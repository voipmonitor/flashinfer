"""SwiGLU clamp support in the tiny-decode W4A8-MX MoE kernel.

The tiny-decode band (1-4 tokens, ``n % 32 == 0``, ``n % 128 != 64``)
specializes ``swiglu_limit`` at compile time and clamps gate to ``max=limit``
and up to ``[-limit, limit]`` before SiLU, like the W4A8 reference and the
compact W4A8 micro activation. Queries carrying SwiGLU alpha/beta or post-SiLU
clamping, which tiny decode does not implement, are not routed to it.

The host-only tests check selection, launch plumbing, and the compile key.
The GPU tests use inputs large enough that the clamp changes the result and
compare against two pure-Torch oracles built from the checkpoint-layout
weights: one with tiny decode's own activation arithmetic (tight tolerance)
and the MXFP8-activation W4A8 reference (looser tolerance). The clamp test
fails on a kernel that ignores the limit.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.moe.fused_moe import _impl
from b12x.moe._shared.kernels.reference import (
    _gated_row_slices,
    _make_fp4_lut,
    _w4a8_effective_weight,
    compare_to_reference,
    moe_reference_w4a8_mx,
)
from b12x.testing.reference.helpers import require_b12x
from b12x.moe._shared.kernels.tiny_decode import (
    MoETinyDecodeKernelBackend,
    MoETinyDecodeKernelBackendPhase1,
    MoETinyDecodeKernelBackendPhase2,
)

from .test_cute_migration_moe_standard_corpus import (
    _Inputs,
    _Weights,
    _assert_oracle,
    _make_inputs,
    _prepare_and_bind,
    _reset_dispatch_environment,
    _run_live_graph_check,
)

_LIMIT = 10.0

# (n, expected route) for DeepSeek-V4-style hidden 5120 / top-6 / 384 experts.
# n % 128 == 64 shards take the compact micro kernel, which clamps on its own.
_WIDTHS = [
    (352, "tiny"),  # GLM 2048/TP6-style tail shard (n % 128 == 96)
    (576, "compact"),  # n % 128 == 64
    (768, "tiny"),  # 2304 / TP3
    (1152, "tiny"),  # 2304 / TP2
    (2304, "tiny"),  # 2304 / TP1
]


def _supported(n: int, *, num_tokens: int = 1, **swiglu) -> bool:
    return _impl._w4a8_mx_micro_supported(
        num_tokens=num_tokens,
        k=5120,
        n=n,
        num_topk=6,
        weight_E=384,
        activation="silu",
        source_format="fp4_e8m0_k32",
        **swiglu,
    )


@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("n,route", _WIDTHS)
def test_tiny_decode_selection_accepts_swiglu_limit(monkeypatch, n, route, num_tokens):
    monkeypatch.delenv("B12X_W4A8_TINY_DECODE", raising=False)
    # The limit is a compile-time kernel parameter, not a selection input.
    assert _supported(n, num_tokens=num_tokens)
    assert _supported(n, num_tokens=num_tokens, swiglu_alpha=None, swiglu_beta=None)
    # Normalized SiLU queries carry the identity alpha/beta.
    assert _supported(n, num_tokens=num_tokens, swiglu_alpha=1.0, swiglu_beta=0.0)
    # Disabling tiny decode separates the two micro families.
    monkeypatch.setenv("B12X_W4A8_TINY_DECODE", "0")
    assert _supported(n, num_tokens=num_tokens) == (route == "compact")


@pytest.mark.parametrize(
    "swiglu",
    [
        {"swiglu_alpha": 1.702},
        {"swiglu_beta": 1.0},
        {"swiglu_alpha": 1.702, "swiglu_beta": 1.0},
        {"swiglu_alpha": "+inf"},
        {"swiglu_clamp_after_silu": True},
    ],
    ids=["alpha", "beta", "alpha-beta", "codec-alpha", "clamp-after-silu"],
)
@pytest.mark.parametrize("n", [n for n, route in _WIDTHS if route == "tiny"])
def test_tiny_decode_rejects_unimplemented_swiglu_variants(monkeypatch, n, swiglu):
    monkeypatch.delenv("B12X_W4A8_TINY_DECODE", raising=False)
    assert _supported(n)
    assert not _supported(n, **swiglu)


def _query(**overrides):
    from b12x.preparation import FrozenMapping
    from b12x.moe.fused_moe._tuning import MoeDecodeQuery

    values = dict(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=384,
        hidden_size=5120,
        intermediate_size=768,
        top_k=6,
        num_tokens=1,
        routed_rows=6,
        route_num_experts=384,
        route_logits_dtype="float32",
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=_LIMIT,
        swiglu_alpha=None,
        swiglu_beta=None,
        w13_layout="w31",
        weight_layouts=("w31",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=True,
        numerical_recipe=None,
        controls=FrozenMapping(),
    )
    values.update(overrides)
    return MoeDecodeQuery(**values)


@pytest.mark.parametrize("n", [n for n, route in _WIDTHS if route == "tiny"])
def test_heuristic_keeps_limited_queries_on_tiny_decode(monkeypatch, n):
    from b12x.preparation import DeviceIdentity

    monkeypatch.delenv("B12X_W4A8_TINY_DECODE", raising=False)
    device = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="NVIDIA RTX PRO 6000 Blackwell",
    )
    for tokens in (1, 2, 3, 4):
        query = _query(intermediate_size=n, num_tokens=tokens, routed_rows=tokens * 6)
        assert _impl._policy_micro_supported(query)
        assert _impl._heuristic_moe_decode_config(query, device).backend == "micro"
        for alpha, beta in ((1.702, None), (None, 1.0)):
            unsupported = replace(query, swiglu_alpha=alpha, swiglu_beta=beta)
            assert not _impl._policy_micro_supported(unsupported)
            assert (
                _impl._heuristic_moe_decode_config(unsupported, device).backend
                == "dynamic"
            )


# (hidden, n, experts, family) for model tensor-parallel shards with
# swiglu_limit=10: DeepSeek-V4.1-Flash (5120 / 2304, 384 experts) at TP3/TP4
# and DeepSeek-V4-Flash (4096 / 2048, 256 experts) at TP2/TP4, all top-6.
_MODEL_SHARDS = [
    pytest.param(5120, 768, 384, "tiny", id="dsv41-tp3"),
    pytest.param(5120, 576, 384, "compact", id="dsv41-tp4"),
    pytest.param(4096, 1024, 256, "tiny", id="dsv4-tp2"),
    pytest.param(4096, 512, 256, "tiny", id="dsv4-tp4"),
]


@pytest.mark.parametrize("k,n,experts,family", _MODEL_SHARDS)
def test_model_shard_selection(monkeypatch, k, n, experts, family):
    """Which micro family the heuristic picks for 1-4 token decode."""
    from b12x.preparation import DeviceIdentity

    monkeypatch.delenv("B12X_W4A8_TINY_DECODE", raising=False)
    device = DeviceIdentity(
        vendor="nvidia",
        compute_capability=(12, 0),
        sm_count=188,
        product_name="NVIDIA RTX PRO 6000 Blackwell",
    )
    for tokens in (1, 2, 3, 4):
        query = _query(
            hidden_size=k,
            intermediate_size=n,
            num_experts=experts,
            route_num_experts=experts,
            num_tokens=tokens,
            routed_rows=tokens * 6,
        )
        assert _impl._heuristic_moe_decode_config(query, device).backend == "micro"
        assert (n % 128 == 64) == (family == "compact")
    # The tiny kill switch leaves the compact family (which clamps) in place.
    monkeypatch.setenv("B12X_W4A8_TINY_DECODE", "0")
    query = _query(
        hidden_size=k,
        intermediate_size=n,
        num_experts=experts,
        route_num_experts=experts,
    )
    expected = "micro" if family == "compact" else "dynamic"
    assert _impl._heuristic_moe_decode_config(query, device).backend == expected


@pytest.mark.parametrize("swiglu_limit", [None, _LIMIT])
def test_tiny_decode_launch_receives_swiglu_limit(monkeypatch, swiglu_limit):
    m, k, n, topk, experts = 2, 512, 128, 2, 4
    seen = []

    def fake_get_tiny_decode_kernel(weight_E, m_, k_, n_, num_topk, **kwargs):
        seen.append(((weight_E, m_, k_, n_, num_topk), kwargs))
        return "fc1", "fc2"

    launches = []
    monkeypatch.setattr(_impl, "_get_tiny_decode_kernel", fake_get_tiny_decode_kernel)
    monkeypatch.setattr(
        MoETinyDecodeKernelBackend,
        "launch",
        staticmethod(lambda fc1, fc2, **kwargs: launches.append((fc1, fc2))),
    )
    workspace = SimpleNamespace(
        barrier_count=torch.zeros(1, dtype=torch.int32),
        barrier_epoch=torch.zeros(1, dtype=torch.int32),
        micro_intermediate=torch.zeros(m * topk * 2 * n, dtype=torch.float32),
        volatile_launch_state=False,
    )
    weights = SimpleNamespace(
        w1_storage=torch.zeros(1, dtype=torch.uint8),
        w1_scale_storage=torch.zeros(1, dtype=torch.uint8),
        w2_storage=torch.zeros(1, dtype=torch.uint8),
        w2_scale_storage=torch.zeros(1, dtype=torch.uint8),
    )
    _impl._launch_micro(
        workspace=workspace,
        weights=weights,
        a=torch.zeros(m, k, dtype=torch.bfloat16),
        flat_ids=torch.zeros(m * topk, dtype=torch.int32),
        flat_weights=torch.zeros(m * topk, dtype=torch.float32),
        input_gs=torch.ones(1),
        down_input_scale=torch.ones(1),
        scatter_output=torch.zeros(m, k, dtype=torch.bfloat16),
        weight_E=experts,
        m=m,
        k=k,
        n=n,
        num_topk=topk,
        routed_rows=m * topk,
        topk_ids_dtype=torch.int32,
        fast_math=False,
        stream=None,
        quant_mode="w4a8_mx",
        swiglu_limit=swiglu_limit,
    )
    assert launches == [("fc1", "fc2")]
    assert len(seen) == 1
    assert seen[0][0] == (experts, m, k, n, topk)
    assert seen[0][1]["swiglu_limit"] == swiglu_limit


def test_tiny_decode_compile_key_specializes_swiglu_limit():
    def phase2(limit):
        kernel = MoETinyDecodeKernelBackendPhase2(swiglu_limit=limit)
        kernel.configure(1, 5120, 768, 6, 384)
        return kernel

    plain, limited = phase2(None), phase2(_LIMIT)
    assert plain.swiglu_limit is None
    assert limited.swiglu_limit == _LIMIT
    assert plain.__cache_key__ != limited.__cache_key__
    assert plain.__cache_key__[:-1] == limited.__cache_key__[:-1]
    assert phase2(int(_LIMIT)).__cache_key__ == limited.__cache_key__
    # FC1 is activation-free and keeps a single compile identity.
    phase1 = MoETinyDecodeKernelBackendPhase1()
    assert phase1.swiglu_limit is None
    with pytest.raises(ValueError, match="phase 2"):
        MoETinyDecodeKernelBackend(compile_time_phase=1, swiglu_limit=_LIMIT)
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="swiglu_limit"):
            MoETinyDecodeKernelBackendPhase2(swiglu_limit=bad)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------

# FP4 E2M1 codes drawn uniformly have E[w^2] = 68.5 / 8; with the 2^-5 block
# scale below a K-long dot product with unit inputs has this standard deviation.
_FP4_UNIFORM_RMS = math.sqrt(68.5 / 8) / 32

# Tiny decode keeps BF16 activations (as f16) and an f16 activated
# intermediate; it does not quantize either to MXFP8. Against an oracle with
# that arithmetic the only differences are fp32 accumulation order and the
# BF16 scatter-add of FC2 partials.
_TINY_MIN_COS = 0.9999
_TINY_MAX_NRMSE = 0.01
# moe_reference_w4a8_mx models the MXFP8 activation recipe of the other W4A8
# kernels, which tiny decode does not use (~6% normalized RMSE at these
# input scales). A kernel that ignores the limit is off by >200% here.
_W4A8_MIN_COS = 0.995
_W4A8_MAX_NRMSE = 0.1


def _make_weights(
    device: torch.device, *, seed: int, experts: int, k: int, n: int
) -> _Weights:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1_fp4 = torch.randint(
        0,
        256,
        (experts, 2 * n, k // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_fp4 = torch.randint(
        0,
        256,
        (experts, k, n // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    # E8M0 byte 122 is exactly 2^-5.
    w1_scale = torch.full(
        (experts, 2 * n, k // 32), 122, dtype=torch.uint8, device=device
    )
    w2_scale = torch.full((experts, k, n // 32), 122, dtype=torch.uint8, device=device)
    ones = torch.ones(experts, dtype=torch.float32, device=device)
    return _Weights(
        w1_fp4=w1_fp4,
        w1_scale=w1_scale,
        w1_alpha=ones,
        a1_scale=ones.clone(),
        w2_fp4=w2_fp4,
        w2_scale=w2_scale,
        w2_alpha=ones.clone(),
        a2_scale=ones.clone(),
    )


def _make_scaled_inputs(
    device, *, m, seed, route_shift, experts, k, topk, projection_rms
) -> _Inputs:
    inputs = _make_inputs(
        device,
        m=m,
        seed=seed,
        route_shift=route_shift,
        num_experts=experts,
        hidden_size=k,
        topk=topk,
    )
    # _make_inputs draws N(0, 0.35^2); rescale so gate/up projections have
    # roughly the requested RMS.
    scale = projection_rms / (0.35 * _FP4_UNIFORM_RMS * math.sqrt(k))
    return replace(inputs, a=(inputs.a.float() * scale).to(torch.bfloat16))


def _w4a8_oracle(weights, inputs, *, experts, k, n, swiglu_limit):
    """Checkpoint-layout W4A8 reference (MXFP8 activation recipe)."""
    return moe_reference_w4a8_mx(
        inputs.a.float(),
        weights.w1_fp4,
        weights.w1_scale,
        None,
        weights.w1_alpha,
        weights.w2_fp4,
        weights.w2_scale,
        None,
        weights.w2_alpha,
        inputs.topk_ids,
        inputs.topk_weights,
        experts,
        k,
        n,
        activation="silu",
        swiglu_limit=swiglu_limit,
        w13_layout="w13",
    )


def _tiny_oracle(weights, inputs, *, experts, k, n, swiglu_limit):
    """Checkpoint-layout reference with tiny decode's activation arithmetic.

    Exact FP4 x E8M0 weights, f16 inputs, float64 projections, the SwiGLU clamp
    of ``reference._apply_gated_activation``, and the route-weighted activated
    intermediate rounded to f16 before FC2, as the tiny kernel packs it.
    """
    lut = _make_fp4_lut(inputs.a.device)
    gate_rows, up_rows = _gated_row_slices("silu", n, w13_layout="w13")
    x = inputs.a.to(torch.float16).double()
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    dequant = {}
    for token in range(x.shape[0]):
        for route in range(inputs.topk_ids.shape[1]):
            eid = int(inputs.topk_ids[token, route])
            if not 0 <= eid < experts:
                continue
            if eid not in dequant:
                w13 = _w4a8_effective_weight(
                    weights.w1_fp4[eid], weights.w1_scale[eid], None, 2 * n, k, lut
                )
                w2 = _w4a8_effective_weight(
                    weights.w2_fp4[eid], weights.w2_scale[eid], None, k, n, lut
                )
                dequant[eid] = (
                    w13.double() * float(weights.w1_alpha[eid]),
                    w2.double() * float(weights.w2_alpha[eid]),
                )
            w13, w2 = dequant[eid]
            projected = w13 @ x[token]
            gate, up = projected[gate_rows], projected[up_rows]
            if swiglu_limit is not None:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            weight = float(inputs.topk_weights[token, route])
            act = (gate * torch.sigmoid(gate) * up * weight).to(torch.float16)
            out[token] += w2 @ act.double()
    return out.float()


def _check(actual, reference, *, context, min_cos, max_normalized_rmse):
    metrics = compare_to_reference(actual.float(), reference.float())
    rms = reference.float().square().mean().sqrt().item()
    # Logged for tolerance calibration (pytest -rP shows it for passing tests).
    print(f"{context}: cos={metrics.cos:.6f} nrmse={metrics.rmse / rms:.5f}")
    _assert_oracle(
        actual,
        reference,
        context=context,
        min_cos=min_cos,
        max_normalized_rmse=max_normalized_rmse,
    )


def _spy_tiny_launches(monkeypatch):
    calls = []
    launch = _impl._launch_tiny_decode_flat

    def spy(**kwargs):
        calls.append(kwargs.get("swiglu_limit"))
        return launch(**kwargs)

    monkeypatch.setattr(_impl, "_launch_tiny_decode_flat", spy)
    return calls


def _run_eager(weights, inputs, *, topk, swiglu_limit, monkeypatch):
    _reset_dispatch_environment(monkeypatch)
    calls = _spy_tiny_launches(monkeypatch)
    case = _prepare_and_bind(
        weights,
        inputs,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        num_topk=topk,
        swiglu_limit=swiglu_limit,
    )
    assert case.binding.implementation == "micro"
    _impl.b12x_moe_fp4(binding=case.binding)
    torch.cuda.synchronize()
    assert len(calls) == 1, "tiny decode kernel was not selected"
    return case.binding.output.clone(), calls


# (experts, k, n, topk): the standard corpus shape, a 352-wide tail shard
# (bounds-checked FC2 branch), and a DeepSeek-V4 TP3 expert shard.
_SHAPES = [
    pytest.param(4, 512, 128, 2, id="k512-n128"),
    pytest.param(4, 512, 352, 2, id="k512-n352-tail"),
    pytest.param(8, 5120, 768, 6, id="k5120-n768-top6"),
    # DeepSeek-V4-Flash (hidden 4096, moe_intermediate 2048, 256 experts,
    # top-6, swiglu_limit 10) tensor-parallel shards: TP4 and TP2.
    pytest.param(256, 4096, 512, 6, id="dsv4-tp4-k4096-n512-e256-top6"),
    pytest.param(256, 4096, 1024, 6, id="dsv4-tp2-k4096-n1024-e256-top6"),
]

# Gate/up RMS well above the limit: the clamp changes most activations.
_CLAMPING_RMS = 15.0


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("experts,k,n,topk", _SHAPES)
def test_tiny_decode_applies_swiglu_limit(monkeypatch, experts, k, n, topk, m):
    """Fails on a tiny-decode kernel that ignores swiglu_limit."""
    device = require_b12x()
    _reset_dispatch_environment(monkeypatch)
    weights = _make_weights(device, seed=401, experts=experts, k=k, n=n)
    make = dict(experts=experts, k=k, topk=topk, projection_rms=_CLAMPING_RMS)
    initial = _make_scaled_inputs(device, m=m, seed=402, route_shift=0, **make)
    changed = _make_scaled_inputs(device, m=m, seed=403, route_shift=1, **make)
    shape = dict(experts=experts, k=k, n=n)
    # Preparation repacks the source weights in place: build oracles first.
    initial_tiny = _tiny_oracle(weights, initial, swiglu_limit=_LIMIT, **shape)
    changed_tiny = _tiny_oracle(weights, changed, swiglu_limit=_LIMIT, **shape)
    initial_w4a8 = _w4a8_oracle(weights, initial, swiglu_limit=_LIMIT, **shape)

    # The inputs must make the clamp matter, or this test proves nothing.
    unclamped = _tiny_oracle(weights, initial, swiglu_limit=None, **shape)
    assert compare_to_reference(unclamped, initial_tiny).cos < 0.95

    calls = _spy_tiny_launches(monkeypatch)
    case = _prepare_and_bind(
        weights,
        initial,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        num_topk=topk,
        swiglu_limit=_LIMIT,
    )
    assert case.binding.implementation == "micro"
    context = f"tiny-decode-swiglu-limit-m{m}-k{k}-n{n}"
    _impl.b12x_moe_fp4(binding=case.binding)
    torch.cuda.synchronize()
    print(
        f"{context}: selected implementation={case.binding.implementation} "
        f"tiny_launches={len(calls)} limit={calls[:1]}"
    )
    assert calls, "tiny decode kernel was not selected"
    _check(
        case.binding.output,
        initial_tiny,
        context=f"{context}:tiny-oracle",
        min_cos=_TINY_MIN_COS,
        max_normalized_rmse=_TINY_MAX_NRMSE,
    )
    _check(
        case.binding.output,
        initial_w4a8,
        context=f"{context}:w4a8-reference",
        min_cos=_W4A8_MIN_COS,
        max_normalized_rmse=_W4A8_MAX_NRMSE,
    )
    _run_live_graph_check(
        case,
        initial=initial,
        changed=changed,
        initial_reference=initial_tiny,
        changed_reference=changed_tiny,
        context=context,
        min_cos=_TINY_MIN_COS,
        max_normalized_rmse=_TINY_MAX_NRMSE,
    )
    assert set(calls) == {_LIMIT}


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("experts,k,n,topk", _SHAPES)
def test_tiny_decode_without_limit_is_plain_silu(monkeypatch, experts, k, n, topk, m):
    """No limit: unclamped SiLU-GLU (also the calibration baseline on main)."""
    device = require_b12x()
    weights = _make_weights(device, seed=411, experts=experts, k=k, n=n)
    make = dict(experts=experts, k=k, topk=topk, projection_rms=_CLAMPING_RMS)
    inputs = _make_scaled_inputs(device, m=m, seed=412, route_shift=0, **make)
    shape = dict(experts=experts, k=k, n=n)
    tiny_reference = _tiny_oracle(weights, inputs, swiglu_limit=None, **shape)
    w4a8_reference = _w4a8_oracle(weights, inputs, swiglu_limit=None, **shape)

    output, calls = _run_eager(
        weights, inputs, topk=topk, swiglu_limit=None, monkeypatch=monkeypatch
    )
    assert calls == [None]
    context = f"tiny-decode-no-limit-m{m}-k{k}-n{n}"
    _check(
        output,
        tiny_reference,
        context=f"{context}:tiny-oracle",
        min_cos=_TINY_MIN_COS,
        max_normalized_rmse=_TINY_MAX_NRMSE,
    )
    _check(
        output,
        w4a8_reference,
        context=f"{context}:w4a8-reference",
        min_cos=_W4A8_MIN_COS,
        max_normalized_rmse=_W4A8_MAX_NRMSE,
    )


@pytest.mark.parametrize("experts,k,n,topk", _SHAPES)
def test_tiny_decode_inactive_limit_matches_no_limit(monkeypatch, experts, k, n, topk):
    """A limit no activation reaches compiles the clamp but changes nothing.

    FC1/FC2 partials are combined with atomics, so two launches agree to
    accumulation-order noise rather than bitwise.
    """
    device = require_b12x()
    make = dict(experts=experts, k=k, topk=topk, projection_rms=_CLAMPING_RMS)
    inputs = _make_scaled_inputs(device, m=2, seed=422, route_shift=0, **make)
    outputs = {}
    for limit in (None, 1.0e6):
        weights = _make_weights(device, seed=421, experts=experts, k=k, n=n)
        outputs[limit], calls = _run_eager(
            weights, inputs, topk=topk, swiglu_limit=limit, monkeypatch=monkeypatch
        )
        assert calls == [limit]
    _check(
        outputs[1.0e6],
        outputs[None],
        context=f"tiny-decode-inactive-limit-k{k}-n{n}",
        min_cos=_TINY_MIN_COS,
        max_normalized_rmse=_TINY_MAX_NRMSE,
    )


def _spy_compact_launches(monkeypatch):
    import b12x.moe._shared.kernels.w4a8_compact_micro as compact_module

    calls = []
    launch = compact_module.launch_w4a8_compact_micro

    def spy(*args, **kwargs):
        calls.append(kwargs.get("swiglu_limit"))
        return launch(*args, **kwargs)

    monkeypatch.setattr(compact_module, "launch_w4a8_compact_micro", spy)
    return calls


@pytest.mark.parametrize("m", [1, 4])
def test_dsv41_tp4_shard_uses_clamping_compact_micro(monkeypatch, m):
    """DeepSeek-V4.1 TP4 (n=576): compact micro is selected and clamps.

    Passes on an unpatched tree too; it backs the "TP4 is not affected" claim.
    """
    device = require_b12x()
    experts, k, n, topk = 384, 5120, 576, 6
    weights = _make_weights(device, seed=431, experts=experts, k=k, n=n)
    make = dict(experts=experts, k=k, topk=topk, projection_rms=_CLAMPING_RMS)
    inputs = _make_scaled_inputs(device, m=m, seed=432, route_shift=0, **make)
    shape = dict(experts=experts, k=k, n=n)
    clamped = _w4a8_oracle(weights, inputs, swiglu_limit=_LIMIT, **shape)
    unclamped = _w4a8_oracle(weights, inputs, swiglu_limit=None, **shape)
    assert compare_to_reference(unclamped, clamped).cos < 0.95

    _reset_dispatch_environment(monkeypatch)
    tiny_calls = _spy_tiny_launches(monkeypatch)
    compact_calls = _spy_compact_launches(monkeypatch)
    case = _prepare_and_bind(
        weights,
        inputs,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        num_topk=topk,
        swiglu_limit=_LIMIT,
    )
    # Serving binds through the prepared fused-MoE state, which attaches the
    # compact kernels compiled at preparation (_FusedMoeState.bind). The
    # scratch-plan binding used here carries none, so build them the same way.
    from b12x.moe.fused_moe._preparation import _compact_launches

    compact = _compact_launches(case.scratch_plan.launch_plan, case.scratch_plan.caps)
    assert compact is not None, "n % 128 == 64 shard did not plan compact micro"
    binding = replace(case.binding, compact_launches=compact)
    _impl.b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    context = f"dsv41-tp4-compact-m{m}-k{k}-n{n}"
    print(
        f"{context}: selected implementation={binding.implementation} "
        f"compact_launches={len(compact_calls)} limit={compact_calls[:1]} "
        f"tiny_launches={len(tiny_calls)}"
    )
    assert binding.implementation == "micro"
    assert tiny_calls == []
    assert compact_calls == [_LIMIT]
    # Compact micro uses the MXFP8 activation recipe of the W4A8 reference.
    _check(
        binding.output,
        clamped,
        context=f"{context}:w4a8-reference",
        min_cos=_W4A8_MIN_COS,
        max_normalized_rmse=_W4A8_MAX_NRMSE,
    )
