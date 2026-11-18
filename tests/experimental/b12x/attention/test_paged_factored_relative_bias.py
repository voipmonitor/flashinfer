"""Rank-16 relative bias fused into BF16 paged extend over FP8 K/V."""

import pytest
import torch
from b12x.attention import paged
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x
from . import test_paged_decode_relative_bias as shared


def _prepare(layer, session, inputs):
    geometry = dict(
        num_q_heads=shared.HQ,
        num_kv_heads=shared.HKV,
        head_dim_qk=shared.DIM,
        head_dim_vo=shared.DIM,
        page_size=shared.PAGE,
    )
    capacity = paged.extend_graph_capacity(
        device="cuda",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        batch=layer.max_batch,
        max_cache_page_count=layer.width,
        window_left=layer.window,
        total_q_capacity=layer.rows,
        **geometry,
    )
    caps = paged.Caps(
        device=torch.device("cuda"),
        mode="extend",
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        max_total_q=layer.rows,
        max_batch=layer.max_batch,
        max_page_table_width=layer.width,
        num_cache_pages=layer.pool,
        max_work_items=capacity.max_work_items,
        max_partial_rows=0,
        use_cuda_graph=True,
        copy_runtime_metadata=False,
        **geometry,
    )
    arguments = dict(inputs, output=torch.empty_like(inputs["q"]))
    plan = paged.plan(
        caps,
        invocation=dict(
            paged.invocation_from_tensors(caps, **arguments), window_left=layer.window
        ),
    )

    def prepare_call(state):
        scratch = [
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in state.scratch_plan.scratch_specs()
        ]
        binding = state.bind(scratch=scratch, **arguments, window_left=layer.window)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    result = session.prepare(
        (plan.request(name="factored", prepare_call=prepare_call),)
    )
    scratch = [
        torch.empty(s.shape, dtype=s.dtype, device="cuda") for s in plan.scratch_specs()
    ]

    def run():
        paged.run(
            binding=paged.bind(
                plan, scratch=scratch, **arguments, window_left=layer.window
            ),
            plan=plan,
        )

    return result, plan, run, arguments["output"]


def _draw(layer, lengths, q_lens=None, *, strided=False):
    inputs = layer.draw(lengths=lengths, q_lens=q_lens)
    factors = layer._randn((layer.rows, shared.HQ, 16), 0.5).bfloat16()
    projection = layer._randn((shared.EXTENT, 16), 0.5).bfloat16()
    if strided:
        factors = torch.stack((factors, factors), dim=-2)[..., 0, :]
        projection = torch.stack((projection, projection), dim=1)[:, 0]
    inputs.pop("relative_attention_bias")
    inputs.update(relative_bias_factors=factors, relative_bias_projection=projection)
    return inputs


def _check(layer, out, inputs):
    bias = torch.einsum(
        "thr,dr->thd",
        inputs["relative_bias_factors"].float(),
        inputs["relative_bias_projection"].float(),
    ).bfloat16()
    reference_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in ("relative_bias_factors", "relative_bias_projection")
    }
    reference_inputs["relative_attention_bias"] = bias
    shared._check(layer, out, reference_inputs)


@pytest.mark.parametrize("window", [-1, 1023])
@pytest.mark.parametrize("strided", [False, True])
def test_factored_bias_ragged_graph_replay(window, strided):
    require_b12x()
    name = "factored_extend"
    shared.CASES[name] = (window, 3, 37, 2048, 0.25, 3.0, True)
    try:
        layer = shared._Layer(name, seed=101)
    finally:
        del shared.CASES[name]
    inputs = _draw(layer, [2048, 1024, 1025], [37, 19, 0], strided=strided)
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan, run, out = _prepare(layer, session, inputs)
        run()
        torch.cuda.synchronize()
        _check(layer, out, inputs)
        kernel = next(iter(plan.prepared.state.programs.values())).forward_kernel
        assert kernel.has_factored_relative_bias
        assert not kernel.use_native_fp8_qk_mma
        assert not kernel.use_native_fp8_pv_mma
        assert not kernel.use_kv_repack_fp16
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for lengths, q_lens in (
            ([1023, 1024, 1025], [1, 37, 23]),
            ([37, 128, 2048], [37, 0, 35]),
        ):
            draw = _draw(layer, lengths, q_lens, strided=strided)
            for key in (
                "q",
                "page_table",
                "cache_seqlens",
                "cu_seqlens_q",
                "attention_sink_bias",
                "relative_bias_factors",
                "relative_bias_projection",
            ):
                inputs[key].copy_(draw[key])
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            _check(layer, out, inputs)
        result.close()


@pytest.mark.parametrize("window", [-1, 1023])
@pytest.mark.parametrize("target", [0, 1023])
def test_factored_bias_distance_endpoint_and_infinite_sink(window, target):
    require_b12x()
    name = "factored_endpoint"
    shared.CASES[name] = (window, 1, 35, 2048, 0.25, 1.0, True)
    try:
        layer = shared._Layer(name, seed=target + 123)
    finally:
        del shared.CASES[name]
    inputs = _draw(layer, [2048])
    inputs["q"].zero_()
    inputs["relative_bias_factors"].zero_()
    inputs["relative_bias_factors"][..., 0] = 1
    inputs["relative_bias_projection"].zero_()
    inputs["relative_bias_projection"][target, 0] = 96
    sink = inputs["attention_sink_bias"]
    sink.fill_(-float("inf"))
    sink[1::3, ::5] = float("inf")
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, _, run, out = _prepare(layer, session, inputs)
        run()
        torch.cuda.synchronize()
        result.close()
    position = 2048 - layer.rows + torch.arange(layer.rows, device="cuda") - target
    pages = inputs["page_table"][0, position // shared.PAGE].long()
    expected = (
        layer.v_cache[pages, position % shared.PAGE]
        .float()
        .repeat_interleave(shared.GROUP, 1)
    )
    expected.masked_fill_(torch.isposinf(sink)[:, :, None], 0)
    assert torch.equal(out.float(), expected)


@pytest.mark.parametrize("window", [-1, 1023])
@pytest.mark.parametrize("rows", [320, 1025])
def test_factored_bias_large_extend_tiles(window, rows):
    require_b12x()
    name = "factored_large_extend"
    shared.CASES[name] = (window, 1, rows, 2048, 0.25, 3.0, True)
    try:
        layer = shared._Layer(name, seed=rows + 131)
    finally:
        del shared.CASES[name]
    inputs = _draw(layer, [2048])
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:
        result, plan, run, out = _prepare(layer, session, inputs)
        run()
        torch.cuda.synchronize()
        _check(layer, out, inputs)
        kernel = next(iter(plan.prepared.state.programs.values())).forward_kernel
        assert kernel.traits.cta_tile_q >= 64
        assert not kernel.use_native_fp8_qk_mma
        assert not kernel.use_native_fp8_pv_mma
        assert not kernel.use_kv_repack_fp16
        result.close()
