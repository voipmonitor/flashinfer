from __future__ import annotations

import torch

import b12x.moe.fused_moe._impl as tp_moe
from b12x._lib.intrinsics.reference import pack_grouped_fp4_values, swizzle_block_scale
from b12x.moe._shared.kernels.reference import (
    moe_reference_w4a16_f32,
    moe_reference_w4a16_fp4_e8m0_k32,
)

from b12x.moe._shared.kernels.micro.kernel import (
    MoEMicroKernelBackend as NVFP4MoEMicroKernelBackend,
)
from b12x.testing.reference.w4a16_reference import moe_reference_w4a16


def _packed_fp4_constant(
    value: float,
    *,
    groups: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    dense = torch.full((groups, rows, cols), value, dtype=torch.float32)
    return pack_grouped_fp4_values(dense).permute(2, 0, 1).contiguous()


def _blockscale_constant(
    value: float,
    *,
    groups: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scales = torch.full(
        (groups, rows, cols // 16),
        value,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    return swizzle_block_scale(scales)


def _e8m0_blockscale_constant(
    value: float,
    *,
    groups: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scales = torch.full(
        (groups, rows, cols // 32),
        value,
        dtype=torch.float32,
    )
    return scales.to(torch.float8_e8m0fnu)


def test_w4a16_reference_uses_bf16_activation_and_intermediate_without_activation_scales() -> (
    None
):
    experts, hidden, intermediate, topk = 1, 16, 16, 1
    x = torch.full((1, hidden), 0.25, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, topk, dtype=torch.int32)
    topk_weights = torch.ones(1, topk, dtype=torch.float32)

    w1_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )
    w1_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )

    actual = moe_reference_w4a16(
        x,
        w1_fp4,
        w1_blockscale,
        torch.ones(experts, dtype=torch.float32),
        w2_fp4,
        w2_blockscale,
        torch.ones(experts, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation="relu2",
    )

    torch.testing.assert_close(
        actual.float(),
        torch.full((1, hidden), 256.0, dtype=torch.float32),
    )


def test_w4a16_f32_oracle_uses_weight_only_scales_without_activation_quant() -> None:
    experts, hidden, intermediate, topk = 1, 16, 16, 1
    x = torch.full((1, hidden), 0.25, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, topk, dtype=torch.int32)
    topk_weights = torch.ones(1, topk, dtype=torch.float32)

    w1_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )
    w1_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_blockscale = _blockscale_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )

    actual = moe_reference_w4a16_f32(
        x,
        w1_fp4,
        w1_blockscale,
        torch.full((experts,), 2.0, dtype=torch.float32),
        w2_fp4,
        w2_blockscale,
        torch.full((experts,), 3.0, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation="relu2",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.full((1, hidden), 3072.0, dtype=torch.float32),
    )


def test_w4a16_fp4_e8m0_k32_oracle_uses_raw_k32_scale_grid() -> None:
    experts, hidden, intermediate, topk = 1, 32, 32, 1
    x = torch.full((1, hidden), 0.25, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, topk, dtype=torch.int32)
    topk_weights = torch.ones(1, topk, dtype=torch.float32)

    w1_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    )
    w2_fp4 = _packed_fp4_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    )
    w1_blockscale = _e8m0_blockscale_constant(
        1.0,
        groups=experts,
        rows=intermediate,
        cols=hidden,
    ).view(torch.uint8)
    w2_blockscale = _e8m0_blockscale_constant(
        1.0,
        groups=experts,
        rows=hidden,
        cols=intermediate,
    ).view(torch.uint8)

    actual = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        w1_fp4,
        w1_blockscale,
        torch.full((experts,), 2.0, dtype=torch.float32),
        w2_fp4,
        w2_blockscale,
        torch.full((experts,), 3.0, dtype=torch.float32),
        topk_ids,
        topk_weights,
        experts,
        hidden,
        intermediate,
        activation="relu2",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.full((1, hidden), 24576.0, dtype=torch.float32),
    )


def test_nvfp4_direct_micro_supports_partial_512_k_groups() -> None:
    for batch_size in (1, 2, 4, 8):
        assert NVFP4MoEMicroKernelBackend.is_supported(
            m=batch_size,
            k=2688,
            n=1856,
            num_topk=6,
            weight_E=128,
        )

    assert not NVFP4MoEMicroKernelBackend.is_supported(
        m=1,
        k=2720,
        n=1856,
        num_topk=6,
        weight_E=128,
    )

    plan = tp_moe._plan_core_workspace(
        "micro",
        "nvfp4",
        state_E=128,
        weight_E=128,
        k=2688,
        n=1856,
        num_topk=6,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        routed_rows=6,
        max_rows=6,
    )
    barrier_spec = next(
        spec for spec in plan.tensor_specs if spec.name == "barrier_count"
    )
    assert barrier_spec.shape == (22,)


def test_mxfp4_oracle_ignores_inactive_routes() -> None:
    experts, hidden, intermediate = 2, 32, 32
    x = torch.full((2, hidden), 0.25, dtype=torch.bfloat16)
    w13 = _packed_fp4_constant(1.0, groups=experts, rows=2 * intermediate, cols=hidden)
    w2 = _packed_fp4_constant(1.0, groups=experts, rows=hidden, cols=intermediate)
    s13 = _e8m0_blockscale_constant(
        0.125, groups=experts, rows=2 * intermediate, cols=hidden
    )
    s2 = _e8m0_blockscale_constant(
        0.125, groups=experts, rows=hidden, cols=intermediate
    )
    one = torch.ones(experts)

    def reference(ids, probabilities):
        return moe_reference_w4a16_fp4_e8m0_k32(
            x,
            w13,
            s13,
            one,
            w2,
            s2,
            one,
            ids,
            probabilities,
            experts,
            hidden,
            intermediate,
            activation="situ",
        )

    expected = reference(
        torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0.3, 0.7], [0.0, 0.0]])
    )
    actual = reference(
        torch.tensor([[0, -1, 1, experts], [-1, -7, experts, experts + 4]]),
        torch.tensor([[0.3, 1.0, 0.7, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0]) == hidden
    assert torch.count_nonzero(actual[1]) == 0
