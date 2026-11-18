# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert preparation writes final weight planes into supplied destinations."""

import pytest
import torch

from b12x._lib.intrinsics.reference import swizzle_block_scale
from b12x._lib.quant.block_codec import block_codec
from b12x.loader._csf_source import stage_csf_sources
from b12x.moe._shared.kernels.w4a16 import prepare
from b12x.moe._shared.kernels.w4a16.iq2_xs import pack_iq2_xs_matrix
from b12x.moe.fused_moe import _impl


class Allocator:
    def __init__(self):
        self.outputs = []

    def __call__(self, shape, *, dtype, device):
        output = torch.empty(shape, dtype=dtype, device=device)
        self.outputs.append(output)
        return output

    def assert_retained(self, *tensors):
        retained = {tensor.untyped_storage().data_ptr() for tensor in tensors}
        assert all(
            output.untyped_storage().data_ptr() in retained for output in self.outputs
        )


def assert_identical(actual, expected):
    assert (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.stride() == expected.stride()
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
@pytest.mark.parametrize("swap_halves", [False, True])
def test_expert_block_codec_destinations(codec, swap_halves):
    spec = block_codec(codec)
    blocks = torch.randint(
        0, 256, (2, 128, 256 // spec.block_weights, spec.block_bytes), dtype=torch.uint8
    )
    blocks[..., :2] = torch.tensor([0, 60], dtype=torch.uint8)
    options = dict(
        codec=codec, swap_halves=swap_halves, tile_descriptors=True, tile_scales=True
    )
    expected = pack_iq2_xs_matrix(blocks, **options)
    allocator = Allocator()
    actual = pack_iq2_xs_matrix(blocks, **options, allocate=allocator)
    for left, right in zip(actual, expected, strict=True):
        assert_identical(left, right)
    assert len(allocator.outputs) == 2
    allocator.assert_retained(*actual)


@pytest.mark.parametrize("rotation", [None, 64])
@pytest.mark.parametrize("reuse", [False, True])
def test_packed_scale_grids_preserve_reuse_and_destination(rotation, reuse):
    scales = torch.randint(0, 256, (2, 128, 8), dtype=torch.uint8)
    allocator = Allocator()
    options = dict(
        size_k=256, size_n=128, row_rotation=rotation, reuse_input_storage=reuse
    )
    expected = prepare._pack_e8m0_k32_scales(scales.clone(), **options)
    source = scales.clone()
    actual = prepare._pack_e8m0_k32_scales(source, **options, allocate=allocator)
    assert_identical(actual, expected)
    assert len(allocator.outputs) == (0 if reuse else 1)
    if reuse:
        assert (
            actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
    allocator.assert_retained(actual)


@pytest.mark.parametrize("kind", ["native_mxfp4", "packed_mxfp4", "packed_nvfp4"])
@pytest.mark.parametrize("transient_scales", [False, True])
def test_prepared_expert_planes_and_aliases(kind, transient_scales, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "_make_workspace",
        lambda device, **kwargs: torch.zeros(8, dtype=torch.int32, device=device),
    )
    w13 = torch.randint(0, 256, (2, 256, 128), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (2, 256, 64), dtype=torch.uint8)
    g13, g2 = torch.tensor([0.5, 0.75]), torch.tensor([1.25, 1.5])
    if kind == "packed_nvfp4":
        s13 = swizzle_block_scale(torch.ones(2, 256, 16).to(torch.float8_e4m3fn))
        s2 = swizzle_block_scale(torch.ones(2, 256, 8).to(torch.float8_e4m3fn))
        prepare_weight = prepare.prepare_w4a16_modelopt_nvfp4_weights
    else:
        s13 = torch.randint(0, 256, (2, 256, 8), dtype=torch.uint8)
        s2 = torch.randint(0, 256, (2, 256, 4), dtype=torch.uint8)
        prepare_weight = (
            prepare.prepare_w4a16_e8m0_native_weights
            if kind == "native_mxfp4"
            else prepare.prepare_w4a16_fp4_e8m0_k32_weights
        )
    options = dict(activation="silu", params_dtype=torch.bfloat16, w13_layout="w13")
    expected = prepare_weight(w13, s13, g13, w2, s2, g2, **options)
    allocator = Allocator()
    actual = prepare_weight(
        w13,
        s13,
        g13,
        w2,
        s2,
        g2,
        **options,
        allocate=allocator,
        scale_allocate=torch.empty if transient_scales else None,
    )
    fields = [
        "w13",
        "w13_scale",
        "w13_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    ]
    if kind == "native_mxfp4":
        fields.extend(
            [
                "micro_w13_scale",
                "micro_w13_global_scale",
                "micro_w2_scale",
                "micro_w2_global_scale",
            ]
        )
        assert actual.w13 is w13 and actual.w2 is w2
        assert actual.micro_w2_scale is actual.w2_scale
        assert actual.micro_w13_global_scale is actual.w13_global_scale
    for field in fields:
        assert_identical(getattr(actual, field), getattr(expected, field))
    allocator.assert_retained(*(getattr(actual, field) for field in fields))
    if transient_scales:
        allocated = {
            tensor.untyped_storage().data_ptr() for tensor in allocator.outputs
        }
        assert actual.w13_scale.untyped_storage().data_ptr() not in allocated
        assert actual.w2_scale.untyped_storage().data_ptr() not in allocated
    assert actual.workspace.untyped_storage().data_ptr() not in {
        output.untyped_storage().data_ptr() for output in allocator.outputs
    }


def test_runtime_alphas_are_destination_values():
    args = (
        torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
        torch.tensor([2.0, 4.0]),
        torch.tensor([10.0, 20.0]),
        torch.tensor([5.0, 10.0]),
    )
    allocator = Allocator()
    actual = _impl._prepare_modelopt_nvfp4_runtime_alphas(*args, allocate=allocator)
    assert torch.equal(actual[0], torch.tensor([[1.0, 2.0], [1.5, 2.0]]))
    assert torch.equal(actual[1], torch.tensor([2.0, 2.0]))
    assert len(allocator.outputs) == 2
    allocator.assert_retained(*actual)


def test_csf_staging_copies_components_and_segment_offsets_into_destinations():
    components = [
        (
            torch.full((2, 80), index + 1, dtype=torch.uint8),
            torch.arange(index + 1, dtype=torch.int64).to(torch.uint32),
        )
        for index in range(2)
    ]
    allocator = Allocator()
    actual = stage_csf_sources(
        components,
        rows=32,
        columns=8,
        quantization="nvfp4",
        device="cpu",
        encoding="csf",
        allocate=allocator,
    )
    assert torch.equal(
        actual.components[0], torch.stack([part[0] for part in components])
    )
    assert torch.equal(
        actual.components[1], torch.cat([part[1] for part in components])
    )
    assert torch.equal(actual.starts, torch.tensor([0, 1, 3]))
    assert len(allocator.outputs) == 3
    allocator.assert_retained(*actual.components, actual.starts)


@pytest.mark.parametrize("layout", ["w13", "w31"])
def test_public_nvfp4_preparation_retains_generated_scale_destinations(layout):
    from b12x.moe import fused_moe as moe

    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout=layout),
        activation=moe.ActivationSpec(
            mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=2, hidden_size=256, intermediate_size=128),
    )
    packed = moe.PackedWeights(
        w13=torch.zeros((2, 256, 128), dtype=torch.uint8),
        w2=torch.zeros((2, 256, 64), dtype=torch.uint8),
        w13_block_scales=swizzle_block_scale(
            torch.ones(2, 256, 16).to(torch.float8_e4m3fn)
        ),
        w2_block_scales=swizzle_block_scale(
            torch.ones(2, 256, 8).to(torch.float8_e4m3fn)
        ),
        w13_global_scales=torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
        w2_global_scales=torch.tensor([10.0, 20.0]),
        input_scale=torch.tensor([2.0, 4.0]),
        intermediate_scale=torch.tensor([5.0, 10.0]),
    )
    checkpoint_gains = packed.w13_global_scales.clone()
    expected = moe.prepare_weights(plan=plan, weights=packed)
    allocator = Allocator()
    actual = moe.prepare_weights(plan=plan, weights=packed, allocate=allocator)
    fields = (
        "a1_gscale",
        "a2_gscale",
        "w1_fp4",
        "w2_fp4",
        "w1_blockscale",
        "w2_blockscale",
        "w1_alphas",
        "w2_alphas",
    )
    for field in fields:
        assert_identical(getattr(actual._impl, field), getattr(expected._impl, field))
    allocator.assert_retained(*(getattr(actual._impl, field) for field in fields))
    assert actual._impl.w1_fp4 is packed.w13 and actual._impl.w2_fp4 is packed.w2
    assert torch.equal(packed.w13_global_scales, checkpoint_gains)
    assert actual._impl.w1_alphas.data_ptr() != packed.w13_global_scales.data_ptr()
