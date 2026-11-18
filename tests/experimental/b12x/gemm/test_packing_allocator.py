# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent destinations preserve packed tensor values and layouts."""

import pytest
import torch

from b12x._lib.intrinsics.reference import swizzle_block_scale
from b12x._lib.quant.block_codec import block_codec
from b12x.gemm._shared import wo_mxfp8
from b12x.gemm.blockscaled import _linear
from b12x.gemm.blockscaled._iq2_xs import _retile_dense_blocks


class Allocator:
    def __init__(self):
        self.outputs = []

    def __call__(self, shape, *, dtype, device):
        result = torch.empty(shape, dtype=dtype, device=device)
        self.outputs.append(result)
        return result

    def assert_final_planes(self, *tensors):
        addresses = {tensor.untyped_storage().data_ptr() for tensor in tensors}
        assert all(
            tensor.untyped_storage().data_ptr() in addresses for tensor in self.outputs
        )


def assert_identical(left, right):
    assert left.shape == right.shape
    assert left.stride() == right.stride()
    assert left.dtype == right.dtype
    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize("shape", [(17, 7), (128, 4), (2, 129, 5)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_nvfp4_swizzle_destination(shape, dtype):
    source = (
        torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64).reshape(
            shape
        )
        % 15
    ).to(dtype)
    allocator = Allocator()
    expected = swizzle_block_scale(source)
    actual = swizzle_block_scale(source, allocate=allocator)
    assert_identical(actual, expected)
    assert len(allocator.outputs) == 1
    allocator.assert_final_planes(actual)


@pytest.mark.parametrize("shape", [(3, 128), (129, 256), (5, 128, 2)])
def test_mxfp8_quantization_destinations(monkeypatch, shape):
    monkeypatch.setattr(wo_mxfp8, "_check_gpu_tensor", lambda *args: None)
    torch.manual_seed(7)
    source = torch.randn(shape, dtype=torch.bfloat16)
    allocator = Allocator()
    expected = wo_mxfp8.quantize_mxfp8_rows_torch(source)
    actual = wo_mxfp8.quantize_mxfp8_rows_torch(source, allocate=allocator)
    for name in ("values", "scale_rows", "scale_mma"):
        assert_identical(getattr(actual, name), getattr(expected, name))
    assert len(allocator.outputs) == 3
    allocator.assert_final_planes(actual.values, actual.scale_rows, actual.scale_mma)


@pytest.mark.parametrize("columns", [128, 160])
def test_mxfp8_packing_destinations(monkeypatch, columns):
    monkeypatch.setattr(wo_mxfp8, "_check_gpu_tensor", lambda *args: None)
    monkeypatch.setattr(_linear, "_check_gpu_tensor", lambda *args: None)
    torch.manual_seed(8)
    weight = torch.randn((8, columns)).to(torch.float8_e4m3fn)
    scales = torch.randint(1, 254, (8, columns // 32), dtype=torch.uint8)
    allocator = Allocator()
    expected = _linear.pack_weight(weight, scales, recipe="mxfp8")
    actual = _linear.pack_weight(weight, scales, recipe="mxfp8", allocate=allocator)
    for name in ("values", "scale_rows", "scale_mma"):
        assert_identical(getattr(actual.weight, name), getattr(expected.weight, name))
    assert len(allocator.outputs) == (1 if columns == 128 else 3)
    allocator.assert_final_planes(
        actual.weight.values, actual.weight.scale_rows, actual.weight.scale_mma
    )


@pytest.mark.parametrize("columns", [128, 160])
def test_tensor_fp8_packing_destinations(monkeypatch, columns):
    monkeypatch.setattr(wo_mxfp8, "_check_gpu_tensor", lambda *args: None)
    monkeypatch.setattr(_linear, "_check_gpu_tensor", lambda *args: None)
    weight = torch.randn((128, columns)).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.25], dtype=torch.float32)
    allocator = Allocator()
    expected = _linear.pack_weight(weight, scale, recipe="tensor_fp8")
    actual = _linear.pack_weight(weight, scale, recipe="tensor_fp8", allocate=allocator)
    for name in ("values", "scale_mma", "block_scale", "output_scale"):
        assert_identical(getattr(actual, name), getattr(expected, name))
    assert len(allocator.outputs) == (2 if columns == 128 else 3)
    allocator.assert_final_planes(
        actual.values, actual.scale_mma, actual.block_scale, actual.output_scale
    )


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
@pytest.mark.parametrize("rows", [8, 128])
def test_dense_block_retile_destinations(codec, rows):
    spec = block_codec(codec)
    blocks = torch.randint(0, 256, (rows, 3, spec.block_bytes), dtype=torch.uint8)
    bases = blocks[..., :2].contiguous()
    allocator = Allocator()
    expected = _retile_dense_blocks(blocks, bases, spec, codec=codec)
    actual = _retile_dense_blocks(blocks, bases, spec, codec=codec, allocate=allocator)
    assert_identical(actual.values, expected.values)
    assert_identical(actual.metadata, expected.metadata)
    assert len(allocator.outputs) == 2
    allocator.assert_final_planes(actual.values, actual.metadata)


@pytest.mark.parametrize("block_size", [(32, 32), (128, 128)])
def test_wo_projection_packing_destinations(monkeypatch, block_size):
    monkeypatch.setattr(wo_mxfp8, "_check_gpu_tensor", lambda *args: None)
    torch.manual_seed(11)
    groups, rank, width, hidden = 2, 128, 128, 128
    a = torch.randn((groups * rank, width)).to(torch.float8_e4m3fn)
    b = torch.randn((hidden, groups * rank)).to(torch.float8_e4m3fn)
    block = block_size[0]
    if block == 32:
        sa = torch.full(
            (groups * (rank // block), width // block), 127, dtype=torch.uint8
        )
        sb = torch.full(
            (hidden // block, groups * rank // block), 127, dtype=torch.uint8
        )
    else:
        sa = torch.full(
            (groups * (rank // block), width // block), 0.3, dtype=torch.float32
        )
        sb = torch.full(
            (hidden // block, groups * rank // block), 0.2, dtype=torch.float32
        )
    options = dict(
        groups=groups,
        rank=rank,
        group_width=width,
        hidden=hidden,
        block_size=block_size,
    )
    allocator = Allocator()
    expected = wo_mxfp8.pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        a, sa, b, sb, **options
    )
    actual = wo_mxfp8.pack_wo_projection_fp8_block_scaled_weights_mxfp8(
        a, sa, b, sb, allocate=allocator, **options
    )
    final = []
    for part in ("wo_a", "wo_b"):
        actual_part, expected_part = getattr(actual, part), getattr(expected, part)
        for name in ("values", "scale_rows", "scale_mma"):
            assert_identical(getattr(actual_part, name), getattr(expected_part, name))
            final.append(getattr(actual_part, name))
    allocator.assert_final_planes(*final)
