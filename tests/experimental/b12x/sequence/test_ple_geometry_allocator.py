# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-owned PLE geometry storage preserves the host geometry."""

import torch

from b12x.sequence.ple_hash.geometry import Geometry, allocate_geometry


def test_geometry_allocator_preserves_values_and_storage():
    geometry = Geometry((3, 5), (0, 3), (1, 7, 11), 8, 16)
    allocations = []

    def allocate(shape, *, dtype, device):
        value = torch.empty(shape, dtype=dtype, device=device)
        allocations.append(value)
        return value

    expected = allocate_geometry(geometry, device="cpu")
    actual = allocate_geometry(geometry, device="cpu", allocate=allocate)
    assert actual.geometry is geometry
    assert len(allocations) == 3
    for name, allocation in zip(
        ("prime_sizes", "table_offsets", "multipliers"), allocations, strict=True
    ):
        observed, reference = getattr(actual, name), getattr(expected, name)
        assert observed is allocation
        assert observed.dtype == reference.dtype == torch.int64
        assert (
            observed.shape == reference.shape
            and observed.stride() == reference.stride()
        )
        assert torch.equal(observed, reference)
