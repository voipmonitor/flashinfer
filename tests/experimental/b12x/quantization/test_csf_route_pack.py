"""CSF expansion and bounded route packing share one caller-owned launch."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder, compile_nvfp4_csf_pair
from b12x._lib.runtime_control import kernel_resolution_guard
from ..conftest import require_b12x
from .test_csf_blocks import planes_and_native


@pytest.mark.parametrize(
    "plane_configs",
    [
        ((8, True), (8, True)),
        ((4, True), (8, False)),
        ((16, False), (4, True)),
        ((32, True), (16, True)),
        ((8, False), (8, False)),
    ],
)
@pytest.mark.parametrize("capacity", [16, 64, 128])
@pytest.mark.parametrize("block_m", [8, 16, 32, 48, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_expansion_preserves_bytes_routes_and_mutable_replay(
    capacity, block_m, dtype, plane_configs
):
    device = require_b12x()
    experts = 16
    planes, outputs, native = [], [], []
    for (rows, columns), (groups, shared) in zip(
        ((128, 8), (256, 4)), plane_configs, strict=True
    ):
        payload, expected = planes_and_native(rows, columns, experts, dense=False)
        planes.append(
            Nvfp4CompactBlockPlane(
                upload_csf_block_payload(payload, device),
                groups_per_task=groups,
                shared_exceptions=shared,
            )
        )
        outputs.append(torch.empty_like(expected, device=device))
        native.append(expected.to(device))
    geometry = (capacity, block_m, capacity * block_m, capacity)
    decoder = replace(
        Nvfp4CsfDecoder.prepare(*planes, *outputs),
        direct_routes=True,
        packed_route_geometry=geometry,
        packed_route_programs=tuple(
            compile_nvfp4_csf_pair(
                planes[0].geometry, planes[1].geometry, ids64, packed_routes=geometry
            )
            for ids64 in (False, True)
        ),
    )
    ids = torch.empty(capacity, dtype=dtype, device=device)
    packed = (
        torch.empty(geometry[2], dtype=torch.int32, device=device),
        torch.empty(geometry[3], dtype=torch.int32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
    )
    barriers = tuple(torch.empty(7, dtype=torch.int32, device=device) for _ in range(2))
    for live in (0, min(9, capacity), capacity):
        view = ids[:live]
        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard(), torch.cuda.graph(graph):
            decoder.decode(view, *outputs, packed_routes=packed, barriers=barriers)
        for pattern in (0, 1, 2, 3, 4, 0):
            values = [(i * 7 + i // 5) % experts for i in range(live)]
            if pattern == 1:
                values = [3 if i % 3 else -1 for i in range(live)]
            elif pattern == 2:
                values = [-1 if i % 2 else experts for i in range(live)]
            elif pattern == 3:
                values = [value if i < 9 else -1 for i, value in enumerate(values)]
            elif pattern == 4:
                values = [3 if i == 0 else -1 for i in range(live)]
            ids[:live].copy_(torch.tensor(values, dtype=dtype, device=device))
            for output in outputs:
                output.fill_(0xD3)
            for tensor in (*packed, *barriers):
                tensor.fill_(-42)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            assert all(torch.count_nonzero(tensor).item() == 0 for tensor in barriers)
            expected_routes = torch.full_like(packed[0], live)
            expected_experts = torch.full_like(packed[1], -1)
            offset = 0
            for expert in range(experts):
                routes = [i for i, value in enumerate(values) if value == expert]
                for index, route in enumerate(routes):
                    expected_routes[offset + index] = route
                blocks = (len(routes) + block_m - 1) // block_m
                expected_experts[offset // block_m : offset // block_m + blocks] = (
                    expert
                )
                offset += blocks * block_m
            torch.testing.assert_close(packed[0], expected_routes, rtol=0, atol=0)
            torch.testing.assert_close(packed[1], expected_experts, rtol=0, atol=0)
            assert packed[2].item() == offset
            active = sorted({value for value in values if 0 <= value < experts})
            for output, expected in zip(outputs, native, strict=True):
                reference = torch.full_like(expected, 0xD3)
                reference[active] = expected[active]
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        graph.reset()


def test_packed_route_compile_rejects_insufficient_decoder_threads():
    device = require_b12x()
    payload, _ = planes_and_native(128, 8, 16, dense=False)
    plane = Nvfp4CompactBlockPlane(
        upload_csf_block_payload(payload, device), groups_per_task=1
    )
    with pytest.raises(ValueError, match="CTA thread count"):
        compile_nvfp4_csf_pair(
            plane.geometry, plane.geometry, packed_routes=(128, 16, 2048, 128)
        )


@pytest.mark.parametrize("capacity", [1, 31, 32, 33, 63, 65, 96, 127])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_expansion_at_warp_route_boundaries(capacity, dtype):
    test_packed_expansion_preserves_bytes_routes_and_mutable_replay(
        capacity, 8, dtype, ((8, True), (8, True))
    )
