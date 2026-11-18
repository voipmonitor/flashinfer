"""Mapped CSF routes preserve exact bytes through live graph replay."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.csf_routing import compile_csf_active_experts
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder, decode_nvfp4_csf_pair
from b12x._lib.runtime_control import kernel_resolution_guard
from ..conftest import require_b12x
from .test_csf_blocks import planes_and_native
from .test_nvfp4_csf import _fixture
from b12x._lib.quant.nvfp4_csf_indexed import prepare_indexed_scales


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("routing", ["direct", "unique", "scan", "bitmap"])
def test_mapped_scale_bytes_with_aliases_and_invalid_routes(dtype, routing):
    device = require_b12x()
    experts, route_experts = 16, 21
    planes, outputs, native = [], [], []
    for rows, columns in ((128, 8), (256, 4)):
        payload, expected = planes_and_native(rows, columns, experts, dense=False)
        planes.append(Nvfp4CompactBlockPlane(upload_csf_block_payload(payload, device)))
        outputs.append(torch.empty_like(expected, device=device))
        native.append(expected.to(device))
    decoder = replace(
        Nvfp4CsfDecoder.prepare(*planes, *outputs),
        direct_routes=routing == "direct",
        mapped_routing_programs=tuple(
            compile_csf_active_experts(
                ids64, experts if routing == "bitmap" else 0, mapped=True
            )
            for ids64 in (False, True)
        ),
    )
    ids = torch.empty(256, dtype=dtype, device=device)
    mapping = torch.empty(route_experts, dtype=torch.int32, device=device)
    for live in (0, 9, 64, 256):
        graph = torch.cuda.CUDAGraph()
        with kernel_resolution_guard(), torch.cuda.graph(graph):
            if routing == "unique":
                decode_nvfp4_csf_pair(
                    *planes,
                    ids[:live],
                    *outputs,
                    mode=1,
                    expert_map=mapping,
                    program=decoder.mapped_programs[int(dtype == torch.int64)],
                )
            else:
                decoder.decode(ids[:live], *outputs, expert_map=mapping)
        for pattern in (0, 1, 2):
            mapped = [(i * 7 + 3) % experts for i in range(route_experts)]
            mapped[0], mapped[1], mapped[2], mapped[3] = 15, 15, -1, experts
            values = [i % (route_experts + 2) - 1 for i in range(live)]
            if pattern == 1:
                mapped[0] = mapped[1] = 5
                values = [i % 2 for i in range(live)]
            elif pattern == 2:
                values = [
                    -1 if i % 2 else (2**32 if dtype == torch.int64 else route_experts)
                    for i in range(live)
                ]
            mapping.copy_(torch.tensor(mapped, dtype=torch.int32, device=device))
            ids[:live].copy_(torch.tensor(values, dtype=dtype, device=device))
            for output in outputs:
                output.fill_(0xD3)
            decoder.active.fill_(-42)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            active = sorted(
                {
                    mapped[value]
                    for value in values
                    if 0 <= value < route_experts and 0 <= mapped[value] < experts
                }
            )
            if live >= 16 * experts and routing != "unique":
                active = list(range(experts))
            for output, expected in zip(outputs, native, strict=True):
                reference = torch.full_like(expected, 0xD3)
                reference[active] = expected[active]
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
            if 64 <= live < 16 * experts and routing in ("scan", "bitmap"):
                expected_active = torch.zeros_like(decoder.active)
                expected_active[active] = 1
                torch.testing.assert_close(
                    decoder.active, expected_active, rtol=0, atol=0
                )
        graph.reset()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_indexed_mapped_scale_replay(dtype):
    device = require_b12x()
    plane, expected = _fixture(128, 8, 0, device)
    outputs = tuple(
        torch.empty_like(expected).view(torch.float8_e4m3fn) for _ in range(2)
    )
    indexed = prepare_indexed_scales(plane)
    decoder = Nvfp4CsfDecoder.prepare(
        plane, plane, *outputs, indexed_scales=(indexed, indexed)
    )
    ids = torch.tensor(
        [0, 1, 2, -1, 2**32 if dtype == torch.int64 else 4], dtype=dtype, device=device
    )
    mapping = torch.empty(4, dtype=torch.int32, device=device)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard(), torch.cuda.graph(graph):
        decoder.decode(ids, *outputs, expert_map=mapping)
    for values, active in (
        ([2, 2, -1, 4], [2]),
        ([1, 3, 3, -1], [1, 3]),
        ([-1] * 4, []),
    ):
        mapping.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        for output in outputs:
            output.view(torch.uint8).fill_(0xD3)
        before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
        reference = torch.full_like(expected, 0xD3)
        reference[active] = expected[active]
        for output in outputs:
            torch.testing.assert_close(
                output.view(torch.uint8), reference, rtol=0, atol=0
            )
    graph.reset()
