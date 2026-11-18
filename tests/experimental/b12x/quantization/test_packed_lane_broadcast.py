"""Check packed 16-bit lane broadcasts, including live captured inputs."""

import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.intrinsics.packed import bfloat2_broadcast_lane
from b12x._lib.utils.device import current_cuda_stream

from ..conftest import require_b12x


class _Broadcast:
    @cute.jit
    def __call__(self, source, lanes, output, count: Int32, stream):
        self.kernel(source, lanes, output, count).launch(
            grid=[cute.ceil_div(count, 128), 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, source, lanes, output, count: Int32):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = Int32(block) * 128 + Int32(thread)
        if index < count:
            value = Uint32(source[index])
            output[index, 0] = bfloat2_broadcast_lane(value, Int32(0))
            output[index, 1] = bfloat2_broadcast_lane(value, Int32(1))
            output[index, 2] = bfloat2_broadcast_lane(value, Int32(lanes[index]))


def test_packed_lane_broadcast_all_halfword_values():
    device = require_b12x()
    low = torch.arange(65536, dtype=torch.int64, device=device)
    high = (low * 17 + 31) % 65536
    source = (low | (high << 16)).to(torch.uint32)
    lanes = ((low % 3) - 1).to(torch.int32)
    selected = torch.where(lanes == 0, low, high)
    expected = torch.stack(
        (low | (low << 16), high | (high << 16), selected | (selected << 16)), dim=1
    ).to(torch.uint32)
    output = torch.empty_like(expected)
    arrays = tuple(from_dlpack(t, assumed_align=16) for t in (source, lanes, output))
    kernel = compile_kernel(
        _Broadcast(),
        *arrays,
        Int32(source.numel()),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_fields(
            "test.packed-lane-broadcast", 1, ("capacity", source.numel())
        ),
    )
    kernel(*arrays, Int32(source.numel()), current_cuda_stream())
    torch.cuda.synchronize()
    assert torch.equal(output, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(*arrays, Int32(source.numel()), current_cuda_stream())
    source.copy_(source.flip(0))
    lanes.copy_(lanes.flip(0))
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected.flip(0))
