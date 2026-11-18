"""Check the exact packed BF16 bits of E8M0 compute-scale conversion."""

import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.intrinsics.packed import packed_dequant_e8m0x4_to_bfloat2x2
from b12x._lib.utils.device import current_cuda_stream

from ..conftest import require_b12x


class _Decode:
    @cute.jit
    def __call__(self, source, output, count: Int32, stream):
        self.kernel(source, output, count).launch(
            grid=[cute.ceil_div(count, 128), 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, source, output, count: Int32):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = Int32(block) * 128 + Int32(thread)
        if index < count:
            lo, hi = packed_dequant_e8m0x4_to_bfloat2x2(Uint32(source[index]))
            output[index, 0] = lo
            output[index, 1] = hi


def test_e8m0_compute_scale_all_byte_pairs():
    device = require_b12x()
    index = torch.arange(65536, dtype=torch.int64, device=device)
    b0, b2 = index % 256, index // 256
    b1, b3 = (b0 * 17 + 31) % 256, (b2 * 19 + 7) % 256
    source = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(torch.uint32)
    expected = torch.stack(
        (((b0 + 71) << 7) | ((b2 + 71) << 23), ((b1 + 71) << 7) | ((b3 + 71) << 23)),
        dim=1,
    ).to(torch.uint32)
    output = torch.empty_like(expected)
    arrays = tuple(from_dlpack(t, assumed_align=16) for t in (source, output))
    stream = current_cuda_stream()
    kernel = compile_kernel(
        _Decode(),
        *arrays,
        Int32(source.numel()),
        stream,
        compile_spec=KernelCompileSpec.from_fields(
            "test.e8m0-packed-decode", 1, ("capacity", source.numel())
        ),
    )
    kernel(*arrays, Int32(source.numel()), stream)
    torch.cuda.synchronize()
    assert torch.equal(output, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(*arrays, Int32(source.numel()), current_cuda_stream())
    source.copy_(source.flip(0))
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected.flip(0))
