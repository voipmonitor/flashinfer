"""Check native shared-memory weight loads against independent FP4 lane order."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import pytest
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.utils.device import current_cuda_stream
from b12x.moe._shared.kernels.w4a16.kernel.gemm import W4A16GemmKernel

from ..conftest import require_b12x


class _NativePair:
    def __init__(self, tile_n, tile_k, block_m, pipe, paired):
        self.consumer = W4A16GemmKernel(
            size_m=block_m,
            size_n=tile_n,
            size_k=tile_k,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=tile_n,
            tile_k=tile_k,
            moe_block_size=block_m,
            max_m_blocks=1,
            element_dtype="bf16",
            weight_layout="modelopt",
            scale_format="e8m0_k32",
        )
        self.pipe = pipe
        self.paired = paired
        self.words = tile_n * tile_k // 8
        self.offset_words = (
            self.consumer.sh_b_off + pipe * self.consumer.b_sh_stage
        ) * 4

    @cute.jit
    def __call__(self, source, output, stream):
        self.kernel(source, output).launch(
            grid=[1, 1, 1], block=[self.consumer.cta_threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, source, output):
        tid = Int32(cute.arch.thread_idx()[0])
        shared = SmemAllocator().allocate_tensor(
            Uint32, cute.make_layout(self.offset_words + self.words), 16
        )
        for i in cutlass.range_constexpr(self.words // self.consumer.cta_threads):
            index = tid + Int32(i * self.consumer.cta_threads)
            physical = self.consumer._modelopt_smem_permuted_offset(index * Int32(4))
            shared[self.offset_words + physical // Int32(4)] = source[index]
        cute.arch.sync_threads()
        address = shared_ptr_to_u32(shared.iterator)
        read_index = tid + (tid // Int32(self.consumer.b_sh_stride)) * Int32(
            self.consumer.b_sh_stride * (self.consumer.b_sh_wr_iters - 1)
        )
        if cutlass.const_expr(self.paired):
            regs = cute.make_rmem_tensor((3, 4), Uint32)
            for pair in cutlass.range_constexpr(self.consumer.b_sh_wr_iters // 2):
                self.consumer._load_b_registers_modelopt_shared_pair(
                    regs, address, read_index, Int32(self.pipe), Int32(pair * 2)
                )
                for jj in cutlass.range_constexpr(4):
                    output[tid, pair * 2, jj] = regs[0, jj]
                    output[tid, pair * 2 + 1, jj] = regs[2, jj]

        else:
            for kk in cutlass.range_constexpr(self.consumer.b_sh_wr_iters):
                q0, q1, q2, q3 = self.consumer._load_b_registers_modelopt_shared(
                    address, read_index, Int32(self.pipe), Int32(kk)
                )
                output[tid, kk, 0] = q0
                output[tid, kk, 1] = q1
                output[tid, kk, 2] = q2
                output[tid, kk, 3] = q3


@pytest.mark.parametrize(
    "tile_n,tile_k,block_m",
    [(128, 64, 16), (64, 128, 16), (128, 128, 16), (256, 64, 16), (512, 32, 8)],
)
@pytest.mark.parametrize("pipe", [0, 1])
@pytest.mark.parametrize("paired", [False, True])
def test_native_paired_shared_load_preserves_fp4_lane_order(
    tile_n, tile_k, block_m, pipe, paired
):
    device = require_b12x()
    generator = torch.Generator().manual_seed(940 + tile_n + tile_k + pipe)
    source_bytes = torch.randint(
        0, 256, (tile_n, tile_k // 2), dtype=torch.uint8, generator=generator
    )
    consumer = _NativePair(tile_n, tile_k, block_m, pipe, paired)
    iterations = consumer.consumer.b_sh_wr_iters
    threads = consumer.consumer.cta_threads
    expected = torch.zeros((threads, iterations, 4), dtype=torch.int64)
    for thread in range(threads):
        warp, lane = divmod(thread, 32)
        n_base = (warp % (tile_n // 64)) * 64 + lane // 4
        k_base = (warp // (tile_n // 64)) * iterations * 8 + lane % 4
        for kk in range(iterations):
            for jj in range(4):
                n, k = n_base + jj * 16, k_base + kk * 8
                values = [
                    int(source_bytes[n, k]),
                    int(source_bytes[n, k + 4]),
                    int(source_bytes[n + 8, k]),
                    int(source_bytes[n + 8, k + 4]),
                ]
                for index, value in enumerate(values):
                    expected[thread, kk, jj] |= (value & 15) << (index * 4)
                    expected[thread, kk, jj] |= (value >> 4) << (16 + index * 4)
    source = source_bytes.flatten().view(torch.uint32).to(device)
    output = torch.empty(expected.shape, device=device, dtype=torch.uint32)
    arrays = tuple(from_dlpack(t, assumed_align=16) for t in (source, output))
    kernel = compile_kernel(
        consumer,
        *arrays,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_fields(
            "test.native-paired-weight-load",
            1,
            ("tile", (tile_n, tile_k)),
            ("block_m", block_m),
            ("pipe", pipe),
            ("paired", paired),
        ),
    )
    kernel(*arrays, current_cuda_stream())
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu().to(torch.int64), expected, rtol=0, atol=0)
