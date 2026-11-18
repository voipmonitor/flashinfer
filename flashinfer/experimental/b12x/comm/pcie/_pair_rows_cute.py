# SPDX-License-Identifier: Apache-2.0
"""Gather four BF16[224]/FP32[56] row pairs with one CTA per row and one warp per peer."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _DCPA2ABase, _GRAPH_ARRIVED_INDEX, _GRAPH_EPOCH_INDEX, _MAX_BLOCKS,
    _SLOT_ALIGNMENT, _a2a_graph_epoch_arrive, _copy_16b, _u32_ptr, _u8_ptr,
)
from ._dcp_cute_common import block_pair_barrier


class RowBlockAllGatherPair(_DCPA2ABase):
    @cute.jit
    def __call__(self, first: cute.Pointer, second: cute.Pointer, out_first: cute.Pointer,
                 out_second: cute.Pointer, stages: tuple, signals: tuple, stride: Int32,
                 stream: cuda.CUstream):
        self.kernel(first, second, out_first, out_second, stages, signals, stride).launch(
            grid=(4, 1, 1), block=(512, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, first, second, out_first, out_second, staging, signals, stride):
        tid, _, _ = cute.arch.thread_idx()
        first_packs, second_packs = Int32(28), Int32(14)
        epoch = ld_relaxed_gpu_u32((signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint())
        slot_offset = Int64(epoch % Uint32(2)) * Int64(stride) * Int64(_SLOT_ALIGNMENT)
        row, _, _ = cute.arch.block_idx()
        combined = first_packs + second_packs
        base = Int64(row) * Int64(combined) * 4
        local = self._staging_words(staging[self._rank] + slot_offset) + base
        if tid < first_packs:
            _copy_16b(first + Int64(row * first_packs + tid) * 4, local + Int64(tid) * 4)
        if tid < second_packs:
            _copy_16b(
                second + Int64(row * second_packs + tid) * 4,
                local + Int64(first_packs + tid) * 4,
            )
        block_pair_barrier(
            signals, self_signal=signals[self._rank], rank=self._rank,
            world_size=self._world_size, max_blocks=_MAX_BLOCKS,
        )
        # One warp reads both fields of one peer. Each field fits in 32 aligned packs.
        lane, warp = tid % 32, tid // 32
        for peer in cutlass.range_constexpr(self._world_size):
            if warp == Int32(peer):
                source = self._staging_words(staging[peer] + slot_offset) + base
                out_row = Int64(row) * self._world_size + peer
                if lane < first_packs:
                    _copy_16b(source + Int64(lane) * 4,
                              out_first + (out_row * first_packs + lane) * 4)
                if lane < second_packs:
                    _copy_16b(source + Int64(first_packs + lane) * 4,
                              out_second + (out_row * second_packs + lane) * 4)
        cute.arch.sync_threads()
        if tid == 0:
            flags = signals[self._rank]
            _a2a_graph_epoch_arrive(
                (flags + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                (flags + Int64(_GRAPH_ARRIVED_INDEX)).toint(), Uint32(4),
            )


@program_cache
def compile_pair_rows(rank):
    launch = RowBlockAllGatherPair(16, rank, 512, True)
    word = _u32_ptr(16, align=4)
    program = compile_kernel(
        launch, word, word, word, word, (_u8_ptr(16),) * 16, (word,) * 16, 1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.dcp_a2a.pair_rows", 1, (rank,), labels=("rank",)
        ),
    )

    def run(local_first, local_second, bias, out_first, out_second, ids, stages, signals,
            rows, first_packs, second_packs, stride):
        if (rows, first_packs, second_packs) != (4, 28, 14):
            raise ValueError("paired gather requires four rows with 28 and 14 aligned packs")
        program(
            _u32_ptr(local_first, align=4), _u32_ptr(local_second, align=4),
            _u32_ptr(out_first, align=4), _u32_ptr(out_second, align=4),
            tuple(_u8_ptr(ptr) for ptr in stages), tuple(_u32_ptr(ptr, align=4) for ptr in signals),
            stride, current_cuda_stream(),
        )

    return attach_programs(run, program)
