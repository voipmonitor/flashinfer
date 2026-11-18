# SPDX-License-Identifier: Apache-2.0

"""TP16 score/ID gather with replicated aligned-prefix metadata.

Each row supplies sixteen prefix lengths in uint32 words, followed by an
all-prefixes-full flag and the maximum prefix length in sixteen-byte packets.
All ranks must supply identical metadata. Prefixes are aligned to four words;
all omitted words must repeat the pair (FP32 negative infinity, int32 minus one).
The recursive peer protocol, eight physical CTAs and bank lifetimes are retained.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _a2a_graph_epoch_arrive,
    _copy_16b,
)
from ._joint_cluster_cute import _threadfence_sys
from ._recursive_gather_cute import RecursiveGather


class PrefixGather(RecursiveGather):
    def __init__(
        self,
        world,
        rank,
        row_words,
        threads,
        blocks,
        chunks,
        acknowledge,
        prefix_stride,
    ):
        super().__init__(world, rank, row_words, threads, blocks, chunks, acknowledge)
        self.prefix_stride = prefix_stride

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        selector: cute.Pointer,
        stages: tuple,
        signals: tuple,
        rows: Int32,
        stride: Int64,
        offset: Int64,
        counter: Int64,
        prefixes: cute.Pointer,
        stream: cuda.CUstream,
    ):
        self.selected(
            source,
            output,
            selector,
            stages,
            signals,
            rows,
            stride,
            offset,
            counter,
            prefixes,
        ).launch(grid=(self.blocks, 1, 1), block=(self._threads, 1, 1), stream=stream)

    @cute.kernel
    def selected(
        self,
        source,
        output,
        selector,
        stages,
        signals,
        rows,
        stride,
        offset,
        counter,
        prefixes,
    ):
        full, maximum, row = Int32(1), Int32(0), Int32(0)
        while row < rows:
            full &= cute.arch.load(
                prefixes + Int64(row * self.prefix_stride + 16), Int32
            )
            maximum = cutlass.max(
                maximum,
                cute.arch.load(prefixes + Int64(row * self.prefix_stride + 17), Int32),
            )
            row += Int32(1)
        if full != Int32(0):
            self.exchange_full(
                source, output, selector, stages, signals, rows, stride, offset, counter
            )
        else:
            self.exchange_prefix(
                source,
                output,
                selector,
                stages,
                signals,
                rows,
                stride,
                offset,
                counter,
                prefixes,
                maximum,
            )

    @cute.jit
    def exchange_prefix(
        self,
        source,
        output,
        selector,
        stages,
        signals,
        rows,
        stride,
        offset,
        counter,
        prefixes,
        maximum,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        if profile == Uint32(8):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            bank = Int64(generation % Uint32(2)) * stride
            local = self._staging_words(stages[self._rank] + bank)
            lane = Int32(tid) % Int32(32)
            first = Int32(tid) // Int32(32) * Int32(8) + Int32(block)
            step = Int32(8 * self._warps_per_block)
            packs = Int32(self.row_words // 4)
            row_tiles = (maximum + Int32(31)) // Int32(32)
            tiles = rows * row_tiles
            for phase in cutlass.range_constexpr(4):
                group = 1 << phase
                peer = self._rank ^ group
                own_begin = self._rank & ~(group - 1)
                peer_begin = peer & ~(group - 1)
                remote = self._staging_words(stages[peer] + bank)
                for segment in cutlass.range_constexpr(group):
                    sending = own_begin + segment
                    destination = sending if sending < peer else sending - 1
                    tile = first
                    while tile < tiles:
                        row = tile % rows
                        pack = (tile // rows) * Int32(32) + lane
                        prefix = cute.arch.load(
                            prefixes + Int64(row * self.prefix_stride + sending), Int32
                        )
                        if pack < packs:
                            out_pack = (
                                Int64(row) * Int64(16) + Int64(sending)
                            ) * Int64(packs) + Int64(pack)
                            words = (
                                Uint32(0xFF800000),
                                Uint32(0xFFFFFFFF),
                                Uint32(0xFF800000),
                                Uint32(0xFFFFFFFF),
                            )
                            if pack < prefix // Int32(4):
                                if cutlass.const_expr(phase == 0):
                                    source_pack = Int64(row) * Int64(packs) + Int64(
                                        pack
                                    )
                                    words = ld_global_v4_u32(
                                        (source + source_pack * Int64(4)).toint()
                                    )
                                else:
                                    words = ld_global_v4_u32(
                                        (output + out_pack * Int64(4)).toint()
                                    )
                                destination_pack = (
                                    Int64(row) * Int64(15) + Int64(destination)
                                ) * Int64(packs) + Int64(pack)
                                st_global_v4_u32(
                                    (remote + destination_pack * Int64(4)).toint(),
                                    *words,
                                )
                                if cutlass.const_expr(phase == 0):
                                    st_global_v4_u32(
                                        (output + out_pack * Int64(4)).toint(), *words
                                    )
                        tile += step
                _threadfence_sys()
                cute.arch.sync_threads()
                self._pair_barrier(signals, peer, block, tid)
                for segment in cutlass.range_constexpr(group):
                    receiving = peer_begin + segment
                    origin = receiving if receiving < self._rank else receiving - 1
                    tile = first
                    while tile < tiles:
                        row = tile % rows
                        pack = (tile // rows) * Int32(32) + lane
                        prefix = cute.arch.load(
                            prefixes + Int64(row * self.prefix_stride + receiving),
                            Int32,
                        )
                        if pack < packs:
                            incoming_pack = (
                                Int64(row) * Int64(15) + Int64(origin)
                            ) * Int64(packs) + Int64(pack)
                            out_pack = (
                                Int64(row) * Int64(16) + Int64(receiving)
                            ) * Int64(packs) + Int64(pack)
                            words = (
                                Uint32(0xFF800000),
                                Uint32(0xFFFFFFFF),
                                Uint32(0xFF800000),
                                Uint32(0xFFFFFFFF),
                            )
                            if pack < prefix // Int32(4):
                                words = ld_global_v4_u32(
                                    (local + incoming_pack * Int64(4)).toint()
                                )
                                st_global_v4_u32(
                                    (output + out_pack * Int64(4)).toint(), *words
                                )
                        tile += step
                # All physical CTAs retain their peer generations. Empty payloads
                # still acknowledge readiness and obey the same bank lifetimes.
                _threadfence_sys()
                cute.arch.sync_threads()
                if cutlass.const_expr(self.acknowledge_consumption):
                    self._pair_barrier(signals, peer, block, tid)
            # Padding never participates in forwarding. Fill the complete suffix
            # once with contiguous local stores after the last peer exchange.
            flat = Int64(block) * Int64(self._threads) + Int64(tid)
            total = Int64(rows) * Int64(16) * Int64(packs)
            while flat < total:
                row = flat // (Int64(16) * Int64(packs))
                owner = flat // Int64(packs) % Int64(16)
                pack = flat % Int64(packs)
                prefix = cute.arch.load(
                    prefixes + row * Int64(self.prefix_stride) + owner, Int32
                )
                if pack >= Int64(prefix // Int32(4)):
                    st_global_v4_u32(
                        (output + flat * Int64(4)).toint(),
                        Uint32(0xFF800000),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFF800000),
                        Uint32(0xFFFFFFFF),
                    )
                flat += Int64(8 * self._threads)
            cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )

    @cute.jit
    def exchange_full(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        if profile == Uint32(8):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            bank = Int64(generation % Uint32(2)) * stride
            local = self._staging_words(stages[self._rank] + bank)
            lane = Int32(tid) % Int32(32)
            first = Int32(block) * Int32(self._warps_per_block) + Int32(tid) // Int32(
                32
            )
            step = Int32(8 * self._warps_per_block)
            packs = Int32(self.row_words // 4)
            row_tiles = (packs + Int32(31)) // Int32(32)
            tiles = rows * row_tiles
            for phase in cutlass.range_constexpr(4):
                group = 1 << phase
                peer = self._rank ^ group
                own_begin = self._rank & ~(group - 1)
                peer_begin = peer & ~(group - 1)
                remote = self._staging_words(stages[peer] + bank)
                # A CTA owns the same row/column tiles in every rank segment.
                # It only forwards bytes that it completed in an earlier phase.
                for segment in cutlass.range_constexpr(group):
                    sending = own_begin + segment
                    destination = sending if sending < peer else sending - 1
                    tile = first
                    while tile < tiles:
                        row = tile // row_tiles
                        pack = (tile - row * row_tiles) * Int32(32) + lane
                        if pack < packs:
                            out_pack = (
                                Int64(row) * Int64(16) + Int64(sending)
                            ) * Int64(packs) + Int64(pack)
                            if cutlass.const_expr(phase == 0):
                                source_pack = Int64(row) * Int64(packs) + Int64(pack)
                                words = ld_global_v4_u32(
                                    (source + source_pack * Int64(4)).toint()
                                )
                                st_global_v4_u32(
                                    (output + out_pack * Int64(4)).toint(), *words
                                )
                            else:
                                words = ld_global_v4_u32(
                                    (output + out_pack * Int64(4)).toint()
                                )
                            destination_pack = (
                                Int64(row) * Int64(15) + Int64(destination)
                            ) * Int64(packs) + Int64(pack)
                            st_global_v4_u32(
                                (remote + destination_pack * Int64(4)).toint(), *words
                            )
                        tile += step
                _threadfence_sys()
                cute.arch.sync_threads()
                self._pair_barrier(signals, peer, block, tid)
                for segment in cutlass.range_constexpr(group):
                    receiving = peer_begin + segment
                    origin = receiving if receiving < self._rank else receiving - 1
                    tile = first
                    while tile < tiles:
                        row = tile // row_tiles
                        pack = (tile - row * row_tiles) * Int32(32) + lane
                        if pack < packs:
                            incoming_pack = (
                                Int64(row) * Int64(15) + Int64(origin)
                            ) * Int64(packs) + Int64(pack)
                            out_pack = (
                                Int64(row) * Int64(16) + Int64(receiving)
                            ) * Int64(packs) + Int64(pack)
                            _copy_16b(
                                local + incoming_pack * Int64(4),
                                output + out_pack * Int64(4),
                            )
                        tile += step
                # With consumption acknowledgements, READY and ACK use separate
                # paired generations. Otherwise the two receive banks protect
                # staging: completing the intervening collective requires every
                # rank to enter it, hence finish this kernel before bank reuse.
                _threadfence_sys()
                cute.arch.sync_threads()
                if cutlass.const_expr(self.acknowledge_consumption):
                    self._pair_barrier(signals, peer, block, tid)
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )


def _pointer(dtype, address=16, alignment=16):
    return make_ptr(
        dtype, address, mem_space=cute.AddressSpace.gmem, assumed_align=alignment
    )


@program_cache(scope="preparation")
def compile_prefix_gather(world, rank, row_words, prefix_stride):
    if (
        world != 16
        or not 0 <= rank < world
        or type(row_words) is not int
        or not 0 < row_words < 2**27
        or row_words % 4
        or type(prefix_stride) is not int
        or prefix_stride < 18
    ):
        raise ValueError(
            "prefix gather requires TP16, aligned positive rows and eighteen metadata columns"
        )
    operation = PrefixGather(world, rank, row_words, 128, 8, 0, True, prefix_stride)
    raw = compile_kernel(
        operation,
        _pointer(Uint32),
        _pointer(Uint32),
        _pointer(Uint32, alignment=4),
        tuple(_pointer(Uint8) for _ in range(16)),
        tuple(_pointer(Uint32, alignment=4) for _ in range(16)),
        Int32(1),
        Int64(256),
        Int64(0),
        Int64(512),
        _pointer(Int32, alignment=4),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.candidate_prefix_gather",
            1,
            (world, rank, row_words, prefix_stride),
        ),
    )

    def run(
        source,
        output,
        selector,
        stages,
        signals,
        rows,
        stride,
        offset,
        counter,
        prefixes,
    ):
        if len(stages) != world or len(signals) != world or not 1 <= rows <= 8:
            raise ValueError(
                "prefix gather requires sixteen peer addresses and one through eight rows"
            )
        if source % 16 or output % 16 or prefixes % 4:
            raise ValueError(
                "prefix gather requires sixteen-byte data and four-byte metadata alignment"
            )
        raw(
            _pointer(Uint32, source),
            _pointer(Uint32, output),
            _pointer(Uint32, selector, 4),
            tuple(_pointer(Uint8, address) for address in stages),
            tuple(_pointer(Uint32, address, 4) for address in signals),
            Int32(rows),
            Int64(stride),
            Int64(offset),
            Int64(counter),
            _pointer(Int32, prefixes, 4),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
