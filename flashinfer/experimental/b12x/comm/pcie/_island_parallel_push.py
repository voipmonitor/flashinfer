# SPDX-License-Identifier: Apache-2.0
"""Four-group TP16 reduction with concurrent readiness publication by the three peer lanes."""

from collections.abc import Sequence

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Uint32
from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import cuda_stream_from_int_or_current, cuda_stream_to_int, current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._island_rs_cute import (
    _IslandRSLaunch, _AG_ARRIVED, _ISLAND_SIZE, _MAX_ISLANDS, _MAX_WORLD_SIZE, _RS_ARRIVED,
    _XS_ARRIVED, _fence_sc_sys, _flag_address, _store_release_sys_u32, _wait_for,
    pack_f32x2_to_bf16x2, unpack_bf16x2,
)


class ParallelPushIslandRS(_IslandRSLaunch):
    @cute.jit
    def _signal_peer_phase(
        self, slabs, self_base, tidx, bidx, region: int, generation, partners: bool
    ):
        own = self._island if cutlass.const_expr(partners) else self._lane
        peer_base = Int64(0)
        for peer in cutlass.range_constexpr(_ISLAND_SIZE):
            if cutlass.const_expr(peer != own):
                if tidx == Int32(peer):
                    if cutlass.const_expr(partners):
                        peer_base = Int64(
                            slabs[peer * _ISLAND_SIZE + self._lane].toint()
                        )
                    else:
                        peer_base = Int64(
                            slabs[self._island * _ISLAND_SIZE + peer].toint()
                        )
        # The producer CTA barrier precedes this phase. Every signaling lane
        # retains a system fence and release/acquire ordering; the caller's
        # next CTA barrier joins all three peers before payload consumption.
        if tidx < Int32(_ISLAND_SIZE) and tidx != Int32(own):
            _fence_sc_sys()
            _store_release_sys_u32(
                _flag_address(peer_base, region, bidx, Int32(own)), generation
            )
            _wait_for(
                _flag_address(self_base, region, bidx, tidx),
                generation,
                self._wait_nanosleep_cycles,
            )

    @cute.jit
    def _participate(
        self,
        slabs,
        input_ptr,
        output_ptr,
        stage_offset,
        part_offset,
        final_offset,
        quarter_capacity,
        elements,
        data_generation,
        generation,
        self_base,
        tidx,
        bidx,
        gdim,
    ):
        if data_generation % Uint32(2) != Uint32(0):
            stage_offset += quarter_capacity * Int64(_ISLAND_SIZE * 4)
            part_offset += quarter_capacity * Int64(4)
            final_offset += quarter_capacity * Int64(_ISLAND_SIZE * 4)

        # Everything below counts in bf16x2 words; callers guarantee an even
        # element count so there is no scalar tail to chase.
        pairs = elements // Int64(2)
        quarter = (pairs + Int64(_ISLAND_SIZE - 1)) // Int64(_ISLAND_SIZE)
        stride = Int64(gdim) * Int64(self._threads)
        start = Int64(bidx) * Int64(self._threads) + Int64(tidx)

        input_words = cute.recast_ptr(input_ptr.align(4), dtype=Uint32)
        output_words = cute.recast_ptr(output_ptr.align(4), dtype=Uint32)

        # Each rank reduces one quarter. Local push delivers that quarter's
        # four input contributions into its staging plane before the reduction.
        self_stage = cute.recast_ptr(
            cute.make_ptr(
                cutlass.BFloat16,
                self_base + stage_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ).align(4),
            dtype=Uint32,
        )
        # A block publishes the same shifted residue class that the matching
        # block on each destination lane consumes after its per-block arrival
        # flag. This keeps data ownership aligned with synchronization even
        # when a quarter boundary is not divisible by the launch stride.
        for lane in cutlass.range_constexpr(_ISLAND_SIZE):
            begin = Int64(lane) * quarter
            stop = begin + quarter
            if stop > pairs:
                stop = pairs
            index = start
            destination = self._island * _ISLAND_SIZE + lane
            destination_stage = cute.recast_ptr(
                cute.make_ptr(
                    cutlass.BFloat16,
                    Int64(slabs[destination].toint()) + stage_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ).align(4),
                dtype=Uint32,
            )
            while index < stop - begin:
                destination_stage[Int64(self._lane) * quarter + index] = (
                    input_words[begin + index]
                )
                index += stride
        cute.arch.barrier()
        self._signal_peer_phase(
            slabs, self_base, tidx, bidx, _RS_ARRIVED, generation, False
        )
        cute.arch.barrier()

        mine_begin = Int64(self._lane) * quarter
        mine_stop = mine_begin + quarter
        if mine_stop > pairs:
            mine_stop = pairs
        mine_length = mine_stop - mine_begin

        self_part = cute.recast_ptr(
            cute.make_ptr(
                cutlass.BFloat16,
                self_base + part_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ).align(4),
            dtype=Uint32,
        )
        index = start
        # Read four independent stride positions before consuming peer values.
        # Each block retains the residue classes covered by its arrival flag.
        while index + Int64(3) * stride < mine_length:
            words = ()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                peer = self._island * _ISLAND_SIZE + p
                peer_stage = self_stage
                peer_begin = Int64(p) * quarter
                for item in cutlass.range_constexpr(4):
                    words += (peer_stage[peer_begin + index + Int64(item) * stride],)
            for item in cutlass.range_constexpr(4):
                total_lo = Float32(0.0)
                total_hi = Float32(0.0)
                for p in cutlass.range_constexpr(_ISLAND_SIZE):
                    lo, hi = unpack_bf16x2(words[p * 4 + item])
                    total_lo += lo
                    total_hi += hi
                self_part[index + Int64(item) * stride] = pack_f32x2_to_bf16x2(
                    total_lo, total_hi
                )
            index += Int64(4) * stride
        while index < mine_length:
            total_lo = Float32(0.0)
            total_hi = Float32(0.0)
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                peer = self._island * _ISLAND_SIZE + p
                peer_stage = self_stage
                peer_begin = Int64(p) * quarter
                lo, hi = unpack_bf16x2(peer_stage[peer_begin + index])
                total_lo += lo
                total_hi += hi
            self_part[index] = pack_f32x2_to_bf16x2(total_lo, total_hi)
            index += stride
        cute.arch.barrier()

        # The XS mailbox publishes partial-ready (2*g), then final-ready (2*g+1)
        # for scattered stripes. A later tag also certifies the still-live partials.
        # Both modes sum island partials in the same order before BF16 conversion.
        self._signal_peer_phase(
            slabs, self_base, tidx, bidx, _XS_ARRIVED, generation * Uint32(2), True
        )
        cute.arch.barrier()

        self_final = cute.recast_ptr(
            cute.make_ptr(
                cutlass.BFloat16,
                self_base + final_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ).align(4),
            dtype=Uint32,
        )
        # Distribute stride positions across islands without changing which
        # CTA owns their publication flags. The final plane already reserves a
        # full vector and can hold both partial and assembled final quarters.
        cross_stride = stride
        index = start
        # Read four independent cross_stride positions before consuming peer values.
        # Each block retains the residue classes covered by its arrival flag.
        while index + Int64(3) * cross_stride < mine_length:
            words = ()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                partner = p * _ISLAND_SIZE + self._lane
                partner_part = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[partner].toint()) + part_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                for item in cutlass.range_constexpr(4):
                    words += (partner_part[index + Int64(item) * cross_stride],)
            for item in cutlass.range_constexpr(4):
                total_lo = Float32(0.0)
                total_hi = Float32(0.0)
                for p in cutlass.range_constexpr(_ISLAND_SIZE):
                    lo, hi = unpack_bf16x2(words[p * 4 + item])
                    total_lo += lo
                    total_hi += hi
                value = pack_f32x2_to_bf16x2(total_lo, total_hi)
                self_final[mine_begin + index + Int64(item) * cross_stride] = value
                output_words[mine_begin + index + Int64(item) * cross_stride] = value
            index += Int64(4) * cross_stride
        while index < mine_length:
            total_lo = Float32(0.0)
            total_hi = Float32(0.0)
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                partner = j * _ISLAND_SIZE + self._lane
                partner_part = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[partner].toint()) + part_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                lo, hi = unpack_bf16x2(partner_part[index])
                total_lo += lo
                total_hi += hi
            value = pack_f32x2_to_bf16x2(total_lo, total_hi)
            self_final[mine_begin + index] = value
            output_words[mine_begin + index] = value
            index += cross_stride
        cute.arch.barrier()


        # Publish completed quarters. Pushed quarters occupy disjoint ranges
        # in each recipient's existing final plane and are copied locally.
        for p in cutlass.range_constexpr(_ISLAND_SIZE):
            if cutlass.const_expr(p != self._lane):
                push_peer = self._island * _ISLAND_SIZE + p
                destination_final = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[push_peer].toint()) + final_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                index = start
                while index < mine_length:
                    destination_final[mine_begin + index] = self_final[
                        mine_begin + index
                    ]
                    index += stride
        cute.arch.barrier()
        self._signal_peer_phase(
            slabs, self_base, tidx, bidx, _AG_ARRIVED, generation, False
        )
        cute.arch.barrier()

        for p in cutlass.range_constexpr(_ISLAND_SIZE):
            if cutlass.const_expr(p != self._lane):
                peer = self._island * _ISLAND_SIZE + p
                peer_final = self_final
                begin = Int64(p) * quarter
                stop = begin + quarter
                if stop > pairs:
                    stop = pairs
                index = start
                while index < stop - begin:
                    output_words[begin + index] = peer_final[begin + index]
                    index += stride


@program_cache
def compile_parallel_push(rank, threads):
    if threads not in (512, 1024):
        raise ValueError("parallel peer publication requires 512 or 1024 threads")
    launch = ParallelPushIslandRS(16, rank, threads=threads, wait_nanosleep_cycles=24,
                                  cross_island_rsag=False, local_push=True)
    slab_placeholder = make_ptr(
        cutlass.Uint8,
        256,
        cute.AddressSpace.gmem,
        assumed_align=256,
    )
    raw = b12x_compile(
        launch,
        *(slab_placeholder for _ in range(_MAX_WORLD_SIZE)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island_rs.parallel_push", 1, (rank, threads), labels=("rank", "threads"),
        ),
    )

    def run(
        slab_addresses: Sequence[int],
        input_address: int,
        output_address: int,
        stage_offset: int,
        part_offset: int,
        final_offset: int,
        quarter_capacity: int,
        elements: int,
        blocks: int,
        stream: object = None,
        active_blocks: int | None = None,
    ) -> None:
        """Launch the compiled island reduce-scatter with runtime arguments."""
        if len(slab_addresses) != 16:
            raise ValueError(
                f"expected sixteen slab addresses, got {len(slab_addresses)}"
            )
        if blocks != 64 or active_blocks != 4:
            raise ValueError("parallel peer publication requires four participants in a 64-CTA channel")
        raw(
            *(
                make_ptr(
                    cutlass.Uint8,
                    int(address),
                    cute.AddressSpace.gmem,
                    assumed_align=256,
                )
                for address in slab_addresses
            ),
            make_ptr(
                cutlass.BFloat16,
                input_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            make_ptr(
                cutlass.BFloat16,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            stage_offset,
            part_offset,
            final_offset,
            quarter_capacity,
            elements,
            blocks,
            active_blocks,
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)
