"""CuTeDSL TP16 all-reduce over four four-rank islands.

Rank ``island * 4 + lane`` owns one quarter of the vector and maps six peers:
the other three lanes and the same lane in each other island. Local-push RSAG
streams ordered BF16 partials through four warp groups with slice readiness.
Other specializations use per-CTA phase readiness over the same double banks.
"""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Callable, Sequence
from typing import Tuple

from b12x._lib.compile_plan import attach_programs
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import (
    cuda_stream_from_int_or_current,
    cuda_stream_to_int,
    current_cuda_stream,
)
from b12x._lib.utils.pointers import make_ptr
from ._joint_startup_cute import _globaltimer


GROUP_TIMING_FIELDS = (
    "generation",
    "entered_ns",
    "ready_ns",
    "before_fence_ns",
    "fence_done_ns",
    "published_ns",
    "ag_wait_entered_ns",
    "ag_ready_ns",
    *(f"warp_completed_{warp}_ns" for warp in range(8)),
)
_GROUP_TIMING_WORDS = 16


@cute.jit
def _record_group_time(records, base, field: cutlass.Constexpr, enabled, group_tid):
    if enabled:
        if group_tid == Int32(0):
            records[base + Int64(field)] = _globaltimer()


@cute.jit
def _start_group_timing(records, base, enabled, generation, group_tid):
    if enabled:
        if group_tid == Int32(0):
            records[base] = Int64(generation)
            records[base + Int64(1)] = _globaltimer()


@cute.jit
def _finish_group_timing(records, base, enabled, group_tid):
    if enabled:
        if group_tid % Int32(32) == Int32(0):
            records[base + Int64(8) + Int64(group_tid // Int32(32))] = _globaltimer()


@dsl_user_op
def _atomic_add_global_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _load_acquire_sys_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_release_sys_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sc_sys(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.sc.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _nanosleep(cycles: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Uint32(cycles).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _wait_for(address: Int64, generation: Uint32, cycles: int) -> None:
    observed = _load_acquire_sys_u32(address)
    # A later ready tag also suffices. Signed modular comparison preserves this
    # ordering across wrap; the two-slot protocol bounds producer advancement.
    while Int32(observed - generation) < Int32(0):
        if cutlass.const_expr(cycles > 0):
            _nanosleep(Uint32(cycles))
        observed = _load_acquire_sys_u32(address)


@cute.jit
def _flag_address(base: Int64, region: int, block: Int32, index: Int32) -> Int64:
    return (
        base
        + Int64(region)
        + (Int64(block) * Int64(_ISLAND_SIZE) + Int64(index)) * Int64(128)
    )


@dsl_user_op
def unpack_bf16x2(value: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f32x2_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Match two scalar ``__float2bfloat16`` conversions without saturation."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 blo, bhi;
                cvt.rn.bf16.f32 blo, $1;
                cvt.rn.bf16.f32 bhi, $2;
                mov.b32 $0, {blo, bhi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _load_payload_words(pointer, index, width: cutlass.Constexpr[int]):
    if cutlass.const_expr(width == 4):
        return ld_global_v4_u32(Int64(pointer.toint()) + index * Int64(4))
    else:
        return (pointer[index],)


@cute.jit
def _store_payload_words(pointer, index, words, width: cutlass.Constexpr[int]):
    if cutlass.const_expr(width == 4):
        st_global_v4_u32(
            Int64(pointer.toint()) + index * Int64(4),
            words[0],
            words[1],
            words[2],
            words[3],
        )
    else:
        pointer[index] = words[0]


@cute.jit
def _accumulate_payload_words(totals, words, width: cutlass.Constexpr[int]):
    result = ()
    for item in cutlass.range_constexpr(width):
        lo, hi = unpack_bf16x2(words[item])
        result += (totals[item * 2] + lo, totals[item * 2 + 1] + hi)
    return result


@cute.jit
def _pack_payload_words(totals, width: cutlass.Constexpr[int]):
    result = ()
    for item in cutlass.range_constexpr(width):
        result += (pack_f32x2_to_bf16x2(totals[item * 2], totals[item * 2 + 1]),)
    return result


_ISLAND_SIZE = 4
_MAX_ISLANDS = 4
_MAX_WORLD_SIZE = _ISLAND_SIZE * _MAX_ISLANDS

# Generation counters precede three regions with one cache line per CTA and peer.
MAX_BLOCKS = 64
_ACTIVE_EPOCH_OFFSET = MAX_BLOCKS * 4
_RS_ARRIVED = ((_ACTIVE_EPOCH_OFFSET + MAX_BLOCKS * 4 + 255) // 256) * 256
_XS_ARRIVED = _RS_ARRIVED + MAX_BLOCKS * _ISLAND_SIZE * 128
_AG_ARRIVED = _XS_ARRIVED + MAX_BLOCKS * _ISLAND_SIZE * 128
HEADER_BYTES = _AG_ARRIVED + MAX_BLOCKS * _ISLAND_SIZE * 128
# Slice progress occupies spare words in the legacy flag cache lines. Its
# counter advances only for streaming calls, independently of invocation tags.
_STREAM_COUNTER = 4
_STREAM_READY = 16
_STREAM_FINAL = 20
_STREAM_WORDS = 512


def island_rs_peers(rank: int, world_size: int) -> tuple[int, ...]:
    """Slabs this rank maps: three island lanes, three same-lane partners."""

    if world_size != _MAX_WORLD_SIZE:
        raise ValueError(f"island reduce-scatter requires TP16, got TP{world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for TP{world_size}")
    island, lane = divmod(rank, _ISLAND_SIZE)
    lanes = {island * _ISLAND_SIZE + p for p in range(_ISLAND_SIZE)}
    partners = {j * _ISLAND_SIZE + lane for j in range(_MAX_ISLANDS)}
    return tuple(sorted((lanes | partners) - {rank}))


class _IslandRSLaunch:
    """Rank-specialized launcher; every peer index folds away at compile time."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        *,
        threads: int = 128,
        wait_nanosleep_cycles: int = 24,
        cross_island_rsag: bool = False,
        local_push: bool = False,
    ) -> None:
        if world_size != _MAX_WORLD_SIZE:
            raise ValueError(f"unsupported world size {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if not 32 <= threads <= 1024 or threads % 32:
            raise ValueError("threads must be a multiple of 32 in [32, 1024]")
        if type(cross_island_rsag) is not bool:
            raise TypeError("cross_island_rsag must be a boolean")
        if type(local_push) is not bool:
            raise TypeError("local_push must be a boolean")
        self._local_push = local_push
        self._cross_island_rsag = cross_island_rsag
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._wait_nanosleep_cycles = int(wait_nanosleep_cycles)
        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE

    @cute.jit
    def __call__(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        part_offset: Int64,
        final_offset: Int64,
        quarter_capacity: Int64,
        elements: Int64,
        blocks: Int32,
        active_blocks: Int32,
        timing_records: cute.Pointer,
        timing_calls: Int32,
        timing_blocks: Int32,
        timing_slices: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
            input_ptr,
            output_ptr,
            stage_offset,
            part_offset,
            final_offset,
            quarter_capacity,
            elements,
            active_blocks,
            timing_records,
            timing_calls,
            timing_blocks,
            timing_slices,
        ).launch(
            grid=(blocks, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        part_offset: Int64,
        final_offset: Int64,
        quarter_capacity: Int64,
        elements: Int64,
        active_blocks: Int32,
        timing_records: cute.Pointer,
        timing_calls: Int32,
        timing_blocks: Int32,
        timing_slices: Int32,
    ) -> None:
        slabs = (
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        self_base = Int64(slabs[self._rank].toint())

        allocator = cutlass.utils.SmemAllocator()
        shared_generation = allocator.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_layout((3,)),
            byte_alignment=4,
        )
        if tidx == Int32(0):
            shared_generation[0] = _atomic_add_global_u32(
                self_base + Int64(bidx) * Int64(4), Uint32(1)
            ) + Uint32(1)
            epoch = Uint32(0)
            if bidx < active_blocks:
                epoch = _atomic_add_global_u32(
                    self_base + Int64(_ACTIVE_EPOCH_OFFSET) + Int64(bidx) * Int64(4),
                    Uint32(1),
                ) + Uint32(1)
            shared_generation[1] = epoch
            if cutlass.const_expr(
                self._local_push
                and self._cross_island_rsag
                and self._threads % 128 == 0
            ):
                if bidx < active_blocks:
                    quarter = (elements // Int64(2) + Int64(3)) // Int64(4)
                    partition = (quarter + Int64(3)) // Int64(4)
                    span = Int64(active_blocks) * Int64(_STREAM_WORDS)
                    slices = (partition + span - Int64(1)) // span
                    shared_generation[2] = _atomic_add_global_u32(
                        _flag_address(self_base, _RS_ARRIVED, bidx, Int32(0))
                        + Int64(_STREAM_COUNTER),
                        Uint32(slices),
                    )
        cute.arch.barrier()
        data_generation = Uint32(shared_generation[0])
        # Every physical CTA advances buffer parity. Communication epochs advance
        # only for participating CTAs, keeping their last flags within one epoch
        # even when a CTA has been inactive across a global counter rollover.
        if bidx < active_blocks:
            if cutlass.const_expr(
                self._local_push
                and self._cross_island_rsag
                and self._threads % 128 == 0
            ):
                self._stream_participate(
                    slabs,
                    input_ptr,
                    output_ptr,
                    stage_offset,
                    part_offset,
                    final_offset,
                    quarter_capacity,
                    elements,
                    data_generation,
                    Uint32(shared_generation[1]),
                    Uint32(shared_generation[2]),
                    self_base,
                    tidx,
                    bidx,
                    active_blocks,
                    timing_records,
                    timing_calls,
                    timing_blocks,
                    timing_slices,
                )
            else:
                self._participate(
                    slabs,
                    input_ptr,
                    output_ptr,
                    stage_offset,
                    part_offset,
                    final_offset,
                    quarter_capacity,
                    elements,
                    data_generation,
                    Uint32(shared_generation[1]),
                    self_base,
                    tidx,
                    bidx,
                    active_blocks,
                )

    @cute.jit
    def _stream_publish(
        self,
        slabs,
        region: cutlass.Constexpr,
        offset: cutlass.Constexpr,
        cross: cutlass.Constexpr,
        bidx,
        tag,
        group_tid,
        barrier: cutlass.Constexpr,
        timing_records,
        timing_record,
        timing_enabled,
    ):
        _record_group_time(timing_records, timing_record, 3, timing_enabled, group_tid)
        _fence_sc_sys()
        cute.arch.barrier(barrier_id=barrier, number_of_threads=self._threads // 4)
        _record_group_time(timing_records, timing_record, 4, timing_enabled, group_tid)
        if group_tid == Int32(0):
            for peer in cutlass.range_constexpr(4):
                if cutlass.const_expr(cross):
                    destination = peer * 4 + self._lane
                    sender = self._island
                else:
                    destination = self._island * 4 + peer
                    sender = self._lane
                _store_release_sys_u32(
                    _flag_address(
                        Int64(slabs[destination].toint()), region, bidx, Int32(sender)
                    )
                    + Int64(offset),
                    tag,
                )
        _record_group_time(timing_records, timing_record, 5, timing_enabled, group_tid)

    @cute.jit
    def _stream_wait(
        self,
        self_base,
        region: cutlass.Constexpr,
        offset: cutlass.Constexpr,
        bidx,
        tag,
        group_tid,
        barrier: cutlass.Constexpr,
    ):
        if group_tid == Int32(0):
            for peer in cutlass.range_constexpr(4):
                _wait_for(
                    _flag_address(self_base, region, bidx, Int32(peer)) + Int64(offset),
                    tag,
                    self._wait_nanosleep_cycles,
                )
        cute.arch.barrier(barrier_id=barrier, number_of_threads=self._threads // 4)

    @cute.jit
    def _stream_participate(
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
        first_slice,
        self_base,
        tidx,
        bidx,
        gdim,
        timing_records,
        timing_calls,
        timing_blocks,
        timing_slices,
    ):
        # Both transfer widths cover the same CTA slice before publishing it.
        if (
            Int64(input_ptr.toint()) % Int64(16) == Int64(0)
            and Int64(output_ptr.toint()) % Int64(16) == Int64(0)
            and elements % Int64(128) == Int64(0)
        ):
            self._stream_participate_words(
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
                first_slice,
                self_base,
                tidx,
                bidx,
                gdim,
                timing_records,
                timing_calls,
                timing_blocks,
                timing_slices,
                4,
            )
        else:
            self._stream_participate_words(
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
                first_slice,
                self_base,
                tidx,
                bidx,
                gdim,
                timing_records,
                timing_calls,
                timing_blocks,
                timing_slices,
                1,
            )

    @cute.jit
    def _stream_participate_words(
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
        first_slice,
        self_base,
        tidx,
        bidx,
        gdim,
        timing_records,
        timing_calls,
        timing_blocks,
        timing_slices,
        width: cutlass.Constexpr[int],
    ):
        if data_generation % Uint32(2) != Uint32(0):
            stage_offset += quarter_capacity * Int64(16)
            part_offset += quarter_capacity * Int64(4)
            final_offset += quarter_capacity * Int64(16)
        pairs = elements // Int64(2)
        quarter = (pairs + Int64(3)) // Int64(4)
        partition = (quarter + Int64(3)) // Int64(4)
        span = Int64(gdim) * Int64(_STREAM_WORDS)
        slices = (partition + span - Int64(1)) // span
        group_threads = self._threads // 4
        group = tidx // Int32(group_threads)
        group_tid = tidx % Int32(group_threads)
        timing_enabled = timing_calls > Int32(0)
        timing_group = Int64(0)
        if timing_enabled:
            slot = Int64(data_generation % Uint32(timing_calls))
            timing_group = (
                ((slot * Int64(timing_blocks) + Int64(bidx)) * Int64(4) + Int64(group))
                * Int64(timing_slices)
                * Int64(_GROUP_TIMING_WORDS)
            )
        input_words = cute.recast_ptr(input_ptr.align(4), dtype=Uint32)
        output_words = cute.recast_ptr(output_ptr.align(4), dtype=Uint32)
        self_stage = cute.make_ptr(
            Uint32, self_base + stage_offset, cute.AddressSpace.gmem, assumed_align=4
        )
        self_part = cute.make_ptr(
            Uint32, self_base + part_offset, cute.AddressSpace.gmem, assumed_align=4
        )
        self_final = cute.make_ptr(
            Uint32, self_base + final_offset, cute.AddressSpace.gmem, assumed_align=4
        )
        mine_begin = Int64(self._lane) * quarter

        # Every slice owns distinct payload offsets until the invocation retires.
        # Slice flags may advance while a consumer reads an earlier slice.
        if group == Int32(0):
            for step in cutlass.range(slices):
                timing_record = timing_group + Int64(step) * Int64(_GROUP_TIMING_WORDS)
                _start_group_timing(
                    timing_records,
                    timing_record,
                    timing_enabled,
                    data_generation,
                    group_tid,
                )
                _record_group_time(
                    timing_records, timing_record, 2, timing_enabled, group_tid
                )
                tag = first_slice + Uint32(step) + Uint32(1)
                begin = Int64(step) * span + Int64(bidx) * Int64(_STREAM_WORDS)
                for lane in cutlass.range_constexpr(4):
                    if cutlass.const_expr(lane != self._lane):
                        peer = self._island * 4 + lane
                        destination = cute.make_ptr(
                            Uint32,
                            Int64(slabs[peer].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        for part in cutlass.range_constexpr(4):
                            item = Int64(group_tid) * Int64(width)
                            while item < Int64(_STREAM_WORDS):
                                position = begin + item
                                index = Int64(part) * partition + position
                                source = Int64(lane) * quarter + index
                                if (
                                    position < partition
                                    and index < quarter
                                    and source < pairs
                                ):
                                    words = _load_payload_words(
                                        input_words, source, width
                                    )
                                    _store_payload_words(
                                        destination,
                                        Int64(self._lane) * quarter + index,
                                        words,
                                        width,
                                    )
                                item += Int64(group_threads * width)
                self._stream_publish(
                    slabs,
                    _RS_ARRIVED,
                    _STREAM_READY,
                    False,
                    bidx,
                    tag,
                    group_tid,
                    1,
                    timing_records,
                    timing_record,
                    timing_enabled,
                )
                _finish_group_timing(
                    timing_records, timing_record, timing_enabled, group_tid
                )
        elif group == Int32(1):
            for step in cutlass.range(slices):
                timing_record = timing_group + Int64(step) * Int64(_GROUP_TIMING_WORDS)
                _start_group_timing(
                    timing_records,
                    timing_record,
                    timing_enabled,
                    data_generation,
                    group_tid,
                )
                tag = first_slice + Uint32(step) + Uint32(1)
                begin = Int64(step) * span + Int64(bidx) * Int64(_STREAM_WORDS)
                self._stream_wait(
                    self_base, _RS_ARRIVED, _STREAM_READY, bidx, tag, group_tid, 2
                )
                _record_group_time(
                    timing_records, timing_record, 2, timing_enabled, group_tid
                )
                for part in cutlass.range_constexpr(4):
                    item = Int64(group_tid) * Int64(width)
                    while item < Int64(_STREAM_WORDS):
                        position = begin + item
                        index = Int64(part) * partition + position
                        if (
                            position < partition
                            and index < quarter
                            and mine_begin + index < pairs
                        ):
                            totals = ()
                            for item_word in cutlass.range_constexpr(width):
                                totals += (Float32(0.0), Float32(0.0))
                            for lane in cutlass.range_constexpr(4):
                                if cutlass.const_expr(lane == self._lane):
                                    words = _load_payload_words(
                                        input_words, mine_begin + index, width
                                    )
                                else:
                                    words = _load_payload_words(
                                        self_stage,
                                        Int64(lane) * quarter + index,
                                        width,
                                    )
                                totals = _accumulate_payload_words(totals, words, width)
                            _store_payload_words(
                                self_part,
                                index,
                                _pack_payload_words(totals, width),
                                width,
                            )
                        item += Int64(group_threads * width)
                self._stream_publish(
                    slabs,
                    _XS_ARRIVED,
                    _STREAM_READY,
                    True,
                    bidx,
                    tag,
                    group_tid,
                    2,
                    timing_records,
                    timing_record,
                    timing_enabled,
                )
                _finish_group_timing(
                    timing_records, timing_record, timing_enabled, group_tid
                )
        elif group == Int32(2):
            for step in cutlass.range(slices):
                timing_record = timing_group + Int64(step) * Int64(_GROUP_TIMING_WORDS)
                _start_group_timing(
                    timing_records,
                    timing_record,
                    timing_enabled,
                    data_generation,
                    group_tid,
                )
                tag = first_slice + Uint32(step) + Uint32(1)
                begin = Int64(step) * span + Int64(bidx) * Int64(_STREAM_WORDS)
                self._stream_wait(
                    self_base, _XS_ARRIVED, _STREAM_READY, bidx, tag, group_tid, 3
                )
                _record_group_time(
                    timing_records, timing_record, 2, timing_enabled, group_tid
                )
                item = Int64(group_tid) * Int64(width)
                while item < Int64(_STREAM_WORDS):
                    position = begin + item
                    index = Int64(self._island) * partition + position
                    if (
                        position < partition
                        and index < quarter
                        and mine_begin + index < pairs
                    ):
                        totals = ()
                        for item_word in cutlass.range_constexpr(width):
                            totals += (Float32(0.0), Float32(0.0))
                        for island in cutlass.range_constexpr(4):
                            peer = island * 4 + self._lane
                            source = cute.make_ptr(
                                Uint32,
                                Int64(slabs[peer].toint()) + part_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            words = _load_payload_words(
                                source,
                                index,
                                width,
                            )
                            totals = _accumulate_payload_words(totals, words, width)
                        _store_payload_words(
                            self_final,
                            mine_begin + index,
                            _pack_payload_words(totals, width),
                            width,
                        )
                    item += Int64(group_threads * width)
                self._stream_publish(
                    slabs,
                    _XS_ARRIVED,
                    _STREAM_FINAL,
                    True,
                    bidx,
                    tag,
                    group_tid,
                    3,
                    timing_records,
                    timing_record,
                    timing_enabled,
                )
                _finish_group_timing(
                    timing_records, timing_record, timing_enabled, group_tid
                )
        else:
            for step in cutlass.range(slices):
                timing_record = timing_group + Int64(step) * Int64(_GROUP_TIMING_WORDS)
                _start_group_timing(
                    timing_records,
                    timing_record,
                    timing_enabled,
                    data_generation,
                    group_tid,
                )
                tag = first_slice + Uint32(step) + Uint32(1)
                begin = Int64(step) * span + Int64(bidx) * Int64(_STREAM_WORDS)
                self._stream_wait(
                    self_base, _XS_ARRIVED, _STREAM_FINAL, bidx, tag, group_tid, 4
                )
                _record_group_time(
                    timing_records, timing_record, 2, timing_enabled, group_tid
                )
                for part in cutlass.range_constexpr(4):
                    peer = part * 4 + self._lane
                    source = cute.make_ptr(
                        Uint32,
                        Int64(slabs[peer].toint()) + final_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    item = Int64(group_tid) * Int64(width)
                    while item < Int64(_STREAM_WORDS):
                        position = begin + item
                        index = Int64(part) * partition + position
                        if (
                            position < partition
                            and index < quarter
                            and mine_begin + index < pairs
                        ):
                            words = _load_payload_words(
                                source, mine_begin + index, width
                            )
                            _store_payload_words(
                                output_words, mine_begin + index, words, width
                            )
                            for lane in cutlass.range_constexpr(4):
                                if cutlass.const_expr(lane != self._lane):
                                    peer = self._island * 4 + lane
                                    destination = cute.make_ptr(
                                        Uint32,
                                        Int64(slabs[peer].toint()) + final_offset,
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    _store_payload_words(
                                        destination, mine_begin + index, words, width
                                    )
                        item += Int64(group_threads * width)
                self._stream_publish(
                    slabs,
                    _AG_ARRIVED,
                    _STREAM_READY,
                    False,
                    bidx,
                    tag,
                    group_tid,
                    4,
                    timing_records,
                    timing_record,
                    timing_enabled,
                )
                _record_group_time(
                    timing_records, timing_record, 6, timing_enabled, group_tid
                )
                self._stream_wait(
                    self_base, _AG_ARRIVED, _STREAM_READY, bidx, tag, group_tid, 4
                )
                _record_group_time(
                    timing_records, timing_record, 7, timing_enabled, group_tid
                )
                for lane in cutlass.range_constexpr(4):
                    if cutlass.const_expr(lane != self._lane):
                        for part in cutlass.range_constexpr(4):
                            item = Int64(group_tid) * Int64(width)
                            while item < Int64(_STREAM_WORDS):
                                position = begin + item
                                index = Int64(part) * partition + position
                                target = Int64(lane) * quarter + index
                                if (
                                    position < partition
                                    and index < quarter
                                    and target < pairs
                                ):
                                    _store_payload_words(
                                        output_words,
                                        target,
                                        _load_payload_words(self_final, target, width),
                                        width,
                                    )
                                item += Int64(group_threads * width)
                _finish_group_timing(
                    timing_records, timing_record, timing_enabled, group_tid
                )
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for peer in cutlass.range_constexpr(4):
                if cutlass.const_expr(peer != self._lane):
                    base = Int64(slabs[self._island * 4 + peer].toint())
                    _store_release_sys_u32(
                        _flag_address(base, _RS_ARRIVED, bidx, Int32(self._lane)),
                        generation,
                    )
                    _store_release_sys_u32(
                        _flag_address(base, _AG_ARRIVED, bidx, Int32(self._lane)),
                        generation,
                    )
                if cutlass.const_expr(peer != self._island):
                    base = Int64(slabs[peer * 4 + self._lane].toint())
                    _store_release_sys_u32(
                        _flag_address(base, _XS_ARRIVED, bidx, Int32(self._island)),
                        generation * Uint32(2) + Uint32(1),
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
            if cutlass.const_expr(self._local_push):
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
                # In the non-RSAG push specialization, the owned contribution
                # is read directly from the input during reduction instead of
                # staging it locally and reading it back.
                if cutlass.const_expr(self._cross_island_rsag or lane != self._lane):
                    while index < stop - begin:
                        destination_stage[Int64(self._lane) * quarter + index] = (
                            input_words[begin + index]
                        )
                        index += stride
            else:
                while index < stop - begin:
                    self_stage[begin + index] = input_words[begin + index]
                    index += stride
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    peer = self._island * _ISLAND_SIZE + p
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[peer].toint()),
                            _RS_ARRIVED,
                            bidx,
                            Int32(self._lane),
                        ),
                        generation,
                    )
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    _wait_for(
                        _flag_address(self_base, _RS_ARRIVED, bidx, Int32(p)),
                        generation,
                        self._wait_nanosleep_cycles,
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
                if cutlass.const_expr(self._local_push):
                    if cutlass.const_expr(
                        p == self._lane and not self._cross_island_rsag
                    ):
                        peer_stage = input_words
                        peer_begin = mine_begin
                    else:
                        peer_stage = self_stage
                        peer_begin = Int64(p) * quarter
                else:
                    peer_begin = mine_begin
                    peer_stage = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
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
                if cutlass.const_expr(self._local_push):
                    if cutlass.const_expr(
                        p == self._lane and not self._cross_island_rsag
                    ):
                        peer_stage = input_words
                        peer_begin = mine_begin
                    else:
                        peer_stage = self_stage
                        peer_begin = Int64(p) * quarter
                else:
                    peer_begin = mine_begin
                    peer_stage = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                lo, hi = unpack_bf16x2(peer_stage[peer_begin + index])
                total_lo += lo
                total_hi += hi
            self_part[index] = pack_f32x2_to_bf16x2(total_lo, total_hi)
            index += stride
        cute.arch.barrier()

        # The XS mailbox publishes partial-ready (2*g), then final-ready (2*g+1)
        # for scattered stripes. A later tag also certifies the still-live partials.
        # Both modes sum island partials in the same order before BF16 conversion.
        if tidx == Int32(0):
            _fence_sc_sys()
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    partner = j * _ISLAND_SIZE + self._lane
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[partner].toint()),
                            _XS_ARRIVED,
                            bidx,
                            Int32(self._island),
                        ),
                        generation * Uint32(2),
                    )
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    _wait_for(
                        _flag_address(self_base, _XS_ARRIVED, bidx, Int32(j)),
                        generation * Uint32(2),
                        self._wait_nanosleep_cycles,
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
        if cutlass.const_expr(self._cross_island_rsag):
            index += Int64(self._island) * stride
            cross_stride *= Int64(_MAX_ISLANDS)
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
                if cutlass.const_expr(not self._local_push or self._cross_island_rsag):
                    self_final[mine_begin + index + Int64(item) * cross_stride] = value
                output_words[mine_begin + index + Int64(item) * cross_stride] = value
                if cutlass.const_expr(self._local_push and not self._cross_island_rsag):
                    for destination_lane in cutlass.range_constexpr(_ISLAND_SIZE):
                        if cutlass.const_expr(destination_lane != self._lane):
                            destination_final = cute.recast_ptr(
                                cute.make_ptr(
                                    cutlass.BFloat16,
                                    Int64(
                                        slabs[
                                            self._island * _ISLAND_SIZE
                                            + destination_lane
                                        ].toint()
                                    )
                                    + final_offset,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ).align(4),
                                dtype=Uint32,
                            )
                            destination_final[
                                mine_begin + index + Int64(item) * cross_stride
                            ] = value
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
            if cutlass.const_expr(not self._local_push or self._cross_island_rsag):
                self_final[mine_begin + index] = value
            output_words[mine_begin + index] = value
            if cutlass.const_expr(self._local_push and not self._cross_island_rsag):
                for destination_lane in cutlass.range_constexpr(_ISLAND_SIZE):
                    if cutlass.const_expr(destination_lane != self._lane):
                        destination_final = cute.recast_ptr(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                Int64(
                                    slabs[
                                        self._island * _ISLAND_SIZE + destination_lane
                                    ].toint()
                                )
                                + final_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ).align(4),
                            dtype=Uint32,
                        )
                        destination_final[mine_begin + index] = value
            index += cross_stride
        cute.arch.barrier()

        if cutlass.const_expr(self._cross_island_rsag):
            if tidx == Int32(0):
                _fence_sc_sys()
                for j in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(j != self._island):
                        partner = j * _ISLAND_SIZE + self._lane
                        _store_release_sys_u32(
                            _flag_address(
                                Int64(slabs[partner].toint()),
                                _XS_ARRIVED,
                                bidx,
                                Int32(self._island),
                            ),
                            generation * Uint32(2) + Uint32(1),
                        )
                for j in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(j != self._island):
                        _wait_for(
                            _flag_address(self_base, _XS_ARRIVED, bidx, Int32(j)),
                            generation * Uint32(2) + Uint32(1),
                            self._wait_nanosleep_cycles,
                        )
            cute.arch.barrier()
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    partner = j * _ISLAND_SIZE + self._lane
                    partner_final = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[partner].toint()) + final_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                    index = start + Int64(j) * stride
                    while index < mine_length:
                        value = partner_final[mine_begin + index]
                        self_final[mine_begin + index] = value
                        output_words[mine_begin + index] = value
                        index += cross_stride
            cute.arch.barrier()

        # RSAG publishes completed quarters in a separate pass. The non-RSAG
        # push specialization above already forwarded each rounded value to all
        # three local peers directly from registers.
        if cutlass.const_expr(self._local_push and self._cross_island_rsag):
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
        if tidx == Int32(0):
            _fence_sc_sys()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    peer = self._island * _ISLAND_SIZE + p
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[peer].toint()),
                            _AG_ARRIVED,
                            bidx,
                            Int32(self._lane),
                        ),
                        generation,
                    )
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    _wait_for(
                        _flag_address(self_base, _AG_ARRIVED, bidx, Int32(p)),
                        generation,
                        self._wait_nanosleep_cycles,
                    )
        cute.arch.barrier()

        for p in cutlass.range_constexpr(_ISLAND_SIZE):
            if cutlass.const_expr(p != self._lane):
                peer = self._island * _ISLAND_SIZE + p
                if cutlass.const_expr(self._local_push):
                    peer_final = self_final
                else:
                    peer_final = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer].toint()) + final_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                begin = Int64(p) * quarter
                stop = begin + quarter
                if stop > pairs:
                    stop = pairs
                index = start
                while index < stop - begin:
                    output_words[begin + index] = peer_final[begin + index]
                    index += stride


@program_cache
def get_island_rs_launcher(
    world_size: int,
    rank: int,
    device_index: int,
    *,
    threads: int = 128,
    wait_nanosleep_cycles: int = 24,
    cross_island_rsag: bool = False,
    local_push: bool = False,
) -> Callable[..., None]:
    """Return the rank-specialized TP16 island reduce-scatter launcher."""

    del device_index  # retained in the process-local cache key
    launch = _IslandRSLaunch(
        world_size,
        rank,
        threads=threads,
        wait_nanosleep_cycles=wait_nanosleep_cycles,
        cross_island_rsag=cross_island_rsag,
        local_push=local_push,
    )
    cache_key = (
        int(world_size),
        int(rank),
        int(threads),
        int(wait_nanosleep_cycles),
        bool(cross_island_rsag),
        bool(local_push),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
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
        make_ptr(cutlass.Int64, 256, cute.AddressSpace.gmem, assumed_align=8),
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island_rs.bf16",
            3,
            cache_key,
            labels=(
                "world_size",
                "rank",
                "threads",
                "wait_nanosleep_cycles",
                "cross_island_rsag",
                "local_push",
            ),
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
        timing_address: int = 0,
        timing_calls: int = 0,
        timing_blocks: int = 0,
        timing_slices: int = 0,
    ) -> None:
        """Launch the compiled island reduce-scatter with runtime arguments."""
        if len(slab_addresses) != world_size:
            raise ValueError(
                f"expected {world_size} slab addresses, got {len(slab_addresses)}"
            )
        if not 1 <= int(blocks) <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
        active_blocks = blocks if active_blocks is None else active_blocks
        if not 1 <= int(active_blocks) <= int(blocks):
            raise ValueError("active blocks must fit the fixed physical grid")
        if timing_calls:
            if not local_push or not cross_island_rsag or threads % 128:
                raise ValueError("group timing requires streaming local-push RSAG")
            partition = (int(elements) + 31) // 32
            required_slices = (partition + int(active_blocks) * _STREAM_WORDS - 1) // (
                int(active_blocks) * _STREAM_WORDS
            )
            if (
                timing_address <= 0
                or timing_address % 8
                or timing_calls < 0
                or timing_blocks < active_blocks
                or timing_slices < required_slices
            ):
                raise ValueError("group timing storage does not cover the launch")
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
            make_ptr(
                cutlass.Int64, timing_address, cute.AddressSpace.gmem, assumed_align=8
            ),
            timing_calls,
            timing_blocks,
            timing_slices,
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


__all__ = ["HEADER_BYTES", "MAX_BLOCKS", "get_island_rs_launcher", "island_rs_peers"]
