"""Neighbor-only byte gather using the DCP channel's banks and paired flags."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _MAX_BLOCKS,
    _a2a_graph_epoch_arrive,
)
from ._dcp_cute_common import (
    _FLAG_STRIDE,
    _MAX_RANKS,
    _load_relaxed_sys_u32,
    _membar_sys,
    _store_relaxed_sys_u32,
)
from ._joint_cluster_cute import _threadfence_sys
from ._runtime_gather_cute import SelectedGather


class RingGather(SelectedGather):
    def __init__(self, world, rank, row_words, threads, blocks, chunks):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        if world not in (8, 16) or threads not in (128, 512) or blocks != 8:
            raise ValueError(
                "ring gather requires DCP8 or DCP16 with eight 128- or 512-thread CTAs"
            )
        self.next = (rank + 1) % world
        self.previous = (rank - 1) % world

    @cute.jit
    def _flag(self, signal, sender, block, value):
        slot = Int64(value % Uint32(2))
        index = (
            Int64(_MAX_BLOCKS * _MAX_RANKS)
            + (
                (slot * Int64(_MAX_BLOCKS) + Int64(block))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            + Int64(sender * _FLAG_STRIDE)
        )
        return signal + index

    @cute.kernel
    def selected(
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
            remote = self._staging_words(stages[self.next] + bank)
            first = Int32(block) * Int32(self._threads) + Int32(tid)
            step = Int32(8 * self._threads)
            packs = Int32(self.row_words // 4)
            total = rows * packs
            destination = self._rank if self._rank < self.next else self._rank - 1
            index = first
            while index < total:
                row = index // packs
                pack = index - row * packs
                out_pack = (
                    Int64(row) * Int64(self._world_size) + Int64(self._rank)
                ) * Int64(packs) + Int64(pack)
                destination_pack = (
                    Int64(row) * Int64(self._world_size - 1) + Int64(destination)
                ) * Int64(packs) + Int64(pack)
                words = ld_global_v4_u32((source + Int64(index) * Int64(4)).toint())
                st_global_v4_u32((output + out_pack * Int64(4)).toint(), *words)
                st_global_v4_u32((remote + destination_pack * Int64(4)).toint(), *words)
                index += step
            incoming, outgoing = Uint32(0), Uint32(0)
            for phase in cutlass.range_constexpr(self._world_size - 1):
                receiving = (self._rank - phase - 1) % self._world_size
                origin = receiving if receiving < self._rank else receiving - 1
                if cutlass.const_expr(phase == 0):
                    # Source stores have no preceding receive-side release.
                    _threadfence_sys()
                    cute.arch.sync_threads()
                if Int32(tid) == Int32(0):
                    own = signals[self._rank] + Int64(block) * Int64(16)
                    outgoing = Uint32(
                        cute.arch.load(own + Int64(self.next), Uint32)
                    ) + Uint32(1)
                    incoming = Uint32(
                        cute.arch.load(own + Int64(self.previous), Uint32)
                    ) + Uint32(1)
                    if cutlass.const_expr(phase >= 2):
                        # Reusing a parity slot requires its earlier notification
                        # to have been consumed by the downstream CTA.
                        retired = outgoing - Uint32(2)
                        acknowledged = self._flag(
                            signals[self._rank], self.next, block, retired
                        )
                        observed = _load_relaxed_sys_u32(acknowledged)
                        while observed != retired:
                            observed = _load_relaxed_sys_u32(acknowledged)
                        _membar_sys()
                    cute.arch.store(own + Int64(self.next), outgoing)
                    cute.arch.store(own + Int64(self.previous), incoming)
                    _store_relaxed_sys_u32(
                        self._flag(signals[self.next], self._rank, block, outgoing),
                        outgoing,
                    )
                    ready = self._flag(
                        signals[self._rank], self.previous, block, incoming
                    )
                    observed = _load_relaxed_sys_u32(ready)
                    while observed != incoming:
                        observed = _load_relaxed_sys_u32(ready)
                    _membar_sys()
                cute.arch.sync_threads()
                index = first
                while index < total:
                    row = index // packs
                    pack = index - row * packs
                    received_pack = (
                        Int64(row) * Int64(self._world_size - 1) + Int64(origin)
                    ) * Int64(packs) + Int64(pack)
                    out_pack = (
                        Int64(row) * Int64(self._world_size) + Int64(receiving)
                    ) * Int64(packs) + Int64(pack)
                    words = ld_global_v4_u32((local + received_pack * Int64(4)).toint())
                    st_global_v4_u32((output + out_pack * Int64(4)).toint(), *words)
                    if cutlass.const_expr(phase < self._world_size - 2):
                        destination = (
                            receiving if receiving < self.next else receiving - 1
                        )
                        destination_pack = (
                            Int64(row) * Int64(self._world_size - 1)
                            + Int64(destination)
                        ) * Int64(packs) + Int64(pack)
                        st_global_v4_u32(
                            (remote + destination_pack * Int64(4)).toint(), *words
                        )
                    index += step
                # Distinct origin slots retain forwarding bytes while two ready
                # notifications overlap. ACK permits matching parity slot reuse.
                _threadfence_sys()
                cute.arch.sync_threads()
                if Int32(tid) == Int32(0):
                    _store_relaxed_sys_u32(
                        self._flag(signals[self.previous], self._rank, block, incoming),
                        incoming,
                    )
                cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                # Ordered receives make the final ACK drain every notification.
                acknowledged = self._flag(
                    signals[self._rank], self.next, block, outgoing
                )
                observed = _load_relaxed_sys_u32(acknowledged)
                while observed != outgoing:
                    observed = _load_relaxed_sys_u32(acknowledged)
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )
