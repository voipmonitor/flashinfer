"""Bounded expert-route metadata produced beside CSF scale expansion."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64


class CsfRoutePack:
    def __init__(self, geometry):
        self.capacity, self.block_m, self.slots, self.blocks = geometry
        self.rounded = 1 << (self.capacity - 1).bit_length()
        self.levels = (self.rounded - 1).bit_length()
        exchanges = sum(max(0, level - 5) for level in range(1, self.levels + 1))
        self.key_offset = (exchanges % 2) * self.rounded
        self.scan_offset = (1 - exchanges % 2) * self.rounded

    @cute.jit
    def run(
        self,
        ids,
        routes,
        experts,
        count,
        live: Int32,
        num_experts: Int32,
        tid: Int32,
        threads: cutlass.Constexpr,
        shared=None,
    ):
        if cutlass.const_expr(shared is None):
            smem = cutlass.utils.SmemAllocator()
            shared = smem.allocate_tensor(
                Int32, cute.make_layout(2 * self.rounded), byte_alignment=16
            )
        else:
            shared = cute.recast_tensor(shared, Int32)
        slot = tid
        while slot < Int32(self.slots):
            routes[slot] = live
            slot += Int32(threads)
        slot = tid
        while slot < Int32(self.blocks):
            experts[slot] = Int32(-1)
            slot += Int32(threads)
        if tid == Int32(0):
            count[Int32(0)] = Int32(0)

        expert = num_experts
        if tid < live:
            raw = ids[tid].to(Int64)
            if raw >= Int64(0) and raw < num_experts.to(Int64):
                expert = raw.to(Int32)
        key = expert * Int32(self.rounded) + tid
        for level in cutlass.range_constexpr(1, self.levels + 1):
            for step in cutlass.range_constexpr(level):
                distance = 1 << (level - step - 1)
                other = key
                if cutlass.const_expr(distance < 32):
                    other = cute.arch.shuffle_sync_bfly(key, offset=distance)
                else:
                    # Alternate halves so writes preserve the preceding exchange.
                    exchange = (level - 6) * (level - 5) // 2 + step
                    offset = (exchange % 2) * self.rounded
                    if tid < Int32(self.rounded):
                        shared[Int32(offset) + tid] = key
                    cute.arch.sync_threads()
                    if tid < Int32(self.rounded):
                        other = shared[Int32(offset) + (tid ^ Int32(distance))]
                take_min = ((tid & Int32(1 << level)) == Int32(0)) == (
                    (tid & Int32(distance)) == Int32(0)
                )
                if take_min:
                    if other < key:
                        key = other
                else:
                    if other > key:
                        key = other
        if tid < Int32(self.rounded):
            shared[Int32(self.key_offset) + tid] = key
        cute.arch.sync_threads()
        expert = key // Int32(self.rounded)
        first = tid == Int32(0)
        if tid > Int32(0) and tid < Int32(self.rounded):
            first = expert != shared[Int32(self.key_offset) + tid - Int32(1)] // Int32(
                self.rounded
            )
        start = Int32(0)
        if first:
            start = tid
        lane = tid % Int32(32)
        warp = tid // Int32(32)
        for step in cutlass.range_constexpr(5):
            prior = cute.arch.shuffle_sync_up(start, offset=1 << step)
            if lane >= Int32(1 << step) and prior > start:
                start = prior
        if tid < Int32(self.rounded):
            shared[Int32(self.scan_offset) + tid] = start
        cute.arch.sync_threads()
        carry = Int32(0)
        for preceding in cutlass.range_constexpr(self.rounded // 32):
            if warp > Int32(preceding):
                prior = shared[Int32(self.scan_offset + (preceding + 1) * 32 - 1)]
                if prior > carry:
                    carry = prior
        if carry > start:
            start = carry
        previous_start = cute.arch.shuffle_sync_up(start, offset=1)
        if lane == Int32(0):
            previous_start = carry
        padding = Int32(0)
        if first and tid > Int32(0) and tid < Int32(self.rounded):
            padding = (
                Int32(self.block_m) - (tid - previous_start) % Int32(self.block_m)
            ) % Int32(self.block_m)
        for step in cutlass.range_constexpr(5):
            prior = cute.arch.shuffle_sync_up(padding, offset=1 << step)
            if lane >= Int32(1 << step):
                padding += prior
        cute.arch.sync_threads()
        if tid < Int32(self.rounded):
            shared[Int32(self.scan_offset) + tid] = padding
        cute.arch.sync_threads()
        for preceding in cutlass.range_constexpr(self.rounded // 32):
            if warp > Int32(preceding):
                padding += shared[Int32(self.scan_offset + (preceding + 1) * 32 - 1)]
        slot = tid + padding
        if tid < Int32(self.rounded) and expert < num_experts:
            routes[slot] = key % Int32(self.rounded)
            if (tid - start) % Int32(self.block_m) == Int32(0):
                experts[slot // Int32(self.block_m)] = expert
            last = tid == Int32(self.rounded - 1)
            if tid < Int32(self.rounded - 1):
                last = (
                    shared[Int32(self.key_offset) + tid + Int32(1)]
                    // Int32(self.rounded)
                    == num_experts
                )
            if last:
                count[Int32(0)] = (slot // Int32(self.block_m) + Int32(1)) * Int32(
                    self.block_m
                )
