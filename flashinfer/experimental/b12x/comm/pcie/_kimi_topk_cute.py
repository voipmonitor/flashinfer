# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi top-16 selection using per-warp candidates and one block synchronization.

The score arithmetic and integer ordering are the b12x Kimi selector's contract.
Every global top-16 key belongs to the local top-16 of its warp, so merging those
sorted lists preserves exact selection and the final normalization order.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint64

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from ._dcp_a2a_cute import (
    _f32_ptr,
    _i32_ptr,
    _kimi_key_expert,
    _kimi_pack_key,
    _kimi_warp_max_key,
    raise_if_kernel_resolution_frozen,
)


class WarpTopK16Launch:
    @cute.jit
    def __call__(
        self,
        router_logits: cute.Pointer,
        correction_bias: cute.Pointer,
        output_weights: cute.Pointer,
        output_ids: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(router_logits, correction_bias, output_weights, output_ids).launch(
            grid=(rows, 1, 1),
            block=(256, 1, 1),
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        router_logits: cute.Pointer,
        correction_bias: cute.Pointer,
        output_weights: cute.Pointer,
        output_ids: cute.Pointer,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) % Int32(32)
        warp = Int32(tidx) // Int32(32)
        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            unbiased_scores: cute.struct.Align[cute.struct.MemRange[Float32, 896], 16]
            warp_keys: cute.struct.Align[cute.struct.MemRange[Uint64, 128], 16]

        storage = smem_alloc.allocate(SharedStorage)
        unbiased_scores = storage.unbiased_scores.get_tensor(
            cute.make_layout((896,), stride=(1,))
        )
        warp_keys = storage.warp_keys.get_tensor(cute.make_layout((128,), stride=(1,)))
        keys = cute.make_rmem_tensor((4,), Uint64)
        row_offset = Int64(bidx) * Int64(896)
        for item in cutlass.range_constexpr(4):
            expert = Int32(tidx) + Int32(item * 256)
            key = Uint64(0)
            if expert < Int32(896):
                router_value = cute.arch.load(
                    router_logits + row_offset + Int64(expert), Float32
                )
                bias = cute.arch.load(correction_bias + Int64(expert), Float32)
                unbiased = Float32(1.0) / (
                    Float32(1.0) + cute.math.exp(-router_value, fastmath=True)
                )
                if not cute.math.isfinite(unbiased):
                    unbiased = Float32(0.0)
                selection = unbiased + bias
                if not cute.math.isfinite(selection):
                    if selection > Float32(0.0):
                        selection = Float32(float("inf"))
                    else:
                        selection = Float32(float("-inf"))
                if selection == Float32(0.0):
                    selection = Float32(0.0)
                key = _kimi_pack_key(selection, expert)
                unbiased_scores[expert] = unbiased
            keys[item] = key

        # Packed keys are unique because they include the expert ID. Each warp
        # therefore removes exactly one candidate per selection without a CTA
        # barrier or an active-expert table in shared memory.
        for selected in cutlass.range_constexpr(16):
            # Score keys occupy the unsigned high half; load register elements
            # explicitly in that domain before reducing them.
            thread_key = Uint64(keys[0])
            for item in cutlass.range_constexpr(1, 4):
                thread_key = cutlass.max(thread_key, Uint64(keys[item]))
            winner = _kimi_warp_max_key(thread_key)
            if lane == Int32(0):
                warp_keys[warp * Int32(16) + Int32(selected)] = winner
            for item in cutlass.range_constexpr(4):
                if keys[item] == winner:
                    keys[item] = Uint64(0)
        cute.arch.sync_threads()

        if warp == Int32(0):
            cursor = Int32(0)
            head = Uint64(0)
            if lane < Int32(8):
                head = warp_keys[lane * Int32(16)]
            selected_ids = cute.make_rmem_tensor((16,), Int32)
            selected_weights = cute.make_rmem_tensor((16,), Float32)
            for selected in cutlass.range_constexpr(16):
                winner = _kimi_warp_max_key(head)
                expert = _kimi_key_expert(winner)
                if lane == Int32(0):
                    selected_ids[selected] = expert
                    selected_weights[selected] = unbiased_scores[expert]
                owner_warp = (expert % Int32(256)) // Int32(32)
                if lane == owner_warp:
                    cursor += Int32(1)
                    head = Uint64(0)
                    if cursor < Int32(16):
                        head = warp_keys[lane * Int32(16) + cursor]
            if lane == Int32(0):
                weight_sum = Float32(0.0)
                for selected in cutlass.range_constexpr(16):
                    weight_sum += selected_weights[selected]
                denominator = Float32(1.0)
                if weight_sum > Float32(0.0):
                    denominator = weight_sum
                scale = Float32(1.0) / denominator
                output_offset = Int64(bidx) * Int64(16)
                for selected in cutlass.range_constexpr(16):
                    cute.arch.store(
                        output_weights + output_offset + Int64(selected),
                        selected_weights[selected] * scale,
                    )
                    cute.arch.store(
                        output_ids + output_offset + Int64(selected),
                        selected_ids[selected],
                    )


@program_cache
def compile_kimi_topk16():
    launch = WarpTopK16Launch()
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=(256,))
    raw = compile_kernel(
        launch,
        _f32_ptr(16),
        _f32_ptr(16),
        _f32_ptr(16),
        _i32_ptr(16),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.kimi_topk16.warp_merge", 1, (256,), labels=("threads",)
        ),
    )

    def run(logits, bias, weights, ids, rows):
        raw(
            _f32_ptr(logits),
            _f32_ptr(bias),
            _f32_ptr(weights),
            _i32_ptr(ids),
            rows,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
