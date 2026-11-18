"""Build an expert-presence mask once per compressed-scale invocation.

One warp scans the route vector for one expert. The bounded output is fully
overwritten on every call, including empty and invalid-only route vectors.
Scale decoders can share the result across all scale tiles without repeating
duplicate searches or expanding experts absent from the routed batch.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache, register_program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

_LAUNCH_PROGRAMS = {}
register_program_cache(_LAUNCH_PROGRAMS)


@cute.jit
def mapped_expert_id(raw: Int64, mapping, route_experts: Int32, experts: Int32):
    expert = Int64(-1)
    if raw >= Int64(0) and raw < route_experts.to(Int64):
        local = mapping[raw.to(Int32)].to(Int64)
        if local >= Int64(0) and local < experts.to(Int64):
            expert = local
    return expert


class _ActiveExperts:
    def __init__(self, mapped=False):
        self.mapped = mapped

    @cute.jit
    def __call__(
        self,
        routes: cute.Pointer,
        mask: cute.Pointer,
        map_ptr: cute.Pointer,
        route_experts: Int32,
        count: Int32,
        experts: Int32,
        stream: cuda.CUstream,
    ):
        ids = cute.make_tensor(routes, cute.make_layout((count,)))
        active = cute.make_tensor(mask, cute.make_layout((experts,)))
        mapping = cute.make_tensor(map_ptr, cute.make_layout((route_experts,)))
        self.kernel(ids, active, mapping, route_experts, count, experts).launch(
            grid=((experts + Int32(7)) // Int32(8), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, ids, active, mapping, route_experts: Int32, count: Int32, experts: Int32
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        expert = Int32(block) * Int32(8) + Int32(tid) // Int32(32)
        lane = Int32(tid) % Int32(32)
        if expert < experts:
            found = cutlass.Boolean(False)
            route = lane
            while route < count:
                raw = ids[route].to(Int64)
                if cutlass.const_expr(self.mapped):
                    raw = mapped_expert_id(raw, mapping, route_experts, experts)
                found = found | (raw == expert.to(Int64))
                route += Int32(32)
            present = cute.arch.vote_any_sync(found)
            if lane == Int32(0):
                active[expert] = present.to(Int32)


class _ActiveExpertBitmap:
    """Build per-warp bitmaps with work proportional to routes plus experts."""

    def __init__(self, experts, mapped=False):
        self.words = (experts + 31) // 32
        self.mapped = mapped

    @cute.jit
    def __call__(
        self,
        routes: cute.Pointer,
        mask: cute.Pointer,
        map_ptr: cute.Pointer,
        route_experts: Int32,
        count: Int32,
        experts: Int32,
        stream: cuda.CUstream,
    ):
        ids = cute.make_tensor(routes, cute.make_layout((count,)))
        active = cute.make_tensor(mask, cute.make_layout((experts,)))
        mapping = cute.make_tensor(map_ptr, cute.make_layout((route_experts,)))
        self.kernel(ids, active, mapping, route_experts, count, experts).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self, ids, active, mapping, route_experts: Int32, count: Int32, experts: Int32
    ):
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        bitmap = smem.allocate_tensor(
            cutlass.Uint32, cute.make_layout(8 * self.words), byte_alignment=16
        )
        index = Int32(tid)
        while index < Int32(8 * self.words):
            bitmap[index] = Uint32(0)
            index += Int32(256)
        cute.arch.sync_threads()
        route = Int32(tid)
        while route < count:
            expert = ids[route].to(Int64)
            if cutlass.const_expr(self.mapped):
                expert = mapped_expert_id(expert, mapping, route_experts, experts)
            if expert >= Int64(0) and expert < experts.to(Int64):
                word = Int32(tid) // Int32(32) * Int32(self.words) + expert.to(
                    Int32
                ) // Int32(32)
                bit = Uint32(1) << (expert.to(Uint32) % Uint32(32))
                cute.arch.atomic_or(
                    bitmap.iterator + word, bit, sem="relaxed", scope="cta"
                )
            route += Int32(256)
        cute.arch.sync_threads()
        expert = Int32(tid)
        while expert < experts:
            bits = Uint32(0)
            for warp in cutlass.range_constexpr(8):
                bits |= bitmap[Int32(warp * self.words) + expert // Int32(32)]
            active[expert] = ((bits >> (expert % Int32(32)).to(Uint32)) & Uint32(1)).to(
                Int32
            )
            expert += Int32(256)


@program_cache
def compile_csf_active_experts(ids64=False, experts=0, mapped=False):
    kernel = _ActiveExpertBitmap(experts, mapped) if experts else _ActiveExperts(mapped)
    key = (bool(ids64), int(experts), bool(mapped))
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    return b12x_compile(
        kernel,
        make_ptr(
            cutlass.Int64 if ids64 else cutlass.Int32,
            8 if ids64 else 4,
            cute.AddressSpace.gmem,
            assumed_align=8 if ids64 else 4,
        ),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        0,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("quant.csf_active_experts", 3, key),
    )


def mark_active_experts(ids, active, program, expert_map=None):
    if (
        ids.dtype not in (torch.int32, torch.int64)
        or not ids.is_contiguous()
        or ids.device != active.device
        or active.dtype != torch.int32
    ):
        raise ValueError("CSF presence masks require contiguous CUDA integer routes")
    if expert_map is not None and (
        expert_map.dtype != torch.int32
        or expert_map.device != ids.device
        or expert_map.ndim != 1
        or not expert_map.is_contiguous()
    ):
        raise ValueError("CSF expert maps must be contiguous CUDA int32 vectors")
    handle = id(program)
    _LAUNCH_PROGRAMS[handle] = program
    _mark_active_experts_op(ids, active, expert_map, handle)


@torch.library.custom_op("b12x::csf_active_experts", mutates_args=("active",))
def _mark_active_experts_op(
    ids: torch.Tensor,
    active: torch.Tensor,
    expert_map: torch.Tensor | None,
    program_handle: int,
) -> None:
    program = _LAUNCH_PROGRAMS[program_handle]
    program(
        make_ptr(
            cutlass.Int64 if ids.dtype == torch.int64 else cutlass.Int32,
            ids.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=8 if ids.dtype == torch.int64 else 4,
        ),
        make_ptr(
            cutlass.Int32, active.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ),
        make_ptr(
            cutlass.Int32,
            (active if expert_map is None else expert_map).data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        0 if expert_map is None else expert_map.numel(),
        ids.numel(),
        active.numel(),
        current_cuda_stream(),
    )


@_mark_active_experts_op.register_fake
def _mark_active_experts_fake(ids, active, expert_map, program_handle) -> None:
    pass
