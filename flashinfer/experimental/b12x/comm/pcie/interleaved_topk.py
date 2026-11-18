# SPDX-License-Identifier: Apache-2.0
"""Select only a context shard's winners of a stable, rank-major FP32 top-2048."""

from dataclasses import dataclass

import torch

from b12x.preparation import MemoryRequirements, PersistentMemory, Plan, make_fixed_contract

from .pcie_dcp_a2a import _SIGNAL_BYTES


@dataclass(frozen=True, kw_only=True)
class InterleavedTopKQuery:
    rows: int
    world: int
    rank: int
    cache_pages: int


_CONTRACT = make_fixed_contract(
    component_id="comm.pcie.interleaved_topk_slots", query_type=InterleavedTopKQuery, backend="cuda"
)


def supports_channel(runtime):
    """Use flag-line padding only in an owned, zero-initialized sixteen-rank channel."""
    return (
        runtime is not None
        and runtime.world_size == 16
        and runtime.max_batch_size >= 4
        and getattr(runtime, "_signal_bytes", 0) >= _SIGNAL_BYTES
        and runtime._slot_bytes >= 4 * (2048 + 64) * 8
    )


@dataclass
class _State:
    run_kernel: object
    runtime: object
    cache_pages: int

    def run(self, ids, scores, positions, tables, owners, slots, counts):
        runtime = self.runtime
        runtime._check_stream()
        if runtime._closed:
            raise RuntimeError("interleaved top-k channel is closed")
        if ids.device != runtime.device:
            raise ValueError("interleaved top-k inputs must use the prepared IPC device")
        if not runtime._device_slot_selection:
            runtime._graph_base_slot = runtime._next_slot & 1
            runtime._device_slot_selection = True
        bank = runtime._graph_base_slot
        self.run_kernel(
            ids, scores, positions, tables, owners, slots, counts,
            runtime._staging_ptrs[bank], runtime._signal_ptrs,
            runtime._slot_bytes if bank == 0 else -runtime._slot_bytes,
            runtime.rank, self.cache_pages,
        )


def plan(runtime, *, cache_pages):
    """Prepare the fixed CUDA kernel for four interleaved rows and 128-token pages.

    Ranks must provide identical global positions and issue channel operations
    in the same order. Each local candidate list has a visible prefix of at most
    2048 records followed by canonical padding. The operation mutates only its
    three output tensors and the existing IPC banks and signal lines.
    """
    if not supports_channel(runtime) or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError("interleaved top-k requires sixteen initialized IPC peers and addressable cache pages")
    query = InterleavedTopKQuery(rows=4, world=16, rank=runtime.rank, cache_pages=cache_pages)

    def materialize(selection, device):
        from flashinfer.jit.interleaved_topk_slots import load_interleaved_topk_slots

        with torch.cuda.device(device.ordinal):
            return _State(load_interleaved_topk_slots().run, runtime, cache_pages)

    size = sum(buffer.nbytes for buffer in runtime._owned_buffers)
    return Plan(
        contract=_CONTRACT, query=query, _device=runtime.device,
        # FlashInfer builds and loads the single CUDA C++ entry during materialization.
        _compile_jobs=lambda config, device: (),
        _materialize=materialize,
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(PersistentMemory(("pcie-dcp-runtime", id(runtime)), size, size),)
        ),
    )
