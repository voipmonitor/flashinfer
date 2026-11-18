"""Caller-owned replay schedules preserve aliases without retaining declaration storage."""

from dataclasses import replace
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from b12x.attention import paged
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    _B12XPagedPlanMetadataCache,
    _materialize_paged_attention_scratch,
    plan_paged_attention_scratch,
)


def _plan():
    caps = B12XPagedAttentionScratchCaps(
        device="cpu",
        mode="decode",
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim_qk=16,
        head_dim_vo=16,
        page_size=4,
        max_total_q=2,
        max_batch=2,
        max_page_table_width=3,
        max_work_items=4,
        max_partial_rows=4,
        num_cache_pages=8,
        use_cuda_graph=True,
    )
    plan = plan_paged_attention_scratch(caps)
    plan._plan = SimpleNamespace(
        enable_cuda_graph=True,
        split_kv=True,
        msa_block_sparse=False,
        padded_batch_size=4,
        total_num_partial_rows=4,
        kv_chunk_size=16,
        total_q=2,
    )
    plan._plan_metadata_cache = _B12XPagedPlanMetadataCache(
        *(torch.arange(4, dtype=torch.int32) for _ in range(7)),
        torch.tensor([16], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        16,
        2,
    )
    plan._decode_graph_chunk_pages_lut = torch.tensor([1, 2, 4], dtype=torch.int32)
    return plan


def test_schedule_handoff_releases_declaration_tensors_and_preserves_seeds():
    plan = _plan()
    original = paged.replay_metadata(plan)
    owners = [weakref.ref(tensor) for tensor in original.tensors]
    seeds = paged.ReplayMetadata(
        tuple(tensor.clone() for tensor in original.tensors),
        original.chunk_pages_lut.clone(),
        original.live,
    )
    assert seeds.live
    del original
    paged.release_replay_metadata(plan)
    gc.collect()
    assert all(owner() is None for owner in owners)
    assert plan._plan_metadata_cache is None
    assert plan._decode_graph_chunk_pages_lut is None
    with pytest.raises(RuntimeError, match="explicit caller operands"):
        paged.replay_metadata(plan)
    cache, lut = plan._bind_replay_metadata(seeds)
    assert cache.request_indices is seeds.tensors[0]
    assert lut is seeds.chunk_pages_lut
    assert cache.kv_chunk_size == 16 and cache.total_q == 2


def test_decode_schedule_aliases_invocation_metadata_not_shared_workspace():
    plan = _plan()
    seeds = paged.replay_metadata(plan)
    live = replace(seeds, tensors=tuple(tensor.clone() for tensor in seeds.tensors))
    cache, lut = plan._bind_replay_metadata(live)
    storage = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8)
    scratch = _materialize_paged_attention_scratch(
        plan.caps,
        storage,
        plan.layout,
        plan=plan._plan,
        plan_q=plan._plan_q,
        plan_output=plan._plan_output,
        plan_k_cache=plan._plan_k_cache,
        plan_v_cache=plan._plan_v_cache,
        planner_budget=plan._planner_budget,
        plan_metadata_cache=cache,
        decode_graph_chunk_pages_lut=lut,
        decode_graph_max_chunks_per_req=2,
        use_regular_decode_graph_replay=True,
        owner_scratch_plan=plan,
    )
    scratch._copy_cached_plan_metadata(cache)
    assert scratch.merge_indptr is live.tensors[3]
    assert scratch.merge_indptr.data_ptr() != seeds.tensors[3].data_ptr()
    scratch.merge_indptr.fill_(9)
    assert (seeds.tensors[3] == torch.arange(4)).all()
    scratch.merge_indptr = seeds.tensors[3]
    with pytest.raises(RuntimeError, match="must alias"):
        scratch._copy_cached_plan_metadata(cache)


@pytest.mark.parametrize("kind", ["shape", "dtype", "count", "stride"])
def test_schedule_handoff_rejects_incompatible_operands(kind):
    plan = _plan()
    metadata = paged.replay_metadata(plan)
    paged.release_replay_metadata(plan)
    tensors = list(metadata.tensors)
    if kind == "shape":
        tensors[0] = tensors[0][:-1]
    elif kind == "dtype":
        tensors[0] = tensors[0].long()
    elif kind == "stride":
        tensors[0] = torch.empty(8, dtype=torch.int32)[::2]
    else:
        tensors.pop()
    with pytest.raises(ValueError, match="prepared layout"):
        plan._bind_replay_metadata(replace(metadata, tensors=tuple(tensors)))


def test_retired_plan_owned_bindings_fail_closed_after_capture():
    plan = _plan()
    cache = plan._plan_metadata_cache
    plan._mark_decode_graph_replay_state_captured(cache)
    paged.release_replay_metadata(plan)
    with pytest.raises(RuntimeError, match="stale paged attention binding"):
        plan._mark_decode_graph_replay_state_captured(cache)
    assert plan._decode_graph_replay_state_captured


def test_explicit_metadata_keeps_sparse_indices_local_to_each_binding(monkeypatch):
    from b12x.attention.paged._scratch import B12XPagedAttentionScratch

    source = _plan()
    plan = plan_paged_attention_scratch(
        replace(
            source.caps,
            msa_block_sparse=True,
            num_q_heads=16,
            head_dim_qk=128,
            head_dim_vo=128,
        )
    )
    plan._plan = source._plan
    plan._plan_metadata_cache = source._plan_metadata_cache
    plan._decode_graph_chunk_pages_lut = source._decode_graph_chunk_pages_lut
    seeds = paged.replay_metadata(plan)
    first = torch.zeros((1, 2, 16), dtype=torch.int32)
    second = torch.ones_like(first)
    plan._q2k_indices_data_ptr = first.data_ptr()

    def prepare(scratch, pages, lengths, cu, **kwargs):
        scratch.page_table, scratch.cache_seqlens, scratch.cu_seqlens_q = (
            pages,
            lengths,
            cu,
        )
        return scratch

    monkeypatch.setattr(B12XPagedAttentionScratch, "prepare", prepare)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    arguments = dict(
        scratch=torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8),
        q=torch.empty((2, 16, 128), dtype=torch.bfloat16),
        k_cache=torch.empty((8, 4, 1, 128), dtype=torch.bfloat16),
        v_cache=torch.empty((8, 4, 1, 128), dtype=torch.bfloat16),
        output=torch.empty((2, 16, 128), dtype=torch.bfloat16),
        page_table=torch.zeros((2, 3), dtype=torch.int32),
        cache_seqlens=torch.ones(2, dtype=torch.int32),
        cu_seqlens_q=torch.arange(3, dtype=torch.int32),
        active_total_q=2,
    )
    for indices in (first, second, first):
        binding = plan.bind(**arguments, q2k_indices=indices, replay_metadata=seeds)
        assert binding.scratch.q2k_indices is indices
        assert plan._q2k_indices_data_ptr == first.data_ptr()
    with pytest.raises(ValueError, match="data_ptr changed"):
        plan.bind(**arguments, q2k_indices=second)
