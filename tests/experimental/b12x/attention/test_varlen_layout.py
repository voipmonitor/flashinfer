"""Packed attention layouts retain row strides in validation and tuning queries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.attention._shared.contiguous.api import (
    _validate_input_layout,
    _validate_varlen_inputs_against_plan,
)
from b12x.attention.varlen._tuning import TUNING, VarlenAttentionQuery


@pytest.mark.cpu_only
def test_interleaved_qkv_layout_is_valid_only_for_packed_attention():
    packed = torch.empty(7, 3, 12, 128, dtype=torch.bfloat16)
    for tensor in packed.unbind(1):
        _validate_input_layout(tensor, allow_row_padding=True)
        with pytest.raises(ValueError, match="must be contiguous"):
            _validate_input_layout(tensor, allow_row_padding=False)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "stride", [(4608, 1, 12), (4608, 129, 1), (768, 128, 1), (4609, 128, 1)]
)
def test_packed_attention_rejects_unsupported_layouts(stride):
    tensor = torch.empty_strided((7, 12, 128), stride, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous heads"):
        _validate_input_layout(tensor, allow_row_padding=True)


@pytest.mark.cpu_only
def test_live_rows_keep_prepared_strides():
    plan = SimpleNamespace(
        q_shape=(257, 12, 128),
        k_shape=(257, 12, 128),
        v_shape=(257, 12, 128),
        cu_seqlens_q_shape=(4,),
        cu_seqlens_k_shape=(4,),
        q_stride=(4608, 128, 1),
        k_stride=(4608, 128, 1),
        v_stride=(4608, 128, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    arguments = dict(vars(plan), plan=plan)
    arguments.update(
        q_shape=(129, 12, 128),
        k_shape=(129, 12, 128),
        v_shape=(129, 12, 128),
        cu_seqlens_q_shape=(3,),
        cu_seqlens_k_shape=(3,),
    )
    _validate_varlen_inputs_against_plan(**arguments)
    arguments["v_stride"] = (1536, 128, 1)
    with pytest.raises(ValueError, match="strides must match"):
        _validate_varlen_inputs_against_plan(**arguments)


@pytest.mark.cpu_only
def test_tuning_query_distinguishes_row_padding():
    query = VarlenAttentionQuery(
        variant="varlen",
        dtype="bfloat16",
        causal=False,
        batch_size=1,
        q_heads=12,
        kv_heads=12,
        q_head_dim=128,
        v_head_dim=128,
        query_rows=257,
        kv_rows=257,
        max_seqlen_q=257,
        max_seqlen_k=257,
        q_row_stride=1536,
        k_row_stride=1536,
        v_row_stride=1536,
    )
    padded = replace(query, q_row_stride=4608, k_row_stride=4608, v_row_stride=4608)
    assert TUNING.encode_query(query) != TUNING.encode_query(padded)


@pytest.mark.cpu_only
def test_packed_attention_rejects_misaligned_storage_offset():
    storage = torch.empty(7 * 4608 + 1, dtype=torch.bfloat16)
    tensor = storage[1:].as_strided((7, 12, 128), (4608, 128, 1))
    with pytest.raises(ValueError, match="contiguous heads"):
        _validate_input_layout(tensor, allow_row_padding=True)
