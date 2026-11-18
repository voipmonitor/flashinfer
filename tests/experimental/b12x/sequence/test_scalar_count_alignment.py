"""Prepared convolution and recurrent count-pointer alignment requirements."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.sequence.gdn_decode import _preparation as gdn
from b12x.sequence.gdn_decode._impl import Caps
from b12x.sequence.gdn_decode._tuning import _OPERANDS as GDN_OPERANDS
from b12x.sequence.short_conv import _preparation as conv
from b12x.sequence.short_conv._tuning import OPERANDS as CONV_OPERANDS, ShortConvQuery


def _byte_view(shape, dtype, offset):
    count = torch.Size(shape).numel()
    storage = bytearray(offset + count * dtype.itemsize + 16)
    return torch.frombuffer(storage, dtype=dtype, count=count, offset=offset).reshape(
        shape
    )


def _convolution(tensors=None):
    if tensors is None:
        tensors = dict(
            x=torch.empty((2, 128), dtype=torch.bfloat16),
            conv_weight=torch.empty((128, 4), dtype=torch.bfloat16),
            query_start_loc=torch.empty(3, dtype=torch.int32),
            state_slot_ids=torch.empty(2, dtype=torch.int64),
            state_is_fresh=torch.empty(2, dtype=torch.bool),
            num_accepted_tokens=torch.empty(2, dtype=torch.int32),
            request_is_prefill=torch.empty(2, dtype=torch.bool),
            num_seqs=torch.empty(1, dtype=torch.int32),
            num_tokens=torch.empty(1, dtype=torch.int32),
            conv_state=torch.empty((4, 128, 3), dtype=torch.bfloat16),
            out=torch.empty((2, 128), dtype=torch.bfloat16),
            gathered_state=torch.empty((2, 128, 3), dtype=torch.bfloat16),
            request_ids=torch.empty(2, dtype=torch.int32),
            conv_bias=None,
        )
    query = ShortConvQuery(
        mode="decode",
        channels=128,
        kernel_size=4,
        dilation=1,
        max_tokens=2,
        max_seqs=2,
        max_state_slots=4,
        max_speculative_tokens=0,
        **conv.invocation_from_tensors(**tensors),
    )
    return (
        tensors,
        conv.PreparedShortConv(query, torch.device("cpu"), ()),
        CONV_OPERANDS,
    )


def _recurrent(kda, tensors=None):
    key_heads, value_heads = (1, 1) if kda else (1, 2)
    if tensors is None:
        tensors = dict(
            mixed_qkv=torch.empty(
                (2, (2 * key_heads + value_heads) * 128), dtype=torch.bfloat16
            ),
            a=torch.empty((2, value_heads, 128), dtype=torch.bfloat16),
            b=torch.empty((2, value_heads), dtype=torch.bfloat16),
            z=torch.empty((2, value_heads, 128), dtype=torch.bfloat16),
            A_log=torch.empty(value_heads, dtype=torch.float32),
            dt_bias=torch.empty(
                (value_heads, 128), dtype=torch.float32 if kda else torch.bfloat16
            ),
            norm_weight=torch.empty(128, dtype=torch.bfloat16),
            recurrent_state=torch.empty(
                (4, value_heads, 128, 128), dtype=torch.float32
            ),
            query_start_loc=torch.empty(3, dtype=torch.int32),
            num_accepted_tokens=torch.empty(2, dtype=torch.int32),
            state_indices=torch.empty((2, 1), dtype=torch.int32),
            num_seqs=torch.empty(1, dtype=torch.int32),
            num_tokens=torch.empty(1, dtype=torch.int32),
            output=torch.empty((2, value_heads, 128), dtype=torch.bfloat16),
        )
    caps = Caps(
        device="cuda:0",
        max_tokens=2,
        max_seqs=2,
        max_state_slots=4,
        key_heads=key_heads,
        value_heads=value_heads,
    )
    arguments = tensors | (
        {"raw_g": tensors["a"], "raw_beta": tensors["b"]} if kda else {}
    )
    query = gdn._query_from_caps(caps, gdn.invocation_from_tensors(caps, **arguments))
    layout = SimpleNamespace(
        caps=SimpleNamespace(device=torch.device("cpu")), recurrent_block_v=32
    )
    return tensors, gdn._GdnState(query, layout, None, None), GDN_OPERANDS


def _case(kind, tensors=None):
    return (
        _convolution(tensors) if kind == "conv" else _recurrent(kind == "kda", tensors)
    )


@pytest.mark.parametrize("kind", ["conv", "qwen", "kda"])
@pytest.mark.parametrize("example_offset", [0, 4, 8, 12])
@pytest.mark.parametrize("runtime_offset", [0, 4, 8, 12])
def test_scalar_alignment_is_independent_of_example_allocation(
    kind, example_offset, runtime_offset
):
    tensors, _, _ = _case(kind)
    for name in ("num_seqs", "num_tokens"):
        tensors[name] = _byte_view((1,), torch.int32, example_offset)
    _, prepared, names = _case(kind, tensors)
    for name in ("num_seqs", "num_tokens"):
        if kind == "conv":
            assert prepared.query.input_alignments[names.index(name)] == 4
        else:
            assert prepared.query.pointer_alignments[name] == 4
        tensors[name] = _byte_view((1,), torch.int32, runtime_offset)
    prepared._check(tuple(tensors[name] for name in names))


@pytest.mark.parametrize("kind", ["conv", "qwen", "kda"])
@pytest.mark.parametrize("name", ["num_seqs", "num_tokens"])
def test_scalar_counts_reject_less_than_natural_alignment(kind, name):
    tensors, prepared, names = _case(kind)
    tensors[name] = _byte_view((1,), torch.int32, 2)
    assert conv._alignment(tensors[name]) == 2
    with pytest.raises(ValueError, match=name + ".*alignment"):
        prepared._check(tuple(tensors[name] for name in names))


@pytest.mark.parametrize(
    "kind,name",
    [
        ("conv", "x"),
        ("conv", "conv_weight"),
        ("conv", "conv_state"),
        ("qwen", "norm_weight"),
        ("kda", "mixed_qkv"),
        ("kda", "recurrent_state"),
    ],
)
def test_vectors_and_history_retain_prepared_alignment(kind, name):
    tensors, prepared, names = _case(kind)
    tensors[name] = _byte_view(tensors[name].shape, tensors[name].dtype, 8)
    assert conv._alignment(tensors[name]) == 8
    with pytest.raises(ValueError, match=name + ".*alignment"):
        prepared._check(tuple(tensors[name] for name in names))


@pytest.mark.parametrize(
    "kind,name", [("conv", "x"), ("qwen", "norm_weight"), ("kda", "mixed_qkv")]
)
def test_stronger_vector_alignment_satisfies_prepared_requirement(kind, name):
    tensors, prepared, names = _case(kind)
    if kind == "conv":
        alignments = list(prepared.query.input_alignments)
        alignments[names.index(name)] = 8
        query = replace(prepared.query, input_alignments=tuple(alignments))
    else:
        query = replace(
            prepared.query,
            pointer_alignments=dict(prepared.query.pointer_alignments) | {name: 8},
        )
    prepared = replace(prepared, query=query)
    assert conv._alignment(tensors[name]) == 16
    prepared._check(tuple(tensors[name] for name in names))
