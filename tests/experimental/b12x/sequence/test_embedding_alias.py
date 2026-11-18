"""Embedding alias checks for byte ranges inside shared allocation pools."""

import pytest
import torch

from b12x.sequence.embedding.api import _check_tensors, _overlaps


@pytest.mark.parametrize(
    "start,count,expected",
    [
        (0, 16, False),
        (16, 16, True),
        (40, 16, True),
        (72, 16, True),
        (80, 16, False),
        (120, 16, False),
        (16, 0, False),
        (40, 0, False),
        (80, 0, False),
    ],
)
def test_byte_spans_respect_strided_weight_rows_and_empty_views(start, count, expected):
    pool = torch.empty(160, dtype=torch.uint8)
    weight = pool.view(torch.bfloat16).as_strided((3, 8), (12, 1), storage_offset=8)
    other = pool[start : start + count]
    assert torch._C._overlaps(weight, other)
    assert _overlaps(weight, other) is expected
    assert _overlaps(other, weight) is expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_single_weight_row_ignores_unaddressed_row_stride(dtype):
    pool = torch.empty(128, dtype=dtype)
    weight = pool.as_strided((1, 8), (64, 1), storage_offset=8)
    assert not _overlaps(weight, pool[16:24])
    assert _overlaps(weight, pool[15:23])


def test_contiguous_q8_weight_span_includes_all_block_bytes():
    pool = torch.empty(256, dtype=torch.uint8)
    weight = pool[16:220].reshape(2, 3, 34)
    assert _overlaps(weight, pool[216:224].view(torch.bfloat16))
    assert not _overlaps(weight, pool[220:228].view(torch.bfloat16))


def test_byte_overlap_is_rejected_for_distinct_storage_wrappers():
    pool = bytearray(128)
    left = torch.frombuffer(pool, dtype=torch.uint8, count=64)
    overlapping = torch.frombuffer(pool, dtype=torch.int64, count=2, offset=48)
    disjoint = torch.frombuffer(pool, dtype=torch.int32, count=4, offset=64)
    assert not torch._C._overlaps(left, overlapping)
    assert _overlaps(left, overlapping)
    assert not _overlaps(left, disjoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_embedding_validation_accepts_disjoint_pool_ranges_and_rejects_overlap():
    pool = torch.empty(128, dtype=torch.uint8, device="cuda")
    weight = pool[:64].view(torch.bfloat16).reshape(4, 8)
    out = pool[64:80].view(torch.bfloat16).reshape(1, 8)
    ids = pool[80:88].view(torch.int64)
    count = pool[88:92].view(torch.int32)
    _check_tensors(weight, ids, out, count)
    _check_tensors(weight, ids[:0], out[:0])
    with pytest.raises(ValueError, match="alias inputs"):
        _check_tensors(
            weight, ids, pool[48:64].view(torch.bfloat16).reshape(1, 8), count
        )
    with pytest.raises(ValueError, match="alias inputs"):
        _check_tensors(weight, pool[64:72].view(torch.int64), out, count)
    with pytest.raises(ValueError, match="alias num_rows"):
        _check_tensors(weight, ids, out, pool[64:68].view(torch.int32))
