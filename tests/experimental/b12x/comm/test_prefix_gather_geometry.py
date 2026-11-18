"""Validate the padded-gather metadata and mutation contract without a CUDA context."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie._prefix_gather_preparation import validate_inputs


def inputs():
    source = torch.empty((2, 1, 16), dtype=torch.bfloat16)
    output = torch.empty((2, 16, 16), dtype=torch.bfloat16)
    prefixes = torch.empty((2, 64), dtype=torch.int32)
    query = SimpleNamespace(
        call=dict(
            shape=source.shape, out_shape=output.shape, prefix_shape=prefixes.shape
        )
    )
    return query, source, output, prefixes


def test_disjoint_prepared_geometry():
    query, source, output, prefixes = inputs()
    validate_inputs(query, source, output, prefixes, source.device)


@pytest.mark.parametrize("name", ["source", "output", "prefixes"])
def test_wrong_dtype_rejected(name):
    query, source, output, prefixes = inputs()
    tensors = dict(source=source, output=output, prefixes=prefixes)
    tensors[name] = tensors[name].float()
    with pytest.raises(ValueError, match="shape or dtype"):
        validate_inputs(query, **tensors, device=source.device)


def test_output_cannot_alias_source():
    query, _, output, prefixes = inputs()
    source = output.reshape(-1)[:32].view(2, 1, 16)
    with pytest.raises(ValueError, match="overlaps a live input"):
        validate_inputs(query, source, output, prefixes, source.device)


def test_output_cannot_alias_replicated_prefix_metadata():
    query, source, output, _ = inputs()
    prefixes = output.view(torch.int32).reshape(-1)[:128].view(2, 64)
    with pytest.raises(ValueError, match="overlaps a live input"):
        validate_inputs(query, source, output, prefixes, source.device)


def test_prefix_metadata_must_have_prepared_contiguous_stride():
    query, source, output, _ = inputs()
    prefixes = torch.empty((64, 2), dtype=torch.int32).t()
    with pytest.raises(ValueError, match="contiguous"):
        validate_inputs(query, source, output, prefixes, source.device)
