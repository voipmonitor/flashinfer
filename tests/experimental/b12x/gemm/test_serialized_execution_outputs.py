"""Serialized fixed GEMM execution uses caller-owned outputs and split workspace."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.gemm.blockscaled import _a16
from b12x.gemm.blockscaled._preparation import _FixedExecutionState


@pytest.mark.parametrize("rows", [0, 2])
def test_serialized_execution_preserves_explicit_destinations(monkeypatch, rows):
    query = SimpleNamespace(
        call_kind="serialized",
        recipe="block_fp8",
        in_features=128,
        padded_in_features=128,
        out_features=128,
        max_rows=2,
        expected_m=None,
        input_dtype="float8_e4m3fn",
        output_dtype="bfloat16",
    )
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(3)
        return kwargs["out"]

    state = _FixedExecutionState(
        query, torch.device("cpu"), SimpleNamespace(run=run), None, None, True
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(_a16, "_stream_context", lambda *args: nullcontext())
    source = torch.empty((rows, 128), dtype=torch.float8_e4m3fn)
    weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
    output = torch.empty((rows, 128), dtype=torch.bfloat16)
    workspace = torch.empty(256)
    result = state.run_serialized(
        source,
        torch.empty((rows, 1)),
        weight,
        torch.empty((1, 1)),
        None,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        sf_vec_size=128,
        block_fp8=True,
        stream=None,
        out=output,
        workspace=workspace,
    )
    assert result.untyped_storage() is output.untyped_storage()
    if rows:
        assert calls[0]["split_k_workspace"] is workspace
        torch.testing.assert_close(output, torch.full_like(output, 3), atol=0, rtol=0)
    else:
        assert result is output and not calls
