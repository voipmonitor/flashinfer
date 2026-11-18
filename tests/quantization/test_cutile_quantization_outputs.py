"""Caller-owned row-major quantization storage contracts."""

from importlib import import_module

import pytest
import torch


@pytest.fixture
def quantizer(monkeypatch):
    module = import_module(
        "flashinfer.quantization.kernels.cutile.per_token_group_quant_8bit_cutile"
    )
    launches = []
    monkeypatch.setattr(module.ct, "launch", lambda *args: launches.append(args))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    return module.per_token_group_quant_8bit_cutile, launches


def test_quantization_uses_disjoint_ranges_in_shared_storage(quantizer):
    quantize, launches = quantizer
    storage = torch.empty(256, dtype=torch.uint8)
    source = storage[:128].view(torch.bfloat16).view(2, 32)
    values = storage[128:192].view(torch.float8_e4m3fn).view(2, 32)
    scales = storage[192:200].view(torch.float32).view(2, 1)
    actual = quantize(source, 32, out=(values, scales))
    assert actual[0] is values and actual[1] is scales
    assert len(launches) == 1
    arguments = launches[0][3]
    assert [tensor.data_ptr() for tensor in arguments[:3]] == [
        tensor.data_ptr() for tensor in (source, values, scales)
    ]


@pytest.mark.parametrize("overlap", ["source", "scales"])
def test_quantization_rejects_actual_output_overlap(quantizer, overlap):
    quantize, launches = quantizer
    storage = torch.empty(256, dtype=torch.uint8)
    source = storage[:128].view(torch.bfloat16).view(2, 32)
    values = storage[64:128] if overlap == "source" else storage[128:192]
    values = values.view(torch.float8_e4m3fn).view(2, 32)
    scales = storage[128:136] if overlap == "scales" else storage[192:200]
    scales = scales.view(torch.float32).view(2, 1)
    with pytest.raises(ValueError, match="byte ranges must be disjoint"):
        quantize(source, 32, out=(values, scales))
    assert not launches


def test_quantization_rejects_wrong_output_metadata(quantizer):
    quantize, launches = quantizer
    source = torch.empty((2, 32), dtype=torch.bfloat16)
    values = torch.empty((2, 32), dtype=torch.int8)
    scales = torch.empty((2, 1))
    with pytest.raises(ValueError, match="quantized values must match"):
        quantize(source, 32, out=(values, scales))
    assert not launches
