# SPDX-License-Identifier: Apache-2.0

"""Prepared launch adapters consume explicit operands without retaining declaration storage."""

from contextlib import nullcontext
from types import SimpleNamespace
import weakref

import pytest
import torch


def test_paged_mqa_uses_runtime_storage_and_preserves_untouched_logits(monkeypatch):
    import tvm_ffi
    from flashinfer.experimental.deepgemm_dense_mqa.paged_mqa import PagedMqaPlan

    monkeypatch.setattr(tvm_ffi, "use_torch_stream", nullcontext)
    calls = []

    def launch(query, cache, scales, weights, lengths, table, output):
        calls.append((query, cache, scales, weights, lengths, table))
        output[:, :2].copy_(weights[:, :1].expand(-1, 2))

    names = ("Q", "KV", "KV_scales", "Weights", "context_lens", "block_table", "Logits")
    record = {"ffi_entry": "run", "arg_plan": [("tensor", name) for name in names]}
    op = object.__new__(PagedMqaPlan)
    template = torch.zeros(1)
    reference = weakref.ref(template)
    op._submissions = [(launch, (template,))]
    op.output = op.logical_output = op.schedule_meta = template
    op._retained = (template,)
    op._programs, op.num_sms, op.max_context_len = [({"run": launch}, record)], 4, 5
    op.functional()
    del template
    assert reference() is None and op.launch_count == 1
    query = torch.zeros(2, 1, 2, 128, dtype=torch.float8_e4m3fn)
    cache = torch.zeros(3, 64, 1, 132, dtype=torch.uint8)
    weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    lengths = torch.tensor([[1], [2]], dtype=torch.int32)
    table = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    output = torch.full((2, 8), -999.0)
    result = op.run(query, cache, weights, lengths, table, output=output)
    assert result.shape == (2, 5) and result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result[:, :2], weights[:, :1].expand(-1, 2))
    assert torch.all(output[:, 2:] == -999)
    assert calls[0][0].data_ptr() == query.data_ptr()
    assert calls[0][1].data_ptr() == cache.data_ptr()
    assert calls[0][3] is weights
    assert op._retained is op.output is op._submissions is None
    with pytest.raises(ValueError, match="every operand"):
        op.run(query)


def test_cake_functional_launch_has_no_declaration_owner(monkeypatch):
    from flashinfer.experimental.cake_dsa_train import cake_backend as module

    monkeypatch.setattr(module, "_ffi_stream_context", lambda _: nullcontext())
    calls = []

    def launcher(run, values, grid):
        calls.append(values)
        values["out"].copy_(values["q_latent"])
        values["lse"].copy_(values["topk_length"][:, None].expand_as(values["lse"]))
        values["o_lo"].zero_()

    plan = SimpleNamespace(
        backward=False,
        device_index=0,
        constants={"idx_stride": 4},
        entries={
            key: SimpleNamespace(launcher=launcher, run=None, grid=(1, 1, 1))
            for key in module.FORWARD_STAGES
        },
    )
    runner = object.__new__(module.DSATrainRunner)
    runner.plan = plan
    template = torch.zeros(1)
    reference = weakref.ref(template)
    runner.tensors = {"q_latent": template}
    runner.launches = {"forward": template}
    functional = runner.functional()
    del template, runner
    assert reference() is None
    q, qr, kv, kr = (
        torch.randn(2, 64, 512),
        torch.zeros(2, 64, 64),
        torch.randn(4, 512),
        torch.zeros(4, 64),
    )
    ids, lengths = (
        torch.zeros(2, 4, dtype=torch.int32),
        torch.tensor([2, 3], dtype=torch.int32),
    )
    output, lse, residual = torch.empty_like(q), torch.empty(2, 64), torch.empty_like(q)
    result = functional.forward_out(
        q, qr, kv, kr, ids, lengths, out=output, lse=lse, o_lo=residual
    )
    assert all(
        actual is expected
        for actual, expected in zip(result, (output, lse, residual), strict=True)
    )
    torch.testing.assert_close(output, q)
    torch.testing.assert_close(lse, lengths[:, None].float().expand_as(lse))
    assert not residual.count_nonzero()
    assert calls[0]["q_latent"] is q and calls[0]["indices"] is ids
    assert functional.topk_length is None
    with pytest.raises(ValueError, match="explicit top-k"):
        functional.forward_out(
            q, qr, kv, kr, ids, None, out=output, lse=lse, o_lo=residual
        )
