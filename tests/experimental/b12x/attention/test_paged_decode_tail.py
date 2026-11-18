"""Private append pages preserve noncausal proposals beyond reserved cache pages."""

import pytest
import torch
from b12x.attention import paged_decode
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("prefixes", [[127], [127, 128, 0], [4095, 256, 129]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_private_tail_ragged_replay(split, prefixes, dtype):
    require_b12x()
    batch, heads, kv_heads, dim, width = len(prefixes), 12, 2, 128, 8
    page_width = max(1, (max(prefixes) + 1 + 127) // 128)
    key = (torch.randn(batch * page_width, 128, kv_heads, dim, device="cuda") * 0.3).to(
        dtype
    )
    value = (torch.randn_like(key.float()) * 0.3).to(dtype)
    if batch == 1:
        value.zero_()
    before_key, before_value = key.clone(), value.clone()
    pages = torch.arange(batch * page_width, dtype=torch.int32, device="cuda").view(
        batch, page_width
    )
    q = torch.empty(batch * width, heads, dim, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(q)
    tail_key = torch.empty(batch * 2, 128, kv_heads, dim, dtype=dtype, device="cuda")
    tail_value = torch.empty_like(tail_key)
    lengths = torch.empty(batch, dtype=torch.int32, device="cuda")
    cu = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
    scale = torch.ones(1, device="cuda")

    def draw(counts):
        q.normal_(std=0.3)
        lengths.copy_(
            torch.tensor(
                [p + n if n else 0 for p, n in zip(prefixes, counts)], device="cuda"
            )
        )
        cu.copy_(
            torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda")
        )
        tail_key.zero_()
        tail_value.zero_()
        for b, (prefix, count) in enumerate(zip(prefixes, counts)):
            partial = prefix % 128
            if partial:
                page = pages[b, prefix // 128].item()
                tail_key[b * 2, :partial].copy_(key[page, :partial])
                tail_value[b * 2, :partial].copy_(value[page, :partial])
            for dest in (tail_key, tail_value):
                append = (torch.randn(count, kv_heads, dim, device="cuda") * 0.3).to(
                    dtype
                )
                dest[b * 2 : b * 2 + 2].view(256, kv_heads, dim)[
                    partial : partial + count
                ].copy_(append)

    def check(counts):
        row = 0
        for b, (prefix, count) in enumerate(zip(prefixes, counts)):
            if not count:
                continue
            k = key[pages[b].long()].reshape(-1, kv_heads, dim)[:prefix].double()
            v = value[pages[b].long()].reshape(-1, kv_heads, dim)[:prefix].double()
            offset = prefix % 128
            k = torch.cat(
                (
                    k,
                    tail_key[b * 2 : b * 2 + 2]
                    .reshape(256, kv_heads, dim)[offset : offset + count]
                    .double(),
                )
            )
            v = torch.cat(
                (
                    v,
                    tail_value[b * 2 : b * 2 + 2]
                    .reshape(256, kv_heads, dim)[offset : offset + count]
                    .double(),
                )
            )
            k, v = k.repeat_interleave(6, 1), v.repeat_interleave(6, 1)
            scores = (
                torch.einsum("thd,khd->thk", q[row : row + count].double(), k)
                / dim**0.5
            )
            probability = scores.softmax(-1)
            expected = torch.einsum("thk,khd->thd", probability, v)
            mass = torch.einsum("thk,khd->thd", probability, v.abs())
            bound = 2**-8 * (mass + expected.abs()) + 2**-11 * mass
            assert (
                (out[row : row + count].double() - expected).abs() - bound
            ).max().item() <= 0
            row += count
        assert torch.equal(key.view(torch.uint8), before_key.view(torch.uint8))
        assert torch.equal(value.view(torch.uint8), before_value.view(torch.uint8))

    caps = paged_decode.Caps(
        device="cuda",
        num_q_heads=heads,
        num_kv_heads=kv_heads,
        head_dim_qk=dim,
        head_dim_vo=dim,
        page_size=128,
        max_batch=batch,
        max_q_per_req=width,
        kv_dtype=dtype,
        causal=False,
        tail_pages_per_request=2,
    )
    config = paged_decode.PagedDecodeConfig(
        split_kv=split,
        max_splits=8 if split else 1,
        head_splits=1,
        stage_rows=16,
        num_stages=8,
        min_tiles_per_split=4,
    )
    plan = paged_decode.plan(caps, override=config)
    arguments = dict(
        q=q,
        k_cache=key,
        v_cache=value,
        output=out,
        page_table=pages,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        tail_k_cache=tail_key,
        tail_v_cache=tail_value,
        k_descale=scale,
        v_descale=scale,
    )
    draw([8] * batch)
    with PreparationSession(device=torch.device("cuda"), autotune=False) as session:

        def prepare(state):
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device="cuda")
                for s in plan.scratch_specs()
            ]
            binding = state.bind(plan=plan, scratch=scratch, **arguments)
            return PreparedCall(
                run=lambda: state.run(binding), owners=(scratch, binding)
            )

        result = session.prepare((plan.request(name="tail", prepare_call=prepare),))
        scratch = [
            torch.empty(s.shape, dtype=s.dtype, device="cuda")
            for s in plan.scratch_specs()
        ]
        binding = paged_decode.bind(plan, scratch=scratch, **arguments)
        paged_decode.run(binding)
        torch.cuda.synchronize()
        check([8] * batch)
        if batch == 1:
            q.zero_()
            tail_value.zero_()
            tail_value.view(256, kv_heads, dim)[prefixes[0] % 128 + 7].fill_(1)
            paged_decode.run(binding)
            torch.cuda.synchronize()
            assert torch.equal(out, torch.full_like(out, 1 / (prefixes[0] + 8)))
            draw([8])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)
        for counts in ([8, 3, 0], [1, 8, 4]):
            counts = counts[:batch]
            draw(counts)
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            check(counts)
        result.close()
