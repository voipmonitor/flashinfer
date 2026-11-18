"""Exact prepared Kimi selection across thread counts and mutable CUDA graph inputs."""

import pytest
import torch
from cuda.bindings import driver as cuda

from b12x.comm import pcie
from b12x.comm.pcie import _dcp_preparation
from b12x.preparation import PreparationSession

pytestmark = pytest.mark.gpu_exclusive


def _fill(logits, bias, mode, seed):
    generator = torch.Generator(device=logits.device).manual_seed(seed)
    if mode == "random":
        logits.normal_(generator=generator).mul_(7)
        bias.normal_(generator=generator).mul_(0.1)
    elif mode == "ties":
        logits.zero_()
        bias.zero_()
    elif mode == "bias_ties":
        logits.zero_()
        bias.copy_((torch.arange(896, device=bias.device) % 5).float() * 0.125)
    else:
        values = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 90.0, -90.0],
            device=logits.device,
        )
        logits.copy_(
            values[
                torch.arange(logits.numel(), device=logits.device).reshape(logits.shape)
                % 7
            ]
        )
        if mode == "invalid_scores":
            logits.fill_(float("nan"))
            bias.fill_(float("nan"))
        else:
            bias.copy_(values[(torch.arange(896, device=bias.device) * 3 + 1) % 7])


@pytest.mark.parametrize("rows", [1, 2, 4, 7, 8])
@pytest.mark.parametrize("threads", [128, 256, 512])
@torch.no_grad()
def test_prepared_kimi_topk_exact_graph_replay(rows, threads):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires hardware warp integer reduction")
    device = torch.device("cuda", torch.cuda.current_device())
    logits = torch.zeros((rows, 896), device=device)
    bias = torch.zeros(896, device=device)
    reference = (
        torch.empty((rows, 16), device=device),
        torch.empty((rows, 16), dtype=torch.int32, device=device),
    )
    output = (torch.empty_like(reference[0]), torch.empty_like(reference[1]))
    with PreparationSession(
        device=device, autotune=True, compile_workers=0
    ) as session:
        plans = {}
        for count in sorted({128, threads}):
            call = dict(
                router_logits=logits,
                correction_bias=bias,
                output_weights=output[0],
                output_ids=output[1],
                threads=count,
            )
            query = pcie.query_from_runtime(None, surface="kimi_topk16", call=call)
            plan = pcie.plan(query)
            session.prepare(
                (
                    plan.request(
                        name=f"selector-{count}",
                        prepare_call=lambda state, call=call: (
                            _dcp_preparation.prepare_call(state, **call)
                        ),
                    ),
                )
            )
            assert plan.selection.source == "fixed"
            plans[count] = plan
        session.freeze()
        pcie.kimi_topk16(logits, bias, *output, plan=plans[threads], threads=threads)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with session.capture(), torch.cuda.graph(graph):
                returned = pcie.kimi_topk16(
                    logits, bias, *output, plan=plans[threads], threads=threads
                )
            assert [value.data_ptr() for value in returned] == [
                value.data_ptr() for value in output
            ]
            handle = cuda.CUgraph(graph.raw_cuda_graph())
            status, _, count = cuda.cuGraphGetNodes(handle)
            assert status == cuda.CUresult.CUDA_SUCCESS and count == 1
            status, nodes, count = cuda.cuGraphGetNodes(handle, count)
            assert status == cuda.CUresult.CUDA_SUCCESS
            status, parameters = cuda.cuGraphKernelNodeGetParams(nodes[0])
            assert status == cuda.CUresult.CUDA_SUCCESS
            status, symbol = cuda.cuFuncGetName(parameters.func)
            assert status == cuda.CUresult.CUDA_SUCCESS
            assert (b"WarpTopK16Launch" in symbol) == (threads == 256)
            for index, mode in enumerate(
                ("random", "ties", "bias_ties", "nonfinite", "invalid_scores")
            ):
                _fill(logits, bias, mode, 813 + rows + index)
                pcie.kimi_topk16(logits, bias, *reference, plan=plans[128], threads=128)
                output[0].fill_(float("nan"))
                output[1].fill_(-777)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                assert torch.equal(output[1], reference[1])
                assert torch.equal(
                    output[0].view(torch.int32), reference[0].view(torch.int32)
                )
                if mode in ("ties", "invalid_scores"):
                    expected = torch.arange(
                        16, device=device, dtype=torch.int32
                    ).expand(rows, -1)
                    assert torch.equal(output[1], expected)
        finally:
            graph.reset()
