"""Tune TP16 IslandRS launches and replay mixed shapes on one fixed-grid owner."""

from dataclasses import asdict
from datetime import timedelta
import os
from pathlib import Path

import pytest

from .test_collective_tuning_gpu import _drive
from .test_island_rs_load_batches_gpu import _input, _ordered_reference


def _worker(rank, directory, blocks):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import plan, query_from_runtime
    from b12x.comm.pcie import _owner_preparation
    from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce
    from b12x.preparation import CollectiveRequirement, PreparationSession

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=16,
        init_method=(Path(directory) / "group").as_uri(),
        timeout=timedelta(minutes=10),
    )
    owner = PCIeIslandRSAllReduce(
        exchange_group=dist.group.WORLD,
        device=device,
        max_elements=1024 * 7168,
        blocks=blocks,
    )
    assert not owner.threads_override, "run without B12X_PCIE_ISLAND_RS_THREADS"
    try:
        for cached in (False, True):
            session = PreparationSession(
                device=device,
                cache_dir=Path(directory) / f"rank{rank}",
                namespace={"test": "island-active-tuning"},
                cache_only=cached,
                compile_workers=4,
                rounds=2,
                collective_barrier=lambda key, ranks: dist.barrier(),
            )
            cases, requests = [], []
            observed_alignments = set()
            for rows in (2, 128, 1024):
                for inplace in (False, True):
                    source_offset, output_offset = {
                        2: (2, 4),
                        128: (4, 2),
                        1024: (0, 0),
                    }[rows]
                    shape = (rows, 7168)
                    source = (
                        torch.empty(
                            rows * 7168 + source_offset,
                            device=device,
                            dtype=torch.bfloat16,
                        )[source_offset:].view(shape)
                        if source_offset
                        else torch.empty(shape, device=device, dtype=torch.bfloat16)
                    )
                    output = (
                        source
                        if inplace
                        else (
                            torch.empty(
                                rows * 7168 + output_offset,
                                device=device,
                                dtype=torch.bfloat16,
                            )[output_offset:].view(shape)
                            if output_offset
                            else torch.empty_like(source)
                        )
                    )
                    seed = torch.zeros_like(source)
                    query = query_from_runtime(
                        owner,
                        surface="PCIeIslandRSAllReduce.all_reduce",
                        call={"inp": source, "out": output},
                    )
                    prepared = plan(query, runtime=owner)

                    def benchmark(
                        state,
                        factory=prepared._benchmark_call,
                        inplace=inplace,
                        observed=observed_alignments,
                    ):
                        call = factory(state)
                        operands = call.owners[0]
                        input_alignment = _owner_preparation._alignment(operands.source)
                        output_alignment = _owner_preparation._alignment(operands.out)
                        assert input_alignment == state.query.call["inp"]["alignment"]
                        assert (
                            output_alignment == state.query.call["output"]["alignment"]
                        )
                        assert (
                            operands.source.data_ptr() == operands.out.data_ptr()
                        ) == inplace
                        observed.add((input_alignment, output_alignment, inplace))
                        return call

                    name = f"rows{rows}-inplace{int(inplace)}"
                    requests.append(
                        prepared.request(
                            name=name,
                            prepare_call=lambda state, source=source, output=output: (
                                _owner_preparation.prepared_call(
                                    state, inp=source, out=output
                                )
                            ),
                            benchmark_call=benchmark,
                            collective=CollectiveRequirement(name, tuple(range(16))),
                        )
                    )
                    cases.append((source, output, seed, prepared, inplace))
            result = _drive(session, requests)
            assert (result.benchmarked_candidates == 0) == cached
            if not cached:
                assert observed_alignments == {
                    (4, 8, False),
                    (4, 4, True),
                    (8, 4, False),
                    (8, 8, True),
                    (256, 256, False),
                    (256, 256, True),
                }
            for source, output, seed, prepared, inplace in cases:
                assert prepared.selection.source == ("cached" if cached else "tuned")
                config = prepared.selection.config
                assert config.blocks == owner.blocks
                assert 1 <= config.active_blocks <= config.blocks
                if blocks is not None:
                    assert config.active_blocks == blocks
                peers = [None] * 16
                dist.all_gather_object(peers, asdict(config))
                assert all(value == peers[0] for value in peers)
                source.zero_()
                owner.all_reduce(
                    source,
                    out=output,
                    plan=prepared,
                    active_blocks=config.active_blocks,
                )
            torch.cuda.synchronize()
            graphs = []
            for order in (cases, list(reversed(cases))):
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with session.capture(), torch.cuda.graph(graph):
                    for source, output, seed, prepared, inplace in order:
                        if not inplace:
                            output.fill_(float("nan"))
                        source.copy_(seed)
                        owner.all_reduce(source, out=output, plan=prepared)
                graphs.append(graph)
            session.freeze()
            for variation in range(3):
                expected = []
                for source, output, seed, prepared, inplace in cases:
                    seed.copy_(
                        _input(source.numel(), rank, variation, device).view_as(source)
                    )
                    expected.append(
                        _ordered_reference(source.numel(), variation, device).view_as(
                            source
                        )
                    )
                dist.barrier()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                if rank == variation:
                    torch.cuda._sleep(200000)
                for graph in (*graphs, *graphs):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                for case, reference in zip(cases, expected, strict=True):
                    torch.testing.assert_close(case[1], reference, rtol=0, atol=0)
            for graph in graphs:
                graph.reset()
            result.close()
            session.close()
    except BaseException:
        import sys
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        owner.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_COLLECTIVE_TUNING_TEST") != "1",
    reason="requires sixteen peer-accessible SM120 GPUs",
)
@pytest.mark.parametrize("blocks", [None, 16])
def test_island_rs_participation_tuning_and_cached_replay(tmp_path, blocks):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() != 16 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires sixteen SM120 GPUs")
    mp.spawn(_worker, args=(str(tmp_path), blocks), nprocs=16, join=True)
