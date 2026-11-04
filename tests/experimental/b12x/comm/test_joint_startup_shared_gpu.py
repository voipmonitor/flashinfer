"""Startup races of one runtime share eviction and timing storage across replays."""

from datetime import timedelta
import gc
import os
from pathlib import Path

import pytest

SIZES = (4112, 163840, 327680, 1310720)
CAPACITY_BYTES = 1310720
PROFILES = (("payload_rsag", 42), ("bulk_push", 8), ("pull", 7))


def _exercise(rank):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import (
        CollectiveConfig,
        joint_startup_spec,
        prepare_joint_startup_tuner,
    )
    from b12x.comm.pcie._joint_profile import JOINT_MODES
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.preparation import CollectiveRequirement, PreparationSession

    device = torch.device("cuda", rank)
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        joint=True,
        max_size=CAPACITY_BYTES,
        eager_buffer_bytes=CAPACITY_BYTES,
    )
    manager.prepare_channels(("startup-shared",))
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    plans, requests, graphs = [], [], []
    try:
        for nbytes in SIZES:
            source = torch.full(
                (nbytes // 2,), rank + 1, dtype=torch.bfloat16, device=device
            )
            out = torch.empty_like(source)
            plan = manager.plan(
                source,
                out=out,
                joint_only=True,
                joint_cluster_size=1,
                override=CollectiveConfig(
                    implementation="joint_bandwidth",
                    threads=512,
                    blocks=64,
                    submode="payload_rsag",
                    active_blocks=1,
                ),
            )
            name = f"startup-shared-{nbytes}"
            requests.append(
                plan.request(
                    name=name,
                    collective=CollectiveRequirement(name, tuple(range(4))),
                    prepare_call=lambda state, source=source, out=out: (
                        _prepare_plain_call(state, inp=source, out=out)
                    ),
                )
            )
            plans.append(plan)
        session.prepare(
            tuple(requests),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        producers = [
            prepare_joint_startup_tuner(
                (joint_startup_spec(plan, profiles=PROFILES),), cache_policy="cold"
            )
            for plan in plans
        ]
        first = producers[0]
        assert {p.eviction.data_ptr() for p in producers} == {first.eviction.data_ptr()}
        assert {p.shared.local_ptr for p in producers} == {first.shared.local_ptr}
        assert int(torch.count_nonzero(first.eviction)) == 0
        for producer in producers:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                producer.run()
            graphs.append(graph)
        session.freeze()

        def replay(producer, graph):
            producer.status.fill_(-1)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            return producer.status.item()

        allowed = {(JOINT_MODES[mode] << 16) | active for mode, active in PROFILES}
        for round_index in range(3):
            for producer, graph, plan in zip(producers, graphs, plans):
                assert replay(producer, graph) == 0, round_index
                winner = int(plan.prepared.state.selector.item())
                assert winner in allowed
                winners = [None] * 4
                dist.all_gather_object(winners, winner)
                assert len(set(winners)) == 1
        # A wrong collective result keeps the race from publishing a profile.
        producer, graph = producers[1], graphs[1]
        expected = producer.inputs.reference.expected[0]
        saved = expected.clone()
        expected.view(-1)[0] += 1
        assert replay(producer, graph) == -1
        expected.copy_(saved)
        assert replay(producer, graph) == 0
        assert replay(producers[2], graphs[2]) == 0
    finally:
        for graph in graphs:
            graph.reset()
        session.close()
        manager.close()


def _worker(rank, directory):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=240),
    )
    try:
        _exercise(rank)
    finally:
        gc.collect()
        dist.destroy_process_group()
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_CLUSTER_TEST") != "1",
    reason="requires an assigned TP4 SM120/SM121 group",
)
def test_startup_races_share_eviction_and_timing_storage(tmp_path):
    import torch
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 4
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=4, join=True)
