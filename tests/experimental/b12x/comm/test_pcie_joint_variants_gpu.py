"""Guarded prepared variants share one graph and advance one collective epoch."""

from datetime import timedelta
import ctypes
import os
from pathlib import Path

import pytest

from .test_pcie_joint_gpu import _operands, _reference, _phase_completions


def _epoch(state):
    import torch
    from b12x.comm.pcie._joint_profile import JOINT_EPOCH_OFFSET

    word = ctypes.c_uint32()
    stream = torch.cuda.current_stream()
    state.runtime._ipc.cudaMemcpyAsync(
        ctypes.addressof(word),
        state.workspace.shared.local_ptr + JOINT_EPOCH_OFFSET,
        4,
        stream.cuda_stream,
    )
    stream.synchronize()
    return word.value


def _worker(rank, directory, alias):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import (
        AllReduce,
        CollectiveConfig,
        JointLaunchVariant,
        joint_startup_spec,
        prepare_joint_startup_tuner,
    )
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.joint_profiles import joint_mode_spec, JOINT_MODE_SPECS
    from b12x.preparation import CollectiveRequirement, PreparationSession

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=600),
    )
    device = torch.device("cuda", rank)
    size = 2621440
    source = torch.empty((size // 256, 128), dtype=torch.bfloat16, device=device)
    out = source if alias else torch.empty_like(source)
    variants = (
        JointLaunchVariant(
            "joint_bandwidth",
            512,
            64,
            (("bulk_push", 4), ("bulk_pull", 64), ("bulk_push_phased", 1)),
        ),
        JointLaunchVariant(
            "joint_bandwidth",
            512,
            4,
            (("bulk_push_phased", 4), ("payload_rsag_phased", 4)),
            True,
        ),
        JointLaunchVariant("joint_latency", 256, 64, (("full", 8), ("full_phased", 1))),
        JointLaunchVariant("joint_latency", 256, 8, (("full_phased", 8),), True),
    )
    manager = AllReduce.from_exchange_group(
        joint=True,
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=size,
        eager_buffer_bytes=size,
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    graph = startup_graph = None
    try:
        plan = manager.plan(source, out=out, joint_only=True, joint_variants=variants)
        controls = {
            family: manager.plan(
                source,
                out=out,
                joint_only=True,
                override=CollectiveConfig(
                    implementation=family,
                    threads=threads,
                    blocks=64,
                    submode=mode,
                    active_blocks=8,
                ),
            )
            for family, threads, mode in (
                ("joint_bandwidth", 512, "bulk_push"),
                ("joint_latency", 256, "full"),
            )
        }
        session.prepare(
            tuple(
                candidate.request(
                    name=name,
                    collective=CollectiveRequirement(name, (0, 1, 2, 3)),
                    prepare_call=lambda state: _prepare_plain_call(
                        state, inp=source, out=out
                    ),
                )
                for name, candidate in (("variants", plan), *controls.items())
            ),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        state = plan.prepared.state
        spec = joint_startup_spec(plan)
        predicates = spec.launch_predicates()
        assert len(predicates) == len(variants)
        assert len({p.function for p in predicates}) == len(variants)
        assert all(p.allowed_values == predicates[0].allowed_values for p in predicates)
        tuner = prepare_joint_startup_tuner((spec,), cache_policy="warm")
        graph, startup_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with session.capture(), manager.capture(), torch.cuda.graph(graph):
            state.run_plain(source, out)
        with session.capture(), manager.capture(), torch.cuda.graph(startup_graph):
            tuner.run()
        session.freeze()
        for step in range(6):
            operands = _operands(4, source.numel(), step)
            for variant_id, variant in enumerate(variants):
                for mode, active in variant.profiles:
                    metadata = joint_mode_spec(mode)
                    base = next(
                        s.name for s in JOINT_MODE_SPECS if s.mode == metadata.base
                    )
                    control = controls[variant.family].prepared.state
                    control.select_profile(base, active)
                    source.copy_(operands[rank].reshape_as(source))
                    control.run_plain(source, out)
                    torch.cuda.synchronize()
                    expected_bits = out.cpu().view(torch.int16)
                    expected = _reference(operands, rank, base).reshape_as(source)
                    state.select_profile(mode, active, variant_id=variant_id)
                    epoch = _epoch(state)
                    phases = _phase_completions(state)
                    for replay in range(3):
                        source.copy_(operands[rank].reshape_as(source))
                        if not alias:
                            out.fill_(float("nan"))
                        before = torch.cuda.memory_stats()
                        graph.replay()
                        torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        for counter in (
                            "allocation.all.allocated",
                            "allocation.all.freed",
                            "segment.all.allocated",
                            "segment.all.freed",
                        ):
                            assert before[counter] == after[counter]
                        actual = out.cpu()
                        torch.testing.assert_close(
                            actual, expected, rtol=0, atol=0, equal_nan=True
                        )
                        assert torch.equal(actual.view(torch.int16), expected_bits), (
                            variant_id,
                            mode,
                            step,
                        )
                        assert _epoch(state) == (epoch + replay + 1) & 0xFFFFFFFF
                        increments = (
                            3 * (1 if base == "full" else 2) if metadata.phased else 0
                        )
                        assert (
                            _phase_completions(state)
                            == (phases + increments * (replay + 1)) & 0xFFFFFFFF
                        )
        startup_graph.replay()
        torch.cuda.synchronize()
        assert tuner.status.item() == 0
        selected = state.selector.item()
        assert selected in spec.selector_values
        values = [None] * 4
        dist.all_gather_object(values, selected)
        assert len(set(values)) == 1
        operands = _operands(4, source.numel(), 1)
        source.copy_(operands[rank].reshape_as(source))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(out).item() == 0
    finally:
        for captured in (graph, startup_graph):
            if captured is not None:
                captured.reset()
        session.close()
        manager.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_PCIE_TEST") != "1",
    reason="requires an assigned peer-accessible four-GPU group",
)
@pytest.mark.parametrize("alias", (False, True))
def test_prepared_joint_launch_variants(alias, tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 4:
        pytest.skip("requires four assigned GPUs")
    mp.spawn(_worker, args=(str(tmp_path), alias), nprocs=4, join=True)
