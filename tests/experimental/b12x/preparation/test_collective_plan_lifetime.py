"""Collective metadata planning does not retain native candidate plans."""

import gc
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from b12x.comm.pcie import _owner_preparation, _preparation
from b12x.comm.pcie._island_rs_cute import HEADER_BYTES, island_rs_peers
from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce
from b12x.preparation import (
    DetectedDevice,
    DeviceIdentity,
    FrozenMapping,
    PreparationSession,
    autotune_effort,
)
from b12x.preparation.types import _plan_scope
from .test_joint_clusters import native_query as joint_query
from .test_static_gather import native_query as gather_query


DEVICE = DetectedDevice(None, DeviceIdentity("nvidia", (12, 0), 188, "SM120"))


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("metadata planning must not initialize or probe CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", refuse)
    monkeypatch.setattr(torch.cuda, "is_available", refuse)


def island_plan():
    owner = object.__new__(PCIeIslandRSAllReduce)
    owner.rank, owner.world_size, owner.device = 0, 16, torch.device("cpu")
    owner.blocks, owner.blocks_override = 64, False
    owner.threads, owner.threads_override, owner.wait_nanosleep_cycles = 512, False, 24
    owner.max_elements = 64 * 7168
    owner.quarter_capacity = ((owner.max_elements // 2 + 3) // 4 + 7) // 8 * 8
    owner.stage_offset = HEADER_BYTES
    owner.part_offset = owner.stage_offset + 2 * owner.quarter_capacity * 16
    owner.final_offset = owner.part_offset + 2 * owner.quarter_capacity * 4
    owner.slab_bytes = owner.final_offset + 2 * owner.quarter_capacity * 16
    owner._mapped_peers = island_rs_peers(0, 16)
    source = torch.empty((64, 7168), dtype=torch.bfloat16)
    output = torch.empty_like(source)
    query = _preparation.query_from_runtime(
        owner,
        surface="PCIeIslandRSAllReduce.all_reduce",
        call={"inp": source, "out": output},
    )
    return _preparation.plan(query, runtime=owner), owner, source, output


def configurations(plan):
    with autotune_effort("default"):
        configured = plan.contract.configure(plan.query, device=DEVICE.identity)
        return [config for _, config in plan.contract.iterate(configured)]


def test_full_candidate_envelope_releases_real_native_plans(monkeypatch):
    children = []
    native = _owner_preparation.plan

    def observe(*args, **kwargs):
        child = native(*args, **kwargs)
        children.append(weakref.ref(child))
        return child

    monkeypatch.setattr(_owner_preparation, "plan", observe)
    plans, requests = [], []
    with autotune_effort("default"):
        for index in range(4):
            plan, owner, source, output = island_plan()
            plans.append(plan)
            requests.append(
                plan.request(
                    name=f"layer{index}.reduce",
                    prepare_call=lambda state, source=source, output=output: (
                        _owner_preparation.prepared_call(state, inp=source, out=output)
                    ),
                )
            )
        expected = len(plans) * len(configurations(plans[0]))
        session = PreparationSession(device=DEVICE, compile_workers=0)
        try:
            envelope = session.candidate_memory_envelope(requests)
            gc.collect()
            assert len(children) == expected
            assert all(child() is None for child in children)
            assert envelope.scratch_nbytes == envelope.pending_persistent_nbytes == 0
            assert len(envelope.persistent) == len(plans)
        finally:
            session.close()


@pytest.mark.parametrize("implementation", ["island_push_rsag", "island_hybrid_rsag"])
def test_compilation_and_materialization_preserve_the_selected_owner(
    monkeypatch, implementation
):
    plan, owner, source, output = island_plan()
    config = next(c for c in configurations(plan) if c.implementation == implementation)
    (job,) = plan._compile_jobs(config, DEVICE)
    payload = job.args[0]
    assert payload["call"]["threads"] == config.threads
    assert payload["call"]["active_blocks"] == config.active_blocks
    assert payload["call"]["local_push"] is True
    assert payload["call"]["cross_island_rsag"] is True
    assert payload["call"]["cross_island_dma"] == (
        implementation == "island_hybrid_rsag"
    )
    assert tuple(payload["call"]["inp"]["shape"]) == tuple(source.shape)
    launcher = SimpleNamespace()
    monkeypatch.setattr(
        _owner_preparation, "compile_owner_surface", lambda *args: launcher
    )
    state = plan._materialize(SimpleNamespace(config=config), DEVICE)
    gc.collect()
    assert state.runtime is owner
    assert state.query.call == payload["call"]
    assert state.launcher() is launcher


@pytest.mark.parametrize("kind", ["gather", "joint"])
def test_profile_memory_keys_follow_binding_and_configuration(kind):
    if kind == "gather":
        from b12x.comm.pcie import _gather_preparation as provider

        query, runtime = gather_query()
        query = _preparation._selected_gather_query(
            query, "selected_phased_barrier", (128, 256, 512)
        )
        role = "pcie-gather-selector"
    else:
        from b12x.comm.pcie import _joint_preparation as provider

        query, runtime = joint_query(), SimpleNamespace(device=torch.device("cpu"))
        role = "pcie-joint-profile"
    first = (
        provider.plan(query, runtime=runtime, invocation=FrozenMapping(), override=None)
        if kind == "gather"
        else provider.plan(query, runtime=runtime)
    )
    second = (
        provider.plan(
            replace(query), runtime=runtime, invocation=FrozenMapping(), override=None
        )
        if kind == "gather"
        else provider.plan(replace(query), runtime=runtime)
    )
    config = first.contract.default_config(query, DEVICE.identity)
    other_config = replace(config, backend="other")

    def key(child, binding, selected):
        with _plan_scope(binding):
            memory = child._memory_requirements(selected, DEVICE)
        return next(item.key for item in memory.persistent if item.key[0] == role)

    assert key(first, first, config) == key(second, first, config)
    assert key(first, first, config) != key(second, second, config)
    assert key(first, first, config) != key(first, first, other_config)
