"""Parallel peer signaling stays within its measured geometry and hardware band."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie._owner_preparation import _parallel_island_peer_flags


def query(shape=(4, 3584), threads=512, **changes):
    call = dict(
        blocks=64,
        active_blocks=4,
        wait_nanosleep_cycles=24,
        local_push=True,
        cross_island_rsag=False,
        threads=threads,
        inp=dict(shape=shape, dtype="torch.bfloat16"),
    )
    call.update(changes)
    return SimpleNamespace(world_size=16, call=call)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("policy must not initialize CUDA")
    )


@pytest.mark.parametrize(
    "shape,threads",
    [((4, 3584), 512), ((1, 14336), 512), ((4, 7168), 1024), ((8, 3584), 1024)],
)
def test_flat_qualified_geometries(shape, threads):
    assert _parallel_island_peer_flags(query(shape, threads), (12, 0), 188)


@pytest.mark.parametrize(
    "changes",
    [
        dict(active_blocks=8),
        dict(blocks=32),
        dict(local_push=False),
        dict(cross_island_rsag=True),
        dict(wait_nanosleep_cycles=0),
        dict(threads=256),
        dict(inp=dict(shape=(4, 3584), dtype="torch.float16")),
        dict(inp=dict(shape=(7, 3584), dtype="torch.bfloat16")),
        dict(inp=dict(shape=(512, 7168), dtype="torch.bfloat16")),
        dict(inp={}),
    ],
)
def test_unmeasured_geometry_retains_serial_signaling(changes):
    assert not _parallel_island_peer_flags(query(**changes), (12, 0), 188)


@pytest.mark.parametrize(
    "capability,sm_count,world",
    [((12, 1), 188, 16), ((10, 3), 188, 16), ((12, 0), 170, 16), ((12, 0), 188, 8)],
)
def test_other_hardware_retains_serial_signaling(capability, sm_count, world):
    selected = query()
    selected.world_size = world
    assert not _parallel_island_peer_flags(selected, capability, sm_count)


def test_flag_strategy_changes_compile_identity(monkeypatch):
    from b12x.comm.pcie import _island_rs_cute as serial, _island_parallel_push as parallel
    from b12x._lib.compile_plan import ProgramKey

    observed = []
    def compile(target, *args, **kwargs):
        observed.append((type(target), kwargs["compile_spec"]))
        return SimpleNamespace(__b12x_programs__=(ProgramKey("cute", str(len(observed))),))

    for module in (serial, parallel):
        monkeypatch.setattr(module, "b12x_compile", compile)
        monkeypatch.setattr(module, "make_ptr", lambda *args, **kwargs: 0)
        monkeypatch.setattr(module, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(serial, "raise_if_kernel_resolution_frozen", lambda *args, **kwargs: None)
    serial.get_island_rs_launcher.__wrapped__(16, 0, 0, threads=512, local_push=True)
    parallel.compile_parallel_push.__wrapped__(0, 512)
    assert observed[0][0] is serial._IslandRSLaunch
    assert observed[1][0] is parallel.ParallelPushIslandRS
    assert observed[0][1] != observed[1][1]
