"""Delta-prefill memory queries release candidate layout metadata."""

import gc
import weakref

import pytest
import torch

from b12x.preparation import DetectedDevice, DeviceIdentity, MemoryRequirements
from b12x.sequence import gdn_prefill, kda_prefill
from b12x.sequence.gdn_prefill import _impl as gdn_impl
from b12x.sequence.kda_prefill import _impl as kda_impl


DEVICE = DetectedDevice(0, DeviceIdentity("nvidia", (12, 0), 188, "SM120"))


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("prefill memory queries must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", refuse)


@pytest.mark.parametrize(
    "recipe,algorithm,intermediate_dtype",
    (
        ("kda", "sequential", torch.bfloat16),
        ("gdn", "sequential", torch.bfloat16),
        ("gdn", "chunk_parallel", torch.bfloat16),
        ("gdn", "sequential", torch.float32),
    ),
    ids=("kda-bf16", "gdn-sequential-bf16", "gdn-parallel-bf16", "gdn-fp32"),
)
def test_memory_queries_release_real_candidate_layouts(
    monkeypatch, recipe, algorithm, intermediate_dtype
):
    geometry = dict(device="cuda:0", max_tokens=4096, max_seqs=1, max_state_slots=4)
    if recipe == "kda":
        caps = kda_prefill.Caps(**geometry, heads=6)
        plan, impl = kda_prefill.plan(caps), kda_impl
    else:
        caps = gdn_prefill.Caps(
            **geometry,
            key_heads=2,
            value_heads=6,
            intermediate_dtype=intermediate_dtype,
            staging_key="gdn-staging",
        )
        plan, impl = gdn_prefill.plan(caps), gdn_impl
    configuration = plan.contract.configure(plan.query, device=DEVICE.identity)
    configs = tuple(
        config
        for _, config in plan.contract.iterate(configuration)
        if getattr(config, "algorithm", "sequential") == algorithm
    )
    assert configs
    native = impl._materialize_layout
    expected = tuple(
        MemoryRequirements(
            scratch=native(caps, config).scratch_specs(),
            persistent=impl.staging_memory(caps).persistent,
        )
        for config in configs
    )
    layouts = []

    def observe(*args, **kwargs):
        layout = native(*args, **kwargs)
        layouts.append(weakref.ref(layout))
        if getattr(layout, "parallel", None) is not None:
            layouts.append(weakref.ref(layout.parallel.segment_plan))
        return layout

    monkeypatch.setattr(impl, "_materialize_layout", observe)
    for _ in range(2):
        assert (
            tuple(plan._memory_requirements(config, DEVICE) for config in configs)
            == expected
        )
        gc.collect()
        assert layouts
        assert all(layout() is None for layout in layouts)
