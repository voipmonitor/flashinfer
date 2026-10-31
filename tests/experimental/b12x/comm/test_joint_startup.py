"""Prepared mode retention uses the established finalist cache contract."""

from b12x.comm.pcie import joint_finalist_policy, joint_startup_spec
import pytest


def test_joint_finalist_policy_preserves_mode_and_compiled_geometry():
    policy = joint_finalist_policy()
    assert policy.family_fields == ("implementation", "threads", "blocks", "submode")
    assert policy.per_family == 1
    assert policy.max_total == 55


def test_unprepared_joint_startup_rejected():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="prepared joint"):
        joint_startup_spec(SimpleNamespace(prepared=None))


@pytest.fixture
def prepared_joint(monkeypatch):
    from types import SimpleNamespace

    import torch
    from b12x.comm.pcie._joint_preparation import _JointExecutionState

    state = _JointExecutionState.__new__(_JointExecutionState)
    state.query = SimpleNamespace(
        call=dict(
            transport="joint_bandwidth",
            threads=512,
            blocks=64,
            dtype="bfloat16",
            size_packs=65536,
        ),
        setup=dict(shape=(512, 1024)),
        world_size=4,
        rank=0,
    )
    state.workspace = SimpleNamespace(capacity_packs=65536)
    state.selector = torch.zeros(1, dtype=torch.uint32)
    state.runtime = SimpleNamespace(exchange_group=object())
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3]
    )
    return SimpleNamespace(
        prepared=SimpleNamespace(state=state, finalists=()),
        query=SimpleNamespace(inplace=False),
    )


def test_explicit_runtime_profiles_need_no_measured_finalists(prepared_joint):
    spec = joint_startup_spec(
        prepared_joint,
        profiles=(("bulk_push", 32), ("bulk_push", 64), ("bulk_pull", 8)),
    )
    assert spec.profiles == ((2, 32), (2, 64), (3, 8))
    assert spec.selector is prepared_joint.prepared.state.selector
    assert spec.state is prepared_joint.prepared.state
    other = joint_startup_spec(prepared_joint, profiles=(("bulk_push", 32),))
    assert spec.key != other.key
    with pytest.raises(ValueError, match="measured finalists"):
        joint_startup_spec(prepared_joint)


@pytest.mark.parametrize(
    "profiles",
    [
        (),
        (("bulk_push", 0),),
        (("bulk_push", 65),),
        (("bulk_push", True),),
        (("full", 1),),
        (("bulk_push_phased", 2),),
        (("payload_rsag_phased", 64),),
        (("bulk_pull", 8), ("bulk_pull", 8)),
        ("bulk_pull",),
        (("bulk_pull",),),
    ],
)
def test_invalid_explicit_runtime_profiles(prepared_joint, profiles):
    with pytest.raises(ValueError):
        joint_startup_spec(prepared_joint, profiles=profiles)


def test_startup_choice_labels_are_defined_by_protocol_metadata(prepared_joint):
    spec = joint_startup_spec(
        prepared_joint, profiles=(("bulk_push_phased", 1), ("bulk_pull", 8))
    )
    assert spec.choice_labels() == (
        ((7 << 16) | 1, "Phased bulk push (1 CTA)"),
        ((3 << 16) | 8, "Bulk pull (8 CTAs)"),
    )


def test_startup_profiles_reject_phased_tp2(prepared_joint):
    prepared_joint.prepared.state.query.world_size = 2
    with pytest.raises(ValueError, match="rank group"):
        joint_startup_spec(prepared_joint, profiles=(("bulk_push_phased", 1),))


def test_variant_selectors_and_launch_predicates(prepared_joint, monkeypatch):
    from b12x.comm.pcie import JointLaunchVariant
    from b12x.comm.pcie import _joint_functions

    state = prepared_joint.prepared.state
    state.variants = (
        JointLaunchVariant(
            "joint_bandwidth", 512, 64, (("bulk_push", 16), ("bulk_pull", 32))
        ),
        JointLaunchVariant(
            "joint_latency", 256, 8, (("full_phased", 8),), cooperative=True
        ),
    )
    state.variant_launchers = (object(), object())
    monkeypatch.setattr(_joint_functions, "joint_function", id)
    spec = joint_startup_spec(prepared_joint)
    expected = ((2 << 16) | 16, (3 << 16) | 32, (1 << 24) | (5 << 16) | 8)
    assert spec.selector_values == expected
    assert spec.candidate_launcher(0) is state.variant_launchers[0]
    assert spec.candidate_launcher(1) is state.variant_launchers[0]
    assert spec.candidate_launcher(2) is state.variant_launchers[1]
    predicates = spec.launch_predicates()
    assert len(predicates) == 2
    domain = tuple(value.to_bytes(4, "little") for value in expected)
    for index, predicate in enumerate(predicates):
        assert predicate.function == id(state.variant_launchers[index])
        assert predicate.selector_argument_index == 3
        assert predicate.selector_argument_byte_offset == 0
        assert predicate.selector is state.selector
        assert predicate.allowed_values == domain
    assert predicates[0].enabled_values == domain[:2]
    assert predicates[1].enabled_values == domain[2:]
    labels = dict(spec.choice_labels())
    assert "joint_latency" in labels[expected[2]]
    assert (
        "256 threads, 8 physical CTAs, 8 active CTAs; cooperative"
        in labels[expected[2]]
    )
    assert "ordinary" in labels[expected[0]]
    with pytest.raises(ValueError, match="already declare"):
        joint_startup_spec(prepared_joint, profiles=(("bulk_push", 16),))


@pytest.mark.parametrize(
    "cooperative,mode,physical,active",
    [
        (True, "bulk_push", 8, 8),
        (True, "bulk_push_phased", 8, 4),
        (False, "bulk_push_phased", 8, 8),
        (True, "bulk_push_phased", 65, 65),
        (1, "bulk_push_phased", 8, 8),
    ],
)
def test_invalid_launch_variant(cooperative, mode, physical, active):
    from b12x.comm.pcie import JointLaunchVariant

    with pytest.raises(ValueError):
        JointLaunchVariant(
            "joint_bandwidth", 512, physical, ((mode, active),), cooperative
        )


def test_variant_validation_rejects_duplicate_geometry_and_tp2_phases():
    from b12x.comm.pcie.joint_profiles import (
        JointLaunchVariant,
        validate_joint_variants,
    )

    variant = JointLaunchVariant(
        "joint_bandwidth", 512, 8, (("bulk_push_phased", 8),), True
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_joint_variants((variant, variant), world=4)
    with pytest.raises(ValueError, match="rank group"):
        validate_joint_variants((variant,), world=2)


def test_variant_catalog_participates_in_compile_plan():
    from dataclasses import asdict
    from types import SimpleNamespace
    from b12x.comm.pcie._joint_preparation import plan
    from b12x.comm.pcie.joint_profiles import JointLaunchVariant
    from b12x.comm.pcie._tuning import PcieQuery
    from b12x.preparation import FrozenMapping

    variants = (
        JointLaunchVariant("joint_bandwidth", 512, 64, (("bulk_push", 4),)),
        JointLaunchVariant("joint_bandwidth", 512, 4, (("bulk_push_phased", 4),), True),
    )
    query = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=4,
        rank=1,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                transport="joint_bandwidth",
                threads=512,
                blocks=64,
                submode="bulk_push",
                active_blocks=4,
                dtype="bfloat16",
                stage_input=True,
                joint_variants=tuple(asdict(v) for v in variants),
            )
        ),
        setup=FrozenMapping(
            dict(
                joint_max_blocks=64,
                rank_data_nbytes=16,
                owned_slab_nbytes=16,
                owned_slab_count=1,
                joint_workspace_nbytes=4096,
            )
        ),
    )
    declaration = plan(query, runtime=SimpleNamespace(device="cuda:1"))
    jobs = declaration._compile_jobs(None, None)
    assert len(jobs) == 2
    assert jobs[0] != jobs[1]


@pytest.mark.parametrize("cooperative_first", (False, True))
def test_declared_variants_admit_counts_outside_static_tuning_axes(cooperative_first):
    from types import SimpleNamespace
    from b12x.comm.pcie import JointLaunchVariant
    from b12x.comm.pcie._collective_tuning import plan, _configs
    from b12x.comm.pcie._tuning import PcieQuery
    from b12x.preparation import FrozenMapping

    ordinary = JointLaunchVariant("joint_latency", 256, 64, (("full", 25), ("pull", 7)))
    cooperative = JointLaunchVariant(
        "joint_latency", 256, 37, (("full_phased", 37),), True
    )
    variants = (cooperative, ordinary) if cooperative_first else (ordinary, cooperative)
    query = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                transport="joint_latency",
                threads=256,
                blocks=64,
                submode="full",
                active_blocks=16,
                size_packs=8192,
                dtype="bfloat16",
                stage_input=True,
            )
        ),
        setup=FrozenMapping(
            dict(
                joint_max_blocks=64,
                rank_data_nbytes=16,
                owned_slab_nbytes=16,
                owned_slab_count=1,
                joint_workspace_nbytes=4096,
            )
        ),
    )
    declaration = plan(
        {"joint_latency": (query, SimpleNamespace(device="cuda:0"))},
        default="joint_latency",
        joint_variants=variants,
    )
    choices = tuple(_configs(declaration.query))
    assert len(choices) == 1
    expected = variants[0]
    assert choices[0].active_blocks == expected.profiles[0][1]
    assert choices[0].blocks == expected.blocks
    assert choices[0].cooperative == expected.cooperative
    declaration.contract.validate_config(declaration.query, choices[0], None)
    jobs = declaration._compile_jobs(choices[0], SimpleNamespace(ordinal=0))
    assert len(jobs) == 2


def test_single_launch_startup_keys_distinguish_cooperative_policy(prepared_joint):
    state = prepared_joint.prepared.state
    state.query.call.update(transport="joint_latency", threads=256, blocks=1)
    profiles = (("full_phased", 1), ("payload_rsag_phased", 1))
    ordinary = joint_startup_spec(prepared_joint, profiles=profiles)
    state.query.call["cooperative"] = True
    cooperative = joint_startup_spec(prepared_joint, profiles=profiles)
    assert ordinary.profiles == cooperative.profiles
    assert ordinary.key != cooperative.key


def test_typed_variant_startup_key_retains_its_complete_identity(prepared_joint):
    from b12x.comm.pcie import JointLaunchVariant

    state = prepared_joint.prepared.state
    state.variants = (
        JointLaunchVariant("joint_bandwidth", 512, 64, (("bulk_pull", 7),)),
        JointLaunchVariant("joint_latency", 256, 37, (("full_phased", 37),), True),
    )
    spec = joint_startup_spec(prepared_joint)
    assert spec.key == (
        "joint_startup",
        2,
        4,
        (0, 1, 2, 3),
        "joint_bandwidth",
        512,
        64,
        (512, 1024),
        "bfloat16",
        65536,
        65536,
        False,
        ((3, 7), (5, 37)),
        state.variants,
    )


@pytest.mark.parametrize("variant_id", (0, 1, 255))
def test_single_launch_profile_rejects_variant_identity(
    prepared_joint, monkeypatch, variant_id
):
    from b12x.comm.pcie import _joint_preparation

    state = prepared_joint.prepared.state
    state.variants = ()
    state.runtime._closed = False
    state.runtime.device = "cuda:0"
    monkeypatch.setattr(
        _joint_preparation, "_is_current_stream_capturing", lambda device: False
    )
    with pytest.raises(ValueError, match="standalone.*variant identity"):
        state.select_profile("pull", 7, variant_id=variant_id)


def test_typed_variant_profile_selection_preserves_valid_identities(
    prepared_joint, monkeypatch
):
    import torch
    from b12x.comm.pcie import JointLaunchVariant, _joint_preparation

    state = prepared_joint.prepared.state
    state.variants = (
        JointLaunchVariant("joint_bandwidth", 512, 64, (("bulk_pull", 7),)),
        JointLaunchVariant("joint_latency", 256, 37, (("full_phased", 37),), True),
    )
    state.runtime._closed = False
    state.runtime.device = "cuda:0"
    monkeypatch.setattr(
        _joint_preparation, "_is_current_stream_capturing", lambda device: False
    )
    monkeypatch.setattr(
        _joint_preparation,
        "_run_collective_preallocation_setup",
        lambda **kwargs: kwargs["setup"](),
    )
    monkeypatch.setattr(
        _joint_preparation, "_require_collective_contract", lambda **kwargs: None
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    state.select_profile("bulk_pull", 7, variant_id=0)
    assert state.selector.item() == (3 << 16) | 7
    state.select_profile("full_phased", 37, variant_id=1)
    assert state.selector.item() == (1 << 24) | (5 << 16) | 37
