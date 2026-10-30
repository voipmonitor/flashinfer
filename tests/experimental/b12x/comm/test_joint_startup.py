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
