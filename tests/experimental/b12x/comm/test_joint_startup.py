"""Prepared mode retention uses the established finalist cache contract."""

from b12x.comm.pcie import joint_finalist_policy, joint_startup_spec
import pytest


def test_joint_finalist_policy_preserves_mode_and_compiled_geometry():
    policy = joint_finalist_policy()
    assert policy.family_fields == ("implementation", "threads", "blocks", "submode")
    assert policy.per_family == 1
    assert policy.max_total == 35


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
        (("bulk_pull", 8), ("bulk_pull", 8)),
        ("bulk_pull",),
        (("bulk_pull",),),
    ],
)
def test_invalid_explicit_runtime_profiles(prepared_joint, profiles):
    with pytest.raises(ValueError):
        joint_startup_spec(prepared_joint, profiles=profiles)


@pytest.mark.parametrize("kind", ("reduction", "gather"))
def test_startup_references_of_fewer_rows_are_leading_rows(kind):
    import torch
    from b12x.comm.pcie._joint_startup_fixtures import _gather_bank, _reduction_bank

    device = torch.device("cpu")
    if kind == "reduction":
        small, large = (
            _reduction_bank(4, 1, rows, (16,), torch.bfloat16, device) for rows in (3, 8)
        )
        leading = lambda tensor, rows: tensor[:rows]
    else:
        small, large = (_gather_bank(4, 2, rows, 24, device) for rows in (3, 8))
        leading = lambda tensor, rows: tensor[: rows * tensor.numel() // 8]
    for case in range(3):
        assert torch.equal(small.seeds[case], leading(large.seeds[case], 3))
        assert torch.equal(small.expected[case], leading(large.expected[case], 3))
    assert large.seeds[0].abs().sum() > 0
