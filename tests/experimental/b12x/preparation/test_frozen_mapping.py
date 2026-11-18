"""Immutable preparation metadata preserves mapping and serialization semantics."""

import pickle
from collections import UserDict
from dataclasses import FrozenInstanceError

import pytest

from b12x.preparation import FrozenMapping
from b12x.preparation.types import _restore_frozen_mapping


class DerivedFrozenMapping(FrozenMapping):
    pass


class CountingHash:
    def __init__(self):
        self.calls = 0

    def __hash__(self):
        self.calls += 1
        return 17


def test_nested_metadata_is_frozen_without_copying_arbitrary_leaves():
    leaf = object()
    nested = {"sequence": [leaf, {"value": 3}]}
    source = {"z": UserDict(nested), "a": (1, [2, 3])}
    frozen = FrozenMapping(source)
    copied = FrozenMapping(frozen)
    nested["sequence"].append(4)
    source["a"] = 5

    for mapping in (frozen, copied):
        assert list(mapping) == ["a", "z"]
        assert mapping["a"] == (1, (2, 3))
        assert isinstance(mapping["z"], FrozenMapping)
        assert mapping["z"]["sequence"][0] is leaf
        assert len(mapping["z"]["sequence"]) == 2
        assert mapping["z"]["sequence"][1]["value"] == 3

    assert copied["z"] is frozen["z"]
    thawed = copied.to_dict()
    assert thawed == {"a": [1, [2, 3]], "z": {"sequence": [leaf, {"value": 3}]}}
    assert thawed["z"]["sequence"][0] is leaf
    thawed["z"]["sequence"][1]["value"] = 7
    assert frozen["z"]["sequence"][1]["value"] == 3


def test_sorted_mapping_operations_and_errors():
    mapping = FrozenMapping({"z": 3, "a": 1, "m": 2})
    assert list(mapping) == ["a", "m", "z"]
    assert list(mapping.items()) == [("a", 1), ("m", 2), ("z", 3)]
    assert list(mapping.values()) == [1, 2, 3]
    assert len(mapping) == 3
    assert "m" in mapping
    assert "missing" not in mapping
    assert [] not in mapping
    with pytest.raises(KeyError):
        mapping["missing"]
    with pytest.raises(TypeError):
        mapping[[]]
    with pytest.raises(TypeError):
        mapping["a"] = 4
    with pytest.raises(FrozenInstanceError):
        mapping.extra = 4
    assert FrozenMapping().to_dict() == {}
    assert FrozenMapping({}) == FrozenMapping()
    assert not FrozenMapping()


def test_equality_requires_the_same_class_and_hashes_sorted_items():
    first = FrozenMapping({"z": [3, 4], "a": {"value": 1}})
    second = FrozenMapping({"a": {"value": 1}, "z": (3, 4)})
    assert first == second
    assert hash(first) == hash(second)
    assert hash(first) == hash((("a", first["a"]), ("z", (3, 4))))
    assert {first: "selected"}[second] == "selected"
    assert first != FrozenMapping({"a": {"value": 2}, "z": [3, 4]})
    assert first != dict(first)
    assert dict(first) != first
    assert first != DerivedFrozenMapping(first)
    assert DerivedFrozenMapping(first) != first
    assert DerivedFrozenMapping(first) == DerivedFrozenMapping(second)


def test_hash_is_cached_and_unhashable_leaves_remain_unhashable():
    leaf = CountingHash()
    mapping = FrozenMapping({"leaf": leaf})
    expected = hash((("leaf", leaf),))
    leaf.calls = 0
    assert hash(mapping) == expected
    assert leaf.calls == 1
    assert hash(mapping) == expected
    assert mapping["leaf"] is leaf
    assert hash(mapping) == expected
    assert leaf.calls == 1

    unhashable = FrozenMapping({"leaf": {1, 2}})
    for _ in range(2):
        with pytest.raises(TypeError):
            hash(unhashable)


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
@pytest.mark.parametrize("access", ["unread", "read", "hashed"])
def test_pickle_round_trip_preserves_nested_metadata(protocol, access):
    mapping = FrozenMapping({"z": 3, "a": {"nested": [1, 2]}})
    if access == "read":
        assert mapping["a"]["nested"] == (1, 2)
    elif access == "hashed":
        hash(mapping)
    restored = pickle.loads(pickle.dumps(mapping, protocol=protocol))
    assert type(restored) is FrozenMapping
    assert restored == mapping
    assert hash(restored) == hash(mapping)
    assert list(restored) == ["a", "z"]
    assert restored.to_dict() == {"a": {"nested": [1, 2]}, "z": 3}


def test_tuple_item_restore_preserves_leaf_identity():
    leaf = object()
    nested = FrozenMapping({"leaf": leaf})
    items = (("a", nested), ("z", (1, 2)))
    restored = _restore_frozen_mapping(items)
    assert restored["a"] is nested
    assert restored["a"]["leaf"] is leaf
    assert restored["z"] == (1, 2)
    assert hash(restored) == hash(items)
    restore, args = restored.__reduce__()
    assert restore is _restore_frozen_mapping
    assert restore(*args) == restored


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle_preserves_shared_metadata_payload(protocol):
    mapping = FrozenMapping(
        {f"metadata_field_{index:04d}": index for index in range(256)}
    )
    copied = FrozenMapping(mapping)
    assert copied is not mapping
    hash(mapping)
    assert copied["metadata_field_0000"] == 0

    single_size = len(pickle.dumps((mapping,), protocol=protocol))
    payload = pickle.dumps((mapping, copied), protocol=protocol)
    assert len(payload) < single_size * 1.1
    first, second = pickle.loads(payload)
    assert first is not second
    assert first == second == mapping
    assert len(pickle.dumps((first, second), protocol=protocol)) < single_size * 1.1


def test_pickle_from_tuple_backed_mapping_remains_readable():
    payload = (
        b"cb12x.preparation.types\n_restore_frozen_mapping\np0\n(((Va\np1\ng0\n"
        b"(((Vnested\np2\n(I1\nI2\ntp3\ntp4\ntp5\ntp6\nRp7\ntp8\n"
        b"(Vz\np9\nI3\ntp10\ntp11\ntp12\nRp13\n."
    )
    mapping = pickle.loads(payload)
    assert type(mapping) is FrozenMapping
    assert list(mapping) == ["a", "z"]
    assert mapping.to_dict() == {"a": {"nested": [1, 2]}, "z": 3}
    expected = FrozenMapping({"z": 3, "a": {"nested": [1, 2]}})
    assert hash(mapping) == hash(expected)
