"""Selector edits invalidate its programs without invalidating unrelated transfers."""

import shutil

from b12x._lib import compiler
from b12x._lib.source_dependencies import SourceDependencies, callable_sources
from b12x.comm.pcie._dcp_a2a_cute import (
    _AllGatherHeadsLaunch,
    _AllGatherPairLaunch,
    _LseReduceScatterLaunch,
)
from b12x.comm.pcie._kimi_topk_cute import WarpTopK16Launch
from b12x.comm.pcie._recursive_gather_cute import RecursiveGather


def test_selector_source_change_preserves_transfer_fingerprints(tmp_path):
    package = compiler._PACKAGE_ROOT
    targets = {
        "selector": WarpTopK16Launch,
        "head_gather": _AllGatherHeadsLaunch,
        "pair_gather": _AllGatherPairLaunch,
        "lse_reduce": _LseReduceScatterLaunch,
        "recursive_gather": RecursiveGather,
    }
    common = (
        ("b12x._lib.compiler", compiler._COMPILER_SOURCE),
        ("b12x._lib.runtime_patches", package / "_lib/runtime_patches.py"),
    )
    roots = {
        name: (*callable_sources(target, package), *common)
        for name, target in targets.items()
    }
    discovery = SourceDependencies(package)
    files = {
        path
        for group in roots.values()
        for name, root in group
        for _, path in discovery.closure(name, root)
    }
    copied = tmp_path / "b12x"
    for path in files:
        destination = copied / path.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    local_roots = {
        name: tuple(
            (module, copied / path.relative_to(package)) for module, path in group
        )
        for name, group in roots.items()
    }
    before = SourceDependencies(copied)
    expected = {name: before.fingerprint(group) for name, group in local_roots.items()}
    selector = copied / "comm/pcie/_kimi_topk_cute.py"
    with selector.open("a") as file:
        file.write("\n# Source-fingerprint regression probe.\n")
    after = SourceDependencies(copied)
    actual = {name: after.fingerprint(group) for name, group in local_roots.items()}
    assert actual["selector"] != expected["selector"]
    for name in targets.keys() - {"selector"}:
        assert actual[name] == expected[name], name
