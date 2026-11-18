"""Prefix kernel changes preserve cache identities of existing transfers."""

import shutil

from b12x._lib import compiler
from b12x._lib.source_dependencies import SourceDependencies, callable_sources
from b12x.comm.pcie._dcp_a2a_cute import _AllGatherHeadsLaunch, _LseReduceScatterLaunch
from b12x.comm.pcie._kimi_topk_cute import WarpTopK16Launch
from b12x.comm.pcie._prefix_gather_cute import PrefixGather
from b12x.comm.pcie._recursive_gather_cute import RecursiveGather


def test_prefix_source_edit_only_invalidates_prefix_programs(tmp_path):
    package = compiler._PACKAGE_ROOT
    targets = dict(prefix=PrefixGather, recursive=RecursiveGather, heads=_AllGatherHeadsLaunch,
                   lse=_LseReduceScatterLaunch, expert_selector=WarpTopK16Launch)
    roots = {name: callable_sources(target, package) for name, target in targets.items()}
    discovery = SourceDependencies(package)
    files = {path for group in roots.values() for name, root in group
             for _, path in discovery.closure(name, root)}
    copied = tmp_path / 'b12x'
    for path in files:
        destination = copied / path.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    local = {name: tuple((module, copied / path.relative_to(package)) for module, path in group)
             for name, group in roots.items()}
    before = SourceDependencies(copied)
    expected = {name: before.fingerprint(group) for name, group in local.items()}
    with (copied / 'comm/pcie/_prefix_gather_cute.py').open('a') as file:
        file.write('\n# Dependency invalidation probe.\n')
    after = SourceDependencies(copied)
    actual = {name: after.fingerprint(group) for name, group in local.items()}
    assert actual['prefix'] != expected['prefix']
    for name in targets.keys() - {'prefix'}:
        assert actual[name] == expected[name], name
