"""Exact captured selector-address decoding for the pruning benchmark."""

import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _benchmark():
    path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/experimental/b12x/benchmark_joint_variant_pruning.py"
    )
    spec = importlib.util.spec_from_file_location("joint_variant_pruning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_address_comes_from_captured_argument_storage():
    benchmark = _benchmark()
    selector_argument = (ctypes.c_uint64 * 2)(0xA000, 0xB000)
    parameters = (ctypes.c_void_p * 4)(0, 0, 0, ctypes.addressof(selector_argument))
    params = SimpleNamespace(kernelParams=ctypes.addressof(parameters), extra=0)
    predicate = SimpleNamespace(
        selector_argument_index=3, selector_argument_byte_offset=0
    )
    assert benchmark._selector_pointer(params, predicate) == 0xA000
    predicate.selector_argument_byte_offset = 8
    assert benchmark._selector_pointer(params, predicate) == 0xB000


@pytest.mark.parametrize(
    "params",
    [
        SimpleNamespace(kernelParams=0, extra=0),
        SimpleNamespace(kernelParams=1, extra=1),
    ],
)
def test_missing_or_packed_parameter_arrays_fail_closed(params):
    with pytest.raises(RuntimeError, match="parameter array"):
        _benchmark()._selector_pointer(params, SimpleNamespace())


def test_source_pruning_preserves_fork_and_join_dependencies(monkeypatch):
    from cuda.bindings import driver

    benchmark = _benchmark()
    success = driver.CUresult.CUDA_SUCCESS
    kernel = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
    empty = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY
    nodes = {1: empty, 2: kernel, 3: kernel, 4: kernel, 5: empty}
    edges = {(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)}
    annotation = SimpleNamespace(from_port=0, to_port=0, type=0, reserved=b"\0" * 5)

    def get_edges(graph, count=0):
        pairs = sorted(edges)
        return (
            success,
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            [annotation] * len(pairs),
            len(pairs),
        )

    def adjacent(node, count=0, *, incoming):
        values = sorted(
            a if incoming else b
            for a, b in edges
            if (b if incoming else a) == int(node)
        )
        return success, values, [annotation] * len(values), len(values)

    def add_empty(graph, dependencies, count):
        node = max(nodes) + 100
        nodes[node] = empty
        edges.update((int(dependency), node) for dependency in dependencies)
        return success, node

    def add_dependencies(graph, sources, targets, annotations, count):
        edges.update((int(a), int(b)) for a, b in zip(sources, targets, strict=True))
        return (success,)

    def destroy_node(node):
        del nodes[int(node)]
        edges.difference_update({edge for edge in edges if int(node) in edge})
        return (success,)

    monkeypatch.setattr(driver, "cuGraphClone", lambda graph: (success, 99))
    monkeypatch.setattr(driver, "cuGraphGetEdges", get_edges)
    monkeypatch.setattr(
        driver,
        "cuGraphGetNodes",
        lambda graph, count=0: (success, list(nodes), len(nodes)),
    )
    monkeypatch.setattr(
        driver, "cuGraphNodeGetType", lambda node: (success, nodes[int(node)])
    )
    monkeypatch.setattr(
        driver,
        "cuGraphKernelNodeGetParams",
        lambda node: (
            success,
            SimpleNamespace(
                func=int(node) * 10,
                gridDimX=1,
                gridDimY=1,
                gridDimZ=1,
                blockDimX=32,
                blockDimY=1,
                blockDimZ=1,
            ),
        ),
    )
    monkeypatch.setattr(
        driver,
        "cuGraphKernelNodeGetAttribute",
        lambda node, kind: (success, SimpleNamespace(cooperative=int(node) == 3)),
    )
    monkeypatch.setattr(
        driver,
        "cuGraphNodeGetDependencies",
        lambda node, count=0: adjacent(node, count, incoming=True),
    )
    monkeypatch.setattr(
        driver,
        "cuGraphNodeGetDependentNodes",
        lambda node, count=0: adjacent(node, count, incoming=False),
    )
    monkeypatch.setattr(driver, "cuGraphAddEmptyNode", add_empty)
    monkeypatch.setattr(driver, "cuGraphAddDependencies", add_dependencies)
    monkeypatch.setattr(driver, "cuGraphDestroyNode", destroy_node)
    monkeypatch.setattr(
        driver, "cuGraphInstantiate", lambda graph, flags: (success, 1000)
    )
    monkeypatch.setattr(
        benchmark, "_selector_pointer", lambda params, predicate: 0xA000
    )
    selector = SimpleNamespace(data_ptr=lambda: 0xA000)
    selected = (1).to_bytes(4, "little")
    spec = SimpleNamespace(
        variants=tuple(
            SimpleNamespace(blocks=1, threads=32, cooperative=index == 1)
            for index in range(3)
        ),
        launch_predicates=lambda: tuple(
            SimpleNamespace(
                function=node * 10,
                selector=selector,
                allowed_values=(selected, (2).to_bytes(4, "little")),
                enabled_values=(selected,)
                if node == 2
                else ((2).to_bytes(4, "little"),),
            )
            for node in (2, 3, 4)
        ),
    )
    source = SimpleNamespace(raw_cuda_graph=lambda: 77)
    graph, inspection = benchmark._prune(source, spec, 1, 1)
    assert inspection["dependency_graph_preserved"]
    assert inspection["empty_nodes"] == 2
    assert inspection["enabled_nodes"] == 1
    assert graph.owners == (source, spec)
    assert nodes[2] == kernel and 3 not in nodes and 4 not in nodes
    assert edges == {(1, 2), (1, 105), (2, 205), (105, 205), (205, 5)}
