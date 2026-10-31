"""Measure standalone, guarded, and once-pruned prepared collective graphs.

Run under torchrun on four explicitly assigned peer-accessible GPUs. Every
timed chain follows one untimed invocation of the same graph path. External
CUDA events bracket only the chain; samples retain every rank's duration and
use the maximum rank duration for each matching replay. Inputs and outputs are
disjoint, and the timed graphs contain complete collective invocations.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flashinfer/experimental/b12x/_compat"))


def _checked(result):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA graph inspection failed: {result[0]}")
    return result[1:]


def _selector_pointer(params, predicate):
    if not params.kernelParams or params.extra:
        raise RuntimeError("joint graph kernel must expose its parameter array")
    parameters = ctypes.cast(int(params.kernelParams), ctypes.POINTER(ctypes.c_void_p))
    argument = parameters[predicate.selector_argument_index]
    if not argument:
        raise RuntimeError("joint selector argument is null")
    return ctypes.c_uint64.from_address(
        argument + predicate.selector_argument_byte_offset
    ).value


def _edges(graph):
    from cuda.bindings import driver

    count = _checked(driver.cuGraphGetEdges(graph))[-1]
    if not count:
        return set()
    sources, targets, annotations, _ = _checked(driver.cuGraphGetEdges(graph, count))
    _require_full_dependencies(annotations)
    return {
        (int(source), int(target))
        for source, target in zip(sources, targets, strict=True)
    }


def _require_full_dependencies(annotations):
    if any(
        edge.from_port or edge.to_port or edge.type or any(edge.reserved)
        for edge in annotations
    ):
        raise RuntimeError("source pruning requires full-completion dependency edges")


def _adjacent(query, node):
    count = _checked(query(node))[-1]
    if not count:
        return []
    nodes, annotations, _ = _checked(query(node, count))
    _require_full_dependencies(annotations)
    return nodes


class _SourcePrunedGraph:
    def __init__(self, graph, executable, captured, spec):
        self.graph, self.executable = graph, executable
        self.owners = (captured, spec)

    def replay(self):
        import torch
        from cuda.bindings import driver

        _checked(
            driver.cuGraphLaunch(
                self.executable,
                driver.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )

    def reset(self):
        import torch
        from cuda.bindings import driver

        if self.executable is not None:
            torch.cuda.synchronize()
            _checked(driver.cuGraphExecDestroy(self.executable))
            self.executable = None
            _checked(driver.cuGraphDestroy(self.graph))
            self.graph = None
            self.owners[0].reset()
            self.owners = ()


def _prune(graph, spec, selected, invocations):
    """Replace losing source kernels by empty dependency nodes before instantiation."""
    from cuda.bindings import driver

    (graph_handle,) = _checked(
        driver.cuGraphClone(driver.CUgraph(graph.raw_cuda_graph()))
    )
    original_edges = _edges(graph_handle)
    replacements = {}
    _, count = _checked(driver.cuGraphGetNodes(graph_handle))
    nodes, _ = _checked(driver.cuGraphGetNodes(graph_handle, count))
    predicates = spec.launch_predicates()
    by_function = {
        record.function: (index, record) for index, record in enumerate(predicates)
    }
    if len(by_function) != len(predicates):
        raise RuntimeError("distinct variants must retain distinct CUDA functions")
    value = selected.to_bytes(4, "little")
    hits = [0] * len(predicates)
    inspections = []
    losing_nodes = []
    for node in nodes:
        (kind,) = _checked(driver.cuGraphNodeGetType(node))
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        (params,) = _checked(driver.cuGraphKernelNodeGetParams(node))
        identity = by_function.get(int(params.func))
        if identity is None:
            raise RuntimeError(
                "serving benchmark graph contains an undeclared CUDA function"
            )
        index, predicate = identity
        if _selector_pointer(params, predicate) != predicate.selector.data_ptr():
            raise RuntimeError(
                "captured joint selector differs from its prepared predicate"
            )
        if value not in predicate.allowed_values:
            raise RuntimeError(
                "selected value is outside the captured predicate domain"
            )
        variant = spec.variants[index]
        (attribute,) = _checked(
            driver.cuGraphKernelNodeGetAttribute(
                node,
                driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
            )
        )
        geometry = (
            params.gridDimX,
            params.gridDimY,
            params.gridDimZ,
            params.blockDimX,
            params.blockDimY,
            params.blockDimZ,
            bool(attribute.cooperative),
        )
        expected = (variant.blocks, 1, 1, variant.threads, 1, 1, variant.cooperative)
        if geometry != expected:
            raise RuntimeError(
                f"captured variant geometry {geometry} differs from {expected}"
            )
        enabled = value in predicate.enabled_values
        if not enabled:
            losing_nodes.append(node)
        if hits[index] == 0:
            inspections.append(
                dict(variant_id=index, enabled=enabled, geometry=list(geometry))
            )
        hits[index] += 1
    if hits != [invocations] * len(predicates):
        raise RuntimeError(
            f"captured variant node counts {hits} differ from {invocations}"
        )
    for node in losing_nodes:
        incoming = _adjacent(driver.cuGraphNodeGetDependencies, node)
        outgoing = _adjacent(driver.cuGraphNodeGetDependentNodes, node)
        (empty,) = _checked(
            driver.cuGraphAddEmptyNode(graph_handle, incoming, len(incoming))
        )
        if outgoing:
            _checked(
                driver.cuGraphAddDependencies(
                    graph_handle,
                    [empty] * len(outgoing),
                    outgoing,
                    None,
                    len(outgoing),
                )
            )
        replacements[int(node)] = int(empty)
        _checked(driver.cuGraphDestroyNode(node))
    expected_edges = {
        (replacements.get(source, source), replacements.get(target, target))
        for source, target in original_edges
    }
    if _edges(graph_handle) != expected_edges:
        raise RuntimeError("source pruning changed the captured dependency graph")
    (_, final_count) = _checked(driver.cuGraphGetNodes(graph_handle))
    if final_count != count:
        raise RuntimeError(
            "source pruning must retain one dependency node per captured node"
        )
    (executable,) = _checked(driver.cuGraphInstantiate(graph_handle, 0))
    return _SourcePrunedGraph(graph_handle, executable, graph, spec), dict(
        nodes_per_variant=hits,
        variants=sorted(inspections, key=lambda row: row["variant_id"]),
        enabled_nodes=sum(
            hits[row["variant_id"]] for row in inspections if row["enabled"]
        ),
        empty_nodes=len(losing_nodes),
        dependency_edges=len(expected_edges),
        dependency_graph_preserved=True,
    )


def _variants():
    from b12x.comm.pcie import JointLaunchVariant

    return (
        JointLaunchVariant("joint_bandwidth", 512, 64, (("bulk_push", 16),)),
        JointLaunchVariant("joint_bandwidth", 512, 8, (("bulk_push_phased", 8),), True),
        JointLaunchVariant("joint_latency", 256, 64, (("full", 8),)),
        JointLaunchVariant("joint_latency", 256, 8, (("full_phased", 8),), True),
    )


def _capture(session, manager, state, source, output, chain):
    import torch

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with session.capture(), manager.capture(), torch.cuda.graph(graph):
        state.run_plain(source, output)
        begin.record()
        for _ in range(chain):
            state.run_plain(source, output)
        end.record()
    graph.instantiate()
    return graph, begin, end


def _run(args, write):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import AllReduce, CollectiveConfig, joint_startup_spec
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.preparation import CollectiveRequirement, PreparationSession

    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    variants = _variants()
    manager = AllReduce.from_exchange_group(
        joint=True,
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=max(args.sizes),
        eager_buffer_bytes=max(args.sizes),
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    cases, requests, graphs = {}, [], []
    try:
        for size in args.sizes:
            source = torch.full(
                (size // 16, 8), rank + 1, dtype=torch.bfloat16, device=device
            )
            output = torch.empty_like(source)
            plan = manager.plan(
                source, out=output, joint_only=True, joint_variants=variants
            )
            controls = {}
            for variant_id in (0, 1):
                variant = variants[variant_id]
                mode, active = variant.profiles[0]
                controls[variant_id] = manager.plan(
                    source,
                    out=output,
                    joint_only=True,
                    override=CollectiveConfig(
                        implementation=variant.family,
                        threads=variant.threads,
                        blocks=variant.blocks,
                        submode=mode,
                        active_blocks=active,
                        cooperative=variant.cooperative,
                    ),
                )
            cases[size] = (source, output, plan, controls)
            for name, declaration in (("guarded", plan), *controls.items()):
                key = f"pruning-{size}-{name}"
                requests.append(
                    declaration.request(
                        name=key,
                        collective=CollectiveRequirement(key, (0, 1, 2, 3)),
                        prepare_call=lambda state, source=source, output=output: (
                            _prepare_plain_call(
                                state,
                                inp=source,
                                out=output,
                            )
                        ),
                    )
                )
        session.prepare(
            tuple(requests),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        props = torch.cuda.get_device_properties(device)
        devices = [None] * 4
        dist.all_gather_object(
            devices,
            dict(
                rank=rank,
                name=props.name,
                uuid=str(props.uuid),
                capability=[props.major, props.minor],
            ),
        )
        write(
            dict(
                kind="metadata",
                devices=devices,
                variants=[asdict(v) for v in variants],
                sizes=args.sizes,
                chain=args.chain,
                warmup=args.warmup,
                samples=args.samples,
                replays=args.replays,
                alias=False,
                cache_policy="one untimed same-path invocation before each timed chain",
                metric="maximum rank-local CUDA-event microseconds per invocation",
                ordering="rotated path order per sample",
                node_selection="source clone with losing kernels replaced by empty dependency nodes; exact CUfunction and selector argument",
            )
        )
        for size, (source, output, plan, controls) in cases.items():
            state = plan.prepared.state
            spec = joint_startup_spec(plan)
            for variant_id, control in controls.items():
                variant = variants[variant_id]
                mode, active = variant.profiles[0]
                state.select_profile(mode, active, variant_id=variant_id)
                selected = variant.encoded_values(variant_id)[0]
                measured = {
                    "standalone": _capture(
                        session,
                        manager,
                        control.prepared.state,
                        source,
                        output,
                        args.chain,
                    ),
                    "guarded": _capture(
                        session, manager, state, source, output, args.chain
                    ),
                    "pruned": _capture(
                        session, manager, state, source, output, args.chain
                    ),
                }
                pruned, inspection = _prune(
                    measured["pruned"][0], spec, selected, args.chain + 1
                )
                measured["pruned"] = (pruned, *measured["pruned"][1:])
                graphs.extend(value[0] for value in measured.values())
                for path, (graph, _, end) in measured.items():
                    write(
                        dict(
                            kind="warmup_begin",
                            bytes=size,
                            winner_variant=variant_id,
                            path=path,
                        )
                    )
                    for _ in range(args.warmup):
                        graph.replay()
                    end.synchronize()
                    if not torch.equal(output, torch.full_like(output, 10)):
                        raise RuntimeError(
                            "prepared graph output differs from the exact rank-sum fixture"
                        )
                    write(
                        dict(
                            kind="warmup_complete",
                            bytes=size,
                            winner_variant=variant_id,
                            path=path,
                        )
                    )
                names = tuple(measured)
                local = {name: [] for name in names}
                for sample in range(args.samples):
                    order = names[sample % len(names) :] + names[: sample % len(names)]
                    for name in order:
                        graph, begin, end = measured[name]
                        values = []
                        for _ in range(args.replays):
                            dist.barrier()
                            graph.replay()
                            end.synchronize()
                            values.append(begin.elapsed_time(end) * 1000 / args.chain)
                        local[name].append(values)
                ranks = [None] * 4
                dist.all_gather_object(ranks, local)
                for name in names:
                    maxima = [
                        [
                            max(value[name][sample][replay] for value in ranks)
                            for replay in range(args.replays)
                        ]
                        for sample in range(args.samples)
                    ]
                    summary = median(sum(sample) / len(sample) for sample in maxima)
                    write(
                        dict(
                            kind="measurement",
                            bytes=size,
                            winner_variant=variant_id,
                            winner_label=variant.choice_label(mode, active),
                            path=name,
                            rank_samples_us=[value[name] for value in ranks],
                            max_rank_samples_us=maxima,
                            median_sample_mean_us=summary,
                            correctness="passed",
                            inspection=inspection if name == "pruned" else None,
                        )
                    )
                for graph, _, _ in measured.values():
                    graph.reset()
                    graphs.remove(graph)
    finally:
        for graph in graphs:
            graph.reset()
        session.close()
        manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(1024, 81920, 2621440))
    parser.add_argument("--chain", type=int, default=32)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if any(
        value <= 0 for value in (args.chain, args.samples, args.replays, args.warmup)
    ):
        parser.error("chain, samples, replays, and warmup must be positive")
    if any(value < 16 or value % 16 for value in args.sizes):
        parser.error("sizes must be positive 16-byte multiples")
    if args.output.exists():
        parser.error("output already exists")
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    if dist.get_world_size() != 4:
        raise ValueError("benchmark requires four ranks")
    rank = dist.get_rank()
    stream = args.output.open("x") if rank == 0 else None

    def write(record):
        if stream is not None:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    try:
        _run(args, write)
    except BaseException:
        failure = dict(kind="failure", rank=rank, traceback=traceback.format_exc())
        write(failure)
        failure_path = args.output.with_name(
            args.output.name + f".rank{rank}.failure.json"
        )
        with failure_path.open("x") as report:
            json.dump(failure, report, indent=2)
        raise
    finally:
        if stream is not None:
            stream.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
