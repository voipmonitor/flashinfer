"""Compare prepared BF16 all-reduce with NCCL on independent CUDA graph chunks.

Run with torchrun on 16 explicitly assigned GPUs and set
``B12X_PCIE_ALLREDUCE_ALGORITHM=island_rs``. Supply ``CollectiveConfig`` JSON
through ``--config``, or use ``--autotune`` to select a configuration before
capture. NCCL uses either separate send/receive buffers
or a timed input copy followed by an in-place reduction. Inputs remain immutable.
An input offset of two BF16 elements exercises four-byte-aligned source views.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flashinfer/experimental/b12x/_compat"))

import torch
import torch.distributed as dist

from b12x.comm.pcie import AllReduce, CollectiveConfig
from b12x.comm.pcie import _owner_preparation
from b12x.preparation import CollectiveRequirement, PreparationSession


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


class NativeNccl:
    def __init__(self, world, rank):
        self.library = ctypes.CDLL(
            str(Path(torch.__file__).parent.parent / "nvidia/nccl/lib/libnccl.so.2")
        )
        self.library.ncclGetErrorString.restype = ctypes.c_char_p
        self.library.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
        self.library.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            NcclUniqueId,
            ctypes.c_int,
        ]
        self.library.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        uid = NcclUniqueId()
        if rank == 0:
            self.check(self.library.ncclGetUniqueId(ctypes.byref(uid)))
        shared = [bytes(uid)]
        dist.broadcast_object_list(shared, src=0)
        self.comm = ctypes.c_void_p()
        self.check(
            self.library.ncclCommInitRank(
                ctypes.byref(self.comm),
                world,
                NcclUniqueId.from_buffer_copy(shared[0]),
                rank,
            )
        )

    def check(self, result):
        if result != 0:
            raise RuntimeError(self.library.ncclGetErrorString(result).decode())

    def all_reduce(self, source, output):
        self.check(
            self.library.ncclAllReduce(
                source.data_ptr(),
                output.data_ptr(),
                source.numel(),
                9,
                0,
                self.comm,
                torch.cuda.current_stream().cuda_stream,
            )
        )

    def close(self):
        self.check(self.library.ncclCommDestroy(self.comm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1381)
    parser.add_argument("--input-offset-elements", type=int, choices=(0, 2), default=0)
    parser.add_argument("--include-full-tensor", action="store_true")
    parser.add_argument("--profile-b12x", action="store_true")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config")
    selection.add_argument("--autotune", action="store_true")
    parser.add_argument("--tuning-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-timing-output", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.chunks, args.samples, args.replays) <= 0:
        parser.error("shape, chunks, samples, and replays must be positive")
    if args.warmup < 1:
        parser.error("warmup must be positive")
    config = (
        None if args.config is None else CollectiveConfig(**json.loads(args.config))
    )
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 16, "the ordered reference requires four four-rank islands"
    nccl_group = dist.new_group(backend="nccl", timeout=timedelta(minutes=3))
    native_nccl = NativeNccl(world, rank)
    runtime = AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=args.rows * args.columns * 2,
    )
    session = PreparationSession(
        device=device,
        autotune=args.autotune,
        compile_workers=1,
        cache_dir=args.tuning_cache_dir,
        collective_barrier=(lambda key, ranks: dist.barrier())
        if args.autotune
        else None,
    )
    prepared = None
    graphs = {}
    names = ("b12x", "nccl_out_of_place", "nccl_copy_in_place")
    if args.include_full_tensor:
        names += ("nccl_full_tensor",)
    parent_shape = (args.rows * args.chunks, args.columns)
    input_storage = None
    if args.input_offset_elements:
        parent_elements = parent_shape[0] * parent_shape[1]
        input_storage = torch.full(
            (parent_elements + args.input_offset_elements + 8,),
            -123.0,
            device=device,
            dtype=torch.bfloat16,
        )
        input_parent = input_storage[
            args.input_offset_elements : args.input_offset_elements + parent_elements
        ].view(parent_shape)
        input_parent.zero_()
    else:
        input_parent = torch.zeros(parent_shape, device=device, dtype=torch.bfloat16)
    inputs = input_parent.split(args.rows)
    output_parents = {name: torch.empty_like(input_parent) for name in names}
    outputs = {name: parent.split(args.rows) for name, parent in output_parents.items()}

    def check_input_canaries():
        if input_storage is not None:
            assert (
                (input_storage[: args.input_offset_elements] == -123.0).all().item()
            ), "input prefix canary changed"
            assert (
                (
                    input_storage[args.input_offset_elements + input_parent.numel() :]
                    == -123.0
                )
                .all()
                .item()
            ), "input suffix canary changed"

    group_timings = None
    try:
        plan = runtime.plan(inputs[0], out=outputs["b12x"][0], override=config)
        request = plan.request(
            name="bf16-reduce",
            collective=CollectiveRequirement(
                key="bf16-reduce", ranks=tuple(range(world))
            ),
            prepare_call=lambda state: _owner_preparation.prepared_call(
                state, inp=inputs[0], out=outputs["b12x"][0]
            ),
        )
        if args.autotune:
            from benchmark_pcie_allreduce_transports import prepare_tuned

            prepared = prepare_tuned(session, (request,))
        else:
            prepared = session.prepare(
                (request,),
                coordinator=lambda progress: progress.ready_collectives[0].key
                if progress.ready_collectives
                else None,
            )
        surface = plan.prepared.state.query.surface
        assert surface == "PCIeIslandRSAllReduce.all_reduce", surface
        if args.group_timing_output is not None:
            from b12x.comm.pcie._island_rs_cute import (
                GROUP_TIMING_FIELDS,
                _STREAM_WORDS,
            )

            selected = plan.contract.config_payload(plan.selection.config).to_dict()
            assert selected["implementation"] == "island_push_rsag", selected
            assert selected["threads"] % 128 == 0, selected
            active = selected["active_blocks"] or selected["blocks"]
            partition = (inputs[0].numel() + 31) // 32
            slices = (partition + active * _STREAM_WORDS - 1) // (
                active * _STREAM_WORDS
            )
            group_timings = torch.zeros(
                (args.chunks, active, 4, slices, len(GROUP_TIMING_FIELDS)),
                dtype=torch.int64,
                device=device,
            )
            runtime._island_rs.bind_group_timing(group_timings)

        def run(name):
            if name == "nccl_full_tensor":
                native_nccl.all_reduce(input_parent, output_parents[name])
                return
            for source, output in zip(inputs, outputs[name]):
                if name == "b12x":
                    runtime.all_reduce(source, out=output, plan=plan)
                elif name == "nccl_out_of_place":
                    native_nccl.all_reduce(source, output)
                else:
                    output.copy_(source)
                    dist.all_reduce(output, group=nccl_group)

        for name in names:
            for _ in range(3):
                run(name)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph(keep_graph=name == "b12x")
            graphs[name] = graph
            with session.capture(), runtime.capture(), torch.cuda.graph(graph):
                run(name)
            if name == "b12x":
                import cuda.bindings.driver as driver

                raw_graph = driver.CUgraph(graph.raw_cuda_graph())
                status, _, count = driver.cuGraphGetNodes(raw_graph)
                assert not status, status
                status, nodes, count = driver.cuGraphGetNodes(raw_graph, count)
                assert not status, status
                graph_node_types = Counter()
                for node in nodes[:count]:
                    status, kind = driver.cuGraphNodeGetType(node)
                    assert not status, status
                    graph_node_types[kind.name] += 1
                graph.instantiate()

        for _ in range(args.warmup):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        check_input_canaries()

        if args.profile_b12x:
            dist.barrier()
            if rank == 0:
                torch.cuda.profiler.start()
            dist.barrier()
            for _ in range(4):
                graphs["b12x"].replay()
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                torch.cuda.profiler.stop()

        errors = {name: [] for name in names}
        saved = []
        expected = []
        absolute_sums = []
        ordered = []
        for chunk, source in enumerate(inputs):
            total = torch.zeros(source.shape, dtype=torch.float32)
            magnitude = torch.zeros_like(total)
            island_total = torch.zeros_like(total)
            partial = torch.zeros_like(total)
            for peer in range(world):
                generator = torch.Generator().manual_seed(
                    args.seed + chunk * world + peer
                )
                local = torch.randn(source.shape, generator=generator).bfloat16()
                if peer == rank:
                    saved.append(local)
                    source.copy_(local)
                value = local.float()
                total.add_(value)
                magnitude.add_(value.abs())
                partial.add_(value)
                if peer % 4 == 3:
                    island_total.add_(partial.bfloat16().float())
                    partial.zero_()
            expected.append(total)
            absolute_sums.append(magnitude)
            ordered.append(island_total.bfloat16())

        # Sixteen BF16 operands require at most fifteen rounded additions.
        unit_roundoff = 2**-8
        gamma = (world - 1) * unit_roundoff / (1 - (world - 1) * unit_roundoff)
        for name, graph in graphs.items():
            output_parents[name].fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            for chunk, output in enumerate(outputs[name]):
                actual = output.cpu()
                assert torch.isfinite(actual).all(), name
                if name == "b12x":
                    torch.testing.assert_close(actual, ordered[chunk], rtol=0, atol=0)
                difference = (actual.float() - expected[chunk]).abs()
                assert (difference <= gamma * absolute_sums[chunk]).all(), name
                errors[name].append(
                    {
                        "max_absolute_error": difference.max().item(),
                        "rms_error": difference.square().mean().sqrt().item(),
                    }
                )
                torch.testing.assert_close(
                    inputs[chunk].cpu(), saved[chunk], rtol=0, atol=0
                )
            check_input_canaries()

        changed_inputs = 6
        for step in range(changed_inputs):
            exact_expected = []
            exact_saved = []
            for chunk, source in enumerate(inputs):
                coordinates = torch.arange(source.numel(), dtype=torch.int32).view(
                    source.shape
                )
                total = torch.zeros(source.shape, dtype=torch.float32)
                for peer in range(world):
                    value = (
                        (coordinates * (2 * peer + 1) + step * 7 + chunk * 3 + peer) % 5
                        - 2
                    ).float() / 16
                    total.add_(value)
                    if peer == rank:
                        local = value.bfloat16()
                        source.copy_(local)
                        exact_saved.append(local)
                exact_expected.append(total.bfloat16())
            for name, graph in graphs.items():
                output_parents[name].fill_(float("nan"))
                torch.cuda.synchronize()
                dist.barrier()
                before = torch.cuda.memory_stats(device)
                graph.replay()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for counter in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[counter] == after[counter], (name, counter)
                for source, output, original, reference in zip(
                    inputs, outputs[name], exact_saved, exact_expected
                ):
                    torch.testing.assert_close(source.cpu(), original, rtol=0, atol=0)
                    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
                check_input_canaries()

        for source, original in zip(inputs, saved):
            source.copy_(original)
        for _ in range(args.warmup):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        end.record()
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for sample in range(args.samples):
            order = names[sample % len(names) :] + names[: sample % len(names)]
            if sample % 2:
                order = tuple(reversed(order))
            for name in order:
                dist.barrier()
                start.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / args.replays)
        rank_samples = [None] * world
        dist.all_gather_object(rank_samples, samples)
        for source, original in zip(inputs, saved):
            torch.testing.assert_close(source.cpu(), original, rtol=0, atol=0)
        check_input_canaries()
        for name in names:
            for chunk, output in enumerate(outputs[name]):
                actual = output.cpu()
                assert torch.isfinite(actual).all(), name
                if name == "b12x":
                    torch.testing.assert_close(actual, ordered[chunk], rtol=0, atol=0)
                assert (
                    (actual.float() - expected[chunk]).abs()
                    <= gamma * absolute_sums[chunk]
                ).all(), name
        timings = {}
        for name in names:
            per_rank = [values[name] for values in rank_samples]
            maximums = [max(values) for values in zip(*per_rank)]
            calls = 1 if name == "nccl_full_tensor" else args.chunks
            timings[name] = {
                "calls_per_graph": calls,
                "input_bytes_per_call": input_parent.nbytes // calls,
                "rank_graph_samples_us": per_rank,
                "max_rank_graph_samples_us": maximums,
                "max_rank_graph_median_us": median(maximums),
                "max_rank_collective_samples_us": [value / calls for value in maximums],
                "max_rank_collective_median_us": median(maximums) / calls,
            }
        properties = torch.cuda.get_device_properties(device)
        if group_timings is not None:
            raw_group_timings = group_timings.cpu().tolist()
            group_warps = selected["threads"] // 128
            for call in raw_group_timings:
                for block in call:
                    generations = {row[0] for group in block for row in group}
                    assert len(generations) == 1, generations
                    for group_index, group in enumerate(block):
                        for row in group:
                            assert row[1] > 0 and row[1:6] == sorted(row[1:6]), row
                            warp_ends = row[8 : 8 + group_warps]
                            assert all(value > 0 for value in warp_ends), row
                            assert max(warp_ends) >= row[5], row
                            assert not any(row[8 + group_warps :]), row
                            if group_index == 3:
                                assert row[5] <= row[6] <= row[7] <= max(warp_ends), row
                            else:
                                assert row[6:8] == [0, 0], row
            rank_group_timings = [None] * world if rank == 0 else None
            dist.gather_object(raw_group_timings, rank_group_timings, dst=0)
            if rank == 0:
                args.group_timing_output.parent.mkdir(parents=True, exist_ok=True)
                args.group_timing_output.write_text(
                    json.dumps(
                        {
                            "fields": GROUP_TIMING_FIELDS,
                            "groups": [
                                "local_scatter",
                                "local_reduction",
                                "island_reduction",
                                "local_distribution",
                            ],
                            "layout": [
                                "rank",
                                "generation_mod_calls",
                                "CTA",
                                "group",
                                "slice",
                                "field",
                            ],
                            "record_shape_per_rank": list(group_timings.shape),
                            "device_bytes_per_rank": group_timings.nbytes,
                            "input_shape": list(inputs[0].shape),
                            "selected_config": selected,
                            "words_per_slice": _STREAM_WORDS,
                            "threads_per_group": selected["threads"] // 4,
                            "clock": "device-local globaltimer nanoseconds",
                            "timing_scope": "Last calls retained after the complete prepared graph benchmark. Leaders record entry, readiness and publication; every warp records completion. Fence intervals include group skew. Absolute clocks are not compared across devices.",
                            "rank_records": rank_group_timings,
                        }
                    )
                    + "\n"
                )
        record = {
            "command": shlex.join(sys.orig_argv),
            "world_size": world,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "gpu": str(properties),
            "gpu_uuid": str(properties.uuid),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "input_shape": list(inputs[0].shape),
            "input_offset_elements": args.input_offset_elements,
            "input_boundary_canaries_checked": bool(args.input_offset_elements),
            "input_bytes_per_collective": inputs[0].nbytes,
            "chunks_per_graph": args.chunks,
            "b12x_graph_node_counts": dict(graph_node_types),
            "b12x_runtime_algorithm": os.getenv("B12X_PCIE_ALLREDUCE_ALGORITHM"),
            "dtype": str(inputs[0].dtype),
            "prepared_surface": surface,
            "autotune": args.autotune,
            "selection_source": plan.selection.source,
            "preparation_cache_hits": prepared.cache_hits,
            "preparation_benchmarked_candidates": prepared.benchmarked_candidates,
            "selected_config": plan.contract.config_payload(
                plan.selection.config
            ).to_dict(),
            "samples": args.samples,
            "replays_per_sample": args.replays,
            "warmup_replays": args.warmup,
            "seed": args.seed,
            "changed_graph_inputs": changed_inputs,
            "exact_changed_input_sums": True,
            "exact_b12x_ordered_reference": True,
            "random_sum_error_bound": "gamma_15 * sum(abs(BF16 inputs)), unit roundoff 2^-8",
            "random_sum_error": errors,
            "replay_allocation_counters_unchanged": True,
            "timings": timings,
        }
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        name: value["max_rank_collective_median_us"]
                        for name, value in timings.items()
                    }
                ),
                flush=True,
            )
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        if prepared is not None:
            prepared.close()
        session.close()
        runtime.close()
        native_nccl.close()
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
