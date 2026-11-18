"""Compare prepared b12x and NCCL BF16 column all-gather in CUDA graphs.

Launch with torchrun and an explicit CUDA_VISIBLE_DEVICES assignment. Each graph
gathers rank-local columns into rank order for every independent chunk. CPU Gloo
references and input updates run outside CUDA event timing. NCCL can retain its
rank-major output or include conversion to the token-major layout consumed by Kimi.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
from statistics import mean, median
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flashinfer/experimental/b12x/_compat"))

import torch
import torch.distributed as dist

from b12x.comm import pcie
from b12x.preparation import (
    CollectiveRequirement,
    PreparationSession,
    PreparedCall,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=336)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1381)
    parser.add_argument(
        "--nccl-output-layout",
        choices=("token-major", "rank-major"),
        default="token-major",
        help="rank-major times NCCL directly without an output-layout copy",
    )
    parser.add_argument(
        "--config", help="CollectiveConfig fields as an inline JSON object"
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune normal gather alternatives before graph capture",
    )
    parser.add_argument("--tuning-cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.chunks, args.samples, args.replays) <= 0:
        parser.error("rows, columns, chunks, samples, and replays must be positive")
    if args.columns % 8:
        parser.error("columns must be divisible by eight")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if args.autotune and args.config is not None:
        parser.error("--autotune cannot be combined with --config")
    config = None
    if args.config is not None:
        try:
            payload = json.loads(args.config)
            if not isinstance(payload, dict):
                raise TypeError("config must be a JSON object")
            config = pcie.CollectiveConfig(**payload)
        except (ValueError, TypeError) as error:
            parser.error(str(error))

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    nccl_group = dist.new_group(backend="nccl", timeout=timedelta(minutes=3))
    pool = pcie.DcpAllToAllPool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=args.rows,
        total_heads=world,
        head_dim=args.columns,
        query_head_dim=args.columns,
        single_channel=True,
    )
    runtime = pool.for_stream()
    session = PreparationSession(
        device=device,
        autotune=args.autotune,
        compile_workers=1,
        cache_dir=args.tuning_cache_dir,
        collective_barrier=(lambda key, ranks: dist.barrier())
        if args.autotune
        else None,
    )
    preparation = None
    graphs = {}
    try:
        inputs = [
            torch.zeros(
                (args.rows, 1, args.columns), device=device, dtype=torch.bfloat16
            )
            for _ in range(args.chunks)
        ]
        output_shapes = {
            "b12x": (args.rows, world, args.columns),
            "nccl": (world, args.rows, args.columns)
            if args.nccl_output_layout == "rank-major"
            else (args.rows, world, args.columns),
        }
        outputs = {
            name: [
                torch.empty(shape, device=device, dtype=torch.bfloat16)
                for _ in range(args.chunks)
            ]
            for name, shape in output_shapes.items()
        }
        nccl_rank_major = (
            outputs["nccl"]
            if args.nccl_output_layout == "rank-major"
            else [
                torch.empty(
                    (world, args.rows, args.columns),
                    device=device,
                    dtype=torch.bfloat16,
                )
                for _ in range(args.chunks)
            ]
        )
        for _ in range(3):
            dist.all_gather_into_tensor(
                nccl_rank_major[0].view(world * args.rows, args.columns),
                inputs[0].view(args.rows, args.columns),
                group=nccl_group,
            )
        torch.cuda.synchronize()
        call = {"local_input": inputs[0], "out": outputs["b12x"][0]}
        query = pcie.query_from_runtime(
            runtime, surface="DcpAllToAll.all_gather_heads", call=call
        )
        plan = pcie.plan(query, runtime=runtime, override=config)
        name = f"gather-{args.rows}-{args.columns}"
        requests = (
            plan.request(
                name=name,
                collective=CollectiveRequirement(key=name, ranks=tuple(range(world))),
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(**call),
                    output=outputs["b12x"][0],
                    owners=(runtime,),
                ),
            ),
        )
        if args.autotune:
            from benchmark_pcie_allreduce_transports import prepare_tuned

            preparation = prepare_tuned(session, requests)
        else:
            preparation = session.prepare(
                requests,
                coordinator=lambda progress: progress.ready_collectives[0].key
                if progress.ready_collectives
                else None,
            )
        for name in outputs:
            graph = torch.cuda.CUDAGraph()
            graphs[name] = graph
            dist.barrier()
            with session.capture(), pool.capture(), torch.cuda.graph(graph):
                for source, output, staging in zip(
                    inputs, outputs[name], nccl_rank_major
                ):
                    if name == "b12x":
                        runtime.all_gather_heads(source, out=output, plan=plan)
                    else:
                        dist.all_gather_into_tensor(
                            staging.view(world * args.rows, args.columns),
                            source.view(args.rows, args.columns),
                            group=nccl_group,
                        )
                        if args.nccl_output_layout == "token-major":
                            output.copy_(staging.permute(1, 0, 2))

        for _ in range(max(1, args.warmup)):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        changed_inputs = 6
        for step in range(changed_inputs):
            expected, saved = {name: [] for name in graphs}, []
            for chunk, source in enumerate(inputs):
                generator = torch.Generator().manual_seed(
                    args.seed + (step * args.chunks + chunk) * world + rank
                )
                local = torch.randn(source.shape, generator=generator).bfloat16()
                gathered = [torch.empty_like(local) for _ in range(world)]
                dist.all_gather(gathered, local)
                expected["b12x"].append(torch.cat(gathered, dim=1))
                expected["nccl"].append(
                    torch.cat(gathered, dim=0).view(world, args.rows, args.columns)
                    if args.nccl_output_layout == "rank-major"
                    else expected["b12x"][-1]
                )
                saved.append(local)
                source.copy_(local)
            for name, graph in graphs.items():
                for output in outputs[name]:
                    output.fill_(float("nan"))
                for staging in nccl_rank_major:
                    staging.fill_(float("nan"))
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
                    inputs, outputs[name], saved, expected[name]
                ):
                    torch.testing.assert_close(source.cpu(), original, rtol=0, atol=0)
                    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)

        for _ in range(args.warmup):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.samples)
        ]
        for start, end in events:
            start.record()
            end.record()
        torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        for sample, (start, end) in enumerate(events):
            order = ("b12x", "nccl") if sample % 2 == 0 else ("nccl", "b12x")
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
        for name, buffers in outputs.items():
            for output, reference in zip(buffers, expected[name]):
                torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)

        timings = {}
        for name in graphs:
            per_rank = [values[name] for values in rank_samples]
            maximums = [max(values) for values in zip(*per_rank)]
            timings[name] = {
                "rank_graph_samples_us": per_rank,
                "rank_collective_mean_us": [
                    mean(values) / args.chunks for values in per_rank
                ],
                "max_rank_graph_samples_us": maximums,
                "max_rank_graph_mean_us": mean(maximums),
                "max_rank_graph_median_us": median(maximums),
                "max_rank_collective_samples_us": [
                    value / args.chunks for value in maximums
                ],
                "max_rank_collective_mean_us": mean(maximums) / args.chunks,
                "max_rank_collective_median_us": median(maximums) / args.chunks,
            }
        properties = torch.cuda.get_device_properties(device)
        record = {
            "command": shlex.join(sys.orig_argv),
            "world_size": world,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "nccl_output_layout": args.nccl_output_layout,
            "nccl_includes_output_layout_copy": args.nccl_output_layout
            == "token-major",
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu": str(properties),
            "gpu_uuid": str(properties.uuid),
            "input_shape": list(inputs[0].shape),
            "output_shape": list(outputs["b12x"][0].shape),
            "nccl_output_shape": list(outputs["nccl"][0].shape),
            "input_bytes_per_collective": inputs[0].nbytes,
            "output_bytes_per_collective": outputs["b12x"][0].nbytes,
            "capacity_rows": args.rows,
            "dtype": str(inputs[0].dtype),
            "stream_affine": runtime._stream_affine,
            "prepared_surface": plan.prepared.state.query.surface,
            "selected_config": plan.contract.config_payload(
                plan.selection.config
            ).to_dict(),
            "autotune": args.autotune,
            "selection_source": plan.selection.source,
            "preparation_cache_hits": preparation.cache_hits,
            "preparation_benchmarked_candidates": preparation.benchmarked_candidates,
            "selected_transport": plan.prepared.state.query.call.get(
                "gather_transport"
            ),
            "chunks_per_graph": args.chunks,
            "warmup_replays": args.warmup,
            "initial_warmup_replays": max(1, args.warmup),
            "changed_graph_inputs": changed_inputs,
            "exact_rank_concatenation": True,
            "replay_allocation_counters_unchanged": True,
            "samples": args.samples,
            "replays_per_sample": args.replays,
            "timings": timings,
            "b12x_over_nccl_latency_ratio": timings["b12x"][
                "max_rank_collective_median_us"
            ]
            / timings["nccl"]["max_rank_collective_median_us"],
        }
        if rank == 0:
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        if preparation is not None:
            preparation.close()
        session.close()
        pool.close()
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
