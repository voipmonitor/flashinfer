"""Exercise TP16 reduction tails and changing inputs through prepared graphs."""

from datetime import timedelta
import ctypes
import os
from pathlib import Path
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = [
    pytest.mark.node_exclusive,
    pytest.mark.skipif(
        os.getenv("B12X_RUN_PCIE_ISLAND_RS_TEST") != "1",
        reason="requires sixteen peer-accessible GPUs",
    ),
]


def _input(elements, rank, variation, device):
    generator = torch.Generator(device=device).manual_seed(
        19731 + elements + variation * 103 + rank * 701
    )
    values = torch.randn(elements, device=device, generator=generator)
    # Alternate large and small cancelling contributions without overflow.
    values.mul_(16.0 if rank % 4 < 2 else 0.125)
    values.add_((1 if rank % 2 == 0 else -1) * (32 + variation))
    return values.bfloat16()


def _ordered_reference(elements, variation, device):
    result = torch.zeros(elements, dtype=torch.float32, device=device)
    for island in range(4):
        partial = torch.zeros_like(result)
        for rank in range(island * 4, island * 4 + 4):
            partial.add_(_input(elements, rank, variation, device).float())
        result.add_(partial.bfloat16().float())
    return result.bfloat16()


def _assert_canaries(regions, elements, context):
    for storage, offset in regions:
        assert (storage[:offset] == -123.0).all().item(), (context, "prefix canary")
        assert (storage[offset + elements :] == -123.0).all().item(), (
            context,
            "suffix canary",
        )


def _seed_generation(owner, generation):
    from b12x.comm.pcie._island_rs_cute import (
        HEADER_BYTES,
        _ACTIVE_EPOCH_OFFSET,
        MAX_BLOCKS,
        _RS_ARRIVED,
        _XS_ARRIVED,
        _AG_ARRIVED,
        _STREAM_COUNTER,
        _STREAM_READY,
        _STREAM_FINAL,
    )

    words = (ctypes.c_uint32 * (HEADER_BYTES // 4))()
    for block in range(MAX_BLOCKS):
        words[block] = generation
        words[_ACTIVE_EPOCH_OFFSET // 4 + block] = generation
        words[(_RS_ARRIVED + block * 4 * 128 + _STREAM_COUNTER) // 4] = generation
        for peer in range(4):
            offset = (block * 4 + peer) * 128
            words[(_RS_ARRIVED + offset) // 4] = generation
            words[(_AG_ARRIVED + offset) // 4] = generation
            words[(_XS_ARRIVED + offset) // 4] = (2 * generation + 1) & 0xFFFFFFFF
            for region in (_RS_ARRIVED, _XS_ARRIVED, _AG_ARRIVED):
                words[(region + offset + _STREAM_READY) // 4] = generation
            words[(_XS_ARRIVED + offset + _STREAM_FINAL) // 4] = generation
    stream = torch.cuda.current_stream(owner.device)
    owner._ipc.cudaMemset(
        owner._local_ptr + owner.stage_offset,
        0xFF,
        owner.slab_bytes - owner.stage_offset,
    )
    owner._ipc.cudaMemcpyAsync(
        owner._local_ptr, ctypes.addressof(words), HEADER_BYTES, stream.cuda_stream
    )
    stream.synchronize()


def _worker(
    rank,
    rendezvous,
    cross_scatter=False,
    local_push=False,
    mixed_alignment=False,
    inplace_protocol=None,
):
    from dataclasses import replace

    from b12x.comm.pcie import _owner_preparation as preparation
    from b12x.comm.pcie._collective_tuning import ISLAND_VARIANTS
    from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce
    from b12x.preparation import FrozenMapping, PreparationSession

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=16,
        init_method=rendezvous,
        timeout=timedelta(minutes=10),
    )
    capacity = 1024 * 7168
    protocols = (
        tuple(ISLAND_VARIANTS)
        if local_push
        else (
            ("island_rs", "island_rsag", "island_rsag")
            if cross_scatter
            else ("island_rs",) * 3
        )
    )
    if inplace_protocol is not None:
        protocols = tuple(p for p in protocols if p != inplace_protocol) + (
            inplace_protocol,
        )
    try:
        # Use a separate fixed-grid owner for each geometry. Changing the
        # physical grid would stop advancing some per-CTA generation counters.
        for blocks, threads in ((16, 512), (32, 1024), (32, 768)):
            stride = blocks * threads
            shapes = (
                (128, 7168, capacity)
                if mixed_alignment
                else (
                    2,
                    6,
                    7168,
                    24 * stride - 2,
                    24 * stride + 2,
                    32 * stride - 2,
                    32 * stride + 2,
                    capacity - 2,
                )
            )
            owner = PCIeIslandRSAllReduce(
                exchange_group=dist.group.WORLD,
                device=device,
                max_elements=capacity,
                blocks=blocks,
            )
            session = PreparationSession(
                device=device,
                autotune=False,
                compile_workers=4,
            )
            cases = []
            graphs = []
            try:
                for elements in shapes:
                    # Vary both caller alignments across local and cross peers.
                    source_offset = 8 + (
                        2 * ((rank ^ (rank >> 2)) & 1) if mixed_alignment else 2
                    )
                    input_storage = torch.full(
                        (elements + 24,), -123.0, dtype=torch.bfloat16, device=device
                    )
                    source = input_storage[source_offset : source_offset + elements]
                    inputs = [
                        _input(elements, rank, n, device) for n in range(len(protocols))
                    ]
                    expected = [
                        _ordered_reference(elements, n, device)
                        for n in range(len(protocols))
                    ]
                    if cross_scatter:
                        for sample in sorted({2, len(protocols) - 1}):
                            if rank == 0:
                                inputs[sample][0] = float("nan")
                                inputs[sample][1] = float("inf")
                            expected[sample][0] = float("nan")
                            expected[sample][1] = float("inf")
                            if elements > 2:
                                if rank in (0, 4):
                                    inputs[sample][2] = (
                                        float("inf") if rank == 0 else -float("inf")
                                    )
                                expected[sample][2] = float("nan")
                    output_offset = 8 + (
                        2 * (((rank >> 1) ^ (rank >> 3)) & 1) if mixed_alignment else 0
                    )
                    output_storages = [
                        torch.full(
                            (elements + 24,),
                            -123.0,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        for _ in range(len(protocols))
                    ]
                    outputs = [
                        source
                        if inplace_protocol == protocols[index]
                        else storage[output_offset : output_offset + elements]
                        for index, storage in enumerate(output_storages)
                    ]
                    guard_regions = [(input_storage, source_offset)] + [
                        (storage, output_offset) for storage in output_storages
                    ]
                    plans = []
                    for index, output in enumerate(outputs):
                        push, scattered, dma = ISLAND_VARIANTS[protocols[index]]
                        query = preparation.query_from_runtime(
                            owner,
                            call=dict(inp=source, out=output, blocks=blocks),
                        )
                        query = replace(
                            query,
                            call=FrozenMapping(
                                {
                                    **query.call,
                                    "threads": threads,
                                    "cross_island_rsag": scattered,
                                    "local_push": push,
                                    "cross_island_dma": dma,
                                }
                            ),
                        )
                        plan = preparation.plan(query, runtime=owner)
                        session.prepare(
                            (
                                plan.request(
                                    name=f"reduce-{elements}-{index}",
                                    prepare_call=lambda state,
                                    source=source,
                                    output=output: (
                                        preparation.prepared_call(
                                            state,
                                            inp=source,
                                            out=output,
                                        )
                                    ),
                                ),
                            )
                        )
                        plans.append(plan)
                    with session.capture():
                        for values, output, plan in zip(
                            inputs, outputs, plans, strict=True
                        ):
                            source.copy_(values)
                            owner.all_reduce(source, out=output, plan=plan)
                    torch.cuda.synchronize()
                    for index, (output, reference) in enumerate(
                        zip(outputs, expected, strict=True)
                    ):
                        context = f"rank={rank} elements={elements} blocks={blocks} threads={threads} protocol={protocols[index]} eager"
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                            msg=lambda detail: f"{context}\n{detail}",
                        )
                    _assert_canaries(
                        guard_regions,
                        elements,
                        f"rank={rank} elements={elements} blocks={blocks} eager",
                    )

                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        for values, output, plan in zip(
                            inputs, outputs, plans, strict=True
                        ):
                            source.copy_(values)
                            if output is not source:
                                output.fill_(float("nan"))
                            owner.all_reduce(source, out=output, plan=plan)
                    graphs.append(graph)
                    cases.append(
                        (elements, source, inputs, outputs, expected, guard_regions)
                    )

                if inplace_protocol is not None:
                    for (
                        elements,
                        source,
                        inputs,
                        outputs,
                        expected,
                        guard_regions,
                    ) in cases:
                        inputs[-1].copy_(_input(elements, rank, len(protocols), device))
                        expected[-1].copy_(
                            _ordered_reference(elements, len(protocols), device)
                        )
                torch.cuda.synchronize()
                if cross_scatter:
                    dist.barrier()
                    _seed_generation(owner, 0xFFFFFFFE)
                    dist.barrier()
                before = torch.cuda.memory_stats(device)
                for repetition in range(5):
                    if rank == repetition:
                        time.sleep(0.025)
                    sequence = graphs if repetition % 2 == 0 else reversed(graphs)
                    for graph in sequence:
                        graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "segment.all.allocated"):
                    assert before[key] == after[key], key
                for elements, source, inputs, outputs, expected, guard_regions in cases:
                    if inplace_protocol is None:
                        assert torch.equal(
                            source.view(torch.int16), inputs[-1].view(torch.int16)
                        ), (
                            rank,
                            elements,
                            "input mutation",
                        )
                    for index, (output, reference) in enumerate(
                        zip(outputs, expected, strict=True)
                    ):
                        context = f"rank={rank} elements={elements} blocks={blocks} threads={threads} protocol={protocols[index]} replay"
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                            msg=lambda detail: f"{context}\n{detail}",
                        )
                    _assert_canaries(
                        guard_regions,
                        elements,
                        f"rank={rank} elements={elements} blocks={blocks} replay",
                    )
                if cross_scatter:
                    counters = (ctypes.c_uint32 * blocks)()
                    stream = torch.cuda.current_stream(device)
                    owner._ipc.cudaMemcpyAsync(
                        ctypes.addressof(counters),
                        owner._local_ptr,
                        blocks * 4,
                        stream.cuda_stream,
                    )
                    stream.synchronize()
                    expected_generation = (
                        0xFFFFFFFE + len(shapes) * len(protocols) * 5
                    ) & 0xFFFFFFFF
                    assert tuple(counters) == (expected_generation,) * blocks
                dist.barrier()
                if rank == 0:
                    print(
                        f"TP16 blocks={blocks} threads={threads}: {len(shapes)} shapes, exact eager and queued replay passed",
                        flush=True,
                    )
            except BaseException:
                import sys
                import traceback

                traceback.print_exc()
                sys.stderr.flush()
                raise
            finally:
                for graph in graphs:
                    graph.reset()
                torch.cuda.synchronize()
                session.close()
                owner.close()
    finally:
        dist.destroy_process_group()


def test_island_rs_prepared_graphs_with_load_batch_tails():
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-rs-load-test-") as directory:
        mp.spawn(
            _worker, args=(Path(directory, "gloo").as_uri(),), nprocs=16, join=True
        )


def test_island_cross_scatter_mixed_protocols_and_rollover():
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-scatter-test-") as directory:
        mp.spawn(
            _worker, args=(Path(directory, "gloo").as_uri(), True), nprocs=16, join=True
        )


@pytest.mark.parametrize(
    "mixed_alignment", [False, True], ids=["offset-input", "mixed-alignment"]
)
def test_island_local_push_mixed_protocols_and_rollover(mixed_alignment):
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-push-test-") as directory:
        mp.spawn(
            _worker,
            args=(Path(directory, "gloo").as_uri(), True, True, mixed_alignment),
            nprocs=16,
            join=True,
        )


@pytest.mark.parametrize("inplace_protocol", ["island_push_rsag", "island_hybrid_rsag"])
def test_island_stream_inplace_mixed_alignment_and_rollover(inplace_protocol):
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-stream-inplace-test-") as directory:
        mp.spawn(
            _worker,
            args=(Path(directory, "gloo").as_uri(), True, True, True, inplace_protocol),
            nprocs=16,
            join=True,
        )
