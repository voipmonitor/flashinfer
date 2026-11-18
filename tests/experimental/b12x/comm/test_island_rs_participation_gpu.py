"""Exercise inactive CTA reactivation and independent buffer/flag counter rollover."""

import ctypes
from datetime import timedelta
import os
from pathlib import Path

import pytest


def _worker(rank, directory, count, threads, cases, hybrid):
    import torch
    import torch.distributed as dist
    from cuda.bindings import driver as cuda
    from b12x.comm.pcie import _island_rs_cute as kernel
    from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=16,
        init_method=(Path(directory) / "group").as_uri(),
        timeout=timedelta(minutes=5),
    )
    physical = 64
    owner = PCIeIslandRSAllReduce(
        exchange_group=dist.group.WORLD,
        device=torch.device("cuda", rank),
        max_elements=count,
        blocks=physical,
    )
    if hybrid:
        from b12x.comm.pcie._island_copy import get_packet_programs

        transport = owner.packet_transport()
        capture_streams = {
            torch.cuda.Stream(device=owner.device).cuda_stream for _ in range(64)
        }
        assert transport.stream.cuda_stream not in capture_streams
        launcher = get_packet_programs(rank, threads, rank).bind(transport)
        with pytest.raises(
            ValueError, match="compute and copy streams must be distinct"
        ):
            launcher(
                owner._slab_ptrs,
                0,
                0,
                owner.stage_offset,
                owner.part_offset,
                owner.final_offset,
                owner.quarter_capacity,
                count,
                physical,
                stream=transport.stream,
            )
    else:
        launcher = kernel.get_island_rs_launcher(
            16, rank, rank, threads=threads, local_push=True, cross_island_rsag=True
        )
    try:
        for generation, previously_active, sequence in cases:
            # This state is reachable after repeated calls with a fixed active
            # subset. Inactive CTAs retain their last communication epoch, while
            # every physical CTA tracks the same data-buffer generation.
            words = (ctypes.c_uint32 * (kernel.HEADER_BYTES // 4))()
            for block in range(physical):
                words[block] = generation
            for block in range(previously_active):
                words[kernel._ACTIVE_EPOCH_OFFSET // 4 + block] = generation
                words[
                    (kernel._RS_ARRIVED + block * 4 * 128 + kernel._STREAM_COUNTER) // 4
                ] = generation
                for peer in range(4):
                    offset = (block * 4 + peer) * 128
                    words[(kernel._RS_ARRIVED + offset) // 4] = generation
                    words[(kernel._AG_ARRIVED + offset) // 4] = generation
                    words[(kernel._XS_ARRIVED + offset) // 4] = (
                        2 * generation + 1
                    ) & 0xFFFFFFFF
                    for region in (
                        kernel._RS_ARRIVED,
                        kernel._XS_ARRIVED,
                        kernel._AG_ARRIVED,
                    ):
                        words[(region + offset + kernel._STREAM_READY) // 4] = (
                            generation
                        )
                    words[(kernel._XS_ARRIVED + offset + kernel._STREAM_FINAL) // 4] = (
                        generation
                    )
            torch.cuda.synchronize()
            dist.barrier()
            assert (
                cuda.cuMemcpyHtoD(
                    owner._local_ptr, ctypes.addressof(words), ctypes.sizeof(words)
                )[0]
                == cuda.CUresult.CUDA_SUCCESS
            )
            assert (
                cuda.cuMemsetD8(
                    owner._local_ptr + owner.stage_offset,
                    255,
                    owner.slab_bytes - owner.stage_offset,
                )[0]
                == cuda.CUresult.CUDA_SUCCESS
            )
            sources = [
                torch.full(
                    (count,), (rank + 1) * (i + 1), dtype=torch.bfloat16, device="cuda"
                )
                for i in range(len(sequence))
            ]
            outputs = [torch.empty_like(source) for source in sources]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                if rank == 12:
                    torch.cuda._sleep(2_000_000)
                for active, source, output in zip(
                    sequence, sources, outputs, strict=True
                ):
                    launcher(
                        owner._slab_ptrs,
                        source.data_ptr(),
                        output.data_ptr(),
                        owner.stage_offset,
                        owner.part_offset,
                        owner.final_offset,
                        owner.quarter_capacity,
                        count,
                        physical,
                        active_blocks=active,
                    )
            dist.barrier()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            for _ in range(3):
                for output in outputs:
                    output.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for index, output in enumerate(outputs):
                torch.testing.assert_close(
                    output,
                    torch.full_like(output, 136 * (index + 1)),
                    rtol=0,
                    atol=0,
                )
            counters = (ctypes.c_uint32 * (2 * kernel.MAX_BLOCKS))()
            assert (
                cuda.cuMemcpyDtoH(
                    ctypes.addressof(counters),
                    owner._local_ptr,
                    ctypes.sizeof(counters),
                )[0]
                == cuda.CUresult.CUDA_SUCCESS
            )
            expected_data = (generation + 3 * len(sequence)) & 0xFFFFFFFF
            assert tuple(counters[:physical]) == (expected_data,) * physical
            expected_epochs = tuple(
                (
                    (generation if block < previously_active else 0)
                    + 3 * sum(block < active for active in sequence)
                )
                & 0xFFFFFFFF
                for block in range(physical)
            )
            first = kernel._ACTIVE_EPOCH_OFFSET // 4
            assert tuple(counters[first : first + physical]) == expected_epochs
            graph.reset()
    finally:
        owner.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_COLLECTIVE_TUNING_TEST") != "1",
    reason="requires sixteen peer-accessible SM120 GPUs",
)
@pytest.mark.parametrize(
    "count, threads, cases",
    [
        pytest.param(
            32768,
            128,
            (
                (1 << 30, 1, (64,)),
                (0xFFFFFFFE, 64, (64, 16, 64)),
            ),
            id="inactive-reactivation",
        ),
        pytest.param(
            1024 * 7168,
            1024,
            ((0xFFFFFFFE, 32, (32, 16, 32)),),
            id="full-capacity-wrap",
        ),
        pytest.param(
            1024 * 7168,
            768,
            ((0xFFFFFFFE, 32, (32, 16, 32)),),
            id="full-capacity-768-threads",
        ),
    ],
)
@pytest.mark.parametrize("hybrid", [False, True], ids=["sm", "hybrid"])
def test_island_rs_inactive_ctas_and_counter_rollover(
    tmp_path, count, threads, cases, hybrid
):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() != 16 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires sixteen SM120 GPUs")
    mp.spawn(
        _worker,
        args=(str(tmp_path), count, threads, cases, hybrid),
        nprocs=16,
        join=True,
    )
