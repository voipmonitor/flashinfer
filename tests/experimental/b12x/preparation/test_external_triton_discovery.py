"""Compiler discovery must release placeholders for caller-defined Triton kernels."""

import torch
import triton
import triton.language as tl

from b12x._lib.compile_plan import compile_only_launches, launch_triton
from b12x._lib.compile_pool import CompileJob, CompilePool

from ..conftest import require_b12x
from .test_compile_pool_fake_indexing import _run_offline_worker


@triton.jit
def _add_one(source, output, N: tl.constexpr):
    index = tl.arange(0, N)
    tl.store(output + index, tl.load(source + index) + 1)


def compile_add_one():
    operands = tuple(
        torch.empty(128, dtype=torch.float32, device="meta") for _ in range(2)
    )
    with compile_only_launches():
        return launch_triton(_add_one, (1,), *operands, N=128, num_warps=4)


FACTORY = (
    "tests.experimental.b12x.preparation.test_external_triton_discovery:compile_add_one"
)


def test_external_triton_describe_then_compile_in_same_worker():
    _run_offline_worker(
        """
        import pickle
        from tempfile import TemporaryDirectory
        from triton import knobs
        from b12x._lib.compile_plan import compiled_program_available
        from b12x._lib.compile_pool import CompileJob
        from tests.experimental.b12x.preparation.test_external_triton_discovery import FACTORY

        initialize_worker()
        payload = pickle.dumps(CompileJob.create(FACTORY), protocol=5)
        with TemporaryDirectory() as cache:
            knobs.cache.dir = cache
            ((_, programs),) = compile_pool._describe_jobs((payload,))
            assert programs and all(not compiled_program_available(p) for p in programs)
            completed = compile_pool._run_job(payload, programs)
            assert completed[2] == 1
            assert all(compiled_program_available(p) for p in programs)
        assert not torch.cuda.is_initialized()
        """
    )


def test_external_triton_compiler_artifact_replays_on_cuda(tmp_path, monkeypatch):
    from b12x._lib.compile_plan import (
        compiled_program_available,
        forbid_lowering,
        load_programs,
    )

    from b12x.preparation.device import detect_device

    device = detect_device(require_b12x())
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(triton.knobs.cache, "dir", str(tmp_path))
    with CompilePool(
        device_ordinal=device.ordinal,
        compute_capability=device.identity.compute_capability,
        device_uuid=device.uuid,
        product_name=device.identity.product_name,
        sm_count=device.identity.sm_count,
        max_shared_memory_per_block=device.max_shared_memory_per_block,
        max_shared_memory_per_multiprocessor=device.max_shared_memory_per_multiprocessor,
        workers=1,
    ) as pool:
        descriptions = pool.plan((CompileJob.create(FACTORY),))
        programs = pool.submit_plans(descriptions)
        pool.wait(programs)
        assert all(compiled_program_available(p) for p in programs)
    with forbid_lowering():
        compiled = compile_add_one()
    load_programs(compiled)
    source = torch.arange(128, device="cuda", dtype=torch.float32)
    output = torch.empty_like(source)
    compiled[(1, 1, 1)](source, output, 128)
    torch.testing.assert_close(output, source + 1, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled[(1, 1, 1)](source, output, 128)
    source.mul_(2)
    graph.replay()
    torch.testing.assert_close(output, source + 1, rtol=0, atol=0)
