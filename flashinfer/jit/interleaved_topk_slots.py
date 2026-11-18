# SPDX-License-Identifier: Apache-2.0
"""Compile stable owner-local top-2048 slot selection for a sixteen-rank CUDA IPC channel."""

from functools import cache
import hashlib

from . import env as jit_env
from .core import gen_jit_spec


@cache
def load_interleaved_topk_slots():
    source = jit_env.FLASHINFER_CSRC_DIR / "interleaved_topk_slots.cu"
    headers = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer" / "comm"
    digest = hashlib.sha256()
    for path in (source, headers / "interleaved_topk_slots.cuh", headers / "interleaved_topk_cutoff.cuh"):
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    spec = gen_jit_spec(
        "interleaved_topk_slots_" + digest.hexdigest()[:24],
        [source], extra_cuda_cflags=["--frandom-seed=31021"], use_fast_math=False,
    )
    return spec.build_and_load()
