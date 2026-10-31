"""Resident cooperative grids for serialized destination-phase PCIe transfers."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ._joint_cute import _JointLaunch
from ._joint_profile import JOINT_PHASE_COMPLETIONS_OFFSET
from ._cute_intrinsics import threadfence_sys


@dsl_user_op
def phase_barrier(signal: Int64, blocks: Uint32, *, loc=None, ip=None):
    """One leader per co-resident CTA arrives after all its writers fence.

    Arrival and generation occupy padding in the aligned IPC signal allocation.
    The final arrival resets the count before publishing a release generation.
    This wait is legal only under cooperative launch with a resident grid.
    """
    llvm.inline_asm(
        None,
        [
            Int64(signal).ir_value(loc=loc, ip=ip),
            Uint32(blocks).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .u64 arrived_addr, released_addr;
            .reg .u32 generation, arrived, last, seen, next_generation, ignored;
            .reg .pred final, done;
            add.u64 arrived_addr, $0, 4;
            add.u64 released_addr, $0, 8;
            ld.acquire.gpu.global.u32 generation, [released_addr];
            atom.acq_rel.gpu.global.add.u32 arrived, [arrived_addr], 1;
            sub.u32 last, $1, 1;
            setp.eq.u32 final, arrived, last;
            @!final bra WAIT_PHASE;
            st.relaxed.gpu.global.u32 [arrived_addr], 0;
            atom.relaxed.gpu.global.add.u32 ignored, [$0], 1;
            add.u32 next_generation, generation, 1;
            st.release.gpu.global.u32 [released_addr], next_generation;
            bra DONE_PHASE;
        WAIT_PHASE:
            ld.acquire.gpu.global.u32 seen, [released_addr];
            setp.ne.u32 done, seen, generation;
            @!done bra WAIT_PHASE;
        DONE_PHASE:
        }""",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _CooperativeJointLaunch(_JointLaunch):
    @cute.jit
    def __call__(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        local_signal: cute.Pointer,
        selector: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        size: Int64,
        capacity: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            peers, signals, local_signal, selector, payload, output, size, capacity
        ).launch(
            grid=(self._max_blocks, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            cooperative=True,
            stream=stream,
        )

    @cute.jit
    def _finish_write_phase(self, local_signal: Int64):
        threadfence_sys()
        cute.arch.sync_threads()
        tid, _, _ = cute.arch.thread_idx()
        if tid == Int32(0):
            phase_barrier(
                local_signal + Int64(JOINT_PHASE_COMPLETIONS_OFFSET),
                Uint32(self._max_blocks),
            )
        cute.arch.sync_threads()
