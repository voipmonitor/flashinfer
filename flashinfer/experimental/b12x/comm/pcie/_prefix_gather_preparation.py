"""Fixed preparation for a padded score/ID gather on an existing TP16 channel."""

from dataclasses import dataclass, replace
from math import prod

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping
from ._gather_preparation import plan as selected_gather_plan
from ._preparation import _selected_gather_query


def validate_inputs(query, source, output, prefixes, device):
    """Check tensor geometry and aliases without inspecting device metadata."""
    c = query.call
    expected = (
        ("source", source, tuple(c["shape"]), torch.bfloat16),
        ("output", output, tuple(c["out_shape"]), torch.bfloat16),
        ("prefixes", prefixes, tuple(c["prefix_shape"]), torch.int32),
    )
    for name, tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"prefix gather {name} differs from its prepared shape or dtype"
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"prefix gather {name} must be contiguous on {device}")
    spans = [
        (
            tensor.untyped_storage()._cdata,
            tensor.storage_offset() * tensor.element_size(),
            tensor.nbytes,
        )
        for _, tensor, _, _ in expected
    ]
    owner, begin, size = spans[1]
    for other, offset, extent in (spans[0], spans[2]):
        if owner == other and begin < offset + extent and offset < begin + size:
            raise ValueError("prefix gather output overlaps a live input")


@dataclass
class PrefixGatherState:
    query: object
    runtime: object
    launcher: object
    selector: torch.Tensor

    def run(self, *, local_input, out, prefixes):
        runtime = self.runtime
        runtime._check_stream()
        if runtime._closed:
            raise RuntimeError("prefix gather channel is closed")
        validate_inputs(self.query, local_input, out, prefixes, runtime.device)
        if not runtime._device_slot_selection:
            runtime._graph_base_slot = runtime._next_slot & 1
            runtime._device_slot_selection = True
        bank = runtime._graph_base_slot
        self.launcher(
            local_input.data_ptr(),
            out.data_ptr(),
            self.selector.data_ptr(),
            runtime._staging_ptrs[bank],
            runtime._signal_ptrs,
            local_input.shape[0],
            runtime._slot_bytes if bank == 0 else -runtime._slot_bytes,
            runtime._exchange_offset,
            runtime._exchange_counter_offset,
            prefixes.data_ptr(),
        )
        return out


def plan(query, *, runtime, prefix_stride):
    """Declare exact padded gathering, retaining eight-CTA recursive acknowledgements.

    Prefix metadata is a replicated int32 matrix. Columns 0..15 give aligned
    per-peer prefix word counts, column 16 states that all prefixes fill the row,
    and column 17 gives the largest prefix packet count. Counts must lie within
    the row and match across ranks; the omitted suffix is the canonical pair
    (-inf,-1). Callers produce metadata and serialize use of the channel.
    """
    if type(prefix_stride) is not int or prefix_stride < 18:
        raise ValueError("prefix metadata requires at least eighteen int32 columns")
    selected = _selected_gather_query(query, "selected_recursive", (128,))
    if (
        selected is None
        or selected.call["dtype"] != "torch.bfloat16"
        or not 1 <= selected.call["shape"][0] <= 8
    ):
        raise ValueError(
            "prefix gather requires aligned TP16 BF16 rows and the eight-CTA recursive protocol"
        )
    selected = replace(
        selected,
        call=FrozenMapping(
            dict(
                selected.call,
                prefix_shape=(selected.call["shape"][0], prefix_stride),
                prefix_padding_words=(0xFF800000, 0xFFFFFFFF),
            )
        ),
    )
    base = selected_gather_plan(
        selected, runtime=runtime, invocation=FrozenMapping(), override=None
    )
    args = (
        selected.world_size,
        selected.rank,
        prod(selected.call["shape"][1:]) // 2,
        prefix_stride,
    )

    def materialize(selection, device):
        from ._prefix_gather_cute import compile_prefix_gather

        launcher = load_programs(compile_prefix_gather(*args))
        selector = torch.tensor([8], dtype=torch.uint32, device=runtime.device)
        return PrefixGatherState(selected, runtime, launcher, selector)

    return replace(
        base,
        _materialize=materialize,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.pcie._prefix_gather_cute:compile_prefix_gather", *args
            ),
        ),
    )
