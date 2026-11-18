"""Compressed scale declarations release storage and bind invocation-local planes."""

from copy import copy
from dataclasses import dataclass
import gc
from types import MappingProxyType, SimpleNamespace
import weakref

import pytest
import torch

from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from b12x.moe.fused_moe import _binding_ops as ops, _impl
from b12x.moe.fused_moe import _token_outer
from b12x.moe.fused_moe._preparation import _FusedMoeCapacityState, _FusedMoeState
from b12x.moe.fused_moe._token_outer import State


@dataclass(frozen=True)
class Scales:
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    micro_w13_scale: torch.Tensor
    micro_w2_scale: torch.Tensor


@dataclass(frozen=True)
class Representation:
    value: Scales


@dataclass(frozen=True)
class Experts:
    _impl: object
    device: torch.device = torch.device("cpu")


def test_release_declaration_scales_preserves_runtime_aliases(monkeypatch):
    first, second = torch.empty(32), torch.empty(16)
    active = torch.empty(8, dtype=torch.int32)
    references = [weakref.ref(tensor) for tensor in (first, second, active)]
    impl = SimpleNamespace(
        w1_fp4=torch.empty(16, dtype=torch.uint8),
        w2_fp4=torch.empty(16, dtype=torch.uint8),
        w1_blockscale=first,
        w2_blockscale=second,
        nvfp4_csf=Nvfp4CsfDecoder(None, None, (), (), active),
        representation=Representation(Scales(first, second, first[4:12], second[2:6])),
    )
    layouts = ops._scale_templates(impl)
    execution = copy(impl)
    experts = SimpleNamespace(_impl=impl)
    state = _FusedMoeState(
        experts, None, None, None, None, None, execution_experts=execution
    )
    capacity = _FusedMoeCapacityState(MappingProxyType({1: state, 2: state}))
    monkeypatch.setattr(ops, "require_prepared", lambda *args: capacity)
    key = (impl.w1_fp4.data_ptr(), 0, impl.w2_fp4.data_ptr(), 0)
    monkeypatch.setattr(_impl, "_WEIGHT_CACHE", {key: (first, second)})
    monkeypatch.setattr(_impl, "_LAST_WEIGHTS", (key, (first, second)))
    del first, second, active
    ops.release_scale_templates(experts, object())
    ops.release_scale_templates(experts, object())
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not _impl._WEIGHT_CACHE and _impl._LAST_WEIGHTS == (None, None)
    assert impl.w1_blockscale.is_meta and execution.w2_blockscale.is_meta
    buffers = [
        torch.empty_strided(
            layout.shape, layout.stride, dtype=layout.dtype, device=layout.device
        ).view(torch.uint8)
        for layout in layouts.planes
    ]
    buffers.append(torch.empty(layouts.active.shape, dtype=layouts.active.dtype))
    runtime = ops._scale_view(impl, buffers)
    assert runtime.w1_blockscale.data_ptr() == buffers[0].data_ptr()
    assert (
        runtime.representation.value.micro_w13_scale.data_ptr()
        == buffers[0].data_ptr() + 16
    )
    assert (
        runtime.representation.value.micro_w2_scale.data_ptr()
        == buffers[1].data_ptr() + 8
    )
    with pytest.raises(ValueError, match="preserve prepared tensor metadata"):
        ops._scale_view(impl, [buffers[0][:-4], *buffers[1:]])


def test_token_outer_releases_separate_routing_storage(monkeypatch):
    first, second = torch.empty(32), torch.empty(16)
    active, token_active = (
        torch.empty(8, dtype=torch.int32),
        torch.empty(8, dtype=torch.int32),
    )
    references = [
        weakref.ref(tensor) for tensor in (first, second, active, token_active)
    ]
    impl = SimpleNamespace(
        w1_fp4=torch.empty(16, dtype=torch.uint8),
        w2_fp4=torch.empty(16, dtype=torch.uint8),
        w1_blockscale=first,
        w2_blockscale=second,
        representation=None,
        nvfp4_csf=Nvfp4CsfDecoder(None, None, (), (), active),
    )
    experts = SimpleNamespace(_impl=impl)
    state = State(
        experts,
        None,
        None,
        None,
        scale_decoder=Nvfp4CsfDecoder(None, None, (), (), token_active),
    )
    monkeypatch.setattr(ops, "require_prepared", lambda *args: state)
    del first, second, active, token_active
    ops.release_scale_templates(experts, object())
    ops.release_scale_templates(experts, object())
    gc.collect()
    assert all(reference() is None for reference in references)
    assert state.scale_decoder.active.is_meta


def test_token_outer_binds_live_scale_planes_after_declaration_release(monkeypatch):
    first, second = torch.empty(32), torch.empty(16)
    active = torch.empty(8, dtype=torch.int32)
    impl = SimpleNamespace(
        w1_fp4=torch.empty(16, dtype=torch.uint8),
        w2_fp4=torch.empty(16, dtype=torch.uint8),
        w1_blockscale=first,
        w2_blockscale=second,
        representation=None,
        nvfp4_csf=Nvfp4CsfDecoder(None, None, (), (), active),
    )
    experts = Experts(impl)
    query = SimpleNamespace(
        hidden=8,
        tokens=2,
        top_k=3,
        dtype="bfloat16",
        output_layout="token",
        route_experts=0,
    )
    state = State(experts, query, None, None, scale_decoder=impl.nvfp4_csf)
    monkeypatch.setattr(ops, "require_prepared", lambda *args: state)
    ops.release_scale_templates(experts, object())
    monkeypatch.setattr(State, "scratch_specs", lambda self: ())
    monkeypatch.setattr(
        _token_outer, "scratch_tensor", lambda *args, **kwargs: torch.empty(0)
    )
    monkeypatch.setattr(_token_outer, "_layout", lambda *args: ((), 0))
    buffers = [
        torch.empty_like(first).view(torch.uint8),
        torch.empty_like(second).view(torch.uint8),
        torch.zeros_like(active),
    ]
    binding = state.bind(
        a=torch.empty((2, 8), dtype=torch.bfloat16),
        topk_ids=torch.zeros((2, 3), dtype=torch.int32),
        topk_weights=torch.ones((2, 3)),
        output=torch.empty((2, 8), dtype=torch.bfloat16),
        scratch=torch.empty(0),
        _scale_buffers=buffers,
    )
    assert binding.state is not state and state.experts._impl.w1_blockscale.is_meta
    assert binding.state.experts._impl.w1_blockscale.data_ptr() == buffers[0].data_ptr()
    assert binding.state.scale_decoder.active is buffers[2]


def test_reusable_experts_rematerialize_and_release_declaration_planes(monkeypatch):
    first, second = torch.empty(32), torch.empty(16)
    impl = SimpleNamespace(
        w1_fp4=torch.empty(16, dtype=torch.uint8),
        w2_fp4=torch.empty(16, dtype=torch.uint8),
        w1_blockscale=first,
        w2_blockscale=second,
        representation=Representation(Scales(first, second, first[4:12], second[2:6])),
        nvfp4_csf=Nvfp4CsfDecoder(
            None, None, (), (), torch.empty(8, dtype=torch.int32)
        ),
    )
    experts = Experts(impl)
    state = _FusedMoeState(experts, None, None, None, None, None)
    monkeypatch.setattr(ops, "require_prepared", lambda *args: state)
    del first, second
    for _ in range(2):
        ops.release_scale_templates(experts, object())
        assert impl.w1_blockscale.is_meta
        ops.materialize_scale_templates(experts)
        assert not impl.w1_blockscale.is_meta and not impl.w2_blockscale.is_meta
        assert impl.nvfp4_csf.active.eq(0).all()
        payload = impl.representation.value
        assert payload.micro_w13_scale.data_ptr() == impl.w1_blockscale.data_ptr() + 16
        assert payload.micro_w2_scale.data_ptr() == impl.w2_blockscale.data_ptr() + 8
        references = [
            weakref.ref(tensor)
            for tensor in (
                impl.w1_blockscale,
                impl.w2_blockscale,
                impl.nvfp4_csf.active,
            )
        ]
        del payload
        ops.release_scale_templates(experts, object())
        gc.collect()
        assert all(reference() is None for reference in references)
