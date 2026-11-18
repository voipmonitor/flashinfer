"""Prepared paged resources follow the selected kernel's address dependencies."""

from types import SimpleNamespace

import pytest
import torch

from b12x.attention.paged import _preparation
from b12x.attention.paged._forward import _PagedLaunchers
from b12x.preparation import FrozenMapping


class ScratchPlan:
    _q2k_indices_data_ptr = None

    def __init__(self, **options):
        self.options = dict(k_descale=None, v_descale=None, attention_sink_bias=None)
        self.options.update(options)

    def bind(self, *, metadata_launchers, k_cache, v_cache):
        return SimpleNamespace(
            k_cache=k_cache,
            v_cache=v_cache,
            scratch=SimpleNamespace(plan=SimpleNamespace(num_kv_heads=2)),
            **self.options,
        )


@pytest.mark.parametrize(
    "resource_kind",
    [
        None,
        "use_paged_kv_tma_raw_desc_issue",
        "use_paged_kv_tma_fp8_raw_issue",
        "k_descale",
        "v_descale",
        "sink_dtype",
        "sink_stride",
    ],
)
def test_capture_rebinds_only_constant_resources(monkeypatch, resource_kind):
    address_field = (
        resource_kind
        if resource_kind and resource_kind.startswith("use_paged")
        else None
    )
    kernel = SimpleNamespace(**({address_field: True} if address_field else {}))
    options = {}
    if resource_kind in ("k_descale", "v_descale"):
        options[resource_kind] = torch.ones(4, 2)[:, :1]
    elif resource_kind == "sink_dtype":
        options["attention_sink_bias"] = torch.ones(4, dtype=torch.bfloat16)
    elif resource_kind == "sink_stride":
        options["attention_sink_bias"] = torch.ones(8)[::2]
    launchers = _PagedLaunchers(controls={}, forward_kernel=kernel)
    state = _preparation._PagedState(
        scratch_plan=ScratchPlan(**options),
        config=None,
        programs={"route": launchers},
        metadata={},
        controls=FrozenMapping({}),
    )
    monkeypatch.setattr(_preparation, "variant_key", lambda binding: "route")
    if address_field:
        monkeypatch.setattr(
            _preparation,
            "materialize_paged_resources",
            lambda **kwargs: {
                "plane_tma_descs": (
                    torch.empty(2, 16),
                    torch.empty(2, 16),
                    torch.zeros(2),
                    torch.zeros(2),
                ),
                "optional_buffers": {},
            },
        )
    first_key, first_value, second_key, second_value = (
        torch.empty((2, 128, 2, 64), dtype=torch.bfloat16) for _ in range(4)
    )
    first = state.bind(k_cache=first_key, v_cache=first_value)
    resources = first.scratch._prepared_plane_tma_descs
    monkeypatch.setattr(_preparation, "_capturing", lambda: True)
    monkeypatch.setattr(
        _preparation,
        "materialize_paged_resources",
        lambda **kwargs: pytest.fail("capture must not materialize paged resources"),
    )
    repeated = state.bind(_preparing=False, k_cache=first_key, v_cache=first_value)
    assert repeated.scratch._prepared_plane_tma_descs is resources
    if resource_kind is not None:
        with pytest.raises(ValueError, match="resources cannot be created during CUDA"):
            state.bind(_preparing=False, k_cache=second_key, v_cache=second_value)
    else:
        rebound = state.bind(_preparing=False, k_cache=second_key, v_cache=second_value)
        assert rebound.k_cache is second_key and rebound.v_cache is second_value
        assert rebound.scratch._prepared_plane_tma_descs is resources
        assert torch.count_nonzero(resources[2]) == 0
    assert len(state.resources) == 1


def test_constant_resource_layout_does_not_hide_sink_conversion(monkeypatch):
    scratch = ScratchPlan(attention_sink_bias=torch.ones(4))
    state = _preparation._PagedState(
        scratch_plan=scratch,
        config=None,
        programs={
            "route": _PagedLaunchers(controls={}, forward_kernel=SimpleNamespace())
        },
        metadata={},
        controls=FrozenMapping({}),
    )
    monkeypatch.setattr(_preparation, "variant_key", lambda binding: "route")
    key, value = (torch.empty((2, 128, 2, 64), dtype=torch.bfloat16) for _ in range(2))
    state.bind(k_cache=key, v_cache=value)
    scratch.options["attention_sink_bias"] = torch.ones(4, dtype=torch.bfloat16)
    monkeypatch.setattr(_preparation, "_capturing", lambda: True)
    with pytest.raises(ValueError, match="resources cannot be created during CUDA"):
        state.bind(_preparing=False, k_cache=key.clone(), v_cache=value.clone())
