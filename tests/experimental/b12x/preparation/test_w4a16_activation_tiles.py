"""W4A16 launch eligibility follows each activation's FC1 output width."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import MoeDecodeConfig, TUNING
from b12x.preparation import DeviceIdentity
from .test_precision_choices import _nvfp4_query


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000 Blackwell")


@pytest.mark.parametrize("activation", ("situ", "silu", "relu2"))
@pytest.mark.parametrize("rows", (2, 4, 6, 8))
def test_native_mxfp4_candidates_follow_fc1_width(activation, rows):
    query = replace(
        _nvfp4_query(),
        quant_mode="w4a16",
        quant_modes=("w4a16",),
        source_format="fp4_e8m0_k32",
        activation=activation,
        num_experts=896,
        hidden_size=3584,
        intermediate_size=192,
        top_k=16,
        num_tokens=rows,
        routed_rows=rows * 16,
        route_num_experts=0,
        w13_layout="w31",
        weight_layouts=("source_native",),
        w4a16_weight_layout="modelopt",
        w4a16_scale_format="e8m0_k32",
        shared_input_scales=False,
        csf_inline_scales=True,
    )
    candidates = {
        config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
    }
    narrow = MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode="packed",
        w4a16_tile_config=(128, 64, 64, 128),
        w4a16_block_size_m=16,
        w4a16_pipeline_stages=3,
    )
    assert narrow in candidates
    paired = replace(narrow, w4a16_tile_config=(64, 128, 64, 128))
    if activation in {"situ", "silu"}:
        assert paired in candidates
        assert TUNING.lower(query, DEVICE, TUNING.config_payload(paired)) == paired
    else:
        assert paired not in candidates
        with pytest.raises(ValueError, match="geometry or shared-memory"):
            TUNING.lower(query, DEVICE, TUNING.config_payload(paired))

    tail = replace(narrow, w4a16_tile_config=(64, 256, 64, 256))
    assert tail not in candidates
    with pytest.raises(ValueError, match="geometry or shared-memory"):
        TUNING.lower(query, DEVICE, TUNING.config_payload(tail))
