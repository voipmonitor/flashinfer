"""Typed protocol identities and admission for fixed-grid PCIe collectives."""

from dataclasses import dataclass
from enum import IntEnum


class JointMode(IntEnum):
    """Protocol identities stored in the upper half of each uint32 selector."""

    FULL = 0
    PAYLOAD_RSAG = 1
    BULK_PUSH = 2
    BULK_PULL = 3
    PULL = 4
    FULL_PHASED = 5
    PAYLOAD_RSAG_PHASED = 6
    BULK_PUSH_PHASED = 7


@dataclass(frozen=True)
class JointModeSpec:
    """Family membership, arithmetic protocol, and labels for one mode."""

    mode: JointMode
    name: str
    label: str
    families: tuple[str, ...]
    base: JointMode
    phased: bool = False


JOINT_MODE_SPECS = (
    JointModeSpec(JointMode.FULL, "full", "Full", ("joint_latency",), JointMode.FULL),
    JointModeSpec(
        JointMode.PAYLOAD_RSAG,
        "payload_rsag",
        "Payload RSAG",
        ("joint_latency", "joint_bandwidth"),
        JointMode.PAYLOAD_RSAG,
    ),
    JointModeSpec(
        JointMode.BULK_PUSH,
        "bulk_push",
        "Bulk push",
        ("joint_bandwidth",),
        JointMode.BULK_PUSH,
    ),
    JointModeSpec(
        JointMode.BULK_PULL,
        "bulk_pull",
        "Bulk pull",
        ("joint_bandwidth",),
        JointMode.BULK_PULL,
    ),
    JointModeSpec(
        JointMode.PULL,
        "pull",
        "Pull",
        ("joint_latency", "joint_bandwidth"),
        JointMode.PULL,
    ),
    JointModeSpec(
        JointMode.FULL_PHASED,
        "full_phased",
        "Phased full",
        ("joint_latency",),
        JointMode.FULL,
        True,
    ),
    JointModeSpec(
        JointMode.PAYLOAD_RSAG_PHASED,
        "payload_rsag_phased",
        "Phased payload RSAG",
        ("joint_latency", "joint_bandwidth"),
        JointMode.PAYLOAD_RSAG,
        True,
    ),
    JointModeSpec(
        JointMode.BULK_PUSH_PHASED,
        "bulk_push_phased",
        "Phased bulk push",
        ("joint_bandwidth",),
        JointMode.BULK_PUSH,
        True,
    ),
)
JOINT_MODES = {spec.name: spec.mode for spec in JOINT_MODE_SPECS}
JOINT_FAMILIES = {
    family: tuple(spec.name for spec in JOINT_MODE_SPECS if family in spec.families)
    for family in ("joint_latency", "joint_bandwidth")
}
_MODE_BY_NAME = {spec.name: spec for spec in JOINT_MODE_SPECS}
_MODE_BY_ID = {spec.mode: spec for spec in JOINT_MODE_SPECS}


def joint_modes(implementation, world=None):
    """Return protocol names; TP2 excludes redundant phased transfers."""
    if implementation not in JOINT_FAMILIES:
        raise ValueError("joint profiles require a latency/bandwidth family")
    return tuple(
        name
        for name in JOINT_FAMILIES[implementation]
        if world != 2 or not _MODE_BY_NAME[name].phased
    )


def joint_mode_spec(name):
    """Describe a named protocol without initializing CUDA."""
    return _MODE_BY_NAME[name]


def profile_records(implementation, max_blocks, *, world=None):
    if type(max_blocks) is not int or not 1 <= max_blocks <= 64:
        raise ValueError(
            "joint profiles require a latency/bandwidth family and 1–64 CTAs"
        )
    return tuple(
        (JOINT_MODES[mode], blocks)
        for mode in joint_modes(implementation, world)
        for blocks in range(1, (1 if _MODE_BY_NAME[mode].phased else max_blocks) + 1)
    )


def profile_index(implementation, max_blocks, submode, active_blocks, *, world=None):
    if type(active_blocks) is not int or not 1 <= active_blocks <= max_blocks:
        raise ValueError("active CTA count must fit the prepared maximum grid")
    if submode not in joint_modes(implementation, world):
        raise ValueError(
            "submode does not belong to the prepared joint family and rank group"
        )
    if _MODE_BY_NAME[submode].phased and active_blocks != 1:
        raise ValueError("phased joint profiles require exactly one active CTA")
    return profile_records(implementation, max_blocks, world=world).index(
        (JOINT_MODES[submode], active_blocks)
    )


def selected_profile(implementation, max_blocks, submode, active_blocks, *, world=None):
    """Encode a validated selector; phased profiles admit one active CTA only.

    Restricting phased work to one CTA makes destination ordering independent of
    grid residency, including concurrent streams, MPS, and CUDA green contexts.
    Supply the rank-group size to reject phased profiles at TP2.
    """
    index = profile_index(
        implementation, max_blocks, submode, active_blocks, world=world
    )
    mode, blocks = profile_records(implementation, max_blocks, world=world)[index]
    return (mode << 16) | blocks


def joint_profile_label(encoded):
    """Render a packed selector using the same mode identities as preparation."""
    spec = _MODE_BY_ID[JointMode(encoded >> 16)]
    active = encoded & 0xFFFF
    return f"{spec.label} ({active} {'CTA' if active == 1 else 'CTAs'})"


JOINT_MAX_BLOCKS = 64
JOINT_MAX_RANKS = 16
JOINT_EPOCH_OFFSET = JOINT_MAX_BLOCKS * JOINT_MAX_RANKS * (1 + 2 * 32) * 4
JOINT_PHASE_COMPLETIONS_OFFSET = JOINT_EPOCH_OFFSET + 8
JOINT_SIGNAL_BYTES = JOINT_PHASE_COMPLETIONS_OFFSET + 4


def joint_slot_packs(capacity, world):
    """Reserve staging, payload-ready, bulk scatter, and bulk gather regions."""
    if type(capacity) is not int or capacity <= 0 or world not in (2, 3, 4, 6, 8, 10):
        raise ValueError(
            "joint storage requires positive capacity and a supported peer count"
        )
    return (world + 1) * capacity + 2 * world * ((capacity + world - 1) // world)
