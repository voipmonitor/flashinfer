"""Typed protocol identities and admission for fixed-grid PCIe collectives."""

from dataclasses import dataclass
from enum import IntEnum


class JointMode(IntEnum):
    """Protocol identities stored in bits 16–23 of each uint32 selector."""

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


def selected_profile(
    implementation,
    max_blocks,
    submode,
    active_blocks,
    *,
    world=None,
    cooperative=False,
    variant_id=None,
):
    """Encode a validated selector; phased profiles admit one active CTA only.

    Restricting phased work to one CTA makes destination ordering independent of
    grid residency, including concurrent streams, MPS, and CUDA green contexts.
    Supply the rank-group size to reject phased profiles at TP2.
    """
    if type(cooperative) is not bool:
        raise ValueError("cooperative must be boolean")
    if variant_id is not None and (
        type(variant_id) is not int or not 0 <= variant_id <= 255
    ):
        raise ValueError("joint variant identity must fit eight bits")
    if cooperative:
        if type(max_blocks) is not int or not 1 <= max_blocks <= 64:
            raise ValueError("cooperative joint profiles require 1–64 CTAs")
        if type(active_blocks) is not int or active_blocks != max_blocks:
            raise ValueError("cooperative physical and active CTA counts must match")
        if submode not in joint_modes(implementation, world):
            raise ValueError(
                "submode does not belong to the prepared joint family and rank group"
            )
        if not joint_mode_spec(submode).phased:
            raise ValueError("cooperative joint profiles require a phased protocol")
        mode, blocks = JOINT_MODES[submode], active_blocks
    else:
        index = profile_index(
            implementation, max_blocks, submode, active_blocks, world=world
        )
        mode, blocks = profile_records(implementation, max_blocks, world=world)[index]
    return ((0 if variant_id is None else variant_id) << 24) | (mode << 16) | blocks


def joint_profile_label(encoded):
    """Render a packed selector using the same mode identities as preparation."""
    spec = _MODE_BY_ID[JointMode((encoded >> 16) & 0xFF)]
    active = encoded & 0xFFFF
    return f"{spec.label} ({active} {'CTA' if active == 1 else 'CTAs'})"


JOINT_MAX_BLOCKS = 64
JOINT_MAX_RANKS = 16
JOINT_EPOCH_OFFSET = JOINT_MAX_BLOCKS * JOINT_MAX_RANKS * (1 + 2 * 32) * 4
JOINT_PHASE_COMPLETIONS_OFFSET = JOINT_EPOCH_OFFSET + 8
JOINT_SIGNAL_BYTES = JOINT_PHASE_COMPLETIONS_OFFSET + 12


@dataclass(frozen=True)
class JointLaunchVariant:
    """One compiled geometry and its admitted runtime protocol/CTA profiles.

    Cooperative grids must fit resident capacity and use every physical CTA.
    Ordinary phased profiles use one active CTA, without a residency dependency.
    """

    family: str
    threads: int
    blocks: int
    profiles: tuple[tuple[str, int], ...]
    cooperative: bool = False

    def __post_init__(self):
        object.__setattr__(self, "profiles", tuple(tuple(p) for p in self.profiles))
        if self.family not in JOINT_FAMILIES:
            raise ValueError("joint variant requires a latency/bandwidth family")
        if type(self.threads) is not int or self.threads not in (32, 64, 128, 256, 512):
            raise ValueError("invalid joint variant thread count")
        if not self.profiles or any(len(p) != 2 for p in self.profiles):
            raise ValueError("joint variant requires nonempty (mode, active CTA) pairs")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("joint variant profiles must be distinct")
        self.validate()

    def validate(self, world=None):
        for mode, active in self.profiles:
            selected_profile(
                self.family,
                self.blocks,
                mode,
                active,
                world=world,
                cooperative=self.cooperative,
            )

    def encoded_values(self, variant_id):
        return tuple(
            selected_profile(
                self.family,
                self.blocks,
                mode,
                active,
                cooperative=self.cooperative,
                variant_id=variant_id,
            )
            for mode, active in self.profiles
        )

    def choice_label(self, mode, active):
        launch = "cooperative" if self.cooperative else "ordinary"
        return (
            f"{self.family}: {joint_mode_spec(mode).label}; {self.threads} threads, "
            f"{self.blocks} physical CTAs, {active} active CTAs; {launch}"
        )


def validate_joint_variants(variants, *, world=None):
    variants = tuple(variants)
    if not variants or len(variants) > 256:
        raise ValueError("joint startup requires 1–256 compiled variants")
    if any(not isinstance(variant, JointLaunchVariant) for variant in variants):
        raise TypeError("joint variants must be JointLaunchVariant objects")
    geometries = [(v.family, v.threads, v.blocks, v.cooperative) for v in variants]
    if len(set(geometries)) != len(geometries):
        raise ValueError("joint compiled variant geometries must be distinct")
    if sum(len(v.profiles) for v in variants) > 512:
        raise ValueError("joint startup admits at most 512 candidates")
    for variant in variants:
        variant.validate(world)
    return variants


def joint_slot_packs(capacity, world):
    """Reserve staging, payload-ready, bulk scatter, and bulk gather regions."""
    if type(capacity) is not int or capacity <= 0 or world not in (2, 3, 4, 6, 8, 10):
        raise ValueError(
            "joint storage requires positive capacity and a supported peer count"
        )
    return (world + 1) * capacity + 2 * world * ((capacity + world - 1) // world)
