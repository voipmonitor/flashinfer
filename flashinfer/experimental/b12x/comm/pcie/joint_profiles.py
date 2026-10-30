"""CPU-only joint collective mode metadata, selector admission, and labels."""

from ._joint_profile import (
    JOINT_MODE_SPECS,
    JointMode,
    JointModeSpec,
    joint_mode_spec,
    joint_modes,
    joint_profile_label,
    selected_profile,
)

__all__ = [
    "JOINT_MODE_SPECS",
    "JointMode",
    "JointModeSpec",
    "joint_mode_spec",
    "joint_modes",
    "joint_profile_label",
    "selected_profile",
]
