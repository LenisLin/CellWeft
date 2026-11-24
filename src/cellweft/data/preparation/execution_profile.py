"""Owner-bound execution profiles for immutable segmentation requests."""
from __future__ import annotations


DEFAULT_PROFILES = {
    "72": {"precision": "fp32", "batch_size": 8},
    "73": {"precision": "bf16", "batch_size": 16},
}
ALLOWED_PROFILES = {("fp32", 8), ("bf16", 16)}


def execution_profile(owner: str, explicit: dict | None = None) -> dict:
    """Resolve an optional frozen profile without changing legacy deliveries."""
    if owner not in DEFAULT_PROFILES:
        raise ValueError("Unsupported delivery owner")
    if explicit is None:
        return dict(DEFAULT_PROFILES[owner])
    if (not isinstance(explicit, dict) or set(explicit) != {"precision", "batch_size"}
            or not isinstance(explicit["precision"], str)
            or type(explicit["batch_size"]) is not int
            or (explicit["precision"], explicit["batch_size"]) not in ALLOWED_PROFILES):
        raise ValueError("Unsupported explicit segmentation execution profile")
    if owner == "72" and explicit != DEFAULT_PROFILES[owner]:
        raise ValueError("Owner 72 retains its fp32/batch8 profile")
    return dict(explicit)
