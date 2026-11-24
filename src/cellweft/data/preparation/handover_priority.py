"""Order feeder study visits using persisted mask delivery receipts."""
from __future__ import annotations


KNOWN_MASK_STATUSES = {"bound", "absent", "unusable"}
UNKNOWN_MASK_POLICY = {"unknown_status": "cpsam_v2"}


def prefer_mask_updates(
    rotated_studies: list[str],
    scopes: dict[str, dict],
    masks_by_study: dict[str, dict[str, dict]],
    pending_study: str | None = None,
    *,
    allow_unknown_mask_segmentation: bool = False,
) -> list[str]:
    """Order work by resumability, policy updates, mask updates, and new input.

    Unknown-mask policy changes are versioned delivery changes even though the
    biological mask binding is unchanged.  They therefore need their own
    priority bucket when the route enables self-segmentation.
    """
    needs_update = set()
    needs_policy_update = set()
    needs_new = set()
    has_pending = set()
    for key, receipt in scopes.items():
        study = key.split("/", 1)[0]
        if receipt.get("pending"):
            has_pending.add(study)
        masks = masks_by_study.get(study, {})
        delivered = receipt.get("delivered", {})
        delivered_versions = receipt.get("delivered_versions", {})
        for image in receipt["binding"]["image_ids"]:
            mask = masks.get(image)
            if mask is None:
                if not allow_unknown_mask_segmentation:
                    continue
                # A missing registry row is an unresolved source status, not
                # evidence that the image has no mask.  The opt-in route
                # already freezes this as the cpsam_v2 policy at delivery.
                mask = {"status": "unknown"}
            if image not in delivered:
                # New D01 input is useful work regardless of author-mask state;
                # unknown images remain subject to the route's unknown burst.
                needs_new.add(study)
                continue
            if (allow_unknown_mask_segmentation and mask.get("status") == "unknown"
                    and delivered_versions.get(image, {}).get("mask_policy")
                    != UNKNOWN_MASK_POLICY):
                needs_policy_update.add(study)
            if mask.get("status") in KNOWN_MASK_STATUSES and (
                    delivered_versions.get(image, {}).get("mask") != mask):
                needs_update.add(study)
    first = [pending_study] if pending_study in rotated_studies else []
    remaining = [study for study in rotated_studies if study not in first]
    pending = [study for study in remaining if study in has_pending]
    remaining = [study for study in remaining if study not in has_pending]
    policy_updates = [study for study in remaining if study in needs_policy_update]
    remaining = [study for study in remaining if study not in needs_policy_update]
    mask_updates = [study for study in remaining if study in needs_update]
    remaining = [study for study in remaining if study not in needs_update]
    new_input = [study for study in remaining if study in needs_new]
    stable = [study for study in remaining if study not in needs_new]
    return first + pending + policy_updates + mask_updates + new_input + stable
