"""Track packet dependencies and source revisions across handover retries."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

from cellweft.data.io.writers import json_value
from cellweft.data.preparation.inputs import file_revision, same_processing_binding
from cellweft.data.preparation.execution_profile import execution_profile as checked_execution_profile


def refresh_scope_receipt(previous: dict | None, binding: dict) -> dict:
    """Retain delivery progress when only acceptance metadata changed."""
    reusable = previous is not None and same_processing_binding(previous["binding"], binding)
    return {"binding": binding,
            "delivered": previous.get("delivered", {}) if previous else {},
            "delivered_versions": previous.get("delivered_versions", {}) if reusable else {},
            "pending": previous.get("pending", {}) if reusable else {}}


def image_route_shard(image_id: str, shard_count: int) -> int:
    """Keep image ownership stable as a study's accepted scope gains images."""
    if shard_count <= 0:
        raise ValueError("Image shard count must be positive")
    return sum(image_id.encode("utf-8")) % shard_count


def save_handover_state(path: Path, state: dict) -> None:
    """Publish a complete handover state through a private file beside the target."""
    payload = json.dumps(json_value(state), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            exc.add_note(f"Failed to remove private state file {temporary}: {cleanup_error}")
        raise


def delivery_version(binding: dict, data_root: Path, resources: dict[str, dict], weights: dict,
                     *, owner: str | None = None, execution_profile: dict | None = None,
                     allow_unknown_mask_segmentation: bool = False) -> dict:
    """Identify explicit mask evidence and all frozen downstream resources."""
    mask = dict(binding)
    identity = None
    if mask["status"] == "bound":
        source = (data_root / mask["path"]).resolve()
        mask["path"] = str(source.relative_to(data_root.resolve()))
        identity = file_revision(source)
    version = {"mask": mask, "mask_file": identity,
               "resources": resources, "weights": weights}
    if execution_profile is not None:
        if owner is None:
            raise ValueError("Explicit execution profile requires a delivery owner")
        version["execution_profile"] = checked_execution_profile(owner, execution_profile)
    if allow_unknown_mask_segmentation and mask["status"] == "unknown":
        version["mask_policy"] = {"unknown_status": "cpsam_v2"}
    return version


def packet_version(envelope: dict, image_id: str) -> dict:
    """Recover dependencies from receipts created before explicit version tracking."""
    version = {"mask": envelope["mask_bindings"][image_id],
            "mask_file": envelope["requests"][image_id]["mask"].get("file_identity"),
            "resources": {item["replica"]: item["revision"] for item in envelope["files"]
                          if item["replica"].startswith("resources/")},
            "weights": envelope.get("weights_revision")}
    if "execution_profile" in envelope:
        if envelope["execution_profile"] is None:
            raise ValueError("Explicit execution profile cannot be null")
        version["execution_profile"] = checked_execution_profile(
            envelope["node_id"], envelope["execution_profile"])
    policy = envelope.get("mask_policy", {})
    if policy not in ({}, {"unknown_status": "cpsam_v2"}):
        raise ValueError("Unsupported mask policy")
    if policy and envelope["mask_bindings"][image_id]["status"] == "unknown":
        version["mask_policy"] = policy
    return version


def packet_sources_current(envelope: dict, *, study_root: Path | None = None,
                           data_root: Path | None = None,
                           asset_workers: int = 1) -> bool:
    """Check sources, optionally retaining table snapshots under current D01."""
    if type(asset_workers) is not int or not 1 <= asset_workers <= 16:
        raise ValueError("asset_workers must be an integer in 1..16")
    if study_root is not None:
        if data_root is None:
            raise ValueError("Scoped source checks require the canonical data root")
        from cellweft.data.io.publication import resource_lock
        from cellweft.data.preparation.inputs import assert_reviewed_binding
        with resource_lock(study_root, shared=True):
            assert_reviewed_binding(study_root, data_root=data_root,
                                    binding=envelope["source_bindings"],
                                    asset_workers=asset_workers)
    current = True
    for item in envelope["files"]:
        if study_root is not None and Path(item["source"]["path"]) in {
                study_root / "images.parquet", study_root / "review_results.csv"}:
            continue
        if file_revision(Path(item["source"]["path"])) != item["source"]:
            current = False
    return current


def pending_packet_matches(pending: dict, envelope: dict, image_id: str,
                           version: dict, sources_current: bool) -> bool:
    """Decide whether a ready staging packet can keep its delivery ID."""
    return (pending.get("version") == version and
            packet_version(envelope, image_id) == version and sources_current)


def new_pending(staging: Path, queue: Path, study: str, version: dict) -> dict:
    """Allocate a fresh local and remote path for an immutable delivery."""
    delivery_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + uuid.uuid4().hex[:8]
    return {"local": str(staging / delivery_id), "remote": str(queue / study / delivery_id),
            "version": version}


def resume_cursor(state: dict, studies: list[str]) -> dict:
    """Recover the next owned study, giving an unfinished packet first turn."""
    current = state.get("current") or {}
    scopes = state.get("scopes", {})
    if current.get("study") in studies:
        study = current["study"]
        pending_keys = [key for key, receipt in scopes.items()
                        if key.startswith(f"{study}/") and
                        current.get("image_id") in receipt.get("pending", {})]
        if pending_keys:
            return {"study": study, "scope": pending_keys[0], "origin": "current"}
    for study in studies:
        for key, receipt in scopes.items():
            if key.startswith(f"{study}/") and receipt.get("pending"):
                return {"study": study, "scope": key, "origin": "pending"}
    if current.get("study") in studies:
        return {"study": current["study"], "scope": None, "origin": "current"}
    cursor = state.get("cursor") or {}
    if cursor.get("study") in studies:
        return cursor
    published = [(Path(remote).name, study) for study in studies
                 for key, receipt in scopes.items() if key.startswith(f"{study}/")
                 for remote in receipt.get("delivered", {}).values() if isinstance(remote, str)]
    if published:
        # Delivery IDs identify allocation order, not completion order.
        return {"study": max(published)[1], "scope": None,
                "origin": "legacy_published_creation_hint"}
    return {"study": studies[0], "scope": None, "origin": "initial"}


def next_study_cursor(studies: list[str], study: str) -> dict:
    """Advance after a study has no more scopes to visit in this pass."""
    return {"study": studies[(studies.index(study) + 1) % len(studies)], "scope": None}


def image_scope_owners(scopes: list[dict], scope_keys: list[str],
                       receipts: dict[str, dict]) -> dict[str, str]:
    """Select one current scope per physical image, retaining pending ownership."""
    owners = {image: key for scope, key in zip(scopes, scope_keys, strict=True)
              for image in scope["binding"]["image_ids"]}
    pinned = set()
    for scope, key in zip(scopes, scope_keys, strict=True):
        receipt = receipts.get(key, {})
        if receipt.get("binding") != scope["binding"]:
            continue
        for image in scope["binding"]["image_ids"]:
            if image in receipt.get("pending", {}) and image not in pinned:
                owners[image] = key
                pinned.add(image)
    return owners


def pending_first(remaining: list[str], pending: dict,
                  masks: dict[str, dict] | None = None) -> str:
    """Resume pending first, then stable-partition current images by mask evidence."""
    for image in remaining:
        if image in pending:
            return image
    masks = masks or {}
    return next((image for image in remaining
                 if masks.get(image, {}).get("status") in {"bound", "absent", "unusable"}),
                remaining[0])
