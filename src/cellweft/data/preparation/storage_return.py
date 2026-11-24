"""Single-image return of accepted downstream products to formal NAS storage.

The return contract is intentionally narrower than preparation: source and
target image directories are explicit, only the six downstream products move,
and the source metadata/request is carried forward without scientific edits.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_write_lease


RESERVE_BYTES = 100 * 1024**3
COPY_CHUNK_BYTES = 4 * 1024**2
COPY_SYNC_BYTES = 64 * 1024**2
PRODUCT_FILENAMES = {
    "image": "image.tiff",
    "channels": "channels.parquet",
    "mask": "mask.tiff",
    "cells": "cells.parquet",
    "expression": "expression.parquet",
    "neighbors": "neighbors.parquet",
}
PRODUCT_KEYS = tuple(PRODUCT_FILENAMES)


class StorageReturnError(ValueError):
    """The accepted evidence or the formal product contract is invalid."""


class StorageReturnCapacityError(OSError):
    """The image cannot be returned while preserving the required reserve."""


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _revision(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StorageReturnError(f"Expected a regular file: {path}")
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _version(value: Any, label: str) -> dict[str, Any]:
    """Validate one current frozen revision from the accepted index."""
    if not isinstance(value, Mapping):
        raise StorageReturnError(f"{label} must contain path, size and mtime_ns")
    path = value.get("path")
    size = value.get("size")
    mtime_ns = value.get("mtime_ns")
    if not isinstance(path, str) or not path:
        raise StorageReturnError(f"{label}.path is required")
    if type(size) is not int or size < 0 or type(mtime_ns) is not int or mtime_ns < 0:
        raise StorageReturnError(f"{label} has invalid size/mtime_ns")
    return {"path": str(Path(path).resolve()), "size": size, "mtime_ns": mtime_ns}


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise StorageReturnError(f"{label} escapes image directory: {path}") from error
    return resolved


def _metadata_evidence(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if "metadata_revision" not in record:
        raise StorageReturnError("accepted_record.metadata_revision is required")
    current = _version(record["metadata_revision"], "metadata_revision")
    original = record.get("original_accepted_metadata_revision")
    if original is not None:
        original = _version(original, "original_accepted_metadata_revision")
    return current, original


def _input_identity(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "source_input_identity" not in record:
        raise StorageReturnError("accepted_record.source_input_identity is required")
    raw = record["source_input_identity"]
    if not isinstance(raw, list):
        raise StorageReturnError("source_input_identity must be a list")
    values = raw
    return [_version(value, "source_input_identity") for value in values]


def _source_delivery(metadata: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[bool, Any]:
    if "source_delivery" in metadata:
        return True, copy.deepcopy(metadata["source_delivery"])
    if "source_delivery" in record:
        return True, copy.deepcopy(record["source_delivery"])
    return False, None


@contextmanager
def _source_publication_read_lock(source_dir: Path, *, formal_path: Path,
                                  io_lock_root: Path | None,
                                  data_root: Path | None) -> Iterator[None]:
    """Acquire the NAS lease first, then open the existing source lock read-only."""
    with formal_write_lease(formal_path, io_lock_root=io_lock_root, data_root=data_root):
        with _source_publication_read_lock_only(source_dir):
            yield


@contextmanager
def _source_publication_read_lock_only(source_dir: Path) -> Iterator[None]:
    """Open an existing source lock read-only; never create a source lock."""
    path = source_dir / ".publication.lock"
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        raise StorageReturnError(f"Existing source .publication.lock is required: {path}") from error
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageReturnError(f"Invalid metadata JSON: {path}") from error
    if not isinstance(value, dict):
        raise StorageReturnError(f"Metadata must be an object: {path}")
    return value


def _input_identity_check(identities: list[dict[str, Any]]) -> None:
    for expected in identities:
        actual = _revision(Path(expected["path"]))
        if actual != expected:
            raise StorageReturnError(f"Source input identity changed: {expected['path']}")


def _source_info(source_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    metadata_path = source_dir / "metadata.json"
    expected_metadata, original_metadata = _metadata_evidence(record)
    actual_metadata = _revision(metadata_path)
    if actual_metadata != expected_metadata:
        raise StorageReturnError("Current source metadata differs from accepted evidence")
    metadata = _read_json(metadata_path)
    if record.get("study") is not None and metadata.get("study") != record["study"]:
        raise StorageReturnError("Source metadata study differs from accepted record")
    if record.get("image_id") is not None and metadata.get("image_id") != record["image_id"]:
        raise StorageReturnError("Source metadata image_id differs from accepted record")
    if "request" in record and metadata.get("request") != record["request"]:
        raise StorageReturnError("Source scientific request differs from accepted evidence")
    if "request" not in metadata or "request_digest" not in record:
        raise StorageReturnError("Source request and accepted request_digest are required")
    if _stable_digest(metadata["request"]) != record["request_digest"]:
        raise StorageReturnError("Source request does not match accepted request_digest")
    if metadata.get("publication_incomplete"):
        raise StorageReturnError("Source metadata is publication_incomplete")

    completed = set(metadata.get("completed", []))
    if not set(PRODUCT_KEYS).issubset(completed):
        raise StorageReturnError("Source metadata does not declare all six products complete")
    declared = metadata.get("products")
    accepted_products = record.get("product_revisions")
    if not isinstance(declared, Mapping) or not isinstance(accepted_products, Mapping):
        raise StorageReturnError("accepted_record.product_revisions must contain all six products")

    products: dict[str, dict[str, Any]] = {}
    original_products: dict[str, dict[str, Any]] = {}
    for key, filename in PRODUCT_FILENAMES.items():
        if key not in declared or key not in accepted_products:
            raise StorageReturnError(f"Missing accepted product evidence: {key}")
        declared_path = Path(declared[key])
        product_path = declared_path if declared_path.is_absolute() else source_dir / declared_path
        product_path = _inside(product_path, source_dir, f"products.{key}")
        if product_path.name != filename:
            raise StorageReturnError(f"Unexpected source filename for {key}: {product_path.name}")
        accepted = _version(accepted_products[key], f"products.{key}")
        if accepted["path"] != str(product_path):
            raise StorageReturnError(f"Accepted product path differs from metadata: {key}")
        actual = _revision(product_path)
        if actual != accepted:
            raise StorageReturnError(f"Current source product differs from accepted evidence: {key}")
        products[key] = actual
        original_map = record.get("original_accepted_product_revisions", {})
        original = original_map.get(key) if isinstance(original_map, Mapping) else None
        original_products[key] = original or actual

    identities = _input_identity(record)
    _input_identity_check(identities)
    has_delivery, delivery = _source_delivery(metadata, record)
    return {
        "metadata": metadata,
        "metadata_revision": actual_metadata,
        "original_metadata_revision": original_metadata,
        "products": products,
        "original_products": original_products,
        "request": copy.deepcopy(metadata.get("request")),
        "request_digest": record["request_digest"],
        "source_input_identity": identities,
        "has_source_delivery": has_delivery,
        "source_delivery": delivery,
    }


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _partial_path(path: Path, tag: str) -> Path:
    return path.with_name(f".{path.name}.{tag}.{os.getpid()}.{uuid.uuid4().hex}.partial")


def _atomic_json(path: Path, value: Mapping[str, Any], *, tag: str) -> None:
    partial = _partial_path(path, tag)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        partial.unlink(missing_ok=True)


def _append_manifest(path: Path, entry: Mapping[str, Any]) -> None:
    if path.suffix == ".jsonl":
        previous: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        previous.append(value)
        identity = (entry.get("target_metadata", {}).get("path"), entry.get("source_metadata", {}).get("path"))
        if any((item.get("target_metadata", {}).get("path"), item.get("source_metadata", {}).get("path")) == identity
               and item.get("source_metadata", {}).get("mtime_ns") == entry.get("source_metadata", {}).get("mtime_ns")
               for item in previous):
            return
        partial = _partial_path(path, "storage_return_manifest")
        try:
            with partial.open("wb") as handle:
                for item in previous:
                    handle.write((json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                handle.write((json.dumps(dict(entry), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
            _fsync_directory(path.parent)
        finally:
            partial.unlink(missing_ok=True)
        return
    previous: Any = []
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageReturnError(f"Migration manifest is not valid JSON: {path}") from error
    if isinstance(previous, dict):
        entries = list(previous.get("entries", []))
        wrapper = {key: value for key, value in previous.items() if key != "entries"}
        wrapper["entries"] = entries
    elif isinstance(previous, list):
        wrapper = {"schema": "cellweft.storage_return_manifest.v1", "entries": previous}
        entries = previous
    else:
        raise StorageReturnError("Migration manifest must be an object or list")
    identity = (entry.get("target_metadata", {}).get("path"), entry.get("source_metadata", {}).get("path"))
    if not any((item.get("target_metadata", {}).get("path"), item.get("source_metadata", {}).get("path")) == identity
               and item.get("source_metadata", {}).get("mtime_ns") == entry.get("source_metadata", {}).get("mtime_ns")
               for item in entries if isinstance(item, Mapping)):
        entries.append(dict(entry))
    _atomic_json(path, wrapper, tag="storage_return_manifest")


def _target_metadata_reusable(current: Mapping[str, Any], source: Mapping[str, Any],
                              target_dir: Path, request: Any) -> bool:
    if not _target_products_reusable(current, source, target_dir, request):
        return False
    consolidation = current.get("storage_consolidation")
    return isinstance(consolidation, Mapping) and consolidation.get("source_metadata") == source["metadata_revision"]


def _target_products_reusable(current: Mapping[str, Any], source: Mapping[str, Any],
                              target_dir: Path, request: Any) -> bool:
    if current.get("publication_incomplete") or not set(current.get("completed", [])) >= set(PRODUCT_KEYS):
        return False
    statuses = current.get("product_status")
    if isinstance(statuses, Mapping) and any(statuses.get(key) != "complete" for key in PRODUCT_KEYS):
        return False
    if request is not None and current.get("request") != request:
        return False
    consolidation = current.get("storage_consolidation")
    if not isinstance(consolidation, Mapping) or consolidation.get("status") not in {"received", "complete"}:
        return False
    if consolidation.get("source_products") != source["products"]:
        return False
    target_products = consolidation.get("target_products")
    declared = current.get("products")
    if not isinstance(target_products, Mapping) or not isinstance(declared, Mapping):
        return False
    for key, filename in PRODUCT_FILENAMES.items():
        target = target_dir / filename
        if declared.get(key) != str(target.resolve()):
            return False
        expected = target_products.get(key)
        if not isinstance(expected, Mapping) or _revision(target) != expected:
            return False
    return True


def _final_metadata(source: Mapping[str, Any], target_dir: Path, metadata_path: Path,
                    target_products: Mapping[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    consolidated = {
        "schema": "cellweft.storage_consolidation.v1",
        "status": "received",
        "source_metadata": source["metadata_revision"],
        "source_products": source["products"],
        "original_accepted_metadata_revision": source["original_metadata_revision"],
        "original_accepted_product_revisions": source["original_products"],
        "target_metadata": {"path": str(metadata_path.resolve())},
        "target_products": dict(target_products),
        "source_input_identity": source["source_input_identity"],
        "request_digest": source["request_digest"],
        "receipt_reused_after_metadata_only_change": bool(
            source["original_metadata_revision"]
            and source["original_metadata_revision"] != source["metadata_revision"]),
    }
    if source["has_source_delivery"]:
        consolidated["source_delivery"] = {"historical": True, "value": source["source_delivery"]}
    final_metadata = copy.deepcopy(source["metadata"])
    final_metadata["products"] = {
        **dict(final_metadata.get("products", {})),
        **{key: str((target_dir / PRODUCT_FILENAMES[key]).resolve()) for key in PRODUCT_KEYS},
    }
    completed = list(final_metadata.get("completed", []))
    for key in PRODUCT_KEYS:
        if key not in completed:
            completed.append(key)
    final_metadata["completed"] = completed
    product_status = dict(final_metadata.get("product_status", {}))
    product_status.update({key: "complete" for key in PRODUCT_KEYS})
    final_metadata["product_status"] = product_status
    final_metadata.pop("publication_incomplete", None)
    if source["has_source_delivery"]:
        final_metadata["source_delivery"] = source["source_delivery"]
    final_metadata["storage_consolidation"] = consolidated
    return final_metadata, consolidated


def _load_checkpoint(path: Path | None, source: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"schema": "cellweft.storage_return_checkpoint.v1", "products": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageReturnError(f"Invalid image migration checkpoint: {path}") from error
    if not isinstance(value, dict) or value.get("schema") != "cellweft.storage_return_checkpoint.v1":
        raise StorageReturnError("Image migration checkpoint has an unsupported schema")
    if value.get("study") != record.get("study") or value.get("image_id") != record.get("image_id"):
        raise StorageReturnError("Image migration checkpoint identity differs")
    if value.get("source_products") != source["products"]:
        raise StorageReturnError("Image migration checkpoint product version differs")
    return value


def _checkpoint_reusable(checkpoint: Mapping[str, Any], key: str, source: Mapping[str, Any], target: Path) -> bool:
    item = checkpoint.get("products", {}).get(key)
    if not isinstance(item, Mapping) or item.get("source") != source["products"].get(key):
        return False
    expected = item.get("target")
    return isinstance(expected, Mapping) and expected == _revision(target)


def _write_checkpoint(path: Path | None, state: dict[str, Any], key: str,
                      source: Mapping[str, Any], target: Path) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("products", {})[key] = {
        "source": copy.deepcopy(source["products"][key]),
        "target": _revision(target),
    }
    _atomic_json(path, state, tag="storage_return_checkpoint")


@contextmanager
def _reservation(factory: Any, work: Path, required_bytes: int) -> Iterator[None]:
    if factory is None:
        yield
        return
    if not callable(factory):
        raise TypeError("shared_disk_reservation must be a context factory")
    with factory(work, required_bytes):
        yield


def _copy_product(source: Path, target: Path) -> dict[str, Any]:
    expected = _revision(source)
    partial = _partial_path(target, "storage_return")
    try:
        with source.open("rb") as source_handle, partial.open("xb") as target_handle:
            pending_bytes = 0
            while chunk := source_handle.read(COPY_CHUNK_BYTES):
                target_handle.write(chunk)
                pending_bytes += len(chunk)
                # Bound dirty NFS data instead of committing an entire large TIFF at EOF.
                if pending_bytes >= COPY_SYNC_BYTES:
                    target_handle.flush()
                    os.fsync(target_handle.fileno())
                    pending_bytes = 0
            target_handle.flush()
            os.fsync(target_handle.fileno())
        if _revision(source) != expected:
            raise StorageReturnError(f"Source changed during product copy: {source}")
        os.utime(partial, ns=(expected["mtime_ns"], expected["mtime_ns"]))
        with partial.open("rb") as target_handle:
            os.fsync(target_handle.fileno())
        partial_stat = partial.stat()
        if partial_stat.st_size != expected["size"] or partial_stat.st_mtime_ns != expected["mtime_ns"]:
            raise StorageReturnError(f"Staged product revision differs: {source}")
        os.replace(partial, target)
        _fsync_directory(target.parent)
        result = _revision(target)
        if result["size"] != expected["size"] or result["mtime_ns"] != expected["mtime_ns"]:
            raise StorageReturnError(f"Published product revision differs: {target}")
        return result
    finally:
        partial.unlink(missing_ok=True)


@contextmanager
def _target_locks(target_dir: Path, *, io_lock_root: Path | None,
                  data_root: Path | None) -> Iterator[None]:
    """Acquire image-level locks after the caller's formal NAS lease/source lock."""
    with resource_lock(target_dir, name=".storage_return.lock"):
        with resource_lock(target_dir, name=".input.lock"):
            with resource_lock(target_dir, name=".preparation.lock"):
                with resource_lock(target_dir, name=".publication.lock"):
                    yield


def _return_image(source_dir: Path, target_dir: Path,
                  accepted_record: Mapping[str, Any], *,
                          io_lock_root: Path | None = None,
                          data_root: Path | None = None,
                          shared_disk_reservation: Any = None,
                          migration_manifest: Path | None = None,
                          checkpoint_path: Path | None = None,
                          prepublish_check: Callable[[], None],
                          reserve_bytes: int = RESERVE_BYTES) -> dict[str, Any]:
    """Copy one accepted image's six products and publish metadata last.

    ``accepted_record`` contains ``study``, ``image_id``, ``metadata_revision``,
    ``product_revisions`` and ``source_input_identity`` evidence. Each revision
    has ``path``, ``size`` and ``mtime_ns``. ``request`` is copied and compared
    exactly when supplied. Input identities are checked before and after the
    transfer but are never copied or rewritten.
    """
    if not isinstance(accepted_record, Mapping):
        raise TypeError("accepted_record must be a mapping")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    source = Path(source_dir)
    target = Path(target_dir)
    source_resolved = source.resolve()
    target_resolved = target.resolve()
    if not source_resolved.is_dir():
        raise StorageReturnError(f"Source image directory does not exist: {source}")
    if source_resolved == target_resolved or source_resolved.is_relative_to(target_resolved) or target_resolved.is_relative_to(source_resolved):
        raise StorageReturnError("Source and target image directories must be disjoint")
    study, image_id = accepted_record.get("study"), accepted_record.get("image_id")
    if not isinstance(study, str) or not study or not isinstance(image_id, str) or not image_id:
        raise StorageReturnError("accepted_record.study and image_id are required")
    source_info: dict[str, Any]
    with _source_publication_read_lock(
            source_resolved,
            formal_path=target_resolved / "metadata.json",
            io_lock_root=io_lock_root,
            data_root=data_root):
        source_info = _source_info(source_resolved, accepted_record)
        _input_identity_check(source_info["source_input_identity"])
        request = accepted_record.get("request", source_info["request"])
        checkpoint = _load_checkpoint(checkpoint_path, source_info, accepted_record)
        checkpoint.setdefault("study", study)
        checkpoint.setdefault("image_id", image_id)
        checkpoint.setdefault("source_products", copy.deepcopy(source_info["products"]))
        if migration_manifest is None and checkpoint_path is not None:
            migration_manifest = Path(checkpoint_path).with_suffix(".previous.jsonl")

        prepublish_check()
        target_resolved.mkdir(parents=True, exist_ok=True)
        with _target_locks(target_resolved, io_lock_root=io_lock_root, data_root=data_root):
            prepublish_check()
            metadata_path = target_resolved / "metadata.json"
            current = _read_json(metadata_path) if metadata_path.exists() else None
            target_paths = {key: target_resolved / filename for key, filename in PRODUCT_FILENAMES.items()}
            if current is not None and _target_metadata_reusable(current, source_info, target_resolved, request):
                return {
                    "status": "reused",
                    "study": study,
                    "image_id": image_id,
                    "copied": [],
                    "reused": list(PRODUCT_KEYS),
                    "metadata": str(metadata_path),
                    "products": {key: str(path) for key, path in target_paths.items()},
                    "metadata_revision": _revision(metadata_path),
                    "target_products": {key: _revision(path) for key, path in target_paths.items()},
                    "receipt_reused_after_metadata_only_change": bool(
                        source_info["original_metadata_revision"]
                        and source_info["original_metadata_revision"] != source_info["metadata_revision"]),
                }
            if current is not None and _target_products_reusable(current, source_info, target_resolved, request):
                target_products = {key: _revision(path) for key, path in target_paths.items()}
                final_metadata, consolidated = _final_metadata(
                    source_info, target_resolved, metadata_path, target_products)
                metadata_staging_bytes = source_info["metadata_revision"]["size"] + 64 * 1024
                usage = shutil.disk_usage(target_resolved)
                if usage.free - metadata_staging_bytes < reserve_bytes:
                    raise StorageReturnCapacityError(
                        f"Return needs {metadata_staging_bytes} bytes plus {reserve_bytes} reserve; free={usage.free}")
                with _reservation(shared_disk_reservation, target_resolved, metadata_staging_bytes):
                    _atomic_json(metadata_path, final_metadata, tag="storage_return_metadata")
                return {
                    "status": "reused",
                    "study": study,
                    "image_id": image_id,
                    "copied": [],
                    "reused": list(PRODUCT_KEYS),
                    "metadata_only": True,
                    "metadata": str(metadata_path),
                    "products": {key: str(path) for key, path in target_paths.items()},
                    "metadata_revision": _revision(metadata_path),
                    "target_products": target_products,
                    "receipt_reused_after_metadata_only_change": consolidated[
                        "receipt_reused_after_metadata_only_change"],
                }

            existing_product_paths = [path for path in target_paths.values() if path.exists()]
            if (current is not None or existing_product_paths) and migration_manifest is None:
                migration_manifest = target_resolved / ".storage_return.previous.jsonl"

            to_copy = [
                key for key, path in target_paths.items()
                if not _checkpoint_reusable(checkpoint, key, source_info, path)
            ]
            final_growth = sum(
                max(0, source_info["products"][key]["size"] -
                    (target_paths[key].stat().st_size if target_paths[key].is_file() else 0))
                for key in to_copy
            )
            metadata_staging_bytes = source_info["metadata_revision"]["size"] + 64 * 1024
            staging_peak = max(
                max((source_info["products"][key]["size"] for key in to_copy), default=0),
                metadata_staging_bytes,
            )
            required_bytes = final_growth + staging_peak
            usage = shutil.disk_usage(target_resolved)
            if usage.free - required_bytes < reserve_bytes:
                raise StorageReturnCapacityError(
                    f"Return needs {required_bytes} bytes plus {reserve_bytes} reserve; free={usage.free}")

            with _reservation(shared_disk_reservation, target_resolved, required_bytes):
                if current is not None or existing_product_paths:
                    manifest_entry = {
                        "schema": "cellweft.storage_return_manifest_entry.v1",
                        "study": study,
                        "image_id": image_id,
                        "source_metadata": source_info["metadata_revision"],
                        "target_metadata": {"path": str(metadata_path.resolve()),
                                            "revision": _revision(metadata_path) if metadata_path.exists() else None},
                        "previous_metadata": current,
                        "previous_products": {
                            key: _revision(path) for key, path in target_paths.items() if path.is_file()
                        },
                    }
                    manifest_path = Path(migration_manifest)
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    _append_manifest(manifest_path, manifest_entry)
                    if current is not None:
                        invalidated = copy.deepcopy(current)
                        invalidated["publication_incomplete"] = True
                        invalidated["completed"] = [name for name in current.get("completed", []) if name not in PRODUCT_KEYS]
                        status = dict(current.get("product_status", {}))
                        status.update({key: "partial" for key in PRODUCT_KEYS})
                        invalidated["product_status"] = status
                        _atomic_json(metadata_path, invalidated, tag="storage_return_invalidate")

                copied: list[str] = []
                reused: list[str] = []
                for key in PRODUCT_KEYS:
                    destination = target_paths[key]
                    if key not in to_copy:
                        reused.append(key)
                        continue
                    _copy_product(Path(source_info["products"][key]["path"]), destination)
                    copied.append(key)
                    _write_checkpoint(checkpoint_path, checkpoint, key, source_info, destination)
                _input_identity_check(source_info["source_input_identity"])
                target_products = {key: _revision(path) for key, path in target_paths.items()}
                final_metadata, consolidated = _final_metadata(
                    source_info, target_resolved, metadata_path, target_products)
                _atomic_json(metadata_path, final_metadata, tag="storage_return_metadata")
                result = {
                    "status": "published",
                    "study": study,
                    "image_id": image_id,
                    "copied": copied,
                    "reused": reused,
                    "metadata": str(metadata_path),
                    "products": {key: str(path) for key, path in target_paths.items()},
                    "metadata_revision": _revision(metadata_path),
                    "target_products": target_products,
                    "required_bytes": required_bytes,
                    "final_growth_bytes": final_growth,
                    "staging_peak_bytes": staging_peak,
                    "metadata_staging_bytes": metadata_staging_bytes,
                    "receipt_reused_after_metadata_only_change": consolidated[
                        "receipt_reused_after_metadata_only_change"],
                }
                if checkpoint_path is not None:
                    result["checkpoint"] = str(checkpoint_path)
                return result


def return_image(source_dir: Path, target_dir: Path, accepted_record: Mapping[str, Any], *,
                 checkpoint: Path | None = None, lock_root: Path | None = None,
                 shared_disk_reservation: Any = None,
                 migration_manifest: Path | None = None,
                 prepublish_check: Callable[[], None],
                 reserve_bytes: int = RESERVE_BYTES) -> dict[str, Any]:
    """Copy one accepted image's six products and publish metadata last."""
    return _return_image(
        source_dir,
        target_dir,
        accepted_record,
        checkpoint_path=checkpoint,
        io_lock_root=lock_root,
        shared_disk_reservation=shared_disk_reservation,
        migration_manifest=migration_manifest,
        prepublish_check=prepublish_check,
        reserve_bytes=reserve_bytes,
    )


__all__ = [
    "PRODUCT_FILENAMES",
    "PRODUCT_KEYS",
    "RESERVE_BYTES",
    "StorageReturnCapacityError",
    "StorageReturnError",
    "return_image",
]
