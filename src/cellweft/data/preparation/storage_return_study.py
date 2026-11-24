"""Finalize one already-consolidated study on the formal NAS."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable, Mapping

import pandas as pd

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.embeddings import _panel_identity, _same_panel_identity, _validate
from cellweft.data.preparation.inputs import PANEL_INPUT_FIELDS, file_revision
from cellweft.data.preparation.io_coordination import formal_write_lease


_FIVE_TABLES = (
    "images.parquet", "sample_metadata.csv", "field_mapping.csv",
    "review_results.csv", "panel_mapping.csv",
)
_IMMUTABLE_TABLES = _FIVE_TABLES[:4]
_PRODUCT_NAMES = ("image", "channels", "mask", "cells", "expression", "neighbors")
_STUDY_PRODUCTS = ("panel_mapping.csv", "marker_embeddings.pt")


@contextmanager
def _source_read_lock(study_root: Path):
    """Use a read-only flock when the source already exposes its lock inode."""
    lock = study_root / ".publication.lock"
    if not lock.is_file():
        yield
        return
    with lock.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _safe_ids(image_ids: list[str]) -> list[str]:
    if type(image_ids) is not list or not image_ids:
        raise ValueError("image_ids must be a non-empty list")
    if any(type(value) is not str or not value or Path(value).name != value
           or value in {".", ".."} for value in image_ids):
        raise ValueError("image_ids must be safe, non-empty path components")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids contains duplicates")
    return sorted(image_ids)


def _study_name(target: Path, source: Path) -> str:
    if not target.name or target.name in {".", ".."} or target.name != source.name:
        raise ValueError("target and source must be the same study directory")
    return target.name


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _table_revisions(root: Path) -> dict[str, dict]:
    return {name: file_revision(root / name) for name in _FIVE_TABLES}


def _check_image_keys(root: Path, study: str, image_ids: list[str]) -> pd.DataFrame:
    table = pd.read_parquet(root / "images.parquet")
    required = {"study", "image_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"{study}: images.parquet lacks study/image_id keys")
    if table.empty or table.study.nunique() != 1 or str(table.study.iloc[0]) != study:
        raise ValueError(f"{study}: images.parquet contains another study")
    values = table.image_id.astype(str).tolist()
    if len(values) != len(set(values)) or sorted(values) != image_ids:
        raise ValueError(f"{study}: NAS image keys do not equal the approved scope")
    return table


def _path_value(value, image_root: Path, target_root: Path) -> Path:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid product path in {image_root / 'metadata.json'}")
    path = Path(value)
    path = path if path.is_absolute() else image_root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(target_root.resolve()):
        raise ValueError(f"Product path escapes NAS study root: {path}")
    return resolved


def _metadata_products(metadata: dict, image_root: Path, target_root: Path) -> dict[str, Path]:
    products = metadata.get("products")
    if not isinstance(products, dict) or not products:
        raise ValueError(f"{image_root}: metadata.products is missing")
    missing = sorted(set(_PRODUCT_NAMES) - set(products))
    if missing:
        raise ValueError(f"{image_root}: metadata.products lacks required products: {missing}")
    resolved = {str(name): _path_value(value, image_root, target_root)
                for name, value in products.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{image_root}: NAS products are missing: {missing}")
    return resolved


def _revision_value(value, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain path, size and mtime_ns")
    path, size, mtime_ns = value.get("path"), value.get("size"), value.get("mtime_ns")
    if not isinstance(path, str) or type(size) is not int or type(mtime_ns) is not int:
        raise ValueError(f"{label} must contain path, size and mtime_ns")
    return {"path": str(Path(path).resolve()), "size": size, "mtime_ns": mtime_ns}


def _frozen_revisions(source: Path, evidence: dict, study: str) -> dict[str, dict]:
    values = evidence.get("study_product_revisions")
    if not isinstance(values, Mapping):
        raise ValueError(f"{study}: evidence.study_product_revisions is required")
    frozen = {}
    for name in _STUDY_PRODUCTS:
        value = values.get(name)
        if isinstance(value, Mapping):
            for key in ("source", "owner", "current", "revision"):
                if isinstance(value.get(key), Mapping):
                    value = value[key]
                    break
        revision = _revision_value(value, f"{study}: study_product_revisions.{name}")
        expected_path = str((source / name).resolve())
        if revision["path"] != expected_path:
            raise ValueError(f"{study}: frozen {name} path is not the owner path")
        frozen[name] = revision
    return frozen


def _check_frozen_revisions(source: Path, study: str,
                            frozen: dict[str, dict]) -> dict[str, dict]:
    actual = {name: file_revision(source / name) for name in _STUDY_PRODUCTS}
    if actual != frozen:
        raise ValueError(f"{study}: owner panel/vector changed since frozen evidence")
    return actual


def _storage_products(metadata: dict, image_root: Path, target_root: Path,
                      products: dict[str, Path]) -> list[Path]:
    storage = metadata.get("storage_consolidation")
    if not isinstance(storage, dict) or storage.get("status") != "received":
        raise ValueError(f"{image_root}: storage_consolidation.status is not received")
    completed = set(metadata.get("completed", []))
    missing = sorted(set(_PRODUCT_NAMES) - completed)
    if missing:
        raise ValueError(f"{image_root}: completed lacks received products: {missing}")
    revisions = storage.get("target_products")
    if not isinstance(revisions, Mapping):
        raise ValueError(f"{image_root}: storage_consolidation.target_products is missing")
    missing = sorted(set(_PRODUCT_NAMES) - set(revisions))
    if missing:
        raise ValueError(f"{image_root}: target_products lacks required products: {missing}")
    checked = []
    for name in _PRODUCT_NAMES:
        path = products[name]
        actual = file_revision(path)
        expected = _revision_value(revisions[name], f"{image_root}: target_products.{name}")
        if actual != expected:
            raise ValueError(f"{image_root}: target_products.{name} does not match metadata.products")
        checked.append(path)
    return checked


def _declared_asset(value, data_root: Path) -> Path | None:
    if value is None or (isinstance(value, str) and not value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    path = Path(str(value))
    return (path if path.is_absolute() else data_root / path).resolve()


def _check_image_assets(row: pd.Series, target_root: Path, study: str, image_id: str) -> None:
    image_root = target_root / "images" / image_id
    input_path = image_root / "input.tiff"
    _require(input_path, f"{study}/{image_id} canonical input.tiff")
    data_root = target_root.parent.parent
    for field in ("input_image_path", "valid_pixels_path", "sample_region_path"):
        value = row.get(field)
        path = _declared_asset(value, data_root)
        if path is not None and not path.is_file():
            raise FileNotFoundError(
                f"{study}/{image_id}: images.parquet {field} is missing: {path}")


def _image_records(target: Path, study: str, image_ids: list[str],
                   image_table: pd.DataFrame) -> list[dict]:
    rows = image_table.set_index(image_table.image_id.astype(str), drop=False)
    records = []
    for image_id in image_ids:
        image_root = target / "images" / image_id
        _check_image_assets(rows.loc[image_id], target, study, image_id)
        metadata_path = image_root / "metadata.json"
        _require(metadata_path, f"{study}/{image_id} metadata")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("study") != study or metadata.get("image_id") != image_id:
            raise ValueError(f"{study}/{image_id}: metadata identity does not match images.parquet")
        products = _metadata_products(metadata, image_root, target)
        _storage_products(metadata, image_root, target, products)
        records.append({"image_id": image_id, "path": str(metadata_path.resolve()),
                        "metadata": metadata, "products": products})
    return records


def _legacy_panel_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """Expose the pre-S1 panel schema using the established source-name fallback.

    Older accepted panels predate the administrative ``s1_original_marker``
    column.  Embedding construction already treats that schema as equivalent
    to ``source_name``; storage-return identity checks must use the same rule.
    An explicitly present column is never filled or rewritten.
    """
    if "s1_original_marker" not in panel.columns and "source_name" in panel.columns:
        panel = panel.copy()
        panel["s1_original_marker"] = panel["source_name"]
    return panel


def _panel_inputs(panel: pd.DataFrame, study: str) -> str:
    panel = _legacy_panel_inputs(panel)
    missing = [field for field in PANEL_INPUT_FIELDS if field not in panel.columns]
    if missing:
        raise ValueError(f"{study}: panel_mapping.csv lacks input fields: {missing}")
    return _panel_identity(panel[PANEL_INPUT_FIELDS].copy())


def _panel_key(row: pd.Series) -> str:
    row = _legacy_panel_inputs(row.to_frame().T).iloc[0]
    values = []
    for field in PANEL_INPUT_FIELDS:
        value = row[field]
        if pd.isna(value):
            value = None
        elif hasattr(value, "item"):
            value = value.item()
        values.append(value)
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)


def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    if "embedding_row" in panel:
        panel["embedding_row"] = pd.to_numeric(
            panel["embedding_row"], errors="coerce").astype("Int64")
    return panel


def _load_embeddings(root: Path, study: str):
    import torch

    panel_path = root / "panel_mapping.csv"
    vector_path = root / "marker_embeddings.pt"
    _require(panel_path, f"{study} panel_mapping.csv")
    _require(vector_path, f"{study} marker_embeddings.pt")
    panel = _normalize_panel(pd.read_csv(panel_path, keep_default_na=False))
    payload = torch.load(vector_path, map_location="cpu", weights_only=True)
    if payload.get("metadata", {}).get("study") != study:
        raise ValueError(f"{study}: marker_embeddings.pt belongs to another study")
    _validate(payload, panel)
    return payload, panel


def _compare_embeddings(study: str, owner_payload: dict, owner_panel: pd.DataFrame,
                        nas_payload: dict, nas_panel: pd.DataFrame) -> None:
    if not _same_panel_identity(_panel_inputs(owner_panel, study), _panel_inputs(nas_panel, study)):
        raise ValueError(f"{study}: owner/NAS PANEL_INPUT_FIELDS conflict")
    if owner_payload["marker_ids"] != nas_payload["marker_ids"]:
        raise ValueError(f"{study}: owner/NAS marker_ids conflict")
    if owner_payload["mapping_expressions"] != nas_payload["mapping_expressions"]:
        raise ValueError(f"{study}: owner/NAS mapping_expressions conflict")
    import torch
    if not torch.equal(owner_payload["embeddings"], nas_payload["embeddings"]):
        raise ValueError(f"{study}: owner/NAS embedding tensor conflict")


def _compare_payloads(study: str, owner_payload: dict, current_payload: dict) -> None:
    if current_payload.get("metadata", {}).get("study") != study:
        raise ValueError(f"{study}: current marker_embeddings.pt belongs to another study")
    if not _same_panel_identity(owner_payload["metadata"]["panel_identity"],
                                current_payload.get("metadata", {}).get("panel_identity", "")):
        raise ValueError(f"{study}: owner/NAS embedding panel identity conflict")
    if owner_payload["marker_ids"] != current_payload.get("marker_ids"):
        raise ValueError(f"{study}: owner/NAS marker_ids conflict")
    if owner_payload["mapping_expressions"] != current_payload.get("mapping_expressions"):
        raise ValueError(f"{study}: owner/NAS mapping_expressions conflict")
    import torch
    if not torch.equal(owner_payload["embeddings"], current_payload.get("embeddings")):
        raise ValueError(f"{study}: owner/NAS embedding tensor conflict")


def _has_embedding_rows(panel: pd.DataFrame) -> bool:
    panel = _normalize_panel(panel)
    if "embedding_row" not in panel.columns:
        return False
    expected = (panel.keep_channel.astype(str).str.lower().eq("true")
                & panel.role.eq("downstream")
                & panel.mapping_status.eq("mapped"))
    return (not panel.loc[expected, "embedding_row"].isna().any()
            and not panel.loc[~expected, "embedding_row"].notna().any())


def _publish_missing_embedding(target: Path, study: str, owner_payload: dict,
                               owner_panel: pd.DataFrame, nas_panel: pd.DataFrame,
                               owner_vector: Path, current_vector: Path | None = None) -> None:
    if not _same_panel_identity(_panel_inputs(owner_panel, study), _panel_inputs(nas_panel, study)):
        raise ValueError(f"{study}: cannot publish embeddings; owner/NAS panel inputs conflict")
    source_rows = {_panel_key(row): row.embedding_row
                   for _, row in owner_panel.iterrows()}
    if len(source_rows) != len(owner_panel):
        raise ValueError(f"{study}: owner panel has duplicate embedding identities")
    updated = nas_panel.copy()
    try:
        values = [source_rows[_panel_key(row)] for _, row in updated.iterrows()]
    except KeyError as exc:
        raise ValueError(f"{study}: owner/NAS panel rows cannot be paired for embedding publication") from exc
    updated["embedding_row"] = pd.Series(values, index=updated.index, dtype="Int64")
    with tempfile.TemporaryDirectory(prefix=".storage_return_", dir=target) as temporary:
        temporary = Path(temporary)
        vector = temporary / "marker_embeddings.pt"
        panel = temporary / "panel_mapping.csv"
        shutil.copy2(current_vector or owner_vector, vector)
        updated.to_csv(panel, index=False, encoding="utf-8-sig")
        checked_panel = pd.read_csv(panel, keep_default_na=False)
        checked_panel = _normalize_panel(checked_panel)
        import torch
        checked_payload = torch.load(vector, map_location="cpu", weights_only=True)
        _validate(checked_payload, checked_panel)
        _fsync(panel)
        _fsync(vector)
        os.replace(panel, target / "panel_mapping.csv")
        os.replace(vector, target / "marker_embeddings.pt")
        _fsync(target)


def _refs(records: list[dict], key: str) -> list[str]:
    values = []
    for record in records:
        value = record["metadata"].get(key)
        if isinstance(value, str) and value:
            values.append(value)
        elif isinstance(value, Mapping):
            for candidate in (value.get("path"), value.get("source"), value.get("reference")):
                if isinstance(candidate, str) and candidate:
                    values.append(candidate)
    return sorted(set(values))


def _flag(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "pending"


def _count(values) -> dict[str, int]:
    counts = {"true": 0, "false": 0, "pending": 0}
    for value in values:
        counts[_flag(value)] += 1
    return counts


def _quality_summary(records: list[dict]) -> dict:
    metadata = [record["metadata"] for record in records]
    sampling = [item.get("sampling") if isinstance(item.get("sampling"), dict) else {}
                for item in metadata]
    reviewed = []
    evaluated = []
    statuses = []
    overview = []
    for item, sample in zip(metadata, sampling):
        request = sample.get("request") if isinstance(sample.get("request"), dict) else {}
        reviewed.append(sample.get("reviewed", request.get("reviewed")))
        evaluated.append(sample.get("evaluated"))
        statuses.append(sample.get("status"))
        mask = item.get("mask_check")
        if not isinstance(mask, dict):
            mask = {}
        overview.append(mask.get("overview", mask.get("overview_judgment")))
    status_counts = {}
    for status in statuses:
        key = status if isinstance(status, str) and status else "pending"
        status_counts[key] = status_counts.get(key, 0) + 1
    complete = (all(value is True for value in reviewed)
                and all(value is True for value in evaluated)
                and all(value is not None for value in overview))
    return {
        "status": "recorded" if complete else "pending",
        "sampling": {"reviewed": _count(reviewed), "evaluated": _count(evaluated),
                      "status": status_counts},
        "mask_overview": _count(overview),
    }


def _evidence_refs(evidence: dict) -> list:
    values = []
    for key in ("existing_evidence", "accepted_index"):
        if key in evidence:
            values.append({key: evidence[key]})
    return values


def _summary(target: Path, study: str, image_ids: list[str], records: list[dict],
             evidence: dict, raw: dict, revisions: dict, embedding_status: str,
             study_revisions: dict) -> dict:
    metadata = [record["metadata"] for record in records]
    acceptance_keys = ("user_confirmed", "technical_reception", "sampling_policy",
                       "actual_sample_ids_and_results", "unperformed_checks_or_use_limits")
    acceptance = {key: evidence[key] for key in acceptance_keys if key in evidence}
    acceptance["organization_quality"] = _quality_summary(records)
    acceptance["training"] = "pending"
    for key in ("existing_evidence", "accepted_index"):
        if key in evidence:
            acceptance[key] = evidence[key]
    raw_record = dict(raw)
    raw_record.setdefault("status", "unknown")
    mask_origins = sorted({str(item["mask_origin"]) for item in metadata
                           if item.get("mask_origin") is not None})
    quality_refs = _evidence_refs(evidence)
    return {
        "schema": "cellweft.processing_summary.v1",
        "study": study,
        "scope": {"retained_images": len(image_ids),
                  "image_keys": "images.parquet:(study,image_id)",
                  "historical_exclusions": evidence.get("historical_exclusions", [])},
        "original_source": {"status": raw_record.get("status", "unknown"),
                            "candidates": raw_record.get("candidates", raw_record.get("raw_candidates", [])),
                            "record": raw_record},
        "processing": {
            "normalization": "per-image metadata.json",
            "mask_origins": mask_origins,
            "mask_references": _refs(records, "source_mask_check"),
            "segmentation_parameters": "per-image metadata.json",
            "cell_expression_neighbors": "final-mask aligned; per-image metadata.json",
            "crop": {"size": 224, "stride": 112, "edge": "cover", "storage": "cells.crop_origins"},
        },
        "acceptance": acceptance,
        "dependencies": {
            "raw": raw_record.get("candidates", raw_record.get("raw_candidates", [])),
            "masks": evidence.get("mask_dependencies", evidence.get("mask_refs", [])),
            "qa": [*evidence.get("qa_dependencies", evidence.get("evidence", [])),
                   *quality_refs],
        },
        "final_location": str(target.resolve()),
        "embedding_status": embedding_status,
        "study_product_revisions": study_revisions,
        "nas_five_table_revisions": {"before": revisions, "after": revisions,
                                      "unchanged": {name: True for name in revisions}},
        "storage_consolidation": {
            "status": "complete", "received_count": len(image_ids),
            "final_location": str(target.resolve()), "owner_retained": True,
            "no_deletion": True,
        },
        "image_metadata_count": len(metadata),
    }


def finalize_study(target: Path, source: Path, *, image_ids: list[str],
                  evidence: dict, raw: dict, lock_root: Path,
                  prepublish_check: Callable[[], None]) -> dict:
    """Close a study whose image-level NAS consolidation has already completed."""
    target, source, lock_root = Path(target), Path(source), Path(lock_root)
    if not target.is_dir() or not source.is_dir():
        raise FileNotFoundError("target and source must be existing study directories")
    study = _study_name(target, source)
    image_ids = _safe_ids(image_ids)
    if not isinstance(evidence, dict) or not isinstance(raw, dict):
        raise TypeError("evidence and raw must be dictionaries")
    frozen = _frozen_revisions(source, evidence, study)
    with formal_write_lease(target, io_lock_root=lock_root, priority="high"):
        prepublish_check()
        with resource_lock(target):
            prepublish_check()
            for name in _IMMUTABLE_TABLES + ("panel_mapping.csv",):
                _require(target / name, f"{study} NAS {name}")
            before = _table_revisions(target)
            image_table = _check_image_keys(target, study, image_ids)
            records = _image_records(target, study, image_ids, image_table)
            with _source_read_lock(source):
                source_revisions = _check_frozen_revisions(source, study, frozen)
                owner_payload, owner_panel = _load_embeddings(source, study)
                nas_panel = pd.read_csv(target / "panel_mapping.csv", keep_default_na=False)
                nas_panel = _normalize_panel(nas_panel)
                owner_identity = _panel_inputs(owner_panel, study)
                nas_identity = _panel_inputs(nas_panel, study)
                if not _same_panel_identity(owner_identity, nas_identity):
                    raise ValueError(f"{study}: owner/NAS PANEL_INPUT_FIELDS conflict")
                vector_path = target / "marker_embeddings.pt"
                if vector_path.is_file() and _has_embedding_rows(nas_panel):
                    nas_payload, current_panel = _load_embeddings(target, study)
                    _compare_embeddings(study, owner_payload, owner_panel,
                                        nas_payload, current_panel)
                    embedding_status = "present"
                else:
                    current_vector = None
                    if vector_path.is_file():
                        import torch
                        try:
                            current_payload = torch.load(vector_path, map_location="cpu",
                                                         weights_only=True)
                        except Exception as exc:
                            raise ValueError(
                                f"{study}: existing marker_embeddings.pt cannot be read; "
                                "repair or explicitly replace the conflicting asset") from exc
                        _compare_payloads(study, owner_payload, current_payload)
                        current_vector = vector_path
                    _publish_missing_embedding(target, study, owner_payload,
                                               owner_panel, nas_panel,
                                               source / "marker_embeddings.pt", current_vector)
                    embedding_status = "published_from_owner"
                source_revisions = _check_frozen_revisions(source, study, frozen)
            after = _table_revisions(target)
            if any(before[name] != after[name] for name in _IMMUTABLE_TABLES):
                raise RuntimeError(f"{study}: immutable NAS input table changed during finalization")
            if not (target / "marker_embeddings.pt").is_file():
                raise FileNotFoundError(f"{study}: marker_embeddings.pt was not published")
            study_revisions = {"frozen": frozen, "source": source_revisions,
                               "target": {name: file_revision(target / name)
                                          for name in _STUDY_PRODUCTS}}
            summary = _summary(target, study, image_ids, records, evidence, raw,
                               before, embedding_status, study_revisions)
            summary["nas_five_table_revisions"]["after"] = after
            summary["nas_five_table_revisions"]["unchanged"] = {
                name: before[name] == after[name] for name in before
            }
            write_json(target / "processing_summary.json", summary)
            return {"study": study, "status": "complete", **summary["storage_consolidation"],
                    "embedding_status": embedding_status,
                    "summary": str((target / "processing_summary.json").resolve()),
                    "study_product_revisions": study_revisions,
                    "nas_five_table_revisions": summary["nas_five_table_revisions"]}
