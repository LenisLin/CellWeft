"""Dynamic admission of reviewed scopes and study-level scheduling records.

This module only discovers existing D01 acceptance and runtime configuration.
It never writes review results or upgrades a scope to accepted status.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Mapping, MutableMapping
import json

import pandas as pd

from cellweft.data.contracts import MaskRejection
from cellweft.data.io.writers import write_json
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import (
    file_revision,
    load_reviewed_study_inputs,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _config_value(config: Mapping[str, Any] | Path) -> dict:
    if isinstance(config, Path):
        return json.loads(config.read_text())
    return dict(config)


def _cache_path(config: Mapping[str, Any]) -> Path | None:
    value = config.get("discovery_cache")
    if value:
        return Path(value)
    run_root = config.get("run_root")
    return Path(run_root) / "discovery_header_cache.json" if run_root else None


def _load_cache(path: Path | None) -> MutableMapping[str, dict]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != "cellweft.discovery_cache.v1":
        raise ValueError(f"Unsupported discovery cache: {path}")
    images = payload.get("images", {})
    if not isinstance(images, dict):
        raise ValueError(f"Discovery cache images must be an object: {path}")
    return images


def _save_cache(path: Path | None, images: MutableMapping[str, dict]) -> None:
    if path is None:
        return
    write_json(path, {"schema": "cellweft.discovery_cache.v1", "images": dict(images)})


def _cache_lock_path(config: Mapping[str, Any]) -> Path | None:
    path = _cache_path(config)
    if path is None:
        return None
    value = config.get("discovery_cache_lock")
    return Path(value) if value else path.with_name(path.name + ".lock")


def _load_shared_cache(config: Mapping[str, Any]) -> MutableMapping[str, dict]:
    path = _cache_path(config)
    lock_path = _cache_lock_path(config)
    if lock_path is None:
        return _load_cache(path)
    with resource_lock(lock_path.parent, shared=True, name=lock_path.name):
        return _load_cache(path)


def _persist_shared_cache(config: Mapping[str, Any],
                          images: MutableMapping[str, dict]) -> None:
    path = _cache_path(config)
    lock_path = _cache_lock_path(config)
    if path is None or lock_path is None:
        return
    with resource_lock(lock_path.parent, name=lock_path.name):
        current = _load_cache(path)
        changed = any(current.get(key) != value for key, value in images.items())
        if changed:
            current.update(images)
            _save_cache(path, current)
        images.clear()
        images.update(current)


def _status_path(config: Mapping[str, Any]) -> Path | None:
    value = config.get("discovery_status")
    if value:
        return Path(value)
    run_root = config.get("run_root")
    return Path(run_root) / "discovery_status.json" if run_root else None


def _write_progress(config: Mapping[str, Any], payload: dict) -> None:
    path = _status_path(config)
    if path is None:
        return
    lock_path = path.with_name(path.name + ".lock")
    value = {
        "schema": "cellweft.discovery_status.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with resource_lock(lock_path.parent, name=lock_path.name):
        write_json(path, value)


def load_header_cache(config: Mapping[str, Any] | Path) -> MutableMapping[str, dict]:
    """Load the persistent successful TIFF-header validation cache for a run."""
    settings = _config_value(config)
    return _load_shared_cache(settings)


def save_header_cache(config: Mapping[str, Any] | Path,
                      cache: MutableMapping[str, dict]) -> None:
    """Persist successful header checks without changing any review or input table."""
    settings = _config_value(config)
    _persist_shared_cache(settings, cache)


def _default_read_gate(config: Mapping[str, Any]) -> Callable[[Path], object] | None:
    value = config.get("nas_lock_root")
    if not value:
        return None
    from cellweft.data.preparation.remote_worker import limited_slots

    lock_root = Path(value)

    def gate(_path: Path):
        return limited_slots(lock_root, "nas_read", 2)

    return gate


def _latest_d01_scopes(study_root: Path) -> tuple[list[dict], list[dict]]:
    review_path = study_root / "review_results.csv"
    if not review_path.is_file():
        return [], []
    with resource_lock(study_root, shared=True):
        rows = pd.read_csv(review_path, keep_default_na=False)
        required = {"scope", "record_id", "check_id", "result", "details"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"review_results.csv missing columns: {sorted(missing)}")
        latest = rows.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
        accepted = latest[(latest.check_id == "D01") & (latest.result == "pass")]
        scopes = []
        gaps = []
        seen = set()
        for row in accepted.itertuples():
            if not isinstance(row.details, str) or not row.details.lstrip().startswith("{"):
                gaps.append({"study": study_root.name, "scope": row.scope,
                             "record_id": row.record_id,
                             "reason": "D01 pass lacks structured input_binding"})
                continue
            try:
                details = json.loads(row.details)
            except json.JSONDecodeError as error:
                gaps.append({"study": study_root.name, "scope": row.scope,
                             "record_id": row.record_id,
                             "reason": f"invalid D01 details: {error}"})
                continue
            binding = details.get("input_binding") if isinstance(details, dict) else None
            if not isinstance(binding, dict):
                gaps.append({"study": study_root.name, "scope": row.scope,
                             "record_id": row.record_id,
                             "reason": "D01 pass lacks structured input_binding"})
                continue
            image_ids = binding.get("image_ids")
            if (not isinstance(image_ids, list) or not image_ids or
                    any(not isinstance(value, str) or not value for value in image_ids) or
                    len(set(image_ids)) != len(image_ids)):
                gaps.append({"study": study_root.name, "scope": row.scope,
                             "record_id": row.record_id,
                             "reason": "Invalid D01 input_binding image_ids"})
                continue
            key = _canonical(binding)
            if key in seen:
                continue
            seen.add(key)
            scopes.append({"scope": row.scope, "record_id": row.record_id,
                           "binding": binding, "scope_key": key})
        return scopes, gaps


def _mask_bindings(config: Mapping[str, Any], study: str) -> tuple[dict, dict | None]:
    root = config.get("mask_bindings_root")
    if not root:
        return {}, None
    path = Path(root) / f"{study}.json"
    if not path.is_file():
        return {}, None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Mask bindings must be an object: {path}")
    for image_id, binding in payload.items():
        if not isinstance(image_id, str) or not isinstance(binding, dict):
            raise ValueError(f"Invalid mask binding entry in {path}")
        if binding.get("status") not in {"unknown", "absent", "bound", "unusable"}:
            raise ValueError(f"Invalid mask status for {study}/{image_id}")
        if binding["status"] == "unusable":
            if set(binding) != {"status", "reason", "evidence"}:
                raise ValueError(f"Incomplete mask rejection for {study}/{image_id}")
            MaskRejection(binding["reason"], binding["evidence"])
    return payload, file_revision(path)


def tissue_reviewed_panels(config: Mapping[str, Any], study: str) -> tuple[list[str], dict | None]:
    """Read the current representative-tissue review configuration."""
    inline = config.get("tissue_reviewed_panels", {})
    revision = None
    path_value = config.get("tissue_reviewed_panels_path")
    if path_value:
        path = Path(path_value)
        payload = json.loads(path.read_text())
        revision = file_revision(path)
        inline = payload
    if isinstance(inline, dict):
        values = inline.get(study, [])
    elif study == str(config.get("study", "")):
        values = inline
    else:
        values = []
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"tissue_reviewed_panels for {study} must be a list of strings")
    return sorted(set(values)), revision


def _study_scan_order(config: Mapping[str, Any], processed: Path) -> list[Path]:
    """Place studies with an explicit usable/absent mask binding first."""
    studies = [path for path in processed.iterdir() if path.is_dir()]
    priority: set[str] = set()
    root = config.get("mask_bindings_root")
    if root:
        binding_root = Path(root)
        if binding_root.is_dir():
            for path in binding_root.glob("*.json"):
                try:
                    payload = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and any(
                    isinstance(binding, dict) and binding.get("status") in {"bound", "absent", "unusable"}
                    for binding in payload.values()
                ):
                    priority.add(path.stem)
    return sorted(studies, key=lambda path: (path.name not in priority, path.name))


def discover(config: Mapping[str, Any] | Path, *,
             header_cache: MutableMapping[str, dict] | None = None,
             read_gate: Callable[[Path], object] | None = None,
             on_scope: Callable[[dict], None] | None = None,
             on_study: Callable[[list[dict]], None] | None = None,
             on_progress: Callable[[dict], None] | None = None) -> tuple[list[dict], list[dict]]:
    """Discover every currently valid exact D01 scope.

    Each returned study entry corresponds to one exact D01 binding.  Entries
    from the same study are intentionally kept separate so an affected scope
    cannot grant or suppress a different intersecting scope.  Repeated calls
    re-read review rows, mask bindings and tissue-review configuration while
    reusing only successful per-image header checks whose asset revisions are
    unchanged.
    """
    settings = _config_value(config)
    data_root = Path(settings["data_root"])
    processed = data_root / "processed"
    cache_path = _cache_path(settings)
    cache = header_cache if header_cache is not None else _load_shared_cache(settings)
    gate = read_gate if read_gate is not None else _default_read_gate(settings)
    workers = settings.get("discovery_workers", 2)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("discovery_workers must be an integer from 1 to 4")
    studies: list[dict] = []
    gaps: list[dict] = []
    scanned_studies = 0
    _write_progress(settings, {
        "event": "scan_started",
        "study": None,
        "scanned_studies": 0,
        "accepted_scopes": 0,
        "gaps": 0,
        "complete": False,
    })
    if on_progress is not None:
        on_progress({
            "event": "scan_started", "study": None,
            "scanned_studies": 0, "accepted_scopes": 0, "gaps": 0,
            "complete": False,
        })
    if not processed.is_dir():
        missing = {"scope": "study", "record_id": None,
                   "reason": f"processed root is missing: {processed}"}
        gaps.append(missing)
        _write_progress(settings, {
            "event": "scan_complete", "study": None,
            "scanned_studies": 0, "accepted_scopes": 0,
            "gaps": len(gaps), "complete": True,
        })
        return studies, gaps
    study_roots = _study_scan_order(settings, processed)
    events: Queue[tuple] = Queue()
    initial_cache = dict(cache)

    def scan_study(index: int, study_root: Path) -> None:
        try:
            local_cache = initial_cache.copy()
            if cache_path is not None:
                local_cache.update(_load_shared_cache(settings))
            try:
                scopes, scope_gaps = _latest_d01_scopes(study_root)
                for gap in scope_gaps:
                    events.put(("gap", index, gap))
                if not scopes and not scope_gaps:
                    events.put(("gap", index, {"study": study_root.name, "scope": "study",
                                               "record_id": None,
                                               "reason": "no current structured D01 acceptance"}))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                events.put(("gap", index, {"study": study_root.name, "scope": "study",
                                           "record_id": None, "reason": str(error)}))
                scopes = []
            for scope in scopes:
                binding = scope["binding"]
                image_ids = list(binding["image_ids"])
                previous_headers = {image_id: local_cache.get(image_id) for image_id in image_ids}
                try:
                    inputs = load_reviewed_study_inputs(
                        study_root, data_root=data_root, image_ids=image_ids,
                        header_cache=local_cache, read_gate=gate,
                    )
                    masks, masks_revision = _mask_bindings(settings, study_root.name)
                    image_masks = {
                        image_id: masks.get(image_id, {"status": "unknown"})
                        for image_id in image_ids
                    }
                    tissue, tissue_revision = tissue_reviewed_panels(settings, study_root.name)
                    panel_ids = sorted(set(inputs.images.panel_id.astype(str)))
                    entry = {
                        "study": study_root.name,
                        "scope": scope["scope"],
                        "record_id": scope["record_id"],
                        "scope_key": scope["scope_key"],
                        "image_ids": image_ids,
                        "process_image_ids": image_ids.copy(),
                        "panel_ids": panel_ids,
                        "input_binding": inputs.binding,
                        "mask_bindings": image_masks,
                        "mask_bindings_revision": masks_revision,
                        "tissue_reviewed_panels": tissue,
                        "tissue_reviewed_revision": tissue_revision,
                    }
                    changed_headers = {
                        image_id: local_cache[image_id] for image_id in image_ids
                        if local_cache.get(image_id) != previous_headers[image_id]
                    }
                    if changed_headers:
                        _persist_shared_cache(settings, changed_headers.copy())
                    events.put(("scope", index, entry,
                                {**entry, "_reviewed_inputs": inputs}, changed_headers))
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    events.put(("gap", index, {"study": study_root.name, "scope": scope["scope"],
                                               "record_id": scope["record_id"],
                                               "image_ids": image_ids, "reason": str(error)}))
        finally:
            events.put(("done", index))

    ordered_studies: list[list[dict]] = [[] for _ in study_roots]
    ordered_gaps: list[list[dict]] = [[] for _ in study_roots]
    study_callback_entries: list[list[dict]] = [[] for _ in study_roots]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_study, index, root)
                   for index, root in enumerate(study_roots)]
        while scanned_studies < len(study_roots):
            event = events.get()
            kind, index = event[:2]
            study_name = study_roots[index].name
            if kind == "gap":
                gap = event[2]
                gaps.append(gap)
                ordered_gaps[index].append(gap)
                continue
            if kind == "scope":
                _, _, entry, callback_entry, changed_headers = event
                cache.update(changed_headers)
                studies.append(entry)
                ordered_studies[index].append(entry)
                study_callback_entries[index].append(callback_entry)
                try:
                    if on_scope is not None:
                        on_scope(callback_entry)
                    progress = {
                        "event": "scope_accepted", "study": study_name,
                        "scanned_studies": scanned_studies,
                        "accepted_scopes": len(studies), "gaps": len(gaps), "complete": False,
                    }
                    _write_progress(settings, progress)
                    if on_progress is not None:
                        on_progress(progress)
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                    gap = {"study": study_name, "scope": entry["scope"],
                           "record_id": entry["record_id"],
                           "image_ids": entry["image_ids"], "reason": str(error)}
                    gaps.append(gap)
                    ordered_gaps[index].append(gap)
                continue
            futures[index].result()
            scanned_studies += 1
            progress = {
                "event": "study_complete", "study": study_name,
                "scanned_studies": scanned_studies,
                "accepted_scopes": len(studies), "gaps": len(gaps), "complete": False,
            }
            _write_progress(settings, progress)
            if on_progress is not None:
                on_progress(progress)
            if on_study is not None:
                on_study(study_callback_entries[index])
    studies = [entry for group in ordered_studies for entry in group]
    gaps = [gap for group in ordered_gaps for gap in group]
    progress = {
        "event": "scan_complete", "study": None,
        "scanned_studies": scanned_studies,
        "accepted_scopes": len(studies), "gaps": len(gaps), "complete": True,
    }
    _write_progress(settings, progress)
    if on_progress is not None:
        on_progress(progress)
    return studies, gaps


def vector_tasks(studies: list[dict]) -> list[dict]:
    """Group exact image scopes into one panel-union vector task per study."""
    grouped: OrderedDict[str, dict] = OrderedDict()
    for entry in sorted(studies, key=lambda value: (value["study"], value.get("scope_key", ""))):
        study = entry["study"]
        task = grouped.setdefault(study, {
            "study": study, "image_ids": set(), "panel_ids": set(), "scopes": [],
        })
        scope_key = entry.get("scope_key") or _canonical({
            "study": study, "image_ids": sorted(entry.get("image_ids", [])),
        })
        if any(scope.get("scope_key") == scope_key for scope in task["scopes"]):
            continue
        image_ids = list(entry.get("image_ids", []))
        task["image_ids"].update(image_ids)
        task["panel_ids"].update(entry.get("panel_ids", []))
        task["scopes"].append({
            "scope_key": scope_key,
            "image_ids": image_ids,
            "input_binding": entry.get("input_binding"),
            "mask_bindings": entry.get("mask_bindings", {}),
            "mask_bindings_revision": entry.get("mask_bindings_revision"),
            "tissue_reviewed_panels": list(entry.get("tissue_reviewed_panels", [])),
            "tissue_reviewed_revision": entry.get("tissue_reviewed_revision"),
        })
    result = []
    for task in grouped.values():
        result.append({
            "study": task["study"],
            "image_ids": sorted(task["image_ids"]),
            "panel_ids": sorted(task["panel_ids"]),
            "scopes": task["scopes"],
        })
    return result
