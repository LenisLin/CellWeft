"""Closed, revision-bound handover of reviewed study inputs to one node."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import time
import tarfile
import threading
import stat
from typing import TYPE_CHECKING

import pandas as pd
import tifffile

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.execution_profile import execution_profile as checked_execution_profile
from cellweft.data.preparation.inputs import (PANEL_INPUT_FIELDS, ReviewedStudyInputs,
    assert_reviewed_binding, file_revision, input_header_lease,
    load_reviewed_study_inputs, validate_panel_mapping)
from cellweft.data.preparation.requests import preparation_request, source_identity
from cellweft.data.preparation.validation import validate_products
from cellweft.data.preparation.study_input import validate_image_metadata
from cellweft.data.preparation.io_coordination import formal_read_lease

if TYPE_CHECKING:
    from cellweft.data.preparation.execution import Transport


SEND_ENVELOPE = "envelope.json"
SEND_PLAN = ".send-plan.json"
DELIVERY_TIMEOUT_BASE_SECONDS = 900
DELIVERY_TIMEOUT_RATE_BYTES_PER_SECOND = 10 * 1024**2
DELIVERY_TIMEOUT_GRACE_SECONDS = 900
DELIVERY_TIMEOUT_MAX_SECONDS = 14_400
DELIVERY_FAILURE_STDERR_BYTES = 8 * 1024
TABLE_CACHE_NAMES = ("images.parquet", "panel_mapping.csv", "sample_metadata.csv",
                     "field_mapping.csv", "review_results.csv")
TABLE_CACHE_SCHEMA = "cellweft.delivery_table_cache.v1"


def _delivery_stream_timeout(missing_bytes: int) -> int:
    if type(missing_bytes) is not int or missing_bytes < 0:
        raise ValueError("missing_bytes must be a nonnegative integer")
    if missing_bytes == 0:
        return DELIVERY_TIMEOUT_BASE_SECONDS
    payload_seconds = ((missing_bytes + DELIVERY_TIMEOUT_RATE_BYTES_PER_SECOND - 1)
                       // DELIVERY_TIMEOUT_RATE_BYTES_PER_SECOND)
    return min(DELIVERY_TIMEOUT_MAX_SECONDS,
               DELIVERY_TIMEOUT_BASE_SECONDS + payload_seconds
               + DELIVERY_TIMEOUT_GRACE_SECONDS)


def _write_transfer_failure(packet_root: Path, *, timeout_seconds: int,
                            missing: list[str], producer_errors: list[BaseException],
                            transfer_error: BaseException | None) -> None:
    def detail(error: BaseException) -> dict:
        stderr = getattr(error, "stderr", None)
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr[-DELIVERY_FAILURE_STDERR_BYTES:].decode("utf-8", errors="replace")
        return {"type": type(error).__name__, "message": str(error),
                "returncode": getattr(error, "returncode", None),
                "stderr": stderr}

    record = {
        "schema": "cellweft.delivery_transfer_failure.v1",
        "observed_at_ns": time.time_ns(), "timeout_seconds": timeout_seconds,
        "missing": missing,
        "producer_errors": [detail(error) for error in producer_errors],
        "transfer_error": detail(transfer_error) if transfer_error is not None else None,
    }
    (packet_root / ".transfer.failure.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class StudyDelivery:
    root: Path
    study: str
    node_id: str
    image_ids: list[str]
    inputs: ReviewedStudyInputs
    mask_bindings: dict[str, dict]
    requests: dict[str, dict]
    settings_path: Path
    weights_path: Path
    s1_path: Path
    bank_dir: Path
    reuse_metadata: dict[str, Path]
    source_bindings: dict
    execution_profile: dict
    allow_unknown_mask_segmentation: bool = False


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Delivery path escapes packet: {relative}")
    return path


def _stat(path: Path) -> dict:
    value = path.stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns}


def _copy(source: Path, root: Path, relative: str, files: list[dict]) -> Path:
    previous = next((item for item in files if item["replica"] == relative), None)
    if previous is not None:
        if previous["source"] != file_revision(source):
            raise ValueError(f"Repeated delivery path has a different source: {relative}")
        return _inside(root, relative)
    destination = _inside(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = file_revision(source)
    shutil.copyfile(source, destination)
    os.utime(destination, ns=(source.stat().st_atime_ns, before["mtime_ns"]))
    if file_revision(source) != before or _stat(destination) != {"size": before["size"], "mtime_ns": before["mtime_ns"]}:
        raise ValueError(f"Source changed during copy: {source}")
    files.append({"source": before, "replica": relative,
                  "revision": {"size": before["size"], "mtime_ns": before["mtime_ns"]}})
    return destination


def _table_cache_entry(cache_root: Path, source_revision: dict) -> Path:
    source_path = Path(source_revision["path"])
    if not source_path.is_absolute():
        raise ValueError("Table cache source path must be absolute")
    relative = source_path.relative_to(Path(source_path.anchor))
    version = f"{relative.name}.size-{source_revision['size']}.mtime-{source_revision['mtime_ns']}"
    return cache_root.resolve() / "tables" / relative.parent / version


def _read_table_cache(entry: Path, source_revision: dict) -> Path:
    table = entry / "table"
    manifest = entry / "manifest.json"
    if not entry.is_dir() or not table.is_file() or not manifest.is_file():
        raise ValueError(f"Incomplete table cache entry: {entry}")
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid table cache manifest: {manifest}") from error
    expected = {
        "schema": TABLE_CACHE_SCHEMA,
        "source": source_revision,
        "cached": {"size": source_revision["size"],
                    "mtime_ns": source_revision["mtime_ns"]},
    }
    if value != expected:
        raise ValueError(f"Table cache version differs from source: {entry}")
    if _stat(table) != expected["cached"]:
        raise ValueError(f"Table cache content changed: {table}")
    return table


def _materialize_table_cache(source: Path, cache_root: Path,
                             source_revision: dict) -> Path:
    entry = _table_cache_entry(cache_root, source_revision)
    with resource_lock(entry.parent, name=f".{entry.name}.lock"):
        if entry.exists():
            return _read_table_cache(entry, source_revision)
        with formal_read_lease(source):
            if file_revision(source) != source_revision:
                raise ValueError(f"Source changed before table cache copy: {source}")
            temporary = Path(tempfile.mkdtemp(prefix=f".{entry.name}.",
                                              dir=entry.parent))
            try:
                table = temporary / "table"
                shutil.copyfile(source, table)
                os.utime(table, ns=(source_revision["mtime_ns"], source_revision["mtime_ns"]))
                if file_revision(source) != source_revision:
                    raise ValueError(f"Source changed during table cache copy: {source}")
                if _stat(table) != {"size": source_revision["size"],
                                    "mtime_ns": source_revision["mtime_ns"]}:
                    raise ValueError(f"Table cache copy has an unexpected revision: {source}")
                (temporary / "manifest.json").write_text(
                    json.dumps({
                        "schema": TABLE_CACHE_SCHEMA,
                        "source": source_revision,
                        "cached": {"size": source_revision["size"],
                                    "mtime_ns": source_revision["mtime_ns"]},
                    }, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, entry)
                temporary = None
            finally:
                if temporary is not None:
                    shutil.rmtree(temporary)
        return _read_table_cache(entry, source_revision)


def _copy_cached_table(source: Path, root: Path, relative: str,
                       files: list[dict], cache_root: Path) -> Path:
    previous = next((item for item in files if item["replica"] == relative), None)
    if previous is not None:
        if previous["source"] != file_revision(source):
            raise ValueError(f"Repeated delivery path has a different source: {relative}")
        return _inside(root, relative)
    destination = _inside(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = file_revision(source)
    cache_path = _materialize_table_cache(source, cache_root, before)
    cached_revision = _stat(cache_path)
    shutil.copyfile(cache_path, destination)
    os.utime(destination, ns=(before["mtime_ns"], before["mtime_ns"]))
    if _stat(cache_path) != cached_revision:
        raise ValueError(f"Table cache content changed during packet copy: {cache_path}")
    if file_revision(source) != before:
        raise ValueError(f"Source changed during cached table delivery: {source}")
    if _stat(destination) != {"size": before["size"],
                              "mtime_ns": before["mtime_ns"]}:
        raise ValueError(f"Cached table replica has an unexpected revision: {relative}")
    files.append({"source": before, "replica": relative,
                  "revision": {"size": before["size"],
                                "mtime_ns": before["mtime_ns"]}})
    return destination


def _record_source(source: Path, relative: str, files: list[dict], backed: list[str]) -> None:
    before = file_revision(source)
    if any(item["replica"] == relative for item in files):
        raise ValueError(f"Repeated delivery path: {relative}")
    files.append({"source": before, "replica": relative,
                  "revision": {"size": before["size"], "mtime_ns": before["mtime_ns"]}})
    backed.append(relative)


def _send_plan(packet_root: Path) -> tuple[Path, dict, set[str]]:
    plan_path = packet_root / SEND_PLAN
    envelope = packet_root / (SEND_ENVELOPE if plan_path.is_file() else "ready.json")
    ready = json.loads(envelope.read_text())
    names = json.loads(plan_path.read_text())["source_backed"] if plan_path.is_file() else []
    backed = set(names)
    if len(backed) != len(names):
        raise ValueError("Sending plan repeats a source-backed file")
    if backed:
        if (packet_root / "ready.json").exists() or len(backed) > 2:
            raise ValueError("Invalid source-backed sending plan")
        declared = {item["replica"]: item for item in ready["files"]}
        if len(declared) != len(ready["files"]) or not backed.issubset(declared):
            raise ValueError("Source-backed sending plan differs from envelope")
        inputs = pd.read_parquet(packet_root / "images.parquet")
        selected = inputs[inputs.image_id.isin(ready["image_ids"])]
        allowed = set(selected.input_image_path)
        allowed.update(f"reuse/{image_id}/products/image.tiff" for image_id in ready["reuse_metadata"])
        if not backed.issubset(allowed):
            raise ValueError("Sending plan names an ineligible source")
        for item in ready["files"]:
            replica = _inside(packet_root, item["replica"])
            source = Path(item["source"]["path"])
            if item["replica"] in backed:
                if replica.exists() or file_revision(source) != item["source"]:
                    raise ValueError(f"Source-backed file changed or materialized: {replica}")
            elif not replica.is_file() or not stat.S_ISREG(replica.lstat().st_mode) or _stat(replica) != item["revision"]:
                raise ValueError(f"Incomplete staged delivery file: {replica}")
    else:
        validate_delivery(packet_root, envelope, check_weights=False)
    return envelope, ready, backed


def _assert_sources_unchanged(files: list[dict], *, study_root: Path) -> None:
    """Keep table snapshots while the caller rechecks their current D01 scope."""
    for item in files:
        source = Path(item["source"]["path"])
        if source in {study_root / "images.parquet", study_root / "review_results.csv"}:
            continue
        if file_revision(source) != item["source"]:
            raise ValueError(f"Source changed after packet copy: {item['source']['path']}")


@contextmanager
def _source_lease(study_root: Path, data_root: Path, images: pd.DataFrame):
    with ExitStack() as stack:
        if not (study_root / ".publication.lock").is_file():
            raise ValueError("Existing study input lock required for read-only source delivery")
        stack.enter_context(resource_lock(study_root, shared=True))
        paths = sorted({(data_root / value).resolve().parent for value in images.input_image_path})
        for parent in paths:
            if not (parent / ".input.lock").is_file():
                raise ValueError("Existing image input lock required for read-only source delivery")
            stack.enter_context(resource_lock(parent, shared=True, name=".input.lock"))
        yield


def build_delivery(study_root: Path, data_root: Path, image_ids: list[str],
                   mask_bindings: dict[str, dict], settings_path: Path,
                   weights_path: Path, s1_path: Path, bank_dir: Path,
                   node_id: str, packet_root: Path,
                   weights_revision: dict | None = None, *,
                   selected_image_ids: list[str] | None = None,
                   seed_metadata: dict[str, Path] | None = None,
                   header_cache: dict | None = None,
                   source_backed: bool = False,
                   table_cache_root: Path | None = None,
                   execution_profile: dict | None = None,
                   allow_unknown_mask_segmentation: bool = False) -> Path:
    """Build one accepted packet; source-backed packets remain sending plans."""
    if node_id not in {"72", "73"} or packet_root.exists():
        raise ValueError("Node must be 72/73 and packet path must be new")
    if not (study_root / ".publication.lock").is_file():
        raise ValueError("Existing study input lock required for read-only source delivery")
    declared = pd.read_parquet(study_root / "images.parquet")
    delivery_ids = selected_image_ids if selected_image_ids is not None else image_ids
    if not delivery_ids or len(delivery_ids) != len(set(delivery_ids)) or not set(delivery_ids).issubset(image_ids):
        raise ValueError("Selected delivery images must belong to the original accepted D01 scope")
    if source_backed and len(delivery_ids) != 1:
        raise ValueError("Source-backed sending plans require one selected image")
    selected = declared[declared.image_id.isin(delivery_ids)]
    if len(selected) != len(set(delivery_ids)):
        raise ValueError("Requested image IDs do not match the study input table")
    for value in selected.input_image_path:
        if not ((data_root / value).resolve().parent / ".input.lock").is_file():
            raise ValueError("Existing image input lock required for read-only source delivery")
    inputs = load_reviewed_study_inputs(study_root, data_root=data_root, image_ids=delivery_ids,
                                        accepted_scope_image_ids=image_ids, header_cache=header_cache)
    ids = sorted(inputs.images.image_id.tolist())
    if set(mask_bindings) != set(ids):
        raise ValueError("Explicit mask state required for every selected image")
    policy_enabled = (allow_unknown_mask_segmentation and any(
        binding.get("status") == "unknown" for binding in mask_bindings.values()))
    profile = checked_execution_profile(node_id, execution_profile)
    precision, batch = profile["precision"], profile["batch_size"]
    settings = load_settings(settings_path, packet_root / "cache", weights=weights_path,
                             precision=precision, batch_size=batch,
                             allow_unknown_mask_segmentation=policy_enabled)
    files: list[dict] = []
    requests: dict[str, dict] = {}
    copied_masks: dict[str, dict] = {}
    reuse: dict[str, str] = {}
    backed: list[str] = []
    table_cache = (Path(table_cache_root).resolve()
                   if source_backed and table_cache_root is not None else None)
    packet_root.mkdir(parents=True)
    try:
        for name in TABLE_CACHE_NAMES:
            if table_cache is None:
                _copy(study_root / name, packet_root, name, files)
            else:
                _copy_cached_table(study_root / name, packet_root, name, files, table_cache)
        _copy(settings_path, packet_root, "resources/settings.yaml", files)
        _copy(s1_path, packet_root, "resources/s1.csv", files)
        for name in ("manifest.json", "gene_symbols.json", "embeddings.pt"):
            _copy(bank_dir / name, packet_root, f"resources/bank/{name}", files)
        for row in inputs.images.itertuples():
            image_id = row.image_id
            binding = dict(mask_bindings[image_id])
            record, source, panel = inputs.source(image_id, binding)
            for key in ("input_image_path", "valid_pixels_path", "sample_region_path"):
                value = getattr(row, key, None)
                if pd.notna(value) and value:
                    path = (data_root / value).resolve()
                    backed_input = source_backed and key == "input_image_path"
                    with formal_read_lease(path) if source_backed and not backed_input else nullcontext():
                        with input_header_lease(source.path, study_root=study_root):
                            assert_reviewed_binding(study_root, data_root=data_root, binding=inputs.binding)
                            if backed_input:
                                _record_source(path, str(value), files, backed)
                            else:
                                _copy(path, packet_root, str(value), files)
            if binding["status"] == "bound":
                mask_source = (data_root / binding["path"]).resolve()
                if not mask_source.is_relative_to(data_root.resolve()):
                    raise ValueError("Mask source is outside declared source root")
                binding["path"] = str(mask_source.relative_to(data_root.resolve()))
                with formal_read_lease(mask_source) if source_backed else nullcontext():
                    _copy(mask_source, packet_root, binding["path"], files)
            copied_masks[image_id] = binding
            requests[image_id] = preparation_request(source, panel, settings,
                (int(row.height_px), int(row.width_px)))
            if weights_revision is not None:
                requests[image_id]["segmentation"]["weights"] = weights_revision
            metadata = ((seed_metadata or {}).get(image_id) or
                        data_root / "processed" / record.study / "images" / image_id / "metadata.json")
            if metadata.is_file():
                metadata_revision = file_revision(metadata)
                old = json.loads(metadata.read_text())
                products = old.get("products", {})
                old_image_request = old.get("image_request", old.get("request", {}).get("image"))
                if (products and old_image_request == requests[image_id]["image"]
                        and not old.get("publication_incomplete")
                        and old.get("shape_yx") == [int(row.height_px), int(row.width_px)]
                        and set(old.get("completed", [])) >= set(products)
                        and all(Path(value).resolve().is_relative_to(metadata.parent.resolve())
                                and Path(value).is_file() for value in products.values())):
                    copied_products = [Path(path) for product, path in products.items()
                                       if not (source_backed and product == "image" and Path(path).suffix == ".tiff")]
                    with formal_read_lease(copied_products[0]) if source_backed and copied_products else nullcontext():
                        if source_backed and not (metadata.parent / ".publication.lock").is_file():
                            raise ValueError("Existing checkpoint publication lock required for source-backed delivery")
                        with resource_lock(metadata.parent, shared=True) if source_backed else nullcontext():
                            if file_revision(metadata) != metadata_revision:
                                raise ValueError("Closed checkpoint metadata changed during packet build")
                            if not source_backed:
                                validate_products(products, (int(row.height_px), int(row.width_px)))
                            relative = f"reuse/{image_id}/metadata.json"
                            _copy(metadata, packet_root, relative, files)
                            reuse[image_id] = relative
                            for product, path in products.items():
                                replica = f"reuse/{image_id}/products/{product}{Path(path).suffix}"
                                if source_backed and product == "image" and Path(path).suffix == ".tiff":
                                    continue
                                else:
                                    _copy(Path(path), packet_root, replica, files)
                    if source_backed and "image" in products and Path(products["image"]).suffix == ".tiff":
                        with resource_lock(metadata.parent, shared=True):
                            if file_revision(metadata) != metadata_revision:
                                raise ValueError("Closed checkpoint metadata changed during packet build")
                            _record_source(Path(products["image"]),
                                           f"reuse/{image_id}/products/image.tiff", files, backed)
        with _source_lease(study_root, data_root, inputs.images):
            assert_reviewed_binding(study_root, data_root=data_root, binding=inputs.binding)
            _assert_sources_unchanged(files, study_root=study_root)
        ready = {"schema": "cellweft.study_delivery.v1", "study": str(inputs.images.study.iloc[0]),
                 "node_id": node_id, "image_ids": ids, "source_bindings": inputs.binding,
                 "created_at_ns": time.time_ns(),
                 "mask_bindings": copied_masks, "requests": requests, "settings": "resources/settings.yaml",
                 "weights": str(weights_path), "weights_revision": weights_revision,
                 "s1": "resources/s1.csv", "bank_dir": "resources/bank",
                 "reuse_metadata": reuse, "files": files}
        if policy_enabled:
            ready["mask_policy"] = {"unknown_status": "cpsam_v2"}
        if execution_profile is not None:
            ready["execution_profile"] = profile
        envelope = packet_root / (SEND_ENVELOPE if source_backed else "ready.json")
        envelope.write_text(json.dumps(ready, indent=2) + "\n")
        if source_backed:
            (packet_root / SEND_PLAN).write_text(json.dumps({"source_backed": backed}) + "\n")
            _send_plan(packet_root)
        else:
            validate_delivery(packet_root, envelope, check_weights=False)
    except BaseException:
        (packet_root / "ready.json").unlink(missing_ok=True)
        (packet_root / SEND_ENVELOPE).unlink(missing_ok=True)
        (packet_root / SEND_PLAN).unlink(missing_ok=True)
        raise
    return packet_root


def upload_delivery(packet_root: Path, destination: Path, transport: Transport,
                    study_root: Path, data_root: Path, *,
                    reserve_bytes: int = 100 * 1024**3,
                    previous_delivery: Path | None = None) -> dict | None:
    """Stream one closed packet, then publish only a validated remote replica."""
    envelope, ready, backed = _send_plan(packet_root)
    checkpoint_locks = {}
    for image_id, relative in ready["reuse_metadata"].items():
        product = f"reuse/{image_id}/products/image.tiff"
        if product in backed:
            metadata_item = next(item for item in ready["files"] if item["replica"] == relative)
            checkpoint_locks[product] = Path(metadata_item["source"]["path"]).parent
    selected_images = pd.read_parquet(packet_root / "images.parquet")
    selected_images = selected_images[selected_images.image_id.isin(ready["image_ids"])]
    with _source_lease(study_root, data_root, selected_images):
        assert_reviewed_binding(study_root, data_root=data_root, binding=ready["source_bindings"])
        _assert_sources_unchanged(ready["files"], study_root=study_root)
    timings = {"query_started_ns": time.time_ns()}
    exists = ("import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);"
              "print(json.dumps(json.loads(p.read_text())) if p.is_file() else 'null')")
    existing = json.loads(transport.ssh(transport.gateway,
        [transport.gateway["python"], "-c", exists, str(destination / "ready.json")],
        capture_output=True, text=True).stdout)
    if existing is not None:
        if existing != ready:
            raise ValueError("Existing immutable delivery has a different envelope")
        code = "from pathlib import Path;import sys;from cellweft.data.preparation.delivery import load_delivery;load_delivery(Path(sys.argv[1]))"
        transport.ssh(transport.gateway, ["env", f"PYTHONPATH={transport.gateway['code_root']}/src",
            transport.gateway["python"], "-c", code, str(destination)])
        return {"previous_delivery": str(previous_delivery) if previous_delivery is not None else None,
                "missing": [], "streamed_bytes": 0, "source_streamed_bytes": 0,
                "avoided_stream_bytes": sum(item["revision"]["size"] for item in ready["files"]),
                "built_bytes": sum(item["revision"]["size"] for item in ready["files"] if item["replica"] not in backed),
                "source_backed_bytes": sum(item["revision"]["size"] for item in ready["files"] if item["replica"] in backed),
                "phase_seconds": {"query": (time.time_ns() - timings["query_started_ns"]) / 1e9}}
    remote = ["env", f"PYTHONPATH={transport.gateway['code_root']}/src",
              transport.gateway["python"], "-m",
              "cellweft.data.preparation.delivery_transfer"]
    result = transport.ssh(transport.gateway,
        [*remote, "query", str(transport.root), str(destination),
         "--reserve-bytes", str(reserve_bytes),
         *(["--previous-delivery", str(previous_delivery)] if previous_delivery is not None else [])],
        input=json.dumps(ready),
        capture_output=True, text=True, timeout=120)
    token, *missing = json.loads(result.stdout)
    timings["query_finished_ns"] = time.time_ns()
    selected = {item["replica"]: item for item in ready["files"] if item["replica"] in missing}
    if len(missing) != len(set(missing)) or set(selected) != set(missing):
        raise ValueError("Remote query requested an undeclared file")
    streamed_bytes = sum(selected[name]["revision"]["size"] for name in missing)
    stream_timeout = _delivery_stream_timeout(streamed_bytes)
    timings["stream_started_ns"] = time.time_ns()
    if missing:
        reader_fd, writer_fd = os.pipe()
        failures: list[BaseException] = []

        def produce() -> None:
            try:
                with os.fdopen(writer_fd, "wb") as stream, tarfile.open(
                        fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for relative in missing:
                        item = selected[relative]
                        source = Path(item["source"]["path"]) if relative in backed else _inside(packet_root, relative)
                        with formal_read_lease(source) if relative in backed else nullcontext():
                            with _source_lease(study_root, data_root, selected_images) if relative in backed else nullcontext():
                                checkpoint_dir = checkpoint_locks.get(relative)
                                if checkpoint_dir is not None and not (checkpoint_dir / ".publication.lock").is_file():
                                    raise ValueError("Closed checkpoint publication lock disappeared")
                                with resource_lock(checkpoint_dir, shared=True) if checkpoint_dir is not None else nullcontext():
                                    if relative in backed and file_revision(source) != item["source"]:
                                        raise ValueError(f"Source changed before stream: {source}")
                                    if _stat(source) != item["revision"]:
                                        raise ValueError(f"Delivery file changed: {relative}")
                                    info = tarfile.TarInfo(relative)
                                    info.size = item["revision"]["size"]
                                    info.mode = 0o600
                                    with source.open("rb") as payload:
                                        archive.addfile(info, payload)
                                    if relative in backed and file_revision(source) != item["source"]:
                                        raise ValueError(f"Source changed during stream: {source}")
                                    if _stat(source) != item["revision"]:
                                        raise ValueError(f"Delivery file changed during stream: {relative}")
            except BaseException as exc:
                failures.append(exc)

        producer = threading.Thread(target=produce, name="delivery-tar", daemon=True)
        producer.start()
        transfer_error: BaseException | None = None
        try:
            with os.fdopen(reader_fd, "rb") as stream:
                transport.ssh(transport.gateway,
                    [*remote, "receive", str(transport.root), str(destination),
                     "--token", token, "--node", transport.gateway.get("id", ready["node_id"]),
                     "--lock-root", str(getattr(transport, "lock_root", transport.root / "transfer_locks"))],
                    stdin=stream, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    timeout=stream_timeout)
        except BaseException as exc:
            transfer_error = exc
        finally:
            producer.join(timeout=30)
        if producer.is_alive():
            raise TimeoutError("Delivery tar producer did not stop")
        if failures or transfer_error is not None:
            _write_transfer_failure(packet_root, timeout_seconds=stream_timeout,
                                    missing=missing, producer_errors=failures,
                                    transfer_error=transfer_error)
        if failures:
            raise failures[0]
        if transfer_error is not None:
            raise transfer_error
    timings["stream_finished_ns"] = time.time_ns()
    with _source_lease(study_root, data_root, selected_images):
        assert_reviewed_binding(study_root, data_root=data_root, binding=ready["source_bindings"])
        _assert_sources_unchanged(ready["files"], study_root=study_root)
    pending = destination / ".ready.pending.json"
    transport.upload(envelope, pending)
    with _source_lease(study_root, data_root, selected_images):
        assert_reviewed_binding(study_root, data_root=data_root, binding=ready["source_bindings"])
        _assert_sources_unchanged(ready["files"], study_root=study_root)
        code = ("from pathlib import Path;import os,sys;"
                "from cellweft.data.preparation.delivery import validate_delivery;"
                "r=Path(sys.argv[1]);validate_delivery(r,r/'.ready.pending.json',check_reuse_products=True);"
                "os.link(r/'.ready.pending.json',r/'ready.json');"
                "(r/'.ready.pending.json').unlink()")
        transport.ssh(transport.gateway, ["env", f"PYTHONPATH={transport.gateway['code_root']}/src",
            transport.gateway["python"], "-c", code, str(destination)])
    timings["ready_ns"] = time.time_ns()
    return {"previous_delivery": str(previous_delivery) if previous_delivery is not None else None,
            "missing": missing, "streamed_bytes": streamed_bytes,
            "avoided_stream_bytes": sum(item["revision"]["size"] for item in ready["files"]) - streamed_bytes,
            "built_bytes": sum(item["revision"]["size"] for item in ready["files"] if item["replica"] not in backed),
            "source_backed_bytes": sum(item["revision"]["size"] for item in ready["files"] if item["replica"] in backed),
            "source_streamed_bytes": sum(selected[name]["revision"]["size"] for name in missing if name in backed),
            "phase_seconds": {"query": (timings["query_finished_ns"] - timings["query_started_ns"]) / 1e9,
                              "stream": (timings["stream_finished_ns"] - timings["stream_started_ns"]) / 1e9,
                              "validate_and_ready": (timings["ready_ns"] - timings["stream_finished_ns"]) / 1e9}}


def _header(path: Path, shape: tuple[int, int], count: int, dtype: str | None = None) -> None:
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != count or any(page.shape != shape or
                (dtype is not None and str(page.dtype) != dtype) for page in tif.pages):
            raise ValueError(f"Delivery TIFF header changed: {path}")


def load_delivery(packet_root: Path) -> StudyDelivery:
    """Validate a ready envelope, every declared file and copied input headers."""
    return validate_delivery(packet_root, packet_root / "ready.json")


def validate_delivery(packet_root: Path, envelope: Path, *, check_weights: bool = True,
                      check_reuse_products: bool = False) -> StudyDelivery:
    """Validate a pending envelope before its atomic ready publication."""
    root = packet_root.resolve()
    ready = json.loads(envelope.read_text())
    if ready.get("schema") != "cellweft.study_delivery.v1":
        raise ValueError("Unknown delivery schema")
    node_id = ready["node_id"]
    if node_id not in {"72", "73"}:
        raise ValueError("Unsupported delivery node")
    if "execution_profile" in ready and ready["execution_profile"] is None:
        raise ValueError("Explicit execution profile cannot be null")
    profile = checked_execution_profile(node_id, ready.get("execution_profile"))
    policy = ready.get("mask_policy", {})
    if policy not in ({}, {"unknown_status": "cpsam_v2"}):
        raise ValueError("Unsupported mask policy")
    allow_unknown_mask_segmentation = policy == {"unknown_status": "cpsam_v2"}
    image_ids = ready["image_ids"]
    if not image_ids or len(image_ids) != len(set(image_ids)) or any(
            Path(value).name != value or value in {".", ".."} for value in image_ids):
        raise ValueError("Invalid delivery image scope")
    for item in ready["files"]:
        replica = _inside(root, item["replica"])
        if not replica.is_file() or _stat(replica) != item["revision"]:
            raise ValueError(f"Incomplete or changed delivery file: {replica}")
        if item["source"]["size"] != item["revision"]["size"] or item["source"]["mtime_ns"] != item["revision"]["mtime_ns"]:
            raise ValueError(f"Source/replica revision differs: {replica}")
    listed = {item["replica"] for item in ready["files"]}
    provenance = {str(_inside(root, item["replica"])): item["source"]
                  for item in ready["files"]}
    if len(listed) != len(ready["files"]):
        raise ValueError("Duplicate replica path")
    required = {"images.parquet", "panel_mapping.csv", "sample_metadata.csv",
                "field_mapping.csv", "review_results.csv", "resources/settings.yaml",
                "resources/s1.csv", "resources/bank/manifest.json",
                "resources/bank/gene_symbols.json", "resources/bank/embeddings.pt"}
    if not required.issubset(listed):
        raise ValueError("Delivery has missing input tables or resources")
    if check_weights and ready.get("weights_revision") is not None:
        if file_revision(Path(ready["weights"])) != ready["weights_revision"]:
            raise ValueError("Target weights version differs from frozen request")
    images = pd.read_parquet(root / "images.parquet")
    panel = pd.read_csv(root / "panel_mapping.csv", keep_default_na=False)
    samples = pd.read_csv(root / "sample_metadata.csv", keep_default_na=False)
    fields = pd.read_csv(root / "field_mapping.csv", keep_default_na=False)
    images = validate_image_metadata(images, samples, panel, fields)
    if not set(image_ids).issubset(set(images.image_id)) or images.study.nunique() != 1 or images.study.iloc[0] != ready["study"]:
        raise ValueError("Delivery image table disagrees with envelope")
    images = images[images.image_id.isin(image_ids)].copy()
    source_bindings = ready["source_bindings"]
    current_images = images.set_index("image_id").loc[sorted(image_ids)].reset_index()
    original_rows = [row for row in source_bindings["images"] if row["image_id"] in image_ids]
    if json.loads(current_images.to_json(orient="records")) != original_rows:
        raise ValueError("Copied image rows differ from original D01 binding")
    current_panel = panel[panel.panel_id.isin(images.panel_id)]
    current_panel = current_panel[[key for key in PANEL_INPUT_FIELDS if key in current_panel]]
    current_panel = current_panel.sort_values(["panel_id", "source_channel_index"])
    original_panel = [row for row in source_bindings["panel"] if row["panel_id"] in set(images.panel_id)]
    if json.loads(current_panel.to_json(orient="records")) != original_panel:
        raise ValueError("Copied panel differs from original D01 binding")
    bindings = ready["mask_bindings"]
    requests = ready["requests"]
    if set(bindings) != set(image_ids) or set(requests) != set(image_ids):
        raise ValueError("Delivery scope has missing mask bindings or requests")
    for row in images.itertuples():
        shape = (int(row.height_px), int(row.width_px))
        selected = panel[panel.panel_id == row.panel_id]
        validate_panel_mapping(selected, input_channel_count=int(row.channel_count))
        if row.input_image_path not in listed:
            raise ValueError("Input image absent from replica proof")
        _header(_inside(root, row.input_image_path), shape, int(row.channel_count), str(row.source_dtype))
        for name in ("valid_pixels_path", "sample_region_path"):
            value = getattr(row, name, None)
            if pd.notna(value) and value:
                if value not in listed:
                    raise ValueError("Domain image absent from replica proof")
                _header(_inside(root, value), shape, 1, "uint8")
        binding = bindings[row.image_id]
        if binding["status"] == "bound":
            if binding["path"] not in listed:
                raise ValueError("Mask was not included in delivery")
        elif binding["status"] not in {"unknown", "absent", "unusable"}:
            raise ValueError("Invalid mask status")
        request = requests[row.image_id]
        if request.get("schema") != "cellweft.post_input.v2" or request.get("image", {}).get("source", {}).get("image_id") != row.image_id:
            raise ValueError("Frozen request does not match image")
        settings = load_settings(root / "resources/settings.yaml", root / "cache",
            weights=Path(ready["weights"]), precision=profile["precision"],
            batch_size=profile["batch_size"],
            allow_unknown_mask_segmentation=allow_unknown_mask_segmentation)
        physical_source, selected_panel = ReviewedStudyInputs(
            root, root, images, panel, samples, fields, ready["source_bindings"]
        ).source(row.image_id, binding)[1:]
        expected = preparation_request(physical_source, selected_panel, settings, shape)
        canonical_source = source_identity(physical_source)
        for key in ("path", "valid_mask", "sample_region"):
            if canonical_source.get(key):
                canonical_source[key] = provenance[canonical_source[key]]["path"]
        canonical_source["file_identity"] = [provenance[item["path"]]
                                             for item in canonical_source["file_identity"]]
        if request["image"]["source"] != canonical_source:
            raise ValueError("Frozen source identity differs from copied input")
        if request["image"]["panel"] != expected["image"]["panel"] or request["image"]["normalization"] != expected["image"]["normalization"]:
            raise ValueError("Frozen panel or normalization differs from copied inputs")
        if binding["status"] != "bound" and request["mask"] != expected["mask"]:
            raise ValueError("Frozen mask state differs from copied binding")
        if binding["status"] == "bound":
            mask_proof = provenance[str(_inside(root, binding["path"]))]
            if request["mask"].get("file_identity") != mask_proof or request["mask"].get("shape_yx") != list(shape):
                raise ValueError("Frozen mask identity differs from copied author mask")
        if request["segmentation"]["parameters"] != expected["segmentation"]["parameters"] or request["segmentation"]["precision"] != settings.precision or request["sampling"] != expected["sampling"]:
            raise ValueError("Frozen processing settings differ from node contract")
        if request["segmentation"]["weights"] != ready.get("weights_revision"):
            raise ValueError("Frozen weights identity differs from target")
    if not set(image_ids).issubset(source_bindings["image_ids"]):
        raise ValueError("Packet is outside original source D01 scope")
    accepted_assets = {item["path"]: item for item in source_bindings["assets"]}
    for image_id in image_ids:
        identity = requests[image_id]["image"]["source"]
        for file in identity["file_identity"]:
            if accepted_assets.get(file["path"]) != file:
                raise ValueError("Canonical image identity differs from accepted D01")
    for item in ready["files"]:
        if item["replica"] in {"images.parquet", "panel_mapping.csv", "sample_metadata.csv", "field_mapping.csv", "review_results.csv"}:
            continue
        if item["replica"].startswith("reuse/") or item["replica"].startswith("resources/"):
            continue
        if item["source"]["path"] not in accepted_assets and not any(
            item["source"]["path"] == value.get("path") for value in (
                requests[image_id].get("mask", {}).get("file_identity", {}) for image_id in image_ids)):
            raise ValueError("Undeclared copied asset")
    reused = {image_id: _inside(root, relative) for image_id, relative in ready["reuse_metadata"].items()}
    if any(relative not in listed for relative in ready["reuse_metadata"].values()):
        raise ValueError("Reuse metadata not included in delivery")
    if check_reuse_products:
        if not set(reused).issubset(image_ids):
            raise ValueError("Reuse metadata is outside delivery image scope")
        revisions = {item["replica"]: item["revision"] for item in ready["files"]}
        for image_id, metadata_path in reused.items():
            if ready["reuse_metadata"][image_id] != f"reuse/{image_id}/metadata.json":
                raise ValueError(f"Unexpected reuse metadata path: {image_id}")
            old = json.loads(metadata_path.read_text())
            products = old.get("products", {})
            row = images.set_index("image_id").loc[image_id]
            shape = (int(row.height_px), int(row.width_px))
            if (not products or old.get("publication_incomplete")
                    or old.get("image_request", old.get("request", {}).get("image")) != requests[image_id]["image"]
                    or old.get("shape_yx") != list(shape)
                    or not set(old.get("completed", [])).issuperset(products)):
                raise ValueError(f"Reuse metadata differs from frozen request: {image_id}")
            packet_products = {}
            for name, original in products.items():
                relative = f"reuse/{image_id}/products/{name}{Path(original).suffix}"
                if relative not in listed:
                    raise ValueError(f"Reuse product absent from delivery: {relative}")
                packet_products[name] = str(_inside(root, relative))
            expected = {f"reuse/{image_id}/metadata.json", *(
                f"reuse/{image_id}/products/{name}{Path(original).suffix}"
                for name, original in products.items())}
            if {relative for relative in listed if relative.startswith(f"reuse/{image_id}/")} != expected:
                raise ValueError(f"Reuse product list differs from metadata: {image_id}")
            validate_products(packet_products, shape)
            for relative in expected:
                if _stat(_inside(root, relative)) != revisions[relative]:
                    raise ValueError(f"Reuse product changed during validation: {relative}")
    inputs = ReviewedStudyInputs(root, root, images, panel, samples, fields, source_bindings)
    return StudyDelivery(root, ready["study"], node_id, image_ids, inputs, bindings, requests,
                         _inside(root, ready["settings"]), Path(ready["weights"]),
                         _inside(root, ready["s1"]), _inside(root, ready["bank_dir"]), reused,
                         source_bindings, profile, allow_unknown_mask_segmentation)
