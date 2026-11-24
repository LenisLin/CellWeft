"""Study-local downstream worker for an immutable, reviewed delivery."""
from __future__ import annotations

import fcntl
import json
import shutil
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from cellweft.data.io.readers import open_source
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.capacity import image_work_bytes
from cellweft.data.preparation.embeddings import _panel_identity, _same_panel_identity, _validate, build_study_embeddings, publish_study_embeddings
from cellweft.data.preparation.inputs import PANEL_INPUT_FIELDS, assert_reviewed_acceptance, file_revision
from cellweft.data.preparation.pipeline import (_completed_segmentation_result,
    checked_canonical_request, preparation_request, prepare_image)
from cellweft.data.preparation.segmentation import CellposeExecutor
from cellweft.data.preparation.validation import validate_products
from cellweft.data.preparation.remote_worker import shared_disk_reservation


NODE_CONFIG = {"72": (4, "fp32", 8), "73": (2, "bf16", 16)}


def _table_revision(packet) -> dict:
    envelope = json.loads((packet.root / "ready.json").read_text())
    names = {"images.parquet", "sample_metadata.csv", "field_mapping.csv",
             "review_results.csv", "panel_mapping.csv"}
    return {item["replica"]: item["revision"] for item in envelope["files"]
            if item["replica"] in names}


def _binding_current(binding: dict, images, panel, reviews, tables: dict) -> bool:
    """Check current table/review evidence; the loader verifies physical replicas."""
    ids = binding["image_ids"]
    if not set(ids).issubset(images.image_id):
        return False
    rows = images.set_index("image_id").loc[sorted(ids)].reset_index()
    if json.loads(rows.to_json(orient="records")) != binding["images"]:
        return False
    selected_panel = panel[panel.panel_id.isin(rows.panel_id)]
    selected_panel = selected_panel[[key for key in PANEL_INPUT_FIELDS if key in panel]]
    selected_panel = selected_panel.sort_values(["panel_id", "source_channel_index"])
    if json.loads(selected_panel.to_json(orient="records")) != binding["panel"]:
        return False
    for file, key in (("sample_metadata.csv", "samples"), ("field_mapping.csv", "field_mapping")):
        if tables[file] != {name: binding[key][name] for name in ("size", "mtime_ns")}:
            return False
    try:
        assert_reviewed_acceptance(reviews, binding)
    except ValueError:
        return False
    return True


def _refresh_tables(packet, study_root: Path) -> bool:
    """Keep the newest full view and every still-approved delivered scope."""
    revision = _table_revision(packet)
    receipt = study_root / "node_input_snapshot.json"
    with resource_lock(study_root):
        old = json.loads(receipt.read_text()) if receipt.is_file() else None
        original_snapshot = old
        heads = dict(old.get("image_deliveries", {})) if old else {}
        envelope = json.loads((packet.root / "ready.json").read_text())
        sequence = envelope.get("created_at_ns", 0)
        resource_delivery = (old.get("resource_delivery", old["delivery"]) if old else str(packet.root))
        resource_sequence = old.get("resource_delivery_created_at_ns", 0) if old else sequence
        def order(tables):
            return tuple(tables[name]["mtime_ns"] for name in
                         ("review_results.csv", "images.parquet", "panel_mapping.csv",
                          "sample_metadata.csv", "field_mapping.csv"))
        candidates = list(old.get("bindings", [old["source_binding"]])) if old else []
        if packet.source_bindings not in candidates:
            candidates.append(packet.source_bindings)
        table_refresh = old is None or order(revision) > order(old["tables"])
        retained_panel = None
        vector_path = study_root / "marker_embeddings.pt"
        if (table_refresh or sequence > resource_sequence) and vector_path.is_file():
            if (study_root / "panel_mapping.csv").is_file():
                import torch
                previous_panel = pd.read_csv(study_root / "panel_mapping.csv", keep_default_na=False)
                previous_panel['embedding_row'] = pd.to_numeric(
                    previous_panel.embedding_row, errors='coerce').astype('Int64')
                payload = torch.load(vector_path, map_location='cpu', weights_only=True)
                incoming_panel = (pd.read_csv(packet.inputs.root / 'panel_mapping.csv', keep_default_na=False)
                                  if table_refresh else previous_panel)
                def version(value):
                    return {key: value[key] for key in ('size', 'mtime_ns')}
                resources_match = (
                    version(payload['metadata']['bank']) == version(file_revision(packet.bank_dir / 'embeddings.pt'))
                    and payload['metadata']['bank_manifest'] == json.loads((packet.bank_dir / 'manifest.json').read_text())
                    and version(payload['metadata']['gene_keys']) == version(file_revision(packet.bank_dir / 'gene_symbols.json'))
                    and version(payload['metadata']['manifest_file']) == version(file_revision(packet.bank_dir / 'manifest.json'))
                    and version(payload['metadata']['s1']) == version(file_revision(packet.s1_path)))
                selected_current = set(pd.read_parquet(
                    packet.inputs.root / 'images.parquet' if table_refresh
                    else study_root / 'images.parquet').panel_id.astype(str))
                scope_match = ('panel_only_source_versions' not in payload['metadata']
                               or set(payload['metadata']['panel_ids']) == selected_current)
                if (_same_panel_identity(_panel_identity(previous_panel), _panel_identity(incoming_panel))
                        and resources_match and scope_match):
                    _validate(payload, previous_panel)
                    retained_panel = previous_panel
            if retained_panel is None:
                vector_path.unlink(missing_ok=True)
                if not table_refresh and (study_root / 'panel_mapping.csv').is_file():
                    cleared = pd.read_csv(study_root / 'panel_mapping.csv', keep_default_na=False)
                    if 'embedding_row' in cleared:
                        cleared['embedding_row'] = pd.NA
                        temporary = study_root / '.panel_mapping.csv.pending'
                        cleared.to_csv(temporary, index=False, encoding='utf-8-sig')
                        os.replace(temporary, study_root / 'panel_mapping.csv')
        if table_refresh:
            for table in revision:
                target = study_root / table
                temporary = target.with_name(f".{table}.pending")
                if table == 'panel_mapping.csv' and retained_panel is not None:
                    retained_panel.to_csv(temporary, index=False, encoding='utf-8-sig')
                else:
                    shutil.copy2(packet.inputs.root / table, temporary)
                os.replace(temporary, target)
            if retained_panel is None:
                vector_path.unlink(missing_ok=True)
            old = {"tables": revision, "delivery": str(packet.root),
                   "source_binding": packet.source_bindings}
        images = pd.read_parquet(study_root / "images.parquet")
        panel = pd.read_csv(study_root / "panel_mapping.csv", keep_default_na=False)
        reviews = pd.read_csv(study_root / "review_results.csv", keep_default_na=False)
        bindings = [binding for binding in candidates
                    if _binding_current(binding, images, panel, reviews, old["tables"])]
        if packet.source_bindings in bindings:
            if sequence > resource_sequence:
                resource_delivery, resource_sequence = str(packet.root), sequence
            for image_id in packet.image_ids:
                previous = heads.get(image_id)
                if previous is None or sequence > previous["created_at_ns"]:
                    heads[image_id] = {"created_at_ns": sequence, "delivery": str(packet.root)}
        panel_ids = {row["panel_id"] for binding in bindings for row in binding["images"]}
        updated_snapshot = {**old, "bindings": bindings, "panel_ids": sorted(panel_ids),
                            "resource_delivery": resource_delivery,
                            "resource_delivery_created_at_ns": resource_sequence,
                            "image_deliveries": heads}
        if updated_snapshot != original_snapshot:
            write_json(receipt, updated_snapshot)
    return packet.source_bindings in bindings


def _current_delivery(study_root: Path, packet, image_id: str) -> bool:
    snapshot = json.loads((study_root / "node_input_snapshot.json").read_text())
    return (packet.source_bindings in snapshot["bindings"]
            and snapshot["image_deliveries"][image_id]["delivery"] == str(packet.root))


def _reviewed_zero_body_result(metadata: dict, request: dict) -> bool:
    task = metadata.get("segmentation_task")
    if (not task or metadata.get("request") != request
            or task.get("request") != request
            or metadata.get("publication_incomplete")
            or "mask" in metadata.get("completed", [])
            or "new zero-body mask requires one whole-image whole-cell judgment"
            not in metadata.get("gaps", [])):
        return False
    receipt = _completed_segmentation_result(task)
    return receipt is not None and receipt.get("whole_cell_confirmed") is True


def _terminal_status(path: Path, request: dict, *,
                     allow_unknown_mask_segmentation: bool = False) -> dict | None:
    if not path.is_file():
        return None
    status = json.loads(path.read_text())
    if status.get("status") not in {"computed", "quality_pending", "partial"}:
        return None
    metadata_path = Path(status["metadata"])
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    # A previous run may have intentionally stopped after normalization while
    # the mask binding was unknown.  Reopen that partial image when this run
    # explicitly enables the scoped cpsam_v2 fallback.
    if (allow_unknown_mask_segmentation and status.get("status") == "partial"
            and "mask" not in metadata.get("completed", [])
            and "source mask binding unknown" in " ".join(metadata.get("gaps", []))):
        return None
    task = metadata.get("segmentation_task")
    if status.get("status") == "partial" and _reviewed_zero_body_result(metadata, request):
        return None
    if (status.get("status") == "partial" and task is not None
            and "mask" not in metadata.get("completed", [])
            and "segmentation result pending" in metadata.get("gaps", [])
            and Path(task["result_path"]).is_file()
            and _completed_segmentation_result(task) is not None):
        return None
    if (metadata.get("request") == request and not metadata.get("publication_incomplete")
            and metadata.get("products")):
        revisions = {name: file_revision(Path(value)) for name, value in metadata["products"].items()
                     if Path(value).is_file()}
        if set(revisions) != set(metadata["products"]):
            return None
        if "product_revisions" not in status:
            try:
                validate_products(metadata["products"], tuple(metadata["shape_yx"]))
            except (OSError, ValueError, KeyError):
                return None
            status["product_revisions"] = revisions
            write_json(path, status)
        if status["product_revisions"] == revisions:
            return status
    return None


def _install_input_links(packet, image_id: str, node_root: Path) -> None:
    """Keep unchanged input paths in full tables resolvable on the owner NAS."""
    row = packet.inputs.images.set_index("image_id").loc[image_id]
    for name in ("input_image_path", "valid_pixels_path", "sample_region_path"):
        relative = row.get(name)
        if not isinstance(relative, str) or not relative:
            continue
        source = (packet.root / relative).resolve()
        target = (node_root / relative).resolve()
        if not target.is_relative_to(node_root.resolve()):
            raise ValueError("Input replica path is outside the owner NAS")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and (target.stat().st_size, target.stat().st_mtime_ns) == (source.stat().st_size, source.stat().st_mtime_ns):
            continue
        temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
        os.link(source, temporary)
        os.replace(temporary, target)


@contextmanager
def _claim(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _import_closed(packet, image_id: str, destination: Path, request: dict) -> bool:
    """Install a copied closed checkpoint only when every product reopens locally."""
    target = destination / "images" / image_id
    if (target / "metadata.json").exists():
        return False
    source_metadata = packet.reuse_metadata.get(image_id)
    if source_metadata is None:
        return False
    source_metadata = Path(source_metadata)
    old = json.loads(source_metadata.read_text())
    if (old.get("image_request", old.get("request", {}).get("image")) != request["image"]
            or old.get("publication_incomplete")):
        return False
    products = old.get("products", {})
    if not products or not set(old.get("completed", [])).issuperset(products):
        return False
    copied = {}
    packet_products = {}
    for name, original in products.items():
        asset = source_metadata.parent / "products" / f"{name}{Path(original).suffix}"
        if not asset.is_file():
            return False
        copied[name] = str(target / asset.name)
        packet_products[name] = str(asset)
    validate_products(packet_products, tuple(old["shape_yx"]))
    target.mkdir(parents=True, exist_ok=True)
    for name, asset in packet_products.items():
        shutil.copy2(asset, copied[name])
    write_json(target / "metadata.json", {**old, "products": copied})
    return True


def _publish_embeddings(packet, study_root: Path) -> str:
    with _claim(study_root / ".embedding_worker.lock") as owned:
        if not owned:
            return "busy"
        import torch
        snapshot = json.loads((study_root / "node_input_snapshot.json").read_text())
        if snapshot["resource_delivery"] != str(packet.root):
            from cellweft.data.preparation.delivery import load_delivery
            packet = load_delivery(Path(snapshot["resource_delivery"]))
        panel_ids = set(snapshot["panel_ids"])
        def version(value):
            return {key: value[key] for key in ("size", "mtime_ns")}
        panel = pd.read_csv(study_root / "panel_mapping.csv", keep_default_na=False)
        expected_panel = _panel_identity(panel)
        if (study_root / "marker_embeddings.pt").is_file():
            payload = torch.load(study_root / "marker_embeddings.pt", map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            panel_scope_current = (set(metadata["panel_ids"]) == panel_ids
                                   or ('panel_only_source_versions' in metadata
                                       and panel_ids.issubset(set(metadata["panel_ids"]))))
            if (_same_panel_identity(metadata["panel_identity"], expected_panel)
                    and panel_scope_current
                    and version(metadata["bank"]) == version(file_revision(packet.bank_dir / "embeddings.pt"))
                    and metadata["bank_manifest"] == json.loads((packet.bank_dir / "manifest.json").read_text())
                    and version(metadata["gene_keys"]) == version(file_revision(packet.bank_dir / "gene_symbols.json"))
                    and version(metadata["manifest_file"]) == version(file_revision(packet.bank_dir / "manifest.json"))
                    and version(metadata["s1"]) == version(file_revision(packet.s1_path))):
                return "present"
        bundle, mapped = build_study_embeddings(packet.study, panel,
                                                packet.s1_path, packet.bank_dir,
                                                panel_ids=panel_ids)
        bundle.payload["metadata"]["source_delivery"] = str(packet.root)
        publish_study_embeddings(bundle, mapped, study_root)
        return "complete"


def process_study_embeddings(packet_roots: list[Path], *, node_id: str,
                             node_root: Path) -> dict:
    """Refresh all ready scopes of one study, then publish its CPU vector once.

    Call only at a study admission boundary; a stop request closes this task.
    Errors propagate to the CPU scheduler so failed studies remain retryable.
    """
    from cellweft.data.preparation.delivery import _inside, load_delivery

    if node_id not in NODE_CONFIG:
        raise ValueError("Only node 72 or 73 may run downstream work")
    from types import SimpleNamespace

    # Preflight only small envelopes before any study table can be written.
    ordered = []
    study = None
    for root in packet_roots:
        ready = json.loads((root / "ready.json").read_text())
        if not isinstance(ready, dict):
            raise ValueError("Delivery ready must be a JSON object")
        name = ready.get("study")
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in {".", ".."}):
            raise ValueError("Delivery study must be one directory name")
        if not isinstance(ready.get("node_id"), str) or ready["node_id"] != node_id:
            raise ValueError("Delivery belongs to a different node")
        if study is not None and name != study:
            raise ValueError("Deliveries must belong to one study on the requested node")
        study = name
        sequence = ready.get("created_at_ns", 0)
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Delivery created_at_ns must be a nonnegative integer")
        ordered.append((sequence, str(root), root))
    if not ordered:
        raise ValueError("Study has no ready deliveries")
    study_root = node_root / "processed" / study
    for _, _, root in sorted(ordered):
        # Translate malformed envelope structure at the untrusted loader boundary.
        try:
            packet = load_delivery(root)
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid delivery ready field type: {exc}") from exc
        if packet.study != study or str(packet.node_id) != node_id:
            raise ValueError("Deliveries must belong to one study on the requested node")
        study_root.mkdir(parents=True, exist_ok=True)
        _refresh_tables(packet, study_root)
        del packet
    snapshot = json.loads((study_root / "node_input_snapshot.json").read_text())
    result = "superseded"
    if snapshot["bindings"]:
        # The resource head may be a seen packet when new scopes arrive late.
        # Read its small envelope only; never reload its complete input tables.
        resource_root = Path(snapshot["resource_delivery"])
        ready = json.loads((resource_root / "ready.json").read_text())
        if not isinstance(ready, dict):
            raise ValueError("Resource delivery ready must be a JSON object")
        if ready.get("study") != study or ready.get("node_id") != node_id:
            raise ValueError("Resource delivery must belong to this study and node")
        for field in ("s1", "bank_dir"):
            if not isinstance(ready.get(field), str) or not ready[field]:
                raise ValueError(f"Resource delivery {field} must be a nonempty path string")
        representative = SimpleNamespace(root=resource_root, study=study,
                                         s1_path=_inside(resource_root, ready["s1"]),
                                         bank_dir=_inside(resource_root, ready["bank_dir"]))
        result = _publish_embeddings(representative, study_root)
    return {"study": study, "embeddings": result, "packets": len(ordered)}


def process_delivery(packet_root: Path, *, node_id: str, node_root: Path,
                     device: str, cache_root: Path | None = None,
                     should_stop: Callable[[], bool] | None = None,
                     segmenter: CellposeExecutor | None = None,
                     cpu_stage: bool = False, queue_root: Path | None = None,
                     image_ids: tuple[str, ...] | None = None,
                     allow_unknown_mask_segmentation: bool | None = None) -> dict:
    """Process a ready delivery on GPU or queue and finish it on CPU."""
    from cellweft.data.preparation.delivery import load_delivery

    if node_id not in NODE_CONFIG:
        raise ValueError("Only node 72 or 73 may run downstream work")
    device_count, precision, batch = NODE_CONFIG[node_id]
    if not cpu_stage and device not in {f"cuda:{index}" for index in range(device_count)}:
        raise ValueError(f"Node {node_id} requires cuda:0..{device_count - 1}")
    if cpu_stage and device != "cpu":
        raise ValueError("CPU stage requires device='cpu'")
    if cpu_stage and queue_root is None:
        raise ValueError("CPU stage requires an explicit queue_root")
    packet = load_delivery(packet_root)
    if str(packet.node_id) != node_id:
        raise ValueError("Delivery belongs to another node")
    precision = packet.execution_profile["precision"]
    batch = packet.execution_profile["batch_size"]
    if allow_unknown_mask_segmentation is None:
        allow_unknown_mask_segmentation = packet.allow_unknown_mask_segmentation
    if image_ids is not None and not set(image_ids).issubset(packet.image_ids):
        raise ValueError("Selected images must belong to the delivery")
    cache_location = "node_nas" if cache_root is None else "operator_declared"
    if cache_root is None:
        cache_root = node_root / "cache" / "study_node"
    study_root = node_root / "processed" / packet.study
    study_root.mkdir(parents=True, exist_ok=True)
    # The cache path is deliberately on the declared node volume unless the
    # operator explicitly supplies a local scratch mount.
    settings = load_settings(packet.settings_path, cache_root / packet.study / device.replace(":", "_"),
                             weights=packet.weights_path, batch_size=batch,
                             cpu_workers=4, precision=precision,
                             allow_unknown_mask_segmentation=bool(allow_unknown_mask_segmentation))
    current = _refresh_tables(packet, study_root)
    summary = {"study": packet.study, "node_id": node_id, "device": device,
               "image_ids": packet.image_ids, "images": {}, "embeddings": "pending",
               "cache_root": str(cache_root), "cache_location": cache_location}
    if not current:
        summary.update(images={image_id: "superseded" for image_id in packet.image_ids},
                       embeddings="superseded")
        return summary
    device_lease = (nullcontext(True) if cpu_stage else
                    _claim(node_root / "locks" / f"{device.replace(':', '_')}.lock"))
    with device_lease as device_owned:
        if not device_owned:
            raise RuntimeError(f"{node_id}/{device} is already occupied")
        if cpu_stage:
            from cellweft.data.preparation.stage_queue import enqueue_segmentation_task

            class QueueingSegmenter:
                core_size, halo, match_iou = 1024, 128, 0.5

                def execute(self, task: dict) -> None:
                    enqueue_segmentation_task(owner_root=node_root, task=task,
                                              queue_root=queue_root, owner_node=node_id)

            context = nullcontext(QueueingSegmenter())
        else:
            context = (nullcontext(segmenter) if segmenter is not None else
                       CellposeExecutor(device=device, precision=precision))
        with context as segmenter:
            for image_id in (image_ids if image_ids is not None else packet.image_ids):
                if should_stop is not None and should_stop():
                    break
                with _claim(study_root / "claims" / f"{image_id}.lock") as owned:
                    if not owned:
                        continue
                    status_path = study_root / "images" / image_id / "node_status.json"
                    try:
                        if not _current_delivery(study_root, packet, image_id):
                            summary["images"][image_id] = "superseded"
                            continue
                        stale_pending_task = False
                        metadata_path = status_path.parent / "metadata.json"
                        if cpu_stage and metadata_path.is_file():
                            metadata = json.loads(metadata_path.read_text())
                            task = metadata.get("segmentation_task")
                            if (task and "mask" not in metadata.get("completed", [])
                                    and metadata.get("source_delivery") != str(packet.root)
                                    and not Path(task["result_path"]).is_file()):
                                with resource_lock(metadata_path.parent):
                                    current_metadata = json.loads(metadata_path.read_text())
                                    if current_metadata.get("segmentation_task") == task:
                                        write_json(metadata_path, {**current_metadata,
                                                                   "segmentation_task": None,
                                                                   "publication_incomplete": True})
                                        stale_pending_task = True
                        previous = (None if stale_pending_task else
                                    _terminal_status(
                                        status_path, packet.requests[image_id],
                                        allow_unknown_mask_segmentation=settings.allow_unknown_mask_segmentation))
                        if previous is not None:
                            metadata_path = Path(previous["metadata"])
                            metadata = json.loads(metadata_path.read_text())
                            if (metadata.get("source_binding") != packet.source_bindings
                                    or metadata.get("source_delivery") != str(packet.root)):
                                write_json(metadata_path, {**metadata, "source_binding": packet.source_bindings,
                                                           "source_delivery": str(packet.root)})
                            if not previous.get("reused"):
                                write_json(status_path, {**previous, "reused": True})
                            summary["images"][image_id] = previous["status"]
                            continue
                        write_json(status_path, {"status": "running", "node_id": node_id,
                                                 "device": device, "image_id": image_id,
                                                 "delivery": str(packet.root)})
                        _install_input_links(packet, image_id, node_root)
                        record, source, panel = packet.inputs.source(image_id, packet.mask_bindings[image_id])
                        record = replace(record, output_path=study_root,
                                         marker_list_path=study_root / "panel_mapping.csv")
                        image_settings = replace(settings, sampling_reviewed=packet.requests[image_id]["sampling"]["reviewed"])
                        row = packet.inputs.images.set_index("image_id").loc[image_id]
                        estimate = image_work_bytes(int(row.height_px), int(row.width_px), int(row.channel_count),
                            mask_unknown=source.mask_status == "unknown", normalized=False,
                            reuse_shared=image_id in packet.reuse_metadata, tile_size=settings.tile_size)
                        with shared_disk_reservation(node_root / "locks", node_root, estimate,
                                                     owner=f"{packet.study}/{image_id}"), open_source(source) as reader:
                            actual = preparation_request(source, panel, image_settings,
                                                         reader.shape[-2:], segmenter)
                            frozen = checked_canonical_request(actual, packet.requests[image_id])
                            _import_closed(packet, image_id, study_root, frozen)
                            result = prepare_image(record, source, reader, panel, image_settings,
                                                   segmenter, canonical_request=frozen)
                        metadata = json.loads(result.metadata_path.read_text())
                        metadata["source_delivery"] = str(packet.root)
                        metadata["source_binding"] = packet.source_bindings
                        write_json(result.metadata_path, metadata)
                        complete_products = {"image", "channels", "mask", "cells", "expression", "neighbors"}
                        state = "computed" if complete_products.issubset(metadata["completed"]) else "partial"
                        if state == "computed" and metadata["sampling"]["status"] == "pending":
                            state = "quality_pending"
                        image_status = {"status": state, "node_id": node_id,
                                        "device": device, "image_id": image_id,
                                        "metadata": str(result.metadata_path),
                                        "completed": metadata["completed"],
                                        "gaps": metadata["gaps"],
                                        "sampling": metadata["sampling"]["status"],
                                        "reused": result.reused,
                                        "product_revisions": {name: file_revision(Path(value))
                                                              for name, value in metadata["products"].items()}}
                    except Exception as exc:
                        image_status = {"status": "retryable_error" if isinstance(exc, OSError) else "error",
                                        "node_id": node_id, "device": device,
                                        "image_id": image_id, "error": f"{type(exc).__name__}: {exc}"}
                    if not _current_delivery(study_root, packet, image_id):
                        image_status["status"] = "superseded"
                    write_json(status_path, image_status)
                    summary["images"][image_id] = image_status["status"]
        if cpu_stage:
            summary["embeddings"] = "delegated"
        elif should_stop is not None and should_stop():
            summary["embeddings"] = "stopped"
        else:
            try:
                summary["embeddings"] = _publish_embeddings(packet, study_root)
            except Exception as exc:
                summary["embeddings"] = f"error: {type(exc).__name__}: {exc}"
    if not cpu_stage:
        write_json(study_root / f"node_summary_{device.replace(':', '_')}.json", summary)
    return summary
