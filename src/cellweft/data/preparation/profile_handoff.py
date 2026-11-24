"""Commit one owner-73 FP32 packet against its existing image head."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Iterator

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.stage_queue import _claim, _read_json, _safe_name


def _profile_only_change(old: dict, new: dict) -> bool:
    frozen = deepcopy(old)
    segment = frozen["segmentation"]
    if (segment["precision"] != "bf16" or segment["parameters"]["batch_size"] != 16
            or new["segmentation"]["precision"] != "fp32"
            or new["segmentation"]["parameters"]["batch_size"] != 8):
        return False
    segment["precision"] = "fp32"
    segment["parameters"]["batch_size"] = 8
    return frozen == new


def _queued_tasks(queue_root: Path, owner_root: Path, study: str,
                  image_id: str, old_head: dict, source_bindings: dict,
                  old_request: dict, *, allow_discarded: bool = False,
                  new_head: dict | None = None, new_request: dict | None = None) -> list[str]:
    directory = queue_root / "gpu" / study
    records = sorted(directory.glob("*.json"))
    if len(records) > 5000:
        raise ValueError("GPU task history exceeds bounded handoff scan")
    queued = []
    for path in records:
        record = _read_json(path)
        task = record.get("task", {})
        if task.get("image_id") != image_id:
            continue
        if (new_head is not None and record.get("head") ==
                {**new_head, "source_bindings": source_bindings}
                and task.get("request") == new_request):
            continue
        task_id = _safe_name(task["task_id"], "task ID")
        state_path = queue_root / "states" / study / f"{task_id}.json"
        state = _read_json(state_path) if state_path.is_file() else {}
        status = state.get("status", "queued" if path.with_suffix(".ready").is_file() else "unpublished")
        retired = (allow_discarded and status == "discarded"
                   and state.get("reason") in {"source_superseded", "profile_handoff_superseded"}
                   and state.get("attempts") == 0)
        result = Path(task["result_path"]).resolve()
        if not result.is_relative_to(owner_root):
            raise ValueError("Old task result path is outside owner volume")
        if (record.get("owner_root") != str(owner_root)
                or record.get("head") != {**old_head, "source_bindings": source_bindings}
                or task.get("request") != old_request
                or task.get("precision") != old_request["segmentation"]["precision"]
                or task.get("parameters", {}).get("batch_size") !=
                   old_request["segmentation"]["parameters"]["batch_size"]
                or (status != "queued" and not retired) or type(state.get("attempts", 0)) is not int
                or state.get("attempts", 0) != 0 or result.exists()
                or (result.parent / "mask.tiff").exists()
                or any(result.parent.glob(".queue_attempt_*"))):
            raise ValueError(f"Old GPU task has started or changed: {task_id}")
        if not retired:
            queued.append(task_id)
    return queued


@contextmanager
def _held_queued_tasks(queue_root: Path, owner_root: Path, study: str,
                       image_id: str, old_head: dict, source_bindings: dict,
                       old_request: dict, *, allow_discarded: bool = False,
                       new_head: dict | None = None,
                       new_request: dict | None = None) -> Iterator[list[str]]:
    args = (queue_root, owner_root, study, image_id, old_head, source_bindings, old_request)
    options = {"allow_discarded": allow_discarded, "new_head": new_head,
               "new_request": new_request}
    queued = _queued_tasks(*args, **options)
    with ExitStack() as claims:
        for task_id in queued:
            lock = queue_root / "claims" / "tasks" / study / f"{task_id}.lock"
            if not claims.enter_context(_claim(lock)):
                raise RuntimeError(f"Old GPU task claim is busy: {task_id}")
        if _queued_tasks(*args, **options) != queued:
            raise ValueError("Old GPU queue changed while claiming tasks")
        yield queued


def _discard_queued_tasks(queue_root: Path, owner_root: Path, study: str,
                          image_id: str, old_head: dict, source_bindings: dict,
                          old_request: dict, queued: list[str], new_head: dict,
                          new_request: dict) -> None:
    pending = _queued_tasks(queue_root, owner_root, study, image_id, old_head,
                            source_bindings, old_request, allow_discarded=True,
                            new_head=new_head, new_request=new_request)
    if pending != queued:
        raise ValueError("Old GPU queue changed after dispatch")
    for task_id in queued:
        write_json(queue_root / "states" / study / f"{task_id}.json",
                   {"status": "discarded", "attempts": 0, "task_id": task_id,
                    "reason": "profile_handoff_superseded", "updated_at_ns": time.time_ns()})


def commit_profile_handoff(owner_root: Path, queue_root: Path, packet_root: Path,
                           expected: dict, *, staging_root: Path | None = None) -> dict:
    """Advance the head and dispatch while CPU/GPU cannot claim this image."""
    from cellweft.data.preparation.delivery import load_delivery
    from cellweft.data.preparation.stage_dispatch import announce_delivery
    from cellweft.data.preparation.study_node import _refresh_tables

    owner_root = Path(owner_root).resolve()
    queue_root = Path(queue_root)
    packet_root = Path(packet_root).resolve()
    staging_root = Path(staging_root).resolve() if staging_root is not None else None
    source_root = staging_root if staging_root is not None and staging_root.is_dir() else packet_root
    ready = _read_json(source_root / "ready.json")
    study = _safe_name(ready["study"], "study")
    if (packet_root.parent != owner_root / "transfer" / "study_deliveries" / study
            or (staging_root is not None and
                (staging_root.parent != owner_root / "transfer" / "profile_handoff_staging" / study
                 or staging_root.name != packet_root.name))
            or ready.get("node_id") != "73" or len(ready.get("image_ids", [])) != 1
            or ready.get("execution_profile") != {"precision": "fp32", "batch_size": 8}):
        raise ValueError("Packet is not one owner-73 FP32/batch8 delivery")
    image_id = _safe_name(ready["image_ids"][0], "image ID")
    old_head = expected["old_head"]
    old_request = expected["old_request"]
    new_request = ready["requests"][image_id]
    if not _profile_only_change(old_request, new_request):
        raise ValueError("New request changes more than precision and batch")
    if type(ready["created_at_ns"]) is not int or ready["created_at_ns"] <= old_head["created_at_ns"]:
        raise ValueError("New packet does not follow the expected old head")
    study_root = owner_root / "processed" / study
    with _claim(study_root / "claims" / f"{image_id}.lock") as owned:
        if not owned:
            raise RuntimeError("Owner image claim is busy")
        snapshot = _read_json(study_root / "node_input_snapshot.json")
        current_head = snapshot.get("image_deliveries", {}).get(image_id)
        new_head = {"delivery": str(packet_root), "created_at_ns": ready["created_at_ns"]}
        if current_head == new_head:
            packet = load_delivery(packet_root)
            if (packet.source_bindings != expected["source_bindings"]
                    or packet.source_bindings not in snapshot.get("bindings", [])):
                raise ValueError("Resumed packet has different source bindings")
            with _held_queued_tasks(queue_root, owner_root, study, image_id,
                                    old_head, expected["source_bindings"], old_request,
                                    allow_discarded=True, new_head=new_head,
                                    new_request=new_request) as queued:
                event = announce_delivery(packet_root, queue_root=queue_root, node_root=owner_root)
                _discard_queued_tasks(queue_root, owner_root, study, image_id,
                                      old_head, expected["source_bindings"], old_request,
                                      queued, new_head, new_request)
            return {"status": "resumed", "head": new_head, "dispatch": str(event),
                    "superseded_queued_task_ids": queued}
        if current_head != old_head:
            raise ValueError("Existing image head changed")
        old_ready = _read_json(Path(old_head["delivery"]) / "ready.json")
        if (old_ready.get("study") != study or old_ready.get("node_id") != "73"
                or old_ready.get("requests", {}).get(image_id) != old_request
                or old_ready.get("source_bindings") != expected["source_bindings"]
                or ready["source_bindings"] != expected["source_bindings"]):
            raise ValueError("Old request or source bindings changed")
        status_path = study_root / "images" / image_id / "node_status.json"
        status = _read_json(status_path) if status_path.is_file() else {}
        if status.get("status") not in {None, "partial"}:
            raise ValueError("Image status is not pending")
        metadata_path = status_path.parent / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        if (metadata and metadata.get("request") != old_request
                or metadata and metadata.get("image_request", metadata.get("request", {}).get("image")) != new_request["image"]
                or set(metadata.get("completed", [])) & {"mask", "cells", "expression", "neighbors"}):
            raise ValueError("Image request, normalization or products changed")
        if any((status_path.parent / name).exists() for name in
               ("mask.tiff", "cells.parquet", "expression.parquet", "neighbors.parquet")):
            raise ValueError("Published image product exists despite pending metadata")
        queued = _queued_tasks(queue_root, owner_root, study, image_id,
                               old_head, expected["source_bindings"], old_request)
        task = metadata.get("segmentation_task")
        if task and task.get("task_id") not in queued:
            raise ValueError("Metadata segmentation task is not queued")
        with _held_queued_tasks(queue_root, owner_root, study, image_id,
                                old_head, expected["source_bindings"], old_request) as held:
            packet = load_delivery(source_root)
            if packet.source_bindings != expected["source_bindings"]:
                raise ValueError("New packet source bindings changed")
            renamed = False
            try:
                if staging_root is not None:
                    if source_root != staging_root or packet_root.exists():
                        raise ValueError("Private staging is missing or formal destination exists")
                    os.rename(staging_root, packet_root)
                    renamed = True
                    packet = load_delivery(packet_root)
                if not _refresh_tables(packet, study_root):
                    raise ValueError("New packet source binding is no longer current")
                if _read_json(study_root / "node_input_snapshot.json")["image_deliveries"][image_id] != new_head:
                    raise ValueError("New packet did not become the image head")
                event = announce_delivery(packet_root, queue_root=queue_root, node_root=owner_root)
                _discard_queued_tasks(queue_root, owner_root, study, image_id,
                                      old_head, expected["source_bindings"], old_request,
                                      held, new_head, new_request)
            except BaseException:
                if renamed:
                    with resource_lock(study_root):
                        current = _read_json(study_root / "node_input_snapshot.json")
                        if current["image_deliveries"].get(image_id) == old_head:
                            os.rename(packet_root, staging_root)
                raise
        return {"status": "published", "head": new_head, "dispatch": str(event),
                "superseded_queued_task_ids": queued}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True,
                        help="Private JSON with old_head, old_request and source_bindings")
    args = parser.parse_args()
    print(json.dumps(commit_profile_handoff(args.owner_root, args.queue_root,
                                            args.packet, _read_json(args.expected))))


if __name__ == "__main__":
    main()
