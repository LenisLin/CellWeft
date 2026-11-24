"""Announce closed deliveries to independent CPU stage consumers."""
from __future__ import annotations

import json
from pathlib import Path

from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.stage_queue import _immutable_json


def announce_delivery(packet: Path, *, queue_root: Path, node_root: Path) -> Path:
    """Publish a small immutable event after the existing ready proof closes."""
    packet = packet.resolve()
    version = file_revision(packet / "ready.json")
    ready = json.loads((packet / "ready.json").read_text())
    owner, study = ready["node_id"], ready["study"]
    if owner not in {"72", "73"} or not study or Path(study).name != study or study in {".", ".."}:
        raise ValueError("Invalid stage dispatch owner or study")
    if not packet.is_relative_to(node_root.resolve() / "transfer/study_deliveries"):
        raise ValueError("Stage dispatch packet does not belong to its owner root")
    event = {"owner": owner, "node_root": str(node_root.resolve()), "study": study,
             "image_ids": ready["image_ids"], "packet": str(packet), "ready_revision": version}
    path = queue_root / "dispatch" / owner / study / (packet.name + ".json")
    if path.exists():
        if json.loads(path.read_text()) != event:
            raise ValueError("Immutable dispatch event has changed")
        return path
    if file_revision(packet / "ready.json") != version:
        raise ValueError("Ready proof changed during stage dispatch")
    _immutable_json(path, event)
    return path
