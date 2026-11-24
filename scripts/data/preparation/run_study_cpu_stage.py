"""Queue segmentation and finish owner-local image products on a CPU host."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import uuid
from collections import OrderedDict, deque
from pathlib import Path

CPU_THREADS = int(os.environ.get("CELLWEFT_CPU_THREADS", "4"))
if CPU_THREADS <= 0:
    raise ValueError("CELLWEFT_CPU_THREADS must be positive")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(CPU_THREADS)

import torch
torch.set_num_threads(CPU_THREADS)

from cellweft.data.io.writers import write_json
from cellweft.data.preparation.stage_queue import (
    _current_head, completed_events, enqueue_segmentation_task)
from cellweft.data.preparation.pipeline import _completed_segmentation_result
from cellweft.data.preparation.study_node import (
    NODE_CONFIG, _claim, _reviewed_zero_body_result, process_delivery)




def _requires_new_segmentation(request: dict) -> bool:
    """Identify mask requests that will enqueue or consume a segmentation task."""
    mask = request.get("mask", {})
    return (mask.get("status") in {"absent", "unusable"}
            or (mask.get("status") == "unknown"
                and mask.get("unknown_status_policy") == "cpsam_v2"))


def preparation_backpressure(queue_root: Path, node_root: Path, study: str,
                             packet: Path, image_id: str,
                             terminal_gpu_markers: set[Path] | None = None) -> bool:
    """Bound pending GPU signals; never hold mask reuse or CPU finalization."""
    metadata = _metadata(node_root, study, image_id)
    if metadata.get("segmentation_task"):
        return False
    ready = json.loads((packet / "ready.json").read_text())
    if not _requires_new_segmentation(ready["requests"][image_id]):
        return False
    count, total_bytes = 0, 0
    for marker in (queue_root / "gpu").glob("*/*.ready"):
        if terminal_gpu_markers is not None and marker in terminal_gpu_markers:
            continue
        state_path = queue_root / "states" / marker.parent.name / f"{marker.stem}.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if state.get("status") in {"completed", "discarded", "failed"}:
            if terminal_gpu_markers is not None:
                terminal_gpu_markers.add(marker)
            continue
        task = json.loads(marker.with_suffix(".json").read_text())["task"]
        count += task["study"] == study
        total_bytes += Path(task["input_path"]).stat().st_size
    return count >= 16 or total_bytes >= 512 * 1024**3


class Worklist:
    def __init__(self) -> None:
        self.urgent: OrderedDict[str, deque[tuple[Path, str]]] = OrderedDict()
        self.normal: dict[str, deque[tuple[Path, str]]] = {}
        self.queued: set[tuple[Path, str]] = set()
        self.urgent_streak = 0

    def add(self, study: str, packet: Path, image_id: str, *, urgent: bool = False,
            front: bool = False) -> None:
        key = (packet, image_id)
        if key in self.queued:
            if not urgent:
                return
            normal = self.normal.get(study)
            if normal is not None and key in normal:
                normal.remove(key)
                if not normal:
                    del self.normal[study]
            elif front and key in self.urgent.get(study, ()):
                self.urgent[study].remove(key)
            else:
                return
        else:
            self.queued.add(key)
        if urgent:
            lane = self.urgent.setdefault(study, deque())
            if front:
                lane.appendleft(key)
                self.urgent.move_to_end(study, last=False)
            else:
                lane.append(key)
        else:
            self.normal.setdefault(study, deque()).append(key)

    def pop(self) -> tuple[str, Path, str] | None:
        lanes = (self.normal, self.urgent) if self.urgent_streak >= 2 else (self.urgent, self.normal)
        for lane in lanes:
            if not lane:
                continue
            study = next(iter(lane))
            packet, image_id = lane[study].popleft()
            remaining = lane.pop(study)
            if remaining:
                lane[study] = remaining
            self.queued.remove((packet, image_id))
            self.urgent_streak = self.urgent_streak + 1 if lane is self.urgent else 0
            return study, packet, image_id
        return None


def _receipt_path(node_root: Path, study: str, image_id: str, packet: Path) -> Path:
    return (node_root / "runtime" / "study_cpu_stage" / "settled" / study /
            image_id / f"{packet.name}.json")


def _packet_image_claim_path(node_root: Path, study: str, packet: Path,
                             image_id: str) -> Path:
    return node_root / "locks" / "cpu_packets" / study / packet.name / f"{image_id}.lock"


def _write_receipt(path: Path, value: dict) -> None:
    pending = path.with_name(f"{path.name}.{uuid.uuid4().hex}.pending")
    try:
        write_json(pending, value)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
        pending.with_name(pending.name + ".partial").unlink(missing_ok=True)


def _metadata(node_root: Path, study: str, image_id: str) -> dict:
    path = node_root / "processed" / study / "images" / image_id / "metadata.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _prepared_mask_update(node_root: Path, study: str, image_id: str, packet: Path) -> bool:
    metadata = _metadata(node_root, study, image_id)
    products = metadata.get("products", {})
    if (metadata.get("publication_incomplete")
            or not {"image", "channels"} <= set(metadata.get("completed", []))
            or not all(name in products and Path(products[name]).is_file()
                       for name in ("image", "channels"))):
        return False
    request = json.loads((packet / "ready.json").read_text())["requests"][image_id]
    return (_requires_new_segmentation(request)
            and metadata.get("image_request") == request["image"])


def _head(node_root: Path, study: str, image_id: str) -> Path | None:
    path = node_root / "processed" / study / "node_input_snapshot.json"
    if not path.is_file():
        return None
    snapshot = json.loads(path.read_text())
    value = snapshot.get("image_deliveries", {}).get(image_id)
    return Path(value["delivery"]) if value else None


def _enqueue_current_task(node_root: Path, study: str, image_id: str, packet: Path,
                          task: dict, queue_root: Path) -> str:
    if _current_head(node_root, task) is None:
        return "superseded"
    try:
        owner = json.loads((packet / "ready.json").read_text())["node_id"]
        enqueue_segmentation_task(owner_root=node_root, task=task, queue_root=queue_root,
                                  owner_node=owner)
    except ValueError as exc:
        if _current_head(node_root, task) is None:
            return "superseded"
        study_root = node_root / "processed" / study
        with _claim(study_root / "claims" / f"{image_id}.lock") as owned:
            if not owned:
                return "deferred"
            if _current_head(node_root, task) is None:
                return "superseded"
            status_path = study_root / "images" / image_id / "node_status.json"
            write_json(status_path, {"status": "error", "image_id": image_id,
                                     "delivery": str(packet), "task_id": task["task_id"],
                                     "error": f"{type(exc).__name__}: {exc}"})
            _record(node_root, study, image_id, packet, "error")
        return "error"
    return "queued"


def _needs_work(node_root: Path, study: str, image_id: str, packet: Path,
                *, queue_root: Path) -> bool:
    receipt = _receipt_path(node_root, study, image_id, packet)
    state = json.loads(receipt.read_text()) if receipt.is_file() else {}
    if state.get("packet") == str(packet) and state.get("state") == "failed":
        return False
    if (state.get("packet") == str(packet) and state.get("state") == "settled"
            and state.get("status") != "partial"):
        return False
    metadata = _metadata(node_root, study, image_id)
    current_request = json.loads((packet / "ready.json").read_text())["requests"][image_id]
    request_matches = metadata.get("request") == current_request
    if (metadata.get("source_delivery") == str(packet)
            and "new zero-body mask requires one whole-image whole-cell judgment"
            in metadata.get("gaps", [])):
        if _reviewed_zero_body_result(metadata, current_request):
            return _head(node_root, study, image_id) == packet
    task = metadata.get("segmentation_task")
    result_ready = bool(task and Path(task["result_path"]).is_file())
    result_pending = "segmentation result pending" in metadata.get("gaps", [])
    unknown_mask_fallback = (
        _requires_new_segmentation(current_request)
        and current_request.get("mask", {}).get("status") == "unknown"
        and current_request.get("mask", {}).get("unknown_status_policy") == "cpsam_v2"
        and "mask" not in metadata.get("completed", [])
        and not result_ready)
    # A partial receipt can be retried when its segmentation result arrived after
    # the first CPU pass. Keep quality-only partial receipts terminal.
    if state.get("packet") == str(packet) and state.get("state") == "settled":
        if not request_matches:
            return True
        completed_result = (
            _completed_segmentation_result(task)
            if task and result_ready else None)
        retry_completed_segmentation = (
            state.get("status") == "partial"
            and metadata.get("source_delivery") == str(packet)
            and _head(node_root, study, image_id) == packet
            and task is not None
            and task.get("study") == study
            and task.get("image_id") == image_id
            and task.get("request") == current_request
            and result_pending
            and completed_result is not None
            and completed_result.get("whole_cell_confirmed") is True)
        if not retry_completed_segmentation:
            return unknown_mask_fallback
    if (metadata.get("source_delivery") == str(packet) and metadata.get("products")
            and request_matches and not unknown_mask_fallback
            and not metadata.get("publication_incomplete") and not result_pending
            and (not task or "mask" in metadata.get("completed", []) or result_ready)):
        _write_receipt(receipt, {"packet": str(packet), "state": "settled",
                                 "status": "computed" if "mask" in metadata.get("completed", []) else "partial",
                                 "task_id": None})
        return False
    if (state.get("packet") == str(packet) and state.get("state") == "awaiting_segmentation"
            and task and state.get("task_id") == task["task_id"] and not result_ready):
        if _head(node_root, study, image_id) != packet:
            return True
        outcome = _enqueue_current_task(node_root, study, image_id, packet, task, queue_root)
        if outcome != "queued":
            return outcome != "error"
        return False
    if (metadata.get("source_delivery") == str(packet) and task
            and "mask" not in metadata.get("completed", []) and not result_ready):
        if _head(node_root, study, image_id) != packet:
            return True
        outcome = _enqueue_current_task(node_root, study, image_id, packet, task, queue_root)
        if outcome != "queued":
            return outcome != "error"
        _write_receipt(receipt, {"packet": str(packet), "state": "awaiting_segmentation",
                                 "task_id": task["task_id"]})
        return False
    return True


def _bootstrap(node_root: Path, queue_root: Path, work: Worklist) -> None:
    processed = node_root / "processed"
    if not processed.is_dir():
        return
    for study_root in processed.iterdir():
        snapshot_path = study_root / "node_input_snapshot.json"
        if not snapshot_path.is_file():
            continue
        snapshot = json.loads(snapshot_path.read_text())
        for image_id, head in snapshot.get("image_deliveries", {}).items():
            packet = Path(head["delivery"])
            if _needs_work(node_root, study_root.name, image_id, packet, queue_root=queue_root):
                metadata = _metadata(node_root, study_root.name, image_id)
                task = metadata.get("segmentation_task")
                urgent = bool(task and Path(task["result_path"]).is_file())
                urgent = urgent or _prepared_mask_update(
                    node_root, study_root.name, image_id, packet)
                work.add(study_root.name, packet, image_id, urgent=urgent)


def _events(queue_root: Path, owner: str, node_root: Path, work: Worklist,
            seen_dispatch: set[Path], seen_completion: set[tuple[str, str, str]]) -> None:
    # Completion events are the wake-up path for pending segmentation.  Read them
    # before the historical dispatch tree so an old dispatch scan cannot delay a
    # result that is already ready for CPU finalization.
    for event in completed_events(owner_root=node_root, queue_root=queue_root,
                                  seen_task_ids={task_id for _, _, task_id in seen_completion}):
        key = (event["study"], event["image_id"], event["task_id"])
        if key in seen_completion:
            continue
        study, image_id = event["study"], event["image_id"]
        metadata = _metadata(node_root, study, image_id)
        task = metadata.get("segmentation_task")
        if (not task or task["task_id"] != event["task_id"]
                or task["result_path"] != event["result_path"]
                or task["request"] != metadata.get("request")):
            continue
        # Another NFS client can see the event before its cached result lookup refreshes.
        if ("mask" not in metadata.get("completed", [])
                and not Path(event["result_path"]).is_file()):
            continue
        packet = _head(node_root, study, image_id)
        if packet is not None and metadata.get("source_delivery") == str(packet):
            seen_completion.add(key)
            if _needs_work(node_root, study, image_id, packet, queue_root=queue_root):
                work.add(study, packet, image_id, urgent=True, front=True)

    dispatch = queue_root / "dispatch" / owner
    if dispatch.is_dir():
        for study_dir in dispatch.iterdir():
            if not study_dir.is_dir():
                continue
            for event_path in study_dir.glob("*.json"):
                if event_path in seen_dispatch:
                    continue
                event = json.loads(event_path.read_text())
                if event.get("owner") != owner or Path(event["node_root"]) != node_root:
                    raise ValueError(f"Dispatch event has a different owner root: {event_path}")
                packet = Path(event["packet"])
                for image_id in event["image_ids"]:
                    if _needs_work(node_root, event["study"], image_id, packet,
                                   queue_root=queue_root):
                        work.add(event["study"], packet, image_id,
                                 urgent=_prepared_mask_update(
                                     node_root, event["study"], image_id, packet))
                seen_dispatch.add(event_path)


def _record(node_root: Path, study: str, image_id: str, packet: Path,
            outcome: str) -> None:
    if outcome in {"error", "retryable_error"}:
        path = _receipt_path(node_root, study, image_id, packet)
        prior = json.loads(path.read_text()) if path.is_file() else {}
        attempts = prior.get("attempts", 0) + 1
        status_path = node_root / "processed" / study / "images" / image_id / "node_status.json"
        failure = json.loads(status_path.read_text())
        _write_receipt(path, {"packet": str(packet), "state": "failed" if outcome == "error" or attempts >= 3 else "retryable",
                              "status": outcome, "attempts": attempts, "error": failure["error"],
                              "task_id": None})
        return
    metadata = _metadata(node_root, study, image_id)
    task = metadata.get("segmentation_task")
    awaiting = (outcome == "partial" and task is not None
                and "mask" not in metadata.get("completed", [])
                and not Path(task["result_path"]).is_file())
    if outcome in {"computed", "quality_pending", "partial", "superseded"}:
        _write_receipt(_receipt_path(node_root, study, image_id, packet), {
            "packet": str(packet), "state": "awaiting_segmentation" if awaiting else "settled",
            "status": outcome, "task_id": task["task_id"] if awaiting else None})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", choices=NODE_CONFIG, required=True)
    parser.add_argument("--node", choices=("72", "73", "81"), required=True)
    parser.add_argument("--node-root", type=Path, required=True,
                        help="Explicit project root for the logical owner")
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="consume dispatch/completion events without scanning prior study heads")
    parser.add_argument("--allow-unknown-mask-segmentation", action="store_true",
                        help="run cpsam_v2 when the author-mask binding is unknown")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not args.worker_id or Path(args.worker_id).name != args.worker_id:
        parser.error("poll interval must be positive and worker ID must be one path component")
    node_root = args.node_root
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    work = Worklist()
    missed_claims = 0
    seen_dispatch: set[Path] = set()
    seen_completion: set[tuple[str, str, str]] = set()
    terminal_gpu_markers: set[Path] = set()
    status_path = args.queue_root / "runtime" / f"cpu_{args.node}_{args.worker_id}.json"
    identity = {"pid": os.getpid(), "start_ticks": int(Path("/proc/self/stat").read_text()
                .rsplit(")", 1)[1].split()[19]), "physical_node": args.node,
                "logical_owner": args.owner, "worker_id": args.worker_id,
                "cpu_threads": CPU_THREADS}

    def status(stage: str, study: str | None = None) -> None:
        write_json(status_path, {**identity, "stage": stage, "study": study,
                                 "queued": len(work.queued)})

    with _claim(node_root / "locks" / f"cpu_stage_{args.worker_id}.worker.lock") as owned:
        if not owned:
            raise RuntimeError("CPU stage worker identity is occupied")
        try:
            status("starting")
            if not args.no_bootstrap:
                _bootstrap(node_root, args.queue_root, work)
            while not stopped.is_set():
                _events(args.queue_root, args.owner, node_root, work,
                        seen_dispatch, seen_completion)
                item = work.pop()
                if item is None:
                    status("polling")
                    if args.once:
                        break
                    stopped.wait(args.poll_seconds)
                    continue
                study, packet, image_id = item
                status("claiming", study)
                backpressured = False
                with _claim(_packet_image_claim_path(node_root, study, packet, image_id)) as packet_owned:
                    if not packet_owned:
                        status("claim_wait", study)
                        work.add(study, packet, image_id)
                        missed_claims += 1
                        if args.once:
                            break
                        if missed_claims >= len(work.queued):
                            stopped.wait(min(args.poll_seconds, 1))
                            missed_claims = 0
                        continue
                    missed_claims = 0
                    if not _needs_work(node_root, study, image_id, packet,
                                       queue_root=args.queue_root):
                        status("checking", study)
                        continue
                    if preparation_backpressure(args.queue_root, node_root, study, packet, image_id,
                                                terminal_gpu_markers):
                        work.add(study, packet, image_id)
                        status("preparation_backpressure", study)
                        backpressured = True
                    else:
                        status("processing", study)
                        summary = process_delivery(packet, node_id=args.owner, node_root=node_root,
                                                   device="cpu", cpu_stage=True,
                                                   cache_root=node_root / "cache" / "study_cpu_stage" /
                                                   args.worker_id,
                                                   queue_root=args.queue_root, image_ids=(image_id,),
                                                   should_stop=stopped.is_set,
                                                   allow_unknown_mask_segmentation=(
                                                       True if args.allow_unknown_mask_segmentation else None))
                        outcome = summary["images"].get(image_id)
                        if outcome is None:
                            work.add(study, packet, image_id)
                        else:
                            _record(node_root, study, image_id, packet, outcome)
                            print(f"{study}/{image_id}: {outcome}", flush=True)
                            if outcome == "retryable_error" and _needs_work(
                                    node_root, study, image_id, packet, queue_root=args.queue_root):
                                work.add(study, packet, image_id)
                                stopped.wait(args.poll_seconds)
                        status("completed", study)
                if backpressured:
                    stopped.wait(min(args.poll_seconds, 1))
                    continue
                if args.once and work.queued == set():
                    break
        except Exception:
            status("failed")
            raise
        else:
            status("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
