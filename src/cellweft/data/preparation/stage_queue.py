"""Shared-NAS handoff for asynchronous, owner-bound Cellpose work."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator

from cellweft.data.io.writers import write_json
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.pipeline import _completed_segmentation_result
from cellweft.data.preparation.segmentation import CellposeExecutor


DEVICE_COUNTS = {"72": 4, "73": 2}
MAX_ATTEMPTS = 3
# A physical worker may only claim tasks whose immutable execution profile it
# can run.  Node 73 may assist the node 72 fp32/batch-8 workload.
GPU_TASK_PROFILES = {
    "72": frozenset({("fp32", 8)}),
    "73": frozenset({("bf16", 16), ("fp32", 8)}),
}


def _safe_name(value: str, label: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or Path(value).name != value or "/" in value or "\\" in value):
        raise ValueError(f"Invalid {label}")
    return value


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _immutable_json(path: Path, value: dict) -> None:
    """Publish a complete record once; retries may only restate identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        write_json(temporary, value)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_json(path) != value:
                raise ValueError(f"Immutable queue record changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _current_head(owner_root: Path, task: dict) -> dict | None:
    study = _safe_name(task["study"], "study")
    image_id = _safe_name(task["image_id"], "image_id")
    snapshot_path = owner_root / "processed" / study / "node_input_snapshot.json"
    if not snapshot_path.is_file():
        return None
    snapshot = _read_json(snapshot_path)
    head = snapshot.get("image_deliveries", {}).get(image_id)
    if not isinstance(head, dict) or not isinstance(head.get("delivery"), str):
        return None
    delivery = Path(head["delivery"])
    ready_path = delivery / "ready.json"
    if not ready_path.is_file():
        return None
    ready = _read_json(ready_path)
    if (ready.get("study") != study or ready.get("requests", {}).get(image_id) != task["request"]
            or ready.get("source_bindings") not in snapshot.get("bindings", [])):
        return None
    return {"delivery": str(delivery), "created_at_ns": head.get("created_at_ns"),
            "source_bindings": ready["source_bindings"]}


def enqueue_segmentation_task(*, owner_root: Path, task: dict,
                              queue_root: Path, owner_node: str | None = None) -> dict:
    """Freeze an already-persisted pipeline task and return without running GPU work."""
    if owner_node is not None and owner_node not in DEVICE_COUNTS:
        raise ValueError("Unknown logical owner node")
    owner_root = Path(owner_root).resolve()
    queue_root = Path(queue_root)
    for key in ("task_id", "study", "image_id"):
        _safe_name(task[key], key)
    if task["precision"] not in {"fp32", "bf16"}:
        raise ValueError("Unsupported task precision")
    if task["stitching"] != {"core_size": 1024, "halo": 128, "match_iou": 0.5}:
        raise ValueError("Unexpected Cellpose stitching contract")
    if not Path(task["result_path"]).resolve().is_relative_to(owner_root):
        raise ValueError("Segmentation result must be on the owner volume")
    head = _current_head(owner_root, task)
    if head is None:
        raise ValueError("Segmentation task does not match the current owner image head")
    record = {"schema": "cellweft.stage_queue.segmentation.v1",
              "owner_root": str(owner_root), "owner_node": owner_node,
              "head": head, "task": task,
              "submitted_at_ns": time.time_ns()}
    path = queue_root / "gpu" / task["study"] / f"{task['task_id']}.json"
    if path.is_file():
        prior = _read_json(path)
        if {key: prior.get(key) for key in ("schema", "owner_root", "owner_node", "head", "task")} != {
                key: record[key] for key in ("schema", "owner_root", "owner_node", "head", "task")}:
            raise ValueError("Task ID already belongs to a different queue record")
        record = prior
    else:
        try:
            _immutable_json(path, record)
        except ValueError:
            prior = _read_json(path)
            if {key: prior.get(key) for key in ("schema", "owner_root", "owner_node", "head", "task")} != {
                    key: record[key] for key in ("schema", "owner_root", "owner_node", "head", "task")}:
                raise
        record = _read_json(path)
    _immutable_json(path.with_suffix(".ready"), {"task_id": task["task_id"]})
    return record


def completed_events(*, owner_root: Path, queue_root: Path,
                     study: str | None = None,
                     seen_task_ids: set[str] | None = None) -> list[dict]:
    """Read only completion receipts, never historical delivery packets."""
    owner = str(Path(owner_root).resolve())
    event_root = Path(queue_root) / "events"
    studies = [event_root / _safe_name(study, "study")] if study else event_root.glob("*")
    return [event for directory in studies if directory.is_dir()
            for path in directory.glob("*.json")
            if seen_task_ids is None or path.stem not in seen_task_ids
            if (event := _read_json(path)).get("owner_root") == owner]


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


@contextmanager
def _admission(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _is_current(record: dict) -> bool:
    return _current_head(Path(record["owner_root"]), record["task"]) == record["head"]


def _candidates(queue_root: Path, *,
                completed_ids: set[tuple[str, str]] | None = None) -> list[dict]:
    records = []
    for marker in (queue_root / "gpu").glob("*/*.ready"):
        if completed_ids is not None and (marker.parent.name, marker.stem) in completed_ids:
            continue
        path = marker.with_suffix(".json")
        if not path.is_file():
            continue
        record = _read_json(path)
        task = record["task"]
        if marker.stem != task["task_id"] or marker.parent.name != task["study"]:
            raise ValueError(f"Queue marker does not match task: {marker}")
        state_path = queue_root / "states" / task["study"] / f"{task['task_id']}.json"
        state = _read_json(state_path) if state_path.is_file() else {}
        if state.get("status") == "completed" and completed_ids is not None:
            completed_ids.add((task["study"], task["task_id"]))
        if state.get("status") not in {"completed", "discarded", "failed"}:
            records.append(record)
    return records


def _active_study_count(queue_root: Path, physical_node: str, study: str) -> int:
    active = 0
    for slot in range(DEVICE_COUNTS[physical_node]):
        path = queue_root / "claims" / physical_node / study / f"{slot}.lock"
        with _claim(path) as available:
            active += not available
    return active


def choose_tasks(records: list[dict], active: dict[str, int],
                 last_served: dict[str, int] | None = None) -> list[dict]:
    """Prefer idle studies, then the longest-starved study and oldest task."""
    last_served = last_served or {}
    return sorted(records, key=lambda record: (
        active.get(record["task"]["study"], 0),
        last_served.get(record["task"]["study"], 0),
        record["submitted_at_ns"], record["task"]["task_id"]))


def _task_profile(task: dict) -> tuple[str, int] | None:
    """Return a task's exact immutable GPU profile, without defaults."""
    precision = task.get("precision")
    parameters = task.get("parameters")
    batch_size = parameters.get("batch_size") if isinstance(parameters, dict) else None
    if not isinstance(precision, str) or type(batch_size) is not int:
        return None
    return precision, batch_size


def _task_is_compatible(record: dict, physical_node: str) -> bool:
    task = record.get("task")
    return isinstance(task, dict) and _task_profile(task) in GPU_TASK_PROFILES[physical_node]


def _start_ticks() -> int:
    return int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])


class SegmentationQueueWorker:
    """One physical GPU consumer; each task retains its owner's exact request."""

    def __init__(self, *, physical_node: str, device: str,
                 queue_root: Path, physical_root: Path,
                 weights_override: Path | None = None,
                 executor_factory: Callable[..., CellposeExecutor] = CellposeExecutor):
        if physical_node not in DEVICE_COUNTS:
            raise ValueError("Unknown physical node")
        if device not in {f"cuda:{n}" for n in range(DEVICE_COUNTS[physical_node])}:
            raise ValueError("Device is outside this physical node")
        self.physical_node, self.device = physical_node, device
        self.queue_root = Path(queue_root)
        self.physical_root = Path(physical_root)
        self.weights_override = Path(weights_override) if weights_override is not None else None
        self.executor_factory = executor_factory
        self.executors: dict[str, CellposeExecutor] = {}
        self.completed_ids: set[tuple[str, str]] = set()
        self.last_served: dict[str, int] = {}
        self.identity = {"physical_executor": physical_node, "device": device,
                         "pid": os.getpid(), "start_ticks": _start_ticks()}
        self.status_path = (self.queue_root / "runtime" / physical_node /
                            f"{device.replace(':', '_')}.json")

    def _status(self, stage: str, *, record: dict | None = None,
                queue_count: int = 0, active_count: int = 0,
                eligible_studies: int = 0) -> None:
        task = record["task"] if record else {}
        usage = Path("/proc/self/status").read_text()
        rss = next((int(line.split()[1]) * 1024 for line in usage.splitlines()
                    if line.startswith("VmRSS:")), 0)
        write_json(self.status_path, {**self.identity, "stage": stage,
            "logical_owner": (record["owner_node"] or record["owner_root"]) if record else None,
            "owner_root": record["owner_root"] if record else None,
            "study": task.get("study"), "task_id": task.get("task_id"),
            "queue_count": queue_count, "active_study_claims": active_count,
            "eligible_studies": eligible_studies,
            "rss_bytes": rss, "updated_at_ns": time.time_ns()})

    def _state(self, record: dict, status: str, attempts: int, **details: object) -> None:
        task = record["task"]
        write_json(self.queue_root / "states" / task["study"] / f"{task['task_id']}.json",
                   {"status": status, "attempts": attempts, "task_id": task["task_id"],
                    "physical_executor": self.physical_node, "device": self.device,
                    "pid": os.getpid(), "start_ticks": self.identity["start_ticks"],
                    "updated_at_ns": time.time_ns(), **details})

    def _publish_event(self, record: dict) -> None:
        task = record["task"]
        event = {"study": task["study"], "image_id": task["image_id"],
                 "task_id": task["task_id"], "owner_root": record["owner_root"],
                 "result_path": task["result_path"], "physical_executor": self.physical_node,
                 "device": self.device}
        path = self.queue_root / "events" / task["study"] / f"{task['task_id']}.json"
        if path.is_file():
            prior = _read_json(path)
            if any(prior.get(key) != event[key] for key in
                   ("study", "image_id", "task_id", "owner_root", "result_path")):
                raise ValueError("Completion event belongs to a different task")
            return
        try:
            _immutable_json(path, event)
        except ValueError:
            prior = _read_json(path)
            if any(prior.get(key) != event[key] for key in
                   ("study", "image_id", "task_id", "owner_root", "result_path")):
                raise

    def _run_claimed(self, record: dict, queue_count: int, active_count: int,
                     eligible_studies: int) -> str:
        task = record["task"]
        state_path = self.queue_root / "states" / task["study"] / f"{task['task_id']}.json"
        previous = _read_json(state_path) if state_path.is_file() else {}
        if previous.get("status") in {"completed", "discarded", "failed"}:
            return "terminal"
        attempts = int(previous.get("attempts", 0))
        if not _is_current(record):
            self._state(record, "discarded", attempts, reason="source_superseded")
            return "discarded"
        result_dir = Path(task["result_path"]).parent
        attempt_dir = result_dir / f".queue_attempt_{self.physical_node}_{os.getpid()}"
        try:
            receipt = _completed_segmentation_result(task)
            if receipt is not None:
                self._publish_event(record)
                self._state(record, "completed", attempts)
                return "completed"
            attempts += 1
            attempt_dir.mkdir(parents=True, exist_ok=True)
            # Verify transported content before occupying the GPU slot.
            from cellweft.data.preparation.segmentation import verify_task_inputs
            execution_task = task
            if self.weights_override is not None:
                execution_task = {**task, "weights": str(self.weights_override)}
            verify_task_inputs(execution_task, Path(task["input_path"]),
                               Path(execution_task["weights"]))
            if not _is_current(record):
                self._state(record, "discarded", attempts, reason="source_superseded")
                return "discarded"
            with _claim(self.physical_root / "locks" / f"{self.device.replace(':', '_')}.lock") as gpu_owned:
                if not gpu_owned:
                    return "busy"
                self._state(record, "running", attempts)
                self._status("executing", record=record, queue_count=queue_count,
                             active_count=active_count, eligible_studies=eligible_studies)
                executor = self.executors.get(task["precision"])
                if executor is None:
                    executor = self.executor_factory(device=self.device, precision=task["precision"])
                    self.executors[task["precision"]] = executor
                temporary_result = executor.execute(execution_task,
                                                    result_path=attempt_dir / "result.json")
            self._status("validating_result", record=record, queue_count=queue_count,
                         active_count=active_count, eligible_studies=eligible_studies)
            receipt = _read_json(temporary_result)
            if (receipt.get("task_id") != task["task_id"] or receipt.get("request") != task["request"]
                    or receipt.get("precision") != task["precision"]
                    or receipt.get("parameters") != task["parameters"]
                    or receipt.get("stitching") != task["stitching"]
                    or receipt.get("content_identity") != task["content_identity"]
                    or receipt.get("mask") != "mask.tiff"
                    or not (attempt_dir / "mask.tiff").is_file()):
                raise ValueError("Cellpose result differs from the immutable queue task")
            owner_study = Path(record["owner_root"]) / "processed" / task["study"]
            with resource_lock(owner_study):
                if not _is_current(record):
                    self._state(record, "discarded", attempts, reason="source_superseded")
                    return "discarded"
                os.replace(attempt_dir / "mask.tiff", result_dir / "mask.tiff")
                write_json(Path(task["result_path"]), receipt)
                self._publish_event(record)
                self._state(record, "completed", attempts)
            return "completed"
        except Exception as exc:
            retryable = isinstance(exc, (OSError, RuntimeError)) and attempts < MAX_ATTEMPTS
            self._state(record, "retryable" if retryable else "failed", attempts,
                        error=f"{type(exc).__name__}: {exc}")
            return "retryable" if retryable else "failed"
        finally:
            if attempt_dir.is_dir():
                shutil.rmtree(attempt_dir)

    def _claim_candidate(self, record: dict, limit: int) -> ExitStack | None:
        if not _task_is_compatible(record, self.physical_node):
            return None
        task = record["task"]
        study = task["study"]
        for slot in range(limit):
            stack = ExitStack()
            try:
                slot_path = self.queue_root / "claims" / self.physical_node / study / f"{slot}.lock"
                task_path = self.queue_root / "claims" / "tasks" / study / f"{task['task_id']}.lock"
                image_path = (Path(record["owner_root"]) / "processed" / study / "claims" /
                              f"{task['image_id']}.lock")
                if (stack.enter_context(_claim(slot_path))
                        and stack.enter_context(_claim(task_path))
                        and stack.enter_context(_claim(image_path))):
                    return stack
            except BaseException:
                stack.close()
                raise
            stack.close()
        return None

    def run_once(self) -> str:
        records = [record for record in
                   _candidates(self.queue_root, completed_ids=self.completed_ids)
                   if _task_is_compatible(record, self.physical_node)]
        studies = {record["task"]["study"] for record in records}
        selected: tuple[dict, ExitStack] | None = None
        with _admission(self.queue_root / "claims" / self.physical_node / "admission.lock"):
            active = {study: _active_study_count(self.queue_root, self.physical_node, study)
                      for study in studies}
            preferred_limit = (DEVICE_COUNTS[self.physical_node] + 1) // 2
            for limit in (preferred_limit, DEVICE_COUNTS[self.physical_node]):
                for record in choose_tasks(records, active, self.last_served):
                    if active[record["task"]["study"]] >= limit:
                        continue
                    stack = self._claim_candidate(record, limit)
                    if stack is not None:
                        selected = record, stack
                        break
                if selected is not None:
                    break
        self._status("polling", queue_count=len(records), active_count=sum(active.values()),
                     eligible_studies=len(studies))
        if selected is None:
            return "idle"
        record, stack = selected
        with stack:
            result = self._run_claimed(record, len(records), sum(active.values()) + 1,
                                       len(studies))
        if result != "busy":
            self.last_served[record["task"]["study"]] = time.monotonic_ns()
        self._status("polling", queue_count=len(records), active_count=sum(active.values()),
                     eligible_studies=len(studies))
        return result

    def close(self) -> None:
        for executor in self.executors.values():
            executor.__exit__(None, None, None)
        self._status("stopped")
