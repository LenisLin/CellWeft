"""Shared filesystem leases for NAS reads, writes, and relay transfers.

The lease deliberately covers only a formal write. Source decoding, local
staging, validation and scientific computation stay outside this context.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Iterator
import uuid

from cellweft.data.io.publication import resource_lock


DEFAULT_DATA_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
LOCK_ROOT_ENV = "CELLWEFT_NAS_LOCK_ROOT"
DATA_ROOT_ENV = "CELLWEFT_DATA_ROOT"
RESERVED_READ_ENV = "CELLWEFT_RESERVED_READ_SLOTS"
HIGH_PRIORITY_TIER_ENV = "CELLWEFT_HIGH_PRIORITY_TIER"


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _data_root(data_root: Path | None) -> Path:
    return Path(data_root or os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT)


def default_io_root(data_root: Path | None = None) -> Path:
    """Return the single shared NAS I/O root for a configured data root."""
    return _data_root(data_root) / "runtime" / "io"


def io_policy_path(root: Path) -> Path:
    """Return the project policy for the canonical runtime root."""
    root = Path(root)
    if root.name == "io" and root.parent.name == "runtime":
        return Path(__file__).resolve().parents[4] / "configs" / "io_policy.json"
    return root / "io_policy.json"


def resolve_lock_root(path: Path, *, io_lock_root: Path | None = None,
                      data_root: Path | None = None) -> Path | None:
    """Resolve the active controller lock root without creating a new run.

    An explicit root or environment value is authoritative. Otherwise formal
    paths under the configured data root use the one canonical ``runtime/io``
    root. Local staging paths return ``None`` and therefore never consume a
    formal NAS slot. Historical batch directories are never discovered.
    """
    if io_lock_root is not None:
        return Path(io_lock_root)
    root = _data_root(data_root)
    if not _path_is_under(path, root):
        return None
    configured = os.environ.get(LOCK_ROOT_ENV)
    if configured:
        return Path(configured)
    return default_io_root(root)


def _slot_count(root: Path, prefix: str, default: int) -> int:
    if prefix not in {"nas_read", "nas_write"}:
        return default
    path = io_policy_path(root)
    try:
        policy = json.loads(path.read_text())
    except FileNotFoundError:
        return default
    count = policy.get(prefix, default)
    if type(count) is not int or count < 1:
        raise ValueError(f"{path}: {prefix} must be a positive integer")
    return count


def _slot_path(root: Path, prefix: str, index: int) -> Path:
    if prefix == "nas_write" and index == 0:
        return root / "nas_write.lock"
    return root / f"{prefix}_{index}.lock"


def _next_waiter(root: Path, queue: str) -> str | None:
    directory = root / ".io_waiters" / queue
    now = time.time_ns()
    candidates = []
    for path in directory.iterdir():
        parts = path.name.split("-", 2)
        if len(parts) != 3:
            continue
        tier, arrival = int(parts[0]), int(parts[1])
        with path.open("a+b") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Aging lets an old import pass a continuing stream of returns.
                effective = tier + (now - arrival) // 10_000_000_000
                candidates.append((-effective, arrival, path))
            else:
                path.unlink(missing_ok=True)
    for _, _, path in sorted(candidates):
        requests = tuple((str(prefix), int(count)) for prefix, count in
                         json.loads(path.read_text()))
        acquired = []
        try:
            for prefix, count in requests:
                lease = _try_slot(root, prefix, count)
                if lease is None:
                    break
                acquired.append(lease)
            if len(acquired) == len(requests):
                return path.name
        finally:
            for _, handle in acquired:
                handle.close()
    return None


def _try_slot(root: Path, prefix: str, count: int):
    for index in range(_slot_count(root, prefix, count)):
        handle = _slot_path(root, prefix, index).open("a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        return index, handle
    return None


@contextmanager
def _slots(root: Path, requests: tuple[tuple[str, int], ...], *,
           priority: str = "normal") -> Iterator[None]:
    """Admit one or more slots atomically; release all before waiting again."""
    if priority not in {"normal", "high"}:
        raise ValueError("priority must be 'normal' or 'high'")
    tier = 0
    if priority == "high":
        configured_tier = os.environ.get(HIGH_PRIORITY_TIER_ENV, "2")
        try:
            tier = int(configured_tier)
        except ValueError as error:
            raise ValueError(f"{HIGH_PRIORITY_TIER_ENV} must be an integer from 2 to 30") from error
        if not 2 <= tier <= 30:
            raise ValueError(f"{HIGH_PRIORITY_TIER_ENV} must be an integer from 2 to 30")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    queue = "nas" if any(prefix in {"nas_read", "nas_write"}
                         for prefix, _ in requests) else requests[-1][0]
    directory = root / ".io_waiters" / queue
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    name = f"{tier}-{time.time_ns()}-{uuid.uuid4().hex}"
    waiter = directory / name
    # A held-open, merely flushed file can still look empty on another NFS
    # client. Publish its committed contents before exposing the queue name;
    # the open descriptor keeps the same inode locked across the rename.
    temporary = directory / (uuid.uuid4().hex + '.partial')
    waiting = temporary.open('x+b')
    acquired = []
    try:
        fcntl.flock(waiting, fcntl.LOCK_EX)
        waiting.write(json.dumps(requests).encode("utf-8"))
        waiting.flush()
        os.fsync(waiting.fileno())
        with resource_lock(root, name=".io_admission.lock"):
            os.replace(temporary, waiter)
        while True:
            with resource_lock(root, name=".io_admission.lock"):
                if _next_waiter(root, queue) == name:
                    admitted = False
                    try:
                        for prefix, count in requests:
                            lease = _try_slot(root, prefix, count)
                            if lease is None:
                                break
                            acquired.append((prefix, *lease))
                        admitted = len(acquired) == len(requests)
                        if admitted:
                            waiter.unlink()
                            break
                    finally:
                        if not admitted:
                            for _, _, handle in acquired:
                                handle.close()
                            acquired.clear()
            time.sleep(0.05)
        entered = time.monotonic()
        try:
            yield
        finally:
            finished = time.monotonic()
            for _, _, handle in reversed(acquired):
                handle.close()
            event = {"time": time.time(), "pid": os.getpid(), "priority": priority,
                     "priority_tier": tier,
                     "slots": {prefix: index for prefix, index, _ in acquired},
                     "wait_seconds": round(entered - started, 6),
                     "hold_seconds": round(finished - entered, 6)}
            with resource_lock(root, name=".io_metrics.lock"):
                with (root / "io_slot_events.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
    finally:
        with resource_lock(root, name=".io_admission.lock"):
            waiter.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            waiting.close()


@contextmanager
def limited_slots(root: Path, prefix: str, count: int, *,
                  priority: str = "normal") -> Iterator[None]:
    """Lease one filesystem slot; NAS counts follow the shared hot policy."""
    with _slots(root, ((prefix, count),), priority=priority):
        yield


@contextmanager
def _reserved_read_slot(root: Path, reserved: int) -> Iterator[None]:
    """Use only unreserved read slots without joining the NAS admission queue."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        count = _slot_count(root, "nas_read", 2)
        if count <= reserved:
            with limited_slots(root, "nas_read", 2):
                yield
            return
        acquired = None
        for index in range(reserved, count):
            handle = _slot_path(root, "nas_read", index).open("a+b")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            except BaseException:
                handle.close()
                raise
            acquired = (index, handle)
            break
        if acquired is not None:
            break
        time.sleep(0.05)
    index, handle = acquired
    entered = time.monotonic()
    try:
        yield
    finally:
        finished = time.monotonic()
        handle.close()
        event = {"time": time.time(), "pid": os.getpid(), "priority": "normal",
                 "slots": {"nas_read": index}, "reserved": True,
                 "reserved_read_slots": reserved,
                 "wait_seconds": round(entered - started, 6),
                 "hold_seconds": round(finished - entered, 6)}
        with resource_lock(root, name=".io_metrics.lock"):
            with (root / "io_slot_events.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, separators=(",", ":")) + "\n")


@contextmanager
def formal_read_lease(path: Path, *, io_lock_root: Path | None = None,
                      data_root: Path | None = None,
                      priority: str = "normal") -> Iterator[None]:
    """Account for one formal NAS read while staging or verifying a source."""
    root = resolve_lock_root(path, io_lock_root=io_lock_root, data_root=data_root)
    if root is None:
        yield
        return
    lease = (_reserved_read_slot(root, 2)
             if priority == "normal" and os.environ.get(RESERVED_READ_ENV) == "2"
             else limited_slots(root, "nas_read", 2, priority=priority))
    with lease:
        yield


@contextmanager
def formal_write_lease(path: Path, *, io_lock_root: Path | None = None,
                       data_root: Path | None = None,
                       priority: str = "normal") -> Iterator[None]:
    """Serialize one formal NAS write and account for one NAS read slot.

    ``path`` is used to distinguish a formal data-root destination from local
    staging. The global write lease is acquired before any study/image lock so
    upstream input publication and downstream result publication share one
    lock order.
    """
    root = resolve_lock_root(path, io_lock_root=io_lock_root, data_root=data_root)
    if root is None:
        yield
        return
    with _slots(root, (("nas_read", 2), ("nas_write", 1)), priority=priority):
        yield


@contextmanager
def formal_publish_lease(path: Path, *, directory: Path | None = None,
                         lock_name: str = ".publication.lock",
                         io_lock_root: Path | None = None,
                         data_root: Path | None = None,
                         priority: str = "normal") -> Iterator[None]:
    """Guard one direct formal-file publication with the shared NAS lease."""
    target = Path(path)
    lock_directory = Path(directory) if directory is not None else target.parent
    with formal_write_lease(target, io_lock_root=io_lock_root, data_root=data_root,
                            priority=priority):
        with resource_lock(lock_directory, name=lock_name):
            yield
