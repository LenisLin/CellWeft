"""Versioned, bounded-memory receipt of one closed study delivery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import uuid

from cellweft.data.preparation.remote_worker import shared_disk_reservation, transfer_slots


RESERVE_BYTES = 100 * 1024**3


def _within(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"Invalid delivery member: {relative}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Delivery member escapes target: {relative}")
    return target


def _regular_version(path: Path, revision: dict) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    return (stat.S_ISREG(value.st_mode) and value.st_size == revision["size"]
            and value.st_mtime_ns == revision["mtime_ns"])


def _cache(root: Path, item: dict) -> Path | None:
    if not item["replica"].startswith(("resources/bank/", "resources/s1.csv")):
        return None
    origin = Path(item["source"]["path"])
    if not origin.is_absolute() or any(part == ".." for part in origin.parts):
        raise ValueError("Invalid cache origin")
    revision = item["revision"]
    return _within(root, str(Path("_resources") / origin.relative_to("/") /
                             f"{revision['size']}_{revision['mtime_ns']}"))


def _link(source: Path, target: Path, revision: dict) -> None:
    if _regular_version(target, revision):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        os.link(source, partial)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def _previous_files(root: Path, destination: Path, previous_delivery: Path,
                    envelope: dict) -> dict[tuple[str, int, int], Path]:
    previous = previous_delivery.resolve()
    if (not previous.is_relative_to(root) or previous == root or previous == destination
            or previous.parent != destination.parent):
        raise ValueError("Previous delivery must be a distinct packet in the same queue")
    ready_path = previous / "ready.json"
    if not ready_path.is_file() or not stat.S_ISREG(ready_path.lstat().st_mode):
        raise ValueError("Previous delivery is not ready")
    ready = json.loads(ready_path.read_text())
    if (ready.get("schema") != "cellweft.study_delivery.v1"
            or ready.get("study") != envelope.get("study")
            or ready.get("node_id") != envelope.get("node_id")
            or ready.get("image_ids") != envelope.get("image_ids")):
        raise ValueError("Previous delivery belongs to a different scope")
    files = ready["files"]
    if len({item["replica"] for item in files}) != len(files):
        raise ValueError("Previous delivery has duplicate members")
    candidates: dict[tuple[str, int, int], Path] = {}
    for item in files:
        source = item["source"]
        revision = item["revision"]
        replica = _within(previous, item["replica"])
        if source["size"] != revision["size"] or source["mtime_ns"] != revision["mtime_ns"]:
            raise ValueError("Previous delivery has an invalid source revision")
        if _regular_version(replica, revision):
            candidates[(source["path"], source["size"], source["mtime_ns"])] = replica
    return candidates


def query(root: Path, destination: Path, envelope: dict,
          reserve_bytes: int = RESERVE_BYTES,
          previous_delivery: Path | None = None) -> list[str]:
    """Check every version in one pass and stage the exact allowed stream list."""
    if reserve_bytes < 0:
        raise ValueError("Free-space reserve must be nonnegative")
    root = root.resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Delivery destination must be inside target root")
    if (destination / "ready.json").exists():
        raise ValueError("An immutable delivery is already ready")
    files = envelope["files"]
    names = [item["replica"] for item in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate delivery member")
    previous = (_previous_files(root, destination, previous_delivery, envelope)
                if previous_delivery is not None else {})
    missing: list[str] = []
    amount = 0
    for item in files:
        if item["replica"] in {"ready.json", ".ready.pending.json", ".transfer.plan.json"}:
            raise ValueError("Delivery manifest names an envelope or transfer plan")
        target = _within(destination, item["replica"])
        revision = item["revision"]
        if type(revision["size"]) is not int or revision["size"] < 0 or type(revision["mtime_ns"]) is not int:
            raise ValueError("Invalid delivery revision")
        if _regular_version(target, revision):
            continue
        cache = _cache(root, item)
        if cache is not None and _regular_version(cache, revision):
            _link(cache, target, revision)
            continue
        source = item["source"]
        old = previous.get((source["path"], source["size"], source["mtime_ns"]))
        if old is not None and source["size"] == revision["size"] and source["mtime_ns"] == revision["mtime_ns"]:
            _link(old, target, revision)
            continue
        missing.append(item["replica"])
        amount += revision["size"]
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free - amount < reserve_bytes:
        raise OSError("Delivery would violate target free-space reserve")
    destination.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    plan = {"token": token, "files": files, "missing": missing,
            "root": str(root), "destination": str(destination)}
    plan_path = destination / ".transfer.plan.json"
    temporary = destination / f".{token}.plan.partial"
    try:
        temporary.write_text(json.dumps(plan))
        os.replace(temporary, plan_path)
    finally:
        temporary.unlink(missing_ok=True)
    return [token, *missing]


def receive(destination: Path, token: str) -> None:
    """Accept one tar stream with precisely the members selected by query."""
    plan_path = destination / ".transfer.plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["token"] != token or Path(plan["destination"]) != destination.resolve():
        raise ValueError("Delivery stream does not match its query")
    if (destination / "ready.json").exists():
        raise ValueError("An immutable delivery is already ready")
    root = Path(plan["root"])
    expected = {item["replica"]: item for item in plan["files"] if item["replica"] in plan["missing"]}
    if len(expected) != len(plan["missing"]):
        raise ValueError("Invalid delivery plan")
    seen: set[str] = set()
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
        for member in archive:
            if (member.name not in expected or member.name in seen or not member.isfile()
                    or member.size != expected[member.name]["revision"]["size"]):
                raise ValueError(f"Undeclared or invalid tar member: {member.name}")
            seen.add(member.name)
            item = expected[member.name]
            revision = item["revision"]
            target = _within(destination, member.name)
            cache = _cache(root, item)
            output = cache or target
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
            try:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("Tar member has no payload")
                with partial.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                if partial.stat().st_size != revision["size"]:
                    raise ValueError("Delivery stream length differs from manifest")
                os.utime(partial, ns=(revision["mtime_ns"], revision["mtime_ns"]))
                if not _regular_version(partial, revision):
                    raise ValueError("Delivery stream revision differs from manifest")
                if (destination / "ready.json").exists():
                    raise ValueError("An immutable delivery became ready during receipt")
                os.replace(partial, output)
            finally:
                partial.unlink(missing_ok=True)
            if cache is not None:
                _link(cache, target, revision)
    if seen != set(expected):
        raise ValueError("Delivery stream omitted declared members")
    if (destination / "ready.json").exists():
        raise ValueError("An immutable delivery became ready during receipt")
    for item in plan["files"]:
        if not _regular_version(_within(destination, item["replica"]), item["revision"]):
            raise ValueError("Delivery target version changed during receipt")
    plan_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("query", "receive"))
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--token")
    parser.add_argument("--node")
    parser.add_argument("--lock-root", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=RESERVE_BYTES)
    parser.add_argument("--previous-delivery", type=Path)
    args = parser.parse_args()
    if args.action == "query":
        print(json.dumps(query(args.root, args.destination, json.load(sys.stdin),
                               args.reserve_bytes, args.previous_delivery)))
    else:
        if args.token is None or args.node is None or args.lock_root is None:
            parser.error("receive requires --token, --node and --lock-root")
        plan = json.loads((args.destination / ".transfer.plan.json").read_text())
        if Path(plan["root"]) != args.root.resolve():
            raise ValueError("Delivery root differs from its query")
        amount = sum(item["revision"]["size"] for item in plan["files"]
                     if item["replica"] in plan["missing"])
        with transfer_slots(args.lock_root, args.node):
            with shared_disk_reservation(args.lock_root, args.destination, amount):
                receive(args.destination, args.token)


if __name__ == "__main__":
    main()
