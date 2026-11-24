"""Open or stage exact shared accepted-index inodes for storage-reception readers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import ModuleType

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_write_lease

RECEPTION_COMPLETED = Path("processed/_shared/storage_reception/completed_studies")
SHARED_INDEX = RECEPTION_COMPLETED / "accepted_indexes" / (
    "accepted_product_version_index_20261106T215516Z_incremental_22828.json")


def _canonical(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"noncanonical or symlink path: {path}")
    return path


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.lstat()
    if not os.path.isfile(path) or path.is_symlink():
        raise ValueError(f"expected a regular nonsymlink file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_revision(path: Path) -> dict:
    identity = _stamp(path)
    return {"path": str(path), "size": identity[2], "mtime_ns": identity[3]}


def _read_json(path: Path) -> tuple[dict, dict]:
    revision = _file_revision(path)
    value = json.loads(path.read_text())
    if _file_revision(path) != revision or not isinstance(value, dict):
        raise ValueError(f"JSON evidence changed or is invalid: {path}")
    return value, revision


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _base_revision(value: dict) -> dict:
    return {key: value[key] for key in ("path", "size", "mtime_ns")}


def _declared_scope_indexes(runner: ModuleType, source: Path) -> tuple[dict, list[dict]]:
    scope_root = _canonical(Path(runner.ROOT) / "studies")
    if not scope_root.is_dir():
        raise ValueError("pre-activation frozen-scope directory is missing")
    declarations = []
    for path in sorted(scope_root.glob("*.json")):
        _canonical(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"frozen scope is not a regular file: {path}")
        scope, revision = _read_json(path)
        accepted = scope.get("accepted_index")
        if not isinstance(accepted, dict) or accepted.get("path") != str(source):
            continue
        base = _base_revision(accepted)
        if (base["path"] != str(source) or type(base["size"]) is not int
                or type(base["mtime_ns"]) is not int
                or type(accepted.get("inode")) is not int):
            raise ValueError(f"invalid frozen accepted-index declaration: {path}")
        declarations.append({"scope": path, "scope_revision": revision,
                             "accepted_index": accepted})
    if not declarations:
        raise ValueError("no pre-activation frozen scope declares the requested index")
    expected = _base_revision(declarations[0]["accepted_index"])
    inode = declarations[0]["accepted_index"]["inode"]
    if any(_base_revision(item["accepted_index"]) != expected
           or item["accepted_index"]["inode"] != inode for item in declarations):
        raise ValueError("frozen scopes disagree on the exact accepted-index version")
    return {"revision": expected, "inode": inode}, declarations


def _dependency_entry(completed: Path, source: Path, destination: Path,
                      expected: dict | None = None) -> tuple[dict | None, dict | None]:
    path = completed / "dependencies.json"
    if not path.exists():
        return None, None
    record, revision = _read_json(path)
    if record.get("schema") != "cellweft.storage_reception_dependencies.v1":
        raise ValueError("invalid reception dependency schema")
    indexes = record.get("accepted_indexes")
    if (not isinstance(indexes, list)
            or any(not isinstance(item, dict) or not isinstance(item.get("old"), dict)
                   for item in indexes)):
        raise ValueError("invalid accepted-index dependency list")
    matches = [entry for entry in indexes
               if entry.get("old", {}).get("path") == str(source)]
    if len(matches) > 1:
        raise ValueError("duplicate accepted-index dependency mapping")
    if not matches:
        return None, revision
    entry = matches[0]
    old = entry.get("old")
    new = entry.get("new")
    identity = entry.get("source_identity")
    if (not isinstance(old, dict) or not isinstance(new, dict)
            or set(old) != {"path", "size", "mtime_ns"}
            or set(new) != {"path", "size", "mtime_ns"}
            or not isinstance(identity, list) or len(identity) != 4
            or entry.get("method") != "same_inode_hardlink"
            or new.get("path") != str(destination)
            or (expected is not None and old != expected["revision"])
            or new.get("size") != old.get("size")
            or new.get("mtime_ns") != old.get("mtime_ns")):
        raise ValueError("accepted-index dependency mapping differs from the frozen declaration")
    if expected is not None and identity != [identity[0], expected["inode"],
                                              expected["revision"]["size"],
                                              expected["revision"]["mtime_ns"]]:
        raise ValueError("accepted-index dependency identity differs from the frozen declaration")
    if (old.get("size") != identity[2] or old.get("mtime_ns") != identity[3]
            or type(identity[0]) is not int or type(identity[1]) is not int):
        raise ValueError("invalid accepted-index dependency identity")
    return entry, revision


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def stage_shared_accepted_index(runner: ModuleType, source: Path | None = None) -> dict:
    """Hardlink one declared index into processed without reading its body.

    Before activation, the old frozen scopes declare the exact path, revision,
    and inode. After activation, only an already activation-bound dependency
    mapping may authorize staging or reuse; this function never edits that map.
    """
    data_root = _canonical(Path(runner.NAS))
    completed = _canonical(Path(runner.RECEPTION_ROOT))
    source = _canonical(Path(source) if source is not None else data_root / SHARED_INDEX)
    expected_lock_root = data_root / "runtime" / "io"
    if _canonical(Path(runner.LOCK_ROOT)) != expected_lock_root:
        raise ValueError("shared accepted-index staging requires canonical NAS runtime/io")
    if completed != data_root / RECEPTION_COMPLETED:
        raise ValueError("runner roots or shared index differ from the approved paths")
    destination = completed / "accepted_indexes" / source.name
    _canonical(destination)
    if source == data_root / SHARED_INDEX:
        if (destination != source or Path(runner.COMPLETED) != completed):
            raise ValueError("current accepted-index root differs from the processed binding")
        return _read_current_accepted_index(completed, source)

    legacy_value = getattr(runner, "LEGACY_COMPLETED", None)
    if legacy_value is None:
        raise ValueError("staging an old accepted index requires an explicit legacy root")
    legacy = _canonical(Path(legacy_value))
    if (Path(runner.COMPLETED) not in {legacy, completed}
            or not legacy.is_relative_to(data_root)
            or not source.is_relative_to(data_root)):
        raise ValueError("runner roots or source differ from the explicit NAS migration roots")

    with formal_write_lease(legacy, io_lock_root=expected_lock_root, data_root=data_root):
        with resource_lock(legacy):
            with resource_lock(completed):
                if not source.is_file() or source.is_symlink():
                    raise ValueError("accepted-index source must be a regular nonsymlink file")
                source_identity = _stamp(source)
                source_revision = _file_revision(source)
                activation_path = completed / "relocation_activation.json"
                fence_path = completed / "legacy_root_retirement.json"
                active = activation_path.exists() or fence_path.exists()
                dependency_revision = None
                declarations = []
                dependency_entry = None

                if active:
                    activation, activation_revision, fence_revision = _activation_record(completed)
                    dependency_entry, dependency_revision = _dependency_entry(
                        completed, source, destination)
                    if (dependency_entry is None
                            or dependency_revision != activation.get("dependencies_revision")):
                        return {"status": "held", "reason": "activated_dependency_mapping_missing",
                                "source": str(source), "target": str(destination)}
                    expected = {"revision": _base_revision(dependency_entry["old"]),
                                "inode": dependency_entry["source_identity"][1]}
                    expected_identity = dependency_entry["source_identity"]
                else:
                    expected, declarations = _declared_scope_indexes(runner, source)
                    dependency_entry, dependency_revision = _dependency_entry(
                        completed, source, destination, expected)
                    expected_identity = ([source_identity[0], expected["inode"],
                                          expected["revision"]["size"],
                                          expected["revision"]["mtime_ns"]])
                    if dependency_entry is not None:
                        expected_identity = dependency_entry["source_identity"]

                if (source_revision != expected["revision"]
                        or list(source_identity) != expected_identity
                        or source_identity[0] != os.stat(legacy).st_dev):
                    raise ValueError("accepted-index source differs from its declared exact identity")

                if destination.parent.is_symlink():
                    raise ValueError("accepted-index destination directory is a symlink")
                created_parent = not destination.parent.exists()
                destination.parent.mkdir(exist_ok=True)
                if created_parent:
                    _sync_directory(completed)
                if destination.exists() or destination.is_symlink():
                    if destination.is_symlink() or _stamp(destination) != source_identity:
                        raise ValueError("accepted-index destination conflicts with source inode")
                    status = "reused"
                else:
                    try:
                        os.link(source, destination, follow_symlinks=False)
                    except FileExistsError:
                        if destination.is_symlink() or _stamp(destination) != source_identity:
                            raise ValueError("accepted-index destination appeared with another identity")
                        status = "reused"
                    else:
                        status = "linked"

                _fsync_file(destination)
                _sync_directory(destination.parent)
                if (_file_revision(source) != source_revision
                        or _stamp(source) != source_identity
                        or _stamp(destination) != source_identity
                        or source.stat().st_dev != destination.stat().st_dev):
                    raise ValueError("accepted-index hardlink failed exact identity readback")
                for item in declarations:
                    if _file_revision(item["scope"]) != item["scope_revision"]:
                        raise ValueError("frozen accepted-index declaration changed during staging")
                if active:
                    current_activation, current_activation_revision, current_fence_revision = (
                        _activation_record(completed))
                    if (current_activation != activation
                            or current_activation_revision != activation_revision
                            or current_fence_revision != fence_revision):
                        raise ValueError("reception activation changed during index staging")
                if dependency_revision is not None:
                    if _file_revision(completed / "dependencies.json") != dependency_revision:
                        raise ValueError("reception dependency map changed during index staging")
                return {"status": status, "source": str(source), "target": str(destination),
                        "revision": _file_revision(destination),
                        "identity": list(_stamp(destination)),
                        "scope_declarations": len(declarations),
                        "dependency_mapping_present": dependency_entry is not None}


def open_shared_accepted_index(runner: ModuleType) -> Path:
    """Return the existing processed index after its reception binding is checked."""
    data_root = _canonical(Path(runner.NAS))
    source = _canonical(data_root / SHARED_INDEX)
    stage_shared_accepted_index(runner, source)
    return source


def _read_current_accepted_index(completed: Path, source: Path) -> dict:
    with resource_lock(completed, shared=True):
        activation, activation_revision, fence_revision = _activation_record(completed)
        dependency_path = completed / "dependencies.json"
        dependencies, dependency_revision = _read_json(dependency_path)
        indexes = dependencies.get("accepted_indexes")
        if not isinstance(indexes, list):
            raise ValueError("invalid accepted-index dependency list")
        matches = [entry for entry in indexes
                   if isinstance(entry, dict)
                   and isinstance(entry.get("new"), dict)
                   and entry["new"].get("path") == str(source)]
        if len(matches) != 1:
            raise ValueError("processed accepted index lacks one exact reception binding")
        dependency, dependency_revision_readback = _dependency_entry(
            completed, Path(matches[0]["old"]["path"]), source)
        if (dependency is None or dependency_revision_readback != dependency_revision
                or dependency_revision != activation.get("dependencies_revision")):
            raise ValueError("processed accepted-index binding differs from activation")
        identity = _stamp(source)
        source_identity = dependency["source_identity"]
        revision = dependency["new"]
        if (list(identity[1:]) != source_identity[1:]
                or _file_revision(source) != revision):
            raise ValueError("processed accepted-index identity differs from reception")
        current_activation, current_activation_revision, current_fence_revision = (
            _activation_record(completed))
        if (current_activation != activation
                or current_activation_revision != activation_revision
                or current_fence_revision != fence_revision
                or _file_revision(dependency_path) != dependency_revision):
            raise ValueError("processed accepted-index binding changed during read")
        return {"status": "reused", "source": str(source), "target": str(source),
                "revision": revision, "identity": list(identity),
                "scope_declarations": 0, "dependency_mapping_present": True}


def _activation_record(completed: Path) -> tuple[dict, dict, dict]:
    activation_path = completed / "relocation_activation.json"
    fence_path = completed / "legacy_root_retirement.json"
    activation, revision = _read_json(activation_path)
    fence, fence_revision = _read_json(fence_path)
    legacy = activation.get("legacy_completed")
    if (not isinstance(legacy, str)
            or Path(legacy) != Path(legacy).resolve()
            or activation.get("schema") != "cellweft.storage_reception_activation.v1"
            or fence.get("legacy_completed") != legacy
            or fence.get("schema") != "cellweft.storage_reception_retired.v1"
            or fence.get("destination") != str(completed)
            or fence.get("activation") != activation):
        raise ValueError("activation differs from the immutable retirement plan")
    if (_file_revision(activation_path) != revision
            or _file_revision(fence_path) != fence_revision):
        raise ValueError("reception activation changed during read")
    return activation, revision, fence_revision
