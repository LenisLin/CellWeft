"""Frozen, lossless raw-archive publication; source retirement is separate."""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.capacity import CapacityReservations
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.raw_archive import (
    RawArchiveMember, _DigestingReader, _check_identity, _check_timestamp,
    _is_sha256_digest, _member_name, _resolve_public_study_root,
    _resolve_under, _resolved_data_root, restore_raw_members,
)
from cellweft.data.raw_consolidation import file_identity

SCHEMA = "cellweft.raw_archive.v1"
_BLOCK = 4 * 1024**2
_PROTOCOL_LOCKS = {".publication.lock", ".raw_consolidation.lock"}


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json(path: Path, document: dict[str, Any]) -> None:
    """Create once; an interrupted caller can resume from the durable document."""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=True, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _inventory(members: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    return {name: {"size": item["size"], "mtime_ns": item["mtime_ns"]}
            for name, item in members.items()}


def _owned_partial(archive: Path, plan: dict[str, Any]) -> tuple[Path, Path]:
    """Commit ownership of an empty inode before writing any large payload."""
    intent_path = archive.with_name(archive.name + ".intent.json")
    if intent_path.exists():
        if intent_path.is_symlink() or not intent_path.is_file():
            raise ValueError("Raw archive intent must be a regular non-symlink file")
        intent = json.loads(intent_path.read_text())
        if intent.get("schema") != SCHEMA or intent.get("plan") != plan:
            raise ValueError("Existing raw archive intent differs from frozen call")
        temporary = Path(intent["temporary"])
        if temporary.parent != archive.parent or not temporary.name.startswith(f".{archive.name}."):
            raise ValueError("Invalid intent temporary path")
        actual = file_identity(temporary)
        if any(actual[key] != intent["creation_identity"][key] for key in ("device", "inode")):
            raise ValueError("Raw archive partial ownership changed")
        # Retry also closes an earlier intent-directory fsync failure before
        # any further payload growth is allowed.
        _sync_directory(archive.parent)
        return temporary, intent_path
    descriptor, name = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".partial", dir=archive.parent)
    temporary = Path(name)
    try:
        os.fsync(descriptor)
        intent = {"schema": SCHEMA, "plan": plan, "temporary": str(temporary),
                  "creation_identity": file_identity(temporary)}
        _publish_json(intent_path, intent)
    except BaseException:
        if not intent_path.exists():
            temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return temporary, intent_path


def _retire_partial(archive: Path, document: Mapping[str, Any]) -> None:
    temporary = Path(document["temporary"])
    if temporary.exists():
        if temporary.parent != archive.parent or not temporary.name.startswith(f".{archive.name}."):
            raise ValueError("Invalid receipt temporary path")
        _check_identity(temporary, document["archive_identity"])
        temporary.unlink()
    archive.with_name(archive.name + ".intent.json").unlink(missing_ok=True)
    _sync_directory(archive.parent)


def _check_tree(root: Path, members: Mapping[str, Mapping[str, int]],
                excluded: Mapping[str, Mapping[str, Any]]) -> None:
    actual = set()
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        for name in subdirectories:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Raw archive directory is a symlink: {path}")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative in _PROTOCOL_LOCKS:
                if file_identity(path)["size"] != 0:
                    raise ValueError(f"Nonempty protocol lock: {path}")
                continue
            actual.add(relative)
    if actual != members.keys() | excluded.keys():
        raise ValueError("Raw archive tree differs from complete frozen member/exclusion inventory")
    for name, identity in members.items():
        _check_identity(root / name, identity)
    for name, entry in excluded.items():
        _check_identity(root / name, entry["identity"])


def _verify(archive: Path, root: Path, members: Mapping[str, Mapping[str, int]]) -> str:
    """Compare every encoded source byte, without decoding scientific formats."""
    seen = set()
    with archive.open("rb") as compressed:
        encoded_source = _DigestingReader(compressed)
        with gzip.GzipFile(fileobj=encoded_source, mode="rb") as encoded:
            with tarfile.open(fileobj=encoded, mode="r|") as source:
                for entry in source:
                    name = _member_name(entry.name)
                    if name in seen or name not in members or not entry.isfile() or entry.issparse():
                        raise ValueError(f"Unexpected raw archive member: {name}")
                    seen.add(name)
                    expected = members[name]
                    if (entry.size != expected["size"]
                            or entry.pax_headers.get("CELLWEFT.mtime_ns") != str(expected["mtime_ns"])):
                        raise ValueError(f"Raw archive member version differs: {name}")
                    _check_timestamp(entry, expected["mtime_ns"])
                    _check_identity(root / name, expected)
                    with source.extractfile(entry) as restored, (root / name).open("rb") as original:
                        while True:
                            block = original.read(_BLOCK)
                            if restored.read(len(block) if block else 1) != block:
                                raise ValueError(f"Raw archive byte readback differs: {name}")
                            if not block:
                                break
                    _check_identity(root / name, expected)
            # Complete gzip validation, including its trailer and any trailing bytes.
            while encoded.read(_BLOCK):
                pass
        encoded_source.drain()
    if seen != members.keys():
        raise ValueError("Raw archive is missing frozen members")
    return encoded_source.hexdigest()


def _encoded_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def publish_raw_archive(
    study_root: Path, archive: Path, *, members: Mapping[str, Mapping[str, int]],
    excluded: Mapping[str, Mapping[str, Any]], data_root: Path,
    reserve_bytes: int, archive_budget_bytes: int,
) -> dict[str, Any]:
    """Publish one immutable archive and receipt from a complete frozen tree.

    ``members`` binds each study-relative name to device/inode/size/mtime_ns.
    ``excluded`` explicitly classifies files as historical_process or empty_lock,
    with frozen identity and a nonempty evidence reference. Root protocol locks
    are managed by the existing lock API and are never archive members.

    A same-directory transaction record binds any published archive before its
    receipt exists; retry the exact frozen call to recover. Existing unrelated
    archives, receipts and transactions are never replaced. No source is deleted.
    Callers must close source acquisition and consumer dependencies first.
    ``archive_budget_bytes`` is an explicit output cap; reserve is checked before
    work and on each output write, against actual filesystem free bytes.
    """
    import shutil

    root, archive = (Path(value).absolute() for value in (study_root, archive))
    resolved_data_root = _resolved_data_root(data_root)
    resolved_root = _resolve_public_study_root(root, resolved_data_root)
    if archive.is_symlink() or (archive.exists() and not archive.is_file()):
        raise ValueError("Raw archive must be a regular non-symlink file path")
    resolved_archive = _resolve_under(archive, resolved_data_root, label="Raw archive", strict=False)
    if (resolved_archive.parent != resolved_root.parent
            or resolved_archive.name != resolved_root.name + ".tar.gz"):
        raise ValueError("Archive must be the study's sibling <study>.tar.gz")
    root, archive, data_root = resolved_root, resolved_archive, resolved_data_root
    if any(type(value) is not int or value < 0 for value in (reserve_bytes, archive_budget_bytes)):
        raise ValueError("Explicit nonnegative archive budget and reserve are required")
    members = {name: dict(identity) for name, identity in members.items()}
    excluded = {name: dict(entry) for name, entry in excluded.items()}
    if "manifest.json" not in members or not members or members.keys() & excluded.keys():
        raise ValueError("A complete disjoint inventory including manifest.json is required")
    identities = list(members.values()) + [entry["identity"] for entry in excluded.values()]
    for identity in identities:
        if (any(type(identity.get(key)) is not int for key in ("device", "inode", "size", "mtime_ns"))
                or identity["size"] < 0):
            raise ValueError("Frozen regular-file device/inode/size/mtime_ns are required")
    for name in members.keys() | excluded.keys():
        _member_name(name)
        if name in _PROTOCOL_LOCKS:
            raise ValueError("Protocol locks must not be inventoried as source members")
    for name in members:
        if ((name not in {"manifest.json", "metadata.json"}
             and name.split("/")[0] not in {"images", "masks", "metadata"})
                or name.endswith((".log", ".lock"))):
            raise ValueError(f"Raw source layout or process-file violation: {name}")
    for name, entry in excluded.items():
        if (entry.get("reason") not in {"historical_process", "empty_lock"}
                or not isinstance(entry.get("evidence"), str) or not entry["evidence"]):
            raise ValueError(f"Explicit exclusion reason/evidence required: {name}")
        if entry["reason"] == "empty_lock" and (
                not name.endswith(".lock") or entry["identity"]["size"] != 0):
            raise ValueError(f"Excluded lock is not empty: {name}")
    plan = {"study_root": str(root.absolute()), "archive": str(archive.absolute()),
            "members": members, "excluded": excluded}
    transaction = archive.with_name(archive.name + ".transaction.json")
    receipt = archive.with_name(archive.name + ".receipt.json")
    with formal_publish_lease(archive, data_root=data_root):
        with resource_lock(root, name=".raw_consolidation.lock"):
            _check_tree(root, members, excluded)
            if receipt.exists():
                if receipt.is_symlink() or not receipt.is_file():
                    raise ValueError("Raw archive receipt must be a regular non-symlink file")
                accepted = json.loads(receipt.read_text())
                if (accepted.get("schema") != SCHEMA or accepted.get("plan") != plan
                        or accepted.get("status") != "accepted_encoded_bytes"):
                    raise ValueError("Existing raw archive receipt differs from frozen call")
                _check_identity(archive, accepted["archive_identity"])
                if accepted.get("inventory") != _inventory(members):
                    raise ValueError("Raw archive receipt inventory differs from frozen plan")
                if (not _is_sha256_digest(accepted.get("archive_sha256"))
                        or _encoded_sha256(archive) != accepted["archive_sha256"]):
                    raise ValueError("Raw archive content digest differs from accepted receipt")
                _sync_directory(archive.parent)
                _retire_partial(archive, accepted)
                return accepted
            if transaction.exists() and (transaction.is_symlink() or not transaction.is_file()):
                raise ValueError("Raw archive transaction must be a regular non-symlink file")
            known = json.loads(transaction.read_text()) if transaction.exists() else None
            if known is not None and (known.get("schema") != SCHEMA or known.get("plan") != plan):
                raise ValueError("Existing raw archive transaction differs from frozen call")
            if known is not None and not _is_sha256_digest(known.get("archive_sha256")):
                raise ValueError("Existing raw archive transaction lacks its content digest")
            if known is not None and not archive.exists():
                pending = Path(known["temporary"])
                if pending.parent != archive.parent or not pending.name.startswith(f".{archive.name}."):
                    raise ValueError("Invalid transaction temporary path")
                _check_identity(pending, known["archive_identity"])
                if _verify(pending, root, members) != known["archive_sha256"]:
                    raise ValueError("Raw archive transaction content digest differs")
                os.link(pending, archive)
                _sync_directory(archive.parent)
            if archive.exists():
                if known is None:
                    raise FileExistsError(f"Unowned raw archive already exists: {archive}")
                _check_identity(archive, known["archive_identity"])
                if _verify(archive, root, members) != known["archive_sha256"]:
                    raise ValueError("Raw archive transaction content digest differs")
                _sync_directory(archive.parent)
            else:
                temporary, _ = _owned_partial(archive, plan)
                reclaimable = file_identity(temporary)["size"]
                if shutil.disk_usage(archive.parent).free + reclaimable < archive_budget_bytes + reserve_bytes:
                    raise OSError("Raw archive budget plus reserve exceeds available space")
                # The exact owned inode can be reset after an interrupted write.
                # No payload is written before its intent is durable.
                descriptor = os.open(temporary, os.O_WRONLY | os.O_TRUNC)
                with os.fdopen(descriptor, "wb") as output:
                    class BudgetedWriter:
                        def write(self, block: bytes) -> int:
                            if (output.tell() + len(block) > archive_budget_bytes
                                    or shutil.disk_usage(archive.parent).free < len(block) + reserve_bytes):
                                raise OSError("Raw archive exceeds output budget or free-space reserve")
                            return output.write(block)
                        def flush(self) -> None:
                            output.flush()
                    with tarfile.open(fileobj=BudgetedWriter(), mode="w|gz", format=tarfile.PAX_FORMAT) as target:
                        for name in sorted(members):
                            identity = members[name]
                            _check_identity(root / name, identity)
                            item = tarfile.TarInfo(name)
                            item.size = identity["size"]
                            seconds, nanos = divmod(abs(identity["mtime_ns"]), 10**9)
                            sign = "-" if identity["mtime_ns"] < 0 else ""
                            item.pax_headers = {"mtime": f"{sign}{seconds}.{nanos:09d}",
                                                "CELLWEFT.mtime_ns": str(identity["mtime_ns"])}
                            with (root / name).open("rb") as source:
                                target.addfile(item, source)
                            _check_identity(root / name, identity)
                    output.flush()
                    os.fsync(output.fileno())
                archive_sha256 = _verify(temporary, root, members)
                _check_tree(root, members, excluded)
                known = {"schema": SCHEMA, "plan": plan,
                         "archive_identity": file_identity(temporary), "temporary": str(temporary),
                         "archive_sha256": archive_sha256}
                # Publish the archive first without replacing another inode.
                # The transaction is durable before its name becomes visible.
                _publish_json(transaction, known)
                os.link(temporary, archive)
                _sync_directory(archive.parent)
            _check_tree(root, members, excluded)
            accepted = {**known, "status": "accepted_encoded_bytes",
                        "inventory": _inventory(members)}
            _publish_json(receipt, accepted)
            # The archive name and accepted receipt are both directory-durable.
            # Only now can the transaction's recovery hardlink be removed.
            _retire_partial(archive, accepted)
            return accepted


@contextmanager
def accepted_raw_members(
    receipt_path: Path, *, members: Sequence[str], scratch_root: Path,
    data_root: Path, reserve_bytes: int,
    capacity_reservations: CapacityReservations | None = None,
) -> Iterator[dict[str, Path]]:
    """Restore from an explicitly selected accepted receipt for existing readers."""
    receipt_path = Path(receipt_path).absolute()
    resolved_data_root = _resolved_data_root(data_root)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Raw archive receipt must be an existing regular non-symlink file")
    receipt_path = _resolve_under(receipt_path, resolved_data_root, label="Raw archive receipt", strict=True)
    with formal_read_lease(receipt_path, data_root=resolved_data_root):
        receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "accepted_encoded_bytes":
        raise ValueError("An accepted raw archive receipt is required")
    plan = receipt.get("plan")
    if (not isinstance(plan, dict) or not isinstance(plan.get("study_root"), str)
            or not isinstance(plan.get("archive"), str)):
        raise ValueError("Accepted raw archive receipt requires a frozen plan")
    study_root = _resolve_public_study_root(Path(plan["study_root"]), resolved_data_root)
    archive_path = Path(plan["archive"])
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("Accepted raw archive must be an existing regular non-symlink file")
    archive = _resolve_under(archive_path, resolved_data_root, label="Raw archive", strict=True)
    if (archive.parent != study_root.parent
            or archive.name != study_root.name + ".tar.gz"):
        raise ValueError("Accepted raw archive must be the study's sibling <study>.tar.gz")
    if receipt_path != archive.with_name(archive.name + ".receipt.json"):
        raise ValueError("Raw archive receipt location does not match archive")
    if not _is_sha256_digest(receipt.get("archive_sha256")):
        raise ValueError("Accepted raw archive receipt lacks its content digest")
    if receipt.get("inventory") != _inventory(receipt["plan"]["members"]):
        raise ValueError("Raw archive receipt inventory differs from frozen plan")
    inventory = {name: RawArchiveMember(**item) for name, item in receipt["inventory"].items()}
    with restore_raw_members(archive, archive_identity=receipt["archive_identity"],
                             inventory=inventory, members=members, scratch_root=scratch_root,
                             data_root=resolved_data_root, archive_sha256=receipt["archive_sha256"],
                             reserve_bytes=reserve_bytes,
                             capacity_reservations=capacity_reservations) as restored:
        yield restored
