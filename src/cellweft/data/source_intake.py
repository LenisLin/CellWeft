"""Receive exact frozen source ranges without retiring their original files."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager, nullcontext
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Callable, Iterator, Mapping

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_write_lease
from cellweft.data.raw_consolidation import SCHEMA as MANIFEST_SCHEMA, file_identity


SCHEMA = "cellweft.source_intake.v1"
RAW_PUBLIC_MANIFEST_SCHEMA = "cellweft.raw_public_manifest.v1"
STATIC = ("device", "inode", "size", "mtime_ns")


def _publish_source(pending: Path, target: Path) -> None:
    """Publish a prepared 0666 inode without replacing any existing target."""
    os.link(pending, target)


def _identity(path: Path, expected: Mapping[str, int], *, inode_only: bool = False) -> None:
    actual = file_identity(path)
    fields = (("device", "inode") if inode_only else
              tuple(key for key in (*STATIC, "nlink") if key in expected))
    if any(actual[key] != expected[key] for key in fields):
        raise ValueError(f"Source intake version conflict: {path}")


def _verify_provenance(item: Mapping[str, Any]) -> None:
    coverage = item.get("coverage")
    provenance = coverage.get("source_sidecar") if isinstance(coverage, Mapping) else None
    if provenance is not None:
        _identity(Path(provenance["path"]), provenance["identity"])


def _manifest_member_field(document: Mapping[str, Any], study: str) -> str:
    schema = document.get("schema")
    if schema == MANIFEST_SCHEMA:
        if document.get("study") != study:
            raise ValueError("Expected the current study source manifest")
        field = "members"
    elif schema == RAW_PUBLIC_MANIFEST_SCHEMA:
        if document.get("canonical_study") != study:
            raise ValueError("Expected the current study public-source manifest")
        field = "source_intakes"
    else:
        raise ValueError("Expected the current study source manifest")
    if not isinstance(document.get(field, []), list):
        raise ValueError(f"Manifest {field} must be a list")
    return field


def _existing_targets(document: Mapping[str, Any], member_field: str) -> set[str]:
    if member_field == "members":
        entries = document[member_field]
        if not isinstance(entries, list):
            raise ValueError("Manifest members must be a list")
        targets = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("target"), str):
                raise ValueError("Manifest member is malformed")
            targets.add(entry["target"])
        return targets
    targets = set()
    for intake in document.get(member_field, []):
        if not isinstance(intake, Mapping) or not isinstance(intake.get("members"), list):
            raise ValueError("Manifest source intake is malformed")
        for member in intake["members"]:
            if not isinstance(member, Mapping) or not isinstance(member.get("target"), str):
                raise ValueError("Manifest source-intake member is malformed")
            targets.add(member["target"])
    return targets


def _check_manifest_baseline(manifest_path: Path, plan: Mapping[str, Any]) -> None:
    expected = plan["manifest_identity"]
    if expected is None:
        if manifest_path.exists():
            raise ValueError(f"Source intake manifest appeared after freeze: {manifest_path}")
        return
    _identity(manifest_path, expected)
    if json.loads(manifest_path.read_text()) != plan["manifest_document"]:
        raise ValueError("Source intake manifest content conflict")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _check_mode(path: Path, expected: int) -> None:
    if path.stat().st_mode & 0o777 != expected:
        raise ValueError(f"Source intake mode conflict: {path}")


def _restore_owned_mode(pending: Path, target: Path, state: Mapping[str, Any]) -> None:
    """Restore only the open inode recorded as this batch's partial."""
    expected = state["partial_identity"]
    for path in (pending, target):
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            continue
        except OSError as error:
            if error.errno == errno.ELOOP:
                continue
            raise
        try:
            actual = os.fstat(descriptor)
            if (not stat.S_ISREG(actual.st_mode)
                    or actual.st_dev != expected["device"] or actual.st_ino != expected["inode"]):
                continue
            os.fchmod(descriptor, state["source_mode"])
            os.fsync(descriptor)
            return
        finally:
            os.close(descriptor)
    raise ValueError("No journal-owned inode available for permission recovery")


@contextmanager
def _publication_error_recovery(plan: dict[str, Any], journal_path: Path) -> Iterator[None]:
    """Close the permission window on ordinary errors before releasing locks."""
    try:
        yield
    except Exception as error:
        try:
            if journal_path.exists():
                journal = json.loads(journal_path.read_text())
                if journal.get("plan") != plan:
                    raise ValueError("Permission recovery journal differs from frozen plan")
                for item in plan["members"]:
                    state = journal["members"].get(item["target"], {})
                    if state.get("stage") == "publishing" and "source_mode" in state:
                        target = Path(item["target"])
                        pending = target.with_name(f'.{target.name}.{plan["plan_id"]}.partial')
                        _restore_owned_mode(pending, target, state)
        except Exception as recovery_error:
            error.add_note(
                f"Source intake permission recovery failed; journal retained at {journal_path}: "
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        raise


def _save_journal(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".source_intake_", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
        _sync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def _compare(source: Path, target: Path, *, prefix: bool = False) -> int:
    """Compare encoded bytes; a partial may contain only a source prefix."""
    count = 0
    with source.open("rb") as original, target.open("rb") as received:
        while block := received.read(4 * 1024**2):
            if original.read(len(block)) != block:
                raise ValueError(f"Source intake byte mismatch: {target}")
            count += len(block)
        if not prefix and original.read(1):
            raise ValueError(f"Source intake truncated target: {target}")
    return count


def _validate(plan: dict[str, Any], journal_path: Path) -> tuple[Path, Path, Path]:
    if plan.get("schema") != SCHEMA or plan.get("review_status") != "reviewed":
        raise ValueError("Source intake requires an exact reviewed plan")
    study, plan_id = plan.get("study", ""), plan.get("plan_id", "")
    if (not study or Path(study).name != study or study in {".", ".."}
            or not plan_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in plan_id)):
        raise ValueError("Source intake requires one study and a safe plan_id")
    root = Path(plan["data_root"])
    if not root.is_absolute():
        raise ValueError("data_root must be absolute")
    raw = root / "raw/public" / study
    manifest = raw / "manifest.json"
    if journal_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source intake journal belongs on SSD outside data_root")
    baseline = plan["manifest_document"]
    member_field = _manifest_member_field(baseline, study)
    if plan.get("manifest_identity") is None and member_field != "source_intakes":
        raise ValueError("Only a raw-public manifest can be initialized by source intake")
    identity = plan.get("manifest_identity")
    if identity is not None and (not isinstance(identity, Mapping)
                                 or any(type(identity.get(key)) is not int for key in STATIC)):
        raise ValueError("Expected a frozen manifest identity")
    if not plan.get("members"):
        raise ValueError("Source intake requires new manifest members")
    source_id = plan.get("source_id", plan_id)
    if (not isinstance(source_id, str) or not source_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in source_id)):
        raise ValueError("Source intake requires a safe source_id")
    existing = _existing_targets(baseline, member_field)
    if member_field == "source_intakes" and any(
            intake.get("source_id", intake.get("plan_id")) == source_id
            for intake in baseline.get(member_field, [])):
        raise ValueError(f"Source version already registered: {source_id}")
    seen: set[str] = set()
    for item in plan["members"]:
        source, target = Path(item["source"]), Path(item["target"])
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError("Source and target must be absolute")
        try:
            relative = target.relative_to(raw)
        except ValueError as error:
            raise ValueError("Source intake target must be inside the exact study") from error
        if (len(relative.parts) < 4 or relative.parts[0] not in {"images", "masks"}
                or relative.parts[2] != "incomplete" or ".." in relative.parts
                or str(target) in existing or str(target) in seen
                or not target.resolve().is_relative_to(raw.resolve())
                or source.resolve().is_relative_to(raw.resolve())):
            raise ValueError("Expected unique images/ or masks/<source>/incomplete source targets")
        seen.add(str(target))
        if not item.get("coverage") or not item.get("role"):
            raise ValueError("Every source range requires explicit role and coverage")
        for key in STATIC:
            if type(item["source_identity"][key]) is not int:
                raise ValueError("Expected a frozen complete source identity")
        coverage = item["coverage"]
        provenance = coverage.get("source_sidecar") if isinstance(coverage, Mapping) else None
        if provenance is not None:
            sidecar = Path(provenance["path"])
            if (not sidecar.is_absolute() or sidecar.resolve().is_relative_to(raw.resolve())
                    or not isinstance(provenance.get("identity"), Mapping)):
                raise ValueError("Expected an external frozen source sidecar identity")
            for key in STATIC:
                if type(provenance["identity"].get(key)) is not int:
                    raise ValueError("Expected a frozen source sidecar identity")
            if "nlink" in provenance["identity"] and type(provenance["identity"]["nlink"]) is not int:
                raise ValueError("Expected a frozen source sidecar link count")
    return root, raw, manifest


def receive_source_ranges(plan: dict[str, Any], *, journal_path: Path,
                          reserve_bytes: int,
                          allow_world_writable_publication: bool = False,
                          _write_lease_held: bool = False) -> dict[str, Any]:
    """Copy, byte-verify and publish an exact reviewed range intake batch.

    Sources remain untouched. Known partials resume only after inode and full
    prefix comparison. Unknown targets/partials are preserved and rejected.
    The manifest is replaced only from its frozen version; interrupted manifest
    publication is recognized by the journaled candidate inode and exact content.
    Capacity reserve is explicit so the caller applies current NAS policy.
    Unfinished intake requires allow_world_writable_publication=True: verified
    partials are temporarily 0666 for hardlink publication on this NAS. Locks
    coordinate cooperating writers, but other writers can retain a writable FD
    after mode restoration. Complete journal replay needs no opt-in and only
    verifies the accepted manifest and target identities, without reopening the
    original sources. Their byte equivalence was established before completion.
    It may update existing lease logs.
    Refusal of a publishing journal still restores its owned inode's source mode
    under the existing locks, with recovery failures attached to the refusal.
    The source mode is restored and fsynced before final readback and manifest
    publication. Ordinary errors restore the journal-owned inode's mode before
    releasing locks and retain the original exception. Recovery failures are
    attached explicitly to that exception. SIGKILL can still leave a journaled
    0666 partial/target until resume; unknown files and sources are never chmodded.
    """
    journal_path = Path(journal_path)
    root, raw, manifest_path = _validate(plan, journal_path)
    manifest_member_field = _manifest_member_field(plan["manifest_document"], plan["study"])
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    if type(allow_world_writable_publication) is not bool:
        raise ValueError("allow_world_writable_publication must be a boolean")
    if type(_write_lease_held) is not bool:
        raise ValueError("_write_lease_held must be a boolean")
    write_lease = nullcontext() if _write_lease_held else formal_write_lease(raw, data_root=root)
    with write_lease:
        with (
            resource_lock(raw, name=".raw_consolidation.lock"),
            _publication_error_recovery(plan, journal_path),
        ):
            journal = json.loads(journal_path.read_text()) if journal_path.exists() else None
            if journal is not None and journal.get("plan") != plan:
                raise ValueError("Source intake journal belongs to a different frozen plan")
            if journal is not None and journal["status"] == "complete":
                _identity(manifest_path, journal["manifest_candidate_identity"])
                if json.loads(manifest_path.read_text()) != journal["manifest_document"]:
                    raise ValueError("Published source manifest content conflict")
                for item in plan["members"]:
                    target = Path(item["target"])
                    state = journal["members"][str(target)]
                    if state["stage"] != "published":
                        raise ValueError(f"Completed source intake has an unpublished target: {target}")
                    _identity(target, state["target_identity"])
                    _check_mode(target, state["source_mode"])
                return journal
            for item in plan["members"]:
                _identity(Path(item["source"]), item["source_identity"])
                _verify_provenance(item)
            if journal is not None and journal["status"] == "manifest_publishing" and manifest_path.exists():
                expected = journal["manifest_candidate_identity"]
                current = file_identity(manifest_path)
                if all(current[key] == expected[key] for key in STATIC):
                    if json.loads(manifest_path.read_text()) != journal["manifest_document"]:
                        raise ValueError("Published source manifest content conflict")
                    for item in plan["members"]:
                        target = Path(item["target"])
                        state = journal["members"][str(target)]
                        _identity(target, state["target_identity"])
                        _check_mode(target, state["source_mode"])
                        _check_mode(Path(item["source"]), state["source_mode"])
                        _compare(Path(item["source"]), target)
                        _identity(Path(item["source"]), item["source_identity"])
                        _verify_provenance(item)
                    journal["status"] = "complete"
                    _save_journal(journal_path, journal)
                    candidate = manifest_path.with_name(f'.manifest.{plan["plan_id"]}.partial')
                    if candidate.exists():
                        _identity(candidate, expected, inode_only=True)
                        candidate.unlink()
                        _sync_directory(raw)
                    return journal
                raise ValueError("Published source manifest identity conflict")
            _check_manifest_baseline(manifest_path, plan)
            if not allow_world_writable_publication:
                raise PermissionError(
                    "Unfinished source intake requires allow_world_writable_publication=True: "
                    "temporary mode 0666 permits non-cooperating writers to retain writable "
                    "file descriptors, and abrupt termination may leave the inode writable"
                )
            if journal is None:
                journal = {"schema": SCHEMA, "plan": plan, "status": "receiving", "members": {}}
                _save_journal(journal_path, journal)
            _check_manifest_baseline(manifest_path, plan)
            remaining = sum(item["source_identity"]["size"] for item in plan["members"])
            for item in plan["members"]:
                state = journal["members"].get(item["target"], {})
                target = Path(item["target"])
                pending = target.with_name(f'.{target.name}.{plan["plan_id"]}.partial')
                if target.exists() and state.get("stage") in {"publishing", "published"}:
                    _identity(target, state.get("target_identity", state["partial_identity"]),
                              inode_only="target_identity" not in state)
                    remaining -= target.stat().st_size
                elif pending.exists() and state.get("partial_identity"):
                    _identity(pending, state["partial_identity"], inode_only=True)
                    remaining -= pending.stat().st_size
            manifest_bytes = len(json.dumps(plan["manifest_document"]).encode()) + len(json.dumps(plan).encode())
            if shutil.disk_usage(raw).free - remaining - manifest_bytes < reserve_bytes:
                raise OSError("Source intake would violate the explicit NAS reserve")
            for item in plan["members"]:
                source, target = Path(item["source"]), Path(item["target"])
                pending = target.with_name(f'.{target.name}.{plan["plan_id"]}.partial')
                state = journal["members"].get(str(target))
                if state is not None:
                    # The held publishing journal predates mode recording. Its
                    # unchanged source is retained, so record that mode before
                    # any permission change to its already-owned partial inode.
                    if "source_mode" not in state:
                        state["source_mode"] = source.stat().st_mode & 0o777
                        _save_journal(journal_path, journal)
                    _check_mode(source, state["source_mode"])
                if target.exists():
                    if not state or state.get("stage") not in {"publishing", "published"}:
                        raise ValueError(f"Unknown source intake target preserved: {target}")
                    _identity(target, state.get("target_identity", state["partial_identity"]),
                              inode_only="target_identity" not in state)
                    _compare(source, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if pending.exists():
                        if not state or not state.get("partial_identity"):
                            raise ValueError(f"Unknown source intake partial preserved: {pending}")
                        _identity(pending, state["partial_identity"], inode_only=True)
                        copied = _compare(source, pending, prefix=True)
                    else:
                        with pending.open("xb"):
                            pass
                        state = {"stage": "copying", "partial_identity": file_identity(pending),
                                 "source_mode": source.stat().st_mode & 0o777}
                        journal["members"][str(target)] = state
                        _save_journal(journal_path, journal)
                        copied = 0
                    if copied < item["source_identity"]["size"]:
                        with source.open("rb") as incoming, pending.open("ab") as outgoing:
                            incoming.seek(copied)
                            shutil.copyfileobj(incoming, outgoing, length=4 * 1024**2)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                    _identity(source, item["source_identity"])
                    _compare(source, pending)
                    timestamp = item["source_identity"]["mtime_ns"]
                    os.utime(pending, ns=(timestamp, timestamp))
                    state["stage"] = "publishing"
                    _save_journal(journal_path, journal)
                    pending.chmod(0o666)
                    _sync_file(pending)
                    _publish_source(pending, target)
                    _sync_directory(target.parent)
                _identity(source, item["source_identity"])
                _check_mode(source, state["source_mode"])
                if state["stage"] == "publishing":
                    # The target is the journaled partial inode, including when
                    # a prior attempt stopped immediately after os.link/chmod.
                    target.chmod(state["source_mode"])
                    _sync_file(target)
                    _sync_directory(target.parent)
                else:
                    _check_mode(target, state["source_mode"])
                _compare(source, target)
                _identity(source, item["source_identity"])
                _verify_provenance(item)
                if pending.exists():
                    _identity(pending, state["partial_identity"], inode_only=True)
                    pending.unlink()
                    _sync_directory(pending.parent)
                state.update(stage="published", target_identity=file_identity(target))
                _save_journal(journal_path, journal)
            document = deepcopy(plan["manifest_document"])
            if manifest_member_field == "members":
                for item in plan["members"]:
                    entry = {
                        "plan_id": plan["plan_id"], "source": item["source"], "target": item["target"],
                        "source_identity": item["source_identity"], "role": item["role"],
                        "coverage": item["coverage"], "status": "retained", "link_mode": "verified_copy",
                        "target_identity": journal["members"][item["target"]]["target_identity"],
                        "byte_verification": "full_encoded_source_target_comparison; source retained",
                    }
                    if "source_id" in plan:
                        entry["source_id"] = plan["source_id"]
                    if "source_batch" in plan:
                        entry["source_batch"] = plan["source_batch"]
                    document["members"].append(entry)
            else:
                source_id = plan.get("source_id", plan["plan_id"])
                source_members = []
                for item in plan["members"]:
                    source_members.append({
                        "source_id": source_id,
                        "source": item["source"],
                        "target": item["target"],
                        "bytes": item["source_identity"]["size"],
                        "source_identity": item["source_identity"],
                        "target_identity": journal["members"][item["target"]]["target_identity"],
                        "role": item["role"],
                        "coverage": item["coverage"],
                        "status": "retained",
                        "link_mode": "verified_copy",
                        "byte_verification": "full_encoded_source_target_comparison; source retained",
                    })
                document.setdefault("source_intakes", []).append({
                    "plan_id": plan["plan_id"],
                    "source_id": source_id,
                    "source_batch": plan.get("source_batch", plan["plan_id"]),
                    "role": plan.get("source_role", "independent_source_version"),
                    "acceptance": plan.get("acceptance", "reviewed source batch"),
                    "status": "received",
                    "source_retained": True,
                    "members": source_members,
                })
            candidate = manifest_path.with_name(f'.manifest.{plan["plan_id"]}.partial')
            if candidate.exists():
                if journal.get("status") != "manifest_publishing":
                    raise ValueError(f"Unknown manifest partial preserved: {candidate}")
                _identity(candidate, journal["manifest_candidate_identity"])
                if json.loads(candidate.read_text()) != document:
                    raise ValueError("Manifest partial content conflict")
            else:
                with candidate.open("x") as stream:
                    json.dump(document, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                journal.update(status="manifest_publishing", manifest_document=document,
                               manifest_candidate_identity=file_identity(candidate))
                _save_journal(journal_path, journal)
            for item in plan["members"]:
                _identity(Path(item["source"]), item["source_identity"])
                _verify_provenance(item)
            _check_manifest_baseline(manifest_path, plan)
            if plan["manifest_identity"] is None:
                os.link(candidate, manifest_path)
                candidate.unlink()
            else:
                os.replace(candidate, manifest_path)
            _sync_directory(raw)
            _identity(manifest_path, journal["manifest_candidate_identity"])
            if json.loads(manifest_path.read_text()) != document:
                raise ValueError("Source intake manifest readback failed")
            journal["status"] = "complete"
            _save_journal(journal_path, journal)
            return journal


def receive_source_ranges_from_factory(
    plan_factory: Callable[[], dict[str, Any]], *, data_root: Path, study: str,
    journal_path: Path, reserve_bytes: int,
    allow_world_writable_publication: bool = False,
) -> dict[str, Any]:
    """Build live pins and publish ranges under one formal write lease."""
    if not callable(plan_factory):
        raise TypeError("plan_factory must be callable")
    if not study or Path(study).name != study or study in {".", ".."}:
        raise ValueError("Expected one study name")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    if type(allow_world_writable_publication) is not bool:
        raise ValueError("allow_world_writable_publication must be a boolean")
    data_root = Path(data_root)
    if not data_root.is_absolute():
        raise ValueError("data_root must be absolute")
    raw = data_root / "raw/public" / study
    with formal_write_lease(raw, data_root=data_root):
        plan = plan_factory()
        if Path(plan["data_root"]) != data_root or plan["study"] != study:
            raise ValueError("Plan factory returned a different data root or study")
        return receive_source_ranges(
            plan, journal_path=journal_path, reserve_bytes=reserve_bytes,
            allow_world_writable_publication=allow_world_writable_publication,
            _write_lease_held=True,
        )
