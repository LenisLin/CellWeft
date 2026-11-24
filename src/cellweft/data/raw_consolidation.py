"""Exact, resumable same-filesystem consolidation of original source files.

This module only creates hard links and records provenance.  Compression and
source retirement are separate, explicit operations.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from typing import Iterator

from cellweft.data.io.publication import resource_lock


SCHEMA = "cellweft.raw_consolidation.v1"
STATIC_FIELDS = ("device", "inode", "size", "mtime_ns")
IDENTITY_FIELDS = STATIC_FIELDS + ("nlink",)


class ConsolidationError(RuntimeError):
    """A source, target, manifest, or execution boundary is unsafe."""


def file_identity(path: Path) -> dict[str, int]:
    """Return the frozen regular-file identity without reading its contents."""
    item = Path(path)
    try:
        value = item.lstat()
    except OSError as error:
        raise ConsolidationError(f"source is unavailable: {item}") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ConsolidationError(f"source is not a regular non-symlink file: {item}")
    return {"device": value.st_dev, "inode": value.st_ino, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "nlink": value.st_nlink}


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _static_equal(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == expected.get(key) for key in STATIC_FIELDS)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_known(path: Path, *, label: str) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConsolidationError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ConsolidationError(f"unknown {label} schema: {path}")
    return value


def _plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ConsolidationError("plan must use cellweft.raw_consolidation.v1")
    if plan.get("review_status") not in {"approved", "pass_for_exact_batch", "reviewed"}:
        raise ConsolidationError("plan must carry an approved review_status")
    study = plan.get("study")
    if not isinstance(study, str) or not study or "/" in study:
        raise ConsolidationError("plan.study must be one canonical study name")
    if not isinstance(plan.get("plan_id"), str) or not plan["plan_id"]:
        raise ConsolidationError("plan.plan_id is required")
    raw_root = Path(plan.get("raw_root", ""))
    target_root = Path(plan.get("target_root", raw_root / study))
    source_prefixes = [Path(item) for item in plan.get("source_prefixes", [])]
    protected = [Path(item) for item in plan.get("protected_prefixes", [])]
    if not raw_root.is_absolute() or not target_root.is_absolute() or not source_prefixes:
        raise ConsolidationError("absolute raw_root, target_root and source_prefixes are required")
    if not _under(target_root, raw_root) or not _under(target_root, raw_root / study):
        raise ConsolidationError("target_root must be inside raw_root/<study>")
    members = plan.get("members")
    if not isinstance(members, list) or not members:
        raise ConsolidationError("plan.members must be nonempty")
    normalized = dict(plan, raw_root=str(raw_root), target_root=str(target_root),
                      source_prefixes=[str(p) for p in source_prefixes],
                      protected_prefixes=[str(p) for p in protected])
    seen = set()
    for member in members:
        if not isinstance(member, dict):
            raise ConsolidationError("each member must be an object")
        source, target = Path(member.get("source", "")), Path(member.get("target", ""))
        if not source.is_absolute() or not target.is_absolute() or target in seen:
            raise ConsolidationError("members require unique absolute source/target paths")
        seen.add(target)
        if not any(_under(source, root) for root in source_prefixes):
            raise ConsolidationError(f"source outside approved prefixes: {source}")
        if any(_under(source, root) for root in protected):
            raise ConsolidationError(f"protected source in consolidation plan: {source}")
        if not _under(target, target_root) or target.name == "manifest.json":
            raise ConsolidationError(f"target outside target_root: {target}")
        frozen = member.get("identity")
        if not isinstance(frozen, dict) or any(key not in frozen for key in IDENTITY_FIELDS):
            raise ConsolidationError(f"complete source identity required: {source}")
        existing_target = member.get("existing_target_identity")
        if existing_target is not None:
            if (not isinstance(existing_target, dict)
                    or any(key not in existing_target for key in IDENTITY_FIELDS)
                    or any(existing_target[key] != frozen[key] for key in IDENTITY_FIELDS)):
                raise ConsolidationError(f"existing target registration does not match source identity: {source}")
        if not isinstance(member.get("role"), str) or not member["role"]:
            raise ConsolidationError(f"source role required: {source}")
        evidence = member.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) and x for x in evidence):
            raise ConsolidationError(f"source evidence required: {source}")
    return normalized


def _member_map(document: dict) -> dict[str, dict]:
    values = document.get("members", [])
    if not isinstance(values, list):
        raise ConsolidationError("manifest members must be a list")
    result = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("target"), str):
            raise ConsolidationError("manifest member is malformed")
        result[value["target"]] = value
    return result


def _link_mode(member: dict, entry: dict | None = None, receipt_member: dict | None = None) -> str:
    """Return the reviewed link mode; ordinary links are distinct from preexisting ones."""
    if member.get("existing_target_identity") is not None:
        return "preexisting_approved_hardlink"
    return (entry or {}).get("link_mode", (receipt_member or {}).get("link_mode", "created_hardlink"))


@contextmanager
def _study_lock(manifest: Path) -> Iterator[None]:
    with resource_lock(manifest.parent, name=".raw_consolidation.lock"):
        yield


def consolidate_batch(plan: dict, *, journal_path: Path, manifest_path: Path,
                      receipt_path: Path) -> dict:
    """Create exact hard links and a durable raw manifest, preserving sources."""
    plan = _plan(plan)
    journal_path, manifest_path, receipt_path = map(Path, (journal_path, manifest_path, receipt_path))
    if not _under(manifest_path, Path(plan["target_root"])):
        raise ConsolidationError("manifest must be inside target_root")
    with _study_lock(manifest_path):
        journal = _read_known(journal_path, label="journal")
        if journal is not None and journal.get("plan") != plan:
            raise ConsolidationError("journal plan differs; use the exact reviewed plan to resume")
        manifest = _read_known(manifest_path, label="manifest")
        receipt = _read_known(receipt_path, label="receipt")
        if receipt is not None and receipt.get("plan") != plan:
            raise ConsolidationError("receipt plan differs; use the exact reviewed plan to resume")
        if manifest is None:
            manifest = {
                "schema": SCHEMA,
                "study": plan["study"],
                "coverage": plan.get(
                    "coverage",
                    "reviewed_consolidated_members; existing raw source tree retained and not exhaustively frozen",
                ),
                "members": [],
            }
            if "source_acquisition" in plan:
                manifest["source_acquisition"] = plan["source_acquisition"]
        if manifest.get("study") != plan["study"]:
            raise ConsolidationError("manifest study differs from plan")
        if "source_acquisition" in plan:
            existing_acquisition = manifest.get("source_acquisition")
            if existing_acquisition is not None and existing_acquisition != plan["source_acquisition"]:
                raise ConsolidationError("manifest source_acquisition differs from plan")
        states = (journal or {}).get("member_states", {})
        receipt = receipt or {"schema": SCHEMA, "plan": plan, "status": "prepared", "members": {}}
        journal = journal or {"schema": SCHEMA, "plan": plan, "status": "prepared", "member_states": {}}
        _write_json(journal_path, journal)
        _write_json(receipt_path, receipt)
        existing = _member_map(manifest)
        # Freeze-check every pending source and target before creating any link.
        # This prevents a late identity mismatch from producing a partial batch.
        for member in plan["members"]:
            source, target = Path(member["source"]), Path(member["target"])
            frozen = member["identity"]
            source_now = file_identity(source)
            if not _static_equal(source_now, frozen):
                raise ConsolidationError(f"source identity changed before link: {source}")
            state = states.get(str(target), "pending")
            known = existing.get(str(target))
            receipt_member = receipt.get("members", {}).get(str(target))
            committed = (known is not None and known.get("plan_id") == plan["plan_id"]
                         and known.get("status") in {"retained", "retired"}
                         and isinstance(receipt_member, dict)
                         and receipt_member.get("status") in {"committed", "retired"})
            mode = _link_mode(member, known, receipt_member)
            if target.exists():
                target_now = file_identity(target)
                if not _static_equal(target_now, frozen):
                    raise ConsolidationError(f"target identity conflicts: {target}")
                approved_existing = member.get("existing_target_identity") is not None
                if approved_existing and target_now != member["existing_target_identity"]:
                    raise ConsolidationError(f"existing target registration does not match: {target}")
                if not approved_existing and not committed and state not in {"linking", "linked"}:
                    raise ConsolidationError(f"existing target has no matching receipt: {target}")
                expected_nlink = frozen["nlink"] if mode == "preexisting_approved_hardlink" else frozen["nlink"] + 1
                if (source_now["nlink"] != expected_nlink if mode == "preexisting_approved_hardlink"
                        else source_now["nlink"] < expected_nlink):
                    raise ConsolidationError(f"source link count changed: {source}")
            elif state in {"linked", "committed"}:
                raise ConsolidationError(f"journal says linked but target is missing: {target}")
            elif source_now != frozen:
                raise ConsolidationError(f"source identity changed before link: {source}")
            elif mode == "preexisting_approved_hardlink":
                raise ConsolidationError(f"approved existing target is missing: {target}")
        for member in plan["members"]:
            source, target = Path(member["source"]), Path(member["target"])
            frozen = member["identity"]
            source_now = file_identity(source)
            if not _static_equal(source_now, frozen):
                raise ConsolidationError(f"source identity changed before link: {source}")
            state = states.get(str(target), "pending")
            known = existing.get(str(target))
            receipt_member = receipt.get("members", {}).get(str(target))
            committed = (known is not None and known.get("plan_id") == plan["plan_id"]
                         and known.get("status") in {"retained", "retired"}
                         and isinstance(receipt_member, dict)
                         and receipt_member.get("status") in {"committed", "retired"})
            mode = _link_mode(member, known, receipt_member)
            if target.exists():
                target_now = file_identity(target)
                if not _static_equal(target_now, frozen):
                    raise ConsolidationError(f"target identity conflicts: {target}")
                approved_existing = member.get("existing_target_identity") is not None
                if approved_existing and target_now != member["existing_target_identity"]:
                    raise ConsolidationError(f"existing target registration does not match: {target}")
                if not approved_existing and not committed and state not in {"linking", "linked"}:
                    raise ConsolidationError(f"existing target has no matching receipt: {target}")
                expected_nlink = frozen["nlink"] if mode == "preexisting_approved_hardlink" else frozen["nlink"] + 1
                if (source_now["nlink"] != expected_nlink if mode == "preexisting_approved_hardlink"
                        else source_now["nlink"] < expected_nlink):
                    raise ConsolidationError(f"source link count changed: {source}")
                if not committed:
                    states[str(target)] = "approved_existing" if approved_existing else "linked"
            else:
                if state in {"linked", "committed"}:
                    raise ConsolidationError(f"journal says linked but target is missing: {target}")
                if source_now != frozen:
                    raise ConsolidationError(f"source identity changed before link: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                states[str(target)] = "linking"
                journal.update(status="running", member_states=states)
                _write_json(journal_path, journal)
                os.link(source, target, follow_symlinks=False)
                _sync_directory(target.parent)
                target_now = file_identity(target)
                if not _static_equal(target_now, frozen):
                    raise ConsolidationError(f"linked target identity changed: {target}")
                states[str(target)] = "linked"
                journal["member_states"] = states
                _write_json(journal_path, journal)
            if not committed:
                entry = {"plan_id": plan["plan_id"], "study": plan["study"],
                         "source": str(source), "target": str(target), "role": member["role"],
                         "evidence": member["evidence"], "source_identity": frozen,
                         "target_identity": file_identity(target), "link_mode": mode,
                         "status": "retained"}
                for field in ("source_url", "provenance", "source_revision"):
                    if field in member:
                        entry[field] = member[field]
                existing[str(target)] = entry
                manifest["members"] = list(existing.values())
                _write_json(manifest_path, manifest)
                receipt["members"][str(target)] = {"status": "committed", "identity": entry["target_identity"],
                                                    "link_mode": mode}
                states[str(target)] = "committed"
                journal.update(status="running", member_states=states)
                _write_json(journal_path, journal)
                _write_json(receipt_path, receipt)
        journal.update(status="completed", member_states=states)
        receipt.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
        _write_json(journal_path, journal)
        _write_json(receipt_path, receipt)
        return receipt


def retire_sources(plan: dict, *, journal_path: Path, manifest_path: Path,
                   receipt_path: Path, closure_evidence: dict) -> dict:
    """Unlink covered sources only after an exact reader/reference closure."""
    plan = _plan(plan)
    paths = sorted(str(Path(member["source"])) for member in plan["members"])
    if (closure_evidence.get("status") != "closed" or closure_evidence.get("readers_closed") is not True
            or closure_evidence.get("references_checked") is not True
            or sorted(closure_evidence.get("paths", [])) != paths
            or not closure_evidence.get("evidence")):
        raise ConsolidationError("exact reader/reference closure evidence is required")
    manifest_path, receipt_path, journal_path = map(Path, (manifest_path, receipt_path, journal_path))
    if not _under(manifest_path, Path(plan["target_root"])):
        raise ConsolidationError("manifest must be inside target_root")
    with _study_lock(manifest_path):
        journal = _read_known(journal_path, label="journal")
        manifest = _read_known(manifest_path, label="manifest")
        receipt = _read_known(receipt_path, label="receipt")
        if manifest is None or receipt is None or receipt.get("status") not in {"completed", "retiring"}:
            raise ConsolidationError("published consolidation manifest and receipt are required")
        if receipt.get("plan") != plan:
            raise ConsolidationError("receipt plan differs; use the exact reviewed plan to resume")
        if journal is not None and journal.get("plan") != plan:
            raise ConsolidationError("journal plan differs; use the exact reviewed plan to resume")
        existing = _member_map(manifest)
        states = (journal or {}).get("member_states", {})
        states.update({str(Path(m["target"])): states.get(str(Path(m["target"])), "retiring")
                       for m in plan["members"]})
        journal = journal or {"schema": SCHEMA, "plan": plan, "status": "retiring",
                              "closure_evidence": closure_evidence, "member_states": states}
        journal.update(status="retiring", closure_evidence=closure_evidence, member_states=states)
        _write_json(journal_path, journal)
        for member in plan["members"]:
            source, target = Path(member["source"]), Path(member["target"])
            entry = existing.get(str(target))
            if entry is None or entry.get("status") not in {"retained", "retired"}:
                raise ConsolidationError(f"manifest member is not published: {target}")
            receipt_member = receipt.get("members", {}).get(str(target))
            if not isinstance(receipt_member, dict) or receipt_member.get("status") not in {"committed", "retiring", "retired"}:
                raise ConsolidationError(f"receipt member is not published: {target}")
            target_now = file_identity(target)
            if not _static_equal(target_now, member["identity"]):
                raise ConsolidationError(f"target identity changed: {target}")
            mode = _link_mode(member, entry, receipt_member)
            expected_source_nlink = (member["identity"]["nlink"] if mode == "preexisting_approved_hardlink"
                                     else member["identity"]["nlink"] + 1)
            if source.exists():
                source_now = file_identity(source)
                if not _static_equal(source_now, member["identity"]):
                    raise ConsolidationError(f"source identity changed before retirement: {source}")
                if source_now["nlink"] != expected_source_nlink:
                    raise ConsolidationError(f"unexpected source link count: {source}")
            elif not (entry.get("status") == "retired" or
                      (states.get(str(target)) == "unlinking" and
                       receipt_member.get("status") == "retiring" and
                       target_now.get("nlink") == expected_source_nlink - 1)):
                raise ConsolidationError(f"source missing before retirement: {source}")
        for member in plan["members"]:
            source, target = Path(member["source"]), Path(member["target"])
            entry = existing[str(target)]
            target_key = str(target)
            if source.exists() and states.get(target_key) != "retired":
                states[target_key] = "unlinking"
                journal["member_states"] = states
                journal["status"] = "retiring"
                receipt["status"] = "retiring"
                receipt["members"][target_key]["status"] = "retiring"
                _write_json(journal_path, journal)
                _write_json(receipt_path, receipt)
                source.unlink()
                _sync_directory(source.parent)
            if states.get(target_key) in {"unlinking", "retired"}:
                entry["status"] = "retired"
            existing[str(target)] = entry
            manifest["members"] = list(existing.values())
            _write_json(manifest_path, manifest)
            states[target_key] = "retired"
            journal["member_states"] = states
            _write_json(journal_path, journal)
            receipt["members"][target_key]["status"] = "retired"
            _write_json(receipt_path, receipt)
        journal["status"] = "completed"
        receipt["status"] = "sources_retired"
        _write_json(journal_path, journal)
        _write_json(receipt_path, receipt)
        return receipt
