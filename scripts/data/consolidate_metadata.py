"""Consolidate an exact metadata inventory into per-owner raw metadata.json files."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

from cellweft.data.raw_consolidation import file_identity
from cellweft.data.preparation.io_coordination import formal_write_lease
from cellweft.data.io.publication import resource_lock
from cellweft.data.source_metadata import build_metadata_bundle, read_metadata_index, read_metadata_member


ACTIVE_STUDY = "Parsons2025_IMC_TNBC"
REGISTRIES = {"public/studies.csv", "public/candidates.csv", "public/handoff_datasets.tsv",
              "public/patient_eligibility_review.json"}
STATIC_FIELDS = ("device", "inode", "size", "mtime_ns")


def write_state(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def route(relative: str) -> tuple[str, str, str]:
    """Route known single-owner namespaces; preserve mixed catalog evidence once."""
    parts = Path(relative).parts
    if relative == "marker_registry.csv":
        return "shared", "_shared", "shared/marker_registry.csv"
    if parts[0] == "private":
        study = parts[1] if len(parts) > 2 and parts[1] != "case_study_v1" else "_shared"
        return "private", study, relative
    if parts[0] == "raw":
        return "public", parts[1], relative
    if parts[:2] == ("public", "source_followups"):
        group = parts[3]
        name = parts[-1].lower()
        single = {"codexcns_qualified_intake": "SanchezMolina2026_CODEXCNS_Alzheimers",
                  "chl_marker_repair": "Shaban2024_MultiPlatform_cHL",
                  "external_ali": "Ali2020_IMC_BreastCancer",
                  "external_codex": "Greenwald2024_CODEX_Glioblastoma"}
        if group in single:
            return "public", single[group], relative
        if group == "external_breast" and len(parts) > 5:
            return "public", parts[4], relative
        if group == "celltune" and name.startswith("gvhd_"):
            return "public", "Azulay2025_MIBI_GVHD", relative
        if group == "celltune" and name.startswith("melanoma_"):
            return "public", "celltune_melanoma_oct2522", relative
        if group == "external_serial" and name.startswith("ohara_"):
            return "public", "Ohara2024_IMC_UTUC", relative
        if name == "canonical_raw_routes_20261109.json":
            study = {"public_kidney_hrimc_followup_1725": "asghari2025_kidney_codex",
                     "public_zotero_archive_gaps_1650": "zotero_16088"}[group]
            return "public", study, relative
        return "public", "_catalog", relative
    if parts[0] == "public_followup_panel20" or parts[:2] in {
            ("public", "public_followup_panel20"), ("public", "exports")}:
        return "public", "_catalog", relative
    if parts[0] == "public" and len(parts) > 2:
        return "public", parts[1], relative
    raise ValueError(f"No metadata owner: {relative}")


def make_plan(root: Path) -> dict:
    metadata = root / "metadata"
    groups = {}
    protected, locks, directories = [], [], []
    with (root / "processed/_catalog/public/studies.csv").open(encoding="utf-8-sig") as stream:
        catalog = {row["dataset_name"] for row in csv.DictReader(stream)}
    for parent, names, filenames in os.walk(metadata, followlinks=False):
        names.sort()
        filenames.sort()
        directories.append(str(Path(parent)))
        for name in filenames:
            source = Path(parent) / name
            relative = source.relative_to(metadata).as_posix()
            identity = file_identity(source)
            member = {"source": str(source), "relative": relative, "identity": identity}
            if relative in REGISTRIES or relative.startswith(f"public/{ACTIVE_STUDY}/"):
                protected.append(member)
                continue
            if name.endswith(".lock"):
                if identity["size"] != 0:
                    raise ValueError(f"Nonempty runtime lock needs individual review: {source}")
                locks.append(member)
                continue
            scope, study, key = route(relative)
            target = (root / "raw/shared/metadata.json" if scope == "shared" else
                      root / "raw" / scope / study / "metadata.json")
            group = groups.setdefault(str(target), {
                "target": str(target), "owner": {"scope": scope, "study": None if study.startswith("_") else study,
                    "namespace": study, "registered_study": scope == "public" and study in catalog,
                    "raw_directory_preexisting": target.parent.exists(),
                    "source_state": "source_metadata_only; collection eligibility and scientific acceptance unchanged"},
                "members": []})
            group["members"].append({**member, "key": key})
    return {"schema": "cellweft.metadata_consolidation_execution.v1", "root": str(root),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "authorization": "User: 完成归并/移动/多余删除; one metadata.json per dataset; retain unfinished Parsons dependencies",
            "groups": list(groups.values()), "protected": protected, "locks": locks,
            "directories": directories, "source_file_count": sum(len(g["members"]) for g in groups.values()) + len(protected) + len(locks)}


def same_identity(path: Path, expected: dict) -> bool:
    actual = file_identity(path)
    return all(actual[key] == expected[key] for key in STATIC_FIELDS)


def execute(plan: dict, state_path: Path) -> dict:
    # This metadata-only inventory is smaller than one image return; admit it
    # once while preserving a durable receipt and exact retirement per owner.
    root = Path(plan["root"])
    with formal_write_lease(root / "metadata", data_root=root):
        return _execute_received_inventory(plan, state_path)


def _execute_received_inventory(plan: dict, state_path: Path) -> dict:
    root = Path(plan["root"])
    metadata = root / "metadata"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": plan["schema"], "status": "running", "completed": [], "retired_files": 0,
        "retired_logical_bytes": 0, "deduplicated_payloads": 0}
    done = {row["target"] for row in state["completed"]}
    for group in plan["groups"]:
        target = Path(group["target"])
        if str(target) in done:
            continue
        state["current"] = str(target)
        write_state(state_path, state)
        present = [member for member in group["members"] if Path(member["source"]).exists()]
        if target.exists():
            owner, index = read_metadata_index(target)
            if owner != group["owner"] or set(index) != {member["key"] for member in group["members"]}:
                raise ValueError(f"Existing metadata bundle does not match this plan: {target}")
            if len(present) != len(group["members"]):
                received = state.get("received", {})
                if received.get("target") != str(target) or not same_identity(target, received["target_identity"]):
                    raise ValueError(f"Missing source requires the matching reception receipt: {target}")
            for member in present:
                if not same_identity(Path(member["source"]), member["identity"]):
                    raise ValueError(f"Metadata source changed: {member['source']}")
                if read_metadata_member(target, member["key"], index=index) != Path(member["source"]).read_bytes():
                    raise ValueError(f"Existing metadata bundle differs from source: {member['source']}")
            result = {"members": len(index), "bundle_bytes": target.stat().st_size,
                      "source_bytes": sum(member["identity"]["size"] for member in group["members"]),
                      "deduplicated_payloads": sum("same_content_as" in row for row in index.values())}
        else:
            if len(present) != len(group["members"]):
                raise ValueError(f"Missing unreceived metadata source for {target}")
            result = build_metadata_bundle(target,
                {member["key"]: Path(member["source"]) for member in present}, owner=group["owner"],
                expected={member["key"]: member["identity"] for member in present})
        _, index = read_metadata_index(target)
        target_identity = file_identity(target)
        receipt = {"target": str(target), "target_identity": target_identity, **result,
                   "status": "received_pending_source_retirement"}
        state["received"] = receipt
        write_state(state_path, state)
        for member in present:
            source = Path(member["source"])
            if not source.resolve().is_relative_to(metadata.resolve()) or not same_identity(source, member["identity"]):
                raise ValueError(f"Source identity/boundary changed before retirement: {source}")
            source.unlink()
        receipt["status"] = "completed"
        state["completed"].append(receipt)
        state["retired_files"] = sum(row["members"] for row in state["completed"])
        state["retired_logical_bytes"] = sum(row["source_bytes"] for row in state["completed"])
        state["deduplicated_payloads"] += result["deduplicated_payloads"]
        state.pop("received", None)
        write_state(state_path, state)
        print(json.dumps({"study": group["owner"]["namespace"], **result, "status": "completed"}), flush=True)
    for member in plan["locks"]:
        source = Path(member["source"])
        if not source.exists():
            continue
        if not same_identity(source, member["identity"]):
            raise ValueError(f"Runtime lock changed: {source}")
        with source.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source.unlink()
    directories = sorted((Path(p) for p in plan["directories"]), key=lambda p: len(p.parts), reverse=True)
    removed_directories = 0
    for directory in directories:
        if directory != metadata and directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
            removed_directories += 1
    state.update(status="sources_consolidated", current=None, removed_directories=removed_directories,
                 completed_at=datetime.now(timezone.utc).isoformat())
    state["catalog_update"] = _update_catalog(plan)
    state["status"] = "complete"
    write_state(state_path, state)
    return state


def _update_catalog(plan: dict) -> dict:
    root = Path(plan["root"])
    table = root / "processed/_catalog/public/studies.csv"
    member_routes = {member["relative"]: (group["target"], member["key"])
                     for group in plan["groups"] for member in group["members"]}
    branch_routes = {"public/" + group["owner"]["namespace"]: group["target"] for group in plan["groups"]
                     if group["owner"]["scope"] == "public" and not group["owner"]["namespace"].startswith("_")}
    def reference(text: str) -> str:
        relative = text.removeprefix(str(root) + "/")
        if not relative.startswith("metadata/"):
            return text
        relative = relative.removeprefix("metadata/")
        if relative.startswith(f"public/{ACTIVE_STUDY}"):
            return text
        if relative in member_routes:
            target, key = member_routes[relative]
            return Path(target).relative_to(root).as_posix() + "#" + quote(key, safe="/")
        for prefix, target in branch_routes.items():
            if relative == prefix or relative.startswith(prefix + "/"):
                return Path(target).relative_to(root).as_posix()
        return text
    with resource_lock(table.parent):
        before = table.read_bytes()
        reader = csv.DictReader(io.StringIO(before.decode("utf-8-sig")))
        columns = reader.fieldnames
        rows = list(reader)
        original = [dict(row) for row in rows]
        changed = []
        for row in rows:
            for field in ("image_path", "marker_path", "mask_path", "meta_path"):
                row[field] = ";".join(reference(value) for value in row[field].split(";"))
        for old, new in zip(original, rows):
            if old != new:
                changed.append(new["dataset_name"])
            assert all(new[k] == old[k] for k in columns if k not in {"image_path", "marker_path", "mask_path", "meta_path"})
        temporary = table.with_suffix(".csv.partial")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        if table.read_bytes() != before:
            raise ValueError("Resource catalog changed during metadata path publication")
        os.replace(temporary, table)
        with table.open(encoding="utf-8-sig", newline="") as stream:
            assert list(csv.DictReader(stream)) == rows
    return {"updated_rows": len(changed), "studies": changed, "nonpath_fields_unchanged": True,
            "parsons_row_unchanged": next(x for x in rows if x["dataset_name"] == ACTIVE_STUDY) == next(x for x in original if x["dataset_name"] == ACTIVE_STUDY)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        plan = make_plan(args.data_root)
        write_state(args.plan, plan)
        print(json.dumps({"files": plan["source_file_count"], "bundles": len(plan["groups"]),
                          "merge_files": sum(len(g["members"]) for g in plan["groups"]),
                          "protected_files": len(plan["protected"]), "retire_locks": len(plan["locks"])}))
        return
    if args.state is None:
        parser.error("--execute requires --state")
    plan = json.loads(args.plan.read_text())
    if plan["root"] != str(args.data_root):
        raise ValueError("Plan data root does not match requested data root")
    result = execute(plan, args.state)
    print(json.dumps({"status": result["status"], "bundles": len(result["completed"]),
                      "retired_files": result["retired_files"], "retired_logical_bytes": result["retired_logical_bytes"]}))


if __name__ == "__main__":
    main()
