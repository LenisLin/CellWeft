"""Prepare verified adult and fetal pancreas native-intensity inputs.

Inputs: S-BIAD3557 adult/fetal channel TIFFs, collector native proofs, ordered
markers, fetal section metadata, and this study's S1 rows. Outputs: CYX TIFFs and the
per-study image, panel, sample, field and preparer-review tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import (
    read_metadata_index, read_metadata_member,
)

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import uuid
from urllib.parse import quote, unquote

import pandas as pd

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.inputs import (
    file_revision, input_binding, same_processing_binding,
)
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)


STUDY = "MartinezCasals2026_CODEX_Pancreas"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
SOURCE = NAS / "raw/public" / STUDY / "S-BIAD3557"
NATIVE_METADATA_BUNDLE = NAS / "raw/public" / STUDY / "metadata.json"
OUTPUT = NAS / "processed" / STUDY
NATIVE_EVIDENCE = OUTPUT / "preparation_evidence" / "native_source_scope"
PROOFS = NATIVE_EVIDENCE / "proofs"
COLLECTION_REPORT = NATIVE_EVIDENCE / "collection_report.json"
NATIVE_SCOPE_RECEIPT = OUTPUT / "storage_reception_20261110/native_input_scope_receipt.json"
NATIVE_SCOPE_MIGRATION = OUTPUT / "storage_reception_20261110/native_scope_migration.json"
NATIVE_SCOPE_SUCCESSOR = OUTPUT / "storage_reception_20261110/frozen_scope.json"
PREVIOUS_JOB = PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0808_prepare_0"
JOB = Path(os.environ.get("CELLWEFT_STUDY_JOB", str(PREVIOUS_JOB)))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PUBLIC_RECORD = "https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD3557"
PREPARER = JOB.name
GIB = 1024 ** 3
NATIVE_SCOPE_SCHEMA = "cellweft.martinez_native_input_scope.v1"
PREVIOUS_FROZEN_SCOPE = (PROJECT / "cache/data_return_execution_20261107/studies"
                         / f"{STUDY}.json")
PINNED_MIGRATION_BASELINE = {
    "scope": {"path": str(PREVIOUS_FROZEN_SCOPE), "size": 535892,
              "mtime_ns": 1794017053701531275},
    "completion": {
        "path": str(NAS / "results/storage_return_20261107/completed_studies"
                    / f"{STUDY}.json"),
        "size": 9978, "mtime_ns": 1794093815349296282,
    },
    "summary": {"path": str(OUTPUT / "processing_summary.json"), "size": 8758,
                 "mtime_ns": 1794093813715285584},
    "images": {"path": str(OUTPUT / "images.parquet"), "size": 50355,
                "mtime_ns": 1793249948865926530},
}


def _valid_revision(revision: dict) -> bool:
    return (isinstance(revision, dict) and set(revision) == {"path", "size", "mtime_ns"}
            and isinstance(revision["path"], str)
            and type(revision["size"]) is int and revision["size"] >= 0
            and type(revision["mtime_ns"]) is int)


def native_metadata_reference(member: str) -> str:
    """Return one canonical member reference in this study's raw metadata bundle."""
    key = PurePosixPath(member)
    if key.is_absolute() or str(key) != member or any(part in {".", ".."} for part in key.parts):
        raise ValueError(f"Invalid Martinez raw metadata member: {member}")
    return f"raw/public/{STUDY}/metadata.json#public/{STUDY}/{member}"


def read_native_metadata_member(member: str, *,
                                index: dict | None = None) -> bytes:
    """Read a native metadata fact directly from its pinned raw bundle."""
    key = f"public/{STUDY}/{member}"
    return read_metadata_member(NATIVE_METADATA_BUNDLE, key, index=index)


def _native_metadata_json(member: str, *, index: dict) -> dict:
    return json.loads(read_native_metadata_member(member, index=index))


def _valid_core_id(core_id: object) -> bool:
    if not isinstance(core_id, str) or not core_id or "\\" in core_id:
        return False
    path = PurePosixPath(core_id)
    return (not path.is_absolute() and str(path) == core_id
            and all(part not in {".", ".."} for part in path.parts))


def build_native_scope_receipt(collection_report: dict,
                               proof_payloads: dict[str, dict],
                               proof_revisions: dict[str, dict], *,
                               report_revision: dict,
                               metadata_bundle_revision: dict,
                               unavailable_metadata_members: tuple[str, ...] = ()) -> dict:
    """Preserve the collector's accepted scope and complete per-core proofs."""
    accepted = collection_report.get("accepted_core_ids")
    if (not isinstance(accepted, list) or not all(_valid_core_id(core_id)
                                                   for core_id in accepted)
            or len(accepted) != len(set(accepted))):
        raise ValueError("Collection report lacks unique accepted adult core IDs")
    if not _valid_revision(report_revision) or not _valid_revision(metadata_bundle_revision):
        raise ValueError("Native scope receipt requires exact source revisions")
    expected_report = str(COLLECTION_REPORT.resolve())
    expected_bundle = str((NAS / "raw/public" / STUDY / "metadata.json").resolve())
    if (report_revision["path"] != expected_report
            or metadata_bundle_revision["path"] != expected_bundle):
        raise ValueError("Native scope receipt source paths differ from this study")
    if len(unavailable_metadata_members) != len(set(unavailable_metadata_members)):
        raise ValueError("Native scope receipt repeats an unavailable metadata member")
    for member in unavailable_metadata_members:
        native_metadata_reference(member)
    if set(proof_payloads) != set(proof_revisions):
        raise ValueError("Native proof payloads and source revisions differ")
    if not all(_valid_core_id(core_id) for core_id in proof_payloads):
        raise ValueError("Native proof set contains an invalid core ID")
    proof_names = [core_id.replace("/", "__") + ".json" for core_id in proof_payloads]
    if len(proof_names) != len(set(proof_names)):
        raise ValueError("Native core IDs map to duplicate collector proof names")
    if not set(accepted).issubset(proof_payloads):
        raise ValueError("Accepted adult core lacks its collector proof")
    proofs = {}
    for core_id, proof in proof_payloads.items():
        revision = proof_revisions[core_id]
        expected_proof = str((PROOFS / (core_id.replace("/", "__") + ".json")).resolve())
        if (not isinstance(proof, dict) or not _valid_revision(revision)
                or revision["path"] != expected_proof):
            raise ValueError(f"Invalid native proof identity: {core_id}")
        proofs[core_id] = {"source_revision": revision, "proof": proof}
    return {
        "schema": NATIVE_SCOPE_SCHEMA,
        "study": STUDY,
        "collection_report_revision": report_revision,
        "metadata_bundle_revision": metadata_bundle_revision,
        "unavailable_metadata_members": list(unavailable_metadata_members),
        "collection_report": collection_report,
        "proofs": proofs,
    }


def collect_native_scope_receipt() -> dict:
    """Capture collector facts against the current raw metadata bundle revision."""
    bundle = NATIVE_METADATA_BUNDLE
    bundle_revision = file_revision(bundle)
    _, bundle_index = read_metadata_index(bundle)
    required_members = [
        "native_manifest.json", "ordered_markers.json", "fetal_native_manifest.json",
        "fetal_ordered_markers.json", "README_fetal_spatial_proteomics_normal_pancreas.md",
    ]
    preserved_dependencies = [
        "README_adult_spatial_proteomics_normal_pancreas.md",
        "adult_annotation_identity_verification_20261028_v2.json",
    ]
    if f"public/{STUDY}/member_recovered_cores.json" in bundle_index:
        required_members.append("member_recovered_cores.json")
    for member in required_members:
        key = f"public/{STUDY}/{member}"
        if key not in bundle_index:
            raise ValueError(f"Required native raw metadata member is absent: {member}")
    unavailable_members = tuple(
        member for member in preserved_dependencies
        if f"public/{STUDY}/{member}" not in bundle_index)
    report_revision = file_revision(COLLECTION_REPORT)
    report_bytes = COLLECTION_REPORT.read_bytes()
    if file_revision(COLLECTION_REPORT) != report_revision:
        raise ValueError("Collection report changed while staging native scope")
    collection_report = json.loads(report_bytes)
    accepted = collection_report.get("accepted_core_ids")
    if not isinstance(accepted, list):
        raise ValueError("Collection report lacks accepted adult core IDs")
    adult_manifest = _native_metadata_json("native_manifest.json", index=bundle_index)
    adult_ordered = _native_metadata_json("ordered_markers.json", index=bundle_index)
    fetal_manifest = _native_metadata_json("fetal_native_manifest.json", index=bundle_index)
    fetal_ordered = _native_metadata_json("fetal_ordered_markers.json", index=bundle_index)
    recovered = (_native_metadata_json("member_recovered_cores.json", index=bundle_index)["core_proofs"]
                 if f"public/{STUDY}/member_recovered_cores.json" in bundle_index else {})
    adult_records = {item["core_id"]: item for item in adult_manifest["cores"]}
    fetal_records = {item["directory_id"]: item for item in fetal_manifest["directories"]}
    if not set(accepted).issubset(adult_records) or not set(recovered).issubset(adult_records):
        raise ValueError("Native report/recovery scope differs from the adult manifest")
    proof_ids = list(dict.fromkeys([*accepted, *recovered, *fetal_records]))
    proof_payloads, proof_revisions = {}, {}
    for core_id in proof_ids:
        path = PROOFS / (core_id.replace("/", "__") + ".json")
        revision = file_revision(path)
        payload = json.loads(path.read_bytes())
        if file_revision(path) != revision:
            raise ValueError(f"Collector proof changed while staging: {core_id}")
        record = adult_records.get(core_id, fetal_records.get(core_id))
        ordered = adult_ordered.get(core_id, fetal_ordered.get(core_id))
        if not payload.get("all_channels_read") or payload.get("record") != record or ordered != record["channels"]:
            raise ValueError(f"Collector identity/proof differs: {core_id}")
        if core_id in recovered:
            evidence = payload.get("member_recovery", {})
            recovered_revision = recovered[core_id]
            proof_name = core_id.replace("/", "__") + ".json"
            legacy_proof = (NAS / "cache/collection_20261018/pancreas_biad3557_native_0320"
                            / "proofs" / proof_name)
            relocated_recovery_revision = (
                recovered_revision.get("path") == str(legacy_proof)
                and recovered_revision.get("size") == revision["size"]
                and recovered_revision.get("mtime_ns") == revision["mtime_ns"]
                and revision.get("path") == str(PROOFS / proof_name)
            )
            if (not evidence.get("qualified") or evidence.get("complete_archive_received")
                    or (revision != recovered_revision and not relocated_recovery_revision)):
                raise ValueError(f"Recovered member proof version differs: {core_id}")
        proof_payloads[core_id] = payload
        proof_revisions[core_id] = revision
    if file_revision(bundle) != bundle_revision:
        raise ValueError("Raw metadata bundle changed while staging native scope")
    return build_native_scope_receipt(
        collection_report, proof_payloads, proof_revisions,
        report_revision=report_revision,
        metadata_bundle_revision=bundle_revision,
        unavailable_metadata_members=unavailable_members)


def read_native_scope_receipt(path: Path | None = None) -> dict:
    """Read the received scope fact without consulting the retired collection cache."""
    path = NATIVE_SCOPE_RECEIPT if path is None else Path(path)
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("schema") != NATIVE_SCOPE_SCHEMA or receipt.get("study") != STUDY:
        raise ValueError(f"Invalid native input scope receipt: {path}")
    report = receipt.get("collection_report")
    proofs = receipt.get("proofs")
    if not isinstance(report, dict) or not isinstance(proofs, dict):
        raise ValueError(f"Incomplete native input scope receipt: {path}")
    proof_names = [core_id.replace("/", "__") + ".json" for core_id in proofs]
    if len(proof_names) != len(set(proof_names)):
        raise ValueError(f"Duplicate native proof source names in receipt: {path}")
    accepted = report.get("accepted_core_ids")
    if (not isinstance(accepted, list)
            or not all(_valid_core_id(core_id) for core_id in accepted)
            or len(accepted) != len(set(accepted)) or not set(accepted).issubset(proofs)):
        raise ValueError(f"Invalid accepted core set in native input scope receipt: {path}")
    if (not _valid_revision(receipt.get("collection_report_revision"))
            or not _valid_revision(receipt.get("metadata_bundle_revision"))):
        raise ValueError(f"Missing source revisions in native input scope receipt: {path}")
    unavailable_members = receipt.get("unavailable_metadata_members", [])
    if (not isinstance(unavailable_members, list)
            or len(unavailable_members) != len(set(unavailable_members))):
        raise ValueError(f"Invalid unavailable metadata members in native input scope receipt: {path}")
    for member in unavailable_members:
        native_metadata_reference(member)
    if (receipt["collection_report_revision"]["path"] != str(COLLECTION_REPORT.resolve())
            or receipt["metadata_bundle_revision"]["path"]
            != str((NAS / "raw/public" / STUDY / "metadata.json").resolve())):
        raise ValueError(f"Native input scope receipt source paths differ: {path}")
    for core_id, entry in proofs.items():
        if not _valid_core_id(core_id):
            raise ValueError(f"Invalid native proof ID in receipt: {core_id}")
        expected_proof = str((PROOFS / (core_id.replace("/", "__") + ".json")).resolve())
        if not isinstance(entry, dict) or set(entry) != {"source_revision", "proof"}:
            raise ValueError(f"Invalid proof member in native input scope receipt: {core_id}")
        proof = entry["proof"]
        proof_record = proof.get("record") if isinstance(proof, dict) else None
        if not isinstance(proof_record, dict):
            raise ValueError(f"Native proof record is malformed: {core_id}")
        record_id = proof_record.get("core_id", proof_record.get("directory_id"))
        if (not _valid_revision(entry["source_revision"])
                or proof.get("all_channels_read") is not True
                or record_id != core_id
                or entry["source_revision"]["path"] != expected_proof):
            raise ValueError(f"Invalid proof member in native input scope receipt: {core_id}")
    return receipt


def native_scope_proof_reference(core_id: str,
                                 receipt_path: Path | None = None) -> str:
    """Return a canonical opaque reference to one received proof member."""
    receipt_path = NATIVE_SCOPE_RECEIPT if receipt_path is None else Path(receipt_path)
    return f"{receipt_path.relative_to(NAS).as_posix()}#proofs/{quote(core_id, safe='')}"


def proof_from_reference(reference: str,
                         receipt: dict | None = None) -> tuple[str, dict]:
    """Resolve only this study's exact receipt-member reference."""
    path, separator, fragment = reference.partition("#")
    if (not separator or path != NATIVE_SCOPE_RECEIPT.relative_to(NAS).as_posix()
            or not fragment.startswith("proofs/") or fragment.count("/") != 1):
        raise ValueError(f"Invalid Martinez native proof reference: {reference}")
    encoded_id = fragment.removeprefix("proofs/")
    core_id = unquote(encoded_id)
    if quote(core_id, safe="") != encoded_id:
        raise ValueError(f"Noncanonical Martinez native proof reference: {reference}")
    current = receipt if receipt is not None else read_native_scope_receipt()
    try:
        return core_id, current["proofs"][core_id]["proof"]
    except KeyError as error:
        raise ValueError(f"Unknown Martinez native proof member: {core_id}") from error


def rebind_native_split_evidence(images: pd.DataFrame,
                                 receipt: dict) -> pd.DataFrame:
    """Replace only obsolete per-image collector-proof pointers."""
    required = {"study", "image_id", "source_image_id", "image_kind", "split_evidence"}
    if not required.issubset(images.columns):
        raise ValueError("Martinez image table lacks native proof provenance fields")
    result = images.copy()
    seen: set[str] = set()
    for index, row in result[result.study == STUDY].iterrows():
        core_id = str(row.source_image_id)
        if core_id not in receipt["proofs"] or core_id in seen:
            raise ValueError(f"Image table proof identity is missing or repeated: {core_id}")
        seen.add(core_id)
        if str(row.image_id) != core_id.replace("/", "_"):
            raise ValueError(f"Image table key differs from native proof identity: {core_id}")
        if row.image_kind not in {"tma_core", "roi"}:
            raise ValueError(f"Unexpected native image kind: {row.image_id}")
        proof = receipt["proofs"][core_id]["proof"]
        record_id = proof.get("record", {}).get("core_id",
                                               proof.get("record", {}).get("directory_id"))
        if proof.get("all_channels_read") is not True or record_id != core_id:
            raise ValueError(f"Collector proof identity differs: {core_id}")
        manifest = "native_manifest.json" if row.image_kind == "tma_core" else "fetal_native_manifest.json"
        expected_manifest = native_metadata_reference(manifest)
        prefix, separator, old_reference = str(row.split_evidence).partition(" and ")
        legacy_manifest = f"metadata/public/{STUDY}/{manifest}"
        if not separator or prefix not in {expected_manifest, legacy_manifest}:
            raise ValueError(f"Image table manifest evidence differs: {row.image_id}")
        new_reference = native_scope_proof_reference(core_id)
        old_name = core_id.replace("/", "__") + ".json"
        if Path(old_reference).parent.name == "proofs" and Path(old_reference).name == old_name:
            result.at[index, "split_evidence"] = f"{expected_manifest} and {new_reference}"
        elif old_reference != new_reference:
            raise ValueError(f"Unexpected native proof reference: {row.image_id}")
    return result


def bind_frozen_scope_to_rebound_images(frozen: dict, new_revision: dict) -> dict:
    """Create a successor scope whose only changed binding is images.parquet."""
    if not isinstance(frozen, dict) or not _valid_revision(new_revision):
        raise ValueError("Frozen scope and images table revision are required")
    expected_path = str((Path(frozen["target"]) / "images.parquet").resolve())
    if new_revision["path"] != expected_path:
        raise ValueError("Successor scope images revision targets another table")
    result = deepcopy(frozen)
    result["images_table_revision"] = new_revision
    return result


def rebind_active_d01_reviews(reviews: pd.DataFrame, old_binding: dict,
                              new_binding: dict, receipt_revision: dict) -> tuple[pd.DataFrame, int]:
    """Refresh only current exact D01 bindings when processing identity is unchanged."""
    if not same_processing_binding(old_binding, new_binding):
        raise ValueError("Native proof migration changes the processing binding")
    result = reviews.copy()
    current = result[(result.study == STUDY) & (result.check_id == "D01")]
    latest = current.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
    rebound = 0
    for index, row in latest.iterrows():
        if row.result != "pass" or not row.details.lstrip().startswith("{"):
            continue
        details = json.loads(row.details)
        if details.get("input_binding") != old_binding:
            continue
        details["input_binding"] = new_binding
        details["metadata_only_rebind"] = {
            "changed_field": "images.parquet.split_evidence",
            "native_input_scope_receipt": receipt_revision,
        }
        result.at[index, "details"] = json.dumps(
            details, ensure_ascii=False, sort_keys=True)
        rebound += 1
    return result, rebound


def rebind_processing_summary(summary: dict, old_revisions: dict[str, dict],
                              new_revisions: dict[str, dict],
                              receipt_revision: dict) -> dict:
    """Bind administrative table replacements while retaining all science facts."""
    if not _valid_revision(receipt_revision) or receipt_revision["path"] != str(
            NATIVE_SCOPE_RECEIPT.resolve()):
        raise ValueError("Summary migration receipt revision differs from this study")
    result = deepcopy(summary)
    table_revisions = result.get("nas_five_table_revisions")
    if (not isinstance(table_revisions, dict)
            or table_revisions.get("after") != old_revisions
            or not isinstance(table_revisions.get("unchanged"), dict)):
        raise ValueError("Summary table revisions do not match the captured snapshot")
    allowed = {"images.parquet", "review_results.csv"}
    if not set(new_revisions).issubset(allowed):
        raise ValueError("Native proof migration changed an unexpected study table")
    changed = []
    for name, revision in new_revisions.items():
        if not _valid_revision(revision) or revision["path"] != str((OUTPUT / name).resolve()):
            raise ValueError(f"Invalid migrated table revision: {name}")
        if old_revisions.get(name) == revision:
            raise ValueError(f"Migrated table revision did not change: {name}")
        table_revisions["after"][name] = revision
        table_revisions["unchanged"][name] = False
        changed.append(name)
    result["metadata_only_migration"] = {
        "status": "metadata_only",
        "changed_files": changed,
        "changed_columns": ["images.parquet:split_evidence"] + (
            ["review_results.csv:details.input_binding"]
            if "review_results.csv" in changed else []),
        "scientific_content_changed": False,
        "native_input_scope_receipt": receipt_revision,
    }
    return result


def _sync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _migration_target_revision(stage: Path, target: Path) -> dict:
    staged = file_revision(stage)
    return {"path": str(target.resolve()), "size": staged["size"],
            "mtime_ns": staged["mtime_ns"]}


def _publish_migration_member(target: Path, stage: Path, *,
                              old_revision: dict | None,
                              new_revision: dict, immutable: bool) -> None:
    """Publish or forward-recover one journal-owned file by exact revision."""
    target, stage = Path(target), Path(stage)
    if new_revision["path"] != str(target.resolve()):
        raise ValueError(f"Migration journal target path differs: {target}")
    if target.exists():
        current = file_revision(target)
        if current == new_revision:
            if stage.exists():
                staged = file_revision(stage)
                if (staged["size"], staged["mtime_ns"]) != (
                        new_revision["size"], new_revision["mtime_ns"]):
                    raise ValueError(f"Migration stage differs from published file: {target}")
                stage.unlink()
                _sync_directory(stage.parent)
            _sync_file(target)
            _sync_directory(target.parent)
            return
        if immutable or old_revision is None or current != old_revision:
            raise ValueError(f"Migration target has an unknown version: {target}")
    elif old_revision is not None:
        raise ValueError(f"Migration predecessor is absent: {target}")
    if not stage.is_file():
        raise FileNotFoundError(f"Journal-owned migration stage is absent: {stage}")
    staged = file_revision(stage)
    if (staged["size"], staged["mtime_ns"]) != (
            new_revision["size"], new_revision["mtime_ns"]):
        raise ValueError(f"Journal-owned migration stage changed: {stage}")
    if immutable:
        try:
            os.link(stage, target)
        except FileExistsError as error:
            raise ValueError(f"Immutable migration member already exists: {target}") from error
        stage.unlink()
    else:
        os.replace(stage, target)
    _sync_file(target)
    _sync_directory(target.parent)
    if stage.parent != target.parent:
        _sync_directory(stage.parent)
    if file_revision(target) != new_revision:
        raise ValueError(f"Migration publication did not preserve its staged version: {target}")


def _write_migration_journal(path: Path, payload: dict, txid: str,
                             previous_revision: dict | None) -> dict:
    temporary = path.with_name(path.name + f".partial.{txid}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    _sync_file(temporary)
    if previous_revision is None:
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("Martinez native migration journal already exists") from error
        temporary.unlink()
    else:
        if file_revision(path) != previous_revision:
            raise ValueError("Martinez native migration journal changed during update")
        os.replace(temporary, path)
    _sync_directory(path.parent)
    return file_revision(path)


def _runner_module():
    path = Path(__file__).with_name("run_storage_return.py")
    spec = importlib.util.spec_from_file_location("cellweft_martinez_storage_return", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the storage-return completion API")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_revision_map(target: Path) -> dict[str, dict]:
    names = ["images.parquet", "sample_metadata.csv", "field_mapping.csv",
             "review_results.csv", "panel_mapping.csv"]
    return {name: file_revision(target / name) for name in names}


def _current_migration_revision(runner, revision: dict) -> dict:
    relocation = runner._relocation(STUDY)
    if relocation is None:
        raise ValueError("Martinez migration requires the activated processed reception")
    return relocation.translate(revision)


def _validate_migration_journal(journal: dict) -> None:
    txid = journal.get("transaction_id")
    if (journal.get("schema") != "cellweft.martinez_native_scope_migration.v1"
            or journal.get("study") != STUDY
            or not isinstance(txid, str) or re.fullmatch(r"[0-9a-f]{32}", txid) is None):
        raise ValueError("Martinez native migration journal identity is invalid")
    baseline = journal.get("baseline")
    if not isinstance(baseline, dict) or any(
            baseline.get(name) != PINNED_MIGRATION_BASELINE[name]
            for name in ("scope", "completion", "summary", "images")):
        raise ValueError("Martinez migration journal baseline differs from captured identity")
    tables = baseline.get("tables")
    expected_names = {"images.parquet", "sample_metadata.csv", "field_mapping.csv",
                      "review_results.csv", "panel_mapping.csv"}
    if not isinstance(tables, dict) or set(tables) != expected_names:
        raise ValueError("Martinez migration journal table baseline is incomplete")
    for name, revision in tables.items():
        if (not _valid_revision(revision)
                or revision["path"] != str((OUTPUT / name).resolve())):
            raise ValueError(f"Martinez migration journal has an invalid table baseline: {name}")
    expected_backups = {
        "images": str(NATIVE_SCOPE_RECEIPT.parent /
                      f".martinez-native-{txid}-before-images.parquet"),
        "summary": str(NATIVE_SCOPE_RECEIPT.parent /
                       f".martinez-native-{txid}-before-processing_summary.json"),
        "review": str(NATIVE_SCOPE_RECEIPT.parent /
                      f".martinez-native-{txid}-before-review_results.csv"),
    }
    backup_paths = journal.get("backup_paths")
    if backup_paths != expected_backups:
        raise ValueError("Martinez migration journal snapshot paths differ from this transaction")
    backup_revisions = journal.get("backup_revisions")
    if backup_revisions is not None:
        old_by_name = {"images": baseline["images"], "summary": baseline["summary"],
                       "review": tables["review_results.csv"]}
        if set(backup_revisions) != set(expected_backups):
            raise ValueError("Martinez migration journal snapshots are incomplete")
        for name, backup_path in expected_backups.items():
            revision = backup_revisions[name]
            old_revision = old_by_name[name]
            if (not _valid_revision(revision) or revision["path"] != backup_path
                    or (revision["size"], revision["mtime_ns"])
                    != (old_revision["size"], old_revision["mtime_ns"])):
                raise ValueError(f"Martinez migration journal snapshot is invalid: {name}")
    expected_targets = {
        "receipt": str(NATIVE_SCOPE_RECEIPT.resolve()),
        "images": str((OUTPUT / "images.parquet").resolve()),
        "review": str((OUTPUT / "review_results.csv").resolve()),
        "summary": str((OUTPUT / "processing_summary.json").resolve()),
        "scope": str(NATIVE_SCOPE_SUCCESSOR.resolve()),
    }
    targets = journal.get("targets")
    if (not isinstance(targets, dict)
            or any(targets.get(key) not in {value, None} for key, value in expected_targets.items())
            or any(targets.get(key) != value for key, value in expected_targets.items()
                   if key != "review")):
        raise ValueError("Martinez migration journal target paths differ from this study")
    expected_stages = {
        name: str(NATIVE_SCOPE_RECEIPT.parent /
                  f".martinez-native-{txid}-{name}.partial")
        for name in ("receipt", "images", "review", "summary", "scope")
    }
    stages = journal.get("stage_paths")
    if (not isinstance(stages, dict)
            or any(stages.get(key) not in {value, None}
                   for key, value in expected_stages.items())
            or any(stages.get(key) != value for key, value in expected_stages.items()
                   if key != "review")
            or (targets["review"] is None) != (stages["review"] is None)):
        raise ValueError("Martinez migration journal stage paths differ from this transaction")
    revisions = journal.get("target_revisions")
    if revisions is None:
        return
    if backup_revisions is None:
        raise ValueError("Martinez migration cannot journal outputs before source snapshots")
    expected_revision_paths = {
        "receipt": targets["receipt"], "images": targets["images"],
        "summary": targets["summary"], "scope": targets["scope"],
    }
    if targets["review"] is not None:
        expected_revision_paths["review"] = targets["review"]
    if not isinstance(revisions, dict) or set(revisions) != set(expected_revision_paths):
        raise ValueError("Martinez migration journal output revisions are incomplete")
    for name, path in expected_revision_paths.items():
        if not _valid_revision(revisions[name]) or revisions[name]["path"] != path:
            raise ValueError(f"Martinez migration journal output revision is invalid: {name}")
    staged = journal.get("stage_revisions")
    expected_stage_names = {key for key, value in stages.items() if value is not None}
    if not isinstance(staged, dict) or set(staged) != expected_stage_names:
        raise ValueError("Martinez migration journal stage revisions are incomplete")
    for name, revision in staged.items():
        if (not _valid_revision(revision)
                or revision["path"] != stages[name]
                or (revision["size"], revision["mtime_ns"])
                != (revisions[name]["size"], revisions[name]["mtime_ns"])):
            raise ValueError(f"Martinez migration journal stage revision is invalid: {name}")


def _create_or_validate_migration_backups(journal: dict) -> dict[str, dict]:
    sources = {
        "images": OUTPUT / "images.parquet",
        "summary": OUTPUT / "processing_summary.json",
        "review": OUTPUT / "review_results.csv",
    }
    old_revisions = {
        "images": journal["baseline"]["images"],
        "summary": journal["baseline"]["summary"],
        "review": journal["baseline"]["tables"]["review_results.csv"],
    }
    backups = {}
    for name, source in sources.items():
        target = Path(journal["backup_paths"][name])
        if target.exists():
            revision = file_revision(target)
        else:
            if file_revision(source) != old_revisions[name]:
                raise ValueError(f"Martinez predecessor changed before its snapshot: {name}")
            os.link(source, target)
            _sync_directory(target.parent)
            revision = file_revision(target)
        expected = {"path": str(target.resolve()), "size": old_revisions[name]["size"],
                    "mtime_ns": old_revisions[name]["mtime_ns"]}
        if revision != expected:
            raise ValueError(f"Martinez migration snapshot has an unknown version: {name}")
        backups[name] = revision
    return backups


def _migration_candidate(journal: dict, name: str) -> Path:
    target = Path(journal["targets"][name])
    expected = journal["target_revisions"][name]
    if target.exists() and file_revision(target) == expected:
        return target
    stage = Path(journal["stage_paths"][name])
    if stage.exists() and file_revision(stage) == journal["stage_revisions"][name]:
        return stage
    raise ValueError(f"Martinez migration lacks its exact staged or published {name} member")


def _verify_migration_candidates(journal: dict, receipt: dict) -> None:
    """Re-derive every staged successor from retained predecessor snapshots."""
    _validate_migration_journal(journal)
    if journal.get("source_revisions") != {
        "collection_report": receipt["collection_report_revision"],
        "metadata_bundle": receipt["metadata_bundle_revision"],
        "unavailable_metadata_members": receipt.get("unavailable_metadata_members", []),
        "proofs": {core_id: entry["source_revision"]
                   for core_id, entry in receipt["proofs"].items()},
    }:
        raise ValueError("Martinez migration source versions differ from the journal")
    for name, revision in journal["backup_revisions"].items():
        if file_revision(Path(journal["backup_paths"][name])) != revision:
            raise ValueError(f"Martinez migration predecessor snapshot changed: {name}")

    old_images = pd.read_parquet(journal["backup_paths"]["images"])
    old_summary = json.loads(Path(journal["backup_paths"]["summary"]).read_text(
        encoding="utf-8"))
    old_reviews = pd.read_csv(journal["backup_paths"]["review"], keep_default_na=False)
    new_images = rebind_native_split_evidence(old_images, receipt)
    if not new_images.drop(columns="split_evidence").equals(
            old_images.drop(columns="split_evidence")):
        raise ValueError("Recovered migration changes a non-provenance image field")
    candidate_images = pd.read_parquet(_migration_candidate(journal, "images"))
    if not candidate_images.equals(new_images):
        raise ValueError("Recovered image table differs beyond native split evidence")

    receipt_revision = journal["target_revisions"]["receipt"]
    candidate_receipt = json.loads(_migration_candidate(journal, "receipt").read_text(
        encoding="utf-8"))
    if candidate_receipt != receipt:
        raise ValueError("Recovered native input receipt differs from its captured source facts")

    old_revisions = journal["baseline"]["tables"]
    new_revisions = {"images.parquet": journal["target_revisions"]["images"]}
    expected_reviews, active_count = _active_d01_table_rebind(
        old_reviews, old_images, new_images, receipt_revision)
    if active_count:
        if journal["targets"].get("review") is None:
            raise ValueError("Active D01 binding was omitted from the migration journal")
        new_revisions["review_results.csv"] = journal["target_revisions"]["review"]
        candidate_reviews = pd.read_csv(_migration_candidate(journal, "review"),
                                        keep_default_na=False)
        if not candidate_reviews.equals(expected_reviews):
            raise ValueError("Recovered D01 table differs beyond its exact metadata rebind")
    elif journal["targets"].get("review") is not None:
        raise ValueError("Migration journal changed review_results without an active D01 pass")
    if active_count != journal.get("active_d01_rebindings"):
        raise ValueError("Recovered D01 binding count differs from its captured source state")

    expected_summary = rebind_processing_summary(
        old_summary, old_revisions, new_revisions, receipt_revision)
    candidate_summary = json.loads(_migration_candidate(journal, "summary").read_text(
        encoding="utf-8"))
    if candidate_summary != expected_summary:
        raise ValueError("Recovered processing summary changes facts outside the table binding")
    frozen = json.loads(PREVIOUS_FROZEN_SCOPE.read_text(encoding="utf-8"))
    expected_scope = bind_frozen_scope_to_rebound_images(
        frozen, journal["target_revisions"]["images"])
    candidate_scope = json.loads(_migration_candidate(journal, "scope").read_text(
        encoding="utf-8"))
    if candidate_scope != expected_scope:
        raise ValueError("Recovered frozen scope changes facts beyond images_table_revision")
    if (journal.get("table_image_ids") != old_images[
            old_images.study.astype(str) == STUDY].image_id.astype(str).tolist()
            or journal.get("frozen_image_ids") != [row["image_id"] for row in frozen["images"]]):
        raise ValueError("Recovered migration image order differs from predecessor snapshots")


def _binding_with_rebound_images(binding: dict, images: pd.DataFrame) -> dict:
    image_ids = binding.get("image_ids")
    if (not isinstance(image_ids, list) or not image_ids
            or len(image_ids) != len(set(image_ids))):
        raise ValueError("Current D01 pass has an invalid image scope")
    if images.image_id.astype(str).duplicated().any() or not set(image_ids).issubset(
            set(images.image_id.astype(str))):
        raise ValueError("Rebound image table differs from the current D01 scope")
    selected = images.set_index(images.image_id.astype(str), drop=False).loc[
        sorted(image_ids)].reset_index(drop=True)
    result = deepcopy(binding)
    result["images"] = json.loads(selected.to_json(orient="records"))
    return result


def _input_binding_for_image_table(images: pd.DataFrame, image_ids: list[str]) -> dict:
    """Build the existing full binding against a captured table without rewriting it."""
    with tempfile.TemporaryDirectory(prefix=f"{STUDY}-binding-") as directory:
        root = Path(directory)
        images.to_parquet(root / "images.parquet", index=False)
        for name in ("sample_metadata.csv", "field_mapping.csv", "panel_mapping.csv"):
            os.symlink(OUTPUT / name, root / name)
        return input_binding(root, NAS, image_ids)


def _active_d01_table_rebind(reviews: pd.DataFrame, old_images: pd.DataFrame,
                             new_images: pd.DataFrame,
                             receipt_revision: dict) -> tuple[pd.DataFrame, int]:
    """Rebind only latest D01 passes that exactly consume the captured table."""
    result = reviews.copy()
    current = result[(result.study == STUDY) & (result.check_id == "D01")]
    latest = current.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
    rebound = 0
    for index, row in latest.iterrows():
        if row.result != "pass" or not row.details.lstrip().startswith("{"):
            continue
        details = json.loads(row.details)
        old_binding = details.get("input_binding")
        if (not isinstance(old_binding, dict)
                or not isinstance(old_binding.get("image_ids"), list)
                or not old_binding["image_ids"]):
            continue
        current_binding = _input_binding_for_image_table(
            old_images, old_binding["image_ids"])
        if old_binding != current_binding:
            continue
        new_binding = _binding_with_rebound_images(old_binding, new_images)
        rebound_rows, count = rebind_active_d01_reviews(
            result.loc[[index]], old_binding, new_binding, receipt_revision)
        result.loc[index, "details"] = rebound_rows.loc[index, "details"]
        rebound += count
    return result, rebound


def _baseline_snapshot(runner) -> dict:
    frozen_revision = file_revision(PREVIOUS_FROZEN_SCOPE)
    completion_path = runner.COMPLETED / f"{STUDY}.json"
    completion_revision = file_revision(completion_path)
    relocation = runner._relocation(STUDY)
    if relocation is None:
        raise ValueError("Martinez migration requires the activated processed reception")
    old_completion = relocation.record["entries"][0]["old"]
    old_scope = relocation.record["entries"][-1]["old"]
    expected_completion = PINNED_MIGRATION_BASELINE["completion"]
    expected_scope = PINNED_MIGRATION_BASELINE["scope"]
    if old_completion != expected_completion or old_scope != expected_scope:
        raise ValueError("Martinez relocation differs from the captured historical baseline")
    if (relocation.translate(expected_completion) != completion_revision
            or relocation.translate(expected_scope) != relocation.scope):
        raise ValueError("Martinez processed reception differs from its historical baseline")
    actual = {
        "scope": frozen_revision,
        "completion": old_completion,
        "summary": file_revision(OUTPUT / "processing_summary.json"),
        "images": file_revision(OUTPUT / "images.parquet"),
    }
    if (actual["scope"] != PINNED_MIGRATION_BASELINE["scope"]
            or actual["completion"] != expected_completion
            or actual["summary"] != PINNED_MIGRATION_BASELINE["summary"]
            or actual["images"] != PINNED_MIGRATION_BASELINE["images"]):
        raise ValueError("Martinez migration baseline changed since owner capture")
    return actual


def _stage_migration_payloads(journal: dict, receipt: dict, frozen: dict,
                              old_images: pd.DataFrame, old_revisions: dict,
                              old_summary: dict) -> tuple[dict, dict]:
    """Build closed successor files and record their exact future versions."""
    new_images = rebind_native_split_evidence(old_images, receipt)
    study_rows = old_images[old_images.study.astype(str) == STUDY]
    frozen_ids = [row["image_id"] for row in frozen["images"]]
    if (study_rows.image_id.astype(str).duplicated().any()
            or set(study_rows.image_id.astype(str)) != set(frozen_ids)):
        raise ValueError("Martinez images.parquet keys differ from the frozen accepted scope")
    if not new_images.drop(columns="split_evidence").equals(
            old_images.drop(columns="split_evidence")):
        raise ValueError("Native proof migration changed a non-provenance image field")
    journal["table_image_ids"] = study_rows.image_id.astype(str).tolist()
    journal["frozen_image_ids"] = frozen_ids

    receipt_stage = Path(journal["stage_paths"]["receipt"])
    images_stage = Path(journal["stage_paths"]["images"])
    summary_stage = Path(journal["stage_paths"]["summary"])
    scope_stage = Path(journal["stage_paths"]["scope"])
    receipt_stage.write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    new_images.to_parquet(images_stage, index=False)
    _sync_file(receipt_stage)
    _sync_file(images_stage)
    targets = journal["targets"]
    receipt_revision = _migration_target_revision(receipt_stage, Path(targets["receipt"]))
    images_revision = _migration_target_revision(images_stage, Path(targets["images"]))

    review_target = targets.get("review")
    new_revisions = {"images.parquet": images_revision}
    review_count = 0
    if review_target is not None:
        reviews = pd.read_csv(journal["backup_paths"]["review"], keep_default_na=False)
        rebound_reviews, review_count = _active_d01_table_rebind(
            reviews, old_images, new_images, receipt_revision)
        if review_count:
            review_stage = Path(journal["stage_paths"]["review"])
            rebound_reviews.to_csv(review_stage, index=False)
            _sync_file(review_stage)
            new_revisions["review_results.csv"] = _migration_target_revision(
                review_stage, Path(review_target))
        else:
            targets["review"] = None
            journal["stage_paths"]["review"] = None

    summary = old_summary
    migrated_summary = rebind_processing_summary(
        summary, old_revisions, new_revisions, receipt_revision)
    summary_stage.write_text(json.dumps(migrated_summary, ensure_ascii=False,
                                        indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    scope = bind_frozen_scope_to_rebound_images(frozen, images_revision)
    scope_stage.write_text(json.dumps(scope, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    _sync_file(summary_stage)
    _sync_file(scope_stage)
    revisions = {
        "receipt": receipt_revision,
        "images": images_revision,
        "summary": _migration_target_revision(summary_stage, Path(targets["summary"])),
        "scope": _migration_target_revision(scope_stage, Path(targets["scope"])),
    }
    if review_count:
        revisions["review"] = new_revisions["review_results.csv"]
    stage_revisions = {
        key: file_revision(Path(journal["stage_paths"][key]))
        for key in journal["stage_paths"] if journal["stage_paths"][key]
    }
    journal["active_d01_rebindings"] = review_count
    journal["stage_revisions"] = stage_revisions
    journal["target_revisions"] = revisions
    journal["source_revisions"] = {
        "collection_report": receipt["collection_report_revision"],
        "metadata_bundle": receipt["metadata_bundle_revision"],
        "unavailable_metadata_members": receipt.get("unavailable_metadata_members", []),
        "proofs": {core_id: entry["source_revision"]
                   for core_id, entry in receipt["proofs"].items()},
    }
    return scope, revisions


def _publish_native_migration_files(journal: dict) -> None:
    targets = journal["targets"]
    stages = journal["stage_paths"]
    old = journal["baseline"]
    new = journal["target_revisions"]
    changed_tables = {"images.parquet"}
    if targets.get("review") is not None:
        changed_tables.add("review_results.csv")
    for name, old_revision in old["tables"].items():
        current = file_revision(OUTPUT / name)
        new_revision = (new["images"] if name == "images.parquet" else
                        new.get("review") if name == "review_results.csv" else None)
        if name in changed_tables:
            if current != old_revision and current != new_revision:
                raise ValueError(f"Martinez table changed outside the migration: {name}")
        elif current != old_revision:
            raise ValueError(f"Martinez table changed outside the migration: {name}")
    _publish_migration_member(Path(targets["images"]), Path(stages["images"]),
                              old_revision=old["images"], new_revision=new["images"],
                              immutable=False)
    if targets.get("review") is not None:
        _publish_migration_member(Path(targets["review"]), Path(stages["review"]),
                                  old_revision=old["tables"]["review_results.csv"],
                                  new_revision=new["review"], immutable=False)
    _publish_migration_member(Path(targets["summary"]), Path(stages["summary"]),
                              old_revision=old["summary"], new_revision=new["summary"],
                              immutable=False)
    _publish_migration_member(Path(targets["scope"]), Path(stages["scope"]),
                              old_revision=None, new_revision=new["scope"], immutable=True)
    _publish_migration_member(Path(targets["receipt"]), Path(stages["receipt"]),
                              old_revision=None, new_revision=new["receipt"], immutable=True)


def _migration_receipt_from_journal(journal: dict) -> dict:
    receipt_stage = Path(journal["stage_paths"]["receipt"])
    if "target_revisions" in journal and receipt_stage.is_file():
        return read_native_scope_receipt(receipt_stage)
    return collect_native_scope_receipt()


def migrate_native_scope() -> dict:
    """Receive collector proofs and bind their reader/table successor without reprocessing."""
    with formal_publish_lease(NATIVE_SCOPE_RECEIPT, directory=OUTPUT):
        runner = _runner_module()
        runner.assert_runner_root_active()
        if not hasattr(runner, "validate_summary_table_revisions"):
            raise RuntimeError("Storage-return runner lacks the current table revision validator")
        journal_path = NATIVE_SCOPE_MIGRATION
        receipt = None
        if journal_path.exists():
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            _validate_migration_journal(journal)
            baseline = journal["baseline"]
            txid = journal["transaction_id"]
        else:
            baseline = _baseline_snapshot(runner)
            if NATIVE_SCOPE_RECEIPT.exists() or NATIVE_SCOPE_SUCCESSOR.exists():
                raise ValueError("Native scope migration target exists without its journal")
            completion = runner.COMPLETED / f"{STUDY}.json"
            with resource_lock(runner.COMPLETED, shared=True):
                chain = runner._completion_chain(STUDY)
                if (len(chain) != 1 or chain[0][2] !=
                        _current_migration_revision(runner, baseline["completion"])):
                    raise ValueError("Martinez generation zero is not the current completion head")
                if runner.completion_binding_path(STUDY, 1).exists():
                    raise ValueError("Martinez generation-one successor already exists")
                if not runner.received_study(STUDY, PREVIOUS_FROZEN_SCOPE, OUTPUT):
                    raise ValueError("Captured Martinez gen0 is not currently received")
            frozen = json.loads(PREVIOUS_FROZEN_SCOPE.read_text(encoding="utf-8"))
            summary = json.loads((OUTPUT / "processing_summary.json").read_text(encoding="utf-8"))
            tables = _current_revision_map(OUTPUT)
            runner.validate_summary_table_revisions(STUDY, OUTPUT)
            if (frozen.get("images_table_revision") != baseline["images"]
                    or tables["images.parquet"] != baseline["images"]):
                raise ValueError("Martinez frozen scope and current images table differ")
            if summary.get("nas_five_table_revisions", {}).get("after") != tables:
                raise ValueError("Martinez summary does not bind the captured five-table snapshot")
            receipt = collect_native_scope_receipt()
            txid = uuid.uuid4().hex
            migration_dir = NATIVE_SCOPE_RECEIPT.parent
            migration_dir.mkdir(parents=True, exist_ok=True)
            journal = {
                "schema": "cellweft.martinez_native_scope_migration.v1",
                "study": STUDY,
                "transaction_id": txid,
                "phase": "preparing",
                "baseline": {**baseline, "tables": tables},
                "backup_paths": {
                    "images": str(migration_dir / f".martinez-native-{txid}-before-images.parquet"),
                    "summary": str(migration_dir / f".martinez-native-{txid}-before-processing_summary.json"),
                    "review": str(migration_dir / f".martinez-native-{txid}-before-review_results.csv"),
                },
                "targets": {
                    "receipt": str(NATIVE_SCOPE_RECEIPT),
                    "images": str(OUTPUT / "images.parquet"),
                    "review": str(OUTPUT / "review_results.csv"),
                    "summary": str(OUTPUT / "processing_summary.json"),
                    "scope": str(NATIVE_SCOPE_SUCCESSOR),
                },
                "stage_paths": {
                    name: str(migration_dir / f".martinez-native-{txid}-{name}.partial")
                    for name in ("receipt", "images", "review", "summary", "scope")
                },
            }
            _write_migration_journal(journal_path, journal, txid, None)
            baseline = journal["baseline"]
        backup_revisions = _create_or_validate_migration_backups(journal)
        if journal.get("backup_revisions") != backup_revisions:
            journal["backup_revisions"] = backup_revisions
            journal_revision = file_revision(journal_path)
            _write_migration_journal(journal_path, journal, txid, journal_revision)
        _validate_migration_journal(journal)
        with resource_lock(runner.COMPLETED, shared=True):
            chain = runner._completion_chain(STUDY)
            if (not chain or chain[0][2] !=
                    _current_migration_revision(runner, baseline["completion"])
                    or len(chain) > 2):
                raise ValueError("Martinez completion chain no longer extends captured gen0")
        if file_revision(PREVIOUS_FROZEN_SCOPE) != baseline["scope"]:
            raise ValueError("Original Martinez frozen scope changed during migration")
        if journal.get("phase") == "successor_received":
            receipt = read_native_scope_receipt()
            _verify_migration_candidates(journal, receipt)
            if (file_revision(runner.completion_binding_path(STUDY, 1))
                    != journal.get("successor_revision")
                    or not runner.received_study(STUDY, NATIVE_SCOPE_SUCCESSOR, OUTPUT)):
                raise ValueError("Completed Martinez migration no longer reads as received")
            cores, _, _ = inventory()
            if any(core["core_id"] not in receipt["proofs"] for core in cores):
                raise ValueError("Current Martinez inventory references an unknown proof member")
            return journal
        if "target_revisions" not in journal:
            if (file_revision(PREVIOUS_FROZEN_SCOPE) != baseline["scope"]
                    or file_revision(OUTPUT / "processing_summary.json") != baseline["summary"]
                    or file_revision(OUTPUT / "images.parquet") != baseline["images"]
                    or _current_revision_map(OUTPUT) != baseline["tables"]):
                raise ValueError("Martinez migration changed before staged versions were journaled")
            frozen = json.loads(PREVIOUS_FROZEN_SCOPE.read_text(encoding="utf-8"))
            old_summary = json.loads(Path(journal["backup_paths"]["summary"]).read_text(
                encoding="utf-8"))
            old_revisions = baseline["tables"]
            old_images = pd.read_parquet(journal["backup_paths"]["images"])
            if receipt is None:
                receipt = collect_native_scope_receipt()
            expected_sources = journal.get("source_revisions", {})
            if (expected_sources.get("collection_report") is not None
                    and receipt["collection_report_revision"]
                    != expected_sources["collection_report"]):
                raise ValueError("Martinez collection report changed during migration recovery")
            if (expected_sources.get("metadata_bundle") is not None
                    and receipt["metadata_bundle_revision"] != expected_sources["metadata_bundle"]):
                raise ValueError("Martinez raw metadata bundle changed during migration recovery")
            if (expected_sources.get("proofs") is not None
                    and {core_id: entry["source_revision"]
                         for core_id, entry in receipt["proofs"].items()}
                    != expected_sources["proofs"]):
                raise ValueError("Martinez collector proof revisions changed during migration recovery")
            if ("unavailable_metadata_members" in expected_sources
                    and receipt.get("unavailable_metadata_members", [])
                    != expected_sources["unavailable_metadata_members"]):
                raise ValueError("Martinez unavailable metadata dependencies changed during recovery")
            _stage_migration_payloads(journal, receipt, frozen, old_images,
                                      old_revisions, old_summary)
            journal_revision = file_revision(journal_path)
            _write_migration_journal(journal_path, journal, txid, journal_revision)
            _validate_migration_journal(journal)
        if receipt is None:
            receipt = _migration_receipt_from_journal(journal)
        _verify_migration_candidates(journal, receipt)
        _publish_native_migration_files(journal)
        runner.validate_summary_table_revisions(STUDY, OUTPUT)
        if journal.get("phase") != "files_published":
            journal["phase"] = "files_published"
            journal_revision = file_revision(journal_path)
            _write_migration_journal(journal_path, journal, txid, journal_revision)
        new_frozen = json.loads(NATIVE_SCOPE_SUCCESSOR.read_text(encoding="utf-8"))

    expected_head = _current_migration_revision(
        runner, journal["baseline"]["completion"])
    migration = {
        "status": "metadata_only",
        "transaction_id": txid,
        "changed_files": (["images.parquet", "review_results.csv"]
                          if journal.get("active_d01_rebindings", 0)
                          else ["images.parquet"]),
        "changed_columns": ["images.parquet:split_evidence"] + (
            ["review_results.csv:details.input_binding"]
            if journal.get("active_d01_rebindings", 0) else []),
        "scientific_content_changed": False,
        "native_input_scope_receipt": journal["target_revisions"]["receipt"],
    }
    migration_receipt = {
        "schema": NATIVE_SCOPE_SCHEMA,
        "study": STUDY,
        "native_input_scope_receipt": journal["target_revisions"]["receipt"],
        "images_table_revision": journal["target_revisions"]["images"],
        "summary_revision": journal["target_revisions"]["summary"],
        "frozen_scope_revision": journal["target_revisions"]["scope"],
        "transaction_id": txid,
    }

    def prepublish_check() -> None:
        if file_revision(runner.COMPLETED / f"{STUDY}.json") != expected_head:
            raise ValueError("Martinez gen0 completion changed before successor publication")
        if file_revision(PREVIOUS_FROZEN_SCOPE) != journal["baseline"]["scope"]:
            raise ValueError("Original Martinez frozen scope changed before successor publication")
        if (file_revision(OUTPUT / "images.parquet") != journal["target_revisions"]["images"]
                or file_revision(OUTPUT / "processing_summary.json")
                != journal["target_revisions"]["summary"]
                or file_revision(NATIVE_SCOPE_SUCCESSOR) != journal["target_revisions"]["scope"]
                or file_revision(NATIVE_SCOPE_RECEIPT) != journal["target_revisions"]["receipt"]):
            raise ValueError("Martinez native migration members changed before successor publication")
        if journal.get("active_d01_rebindings", 0):
            if file_revision(OUTPUT / "review_results.csv") != journal["target_revisions"]["review"]:
                raise ValueError("Martinez D01 binding table changed before successor publication")
        runner.validate_summary_table_revisions(STUDY, OUTPUT)

    runner.complete_study(
        STUDY, NATIVE_SCOPE_SUCCESSOR, new_frozen, migration_receipt,
        completion_path=runner.completion_binding_path(STUDY, 1),
        expected_previous_revision=expected_head,
        migration=migration,
        prepublish_check=prepublish_check,
    )
    with formal_read_lease(NATIVE_SCOPE_RECEIPT):
        with resource_lock(OUTPUT, shared=True):
            with resource_lock(runner.COMPLETED, shared=True):
                if not runner.received_study(STUDY, NATIVE_SCOPE_SUCCESSOR, OUTPUT):
                    raise ValueError("Martinez successor completion did not read back as received")
                receipt = read_native_scope_receipt()
                images = pd.read_parquet(OUTPUT / "images.parquet")
                table_rows = images[images.study.astype(str) == STUDY]
                cores, _, _ = inventory()
                if (len(table_rows) != len(new_frozen["images"])
                        or table_rows.image_id.astype(str).tolist()
                        != journal["table_image_ids"]
                        or [row["image_id"] for row in new_frozen["images"]]
                        != journal["frozen_image_ids"]
                        or not set(table_rows.source_image_id.astype(str)).issubset(
                            set(receipt["proofs"]))
                        or any(core["core_id"] not in receipt["proofs"] for core in cores)):
                    raise ValueError("Martinez reader/table/receipt readback differs from the migration")
                for row in table_rows.itertuples():
                    _, separator, reference = str(row.split_evidence).partition(" and ")
                    if not separator:
                        raise ValueError(f"Martinez proof reference is absent: {row.image_id}")
                    core_id, proof = proof_from_reference(reference, receipt)
                    proof_record_id = proof["record"].get(
                        "core_id", proof["record"].get("directory_id"))
                    if core_id != str(row.source_image_id) or proof_record_id != core_id:
                        raise ValueError(f"Martinez proof reference identity differs: {row.image_id}")
                successor_revision = file_revision(
                    runner.completion_binding_path(STUDY, 1))
    journal["phase"] = "successor_received"
    journal["successor_revision"] = successor_revision
    journal["readback"] = {"received_study": True, "reader_inventory_count": len(cores),
                            "image_table_count": len(table_rows)}
    with formal_publish_lease(NATIVE_SCOPE_MIGRATION, directory=OUTPUT):
        journal_revision = file_revision(NATIVE_SCOPE_MIGRATION)
        _write_migration_journal(NATIVE_SCOPE_MIGRATION, journal, txid, journal_revision)
    return journal


def publish_json(path: Path, payload: dict) -> None:
    staged = Path("/tmp") / f"cellweft-{STUDY}-{os.getpid()}-progress.json"
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    shutil.copyfile(staged, temporary)
    os.replace(temporary, path)
    staged.unlink()


def checkpoint(stage: str, total: int, *, next_core: str = "", reason: str = "") -> None:
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv") if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    publish_json(JOB / "progress.json", {
        "study": STUDY, "phase": "prepare", "scientific_repair_round": 0,
        "stage": stage, "source_reported_adult_cores": total,
        "indexed_images": len(images), "sample_rows": len(samples),
        "sample_linked_images": int(images.sample_id.notna().sum()) if len(images) else 0,
        "patient_linked_samples": int(samples.patient_key.notna().sum()) if len(samples) else 0,
        "next_core": next_core, "reason": reason,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })


def fetal_samples() -> pd.DataFrame:
    """Resolve public section-folder IDs to the official donor and gestational-age table."""
    bundle_revision = file_revision(NATIVE_METADATA_BUNDLE)
    text = read_native_metadata_member(
        "README_fetal_spatial_proteomics_normal_pancreas.md").decode("utf-8")
    if file_revision(NATIVE_METADATA_BUNDLE) != bundle_revision:
        raise ValueError("Raw native metadata bundle changed while reading fetal sample layout")
    layout = text.split("## Slide Layout", 1)[1].split("## Tissue Exclusions", 1)[0]
    identities = re.findall(r"([A-Z]{4}) \((FPA\d+)\).*?([0-9.]+)w", layout)
    if len(identities) != 11 or len({item[1] for item in identities}) != 11:
        raise ValueError("Fetal README section-to-donor layout is incomplete")
    return pd.DataFrame([dict(study=STUDY, sample_id=f"fetal_{folder}",
                              patient_key=f"{STUDY}:{donor}", source_subject_id=donor,
                              source_sample_id=folder, cohort_id="fetal",
                              sample_type="pancreatic tissue section", tissue_type="normal",
                              anatomical_site="pancreas", preservation_method="ffpe",
                              case_control_status="reference", gestational_age_weeks=float(age),
                              metadata_notes="Official fetal slide layout links this section folder to one donor.")
                         for donor, folder, age in identities])


def inventory() -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Restrict input to native cores/sections with received collector proofs."""
    bundle_revision = file_revision(NATIVE_METADATA_BUNDLE)
    receipt = read_native_scope_receipt()
    if receipt["metadata_bundle_revision"] != bundle_revision:
        raise ValueError("Raw native metadata bundle changed since its received scope receipt")
    _, bundle_index = read_metadata_index(NATIVE_METADATA_BUNDLE)
    manifest = _native_metadata_json("native_manifest.json", index=bundle_index)
    ordered = _native_metadata_json("ordered_markers.json", index=bundle_index)
    accepted = set(receipt["collection_report"]["accepted_core_ids"])
    recovered = (_native_metadata_json("member_recovered_cores.json", index=bundle_index)["core_proofs"]
                 if f"public/{STUDY}/member_recovered_cores.json" in bundle_index else {})
    accepted.update(recovered)
    cores = [item for item in manifest["cores"] if item["core_id"] in accepted]
    cores_ready = []
    for item in cores:
        core_id = item["core_id"]
        archive = SOURCE / "archives" / item["archive"]
        if core_id not in recovered and (not archive.is_file() or archive.stat().st_size != next(
            row["size"] for row in manifest["archives"] if row["name"] == item["archive"]
        )):
            continue
        proof_entry = receipt["proofs"].get(core_id)
        if proof_entry is None:
            raise ValueError(f"Native collector proof absent from received scope: {core_id}")
        if core_id in recovered and proof_entry["source_revision"] != recovered[core_id]:
            raise ValueError(f"Recovered member proof version differs: {core_id}")
        proof = proof_entry["proof"]
        if not proof["all_channels_read"] or proof["record"] != item or ordered[core_id] != item["channels"]:
            raise ValueError(f"Collector identity/proof differs: {core_id}")
        paths = [SOURCE / channel["member"] for channel in item["channels"]]
        if core_id in recovered:
            evidence = proof["member_recovery"]
            if not evidence["qualified"] or evidence["complete_archive_received"]:
                raise ValueError(f"Recovered source qualification differs: {core_id}")
            versions = [file_revision(path) for path in paths]
            if versions != evidence["formal_source_versions"]:
                raise ValueError(f"Recovered source version differs: {core_id}")
        if any(not path.is_file() or path.stat().st_size != channel["size"]
               for path, channel in zip(paths, item["channels"], strict=True)):
            continue
        cores_ready.append(dict(item, paths=paths, shape=proof["shape"],
                                kind="adult", panel_id="adult_25", sample_id=None,
                                pixel_size_um=0.325,
                                proof_reference=native_scope_proof_reference(core_id)))
    if not cores_ready:
        if file_revision(NATIVE_METADATA_BUNDLE) != bundle_revision:
            raise ValueError("Raw native metadata bundle changed during inventory")
        return [], pd.DataFrame(), pd.DataFrame()
    names = [channel["marker"] for channel in cores_ready[0]["channels"]]
    if any([channel["marker"] for channel in item["channels"]] != names for item in cores_ready):
        raise ValueError("Adult source panel changed between cores")
    panels = [panel_from_s1(STUDY, "adult_25", names, S1)]
    fetal = _native_metadata_json("fetal_native_manifest.json", index=bundle_index)
    fetal_ordered = _native_metadata_json("fetal_ordered_markers.json", index=bundle_index)
    samples = fetal_samples()
    panel_orders: dict[tuple[str, ...], str] = {}
    for original in fetal["directories"]:
        directory_id = original["directory_id"]
        archive = SOURCE / "archives" / original["archive"]
        archive_size = next(row["size"] for row in fetal["archives"] if row["name"] == original["archive"])
        if not archive.is_file() or archive.stat().st_size != archive_size:
            continue
        proof_entry = receipt["proofs"].get(directory_id)
        if proof_entry is None:
            raise ValueError(f"Fetal collector proof absent from received scope: {directory_id}")
        proof = proof_entry["proof"]
        if not proof["all_channels_read"] or proof["record"] != original or fetal_ordered[directory_id] != original["channels"]:
            raise ValueError(f"Fetal collector identity/proof differs: {directory_id}")
        paths = [SOURCE / channel["member"] for channel in original["channels"]]
        if any(not path.is_file() or path.stat().st_size != channel["size"]
               for path, channel in zip(paths, original["channels"], strict=True)):
            continue
        names = tuple(channel["marker"] for channel in original["channels"])
        if names not in panel_orders:
            panel_id = f"fetal_{len(panel_orders) + 1:02d}"
            panel_orders[names] = panel_id
            panels.append(panel_from_s1(STUDY, panel_id, list(names), S1))
        folder = "FPA" + directory_id.split("/")[1]
        if folder not in set(samples.source_sample_id):
            raise ValueError(f"Fetal directory absent from official section layout: {directory_id}")
        cores_ready.append(dict(original, core_id=directory_id, paths=paths,
                                shape=proof["shape"], kind="fetal",
                                panel_id=panel_orders[names], sample_id=f"fetal_{folder}",
                                pixel_size_um=0.51,
                                proof_reference=native_scope_proof_reference(directory_id)))
    if file_revision(NATIVE_METADATA_BUNDLE) != bundle_revision:
        raise ValueError("Raw native metadata bundle changed during inventory")
    return cores_ready, pd.concat(panels, ignore_index=True), samples


def image_row(item: dict, result: dict) -> dict:
    core_id = item["core_id"]
    image_id = core_id.replace("/", "_")
    height, width = item["shape"]
    archive = item["archive"]
    manifest_name = "native_manifest.json" if item["kind"] == "adult" else "fetal_native_manifest.json"
    return dict(
        study=STUDY, image_id=image_id, sample_id=item["sample_id"], panel_id=item["panel_id"],
        image_kind="tma_core" if item["kind"] == "adult" else "roi", source_image_id=core_id,
        source_files=[path.relative_to(NAS).as_posix() for path in item["paths"]],
        source_format="tiff_planes", source_axes="YX per channel",
        source_dtype=result["source_dtype"], source_revision=f"S-BIAD3557/{archive}",
        slide_id=archive.removesuffix(".zip"), core_id=core_id if item["kind"] == "adult" else None,
        roi_id=core_id if item["kind"] == "fetal" else None,
        source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=result["height_px"], width_px=result["width_px"],
        source_channel_count=result["source_channel_count"],
        channel_count=result["channel_count"],
        pixel_size_x_um=item["pixel_size_um"], pixel_size_y_um=item["pixel_size_um"],
        input_image_path=(OUTPUT / 'images' / image_id / 'input.tiff').relative_to(NAS).as_posix(),
        intensity_description=("Published CODEX Processor TIFF intensities; no CellWeft input normalization"
                               if item["kind"] == "adult" else
                               "Published PhenoCycler-processed TIFF intensities; no CellWeft input normalization"),
        split_method="already_independent",
        split_evidence=f"{native_metadata_reference(manifest_name)} and {item['proof_reference']}",
        metadata_notes=("Physical core confirmed; parent biological taking unresolved; adult annotation identity evidence is recorded separately."
                        if item["kind"] == "adult" else "Official fetal section layout identifies one donor per directory."),
    )


def field_rows() -> pd.DataFrame:
    entries = [
        ("core_id", "image_metadata", "core_id", "Physical core locator from the adult native manifest."),
        ("channels.member", "image_metadata", "source_files", "Ordered per-core native TIFF paths; join on core_id."),
        ("sample/donor", "image_metadata", "sample_id", "Complete adult annotation matrix verified; case_region labels do not establish a unique parent biological taking. Adult sample connection remains unresolved."),
        ("donor", "sample_metadata", "patient_key", "Author README defines numeric case as donor identity. Matrix cell-key bridges and ID discrepancies are recorded separately; no adult patient values are filled without the biological sample connection."),
        ("fetal section layout", "sample_metadata", "sample_id", "Official fetal README joins FPA folder IDs to one donor per section."),
        ("fetal section layout", "sample_metadata", "gestational_age_weeks", "Published fetal age at collection, in weeks; read from the official section layout."),
        ("author annotation identity verification", "image_metadata", "sample_id", "Complete author ZIP and all 184 matrix cores verified. Exact ID/coordinate/area bridges and coordinate-only diagnostic overlaps remain distinct; parent taking/block map and HB relationships are unresolved. Reuse the recorded source evidence."),
    ]
    return pd.DataFrame([dict(study=STUDY, source_file=(native_metadata_reference("adult_annotation_identity_verification_20261028_v2.json")
                              if source_field == "author annotation identity verification" else native_metadata_reference("README_fetal_spatial_proteomics_normal_pancreas.md")
                              if source_field == "fetal section layout" else native_metadata_reference("README_adult_spatial_proteomics_normal_pancreas.md")
                              if target == "sample_metadata" else native_metadata_reference("native_manifest.json")),
                              source_sheet_or_table="fetal slide layout" if source_field == "fetal section layout" else "adult cores",
                              source_field=source_field, target_table=target, target_field=field,
                              join_keys="core_id" if target != "sample_metadata" else "unavailable",
                              value_mapping="", transformation_rule="", time_origin_rule="",
                              conflict_rule="Leave unlinked until an explicit mapping is found.", notes=note)
                         for source_field, target, field, note in entries])


def write_review(total: int, eligible: int, adult_ready: int, indexed: int, linked: int, patients: int) -> None:
    path = OUTPUT / "review_results.csv"
    rows = [
        dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample", record_id=STUDY,
             check_id="M01", result="unresolved", source_evidence=PUBLIC_RECORD,
             output_evidence=(OUTPUT / 'sample_metadata.csv').relative_to(NAS).as_posix(),
             details=f"Adult parent biological taking/block identity remains unresolved; complete annotation matrix and cell-key evidence are recorded separately. Fetal folder-to-donor mapping is explicit in official README. Exported {indexed}/{eligible} locally eligible images; {linked} image-to-sample and {patients} sample-to-patient links."),
        dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
             check_id="D01", result="unresolved", source_evidence=native_metadata_reference("native_manifest.json"),
             output_evidence=OUTPUT.relative_to(NAS).as_posix(),
             details=f"Preparer scope only: {indexed}/{eligible} locally qualified images exported; {adult_ready}/{total} adult cores locally verified. Independent review pending; remaining adult archives pending."),
    ]
    staging = Path("/tmp") / f"cellweft-{STUDY}-{os.getpid()}-review.csv"
    previous = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    if len(previous):
        own = ((previous.preparer == PREPARER) & (previous.reviewer == "") &
               previous.check_id.isin(["M01", "D01"]))
        previous = previous.loc[~own]
    result = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    result = result.reindex(columns=pd.read_csv(TEMPLATES / "review_results.csv").columns)
    result.to_csv(staging, index=False)
    with formal_publish_lease(path, directory=OUTPUT):
        temporary = path.with_suffix(".csv.partial")
        shutil.copyfile(staging, temporary)
        os.replace(temporary, path)
    staging.unlink()


def run(max_images: int | None) -> None:
    checkpoint("inventory", 184,
               reason="Recheck local manifests and collector proofs under one shared read lease")
    with formal_read_lease(NATIVE_SCOPE_RECEIPT):
        cores, panel, samples = inventory()
    total = 184
    if not cores:
        checkpoint("waiting_io", total, reason="No complete verified adult core presently available")
        return
    eligible = len(cores)
    per_image = max(len(panel[(panel.panel_id == item["panel_id"]) & panel.keep_channel])
                    * max(item["shape"]) ** 2 * 2 for item in cores)
    nas_free, local_free = shutil.disk_usage(NAS).free, shutil.disk_usage("/tmp").free
    if nas_free - per_image < 100 * GIB or local_free - per_image < 64 * GIB:
        checkpoint("waiting_io", total, next_core=cores[0]["core_id"],
                   reason=f"Space reservation: NAS free={nas_free}, local free={local_free}, estimated maximum per-image bytes={per_image}")
        return
    fields = field_rows()
    existing = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
    indexed = set(existing.image_id) if len(existing) else set()
    if len(existing):
        save_input_tables(OUTPUT, existing, samples, panel, fields)
    produced = 0
    for item in cores:
        core_id = item["core_id"]
        image_id = core_id.replace("/", "_")
        destination = OUTPUT / "images" / image_id / "input.tiff"
        needs_scale_rewrite = (image_id in indexed and
                               pd.isna(existing.loc[existing.image_id == image_id, "pixel_size_x_um"].iloc[0]))
        if image_id in indexed and not needs_scale_rewrite:
            if not destination.is_file():
                raise ValueError(f"Indexed image missing its input: {image_id}")
            continue
        if max_images is not None and produced >= max_images:
            break
        if shutil.disk_usage(NAS).free - per_image < 100 * GIB or shutil.disk_usage("/tmp").free - per_image < 64 * GIB:
            checkpoint("waiting_io", total, next_core=core_id, reason="Space reservation reached before next image")
            return
        current_panel = panel[panel.panel_id == item["panel_id"]]
        source = SourceSpec(image_id, item["panel_id"], "tiff_planes", item["paths"][0],
                            tuple(0 for _ in item["paths"]), paths=tuple(item["paths"]),
                            source_revision=f"S-BIAD3557/{item['archive']}",
                            pixel_size_um=(item["pixel_size_um"], item["pixel_size_um"]))
        checkpoint("exporting", total, next_core=core_id,
                   reason=(f"Native source export/publication; locally eligible={eligible}; "
                           f"estimated maximum image bytes={per_image}. "
                           "Reuse a closed helper checkpoint when present on I/O retry."))
        with ExitStack() as source_handles:
            with formal_read_lease(source.path):
                reader = source_handles.enter_context(open_source(source))
            if reader.shape[-2:] != tuple(item["shape"]):
                raise ValueError(f"Source shape differs from native proof: {core_id}")
            # Native planes fit in memory; one tile avoids repeatedly decoding
            # full-width source strips for each narrow output window.
            tile_size = ((max(item["shape"]) + 15) // 16) * 16
            result = export_input(reader, current_panel, destination, tile_size=tile_size)
        if not result["admitted"]:
            checkpoint("excluded_no_marker_signal", total, next_core=core_id,
                       reason="All admitted marker channels zero; investigate source scope")
            continue
        # export_input has reopened and checked the formal TIFF before return.
        save_input_tables(OUTPUT, pd.DataFrame([image_row(item, result)]), samples, panel, fields)
        indexed.add(image_id)
        produced += 1
        checkpoint("running", total, next_core=next((c["core_id"] for c in cores if c["core_id"].replace("/", "_") not in indexed), ""))
    final = pd.read_parquet(OUTPUT / "images.parquet")
    write_review(total, eligible, sum(item["kind"] == "adult" for item in cores),
                 len(final), int(final.sample_id.notna().sum()),
                 int(samples.patient_key.notna().sum()))
    checkpoint("prepared" if len(indexed) == eligible else "batch_complete", total,
               next_core=next((c["core_id"] for c in cores if c["core_id"].replace("/", "_") not in indexed), ""),
               reason="Independent review pending; remaining adult archives and adult core-to-sample bridge unresolved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--migrate-native-scope", action="store_true")
    args = parser.parse_args()
    if args.migrate_native_scope:
        if args.max_images is not None:
            parser.error("--migrate-native-scope does not accept --max-images")
        print(json.dumps(migrate_native_scope(), ensure_ascii=False))
    else:
        try:
            run(args.max_images)
        except KeyboardInterrupt:
            previous = json.loads((JOB / "progress.json").read_text()) if (JOB / "progress.json").exists() else {}
            checkpoint("waiting_io", 184, next_core=previous.get("next_core", ""),
                       reason=(f"Execution interrupted during {previous.get('stage', 'inventory')}; "
                               "handles closed. Resume the indexed next core; reuse only an existing "
                               "closed helper checkpoint, otherwise repeat its local export."))
