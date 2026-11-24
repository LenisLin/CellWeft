"""Prepare S-BIAD1507 native MIBI inputs from version-matched local proofs.

Inputs: official manifest, native proofs, current S1 and verified local patient
bridge. Outputs: input.tiff, input tables and resumable progress. Metadata-only
mode applies the independently audited exact cohort/FOV patient bridge.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative, metadata_reference_path

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

METADATA_ONLY = "--metadata-only" in sys.argv
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2" if METADATA_ONLY else "4"
if METADATA_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import SourceReader, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

pa.set_cpu_count(2 if METADATA_ONLY else 4)
pa.set_io_thread_count(2 if METADATA_ONLY else 4)
STUDY = "Ranek2026_MIBI_TNBC"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "source_verification"
SOURCE_METADATA = META / "source_attachments/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
DEFAULT_JOB = OUTPUT / "preparation"
GIB = 1024 ** 3
BRIDGE = CACHE / "verified_patient_bridge.json"
AUDIT_PROOF = CACHE / "independent_identity_audit_result.json"
PATIENT_BRIDGE = {}


def relative(path: Path) -> str:
    return metadata_source_relative(path, NAS).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def checkpoint(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_revision(path: Path) -> dict:
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino, device=stat.st_dev)


def technical_version(path: Path) -> dict:
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, inode=stat.st_ino)


def bridge_rows(path: Path) -> dict[tuple[str, str], dict]:
    """Index verified source identities without merging distinct physical cores."""
    proposal = json.loads(path.read_text())
    rows = {}
    for row in proposal["rows"]:
        key = (row["cohort"], row["source_fov_id"])
        subject = row["source_subject_id"]
        if (key in rows or not isinstance(subject, str) or not subject.isdecimal()
                or int(subject) <= 0 or str(int(subject)) != subject
                or row["patient_key"] != f"{STUDY}__{key[0]}__patient_{subject}"):
            raise ValueError("Invalid or duplicate verified patient bridge row")
        rows[key] = row
    return rows


def field_rules() -> pd.DataFrame:
    rules = [
        (relative(META / "source_file_manifest.json"), "component;source_fov_id;source_class;destination_rel", "image_metadata", "sample_id;source_files;source_image_id;core_id;region_annotation", "component+source_fov_id", "One complete independent source FOV is one image and one physical core-level sample; unresolved repeated control identities are excluded."),
        (relative(META / "source_file_manifest.json"), "component;source_fov_id;source_class", "sample_metadata", "sample_id;source_sample_id;biospecimen_group_id;cohort_id;sample_type;metadata_notes", "component+source_fov_id", "Use the exact physical core identity, retaining cohort and source class; distinct cores remain distinct samples."),
        (relative(META / "native_proofs"), "channels.member;regions;archive_stat", "panel_mapping", "source_channel_index;source_name", "component+source_fov_id", "Retain each proof's explicit TIFF-member order; require archive size, mtime, device and inode to match its native proof."),
        (relative(S1), "Original marker;Unified marker;Segmentation channel", "panel_mapping", "marker_id;keep_channel;role;nuclear;body", "Study+Original marker", "Apply current S1; histone modifications remain excluded and no nuclear channel is promoted."),
        (relative(SOURCE_METADATA / "supplementary_tables.xlsx"), "Supplementary Tables 3 and 4", "sample_metadata", "age_at_collection_years;tnm_t;grade", "cohort summaries only", "Clinical tables are cohort summaries; individual ages, TNM, grade and outcomes remain empty."),
        (relative(BRIDGE), "Zenodo 14290163 rev6 preprocessed H5AD obs; independently audited source Patient_ID", "sample_metadata", "patient_key;source_subject_id", "cohort_id+source_fov_id (exact fov_id/core_id and source archive)", "Use cohort-qualified exact patient_key and positive integral Patient_ID string only for verified FOVs; never infer from core position or cohort totals."),
    ]
    return pd.DataFrame([dict(study=STUDY, source_file=source, source_sheet_or_table="", source_field=field,
        target_table=table, target_field=target, join_keys=join, value_mapping="{}", transformation_rule=rule,
        time_origin_rule="No individual time values assigned", conflict_rule="Unknown individual values remain empty",
        notes="Coverage is reported in the metadata repair receipt; clinical summaries cannot assign individual values.")
        for source, field, table, target, join, rule in rules], columns=columns("field_mapping"))


def sample_for(record: dict) -> dict:
    key = f"{record['component']}__{record['source_fov_id']}"
    row = dict.fromkeys(columns("sample_metadata"), "")
    patient = PATIENT_BRIDGE.get((record["component"], record["source_fov_id"]))
    row.update(study=STUDY, sample_id=f"{STUDY}__core_{key}",
        source_sample_id=record["source_fov_id"], biospecimen_group_id=key,
        cohort_id=record["component"], sample_type="TMA core",
        patient_key=patient["patient_key"] if patient else "",
        source_subject_id=patient["source_subject_id"] if patient else "",
        metadata_notes=(f"Source class={record['source_class']}; exact published physical core identity. "
                        + ("Patient linked by exact cohort+source_fov_id obs bridge; " if patient else "Donor specimen and patient unlinked; ")
                        + "cohort-level clinical summaries not assigned to individuals."))
    return row


def repair_patient_metadata(job: Path, proposal_path: Path, audit_path: Path,
                            summary_path: Path) -> None:
    """Publish only patient identity fields supported by the independent obs audit."""
    proposal = json.loads(proposal_path.read_text())
    audit = json.loads(audit_path.read_text())
    summary = json.loads(summary_path.read_text())
    expected = summary["formal_table_versions"]
    before = {name: technical_version(OUTPUT / name) for name in expected}
    manifest_path = META / "source_file_manifest.json"
    manifest_version = technical_version(manifest_path)
    proposal_version = technical_version(proposal_path)
    record_path = SOURCE_METADATA / "zenodo_record.json"
    record_version = technical_version(record_path)

    def same_version(actual: dict, recorded: dict) -> bool:
        return all(actual[name] == recorded[name] for name in ("size", "mtime_ns"))

    if (audit["status"] != "pass" or audit["study"] != STUDY
            or audit["source_values_logged"] is not False
            or summary["source_values_logged"] is not False
            or proposal["study"] != STUDY
            or summary["current_images"] != 406
            or audit["exact_join_count"] != 406
            or audit["patient_count"] != 246
            or not same_version(proposal_version, summary["proposal"])
            or not same_version(proposal_version, audit["technical_versions"]["proposal"])
            or not same_version(manifest_version, summary["source_manifest"])
            or not same_version(manifest_version, audit["technical_versions"]["manifest"])
            or not same_version(record_version, audit["technical_versions"]["zenodo_record"])
            or any(not same_version(before[name], version) for name, version in expected.items())
            or not same_version(before["images.parquet"], audit["technical_versions"]["images"])
            or not same_version(before["sample_metadata.csv"], audit["technical_versions"]["samples"])):
        raise ValueError("Prewrite audit, proposal, manifest or formal table version mismatch")
    release = proposal["source_release"]
    if (release["id"] != 14290163 or release["revision"] != 6
            or audit["technical_versions"]["release_id"] != release["id"]
            or audit["technical_versions"]["release_revision"] != release["revision"]
            or any(release["files"][cohort]["size"] != size
                   for cohort, size in audit["technical_versions"]["source_h5ad_sizes"].items())):
        raise ValueError("Verified H5AD release identity mismatch")
    controller = json.loads((Path(os.environ["CELLWEFT_NAS_LOCK_ROOT"]) / "status.json").read_text())
    study_state = controller["studies"].get(STUDY, {})
    if (not study_state.get("terminal")
            or any(worker.get("study") == STUDY for worker in controller.get("workers", {}).values())):
        raise RuntimeError("Another study owner is active or controller state is not terminal")

    rows = bridge_rows(proposal_path)
    images = pd.read_parquet(OUTPUT / "images.parquet")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    if (len(images) != 406 or len(samples) != 406 or len(rows) != 406
            or samples.sample_id.nunique() != 406
            or images.image_id.nunique() != 406
            or samples.patient_key.ne("").any()
            or samples.source_subject_id.ne("").any()):
        raise ValueError("Current image/sample scope differs from reviewed bridge")
    manifest = json.loads(manifest_path.read_text())["files"]
    source_rows = {(row["component"], row["source_fov_id"]): row for row in manifest
                   if row["counted_as_independent_acquisition"]}
    sample_index = samples.set_index("sample_id", verify_integrity=True)
    used = set()
    for image in images.itertuples(index=False):
        key = (image.image_id.split("__", 1)[0], image.fov_id)
        source = source_rows.get(key)
        patient = rows.get(key)
        if (source is None or patient is None or key in used
                or image.core_id != image.fov_id
                or image.image_id != f"{key[0]}__{key[1]}"
                or list(image.source_files) != [source["destination_rel"]]
                or image.sample_id != patient["sample_id"]
                or patient["image_id"] != image.image_id
                or image.sample_id not in sample_index.index
                or sample_index.at[image.sample_id, "cohort_id"] != key[0]
                or sample_index.at[image.sample_id, "source_sample_id"] != key[1]
                or sample_index.at[image.sample_id, "biospecimen_group_id"] != image.image_id):
            raise ValueError("Exact cohort/FOV/core/archive/sample patient join failed")
        used.add(key)
    if len(used) != len(rows) or len({row["patient_key"] for row in rows.values()}) != 246:
        raise ValueError("Verified patient coverage differs from independent audit")

    # Freeze only technical versions and the two small CSVs before any formal write.
    snapshot_dir = job / "prewrite_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sample_metadata.csv", "field_mapping.csv"):
        shutil.copy2(OUTPUT / name, snapshot_dir / name)
    input_versions = {}
    for image in images.itertuples(index=False):
        directory = OUTPUT / "images" / image.image_id
        input_versions[image.image_id] = {path.name: technical_version(path)
                                          for path in directory.iterdir() if path.is_file()
                                          and path.name in {"input.tiff", "valid_pixels.tiff", "sample_region.tiff"}}
        if "input.tiff" not in input_versions[image.image_id]:
            raise ValueError("Current input TIFF is missing")
    checkpoint(snapshot_dir / "versions.json", dict(formal_tables=before,
        source_manifest=manifest_version, zenodo_record=record_version,
        proposal=proposal_version, independent_audit=technical_version(audit_path),
        exact_join_summary=technical_version(summary_path), input_files=input_versions))

    for index, sample in samples.iterrows():
        key = (sample["cohort_id"], sample["source_sample_id"])
        patient = rows[key]
        note = sample["metadata_notes"]
        old = "Donor specimen and patient unlinked; "
        if old not in note:
            raise ValueError("Current sample note differs from reviewed baseline")
        samples.at[index, "patient_key"] = patient["patient_key"]
        samples.at[index, "source_subject_id"] = patient["source_subject_id"]
        samples.at[index, "metadata_notes"] = note.replace(
            old, "Patient linked by exact cohort+source_fov_id obs bridge; ")
    fields = field_rules()
    staged_samples = job / "sample_metadata.csv"
    staged_fields = job / "field_mapping.csv"
    samples.to_csv(staged_samples, index=False, encoding="utf-8-sig")
    fields.to_csv(staged_fields, index=False, encoding="utf-8-sig")
    original = pd.read_csv(snapshot_dir / "sample_metadata.csv", dtype=str, keep_default_na=False)
    protected = [name for name in samples if name not in
                 {"patient_key", "source_subject_id", "metadata_notes"}]
    if (not samples[protected].equals(original[protected])
            or not pd.read_csv(staged_samples, dtype=str, keep_default_na=False)[protected].equals(original[protected])):
        raise ValueError("An unapproved sample cell changed during CSV staging")

    # Cache the reviewed evidence under the shared formal NAS publication protocol.
    for source, destination in ((proposal_path, BRIDGE), (audit_path, AUDIT_PROOF)):
        with formal_publish_lease(destination, directory=CACHE):
            partial = destination.with_name(destination.name + ".partial")
            shutil.copyfile(source, partial)
            partial.replace(destination)
    if bridge_rows(BRIDGE) != rows:
        raise ValueError("Persisted patient bridge differs from reviewed proposal")
    with formal_publish_lease(OUTPUT / "sample_metadata.csv", directory=OUTPUT):
        if any(technical_version(OUTPUT / name) != before[name] for name in expected):
            raise ValueError("Formal table changed before metadata publication")
        for name, staged in (("sample_metadata.csv", staged_samples),
                             ("field_mapping.csv", staged_fields)):
            partial = OUTPUT / (name + ".metadata_repair.partial")
            shutil.copyfile(staged, partial)
            partial.replace(OUTPUT / name)

    after = {name: technical_version(OUTPUT / name) for name in expected}
    current = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    if (not current.equals(samples) or not current[protected].equals(original[protected])
            or current.patient_key.nunique() != 246
            or current.patient_key.eq("").any()
            or current.source_subject_id.eq("").any()
            or any(after[name] != before[name] for name in
                   ("images.parquet", "panel_mapping.csv", "review_results.csv"))
            or any({path.name: technical_version(path) for path in (OUTPUT / "images" / image_id).iterdir()
                    if path.is_file() and path.name in versions} != versions
                   for image_id, versions in input_versions.items())):
        raise ValueError("Post-publication patient mapping or preserved file version mismatch")
    receipt = dict(study=STUDY, scientific_repair=1, independent_recheck="pending",
        formal_two_tables_written=True, D01=False, source_values_logged=False,
        modified_files=[str(Path(__file__).resolve()), str(BRIDGE), str(AUDIT_PROOF),
                        str(OUTPUT / "sample_metadata.csv"), str(OUTPUT / "field_mapping.csv")],
        source_release=release, independent_audit_version=technical_version(audit_path),
        source_versions=dict(manifest=manifest_version, zenodo_record=record_version,
                             proposal=proposal_version), formal_before=before, formal_after=after,
        coverage=dict(images=406, samples=406, patient_linked_samples=406, patient_keys=246,
                      exact_image_sample_joins=406, exact_proposal_values=406),
        preserved=dict(other_sample_cells=True, images_parquet=True, panel_mapping=True,
                       review_results=True, input_and_companions=len(input_versions),
                       image_metadata_notes=True),
        command_returncode=0)
    checkpoint(job / "metadata_repair_receipt.json", receipt)
    checkpoint(job / "progress.json", dict(study=STUDY, phase="metadata_repair",
        scientific_repair_round=1, status="prepared_for_independent_recheck",
        indexed_images=406, sample_rows=406, patient_linked_samples=406,
        unique_patient_keys=246, independent_review="pending", D01=False))
    print(json.dumps(dict(status="metadata_repair_complete", images=406,
                          patient_linked_samples=406, patient_keys=246)))


def publish_reviews(images: pd.DataFrame, samples: pd.DataFrame, expected: int, job: str) -> None:
    patient_linked_samples = int(samples.patient_key.fillna("").ne("").sum())
    details = {
        "S01": f"{len(images)}/{expected} version-matched unambiguous local FOV inputs; complete official source has 1046 ZIPs, 16 identity-deferred. In-flight sources unchanged.",
        "I01": "Image IDs and source members follow exact cohort+FOV keys.",
        "I02": f"{len(images)}/{len(images)} images linked to physical-core samples.",
        "I03": "Full native FOV; centered square padding; no resampling.",
        "I04": "Source core labels retained; individual donor specimen bridge unresolved.",
        "P01": "Proof member order and exact current S1 labels used; actual panel variants kept separately.",
        "P02": "Current S1 excludes H3K27me3/H3K9ac. No selected nuclear channel; downstream segmentation gap retained.",
        "P03": "All newly published OME heads, dtype, dimensions and ordered names reopened.",
        "I05": "Preparer compared all retained channels in center and boundary 32x32 windows for each new image; selfcheck.json contains coordinates. Independent review pending.",
        "M01": f"{len(samples)} physical-core samples; {len(images)} sample-linked images; {patient_linked_samples} patient-linked samples. Exact verified cohort/FOV patient bridge used where available; unverified individual values remain empty.",
        "M02": "Individual clinical fields remain empty; supplementary clinical tables provide only cohort summaries.",
        "D01": f"Preparer delivery {len(images)} images; independent review and acceptance binding pending.",
    }
    frame = pd.DataFrame([dict(study=STUDY, preparer=job, reviewer="", scope="study", record_id=STUDY,
        check_id=check, result="unresolved", source_evidence=relative(META)+";"+relative(CACHE),
        output_evidence=relative(OUTPUT), details="Preparer self-check only. "+detail)
        for check, detail in details.items()], columns=columns("review_results"))
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        temporary = OUTPUT / "review_results.csv.partial"
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(OUTPUT / "review_results.csv")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--bridge-proposal", type=Path)
parser.add_argument("--identity-audit", type=Path)
parser.add_argument("--exact-join-summary", type=Path)
args = parser.parse_args()
args.job.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
if args.metadata_only:
    if not all((args.bridge_proposal, args.identity_audit, args.exact_join_summary)):
        parser.error("--metadata-only requires proposal, audit and exact join summary")
    repair_patient_metadata(args.job, args.bridge_proposal, args.identity_audit,
                            args.exact_join_summary)
    sys.exit(0)
if BRIDGE.is_file():
    PATIENT_BRIDGE = bridge_rows(BRIDGE)
staging = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
staging.mkdir(parents=True, exist_ok=True)

# Freeze only complete, version-matched and unambiguous local sources.
snapshot = args.job / "source_snapshot.json"
if snapshot.is_file():
    scope = json.loads(snapshot.read_text())
else:
    manifest = json.loads((META / "source_file_manifest.json").read_text())["files"]
    proofs = {p["archive_rel"]: (path, p) for path in (META / "native_proofs").rglob("*.json")
              for p in [json.loads(path.read_text())]}
    selected, pending = [], []
    for record in manifest:
        path = NAS / record["destination_rel"]
        if not record["counted_as_independent_acquisition"]:
            pending.append(dict(source=record["destination_rel"], reason="source identity deferred"))
        elif record["destination_rel"] not in proofs:
            pending.append(dict(source=record["destination_rel"], reason="no complete native proof"))
        else:
            proof_path, proof = proofs[record["destination_rel"]]
            if (not proof["actual_pixel_read"] or proof["fixture"] or not path.is_file()
                    or source_revision(path) != proof["archive_stat"]):
                pending.append(dict(source=record["destination_rel"], reason="native proof/file version mismatch"))
                continue
            selected.append(dict(record=record, proof_path=relative(proof_path), archive_stat=proof["archive_stat"]))
    scope = dict(selected=selected, pending=pending, official_archives=len(manifest))
    checkpoint(snapshot, scope)
records = scope["selected"]
previous = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
image_rows = {row["image_id"]: row for row in previous.to_dict("records")}
panels, signatures = {}, {}
for entry in records:
    proof = json.loads(metadata_reference_path(NAS, entry["proof_path"]).read_text())
    names = tuple(c["source_label"] for c in proof["channels"])
    signature = (proof["component"], names)
    if signature not in signatures:
        panel_id = f"{STUDY}__{proof['component']}_{sum(k[0] == proof['component'] for k in signatures)+1:03d}"
        signatures[signature] = panel_id
        panels[panel_id] = panel_from_s1(STUDY, panel_id, list(names), S1)
    entry["panel_id"] = signatures[signature]
fields = field_rules()
selfcheck_path = args.job / "selfcheck.json"
selfchecks = json.loads(selfcheck_path.read_text()) if selfcheck_path.is_file() else []
excluded = []
state = dict(study=STUDY, phase="prepare", scientific_repair_round=0, expected_images=len(records),
    indexed_images=len(image_rows), sample_rows=len(image_rows), sample_linked_images=len(image_rows),
    patient_linked_samples=sum(bool(PATIENT_BRIDGE.get((row["image_id"].split("__", 1)[0], row["fov_id"])))
                               for row in image_rows.values()))

# Export each image independently; helpers own formal publication locks.
for position, entry in enumerate(records):
    record = entry["record"]
    image_id = f"{record['component']}__{record['source_fov_id']}"
    source = NAS / record["destination_rel"]
    revision = json.dumps(entry["archive_stat"], sort_keys=True)
    if (image_id in image_rows and image_rows[image_id]["source_revision"] == revision
            and (NAS / image_rows[image_id]["input_image_path"]).is_file()):
        continue
    proof = json.loads(metadata_reference_path(NAS, entry["proof_path"]).read_text())
    panel = panels[entry["panel_id"]]
    shape = tuple(proof["regions"][0]["shape"])
    if {tuple(r["shape"]) for r in proof["regions"]} != {shape}:
        raise ValueError(f"Inconsistent native geometry: {image_id}")
    estimate = max(shape)**2 * 4 * int(panel.keep_channel.sum())
    state.update(next_source_position=position, next_image_id=image_id,
                 estimated_image_bytes=estimate, nas_free_bytes=shutil.disk_usage(NAS).free,
                 local_free_bytes=shutil.disk_usage(staging).free)
    if state["nas_free_bytes"] < 100*GIB + estimate or state["local_free_bytes"] < 64*GIB + 3*estimate + source.stat().st_size:
        state.update(status="waiting_io", waiting_reason="NAS 100 GiB / local 64 GiB reserve")
        checkpoint(args.job / "progress.json", state)
        break
    state.update(status="exporting", updated_at=datetime.now(timezone.utc).isoformat())
    checkpoint(args.job / "progress.json", state)
    destination = OUTPUT / "images" / image_id / "input.tiff"
    with tempfile.TemporaryDirectory(prefix=image_id+"-", dir=staging) as local:
        local = Path(local)
        archive = local / "source.zip"
        with formal_read_lease(source):
            if source_revision(source) != entry["archive_stat"]:
                raise ValueError(f"Native source changed: {image_id}")
            shutil.copyfile(source, archive)
            if source_revision(source) != entry["archive_stat"]:
                raise ValueError(f"Native source changed during staging: {image_id}")
        paths = []
        readable_indices = []
        with zipfile.ZipFile(archive) as z:
            for index, channel in enumerate(proof["channels"]):
                # Some published ZIPs contain an empty excluded instrument plane.
                # Keep its original panel row, but never present it as a TIFF.
                if channel["member_bytes"] == 0:
                    if bool(panel.loc[panel.source_channel_index == index, "keep_channel"].iloc[0]):
                        raise ValueError(f"Empty retained source channel: {image_id}/{index}")
                    continue
                path = local / f"channel_{index:03d}.tiff"
                with z.open(channel["member"]) as incoming, path.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if path.stat().st_size != channel["member_bytes"]:
                    raise ValueError(f"Member size differs: {image_id}/{index}")
                paths.append(path)
                readable_indices.append(index)
        reader_panel = panel.loc[panel.source_channel_index.isin(readable_indices)].copy()
        reader_panel["source_channel_index"] = range(len(readable_indices))
        spec = SourceSpec(image_id, entry["panel_id"], "tiff_planes", paths[0],
                          (0,)*len(paths), paths=tuple(paths))
        with ExitStack() as stack:
            reader = SourceReader(spec, stack)
            result = export_input(reader, reader_panel, destination)
            if result["admitted"]:
                selected_panel = reader_panel.loc[reader_panel.keep_channel].sort_values("output_channel_index")
                windows = [(0, 0, 32, 32), (shape[1]//2-16, shape[0]//2-16, shape[1]//2+16, shape[0]//2+16)]
                with formal_read_lease(destination), tifffile.TiffFile(destination) as output:
                    names = [node.attrib["Name"] for node in ET.fromstring(output.ome_metadata).iter() if node.tag.endswith("}Channel")]
                    assert names == selected_panel.marker_id.tolist()
                    assert len(output.pages) == len(selected_panel)
                    assert all(page.shape == (max(shape), max(shape)) and page.dtype == reader.dtype for page in output.pages)
                    for window in windows:
                        native = reader.read_window(selected_panel.source_channel_index.astype(int).tolist(), window)
                        x0,y0,x1,y1 = window
                        target_window = (x0+result['pad_left_px'],y0+result['pad_top_px'],x1+result['pad_left_px'],y1+result['pad_top_px'])
                        actual = np.stack([read_tiff_window(output, i, target_window) for i in range(len(selected_panel))])
                        assert actual.dtype == native.dtype and np.array_equal(actual, native)
                selfchecks.append(dict(image_id=image_id, windows=windows, channels=len(selected_panel), dtype=str(reader.dtype), equal=True))
    if result["admitted"]:
        row = dict.fromkeys(columns("image_metadata"), None)
        row.update(study=STUDY, image_id=image_id, sample_id=sample_for(record)["sample_id"], panel_id=entry["panel_id"],
            image_kind="fov", source_image_id=record["source_member_path"]+"::"+proof["native_folder"],
            source_files=[record["destination_rel"]], source_format="ZIP of scalar TIFF planes", source_axes="per-channel YX",
            source_dtype="float32", source_revision=revision, core_id=record["source_fov_id"], fov_id=record["source_fov_id"],
            region_annotation=record["source_class"]+";"+record["source_fov_id"],
            source_x0_px=0, source_y0_px=0, source_x1_px=shape[1], source_y1_px=shape[0],
            height_px=result["height_px"], width_px=result["width_px"],
            source_channel_count=len(proof["channels"]), channel_count=result["channel_count"],
            input_image_path=relative(destination), split_method="already_independent",
            split_evidence="Official independent FOV ZIP and version-matched native proof", intensity_description="Published raw float32 MIBI intensities; no rescaling or normalization",
            metadata_notes="Physical core sample identity only; patient unknown. Pixel size not established. S1 has no selected nuclear input.")
        for key in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"):
            row[key] = result[key]
        image_rows[image_id] = row
    else:
        excluded.append(image_id)
    images = pd.DataFrame(image_rows.values(), columns=columns("image_metadata"))
    samples = pd.DataFrame([sample_for(e["record"]) for e in records
        if f"{e['record']['component']}__{e['record']['source_fov_id']}" in image_rows], columns=columns("sample_metadata"))
    panel_rows = pd.concat([p for key,p in panels.items() if key in set(images.panel_id)], ignore_index=True)
    save_input_tables(OUTPUT, images, samples, panel_rows, fields)
    checkpoint(selfcheck_path, selfchecks)
    publish_reviews(images, samples, len(records), args.job.name)
    next_id = (f"{records[position+1]['record']['component']}__{records[position+1]['record']['source_fov_id']}"
               if position+1 < len(records) else "")
    state.update(indexed_images=len(images), sample_rows=len(samples), sample_linked_images=len(images),
        patient_linked_samples=int(samples.patient_key.fillna("").ne("").sum()),
        next_source_position=position+1, next_image_id=next_id,
        last_completed_image=image_id, excluded_zero_signal=excluded)
    queue = Path(os.environ["CELLWEFT_NAS_LOCK_ROOT"]) / ".io_waiters/nas"
    waiting = sum(1 for p in queue.glob("*") if p.is_file() and not p.name.endswith(".partial"))
    if waiting and position < len(records)-1:
        state.update(status="waiting_io", waiting_reason=f"Shared NAS queue has {waiting} requests after a fully closed image and five tables")
        checkpoint(args.job / "progress.json", state)
        break
    checkpoint(args.job / "progress.json", state)
    print(json.dumps(dict(image_id=image_id, indexed_images=len(images))), flush=True)
else:
    state.update(status="prepared", next_source_position=len(records), independent_review="pending")
    checkpoint(args.job / "progress.json", state)
print(json.dumps(state), flush=True)
