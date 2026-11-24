"""Export native MIBI intensities from the S-BIAD1579 per-FOV ZIP release.

Inputs: local source manifest and channel proofs, S1, and public BRUCE identity
tables. Outputs: per-image input.tiff and the four study input tables. The
BRUCE FOV bridge identifies an explicit subset; other TMA cores remain separate
core-level samples with unknown patient identity.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import csv
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

from cellweft.data.contracts import SourceSpec
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Piyadasa2025_MIBI_Glioma"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
MANIFEST = META / "source_manifest.json"
LEGACY_BRIDGE = NAS / "raw/public" / STUDY / "metadata/20241213_vitesscesamples.csv"
BRIDGE = META / "source_attachments/metadata_complete.csv"
PATIENTS = META / "source_attachments/bruce_patients_20261024.json"
SELF_CHECK = CACHE / "pixel_selfcheck_0340.json"
DEFAULT_JOB = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0340_prepare_0"
GIB = 1024 ** 3


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return metadata_source_relative(path, NAS).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def checkpoint(path: Path, *, scientific_repair_round: int = 0, **details: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(study=STUDY, phase="prepare", scientific_repair_round=scientific_repair_round,
                   updated_at=datetime.now(timezone.utc).isoformat(), **details)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def queued_io_requests() -> int:
    queue = Path(os.environ["CELLWEFT_NAS_LOCK_ROOT"]) / ".io_waiters/nas"
    return sum(1 for path in queue.glob("*") if path.is_file() and not path.name.endswith(".partial"))


class ZipPlanes:
    """Read one native TIFF plane at a time from a locally staged FOV ZIP."""

    def __init__(self, path: Path, mapping: dict, panel_id: str, shape: tuple[int, int]):
        self.source = SourceSpec(mapping["fov"], panel_id, "tiff_planes", path,
                                 tuple(range(len(mapping["channels"]))))
        self.archive = zipfile.ZipFile(path)
        self.channels = mapping["channels"]
        self.shape = (len(self.channels), *shape)
        self.dtype = np.dtype("float32")
        self.loaded_index = -1
        self.loaded = None
        try:
            members = {item.filename: item for item in self.archive.infolist()}
            for channel in self.channels:
                info = members[channel["member"]]
                if info.file_size != channel["member_bytes"]:
                    raise ValueError(f"ZIP member size changed: {path.name}/{info.filename}")
        except Exception:
            self.archive.close()
            raise

    def close(self) -> None:
        self.archive.close()

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        planes = []
        for index in channels:
            if index != self.loaded_index:
                payload = self.archive.read(self.channels[index]["member"])
                image = tifffile.imread(BytesIO(payload))
                if image.shape != self.shape[1:] or image.dtype != self.dtype:
                    raise ValueError(f"Native TIFF shape/dtype changed: {self.source.image_id}/{index}")
                self.loaded_index, self.loaded = index, image
            planes.append(self.loaded[y0:y1, x0:x1])
        return np.stack(planes)

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def source_scope() -> list[dict]:
    records = json.loads(MANIFEST.read_text())
    assert len(records) == 1531 and len({(r["panel"], r["fov"]) for r in records}) == 1531
    return records


def load_mapping(record: dict) -> dict:
    key = record["panel"], record["fov"]
    proof = META / "channel_mappings" / f"{key[0]}__{key[1]}.json"
    mapping = json.loads(proof.read_text())
    if (mapping["panel"], mapping["fov"]) != key or mapping["archive_path"] != record["local_path"]:
        raise ValueError(f"Source mapping identity differs: {key}")
    return mapping


def native_shape(mapping: dict, panel: pd.DataFrame) -> tuple[int, int]:
    retained = [mapping["channels"][index]
                for index in panel.loc[panel.keep_channel, "source_channel_index"]]
    shapes = {tuple(channel["shape"]) for channel in retained if "shape" in channel}
    dtypes = {channel["dtype"] for channel in retained if "dtype" in channel}
    if len(shapes) != 1 or dtypes != {"float32"}:
        raise ValueError(f"Retained source planes have different shape/dtype: {mapping['fov']}")
    return next(iter(shapes))


def identity_bridge() -> tuple[dict[tuple[str, str], str], dict[str, dict]]:
    with BRIDGE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    bridge = {(panel, row[field]): row["sample_id"] for row in rows
              for panel, field in (("immune", "fov_immune"), ("tumor", "fov_tumor"))}
    if len(bridge) != 2 * len(rows) or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Public FOV-to-sample bridge has duplicate keys")
    patients = json.loads(PATIENTS.read_text())
    by_sample = {sample: patient for patient in patients for sample in patient["samples"]}
    if len(by_sample) != sum(len(patient["samples"]) for patient in patients):
        raise ValueError("Public sample-to-patient bridge has conflicting keys")
    if any(row["sample_id"] not in by_sample or
           str(by_sample[row["sample_id"]]["patient_id"]) != row["patient_id"] for row in rows):
        raise ValueError("Public metadata and patient API disagree")
    sex_values: dict[str, set[str]] = {}
    for row in rows:
        sex_values.setdefault(row["patient_id"], set()).add(row["sex"])
    conflicted = {patient for patient, values in sex_values.items() if {"M", "F"} <= values}
    for row in rows:
        patient = dict(by_sample[row["sample_id"]])
        if row["patient_id"] in conflicted:
            patient["sex"] = ""
            patient["sex_source_conflict"] = True
        elif patient["sex"] != row["sex"]:
            raise ValueError("Public sample and patient sex values disagree")
        by_sample[row["sample_id"]] = patient
    with LEGACY_BRIDGE.open(newline="", encoding="utf-8") as handle:
        if any(bridge.get((row["panel"], row["sampleID"])) != row["sample"]
               for row in csv.DictReader(handle)):
            raise ValueError("Published identity bridge conflicts with prior exact links")
    return bridge, by_sample


def sample_for(record: dict, bridge: dict, patients: dict) -> dict:
    source_sample = bridge.get((record["panel"], record["fov"]), "")
    core = record["fov"]
    sample_id = (f"{STUDY}__sample_{source_sample}" if source_sample
                 else f"{STUDY}__core_{core}")
    patient = patients.get(source_sample, {})
    row = dict.fromkeys(columns("sample_metadata"), "")
    sex = {"M": "male", "F": "female"}.get(patient.get("sex"), "")
    row.update(study=STUDY, sample_id=sample_id,
               patient_key=f"{STUDY}__patient_{patient['patient_id']}" if patient else "",
               source_subject_id=str(patient.get("patient_id", "")),
               source_sample_id=source_sample, biospecimen_group_id=core if not source_sample else "",
               sample_type="archival FFPE TMA core" if not source_sample else "archival FFPE tissue",
               preservation_method="FFPE", sex_reported=sex,
               metadata_notes=("Public BRUCE FOV/sample and sample/patient bridges." if source_sample else
                               "Core-level identity from the published FOV label; donor specimen and patient unlinked."))
    if patient.get("sex_source_conflict"):
        row["metadata_notes"] += (
            " Published samples under this author patient key report conflicting M/F values; "
            "sex is withheld and sex-dependent clinical analysis remains pending. "
            "Evidence: raw/public/Piyadasa2025_MIBI_Glioma/manifest.json#source_metadata_conflicts.")
    return row


def image_row(record: dict, mapping: dict, result: dict, revision: str,
              sample_id: str, destination: Path) -> dict:
    height = result["height_px"] - result["pad_top_px"] - result["pad_bottom_px"]
    width = result["width_px"] - result["pad_left_px"] - result["pad_right_px"]
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=f"{record['panel']}__{record['fov']}",
               sample_id=sample_id, panel_id=record["panel_id"], image_kind="fov",
               source_image_id=record["fov"], source_files=[record["local_path"]],
               source_format="ZIP of scalar TIFF planes", source_axes="per-channel YX",
               source_dtype="float32", source_revision=revision,
               core_id=record["fov"], fov_id=record["fov"],
               region_annotation=record["fov"],
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=result["height_px"], width_px=result["width_px"],
               source_channel_count=len(mapping["channels"]), channel_count=result["channel_count"],
               input_image_path=relative(destination),
               intensity_description="Published raw float32 MIBI channel intensities; no rescaling",
               split_method="already_independent", split_evidence="Published independent FOV ZIP",
               metadata_notes="FOV/core label retained verbatim; pixel scale not established in source metadata.")
    return row


def field_rules() -> pd.DataFrame:
    rules = [
        (relative(MANIFEST), "panel;fov;local_path;bytes", "image_metadata",
         "source_image_id;source_files;core_id;fov_id", "panel+fov", "One published FOV ZIP is one image; its FOV label identifies a TMA core."),
        (relative(META / "channel_mappings"), "channels.member;source_channel_label;shape;dtype", "panel_mapping",
         "source_name;source_channel_index", "panel+fov", "Use the proven ordered native TIFF members; group identical orders into a panel_id."),
        (relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
         "marker_id;role;keep_channel;nuclear;body", "Study+Original marker", "Apply current S1 channel and segmentation selection."),
        (relative(BRIDGE), "fov_immune;fov_tumor;sample_id", "image_metadata",
         "sample_id", "immune+fov=fov_immune;tumor+fov=fov_tumor",
         "Join the full FOV label in the matching panel column to the explicit biological sample_id; unmatched core identity remains unchanged."),
        (relative(BRIDGE), "sample_id;patient_id", "sample_metadata",
         "source_sample_id;source_subject_id;patient_key", "source_sample_id=sample_id",
         "Use the explicit sample-to-patient key, checked against the public patient API; do not infer participants from FOV labels."),
        (relative(PATIENTS), "patient_id;samples;sex", "sample_metadata",
         "patient_key;source_subject_id;sex_reported", "source_sample_id in samples", "Copy patient identity and sex only for exact linked source samples; unknown sex stays empty."),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC11952471/", "Methods;BRUCE cohort", "sample_metadata",
         "sample_type;preservation_method", "study-level methods", "Cohort and control TMA tissue is archival FFPE; source FOVs are MIBI core scans."),
    ]
    rows = [dict(study=STUDY, source_file=source, source_sheet_or_table="",
                 source_field=field, target_table=table, target_field=target,
                 join_keys=join, value_mapping="{}", transformation_rule=rule,
                 time_origin_rule="", conflict_rule=(
                     "Leave unconfirmed identity and clinical values empty. If source samples under one "
                     "author patient key contain both M and F, leave sex blank for all its samples, "
                     "retain the reported identity and original evidence, and keep M02 pending."),
                 notes=("The complete published BRUCE metadata supplies direct FOV/sample/patient identity; "
                        "the older Vitessce demonstration subset is checked for agreement. "
                        "Unmatched FOVs retain technical core identity and an explicit biological identity gap. "
                        "Patient API any_value clinical summaries are not copied to individual samples."))
            for source, field, table, target, join, rule in rules]
    return pd.DataFrame(rows).reindex(columns=columns("field_mapping"))


def inspect_output(path: Path, panel: pd.DataFrame, side: int) -> bool:
    if not path.is_file():
        return False
    selected = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    with formal_read_lease(path), tifffile.TiffFile(path) as tif:
        if len(tif.pages) != len(selected) or tif.pages[0].shape != (side, side) or tif.pages[0].dtype != np.dtype("float32"):
            return False
        root = ET.fromstring(tif.ome_metadata)
        names = [node.attrib["Name"] for node in root.iter() if node.tag.endswith("}Channel")]
        return names == selected.marker_id.tolist()


def save_reviews(path: Path, n: int, samples: pd.DataFrame, linked_images: int,
                 checked: list[dict], expected: int, preparer: str) -> None:
    known_patients = int(samples.patient_key.astype(str).ne("").sum())
    source = ";".join([relative(MANIFEST), relative(BRIDGE), relative(PATIENTS), relative(S1)])
    checks = {
        "S01": ("pass" if n == expected else "unresolved", f"{n}/{expected} published FOV inputs; remaining source positions await export."),
        "I01": ("pass", f"{n} image IDs use the source panel and FOV label."),
        "I02": ("pass", f"{n}/{n} exported images connect to core-level or public specimen samples."),
        "I03": ("pass", "Complete native FOV dimensions with documented square padding; no split or resampling."),
        "I04": ("unresolved", f"{linked_images}/{n} images have direct published specimen and patient links; independent review and remaining identities are pending."),
        "P01": ("pass", "Ordered original TIFF names checked against collection proofs and S1."),
        "P02": ("pass", "S1-selected native channels retained with continuous output positions."),
        "P03": ("pass", f"{n} output TIFF heads and ordered OME channel names reopened."),
        "M01": ("unresolved" if linked_images < n else "pass",
                f"FOV-to-sample {n}/{n}; public specimen bridge {linked_images}/{n}; patient-linked sample rows {known_patients}/{len(samples)}. Source core labels preserve the other sample identities."),
        "M02": ("unresolved", "Only exact BRUCE patient bridge and reported sex are filled; unverified per-sample diagnosis, grade, treatment and outcomes remain empty."),
        "I05": ("unresolved", f"Preparer pixel comparison completed for {len(checked)} images; independent stratified review pending."),
        "D01": ("unresolved", f"Preparer delivered {n} images; independent reviewer has not bound or accepted D01."),
    }
    rows = [dict(study=STUDY, preparer=preparer, reviewer="", scope="study",
                 record_id=STUDY, check_id=key, result="unresolved", source_evidence=source,
                 output_evidence=relative(OUTPUT),
                 details=f"Preparer self-check ({result}): {details} Independent review pending.")
            for key, (result, details) in checks.items()]
    frame = pd.DataFrame(rows).reindex(columns=columns("review_results"))
    staging_root = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root, prefix="piyadasa-review-") as temporary:
        staged = Path(temporary) / "review_results.csv"
        frame.to_csv(staged, index=False, encoding="utf-8-sig")
        with formal_publish_lease(path, directory=OUTPUT):
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            shutil.copyfile(staged, partial)
            partial.replace(path)


def stage_identity_repair(destination: Path) -> None:
    """Stage exact public identity links while preserving every physical input field."""
    from cellweft.data.preparation.inputs import file_revision
    destination.mkdir(parents=True, exist_ok=False)
    tables = ("images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv", "review_results.csv")
    before = {name: file_revision(OUTPUT / name) for name in tables}
    for name in tables:
        shutil.copyfile(OUTPUT / name, destination / ("before_" + name))
    images = pd.read_parquet(OUTPUT / "images.parquet")
    old_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    by_sample = old_samples.set_index("sample_id").to_dict("index")
    bridge, patients = identity_bridge()
    sources = {(row["panel"], row["fov"]): row for row in source_scope()}
    active_samples: dict[str, dict] = {}
    linked_images = 0
    for index, image in images.iterrows():
        panel, fov = image.image_id.split("__", 1)
        if fov != image.source_image_id:
            raise ValueError("Image/source FOV differs")
        source = sources[(panel, fov)]
        if (panel, fov) in bridge:
            sample = sample_for(source, bridge, patients)
            linked_images += 1
        else:
            sample = dict(sample_id=image.sample_id, **by_sample[image.sample_id])
        images.at[index, "sample_id"] = sample["sample_id"]
        active_samples[sample["sample_id"]] = sample
    images.to_parquet(destination / "images.parquet", index=False)
    pd.DataFrame(active_samples.values()).reindex(columns=old_samples.columns).to_csv(
        destination / "sample_metadata.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(OUTPUT / "panel_mapping.csv", destination / "panel_mapping.csv")
    field_rules().to_csv(destination / "field_mapping.csv", index=False, encoding="utf-8-sig")
    assert before == {name: file_revision(OUTPUT / name) for name in tables}
    checkpoint(destination / "stage_receipt.json", stage="metadata_staged_pending_independent_review",
               scientific_repair_round=1, formal_changes=0, before_versions=before,
               exact_linked_images=linked_images, unlinked_images=len(images)-linked_images,
               sample_rows=len(active_samples), published_bridge=file_revision(BRIDGE))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage-identity-repair", type=Path,
                    help="Stage metadata-only public identity repair and exit without formal publication")
parser.add_argument("--limit", type=int, default=0, help="Pilot count; zero processes the remaining release")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--start-position", type=int, default=0,
                    help="Zero-based position after a previously verified indexed prefix")
parser.add_argument("--progress-path", type=Path, default=DEFAULT_JOB / "progress.json")
args = parser.parse_args()
if args.stage_identity_repair is not None:
    stage_identity_repair(args.stage_identity_repair)
    raise SystemExit(0)
if args.limit < 0 or args.batch_size < 1 or args.start_position < 0:
    parser.error("--limit and --start-position must be nonnegative; --batch-size must be positive")

records = source_scope()
panels: dict[str, pd.DataFrame] = {}
signature_ids: dict[tuple[str, tuple[str, ...]], str] = {}
bridge, patients = identity_bridge()
previous = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
image_rows = {row["image_id"]: row for row in previous.to_dict("records")}
if args.start_position > len(records) or any(
        f"{record['panel']}__{record['fov']}" not in image_rows
        for record in records[:args.start_position]):
    parser.error("--start-position requires a complete previously indexed source prefix")
samples = (pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
           if (OUTPUT / "sample_metadata.csv").exists()
           else pd.DataFrame(columns=columns("sample_metadata")))
checked = [json.loads(SELF_CHECK.read_text())] if SELF_CHECK.is_file() else []
new_count = 0
unsaved = 0
excluded = []
fields = field_rules()
staging_root = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
staging_root.mkdir(parents=True, exist_ok=True)

for position, record in enumerate(records):
    key = record["panel"], record["fov"]
    mapping = load_mapping(record)
    names = tuple(channel["source_channel_label"] for channel in mapping["channels"])
    signature = key[0], names
    if signature not in signature_ids:
        panel_id = f"{STUDY}__{key[0]}_{sum(item[0] == key[0] for item in signature_ids)+1:03d}"
        panels[panel_id] = panel_from_s1(STUDY, panel_id, list(names), S1)
        signature_ids[signature] = panel_id
    record["panel_id"] = signature_ids[signature]
    panel = panels[record["panel_id"]]
    if position < args.start_position:
        continue
    shape = native_shape(mapping, panel)
    side = max(shape)
    image_id = f"{key[0]}__{key[1]}"
    destination = OUTPUT / "images" / image_id / "input.tiff"
    source = NAS / record["local_path"]
    stat = source.stat()
    if stat.st_size != record["bytes"]:
        raise ValueError(f"Local source size differs from official manifest: {image_id}")
    revision = f"S-BIAD1579:{stat.st_size}:{stat.st_mtime_ns}"
    sample = sample_for(record, bridge, patients)
    existing = image_rows.get(image_id)
    if (existing and existing["source_revision"] == revision
            and existing["panel_id"] == record["panel_id"]
            and existing["input_image_path"] == relative(destination)
            and destination.is_file() and destination.stat().st_size > 0):
        checkpoint(args.progress_path, stage="exporting", next_source_position=position+1,
                   indexed_images=len(image_rows), last_completed_image=image_id, reused=True)
        continue
    if destination.is_file() and inspect_output(destination, panel, side):
        result = dict(channel_count=int(panel.keep_channel.sum()), admitted=True,
                      height_px=side, width_px=side, pad_left_px=(side-shape[1])//2,
                      pad_top_px=(side-shape[0])//2,
                      pad_right_px=side-shape[1]-(side-shape[1])//2,
                      pad_bottom_px=side-shape[0]-(side-shape[0])//2)
    else:
        stale = destination.with_name("input.tiff.partial")
        if stale.is_file():
            with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
                stale.unlink(missing_ok=True)
        estimated = side * side * 4 * int(panel.keep_channel.sum())
        if shutil.disk_usage(NAS).free < 100*GIB + estimated or shutil.disk_usage(staging_root).free < 64*GIB + 2*GIB:
            checkpoint(args.progress_path, stage="waiting_io", waiting_reason="NAS 100 GiB or local 64 GiB reserve",
                       next_source_position=position, indexed_images=len(image_rows), last_completed_image="")
            print(json.dumps({"status": "waiting_io", "indexed_images": len(image_rows), "next_source_position": position}))
            break
        with tempfile.TemporaryDirectory(dir=staging_root, prefix="piyadasa-source-") as temporary:
            local_zip = Path(temporary) / source.name
            with formal_read_lease(source):
                shutil.copyfile(source, local_zip)
            if local_zip.stat().st_size != stat.st_size:
                raise OSError(f"Incomplete source staging: {image_id}")
            reader = ZipPlanes(local_zip, mapping, record["panel_id"], shape)
            try:
                result = export_input(reader, panel, destination)
            finally:
                reader.close()
            if result["admitted"] and not inspect_output(destination, panel, side):
                raise ValueError(f"Published input header differs from source panel: {image_id}")
    if result["admitted"]:
        image_rows[image_id] = image_row(record, mapping, result, revision, sample["sample_id"], destination)
    else:
        excluded.append(image_id)
    new_count += 1
    unsaved += 1
    checkpoint(args.progress_path, stage="exporting", next_source_position=position+1,
               indexed_images=len(image_rows)-unsaved, published_images=len(image_rows),
               last_completed_image=image_id, excluded_zero_signal=excluded)
    if unsaved >= args.batch_size or position == len(records)-1 or args.limit and new_count >= args.limit:
        images = pd.DataFrame(image_rows.values())
        active_samples = {}
        for selected_record in records:
            selected_id = f"{selected_record['panel']}__{selected_record['fov']}"
            if selected_id in image_rows:
                item = sample_for(selected_record, bridge, patients)
                active_samples[item["sample_id"]] = item
        samples = pd.DataFrame(active_samples.values()).reindex(columns=columns("sample_metadata"))
        used = {row["panel_id"] for row in image_rows.values()}
        panel_rows = pd.concat([panels[panel_id] for panel_id in sorted(used)], ignore_index=True)
        save_input_tables(OUTPUT, images, samples, panel_rows, fields)
        linked_images = sum((r["panel"], r["fov"]) in bridge for r in records
                            if f"{r['panel']}__{r['fov']}" in image_rows)
        save_reviews(OUTPUT / "review_results.csv", len(image_rows), samples, linked_images,
                     checked, len(records)-len(excluded), args.progress_path.parent.name)
        unsaved = 0
        checkpoint(args.progress_path, stage="exporting", next_source_position=position+1,
                   indexed_images=len(image_rows), sample_rows=len(samples),
                   sample_linked_images=len(image_rows),
                   patient_linked_samples=int(samples.patient_key.astype(str).ne("").sum()),
                   last_completed_image=image_id, excluded_zero_signal=excluded)
        waiting = queued_io_requests()
        if waiting >= 10 and position < len(records)-1:
            checkpoint(args.progress_path, stage="waiting_io",
                       waiting_reason=f"Shared NAS lease queue contains {waiting} pending requests after a closed input batch",
                       next_source_position=position+1, indexed_images=len(image_rows),
                       sample_rows=len(samples), sample_linked_images=len(image_rows),
                       patient_linked_samples=int(samples.patient_key.astype(str).ne("").sum()),
                       last_completed_image=image_id, excluded_zero_signal=excluded)
            print(json.dumps({"status": "waiting_io", "indexed_images": len(image_rows),
                              "next_source_position": position+1, "queued_io_requests": waiting}))
            break
    if args.limit and new_count >= args.limit:
        break
else:
    checkpoint(args.progress_path, stage="prepared_pending_independent_review",
               next_source_position=len(records), indexed_images=len(image_rows),
               sample_rows=len(samples), sample_linked_images=len(image_rows),
               patient_linked_samples=int(samples.patient_key.astype(str).ne("").sum()),
               excluded_zero_signal=excluded)

print(json.dumps({"study": STUDY, "expected_images": len(records),
                  "exported_images": len(image_rows), "new_or_recovered": new_count,
                  "excluded_zero_signal": len(excluded)}, ensure_ascii=False))
