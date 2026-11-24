"""Prepare native IMC and IMC-RNAscope ROIs from Zenodo record 4911135.

Inputs: two verified original ZIPs and ROI tables, ordered channels, S1, and
published Supplementary Data 1. Outputs: per-ROI CYX TIFF and study input tables.
Each ZIP member is staged alone; input publication uses the shared NAS lease.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import openpyxl
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Tietscher2023_IMC_BreastCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_4911135"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0631_prepare_0"
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PROOFS = NAS / "raw/public" / STUDY / "manifest.json"
CLINICAL = NAS / "raw/public" / STUDY / "metadata/supplementary_data_1.xlsx"
PANELS = ("Protein_panel", "RNAscope_panel")


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def update_progress(phase: str, completed: int, **details: object) -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    path = JOB / "progress.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps({"study": STUDY, "phase": phase,
                                     "expected_images": 154, "completed_images": completed,
                                     "updated_at": datetime.now(timezone.utc).isoformat(),
                                     **details}, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def template_columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def field(source: Path, source_field: str, target_table: str, target_field: str,
          rule: str, *, notes: str = "", join: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table=source.name,
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


def clinical_rows() -> dict[str, dict]:
    sheet = openpyxl.load_workbook(CLINICAL, read_only=True, data_only=True).active
    rows = list(sheet.values)
    names = rows[2][:26]
    return {str(values[0]): dict(zip(names, values[:26])) for values in rows[3:] if values[0]}


def make_samples(roi_rows: list[dict], clinical: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sample in sorted({row["sample"] for row in roi_rows}):
        patient = sample.removeprefix("T")
        if not sample.startswith("TBB") or patient not in clinical:
            raise ValueError("Source sample has no exact patient row in Supplementary Data 1")
        source = clinical[patient]
        row = dict.fromkeys(template_columns("sample_metadata"), "")
        row.update(study=STUDY, sample_id=sample, patient_key=patient,
                   source_subject_id=patient, source_sample_id=sample,
                   sample_type="breast tumor resection", tissue_type="breast tumor",
                   anatomical_site="breast", primary_diagnosis="breast cancer",
                   histology=source["Histopathology"] or "",
                   tnm_t=source["T Staging"] or "", tnm_n=source["N Staging"] or "",
                   tnm_m=source["M Staging"] or "", grade=source["Grade"] or "",
                   molecular_subtype=source["Clinical Subtype"] or "",
                   metadata_notes="Clinical values from published Supplementary Data 1; age is an interval and is not converted to a point age.")
        row["age_range_at_surgery"] = source["Age range at Surgery"] or ""
        row["menopause_status"] = source["Menopause Status"] or ""
        row["neoadjuvant_therapy_reported"] = source["Neoadjuvant Therapy Received"] or ""
        rows.append(row)
    return pd.DataFrame(rows)


def source_plan(mapping: dict) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    plan, roi_rows, panels = [], [], []
    for panel_name in PANELS:
        info = mapping[panel_name]
        archive = RAW / info["archive"]
        roi_file = RAW / f"{panel_name}_ROI_info.csv"
        proof_file = PROOFS
        proof = json.loads(proof_file.read_text())["preparation_sources"]["panel_members"][panel_name]
        if proof["status"] != "done" or proof["verified_count"] != 77:
            raise ValueError(f"Incomplete native verification: {proof_file}")
        with roi_file.open(newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        if len(source_rows) != 77:
            raise ValueError(f"Unexpected ROI table count: {roi_file}")
        by_number = {row["ImageNumber"]: row for row in source_rows}
        if len(by_number) != 77:
            raise ValueError(f"Duplicate ROI number: {roi_file}")
        names = [item["marker_key"] for item in info["ordered_channels"]]
        channel_panel = panel_from_s1(STUDY, panel_name, names, S1)
        panels.append(channel_panel)
        with formal_read_lease(archive, data_root=NAS):
            with zipfile.ZipFile(archive) as zipped:
                members = {item.filename: item for item in zipped.infolist() if not item.is_dir()}
        if len(members) != 77:
            raise ValueError(f"Unexpected ZIP member count: {archive}")
        for verified in proof["verified_images"]:
            source = by_number[verified["roi"]["ImageNumber"]]
            if any(source[key] != verified["roi"][key] for key in ("ImageNumber", "sample", "ROI", "panel_type")):
                raise ValueError("Native proof and ROI table disagree")
            member = verified["member"]
            if member not in members or members[member].file_size != verified["member_bytes"]:
                raise ValueError("Native member differs from verified source")
            if verified["channel_ifd_binding"]["series_axes"] != "QYX" or verified["shape"][0] != 46:
                raise ValueError("Native layout differs from accepted IFD binding")
            roi_rows.append(source)
            plan.append(dict(panel=panel_name, archive=archive, roi_file=roi_file,
                             member=member, member_bytes=members[member].file_size,
                             source=source, image_id=f"{'Protein' if panel_name == 'Protein_panel' else 'RNA'}_{int(source['ImageNumber']):03d}"))
    if len(plan) != 154 or len({item["image_id"] for item in plan}) != 154:
        raise ValueError("Native image plan is not 154 unique acquisitions")
    return plan, pd.concat(panels, ignore_index=True), roi_rows


def mapping_rows() -> pd.DataFrame:
    rows = []
    for panel_name in PANELS:
        roi_file = RAW / f"{panel_name}_ROI_info.csv"
        archive = RAW / f"{panel_name}_rawdata_ometiffs.zip"
        rows.extend([
            field(roi_file, "ImageNumber, ROI, panel_type", "image_metadata", "image_id",
                  "Panel-prefixed ImageNumber identifies one native ROI; ROI is retained as roi_id."),
            field(roi_file, "sample", "image_metadata", "sample_id",
                  "Exact source sample code; all ROIs of the same tissue sample share one sample_id."),
            field(roi_file, "TLS, TIG", "image_metadata", "region_annotation",
                  "Preserve both source ROI annotations as a JSON object."),
            field(archive, "native ZIP member / OME Channel / IFD", "panel_mapping", "source_channel_index",
                  "Accepted source proof binds IFD index i to ordered channel i; QYX legacy OME layout."),
        ])
    rows.extend([
        field(CLINICAL, "Patient ID", "sample_metadata", "patient_key",
              "Remove the initial T from source TBB sample code and require exact match to the published BB patient row.",
              join="ROI_info.sample[1:] = Supplementary Data 1.Patient ID"),
        field(CLINICAL, "Histopathology, T Staging, N Staging, M Staging, Grade, Clinical Subtype",
              "sample_metadata", "clinical_fields", "Retain published categorical values without stage or grade inference.",
              join="ROI_info.sample[1:] = Supplementary Data 1.Patient ID"),
        field(CLINICAL, "Age range at Surgery", "sample_metadata", "age_range_at_surgery",
              "Retain source interval, without converting to a point age.",
              join="ROI_info.sample[1:] = Supplementary Data 1.Patient ID",
              notes=json.dumps({"data_type": "string", "definition": "Published age interval at surgery"})),
        field(CLINICAL, "Menopause Status", "sample_metadata", "menopause_status",
              "Retain published category.", join="ROI_info.sample[1:] = Supplementary Data 1.Patient ID",
              notes=json.dumps({"data_type": "string", "definition": "Reported menopause status at surgery"})),
        field(CLINICAL, "Neoadjuvant Therapy Received", "sample_metadata", "neoadjuvant_therapy_reported",
              "Retain published value; no timing or regimen inference.",
              join="ROI_info.sample[1:] = Supplementary Data 1.Patient ID",
              notes=json.dumps({"data_type": "string", "definition": "Published neoadjuvant treatment report"})),
    ])
    return pd.DataFrame(rows, columns=template_columns("field_mapping"))


def image_row(item: dict, result: dict, *, pixel_size: tuple[float, float] | None) -> dict:
    source = item["source"]
    image_id = item["image_id"]
    row = dict.fromkeys(template_columns("image_metadata"), None)
    row.update(study=STUDY, image_id=image_id, sample_id=source["sample"],
               panel_id=item["panel"], image_kind="roi", source_image_id=item["member"],
               source_files=[relative(item["archive"])], source_format="ome_tiff_zip",
               source_axes="QYX (IFD index binds source channel)", source_dtype=result["source_dtype"],
               source_revision="Zenodo 4911135 v1; verified native IFD binding",
               roi_id=source["ROI"],
               region_annotation=json.dumps({"TLS": source["TLS"], "TIG": source["TIG"]}),
               source_x0_px=0, source_y0_px=0, source_x1_px=1000, source_y1_px=1000,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=result["height_px"], width_px=result["width_px"],
               source_channel_count=result["source_channel_count"], channel_count=result["channel_count"],
               pixel_size_x_um=pixel_size[0] if pixel_size else None,
               pixel_size_y_um=pixel_size[1] if pixel_size else None,
               input_image_path=relative(OUTPUT / "images" / image_id / "input.tiff"),
               valid_pixels_path=relative(OUTPUT / "images" / image_id / "valid_pixels.tiff") if result["valid_pixels"] else None,
               intensity_description="Original decoded float32 IMC acquisition intensities; no normalization",
               split_method="already_independent", split_evidence="One verified native OME TIFF per ROI",
               metadata_notes="Protein and RNAscope panels are adjacent sections; matching ROI numbers do not imply identical pixels.")
    return row


def save_review(completed: int, *, excluded: list[str]) -> None:
    checks = [("S01", "study", STUDY, "pass" if completed + len(excluded) == 154 else "unresolved",
               f"Verified source scope: 154 native members; exported {completed}; excluded {len(excluded)} after signal check."),
              ("M01", "sample", STUDY, "pass",
               "12 source tissue samples link to 12 published patient rows through exact TBB-to-BB code transformation."),
              ("D01", "study", STUDY, "unresolved",
               "Preparer self-check only; independent reviewer must check current files and write acceptance binding.")]
    rows = [dict(study=STUDY, preparer="0631_prepare_0", reviewer="", scope=scope,
                 record_id=record, check_id=check, result=status,
                 source_evidence=";".join((relative(RAW / "Protein_panel_ROI_info.csv"),
                                           relative(RAW / "RNAscope_panel_ROI_info.csv"),
                                           relative(CLINICAL), relative(PROOFS) + "#preparation_sources/panel_members/Protein_panel",
                                           relative(PROOFS) + "#preparation_sources/panel_members/RNAscope_panel")),
                 output_evidence=relative(OUTPUT), details=details)
            for check, scope, record, status, details in checks]
    payload = pd.DataFrame(rows, columns=template_columns("review_results")).to_csv(index=False).encode("utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="tietscher-review-") as temporary:
        staged = Path(temporary) / "review_results.csv"
        staged.write_bytes(payload)
        destination = OUTPUT / "review_results.csv"
        with formal_publish_lease(destination, directory=OUTPUT, data_root=NAS):
            partial = destination.with_suffix(".csv.partial")
            shutil.copyfile(staged, partial)
            os.replace(partial, destination)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-images", type=int, help="Stop after this many new images for a bounded run")
parser.add_argument("--waiting-io", help="Close the current batch at its persisted index and record the I/O wait reason")
args = parser.parse_args()
mapping = json.loads((META / "ordered_channel_mapping.json").read_text())
plan, panels, roi_rows = source_plan(mapping)
samples = make_samples(roi_rows, clinical_rows())
fields = mapping_rows()
OUTPUT.mkdir(parents=True, exist_ok=True)
with formal_read_lease(OUTPUT / "images.parquet", data_root=NAS):
    prior = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
completed_ids = set(prior.image_id) if len(prior) else set()
checkpoint = JOB / "progress.json"
excluded = json.loads(checkpoint.read_text()).get("excluded_images", []) if checkpoint.is_file() else []
done_ids = completed_ids | set(excluded)
update_progress("preparing", len(completed_ids), excluded_images=excluded,
                next_image=next((x["image_id"] for x in plan if x["image_id"] not in done_ids), None))
if args.waiting_io:
    save_review(len(completed_ids), excluded=excluded)
    update_progress("waiting_io", len(completed_ids), excluded_images=excluded,
                    next_image=next((x["image_id"] for x in plan if x["image_id"] not in done_ids), None),
                    reason=args.waiting_io, sample_rows=len(samples),
                    sample_linked_images=int(prior.sample_id.notna().sum()) if len(prior) else 0,
                    patient_linked_samples=int(samples.patient_key.astype(bool).sum()))
    raise SystemExit(0)
new_count = 0

for item in plan:
    image_id = item["image_id"]
    if image_id in done_ids:
        continue
    if args.max_images is not None and new_count >= args.max_images:
        break
    image_dir = OUTPUT / "images" / image_id
    destination = image_dir / "input.tiff"
    if shutil.disk_usage(NAS).free < 100 * 1024**3 + 46 * 1000 * 1000 * 4:
        update_progress("waiting_io", len(completed_ids), next_image=image_id,
                        reason="NAS output would violate 100 GiB reserve")
        break
    staging = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
    staging.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(staging).free < item["member_bytes"] + 64 * 1024**3:
        update_progress("waiting_io", len(completed_ids), next_image=image_id,
                        reason="Local staging would violate 64 GiB reserve")
        break
    with tempfile.TemporaryDirectory(prefix="tietscher-member-", dir=staging) as temporary:
        local = Path(temporary) / "source.ome.tiff"
        with formal_read_lease(item["archive"], data_root=NAS):
            with zipfile.ZipFile(item["archive"]) as zipped, zipped.open(item["member"]) as source, local.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
        if local.stat().st_size != item["member_bytes"]:
            raise ValueError(f"Source member size changed: {image_id}")
        with tifffile.TiffFile(local) as tif:
            if len(tif.pages) != 46 or tif.series[0].axes != "QYX" or tif.series[0].shape != (46, 1000, 1000):
                raise ValueError(f"Native IFD layout changed: {image_id}")
            if any(page.dtype.name != "float32" or page.shape != (1000, 1000) for page in tif.pages):
                raise ValueError(f"Native page type or geometry changed: {image_id}")
            root = ElementTree.fromstring(tif.ome_metadata)
            native_names = [node.attrib.get("Name") for node in root.iter() if node.tag.endswith("}Channel")]
            expected_names = [entry["ome_channel_name"] for entry in mapping[item["panel"]]["ordered_channels"]]
            if native_names != expected_names:
                raise ValueError(f"Source channel identity changed: {image_id}")
        spec = SourceSpec(image_id, item["panel"], "tiff", local, tuple(range(46)))
        selected_panel = panels[panels.panel_id == item["panel"]]
        with open_source(spec) as reader:
            result = export_input(reader, selected_panel, destination, data_root=NAS)
        if not result["admitted"]:
            excluded.append(image_id)
            done_ids.add(image_id)
            update_progress("preparing", len(completed_ids),
                            next_image=next((x["image_id"] for x in plan if x["image_id"] not in done_ids), None),
                            excluded_images=excluded)
            continue
        row = image_row(item, result, pixel_size=None)
        save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panels, fields, data_root=NAS)
        completed_ids.add(image_id)
        done_ids.add(image_id)
        new_count += 1
        update_progress("preparing", len(completed_ids), last_image=image_id,
                        next_image=next((x["image_id"] for x in plan if x["image_id"] not in done_ids), None),
                        excluded_images=excluded)

save_review(len(completed_ids), excluded=excluded)
if len(completed_ids) + len(excluded) == 154:
    update_progress("prepared", len(completed_ids), excluded_images=excluded,
                    sample_rows=len(samples), sample_linked_images=len(completed_ids),
                    patient_linked_samples=int(samples.patient_key.astype(bool).sum()))
