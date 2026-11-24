"""Export complete native CRLM MCD acquisitions with public ROI evidence.

Inputs: verified local MCD index, acquisition channel map, public supplement,
and current S1. Outputs: native CYX TIFF inputs and per-study input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
import re
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import openpyxl
import pandas as pd
import tifffile
from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Lin2026_IMC_CRLM"
SOURCE_STUDY = "zenodo_22300873"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(ROOT, SOURCE_STUDY)
SUPPLEMENT = ROOT / "raw/public/Lin2027_IMC_CRLM/zenodo_22300873/Supplementary_Tables_public.xlsx"
DICTIONARY = ROOT / "raw/public/Lin2027_IMC_CRLM/zenodo_22300873/data_dictionary.md"
REGION = ROOT / "raw/public" / SOURCE_STUDY / "metadata/region_17638058.csv"
CLINICAL = ROOT / "raw/public" / SOURCE_STUDY / "metadata/clinical_17638058.csv"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", PROJECT / "cache/study_input_jobs" / STUDY / "0749_prepare_0"))
IO_LOCK_ROOT = Path(os.environ["CELLWEFT_NAS_LOCK_ROOT"])
TEMPLATES = PROJECT / "docs/data/templates/study_input"


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, completed: int, next_image: str | None, **details: object) -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    payload = dict(study=STUDY, stage=stage, completed_images=completed,
                   expected_images=329, next_image=next_image,
                   updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    partial = JOB / "progress.json.partial"
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    partial.replace(JOB / "progress.json")


def mapping(source: Path, source_field: str, table: str, target: str,
            rule: str, *, join: str = "", notes: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table="",
                source_field=source_field, target_table=table, target_field=target,
                join_keys=join, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


# Reconcile complete acquisitions, their native labels, and public ROI evidence.
index_path = META / "local_mcd_acquisition_index.json"
channels_path = META / "local_mcd_channel_mapping.csv"
with formal_read_lease(META, data_root=ROOT):
    index = json.loads(index_path.read_text())
    channels = pd.read_csv(channels_path, keep_default_na=False)
    workbook = openpyxl.load_workbook(SUPPLEMENT, read_only=True, data_only=True)
    roi_rows = list(workbook["Supplementary Table 2"].values)[2:]
    clinical_rows = list(workbook["Supplementary Table 1"].values)[2:]
    workbook.close()
    regions = pd.read_csv(REGION, dtype={"sample": str, "ROI": str})
    older_clinical = pd.read_csv(CLINICAL, dtype={"PID": str})
assert len(index["images"]) == index["n_images"] == 329
assert index["container_count"] == 35 and not index["failures"]
assert all(row["passed"] and row["pixel_records"] == row["expected_pixels"]
           for row in index["images"])
roi = {(str(row[0]), int(str(row[1]).removeprefix("ROI"))): row for row in roi_rows}
assert len(roi) == 311 and len({str(row[0]) for row in clinical_rows}) == 35
assert len(regions) == 331 and regions["sample"].nunique() == 35
assert set(regions["sample"]).issubset(set(older_clinical.PID))
region = {(row["sample"], int(row["ROI"].removeprefix("ROI"))): row
          for _, row in regions.iterrows()}
assert len(region) == 331
clinical = {str(row[0]): row for row in clinical_rows}
channel_groups = {(file, int(acquisition)): group.sort_values("channel_index")
                  for (file, acquisition), group in channels.groupby(["source_file", "acquisition_id"])}
assert len(channel_groups) == 329

panels = {}
panel_for_names = {}
with formal_read_lease(S1, data_root=ROOT):
    for group in channel_groups.values():
        names = tuple(group.marker_label)
        if names not in panel_for_names:
            panel_id = f"{SOURCE_STUDY}__MCD_{len(names)}"
            panel_for_names[names] = panel_id
            panels[panel_id] = panel_from_s1(STUDY, panel_id, list(names), S1)
assert {len(names) for names in panel_for_names} == {41, 44}
panel = pd.concat(panels.values(), ignore_index=True)
assert all(len(group[group.keep_channel]) == 37 for group in panels.values())

acquisitions = {}
for source_name in sorted({Path(row["image_path"]).name for row in index["images"]}):
    source = ROOT / "raw/public/Lin2027_IMC_CRLM/local_mcd" / source_name
    with formal_read_lease(source, io_lock_root=IO_LOCK_ROOT, data_root=ROOT):
        with MCDFile(source) as mcd:
            descriptions = {acquisition.id: acquisition.description
                            for slide in mcd.slides for acquisition in slide.acquisitions}
    acquisitions[source_name] = descriptions

records = []
for item in index["images"]:
    source = ROOT / item["image_path"]
    subject = source.stem.rsplit("_", 1)[-1]
    description = acquisitions[source.name][int(item["acquisition_id"])]
    match = re.fullmatch(r"ROI_?(\d+)(?: - split)?", description)
    assert match, (source.name, item["acquisition_id"], description)
    roi_number = int(match.group(1))
    annotation = roi.get((subject, roi_number))
    sample_region = region.get((subject, roi_number))
    names = tuple(channel_groups[(source.name, int(item["acquisition_id"]))].marker_label)
    assert len(names) == item["channels"]
    image_id = f"{source.stem}__a{int(item['acquisition_id']):03d}"
    records.append((item, source, image_id, description, subject, roi_number, annotation,
                    sample_region, panel_for_names[names]))
assert len({record[2] for record in records}) == 329
assert sum(record[6] is not None for record in records) == 308
assert sum(record[7] is not None for record in records) == 328

# The older region table explicitly links each mapped ROI to one source sample.
samples = pd.DataFrame([
    dict(study=STUDY, sample_id=sample, patient_key=sample,
         source_subject_id=sample, source_sample_id=sample,
         sample_type="tissue", tissue_type="liver", anatomical_site="liver",
         primary_diagnosis="colorectal liver metastases",
         sex_reported=clinical[sample][6], histology=clinical[sample][16],
         rfs_time=float(clinical[sample][2]), rfs_time_unit="month",
         rfs_origin="Completion of primary colorectal tumor and liver metastasis resection",
         rfs_event=int(clinical[sample][1] == "Recurrence"),
         rfs_event_definition="Recurrence during follow-up",
         metadata_notes="Source region sample and public patient pseudonym coincide; clinical values from Supplementary Table 1.")
    for sample in sorted(regions["sample"].unique())
], columns=columns("sample_metadata") + ["rfs_time", "rfs_time_unit", "rfs_origin",
                                  "rfs_event", "rfs_event_definition"])
fields = pd.DataFrame([
    mapping(index_path, "images;excluded_acquisitions", "image_metadata",
            "source_files;source_acquisition_id;source_channel_count",
            "Admit 329 complete native tissue acquisitions; exclude indexed calibration and incomplete acquisitions.",
            notes="35 MCD files; 329 complete tissue acquisitions; 68 calibration and seven incomplete/empty acquisitions are excluded by the existing native index."),
    mapping(channels_path, "source_file;acquisition_id;channel_index;marker_label",
            "panel_mapping", "source_name;source_channel_index",
            "Use native channel order for each acquisition; 41- and 44-channel layouts have separate panels.",
            join="MCD filename + acquisition ID = channel map source_file + acquisition_id"),
    mapping(S1, "Study;Original marker", "panel_mapping", "marker_id;role;keep_channel",
            "Join each native marker_label to current study S1; keep 37 selected channels per panel.",
            join="Study = study; Original marker = source_name"),
    mapping(SUPPLEMENT, "Supplementary Table 2: Patient ID;ROI;PT;TC;IM",
            "image_metadata", "roi_id;region_annotation",
            "Join MCD filename subject and numeric acquisition description ROI to public ROI row; retain the indicated tissue label.",
            join="filename B/W subject = Patient ID; acquisition ROI number = ROI number",
            notes="308 of 329 complete acquisitions join the 311-row public ROI table; 21 acquisitions lack a published ROI row and three published ROIs lack a complete acquisition."),
    mapping(REGION, "sample;ROI", "sample_metadata", "sample_id;source_sample_id",
            "Use the explicit source sample value as the biological sample key for each mapped ROI.",
            join="MCD filename subject + acquisition ROI number = region.sample + region.ROI",
            notes="328 of 329 complete acquisitions link to 35 source sample IDs; one complete acquisition has no region row."),
    mapping(CLINICAL, "PID", "sample_metadata", "patient_key;source_subject_id",
            "Join source sample pseudonym to the public clinical PID; one source sample per patient in this IMC cohort.",
            join="region.sample = clinical.PID = sample_id = patient_key",
            notes="35 of 35 source sample IDs join clinical PID; 328 images link to patient-bearing sample rows."),
    mapping(SUPPLEMENT, "Supplementary Table 1: Gender;Histological subtype",
            "sample_metadata", "sex_reported;histology",
            "Retain the published categorical values without inference.",
            join="sample_id = Patient ID"),
    mapping(SUPPLEMENT, "Supplementary Table 1: RFS time", "sample_metadata", "rfs_time",
            "Retain the published numeric recurrence-free survival time in months.",
            join="sample_id = Patient ID",
            notes=json.dumps({"data_type": "number", "unit": "month", "definition": "Recurrence-free survival time from completion of primary colorectal tumor and liver metastasis resection."})),
    mapping(SUPPLEMENT, "Supplementary Table 1: RFS time (months)", "sample_metadata",
            "rfs_time_unit", "Use the published month unit.", join="sample_id = Patient ID",
            notes=json.dumps({"data_type": "string", "definition": "Unit for rfs_time; month."})),
    mapping(DICTIONARY, "FDZS-1 RFS_time", "sample_metadata", "rfs_origin",
            "Use the public data dictionary time origin.", join="sample_id = Patient ID",
            notes=json.dumps({"data_type": "string", "definition": "Completion of resection of the primary colorectal tumor and liver metastases."})),
    mapping(SUPPLEMENT, "Supplementary Table 1: Recurrence status during follow-up",
            "sample_metadata", "rfs_event", "Recurrence maps to 1; No recurrence maps to 0.",
            join="sample_id = Patient ID",
            notes=json.dumps({"data_type": "integer", "definition": "Recurrence indicator during follow-up; 1 event, 0 censored."})),
    mapping(SUPPLEMENT, "Supplementary Table 1: Recurrence status", "sample_metadata",
            "rfs_event_definition", "Record the source event definition.",
            join="sample_id = Patient ID",
            notes=json.dumps({"data_type": "string", "definition": "Documented recurrence during follow-up."})),
    mapping(SUPPLEMENT, "Study title and IMC cohort", "sample_metadata",
            "sample_type;tissue_type;anatomical_site;primary_diagnosis",
            "Study-level tissue and disease context applies to all 35 mapped source samples.",
            join="sample_id is a member of the IMC cohort"),
    mapping(REGION, "normal;tumor;invasive;TLS;QC", "image_metadata",
            "region_annotation;sample_id",
            "Retain the source region class and sample join; unknown class remains empty.",
            join="MCD filename subject + acquisition ROI number = region.sample + region.ROI",
            notes="328 matched acquisitions; one has no region row. Public QC retained as source evidence; one region row has missing QC."),
    mapping(SUPPLEMENT, "Supplementary Table 2: Patient ID;ROI", "image_metadata",
            "source_patient_key", "For matched ROIs use the public patient pseudonym; for 21 unlisted ROIs retain only the MCD filename subject clue.",
            join="MCD filename subject = public Patient ID for 308 matched ROIs",
            notes=json.dumps({"data_type": "string", "definition": "Public B/W subject code inferred from the MCD filename; 308 newer ROI rows and 328 older region rows independently confirm it."})),
], columns=columns("field_mapping"))

existing = {}
if (OUTPUT / "images.parquet").is_file():
    with formal_read_lease(OUTPUT, data_root=ROOT):
        prior = pd.read_parquet(OUTPUT / "images.parquet")
    existing = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
checkpoint("resuming", len(existing), next((record[2] for record in records
                                          if record[2] not in existing), None))

for item, source, image_id, description, subject, roi_number, annotation, sample_region, panel_id in records:
    destination = OUTPUT / "images" / image_id / "input.tiff"
    if image_id in existing and destination.is_file():
        continue
    estimated = 37 * max(item["width"], item["height"]) ** 2 * 4
    if shutil.disk_usage(ROOT).free < 100 * 1024**3 + estimated:
        checkpoint("waiting_io", len(existing), image_id, reason="NAS 100 GiB reserve")
        raise SystemExit(75)
    if shutil.disk_usage("/tmp").free < 64 * 1024**3 + 2 * estimated:
        checkpoint("waiting_io", len(existing), image_id, reason="local 64 GiB reserve")
        raise SystemExit(75)
    names = tuple(channel_groups[(source.name, int(item["acquisition_id"]))].marker_label)
    spec = SourceSpec(image_id=image_id, panel_id=panel_id, kind="mcd", path=source,
                      channels=tuple(range(len(names))), acquisition_id=int(item["acquisition_id"]),
                      pixel_size_um=tuple(item["pixel_size_um"]))
    checkpoint("exporting", len(existing), image_id, source=relative(source))
    try:
        with ExitStack() as stack:
            with formal_read_lease(source, io_lock_root=IO_LOCK_ROOT, data_root=ROOT):
                reader = stack.enter_context(open_source(spec))
            result = export_input(reader, panels[panel_id], destination, tile_size=256,
                                  io_lock_root=IO_LOCK_ROOT, data_root=ROOT)
    except OSError as error:
        checkpoint("waiting_io", len(existing), image_id,
                   reason=f"{type(error).__name__}: {error}", source=relative(source))
        raise SystemExit(75) from error
    if not result["admitted"]:
        checkpoint("excluded_no_marker_signal", len(existing), image_id)
        continue
    with formal_read_lease(destination, data_root=ROOT):
        with tifffile.TiffFile(destination) as tif:
            assert len(tif.pages) == 37 and tif.pages[0].dtype.name == "float32"
            assert tif.pages[0].shape == (result["height_px"], result["width_px"])
    row = dict.fromkeys(columns("image_metadata"))
    region_label = (next((label for label, field in (("PT", "normal"), ("TC", "tumor"),
                                                   ("IM", "invasive"), ("TLS", "TLS"))
                          if sample_region[field] == 1), None) if sample_region is not None else None)
    row.update(study=STUDY, image_id=image_id,
               sample_id=(sample_region["sample"] if sample_region is not None else None),
               panel_id=panel_id,
               image_kind="roi", source_image_id=description, source_files=[relative(source)],
               source_format="mcd", source_acquisition_id=str(item["acquisition_id"]),
               source_axes="CYX", source_dtype="float32", source_revision="local_mcd",
               slide_id=str(item["slide_id"]), roi_id=f"ROI{roi_number}",
               region_annotation=region_label,
               source_x0_px=0, source_y0_px=0, source_x1_px=item["width"],
               source_y1_px=item["height"], pixel_size_x_um=item["pixel_size_um"][0],
               pixel_size_y_um=item["pixel_size_um"][1], input_image_path=relative(destination),
               valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                  if result["valid_pixels"] else None),
               intensity_description="Native unnormalized float32 MCD acquisition intensity",
               split_method="source_acquisition", split_evidence=relative(index_path),
               metadata_notes=("Public source sample and region row matched."
                               if sample_region is not None else "Acquisition absent from public region table; sample association unresolved."),
               source_patient_key=subject,
               **{key: result[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px",
                                                 "pad_bottom_px", "height_px", "width_px",
                                                 "source_channel_count", "channel_count")})
    existing[image_id] = row
    save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields, data_root=ROOT)
    checkpoint("image_saved", len(existing), image_id, sample_rows=len(samples),
               sample_linked_images=sum(bool(row["sample_id"]) for row in existing.values()),
               patient_linked_samples=len(samples))
    print(f"saved {len(existing)}/329", flush=True)

save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields, data_root=ROOT)
field_path = OUTPUT / "field_mapping.csv"
with formal_read_lease(field_path, data_root=ROOT):
    published_fields = pd.read_csv(field_path, dtype=str, keep_default_na=False)
old_rfs_rule = ((published_fields.study == STUDY) &
                (published_fields.target_field == "rfs_time;rfs_time_unit;rfs_origin;rfs_event;rfs_event_definition"))
if old_rfs_rule.any():
    with formal_publish_lease(field_path, directory=OUTPUT, data_root=ROOT):
        partial = field_path.with_suffix(".csv.partial")
        published_fields.loc[~old_rfs_rule].to_csv(partial, index=False, encoding="utf-8-sig")
        partial.replace(field_path)
review = pd.DataFrame([
    dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="unresolved", source_evidence=relative(index_path),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details=f"Preparer exported {len(existing)} of 329 native-complete acquisitions. Native index excludes 68 calibration and seven incomplete/empty acquisitions. Older region table matches 328; one complete acquisition is absent and three region rows lack a complete acquisition. Newer ROI table matches 308; 19 additional matched regions are TLS. Independent review pending."),
    dict(study=STUDY, preparer=JOB.name, reviewer="", scope="sample", record_id=STUDY,
         check_id="M01", result="unresolved", source_evidence=relative(REGION),
         output_evidence=relative(OUTPUT / "sample_metadata.csv"),
         details="Older region table supplies 35 source sample IDs and links 328/329 complete acquisitions; all 35 IDs join clinical PID and newer clinical supplement. One complete acquisition lacks a region/sample row and retains only its filename subject clue. Independent review pending."),
    dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study", record_id=STUDY,
         check_id="D01", result="unresolved", source_evidence=relative(index_path),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details="Preparer output closed; one image-to-sample gap and independent reviewer/controller acceptance pending."),
], columns=columns("review_results"))
review_path = OUTPUT / "review_results.csv"
with formal_read_lease(review_path, data_root=ROOT):
    previous_review = pd.read_csv(review_path) if review_path.exists() else None
if previous_review is not None:
    current_preparer = (previous_review.preparer.fillna("").eq(JOB.name)
                        & previous_review.reviewer.fillna("").eq(""))
    review = pd.concat([previous_review.loc[~current_preparer], review], ignore_index=True)
with formal_publish_lease(review_path, directory=OUTPUT, data_root=ROOT):
    partial = review_path.with_suffix(".csv.partial")
    review.to_csv(partial, index=False, encoding="utf-8-sig")
    partial.replace(review_path)
checkpoint("prepared_pending_independent_review", len(existing), None,
           sample_rows=len(samples),
           sample_linked_images=sum(bool(row["sample_id"]) for row in existing.values()),
           patient_linked_samples=len(samples))
