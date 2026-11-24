"""Prepare native SPORA IMC images and source-linked pancreatic samples."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from xml.etree import ElementTree

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Malchiodi2024_IMC_PancreaticCancer"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--waiting-io", action="store_true")
parser.add_argument("--repair-round", type=int, default=0)
args = parser.parse_args()
if args.waiting_io and not args.metadata_only:
    parser.error("--waiting-io requires --metadata-only")
root = args.data_root
raw = root / "raw/public" / STUDY / "spora"
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
output = root / "processed" / STUDY
receipts = cache / "image_receipts"
temporary = Path("/tmp/study_input_batch") / STUDY
for directory in (cache, output, receipts, temporary):
    directory.mkdir(parents=True, exist_ok=True)
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = temporary / (path.name + ".json.stage")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with formal_publish_lease(path, directory=path.parent):
        pending = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, pending)
        os.replace(pending, path)
    staged.unlink()


def progress(stage: str, count: int, *, current_image: str | None = None,
             detail: str | None = None) -> None:
    if args.progress is None:
        return
    publish_json(args.progress, dict(study=STUDY, stage=stage,
                 phase="repair" if args.repair_round else "prepare",
                 scientific_repair_round=args.repair_round,
                 expected_source_images=164, eligible_source_images=110,
                 exported_images=count, next_image=current_image,
                 detail=detail, updated_utc=datetime.now(timezone.utc).isoformat()))


def field(source: Path, source_table: str, source_field: str, target_table: str,
          target_field: str, join_keys: str, rule: str, notes: str) -> dict:
    row = dict.fromkeys(columns("field_mapping"))
    row.update(study=STUDY, source_file=relative(source), source_sheet_or_table=source_table,
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join_keys, transformation_rule=rule,
               conflict_rule="Preserve conflicting source values in the source table; leave derived value blank.",
               notes=notes)
    return row


class StagedReader:
    """Read one locally staged CYX array without repeated SPORA shard decoding."""

    def __init__(self, array: np.memmap, source: SourceSpec):
        self.array, self.source = array, source
        self.shape, self.dtype = array.shape, array.dtype

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def one_value(group: pd.DataFrame, name: str) -> str | None:
    values = {str(value).strip() for value in group[name] if str(value).strip()}
    return next(iter(values)) if len(values) == 1 else None


def reported_stage(group: pd.DataFrame) -> str | None:
    values = {str(value).strip() for value in group["Stage"] if str(value).strip()}
    if len(values) <= 1:
        return next(iter(values)) if values else None
    if len({"".join(value.split()) for value in values}) != 1:
        return None
    return max(values, key=lambda value: (len(value.split()), value))


tissues_path = raw / "metadata/tissues.parquet"
metadata = metadata_source_directory(root, STUDY)
clinical_path = metadata / "source_metadata/pancreas_TMA_sample_info.xlsx"
verified_path = metadata / "native_verification.json"
channels_path = raw / "imc/channels.parquet"
availability_path = raw / "imc/channels_per_tissue.parquet"
ordered_path = metadata_source_directory(root, STUDY) / 'spora_inventory' / 'imc_ordered_marker_names.json'
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"

tissues = pd.read_parquet(tissues_path).set_index("tissue_id", verify_integrity=True)
clinical = pd.read_excel(clinical_path, dtype=str).fillna("")
assert len(tissues) == len(clinical) == 164
assert tissues.Tissue.value_counts().to_dict() == clinical.Tissue.value_counts().to_dict()
clinical_columns = clinical.columns.tolist()
source_rows = Counter(tuple(str(value).strip() for value in row)
                      for row in tissues[clinical_columns].itertuples(index=False, name=None))
author_rows = Counter(tuple(str(value).strip() for value in row)
                      for row in clinical[clinical_columns].itertuples(index=False, name=None))
exact_clinical_rows = sum((source_rows & author_rows).values())
assert exact_clinical_rows == 158
source_key_fields = ["Image_Name", "ROI_Grid_Location", "Random #"]
spora_source = tissues.reset_index()[["tissue_id", *clinical_columns]].copy()
author_source = clinical[clinical_columns].copy()
for frame in (spora_source, author_source):
    for name in source_key_fields:
        frame[name] = frame[name].astype(str).str.strip()
assert not author_source.duplicated(source_key_fields).any()
source_crosswalk = spora_source.merge(
    author_source, on=source_key_fields, how="left", suffixes=("_spora", "_author"),
    indicator=True, validate="many_to_one")
assert len(source_crosswalk) == len(tissues) and source_crosswalk["_merge"].eq("both").all()
clinical_value_differences = sum(
    any(str(row[f"{name}_spora"]).strip() != str(row[f"{name}_author"]).strip()
        for name in clinical_columns if name not in source_key_fields)
    for _, row in source_crosswalk.iterrows())
assert clinical_value_differences == 0
spora_source_keys = spora_source[source_key_fields].drop_duplicates()
author_only_source_rows = int(author_source[source_key_fields].merge(
    spora_source_keys, on=source_key_fields, how="left", indicator=True)["_merge"].eq("left_only").sum())
assert author_only_source_rows == 6
verified_document = json.loads(verified_path.read_text())
assert verified_document["full_source_covered"] and len(verified_document["verified"]) == 164
verified = {item["image_id"]: item for item in verified_document["verified"]}
assert set(verified) == set(tissues.index)
ordered = json.loads(ordered_path.read_text())
channels = pd.read_parquet(channels_path)
availability = pd.read_parquet(availability_path)
assert availability.columns.tolist() == channels.channel_name.tolist()
assert len(channels) == 42 and len(availability) == 164
selected = sorted(tissues.index[tissues.Tissue.str.startswith("Pancreas")])
assert len(selected) == 110
excluded = sorted(set(tissues.index) - set(selected))
panel_id = f"{STUDY}__spora_native42"
source_names = channels.channel_name.tolist()
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 34 and panel.nuclear.sum() == 2

inventory = []
for image_id in selected:
    item = verified[image_id]
    group = root / item["path"]
    group_meta = json.loads((group / "zarr.json").read_text())["attributes"]["ome"]
    array_meta = json.loads((group / "0/zarr.json").read_text())
    names = channels.loc[availability.loc[image_id].to_numpy(bool), "channel_name"].tolist()
    assert names == source_names == ordered[image_id]
    assert names == item["canonical_channel_labels"] == item["embedded_labels"]
    assert names == [entry["label"] for entry in group_meta["omero"]["channels"]]
    assert array_meta["shape"] == item["shape"] and array_meta["data_type"] == item["dtype"] == "float32"
    assert [axis["name"] for axis in group_meta["multiscales"][0]["axes"]] == ["c", "y", "x"]
    scale = group_meta["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [1.0, 1.0, 1.0]
    files = sorted(path for path in group.rglob("*") if path.is_file() and not path.name.endswith(".partial"))
    assert len(files) == item["source_object_count"]
    assert sum(path.stat().st_size for path in files) == item["source_bytes"]
    inventory.append(dict(image_id=image_id, path=item["path"], shape=item["shape"],
                          files=[(relative(path), path.stat().st_size, path.stat().st_mtime_ns)
                                 for path in files]))

estimated_output = sum(34 * max(item["shape"][1:]) ** 2 * 4 for item in inventory)
maximum_stage = max(42 * item["shape"][1] * item["shape"][2] * 4 +
                    34 * max(item["shape"][1:]) ** 2 * 4 for item in inventory)
if shutil.disk_usage(root).free < 100 * 1024**3 + estimated_output:
    progress("waiting_io", 0, detail="NAS free space below 100 GiB reserve plus estimated output")
    raise SystemExit(75)
if shutil.disk_usage(temporary).free < 64 * 1024**3 + maximum_stage:
    progress("waiting_io", 0, detail="Local free space below 64 GiB reserve plus largest image staging")
    raise SystemExit(75)
inventory_path = cache / "source_inventory.json"
inventory_record = dict(study=STUDY, source_count=164,
                        selected_count=110, excluded_count=len(excluded), excluded_source_images=excluded,
                        exclusion_rule="Tissue starts with Pancreas; non-pancreatic controls and cell preparations excluded",
                        selected=inventory, estimated_uncompressed_output_bytes=estimated_output,
                        maximum_local_staging_bytes=maximum_stage, reused_verification=relative(verified_path),
                        clinical_source=relative(clinical_path),
                        exact_author_spora_clinical_rows=exact_clinical_rows,
                        source_identity_crosswalk_rows=len(source_crosswalk),
                        author_only_split_source_rows=author_only_source_rows,
                        clinical_value_differences=clinical_value_differences)
with formal_read_lease(inventory_path):
    inventory_unchanged = (inventory_path.is_file() and
                           inventory_path.read_text() == json.dumps(inventory_record, ensure_ascii=False, indent=2) + "\n")
if not inventory_unchanged:
    publish_json(inventory_path, inventory_record)

sample_rows = []
image_samples = {}
patient_cases = tissues.groupby("patient_id").Case_Number.apply(
    lambda values: {str(value).strip() for value in values if str(value).strip()}
)
assert all(len(cases) <= 1 for cases in patient_cases)
assert tissues.groupby("patient_id").original_patient_identifier.nunique().max() == 1
assert tissues.groupby("original_patient_identifier").patient_id.nunique().max() == 1
assert tissues[tissues.Case_Number.astype(str).str.strip().ne("")].groupby("Case_Number").patient_id.nunique().max() == 1
for (source_patient, block), group in tissues.loc[selected].groupby(["patient_id", "Block"], dropna=False):
    sample_id = f"{source_patient}__{str(block).strip() or 'block_unspecified'}"
    cases = patient_cases[source_patient]
    case = next(iter(cases)) if cases else None
    patient_note = ("Patient linkage agrees with the unique source patient_id and Case_Number. " if case else
                    "Patient linkage uses the unique source patient_id and original_patient_identifier; "
                    "the original Case_Number is absent for this source patient. ")
    stage_values = sorted({str(value).strip() for value in group["Stage"] if str(value).strip()})
    stage_value = reported_stage(group)
    stage_note = ("Stage source rows differ only by whitespace; stage_reported keeps the source spelling "
                  "with the most explicit token spacing. "
                  if len(stage_values) > 1 and len({"".join(value.split()) for value in stage_values}) == 1
                  else "")
    sample = dict.fromkeys(columns("sample_metadata"))
    sample.update(study=STUDY, sample_id=sample_id,
                  patient_key=(f"{STUDY}__case_{case}" if case else
                               f"{STUDY}__spora_patient_{source_patient}"),
                  source_subject_id=str(source_patient), source_sample_id=sample_id,
                  biospecimen_group_id=sample_id, sample_type="tissue",
                  anatomical_site=one_value(group, "Tissue"),
                  sex_reported={"Male": "male", "Female": "female"}.get(one_value(group, "Sex")),
                  case_control_status="case" if one_value(group, "Tumor") == "Yes" else None,
                  tissue_type=("tumor" if one_value(group, "Tumor") == "Yes" and
                               not one_value(group, "Normal ") else
                               "normal" if one_value(group, "Normal ") == "Yes" and
                               not one_value(group, "Tumor") else None),
                  pathology_diagnosis_reported=one_value(group, "Dx"),
                  age_reported=one_value(group, "Age"),
                  stage_reported=stage_value,
                  procedure_reported=one_value(group, "Procedure"),
                  tumor_size_cm_reported=one_value(group, "Tumor size (cm)"),
                  ethnicity_reported=one_value(group, "Ethnicity"),
                  race_reported=one_value(group, "Race"),
                  source_sample_label=one_value(group, "Sample"),
                  metadata_notes=("Source spreadsheet and SPORA tissues table agree on tissue counts. "
                                  + patient_note +
                                  "Reported age/stage/procedure and diagnosis retain source semantics. "
                                  "Conflicting within-sample values remain blank; inspect original ROI rows. "
                                  + stage_note))
    sample_rows.append(sample)
    for image_id in group.index:
        image_samples[image_id] = sample_id
samples = pd.DataFrame(sample_rows)
assert len(samples) == 56 and len(image_samples) == 110

fields = pd.DataFrame([
    field(tissues_path, "tissues", "tissue_id;Tissue;Image_Name;ROI_Grid_Location", "image_metadata",
          "image_id;sample_id;region_annotation;roi_id", "tissue_id;patient_id+Block",
          "Keep the complete source ROI when Tissue begins with Pancreas; connect to biological sample by source patient_id and Block.",
          "164 source ROIs, 110 pancreatic ROIs selected; 54 controls or cell preparations excluded."),
    field(tissues_path, "tissues", "patient_id;Block;Case_Number", "sample_metadata",
          "sample_id;patient_key;source_subject_id;source_sample_id;biospecimen_group_id",
          "patient_id+Block;Case_Number",
          "One sample per source patient_id and block; use the unique patient-level SPORA identity. Reuse its nonempty Case_Number across blocks; otherwise retain the pseudonymous patient_id as patient_key.",
          "SPORA patient_id and original_patient_identifier are one-to-one across all 164 ROIs; Case_Number is unique per source patient and does not cross source patients. Eight sample rows have no original Case_Number; one shares a source patient with a known-case ROI. The other seven retain a SPORA-only patient key without an inferred case number."),
    field(tissues_path, "tissues", "Case_Number;Sex;Age;Stage;Dx;Tissue;Tumor;Normal ;Procedure;Tumor size (cm);Ethnicity;Race;Sample",
          "sample_metadata", "clinical columns", "source spreadsheet row;patient_id+Block in SPORA tissues",
          "Use SPORA clinical fields grouped by patient_id+Block; preserve reported semantics and leave substantive within-sample conflicts blank.",
          "All 164 SPORA rows match the author workbook by Image_Name, ROI_Grid_Location and Random #. Six workbook-only rows add no unique values to the five explicit case/block groups; one additional row lacks Block and is not assigned to a sample."),
    field(clinical_path, "Sheet1 (2)", "17 original columns", "sample_metadata",
          "source_crosscheck", "Image_Name+ROI_Grid_Location+Random #; then patient_id+Block",
          "Join source rows by the exact source image key, then aggregate to the SPORA patient_id+Block sample. Do not assign workbook-only rows with no Block.",
          "All 164 SPORA rows match a unique author-workbook source row with zero value differences. Six additional workbook rows add no unique field to the five explicit case/block groups; one has no Block and is not assigned. All 110 selected image rows and 56 sample groups are covered."),
    field(channels_path, "channels;channels_per_tissue", "channel_name;availability", "panel_mapping",
          "source_channel_index;source_name", "tissue_id;canonical channel order",
          "Filter canonical rows by per-tissue availability and verify source OME labels for each selected ROI.",
          "All 110 selected images have the same 42 ordered source channels."),
    field(s1_path, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
          "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index", "Study+Original marker",
          "Retain 32 downstream proteins and both DNA segmentation inputs in source order.",
          "Eight technical channels excluded. No downstream duplicate marker."),
    field(verified_path, "verified", "path;shape;dtype;source_bytes", "image_metadata",
          "source_files;source_revision;source_dtype;geometry;input_image_path", "image_id",
          "Keep the complete native 1 um ROI, centered square padding, float32 intensity, and lossless CYX OME TIFF.",
          "Published SPORA float32 intensities are not claimed to be original MCD counts."),
    *[field(clinical_path, "Sheet1 (2)", source, "sample_metadata", target,
            "Image_Name+ROI_Grid_Location+Random #; patient_id+Block",
            ("Map Male/Female to lowercase labels; leave other or conflicting values blank."
             if source == "Sex" else
             "Preserve author-reported value; do not infer time origin or restage."),
            "Exact source-image crosswalk to SPORA; patient/block grouping follows the SPORA tissue table.")
      for source, target in [("Dx", "pathology_diagnosis_reported"), ("Age", "age_reported"),
                             ("Sex", "sex_reported"), ("Procedure", "procedure_reported"),
                             ("Tumor size (cm)", "tumor_size_cm_reported"),
                             ("Ethnicity", "ethnicity_reported"), ("Race", "race_reported"),
                             ("Sample", "source_sample_label")]],
    field(clinical_path, "Sheet1 (2)", "Stage", "sample_metadata", "stage_reported",
          "Image_Name+ROI_Grid_Location+Random #; patient_id+Block",
          "Preserve source stage tokens; if variants differ only by whitespace, keep the source spelling with the most explicit token spacing; do not restage.",
          "One sample has two source spellings differing only by whitespace; original rows remain available in the source workbook."),
    field(tissues_path, "tissues", "Tissue", "sample_metadata", "anatomical_site",
          "patient_id+Block", "Preserve the unique source-reported tissue value.",
          "SPORA source field Tissue, cross-checked to the author workbook."),
    field(tissues_path, "tissues", "Tumor", "sample_metadata", "case_control_status",
          "patient_id+Block", "Map source Tumor=Yes to case; leave other or conflicting values blank.",
          "SPORA source field Tumor, cross-checked to the author workbook."),
    field(tissues_path, "tissues", "Tumor;Normal ", "sample_metadata", "tissue_type",
          "patient_id+Block", "Set tumor or normal only when one source flag is uniquely Yes; otherwise leave blank.",
          "SPORA source fields Tumor and Normal, cross-checked to the author workbook."),
    field(tissues_path, "tissues", "Tissue", "sample_metadata", "sample_type",
          "Tissue starts with Pancreas", "Set tissue for the selected pancreatic tissue rows.",
          "Non-pancreatic controls and cell preparations are excluded from sample_metadata."),
    *[field(tissues_path, "tissues", source, "sample_metadata", target,
            "tissue_id;patient_id+Block",
            ("Preserve source stage tokens; resolve whitespace-only variants using the spelling with the most explicit token spacing; do not restage."
             if source == "Stage" else
             "Preserve SPORA reported value; no inferred timepoint or category conversion."),
            ("SPORA Stage values were cross-checked against the author workbook; one sample has whitespace-only variants."
             if source == "Stage" else
             f"SPORA source field {source}; reported value, temporal origin unspecified. Author workbook crosswalk has zero value differences."))
      for source, target in [("Dx", "pathology_diagnosis_reported"), ("Age", "age_reported"),
                             ("Stage", "stage_reported"), ("Sex", "sex_reported"),
                             ("Procedure", "procedure_reported"),
                             ("Tumor size (cm)", "tumor_size_cm_reported"),
                             ("Ethnicity", "ethnicity_reported"), ("Race", "race_reported"),
                             ("Sample", "source_sample_label")]],
], columns=columns("field_mapping"))

index_path = output / "images.parquet"
existing = {row["image_id"]: row for row in pd.read_parquet(index_path).to_dict("records")} if index_path.exists() else {}
for image_id, row in existing.items():
    if not str(tissues.loc[image_id, "Case_Number"]).strip():
        row["metadata_notes"] = (
            "Source patient_id and Block identify the sample; unique SPORA patient identity "
            "establishes patient linkage when this ROI lacks an original Case_Number."
        )
save_input_tables(output, pd.DataFrame(existing.values(), columns=columns("image_metadata")), samples, panel, fields,
                  replace_metadata_studies={STUDY})
next_image = next((item["image_id"] for item in inventory if item["image_id"] not in existing), None)
if args.waiting_io:
    progress("waiting_io", len(existing), current_image=next_image,
             detail="Shared NAS admission queue delayed per-image completion; resume at next_image using indexed TIFF receipts")
else:
    progress("metadata_saved", len(existing))
if args.metadata_only:
    review_path = output / "review_results.csv"
    preparer = (f"{STUDY}_repair_{args.repair_round}" if args.repair_round else f"{STUDY}_prepare_0")
    review = dict(study=STUDY, preparer=preparer,
                  reviewer=None, scope="study", record_id=STUDY, check_id="D01",
                  result="unresolved", source_evidence=relative(cache / "source_inventory.json"),
                  output_evidence=relative(output / "images.parquet"),
                  details=f"Repair round {args.repair_round}: {len(existing)}/110 eligible images indexed; next={next_image}. "
                          "All 56 sample groups and 110 selected images have patient keys and sample joins. "
                          "The author workbook crosswalks all SPORA source rows; current D01 binding requires independent review.")
    reviews = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame(columns=columns("review_results"))
    reviews = reviews[~((reviews.preparer == review["preparer"]) &
                        (reviews.check_id == "D01") & (reviews.record_id == STUDY))]
    reviews = pd.concat([reviews, pd.DataFrame([review])], ignore_index=True)
    staged_review = temporary / "review_results.csv"
    reviews.to_csv(staged_review, index=False, encoding="utf-8-sig")
    with formal_publish_lease(review_path, directory=output):
        partial = review_path.with_name(review_path.name + ".partial")
        shutil.copyfile(staged_review, partial)
        os.replace(partial, review_path)
    staged_review.unlink()
    publish_json(cache / "input_delivery_summary.json", dict(
        study=STUDY, status="repaired_pending_independent_review",
        scientific_repair_round=args.repair_round, expected_source_images=164,
        eligible_images=110, exported_images=len(existing), sample_rows=len(samples),
        sample_linked_images=sum(bool(row.get("sample_id")) for row in existing.values()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        source_identity_crosswalk_rows=len(source_crosswalk),
        author_only_split_source_rows=author_only_source_rows,
        independent_review="pending",
        source_verification=relative(verified_path), source_inventory=relative(inventory_path)))
    progress("repair_saved_pending_independent_review", len(existing),
             detail="Patient links and author source crosswalk updated; D01 independent review remains pending")
    print(json.dumps(dict(study=STUDY, metadata_refreshed=True, indexed_images=len(existing))), flush=True)
    raise SystemExit(0)

for item in inventory:
    image_id = item["image_id"]
    source_path = root / item["path"]
    destination = output / "images" / image_id / "input.tiff"
    receipt_path = receipts / f"{image_id}.json"
    revision = json.dumps(dict(spora_scale="imc/1_0mpp", files=item["files"]), separators=(",", ":"))
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    reusable = (receipt is not None and receipt["source_revision"] == revision and
                receipt["channel_names"] == kept.marker_id.tolist() and
                (destination.is_file() if receipt["exported"]["admitted"] else not destination.exists()))
    if reusable and (image_id in existing or not receipt["exported"]["admitted"]):
        continue
    if not reusable:
        side = max(item["shape"][1:])
        estimated = len(kept) * side * side * 4
        stage_bytes = 42 * item["shape"][1] * item["shape"][2] * 4 + estimated
        if shutil.disk_usage(root).free < 100 * 1024**3 + estimated or \
                shutil.disk_usage(temporary).free < 64 * 1024**3 + stage_bytes:
            progress("waiting_io", len(existing), current_image=image_id,
                     detail="Space reserve unavailable before next image")
            raise SystemExit(75)
        source = SourceSpec(image_id, panel_id, "zarr", source_path / "0", tuple(range(42)),
                            pixel_size_um=(1.0, 1.0), source_revision=revision)
        stage_path = temporary / f"{image_id}.npy"
        array = None
        try:
            array = np.lib.format.open_memmap(stage_path, mode="w+", dtype="float32", shape=tuple(item["shape"]))
            with formal_read_lease(source.path):
                with open_source(source) as reader:
                    for window in windows(tuple(item["shape"][1:]), 256):
                        x0, y0, x1, y1 = window
                        array[:, y0:y1, x0:x1] = reader.read_window(list(range(42)), window)
            array.flush()
            local_source = SourceSpec(image_id, panel_id, "zarr", stage_path, tuple(range(42)),
                                      pixel_size_um=(1.0, 1.0), source_revision=revision)
            exported = export_input(StagedReader(array, local_source), panel, destination, tile_size=256)
            if not exported["admitted"]:
                publish_json(receipt_path, dict(image_id=image_id, source_revision=revision,
                             channel_names=kept.marker_id.tolist(), exported=exported,
                             exclusion_reason="All retained downstream markers have zero signal"))
                progress("image_excluded_zero_marker", len(existing), current_image=image_id)
                continue
            assert exported["valid_pixels"] is None
            with formal_read_lease(destination):
                with tifffile.TiffFile(destination) as tif:
                    names = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                             if node.tag.endswith("}Channel")]
                    assert names == kept.marker_id.tolist()
                    assert tif.series[0].shape == (len(kept), side, side)
                    assert tif.series[0].dtype == np.dtype("float32")
                    x1, y1 = min(32, item["shape"][2]), min(32, item["shape"][1])
                    left, top = exported["pad_left_px"], exported["pad_top_px"]
                    for position in (0, len(kept)//2, len(kept)-1):
                        channel = kept.iloc[position]
                        plane = tif.asarray(key=position)
                        actual = plane[top:top+y1, left:left+x1]
                        assert np.array_equal(actual, array[int(channel.source_channel_index), :y1, :x1])
                        if left:
                            assert not np.any(plane[:, :left])
                        if top:
                            assert not np.any(plane[:top, :])
            receipt = dict(image_id=image_id, source_revision=revision,
                           channel_names=kept.marker_id.tolist(), exported=exported,
                           preparer_pixel_check="first, middle and last retained channels, origin 32x32 exact")
            publish_json(receipt_path, receipt)
        finally:
            if array is not None:
                array._mmap.close()
            stage_path.unlink(missing_ok=True)
    if not receipt["exported"]["admitted"]:
        continue
    exported = receipt["exported"]
    source_row = tissues.loc[image_id]
    h, w = item["shape"][1:]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=image_id, sample_id=image_samples[image_id],
               panel_id=panel_id, image_kind="roi", source_image_id=image_id,
               source_files=[item["path"]], source_format="ome_zarr", source_axes="CYX",
               source_dtype="float32", source_revision=revision,
               roi_id=str(source_row.Image_Name), region_annotation=str(source_row.Tissue),
               source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
               pad_left_px=exported["pad_left_px"], pad_top_px=exported["pad_top_px"],
               pad_right_px=exported["pad_right_px"], pad_bottom_px=exported["pad_bottom_px"],
               height_px=exported["height_px"], width_px=exported["width_px"],
               source_channel_count=42, channel_count=len(kept),
               pixel_size_x_um=1.0, pixel_size_y_um=1.0,
               input_image_path=relative(destination),
               intensity_description="Native SPORA-published float32 intensity at 1 um; no normalization; original MCD counts not asserted.",
               split_method="already_independent",
               split_evidence="Full published tissue_id ROI; source tissue annotation selects pancreatic material; no sub-ROI split.",
               metadata_notes="Source patient_id and Block identify the sample; patient linkage follows the unique SPORA patient identity and available Case_Number.")
    existing[image_id] = row
    save_input_tables(output, pd.DataFrame(existing.values()), samples, panel, fields)
    next_image = next((entry["image_id"] for entry in inventory if entry["image_id"] not in existing), None)
    progress("image_saved", len(existing), current_image=next_image)
    print(json.dumps(dict(saved=len(existing), image_id=image_id)), flush=True)

review_path = output / "review_results.csv"
preparer_row = dict(study=STUDY, preparer="Malchiodi2024_IMC_PancreaticCancer_prepare_0",
                    reviewer=None, scope="study", record_id=STUDY, check_id="D01",
                    result="unresolved", source_evidence=relative(cache / "source_inventory.json"),
                    output_evidence=relative(output / "images.parquet"),
                    details="Preparer delivery closed; independent review and controller acceptance pending. "
                            "Patient case-number gaps remain explicit in sample_metadata.csv.")
reviews = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame(columns=columns("review_results"))
reviews = reviews[~((reviews.preparer == preparer_row["preparer"]) &
                    (reviews.check_id == "D01") & (reviews.record_id == STUDY))]
reviews = pd.concat([reviews, pd.DataFrame([preparer_row])], ignore_index=True)
staged_review = temporary / "review_results.csv"
reviews.to_csv(staged_review, index=False, encoding="utf-8-sig")
with formal_publish_lease(review_path, directory=output):
    partial = review_path.with_name(review_path.name + ".partial")
    shutil.copyfile(staged_review, partial)
    os.replace(partial, review_path)
staged_review.unlink()
summary = dict(study=STUDY, status="prepared", expected_source_images=164,
               eligible_images=110, exported_images=len(existing), sample_rows=len(samples),
               sample_linked_images=sum(bool(row.get("sample_id")) for row in existing.values()),
               patient_linked_samples=int(samples.patient_key.notna().sum()),
               independent_review="pending", source_verification=relative(verified_path),
               source_inventory=relative(cache / "source_inventory.json"))
publish_json(cache / "input_delivery_summary.json", summary)
progress("prepared_pending_independent_review", len(existing))
print(json.dumps(summary, ensure_ascii=False), flush=True)
