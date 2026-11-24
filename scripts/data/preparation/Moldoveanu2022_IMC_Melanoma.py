"""Deliver native Moldoveanu IMC ROIs with source-linked sample metadata.

Inputs: complete local SPORA Zarrs, native verification, tissues, ordered channels,
and study S1 rows. Outputs: per-ROI input.tiff and the five study input tables.
The original Zenodo release identifies one directory per sample; SPORA's clinical
Sample_ID supplies a patient link for 72 of the 90 released ROIs.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import ImageBlock, SourceSpec, Window
from cellweft.data.io.readers import windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Moldoveanu2022_IMC_Melanoma"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "spora"
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0254_prepare_0")))
META = metadata_source_directory(ROOT, STUDY)
PROOF = META / "native_verification.json"
OBJECTS = META / "imc_source_objects.json"
ORDERED = META / 'spora_inventory' / 'imc_ordered_marker_names.json'
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
ZENODO = "https://zenodo.org/records/5903190"
CODE_RELEASE = "https://zenodo.org/records/5903179"
ST1 = META / "source_metadata/ST1_sample_info.txt"
zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
blosc.set_nthreads(2)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def publish_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with formal_publish_lease(path, directory=path.parent):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)


class DecodedROI:
    """Read an already decoded source ROI so a sharded chunk is decoded once."""

    def __init__(self, values: np.ndarray, source: SourceSpec):
        self.values = values
        # The source pixels are already decoded under the NAS read lease.
        self.source = replace(source, path=Path(os.environ.get(
            "CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")))
        self.shape, self.dtype = values.shape, values.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def iter_blocks(self, channels: list[int], block_size: int):
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


def source_value(value: object) -> str:
    return "" if pd.isna(value) else str(value)


# Establish source scope and panel identity before any output publication.
tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
ordered = json.loads(ORDERED.read_text())
proof = json.loads(PROOF.read_text())
verified = {row["image_id"]: row for row in proof["verified"]}
assert proof["full_source_covered"] and not proof["failed"]
assert len(tissues) == len(ordered) == len(verified) == 90
assert set(tissues.tissue_id) == set(ordered) == set(verified)
assert tissues.tissue_id.is_unique and tissues.original_patient_identifier.is_unique
assert tissues.original_file_name.notna().all() and tissues.original_file_name.is_unique
assert tissues.loc[tissues.Sample_ID.notna(), "Sample_ID"].eq(
    tissues.loc[tissues.Sample_ID.notna(), "original_patient_identifier"]).all()
st1 = pd.read_csv(ST1, sep="\t", dtype=str, keep_default_na=False).fillna("")
st1_linked = st1.loc[st1.filename.ne(""), ["filename", "Sample_ID"]]
tissues_linked = tissues.loc[tissues.Sample_ID.notna(),
                             ["original_file_name", "Sample_ID"]]
assert len(st1_linked) == 72 and st1_linked.filename.is_unique
assert st1_linked.Sample_ID.ne("").all()
assert dict(zip(st1_linked.filename, st1_linked.Sample_ID)) == dict(
    zip(tissues_linked.original_file_name, tissues_linked.Sample_ID))

object_paths: dict[str, list[tuple[Path, int]]] = {}
prefix = "spora-submissions/moldoveanu2022spatially/"
for obj in json.loads(OBJECTS.read_text()):
    tail = obj["key"].removeprefix(prefix)
    image_id = tail.split("/images/")[1].split(".ome.zarr/")[0]
    object_paths.setdefault(image_id, []).append((RAW / tail, obj["size"]))
assert set(object_paths) == set(verified)

panels = {}
for names in {tuple(value) for value in ordered.values()}:
    panel_id = f"spora_imc_{len(names)}"
    panels[panel_id] = panel_from_s1(STUDY, panel_id, list(names), S1)
panel = pd.concat(panels.values(), ignore_index=True)
assert set(panels) == {"spora_imc_35", "spora_imc_37"}

# The raw release has one directory per sample. Clinical Sample_ID is the
# separately confirmed patient key; it is absent for 18 published source rows.
clinical_fields = [
    "Age_last_alive", "CTLA4", "Cohort", "Drug", "Known_Deceased",
    "Lymphocyte_Infiltration", "PD1", "Primary_Site_Simplified", "Primary_site",
    "Response", "Survival_from_Rx_Start", "Tissue_Source",
    "Tissue_Source_Simplified", "Treatment", "Type", "irRC",
]
sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv", nrows=0).columns.tolist()
sample_rows = []
assert tissues.patient_id.notna().all() and tissues.patient_id.is_unique
assert tissues.original_patient_identifier.notna().all()
for row in tissues.itertuples(index=False):
    sample = {column: "" for column in sample_columns}
    patient = source_value(row.Sample_ID)
    patient_hash = source_value(row.patient_id)
    sample.update(study=STUDY, sample_id=row.tissue_id,
                  source_sample_id=row.original_file_name,
                  patient_key=f"{STUDY}:{patient_hash}",
                  source_subject_id=row.original_patient_identifier,
                  sex_reported={"M": "male", "F": "female"}.get(source_value(row.Sex), ""),
                  metadata_notes=("Patient linked by public Sample_ID; canonical patient_key uses the SPORA patient hash."
                                  if patient else "Public Sample_ID absent; patient_key uses the exact tissue-linked SPORA patient hash."))
    for field in clinical_fields:
        sample["source_" + field.lower()] = source_value(getattr(row, field))
    sample_rows.append(sample)
samples = pd.DataFrame(sample_rows)

field_rows = [
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues", source_field="tissue_id",
         target_table="image_metadata", target_field="image_id",
         join_keys="tissues.tissue_id = Zarr basename",
         value_mapping="", transformation_rule="Copy the exact SPORA tissue identifier to image_id.",
         time_origin_rule="", conflict_rule="Require one exact tissue_id per source image.",
         notes=""),
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues", source_field="tissue_id",
         target_table="sample_metadata", target_field="sample_id",
         join_keys="tissues.tissue_id = Zarr basename = sample_id",
         value_mapping="", transformation_rule="Use one exact SPORA tissue_id as one sample row.",
         time_origin_rule="", conflict_rule="Require one unique tissue_id per sample row.",
         notes=""),
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues", source_field="original_file_name",
         target_table="sample_metadata", target_field="source_sample_id",
         join_keys="tissues.tissue_id = image.source_image_id = sample_metadata.sample_id",
         value_mapping="", transformation_rule="Preserve the exact original source filename as the source sample identifier.",
         time_origin_rule="", conflict_rule="Require one unique original_file_name per tissue_id.",
         notes="SPoRA defines original_file_name as the source trace for a tissue; it is distinct from original_patient_identifier. https://spora.epfl.ch/docs/io/format.html"),
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues", source_field="patient_id",
         target_table="sample_metadata", target_field="patient_key",
         join_keys="tissues.tissue_id = image.source_image_id = sample_metadata.sample_id",
         value_mapping="", transformation_rule=f"Namespace the exact SPoRA pseudonymous patient hash with {STUDY}.",
         time_origin_rule="", conflict_rule="Require one nonmissing patient_id per tissue; do not infer from source filenames.",
         notes=""),
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues", source_field="original_patient_identifier",
         target_table="sample_metadata", target_field="source_subject_id",
         join_keys="tissues.tissue_id = image.source_image_id = sample_metadata.sample_id",
         value_mapping="", transformation_rule="Preserve the exact original dataset patient identifier as the source subject identifier.",
         time_origin_rule="", conflict_rule="Require one source patient identifier per tissue; do not use it as a sample identifier.",
         notes="SPoRA defines original_patient_identifier as the original dataset patient ID; the code-release Sample_ID cross-checks this value for 72 linked records. https://spora.epfl.ch/docs/io/format.html"),
    dict(study=STUDY, source_file=relative(ORDERED), source_sheet_or_table="SPORA native channel order + study S1",
         source_field="image_id;ordered marker labels;Original marker;Unified marker;Segmentation channel",
         target_table="panel_mapping", target_field="source_name;marker_id;role;keep_channel;output_channel_index",
         join_keys="image_id -> actual ordered labels; exact source name -> study S1",
         value_mapping="", transformation_rule="Two native panels; retain S1 admitted measurements and selected DNA inputs in source order.",
         time_origin_rule="", conflict_rule="Require exact proof and embedded label agreement for every image.", notes="35- and 37-channel source panels."),
    dict(study=STUDY, source_file=relative(PROOF), source_sheet_or_table="native verification + source object inventory",
         source_field="shape;dtype;dimension_names;canonical_channel_labels;source object sizes",
         target_table="image_metadata", target_field="source_axes;source_dtype;source_channel_count;source_x1_px;source_y1_px;input_image_path",
         join_keys="image_id -> source Zarr store and verified object list",
         value_mapping="", transformation_rule="Use full native CYX ROI, 1 um pixels; square-pad without rescaling or intensity conversion.",
         time_origin_rule="", conflict_rule="Export only stores whose current object sizes and ordered channels match verified evidence.",
         notes="Source publication identifies one directory per sample; the separate public code release ST1 is cross-checked below."),
    dict(study=STUDY, source_file=relative(ST1),
         source_sheet_or_table="Public code release ST1_sample_info", source_field="filename;Sample_ID",
         target_table="sample_metadata", target_field="source_subject_id",
         join_keys="ST1 filename = SPORA original_file_name; matched Sample_ID = SPORA Sample_ID",
         value_mapping="", transformation_rule="Cross-check the 72 source identifiers; canonical patient_key uses SPORA patient_id for all sample rows.",
         time_origin_rule="", conflict_rule="Use ST1 only for exact filename/Sample_ID matches; do not assign identifiers to unmatched rows from this source.",
         notes=f"Published source: {CODE_RELEASE}; 72 exact filename and Sample_ID matches."),
    dict(study=STUDY, source_file="scripts/data/preparation/Moldoveanu2022_IMC_Melanoma.py",
         source_sheet_or_table="study configuration", source_field="STUDY",
         target_table="sample_metadata", target_field="study",
         join_keys="One canonical study declaration for this output table",
         value_mapping="", transformation_rule="Copy the study identifier declared by the dataset preparer.",
         time_origin_rule="", conflict_rule="All sample rows use the same declared study identifier.",
         notes="Internal canonical identifier; not a patient or sample identity."),
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues patient identifiers", source_field="Sample_ID;patient_id;original_patient_identifier",
         target_table="sample_metadata", target_field="metadata_notes",
         join_keys="images.source_image_id = SPORA tissue_id; images.sample_id = sample_metadata.sample_id",
         value_mapping="", transformation_rule="Record whether the original Sample_ID exists and document the linked SPORA patient hash used for patient_key.",
         time_origin_rule="", conflict_rule="Only describe exact tissue-linked identifiers; do not infer missing clinical values or timepoints.",
         notes="Generated row-level provenance note; source identifiers remain in dedicated fields."),
    dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
         source_sheet_or_table="SPORA tissues clinical columns", source_field="Sex",
         target_table="sample_metadata", target_field="sex_reported",
         join_keys="tissue_id", value_mapping=json.dumps({"M": "male", "F": "female"}),
         transformation_rule="Map only explicit source sex codes.", time_origin_rule="", conflict_rule="Blank source remains blank.", notes=""),
]
for field in clinical_fields:
    field_rows.append(dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
        source_sheet_or_table="SPORA tissues clinical columns", source_field=field,
        target_table="sample_metadata", target_field="source_" + field.lower(),
        join_keys="tissue_id", value_mapping="", transformation_rule="Preserve the published source value as text; no outcome or timing inference.",
        time_origin_rule="Source time origin or unit is not specified here." if field in {"Age_last_alive", "Survival_from_Rx_Start"} else "",
        conflict_rule="Blank source remains blank.",
        notes=json.dumps({"data_type": "string", "definition": f"Published SPORA tissues.{field} source annotation."})))
field_rows.append(dict(study=STUDY, source_file=ZENODO,
    source_sheet_or_table="original raw release description", source_field="one directory per sample; Supplementary Table S1 reference",
    target_table="sample_metadata", target_field="sample_id", join_keys="original sample directory -> SPORA tissue_id",
    value_mapping="", transformation_rule="The original release describes one directory per sample; use each independent SPORA tissue_id as one source sample pending original-archive crosswalk.", time_origin_rule="",
    conflict_rule="Do not infer absent patient metadata from release directory names.",
    notes="Public release inspected; only a 1.2 GB image archive is listed, no separate small sample table. Local SPORA Sample_ID links 72/90 patient records."))
field_columns = pd.read_csv(TEMPLATES / "field_mapping.csv", nrows=0).columns
field_mapping = pd.DataFrame(field_rows, columns=field_columns)

image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv", nrows=0).columns.tolist()
with formal_read_lease(OUTPUT / "images.parquet"):
    images = (pd.read_parquet(OUTPUT / "images.parquet").to_dict("records")
              if (OUTPUT / "images.parquet").is_file() else [])
completed = {row["image_id"] for row in images}
assert completed <= set(verified)
excluded = []


def save_progress(stage: str, current: str = "") -> None:
    publish_json(JOB / "progress.json", dict(study=STUDY, stage=stage,
        expected_images=90, exported_images=len(images), excluded_images=len(excluded),
        sample_rows=len(samples), sample_linked_images=len(images),
        patient_linked_samples=int(samples.patient_key.ne("").sum()),
        next_image=next((name for name in sorted(verified) if name not in completed), None),
        current_image=current))


def header_matches(path: Path, shape: tuple[int, int], dtype: str, names: list[str]) -> bool:
    with formal_read_lease(path):
        with tifffile.TiffFile(path) as tif:
            channels = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                        if node.tag.endswith("}Channel")]
            return (len(tif.pages) == len(names) and tif.pages[0].shape == shape
                    and tif.pages[0].dtype == np.dtype(dtype) and channels == names)


# Prior indexed images were reopened and checked during the completed batch;
# this continuation only reads and publishes images missing from the index.

save_progress("checking_sources")
for tissue in tissues.sort_values("tissue_id").itertuples(index=False):
    image_id = tissue.tissue_id
    if image_id in completed:
        continue
    record = verified[image_id]
    source_path = RAW / "imc/1_0mpp/images" / f"{image_id}.ome.zarr"
    objects = object_paths[image_id]
    if (len(objects) != record["source_object_count"] or
            any(not path.is_file() or path.stat().st_size != size for path, size in objects)):
        raise ValueError(f"Incomplete source objects: {image_id}")
    names = ordered[image_id]
    if (names != record["canonical_channel_labels"] or not record["exact_canonical_order_checked"]
            or not record["all_source_sizes_match"] or record["dimension_names"] != ["c", "y", "x"]):
        raise ValueError(f"Source proof differs: {image_id}")
    panel_id = f"spora_imc_{len(names)}"
    selected = panels[panel_id].loc[panels[panel_id].keep_channel].sort_values("output_channel_index")
    with formal_read_lease(source_path):
        group = json.loads((source_path / "zarr.json").read_text())["attributes"]["ome"]
        labels = [item["label"] for item in group["omero"]["channels"]]
        array = zarr.open_array(str(source_path / "0"), mode="r")
        if (labels != names or list(array.shape) != record["shape"] or
                str(array.dtype) != record["dtype"]):
            raise ValueError(f"Current Zarr header differs: {image_id}")
        pixels = array[:]
        if hasattr(array.store, "close"):
            array.store.close()
    height, width = pixels.shape[-2:]
    side = max(height, width)
    source = SourceSpec(image_id, panel_id, "zarr", source_path / "0",
        tuple(range(len(names))), pixel_size_um=(1.0, 1.0),
        source_revision="SPORA BIA-SPORA-020 native verification 2026-10-18",
        intensity_description="Native float64 SPORA IMC values; no normalization")
    destination = OUTPUT / "images" / image_id / "input.tiff"
    save_progress("exporting", image_id)
    result = export_input(DecodedROI(pixels, source), panels[panel_id], destination, tile_size=256)
    del pixels
    if not result["admitted"]:
        excluded.append(image_id)
        save_progress("excluded", image_id)
        continue
    if not header_matches(destination, (side, side), "float64", selected.marker_id.tolist()):
        raise ValueError(f"Published TIFF header differs: {image_id}")
    image = {column: None for column in image_columns}
    image.update(study=STUDY, image_id=image_id, sample_id=image_id, panel_id=panel_id,
        image_kind="roi", source_image_id=image_id, source_files=[relative(source_path)],
        source_format="zarr", source_axes="CYX", source_dtype="float64",
        source_revision=source.source_revision, roi_id=image_id,
        region_annotation=tissue.original_file_name,
        source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=side, width_px=side, source_channel_count=len(names),
        channel_count=len(selected), pixel_size_x_um=1.0, pixel_size_y_um=1.0,
        input_image_path=relative(destination), intensity_description=source.intensity_description,
        split_method="already_independent",
        split_evidence="SPORA ROI store and original release sample directory are independent frames.",
        metadata_notes=("Patient linked by source Sample_ID." if pd.notna(tissue.Sample_ID)
                        else "Source Sample_ID absent; patient link unresolved."))
    images.append(image)
    completed.add(image_id)
    save_input_tables(OUTPUT, pd.DataFrame(images, columns=image_columns), samples, panel, field_mapping)
    save_progress("exported", image_id)
    print(f"exported {len(images)}/90 {image_id}", flush=True)

save_input_tables(OUTPUT, pd.DataFrame(images, columns=image_columns), samples, panel, field_mapping)

# These are preparer findings; an independent task owns final review and D01.
review_path = OUTPUT / "review_results.csv"
checks = [
    ("S01", "pass", f"90 verified source stores reconciled; {len(images)} admitted, {len(excluded)} marker-zero exclusions."),
    ("P01", "pass", "Two source panel orders matched the native proof, embedded labels and study S1."),
    ("P02", "pass", "S1-admitted measurements and selected DNA inputs retained in source order."),
    ("P03", "pass", f"{len(images)} TIFF headers reopened for dimensions, dtype and OME channel order."),
    ("M01", "unresolved", "90 source samples link to 90 images; public code ST1 confirms 72 patient links and has no matching record for the other 18."),
    ("D01", "unresolved", "Preparer delivery closed; independent review and binding pending."),
]
new_review = pd.DataFrame([dict(study=STUDY, preparer=Path(__file__).name,
    reviewer="", scope="study", record_id=f"{STUDY}:preparer_delivery",
    check_id=check, result=outcome, source_evidence=relative(PROOF),
    output_evidence=relative(OUTPUT), details=details) for check, outcome, details in checks])
with formal_publish_lease(review_path, directory=OUTPUT):
    prior = pd.read_csv(review_path, keep_default_na=False) if review_path.is_file() else new_review.iloc[:0]
    prior = prior[prior.record_id != f"{STUDY}:preparer_delivery"]
    combined = pd.concat([prior, new_review], ignore_index=True)
    temporary = review_path.with_name(review_path.name + ".partial")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, review_path)
save_progress("prepared")
