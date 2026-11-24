"""Prepare Meyer2025 IMC TMA cores from complete local SPORA OME-Zarr.

Inputs: verified 450 source objects, tissue/channel tables, and study S1.
Outputs: native decoded float32 CYX input TIFFs and study input tables.
Each source tissue is one sample row; patient-level metadata is joined by patient_id.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

pa.set_cpu_count(2)
pa.set_io_thread_count(2)
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)

STUDY = "Meyer2025_IMC_BreastCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "spora"
OUTPUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
JOB = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_JOB",
    str(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0265_prepare_0"),
))
META = metadata_source_directory(NAS, STUDY)
VERIFICATION = META / "native_verification.json"
OBJECTS = META / "imc_source_objects.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
DISCOVERY_METADATA_CSV = RAW.parent / "metadata/TNBC_discovery_metadata.csv"
PANEL_ID = "spora_imc_37"
PREPARER = JOB.name


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def template_columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def field(source: Path, source_field: str, target_table: str, target_field: str,
          rule: str, note: str = "", join: str = "tissue_id") -> dict:
    row = dict.fromkeys(template_columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(source), source_sheet_or_table=source.name,
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join, transformation_rule=rule,
               conflict_rule="Use one unique nonmissing value at the documented patient or tissue level; conflicts remain unassigned",
               notes=note)
    return row


def extension_field(source: Path, source_field: str, target_field: str, rule: str,
                    data_type: str, definition: str, *, join: str = "tissue_id",
                    unit: str = "", time_origin: str = "") -> dict:
    note = dict(data_type=data_type, definition=definition)
    if unit:
        note["unit"] = unit
    if time_origin:
        note["time_origin"] = time_origin
    return field(source, source_field, "sample_metadata", target_field, rule,
                 json.dumps(note), join)


def one_value(group: pd.DataFrame, name: str) -> object:
    values = group[name].dropna().unique()
    if len(values) != 1:
        return ""
    value = values[0]
    return value.item() if isinstance(value, np.generic) else value


class NativeCore:
    """Decode one verified sharded core once; retain its full measured frame."""

    def __init__(self, proof: dict):
        source_path = NAS / proof["path"] / "0"
        with formal_read_lease(source_path, data_root=NAS):
            array = zarr.open_array(str(source_path), mode="r")
            self.values = np.asarray(array[:])
            if hasattr(array.store, "close"):
                array.store.close()
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        assert list(self.shape) == proof["shape"] and self.dtype == np.dtype("float32")
        # The source read is complete; export_input only needs a local path for
        # lease classification and the measured pixel scale for TIFF metadata.
        self.source = SimpleNamespace(path=Path("/tmp/study_input_batch"),
                                      pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def check_output(reader: NativeCore, destination: Path, panel: pd.DataFrame,
                 *, compare_pixels: bool) -> None:
    """Check every header and compare all retained pixels on sampled cores."""
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    _, height, width = reader.shape
    side = max(height, width)
    left, top = (side-width)//2, (side-height)//2
    with formal_read_lease(destination, data_root=NAS):
        with tifffile.TiffFile(destination) as tif:
            assert tif.series[0].shape == (len(kept), side, side)
            assert tif.series[0].axes == "CYX" and tif.series[0].dtype == reader.dtype
            names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter()
                     if node.tag.endswith("}Channel")]
            assert names == kept.marker_id.tolist()
            if compare_pixels:
                actual = tif.asarray(maxworkers=2)
                for channel in kept.itertuples(index=False):
                    plane = actual[int(channel.output_channel_index)]
                    expected = reader.values[int(channel.source_channel_index)]
                    assert np.array_equal(plane[top:top+height, left:left+width], expected)
                    assert not plane[:top].any() and not plane[top+height:].any()
                    assert not plane[:, :left].any() and not plane[:, left+width:].any()


def prepare_metadata(proofs: dict[str, dict], tissues: pd.DataFrame, panel: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tissue_path = RAW / "metadata/tissues.parquet"
    evidence_path = OUTPUT / "source_evidence.json"
    discovery = pd.read_csv(DISCOVERY_METADATA_CSV, dtype={"PID": "string"})
    assert discovery.PID.is_unique and len(discovery) == 235
    discovery_by_pid = discovery.set_index("PID", verify_integrity=True)
    assert tissues.PID.astype(str).isin(discovery_by_pid.index).all()
    patient_groups = {patient: group for patient, group in tissues.groupby("patient_id", sort=True)}

    tissue_special = {
        "age", "grade", "pT", "pN", "pM", "OS_months", "DFS_months",
        "status_OS", "status_DFS", "PID", "pretreatment", "tumor_size_cm",
    }
    tissue_identity = {
        "tissue_id", "patient_id", "specimen_id", "sample_id", "original_patient_identifier",
        "original_file_name", "TMANr", "height_px", "width_px", "alignment",
    }
    tissue_extensions: dict[str, tuple[str, str]] = {}
    for name in tissues.columns:
        if name in tissue_special | tissue_identity or not tissues[name].notna().any():
            continue
        dtype = tissues[name].dtype
        data_type = ("boolean" if pd.api.types.is_bool_dtype(dtype) else
                     "integer" if pd.api.types.is_integer_dtype(dtype) else
                     "number" if pd.api.types.is_float_dtype(dtype) else "string")
        tissue_extensions[name] = (f"source_{name.lower()}", data_type)

    discovery_extensions: dict[str, tuple[str, str]] = {}
    for name in discovery.columns:
        if name == "PID":
            continue
        dtype = discovery[name].dtype
        data_type = ("boolean" if pd.api.types.is_bool_dtype(dtype) else
                     "integer" if pd.api.types.is_integer_dtype(dtype) else
                     "number" if pd.api.types.is_float_dtype(dtype) else "string")
        discovery_extensions[name] = (f"source_zenodo_{name.lower()}", data_type)

    def scalar(value: object, data_type: str) -> object:
        if pd.isna(value):
            return ""
        if isinstance(value, np.generic):
            value = value.item()
        return str(value) if data_type == "string" else value

    images = []
    for row in tissues.itertuples(index=False):
        proof = proofs[row.tissue_id]
        _, height, width = proof["shape"]
        side = max(height, width)
        left, top = (side-width)//2, (side-height)//2
        image = dict.fromkeys(template_columns("image_metadata"))
        image.update(study=STUDY, image_id=row.tissue_id,
                     sample_id=row.tissue_id, panel_id=PANEL_ID,
                     image_kind="tma_core", source_image_id=row.tissue_id,
                     source_files=[proof["path"]], source_format="ome-zarr-v3",
                     source_axes="CYX", source_dtype="float32", source_revision="spora 1_0mpp",
                     slide_id=row.TMANr, core_id=row.original_file_name,
                     roi_id=row.tissue_id, region_annotation="",
                     source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                     pad_left_px=left, pad_top_px=top,
                     pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                     height_px=side, width_px=side, source_channel_count=37,
                     channel_count=int(panel.keep_channel.sum()), pixel_size_x_um=1.0,
                     pixel_size_y_um=1.0,
                     input_image_path=relative(OUTPUT / "images" / row.tissue_id / "input.tiff"),
                     intensity_description="Decoded SPORA float32 source intensity; no additional scaling or normalization",
                     split_method="already_independent", split_evidence=relative(RAW / "metadata/tissues.parquet"),
                     metadata_notes="One published SPoRA tissue/core; patient_id links this tissue to its patient record")
        images.append(image)

    samples = []
    for row in tissues.itertuples(index=False):
        group = patient_groups[row.patient_id]
        sample = dict.fromkeys(template_columns("sample_metadata"), "")
        patient_id = str(row.patient_id)
        pid = str(row.PID)
        zenodo_row = discovery_by_pid.loc[pid]
        sample.update(study=STUDY, sample_id=row.tissue_id,
                      patient_key=f"meyer2025:{patient_id}",
                      source_subject_id=one_value(group, "original_patient_identifier"),
                      cohort_id="Meyer2025 TNBC TMA", source_sample_id=row.sample_id,
                      biospecimen_group_id=f"meyer2025:{patient_id}:{row.specimen_id}",
                      source_specimen_index=str(row.specimen_id),
                      sample_type="tissue",
                      tissue_type="tumour", anatomical_site="Breast", lesion_type="primary",
                      preservation_method="ffpe", processing_method="TMA; IMC",
                      primary_diagnosis="Triple-negative breast cancer", primary_site="Breast",
                      source_age_years=one_value(group, "age"),
                      grade=scalar(one_value(group, "grade"), "string"),
                      stage_basis="pathological",
                      tnm_t=scalar(one_value(group, "pT"), "string"),
                      tnm_n=scalar(one_value(group, "pN"), "string"),
                      tnm_m=scalar(one_value(group, "pM"), "string"),
                      source_os_months=one_value(group, "OS_months"),
                      source_dfs_months=one_value(group, "DFS_months"),
                      source_status_os=one_value(group, "status_OS"),
                      source_status_dfs=one_value(group, "status_DFS"),
                      source_pm=scalar(row.pM, "string"),
                      source_pid=pid, source_tmanr=one_value(group, "TMANr"),
                      source_pretreatment=one_value(group, "pretreatment"),
                      source_tumor_size_cm=one_value(group, "tumor_size_cm"),
                      metadata_notes=("One SPoRA tissue/core per sample row; patient-level fields join by patient_id. "
                                      "The public discovery CSV contains 235 PID rows, of which 215 map to this cohort. "
                                      "Conflicting patient pM is blank in tnm_m; source pM is retained per tissue. "
                                      "Age and survival time origins are unspecified; survival source fields are retained."))
        row_values = row._asdict()
        for source_field, (target_field, data_type) in tissue_extensions.items():
            sample[target_field] = scalar(row_values[source_field], data_type)
        for source_field, (target_field, data_type) in discovery_extensions.items():
            sample[target_field] = scalar(zenodo_row[source_field], data_type)
        samples.append(sample)

    mappings = [
        field(tissue_path, "tissue_id; patient_id; specimen_id; original_file_name; TMANr; height_px; width_px",
              "image_metadata", "image_id; sample_id; slide_id; core_id; source_x1_px; source_y1_px",
              "Exact tissue_id identifies the source tissue/core; patient_id supplies the patient link",
              "450 source tissues map to 450 samples and 215 patient keys"),
        field(tissue_path, "tissue_id", "sample_metadata", "sample_id",
              "Use the exact unique SPoRA tissue_id as the project sample key", join="tissue_id"),
        field(tissue_path, "patient_id", "sample_metadata", "patient_key",
              "Namespace the explicit SPoRA patient_id", join="patient_id"),
        field(tissue_path, "original_patient_identifier", "sample_metadata", "source_subject_id",
              "Preserve the source patient label without decoding or normalization", join="patient_id"),
        field(tissue_path, "sample_id", "sample_metadata", "source_sample_id",
              "Copy the unique source TMA core ID to its tissue sample row", join="tissue_id"),
        field(tissue_path, "specimen_id", "sample_metadata", "biospecimen_group_id",
              "Combine patient_id and its per-patient specimen index using the SPoRA tissue identity rule",
              "Same tissue IDs across modalities denote the same registered physical tissue", "tissue_id"),
        extension_field(tissue_path, "specimen_id", "source_specimen_index",
                        "Preserve the source per-patient specimen index", "string",
                        "SPoRA specimen index for this source tissue", join="tissue_id"),
        extension_field(tissue_path, "age", "source_age_years",
                        "Copy the unique nonmissing patient age to each linked tissue record",
                        "number", "Source patient age; collection time is unspecified",
                        join="patient_id", unit="years", time_origin="unconfirmed"),
        field(tissue_path, "age", "sample_metadata", "age_at_collection_years",
              "Leave blank because the source does not define age relative to tissue collection",
              "Source age is retained in source_age_years", "patient_id"),
        field(tissue_path, "grade", "sample_metadata", "grade",
              "Copy the unique nonmissing patient value", join="patient_id"),
        field(tissue_path, "pT; pN; pM", "sample_metadata", "stage_basis",
              "Set pathological from the explicit pT/pN/pM source fields; do not infer a stage group",
              "The source does not publish a clinical-stage alternative", "patient_id"),
        field(tissue_path, "pT", "sample_metadata", "tnm_t",
              "Copy the unique nonmissing pathological T value", join="patient_id"),
        field(tissue_path, "pN", "sample_metadata", "tnm_n",
              "Copy the unique nonmissing pathological N value", join="patient_id"),
        field(tissue_path, "pM", "sample_metadata", "tnm_m",
              "Copy the unique nonmissing patient value; leave blank for the two conflicting patient groups",
              "Raw source pM is preserved separately on each tissue row", "patient_id"),
        extension_field(tissue_path, "pM", "source_pm",
                        "Preserve the raw source pM value on each tissue row",
                        "string", "Raw SPoRA pM value without conflict resolution"),
        extension_field(tissue_path, "OS_months", "source_os_months",
                        "Copy the unique nonmissing patient value without endpoint reinterpretation",
                        "number", "Source overall survival duration; time origin is unspecified",
                        join="patient_id", unit="months", time_origin="unconfirmed"),
        extension_field(tissue_path, "DFS_months", "source_dfs_months",
                        "Copy the unique nonmissing patient value without endpoint reinterpretation",
                        "number", "Source disease-free survival duration; time origin is unspecified",
                        join="patient_id", unit="months", time_origin="unconfirmed"),
        extension_field(tissue_path, "status_OS", "source_status_os",
                        "Copy the unique nonmissing patient value without endpoint reinterpretation",
                        "number", "Source OS event indicator; author analysis uses 0=censored and 1=event",
                        join="patient_id", time_origin="unconfirmed"),
        extension_field(tissue_path, "status_DFS", "source_status_dfs",
                        "Copy the unique nonmissing patient value without endpoint reinterpretation",
                        "number", "Source DFS event indicator; author analysis uses 0=censored and 1=event",
                        join="patient_id", time_origin="unconfirmed"),
        extension_field(tissue_path, "PID", "source_pid",
                        "Preserve the source PID label", "string",
                        "Source patient identifier; one-to-one with patient_id", join="patient_id"),
        extension_field(tissue_path, "TMANr", "source_tmanr",
                        "Copy the unique source TMA label", "string",
                        "Source TMA label; patient-level value", join="patient_id"),
        extension_field(tissue_path, "pretreatment", "source_pretreatment",
                        "Copy the unique nonmissing patient value", "boolean",
                        "Raw source pretreatment flag", join="patient_id"),
        extension_field(tissue_path, "tumor_size_cm", "source_tumor_size_cm",
                        "Copy the unique nonmissing patient value to each linked tissue record",
                        "number", "Source tumor size; no further clinical interpretation",
                        join="patient_id", unit="cm"),
        field(DISCOVERY_METADATA_CSV, "PID", "sample_metadata", "patient_key",
              "Join the public discovery table to SPoRA patient records by exact PID",
              "215 of 235 public PID rows match this cohort", "PID"),
        field(tissue_path, "TMANr", "image_metadata", "slide_id",
              "Copy the source TMA label to the image row", join="tissue_id"),
        field(tissue_path, "original_file_name", "image_metadata", "core_id",
              "Copy the source image filename to the core identity field", join="tissue_id"),
        field(tissue_path, "height_px; width_px", "image_metadata", "source_y1_px; source_x1_px",
              "Copy source dimensions to the recorded full-core bounds", join="tissue_id"),
        field(DISCOVERY_METADATA_CSV, "TMANr; n_samples; grade; pT_simple; pN_simple; pM_simple; OS_data; DFS_data; age_bin; pretreatment",
              "sample_metadata", "; ".join(target for target, _ in discovery_extensions.values()),
              "Join exact PID to the SPoRA patient; preserve each public CSV field in a release-namespaced extension",
              "235 unique PID rows; 215 match this cohort and 20 lack a source tissue match. TMANr and age_bin differences are preserved by release; OS_data/DFS_data flags are not interpreted as outcomes",
              "PID"),
        field(evidence_path, "official publication; author SPoRA dataset; public release and table comparisons",
              "sample_metadata", "study; cohort_id; sample_type; tissue_type; anatomical_site; lesion_type; preservation_method; processing_method; primary_diagnosis; primary_site; metadata_notes",
              "Assign documented study/cohort and tissue attributes; describe source identity, joins, conflicts, and unknown time origins",
              "Author dataset describes an IMC TMA with multiple FFPE cores from 215 TNBC patients"),
        field(RAW / "imc/channels.parquet", "channel_name; original_channel_name; qc_pass",
              "panel_mapping", "source_name; source_channel_index; keep_channel",
              "37 source rows in published channel order; 33 S1-admitted channels", join="source_channel_index"),
        field(S1, "Study; Original marker; Unified marker; Segmentation channel", "panel_mapping",
              "marker_id; role; nuclear; body; output_channel_index",
              "Exact study/source-name join; retain S1 nuclear and body channels", join="Study + Original marker"),
        field(VERIFICATION, "shape; dtype; canonical_channel_labels; decoded source probe", "image_metadata",
              "source_files; source_format; source_axes; source_dtype; source_revision; source_x0_px; source_y0_px; source_x1_px; source_y1_px; pad_left_px; pad_top_px; pad_right_px; pad_bottom_px; height_px; width_px; source_channel_count; channel_count; input_image_path; intensity_description; split_method; split_evidence",
              "Full verified core; center-pad to square; retain decoded float32; 1 um/pixel"),
    ]
    for source_field, (target_field, data_type) in tissue_extensions.items():
        mappings.append(extension_field(tissue_path, source_field, target_field,
            "Preserve the source value on its exact tissue record without reinterpretation",
            data_type, f"Raw SPoRA tissues.parquet value for {source_field}"))
    for source_field, (target_field, data_type) in discovery_extensions.items():
        mappings.append(extension_field(DISCOVERY_METADATA_CSV, source_field, target_field,
            "Join by exact PID and preserve the public release value without reinterpretation",
            data_type, f"Raw Zenodo v3 TNBC_discovery_metadata.csv value for {source_field}",
            join="PID"))
    return pd.DataFrame(images), pd.DataFrame(samples), pd.DataFrame(mappings)


def progress(stage: str, exported: int, **extra: object) -> None:
    write_json(JOB / "progress.json", dict(study=STUDY, phase="prepare", repair=0,
        stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), expected_images=450,
        exported_images=exported, sample_rows=450, sample_linked_images=exported,
        patient_linked_samples=450, unique_patients=215, **extra))


def preparer_review(exported: int, excluded: int, pixel_checked: int) -> None:
    checks = [
        ("S01", "pass" if exported+excluded == 450 else "unresolved", "450 locally verified cores; export and excluded counts in progress.json"),
        ("I01", "pass" if exported else "unresolved", "Published image keys and source paths are in images.parquet"),
        ("I02", "pass" if exported else "unresolved", "Source patient_id joins exported images to tumour samples"),
        ("I03", "pass" if exported else "unresolved", "Full core frames centered and padded to square"),
        ("I04", "pass" if exported else "unresolved", "Each source object is one independent TMA core"),
        ("I05", "pass" if pixel_checked else "unresolved", f"{pixel_checked} selected cores compared at every pixel of all 33 retained channels, including padding"),
        ("P01", "pass", "37 source channels checked against accepted order proof"),
        ("P02", "pass", "33 retained channels from study S1; excluded channels remain in panel table"),
        ("P03", "pass" if exported else "unresolved", "Each TIFF reopened for CYX shape, float32 dtype and ordered names"),
        ("M01", "pass", "450 SPoRA tissue IDs map to 450 sample rows and 450 image foreign keys; 215 patient keys"),
        ("M02", "unresolved", "Source fields and release-specific disagreements are retained; patient pM conflicts remain blank in the canonical field; age and survival time origins are unspecified"),
        ("D01", "unresolved", "Preparer self-check only; independent reviewer and controller acceptance pending"),
    ]
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                 record_id=STUDY, check_id=check, result=result,
                 source_evidence=relative(OUTPUT / "source_evidence.json"),
                 output_evidence=relative(OUTPUT),
                 details=json.dumps(dict(assessment="preparer_self_check", text=detail,
                                         exported_images=exported, excluded_images=excluded), ensure_ascii=False))
            for check, result, detail in checks]
    target = OUTPUT / "review_results.csv"
    with formal_publish_lease(target, directory=OUTPUT, data_root=NAS):
        pending = target.with_name(target.name + ".partial")
        if target.is_file():
            previous = pd.read_csv(target, dtype=str, keep_default_na=False)
            previous = previous[~((previous.preparer == PREPARER) &
                                  (previous.reviewer == ""))]
            rows = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True).to_dict("records")
        pd.DataFrame(rows).to_csv(pending, index=False, encoding="utf-8-sig")
        pending.replace(target)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--limit", type=int, help="Maximum additional source images to process")
parser.add_argument("--yield-io", action="store_true", help="Record I/O wait from saved per-image checkpoint")
parser.add_argument("--metadata-only", action="store_true",
                    help="Refresh source metadata tables without reading or writing image payloads")
args = parser.parse_args()
if args.metadata_only and (args.limit is not None or args.yield_io):
    parser.error("--metadata-only cannot be combined with --limit or --yield-io")
JOB.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

if args.yield_io:
    with formal_read_lease(VERIFICATION, data_root=NAS):
        verified = json.loads(VERIFICATION.read_text())["verified"]
    receipts = json.loads((CACHE / "export_receipts.json").read_text())
    exported = sum(item["admitted"] for item in receipts.values())
    remaining_ids = sorted(item["image_id"] for item in verified
                           if item["image_id"] not in receipts)
    preparer_review(exported, len(receipts)-exported,
                    sum(item.get("full_pixels_checked", False) for item in receipts.values()))
    progress("waiting_io", exported, processed_sources=len(receipts),
             excluded_sources=len(receipts)-exported,
             next_image=remaining_ids[0] if remaining_ids else None,
             issue="Shared NAS read/write admission is temporarily congested; resume unprocessed source objects")
    raise SystemExit(0)

# Reuse complete collection proof and check that each local object remains present.
with formal_read_lease(VERIFICATION, data_root=NAS):
    verification = json.loads(VERIFICATION.read_text())
    objects = json.loads(OBJECTS.read_text())
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    channels = pd.read_parquet(RAW / "imc/channels.parquet")
    available = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
assert verification["full_source_covered"] and not verification["failed"]
assert verification["verified_count"] == len(verification["verified"]) == 450
proofs = {item["image_id"]: item for item in verification["verified"]}
assert len(proofs) == 450 and set(proofs) == set(tissues.tissue_id)
assert tissues.tissue_id.is_unique and tissues.sample_id.is_unique
assert tissues.patient_id.notna().all() and tissues.original_patient_identifier.notna().all()
assert tissues.groupby("patient_id").original_patient_identifier.nunique().max() == 1
assert tissues.patient_id.nunique() == 215
assert channels.shape[0] == 37 and available.shape == (450, 37)
names = channels.channel_name.tolist()
assert available.columns.tolist() == names and bool(available.to_numpy().all())
assert all(item["exact_canonical_order_checked"] and item["canonical_channel_labels"] == names
           and item["dtype"] == "float32" and item["dimension_names"] == ["c", "y", "x"]
           and item["all_source_sizes_match"] for item in proofs.values())
assert len(objects) == 1351
for item in objects:
    suffix = item["key"].split("spora-submissions/meyer2025stratification/", 1)[1]
    assert (RAW / suffix).stat().st_size == item["size"]

panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
assert int(panel.keep_channel.sum()) == 33
images, samples, fields = prepare_metadata(proofs, tissues, panel)
assert len(samples) == 450
assert samples.sample_id.is_unique and samples.source_sample_id.is_unique
assert samples.patient_key.nunique() == 215
discovery = pd.read_csv(DISCOVERY_METADATA_CSV, dtype={"PID": "string"})
matched_discovery = discovery[discovery.PID.isin(set(tissues.PID.astype(str)))].copy()
tissue_by_pid = {str(pid): group for pid, group in tissues.groupby("PID")}
release_comparisons = {}
for column in ("TMANr", "age_bin"):
    exact = compared = 0
    for source_row in matched_discovery.itertuples(index=False):
        source_value = getattr(source_row, column)
        values = tissue_by_pid[str(source_row.PID)][column].dropna().unique()
        if len(values) == 1 and pd.notna(source_value):
            compared += 1
            exact += int(str(values[0]) == str(source_value))
    release_comparisons[column] = dict(exact=exact, compared=compared,
                                       different=compared-exact)
source_evidence = dict(
    study=STUDY, local_verification=relative(VERIFICATION), local_objects=relative(OBJECTS),
    source_tissues=relative(RAW / "metadata/tissues.parquet"),
    source_channels=relative(RAW / "imc/channels.parquet"),
    discovery_metadata=relative(DISCOVERY_METADATA_CSV),
    public_sources=["https://pubmed.ncbi.nlm.nih.gov/40680742/",
                    "https://bodenmillergroup.github.io/imcdatasets/reference/Meyer_2025_TripleNegativeBreastCancer.html",
                    "https://zenodo.org/records/20414755",
                    "https://spora.epfl.ch/docs/io/format.html",
                    "https://github.com/BodenmillerGroup/TNBC_publication/blob/main/pipelines/discovery/07_combined_tumor_TME.Rmd",
                    "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html"],
    sample_identity=("SPoRA tissues.parquet has one row per tissue-modality pair; tissue_id identifies the tissue, "
                     "patient_id is the pseudonymous patient key, and specimen_id is a per-patient specimen index. "
                     "Images are separate only when the SPoRA tissue IDs differ."),
    sample_scope=dict(tissue_rows=450, unique_patient_ids=215, unique_source_sample_ids=450,
                      specimen_count_distribution={str(k): int(v) for k, v in
                          tissues.groupby("patient_id").specimen_id.nunique().value_counts().sort_index().items()}),
    discovery_release=dict(rows=len(discovery), unique_pid=int(discovery.PID.nunique()),
                           matched_current_patients=len(matched_discovery),
                           unmatched_current_cohort_rows=len(discovery)-len(matched_discovery),
                           cross_release_comparisons=release_comparisons),
    all_missing_tissue_columns=tissues.columns[tissues.isna().all()].tolist(),
    grouping="Join each tissue record to one of 215 patients by exact patient_id; retain source sample_id and per-patient specimen_id",
    source_scope="450 complete locally verified SPoRA 1 um OME-Zarr tissue/core objects",
    clinical_gaps="Two patient groups have conflicting pM; age collection time and survival time origin are unspecified",
    survival_status_evidence="Local statuses are 0/1; author analysis passes status_OS and status_DFS to Surv(), where 0 is censored and 1 is an event")
with formal_publish_lease(OUTPUT / "source_evidence.json", directory=OUTPUT, data_root=NAS):
    write_json(OUTPUT / "source_evidence.json", source_evidence)

save_input_tables(OUTPUT, images, samples, panel, fields, data_root=NAS,
                  replace_metadata_studies={STUDY})
if args.metadata_only:
    print(json.dumps(dict(study=STUDY, metadata_only=True, sample_rows=len(samples),
                          unique_patients=int(samples.patient_key.nunique()), image_rows=len(images),
                          image_sample_links=int(images.sample_id.notna().sum()),
                          source_sample_ids=int(samples.source_sample_id.notna().sum())),
                     ensure_ascii=False))
    raise SystemExit(0)

receipt_path = CACHE / "export_receipts.json"
receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
for image_id in list(receipts):
    if receipts[image_id]["admitted"] and not (OUTPUT / "images" / image_id / "input.tiff").is_file():
        receipts.pop(image_id)
if receipts:
    assert set(receipts).issubset(proofs)

remaining = images[~images.image_id.isin(receipts)].sort_values("image_id")
if args.limit is not None:
    remaining = remaining.iloc[:args.limit]
upper_bytes = sum(max(item["shape"][1:])**2 * 33 * 4 for item in proofs.values())
progress("ready", sum(r["admitted"] for r in receipts.values()),
         processed_sources=len(receipts), upper_uncompressed_output_bytes=upper_bytes,
         next_image=remaining.image_id.iloc[0] if len(remaining) else None)

for row in remaining.itertuples(index=False):
    free_nas = shutil.disk_usage(NAS).free
    free_local = shutil.disk_usage("/tmp").free
    if free_nas < 100*1024**3 + 2*max(proofs[row.image_id]["shape"][1:])**2*33*4 or free_local < 64*1024**3 + 2*max(proofs[row.image_id]["shape"][1:])**2*33*4:
        progress("waiting_io", sum(r["admitted"] for r in receipts.values()),
                 processed_sources=len(receipts), next_image=row.image_id,
                 issue="NAS 100 GiB or local 64 GiB reserve unavailable")
        break
    progress("exporting", sum(r["admitted"] for r in receipts.values()),
             processed_sources=len(receipts), current_image=row.image_id)
    reader = NativeCore(proofs[row.image_id])
    destination = NAS / row.input_image_path
    compare_pixels = len(receipts) % 50 == 0 or row.image_id == images.image_id.max()
    if destination.is_file():
        check_output(reader, destination, panel, compare_pixels=compare_pixels)
        result = dict(admitted=True, channel_nonzero_pixels=[
            int(np.count_nonzero(reader.values[int(index)]))
            for index in panel.loc[panel.keep_channel, "source_channel_index"]])
    else:
        result = export_input(reader, panel, destination, data_root=NAS)
        if result["admitted"]:
            check_output(reader, destination, panel, compare_pixels=compare_pixels)
    del reader
    admitted_ids = {key for key, value in receipts.items() if value["admitted"]}
    if result["admitted"]:
        admitted_ids.add(row.image_id)
    save_input_tables(OUTPUT, images[images.image_id.isin(admitted_ids)], samples, panel, fields,
                      data_root=NAS)
    receipts[row.image_id] = dict(admitted=result["admitted"], source_path=proofs[row.image_id]["path"],
                                  output_path=row.input_image_path if result["admitted"] else "",
                                  header_checked=bool(result["admitted"]),
                                  full_pixels_checked=bool(result["admitted"] and compare_pixels),
                                  nonzero_pixels=result["channel_nonzero_pixels"])
    write_json(receipt_path, receipts)
    exported = len(admitted_ids)
    next_images = images[~images.image_id.isin(receipts)].sort_values("image_id")
    progress("exporting", exported, processed_sources=len(receipts),
             excluded_sources=len(receipts)-exported,
             next_image=next_images.image_id.iloc[0] if len(next_images) else None)
    print(json.dumps(dict(image_id=row.image_id, exported_images=exported,
                          processed_sources=len(receipts))), flush=True)

exported = sum(r["admitted"] for r in receipts.values())
pixel_checked = sum(r.get("full_pixels_checked", False) for r in receipts.values())
preparer_review(exported, len(receipts)-exported, pixel_checked)
if len(receipts) == 450:
    progress("prepare_complete", exported, processed_sources=450,
             excluded_sources=450-exported, next_image=None,
             gaps=["Independent review pending", "Two conflicting source pM groups",
                   "Clinical survival time origin unconfirmed"])
