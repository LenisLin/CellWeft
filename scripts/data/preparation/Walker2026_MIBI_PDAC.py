"""Prepare Walker PDAC FOVs from the verified compensated-image release.

Inputs: original Zenodo tar, native member/order inventories, two author metadata
tables and S1. Outputs: native CYX input TIFFs and per-study input tables.
Each FOV is staged locally and published through the shared NAS leases.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

from datetime import datetime, timezone
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import pandas as pd
import numpy as np
import tifffile

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)


STUDY = "Walker2026_MIBI_PDAC"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
ARCHIVE = NAS / "raw/public" / STUDY / "zenodo_21108550/compensated_images.tar"
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0633_prepare_0")))
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TMA = RAW_METADATA / "U54-TMA_Metadata_CRP_CW_231024.csv"
PATIENT = RAW_METADATA / "U54_MIBI_patient_metadata_072924.csv"
ORDER = META / "native_ordered_markers.json"
INVENTORY = META / "native_inventory.json"
PANEL_ID = "compensated_47"
PREPARER = JOB.name
REPAIR_ROUND = int(os.environ.get("CELLWEFT_SCIENTIFIC_REPAIR_ROUND", "0"))


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with formal_publish_lease(path, directory=path.parent, lock_name=".progress.lock"):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)


def progress(phase: str, image_count: int, *, next_fov: str = "", detail: str = "") -> None:
    publish_json(JOB / "progress.json", {
        "study": STUDY, "preparer": PREPARER, "phase": phase,
        "scientific_repair_round": REPAIR_ROUND,
        "expected_images": 58, "completed_images": image_count,
        "sample_rows": 58, "sample_linked_images": image_count,
        "patient_linked_samples": 58, "next_fov": next_fov,
        "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat(),
    })


def ensure_review_header() -> None:
    """Create only the independent review table schema; no preparer pass rows."""
    review = OUTPUT / "review_results.csv"
    if not review.exists():
        with formal_publish_lease(review, directory=OUTPUT):
            if not review.exists():
                temporary = review.with_name(review.name + ".partial")
                pd.DataFrame(columns=columns("review_results")).to_csv(
                    temporary, index=False, encoding="utf-8-sig")
                os.replace(temporary, review)


def field(source: Path, source_field: str, target_table: str, target_field: str,
          join: str, rule: str, notes: str = "") -> dict:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(source),
               source_sheet_or_table=source.name, source_field=source_field,
               target_table=target_table, target_field=target_field,
               join_keys=join, transformation_rule=rule, notes=notes)
    return row


def author_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with formal_read_lease(TMA):
        tma = pd.read_csv(TMA, dtype=str, keep_default_na=False)
    with formal_read_lease(PATIENT):
        patient = pd.read_csv(PATIENT, dtype=str, keep_default_na=False)
    keys = ["fov", "New_Sample_ID", "New_Pt_ID", "Public_Patient_ID"]
    if (len(tma) != 58 or len(patient) != 58 or tma.fov.duplicated().any()
            or patient.fov.duplicated().any()
            or not tma[keys].sort_values("fov").reset_index(drop=True).equals(
                patient[keys].sort_values("fov").reset_index(drop=True))):
        raise ValueError("The two author tables do not identify the same 58 FOV/sample/patient tuples")
    return tma.set_index("fov"), patient.set_index("fov")


def samples_from_author(tma: pd.DataFrame, patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fov, source in tma.iterrows():
        clinical = patient.loc[fov]
        row = dict.fromkeys(columns("sample_metadata"), "")
        tissue = source.Tissue
        source_site = clinical["Primary.Site"]
        if clinical.Diagnosis != "Adenocarcinoma" or source_site not in {"Pancreas", "Ampulla of Vater"}:
            raise ValueError("Unexpected source diagnosis or primary site")
        location = {"PDAC": "pancreas", "Liver_Met": "liver",
                    "Ovary_Met": "ovary", "Peritoneum_Met": "peritoneum",
                    "Lymph node": "lymph node", "Lung_Met": "lung"}[tissue]
        row.update(study=STUDY, sample_id=source.New_Sample_ID,
                   source_sample_id=source.New_Sample_ID,
                   patient_key=source.New_Pt_ID,
                   source_subject_id=source.Public_Patient_ID,
                   sample_type="clinical tissue", tissue_type="tumor",
                   anatomical_site=location,
                   primary_site="pancreas" if source_site == "Pancreas" else source_site,
                   lesion_type="primary" if source.Primary_Mets == "Primary" else "metastasis",
                   primary_diagnosis=("pancreatic ductal adenocarcinoma"
                                      if source_site == "Pancreas" else "adenocarcinoma"),
                   grade=source.Grade if source.Grade not in {"NA", "nan"} else "",
                   sex_reported={"Female": "female", "Male": "male"}.get(clinical.Sex, ""),
                   age_at_diagnosis_years=(clinical["Age.at.Diagnosis"]
                                           if clinical["Age.at.Diagnosis"] not in {"NA", "nan"} else ""),
                   metadata_notes="FOV-to-sample and patient keys agree in both author tables. "
                                  "Source Stage and survival fields retain their published semantics in the author tables; "
                                  "no index event or survival origin is inferred.")
        if source_site == "Ampulla of Vater":
            row["metadata_notes"] += (" TMA Tissue=PDAC is a sampled-tissue label; the patient table "
                                      "identifies the primary and staging site as Ampulla of Vater. "
                                      "Diagnosis is retained at the source specificity of adenocarcinoma.")
        rows.append(row)
    result = pd.DataFrame(rows, columns=columns("sample_metadata"))
    if result.sample_id.duplicated().any() or result.patient_key.eq("").any():
        raise ValueError("Sample or patient identity is incomplete")
    return result


def mappings() -> pd.DataFrame:
    join = "Exact fov in both author tables; New_Sample_ID identifies the biological sample; New_Pt_ID identifies the source patient"
    rows = [
        field(TMA, "fov", "image_metadata", "fov_id", join, "Keep the exact source FOV key"),
        field(TMA, "New_Sample_ID", "image_metadata", "sample_id", join, "Exact sample key; 58/58 FOVs linked"),
        field(TMA, "New_Sample_ID", "sample_metadata", "sample_id", join, "One row per source biological sample; 58 unique samples"),
        field(TMA, "New_Pt_ID", "sample_metadata", "patient_key", join, "Exact source patient key; 58/58 samples linked, 34 distinct keys; independent person count unverified"),
        field(TMA, "Public_Patient_ID", "sample_metadata", "source_subject_id", join, "Retain published de-identified patient identifier"),
        field(TMA, "SampleType", "sample_metadata", "sample_type", join, "Clinical denotes tissue collected in the clinical cohort; record clinical tissue"),
        field(TMA, "Tissue;Primary_Mets", "sample_metadata", "tissue_type", join,
              "All 58 included TMA tissue labels denote primary or metastatic tumor; map to tumor"),
        field(TMA, "Tissue", "sample_metadata", "anatomical_site", join, "Map PDAC and site-specific metastasis labels to anatomical site"),
        field(PATIENT, "Primary.Site;Primary.Site._.Major.Groups.For.Staging", "sample_metadata", "primary_site", join,
              "Pancreas -> pancreas; Ampulla of Vater -> Ampulla of Vater; use patient primary site rather than TMA sampled-tissue label",
              "Both patient source columns agree for the two ampullary samples; TMA Tissue=PDAC is retained as a separate tissue annotation."),
        field(PATIENT, "Diagnosis;Primary.Site", "sample_metadata", "primary_diagnosis", join,
              "Adenocarcinoma with pancreatic primary in the PDAC cohort -> pancreatic ductal adenocarcinoma; "
              "ampullary primary -> adenocarcinoma at source specificity",
              "Do not infer pancreatic primary or ductal subtype for the two ampullary samples."),
        field(TMA, "Primary_Mets", "sample_metadata", "lesion_type", join, "Primary -> primary; Metastasis -> metastasis"),
        field(TMA, "Grade", "sample_metadata", "grade", join, "Retain source grade; missing tokens remain empty"),
        field(PATIENT, "Sex", "sample_metadata", "sex_reported", join, "Female -> female; Male -> male; missing remains empty"),
        field(PATIENT, "Age.at.Diagnosis", "sample_metadata", "age_at_diagnosis_years", join, "Retain numeric years at diagnosis; missing remains empty"),
        field(TMA, "Stage;Days_Survival;Vital_Status", "sample_metadata", "metadata_notes", join,
              "Source clinical fields remain in the author tables; index stage and survival timing are not inferred",
              "Stage origin and Days_Survival time origin require source clarification for time-indexed target fields."),
        field(ORDER, "fovs[*].order;source_name;member", "panel_mapping", "source_channel_index",
              "Exact FOV key to native ordered member list", "Zero-base the verified native source order; all 47 source rows retained in panel"),
        field(INVENTORY, "all_members.name;offset;size", "image_metadata", "source_image_id",
              "Exact archive member path for each FOV/channel", "Use verified original tar member byte ranges; 58 metadata FOVs, excluding 12 extra directories"),
        field(ORDER, "fovs[*].source_name", "image_metadata", "source_dtype",
              "Exact source channel name to indexed TIFF member", "46 float32 planes plus enhanced COL1A1 uint16; uint16 values convert exactly to common float32 output dtype"),
        field(ARCHIVE, "compensated_images/<fov>/<channel>.tiff", "image_metadata", "source_files",
              "Exact FOV and member paths from native_ordered_markers.json", "Original tar remains the source container; local per-image copies are temporary"),
    ]
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def stage_fov(archive, members: dict[str, dict], entries: list[dict], directory: Path) -> list[Path]:
    """Copy one verified native FOV from indexed tar byte ranges into local scratch."""
    paths = []
    for entry in entries:
        member = members[entry["member"]]
        path = directory / f'{entry["order"]:02d}_{entry["source_name"]}.tiff'
        archive.seek(member["offset"])
        with path.open("wb") as output:
            remaining = member["size"]
            while remaining:
                block = archive.read(min(4 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError(f'Incomplete original tar member: {entry["member"]}')
                output.write(block)
                remaining -= len(block)
        paths.append(path)
    return paths


def local_archive_copy() -> Path:
    """Resume one sequential NAS read into local scratch for indexed FOV staging."""
    local = TEMP / ARCHIVE.name
    if local.is_file() and local.stat().st_size == ARCHIVE.stat().st_size:
        return local
    partial = local.with_name(local.name + ".partial")
    copied = partial.stat().st_size if partial.exists() else 0
    total = ARCHIVE.stat().st_size
    if copied > total:
        raise ValueError("Local archive partial exceeds original source size")
    if shutil.disk_usage(TEMP).free - (total - copied) < 64 * 1024**3:
        raise OSError("Local scratch cannot retain 64 GiB after archive staging")
    with formal_read_lease(ARCHIVE), ARCHIVE.open("rb") as source, partial.open("ab") as target:
        source.seek(copied)
        while copied < total:
            block = source.read(min(8 * 1024 * 1024, total - copied))
            if not block:
                raise EOFError("Original archive ended during local staging")
            target.write(block)
            copied += len(block)
    os.replace(partial, local)
    return local


class MixedPlanesReader:
    """Expose source float32 planes and exact uint16 COL1A1 values as float32 CYX."""

    def __init__(self, paths: list[Path], names: list[str]):
        self.stack = ExitStack()
        try:
            self.tiffs = [self.stack.enter_context(tifffile.TiffFile(path)) for path in paths]
            pages = [item.pages[0] for item in self.tiffs]
            native = pages[0].shape
            if len(native) != 2 or any(page.shape != native for page in pages):
                raise ValueError("Source FOV planes do not share native geometry")
            for name, page in zip(names, pages):
                expected = np.dtype("uint16" if name == "COL1A1" else "float32")
                if page.dtype != expected:
                    raise ValueError(f"Source FOV plane dtype differs from verified layout: {name}")
        except Exception:
            self.stack.close()
            raise
        self.shape = (len(paths), *native)
        self.dtype = np.dtype("float32")
        self.source = SimpleNamespace(path=paths[0], pixel_size_um=None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        return np.stack([read_tiff_window(self.tiffs[index], 0, window).astype(np.float32, copy=False)
                         for index in channels])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def check_input(path: Path, panel: pd.DataFrame, shape: tuple[int, int, int], dtype: str) -> None:
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    with formal_read_lease(path), tifffile.TiffFile(path) as image:
        if image.series[0].shape != shape or str(image.pages[0].dtype) != dtype:
            raise ValueError(f"Published input has incorrect native shape/dtype: {path}")
        root = ElementTree.fromstring(image.ome_metadata)
        names = [element.attrib["Name"] for element in root.iter() if element.tag.endswith("}Channel")]
        if names != kept.marker_id.tolist():
            raise ValueError(f"Published input channel names differ from S1 panel: {path}")


def image_row(fov: str, source: pd.Series, shape: tuple[int, int, int],
              dtype: str, result: dict, output_path: Path) -> dict:
    _, height, width = shape
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=fov, sample_id=source.New_Sample_ID,
               panel_id=PANEL_ID, image_kind="fov", source_image_id=f"compensated_images/{fov}",
               source_files=[relative(ARCHIVE)], source_format="tiff_planes_in_tar",
               source_axes="single-channel YX TIFF planes; ordered CYX FOV", source_dtype=dtype,
               source_revision="zenodo_21108550:compensated_images.tar",
               fov_id=fov, region_annotation=source.Tissue,
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               input_image_path=relative(output_path), valid_pixels_path=None,
               intensity_description="Published compensated per-channel TIFF intensities; COL1A1 uses the author's enhanced TIFF; no new normalization",
               split_method="already_independent",
               split_evidence=f"{relative(TMA)}:fov={fov}; {relative(ORDER)}:fovs.{fov}",
               metadata_notes="Pixel size was not stated in the verified native release; source FOV is kept whole. "
                              "COL1A1 source plane is uint16 and is exactly represented in float32 alongside 46 native float32 planes.")
    row.update({key: result[key] for key in ("height_px", "width_px", "pad_left_px", "pad_top_px",
                                                "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count")})
    return row


# Reconcile source inventories, patient joins, and S1 before creating an output.
with formal_read_lease(ORDER):
    ordered = json.loads(ORDER.read_text())
with formal_read_lease(INVENTORY):
    inventory = json.loads(INVENTORY.read_text())
tma, patient = author_tables()
fovs = sorted(ordered["fovs"], key=lambda key: tuple(map(int, key.split("_"))))
if set(fovs) != set(tma.index) or set(fovs) != set(inventory["fovs"]):
    raise ValueError("Native FOV inventory differs from the author sample tables")
archive_stat = ARCHIVE.stat()
if archive_stat.st_size != ordered["file_stat"]["size"] or archive_stat.st_mtime_ns != ordered["file_stat"]["mtime_ns"]:
    raise ValueError("Source archive changed after native verification")
members = {entry["name"]: entry for entry in inventory["all_members"]}
names = [entry["source_name"] for entry in ordered["fovs"][fovs[0]]]
if any([entry["source_name"] for entry in ordered["fovs"][fov]] != names for fov in fovs):
    raise ValueError("FOV channel orders differ; define separate source panels")
panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
samples = samples_from_author(tma, patient)
field_mapping = mappings()
TEMP.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
ensure_review_header()

with formal_read_lease(OUTPUT):
    existing = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
records = {row.image_id: row._asdict() for row in existing.itertuples(index=False)}

if os.environ.get("CELLWEFT_INPUT_REPAIR_ONLY") == "1":
    if len(records) != len(fovs):
        raise ValueError("Metadata-only repair requires the complete existing image index")
    progress("repairing_metadata", len(records), detail="Repairing two clinical rows and three source rules")
    save_input_tables(OUTPUT, existing, samples, panel, field_mapping, data_root=NAS)
    progress("prepared_pending_independent_review", len(records),
             detail="M02 metadata repair published; independent recheck and D01 rebinding pending")
    raise SystemExit(0)

progress("exporting", len(records), next_fov=next((fov for fov in fovs if fov not in records), ""))
for fov in fovs:
    if fov not in records:
        continue
    destination = OUTPUT / "images" / fov / "input.tiff"
    if not destination.is_file() or not destination.stat().st_size:
        raise ValueError(f"Indexed FOV lacks its completed input: {fov}")
    check_input(destination, panel,
                (int(records[fov]["channel_count"]), int(records[fov]["height_px"]),
                 int(records[fov]["width_px"])), str(records[fov]["source_dtype"]))

pending = [fov for fov in fovs if fov not in records]
if pending:
    progress("staging_source", len(records), next_fov=pending[0],
             detail="Sequential copy of the verified original tar to local scratch; partial copy is resumable")
    try:
        local_archive = local_archive_copy()
    except (KeyboardInterrupt, OSError):
        partial = TEMP / (ARCHIVE.name + ".partial")
        copied = partial.stat().st_size if partial.exists() else 0
        progress("waiting_io", len(records), next_fov=pending[0],
                 detail=f"Resumable local source copy: {copied}/{ARCHIVE.stat().st_size} bytes")
        raise
    progress("exporting", len(records), next_fov=pending[0])
for start in range(0, len(pending), 4):
    batch = pending[start:start+4]
    with tempfile.TemporaryDirectory(prefix="walker-batch-", dir=TEMP) as temporary:
        scratch = Path(temporary)
        staged = {}
        with local_archive.open("rb") as archive:
            for fov in batch:
                directory = scratch / fov
                directory.mkdir()
                staged[fov] = stage_fov(archive, members, ordered["fovs"][fov], directory)
        for fov in batch:
            destination = OUTPUT / "images" / fov / "input.tiff"
            with MixedPlanesReader(staged[fov], names) as reader:
                shape, dtype = reader.shape, str(reader.dtype)
                result = export_input(reader, panel, destination, data_root=NAS)
            if not result["admitted"]:
                raise ValueError(f"FOV has no nonzero admitted marker signal: {fov}")
            check_input(destination, panel,
                        (result["channel_count"], result["height_px"], result["width_px"]), dtype)
            row = image_row(fov, tma.loc[fov], shape, dtype, result, destination)
            save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panel, field_mapping, data_root=NAS)
            records[fov] = row
            remaining = next((item for item in fovs if item not in records), "")
            progress("exporting" if remaining else "prepared_pending_independent_review",
                     len(records), next_fov=remaining)
            print(f"{STUDY}: closed {len(records)}/{len(fovs)} FOV inputs", flush=True)

progress("prepared_pending_independent_review", len(records))
