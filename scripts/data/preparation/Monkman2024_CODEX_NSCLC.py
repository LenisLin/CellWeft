"""Prepare native CODEX TMA-core inputs from the qualified Monkman SPORA release.

Inputs: 42 qualified OME-Zarr images, SPORA tables, study S1, and published AnnData.
Outputs: one square native-intensity input.tiff per core and study input tables.
The source Zarr is decoded once per spatial block into a local image buffer.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import h5py
import numpy as np
import pandas as pd
import tifffile
import zarr

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Monkman2024_CODEX_NSCLC"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "spora"
SOURCE = RAW / "codex/0_3775mpp/images"
META = metadata_source_directory(ROOT, STUDY) / 'spora_inventory' / 'codex_ordered_marker_names.json'
PROOF = metadata_source_directory(ROOT, STUDY) / "input_facts/verification.json"
OBJECTS = metadata_source_directory(ROOT, STUDY) / "input_facts/codex_source_objects.json"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
OUTPUT = ROOT / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input_runtime" / STUDY
RAW_METADATA = ROOT / "raw/public" / STUDY / "metadata"
ANALYSIS = RAW_METADATA / "anndata_4301_v4_annotated.h5ad"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    CACHE / "0255_prepare_0")))
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
PANEL_ID = "codex_37"
SCALE_UM = 0.3775
SHAPE = (37, 3024, 2688)
SIDE = 3024
PREPARER = f"{STUDY}:{JOB.name}"

zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})


def relative(path: Path) -> str:
    return metadata_source_relative(path, ROOT).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: dict) -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / (path.name + ".staged")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with formal_publish_lease(path, directory=OUTPUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, temporary)
        os.replace(temporary, path)


def checkpoint(stage: str, images: pd.DataFrame, samples: pd.DataFrame, **details: object) -> None:
    state = dict(study=STUDY, phase="prepare", scientific_repair_round=0, stage=stage,
                 expected_images=42, exported_images=len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    publish_json(JOB / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Bind qualified native objects, source tissue rows, and the ordered S1 panel."""
    proof = json.loads(PROOF.read_text())
    if not (proof["full_source_covered"] and proof["verified_count"] == 42
            and not proof["failed"] and len(proof["verified"]) == 42):
        raise ValueError("The accepted native source proof does not cover 42 images")
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    tissues = tissues[tissues.modality == "codex"].copy()
    ordered = json.loads(META.read_text())
    names = pd.read_parquet(RAW / "codex/channels.parquet").channel_name.tolist()
    available = pd.read_parquet(RAW / "codex/channels_per_tissue.parquet")
    qualified = {item["image_id"]: item for item in proof["verified"]}
    if not (len(tissues) == 42 and tissues.tissue_id.is_unique
            and set(tissues.tissue_id) == set(qualified) == set(ordered)
            == {p.name.removesuffix(".ome.zarr") for p in SOURCE.glob("*.ome.zarr")}):
        raise ValueError("Source, tissue metadata, marker order, and proof disagree")
    if tissues.duplicated(["patient_id", "specimen_id"]).any() or tissues.patient_id.nunique() != 42:
        raise ValueError("Source-declared patient/specimen identity is not unique")
    if len(names) != 37 or available.shape != (42, 37):
        raise ValueError("Source channel table differs from qualified native panel")
    objects = json.loads(OBJECTS.read_text())
    if len(objects) != proof["source_object_count"]:
        raise ValueError("Source object manifest differs from qualified proof")
    bindings: dict[str, list[tuple[str, int, int]]] = {}
    prefix = "spora-submissions/monkman2024spatial/codex/0_3775mpp/images/"
    for item in objects:
        suffix = item["key"].removeprefix(prefix)
        if suffix == item["key"]:
            raise ValueError("Unexpected source object prefix")
        path = SOURCE / suffix
        stat = path.stat()
        if stat.st_size != item["size"]:
            raise ValueError(f"Qualified source object changed: {relative(path)}")
        identity = suffix.split("/", 1)[0].removesuffix(".ome.zarr")
        bindings.setdefault(identity, []).append((suffix.split("/", 1)[1], stat.st_size, stat.st_mtime_ns))
    scope = []
    for row in tissues.sort_values("tissue_id").itertuples():
        identity = row.tissue_id
        path = SOURCE / f"{identity}.ome.zarr"
        group = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
        array = json.loads((path / "0/zarr.json").read_text())
        labels = [channel["label"] for channel in group["omero"]["channels"]]
        declared = [name for name, keep in zip(names, available.loc[identity], strict=True) if keep]
        if not (labels == declared == ordered[identity] and tuple(array["shape"]) == SHAPE
                and array["data_type"] == "uint16" and array["dimension_names"] == ["c", "y", "x"]
                and qualified[identity]["all_source_sizes_match"] and len(bindings[identity]) == 11):
            raise ValueError(f"Native header, panel, or accepted proof disagrees: {identity}")
        scope.append(dict(image_id=identity, path=path, labels=labels,
                          source_revision=json.dumps(sorted(bindings[identity]), separators=(",", ":"))))
    panel = panel_from_s1(STUDY, PANEL_ID, scope[0]["labels"], S1).drop(columns="s1_original_marker")
    if not (len(panel) == 37 and int(panel.keep_channel.sum()) == 37
            and panel.loc[panel.nuclear, "source_name"].tolist() == ["DAPI2"]):
        raise ValueError("Study S1 panel has changed")
    panel["source_qc_pass"] = pd.read_parquet(RAW / "codex/channels.parquet").qc_pass.tolist()
    return tissues, scope, panel


def sample_rows(tissues: pd.DataFrame) -> pd.DataFrame:
    """Use the source's one patient/one declared specimen and retain clinical source values."""
    rows = []
    for source in tissues.itertuples():
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=f"{source.patient_id}_specimen_{source.specimen_id}",
                   patient_key=source.patient_id,
                   source_subject_id=str(source.original_patient_identifier),
                   source_sample_id=f"{source.patient_id}:{source.specimen_id}",
                   biospecimen_group_id=None, sample_type="FFPE TMA core", tissue_type="tumor",
                   anatomical_site=source.Tissue, primary_diagnosis="non-small cell lung cancer",
                   preservation_method="ffpe", timepoint_label="pre_treatment",
                   index_therapy_modality="immunotherapy",
                   vital_status={"Alive": "alive", "Deceased ": "deceased"}[source.Status_text],
                   source_histology=source.Histology,
                   source_response=source.Response if pd.notna(source.Response) else None,
                   source_followup_time=str(source.Folloup_time),
                   source_current_status=str(source.Current_status), source_split=source.split,
                   metadata_notes=("SPORA tissue_id joins a patient-local specimen_id and clinical row; "
                                   "the publication reports one TMA core per patient. Binary response is "
                                   "retained as published and is not a RECIST category. Follow-up origin "
                                   "and endpoint coding are unresolved, so survival fields are blank."))
        if source.Histology == "Metastasis":
            row["lesion_type"] = "metastasis"
        elif source.Histology in {"Adenocarcinoma", "Squamous Cell Carcinoma", "Pleomorphic carcinoma"}:
            row["histology"] = source.Histology
        rows.append(row)
    return pd.DataFrame(rows).where(lambda frame: frame.notna(), None)


def publication_membership(tissues: pd.DataFrame) -> dict[str, bool]:
    """Match the paper's annotated-image cohort to public SPORA region names."""
    with formal_read_lease(ANALYSIS):
        with h5py.File(ANALYSIS) as matrix:
            image_column = matrix["obs/Image"]
            names = [value.decode() for value in image_column["categories"][:]]
            used = set(image_column["codes"][:].tolist())
    source_names = dict(zip(tissues.tissue_id, tissues.original_file_name_2, strict=True))
    if (len(names) != 36 or used != set(range(36)) or len(set(source_names.values())) != 42
            or any(not name.endswith(".ome.tiff") for name in names)):
        raise ValueError("Publication analysis image identities are incomplete")
    published = {name.removesuffix(".ome.tiff") for name in names}
    if not published.issubset(source_names.values()) or len(set(source_names.values()) - published) != 6:
        raise ValueError("Publication analysis images do not match six source QC exclusions")
    return {identity: name in published for identity, name in source_names.items()}


def field_rows() -> pd.DataFrame:
    rows = []
    tissue = relative(RAW / "metadata/tissues.parquet")
    def add(source_file: str, source_field: str, target_table: str, target_field: str,
            rule: str, notes: str = "", source_table: str = "codex tissue rows") -> None:
        rows.append(dict(study=STUDY, source_file=source_file, source_sheet_or_table=source_table,
                         source_field=source_field, target_table=target_table, target_field=target_field,
                         join_keys="tissue_id -> image_id; patient_id + specimen_id -> sample_id",
                         value_mapping="{}", transformation_rule=rule, time_origin_rule="",
                         conflict_rule="Require unique patient/specimen and exact source image identity", notes=notes))
    add(tissue, "tissue_id;patient_id;specimen_id", "image_metadata", "sample_id",
        "One source-declared specimen/core per patient; 42 image-to-sample and 42 sample-to-patient links.")
    add(tissue, "patient_id;specimen_id", "sample_metadata", "sample_id;patient_key;source_sample_id",
        "Preserve patient_id; construct patient-local specimen key. Publication reports one core per patient.")
    add(tissue, "original_patient_identifier", "sample_metadata", "source_subject_id",
        "Preserve source identifier as string; no cross-study identity inference.")
    add(tissue, "Histology;Tissue;Cancer", "sample_metadata", "histology;anatomical_site;primary_diagnosis;lesion_type",
        "Preserve histology source values; treat source 'Metastasis' as lesion_type, not histology.")
    add(tissue, "Status_text;Current_status", "sample_metadata", "vital_status",
        "Status_text Alive/Deceased maps to alive/deceased; Current_status agrees for all source rows.")
    for source, target, kind, definition in (
        ("Histology", "source_histology", "string", "Unmodified SPORA Histology label"),
        ("Response;Response.1", "source_response", "string", "Unmodified binary ICI response; both source columns agree"),
        ("Folloup_time", "source_followup_time", "string", "Unmodified source follow-up number; source spelling retained; origin unresolved"),
        ("Current_status", "source_current_status", "string", "Unmodified source status code"),
        ("split", "source_split", "string", "Unmodified SPORA source split label"),
    ):
        add(tissue, source, "sample_metadata", target, "Preserve source value without endpoint recoding.",
            json.dumps(dict(data_type=kind, definition=definition)))
    add(tissue, "original_file_name;File;tif_csv Sample name", "image_metadata", "core_id;roi_id",
        "The source publishes one TMA core per tissue_id; preserve the source image ID as core_id.")
    add(relative(S1), "Study;Original marker;Segmentation channel", "panel_mapping",
        "source_name;marker_id;role;output_channel_index",
        "Match all 37 native channel labels in order to study-specific S1; retain 36 markers and DAPI2.")
    add(relative(RAW / "codex/channels.parquet"), "qc_pass", "panel_mapping", "source_qc_pass",
        "Preserve source channel QC flag; retention follows S1.")
    add(relative(PROOF), "verified;full_source_covered", "image_metadata", "source_revision",
        "Reuse complete 42-image native qualification; compare every current source object size with official inventory.")
    add(relative(OBJECTS), "key;size", "image_metadata", "source_files",
        "Bind each image to 11 native objects by path, size and current mtime.")
    add(tissue, "0_3775mpp;OME coordinateTransformations", "image_metadata", "pixel_size_x_um;pixel_size_y_um",
        "Use published 0.3775 um/pixel resolution-directory scale; embedded OME transform states 1.0 and conflicts.")
    add(relative(ANALYSIS), "obs/Image", "image_metadata", "publication_analysis_included",
        "Match all used annotated-matrix image categories to source original_file_name_2; 36 present and six absent.",
        json.dumps(dict(data_type="boolean", definition="Source core appears in the paper's public annotated cell matrix; absence agrees with the six reported QC exclusions.",
                        source_url="https://zenodo.org/records/10258578")),
        source_table="AnnData obs")
    add("https://link.springer.com/article/10.1186/s12967-024-05035-8",
        "Materials and methods: Patient information and TMA construction", "sample_metadata",
        "sample_type;preservation_method;timepoint_label;index_therapy_modality",
        "One pre-ICI FFPE TMA core per patient; 42 initial cores, 6 excluded in published analysis.")
    return pd.DataFrame(rows, columns=columns("field_mapping"))


class StagedReader:
    """Read the locally decoded CYX buffer without revisiting all-channel Zarr chunks."""

    def __init__(self, pixels: np.memmap, path: Path):
        self.pixels = pixels
        self.shape = pixels.shape
        self.dtype = pixels.dtype
        self.source = SimpleNamespace(path=path, pixel_size_um=(SCALE_UM, SCALE_UM))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.pixels[channels, y0:y1, x0:x1])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


class FastTiffWriter(tifffile.TiffWriter):
    """Use lossless low-level Zstd for the shared tiled TIFF exporter."""

    def write(self, *args: object, **kwargs: object) -> object:
        if kwargs.get("compression") == "deflate":
            kwargs["compression"] = "zstd"
            kwargs["compressionargs"] = {"level": 1}
        return super().write(*args, **kwargs)


PUBLICATION_NOTE = ("The annotated cell matrix covers 36 of 42 source regions; its six absent "
                    "regions match the paper's QC-exclusion count. Independent visual QC pending. "
                    "Embedded OME scale 1.0 conflicts with the published 0.3775 mpp directory.")


def image_row(item: dict, source: pd.Series, destination: Path,
              publication_included: bool) -> dict:
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"],
               sample_id=f"{source.patient_id}_specimen_{source.specimen_id}",
               panel_id=PANEL_ID, image_kind="tma_core", source_image_id=item["image_id"],
               source_files=[relative(item["path"])], source_format="ome-zarr-v3",
               source_axes="CYX", source_dtype="uint16", source_revision=item["source_revision"],
               core_id=item["image_id"], roi_id=item["image_id"],
               region_annotation=source.Histology,
               source_x0_px=0, source_y0_px=0, source_x1_px=SHAPE[2], source_y1_px=SHAPE[1],
               pad_left_px=(SIDE-SHAPE[2])//2, pad_right_px=(SIDE-SHAPE[2])//2,
               pad_top_px=0, pad_bottom_px=0, height_px=SIDE, width_px=SIDE,
               source_channel_count=37, channel_count=37,
               pixel_size_x_um=SCALE_UM, pixel_size_y_um=SCALE_UM,
               input_image_path=relative(destination),
               intensity_description="Native decoded uint16 CODEX intensity; source upstream deconvolution/stitching retained, no CellWeft scaling",
               split_method="already_independent",
               split_evidence="Source tissue_id is one independently imaged TMA core; full native frame retained",
               publication_analysis_included=publication_included,
               metadata_notes=PUBLICATION_NOTE)
    return row


def review_rows(images: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples():
        sources = [relative(PROOF), relative(OBJECTS), relative(RAW / "metadata/tissues.parquet")]
        if check.check_id in {"S01", "I04", "D01"}:
            sources.append(relative(ANALYSIS))
        if check.check_id.startswith("P"):
            sources.extend([relative(META), relative(S1)])
        detail = dict(preparer_self_check=True, independent_review_pending=True,
                      exported_images=len(images), expected_images=42, sample_rows=len(samples),
                      sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                      patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0)
        if check.check_id in {"S01", "I04", "D01"}:
            detail["gap"] = "Six cores absent from the public 36-image analysis matrix are mapped to source regions; independent visual/source QC must confirm tissue quality and final inclusion."
        if check.check_id == "M02":
            detail["gap"] = "Binary response and follow-up value retained raw; no RECIST category, survival origin or endpoint conversion assigned."
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                         record_id=PANEL_ID if check.scope == "panel" else STUDY,
                         check_id=check.check_id, result="unresolved",
                         source_evidence=json.dumps(sources), output_evidence=relative(OUTPUT),
                         details=json.dumps(detail, ensure_ascii=False)))
    return pd.DataFrame(rows, columns=columns("review_results"))


def publish_reviews(images: pd.DataFrame, samples: pd.DataFrame) -> None:
    path = OUTPUT / "review_results.csv"
    incoming = review_rows(images, samples)
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / "review_results.csv"
    with formal_read_lease(path):
        prior = pd.read_csv(path, keep_default_na=False) if path.exists() else incoming.iloc[:0]
    replace = prior.preparer.eq(PREPARER) & prior.reviewer.eq("") if len(prior) else pd.Series(False, index=prior.index)
    pd.concat([prior.loc[~replace], incoming], ignore_index=True).to_csv(
        staged, index=False, encoding="utf-8-sig")
    with formal_publish_lease(path, directory=OUTPUT):
        temporary = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, temporary)
        os.replace(temporary, path)


def verify_header(path: Path, panel: pd.DataFrame) -> None:
    names = panel.sort_values("output_channel_index").marker_id.tolist()
    with formal_read_lease(path):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            if series.shape != (37, SIDE, SIDE) or series.dtype != np.dtype("uint16"):
                raise ValueError(f"Output header shape/dtype differs: {path}")
            import xml.etree.ElementTree as ET
            pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
            observed = [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")]
            if observed != names:
                raise ValueError(f"Output channel labels differ: {path}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", action="store_true")
parser.add_argument("--waiting-io", action="store_true")
parser.add_argument("--waiting-reason", default="Shared NAS I/O admission temporarily congested")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
if not (args.run or args.waiting_io):
    parser.error("Specify --run or --waiting-io")

with formal_read_lease(SOURCE):
    tissues, scope, panel = source_scope()
membership = publication_membership(tissues)
if args.waiting_io:
    with formal_read_lease(OUTPUT / "images.parquet"):
        completed = pd.read_parquet(OUTPUT / "images.parquet")
        linked = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
    next_image = next((item["image_id"] for item in scope
                       if item["image_id"] not in set(completed.image_id)), None)
    publish_reviews(completed, linked)
    checkpoint("waiting_io", completed, linked, next_image=next_image,
               pending_images=len(scope)-len(completed), reason=args.waiting_reason)
    raise SystemExit(0)

samples = sample_rows(tissues)
fields = field_rows()
OUTPUT.mkdir(parents=True, exist_ok=True)
with formal_read_lease(OUTPUT / "images.parquet"):
    images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists()
              else pd.DataFrame(columns=columns("image_metadata")))
known = {item["image_id"]: item for item in scope}
if any(row.image_id not in known or row.source_revision != known[row.image_id]["source_revision"]
       for row in images.itertuples()):
    raise ValueError("Previously indexed image no longer matches qualified source")
images["publication_analysis_included"] = images.image_id.map(membership).astype("boolean")
images["metadata_notes"] = PUBLICATION_NOTE
save_input_tables(OUTPUT, images, samples, panel, fields)
publish_reviews(images, samples)
checkpoint("metadata_prepared", images, samples, pending_images=42-len(images))

todo = [item for item in scope if item["image_id"] not in set(images.image_id)]
if args.limit is not None:
    todo = todo[:args.limit]
for item in todo:
    identity = item["image_id"]
    destination = OUTPUT / "images" / identity / "input.tiff"
    native_bytes = int(np.prod(SHAPE)) * 2
    output_bytes = 37 * SIDE * SIDE * 2
    if shutil.disk_usage(ROOT).free < 100 * 1024**3 + output_bytes:
        checkpoint("waiting_io", images, samples, current_image=identity, reason="NAS 100 GiB reserve")
        raise SystemExit(75)
    if shutil.disk_usage(TEMP).free < 64 * 1024**3 + native_bytes + output_bytes:
        checkpoint("waiting_io", images, samples, current_image=identity, reason="Local 64 GiB reserve")
        raise SystemExit(75)
    start = time.monotonic()
    TEMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="monkman-native-", dir=TEMP) as directory:
        staged = Path(directory) / "native.dat"
        values = np.memmap(staged, mode="w+", dtype="uint16", shape=SHAPE)
        specification = SourceSpec(identity, PANEL_ID, "zarr", item["path"] / "0", tuple(range(37)))
        with formal_read_lease(item["path"] / "0"):
            with open_source(specification) as reader:
                if reader.shape != SHAPE or reader.dtype != np.dtype("uint16"):
                    raise ValueError(f"Native pixels changed: {identity}")
                for window in windows(SHAPE[-2:], 1024):
                    x0, y0, x1, y1 = window
                    values[:, y0:y1, x0:x1] = reader.read_window(list(range(37)), window)
        values.flush()
        with patch("cellweft.data.preparation.study_input.tifffile.TiffWriter", FastTiffWriter):
            result = export_input(StagedReader(values, staged), panel, destination)
        values._mmap.close()
        del values
    if not result["admitted"]:
        checkpoint("source_excluded", images, samples, current_image=identity,
                   reason="All downstream marker pixels are zero")
        continue
    verify_header(destination, panel)
    source_row = tissues.set_index("tissue_id").loc[identity]
    images = pd.concat([images, pd.DataFrame([image_row(item, source_row, destination,
                                                       membership[identity])])], ignore_index=True)
    save_input_tables(OUTPUT, images, samples, panel, fields)
    checkpoint("image_published", images, samples, last_completed_image=identity,
               image_seconds=round(time.monotonic()-start, 2))

publish_reviews(images, samples)
checkpoint("prepared_awaiting_independent_review" if len(images) == 42 else "partial_prepared",
           images, samples, next_image=next((item["image_id"] for item in scope
                                             if item["image_id"] not in set(images.image_id)), None),
           publication_qc_gap="Six analysis-absent cores mapped; independent tissue-quality QC pending")
