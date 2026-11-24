"""Prepare complete local Hoefflin CODEX Zarr acquisitions as native inputs.

Inputs: SPoRA object inventory, level-0 Zarr, ordered channels, tissue table,
the public study sample table, and study S1. Outputs: square CYX input TIFFs
and the per-study input tables. Independent review follows this preparation.
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
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import openpyxl
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import windows
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Hoefflin2026_CODEX_IDHMutantGlioma"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
RAW_STUDY = NAS / "raw/public" / STUDY
RAW_METADATA = RAW_STUDY / "metadata"
SOURCE = NAS / "raw/public" / STUDY / "spora/codex/0_5mpp/images"
MANIFEST = META / "spora_inventory/codex_source_objects.json"
CLINICAL = RAW_METADATA / "TableS1_SampleMetadata_v13.xlsx"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
PROJECT = Path(__file__).resolve().parents[3]
JOB = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_JOB",
    PROJECT / "cache/study_preparation" / STUDY / "jobs/0757_prepare_0",
))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
RESERVE_NAS = 100 * 1024**3
RESERVE_LOCAL = 64 * 1024**3
CLINICAL_URL = "https://zenodo.org/records/18378034/files/TableS1_SampleMetadata_v13.xlsx?download=1"
AUTHOR_CODE = "https://github.com/tiroshlab/spatial_IDHm/blob/main/9_CODEX_Import_functions_objects.R"
DCX_NOTE = "Author marker_nimbus and marker_heatmap use DCX and comment out DCXv2."
REVIEWED_IMAGE_IDS = ("hoefflin2026spatial_almistmp_0000", "hoefflin2026spatial_crwvufdr_0000")


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def progress(stage: str, *, repair_round: int = 0, **detail: object) -> None:
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").is_file() else pd.DataFrame()
    state = dict(study=STUDY, stage=stage, expected_images=30,
                 exported_images=len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
                 scientific_repair_round=repair_round, updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(detail)
    write_json(JOB / "progress.json", state)


def local_scope() -> tuple[list[dict], int]:
    """Admit only acquisitions with every inventoried object at its declared size."""
    with formal_read_lease(NAS / "raw/public" / STUDY / "metadata.json", data_root=NAS):
        with resource_lock(RAW_STUDY, shared=True, name=".raw_consolidation.lock"):
            objects = json.loads(MANIFEST.read_text())
    groups: dict[str, list[dict]] = {}
    prefix = "spora-submissions/hoefflin2026spatial/codex/0_5mpp/images/"
    for item in objects:
        suffix = item["key"].removeprefix(prefix)
        if suffix == item["key"]:
            raise ValueError("Unexpected inventory source prefix")
        groups.setdefault(suffix.split("/", 1)[0], []).append(item)
    if len(groups) != 30:
        raise ValueError("Acquisition inventory no longer has 30 groups")
    scope = []
    for name, members in sorted(groups.items()):
        versions = []
        for item in members:
            path = SOURCE / item["key"].removeprefix(prefix)
            if not path.is_file():
                break
            stat = path.stat()
            if stat.st_size != item["size"]:
                break
            versions.append((path, stat.st_size, stat.st_mtime_ns))
        if len(versions) != len(members):
            continue
        path = SOURCE / name
        image_id = name.removesuffix(".ome.zarr")
        scope.append(dict(image_id=image_id, path=path, members=versions,
                          revision=f"{relative(MANIFEST)};objects={len(members)};"
                                   f"bytes={sum(size for _, size, _ in versions)};"
                                   f"mtime_ns_sum={sum(mtime for _, _, mtime in versions)}"))
    return scope, len(groups) - len(scope)


def check_versions(record: dict) -> None:
    for path, size, mtime in record["members"]:
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise RuntimeError(f"Source object changed: {path}")


def clinical_rows() -> dict[str, dict]:
    with formal_read_lease(CLINICAL, data_root=NAS):
        with resource_lock(RAW_STUDY, shared=True, name=".raw_consolidation.lock"):
            workbook = openpyxl.load_workbook(CLINICAL, read_only=True, data_only=True)
            try:
                rows = list(workbook["This_cohort"].values)
            finally:
                workbook.close()
    records = {str(row[0]): dict(zip(rows[0], row)) for row in rows[1:] if row[0]}
    if len(records) != 40:
        raise ValueError("Unexpected public sample table scope")
    return records


def metadata(scope: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    tissue = pd.read_parquet(META / "metadata/tissues.parquet")
    tissue = tissue[tissue.modality == "codex"].set_index("tissue_id", verify_integrity=True)
    channels = pd.read_parquet(META / "codex/channels.parquet")
    availability = pd.read_parquet(META / "codex/channels_per_tissue.parquet")
    ordered = json.loads((META / "spora_inventory/codex_ordered_marker_names.json").read_text())
    clinical = clinical_rows()
    panels: dict[tuple[str, ...], str] = {}
    panel_frames = []
    samples = {}
    records = {}
    for record in scope:
        image_id, path = record["image_id"], record["path"]
        if image_id not in tissue.index:
            raise ValueError(f"Tissue metadata missing: {image_id}")
        names = ordered[image_id]
        expected = [name for name in channels.channel_name if bool(availability.loc[image_id, name])]
        if names != expected:
            raise ValueError(f"Ordered panel differs from availability: {image_id}")
        with formal_read_lease(path):
            root = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
            array = json.loads((path / "0/zarr.json").read_text())
        if (root["multiscales"][0]["name"] != image_id or
                [entry["label"] for entry in root["omero"]["channels"]] != names or
                array["dimension_names"] != ["c", "y", "x"] or
                array["shape"][0] != len(names) or array["data_type"] != "uint16"):
            raise ValueError(f"Native header differs from ordered inventory: {image_id}")
        key = tuple(names)
        if key not in panels:
            panel_id = f"panel_{len(panels) + 1:03d}"
            panels[key] = panel_id
            selected = set(range(len(names))) - {names.index("DCXv2")}
            frame = panel_from_s1(STUDY, panel_id, names, S1,
                                  downstream_source_indices=selected)
            frame.loc[frame.source_name == "DCXv2", "exclusion_reason"] = (
                f"Duplicate DCX measurement; author analysis uses DCX and comments out DCXv2: {AUTHOR_CODE}")
            panel_frames.append(frame)
        row = tissue.loc[image_id]
        source_row = clinical.get(str(row.SampleID))
        if source_row is None or any(str(row[field]) != str(source_row[field]) for field in
                                     ("PatientID", "GliomaType", "GradeTumor", "GradeSection",
                                      "Location", "Sex", "Age", "IDH_status")):
            raise ValueError(f"Tissue and public sample metadata differ: {image_id}")
        sample_id = str(row.SampleID)
        sample = dict.fromkeys(columns("sample_metadata"))
        sample.update(study=STUDY, sample_id=sample_id, patient_key=str(row.patient_id),
                      source_subject_id=str(row.PatientID), source_sample_id=sample_id,
                      sample_type="tissue section", tissue_type="glioma tumor",
                      anatomical_site=str(row.Location), primary_site="Brain",
                      primary_diagnosis="IDH-mutant glioma", grade=str(row.GradeTumor),
                      molecular_subtype=str(row.GliomaType),
                      sex_reported={"M": "male", "F": "female"}[str(row.Sex)],
                      metadata_notes="SPoRA tissue_id connects to public Table S1 by SampleID; "
                                     "patient hash and PatientID agree. Age is reported without collection timing; "
                                     "GradeSection and IDH_status remain source annotations.")
        sample["reported_age_years"] = int(row.Age)
        if sample_id in samples and samples[sample_id] != sample:
            raise ValueError(f"Conflicting repeated sample: {image_id}")
        samples[sample_id] = sample
        record.update(shape=array["shape"], names=names, panel_id=panels[key],
                      sample_id=sample_id, region_annotation=str(row.GradeSection),
                      source_sample_id=sample_id)
        records[image_id] = record
    panel = pd.concat(panel_frames, ignore_index=True)
    fields = []
    def field(source: str, source_fields: str, table: str, target: str, join: str, rule: str) -> None:
        item = dict.fromkeys(columns("field_mapping"), "")
        item.update(study=STUDY, source_file=source, source_sheet_or_table="This_cohort" if source == relative(CLINICAL) else "",
                    source_field=source_fields, target_table=table, target_field=target,
                    join_keys=join, transformation_rule=rule)
        fields.append(item)
    field(relative(META / "metadata/tissues.parquet"), "tissue_id;SampleID;patient_id;PatientID",
          "image_metadata", "sample_id", "tissue_id=image_id;SampleID=sample_id",
          "Exact tissue and sample join; patient_id is the public SPoRA pseudonymous patient hash.")
    field(relative(CLINICAL), "SampleID;PatientID;GliomaType;GradeTumor;GradeSection;Location;Sex;Age;IDH_status",
          "sample_metadata", "patient_key;source_subject_id;grade;molecular_subtype;sex_reported;reported_age_years",
          "SampleID=source_sample_id", "All included source rows agree with the SPoRA tissue table. "
          "Reported age is retained separately because collection timing is not stated; section grade remains in image region_annotation.")
    field(relative(META / "spora_inventory/codex_ordered_marker_names.json"), "ordered labels",
          "panel_mapping", "source_channel_index;source_name", "image_id",
          f"Native OME labels and channel availability agree. {DCX_NOTE} {AUTHOR_CODE}")
    field(relative(S1), "Study;Original marker;Unified marker;Segmentation channel",
          "panel_mapping", "marker_id;keep_channel;output_channel_index", "Study;source_name=Original marker",
          f"Exact study S1 mapping in source order; DCX retained by author analysis. {AUTHOR_CODE}")
    field(relative(MANIFEST), "key;size", "image_metadata", "source_revision;source_files",
          "source object prefix=image_id", "Only complete local object groups with exact inventory sizes are exported.")
    field("https://spora.epfl.ch/docs/io/format.html", "0_5mpp", "image_metadata",
          "pixel_size_x_um;pixel_size_y_um", "source directory=0_5mpp",
          "SPoRA resolution directory specifies 0.5 micrometres per pixel; embedded numeric scale is 1.0.")
    return pd.DataFrame(samples.values()), panel, pd.DataFrame(fields), records


class StagedReader:
    def __init__(self, source: SourceSpec, array: np.memmap):
        self.source, self.array = source, array
        self.shape, self.dtype = array.shape, array.dtype

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1 - y0, x1 - x0), dtype=bool)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])


def export_one(record: dict, panel: pd.DataFrame, samples: pd.DataFrame, fields: pd.DataFrame) -> None:
    image_id = record["image_id"]
    destination = OUTPUT / "images" / image_id / "input.tiff"
    shape = tuple(record["shape"])
    stage_bytes = int(np.prod(shape)) * 2
    output_bytes = max(shape[1:]) ** 2 * int(panel.keep_channel.sum()) * 2
    if (shutil.disk_usage(STAGING).free < stage_bytes + output_bytes + RESERVE_LOCAL or
            shutil.disk_usage(NAS).free < output_bytes + RESERVE_NAS):
        progress("waiting_io", reason="NAS or local capacity reserve unavailable", next_image=image_id)
        return
    progress("staging", next_image=image_id, stage_bytes=stage_bytes, output_bytes_upper_bound=output_bytes)
    check_versions(record)
    STAGING.mkdir(parents=True, exist_ok=True)
    zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
    blosc.set_nthreads(1)
    with tempfile.TemporaryDirectory(prefix="hoefflin-native-", dir=STAGING) as directory:
        stage_path = Path(directory) / "native.dat"
        with formal_read_lease(record["path"]):
            native = zarr.open_array(str(record["path"] / "0"), mode="r")
            if native.shape != shape or native.dtype != np.dtype("uint16"):
                raise ValueError(f"Source array changed: {image_id}")
            stage = np.memmap(stage_path, mode="w+", dtype="uint16", shape=shape)
            try:
                for x0, y0, x1, y1 in windows(shape[-2:], 1024):
                    stage[:, y0:y1, x0:x1] = native[:, y0:y1, x0:x1]
                stage.flush()
            finally:
                stage._mmap.close()
        check_versions(record)
        stage = np.memmap(stage_path, mode="r", dtype="uint16", shape=shape)
        spec = SourceSpec(image_id, record["panel_id"], "zarr", stage_path,
                          tuple(range(shape[0])), pixel_size_um=(0.5, 0.5),
                          source_revision=record["revision"],
                          intensity_description="Published SPoRA decoded uint16 intensity")
        try:
            result = export_input(StagedReader(spec, stage), panel, destination)
        finally:
            stage._mmap.close()
    check_versions(record)
    write_json(CACHE / "image_checks" / f"{image_id}.json", result)
    if not result["admitted"]:
        progress("image_excluded", next_image=None, excluded_image=image_id,
                 reason="All retained downstream measurements were zero")
        return
    image = dict.fromkeys(columns("image_metadata"))
    _, height, width = shape
    image.update(study=STUDY, image_id=image_id, sample_id=record["sample_id"],
                 panel_id=record["panel_id"], image_kind="roi", source_image_id=image_id,
                 source_files=[relative(record["path"])], source_format="OME-Zarr",
                 source_axes="CYX", source_dtype="uint16", source_revision=record["revision"],
                 roi_id=image_id, region_annotation=record["region_annotation"],
                 source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                 pixel_size_x_um=0.5, pixel_size_y_um=0.5,
                 input_image_path=relative(destination), split_method="already_independent",
                 split_evidence="One complete level-0 SPoRA tissue image; no tissue crop.",
                 intensity_description="Published SPoRA decoded uint16 intensity; no new normalization.",
                 metadata_notes="Source section grade is region_annotation. OME numeric scale is 1.0; "
                                f"SPoRA 0_5mpp directory defines 0.5 um/px. {DCX_NOTE}")
    for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        image[key] = result[key]
    save_input_tables(OUTPUT, pd.DataFrame([image]), samples, panel, fields)
    progress("image_completed", current_image=image_id, next_image=None)


def self_check(scope: list[dict], panel: pd.DataFrame) -> None:
    images = pd.read_parquet(OUTPUT / "images.parquet")
    for row in images.itertuples(index=False):
        selected = panel[(panel.panel_id == row.panel_id) & panel.keep_channel].sort_values("output_channel_index")
        with formal_read_lease(NAS / row.input_image_path):
            with tifffile.TiffFile(NAS / row.input_image_path) as tif:
                names = [item.get("Name") for item in ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
                if (tif.series[0].shape != (len(selected), row.height_px, row.width_px) or
                        tif.series[0].dtype != np.dtype("uint16") or names != selected.marker_id.tolist()):
                    raise ValueError(f"Published image/header/panel mismatch: {row.image_id}")
    reviews = []
    for check, details in (
        ("S01", f"Preparer: {len(images)}/30 images exported; {30-len(scope)} source groups still incomplete locally."),
        ("P02", f"Preparer: DCX and DCXv2 map to DCX; {DCX_NOTE} {AUTHOR_CODE} Independent review pending."),
        ("M01", f"Preparer: {len(images)} exported images connect to {images.sample_id.nunique()} public samples and their patient hashes; independent review pending."),
        ("D01", f"Preparer checked {len(images)} TIFF headers and table links; independent review and controller acceptance pending."),
    ):
        item = dict.fromkeys(columns("review_results"), "")
        item.update(study=STUDY, preparer=JOB.name, reviewer="", scope="study", record_id=STUDY,
                    check_id=check, result="unresolved", source_evidence=relative(MANIFEST) + ";" +
                    relative(META / "metadata/tissues.parquet") + ";" + relative(CLINICAL),
                    output_evidence=relative(OUTPUT), details=details)
        reviews.append(item)
    target = OUTPUT / "review_results.csv"
    previous = pd.read_csv(target, dtype=str, keep_default_na=False) if target.is_file() else pd.DataFrame()
    if len(previous):
        previous = previous[~((previous.preparer == JOB.name) &
                              (previous.reviewer == "") &
                              previous.check_id.isin(["S01", "P02", "M01", "D01"]))]
    updated = pd.concat([previous, pd.DataFrame(reviews)], ignore_index=True)
    with formal_publish_lease(target, directory=OUTPUT):
        temporary = target.with_suffix(".csv.partial")
        updated.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)


def reusable_images(scope: list[dict]) -> set[str]:
    """Reuse only indexed inputs bound to the current source group and TIFF header."""
    images = pd.read_parquet(OUTPUT / "images.parquet")
    indexed = images.set_index("image_id", verify_integrity=True)
    reusable = set()
    for record in scope:
        image_id = record["image_id"]
        if image_id not in indexed.index:
            continue
        row = indexed.loc[image_id]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if row.source_revision != record["revision"] or not destination.is_file():
            continue
        with formal_read_lease(destination):
            with tifffile.TiffFile(destination) as tif:
                side = max(record["shape"][1:])
                if (tif.series[0].shape != (int(row.channel_count), side, side) or
                        tif.series[0].dtype != np.dtype("uint16")):
                    continue
        reusable.add(image_id)
    return reusable


def repair_dcx_provenance() -> None:
    scope, incomplete = local_scope()
    reviewed = [record for record in scope if record["image_id"] in REVIEWED_IMAGE_IDS]
    if len(reviewed) != len(REVIEWED_IMAGE_IDS):
        raise ValueError("Reviewed source groups are no longer complete")
    samples, panel, fields, _ = metadata(reviewed)
    images = pd.read_parquet(OUTPUT / "images.parquet")
    images = images[images.image_id.isin(REVIEWED_IMAGE_IDS)].copy()
    if set(images.image_id) != set(REVIEWED_IMAGE_IDS):
        raise ValueError("Reviewed image records are missing")
    images["metadata_notes"] = images.metadata_notes.str.replace(
        "DCX channel choice awaits P02 review.", DCX_NOTE, regex=False)
    save_input_tables(OUTPUT, images, samples, panel, fields)

    target = OUTPUT / "review_results.csv"
    previous = pd.read_csv(target, dtype=str, keep_default_na=False)
    repair = dict.fromkeys(columns("review_results"), "")
    repair.update(study=STUDY, preparer=JOB.name, scope="study", record_id=STUDY,
                  check_id="P02", result="unresolved",
                  source_evidence=AUTHOR_CODE,
                  output_evidence=";".join(relative(OUTPUT / name) for name in
                                           ("panel_mapping.csv", "field_mapping.csv", "images.parquet")),
                  details=json.dumps(dict(image_ids=REVIEWED_IMAGE_IDS, round=1,
                                          action="Author-supported DCX provenance recorded in panel, field, and image notes; input pixels and channel positions unchanged.",
                                          independent_review="pending")))
    updated = pd.concat([previous, pd.DataFrame([repair])], ignore_index=True)
    with formal_publish_lease(target, directory=OUTPUT):
        temporary = target.with_suffix(".csv.partial")
        updated.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)
    progress("repair_completed_pending_review", repair_round=1,
             eligible_images=len(scope), incomplete_images=incomplete,
             reviewed_image_ids=list(REVIEWED_IMAGE_IDS), d01_bound=False,
             next_step="Independent P02 and changed-metadata recheck, then exact D01 binding")


def run() -> None:
    scope, in_flight = local_scope()
    samples, panel, fields, records = metadata(scope)
    empty = pd.DataFrame(columns=columns("image_metadata"))
    save_input_tables(OUTPUT, empty, samples, panel, fields)
    completed = reusable_images(scope)
    pending = [record for record in scope if record["image_id"] not in completed]
    progress("metadata_ready", eligible_images=len(scope), incomplete_images=in_flight,
             next_image=pending[0]["image_id"] if pending else None)
    for record in pending:
        selected = panel[panel.panel_id == record["panel_id"]]
        export_one(record, selected, samples, fields)
        if json.loads((JOB / "progress.json").read_text())["stage"] == "waiting_io":
            return
    if pending or not (OUTPUT / "review_results.csv").is_file():
        self_check(scope, panel)
    progress("prepared_local_scope", eligible_images=len(scope), incomplete_images=in_flight,
             next_image=None, independent_review="pending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repair-provenance", action="store_true")
    arguments = parser.parse_args()
    if arguments.repair_provenance:
        repair_dcx_provenance()
    elif arguments.run:
        run()
