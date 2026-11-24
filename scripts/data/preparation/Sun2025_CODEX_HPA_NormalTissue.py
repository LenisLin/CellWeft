"""Prepare native HPA normal-tissue CODEX cores as independent inputs.

Inputs: complete local level-0 core Zarr, native manifest, ordered markers,
and this study's S1 rows. Outputs: CYX input.tiff and five study input tables.
Each core closes and updates its index before the next source is opened.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

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
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Sun2025_CODEX_HPA_NormalTissue"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "hpa_level0"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0535_prepare_0")))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
RESERVE_NAS = 100 * 1024**3
RESERVE_LOCAL = 64 * 1024**3


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def progress(stage: str, scope: list[dict], *, current: str = "", issue: str = "") -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").is_file() else pd.DataFrame()
    completed = set(images.image_id) if len(images) else set()
    publish_json(JOB / "progress.json", dict(
        study=STUDY, phase="prepare", scientific_repair_round=0, stage=stage,
        expected_images=len(scope), exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
        current_image=current,
        next_image=next((item["image_id"] for item in scope if item["image_id"] not in completed), ""),
        issue=issue, updated_utc=datetime.now(timezone.utc).isoformat(),
    ))


def field(source: str, source_field: str, target_table: str, target_field: str,
          join: str, rule: str, notes: str = "") -> dict:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=source, source_sheet_or_table="HPA core metadata",
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join, transformation_rule=rule, notes=notes)
    return row


def inventory(s1_path: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build core and panel identities from the collected native manifest."""
    with formal_read_lease(META):
        manifest = json.loads((META / "native_manifest.json").read_text())
        ordered = json.loads((META / "ordered_markers.json").read_text())["cores"]
    if not manifest["full_current_catalog_enumerated"] or len(manifest["cores"]) != 121:
        raise ValueError("Current HPA core inventory is incomplete")
    if set(ordered) != {str(item["hpa_image_id"]) for item in manifest["cores"]}:
        raise ValueError("Ordered-marker inventory differs from native manifest")
    scope, panels, samples = [], [], []
    panel_cache: dict[tuple[str, ...], pd.DataFrame] = {}
    for item in manifest["cores"]:
        core = str(item["hpa_image_id"])
        source = RAW / core
        names = item["ordered_markers"]
        if names != ordered[core]["ordered_markers"] or names != [
            channel["label"] for channel in item["attrs"]["omero"]["channels"]
        ]:
            raise ValueError(f"Native marker order differs: {core}")
        attrs, array = item["attrs"], item["array"]
        shape = array["shape"]
        if (len(shape) != 5 or shape[:1] != [1] or shape[2] != 1 or
                shape[1] != len(names) or array["dtype"] != "|u1"):
            raise ValueError(f"Unexpected native axes or dtype: {core}")
        scale = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        if len(scale) != 5 or scale[:3] != [1.0, 1.0, 1.0] or scale[-2:] != [scale[-1]] * 2:
            raise ValueError(f"Unexpected native pixel scale: {core}")
        panel_id = item["panel_id"]
        signature = tuple(names)
        if signature not in panel_cache:
            panel_cache[signature] = panel_from_s1(STUDY, panel_id, names, s1_path)
        panel_rows = panel_cache[signature].copy()
        panel_rows["panel_id"] = panel_id
        panels.append(panel_rows)
        sample_id = f"core_{core}"
        sample = dict.fromkeys(columns("sample_metadata"))
        sample.update(study=STUDY, sample_id=sample_id, source_sample_id=item["physical_core_id"],
                      sample_type="tissue", tissue_type="normal", anatomical_site=item["tissue"],
                      metadata_notes="HPA physical core is a source specimen; donor identity and repeated-donor grouping are unavailable.")
        samples.append(sample)
        scope.append(dict(image_id=f"{STUDY}:{core}", core=core, sample_id=sample_id,
                          panel_id=panel_id, names=names, shape=shape, pixel_size_um=scale[-1],
                          tissue=item["tissue"], title=item["title"], source=source,
                          source_url=item["url"], attrs=attrs, array=array))
    fields = pd.DataFrame([
        field(relative(META / "native_manifest.json"), "hpa_image_id;physical_core_id", "image_metadata",
              "image_id;core_id;source_image_id", "hpa_image_id=physical_core_id", "Preserve the public physical core ID."),
        field(relative(META / "native_manifest.json"), "physical_core_id", "sample_metadata",
              "sample_id;source_sample_id", "physical_core_id", "One source physical core is one biological specimen row; donor grouping is unknown."),
        field(relative(META / "native_manifest.json"), "tissue;title", "image_metadata",
              "region_annotation", "hpa_image_id", "Keep source title; tissue is recorded in sample anatomical_site."),
        field(relative(META / "native_manifest.json"), "tissue", "sample_metadata",
              "anatomical_site;tissue_type", "physical_core_id", "HPA identifies a normal-tissue core and source organ."),
        field(relative(META / "native_manifest.json"), "array.shape;attrs.multiscales", "image_metadata",
              "source_axes;pixel_size_x_um;pixel_size_y_um", "hpa_image_id", "Select native level 0, t=0, z=0; use published micrometer scale."),
        field(relative(META / "ordered_markers.json"), "ordered_markers", "panel_mapping",
              "source_name;source_channel_index", "hpa_image_id", "Preserve source channel order and join exact study S1 labels."),
        field("https://www.proteinatlas.org/humanproteome/tissue/highly%2Bmultiplexed%2Bimages",
              "normal tissue TMA cores", "sample_metadata", "patient_key", "physical_core_id",
              "No donor/core crosswalk in checked portal page, article or local metadata; supplementary attachment was inaccessible; leave patient_key empty."),
    ])
    return scope, pd.concat(panels, ignore_index=True), pd.DataFrame(samples), fields


class StagedReader:
    def __init__(self, source: SourceSpec, array: np.memmap):
        self.source, self.array = source, array
        self.shape, self.dtype = array.shape, array.dtype

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])


def image_row(item: dict, result: dict, destination: Path) -> dict:
    _, _, _, height, width = item["shape"]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"], sample_id=item["sample_id"],
               panel_id=item["panel_id"], image_kind="tma_core", source_image_id=item["core"],
               source_files=[relative(item["source"])], source_format="OME-Zarr",
               source_axes="TCZYX; t=0,z=0", source_dtype="uint8",
               source_revision=relative(META / "native_manifest.json"), core_id=item["core"],
               region_annotation=item["title"], source_x0_px=0, source_y0_px=0,
               source_x1_px=width, source_y1_px=height,
               pixel_size_x_um=item["pixel_size_um"], pixel_size_y_um=item["pixel_size_um"],
               input_image_path=relative(destination), split_method="already_independent",
               split_evidence=relative(META / "native_manifest.json"),
               intensity_description="Published decoded native level-0 uint8 intensities; no normalization.",
               metadata_notes=f"HPA organ: {item['tissue']}; donor unknown; paper reports 148 cores but current public portal exposes 121.")
    for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = result[key]
    return row


def export_one(item: dict, panel: pd.DataFrame, samples: pd.DataFrame,
               fields: pd.DataFrame) -> None:
    source = item["source"] / "0"
    _, channels, _, height, width = item["shape"]
    stage_bytes = channels * height * width
    if shutil.disk_usage(STAGING).free < 2 * stage_bytes + RESERVE_LOCAL:
        raise OSError("Local scratch would cross the 64 GiB reserve")
    if shutil.disk_usage(NAS).free < stage_bytes + RESERVE_NAS:
        raise OSError("NAS would cross the 100 GiB reserve")
    with tempfile.TemporaryDirectory(prefix="hpa-native-", dir=STAGING) as directory:
        staged = Path(directory) / "native.dat"
        array = np.memmap(staged, dtype="uint8", mode="w+", shape=(channels, height, width))
        zarr.config.set({"async.concurrency": 4, "threading.max_workers": 4})
        blosc.set_nthreads(1)
        with formal_read_lease(source):
            if (json.loads((item["source"] / ".zattrs").read_text()) != item["attrs"] or
                    json.loads((source / ".zarray").read_text()) != item["array"]):
                raise ValueError(f"Native Zarr metadata differs from manifest: {item['core']}")
            native = zarr.open_array(str(source), mode="r")
            if list(native.shape) != item["shape"] or native.dtype != np.dtype("uint8"):
                raise ValueError(f"Native pixel array changed: {item['core']}")
            for y0 in range(0, height, 1024):
                for x0 in range(0, width, 1024):
                    y1, x1 = min(y0 + 1024, height), min(x0 + 1024, width)
                    array[:, y0:y1, x0:x1] = native[0, :, 0, y0:y1, x0:x1]
            array.flush()
        array._mmap.close()
        array = np.memmap(staged, dtype="uint8", mode="r", shape=(channels, height, width))
        spec = SourceSpec(item["image_id"], item["panel_id"], "zarr", staged,
                          tuple(range(channels)), pixel_size_um=(item["pixel_size_um"],) * 2,
                          source_revision=relative(META / "native_manifest.json"))
        destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
        selected = panel[panel.panel_id == item["panel_id"]]
        try:
            result = export_input(StagedReader(spec, array), selected, destination)
        finally:
            array._mmap.close()
    if not result["admitted"]:
        raise ValueError(f"All retained downstream markers have zero signal: {item['core']}")
    save_input_tables(OUTPUT, pd.DataFrame([image_row(item, result, destination)]),
                      samples, panel, fields)
    with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
        kept = selected[selected.keep_channel].sort_values("output_channel_index")
        xml = ET.fromstring(tif.ome_metadata)
        names = [node.get("Name") for node in xml.findall(".//{*}Channel")]
        if (tif.series[0].shape != (len(kept), max(height, width), max(height, width)) or
                tif.series[0].dtype != np.dtype("uint8") or names != kept.marker_id.tolist()):
            raise ValueError(f"Published TIFF header differs from panel/source: {item['core']}")


def preparer_review(scope: list[dict], stage: str) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
    linked = int(images.sample_id.notna().sum()) if len(images) else 0
    facts = [
        ("S01", f"{len(images)}/{len(scope)} current public HPA cores exported; paper's remaining 27 cores have no local portal object."),
        ("M01", f"{len(samples)} physical-core specimen rows; {linked} image-to-sample links; 0 confirmed patient links. Portal, native manifest, article and public author data-processing repository were checked; supplementary attachment inaccessible and donor crosswalk unavailable."),
        ("D01", f"Preparer closed and checked headers for {len(images)}/{len(scope)} exported files; independent review pending. Stage: {stage}."),
    ]
    rows = []
    for check, detail in facts:
        row = dict.fromkeys(columns("review_results"), "")
        row.update(study=STUDY, preparer=JOB.name, reviewer="", scope="study",
                   record_id=STUDY, check_id=check, result="unresolved",
                   source_evidence=relative(META / "native_manifest.json") + ";" +
                   relative(META / "ordered_markers.json"), output_evidence=relative(OUTPUT),
                   details=detail)
        rows.append(row)
    target = OUTPUT / "review_results.csv"
    with formal_publish_lease(target, directory=OUTPUT):
        if target.is_file():
            prior = pd.read_csv(target, dtype=str).fillna("")
            if prior.reviewer.ne("").any():
                raise ValueError("Independent review exists; preparer cannot replace it")
        temporary = target.with_name(target.name + ".partial")
        pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)


def run() -> None:
    with formal_read_lease(META):
        manifest = json.loads((META / "native_manifest.json").read_text())
    pending_scope = [{"image_id": f"{STUDY}:{item['hpa_image_id']}"} for item in manifest["cores"]]
    progress("inventory", pending_scope)
    with formal_read_lease(S1):
        s1 = pd.read_csv(S1, dtype=str, keep_default_na=False)
    with tempfile.TemporaryDirectory(prefix="hpa-panel-") as directory:
        local_s1 = Path(directory) / "study_s1.csv"
        s1[s1.Study == STUDY].to_csv(local_s1, index=False)
        scope, panel, samples, fields = inventory(local_s1)
    total_uncompressed = sum(item["shape"][1] * item["shape"][3] * item["shape"][4]
                             for item in scope)
    largest = max(item["shape"][1] * item["shape"][3] * item["shape"][4]
                  for item in scope)
    STAGING.mkdir(parents=True, exist_ok=True)
    if (shutil.disk_usage(NAS).free < total_uncompressed + RESERVE_NAS or
            shutil.disk_usage(STAGING).free < 2 * largest + RESERVE_LOCAL):
        progress("waiting_io", scope, issue="NAS or local free space below declared reserve")
        return
    empty = pd.DataFrame(columns=columns("image_metadata"))
    save_input_tables(OUTPUT, empty, samples, panel, fields)
    preparer_review(scope, "metadata_ready")
    progress("metadata_ready", scope)
    with formal_read_lease(OUTPUT):
        completed = set(pd.read_parquet(OUTPUT / "images.parquet").image_id)
    for item in scope:
        if item["image_id"] in completed:
            destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
            if destination.is_file():
                continue
            raise ValueError(f"Indexed input is missing: {item['image_id']}")
        progress("staging", scope, current=item["image_id"])
        try:
            export_one(item, panel, samples, fields)
        except OSError as error:
            progress("waiting_io", scope, current=item["image_id"], issue=str(error))
            preparer_review(scope, "waiting_io")
            return
        completed.add(item["image_id"])
        progress("image_completed", scope, current=item["image_id"])
        print(f"Completed {item['image_id']}", flush=True)
        if item["image_id"] == os.environ.get("CELLWEFT_STOP_AFTER_IMAGE"):
            preparer_review(scope, "waiting_io")
            progress("waiting_io", scope, issue="Shared NAS I/O admission is temporarily congested; resume at next_image.")
            return
    preparer_review(scope, "prepared")
    progress("prepared", scope)


if __name__ == "__main__":
    run()
