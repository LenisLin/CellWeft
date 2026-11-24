"""Prepare verified Ruf2023 OME-Zarr acquisitions as native-intensity inputs.

Run with --run. Clinical and sample identities remain in NAS-only outputs.
The review table records preparer evidence; independent review follows delivery.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import openpyxl
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, _export_input, _file_version, _fsync_directory, panel_from_s1, save_input_tables,
)

STUDY = "Ruf2023_CODEX_HepatocellularCarcinoma"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE_GRID_PREFIX = ("raw/public/Ruf2023_CODEX_HepatocellularCarcinoma/"
                      "images/spora/codex/0_3477mpp/images/")
SOURCE_GRID_UM_PER_PIXEL = 0.3477
SOURCE_GRID_FORMAT_URL = "https://spora.epfl.ch/docs/io/format.html"
SOURCE_GRID_REVIEW = ("processed/Ruf2023_CODEX_HepatocellularCarcinoma/storage_reception/"
                      "ruf_scale_evidence_review_20261026T0005Z_full112/independent_review.json")
META = metadata_source_directory(NAS, STUDY) / 'spora_inventory'
PROOF = NAS / "raw/public" / STUDY / "manifest.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
CLINICAL = NAS / "raw/public" / STUDY / "metadata/TCIA_Clinical-data-Key.xlsx"
OUTPUT = NAS / "processed" / STUDY
STAGING = Path("/tmp/study_input_batch") / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0439_prepare_0")))
PROGRESS = JOB / "progress.json"


class SourceReadDeferred(Exception):
    def __init__(self, next_y: int):
        self.next_y = next_y


class PublicationDeferred(Exception):
    def __init__(self, offset: int):
        self.offset = offset


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, NAS))


def template(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def progress(stage: str, **extra: object) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    state = dict(study=STUDY, stage=stage, expected_images=17,
                 completed_images=len(images), exported_images=len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
                 patient_linked_images=int(images.sample_id.isin(
                     samples.loc[samples.patient_key.notna(), "sample_id"]).sum()) if len(images) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(extra)
    save_json(PROGRESS, state)


def waiting_io(record: dict, issue: str, **extra: object) -> None:
    state = json.loads(PROGRESS.read_text())
    state.pop("current_image", None)
    state.update(stage="waiting_io", next_image=record["image_id"], issue=issue,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    save_json(PROGRESS, state)


def source_scope() -> list[dict]:
    proof = json.loads(PROOF.read_text())["preparation_sources"]["native_images"]
    assert proof["source_image_count"] == proof["verified_count"] == len(proof["verified"]) == 17
    tissues = pd.read_parquet(META / "metadata/tissues.parquet")
    tissues = tissues[tissues.modality.eq("codex")].set_index("tissue_id", verify_integrity=True)
    assert len(tissues) == 17
    scope = []
    for item in proof["verified"]:
        image_id = item["image_id"]
        assert item["path"].startswith(SOURCE_GRID_PREFIX)
        source = NAS / item["path"]
        assert image_id in tissues.index and source.is_dir()
        assert item["all_source_sizes_match"] and item["exact_canonical_order_checked"]
        group = json.loads((source / "zarr.json").read_text())["attributes"]["ome"]
        dataset = group["multiscales"][0]["datasets"][0]
        array_path = dataset["path"]
        array = json.loads((source / array_path / "zarr.json").read_text())
        labels = [channel["label"] for channel in group["omero"]["channels"]]
        assert group["multiscales"][0]["name"] == group["omero"]["name"] == image_id
        assert labels == item["canonical_channel_labels"] and len(labels) == 38
        assert array["shape"] == item["shape"] and array["data_type"] == item["dtype"] == "uint16"
        assert array["dimension_names"] == ["c", "y", "x"]
        scale = dataset["coordinateTransformations"][0]["scale"]
        assert scale == [1, 1, 1]
        tissue = tissues.loc[image_id]
        scope.append(dict(image_id=image_id, path=item["path"], array_path=array_path,
                          shape=item["shape"], labels=labels,
                          pixel_size=[SOURCE_GRID_UM_PER_PIXEL, SOURCE_GRID_UM_PER_PIXEL],
                          sample_id=f"{tissue.patient_id}_{tissue.specimen_id}",
                          source_revision=relative(PROOF)))
    assert len({tuple(record["labels"]) for record in scope}) == 1
    assert len({record["sample_id"] for record in scope}) == 17
    return sorted(scope, key=lambda record: int(np.prod(record["shape"])))


def build_samples(scope: list[dict]) -> pd.DataFrame:
    tissues = pd.read_parquet(META / "metadata/tissues.parquet").set_index("tissue_id", verify_integrity=True)
    workbook = openpyxl.load_workbook(CLINICAL, read_only=True, data_only=True)
    try:
        values = list(workbook.worksheets[0].values)
    finally:
        workbook.close()
    clinical = {row[0]: dict(zip(values[0], row)) for row in values[1:] if row[0]}
    rows = []
    for record in scope:
        tissue = tissues.loc[record["image_id"]]
        original = tissue.original_patient_identifier
        row = dict.fromkeys(template("sample_metadata"))
        row.update(study=STUDY, sample_id=record["sample_id"], source_sample_id=tissue.tissue_id_old,
                   sample_type="tissue", preservation_method=None)
        notes = ["SPoRA tissue_id joins explicit patient_id and specimen_id; original_patient_identifier retained only in NAS metadata."]
        if original in clinical:
            source = clinical[original]
            row.update(patient_key=original, source_subject_id=original,
                       tissue_type="tumor", anatomical_site="Liver",
                       primary_diagnosis="Hepatocellular carcinoma", primary_site="Liver",
                       case_control_status="case", grade=str(source["Grade "]) if source["Grade "] is not None else None,
                       sex_reported={"Female": "female", "Male": "male"}.get(source["Gender"]))
            age = source["Age "]
            row["reported_age_years"] = int(age) if str(age).isdigit() else None
            row["ethnicity_reported"] = source["Ethnicity "]
            row["pretreatment_reported"] = str(source["Pre-treatment "]) if source["Pre-treatment "] is not None else None
            size = source["Size"]
            row["reported_tumor_size_cm"] = float(size.removesuffix(" cm")) if isinstance(size, str) and size.endswith(" cm") else None
            row["underlying_liver_disease_reported"] = source["Underlying liver disease"]
            notes.append("TCIA clinical Sample ID exact match; age timing, treatment timing and grade system unspecified; source values retained.")
        else:
            row.update(tissue_type="validation tissue")
            notes.append("Anonymous non-HCC validation organ; source label does not establish a person-level patient key or clinical row.")
        row["metadata_notes"] = " ".join(notes)
        rows.append(row)
    samples = pd.DataFrame(rows)
    assert len(samples) == 17 and samples.patient_key.notna().sum() == 15
    return samples


def field_mapping() -> pd.DataFrame:
    rows = []
    def add(source: Path | str, source_field: str, target_table: str, target_field: str,
            join_keys: str, rule: str, notes: str = "") -> None:
        row = dict.fromkeys(template("field_mapping"), "")
        row.update(study=STUDY,
                   source_file=relative(source) if isinstance(source, Path) else source,
                   source_field=source_field,
                   target_table=target_table, target_field=target_field, join_keys=join_keys,
                   value_mapping="{}", transformation_rule=rule, notes=notes)
        rows.append(row)
    add(PROOF, "verified;canonical_channel_labels;shape", "image_metadata", "source_files",
        "image_id", "Reuse complete native source proof; resolve OME-declared level-0 array path per image.")
    add(SOURCE_GRID_FORMAT_URL, "Dataset Format > Directory Layout; codex/0_3477mpp/images", "image_metadata",
        "pixel_size_x_um;pixel_size_y_um", "image_id",
        "For the exact 17 current 0_3477mpp stored-grid source groups, use the provider-declared "
        "0.3477 micrometers/pixel on both axes.",
        "Native OME scale [1,1,1] conflicts with provider format; independent scale review: "
        + SOURCE_GRID_REVIEW + "; remaining I03/D01 review unresolved.")
    tissues = META / "metadata/tissues.parquet"
    add(tissues, "tissue_id;patient_id;specimen_id", "image_metadata", "sample_id",
        "source_image_id=tissue_id", "Join explicit patient_id and specimen_id as biological sample key; 17/17 image links.")
    add(tissues, "original_patient_identifier", "sample_metadata", "patient_key",
        "tissue_id", "Only HCC donor labels with exact TCIA clinical Sample ID receive patient_key; 15/17 samples.")
    add(tissues, "tissue_id_old", "sample_metadata", "source_sample_id",
        "tissue_id", "Retain source specimen label; two anonymous validation organs have no person-level key.")
    add(CLINICAL, "Sample ID", "sample_metadata", "source_subject_id",
        "original_patient_identifier=Sample ID", "Exact join of 15 HCC image-linked samples; two extra clinical rows have no source image.")
    for source_field, target, kind, definition in [
        ("Age ", "reported_age_years", "integer", "Age in years reported by TCIA; collection time unspecified"),
        ("Ethnicity ", "ethnicity_reported", "string", "TCIA ethnicity code or label as reported"),
        ("Pre-treatment ", "pretreatment_reported", "string", "TCIA pre-treatment source value; timing unspecified"),
        ("Size", "reported_tumor_size_cm", "number", "TCIA reported tumor size converted from cm text to numeric cm"),
        ("Underlying liver disease", "underlying_liver_disease_reported", "string", "TCIA underlying liver disease code as reported")]:
        add(CLINICAL, source_field, "sample_metadata", target, "Sample ID=original_patient_identifier",
            "Copy reported value; for size remove cm unit and parse number; do not infer timing.",
            json.dumps(dict(data_type=kind, definition=definition), ensure_ascii=False))
    for source_field, target in [("Gender", "sex_reported"), ("Grade ", "grade")]:
        add(CLINICAL, source_field, "sample_metadata", target, "Sample ID=original_patient_identifier",
            "Map Male/Female to male/female; retain numeric grade as reported with unspecified system.")
    add(S1, "Study;Original marker", "panel_mapping", "marker_id;keep_channel;output_channel_index",
        "Study;source_name=Original marker", "Exact study-specific S1 join in native channel order; retain admitted markers and selected segmentation input.")
    return pd.DataFrame(rows)


class LocalNativeReader:
    def __init__(self, record: dict, stage: np.memmap, stage_path: Path):
        self.source = SourceSpec(record["image_id"], "CODEX_P01", "local_memmap",
                                 stage_path, tuple(range(38)),
                                 pixel_size_um=tuple(record["pixel_size"]))
        self.stage = stage
        self.shape, self.dtype = stage.shape, stage.dtype

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.stage[channels, y0:y1, x0:x1])


def stage_native(record: dict, max_source_read_seconds: float | None = None) -> Path:
    STAGING.mkdir(parents=True, exist_ok=True)
    path = STAGING / (record["image_id"] + ".native.dat")
    size = int(np.prod(record["shape"])) * np.dtype("uint16").itemsize
    if path.exists():
        assert path.stat().st_size == size
        return path
    if shutil.disk_usage(STAGING).free < size * 2 + 64*1024**3:
        raise OSError(28, "local stage and export would breach 64 GiB reserve")
    zarr.config.set({"async.concurrency": 4, "threading.max_workers": 4})
    blosc.set_nthreads(1)
    partial = path.with_suffix(".partial")
    checkpoint = STAGING / (record["image_id"] + ".native.progress.json")
    started = time.monotonic()
    with formal_read_lease(NAS / record["path"] / record["array_path"]):
        native = zarr.open_array(str(NAS / record["path"] / record["array_path"]), mode="r")
        assert list(native.shape) == record["shape"] and native.dtype == np.dtype("uint16")
        _, height, width = record["shape"]
        _, chunk_height, _ = native.chunks
        identity = dict(source=record["path"], revision=record["source_revision"],
                        shape=record["shape"], chunks=list(native.chunks))
        saved = json.loads(checkpoint.read_text()) if checkpoint.is_file() else None
        resume = (saved is not None and saved.get("identity") == identity and
                  partial.is_file() and partial.stat().st_size == size)
        next_y = saved["next_y"] if resume else 0
        if not resume:
            checkpoint.unlink(missing_ok=True)
        stage = np.memmap(partial, dtype="uint16", mode="r+" if resume else "w+",
                          shape=tuple(record["shape"]))
        try:
            for y0 in range(next_y, height, chunk_height):
                y1 = min(y0 + chunk_height, height)
                stage[:, y0:y1, :] = native[:, y0:y1, :]
                stage.flush()
                temporary = checkpoint.with_suffix(".partial")
                temporary.write_text(json.dumps(dict(identity=identity, next_y=y1)))
                os.replace(temporary, checkpoint)
                if max_source_read_seconds is not None and time.monotonic() - started >= max_source_read_seconds:
                    raise SourceReadDeferred(y1)
        finally:
            stage._mmap.close()
    os.replace(partial, path)
    checkpoint.unlink(missing_ok=True)
    return path


def export_staged(record: dict, panel: pd.DataFrame, stage_path: Path, destination: Path,
                  max_publication_seconds: float | None = None) -> dict:
    encoded = STAGING / (record["image_id"] + ".input.tiff")
    receipt = encoded.with_suffix(".json")
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    identity = dict(source=record["path"], shape=record["shape"],
                    revision=record["source_revision"],
                    channels=kept.source_channel_index.tolist(), names=kept.marker_id.tolist())
    saved = json.loads(receipt.read_text()) if receipt.is_file() else None
    reusable = saved is not None and saved.get("identity") == identity and encoded.is_file()
    if reusable:
        with tifffile.TiffFile(encoded) as tif:
            reusable = (len(tif.pages) == len(kept) and
                        tif.pages[0].dtype == np.dtype("uint16") and
                        tif.pages[0].shape == (max(record["shape"][1:]),) * 2)
        reusable = reusable and encoded.stat().st_size == saved["encoded_bytes"]
    if not reusable:
        stage = np.memmap(stage_path, dtype="uint16", mode="r", shape=tuple(record["shape"]))
        try:
            stats = _export_input(LocalNativeReader(record, stage, stage_path), panel, encoded)
        finally:
            stage._mmap.close()
        if stats["admitted"]:
            save_json(receipt, dict(identity=identity, stats=stats, encoded_bytes=encoded.stat().st_size))
        else:
            save_json(receipt, dict(identity=identity, stats=stats, encoded_bytes=0))
    else:
        stats = saved["stats"]
    if stats["admitted"]:
        publish_encoded(encoded, destination, max_publication_seconds)
    else:
        with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
            destination.unlink(missing_ok=True)
    return stats


def publish_encoded(encoded: Path, destination: Path,
                    max_publication_seconds: float | None = None) -> None:
    chunk_bytes = 64 * 1024**2
    encoded_stat = encoded.stat()
    encoded_version = _file_version(encoded_stat)
    with tifffile.TiffFile(encoded) as tif:
        expected_header = (len(tif.pages), tif.pages[0].shape, tif.pages[0].dtype,
                           tif.ome_metadata)
    identity = dict(encoded_bytes=encoded_stat.st_size, encoded_mtime_ns=encoded_stat.st_mtime_ns,
                    destination=relative(destination))
    checkpoint = encoded.with_suffix(".transfer.json")
    partial = destination.with_name(destination.name + ".transfer.partial")
    saved = json.loads(checkpoint.read_text()) if checkpoint.is_file() else None
    offset = saved["offset"] if saved is not None and saved.get("identity") == identity else 0
    if offset == encoded_stat.st_size:
        with formal_read_lease(destination):
            if destination.is_file() and destination.stat().st_size == encoded_stat.st_size:
                checkpoint.unlink(missing_ok=True)
                return
    if offset == 0:
        if shutil.disk_usage(NAS).free < encoded_stat.st_size + 100 * 1024**3:
            raise OSError(28, "NAS publication would breach 100 GiB reserve")
        with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("wb"):
                pass
    started = time.monotonic()
    with encoded.open("rb") as source:
        source.seek(offset)
        while offset < encoded_stat.st_size:
            block = source.read(min(chunk_bytes, encoded_stat.st_size - offset))
            assert block
            with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
                with partial.open("r+b") as target:
                    target.seek(0, os.SEEK_END)
                    assert target.tell() >= offset
                    target.seek(offset)
                    if target.write(block) != len(block):
                        raise OSError(5, "Short TIFF publication write")
                    target.flush()
                    os.fsync(target.fileno())
            offset += len(block)
            save_json(checkpoint, dict(identity=identity, offset=offset))
            if (max_publication_seconds is not None and offset < encoded_stat.st_size
                    and time.monotonic() - started >= max_publication_seconds):
                raise PublicationDeferred(offset)
    with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
        assert partial.stat().st_size == encoded_stat.st_size
        assert _file_version(encoded.stat()) == encoded_version
        save_json(encoded.with_suffix(".durable.json"), dict(
            staged_version=encoded_version, partial_version=_file_version(partial.stat())))
        os.replace(partial, destination)
        with tifffile.TiffFile(destination) as tif:
            assert (len(tif.pages), tif.pages[0].shape, tif.pages[0].dtype,
                    tif.ome_metadata) == expected_header
        _fsync_directory(destination.parent)
    checkpoint.unlink(missing_ok=True)


def export_one(record: dict, panel: pd.DataFrame, samples: pd.DataFrame, fields: pd.DataFrame,
               max_source_read_seconds: float | None = None,
               max_publication_seconds: float | None = None) -> None:
    image_id = record["image_id"]
    path = OUTPUT / "images" / image_id / "input.tiff"
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        reusable = image_id in set(images.image_id) and path.is_file()
    if reusable:
        print("Reused", image_id, flush=True)
        return
    stage_path = stage_native(record, max_source_read_seconds)
    progress("exporting", current_image=image_id)
    stats = export_staged(record, panel, stage_path, path, max_publication_seconds)
    save_json(CACHE / "image_checks" / (image_id + ".json"), stats)
    if stats["admitted"]:
        _, h, w = record["shape"]
        row = dict.fromkeys(template("image_metadata"))
        row.update(study=STUDY, image_id=image_id, sample_id=record["sample_id"], panel_id="CODEX_P01",
                   image_kind="roi", source_image_id=image_id, source_files=[record["path"]],
                   source_format="OME-Zarr", source_axes="CYX", source_dtype="uint16",
                   source_revision=record["source_revision"], roi_id=image_id,
                   source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                   pixel_size_x_um=record["pixel_size"][0], pixel_size_y_um=record["pixel_size"][1],
                   input_image_path=relative(path), split_method="already_independent",
                   split_evidence="Complete source level-0 acquisition; no tissue crop or sample split.",
                   intensity_description="Native decoded uint16 intensity; no normalization.",
                   metadata_notes="HCC cohort" if record["sample_id"] in set(samples.loc[samples.patient_key.notna(), "sample_id"])
                   else "Anonymous validation organ; patient identity unavailable.")
        row.update({key: stats[key] for key in ("height_px", "width_px", "pad_left_px", "pad_top_px",
                                             "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count")})
        save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panel, fields)
    stage_path.unlink(missing_ok=True)
    (STAGING / (image_id + ".input.tiff")).unlink(missing_ok=True)
    (STAGING / (image_id + ".input.json")).unlink(missing_ok=True)
    progress("image_completed", last_image=image_id, last_admitted=stats["admitted"])
    print("Completed", image_id, "admitted", stats["admitted"], flush=True)


def write_review(scope: list[dict], *, complete: bool) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
    checks = sorted((CACHE / "image_checks").glob("*.json"))
    excluded = [p.stem for p in checks if not json.loads(p.read_text())["admitted"]]
    rows = []
    for check, detail in [
        ("S01", f"17 proven complete native acquisitions; {len(images)} exported, {len(excluded)} all-zero-marker exclusions."),
        ("I01", "Distinct OME-Zarr acquisitions and source image identities."),
        ("I02", "Image-to-sample and panel foreign keys present in delivered rows."),
        ("I03", "Provider-declared 0_3477mpp stored grid is 0.3477 um/pixel on X/Y; "
         "native OME scale [1,1,1] conflicts; source-scale independent review: " + SOURCE_GRID_REVIEW),
        ("P01", "All 38 source labels joined to study-specific S1 in native order."),
        ("M01", "17 biological sample rows; 15 HCC patient keys, two anonymous validation specimens."),
        ("D01", f"Current delivered scope: {len(images)}/17 images; independent review and acceptance pending.")]:
        rows.append(dict(study=STUDY, preparer="codex_ruf_" + JOB.name, reviewer="", scope="study",
                         record_id=STUDY, check_id=check,
                         result="unresolved",
                         source_evidence=relative(PROOF) + ";" + relative(META / "metadata/tissues.parquet") +
                         ";" + relative(CLINICAL), output_evidence=relative(OUTPUT),
                         details="Preparer self-check: " + detail))
    review = OUTPUT / "review_results.csv"
    with formal_publish_lease(review, directory=OUTPUT):
        temporary = review.with_name(review.name + ".partial")
        pd.DataFrame(rows, columns=template("review_results")).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, review)
    if complete:
        progress("prepared_pending_independent_review", excluded_images=len(excluded))


def run(max_source_read_seconds: float | None = None,
        max_publication_seconds: float | None = None) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scope = source_scope()
    panel = panel_from_s1(STUDY, "CODEX_P01", scope[0]["labels"], S1)
    samples, fields = build_samples(scope), field_mapping()
    save_input_tables(OUTPUT, pd.DataFrame(columns=template("image_metadata")), samples, panel, fields)
    progress("metadata_ready", next_image=scope[0]["image_id"])
    for record in scope:
        try:
            export_one(record, panel, samples, fields, max_source_read_seconds,
                       max_publication_seconds)
        except PublicationDeferred as deferred:
            waiting_io(record, "Shared NAS publication remains slow; resume from the closed transfer checkpoint.",
                       transfer_offset_bytes=deferred.offset,
                       local_encoded_stage=str(STAGING / (record["image_id"] + ".input.tiff")),
                       local_transfer_checkpoint=str(STAGING / (record["image_id"] + ".input.transfer.json")))
            write_review(scope, complete=False)
            return
        except SourceReadDeferred as deferred:
            waiting_io(record, "Shared NAS source read remains slow; resume from the closed local row checkpoint.",
                       next_source_y=deferred.next_y,
                       local_native_stage=str(STAGING / (record["image_id"] + ".native.partial")),
                       local_native_checkpoint=str(STAGING / (record["image_id"] + ".native.progress.json")))
            write_review(scope, complete=False)
            return
        except OSError as error:
            if error.errno not in {5, 28, 110, 116}:
                raise
            waiting_io(record, f"NAS/local I/O or capacity error: errno {error.errno}: {error.strerror}",
                       local_native_stage=str(STAGING / (record["image_id"] + ".native.dat")),
                       local_native_checkpoint=str(STAGING / (record["image_id"] + ".native.progress.json")),
                       local_encoded_stage=str(STAGING / (record["image_id"] + ".input.tiff")),
                       local_transfer_checkpoint=str(STAGING / (record["image_id"] + ".input.transfer.json")))
            write_review(scope, complete=False)
            return
    write_review(scope, complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-source-read-seconds", type=float)
    parser.add_argument("--max-publication-seconds", type=float)
    args = parser.parse_args()
    if args.run:
        run(args.max_source_read_seconds, args.max_publication_seconds)
