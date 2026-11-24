"""Publish native IMC ROI inputs from the verified Sussman 2024 ZIP."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Sussman2024_IMC_PDAC_VascularNiche"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
ZIP = NAS / "raw/public" / STUDY / "zenodo_10246315/Raw_Data_TIFF_Files.zip"
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0553_prepare_0")))
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11250934/"
SUPPLEMENT_URL = "https://pmc.ncbi.nlm.nih.gov/articles/instance/11250934/bin/NIHMS1992488-supplement-1.pdf"
RESERVE_NAS = 100 * 1024**3
RESERVE_LOCAL = 64 * 1024**3


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def publish_bytes(path: Path, payload: bytes) -> None:
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def checkpoint(stage: str, completed: list[dict], samples: pd.DataFrame,
               **detail: object) -> None:
    state = dict(study=STUDY, phase="prepare", repair_round=0, stage=stage,
                 expected_images=35, exported_images=len(completed),
                 sample_linked_images=sum(bool(row.get("sample_id")) for row in completed),
                 sample_rows=len(samples),
                 patient_linked_samples=int(samples.patient_key.notna().sum()),
                 updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(detail)
    publish_bytes(JOB / "progress.json", json.dumps(state, indent=2).encode())


class PlaneReader:
    """Expose locally staged, native scalar TIFFs in full source order."""

    def __init__(self, paths: list[Path | None]):
        self.source = type("Source", (), {"path": STAGING, "pixel_size_um": (1.0, 1.0)})()
        self.arrays = [tifffile.memmap(path, mode="r") if path else None for path in paths]
        present = [array for array in self.arrays if array is not None]
        if not present or any(array.ndim != 2 or array.shape != present[0].shape or
                              array.dtype != present[0].dtype for array in present):
            raise ValueError("Source planes differ in shape or dtype")
        self.dtype = present[0].dtype
        self.shape = (len(paths), *present[0].shape)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.stack([np.asarray(self.arrays[index][y0:y1, x0:x1]) for index in channels])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1 - y0, x1 - x0), dtype=bool)

    def close(self) -> None:
        for array in self.arrays:
            if array is not None:
                array._mmap.close()


def source_scope() -> tuple[list[dict], list[str]]:
    manifest = json.loads((META / "ordered_markers_and_roi_members.json").read_text())
    inventory = json.loads((META / "official_archive_inventory.json").read_text())
    if ZIP.stat().st_size != 1120865095 or len(manifest["roi_bindings"]) != 35:
        raise ValueError("Official ZIP or reviewed ROI scope changed")
    if not inventory["scope_result"]["all_roi_channel_orders_identical"]:
        raise ValueError("The reviewed ROI panels are not identical")
    labels = [f'{row["isotope"]}:{row["marker"]}'
              for row in manifest["source_ordered_panel"]["all_channels"]]
    if len(labels) != 46:
        raise ValueError("Reviewed source panel must contain 46 channels")
    with formal_read_lease(ZIP), zipfile.ZipFile(ZIP) as archive:
        members = set(archive.namelist())
        if len(archive.infolist()) != inventory["scope_result"]["archive_entries_total"]:
            raise ValueError("Official archive member count changed")
        for item in manifest["roi_bindings"]:
            names = item["ordered_channel_member_names"]
            if len(names) != 46 or any(name not in members for name in names):
                raise ValueError("A reviewed ROI lacks a native channel")
            if [Path(name).stem.removesuffix(".ome").replace("_", ":", 1)
                    for name in names] != labels:
                raise ValueError("A native ROI channel differs from the reviewed order")
    return manifest["roi_bindings"], labels


def sample_table(scope: list[dict]) -> pd.DataFrame:
    cases = sorted({item["case_id"] for item in scope})
    if len(cases) != 9:
        raise ValueError("The 35 ROIs must identify nine source cases")
    rows = []
    for case in cases:
        rows.append(dict(study=STUDY, sample_id=f"case_{case}", patient_key=case,
                         source_subject_id=case, source_sample_id=case,
                         sample_type="tissue", tissue_type="tumor", anatomical_site="pancreas",
                         collection_method="surgical resection", preservation_method="ffpe",
                         case_control_status="case", primary_diagnosis="pancreatic ductal adenocarcinoma",
                         metadata_notes="Native ROI folder case label groups the specimen and patient; case-specific supplementary clinical values have not been verified."))
    return pd.DataFrame([dict.fromkeys(columns("sample_metadata"), None) | row for row in rows])


def field_table() -> pd.DataFrame:
    rows = []

    def add(source: str, source_field: str, target_table: str, target_field: str,
            keys: str, rule: str, mapping: dict[str, str] | None = None) -> None:
        rows.append(dict(study=STUDY, source_file=source,
                         source_sheet_or_table="published source", source_field=source_field,
                         target_table=target_table, target_field=target_field, join_keys=keys,
                         value_mapping=json.dumps(mapping) if mapping else "",
                         transformation_rule=rule, time_origin_rule="",
                         conflict_rule="Leave unverified case-specific clinical values null", notes=""))

    add(relative(ZIP), "ROI directory/case label", "sample_metadata",
        "sample_id;patient_key;source_subject_id", "case_id in reviewed ROI bindings",
        "Each native ROI directory has one reviewed case label; 35 ROIs group into nine cases, matching the article's nine patient samples. One sample and patient key per case.")
    add(relative(META / "ordered_markers_and_roi_members.json"),
        "roi_bindings.case_id", "sample_metadata", "source_sample_id",
        "roi_bindings.case_id = sample_metadata.source_subject_id",
        "Copy the de-identified case label from each reviewed ROI binding. All ROIs with the same label map to the same source specimen; the cohort has one specimen per case.")
    add(PAPER_URL, "Surgically resected tissue specimens", "sample_metadata",
        "sample_type", "reviewed nine-case PDAC cohort",
        "Assign the article's tissue specimen type to each source case.",
        {"surgically resected tissue specimen": "tissue"})
    add(PAPER_URL, "PDAC tumor tissue", "sample_metadata", "tissue_type",
        "reviewed nine-case PDAC cohort",
        "Classify the imaged PDAC tumor specimens using the template tissue_type vocabulary.",
        {"PDAC tumor tissue": "tumor"})
    add(PAPER_URL, "Patients with PDAC", "sample_metadata",
        "case_control_status", "reviewed nine-case PDAC cohort",
        "Classify each PDAC patient as a case using the template case_control_status vocabulary.",
        {"patient with PDAC": "case"})
    add(relative(META / "ordered_markers_and_roi_members.json"),
        "roi_bindings;source_ordered_panel", "image_metadata",
        "source_image_id;sample_id;source_files", "ROI directory and case_id",
        "Preserve the full native ROI. Each ZIP directory contributes one image; the reviewed case label connects its sample.")
    add(relative(ZIP), "<ROI>/<isotope>_<marker>.ome.tiff", "image_metadata",
        "source_dtype;source_x1_px;source_y1_px;input_image_path", "ROI directory",
        "Decode native scalar TIFF planes in official ZIP order, retain original float32 intensities, and center-pad to a square.")
    add(relative(S1), "Study;Original marker;Unified marker;Segmentation channel",
        "panel_mapping", "marker_id;role;nuclear;body;keep_channel",
        "Study + isotope:marker", "Match each native channel label exactly to current S1; retain selected markers and both Ir DNA nuclear measurements.")
    add(PAPER_URL, "Patient cohort; image acquisition; Data availability",
        "sample_metadata", "primary_diagnosis;anatomical_site;collection_method;preservation_method",
        "nine source cases", "The article describes surgically resected PDAC, FFPE sections, 35 ROIs across nine patient samples, and 1 um/pixel acquisition.")
    add(SUPPLEMENT_URL, "Table S1 patient characteristics/pathology", "sample_metadata",
        "case-specific clinical fields", "unresolved",
        "Supplementary PDF was listed on the PMC article but direct retrieval returned a browser verification page; no case-specific clinical value is inferred.")
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def prepare_one(item: dict, panel: pd.DataFrame) -> tuple[dict, tuple[int, int]]:
    kept = panel.loc[panel.keep_channel, "source_channel_index"].tolist()
    names = item["ordered_channel_member_names"]
    with tempfile.TemporaryDirectory(prefix="sussman-roi-", dir=STAGING) as directory:
        paths: list[Path | None] = [None] * len(names)
        with formal_read_lease(ZIP), zipfile.ZipFile(ZIP) as archive:
            for index in kept:
                path = Path(directory) / f"{index:03d}.tiff"
                with archive.open(names[index]) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                paths[index] = path
        reader = PlaneReader(paths)
        try:
            h, w = reader.shape[1:]
            side = max(h, w)
            estimate = len(kept) * side * side * reader.dtype.itemsize
            if shutil.disk_usage(STAGING).free < RESERVE_LOCAL + estimate * 2:
                raise RuntimeError("waiting_io: local 64 GiB reserve plus ROI staging unavailable")
            if shutil.disk_usage(NAS).free < RESERVE_NAS + estimate:
                raise RuntimeError("waiting_io: NAS 100 GiB reserve plus ROI output unavailable")
            destination = OUTPUT / "images" / item["roi_directory"] / "input.tiff"
            result = export_input(reader, panel, destination, data_root=NAS)
        finally:
            reader.close()
    return result, (h, w)


def image_row(item: dict, result: dict, shape: tuple[int, int]) -> dict:
    h, w = shape
    return dict(study=STUDY, image_id=item["roi_directory"],
                sample_id=f'case_{item["case_id"]}', panel_id="panel_01", image_kind="roi",
                source_image_id=item["roi_directory"], source_files=[relative(ZIP)],
                source_format="zip_tiff_planes", source_axes="YX scalar planes",
                source_dtype=result["source_dtype"], source_revision="zenodo:10246315:v1",
                roi_id=item["roi_directory"], source_x0_px=0, source_y0_px=0,
                source_x1_px=w, source_y1_px=h,
                pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
                pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
                height_px=result["height_px"], width_px=result["width_px"],
                source_channel_count=result["source_channel_count"],
                channel_count=result["channel_count"], pixel_size_x_um=1.0,
                pixel_size_y_um=1.0,
                input_image_path=relative(OUTPUT / "images" / item["roi_directory"] / "input.tiff"),
                valid_pixels_path=None,
                intensity_description="Native published scalar float32 IMC channel intensities; no normalization",
                split_method="already_independent", split_evidence="Full official ZIP ROI directory",
                metadata_notes="Case-specific supplementary clinical values remain unverified.")


def review_rows(completed: list[dict], samples: pd.DataFrame) -> None:
    observed = [
        ("S01", "pass" if len(completed) == 35 else "unresolved",
         f"Reviewed official ZIP has 35 independent ROI folders; exported {len(completed)}. Each retained ROI had downstream marker signal."),
        ("I01", "pass", "Image records identify the exact ZIP and complete native ROI directory."),
        ("I02", "pass", "Shared table publisher checked all nonempty sample and panel foreign keys."),
        ("I03", "pass", "Shared exporter recorded the full source bounds, 1 um pixel size and centered square padding."),
        ("I04", "pass", "One source case label per independent ROI directory; no multi-sample ROI split was indicated."),
        ("P01", "pass", "46 official ZIP-ordered isotope:marker labels matched the current study S1 exactly."),
        ("P02", "pass", "Shared S1 mapping retained selected downstream and nuclear channels."),
        ("P03", "pass", "Shared exporter reopened each TIFF header and checked the kept channel count, dimensions and dtype."),
        ("I05", "unresolved", "Independent source-to-output pixel blocks remain to be checked by the reviewer."),
        ("M01", "pass", f"Local ROI/member evidence and public article checked: {len(samples)} case-labeled sample/patient keys; {len(completed)} exported ROI links. Public supplementary Table S1 was listed but could not be retrieved."),
        ("M02", "unresolved", "Common cohort fields come from the article; case-specific Table S1 clinical values remain unavailable and null."),
        ("D01", "unresolved", "Preparer output closed; independent review and input binding are pending."),
    ]
    frame = pd.DataFrame([dict(study=STUDY, preparer="study_input_preparer",
                               reviewer="", scope="study", record_id=STUDY,
                               check_id=check, result=status,
                               source_evidence=f"{relative(META)};{PAPER_URL}",
                               output_evidence=relative(OUTPUT), details=detail)
                          for check, status, detail in observed], columns=columns("review_results"))
    publish_bytes(OUTPUT / "review_results.csv", frame.to_csv(index=False).encode("utf-8-sig"))


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    scope, labels = source_scope()
    panel = panel_from_s1(STUDY, "panel_01", labels, S1)
    samples = sample_table(scope)
    fields = field_table()
    with formal_read_lease(OUTPUT):
        existing = (pd.read_parquet(OUTPUT / "images.parquet")
                    if (OUTPUT / "images.parquet").is_file() else pd.DataFrame())
    completed = existing.to_dict("records")
    known = {row["image_id"] for row in completed}
    checkpoint("source_scope_checked", completed, samples,
               next_image=next((item["roi_directory"] for item in scope
                                if item["roi_directory"] not in known), None))
    for item in scope:
        image_id = item["roi_directory"]
        if image_id in known:
            continue
        checkpoint("staging", completed, samples, next_image=image_id)
        try:
            result, shape = prepare_one(item, panel)
            if not result["admitted"]:
                raise ValueError(f"ROI has no retained downstream marker signal: {image_id}")
            row = image_row(item, result, shape)
            save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panel, fields, data_root=NAS)
            completed.append(row)
            known.add(image_id)
            checkpoint("image_completed", completed, samples, last_completed_image=image_id,
                       next_image=next((entry["roi_directory"] for entry in scope
                                        if entry["roi_directory"] not in known), None))
        except (RuntimeError, OSError) as error:
            waiting = (str(error).startswith("waiting_io:") or
                       isinstance(error, OSError) and error.errno in
                       {errno.EIO, errno.ENOSPC, errno.ETIMEDOUT, errno.ESTALE})
            if not waiting:
                raise
            checkpoint("waiting_io", completed, samples, next_image=image_id,
                       reason=f"{type(error).__name__}: {error}")
            raise SystemExit(75) from error
    review_rows(completed, samples)
    checkpoint("prepared_pending_independent_review", completed, samples, next_image=None)


def repair_field_mapping() -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
        panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
    if len(images) != 35 or len(samples) != 9:
        raise ValueError("The reviewed image and sample scope changed")
    save_input_tables(OUTPUT, images, samples, panel, field_table(), data_root=NAS)
    checkpoint("mapping_repaired_pending_independent_review", images.to_dict("records"),
               samples, phase="repair", repair_round=1, next_check="M02 then D01")


if __name__ == "__main__":
    if sys.argv[1:] == ["--repair-field-mapping"]:
        repair_field_mapping()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: Sussman2024_IMC_PDAC_VascularNiche.py [--repair-field-mapping]")
