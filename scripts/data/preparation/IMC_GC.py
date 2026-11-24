"""Export registered IMC_GC ROIs as native, revision-bound study inputs.

Read the existing private metadata bundle and current formal panel/clinical tables.
Write per-image input/validity TIFFs, the standard image/field tables and SSD
receipts. Reuse pixels with current source-supported clinical annotations.
Clinical tables and registered source membership remain unchanged.
Use --export explicitly; this adapter never runs normalization or segmentation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
import os
from pathlib import Path

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.study_input import TEMPLATES, export_input, save_input_tables
from cellweft.data.source_metadata import read_metadata_index, read_metadata_member

STUDY = "IMC_GC"
PROJECT = Path(__file__).resolve().parents[3]
MEMBER = "private/IMC_GC/clinical_metadata/source_image_inventory.parquet"
ANNOTATION_FIELDS = ("sample_id", "patient_key", "patient_link_status", "sample_link_status",
                     "record_phase_reported", "timepoint_evidence_ref")


def save_json(path: Path, value: dict) -> None:
    """Atomically update an SSD receipt; JSON contains no clinical attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def source_spec(row: pd.Series, root: Path) -> SourceSpec:
    """Resolve the frozen native source without substituting a backup or identity."""
    value = (row.source_registered_mcd_path if row.source_format == "mcd"
             else row.backup_tiff_path)
    path = Path(value)
    path = path if path.is_absolute() else root / path
    scale = ((float(row.pixel_size_x_um), float(row.pixel_size_y_um))
             if pd.notna(row.pixel_size_x_um) and pd.notna(row.pixel_size_y_um) else None)
    return SourceSpec(image_id=row.image_id, panel_id=row.panel_id,
        kind=row.source_format, path=path, channels=tuple(range(int(row.source_channel_count))),
        acquisition_id=int(row.source_acquisition_id) if row.source_format == "mcd" else None,
        pixel_size_um=scale, intensity_description="Native source intensities; no normalization")


def current_annotations(record: dict, row: pd.Series) -> dict:
    """Reuse a pixel receipt while receiving the current clinical identity fields."""
    for key in ANNOTATION_FIELDS:
        if key in row:
            record[key] = None if pd.isna(row[key]) else str(row[key])
    return record


def export_one(row: pd.Series, root: Path, output: Path, panel: pd.DataFrame,
               columns: list[str], job: Path, verify_only: bool) -> dict:
    """Export one ROI and return its typed-table candidate or an actionable failure."""
    source = source_spec(row, root)
    destination = output / "images" / row.image_id / "input.tiff"
    receipt_path = job / "receipts" / f"{row.image_id}.json"
    source_version = file_revision(source.path)
    if (source_version["size"] != int(row.source_file_size_bytes) or
            source_version["mtime_ns"] != int(row.source_file_mtime_ns)):
        raise ValueError("Source size/mtime differs from frozen inventory")
    selection = json.loads(panel.to_json(orient="records"))
    previous = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    if previous and destination.is_file():
        if (previous["source_version"] == source_version and previous["panel"] == selection
                and previous["output_version"] == file_revision(destination)):
            for asset in previous["validity_versions"]:
                if file_revision(Path(asset["path"])) != asset:
                    raise ValueError("Published validity version changed")
            with tifffile.TiffFile(destination) as tif:
                saved = previous["row"]
                if (len(tif.pages) != saved["channel_count"] or
                        tif.pages[0].shape != (saved["height_px"], saved["width_px"]) or
                        str(tif.pages[0].dtype) != saved["source_dtype"]):
                    raise ValueError("Published header differs from completed receipt")
            return current_annotations(previous["row"], row)
    if verify_only:
        raise ValueError("No current completed export receipt")
    with open_source(source) as reader:
        # Reader shape is native CYX; padding never changes the acquisition frame.
        _, native_h, native_w = reader.shape
        if (native_h, native_w) != (int(row.source_native_height_px), int(row.source_native_width_px)):
            raise ValueError("Native source geometry differs from frozen inventory")
        if str(reader.dtype) != row.source_dtype:
            raise ValueError("Native source dtype differs from frozen inventory")
        result = export_input(reader, panel, destination, data_root=root)
    if not result["admitted"]:
        raise ValueError("No nonzero retained downstream measurement; no formal input admitted")
    if file_revision(source.path) != source_version:
        raise ValueError("Source version changed during export")
    record = {}
    for key in columns:
        value = row[key]
        if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        record[key] = value
    record.update({key: value for key, value in result.items() if key in columns})
    record.update(source_files=list(row.source_files),
        source_revision=json.dumps(source_version, sort_keys=True),
        source_x0_px=0, source_y0_px=0, source_x1_px=native_w, source_y1_px=native_h,
        input_image_path=destination.relative_to(root).as_posix(),
        valid_pixels_path=(destination.with_name(result["valid_pixels"]).relative_to(root).as_posix()
                           if result["valid_pixels"] else None),
        split_method="source_acquisition" if source.kind == "mcd" else "already_independent",
        split_evidence=f"raw/private/IMC_GC/metadata.json#{MEMBER}",
        intensity_description="Native source intensities; no normalization",
        metadata_notes="Frozen identity retained. Missing sample/patient links remain unknown; "
                       "no patient-level use without a verified link.")
    current_annotations(record, row)
    validity = ([file_revision(destination.with_name(result["valid_pixels"]))]
                if result["valid_pixels"] else [])
    save_json(receipt_path, dict(source_version=source_version, panel=selection,
        output_version=file_revision(destination), validity_versions=validity,
        row=record, channel_nonzero_pixels=result["channel_nonzero_pixels"],
        channel_means=result["channel_means"]))
    return record


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--jobdir", type=Path, default=PROJECT / "cache/imc_gc_input_20261119")
parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
parser.add_argument("--image-id", action="append")
parser.add_argument("--limit", type=int)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--export", action="store_true")
mode.add_argument("--verify-only", action="store_true")
args = parser.parse_args()
root, job = args.data_root, args.jobdir
output = root / "processed" / STUDY
job.mkdir(parents=True, exist_ok=True)

# One adapter owns this study; NAS write leases are acquired only for publication.
with resource_lock(job, name=".adapter.lock"):
    bundle = root / "raw/private/IMC_GC/metadata.json"
    _, index = read_metadata_index(bundle)
    inventory = pd.read_parquet(BytesIO(read_metadata_member(bundle, MEMBER, index=index)))
    assert len(inventory) == 961 and inventory.image_id.is_unique
    samples = pd.read_csv(output / "sample_metadata.csv", dtype=str, keep_default_na=False)
    clinical_bytes = {name: (output / name).read_bytes()
                      for name in ("sample_metadata.csv", "patient_metadata.csv")}
    panel = pd.read_csv(output / "panel_mapping.csv", keep_default_na=False)
    # Empty output positions belong to dropped rows; retained positions are numeric.
    panel["output_channel_index"] = pd.to_numeric(
        panel.output_channel_index.replace("", pd.NA), errors="raise").astype("Int64")
    fields = pd.read_csv(output / "field_mapping.csv", keep_default_na=False)
    dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
    columns = dictionary.loc[dictionary.table_name == "image_metadata", "field_name"].tolist()
    extensions = {"patient_key": "Frozen patient key, empty when source linkage is unresolved.",
        "patient_link_status": "Frozen ROI to patient evidence status; not clinical eligibility.",
        "sample_link_status": "Source-supported ROI to biological sample status; unknown stays empty.",
        "record_phase_reported": "Current source-reported pre/post specimen timing; baseline dates are separate.",
        "timepoint_evidence_ref": "Source reference supporting the current specimen timing."}
    extra_fields = []
    for key, definition in extensions.items():
        if not ((fields.target_table == "image_metadata") & (fields.target_field == key)).any():
            value = dict.fromkeys(fields.columns, "")
            value.update(study=STUDY, source_file=f"raw/private/IMC_GC/metadata.json#{MEMBER}",
                source_sheet_or_table="source_image_inventory", source_field=key,
                target_table="image_metadata", target_field=key, join_keys="image_id",
                transformation_rule="Preserve frozen source value without imputation",
                notes=json.dumps(dict(data_type="string", definition=definition)))
            extra_fields.append(value)
    fields = pd.concat([fields, pd.DataFrame(extra_fields)], ignore_index=True)
    source_pending = inventory[inventory.source_record_status != "available"]
    selected = inventory[inventory.source_record_status == "available"]
    summary_path = output / "processing_summary.json"
    if summary_path.is_file():
        excluded = json.loads(summary_path.read_text())["scope"]["excluded_input_quality"]
        selected = selected[~selected.image_id.isin([item["image_id"] for item in excluded])]
    if args.image_id:
        if not set(args.image_id).issubset(set(selected.image_id)):
            raise ValueError("Requested image is outside the readable frozen source scope")
        selected = selected[selected.image_id.isin(args.image_id)]
    if args.limit is not None:
        selected = selected.head(args.limit)
    records, failures = [], []
    def progress(stage: str) -> None:
        save_json(job / "progress.json", dict(study=STUDY, stage=stage,
            updated_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            selected_images=len(selected), completed_images=len(records), failures=failures,
            source_incomplete_images=source_pending.image_id.tolist(), downstream_started=False))
    progress("exporting" if args.export else "verifying")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_one, row, root, output,
                   panel[panel.panel_id == row.panel_id], columns, job, args.verify_only): row.image_id
                   for _, row in selected.iterrows()}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(dict(image_id=futures[future], error=f"{type(error).__name__}: {error}"))
            progress("exporting" if args.export else "verifying")
            if len(records) and len(records) % 32 == 0:
                save_input_tables(output, pd.DataFrame(records), samples, panel, fields, data_root=root)
    if records:
        save_input_tables(output, pd.DataFrame(records), samples, panel, fields, data_root=root)
    for name, payload in clinical_bytes.items():
        if (output / name).read_bytes() != payload:
            raise RuntimeError(f"Frozen clinical table changed: {name}")
    progress("complete" if not failures else "partial_failure")
    print(json.dumps(dict(completed=len(records), failed=len(failures),
                          source_incomplete=len(source_pending), downstream_started=False)))
    if failures:
        raise SystemExit(1)
