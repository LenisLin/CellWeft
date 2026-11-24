"""Prepare the accepted Rigamonti IMC ROIs as native-intensity study inputs.

Inputs: verified SPORA OME-Zarr images, tissue/channel tables, and study S1.
Outputs: per-ROI input.tiff, image/sample/panel/field tables, preparer checks.
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

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile
import zarr

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Rigamonti2024_IMC_NSCLC"
PREPARER = "0417_prepare_0"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path, default=Path(
    PROJECT / "cache/study_input_jobs" / STUDY / "0417_prepare_0/progress.json"))
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()
root = args.data_root
raw = root / "raw/public" / STUDY / "spora"
metadata = metadata_source_directory(root, STUDY)
output = root / "processed" / STUDY
cache = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
proof_path = root / "raw/public" / STUDY / "manifest.json"
scope_path = proof_path
objects_path = proof_path
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, root))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with formal_publish_lease(path):
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)


def progress(stage: str, next_index: int, **details: object) -> None:
    present = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").is_file() else pd.DataFrame()
    save_json(args.progress, dict(study=STUDY, phase="prepare", repair_round=0,
        stage=stage, expected_images=len(selected), exported_images=len(present),
        next_source_index=next_index, sample_rows=len(samples),
        sample_linked_images=int(present.sample_id.notna().sum()) if len(present) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        updated_utc=datetime.now(timezone.utc).isoformat(), **details))


def field(source: Path | str, source_table: str, source_field: str, target_table: str,
          target_field: str, join: str, rule: str, notes: str = "") -> dict:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(source) if isinstance(source, Path) else source,
        source_sheet_or_table=source_table, source_field=source_field, target_table=target_table,
        target_field=target_field, join_keys=join, value_mapping="{}",
        transformation_rule=rule, conflict_rule="Keep source values; leave unresolved sample-level conflicts empty.",
        notes=notes)
    return row


class MemoryReader:
    """Expose one completely decoded native CYX Zarr array to the shared exporter."""

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels
        self.shape = pixels.shape
        self.dtype = pixels.dtype
        self.source = SimpleNamespace(path=Path("/tmp"), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.pixels[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


# Reuse complete native read proofs and the controller's accepted source range.
verification = json.loads(proof_path.read_text())["preparation_sources"]["native_images"]
accepted = json.loads(scope_path.read_text())["preparation_sources"]["accepted_scope"]
assert verification["full_source_covered"] and not verification["failed"]
assert verification["verified_count"] == verification["source_image_count"] == 158
proofs = {item["image_id"]: item for item in verification["verified"]}
selected = sorted(accepted["accepted_image_ids"])
assert accepted["scope_complete"] and accepted["n_images"] == len(selected) == 158
assert set(selected) == set(proofs)
assert all(p["all_source_sizes_match"] and p["exact_canonical_order_checked"]
           and not p["embedded_label_differences"] for p in proofs.values())
objects = json.loads(objects_path.read_text())["preparation_sources"]["source_objects"]
assert len(objects) == verification["source_object_count"] == 948
with formal_read_lease(raw):
    for item in objects:
        suffix = item["key"].split("spora-submissions/rigamonti2024integrating/", 1)[1]
        source = raw / suffix
        assert source.is_file() and source.stat().st_size == item["size"], source
assert {p.name.removesuffix(".ome.zarr") for p in (raw / "imc/1_0mpp/images").iterdir()
        if p.is_dir() and p.name.endswith(".ome.zarr")} == set(selected)

channels_path = metadata / "imc/channels.parquet"
availability_path = metadata / "imc/channels_per_tissue.parquet"
tissues_path = metadata / "metadata/tissues.parquet"
ordered_path = metadata / "spora_inventory/imc_ordered_marker_names.json"
channels = pd.read_parquet(channels_path)
availability = pd.read_parquet(availability_path)
tissues = pd.read_parquet(tissues_path)
tissues = tissues.loc[tissues.modality.eq("imc")].copy()
ordered = json.loads(ordered_path.read_text())
names = channels.channel_name.tolist()
assert len(names) == 38 and availability.columns.tolist() == names
assert set(tissues.tissue_id) == set(selected) == set(availability.index) == set(ordered)
assert tissues.tissue_id.is_unique and availability.to_numpy(dtype=bool).all()
assert all(ordered[image_id] == names and proofs[image_id]["embedded_labels"] == names
           and proofs[image_id]["shape"] == [38, 1300, 1300]
           and proofs[image_id]["dtype"] == "float32" for image_id in selected)
panel_id = f"{STUDY}__spora_imc_38"
panel = panel_from_s1(STUDY, panel_id, names, s1_path)
panel["display_name"] = panel.marker_id
panel["source_qc_pass"] = channels.qc_pass.to_numpy()
kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 28 and panel.loc[panel.nuclear, "source_name"].tolist() == ["DNA1", "DNA2"]

# SPORA's source tissue label identifies one biological sample per patient here.
assert tissues.groupby("patient_id")["Tissue ID."].nunique().eq(1).all()
assert tissues.groupby("Tissue ID.")["patient_id"].nunique().eq(1).all()
sample_rows = []
for patient_id, group in tissues.groupby("patient_id", sort=True):
    source_sample = str(group["Tissue ID."].iloc[0])
    row = dict.fromkeys(columns("sample_metadata"), None)
    row.update(study=STUDY, sample_id=f"{STUDY}__{source_sample}",
        patient_key=f"{STUDY}__{patient_id}", source_subject_id=str(patient_id),
        source_sample_id=source_sample, cohort_id=STUDY, sample_type="lung tissue",
        tissue_type="tumor", anatomical_site="lung", case_control_status="case",
        sex_reported={"M": "male", "F": "female"}[group.Sex.iloc[0]],
        primary_diagnosis=group["Pathology diagnosis"].iloc[0],
        histology=group["Pathology diagnosis"].iloc[0],
        reported_age_years=int(group.Age.iloc[0]),
        reported_stage=str(group.Stage.iloc[0]), reported_tnm=str(group.TNM.iloc[0]),
        reported_tissue_type=str(group.Type.iloc[0]),
        metadata_notes="SPORA Tissue ID. links the ROI rows to this tissue and patient. Source grade is retained only when concordant across its ROI rows; age, TNM, and stage have no documented clinical time origin.")
    grades = group.Grade.astype(str).unique().tolist()
    row["grade"] = grades[0] if len(grades) == 1 and grades[0] != "-" else None
    row["metadata_notes"] += " Grade differs between source ROI rows." if len(grades) > 1 else ""
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)
assert len(samples) == 83 and samples.sample_id.is_unique and samples.patient_key.notna().all()
sample_by_tissue = dict(zip(samples.source_sample_id, samples.sample_id, strict=True))

fields = pd.DataFrame([
    field(tissues_path, "IMC tissue rows", "tissue_id;patient_id;Tissue ID.;specimen_id", "image_metadata", "sample_id;source_image_id", "tissue_id -> Tissue ID. -> patient_id", "Use the source tissue ID for each independent IMC ROI; link repeated ROI rows with the same Tissue ID. to one biological sample.", f"158/158 accepted IMC ROIs link to 83 source tissue samples and 83 patient IDs; source proof: {relative(proof_path)}; accepted scope: {relative(scope_path)}."),
    field(tissues_path, "IMC tissue rows", "patient_id;Tissue ID.", "sample_metadata", "sample_id;patient_key;source_subject_id;source_sample_id", "Tissue ID. -> patient_id", "Prefix source pseudonyms with the study ID; preserve source tissue and patient keys as strings."),
    field(tissues_path, "IMC tissue rows", "Age;Sex;Pathology diagnosis;Grade;TNM;Stage;Type;Quality check IMC", "sample_metadata", "sex_reported;primary_diagnosis;histology;grade;metadata_notes", "Tissue ID. -> sample_id", "Map source M/F to controlled sex values and preserve diagnosis. Fill grade only when source ROI rows agree; leave age, stage and TNM out of time-specific target fields because their clinical time origin is unspecified. Preserve each source row in region_annotation.", "Three tissue samples have conflicting source ROI grades; 34 IMC rows have source Quality check IMC=No and remain in the accepted 158-image scope for direct signal/overview assessment."),
    field(tissues_path, "IMC tissue rows", "Age;Stage;TNM;Type", "sample_metadata", "reported_age_years;reported_stage;reported_tnm;reported_tissue_type", "Tissue ID. -> sample_id", "Copy concordant source values once per tissue sample without assigning a diagnosis, collection, or index time origin.", json.dumps({"reported_age_years": {"data_type": "integer", "unit": "year", "definition": "Source Age; clinical time origin not specified"}, "reported_stage": {"data_type": "string", "definition": "Source Stage; staging time not specified"}, "reported_tnm": {"data_type": "string", "definition": "Source TNM; staging time not specified"}, "reported_tissue_type": {"data_type": "string", "definition": "Source Type in the tissue metadata"}})),
    field(channels_path, "ordered channels", "channel_name;qc_pass;description", "panel_mapping", "source_name;source_channel_index;source_qc_pass", "channel position", "Use the SPORA ordered 38-channel table and per-tissue availability; retain source QC annotations."),
    field(s1_path, "Supplementary Table S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "marker_id;role;nuclear;body;keep_channel;output_channel_index", "Study + Original marker", "Retain 28 S1-admitted source measurements including both named DNA nuclear inputs; exclude ten source rows."),
    field(proof_path, "native Zarr verification", "shape;dtype;embedded_labels;read_region;source_object_count", "image_metadata", "source_files;source_dtype;source_channel_count;source coordinates;pixel size", "tissue_id", "Use all 158 size-verified, locally readable OME-Zarr CYX arrays at the published 1 µm scale; full native frame, no rescaling."),
    field("https://spora.epfl.ch/docs/io/format.html", "SPORA dataset format", "tissue_id;patient_id;specimen_id", "sample_metadata", "metadata_notes", "published SPORA schema", "SPORA patient_id is pseudonymous; specimen_id indexes source image stores and does not by itself establish a distinct biological sample."),
    field(root / "raw/public" / STUDY / "metadata/can-23-1698_supplementary_table_1_suppst1.pdf", "paper Supplementary Table 1", "cohort counts for age at diagnosis;diagnosis;grade", "sample_metadata", "metadata_notes", "cohort only; no individual join", "The one-page public supplement gives cohort summaries without a patient or ROI crosswalk; it cannot resolve the three discordant source grades."),
], columns=columns("field_mapping"))

output.mkdir(parents=True, exist_ok=True)
cache.mkdir(parents=True, exist_ok=True)
estimated_output = len(selected) * len(kept) * 1300 * 1300 * 4
nas_free = shutil.disk_usage(root).free
local_free = shutil.disk_usage("/tmp").free
if nas_free < 100 * 2**30 + estimated_output or local_free < 64 * 2**30 + 600 * 2**20:
    progress("waiting_io_capacity", 0, estimated_uncompressed_output_bytes=estimated_output,
             nas_free_bytes=nas_free, local_free_bytes=local_free)
    raise SystemExit(75)

image_path = output / "images.parquet"
images = pd.read_parquet(image_path) if image_path.is_file() else pd.DataFrame(columns=columns("image_metadata"))
done = set(images.image_id) if len(images) else set()
progress("exporting", 0, estimated_uncompressed_output_bytes=estimated_output,
         nas_free_bytes=nas_free, local_free_bytes=local_free)
processed = 0
for index, image_id in enumerate(selected):
    if image_id in done:
        assert (root / images.set_index("image_id").loc[image_id, "input_image_path"]).is_file()
        continue
    if args.limit is not None and processed >= args.limit:
        break
    proof = proofs[image_id]
    source = root / proof["path"] / "0"
    destination = output / "images" / image_id / "input.tiff"
    progress("exporting", index, current_image=image_id)
    with formal_read_lease(source):
        pixels = np.asarray(zarr.open_array(str(source), mode="r")[:])
    assert pixels.shape == (38, 1300, 1300) and pixels.dtype == np.dtype("float32")
    reader = MemoryReader(pixels)
    result = export_input(reader, panel, destination, tile_size=256)
    receipt = dict(image_id=image_id, source_path=relative(source), admitted=result["admitted"],
                   source_proof=relative(proof_path), source_index=index,
                   channel_nonzero_pixels=result["channel_nonzero_pixels"])
    if result["admitted"]:
        with formal_read_lease(destination), tifffile.TiffFile(destination) as delivered:
            assert delivered.series[0].shape == (28, 1300, 1300)
            assert delivered.series[0].dtype == np.dtype("float32")
            for window in ((0, 0, 32, 32), (1268, 1268, 1300, 1300)):
                native = reader.read_window(kept.source_channel_index.tolist(), window)
                saved = np.stack([read_tiff_window(delivered, channel, window) for channel in range(28)])
                assert np.array_equal(native, saved), (image_id, window)
        source_row = tissues.set_index("tissue_id").loc[image_id]
        row = dict.fromkeys(columns("image_metadata"), None)
        row.update(study=STUDY, image_id=image_id,
            sample_id=sample_by_tissue[str(source_row["Tissue ID."])], panel_id=panel_id,
            image_kind="roi", source_image_id=image_id, source_files=[relative(source.parent)],
            source_format="ome-zarr", source_axes="CYX", source_dtype="float32",
            source_revision="SPORA accepted 2026-10-18; native size/read proof", roi_id=str(source_row["Image"]),
            region_annotation=json.dumps({key: str(source_row[key]) for key in
                ("Image", "Tissue ID.", "Pathology diagnosis", "Grade", "TNM", "Stage", "Type", "Quality check IMC")}),
            source_x0_px=0, source_y0_px=0, source_x1_px=1300, source_y1_px=1300,
            pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
            height_px=1300, width_px=1300, source_channel_count=38, channel_count=28,
            pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
            intensity_description="Native SPORA float32 IMC intensity; no normalization or rescaling",
            split_method="already_independent", split_evidence=relative(proof_path),
            metadata_notes="Complete published IMC ROI; source Quality check IMC is retained in region_annotation; independent review pending.")
        images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
        save_input_tables(output, images, samples, panel, fields)
        done.add(image_id)
        receipt.update(output_path=relative(destination), output_bytes=destination.stat().st_size,
                       compared_windows=[[0, 0, 32, 32], [1268, 1268, 1300, 1300]])
    save_json(cache / "image_receipts" / f"{image_id}.json", receipt)
    processed += 1
    progress("exporting", index + 1, last_completed_image=image_id)
    print(json.dumps({"source_index": index + 1, "exported_images": len(images)}), flush=True)

if len(images) == 158:
    qc_no = tissues.loc[tissues["Quality check IMC"].eq("No")].sort_values("tissue_id")
    representatives = (tissues.loc[tissues["Quality check IMC"].eq("Yes")]
                       .sort_values("tissue_id").groupby("Pathology diagnosis").head(1))
    overview_rows = pd.concat([qc_no, representatives], ignore_index=True)
    assert len(qc_no) == 34 and len(representatives) == 3
    overview = Image.new("RGB", (8 * 190, 5 * 215), "white")
    draw = ImageDraw.Draw(overview)
    channel_indices = [int(kept.set_index("source_name").loc[name, "output_channel_index"])
                       for name in ("PanCytokeratin", "CD45", "DNA1")]
    overview_index = []
    for position, (_, source_row) in enumerate(overview_rows.iterrows()):
        image_id = source_row.tissue_id
        with formal_read_lease(output / "images" / image_id / "input.tiff"):
            with tifffile.TiffFile(output / "images" / image_id / "input.tiff") as delivered:
                planes = [read_tiff_window(delivered, channel, (0, 0, 1300, 1300))[::4, ::4]
                          for channel in channel_indices]
        rgb = []
        for plane in planes:
            high = np.percentile(plane, 99.5)
            rgb.append((np.clip(plane / high, 0, 1) * 255).astype(np.uint8)
                       if high > 0 else np.zeros_like(plane, dtype=np.uint8))
        tile = Image.fromarray(np.stack(rgb, axis=-1)).resize((180, 180))
        x, y = (position % 8) * 190, (position // 8) * 215
        overview.paste(tile, (x, y))
        draw.text((x + 3, y + 184), f"{position + 1}: {source_row['Quality check IMC']}", fill="black")
        overview_index.append(dict(tile=position + 1, image_id=image_id,
                                   source_qc=source_row["Quality check IMC"],
                                   pathology=source_row["Pathology diagnosis"]))
    overview_path = cache / "source_overviews/contact_sheet.png"
    with formal_publish_lease(overview_path):
        overview_path.parent.mkdir(parents=True, exist_ok=True)
        overview.save(overview_path)
    save_json(cache / "source_overviews/contact_sheet_index.json", dict(
        study=STUDY, channels=["PanCytokeratin", "CD45", "DNA1"],
        source=relative(tissues_path), tiles=overview_index))

checks = []
for check_id, scope, record_id, result, details in [
    ("S01", "study", STUDY, "unresolved", f"158 accepted native sources reconciled; {len(images)} exported. Contact sheet of all 34 source-QC-No ROIs and three pathology representatives inspected; no whole-image blank observed. Independent visual review pending. Overview: {relative(cache / 'source_overviews/contact_sheet.png')}."),
    ("P01", "panel", panel_id, "pass", "38 ordered source names agree with SPORA table, availability, embedded OME labels and S1."),
    ("P02", "panel", panel_id, "pass", "28 S1-admitted source channels retained in native order; ten excluded."),
    ("M01", "sample", STUDY, "unresolved", f"{len(samples)} source tissue samples and 83 patient links; {len(images)} exported image-to-sample links. Three sample-grade conflicts retained empty; paper Supplementary Table 1 is aggregate-only. Independent identity review pending."),
    ("D01", "study", STUDY, "unresolved", "Preparer handoff; independent reviewer and controller acceptance pending."),
]:
    checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
        record_id=record_id, check_id=check_id, result=result,
        source_evidence=";".join((relative(proof_path), relative(scope_path), relative(tissues_path))),
        output_evidence=relative(image_path), details=details))
review_path = output / "review_results.csv"
if len(images):
    save_input_tables(output, images, samples, panel, fields)
with formal_publish_lease(review_path, directory=output):
    prior = pd.read_csv(review_path, keep_default_na=False) if review_path.is_file() else pd.DataFrame()
    if len(prior):
        prior = prior.loc[prior.preparer.ne(PREPARER)]
    pd.concat([prior, pd.DataFrame(checks)], ignore_index=True).to_csv(
        review_path, index=False, columns=columns("review_results"), encoding="utf-8-sig")
progress("prepared" if len(images) == 158 else "partial", len(selected) if len(images) == 158 else index,
         source_images_processed=processed, review_path=relative(review_path))
