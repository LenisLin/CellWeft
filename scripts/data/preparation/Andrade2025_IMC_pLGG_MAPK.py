"""Prepare the accepted Andrade IMC TXT acquisitions at native decoded intensity.

Inputs: original Zenodo ZIP, accepted native proofs, S1, and published clinical
Supplementary Table 1. Outputs: square CYX input.tiff and study input tables.
TXT rasterization is a study-local adapter pending shared TXT reader support;
float32 decoding and XY placement follow readimc.TXTFile, retaining measured zeros.
Run pixel work through the batch shared formal write lease wrapper. No normalization.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import requests
import tifffile

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

STUDY = "Andrade2025_IMC_pLGG_MAPK"
PANEL = STUDY + "__native47"
PREPARER = "codex_cli_0012_prepare_0"
PROJECT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--image", action="append", help="Restrict pixel work to published Sample IDs")
args = parser.parse_args()
root = args.data_root
metadata = metadata_source_directory(root, STUDY)
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
output = root / "processed" / STUDY
temporary_root = Path("/tmp/study_input_batch") / STUDY
progress_path = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0012_prepare_0/progress.json"
archive_path = root / "raw/public" / STUDY / "public_original/LLGZenodo.zip"
templates = PROJECT / "docs/data/templates/study_input"
for directory in (cache, output, temporary_root, progress_path.parent):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(progress_path, dict(study=STUDY, stage=stage, completed_images=completed,
        expected_images=120, updated_at=datetime.now(timezone.utc).isoformat(), **details))


# Reuse the original accepted proofs and inventory every member without rehashing.
verification = json.loads((metadata / "native_verification.json").read_text())
proofs = [record["native_proof"] for record in verification["records"]]
assert verification["scope_complete"] and len(proofs) == 120
source_stat = archive_path.stat()
for proof in proofs:
    assert proof["complete"] and proof["native_order_verified"]
    assert source_stat.st_size == proof["source_file_stat"]["size"]
    assert source_stat.st_mtime_ns == proof["source_file_stat"]["mtime_ns"]
accepted = {proof["source_member"] for proof in proofs}
manifest = json.loads((metadata / "source_manifest.json").read_text())
held = set(manifest["scope_selector"]["held_archive_entries"])
with zipfile.ZipFile(archive_path) as archive:
    inventory = []
    for entry in archive.infolist():
        category = ("accepted_raw_image" if entry.filename in accepted else
                    "held_unknown_diagnosis" if entry.filename in held else
                    "auxiliary_or_alternate_modality")
        inventory.append(dict(member=entry.filename, bytes=entry.file_size, category=category))
    raw_members = {i.filename for i in archive.infolist()
                   if i.filename.startswith("LLGZenodo/LLGRawData/") and i.filename.endswith(".txt")}
    assert raw_members == accepted | held
    for proof in proofs:
        entry = archive.getinfo(proof["source_member"])
        assert all(getattr(entry, key) == value for key, value in proof["zip_member_stat"].items())
write_json(cache / "source_inventory.json", inventory)
existing_count = len(pd.read_parquet(output / "images.parquet")) if (output / "images.parquet").exists() else 0
progress("source_bound", existing_count, gaps=["独立复核待执行"])

# Read the already accepted publisher and author attachments.
source_metadata = metadata / "source_metadata"
clinical_path = source_metadata / "41590_2025_2268_MOESM2_ESM.xlsx"
panel_source = source_metadata / "41590_2025_2268_MOESM3_ESM.xlsx"
repo_url = "https://api.github.com/repos/walsh-quail-labs/Pediatric-Low-Grade-Gliomas/git/trees/main?recursive=1"

clinical = pd.read_excel(clinical_path, sheet_name="IMC_Complete_Cohort_Data", dtype=object, keep_default_na=False)
clinical = clinical[clinical["Sample ID"].ne("")].copy()
assert len(clinical) == 120 and clinical["Sample ID"].is_unique and clinical["Case ID"].is_unique
assert set(clinical["Sample ID"]) == {Path(member).stem for member in accepted}
samples = pd.DataFrame(index=range(len(clinical)), columns=pd.read_csv(templates / "sample_metadata.csv").columns)
clinical = clinical.reset_index(drop=True)
samples["study"] = STUDY
samples["sample_id"] = clinical["Sample ID"].astype(str)
samples["source_sample_id"] = clinical["Sample ID"].astype(str)
samples["patient_key"] = STUDY + "::" + clinical["Case ID"].astype(str)
samples["source_subject_id"] = clinical["Case ID"].astype(str)
samples["age_at_collection_years"] = pd.to_numeric(clinical["Age of Resection (Y)"], errors="coerce")
samples["sex_reported"] = clinical.Gender.map({"Female": "female", "Male": "male", "Unknown": "unknown"})
samples["primary_diagnosis"] = clinical["Standard Outcoming Diagnosis"]
samples["histology"] = clinical["Standard Outcoming Diagnosis"]
samples["anatomical_site"] = clinical["Standard Tumor Location"].replace("unknown", pd.NA)
samples["molecular_subtype"] = clinical["Simplified Alteration"].replace("NA", pd.NA)
samples["tissue_type"] = "tumor"
samples["metadata_notes"] = (
    "Published Supplementary Table 1: exact Sample ID to TXT stem; full Case ID retained as source case identity, "
    "without stripping specimen suffixes or inferring independent person count. "
    "Age <1 retained as source text; numeric age blank. PFS origin/event and response criteria unspecified; "
    "standardized PFS and best response blank."
)
raw_fields = {
    "Age of Resection (Y)": "source_age_at_resection",
    "Genomic Alteration": "source_genomic_alteration",
    "Tumour Location": "source_tumour_location_group",
    "PFS (Days)": "source_pfs_days",
    "PFS comment": "source_pfs_comment",
    "Response Evaluation acc. to radiology": "source_radiology_response",
    "DNA methylation profiling": "source_dna_methylation_profiling",
    "Gene panel sequencing": "source_gene_panel_sequencing",
    "RNA sequencing": "source_rna_sequencing",
}
for source, target in raw_fields.items():
    samples[target] = clinical[source].astype("string")

mapping_rows = []


def field(source: str, table: str, target: str, rule: str, *, source_file: Path = clinical_path,
          source_table: str = "IMC_Complete_Cohort_Data", notes: str = "") -> None:
    mapping_rows.append(dict(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_table,
        source_field=source, target_table=table, target_field=target,
        join_keys="TXT member basename without .txt = Sample ID; clinical Sample ID -> full Case ID",
        value_mapping="{}", transformation_rule=rule, time_origin_rule="Unknown when not explicitly supplied",
        conflict_rule="Keep original values; do not reconcile Case ID suffixes", notes=notes))


for source, target in [("Sample ID", "sample_id"), ("Sample ID", "source_sample_id"),
                       ("Case ID", "source_subject_id"), ("Standard Outcoming Diagnosis", "primary_diagnosis"),
                       ("Standard Outcoming Diagnosis", "histology")]:
    field(source, "sample_metadata", target, "Preserve published value exactly")
field("Simplified Alteration", "sample_metadata", "molecular_subtype", "Preserve published group; explicit source NA -> blank")
field("Case ID", "sample_metadata", "patient_key", "Namespace full published Case ID; preserve punctuation and suffixes; 120/120 source case connections, independent person count unasserted")
field("Gender", "sample_metadata", "sex_reported", "Female -> female; Male -> male; Unknown -> unknown")
field("Age of Resection (Y)", "sample_metadata", "age_at_collection_years", "Numeric resection age in years; <1 remains blank, preserve inequality in source_age_at_resection")
field("Standard Tumor Location", "sample_metadata", "anatomical_site", "Keep source text; explicit unknown -> blank")
field("Standard Outcoming Diagnosis", "sample_metadata", "tissue_type", "Both published included diagnoses are tumor")
for source, target in raw_fields.items():
    field(source, "sample_metadata", target, "Preserve raw published text, including missingness codes",
          notes=json.dumps(dict(data_type="string", definition="Published clinical source value: " + source)))
field("PFS (Days); PFS comment; Response Evaluation acc. to radiology", "sample_metadata", "metadata_notes",
      "Do not interpret PFS=1 as an event; unknown origin/event/censoring and response criteria leave standardized outcome fields blank")
field("Sample ID", "image_metadata", "sample_id", "Exact TXT stem join: 120/120 eligible images -> 120 samples -> 120 published Case IDs")
field("Standard Outcoming Diagnosis; Tumour Location", "image_metadata", "region_annotation", "JSON object of published diagnosis and location for the matching Sample ID")
field("TXT measurement header", "panel_mapping", "source_channel_index",
      "The first six TXT columns are push/XYZ metadata; source_channel_index counts 47 measurement columns from zero; source_txt_column_index = source_channel_index + 6",
      source_file=archive_path, source_table="LLGZenodo/LLGRawData/*.txt")
field("X; Y; Z; measurement values", "image_metadata", "intensity_description",
      "readimc-compatible float32 text decoding; raster[y,x] = row measurement; no normalization, interpolation or orientation change. Source Z is an ancillary row counter, not a selected depth plane; unique XY coordinates define the independent 2D acquisition",
      source_file=archive_path, source_table="LLGZenodo/LLGRawData/*.txt")
fields = pd.DataFrame(mapping_rows, columns=pd.read_csv(templates / "field_mapping.csv").columns)

source_names = proofs[0]["channels"][6:]
assert all(proof["channels"][6:] == source_names for proof in proofs)
panel = panel_from_s1(STUDY, PANEL, source_names,
                      root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv")
panel["source_txt_column_index"] = panel.source_channel_index + 6
assert len(panel) == 47 and int(panel.keep_channel.sum()) == 35
binding = panel[["source_channel_index", "source_name", "marker_id", "role", "keep_channel", "output_channel_index"]].to_json(orient="records")
write_json(cache / "metadata_source_review.json", dict(
    clinical_url="https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41590-025-02268-7/MediaObjects/41590_2025_2268_MOESM2_ESM.xlsx",
    clinical_file=relative(clinical_path), sheet="IMC_Complete_Cohort_Data", sample_rows=120,
    eligible_source_images_linked=120, patient_linked_samples=120, independent_person_count=None,
    patient_linkage_level="published_case_id",
    patient_identity_basis="Full published Case ID, no suffix normalization or independent-person inference",
    repository_tree=relative(source_metadata / "author_repository_tree.json"), repository_url=repo_url,
    repository_finding="Visualization notebook and cell-type color map; clinical mapping provided by publisher Table 1",
    zenodo_url="https://zenodo.org/records/16614871", publisher_live_fetch="Web tool redirected to inaccessible identity page; reused accepted localized publisher HTML and attachments",
    gaps=["PFS time origin/event definition and response criteria not established; raw fields preserved",
          "Physical pixel pitch not established; pixel_size_x_um and pixel_size_y_um blank",
          "Published Case ID connection is retained; independent person count remains unknown",
          "Independent input review pending"]
))
write_json(cache / "capacity_estimate.json", dict(
    eligible_txt_bytes=sum(item["bytes"] for item in inventory if item["category"] == "accepted_raw_image"),
    extraction_mode="One source TXT and one float32 raster at a time on /tmp; remove after publishing image",
    reserve_nas_bytes=100 * 1024**3, reserve_local_bytes=64 * 1024**3,
    representative_source="T1 independently decoded with readimc: 47 x 1000 x 1000 float32",
    provisional_output_bytes=120 * 35 * 1000**2 * 4,
    output_estimate_basis="35 float32 channels * measured bounding rectangle, padded to square; exact shapes learned per TXT"
))
write_json(cache / "controller_requests.json", [
    "请安排不同CLI上下文独立复核输入及来源Case ID连接；整理者仅记录自检，不写D01独立验收。",
    "共享SourceReader尚未支持IMC TXT；本study脚本临时以分块float32解析和XY磁盘栅格适配。建议后续迁移为src中的共享TXT读取器，保留前6列metadata与47测量通道的位置区别。",
    "最终补边复核发现共享read_tiff_window在先访问TiffFile.series后可能接到TiffFrame，其samplesperpixel属性缺失而报错；本study在series物化前进行页窗口检查。建议共享读取器兼容TiffFrame/keyframe。",
])


class TxtRaster:
    """Bounded-memory TXT adapter with explicit acquisition validity.

    Dimensions follow readimc (max XY + 1), so leading acquisition coordinates
    are retained. A local disk raster prevents whole multi-channel decode in RAM.
    """

    def __init__(self, path: Path):
        width = height = count = 0
        self.z_is_row_counter = True
        for chunk in pd.read_csv(path, sep="\t", usecols=["X", "Y", "Z"], dtype=np.float32, chunksize=65536):
            xy = chunk[["X", "Y"]].to_numpy()
            if not np.isfinite(xy).all() or (xy < 0).any() or (xy != np.floor(xy)).any():
                raise ValueError("TXT acquisition requires nonnegative integer XY coordinates")
            self.z_is_row_counter &= bool(np.array_equal(chunk.Z.to_numpy(), np.arange(count, count + len(chunk), dtype=np.float32)))
            width, height = max(width, int(chunk.X.max()) + 1), max(height, int(chunk.Y.max()) + 1)
            count += len(chunk)
        self.shape = (47, height, width)
        self.dtype = np.dtype("float32")
        self.source = SimpleNamespace(pixel_size_um=None)
        self.raster_path = path.with_suffix(".float32")
        self.values = np.memmap(self.raster_path, mode="w+", dtype=self.dtype, shape=self.shape)
        self.values[:] = 0
        self.valid = np.zeros((height, width), dtype=bool)
        for chunk in pd.read_csv(path, sep="\t", dtype=np.float32, chunksize=32768):
            if chunk.columns.tolist()[6:] != source_names:
                raise ValueError("Source TXT measurement order differs from accepted native panel")
            x, y = chunk.X.to_numpy(dtype=int), chunk.Y.to_numpy(dtype=int)
            if self.valid[y, x].any() or len(np.unique(y * width + x)) != len(x):
                raise ValueError("Duplicate acquisition coordinates")
            self.values[:, y, x] = chunk.iloc[:, 6:].to_numpy().T
            self.valid[y, x] = True
        assert int(self.valid.sum()) == count

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.values[channels, y0:y1, x0:x1])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.valid[y0:y1, x0:x1]

    def close(self) -> None:
        self.values._mmap.close()
        self.raster_path.unlink()


records = pd.read_parquet(output / "images.parquet").to_dict("records") if (output / "images.parquet").exists() else []
receipts_dir = cache / "image_receipts"
receipts_dir.mkdir(exist_ok=True)
selected = [p for p in proofs if not args.image or Path(p["source_member"]).stem in args.image]
if args.image and len(selected) != len(set(args.image)):
    raise ValueError("Selected image is outside accepted scope")
with formal_publish_lease(output, directory=output):
    samples.to_csv(output / "sample_metadata.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(output / "panel_mapping.csv", index=False, encoding="utf-8-sig")
    fields.to_csv(output / "field_mapping.csv", index=False, encoding="utf-8-sig")


def publish_reviews() -> None:
    """Record preparer checks separately from the pending independent review."""
    checks = []
    details = {
        "S01": f"Preparer self-check: {len(records)}/120 delivered; 125 raw TXT members, 120 eligible and 5 held for unknown diagnosis; all archive members in source_inventory.json.",
        "M01": f"Preparer self-check: 120 sample rows; {len(records)} delivered images joined exactly to Sample ID; 120 sample rows connected to full published Case ID. Distinct independent people not inferred. See metadata_source_review.json.",
        "M02": "Preparer self-check: selected clinical values copied from publisher Table 1; raw PFS, response and bounded age preserved; standardized PFS/event/response left blank pending definitions.",
        "D01": f"独立复核待执行；整理者自检不构成独立验收。当前{len(records)}/120 input图像；120 samples及120来源Case ID连接。",
    }
    for check, detail in details.items():
        checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study" if check in {"S01", "D01"} else "sample",
            record_id=STUDY, check_id=check, result="unresolved" if check == "D01" else "pass",
            source_evidence=relative(clinical_path) if check.startswith("M") else relative(metadata / "native_verification.json"),
            output_evidence=relative(output), details=detail))
    for check, detail in {
        "P01": "47 actual measurement columns in accepted native order; first 6 TXT columns are acquisition metadata. All labels join this study's S1 exactly.",
        "P02": "35 retained channels in source order: 33 unique downstream markers and both S1 DNA nuclear inputs; 12 excluded measurements remain in panel table.",
    }.items():
        checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="panel", record_id=PANEL,
            check_id=check, result="pass", source_evidence=relative(metadata / "native_verification.json"),
            output_evidence=relative(output / "panel_mapping.csv"), details="Preparer self-check: " + detail))
    for row in records:
        receipt = receipts_dir / (row["image_id"] + ".json")
        for check, detail in {
            "I01": "Unique source TXT member, stable image identity and NAS-relative output path.",
            "I02": "Exact source Sample ID join and actual ordered panel foreign key.",
            "I03": "Full XY acquisition rectangle, centered square padding, unchanged orientation and float32 intensity; pixel scale unknown.",
            "I04": "Already independent source core; TXT stem exactly matches publisher biological Sample ID, no dearray crop or sample-region split.",
            "P03": "Reopened TIFF has 35 pages, float32 dtype and ordered OME channel names matching the retained panel.",
            "I05": "All 35 retained channels compared at central and lower-right 32-pixel source windows; exact equality to TXT raster. Coordinates and signal statistics in receipt; independent source decode review pending.",
        }.items():
            checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="image", record_id=row["image_id"],
                check_id=check, result="pass", source_evidence=relative(archive_path) + "::" + row["source_image_id"],
                output_evidence=relative(receipt), details="Preparer self-check: " + detail))
    review_path = output / "review_results.csv"
    frame = pd.DataFrame(checks, columns=pd.read_csv(templates / "review_results.csv").columns)
    if review_path.exists():
        previous = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        # Keep independent findings through targeted repairs; their binding controls acceptance.
        frame = pd.concat([frame, previous[previous.reviewer.ne("")]], ignore_index=True)
    with formal_publish_lease(review_path, directory=review_path.parent):
        frame.to_csv(review_path, index=False, encoding="utf-8-sig")


publish_reviews()
if args.metadata_only:
    progress("metadata_prepared", len(records), sample_rows=120, patient_linked_samples=120, gaps=["独立复核待执行"])
    raise SystemExit(0)

# Decode and publish one complete acquisition at a time, retaining resumable indices.
with zipfile.ZipFile(archive_path) as archive:
    for proof in sorted(selected, key=lambda item: int(Path(item["source_member"]).stem[1:])):
        image_id = Path(proof["source_member"]).stem
        destination = output / "images" / image_id / "input.tiff"
        receipt_path = receipts_dir / (image_id + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["panel_binding"] != binding:
                raise ValueError("Existing input panel changed; targeted re-export decision required")
            if receipt["admitted"] and not destination.exists():
                raise ValueError("Published receipt is missing its TIFF")
            if receipt["admitted"] and image_id not in {row["image_id"] for row in records}:
                records.append(receipt["image_metadata"])
                save_input_tables(output, pd.DataFrame(records), samples, panel, fields)
            continue
        progress("extracting", len(records), current_image=image_id, gaps=["独立复核待执行"])
        if shutil.disk_usage(root).free < 100 * 1024**3 or shutil.disk_usage(temporary_root).free < 64 * 1024**3 + 3 * archive.getinfo(proof["source_member"]).file_size:
            raise RuntimeError("Insufficient reserved disk space for this acquisition")
        text_path = temporary_root / (image_id + ".txt")
        if not text_path.exists() or text_path.stat().st_size != proof["zip_member_stat"]["file_size"]:
            partial = text_path.with_suffix(".extracting")
            with archive.open(proof["source_member"]) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024**2)
            os.replace(partial, text_path)
        reader = TxtRaster(text_path)
        try:
            progress("exporting", len(records), current_image=image_id, source_shape=list(reader.shape),
                     estimated_square_bytes=35 * max(reader.shape[1:])**2 * 4, gaps=["独立复核待执行"])
            result = export_input(reader, panel, destination)
            row = {}
            if result["admitted"]:
                c = clinical.set_index("Sample ID").loc[image_id]
                row = dict(study=STUDY, image_id=image_id, sample_id=image_id, panel_id=PANEL,
                    image_kind="tma_core", source_image_id=proof["source_member"], source_files=[relative(archive_path)],
                    source_format="txt", source_axes="records(X,Y,Z); measurement_columns=C",
                    source_revision="Zenodo 16614871 v1 (accepted revision 4)",
                    core_id=image_id, region_annotation=json.dumps({"diagnosis": c["Standard Outcoming Diagnosis"],
                        "location": c["Tumour Location"] if pd.notna(c["Tumour Location"]) else None}),
                    source_x0_px=0, source_y0_px=0, source_x1_px=reader.shape[2], source_y1_px=reader.shape[1],
                    input_image_path=relative(destination),
                    valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
                    intensity_description="Original TXT measurement values decoded as float32 following readimc.TXTFile; no normalization or transform",
                    split_method="already_independent", split_evidence="One original TXT acquisition per published Sample ID; retain full XY acquisition rectangle",
                    metadata_notes="Pixel size not asserted without acquisition-scale evidence. First 6 TXT columns are metadata; panel records their measurement-column offset.")
                for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_dtype", "source_channel_count", "channel_count"]:
                    row[key] = result[key]
                # Every TIFF header and representative source/output tiles are preparer checks.
                comparisons = []
                padding_window = None
                kept = panel[panel.keep_channel].sort_values("output_channel_index")
                # Read windows before series materialization: the shared reader
                # currently assumes TiffPage, while lazy series can use TiffFrame.
                with tifffile.TiffFile(destination) as tif:
                    names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter() if node.tag.endswith("}Channel")]
                    assert names == kept.marker_id.tolist()
                    assert len(tif.pages) == 35 and tif.pages[0].dtype == np.dtype("float32")
                    h, w = reader.shape[1:]
                    for x0, y0 in [(max(0, w // 2 - 16), max(0, h // 2 - 16)), (max(0, w - 32), max(0, h - 32))]:
                        window = (x0, y0, min(w, x0 + 32), min(h, y0 + 32))
                        expected = reader.read_window(kept.source_channel_index.tolist(), window)
                        target_window = tuple(v + (result["pad_left_px"] if i % 2 == 0 else result["pad_top_px"]) for i, v in enumerate(window))
                        actual = np.stack([read_tiff_window(tif, i, target_window) for i in range(35)])
                        assert np.array_equal(expected, actual)
                        comparisons.append(list(window))
                    if result["pad_left_px"]:
                        padding_window = (0, result["pad_top_px"], min(32, result["pad_left_px"]), result["pad_top_px"] + min(32, h))
                    elif result["pad_top_px"]:
                        padding_window = (result["pad_left_px"], 0, result["pad_left_px"] + min(32, w), min(32, result["pad_top_px"]))
                    if padding_window:
                        assert not any(read_tiff_window(tif, i, padding_window).any() for i in range(35))
                if image_id == "T1":
                    from readimc import TXTFile
                    with TXTFile(text_path) as native_txt:
                        independent = native_txt.read_acquisition()
                    assert independent.shape == reader.shape and independent.dtype == reader.dtype
                    assert np.array_equal(independent, reader.values)
                    del independent
                if image_id in {"T1", "T48", "T125"}:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    for axis, name in zip(axes, ["DNA1(Ir191Di)", "GFAP(Pr141Di)", "CD68(Tb159Di)"]):
                        plane = reader.values[source_names.index(name), ::2, ::2]
                        limit = np.percentile(plane[plane > 0], 99) if np.any(plane > 0) else 1
                        axis.imshow(plane, cmap="gray", vmin=0, vmax=limit)
                        axis.set_title(name)
                        axis.axis("off")
                    fig.suptitle(image_id + " — display scaling only")
                    fig.savefig(cache / (image_id + "_overview.png"), dpi=120)
                    plt.close(fig)
                records = [record for record in records if record["image_id"] != image_id] + [row]
                save_input_tables(output, pd.DataFrame(records), samples, panel, fields)
            write_json(receipt_path, dict(image_id=image_id, source_member=proof["source_member"],
                panel_binding=binding, image_metadata=row, source_shape=list(reader.shape),
                measured_pixels=int(reader.valid.sum()), acquisition_rectangle_pixels=reader.shape[1] * reader.shape[2],
                source_z_is_row_counter=reader.z_is_row_counter,
                readimc_full_raster_comparison=image_id == "T1" and result["admitted"],
                padding_comparison_window=padding_window if result["admitted"] else None,
                pixel_comparison_windows=comparisons if result["admitted"] else [], **result))
            publish_reviews()
            progress("image_completed", len(records), last_image=image_id, sample_rows=120,
                     sample_linked_images=len(records), patient_linked_samples=120, gaps=["独立复核待执行"])
            print(json.dumps(dict(image=image_id, admitted=result["admitted"], completed=len(records)), ensure_ascii=False), flush=True)
        finally:
            reader.close()
        text_path.unlink()
publish_reviews()
progress("prepared_for_independent_review" if len(selected) == 120 else "selected_batch_completed", len(records), sample_rows=120,
         sample_linked_images=len(records), patient_linked_samples=120,
         selected_source_images=len(selected), remaining_source_images=120 - len(list(receipts_dir.glob("*.json"))),
         gaps=["独立复核待执行", "PFS起点及事件定义不明；标准化结局保持空值",
               "物理像素尺寸未确认；pixel_size_x/y留空",
               "患者连接层级为公开完整Case ID；独立患者人数仍UNKNOWN"])
