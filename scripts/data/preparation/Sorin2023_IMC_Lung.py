"""Prepare Sorin's 536 published IMC core images as native-intensity inputs.

Inputs: Zenodo 7760826 LungData.zip, its clinical workbook and ordered panel.
Outputs: private candidate S1, source inventory, study input tables, per-core TIFFs.
The author MAT files describe nuclei; no whole-cell mask is asserted here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import openpyxl
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)


STUDY = "Sorin2023_IMC_Lung"
NAMES = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3",
         "CD4", "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "MPO",
         "Pancytokeratin", "TTF1"]
BODY = {"CD31", "CD3", "CD68", "HLA-DR", "Pancytokeratin"}
SOURCE_SIGNAL = ("Author-published masked uint8 multiplex signal; sampled protein channels are quantized. "
                 "Retain published source intensity; not instrument raw counts; CellWeft normalization not yet applied.")
MEMBER_ROOT = "LungData/LUAD_IMC_MaskTif/"
CLINICAL_MEMBER = "LungData/LUAD Clinical Data.xlsx"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--output-root", type=Path)
parser.add_argument("--jobdir", type=Path, default=Path(__file__).resolve().parents[3] /
                    "cache/new_studies_20261107" / STUDY)
parser.add_argument("--s1", type=Path)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--limit", type=int)
parser.add_argument("--shard-count", type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
args = parser.parse_args()
assert 0 <= args.shard_index < args.shard_count
root = args.data_root
archive_path = root / "raw/public" / STUDY / "LungData.zip"
# Keep decompressed ZIP members on the SSD; the raw ZIP remains canonical on NAS.
source_staging = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime" / STUDY / "source_members"
job = args.jobdir
output = args.output_root or root / "processed" / STUDY
formal_s1 = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
candidate_s1 = job / "candidate_s1_rows.csv"
panel_id = "sorin2023_zenodo7760826_18"
for directory in (source_staging, job):
    directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def relative(path: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def progress(stage: str, exported: int = 0, issue: str | None = None) -> None:
    progress_path = job / ("progress.json" if args.shard_count == 1 else
                           f"progress_shard{args.shard_index}.json")
    save_json(progress_path, dict(
        study=STUDY, phase="prepare", stage=stage, status="prepared" if exported == 536 else "partial",
        shard_count=args.shard_count, shard_index=args.shard_index,
        source_images=536, expected_images=536, exported_images=exported,
        sample_rows=536, sample_linked_images=536,
        patient_linked_samples=416, patient_linked_images=416,
        unique_patient_keys=416,
        issue=issue, gaps=["validation的120图缺逐行patient crosswalk；编号配对仅作候选证据，patient_key留空",
                           "作者MAT为核分割；whole-cell mask未提供",
                           "像素物理尺度未在来源TIFF中编码",
                           "论文35-plex完整panel与本归档18通道可用panel不同",
                           "TTF1在当前正式S1无可复用同义映射，候选保留mapping pending",
                           "独立复核和D01待主控安排"],
        updated_utc=datetime.now(timezone.utc).isoformat()))


# Fix the candidate row identity before potentially slow image-member reads.
existing = pd.read_csv(formal_s1, dtype=str, keep_default_na=False)
formal_study = existing[existing.Study == STUDY]
assert len(formal_study) in {0, 18}
candidate_rows = []
if formal_study.empty:
    for index, name in enumerate(NAMES, 1):
        matches = existing[existing["Original marker"] == name]
        if name == "TTF1":
            mapping = dict(zip(existing.columns, [STUDY, name, "TTF1 (mapping pending)", "", "", ""]))
            reuse = "No exact original/unified TTF1 row in current formal S1; protein mapping unresolved"
        else:
            values = matches[["Unified marker", "MAP-KG mapping", "UniProt accession"]].drop_duplicates()
            assert len(values) == 1, (name, values)
            mapping = dict(zip(existing.columns, [STUDY, name, *values.iloc[0].tolist(), ""]))
            reuse = "same Original marker: " + ";".join(matches.Study.head(5).tolist())
        mapping["Segmentation channel"] = "1" if name == "DNA1" else "2" if name in BODY else ""
        mapping["source_channel_index_1based"] = index
        mapping["source_evidence"] = (
            "Zenodo 7760826 metadata.description Channel index names; " + MEMBER_ROOT + "*.tif page " + str(index))
        mapping["mapping_reuse_evidence"] = reuse
        mapping["segmentation_choice_evidence"] = (
            "source DNA1 actual nuclear stain" if name == "DNA1" else
            "current S1 same-class body candidate; proposed, downstream intensity untested" if name in BODY else "")
        candidate_rows.append(mapping)
if formal_study.empty and not candidate_s1.is_file():
    pd.DataFrame(candidate_rows).to_csv(candidate_s1, index=False, encoding="utf-8-sig")

# Reconcile all archive image members and their paired source objects.
with zipfile.ZipFile(archive_path) as archive:
    members = archive.infolist()
    tiffs = sorted((item for item in members if item.filename.startswith(MEMBER_ROOT)
                    and item.filename.lower().endswith(".tif")), key=lambda item: item.filename)
    assert len(tiffs) == 536 and archive_path.stat().st_size == 2088454891
    stems = [Path(item.filename).stem for item in tiffs]
    assert len(set(stems)) == 536
    all_names = {item.filename for item in members}
    previous_path = job / "inventory.json"
    prior_items = {}
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text())
        if (previous.get("archive_bytes") == archive_path.stat().st_size
                and previous.get("archive_mtime_ns") == archive_path.stat().st_mtime_ns):
            prior_items = {entry["image_id"]: entry for entry in previous["images"]
                           if "source_shape" in entry}
    if len(prior_items) != len(tiffs):
        save_json(job / "inventory.json", dict(
            study=STUDY, stage="archive_members_verified_tiff_headers_pending",
            source_archive=relative(archive_path), archive_bytes=archive_path.stat().st_size,
            archive_mtime_ns=archive_path.stat().st_mtime_ns,
            archive_members=len(members), source_images=len(tiffs),
            groups={"discovery": sum("LUAD_D" in stem for stem in stems),
                    "validation": sum("LUAD_V" in stem for stem in stems)},
            source_panel_order_1based=NAMES,
            images=[dict(image_id=Path(item.filename).stem, archive_member=item.filename,
                         compressed_bytes=item.compress_size, source_bytes=item.file_size,
                         zip_crc32=item.CRC) for item in tiffs]))
    progress("archive_inventory_written;_tiff_header_scan_pending")
    items = []
    for item in tiffs:
        stem = Path(item.filename).stem
        nucleus = f"LungData/LUAD_IMC_Segmentation/{stem}/nuclei_multiscale.mat"
        regions = f"LungData/LUAD_IMC_Segmentation/{stem}/allRegionIndices.mat"
        cell_type = f"LungData/LUAD_IMC_CellType/{stem}.mat"
        assert {nucleus, regions, cell_type} <= all_names
        prior = prior_items.get(stem)
        if (prior is not None and prior["archive_member"] == item.filename
                and prior["source_bytes"] == item.file_size
                and prior["zip_crc32"] == item.CRC):
            items.append(prior)
            continue
        with tifffile.TiffFile(io.BytesIO(archive.read(item))) as tif:
            shape = list(tif.series[0].shape)
            dtype = str(tif.series[0].dtype)
            axes = tif.series[0].axes
            assert len(tif.pages) == 18 and shape[0] == 18 and dtype == "uint8"
            assert all(page.shape == tuple(shape[1:]) and page.dtype == np.dtype("uint8")
                       for page in tif.pages)
            assert "XResolution" not in tif.pages[0].tags
        items.append(dict(image_id=stem, archive_member=item.filename,
                          compressed_bytes=item.compress_size, source_bytes=item.file_size,
                          zip_crc32=item.CRC, source_shape=shape, source_axes=axes,
                          source_dtype=dtype, source_pixel_size_um=None,
                          author_nucleus_members=[nucleus, regions],
                          author_cell_type_member=cell_type))
    workbook = openpyxl.load_workbook(io.BytesIO(archive.read(CLINICAL_MEMBER)),
                                      read_only=True, data_only=True)
    sheets = {sheet.title: list(sheet.values) for sheet in workbook}
    workbook.close()
assert len(members) == 2698
assert {str(row[0]) for rows in sheets.values() for row in rows[1:]} == set(stems)
assert len(sheets["LUAD_416_Discovery"]) == 417
assert len(sheets["LUAD_120_Validation"]) == 121
inventory = dict(study=STUDY, source_archive=relative(archive_path),
                 archive_bytes=archive_path.stat().st_size,
                 archive_mtime_ns=archive_path.stat().st_mtime_ns,
                 archive_members=len(members), source_images=len(items),
                 groups={"discovery": 416, "validation": 120},
                 source_panel_order_1based=NAMES,
                 panel_source="Zenodo 7760826 metadata.description Channel index names",
                 clinical_member=CLINICAL_MEMBER,
                 author_segmentation_semantics="nuclei only; README and nucleiOccupancyIndexed",
                 source_pixel_size_um=None,
                 images=items)
if len(prior_items) != len(tiffs):
    save_json(job / "inventory.json", inventory)

progress("inventory_and_candidate_s1")

if args.metadata_only:
    print(json.dumps(dict(study=STUDY, source_images=536, inventory=str(job / "inventory.json"),
                          candidate_s1=str(candidate_s1), stage="metadata_only")))
    raise SystemExit(0)

# Keep each core as its own sample. Cohort counts and A/B filenames support a
# pairing proposal, but do not establish the missing row-level patient bridge.
clinical = {}
for sheet_name, rows in sheets.items():
    for row in rows[1:]:
        clinical[str(row[0])] = (sheet_name, dict(zip(rows[0], row)))
validation_keys = {key for key, (sheet, _) in clinical.items()
                   if sheet == "LUAD_120_Validation"}
assert validation_keys == {f"LUAD_V{number:02d}{suffix}"
                           for number in range(1, 61) for suffix in ("A", "B")}
for number in range(1, 61):
    first, second = (clinical[f"LUAD_V{number:02d}{suffix}"][1]
                     for suffix in ("A", "B"))
    assert first["Progression (No: 0, Yes: 1) "] == second["Progression (No: 0, Yes: 1) "]
sample_rows = []
for image_id in stems:
    sheet, record = clinical[image_id]
    patient_key = image_id if sheet == "LUAD_416_Discovery" else None
    row = dict.fromkeys(columns("sample_metadata"), None)
    row.update(study=STUDY, sample_id=image_id,
               patient_key=patient_key, source_subject_id=patient_key,
               source_sample_id=image_id, cohort_id=sheet, sample_type="tumor core",
               anatomical_site="lung", primary_diagnosis="lung adenocarcinoma",
               primary_site="lung", case_control_status="case",
               metadata_notes=json.dumps(record, ensure_ascii=False, default=str))
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)

field_rows = []
def field(source_field: str, target_table: str, target_field: str, rule: str,
          evidence: str) -> None:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(archive_path),
               source_sheet_or_table=(CLINICAL_MEMBER if source_field in
                                      {"Key", "clinical workbook fields"} else "Zenodo 7760826"),
               source_field=source_field, target_table=target_table,
               target_field=target_field, join_keys="workbook Key = TIFF basename",
               transformation_rule=rule, notes=evidence)
    field_rows.append(row)

field("Key", "sample_metadata", "sample_id;source_sample_id;patient_key",
      "Exact Key joins 536/536 TIFFs; discovery Key is one per 416 patients; validation patient_key and source_subject_id remain empty",
      "The paper confirms 60 validation patients/120 cores, but no row-key patient crosswalk. A/B filename pairing remains an unconfirmed proposal: https://www.nature.com/articles/s41586-022-05672-3")
field("clinical workbook fields", "sample_metadata", "metadata_notes",
      "Retain source fields and coded values as JSON without interpreting timing or binary categories",
      "Original workbook values only; no clinical endpoint normalization")
field("Channel index names", "panel_mapping", "source_name;source_channel_index",
      "Zenodo 1-based channel order to zero-based panel index",
      "Zenodo 7760826; 18 uint8 pages per TIFF")
field("TIFF basename; TIFF pages", "image_metadata", "source_image_id;source_axes;source_dtype;source_channel_count",
      "One source image per image; retain full original acquisition rectangle",
      "536 exact image/clinical keys; no documented multi-sample geometry")
fields = pd.DataFrame(field_rows)
panel = panel_from_s1(STUDY, panel_id, NAMES,
                      args.s1 or (formal_s1 if len(formal_study) else candidate_s1))
assert panel.keep_channel.sum() == 18 and panel.nuclear.sum() == 1 and panel.body.sum() == 5

images = (pd.read_parquet(output / "images.parquet") if (output / "images.parquet").is_file()
          else pd.DataFrame(columns=columns("image_metadata")))
completed = set(images.image_id)
if len(images):
    images["intensity_description"] = SOURCE_SIGNAL
remaining = [item for item in items if item["image_id"] not in completed]
if args.shard_count > 1:
    index_by_image = {item["image_id"]: index for index, item in enumerate(items)}
    remaining = [item for item in remaining
                 if index_by_image[item["image_id"]] % args.shard_count == args.shard_index]
initial_qc = ["LUAD_D001", "LUAD_D082", "LUAD_V01A"]
remaining = sorted(remaining, key=lambda item: (
    initial_qc.index(item["image_id"]) if item["image_id"] in initial_qc else len(initial_qc),
    item["image_id"]))
if args.limit is not None:
    remaining = remaining[:args.limit]
progress("exporting", len(images))

# Extract and verify one archive member at a time into SSD-only runtime staging.
source_dir = source_staging
source_dir.mkdir(parents=True, exist_ok=True)
for item in remaining:
    image_id = item["image_id"]
    source_path = source_dir / f"{image_id}.tif"
    if not source_path.is_file() or source_path.stat().st_size != item["source_bytes"]:
        with zipfile.ZipFile(archive_path) as archive:
            temporary = source_path.with_suffix(".tif.partial")
            with archive.open(item["archive_member"]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            assert temporary.stat().st_size == item["source_bytes"]
            os.replace(temporary, source_path)
    source = SourceSpec(image_id=image_id, panel_id=panel_id, kind="tiff", path=source_path,
                        channels=tuple(range(18)), source_revision="Zenodo 7760826 LungData.zip",
                        intensity_description=SOURCE_SIGNAL)
    destination = output / "images" / image_id / "input.tiff"
    with open_source(source) as reader:
        assert reader.shape == tuple(item["source_shape"]) and reader.dtype == np.dtype("uint8")
        result = export_input(reader, panel, destination, tile_size=256, data_root=root)
        assert result["admitted"] and result["channel_count"] == 18
        _, height, width = reader.shape
        side = max(height, width)
        assert result["height_px"] == result["width_px"] == side
        if image_id in initial_qc:
            checked_boxes = []
            with tifffile.TiffFile(destination) as saved:
                names = [node.attrib["Name"] for node in ET.fromstring(saved.ome_metadata)
                         .findall(".//{*}Channel")]
                assert names == panel.sort_values("output_channel_index").marker_id.tolist()
                for box in ((width // 2, height // 2, min(width, width // 2 + 32),
                             min(height, height // 2 + 32)),
                            (0, 0, min(width, 32), min(height, 32))):
                    translated = (box[0] + result["pad_left_px"],
                                  box[1] + result["pad_top_px"],
                                  box[2] + result["pad_left_px"],
                                  box[3] + result["pad_top_px"])
                    for channel in range(18):
                        assert np.array_equal(reader.read_window([channel], box)[0],
                                              read_tiff_window(saved, channel, translated))
                    checked_boxes.append(dict(source_xyxy=box, output_xyxy=translated,
                                              channels=18, exact=True))
            save_json(job / f"{image_id}_pixel_self_check.json", dict(
                image_id=image_id, group="discovery" if "_D" in image_id else "validation",
                source_archive=relative(archive_path),
                source_member=item["archive_member"], source_crc32=item["zip_crc32"],
                source_shape=item["source_shape"], output_shape=[18, side, side],
                source_dtype="uint8", output_dtype="uint8",
                padding_ltrb=[result["pad_left_px"], result["pad_top_px"],
                              result["pad_right_px"], result["pad_bottom_px"]],
                window_checks=checked_boxes, ome_names_match=True))
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=image_id, sample_id=image_id, panel_id=panel_id,
               image_kind="tma_core", source_image_id=item["archive_member"],
               source_files=[relative(archive_path)], source_format="tiff",
               source_axes=item["source_axes"], source_dtype="uint8",
               source_revision=source.source_revision, core_id=image_id,
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=side, width_px=side, source_channel_count=18, channel_count=18,
               input_image_path=relative(destination), intensity_description=source.intensity_description,
               split_method="already_independent",
               split_evidence="Zenodo 7760826 exact TIFF core key and clinical workbook Key",
               metadata_notes="Author MAT supplies nuclei segmentation, not a verified whole-cell mask")
    images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
    save_input_tables(output, images, samples, panel, fields, data_root=root)
    progress("exporting", len(images))
    print(json.dumps(dict(image_id=image_id, exported_images=len(images), expected_images=536)), flush=True)

# Refresh typed metadata once when a resumed run has no remaining pixels.
if args.shard_count == 1 and len(images):
    save_input_tables(output, images, samples, panel, fields, data_root=root)
    mapping_path = output / "field_mapping.csv"
    with formal_publish_lease(mapping_path, directory=output, data_root=root):
        recorded = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
        stale = ((recorded.study == STUDY) &
                 (recorded.source_field == "clinical workbook fields") &
                 (recorded.source_sheet_or_table == "Zenodo 7760826"))
        if stale.any():
            temporary = mapping_path.with_suffix(".csv.partial")
            recorded.loc[~stale].to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, mapping_path)

if args.shard_count > 1:
    progress("shard_complete", len(images))
    raise SystemExit(0)

review_path = output / "review_results.csv"
review_rows = []
for check_id, result, detail in (
        ("S01", "pass", "536/536 archive TIFF members have 18 uint8 pages and exact workbook Key; two groups 416/120"),
        ("M01", "unresolved", "536/536 images link source Key samples; discovery 416 patient keys; validation 120 patient keys remain empty without row-key crosswalk"),
        ("M02", "unresolved", "Original clinical coded values retained; endpoint anchors and clinical normalization not independently reviewed"),
        ("P01", "pass", "Zenodo 7760826 ordered 18-channel panel joined to private S1 candidate"),
        ("P02", "pass", "DNA1 source channel 13 retained as nuclear input; five source protein body candidates proposed"),
        ("D01", "unresolved", "Preparer self-check only; independent review and controller binding pending")):
    review_rows.append(dict(study=STUDY, preparer="Sorin2023_input_adapter", reviewer="",
                            scope="study", record_id=STUDY, check_id=check_id, result=result,
                            source_evidence=relative(archive_path) + ";" + str(job / "inventory.json"),
                            output_evidence=relative(output),
                            details=json.dumps(dict(assessment="preparer_self_check", text=detail))))
pd.DataFrame(review_rows, columns=columns("review_results")).to_csv(
    review_path, index=False, encoding="utf-8-sig")
progress("candidate_export_complete" if len(images) == 536 else "candidate_export_partial", len(images))
