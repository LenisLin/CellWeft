"""Prepare Merkus cSCC native IMC ROI TIFFs and source-linked samples.

Inputs: official S-BIAD2962 TIFFs, reviewed native manifest, ordered PageNames,
and study rows in S1. Outputs: per-ROI input.tiff and study input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    export_input,
    panel_from_s1,
    save_input_tables,
    study_input_staging_root,
)


STUDY = "Merkus2026_IMC_cSCC"
PROJECT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
    "CELLWEFT_DATA_ROOT", "/mnt/NAS_21T/ProjectData/CellWeft")))
parser.add_argument("--progress-path", type=Path)
args = parser.parse_args()

data_root = args.data_root
raw_root = data_root / "raw/public" / STUDY / "S-BIAD2962"
metadata_root = metadata_source_directory(data_root, STUDY)
output_root = data_root / "processed" / STUDY
fact_root = metadata_root / "input_facts"
progress_path = args.progress_path or (study_input_staging_root(output_root) / "progress.json")
manifest_path = metadata_root / "native_manifest.json"
ordered_path = metadata_root / "ordered_markers.json"
s1_path = data_root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
template_root = PROJECT / "docs/data/templates/study_input"
image_columns = pd.read_csv(template_root / "image_metadata.csv", nrows=0).columns.tolist()
sample_columns = pd.read_csv(template_root / "sample_metadata.csv", nrows=0).columns.tolist()
field_columns = pd.read_csv(template_root / "field_mapping.csv", nrows=0).columns.tolist()
manifest = json.loads(manifest_path.read_text())
source_rows = manifest["sources"]
ordered_names = [item["source_name"] for item in json.loads(ordered_path.read_text())["channels"]]
panel_id = "S-BIAD2962_native_49"
panel = panel_from_s1(STUDY, panel_id, ordered_names, s1_path)
kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
expected_names = kept.marker_id.tolist()
assert len(source_rows) == 43 and len(ordered_names) == 49 and len(kept) == 43
assert len({item["roi_id"] for item in source_rows}) == len(source_rows)

# The official file list identifies biological samples and donor labels directly.
by_sample = {}
for item in source_rows:
    key = item["sample"]
    if key in by_sample:
        assert (by_sample[key]["patient_label"], by_sample[key]["group"]) == (
            item["patient_label"], item["group"])
    else:
        by_sample[key] = item

samples = []
for sample_id, item in sorted(by_sample.items()):
    row = {column: None for column in sample_columns}
    row.update(study=STUDY, sample_id=sample_id,
               source_sample_id=sample_id,
               patient_key=f"{STUDY}:patient:{item['patient_label']}",
               source_subject_id=item["patient_label"],
               tissue_type="tumor", anatomical_site="skin", lesion_type="primary",
               case_control_status="case", primary_diagnosis="cutaneous squamous cell carcinoma",
               primary_site="skin",
               metadata_notes="Official BioImages file-list group is preserved in source_group; metastatic status refers to the primary lesion cohort.")
    row["source_group"] = item["group"]
    samples.append(row)
samples_table = pd.DataFrame(samples, columns=[*sample_columns, "source_group"])

official_file_list = (
    "retained source-query evidence "
    "q02_biad2962_BioImageArchive.json.body")
field_rows = [
    dict(study=STUDY, source_file=official_file_list,
         source_sheet_or_table="S-BIAD2962 official file list", source_field="path;id;sample;patient;group",
         target_table="image_metadata;sample_metadata",
         target_field="source_image_id;roi_id;region_annotation;sample_id;source_sample_id;patient_key;source_subject_id;source_group",
         join_keys="file id -> sample; sample -> patient; all repeated sample rows agree",
         value_mapping=None, transformation_rule="One source TIFF is one ROI. Source sample identifies a lesion; source patient label identifies a donor within this study.",
         time_origin_rule=None, conflict_rule="Do not infer acquisition time from sample name or infer un-imaged lesions.",
         notes="43 source ROI files link to 21 imaged samples and five source patient labels. Publication describes 25 lesions in the full cohort."),
    dict(study=STUDY, source_file=official_file_list,
         source_sheet_or_table="S-BIAD2962 official file list", source_field="group",
         target_table="sample_metadata", target_field="source_group", join_keys="sample",
         value_mapping=None, transformation_rule="Preserve the original source group label for each sample.",
         time_origin_rule=None, conflict_rule="No per-sample stage, event date or treatment value is inferred.",
         notes=json.dumps({"data_type": "string", "definition": "Official BioImages metastatic versus non-metastatic primary cSCC group label."})),
    dict(study=STUDY, source_file=str(metadata_source_relative(metadata_root / 'source_facts.reviewed.json', data_root)),
         source_sheet_or_table="reviewed study abstract and specimen protocol",
         source_field="people_basis;sample_preparation_protocol;human",
         target_table="sample_metadata",
         target_field="tissue_type;anatomical_site;lesion_type;case_control_status;primary_diagnosis;primary_site",
         join_keys="study accession S-BIAD2962; applies only to source-linked tumor samples",
         value_mapping=None,
         transformation_rule="Record study-confirmed skin primary cSCC tissue context for the published lesion samples.",
         time_origin_rule=None,
         conflict_rule="Do not propagate cohort-level counts or unstated stage/treatment values to individual samples.",
         notes="Study abstract describes 25 primary cSCC lesions across five recipients; official ROI file list represents 21 lesion samples."),
    dict(study=STUDY, source_file=str(metadata_source_relative(ordered_path, data_root)),
         source_sheet_or_table="native PageName index + S1", source_field="channel_index;source_name;Original marker;Unified marker;Segmentation channel",
         target_table="panel_mapping", target_field="source_name;marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
         join_keys="Study + exact PageName/Original marker", value_mapping=None,
         transformation_rule="Keep S1 admitted protein measurements and both selected DNA nuclear inputs in native order.",
         time_origin_rule=None, conflict_rule="Preserve all 49 source rows and S1 mapping-pending identities.",
         notes="43 retained output channels; six excluded elemental/background channels."),
    dict(study=STUDY, source_file=str(metadata_source_relative(raw_root, data_root)),
         source_sheet_or_table="native TIFF IFD chain", source_field="PageName;ImageWidth;ImageLength;SampleFormat;XResolution;YResolution;ResolutionUnit",
         target_table="image_metadata", target_field="source_dtype;source_channel_count;source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um;input_image_path",
         join_keys="official file path -> TIFF filename", value_mapping=None,
         transformation_rule="Read all 49 scalar pages in IFD order with each file's native dtype; use the complete source ROI and retain native values. Resolution 25400 pixels/inch gives 1 um/pixel.",
         time_origin_rule=None, conflict_rule="Source OME-XML describes individual pages without SizeC; use IFD PageName order.",
         notes="Rectangular ROI frames receive centered square zero padding; each source file contains one sample."),
]
field_mapping = pd.DataFrame(field_rows, columns=field_columns)

images = pd.read_parquet(output_root / "images.parquet").to_dict("records") if (
    output_root / "images.parquet").is_file() else []
completed = {row["image_id"] for row in images}
fact_root.mkdir(parents=True, exist_ok=True)
excluded_path = fact_root / "input_excluded_images.json"
excluded = json.loads(excluded_path.read_text()) if excluded_path.is_file() else []
excluded_ids = {row["image_id"] for row in excluded}
new_count = 0


def save_progress(stage: str, current: str | None = None) -> None:
    progress = dict(study=STUDY, stage=stage, current_image=current,
                    expected_images=len(source_rows), exported_images=len(images),
                    excluded_images=len(excluded), new_images_this_run=new_count,
                    sample_rows=len(samples_table), sample_linked_images=len(images),
                    patient_linked_samples=len(samples_table),
                    next_image=next((item["roi_id"] for item in source_rows
                                     if item["roi_id"] not in completed | excluded_ids), None),
                    updated_at=datetime.now(timezone.utc).isoformat())
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    os.replace(temporary, progress_path)


def publish_tables() -> None:
    save_input_tables(output_root, pd.DataFrame(images, columns=image_columns),
                      samples_table, panel, field_mapping)


def output_header_matches(path: Path, side: int, dtype: str) -> bool:
    with tifffile.TiffFile(path) as tif:
        names = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        return (len(tif.pages) == len(kept) and tif.pages[0].shape == (side, side)
                and tif.pages[0].dtype == np.dtype(dtype) and names == expected_names)


# Reuse only previously indexed outputs whose source identity and TIFF header still agree.
source_by_id = {item["roi_id"]: item for item in source_rows}
for row in images:
    item = source_by_id[row["image_id"]]
    path = raw_root / item["name"]
    output_path = data_root / row["input_image_path"]
    if (path.stat().st_size != item["size"] or row["sample_id"] != item["sample"]
            or row["source_image_id"] != item["roi_id"]
            or not output_header_matches(output_path, int(row["width_px"]), row["source_dtype"])):
        raise ValueError(f"Existing indexed input differs from source: {row['image_id']}")

save_progress("source_check")
for item in source_rows:
    image_id = item["roi_id"]
    if image_id in completed | excluded_ids:
        continue
    source_path = raw_root / item["name"]
    if source_path.stat().st_size != item["size"]:
        raise ValueError(f"Official source size differs: {source_path}")
    with tifffile.TiffFile(source_path, is_ome=False) as tif:
        native_dtype = tif.pages[0].dtype
        if (len(tif.pages) != len(ordered_names)
                or [page.tags["PageName"].value for page in tif.pages] != ordered_names
                or any(page.shape != tif.pages[0].shape or page.dtype != native_dtype
                       for page in tif.pages)):
            raise ValueError(f"Native IFD layout differs: {source_path}")
        height, width = tif.pages[0].shape
        x_resolution = tif.pages[0].tags["XResolution"].value
        y_resolution = tif.pages[0].tags["YResolution"].value
        resolution_unit = tif.pages[0].tags["ResolutionUnit"].value
        if (resolution_unit != 2 or x_resolution[0] / x_resolution[1] != 25400
                or y_resolution[0] / y_resolution[1] != 25400):
            raise ValueError(f"Unexpected native pixel scale: {source_path}")
    side = max(height, width)

    image_dir = output_root / "images" / image_id
    input_path = image_dir / "input.tiff"
    save_progress("exporting", image_id)
    source = SourceSpec(image_id=image_id, panel_id=panel_id, kind="tiff", path=source_path,
                        channels=tuple(range(len(ordered_names))), pixel_size_um=(1.0, 1.0),
                        source_revision=f"S-BIAD2962:{item['name']}:size={item['size']}",
                        intensity_description=f"native {native_dtype} IMC page values; no intensity normalization")
    with open_source(source) as reader:
        result = export_input(reader, panel, input_path, tile_size=1024)
    if not result["admitted"]:
        excluded.append(dict(image_id=image_id, source_file=item["name"],
                             reason="all admitted downstream markers zero across full ROI"))
        excluded_ids.add(image_id)
        excluded_path.write_text(json.dumps(excluded, indent=2) + "\n")
        save_progress("excluded", image_id)
        continue
    if not output_header_matches(input_path, side, str(native_dtype)):
        raise ValueError(f"Published TIFF header differs: {input_path}")
    row = {column: None for column in image_columns}
    row.update(study=STUDY, image_id=image_id, sample_id=item["sample"], panel_id=panel_id,
               image_kind="roi", source_image_id=image_id,
               source_files=[str(metadata_source_relative(source_path, data_root))], source_format="tiff",
               source_axes="CYX", source_dtype=str(native_dtype), source_revision=source.source_revision,
               roi_id=image_id, region_annotation=image_id,
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=side, width_px=side, source_channel_count=len(ordered_names),
               channel_count=len(kept), pixel_size_x_um=1.0, pixel_size_y_um=1.0,
               input_image_path=str(metadata_source_relative(input_path, data_root)),
               intensity_description=source.intensity_description,
               split_method="already_independent",
               split_evidence="Official file id and sample attribute identify a complete independent ROI frame.",
               metadata_notes="Source group and donor label are linked through the official file list to sample_metadata.csv.")
    images.append(row)
    completed.add(image_id)
    new_count += 1
    publish_tables()
    save_progress("exported", image_id)
    print(f"exported {len(images)}/{len(source_rows)} {image_id}", flush=True)

publish_tables()
excluded_path.write_text(json.dumps(excluded, indent=2) + "\n")
pixel_checks = []
checked_layouts = set()
for row in images:
    item = source_by_id[row["image_id"]]
    layout = (row["source_dtype"], int(row["source_x1_px"]), int(row["source_y1_px"]))
    if layout in checked_layouts:
        continue
    checked_layouts.add(layout)
    source_path = raw_root / item["name"]
    input_path = data_root / row["input_image_path"]
    width, height = layout[1:]
    windows = [(0, 0, min(32, width), min(32, height)),
               (max(0, width//2-16), max(0, height//2-16),
                min(width, width//2+16), min(height, height//2+16))]
    with formal_read_lease(source_path), formal_read_lease(input_path):
        with tifffile.TiffFile(source_path, is_ome=False) as source_tif, tifffile.TiffFile(input_path) as output_tif:
            for channel in kept.itertuples(index=False):
                for window in windows:
                    x0, y0, x1, y1 = window
                    output_window = (x0+int(row["pad_left_px"]), y0+int(row["pad_top_px"]),
                                     x1+int(row["pad_left_px"]), y1+int(row["pad_top_px"]))
                    source_values = read_tiff_window(source_tif, int(channel.source_channel_index), window)
                    output_values = read_tiff_window(output_tif, int(channel.output_channel_index), output_window)
                    if not np.array_equal(source_values, output_values):
                        raise ValueError(f"Preparer pixel mismatch: {row['image_id']} channel {channel.source_channel_index} window {window}")
    pixel_checks.append(dict(image_id=row["image_id"], source_dtype=row["source_dtype"],
                             source_width=width, source_height=height, channels=len(kept),
                             source_windows=windows, result="exact_equal"))
(fact_root / "preparer_pixel_checks.json").write_text(json.dumps(pixel_checks, indent=2) + "\n")
summary = dict(study=STUDY, source_files=len(source_rows), exported_images=len(images),
               excluded_images=len(excluded), sample_rows=len(samples_table),
               sample_linked_images=len(images), patient_linked_samples=len(samples_table),
               distinct_patient_labels=samples_table.patient_key.nunique(),
               source_evidence=[str(metadata_source_relative(manifest_path, data_root)), official_file_list],
               metadata_scope="21 imaged lesions among 25 described study lesions; no stage or treatment per sample provided")
(fact_root / "input_delivery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

# Self-checks are preparer evidence; an independent reviewer fills the final D01.
review_path = output_root / "review_results.csv"
review = pd.read_csv(review_path, keep_default_na=False) if review_path.is_file() else pd.read_csv(
    template_root / "review_results.csv", keep_default_na=False)
record_id = f"{STUDY}:preparer_delivery"
review = review[review.record_id != record_id]
checks = [
    ("S01", "pass", f"Official 43 ROI files reconciled: {len(images)} admitted, {len(excluded)} excluded; full-frame marker signal checked on export."),
    ("P01", "pass", "All 49 native IFD PageNames match ordered_markers.json and exact S1 source labels."),
    ("P02", "pass", "43 channels retained in source order, including both S1-selected DNA nuclear inputs."),
    ("P03", "pass", f"All {len(images)} saved TIFF headers reopened for CYX native dtype, 43 channel names and geometry."),
    ("I05", "pass", f"Preparer exact pixel comparison across all 43 retained channels, border and center blocks for each native dtype/geometry layout; see preparer_pixel_checks.json ({len(pixel_checks)} images)."),
    ("M01", "pass", f"Official file-list id -> sample -> patient attributes link {len(images)} images to {len(samples_table)} imaged samples and five donor labels; 25 lesions in study cohort, 21 represented in published ROI files."),
    ("D01", "unresolved", "Preparer delivery closed; independent review and acceptance pending."),
]
review_rows = [dict(study=STUDY, preparer=Path(__file__).name, reviewer="", scope="study",
                    record_id=record_id, check_id=check_id, result=outcome,
                    source_evidence=official_file_list, output_evidence=str(metadata_source_relative(output_root, data_root)),
                    details=details) for check_id, outcome, details in checks]
review = pd.concat([review, pd.DataFrame(review_rows)], ignore_index=True)
with formal_publish_lease(review_path, directory=output_root):
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
save_progress("prepared_awaiting_independent_review")
print(json.dumps(summary, indent=2), flush=True)
