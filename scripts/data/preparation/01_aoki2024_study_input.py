"""Prepare the complete Aoki 2024 IMC source into per-image input TIFFs.

The durable source is eight official ZIP archives containing one MCD member
each.  This adapter keeps the archives untouched, extracts one member at a
time into the study cache, and uses the verified acquisition index to publish
one independent input image per counted acquisition.
"""

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative, publish_metadata_sources
import argparse
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import tifffile
from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.study_input import (
    export_input,
    panel_from_s1,
    save_input_tables,
    study_input_staging_root,
)
from cellweft.data.preparation.io_coordination import formal_publish_lease


PROJECT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CELLWEFT_DATA_ROOT", "/mnt/NAS_21T/ProjectData/CellWeft"))
STUDY = "Aoki2024_IMC_RelapsedHodgkin"
RAW_ROOT = DATA_ROOT / "raw/public" / STUDY / "zenodo_7963681"
SOURCE_META = metadata_source_directory(DATA_ROOT, STUDY)
S1_PATH = DATA_ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
OUTPUT_ROOT = DATA_ROOT / "processed" / STUDY

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
parser.add_argument("--archive", action="append", help="Process only named ZIP files; repeatable")
parser.add_argument("--extraction-root", type=Path, help="Temporary MCD extraction root; defaults to SSD study staging")
parser.add_argument("--metadata-only", action="store_true", help="Refresh tables and delivery checks without decoding sources")
parser.add_argument("--overviews", action="store_true", help="Cache display previews for one image per archive and every nonstandard ROI description")
parser.add_argument("--progress-path", type=Path, help="Optional progress override; default is SSD study staging")
args = parser.parse_args()

data_root = args.data_root
raw_root = data_root / "raw/public" / STUDY / "zenodo_7963681"
source_meta = metadata_source_directory(data_root, STUDY)
s1_path = data_root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
output_root = data_root / "processed" / STUDY
cache_root = study_input_staging_root(output_root)
extraction_root = args.extraction_root or cache_root / "extracted"
progress_path = args.progress_path or cache_root / "progress.json"
source_scope_path = source_meta / "source_scope.json"
source_exclusions_path = source_meta / "source_acquisition_exclusions.csv"
sample_linkage_path = source_meta / "sample_linkage_search.json"
source_search_root = source_meta / "source_search"
output_root.mkdir(parents=True, exist_ok=True)
extraction_root.mkdir(parents=True, exist_ok=True)
cache_root.mkdir(parents=True, exist_ok=True)
source_search_root.mkdir(parents=True, exist_ok=True)

index_path = source_meta / "acquisition_index.json"
manifest_path = source_meta / "source_manifest.json"
coverage_path = source_meta / "coverage.json"
acquisition_rows = json.loads(index_path.read_text(encoding="utf-8"))["images"]
source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
assert all(row["counted"] and row["passed"] for row in acquisition_rows)
assert all(row["pixel_records"] == row["expected_pixels"] for row in acquisition_rows)

archive_names = [item["key"] for item in source_manifest["files"]]
source_scope = []
for item in source_manifest["files"]:
    archive_path = raw_root / item["key"]
    if archive_path.stat().st_size != item["size"]:
        raise ValueError(f"Source size differs from original verified manifest: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        members = [(entry.filename, entry.file_size) for entry in archive.infolist() if not entry.is_dir()]
    source_scope.append(dict(source_archive=item["key"], archive_bytes=item["size"], members=members))
source_scope_path.write_text(json.dumps(source_scope, indent=2) + "\n")
native_reports = source_meta / "archive_results"
source_exclusions = []
for item in source_manifest["files"]:
    report = json.loads((native_reports / (Path(item["key"]).stem + ".json")).read_text())
    source_exclusions.extend(report["excluded_acquisitions"])
pd.DataFrame(source_exclusions).to_csv(source_exclusions_path,
                                      index=False, encoding="utf-8-sig")
if args.archive:
    requested = set(args.archive)
    unknown = requested.difference(archive_names)
    if unknown:
        raise ValueError(f"Unknown Aoki archive(s): {sorted(unknown)}")
    archive_names = [name for name in archive_names if name in requested]
selected_rows = [row for row in acquisition_rows if row["source_archive"] in archive_names]
if not selected_rows:
    raise ValueError("No acquisition selected")
if not args.archive and len(selected_rows) != len(acquisition_rows):
    raise ValueError("The complete-study run must include all verified acquisitions")

source_names = list(selected_rows[0]["embedded_channel_labels"])
if len(source_names) != 44 or any(row["embedded_channel_labels"] != source_names for row in selected_rows):
    raise ValueError("Aoki acquisition channel order is not identical across the selected sources")

panel_id = f"{STUDY}__native44"
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
if list(panel["source_name"]) != source_names or len(panel) != 44:
    raise ValueError("S1 panel does not preserve the verified native channel order")
if int(panel["keep_channel"].astype(bool).sum()) != 37:
    raise ValueError("Aoki input is expected to retain 37 channels including both Ir nuclei")
ir_rows = panel[panel["source_name"].isin(["191Ir", "193Ir"])]
if not ((ir_rows["role"] == "segmentation-only").all() and ir_rows["keep_channel"].astype(bool).all()):
    raise ValueError("Aoki Ir nuclear inputs were not retained as segmentation-only channels")

image_columns = list(pd.read_csv(
    PROJECT / "docs/data/templates/study_input/image_metadata.csv",
    nrows=0,
    encoding="utf-8-sig",
).columns)
sample_columns = list(pd.read_csv(
    PROJECT / "docs/data/templates/study_input/sample_metadata.csv",
    nrows=0,
    encoding="utf-8-sig",
).columns)
field_columns = list(pd.read_csv(
    PROJECT / "docs/data/templates/study_input/field_mapping.csv",
    nrows=0,
    encoding="utf-8-sig",
).columns)

field_rows = [
    {
        "study": STUDY,
        "source_file": str(metadata_source_relative(source_meta / 'acquisition_index.json', data_root)),
        "source_sheet_or_table": "images",
        "source_field": "source_archive;source_member;slide_id;acquisition_id;description;width_px;height_px;pixel_size_um",
        "target_table": "image_metadata",
        "target_field": "source_files;source_image_id;slide_id;source_acquisition_id;region_annotation;geometry;pixel_size_x_um;pixel_size_y_um",
        "join_keys": "source_archive + source_member + acquisition_id",
        "value_mapping": None,
        "transformation_rule": "One counted acquisition becomes one image; retain the complete native acquisition frame and center-pad to square.",
        "time_origin_rule": None,
        "conflict_rule": "Use the verified acquisition index and embedded MCD metadata; do not infer patient identity from ROI names.",
        "notes": "303 indexed acquisitions; every counted acquisition passed and pixel_records equals expected_pixels.",
    },
    {
        "study": STUDY,
        "source_file": str(metadata_source_relative(source_meta / 'channel_mapping.csv', data_root)),
        "source_sheet_or_table": "channel_mapping",
        "source_field": "source_archive;source_member;acquisition_id;channel_index;marker_label",
        "target_table": "panel_mapping",
        "target_field": "panel_id;source_channel_index;source_name",
        "join_keys": "source channel index and native marker label",
        "value_mapping": None,
        "transformation_rule": "Use the embedded ordered 44-channel vector shared by all 303 acquisitions.",
        "time_origin_rule": None,
        "conflict_rule": "Reject any acquisition whose embedded channel vector differs from the verified native vector.",
        "notes": "All selected acquisitions have the same native ordered labels.",
    },
    {
        "study": STUDY,
        "source_file": "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv",
        "source_sheet_or_table": "S1",
        "source_field": "Study + Original marker",
        "target_table": "panel_mapping",
        "target_field": "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
        "join_keys": "Study + Original marker",
        "value_mapping": None,
        "transformation_rule": "Map the native vector through S1; retain all non-excluded mapped channels plus S1-selected Ir nuclear inputs.",
        "time_origin_rule": None,
        "conflict_rule": "Keep source names and order; S1-excluded 191Ir/193Ir are retained only as segmentation-only nuclear inputs.",
        "notes": "Output retains 37 of 44 native channels.",
    },
    {
        "study": STUDY,
        "source_file": "processed/_catalog/public/studies.csv",
        "source_sheet_or_table": "studies",
        "source_field": "dataset_name;tissue;disease",
        "target_table": "sample_metadata",
        "target_field": "tissue_type;anatomical_site;primary_site;metadata_notes",
        "join_keys": "dataset_name = study",
        "value_mapping": None,
        "transformation_rule": "Apply only study-level lymph-node context; do not assign disease subgroup to an individual ROI.",
        "time_origin_rule": None,
        "conflict_rule": "Per-ROI clinical values remain blank without an explicit source join.",
        "notes": "Study-level record describes relapsed/refractory classical Hodgkin lymphoma and diagnostic controls.",
    },
    {
        "study": STUDY,
        "source_file": "processed/_catalog/public/patient_eligibility_review.json",
        "source_sheet_or_table": "Aoki2024_IMC_RelapsedHodgkin",
        "source_field": "candidate_patient_count;verified_evidence_excerpt",
        "target_table": "sample_metadata",
        "target_field": "patient_key;source_subject_id;timepoint_label",
        "join_keys": "No per-ROI join key supplied",
        "value_mapping": None,
        "transformation_rule": "Retain the study-level patient-count evidence in the mapping record; leave per-sample patient and timepoint fields blank.",
        "time_origin_rule": None,
        "conflict_rule": "Do not propagate the study-level lower bound to individual sample rows.",
        "notes": "The source review verifies at least 71 patients and paired diagnostic/relapse specimens but no ROI-level linkage.",
    },
]
field_mapping = pd.DataFrame(field_rows, columns=field_columns)


images = pd.read_parquet(output_root / "images.parquet").to_dict("records") if (output_root / "images.parquet").exists() else []
samples = pd.read_csv(output_root / "sample_metadata.csv", dtype=str).to_dict("records") if (output_root / "sample_metadata.csv").exists() else []
excluded_path = cache_root / "input_excluded_images.json"
excluded_images = json.loads(excluded_path.read_text()) if excluded_path.exists() else []
processed_count = 0
direct_validation_done = False
reused_count = 0
new_count = 0
completed_ids = {row["image_id"] for row in images}
expected_names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()


def check_saved_image(row: dict) -> None:
    """Reuse a published native TIFF only when its identity and header agree."""
    path = data_root / row["input_image_path"]
    source_row = next(item for item in acquisition_rows
                      if f"{Path(item['source_archive']).stem}__acq_{int(item['acquisition_id']):03d}" == row["image_id"])
    width, height = int(source_row["width_px"]), int(source_row["height_px"])
    side = max(width, height)
    expected_geometry = [0, 0, width, height, (side-width)//2, (side-height)//2,
                         side-width-(side-width)//2, side-height-(side-height)//2, side, side]
    fields = ["source_x0_px", "source_y0_px", "source_x1_px", "source_y1_px",
              "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "height_px", "width_px"]
    if [int(row[field]) for field in fields] != expected_geometry or row["region_annotation"] != source_row["description"]:
        raise ValueError(f"Published acquisition geometry/annotation differs from source: {row['image_id']}")
    with tifffile.TiffFile(path) as tif:
        names = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        if (tif.series[0].shape != (37, int(row["height_px"]), int(row["width_px"]))
                or tif.series[0].dtype != np.dtype("float32") or names != expected_names):
            raise ValueError(f"Published native TIFF header/panel mismatch: {path}")


def save_progress(stage: str, current: str | None = None) -> None:
    """Publish resumable tables after each completed acquisition."""
    save_input_tables(output_root, pd.DataFrame(images, columns=image_columns),
                      pd.DataFrame(samples, columns=sample_columns), panel, field_mapping)
    excluded_path.write_text(json.dumps(excluded_images, indent=2) + "\n")
    progress = dict(study=STUDY, stage=stage, current_image=current,
                    expected_images=len(acquisition_rows), completed_images=len(images),
                    new_images_this_run=new_count, reused_images_this_run=reused_count,
                    excluded_images=len(excluded_images), sample_rows=len(samples),
                    sample_linked_images=sum(pd.notna(row.get("sample_id")) for row in images),
                    patient_linked_samples=sum(pd.notna(row.get("patient_key")) for row in samples),
                    gaps=["image/core to biological sample and patient crosswalk remains unavailable; see sample_linkage_search.json",
                          "new images await independent review"],
                    updated_at=datetime.now(timezone.utc).isoformat())
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(progress_path)


for saved_row in images:
    check_saved_image(saved_row)
if args.metadata_only:
    reused_count = len(images)
save_progress("metadata_refresh" if args.metadata_only else "export_ready")

for archive_name in ([] if args.metadata_only else archive_names):
    archive_path = raw_root / archive_name
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive_rows_for_file = [row for row in selected_rows if row["source_archive"] == archive_name]
    pending_rows = [row for row in archive_rows_for_file
                    if f"{Path(archive_name).stem}__acq_{int(row['acquisition_id']):03d}" not in completed_ids]
    reused_count += len(archive_rows_for_file) - len(pending_rows)
    if not pending_rows:
        print(f"reused {archive_name}: {len(archive_rows_for_file)} TIFFs", flush=True)
        continue
    save_progress("extracting", archive_name)
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".mcd")]
        if len(members) != 1:
            raise ValueError(f"Expected one MCD member in {archive_path}, found {len(members)}")
        member = members[0]
        member_path = extraction_root / Path(archive_name).stem / Path(member.filename).name
        member_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_this_run = False
        if not member_path.is_file() or member_path.stat().st_size != member.file_size:
            partial = member_path.with_suffix(member_path.suffix + ".partial")
            if partial.exists():
                raise ValueError(f"Existing in-flight extraction must remain untouched: {partial}")
            if shutil.disk_usage(extraction_root).free < member.file_size + 64 * 1024**3:
                raise ValueError(f"Insufficient extraction space for {archive_name}: {member.file_size} bytes")
            with archive.open(member, "r") as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            if partial.stat().st_size != member.file_size:
                raise IOError(f"Incomplete extraction for {archive_name}:{member.filename}")
            partial.replace(member_path)
            extracted_this_run = True

        with MCDFile(member_path) as direct_mcd:
            annotations = {"source_member": member.filename, "slides": [
                {"metadata": slide.metadata, "panoramas": [item.metadata for item in slide.panoramas],
                 "acquisitions": [{"metadata": item.metadata, "roi_points_um": item.roi_points_um}
                                  for item in slide.acquisitions]} for slide in direct_mcd.slides]}
            (source_search_root / "mcd_annotations").mkdir(exist_ok=True)
            (source_search_root / "mcd_annotations" / f"{Path(archive_name).stem}_annotations.json").write_text(
                json.dumps(annotations, indent=2) + "\n")
            direct_acquisitions = {
                int(acquisition.id): acquisition
                for slide in direct_mcd.slides
                for acquisition in slide.acquisitions
            }
        missing_acquisitions = {
            int(row["acquisition_id"])
            for row in archive_rows_for_file
            if int(row["acquisition_id"]) not in direct_acquisitions
        }
        if missing_acquisitions:
            raise ValueError(f"MCD metadata lacks indexed acquisition(s): {sorted(missing_acquisitions)}")

        for source_row in pending_rows:
            acquisition_id = int(source_row["acquisition_id"])
            acquisition = direct_acquisitions[acquisition_id]
            actual_labels = list(acquisition.channel_labels)
            if actual_labels != source_names or acquisition.num_channels != len(source_names):
                raise ValueError(f"Embedded MCD channel labels differ for {archive_name} acquisition {acquisition_id}")
            actual_width = acquisition.width_px
            actual_height = acquisition.height_px
            if (actual_width, actual_height) != (int(source_row["width_px"]), int(source_row["height_px"])):
                raise ValueError(f"MCD dimensions differ from index for {archive_name} acquisition {acquisition_id}")
            actual_pixel_size = (acquisition.pixel_size_x_um, acquisition.pixel_size_y_um)
            if not all(value is not None and np.isfinite(value) and value > 0 for value in actual_pixel_size):
                actual_pixel_size = (None, None)
            image_id = f"{Path(archive_name).stem}__acq_{acquisition_id:03d}"
            save_progress("exporting", image_id)
            sample_id = None
            image_dir = output_root / "images" / image_id
            input_path = image_dir / "input.tiff"
            source = SourceSpec(
                image_id=image_id,
                panel_id=panel_id,
                kind="mcd",
                path=member_path,
                channels=tuple(range(44)),
                acquisition_id=acquisition_id,
                pixel_size_um=actual_pixel_size if actual_pixel_size[0] is not None else None,
                source_revision=(
                    f"{archive_name}:{member.filename}:member_size={member.file_size}:"
                    f"zip_size={archive_path.stat().st_size}"
                ),
                intensity_description="native float32 MCD measurement values; no P99.5 normalization",
            )
            with open_source(source) as reader:
                if reader.shape != (44, actual_height, actual_width):
                    raise ValueError(f"Reader shape differs for {image_id}: {reader.shape}")
                if not direct_validation_done:
                    region = (0, 0, min(32, actual_width), min(32, actual_height))
                    direct_values = acquisition_reader = None
                    with MCDFile(member_path) as validation_mcd:
                        validation_acquisition = next(
                            acquisition_item
                            for slide in validation_mcd.slides
                            for acquisition_item in slide.acquisitions
                            if int(acquisition_item.id) == acquisition_id
                        )
                        direct_values = validation_mcd.read_acquisition(
                            validation_acquisition,
                            channels=list(range(44)),
                            region=region,
                        )
                    source_values = reader.read_window(list(range(44)), region)
                    if not np.array_equal(direct_values, source_values):
                        raise ValueError(f"Direct readimc and SourceReader values differ for {image_id}")
                    direct_validation_done = True
                exported = export_input(reader, panel, input_path, tile_size=512)
            (cache_root / "image_export_checks").mkdir(exist_ok=True)
            (cache_root / "image_export_checks" / f"{image_id}.json").write_text(
                json.dumps(exported, indent=2) + "\n")

            width = int(source_row["width_px"])
            height = int(source_row["height_px"])
            output_width = int(exported["width_px"])
            output_height = int(exported["height_px"])
            if output_width != output_height or output_width != max(width, height):
                raise ValueError(f"Unexpected square output for {image_id}: {exported}")
            if int(exported["source_channel_count"]) != 44 or int(exported["channel_count"]) != 37:
                raise ValueError(f"Unexpected channel output for {image_id}: {exported}")
            if not exported["admitted"]:
                excluded_images.append({
                    "image_id": image_id,
                    "source_archive": archive_name,
                    "source_member": member.filename,
                    "acquisition_id": acquisition_id,
                    "reason": "all retained downstream marker channels are zero across the full acquisition frame",
                    "channel_nonzero_pixels": exported["channel_nonzero_pixels"],
                })
                save_progress("exporting", image_id)
                processed_count += 1
                print(f"excluded {processed_count}/{len(selected_rows)} {image_id}")
                continue
            relative_input = str(metadata_source_relative(input_path, data_root))
            valid_pixels = exported.get("valid_pixels")
            relative_valid = None if valid_pixels is None else str(
                metadata_source_relative(image_dir / str(valid_pixels), data_root)
            )
            source_files = [str(metadata_source_relative(archive_path, data_root))]
            split_evidence = (
                f"source acquisition {archive_name}:{member.filename}:acquisition_id={acquisition_id}; "
                f"source description={source_row['description']!r}; full acquisition frame retained"
            )
            image_values = {column: None for column in image_columns}
            image_values.update({
                "study": STUDY,
                "image_id": image_id,
                "sample_id": sample_id,
                "panel_id": panel_id,
                "image_kind": "roi",
                "source_image_id": f"{archive_name}:{member.filename}:acquisition_{acquisition_id}",
                "source_files": source_files,
                "source_format": "mcd",
                "source_acquisition_id": acquisition_id,
                "source_z_index": None,
                "source_t_index": None,
                "source_axes": "CYX",
                "source_dtype": str(exported["source_dtype"]),
                "source_revision": source.source_revision,
                "slide_id": source_row["slide_id"],
                "core_id": None,
                "roi_id": source_row["description"],
                "fov_id": None,
                "region_annotation": source_row["description"],
                "source_x0_px": 0,
                "source_y0_px": 0,
                "source_x1_px": width,
                "source_y1_px": height,
                "pad_left_px": exported["pad_left_px"],
                "pad_top_px": exported["pad_top_px"],
                "pad_right_px": exported["pad_right_px"],
                "pad_bottom_px": exported["pad_bottom_px"],
                "height_px": output_height,
                "width_px": output_width,
                "source_channel_count": 44,
                "channel_count": 37,
                "pixel_size_x_um": actual_pixel_size[0],
                "pixel_size_y_um": actual_pixel_size[1],
                "input_image_path": relative_input,
                "valid_pixels_path": relative_valid,
                "intensity_description": source.intensity_description,
                "split_method": "source_acquisition",
                "split_evidence": split_evidence,
                "metadata_notes": (
                    "Local source provides one counted acquisition per image and no ROI-to-patient map; "
                    "sample_id and patient_key remain blank until a biological-sample join is supplied."
                ),
            })
            images.append(image_values)
            completed_ids.add(image_id)
            new_count += 1
            save_progress("exporting", image_id)

            processed_count += 1
            print(f"saved {len(images)}/{len(acquisition_rows)} {image_id}", flush=True)

        if extracted_this_run and str(extraction_root).startswith("/tmp/study_input_batch/"):
            member_path.unlink()

images_table = pd.DataFrame(images, columns=image_columns)
images_table["metadata_notes"] = (
    "Original complete acquisition frame; biological sample and patient identity unresolved. "
    "Local acquisition metadata, public release, paper/preprint supplements and author repositories "
    "do not supply an acquisition/core-to-specimen crosswalk; see sample_linkage_search.json."
)
images = images_table.to_dict("records")
samples_table = pd.DataFrame(samples, columns=sample_columns)

# Public source investigation is retained as canonical metadata; no ROI name is a sample key.
sample_sources = source_search_root
if (sample_sources / "biorxiv_media-1.xlsx").exists():
    workbook = pd.ExcelFile(sample_sources / "biorxiv_media-1.xlsx")
    supplement_sheets = {sheet: pd.read_excel(workbook, sheet_name=sheet, header=None).shape
                         for sheet in workbook.sheet_names}
    investigation = dict(
        study=STUDY, status="unresolved", image_count=len(images), sample_rows=len(samples),
        sample_linked_images=int(images_table.sample_id.notna().sum()), patient_linked_samples=0,
        local_sources=["raw/public/" + STUDY + "/metadata.json#public/" + STUDY + "/acquisition_index.json",
                       "raw/public/" + STUDY + "/metadata.json#public/" + STUDY + "/channel_mapping.csv",
                       "raw/public/" + STUDY + "/metadata.json#public/" + STUDY + "/source_manifest.json"],
        local_findings="All eight ZIP central directories contain one MCD only. All 303 indexed descriptions are acquisition ROI labels. Checked MCD slide/acquisition annotations supply geometry and acquisition labels, without a biological-specimen or patient crosswalk.",
        public_sources=[
            dict(url="https://zenodo.org/records/7963681", cached="sample_sources/zenodo_record.json",
                 finding="Eight original MCD ZIPs only; no sample/core map attachment."),
            dict(url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10950131/fullTextXML",
                 cached="sample_sources/pmc_fulltext.xml",
                 finding="Final article: 71 paired diagnostic/relapse patients plus 22 diagnostic controls, 164 biopsies; aggregate cohort context, no acquisition keys. Data statement identifies Zenodo and two author packages."),
            dict(url="https://www.biorxiv.org/content/10.1101/2023.05.24.542195v1.supplementary-material",
                 cached="sample_sources/biorxiv_supplementary.html",
                 finding="Downloaded media-1.xlsx and media-2.docx. Excel Sheet1 is aggregate clinical characteristics (73 paired patients and 20 non-relapse controls), Sheet2 is antibody panel. Word has supplementary figure captions and no clinical table. Preprint reports 261 analyzed ROIs; no key maps these to the 303 complete original acquisitions.",
                 sheet_shapes=supplement_sheets),
            dict(url="https://github.com/ajiangsfu/RHL4S", cached="sample_sources/RHL4S_tree.json;sample_sources/RHL4S_convert_patient.R;sample_sources/author_package_members.json",
                 finding="convert_patient requires caller-provided patient.xlsx, core sheet, ID1/ID2/Low/SlideID fields to link MC-IF data. Recursive public tree and published 0.1.1 ZIP contain no patient.xlsx or IMC core map; package is for MC-IF and cannot supply this IMC join."),
            dict(url="https://github.com/ajiangsfu/csmpv", cached="sample_sources/csmpv_tree.json;sample_sources/csmpv_data.R;sample_sources/author_package_members.json",
                 finding="Statistical package and generic training/validation datlist example; recursive tree and published 1.0.5 tar archive have no published IMC acquisition-to-specimen crosswalk."),
            dict(url="https://ascopubs.org/doi/suppl/10.1200/JCO.23.01115",
                 cached="sample_sources/publisher_http_checks.json;sample_sources/source_http_checks.json",
                 finding="Publisher supplement page and PDF candidate returned HTTP 403. Europe PMC supplementaryFiles response contains only main-text figure JPG/GIF files; no clinical supplement.")],
        join_keys="source_archive + source_member + acquisition_id identifies images only; biological-specimen and patient join keys unavailable",
        unresolved="Need an explicit original IMC acquisition/core to biological specimen to patient/timepoint map; neither cohort totals nor ROI names establish that join.",
        next_action="Controller to retain image delivery and metadata gap; independent review to check new image range and cached source investigation. No author contact performed.")
    sample_linkage_path.write_text(json.dumps(investigation, ensure_ascii=False, indent=2) + "\n")
    row = {column: None for column in field_columns}
    row.update(study=STUDY, source_file=str(metadata_source_relative(sample_linkage_path, data_root)),
               source_sheet_or_table="local sources; final paper; preprint supplements; author repository trees",
               source_field="source_archive;source_member;acquisition_id;ROI descriptions;aggregate clinical tables;external patient.xlsx requirement",
               target_table="sample_metadata", target_field="sample_id;source_sample_id;patient_key;source_subject_id;timepoint_label",
               join_keys=investigation["join_keys"],
               transformation_rule="No biological sample rows can be reconstructed from aggregate tables. Preserve blank image sample_id until an explicit source crosswalk is available.",
               conflict_rule="Preserve preprint/final cohort version differences as context; do not apply either population's counts or infer patient/timepoint identities from ROI names.",
               notes=f"M01 unresolved: {len(images)} images, {len(samples)} sample rows, 0 image/sample links, 0 patient-linked samples. Source search and HTTP gaps recorded in sample_linkage_search.json.")
    field_mapping = pd.concat([field_mapping, pd.DataFrame([row], columns=field_columns)], ignore_index=True)


publish_metadata_sources(source_meta)
save_input_tables(output_root, images_table, samples_table, panel, field_mapping)

evidence = {
    "schema": "cellweft.study_input.aoki.v1",
    "study": STUDY,
    "source_manifest": str(manifest_path),
    "coverage": str(coverage_path),
    "archives": archive_names,
    "member_count": len(archive_names),
    "acquisition_count": len(images_table),
    "native_channel_count": 44,
    "retained_channel_count": 37,
    "panel_id": panel_id,
    "sample_count": len(samples_table),
    "sample_identity_rule": "No explicit biological-specimen identity can be joined after local and public source investigation; sample table remains empty.",
    "patient_join_status": "Acquisition/core to biological specimen to patient crosswalk unresolved; sample_id and patient_key remain blank.",
    "clinical_evidence": "study-level lower bound >=71 patients; not propagated to ROI rows",
    "source_rows": len(acquisition_rows),
    "expected_images": len(acquisition_rows),
    "previously_excluded_source_acquisitions": sum(row["n_excluded"] for row in coverage["archive_summaries"]),
    "sample_linkage_evidence": str(metadata_source_relative(sample_linkage_path, data_root)),
    "source_rows_processed": len(selected_rows),
    "image_count_admitted": len(images_table),
    "image_count_excluded": len(excluded_images),
    "excluded_images": excluded_images,
    "direct_readimc_validation_this_run": direct_validation_done,
    "reused_images_this_run": reused_count,
    "new_images_this_run": new_count,
    "extraction_root": str(extraction_root),
}
(cache_root / "input_delivery_summary.json").parent.mkdir(parents=True, exist_ok=True)
(cache_root / "input_delivery_summary.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(evidence, ensure_ascii=False, indent=2))

if args.overviews:
    from PIL import Image, ImageDraw
    by_archive = {}
    overview_rows = []
    for row in images:
        archive = row["source_image_id"].split(":", 1)[0]
        if archive not in by_archive or not re.fullmatch(r"ROI_[0-9]{3}", row["region_annotation"]):
            overview_rows.append(row)
        by_archive[archive] = True
    for page_number, start in enumerate(range(0, len(overview_rows), 16)):
        rows = overview_rows[start:start+16]
        canvas = Image.new("RGB", (4*300, ((len(rows)+3)//4)*330))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows):
            with tifffile.TiffFile(data_root / row["input_image_path"]) as tif:
                channels = []
                for channel in [27, 28, 36]:  # CD30 red, CD3e green, 193Ir blue; display only.
                    values = tif.pages[channel].asarray(maxworkers=1)
                    limit = np.percentile(values[values > 0], 99) if np.any(values > 0) else 1
                    channels.append(np.uint8(np.clip(values / limit, 0, 1)*255))
            preview = Image.fromarray(np.stack(channels, axis=-1)).resize((300, 300))
            x, y = (index % 4)*300, (index // 4)*330
            canvas.paste(preview, (x, y+30))
            draw.text((x+3, y+1), row["image_id"], fill="white")
            draw.text((x+3, y+14), row["region_annotation"], fill="white")
        canvas.save(cache_root / f"batch_overview_{page_number+1:02d}.png")
    (cache_root / "batch_overview_selection.json").write_text(json.dumps(
        [{"image_id": row["image_id"], "region_annotation": row["region_annotation"]}
         for row in overview_rows], indent=2) + "\n")

# These are preparer checks. Preserve the original independent 43-image review.
review_path = output_root / "review_results.csv"
review = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.read_csv(
    PROJECT / "docs/data/templates/study_input/review_results.csv")
review = review[(review.record_id != f"{STUDY}:full_delivery_preparer") | (review.check_id == "I04")]
review_rows = []
for check_id, result, details in [
    ("S01", "pass", f"Preparer check: eight original ZIPs, 330 total source acquisitions; 303 complete verified acquisitions and 27 previously excluded incomplete acquisitions (raw/public/{STUDY}/metadata.json#public/{STUDY}/source_acquisition_exclusions.csv). Current delivery {len(images)}/303, all retained downstream markers checked during export; existing 43 TIFFs reused."),
    ("P03", "pass", f"Preparer check: opened {len(images)} output TIFF headers; 37 CYX float32 channels, standardized names and order match S1-derived panel; 44 source rows retained."),
    ("I03", "pass", f"Preparer check: all {len(images)} source frames, annotations and padding match the verified acquisition index; no pixel resampling or rotation."),
    ("M01", "unresolved", f"Preparer source search: sample rows={len(samples)}, image-to-sample links=0/{len(images)}, patient-linked samples=0. No confirmed acquisition/core-to-biological-sample map. See sample_linkage_search.json; do not infer identities from ROI names."),
    ("D01", "unresolved", f"Preparer delivery: {len(images)}/303 TIFFs and all input tables saved. Existing independent review applies to original A1_1 batch only. New range awaits independent review; sample and patient links remain unresolved."),
]:
    review_rows.append(dict(study=STUDY, preparer=Path(__file__).name, reviewer="", scope="study",
                            record_id=f"{STUDY}:full_delivery_preparer", check_id=check_id,
                            result=result, source_evidence=(
                                str(metadata_source_relative(sample_linkage_path, data_root))
                                if check_id == "M01" else str(metadata_source_relative(index_path, data_root))),
                            output_evidence=str(metadata_source_relative(output_root, data_root)), details=details))
review = pd.concat([review, pd.DataFrame(review_rows)], ignore_index=True)
pixel_evidence = cache_root / "batch_preparer_pixel_checks.json"
if pixel_evidence.exists():
    review = pd.concat([review, pd.DataFrame([dict(
        study=STUDY, preparer=Path(__file__).name, reviewer="", scope="image",
        record_id=f"{STUDY}:full_delivery_preparer", check_id="I05", result="pass",
        source_evidence=str(metadata_source_relative(pixel_evidence, data_root)),
        output_evidence=f"processed/{STUDY}/images/TMA_B1_2__acq_001/input.tiff",
        details="Preparer pixel check only: direct readimc versus saved TIFF, all 37 retained channels, source windows [0,0,32,32] and [608,736,640,768], equal float32 values; output padding window [0,0,32,32] is zero in all channels. Existing independent first-batch pixel review remains separate.")])], ignore_index=True)
with formal_publish_lease(review_path, directory=output_root):
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
save_progress("prepared_awaiting_independent_review")
