"""Deliver verified Krop 2022 decidua MCD acquisitions as native inputs.

Inputs: the original Mendeley ZIP, selected-subset/native proofs, and S1.
Outputs: independent input TIFFs and the per-study image, sample, panel,
field-mapping, and pending-review tables. Each MCD member is closed in turn.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import tifffile
from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Krop2022_IMC_HealthyDecidua"
DATA_ROOT = Path(os.environ.get("CELLWEFT_DATA_ROOT", "/mnt/NAS_21T/ProjectData/CellWeft"))
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(DATA_ROOT, STUDY)
OUTPUT = DATA_ROOT / "processed" / STUDY
ARCHIVE = DATA_ROOT / "raw/public" / STUDY / "public_original/raw IMC data files.zip"
S1 = DATA_ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PAPER = META / "krop2022_decidua.bioc.xml"
TEMPLATES = PROJECT / "docs/data/templates/study_input"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--progress-path", type=Path, required=True)
parser.add_argument("--metadata-only", action="store_true",
                    help="Repair metadata for the complete, previously verified indexed image set.")
args = parser.parse_args()

selected_path = META / "selected_subset_closeout_20261024.json"
native_path = META / "native_verification.json"
with formal_read_lease(META):
    selected = json.loads(selected_path.read_text())
    native = json.loads(native_path.read_text())
chosen = selected["selected_images"]
proofs = {row["physical_image_id"]: row["native_proof"] for row in native["records"]}
assert selected["qualified_usable_subset"] and len(chosen) == selected["n_images"] == 87
assert all(row["acquisition_id"] in proofs and proofs[row["acquisition_id"]]["complete"] for row in chosen)

with formal_read_lease(PAPER):
    paper_root = ET.parse(PAPER).getroot()
passages = (node.text or "" for node in paper_root.iter("text"))
table_text = next(text for text in passages if text.startswith("\tSampleID\tTrimester\t"))
table_cells = [cell.strip() for cell in table_text.split("\t") if cell.strip()]
assert table_cells[:5] == ["SampleID", "Trimester", "Gestational age (weeks\xa0+ days)",
                           "Delivery mode", "Maternal age"]
table_rows = table_cells[10:]
assert len(table_rows) == 13 * 11
clinical = {}
for offset in range(0, len(table_rows), 11):
    ordinal, specimen, trimester, gestation, delivery, maternal_age, *_ = table_rows[offset:offset + 11]
    assert int(ordinal) == len(clinical) + 1 and specimen not in clinical
    assert trimester in {"First", "Second", "Term"}
    clinical[specimen] = dict(timepoint_label="other", trimester_reported=trimester,
                              gestational_age_reported=gestation,
                              delivery_mode_reported=delivery,
                              age_at_collection_years=(maternal_age if maternal_age != "unknown" else None))
assert {Path(row["source_member"]).stem for row in chosen} <= clinical.keys()

with formal_read_lease(ARCHIVE):
    with zipfile.ZipFile(ARCHIVE) as z:
        members = {item.filename: item for item in z.infolist()}
        member_order = list(dict.fromkeys(row["source_member"] for row in chosen))
        assert len(member_order) == 12
        assert all(members[name].file_size == proofs[next(row["acquisition_id"] for row in chosen
                                                            if row["source_member"] == name)]["archive_entry"]["exact_archive_entry_size"]
                   for name in member_order)

source_names = [item["source_label"] for item in chosen[0]["source_channel"]][3:]
assert all([item["source_label"] for item in row["source_channel"]][3:] == source_names for row in chosen)
assert len(source_names) == 46 and source_names[2:4] == ["CD33", "CD33"]
panel_id = f"{STUDY}__native46"
# Native fields X/Y/Z precede the 46 intensity channels. The first tissue
# acquisition has slightly more CD33-positive pixels in Cd(114) than Cd(112).
with formal_read_lease(S1):
    panel = panel_from_s1(STUDY, panel_id, source_names, S1,
                          drop_indices={2: "Duplicate CD33 Cd(112); Cd(114) selected by native signal QC"})
assert panel.keep_channel.sum() == 45

def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / name, nrows=0, encoding="utf-8-sig").columns.tolist()


image_columns = columns("image_metadata.csv")
sample_columns = columns("sample_metadata.csv") + ["gestational_age_reported", "delivery_mode_reported",
                                                   "trimester_reported"]
field_columns = columns("field_mapping.csv")
with formal_read_lease(OUTPUT):
    images = (pd.read_parquet(OUTPUT / "images.parquet").to_dict("records")
              if (OUTPUT / "images.parquet").exists() else [])
    samples = (pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str).to_dict("records")
               if (OUTPUT / "sample_metadata.csv").exists() else [])
for sample in samples:
    sample.update(clinical[sample["source_sample_id"]])
    sample.update(tissue_type="normal", disease_status_at_collection=None)
done = {row["image_id"] for row in images}
sample_done = {row["sample_id"] for row in samples}
for specimen, source_values in clinical.items():
    sample_id = f"sample_{specimen}"
    if sample_id in sample_done:
        continue
    item = {key: None for key in sample_columns}
    item.update(study=STUDY, sample_id=sample_id, source_sample_id=specimen,
                sample_type="decidua basalis", tissue_type="normal",
                anatomical_site="maternal-fetal interface",
                disease_status_at_collection=None,
                metadata_notes="Article Table 1 sample with a matching source MCD member; no image is included in the current selected processing scope, and patient identity is unlinked.")
    item.update(source_values)
    samples.append(item)
    sample_done.add(sample_id)
new_count = 0
excluded = []

field_values = [
    dict(source_file=str(metadata_source_relative(selected_path, DATA_ROOT)), source_sheet_or_table="selected_images",
         source_field="source_member;acquisition_id;native_proof", target_table="image_metadata",
         target_field="source_files;source_image_id;source_acquisition_id;geometry;region_annotation",
         join_keys="ZIP member + slide 0 + acquisition ID",
         transformation_rule="One selected complete tissue acquisition per image; full native frame, centered square padding.",
         notes="87 selected acquisitions in 12 MCD members; held controls, overlap/identity uncertainties and defective 2394 (2) remain excluded."),
    dict(source_file=str(metadata_source_relative(native_path, DATA_ROOT)), source_sheet_or_table="ordered_channel_rows",
         source_field="channel_label;channel_name;order", target_table="panel_mapping",
         target_field="source_name;source_channel_index;keep_channel;output_channel_index",
         join_keys="member + acquisition + ordered channel index; Study + Original marker to S1",
         transformation_rule="Keep S1 admitted markers and nuclear inputs; CD33 Cd(114) retained, Cd(112) dropped after first native tissue image QC.",
         notes="First 800x800 tissue image: Cd(112) 9403 nonzero pixels, mean 0.01654; Cd(114) 9677 nonzero pixels, mean 0.01719."),
    dict(source_file=str(metadata_source_relative(ARCHIVE, DATA_ROOT)), source_sheet_or_table="MCD slide Filename; acquisition Description",
         source_field="slide Filename; MCD member basename; Description", target_table="sample_metadata",
         target_field="sample_id;source_sample_id;patient_key;timepoint_label",
         join_keys="ZIP member basename to native slide Filename and all selected acquisitions in that member",
         transformation_rule="MCD member stem identifies the specimen; article Table 1 SampleID joins clinical values; patient key left empty.",
         conflict_rule="Do not infer patient identity from specimen IDs or aggregate cohort counts; exclude control acquisitions.",
         notes="Mendeley v1 (https://data.mendeley.com/datasets/gs2bj33r6f/1) describes 97 ROIs from 13 samples and control ROIs; the archive MCD stems match all 13 Table 1 SampleIDs. The current selected processing subset covers 12 specimens and 87 acquisitions. Neither Table 1 nor the ZIP assigns a cross-specimen patient key."),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Table 1",
         source_field="SampleID;Trimester;Gestational age;Delivery mode;Maternal age",
         target_table="sample_metadata",
         target_field="sample_id;source_sample_id;timepoint_label;gestational_age_reported;delivery_mode_reported;age_at_collection_years",
         join_keys="MCD ZIP member stem = Table 1 SampleID; source_sample_id = Table 1 SampleID",
         transformation_rule="Set sample_id to sample_<SampleID>; preserve source_sample_id; keep gestational age and delivery mode as reported; use numeric maternal age only where stated; encode the custom gestational timepoint as other and preserve Table 1 Trimester in trimester_reported.",
         notes="Article DOI 10.1016/j.isci.2022.104648; all 13 Table 1 specimens match MCD member stems. The processed image selection includes 12 specimens; the remaining public sample row is retained without image links. timepoint_label=other denotes the source trimester recorded in trimester_reported."),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Table 1",
         source_field="Trimester", target_table="sample_metadata",
         target_field="trimester_reported", join_keys="Table 1 SampleID = source_sample_id",
         transformation_rule="Preserve the source category verbatim: First, Second, or Term.",
         notes=json.dumps({"data_type": "string", "definition": "Gestational trimester reported in article Table 1; defines the custom timepoint_label=other."})),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Summary; STAR Methods",
         source_field="decidua basalis tissue sections", target_table="sample_metadata",
         target_field="sample_type", join_keys="Study cohort; Table 1 SampleID = source_sample_id",
         transformation_rule="Record decidua basalis as the source tissue specimen type.",
         notes="Applies to all 13 Table 1 specimen records; no patient-level inference."),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Summary; STAR Methods",
         source_field="decidua basalis of healthy pregnancy", target_table="sample_metadata",
         target_field="tissue_type", join_keys="Study cohort; Table 1 SampleID = source_sample_id",
         transformation_rule="Classify independent healthy decidual tissue as normal under the current tissue_type vocabulary.",
         value_mapping=json.dumps({"decidua basalis of healthy pregnancy": "normal"}),
         notes="Anatomical tissue identity remains in sample_type; normal denotes the non-tumor tissue group."),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Summary; Introduction",
         source_field="human maternal-fetal interface; decidua basalis", target_table="sample_metadata",
         target_field="anatomical_site", join_keys="Study cohort; Table 1 SampleID = source_sample_id",
         transformation_rule="Use maternal-fetal interface as the reported anatomical site of the decidua basalis specimens.",
         notes="Cohort-level anatomical description; no ROI-specific anatomical subtype inferred."),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Summary; Table 1",
         source_field="healthy pregnancy; no disease-course status field", target_table="sample_metadata",
         target_field="disease_status_at_collection", join_keys="Study cohort; Table 1 SampleID = source_sample_id",
         transformation_rule="Leave blank: healthy pregnancy is cohort context and does not encode a disease-course status in the current disease_status vocabulary.",
         notes="All 13 sample rows remain empty; do not infer remission, stable disease, or no_evidence_of_disease. Healthy-pregnancy context is retained in tissue_type mapping and the cited source; this optional field is unavailable."),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Table 1",
         source_field="Gestational age (weeks + days)", target_table="sample_metadata",
         target_field="gestational_age_reported", join_keys="Table 1 SampleID = source_sample_id",
         transformation_rule="Preserve the source string, including day component where present.",
         notes=json.dumps({"data_type": "string", "definition": "Gestational age in weeks and optional days as reported in article Table 1."})),
    dict(source_file=str(metadata_source_relative(PAPER, DATA_ROOT)), source_sheet_or_table="Table 1",
         source_field="Delivery mode", target_table="sample_metadata",
         target_field="delivery_mode_reported", join_keys="Table 1 SampleID = source_sample_id",
         transformation_rule="Preserve the article category verbatim.",
         notes=json.dumps({"data_type": "string", "definition": "Delivery or termination mode reported in article Table 1."})),
]
fields = pd.DataFrame([{**{key: None for key in field_columns}, "study": STUDY, **row}
                       for row in field_values], columns=field_columns)

def image_id_for(row: dict) -> str:
    return f"{Path(row['source_member']).stem}__acq_{int(proofs[row['acquisition_id']]['source_acquisition_id']):03d}"


def save_progress(stage: str, current: str | None = None) -> None:
    if images:
        save_input_tables(OUTPUT, pd.DataFrame(images, columns=image_columns),
                          pd.DataFrame(samples, columns=sample_columns), panel, fields)
    progress = dict(study=STUDY, stage=stage, current_image=current,
                    expected_images=len(chosen), exported_images=len(images),
                    new_images_this_run=new_count, excluded_images=excluded,
                    sample_rows=len(samples),
                    sample_linked_images=sum(pd.notna(row.get("sample_id")) for row in images),
                    patient_linked_samples=sum(pd.notna(row.get("patient_key")) for row in samples),
                    next_image=next((image_id_for(row) for row in chosen if image_id_for(row) not in done), None),
                    updated_at=datetime.now(timezone.utc).isoformat())
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.progress_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.progress_path)


if args.metadata_only:
    assert done == {image_id_for(row) for row in chosen}
else:
    with formal_read_lease(OUTPUT):
        for row in images:
            path = DATA_ROOT / row["input_image_path"]
            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) != 45 or str(tif.pages[0].dtype) != "float32":
                    raise ValueError(f"Previously indexed input differs: {path}")

by_member = defaultdict(list)
for row in chosen:
    by_member[row["source_member"]].append(row)
save_progress("ready")

for member in member_order:
    pending = [row for row in by_member[member] if image_id_for(row) not in done]
    if not pending:
        continue
    save_progress("extracting", member)
    with tempfile.TemporaryDirectory(prefix="cellweft-krop-") as directory:
        source_path = Path(directory) / Path(member).name
        with formal_read_lease(ARCHIVE):
            with zipfile.ZipFile(ARCHIVE) as z, z.open(member) as source, source_path.open("wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        with MCDFile(source_path) as mcd:
            acquisitions = {int(a.id): a for slide in mcd.slides for a in slide.acquisitions}
            slide_filename = mcd.slides[0].metadata["Filename"].split("\\")[-1]
            specimen = Path(member).stem
            sample_id = f"sample_{specimen}"
            if sample_id not in sample_done:
                item = {key: None for key in sample_columns}
                item.update(study=STUDY, sample_id=sample_id, source_sample_id=specimen,
                            sample_type="decidua basalis", tissue_type="normal",
                            anatomical_site="maternal-fetal interface",
                            disease_status_at_collection=None,
                            metadata_notes=f"MCD member {member}; native slide Filename {slide_filename}; patient identity unlinked.")
                item.update(clinical[specimen])
                samples.append(item)
                sample_done.add(sample_id)
            for row in pending:
                proof = proofs[row["acquisition_id"]]
                acquisition_id = int(proof["source_acquisition_id"])
                acquisition = acquisitions[acquisition_id]
                meta = acquisition.metadata
                width, height = int(meta["MaxX"]), int(meta["MaxY"])
                assert (width, height) == (proof["data_boundary"]["width"], proof["data_boundary"]["height"])
                image_id = image_id_for(row)
                save_progress("exporting", image_id)
                revision = (f"{ARCHIVE.stat().st_size}:{members[member].file_size}:"
                            f"{members[member].CRC}:{member}")
                source = SourceSpec(image_id=image_id, panel_id=panel_id, kind="mcd",
                                    path=source_path, channels=tuple(range(46)),
                                    acquisition_id=acquisition_id, pixel_size_um=(1.0, 1.0),
                                    source_revision=revision,
                                    intensity_description="native float32 MCD values; unnormalized")
                input_path = OUTPUT / "images" / image_id / "input.tiff"
                with open_source(source) as reader:
                    result = export_input(reader, panel, input_path, tile_size=512)
                if not result["admitted"]:
                    excluded.append(dict(source_member=member, acquisition_id=acquisition_id,
                                         reason="No admitted marker signal across source frame"))
                    save_progress("exporting", image_id)
                    continue
                side = max(width, height)
                item = {key: None for key in image_columns}
                item.update(study=STUDY, image_id=image_id, sample_id=sample_id, panel_id=panel_id,
                            image_kind="roi", source_image_id=row["acquisition_id"],
                            source_files=[str(metadata_source_relative(ARCHIVE, DATA_ROOT))], source_format="mcd",
                            source_acquisition_id=acquisition_id, source_axes="CYX",
                            source_dtype="float32", source_revision=revision,
                            slide_id=str(proof["slide_id"]), roi_id=meta["Description"],
                            region_annotation=meta["Description"],
                            source_x0_px=0, source_y0_px=0,
                            source_x1_px=width, source_y1_px=height,
                            pad_left_px=(side-width)//2, pad_right_px=side-width-(side-width)//2,
                            pad_top_px=(side-height)//2, pad_bottom_px=side-height-(side-height)//2,
                            height_px=side, width_px=side, source_channel_count=46,
                            channel_count=int(result["channel_count"]),
                            pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                            input_image_path=str(metadata_source_relative(input_path, DATA_ROOT)),
                            valid_pixels_path=(str(metadata_source_relative(input_path.parent / 'valid_pixels.tiff', DATA_ROOT))
                                               if result["valid_pixels"] else None),
                            intensity_description=source.intensity_description,
                            split_method="source_acquisition",
                            split_evidence=f"{row['acquisition_id']}; native proof {row['native_proof']}",
                            metadata_notes="Specimen linked by MCD member and slide label; patient crosswalk unavailable.")
                images.append(item)
                done.add(image_id)
                new_count += 1
                save_progress("exporting", image_id)
                print(f"exported {len(images)}/{len(chosen)} {image_id}", flush=True)

# The independent reviewer owns all result rows; this header marks pending review.
review = OUTPUT / "review_results.csv"
if not review.exists():
    payload = (TEMPLATES / "review_results.csv").read_bytes()
    with formal_publish_lease(review, directory=OUTPUT):
        if not review.exists():
            temporary = review.with_suffix(".csv.partial")
            temporary.write_bytes(payload)
            os.replace(temporary, review)
save_progress("prepared_awaiting_independent_review")
