"""Prepare Plattner CRC native IMC ROIs from the complete public imaging archive.

Inputs: imaging.zip, its native verification, source sample_info.tsv, published
article evidence, and the study's S1 marker mapping. Outputs: native CYX
input.tiff per ROI and the panel/image/sample/field/preparer-review tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import Window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Plattner2023_IMC_CRC"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path, default=Path(
    PROJECT / "cache/study_input_jobs" / STUDY / "0343_prepare_0/progress.json"))
parser.add_argument("--repair-image", help="Republish one previously indexed image and retain reviewed tables")
args = parser.parse_args()
root = args.data_root
metadata = metadata_source_directory(root, STUDY)
archive = root / "raw/public" / STUDY / "public_original/imaging.zip"
cache = PROJECT / "cache/study_input" / STUDY
output = root / "processed" / STUDY
local = Path("/tmp/study_input_batch") / STUDY
for directory in (cache, local, args.progress.parent):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(PROJECT / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, root))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def save_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def progress(stage: str, completed: int, **details: object) -> None:
    save_json(args.progress, dict(study=STUDY, stage=stage, completed_images=completed,
                                  expected_images=37, updated_utc=datetime.now(timezone.utc).isoformat(),
                                  **details))


previous_progress = json.loads(args.progress.read_text()) if args.progress.exists() else {}
progress("opening_sources", previous_progress.get("completed_images", 0),
         next_member=previous_progress.get("next_member"))


class NativeReader:
    """Read one locally staged, uncompressed OME member in its source CYX order."""

    def __init__(self, pages: list[np.memmap], path: Path):
        self.pages = pages
        self.shape = (len(pages), *pages[0].shape)
        self.dtype = pages[0].dtype
        self.source = SimpleNamespace(path=path, pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.stack([self.pages[channel][y0:y1, x0:x1] for channel in channels])

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


# Reconcile the public archive, native proof, S1 and the source specimen map.
verification_path = metadata / "native_verification.json"
verification = json.loads(verification_path.read_text())
manifest = json.loads((metadata / "source_manifest.json").read_text())
records = verification["records"]
members = manifest["scope_selector"]["eligible_archive_entries"]
assert verification["native_result"]["actual_n_images"] == len(records) == len(members) == 37
assert verification["scope_complete"] and not verification["source_defects"]
assert archive.stat().st_size == manifest["source_bytes"]
proofs = {row["native_proof"]["source_member_name"]: row["native_proof"] for row in records}
assert set(proofs) == set(members)
with formal_read_lease(archive):
    with zipfile.ZipFile(archive) as source_zip:
        source_names = [info.filename for info in source_zip.infolist()]
        assert set(members).issubset(source_names)
        specimen_lines = source_zip.read("imaging/sample_info.tsv").decode().splitlines()
specimen_map = dict(line.split("\t") for line in specimen_lines)
assert len(specimen_map) == len(specimen_lines) == 6
channels = proofs[members[0]]["channels"]
assert all(proofs[member]["channels"] == channels and
           proofs[member]["common_shape"] == [1000, 1000] and
           proofs[member]["native_page_count"] == 50 and
           proofs[member]["complete"] for member in members)
panel_id = f"{STUDY}__original_50"
panel = panel_from_s1(STUDY, panel_id, channels,
                      root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv",
                      drop_indices={3: "Duplicate CD39 measurement; representative native ROI has stronger 112Cd signal."})
kept = panel[panel.keep_channel].sort_values("output_channel_index")

# Join the original specimen key to clinical records; preserve the primary and
# metastatic specimens as separate samples under the same source patient.
article_path = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10692669/"
clinical_path = metadata / "source_attachments/clinical_data_organoids.tsv"
with formal_read_lease(clinical_path):
    clinical = pd.read_csv(clinical_path, sep="\t", dtype=str).set_index("Organoids code", verify_integrity=True)
assert set(specimen_map.values()).issubset(set(clinical.index))
samples_by_specimen = {}
for specimen, subject in specimen_map.items():
    subject_root = subject.removesuffix("LM")
    source_clinical = clinical.loc[subject]
    assert source_clinical["Patient"] == subject_root
    assert source_clinical["Origin"] == ("liver metastasis" if subject.endswith("LM") else "primary tumor")
    sample_id = f"{STUDY}__{specimen}"
    sample = dict.fromkeys(columns("sample_metadata"))
    sample.update(study=STUDY, sample_id=sample_id, patient_key=f"{STUDY}__{source_clinical['Patient']}",
                  source_subject_id=source_clinical["Patient"], source_sample_id=specimen,
                  sample_type="tissue", tissue_type="tumor", primary_diagnosis="colorectal carcinoma",
                  anatomical_site=source_clinical["Location"],
                  lesion_type="metastatic" if subject.endswith("LM") else "primary",
                  primary_site="colorectum", preservation_method="ffpe",
                  processing_method="IMC tissue section",
                  sex_reported={"M": "male", "F": "female"}[source_clinical["Sex"]],
                  treatment_status_at_collection=f"Preoperative chemotherapy reported: {source_clinical['preop CTx']}",
                  reported_age_years=int(source_clinical["Age"]),
                  reported_uicc_stage=source_clinical["UICC"],
                  reported_msi_status=source_clinical["MSI / MSS (IHC)"],
                  metadata_notes="Archive sample_info.tsv links native ROI specimen to Zenodo clinical_data_organoids.tsv by study code. Age is source-reported without an explicit collection-time definition. The linked paper Table S1 workbook could not be retrieved; CMS/iCMS and any additional fields remain unassigned.")
    samples_by_specimen[specimen] = sample


def source_specimen(member: str) -> str:
    stem = Path(member).name.split("_")
    year, number = stem[:2]
    candidate = f"K20{year}-{int(number)}"
    assert candidate in specimen_map, member
    return candidate


selected_specimens = {source_specimen(member) for member in members}
samples = pd.DataFrame([samples_by_specimen[key] for key in sorted(selected_specimens)],
                       columns=columns("sample_metadata") + ["reported_age_years", "reported_uicc_stage", "reported_msi_status"])
assert len(samples) == 5 and samples.patient_key.nunique() == 4


def field(source: Path | str, source_field: str, target_table: str, target_field: str,
          rule: str, *, join: str = "", notes: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(source) if isinstance(source, Path) else source, source_sheet_or_table="",
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


fields = pd.DataFrame([
    field(archive, "imaging/sample_info.tsv", "sample_metadata", "sample_id;patient_key;source_subject_id",
          "Map archive specimen key to study ID; metastatic suffix shares the primary patient key.",
          join="archive member specimen prefix -> K-year-number"),
    field(article_path, "linked Table S1 and IMC cohort description", "sample_metadata", "metadata_notes",
          "Paper states six tumor tissues and links Table S1; Zenodo clinical_data_organoids.tsv supplies available individual fields, while additional workbook fields remain unassigned.",
          join="public study ID"),
    field(clinical_path, "Patient;Organoids code", "sample_metadata", "patient_key;source_subject_id",
          "Join the specimen's public study code to unique Organoids code and check Patient identity.",
          join="sample_info.tsv public study code = Organoids code"),
    field(clinical_path, "Sex;Location;Origin;preop CTx", "sample_metadata",
          "sex_reported;anatomical_site;lesion_type;treatment_status_at_collection",
          "Map M/F to male/female; retain source Location and preoperative chemotherapy wording; distinguish primary and liver metastasis.",
          join="Organoids code"),
    field(verification_path, "source_member_name;channels;common_shape", "image_metadata",
          "source_files;source_image_id;source_channel_count;source_x1_px;source_y1_px",
          "Use complete native TIFF member and verified 50-channel 1000x1000 frame.", join="source_member_name"),
    field(archive, "native OME pages", "panel_mapping", "source_name;source_channel_index",
          "Use native OME page order; match the study S1. Retain CD39 112Cd based on representative ROI signal (88,297 nonzero pixels versus 68,864 for 114Cd).",
          join="member and ordered source channel"),
    field(archive, "native source pixels", "image_metadata", "input_image_path;source_dtype",
          "Retain native float32 values and full 1000x1000 frame; no normalization.", join="source_member_name"),
    field(archive, "source inventory", "review_results", "S01;M01",
          "37 public native members cover five specimen keys; paper describes six tissues and 40 measured ROIs, three QC exclusions.",
          notes="One described tissue has no public native TIFF in this archive; preserve published coverage gap."),
    field(clinical_path, "Age", "sample_metadata", "reported_age_years",
          "Preserve source-reported age without assigning a collection or index time origin.",
          join="Organoids code", notes=json.dumps(dict(data_type="integer", definition="Age in years as reported by Zenodo clinical table; time origin unspecified."))),
    field(clinical_path, "UICC", "sample_metadata", "reported_uicc_stage",
          "Preserve source stage label without assigning an index-time stage.",
          join="Organoids code", notes=json.dumps(dict(data_type="string", definition="Verbatim UICC stage from Zenodo clinical table; edition and time origin unspecified."))),
    field(clinical_path, "MSI / MSS (IHC)", "sample_metadata", "reported_msi_status",
          "Preserve the source IHC MSI/MSS designation.",
          join="Organoids code", notes=json.dumps(dict(data_type="string", definition="Verbatim MSI/MSS IHC status from Zenodo clinical table."))),
], columns=columns("field_mapping"))

existing_path = output / "images.parquet"
with formal_read_lease(existing_path):
    existing = {row["image_id"]: row for row in pd.read_parquet(existing_path).to_dict("records")} if existing_path.exists() else {}
if args.repair_image and args.repair_image not in existing:
    parser.error(f"repair image is absent from the published index: {args.repair_image}")
archive_revision = json.dumps(dict(size=archive.stat().st_size, mtime_ns=archive.stat().st_mtime_ns),
                              separators=(",", ":"))
progress("source_reconciled", len(existing), sample_rows=len(samples), next_member=members[len(existing)] if len(existing) < 37 else None)

# Stage one ZIP member outside the publication lease; the shared helper stages
# and publishes each image, then the four-table helper merges the new index row.
for member in members:
    image_id = f"{STUDY}__{Path(member).stem.removesuffix('.ome')}"
    if args.repair_image and image_id != args.repair_image:
        continue
    destination = output / "images" / image_id / "input.tiff"
    if not args.repair_image and image_id in existing and destination.is_file() and destination.stat().st_size and existing[image_id].get("source_revision") == archive_revision:
        continue
    if args.repair_image and existing[image_id].get("source_revision") != archive_revision:
        raise ValueError(f"Source revision changed for {image_id}; a full source review is required")
    next_bytes = 50 * 1000 * 1000 * 4
    if shutil.disk_usage(root).free < 100 * 1024**3 + next_bytes:
        progress("waiting_nas_capacity", len(existing), next_member=member)
        raise SystemExit(75)
    if shutil.disk_usage(local).free < 64 * 1024**3 + 2 * next_bytes:
        progress("waiting_local_capacity", len(existing), next_member=member)
        raise SystemExit(75)
    progress("staging_member", len(existing), next_member=member)
    with tempfile.TemporaryDirectory(prefix="plattner-member-", dir=local) as temporary:
        staged = Path(temporary) / "native.ome.tiff"
        with formal_read_lease(archive):
            with zipfile.ZipFile(archive) as source_zip, source_zip.open(member) as source, staged.open("wb") as target:
                assert source_zip.getinfo(member).CRC == proofs[member]["source_member_stat"]["CRC"]
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        assert staged.stat().st_size == proofs[member]["source_member_stat"]["file_size"]
        with tifffile.TiffFile(staged) as tif:
            names = [next(element.attrib["Name"] for element in ET.fromstring(page.description).iter()
                          if element.tag.endswith("}Image")) for page in tif.pages]
            assert names == channels and len(tif.pages) == 50
            assert all(page.shape == (1000, 1000) and page.dtype == np.dtype("float32")
                       for page in tif.pages)
            page_offsets = [page.dataoffsets[0] for page in tif.pages]
            assert len(page_offsets) == 50 and all(page.databytecounts == (4_000_000,) for page in tif.pages)
        pages = [np.memmap(staged, mode="r", dtype="float32", offset=offset, shape=(1000, 1000))
                 for offset in page_offsets]
        try:
            result = export_input(NativeReader(pages, staged), panel, destination, tile_size=256)
            assert result["admitted"], member
            with formal_read_lease(destination):
                with tifffile.TiffFile(destination) as tif:
                    assert tif.series[0].shape == (len(kept), 1000, 1000)
                    assert tif.series[0].dtype == np.dtype("float32")
                    output_names = [element.attrib["Name"] for element in ET.fromstring(tif.ome_metadata).iter()
                                    if element.tag.endswith("}Channel")]
                    assert output_names == kept.marker_id.tolist()
                    if args.repair_image:
                        pixels = next(element for element in ET.fromstring(tif.ome_metadata).iter()
                                      if element.tag.endswith("}Pixels"))
                        assert float(pixels.attrib["PhysicalSizeX"]) == 1.0
                        assert float(pixels.attrib["PhysicalSizeY"]) == 1.0
                        for output_index, source_index in enumerate(kept.source_channel_index):
                            values = tif.pages[output_index].asarray()
                            for y, x in ((0, 0), (384, 384)):
                                assert np.array_equal(values[y:y+64, x:x+64],
                                                      pages[source_index][y:y+64, x:x+64])
        finally:
            for page in pages:
                page._mmap.close()
    if args.repair_image:
        progress("repair_published_pending_independent_review", len(existing),
                 repaired_image_id=image_id, next_member=None, repair_round=1,
                 pixel_blocks_compared=2 * len(kept), sample_rows=len(samples),
                 sample_linked_images=len(existing),
                 patient_linked_samples=int(samples.patient_key.notna().sum()))
        break
    specimen = source_specimen(member)
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=image_id, sample_id=samples_by_specimen[specimen]["sample_id"],
               panel_id=panel_id, image_kind="roi", source_image_id=member,
               source_files=[relative(archive)], source_format="ome_tiff_zip_member",
               source_acquisition_id=Path(member).name, source_axes="CYX", source_dtype="float32",
               source_revision=archive_revision, roi_id=Path(member).stem.removesuffix(".ome"),
               region_annotation=Path(member).name, source_x0_px=0, source_y0_px=0,
               source_x1_px=1000, source_y1_px=1000, pad_left_px=0, pad_top_px=0,
               pad_right_px=0, pad_bottom_px=0, height_px=1000, width_px=1000,
               source_channel_count=50, channel_count=len(kept), pixel_size_x_um=1.0,
               pixel_size_y_um=1.0, input_image_path=relative(destination),
               intensity_description="Native unnormalized float32 IMC OME TIFF intensities",
               split_method="already_independent", split_evidence="Original independent Raw_tiffs ROI member",
               metadata_notes="Original public ROI; source specimen linked by sample_info.tsv. Full source frame retained.")
    existing[image_id] = row
    save_input_tables(output, pd.DataFrame(existing.values(), columns=columns("image_metadata")),
                      samples, panel, fields)
    progress("image_saved", len(existing), next_member=member,
             sample_rows=len(samples), sample_linked_images=len(existing),
             patient_linked_samples=int(samples.patient_key.notna().sum()))
    print(f"saved {len(existing)}/37", flush=True)

if args.repair_image:
    raise SystemExit(0)

review = pd.DataFrame([
    dict(study=STUDY, preparer="0343_prepare_0", reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="unresolved", source_evidence=relative(verification_path),
         output_evidence=relative(output / "images.parquet"),
         details="Preparer reconciled 37 complete native archive members. Publication describes six tissues/40 acquired ROIs/three QC exclusions; this public raw archive maps to five tissue specimens."),
    dict(study=STUDY, preparer="0343_prepare_0", reviewer="", scope="study", record_id=STUDY,
         check_id="M01", result="unresolved", source_evidence=relative(clinical_path),
         output_evidence=relative(output / "sample_metadata.csv"),
         details="Preparer linked 37 images to five source specimens and four public patient keys; Zenodo clinical fields were joined by exact study code. Paper Table S1 workbook and its additional fields remain unavailable; independent verification pending."),
    dict(study=STUDY, preparer="0343_prepare_0", reviewer="", scope="study", record_id=STUDY,
         check_id="D01", result="unresolved", source_evidence=relative(verification_path),
         output_evidence=relative(output / "images.parquet"),
         details="Preparer export complete; independent reviewer and controller acceptance pending.")
], columns=columns("review_results"))
review_path = output / "review_results.csv"
with formal_publish_lease(review_path, directory=output):
    pending = review_path.with_suffix(".csv.partial")
    review.to_csv(pending, index=False)
    pending.replace(review_path)
progress("prepared_pending_independent_review", len(existing), sample_rows=len(samples),
         sample_linked_images=len(existing), patient_linked_samples=int(samples.patient_key.notna().sum()))
