"""Prepare native Tanoue 2024 IMC ROIs and source-linked biological samples.

Inputs: verified Zenodo 11421121 raw TIFF members, S1, public Source Data.
Outputs: square native float32 input.tiff per ROI, study input tables, progress.
The independent reviewer supplies acceptance after this preparer closes files.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import openpyxl
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Tanoue2024_IMC_EsophagealCancer"
PIXEL_SIZE_UM = 1.0
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--max-new-images", type=int)
parser.add_argument("--metadata-only", action="store_true")
args = parser.parse_args()
if args.max_new_images is not None and args.max_new_images < 1:
    parser.error("--max-new-images must be positive")
root = args.data_root
archive = root / f"raw/public/{STUDY}/zenodo_11421121/repository_data.zip"
verification_path = metadata_source_directory(root, STUDY) / 'acquisition_verification.json'
RAW_METADATA = root / "raw/public" / STUDY / "metadata"
output = root / "processed" / STUDY
job = Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
scratch = Path("/tmp/study_input_batch") / STUDY
lock_root = Path(os.environ["CELLWEFT_NAS_LOCK_ROOT"])
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
source_data = RAW_METADATA / "source_data.xlsx"
for directory in (scratch, output, job):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return metadata_source_relative(path, root).as_posix()


def publish_json(path: Path, payload: dict) -> None:
    staged = scratch / (path.name + ".staged")
    staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with formal_publish_lease(path, directory=path.parent, io_lock_root=lock_root, data_root=root):
        pending = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, pending)
        os.replace(pending, path)


def field(source: str, source_field: str, target: str, target_field: str,
          join: str, rule: str, notes: str = "") -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=source_field, target_table=target, target_field=target_field,
                join_keys=join, value_mapping="{}", transformation_rule=rule,
                time_origin_rule="ICB start separates pretreatment and post-ICB samples; exact dates unavailable.",
                conflict_rule="Retain source labels; unconfirmed optional values remain blank.", notes=notes)


def patient_id(image: dict) -> str:
    return re.sub(r"(?:TLStu|_tumor)$", "", image["image_id"])


def sample_kind(image: dict) -> str:
    folder = image["roi_path"].split("/")[-2]
    return {"Post_ICB_ROI": "post_icb_tumor", "SLO_ROI": "slo",
            "TLS+tumorROI": "pretreatment_tumor", "Tumor_ROI": "pretreatment_tumor"}[folder]


def image_id(proof: dict) -> str:
    return proof["roi_path"].split("IMC_raw_dataset/")[1].replace("/", "__")


def progress(stage: str, images: pd.DataFrame, **details: object) -> None:
    scaled = int((images.pixel_size_x_um.eq(PIXEL_SIZE_UM) &
                  images.pixel_size_y_um.eq(PIXEL_SIZE_UM)).sum()) if len(images) else 0
    publish_json(job / "progress.json", dict(study=STUDY, stage=stage,
        updated_at_utc=datetime.now(timezone.utc).isoformat(), expected_images=len(proofs),
        exported_images=len(images), sample_rows=len(samples),
        scale_corrected_images=scaled,
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.ne("").sum()), **details))


def check_header(path: Path, selected: pd.DataFrame, side: int) -> None:
    with formal_read_lease(path, io_lock_root=lock_root, data_root=root):
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].axes == "CYX"
            assert tif.series[0].shape == (len(selected), side, side)
            assert all(page.dtype == "float32" for page in tif.pages)
            ome = ET.fromstring(tif.ome_metadata)
            pixels = next(element for element in ome.iter() if element.tag.endswith("}Pixels"))
            assert float(pixels.attrib["PhysicalSizeX"]) == PIXEL_SIZE_UM
            assert float(pixels.attrib["PhysicalSizeY"]) == PIXEL_SIZE_UM
            names = [element.attrib["Name"] for element in ome.iter()
                     if element.tag.endswith("}Channel")]
            assert names == selected.marker_id.tolist()


# The accepted verification fixes all 40 raw ROIs and their complete native plane order.
with formal_read_lease(verification_path, io_lock_root=lock_root, data_root=root):
    verification = json.loads(verification_path.read_text())
assert verification["complete"] and verification["verified_acquisitions"] == 40
assert len(verification["archives"]) == 1
accepted = verification["archives"][0]
assert accepted["complete"] and not accepted["failures"] and len(accepted["images"]) == 40
proofs = accepted["images"]
identity = proofs[0]["file_stat"]
stat = archive.stat()
assert (stat.st_size, stat.st_mtime_ns) == (identity["size"], identity["mtime_ns"])
assert all(p["complete"] and p["file_stat"]["size"] == stat.st_size and
           p["file_stat"]["mtime_ns"] == stat.st_mtime_ns for p in proofs)

panels = []
for panel_id, proof in {p["panel"]: p for p in proofs}.items():
    panel = panel_from_s1(STUDY, panel_id, proof["source_marker_order"], s1_path)
    if panel_id != "panel_002_45ch":
        low_signal = panel.source_name.isin(["CD45", "CD28"])
        panel.loc[low_signal & panel.keep_channel, "keep_reason"] += "; author reports low signal-to-noise in tumor ROIs"
    panels.append(panel)
panel = pd.concat(panels, ignore_index=True)
for proof in proofs:
    native = proof["native_channel_map"]
    assert [(row["isotope"], row["marker"]) for row in native] == [
        (read["isotope"], read["marker"]) for read in proof["bounded_native_reads"]]
    assert [row["marker"] for row in native] == proof["source_marker_order"]
    assert len(native) == proof["raw_channel_count"]
    assert [row["source_name"] for row in panel.loc[panel.panel_id.eq(proof["panel"])].to_dict("records")] == proof["source_marker_order"]

# Public figure source data resolves all patient labels; ROI folder determines sample type.
with formal_read_lease(source_data, io_lock_root=lock_root, data_root=root):
    workbook = openpyxl.load_workbook(source_data, read_only=True, data_only=True)
    response_rows = list(workbook["Fig.5a"].values)
    cps_rows = list(workbook["Fig.1d"].values)
    workbook.close()
assert response_rows[0][:3] == ("Patient ID", "Response", "TLS(1=present、0=absent)")
assert cps_rows[0][:3] == ("Patient ID", "Response", "CPS")
response = {str(row[0]).strip(): (str(row[1]).strip(), row[2]) for row in response_rows[1:]}
cps = {str(row[0]).strip(): row[2] for row in cps_rows[1:]}
assert len(response) == 27 and {patient_id(p) for p in proofs} == set(response)
assert len(set(cps) & set(response)) == 20

sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv", nrows=0).columns.tolist()
sample_rows = {}
for proof in proofs:
    subject = patient_id(proof)
    kind = sample_kind(proof)
    sample_id = f"{subject}__{kind}"
    if sample_id in sample_rows:
        continue
    row = dict.fromkeys(sample_columns, "")
    row.update(study=STUDY, sample_id=sample_id, patient_key=subject,
               source_subject_id=subject, source_sample_id=sample_id,
               sample_type="tumor tissue" if kind != "slo" else "secondary lymphoid organ",
               tissue_type="tumor" if kind != "slo" else "other",
               anatomical_site="regional lymph node" if kind == "slo" else "",
               timepoint_label="post_treatment" if kind == "post_icb_tumor" else "pre_treatment",
               treatment_status_at_collection="post ICB" if kind == "post_icb_tumor" else "pre ICB",
               preservation_method="ffpe", case_control_status="case",
               primary_diagnosis="esophageal squamous cell carcinoma",
               primary_site="esophagus", histology="squamous cell carcinoma",
               source_response_group=response[subject][0],
               source_tls_present=response[subject][1] if kind == "pretreatment_tumor" else "",
               pd_l1_cps=cps.get(subject, "") if kind == "pretreatment_tumor" else "",
               metadata_notes="Fig.5a response group is patient-level; TLS status and Fig.1d CPS apply to pretreatment tumor only. Source response R/NR/NE is not a RECIST category. Individual age/stage, exact collection date and tumor collection site unavailable.")
    sample_rows[sample_id] = row
samples = pd.DataFrame(sample_rows.values())
assert len(samples) == 33

proof_rel = relative(verification_path)
data_rel = relative(source_data)
archive_rel = relative(archive)
fields = pd.DataFrame([
    field(proof_rel, "roi_path; native_channel_map; bounded_native_reads", "image_metadata",
          "source_image_id,source_files,panel_id,roi_id,region_annotation", "archive member + ROI folder",
          "Use verified complete raw member order; ROI folder is the original region label."),
    field(proof_rel, "bounded_native_reads.shape", "image_metadata",
          "source_x0_px,source_y0_px,source_x1_px,source_y1_px", "exact ROI member",
          "Use entire native TIFF plane; no cropped field or orientation change."),
    field(archive_rel, "OME TIFF XResolution; YResolution; ResolutionUnit", "image_metadata",
          "pixel_size_x_um,pixel_size_y_um", "all 40 raw ROI first-channel planes",
          "25,400 pixels per inch in both axes with inch unit; 25,400 um per inch gives 1 um/pixel."),
    field(data_rel, "Patient ID", "sample_metadata", "patient_key,source_subject_id",
          "ROI stem after documented _tumor/TLStu suffix", "Exact public Source Data Fig.5a patient ID match for all 27 distinct IDs."),
    field(proof_rel, "roi_path folder", "sample_metadata", "sample_id,source_sample_id,timepoint_label,tissue_type",
          "patient ID + ROI folder class", "Tumor_ROI and TLS+tumorROI share pretreatment tumour sample; Post_ICB_ROI and SLO_ROI are distinct biological samples."),
    field(data_rel, "Fig.5a Response", "sample_metadata", "source_response_group", "Patient ID",
          "Preserve R/NR/NE source class; do not convert into CR/PR/SD/PD.",
          json.dumps({"data_type": "string", "definition": "Author response group in Source Data Fig.5a"})),
    field(data_rel, "Fig.5a TLS(1=present,0=absent)", "sample_metadata", "source_tls_present", "Patient ID",
          "Preserve original 1/0 status on pretreatment tumor samples only.",
          json.dumps({"data_type": "string", "definition": "Author TLS presence code in Source Data Fig.5a"})),
    field(data_rel, "Fig.1d CPS", "sample_metadata", "pd_l1_cps", "Patient ID",
          "Copy numeric combined positive score only for exact ID-matched pretreatment tumor samples.",
          json.dumps({"data_type": "number", "definition": "PD-L1 combined positive score from Source Data Fig.1d"})),
    field("https://www.nature.com/articles/s41467-024-53262-w", "Methods; Supplementary Table 1", "sample_metadata",
          "preservation_method,primary_diagnosis,primary_site,histology", "study cohort",
          "Archival FFPE ESCC material; individual age and stage omitted because Table 1 is aggregate."),
    field("https://www.nature.com/articles/s41467-024-53262-w", "Methods: Antibody validation", "panel_mapping",
          "keep_reason", "tumor ROI source panel",
          "Author reports low signal-to-noise for CD45-152Sm and CD28-173Yb in tumor ROIs; current S1 input selection is preserved pending central review."),
    field("https://zenodo.org/records/11421121", "Description: IMC_raw_dataset", "image_metadata",
          "intensity_description,source_revision", "Zenodo record 11421121",
          "Raw unprocessed 32-bit exported IMC TIFF planes; denoised and normalized data excluded."),
])
template_columns = pd.read_csv(TEMPLATES / "image_metadata.csv", nrows=0).columns.tolist()
with formal_read_lease(output, io_lock_root=lock_root, data_root=root):
    images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").is_file() else pd.DataFrame(columns=template_columns)
if args.metadata_only:
    assert len(images) > 0
    save_input_tables(output, pd.DataFrame(columns=template_columns), samples, panel, fields,
                      io_lock_root=lock_root, data_root=root)
    progress("complete" if len(images) == len(proofs) else "partial", images,
             metadata_updated=True)
    raise SystemExit(0)
revision = f"Zenodo 11421121; archive_size={stat.st_size}; archive_mtime_ns={stat.st_mtime_ns}"
def scaled_ids(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    scaled = frame.pixel_size_x_um.eq(PIXEL_SIZE_UM) & frame.pixel_size_y_um.eq(PIXEL_SIZE_UM)
    return set(frame.loc[scaled.fillna(False), "image_id"])

output_estimate = sum(len(panel.loc[panel.panel_id.eq(p["panel"]) & panel.keep_channel]) *
                      max(p["bounded_native_reads"][0]["bounded_read"]["shape"]) ** 2 * 4 for p in proofs)
max_roi_bytes = max(sum(read["source_member_bytes"] for read in p["bounded_native_reads"]) for p in proofs)
assert shutil.disk_usage(root).free > 100 * 1024**3 + output_estimate
assert shutil.disk_usage(scratch).free > 64 * 1024**3 + 2 * max_roi_bytes
progress("ready", images, output_estimate_bytes=output_estimate, max_roi_stage_bytes=max_roi_bytes,
         next_image=next((image_id(p) for p in proofs if image_id(p) not in scaled_ids(images)), ""))


# Extract one verified ROI to local scratch, export it, close it, and merge its row.
new_images = 0
with formal_read_lease(archive, io_lock_root=lock_root, data_root=root):
    with zipfile.ZipFile(archive) as source_zip:
        members = {entry.filename: entry for entry in source_zip.infolist()}
for proof in proofs:
    current_id = image_id(proof)
    destination = output / "images" / current_id / "input.tiff"
    existing = images.loc[images.image_id.eq(current_id)] if len(images) else images
    selected = panel.loc[panel.panel_id.eq(proof["panel"]) & panel.keep_channel].sort_values("output_channel_index")
    height, width = proof["bounded_native_reads"][0]["bounded_read"]["shape"]
    side = max(height, width)
    if (len(existing) == 1 and existing.iloc[0].source_revision == revision and
            current_id in scaled_ids(images) and destination.is_file()):
        check_header(destination, selected, side)
        continue
    if args.max_new_images is not None and new_images >= args.max_new_images:
        break
    try:
        with tempfile.TemporaryDirectory(prefix="roi-", dir=scratch) as temporary_root:
            temporary = Path(temporary_root)
            local_planes = []
            with formal_read_lease(archive, io_lock_root=lock_root, data_root=root):
                with zipfile.ZipFile(archive) as source_zip:
                    for read in proof["bounded_native_reads"]:
                        member = read["source_member"]
                        assert member in members and members[member].file_size == read["source_member_bytes"]
                        target = temporary / Path(member).name
                        with source_zip.open(member) as original, target.open("wb") as copied:
                            shutil.copyfileobj(original, copied, length=1024 * 1024)
                        assert target.stat().st_size == read["source_member_bytes"]
                        local_planes.append(target)
            with tifffile.TiffFile(local_planes[0]) as first_plane:
                native = first_plane.pages[0]
                x_res = native.tags["XResolution"].value
                y_res = native.tags["YResolution"].value
                assert x_res[0] / x_res[1] == y_res[0] / y_res[1] == 25400
                assert int(native.tags["ResolutionUnit"].value) == 2
            source = SourceSpec(image_id=current_id, panel_id=proof["panel"], kind="tiff_planes",
                                path=local_planes[0], paths=tuple(local_planes),
                                channels=tuple(0 for _ in local_planes), pixel_size_um=(PIXEL_SIZE_UM, PIXEL_SIZE_UM),
                                source_revision=revision,
                                intensity_description="Unprocessed MCD Viewer 32-bit IMC TIFF intensity")
            with open_source(source) as reader:
                assert reader.shape == (len(local_planes), height, width)
                assert str(reader.dtype) == "float32"
                result = export_input(reader, panel.loc[panel.panel_id.eq(proof["panel"])], destination,
                                      io_lock_root=lock_root, data_root=root)
        if not result["admitted"]:
            raise ValueError(f"No nonzero admitted marker in {current_id}")
        check_header(destination, selected, side)
        folder = proof["roi_path"].split("/")[-2]
        subject = patient_id(proof)
        row = dict(study=STUDY, image_id=current_id,
                   sample_id=f"{subject}__{sample_kind(proof)}", panel_id=proof["panel"],
                   image_kind="roi", source_image_id=proof["roi_path"],
                   source_files=[archive_rel], source_format="ome_tiff_planes_in_zip",
                   source_axes="YX per ordered channel file", source_dtype="float32",
                   source_revision=revision, roi_id=proof["image_id"],
                   region_annotation=folder, source_x0_px=0, source_y0_px=0,
                   source_x1_px=width, source_y1_px=height,
                   pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
                   pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
                   height_px=side, width_px=side, source_channel_count=len(local_planes),
                   channel_count=len(selected), pixel_size_x_um=PIXEL_SIZE_UM,
                   pixel_size_y_um=PIXEL_SIZE_UM, input_image_path=relative(destination),
                   intensity_description="Unprocessed MCD Viewer 32-bit IMC TIFF intensity",
                   split_method="already_independent", split_evidence=proof_rel + " :: " + proof["roi_path"],
                   metadata_notes="Native TIFF resolution tags specify 1 um/pixel in both axes.")
        if result["valid_pixels"]:
            row["valid_pixels_path"] = relative(destination.with_name("valid_pixels.tiff"))
        incoming = pd.DataFrame([row])
        save_input_tables(output, incoming, samples, panel, fields,
                          io_lock_root=lock_root, data_root=root)
        with formal_read_lease(output, io_lock_root=lock_root, data_root=root):
            images = pd.read_parquet(output / "images.parquet")
        new_images += 1
        progress("exporting", images, last_image=current_id,
                 next_image=next((image_id(p) for p in proofs if image_id(p) not in scaled_ids(images)), ""))
    except (OSError, TimeoutError) as error:
        progress("waiting_io", images, last_image=current_id, next_image=current_id,
                 waiting_reason=f"{type(error).__name__}: {error}")
        raise

progress("complete" if len(scaled_ids(images)) == len(proofs) else "partial", images,
         next_image=next((image_id(p) for p in proofs if image_id(p) not in scaled_ids(images)), ""))

# This preparer record leaves D01 unresolved until a separate reviewer checks the files.
review_path = output / "review_results.csv"
if len(images) == len(proofs) and not review_path.exists():
    columns = pd.read_csv(TEMPLATES / "review_results.csv", nrows=0).columns.tolist()
    review = pd.DataFrame([dict(study=STUDY, preparer=Path(__file__).name,
                                reviewer="", scope="study", record_id=STUDY, check_id="D01",
                                result="unresolved", source_evidence=proof_rel + "; " + data_rel,
                                output_evidence=relative(output / "images.parquet"),
                                details=json.dumps(dict(exported_images=len(images), sample_rows=len(samples),
                                                        sample_linked_images=int(images.sample_id.notna().sum()),
                                                        patient_linked_samples=int(samples.patient_key.ne("").sum()),
                                                        conclusion="Independent reviewer pending")))], columns=columns)
    staged = scratch / "review_results.csv"
    review.to_csv(staged, index=False, encoding="utf-8-sig")
    with formal_publish_lease(review_path, directory=output, io_lock_root=lock_root, data_root=root):
        if not review_path.exists():
            pending = review_path.with_name("review_results.csv.partial")
            shutil.copyfile(staged, pending)
            os.replace(pending, review_path)
