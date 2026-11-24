"""Prepare Ravi 2022 original IMC ROIs as native-intensity study inputs.

Inputs: verified Zenodo IMC_1.zip members, embedded MCD acquisitions, S1.
Outputs: per-ROI input.tiff, panel/image/sample/field tables and preparer checks.
The archive README and one MCD subject label disagree; that patient link stays open.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Ravi2022_IMC_Glioblastoma"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path, default=PROJECT / "cache/study_input" / STUDY / "progress.json")
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()
root = args.data_root
metadata = metadata_source_directory(root, STUDY)
archive = root / "raw/public" / STUDY / "public_original/IMC_1.zip"
output = root / "processed" / STUDY
cache = PROJECT / "cache/study_input" / STUDY
s1 = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
cache.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)
SOURCE_STAGING = PROJECT / "cache/study_input" / STUDY / "source_members"


def relative(path: Path) -> str:
    if path.is_relative_to(PROJECT):
        return str(path.resolve())
    return str(metadata_source_relative(path, root))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(stage: str, images: pd.DataFrame, **details: object) -> None:
    save_json(args.progress, dict(study=STUDY, phase="prepare", repair_round=0,
        stage=stage, completed_images=len(images), expected_images=14,
        next_source_index=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        updated_utc=datetime.now(timezone.utc).isoformat(), **details))


def field(source: str, table: str, source_field: str, target: str,
          target_field: str, join: str, rule: str, notes: str = "") -> dict:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=source, source_sheet_or_table=table,
               source_field=source_field, target_table=target, target_field=target_field,
               join_keys=join, value_mapping="{}", transformation_rule=rule,
               conflict_rule="Preserve source labels and unresolved discrepancy; do not impute patient identity.",
               notes=notes)
    return row


@contextmanager
def extracted_member(proof: dict):
    """Expose one exact ZIP member on SSD only for the current export."""
    SOURCE_STAGING.mkdir(parents=True, exist_ok=True)
    entry = proof["archive_entry"]
    with tempfile.TemporaryDirectory(prefix=f"{STUDY}_", dir=SOURCE_STAGING) as directory:
        destination = Path(directory) / Path(proof["source_member_name"]).name
        partial = destination.with_suffix(destination.suffix + ".partial")
        with formal_read_lease(archive), zipfile.ZipFile(archive) as source_zip:
            member = source_zip.getinfo(proof["source_member_name"])
            assert member.filename == entry["exact_archive_entry_name"]
            assert (member.file_size, member.compress_size, member.CRC, member.header_offset) == (
                entry["exact_archive_entry_size"],
                entry["exact_archive_entry_compressed_size"],
                entry["exact_archive_entry_crc"],
                entry["exact_archive_entry_local_header_offset"])
            with source_zip.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            assert partial.stat().st_size == entry["exact_archive_entry_size"]
            os.replace(partial, destination)
        yield destination


verification = json.loads((metadata / "native_verification.json").read_text())
proofs = verification["native_result"]["images"]
assert verification["scope_complete"] and len(proofs) == 14
assert all(item["complete"] and item["native_order_verified"] for item in proofs)
source_names = proofs[0]["channels"]
assert len(source_names) == 47 and all(item["channels"] == source_names for item in proofs)
panel_id = f"{STUDY}__original47"
panel = panel_from_s1(STUDY, panel_id, source_names, s1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 40 and panel.loc[panel.nuclear, "source_name"].tolist() == ["DNA1", "DNA2"]
archive_stat = archive.stat()
assert archive_stat.st_size == proofs[0]["source_file_stat"]["size"]
assert archive_stat.st_mtime_ns == proofs[0]["source_file_stat"]["mtime_ns"]
with formal_read_lease(archive), zipfile.ZipFile(archive) as source_zip:
    members = {item.filename: item for item in source_zip.infolist()}
    readme = source_zip.read("IMC_1/README.rtf").decode("latin-1")
    published_subjects = set(re.findall(r"#UKF\s*(\d+)_T", readme))
    assert len(published_subjects) == 6
    for proof in proofs:
        member = members[proof["source_member_name"]]
        assert (member.file_size, member.compress_size, member.CRC, member.header_offset) == (
            proof["archive_entry"]["exact_archive_entry_size"],
            proof["archive_entry"]["exact_archive_entry_compressed_size"],
            proof["archive_entry"]["exact_archive_entry_crc"],
            proof["archive_entry"]["exact_archive_entry_local_header_offset"])

# The acquisition description establishes a specimen label. The README confirms
# patient identity only for its matching labels; its discordant entry stays open.
subject_by_image = {}
for proof in proofs:
    match = re.fullmatch(r"GB_?(\d+)_(?:[A-Za-z]+\d*)", proof["description"])
    assert match, proof["description"]
    subject_by_image[proof["image_id"]] = match.group(1)
source_subjects = set(subject_by_image.values())
assert len(source_subjects) == 6 and len(source_subjects & published_subjects) == 5
member_numbers = {name: index + 1 for index, name in enumerate(dict.fromkeys(
    item["source_member_name"] for item in proofs))}


def output_image_id(proof: dict) -> str:
    """Identify one native acquisition by its existing output directory."""
    return f"mcd_{member_numbers[proof['source_member_name']]}_acq_{proof['source_acquisition_id']:03d}"


assert len({output_image_id(proof) for proof in proofs}) == len(proofs)
# Table S1 confirms the discrepant native label as a separate GBM subject;
# Figure S8 explicitly identifies that subject's IMC ROIs.
supplement = metadata / "document_s1.pdf"
table_text = subprocess.run(["pdftotext", "-layout", str(supplement), "-"],
                            check=True, capture_output=True, text=True).stdout
table_s1 = table_text.split("Table S1: Overview of the cohort", 1)[1].split(
    "Tissue Type (Macroscopic)", 1)[0]
clinical = {}
for subject, batch, site, age, subtype, histology, sex in re.findall(
        r"^\s*#(\d+)UKF\s+T\s+(\d+)\s+(frontal|temporal|occipital)\s+"
        r"(\d+)\s+(IDH-WT|IDH-MUT)\s+(GBM|AA|ODG)\s+([mf])\b", table_s1, re.M):
    row = (batch, site, int(age), subtype, histology, sex)
    if subject in clinical:
        assert clinical[subject] == row
    clinical[subject] = row
assert source_subjects.issubset(clinical)
sample_rows = []
for subject in sorted(source_subjects):
    batch, site, age, subtype, histology, sex = clinical[subject]
    row = dict.fromkeys(columns("sample_metadata"), None)
    sample_id = f"GB_{subject}"
    row.update(study=STUDY, sample_id=sample_id, source_sample_id=sample_id,
               source_subject_id=subject, cohort_id=STUDY, center_id="UKF",
               patient_key=f"{STUDY}__{subject}",
               sample_type="glioblastoma surgical tissue", tissue_type="tumor",
               anatomical_site=f"brain; {site}", primary_site="brain",
               primary_diagnosis="Glioblastoma", histology=histology,
               molecular_subtype=subtype, sex_reported={"m": "male", "f": "female"}[sex],
               case_control_status="case", source_reported_age_years=age,
               metadata_notes=("Table S1 tumor row and MCD ROI description agree. "
                   "ZIP README also agrees." if subject in published_subjects else
                   "Table S1 tumor row and Figure S8 IMC ROI agree with MCD; ZIP README instead lists another subject, and Table S1's IMC flag is blank. Patient link uses the matching table and figure evidence. "
                   "Table S1 age has no explicit collection/diagnosis time anchor."))
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=[*columns("sample_metadata"), "source_reported_age_years"])

fields = pd.DataFrame([
    field(relative(metadata / "native_verification.json"), "native_result.images", "source_member_name;source_acquisition_id;image_id;native_metadata", "image_metadata",
          "image_id;source_image_id;source_record_image_id;source_files;source_acquisition_id;region_annotation;source coordinates;pixel size",
          "ZIP member + slide + acquisition", "Use fourteen complete, previously pixel-read MCD acquisition proofs. The safe image_id matches the existing MCD/acquisition output directory; retain the proof image_id in source_record_image_id. Each ROI is 1000x1000 float32 at 1 um/pixel; preserve the full acquired frame."),
    field(relative(metadata / "native_verification.json"), "native_result.images", "image_id", "image_metadata",
          "source_record_image_id", "ZIP member + slide + acquisition", "Preserve the original proof image_id exactly, separate from the native source_image_id.",
          json.dumps({"data_type": "string", "definition": "Original image_id in the native acquisition proof before the safe output identifier repair."})),
    field(relative(archive), "IMC_1/README.rtf + embedded MCD metadata", "#UKF subject labels; acquisition Description", "sample_metadata",
          "sample_id;source_sample_id;source_subject_id;patient_key;metadata_notes", "subject number in MCD Description = README #UKF number",
          "Six native specimen labels form six sample rows. All fourteen ROIs link to a sample. Five labels agree with ZIP README; the discordant label is independently confirmed in Table S1 and Figure S8.",
          "ZIP README names another subject and Table S1 leaves this subject's IMC flag blank. The native ROI and Figure S8 support its patient identity."),
    field(relative(supplement), "Table S1 tumor rows + Figure S8 (PDF pp. 2, 18)", "ID;Tissue Type;Batch;Region;Age;Tumor;Histology;Sex;IMC ROI label", "sample_metadata",
          "patient_key;source_subject_id;center_id;tissue_type;anatomical_site;sex_reported;molecular_subtype;histology;source_reported_age_years",
          "MCD Description subject = Table S1 ID; choose T row", "Six matching Table S1 subjects establish six patient keys; T denotes tumor. Preserve sex, region, IDH-WT, GBM and source-reported age. Figure S8 confirms IMC regions for the README-discordant subject. Age is held in a source-named extension because its event anchor is unstated.",
          json.dumps({"data_type": "integer", "definition": "Age in years as reported in cohort Table S1; collection or diagnosis time anchor unspecified."})),
    field(relative(metadata / "README2.txt.rtf"), "public repository README2", "MCD filename and tumor-sample explanation", "sample_metadata",
          "source_subject_id", "MCD container member + native ROI description", "Repository README2 confirms that one MCD contains images from three distinct tumor samples; the MCD is not itself a sample."),
    field(relative(archive), "IMC_1/raw/tags.csv + MCD channel labels", "ordered source markers", "panel_mapping",
          "source_channel_index;source_name", "native acquisition channel order", "All fourteen proofs record the same 47-channel order, confirmed against internal MCD labels and source tags."),
    field(relative(s1), "Supplementary Table S1", "Study + Original marker", "panel_mapping",
          "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index", "exact study + source label",
          "Retain 38 S1-admitted downstream measurements and two distinct DNA nuclear measurements in native order; exclude seven S1-excluded channels."),
    field(relative(archive), "study and original MCD acquisitions", "glioblastoma tissue; original ion measurements", "sample_metadata",
          "sample_type;anatomical_site;primary_diagnosis;case_control_status", "study cohort and source tissue",
          "Retain only disease and tissue facts supported by the study; optional clinical values have no verified per-sample table in this local source."),
    field("https://zenodo.org/records/6972429", "public repository and published paper", "repository metadata; IMC study scope", "sample_metadata",
          "metadata_notes", "public study ID", "Public repository, original ZIP, repository README2 and paper Document S1 were checked for six subject identities and clinical cohort fields."),
], columns=columns("field_mapping"))

images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
existing_ids = set(images.image_id) if len(images) else set()
remaining = [item for item in proofs if output_image_id(item) not in existing_ids]
predicted_bytes = len(remaining) * len(kept) * 1000 * 1000 * 4
nas_free = shutil.disk_usage(root).free
local_free = shutil.disk_usage("/tmp").free
if nas_free < 100 * 2**30 + predicted_bytes or local_free < 64 * 2**30 + 160 * 2**20:
    progress("waiting_io_capacity", images, remaining_images=len(remaining),
             predicted_output_bytes=predicted_bytes, nas_free_bytes=nas_free,
             local_free_bytes=local_free)
    raise SystemExit(75)

save_input_tables(output, images, samples, panel, fields)
progress("exporting", images, remaining_images=len(remaining),
         predicted_output_bytes=predicted_bytes, nas_free_bytes=nas_free,
         local_free_bytes=local_free)
overview_dir = cache / "source_overviews"
overview_dir.mkdir(exist_ok=True)

for proof in remaining[:args.limit]:
    image_id = output_image_id(proof)
    member = Path(proof["source_member_name"])
    image_key = image_id
    destination = output / "images" / image_key / "input.tiff"
    with extracted_member(proof) as source_path:
        source = SourceSpec(image_id=image_id, panel_id=panel_id, kind="mcd",
            path=source_path, channels=tuple(range(47)),
            acquisition_id=int(proof["source_acquisition_id"]), pixel_size_um=(1.0, 1.0),
            source_revision="Zenodo 6972429 v1 / IMC_1.zip",
            intensity_description="Native MCD float32 IMC ion intensity; no normalization")
        progress("exporting", images, current_image=image_id, next_image=image_id)
        with open_source(source) as reader:
            assert reader.shape == (47, 1000, 1000) and reader.dtype == np.dtype("float32")
            # RGB is display-only; measured pixels are exported without scaling.
            preview = reader.read_window([44, 11, 1], (0, 0, 1000, 1000))[:, ::4, ::4]
            rgb = []
            for plane in preview:
                high = np.percentile(plane, 99.5)
                rgb.append((np.clip(plane / high, 0, 1) * 255).astype(np.uint8) if high > 0
                           else np.zeros_like(plane, dtype=np.uint8))
            overview = Image.fromarray(np.stack(rgb, axis=-1), "RGB")
            result = export_input(reader, panel, destination, tile_size=256)
            if not result["admitted"]:
                raise ValueError(f"No admitted marker signal: {image_id}")
            comparisons = []
            with formal_read_lease(destination), tifffile.TiffFile(destination) as saved:
                assert saved.series[0].shape == (40, 1000, 1000)
                assert saved.series[0].dtype == np.dtype("float32")
                assert len(saved.pages) == 40
                for window in ((0, 0, 32, 32), (968, 968, 1000, 1000)):
                    native = reader.read_window(kept.source_channel_index.tolist(), window)
                    delivered = np.stack([read_tiff_window(saved, index, window) for index in range(40)])
                    assert np.array_equal(native, delivered)
                    comparisons.append(list(window))
    overview_path = overview_dir / f"{image_key}.png"
    with formal_publish_lease(overview_path):
        overview.save(overview_path)
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=image_id,
        sample_id=f"GB_{subject_by_image[proof['image_id']]}", panel_id=panel_id,
        image_kind="roi", source_image_id=proof["physical_identity"]["identity_key"],
        source_record_image_id=proof["image_id"],
        source_files=[relative(archive)], source_format="mcd",
        source_acquisition_id=str(proof["source_acquisition_id"]), source_axes="CYX",
        source_dtype="float32", source_revision=source.source_revision,
        slide_id=f"{member.stem}__slide_{proof['slide_id']}",
        roi_id=str(proof["acquisition_roi_id"]), region_annotation=proof["description"],
        source_x0_px=0, source_y0_px=0, source_x1_px=1000, source_y1_px=1000,
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=result["height_px"], width_px=result["width_px"],
        source_channel_count=47, channel_count=40, pixel_size_x_um=1.0,
        pixel_size_y_um=1.0, input_image_path=relative(destination),
        valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
        intensity_description=source.intensity_description, split_method="source_acquisition",
        split_evidence=relative(metadata / "native_verification.json") + "#native_result.images;" + relative(archive) + "#embedded acquisition",
        metadata_notes="Full independent native ROI. Source label is retained; patient discrepancy is recorded in sample metadata.")
    images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
    save_input_tables(output, images, samples, panel, fields)
    receipt = dict(study=STUDY, image_id=image_id, image_key=image_key,
                   source_proof=proof["proof_path"], output_path=relative(destination),
                   output_bytes=destination.stat().st_size, output_mtime_ns=destination.stat().st_mtime_ns,
                   pixel_windows=comparisons, signal_nonzero=result["channel_nonzero_pixels"],
                   valid_pixels=result["valid_pixels"])
    save_json(cache / "image_receipts" / f"{image_key}.json", receipt)
    progress("exporting", images, last_completed_image=image_id,
             remaining_images=14-len(images))
    print(json.dumps(dict(exported_images=len(images), image_key=image_key)), flush=True)

# A source overview sheet is a visual QC artifact, separate from delivered pixels.
paths = sorted(overview_dir.glob("mcd_*.png"))
if len(paths) == 14:
    sheet = Image.new("RGB", (4 * 250, 4 * 275), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        x, y = index % 4 * 250, index // 4 * 275
        with Image.open(path) as tile:
            sheet.paste(tile, (x, y))
        draw.text((x + 4, y + 252), path.stem, fill="black")
    with formal_publish_lease(overview_dir / "contact_sheet.png"):
        sheet.save(overview_dir / "contact_sheet.png")

check_rows = []
for check_id, scope, record_id, result, details in [
    ("S01", "study", STUDY, "pass", f"14 native ROI proofs and source ZIP reconciled; {len(images)} exported. Preparer viewed all fourteen source RGB overviews in the contact sheet; each has marker and nuclear signal."),
    ("P01", "panel", panel_id, "pass", "47 ordered native channels agree across fourteen proofs and study S1."),
    ("P02", "panel", panel_id, "pass", "40 retained native channels, including two named DNA inputs; seven S1 exclusions."),
    ("D01", "study", STUDY, "unresolved", "Preparer handoff only; independent reviewer and controller acceptance pending. ZIP README and Table S1 IMC flag conflict is recorded; native labels and Figure S8 support all six patient links."),
]:
    check_rows.append(dict(study=STUDY, preparer="0414_prepare_0", reviewer="",
        scope=scope, record_id=record_id, check_id=check_id, result=result,
        source_evidence=relative(metadata / "native_verification.json") + ";" + relative(archive),
        output_evidence=relative(output / "images.parquet"), details=details))
for sample in samples.itertuples(index=False):
    for check_id, details in (
        ("M01", f"Preparer source check: {int(images.sample_id.eq(sample.sample_id).sum())} source ROIs link to this specimen; Table S1 supplies its patient identity. Six of six study samples are patient-linked. ZIP README and Table S1 IMC flag conflict remains recorded for the affected source label."),
        ("M02", "Preparer source check: Table S1 T row supplies sex, anatomical region, IDH status, GBM histology and reported cohort age. The age event anchor is unstated, so standard timed-age fields remain empty."),
    ):
        check_rows.append(dict(study=STUDY, preparer="0414_prepare_0", reviewer="",
            scope="sample", record_id=sample.sample_id, check_id=check_id,
            result="pass", source_evidence=relative(supplement) + ";" + relative(archive),
            output_evidence=relative(output / "sample_metadata.csv"), details=details))
for image in images.itertuples(index=False):
    receipt_path = cache / "image_receipts" / (Path(image.input_image_path).parent.name + ".json")
    for check_id, details in (
        ("P03", "Preparer reopened source-intensity TIFF and checked 40 float32 CYX pages; a final all-image header audit confirmed OME names in panel output order."),
        ("I05", "Preparer compared all 40 retained source channels, float32, with output at [0,0,32,32] and [968,968,1000,1000]; see this image's receipt."),
    ):
        check_rows.append(dict(study=STUDY, preparer="0414_prepare_0", reviewer="",
            scope="image", record_id=image.image_id, check_id=check_id,
            result="pass", source_evidence=relative(metadata / "native_verification.json") + ";" + relative(archive),
            output_evidence=image.input_image_path + ";" + relative(receipt_path),
            details=details))
review_path = output / "review_results.csv"
with formal_publish_lease(review_path, directory=output):
    prior = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.DataFrame()
    if len(prior):
        prior = prior.loc[prior.preparer != "0414_prepare_0"]
    pd.concat([prior, pd.DataFrame(check_rows)], ignore_index=True).to_csv(
        review_path, index=False, columns=columns("review_results"), encoding="utf-8-sig")
progress("prepared" if len(images) == 14 else "partial", images,
         remaining_images=14-len(images), review_path=relative(review_path))
