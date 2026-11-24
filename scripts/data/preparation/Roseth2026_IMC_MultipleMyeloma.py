"""Prepare Roseth's original IMC marrow ROIs as native-intensity inputs.

Inputs: verified original MCD acquisitions, published clinical CSV, study S1.
Outputs: per-ROI CYX input TIFF and study input tables; independent review follows.
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

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import _export_input, panel_from_s1, save_input_tables


STUDY = "Roseth2026_IMC_MultipleMyeloma"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--job", type=Path, default=None)
parser.add_argument("--metadata-only", action="store_true", help="Refresh clinical tables after completed image export")
parser.add_argument("--repair-review-1", action="store_true", help="Apply the first independent review's scoped repair")
args = parser.parse_args()
root = args.data_root
job = args.job or Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
metadata = metadata_source_directory(root, STUDY)
output = root / "processed" / STUDY
clinical_path = root / "raw/public" / STUDY / "metadata/metadata_imc_myeloma.csv"
author_images_path = clinical_path.with_name("images.csv")
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
stage_root = Path("/tmp/study_input_batch") / STUDY
stage_root.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return metadata_source_relative(path, root).as_posix()


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, next_image_id: str | None, completed: int, expected: int,
               sample_rows: int) -> None:
    value = dict(study=STUDY, stage=stage, expected_images=expected,
                 completed_images=completed, next_image_id=next_image_id,
                 sample_rows=sample_rows, sample_linked_images=completed,
                 patient_linked_samples=sample_rows,
                 phase="repair" if args.repair_review_1 else "prepare",
                 repair_round=1 if args.repair_review_1 else 0,
                 updated_at_utc=datetime.now(timezone.utc).isoformat())
    job.mkdir(parents=True, exist_ok=True)
    pending = job / "progress.json.partial"
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(pending, job / "progress.json")


def field(source: str, source_field: str, table: str, target: str,
          rule: str, join: str = "", notes: str = "") -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=source_field, target_table=table, target_field=target,
                join_keys=join, value_mapping="{}", transformation_rule=rule,
                time_origin_rule="", conflict_rule="Unmatched or conflicting joins stay empty",
                notes=notes)


def verify_tiff(path: Path, side: int, names: list[str], valid: Path | None) -> None:
    with formal_read_lease(path), tifffile.TiffFile(path) as image:
        assert image.series[0].shape == (len(names), side, side)
        assert image.series[0].dtype == "float32"
        assert all(f'Name="{name}"' in image.ome_metadata for name in names)
    if valid is not None:
        with formal_read_lease(valid), tifffile.TiffFile(valid) as image:
            assert image.pages[0].shape == (side, side)
            assert image.pages[0].dtype == "uint8"


if args.repair_review_1:
    checkpoint("awaiting_source_verification_read", None, 128, 127, 65)

# Published member proofs identify native acquisitions; LEtest acquisitions are
# instrument tests, and three defective acquisitions were excluded by the proof.
with formal_read_lease(metadata / "native_verification.json"):
    verification = json.loads((metadata / "native_verification.json").read_text())
assert verification["scope_complete"] and verification["native_result"]["actual_n_images"] == 139
records = verification["native_result"]["images"]
with formal_read_lease(author_images_path):
    author_images = pd.read_csv(author_images_path, dtype=str, keep_default_na=False)
author_acquisitions = {
    (row.source_file, int(float(row.acquisition_id)))
    for row in author_images.itertuples(index=False)
    if row.source_file.endswith(".mcd") and row.acquisition_id
}
accepted = [record for record in records
            if not record["description"].startswith("LEtest")
            and (Path(record["source_member_name"]).name,
                 record["source_acquisition_id"]) in author_acquisitions]
assert len(accepted) == 127 and len(records) - len(accepted) == 12

with formal_read_lease(clinical_path):
    clinical = pd.read_csv(clinical_path, dtype=str, keep_default_na=False)
assert not clinical.duplicated(["IMC label", "Cohort"]).any()
source_keys = {(record["source_member_manifest"]["source_label"],
                record["source_member_manifest"]["clinical_cohort"]) for record in accepted}
clinical = clinical[clinical[["IMC label", "Cohort"]].apply(tuple, axis=1).isin(source_keys)]
assert len(source_keys) == len(clinical) == 65
clinical_by_key = clinical.set_index(["IMC label", "Cohort"])

# The exact archive label and cohort pair links each MCD to one published row.
sample_rows = []
for (label, cohort), source in clinical_by_key.iterrows():
    row = dict.fromkeys(columns("sample_metadata"), "")
    row.update(study=STUDY, sample_id=f"{STUDY}__{label}",
               patient_key=f"{STUDY}__{label}", source_subject_id=label,
               source_sample_id=label, cohort_id=cohort,
               sample_type="bone marrow trephine", anatomical_site="bone marrow",
               collection_method="iliac crest trephine biopsy", preservation_method="FFPE",
               timepoint_label="diagnosis", treatment_status_at_collection="treatment-naive",
               primary_diagnosis="multiple myeloma", sex_reported=source["Gender"],
               age_at_diagnosis_years=source["Age time of diagnosis"],
               metadata_notes="Published IMC label and cohort match the original MCD member exactly; clinical fields retain source time semantics.")
    row.update(source_r_iss_stage=source["R-ISS"],
               source_pfs_from_diagnosis_days=source["PFS from first time of diagnosis until relapse (days)"],
               source_pfs_event=source["Event (Progress) occurred"],
               source_os_days=source["OS (days)"],
               source_os_event=source["Event (Death) occurred"],
               source_bone_disease_at_diagnosis=source["Bone disease upon time of diagnosis/biopsy"])
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)

source_labels = records[0]["channels"]
assert all(record["channels"] == source_labels for record in records)
panel_id = STUDY + "__native42"
with formal_read_lease(s1_path):
    panel = panel_from_s1(STUDY, panel_id, source_labels, s1_path,
                          aliases={"": "<Unnamed channel>"})
panel["display_name"] = panel.marker_id
panel.loc[panel.role.eq("segmentation-only"), "display_name"] = panel.source_name
assert len(panel) == 42
kept = panel[panel.keep_channel].sort_values("output_channel_index")
names = kept.marker_id.tolist()

fields = pd.DataFrame([
    field(relative(metadata / "native_verification.json"), "physical_identity", "image_metadata", "image_id",
          "Use verified original instrument acquisition/ROI identity"),
    field(relative(metadata / "native_verification.json"), "ordered_channel_rows", "panel_mapping", "source_name",
          "Preserve native channel order and join source labels to study S1"),
    field(relative(clinical_path), "IMC label; Cohort", "image_metadata", "sample_id",
          "Exact original MCD label and cohort pair; one published row per source specimen",
          join="IMC label + Cohort = source_label + clinical_cohort",
          notes="65 matched source specimens; no ROI-number patient inference"),
    field(relative(clinical_path), "IMC label", "sample_metadata", "patient_key",
          "Preserve published anonymous specimen label", join="IMC label + Cohort"),
    field(relative(clinical_path), "Gender", "sample_metadata", "sex_reported", "Preserve published value"),
    field(relative(clinical_path), "Age time of diagnosis", "sample_metadata", "age_at_diagnosis_years",
          "Published age at diagnosis, in years"),
    field(relative(clinical_path), "Cohort", "sample_metadata", "cohort_id", "Preserve published cohort code"),
    field(relative(clinical_path), "R-ISS", "sample_metadata", "source_r_iss_stage",
          "Preserve source stage string without inferring index-stage timing",
          notes=json.dumps({"data_type": "string", "definition": "Published R-ISS stage"})),
    field(relative(clinical_path), "PFS from first time of diagnosis until relapse (days)", "sample_metadata",
          "source_pfs_from_diagnosis_days", "Preserve source duration without changing event definition",
          notes=json.dumps({"data_type": "string", "definition": "Source PFS duration from diagnosis to relapse, days"})),
    field(relative(clinical_path), "Event (Progress) occurred", "sample_metadata", "source_pfs_event",
          "Preserve source event string",
          notes=json.dumps({"data_type": "string", "definition": "Source progression event indicator"})),
    field(relative(clinical_path), "OS (days)", "sample_metadata", "source_os_days",
          "Preserve source duration pending origin confirmation",
          notes=json.dumps({"data_type": "string", "definition": "Source OS duration in days; origin unconfirmed"})),
    field(relative(clinical_path), "Event (Death) occurred", "sample_metadata", "source_os_event",
          "Preserve source event string",
          notes=json.dumps({"data_type": "string", "definition": "Source death event indicator"})),
    field(relative(clinical_path), "Bone disease upon time of diagnosis/biopsy", "sample_metadata",
          "source_bone_disease_at_diagnosis", "Preserve source value",
          notes=json.dumps({"data_type": "string", "definition": "Bone disease at diagnosis or biopsy, source value"})),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "collection_method", "Diagnosis-time treatment-naive FFPE iliac crest trephine biopsy"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "sample_type", "Study specimen is a bone marrow trephine biopsy"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "anatomical_site", "Iliac crest bone marrow collection site"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "preservation_method", "Study specimen is formalin-fixed paraffin-embedded"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "timepoint_label", "Diagnosis-time specimen; no later timepoint inferred",
          notes="Collection timepoint, not the origin of source OS duration"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "treatment_status_at_collection", "Diagnosis-time treatment-naive cohort"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC13139845/", "Study cohort methods", "sample_metadata",
          "primary_diagnosis", "Study cohort has multiple myeloma"),
], columns=columns("field_mapping"))

if args.repair_review_1:
    image_path = output / "images.parquet"
    field_path = output / "field_mapping.csv"
    with formal_read_lease(image_path):
        image_table = pq.read_table(image_path)
    with formal_read_lease(field_path):
        old_fields = pd.read_csv(field_path, dtype=str, keep_default_na=False)
    accepted_ids = {
        STUDY + "__" + re.sub(r"[^A-Za-z0-9]+", "_", Path(record["source_member_name"]).stem).strip("_")
        + f"__acq_{record['source_acquisition_id']}" for record in accepted
    }
    published_ids = set(image_table.column("image_id").to_pylist())
    omitted = [record for record in records if not record["description"].startswith("LEtest")
               and (Path(record["source_member_name"]).name,
                    record["source_acquisition_id"]) not in author_acquisitions]
    assert len(omitted) == 1 and omitted[0]["data_boundary"]["height"] == 47
    omitted_id = (STUDY + "__" + re.sub(r"[^A-Za-z0-9]+", "_", Path(omitted[0]["source_member_name"]).stem).strip("_")
                  + f"__acq_{omitted[0]['source_acquisition_id']}")
    assert len(accepted_ids) == 127 and len(image_table) == len(published_ids)
    assert accepted_ids <= published_ids <= accepted_ids | {omitted_id}
    missing_fields = fields[~fields.target_field.isin(old_fields.target_field)]
    repair_fields = {
        "sample_type", "anatomical_site", "preservation_method", "timepoint_label",
        "treatment_status_at_collection", "primary_diagnosis",
    }
    assert set(missing_fields.target_field) <= repair_fields
    repaired_images = image_table.filter(pc.not_equal(image_table.column("image_id"), omitted_id))
    repaired_fields = pd.concat([old_fields, missing_fields], ignore_index=True)
    if omitted_id in published_ids or len(missing_fields):
        checkpoint("repair_staged", omitted_id, len(published_ids), 127, len(samples))
        with tempfile.TemporaryDirectory(prefix="roseth-repair-", dir=stage_root) as temporary:
            temporary = Path(temporary)
            staged_images = temporary / "images.parquet"
            staged_fields = temporary / "field_mapping.csv"
            if omitted_id in published_ids:
                pq.write_table(repaired_images, staged_images)
            if len(missing_fields):
                repaired_fields.to_csv(staged_fields, index=False, encoding="utf-8-sig")
            versions = tuple((path.stat().st_size, path.stat().st_mtime_ns)
                             for path in (image_path, field_path))
            with formal_publish_lease(output, directory=output):
                assert versions == tuple((path.stat().st_size, path.stat().st_mtime_ns)
                                         for path in (image_path, field_path))
                for staged, destination in ((staged_images, image_path), (staged_fields, field_path)):
                    if staged.is_file():
                        pending = destination.with_name(destination.name + ".partial")
                        shutil.copyfile(staged, pending)
                        os.replace(pending, destination)
    checkpoint("index_repaired", omitted_id, 127, 127, len(samples))
    rejected_dir = output / "images" / omitted_id
    rejected_image = rejected_dir / "input.tiff"
    if rejected_image.is_file() or (rejected_dir / "valid_pixels.tiff").is_file():
        with formal_publish_lease(rejected_image, directory=rejected_dir, lock_name=".input.lock"):
            rejected_image.unlink(missing_ok=True)
            (rejected_dir / "valid_pixels.tiff").unlink(missing_ok=True)
    checkpoint("repaired_awaiting_independent_review", None, 127, 127, len(samples))
    print(json.dumps(dict(study=STUDY, expected_images=127, exported_images=127,
                          sample_rows=len(samples), sample_linked_images=127,
                          patient_linked_samples=len(samples))))
    raise SystemExit(0)

with formal_read_lease(output / "images.parquet"):
    prior = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").is_file() else pd.DataFrame()
images = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
if args.metadata_only and len(images) != len(accepted):
    raise ValueError("Metadata-only refresh requires the complete previously verified image scope")
checkpoint("source_checked", accepted[0]["image_id"], len(images), len(accepted), len(samples))

for record in ([] if args.metadata_only else accepted):
    identity = record["physical_identity"]
    image_id = STUDY + "__" + re.sub(r"[^A-Za-z0-9]+", "_", Path(record["source_member_name"]).stem).strip("_") + f"__acq_{record['source_acquisition_id']}"
    source = Path(record["source_file_path"])
    proof = record["source_file_stat"]
    stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (proof["size"], proof["mtime_ns"]):
        raise ValueError(f"Source changed since native proof: {source.name}")
    boundary = record["data_boundary"]
    side = max(boundary["height"], boundary["width"])
    expected_valid = boundary["record_count"] != boundary["height"] * boundary["width"]
    destination = output / "images" / image_id / "input.tiff"
    valid_destination = destination.with_name("valid_pixels.tiff")
    if image_id in images and destination.is_file() and (not expected_valid or valid_destination.is_file()):
        verify_tiff(destination, side, names, valid_destination if expected_valid else None)
        checkpoint("reused", image_id, len(images), len(accepted), len(samples))
        continue
    source_spec = SourceSpec(image_id=image_id, panel_id=panel_id, kind="mcd", path=source,
                             channels=tuple(range(42)), acquisition_id=record["source_acquisition_id"])
    checkpoint("reading_source", image_id, len(images), len(accepted), len(samples))
    with tempfile.TemporaryDirectory(prefix="roseth-", dir=stage_root) as temporary:
        staged = Path(temporary) / "input.tiff"
        with formal_read_lease(source), open_source(source_spec) as reader:
            geometry = _export_input(reader, panel, staged, tile_size=256)
        if not geometry["admitted"]:
            raise ValueError(f"Verified tissue ROI has no admitted marker signal: {image_id}")
        staged_valid = staged.with_name("valid_pixels.tiff") if geometry["valid_pixels"] else None
        verify_tiff(staged, side, names, staged_valid)
        checkpoint("publishing_image", image_id, len(images), len(accepted), len(samples))
        with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name("input.tiff.partial")
            shutil.copyfile(staged, pending)
            os.replace(pending, destination)
            if staged_valid is not None:
                pending_valid = valid_destination.with_name("valid_pixels.tiff.partial")
                shutil.copyfile(staged_valid, pending_valid)
                os.replace(pending_valid, valid_destination)
        verify_tiff(destination, side, names, valid_destination if staged_valid else None)
    label = record["source_member_manifest"]["source_label"]
    cohort = record["source_member_manifest"]["clinical_cohort"]
    assert (label, cohort) in clinical_by_key.index
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=image_id, sample_id=f"{STUDY}__{label}", panel_id=panel_id,
               image_kind="roi", source_image_id=identity["unit_id"], source_files=[relative(source)],
               source_format="MCD", source_acquisition_id=str(record["source_acquisition_id"]),
               source_axes="CYX", source_revision="zenodo:17093203", slide_id=identity["slide_id"],
               roi_id=str(identity["roi_id"]), region_annotation=record["description"],
               source_x0_px=0, source_y0_px=0, source_x1_px=boundary["width"],
               source_y1_px=boundary["height"], input_image_path=relative(destination),
               valid_pixels_path=relative(valid_destination) if geometry["valid_pixels"] else None,
               intensity_description="Original MCD float32 counts; no normalization or clipping",
               split_method="already_independent", split_evidence=relative(metadata / "native_verification.json"),
               metadata_notes="Native acquisition coordinates and accepted source proof; centered zero padding.",
               **{key: geometry[key] for key in ("height_px", "width_px", "pad_left_px", "pad_top_px",
                                                  "pad_right_px", "pad_bottom_px", "source_dtype",
                                                  "source_channel_count", "channel_count")})
    images[image_id] = row
    save_input_tables(output, pd.DataFrame(list(images.values())), samples, panel, fields)
    checkpoint("image_committed", image_id, len(images), len(accepted), len(samples))

# A metadata-only rerun must publish updated clinical rules even when every TIFF is reused.
save_input_tables(output, pd.DataFrame(list(images.values())), samples, panel, fields)

# These rows document preparer checks; D01 is reserved for the independent reviewer.
review_path = output / "review_results.csv"
checks = [
    dict(study=STUDY, preparer=job.name, reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="pass", source_evidence=relative(metadata / "native_verification.json"),
         output_evidence=relative(output / "images.parquet"),
         details="139 complete instrument acquisitions: 127 author-listed tissue ROIs exported; 11 LEtest and one 47-row partial acquisition excluded. Three other defective acquisitions were excluded upstream."),
    dict(study=STUDY, preparer=job.name, reviewer="", scope="study", record_id=STUDY,
         check_id="M01", result="pass", source_evidence=relative(clinical_path),
         output_evidence=relative(output / "sample_metadata.csv"),
         details="Exact IMC label and cohort join covers 65 source specimens, 127 images and 65 patient keys; independent clinical review pending."),
]
rows = pd.DataFrame(checks, columns=columns("review_results"))
if not review_path.is_file():
    with formal_publish_lease(review_path, directory=output):
        if not review_path.is_file():
            pending = review_path.with_name("review_results.csv.partial")
            rows.to_csv(pending, index=False, encoding="utf-8-sig")
            os.replace(pending, review_path)
checkpoint("exported_awaiting_independent_review", None, len(images), len(accepted), len(samples))
print(json.dumps(dict(study=STUDY, expected_images=len(accepted), exported_images=len(images),
                      sample_rows=len(samples), sample_linked_images=len(images),
                      patient_linked_samples=len(samples))))
