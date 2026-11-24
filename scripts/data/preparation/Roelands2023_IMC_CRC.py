"""Export verified Roelands CRC MCD acquisitions as native intensity inputs.

Inputs: four local S-BIAD587 MCD files, native verification, ROI annotation,
source manifest, and study S1. Outputs: per-ROI CYX TIFF and study input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Roelands2023_IMC_CRC"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_JOB",
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0419_prepare_0",
))
TEMPLATES = PROJECT / "docs/data/templates/study_input"
PROGRESS = JOB / "progress.json"


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, completed: int, next_image: str | None, **details: object) -> None:
    payload = dict(study=STUDY, stage=stage, completed_images=completed,
                   expected_images=25, next_image=next_image,
                   updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    JOB.mkdir(parents=True, exist_ok=True)
    pending = PROGRESS.with_suffix(".json.partial")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pending.replace(PROGRESS)


def field(path: Path, source_field: str, target_table: str, target_field: str,
          rule: str, notes: str = "", join_keys: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(path), source_sheet_or_table="",
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join_keys, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


# Reconcile the exact native scope and public ROI-to-specimen bridge.
verification_path = META / "native_verification.json"
mapping_path = META / "roi_mapping.json"
manifest_path = META / "source_manifest.json"
with formal_read_lease(META, data_root=ROOT):
    verification = json.loads(verification_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    manifest = json.loads(manifest_path.read_text())
proofs = {row["native_proof"]["description"]: row["native_proof"]
          for row in verification["records"]}
annotations = {row["id"]: row for row in mapping["rows"]}
assert len(proofs) == len(annotations) == 25
assert verification["scope_complete"] and mapping["source_annotation_complete"]
assert set(proofs) == set(annotations)
assert len(verification["source_defects"]) == 3
assert all(item["scope_status"] == "held_not_in_reviewed_scope"
           for item in verification["source_defects"])
assert "eight-patient" in manifest["qualification"]["people_basis"]
assert len({row["sample"] for row in annotations.values()}) == 8

sources = ROOT / "raw/public" / STUDY / "S-BIAD587"
for proof in proofs.values():
    source = sources / proof["source_member_name"]
    assert source.stat().st_size == proof["source"]["size"]
    assert proof["complete"] and proof["counted"]
    assert proof["physical_identity"]["sample_id"] == annotations[proof["description"]]["sample"]

names = next(iter(proofs.values()))["channels"]
assert all(proof["channels"] == names for proof in proofs.values())
panel_id = f"{STUDY}__S-BIAD587_48"
s1_path = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
with formal_read_lease(s1_path, data_root=ROOT):
    panel = panel_from_s1(STUDY, panel_id, names, s1_path)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(names) == 48 and len(kept) == 42

# An ESD specimen is one source sample; ROI histology varies within specimens.
samples = pd.DataFrame([
    dict(study=STUDY, sample_id=sample, patient_key=sample,
         source_subject_id=sample, source_sample_id=sample,
         sample_type="tissue", tissue_type="colorectum",
         primary_diagnosis="early-stage colorectal cancer",
         metadata_notes="Public ESD specimen and patient grouping from ROI annotation and source manifest; ROI-level normal/carcinoma labels are in image metadata. Other clinical fields are unreported here.")
    for sample in sorted({row["sample"] for row in annotations.values()})
], columns=columns("sample_metadata"))
fields = pd.DataFrame([
    field(sources / "ROIs_annotation.xlsx", "id;sample;ROI;Histology", "image_metadata",
          "source_image_id;sample_id;roi_id;region_annotation",
          "Exact annotation ID joins native acquisition description; sample is the ESD specimen; retain Histology verbatim.",
          f"Parsed row binding: {relative(mapping_path)}; 25 of 25 admitted ROIs linked to eight samples.",
          "annotation.id = native_proof.description = image_id; annotation.sample = sample_id; ROI is scoped by sample"),
    field(sources / "ROIs_annotation.xlsx", "sample", "sample_metadata",
          "sample_id;source_sample_id", "One biological ESD specimen per distinct source sample ID.",
          "Eight source specimen rows represented among 25 admitted ROIs.",
          "annotation.sample = sample_id = source_sample_id; distinct ESD specimen IDs"),
    field(manifest_path, "qualification.people_basis", "sample_metadata",
          "patient_key;source_subject_id", "Eight public ESD specimen IDs identify the eight patient groups.",
          "25 of 25 source ROIs link to eight samples and eight patient groups; no additional clinical fields inferred. Published study: https://doi.org/10.1136/gutjnl-2022-327608.",
          "annotation.sample = sample_id = source_sample_id; one patient group per ESD specimen in manifest qualification.people_basis"),
    field(manifest_path, "tissue;disease", "sample_metadata",
          "sample_type;tissue_type;primary_diagnosis",
          "Study-level ESD tissue and early-stage colorectal cancer; region-specific normal/carcinoma is retained per image.",
          "No patient-level stage, timing, treatment or other clinical values inferred.",
          "manifest.dataset_name = study; applies to eight ESD specimens in the admitted study scope"),
    field(verification_path, "records.native_proof", "image_metadata",
          "source_files;source_acquisition_id;source_channel_count",
          "Use complete counted native proofs and original MCD acquisition IDs.",
          join_keys="native_proof.description = image_id; source_member_name and source_acquisition_id identify MCD acquisition"),
    field(s1_path, "Study;Original marker", "panel_mapping", "marker_id;role;keep_channel",
          "Join all 48 ordered native source labels to current study S1.",
          join_keys="S1.Study = study; S1.Original marker = ordered native source_name"),
], columns=columns("field_mapping"))

existing = {}
if (OUTPUT / "images.parquet").exists():
    with formal_read_lease(OUTPUT, data_root=ROOT):
        prior = pd.read_parquet(OUTPUT / "images.parquet")
    existing = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
    for image_id in set(existing) & set(annotations):
        existing[image_id]["roi_id"] = str(annotations[image_id]["ROI"])

# Stage one image outside the writer lease; helper owns read and publish leases.
for image_id in sorted(proofs):
    proof = proofs[image_id]
    destination = OUTPUT / "images" / image_id / "input.tiff"
    if image_id in existing and destination.is_file():
        continue
    next_bytes = len(kept) * int(proof["metadata"]["MaxX"]) * int(proof["metadata"]["MaxY"]) * 4
    if shutil.disk_usage(ROOT).free < (100 * 1024**3 + next_bytes):
        checkpoint("waiting_io", len(existing), image_id, reason="NAS 100 GiB reserve")
        raise SystemExit(75)
    if shutil.disk_usage("/tmp").free < (64 * 1024**3 + 2 * next_bytes):
        checkpoint("waiting_io", len(existing), image_id, reason="local 64 GiB reserve")
        raise SystemExit(75)
    source = sources / proof["source_member_name"]
    spec = SourceSpec(image_id=image_id, panel_id=panel_id, kind="mcd", path=source,
                      channels=tuple(range(48)), acquisition_id=int(proof["source_acquisition_id"]),
                      pixel_size_um=(1.0, 1.0))
    checkpoint("exporting", len(existing), image_id, source=source.name)
    with open_source(spec) as reader:
        result = export_input(reader, panel, destination, tile_size=256, data_root=ROOT)
    if not result["admitted"]:
        checkpoint("excluded_no_marker_signal", len(existing), image_id)
        continue
    with formal_read_lease(destination, data_root=ROOT):
        with tifffile.TiffFile(destination) as tif:
            assert tif.series[0].shape == (len(kept), result["height_px"], result["width_px"])
            assert tif.series[0].dtype.name == "float32"
            channel_names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter()
                             if node.tag.endswith("}Channel")]
            assert channel_names == kept.marker_id.tolist()
    annotation = annotations[image_id]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=image_id, sample_id=annotation["sample"],
               panel_id=panel_id, image_kind="roi", source_image_id=image_id,
               source_files=[relative(source)], source_format="mcd",
               source_acquisition_id=str(proof["source_acquisition_id"]),
               source_axes="CYX", source_dtype="float32", source_revision="S-BIAD587",
               slide_id=str(proof["slide_id"]), roi_id=str(annotation["ROI"]),
               region_annotation=annotation["Histology"],
               source_x0_px=0, source_y0_px=0,
               source_x1_px=int(proof["metadata"]["MaxX"]),
               source_y1_px=int(proof["metadata"]["MaxY"]),
               pixel_size_x_um=1.0, pixel_size_y_um=1.0,
               input_image_path=relative(destination),
               valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                  if result["valid_pixels"] else None),
               intensity_description="Native unnormalized float32 MCD acquisition intensity",
               split_method="source_acquisition", split_evidence=relative(mapping_path),
               metadata_notes="Complete native acquisition; original ROI annotation linked by description.",
               **{key: result[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px",
                                                 "pad_bottom_px", "height_px", "width_px",
                                                 "source_channel_count", "channel_count")})
    existing[image_id] = row
    save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields, data_root=ROOT)
    checkpoint("image_saved", len(existing), image_id, sample_rows=len(samples),
               sample_linked_images=len(existing), patient_linked_samples=len(samples))
    print(f"saved {len(existing)}/25", flush=True)

save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields, data_root=ROOT)
field_path = OUTPUT / "field_mapping.csv"
with formal_read_lease(field_path, data_root=ROOT):
    published_fields = pd.read_csv(field_path, dtype=str, keep_default_na=False)
repaired_keys = set(zip(fields.source_file, fields.source_field,
                        fields.target_table, fields.target_field))
stale = ((published_fields.study == STUDY) &
         (published_fields.join_keys == "") &
         published_fields.apply(
             lambda row: (row.source_file, row.source_field, row.target_table,
                          row.target_field) in repaired_keys, axis=1))
if stale.any():
    with formal_publish_lease(field_path, directory=OUTPUT, data_root=ROOT):
        partial = field_path.with_suffix(".csv.partial")
        published_fields.loc[~stale].to_csv(partial, index=False, encoding="utf-8-sig")
        partial.replace(field_path)
review = pd.DataFrame([
    dict(study=STUDY, preparer="0419_prepare_0", reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="unresolved", source_evidence=relative(verification_path),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details=f"Preparer indexed {len(existing)} of 25 eligible complete normal/carcinoma ROIs. Source manifest lists 60 annotated biological ROIs, 4 unannotated ROIs and 4 laser tests; transition/dysplasia histology is deferred. Native proof holds 3 defective acquisitions outside reviewed scope. Independent review pending."),
    dict(study=STUDY, preparer="0419_prepare_0", reviewer="", scope="sample", record_id=STUDY,
         check_id="M01", result="unresolved", source_evidence=relative(mapping_path),
         output_evidence=relative(OUTPUT / "sample_metadata.csv"),
         details=f"Public ROI annotation links {len(existing)} images to eight ESD samples; source manifest identifies eight patient groups. Additional clinical fields have no local source; independent review pending."),
    dict(study=STUDY, preparer="0419_prepare_0", reviewer="", scope="study", record_id=STUDY,
         check_id="D01", result="unresolved", source_evidence=relative(verification_path),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details="Preparer output closed; independent reviewer and controller acceptance pending."),
], columns=columns("review_results"))
review_path = OUTPUT / "review_results.csv"
with formal_read_lease(review_path, data_root=ROOT):
    previous_review = pd.read_csv(review_path) if review_path.exists() else None
if previous_review is None or (
    previous_review.preparer.fillna("").eq("0419_prepare_0").all()
    and previous_review.reviewer.fillna("").eq("").all()
    and previous_review.result.eq("unresolved").all()
):
    with formal_publish_lease(review_path, directory=OUTPUT, data_root=ROOT):
        partial = OUTPUT / "review_results.csv.partial"
        review.to_csv(partial, index=False, encoding="utf-8-sig")
        partial.replace(review_path)
checkpoint("prepared_pending_independent_review", len(existing), None,
           sample_rows=len(samples), sample_linked_images=len(existing),
           patient_linked_samples=len(samples))
