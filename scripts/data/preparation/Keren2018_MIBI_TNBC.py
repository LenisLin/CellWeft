"""Prepare the original 41-Point Keren TNBC release without changing intensity.

Inputs: original TNBC.zip, accepted native proofs, S1, original Table S2,
author patient-data readme and cached Point/patient-label correspondence proof.
Outputs: per-Point input.tiff and study input tables.
Run through batch shared formal I/O coordination: write for export, read for metadata-only refresh.
Clinical identities stay on NAS.
The ZIP panel table defines the complete mass order; the older native index
contains an incomplete isotope annotation and is used only for member proofs.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import io
import json
import os
from pathlib import Path
import shutil
from xml.etree import ElementTree
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import openpyxl
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Keren2018_MIBI_TNBC"
PROJECT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--points", type=int, nargs="+", help="Optional bounded batch of native Point numbers")
parser.add_argument("--metadata-only", action="store_true", help="Refresh tables for already exported images")
parser.add_argument("--progress-path", type=Path, help="Progress JSON for the current batch job")
args = parser.parse_args()
root = args.data_root
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
output = root / "processed" / STUDY
metadata = metadata_source_directory(root, STUDY)
proofs = metadata / "native_verification"
progress_path = args.progress_path or Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0008_prepare_0/progress.json"
archive_path = root / "raw/public" / STUDY / "angelolab_tnbc/TNBC.zip"
temporary = Path("/tmp/study_input_batch") / STUDY
templates = PROJECT / "docs/data/templates/study_input"
for directory in (cache, output, cache / "receipts", progress_path.parent):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return metadata_source_relative(path, root).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(templates / f"{name}.csv", nrows=0).columns.tolist()


def write_json(path: Path, value: dict) -> None:
    stage = path.with_name(path.name + ".partial")
    stage.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    stage.replace(path)


# These small original-source attachments were retrieved during source tracing.
clinical_path = metadata / "source_metadata/1-s2.0-S0092867418311000-mmc2.xlsx"
search_path = cache / "metadata_source_search.json"
source_search = json.loads((metadata / "source_metadata/metadata_source_search.json").read_text())
linkage_path = metadata / "source_metadata/patient_image_linkage_pixel_evidence.json"
linkage_evidence = json.loads(linkage_path.read_text())
assert all(check["boundary_complement_equal"] for check in linkage_evidence["checks"])
paper_text = (metadata / "source_metadata/paper_text.txt").read_text()
assert "two cores were included in the dataset (#15 and #22)" in paper_text
assert "800um" in paper_text and "2048" in paper_text
assert "patient ID" in (metadata / "source_metadata/readme.rtf").read_text()
patient_classes = pd.read_csv(metadata / "source_metadata/patient_class.csv", header=None, names=["InternalId", "class"])
assert sorted(patient_classes.InternalId.tolist()) == list(range(1, 42))
sheet = openpyxl.load_workbook(clinical_path, read_only=True, data_only=True).active
clinical_values = list(sheet.values)
clinical = pd.DataFrame(clinical_values[2:], columns=clinical_values[1]).iloc[:, :29]
clinical = clinical[clinical.InternalId.map(lambda value: isinstance(value, int))].copy()
clinical.InternalId = clinical.InternalId.astype(int)
assert len(clinical) == 39 and set(range(1, 42)) - set(clinical.InternalId) == {22, 38}
assert not clinical.DONOR_NO.duplicated().any()
clinical = clinical.set_index("InternalId", verify_integrity=True)


# The paper identifies cores 15/22 as one patient, without confirming a shared
# biological specimen. Only core15 has a Table S2 specimen record; core22 stays unlinked.
# A public InternalId remains available for 38, while its clinical accession is absent.
repair_path = cache / "point22_sample_linkage_repair.json"
patient_key_15 = f"{STUDY}__patient_15"
specimen_gap = "Point22 shares the patient of Point15; its biological specimen and any shared-biopsy relationship remain unconfirmed. sample_id is empty."
write_json(repair_path, dict(
    repair_round=1, image_ids=["Point15", "Point22"],
    verified_patient_relation=dict(image_ids=["Point15", "Point22"], patient_key=patient_key_15,
                                   evidence=relative(metadata / "source_metadata/paper.html"),
                                   source_statement="For one of the patients two cores were included in the dataset (#15 and #22)."),
    specimen_relation=dict(Point15="biopsy_15", Point22=None, status="unresolved", reason=specimen_gap),
    reviewed_sources=[relative(metadata / "source_metadata" / name) for name in
                      ("paper.html", clinical_path.name, "readme.rtf", "patient_class.csv", "author_tree.json")],
    prior_independent_evidence="Earlier independent metadata-validation review; original review session has closed.",
))
source_search["unresolved"] = [
    "No clinical row or donor accession for InternalId38",
    specimen_gap,
    "B7H3/CD163/CSF-1R original antibody QC fails but S1 maps them; current S1-admitted pixels retained and flagged for controller",
]
source_search["coverage"].update(sample_rows=40, sample_linked_images=40, patient_linked_samples=40,
                                  unlinked_sample_images=["Point22"])
source_search["specimen_linkage_repair"] = relative(repair_path)
write_json(search_path, source_search)
sample_rows = []
sample_extras = {
    "source_stage_code": ("string", "Untranslated original Table S2 STAGE code"),
    "source_ajcc_p": ("string", "Original Table S2 AJCC_P code; edition unspecified"),
    "source_site_code": ("string", "Original Table S2 SITE_02 code; coding version unspecified"),
    "source_recurrence_label": ("string", "Original recurrence label without an inferred time origin"),
    "source_survival_days_capped": ("number", "Original survival from diagnosis capped at 2016-01-01, in days"),
    "source_censored": ("integer", "Original Table S2 Censored flag; not a current vital status"),
    "source_til_score": ("number", "Original pathologist four-tier TIL score, including averaged half scores"),
}
clinical_targets = {
    "AGE_AT_DX": "age_at_diagnosis_years", "TCODE_P": "tnm_t", "NCODE_P": "tnm_n",
    "MCODE_P": "tnm_m", "GRADE": "grade", "STAGE": "source_stage_code",
    "AJCC_P": "source_ajcc_p", "SITE_02": "source_site_code",
    "RECURRENCE_LABEL": "source_recurrence_label", "Survival_days_capped*": "source_survival_days_capped",
    "Censored": "source_censored", "TIL_score": "source_til_score",
}


def source_value(value: object) -> object:
    if value is None or str(value).strip() in {"", "NaN"}:
        return None
    return value


for internal in sorted(set(range(1, 42)) - {22}):
    row = dict.fromkeys(columns("sample_metadata") + list(sample_extras))
    row.update(study=STUDY, sample_id=f"biopsy_{internal:02d}",
               patient_key=f"{STUDY}__patient_{internal:02d}", source_subject_id=str(internal),
               cohort_id=STUDY, center_id="Stanford", sample_type="biopsy",
               tissue_type="breast", anatomical_site="breast", preservation_method="FFPE",
               case_control_status="case", primary_diagnosis="triple-negative breast cancer",
               primary_site="breast", histology="carcinoma of no special type", molecular_subtype="TNBC")
    if internal in clinical.index:
        original = clinical.loc[internal]
        row["source_sample_id"] = str(int(original.DONOR_NO))
        for source, target in clinical_targets.items():
            value = source_value(original[source])
            # Unknown/unspecified pathology codes are retained as missing; raw table remains available.
            if source in {"TCODE_P", "NCODE_P", "MCODE_P", "GRADE"} and str(value) in {"X", "9", "99"}:
                value = None
            row[target] = str(value) if value is not None and target in {
                "tnm_t", "tnm_n", "tnm_m", "grade", "source_stage_code", "source_ajcc_p", "source_site_code"
            } else value
        row["metadata_notes"] = (
            "Table S2 InternalId matched to original numbered Point/core; DONOR_NO identifies the source biopsy. "
            "Capped survival and censoring retained as source extensions; no uncapped OS or present vital status inferred. "
            "Collection age, treatment, staging edition and sex are not established."
        )
        if internal == 15:
            row["metadata_notes"] += (
                " STAR Methods explicitly identifies cores 15 and 22 as one patient. "
                "Table S2 links core15 to this specimen; the biological specimen for core22 is unconfirmed, "
                "so Point22 has no sample_id and is not assigned to this biopsy."
            )
    else:
        row["metadata_notes"] = (
            "Original author patient_class.csv/readme and cohort imaging establish InternalId38 as a TNBC biopsy/patient. "
            "Table S2 omits InternalId38; DONOR_NO and clinical attributes remain unavailable."
        )
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)
for key in ("age_at_diagnosis_years", "source_survival_days_capped", "source_til_score"):
    samples[key] = pd.to_numeric(samples[key], errors="raise").astype("Float64")
samples["source_censored"] = pd.to_numeric(samples.source_censored, errors="raise").astype("Int64")

field_rows = []


def field(source_file: Path, source_field: str, target_table: str, target_field: str,
          rule: str, *, join: str = "", notes: str = "", sheet_name: str = "") -> None:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(source_file), source_sheet_or_table=sheet_name,
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join, value_mapping="{}", transformation_rule=rule, notes=notes)
    field_rows.append(row)


field(clinical_path, "InternalId;DONOR_NO", "sample_metadata", "sample_id",
      "Use original biopsy identity indexed by InternalId for source-supported specimens. Core22 has no confirmed specimen link; create no sample row for it. "
      "Create InternalId38 biopsy from the author patient release; retain missing donor/clinical information.",
      join="PointN -> InternalId N -> biopsy_N, except Point22 -> missing sample_id", sheet_name="Sheet1",
      notes="40 biological-sample rows, 40/41 image links, 40 patient-linked samples; 39 rows have Table S2 donor and clinical data. Point22 specimen identity remains unresolved.")
field(metadata / "source_metadata/readme.rtf", "Sample ID; numbered label images", "image_metadata", "sample_id",
      "Original author readme identifies numbered Sample ID as patient ID and links it to numbered label image. "
      "Cross-check image Point numbering and clinical InternalId. The paper confirms a common patient for cores15/22; "
      "it does not establish a common specimen. Leave Point22 sample_id empty.",
      join="PointN -> original patient/core N -> biopsy_N, except Point22 -> missing sample_id")
field(linkage_path, "Point1/Point38 binary boundaries; documented patient-numbered label images", "image_metadata", "metadata_notes",
      "For two full-frame representative images, original Point Segmentation.tif equals the zero-labelled boundary "
      "of the author patient pN_labeledcellData.tiff exactly at every pixel. This verifies the numbering bridge "
      "between the paired author releases, including the clinically missing patient38. Other IDs follow the same "
      "documented paired-release numbering. This is identity evidence, not acceptance of downstream mask reuse.",
      join="PointN -> documented author patient label pN -> InternalId N")
field(repair_path, "verified_patient_relation;specimen_relation", "image_metadata", "metadata_notes",
      "Preserve the directly documented same-patient relation for Point15/Point22 using public patient_key "
      f"{patient_key_15}; leave Point22 sample_id empty because the biological specimen is unconfirmed.",
      join="Point15 and Point22 -> same patient; Point22 -> unresolved specimen",
      notes="Repair round1: patient identity and biological-sample identity are separate; no Point22 clinical/sample values are inherited.")
for source, target in clinical_targets.items():
    field(clinical_path, source, "sample_metadata", target,
          "Copy original value; blank and literal NaN become missing. Pathological X/9/99 become missing for TNM/grade only. "
          "Do not decode unknown stage/site code systems or infer unstated collection/time semantics.",
          join="InternalId -> confirmed original biopsy sample; no clinical/sample row for22", sheet_name="Sheet1")
for target, (kind, definition) in sample_extras.items():
    mapping = next(row for row in field_rows if row["target_field"] == target)
    mapping["notes"] = json.dumps({"data_type": kind, "definition": definition})
    if target == "source_survival_days_capped":
        mapping["time_origin_rule"] = "diagnosis; capped at 2016-01-01 per original Table S2 footer"
for target in ("patient_key", "source_subject_id", "source_sample_id"):
    field(clinical_path, "InternalId;DONOR_NO", "sample_metadata", target,
          "For each confirmed sample row, namespaced InternalId defines public patient_key; InternalId is source_subject_id. "
          "The paper identifies core22 with patient15; preserve that image/patient relation in metadata_notes and the repair evidence, "
          "with no inferred Point22 specimen/sample assignment. DONOR_NO supplies source_sample_id when available; "
          "InternalId38 has no source biopsy accession and that field remains empty.",
          join="InternalId -> sample_id", sheet_name="Sheet1")
field(metadata / "source_metadata/paper.html", "STAR Methods; cohort description", "sample_metadata",
      "sample_type;tissue_type;anatomical_site;preservation_method;histology;primary_diagnosis;primary_site;molecular_subtype;case_control_status;cohort_id;center_id",
      "Apply source cohort-wide TNBC biopsy, breast, FFPE, no-special-type and Stanford annotations.")
field(metadata / "source_metadata/paper.html", "800um at 2048 x 2048 pixels", "image_metadata", "pixel_size_x_um;pixel_size_y_um",
      "800 / 2048 = 0.390625 um/pixel. The separately stated 500nm probe resolution is not the sampling pitch.")


# Correct the incomplete historical source panel annotation from the original ZIP attachment.
with zipfile.ZipFile(archive_path) as archive:
    source_panel_bytes = archive.read("TNBCShareData/TNBCpanelInfo.csv")
    panel_source_path = metadata / "source_metadata/TNBCpanelInfo.csv"
    panel_source_path.write_bytes(source_panel_bytes)
    native_panel = pd.read_csv(io.BytesIO(source_panel_bytes), keep_default_na=False).sort_values("Isotope")
    names = native_panel.Label.tolist()
    assert len(names) == 49 and len(set(names)) == 49
    bad_rows = {i: f"Original TNBCpanelInfo.csv source QC: {native_panel.iloc[i].Comment}"
                for i, name in enumerate(names) if name in {"B7H3", "CD163", "CSF-1R"}}
    panel_id = f"{STUDY}__native49_mass_order"
    panel = panel_from_s1(STUDY, panel_id, names,
                         root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv")
    # S1 is the authorized admission table. Preserve its admitted pixels and
    # expose original antibody QC for the controller's separate S1 decision.
    for index, reason in bad_rows.items():
        panel.loc[index, "keep_reason"] += "; " + reason + "; retained under current S1 pending controller QC decision"
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    field(panel_source_path, "Isotope;Label;Comment", "panel_mapping", "source_channel_index;source_name;keep_channel;role",
          "Source index is ascending native Isotope order, covering all 49 individual files. Join exact study/source labels to S1. "
          "B7H3/CD163/CSF-1R have original bad-antibody comments; retain their admitted pixels under current S1 and flag the QC conflict for controller review.",
          notes="Historical ordered_channel_mapping.json appends unidentified channels and is not a complete mass order; "
          "its member size/CRC proofs are reused. This study-local interpretation should replace the historical annotation after controller review.")
    field_mapping = pd.DataFrame(field_rows, columns=columns("field_mapping"))
    old_index = json.loads((metadata / "ordered_channel_mapping.json").read_text())["acquisitions"]
    archive_stat = archive_path.stat()
    revision = f"angelolab:Keren2018/TNBC.zip:size={archive_stat.st_size}:mtime_ns={archive_stat.st_mtime_ns}:native49_mass_order_v1"
    for acquisition in old_index.values():
        assert acquisition["archive"]["stat"] == {"size": archive_stat.st_size, "mtime_ns": archive_stat.st_mtime_ns}
        for channel in acquisition["ordered_channels"]:
            member = archive.getinfo(channel["member"])
            assert (member.file_size, member.CRC) == (channel["member_bytes"], channel["zip_crc"])
    members = [i for i in archive.infolist() if not i.is_dir()]
    point_tiffs = [i for i in members if len(Path(i.filename).parts) == 3
                   and Path(i.filename).parts[0] == "TNBCShareData" and i.filename.endswith(".tif")]
    native_members = [i for i in point_tiffs if Path(i.filename).stem in names]
    derived_masks = [i.filename for i in point_tiffs if Path(i.filename).stem not in names]
    assert len(native_members) == 41 * 49
    assert len(derived_masks) == 82 and {Path(i).stem for i in derived_masks} == {"Segmentation", "SegmentationInterior"}
    cell_type_masks = [i.filename for i in members if i.filename.startswith("TNBCcellTypes/") and i.filename.endswith(".tiff")]
    assert len(cell_type_masks) == 41
    source_inventory = dict(archive=relative(archive_path), complete_file_count=len(members),
                            measurement_tiffs=len(native_members), point_boundary_interior_masks=len(derived_masks),
                            cell_type_label_images=len(cell_type_masks), native_points=41,
                            non_image_members=[i.filename for i in members if not i.filename.lower().endswith((".tif", ".tiff"))],
                            uncompressed_archive_bytes=sum(i.file_size for i in members))
    write_json(cache / "source_inventory.json", source_inventory)

    image_columns = columns("image_metadata")
    images = pd.read_parquet(output / "images.parquet").to_dict("records") if (output / "images.parquet").exists() else []
    existing = {row["image_id"]: row for row in images}
    overview_review_path = cache / "overview_review.json"
    if overview_review_path.exists():
        # Human inspection follows the first export and is reapplied on metadata refresh.
        overview_review = json.loads(overview_review_path.read_text())
        corrected_notes = {
            "Point15": "Preparer viewed the complete source-frame composite. One source core field and surrounding background; no separate adjacent core is evident. "
                       f"Core15 and core22 are the same patient ({patient_key_15}) per STAR Methods. Point15 links to biopsy_15; Point22 specimen identity is unresolved.",
            "Point22": "Preparer viewed the complete source-frame composite. One source core field; no separate adjacent core is evident. "
                       f"Core22 and core15 are the same patient ({patient_key_15}) per STAR Methods. Point22 sample_id is empty because its biological specimen is unconfirmed.",
        }
        for inspection in overview_review["images"]:
            old_note = inspection["notes"]
            if inspection["image_id"] in corrected_notes:
                inspection["notes"] = corrected_notes[inspection["image_id"]]
            if inspection["image_id"] in existing:
                row = existing[inspection["image_id"]]
                if old_note != inspection["notes"]:
                    row["metadata_notes"] = row["metadata_notes"].replace(old_note, "").rstrip()
                if inspection["notes"] not in row["metadata_notes"]:
                    row["metadata_notes"] += " " + inspection["notes"]
        write_json(overview_review_path, overview_review)
        field(overview_review_path, "image_id;notes", "image_metadata", "metadata_notes",
              "Append actual preparer full-source-frame inspection notes; preserve original ROI and intensities.",
              join="image_id")
        field_mapping = pd.DataFrame(field_rows, columns=columns("field_mapping"))
    # Apply the specimen repair before the first publication/checkpoint.
    for point, row in existing.items():
        number = int(point.removeprefix("Point"))
        row["sample_id"] = None if number == 22 else f"biopsy_{number:02d}"
    exclusions_path = cache / "excluded_images.json"
    excluded = json.loads(exclusions_path.read_text()).get("images", []) if exclusions_path.exists() else []
    pixel_checks = []
    overviews = []
    selected_points = args.points or list(range(1, 42))
    assert set(selected_points).issubset(set(range(1, 42)))
    if args.metadata_only:
        selected_points = [int(row["image_id"].removeprefix("Point")) for row in images]
    reusable = new_count = 0

    def checkpoint(stage: str, current: str = "", *, save_tables: bool = True) -> None:
        if save_tables:
            save_input_tables(output, pd.DataFrame(list(existing.values()), columns=image_columns), samples, panel, field_mapping)
            write_json(exclusions_path, {"images": excluded})
        write_json(progress_path, dict(stage=stage, current_image=current, expected_images=41,
                   completed_images=len(existing), excluded_images=len(excluded), sample_rows=len(samples),
                   sample_linked_images=sum(bool(r["sample_id"]) for r in existing.values()),
                   patient_linked_samples=int(samples.patient_key.notna().sum()),
                   gaps=[specimen_gap, "InternalId38 lacks Table S2 donor/clinical data", "Independent metadata recheck and D01 binding refresh pending",
                         "Source bad-antibody QC differs from S1 for B7H3/CD163/CSF-1R"] ))

    checkpoint("source_metadata_ready")
    print(f"Source 41 x 49 channels; retaining {len(kept)}; 40 samples; uncompressed output upper bound {41*len(kept)*2048**2*2} bytes", flush=True)

    for point_number in selected_points:
        point = f"Point{point_number}"
        input_path = output / "images" / point / "input.tiff"
        receipt_path = cache / "receipts" / f"{point}.json"
        sample_id = None if point_number == 22 else f"biopsy_{point_number:02d}"
        if point in existing:
            # Reuse only a previously completed export of this exact source and panel.
            receipt = json.loads(receipt_path.read_text())
            assert receipt["source_revision"] == revision and receipt["source_names"] == names
            assert receipt["output_names"] == kept.marker_id.tolist()
            with tifffile.TiffFile(input_path) as tif:
                assert tif.series[0].shape == (len(kept), 2048, 2048) and tif.series[0].dtype == np.dtype("uint16")
                ome = ElementTree.fromstring(tif.ome_metadata)
                assert [el.attrib["Name"] for el in ome.iter() if el.tag.endswith("}Channel")] == kept.marker_id.tolist()
            reusable += 1
            checkpoint("reusing_complete_inputs", point, save_tables=False)
            continue
        if args.metadata_only:
            continue
        if shutil.disk_usage(root).free < 100 * 1024**3 + 320 * 1024**2 or shutil.disk_usage("/tmp").free < 64 * 1024**3 + 300 * 1024**2:
            raise RuntimeError("Insufficient capacity for this Point while preserving the 100/64 GiB reserves")
        proof = json.loads((proofs / f"{point}.json").read_text())
        assert proof["actual_pixel_read"] and proof["shape"] == [2048, 2048]
        native_dir = temporary / point
        native_dir.mkdir(parents=True, exist_ok=True)
        source_paths = []
        for name in names:
            member_name = f"TNBCShareData/{point}/{name}.tif"
            member = archive.getinfo(member_name)
            destination = native_dir / f"{name}.tif"
            if not destination.exists() or destination.stat().st_size != member.file_size:
                partial = destination.with_name(destination.name + ".partial")
                with archive.open(member) as stream, partial.open("wb") as target:
                    shutil.copyfileobj(stream, target, length=8*1024**2)
                assert partial.stat().st_size == member.file_size
                partial.replace(destination)
            source_paths.append(destination)
        source = SourceSpec(image_id=point, panel_id=panel_id, kind="tiff_planes", path=source_paths[0],
                            paths=tuple(source_paths), channels=(0,)*49, pixel_size_um=(800/2048, 800/2048),
                            source_revision=revision,
                            intensity_description="Original author-published uint16 MIBI channel TIFF counts; upstream processing is not independently reconstructed; no CellWeft intensity normalization")
        with open_source(source) as reader:
            assert reader.shape == (49, 2048, 2048) and reader.dtype == np.dtype("uint16")
            stats = export_input(reader, panel, input_path)
            if not stats["admitted"]:
                excluded.append({"image_id": point, "reason": "All admitted marker channels are zero across the entire image"})
                checkpoint("exporting", point)
                shutil.rmtree(native_dir)
                continue
            checks = []
            if point_number in {1, 15, 22, 30, 38, 41}:
                with tifffile.TiffFile(input_path) as saved:
                    for window in ((960, 960, 992, 992), (2016, 2016, 2048, 2048)):
                        native = reader.read_window(kept.source_channel_index.tolist(), window)
                        observed = np.stack([read_tiff_window(saved, i, window) for i in range(len(kept))])
                        assert native.dtype == observed.dtype and np.array_equal(native, observed)
                        checks.append({"window_xyxy": list(window), "all_retained_channels": len(kept),
                                       "nonzero_pixels": int(np.count_nonzero(native)), "equal": True})
                # Small composite overviews retain the entire acquisition frame.
                from PIL import Image
                channels = [names.index(label) for label in ("Pan-Keratin", "CD45", "dsDNA")]
                composite = []
                for channel in channels:
                    plane = reader.read_window([channel], (0, 0, 2048, 2048))[0]
                    reduced = np.asarray(Image.fromarray(plane.astype(np.float32)).resize((384, 384), Image.Resampling.BOX), dtype=float)
                    scale = np.percentile(reduced[reduced > 0], 99.5) if np.any(reduced) else 1
                    composite.append(np.clip(reduced/scale, 0, 1))
                overview_path = cache / f"{point}_overview.png"
                Image.fromarray((np.stack(composite, axis=-1)*255).astype("uint8")).save(overview_path)
                overviews.append(relative(overview_path))
        with tifffile.TiffFile(input_path) as saved:
            assert saved.series[0].shape == (len(kept), 2048, 2048)
            ome = ElementTree.fromstring(saved.ome_metadata)
            saved_names = [el.attrib["Name"] for el in ome.iter() if el.tag.endswith("}Channel")]
            assert saved_names == kept.marker_id.tolist()
        image = dict.fromkeys(image_columns)
        image.update(study=STUDY, image_id=point, sample_id=sample_id, panel_id=panel_id,
                     image_kind="tma_core", source_image_id=f"TNBCShareData/{point}",
                     source_files=[relative(archive_path)], source_format="tiff_planes", source_axes="CYX",
                     source_dtype="uint16", source_revision=revision, core_id=point, roi_id=point,
                     source_x0_px=0, source_y0_px=0, source_x1_px=2048, source_y1_px=2048,
                     pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                     height_px=2048, width_px=2048, source_channel_count=49, channel_count=len(kept),
                     pixel_size_x_um=800/2048, pixel_size_y_um=800/2048,
                     input_image_path=relative(input_path), intensity_description=source.intensity_description,
                     split_method="already_independent",
                     split_evidence="Original author TNBCShareData/PointN independent full 2048x2048 acquisition; per-core TMA biopsy in STAR Methods",
                     metadata_notes="Channel filenames and acquisition membership match accepted native proof; complete mass order from original TNBCpanelInfo.csv. "
                     + (f"Point15/Point22 are the same patient ({patient_key_15}); Point22 specimen identity is unresolved. " if point_number in {15, 22} else "")
                     + ("Paper excludes Point30 from its analyses due to noise; retained as a complete local input under current full-frame nonzero-marker rule. " if point_number == 30 else "")
                     + ("Clinical Table S2 absent for InternalId38; original author patient release confirms identity." if point_number == 38 else ""))
        existing[point] = image
        write_json(receipt_path, dict(source_revision=revision, source_names=names, output_names=kept.marker_id.tolist(),
                                     stats=stats, pixel_checks=checks, original_native_proof=relative(proofs / f"{point}.json")))
        pixel_checks.extend({"image_id": point, **check} for check in checks)
        new_count += 1
        checkpoint("exporting", point)
        print(f"saved {len(existing)}/41 {point}", flush=True)
        shutil.rmtree(native_dir)


all_pixel_checks = [dict(image_id=point, **check) for point in existing
                    for check in json.loads((cache / "receipts" / f"{point}.json").read_text())["pixel_checks"]]
summary = dict(study=STUDY, expected_images=41, exported_images=len(existing), new_images=new_count,
               reused_images=reusable, excluded_images=excluded, sample_rows=len(samples),
               sample_linked_images=sum(bool(row["sample_id"]) for row in existing.values()),
               patient_linked_samples=int(samples.patient_key.notna().sum()), clinical_source_rows=39,
               source_channel_count=49, output_channel_count=len(kept), source_order="TNBCpanelInfo.csv ascending Isotope",
               original_derived_masks_not_input_channels=derived_masks,
               source_inventory=relative(cache / "source_inventory.json"),
               pixel_checks_this_run=pixel_checks, overview_paths_this_run=overviews,
               pixel_checks=all_pixel_checks,
               overview_review=relative(overview_review_path) if overview_review_path.exists() else None,
               source_search=relative(search_path), clinical_missing=[38],
               patient_linkage_pixel_evidence=relative(linkage_path),
               specimen_linkage_repair=relative(repair_path), unresolved_sample_images=["Point22"], repair_round=1,
               controller_requests=[
                   "核对本study S1中的B7H3、CD163、CSF-1R：原始TNBCpanelInfo.csv明确记载抗体失败；当前按唯一S1保留其准入像素并在keep_reason标注来源QC，建议主控核实是否更改S1。",
                   "将本study旧ordered_channel_mapping中的附加unknown通道更新为归档原始panel的完整49通道同位素及顺序；现有member核验证明保持有效。",
                   "请独立复查Point22 sample_id留空、15/22同患者关系及相关表述修补；复用0122技术核验，刷新共享metadata变化后其余40张的D01绑定。",
               ])
write_json(cache / "input_delivery_summary.json", summary)

# Only preparer checks are recorded here; reviewer stays empty for a separate CLI.
review_path = output / "review_results.csv"
review = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.DataFrame(columns=columns("review_results"))
review = review[~((review.record_id == f"{STUDY}:preparer") & (review.reviewer == ""))]
check_details = {
    "S01": f"Preparer: 41 native Point acquisitions, all 49 source channels reconciled to ZIP; {len(existing)} exported; {len(excluded)} all-zero exclusions. Original labeled-cell-type masks are separate derived objects.",
    "I01": "Preparer: unique Point IDs, archive members and NAS-relative paths recorded.",
    "I02": f"Preparer repair1: {sum(bool(row['sample_id']) for row in existing.values())}/{len(existing)} image/sample links; 40 unique sample rows; 40 patient-linked samples. Point22 sample_id is empty; nonempty foreign keys resolve.",
    "I03": "Preparer: complete 2048-square frames, uint16, zero padding. Sampling pitch 800/2048 um, independently distinguished from probe resolution.",
    "I04": "Preparer repair1: original separate per-core acquisition frames; no image region annotations provided. Cores15/22 are the same patient per STAR Methods; a shared specimen is unconfirmed. Point22 sample_id is empty. Existing overview images and technical review are reused.",
    "P01": "Preparer: all 49 rows in original isotope order; exact study and source-name S1 joins. Old index order corrected locally using ZIP panel attachment.",
    "P02": f"Preparer: {len(kept)} kept, dsDNA nucleus retained; B7H3/CD163/CSF-1R source QC conflicts flagged in keep_reason and retained under current S1. No duplicate downstream markers.",
    "P03": f"Preparer: {len(existing)} TIFF headers reopened, CYX uint16 and standardized channel names/order checked.",
    "I05": "Preparer: all retained channels compared exactly in central and boundary 32x32 windows for Points1/15/22/30/38/41 when newly exported; per-image receipt records actual comparison coverage.",
    "M01": json.dumps(dict(repair_round=1, image_ids=["Point22"], sample_rows=40,
                           sample_linked_images=sum(bool(row["sample_id"]) for row in existing.values()),
                           patient_linked_samples=40, clinical_source_rows=39,
                           summary="Reused tracing of local proof/index/ZIP, original paper/Table S2, author repo, patient archive readme and patient_class. "
                                   "Point22 sample_id cleared: same-patient evidence does not establish a common specimen. "
                                   "Point15 retains biopsy_15 and both cores retain documented patient15 identity in image notes and repair evidence. "
                                   "InternalId38 donor/clinical fields remain empty.", evidence=relative(repair_path))),
    "M02": "Preparer repair1: removed unsupported shared-biopsy statements in sample15, image15/22, field_mapping, overview notes and tracing summary. Existing Table S2 values/units, unknown codes and seven source extensions retained. Independent M02 recheck pending.",
    "D01": json.dumps(dict(repair_round=1, image_ids=list(existing),
                           summary=f"Preparer delivery: {len(existing)}/41 input TIFFs retained; 40/41 sample links, 40 samples and 40 patient links. "
                                   "Point22 specimen identity remains unresolved. Prior independent technical findings are preserved. "
                                   "Shared metadata changed, so prior D01 bindings require independent refresh; preparer grants no independent acceptance.")),
}
review_rows = []
for check, details in check_details.items():
    review_rows.append(dict(study=STUDY, preparer=Path(__file__).name, reviewer="", scope="study",
                            record_id=f"{STUDY}:preparer", check_id=check,
                            result="unresolved" if check in {"M01", "D01"} else "pass",
                            source_evidence=relative(search_path), output_evidence=relative(output), details=details))
review_stage = review_path.with_name(review_path.name + ".partial")
with formal_publish_lease(review_path, directory=output):
    pd.concat([review, pd.DataFrame(review_rows)], ignore_index=True).to_csv(review_stage, index=False, encoding="utf-8-sig")
    review_stage.replace(review_path)
checkpoint("prepared_awaiting_independent_review" if len(existing) + len(excluded) == 41
           else "metadata_ready_waiting_export", save_tables=False)
if temporary.exists() and not any(temporary.iterdir()):
    temporary.rmdir()
print(json.dumps({key: value for key, value in summary.items()
                  if key != "original_derived_masks_not_input_channels"}, ensure_ascii=False, indent=2), flush=True)
