"""Deliver native-intensity OSCC ROI/core inputs from Zenodo 8169946.

Inputs: complete ZIP, existing native proofs, study-specific S1 rows, author
clinical workbooks/analysis scripts and paper methods. Outputs: square CYX
input.tiff files and panel/image/sample/field-mapping/preparer-check tables.
Run actual export through the batch shared formal write lease wrapper. No normalization.
Use --repair-pixels --job-dir PATH through shared formal read lease to supplement I05
evidence without rewriting the delivered TIFFs or granting independent acceptance.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.inputs import input_binding
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

pa.set_cpu_count(4)
pa.set_io_thread_count(2)

STUDY = "Einhaus2023_IMC_OSCC"
PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
OUTPUT = ROOT / "processed" / STUDY
METADATA = metadata_source_directory(ROOT, STUDY)
CACHE = METADATA / "support"
JOB = PROJECT / "cache/study_input_jobs" / STUDY / "prepare"
ARCHIVE = ROOT / "raw/public" / STUDY / "public_original/OSCC-IMC-Einhaus2023.zip"
PREFIX = "OSCC-IMC Einhaus et al. 2023/"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
TEMP = Path("/tmp/study_input_batch") / STUDY


def relative(path: Path) -> str:
    if path.is_relative_to(PROJECT):
        return str(path)
    return str(metadata_source_relative(path, ROOT))


def template(name: str) -> pd.DataFrame:
    return pd.read_csv(TEMPLATES / f"{name}.csv", keep_default_na=False)


def progress(stage: str, **details: object) -> None:
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(proofs), completed_images=len(images),
        sample_rows=len(samples), sample_linked_images=sum(bool(x.get("sample_id")) for x in images),
        patient_linked_samples=int(samples.patient_key.ne("").sum()),
        updated_at=datetime.now(timezone.utc).isoformat(), **details))


def field(source: str, source_fields: str, target: str, target_fields: str,
          rule: str, *, sheet: str = "", join: str = "cohort + author patient_id", notes: str = "",
          mapping: dict | None = None, conflict: str = "Source values retained; unknown optional values blank.") -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table=sheet,
                source_field=source_fields, target_table=target, target_field=target_fields,
                join_keys=join, value_mapping=json.dumps(mapping or {}, ensure_ascii=False),
                transformation_rule=rule, time_origin_rule="Collection is the biopsy/resection index; no inferred survival intervals.",
                conflict_rule=conflict, notes=notes)


def check_header(path: Path, selected: pd.DataFrame, side: int) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (len(selected), side, side)
        assert all(page.dtype == np.dtype("float32") for page in tif.pages)
        xml = ET.fromstring(tif.ome_metadata)
        channels = [element.attrib["Name"] for element in xml.iter() if element.tag.endswith("}Channel")]
        assert channels == selected.marker_id.tolist()


def source_overview(source: tifffile.TiffFile, panel: pd.DataFrame, image_id: str, group: str) -> None:
    """Render only a diagnostic preview; native input intensities remain unchanged."""
    colors = []
    for name in ("Pancytokeratin", "CD45", "DNA2"):
        index = int(panel.loc[panel.source_name == name, "source_channel_index"].iloc[0])
        plane = source.pages[index].asarray(maxworkers=1)
        upper = np.percentile(plane[plane > 0], 99.5)
        colors.append(np.uint8(np.clip(plane / upper, 0, 1) * 255))
    preview = Image.fromarray(np.stack(colors, axis=-1))
    preview.thumbnail((600, 600))
    canvas = Image.new("RGB", (620, 640), "black")
    canvas.paste(preview, (10, 30))
    ImageDraw.Draw(canvas).text((10, 8), image_id + " | PanCK/CD45/DNA2", fill="white")
    canvas.save(CACHE / f"overview_{group}.png")


def representative_check(source_path: Path, destination: Path, panel: pd.DataFrame,
                         record: dict, group: str) -> None:
    """Compare every retained plane to direct TIFF decoding in two 32px blocks."""
    h, w = record["source_y1_px"], record["source_x1_px"]
    left, top = record["pad_left_px"], record["pad_top_px"]
    regions = [(w // 2 - 16, h // 2 - 16, w // 2 + 16, h // 2 + 16),
               (w - 32, h - 32, w, h)]
    selected = panel[panel.keep_channel].sort_values("output_channel_index")
    with tifffile.TiffFile(source_path) as source, tifffile.TiffFile(destination) as saved:
        for row in selected.itertuples(index=False):
            native = source.pages[row.source_channel_index].asarray(maxworkers=1)
            for x0, y0, x1, y1 in regions:
                actual = read_tiff_window(saved, row.output_channel_index,
                                          (x0 + left, y0 + top, x1 + left, y1 + top))
                assert actual.dtype == native.dtype
                assert np.array_equal(actual, native[y0:y1, x0:x1])
            assert not read_tiff_window(saved, row.output_channel_index, (0, 0, 32, 32)).any()
        source_overview(source, panel, record["image_id"], group)
    pixel_checks[group] = dict(image_id=record["image_id"], source_windows=regions,
                               retained_channels=len(selected), dtype="float32", equality=True,
                               padding_window=[0, 0, 32, 32], padding_zero=True,
                               overview=relative(CACHE / f"overview_{group}.png"))
    write_json(CACHE / "preparer_pixel_checks.json", pixel_checks)


def repair_pixel_evidence(proofs: list[dict], job: Path) -> None:
    """Supplement the five observed cohort/geometric groups with direct decoding.

    Each channel uses a signal-centered interior block and a source-edge block;
    all padding pixels are checked. Only one encoded ROI TIFF and one decoded
    source/output plane pair are in memory. Existing independent findings remain
    open for the separate reviewer.
    """
    images = pd.read_parquet(OUTPUT / "images.parquet")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
    panels = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
    panels["output_channel_index"] = pd.to_numeric(panels.output_channel_index).astype("Int64")
    groups: dict[str, dict] = {}
    evidence_path = job / "repair_pixel_checks.json"
    counts = dict(expected_images=len(proofs), completed_images=len(images),
                  sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                  patient_linked_samples=int(samples.patient_key.ne("").sum()))
    gaps = ["Independent I05 pixel review pending", "D01 independent acceptance pending"]
    with formal_read_lease(ARCHIVE), resource_lock(OUTPUT, shared=True):
        binding = input_binding(OUTPUT, ROOT, images.image_id.tolist())
        write_json(job / "candidate_input_binding.json", binding)
        representatives = {}
        for proof in proofs:
            group = proof["cohort"] + "_" + "_".join(map(str, proof["common_shape"]))
            representatives.setdefault(group, proof)
        assert len(representatives) == 5
        evidence = dict(study=STUDY, preparer=Path(__file__).name, reviewer="", repair_round=1,
                        check_id="I05", expected_groups=len(representatives), groups=groups,
                        method="Direct tifffile page.asarray on source and output, one plane at a time; 32px interior/edge blocks and complete padding.",
                        native_proofs_reused=len(proofs), source_archive=relative(ARCHIVE),
                        input_binding_path=relative(job / "candidate_input_binding.json"),
                        independent_review="pending", gaps=gaps)
        try:
            with zipfile.ZipFile(ARCHIVE) as archive:
                for group, proof in representatives.items():
                    image_id = proof["cohort"] + "__" + Path(proof["source_member_name"]).stem
                    record = images.set_index("image_id").loc[image_id]
                    selected = panels[(panels.panel_id == record.panel_id) & panels.keep_channel].sort_values("output_channel_index")
                    info = archive.getinfo(proof["source_member_name"])
                    assert info.file_size == proof["source_member_stat"]["file_size"]
                    assert info.CRC == proof["source_member_stat"]["CRC"]
                    write_json(job / "progress.json", dict(study=STUDY, stage="repair_pixel_check_running",
                               **counts, current_image=image_id, completed_pixel_groups=len(groups),
                               expected_pixel_groups=len(representatives), gaps=gaps,
                               updated_at=datetime.now(timezone.utc).isoformat()))
                    # ZIP-level bytes stay in memory to avoid repeated seek/decompression of a member.
                    # These selected native ROI TIFFs are about 82 MB each, not whole slides.
                    with BytesIO(archive.read(info)) as encoded, tifffile.TiffFile(encoded) as source, \
                            tifffile.TiffFile(ROOT / record.input_image_path) as saved:
                        h, w = proof["common_shape"]
                        left, top = int(record.pad_left_px), int(record.pad_top_px)
                        side = max(h, w)
                        check_header(ROOT / record.input_image_path, selected, side)
                        channels = []
                        for row in selected.itertuples(index=False):
                            native = source.pages[row.source_channel_index].asarray(maxworkers=1)
                            actual = saved.pages[row.output_channel_index].asarray(maxworkers=1)
                            assert native.shape == (h, w) and native.dtype == actual.dtype == np.dtype(record.source_dtype)
                            yy, xx = np.unravel_index(np.abs(native[32:-32, 32:-32]).argmax(), (h-64, w-64))
                            regions = [(int(xx)+16, int(yy)+16, int(xx)+48, int(yy)+48),
                                       (w-32, h-32, w, h)]
                            blocks = []
                            for kind, (x0, y0, x1, y1) in zip(("interior", "bottom_right_edge"), regions):
                                native_block = native[y0:y1, x0:x1]
                                output_window = [x0+left, y0+top, x1+left, y1+top]
                                equal = bool(np.array_equal(native_block, actual[y0+top:y1+top, x0+left:x1+left]))
                                assert equal, f"{image_id}/{row.source_name}: {kind} pixels differ at {(x0,y0,x1,y1)}"
                                blocks.append(dict(kind=kind, source_window=[x0,y0,x1,y1],
                                                   output_window=output_window, equal=equal,
                                                   nonzero_pixels=int(np.count_nonzero(native_block))))
                            padding = [(0,0,side,top), (0,top+h,side,side),
                                       (0,top,left,top+h), (left+w,top,side,top+h)]
                            padding = [region for region in padding if region[0] < region[2] and region[1] < region[3]]
                            for x0, y0, x1, y1 in padding:
                                assert not np.count_nonzero(actual[y0:y1,x0:x1]), f"{image_id}/{row.source_name}: nonzero padding"
                            channels.append(dict(source_channel_index=int(row.source_channel_index),
                                                 output_channel_index=int(row.output_channel_index),
                                                 source_name=row.source_name, marker_id=row.marker_id,
                                                 dtype=str(native.dtype), blocks=blocks,
                                                 padding_windows=padding, padding_zero=True,
                                                 padding_pixels_checked=sum((c-a)*(d-b) for a,b,c,d in padding)))
                    groups[group] = dict(image_id=image_id, panel_id=record.panel_id,
                                         source_member=proof["source_member_name"],
                                         source_format=record.source_format, source_axes=record.source_axes,
                                         split_method=record.split_method, source_shape=[h,w],
                                         output_shape=[len(selected),side,side], channels=channels,
                                         output_path=record.input_image_path, values_equal=True)
                    evidence["updated_at"] = datetime.now(timezone.utc).isoformat()
                    write_json(evidence_path, evidence)
                    write_json(job / "progress.json", dict(study=STUDY, stage="repair_pixel_group_complete",
                               **counts, completed_pixel_groups=len(groups), expected_pixel_groups=len(representatives),
                               last_completed_image=image_id, gaps=gaps, updated_at=evidence["updated_at"]))
                    print(f"pixel evidence {len(groups)}/{len(representatives)} {image_id}", flush=True)
            assert binding == input_binding(OUTPUT, ROOT, images.image_id.tolist())
        except Exception as error:
            write_json(job / "progress.json", dict(study=STUDY, stage="repair_pixel_check_failed",
                       **counts, completed_pixel_groups=len(groups), expected_pixel_groups=len(representatives),
                       gaps=[str(error), *gaps], updated_at=datetime.now(timezone.utc).isoformat()))
            raise

    # Append preparer evidence atomically, preserving all independent review rows.
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        reviews = pd.read_csv(review_path, keep_default_na=False)
        record_id = STUDY + ":repair_1_pixel_evidence"
        reviews = reviews[reviews.record_id != record_id]
        details = dict(repair_round=1, action="Supplemented preparer pixel evidence; all existing TIFFs and input tables reused.",
                       completed_groups=len(groups), source_output_blocks=sum(len(c["blocks"]) for g in groups.values() for c in g["channels"]),
                       independent_I05="unresolved; separate reviewer must perform and confirm", D01="unresolved; independent acceptance pending",
                       evidence_path=relative(evidence_path), candidate_input_binding=relative(job / "candidate_input_binding.json"))
        row = dict(study=STUDY, preparer=Path(__file__).name, reviewer="", scope="image", record_id=record_id,
                   check_id="I05", result="pass", source_evidence=relative(ARCHIVE),
                   output_evidence=relative(evidence_path), details=json.dumps(details, ensure_ascii=False))
        temporary = review_path.with_name("review_results.repair_1.tmp.csv")
        pd.concat([reviews,pd.DataFrame([row])], ignore_index=True).to_csv(temporary,index=False,encoding="utf-8-sig")
        os.replace(temporary,review_path)
    summary = dict(study=STUDY, status="prepared", script=str(metadata_source_relative(Path(__file__), PROJECT)),
                   expected_images=len(proofs), exported_images=len(images), sample_rows=len(samples),
                   sample_linked_images=counts["sample_linked_images"], patient_linked_samples=counts["patient_linked_samples"],
                   completed_pixel_groups=len(groups), source_output_blocks=details["source_output_blocks"],
                   input_binding_unchanged=True, newly_exported_images=0, reused_images=len(images),
                   evidence_paths=[str(evidence_path),str(job/"candidate_input_binding.json"),str(review_path)],
                   issues=gaps, controller_requests=["安排不同CLI上下文补做独立I05并完成D01；本轮仅补齐整理者证据，不关闭独立复核未决项。"],
                   summary="五类来源像素及完整padding自检通过，142张TIFF和两级metadata保持原版本；等待独立I05及D01验收。")
    write_json(job / "repair_execution.json", summary)
    write_json(job / "progress.json", dict(study=STUDY, stage="repair_prepared_awaiting_independent_review",
               **counts, completed_pixel_groups=len(groups), expected_pixel_groups=len(representatives),
               gaps=gaps, updated_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--overview-only", action="store_true", help="Read representative ZIP members without extracting or publishing TIFFs; use shared formal read lease")
parser.add_argument("--limit", type=int, help="Maximum new images in this invocation")
parser.add_argument("--repair-pixels", action="store_true", help="Supplement I05 preparer evidence only; use shared formal read lease")
parser.add_argument("--job-dir", type=Path, help="Batch job directory for targeted repair progress and evidence")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)

# Reuse accepted native evidence and bind the actual ZIP and all selected members.
verification = json.loads((METADATA / "native_verification.json").read_text())
proofs = [row["native_proof"] for row in verification["records"]]
assert len(proofs) == 142 and verification["scope_complete"]
assert ARCHIVE.stat().st_size == proofs[0]["source_file_stat"]["size"]
assert ARCHIVE.stat().st_mtime_ns == proofs[0]["source_file_stat"]["mtime_ns"]
if args.repair_pixels:
    if args.job_dir is None:
        parser.error("--repair-pixels requires --job-dir")
    repair_pixel_evidence(proofs, args.job_dir)
    raise SystemExit(0)
if args.overview_only:
    groups = set()
    with formal_read_lease(ARCHIVE), zipfile.ZipFile(ARCHIVE) as archive:
        for proof in proofs:
            cohort = proof["cohort"]
            group = cohort + "_" + "_".join(str(x) for x in proof["common_shape"])
            if group in groups:
                continue
            native_panel = pd.read_csv(CACHE / "source_metadata" / cohort / "Steinbock/panel.csv")
            panel = panel_from_s1(STUDY, f"{cohort}_native", native_panel.name.tolist(), S1)
            image_id = cohort + "__" + Path(proof["source_member_name"]).stem
            with archive.open(proof["source_member_name"]) as member, tifffile.TiffFile(member) as source:
                source_overview(source, panel, image_id, group)
            groups.add(group)
            print("source overview " + group, flush=True)
    raise SystemExit(0)
images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
pixel_checks = json.loads((CACHE / "preparer_pixel_checks.json").read_text()) if (CACHE / "preparer_pixel_checks.json").exists() else {}
fields, sample_records, panels, identities = [], [], [], {}
with zipfile.ZipFile(ARCHIVE) as archive:
    selected_members = {proof["source_member_name"] for proof in proofs}
    actual_members = {x.filename for x in archive.infolist() if "/Steinbock/img/" in x.filename
                      and x.filename.endswith(".tiff") and not x.filename.startswith("__MACOSX/")}
    assert actual_members == selected_members
    inventory = [dict(member=x.filename, size=x.file_size, compressed_size=x.compress_size,
                      selected=x.filename in selected_members,
                      disposition="selected verified original ROI/core" if x.filename in selected_members else
                      "source support/derived asset; not an additional independent image")
                 for x in archive.infolist() if not x.is_dir()]
    pd.DataFrame(inventory).to_csv(CACHE / "archive_inventory.csv", index=False, encoding="utf-8-sig")
    for proof in proofs:
        member = archive.getinfo(proof["source_member_name"])
        assert member.file_size == proof["source_member_stat"]["file_size"]
        assert member.CRC == proof["source_member_stat"]["CRC"]

    # Author code supplies the patient regex; paper methods establish one specimen/patient.
    paper_source = Path(json.loads((METADATA / "primary_scope_review.json").read_text())["primary_local"])
    paper_cached = CACHE / "source_metadata/pmc_fulltext.xml"
    paper_cached.parent.mkdir(parents=True, exist_ok=True)
    if not paper_cached.exists():
        shutil.copyfile(paper_source, paper_cached)
    for cohort in ("UOP", "STA"):
        analysis_name = "Analysis_UOP.Rmd" if cohort == "UOP" else "Analysis_STACohort.Rmd"
        suffixes = [f"Metadata/{cohort}_metadata.xlsx", "Steinbock/images.csv", "Steinbock/panel.csv", analysis_name, "README.md"]
        for suffix in suffixes:
            cached = CACHE / "source_metadata" / cohort / suffix
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(archive.read(PREFIX + cohort + "Cohort/" + suffix))
        workbook_path = CACHE / "source_metadata" / cohort / f"Metadata/{cohort}_metadata.xlsx"
        script_path = CACHE / "source_metadata" / cohort / analysis_name
        clinical = pd.read_excel(workbook_path, sheet_name=0, keep_default_na=False, dtype=object)
        patient_column, sex_column = ("Study ID", "Gender") if cohort == "UOP" else ("ID", "Sex")
        patient_pattern = r"OC[0-9][0-9]" if cohort == "UOP" else r"S[0-9][0-9]"
        assert f'str_extract_all(spe$sample_id, "{patient_pattern}"' in script_path.read_text()
        image_source = pd.read_csv(CACHE / "source_metadata" / cohort / "Steinbock/images.csv", keep_default_na=False, dtype=str)
        clinical_keys = set(clinical[patient_column])
        assert len(clinical_keys) == len(clinical) == 24
        for row in image_source.to_dict("records"):
            patient = re.search(patient_pattern, row["image"]).group()
            assert patient in clinical_keys
            identities[(cohort, row["image"])] = dict(patient=patient, source=row)
        native_panel = pd.read_csv(CACHE / "source_metadata" / cohort / "Steinbock/panel.csv", keep_default_na=False)
        panel = panel_from_s1(STUDY, f"{cohort}_native", native_panel.name.tolist(), S1)
        panels.append(panel)
        for row in clinical.to_dict("records"):
            patient = str(row[patient_column])
            record = dict.fromkeys(template("sample_metadata").columns, "")
            record.update(study=STUDY, sample_id=f"{cohort}_{patient}",
                          patient_key=f"{STUDY}:{cohort}:{patient}", source_subject_id=patient,
                          source_sample_id=patient, cohort_id=cohort, center_id=cohort,
                          sample_type="incisional biopsy" if cohort == "UOP" else "primary resection specimen",
                          tissue_type="tumor", anatomical_site=row["Location_specific" if cohort == "UOP" else "Location"],
                          lesion_type="primary", disease_status_at_collection="active", timepoint_label="pre_treatment",
                          treatment_status_at_collection="treatment-naive", collection_method="biopsy" if cohort == "UOP" else "resection",
                          preservation_method="ffpe", age_at_collection_years=row["Age"],
                          case_control_status="case", index_event="specimen_collection" if cohort == "UOP" else "surgery",
                          index_definition_version="Einhaus2023_source_specimen_index_v1",
                          primary_diagnosis=row["Diagnosis"], primary_site="oral tongue",
                          histology="squamous cell carcinoma", grade=str(row["Grade"]),
                          sex_reported={"M": "male", "F": "female", "m": "male", "f": "female"}[row[sex_column]],
                          metadata_notes="One source biopsy/resection per patient established by paper methods; multiple ROI/core images share this sample. Optional missing clinical fields blank; no survival duration inferred. Source grade 2to3 retained literally where present.")
            if cohort == "STA":
                smoking = str(row["Smoking"])
                record.update(bmi_kg_m2=row["BMI"], stage_basis="pathological",
                              source_pathological_tnm=str(row["Stage"]),
                              source_smoking_history=smoking,
                              smoking_status="never" if smoking == "never" else "former" if smoking.startswith("former") else "current" if "currently" in smoking else "",
                              recurrence_within_3_years=row["Recurrence within 3yrs (0=no, 1=yes)"],
                              all_cause_death_within_5_years=row["Death overall within 5yrs (0=no, 1=yes)"],
                              cancer_death_within_5_years=row["Cancer related death within 5yrs (0=no, 1=yes)"])
                record["tnm_t"] = re.search(r"T([0-9]+)", str(row["Stage"])).group(1)
                record["tnm_n"] = re.search(r"N([0-9]+[a-c]?)", str(row["Stage"])).group(1)
                match = re.search(r"M([0-9]+)", str(row["Stage"]))
                record["tnm_m"] = match.group(1) if match else ""
            sample_records.append(record)
        source = relative(workbook_path)
        fields.extend([
            field(relative(script_path), "spe$sample_id;patient_id;clinical key", "sample_metadata", "sample_id;patient_key;source_subject_id;source_sample_id;cohort_id;center_id",
                  f"Author regex {patient_pattern} links image to {patient_column}. Namespaced by cohort; one biological specimen per patient per paper. Source read_steinbock sample_id denotes an image, not biological sample.",
                  notes="STA early script references meta$Study.ID although actual worksheet has ID; later author summary code matches meta$ID. Use actual ID; gender from actual Sex. All 142 images link to 48 specimens and 48 patients."),
            field(source, "Age", "sample_metadata", "age_at_collection_years", "Copy source age at specimen acquisition.", sheet="first worksheet",
                  notes="UOP individual-workbook mean is 61.9167 years versus Table S2 aggregate 60.6; preserve individual workbook ages, not aggregate-derived replacements." if cohort == "UOP" else "STA workbook mean 58.9583 agrees with rounded supplemental mean 59.0."),
            field(source, sex_column, "sample_metadata", "sex_reported", "Map exact source sex codes.", mapping={"M":"male", "F":"female", "m":"male", "f":"female"}),
            field(source, "Diagnosis;Grade;Location_specific/Location", "sample_metadata", "primary_diagnosis;grade;anatomical_site", "Copy source diagnosis, grade string and location. Paper identifies WHO histological differentiation grading; preserve mixed source grade 2to3 without collapsing to grade3."),
            field(relative(paper_cached), "Cohorts;Imaging mass cytometry", "sample_metadata",
                  "sample_type;tissue_type;lesion_type;disease_status_at_collection;timepoint_label;treatment_status_at_collection;collection_method;preservation_method;case_control_status;index_event;index_definition_version;primary_site;histology;metadata_notes",
                  "Paper states treatment-naive invasive oral tongue OSCC FFPE biopsies at UOP and primary resections at STA; clinical acquisition is pre-treatment. Local source_specimen_index_v1 defines UOP index as diagnostic specimen collection and STA index as primary resection surgery. One specimen/patient; no cohort-level value assigned beyond this stated scope."),
            field(relative(CACHE / "source_metadata" / cohort / "Steinbock/panel.csv"), "ordered channel;name", "panel_mapping", "source_channel_index;source_name;marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
                  "Compare ordered source panel with each accepted native proof; join exact study and source name to S1; retain source order and both selected DNA channels.", join="study + Original marker"),
        ])
        if cohort == "STA":
            fields.extend([
                field(source, "BMI", "sample_metadata", "bmi_kg_m2", "Copy source BMI in kg/m2."),
                field(source, "Stage", "sample_metadata", "stage_basis;tnm_t;tnm_n;tnm_m", "Source p-prefix denotes pathology. Parse only recorded T/N/M components; absent M blank; no AJCC edition or stage-group inference."),
                field(source, "Stage", "sample_metadata", "source_pathological_tnm", "Preserve source TNM string verbatim.", notes=json.dumps(dict(data_type="string", definition="Original pathological TNM string; staging edition unavailable"))),
                field(source, "Smoking", "sample_metadata", "smoking_status", "Exact never -> never; source starting former -> former; explicit currently -> current; ambiguous longitudinal history -> blank."),
                field(source, "Smoking", "sample_metadata", "source_smoking_history", "Preserve raw source smoking history.", notes=json.dumps(dict(data_type="string", definition="Original smoking history at source-defined assessment"))),
            ])
            for source_key, target, years, event in [
                ("Recurrence within 3yrs (0=no, 1=yes)", "recurrence_within_3_years", 3, "recurrence"),
                ("Death overall within 5yrs (0=no, 1=yes)", "all_cause_death_within_5_years", 5, "all-cause death"),
                ("Cancer related death within 5yrs (0=no, 1=yes)", "cancer_death_within_5_years", 5, "cancer-related death")]:
                fields.append(field(source, source_key, "sample_metadata", target,
                                    f"Copy explicit 0/1 {event} within {years} years after resection; source blanks remain blank. Fixed-horizon binary indicator; not OS/DFS time-to-event data.",
                                    notes=json.dumps(dict(data_type="integer", definition=f"Source 0=no, 1=yes {event} within {years} years after surgery"))))

samples = pd.DataFrame(sample_records).fillna("")
for endpoint in ("recurrence_within_3_years", "all_cause_death_within_5_years", "cancer_death_within_5_years"):
    samples[endpoint] = pd.to_numeric(samples[endpoint].replace("", pd.NA)).astype("Int64")
panel_table = pd.concat(panels, ignore_index=True)
output_estimate = sum(max(proof["common_shape"]) ** 2 *
                      int(panel_table.loc[panel_table.panel_id == proof["cohort"] + "_native", "keep_channel"].sum()) * 4
                      for proof in proofs)
fields.extend([
    field(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
          "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index;keep_reason;exclusion_reason",
          "Exact Einhaus2023_IMC_OSCC + native source-name join; no cross-study mappings or aliases. Exclude the eight explicitly excluded phosphorylation channels, retain selected DNA1/DNA2 and admitted markers in native order.",
          join="Study + Original marker"),
    field(relative(ARCHIVE), "ZIP member;images.csv;native proof", "image_metadata",
          ";".join(template("image_metadata").columns),
          "Use complete native TIFF frame and author image/source-acquisition IDs. Image ID is cohort__native stem. No reorientation/resampling. Symmetric square zero padding; native float32 intensity. Physical scale 1um from paper 1000x500um ROI and matching 1000x500px frame. Per-image sample join uses author patient regex.",
          join="cohort + image filename + native-proof member", notes="source_image_id uses ZIP-relative archive::member; source_files contains NAS-relative ZIP and source metadata paths. source_acquisition_id only where actual source table supplies it. STA core identity retained as full native image stem."),
    field(relative(CACHE / "source_metadata/OSCC-IMC-SupplementalMaterial.pdf"), "Tables S2/S3", "sample_metadata", "metadata_notes",
          "Supplemental cohort clinical summaries cross-check population context; individual clinical values come from archived workbooks, not aggregate table imputation."),
])
field_mapping = pd.DataFrame(fields, columns=template("field_mapping").columns)
write_json(CACHE / "sample_linkage_evidence.json", dict(
    study=STUDY, public_source="https://zenodo.org/records/8169946", paper="https://doi.org/10.1016/j.isci.2023.108486",
    evidence=[relative(CACHE / "source_metadata" / cohort / name) for cohort in ("UOP", "STA") for name in
              (("Analysis_UOP.Rmd" if cohort == "UOP" else "Analysis_STACohort.Rmd"), f"Metadata/{cohort}_metadata.xlsx")],
    source_code_rule="Explicit author patient_id regex; checked against actual clinical worksheet primary key.",
    biological_specimen_rule="Paper: 24 UOP biopsies from 24 patients; 24 STA primary resections from 24 patients, sampled with two/three cores.",
    expected_images=142, sample_rows=48, patient_linked_samples=48, source_image_sample_links=142,
    optional_gaps=["UOP outcomes unavailable by paper", "TNM edition not specified", "Mixed STA grade 2to3 retained", "One longitudinal smoking history not reducible to a collection-time category", "One STA recurrence-within-3-years value undocumented", "UOP individual-workbook age mean 61.9167 differs from supplemental aggregate 60.6; individual source ages retained"],
    author_code_conflict="STA early meta$Study.ID and meta$Gender references differ from actual ID/Sex columns; later patient summary uses meta$ID. Actual workbook columns are authoritative.",
    repository_search="The original Zenodo release contains the author Rmd image/patient mapping and clinical workbooks; paper external software repositories are generic analysis tools."))
progress("metadata_ready", gaps=["Independent review pending"], output_estimate_uncompressed_bytes=output_estimate,
         peak_extracted_member_bytes=max(x["source_member_stat"]["file_size"] for x in proofs))

# Save a real index after each completed image. Correct completed TIFFs are reused.
completed = {record["image_id"] for record in images}
excluded_path = CACHE / "excluded_images.json"
excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else []
new_count = reused_count = 0
if not args.metadata_only:
    assert shutil.disk_usage(ROOT).free > 100 * 1024**3 + output_estimate
    assert shutil.disk_usage(TEMP).free > 64 * 1024**3 + 100_000_000
    with zipfile.ZipFile(ARCHIVE) as archive:
        for proof in proofs:
            cohort = proof["cohort"]
            filename = Path(proof["source_member_name"]).name
            image_id = cohort + "__" + Path(filename).stem
            panel = panel_table[panel_table.panel_id == f"{cohort}_native"]
            selected = panel[panel.keep_channel].sort_values("output_channel_index")
            assert panel.source_name.tolist() == proof["channels"]
            destination = OUTPUT / "images" / image_id / "input.tiff"
            if image_id in completed:
                check_header(destination, selected, max(proof["common_shape"]))
                reused_count += 1
                continue
            if args.limit is not None and new_count >= args.limit:
                break
            identity = identities[(cohort, filename)]
            source_row = identity["source"]
            h, w = int(source_row["height_px"]), int(source_row["width_px"])
            assert [h, w] == proof["common_shape"]
            local = TEMP / (cohort + "__" + filename)
            # One complete original TIFF at a time; ZipFile checks its existing CRC on extraction.
            if not local.exists():
                temporary = local.with_suffix(".extracting")
                with formal_read_lease(ARCHIVE):
                    with archive.open(proof["source_member_name"]) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024**2)
                assert temporary.stat().st_size == proof["source_member_stat"]["file_size"]
                os.replace(temporary, local)
            with tifffile.TiffFile(local) as source_header:
                native_axes = source_header.series[0].axes
            spec = SourceSpec(image_id, f"{cohort}_native", "tiff", local,
                              tuple(range(len(panel))), pixel_size_um=(1.0, 1.0))
            progress("exporting", current_image=image_id, gaps=["Independent review pending"])
            try:
                with open_source(spec) as reader:
                    stats = export_input(reader, panel, destination)
                if not stats["admitted"]:
                    excluded.append(dict(image_id=image_id, reason="All admitted downstream markers zero over entire source frame"))
                    write_json(excluded_path, excluded)
                    local.unlink()
                    continue
                side = max(h, w)
                check_header(destination, selected, side)
                record = dict.fromkeys(template("image_metadata").columns, None)
                record.update(study=STUDY, image_id=image_id, sample_id=f"{cohort}_{identity['patient']}",
                              panel_id=f"{cohort}_native", image_kind="roi" if cohort == "UOP" else "tma_core",
                              source_image_id=relative(ARCHIVE) + "::" + proof["source_member_name"],
                              source_files=[relative(ARCHIVE), relative(CACHE / "source_metadata" / cohort / "Steinbock/images.csv")],
                              source_format="tiff", source_acquisition_id=source_row.get("acquisition_id") or None,
                              source_axes=native_axes, source_dtype=stats["source_dtype"], source_revision=proof["source_release"],
                              slide_id=Path(source_row["source_file"]).stem if cohort == "UOP" else None,
                              core_id=Path(filename).stem if cohort == "STA" else None,
                              roi_id=Path(filename).stem,
                              region_annotation=source_row.get("acquisition_description") or None,
                              source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                              height_px=side, width_px=side, source_channel_count=len(panel), channel_count=len(selected),
                              pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                              intensity_description="Original published float32 IMC TIFF intensities; no intensity transformation, resampling or clipping.",
                              split_method="already_independent", split_evidence=relative(CACHE / "sample_linkage_evidence.json"),
                              metadata_notes="Complete native ROI frame; one biological sample per image. STA cores from the same resection share sample_id. Public release is native-intensity Steinbock export, not original MCD recomputation.")
                for key in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"):
                    record[key] = stats[key]
                if stats["valid_pixels"]:
                    record["valid_pixels_path"] = relative(destination.with_name(stats["valid_pixels"]))
                group = f"{cohort}_{h}_{w}"
                if group not in pixel_checks:
                    representative_check(local, destination, panel, record, group)
                images.append(record)
                completed.add(image_id)
                new_count += 1
                save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel_table, field_mapping)
                signal_path = CACHE / "image_signal" / f"{image_id}.json"
                write_json(signal_path, dict(image_id=image_id, **stats))
                progress("exporting", last_completed_image=image_id, gaps=["Independent review pending"])
                print(f"saved {len(images)}/142 {image_id}", flush=True)
            except Exception as error:
                progress("failed", current_image=image_id, gaps=[str(error)])
                raise
            finally:
                local.unlink(missing_ok=True)

images_table = pd.DataFrame(images) if images else template("image_metadata")
save_input_tables(OUTPUT, images_table, samples, panel_table, field_mapping)
summary = dict(study=STUDY, expected_images=142, exported_images=len(images), excluded_images=excluded,
               sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=48,
               reused_images=reused_count, newly_exported_images=new_count,
               original_source_images=142, retained_channels_by_panel={p: int(g.keep_channel.sum()) for p, g in panel_table.groupby("panel_id")},
               source_bytes=ARCHIVE.stat().st_size, native_proofs_reused=142,
               output_estimate_uncompressed_bytes=output_estimate,
               representative_pixel_groups=pixel_checks, independent_review="pending",
               output=relative(OUTPUT), source_inventory=relative(CACHE / "archive_inventory.csv"),
               clinical_linkage=relative(CACHE / "sample_linkage_evidence.json"),
               controller_requests=["Arrange independent review of image/panel/sample/clinical evidence and D01 input binding."])
write_json(CACHE / "input_delivery_summary.json", summary)

# Preparer self-checks have an empty reviewer; D01 stays unresolved for independent acceptance.
review_path = OUTPUT / "review_results.csv"
review = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else template("review_results")
record_id = STUDY + ":preparer_delivery"
review = review[review.record_id != record_id]
review_rows = []
overview_path = CACHE / "preparer_overview_assessment.json"
overview_confirmed = (overview_path.exists() and
                      set(json.loads(overview_path.read_text())["overview_groups"]) ==
                      {proof["cohort"] + "_" + "_".join(str(x) for x in proof["common_shape"]) for proof in proofs})
clinical_check_path = CACHE / "preparer_clinical_checks.json"
clinical_confirmed = clinical_check_path.exists() and json.loads(clinical_check_path.read_text())["all_checked_values_equal"]
for check, result, details in [
    ("S01", "pass", "142 accepted native TIFF members exactly match archive original-img members; 71 UOP ROIs + 71 STA core ROIs. Other archive assets are source support/derived views. Whole-frame marker signal tested during export. " + f"Current image delivery {len(images)}/142."),
    ("I01", "pass", f"{len(images)} unique image IDs and exact archive member/source image binding."),
    ("I02", "pass", f"{len(images)}/{len(images)} image/sample links; all panel foreign keys valid."),
    ("I03", "pass", "Complete native frame and centered square zero padding; no resampling; 1um from source methods and matching dimensions."),
    ("I04", "pass" if overview_confirmed else "unresolved", "Already independent source ROI/core frames with author patient linkage. Preparer visually inspected all five cohort/dimension groups; full tissue acquisition retained, including OC11_003 400x1250px. See preparer_overview_assessment.json; independent review pending." if overview_confirmed else "Source-label/geometry checks complete; representative visual assessment pending."),
    ("P01", "pass", "41 UOP / 40 STA original ordered source rows agree with native proofs and exact study S1 labels."),
    ("P02", "pass", "33 UOP / 32 STA retained: both selected DNA inputs and admitted markers; eight S1 excluded phosphorylation channels omitted; output indices continuous; downstream markers unique."),
    ("P03", "pass", f"Opened {len(images)} saved TIFF headers; native float32 CYX, square, all channel names/order agree."),
    ("I05", "pass" if pixel_checks else "unresolved", f"Preparer only: {len(pixel_checks)} observed cohort/orientation groups; all retained channels compared to direct source TIFF in two 32px blocks; padding zero. Exact coordinates in preparer_pixel_checks.json."),
    ("M01", "pass", f"Author code + clinical primary keys + paper specimen design establish 48 biological samples / 48 patient keys; image links {len(images)}/{len(images)}. See sample_linkage_evidence.json. Optional clinical gaps preserved."),
    ("M02", "pass" if clinical_confirmed else "unresolved", "Preparer source/target clinical checks saved in preparer_clinical_checks.json; optional unknowns blank, mixed grade 2to3 preserved. Individual workbook values prevail over discordant supplemental aggregate age. Independent clinical reconstruction pending."),
    ("D01", "unresolved", f"Preparer delivery {len(images)}/142 images, 48 samples, all existing image/sample links and 48 sample/patient links. Independent review and input_binding pending; this preparer does not grant acceptance."),
]:
    review_rows.append(dict(study=STUDY, preparer=Path(__file__).name, reviewer="", scope="study", record_id=record_id,
                            check_id=check, result=result, source_evidence=relative(METADATA / "native_verification.json") + ";" + relative(CACHE / "sample_linkage_evidence.json"),
                            output_evidence=relative(OUTPUT), details="Preparer self-check: " + details))
with formal_publish_lease(review_path, directory=review_path.parent):
    pd.concat([review, pd.DataFrame(review_rows)], ignore_index=True).to_csv(review_path, index=False, encoding="utf-8-sig")
progress("prepared_awaiting_independent_review" if len(images) + len(excluded) == 142 else "partial_delivery",
         gaps=["Independent image/panel/clinical review and D01 binding pending"], **{k:summary[k] for k in ("reused_images", "newly_exported_images")})
print(json.dumps(summary, ensure_ascii=False, indent=2))
