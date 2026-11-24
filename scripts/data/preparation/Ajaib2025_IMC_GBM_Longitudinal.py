"""Deliver Ajaib 2025 GBM native ROI intensities and biological-sample metadata.

Inputs: accepted native_verification, original Zenodo data.zip, study-specific S1,
author analysis code, and the paper's supplementary Tables 1--3 / Figure 1.
Outputs: independent square CYX input.tiff, 39-field images.parquet, panel, sample,
field_mapping, and preparer self-checks. Run exports through batch shared formal write lease.
Source-only display previews are cached; stored TIFF intensities are unchanged.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import requests
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

PROJECT = Path(__file__).resolve().parents[3]
STUDY = "Ajaib2025_IMC_GBM_Longitudinal"
PANEL = STUDY + "__native34"
PREPARER = "codex_cli_0010_prepare_0"
CONTROLLER_REQUESTS = ["共享 read_tiff_window 在 OME series 解析后遇到 TiffFrame，访问 samplesperpixel 失败；本 study 用公开 aspage() 转换 IFD 后复用原窗口读取。请主控将兼容性修复迁入 src/cellweft/data/io/readers.py，并安排独立复核。"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--repair-field-mapping", action="store_true",
                    help="Publish only the corrected field mapping and a preparer repair record")
parser.add_argument("--progress-path", type=Path, help="Progress file for the current batch task")
parser.add_argument("--image", action="append", help="Restrict export to specified source ROI IDs")
args = parser.parse_args()
root = args.data_root
metadata = metadata_source_directory(root, STUDY)
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
source_metadata = metadata / "source_metadata"
output = root / "processed" / STUDY
progress_path = args.progress_path or Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0010_prepare_0/progress.json"
archive_path = root / "raw/public" / STUDY / "public_original/data.zip"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
templates = PROJECT / "docs/data/templates/study_input"
for directory in (cache, output, progress_path.parent):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def scalar_tiff(tif: tifffile.TiffFile) -> SimpleNamespace:
    """Study-local TiffFrame adaptation for the shared scalar window reader.

    OME series access materializes TiffFrame objects without samplesperpixel.
    Convert their IFD metadata through public aspage(), preserving each frame's
    offsets. The controller should migrate this compatibility fix to readers.py.
    """
    return SimpleNamespace(filehandle=tif.filehandle, pages=[
        page.aspage() if isinstance(page, tifffile.TiffFrame) else page for page in tif.pages])


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(progress_path, dict(study=STUDY, stage=stage, completed_images=completed,
                                  expected_images=len(proofs), updated_at=datetime.now(timezone.utc).isoformat(),
                                  controller_requests=CONTROLLER_REQUESTS,
                                  **details))


write_json(cache / "controller_requests.json", CONTROLLER_REQUESTS)


# Reuse accepted native proofs, bound by source stat and ZIP central-directory identity.
verification = json.loads((metadata / "native_verification.json").read_text())
proofs = [record["native_proof"] for record in verification["records"]]
assert verification["scope_complete"] and len(proofs) == 34
assert all(p["complete"] and p["counted"] and p["native_order_verified"] for p in proofs)
archive_stat = archive_path.stat()
for proof in proofs:
    assert archive_stat.st_size == proof["source_file_stat"]["size"]
    assert archive_stat.st_mtime_ns == proof["source_file_stat"]["mtime_ns"]
with zipfile.ZipFile(archive_path) as archive:
    inventory = []
    accepted = {p["source_member_name"] for p in proofs}
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        category = "accepted_raw_roi" if entry.filename in accepted else (
            "source_metadata" if entry.filename in {"data/marker_panel_info.csv", "data/raw_image_info.csv"}
            else "alternate_processed_view_or_auxiliary")
        inventory.append(dict(member=entry.filename, bytes=entry.file_size, category=category))
    assert {i.filename for i in archive.infolist() if i.filename.startswith("data/raw_fullstack_tiff_images/") and i.filename.endswith(".tiff")} == accepted
    for proof in proofs:
        entry = archive.getinfo(proof["source_member_name"])
        for name in ("file_size", "compress_size", "CRC", "header_offset", "flag_bits"):
            assert getattr(entry, name) == proof["source_member_stat"][name]
    for member in ("data/marker_panel_info.csv", "data/raw_image_info.csv"):
        destination = source_metadata / Path(member).name
        if not destination.exists():
            destination.write_bytes(archive.read(member))
write_json(cache / "archive_inventory.json", inventory)
progress("sample_and_panel_metadata", 0, gaps=["独立复核待主控安排"])

# Public metadata: preserve author code and the complete small supplementary attachment.
author_base = "https://raw.githubusercontent.com/GliomaGenomics/GBM_IMC_Analysis/main/"
author_files = ["README.md", "inst/region_mapping.xlsx", "inst/IMC_cell_type_markers.yml",
                "process/Downstream/1_Read_in_Data.R", "process/Downstream/4_QC_and_Batch_Correction.R",
                "process/Downstream/5c_Expression_Ranks_Gating_Logic.R"]
for filename in author_files:
    destination = source_metadata / "author_repository" / filename
    if not destination.exists():
        response = requests.get(author_base + filename, timeout=45)
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
supp_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12916744/supplementaryFiles"
supp_path = source_metadata / "supplementary.pdf"
if not supp_path.exists():
    import io
    response = requests.get(supp_url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as supplementary:
        member = "noaf190_supplementary_figures_1-5_tables_1-12.pdf"
        supp_path.write_bytes(supplementary.read(member))
        (source_metadata / "supplementary.url").write_text(supp_url + "::" + member)
if not (source_metadata / "supplementary.txt").exists():
    subprocess.run(["pdftotext", "-layout", str(supp_path), str(source_metadata / "supplementary.txt")], check=True)
paper_path = source_metadata / "paper.xml"
if not paper_path.exists():
    response = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12916744/fullTextXML", timeout=45)
    response.raise_for_status()
    paper_path.write_bytes(response.content)
text = (source_metadata / "supplementary.txt").read_text()
author_code = (source_metadata / "author_repository/process/Downstream/1_Read_in_Data.R").read_text()
assert 'pattern = "^\\\\d{2}"' in author_code and 'pattern = "(?i)prim|rec"' in author_code

# Extract clinical values from the actual source table; no patient list is embedded here.
clinical_section = text.split("Supplementary Table 1.", 1)[1].split("Supplementary Figure 1.", 1)[0]
clinical_rows = re.findall(r"^\s*(\d+)\s+(Male|Female)\s+(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(up|down)\s*$", clinical_section, re.M)
clinical = pd.DataFrame(clinical_rows, columns=["patient", "sex", "primary_tumour_location", "diagnosis_age", "pfs_months", "os_months", "status", "responder_type"])
assert len(clinical) == 5 and clinical.patient.is_unique
clinical.to_csv(cache / "supplementary_table_1.csv", index=False, encoding="utf-8-sig")
roi_section = text.split("Supplementary Table 2.", 1)[1].split("Supplementary Figure 2.", 1)[0]
roi_rows = re.findall(r"^\s*(\d+(?:Prim|Rec)_\d{3})\s+(\d+)\s+(primary|recurrent)\s+(\d{3})\s+(\w+)\s*$", roi_section, re.M)
roi_table = pd.DataFrame(roi_rows, columns=["image_id", "patient", "surgery", "roi", "annotation"])
assert len(roi_table) == 30 and roi_table.image_id.is_unique
roi_table.to_csv(cache / "supplementary_table_2.csv", index=False, encoding="utf-8-sig")
raw = pd.read_csv(source_metadata / "raw_image_info.csv", dtype=str, keep_default_na=False)
raw["image_id"] = raw.image.str.removesuffix(".tiff")
matched = roi_table.merge(raw, on="image_id", validate="one_to_one")
assert len(matched) == 30
assert (matched.annotation == matched.region_type).all()
# A source MCD is one patient/surgery biological specimen, demonstrated by Table 2.
specimen_map = matched[["source_file", "patient", "surgery"]].drop_duplicates()
assert len(specimen_map) == 10 and specimen_map.source_file.is_unique
source_rows = raw.merge(specimen_map, on="source_file", validate="many_to_one")
source_rows["sample_id"] = source_rows.source_file.str.removesuffix(".mcd")
source_rows = source_rows.set_index("image_id", verify_integrity=True)
clinical = clinical.set_index("patient", verify_integrity=True)
sample_columns = pd.read_csv(templates / "sample_metadata.csv", nrows=0).columns.tolist()
samples = []
for specimen in specimen_map.itertuples(index=False):
    c = clinical.loc[specimen.patient]
    sample_id = Path(specimen.source_file).stem
    primary = specimen.surgery == "primary"
    row = dict.fromkeys(sample_columns, None)
    row.update(study=STUDY, sample_id=sample_id, source_sample_id=sample_id,
               patient_key=STUDY + "__" + specimen.patient, source_subject_id=specimen.patient,
               cohort_id=STUDY, biospecimen_group_id=sample_id, sample_type="surgical tissue",
               tissue_type="tumor", anatomical_site="brain", lesion_type="primary" if primary else "recurrence",
               timepoint_label="pre_treatment" if primary else "recurrence",
               treatment_status_at_collection="pre-treatment" if primary else "post-radiotherapy and temozolomide",
               collection_method="resection", preservation_method="ffpe",
               sex_reported=c.sex.lower(), age_at_diagnosis_years=int(c.diagnosis_age),
               primary_diagnosis="IDH-wildtype glioblastoma", primary_site="brain; " + c.primary_tumour_location,
               molecular_subtype="IDH-wildtype", case_control_status="case",
               vital_status="deceased" if float(c.status) == 1 else "alive",
               os_time=float(c.os_months), os_time_unit="months", os_origin="primary surgery",
               os_event=int(float(c.status)), os_event_definition="death; Supplementary Table 1 status 1=deceased, 0=alive",
               metadata_notes="Table 1 diagnosis age is patient-level; collection age unknown. OS origin is primary surgery per Supplementary Figure 1 (visually checked). PFS values retained in cached source Table 1; formal PFS event/assessment definition absent, standardized PFS fields left empty. Brain lobe in primary_site describes the primary tumour, not independently measured recurrence location. Source responder_type is not a clinical response category. Matched primary/recurrent samples per paper Methods and Supplementary Figure 2.")
    samples.append(row)
samples = pd.DataFrame(samples, columns=sample_columns).sort_values("sample_id")

# Native panel order is author channel_number 1..34; S1 matches only this study.
source_panel = pd.read_csv(source_metadata / "marker_panel_info.csv", keep_default_na=False).sort_values("channel_number")
source_names = source_panel.name.tolist()
assert len(source_names) == 34
assert all(p["channels"] == source_names for p in proofs)
assert "validated (by KO)" in text and "JARID2_N_Terminus" in text
# Select the KO-validated N-terminal antibody (Table 3, Abcam ab251123), rather
# than the in-house C-terminal functional measurement. Both source rows survive
# in panel_mapping; S1 identity is retained. This is a source-based adapter choice.
downstream_indices = set(range(34)) - {0}
panel = panel_from_s1(STUDY, PANEL, source_names, s1_path, downstream_source_indices=downstream_indices)
panel.loc[panel.source_channel_index == 0, "exclusion_reason"] = "Duplicate JARID2: select N-terminus antibody ab251123 explicitly validated by KO in Supplementary Table 3; C-terminus is a distinct in-house activating-form measurement."
panel.loc[panel.source_channel_index == 29, "keep_reason"] = "Unique downstream JARID2 measurement: N-terminal ab251123 has explicit KO validation (Supplementary Table 3)."
assert int(panel.keep_channel.sum()) == 33
assert panel.loc[panel.source_name.isin(["DNA1", "DNA2"]), "keep_channel"].all()

# Document each source interpretation once, including actual identity coverage.
field_columns = pd.read_csv(templates / "field_mapping.csv", nrows=0).columns.tolist()
fields = []
def field(source_file: str, source_table: str, source_field: str, target: str, target_field: str,
          keys: str, rule: str, notes: str = "", value_mapping: dict | None = None) -> None:
    row = dict.fromkeys(field_columns, None)
    row.update(study=STUDY, source_file=source_file, source_sheet_or_table=source_table,
               source_field=source_field, target_table=target, target_field=target_field,
               join_keys=keys, transformation_rule=rule, notes=notes,
               value_mapping=json.dumps(value_mapping, ensure_ascii=False) if value_mapping else None,
               conflict_rule="Keep source values; do not impute absent clinical fields.")
    fields.append(row)
field(relative(metadata / "native_verification.json"), "records[].native_proof", "source_file_rel;source_member_name;physical_identity;source_metadata;common_shape", "image_metadata", "study;image_id;image_kind;source_image_id;source_files;source_format;source_acquisition_id;source_axes;source_dtype;source_revision;roi_id;source_x0_px;source_y0_px;source_x1_px;source_y1_px;source_channel_count", "ZIP member + source MCD acquisition", "Reuse accepted 34 native proofs; confirm size/mtime and ZIP member identity, no rehash. Each accepted raw_fullstack TIFF is one complete source MCD acquisition ROI; assign image_kind=roi. Publish complete source ROI without resizing or flipping.")
field(relative(source_metadata / "raw_image_info.csv"), "raw_image_info", "image;source_file;region_type;acquisition_width_um;acquisition_height_um;width_px;height_px", "image_metadata", "sample_id;region_annotation;pixel_size_x_um;pixel_size_y_um;split_method;split_evidence;metadata_notes", "image=ROI+.tiff; source_file=sample_id+.mcd", "Preserve raw region_type, including literal unknown. Pixel scale is acquisition physical width/height divided by native width/height (1 micrometre). Full independent ROI, centered square padding.", "Four additional 67Prim ROI 004--007 share the exact source MCD specimen already explicitly assigned by Table 2; Supplementary Figure 3 (PDF p.6) also explicitly labels all four and colors them by patient. Four energy-test metadata rows 008--011 have no TIFF and are held; not images.")
field(relative(source_metadata / "supplementary.pdf"), "Supplementary Table 2 + Figure 2 (PDF pp.3--4)", "image id;patient;surgery;ROI", "sample_metadata", "study;sample_id;source_sample_id;patient_key;source_subject_id;cohort_id;biospecimen_group_id;sample_type;lesion_type;timepoint_label;metadata_notes", "image id -> raw_image_info.image -> source_file -> biological specimen; Table 2 patient", "Collapse ROI rows by source_file + patient + surgery into ten biological samples; five explicit patient identities. The four additional raw TIFFs map via their exact same source_file and author identity rules.", "34/34 expected image-to-sample; 10/10 sample-to-patient, five patients. Same specimen ROIs share sample_id. Author 1_Read_in_Data.R lines 69--79 independently defines patient/surgery/ROI parsing; region_mapping.xlsx is ROI annotation only and has one blank key. Do not treat its sample_id column as biological sampling identity.")
field(relative(source_metadata / "author_repository/process/Downstream/1_Read_in_Data.R"), "ADD SINGLE CELL METADATA", "sample_id -> patient,surgery,ROI", "image_metadata", "sample_id;roi_id", "author ROI ID and raw source_file", "Use author's explicit patient and surgery semantics, validated against all 30 Supplementary Table 2 assignments; extra ROIs retain source-file specimen linkage.", "Original GitHub code downloaded from the Zenodo-linked author repository; identifiers remain source identifiers, without embedded patient lists in code.")
field(relative(source_metadata / "supplementary.pdf"), "Table 1 and Figure 1 (PDF p.2)", "patient;sex;primary tumour location;diagnosis age;OS (months);status", "sample_metadata", "sex_reported;primary_site;age_at_diagnosis_years;vital_status;os_time;os_time_unit;os_origin;os_event;os_event_definition", "patient = source_subject_id", "Patient clinical data replicated to paired samples. Sex lowercase; diagnosis age retained at diagnosis. OS months unchanged; Figure 1 places primary surgery at zero and death at OS. Status 1 -> deceased / death event.", "Figure 1 visually checked. PFS origin/criteria and responder_type lack a sufficiently explicit clinical event/response definition; source Table 1 is retained in cache and corresponding standard fields remain empty.", {"Male":"male", "Female":"female", "1":"deceased", "0":"alive"})
field(relative(paper_path), "Methods; Results; Ethics; Supplementary Methods", "paired pre/post-treatment IDH-wildtype GBM; resection;5um FFPE;RT/TMZ", "sample_metadata", "tissue_type;anatomical_site;collection_method;preservation_method;primary_diagnosis;molecular_subtype;case_control_status;treatment_status_at_collection", "study cohort + primary/recurrent surgery", "Cohort is paired primary/recurrent glioblastoma surgical tissue; FFPE. Primary before treatment, recurrence following radiotherapy and temozolomide. All samples confirmed tumour.")
field(relative(cache / "image_receipts"), "per-image export receipts", "native source shape;selected panel;export result;source/output pixel comparison", "image_metadata", "pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;channel_count;input_image_path;valid_pixels_path;intensity_description", "image_id", "Shared export_input preserves decoded dtype and all ROI pixels; center-pad to square, write lossless OME BigTIFF in output channel order. Full-image downstream nonzero counts establish admission. Source and saved 32x32 interior/border blocks are compared for every kept channel.")
field(relative(source_metadata / "marker_panel_info.csv"), "marker_panel_info", "channel_number;name", "panel_mapping", "panel_id;source_channel_index;source_name", "channel_number - 1", "Preserve all 34 source rows in author-published native order; bind to accepted native header proof.")
field(relative(source_metadata / "marker_panel_info.csv"), "marker_panel_info + records[].native_proof.channels", "channel_number;name;native_proof.channels", "image_metadata", "panel_id", "ordered native_proof.channels = marker_panel_info.name sorted by channel_number", "All 34 accepted ROI channel lists exactly equal the same ordered 34-channel source panel. Assign panel_id=" + PANEL + " to every image; this is the foreign key to panel_mapping.panel_id. Channel order or membership changes require a different panel_id.", "Native channel lists are preserved in " + relative(metadata / "native_verification.json") + ".")
field(relative(s1_path), "Supplementary Table S1", "Study + Original marker", "panel_mapping", "s1_original_marker;marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index;keep_reason;exclusion_reason", "exact study + source marker name", "Match S1 Study=" + STUDY + " and Original marker=source_name exactly; set s1_original_marker to that matched Original marker, without aliases. Retain S1-admitted markers and both DNA nuclear channels. Choose N-terminal JARID2 by explicit KO validation in source Table 3; exclude C-terminal duplicate from physical output. Output indices follow retained source order.", "33 output channels, 31 downstream unique markers, 2 nuclear-only channels. S1 is unchanged; duplicate antibody functions are distinct in source and this selection does not claim equivalence of their biological activity.")
field(relative(source_metadata / "supplementary.pdf"), "Supplementary Table 3 (PDF p.5)", "JARID2 antibody validation", "panel_mapping", "keep_channel;keep_reason;exclusion_reason", "JARID2_N_Terminus vs JARID2_C_Terminus", "N-terminal ab251123 is explicitly KO validated; C-terminal is developed in house. Use N-terminal only for downstream JARID2.")
field(relative(cache / "public_metadata_sources.json"), "source search receipt", "Zenodo;paper;Supplementary Tables 1--3;author repository tree;region map;author identity code", "sample_metadata", "metadata_notes", "study + source-file specimen + explicit patient", "Sources actually read and downloaded; optional absent clinical fields stay empty. Source-side region map corrections in an unlabeled column do not overwrite raw_image_info annotation.", "Ten biological samples, five patients; complete image/sample/patient identity coverage. No unverified inference from ROI suffixes.")
for mapping in fields:
    if "os_time" in mapping["target_field"].split(";"):
        mapping["time_origin_rule"] = "OS months start at primary surgery (Supplementary Figure 1), not at recurrent sample collection; event is death."
field_mapping = pd.DataFrame(fields, columns=field_columns)

# Repair review FM01--FM03 without republishing independently verified tables/TIFFs.
if args.repair_field_mapping:
    images = pd.read_parquet(output / "images.parquet")
    saved_panel = pd.read_csv(output / "panel_mapping.csv", keep_default_na=False)
    assert len(images) == len(proofs) and images.image_kind.eq("roi").all()
    assert images.panel_id.eq(PANEL).all() and set(images.panel_id) == set(saved_panel.panel_id)
    assert saved_panel.source_name.tolist() == source_names
    assert saved_panel.s1_original_marker.tolist() == panel.s1_original_marker.tolist() == source_names
    repairs = [{"issue_id": issue, "target_table": table, "target_field": name}
               for issue, table, name in [("FM01", "image_metadata", "image_kind"),
                                          ("FM02", "image_metadata", "panel_id"),
                                          ("FM03", "panel_mapping", "s1_original_marker")]]
    details = dict(self_check=True, repair_round=1, repairs=repairs,
                   action="Regenerated source rules only; existing image/sample/panel tables and TIFFs reused.",
                   prior_review=relative(progress_path.parent.parent / "0138_review_0/report.json"),
                   independent_recheck="pending", input_binding_written=False)
    review_path = output / "review_results.csv"
    with nullcontext():
        temporary = output / "field_mapping.csv.tmp"
        with formal_publish_lease(output / "field_mapping.csv", directory=output):
            field_mapping.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, output / "field_mapping.csv")
        reviews = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        repair = dict(study=STUDY, preparer="codex_cli_0142_repair_1", reviewer="", scope="study",
                      record_id=STUDY, check_id="D01", result="unresolved",
                      source_evidence=details["prior_review"], output_evidence=relative(output / "field_mapping.csv"),
                      details=json.dumps(details, ensure_ascii=False))
        review_temporary = review_path.with_suffix(".csv.tmp")
        with formal_publish_lease(review_path, directory=review_path.parent):
            pd.concat([reviews, pd.DataFrame([repair])], ignore_index=True).to_csv(
                review_temporary, index=False, encoding="utf-8-sig")
            os.replace(review_temporary, review_path)
    progress("field_mapping_repaired_pending_independent_review", len(images),
             sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()), repairs=repairs,
             gaps=["FM01--FM03 已修补，待不同 CLI 上下文独立复查及主控更新 D01。",
                   "PFS 正式事件与评估定义缺失，标准字段按来源说明留空。"])
    print(json.dumps(dict(study=STUDY, repaired_fields=repairs, reused_images=len(images)), ensure_ascii=False), flush=True)
    raise SystemExit(0)

write_json(cache / "public_metadata_sources.json", dict(
    zenodo="https://zenodo.org/records/14679443", paper="https://pmc.ncbi.nlm.nih.gov/articles/PMC12916744/",
    supplementary_url=(source_metadata / "supplementary.url").read_text(), author_repository="https://github.com/GliomaGenomics/GBM_IMC_Analysis",
    author_files=[dict(url=author_base + name, path=relative(source_metadata / "author_repository" / name)) for name in author_files],
    connection=dict(expected_images=34, biological_samples=10, patients=5, explicit_table2_rois=30,
                    additional_rois="4 linked through exact same source_file, confirmed author identity code, and patient-colored Supplementary Figure 3 labels"),
    clinical_limitations=["PFS source value retained but formal assessment/event definition absent; standard PFS fields empty", "Collection age, exact recurrence lobe, detailed regimen/dates unavailable"],
    retrieval_notes=["Direct PMC supplementary endpoint returned HTML; Europe PMC supplementaryFiles returned the original PDF", "Primary Figure 1 visually inspected; primary surgery is OS time origin"],
    panel_selection="JARID2 N-terminal ab251123 KO validation in Table 3; C-terminal duplicate not retained"))

# Merge existing image rows; completed image receipts bind reuse to panel and native source.
image_columns = pd.read_csv(templates / "image_metadata.csv", nrows=0).columns.tolist()
images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists() else pd.DataFrame(columns=image_columns)
receipts_dir = cache / "image_receipts"
receipts_dir.mkdir(exist_ok=True)
panel_binding = panel[["source_channel_index", "source_name", "marker_id", "role", "keep_channel", "output_channel_index"]].to_json(orient="records")
selected = [p for p in proofs if not args.image or Path(p["source_member_name"]).name.removesuffix("_fullstack.tiff") in args.image]
if args.image and len(selected) != len(set(args.image)):
    raise ValueError("Unknown selected ROI ID")
write_json(cache / "capacity_estimate.json", dict(extracted_tiff_bytes=sum(p["source_member_stat"]["file_size"] for p in proofs),
    output_uncompressed_bytes=sum(33 * max(p["common_shape"]) ** 2 * 4 for p in proofs),
    reserve_nas_bytes=100 * 1024 ** 3, reserve_local_bytes=64 * 1024 ** 3))


def review_row(scope: str, record_id: str, check: str, result: str, details: object,
               source: str = "", destination: str = "") -> dict:
    return dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                record_id=record_id, check_id=check, result=result,
                source_evidence=source or relative(cache / "public_metadata_sources.json"),
                output_evidence=destination or relative(output),
                details=json.dumps(dict(self_check=True, evidence=details), ensure_ascii=False))


def save_tables(stage: str) -> None:
    save_input_tables(output, images, samples, panel, field_mapping)
    visual_path = cache / "visual_selfcheck.json"
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else None
    checks = []
    for image in images.itertuples(index=False):
        receipt = json.loads((receipts_dir / f"{image.image_id}.json").read_text())
        for check in ("I01", "I02", "I03", "P03"):
            checks.append(review_row("image", image.image_id, check, "pass", receipt["structure_check"],
                                     image.source_image_id, image.input_image_path))
        viewed = visual is not None and image.image_id in visual["image_ids"]
        checks.append(review_row("image", image.image_id, "I04", "pass" if viewed else "unresolved",
                                 dict(source_identity=receipt["structure_check"], visual_evidence=visual if viewed else "native overview awaits preparer viewing"),
                                 image.source_image_id, image.input_image_path))
        if receipt.get("pixel_checks"):
            checks.append(review_row("image", image.image_id, "I05", "pass", receipt["pixel_checks"],
                                     image.source_image_id, image.input_image_path))
    checks.append(review_row("study", STUDY, "S01", "pass", dict(raw_tiffs=34, accepted_native_proofs=34,
        published_images=len(images), energy_test_metadata_without_tiff=4, alternate_processed_views="not additional physical images", inventory=relative(cache / "archive_inventory.json"))))
    for check in ("P01", "P02"):
        checks.append(review_row("panel", PANEL, check, "pass", dict(native_channels=34, output_channels=33,
            downstream_unique=31, selected_duplicate="KO-validated JARID2_N_Terminus", nuclear=["DNA1", "DNA2"])))
    checks.append(review_row("sample", STUDY, "M01", "pass", dict(sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum()),
        distinct_patients=int(samples.patient_key.nunique()), sources=relative(cache / "public_metadata_sources.json"),
        join_keys="Table 2 image id -> raw_image_info source_file -> patient/surgery biological specimen; Table 1 patient")))
    checks.append(review_row("sample", STUDY, "M02", "pass", dict(
        fields="Source Table 1 age/sex/location/OS/status and Methods collection/preservation/diagnosis/timepoint values reconstructed",
        optional_gap="Standard PFS fields empty: source PFS formal event and assessment definition not explicit", os_origin="Figure 1 primary surgery")))
    checks.append(review_row("study", STUDY, "D01", "unresolved", dict(
        stage=stage, actual_images=len(images), expected_images=34,
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        reason="整理者自检完成范围见逐图记录；待不同 CLI 上下文独立复核及主控验收。未写入独立 reviewer 身份或程序验收绑定。")))
    check_frame = pd.DataFrame(checks, columns=pd.read_csv(templates / "review_results.csv", nrows=0).columns)
    review_path = output / "review_results.csv"
    if review_path.exists():
        existing_reviews = pd.read_csv(review_path, keep_default_na=False)
        # Preserve later independent review history on a metadata-only rerun.
        independent = existing_reviews[(existing_reviews.preparer != PREPARER) | (existing_reviews.reviewer != "")]
        check_frame = pd.concat([check_frame, independent], ignore_index=True)
    with formal_publish_lease(review_path, directory=review_path.parent):
        check_frame.to_csv(review_path, index=False, encoding="utf-8-sig")
    progress(stage, len(images), sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()), gaps=["待不同 CLI 上下文独立复核；PFS 定义缺口保留空值"])


visual_complete = (cache / "visual_selfcheck.json").exists() and len(images) == len(proofs)
save_tables("prepared_pending_independent_review" if visual_complete else "metadata_ready")
if args.metadata_only:
    raise SystemExit(0)
if shutil.disk_usage(root).free < 100 * 1024 ** 3 + 10 * 1024 ** 3 or shutil.disk_usage("/tmp").free < 64 * 1024 ** 3:
    raise SystemExit("Insufficient export reserve; metadata retained")

# Export one complete ROI at a time; save index/receipt immediately after each image.
# Native TIFFs are about 1k pixels wide; strips are decoded through the shared
# window reader. No whole-slide all-channel allocation is performed.
extracted = cache / "extracted"
extracted.mkdir(exist_ok=True)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
source_indices = kept.source_channel_index.tolist()
overviews = cache / "overviews"
overviews.mkdir(exist_ok=True)
for proof in selected:
    member = proof["source_member_name"]
    image_id = Path(member).name.removesuffix("_fullstack.tiff")
    destination = output / "images" / image_id / "input.tiff"
    receipt_path = receipts_dir / f"{image_id}.json"
    if destination.exists() and receipt_path.exists() and image_id in set(images.image_id):
        prior = json.loads(receipt_path.read_text())
        actual = destination.stat()
        if (prior["panel_binding"] == panel_binding and prior["output_stat"] == dict(size=actual.st_size, mtime_ns=actual.st_mtime_ns)
                and prior["source_member"] == member and prior["source_archive_stat"] == dict(size=archive_stat.st_size, mtime_ns=archive_stat.st_mtime_ns)):
            print(json.dumps(dict(image_id=image_id, reused=True)), flush=True)
            continue
        raise ValueError(f"Existing output binding changed; targeted review required: {image_id}")
    progress("exporting", len(images), active_image=image_id, gaps=["独立复核待主控安排"])
    path = extracted / Path(member).name
    if not path.exists():
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source, path.with_suffix(".extracting").open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 ** 2)
        os.replace(path.with_suffix(".extracting"), path)
    assert path.stat().st_size == proof["source_member_stat"]["file_size"]
    info = source_rows.loc[image_id]
    height, width = proof["common_shape"]
    pixel_size = (float(info.acquisition_width_um) / width, float(info.acquisition_height_um) / height)
    spec = SourceSpec(image_id, PANEL, "tiff", path, tuple(range(34)), pixel_size_um=pixel_size,
                      source_revision="Zenodo14679443_revision8", intensity_description="raw source float32 IMC intensities")
    with open_source(spec) as reader:
        assert reader.shape == (34, height, width) and str(reader.dtype) == "float32"
        if destination.exists() and not receipt_path.exists():
            # The first export finished writing but its TiffFrame QA failed.
            # Recover that native file through the same header and pixel checks;
            # source-matching positive marker pixels below prove admission.
            side = max(height, width)
            left, top = (side-width)//2, (side-height)//2
            result = dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                          pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                          source_dtype="float32", source_channel_count=34, channel_count=33,
                          admitted=True, valid_pixels=None, reused_written_tiff=True,
                          admission_evidence="Nonzero retained downstream signal in source/output blocks verified below; full export counts were not persisted before the original QA failure.")
        else:
            result = export_input(reader, panel, destination, tile_size=512)
        if not result["admitted"]:
            write_json(receipt_path, dict(source_member=member, exclusion="all retained downstream markers zero", export=result))
            print(json.dumps(dict(image_id=image_id, excluded="all downstream markers zero")), flush=True)
            continue
        row = dict.fromkeys(image_columns, None)
        row.update(study=STUDY, image_id=image_id, sample_id=info.sample_id, panel_id=PANEL, image_kind="roi",
                   source_image_id=relative(archive_path) + "::" + member, source_files=[relative(archive_path)],
                   source_format="tiff", source_acquisition_id=int(info.acquisition_id), source_axes="CYX", source_dtype="float32",
                   source_revision="Zenodo14679443_revision8", roi_id=image_id.rsplit("_", 1)[1], region_annotation=info.region_type,
                   source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                   height_px=result["height_px"], width_px=result["width_px"], source_channel_count=34, channel_count=33,
                   pixel_size_x_um=pixel_size[0], pixel_size_y_um=pixel_size[1], input_image_path=relative(destination),
                   intensity_description="Native decoded float32 IMC raw_fullstack TIFF intensities; no rescaling, normalization or spillover compensation.",
                   split_method="already_independent", split_evidence=relative(source_metadata / "raw_image_info.csv"),
                   metadata_notes="Physical identity: " + proof["physical_identity"] + "; full ROI. Source MCD is not distributed separately. Absolute stage coordinates remain in raw_image_info.csv and native proof; local TIFF ROI origin is (0,0). Region annotation follows source raw_image_info, not unlabeled spreadsheet suggestions.")
        for name in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"):
            row[name] = result[name]
        if result["valid_pixels"]:
            row["valid_pixels_path"] = relative(destination.with_name(result["valid_pixels"]))
        # All native layouts are checked using all retained channels, one interior
        # and one border block. Square padding is checked independently.
        pixel_checks = []
        observed_marker_signal = False
        with tifffile.TiffFile(destination) as tif:
            assert tif.series[0].shape == (33, max(height, width), max(height, width))
            assert tif.series[0].dtype == np.dtype("float32")
            names = [element.attrib["Name"] for element in ET.fromstring(tif.ome_metadata).iter() if element.tag.endswith("}Channel")]
            assert names == kept.marker_id.tolist()
            scalar = scalar_tiff(tif)
            for x0, y0 in ((width // 2 - 16, height // 2 - 16), (width - 32, height - 32)):
                window = (x0, y0, x0 + 32, y0 + 32)
                native = reader.read_window(source_indices, window)
                target_window = (x0 + result["pad_left_px"], y0 + result["pad_top_px"], x0 + 32 + result["pad_left_px"], y0 + 32 + result["pad_top_px"])
                saved = np.stack([read_tiff_window(scalar, i, target_window) for i in range(33)])
                assert native.dtype == saved.dtype and np.array_equal(native, saved)
                marker_nonzero = int(np.count_nonzero(native[(kept.role == "downstream").to_numpy()]))
                observed_marker_signal |= marker_nonzero > 0
                pixel_checks.append(dict(source_window=window, output_window=target_window, channels=list(range(33)), equal=True, dtype="float32", downstream_nonzero_pixels=marker_nonzero))
            if result.get("reused_written_tiff"):
                assert observed_marker_signal, "Recovered TIFF requires positive, source-matching marker signal"
            for i in range(33):
                side = max(height, width)
                left, top = result["pad_left_px"], result["pad_top_px"]
                for window in ((0, 0, left, side), (left+width, 0, side, side), (0, 0, side, top), (0, top+height, side, side)):
                    if window[0] < window[2] and window[1] < window[3]:
                        assert not np.any(read_tiff_window(scalar, i, window))
        # Display-only RGB: GFAP(red), CD45(green), DNA2(blue); output remains native.
        view = reader.read_window([6, 9, 33], (0, 0, width, height))[:, ::4, ::4]
        rgb = []
        for values in view:
            upper = np.percentile(values, 99.5)
            rgb.append(np.clip(np.arcsinh(values) / np.arcsinh(upper) if upper > 0 else values, 0, 1))
        from PIL import Image
        Image.fromarray((np.stack(rgb, axis=-1) * 255).astype(np.uint8)).save(overviews / f"{image_id}.png")
    stat = destination.stat()
    write_json(receipt_path, dict(source_member=member, source_archive_stat=dict(size=archive_stat.st_size, mtime_ns=archive_stat.st_mtime_ns),
        panel_binding=panel_binding, output_stat=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
        export=result, pixel_checks=pixel_checks,
        structure_check=dict(shape=[33, result["height_px"], result["width_px"]], dtype="float32", channel_names=kept.marker_id.tolist(),
                             sample_id=info.sample_id, native_frame=[0, 0, width, height],
                             padding=[result[name] for name in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px")],
                             region_annotation=info.region_type, all_padding_zero=True)))
    images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
    save_tables("exporting")
    print(json.dumps(dict(image_id=image_id, completed_images=len(images), shape=[33, result["height_px"], result["width_px"]], sample_id=info.sample_id)), flush=True)

save_tables("export_complete_pending_visual_selfcheck")
# The preparer examines the cached native-channel montage before completing handoff.
import matplotlib.pyplot as plt
from PIL import Image
ordered = sorted(images.image_id)
fig, axes = plt.subplots((len(ordered) + 3) // 4, 4, figsize=(12, 3 * ((len(ordered) + 3) // 4)), squeeze=False)
for axis, image_id in zip(axes.flat, ordered):
    axis.imshow(Image.open(overviews / f"{image_id}.png"))
    axis.set_title(image_id, fontsize=9)
for axis in axes.flat:
    axis.axis("off")
fig.suptitle("Native ROI previews: GFAP red / CD45 green / DNA2 blue (display-only asinh)")
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(cache / "native_roi_overview.png", dpi=100)
plt.close(fig)
print(json.dumps(dict(study=STUDY, exported_images=len(images), expected_images=len(proofs), sample_rows=len(samples),
                      sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum())), ensure_ascii=False), flush=True)
