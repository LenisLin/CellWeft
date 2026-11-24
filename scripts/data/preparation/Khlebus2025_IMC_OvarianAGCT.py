"""Prepare Khlebus AGCT native IMC ROI inputs and source-linked clinical metadata.

Inputs: Zenodo 15384724 original ZIP, accepted native proofs, study S1, and the
public clinical workbook. Outputs: independent CYX input.tiff, complete image,
sample, panel, field mapping, and preparer self-check tables. No intensity
normalization is performed.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import io
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import openpyxl
import pandas as pd
import requests
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

STUDY = "Khlebus2025_IMC_OvarianAGCT"
PANEL = STUDY + "__native44"
PREPARER = "codex_cli_0330_repair_1"
PROJECT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--image", action="append", help="Export only named ROI(s), merging the existing index")
args = parser.parse_args()
root = args.data_root
metadata = metadata_source_directory(root, STUDY)
cache = PROJECT / ".cache/cellweft/runtime/study_input" / STUDY
output = root / "processed" / STUDY
job = PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0330_repair_1"
templates = PROJECT / "docs/data/templates/study_input"
archive_path = root / "raw/public" / STUDY / "zenodo_15384724/IMC-images_Raw_OME-TIFF.zip"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
for directory in (cache, output, job):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")


def relative(path: Path) -> str:
    if path.is_relative_to(cache) or path.is_relative_to(job):
        return str(path)
    return str(metadata_source_relative(path, root))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(stage: str, count: int, **details: object) -> None:
    write_json(job / "progress.json", dict(study=STUDY, stage=stage, completed_images=count,
               expected_images=130, updated_at=datetime.now(timezone.utc).isoformat(), **details))


def fetch_small(name: str, url: str, expected_bytes: int | None = None) -> Path:
    """Read the formally received source attachment."""
    path = metadata / "source_metadata" / name
    assert path.is_file(), f"Missing received attachment: {name}"
    return path


# Bind the accepted native verification to the unchanged complete archive.
verification = json.loads((metadata / "native_verification.json").read_text())
proofs = sorted(verification["native_result"]["images"], key=lambda item: item["physical_identity"])
assert verification["scope_complete"] and len(proofs) == 130
stat = archive_path.stat()
for proof in proofs:
    assert proof["complete"] and proof["counted"] and proof["native_order_verified"]
    assert (stat.st_size, stat.st_mtime_ns) == (proof["source_file_stat"]["size"], proof["source_file_stat"]["mtime_ns"])
assert len({tuple(proof["channels"]) for proof in proofs}) == 1
names = proofs[0]["channels"]
panel = panel_from_s1(STUDY, PANEL, names, s1_path)
assert len(panel) == 44 and int(panel.keep_channel.sum()) == 34
if (output / "panel_mapping.csv").exists():
    previous = pd.read_csv(output / "panel_mapping.csv", keep_default_na=False)
    fields = ["source_name", "marker_id", "role", "keep_channel"]
    assert previous[fields].to_dict("records") == panel[fields].to_dict("records"), "Existing pixel selection changed"
with formal_read_lease(archive_path), zipfile.ZipFile(archive_path) as archive:
    members = archive.infolist()
    assert {member.filename for member in members if member.filename.endswith(".ome.tiff") and not member.filename.startswith("__MACOSX/")} == {proof["source_member_name"] for proof in proofs}
    inventory = []
    for member in members:
        category = ("directory" if member.is_dir() else "macos_resource_fork" if member.filename.startswith("__MACOSX/")
                    else "native_roi" if member.filename.endswith(".ome.tiff") else "roi_channel_summary")
        inventory.append(dict(member=member.filename, bytes=member.file_size, category=category))
    for proof in proofs:
        entry = archive.getinfo(proof["source_member_name"])
        for key in ("file_size", "compress_size", "CRC", "header_offset", "flag_bits"):
            assert getattr(entry, key) == proof["source_member_stat"][key]
        data = archive.read(proof["summary_member_name"])
        summary = pd.read_csv(io.BytesIO(data), sep="\t")
        assert summary.Page.tolist() == list(range(44)) and summary.Label.tolist() == proof["channels"]
        summary_path = cache / "summaries" / Path(proof["summary_member_name"]).name
        summary_path.parent.mkdir(exist_ok=True)
        if not summary_path.exists():
            summary_path.write_bytes(data)
write_json(cache / "archive_inventory.json", inventory)

# The release explicitly states that ROI filenames relate to clinical Sample IDs.
clinical_url = "https://zenodo.org/records/15384724/files/Clinical%20characteristics%20by%20patient.xlsx?download=1"
clinical_path = fetch_small("Clinical characteristics by patient.xlsx", clinical_url, 13063)
roi_url = "https://zenodo.org/records/15384724/files/ROIs_Selection.zip?download=1"
roi_path = fetch_small("ROIs_Selection.zip", roi_url, 13917506)
with formal_read_lease(roi_path), zipfile.ZipFile(roi_path) as archive:
    write_json(cache / "roi_selection_inventory.json", [dict(member=m.filename, bytes=m.file_size) for m in archive.infolist()])
    roi_samples = {Path(name).stem.split("_")[0] for name in archive.namelist()
                   if name.startswith("ROIs_Selection/") and name.lower().endswith(".png")}
    for sid in ("GCT001", "GCT003", "GCT012", "GCT024"):
        member = next(name for name in archive.namelist() if name.startswith("ROIs_Selection/" + sid + "_") and name.lower().endswith(".png"))
        destination = cache / "roi_selection" / Path(member).name
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            destination.write_bytes(archive.read(member))
scope = json.loads((metadata / "source_scope.json").read_text())
article_path = metadata / "source_metadata/article_PMC12555029.xml"
article = ET.parse(article_path).getroot()
sections = {" ".join(section.find("title").itertext()): " ".join(section.itertext())
            for section in article.iter("sec") if section.find("title") is not None}
assert "24 patients" in sections["Patients and samples"] and "nonmatched recurrent" in sections["Patients and samples"]
release = json.loads((metadata / "official_zenodo_record.json").read_text())
assert "file names related to Sample IDs from clinical table" in release["metadata"]["description"]
assert "Code used for processing and analysis is available upon request" in sections["Data Availability"]
sheet = openpyxl.load_workbook(clinical_path, data_only=True).worksheets[0]
clinical = pd.DataFrame(list(sheet.values)[2:], columns=[str(value).strip() for value in list(sheet.values)[1]])
clinical = clinical[clinical["Sample ID"].notna()].copy()
assert clinical["Sample ID"].is_unique
source_samples = {proof["physical_identity"].split("_")[0] for proof in proofs}
assert source_samples == set(clinical["Sample ID"]) == roi_samples and len(source_samples) == 23
clinical.to_csv(cache / "clinical_source_rows.csv", index=False, encoding="utf-8-sig")
issues = ["论文称24位患者/24份标本；当前发布的130个ROI及临床附件均覆盖23份标本，GCT013未出现。未补造第24份标本。",
          "来源TIFF未提供物理尺度metadata，论文的spot单位存在冲突；pixel_size_x_um和pixel_size_y_um留空。"]
requests_to_controller = ["安排不同CLI上下文独立复核本study产物；D01保持unresolved，未写独立验收pass。",
                          "登记公开发布23份标本与论文24份的范围差异；现有130张ROI均有临床sample和patient连接，无需删图或重导出。"]

# Preserve clinical time semantics: diagnosis age/stage are not collection age/stage.
samples = pd.read_csv(templates / "sample_metadata.csv")
sample_rows = []
field_rows = []


def mapping(source: Path, source_field: str, table: str, field: str, rule: str,
            *, notes: str = "", values: dict | None = None, time: str = "") -> None:
    field_rows.append(dict(study=STUDY, source_file=relative(source), source_sheet_or_table=sheet.title if source == clinical_path else "",
        source_field=source_field, target_table=table, target_field=field,
        join_keys="ROI stem before final '_' = clinical Sample ID" if source == clinical_path else "physical_identity / archive member",
        value_mapping=json.dumps(values or {}, ensure_ascii=False), transformation_rule=rule,
        time_origin_rule=time, conflict_rule="保留来源；不从ROI名称推测患者；未知值留空", notes=notes))


direct_fields = {"source_sample_id": "Sample ID", "age_at_diagnosis_years": "Age at diagnosis",
                 "anatomical_site": "Anatomical tumor site", "stage_group_at_index": "Initial stage at diagnosis (Detailed)"}
raw_fields = {
    "source_condition": ("Condition", "string", "Source primary/recurrent condition at specimen collection"),
    "source_stage_group": ("Initial stage at diagnosis (In main manuscript)", "string", "Author early/advanced stage grouping at initial diagnosis"),
    "menopause_status_at_diagnosis": ("Menopause status at diagnosis", "string", "Source menopause status at initial diagnosis"),
    "tumor_size_at_diagnosis_mm": ("Tumor size at diagnosis (mm)", "number", "Tumor size at initial diagnosis", "mm"),
    "tumor_size_at_collection_mm": ("Tumor size at time of sample collection (mm)", "number", "Tumor size at specimen collection", "mm"),
    "recurrence_number_at_collection": ("Recurrence number at the time of sample collection", "integer", "Recurrence ordinal at specimen collection"),
    "first_recurrence_from_diagnosis_years": ("Time of first recurrence from diagnosis (years)", "number", "Time from diagnosis to first recurrence", "years"),
    "prior_systemic_treatment_count": (clinical.columns[12], "integer", "Source count of systemic treatments before collection; not therapy line count"),
    "prior_chemotherapy_count": (clinical.columns[13], "integer", "Source count of chemotherapy before collection"),
    "prior_hormonal_therapy_count": (clinical.columns[14], "integer", "Source count of hormonal therapy before collection"),
    "prior_targeted_therapy_count": (clinical.columns[15], "integer", "Source count of targeted therapy before collection"),
    "prior_radiotherapy_count": (clinical.columns[16], "integer", "Source count of radiotherapy before collection"),
    "chemotherapy_before_collection": ("Chemotherapy before sample collection", "string", "Source yes/no for prior chemotherapy"),
    "source_vital_status": ("Vital status", "string", "Source vital/disease status at unspecified follow-up; not collection status"),
    "source_cause_of_death": ("Cause of death", "string", "Source cause of death; unknown retained explicitly"),
    "source_overall_survival_years": ("Overal survival (years)", "number", "Reported overall survival; origin/cutoff not specified, so standardized OS fields stay empty", "years"),
    "source_archival_years_before_sequencing": (clinical.columns[21], "number", "Source storage duration before sequencing; not a collection date", "years"),
    "source_kmt2d_frameshift_mutation": (clinical.columns[22], "string", "Source KMT2D frameshift yes/no"),
    "source_chr3p29_deletion": (clinical.columns[23], "string", "Source chromosome 3p29 deletion yes/no"),
    "source_chr21q22_3_deletion": (clinical.columns[24], "string", "Source chromosome 21q22.3 deletion yes/no"),
    "source_foxl2_c_c402g_detected": (clinical.columns[25], "string", "Source FOXL2 c.C402G detected yes/no"),
    "source_foxl2_c_c402g_vaf": (clinical.columns[26], "number", "Source FOXL2 c.C402G variant allele fraction"),
}
for target, source_field in direct_fields.items():
    mapping(clinical_path, source_field, "sample_metadata", target, "原值复制；数值Unknown置空", time="initial diagnosis" if "diagnosis" in source_field.lower() else "specimen collection")
for target, spec in raw_fields.items():
    source_field, kind, definition, *unit = spec
    mapping(clinical_path, source_field, "sample_metadata", target, "复制原值；数值Unknown/N/A及缺失置空",
            notes=json.dumps(dict(data_type=kind, definition=definition, **({"unit": unit[0]} if unit else {})), ensure_ascii=False))
for target in ("sample_id", "patient_key", "source_subject_id"):
    mapping(clinical_path, "Sample ID", "sample_metadata", target,
            "sample_id与source_subject_id沿用Sample ID；patient_key加study命名空间。附件按patient发布、论文为不同患者的非配对取样。",
            notes="23 sample rows; 130/130 image→sample and image→patient; 23 distinct patient keys. Article claims24; release provides23.")
mapping(clinical_path, "Donor gender", "sample_metadata", "sex_reported", "来源性别映射", values={"Female": "female"})
mapping(clinical_path, "Condition", "sample_metadata", "lesion_type", "来源取样时原发/复发状态映射", values={"Primary tumor": "primary", "Recurrent tumor": "recurrence"})
mapping(clinical_path, "Condition", "sample_metadata", "disease_status_at_collection", "来源取样时状态映射", values={"Primary tumor": "active", "Recurrent tumor": "relapsed"})
mapping(clinical_path, "Vital status", "sample_metadata", "vital_status", "来源随访存活状态映射；不用于取样时疾病状态", values={"Deceased": "deceased", "Alive with disease": "alive", "Alive - no evidence of disease": "alive"})
constants = dict(cohort_id="AGCT_IMC", center_id="MD_Anderson", sample_type="tumor_tissue_section", tissue_type="tumor",
                 collection_method="resection", preservation_method="ffpe", case_control_status="case", index_event="diagnosis",
                 index_definition_version="source_initial_diagnosis_v1", primary_diagnosis="adult-type ovarian granulosa cell tumor",
                 primary_site="ovary", histology="adult-type granulosa cell tumor")
for target, value in constants.items():
    mapping(article_path, "Patients and samples; Figure 1 caption", "sample_metadata", target, "study级来源常量: " + value)
mapping(clinical_path, "Sample ID; all clinical columns", "sample_metadata", "metadata_notes", "记录连接来源、时点及未明确OS起点；不以来源Unknown补值")
for _, source in clinical.iterrows():
    row = {key: None for key in samples.columns}
    sid = source["Sample ID"]
    row.update(study=STUDY, sample_id=sid, source_subject_id=sid, patient_key=f"{STUDY}__patient_{sid}", **constants)
    for target, source_field in direct_fields.items():
        row[target] = None if source[source_field] == "Unknown" else source[source_field]
    for target, (source_field, kind, *_) in raw_fields.items():
        value = source[source_field]
        row[target] = None if pd.isna(value) or (kind != "string" and value in ("Unknown", "N/A")) else value
    row.update(sex_reported={"Female": "female"}[source["Donor gender"]],
               lesion_type={"Primary tumor": "primary", "Recurrent tumor": "recurrence"}[source["Condition"]],
               disease_status_at_collection={"Primary tumor": "active", "Recurrent tumor": "relapsed"}[source["Condition"]],
               vital_status="deceased" if source["Vital status"] == "Deceased" else "alive",
               metadata_notes="Clinical workbook is per patient; release links ROI stems to Sample ID. Nonmatched specimens in paper. Diagnosis fields retain diagnosis timing; collection age and standard OS origin/cutoff unavailable. Prior treatment counts refer to collection, not the diagnosis index; index prior_systemic_therapy is blank. Source raw clinical extensions retain reported values.")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=list(samples.columns) + list(raw_fields))
source_links = pd.DataFrame([dict(image_id=proof["physical_identity"], source_image_id=proof["source_member_name"],
    sample_id=proof["physical_identity"].split("_")[0], source_summary=proof["summary_member_name"])
    for proof in proofs]).merge(samples[["sample_id", "patient_key", "source_subject_id"]], on="sample_id", validate="many_to_one")
assert len(source_links) == 130 and source_links.patient_key.notna().all()
source_links.to_csv(cache / "source_image_sample_map.csv", index=False, encoding="utf-8-sig")

for target in pd.read_csv(templates / "image_metadata.csv").columns:
    mapping(archive_path, "ZIP member / own summary / native verification", "image_metadata", target,
            "逐ROI完整1000×1000框；44来源通道按S1保留34；原生uint16；方形无需padding；以member stem连接临床Sample ID。",
            notes="Pixel size left empty: source TIFF has no embedded OME physical metadata; article's spot unit is inconsistent. No native intensity rescaling.")
mapping(s1_path, "Study + Original marker", "panel_mapping", "source_channel_index", "按每个ROI自己的summary Page连续编号，精确study和source label连接S1；全部44行保留")
mapping(s1_path, "Unified marker; Segmentation channel", "panel_mapping", "output_channel_index", "保留32个marker及191Ir/193Ir核通道；原始顺序输出34通道；CD45和CD45RO为不同测量身份")
field_mapping = pd.DataFrame(field_rows, columns=pd.read_csv(templates / "field_mapping.csv").columns)
write_json(cache / "metadata_provenance.json", dict(clinical_url=clinical_url, roi_selection_url=roi_url,
    article_url=scope["primary_url"], release_url=release["links"]["self_html"],
    author_repository_evidence="Article Data Availability: code available upon request; no public author repository linked.",
    join_evidence="Zenodo description explicitly associates filenames with clinical Sample IDs; clinical workbook per patient and article nonmatched cohort.",
    sample_rows=len(samples), source_sample_linked_images=len(source_links), patient_linked_samples=len(samples),
    missing_optional=["collection age", "collection date", "clinical cutoff", "standardized OS origin"], issues=issues))

# Merge stable existing deliveries; save a reconstructible record after each image.
images_path = output / "images.parquet"
images = pd.read_parquet(images_path) if images_path.exists() else pd.read_csv(templates / "image_metadata.csv")
receipt_path = cache / "export_records.json"
receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
progress("metadata_ready" if args.metadata_only else "exporting", len(images), gaps=issues,
         sample_rows=len(samples), sample_linked_images=len(images), controller_requests=requests_to_controller,
         estimated_native_extraction_bytes=88014281, estimated_uncompressed_output_bytes=130*34*1000*1000*2,
         disk_free_bytes=shutil.disk_usage(root).free,
         local_staging_free_bytes=shutil.disk_usage(Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")).parent).free)


def save_tables() -> None:
    save_input_tables(output, images, samples, panel, field_mapping)


if not args.metadata_only:
    if shutil.disk_usage(root).free < 100*1024**3 + 130*34*1000*1000*2:
        raise RuntimeError("NAS reserve plus conservative uncompressed output estimate unavailable")
    staging_root = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
    if shutil.disk_usage(staging_root.parent).free < 64*1024**3 + 34*1000*1000*2:
        raise RuntimeError("Local reserve plus one uncompressed image unavailable")
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    with zipfile.ZipFile(archive_path) as archive:
        for proof in proofs:
            image_id = proof["physical_identity"]
            if args.image and image_id not in args.image:
                continue
            destination = output / "images" / image_id / "input.tiff"
            if image_id in receipts and destination.exists():
                receipt = receipts[image_id]
                assert (destination.stat().st_size, destination.stat().st_mtime_ns) == tuple(receipt["output_stat"])
                if image_id not in set(images.image_id):
                    images = pd.concat([images, pd.DataFrame([receipt["row"]])], ignore_index=True)
                    save_tables()
                    progress("exporting", len(images), last_completed_image=image_id,
                             sample_rows=len(samples), sample_linked_images=len(images),
                             patient_linked_samples=len(samples), gaps=issues,
                             controller_requests=requests_to_controller)
                continue
            extracted = staging_root / STUDY / "native_roi" / Path(proof["source_member_name"]).name
            extracted.parent.mkdir(parents=True, exist_ok=True)
            if not extracted.exists():
                temporary = extracted.with_suffix(extracted.suffix + ".extracting")
                with formal_read_lease(archive_path):
                    with archive.open(proof["source_member_name"]) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=4*1024**2)
                assert temporary.stat().st_size == proof["source_member_stat"]["file_size"]
                os.replace(temporary, extracted)
            assert extracted.stat().st_size == proof["source_member_stat"]["file_size"]
            source = SourceSpec(image_id, PANEL, "tiff", extracted, tuple(range(44)), source_revision="zenodo_15384724_revision4")
            with open_source(source) as reader:
                assert reader.shape == (44, 1000, 1000) and reader.dtype == np.dtype("uint16")
                result = export_input(reader, panel, destination)
                assert result["admitted"], "All admitted markers are zero: record exclusion before changing source scope"
                checks = []
                # One actual format/panel/split group; cover signal and boundary blocks.
                if image_id == proofs[0]["physical_identity"]:
                    with formal_read_lease(destination), tifffile.TiffFile(destination) as saved:
                        for window in ((400, 400, 432, 432), (0, 0, 32, 32)):
                            actual = np.stack([read_tiff_window(saved, index, window) for index in range(34)])
                            expected = reader.read_window(kept.source_channel_index.tolist(), window)
                            assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
                            checks.append(dict(window=list(window), channels="all 34 retained channels", equal=True,
                                               nonzero_pixels=int(np.count_nonzero(expected))))
                if image_id in {proofs[0]["physical_identity"], "GCT003_A", "GCT012_A", proofs[-1]["physical_identity"]}:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                    with formal_read_lease(extracted):
                        for ax, channel in zip(axes, (41, 4, 29, 33)):
                            values = reader.read_window([channel], (0, 0, 1000, 1000))[0]
                            ax.imshow(np.arcsinh(values / 5), cmap="gray")
                            ax.set_title(names[channel]); ax.axis("off")
                    fig.suptitle(image_id + " | full native ROI | display-only arcsinh(x/5)")
                    fig.tight_layout()
                    preview = cache / "overviews" / (image_id + ".png")
                    preview.parent.mkdir(exist_ok=True)
                    fig.savefig(preview, dpi=120); plt.close(fig)
            with formal_read_lease(destination), tifffile.TiffFile(destination) as saved:
                ome = ET.fromstring(saved.ome_metadata)
                channels = [element.attrib["Name"] for element in ome.iter() if element.tag.endswith("}Channel")]
                assert channels == kept.marker_id.tolist()
                assert len(saved.pages) == 34 and all(page.shape == (1000, 1000) and page.dtype == np.dtype("uint16") for page in saved.pages)
            row = dict(study=STUDY, image_id=image_id, sample_id=image_id.split("_")[0], panel_id=PANEL,
                       image_kind="roi", source_image_id=proof["source_member_name"], source_files=[relative(archive_path)],
                       source_format="ome-tiff in zip", source_axes="CYX", source_dtype="uint16",
                       source_revision="zenodo_15384724_revision4", roi_id=image_id,
                       source_x0_px=0, source_y0_px=0, source_x1_px=1000, source_y1_px=1000,
                       pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                       height_px=1000, width_px=1000, source_channel_count=44, channel_count=34,
                       input_image_path=relative(destination), intensity_description="Published raw 16-bit MCD Viewer OME-TIFF values; no normalization or intensity rescaling",
                       split_method="already_independent", split_evidence=relative(metadata / "official_zenodo_record.json") + "; " + relative(roi_path),
                       metadata_notes="Complete published ROI frame; native proof reused. Clinical Sample ID linkage explicitly documented by Zenodo. Pixel size unspecified in native TIFF; paper spot units inconsistent.")
            receipts[image_id] = dict(row=row, export=result, pixel_checks=checks,
                                     output_stat=[destination.stat().st_size, destination.stat().st_mtime_ns])
            write_json(receipt_path, receipts)
            images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
            save_tables()
            progress("exporting", len(images), last_completed_image=image_id, sample_rows=len(samples),
                     sample_linked_images=len(images), patient_linked_samples=len(samples), gaps=issues,
                     controller_requests=requests_to_controller)
            print(json.dumps(dict(image_id=image_id, completed=len(images), expected=130)), flush=True)
            extracted.unlink()
save_tables()

# Keep the prior independent review effective until a separate reviewer rechecks this delivery.
assert (output / "review_results.csv").exists()
summaries = {
    "S01": f"Preparer self-check: 130 raw ROI TIFFs +130 channel summaries; other members are macOS forks/directory. {len(images)}/130 exported. " + issues[0],
    "I01": "Preparer self-check: unique ROI/member/output identity and source stat binding.",
    "I02": f"Preparer self-check: {len(images)}/{len(images)} images connect to23 biological specimens and23 distinct patients via source clinical Sample ID.",
    "I03": "Preparer self-check: complete1000×1000 ROI; uint16; no padding or scaling. Native pixel scale unavailable; blank.",
    "I04": "Preparer self-check: independent published ROI per single source specimen; source explicitly documents Sample ID linkage; no multi-sample splitting.",
    "P01": "Preparer self-check: every ROI own summary matches all44 ordered channels; exact study/source S1 join; no alias substitution.",
    "P02": "Preparer self-check:32 marker channels +191Ir/193Ir nuclear channels kept;10 blank mass channels omitted. CD45RO/CD45 retain distinct marker identities.",
    "P03": f"Preparer self-check: {len(images)} TIFF headers checked:34 channels,1000×1000,uint16, OME names equal output order.",
    "I05": "Preparer self-check: GCT001_A all34 retained channels at [400,400,432,432) and[0,0,32,32) exactly compared; see export_records.json. Single format/panel/split group; no padding.",
    "M01": "Preparer self-check: public clinical workbook23 rows;130/130 source ROI connect to23 specimens and23 patients. Release explicitly relates ROI names to clinical Sample ID; paper describes nonmatched patients. Cohort count24 vs23 discrepancy retained; author code upon request, no repository linked.",
    "M02": "Preparer self-check: public clinical fields mapped with diagnosis/collection/follow-up timing preserved. NumericUnknown/N/A blank; raw vital status retained. OS reported years preserved as extension; standard OS origin/cutoff unknown. No therapy line inference from treatment count.",
    "D01": f"Repair round 1 preparer delivery: {len(images)}/130 images;23 samples/patients. Missing TIFF/index delivery repaired when count is130. Independent recheck/controller acceptance pending; no acceptance binding written.",
}
overview_judgment = cache / "overview_judgment.json"
self_checks = []
for check, details in summaries.items():
    result = "unresolved" if check == "D01" or (check.startswith("I") or check in {"S01", "P03"}) and len(images) != 130 else "pass"
    if check == "I04":
        if overview_judgment.exists():
            details += " " + overview_judgment.read_text()
        else:
            result = "unresolved"; details += " Representative full-ROI overviews await preparer viewing."
    if check == "I05" and not any(r["pixel_checks"] for r in receipts.values()):
        result = "unresolved"
        details = "Pending preparer pixel checks: no TIFF has yet been exported. Compare all retained channels in a signal block and a boundary block after export."
    if check == "M01":
        result = "unresolved"
        details += " Current delivery scope is the published130 ROI only; paper24 vs release23 specimens remains a source-level discrepancy."
    self_checks.append(dict(check_id=check, result=result, details=details))
write_json(cache / "preparer_self_check.json", dict(preparer=PREPARER, checks=self_checks,
    independent_review="pending", source_evidence=relative(metadata / "native_verification.json"),
    output_evidence=relative(output)))
write_json(cache / "delivery_summary.json", dict(study=STUDY, expected_images=130, exported_images=len(images),
    sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=len(samples),
    issues=issues, controller_requests=requests_to_controller, independent_review="pending"))
write_json(cache / "controller_requests.json", requests_to_controller)
final_stage = "prepared" if len(images) == len(proofs) else "metadata_ready" if args.metadata_only else "partial"
progress(final_stage, len(images), sample_rows=len(samples),
         sample_linked_images=len(images), patient_linked_samples=len(samples), gaps=issues,
         independent_review="pending", controller_requests=requests_to_controller)
