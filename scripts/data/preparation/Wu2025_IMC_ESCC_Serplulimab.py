"""Prepare the 21 source-verified Wu ESCC MCD1 ROIs at native intensity.

Inputs: complete original MCD1, selected-subset/native proofs, current S1,
and the locally collected article supplements. Outputs: input TIFFs, four
metadata tables, preparer checks and pending independent-review records.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import openpyxl
import pandas as pd
import requests
import tifffile

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Wu2025_IMC_ESCC_Serplulimab"
ROOT = Path(os.environ.get("CELLWEFT_DATA_ROOT", "/mnt/NAS_21T/ProjectData/CellWeft"))
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
CACHE = PROJECT / "cache/study_input" / STUDY
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
SUPPLEMENTS = ROOT / "raw/public" / STUDY / "metadata"
SOURCE_MANIFEST = ROOT / "raw/public" / STUDY / "manifest.json"
SOURCE = ROOT / "raw/public" / STUDY / "public_original/ZLJ0303_1.mcd"
LOCAL = Path("/tmp/cellweft-study-input") / STUDY / SOURCE.name
PANEL_ID = f"{STUDY}__native50"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--progress-path", type=Path, required=True)
parser.add_argument("--metadata-only", action="store_true",
                    help="Publish the public patient metadata without staging or exporting MCD images")
args = parser.parse_args()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / name, nrows=0).columns.tolist()


def publish_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with formal_publish_lease(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(encoded)
        temporary.replace(path)


with formal_read_lease(META):
    selected = json.loads((META / "selected_subset_closeout_20261024.json").read_text())
    chosen = selected["selected_images"]
    proofs = json.loads(SOURCE_MANIFEST.read_text())["preparation_sources"]["native_images"]
assert selected["qualified_usable_subset"] and len(chosen) == 21
assert all(proof["complete"] for proof in proofs)
source_stat = SOURCE.stat()
assert all(all(getattr(source_stat, "st_" + key) == proof["source_file_stat"][key]
               for key in ("size", "mtime_ns")) for proof in proofs)
names = proofs[0]["channels"][3:]
assert all(proof["channels"][3:] == names for proof in proofs)
with formal_read_lease(S1):
    panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
assert len(panel) == 50 and panel.keep_channel.sum() == 44

# Reserve disk headroom before exporting the image inputs.
if args.metadata_only:
    space = {"disk_check": "skipped for metadata-only publication"}
else:
    estimated_output = sum(max(p["data_boundary"]["width"], p["data_boundary"]["height"]) ** 2
                           * 44 * 4 for p in proofs)
    space = dict(local_free=shutil.disk_usage(LOCAL.parent if LOCAL.parent.exists() else Path("/tmp")).free,
                 nas_free=shutil.disk_usage(ROOT).free, estimated_output_bytes=estimated_output,
                 estimated_local_bytes=source_stat.st_size + 2 * 800 * 800 * 50 * 4)
    if space["nas_free"] - estimated_output < 100 * 1024**3 or space["local_free"] - space["estimated_local_bytes"] < 64 * 1024**3:
        publish_json(args.progress_path, dict(study=STUDY, status="waiting_io", stage="space", **space))
        raise SystemExit("Insufficient reserved free space; checkpoint saved")

# Load the official patient-level workbook and retain every source field for the IMC subset.
article_url = "https://www.nature.com/articles/s41467-025-67589-5"
api_url = "https://api.figshare.com/v2/articles/30466160"
workbook_path = SUPPLEMENTS / "MOESM4_ESM.xlsx"
workbook_source = str(metadata_source_relative(workbook_path, ROOT))
with formal_read_lease(META):
    article = {"files": json.loads((META / "source_manifest.json").read_text())["sources"]}
assert {item["name"] for item in article["files"]} == {f"ZLJ0303_{i}.mcd" for i in (1, 2, 3)}
sheet_keys = {
    "Table 1": "table_1",
    "Table 2": "table_2",
    "Table 3": "table_3",
    "Figure 2": "figure_2",
    "Supplementary Table 1": "supplementary_table_1",
    "Supplementary Figure 1": "supplementary_figure_1",
}
with formal_read_lease(SUPPLEMENTS):
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    assert set(workbook.sheetnames) == set(sheet_keys)
    source_tables = {}
    for sheet in workbook:
        rows = list(sheet.iter_rows())
        headers = [str(cell.value).strip() for cell in rows[0]]
        patient_column = headers.index("Patient ID")
        records = {}
        for row in rows[1:]:
            value = row[patient_column].value
            if value is None:
                continue
            patient_id = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value).strip()
            assert patient_id not in records, f"Duplicate Patient ID in {sheet.title}"
            records[patient_id] = [cell.value for cell in row]
        source_tables[sheet.title] = dict(headers=headers, records=records)
    workbook.close()
clinical_cohort_ids = list(source_tables["Supplementary Table 1"]["records"])
assert len(clinical_cohort_ids) == 20
assert all(set(clinical_cohort_ids).issubset(table["records"]) for table in source_tables.values())

field_rows = [
    dict(source_file=str(metadata_source_relative(META / 'selected_subset_closeout_20261024.json', ROOT)),
         source_sheet_or_table="selected_images + native_proof", source_field="acquisition_id;native_metadata;data_boundary",
         target_table="image_metadata", target_field="source_image_id;source_acquisition_id;roi_id;region_annotation;geometry",
         join_keys="exact original file + acquisition ID", transformation_rule="One complete independent native ROI per image; full 800x800 frame; native coordinate reader; preserve Description verbatim.",
         notes="21 selected MCD1 acquisitions; 29 zero-offset acquisitions omitted; MCD2/3 excluded under prior source-defect evidence. Embedded slide Filename differs from actual published filename; bind actual file by original source proof."),
    dict(source_file=str(metadata_source_relative(S1, ROOT)), source_sheet_or_table="Study rows",
         source_field="Original marker;Unified marker;Segmentation channel", target_table="panel_mapping",
         target_field="marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
         join_keys="Study + exact source label", transformation_rule="Use all 50 intensity-channel rows in native order, excluding coordinate fields X/Y/Z. Retain 44 S1 admitted marker and nuclear channels.", notes="No intensity transformation; DNA1 and DNA2 retain their separate measurement identities."),
    dict(source_file=str(metadata_source_relative(META / 'source_metadata_gap.json', ROOT)),
         source_sheet_or_table="MCD descriptions; Figshare v1; article; clinical workbook",
         source_field="Description;Patient ID", target_table="sample_metadata",
         target_field="sample_id;source_sample_id;patient_key;clinical fields",
         join_keys="unresolved ROI-to-biological-specimen-to-clinical-Patient-ID bridge",
         transformation_rule="Store the 20 public patient-level IMC records using project patient-record sample keys; keep source_sample_id and all image sample_id values blank because no biospecimen or acquisition crosswalk is published.",
         conflict_rule="Do not infer specimen/patient identity or response class from ROI tokens or aggregate counts.",
         notes="The source Patient ID links clinical records across the six workbook sheets but does not link any MCD acquisition to a patient or biological specimen. Project sample_id values represent patient-level records, not source sample IDs."),
]

def source_value(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return str(value)


def has_value(value) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def source_extension_name(sheet_key: str, header: str, occurrence: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", header.casefold()).strip("_")
    suffix = f"_{occurrence}" if occurrence > 1 else ""
    return f"source_{sheet_key}_{slug}{suffix}"


def add_field_mapping(*, source_file: str, sheet: str, source_field: str,
                      target_field: str, join_keys: str, transformation: str,
                      value_mapping: dict | None = None, time_origin: str | None = None,
                      notes: str | None = None) -> None:
    field_rows.append(dict(source_file=source_file, source_sheet_or_table=sheet,
        source_field=source_field, target_table="sample_metadata", target_field=target_field,
        join_keys=join_keys, transformation_rule=transformation,
        value_mapping=json.dumps(value_mapping, ensure_ascii=False) if value_mapping is not None else None,
        time_origin_rule=time_origin, notes=notes))


sample_columns = columns("sample_metadata.csv")
sample_records = {}
for patient_id in clinical_cohort_ids:
    record = dict.fromkeys(sample_columns, None)
    record.update(study=STUDY,
        sample_id=f"{STUDY}__patient_record__{patient_id}",
        patient_key=f"{STUDY}:{patient_id}", source_subject_id=patient_id,
        cohort_id="IMC", metadata_notes=(
            "Patient-level IMC cohort record from the public source workbook. "
            "sample_id is a project record key because no source sample or biospecimen ID is published; "
            "no MCD acquisition is linked to this row."))
    sample_records[patient_id] = record

source_extension_fields = []
for sheet_name, table in source_tables.items():
    headers = table["headers"]
    counts = {header: headers.count(header) for header in set(headers)}
    seen = {}
    for column_index, header in enumerate(headers):
        if header == "Patient ID":
            continue
        seen[header] = seen.get(header, 0) + 1
        occurrence = seen[header] if counts[header] > 1 else 1
        extension = source_extension_name(sheet_keys[sheet_name], header, occurrence)
        source_field = f"{header} (column {column_index + 1})" if counts[header] > 1 else header
        raw_values = [table["records"][patient_id][column_index]
                      for patient_id in clinical_cohort_ids]
        cohort_values = [source_value(value) for value in raw_values]
        if not any(value is not None for value in cohort_values):
            continue
        source_extension_fields.append(extension)
        for patient_id, value in zip(clinical_cohort_ids, cohort_values):
            sample_records[patient_id][extension] = value
        value_types = sorted({type(value).__name__ for value in raw_values if value is not None})
        field_rows.append(dict(source_file=workbook_source, source_sheet_or_table=sheet_name,
            source_field=source_field, target_table="sample_metadata", target_field=extension,
            join_keys="Exact Patient ID match to Supplementary Table 1; retain only its 20-patient IMC subset",
            transformation_rule="Store the source cell's scalar representation as text in a source_* extension column.",
            notes=f"Source extension; original workbook value type(s): {', '.join(value_types)}; header units/codes are retained."))
    field_rows.append(dict(source_file=workbook_source, source_sheet_or_table=sheet_name,
        source_field="Patient ID", target_table="sample_metadata", target_field="source_subject_id",
        join_keys="Exact Patient ID match across workbook sheets",
        transformation_rule="Convert the workbook's numeric identifier to a string; no image or specimen linkage is implied.",
        notes="Patient-level source key."))

table1 = source_tables["Table 1"]
table1_columns = {name: table1["headers"].index(name) for name in table1["headers"]}
table1_records = table1["records"]
response_map = {1: "SD", 2: "PR", 3: "CR", 4: "PD"}
for patient_id in clinical_cohort_ids:
    values = table1_records[patient_id]
    raw_tnm = str(values[table1_columns["Clinical TNM stage"]]).strip()
    tnm = re.fullmatch(r"(c?T[^N]+)(N[^M]+)(M.+)", raw_tnm)
    assert tnm is not None
    record = sample_records[patient_id]
    record.update(
        index_event="trial enrollment",
        index_definition_version="NCT05659251",
        age_at_index_years=values[table1_columns["AGE"]],
        bmi_kg_m2=values[table1_columns["BMI"]],
        ecog_at_index=values[table1_columns["ECOG performance status"]],
        primary_diagnosis="oesophageal squamous cell carcinoma",
        primary_site="oesophagus",
        histology="oesophageal squamous cell carcinoma",
        stage_group_at_index=values[table1_columns["Clinical stage"]],
        stage_system="AJCC 8th edition",
        stage_basis="clinical",
        tnm_t=tnm.group(1), tnm_n=tnm.group(2), tnm_m=tnm.group(3),
        index_therapy_modality="other",
        index_regimen="serplulimab, carboplatin, and albumin-bound paclitaxel",
        response_criteria="RECIST 1.1",
    )
    assert values[table1_columns["Objective response  (1=SD;2=PR;3=CR;4=PD)"]] in response_map

objective_response_field = source_extension_name(
    sheet_keys["Table 1"], "Objective response  (1=SD;2=PR;3=CR;4=PD)", 1)

figure2 = source_tables["Figure 2"]
figure2_columns = {name: figure2["headers"].index(name) for name in figure2["headers"]}
supplementary_figure = source_tables["Supplementary Figure 1"]
supplementary_columns = {name: supplementary_figure["headers"].index(name)
                         for name in supplementary_figure["headers"]}
for patient_id in clinical_cohort_ids:
    figure_values = figure2["records"][patient_id]
    os_state = figure_values[figure2_columns["Survival state (0=alive;1=dead)"]]
    record = sample_records[patient_id]
    record.update(
        vital_status={0: "alive", 1: "deceased"}[os_state],
        clinical_data_cutoff="2025-04-01",
        os_time=figure_values[figure2_columns["Survival time (month)"]],
        os_time_unit="months", os_origin="trial enrollment",
        os_event=int(bool(os_state)), os_event_definition="death from any cause",
    )
    dfs_state = supplementary_figure["records"][patient_id][
        supplementary_columns["DFS state (0=alive;1=Tumor recurrence or death)"]]
    record.update(
        dfs_time=supplementary_figure["records"][patient_id][
            supplementary_columns["DFS time （months）"]],
        dfs_time_unit="months", dfs_origin="surgery",
        dfs_event=int(bool(dfs_state)),
        dfs_event_definition="tumor recurrence or death from any cause",
    )

field_rows.append(dict(source_file=article_url, source_sheet_or_table="Article methods",
    source_field="study population and index regimen", target_table="sample_metadata",
    target_field="cohort_id;primary_diagnosis;primary_site;histology;index_event;index_regimen;index_therapy_modality;index_definition_version",
    join_keys="Apply the single-arm trial definition to the 20 Patient IDs selected for IMC in Supplementary Table 1",
    transformation_rule="Record the study-defined IMC cohort, ESCC diagnosis, enrollment index, neoadjuvant regimen, and trial registration.",
    value_mapping={"index_therapy_modality": "other: combined chemoimmunotherapy is not represented by a single controlled category"},
    notes="The source patient subset is 10 PCR and 10 non-PCR patients used for IMC."))

add_field_mapping(source_file=workbook_source, sheet="Supplementary Table 1",
    source_field="Patient ID", target_field="sample_id",
    join_keys="One project patient-record key per Patient ID in the 20-patient IMC subset",
    transformation="Create a project key from Patient ID because no source sample or biospecimen ID is published.",
    notes="This key identifies the patient-level clinical record and does not establish a biological sample or image relationship.")
add_field_mapping(source_file=workbook_source, sheet="Supplementary Table 1",
    source_field="Patient ID", target_field="patient_key",
    join_keys="Exact Patient ID from the 20-patient IMC subset",
    transformation="Namespace the source Patient ID by study.",
    notes="The published Patient ID is the de-identified patient key.")

field_rows.extend([
    dict(source_file=workbook_source, source_sheet_or_table="Table 1",
         source_field="AGE;BMI;ECOG performance status", target_table="sample_metadata",
         target_field="age_at_index_years;bmi_kg_m2;ecog_at_index",
         join_keys="Patient ID exact match to the 20-patient Supplementary Table 1 IMC subset",
         transformation_rule="Directly copy baseline age in years, BMI in kg/m2, and ECOG score.",
         time_origin_rule="Trial enrollment baseline.",
         notes="Table 1 and Supplementary Table 1 values agree for all 20 selected Patient IDs."),
    dict(source_file=workbook_source, source_sheet_or_table="Table 1",
         source_field="Clinical TNM stage;Clinical stage", target_table="sample_metadata",
         target_field="tnm_t;tnm_n;tnm_m;stage_group_at_index;stage_system;stage_basis",
         join_keys="Patient ID exact match to the 20-patient Supplementary Table 1 IMC subset",
         transformation_rule="Parse the source clinical TNM string into its T/N/M components; copy Clinical stage unchanged.",
         value_mapping={"stage_system": "AJCC 8th edition", "stage_basis": "clinical"},
         time_origin_rule="Clinical stage at trial enrollment, before neoadjuvant therapy.",
         notes="Original clinical TNM stage and Clinical stage are also retained in source_* columns."),
    dict(source_file=workbook_source, source_sheet_or_table="Table 1",
         source_field="Objective response  (1=SD;2=PR;3=CR;4=PD)", target_table="sample_metadata",
         target_field=objective_response_field,
         join_keys="Patient ID exact match to the 20-patient Supplementary Table 1 IMC subset",
         transformation_rule="Retain the coded post-neoadjuvant objective response; do not relabel it as best_overall_response.",
         value_mapping=response_map,
         time_origin_rule="Response assessed three weeks after the final neoadjuvant cycle.",
         notes="Article methods identify RECIST 1.1; the assessment timepoint is not a best-overall-response summary."),
    dict(source_file=article_url, source_sheet_or_table="Methods",
         source_field="RECIST 1.1 objective response assessment", target_table="sample_metadata",
         target_field="response_criteria",
         join_keys="Study-wide protocol applied to the 20-patient IMC subset",
         transformation_rule="Record the published RECIST version.",
         notes="Objective response is preserved as a source extension because it is a fixed post-treatment assessment."),
    dict(source_file=workbook_source, source_sheet_or_table="Figure 2",
         source_field="Survival state (0=alive;1=dead);Survival time (month)",
         target_table="sample_metadata", target_field="vital_status;os_event;os_time;os_time_unit;os_origin;os_event_definition;clinical_data_cutoff",
         join_keys="Patient ID exact match to the 20-patient Supplementary Table 1 IMC subset",
         transformation_rule="Map 0 to alive and 1 to deceased; map state 1 to an OS event; copy time in months.",
         value_mapping={"0": "alive", "1": "deceased"},
         time_origin_rule="Enrollment to death or the last follow-up on 2025-04-01.",
         notes="OS definition and cutoff date follow the article Methods."),
    dict(source_file=article_url, source_sheet_or_table="Methods",
         source_field="Overall survival definition and follow-up cutoff", target_table="sample_metadata",
         target_field="os_origin;os_event_definition;clinical_data_cutoff",
         join_keys="Study-wide endpoint definition applied to Figure 2 patient records",
         transformation_rule="Use enrollment as OS origin, death from any cause as event, and 2025-04-01 as cutoff.",
         time_origin_rule="Enrollment.",
         notes="Survival time is in months in Figure 2."),
    dict(source_file=workbook_source, source_sheet_or_table="Supplementary Figure 1",
         source_field="DFS state (0=alive;1=Tumor recurrence or death);DFS time （months）",
         target_table="sample_metadata", target_field="dfs_time;dfs_time_unit;dfs_origin;dfs_event;dfs_event_definition",
         join_keys="Patient ID exact match to the 20-patient Supplementary Table 1 IMC subset",
         transformation_rule="Map state 1 to a DFS event and copy time in months.",
         value_mapping={"0": False, "1": True},
         time_origin_rule="Date of surgery.",
         notes="DFS definition follows the article Methods; source state/time are also retained as extensions."),
])
for mapping in field_rows:
    if isinstance(mapping.get("value_mapping"), dict):
        mapping["value_mapping"] = json.dumps(mapping["value_mapping"], ensure_ascii=False)

samples = pd.DataFrame([sample_records[patient_id] for patient_id in clinical_cohort_ids],
                       columns=sample_columns + source_extension_fields)
fields = pd.DataFrame([dict.fromkeys(columns("field_mapping.csv"), None) | dict(study=STUDY) | r
                       for r in field_rows])
with formal_read_lease(OUTPUT):
    images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
    checks_path = CACHE / "preparer_checks.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
done = {item["image_id"] for item in images}

identity_evidence = dict(study=STUDY, public_sources=[article_url, api_url],
    checked_local_sources=[str(workbook_path), str(META / "source_marker_order_evidence.json"), str(SOURCE)],
    clinical_sheets={name: dict(rows=len(source_tables[name]["records"]),
        headers=source_tables[name]["headers"], imc_patient_matches=len(clinical_cohort_ids))
        for name in sheet_keys},
    available_files=[item["name"] for item in article["files"]],
    conclusion="The official workbook provides 20 patient-level records for the IMC cohort and the Patient ID matches exactly across all six source sheets. The workbook does not provide source sample/biospecimen IDs or a crosswalk from MCD acquisitions to patients; project sample_id values are explicitly patient-record keys, and image sample_id remains blank.",
    sample_rows=len(samples), sample_linked_images=sum(has_value(item.get("sample_id")) for item in images),
    patient_linked_samples=int(samples.patient_key.notna().sum()),
    source_gap="Public patient-level metadata is available. The acquisition-to-patient/biospecimen bridge remains unresolved; MCD2/3 retain the previously verified source-metadata defect.",
    space_estimate=space)
publish_json(CACHE / "identity_source_review.json", identity_evidence)


def progress(stage: str, current: str | None = None) -> None:
    publish_json(checks_path, checks)
    save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel, fields)
    publish_json(args.progress_path, dict(study=STUDY, stage=stage, current_image=current,
        expected_images=21, exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=sum(has_value(item.get("sample_id")) for item in images),
        patient_linked_samples=int(samples.patient_key.notna().sum()), completed_image_ids=sorted(done),
        next_image=next((f"ZLJ0303_1__acq_{p['source_acquisition_id']:03d}" for p in proofs
                         if f"ZLJ0303_1__acq_{p['source_acquisition_id']:03d}" not in done), None),
        local_source=str(LOCAL), updated_at=datetime.now(timezone.utc).isoformat()))


if args.metadata_only:
    progress("metadata_only_published")
    summary = dict(study=STUDY, exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=sum(has_value(item.get("sample_id")) for item in images),
        patient_linked_samples=int(samples.patient_key.notna().sum()))
    print(json.dumps(summary))
    raise SystemExit(0)

progress("staging_source")
LOCAL.parent.mkdir(parents=True, exist_ok=True)
if not LOCAL.exists():
    with formal_read_lease(SOURCE), SOURCE.open("rb") as source, LOCAL.with_suffix(".partial").open("wb") as target:
        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    LOCAL.with_suffix(".partial").replace(LOCAL)
assert LOCAL.stat().st_size == source_stat.st_size
revision = f"figshare30466160v1:file59123387:{source_stat.st_size}:{source_stat.st_mtime_ns}"
kept = panel[panel.keep_channel].sort_values("output_channel_index")
for selected_row, proof in zip(chosen, proofs):
    acquisition_id = int(proof["source_acquisition_id"])
    image_id = f"ZLJ0303_1__acq_{acquisition_id:03d}"
    if image_id in done and image_id in checks:
        continue
    progress("exporting", image_id)
    source = SourceSpec(image_id=image_id, panel_id=PANEL_ID, kind="mcd", path=LOCAL,
                        channels=tuple(range(50)), acquisition_id=acquisition_id,
                        pixel_size_um=(1.0, 1.0), source_revision=revision,
                        intensity_description="native unnormalized MCD float32 intensities")
    destination = OUTPUT / "images" / image_id / "input.tiff"
    with open_source(source) as reader:
        if destination.exists():
            # Recover a closed TIFF when execution stopped before indexing it.
            valid = reader.valid_window((0, 0, 800, 800))
            assert valid.all(), "Unindexed incomplete acquisitions need their original export checkpoint"
            nonzero, means = [], []
            with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
                assert len(tif.pages) == 44
                for index, source_index in enumerate(kept.source_channel_index):
                    values = reader.read_window([int(source_index)], (0, 0, 800, 800))[0]
                    observed = tif.pages[index].asarray(maxworkers=4)
                    assert observed.dtype == values.dtype
                    np.testing.assert_array_equal(observed, values)
                    nonzero.append(int(np.count_nonzero(values)))
                    means.append(float(values.mean(dtype=np.float64)))
            result = dict(height_px=800, width_px=800, source_dtype="float32",
                source_channel_count=50, channel_count=44, admitted=True, valid_pixels=None,
                channel_nonzero_pixels=nonzero, channel_means=means,
                recovered_closed_input_full_pixel_comparison=True)
        else:
            result = export_input(reader, panel, destination)
        assert result["admitted"]
        # Preparer self-check covers every retained channel at two native windows.
        windows: list[Window] = [(320, 320, 352, 352), (768, 768, 800, 800)]
        with formal_read_lease(destination):
            with tifffile.TiffFile(destination) as tif:
                assert len(tif.pages) == 44 and tif.pages[0].shape == (800, 800)
                assert tif.pages[0].dtype == np.dtype("float32")
                for window in windows:
                    native = reader.read_window(list(range(50)), window)
                    x0, y0, x1, y1 = window
                    for out_index, source_index in enumerate(kept.source_channel_index):
                        observed = tif.pages[out_index].asarray(maxworkers=4)[y0:y1, x0:x1]
                        np.testing.assert_array_equal(observed, native[int(source_index)])
        checks[image_id] = dict(header="800x800 CYX float32, 44 channels", result=result,
            pixel_windows=[list(w) for w in windows],
            all_retained_channels_equal=True)
        if acquisition_id in {1, 11, 21}:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            signal = reader.read_window([names.index(n) for n in ("DNA1", "CD45", "Pan-cytokeratin")],
                                        (0, 0, 800, 800))
            display = np.stack([np.clip(channel / np.percentile(channel[channel > 0], 99.5), 0, 1)
                                for channel in signal], axis=-1)
            figure, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(display[:, :, 0], cmap="gray")
            axes[0].set_title("DNA1")
            axes[1].imshow(display)
            axes[1].set_title("R DNA1 / G CD45 / B PanCK")
            for axis in axes:
                axis.axis("off")
            figure.suptitle(image_id + "; display-only channel P99.5 scaling")
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            plt.close(figure)
            overview = CACHE / f"{image_id}_overview.png"
            with formal_publish_lease(overview):
                overview.write_bytes(buffer.getvalue())
            checks[image_id]["overview_path"] = str(metadata_source_relative(overview, ROOT))
    row = dict.fromkeys(columns("image_metadata.csv"), None)
    row.update(study=STUDY, image_id=image_id, panel_id=PANEL_ID, image_kind="roi",
        source_image_id=selected_row["acquisition_id"], source_files=[str(metadata_source_relative(SOURCE, ROOT))],
        source_format="mcd", source_acquisition_id=acquisition_id, source_axes="CYX", source_dtype="float32",
        source_revision=revision, slide_id="0", roi_id=str(proof["acquisition_roi_id"]),
        region_annotation=proof["description"], source_x0_px=0, source_y0_px=0,
        source_x1_px=800, source_y1_px=800, pad_left_px=0, pad_top_px=0, pad_right_px=0,
        pad_bottom_px=0, height_px=800, width_px=800, source_channel_count=50, channel_count=44,
        pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=str(metadata_source_relative(destination, ROOT)),
        valid_pixels_path=(str(metadata_source_relative(destination.with_name('valid_pixels.tiff'), ROOT)) if result["valid_pixels"] else None),
        intensity_description=source.intensity_description, split_method="source_acquisition",
        split_evidence=str(metadata_source_relative(META / 'selected_subset_closeout_20261024.json', ROOT)),
        metadata_notes="ROI identity confirmed; biological specimen and patient bridge unresolved. See identity_source_review.json.")
    if image_id not in done:
        images.append(row)
    done.add(image_id)
    progress("image_closed", image_id)

progress("prepared_pending_independent_review")
review_path = OUTPUT / "review_results.csv"
if not review_path.exists():
    with formal_publish_lease(review_path, directory=OUTPUT):
        temporary = review_path.with_suffix(".csv.partial")
        temporary.write_bytes((TEMPLATES / "review_results.csv").read_bytes())
        temporary.replace(review_path)
summary = dict(study=STUDY, exported_images=len(images), sample_rows=len(samples),
    sample_linked_images=sum(has_value(item.get("sample_id")) for item in images),
    patient_linked_samples=int(samples.patient_key.notna().sum()))
print(json.dumps(summary))
