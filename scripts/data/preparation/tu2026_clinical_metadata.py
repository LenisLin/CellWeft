"""Extract the published Tu supplemental clinical table into study input metadata."""

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative, publish_metadata_sources
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree

import pandas as pd

STUDY = "Tu2026_PhenoCycler_Neuroblastoma"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
FACTS = metadata_source_directory(ROOT, STUDY) / 'input_facts'
SOURCE = FACTS / "supplement_springer_MOESM1.pdf"
OUT = FACTS
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
COLUMNS = ["tma", "geomx", "codex", "source_row_id", "tma_map", "source_patient_id", "treatment", "staging", "age_diagnosis_years", "age_diagnosis_month_group", "histology", "mycn_amplification", "risk_group", "gender", "followup_months", "overall_survival_months", "overall_survival_years", "death", "residual_recurrence", "stroma", "schwann_cells", "index_mc", "shimada"]
BOUNDS = [36, 62, 91, 120, 137, 173, 207, 245, 275, 307, 339, 382, 427, 484, 514, 545, 577, 608, 638, 676, 705, 739, 764, 806]

raw = subprocess.check_output(["pdftotext", "-bbox", str(SOURCE), "-"])
root = ElementTree.fromstring(raw)
ns = {"h": "http://www.w3.org/1999/xhtml"}
rows = []
for page_number, page in enumerate(root.findall(".//h:page", ns), 1):
    words = [(word.text or "", float(word.attrib["xMin"]), float(word.attrib["yMin"])) for word in page.findall("h:word", ns)]
    anchors = sorted((y, text) for text, x, y in words if re.fullmatch(r"TMA[135]_[A-E][1-4]", text))
    for index, (y, label) in enumerate(anchors):
        lower = (anchors[index - 1][0] + y) / 2 if index else y - 9
        upper = (y + anchors[index + 1][0]) / 2 if index < len(anchors) - 1 else y + 9
        row = {name: " ".join(text for yy, x, text in sorted((yy, x, text) for text, x, yy in words if lower <= yy < upper and BOUNDS[col] <= x < BOUNDS[col + 1])) for col, name in enumerate(COLUMNS)}
        if row["tma_map"] != label:
            raise ValueError(f"Clinical row geometry mismatch: {label}")
        row.update(source_page=page_number, Parent=label, sample_id=f"sample_{label}")
        rows.append(row)
source = pd.DataFrame(rows)
if len(source) != 38 or set(source.source_row_id) != {str(i) for i in range(1, 39)}:
    raise ValueError("Expected the 38 published clinical rows")
eligible = source.loc[source.codex.eq("Y")].copy()
if len(eligible) != 29 or eligible.tma_map.duplicated().any():
    raise ValueError("Expected 29 unique CODEX=Y specimens")

# Preserve reported endpoints as source fields because the table gives no time origin.
rename = {
    "source_patient_id": "source_subject_id", "tma_map": "source_sample_id",
    "age_diagnosis_years": "age_at_diagnosis_years", "histology": "histology",
    "staging": "stage_reported", "mycn_amplification": "mycn_amplification_reported",
    "risk_group": "risk_group_reported", "followup_months": "reported_followup_months",
    "overall_survival_months": "reported_os_months", "death": "death_reported",
    "residual_recurrence": "residual_recurrence_reported", "stroma": "stroma_reported",
    "schwann_cells": "schwann_cells_reported", "index_mc": "index_mc_reported",
    "shimada": "shimada_reported", "age_diagnosis_month_group": "age_diagnosis_month_group_reported",
}
samples = pd.DataFrame(index=eligible.index, columns=pd.read_csv(TEMPLATES / "sample_metadata.csv").columns).fillna("")
samples["study"] = STUDY
samples["sample_id"] = eligible.sample_id
for original, target in rename.items():
    samples[target] = eligible[original].replace({"Unknown": "", "unknown": ""})
samples["sex_reported"] = eligible.gender.map({"M": "male", "F": "female"})
treatment = eligible.treatment.str.replace(r"\s+", "", regex=True).str.lower()
samples["timepoint_label"] = treatment.map({"pre-treatment": "pre_treatment", "post-treatment": "post_treatment"})
if samples[["sex_reported", "timepoint_label"]].isna().any().any():
    raise ValueError("Unrecognized published sex or treatment value")
samples["primary_diagnosis"] = "Neuroblastoma"
samples["preservation_method"] = "ffpe"
samples["case_control_status"] = "case"
samples["metadata_notes"] = "Published CODEX=Y specimen linked by TMAMap to author Parent. PatientID retained verbatim as source_subject_id; cross-specimen patient identity is not established. Reported follow-up and survival lack an explicit time origin."

mapping = []
for original, target in {**rename, "gender": "sex_reported", "treatment": "timepoint_label", "tma_map": "source_sample_id;sample_id"}.items():
    rule = "Preserve source text; Unknown becomes empty"
    if original == "gender":
        rule = "M=male; F=female"
    elif original == "treatment":
        rule = "Pre-treatment=pre_treatment; Post-treatment=post_treatment; remove whitespace before matching"
    elif original == "tma_map":
        rule = "source_sample_id=TMAMap; sample_id=sample_ + TMAMap; include CODEX=Y specimens"
    mapping.append(dict(study=STUDY, source_file=metadata_source_relative(SOURCE, ROOT).as_posix(), source_sheet_or_table="Supplemental Table S1, pages 1-2", source_field=original, target_table="sample_metadata", target_field=target, join_keys="TMAMap = author Parent", value_mapping=rule, transformation_rule="PDF table columns extracted by fixed page coordinates; both pages visually checked", time_origin_rule="Reported month values preserved; source does not define an endpoint origin" if "months" in original else "", conflict_rule="Published clinical table", notes="Extension *_reported preserves the published category or value; *_months is numeric months; age_at_diagnosis_years is numeric years. PatientID is an original source identifier; patient_key remains empty pending cross-specimen linkage."))
mapping.append(dict(study=STUDY, source_file=metadata_source_relative(FACTS / "pmc13041199_fulltext.xml", ROOT).as_posix(), source_sheet_or_table="Methods: cohort and tissue microarrays", source_field="Neuroblastoma FFPE tumor specimens", target_table="sample_metadata", target_field="primary_diagnosis;preservation_method;case_control_status", join_keys="study", value_mapping="Neuroblastoma;ffpe;case", transformation_rule="Study-level source description", time_origin_rule="", conflict_rule="Source study methods", notes="29 CODEX specimens; manuscript reports 27 patients. Clinical table alone does not establish the paired patient crosswalk."))
OUT.mkdir(parents=True, exist_ok=True)
source.to_csv(OUT / "clinical_source_table.csv", index=False, encoding="utf-8-sig")
samples.to_csv(OUT / "sample_metadata.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(mapping, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns).to_csv(OUT / "clinical_field_mapping.csv", index=False, encoding="utf-8-sig")
publish_metadata_sources(FACTS)
print(f"Clinical source rows={len(source)}; CODEX samples={len(samples)}; output={OUT}")
