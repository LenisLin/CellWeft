"""Prepare the public SPoRA patient and tissue table for the Borner cohort."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.source_metadata import metadata_source_relative


STUDY = "Borner2025_CODEX_MultiTissue"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
OUTPUT = ROOT / "processed" / STUDY
SOURCE = ROOT / "raw/public/hubmap2019human/spora/metadata/tissues.parquet"
ALIAS_RECEIPT = PROJECT / "cache/two_node_workflow_20261024/Borner_native_alias_20261028T2020Z/root_source_scope_check.json"
SPORA_DATASET = "hubmap2019human"
BIA_ACCESSION = "BIA-SPORA-035"
BIA_URL = "https://spora.bioimagearchive.org/bioimage-archive/projects/spora/dataset"
FORMAT_URL = "https://spora.epfl.ch/docs/io/format.html"
PAPER_URL = "https://www.nature.com/articles/s41592-024-02563-5"
PREPARER = f"{STUDY}:metadata_only"
CLINICAL_COLUMNS = {
    "age_at_collection_years", "age_at_index_years", "age_at_diagnosis_years",
    "sex_reported", "bmi_kg_m2", "ecog_at_index", "primary_diagnosis",
    "primary_diagnosis_code", "diagnosis_code_system", "primary_site", "histology",
    "stage_group_at_index", "stage_system", "stage_basis", "tnm_t", "tnm_n",
    "tnm_m", "grade", "molecular_subtype", "prior_systemic_therapy",
    "prior_therapy_lines", "index_therapy_modality", "index_regimen",
    "response_criteria", "best_overall_response", "vital_status", "os_time",
    "os_event", "pfs_time", "pfs_event", "dfs_time", "dfs_event",
}
SOURCE_EXTENSIONS = (
    ("patient_id", "source_spora_patient_id", "Pseudonymous patient hash in SPoRA."),
    ("specimen_id", "source_spora_specimen_id", "SPoRA specimen index within a patient."),
    ("modality", "source_spora_modality", "SPoRA imaging modality."),
    ("original_file_name", "source_spora_original_file_name",
     "Original filename recorded by SPoRA; not used as a unique sample key."),
    ("split", "source_spora_split", "SPoRA train/test split label."),
    ("tissue_id_old", "source_spora_tissue_id_old", "Prior SPoRA tissue ID."),
)
METADATA_NOTE = (
    "SPoRA public tissue row. The source provides pseudonymous patient and specimen "
    "identifiers but no clinical covariates. Exact BIA image namespace correspondence "
    "is recorded in the source evidence; original_file_name is not a unique sample key."
)


def relative(path: Path) -> str:
    return metadata_source_relative(path, ROOT).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def validate_aliases(tissue_ids: set[str]) -> None:
    evidence = json.loads(ALIAS_RECEIPT.read_text())
    aliases = evidence["namespace_correspondences"]
    expected_prefix = f"https://livingobjects.ebi.ac.uk/bioimaging-01-pub/spora-submissions/{SPORA_DATASET}/codex/"
    source_ids = {row["image_id"] for row in aliases}
    normalized_names = {
        row["native_name"].replace("borner2025collection_", f"{SPORA_DATASET}_", 1)
        for row in aliases
    }
    if (not evidence["namespace_alias_proven"] or len(aliases) != 80
            or len(source_ids) != 80 or len(normalized_names) != 80
            or source_ids != tissue_ids or normalized_names != tissue_ids
            or any(not row["source_url"].startswith(expected_prefix) for row in aliases)):
        raise ValueError("The Borner-to-SPoRA public tissue correspondence changed")


def source_table() -> pd.DataFrame:
    with formal_read_lease(SOURCE, data_root=ROOT):
        data = pd.read_parquet(SOURCE)
    required = {
        "dataset", "tissue_id", "patient_id", "specimen_id", "modality",
        "original_patient_identifier", "original_file_name", "split", "tissue_id_old",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"SPoRA tissues.parquet is missing fields: {sorted(required - set(data.columns))}")
    if (len(data) != 80 or data.tissue_id.isna().any() or data.tissue_id.duplicated().any()
            or data.dataset.ne(SPORA_DATASET).any() or data.modality.ne("codex").any()
            or data.patient_id.isna().any() or data.original_patient_identifier.isna().any()
            or data.specimen_id.isna().any() or data.original_file_name.isna().any()):
        raise ValueError("SPoRA tissue metadata no longer matches the reviewed 80-row Borner scope")
    if (data.patient_id.nunique() != 15
            or data.groupby("patient_id").original_patient_identifier.nunique().ne(1).any()):
        raise ValueError("SPoRA patient identifiers do not map consistently")
    validate_aliases(set(data.tissue_id.astype(str)))
    return data


def sample_metadata(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for source in data.to_dict(orient="records"):
        tissue_id = str(source["tissue_id"])
        patient_id = str(source["patient_id"])
        row = dict.fromkeys(columns("sample_metadata"), "")
        row.update(
            study=STUDY,
            sample_id=tissue_id,
            patient_key=f"{STUDY}::{patient_id}",
            source_subject_id=str(source["original_patient_identifier"]),
            cohort_id=str(source["dataset"]),
            source_sample_id=tissue_id,
            metadata_notes=METADATA_NOTE,
        )
        for source_field, target_field, _ in SOURCE_EXTENSIONS:
            value = source[source_field]
            row[target_field] = "" if pd.isna(value) else str(value)
        records.append(row)
    extension_columns = [target for _, target, _ in SOURCE_EXTENSIONS]
    result = pd.DataFrame(records, columns=columns("sample_metadata") + extension_columns)
    if (len(result) != 80 or result.sample_id.duplicated().any()
            or result.patient_key.nunique() != 15 or result.patient_key.isna().any()):
        raise ValueError("Generated Borner sample rows failed identity checks")
    populated_clinical = [column for column in CLINICAL_COLUMNS
                          if column in result and result[column].fillna("").ne("").any()]
    if populated_clinical:
        raise ValueError(f"Unexpected clinical covariates in source table: {populated_clinical}")
    return result


def mapping(source_file: str, source_field: str, target_field: str, rule: str,
            *, notes: str = "", source_table: str = "tissues.parquet",
            join_keys: str = "Exact source tissue_id row") -> dict[str, str]:
    return dict(
        study=STUDY,
        source_file=source_file,
        source_sheet_or_table=source_table,
        source_field=source_field,
        target_table="sample_metadata",
        target_field=target_field,
        join_keys=join_keys,
        value_mapping="{}",
        transformation_rule=rule,
        time_origin_rule="",
        conflict_rule="Preserve source values; leave unreported clinical fields blank.",
        notes=notes,
    )


def field_mapping() -> pd.DataFrame:
    source_file = relative(SOURCE)
    script_file = "scripts/data/preparation/borner2025_codex_multitissue.py"
    tissue_join = (
        "SPoRA tissue_id; exact 80/80 BIA image namespace correspondence "
        "borner2025collection_* -> hubmap2019human_*"
    )
    rows = [
        mapping(script_file, "STUDY", "study",
                "Assign the canonical CellWeft dataset identifier.",
                notes="Internal project identifier; not a source clinical value.",
                source_table="preparation script", join_keys=""),
        mapping(source_file, "tissue_id", "sample_id",
                "Copy the unique SPoRA tissue ID as the project sample key.",
                notes="The public tissue ID, not original_file_name, defines the row identity.",
                join_keys=tissue_join),
        mapping(source_file, "tissue_id", "source_sample_id",
                "Preserve the public SPoRA tissue ID as the available source sample key.",
                notes="No separate original biospecimen accession is present in tissues.parquet.",
                join_keys=tissue_join),
        mapping(source_file, "dataset", "cohort_id",
                "Copy the source SPoRA cohort identifier."),
        mapping(source_file, "patient_id", "patient_key",
                f"Namespace the pseudonymous patient hash with {STUDY}.",
                notes="SPoRA defines patient_id as a pseudonymous hash of the source patient identifier."),
        mapping(source_file, "original_patient_identifier", "source_subject_id",
                "Copy the source patient identifier recorded by SPoRA.",
                notes="One source patient identifier maps to each SPoRA patient hash."),
        mapping(script_file, "generated provenance note", "metadata_notes",
                "Record source scope, available patient identity fields, and the absence of clinical covariates.",
                notes="Generated annotation; it does not add a clinical value.",
                source_table="preparation script", join_keys=""),
    ]
    for source_field, target_field, definition in SOURCE_EXTENSIONS:
        rows.append(mapping(
            source_file,
            source_field,
            target_field,
            "Preserve the source value in a typed SPoRA extension.",
            notes=json.dumps({"data_type": "string", "definition": definition}, ensure_ascii=True),
        ))
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def preparer_reviews(data: pd.DataFrame, samples: pd.DataFrame,
                     fields: pd.DataFrame) -> pd.DataFrame:
    common = dict(
        study=STUDY,
        preparer=PREPARER,
        reviewer="",
        record_id=STUDY,
        source_evidence=";".join((relative(SOURCE), str(ALIAS_RECEIPT.relative_to(PROJECT)),
                                  BIA_URL, FORMAT_URL, PAPER_URL)),
        output_evidence=f"processed/{STUDY}/sample_metadata.csv;processed/{STUDY}/field_mapping.csv",
    )
    details = (
        f"Mapped {len(samples)} public tissue rows to {samples.patient_key.nunique()} patients. "
        "All 80 sample keys match the official BIA image namespace receipt. The source table "
        "contains no clinical covariates; those template fields remain blank. No local image "
        "table was created or modified; independent review pending."
    )
    rows = [dict(common, scope=scope, check_id=check_id, result=result, details=details)
            for scope in ("study", "sample")
            for check_id, result in (("M01", "pass"), ("M02", "pass"))]
    return pd.DataFrame(rows, columns=columns("review_results"))


def publish_frame(path: Path, frame: pd.DataFrame) -> None:
    with formal_publish_lease(path, directory=OUTPUT, data_root=ROOT):
        temporary = path.with_suffix(path.suffix + ".partial")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


def metadata_only() -> None:
    data = source_table()
    samples = sample_metadata(data)
    fields = field_mapping()
    review_path = OUTPUT / "review_results.csv"
    with formal_read_lease(review_path, data_root=ROOT):
        prior = pd.read_csv(review_path, keep_default_na=False)
    incoming = preparer_reviews(data, samples, fields)
    replace = (prior.preparer.eq(PREPARER) & prior.reviewer.eq("")
               & prior.scope.isin({"study", "sample"})
               & prior.check_id.isin({"M01", "M02"}))
    merged_reviews = pd.concat([prior.loc[~replace], incoming], ignore_index=True)
    publish_frame(OUTPUT / "sample_metadata.csv", samples)
    publish_frame(OUTPUT / "field_mapping.csv", fields)
    publish_frame(review_path, merged_reviews)
    print("published 80 public tissue records for 15 patients; clinical covariates remain absent", flush=True)


if __name__ == "__main__":
    metadata_only()
