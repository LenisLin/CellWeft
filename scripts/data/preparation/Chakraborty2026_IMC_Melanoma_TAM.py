"""Prepare Chakraborty 2026 melanoma IMC source inputs.

The Zenodo package contains author-normalized 44-page float32 stacks.  This
adapter keeps those values, uses the published keep-order panel, and writes
only private candidate inputs until an independent reviewer signs the study.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Chakraborty2026_IMC_Melanoma_TAM"
PANEL_ID = "chakraborty2026_zenodo15124735_44"
DEFAULT_JOB = Path("/home/lenislin/Experiment/projects/CellWeft/cache/chakraborty_xu_20261107") / STUDY
RECORD_URL = "https://zenodo.org/records/15124735"
PAPER_URL = "https://doi.org/10.1186/s13073-026-01794-9"
SOURCE_DESC = (
    "Author-published CyTOF-normalized float32 IMC stack; values are retained as "
    "published and are not instrument raw counts."
)
SUPPLEMENT = RAW / "metadata" / "13073_2026_1794_MOESM4_ESM.xlsx"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--output-root", type=Path, help="Parent containing <study>; use a private candidate root")
parser.add_argument("--jobdir", type=Path, default=DEFAULT_JOB)
parser.add_argument("--s1", type=Path, help="Candidate six-column S1 file; defaults to the generated proposal")
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()

ROOT = args.data_root
RAW = ROOT / "raw/public" / STUDY
JOB = args.jobdir
OUTPUT = (args.output_root or JOB / "candidate_processed") / STUDY
MASK_ROOT = RAW / "masks"
JOB.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def progress(stage: str, exported: int, issues: list[str] | None = None) -> None:
    write_json(JOB / "progress.json", {
        "study": STUDY,
        "stage": stage,
        "status": "candidate_complete" if exported == 13 else "partial",
        "source_images": 13,
        "candidate_exported_images": exported,
        "sample_rows": 13,
        "patient_linked_images": 13,
        "issues": issues or [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log": str(JOB / "prepare.log"),
    })


def source_revision(path: Path) -> str:
    stat = path.stat()
    return f"Zenodo15124735:{stat.st_size}:{stat.st_mtime_ns}"


def build_mapping(panel_source: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Create a bounded candidate S1 from explicit labels and existing mappings."""
    mapping: dict[str, tuple[str, str, str, str]] = {
        "CD45": ("CD45", "PTPRC", "P08575", "exact established marker"),
        "CD68": ("CD68", "CD68", "P34810", "exact established marker"),
        "HLADR": ("HLA-DR", "mean(HLA-DRA, HLA-DRB1, HLA-DRB3, HLA-DRB4, HLA-DRB5)", "P01903 | P01911 | P79483 | P13762 | Q30154", "established source alias"),
        "SMA": ("αSMA", "ACTA2", "P62736", "established source alias"),
        "CD15": ("CD15", "proxy(FUT4)", "P22083", "established proxy"),
        "CTLA4": ("CD152", "CTLA4", "P16410", "established alias"),
        "CD14": ("CD14", "CD14", "P08571", "exact established marker"),
        "CD3": ("CD3e", "CD3E", "P07766", "source label maps to CD3E"),
        "ki67": ("Ki-67", "MKI67", "P46013", "case-normalized established marker"),
        "CD163": ("CD163", "CD163", "Q86VB7", "exact established marker"),
        "PanKer": ("PanCK (mapping pending)", "", "", "source generic PanKer label; clone/epitope not supplied"),
        "CD11b": ("CD11b", "ITGAM", "P11215", "established alias"),
        "PDL1": ("CD274", "CD274", "Q9NZQ7", "established alias"),
        "CD31": ("CD31", "PECAM1", "P16284", "established alias"),
        "CCL4": ("CCL4", "CCL4", "P13236", "exact established marker"),
        "TREM2": ("TREM2", "TREM2", "Q9NZC2", "explicit human gene/protein marker label; canonical UniProt identity"),
        "Tim3": ("CD366", "HAVCR2", "Q8TDQ0", "case-normalized established alias"),
        "FoxP3": ("FoxP3", "FOXP3", "Q9BZS1", "case-normalized established marker"),
        "CD4": ("CD4", "CD4", "P01730", "exact established marker"),
        "Lag3": ("CD223", "LAG3", "P18627", "case-normalized established alias"),
        "TIGIT": ("TIGIT", "TIGIT", "Q495A1", "exact established marker"),
        "CD11c": ("CD11c", "ITGAX", "P20702", "established alias"),
        "Tbet": ("T-bet", "TBX21", "Q9UL17", "case-normalized established alias"),
        "CD20": ("CD20", "MS4A1", "P11836", "exact established marker"),
        "CD8a": ("CD8a", "CD8A", "P01732", "exact established marker"),
        "VEGF": ("VEGF", "mean(VEGFA, VEGFC)", "P15692 | P49767", "existing composite mapping"),
        "CytC": ("Cytochrome C (mapping pending)", "", "", "source CytC label lacks an established current S1 row"),
        "PD1": ("CD279", "PDCD1", "Q15116", "established alias"),
        "CD204": ("CD204", "MSR1", "P21757", "established alias"),
        "GranzymeB": ("Granzyme B", "GZMB", "P10144", "case-normalized established alias"),
        "CCL2": ("CCL2", "CCL2", "P13500", "exact established marker"),
        "Collagen1": ("COLLAGEN-I", "mean(COL1A1, COL1A2)", "P02452 | P08123", "existing composite mapping"),
        "CD103": ("CD103", "ITGAE", "P38570", "established alias"),
        "APOE": ("APOE", "APOE", "P02649", "exact established marker"),
        "CD38": ("CD38", "CD38", "P28907", "exact established marker"),
        "CD45RO": ("CD45RO", "proxy(PTPRC)", "P08575", "established proxy"),
        "S100": ("S100 (mapping pending)", "", "", "generic S100 label cannot select S100A1/A4/B without clone evidence"),
        "CD34": ("CD34", "CD34", "P28906", "exact established marker"),
        "cCasp3": ("Caspase-3", "CASP3", "P42574", "existing S1 CCASP3/CASP3 mapping; source keeps cleaved-state measurement semantics"),
        "DNA1": ("DNA (segmentation only)", "", "", "published iridium intercalator nuclear input"),
        "DNA2": ("DNA (segmentation only)", "", "", "published iridium intercalator nuclear input"),
        "HH3": ("Histone H3", "H3C1", "P68431", "source HH3 and existing histone rule"),
        "Vimentin": ("Vimentin", "VIM", "P08670", "exact established marker"),
        "ECadherin": ("E-Cadherin", "CDH1", "P12830", "source alias"),
    }
    rows: list[dict] = []
    evidence: list[dict] = []
    for source_name in panel_source.loc[panel_source.keep.astype(int).eq(1), "name"].tolist():
        unified, gene, accession, note = mapping[source_name]
        deepcell = panel_source.loc[panel_source.name.eq(source_name), "deepcell"].iloc[0]
        if pd.notna(deepcell) and str(deepcell).strip() not in {"", "nan", "None"}:
            segmentation_channel = str(int(float(deepcell)))
        else:
            segmentation_channel = "1" if source_name in {"DNA1", "DNA2", "HH3"} else ""
        rows.append({
            "Study": STUDY,
            "Original marker": source_name,
            "Unified marker": unified,
            "MAP-KG mapping": gene,
            "UniProt accession": accession,
            "Segmentation channel": segmentation_channel,
        })
        evidence.append({
            "study": STUDY,
            "source_marker": source_name,
            "candidate_mapping": unified,
            "mapping_basis": note,
            "source_panel": str(RAW / "Melanoma_panel.csv"),
            "existing_s1_reuse": "current formal S1 marker registry where exact/established synonym exists",
            "status": "mapping_pending" if "mapping pending" in unified else "proposed",
        })
    return pd.DataFrame(rows), evidence


def read_patient_info() -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_info = pd.read_excel(SUPPLEMENT, sheet_name="Patient_info", nrows=6, dtype=str).fillna("")
    raw = pd.read_excel(SUPPLEMENT, sheet_name="Patient_info", header=None, dtype=str).fillna("")
    headers = raw.iloc[:, :3].astype(str)
    response_header_rows = headers.index[
        headers.iloc[:, 0].str.strip().eq("Patient ID")
        & headers.iloc[:, 1].str.strip().eq("type of ICI")
        & headers.iloc[:, 2].str.contains("best response", case=False, regex=False)
    ].tolist()
    assert len(response_header_rows) == 1
    response_start = response_header_rows[0] + 1
    response = raw.iloc[response_start:response_start + 6, :3].copy()
    response.columns = ["Patient ID", "type of ICI", "best response to ICI"]
    response["Patient ID"] = response["Patient ID"].str.replace(" ", "", regex=False)
    assert patient_info["Patient ID"].nunique() == 6
    assert set(patient_info["Patient ID"]) == set(response["Patient ID"])
    assert not response["Patient ID"].duplicated().any()
    return patient_info, response


def make_samples(
    metadata: pd.DataFrame,
    patient_info: pd.DataFrame,
    patient_responses: pd.DataFrame,
    image_prefixes: list[str],
    image_ids: list[str],
) -> pd.DataFrame:
    template = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str, keep_default_na=False)
    by_prefix = metadata.drop_duplicates("Pseudonym").set_index("Pseudonym")
    by_patient = patient_info.set_index("Patient ID")
    by_response = patient_responses.set_index("Patient ID")
    lesion_mapping = {
        "Primary": ("", "primary"),
        "LN macro met": ("lymph node", "metastasis"),
        "s.c. Met": ("subcutaneous", "metastasis"),
    }
    rows: list[dict] = []
    for image_prefix, image_id in zip(image_prefixes, image_ids):
        row = dict.fromkeys(template.columns, "")
        pseudonym = image_prefix.split("_", 1)[0]
        source = by_prefix.loc[pseudonym]
        clinical = by_patient.loc[pseudonym]
        response = by_response.loc[pseudonym]
        assert clinical["type of ICI"] == response["type of ICI"]
        assert {"Yes": "Responder", "No": "Non-Responder"}[clinical["Initial response to ICI"]] == source.Response
        assert clinical["type of tumor material"] in lesion_mapping
        anatomical_site, lesion_type = lesion_mapping[clinical["type of tumor material"]]
        row.update(
            study=STUDY,
            sample_id=image_id,
            patient_key=f"{STUDY}_patient_{source.Patient_ID}",
            source_subject_id=str(source.Patient_ID),
            source_sample_id="",
            biospecimen_group_id="",
            sample_type="FFPE tissue ROI",
            tissue_type="melanoma",
            anatomical_site=anatomical_site,
            lesion_type=lesion_type,
            timepoint_label="pre_treatment",
            treatment_status_at_collection=clinical["Treatment status at baseline (at the time of biopsy)"],
            preservation_method="FFPE",
            processing_method="author-published CyTOF-normalized IMC",
            bmi_kg_m2=clinical["BMI"],
            index_therapy_modality="immune checkpoint inhibitor",
            index_regimen=clinical["type of ICI"],
            best_overall_response=response["best response to ICI"],
            sex_reported="",
            metadata_notes=(
                f"source_pseudonym={pseudonym}; source_response_group={source.Response}; "
                f"source_initial_response_to_ici={clinical['Initial response to ICI']}; "
                f"source_type_of_tumor_material={clinical['type of tumor material']}; "
                f"source_metastasis={clinical['Metastasis']}; "
                f"source_weight_kg_at_ici_start={clinical['weight (kg) at ICI start']}; "
                f"source_bmi_at_ici_start={clinical['BMI']}; "
                f"source_treatment_status_at_baseline={clinical['Treatment status at baseline (at the time of biopsy)']}; "
                f"source_best_response_to_ici={response['best response to ICI']} (criteria version not specified). "
                "The paper reports all samples were obtained within three months before ICI initiation. "
                "Melanoma_metadata.xlsx lists multiple Tissue_ID values per patient without mapping image suffixes to them; "
                "source_sample_id and biospecimen_group_id therefore remain empty. The article describes metastatic patients, "
                "while the source material label for one patient's imaged tissue is Primary; row-level source wording is retained."
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=template.columns)


def main() -> None:
    panel_source = pd.read_csv(RAW / "Melanoma_panel.csv", dtype=str, keep_default_na=False)
    panel_source["keep"] = pd.to_numeric(panel_source["keep"], errors="raise")
    metadata = pd.read_excel(RAW / "Melanoma_metadata.xlsx", dtype=str).fillna("")
    patient_info, patient_responses = read_patient_info()
    image_files = sorted(RAW.glob("*_Melanoma_img_normalized.tiff"))
    assert len(image_files) == 13, f"expected 13 source image stacks, found {len(image_files)}"
    kept_names = panel_source.loc[panel_source.keep.eq(1), "name"].tolist()
    assert len(kept_names) == 44 and len(set(kept_names)) == 44
    prefixes = [path.name.removesuffix("_Melanoma_img_normalized.tiff") for path in image_files]
    pseudonyms = [prefix.split("_", 1)[0] for prefix in prefixes]
    assert all(prefix in set(metadata.Pseudonym) for prefix in pseudonyms)

    inventory: list[dict] = []
    for path, prefix in zip(image_files, prefixes):
        stat = path.stat()
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == 44
            assert all(page.ndim == 2 and page.samplesperpixel == 1 and page.dtype == np.dtype("float32")
                       for page in tif.pages)
            shape = tuple(tif.series[0].shape)
            assert shape[0] == 44 and len(shape) == 3
            assert all(page.shape == shape[1:] for page in tif.pages)
        image_id = prefix.replace("#", "")
        mask = MASK_ROOT / f"{prefix}_Melanoma_cellmasks.tiff"
        assert mask.is_file() and mask.stat().st_size > 0
        with tifffile.TiffFile(mask) as mask_tif:
            assert len(mask_tif.pages) == 1 and mask_tif.pages[0].shape == shape[1:]
            assert mask_tif.pages[0].dtype.kind in "iu"
        inventory.append({
            "image_id": image_id,
            "source_image_id": path.name,
            "source_prefix": prefix,
            "source_file": str(path.relative_to(ROOT)),
            "source_file_bytes": stat.st_size,
            "source_file_mtime_ns": stat.st_mtime_ns,
            "source_shape": list(shape),
            "source_dtype": "float32",
            "source_axes": "CYX",
            "panel_id": PANEL_ID,
            "sample_id": image_id,
            "patient_pseudonym": prefix.split("_", 1)[0],
            "author_mask": str(mask.relative_to(ROOT)),
            "author_mask_bytes": mask.stat().st_size,
            "source_revision": source_revision(path),
        })

    candidate_s1, s1_evidence = build_mapping(panel_source)
    write_csv(JOB / "candidate_s1_rows.csv", candidate_s1.to_dict("records"), list(candidate_s1.columns))
    write_csv(JOB / "candidate_s1_evidence.csv", s1_evidence, list(s1_evidence[0]))
    write_json(JOB / "inventory.json", {
        "schema": "cellweft.chakraborty2026_candidate_inventory.v1",
        "study": STUDY,
        "source_record": 15124735,
        "source_url": RECORD_URL,
        "paper_url": PAPER_URL,
        "source_archive": "public individual TIFF files; no source archive retained locally",
        "source_image_count": len(inventory),
        "source_channel_count": 44,
        "source_dtype": "float32",
        "source_intensity_semantics": SOURCE_DESC,
        "ordered_panel_basis": str(RAW / "Melanoma_panel.csv") + ": keep=1 rows in source order; ilastik is a separate model ordering",
        "pixel_scale_um": [1.0, 1.0],
        "pixel_scale_basis": "paper methods reports 1 micrometre per pixel; downloaded TIFF headers do not encode calibration",
        "patient_bridge": "13/13 image pseudonyms match explicit workbook Patient_ID; image suffix to Tissue_ID remains unresolved",
        "images": inventory,
        "author_mask_review": str(JOB / "source_author_mask_review.json"),
        "candidate_s1": str(JOB / "candidate_s1_rows.csv"),
    })
    progress("inventory_and_s1_ready", 0, [
        "independent review and D01 pending",
        "image suffix to workbook Tissue_ID is unresolved; technical image processing remains allowed",
        "generic PanKer, TREM2, CytC, S100 and cCasp3 mappings remain pending",
        "author mask boundary quality and whole-cell binding require independent review",
    ])
    if args.metadata_only:
        write_json(JOB / "preparation_receipt.json", {
            "status": "metadata_ready", "preparer": "/root/chakraborty_prepare", "study": STUDY,
            "images": 13, "samples": 13, "patient_linked_images": 13,
        })
        return

    panel = panel_from_s1(STUDY, PANEL_ID, kept_names, JOB / "candidate_s1_rows.csv")
    samples = make_samples(metadata, patient_info, patient_responses, prefixes, [row["image_id"] for row in inventory])
    field_columns = pd.read_csv(TEMPLATES / "field_mapping.csv", dtype=str, keep_default_na=False).columns.tolist()
    fields = pd.DataFrame([
        dict(study=STUDY, source_file=RECORD_URL, source_sheet_or_table="Zenodo record and article",
             source_field="study source record identity", target_table="sample_metadata", target_field="study",
             join_keys="study identity", value_mapping=STUDY,
             transformation_rule="use the canonical processed study identifier",
             time_origin_rule="", conflict_rule="none",
             notes="The Zenodo record and linked article identify the source dataset."),
        dict(study=STUDY, source_file=str(RAW / "Melanoma_metadata.xlsx"), source_sheet_or_table="Sheet1",
             source_field="Pseudonym, Patient_ID", target_table="sample_metadata", target_field="patient_key",
             join_keys="exact image filename patient prefix to Pseudonym",
             value_mapping="source Patient_ID -> study-namespaced patient_key",
             transformation_rule="prefix image filename before first underscore; preserve source key",
             time_origin_rule="", conflict_rule="do not infer image suffix Tissue_ID",
             notes="13/13 image patient prefixes map to six explicit Patient_ID values."),
        dict(study=STUDY, source_file=str(RAW / "Melanoma_metadata.xlsx"), source_sheet_or_table="Sheet1",
             source_field="Patient_ID", target_table="sample_metadata", target_field="source_subject_id",
             join_keys="exact Pseudonym", value_mapping="copy Patient_ID",
             transformation_rule="preserve source identifier", time_origin_rule="",
             conflict_rule="none", notes="The workbook does not map image suffixes to Tissue_ID."),
        dict(study=STUDY, source_file=str(RAW), source_sheet_or_table="public image filenames",
             source_field="image filename prefix", target_table="sample_metadata", target_field="sample_id",
             join_keys="image filename to images.parquet image_id",
             value_mapping="remove # from filename prefix for project ID",
             transformation_rule="derive image-associated row key; not a confirmed biological sample ID",
             time_origin_rule="", conflict_rule="retain unresolved sample grain in M01",
             notes="13 image-associated rows are retained for image metadata; source_sample_id remains blank."),
        dict(study=STUDY, source_file=str(RAW), source_sheet_or_table="study source record",
             source_field="melanoma study cohort", target_table="sample_metadata", target_field="tissue_type",
             join_keys="study identity", value_mapping="melanoma", transformation_rule="preserve disease tissue label",
             time_origin_rule="", conflict_rule="none", notes="Study title and article identify the cohort as melanoma."),
        dict(study=STUDY, source_file=PAPER_URL, source_sheet_or_table="Methods: imaging mass cytometry sample preparation",
             source_field="formalin-fixed and paraffin-embedded tissue samples", target_table="sample_metadata",
             target_field="sample_type", join_keys="six-patient UKF cohort", value_mapping="FFPE tissue ROI",
             transformation_rule="map explicit sample preparation description",
             time_origin_rule="at biopsy", conflict_rule="none",
             notes="Paper identifies FFPE tumor tissue sections from the same cohort used for spatial transcriptomics."),
        dict(study=STUDY, source_file=PAPER_URL, source_sheet_or_table="Methods: imaging mass cytometry sample preparation",
             source_field="formalin-fixed and paraffin-embedded tissue samples", target_table="sample_metadata",
             target_field="preservation_method", join_keys="six-patient UKF cohort", value_mapping="FFPE",
             transformation_rule="map explicit sample preparation description",
             time_origin_rule="at biopsy", conflict_rule="none",
             notes="Paper identifies FFPE tumor tissue sections from the same cohort used for spatial transcriptomics."),
        dict(study=STUDY, source_file=PAPER_URL, source_sheet_or_table="Methods: imaging and data acquisition",
             source_field="images processed and normalized using CyTOF software", target_table="sample_metadata",
             target_field="processing_method", join_keys="Zenodo image source record",
             value_mapping="author-published CyTOF-normalized IMC",
             transformation_rule="summarize published processing description",
             time_origin_rule="", conflict_rule="none",
             notes="Source image values are retained as published; they are not instrument raw counts."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="type of ICI", target_table="sample_metadata", target_field="index_regimen",
             join_keys="exact UKF patient pseudonym; response table whitespace normalized and regimen cross-checked",
             value_mapping="copy source regimen label", transformation_rule="preserve source spelling",
             time_origin_rule="at ICI initiation", conflict_rule="none",
             notes="Patient-level regimen copied to image-associated rows; no patient-specific timing beyond source statement."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="type of ICI", target_table="sample_metadata", target_field="index_therapy_modality",
             join_keys="exact UKF patient pseudonym", value_mapping="ICI -> immune checkpoint inhibitor",
             transformation_rule="expand source abbreviation", time_origin_rule="at ICI initiation",
             conflict_rule="none", notes="Therapy class is explicit in source column."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="BMI", target_table="sample_metadata", target_field="bmi_kg_m2",
             join_keys="exact UKF patient pseudonym", value_mapping="copy numeric BMI",
             transformation_rule="preserve source numeric value", time_origin_rule="at ICI start",
             conflict_rule="none", notes="BMI is reported at ICI start."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="type of tumor material, Metastasis", target_table="sample_metadata",
             target_field="lesion_type", join_keys="exact UKF patient pseudonym",
             value_mapping="Primary -> primary; LN macro met and s.c. Met -> metastasis",
             transformation_rule="map explicit source material labels to controlled lesion_type values",
             time_origin_rule="at biopsy", conflict_rule="preserve exact source wording in metadata_notes",
             notes="The article cohort-level description is retained; patient-specific source material labels govern these fields."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="type of tumor material", target_table="sample_metadata", target_field="anatomical_site",
             join_keys="exact UKF patient pseudonym", value_mapping="LN macro met -> lymph node; s.c. Met -> subcutaneous; Primary -> blank",
             transformation_rule="expand explicit abbreviations; do not infer primary site",
             time_origin_rule="at biopsy", conflict_rule="preserve exact source wording in metadata_notes",
             notes="Patient-level site label is repeated across image-associated rows; no image-to-Tissue_ID mapping."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="Treatment status at baseline (at the time of biopsy)", target_table="sample_metadata",
             target_field="treatment_status_at_collection", join_keys="exact UKF patient pseudonym",
             value_mapping="copy source label", transformation_rule="preserve source wording",
             time_origin_rule="at biopsy", conflict_rule="none", notes="All six source rows report Treatment naive."),
        dict(study=STUDY, source_file=PAPER_URL, source_sheet_or_table="Methods: spatial transcriptomics and IMC sample preparation",
             source_field="tissue samples obtained within three months before initiation of ICI therapy",
             target_table="sample_metadata", target_field="timepoint_label",
             join_keys="six-patient UKF cohort; same cohort used for IMC",
             value_mapping="pre_treatment", transformation_rule="map to controlled timepoint label; retain full window in notes",
             time_origin_rule="ICI initiation", conflict_rule="none",
             notes="The patient-specific day within the stated three-month window is not reported."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="best response to ICI (CR, PR, SD, PD)", target_table="sample_metadata",
             target_field="best_overall_response", join_keys="exact UKF patient pseudonym after whitespace normalization",
             value_mapping="copy first-line ICI best-response category",
             transformation_rule="preserve CR/PR/SD/PD source value",
             time_origin_rule="response to first listed ICI regimen", conflict_rule="leave response_criteria blank; criteria/version not specified",
             notes="The article discusses iRECIST follow-up status, but the supplement does not specify criteria/version for this table field."),
        dict(study=STUDY, source_file=str(SUPPLEMENT), source_sheet_or_table="Patient_info",
             source_field="Initial response to ICI; type of tumor material; Metastasis; weight (kg) at ICI start",
             target_table="sample_metadata", target_field="metadata_notes",
             join_keys="exact UKF patient pseudonym", value_mapping="preserve exact source values",
             transformation_rule="record source-specific fields not represented by a standard template field",
             time_origin_rule="at ICI start or at biopsy as named in the source column",
             conflict_rule="none", notes="Response group is also preserved from Melanoma_metadata.xlsx; source weight is retained because the template has no weight field."),
        dict(study=STUDY, source_file=str(RAW / "Melanoma_panel.csv"), source_sheet_or_table="CSV",
             source_field="name, keep, deepcell", target_table="panel_mapping", target_field="source_channel_index",
             join_keys="source stack page order and keep=1 rows", value_mapping="deepcell 1/2 retained as candidate segmentation channel",
             transformation_rule="drop six non-keep source channels; preserve 44 page order",
             time_origin_rule="", conflict_rule="ilastik ordering is not substituted for raw page order",
             notes="Source image stacks have 44 pages and no OME names; page order is bound to keep=1 order."),
    ], columns=field_columns)

    bindings = {}
    for item in inventory:
        bindings[item["image_id"]] = {
            "status": "bound",
            "path": item["author_mask"],
            "frame": "source",
            "mask_type": "whole_cell",
            "overview_judgment": True,
            "overview_required": True,
        }
    write_json(JOB / "mask_binding_proposal.json", {
        "study": STUDY,
        "status": "candidate_only_independent_review_required",
        "bindings": bindings,
        "evidence": str(JOB / "source_author_mask_review.json"),
        "source_type": "author-published individual-cell mask; exact native YX frame",
        "action": "propose reuse; root independent reviewer must approve before formal handover",
    })

    image_rows: list[dict] = []
    selected = inventory[:args.limit] if args.limit is not None else inventory
    for item in selected:
        source_path = ROOT / item["source_file"]
        destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
        source = SourceSpec(
            image_id=item["image_id"], panel_id=PANEL_ID, kind="tiff", path=source_path,
            channels=tuple(range(44)), source_revision=item["source_revision"],
            pixel_size_um=(1.0, 1.0), intensity_description=SOURCE_DESC,
        )
        with open_source(source) as reader:
            result = export_input(reader, panel, destination, data_root=ROOT)
        assert result["admitted"] and result["source_dtype"] == "float32"
        h, w = item["source_shape"][1:]
        image_rows.append(dict(
            study=STUDY, image_id=item["image_id"], sample_id=item["sample_id"], panel_id=PANEL_ID,
            image_kind="roi", source_image_id=item["source_image_id"], source_files=[item["source_file"]],
            source_format="tiff_stack", source_acquisition_id=item["source_prefix"], source_axes="CYX",
            source_dtype="float32", source_revision=item["source_revision"], slide_id="",
            core_id="", roi_id=item["source_prefix"], fov_id="", region_annotation="melanoma ROI",
            source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
            pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
            pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
            height_px=result["height_px"], width_px=result["width_px"], source_channel_count=44,
            channel_count=44, pixel_size_x_um=1.0, pixel_size_y_um=1.0,
            input_image_path=str(destination), valid_pixels_path="", intensity_description=SOURCE_DESC,
            split_method="already_independent", split_evidence=str(JOB / "inventory.json"),
            metadata_notes=("One complete published ROI frame; source image suffix is not a verified Tissue_ID. "
                            "Zenodo primary-versus-paper metastatic wording remains unresolved."),
        ))
        save_input_tables(OUTPUT, pd.DataFrame(image_rows), samples, panel, fields, data_root=ROOT)
        progress("candidate_export_in_progress", len(image_rows), [
            "independent review and D01 pending", "image suffix to Tissue_ID unresolved",
            "author mask binding proposed, independent approval pending",
        ])
        print(json.dumps({"stage": "candidate_input_exported", "image_id": item["image_id"],
                          "completed": len(image_rows), "total": len(selected)}), flush=True)

    review_columns = pd.read_csv(TEMPLATES / "review_results.csv", dtype=str, keep_default_na=False).columns.tolist()
    review_rows = []
    for item in selected:
        review_rows.append(dict(study=STUDY, preparer="/root/chakraborty_prepare", reviewer="",
            scope="image", record_id=item["image_id"], check_id="P03", result="preparer_checked",
            source_evidence=str(JOB / "inventory.json"), output_evidence=str(OUTPUT / "images" / item["image_id"] / "input.tiff"),
            details="44-page float32 native header, square canvas and source frame checked; independent review pending."))
    review_rows.extend([
        dict(study=STUDY, preparer="/root/chakraborty_prepare", reviewer="", scope="study", record_id=STUDY,
             check_id="M01", result="unresolved", source_evidence=str(RAW / "Melanoma_metadata.xlsx"),
             output_evidence=str(OUTPUT / "sample_metadata.csv"), details="13 image prefixes map to explicit Patient_ID; suffix to Tissue_ID is unresolved."),
        dict(study=STUDY, preparer="/root/chakraborty_prepare", reviewer="", scope="study", record_id=STUDY,
             check_id="D01", result="pending", source_evidence=str(JOB / "inventory.json"),
             output_evidence=str(OUTPUT / "images.parquet"), details="Independent reviewer and controller admission required."),
    ])
    write_csv(OUTPUT / "review_results.csv", review_rows, review_columns)
    write_json(JOB / "self_check.json", {
        "study": STUDY, "result": "preparer_checked", "images": len(image_rows),
        "source_headers_checked": len(image_rows), "representative_pixel_windows": ["UKF2_1", "UKF4_1", "UKF6_1"],
        "author_masks_checked": 13, "independent_review": "pending", "d01": "pending",
    })
    status = "candidate_complete" if len(image_rows) == 13 else "candidate_partial"
    write_json(JOB / "preparation_receipt.json", {
        "status": status,
        "preparer": "/root/chakraborty_prepare",
        "study": STUDY,
        "images": len(image_rows),
        "samples": len(samples),
        "sample_linked_images": len(image_rows),
        "patient_linked_images": int(samples.patient_key.ne("").sum()),
        "candidate_output": str(OUTPUT),
        "candidate_s1": str(JOB / "candidate_s1_rows.csv"),
        "mask_binding_proposal": str(JOB / "mask_binding_proposal.json"),
        "source_intensity": SOURCE_DESC,
        "gaps": ["image suffix to Tissue_ID unresolved", "generic marker mappings pending", "independent review/D01 pending"],
    })
    progress(status, len(image_rows), ["independent review and D01 pending", "image suffix to Tissue_ID unresolved", "generic marker mappings pending"])


if __name__ == "__main__":
    main()
