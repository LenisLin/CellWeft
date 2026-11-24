"""Prepare the local S-BIAD2091 NASH-HCC TMA cores as native IMC inputs.

Inputs are the 43 complete TIFF/ordered-summary pairs and the study-specific S1
panel. Outputs are per-core native-intensity TIFFs and study input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd

from cellweft.data.contracts import SourceSpec
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease


STUDY = "gerber_biad2091_imc_mimic"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
RAW = ROOT / "raw/public" / STUDY / "S-BIAD2091"
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0714_prepare_0")))
MANIFEST = META / "native_manifest.json"
ORDER = META / "ordered_markers.json"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PAPER = "https://www.nature.com/articles/s42003-026-09961-9"
PUBLIC_SCRIPTS = META / "source_metadata/imc_to_ims_methods_analysis_scripts-master.zip"
SUPPLEMENT = META / "source_metadata/42003_2026_9961_MOESM2_ESM.pdf"
CLINICAL_SUPPLEMENT = META / "source_metadata/2023_Hepatology_HEP_H992_clinical_supplement.pdf"
PANEL_ID = "nash_41"
SOURCE_REVISION = "S-BIAD2091 NASH_HCC_TMA"
PREPARER = JOB.name
CLINICAL_PAPER = "https://doi.org/10.1097/HEP.0000000000000591"
CLINICAL_TABLE = "Table S1. Clinical information for all HCC patients"
CLINICAL_EXTENSION_FIELDS = (
    ("No.", "source_table_s1_no", "Patient row number in Table S1; not a source patient or specimen ID", ""),
    ("Sex", "source_table_s1_sex", "Sex as reported in Table S1", ""),
    ("Age", "source_table_s1_age", "Age as reported; source time origin is unspecified", "years"),
    ("BMI", "source_table_s1_bmi", "BMI as reported; source time origin is unspecified", "kg/m2"),
    ("Stage of liver fibrosis", "source_table_s1_liver_fibrosis_stage", "Liver fibrosis stage as reported", ""),
    ("TNM stage", "source_table_s1_tnm_stage", "TNM stage as reported, including any appended subgroup", ""),
    ("BCLC", "source_table_s1_bclc", "BCLC stage as reported", ""),
    ("HBsAg/HBV-DNA", "source_table_s1_hbsag_hbv_dna", "HBsAg/HBV-DNA result as reported", ""),
    ("Hepatitis C", "source_table_s1_hepatitis_c", "Hepatitis C result as reported", ""),
    ("AFP (ng/mL)", "source_table_s1_afp_ng_ml", "AFP value as reported", "ng/mL"),
    ("Diagnosis", "source_table_s1_diagnosis", "Diagnosis label as reported", ""),
    ("Steatosis (%)", "source_table_s1_steatosis_pct", "Steatosis percentage as reported", "%"),
    ("Lobular inflammation score", "source_table_s1_lobular_inflammation_score", "Lobular inflammation score as reported", ""),
    ("Hepatocyte ballooning degeneration score", "source_table_s1_hepatocyte_ballooning_score", "Hepatocyte ballooning degeneration score as reported", ""),
    ("Total ROIs for peri-tumour", "source_table_s1_rois_peri_tumour", "Peri-tumour ROI count as reported", "ROIs"),
    ("Total ROIs for margin", "source_table_s1_rois_margin", "Margin ROI count as reported", "ROIs"),
    ("Total ROIs for tumour", "source_table_s1_rois_tumour", "Tumour ROI count as reported", "ROIs"),
    ("ROI size", "source_table_s1_roi_size_um", "ROI width and height as reported", "um x um"),
)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, completed: int, next_image: str | None,
               sample_rows: int = 0, **details: object) -> None:
    payload = dict(study=STUDY, stage=stage, expected_images=43,
                   completed_images=completed, next_image=next_image,
                   sample_rows=sample_rows, sample_linked_images=0, patient_linked_samples=0,
                   scientific_repair_round=0,
                   updated_utc=pd.Timestamp.now(tz="UTC").isoformat(), **details)
    JOB.mkdir(parents=True, exist_ok=True)
    with formal_publish_lease(JOB / "progress.json", directory=JOB, data_root=ROOT):
        temporary = JOB / "progress.json.partial"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, JOB / "progress.json")


def mapping(source: Path | str, field: str, table: str, target: str,
            rule: str, *, notes: str = "", join: str = "", source_table: str = "",
            value_mapping: dict[str, str] | None = None) -> dict[str, str]:
    source_file = str(source) if isinstance(source, str) else relative(source)
    return dict(study=STUDY, source_file=source_file, source_sheet_or_table=source_table,
                source_field=field, target_table=table, target_field=target,
                join_keys=join, value_mapping=json.dumps(value_mapping or {}, ensure_ascii=False),
                transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


def clinical_samples() -> pd.DataFrame:
    result = subprocess.run(
        ["pdftotext", "-layout", "-f", "1", "-l", "1", str(CLINICAL_SUPPLEMENT), "-"],
        check=True, capture_output=True, text=True)
    rows: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        tokens = line.split()
        if not tokens or not re.fullmatch(r"\d{1,2}", tokens[0]):
            continue
        patient_no = int(tokens[0])
        if not 1 <= patient_no <= 16:
            continue
        if patient_no in rows:
            raise ValueError(f"Duplicate clinical Table S1 row: {patient_no}")
        if len(tokens) < 7 or tokens[1] not in {"Male", "Female"}:
            raise ValueError(f"Unexpected clinical Table S1 row: {line}")

        index = 3
        bmi = ""
        is_fibrosis = lambda value: bool(re.fullmatch(r"N|S\d(?:-S\d)?", value))
        is_number = lambda value: bool(re.fullmatch(r"\d+(?:\.\d+)?", value))
        if index + 1 < len(tokens) and is_number(tokens[index]) and is_fibrosis(tokens[index + 1]):
            bmi = tokens[index]
            index += 1
        if index >= len(tokens) or not is_fibrosis(tokens[index]):
            raise ValueError(f"Missing liver fibrosis value in Table S1 row {patient_no}")
        fibrosis = tokens[index]
        index += 1
        if index >= len(tokens) or not re.fullmatch(r"T\d+N\d+M\d+", tokens[index]):
            raise ValueError(f"Missing TNM stage in Table S1 row {patient_no}")
        tnm = tokens[index]
        index += 1
        if index < len(tokens) and re.fullmatch(r"I{1,3}[a-c]?", tokens[index]):
            tnm = f"{tnm} {tokens[index]}"
            index += 1
        bclc = tokens[index] if index < len(tokens) and tokens[index] in {"A", "B"} else ""
        if bclc:
            index += 1
        if len(tokens) - index != 13:
            raise ValueError(f"Unexpected field count in clinical Table S1 row {patient_no}: {tokens}")

        hbv, hepatitis_c, afp, diagnosis = tokens[index:index + 4]
        steatosis, inflammation, ballooning = tokens[index + 4:index + 7]
        peri_tumour, margin, tumour = tokens[index + 7:index + 10]
        roi_width, multiply, roi_height = tokens[index + 10:index + 13]
        if hbv not in {"+", "-"} or hepatitis_c not in {"+", "-"} or multiply != "×":
            raise ValueError(f"Unexpected marker or ROI-size value in Table S1 row {patient_no}")
        if diagnosis != "NASH-HCC":
            raise ValueError(f"Non-NASH record found in the local NASH-HCC subset: {patient_no}")

        source_values = {
            "No.": tokens[0], "Sex": tokens[1], "Age": tokens[2], "BMI": bmi,
            "Stage of liver fibrosis": fibrosis, "TNM stage": tnm, "BCLC": bclc,
            "HBsAg/HBV-DNA": hbv, "Hepatitis C": hepatitis_c, "AFP (ng/mL)": afp,
            "Diagnosis": diagnosis, "Steatosis (%)": steatosis,
            "Lobular inflammation score": inflammation,
            "Hepatocyte ballooning degeneration score": ballooning,
            "Total ROIs for peri-tumour": peri_tumour,
            "Total ROIs for margin": margin, "Total ROIs for tumour": tumour,
            "ROI size": f"{roi_width} {multiply} {roi_height}",
        }
        record_key = f"patient_record_{patient_no:02d}"
        row = dict.fromkeys(columns("sample_metadata"), "")
        row.update(study=STUDY, sample_id=record_key,
                   patient_key=f"{STUDY}:{record_key}", cohort_id=diagnosis,
                   sex_reported=tokens[1].lower(), primary_diagnosis=diagnosis,
                   metadata_notes=("Patient-level clinical row from the official Table S1. "
                                   "sample_id and patient_key are project patient-record keys "
                                   "derived from the table row number, not source biospecimen IDs. "
                                   "No published key links these rows to the local TMA cores. "
                                   "Age, BMI, stage and AFP time origins are unspecified."))
        row.update({target: source_values[source] for source, target, _, _ in CLINICAL_EXTENSION_FIELDS})
        rows[patient_no] = row

    if sorted(rows) != list(range(1, 17)):
        raise ValueError(f"Expected 16 NASH-HCC rows from Table S1, found {sorted(rows)}")
    sample_columns = columns("sample_metadata") + [target for _, target, _, _ in CLINICAL_EXTENSION_FIELDS]
    return pd.DataFrame([rows[index] for index in sorted(rows)], columns=sample_columns)


def source_inventory() -> tuple[list[dict], pd.DataFrame]:
    import tifffile
    from cellweft.data.preparation.study_input import panel_from_s1

    with formal_read_lease(MANIFEST, data_root=ROOT):
        manifest = json.loads(MANIFEST.read_text())
        ordered = json.loads(ORDER.read_text())
    sources = manifest["sources"]
    names = [item["source_name"] for item in ordered[PANEL_ID]["channels"]]
    if (len(sources) != 43 or len({item["roi_id"] for item in sources}) != 43
            or len(names) != 41 or manifest["full_known_eligible_extent"] != 43):
        raise ValueError("Manifest does not describe 43 distinct NASH-HCC cores")
    with formal_read_lease(S1, data_root=ROOT):
        panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
    if int(panel.keep_channel.sum()) != 32:
        raise ValueError("The current S1 selection differs from the 32-channel panel")
    with formal_read_lease(RAW, data_root=ROOT):
        for item in sources:
            image = RAW / item["name"]
            summary = RAW / item["summary"]["name"]
            if (item["component"] != "NASH_HCC_TMA" or item["panel_id"] != PANEL_ID
                    or image.stat().st_size != item["size"]
                    or summary.stat().st_size != item["summary"]["size"]):
                raise ValueError(f"Incomplete or changed source pair: {item['name']}")
            source_panel = pd.read_csv(summary, keep_default_na=False)
            with tifffile.TiffFile(image) as tif:
                pages = tif.pages
                if (len(pages) != 41 or any(page.dtype != np.dtype("float32") or
                        page.shape != pages[0].shape for page in pages)):
                    raise ValueError(f"Unexpected native TIFF layout: {image}")
                item["height_px"], item["width_px"] = pages[0].shape
            if (source_panel["name"].tolist() != names or
                    source_panel["channel"].tolist() !=
                    [row["channel"] for row in ordered[PANEL_ID]["channels"]]):
                raise ValueError(f"Per-core channel order differs: {summary}")
    return sources, panel


def field_mappings() -> pd.DataFrame:
    example = RAW / "NASH_HCC_TMA/IMC_summary_panel/NASH_HCC_TMA-2_001_summary.csv"
    rows = [
        mapping(MANIFEST, "roi_id", "image_metadata", "image_id",
                "use the component-qualified physical TMA core identity"),
        mapping(MANIFEST, "tma_core", "image_metadata", "core_id",
                "preserve the published core grid coordinate"),
        mapping(MANIFEST, "component", "image_metadata", "slide_id",
                "preserve the source TMA component"),
        mapping(example, "name", "panel_mapping", "source_name",
                "use the complete per-core CSV row order; verify all 43 summaries"),
        mapping(S1, "Original marker", "panel_mapping", "marker_id",
                "exact study-specific S1 mapping and selection"),
        mapping(MANIFEST, "tma_core", "image_metadata", "sample_id",
                "leave empty pending a core-to-biospecimen key",
                notes=("The source manifest and per-core summaries identify physical TMA positions, "
                       "not a biological specimen or patient. The public BioStudies record, "
                       f"paper ({PAPER}), its supplement ({relative(SUPPLEMENT)}), and "
                       f"Zenodo analysis scripts ({relative(PUBLIC_SCRIPTS)}) "
                       f"The clinical patient table ({relative(CLINICAL_SUPPLEMENT)}) was also checked; "
                       "it has no core or image key, so no verified core-to-specimen-to-patient "
                       "mapping was found. "
                       "The paper describes tumor and adjacent liver cores within this TMA, "
                       "so core position alone does not establish tissue type.")),
    ]
    clinical_join = "Table S1 No. row; project record key generated from this patient row"
    preparer_script = "scripts/data/preparation/gerber_biad2091_imc_mimic.py"
    rows.extend([
        mapping(preparer_script, "STUDY", "sample_metadata", "study",
                "copy the canonical dataset identifier assigned by this preparer",
                notes="Internal project identifier; not a clinical source value.",
                source_table="preparation script"),
        mapping(CLINICAL_SUPPLEMENT, "No.", "sample_metadata", "sample_id",
                "create patient_record_NN from the Table S1 row number; this is not a source sample ID",
                notes="Each Table S1 row reports one patient; the No. is a row number, not a published biospecimen identifier.",
                join=clinical_join, source_table=CLINICAL_TABLE),
        mapping(CLINICAL_SUPPLEMENT, "No.", "sample_metadata", "patient_key",
                f"namespace the generated patient-record key with {STUDY}",
                notes="One key per patient row; no source patient ID or link to a TMA core is published.",
                join=clinical_join, source_table=CLINICAL_TABLE),
        mapping(CLINICAL_SUPPLEMENT, "Sex", "sample_metadata", "sex_reported",
                "map the reported category to lowercase without inferring sex assigned at birth",
                join=clinical_join, source_table=CLINICAL_TABLE,
                value_mapping={"Male": "male", "Female": "female"}),
        mapping(CLINICAL_SUPPLEMENT, "Diagnosis", "sample_metadata", "cohort_id",
                "copy the reported diagnosis label for the local NASH-HCC subset",
                join=clinical_join, source_table=CLINICAL_TABLE),
        mapping(CLINICAL_SUPPLEMENT, "Diagnosis", "sample_metadata", "primary_diagnosis",
                "copy the reported diagnosis label without adding a code or timepoint",
                join=clinical_join, source_table=CLINICAL_TABLE),
        mapping(preparer_script, "clinical_samples provenance note", "sample_metadata",
                "metadata_notes",
                "record the row-level source, generated-key limits, missing image crosswalk, "
                "and unspecified time origins",
                notes="Generated annotation accompanying each Table S1-derived patient row.",
                join=clinical_join, source_table="clinical_samples()"),
    ])
    for source_field, target_field, definition, unit in CLINICAL_EXTENSION_FIELDS:
        notes = json.dumps({"data_type": "string", "definition": definition,
                            "unit": unit or None}, ensure_ascii=False)
        rows.append(mapping(CLINICAL_SUPPLEMENT, source_field, "sample_metadata", target_field,
                            "retain the source value as rendered in the official PDF; leave blank cells empty",
                            notes=notes, join=clinical_join, source_table=CLINICAL_TABLE))
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def header_matches(path: Path, panel: pd.DataFrame, side: int) -> bool:
    import tifffile

    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    with tifffile.TiffFile(path) as tif:
        names = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        return (len(tif.pages) == len(kept) and names == kept.marker_id.tolist()
                and all(page.shape == (side, side) and page.dtype == np.dtype("float32")
                        for page in tif.pages))


def preparer_review(completed: int, sample_rows: int, *, include_d01: bool = True) -> None:
    review = OUTPUT / "review_results.csv"
    rows = [
        dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample", record_id=STUDY,
             check_id="M01", result="unresolved",
             source_evidence=";".join((relative(MANIFEST), relative(PUBLIC_SCRIPTS),
                                       relative(SUPPLEMENT), relative(CLINICAL_SUPPLEMENT),
                                       PAPER, CLINICAL_PAPER)),
             output_evidence=relative(OUTPUT / "sample_metadata.csv"),
             details=(f"Preparer trace: {sample_rows} patient-level records transcribed from the "
                      f"official Table S1; {completed}/43 native images indexed. Table S1 No. "
                      "is a row number; generated project patient-record keys do not identify "
                      "biospecimens. No public image/core-to-patient key was found, so image links "
                      "remain unresolved. Independent review pending.")),
    ]
    if include_d01:
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
                         check_id="D01", result="unresolved", source_evidence=relative(MANIFEST),
                         output_evidence=relative(OUTPUT / "images.parquet"),
                         details=(f"Preparer export {completed}/43; biological sample and patient "
                                  "links remain unresolved. Independent review and acceptance binding pending.")))
    incoming = pd.DataFrame(rows, columns=columns("review_results"))
    with formal_read_lease(review, data_root=ROOT):
        prior = (pd.read_csv(review, keep_default_na=False) if review.is_file() else
                 pd.DataFrame(columns=incoming.columns))
    replace = (prior.preparer.eq(PREPARER) & prior.reviewer.eq("") &
               prior.check_id.isin({row["check_id"] for row in rows}))
    merged = pd.concat([prior.loc[~replace], incoming], ignore_index=True)
    with formal_publish_lease(review, directory=OUTPUT, data_root=ROOT):
        temporary = review.with_suffix(".csv.partial")
        merged.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, review)


def metadata_only() -> None:
    samples = clinical_samples()
    fields = field_mappings()
    index = OUTPUT / "images.parquet"
    with formal_read_lease(OUTPUT, data_root=ROOT):
        if not index.is_file():
            raise FileNotFoundError(index)
        completed = len(pd.read_parquet(index))
    for name, frame in (("sample_metadata.csv", samples), ("field_mapping.csv", fields)):
        path = OUTPUT / name
        with formal_publish_lease(path, directory=OUTPUT, data_root=ROOT):
            temporary = path.with_suffix(path.suffix + ".partial")
            frame.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, path)
    preparer_review(completed, len(samples), include_d01=False)
    print(f"published {len(samples)} patient records; image links remain unresolved", flush=True)


def main() -> None:
    from cellweft.data.io.readers import open_source
    from cellweft.data.preparation.study_input import export_input, save_input_tables

    sources, panel = source_inventory()
    fields = field_mappings()
    samples = clinical_samples()
    existing = {}
    index = OUTPUT / "images.parquet"
    verified_existing = set()
    with formal_read_lease(OUTPUT, data_root=ROOT):
        if index.is_file():
            existing = {row.image_id: row._asdict() for row in
                        pd.read_parquet(index).itertuples(index=False)}
        if set(existing) - {item["roi_id"] for item in sources}:
            raise ValueError("Indexed image is outside the current source manifest")
        for item in sources:
            image_id = item["roi_id"]
            destination = OUTPUT / "images" / image_id / "input.tiff"
            if image_id in existing and destination.is_file():
                if not header_matches(destination, panel,
                                      max(item["height_px"], item["width_px"])):
                    raise ValueError(f"Indexed input header changed: {image_id}")
                verified_existing.add(image_id)

    pending = 0
    for item in sources:
        image_id = item["roi_id"]
        source = RAW / item["name"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in verified_existing:
            continue
        side = max(item["height_px"], item["width_px"])
        predicted = side * side * int(panel.keep_channel.sum()) * 4
        if shutil.disk_usage(ROOT).free < 100 * 1024**3 + 2 * predicted:
            checkpoint("waiting_io", len(existing), image_id, sample_rows=len(samples),
                       reason="NAS 100 GiB reserve")
            preparer_review(len(existing), len(samples))
            return
        if shutil.disk_usage("/tmp").free < 64 * 1024**3 + 2 * predicted:
            checkpoint("waiting_io", len(existing), image_id, sample_rows=len(samples),
                       reason="local 64 GiB reserve")
            preparer_review(len(existing), len(samples))
            return
        spec = SourceSpec(image_id=image_id, panel_id=PANEL_ID, kind="tiff", path=source,
                          channels=tuple(range(41)), pixel_size_um=(1.0, 1.0),
                          source_revision=SOURCE_REVISION)
        with open_source(spec) as reader:
            result = export_input(reader, panel, destination, tile_size=256, data_root=ROOT)
        if not result["admitted"]:
            checkpoint("excluded_no_marker_signal", len(existing), image_id,
                       sample_rows=len(samples))
            continue
        with formal_read_lease(destination, data_root=ROOT):
            if not header_matches(destination, panel, side):
                raise ValueError(f"Published input header differs: {image_id}")
        row = dict.fromkeys(columns("image_metadata"))
        row.update(study=STUDY, image_id=image_id, sample_id=None, panel_id=PANEL_ID,
                   image_kind="tma_core", source_image_id=Path(item["name"]).stem,
                   source_files=[relative(source), relative(RAW / item["summary"]["name"])],
                   source_format="tiff", source_axes="CYX", source_dtype="float32",
                   source_revision=SOURCE_REVISION, slide_id=item["component"],
                   core_id=item["tma_core"], roi_id=image_id,
                   region_annotation=item["tma_core"], source_x0_px=0, source_y0_px=0,
                   source_x1_px=item["width_px"], source_y1_px=item["height_px"],
                   pixel_size_x_um=1.0,
                   pixel_size_y_um=1.0, input_image_path=relative(destination),
                   intensity_description="Native float32 IMC TIFF intensity, no normalization",
                   split_method="already_independent", split_evidence=relative(MANIFEST),
                   metadata_notes="Physical TMA core only; specimen, patient and tissue subtype unconfirmed.",
                   **{key: result[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px",
                                                   "pad_bottom_px", "height_px", "width_px",
                                                   "source_channel_count", "channel_count")})
        existing[image_id] = row
        pending += 1
        checkpoint("published_unindexed", len(existing) - pending, image_id,
                   sample_rows=len(samples),
                   next_step="reopen this published TIFF and merge its index on resume")
        if pending == 4:
            save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields,
                              data_root=ROOT)
            pending = 0
            following = next((source_item["roi_id"] for source_item in sources
                              if source_item["roi_id"] not in existing), None)
            checkpoint("batch_saved", len(existing), following, sample_rows=len(samples))
            print(f"indexed {len(existing)}/43 through {image_id}", flush=True)

    if existing:
        save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields,
                          data_root=ROOT)
    preparer_review(len(existing), len(samples))
    checkpoint("prepared", len(existing), None, sample_rows=len(samples),
               next_step="independent S01/I01-I05/P01-P03/M01-M02/D01 review")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true",
                        help="publish patient metadata without changing image artifacts")
    args = parser.parse_args()
    metadata_only() if args.metadata_only else main()
