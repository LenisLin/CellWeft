"""Deliver confirmed native Yeo mIF TMA cores as independent study inputs."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Yeo2026_mIF_DLBCL"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
MANIFEST = META / "tma_source_manifest.json"
PROOFS = META / "tma_native_proofs"
ORDER = META / "tma_ordered_markers.json"
S11 = NAS / "raw/public" / STUDY / "metadata/cd-25-0775_supp_table_11_suppst11.xlsx"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0652_prepare_0")))
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
PANEL_ID = "zenodo18379156_mif_27"
PREPARER = f"{STUDY}:{JOB.name}"
GAP = ("Supplementary Table S11 provides patient-associated TMA core records grouped by source "
       "tumor block (TB#). Exact TMA/Seq-to-core joins link 20 of 47 released images to 18 "
       "biological samples; DFCI and remaining core identities have no verified crosswalk and "
       "remain unlinked.")


def relative(path: Path) -> str:
    return metadata_source_relative(path, NAS).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def publish_local(source: Path, destination: Path) -> None:
    with formal_publish_lease(destination, directory=destination.parent):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)


def progress(stage: str, images: pd.DataFrame, scope: list[dict],
             samples: pd.DataFrame | None = None, **details: object) -> None:
    finished = set(images.image_id) if len(images) else set()
    sample_rows = 0 if samples is None else len(samples)
    patient_linked_samples = (0 if samples is None else
                              int(samples.patient_key.notna().sum()))
    sample_linked_images = (int(images.sample_id.notna().sum()) if len(images) else 0)
    state = dict(study=STUDY, phase="prepare", scientific_repair_round=1,
                 stage=stage, expected_images=len(scope), exported_images=len(images),
                 sample_rows=sample_rows, sample_linked_images=sample_linked_images,
                 patient_linked_samples=patient_linked_samples,
                 next_image=next((item["image_id"] for item in scope
                                  if item["image_id"] not in finished), None),
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    STAGING.mkdir(parents=True, exist_ok=True)
    local = STAGING / "progress.json"
    local.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    publish_local(local, JOB / "progress.json")
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> tuple[list[dict], pd.DataFrame]:
    scope = []
    with formal_read_lease(MANIFEST):
        manifest = json.loads(MANIFEST.read_text())
        official = json.loads(ORDER.read_text())["ordered_channels"]
        if manifest["ordered_channels"] != official or len(official) != 27:
            raise ValueError("Manifest and official marker order disagree")
        for core in manifest["cores"]:
            path = NAS / core["image_rel"]
            proof_path = PROOFS / f'{core["proof_id"]}.json'
            proof = json.loads(proof_path.read_text())
            if not (proof["complete"] and proof["image_rel"] == core["image_rel"]
                    and proof["identity_confirmed"] == core["identity_confirmed"]):
                raise ValueError(f"Native proof disagrees: {core['proof_id']}")
            if not core["identity_confirmed"]:
                continue
            stat = path.stat()
            if (stat.st_size != proof["file_stat"]["size"] or
                    stat.st_mtime_ns != proof["file_stat"]["mtime_ns"] or
                    stat.st_size != core["uncompressed_bytes"]):
                raise ValueError(f"Qualified source changed: {core['image_rel']}")
            if (proof["native_axes"] != "CYX" or proof["shape"][0] != 27 or
                    proof["native_channel_labels"] != [
                        "Myc-Cy5-200" if name == "Myc" else name for name in official]):
                raise ValueError(f"Native panel or axes changed: {core['proof_id']}")
            scope.append(dict(image_id=f'{core["parent_tma"]}_{core["core_id"]}',
                              core=core, proof=proof, path=path,
                              shape=tuple(proof["shape"]),
                              source_revision=f'zenodo18379156-v1:{stat.st_size}:{stat.st_mtime_ns}'))
    if len(scope) != 47 or len({item["image_id"] for item in scope}) != 47:
        raise ValueError("Expected 47 distinct identity-confirmed cores")
    panel = panel_from_s1(STUDY, PANEL_ID, scope[0]["proof"]["native_channel_labels"],
                          S1, aliases={"Myc-Cy5-200": "Myc"})
    if (len(panel) != 27 or int(panel.keep_channel.sum()) != 26 or
            panel.loc[~panel.keep_channel, "source_name"].tolist() != ["LMP1"] or
            panel.loc[panel.nuclear, "source_name"].tolist() != ["DAPI"]):
        raise ValueError("Current S1 selection differs from confirmed Yeo panel")
    return scope, panel


def field_rows() -> pd.DataFrame:
    rows = []

    def add(source: Path | str, source_field: str, table: str, target: str,
            rule: str, *, join: str = "", notes: str = "") -> None:
        rows.append(dict(study=STUDY, source_file=relative(source) if isinstance(source, Path) else source,
                         source_sheet_or_table=("manifest cores / native OME" if source == MANIFEST else
                                                "DLBCL_TMA" if source == S11 else "published source"),
                         source_field=source_field, target_table=table, target_field=target,
                         join_keys=join, value_mapping="", transformation_rule=rule,
                         time_origin_rule="", conflict_rule="No inferred sample or patient identity",
                         notes=notes))

    add(MANIFEST, "member;image_rel;proof_id;identity_confirmed", "image_metadata",
        "image_id;source_image_id;source_files;core_id;slide_id",
        "Use identity-confirmed source TIFF only; image_id combines TMA and exact core token.",
        join="proof_id -> tma_native_proofs/<proof_id>.json",
        notes="47 confirmed of 49 members; Rochester/C04_1 and Tub972/None_1 remain excluded pending identity.")
    add(MANIFEST, "parent_tma;core_id", "image_metadata", "sample_id",
        "Link only when the source TMA and numeric core token uniquely equal the Table S11 TMA and Seq values.",
        join="image_metadata.slide_id;core_id = sample_metadata.source_tma;source_seq",
        notes=(GAP + " TMA aliases follow the public archive names: 97.1=Tub971 and 97.2=Tub972; "
               "core token cNN is normalized to numeric Seq NN."))
    add("https://pmc.ncbi.nlm.nih.gov/articles/PMC13091187/", "Methods: DLBCL TMA cohorts; Supplementary Table S11",
        "sample_metadata", "cohort_id;source_subject_id",
        "Paper identifies the DFCI, Rochester, and Tübingen TMA cohorts and describes Supplementary Table S11 as deidentified patient metadata.")
    add(S11, "TB#", "sample_metadata", "sample_id;source_sample_id",
        "Group retained TMA/Seq core records by source TB#; each TB# maps to one PT# and defines one biological tissue sample.",
        join="TB# -> PT#; exact TMA + normalized Seq joins image core to its TB# sample where unique",
        notes="The worksheet has 141 retained patient-associated TMA/Seq records but 73 unique TB# and PT# values. Repeated cores from one TB# are represented by one sample row.")
    add(S11, "study context", "sample_metadata", "study",
        "Assign the registered CellWeft study identifier to each retained S11 patient record.")
    add(S11, "PT#", "sample_metadata", "patient_key;source_subject_id",
        "Namespace the source patient identifier by study; repeated PT# values share a patient_key. Control rows are excluded.",
        notes="Patient identity follows the source PT# column; no cross-institution identity inference is performed.")
    add(S11, "TMA;seq", "sample_metadata", "cohort_id;source_tma;source_seq",
        "Set cohort_id only when a TB# has one TMA label. Preserve distinct TMA and Seq values in source extensions; multiple values are encoded as JSON arrays.",
        join="TB# group",
        notes="TMA/Seq rows are retained as provenance for the biological sample; they do not create separate sample rows.")
    add(S11, "Dx, Prime;Sub, Prime;Sex", "sample_metadata",
        "primary_diagnosis;histology;sex_reported",
        "Copy a standardized value only when all non-placeholder values in the TB# group agree; otherwise leave it blank. Map consistent source M/F to male/female.")
    for source_field, target_field, definition in [
        ("TMA", "source_tma", "Raw TMA label in Supplementary Table S11."),
        ("seq", "source_seq", "Raw Seq value in Supplementary Table S11, normalized from Excel numeric representation."),
        ("TB#", "source_tb_id", "Raw TB# value in Supplementary Table S11."),
        ("PT#", "source_pt_id", "Raw PT# value in Supplementary Table S11."),
        ("Dx, Prime", "source_diagnosis_prime", "Raw Dx, Prime value in Supplementary Table S11."),
        ("Sub, Prime", "source_subtype_prime", "Raw Sub, Prime value in Supplementary Table S11."),
        ("Sex", "source_sex", "Raw Sex value in Supplementary Table S11."),
        ("Age", "source_age", "Raw Age value in Supplementary Table S11; no age time origin is assigned."),
        ("Source", "source_source", "Raw Source value in Supplementary Table S11."),
        ("Source2", "source_source2", "Raw Source2 value in Supplementary Table S11."),
        ("EBV", "source_ebv", "Raw EBV code in Supplementary Table S11; source coding is not reinterpreted."),
        ("Block", "source_block", "Raw Block value in Supplementary Table S11."),
        ("GCB/non-GCB", "source_gcb_non_gcb", "Raw GCB/non-GCB value in Supplementary Table S11."),
    ]:
        add(S11, source_field, "sample_metadata", target_field,
            "Within each TB# group, collapse identical source values; store one value as a scalar and multiple distinct values as a JSON array.",
            notes=json.dumps(dict(data_type="string", definition=definition), ensure_ascii=False))
    add(S11, "Age", "sample_metadata", "age_at_collection_years;age_at_diagnosis_years",
        "Do not assign Age to a standardized age field because the worksheet does not define its time origin.",
        notes="Raw Age is retained in source_age.")
    add(S11, "worksheet row numbers; TB#; grouped source fields", "sample_metadata", "metadata_notes",
        "Within each TB# group, record retained worksheet rows, source-field conflicts, and the unspecified Age time origin; do not infer identity.",
        join="TB# -> sample_id; retained S11 rows grouped by TB#",
        notes="Source values remain in the mapped source_* extension fields; notes summarize row membership and conflicts.")
    add(ORDER, "ordered_channels", "panel_mapping", "source_name;source_channel_index",
        "Use native OME channel order; exact native Myc-Cy5-200 at official Myc position 1.",
        notes="Native aliases verified in all 47 local proof records.")
    add(S1, "Study;Original marker", "panel_mapping", "marker_id;role;keep_channel",
        "Apply current study S1 to all 27 native source rows; exclude LMP1.")
    add(MANIFEST, "source full frame;native proof shape", "image_metadata",
        "source_x0_px;source_y0_px;source_x1_px;source_y1_px;split_method",
        "Each source TIFF is one complete dearrayed core; retain its full native frame and center-pad only.")
    return pd.DataFrame(rows, columns=columns("field_mapping"))


_OOXML_MAIN = "http://purl.oclc.org/ooxml/spreadsheetml/main"
_OOXML_REL = "http://purl.oclc.org/ooxml/officeDocument/relationships"
_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_SAMPLE_EXTENSIONS = {
    "TMA": "source_tma", "seq": "source_seq", "TB#": "source_tb_id",
    "PT#": "source_pt_id", "Dx, Prime": "source_diagnosis_prime",
    "Sub, Prime": "source_subtype_prime", "Sex": "source_sex",
    "Age": "source_age", "Source": "source_source", "Source2": "source_source2",
    "EBV": "source_ebv", "Block": "source_block",
    "GCB/non-GCB": "source_gcb_non_gcb",
}


def _column_name(reference: str) -> str:
    match = re.fullmatch(r"([A-Z]+)\d+", reference)
    if match is None:
        raise ValueError(f"Invalid OOXML cell reference: {reference}")
    return match.group(1)


def _excel_scalar(value: str, cell_type: str, shared_strings: list[str]) -> str:
    if cell_type == "s":
        return shared_strings[int(value)] if value else ""
    if cell_type == "inlineStr":
        return value
    if not value:
        return ""
    try:
        numeric = float(value)
    except ValueError:
        return value
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite value in Supplementary Table S11: {value}")
    return format(numeric, ".15g")


def supplementary_table_rows() -> list[tuple[int, dict[str, str]]]:
    """Read the author's strict-OOXML S11 workbook without rewriting its bytes."""
    with zipfile.ZipFile(S11) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in rels.findall(f"{{{_PACKAGE_REL}}}Relationship")
        }
        sheet = workbook.find(f"{{{_OOXML_MAIN}}}sheets/{{{_OOXML_MAIN}}}sheet")
        if sheet is None or sheet.attrib.get("name") != "DLBCL_TMA":
            raise ValueError("Supplementary Table S11 is missing the DLBCL_TMA worksheet")
        sheet_target = relationship_targets[sheet.attrib[f"{{{_OOXML_REL}}}id"]]
        worksheet_name = str(Path("xl") / sheet_target)
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.iter(f"{{{_OOXML_MAIN}}}t"))
            for item in strings_root.findall(f"{{{_OOXML_MAIN}}}si")
        ]
        root = ET.fromstring(archive.read(worksheet_name))
        xml_rows = root.findall(f".//{{{_OOXML_MAIN}}}sheetData/{{{_OOXML_MAIN}}}row")
        if not xml_rows:
            raise ValueError("Supplementary Table S11 worksheet is empty")
        parsed = []
        for row in xml_rows:
            values = {}
            for cell in row.findall(f"{{{_OOXML_MAIN}}}c"):
                scalar = cell.find(f"{{{_OOXML_MAIN}}}v")
                if scalar is None:
                    inline = cell.find(f"{{{_OOXML_MAIN}}}is")
                    value = ("" if inline is None else "".join(
                        node.text or "" for node in inline.iter(f"{{{_OOXML_MAIN}}}t")))
                else:
                    value = _excel_scalar(scalar.text or "", cell.attrib.get("t", ""), shared_strings)
                values[_column_name(cell.attrib["r"])] = value
            parsed.append((int(row.attrib["r"]), values))
    header_row, raw_headers = parsed[0]
    if header_row != 1 or len(set(raw_headers.values())) != len(raw_headers):
        raise ValueError("Supplementary Table S11 has an invalid header row")
    required = set(_SAMPLE_EXTENSIONS)
    if set(raw_headers.values()) != required:
        raise ValueError(f"Unexpected Supplementary Table S11 columns: {set(raw_headers.values())}")
    headers = raw_headers
    return [(row_number, {headers[column]: value for column, value in values.items()})
            for row_number, values in parsed[1:]]


def sample_metadata() -> pd.DataFrame:
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for source_row, values in supplementary_table_rows():
        patient_id = values.get("PT#", "").strip()
        if patient_id in {"", "/", "TONSIL"}:
            continue
        tumor_block = values.get("TB#", "").strip()
        if not tumor_block or tumor_block == "/":
            raise ValueError(f"Retained S11 record lacks TB#: row {source_row}")
        grouped.setdefault(tumor_block, []).append((source_row, values))

    def source_values(records: list[tuple[int, dict[str, str]]], field: str) -> list[str]:
        return list(dict.fromkeys(values.get(field, "").strip()
                                  for _, values in records if values.get(field, "").strip()))

    def consensus(records: list[tuple[int, dict[str, str]]], field: str) -> str | None:
        values = [value for value in source_values(records, field) if value != "/"]
        return values[0] if len(values) == 1 else None

    def encoded_source_values(records: list[tuple[int, dict[str, str]]], field: str) -> str | None:
        values = source_values(records, field)
        if not values:
            return None
        return values[0] if len(values) == 1 else json.dumps(values, ensure_ascii=False)

    rows = []
    for tumor_block, records in grouped.items():
        patient_ids = source_values(records, "PT#")
        if len(patient_ids) != 1:
            raise ValueError(f"One S11 TB# maps to multiple PT# values: {tumor_block}")
        patient_id = patient_ids[0]
        source_tmas = [value for value in source_values(records, "TMA") if value != "/"]
        cohort_id = source_tmas[0] if len(source_tmas) == 1 else pd.NA
        diagnosis = consensus(records, "Dx, Prime")
        histology = consensus(records, "Sub, Prime")
        sex = consensus(records, "Sex")
        source_sex = {"M": "male", "F": "female"}.get(str(sex), pd.NA)
        conflict_fields = [field for field in _SAMPLE_EXTENSIONS
                           if len(source_values(records, field)) > 1]
        source_rows = ",".join(str(number) for number, _ in records)
        notes = (f"Table S11 TB# group; source rows {source_rows}; "
                 f"conflicting source fields: {', '.join(conflict_fields) if conflict_fields else 'none'}; "
                 "Age time origin is unspecified.")
        row = dict.fromkeys(columns("sample_metadata"), pd.NA)
        row.update(study=STUDY, sample_id=f"S11:TB#:{tumor_block}",
                   patient_key=f"{STUDY}:PT#:{patient_id}",
                   source_subject_id=patient_id, cohort_id=cohort_id,
                   source_sample_id=tumor_block,
                   primary_diagnosis=diagnosis, histology=histology,
                   sex_reported=source_sex, metadata_notes=notes)
        for source_field, target_field in _SAMPLE_EXTENSIONS.items():
            row[target_field] = encoded_source_values(records, source_field)
        rows.append(row)
    samples = pd.DataFrame(rows)
    expected = 73
    if len(samples) != expected or samples.sample_id.duplicated().any():
        raise ValueError(f"Expected {expected} unique S11 TB# samples, got {len(samples)}")
    if samples.patient_key.isna().any() or samples.patient_key.nunique() != expected:
        raise ValueError("S11 patient IDs do not produce the expected 73 unique patient keys")
    return samples


def link_sample_metadata(images: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    slide_to_tma = {"DFCI": "DFCI", "Rochester": "Rochester",
                    "Tub971": "97.1", "Tub972": "97.2"}
    sample_id_by_block = {str(row.source_sample_id): row.sample_id
                          for row in samples.itertuples(index=False)}
    sample_ids_by_tma_sequence: dict[tuple[str, str], set[str]] = {}
    for source_row, values in supplementary_table_rows():
        if values.get("PT#", "").strip() in {"", "/", "TONSIL"}:
            continue
        sequence = values.get("seq", "").strip()
        if sequence in {"", "/"}:
            continue
        try:
            sequence = format(float(sequence), ".15g")
        except ValueError:
            sequence = sequence.strip()
        block = values.get("TB#", "").strip()
        sample_id = sample_id_by_block.get(block)
        if sample_id is None:
            raise ValueError(f"S11 row {source_row} has no grouped TB# sample: {block}")
        sample_ids_by_tma_sequence.setdefault((values.get("TMA", "").strip(), sequence), set()).add(sample_id)
    sample_by_tma_sequence = {
        key: next(iter(sample_ids)) if len(sample_ids) == 1 else None
        for key, sample_ids in sample_ids_by_tma_sequence.items()
    }

    result = images.copy()
    result["sample_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["metadata_notes"] = GAP
    for index, row in result.iterrows():
        tma = slide_to_tma.get(str(row.slide_id))
        core = re.fullmatch(r"[cC](\d+)", str(row.core_id))
        if tma is None or core is None:
            continue
        sequence = str(int(core.group(1)))
        sample_id = sample_by_tma_sequence.get((tma, sequence))
        if sample_id:
            result.at[index, "sample_id"] = sample_id
            result.at[index, "metadata_notes"] = (
                "Linked to Supplementary Table S11 by exact TMA and numeric Seq/core_id match, then grouped by TB#.")
    return result


class FastTiffWriter(tifffile.TiffWriter):
    def write(self, *args: object, **kwargs: object) -> object:
        if kwargs.get("compression") == "deflate":
            kwargs["compression"] = "zstd"
            kwargs["compressionargs"] = {"level": 1}
        return super().write(*args, **kwargs)


def image_row(item: dict, result: dict, destination: Path) -> dict:
    core = item["core"]
    _, height, width = item["shape"]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"], panel_id=PANEL_ID,
               image_kind="tma_core", source_image_id=core["member"],
               source_files=[core["image_rel"]], source_format="ome_tiff", source_axes="CYX",
               source_dtype=result["source_dtype"], source_revision=item["source_revision"],
               slide_id=core["parent_tma"], core_id=core["core_id"],
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               pad_left_px=result["pad_left_px"], pad_right_px=result["pad_right_px"],
               pad_top_px=result["pad_top_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=result["height_px"], width_px=result["width_px"],
               source_channel_count=27, channel_count=result["channel_count"],
               input_image_path=relative(destination),
               intensity_description="Source-decoded native uint16 mIF intensities; no CellWeft scaling",
               split_method="already_independent",
               split_evidence=f'{relative(MANIFEST)}: {core["member"]}; {relative(PROOFS / (core["proof_id"] + ".json"))}',
               metadata_notes=GAP)
    return row


def verify_header(destination: Path, item: dict, panel: pd.DataFrame) -> None:
    side = max(item["shape"][1:])
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
        series = tif.series[0]
        names = [channel.attrib["Name"] for channel in
                 ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
        if (series.shape != (len(kept), side, side) or series.axes != "CYX" or
                series.dtype != np.dtype("uint16") or names != kept.marker_id.tolist()):
            raise ValueError(f"Published header differs from source and panel: {destination}")


def review_rows(images: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    sources = [relative(MANIFEST), relative(ORDER), relative(PROOFS), relative(S11),
               "https://zenodo.org/records/18379156",
               "https://pmc.ncbi.nlm.nih.gov/articles/PMC13091187/"]
    rows = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples():
        detail = dict(preparer_self_check=True, independent_review_pending=True,
                      expected_images=47, exported_images=len(images), sample_rows=len(samples),
                      sample_linked_images=int(images.sample_id.notna().sum()),
                      patient_linked_samples=int(samples.patient_key.notna().sum()),
                      unique_patient_keys=int(samples.patient_key.nunique()),
                      identity_unresolved=["Rochester/C04_1", "Tub972/None_1"])
        if check.check_id in {"M01", "M02", "D01"}:
            detail["metadata_gap"] = GAP
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="",
                         scope=check.scope, record_id=PANEL_ID if check.scope == "panel" else STUDY,
                         check_id=check.check_id, result="unresolved",
                         source_evidence=json.dumps(sources), output_evidence=relative(OUTPUT),
                         details=json.dumps(detail, ensure_ascii=False)))
    return pd.DataFrame(rows, columns=columns("review_results"))


def publish_reviews(images: pd.DataFrame, samples: pd.DataFrame) -> None:
    path = OUTPUT / "review_results.csv"
    with formal_read_lease(path):
        prior = pd.read_csv(path, keep_default_na=False) if path.is_file() else pd.DataFrame(columns=columns("review_results"))
    incoming = review_rows(images, samples)
    retained = prior.loc[~(prior.preparer.eq(PREPARER) & prior.reviewer.eq(""))]
    STAGING.mkdir(parents=True, exist_ok=True)
    staged = STAGING / "review_results.csv"
    pd.concat([retained, incoming], ignore_index=True).to_csv(staged, index=False, encoding="utf-8-sig")
    publish_local(staged, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    STAGING.mkdir(parents=True, exist_ok=True)
    scope, panel = source_scope()
    samples = sample_metadata()
    fields = field_rows()
    with formal_read_lease(OUTPUT / "images.parquet"):
        images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file()
                  else pd.DataFrame(columns=columns("image_metadata")))
    known = {item["image_id"]: item for item in scope}
    for row in images.itertuples():
        if row.image_id not in known or row.source_revision != known[row.image_id]["source_revision"]:
            raise ValueError(f"Previously indexed source changed: {row.image_id}")
        if not (NAS / row.input_image_path).is_file():
            raise ValueError(f"Previously indexed TIFF missing: {row.image_id}")
    images = link_sample_metadata(images, samples)
    if args.metadata_only:
        if len(samples) != 73 or int(images.sample_id.notna().sum()) != 20:
            raise ValueError("Metadata-only update expected 47 indexed images and 20 exact sample links")
        save_input_tables(OUTPUT, images, samples, panel, fields,
                          replace_metadata_studies={STUDY})
        publish_reviews(images, samples)
        progress("metadata_prepared", images, scope, samples=samples, excluded_unresolved=2)
        print(json.dumps(dict(sample_rows=len(samples), unique_patients=int(samples.patient_key.nunique()),
                              image_rows=len(images), image_sample_links=int(images.sample_id.notna().sum()),
                              source_sample_ids=int(samples.source_sample_id.notna().sum())),
                         ensure_ascii=False))
        return
    save_input_tables(OUTPUT, images, samples, panel, fields,
                      replace_metadata_studies={STUDY})
    publish_reviews(images, samples)
    progress("metadata_prepared", images, scope, samples=samples, excluded_unresolved=2)
    todo = [item for item in scope if item["image_id"] not in set(images.image_id)]
    if args.limit is not None:
        todo = todo[:args.limit]
    for item in todo:
        identity = item["image_id"]
        destination = OUTPUT / "images" / identity / "input.tiff"
        side = max(item["shape"][1:])
        native_bytes = int(np.prod(item["shape"])) * 2
        output_bytes = 26 * side * side * 2
        if shutil.disk_usage(NAS).free < 100 * 1024**3 + output_bytes:
            progress("waiting_io", images, scope, samples=samples,
                     current_image=identity, reason="NAS 100 GiB reserve")
            return
        if shutil.disk_usage(STAGING).free < 64 * 1024**3 + native_bytes + output_bytes:
            progress("waiting_io", images, scope, samples=samples,
                     current_image=identity, reason="Local 64 GiB reserve")
            return
        specification = SourceSpec(identity, PANEL_ID, "tiff", item["path"], tuple(range(27)))
        with open_source(specification) as reader:
            if reader.shape != item["shape"] or reader.dtype != np.dtype("uint16"):
                raise ValueError(f"Native shape/dtype differs from proof: {identity}")
            with patch("cellweft.data.preparation.study_input.tifffile.TiffWriter", FastTiffWriter):
                result = export_input(reader, panel, destination)
        if not result["admitted"]:
            progress("source_excluded", images, scope, samples=samples, current_image=identity,
                     reason="All admitted downstream marker pixels are zero")
            continue
        verify_header(destination, item, panel)
        images = pd.concat([images, pd.DataFrame([image_row(item, result, destination)])], ignore_index=True)
        images = link_sample_metadata(images, samples)
        save_input_tables(OUTPUT, images, samples, panel, fields)
        progress("image_published", images, scope, samples=samples,
                 last_completed_image=identity)
    publish_reviews(images, samples)
    progress("prepared_awaiting_independent_review" if len(images) == len(scope) else "partial_prepared",
             images, scope, samples=samples, metadata_gap=GAP)


if __name__ == "__main__":
    main()
