"""Prepare native Lyman human PanIN IMC acquisitions from Zenodo 8336719.

Inputs: 13 verified MCD files, acquisition verification, S1 and published panel.
Outputs: native-intensity CYX TIFFs, image/panel/sample/field tables, preparer
review records and a resumable per-acquisition checkpoint. Table1 supplies
four patient links; one specimen and ROI pathology classes remain unresolved.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile
from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Lyman2025_IMC_PanIN"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_8336719"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0234_prepare_0"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "lyman_zenodo8336719_native_49"
PREPARER = "Lyman2025_IMC_PanIN_0234_prepare_0"
LOCAL = Path("/tmp/study_input_batch") / STUDY


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def checkpoint(stage: str, images: pd.DataFrame, decisions: dict, **extra: object) -> None:
    save_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(acquisitions),
        completed_images=len(images), reviewed_acquisitions=len(decisions),
        excluded_acquisitions=sum(not item["admitted"] for item in decisions.values()),
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        next_acquisition=next((a["native_acquisition_id"] for a in acquisitions
                              if decisions.get(a["native_acquisition_id"], {}).get("pixel_size_um") != 1.0), None),
        updated_at=datetime.now(timezone.utc).isoformat(), **extra))


def field(source: Path, source_field: str, target_table: str, target_field: str,
          join_keys: str, rule: str, notes: str) -> dict:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table="native/public release",
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join_keys, value_mapping="", transformation_rule=rule,
                time_origin_rule="No collection time inferred", conflict_rule="Keep unresolved without direct source key",
                notes=notes)


class RepeatedOriginReader:
    """Read the one MCD ROI whose unpositioned records all claim (0, 0).

    Only unique, non-origin coordinates are spatially recoverable. The origin
    and unmeasured positions remain invalid; their native values are not
    assigned to invented locations. Other acquisitions use the shared reader.
    """

    def __init__(self, source: SourceSpec, acquisition: dict):
        self.source = source
        self.shape = (49, acquisition["height"], acquisition["width"])
        self.dtype = np.dtype("float32")
        metadata = acquisition["native_metadata"]
        start, end = int(metadata["DataStartOffset"]), int(metadata["DataEndOffset"])
        records = (end - start) // ((49 + 3) * 4)
        assert records == self.shape[1] * self.shape[2]
        self.raw = np.memmap(source.path, dtype="float32", mode="r", offset=start,
                             shape=(records, 52))
        width, height = self.shape[2], self.shape[1]
        coords = self.raw[:, :2]
        assert np.isfinite(coords).all() and np.equal(coords, np.floor(coords)).all()
        x, y = coords[:, 0].astype(np.int64), coords[:, 1].astype(np.int64)
        assert ((0 <= x) & (x < width) & (0 <= y) & (y < height)).all()
        key = y * width + x
        repeated = key == 0
        assert int(repeated.sum()) == 72725
        assert len(np.unique(key[~repeated])) == int((~repeated).sum())
        self.record_index = np.full((height, width), -1, dtype=np.int64)
        self.record_index[y[~repeated], x[~repeated]] = np.flatnonzero(~repeated)
        save_json(CACHE / "duplicate_coordinate_review.json", dict(
            native_acquisition_id=acquisition["native_acquisition_id"],
            total_records=records, repeated_origin_records=int(repeated.sum()),
            unique_spatial_records=int((~repeated).sum()),
            unmapped_nonzero_values=int(np.count_nonzero(self.raw[repeated, 3:])),
            valid_pixels=int((self.record_index >= 0).sum()),
            rule="Only unique non-origin coordinates are valid; no value from the ambiguous origin records is reassigned."))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        index = self.record_index[y0:y1, x0:x1]
        valid = index >= 0
        values = np.zeros((len(channels), y1-y0, x1-x0), dtype=self.dtype)
        for output, channel in enumerate(channels):
            values[output, valid] = self.raw[index[valid], self.source.channels[channel] + 3]
        return values

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.record_index[y0:y1, x0:x1] >= 0

    def close(self) -> None:
        self.raw._mmap.close()


@contextmanager
def acquisition_reader(source: SourceSpec, acquisition: dict):
    if acquisition["native_acquisition_id"] == "12.mcd#acquisition_id=1":
        reader = RepeatedOriginReader(source, acquisition)
        try:
            yield reader
        finally:
            reader.close()
    else:
        with open_source(source) as reader:
            yield reader


def clinical_links(supplement: Path, author_script: Path, codes: set[str]) -> tuple[dict[str, str], str]:
    """Connect native slide accessions to unique Table1 patient labels and the author diagnosis note."""
    import openpyxl

    with formal_read_lease(supplement):
        workbook = openpyxl.load_workbook(supplement, read_only=True, data_only=True)
        rows = iter(workbook["Table1"].values)
        assert next(rows)[:2] == ("Patient", "LCM sample ID")
        matched = {code: set() for code in codes}
        for patient, lcm_id, *_ in rows:
            for code in codes:
                if re.search(r"(?<!\d)" + re.escape(code) + r"(?!\d)", str(lcm_id)):
                    matched[code].add(str(patient))
        workbook.close()
    assert all(len(patients) <= 1 for patients in matched.values())
    assert len([patients for patients in matched.values() if patients]) == 4
    patient_keys = {code: "lyman_patient_" + next(iter(patients))
                    for code, patients in matched.items() if patients}

    with formal_read_lease(author_script):
        lines = author_script.read_text().splitlines()
    note = next(i for i, line in enumerate(lines) if "this patient has cholangio" in line)
    cholangio_codes = {match for line in lines[note + 1:note + 5]
                       for match in re.findall(r"_(\d{5})_", line)}
    assert len(cholangio_codes) == 1 and cholangio_codes <= codes
    return patient_keys, cholangio_codes.pop()


def repair_clinical_metadata() -> None:
    """Repair only the independently reported identity and diagnosis discrepancies."""
    supplement = META / "source_metadata/jci.insight.191595.sdt1-10.xlsx"
    author_script = META / "source_metadata/IMCpipeline_humanpanin_withfunctions_TLSsample_analysis.R"
    with formal_read_lease(OUTPUT):
        with resource_lock(OUTPUT, shared=True):
            images = pd.read_parquet(OUTPUT / "images.parquet")
            samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
            fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str).fillna("")
            panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    assert len(images) == 75 and len(samples) == 5
    assert set(images.sample_id) == set(samples.sample_id)
    patient_keys, cholangio_code = clinical_links(supplement, author_script,
                                                   set(samples.source_sample_id))
    save_json(REPAIR_JOB / "progress.json", dict(study=STUDY, phase="repair",
              scientific_repair_round=1, stage="sources_checked", expected_images=75,
              exported_images=75, sample_rows=5, sample_linked_images=75,
              patient_linked_samples=0, next_step="publish clinical metadata",
              updated_at=datetime.now(timezone.utc).isoformat()))

    for index, sample in samples.iterrows():
        code = sample.source_sample_id
        samples.at[index, "patient_key"] = patient_keys.get(code, pd.NA)
        if code == cholangio_code:
            samples.at[index, "primary_diagnosis"] = "distal cholangiocarcinoma"
            samples.at[index, "metadata_notes"] = (
                "Native slide accession matches the author analysis exclusion for cholangiocarcinoma; "
                "the article identifies one distal cholangiocarcinoma case. ROI pathology class remains unassigned.")
        elif code in patient_keys:
            samples.at[index, "metadata_notes"] = (
                "Native slide accession uniquely matches a Table1 LCM sample ID and its patient label; "
                "the exact tissue block and ROI pathology class remain unassigned.")
        else:
            samples.at[index, "metadata_notes"] = (
                "Native slide accession groups the specimen; Table1 and the author analysis script do not "
                "provide a direct patient match. ROI pathology class remains unassigned.")

    patient_field = fields.target_table.eq("sample_metadata") & fields.target_field.eq("patient_key;sample_id")
    fields.loc[patient_field, "transformation_rule"] = (
        "Match five-digit native slide accession to Table1 LCM sample ID only when its patient label is unique")
    fields.loc[patient_field, "notes"] = (
        "Five specimen groups and 75 image-to-sample links; four unique Table1 patient matches "
        "cover 60 images, with one group and 15 images unresolved. Exact tissue block and ROI class remain unknown.")
    fields.loc[patient_field, "conflict_rule"] = "Leave unmatched patient keys empty"
    author_field = fields.source_file.eq(relative(author_script))
    author_field &= fields.target_field.eq("sample_id;patient_key")
    fields.loc[author_field, "transformation_rule"] = (
        "Use author labels to corroborate specimen accessions; do not infer the missing patient bridge")
    fields.loc[author_field, "notes"] = (
        "Author labels corroborate specimen accessions; the external metadata workbook and "
        "FCS-to-MCD ROI bridge remain unavailable.")
    fields = pd.concat([fields, pd.DataFrame([field(
        author_script, "cholangio exclusion comment; sample_id", "sample_metadata",
        "primary_diagnosis", "author sample accession -> native slide accession",
        "Set the code-matched specimen to distal cholangiocarcinoma; retain other cohort diagnoses",
        "Article methods report four PDAC cases and one distal cholangiocarcinoma case; "
        "the author script identifies the corresponding specimen accession.")])], ignore_index=True)
    save_input_tables(OUTPUT, pd.DataFrame(columns=images.columns), samples, panel, fields)

    review_path = OUTPUT / "review_results.csv"
    with formal_read_lease(review_path):
        with resource_lock(OUTPUT, shared=True):
            reviews = pd.read_csv(review_path, dtype=str).fillna("")
    reviews = reviews.loc[reviews.preparer.ne(REPAIR_PREPARER)]
    evidence = ";".join(map(relative, (supplement, author_script)))
    additions = [
        dict(study=STUDY, preparer=REPAIR_PREPARER, reviewer="", scope="sample",
             record_id=STUDY, check_id="M01", result="unresolved", source_evidence=evidence,
             output_evidence=";".join(map(relative, (OUTPUT / "sample_metadata.csv", OUTPUT / "field_mapping.csv"))),
             details=json.dumps(dict(scientific_repair_round=1, action="Linked four unique Table1 patient labels; corrected coverage mapping",
                                     sample_rows=5, sample_linked_images=75, patient_linked_samples=4,
                                     patient_linked_images=60, unmatched_sample_rows=1, unmatched_images=15,
                                     next_step="Independent recheck of patient links and remaining gap"))),
        dict(study=STUDY, preparer=REPAIR_PREPARER, reviewer="", scope="sample",
             record_id=STUDY, check_id="M02", result="unresolved", source_evidence=evidence,
             output_evidence=";".join(map(relative, (OUTPUT / "sample_metadata.csv", OUTPUT / "field_mapping.csv"))),
             details=json.dumps(dict(scientific_repair_round=1, action="Corrected one code-matched diagnosis from author note and article methods",
                                     next_step="Independent recheck of diagnosis and source scope"))),
        dict(study=STUDY, preparer=REPAIR_PREPARER, reviewer="", scope="study",
             record_id=STUDY, check_id="D01", result="unresolved", source_evidence=evidence,
             output_evidence=relative(review_path),
             details=json.dumps(dict(scientific_repair_round=1, image_files=75,
                                     sample_rows=5, patient_linked_samples=4,
                                     reason="Independent recheck and current input binding pending"))),
    ]
    staged = pd.concat([reviews, pd.DataFrame(additions)], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="lyman-repair-", dir=LOCAL) as temporary:
        local_review = Path(temporary) / "review_results.csv"
        staged.to_csv(local_review, index=False, encoding="utf-8-sig")
        with formal_publish_lease(review_path, directory=OUTPUT):
            partial = review_path.with_name(review_path.name + ".partial")
            shutil.copyfile(local_review, partial)
            os.replace(partial, review_path)
    save_json(REPAIR_JOB / "progress.json", dict(study=STUDY, phase="repair",
              scientific_repair_round=1, stage="repair_published_pending_independent_review",
              expected_images=75, exported_images=75, sample_rows=5,
              sample_linked_images=75, patient_linked_samples=4, patient_linked_images=60,
              next_step="independent review M01 M02 D01", updated_at=datetime.now(timezone.utc).isoformat()))


REPAIR_JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0292_repair_1"
REPAIR_PREPARER = "Lyman2025_IMC_PanIN_0292_repair_1"
REPAIR_2_JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0302_repair_2"
REPAIR_2_PREPARER = "Lyman2025_IMC_PanIN_0302_repair_2"


def repair_duplicate_field_mapping() -> None:
    """Remove the two duplicate clinical rules found by independent review."""
    field_path = OUTPUT / "field_mapping.csv"
    review_path = OUTPUT / "review_results.csv"
    with formal_read_lease(OUTPUT):
        with resource_lock(OUTPUT, shared=True):
            field_version = (field_path.stat().st_size, field_path.stat().st_mtime_ns)
            review_version = (review_path.stat().st_size, review_path.stat().st_mtime_ns)
            fields = pd.read_csv(field_path, dtype=str).fillna("")
            reviews = pd.read_csv(review_path, dtype=str).fillna("")

    patient = (fields.target_table.eq("sample_metadata") &
               fields.target_field.eq("patient_key;sample_id") &
               fields.source_field.eq("Tables 1,4,5,10"))
    diagnosis = (fields.target_table.eq("sample_metadata") &
                 fields.target_field.eq("primary_diagnosis") &
                 fields.source_field.eq("cholangio exclusion comment; sample_id"))
    assert int(patient.sum()) in (1, 2) and int(diagnosis.sum()) in (1, 2)
    assert len(fields) in (9, 11)
    if len(fields) == 11:
        assert int(patient.sum()) == int(diagnosis.sum()) == 2
        patient_rows = fields.loc[patient]
        assert patient_rows.drop(columns="join_keys").nunique().eq(1).all()
        assert set(patient_rows.join_keys) == {
            "No native MCD file/acquisition key supplied",
            "No native MCD file/acquisition-to-patient key supplied",
        }
        assert fields.loc[diagnosis].nunique().eq(1).all()
        remove_patient = patient & fields.join_keys.eq("No native MCD file/acquisition key supplied")
        remove_diagnosis = diagnosis & fields.duplicated(keep="first")
        fields = fields.loc[~(remove_patient | remove_diagnosis)].reset_index(drop=True)
    assert len(fields) == 9 and int((fields.target_field == "primary_diagnosis").sum()) == 1

    existing = reviews.preparer.eq(REPAIR_2_PREPARER)
    if not existing.any():
        source = relative(META / "source_metadata/jci.insight.191595.sdt1-10.xlsx") + ";" + relative(
            META / "source_metadata/IMCpipeline_humanpanin_withfunctions_TLSsample_analysis.R")
        output = relative(field_path)
        additions = [dict(study=STUDY, preparer=REPAIR_2_PREPARER, reviewer="",
                          scope="sample" if check == "M02" else "study", record_id=STUDY,
                          check_id=check, result="unresolved", source_evidence=source,
                          output_evidence=output + ";" + relative(review_path),
                          details=json.dumps({"scientific_repair_round": 2,
                                              "action": "Removed duplicate patient and diagnosis mapping rules",
                                              "next_step": "Independent M02 review and D01 rebinding for 60 previously accepted images",
                                              "remaining_gap": "One sample group and 15 images lack a direct patient link"}))
                     for check in ("M02", "D01")]
        reviews = pd.concat([reviews, pd.DataFrame(additions)], ignore_index=True)

    save_json(REPAIR_2_JOB / "progress.json", dict(
        study=STUDY, phase="repair", scientific_repair_round=2, stage="staged",
        expected_images=75, exported_images=75, sample_rows=5, sample_linked_images=75,
        patient_linked_samples=4, next_step="publish field_mapping and preparer review rows"))
    with tempfile.TemporaryDirectory(prefix="lyman-repair-2-", dir=LOCAL) as temporary:
        staged_field = Path(temporary) / field_path.name
        staged_review = Path(temporary) / review_path.name
        fields.to_csv(staged_field, index=False, encoding="utf-8-sig")
        reviews.to_csv(staged_review, index=False, encoding="utf-8-sig")
        with formal_publish_lease(field_path, directory=OUTPUT):
            assert (field_path.stat().st_size, field_path.stat().st_mtime_ns) == field_version
            assert (review_path.stat().st_size, review_path.stat().st_mtime_ns) == review_version
            if len(pd.read_csv(field_path)) == 11:
                partial_field = field_path.with_name(field_path.name + ".repair-2.partial")
                shutil.copyfile(staged_field, partial_field)
                os.replace(partial_field, field_path)
            if not existing.any():
                partial_review = review_path.with_name(review_path.name + ".repair-2.partial")
                shutil.copyfile(staged_review, partial_review)
                os.replace(partial_review, review_path)
    save_json(REPAIR_2_JOB / "progress.json", dict(
        study=STUDY, phase="repair", scientific_repair_round=2,
        stage="repair_published_pending_independent_review", expected_images=75,
        exported_images=75, sample_rows=5, sample_linked_images=75,
        patient_linked_samples=4, next_step="independent M02 review and D01 rebinding"))


if "--repair-2" in sys.argv:
    LOCAL.mkdir(parents=True, exist_ok=True)
    repair_duplicate_field_mapping()
    raise SystemExit(0)
if "--repair" in sys.argv:
    LOCAL.mkdir(parents=True, exist_ok=True)
    repair_clinical_metadata()
    raise SystemExit(0)


proof_path = META / "acquisition_verification.json"
proof = json.loads(proof_path.read_text())
manifest = json.loads((META / "source_manifest.json").read_text())
accepted = set(json.loads((META / "accepted_images.json").read_text())["accepted_image_ids"])
assert len(proof["files"]) == len(manifest["source_files"]) == 13
assert all(item["verification_complete"] and item["file_readable"] for item in proof["files"])
for file_record in proof["files"]:
    path = RAW / file_record["source_file"]
    assert path.is_file() and path.stat().st_size == file_record["expected_bytes"] == file_record["actual_bytes"]
acquisitions = [acq for file_record in proof["files"] for acq in file_record["acquisitions"]]
assert len(acquisitions) == len(accepted) == 75
assert {acq["native_acquisition_id"] for acq in acquisitions} == accepted
assert all(acq["counted"] and acq["source_order_header_match"] and
           acq["bounded_native_read"]["valid"] for acq in acquisitions)
source_names = [item["native_label"] for item in acquisitions[0]["channels"]]
assert all([item["native_label"] for item in acq["channels"]] == source_names
           for acq in acquisitions)
assert len(source_names) == 49

# The published human antibody table identifies both CD45 isotopes. The
# collection mapping selects 89Y as the counted measurement and 152 as duplicate.
panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1, drop_indices={19: "duplicate CD45; 89Y selected"})
assert panel.loc[1, "source_name"] == panel.loc[19, "source_name"] == "CD45"
assert panel.loc[1, "role"] == "downstream" and not panel.loc[19, "keep_channel"]
for index in (45, 46, 47):
    assert panel.loc[index, "role"] == "segmentation-only"
    panel.loc[index, ["body", "keep_channel", "keep_reason", "exclusion_reason"]] = [
        True, True, "Published membrane segmentation kit channel", ""]
panel.loc[panel.keep_channel, "output_channel_index"] = range(int(panel.keep_channel.sum()))
panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
kept = panel[panel.keep_channel].sort_values("output_channel_index")

image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
# The native slide filenames preserve surgical accessions that bridge four
# specimen groups to unique patient labels in the public supplemental table.
slides = {}
for file_record in proof["files"]:
    path = RAW / file_record["source_file"]
    with formal_read_lease(path):
        with MCDFile(path) as mcd:
            assert len(mcd.slides) == 1
            slide = mcd.slides[0]
            basename = Path(slide.metadata["Filename"].replace("\\", "/")).stem
            codes = re.findall(r"(?<!\d)\d{5}(?!\d)", basename)
            assert len(codes) == 1, path
            steps = {(acq.metadata["AblationDistanceBetweenShotsX"],
                      acq.metadata["AblationDistanceBetweenShotsY"])
                     for acq in slide.acquisitions}
            assert steps == {("1", "1")}, path
            slides[file_record["source_file"]] = dict(source_filename=basename,
                                                        source_sample_id=codes[0], pixel_size_um=1.0)
assert len({slide["source_sample_id"] for slide in slides.values()}) == 5
save_json(CACHE / "source_slide_mapping.json", slides)
supplement = META / "source_metadata/jci.insight.191595.sdt1-10.xlsx"
author_script = META / "source_metadata/IMCpipeline_humanpanin_withfunctions_TLSsample_analysis.R"
patient_keys, cholangio_code = clinical_links(
    supplement, author_script, {slide["source_sample_id"] for slide in slides.values()})
samples = pd.DataFrame([{
    **{name: pd.NA for name in sample_columns}, "study": STUDY,
    "sample_id": "resection_" + code, "source_sample_id": code,
    "patient_key": patient_keys.get(code, pd.NA),
    "sample_type": "surgical resection specimen", "anatomical_site": "pancreas",
    "collection_method": "resection", "preservation_method": "ffpe",
    "primary_diagnosis": ("distal cholangiocarcinoma" if code == cholangio_code
                          else "pancreatic ductal adenocarcinoma"),
    "metadata_notes": ("Native slide accession matches the author analysis exclusion for cholangiocarcinoma; "
                       "the article identifies one distal cholangiocarcinoma case. ROI pathology class remains unassigned."
                       if code == cholangio_code else
                       "Native slide accession uniquely matches a Table1 LCM sample ID and its patient label; "
                       "the exact tissue block and ROI pathology class remain unassigned."
                       if code in patient_keys else
                       "Native slide accession groups the specimen; Table1 and the author analysis script do not "
                       "provide a direct patient match. ROI pathology class remains unassigned.")
} for code in sorted({slide["source_sample_id"] for slide in slides.values()})], columns=sample_columns)
fields = pd.DataFrame([
    field(proof_path, "source_file;native_acquisition_id;native_metadata.Description", "image_metadata",
          "source_image_id;source_acquisition_id;roi_id;region_annotation", "source_file + acquisition_id",
          "Preserve each complete native acquisition and source description; no disease class inferred",
          "75 qualified acquisitions in 13 containers; ROI descriptions repeat across containers and split labels remain distinct."),
    field(proof_path, "width;height;native_channel_count;bounded_native_read", "image_metadata",
          "source_x1_px;source_y1_px;source_channel_count;source_dtype", "native_acquisition_id",
          "Use complete native frame, float32 decoded MCD counts and centered square padding",
          "Native verified read covers a bounded block; full-image admission requires downstream signal."),
    field(CACHE / "duplicate_coordinate_review.json", "repeated_origin_records;valid_pixels", "image_metadata",
          "valid_pixels_path;metadata_notes", "12.mcd#acquisition_id=1",
          "Preserve uniquely positioned raw values and mark ambiguous origin/unmeasured pixels invalid",
          "One source acquisition contains repeated (0,0) coordinates; the source does not locate those records elsewhere."),
    field(META / "channel_mapping.json", "index;mass;metal;native_label;counted", "panel_mapping",
          "source_channel_index;source_name;keep_channel", "native index + isotope",
          "Match exact source order to S1; retain 89Y CD45 and omit duplicate 152 channel",
          "The published human panel lists both CD45 measurements; 152 metal differs from the native annotation and remains a review item."),
    field(S1, "Original marker;Unified marker;Segmentation channel", "panel_mapping",
          "marker_id;mapping_status;role;nuclear;body", "Study + Original marker",
          "Use study-specific S1 names; retain Ir nuclei and published PM2/3/4 membrane inputs",
          "S1 segmentation status for PM2/3/4 is blank; source label and paper methods identify segmentation-kit channels."),
    field(supplement, "Tables 1,4,5,10", "sample_metadata", "patient_key;sample_id",
          "native slide accession -> Table1 LCM sample ID -> Patient",
          "Assign a patient key only for unique Table1 accession matches",
          "Five specimen groups and 75 image-to-sample links; four patient matches cover 60 images, one group remains unresolved."),
    field(CACHE / "source_slide_mapping.json", "Filename;AblationDistanceBetweenShotsX/Y", "image_metadata",
          "sample_id;pixel_size_x_um;pixel_size_y_um", "source MCD file -> slide Filename -> five-digit accession",
          "Group acquisitions by source accession and use native 1 um by 1 um ablation step",
          "Five source accession groups across 13 slides; these are specimen-level groups, without a confirmed patient key or exact tissue block bridge."),
    field(author_script, "metaDataFile;file_name;sample_id;Patient;Region", "sample_metadata", "sample_id;patient_key",
          "External metadata workbook not published in repository", "Do not infer keys from analysis sample labels",
          "Public author TLS labels corroborate accessions; the referenced metadata workbook and MCD-to-FCS bridge are absent."),
    field(author_script, "cholangio exclusion comment; sample_id", "sample_metadata", "primary_diagnosis",
          "author sample accession -> native slide accession",
          "Set the code-matched specimen to distal cholangiocarcinoma; retain other cohort diagnoses",
          "Article methods report four PDAC cases and one distal cholangiocarcinoma case."),
])

LOCAL.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
index_path = OUTPUT / "images.parquet"
images = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame(columns=image_columns)
decisions_path = CACHE / "acquisition_decisions.json"
decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
save_input_tables(OUTPUT, pd.DataFrame(columns=image_columns), samples, panel, fields)
max_side = max(max(acq["width"], acq["height"]) for acq in acquisitions)
output_bound = sum(len(kept) * max(acq["width"], acq["height"]) ** 2 * 4 for acq in acquisitions)
largest_source = max(item["expected_bytes"] for item in manifest["source_files"])
assert shutil.disk_usage(NAS).free > (100 << 30) + output_bound
assert shutil.disk_usage(LOCAL).free > (64 << 30) + largest_source + len(kept) * max_side ** 2 * 4
checkpoint("preflight", images, decisions, output_bound_bytes=output_bound,
           largest_local_source_bytes=largest_source)

# Reuse published images and decisions; copy one verified source at a time so
# decoding and TIFF construction use local storage outside the NAS writer slot.
for file_record in proof["files"]:
    pending = [acq for acq in file_record["acquisitions"]
               if decisions.get(acq["native_acquisition_id"], {}).get("pixel_size_um") != 1.0]
    if not pending:
        continue
    source = RAW / file_record["source_file"]
    local_source = LOCAL / source.name
    with formal_read_lease(source):
        if not local_source.exists() or local_source.stat().st_size != source.stat().st_size:
            partial = local_source.with_name(local_source.name + ".partial")
            shutil.copyfile(source, partial)
            os.replace(partial, local_source)
    assert local_source.stat().st_size == file_record["expected_bytes"]
    for acq in pending:
        native_id = acq["native_acquisition_id"]
        image_id = f"lyman_{source.stem}_a{int(acq['acquisition_id']):03d}"
        destination = OUTPUT / "images" / image_id / "input.tiff"
        specimen = SourceSpec(image_id, PANEL_ID, "mcd", local_source, tuple(range(49)),
                              acquisition_id=int(acq["acquisition_id"]),
                              pixel_size_um=(1.0, 1.0),
                              source_revision="zenodo:8336719",
                              intensity_description="Native MCD decoded float32 intensities; no normalization")
        with acquisition_reader(specimen, acq) as reader:
            assert reader.shape == (49, acq["height"], acq["width"])
            result = export_input(reader, panel, destination)
        if result["admitted"]:
            with formal_read_lease(destination):
                with tifffile.TiffFile(destination) as tif:
                    assert tif.series[0].shape == (len(kept), result["height_px"], result["width_px"])
                    assert tif.series[0].dtype.name == "float32" and tif.series[0].axes == "CYX"
            row = {name: pd.NA for name in image_columns}
            row.update(study=STUDY, image_id=image_id,
                       sample_id="resection_" + slides[source.name]["source_sample_id"],
                       panel_id=PANEL_ID, image_kind="roi",
                       source_image_id=native_id, source_files=[relative(source)], source_format="mcd",
                       source_acquisition_id=str(acq["acquisition_id"]), source_axes="CYX",
                       source_dtype="float32", source_revision="zenodo:8336719",
                       roi_id=native_id, region_annotation=acq["native_metadata"].get("Description"),
                       source_x0_px=0, source_y0_px=0, source_x1_px=acq["width"],
                       source_y1_px=acq["height"], height_px=result["height_px"],
                       width_px=result["width_px"], source_channel_count=49,
                       channel_count=len(kept), pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                       input_image_path=relative(destination),
                       valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                          if result["valid_pixels"] else pd.NA),
                       intensity_description=specimen.intensity_description,
                       split_method="source_acquisition", split_evidence="Complete MCD acquisition frame",
                       metadata_notes=("Source accession groups the resection specimen; patient identity and ROI disease region lack a direct public acquisition map. "
                                       + ("Repeated-origin source records have no recoverable coordinates; consult valid_pixels.tiff and duplicate_coordinate_review.json."
                                          if native_id == "12.mcd#acquisition_id=1" else "")),
                       pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
                       pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"])
            images = pd.concat([images.loc[images.image_id.ne(image_id)], pd.DataFrame([row])],
                               ignore_index=True)
            save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panel, fields)
        decisions[native_id] = dict(admitted=bool(result["admitted"]), image_id=image_id,
                                    reason="downstream signal" if result["admitted"] else "all downstream channels zero",
                                    source_file=source.name, pixel_size_um=1.0)
        save_json(decisions_path, decisions)
        checkpoint("exporting", images, decisions, last_acquisition=native_id)
        print(f"{len(decisions)}/{len(acquisitions)} {image_id} admitted={result['admitted']}", flush=True)
    local_source.unlink()

# The preparer records source accounting and the unresolved identity bridge;
# an independent reviewer will provide the actual I/P/M/D conclusions.
review_path = OUTPUT / "review_results.csv"
rows = [
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="unresolved",
         source_evidence=";".join((relative(proof_path), relative(CACHE / "duplicate_coordinate_review.json"))),
         output_evidence=";".join((relative(decisions_path), relative(CACHE / "overview_initial.jpg"),
                                   relative(CACHE / "overview_tiny_06.jpg"), relative(CACHE / "overview_tiny_10.jpg"),
                                   relative(CACHE / "overview_split_11_13.jpg"), relative(CACHE / "overview_12_valid.jpg"))),
         details=json.dumps({"candidate_acquisitions": len(acquisitions), "admitted": len(images),
                             "excluded_zero_signal": sum(not item["admitted"] for item in decisions.values()),
                             "qualification": "All 75 native acquisitions readable; inspected representative, narrow, split and eight 50x50 tissue frames.",
                             "coordinate_gap": "12.mcd acquisition 1 has 72725 ambiguous origin records; 567275 uniquely positioned pixels retained with a valid mask.",
                             "pending": "Independent tissue/use and native coordinate review; ROI disease class unavailable."})),
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample", record_id=STUDY,
         check_id="M01", result="unresolved", source_evidence=";".join((relative(CACHE / "source_slide_mapping.json"),
                                                                          relative(supplement), relative(author_script))),
         output_evidence=relative(OUTPUT / "sample_metadata.csv"),
         details=json.dumps({"sample_rows": len(samples), "sample_linked_images": int(images.sample_id.notna().sum()),
                             "patient_linked_samples": 0,
                             "gap": "Native slide accession groups are connected; no explicit accession-to-patient map or ROI disease annotation."})),
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
         check_id="D01", result="unresolved", source_evidence=relative(proof_path),
         output_evidence=relative(index_path),
         details="Preparer export closed; independent review and current input binding pending."),
]
with tempfile.TemporaryDirectory(prefix="lyman-review-", dir=LOCAL) as temporary:
    staged = Path(temporary) / "review_results.csv"
    with staged.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pd.read_csv(TEMPLATES / "review_results.csv").columns)
        writer.writeheader()
        writer.writerows(rows)
    with formal_publish_lease(review_path, directory=OUTPUT):
        partial = review_path.with_name(review_path.name + ".partial")
        shutil.copyfile(staged, partial)
        os.replace(partial, review_path)
checkpoint("prepared_pending_independent_review", images, decisions)
