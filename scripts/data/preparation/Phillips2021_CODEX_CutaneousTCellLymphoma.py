"""Prepare the 70 verified Phillips CODEX TMA spots as native intensity inputs.

The Spora tissue table links TMA spots to biopsies and patient identifiers.
Each image is closed and indexed before the next source is opened. Independent
review remains a separate task.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
from types import SimpleNamespace
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile
import zarr

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, _export_input, panel_from_s1, save_input_tables,
)


STUDY = "Phillips2021_CODEX_CutaneousTCellLymphoma"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "spora"
SOURCE = RAW / "codex/0_37744mpp/images"
OUTPUT = ROOT / "processed" / STUDY
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0280_prepare_0")))
PROOF = ROOT / "raw/public" / STUDY / "manifest.json"
OBJECTS = PROOF
ORDERED = metadata_source_directory(ROOT, STUDY) / 'spora_inventory' / 'codex_ordered_marker_names.json'
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "codex_native56"
PIXEL_SIZE_UM = 0.37744
PREPARER = "codex_phillips_0280_prepare_0"
GIB = 1024 ** 3
CLINICAL_SOURCE_FIELDS = {
    "Age_at_screening_years": "source_age_at_screening_years",
    "Biopsy_timepoints": "source_biopsy_timepoints",
    "Disease_stage": "source_disease_stage",
    "Response_status": "source_response_status",
    "Prior_therapies": "source_prior_therapies",
    "mSWAT_change": "source_mswat_change",
    "C02_weeks": "source_c02_weeks",
    "Death_time_weeks": "source_death_time_weeks",
    "EOT_time_weeks": "source_eot_time_weeks",
    "Follow_up_time_weeks": "source_follow_up_time_weeks",
    "Response_progression_time_weeks": "source_response_progression_time_weeks",
}
CLINICAL_FIELD_SPECS = {
    "Age_at_screening_years": ("number", "Age at screening as reported by the source", "years", "Screening; relation to biopsy and treatment index is not established"),
    "Biopsy_timepoints": ("string", "Source-reported biopsy timepoint summary", None, "Source labels; no elapsed-time origin is defined"),
    "Disease_stage": ("string", "Source-reported disease stage", None, "Stage assessment time relative to each biopsy is not established"),
    "Response_status": ("string", "Source-reported response status", None, "Response assessment time relative to each biopsy is not established"),
    "Prior_therapies": ("integer", "Source-reported number of prior therapies", "therapies", "Look-back endpoint and biopsy-specific timing are not established"),
    "mSWAT_change": ("number", "Source-reported change in modified Severity Weighted Assessment Tool score", None, "Change interval and calculation unit are not independently established"),
    "C02_weeks": ("number", "Source-reported C02 time", "weeks", "Reference event and C02 event definition are not independently established"),
    "Death_time_weeks": ("number", "Source-reported death time", "weeks", "Reference event and event ascertainment are not independently established"),
    "EOT_time_weeks": ("number", "Source-reported end-of-treatment time", "weeks", "Reference event is not independently established"),
    "Follow_up_time_weeks": ("number", "Source-reported follow-up time", "weeks", "Reference event and censoring definition are not independently established"),
    "Response_progression_time_weeks": ("number", "Source-reported response or progression time", "weeks", "Reference event and response/progression event definition are not independently established"),
}
COUNTS = {"exported_images": 0, "sample_rows": 0,
          "sample_linked_images": 0, "patient_linked_samples": 0}

zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})


class WaitingIO(RuntimeError):
    """A closed image is waiting for a shared publication slot or space."""


def relative(path: Path) -> str:
    return metadata_source_relative(path, ROOT).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


@contextmanager
def publish_slot(destination: Path, *, directory: Path, lock_name: str = ".publication.lock"):
    def timeout(signum: int, frame: object) -> None:
        raise WaitingIO("Shared NAS publication lease unavailable for 60 seconds")

    with ExitStack() as stack:
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            stack.enter_context(formal_publish_lease(destination, directory=directory,
                                                    lock_name=lock_name))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        yield


def publish_file(source: Path, destination: Path, *, directory: Path,
                 lock_name: str = ".publication.lock") -> None:
    with publish_slot(destination, directory=directory, lock_name=lock_name):
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, partial)
        os.replace(partial, destination)


def progress(stage: str, *, phase: str = "prepare", repair_round: int = 0,
             **details: object) -> None:
    state = dict(study=STUDY, phase=phase, scientific_repair_round=repair_round,
                 stage=stage, status="waiting_io" if stage == "waiting_io" else stage,
                 expected_images=70, **COUNTS,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    TEMP.mkdir(parents=True, exist_ok=True)
    local = TEMP / "progress.json"
    write_json(local, state)
    write_json(JOB / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> tuple[pd.DataFrame, list[dict]]:
    with formal_read_lease(SOURCE):
        proof = json.loads(PROOF.read_text())["preparation_sources"]["native_images"]
        objects = json.loads(OBJECTS.read_text())["preparation_sources"]["source_objects"]
        ordered = json.loads(ORDERED.read_text())
        tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
        channel_names = pd.read_parquet(RAW / "codex/channels.parquet").channel_name.tolist()
        availability = pd.read_parquet(RAW / "codex/channels_per_tissue.parquet")
        assert proof["source_image_count"] == proof["verified_count"] == 70
        assert proof["source_object_count"] == len(objects) == 420
        assert proof["source_bytes"] == sum(item["size"] for item in objects)
        tissue = tissues[tissues.modality.eq("codex")].copy()
        assert len(tissue) == tissue.tissue_id.nunique() == 70
        verified = {item["image_id"]: item for item in proof["verified"]}
        assert set(tissue.tissue_id) == set(verified) == set(ordered)
        assert {path.name.removesuffix(".ome.zarr") for path in SOURCE.glob("*.ome.zarr")} == set(verified)
        assert channel_names == availability.columns.tolist()
        assert len(channel_names) == 56 and len(availability) == 70
        expected = {}
        prefix = "spora-submissions/phillips2021immune/codex/0_37744mpp/images/"
        for item in objects:
            suffix = item["key"].removeprefix(prefix)
            assert suffix != item["key"]
            path = SOURCE / suffix
            stat = path.stat()
            assert stat.st_size == item["size"], relative(path)
            identity = suffix.split("/", 1)[0].removesuffix(".ome.zarr")
            expected.setdefault(identity, []).append(item["size"])
        assert set(expected) == set(verified)
        scope = []
        for row in tissue.sort_values("tissue_id").itertuples():
            path = SOURCE / f"{row.tissue_id}.ome.zarr"
            group = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
            array = json.loads((path / "0/zarr.json").read_text())
            labels = [channel["label"] for channel in group["omero"]["channels"]]
            declared = [name for name, keep in zip(channel_names, availability.loc[row.tissue_id], strict=True) if keep]
            proof_row = verified[row.tissue_id]
            assert labels == declared == ordered[row.tissue_id]
            assert array["shape"] == proof_row["shape"] == [56, 1440, 1920]
            assert array["data_type"] == proof_row["dtype"] == "uint16"
            assert array["dimension_names"] == ["c", "y", "x"]
            assert len(expected[row.tissue_id]) == proof_row["source_object_count"] == 6
            assert sum(expected[row.tissue_id]) == proof_row["source_bytes"]
            assert group["multiscales"][0]["datasets"][0]["path"] == "0"
            scope.append(dict(image_id=row.tissue_id, path=path, labels=labels,
                              revision=f"spora:phillips2021immune;bytes={proof_row['source_bytes']}",
                              shape=array["shape"]))
    return tissue, scope


def sample_tables(tissue: pd.DataFrame) -> pd.DataFrame:
    tissue = tissue.copy()
    tissue["biopsy_group"] = tissue.Biopsy.str.replace(r"_[123]$", "", regex=True)
    assert tissue.Biopsy.str.fullmatch(r"(?:Pre|EOT|Resp|C02)_[123]").all()
    assert (tissue.PatientID.astype(str) == tissue.Patient_ID.astype(str)).all()
    assert (tissue.PatientID.astype(str) == tissue.original_patient_identifier.astype(str)).all()
    assert tissue.groupby("patient_id").PatientID.nunique().eq(1).all()
    grouped = tissue.groupby(["patient_id", "biopsy_group"], sort=True)
    fields = ("Age_at_screening_years", "Diagnosis", *CLINICAL_SOURCE_FIELDS)
    assert all(grouped[field].nunique(dropna=False).le(1).all() for field in fields)
    assert len(grouped) == 33 and grouped.size().between(2, 3).all()
    samples = []
    for (patient, timepoint), group in grouped:
        source = group.iloc[0]
        sample = dict.fromkeys(columns("sample_metadata"))
        sample.update(study=STUDY, sample_id=f"{patient}_{timepoint}",
                      patient_key=patient, source_subject_id=str(source.PatientID),
                      source_sample_id=f"{source.PatientID}_{timepoint}",
                      sample_type="skin biopsy", tissue_type="tumor",
                      primary_site="Skin", primary_diagnosis="Cutaneous T-cell lymphoma",
                      histology={"MF": "Mycosis fungoides", "SS": "Sezary syndrome"}[source.Diagnosis],
                      timepoint_label=timepoint,
                      preservation_method="ffpe", processing_method="tissue microarray",
                      case_control_status="case", index_therapy_modality="immunotherapy",
                      index_regimen="pembrolizumab",
                      metadata_notes=("Spora patient_id and source PatientID agree; Biopsy suffixes 1-3 "
                                      "are separate TMA spots from the same biopsy timepoint. "
                                      "Source endpoints lack an explicit per-field origin and are not converted."))
        for source_field, target_field in CLINICAL_SOURCE_FIELDS.items():
            value = source[source_field]
            sample[target_field] = None if pd.isna(value) else value
        samples.append(sample)
    return pd.DataFrame(samples, columns=[*columns("sample_metadata"), *CLINICAL_SOURCE_FIELDS.values()])


def field_rows() -> pd.DataFrame:
    basis = dict(study=STUDY, source_sheet_or_table="", value_mapping="{}",
                 time_origin_rule="", conflict_rule="Conflicting source values remain empty pending review")
    rules = [
        (RAW / "metadata/tissues.parquet", "patient_id; PatientID; Patient_ID; original_patient_identifier; Biopsy", "image_metadata", "sample_id", "patient_id + Biopsy without TMA spot suffix", "Group 2-3 spots of each biopsy timepoint into one biological sample"),
        (RAW / "metadata/tissues.parquet", "patient_id; PatientID; Patient_ID; original_patient_identifier; Biopsy", "sample_metadata", "sample_id; patient_key; source_subject_id; source_sample_id; timepoint_label", "patient_id + Biopsy timepoint", "Require all patient aliases to agree; retain original ID and pre/post biopsy timepoint"),
        (RAW / "metadata/tissues.parquet", "Diagnosis", "sample_metadata", "primary_diagnosis; histology", "patient_id + Biopsy timepoint", "MF and SS use their published disease names"),
        (RAW / "metadata/tissues.parquet", "Biopsy; TMA_spot; original_file_name", "image_metadata", "region_annotation; core_id; roi_id; source_image_id", "exact tissue_id", "Preserve individual biopsy spot and physical TMA spot; one ROI per original acquisition"),
        (ORDERED, "ordered marker names; channels.parquet; channels_per_tissue.parquet", "panel_mapping", "source_name; marker_id; role; keep_channel; output_channel_index", "tissue_id and source channel index", "Match exact embedded channel order to S1; preserve both nuclear measurements"),
        (SOURCE / "<tissue_id>.ome.zarr", "0/zarr.json; zarr.json; decoded array", "image_metadata", "source_files; source_format; source_axes; source_dtype; source_revision; source_x0_px; source_y0_px; source_x1_px; source_y1_px; pixel_size_x_um; pixel_size_y_um; input_image_path; height_px; width_px; pad_left_px; pad_right_px; pad_top_px; pad_bottom_px; source_channel_count; channel_count", "tissue_id", "Use the highest published 0_37744mpp CYX array, original uint16 values and full ROI; square center padding. Embedded OME scale [1,1,1] conflicts with resolution-named source path and needs independent scale review"),
        ("https://www.nature.com/articles/s41467-021-26974-6", "Results: patient cohort and multimodal experimental approach", "sample_metadata", "sample_type; tissue_type; primary_site; preservation_method; processing_method; case_control_status; index_therapy_modality; index_regimen", "study cohort and patient_id + Biopsy timepoint", "Original study describes skin tumor biopsies, FFPE TMA spots and pembrolizumab treatment; immunotherapy is the therapy class"),
        ("https://www.nature.com/articles/s41467-021-26974-6", "Results: 70 TMA regions from 14 patients", "image_metadata", "image_kind; split_method; split_evidence", "tissue_id and source TMA_spot", "Each source object is an independent imaged TMA spot, not a multiscale duplicate"),
    ]
    rows = []
    for path, source_field, table, target, keys, rule in rules:
        source_location = path if isinstance(path, str) else relative(path)
        rows.append(dict(**basis, source_file=source_location, source_field=source_field,
                         target_table=table, target_field=target, join_keys=keys,
                         transformation_rule=rule,
                         notes="Local Spora tissue table, ordered channels, 70-image source proof and original article checked; 70 image links, 33 biopsy samples, 14 patient keys."))
    for source_field, target_field in CLINICAL_SOURCE_FIELDS.items():
        data_type, definition, unit, time_semantics = CLINICAL_FIELD_SPECS[source_field]
        row = dict(basis)
        row.update(source_file=relative(RAW / "metadata/tissues.parquet"),
                   source_field=source_field, target_table="sample_metadata",
                   target_field=target_field, join_keys="patient_id + Biopsy timepoint",
                   transformation_rule="Copy reported source value without endpoint or response reclassification",
                   time_origin_rule=time_semantics,
                   notes=json.dumps(dict(data_type=data_type, definition=definition,
                                         unit=unit, time_semantics=time_semantics,
                                         missing="Source null remains empty; no endpoint reclassification")))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def repair_m02() -> None:
    """Correct the reviewed clinical mapping while retaining independent results."""
    with formal_read_lease(OUTPUT):
        with resource_lock(OUTPUT, shared=True):
            fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str).fillna("")
            review = pd.read_csv(OUTPUT / "review_results.csv", dtype=str).fillna("")
            images = pd.read_parquet(OUTPUT / "images.parquet", columns=["image_id", "sample_id", "input_image_path"])
            samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
    assert len(images) == 70 and len(samples) == 33
    assert images.sample_id.notna().sum() == 70 and samples.patient_key.notna().sum() == 33
    COUNTS.update(exported_images=70, sample_rows=33, sample_linked_images=70,
                  patient_linked_samples=33)
    wrong = (fields.source_field.eq("Diagnosis; Age_at_screening_years") &
             fields.target_table.eq("sample_metadata") &
             fields.target_field.eq("primary_diagnosis; histology; age_at_index_years; index_event"))
    assert wrong.sum() <= 1
    fields = fields.loc[~wrong].copy()
    corrected = field_rows()
    for source_field, target_field in CLINICAL_SOURCE_FIELDS.items():
        selection = (fields.source_field.eq(source_field) & fields.target_table.eq("sample_metadata") &
                     fields.target_field.eq(target_field))
        assert selection.sum() == 1
        replacement = corrected[(corrected.source_field == source_field) &
                                (corrected.target_field == target_field)].iloc[0]
        fields.loc[selection, "time_origin_rule"] = replacement.time_origin_rule
        fields.loc[selection, "notes"] = replacement.notes
    assert not fields.target_field.str.contains("age_at_index_years|index_event").any()
    assert any(fields.target_field.eq("source_age_at_screening_years"))

    previous_d01 = review[(review.scope == "study") & (review.check_id == "D01")].iloc[-1]
    if previous_d01.result == "pass" and not wrong.any():
        progress("repair_already_reviewed", phase="repair", repair_round=1)
        return
    if previous_d01.result == "pass":
        revoked = dict.fromkeys(columns("review_results"), "")
        revoked.update(study=STUDY, preparer="codex_phillips_0398_repair_1", reviewer="",
                       scope="study", record_id=STUDY, check_id="D01", result="unresolved",
                       source_evidence=previous_d01.source_evidence,
                       output_evidence=previous_d01.output_evidence,
                       details=json.dumps(dict(repair_round=1, preparer_only=True,
                                               reason="M02 field mapping repaired; independent M02 review and D01 rebinding pending",
                                               affected_check="M02", image_ids=[])))
        review = pd.concat([review, pd.DataFrame([revoked])], ignore_index=True)
    else:
        assert previous_d01.result == "unresolved"

    TEMP.mkdir(parents=True, exist_ok=True)
    staged_review = TEMP / "repair_review_results.csv"
    staged_fields = TEMP / "repair_field_mapping.csv"
    review.to_csv(staged_review, index=False, encoding="utf-8-sig")
    fields.to_csv(staged_fields, index=False, encoding="utf-8-sig")
    progress("repair_staged", phase="repair", repair_round=1,
             corrected_mapping_rows=12, next_action="publish_review_then_field_mapping")
    try:
        with publish_slot(OUTPUT / "field_mapping.csv", directory=OUTPUT):
            for source, name in ((staged_review, "review_results.csv"),
                                 (staged_fields, "field_mapping.csv")):
                partial = OUTPUT / f"{name}.partial"
                shutil.copyfile(source, partial)
                os.replace(partial, OUTPUT / name)
    except WaitingIO as error:
        progress("waiting_io", phase="repair", repair_round=1,
                 reason=str(error), completed_scope="70 existing images and 33 samples unchanged",
                 next_action="publish_review_then_field_mapping")
        return
    progress("repair_pending_independent_review", phase="repair", repair_round=1,
             corrected_mapping_rows=12, next_action="independent_M02_review_and_D01_rebinding")


def image_row(item: dict, tissue: pd.Series, geometry: dict) -> dict:
    image_id = item["image_id"]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=image_id,
               sample_id=f"{tissue.patient_id}_{str(tissue.Biopsy).rsplit('_', 1)[0]}",
               panel_id=PANEL_ID, image_kind="tma_core", source_image_id=image_id,
               source_files=[relative(item["path"])], source_format="ome-zarr-v3",
               source_axes="CYX", source_dtype="uint16", source_revision=item["revision"],
               core_id=str(tissue.TMA_spot), roi_id=image_id,
               region_annotation=str(tissue.Biopsy),
               source_x0_px=0, source_y0_px=0, source_x1_px=1920, source_y1_px=1440,
               pixel_size_x_um=PIXEL_SIZE_UM, pixel_size_y_um=PIXEL_SIZE_UM,
               input_image_path=relative(OUTPUT / "images" / image_id / "input.tiff"),
               intensity_description="Published highest-resolution CODEX uint16 source, no intensity transform",
               split_method="already_independent",
               split_evidence="Spora independent TMA spot; original paper reports 70 regions from 14 patients",
               metadata_notes="Source biopsy and TMA spot from Spora tissues.parquet")
    for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = geometry[key]
    return row


def recovered_geometry(path: Path, panel: pd.DataFrame) -> dict:
    """Validate a previously published, unindexed TIFF before restoring its row."""
    expected_names = panel.sort_values("output_channel_index").marker_id.tolist()
    with formal_read_lease(path):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            assert series.axes == "CYX" and series.shape == (56, 1920, 1920)
            assert series.dtype == np.dtype("uint16") and len(tif.pages) == 56
            ome = ElementTree.fromstring(tif.ome_metadata or "")
            names = [channel.get("Name") for channel in ome.findall(".//{*}Channel")]
            assert names == expected_names
    return dict(height_px=1920, width_px=1920, pad_left_px=0, pad_right_px=0,
                pad_top_px=240, pad_bottom_px=240, source_channel_count=56,
                channel_count=56)


class LocalReader:
    def __init__(self, array: np.memmap, source: SourceSpec):
        self.array, self.source = array, source
        self.shape, self.dtype = array.shape, array.dtype

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def stage_source(item: dict, destination: Path) -> None:
    array = zarr.open_array(str(item["path"] / "0"), mode="r")
    assert list(array.shape) == item["shape"] and str(array.dtype) == "uint16"
    local = np.memmap(destination, dtype="uint16", mode="w+", shape=tuple(item["shape"]))
    try:
        shard_y, shard_x = array.metadata.chunk_grid.chunk_shape[1:]
        for y in range(0, item["shape"][1], shard_y):
            for x in range(0, item["shape"][2], shard_x):
                local[:, y:y + shard_y, x:x + shard_x] = array[:, y:y + shard_y, x:x + shard_x]
        local.flush()
    finally:
        local._mmap.close()


def review_table() -> pd.DataFrame:
    sources = [relative(PROOF), relative(OBJECTS), relative(RAW / "metadata/tissues.parquet"),
               relative(ORDERED), relative(S1)]
    outputs = [relative(OUTPUT / name) for name in
               ("images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv")]
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
                 check_id="D01", result="unresolved", source_evidence=json.dumps(sources),
                 output_evidence=json.dumps(outputs),
                 details=json.dumps(dict(preparer_only=True, independent_review="pending",
                                         image_ids=[], reason="Independent S01/I01-I05/P01-P03/M01-M02 review and D01 binding pending")))]
    return pd.DataFrame(rows, columns=columns("review_results"))


def run() -> None:
    tissue, scope = source_scope()
    samples = sample_tables(tissue)
    panel = panel_from_s1(STUDY, PANEL_ID, scope[0]["labels"], S1).drop(columns="s1_original_marker")
    assert len(panel) == 56 and panel.keep_channel.sum() == 56
    fields = field_rows()
    by_image = tissue.set_index("tissue_id", verify_integrity=True)
    with formal_read_lease(OUTPUT):
        prior = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
        prior_samples = (pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
                         if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame())
    indexed = {row.image_id: row for row in prior.itertuples(index=False)} if len(prior) else {}
    COUNTS.update(exported_images=len(prior), sample_rows=len(prior_samples),
                  sample_linked_images=int(prior.sample_id.notna().sum()) if len(prior) else 0,
                  patient_linked_samples=int(prior_samples.patient_key.notna().sum()) if len(prior_samples) else 0)
    pending: list[dict] = []

    def commit_batch() -> None:
        if not pending:
            return
        save_input_tables(OUTPUT, pd.DataFrame(pending), samples, panel, fields)
        for record in pending:
            indexed[record["image_id"]] = SimpleNamespace(**record)
        COUNTS.update(exported_images=len(indexed), sample_rows=len(samples),
                      sample_linked_images=sum(bool(value.sample_id) for value in indexed.values()),
                      patient_linked_samples=int(samples.patient_key.notna().sum()))
        progress("batch_completed", last_image=pending[-1]["image_id"], batch_images=len(pending))
        pending.clear()

    for item in scope:
        image_id = item["image_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id not in indexed and destination.is_file() and destination.stat().st_size > 0:
            geometry = recovered_geometry(destination, panel)
            pending.append(image_row(item, by_image.loc[image_id], geometry))
            if len(pending) == 4:
                commit_batch()
    commit_batch()

    progress("source_scope_checked", source_images=len(scope), biological_samples=len(samples),
             source_patient_keys=tissue.patient_id.nunique(),
             next_image=next((item["image_id"] for item in scope
                              if item["image_id"] not in indexed or
                              not (OUTPUT / "images" / item["image_id"] / "input.tiff").is_file()), None))
    staged_ids: set[str] = set()
    for position, item in enumerate(scope):
        image_id = item["image_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        previous = indexed.get(image_id)
        complete = (previous is not None and destination.is_file() and destination.stat().st_size > 0
                    and previous.source_revision == item["revision"])
        if complete:
            continue
        bytes_per_image = int(np.prod(item["shape"])) * 2
        output_bound = 56 * 1920 * 1920 * 2
        if not staged_ids:
            batch = []
            for candidate in scope[position:]:
                earlier = indexed.get(candidate["image_id"])
                target = OUTPUT / "images" / candidate["image_id"] / "input.tiff"
                if earlier is not None and target.is_file() and target.stat().st_size > 0 and earlier.source_revision == candidate["revision"]:
                    continue
                batch.append(candidate)
                if len(batch) == 4:
                    break
            staging_bytes = sum(int(np.prod(candidate["shape"])) * 2 for candidate in batch)
            if shutil.disk_usage(TEMP.parent).free < 64 * GIB + staging_bytes + output_bound:
                commit_batch()
                progress("waiting_io", next_image=image_id, reason="Local 64 GiB reserve plus one four-image staging batch unavailable")
                return
            TEMP.mkdir(parents=True, exist_ok=True)
            progress("batch_staging", next_image=image_id, batch_images=len(batch))
            try:
                with formal_read_lease(SOURCE):
                    for candidate in batch:
                        stage_source(candidate, TEMP / f"{candidate['image_id']}.native.dat")
            except BaseException:
                for candidate in batch:
                    (TEMP / f"{candidate['image_id']}.native.dat").unlink(missing_ok=True)
                raise
            staged_ids = {candidate["image_id"] for candidate in batch}
        if shutil.disk_usage(ROOT).free < 100 * GIB + output_bound:
            commit_batch()
            for staged_id in staged_ids:
                (TEMP / f"{staged_id}.native.dat").unlink(missing_ok=True)
            progress("waiting_io", next_image=image_id, reason="NAS 100 GiB reserve plus one output unavailable")
            return
        staged_source = TEMP / f"{image_id}.native.dat"
        staged_tiff = TEMP / f"{image_id}.input.tiff"
        array = None
        reader = None
        try:
            progress("image_exporting", next_image=image_id)
            array = np.memmap(staged_source, dtype="uint16", mode="r", shape=tuple(item["shape"]))
            local_spec = SourceSpec(image_id, PANEL_ID, "zarr", staged_source,
                                    tuple(range(56)), pixel_size_um=(PIXEL_SIZE_UM, PIXEL_SIZE_UM))
            reader = LocalReader(array, local_spec)
            geometry = _export_input(reader, panel, staged_tiff, tile_size=512)
            if not geometry["admitted"]:
                progress("image_excluded_no_marker_signal", next_image=image_id)
                continue
            with tifffile.TiffFile(staged_tiff) as tif:
                assert tif.series[0].shape == (56, 1920, 1920)
                assert tif.series[0].dtype == np.dtype("uint16")
            publish_file(staged_tiff, destination, directory=destination.parent, lock_name=".input.lock")
            row = image_row(item, by_image.loc[image_id], geometry)
            pending.append(row)
            if len(pending) == 4:
                commit_batch()
        except WaitingIO as error:
            commit_batch()
            for staged_id in staged_ids:
                (TEMP / f"{staged_id}.native.dat").unlink(missing_ok=True)
            progress("waiting_io", next_image=image_id, reason=str(error))
            return
        finally:
            del reader, array
            staged_source.unlink(missing_ok=True)
            staged_tiff.unlink(missing_ok=True)
            staged_ids.discard(image_id)
    commit_batch()
    TEMP.mkdir(parents=True, exist_ok=True)
    local_review = TEMP / "review_results.csv"
    review_table().to_csv(local_review, index=False, encoding="utf-8-sig")
    publish_file(local_review, OUTPUT / "review_results.csv", directory=OUTPUT)
    progress("prepared_pending_independent_review", next_image=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--repair-m02", action="store_true")
    arguments = parser.parse_args()
    if arguments.run:
        run()
    elif arguments.repair_m02:
        repair_m02()
    else:
        parser.print_help()
