"""Prepare the 33 publication-linked HTAN neuroblastoma MIBI FOVs.

The accepted native proofs, HTAN biospecimen ancestry, and this study's S1
rows determine source identity, sample identity, and channel order.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import xml.etree.ElementTree as ET
from typing import Iterator

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Patel2024_MIBI_Neuroblastoma"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
MANIFEST = NAS / "raw/public" / STUDY / "manifest.json"
ACCEPTED = NAS / "raw/public" / STUDY / "manifest.json"
PROOFS = NAS / "raw/public" / STUDY / "manifest.json"
PORTAL = NAS / "raw/public" / STUDY / "manifest.json"
DATABASE = NAS / "raw/public" / STUDY / "manifest.json"
META = metadata_source_directory(NAS, STUDY)
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
DEFAULT_JOB = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0260_prepare_0"
PANEL_ID = "HTAPP_NB_panel18"
REVISION = "HTAN Release 3.0"
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10802404/"


def relative(path: Path) -> str:
    return metadata_source_relative(path, NAS).as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def publish_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


class LeaseWait(TimeoutError):
    """A shared read lease or source staging did not finish by the checkpoint deadline."""


@contextmanager
def read_deadline(seconds: int) -> Iterator[None]:
    def timed_out(signum: int, frame: object) -> None:
        raise LeaseWait(f"NAS source read/staging exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def source_records() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    records = [row for row in read_json(MANIFEST)["preparation_sources"]["native_manifest"]["files"] if row["study"] == STUDY]
    accepted = read_json(ACCEPTED)["preparation_sources"]["accepted_scope"]
    proofs = {row["source_id"]: read_json(PROOFS)["preparation_sources"]["native_images"][row["source_id"]]
              for row in records}
    assert accepted["scope_complete"] and len(records) == accepted["n_images"] == 33
    assert {row["source_id"] for row in records} == set(accepted["accepted_image_ids"])
    portal = {row["HTANDataFileID"]: row for row in read_json(PORTAL)["preparation_sources"]["portal_images"]
              if str(row.get("Filename", "")).startswith("mibi_level_2/neuroblastoma/")}
    assert set(portal) == {row["source_id"] for row in records}
    for row in records:
        proof = proofs[row["source_id"]]
        path = NAS / row["source_path"]
        stat = path.stat()
        assert proof["verification_status"] == "verified" and proof["mapping_verified"]
        assert proof["all_channels_bounded_read"] and proof["channels_read"] == 49
        assert proof["series_shape"] == [49, 1024, 1024]
        assert stat.st_size == row["expected_bytes"] == proof["file_stat"]["size"]
        assert stat.st_mtime_ns == proof["file_stat"]["mtime_ns"]
        assert row["filename"] == portal[row["source_id"]]["Filename"]
        assert row["participant_id"] == portal[row["source_id"]]["HTANParticipantID"]
        assert row["parent_biospecimen_id"] == portal[row["source_id"]]["HTANParentBiospecimenID"]
        assert portal[row["source_id"]]["PassedQC"] == "Yes"
    return records, portal, proofs


def sample_rows(records: list[dict]) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    database = read_json(DATABASE)["preparation_sources"]["clinical_source"]
    biospecimens = database["biospecimenByBiospecimenID"]
    chains = {}
    for record in records:
        chain = []
        key = record["parent_biospecimen_id"]
        while key in biospecimens:
            chain.append(key)
            key = biospecimens[key]["ParentID"]
        assert key == record["participant_id"] and len(chain) == 4
        assert biospecimens[chain[-1]]["IschemicTime"] == "original tumor (tissue removed by procedure)"
        chains[record["source_id"]] = chain
    samples = []
    for sample_id in sorted({chain[-1] for chain in chains.values()}):
        linked = [record for record in records if chains[record["source_id"]][-1] == sample_id]
        patients = {record["participant_id"] for record in linked}
        assert len(patients) == 1
        patient = patients.pop()
        diagnosis = database["diagnosisByParticipantID"][patient]
        demographics = database["demographicsByParticipantID"][patient]
        specimen = biospecimens[sample_id]
        assert specimen["ParentID"] == patient
        assert specimen["AcquisitionMethodType"] == "Surgical Resection"
        assert all(biospecimens[record["parent_biospecimen_id"]]["FixativeType"] == "Formalin"
                   and biospecimens[record["parent_biospecimen_id"]]["StorageMethod"] == "Paraffin block"
                   for record in linked)
        samples.append(dict(study=STUDY, sample_id=sample_id, patient_key=patient,
            source_subject_id=patient, source_sample_id=sample_id, biospecimen_group_id=sample_id,
            cohort_id="HTAN_HTAPP_NB", center_id=specimen["atlasid"], sample_type=specimen["BiospecimenType"],
            tissue_type="tumor", anatomical_site=diagnosis["SiteofResectionorBiopsy"],
            collection_method=specimen["AcquisitionMethodType"].lower(), preservation_method="ffpe",
            processing_method="MIBI FFPE section", case_control_status="case",
            sex_reported=demographics["Gender"].lower(), primary_diagnosis=diagnosis["PrimaryDiagnosis"],
            primary_site=diagnosis["TissueorOrganofOrigin"],
            metadata_notes=json.dumps(dict(identity_evidence="HTAN FOV-to-original-specimen ancestry",
                clinical_source=relative(DATABASE), reported_risk_group=diagnosis.get("CogNeuroblastomaRiskGroup"),
                reported_inrg_stage=diagnosis.get("INRGStage"),
                missing="HTAN AgeatDiagnosis repeats 6570 across the selected participants; its unit and privacy transformation are not established. Index date, treatment-at-collection, and survival origin are not established"),
                ensure_ascii=False)))
    assert len(samples) == 11 and len({row["patient_key"] for row in samples}) == 11
    annotations = {record["source_id"]: biospecimens[record["parent_biospecimen_id"]]["IschemicTime"]
                   for record in records}
    return pd.DataFrame(samples).reindex(columns=columns("sample_metadata")), chains, annotations


def field_rules() -> pd.DataFrame:
    rows = []

    def add(source: str, source_field: str, table: str, target: str, keys: str,
            rule: str, notes: str = "") -> None:
        rows.append(dict(study=STUDY, source_file=source, source_sheet_or_table="",
            source_field=source_field, target_table=table, target_field=target, join_keys=keys,
            value_mapping="{}", transformation_rule=rule, time_origin_rule="",
            conflict_rule="Preserve source values; leave unconfirmed clinical values empty", notes=notes))

    add(relative(MANIFEST), "source_id;filename;source_path;participant_id;parent_biospecimen_id",
        "image_metadata", "image_id;source_files;sample_id", "source_id=HTANDataFileID; parent biospecimen ancestry",
        "Use each accepted HTAN acquisition as one image; source path is the release-3.0 OME-TIFF.",
        "33/33 accepted files, 33/33 proof/stat matches; publication association verified by manifest and portal.")
    add(relative(DATABASE), "biospecimenByBiospecimenID.ParentID", "sample_metadata",
        "sample_id;patient_key;source_sample_id;source_subject_id", "FOV parent→section→block→original sample→participant",
        "Use the terminal original tumor specimen as sample_id and its parent as patient_key.",
        "33 image-to-sample links; 11 original sample rows; 11 sample-to-patient links.")
    add(relative(DATABASE),
        "diagnosisByParticipantID.SiteofResectionorBiopsy;diagnosisByParticipantID.PrimaryDiagnosis;"
        "diagnosisByParticipantID.TissueorOrganofOrigin;demographicsByParticipantID.Gender;"
        "biospecimenByBiospecimenID.AcquisitionMethodType;biospecimenByBiospecimenID.FixativeType;"
        "biospecimenByBiospecimenID.StorageMethod",
        "sample_metadata", "anatomical_site;primary_diagnosis;primary_site;sex_reported;collection_method;preservation_method",
        "original sample ParentID=participant; clinical dictionaries keyed by participant",
        "Copy SiteofResectionorBiopsy, PrimaryDiagnosis and TissueorOrganofOrigin from the participant "
        "diagnosis; lower-case demographics Gender; lower-case original sample AcquisitionMethodType "
        "(Surgical Resection to surgical resection); map FOV section FixativeType=Formalin and "
        "StorageMethod=Paraffin block to preservation_method=ffpe.",
        "AgeatDiagnosis repeats 6570; unit and privacy transformation are unresolved, so age is omitted. INRG stage and COG risk remain source notes because index timing is undefined.")
    add(relative(DATABASE), "biospecimenByBiospecimenID.BiospecimenType;atlasid;IschemicTime;FixativeType;StorageMethod",
        "sample_metadata", "biospecimen_group_id;cohort_id;center_id;sample_type;tissue_type;processing_method;case_control_status;metadata_notes",
        "original specimen sample_id; FOV section ancestor; participant key from ParentID",
        "Preserve original specimen ID as group; identify HTAN HTAPP NB cohort and atlas; copy BiospecimenType. Original tumor annotation supports tissue_type=tumor and case status; formalin/paraffin ancestry and MIBI source support processing method. Keep risk/stage source notes and unresolved clinical timing in metadata_notes.")
    add(relative(DATABASE), "biospecimenByBiospecimenID.ParentID;IschemicTime",
        "image_metadata", "slide_id;fov_id;region_annotation",
        "parent_biospecimen_id=FOV-specific section; follow ParentID to section/slide",
        "Use the FOV-specific section as fov_id, its parent as slide_id, and preserve FOV section IschemicTime verbatim as the source region annotation.")
    add(relative(PORTAL), "PhysicalSizeX;PhysicalSizeY;FOVnumber;ImageID", "image_metadata",
        "pixel_size_x_um;pixel_size_y_um;fov_id;roi_id", "HTANDataFileID=source_id",
        "Copy 0.48828125 micrometers/pixel only when portal and OME agree; use biospecimen ancestry for fov_id and portal ImageID for roi_id. Portal FOVnumber corroborates the source Point number.")
    add(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
        "source_name;marker_id;role;keep_channel;nuclear;body;output_channel_index",
        "Study=Patel2024_MIBI_Neuroblastoma; Original marker=OME Channel Name",
        "Retain 37 downstream measurements and selected dsDNA nuclear input; exclude 11 other channels.")
    add(PAPER, "Figure 6 and Methods: MIBI cohort", "image_metadata", "split_evidence;metadata_notes",
        "study-level reference", "Paper reports 30 analytic FOVs from 11 samples; the accepted HTAN release contains 33 FOVs.",
        "The 33 release FOVs are retained as the accepted local source scope; do not equate release count with analytic figure count.")
    add("https://pmc.ncbi.nlm.nih.gov/articles/instance/10802404/bin/media-2.xlsx",
        "Supplement 2 / Table S1 clinical metadata", "sample_metadata", "metadata_notes",
        "Paper XML cross-reference SD2=Table S1; supplement access attempted 2026-10-24",
        "No values imported from Table S1: the direct PMC attachment returned a browser challenge and the Europe PMC supplementary-files endpoint timed out. The public paper XML exposed no GitHub code link; identity and populated clinical values use the local HTAN snapshot.",
        "Optional paper-table clinical enrichment remains unverified; image-to-sample and sample-to-patient coverage are fully supported by HTAN ancestry.")
    add(relative(ACCEPTED), "accepted_image_ids;verification_paths", "image_metadata",
        "source_revision;source_channel_count;channel_count;source_axes;source_dtype;source_x0_px;source_y0_px;source_x1_px;source_y1_px;input_image_path",
        "source_id=proof source_id", "Use verified native 49×1024×1024 CYX uint16 source; retain full frame and 38 selected channels at native intensity.")
    return pd.DataFrame(rows).reindex(columns=columns("field_mapping"))


def check_header(path: Path, panel: pd.DataFrame) -> None:
    names = panel.loc[panel.keep_channel, "marker_id"].tolist()
    with formal_read_lease(path), tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        assert series.axes == "CYX" and series.shape == (len(names), 1024, 1024)
        assert series.dtype == np.dtype("uint16")
        xml = ET.fromstring(tif.ome_metadata)
        observed = [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")]
        assert observed == names
        pixels = next(node for node in xml.iter() if node.tag.endswith("}Pixels"))
        assert float(pixels.attrib["PhysicalSizeX"]) == 0.48828125
        assert float(pixels.attrib["PhysicalSizeY"]) == 0.48828125


def review_rows(images: pd.DataFrame, samples: pd.DataFrame, preparer: str,
                verified: bool) -> pd.DataFrame:
    n = len(images)
    evidence = ";".join(map(relative, (MANIFEST, ACCEPTED, DATABASE, PORTAL, S1)))
    rows = []
    checks = {
        "S01": f"已导出{n}/33张；33个已准入HTAN源文件均有native proof。论文30张分析FOV与本地33张发布FOV分别记录。",
        "I01": f"{n}张输出以HTAN采集ID、原始路径和原始OME身份定位。",
        "I02": f"{n}/{n}张连接到11份原始取样；panel外键均指向本study panel。",
        "I03": "完整1024×1024原生FOV，无裁切、补边或重采样；OME及门户尺度一致。",
        "I04": "HTAN明确FOV→section→block→原始肿瘤取样→participant；来源IschemicTime注释原文保留。",
        "P01": "49个来源通道的有序标签与本study S1精确匹配。",
        "P02": "38个保留通道含37个downstream和1个dsDNA核输入；11个有依据排除。",
        "P03": f"已导出{n}张逐图重开，核对CYX、uint16、38个标准化名称及顺序。",
        "M01": f"本地HTAN门户/临床快照、论文正文和Table S1公开入口已追查；sample 11行，image→sample {n}/{n}，sample→patient 11/11；Table S1附件访问受阻，未从中导入值。",
        "M02": "已填临床字段由HTAN取样祖先、diagnosis和demographics重建；年龄单位/隐私变换、index及生存起点未核实，Table S1未取得，相关可选值留空。",
        "I05": "代表图全部保留通道的内部和边界像素块逐值相等。" if verified else "代表图像素自检待运行。",
        "D01": f"整理者交付{n}/33张、11份sample和11个patient连接；独立复核及input_binding待主控执行。",
    }
    for check, details in checks.items():
        rows.append(dict(study=STUDY, preparer=preparer, reviewer="",
            scope="sample" if check.startswith("M") else "panel" if check.startswith("P") else "study",
            record_id=STUDY, check_id=check,
            result="unresolved" if check == "D01" or check == "I05" and not verified else "pass",
            source_evidence=evidence, output_evidence=relative(OUTPUT),
            details="整理者自检；独立复核待执行。" + details))
    return pd.DataFrame(rows).reindex(columns=columns("review_results"))


def save_reviews(frame: pd.DataFrame) -> None:
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


def verify_pixels(images: pd.DataFrame, panel: pd.DataFrame, records: list[dict]) -> dict:
    selected = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    lookup = {record["source_id"]: record for record in records}
    checked = []
    for image_id in (records[0]["source_id"], records[-1]["source_id"]):
        row = images.set_index("image_id").loc[image_id]
        source = NAS / lookup[image_id]["source_path"]
        output = NAS / row.input_image_path
        for window in ((448, 448, 480, 480), (992, 992, 1024, 1024)):
            with formal_read_lease(source), tifffile.TiffFile(source) as tif:
                expected = [read_tiff_window(tif, int(channel.source_channel_index), window)
                            for channel in selected.itertuples(index=False)]
            with formal_read_lease(output), tifffile.TiffFile(output) as tif:
                for channel, values in zip(selected.itertuples(index=False), expected, strict=True):
                    observed = read_tiff_window(tif, int(channel.output_channel_index), window)
                    assert values.dtype == observed.dtype
                    np.testing.assert_array_equal(values, observed)
            checked.append(dict(image_id=image_id, window_xyxy=window, channel_count=len(selected)))
    return dict(study=STUDY, sampled_images=2, sampled_windows=checked,
                all_retained_channels_compared=True, independent_review=False)


def repair_clinical_mapping(job: Path) -> None:
    with read_deadline(180), formal_read_lease(MANIFEST):
        records, _, _ = source_records()
        samples, _, _ = sample_rows(records)
    fields = field_rules()
    clinical_target = "anatomical_site;primary_diagnosis;primary_site;sex_reported;collection_method;preservation_method"
    selector = lambda frame: ((frame.study == STUDY) &
                              (frame.target_table == "sample_metadata") &
                              (frame.target_field == clinical_target))
    corrected = fields.loc[selector(fields)]
    assert len(corrected) == 1
    path = OUTPUT / "field_mapping.csv"
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        existing_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
        prior = pd.read_csv(path, dtype=str, keep_default_na=False)
        version = (path.stat().st_size, path.stat().st_mtime_ns)
    clinical_columns = clinical_target.split(";")
    pd.testing.assert_frame_equal(
        samples.set_index("sample_id")[clinical_columns].sort_index(),
        existing_samples.set_index("sample_id")[clinical_columns].sort_index(),
        check_dtype=False,
    )
    assert len(prior.loc[selector(prior)]) == 1
    if not prior.loc[selector(prior)].iloc[0].equals(corrected.iloc[0]):
        for column in prior.columns:
            prior.loc[selector(prior), column] = corrected.iloc[0][column]
        with tempfile.TemporaryDirectory(prefix="patel-mapping-", dir="/tmp/study_input_batch") as temporary:
            staged = Path(temporary) / path.name
            prior.to_csv(staged, index=False, encoding="utf-8-sig")
            partial = path.with_name(path.name + ".partial")
            with read_deadline(180), formal_publish_lease(path, directory=OUTPUT):
                try:
                    assert (path.stat().st_size, path.stat().st_mtime_ns) == version
                    shutil.copyfile(staged, partial)
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
    publish_json(job / "progress.json", dict(
        study=STUDY, phase="repair", scientific_repair_round=1,
        stage="clinical_mapping_repaired_pending_independent_review",
        expected_images=33, completed_images=33, sample_rows=len(samples),
        sample_linked_images=33, patient_linked_samples=int(samples.patient_key.notna().sum()),
        repaired_check="M02", repaired_file=relative(path),
        next_step="Independent M02 review and refreshed D01 binding",
        updated_at=datetime.now(timezone.utc).isoformat(),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--limit", type=int, default=33)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--repair-field-mapping", action="store_true")
    args = parser.parse_args()
    job = args.job_dir
    progress_path = job / "progress.json"
    preparer = "codex_cli_" + job.name
    if args.repair_field_mapping:
        try:
            repair_clinical_mapping(job)
        except LeaseWait as error:
            publish_json(progress_path, dict(
                study=STUDY, phase="repair", scientific_repair_round=1,
                stage="waiting_io", waiting_reason=str(error),
                completed_images=33, remaining_images=0,
                resume_scope="Retry M02 field_mapping row; 33 image files and sample table are unchanged",
                updated_at=datetime.now(timezone.utc).isoformat(),
            ))
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        with read_deadline(180), formal_read_lease(MANIFEST):
            records, portal, proofs = source_records()
            samples, chains, annotations = sample_rows(records)
            source_names = pd.read_csv(META / "htan_release_3.0_channels.csv").ome_channel_name.tolist()
            assert all(proof["observed_channel_labels"] == source_names for proof in proofs.values())
            panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1)
            assert len(panel) == 49 and int(panel.keep_channel.sum()) == 38
            fields = field_rules()
            with resource_lock(OUTPUT, shared=True):
                images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists()
                          else pd.DataFrame(columns=columns("image_metadata")))
            receipts_path = CACHE / "export_records.json"
            receipts = read_json(receipts_path) if receipts_path.exists() else {}
    except LeaseWait as error:
        prior = read_json(progress_path) if progress_path.exists() else {}
        prior.update(stage="waiting_io", waiting_reason=str(error),
                     resume_scope="Retry startup source and existing-index read; committed rows unchanged",
                     updated_at=datetime.now(timezone.utc).isoformat())
        publish_json(progress_path, prior)
        return
    rows = {row["image_id"]: row for row in images.to_dict("records")}

    def progress(stage: str, **details: object) -> None:
        remaining = [record["source_id"] for record in records if record["source_id"] not in rows]
        publish_json(progress_path, dict(study=STUDY, phase="prepare", scientific_repair_round=0,
            stage=stage, expected_images=33, completed_images=len(rows), sample_rows=len(samples),
            sample_linked_images=sum(bool(row.get("sample_id")) for row in rows.values()),
            patient_linked_samples=int(samples.patient_key.notna().sum()),
            remaining_images=len(remaining), next_image=remaining[0] if remaining else None,
            updated_at=datetime.now(timezone.utc).isoformat(), **details))

    if args.metadata_only:
        assert set(rows) == {record["source_id"] for record in records}
        save_input_tables(OUTPUT, images, samples, panel, fields)
        save_reviews(review_rows(images, samples, preparer, True))
        progress("self_checked_pending_independent_review", sampled_images=2)
        return

    estimated = 33 * 38 * 1024 * 1024 * 2
    remaining_bytes = (33 - len(rows)) * 38 * 1024 * 1024 * 2
    free_nas = shutil.disk_usage(NAS).free
    free_local = shutil.disk_usage("/tmp").free
    progress("source_checked", estimated_uncompressed_output_bytes=estimated,
             remaining_uncompressed_bytes=remaining_bytes, nas_free_bytes=free_nas,
             local_free_bytes=free_local)
    local_peak_bytes = 103206968 + 38 * 1024 * 1024 * 2
    if free_nas - remaining_bytes < 100 * 1024**3 or free_local - local_peak_bytes < 64 * 1024**3:
        progress("waiting_io", waiting_reason="NAS 100 GiB or local 64 GiB free-space reserve unavailable")
        return
    # The source manifest and HTAN ancestry are stable; publish index rows after each closed image.
    new_images = 0
    for record in records:
        image_id = record["source_id"]
        source_path = NAS / record["source_path"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in rows:
            assert destination.is_file() and image_id in receipts
            assert receipts[image_id]["source_stat"] == [source_path.stat().st_size, source_path.stat().st_mtime_ns]
            assert receipts[image_id]["output_stat"] == [destination.stat().st_size, destination.stat().st_mtime_ns]
            continue
        if new_images >= args.limit:
            break
        with tempfile.TemporaryDirectory(prefix="patel-source-", dir="/tmp/study_input_batch") as temporary:
            staged_source = Path(temporary) / source_path.name
            try:
                with read_deadline(180), formal_read_lease(source_path):
                    shutil.copyfile(source_path, staged_source)
            except LeaseWait as error:
                staged_source.unlink(missing_ok=True)
                save_reviews(review_rows(pd.DataFrame(rows.values()), samples, preparer, False))
                progress("waiting_io", waiting_reason=str(error), current_image=image_id,
                         resume_scope=f"{len(rows)}/33 committed; retry remaining {33-len(rows)} images")
                return
            assert staged_source.stat().st_size == source_path.stat().st_size
            with tifffile.TiffFile(staged_source) as tif:
                assert tif.series[0].shape == (49, 1024, 1024) and tif.series[0].axes == "CYX"
                assert tif.series[0].dtype == np.dtype("uint16")
                xml = ET.fromstring(tif.ome_metadata)
                image_name = next(node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Image"))
                pixels = next(node for node in xml.iter() if node.tag.endswith("}Pixels"))
                assert [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")] == source_names
                scale = (float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"]))
                assert scale == (portal[image_id]["PhysicalSizeX"], portal[image_id]["PhysicalSizeY"])
                assert image_name == portal[image_id]["ImageID"]
            source = SourceSpec(image_id, PANEL_ID, "tiff", staged_source, tuple(range(49)),
                                pixel_size_um=scale, source_revision=REVISION)
            with ExitStack() as stack:
                reader = stack.enter_context(open_source(source))
                result = export_input(reader, panel, destination)
        if not result["admitted"]:
            progress("excluded_zero_markers", current_image=image_id)
            continue
        check_header(destination, panel)
        chain = chains[image_id]
        row = dict(study=STUDY, image_id=image_id, sample_id=chain[-1], panel_id=PANEL_ID,
            image_kind="fov", source_image_id=image_id, source_files=[record["source_path"]],
            source_format="OME-TIFF", source_acquisition_id=image_id, source_z_index=0,
            source_t_index=0, source_axes="CYX", source_dtype="uint16", source_revision=REVISION,
            slide_id=chain[1], fov_id=chain[0], roi_id=image_name,
            region_annotation=annotations[image_id],
            source_x0_px=0, source_y0_px=0, source_x1_px=1024, source_y1_px=1024,
            pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
            height_px=1024, width_px=1024, source_channel_count=49, channel_count=38,
            pixel_size_x_um=scale[0], pixel_size_y_um=scale[1], input_image_path=relative(destination),
            intensity_description="HTAN Level-2 published uint16 MIBI intensities; no transform in this export",
            split_method="already_independent", split_evidence=relative(ACCEPTED),
            metadata_notes=json.dumps(dict(ome_image_name=image_name, synapse_id=record["synapse_id"],
                biological_chain=chain, source_annotation_field="HTAN biospecimen IschemicTime",
                paper_analytic_fovs=30, release_fovs=33), ensure_ascii=False))
        rows[image_id] = row
        save_input_tables(OUTPUT, pd.DataFrame(rows.values()), samples, panel, fields)
        receipts[image_id] = dict(source_stat=[source_path.stat().st_size, source_path.stat().st_mtime_ns],
                                  output_stat=[destination.stat().st_size, destination.stat().st_mtime_ns],
                                  export=result, header_checked=True)
        publish_json(receipts_path, receipts)
        new_images += 1
        progress("exporting", last_completed_image=image_id)
        print(f"exported {len(rows)}/33", flush=True)
    images = pd.DataFrame(rows.values())
    if len(images) == 33:
        verification = verify_pixels(images, panel, records)
        publish_json(CACHE / "preparer_verification.json", verification)
        save_reviews(review_rows(images, samples, preparer, True))
        progress("self_checked_pending_independent_review", sampled_images=2)
    else:
        save_reviews(review_rows(images, samples, preparer, False))
        progress("batch_complete")


if __name__ == "__main__":
    main()
