"""Prepare the ten admitted native lung TMA acquisitions from Zenodo 13947395.

Inputs: original nested ZIP, accepted native TXT proofs, Appendix S6, release
workbook, NAS-only original-cohort clinical transcription, and study S1 rows.
Outputs: square CYX input.tiff,
39-field images.parquet, sample/panel/field tables and labelled preparer checks.
TXT window support remains study-local pending migration into the shared reader.
Run pixel work through the batch shared formal write lease wrapper; no normalization.
Use --repair-cohort-id for the metadata-only M02 repair requested by review 0.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "4"

import numpy as np
import openpyxl
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Barone2026_IMC_LungAdenocarcinoma"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = META / "preparation_evidence"
OUTPUT = NAS / "processed" / STUDY
RAW = NAS / "raw/public" / STUDY / "public_original/OT3 - Lung TMA block 1.zip"
NESTED_NAME = "OT3 - Lung TMA block 1/IM_20220623_OT3.zip"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PROGRESS = PROJECT / "cache/study_input" / STUDY / "jobs/0014_prepare_0/progress.json"
PANEL_ID = "OT3_native_TXT_45"
PREPARER = "0014_prepare_0"
COHORT_RULE = ("Leave blank; Batch=OT3 identifies the TMA block/acquisition batch, "
               "with no source definition of a patient-level cohort. "
               "Preserve Batch in image_metadata.slide_id.")
GAPS = [
    "原始TXT未声明物理像素尺度；说明中的2000 µm名义ROI不能代替实际采集像素尺度，保持空值。",
    "归档早期core说明的前4个肿瘤/正常标签与正式Appendix S6冲突；采用S6及一致的Zenodo采集表。",
    "独立复核待另一CLI上下文完成；整理者自检不生成D01通过绑定。",
]
CONTROLLER_REQUESTS = [
    "共享readers.py缺少原生IMC TXT流式读取：本study用readimc一致的float32解码、坐标索引与局部窗口临时适配；后续迁入src。",
    "请组织独立复核并写入D01验收绑定；重点核对S6与早期core说明的标签冲突处理、Bischoff2021 Fig1D临床转录、TXT几何及像素。",
]


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


@contextmanager
def nested_source():
    """Read the published nested ZIP from an operation-scoped SSD stage."""
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nested_", dir=scratch_root) as directory:
        nested = Path(directory) / Path(NESTED_NAME).name
        with formal_read_lease(RAW), zipfile.ZipFile(RAW) as outer:
            info = outer.getinfo(NESTED_NAME)
            if shutil.disk_usage(scratch_root).free < info.file_size + 64 * 1024**3:
                raise RuntimeError("Nested source would violate the local 64 GiB reserve")
            with outer.open(info) as source, nested.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024**2)
            if nested.stat().st_size != info.file_size:
                raise ValueError("Incomplete nested source stage")
        with zipfile.ZipFile(nested) as inner:
            yield inner


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress(phase: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, expected_images=10,
                             completed_images=completed, gaps=GAPS,
                             updated_at=datetime.now(timezone.utc).isoformat(), **details))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def field_row(field: str, source: str, rule: str, *, table: str = "image_metadata",
              file: Path | None = None, notes: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(file or META / "native_verification.json"),
                source_sheet_or_table="Appendix S6 / native TXT / study preparation",
                source_field=source, target_table=table, target_field=field,
                join_keys="Ablation_ID -> ROI_ID -> exact TXT member; Patient_ID + TumorType -> biological sample",
                value_mapping="", transformation_rule=rule, time_origin_rule="",
                conflict_rule="Published Appendix S6, corroborated by Zenodo workbook, overrides early draft core labels",
                notes=notes)


class TxtRecordReader:
    """Window disk-mapped float32 records using their measured integer XY positions.

    The numeric decode and coordinate placement follow readimc.TXTFile. Native
    records, including measured zeros, define validity; no tissue mask is inferred.
    """

    def __init__(self, records: np.memmap):
        xy = records[:, :2]
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy != np.floor(xy)):
            raise ValueError("Invalid native TXT pixel coordinates")
        width, height = xy.max(axis=0).astype(int) + 1
        self.index = np.full((height, width), -1, dtype=np.int64)
        # Bounded blocks avoid copying the complete multi-channel acquisition.
        for start in range(0, len(records), 65536):
            end = min(start + 65536, len(records))
            x, y = records[start:end, :2].astype(np.int64).T
            self.index[y, x] = np.arange(start, end)
        if np.count_nonzero(self.index >= 0) != len(records):
            raise ValueError("Duplicate source XY records need an explicit source decision")
        self.records = records
        self.shape = (45, int(height), int(width))
        self.dtype = np.dtype("float32")
        self.source = SimpleNamespace(pixel_size_um=None)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        index = self.index[y0:y1, x0:x1]
        valid = index >= 0
        result = np.zeros((len(channels), y1-y0, x1-x0), dtype=self.dtype)
        for output, channel in enumerate(channels):
            result[output, valid] = self.records[index[valid], channel + 2]
        return result

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.index[y0:y1, x0:x1] >= 0


def decode_txt(archive: zipfile.ZipFile, proof: dict, scratch: Path,
               completed: int) -> np.memmap:
    """Decode one exact accepted TXT member in bounded pandas chunks to local disk."""
    evidence = proof["native_read_evidence"]
    member = proof["native_metadata"]["member_name"]
    expected = evidence["rows_read"]
    scratch_bytes = expected * 47 * 4
    if shutil.disk_usage(scratch.parent).free < scratch_bytes + 64 * 1024**3:
        raise RuntimeError("Native TXT scratch would violate the local 64 GiB reserve")
    records = np.memmap(scratch, mode="w+", dtype=np.float32, shape=(expected, 47))
    with archive.open(member) as stream:
        header = stream.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
        if header != evidence["header_contract"]["header"]:
            raise ValueError(f"TXT header differs from accepted native proof: {member}")
        count = 0
        chunks = pd.read_csv(stream, sep="\t", names=header, usecols=[3, 4, *range(6, 51)],
                             dtype=np.float32, chunksize=32768, engine="c")
        for chunk in chunks:
            end = count + len(chunk)
            if end > expected:
                raise ValueError(f"More native TXT records than accepted proof: {member}")
            records[count:end] = chunk.to_numpy(dtype=np.float32)
            count = end
            if count % (32768 * 16) == 0:
                progress("decode_native_txt", completed, active_member=member,
                         decoded_records=count, expected_records=expected)
        if count != expected:
            raise ValueError(f"Native TXT record count differs from accepted proof: {member}")
    records.flush()
    return records


def check_header(path: Path, row: dict, panel: pd.DataFrame) -> None:
    """Reopen each TIFF and compare actual structure and OME channel identity."""
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (36, int(row["height_px"]), int(row["width_px"]))
        assert len(tif.pages) == 36 and all(p.dtype == np.dtype("float32") for p in tif.pages)
        root = ET.fromstring(tif.ome_metadata)
        names = [e.attrib["Name"] for e in root.iter() if e.tag.endswith("}Channel")]
        assert names == panel.loc[panel.keep_channel, "marker_id"].tolist()


def pixel_checks(reader: TxtRecordReader, path: Path, row: dict,
                 panel: pd.DataFrame) -> list[dict]:
    """Compare every retained channel in signal and boundary windows, plus padding."""
    _, h, w = reader.shape
    top, left = row["pad_top_px"], row["pad_left_px"]
    channels = panel.loc[panel.keep_channel, "source_channel_index"].tolist()
    # Choose a measured row with marker signal, rather than assuming the center is tissue.
    signal = None
    for start in range(0, len(reader.records), 65536):
        block = reader.records[start:start+65536]
        hit = np.flatnonzero(block[:, 6] != 0)  # native CD38, source index 4
        if len(hit):
            signal = tuple(int(value) for value in block[hit[0], :2])
            break
    if signal is None:
        raise ValueError("CD38 overview reference has no signal; choose another measured marker")
    x, y = signal
    x0, y0 = min(max(x-16, 0), max(w-32, 0)), min(max(y-16, 0), max(h-32, 0))
    windows = [(x0, y0, min(w, x0+32), min(h, y0+32)), (0, 0, min(w, 32), min(h, 32))]
    evidence = []
    with tifffile.TiffFile(path) as tif:
        for region in windows:
            x0, y0, x1, y1 = region
            target = (x0+left, y0+top, x1+left, y1+top)
            source = reader.read_window(channels, region)
            saved = np.stack([read_tiff_window(tif, i, target) for i in range(len(channels))])
            assert source.dtype == saved.dtype and np.array_equal(source, saved)
            evidence.append(dict(source_window=list(region), output_window=list(target),
                                 all_retained_channels=36, exact_equal=True))
        side = row["height_px"]
        padding = []
        if top: padding.append((0, 0, side, top))
        if left: padding.append((0, top, left, top+h))
        if row["pad_bottom_px"]: padding.append((0, top+h, side, side))
        if row["pad_right_px"]: padding.append((left+w, top, side, top+h))
        for region in padding:
            assert all(not np.any(read_tiff_window(tif, c, region)) for c in range(36))
    return evidence


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--extract-only", action="store_true")
parser.add_argument("--image-id", type=int, action="append", help="Process only specified source ROI IDs")
parser.add_argument("--overview-observation", help="Preparer observation supplied after viewing contact sheet")
parser.add_argument("--repair-cohort-id", action="store_true",
                    help="Clear unsupported patient cohort values and queue M02/D01 for independent recheck")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# Review 0 isolated a clinical-field semantic error; preserve its pixel acceptance evidence.
if args.repair_cohort_id:
    from cellweft.data.preparation.inputs import input_binding

    repair_job = PROGRESS.parent.parent / "0157_repair_1"
    PROGRESS = repair_job / "progress.json"
    repair_id = repair_job.name
    prior_job = repair_job.parent / "0154_review_0"
    GAPS = GAPS[:2] + ["M02 cohort_id已修补后需独立复查，D01需按新metadata版本重新绑定。"]
    progress("repair_source_check", 10)
    source_file = CACHE / "meta_IMlungpanel_CD3polygon.xlsx"
    source = pd.read_excel(source_file, sheet_name="Sheet1", dtype=str,
                           keep_default_na=False).iloc[:, :6]
    source = source[source.ROI_ID.isin([str(i) for i in range(1, 11)])]
    assert len(source) == 10 and source.Batch.eq("OT3").all()
    appendix = (CACHE / "appendix_MOESM1.txt").read_text()
    for row in source.itertuples(index=False):
        assert re.search(r"\s+".join(re.escape(str(value)) for value in row), appendix)
    expected_samples = set(source.Patient_ID + "_" + source.TumorType)
    verification = json.loads((META / "native_verification.json").read_text())
    stat = RAW.stat()
    assert {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} == {
        key: verification["native_result"]["source_file_stat"][key] for key in ("size", "mtime_ns")}

    with nullcontext():
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples_before = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        fields_before = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
        review_before = pd.read_csv(OUTPUT / "review_results.csv", dtype=str, keep_default_na=False)
        image_ids = sorted(images.image_id.tolist())
        assert len(image_ids) == 10 and set(samples_before.sample_id) == expected_samples
        assert len(samples_before) == 10 and samples_before.cohort_id.isin(["OT3", ""]).all()
        before_binding = input_binding(OUTPUT, NAS, image_ids)
        previous_d01 = review_before[(review_before.check_id == "D01") &
                                     (review_before.reviewer == "0154_review_0")].iloc[-1]
        accepted_binding = json.loads(previous_d01.details)["input_binding"]
        for key in ("image_ids", "images", "panel", "assets"):
            assert before_binding[key] == accepted_binding[key], key
        preserved_paths = [OUTPUT / "images.parquet", OUTPUT / "panel_mapping.csv"]
        preserved_revisions = [(path.stat().st_size, path.stat().st_mtime_ns) for path in preserved_paths]
        samples = samples_before.copy()
        samples["cohort_id"] = ""
        fields = fields_before.copy()
        affected_mapping = (fields.target_table == "sample_metadata") & (fields.target_field == "cohort_id")
        assert affected_mapping.sum() == 1
        fields.loc[affected_mapping, "transformation_rule"] = COHORT_RULE
        progress("repair_metadata_write", 10, affected_sample_rows=int(samples_before.cohort_id.ne("").sum()))
        for name, frame in [("sample_metadata", samples), ("field_mapping", fields)]:
            temporary = OUTPUT / f"{name}.csv.tmp"
            with formal_publish_lease(OUTPUT / f"{name}.csv", directory=OUTPUT):
                frame.to_csv(temporary, index=False, encoding="utf-8-sig")
                temporary.replace(OUTPUT / f"{name}.csv")

        # Only an independent reviewer may approve the new metadata revision.
        pending_binding = input_binding(OUTPUT, NAS, image_ids)
        write_json(repair_job / "pending_input_binding.json", pending_binding)
        additions = []
        for check_id, scope in [("M02", "sample"), ("D01", "study")]:
            details = dict(round=1, image_ids=image_ids, action="Clear all 10 cohort_id values; correct its field_mapping rule",
                           preparer_self_check=True, independent_review="pending",
                           note="OT3保留于image.slide_id；其余sample字段、图像索引、panel与TIFF复用。",
                           pending_input_binding=relative(repair_job / "pending_input_binding.json"))
            additions.append(dict(study=STUDY, preparer=repair_id, reviewer="", scope=scope,
                                  record_id=STUDY, check_id=check_id, result="unresolved",
                                  source_evidence="; ".join(map(relative, [source_file, CACHE / "appendix_MOESM1.txt", prior_job / "report.json"])),
                                  output_evidence="; ".join(relative(OUTPUT / f"{name}.csv") for name in ["sample_metadata", "field_mapping"]),
                                  details=json.dumps(details, ensure_ascii=False)))
        review = pd.concat([review_before, pd.DataFrame(additions, columns=review_before.columns)], ignore_index=True)
        temporary = OUTPUT / "review_results.csv.tmp"
        with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
            review.to_csv(temporary, index=False, encoding="utf-8-sig")
            temporary.replace(OUTPUT / "review_results.csv")

        # Verify the actual saved repair and preservation of all unaffected inputs.
        saved_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        saved_fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
        saved_review = pd.read_csv(OUTPUT / "review_results.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(saved_samples, samples)
        pd.testing.assert_frame_equal(saved_samples.drop(columns="cohort_id"), samples_before.drop(columns="cohort_id"))
        pd.testing.assert_frame_equal(saved_fields, fields)
        pd.testing.assert_frame_equal(saved_review.iloc[:len(review_before)], review_before)
        assert [(path.stat().st_size, path.stat().st_mtime_ns) for path in preserved_paths] == preserved_revisions
        for key in ("image_ids", "images", "panel", "assets"):
            assert pending_binding[key] == before_binding[key], key
        receipt = dict(study=STUDY, repair_id=repair_id, expected_images=10, exported_images=len(images),
                       reused_images=len(images), sample_rows=len(saved_samples),
                       sample_linked_images=int(images.sample_id.isin(saved_samples.sample_id).sum()),
                       patient_linked_samples=int(saved_samples.patient_key.ne("").sum()),
                       distinct_patients=int(saved_samples.patient_key.nunique()),
                       cohort_values_cleared=int(samples_before.cohort_id.ne("").sum()),
                       cohort_values_nonempty=int(saved_samples.cohort_id.ne("").sum()),
                       preserved_review_rows=len(review_before), independent_review="pending",
                       unchanged_tiff_assets=before_binding["assets"],
                       changed_sample_fields=["cohort_id"], changed_mapping_rows=1,
                       source_evidence=[relative(source_file), relative(CACHE / "appendix_MOESM1.txt"), relative(prior_job / "metadata_review.json")],
                       cohort_rule=COHORT_RULE, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(repair_job / "repair_verification.json", receipt)
    progress("prepared_for_independent_recheck", len(images), sample_rows=len(samples),
             sample_linked_images=receipt["sample_linked_images"], patient_linked_samples=receipt["patient_linked_samples"],
             repaired_fields=["sample_metadata.cohort_id"], independent_review="pending")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    raise SystemExit(0)

scratch_root = Path("/tmp/study_input_batch") / STUDY
scratch_root.mkdir(parents=True, exist_ok=True)

# Reuse the accepted source binding and physical-core scope, without rehashing.
verification = json.loads((META / "native_verification.json").read_text())
proofs = [row["native_proof"] for row in verification["records"]]
source_stat = verification["native_result"]["source_file_stat"]
assert RAW.stat().st_size == source_stat["size"]
assert RAW.stat().st_mtime_ns == source_stat["mtime_ns"]
assert len(proofs) == 10 and all(p["complete"] and p["native_order_verified"] for p in proofs)
names = proofs[0]["channels"]
assert all(p["channels"] == names for p in proofs)
panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
assert len(panel) == 45 and int(panel.keep_channel.sum()) == 36
assert panel.loc[panel.nuclear, "source_name"].tolist() == ["DNA1", "DNA2"]

# The official metadata and Appendix S6 independently state the specimen/subject map.
with zipfile.ZipFile(RAW) as outer:
    for member in outer.namelist():
        if not member.endswith(".docx"): continue
        target = CACHE / Path(member).name
        if not target.exists(): target.write_bytes(outer.read(member))
        with zipfile.ZipFile(target) as doc:
            root = ET.fromstring(doc.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            text = "\n".join("".join(t.text or "" for t in p.findall(".//w:t", ns))
                             for p in root.findall(".//w:p", ns))
            (CACHE / (target.stem + ".txt")).write_text(text, encoding="utf-8")
workbook = openpyxl.load_workbook(CACHE / "meta_IMlungpanel_CD3polygon.xlsx", read_only=True, data_only=True)
values = list(workbook["Sheet1"].values)
source_meta = pd.DataFrame(values[1:], columns=values[0]).iloc[:,:6]
workbook.close()
source_meta = source_meta[source_meta.ROI_ID.isin(range(1, 11))].copy()
appendix = (CACHE / "appendix_MOESM1.txt").read_text()
for source in source_meta.itertuples(index=False):
    pattern = rf"{re.escape(source.Ablation_ID)}\s+{source.ROI_ID}\s+{source.Tissue}\s+{source.Batch}\s+{source.Patient_ID}\s+{source.TumorType}"
    assert re.search(pattern, appendix), f"S6/workbook mismatch: ROI {source.ROI_ID}"
assert len(source_meta) == 10 and source_meta.Patient_ID.nunique() == 5

samples = []
for source in source_meta.itertuples(index=False):
    sample = dict.fromkeys(columns("sample_metadata"), "")
    patient, condition = source.Patient_ID, source.TumorType
    paired_condition = "Healthy" if condition == "Tumor" else "Tumor"
    sample.update(study=STUDY, sample_id=f"{patient}_{condition}", patient_key=f"{STUDY}::{patient}",
                  source_subject_id=patient, center_id="Charite",
                  sample_type="tissue", tissue_type="tumor" if condition == "Tumor" else "adjacent_normal",
                  anatomical_site=source.Tissue, lesion_type="primary" if condition == "Tumor" else "non_malignant",
                  tumor_normal_pair_id=f"{STUDY}::{patient}", matched_sample_id=f"{patient}_{paired_condition}",
                  collection_method="resection", preservation_method="other", processing_method="paraffin-embedded TMA",
                  primary_diagnosis="lung adenocarcinoma", primary_site="lung",
                  histology="adenocarcinoma", metadata_notes="S6和Zenodo表确认患者/取样类型；sample_id按Patient_ID+TumorType构造，非ROI替代。各取样本批1个core。Methods确认原发手术和石蜡组织；preservation_method=other表示石蜡包埋，当前IMC段未独立声明固定剂。独立标本编号、确切时点、完整治疗史及随访未提供，保持空值；人口统计见原队列来源补充。")
    samples.append(sample)
samples = pd.DataFrame(samples, columns=columns("sample_metadata"))
clinical_file = META / "source_clinical_figure1d.csv"
clinical = pd.read_csv(clinical_file, dtype=str, keep_default_na=False).set_index("Patient_ID", verify_integrity=True)
assert set(source_meta.Patient_ID) == set(clinical.index)
for index, sample in samples.iterrows():
    source = clinical.loc[sample.source_subject_id]
    samples.loc[index, "age_at_collection_years"] = int(source.Age)
    samples.loc[index, "sex_reported"] = {"F":"female", "M":"male"}[source.Sex]
    samples.loc[index, "primary_site"] = "lung; " + source.Topography
    samples.loc[index, "histology"] = source.Histology + " adenocarcinoma"
    samples.loc[index, "tnm_t"] = source.Tumor_pT
    samples.loc[index, "tnm_n"] = source.Tumor_pN
    samples.loc[index, "stage_basis"] = "pathological"
    samples.loc[index, "treatment_status_at_collection"] = "previously untreated"
    samples.loc[index, "metadata_notes"] += " Bischoff2021 Fig1D按同一Patient_ID补充手术队列年龄/性别/肿瘤部位/组织学/pT/pN；包括邻近正常sample的字段均描述供者原发肿瘤，非正常组织病理。Methods为既往未治疗原发手术；分期版次、M及stage group未推算。"

# Record all base fields, blank-value semantics, and source-specific conflict rules.
image_rules = {
    "study": ("canonical dataset", "Fixed canonical study identifier"),
    "image_id": ("ROI_ID", "IM_20220623_OT3_ROI_{ROI_ID:03d}"),
    "sample_id": ("Patient_ID; TumorType", "Patient_ID + '_' + TumorType; one distinct biological tissue per patient/condition"),
    "panel_id": ("native ordered 45 labels", "One verified order shared by ten TXT acquisitions"),
    "image_kind": ("Methods; Appendix S6", "tma_core"),
    "source_image_id": ("nested ZIP exact member", "NAS-relative outer ZIP::inner ZIP member::TXT member"),
    "source_files": ("original ZIP", "List containing unchanged original outer ZIP"),
    "source_format": ("TXT", "txt"),
    "source_acquisition_id": ("Ablation_ID", "Exact published ablation identifier"),
    "source_axes": ("X;Y;measurement columns", "XY table; window placement follows readimc.TXTFile"),
    "source_dtype": ("native TXT reader", "float32, identical numeric decode to readimc.TXTFile"),
    "source_revision": ("source_version", "10.5281/zenodo.13947395"),
    "slide_id": ("Batch", "OT3"), "core_id": ("ROI_ID", "OT3_core_{ROI_ID:02d}"),
    "roi_id": ("ROI_ID", "Exact string ROI identifier"),
    "region_annotation": ("Tissue; TumorType", "Retain original categorical strings as 'Tissue; TumorType'"),
    "source_x0_px": ("X", "0; full TXT acquisition coordinate frame"),
    "source_y0_px": ("Y", "0; full TXT acquisition coordinate frame"),
    "source_x1_px": ("X", "max(X)+1, half-open"),
    "source_y1_px": ("Y", "max(Y)+1, half-open"),
    "intensity_description": ("native TXT measurement columns", "Unscaled float32 native values; no spillover correction or asinh transform"),
    "split_method": ("one TXT per independent acquisition", "source_acquisition"),
    "split_evidence": ("Appendix S6; release workbook", "Exact Ablation_ID/ROI_ID mapping, one core per selected acquisition"),
    "metadata_notes": ("source conflict and missing optional metadata", "Document in each record; unknown scale is blank"),
}
for field in ["pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "height_px", "width_px", "channel_count", "source_channel_count"]:
    image_rules[field] = ("native geometry; export_input", "Full acquisition with centered square zero padding; kept channels in source order")
for field in ["input_image_path", "valid_pixels_path"]:
    image_rules[field] = ("actual output asset", "NAS-relative path; validity is emitted only for genuinely absent acquisition coordinates")
fields = [field_row(field, *image_rules.get(field, ("not supplied / no layer selection", "Leave blank; no inferred value")))
          for field in columns("image_metadata")]
for field in columns("sample_metadata"):
    populated = samples[field].astype(str).ne("").any()
    identity_fields = {"study":"canonical study identifier", "sample_id":"Patient_ID; TumorType",
                       "patient_key":"Patient_ID", "source_subject_id":"Patient_ID", "cohort_id":"Batch",
                       "tissue_type":"TumorType", "anatomical_site":"Tissue", "lesion_type":"TumorType",
                       "sample_type":"Tissue", "tumor_normal_pair_id":"Patient_ID", "matched_sample_id":"Patient_ID; TumorType"}
    method_fields = {"collection_method", "preservation_method", "processing_method", "primary_diagnosis",
                     "primary_site", "histology", "center_id"}
    source_file = CACHE / ("paper_text.txt" if field in method_fields else "meta_IMlungpanel_CD3polygon.xlsx")
    source_field = "Methods: lung tissue samples / IMC TMA preparation" if field in method_fields else identity_fields.get(field,"not supplied")
    rule = "Derive only from official S6, agreeing Zenodo metadata and Methods" if populated else "Leave blank; no patient-level clinical value inferred"
    if field == "cohort_id": rule = COHORT_RULE
    if field == "preservation_method": rule = "other = paraffin embedded; fixation reagent not independently stated for the IMC block"
    clinical_fields = {"age_at_collection_years":"Age", "sex_reported":"Sex", "primary_site":"Topography",
                       "histology":"Histology", "tnm_t":"Tumor_pT", "tnm_n":"Tumor_pN", "stage_basis":"Tumor_pT; Tumor_pN"}
    if field in clinical_fields:
        source_file, source_field = clinical_file, clinical_fields[field]
        rule = "Exact Patient_ID join to original cohort Fig1D; donor tumor characteristics also apply to paired adjacent-normal sample metadata"
        if field == "age_at_collection_years": rule = "Source surgical-cohort Age in years; no age-at-diagnosis inference"
        if field == "sex_reported": rule = "F -> female; M -> male"
        if field == "stage_basis": rule = "Source pT/pN prefix denotes pathological assessment; edition/M/stage group left blank"
    if field == "treatment_status_at_collection":
        source_file, source_field = CACHE / "bischoff2021_paper.txt", "Methods: Collection of tissue specimens"
        rule = "Retain source statement 'previously untreated' for donor tissue collected at primary surgery"
    fields.append(field_row(field, source_field, rule, table="sample_metadata", file=source_file,
                            notes="S6与发布表一致，覆盖10 image、10 sample、5 patient。source_sample_id因无独立标本编号留空。"))
fields.append(field_row("source_channel_index", "header columns[6:]", "0..44; S1 joins exact study+source_name; output indices separately assigned 0..35",
                        table="panel_mapping", file=S1))
field_mapping = pd.DataFrame(fields, columns=columns("field_mapping"))
images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
completed = len(images)
if args.overview_observation:
    write_json(CACHE / "overview_observation.json", dict(preparer=PREPARER, observation=args.overview_observation))

scope = dict(study=STUDY, source_archive=relative(RAW), source_revision="10.5281/zenodo.13947395",
             native_verification_reused=relative(META / "native_verification.json"), hashes_recomputed=False,
             expected_images=10, sample_rows=10, sample_linked_images=10, patient_linked_samples=10, distinct_patients=5,
             selected_core_ids=list(range(1,11)), excluded=[dict(roi_id=11, reason="Tonsil positive control; outside accepted lung scope")],
             duplicate_representations="MCD and per-acquisition TXT represent the same physical cores; accepted native TXT is used once",
             source_defects=verification["source_defects"], gaps=GAPS, controller_requests=CONTROLLER_REQUESTS,
             metadata_evidence=[relative(CACHE / "appendix_MOESM1.txt"), relative(CACHE / "meta_IMlungpanel_CD3polygon.xlsx"),
                                relative(CACHE / "Lung TMA description.txt"), relative(CACHE / "Slide_Description_IM_OT3_OT4_LungTMA.txt"),
                                relative(CACHE / "paper_text.txt"), relative(CACHE / "zenodo_record.json"),
                                relative(clinical_file), relative(META / "source_clinical_figure1d_evidence.json")],
             public_sources=["https://zenodo.org/records/13947395", "https://link.springer.com/article/10.1038/s44319-026-00732-5",
                             "https://github.com/SawitzkiGroup/KLRF1-Tpex"],
             source_followup=dict(author_repository="SawitzkiGroup/KLRF1-Tpex API returned HTTP 404; recorded in author_repository_check.json",
                                  original_cohort="Bischoff2021, DOI 10.1038/s41388-021-02054-3, explicitly cited by current IMC paper",
                                  original_cohort_table="Supplementary Table 1 is endothelial cell counts; original Figure 1D provides five patients' age/sex/topography/histology/pT/pN, manually transcribed to study metadata",
                                  original_cohort_repository="bischofp/single_cell_lung_adenocarcinoma main tree inspected; code and ligand/receptor tables, no clinical/sample map"),
             capacity=dict(nested_zip_bytes=2906143727, largest_txt_record_scratch_bytes=6024528*47*4,
                           planned_native_output_upper_estimate_bytes=10*36*3200*3200*4,
                           nas_reserve_bytes=100*1024**3, local_reserve_bytes=64*1024**3))
write_json(CACHE / "source_scope.json", scope)
save_input_tables(OUTPUT, pd.DataFrame(images, columns=columns("image_metadata")), samples, panel, field_mapping)
progress("metadata_ready", completed, sample_rows=10, sample_linked_images=completed, patient_linked_samples=10)

if not args.metadata_only:
    if shutil.disk_usage(OUTPUT).free < scope["capacity"]["planned_native_output_upper_estimate_bytes"] + 100 * 1024**3:
        raise RuntimeError("Native TIFF outputs would violate the NAS 100 GiB reserve")
    with nested_source() as inner:
        write_json(CACHE / "archive_members.json", [dict(name=i.filename, size=i.file_size, compressed_size=i.compress_size) for i in inner.infolist()])
        if not args.extract_only:
            selected = set(args.image_id or range(1,11))
            checks = json.loads((CACHE / "preparer_pixel_checks.json").read_text()) if (CACHE / "preparer_pixel_checks.json").exists() else {}
            for source in source_meta.sort_values("ROI_ID").itertuples(index=False):
                roi = int(source.ROI_ID)
                if roi not in selected: continue
                image_id = f"IM_20220623_OT3_ROI_{roi:03d}"
                destination = OUTPUT / "images" / image_id / "input.tiff"
                old = next((r for r in images if r["image_id"] == image_id), None)
                if old is not None and destination.exists():
                    check_header(destination, old, panel)
                    print(json.dumps(dict(image_id=image_id, reused=True)), flush=True)
                    continue
                proof = next(p for p in proofs if p["native_metadata"]["member_name"] == f"IM_20220623_OT3_OT3_ROI_{roi}_{roi}.txt")
                member = proof["native_metadata"]["member_name"]
                assert inner.getinfo(member).file_size == proof["inner_archive_entry"]["uncompressed_bytes"]
                scratch = scratch_root / f"{image_id}.records.float32"
                progress("decode_native_txt", len(images), active_image=image_id)
                records = decode_txt(inner, proof, scratch, len(images))
                try:
                    reader = TxtRecordReader(records)
                    _, h, w = reader.shape
                    output_bytes = 36 * max(h,w)**2 * 4
                    if shutil.disk_usage(OUTPUT).free < output_bytes + 100*1024**3:
                        raise RuntimeError(f"{image_id}: {output_bytes} uncompressed TIFF bytes would violate NAS reserve")
                    progress("export_native_tiff", len(images), active_image=image_id, source_shape=list(reader.shape),
                             estimated_uncompressed_tiff_bytes=output_bytes)
                    receipt = CACHE / "export_receipts" / f"{image_id}.json"
                    # A completed TIFF may precede an interrupted metadata checkpoint.
                    # Recheck its pixels below and publish the missing row without rewriting it.
                    if destination.exists() and receipt.exists():
                        result = json.loads(receipt.read_text())
                    else:
                        result = export_input(reader, panel, destination, tile_size=512)
                        result["estimated_uncompressed_tiff_bytes"] = output_bytes
                        write_json(receipt, result)
                    if not result["admitted"]:
                        raise ValueError(f"Selected source acquisition has all admitted markers zero: {image_id}")
                    row = dict.fromkeys(columns("image_metadata"), None)
                    row.update(study=STUDY, image_id=image_id, sample_id=f"{source.Patient_ID}_{source.TumorType}",
                               panel_id=PANEL_ID, image_kind="tma_core",
                               source_image_id=f"{relative(RAW)}::{NESTED_NAME}::{member}", source_files=[relative(RAW)],
                               source_format="txt", source_acquisition_id=source.Ablation_ID,
                               source_axes="XY table; 45 measurement columns", source_dtype="float32",
                               source_revision="10.5281/zenodo.13947395", slide_id=source.Batch,
                               core_id=f"OT3_core_{roi:02d}", roi_id=str(roi),
                               region_annotation=f"{source.Tissue}; {source.TumorType}",
                               source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                               input_image_path=relative(destination),
                               valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
                               intensity_description="Native IMC TXT measurement values decoded as float32, as in readimc.TXTFile; no normalization, spillover correction or asinh transform",
                               split_method="source_acquisition", split_evidence="Appendix S6 + Zenodo workbook exact Ablation_ID/ROI_ID; complete independent TXT acquisition",
                               metadata_notes="Full native acquisition frame; physical pixel scale unconfirmed. Published S6 and agreeing release workbook override early draft core n/t labels. Source MCD proof has historical I/O failure; accepted native TXT used without duplicate credit.")
                    for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]:
                        row[key] = result[key]
                    check_header(destination, row, panel)
                    checks[image_id] = dict(pixels=pixel_checks(reader, destination, row, panel),
                                            measured_pixels=int(np.count_nonzero(reader.index >= 0)),
                                            source_shape=list(reader.shape), channel_nonzero_pixels=result["channel_nonzero_pixels"])
                    write_json(CACHE / "preparer_pixel_checks.json", checks)
                    # Make the actual image index durable after each completed TIFF.
                    images = [r for r in images if r["image_id"] != image_id] + [row]
                    save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel, field_mapping)
                    progress("image_complete", len(images), latest_image=image_id, sample_linked_images=len(images), patient_linked_samples=10)
                    print(json.dumps(dict(image_id=image_id, exported=True, completed_images=len(images), source_shape=list(reader.shape))), flush=True)
                finally:
                    records._mmap.close()
                    scratch.unlink(missing_ok=True)

# Generate a display-only contact sheet from three raw channels at bounded resolution.
if images and not args.extract_only:
    sheet = Image.new("RGB", (5*360, 2*390), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(sorted(images, key=lambda r:r["image_id"])):
        path = NAS / row["input_image_path"]
        channels = []
        with tifffile.TiffFile(path) as tif:
            for source_name in ["DNA1", "CK18", "CD45"]:
                output_index = int(panel.loc[panel.source_name == source_name, "output_channel_index"].iloc[0])
                values = tif.pages[output_index].asarray(maxworkers=2)[::4, ::4]
                positive = values[values > 0]
                scale = float(np.percentile(positive, 99)) if len(positive) else 1.0
                channels.append(np.uint8(np.clip(values / scale, 0, 1)*255))
        rgb = Image.fromarray(np.stack(channels, axis=-1), "RGB")
        rgb.thumbnail((350,350))
        x, y = (i%5)*360, (i//5)*390
        sheet.paste(rgb, (x, y+35))
        draw.text((x+5,y+4), row["image_id"], fill="black")
        draw.text((x+5,y+19), row["region_annotation"] + " | RGB=DNA1/CK18/CD45", fill="black")
    sheet.save(CACHE / "native_overviews.png")

# Preparer checks are explicitly labelled; independent acceptance is outstanding.
review_rows = []
for check in pd.read_csv(TEMPLATES / "review_checks.csv", keep_default_na=False).itertuples(index=False):
    result = "pass" if len(images) == 10 else "unresolved"
    details = "Preparer self-check only; independent reviewer pending."
    if check.check_id == "M01":
        result = "pass"
        details += " Appendix S6 and Zenodo workbook agree: 10 samples, 10/10 image-sample links, 10/10 sample-patient links, 5 distinct patients; key=Patient_ID+TumorType, not ROI."
    elif check.check_id == "M02":
        result = "pass"
        details += " Identity/tissue/pair fields follow S6; resection/paraffin and donor diagnosis follow Methods. Age/sex/topography/histology/pT/pN join original cohort Fig1D by exact Patient_ID; source image and NAS-only transcription preserved. Original cohort Methods confirms previously untreated. Other clinical/outcome values left blank. Independent transcription review pending."
    elif check.check_id == "I04":
        observation = CACHE / "overview_observation.json"
        result = "pass" if observation.exists() and len(images)==10 else "unresolved"
        details += " " + (observation.read_text() if observation.exists() else "Source identity is mapped; contact sheet visual assessment pending.")
    elif check.check_id == "I05":
        details += " Every delivered image: two <=32x32 native/output windows, all 36 retained channels, float32 exact equality; all padding strips checked zero. Coordinates: cache preparer_pixel_checks.json."
    elif check.check_id == "D01":
        result = "unresolved"
        details += f" Exported {len(images)}/10; all metadata saved. Independent review and D01 input_binding pending."
    elif check.check_id == "S01":
        details += " Ten accepted lung cores, tonsil ROI 11 excluded; MCD/TXT alternate representations counted once. Native proof reused; no full rehash."
    elif check.check_id in {"P01", "P02", "P03"}:
        details += " 45 source channels -> 36 outputs: 34 downstream markers plus DNA1/DNA2; exact study-specific S1 labels; all delivered headers reopened."
    review_rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                            record_id=PANEL_ID if check.scope=="panel" else STUDY,
                            check_id=check.check_id, result=result,
                            source_evidence=relative(META / "native_verification.json") + "; " + relative(CACHE / "appendix_MOESM1.txt") + "; " + relative(CACHE / "meta_IMlungpanel_CD3polygon.xlsx"),
                            output_evidence=relative(OUTPUT) + "; " + relative(CACHE / "preparer_pixel_checks.json"), details=details))
review = pd.DataFrame(review_rows, columns=columns("review_results"))
review_path = OUTPUT / "review_results.csv"
if review_path.exists():
    prior_review = pd.read_csv(review_path, keep_default_na=False)
    review = pd.concat([prior_review[prior_review.preparer != PREPARER], review], ignore_index=True)
with formal_publish_lease(review_path, directory=review_path.parent):
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
write_json(CACHE / "preparation_summary.json", dict(study=STUDY, expected_images=10, exported_images=len(images),
           sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=10,
           distinct_patients=5, channels_source=45, channels_output=36, gaps=GAPS,
           controller_requests=CONTROLLER_REQUESTS, independent_review="pending"))
progress("prepared" if len(images)==10 else "partial", len(images), sample_rows=10,
         sample_linked_images=len(images), patient_linked_samples=10, controller_requests=CONTROLLER_REQUESTS)
