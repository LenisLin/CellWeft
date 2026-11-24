"""Prepare Jackson 2020 native SPoRA IMC inputs and source-supported sample links.

Inputs: complete 1.0-mpp Zarrs, original author clinical/core tables, and study S1.
Outputs: square native float32 input TIFFs and the five per-study input tables.
Run metadata retrieval first; --export uses the controller's shared formal write lease.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import datetime as dt
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import requests

STUDY = "Jackson2020_IMC_BreastCancer"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
RAW = ROOT / "raw/public" / STUDY / "spora"
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0294_prepare_0")))
PANEL = "jackson_spora_imc_39"
CACHE.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: object) -> None:
    """Replace a small resumable study record after its contents are complete."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RemoteZip(io.RawIOBase):
    """Read only ZIP directory and requested metadata members via verified ranges."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self.ranges: list[dict] = []

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, length: int = -1) -> bytes:
        end = self.size - 1 if length < 0 else min(self.size, self.position + length) - 1
        if end < self.position:
            return b""
        start = self.position
        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                              params={"cellweft_range": f"{start}-{end}"}, stream=True, timeout=60) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Metadata archive server did not honor the bounded byte range")
            value = response.content
        if len(value) != end - start + 1:
            raise ValueError("Incomplete archive metadata range")
        self.ranges.append(dict(start=start, end=end, bytes=len(value)))
        self.position = end + 1
        return value

    def close(self) -> None:
        self.session.close()
        super().close()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fetch-metadata", action="store_true")
parser.add_argument("--export", action="store_true")
parser.add_argument("--overview", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()

if args.fetch_metadata:
    for name in ("Basel_PatientMetadata.csv", "Zuri_PatientMetadata.csv", "Basel_Zuri_StainingPanel.csv"):
        assert (META / "source_metadata" / name).is_file(), f"Missing received source: {name}"
    print(json.dumps(dict(metadata_sources=3, source="received raw metadata bundle")))
    raise SystemExit(0)

import shutil
import xml.etree.ElementTree as ET

import numpy as np
import numcodecs.blosc as blosc
import pandas as pd
import tifffile
import zarr

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

blosc.set_nthreads(1)
zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
OUTPUT.mkdir(parents=True, exist_ok=True)
prior_receipts_path = CACHE / "export_receipts.json"
prior_receipts = json.loads(prior_receipts_path.read_text()) if prior_receipts_path.exists() else {}
save_json(JOB / "progress.json", dict(study=STUDY, stage="checking_sources",
    processed_sources=len(prior_receipts), exported_images=sum(item["admitted"] for item in prior_receipts.values()),
    expected_images=723, resume_after="source_manifest_validation",
    updated_at=dt.datetime.now(dt.timezone.utc).isoformat()))


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE) or path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, ROOT))


def progress(stage: str, gaps: list[str] | None = None, *,
             verified_headers: int | None = None, resume_after: str | None = None) -> None:
    """Record the last committed image batch and its identity coverage."""
    state = dict(study=STUDY, stage=stage,
        completed_images=committed_images, expected_images=len(tissues),
        processed_sources=committed_sources, all_zero_exclusions=committed_exclusions,
        sample_rows=len(samples), sample_linked_images=committed_images,
        patient_linked_samples=int(samples.patient_key.ne("").sum()),
        distinct_patients=samples.patient_key.nunique(), gaps=gaps or ["独立review待执行"],
        updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
    if verified_headers is not None:
        state["verified_headers"] = verified_headers
    if resume_after is not None:
        state["resume_after"] = resume_after
    save_json(JOB / "progress.json", state)


def one_value(frame: pd.DataFrame, column: str) -> str:
    """Retain one source value per specimen, treating author [] as missing."""
    values = frame[column].replace({"[]": "", "NA": ""}).dropna()
    values = values[values.ne("")].unique()
    if len(values) > 1:
        raise ValueError(f"Conflicting source {column} within sample {frame.sample_id.iloc[0]}")
    return str(values[0]) if len(values) else ""


# Reuse complete native verification and compare the current objects to its source manifest.
native_path = META / "native_verification.json"
manifest_path = META / "imc_source_objects.json"
native = json.loads(native_path.read_text())
assert native["verified_count"] == native["source_image_count"] == 723 and not native["failed"]
proofs = {item["image_id"]: item for item in native["verified"]}
accepted_path = META / "source_metadata/accepted_inventory.json"
accepted = json.loads(accepted_path.read_text())
assert Path(accepted["native_verification"]).name == "verification.json"
assert Path(accepted["source_manifest"]).name == "imc_source_objects.json"
inventory = accepted["images"]
objects = {item["image_id"]: item["source_objects"] for item in inventory}
manifest_objects = {
    relative(RAW / item["key"].removeprefix("spora-submissions/jackson2020single/")): item["size"]
    for item in json.loads(manifest_path.read_text())
}
assert len(manifest_objects) == sum(map(len, objects.values())) == 2170
assert {entry["path"]: entry["size"] for group in objects.values() for entry in group} == manifest_objects
ordered_path = metadata_source_directory(ROOT, STUDY) / 'spora_inventory' / 'imc_ordered_marker_names.json'
ordered = json.loads(ordered_path.read_text())
tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
channels = pd.read_parquet(RAW / "imc/channels.parquet")
availability = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
assert set(tissues.tissue_id) == set(proofs) == set(objects) == set(ordered)
assert {item["image_id"] for item in inventory} == set(proofs)
source_names = channels.channel_name.tolist()
assert availability.columns.tolist() == source_names and availability.to_numpy().all()
panel = panel_from_s1(STUDY, PANEL, source_names,
    ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv")
panel["source_original_channel_name"] = channels.original_channel_name.tolist()
kept = panel[panel.keep_channel].sort_values("output_channel_index")
for row in tissues.itertuples(index=False):
    item = next(item for item in inventory if item["image_id"] == row.tissue_id)
    assert item["shape"] == proofs[row.tissue_id]["shape"] == [39, row.image_height, row.image_width]
    assert ordered[row.tissue_id] == source_names == proofs[row.tissue_id]["canonical_channel_labels"]
inventory_by_id = {item["image_id"]: item for item in inventory}
output_estimate = sum(len(kept) * max(item["shape"][1:])**2 * 4 for item in inventory)
save_json(JOB / "progress.json", dict(study=STUDY, stage="linking_metadata",
    processed_sources=len(prior_receipts), exported_images=sum(item["admitted"] for item in prior_receipts.values()),
    expected_images=len(tissues), resume_after="metadata_linkage",
    updated_at=dt.datetime.now(dt.timezone.utc).isoformat()))

# Exact author filename/core/PID joins distinguish normal tissue from the sampled tumour.
author_frames = []
for cohort, filename in [("Basel", "Basel_PatientMetadata.csv"), ("Zurich", "Zuri_PatientMetadata.csv")]:
    original = pd.read_csv(META / "source_metadata" / filename, dtype=str, keep_default_na=False)
    original["author_cohort"] = cohort
    original["author_table"] = relative(META / "source_metadata" / filename)
    author_frames.append(original)
authors = pd.concat(author_frames, ignore_index=True).fillna("")
linked = tissues.merge(authors, left_on="original_file_name", right_on="FileName_FullStack", validate="one_to_one", suffixes=("_spora", ""))
assert len(linked) == len(tissues)
assert linked.image_name.eq(linked.core).all()
assert pd.to_numeric(linked.Height_FullStack).eq(linked.image_height).all()
assert pd.to_numeric(linked.Width_FullStack).eq(linked.image_width).all()
assert linked.original_patient_identifier.eq(linked.author_cohort + "_" + linked.PID).all()
assert linked.groupby("original_patient_identifier").patient_id.nunique().eq(1).all()
assert linked.groupby("patient_id").original_patient_identifier.nunique().eq(1).all()
linked["biological_tissue"] = np.where(linked.patient_cohort.eq("Basel"),
    linked.diseasestatus.map({"tumor": "tumor", "non-tumor": "normal"}),
    linked.location.map({"CENTER": "tumor", "PERIPHERY": "tumor", "[]": "normal"}))
assert linked.biological_tissue.notna().all()
linked["sample_id"] = linked.original_patient_identifier + "_" + linked.biological_tissue
linked["source_region"] = np.where(linked.patient_cohort.eq("Basel"), linked.diseasestatus, linked.location)
linked["histology_from_type"] = linked.tumor_type.replace({"INVASIV_DUCTAL": "INVASIVE_DUCTAL"})
sample_rows = []
extra_fields = ["source_age_years", "source_tnm_t", "source_tnm_n", "source_tnm_m", "source_os_months", "source_dfs_months", "source_collection_year"]
source_columns = dict(source_age_years="age", source_tnm_t="PTNM_T", source_tnm_n="PTNM_N", source_tnm_m="PTNM_M",
                      source_os_months="OSmonth", source_dfs_months="DFSmonth", source_collection_year="Yearofsamplecollection")
for sample_id, group in linked.groupby("sample_id", sort=True):
    first = group.iloc[0]
    row = dict.fromkeys(pd.read_csv(TEMPLATES / "sample_metadata.csv").columns, "")
    row.update(study=STUDY, sample_id=sample_id, patient_key=first.patient_id,
        source_subject_id=first.original_patient_identifier, cohort_id=first.patient_cohort,
        center_id=first.patient_cohort, sample_type="tissue", tissue_type=first.biological_tissue,
        anatomical_site="breast", lesion_type="primary" if first.biological_tissue == "tumor" else "non_malignant",
        preservation_method="ffpe", processing_method="tissue microarray", sex_reported="female",
        primary_diagnosis="breast cancer", primary_site="breast",
        histology=one_value(group, "histology") or one_value(group, "histology_from_type"),
        grade=one_value(group, "grade"), molecular_subtype=one_value(group, "clinical_type"),
        vital_status={"alive": "alive", "alive w metastases": "alive", "death": "deceased", "death by primary disease": "deceased"}.get(one_value(group, "Patientstatus"), ""),
        metadata_notes="Derived specimen group: same patient and tumor/normal tissue; multiple cores from the same sampled tumor share a sample. Author tables and manuscript Clinical data provide identity. Age time origin, TNM edition/index, OS/DFS origins/events unspecified: retained in source_* fields; standardized age, TNM and endpoint groups left blank. Source []/NA are missing. No independent donor-block identifier supplied.")
    row.update({target: one_value(group, source) for target, source in source_columns.items()})
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=list(pd.read_csv(TEMPLATES / "sample_metadata.csv").columns) + extra_fields)
assert len(samples) == 502 and samples.patient_key.nunique() == 357
save_json(CACHE / "sample_linkage.json", dict(sample_rows=len(samples), patient_count=samples.patient_key.nunique(),
    source_core_rows=len(authors), local_images=len(linked), image_to_sample=linked[["tissue_id", "sample_id", "patient_id", "original_patient_identifier", "original_file_name", "core", "source_region", "author_table"]].to_dict("records"),
    author_rows_outside_local_scope=authors.loc[~authors.FileName_FullStack.isin(linked.original_file_name), ["author_cohort", "core", "location", "FileName_FullStack"]].to_dict("records"),
    biological_group_evidence="Manuscript Clinical data: several cores sampled from each Zurich tumor and matched normal samples in Basel. Author ZurichTMA_pipeline.Rmd lines 467-469: location=[] denotes NORMAL. Normal and tumor are separate specimens; SPoRA specimen_id enumerates core views and is not the biological sample key."))

# Define every filled field and preserve uncertain clinical time semantics explicitly.
fields = pd.read_csv(TEMPLATES / "field_mapping.csv", dtype=str)
field_rows = []


def mapping(source: str, target_table: str, target: str, rule: str, *, source_file: str = "", notes: str = "", values: dict | None = None) -> None:
    record = dict.fromkeys(fields.columns, "")
    record.update(study=STUDY, source_file=source_file or relative(RAW / "metadata/tissues.parquet"),
        source_sheet_or_table="tissues / original author patient-core metadata", source_field=source,
        target_table=target_table, target_field=target, join_keys="tissues.original_file_name = author.FileName_FullStack; image_name = core; original_patient_identifier = author_cohort + '_' + PID",
        value_mapping=json.dumps(values or {}, ensure_ascii=False), transformation_rule=rule,
        time_origin_rule="Unspecified clinical time origins remain blank in standardized fields.",
        conflict_rule="Multiple nonmissing values within a specimen raise an error; source [] and NA become blank.", notes=notes)
    field_rows.append(record)


mapping("patient_id;original_patient_identifier;PID;diseasestatus;location", "sample_metadata", "study;sample_id;patient_key;source_subject_id;cohort_id;center_id;source_sample_id;biospecimen_group_id",
    "Exact filename joins for all 723 images. 502 biological specimen groups = 352 tumors + 150 normal samples; 357 explicit patient keys. Group same patient tumor cores across CENTER/PERIPHERY; normal separate. source_sample_id/biospecimen_group_id blank: no separate original donor-block/sample identifier. Same local cohort name defines center. Author 12 METASTASIS cores outside local source scope are not added.",
    source_file=relative(CACHE / "sample_linkage.json"), notes="Sources searched: local tissues/panel/native manifests; original Zenodo 3518284 metadata CSVs; Nature Supplementary Tables 1/9; author README and both R pipelines. All 723 image→sample and all 502 sample→patient links confirmed; no identity gap.")
mapping("Clinical data;diseasestatus;location", "sample_metadata", "sample_type;tissue_type;anatomical_site;lesion_type;preservation_method;processing_method;primary_diagnosis;primary_site",
    "Breast cancer patient cohorts; tumor vs matched normal as explicitly labelled by authors. Zurich location=[] is NORMAL per author pipeline lines 467-469. Primary tumor cores group across center/periphery; normal non_malignant. FFPE and tissue microarray from manuscript Methods.", source_file=relative(META / "source_metadata/accepted_manuscript.pdf"))
for target, source in [("sex_reported", "gender"), ("histology", "histology;tumor_type"), ("grade", "grade"), ("molecular_subtype", "clinical_type"), ("vital_status", "Patientstatus")]:
    mapping(source, "sample_metadata", target,
        "Aggregate unique nonmissing value within sample. Female from FEMALE. Histology prefers original histology, otherwise author tumor_type code; INVASIV_DUCTAL→INVASIVE_DUCTAL corrects the same patient's obvious spelling variant across tumor cores. Molecular_subtype preserves clinical HR/HER2 category. Patientstatus alive/alive w metastases→alive; death/death by primary disease→deceased; CENSORED/[]→blank.",
        source_file=";".join(relative(META / "source_metadata" / name) for name in ["Basel_PatientMetadata.csv", "Zuri_PatientMetadata.csv"]))
for target, source in source_columns.items():
    definition = dict(source_age_years="Author-reported patient age in years; time origin unspecified", source_tnm_t="Original T category; staging edition and clinical index unspecified", source_tnm_n="Original N category; staging edition and clinical index unspecified", source_tnm_m="Original M category; staging edition and clinical index unspecified", source_os_months="Author-reported OS months; origin and event indicator not inferred", source_dfs_months="Author-reported DFS months; origin and event indicator not inferred", source_collection_year="Author-reported specimen collection year")[target]
    mapping(source, "sample_metadata", target, "Preserve unique original value; []/NA/blank→blank. No clinical index or event inferred.",
        source_file=";".join(relative(META / "source_metadata" / name) for name in ["Basel_PatientMetadata.csv", "Zuri_PatientMetadata.csv"]),
        notes=json.dumps(dict(data_type="string", definition=definition)))
mapping("clinical absent/undefined;age;PTNM_*;OSmonth;DFSmonth", "sample_metadata", "metadata_notes;age_at_collection_years;age_at_index_years;age_at_diagnosis_years;tnm_t;tnm_n;tnm_m;os_time;dfs_time;source_sample_id",
    "Blank standardized ages, staging and outcomes when time origin/edition/event is not specified; source versions retained as extensions. Other unavailable optional template fields blank. Record reason per sample.")
mapping("channel row order;channels_per_tissue;Zarr omero.channels", "panel_mapping", ";".join(panel.columns),
    "39 measured source channels exactly agree in channel table, per-tissue presence, native embedded label order and accepted native proof. Study-specific exact S1 join; six excluded channels omitted from TIFF; source/output indices distinct.", source_file=relative(ordered_path))
mapping("tissue_id;original_file_name;image_name;patient_id;TMA_block_label;diseasestatus;location", "image_metadata", "study;image_id;source_image_id;source_files;sample_id;panel_id;image_kind;slide_id;core_id;roi_id;region_annotation;split_method;split_evidence;metadata_notes",
    "Use canonical directory tissue_id, exact author core and original filename join. Native embedded group display-name defects ignored as identity keys. Full independent core acquisition, no region crop or sample splitting. Basel slide=TMA block; Zurich slide=ZTMA208 from author core prefix. core_id=author core, ROI=author core; region annotation preserves author diseasestatus or location including [].")
mapping("native Zarr shape/dtype/scale;accepted proof;source manifest", "image_metadata", "source_format;source_axes;source_dtype;source_revision;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;source_channel_count;channel_count;pixel_size_x_um;pixel_size_y_um;input_image_path;valid_pixels_path;intensity_description",
    "Reuse 723 accepted source proofs and verify 2170 manifest object sizes; no full hashing. CYX float32, scale [1,1,1], complete acquisition bounding rectangle; center-pad to square with zeros. All source pixels, including measured zeros/background, retained. Published Zarr float32 values preserved; prior processing not independently established. No missing acquisition domain declared.", source_file=relative(META / "source_metadata/accepted_inventory.json"))
fields = pd.DataFrame(field_rows, columns=fields.columns)

if (OUTPUT / "images.parquet").exists():
    with formal_read_lease(OUTPUT / "images.parquet"):
        images = pd.read_parquet(OUTPUT / "images.parquet")
else:
    images = pd.read_csv(TEMPLATES / "image_metadata.csv")
images = images.set_index("image_id", drop=False)
receipts_path = CACHE / "export_receipts.json"
receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
committed_images = len(images)
committed_sources = len(receipts)
committed_exclusions = sum(not item["admitted"] for item in receipts.values())


def save_tables() -> None:
    save_input_tables(OUTPUT, images.reset_index(drop=True), samples, panel, fields)


def checkpoint(stage: str) -> None:
    """Commit a small completed image batch before reporting its coverage."""
    global committed_images, committed_sources, committed_exclusions
    save_tables()
    save_json(receipts_path, receipts)
    committed_images = len(images)
    committed_sources = len(receipts)
    committed_exclusions = sum(not item["admitted"] for item in receipts.values())
    progress(stage)


def write_reviews(verified: dict | None = None) -> None:
    """Save preparer evidence; independent D01 acceptance remains unresolved."""
    template = pd.read_csv(TEMPLATES / "review_results.csv")
    rows = []
    items = [
        ("S01", "study", "pass" if len(receipts) == len(tissues) else "unresolved", f"整理者自检：723个完整Zarr，2170个正式对象；既有native proof复用，当前size核对；整图marker准入完成{len(receipts)}/723，输出{len(images)}。6个排除通道；未改动.partial。论文分析范围与发布范围分开，12个仅作者表转移core未加入。准入与逐图信号由实际export_receipts记录；概览见representative_overview.jpg。"),
        ("P01", "panel", "pass", "整理者自检：39个来源通道顺序与逐图标签、channels、channels_per_tissue、有序证明及本study S1精确对应。"),
        ("P02", "panel", "pass", "整理者自检：33个输出通道=31个marker+2个DNA；所有已选核/膜输入保留；无重复downstream marker。"),
        ("M01", "sample", "pass", "整理者来源追查：723/723图像经原文件名与core精确连接作者原表；502个生物学sample全部连接357个患者。352肿瘤sample+150正常sample；同患者同肿瘤多core共享sample。已查本地metadata、原Zenodo两份patient/core CSV、作者流程、论文Methods与补充表1/9。来源单独donor-block标识缺失已说明，未用SPoRA逐core specimen_id充当sample。"),
        ("M02", "sample", "pass", "整理者自检：全部非空目标值从作者表按sample重建；来源年龄、TNM、OS/DFS保留source_*字段，临床起点/分期版本/事件语义未明时标准字段留空。CENSORED不映射alive。[]按missing处理（Zurich区域注释[]原值保留；其正常组织含义有作者代码证据）。"),
        ("D01", "study", "unresolved", f"整理者交付{len(images)}张TIFF；身份和panel记录已保存。本表为整理者自检，尚待其他CLI独立复核，不写input_binding验收。"),
    ]
    if verified:
        items += [(check, "study", "pass", detail) for check, detail in [
            ("I01", f"整理者自检：全部{len(images)}行主键、39基础字段及来源路径有效。"),
            ("I02", "整理者自检：全部image的sample与panel外键有效。"),
            ("I03", "整理者自检：全部原生完整ROI框、float32、1um尺度与中心方形补边核对。"),
            ("I04", "整理者自检：逐图作者core/采集注释与sample归属核对；代表图覆盖队列/组织类型、最小/最大/极端宽高比和多shard布局；没有多sample拆分。"),
            ("P03", f"整理者自检：全部{verified['headers_checked']}个输出TIFF重开，CYX、float32、33通道名称与顺序一致。"),
            ("I05", json.dumps(dict(note="整理者自检；每个代表图全部33通道的信号/边界窗口逐值比较，padding四侧检查", checks=verified["pixel_checks"]), ensure_ascii=False)),
        ]]
    for check, scope, result, detail in items:
        record = dict.fromkeys(template.columns, "")
        record.update(study=STUDY, preparer=f"Jackson2020 {JOB.name}", scope=scope,
            record_id=PANEL if scope == "panel" else STUDY, check_id=check, result=result,
            source_evidence=relative(CACHE / "source_trace.json") + ";" + relative(CACHE / "sample_linkage.json") + ";" + relative(native_path),
            output_evidence=relative(OUTPUT) + (";" + relative(CACHE / "preparer_verification.json") if verified else ""), details=detail)
        rows.append(record)
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(rows, columns=template.columns).to_csv(review_path, index=False, encoding="utf-8-sig")


class CoreReader:
    """Decode one native core and stage its pixels for local TIFF construction.

    This study's measured cores are at most 883 x 1032 pixels, < 143 MiB for all
    39 float32 channels. No whole-slide source is handled by this local adapter.
    The shared Zarr reader currently decompresses all-channel inner chunks on
    each selected-channel read; this is a study-local performance adaptation.
    """

    def __init__(self, image_id: str):
        item = inventory_by_id[image_id]
        self.source = SourceSpec(image_id, PANEL, "zarr", ROOT / item["path"] / "0", tuple(range(39)), pixel_size_um=(1.0, 1.0))
        array = zarr.open_array(str(self.source.path), mode="r")
        self.values = np.asarray(array[:])
        self.shape, self.dtype = self.values.shape, self.values.dtype
        assert list(self.shape) == item["shape"] and self.dtype == np.dtype("float32")

    def stage_local(self, path: Path) -> None:
        """Make the in-memory native decode an actual local source for export_input."""
        np.save(path, self.values)
        self.values = np.load(path, mmap_mode="r")
        self.source = SourceSpec(self.source.image_id, PANEL, "npy", path,
                                 tuple(range(39)), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


if args.export or not (args.overview or args.verify):
    save_tables()
    write_reviews()
    progress("metadata_ready")

if args.export:
    assert shutil.disk_usage(ROOT).free - output_estimate > 100 * 1024**3
    assert shutil.disk_usage("/tmp").free > 64 * 1024**3
    staging_root = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
    staging_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in linked.to_dict("records"):
        image_id = row["tissue_id"]
        item = inventory_by_id[image_id]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in receipts:
            receipt = receipts[image_id]
            assert receipt["source_objects"] == item["source_objects"]
            assert receipt["kept_source_indices"] == kept.source_channel_index.tolist()
            assert receipt["channel_names"] == kept.marker_id.tolist()
            if receipt["admitted"]:
                assert image_id in images.index and destination.stat().st_size == receipt["output_size"]
                assert destination.stat().st_mtime_ns == receipt["output_mtime_ns"]
            continue
        if args.limit is not None and count >= args.limit:
            break
        try:
            progress("exporting", [f"当前{image_id}", "独立review待执行"])
            with tempfile.TemporaryDirectory(prefix="jackson-native-", dir=staging_root) as temporary:
                with formal_read_lease(ROOT / item["path"]):
                    for source_object in item["source_objects"]:
                        source_stat = (ROOT / source_object["path"]).stat()
                        assert (source_stat.st_size, source_stat.st_mtime_ns) == (
                            source_object["size"], source_object["mtime_ns"]), source_object["path"]
                    reader = CoreReader(image_id)
                reader.stage_local(Path(temporary) / "source.npy")
                report = export_input(reader, panel, destination)
                del reader
            if report["admitted"]:
                record = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
                record.update(study=STUDY, image_id=image_id, sample_id=row["sample_id"], panel_id=PANEL,
                    image_kind="tma_core", source_image_id=image_id, source_files=[item["path"]],
                    source_format="zarr", source_axes="CYX", source_dtype="float32",
                    source_revision="SPoRA jackson2020single 1.0mpp; accepted 723 native proofs and source object manifest",
                    slide_id=row["TMA_block_label"] if row["patient_cohort"] == "Basel" else "ZTMA208",
                    core_id=row["core"], roi_id=row["core"], region_annotation=row["source_region"],
                    source_x0_px=0, source_y0_px=0, source_x1_px=row["image_width"], source_y1_px=row["image_height"],
                    pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                    intensity_description="Native published SPoRA float32 intensities; unchanged decoded values, no normalization or clipping; upstream processing not independently established.",
                    split_method="already_independent", split_evidence=relative(RAW / "metadata/tissues.parquet") + ";" + row["author_table"],
                    metadata_notes="Original file: " + row["original_file_name"] + "; source display group names not used as identity; complete core acquisition including measured zero/background pixels. Author core/PID define specimen group.")
                record.update({key: report[key] for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]})
                images.loc[image_id] = pd.Series(record)
                report.update(output_size=destination.stat().st_size, output_mtime_ns=destination.stat().st_mtime_ns)
            report.update(source_objects=item["source_objects"], kept_source_indices=kept.source_channel_index.tolist(), channel_names=kept.marker_id.tolist())
            receipts[image_id] = report
            count += 1
            if count % 5 == 0:
                checkpoint("exporting")
            print(json.dumps(dict(image_id=image_id, completed_images=len(images), admitted=report["admitted"])), flush=True)
        except Exception as error:
            if count % 5:
                checkpoint("exporting")
            progress("export_failed", [f"{image_id}: {type(error).__name__}: {error}"])
            raise
    if count % 5:
        checkpoint("exporting")
    write_reviews()
    progress("export_complete" if len(receipts) == len(linked) else "export_partial")

if args.overview:
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = []
    for _, group in linked.groupby(["patient_cohort", "biological_tissue"]):
        selected.append(group.tissue_id.iloc[0])
    selected += [min(inventory, key=lambda x: np.prod(x["shape"][1:]))["image_id"],
                 max(inventory, key=lambda x: np.prod(x["shape"][1:]))["image_id"],
                 max(inventory, key=lambda x: max(x["shape"][1:]) / min(x["shape"][1:]))["image_id"],
                 max(inventory, key=lambda x: len(x["source_objects"]))["image_id"]]
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 2.8*len(selected)), squeeze=False)
    for i, image_id in enumerate(selected):
        with formal_read_lease(ROOT / inventory_by_id[image_id]["path"]):
            source = zarr.open_array(str(ROOT / inventory_by_id[image_id]["path"] / "0"), mode="r")
            for j, (label, indices) in enumerate([("DNA1", [37]), ("DNA2", [38]), ("PanCK/CD45/Vimentin", [35, 22, 10])]):
                values = np.asarray(source.oindex[indices, :, :])
                display = [np.clip(value / max(float(np.quantile(value[value > 0], .995)) if np.any(value > 0) else 1, 1e-12), 0, 1) for value in values]
                axes[i, j].imshow(display[0] if len(display) == 1 else np.stack(display, axis=-1), cmap="gray", vmin=0, vmax=1)
                region = linked.set_index("tissue_id").loc[image_id]
                axes[i, j].set_title(image_id + "\n" + str(region.source_region) + " / " + label, fontsize=7)
                axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(CACHE / "representative_overview.jpg", dpi=110)
    plt.close(fig)
    save_json(CACHE / "overview_selection.json", dict(image_ids=selected, selection="Each cohort/tissue group; minimum and maximum area, largest aspect ratio, multi-shard source.", display_only="Positive-pixel P99.5 scaling for overview; native TIFF values unchanged."))

if args.verify:
    selected = json.loads((CACHE / "overview_selection.json").read_text())["image_ids"]
    expected_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    assert expected_samples.to_dict("records") == samples.to_dict("records")
    verify_path = CACHE / "preparer_verify_progress.json"
    prior_verify = json.loads(verify_path.read_text()) if verify_path.exists() else {}
    completed = prior_verify.get("completed", {})
    checks = prior_verify.get("pixel_checks", [])
    assert set(completed).issubset(images.index)
    for image_id, version in completed.items():
        receipt = receipts[image_id]
        assert version == [receipt["output_size"], receipt["output_mtime_ns"]]
        current = (ROOT / images.loc[image_id, "input_image_path"]).stat()
        assert version == [current.st_size, current.st_mtime_ns]
    progress("verifying", verified_headers=len(completed), resume_after="verify_remaining_images")
    for record in images.to_dict("records"):
        image_id = record["image_id"]
        if image_id in completed:
            continue
        height, width = inventory_by_id[image_id]["shape"][1:]
        side = max(height, width)
        left, top = (side-width)//2, (side-height)//2
        assert [record["source_x0_px"], record["source_y0_px"], record["source_x1_px"], record["source_y1_px"]] == [0, 0, width, height]
        assert [record["pad_left_px"], record["pad_top_px"], record["pad_right_px"], record["pad_bottom_px"]] == [left, top, side-width-left, side-height-top]
        assert record["height_px"] == record["width_px"] == side
        with formal_read_lease(ROOT / record["input_image_path"]), tifffile.TiffFile(ROOT / record["input_image_path"]) as tif:
            assert tif.series[0].shape == (33, side, side) and tif.series[0].dtype == np.dtype("float32")
            assert tif.series[0].axes == "CYX" and len(tif.pages) == 33
            xml = ET.fromstring(tif.ome_metadata)
            assert [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")] == kept.marker_id.tolist()
            pixels = next(node for node in xml.iter() if node.tag.endswith("}Pixels"))
            assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0
            assert pixels.attrib["PhysicalSizeXUnit"] == pixels.attrib["PhysicalSizeYUnit"] == "µm"
            if image_id in selected:
                # Shared window reader needs TiffPage tags, absent on cached OME TiffFrames.
                tif.pages.cache = False
                tif.pages.useframes = False
                source = zarr.open_array(str(ROOT / inventory_by_id[image_id]["path"] / "0"), mode="r")
                dna = np.asarray(source[37])
                y, x = np.unravel_index(dna.argmax(), dna.shape)
                x, y = max(0, min(int(x)-16, width-32)), max(0, min(int(y)-16, height-32))
                for kind, box in [("signal", (x, y, min(x+32,width), min(y+32,height))), ("boundary", (max(0,width-32), max(0,height-32), width, height))]:
                    x0,y0,x1,y1 = box
                    expected = np.asarray(source.oindex[kept.source_channel_index.tolist(), slice(y0,y1), slice(x0,x1)])
                    for out in range(33):
                        actual = read_tiff_window(tif, out, (x0+left,y0+top,x1+left,y1+top))
                        assert actual.dtype == expected.dtype and np.array_equal(actual, expected[out]), (image_id, out, box)
                    checks.append(dict(image_id=image_id, source_window=box, kind=kind, channels=33, exact_equal=True))
                for channel in range(33):
                    for box in [(0,0,side,top),(0,0,left,side),(0,top+height,side,side),(left+width,0,side,side)]:
                        if box[0] < box[2] and box[1] < box[3]:
                            assert not read_tiff_window(tif, channel, box).any()
        receipt = receipts[image_id]
        completed[image_id] = [receipt["output_size"], receipt["output_mtime_ns"]]
        if len(completed) % 10 == 0 or len(completed) == len(images):
            save_json(verify_path, dict(completed=completed, pixel_checks=checks))
            progress("verifying", verified_headers=len(completed), resume_after="verify_remaining_images")
            print(json.dumps(dict(verified_headers=len(completed), expected_images=len(images))), flush=True)
    verification = dict(study=STUDY, role="preparer_self_check", headers_checked=len(images), pixel_checks=checks,
        sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=len(samples),
        distinct_patients=samples.patient_key.nunique(), independent_review="pending", output_estimate_bytes=output_estimate)
    save_json(CACHE / "preparer_verification.json", verification)
    write_reviews(verification)
    progress("prepared_self_checked")

print(json.dumps(dict(study=STUDY, expected_images=len(tissues), exported_images=len(images),
    sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=len(samples),
    distinct_patients=samples.patient_key.nunique(), retained_channels=len(kept), output_estimate_bytes=output_estimate)), flush=True)
