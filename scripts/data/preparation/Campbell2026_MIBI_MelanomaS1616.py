"""Deliver the 202 complete S1616 MIBI ROI as native-intensity study inputs.

Inputs: Zenodo 15547911 ZIP, existing native proofs, study-specific S1 rows,
published supplementary Tables S1/S2, and the author's public analysis code.
Outputs: independent CYX float32 input TIFFs, 39-field image metadata, sample,
panel, field-mapping tables, preparer checks, and resumable per-image receipts.
The study-local reader caches one scalar 2048-square page (16 MiB), avoiding
repeated strip decompression; migrate this optimization through the controller.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "4"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Campbell2026_MIBI_MelanomaS1616"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_15547911/MIBI_dataset.zip"
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
PROGRESS = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_PROGRESS",
    str(PROJECT / "cache/study_preparation" / STUDY / "jobs/0020_prepare_0/progress.json"),
))
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = STUDY + "__20220705_PICI_P138"
EXPECTED = 202
GAPS = ["论文/归档为202个ROI；补充Table S2的mibi.n_roi合计161，17份标本计数不符，未据此删图。",
        "整理者完成自检；独立CLI复核和D01验收由主控组织。"]


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress(phase: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, expected_images=EXPECTED,
                             completed_images=completed, gaps=GAPS,
                             updated_at=datetime.now(timezone.utc).isoformat(), **details))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def source_inventory() -> list[dict]:
    """Reuse accepted native acquisition proofs, retaining compact source headers.

    No source hash or whole-image native verification is repeated. ZIP member
    length/CRC/header offset and original source size/mtime bind the old proof.
    """
    path = CACHE / "source_inventory.json"
    if path.exists():
        return json.loads(path.read_text())
    verification = json.loads((META / "native_verification.json").read_text())
    assert verification["scope_complete"] and not verification["source_defects"]
    result = []
    for image in verification["native_result"]["images"]:
        assert image["complete"] and image["counted"] and image["native_order_verified"]
        result.append(dict(source_member=image["source_member_name"], physical_image_id=image["image_id"],
                           physical_identity=image["physical_identity_fields"], file_stat=image["source_file_stat"],
                           member_stat=image["source_member_stat"],
                           native_header=image["native_page_metadata"][0]["metadata"],
                           source_channel_names=image["channels"], shape=image["common_shape"],
                           proof_path=relative(Path(image["proof_path"])), complete=image["complete"]))
    write_json(path, result)
    return result


class ScalarPageReader:
    """Window a native MIBI ROI, decoding at most one scalar channel into RAM.

    The shared exporter visits channels sequentially. This specific source has
    narrow compressed strips; a 16-MiB page cache avoids decoding each strip
    four times for the four output tile columns. Measured zeros remain pixels.
    """

    def __init__(self, tif: tifffile.TiffFile):
        self.tif = tif
        self.shape = (len(tif.pages), *tif.pages[0].shape)
        self.dtype = tif.pages[0].dtype
        self.source = SimpleNamespace(pixel_size_um=(0.390625, 0.390625))
        self.page_index = None
        self.page = None

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        values = []
        for channel in channels:
            if channel != self.page_index:
                self.page = self.tif.pages[channel].asarray(maxworkers=1)
                self.page_index = channel
            values.append(self.page[y0:y1, x0:x1])
        return np.stack(values)

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def extract_member(archive: zipfile.ZipFile, member: str, root: Path) -> Path:
    """Stage only the current expression image; preserve original ZIP members."""
    info = archive.getinfo(member)
    path = root / (member.split('/')[0] + ".tiff")
    if path.exists() and path.stat().st_size == info.file_size:
        return path
    assert shutil.disk_usage(root).free >= info.file_size + 64 * 1024**3
    temporary = path.with_suffix(".tiff.partial")
    with archive.open(info) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024**2)
    assert temporary.stat().st_size == info.file_size
    temporary.replace(path)
    return path


def check_header(path: Path, panel: pd.DataFrame) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX" and tif.series[0].shape == (33, 2048, 2048)
        assert all(page.shape == (2048, 2048) and page.dtype == np.dtype("float32") for page in tif.pages)
        xml = ElementTree.fromstring(tif.ome_metadata)
        assert [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")] == panel.marker_id.tolist()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--limit", type=int, help="Initial bounded smoke batch; a successful positive batch resumes the full scope in the same IO lease")
parser.add_argument("--stop-after-limit", action="store_true", help="Stop after the bounded batch instead of continuing the full study")
parser.add_argument("--overview-observation", help="Preparer observation supplied after viewing the actual contact sheet")
parser.add_argument("--temporary-root", type=Path, default=Path("/tmp/study_input_batch") / STUDY)
args = parser.parse_args()
for directory in (CACHE, CACHE / "metadata", CACHE / "receipts", CACHE / "overview", OUTPUT, args.temporary_root):
    directory.mkdir(parents=True, exist_ok=True)

# Reconcile the canonical archive and old native physical identities.
inventory = source_inventory()
assert len(inventory) == EXPECTED and len({row["physical_image_id"] for row in inventory}) == EXPECTED
assert RAW.stat().st_size == 3210188184
for row in inventory:
    assert row["complete"] and row["shape"] == [2048, 2048]
    assert RAW.stat().st_size == row["file_stat"]["size"]
    assert RAW.stat().st_mtime_ns == row["file_stat"]["mtime_ns"]
    assert row["native_header"]["mibi.panel"] == "20220705_PICI P138"
    assert row["source_channel_names"] == inventory[0]["source_channel_names"]
with zipfile.ZipFile(RAW) as archive:
    source_members = [info for info in archive.infolist()
                      if not info.filename.startswith("__MACOSX/") and info.filename.endswith("/expression.tiff")]
    assert {info.filename for info in source_members} == {row["source_member"] for row in inventory}
    for row in inventory:
        info = archive.getinfo(row["source_member"])
        assert (info.file_size, info.compress_size, info.CRC, info.header_offset) == tuple(
            row["member_stat"][key] for key in ("file_size", "compress_size", "CRC", "header_offset"))
    archive_counts = dict(archive_entries=len(archive.infolist()), expression_images=len(source_members),
                          other_biological_files=len([i for i in archive.infolist() if not i.is_dir()
                                                     and not i.filename.startswith("__MACOSX/")]) - EXPECTED)
panel = panel_from_s1(STUDY, PANEL_ID, inventory[0]["source_channel_names"], S1)
assert len(panel) == 33 and panel.keep_channel.all()
assert panel.loc[panel.source_channel_index == 20, "role"].item() == "segmentation-only"
panel.loc[panel.source_channel_index == 20, "keep_reason"] = "S1 selected body input; mixed HLA-I/Na-K-ATPase excluded from downstream measurements"
panel.loc[panel.source_channel_index == 16, "keep_reason"] = "S1 selected dsDNA nuclear input"

# Use the publication's explicit mibi.id -> specimen -> subject crosswalk.
# The PMC XLSX endpoint returned an HTML interstitial; the identical published
# attachment is acquired from AACR's figshare record 33402734, file 68090540.
clinical_file = NAS / "raw/public" / STUDY / "metadata/supplement_tables_figshare.xlsx"
patients = pd.read_excel(clinical_file, sheet_name="Supplementary Table S1", dtype=str).fillna("")
specimens = pd.read_excel(clinical_file, sheet_name="Supplementary Table S2", dtype=str).fillna("")
specimens = specimens[specimens["mibi.id"] != ""].copy()
assert len(specimens) == 45 and specimens.subject.nunique() == 26
assert not specimens["mibi.id"].duplicated().any() and not specimens.specimen.duplicated().any()
patients = patients[patients.subject.isin(specimens.subject)]
assert not patients.subject.duplicated().any()
specimens = specimens.merge(patients[["subject", "arm", "subtype", "prior_mapki", "resp", "resp_group"]],
                             on="subject", validate="many_to_one")
sample_map = specimens.set_index("mibi.id").to_dict("index")
source_keys = [row["source_member"].split('/')[0].rsplit("_ROI", 1)[0] for row in inventory]
assert set(source_keys) == set(sample_map)
source_labels = pd.DataFrame([
    dict(mibi_id=key, embedded_specimen=row["native_header"]["mibi.fov_name"].rsplit("_ROI", 1)[0])
    for key, row in zip(source_keys, inventory)
])
assert (source_labels.groupby("mibi_id").embedded_specimen.nunique() == 1).all()
assert (source_labels.groupby("embedded_specimen").mibi_id.nunique() == 1).all()
assert all(row["source_member"].split('/')[0].rsplit("_ROI", 1)[1] ==
           row["native_header"]["mibi.fov_name"].rsplit("_ROI", 1)[1] for row in inventory)
counts = pd.Series(source_keys).value_counts().to_dict()
discrepancies = [dict(mibi_id=key, table_s2_roi_count=int(value["mibi.n_roi"]), archive_roi_count=counts[key])
                 for key, value in sample_map.items() if int(value["mibi.n_roi"]) != counts[key]]
sample_rows = []
for record in specimens.to_dict("records"):
    row = dict.fromkeys(columns("sample_metadata"), "")
    row.update(study=STUDY, sample_id=record["specimen"], source_sample_id=record["specimen"],
               patient_key=STUDY + "__" + record["subject"], source_subject_id=record["subject"],
               cohort_id="SWOG S1616", sample_type="tumor biopsy", tissue_type="tumor",
               timepoint_label={"baseline": "baseline", "ontx c2": "on_treatment"}[record["timepoint"]],
               preservation_method="ffpe", primary_diagnosis="melanoma",
               histology=record["subtype"] if record["subtype"] != "Unknown" else "",
               index_therapy_modality="immunotherapy",
               index_regimen={"Combination": "ipilimumab + nivolumab", "Ipilimumab": "ipilimumab"}[record["arm"]],
               response_criteria="RECIST 1.1", best_overall_response=record["resp"],
               index_event="treatment_start",
               index_definition_version="published trial S1616 treatment baseline",
               metadata_notes=json.dumps(dict(source_sheet="Supplementary Table S2", mibi_id=record["mibi.id"],
                                                specimen_type=record["specimen_type"], source_timepoint=record["timepoint"],
                                                source_collection_window=("before treatment" if record["timepoint"] == "baseline"
                                                                          else "28-35 days after treatment start"),
                                                source_melanoma_histology=record["subtype"], path_note=record["path.note"],
                                                path_exclusion=record["path.exclusion"],
                                                prior_mapki=record["prior_mapki"],
                                                source_response_group=record["resp_group"],
                                                table_roi_count=int(record["mibi.n_roi"]),
                                                archive_roi_count=counts[record["mibi.id"]],
                                                clinical_sheet="Supplementary Table S1"), ensure_ascii=False))
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=columns("sample_metadata"))
dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
for field_definition in dictionary[(dictionary.table_name == "sample_metadata") &
                                   (dictionary.value_set != "")].itertuples():
    actual = set(samples[field_definition.field_name]) - {""}
    allowed = set(vocabulary.loc[vocabulary.value_set == field_definition.value_set, "allowed_value"])
    assert actual <= allowed, (field_definition.field_name, actual - allowed)
investigation = dict(
    sources=[dict(path=relative(META), result="既有202份native proof、33通道顺序和物理采集身份"),
             dict(url="https://zenodo.org/records/15547911", path=relative(META / "official_zenodo_record.json"),
                  result="完整MIBI_dataset.zip；all_segments.tsv为非必要细胞定量产物，未下载"),
             dict(url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13274726/", path=relative(RAW_METADATA / "paper_live.html"),
                  result="202 ROI、45份biopsy、26位患者；800µm视野、2048像素、0.390625µm/pixel"),
             dict(url="https://api.figshare.com/v2/articles/33402734", path=relative(RAW_METADATA / "figshare_tables.json"),
                  attachment_url="https://ndownloader.figshare.com/files/68090540", attachment_path=relative(clinical_file),
                  result="Supplementary Table S2明确mibi.id/specimen/subject，Table S1提供患者治疗臂/反应/亚型"),
             dict(url="https://github.com/kcampbel/s1616-public", path=relative(RAW_METADATA / "github_tree.json"),
                  code_path=relative(RAW_METADATA / "author_analysis_mibi.R"),
                  result="完整main树仅4份R脚本；analysis_mibi.R第27–40行明确用Table S1/S2连接sample与patient；无单独roi-map附件")],
    join_keys="ZIP ROI folder minus terminal _ROI<number> == Table S2 mibi.id; Table S2 subject == Table S1 subject",
    expected_images=EXPECTED, sample_linked_images=202, sample_rows=45, patient_linked_samples=45, patients=26,
    table_s2_roi_sum=int(pd.to_numeric(specimens["mibi.n_roi"]).sum()), roi_count_discrepancies=discrepancies,
    discrepancy_resolution="精确身份键均匹配；以完整归档与物理采集证明确定202图范围，mibi.n_roi不作为图像排除规则",
    optional_fields_missing="精确采样日、解剖位点、年龄、性别、生存信息未随已连接表提供；on-therapy采样窗口已按论文记录",
    source_name_caution="embedded mibi.fov_name用于保留采集标本标签；患者身份来自S2 subject，未从名称猜测")
write_json(CACHE / "sample_investigation.json", investigation)
write_json(CACHE / "source_scope.json", dict(study=STUDY, canonical_release="zenodo_15547911", **archive_counts,
           native_verification_reused=relative(META / "native_verification.json"), hashes_recomputed=False,
           physical_images=EXPECTED, panel_source_channels=33, retained_channels=33,
           downstream_channels=int((panel.role == "downstream").sum()),
           maximum_output_uncompressed_bytes=EXPECTED * 33 * 2048 * 2048 * 4,
           local_one_expression_member_bytes=max(row["member_stat"]["file_size"] for row in inventory),
           nas_reserve_bytes=100 * 1024**3, temporary_reserve_bytes=64 * 1024**3,
           exclusions="AppleDouble资源叉/目录不作图像；segments/tissue/tumor/endothelial mask和细胞表留在原ZIP，后续阶段另行处理"))

# Define every template field, including optional empty fields and clinical origins.
field_rows = []


def field(source: str, source_field: str, table: str, target: str, rule: str,
          notes: str = "", sheet: str = "", join: str = "") -> None:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=source, source_sheet_or_table=sheet, source_field=source_field,
               target_table=table, target_field=target, join_keys=join, transformation_rule=rule,
               conflict_rule="来源身份精确连接；未知留空；不从ROI名称推断患者", notes=notes)
    field_rows.append(row)


image_rules = {
    "study": "任务确认的canonical study ID",
    "image_id": "完整ZIP ROI目录名；每个来源物理采集一行",
    "sample_id": "ROI目录去末尾_ROI<number>，精确连接S2 mibi.id，取specimen；202/202连接45份标本",
    "panel_id": "实际mibi.panel=20220705_PICI P138；全202图有序33通道完全一致",
    "image_kind": "roi；来源病理学家选择的独立800µm×800µm ROI",
    "source_image_id": "ZIP内完整expression.tiff成员路径",
    "source_files": "仅原始MIBI_dataset.zip的NAS相对路径列表；不是临时提取副本",
    "source_format": "tiff_zip_member；独立expression.tiff按marker page读取",
    "source_acquisition_id": "来源非MCD；物理身份mibi.run/fov_id/coordinates存metadata_notes；此字段空",
    "source_z_index": "来源33个二维marker pages；无Z层，空",
    "source_t_index": "来源无时间轴，空；取样时点保存于sample",
    "source_axes": "IYX；tifffile原生33个page轴I逐页对应marker，输出CYX",
    "source_dtype": "原始TIFF解码float32，保持原值",
    "source_revision": "zenodo_15547911；原ZIP size/mtime和member元数据绑定既有native proof",
    "slide_id": "原始mibi.slide转字符串",
    "core_id": "独立活检ROI，未发布物理TMA core身份，空",
    "roi_id": "完整ZIP ROI目录名，保留来源ROI身份",
    "fov_id": "原始mibi.fov_id；与run和coordinates结合识别唯一采集",
    "region_annotation": "mibi.tissue原值；本来源202图均为空；不把样本级path.note当作ROI级注释",
    "source_x0_px": "完整原生ROI框左界0，含边界",
    "source_y0_px": "完整原生ROI框上界0，含边界",
    "source_x1_px": "完整原生ROI框右界2048，不含边界",
    "source_y1_px": "完整原生ROI框下界2048，不含边界",
    "pad_left_px": "原生ROI已方形，0", "pad_top_px": "原生ROI已方形，0",
    "pad_right_px": "原生ROI已方形，0", "pad_bottom_px": "原生ROI已方形，0",
    "height_px": "完整采集2048", "width_px": "完整采集2048",
    "source_channel_count": "202份原生proof及实际TIFF headers均33通道",
    "channel_count": "S1保留33通道；31个downstream、dsDNA核输入和混合HLA-I/Na-K-ATPase仅分割输入",
    "pixel_size_x_um": "论文Methods:800µm×800µm、2048²，明示0.390625µm/pixel；与mibi.size一致",
    "pixel_size_y_um": "论文Methods明示0.390625µm/pixel，与原生ROI尺度一致",
    "input_image_path": "processed/<study>/images/<ROI目录名>/input.tiff，NAS相对路径",
    "valid_pixels_path": "来源完整dense scalar TIFF，未标示未采集域；不按全零像素删除采集，此列空",
    "intensity_description": "保持来源SIMS expression.tiff的float32信号；具体上游强度校正未随TIFF完整说明；本步骤不变换",
    "split_method": "already_independent；完整来源ROI无需sample拆分",
    "split_evidence": "source_inventory逐ROI物理采集、ZIP路径及S2 mibi.id->specimen对应",
    "metadata_notes": "JSON保留physical_image_id、mibi.fov_name、mibi.run、coordinates、panel、sample连接键与proof路径",
}
assert set(image_rules) == set(columns("image_metadata"))
for target, rule in image_rules.items():
    source = relative(CACHE / "source_inventory.json")
    if target == "sample_id":
        source = relative(clinical_file)
    field(source, "mibi.id; specimen" if target == "sample_id" else "source_member; native_header; accepted native proof",
          "image_metadata", target, rule,
          sheet="Supplementary Table S2" if target == "sample_id" else "",
          join="ZIP ROI prefix == mibi.id" if target == "sample_id" else "source_member")
for target, source_field in {"sample_id": "specimen", "source_sample_id": "specimen", "patient_key": "subject",
                             "source_subject_id": "subject", "timepoint_label": "timepoint",
                             "preservation_method": "specimen_type", "metadata_notes": "mibi.id; specimen_type; path.note; path.exclusion; mibi.n_roi"}.items():
    field(relative(clinical_file), source_field, "sample_metadata", target,
          "精确mibi.id连接；patient_key添加study命名空间；ffpe block->ffpe；baseline->baseline、ontx c2->on_treatment，原时点保留notes；病理注释及ROI计数存notes",
          "45份标本、202张图、45份sample连接26位患者；17份mibi.n_roi与归档不符，详见sample_investigation.json",
          sheet="Supplementary Table S2", join="ROI prefix == mibi.id; subject == S1 subject")
for target, source_field, rule in [
    ("histology", "subtype", "论文Methods称melanoma histology；保留Cutaneous/Mucosal；Unknown留空，原值均保留notes"),
    ("index_regimen", "arm", "Combination=ipilimumab+nivolumab；Ipilimumab=ipilimumab；论文Methods确认药物"),
    ("best_overall_response", "resp", "保留CR/PR/SD/PD；患者层级反应按subject传播到其样本"),
]:
    field(relative(clinical_file), source_field, "sample_metadata", target, rule,
          sheet="Supplementary Table S1", join="S2 subject == S1 subject; validated many_to_one")
field(relative(clinical_file), "prior_mapki; subtype; resp_group", "sample_metadata", "metadata_notes",
      "保留来源prior_mapki、原始melanoma histology和resp_group；NA按来源缺失读为空，不解释为从未使用MAPK抑制剂",
      sheet="Supplementary Table S1", join="S2 subject == S1 subject; validated many_to_one")
field(relative(RAW_METADATA / "paper_live.html"),
      "baseline before treatment; on-therapy biopsy 28-35 days after treatment start",
      "sample_metadata", "metadata_notes",
      "保留论文报告的样本采集时间窗；不将范围换算为具体采样日",
      sheet="Methods: Patient sample acquisition", join="Table S2 timepoint")
for target in ("study", "cohort_id", "sample_type", "tissue_type", "primary_diagnosis", "index_therapy_modality",
               "response_criteria", "index_event", "index_definition_version"):
    field(relative(RAW_METADATA / "paper_live.html"), "Methods: Clinical samples, trial S1616; RECIST 1.1",
          "sample_metadata", target, "按论文保留试验、肿瘤活检、治疗和RECIST语义；无精确采样日/生存起点推断")
field(relative(S1), "Study; Original marker; Unified marker; Segmentation channel", "panel_mapping",
      "source_channel_index", "Study+原始标签精确连接；source_channel_index按原TIFF page从0编号；无别名")
field(relative(S1), "Unified marker; Segmentation channel", "panel_mapping", "output_channel_index",
      "全部33通道保留，output位置连续；dsDNA核输入；混合HLA-I/Na-K-ATPase的mapping_status=excluded但role=segmentation-only，保留S1所选body像素")
fields = pd.DataFrame(field_rows, columns=columns("field_mapping"))

# Save a real index after each ROI; resume only files already bound to receipts.
image_index = OUTPUT / "images.parquet"
images = pd.read_parquet(image_index) if image_index.exists() else pd.DataFrame(columns=columns("image_metadata"))
images = images.set_index("image_id", drop=False)
receipt_rows = []
pixel_checks = []
if args.metadata_only:
    # A visual-observation refresh retains the completed export's pixel checks.
    for image_id in images.image_id:
        receipt = json.loads((CACHE / "receipts" / f"{image_id}.json").read_text())
        receipt_rows.append(receipt)
        pixel_checks.extend(receipt.get("pixel_checks", []))
errors = []
overview_indices = set(np.linspace(0, EXPECTED-1, 9, dtype=int).tolist())
attempted = 0
progress("metadata_prepared", len(images), sample_rows=45, sample_linked_images=len(images), patient_linked_samples=45)
save_input_tables(OUTPUT, images.reset_index(drop=True), samples, panel, fields)
if not args.metadata_only:
    assert shutil.disk_usage(NAS).free > 100 * 1024**3 + EXPECTED * 33 * 2048 * 2048 * 4
    with zipfile.ZipFile(RAW) as archive:
        for number, record in enumerate(inventory):
            member = record["source_member"]
            image_id = member.split('/')[0]
            destination = OUTPUT / "images" / image_id / "input.tiff"
            receipt_path = CACHE / "receipts" / f"{image_id}.json"
            if image_id in images.index and destination.exists() and receipt_path.exists():
                check_header(destination, panel)
                receipt = json.loads(receipt_path.read_text())
                assert receipt["source_member"] == member and receipt["export"]["admitted"]
                receipt_rows.append(receipt)
                pixel_checks.extend(receipt.get("pixel_checks", []))
                continue
            if args.limit is not None and attempted >= args.limit:
                break
            attempted += 1
            progress("exporting", len(images), current_image=image_id)
            staged = extract_member(archive, member, args.temporary_root)
            try:
                with tifffile.TiffFile(staged) as tif:
                    assert len(tif.pages) == 33 and tif.pages[0].shape == (2048, 2048)
                    actual_names = [json.loads(page.description)["channel.target"] for page in tif.pages]
                    assert actual_names == record["source_channel_names"]
                    assert json.loads(tif.pages[0].description) == record["native_header"]
                    assert all(page.dtype == np.dtype("float32") for page in tif.pages)
                    exported = export_input(ScalarPageReader(tif), panel, destination)
                    if not exported["admitted"]:
                        write_json(receipt_path, dict(source_member=member, export=exported,
                                                     exclusion="whole ROI all admitted downstream markers zero"))
                        progress("exporting", len(images), excluded_image=image_id)
                        continue
                    check_header(destination, panel)
                    checks = []
                    if number in overview_indices:
                        # Select an internal block containing an admitted marker,
                        # rather than relying on the geometric center being tissue.
                        for source_channel in panel.loc[panel.role == "downstream", "source_channel_index"]:
                            signal = tif.pages[source_channel].asarray(maxworkers=1)[32:-32, 32:-32]
                            positions = np.flatnonzero(signal)
                            if len(positions):
                                sy, sx = divmod(int(positions[len(positions)//2]), signal.shape[1])
                                internal = (sx+16, sy+16, sx+48, sy+48)
                                del signal, positions
                                break
                        else:
                            raise ValueError("Representative ROI has no internal admitted marker signal for pixel comparison")
                        with tifffile.TiffFile(destination) as saved:
                            for window in [(0, 0, 32, 32), internal]:
                                for channel in range(33):
                                    assert np.array_equal(read_tiff_window(tif, channel, window),
                                                          read_tiff_window(saved, channel, window))
                                checks.append(dict(image_id=image_id, window=list(window),
                                                   source_channel_indices=list(range(33)), exact_equal=True, dtype="float32",
                                                   block_kind="internal_with_marker_signal" if window == internal else "boundary"))
                        # Display-only scaling; TIFF intensities remain unchanged.
                        rgb = []
                        for channel in (30, 10, 16):
                            plane = tif.pages[channel].asarray(maxworkers=1)[::8, ::8]
                            positive = plane[plane > 0]
                            scale = np.percentile(positive, 99) if len(positive) else 1
                            rgb.append(np.clip(plane / scale, 0, 1))
                        np.save(CACHE / "overview" / f"{image_id}.npy", np.stack(rgb, axis=-1))
                metadata = dict.fromkeys(columns("image_metadata"), None)
                sample_key = image_id.rsplit("_ROI", 1)[0]
                header = record["native_header"]
                metadata.update(study=STUDY, image_id=image_id, sample_id=sample_map[sample_key]["specimen"],
                    panel_id=PANEL_ID, image_kind="roi", source_image_id=member, source_files=[relative(RAW)],
                    source_format="tiff_zip_member", source_axes="IYX", source_dtype="float32",
                    source_revision="zenodo_15547911", slide_id=str(header["mibi.slide"]), roi_id=image_id,
                    fov_id=header["mibi.fov_id"], region_annotation=header["mibi.tissue"] or None,
                    source_x0_px=0, source_y0_px=0, source_x1_px=2048, source_y1_px=2048,
                    pad_left_px=0, pad_right_px=0, pad_top_px=0, pad_bottom_px=0,
                    height_px=2048, width_px=2048, source_channel_count=33, channel_count=33,
                    pixel_size_x_um=0.390625, pixel_size_y_um=0.390625,
                    input_image_path=relative(destination),
                    intensity_description="Native float32 SIMS expression.tiff signal; release upstream correction unspecified; no input-stage intensity transform",
                    split_method="already_independent", split_evidence=relative(CACHE / "source_inventory.json") + "; " + relative(clinical_file) + "#Supplementary Table S2:mibi.id",
                    metadata_notes=json.dumps(dict(physical_image_id=record["physical_image_id"],
                        native_fov_name=header["mibi.fov_name"], native_run=header["mibi.run"],
                        native_coordinates=header["mibi.coordinates"], native_panel=header["mibi.panel"],
                        sample_join_key=sample_key, native_proof=record["proof_path"]), ensure_ascii=False))
                new_image = pd.DataFrame([metadata]).set_index("image_id", drop=False)
                images = pd.concat([images.drop(index=image_id, errors="ignore"), new_image])
                receipt = dict(image_id=image_id, source_member=member, native_proof_reused=record["proof_path"],
                               source_file_stat=record["file_stat"], member_stat=record["member_stat"],
                               export=exported, pixel_checks=checks)
                write_json(receipt_path, receipt)
                save_input_tables(OUTPUT, images.reset_index(drop=True), samples, panel, fields)
                receipt_rows.append(receipt)
                pixel_checks.extend(checks)
                progress("exporting", len(images), last_completed_image=image_id,
                         sample_rows=45, sample_linked_images=len(images), patient_linked_samples=45)
                print(json.dumps(dict(completed=len(images), expected=EXPECTED, image_id=image_id)), flush=True)
            except Exception as error:
                failure = dict(image_id=image_id, error_type=type(error).__name__, message=str(error))
                errors.append(failure)
                write_json(CACHE / "export_errors.json", errors)
                progress("export_error_continuing", len(images), error=failure)
                print(json.dumps(failure), flush=True)
            finally:
                staged.unlink(missing_ok=True)

# Build the contact sheet only from actually exported representative images.
overview_files = sorted((CACHE / "overview").glob("*.npy"))
if overview_files:
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 3, figsize=(12, 12))
    for axis, path in zip(axes.flat, overview_files):
        axis.imshow(np.load(path))
        axis.set_title(path.stem, fontsize=8)
        axis.axis("off")
    for axis in list(axes.flat)[len(overview_files):]:
        axis.axis("off")
    figure.suptitle("Native ROI overview: SOX10 red / CD45 green / dsDNA blue; display P99 only", fontsize=11)
    figure.tight_layout()
    figure.savefig(CACHE / "overview/contact_sheet.jpg", dpi=140)
    plt.close(figure)
if args.overview_observation:
    write_json(CACHE / "overview/observation.json", dict(preparer=STUDY + "__0020_prepare_0",
        evidence=relative(CACHE / "overview/contact_sheet.jpg"), observation=args.overview_observation))

# Self-check evidence is explicitly attributed to the preparer. D01 stays unresolved.
review_rows = []
for check, scope, result, details in [
    ("S01", "study", "pass", dict(source_images=202, delivered_images=len(images), apple_double_excluded=True,
                                    accepted_native_proofs_reused=202, supplementary_roi_count_discrepancies=discrepancies)),
    ("I01", "study", "pass", dict(unique_image_ids=len(images), source_member_and_physical_identity_preserved=True)),
    ("I02", "study", "pass", dict(sample_linked_images=len(images), sample_rows=45, patient_linked_samples=45, patients=26)),
    ("I03", "study", "pass", dict(source_bounds=[0, 0, 2048, 2048], padding=[0, 0, 0, 0], pixel_size_um=0.390625)),
    ("I04", "study", "pass" if (CACHE / "overview/observation.json").exists() else "unresolved",
     dict(sample_join="ZIP ROI prefix exact mibi.id -> specimen", split_method="already_independent",
          overview=relative(CACHE / "overview/contact_sheet.jpg"), observation=relative(CACHE / "overview/observation.json"))),
    ("P01", "panel", "pass", dict(source_channels=33, native_order_matches_all_images=True, s1_join="exact study+source label")),
    ("P02", "panel", "pass", dict(retained_channels=33, downstream=31, nuclear="dsDNA", mixed_body_channel_retained=True)),
    ("P03", "study", "pass", dict(headers_checked=len(images), axes="CYX", dtype="float32", shape=[33, 2048, 2048],
         header_evidence="Per-image successful export/check_header; metadata-only refresh reuses those checks")),
    ("I05", "study", "pass" if pixel_checks else "unresolved", dict(group="tiff_zip_member/IYX/single panel/already_independent/float32",
         sampled_images=len({row['image_id'] for row in pixel_checks}), checks=pixel_checks)),
    ("M01", "study", "pass", dict(source_investigation=relative(CACHE / "sample_investigation.json"),
         source_linked_images=202, delivered_linked_images=len(images), sample_rows=45, patient_linked_samples=45, patients=26)),
    ("M02", "study", "pass", dict(source_sheets=["Supplementary Table S1", "Supplementary Table S2"],
         source_fields="specimen,subject,timepoint,specimen_type,subtype,arm,resp", unknown_optional_values="blank",
         detailed_rules=relative(OUTPUT / "field_mapping.csv"))),
    ("D01", "study", "unresolved", dict(image_ids=images.image_id.tolist(), delivered_images=len(images), expected_images=EXPECTED,
         sample_rows=45, sample_linked_images=len(images), patient_linked_samples=45, patients=26,
         reason="整理者自检完成；独立CLI上下文复核与主控验收待执行", errors=errors)),
]:
    row = dict.fromkeys(columns("review_results"), "")
    row.update(study=STUDY, preparer=STUDY + "__0020_prepare_0", reviewer="", scope=scope,
        record_id=PANEL_ID if scope == "panel" else STUDY, check_id=check, result=result,
        source_evidence=relative(CACHE / "source_inventory.json") + ";" + relative(CACHE / "sample_investigation.json"),
        output_evidence=relative(OUTPUT), details=json.dumps(dict(preparer_self_check=True, **details), ensure_ascii=False))
    review_rows.append(row)
review_path = OUTPUT / "review_results.csv"
with formal_publish_lease(review_path, directory=OUTPUT):
    pd.DataFrame(review_rows, columns=columns("review_results")).to_csv(review_path, index=False, encoding="utf-8-sig")
write_json(CACHE / "export_summary.json", dict(study=STUDY, expected_images=EXPECTED, exported_images=len(images),
    sample_rows=45, sample_linked_images=len(images), patient_linked_samples=45, patients=26,
    native_dtype="float32", shape=[33, 2048, 2048], pixel_size_um=0.390625, errors=errors,
    overview_viewed=(CACHE / "overview/observation.json").exists(),
    pixel_sample_images=len({row['image_id'] for row in pixel_checks}),
    controller_requests=["安排不同CLI上下文独立复核及D01 input_binding验收", "按需将study-local单page缓存读取优化迁移到共享reader；本study已适配，无待修复阻断"],
    gaps=GAPS))
progress("prepared" if len(images) == EXPECTED and not errors else "partial", len(images),
         sample_rows=45, sample_linked_images=len(images), patient_linked_samples=45, patients=26, errors=errors)
print(json.dumps(dict(study=STUDY, exported_images=len(images), expected_images=EXPECTED, sample_rows=45,
                      sample_linked_images=len(images), patient_linked_samples=45, patients=26, errors=errors)), flush=True)
if errors:
    raise SystemExit(1)
if (args.limit is not None and args.limit > 0 and not args.stop_after_limit
        and not args.metadata_only and 0 < len(images) < EXPECTED):
    # The shared publication controller retains the NAS lease across exec. This
    # follows a real successful smoke export without a second long queue wait.
    print("Smoke batch passed; continuing the complete study within the same IO lease", flush=True)
    os.execv(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()),
                             "--temporary-root", str(args.temporary_root)])
