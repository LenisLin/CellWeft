"""Prepare the 35 localized Kim2023 IMC ROIs at native float32 intensity.

Inputs: Zenodo v1.0 TIFF planes, formal raw source metadata, the current
processed image scope, paper/supplement and author code, and the unique S1.
Outputs: independent square CYX input.tiff and the five study input tables.
Run export directly; shared helpers stage locally and lease each publication.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Kim2023_IMC_EsophagealAdenocarcinoma"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW_ROOT = NAS / "raw/public" / STUDY
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
EVIDENCE = META
PROGRESS = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_PROGRESS",
    f"/tmp/study_input_batch/{STUDY}/progress.json",
))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "kim2023_imc_roi_channel_filename_map"
PREPARER = "Kim2023_0317_prepare_0"
SOURCE_REVISION = "zenodo:7576005/version:1.0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["prepare", "overview", "export", "verify"], nargs="?", default="export")
args = parser.parse_args()
for directory in (CACHE / "image_receipts", EVIDENCE, OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def save_json(path: Path, value: dict | list) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def identity(path: Path) -> dict:
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def progress(stage: str, **details: object) -> None:
    save_json(PROGRESS, dict(
        study=STUDY, status="prepared" if stage == "prepared_awaiting_independent_review" else "partial",
        stage=stage, updated_at=datetime.now(timezone.utc).isoformat(),
        expected_images=len(manifest["roi_ids"]), exported_images=len(images),
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=0,
        gaps=([f"图像完成{len(images)}/{len(eligible)}"] if len(images) < len(eligible) else [])
             + ["公开标本未连接到明确患者编号；patient_key留空", "独立复核与主控验收待执行"],
        **details))


def processed_source_paths(images: pd.DataFrame, roi_ids: list[str]) -> dict[str, tuple[str, ...]]:
    """Bind the received Zenodo version to its processed ROI plane paths."""
    if images.image_id.duplicated().any() or set(images.image_id) != set(roi_ids):
        raise ValueError("Processed image scope differs from the raw source manifest")
    result = {}
    for row in images.itertuples():
        paths = tuple(row.source_files)
        if (row.source_revision != SOURCE_REVISION or row.source_dtype != "float32"
                or row.source_channel_count != 47 or len(paths) != 47):
            raise ValueError(f"Processed source version or channel scope differs: {row.image_id}")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
            raise ValueError(f"Processed source path is not NAS-relative: {row.image_id}")
        result[row.image_id] = paths
    return result


class NativePlanes:
    """Read contiguous uncompressed source planes by bounded mmap slices.

    The source stores one full-height strip per channel. Shared strip decoding
    otherwise rereads that strip for every square output tile. This study-local
    adapter preserves exact float32 values without decoding a whole CYX image.
    A shared uncompressed-TIFF fast path requires controller-coordinated migration.
    """

    def __init__(self, source: SourceSpec, stack: ExitStack):
        self.source = source
        self.arrays = []
        for path in source.paths:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                assert len(tif.pages) == 1 and page.is_contiguous and int(page.compression) == 1
            plane = tifffile.memmap(path, mode="r")
            stack.callback(plane._mmap.close)
            self.arrays.append(plane)
        self.shape = (len(self.arrays), *self.arrays[0].shape)
        self.dtype = self.arrays[0].dtype
        assert self.dtype == np.dtype("float32")
        assert all(a.shape == self.shape[1:] and a.dtype == self.dtype for a in self.arrays)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.stack([self.arrays[c][y0:y1, x0:x1] for c in channels])

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def check_image(path: Path, reader: NativePlanes) -> dict:
    """Check every header and two all-channel blocks, plus actual zero padding."""
    _, h, w = reader.shape
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    checks = []
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (len(retained), side, side)
        assert tif.series[0].axes == "CYX" and tif.series[0].dtype == reader.dtype
        ome = ET.fromstring(tif.ome_metadata)
        assert [c.attrib["Name"] for c in ome.findall(".//{*}Channel")] == retained.marker_id.tolist()
        physical = ome.find(".//{*}Pixels").attrib
        assert float(physical["PhysicalSizeX"]) == float(physical["PhysicalSizeY"]) == 1.0
        native_boxes = [(max(0, w//2-16), max(0, h//2-16), min(w, w//2+16), min(h, h//2+16)),
                        (0, 0, min(32, w), min(32, h))]
        for box in native_boxes:
            x0, y0, x1, y1 = box
            saved_box = (x0+left, y0+top, x1+left, y1+top)
            source = reader.read_window(retained.source_channel_index.tolist(), box)
            saved = np.stack([read_tiff_window(tif, c, saved_box) for c in range(len(retained))])
            assert source.dtype == saved.dtype and np.array_equal(source, saved)
            checks.append(dict(source_xyxy=list(box), output_xyxy=list(saved_box),
                               source_channels=retained.source_channel_index.tolist(),
                               exact_equal=True, nonzero_pixels=int(np.count_nonzero(source))))
        for box in [(0, 0, side, top), (0, top+h, side, side),
                    (0, top, left, top+h), (left+w, top, side, top+h)]:
            x0, y0, x1, y1 = box
            if x1 > x0 and y1 > y0:
                assert all(not read_tiff_window(tif, c, box).any() for c in range(len(retained)))
    return dict(header_pass=True, pixel_checks=checks, all_padding_zero=True)


def overview(reader: NativePlanes) -> Image.Image:
    """Render source DNA/PanCK/CD3 for inspection; display scaling affects only PNG."""
    _, h, w = reader.shape
    colors = reader.read_window([44, 16, 37], (0, 0, w, h))
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        high = float(np.quantile(colors[c], .995))
        rgb[:, :, c] = np.uint8(np.clip(colors[c]/high, 0, 1)**.5*255) if high else 0
    tile = Image.fromarray(rgb)
    tile.thumbnail((1100, 320))
    return tile


def save_overviews(rows: list[tuple[str, Image.Image]]) -> None:
    canvas = Image.new("RGB", (1160, 390*len(rows)), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (roi, tile) in enumerate(rows):
        y = index*390
        draw.text((10, y+10), f"{roi}: red=Ir191 DNA, green=PanCK, blue=CD3; native ROI; display only P99.5/sqrt", fill="white")
        canvas.paste(tile, (10, y+45))
    canvas.save(CACHE / "representative_overviews.png")


def publish(*, review: bool = True) -> None:
    save_input_tables(OUTPUT, images, samples, panel, fields)
    if not review:
        return
    rows = []
    for check, scope, record, result, details in [
        ("S01", "study", STUDY, "pass", f"本地候选范围35 ROI；实际交付{len(images)}；全零marker排除{len(excluded)}。归档IF、HE、注册结果、病理area mask和Apple元数据不另计image。"),
        ("I01", "study", STUDY, "pass", "逐image主键、来源47平面及输出路径可定位；逐图保存receipt与索引。"),
        ("I02", "study", STUDY, "pass", f"{len(images)} image共享1份有来源标本；panel外键一致。"),
        ("I03", "study", STUDY, "pass", "保持完整原生ROI框、1um尺度和float32；按共享公式居中方形补零。"),
        ("I04", "study", STUDY, "pass", "归档取样标签、35份采集坐标文件名及作者sample处理代码支持同一EMR组织标本；未拆分；病理area mask不直接转为单ROI诊断。"),
        ("P01", "panel", PANEL_ID, "pass", "47来源行按采集header核对；S1仅匹配本study原名或由文件名蛋白后缀证实的别名。"),
        ("P02", "panel", PANEL_ID, "pass", "36唯一蛋白测量+两条来源DNA核输入；Qdot800/EPR3653标记112/114Cd为同一CD14测量，保留112Cd，114Cd不另作表达。"),
        ("P03", "study", STUDY, "pass" if len(images) else "unresolved", "全部已输出图重开，38通道CYX及具体DNA名称一致；全量header和像素自检见逐图receipt。"),
        ("I05", "study", STUDY, "pass" if len(images) else "unresolved", "每张输出图、全部38保留通道，中心与左上边界各至多32x32逐值一致；全部padding实读为零。"),
        ("M01", "sample", sample_id, "unresolved", f"sample实际1行；image→sample={len(images)}/35；sample→patient=0/1。已查正式来源manifest、采集标签、Zenodo发布页、论文Experimental model and subject details、补充Figures S1–S7/Table S1、作者GitHub树及sample处理代码；标本无明确patient编号连接，未从标本字符串推断患者。"),
        ("M02", "sample", sample_id, "pass", "只填已证实的EMR组织、食管、FFPE；患者、年龄、性别、分期、治疗与生存未知留空。论文明确年龄与性别因匿名化不可提供。"),
        ("D01", "study", STUDY, "unresolved", f"整理者自检交付{len(images)}/35 image；sample1、连接{len(images)}、patient0。独立复核与主控验收待执行，无独立input_binding pass。"),
    ]:
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                         record_id=record, check_id=check, result=result,
                         source_evidence=relative(CACHE / "source_evidence.json"),
                         output_evidence=relative(OUTPUT),
                         details=json.dumps(dict(assessment="preparer_self_check", text=details), ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8-sig")


# Bind the formal raw source version to its current processed scope.
with formal_read_lease(RAW_ROOT / "metadata.json", data_root=NAS):
    with resource_lock(RAW_ROOT, shared=True), resource_lock(OUTPUT, shared=True):
        manifest = json.loads((META / "source_manifest.json").read_text())
        channel_map = json.loads((META / "source_ordered_channel_map.json").read_text())
        archive = RAW_ROOT / "zenodo_7576005" / manifest["source"]["filename"]
        assert archive.stat().st_size == manifest["source"]["size"]
        images = pd.read_parquet(OUTPUT / "images.parquet")
        source_paths_by_roi = processed_source_paths(images, manifest["roi_ids"])
channels = channel_map["channels"]
source_names = [c["source_channel"] for c in channels]
assert len(source_names) == 47 and [c["channel_index"] for c in channels] == list(range(47))
specimen = manifest["member_root"].split("/")[0].split("_", 2)[2]
sample_id = "EMR_" + specimen
header = (EVIDENCE / "acquisition_header.txt").read_text().splitlines()[1].split("\t")[6:]
assert len(header) == 47
assert all(c["isotope_prefix"][:re.match(r"\d+", c["isotope_prefix"]).end()] in h
           for c, h in zip(channels, header))
assert all(c["protein_marker"] in h for c, h in zip(channels, header) if c["protein_marker"])
eligible = list(manifest["roi_ids"])
shape_by_roi = images.set_index("image_id").apply(
    lambda row: (int(row.source_y1_px), int(row.source_x1_px)), axis=1).to_dict()
slide_labels = images.set_index("image_id").slide_id.to_dict()

# S1 source aliases come from actual isotope/antibody labels, not another study.
supplement = (EVIDENCE / "supplement_mmc1.txt").read_text()
assert re.search(r"Qdot\s+112/114\s+EPR3653", supplement)
aliases = {c["source_channel"]: c["protein_marker"] for c in channels if c["protein_marker"]}
selection = {c["channel_index"] for c in channels if c["protein_marker"] not in (None, "DNA")}
selection.remove(2)
panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1, aliases=aliases,
                      downstream_source_indices=selection)
panel.loc[panel.source_channel_index == 1, "keep_reason"] = (
    "Supplementary Table S1: Qdot800 112/114, EPR3653 CD14; retain source 112Cd as the single CD14 measurement")
panel.loc[panel.source_channel_index == 2, "exclusion_reason"] = (
    "Duplicate isotope readout of the same Qdot800/EPR3653 CD14 conjugate; 112Cd retained")
panel.loc[panel.source_channel_index.isin([44, 45]), "keep_reason"] = (
    "S1-selected DNA nuclear input; original registration code reads both Ir191 and Ir193 DNA planes")
retained = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(retained) == 38 and len(retained[retained.role == "downstream"]) == 36

# One source-labelled biological specimen is shared by all 35 acquisition ROIs.
samples = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str)
sample = {key: "" for key in samples.columns}
sample.update(study=STUDY, sample_id=sample_id, source_sample_id=specimen,
              sample_type="tissue", anatomical_site="esophagus", collection_method="resection",
              preservation_method="ffpe", processing_method="endoscopic mucosal resection (EMR)",
              metadata_notes="来源归档、35份采集标签及作者代码确认同一EMR标本；患者连接未知，未从名称猜测。论文明确匿名化后年龄/性别不可提供；ROI病理注释未直接提供。")
samples = pd.DataFrame([sample], columns=samples.columns)
fields = pd.read_csv(TEMPLATES / "field_mapping.csv", dtype=str)
field_rows = []
for source, source_field, table, target, rule, note in [
    (relative(META / "source_manifest.json"), "member_root; processed ROI scope", "sample_metadata", "sample_id", "同一标本标签共享一行；sample_id=EMR_+来源完整标本标签", "35 ROI→1 sample；此标本身份不是patient身份"),
    (relative(OUTPUT / "images.parquet"), "source_sample_id", "sample_metadata", "source_sample_id", "逐字保留来源标本标签；不拆取患者键", "当前处理范围中的35个ROI连接到同一标本标签"),
    (relative(RAW_METADATA / "paper.xml"), "Experimental model and subject details", "sample_metadata", "patient_key", "无来源明确subject连接，留空", "查阅Zenodo、完整归档、论文补充Table S1及作者GitHub sample处理；连接0/1"),
    (relative(RAW_METADATA / "paper.xml"), "Human subjects; Dewax and antigen retrieval", "sample_metadata", "sample_type;anatomical_site;collection_method;preservation_method;processing_method", "tissue;esophagus;resection;ffpe;endoscopic mucosal resection (EMR)", "FFPE EMR，非按ROI推断疾病分期"),
    (relative(RAW_METADATA / "paper.xml"), "Human subjects: anonymized age and gender", "sample_metadata", "age_at_collection_years;sex_reported", "来源明确不可提供，留空", "不填队列统计作为个体值"),
    (relative(RAW_METADATA / "supplementary_files.zip") + "#mmc1.pdf", "Supplementary Table S1: Qdot800 112/114 EPR3653 CD14", "panel_mapping", "source_name;keep_channel;output_channel_index", "仅保留112Cd单一CD14表达测量，排除114Cd重复；两DNA按S1核输入保留", "两Cd为同一抗体同一Qdot的同位素读出；不声称作者偏好哪条Cd通道"),
    (relative(META / "source_ordered_channel_map.json"), "filename protein_marker suffix; numeric isotope order", "panel_mapping", "source_channel_index;s1_original_marker", "由47原始文件标签蛋白后缀对接本study S1；source与output位置分别记录", "顺序已与采集txt header核对；不是IFD堆叠推断"),
    (relative(RAW_METADATA / "IF_IMC_ROI_registration.m"), "IMC_scan = 1.0", "image_metadata", "pixel_size_x_um;pixel_size_y_um", "1.0um/pixel；TIFF ResolutionUnit=inch且X/YResolution=25400交叉支持", "保持采集朝向，不执行来源注册旋转或插值"),
    (relative(OUTPUT / "images.parquet"), "slide_id;roi_id;region_annotation", "image_metadata", "sample_id;slide_id;roi_id;region_annotation", "沿用当前正式处理ROI身份；病理区无直接ROI对应则留空", "area mask为整片病理区域，未擅自赋予单ROI病理标签"),
]:
    row = {key: "" for key in fields.columns}
    row.update(study=STUDY, source_file=source, source_field=source_field, target_table=table,
               target_field=target, join_keys="archive member_root / specimen label / ROI identifier",
               transformation_rule=rule, notes=note)
    field_rows.append(row)
fields = pd.DataFrame(field_rows, columns=fields.columns)
image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
excluded = json.loads((CACHE / "excluded_images.json").read_text()) if (CACHE / "excluded_images.json").exists() else {}
source_evidence = dict(
    study=STUDY, source_record_url="https://zenodo.org/records/7576005", version="1.0",
    paper_url="https://doi.org/10.1016/j.crmeth.2023.100595", author_repository="https://github.com/enkim/IMC-IF",
    supplementary_url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10626190/supplementaryFiles",
    source_manifest=relative(META / "source_manifest.json"),
    processed_scope=relative(OUTPUT / "images.parquet"), accepted_version=SOURCE_REVISION,
    representative_overview=relative(CACHE / "representative_overviews.png"),
    preparer_overview_assessment=relative(CACHE / "overview_assessment.json"),
    source_scope=dict(native_rois=eligible, native_channel_files=sum(len(paths) for paths in source_paths_by_roi.values()),
                      uncompressed_square_bytes=sum(max(shape)**2*38*4 for shape in shape_by_roi.values())),
    sample_identity=dict(source_sample_id=specimen, sample_rows=1, image_linkable=35, patient_linked_samples=0,
                         evidence=([relative(OUTPUT / "images.parquet"), relative(EVIDENCE / "acquisition_header.txt")]
                                   + [relative(RAW_METADATA / name) for name in ["paper.xml", "author_05_Feature extraction_merge.ipynb", "author_06_area masked dataframe select.ipynb", "github_tree.json"]]
                                   + [relative(RAW_METADATA / "supplementary_files.zip") + "#mmc1.pdf"])),
    duplicate_selection="Table S1 confirms same Qdot800/EPR3653 CD14 at 112/114; keep native112, drop duplicate114",
    controller_requests=["完成剩余ROI正式导出后安排独立复核；M01患者连接缺口须保留", "共享readers.py的连续未压缩TIFF有界memmap适配由主控协调"],
)
if args.action != "overview":
    save_json(CACHE / "source_evidence.json", source_evidence)
    publish(review=args.action == "prepare")
    progress("metadata_prepared")


# A read-slot action permits source inspection while the formal write slot is busy.
if args.action == "overview":
    overview_rows = []
    for roi in ("ROI001", "ROI002", "ROI008", "ROI024", "ROI035"):
        paths = tuple(NAS / path for path in source_paths_by_roi[roi])
        source = SourceSpec(roi, PANEL_ID, "tiff_planes", paths[0], (0,)*47, paths=paths)
        with ExitStack() as stack:
            overview_rows.append((roi, overview(NativePlanes(source, stack))))
    save_overviews(overview_rows)
    progress("source_overview_ready_waiting_write_slot")


# Export only new or source-changed images, and publish a resumable index per image.
if args.action in {"export", "verify"}:
    estimate = source_evidence["source_scope"]["uncompressed_square_bytes"]
    if args.action == "export":
        assert shutil.disk_usage(NAS).free - estimate >= 100*1024**3
        assert shutil.disk_usage("/tmp").free >= 64*1024**3
    overview_rows = []
    for roi in eligible:
        source_paths = tuple(NAS / path for path in source_paths_by_roi[roi])
        source = SourceSpec(roi, PANEL_ID, "tiff_planes", source_paths[0], (0,)*47,
                            paths=source_paths, pixel_size_um=(1.0, 1.0))
        destination = OUTPUT / "images" / roi / "input.tiff"
        receipt_path = CACHE / "image_receipts" / f"{roi}.json"
        binding = dict(source_revision=SOURCE_REVISION,
                       source_files=[relative(p) for p in source_paths],
                       retained_source_channels=retained.source_channel_index.tolist(),
                       output_names=retained.marker_id.tolist())
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        reuse = (receipt is not None and receipt["source_binding"] == binding and
                 destination.exists() and receipt["output_stat"] == identity(destination))
        if (reuse and roi in set(images.image_id) and
                receipt.get("self_check", {}).get("header_pass") and
                receipt["self_check"].get("all_padding_zero")):
            continue
        progress("checking_existing" if reuse else "exporting", image_id=roi)
        with ExitStack() as stack:
            reader = NativePlanes(source, stack)
            _, h, w = reader.shape
            assert [h, w] == shape_by_roi[roi]
            with tifffile.TiffFile(source_paths[0]) as tif:
                tags = tif.pages[0].tags
                assert int(tags["ResolutionUnit"].value) == 2
                for key in ("XResolution", "YResolution"):
                    num, den = tags[key].value
                    assert 25400*den/num == 1.0
            if args.action == "verify":
                assert reuse, f"{roi}: missing/current input receipt required before verification"
            if reuse:
                export = receipt["export"]
            else:
                export = export_input(reader, panel, destination)
                if not export["admitted"]:
                    excluded[roi] = dict(reason="all admitted markers zero over the entire acquisition", export=export)
                    save_json(CACHE / "excluded_images.json", excluded)
                    images = images[images.image_id != roi]
                    publish(review=False)
                    progress("all_zero_image_excluded", image_id=roi)
                    continue
            with formal_read_lease(source_paths[0]):
                check = check_image(destination, reader)
                if roi in {"ROI001", "ROI002", "ROI008", "ROI024", "ROI035"}:
                    overview_rows.append((roi, overview(reader)))
        save_json(receipt_path, dict(image_id=roi, source_binding=binding,
                                    output_stat=identity(destination), export=export, self_check=check))
        row = {key: pd.NA for key in image_columns}
        slide_label = slide_labels[roi]
        row.update(study=STUDY, image_id=roi, sample_id=sample_id, panel_id=PANEL_ID,
                   image_kind="roi", source_image_id=f"{manifest['member_root']}/{roi}",
                   source_files=[relative(p) for p in source_paths], source_format="tiff_planes",
                   source_axes="YX (47 independent planes)", source_dtype="float32",
                   source_revision=SOURCE_REVISION,
                   slide_id=slide_label, roi_id=roi, source_x0_px=0, source_y0_px=0,
                   source_x1_px=w, source_y1_px=h, pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                   input_image_path=relative(destination), intensity_description="Fluidigm MCD Viewer native float32 IMC intensities; no normalization or resampling",
                   split_method="already_independent", split_evidence=relative(CACHE / "source_evidence.json"),
                   metadata_notes="35原始采集ROI共享同一来源EMR标本；患者连接及逐ROI病理注释未知；仅整理者自检，独立复核待执行")
        row.update({key: export[key] for key in ["pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "height_px", "width_px", "source_channel_count", "channel_count"]})
        images = pd.concat([images[images.image_id != roi], pd.DataFrame([row])], ignore_index=True).sort_values("image_id")
        publish(review=False)
        progress("image_completed", image_id=roi, reused=reuse)
        print(json.dumps(dict(image_id=roi, exported_images=len(images), reused=reuse), ensure_ascii=False), flush=True)
    if not (CACHE / "representative_overviews.png").exists():
        save_overviews(overview_rows)
    publish()
    progress("prepared_awaiting_independent_review", overview=relative(CACHE / "representative_overviews.png"))
