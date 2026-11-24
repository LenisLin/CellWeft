"""Deliver the 16 complete CellSighter melanoma LN fields as native input TIFFs.

Inputs: Mendeley zcz8743fcv v2 NPZ images, channels.txt, existing native
verification, the study-specific S1 rows, and cached public metadata evidence.
Outputs: independent CYX input.tiff, complete input tables and preparer checks.
NPZ support is study-local pending migration into the shared source reader.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "4"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Amitay2023_MIBI_Melanoma"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "mendeley_zcz8743fcv_v2"
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
NATIVE = META / "native_verification.json"
PRIOR_REVIEW = NAS / "processed/_shared/paper/segmentation_channel_review/amitay_dsdna_recheck"
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0005_prepare_0/progress.json"
GAPS = [
    "16个来源image ID尚无可验证的生物学取样及患者连接；sample表0行，M01 unresolved。",
    "像素尺度未确认：发布页/Methods为0.8 mm视野，2048像素；Methods另述0.5 µm/pixel，未据此推算。",
    "当前25通道分类发布不含dsDNA；已有作者mask保留，类型/配准在后续阶段核验。",
    "整理者自检已记录；独立CLI复核待主控组织。",
]


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress(phase: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, expected_images=16,
                             completed_images=completed, gaps=GAPS,
                             updated_at=datetime.now(timezone.utc).isoformat(), **details))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


class NpyFieldReader:
    """Window the release's disk-mapped YXC data.npy without loading all channels.

    This release contains dense, already cleaned image arrays. Zeros are source
    measurements/background; no acquisition validity or tissue mask is inferred.
    """

    def __init__(self, array: np.memmap):
        self.array = array
        self.shape = (array.shape[2], array.shape[0], array.shape[1])
        self.dtype = array.dtype
        self.source = SimpleNamespace(pixel_size_um=None)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.stack([self.array[y0:y1, x0:x1, channel] for channel in channels])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def check_tiff(path: Path, panel: pd.DataFrame) -> None:
    """Check every delivered header, including standardized channel ordering."""
    names = panel.loc[panel.keep_channel, "marker_id"].tolist()
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (24, 2048, 2048)
        assert tif.series[0].axes == "CYX"
        assert len(tif.pages) == 24
        assert all(page.shape == (2048, 2048) and page.dtype == np.dtype("float64") for page in tif.pages)
        xml = ElementTree.fromstring(tif.ome_metadata)
        assert [element.attrib["Name"] for element in xml.iter() if element.tag.endswith("}Channel")] == names


def extract_field(path: Path, temporary_root: Path) -> Path:
    """Extract exactly one NPY to local disk; ZIP CRC is checked during extraction."""
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["data.npy"]
        info = archive.getinfo("data.npy")
        if shutil.disk_usage(temporary_root).free < info.file_size + 64 * 1024**3:
            raise RuntimeError("Local NPY extraction would violate the 64 GiB reserve")
        destination = temporary_root / f"{path.stem}.npy"
        if destination.exists() and destination.stat().st_size == info.file_size:
            return destination
        temporary = destination.with_suffix(".npy.partial")
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024**2)
        assert temporary.stat().st_size == info.file_size
        temporary.replace(destination)
        return destination


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image-id", action="append", help="Source numeric ID to process first; repeatable")
parser.add_argument("--selected-only", action="store_true", help="Stop after the explicitly requested IDs")
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--overview-observation", help="Preparer observation, supplied only after viewing the contact sheet")
parser.add_argument("--temporary-root", type=Path, default=Path("/tmp/study_input_batch") / STUDY)
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
args.temporary_root.mkdir(parents=True, exist_ok=True)

# Reuse the original native proof and reconcile the complete canonical release.
verification = json.loads(NATIVE.read_text())
manifest = json.loads((META / "source_images_manifest.json").read_text())
source_names = (RAW / "channels.txt").read_text().splitlines()
assert len(verification) == len(manifest) == 16
assert {row["filename"] for row in manifest} == {f"{i}.npz" for i in range(1, 17)}
assert all(row["complete"] and row["shape"] == [2048, 2048, 25] and row["dtype"] == "float64" for row in verification)
assert all([channel["source_marker"] for channel in row["channel_regions"]] == source_names for row in verification)
for row in verification:
    path = NAS / row["path"]
    assert path.stat().st_size == row["source_size"] == row["local_size"]
scope = dict(canonical_study=STUDY, release="10.17632/zcz8743fcv.2",
             complete_images=[dict(source_image_id=row["acquisition_id"], path=row["path"],
                                   source_file_id=row["source_file_id"], size=row["source_size"])
                              for row in verification],
             native_verification_reused=relative(NATIVE), hashes_recomputed=False,
             not_admitted=[dict(path=relative(RAW / "data/images/1.npz"),
                                reason="旧未完成下载，573440字节、无ZIP中心目录；对应正式images/1.npz已完整。保留原件，不另计image。")],
             masks="16个来源NPZ mask保留于raw，当前阶段不转换或验收。",
             capacity=dict(maximum_uncompressed_outputs_bytes=16*24*2048*2048*8,
                           one_local_npy_bytes=838860928, nas_reserve_bytes=100*1024**3,
                           local_reserve_bytes=64*1024**3))
write_json(CACHE / "source_scope.json", scope)
panel_id = STUDY + "__mendeley_v2_25"
panel = panel_from_s1(STUDY, panel_id, source_names, S1)
assert len(panel) == 25 and int(panel.keep_channel.sum()) == 24
assert panel.loc[~panel.keep_channel, "source_name"].tolist() == ["MECA-79"]

# Evidence chain: local source -> official release and supplement -> author repo.
# Cached article attachments are figure/source-data tables, not sample identity maps.
metadata_cache = META / "source_metadata"
metadata_evidence = dict(
    sources=[
        dict(path=relative(META), result="图像和mask来源manifest、channels；仅有image ID，无标本/患者映射"),
        dict(url="https://data.mendeley.com/datasets/zcz8743fcv/2", path=relative(metadata_cache / "mendeley_landing.html"),
             result="发布范围为多个患者的16个独立视野；四层目录无sample/core/clinical map"),
        dict(url="https://www.nature.com/articles/s41467-023-40066-7", path=relative(metadata_cache / "paper_html.html"),
             result="Methods区分16个labeled视野及33个unlabeled视野；本地仅16个正式发布视野。不提供image至标本连接"),
        dict(path=relative(metadata_cache / "prior_supplement.txt"), result="补充图与图例；未发现临床表、标本/患者或image ID连接"),
        dict(path=relative(metadata_cache / "source_data_inventory.json"), result="已查看Source Data目录及所有表头，小表完整解码；fig2_B仅查看大数值矩阵表头/前缀。未找到标本/患者桥接列；具体覆盖见inventory"),
        dict(url="https://github.com/KerenLab/CellSighter", path=relative(metadata_cache / "github_tree.json"),
             result="完整main文件树仅代码、分类示例与模型配置；无melanoma sample/core map"),
    ], join_keys="source image basename / Mendeley file UUID；未找到至生物学取样的桥接键",
    image_count=16, sample_rows=0, sample_linked_images=0, patient_linked_samples=0,
    gap=GAPS[0], no_roi_as_sample=True)
write_json(CACHE / "sample_investigation.json", metadata_evidence)
samples = pd.DataFrame(columns=columns("sample_metadata"))
field_rows = []


def field(source: str, source_field: str, table: str, target: str, rule: str,
          notes: str = "", join_keys: str = "") -> None:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=source, source_field=source_field,
               target_table=table, target_field=target, join_keys=join_keys,
               transformation_rule=rule, conflict_rule="保留来源；未知不推断", notes=notes)
    field_rows.append(row)


field(relative(META / "source_images_manifest.json"), "filename;id", "image_metadata", "image_id",
      "STUDY + '__' + NPZ basename；source_image_id保存官方data/images/<id>.npz:data.npy对象身份",
      "同一公开来源UUID只发布一份；旧data/images/1.npz不完整副本不计数", "filename == local NPZ basename")
field(relative(CACHE / "sample_investigation.json"), "source image basename; source_file_id", "image_metadata", "sample_id",
      "已查本地metadata、官方发布页/全目录、论文/补充附件及作者完整仓库；无桥接键，16/16留空",
      "sample_rows=0; sample_linked_images=0/16; patient_linked_samples=0; M01 unresolved")
field(relative(CACHE / "sample_investigation.json"), "biological specimen and subject identity", "sample_metadata", "patient_key",
      "无来源标本身份，暂不创建sample行；未知patient_key留空", "不能将ROI编号作为sample或患者")
field(relative(RAW / "channels.txt"), "line order", "panel_mapping", "source_channel_index",
      "按25个来源行从0编号；以Study+Original marker精确连接唯一S1，未使用跨study别名")
field(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "output_channel_index",
      "保留24个准入marker，按来源顺序从0连续编号；MECA-79按S1排除；5个body候选全部保留",
      "分类发布无核输入，未伪造核通道；作者mask另阶段验证")
for target in ("source_files", "source_format", "source_axes", "source_dtype", "source_revision", "fov_id"):
    field(relative(NATIVE), "path;npz_key;axes;dtype;acquisition_id", "image_metadata", target,
          "复用16个原生核验；格式npz，YXC float64，Mendeley v2，FOV沿用basename")
for target in ("source_x0_px", "source_y0_px", "source_x1_px", "source_y1_px", "height_px", "width_px",
               "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "split_method", "region_annotation"):
    field(relative(NATIVE), "shape;acquisition_id", "image_metadata", target,
          "保留完整[0,2048)×[0,2048)视野，原生朝向；无需补边，already_independent；无来源区域注释则空")
for target in ("pixel_size_x_um", "pixel_size_y_um"):
    field(relative(metadata_cache / "paper_html.html"), "Methods: field extent; image resolution", "image_metadata", target,
          "0.8mm视野/2048像素与另述0.5µm/pixel不一致，留空等待来源像素尺度证据")
field(relative(metadata_cache / "mendeley_landing.html"), "Description: clean images", "image_metadata", "intensity_description",
      "保留发布方已清理的float64像素值；不追加强度变换；具体来源清理步骤未随文件附带")
for target, rule in {
    "study": "采用任务确认的canonical study ID，与正式来源目录一致",
    "panel_id": "同一25行有序panel标识为STUDY__mendeley_v2_25",
    "image_kind": "官方按image ID发布独立MIBI视野，填写fov",
    "source_image_id": "官方data/images/<numeric ID>.npz:data.npy；UUID同时保留于metadata_notes与export_receipt",
    "source_acquisition_id": "NPZ不提供MCD acquisition字段，留空",
    "source_z_index": "来源只有YXC三轴，无Z层选择，留空",
    "source_t_index": "来源只有YXC三轴，无时间层选择，留空",
    "slide_id": "来源未提供slide对应，留空",
    "core_id": "来源为独立FOV，未提供core身份，留空",
    "roi_id": "沿用明确的FOV身份；无额外来源ROI注释，留空",
    "source_channel_count": "复用原生核验NPY末轴长度25，与channels.txt行数一致",
    "channel_count": "本study S1选择后保留24通道，与TIFF头逐图核对",
    "input_image_path": "processed/STUDY/images/<image_id>/input.tiff，相对NAS根目录",
    "valid_pixels_path": "来源为完整密集NPZ网格，无来源记录层级缺测证据，未另造实测位置掩码；留空",
    "split_evidence": "引用本study source_scope.json所记录的官方独立来源及准入范围",
    "metadata_notes": "逐图保留来源UUID以及sample连接、尺度、核输入的实际缺口",
}.items():
    field(relative(CACHE / "source_scope.json"), "canonical release; native verification; S1 selection",
          "image_metadata", target, rule)
fields = pd.DataFrame(field_rows, columns=columns("field_mapping"))
assert set(columns("image_metadata")) <= set(fields.loc[fields.target_table == "image_metadata", "target_field"])

images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
records = images.to_dict("records")
receipt_path = CACHE / "export_receipt.json"
receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
if args.overview_observation:
    assert (CACHE / "image_overview.png").exists()
    write_json(CACHE / "overview_observation.json", dict(preparer="Codex CLI study preparer",
               independent_reviewer=None, observation=args.overview_observation,
               figure=relative(CACHE / "image_overview.png")))


def publish(phase: str) -> None:
    frame = pd.DataFrame(records, columns=columns("image_metadata"))
    save_input_tables(OUTPUT, frame, samples, panel, fields)
    write_json(receipt_path, receipts)
    observation = CACHE / "overview_observation.json"
    rows = []
    details = {
        "S01": ("pass", "16个官方完整来源；旧未完成副本不纳入。所有已导出image均有全图准入marker非零证据。"),
        "I01": ("pass", "已导出索引主键、源路径、来源image UUID及独立TIFF可定位。"),
        "I02": ("pass", "panel外键有效；sample_id全空，连接缺口详见M01。"),
        "I03": ("pass", "每图完整2048×2048来源框，四侧padding=0；像素尺度尚无一致来源证据。"),
        "I04": ("pass" if observation.exists() else "unresolved", "来源为独立FOV，无core拆分；逐图概览见overview_observation.json。" if observation.exists() else "概览待整理者查看；无来源sample identity map。"),
        "P01": ("pass", "25个来源通道按原始channels.txt逐行精确连接本study S1。"),
        "P02": ("pass", "保留24个准入marker；排除MECA-79，保留全部5个S1 body候选；无来源核通道。"),
        "P03": ("pass", "每张已交付TIFF重开核对CYX、24×2048×2048、float64与全部标准名称顺序。"),
        "I05": ("pass", "每张已导出图像全部24通道抽查两个32×32块，来源磁盘映射与输出逐值相同；坐标见export_receipt.json。"),
        "M01": ("unresolved", GAPS[0] + "追查来源、键与覆盖见sample_investigation.json。"),
        "M02": ("not_applicable", "无可验证sample身份；sample表0行，无临床值转换。"),
        "D01": ("unresolved", f"当前交付{len(records)}/16张；sample/patient连接未完成，独立复核待主控安排。"),
    }
    for check_id, (result, detail) in details.items():
        row = dict.fromkeys(columns("review_results"), "")
        row.update(study=STUDY, preparer="Codex CLI study preparer", scope="study",
                   record_id=STUDY, check_id=check_id, result=result,
                   source_evidence=relative(CACHE / ("sample_investigation.json" if check_id.startswith("M") else "source_scope.json")),
                   output_evidence=relative(OUTPUT), details="整理者自检；独立reviewer留空。" + detail)
        rows.append(row)
    review_path = OUTPUT / "review_results.csv"
    if review_path.exists():
        previous = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        independent = previous[previous.reviewer.ne("")]
        review = pd.concat([pd.DataFrame(rows), independent], ignore_index=True)
    else:
        review = pd.DataFrame(rows)
    with formal_publish_lease(review_path, directory=review_path.parent):
        review.to_csv(review_path, index=False, encoding="utf-8-sig")
    progress(phase, len(records), sample_rows=0, sample_linked_images=0, patient_linked_samples=0,
             image_ids=[row["image_id"] for row in records], output_root=relative(OUTPUT))


publish(("inputs_prepared_pending_metadata_and_independent_review" if len(records) == 16 else "metadata_ready")
        if args.metadata_only else "exporting")
requested = set(args.image_id or [])
assert requested <= {str(i) for i in range(1, 17)}
selected = requested if args.selected_only else {str(i) for i in range(1, 17)}
assert selected
if not args.metadata_only:
    for native in sorted(verification, key=lambda row: (row["acquisition_id"] not in requested, int(row["acquisition_id"]))):
        source_id = native["acquisition_id"]
        if source_id not in selected:
            continue
        image_id = STUDY + "__" + source_id
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if any(row["image_id"] == image_id for row in records):
            check_tiff(destination, panel)
            assert image_id in receipts and receipts[image_id]["admitted"]
            assert receipts[image_id]["source_size"] == (NAS / native["path"]).stat().st_size
            assert receipts[image_id]["source_mtime_ns"] == (NAS / native["path"]).stat().st_mtime_ns
            print(json.dumps(dict(image_id=image_id, status="reused")), flush=True)
            continue
        progress("extracting_npz", len(records), current_image=image_id)
        npy_path = extract_field(NAS / native["path"], args.temporary_root)
        array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        assert array.shape == (2048, 2048, 25) and array.dtype == np.dtype("float64")
        try:
            reader = NpyFieldReader(array)
            progress("writing_tiff", len(records), current_image=image_id)
            result = export_input(reader, panel, destination)
            result["source_file_id"] = native["source_file_id"]
            result["source_path"] = native["path"]
            result["source_size"] = native["source_size"]
            result["source_mtime_ns"] = (NAS / native["path"]).stat().st_mtime_ns
            if result["admitted"]:
                check_tiff(destination, panel)
                checks = []
                signal_window = None
                for channel in panel.loc[panel.keep_channel, "source_channel_index"]:
                    ys, xs = np.nonzero(array[32:-32, 32:-32, channel])
                    if len(xs):
                        x, y = int(xs[len(xs)//2])+32, int(ys[len(ys)//2])+32
                        signal_window = (x-16, y-16, x+16, y+16)
                        break
                assert signal_window is not None, "No interior marker signal for the required pixel self-check"
                with tifffile.TiffFile(destination) as tif:
                    for window in (signal_window, (0, 0, 32, 32)):
                        for row in panel[panel.keep_channel].itertuples(index=False):
                            expected = reader.read_window([row.source_channel_index], window)[0]
                            actual = read_tiff_window(tif, int(row.output_channel_index), window)
                            assert expected.dtype == actual.dtype and np.array_equal(expected, actual)
                        checks.append(dict(window=list(window), compared_channels=24, equal=True))
                result["pixel_checks"] = checks
                record = dict.fromkeys(columns("image_metadata"), None)
                record.update(study=STUDY, image_id=image_id, panel_id=panel_id, image_kind="fov",
                              source_image_id=f"data/images/{source_id}.npz:data.npy",
                              source_files=[native["path"]], source_format="npz", source_axes="YXC",
                              source_dtype="float64", source_revision="Mendeley zcz8743fcv v2 (10.17632/zcz8743fcv.2)",
                              fov_id=source_id, source_x0_px=0, source_y0_px=0,
                              source_x1_px=2048, source_y1_px=2048,
                              pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                              height_px=2048, width_px=2048, source_channel_count=25, channel_count=24,
                              input_image_path=relative(destination),
                              intensity_description="Native float64 values of publisher-cleaned protein images; source cleaning details unspecified; no additional intensity transformation",
                              split_method="already_independent", split_evidence=relative(CACHE / "source_scope.json"),
                              metadata_notes="来源UUID=" + native["source_file_id"] + "; sample/patient连接未知；来源尺度描述不一致，pixel_size留空；发布panel无核输入，作者mask留待后续验证")
                records.append(record)
                # Display-only percentiles never affect the saved native pixels.
                rgb = np.stack([array[::4, ::4, channel] for channel in (17, 19, 2)], axis=-1).copy()
                for channel in range(3):
                    positive = rgb[:, :, channel][rgb[:, :, channel] > 0]
                    scale = np.percentile(positive, 99) if len(positive) else 1
                    rgb[:, :, channel] = np.clip(rgb[:, :, channel] / scale, 0, 1)
                plt.imsave(CACHE / f"overview_{source_id}.png", rgb)
            receipts[image_id] = result
            publish("exporting")
            print(json.dumps(dict(image_id=image_id, admitted=result["admitted"], completed_images=len(records))), flush=True)
        finally:
            array._mmap.close()
            npy_path.unlink()
    overview_ids = [str(i) for i in range(1, 17) if (CACHE / f"overview_{i}.png").exists()]
    if overview_ids:
        figure, axes = plt.subplots(4, 4, figsize=(14, 14))
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, source_id in zip(axes.flat, overview_ids):
            axis.imshow(plt.imread(CACHE / f"overview_{source_id}.png"))
            axis.set_title("Source FOV " + source_id, fontsize=10)
        figure.suptitle("Amitay v2 full fields: MelanA red / CD45 green / CD20 blue (display only)")
        figure.tight_layout()
        figure.savefig(CACHE / "image_overview.png", dpi=120)
        plt.close(figure)
    publish("inputs_prepared_pending_metadata_and_independent_review")

print(json.dumps(dict(study=STUDY, expected_images=16, exported_images=len(records),
                     sample_rows=0, sample_linked_images=0, patient_linked_samples=0,
                     output=relative(OUTPUT)), ensure_ascii=False), flush=True)
