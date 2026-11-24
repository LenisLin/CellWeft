"""Prepare Cai's 157 accepted original Hyperion TXT ROIs without normalization.

Inputs: three original 7z archives, accepted native proofs, public specimen map,
paper/supplement and the exact study rows of S1. Outputs: native float32 CYX
input.tiff, complete image/sample/panel/field tables and preparer check records.
The libarchive/TXT window adapter is study-local pending shared-reader migration.
Run --export through the batch shared formal write lease wrapper. Metadata/verification
modes do not re-export completed images. No patient identity is inferred from IDs.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import ctypes
import ctypes.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Cai2026_IMC_HER2_BreastCancer"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
SCRATCH = Path("/tmp/study_input_batch") / STUDY
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0019_prepare_0/progress.json"
PANEL_ID = "Hyperion_detector_44"
PREPARER = "0019_prepare_0"
GAPS = [
    "公开sample_Group.csv的sample_id逐ROI唯一；尚缺多个ROI到生物学取样的明确归并及patient map，不从编号前缀推断。",
    "TXT未提供可核实的物理像素尺度，pixel_size字段留空。",
    "独立复核尚待另一CLI上下文执行；整理者不写D01通过绑定。",
]
CONTROLLER_REQUESTS = [
    "共享读取器缺少原生IMC TXT及7z流式入口；本study按readimc的float32/XY语义临时适配，后续可迁入src。",
    "请独立复核157张原生ROI及S1；尤其核对sample_Group.csv中逐ROI sample_id的生物学层级，缺乏来源依据时保持患者连接为空。",
    "请重点复核pCR-34及其他终末全零占位记录：仅在完整前缀Z连续时省略重复空记录，保留真实原点；TXT只给出实测坐标，未声明MCD计划扫描框。",
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


def progress(phase: str, count: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, expected_images=157,
                             completed_images=count, gaps=GAPS,
                             updated_at=datetime.now(timezone.utc).isoformat(), **details))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


class Archive:
    """Single-pass libarchive 7z extraction, with one bounded member on local disk."""

    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(ctypes.util.find_library("archive"))
        declarations = {
            "archive_read_new": (ctypes.c_void_p, []),
            "archive_read_support_format_7zip": (ctypes.c_int, [ctypes.c_void_p]),
            "archive_read_support_filter_all": (ctypes.c_int, [ctypes.c_void_p]),
            "archive_read_open_filename": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]),
            "archive_read_next_header": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
            "archive_read_data": (ctypes.c_ssize_t, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]),
            "archive_read_data_skip": (ctypes.c_int, [ctypes.c_void_p]),
            "archive_read_free": (ctypes.c_int, [ctypes.c_void_p]),
            "archive_entry_pathname": (ctypes.c_char_p, [ctypes.c_void_p]),
            "archive_entry_size": (ctypes.c_int64, [ctypes.c_void_p]),
            "archive_error_string": (ctypes.c_char_p, [ctypes.c_void_p]),
        }
        for name, (result, arguments) in declarations.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, arguments

    def __enter__(self):
        self.handle = self.lib.archive_read_new()
        self.lib.archive_read_support_format_7zip(self.handle)
        self.lib.archive_read_support_filter_all(self.handle)
        if self.lib.archive_read_open_filename(self.handle, os.fsencode(self.path), 1024*1024) != 0:
            self.__exit__(None, None, None)
            raise OSError(f"Cannot open original 7z: {self.path.name}")
        return self

    def __exit__(self, *exc):
        self.lib.archive_read_free(self.handle)

    def __iter__(self):
        entry = ctypes.c_void_p()
        while True:
            status = self.lib.archive_read_next_header(self.handle, ctypes.byref(entry))
            if status == 1:
                break
            if status != 0:
                raise OSError(self.lib.archive_error_string(self.handle).decode())
            yield self.lib.archive_entry_pathname(entry).decode(), int(self.lib.archive_entry_size(entry))

    def skip(self) -> None:
        if self.lib.archive_read_data_skip(self.handle) < 0:
            raise OSError(self.lib.archive_error_string(self.handle).decode())

    def extract(self, destination: Path, size: int) -> None:
        if shutil.disk_usage(SCRATCH).free < size*3 + 64*1024**3:
            raise RuntimeError("TXT scratch would violate the 64 GiB local reserve")
        temporary = destination.with_suffix(".partial")
        buffer = ctypes.create_string_buffer(1024*1024)
        count = 0
        with temporary.open("wb") as stream:
            while True:
                length = self.lib.archive_read_data(self.handle, buffer, len(buffer))
                if length < 0:
                    raise OSError(self.lib.archive_error_string(self.handle).decode())
                if not length:
                    break
                stream.write(buffer.raw[:length])
                count += length
        if count != size:
            raise ValueError(f"Incomplete archive member: {count}/{size}")
        temporary.replace(destination)


class TxtReader:
    """Disk-mapped native float32 records; coordinates define measured validity.

    The complete source table has 50 columns, including six acquisition fields.
    Source channels are the 44 detector columns, starting at table column six.
    Geometric dimensions follow readimc.TXTFile: max(X/Y)+1 with no crop or flip.
    """

    def __init__(self, records: np.memmap):
        # pCR-34 revealed an instrument footer: 500 entirely zero records reset
        # X/Y/Z to zero after a sequential scan prefix. These are not measured
        # zero-intensity pixels: they duplicate the acquired origin and reset Z.
        # Keep the real origin and every measured zero; resolve only this exact
        # footer pattern, requiring the complete prefix Z sequence as evidence.
        measured_end = len(records)
        end = len(records)
        while end > 1:
            start = max(1, end-65536)
            nonzero = np.flatnonzero(np.any(records[start:end] != 0, axis=1))
            if len(nonzero):
                measured_end = start + int(nonzero[-1]) + 1
                break
            end = start
        self.zero_footer_records = len(records)-measured_end
        if self.zero_footer_records:
            if not np.array_equal(records[:measured_end, 5], np.arange(measured_end, dtype=np.float32)):
                raise ValueError("Zero coordinate footer lacks a complete sequential source Z prefix")
            records = records[:measured_end]
        xy = records[:, 3:5]
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy != np.floor(xy)):
            raise ValueError("Invalid native TXT XY coordinates")
        width, height = xy.max(axis=0).astype(int) + 1
        self.index = np.full((height, width), -1, dtype=np.int64)
        for start in range(0, len(records), 65536):
            end = min(start+65536, len(records))
            x, y = records[start:end, 3:5].astype(np.int64).T
            self.index[y, x] = np.arange(start, end)
        if np.count_nonzero(self.index >= 0) != len(records):
            raise ValueError("Duplicate native XY records require a source decision")
        self.records = records
        self.shape = (44, int(height), int(width))
        self.dtype = np.dtype("float32")
        self.source = SimpleNamespace(pixel_size_um=None)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        index = self.index[y0:y1, x0:x1]
        valid = index >= 0
        result = np.zeros((len(channels), y1-y0, x1-x0), dtype=self.dtype)
        for output, channel in enumerate(channels):
            result[output, valid] = self.records[index[valid], channel+6]
        return result

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.index[y0:y1, x0:x1] >= 0


def decode_txt(path: Path, labels: list[str], record_path: Path) -> np.memmap:
    """Parse the complete TXT in bounded chunks, matching readimc float32 decode."""
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
    if header != labels:
        raise ValueError(f"Source header differs from accepted native proof: {path.name}")
    count = 0
    with record_path.open("wb") as stream:
        for chunk in pd.read_csv(path, sep="\t", dtype=np.float32, chunksize=32768):
            if chunk.shape[1] != 50:
                raise ValueError("Incomplete native source columns")
            chunk.to_numpy(dtype=np.float32).tofile(stream)
            count += len(chunk)
    return np.memmap(record_path, mode="r", dtype=np.float32, shape=(count, 50))


def header_check(path: Path, row: dict, panel: pd.DataFrame) -> None:
    with tifffile.TiffFile(path) as tif:
        kept = panel[panel.keep_channel].sort_values("output_channel_index")
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (len(kept), int(row["height_px"]), int(row["width_px"]))
        assert len(tif.pages) == len(kept) and all(p.dtype == np.dtype("float32") for p in tif.pages)
        xml = ET.fromstring(tif.ome_metadata)
        names = [e.attrib["Name"] for e in xml.iter() if e.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()


def pixel_check(reader: TxtReader, path: Path, row: dict, panel: pd.DataFrame) -> list[dict]:
    """Check all retained channels at source interior/edge and all padding pixels."""
    _, height, width = reader.shape
    channels = panel.loc[panel.keep_channel, "source_channel_index"].tolist()
    regions = [(max(0, width//2-16), max(0, height//2-16), min(width, width//2+16), min(height, height//2+16)),
               (0, 0, min(32, width), min(32, height))]
    left, top = int(row["pad_left_px"]), int(row["pad_top_px"])
    checks = []
    with tifffile.TiffFile(path) as tif:
        for region in regions:
            x0, y0, x1, y1 = region
            target = (x0+left, y0+top, x1+left, y1+top)
            saved = np.stack([read_tiff_window(tif, c, target) for c in range(len(channels))])
            source = reader.read_window(channels, region)
            assert saved.dtype == source.dtype and np.array_equal(saved, source)
            checks.append(dict(source_window=region, output_window=target,
                               channels=len(channels), exact_equal=True))
        side = int(row["height_px"])
        padding = []
        if top: padding.append((0, 0, side, top))
        if left: padding.append((0, top, left, top+height))
        if row["pad_bottom_px"]: padding.append((0, top+height, side, side))
        if row["pad_right_px"]: padding.append((left+width, top, side, top+height))
        for region in padding:
            assert all(not np.any(read_tiff_window(tif, c, region)) for c in range(len(channels)))
    if row["valid_pixels_path"]:
        with tifffile.TiffFile(NAS / row["valid_pixels_path"]) as validity:
            saved_valid = validity.pages[0].asarray(maxworkers=2)
        expected = np.zeros((side, side), dtype=np.uint8)
        expected[top:top+height, left:left+width] = reader.valid_window((0, 0, width, height))
        assert saved_valid.dtype == np.dtype("uint8") and np.array_equal(saved_valid, expected)
        checks.append(dict(kind="valid_pixels", full_domain_exact_equal=True,
                           measured_pixels=int(expected.sum()), padding_zero=True))
    return checks


def field_mapping() -> pd.DataFrame:
    """Define every image/sample field and the actual source-map uncertainty."""
    rows = []
    for table in ["image_metadata", "sample_metadata", "panel_mapping"]:
        for field in columns(table):
            source = "native TXT / accepted native_verification / paper"
            rule = "Source-backed value; unreported optional values remain null, never defaulted"
            if table == "image_metadata":
                rule = {
                    "study": "Fixed canonical study Cai2026_IMC_HER2_BreastCancer",
                    "image_id": "Exact native archive member stem; unique within study",
                    "sample_id": "Null: release sample_id is unique per ROI and its biological hierarchy is unconfirmed; retain it separately as source_sample_id",
                    "panel_id": "Hyperion_detector_44; identical ordered detector labels verified for all 157 members",
                    "image_kind": "roi; accepted original non-TMA instrument acquisition",
                    "source_files": "NAS-relative original 7z container, not temporary extracted TXT",
                    "source_image_id": "Exact original 7z member name",
                    "source_format": "txt; tab-separated native instrument export stored inside original 7z",
                    "source_acquisition_id": "Null; no MCD acquisition ID in this public TXT release",
                    "source_z_index": "Null; Z is a sequential instrument record column, not a selected optical section",
                    "source_t_index": "Null; no temporal image axis",
                    "source_axes": "XY records; scalar detector channels in ordered source columns",
                    "source_revision": "Zenodo record 18041847 v1; reuse accepted original size/mtime source binding",
                    "slide_id": "Null; no explicit source slide ID",
                    "core_id": "Null; original non-TMA ROI",
                    "roi_id": "Exact native TXT member stem",
                    "fov_id": "Null; source identifies instrument ROI, not separate FOV",
                    "source_channel_count": "44 detector channels; six acquisition/coordinate table fields are not image channels",
                    "channel_count": "Number of panel_mapping.keep_channel=true; 32 after exact S1 selection",
                    "source_dtype": "float32 native TXT decode following readimc.TXTFile; no intensity transform",
                    "source_x0_px": "0; full native acquisition frame",
                    "source_y0_px": "0; full native acquisition frame",
                    "source_x1_px": "max(native X)+1 from complete numeric records",
                    "source_y1_px": "max(native Y)+1 from complete numeric records",
                    "region_annotation": "Exact public sample_Group.Group value",
                    "pad_left_px": "floor((max(H,W)-W)/2)",
                    "pad_right_px": "max(H,W)-W-pad_left_px",
                    "pad_top_px": "floor((max(H,W)-H)/2)",
                    "pad_bottom_px": "max(H,W)-H-pad_top_px",
                    "height_px": "max(native H,native W)",
                    "width_px": "max(native H,native W)",
                    "input_image_path": "NAS-relative processed/<study>/images/<native member stem>/input.tiff",
                    "valid_pixels_path": "Null if every location acquired; otherwise coordinate-derived uint8 measured-pixel TIFF with padding zero. Entirely zero terminal duplicate-origin placeholders are omitted only after verifying sequential prefix Z; real origin and measured zero pixels remain valid",
                    "intensity_description": "Native TXT detector values decoded as float32, consistent with readimc; no scaling or normalization",
                    "split_method": "already_independent; original non-TMA instrument ROI",
                    "split_evidence": "NAS-relative primary_scope_review.json confirming original non-TMA instrument ROI scope",
                    "metadata_notes": "Source-column count semantics, exact publisher identity extension and unresolved biological/sample map",
                    "pixel_size_x_um": "Null; TXT lacks documented physical calibration",
                    "pixel_size_y_um": "Null; TXT lacks documented physical calibration",
                }.get(field, rule)
            if table == "sample_metadata":
                source = "No confirmed biological sample row; sample_Group.csv identity hierarchy unresolved"
                rule = {
                    "sample_id": "No row created until the biological sampling identity is supported; do not substitute one row per ROI",
                    "source_sample_id": "Exact public source sample_id string including leading zeros",
                    "patient_key": "Null: source does not publish the ROI/specimen-to-patient crosswalk",
                    "source_subject_id": "Null: no source subject field",
                    "sex_reported": "No individual value emitted while biological sample identity is unresolved; paper states cohort 100% female",
                    "timepoint_label": "No sample value emitted; Results associate PCR/nonPCR-Pre with pretreatment and nonPCR-Post with post-treatment",
                    "collection_method": "No sample value emitted; Results describe pretreatment biopsies and post-treatment surgical specimens",
                    "matched_sample_id": "Null: paired cohort design does not identify individual specimen pairs",
                    "index_regimen": "No sample value emitted; study restricted to SHR-A1811 monotherapy arm",
                    "best_overall_response": "Null: pCR group is pathological outcome, not RECIST CR",
                }.get(field, rule)
            if table == "panel_mapping":
                source = "native TXT ordered columns -> exact Study+Original marker in S1"
                rule = "panel_from_s1; all source positions retained in mapping; selected detector pixels retain source order"
            rows.append(dict(study=STUDY, source_file=relative(META / "source_metadata/cai_zenodo_sample_Group.csv") if table == "sample_metadata" else relative(META / "native_verification.json"),
                             source_sheet_or_table=source, source_field=field, target_table=table,
                             target_field=field, join_keys="Exact Sample=TXT member stem; exact source sample_id; no prefix patient inference",
                             value_mapping="", transformation_rule=rule, time_origin_rule="No survival/time origin inferred",
                             conflict_rule="Original source values preserved; unresolved biological identity remains explicit",
                             notes="Preparer mapping; source sample_id biological hierarchy remains unresolved" if table == "sample_metadata" else ""))
    rows.append(dict(study=STUDY, source_file=relative(META / "source_metadata/cai_zenodo_sample_Group.csv"),
                     source_sheet_or_table="sample_Group.csv", source_field="sample_id",
                     target_table="image_metadata", target_field="source_sample_id",
                     join_keys="Sample = exact native TXT member stem", value_mapping="",
                     transformation_rule="Preserve the exact release ID; do not assert biological sample identity",
                     time_origin_rule="", conflict_rule="Biological sample grouping and patient identity unresolved",
                     notes=json.dumps(dict(data_type="string", definition="Exact publisher sample_id associated with this ROI; biological sampling hierarchy unconfirmed"))))
    rows.append(dict(study=STUDY, source_file=relative(META / "native_verification.json"),
                     source_sheet_or_table="Native TXT header", source_field="0-based native table column position",
                     target_table="panel_mapping", target_field="source_table_column_index",
                     join_keys="Exact detector label", value_mapping="",
                     transformation_rule="source_channel_index + 6; the first six table columns are acquisition/XYZ metadata",
                     time_origin_rule="", conflict_rule="Use detector order for source_channel_index; preserve table order separately",
                     notes=json.dumps(dict(data_type="integer", definition="Zero-based original TXT table column containing the detector measurement"))))
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def save_reviews(images: pd.DataFrame, samples: pd.DataFrame, inventory: list[dict], complete: bool) -> None:
    """Publish labelled preparer checks, without impersonating independent review."""
    records = []
    for check, result, details in [
        ("S01", "pass" if complete else "unresolved", f"本批已输出{len(images)}/157；原归档全部成员见archive_inventory.json；不重hash，复用原size/mtime绑定及准入证明。"),
        ("I01", "pass", "已输出image主键唯一，来源精确成员及原始7z可定位。"),
        ("I02", "pass", f"全部{len(images)}个image连接实际panel；sample_id为空，无悬挂外键；来源ID保存在source_sample_id扩展列，生物学归并另见M01。"),
        ("I03", "pass", "保留每个TXT全部实测坐标的max(X/Y)+1框，方形补边，逐图重开TIFF头。TXT未声明MCD计划框，不推断未公开尺寸；尾部占位判定见receipt，实测不足矩形处保存valid_pixels。"),
        ("I04", "unresolved", "独立非TMA ROI已确认；逐ROI概览见overview；生物学取样身份归并尚缺明确来源。"),
        ("P01", "pass", "44个检测通道按原顺序与该study S1原名精确匹配，无别名替换；另6个采集/XYZ字段不计为图像通道，原表列位置单独保留。"),
        ("P02", "pass", "S1核输入DNA1/DNA2及已准入marker保留；无重复downstream marker。"),
        ("P03", "pass", "全部已输出TIFF的CYX、float32、通道数量/OME名称/顺序逐图自检；见每图receipt。"),
        ("I05", "pass", "全部已输出图的全部保留通道，两处<=32x32块与原TXT解码逐值相等；补边全量零检查；存在valid_pixels时全域与来源坐标索引逐值相等；见receipts。"),
        ("M01", "unresolved", f"已查Zenodo全部4文件、sample_Group.csv、全文及出版方mmc1.pdf；补充S1为panel，S2为细胞系ChIP，无临床crosswalk；论文声明无原创代码，公开检索未定位作者sample/core map。sample_Group的157个sample_id逐ROI唯一，生物学层级未定义；sample表{len(samples)}行，image→生物学取样0，sample→patient0。47份治疗前/13份术后取样及47患者的设计已知，具体ROI归并及患者crosswalk缺失，未从名称前缀猜测。详见source_trace.json。"),
        ("M02", "unresolved", "生物学sample身份缺失，sample表为空；Group保留在image.region_annotation，精确来源ID保留在image.source_sample_id；未将组级属性冒充个体临床值。"),
        ("D01", "unresolved", f"整理者交付{len(images)}/157张及配套表；sample/patient缺口如M01；待独立CLI上下文复核和主控验收，无input_binding pass。"),
    ]:
        if images.empty and check not in {"M01", "M02", "D01"}:
            continue
        records.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample" if check.startswith("M") else "study",
                            record_id=STUDY, check_id=check, result=result,
                            source_evidence=relative(META / "native_verification.json") + ";" + relative(CACHE),
                            output_evidence=relative(OUTPUT),
                            details="整理者自检，非独立复核：" + details))
    frame = pd.DataFrame(records, columns=columns("review_results"))
    temporary = OUTPUT / "review_results.csv.tmp"
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(OUTPUT / "review_results.csv")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export", action="store_true")
parser.add_argument("--verify-only", action="store_true")
parser.add_argument("--inventory-only", action="store_true", help="Read archive headers under the shared formal read lease")
parser.add_argument("--overview-observation")
args = parser.parse_args()
for path in [CACHE, OUTPUT, SCRATCH, CACHE / "receipts", CACHE / "overviews"]:
    path.mkdir(parents=True, exist_ok=True)

# Reuse existing native proof and exact original container binding, without hashing.
verification = json.loads((META / "native_verification.json").read_text())
proofs = [r["native_proof"] for r in verification["records"]]
assert len(proofs) == 157 and all(p["complete"] and p["native_order_verified"] and not p["fixture"] for p in proofs)
labels = proofs[0]["channels"]
assert len(labels) == 50 and all(p["channels"] == labels for p in proofs)
for p in proofs:
    stat = Path(p["source_file_path"]).stat()
    assert stat.st_size == p["source_file_stat"]["size"] and stat.st_mtime_ns == p["source_file_stat"]["mtime_ns"]
if args.inventory_only:
    inventory = []
    for archive_path in dict.fromkeys(p["source_file_path"] for p in proofs):
        selected = {p["source_member"]: p for p in proofs if p["source_file_path"] == archive_path}
        seen = set()
        with Archive(Path(archive_path)) as archive:
            for member, size in archive:
                inventory.append(dict(archive=relative(Path(archive_path)), member=member,
                                      size=size, admitted=member in selected))
                if member in selected:
                    assert size == selected[member]["source_member_stat"]["size"]
                    seen.add(member)
                archive.skip()
        assert seen == set(selected)
    write_json(CACHE / "inventory_preflight.json", inventory)
    print(json.dumps(dict(total_members=len(inventory), accepted_members=sum(r["admitted"] for r in inventory),
                          unpacked_bytes=sum(r["size"] for r in inventory))))
    raise SystemExit(0)
panel = panel_from_s1(STUDY, PANEL_ID, labels[6:], S1)
panel["source_table_column_index"] = panel.source_channel_index + 6
assert panel.loc[panel.nuclear, "source_channel_index"].tolist() == [41, 42]
with formal_publish_lease(OUTPUT, directory=OUTPUT):
    panel.to_csv(OUTPUT / "panel_mapping.csv", index=False, encoding="utf-8-sig")

# The release explicitly labels sample_id, but does not define its biological hierarchy.
source_map = pd.read_csv(META / "source_metadata/cai_zenodo_sample_Group.csv", dtype=str, keep_default_na=False)
assert len(source_map) == 157 and source_map.Sample.is_unique and source_map.sample_id.is_unique
source_map = source_map.set_index("Sample", verify_integrity=True)
for proof in proofs:
    member = proof["source_member"]
    source = source_map.loc[Path(member).stem]
    binding = proof["binding"]
    assert source.sample_id == binding["sample_id"] and source.Group == binding["group"]
samples = pd.DataFrame(columns=columns("sample_metadata"))
fields = field_mapping()
with formal_publish_lease(OUTPUT, directory=OUTPUT):
    samples.to_csv(OUTPUT / "sample_metadata.csv", index=False, encoding="utf-8-sig")
    fields.to_csv(OUTPUT / "field_mapping.csv", index=False, encoding="utf-8-sig")
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
# The first 27 TIFFs already contain the correct 32 detector pixels and names.
# Correct only the earlier source-table/channel count conflation in their index.
if not images.empty:
    images["panel_id"] = PANEL_ID
    images["source_channel_count"] = 44
    images["metadata_notes"] = images.metadata_notes.str.replace(
        "50 source table columns include 6 excluded acquisition fields and 44 detector columns.",
        "44 detector channels from 50 source table columns; six acquisition/XYZ fields are not image channels.",
        regex=False)
    save_input_tables(OUTPUT, images, samples, panel, fields)
    for receipt_path in (CACHE / "receipts").glob("*.json"):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("source_shape", [None])[0] == 50:
            receipt["source_shape"][0] = 44
            receipt["source_table_column_count"] = 50
            write_json(receipt_path, receipt)
rows = images.to_dict("records")
done = {r["image_id"] for r in rows}
inventory_path = CACHE / "archive_inventory.json"
inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else []
paused_path = CACHE / "paused_images.json"
paused = json.loads(paused_path.read_text()) if paused_path.exists() else {}
save_reviews(images, samples, inventory, False)
progress("metadata_ready" if not args.export else "waiting_or_export", len(rows), source_sample_rows=len(samples),
         patient_linked_samples=0, retained_channels=int(panel.keep_channel.sum()))

# Single-pass original 7z traversal; publish and checkpoint each independent image.
if args.export:
    archive_paths = list(dict.fromkeys(p["source_file_path"] for p in proofs))
    for archive_path in archive_paths:
        selected = {p["source_member"]: p for p in proofs if p["source_file_path"] == archive_path}
        seen = set()
        with Archive(Path(archive_path)) as archive:
            for member, size in archive:
                record = dict(archive=relative(Path(archive_path)), member=member, size=size,
                              admitted=member in selected, exclusion_reason="" if member in selected else "outside accepted native ROI scope")
                inventory = [r for r in inventory if (r["archive"], r["member"]) != (record["archive"], member)] + [record]
                write_json(inventory_path, inventory)
                if member not in selected:
                    archive.skip()
                    continue
                seen.add(member)
                proof = selected[member]
                assert size == proof["source_member_stat"]["size"]
                image_id = Path(member).stem
                destination = OUTPUT / "images" / image_id / "input.tiff"
                receipt_path = CACHE / "receipts" / (image_id + ".json")
                if image_id in done:
                    header_check(destination, next(r for r in rows if r["image_id"] == image_id), panel)
                    archive.skip()
                    continue
                progress("extract_native_member", len(rows), active_image=image_id, active_archive=Path(archive_path).name)
                txt = SCRATCH / (image_id + ".txt")
                if txt.exists() and txt.stat().st_size == size:
                    archive.skip()
                else:
                    archive.extract(txt, size)
                records_path = SCRATCH / (image_id + ".float32")
                progress("decode_native_txt", len(rows), active_image=image_id)
                records = decode_txt(txt, labels, records_path)
                try:
                    reader = TxtReader(records)
                except ValueError as error:
                    # A concrete native-coordinate failure pauses this ROI, not
                    # the remaining independent members of the complete archive.
                    paused[image_id] = dict(source_member=member, source_file=relative(Path(archive_path)),
                                            reason=str(error), source_record_count=len(records))
                    write_json(paused_path, paused)
                    progress("source_roi_paused", len(rows), active_image=image_id, paused_images=paused)
                    records._mmap.close()
                    txt.unlink()
                    records_path.unlink()
                    print(json.dumps(dict(image=image_id, paused=str(error))), flush=True)
                    continue
                _, height, width = reader.shape
                required = max(height, width)**2 * int(panel.keep_channel.sum())*4
                if shutil.disk_usage(NAS).free < required + 100*1024**3:
                    raise RuntimeError(f"Output for {image_id} would violate 100 GiB NAS reserve")
                row = dict.fromkeys(columns("image_metadata"), None)
                source = source_map.loc[image_id]
                row.update(study=STUDY, image_id=image_id, sample_id=None, source_sample_id=source.sample_id, panel_id=PANEL_ID,
                           image_kind="roi", source_image_id=member, source_files=[relative(Path(archive_path))],
                           source_format="txt", source_axes="XY records", source_dtype="float32",
                           source_revision="zenodo18041847_v1; exact size/mtime native proof binding",
                           roi_id=image_id, region_annotation=source.Group,
                           source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                           input_image_path=relative(destination), intensity_description="Native Hyperion TXT detector intensities decoded as float32, matching readimc; no normalization",
                           split_method="already_independent", split_evidence=relative(META / "primary_scope_review.json"),
                           metadata_notes="Original non-TMA instrument ROI; 44 detector channels from 50 source table columns; six acquisition/XYZ fields are not image channels. Publisher sample_id retained as source_sample_id; biological sample_id and patient crosswalk unresolved.")
                progress("export_native_tiff", len(rows), active_image=image_id, source_height=height, source_width=width,
                         estimated_uncompressed_output_bytes=required)
                result = export_input(reader, panel, destination)
                receipt = dict(image_id=image_id, source_file=relative(Path(archive_path)), source_member=member,
                               source_record_count=len(records), measured_record_count=len(reader.records),
                               terminal_zero_placeholder_records=reader.zero_footer_records,
                               placeholder_decision="Only all-50-column zero terminal rows after verified sequential Z prefix; retain real origin and all measured zeros",
                               source_shape=[44, height, width], source_table_column_count=50,
                               source_geometry="max(X/Y)+1 from all measured TXT records; no planned MCD frame dimensions published",
                               source_proof=relative(Path(proof["proof_path"])), result=result)
                if result["admitted"]:
                    if reader.zero_footer_records:
                        row["metadata_notes"] += f" {reader.zero_footer_records} terminal all-column-zero instrument placeholder records omitted after verifying sequential source Z prefix; real (0,0) retained. Measured zeros remain valid; unmeasured native positions use valid_pixels.tiff."
                    row.update({k: v for k, v in result.items() if k in row})
                    row["valid_pixels_path"] = relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None
                    header_check(destination, row, panel)
                    receipt["pixel_checks"] = pixel_check(reader, destination, row, panel)
                    rows.append(row)
                    done.add(image_id)
                    images = pd.DataFrame(rows)
                    save_input_tables(OUTPUT, images, samples, panel, fields)
                    save_reviews(images, samples, inventory, False)
                write_json(receipt_path, receipt)
                paused.pop(image_id, None)
                write_json(paused_path, paused)
                records._mmap.close()
                del reader, records
                txt.unlink()
                records_path.unlink()
                progress("image_saved", len(rows), last_image=image_id, last_admitted=result["admitted"])
                print(json.dumps(dict(image=image_id, completed=len(rows), admitted=result["admitted"])), flush=True)
        if seen != set(selected):
            raise ValueError(f"Missing accepted members in {Path(archive_path).name}: {set(selected)-seen}")

# Reopen every exported file; overview colors are display-only, not input transforms.
if args.export or args.verify_only:
    for row in rows:
        header_check(NAS / row["input_image_path"], row, panel)
    chosen = panel[panel.keep_channel].set_index("source_channel_index")
    overview_channels = [int(chosen.loc[c, "output_channel_index"]) for c in [6, 5, 41]]
    sheet = Image.new("RGB", (8*200, ((len(rows)+7)//8)*220), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        with tifffile.TiffFile(NAS / row["input_image_path"]) as tif:
            rgb = []
            for channel in overview_channels:
                plane = tif.pages[channel].asarray(maxworkers=2)
                nonzero = plane[plane > 0]
                scale = np.percentile(nonzero, 99) if len(nonzero) else 1
                rgb.append(np.clip(plane/scale*255, 0, 255).astype(np.uint8))
            thumbnail = Image.fromarray(np.stack(rgb, axis=-1))
            thumbnail.thumbnail((196, 196))
        x, y = (i % 8)*200, (i//8)*220
        sheet.paste(thumbnail, (x, y))
        draw.text((x+2, y+198), row["image_id"], fill="black")
    if rows:
        sheet.save(CACHE / "overviews/all_ROIs_PanCK_SMA_DNA.png")
    if args.overview_observation:
        write_json(CACHE / "overview_observation.json", dict(preparer=PREPARER,
                   image_ids=[r["image_id"] for r in rows], observation=args.overview_observation,
                   display="R=PanCK, G=SMA, B=DNA1; per-channel nonzero P99 for display only"))
    complete = len(rows) == 157
    footer_images = []
    for receipt_path in (CACHE / "receipts").glob("*.json"):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("terminal_zero_placeholder_records", 0):
            footer_images.append(dict(image_id=receipt["image_id"],
                                      zero_placeholder_rows=receipt["terminal_zero_placeholder_records"],
                                      measured_records=receipt["measured_record_count"]))
    if footer_images:
        write_json(CACHE / "terminal_zero_record_decisions.json", footer_images)
        GAPS.append("部分TXT含重复原点的全零尾部占位记录，已按连续Z前缀证据处理；实测不足完整矩形处保存valid_pixels，详见terminal_zero_record_decisions.json，需独立复核。")
    if paused:
        GAPS.append("坐标解释仍有必要缺口的ROI已单独暂停：" + ", ".join(sorted(paused)))
    save_reviews(pd.DataFrame(rows), samples, inventory, complete)
    summary = dict(study=STUDY, status="prepared" if complete else "partial", expected_images=157,
                   exported_images=len(rows), sample_rows=len(samples), sample_linked_images=0,
                   patient_linked_samples=0, retained_channels=int(panel.keep_channel.sum()),
                   paused_images=paused, terminal_zero_record_images=footer_images,
                   issues=GAPS, controller_requests=CONTROLLER_REQUESTS,
                   summary="原生image及157个精确来源sample_id已整理；生物学取样归并和患者crosswalk未解决，sample表为空，尚待独立复核。")
    write_json(CACHE / "preparation_summary.json", summary)
    progress("prepared_pending_independent_review" if complete else "partial", len(rows),
             sample_rows=len(samples), sample_linked_images=0, patient_linked_samples=0,
             overview_observed=(CACHE / "overview_observation.json").exists())
    print(json.dumps(summary, ensure_ascii=False), flush=True)

# Record the human observation after viewing existing overviews without reading
# every native-intensity TIFF again or regenerating already correct pixels.
if args.overview_observation and not (args.export or args.verify_only):
    write_json(CACHE / "overview_observation.json", dict(preparer=PREPARER,
               image_ids=[r["image_id"] for r in rows], observation=args.overview_observation,
               display="R=PanCK, G=SMA, B=DNA1; per-channel nonzero P99 for display only",
               evidence=[relative(CACHE / "overviews/all_ROIs_PanCK_SMA_DNA.png"),
                         relative(CACHE / "pCR-34_source_overview_observation.json")]))
    summary = json.loads((CACHE / "preparation_summary.json").read_text())
    summary["overview_observed"] = True
    write_json(CACHE / "preparation_summary.json", summary)
    GAPS[:] = summary["issues"]
    save_reviews(pd.DataFrame(rows), samples, inventory, len(rows) == 157)
    progress("prepared_pending_independent_review" if len(rows) == 157 else "partial", len(rows),
             sample_rows=len(samples), sample_linked_images=0, patient_linked_samples=0,
             overview_observed=True)
