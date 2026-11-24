"""整理本地肝癌 MCD 的输入来源、身份连接和 study 专属 S1 准入。

输入为 local_intake_v1、完整原始 MCD、来源 panel/采集表和唯一 S1。
输出为本 study 的来源证据、映射建议、输入表、逐图 TIFF 与进度。
MCD 容器标签不自动成为生物学 sample；S1 缺本 study 时暂停像素发布。
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "IMC_20240725_91"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
INTAKE = metadata_source_directory(NAS, STUDY, scope='private') / 'local_intake_v1'
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
LEGACY = Path("/mnt/public/lyx/horizontalProject") / STUDY
PROGRESS = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_PROGRESS",
    str(PROJECT / "cache/study_preparation" / STUDY / "jobs/0004_prepare_0/progress.json"),
))
PREPARER = "0004_prepare_0"
pa.set_cpu_count(4)
pa.set_io_thread_count(2)


def relative(path: Path) -> str:
    """NAS 内正式引用使用相对路径，外部只读证据保留其实际位置。"""
    return str(metadata_source_relative(path, NAS)) if path.is_relative_to(NAS) else str(path)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, completed: int, expected: int, gaps: list[str], **details: object) -> None:
    save_json(PROGRESS, dict(study=STUDY, stage=stage, completed_images=completed,
                            expected_images=expected, gaps=gaps,
                            updated_at=datetime.now(timezone.utc).isoformat(), **details))


def csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve().is_relative_to(OUTPUT.resolve()):
        with formal_publish_lease(path, directory=path.parent):
            frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def headers(files: pd.DataFrame, rois: pd.DataFrame, panels: pd.DataFrame) -> list[dict]:
    """复用既有原生核验，核对 size/mtime、实际采集及有序标签，不重做 hash。"""
    records = []
    for number, item in enumerate(files.itertuples(index=False), 1):
        path = NAS / item.logical_path
        stat = path.stat()
        if stat.st_size != item.size_bytes or stat.st_mtime_ns != item.mtime_ns:
            raise ValueError(f"Source identity changed: {item.logical_path}")
        indexed = rois[rois.mcd_path.eq(item.logical_path)]
        with MCDFile(path) as mcd:
            acquisitions = {a.id: a for slide in mcd.slides for a in slide.acquisitions}
            for roi in indexed.itertuples(index=False):
                acq = acquisitions[int(roi.acquisition_id)]
                native = panels[panels.panel_id.eq(roi.panel_id)].sort_values("channel_index")
                if list(acq.channel_labels) != native.source_label.tolist():
                    raise ValueError(f"Native ordered panel differs: {roi.image_id}")
                if (int(acq.metadata["MaxX"]), int(acq.metadata["MaxY"]), acq.num_channels) != (
                        int(roi.width), int(roi.height), int(roi.channel_count)):
                    raise ValueError(f"Native acquisition shape differs: {roi.image_id}")
                if acq.metadata.get("Description", "") != roi.description:
                    raise ValueError(f"Native acquisition annotation differs: {roi.image_id}")
            identity_nodes = []
            for node in ElementTree.fromstring(mcd.schema_xml).iter():
                key = node.tag.split("}")[-1]
                if any(term in key.lower() for term in ("patient", "sample", "specimen", "clinical", "donor", "subject", "tissue")):
                    identity_nodes.append({"field": key, "value": node.text or ""})
            records.append(dict(source_file=item.logical_path, size_bytes=stat.st_size,
                                mtime_ns=stat.st_mtime_ns, indexed_tissue_acquisitions=len(indexed),
                                actual_all_acquisitions=len(acquisitions),
                                slides=[dict(slide_id=s.id, metadata=s.metadata,
                                             panoramas=[dict(panorama_id=p.id, metadata=p.metadata) for p in s.panoramas])
                                        for s in mcd.slides], identity_nodes=identity_nodes,
                                ordered_panel_and_acquisition_match=True))
        save_json(CACHE / "native_header_evidence.json", records)
        if number % 10 == 0 or number == len(files):
            progress("native_metadata_checked", 0, int(rois.passed.sum()), [], checked_files=number,
                     source_files=len(files), biological_sample_link_status="unresolved")
            print(f"metadata {number}/{len(files)}", flush=True)
    return records


def field_rules() -> pd.DataFrame:
    """逐项说明39个image字段，以及未建立sample/patient外键的实际原因。"""
    definitions = {
        "study": ("dataset_id", "固定来源 study 标识"),
        "image_id": ("image_id", "复用 rois.csv 的容器别名+acquisition 身份；与原 MCD 核对"),
        "sample_id": ("sample_unit;specimen_label_basis", "MCD_container/source_filename_only 不构成生物学取样证据；留空"),
        "panel_id": ("panel_id;ordered native channel labels", "复用原 panel_id 并逐采集核对来源有序标签"),
        "image_kind": ("tissue", "来源组织 acquisition 记 roi"),
        "source_image_id": ("mcd_path;acquisition_id", "NAS相对MCD路径#Acquisition:来源ID"),
        "source_files": ("mcd_path", "单元素 NAS 相对路径列表"),
        "source_format": ("source", "mcd"),
        "source_acquisition_id": ("acquisition_id", "原始采集ID转换为字符串"),
        "source_z_index": ("", "不适用，留空"),
        "source_t_index": ("", "不适用，留空"),
        "source_axes": ("native MCD records", "MCD按坐标解码后的逻辑CYX；保留原生float32"),
        "source_dtype": ("ValueBytes", "MCD原始4字节浮点解码为float32"),
        "source_revision": ("size_bytes;mtime_ns", "local_intake_v1及逐源size/mtime绑定；复用原核验，不重做hash"),
        "slide_id": ("Slide.ID", "容器别名+原Slide.ID，仅定位容器内切片，不宣称生物学样本"),
        "core_id": ("", "无物理core对应证据，留空"),
        "roi_id": ("acquisition_id", "原始acquisition ID"),
        "fov_id": ("", "来源无独立FOV身份，留空"),
        "region_annotation": ("description", "保留MCD acquisition description原值"),
        "source_x0_px": ("MaxX", "完整来源ROI，x0=0"),
        "source_y0_px": ("MaxY", "完整来源ROI，y0=0"),
        "source_x1_px": ("width", "完整ROI半开框右界=原width"),
        "source_y1_px": ("height", "完整ROI半开框下界=原height"),
        "pad_left_px": ("width;height", "(max(width,height)-width)//2"),
        "pad_top_px": ("width;height", "(max(width,height)-height)//2"),
        "pad_right_px": ("width;height", "max(width,height)-width-pad_left_px"),
        "pad_bottom_px": ("width;height", "max(width,height)-height-pad_top_px"),
        "height_px": ("width;height", "max(width,height)"),
        "width_px": ("width;height", "max(width,height)"),
        "source_channel_count": ("channel_count", "来源全部通道数"),
        "channel_count": ("S1;panel_mapping.keep_channel", "本study专属S1准入后实际输出数；缺S1时候选表留空"),
        "pixel_size_x_um": ("pixel_size_x_um", "原MCD采集尺度，不从图像大小推断"),
        "pixel_size_y_um": ("pixel_size_y_um", "原MCD采集尺度，不从图像大小推断"),
        "input_image_path": ("export_input", "仅实际写入、重开检查通过后发布NAS相对路径"),
        "valid_pixels_path": ("MCD record coordinates", "存在未采集位置时记录实际生成的uint8有效域；padding字段另行保留"),
        "intensity_description": ("native MCD pixel records", "native float32 MCD ion intensities; no normalization"),
        "split_method": ("acquisition_id", "source_acquisition，完整原生ROI"),
        "split_evidence": ("rois.csv;MCD acquisition metadata", "逐采集核对原尺寸、有序panel与description；未声称sample归属已确认"),
        "metadata_notes": ("sample/clinical crosswalk search", "记录case_study、禁止pretraining、split待临床/医院确认及身份缺口"),
    }
    rows = []
    for field, (source, rule) in definitions.items():
        rows.append(dict(study=STUDY, source_file=relative(INTAKE / "rois.csv"),
                         source_sheet_or_table="rois;files;samples;native_header_evidence", source_field=source,
                         target_table="image_metadata", target_field=field,
                         join_keys="mcd_path + acquisition_id", value_mapping="{}", transformation_rule=rule,
                         time_origin_rule="", conflict_rule="以原MCD标签为准；不从容器名推断sample/patient", notes=""))
    for field in ("sample_id", "patient_key"):
        rows.append(dict(study=STUDY, source_file=relative(CACHE / "sample_link_evidence.json"),
                         source_sheet_or_table="local source/clinical search", source_field="",
                         target_table="sample_metadata", target_field=field, join_keys="尚无已证实生物学取样连接键",
                         value_mapping="{}", transformation_rule="身份未获来源确认，保留空表及明确缺口",
                         time_origin_rule="", conflict_rule="不使用ROI、MCD文件名或病理文件名猜患者", notes="M01 unresolved"))
    return pd.DataFrame(rows, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)


def candidate_image(roi: pd.Series, file: pd.Series, panel: pd.DataFrame | None) -> dict:
    row = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
    width, height = int(roi.width), int(roi.height)
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    row.update(study=STUDY, image_id=roi.image_id, panel_id=roi.panel_id, image_kind="roi",
               source_image_id=f"{roi.mcd_path}#Acquisition:{roi.acquisition_id}", source_files=[roi.mcd_path],
               source_format="mcd", source_acquisition_id=str(roi.acquisition_id), source_axes="CYX",
               source_dtype="float32", source_revision=f"local_intake_v1;size={file.size_bytes};mtime_ns={file.mtime_ns}",
               slide_id=f"{Path(roi.mcd_path).stem}__slide_{roi.slide_id}", roi_id=str(roi.acquisition_id),
               region_annotation=roi.description, source_x0_px=0, source_y0_px=0, source_x1_px=width,
               source_y1_px=height, pad_left_px=left, pad_top_px=top, pad_right_px=side-width-left,
               pad_bottom_px=side-height-top, height_px=side, width_px=side,
               source_channel_count=int(roi.channel_count),
               channel_count=int(panel.keep_channel.sum()) if panel is not None else None,
               pixel_size_x_um=roi.pixel_size_x_um, pixel_size_y_um=roi.pixel_size_y_um,
               intensity_description="native float32 MCD ion intensities; no normalization",
               split_method="source_acquisition", split_evidence=relative(INTAKE / "rois.csv"),
               metadata_notes="private;case_study;pretraining_allowed=false;patient_split_status=pending_clinical_hospital_confirmation;biological_sample_and_patient_crosswalk_unresolved")
    return row


def unresolved_panel(source: pd.DataFrame, reason: str) -> pd.DataFrame:
    """S1未裁定时仍保留该panel全部来源行，正式取舍和输出位置留空。"""
    frame = pd.DataFrame(columns=pd.read_csv(TEMPLATES / "panel_mapping.csv").columns)
    frame["panel_id"] = source.panel_id.to_numpy()
    frame["source_channel_index"] = source.channel_index.to_numpy()
    frame["source_name"] = source.source_label.to_numpy()
    frame["source_isotope"] = source.isotope.to_numpy()
    frame["source_role"] = source.role.to_numpy()
    frame["s1_resolution"] = reason
    return frame


# 1. 对账完整本地来源、既有准入和归档范围。
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export", action="store_true", help="须由shared formal write lease封装；仅S1逐来源行完整时导出")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
files = pd.read_csv(INTAKE / "files.csv", keep_default_na=False)
rois = pd.read_csv(INTAKE / "rois.csv", keep_default_na=False)
native_panel = pd.read_csv(INTAKE / "panels.csv", keep_default_na=False)
native_verification = json.loads((INTAKE / "native_verification.json").read_text())
archive = json.loads((INTAKE / "archive_member_inventory.json").read_text())
source_files = json.loads((INTAKE / "source_files.json").read_text())
accepted = rois[rois.passed].copy()
actual_files = {relative(p) for p in (NAS / "raw/private" / STUDY / "original_mcd_v1").rglob("*.mcd")}
if actual_files != set(files.logical_path) or set(files.logical_path) != {item["path"] for item in source_files}:
    raise ValueError("Local complete MCD scope differs from the original source manifest")
if len(archive) != len(source_files) or any(not member["member"].lower().endswith(".mcd") for member in archive):
    raise ValueError("Archive attachment scope changed; inspect new members before publishing")
header_evidence = headers(files, rois, native_panel)

# 2. 主动追查本地身份资料；所有真实来源标签只保存到NAS。
legacy_images_path = LEGACY / "steinbock/images.csv"
legacy_images = pd.read_csv(legacy_images_path, keep_default_na=False)
legacy_keys = {(Path(row.source_file).name, int(row.acquisition_id)) for row in legacy_images.itertuples(index=False)}
source_links = []
for record in header_evidence:
    source = Path(record["source_file"])
    source_links.append(dict(mcd_path=record["source_file"], source_container_label=source.stem,
                             embedded_slide_filenames=[s["metadata"].get("Filename", "") for s in record["slides"]],
                             biological_sample_id=None, patient_key=None,
                             status="container_and_embedded_filename_only_not_biological_crosswalk"))
save_json(CACHE / "container_identity_evidence.json", source_links)
local_tables = sorted(str(path) for pattern in ("*.csv", "*.xlsx", "*.xls") for path in (LEGACY / "data").rglob(pattern))
identity_nodes = [node for record in header_evidence for node in record["identity_nodes"]]
sample_evidence = dict(study=STUDY, archive_members=len(archive), archive_metadata_attachments=0,
                       source_intake_samples_unit="MCD_container", source_specimen_label_basis="source_filename_only",
                       local_data_directory=str(LEGACY / "data"), local_metadata_tables=local_tables,
                       legacy_analysis_entry=str(LEGACY / "code/1_readData.R"),
                       missing_legacy_crosswalk=[str(LEGACY / "data/FileName_DF.csv"),
                                                "/mnt/public/lyx/horizontalProject/IMC_xuanwu/data/FileName_DF.csv"],
                       legacy_crosswalk_exists=[(LEGACY / "data/FileName_DF.csv").exists(),
                                                Path("/mnt/public/lyx/horizontalProject/IMC_xuanwu/data/FileName_DF.csv").exists()],
                       native_explicit_identity_nodes=identity_nodes,
                       legacy_images_path=str(legacy_images_path), legacy_acquisition_rows=len(legacy_images),
                       accepted_images_with_legacy_acquisition_reference=sum(
                           (Path(row.mcd_path).name, int(row.acquisition_id)) in legacy_keys for row in accepted.itertuples(index=False)),
                       accepted_source_images=len(accepted), sample_rows=0, sample_linked_images=0,
                       patient_linked_samples=0, required_evidence="image/acquisition→biological specimen crosswalk and specimen→patient clinical table",
                       conclusion="尚无已证实生物学取样或患者连接；容器/内嵌文件名保留为线索，不推断sample/patient。",
                       role="case_study", pretraining_allowed=False,
                       patient_split_status="pending_clinical_hospital_confirmation")
save_json(CACHE / "sample_link_evidence.json", sample_evidence)

# 3. 本study专属S1缺行时，保存完整panel身份与有来源的建议，保持准入未裁定。
s1 = pd.read_csv(S1, dtype=str, keep_default_na=False)
study_s1 = s1[s1.Study.eq(STUDY)]
panel_frames = []
panel_gaps = []
resolved_panels = set()
proposals = []
reference_panel = pd.read_csv(NAS / "raw/private" / STUDY / "source_tables_v1/panel.csv", dtype=str, keep_default_na=False).set_index("channel")
for panel_id, source in native_panel.groupby("panel_id", sort=False):
    source = source.sort_values("channel_index")
    for channel in source.itertuples(index=False):
        reference = reference_panel.loc[channel.isotope]
        proposals.append(dict(study=STUDY, panel_id=panel_id, source_channel_index=channel.channel_index,
                              source_name=channel.source_label, isotope=channel.isotope,
                              native_marker_label=channel.marker_name, native_role=channel.role,
                              proposed_original_marker=channel.source_label or f"{channel.isotope} (unlabelled native channel)",
                              proposed_display_name=channel.marker_name,
                              source_keep=reference["keep"], source_nuclear=reference["cellpose"] == "1",
                              source_body=reference["cellpose"] == "2",
                              proposal_status="requires_study_specific_S1_adjudication",
                              duplicate_decision=("Gd156与Tm169均为原生CD31；Gd156有来源body=2；需裁定downstream测量，保留已选分割输入"
                                                  if panel_id == "panel_03" and channel.marker_name == "CD31" else ""),
                              evidence=relative(INTAKE / "panels.csv") + ";" + relative(NAS / "raw/private" / STUDY / "source_tables_v1/panel.csv"),
                              identity_note=channel.identity_status,
                              notes="仅来源身份及用途建议；未借用其他study映射，未赋予MAP-KG靶标或正式S1准入"))
    if study_s1.empty:
        panel_gaps.append(f"{panel_id}: S1 has no rows for {STUDY}")
        frame = unresolved_panel(source, "unresolved_study_absent_from_S1")
    else:
        # S1 必须使用本study的原始标签；其他study及来源panel用途不能代替准入。
        try:
            frame = panel_from_s1(STUDY, panel_id, source.source_label.tolist(), S1)
        except ValueError as error:
            panel_gaps.append(str(error))
            frame = unresolved_panel(source, str(error))
        else:
            resolved_panels.add(panel_id)
    panel_frames.append(frame)
panel = pd.concat(panel_frames, ignore_index=True) if panel_frames else pd.DataFrame()
csv(CACHE / "s1_mapping_proposals.csv", pd.DataFrame(proposals))
samples = pd.read_csv(TEMPLATES / "sample_metadata.csv")
mapping = field_rules()
images_path = OUTPUT / "images.parquet"
images = pd.read_parquet(images_path) if images_path.exists() else pd.read_csv(TEMPLATES / "image_metadata.csv")
indexed_files = files.set_index("logical_path")
candidates = [candidate_image(roi, indexed_files.loc[roi.mcd_path],
                              panel[panel.panel_id.eq(roi.panel_id)] if roi.panel_id in resolved_panels else None)
              for _, roi in accepted.iterrows()]
csv(CACHE / "candidate_images.csv", pd.DataFrame(candidates).assign(
    source_files=lambda frame: frame.source_files.map(json.dumps)))
scope = dict(study=STUDY, complete_mcd_files=len(files), archive_members=len(archive),
             tissue_acquisitions=len(rois), all_acquisitions=len(native_verification["acquisitions"]),
             native_accepted_images=len(accepted), native_excluded_tissue_images=len(rois)-len(accepted),
             excluded_tissue_reasons=rois[~rois.passed].exclusion_reason.value_counts().to_dict(),
             non_tissue_acquisitions=sum(not row["tissue"] for row in native_verification["acquisitions"]),
             accepted_images_per_panel=accepted.panel_id.value_counts().to_dict(),
             output_uncompressed_upper_bound_bytes=int(sum(max(row.width, row.height)**2 * 31 * 4
                                                          for row in accepted.itertuples(index=False))),
             extraction_required=False, native_proof_reused=True, new_full_source_hashes=0,
             complete_source_scope_matches=True, native_metadata_checked_files=len(header_evidence),
             source_only_or_duplicate_publish=False, role="case_study", pretraining_allowed=False)
save_json(CACHE / "source_scope.json", scope)

# 4. 只有本study的panel准入完整时才逐图保存；每图立即合并发布索引。
if resolved_panels and args.export:
    excluded = []
    for row in candidates:
        if row["panel_id"] not in resolved_panels:
            continue
        if row["image_id"] in set(images.image_id):
            continue
        selected = panel[panel.panel_id.eq(row["panel_id"])]
        source = SourceSpec(row["image_id"], row["panel_id"], "mcd", NAS / row["source_files"][0],
                            tuple(range(row["source_channel_count"])),
                            acquisition_id=int(row["source_acquisition_id"]),
                            pixel_size_um=(row["pixel_size_x_um"], row["pixel_size_y_um"]))
        destination = OUTPUT / "images" / row["image_id"] / "input.tiff"
        with open_source(source) as reader:
            result = export_input(reader, selected, destination)
            if not result["admitted"]:
                excluded.append(dict(image_id=row["image_id"], reason="all_retained_downstream_markers_zero"))
                csv(CACHE / "signal_exclusions.csv", pd.DataFrame(excluded))
                continue
            kept = selected[selected.keep_channel].sort_values("output_channel_index")
            # 全保留通道的内部及边界小块逐值自检，独立review另行执行。
            with tifffile.TiffFile(destination) as tif:
                for window in ((0, 0, min(32, reader.shape[2]), min(32, reader.shape[1])),
                               (max(0, reader.shape[2]-32), max(0, reader.shape[1]-32), reader.shape[2], reader.shape[1])):
                    x0, y0, x1, y1 = window
                    output_window = (x0+row["pad_left_px"], y0+row["pad_top_px"],
                                     x1+row["pad_left_px"], y1+row["pad_top_px"])
                    for channel in kept.itertuples(index=False):
                        observed = read_tiff_window(tif, int(channel.output_channel_index), output_window)
                        expected = reader.read_window([channel.source_channel_index], window)[0]
                        if not np.array_equal(observed, expected):
                            raise ValueError(f"Native pixel mismatch: {row['image_id']}")
        row["input_image_path"] = relative(destination)
        row["valid_pixels_path"] = relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None
        images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
        save_input_tables(OUTPUT, images, samples, panel, mapping)
        progress("exporting", len(images), len(accepted), ["sample/patient crosswalk unresolved"])
elif resolved_panels:
    panel_gaps.append("Panel resolved; rerun under shared formal write lease with --export for native TIFF delivery")

if images.empty:
    save_input_tables(OUTPUT, images, samples, panel, mapping)

# 5. 整理者自检记录；reviewer留空，由独立上下文完成正式复核。
review = []
for check, result, detail in [
    ("S01", "pass", f"整理者来源核对：{len(files)} MCD；{len(rois)}组织采集；{len(accepted)}原核验通过候选；45个组织采集排除详见source_scope.json。尚未进行准入marker全图信号判定。"),
    ("I01", "unresolved" if images.empty else "pass", f"来源image/acquisition身份已核对；实际已发布{len(images)}，候选不写成已交付image。"),
    ("I02", "unresolved", "sample外键留空；panel来源身份已确认，正式S1准入仍依实际panel缺口。"),
    ("I03", "unresolved" if images.empty else "pass", "候选完整ROI框及方形padding已计算；TIFF交付范围按images.parquet。"),
    ("I04", "unresolved", "来源ROI描述已核对；缺生物学sample对应，未进行多sample拆分或宣称sample区域已确认。"),
    ("P01", "unresolved" if panel_gaps else "pass", "; ".join(panel_gaps) or "本study专属S1精确连接"),
    ("P02", "unresolved" if panel_gaps else "pass", "原生两panel及全部70来源行保留；panel_03有两个CD31测量，需来源支持的重复通道选择。"),
    ("P03", "unresolved" if images.empty else "pass", f"实际TIFF {len(images)}；无输出的候选未填写虚构输出通道位置。"),
    ("I05", "unresolved" if images.empty else "pass", "无TIFF时无像素相等结论；导出自检仅覆盖代码所列全部保留通道的两个32像素窗口；独立复核待执行。"),
    ("M01", "unresolved", "已查原归档成员、109份MCD内嵌元信息、local_intake、来源采集表和旧分析入口。sample行0；image→sample 0/889；sample→patient 0；缺标本及临床crosswalk。"),
    ("M02", "unresolved", "无已确认sample/临床目标值；无可验证的生物学身份连接。"),
    ("D01", "unresolved", f"整理者交付：实际image {len(images)}/{len(accepted)}，完整来源panel身份及映射建议已保存；sample/patient未连接；待主控裁定S1后续并安排独立复核。"),
]:
    review.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample" if check.startswith("M") else "study",
                       record_id=STUDY, check_id=check, result=result,
                       source_evidence=relative(INTAKE) + ";" + relative(S1),
                       output_evidence=relative(CACHE) + ";" + relative(OUTPUT),
                       details="preparer self-check; independent review pending; " + detail))
csv(OUTPUT / "review_results.csv", pd.DataFrame(review, columns=pd.read_csv(TEMPLATES / "review_results.csv").columns))
gaps = panel_gaps + ["缺image/acquisition→生物学标本及标本→patient临床crosswalk；MCD容器不能直接视为sample"]
summary = dict(study=STUDY, status="partial", script=str(metadata_source_relative(Path(__file__), Path.cwd())),
               expected_images=len(accepted), exported_images=len(images), sample_rows=len(samples),
               sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=0,
               issues=gaps, controller_requests=[
                   "在唯一S1增加本study专属原始标签/统一marker/分割输入裁定，详见s1_mapping_proposals.csv；不能借用其他study映射。",
                   "裁定panel_03的Gd156/Tm169重复CD31下游选择；原MCD的169Tm_CD31权威标签保持。",
                   "提供可核验的acquisition→生物学标本→patient本地crosswalk，并确认医院/正式split；现有case_study、pretraining_allowed=false保持。",
                   "S1补齐后由本study脚本在shared formal write lease封装下--export，随后安排不同上下文独立复核。"],
               evidence_paths=[relative(CACHE / name) for name in ("source_scope.json", "native_header_evidence.json", "sample_link_evidence.json", "container_identity_evidence.json", "s1_mapping_proposals.csv", "candidate_images.csv")]
                              + [relative(OUTPUT / name) for name in ("images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv", "review_results.csv")],
               summary="已实际完成来源、原生metadata和身份线索整理；S1无本study条目，TIFF导出暂停，未发布虚构sample或患者。")
save_json(CACHE / "preparation_summary.json", summary)
progress("partial_S1_and_biological_crosswalk_unresolved", len(images), len(accepted), gaps,
         sample_rows=len(samples), sample_linked_images=summary["sample_linked_images"], patient_linked_samples=0,
         metadata_files_checked=len(header_evidence), formal_independent_review="pending")
print(json.dumps(summary, ensure_ascii=False), flush=True)
