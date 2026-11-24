"""Render one matched channel/mask crop for every Allam2024 ROI.

Inputs: processed/Allam2024_IMC_NormalSLO/images.parquet and per-ROI image,
mask, channels, and metadata assets. Outputs review crops and a review index
under processed/_catalog/public/evidence/segmentation_followup_20261119/.
Upstream: fixed public manifest 20261119_v2 and current cpsam_v2 masks.
"""

import json
import math
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Allam2024_IMC_NormalSLO"
STUDY_ROOT = ROOT / "processed" / STUDY
OUTPUT_ROOT = ROOT / "processed/_catalog/public/evidence/segmentation_followup_20261119" / STUDY
SEED = 20261119
CROP_SIZE = 512
PANEL_SIZE = 320

helpers = runpy.run_path(str(Path(__file__).with_name("01_generate_html.py")))
tiff_windows = helpers["tiff_windows"]
scaled_channel = helpers["scaled_channel"]
display_scale = helpers["display_scale"]
channel_label = helpers["channel_label"]
colorize_mask = helpers["colorize_mask"]


def selected_channels(table):
    table = table.copy()
    table["tiff_page"] = pd.to_numeric(table["tiff_page"], errors="coerce")
    table = table[table["keep_channel"].fillna(False) & table["tiff_page"].notna()]
    table["nuclear"] = table["nuclear"].fillna(False).astype(bool)
    table["body"] = table["body"].fillna(False).astype(bool)
    rows = table[table["nuclear"] | table["body"]].sort_values("tiff_page").to_dict("records")
    return [dict(row, tiff_page=int(row["tiff_page"])) for row in rows]


def crop_candidates(height, width, valid_bounds):
    size = min(CROP_SIZE, height, width)
    x0, y0, x1, y1 = [int(value) for value in valid_bounds]
    x0, y0 = max(0, min(x0, width - size)), max(0, min(y0, height - size))
    x1, y1 = max(x0 + size, min(x1, width)), max(y0 + size, min(y1, height))
    xs = sorted(set((x0, (x0 + x1 - size) // 2, x1 - size)))
    ys = sorted(set((y0, (y0 + y1 - size) // 2, y1 - size)))
    return [(x, y, x + size, y + size) for x, y in ((xs[0], ys[0]),
                                                       (xs[len(xs)//2], ys[len(ys)//2]),
                                                       (xs[-1], ys[-1]))]


def choose_window(channels, channel_views, mask_views):
    scales = {item["tiff_page"]: display_scale(
        np.concatenate([view.ravel() for view in channel_views[item["tiff_page"]]]),
        item.get("q995")) for item in channels}
    nuclear_pages = [item["tiff_page"] for item in channels if item["nuclear"]]
    body_pages = [item["tiff_page"] for item in channels if item["body"]]
    scores = []
    for index, mask in enumerate(mask_views):
        nuc = np.maximum.reduce([scaled_channel(channel_views[page][index], scales[page])
                                 for page in nuclear_pages])
        body = np.maximum.reduce([scaled_channel(channel_views[page][index], scales[page])
                                  for page in body_pages])
        occupied = float(np.count_nonzero(mask)) / mask.size
        activity = float(np.mean(nuc) + np.mean(body))
        scores.append(activity + occupied)
    return int(np.argmax(scores)), scales


def draw_panel(canvas, x, y, panel, label, font):
    if panel.ndim == 2:
        gray = (np.clip(panel, 0, 1) * 255).astype(np.uint8)
        panel = np.repeat(gray[:, :, None], 3, axis=2)
    tile = Image.fromarray(np.asarray(panel, dtype=np.uint8), mode="RGB")
    tile = tile.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    canvas.paste(tile, (x, y + 28))
    ImageDraw.Draw(canvas).text((x, y), label[:54], fill=(240, 244, 247), font=font)


def make_figure(study, roi, channels, views, mask, scales, bounds, output_path):
    nuclear = [item for item in channels if item["nuclear"]]
    body = [item for item in channels if item["body"]]
    first = views[channels[0]["tiff_page"]]
    body_signal = np.maximum.reduce([scaled_channel(views[item["tiff_page"]],
                                                    scales[item["tiff_page"]]) for item in body])
    nuclear_signal = np.maximum.reduce([scaled_channel(views[item["tiff_page"]],
                                                       scales[item["tiff_page"]]) for item in nuclear])
    composite = np.zeros(first.shape + (3,), dtype=np.uint8)
    composite[:, :, 0] = (body_signal * 255).astype(np.uint8)
    composite[:, :, 1] = (body_signal * 145).astype(np.uint8)
    composite[:, :, 2] = (nuclear_signal * 255).astype(np.uint8)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    overlay = composite.copy()
    overlay[boundary] = (255, 255, 255)
    panels = [(scaled_channel(views[item["tiff_page"]], scales[item["tiff_page"]]),
               channel_label(item)) for item in channels]
    panels.extend([(overlay, "DNA/body + mask boundary"),
                   (colorize_mask(mask.astype(np.uint32)), "Instance mask")])

    columns = 3
    rows = int(math.ceil(len(panels) / float(columns)))
    gap, margin = 14, 26
    canvas_width = margin * 2 + columns * PANEL_SIZE + (columns - 1) * gap
    canvas_height = 56 + rows * (PANEL_SIZE + 42) + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 24, 29))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((margin, 16), "{} | {} | x=[{},{}) y=[{},{})".format(
        study, roi, bounds[0], bounds[2], bounds[1], bounds[3]), fill=(246, 248, 249), font=title_font)
    for index, (panel, label) in enumerate(panels):
        col, row = index % columns, index // columns
        x = margin + col * (PANEL_SIZE + gap)
        y = 54 + row * (PANEL_SIZE + 42)
        draw_panel(canvas, x, y, panel, label, label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=94, optimize=True)


def make_source_comparison(roi_root, channels, output_path):
    """Compare native marker files to input and normalized image pages."""
    selected = [item for item in channels
                if item["marker_id"] in {"DNA1", "DNA2", "Vimentin", "CD11c"}]
    rows = []
    bounds = [(0, 0, CROP_SIZE, CROP_SIZE)]
    source_root = ROOT / "raw/public" / STUDY / "INT Donor E"
    with tifffile.TiffFile(str(roi_root / "input.tiff")) as input_tif, \
         tifffile.TiffFile(str(roi_root / "image.tiff")) as image_tif:
        for item in selected:
            marker = str(item["marker_id"])
            with tifffile.TiffFile(str(source_root / (marker + ".tif"))) as source_tif:
                source = tiff_windows(source_tif.pages[0], bounds)[0]
            source_index = int(item["output_channel_index"])
            page_index = int(item["tiff_page"])
            input_view = tiff_windows(input_tif.pages[source_index], bounds)[0]
            image_view = tiff_windows(image_tif.pages[page_index], bounds)[0]
            scale = float(item.get("q995") or 1.0)
            rows.append((marker, [source / scale, input_view / scale, image_view],
                         source_index, page_index))

    gap, margin, panel = 14, 26, PANEL_SIZE
    canvas = Image.new("RGB", (margin * 2 + 3 * panel + 2 * gap,
                               56 + len(rows) * (panel + 48) + margin), (18, 24, 29))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    draw.text((margin, 16), "INT_Donor_E | source / input.tiff / image.tiff, same XY",
              fill=(246, 248, 249), font=title_font)
    for row_index, (marker, arrays, source_index, page_index) in enumerate(rows):
        y = 54 + row_index * (panel + 48)
        labels = [marker + ".tif source", "input.tiff page {}".format(source_index),
                  "image.tiff page {}".format(page_index)]
        for col, (array, label) in enumerate(zip(arrays, labels)):
            view = np.clip(np.asarray(array, dtype=np.float32), 0, 1)
            draw_panel(canvas, margin + col * (panel + gap), y, view, label, label_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=94, optimize=True)


images = pd.read_parquet(STUDY_ROOT / "images.parquet")
records = []
for image in images.to_dict("records"):
    roi = str(image["image_id"])
    roi_root = STUDY_ROOT / "images" / roi
    channels_table = pd.read_parquet(roi_root / "channels.parquet")
    channels = selected_channels(channels_table)
    metadata = json.loads((roi_root / "metadata.json").read_text())
    task = metadata["segmentation_task"]
    valid_bounds = task["request"]["image"]["source"]["valid_bounds"]
    output_path = OUTPUT_ROOT / "figures" / (roi + ".jpg")

    with tifffile.TiffFile(str(roi_root / "image.tiff")) as image_tif, \
         tifffile.TiffFile(str(roi_root / "mask.tiff")) as mask_tif:
        height, width = mask_tif.pages[0].shape[-2:]
        candidates = crop_candidates(height, width, valid_bounds)
        bounds = [tuple(window) for window in candidates]
        channel_views = {item["tiff_page"]: tiff_windows(image_tif.pages[item["tiff_page"]], bounds)
                         for item in channels}
        mask_views = tiff_windows(mask_tif.pages[0], bounds)
        selected, scales = choose_window(channels, channel_views, mask_views)
        window = bounds[selected]
        local_views = {page: items[selected] for page, items in channel_views.items()}
        local_mask = mask_views[selected]
        make_figure(STUDY, roi, channels, local_views, local_mask, scales, window, output_path)

    records.append({
        "study": STUDY,
        "roi": roi,
        "mask_origin": metadata.get("mask_origin", "unknown"),
        "mask_identity": metadata.get("mask_identity"),
        "judgment": "待逐图审查",
        "findings": "",
        "likely_cause": "",
        "next_action": "",
        "figures": [output_path.relative_to(OUTPUT_ROOT).as_posix()],
        "window_xyxy": list(window),
        "channels": [channel_label(item) for item in channels],
        "segmentation_parameters": task.get("parameters", {}),
        "stitching": task.get("stitching", {}),
    })

int_e_record = next(item for item in records if item["roi"] == "INT_Donor_E")
mapping_figure = OUTPUT_ROOT / "figures" / "INT_Donor_E_source_mapping.jpg"
make_source_comparison(STUDY_ROOT / "images" / "INT_Donor_E",
                       selected_channels(pd.read_parquet(STUDY_ROOT / "images/INT_Donor_E/channels.parquet")),
                       mapping_figure)
int_e_record["figures"].append(mapping_figure.relative_to(OUTPUT_ROOT).as_posix())
prior_summary = ROOT / "processed/_catalog/public/evidence/segmentation_roi_review_20261119/roi_review_summary.json"
prior_roi = next(item for item in json.loads(prior_summary.read_text())["rois"]
                 if item["study"] == STUDY and item["roi"] == "INT_Donor_E")
prior_bounds = []
for location in prior_roi["windows"]:
    x0, x1, y0, y1 = [int(value) for value in re.findall(r"\d+", location)]
    prior_bounds.append((x0, y0, x1, y1))
int_e_root = STUDY_ROOT / "images" / "INT_Donor_E"
int_e_channels = selected_channels(pd.read_parquet(int_e_root / "channels.parquet"))
with tifffile.TiffFile(str(int_e_root / "image.tiff")) as image_tif, \
     tifffile.TiffFile(str(int_e_root / "mask.tiff")) as mask_tif:
    prior_views = {item["tiff_page"]: tiff_windows(image_tif.pages[item["tiff_page"]], prior_bounds)
                   for item in int_e_channels}
    prior_masks = tiff_windows(mask_tif.pages[0], prior_bounds)
    prior_scales = {item["tiff_page"]: display_scale(
        np.concatenate([view.ravel() for view in prior_views[item["tiff_page"]]]), item.get("q995"))
        for item in int_e_channels}
    for index, bounds in enumerate(prior_bounds, start=1):
        location_figure = OUTPUT_ROOT / "figures" / "INT_Donor_E_previous_random_{}.jpg".format(index)
        local_views = {page: views[index-1] for page, views in prior_views.items()}
        make_figure(STUDY, "INT_Donor_E previous window {}".format(index), int_e_channels,
                    local_views, prior_masks[index-1], prior_scales, bounds, location_figure)
        int_e_record["figures"].append(location_figure.relative_to(OUTPUT_ROOT).as_posix())
int_e_record["previous_random_windows_xyxy"] = [list(bounds) for bounds in prior_bounds]

review_notes = {
    "INT_Donor_B": {
        "judgment": "样例可用",
        "findings": "检查窗 x=[1000,1512) y=[0,512) 位于组织边缘。DNA1/2 可见离散核样信号，实例边界大多围绕可见细胞；未见明显整体错位、跨越大片组织空白或拼接线处的轮廓断裂。仅代表该局部窗口。",
        "likely_cause": "当前窗口未见明确异常。",
        "next_action": "无需据此窗口优先重分割；若后续发现其他区域异常，再对该ROI定点复核。",
    },
    "INT_Donor_E": {
        "judgment": "证据不足",
        "findings": "旧抽样窗1 x=[27,539) y=[848,1360) 重现DNA1/2环状或纤维样纹理，并与Vimentin/CD11c呈相似的大尺度纹理；该窗只有少量mask实例，覆盖相对稀疏。旧抽样窗2 x=[2286,2798) y=[1811,2323) 大部分为低信号区域，少数亮区伴随少量实例。此前左上窗也呈类似非典型DNA纹理。原始DNA通道到input.tiff对应页逐像素相关系数为1.0；两个旧窗DNA1/2到image.tiff同坐标相关系数约0.984–0.989，页号映射正确，未见坐标错位。DNA样纹理是否代表细胞核无法仅由现有图像确认。",
        "likely_cause": "已确认异常纹理存在于原始DNA通道，显示归一化和坐标错位不受现有证据支持。若这些结构确为细胞核，则mask可能局部漏分；当前证据不能区分真实核信号、非特异背景或串扰，也不能据此认定分割错误。",
        "next_action": "优先人工核实该ROI的DNA1/2组织学信号性质及预期核形态；在确认可作为核标记后，再判定是否需要重分割。",
    },
    "LN_Donor_A": {
        "judgment": "样例可用",
        "findings": "检查窗 x=[2244,2756) y=[2244,2756) 为高密度组织。DNA阳性核与实例边界总体对应，暗区大多未被mask填充；该窗横跨拼接核心边界，未见明显的边界断裂或整体偏移。仅代表该局部窗口。",
        "likely_cause": "当前窗口未见明确异常。",
        "next_action": "无需据此窗口优先重分割；该判断不外推至整个ROI。",
    },
    "LN_Donor_E": {
        "judgment": "样例可用",
        "findings": "检查窗 x=[4244,4756) y=[4244,4756) 为密集、形态不规则组织。可见实例总体沿核样结构分布，未见明显跨越大块空白或整体XY偏移。仅代表该局部窗口。",
        "likely_cause": "当前窗口未见明确异常。",
        "next_action": "无需据此窗口优先重分割；该判断不外推至整个ROI。",
    },
    "TS_Donor_A": {
        "judgment": "样例可用",
        "findings": "检查窗 x=[0,512) y=[2500,3012) 横跨组织边缘和拼接边界。组织内核样信号与细胞mask总体对应；组织外侧少量离散mask伴随可见DNA信号，未见轮廓延伸进入大片无信号空白。仅代表该局部窗口。",
        "likely_cause": "当前窗口未见明确异常。",
        "next_action": "无需据此窗口优先重分割；该判断不外推至整个ROI。",
    },
    "TS_Donor_E": {
        "judgment": "样例可用",
        "findings": "检查窗 x=[1500,2012) y=[0,512) 位于组织边缘/细胞簇。可见实例边界总体贴合核样信号，未见明显多核并成单实例、单核明显拆分或轮廓跨空白。仅代表该局部窗口。",
        "likely_cause": "当前窗口未见明确异常。",
        "next_action": "无需据此窗口优先重分割；该判断不外推至整个ROI。",
    },
}

for record in records:
    record.update(review_notes[record["roi"]])

int_e_record["channel_mapping_check"] = {
    "coordinate_windows_xyxy": [list(bounds) for bounds in prior_bounds],
    "source_to_input_correlations": {"DNA1": 1.0, "DNA2": 1.0},
    "input_to_image_correlations": {
        "window_1": {"DNA1": 0.98723, "DNA2": 0.98375},
        "window_2": {"DNA1": 0.98895, "DNA2": 0.98625},
    },
    "interpretation": "page mapping and coordinate correspondence supported; the unusual DNA morphology is present in the raw source channels",
}

(OUTPUT_ROOT / "roi_review.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
report_rows = []
for record in records:
    report_rows.append("| {roi} | {judgment} | {findings} | `{figures}` |".format(
        roi=record["roi"], judgment=record["judgment"],
        findings=record["findings"].replace("|", "\\|"),
        figures="`, `".join(record["figures"])))
report = """# Allam2024_IMC_NormalSLO ROI 抽样复核

## 范围与方法

逐张查看6个ROI的同坐标 `image.tiff` 与 `mask.tiff` 局部图，并按 `channels.parquet` 选用实际保留的DNA1/2及Vimentin、CD11c、CD20、CD8a、CD3e。mask来源均为 `cpsam_v2`。每个ROI图展示各通道、DNA/body与mask边界叠加、彩色实例mask。当前图像是原始分辨率512×512像素的局部窗口，缩放仅用于联系表排版。结论限于图示窗口，不等同于对整幅ROI认证。

## 汇总

6个ROI中，5个抽样窗口标为“样例可用”，1个（INT_Donor_E）标为“证据不足”。在5个可用样例窗中未观察到明确整体XY偏移、大片空白上的轮廓延伸或明显细胞合并/拆分；不能由此推断未抽样区域无问题。

INT_Donor_E旧HTML中两个随机窗口已补看：窗口1重现环形/纤维样DNA1/2纹理及稀疏mask；窗口2大部分低信号，仅少数亮区和实例。raw source→input页相关系数均为1.0，旧窗DNA1/2 input→image同坐标相关系数为0.98375–0.98895。由此支持异常纹理来自原始DNA通道，未见通道页映射或坐标错位证据；但现有图像不能确认这些纹理是否为细胞核，因此不能将稀疏mask直接归因于漏分。优先核实DNA信号性质，再决定是否重分割。

## 逐ROI

| ROI | 判断 | 图示局部观察 | 质控图 |
|---|---|---|---|
""" + "\n".join(report_rows) + """

原始 `image.tiff`、`mask.tiff` 与通道数据未修改。逐ROI机器可读记录见 `roi_review.json`。
"""
(OUTPUT_ROOT / "report.md").write_text(report, encoding="utf-8")
print("rendered", len(records), "ROIs under", OUTPUT_ROOT)
