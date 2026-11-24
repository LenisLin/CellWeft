"""Review Lyman2025_IMC_PanIN channel and mask resolution.

Inputs: current processed images.parquet, per-ROI image/input/mask TIFFs,
channels.parquet, metadata.json, and source MCD headers.
Outputs: per-ROI evidence figures, contact sheets, and roi_review.json under
the study follow-up evidence directory.
Upstream: public manifest 20261119_v2 and existing cPSAM segmentation assets.
"""

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from imctools.io.mcd.mcdparser import McdParser


NAS_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Lyman2025_IMC_PanIN"
PROCESSED = NAS_ROOT / "processed" / STUDY
RAW_ROOT = NAS_ROOT / "raw/public/Lyman2025_IMC_PanIN/zenodo_8336719"
OUTPUT = NAS_ROOT / "processed/_catalog/public/evidence/segmentation_followup_20261119" / STUDY
HELPER_PATH = Path(__file__).with_name("01_generate_html.py")
WINDOW_SIZE = 256
PANEL_SIZE = 320
SHEET_PANEL_SIZE = 256

spec = importlib.util.spec_from_file_location("segmentation_review_helper", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
tiff_windows = helper.tiff_windows
segment_plane = helper.segment_plane


def read_source_dimensions() -> Dict[Tuple[str, int], dict]:
    """Read native acquisition bounds from each MCD metadata header."""
    result = {}
    for path in sorted(RAW_ROOT.glob("*.mcd")):
        with McdParser(path) as parser:
            for acquisition_id, acquisition in parser.session.acquisitions.items():
                result[(path.name, int(acquisition_id))] = {
                    "width": int(acquisition.max_x), "height": int(acquisition.max_y),
                    "name": acquisition.metaname, "pixel_size_x_um": acquisition.ablation_distance_between_shots_x,
                    "pixel_size_y_um": acquisition.ablation_distance_between_shots_y,
                }
    return result


def channel_scale(values: np.ndarray) -> float:
    """Use the stored shared-image scale, with a crop-derived display fallback."""
    values = np.asarray(values, dtype=np.float32)
    maximum = float(np.max(values)) if values.size else 1.0
    if maximum <= 1.05:
        return 1.0
    return max(float(np.quantile(values, 0.995)), 1e-6)


def to_gray(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32) / max(scale, 1e-6), 0.0, 1.0)


def best_signal_window(nuclear: np.ndarray, height: int, width: int) -> Tuple[int, int, int, int]:
    """Choose a 256-pixel window with the strongest available nuclear signal."""
    size = min(WINDOW_SIZE, height, width)
    if size == height and size == width:
        return (0, 0, width, height)
    stride = max(1, size // 8)
    y_starts = list(range(0, max(1, height - size + 1), stride)) + [height - size]
    x_starts = list(range(0, max(1, width - size + 1), stride)) + [width - size]
    integral = np.pad(np.asarray(nuclear, dtype=np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best = (0, 0)
    best_score = -1.0
    for y0 in sorted(set(y_starts)):
        y1 = y0 + size
        for x0 in sorted(set(x_starts)):
            x1 = x0 + size
            score = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            if score > best_score:
                best, best_score = (x0, y0), score
    return (best[0], best[1], best[0] + size, best[1] + size)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    labels, inverse = np.unique(mask.astype(np.uint32), return_inverse=True)
    values = labels.astype(np.uint64)
    hashed = values * np.uint64(2654435761) + np.uint64(2246822519)
    colors = np.stack((72 + ((hashed >> 8) % 184),
                       72 + ((hashed >> 17) % 184),
                       72 + ((hashed >> 25) % 184)), axis=1).astype(np.uint8)
    colors[labels == 0] = 0
    return colors[inverse].reshape(mask.shape + (3,))


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def signal_overlay(nuclear: np.ndarray, body: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    rgb = np.zeros(nuclear.shape + (3,), dtype=np.uint8)
    rgb[:, :, 0] = (body * 230).astype(np.uint8)
    rgb[:, :, 1] = np.minimum(255, body * 120 + nuclear * 70).astype(np.uint8)
    rgb[:, :, 2] = (nuclear * 255).astype(np.uint8)
    if mask is not None:
        rgb[mask_boundary(mask)] = (255, 235, 45)
    return rgb


def fit_panel(array: np.ndarray, panel_size: int, nearest: bool = True) -> Image.Image:
    image = Image.fromarray(array)
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize((panel_size, panel_size), resampling)


def paste_panel(canvas: Image.Image, x: int, y: int, array: np.ndarray,
                label: str, panel_size: int, font: ImageFont.ImageFont) -> None:
    if array.ndim == 2:
        gray = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        array = np.repeat(gray[:, :, None], 3, axis=2)
    panel = fit_panel(np.asarray(array, dtype=np.uint8), panel_size, nearest=True)
    canvas.paste(panel, (x, y + 24))
    ImageDraw.Draw(canvas).text((x, y), label[:40], fill=(242, 246, 248), font=font)


def source_file_name(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return Path(str(value[0])).name
    return Path(str(value)).name


def render_roi(row: pd.Series, source_dimensions: Dict[Tuple[str, int], dict]) -> dict:
    roi = str(row.image_id)
    roi_dir = PROCESSED / "images" / roi
    channels = pd.read_parquet(roi_dir / "channels.parquet")
    metadata = json.loads((roi_dir / "metadata.json").read_text(encoding="utf-8"))
    signal_groups = metadata["segmentation_task"]["signal"]["groups"]
    nuclear_pages = [int(page) for page in signal_groups["nuclear"]]
    body_pages = [int(page) for page in signal_groups["body"]]
    selected_pages = nuclear_pages + body_pages
    page_rows = {int(channel.tiff_page): channel for channel in channels.itertuples()
                 if not pd.isna(channel.tiff_page) and int(channel.tiff_page) in selected_pages}

    with tifffile.TiffFile(roi_dir / "image.tiff") as image_tif, \
            tifffile.TiffFile(roi_dir / "input.tiff") as input_tif, \
            tifffile.TiffFile(roi_dir / "mask.tiff") as mask_tif:
        height, width = image_tif.pages[0].shape[-2:]
        full_bounds = (0, 0, width, height)
        full_planes = {
            page: tiff_windows(image_tif.pages[page], [full_bounds])[0]
            for page in selected_pages
        }
        scales = {page: channel_scale(full_planes[page]) for page in selected_pages}
        normalized = {page: to_gray(full_planes[page], scales[page]) for page in selected_pages}
        nuclear = np.maximum.reduce([normalized[page] for page in nuclear_pages])
        body = np.maximum.reduce([normalized[page] for page in body_pages])
        mask = tiff_windows(mask_tif.pages[0], [full_bounds])[0].astype(np.uint32)
        bounds = best_signal_window(nuclear, height, width)
        x0, y0, x1, y1 = bounds
        crop = np.s_[y0:y1, x0:x1]
        crop_planes = {page: normalized[page][crop] for page in selected_pages}
        crop_nuclear, crop_body, crop_mask = nuclear[crop], body[crop], mask[crop]
        crop_overlay = signal_overlay(crop_nuclear, crop_body, crop_mask)
        crop_mask_rgb = colorize_mask(crop_mask)

        full_overlay = signal_overlay(nuclear, body, mask)
        full_mask_rgb = colorize_mask(mask)
        ratio = min(512.0 / width, 512.0 / height)
        overview_size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
        overview_filter = Image.Resampling.NEAREST if min(width, height) <= 64 else Image.Resampling.LANCZOS
        overview = Image.fromarray(full_overlay, mode="RGB").resize(overview_size, overview_filter)
        overview_mask = Image.fromarray(full_mask_rgb, mode="RGB").resize(overview_size, overview_filter)
        draw = ImageDraw.Draw(overview)
        bx0, by0 = int(x0 * ratio), int(y0 * ratio)
        bx1, by1 = int(x1 * ratio), int(y1 * ratio)
        draw.rectangle((bx0, by0, bx1, by1), outline=(255, 245, 45), width=2)

        input_image_error = 0.0
        input_image_error_max = 0.0
        for page, channel in page_rows.items():
            raw_q995 = float(channel.q995) if not pd.isna(channel.q995) else channel_scale(
                tiff_windows(input_tif.pages[int(channel.output_channel_index)], [full_bounds])[0])
            input_plane = tiff_windows(input_tif.pages[int(channel.output_channel_index)], [full_bounds])[0]
            expected = np.clip(input_plane.astype(np.float32) / max(raw_q995, 1e-6), 0.0, 1.0)
            delta = np.abs(expected - full_planes[page].astype(np.float32))
            input_image_error = max(input_image_error, float(delta.mean()))
            input_image_error_max = max(input_image_error_max, float(delta.max()))

    # Render the full acquisition context plus all actual segmentation channels at one matched crop.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 21)
        label_font = ImageFont.truetype(font_path, 14)
        small_font = ImageFont.truetype(font_path, 13)
    except OSError:
        title_font = label_font = small_font = ImageFont.load_default()
    columns = 4
    panel_size = PANEL_SIZE
    panel_count = len(selected_pages) + 3
    rows = int(math.ceil(panel_count / float(columns)))
    canvas_width = 56 + columns * panel_size + (columns - 1) * 14
    row_height = panel_size + 42
    canvas_height = 820 + rows * row_height + 24
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 24, 28))
    draw = ImageDraw.Draw(canvas)
    source_file = source_file_name(row.source_files)
    acquisition = int(row.source_acquisition_id)
    native = source_dimensions.get((source_file, acquisition))
    native_text = "{}x{} source".format(native["width"], native["height"]) if native else "source geometry unavailable"
    draw.text((28, 12), "{} | {} | final {}x{} | {} | {}".format(
        STUDY, roi, width, height, native_text, metadata.get("mask_origin", "unknown")),
        fill=(246, 249, 250), font=title_font)
    draw.text((28, 40), "Local window x=[{},{}) y=[{},{}) | display uses nearest-neighbor for detail".format(
        x0, x1, y0, y1), fill=(190, 202, 207), font=small_font)

    # Full-resolution overview panels preserve acquisition context.
    overview_y = 68
    draw.text((28, overview_y), "Whole ROI: nuclear (blue) + body (red); mask contours (yellow)",
              fill=(235, 240, 242), font=label_font)
    overview_x = 28
    canvas.paste(overview, (overview_x, overview_y + 22))
    canvas.paste(overview_mask, (overview_x + overview.width + 24, overview_y + 22))
    draw.text((overview_x + overview.width + 24, overview_y), "Whole ROI instance mask",
              fill=(235, 240, 242), font=label_font)
    local_y = 68 + max(overview.height, overview_mask.height) + 48
    draw.text((28, local_y), "Matched local channel and mask views", fill=(235, 240, 242), font=label_font)
    panels = []
    for page in selected_pages:
        channel = page_rows[page]
        marker = str(channel.marker_id) if not pd.isna(channel.marker_id) else str(channel.source_name)
        role = "DNA" if page in nuclear_pages else "body"
        panels.append((crop_planes[page], "{} | {} | page {}".format(role, marker, page)))
    panels.extend([(crop_overlay, "Nuclear/body + mask contours"), (crop_mask_rgb, "Instance mask")])
    for index, (panel, label) in enumerate(panels):
        col, panel_row = index % columns, index // columns
        paste_panel(canvas, 28 + col * (panel_size + 14), local_y + 22 + panel_row * row_height,
                    panel, label, panel_size, label_font)
    draw.text((28, canvas_height - 20), "mask labels: {} | mask foreground: {:.1%} | input→image max abs error: {:.2g}".format(
        len(np.unique(mask)) - int(0 in mask), float(np.count_nonzero(mask)) / float(mask.size), input_image_error_max),
        fill=(180, 192, 198), font=small_font)

    OUTPUT.joinpath("assets").mkdir(parents=True, exist_ok=True)
    figure = "assets/{}.jpg".format(roi)
    canvas.save(OUTPUT / figure, format="JPEG", quality=93, optimize=True)

    # A compact contact-sheet view keeps the same source-coordinate crop across channels and mask.
    key_pages = [nuclear_pages[0], nuclear_pages[1],
                 next(page for page in body_pages if str(page_rows[page].marker_id) == "Cytokeratin"),
                 next(page for page in body_pages if str(page_rows[page].marker_id) == "Vimentin")]
    sheet_panels = [to_gray(crop_planes[page], 1.0) for page in key_pages]
    sheet_panels.extend([crop_overlay, crop_mask_rgb])

    # Source and output geometry separate native small acquisitions from padded square views.
    src_x0, src_y0, src_x1, src_y1 = [int(getattr(row, field)) for field in
                                      ("source_x0_px", "source_y0_px", "source_x1_px", "source_y1_px")]
    pad = {key: int(getattr(row, field)) for key, field in
           (("left", "pad_left_px"), ("top", "pad_top_px"),
            ("right", "pad_right_px"), ("bottom", "pad_bottom_px"))}
    expected_native = (int(src_y1 - src_y0), int(src_x1 - src_x0))
    input_shape = tuple(int(value) for value in input_tif_shape(roi_dir / "input.tiff"))
    source_match = bool(native and expected_native == (native["height"], native["width"]))
    mask_labels = int(len(np.unique(mask)) - int(0 in mask))
    mask_origin = str(metadata.get("mask_origin", "unknown"))
    resolution_note = ""
    if native and (native["height"] <= 64 or native["width"] <= 64):
        resolution_note = "原始 MCD acquisition 本身为{}x{} px；input/image/mask均保留该原生尺寸。".format(
            native["width"], native["height"])
    elif any(pad.values()):
        resolution_note = "源ROI为{}x{} px，当前图像按合同方形padding到{}x{}；padding={}.".format(
            expected_native[1], expected_native[0], width, height, pad)
    else:
        resolution_note = "源ROI与当前图像的有效尺寸一致。"
    return {
        "study": STUDY,
        "roi": roi,
        "mask_origin": mask_origin,
        "judgment": "待逐图确认",
        "findings": resolution_note,
        "likely_cause": "",
        "next_action": "完成核/膜信号与mask同坐标图像复核。",
        "figures": [figure],
        "contact_sheet_panels": sheet_panels,
        "window_xyxy": [x0, y0, x1, y1],
        "indexed_shape_yx": [int(row.height_px), int(row.width_px)],
        "input_shape_cyx": list(input_shape),
        "image_shape_cyx": [int(row.channel_count), height, width],
        "mask_shape_yx": [int(mask.shape[0]), int(mask.shape[1])],
        "native_mcd_shape_yx": [int(native["height"]), int(native["width"])] if native else None,
        "native_mcd_match": source_match,
        "source_mcd": source_file,
        "source_acquisition_id": acquisition,
        "source_crop_xyxy": [src_x0, src_y0, src_x1, src_y1],
        "padding_ltrb": pad,
        "pixel_size_um_xy": [float(row.pixel_size_x_um), float(row.pixel_size_y_um)],
        "mask_label_count": mask_labels,
        "mask_foreground_fraction": float(np.count_nonzero(mask)) / float(mask.size),
        "input_to_image_max_abs_error": input_image_error_max,
        "input_to_image_mean_abs_error_max_channel": input_image_error,
    }


def input_tif_shape(path: Path) -> Tuple[int, int, int]:
    with tifffile.TiffFile(path) as image:
        return (len(image.pages), int(image.pages[0].shape[-2]), int(image.pages[0].shape[-1]))


def write_contact_sheets(records: List[dict]) -> None:
    """Write 3-ROI sheets with identical signal/mask crop coordinates."""
    output_dir = OUTPUT / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = label_font = ImageFont.load_default()
    labels = ["191Ir", "193Ir", "Cytokeratin", "Vimentin", "Overlay + contours", "Instance mask"]
    columns = len(labels)
    row_height = SHEET_PANEL_SIZE + 42
    sheet_width = 32 + columns * SHEET_PANEL_SIZE + (columns - 1) * 8
    for start in range(0, len(records), 3):
        group = records[start:start + 3]
        sheet = Image.new("RGB", (sheet_width, 30 + len(group) * row_height), (18, 24, 28))
        draw = ImageDraw.Draw(sheet)
        draw.text((14, 7), "Lyman ROI contact sheet {}-{}".format(start + 1, start + len(group)),
                  fill=(246, 249, 250), font=font)
        for row_index, record in enumerate(group):
            y = 30 + row_index * row_height
            draw.text((14, y), "{} | {} | source {}".format(
                record["roi"], record["indexed_shape_yx"][1], record["source_mcd"]),
                fill=(221, 231, 235), font=label_font)
            for col, (array, label) in enumerate(zip(record["contact_sheet_panels"], labels)):
                x = 14 + col * (SHEET_PANEL_SIZE + 8)
                draw.text((x, y + 18), label, fill=(235, 241, 243), font=label_font)
                panel_array = np.asarray(array)
                if panel_array.ndim == 2:
                    gray = (np.clip(panel_array, 0.0, 1.0) * 255).astype(np.uint8)
                    panel_array = np.repeat(gray[:, :, None], 3, axis=2)
                panel = fit_panel(panel_array.astype(np.uint8), SHEET_PANEL_SIZE, nearest=True)
                sheet.paste(panel, (x, y + 37))
        filename = "sheet_{:03d}_{:03d}.jpg".format(start + 1, start + len(group))
        sheet.save(output_dir / filename, format="JPEG", quality=94, optimize=True)
        for record in group:
            record["figures"].append("contact_sheets/" + filename)
            del record["contact_sheet_panels"]


def write_report(records: List[dict]) -> None:
    rows = []
    for item in records:
        rows.append("| `{roi}` | {source} {acq} | {native} | {current} | {mask} | 待逐图确认 |".format(
            roi=item["roi"], source=item["source_mcd"], acq=item["source_acquisition_id"],
            native=item["native_mcd_shape_yx"], current=item["indexed_shape_yx"],
            mask=item["mask_origin"]))
    report = """# Lyman2025_IMC_PanIN 逐 ROI 通道与 mask 复核

## 范围

- 共 {n} 个 ROI；每个 ROI 均生成同坐标通道与实例 mask 证据图。
- 通道按每图 segmentation_task.signal.groups 选取，核标记为191Ir/193Ir，body组为8个实际页。
- 证据窗口优先定位到核信号较强区域；50×50 ROI显示整个原生图像。局部面板使用最近邻放大，原始像素没有插值重采样。
- `input.tiff` 到 `image.tiff` 按 q995 归一化逐像素比较；尺度和有效视野另见 roi_review.json。
- 联系表位于 `contact_sheets/`；单 ROI 完整通道图位于 `assets/`。

## ROI 清单

| ROI | 源 acquisition | MCD 原生尺寸 Y×X | processed 尺寸 Y×X | mask 来源 | 视觉判断 |
|---|---|---:|---:|---|---|
{rows}

## 逐图结论

待完成全部联系表的逐 ROI 图像查看后填写。单个高信号窗口不足以证明整张 ROI 的 mask 均无问题。
""".format(n=len(records), rows="\n".join(rows))
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    public_records = [{key: value for key, value in item.items() if key != "contact_sheet_panels"}
                      for item in records]
    (OUTPUT / "roi_review.json").write_text(json.dumps(public_records, ensure_ascii=False, indent=2),
                                             encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(PROCESSED / "images.parquet").sort_values("image_id")
    source_dimensions = read_source_dimensions()
    records = [render_roi(row, source_dimensions) for _, row in images.iterrows()]
    write_contact_sheets(records)
    write_report(records)
    print("ROIs", len(records), "output", OUTPUT)


if __name__ == "__main__":
    main()
