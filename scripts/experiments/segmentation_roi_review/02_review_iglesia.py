"""Create matched crops for visual review of all Iglesia CODEX breast ROIs.

Inputs: current public Iglesia images.parquet and each complete ROI's
channels.parquet, metadata.json, image.tiff, input.tiff, and mask.tiff.
Outputs: one full-resolution channel/mask comparison per ROI and a pending
review record under the NAS segmentation-followup evidence directory.
Upstream: segmentation ROI gallery generated from manifest 20261119_v2.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/home/lenislin/Experiment/projects/CellWeft")
NAS_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Iglesia2024_CODEX_BreastCancer"
STUDY_ROOT = NAS_ROOT / "processed" / STUDY
OUTPUT_ROOT = NAS_ROOT / "processed/_catalog/public/evidence/segmentation_followup_20261119" / STUDY
FIGURE_DIR = OUTPUT_ROOT / "figures"
CROP_SIZE = 512
PANEL_SIZE = 480
PANEL_COLUMNS = 3
SEED = 20261119


# Reuse the bounded TIFF-tile reader and instance color mapping from the gallery.
spec = importlib.util.spec_from_file_location(
    "segmentation_roi_gallery",
    PROJECT_ROOT / "scripts/experiments/segmentation_roi_review/01_generate_html.py",
)
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


def display(array: np.ndarray, q995: object = None) -> np.ndarray:
    """Scale one matched channel crop using its existing image normalization."""
    values = np.asarray(array, dtype=np.float32)
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum <= 1.05:
        scale = 1.0
    elif q995 is not None and not pd.isna(q995) and float(q995) > 0:
        scale = float(q995)
    else:
        scale = max(float(np.quantile(values, 0.995)), 1e-6)
    return np.clip(values / max(scale, 1e-6), 0, 1)


def tissue_window(page, mask_page, q995: object, height: int, width: int):
    """Choose an informative crop using DAPI and mask foreground only."""
    size = min(CROP_SIZE, height, width)
    max_x, max_y = width - size, height - size
    positions = sorted(set(int(round(i * max_x / 4)) for i in range(5)))
    x_positions = positions
    y_positions = sorted(set(int(round(i * max_y / 4)) for i in range(5)))
    bounds = [(x, y, x + size, y + size) for y in y_positions for x in x_positions]
    crops = gallery.tiff_windows(page, bounds)
    mask_crops = gallery.tiff_windows(mask_page, bounds)
    if float(np.max(crops[0])) <= 1.05:
        scale = 1.0
    elif q995 is not None and not pd.isna(q995) and float(q995) > 0:
        scale = float(q995)
    else:
        scale = max(float(np.quantile(crops[0], 0.995)), 1e-6)
    threshold = max(scale * 0.05, 0.005)
    dapi_fractions = [float(np.count_nonzero(crop > threshold)) / float(size * size)
                      for crop in crops]
    mask_fractions = [float(np.count_nonzero(crop)) / float(size * size)
                      for crop in mask_crops]
    scores = [dapi + mask for dapi, mask in zip(dapi_fractions, mask_fractions)]
    index = int(np.argmax(scores))
    return bounds[index], dapi_fractions[index], mask_fractions[index]


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def labeled_panel(panel: np.ndarray, label: str) -> tuple[np.ndarray, str]:
    return panel, label


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel boundary for the local instance-label crop."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def render_roi(roi: str, row: pd.Series) -> dict:
    roi_dir = STUDY_ROOT / "images" / roi
    channel_table = pd.read_parquet(roi_dir / "channels.parquet")
    metadata = json.loads((roi_dir / "metadata.json").read_text(encoding="utf-8"))
    groups = metadata["segmentation_task"]["signal"]["groups"]
    nuclear_pages = {int(page) for page in groups.get("nuclear", [])}
    body_pages = {int(page) for page in groups.get("body", [])}
    channel_table = channel_table[channel_table["tiff_page"].notna()].copy()
    channel_table["tiff_page"] = channel_table["tiff_page"].astype(int)
    selected = channel_table[
        channel_table["keep_channel"].fillna(False).astype(bool)
        & channel_table["tiff_page"].isin(nuclear_pages | body_pages)
    ].copy()
    selected["tiff_page"] = selected["tiff_page"].astype(int)
    selected = selected.sort_values("tiff_page")
    nuclei = selected[selected["tiff_page"].isin(nuclear_pages)]
    if nuclei.empty:
        raise ValueError("No recorded nuclear input for {}".format(roi))
    dapi = nuclei.iloc[0]

    with tifffile.TiffFile(str(roi_dir / "image.tiff")) as image_tif, \
            tifffile.TiffFile(str(roi_dir / "mask.tiff")) as mask_tif:
        height, width = image_tif.pages[0].shape
        bounds, dapi_fraction, mask_fraction = tissue_window(
            image_tif.pages[int(dapi["tiff_page"])], mask_tif.pages[0],
            dapi.get("q995"), height, width
        )
        channels = []
        for _, channel in selected.iterrows():
            page_index = int(channel["tiff_page"])
            crop = gallery.tiff_windows(image_tif.pages[page_index], [bounds])[0]
            channels.append((channel, crop, display(crop, channel.get("q995"))))
        mask = gallery.tiff_windows(mask_tif.pages[0], [bounds])[0].astype(np.uint32)

    with tifffile.TiffFile(str(roi_dir / "input.tiff")) as input_tif:
        raw_crops = {}
        for marker in ("DAPI", "PanCK"):
            marker_rows = selected[selected["marker_id"].astype(str).eq(marker)]
            if marker_rows.empty:
                continue
            channel = marker_rows.iloc[0]
            page_index = int(channel["output_channel_index"])
            raw = gallery.tiff_windows(input_tif.pages[page_index], [bounds])[0]
            raw_crops[marker] = display(raw, channel.get("q995"))

    nuclear_image = channels[next(i for i, (ch, _, _) in enumerate(channels)
                                  if int(ch["tiff_page"]) in nuclear_pages)][2]
    body_items = [(ch, image) for ch, _, image in channels
                  if int(ch["tiff_page"]) in body_pages]
    panck = next((image for ch, image in body_items if str(ch["marker_id"]) == "PanCK"), None)
    other_body = [image for ch, image in body_items if str(ch["marker_id"]) != "PanCK"]
    body_composite = np.maximum.reduce(other_body) if other_body else np.zeros_like(nuclear_image)
    if panck is None:
        panck = body_composite
    overlay = np.zeros(nuclear_image.shape + (3,), dtype=np.uint8)
    overlay[:, :, 0] = (panck * 255).astype(np.uint8)
    overlay[:, :, 1] = (body_composite * 190).astype(np.uint8)
    overlay[:, :, 2] = (nuclear_image * 255).astype(np.uint8)
    overlay[mask_boundary(mask)] = (255, 255, 255)

    panels = [labeled_panel(image, "{} | page {} | {}".format(
        channel["marker_id"], channel["tiff_page"],
        "nuclear" if int(channel["tiff_page"]) in nuclear_pages else "body"))
        for channel, _, image in channels]
    for marker, image in raw_crops.items():
        panels.append(labeled_panel(image, "input.tiff {} / same q995".format(marker)))
    panels.extend([
        labeled_panel(overlay, "DAPI / PanCK / body composite + mask contours"),
        labeled_panel(gallery.colorize_mask(mask), "Instance mask"),
    ])

    rows = int(math.ceil(len(panels) / float(PANEL_COLUMNS)))
    gap = 18
    margin = 30
    canvas_width = margin * 2 + PANEL_COLUMNS * PANEL_SIZE + (PANEL_COLUMNS - 1) * gap
    header_height = 90
    row_height = PANEL_SIZE + 50
    canvas_height = header_height + rows * row_height + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (17, 23, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), "{} | {}".format(STUDY, roi), fill=(245, 248, 249), font=font(23))
    pixel_size = float(row["pixel_size_x_um"])
    location = "x=[{}, {}) y=[{}, {}) | {:.6f} um/px".format(bounds[0], bounds[2], bounds[1], bounds[3], pixel_size)
    draw.text((margin, 52), location, fill=(196, 207, 212), font=font(16))
    for index, (panel, label) in enumerate(panels):
        column, panel_row = index % PANEL_COLUMNS, index // PANEL_COLUMNS
        x = margin + column * (PANEL_SIZE + gap)
        y = header_height + panel_row * row_height
        draw.text((x, y), str(label)[:60], fill=(237, 241, 243), font=font(15))
        values = np.asarray(panel)
        if values.ndim == 2:
            values = np.repeat((np.clip(values, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode="RGB")
        image = image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y + 24))

    figure_path = FIGURE_DIR / (roi.replace("/", "_") + ".jpg")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(figure_path), format="JPEG", quality=94, optimize=True)
    mask_check = metadata.get("mask_check", {})
    return {
        "study": STUDY,
        "roi": roi,
        "panel": str(row["panel_id"]),
        "mask_origin": metadata.get("mask_origin", "unknown"),
        "mask_type": mask_check.get("mask_type", "unknown"),
        "judgment": "证据不足",
        "findings": ["待逐图复核；当前记录仅描述裁片和通道来源。"],
        "likely_cause": None,
        "next_action": "逐图检查通道清晰度、同坐标关系和实例轮廓。",
        "pixel_size_um": pixel_size,
        "window_xyxy": list(bounds),
        "window_dapi_signal_fraction": float(dapi_fraction),
        "window_mask_foreground_fraction": float(mask_fraction),
        "channels": [
            {
                "marker_id": str(channel["marker_id"]),
                "source_name": str(channel["source_name"]),
                "tiff_page": int(channel["tiff_page"]),
                "source_channel_index": int(channel["source_channel_index"]),
                "output_channel_index": int(channel["output_channel_index"]),
                "role": "nuclear" if int(channel["tiff_page"]) in nuclear_pages else "body",
            }
            for channel, _, _ in channels
        ],
        "figures": [figure_path.relative_to(OUTPUT_ROOT).as_posix()],
    }


# Inventory exact current study members, then render matched signal and mask crops.
images = pd.read_parquet(STUDY_ROOT / "images.parquet")
records = []
for _, image_row in images.iloc[:24].iterrows():
    records.append(render_roi(str(image_row["image_id"]), image_row))

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "roi_review_first24.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(OUTPUT_ROOT / "report_first24.md").write_text(
    "# Iglesia2024_CODEX_BreastCancer\n\n"
    "48个ROI的逐ROI通道与mask证据图已生成。图像使用同一512×512 XY窗口，"
    "窗口由DAPI信号与mask前景共同用于定位，不作为质量判定。图中包含 segmentation_task 标记的核/胞质通道、"
    "对应 input.tiff 的DAPI/PanCK信号、mask轮廓叠加及实例mask。\n\n"
    "本文件对应前24个ROI的原生512×512裁片。最终判断需逐图复核并合并后续24个ROI。\n",
    encoding="utf-8",
)
print("rendered {} ROI figures under {}".format(len(records), OUTPUT_ROOT))
