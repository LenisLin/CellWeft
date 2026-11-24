"""Render replacement evidence windows for eight low-signal Iglesia crops."""

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
FIGURE_DIR = OUTPUT_ROOT / "figures" / "reselected"
SIZE = 512
PANEL_SIZE = 480

WINDOWS = {
    "HT565B1-H2__20230427": (8333, 6944),
    "HT339B1-H1A1__20220302": (7136, 7136),
    "HT171B1-S1H9A1-A4_left__20230526": (9439, 6542),
    "HT243B1-S1H4A4_left__20230526": (8693, 8193),
    "HT271B1-S1H6A5_middle__20230526": (2744, 4939),
    "HT308B1-S1H5A4_left__20230526": (4395, 7342),
    "HT308B1-S1H5A4_right__20230526": (4895, 8042),
    "HT339B2-H2_right__20230701": (2846, 6642),
}

spec = importlib.util.spec_from_file_location(
    "segmentation_roi_gallery",
    PROJECT_ROOT / "scripts/experiments/segmentation_roi_review/01_generate_html.py",
)
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


def scale_channel(array: np.ndarray, q995: object = None) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.size and float(values.max()) <= 1.05:
        scale = 1.0
    elif q995 is not None and not pd.isna(q995) and float(q995) > 0:
        scale = float(q995)
    else:
        scale = max(float(np.quantile(values, 0.995)), 1e-6)
    return np.clip(values / scale, 0, 1)


def boundary(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return edge


def render(roi: str, x: int, y: int) -> dict:
    roi_dir = STUDY_ROOT / "images" / roi
    channels = pd.read_parquet(roi_dir / "channels.parquet")
    metadata = json.loads((roi_dir / "metadata.json").read_text(encoding="utf-8"))
    groups = metadata["segmentation_task"]["signal"]["groups"]
    nuclear_pages = {int(value) for value in groups.get("nuclear", [])}
    body_pages = {int(value) for value in groups.get("body", [])}
    channels = channels[channels["tiff_page"].notna()].copy()
    channels["tiff_page"] = channels["tiff_page"].astype(int)
    channels = channels[
        channels["keep_channel"].fillna(False).astype(bool)
        & channels["tiff_page"].isin(nuclear_pages | body_pages)
    ].sort_values("tiff_page")
    bounds = (x, y, x + SIZE, y + SIZE)

    with tifffile.TiffFile(str(roi_dir / "image.tiff")) as image_tif, \
            tifffile.TiffFile(str(roi_dir / "mask.tiff")) as mask_tif:
        image_h, image_w = image_tif.pages[0].shape
        if bounds[2] > image_w or bounds[3] > image_h:
            raise ValueError("Replacement window outside ROI: {} {}".format(roi, bounds))
        crops = []
        for _, channel in channels.iterrows():
            crop = gallery.tiff_windows(image_tif.pages[int(channel["tiff_page"])], [bounds])[0]
            crops.append((channel, scale_channel(crop, channel.get("q995"))))
        mask = gallery.tiff_windows(mask_tif.pages[0], [bounds])[0].astype(np.uint32)

    raw = {}
    with tifffile.TiffFile(str(roi_dir / "input.tiff")) as input_tif:
        for marker in ("DAPI", "PanCK"):
            selected = channels[channels["marker_id"].astype(str).eq(marker)]
            if selected.empty:
                continue
            channel = selected.iloc[0]
            source = gallery.tiff_windows(
                input_tif.pages[int(channel["output_channel_index"])], [bounds]
            )[0]
            raw[marker] = scale_channel(source, channel.get("q995"))

    nucleus = next(image for channel, image in crops
                   if int(channel["tiff_page"]) in nuclear_pages)
    body = [(channel, image) for channel, image in crops
            if int(channel["tiff_page"]) in body_pages]
    panck = next((image for channel, image in body
                  if str(channel["marker_id"]) == "PanCK"), None)
    other_body = [image for channel, image in body
                  if str(channel["marker_id"]) != "PanCK"]
    body_composite = np.maximum.reduce(other_body) if other_body else np.zeros_like(nucleus)
    if panck is None:
        panck = body_composite
    overlay = np.zeros(nucleus.shape + (3,), dtype=np.uint8)
    overlay[:, :, 0] = (panck * 255).astype(np.uint8)
    overlay[:, :, 1] = (body_composite * 190).astype(np.uint8)
    overlay[:, :, 2] = (nucleus * 255).astype(np.uint8)
    overlay[boundary(mask)] = (255, 255, 255)

    panels = [(image, "{} | page {}".format(ch["marker_id"], ch["tiff_page"]))
              for ch, image in crops]
    panels.extend((image, "input.tiff {} / same q995".format(marker))
                  for marker, image in raw.items())
    panels.extend([
        (overlay, "DAPI / PanCK / body + mask contours"),
        (gallery.colorize_mask(mask), "Instance mask"),
    ])
    columns = 3
    rows = int(math.ceil(len(panels) / float(columns)))
    gap, margin, header, row_h = 18, 30, 90, PANEL_SIZE + 50
    canvas = Image.new(
        "RGB",
        (margin * 2 + columns * PANEL_SIZE + (columns - 1) * gap,
         header + rows * row_h + margin),
        (17, 23, 28),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        face = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        title_face = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    except OSError:
        face = title_face = ImageFont.load_default()
    draw.text((margin, 14), "{} | {}".format(STUDY, roi), fill="white", font=title_face)
    pixel_size = float(metadata["pixel_size_um"][0])
    draw.text((margin, 52), "reselected x=[{}, {}) y=[{}, {}) | {:.6f} um/px".format(
        bounds[0], bounds[2], bounds[1], bounds[3], pixel_size), fill=(196, 207, 212), font=face)
    for index, (panel, label) in enumerate(panels):
        col, row = index % columns, index // columns
        px, py = margin + col * (PANEL_SIZE + gap), header + row * row_h
        draw.text((px, py), label[:60], fill=(237, 241, 243), font=face)
        values = np.asarray(panel)
        if values.ndim == 2:
            values = np.repeat((np.clip(values, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode="RGB")
        canvas.paste(image.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS), (px, py + 24))

    figure = FIGURE_DIR / (roi + ".jpg")
    figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(figure), format="JPEG", quality=94, optimize=True)
    dapi = nucleus.astype(np.float32)
    return {
        "study": STUDY,
        "roi": roi,
        "mask_origin": metadata.get("mask_origin", "unknown"),
        "window_xyxy": list(bounds),
        "window_dapi_fraction_gt_0_2": float(np.mean(dapi > 0.2)),
        "window_mask_foreground_fraction": float(np.mean(mask > 0)),
        "figures": [figure.relative_to(OUTPUT_ROOT).as_posix()],
    }


records = [render(roi, x, y) for roi, (x, y) in WINDOWS.items()]
(OUTPUT_ROOT / "replacement_windows.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("rendered {} replacement windows".format(len(records)))
