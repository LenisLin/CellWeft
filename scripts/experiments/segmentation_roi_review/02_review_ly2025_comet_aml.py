"""Inspect all Ly2025 COMET ROI channels and masks at matched coordinates.

Inputs: current processed image/mask/input TIFFs, channel tables and the original
Figshare OME-TIFFs. Outputs: one local-window figure per ROI plus a JSON inventory
under NAS processed/_catalog/public/evidence/segmentation_followup_20261119/.
Upstream: public manifest 20261119_v2 and existing cpsam_v2 segmentation assets.
The workflow compares source-to-input pixels, selects two signal/mask-supported
windows, and renders the six effective segmentation channels with their masks.
"""

import json
import math
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NAS_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Ly2025_COMET_AML"
PROCESSED = NAS_ROOT / "processed" / STUDY
RAW_ROOT = NAS_ROOT / "raw/public" / STUDY / "figshare_28861841_v3"
OUTPUT = NAS_ROOT / "processed/_catalog/public/evidence/segmentation_followup_20261119" / STUDY
SEED = 20261119
CROP_SIZE = 512
PREVIEW_MAX = 720
PANEL_SIZE = 384
PANEL_COLUMNS = 4
SEGMENTATION_MARKERS = ("DAPI", "CD34", "CD68", "CD3e", "CD33", "CD45RB")


def load_crop_reader():
    """Reuse the established decoder that reads only tiles intersecting crops."""
    helper_path = PROJECT_ROOT / "scripts/experiments/segmentation_roi_review/01_generate_html.py"
    spec = importlib.util.spec_from_file_location("segmentation_review_renderer", helper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.tiff_windows, module.segment_plane


tiff_windows, segment_plane = load_crop_reader()


def tile_preview(page, step: int, segment_plane) -> np.ndarray:
    """Sample a tiled page into a small overview without materializing the full image."""
    height, width = page.shape[-2:]
    preview = np.zeros((int(math.ceil(height / float(step))),
                        int(math.ceil(width / float(step)))), dtype=page.dtype)
    for segment, position, shape in page.segments():
        if segment is None:
            continue
        tile = segment_plane(segment)
        tile_y, tile_x = int(position[2]), int(position[3])
        tile_height, tile_width = int(shape[1]), int(shape[2])
        sample_y = np.arange(((tile_y + step - 1) // step) * step,
                             min(tile_y + tile_height, height), step)
        sample_x = np.arange(((tile_x + step - 1) // step) * step,
                             min(tile_x + tile_width, width), step)
        if not len(sample_y) or not len(sample_x):
            continue
        preview[np.ix_(sample_y // step, sample_x // step)] = tile[
            np.ix_(sample_y - tile_y, sample_x - tile_x)]
    return preview


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return instance edges including the foreground/background boundary."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def frame_windows(page, bounds: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """Read selected TIFF windows while honoring frame-local tile offsets."""
    layout = getattr(page, "keyframe", page)
    page_height, page_width = page.shape[-2:]
    outputs = [np.zeros((y1 - y0, x1 - x0), dtype=page.dtype)
               for x0, y0, x1, y1 in bounds]
    if layout.is_tiled:
        tile_height, tile_width = int(layout.tilelength), int(layout.tilewidth)
        tiles_per_row = int(math.ceil(page_width / float(tile_width)))
    else:
        tile_height, tile_width = int(layout.rowsperstrip), page_width
        tiles_per_row = 1
    indices = set()
    for x0, y0, x1, y1 in bounds:
        first_row, last_row = y0 // tile_height, (y1 - 1) // tile_height
        first_col, last_col = x0 // tile_width, (x1 - 1) // tile_width
        for tile_row in range(first_row, last_row + 1):
            for tile_col in range(first_col, last_col + 1):
                indices.add(tile_row * tiles_per_row + tile_col)

    for index in sorted(indices):
        byte_count = page.databytecounts[index]
        if byte_count:
            page.parent.filehandle.seek(page.dataoffsets[index])
            raw = page.parent.filehandle.read(byte_count)
            segment, position, shape = page.decode(raw, index)
        else:
            segment, position, shape = page.decode(None, index)
        if segment is None:
            continue
        tile = segment_plane(segment)
        tile_y, tile_x = int(position[2]), int(position[3])
        decoded_height, decoded_width = int(shape[1]), int(shape[2])
        for out, (x0, y0, x1, y1) in zip(outputs, bounds):
            ix0, iy0 = max(x0, tile_x), max(y0, tile_y)
            ix1 = min(x1, tile_x + decoded_width, page_width)
            iy1 = min(y1, tile_y + decoded_height, page_height)
            if ix0 < ix1 and iy0 < iy1:
                out[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = tile[
                    iy0-tile_y:iy1-tile_y, ix0-tile_x:ix1-tile_x]
    return outputs


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map positive instance labels to stable colors and keep zero black."""
    labels, inverse = np.unique(mask.astype(np.uint32), return_inverse=True)
    values = labels.astype(np.uint64)
    hashed = values * np.uint64(2654435761) + np.uint64(2246822519)
    colors = np.stack((72 + ((hashed >> 8) % 184),
                       72 + ((hashed >> 17) % 184),
                       72 + ((hashed >> 25) % 184)), axis=1).astype(np.uint8)
    colors[labels == 0] = 0
    return colors[inverse].reshape(mask.shape + (3,))


def choose_windows(mask_preview: np.ndarray, dapi_preview: np.ndarray,
                   body_preview: np.ndarray, image_shape: Tuple[int, int],
                   step: int) -> List[dict]:
    """Choose two separated regions with mask foreground and supporting signal."""
    height, width = image_shape
    size = min(CROP_SIZE, height, width)
    max_x, max_y = width - size, height - size
    starts_x = list(range(0, max_x + 1, max(1, size // 2))) + [max_x]
    starts_y = list(range(0, max_y + 1, max(1, size // 2))) + [max_y]
    candidates = []
    for y0 in sorted(set(starts_y)):
        for x0 in sorted(set(starts_x)):
            px0, py0 = x0 // step, y0 // step
            px1 = min(mask_preview.shape[1], int(math.ceil((x0 + size) / float(step))))
            py1 = min(mask_preview.shape[0], int(math.ceil((y0 + size) / float(step))))
            mask_crop = mask_preview[py0:py1, px0:px1]
            dapi_crop = dapi_preview[py0:py1, px0:px1]
            body_crop = body_preview[py0:py1, px0:px1]
            foreground = float(np.mean(mask_crop > 0)) if mask_crop.size else 0.0
            dapi_signal = float(np.mean(dapi_crop)) if dapi_crop.size else 0.0
            body_signal = float(np.mean(body_crop)) if body_crop.size else 0.0
            score = 0.60 * foreground + 0.20 * dapi_signal + 0.20 * body_signal
            candidates.append({"start": (x0, y0), "score": score,
                               "foreground_fraction_preview": foreground,
                               "dapi_mean_preview": dapi_signal,
                               "body_mean_preview": body_signal})
    candidates.sort(key=lambda item: (-item["score"], item["start"][1], item["start"][0]))
    first = candidates[0]
    fx, fy = first["start"]
    separate = [item for item in candidates[1:]
                if (item["start"][0] + size <= fx or fx + size <= item["start"][0]
                    or item["start"][1] + size <= fy or fy + size <= item["start"][1])]
    if separate:
        second = separate[0]
    elif len(candidates) > 1:
        second = max(candidates[1:], key=lambda item: abs(item["start"][0] - fx)
                     + abs(item["start"][1] - fy))
    else:
        second = first
    windows = []
    for index, item in enumerate((first, second), start=1):
        x0, y0 = item["start"]
        x1, y1 = x0 + size, y0 + size
        windows.append({"label": "Window {}".format(index),
                        "bounds": (x0, y0, x1, y1),
                        "coordinate": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "selection_preview": {k: v for k, v in item.items() if k != "start"}})
    return windows


def overview_rgb(dapi: np.ndarray, body: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render nuclear/body overview with instance boundaries for crop context."""
    rgb = np.zeros(dapi.shape + (3,), dtype=np.uint8)
    rgb[:, :, 0] = (np.clip(body, 0, 1) * 245).astype(np.uint8)
    rgb[:, :, 1] = (np.clip(body, 0, 1) * 190).astype(np.uint8)
    rgb[:, :, 2] = (np.clip(dapi, 0, 1) * 255).astype(np.uint8)
    rgb[:, :, 1] = np.maximum(rgb[:, :, 1], (np.clip(dapi, 0, 1) * 80).astype(np.uint8))
    rgb[mask_boundary(mask)] = (255, 255, 255)
    return rgb


def draw_panel(canvas: Image.Image, x: int, y: int, array: np.ndarray,
               title: str, font) -> None:
    if array.ndim == 2:
        panel = np.repeat((np.clip(array, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    else:
        panel = np.asarray(array, dtype=np.uint8)
    tile = Image.fromarray(panel, mode="RGB").resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    canvas.paste(tile, (x, y + 26))
    ImageDraw.Draw(canvas).text((x, y), title, fill=(235, 240, 243), font=font)


def compare_source_and_processed(source_tif, input_tif, image_tif, channels: pd.DataFrame,
                                 windows: List[dict], source_info: dict) -> dict:
    """Compare matched pixels after reversing the source placement/padding."""
    processed_valid = source_info.get("valid_bounds")
    if processed_valid is None:
        processed_valid = [0, 0, int(input_tif.pages[0].shape[-1]),
                           int(input_tif.pages[0].shape[-2])]
    valid_x0, valid_y0, valid_x1, valid_y1 = [int(value) for value in processed_valid]
    source_window = source_info.get("window")
    source_x0, source_y0 = ((int(source_window[0]), int(source_window[1]))
                            if source_window is not None else (0, 0))
    source_bounds, processed_bounds = [], []
    for window in windows:
        x0, y0, x1, y1 = window["bounds"]
        px0, py0 = max(x0, valid_x0), max(y0, valid_y0)
        px1, py1 = min(x1, valid_x1), min(y1, valid_y1)
        if px0 < px1 and py0 < py1:
            processed_bounds.append((px0, py0, px1, py1))
            source_bounds.append((source_x0 + px0 - valid_x0,
                                  source_y0 + py0 - valid_y0,
                                  source_x0 + px1 - valid_x0,
                                  source_y0 + py1 - valid_y0))
    comparisons = []
    for row in channels.itertuples(index=False):
        if row.marker_id not in SEGMENTATION_MARKERS or pd.isna(row.tiff_page):
            continue
        source_page = int(row.source_channel_index)
        input_page = int(row.output_channel_index)
        image_page = int(row.tiff_page)
        q995 = float(row.q995)
        source_crops = frame_windows(source_tif.pages[source_page], source_bounds)
        input_crops = frame_windows(input_tif.pages[input_page], processed_bounds)
        image_crops = frame_windows(image_tif.pages[image_page], [w["bounds"] for w in windows])
        source_equal = all(np.array_equal(a, b) for a, b in zip(source_crops, input_crops))
        source_differences = [np.abs(a.astype(np.int16) - b.astype(np.int16))
                              for a, b in zip(source_crops, input_crops)]
        source_left = np.concatenate([a.ravel() for a in source_crops])
        source_right = np.concatenate([b.ravel() for b in input_crops])
        source_correlation = (float(np.corrcoef(source_left, source_right)[0, 1])
                              if np.std(source_left) > 0 and np.std(source_right) > 0 else None)
        expected_crops = [np.clip(a.astype(np.float32) / max(q995, 1e-6), 0, 1)
                          for a in input_crops]
        errors = [np.abs(expected - actual.astype(np.float32))
                  for expected, actual in zip(expected_crops, image_crops)]
        comparisons.append({
            "marker_id": row.marker_id,
            "source_name": row.source_name,
            "source_channel_index": source_page,
            "input_tiff_page": input_page,
            "image_tiff_page": image_page,
            "source_equals_input_on_windows": source_equal,
            "source_input_max_abs_error": int(max(np.max(error) for error in source_differences)),
            "source_input_mae": float(np.mean(np.concatenate([e.ravel() for e in source_differences]))),
            "source_input_correlation": source_correlation,
            "q995": q995,
            "linear_normalization_max_abs_error": float(max(np.max(error) for error in errors)),
            "linear_normalization_rmse": float(np.sqrt(np.mean(np.concatenate([e.ravel() for e in errors]) ** 2))),
        })
    return {"checked_channels": len(comparisons),
            "source_coordinate_mapping": {
                "processed_valid_bounds_xyxy": [valid_x0, valid_y0, valid_x1, valid_y1],
                "source_window_xyxy": source_window,
                "source_origin_xy": [source_x0, source_y0],
                "compared_processed_bounds_xyxy": [list(bounds) for bounds in processed_bounds],
                "compared_source_bounds_xyxy": [list(bounds) for bounds in source_bounds],
            },
            "channels": comparisons}


def render_roi(study_row, output_dir: Path) -> dict:
    """Create one matched overview/local-window figure and evidence record."""
    roi = str(study_row.image_id)
    roi_dir = PROCESSED / "images" / roi
    channels = pd.read_parquet(roi_dir / "channels.parquet")
    metadata = json.loads((roi_dir / "metadata.json").read_text())
    selected = channels[channels.marker_id.isin(SEGMENTATION_MARKERS)
                        & channels.tiff_page.notna()].copy()
    selected = selected.sort_values("tiff_page")
    image_tif = tifffile.TiffFile(roi_dir / "image.tiff")
    input_tif = tifffile.TiffFile(roi_dir / "input.tiff")
    mask_tif = tifffile.TiffFile(roi_dir / "mask.tiff")
    source_files = study_row.source_files
    if isinstance(source_files, np.ndarray):
        source_files = source_files.tolist()
    source_rel = str(source_files[0] if isinstance(source_files, (list, tuple)) else source_files)
    source_path = NAS_ROOT / source_rel
    source_tif = tifffile.TiffFile(source_path)

    height, width = mask_tif.pages[0].shape[-2:]
    step = max(1, int(math.ceil(max(height, width) / float(PREVIEW_MAX))))
    dapi_page = int(selected.loc[selected.marker_id == "DAPI", "tiff_page"].iloc[0])
    body_pages = selected.loc[selected.body.astype(bool), "tiff_page"].astype(int).tolist()
    dapi_preview = tile_preview(image_tif.pages[dapi_page], step, segment_plane).astype(np.float32)
    body_planes = [tile_preview(image_tif.pages[page], step, segment_plane).astype(np.float32)
                   for page in body_pages]
    body_preview = np.mean(np.stack(body_planes), axis=0) if body_planes else np.zeros_like(dapi_preview)
    mask_preview = tile_preview(mask_tif.pages[0], step, segment_plane).astype(np.uint32)
    windows = choose_windows(mask_preview, dapi_preview, body_preview, (height, width), step)

    # Compare native OME, unnormalized input, and normalized shared channels on the same pixels.
    source_info = metadata["image_request"]["source"]
    fidelity = compare_source_and_processed(source_tif, input_tif, image_tif, selected,
                                            windows, source_info)
    bounds = [window["bounds"] for window in windows]
    local_planes = {}
    for row in selected.itertuples(index=False):
        local_planes[int(row.tiff_page)] = frame_windows(image_tif.pages[int(row.tiff_page)], bounds)
    local_masks = frame_windows(mask_tif.pages[0], bounds)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 24)
        label_font = ImageFont.truetype(font_path, 15)
        section_font = ImageFont.truetype(font_path, 19)
    except OSError:
        title_font = label_font = section_font = ImageFont.load_default()

    overview = overview_rgb(dapi_preview, body_preview, mask_preview)
    overview_image = Image.fromarray(overview, mode="RGB")
    overview_size = min(760, overview_image.width)
    overview_height = int(overview_image.height * overview_size / float(overview_image.width))
    overview_image = overview_image.resize((overview_size, overview_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(overview_image)
    for index, window in enumerate(windows):
        x0, y0, x1, y1 = window["bounds"]
        box = (int(x0 / step * overview_size / dapi_preview.shape[1]),
               int(y0 / step * overview_height / dapi_preview.shape[0]),
               int(x1 / step * overview_size / dapi_preview.shape[1]),
               int(y1 / step * overview_height / dapi_preview.shape[0]))
        color = (255, 232, 70) if index == 0 else (40, 240, 255)
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 4, box[1] + 4), str(index + 1), fill=color)

    panel_width = PANEL_COLUMNS * PANEL_SIZE + (PANEL_COLUMNS - 1) * 16
    canvas_width = panel_width + 64
    rows = 2
    row_height = PANEL_SIZE + 46
    window_height = 34 + rows * row_height + 14
    canvas_height = 58 + overview_height + 48 + 2 * window_height + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 24, 28))
    canvas.paste(overview_image, (32, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 14), "{} | {} | source / input / image fidelity: {} channels".format(
        STUDY, roi, fidelity["checked_channels"]), fill=(244, 248, 249), font=title_font)
    draw.text((32, 50 + overview_height + 6),
              "Overview: blue=DAPI, red/yellow=body mean, white=mask boundary; boxes=local crops",
              fill=(195, 207, 214), font=label_font)

    section_start = 50 + overview_height + 52
    for window_index, window in enumerate(windows):
        header_y = section_start + window_index * window_height
        coord = window["coordinate"]
        draw.text((32, header_y), "Window {} | x=[{}, {}) y=[{}, {})".format(
            window_index + 1, coord["x0"], coord["x1"], coord["y0"], coord["y1"]),
            fill=(246, 248, 249), font=section_font)
        planes = {int(row.tiff_page): local_planes[int(row.tiff_page)][window_index]
                  for row in selected.itertuples(index=False)}
        panels = [(planes[int(row.tiff_page)], row.marker_id)
                  for row in selected.itertuples(index=False)]
        mask = local_masks[window_index].astype(np.uint32)
        dapi = planes[dapi_page]
        body = np.mean([planes[p] for p in body_pages], axis=0) if body_pages else np.zeros_like(dapi)
        composite = np.zeros(dapi.shape + (3,), dtype=np.uint8)
        composite[:, :, 0] = (np.clip(body, 0, 1) * 240).astype(np.uint8)
        composite[:, :, 1] = (np.clip(body, 0, 1) * 175).astype(np.uint8)
        composite[:, :, 2] = (np.clip(dapi, 0, 1) * 255).astype(np.uint8)
        composite[mask_boundary(mask)] = (255, 255, 255)
        panels.extend([(colorize_mask(mask), "Instance mask"),
                       (composite, "DAPI/body + mask boundary")])
        for panel_index, (panel, label) in enumerate(panels):
            col, row = panel_index % PANEL_COLUMNS, panel_index // PANEL_COLUMNS
            x, y = 32 + col * (PANEL_SIZE + 16), header_y + 31 + row * row_height
            draw_panel(canvas, x, y, panel, label, label_font)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = output_dir / "figures" / (roi + ".jpg")
    figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure, format="JPEG", quality=92, optimize=True, progressive=True)
    image_tif.close()
    input_tif.close()
    mask_tif.close()
    source_tif.close()

    mask_check = metadata.get("mask_check") or {}
    selection_data = [{"marker_id": row.marker_id,
                       "source_name": row.source_name,
                       "source_channel_index": int(row.source_channel_index),
                       "output_channel_index": int(row.output_channel_index),
                       "image_page": int(row.tiff_page),
                       "nuclear": bool(row.nuclear), "body": bool(row.body),
                       "q995": float(row.q995)} for row in selected.itertuples(index=False)]
    label_counts = [int(len(np.unique(mask[mask > 0]))) for mask in local_masks]
    foreground_fraction = [float(np.mean(mask > 0)) for mask in local_masks]
    return {
        "study": STUDY,
        "roi": roi,
        "mask_origin": metadata.get("mask_origin"),
        "mask_type": mask_check.get("mask_type"),
        "judgment": "待视觉复核",
        "findings": "",
        "likely_cause": "",
        "next_action": "等待逐图判断；保留当前分割产物",
        "panel_id": str(study_row.panel_id),
        "source_file": source_path.name,
        "shape_yx": [int(height), int(width)],
        "pixel_size_um": metadata.get("pixel_size_um"),
        "segmentation_model": "cpsam_v2",
        "segmentation_precision": (metadata.get("segmentation_task") or {}).get("precision"),
        "segmentation_parameters": (metadata.get("segmentation_task") or {}).get("parameters"),
        "segmentation_signal_groups": (metadata.get("segmentation_task") or {}).get("signal", {}).get("groups"),
        "channels": selection_data,
        "windows": [{**window["coordinate"],
                     "preview_selection": window["selection_preview"],
                     "mask_labels_in_window": label_counts[index],
                     "mask_foreground_fraction": foreground_fraction[index]}
                    for index, window in enumerate(windows)],
        "source_input_image_fidelity": fidelity,
        "source_valid_bounds_xyxy": source_info.get("valid_bounds"),
        "source_window_xyxy": source_info.get("window"),
        "mask_check_metadata": mask_check,
        "figure": str(figure.relative_to(output_dir)),
    }


def render_previous_gallery_window(output_dir: Path) -> dict:
    """Recheck the previously displayed TMA8_3C crop at its exact coordinates."""
    roi = "TMA8_3C"
    roi_dir = PROCESSED / "images" / roi
    channels = pd.read_parquet(roi_dir / "channels.parquet")
    selected = channels[channels.marker_id.isin(SEGMENTATION_MARKERS)
                        & channels.tiff_page.notna()].sort_values("tiff_page")
    image_tif = tifffile.TiffFile(roi_dir / "image.tiff")
    mask_tif = tifffile.TiffFile(roi_dir / "mask.tiff")
    input_tif = tifffile.TiffFile(roi_dir / "input.tiff")
    source_tif = tifffile.TiffFile(RAW_ROOT / "TMA8_3C.tiff")
    metadata = json.loads((roi_dir / "metadata.json").read_text())
    bounds = [(4387, 1534, 4899, 2046)]
    planes = {int(row.tiff_page): frame_windows(image_tif.pages[int(row.tiff_page)], bounds)[0]
              for row in selected.itertuples(index=False)}
    mask = frame_windows(mask_tif.pages[0], bounds)[0].astype(np.uint32)
    dapi_page = int(selected.loc[selected.marker_id == "DAPI", "tiff_page"].iloc[0])
    body_pages = selected.loc[selected.body.astype(bool), "tiff_page"].astype(int).tolist()
    dapi = planes[dapi_page]
    body = np.mean([planes[page] for page in body_pages], axis=0)
    composite = np.zeros(dapi.shape + (3,), dtype=np.uint8)
    composite[:, :, 0] = (np.clip(body, 0, 1) * 240).astype(np.uint8)
    composite[:, :, 1] = (np.clip(body, 0, 1) * 175).astype(np.uint8)
    composite[:, :, 2] = (np.clip(dapi, 0, 1) * 255).astype(np.uint8)
    composite[mask_boundary(mask)] = (255, 255, 255)
    panels = [(planes[int(row.tiff_page)], row.marker_id)
              for row in selected.itertuples(index=False)]
    panels.extend([(colorize_mask(mask), "Instance mask"),
                   (composite, "DAPI/body + mask boundary")])
    canvas = Image.new("RGB", (4 * PANEL_SIZE + 3 * 16 + 64,
                                80 + 2 * (PANEL_SIZE + 46) + 20), (18, 24, 28))
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        title_font = label_font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 16), "Ly2025_COMET_AML | TMA8_3C | previous gallery window",
              fill=(244, 248, 249), font=title_font)
    draw.text((32, 48), "x=[4387,4899) y=[1534,2046) | 512x512 source crop",
              fill=(195, 207, 214), font=label_font)
    for panel_index, (panel, label) in enumerate(panels):
        col, row = panel_index % 4, panel_index // 4
        x, y = 32 + col * (PANEL_SIZE + 16), 76 + row * (PANEL_SIZE + 46)
        draw_panel(canvas, x, y, panel, label, label_font)
    figure = output_dir / "figures" / "TMA8_3C_previous_window.jpg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure, format="JPEG", quality=94, optimize=True)
    fidelity = compare_source_and_processed(source_tif, input_tif, image_tif,
                                            selected, [{"bounds": bounds[0]}],
                                            metadata["image_request"]["source"])
    image_tif.close()
    input_tif.close()
    mask_tif.close()
    source_tif.close()
    return {"bounds_xyxy": list(bounds[0]),
            "mask_labels": int(len(np.unique(mask[mask > 0]))),
            "mask_foreground_fraction": float(np.mean(mask > 0)),
            "source_input_image_fidelity": fidelity,
            "figure": str(figure.relative_to(output_dir))}


# Build matched channel/mask evidence for every current ROI in the study.
images = pd.read_parquet(PROCESSED / "images.parquet")
records = [render_roi(row, OUTPUT) for row in images.itertuples(index=False)]
OUTPUT.mkdir(parents=True, exist_ok=True)
prior_window = render_previous_gallery_window(OUTPUT)
next(record for record in records if record["roi"] == "TMA8_3C")["previous_gallery_window"] = prior_window
(OUTPUT / "roi_review.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# Ly2025_COMET_AML 逐ROI分割复核", "",
         "当前清单包含{}个ROI；本报告逐ROI生成DAPI、5个body分割输入通道、实例mask及同坐标边界叠加图。".format(len(records)),
         "源图比较先按metadata的valid_bounds/source.window反映射，去除padding后逐像素比较；input→image按q995线性归一对照。",
         "源坐标的映射为 source_xy = source_window_origin + processed_xy - valid_bounds_origin；valid_bounds外为padding，不参与源像素一致性。",
         "每图的512×512局部像素从原始 image.tiff/mask.tiff 按同一坐标读取；拼图仅为展示缩放。", "",
         "| ROI | Mask来源 | 源→input完全一致通道数 | 源→input最大绝对差 | 最低相关 | input→image最大误差 | 窗口坐标 | 图像 |",
         "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
for record in records:
    fidelity = record["source_input_image_fidelity"]
    exact = sum(item["source_equals_input_on_windows"] for item in fidelity["channels"])
    source_max_error = max(item["source_input_max_abs_error"] for item in fidelity["channels"])
    correlations = [item["source_input_correlation"] for item in fidelity["channels"]
                    if item["source_input_correlation"] is not None]
    min_correlation = min(correlations) if correlations else float("nan")
    max_error = max(item["linear_normalization_max_abs_error"] for item in fidelity["channels"])
    coords = "; ".join("({},{})-({},{})".format(
        w["x0"], w["y0"], w["x1"], w["y1"]) for w in record["windows"])
    lines.append("| {} | {} | {}/{} | {} | {:.5f} | {:.2e} | {} | `{}` |".format(
        record["roi"], record["mask_origin"], exact, fidelity["checked_channels"],
        source_max_error, min_correlation, max_error, coords, record["figure"]))
lines.extend(["", "## TMA8_3C 旧图库窗口", "",
              "旧图库窗口1 `(4387,1534)-(4899,2046)` 已按原坐标补做DAPI、5个body通道、mask及边界叠加，见 `figures/TMA8_3C_previous_window.jpg`。旧图库把512×512裁切缩放到480×480作显示；缩放只用于拼图显示，不改变被截取的原始像素或mask。",
              "旧图库窗口2 `(529,1909)-(1041,2421)` 也位于低信号区；本次重点复核窗口1的组织内mask及逐通道空间对应。"])
(OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("reviewed", len(records), "ROIs; figures", OUTPUT / "figures")
