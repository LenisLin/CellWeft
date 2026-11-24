"""Visual audit of the Bandyopadhyay CODEX bone-marrow masks.

Inputs: current processed images, masks, channel tables, metadata, and source masks.
Outputs: per-ROI matched channel/mask figures and a structured review record on NAS.
Upstream: public experiment manifest 20261119_v2 and current processed ROI assets.

The script samples mask-bearing windows inside each valid image window, compares
source-mask crops with the published mask TIFF, and leaves visual judgments for
manual review of the matched evidence figures.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NAS_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Bandyopadhyay2024_CODEX_BoneMarrow"
STUDY_ROOT = NAS_ROOT / "processed" / STUDY
OUTPUT_ROOT = NAS_ROOT / "processed/_catalog/public/evidence/segmentation_followup_20261119" / STUDY
FIGURE_ROOT = OUTPUT_ROOT / "figures"
ROI_SIZE = 512
PANEL_SIZE = 512
PANEL_COLUMNS = 3
LEGACY_WINDOWS = {
    "H37_Scan1": [
        (18141, 12046, 18653, 12558),
        (13574, 2126, 14086, 2638),
    ],
    "AML3_1329_Scan2": [
        (4215, 1111, 4727, 1623),
        (10180, 778, 10692, 1290),
    ],
}


# Reuse the tested local TIFF-tile reader from the public review generator.
HELPER_PATH = PROJECT_ROOT / "scripts/experiments/segmentation_roi_review/01_generate_html.py"
SPEC = importlib.util.spec_from_file_location("segmentation_roi_review_helpers", HELPER_PATH)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def candidate_bounds(valid_bounds, height, width):
    """Return a 3-by-3 set of 512-pixel windows inside the valid source area."""
    x0, y0, x1, y1 = [int(value) for value in valid_bounds]
    size = min(ROI_SIZE, y1 - y0, x1 - x0)
    max_x, max_y = min(width - size, x1 - size), min(height - size, y1 - size)
    xs = sorted(set((x0, (x0 + x1 - size) // 2, x1 - size)))
    ys = sorted(set((y0, (y0 + y1 - size) // 2, y1 - size)))
    return [(x, y, x + size, y + size) for y in ys for x in xs]


def select_windows(mask_page, dapi_page, valid_bounds, height, width, roi):
    """Choose mask-rich, DAPI-rich, and previously shown local windows."""
    bounds = candidate_bounds(valid_bounds, height, width)
    masks = HELPERS.tiff_windows(mask_page, bounds)
    scores = []
    for window, mask in zip(bounds, masks):
        labels = np.unique(mask)
        scores.append({"bounds": window, "mask_nonzero_pixels": int(np.count_nonzero(mask)),
                       "mask_label_count": int(np.count_nonzero(labels))})
    ranked = sorted(scores, key=lambda item: (-item["mask_nonzero_pixels"],
                                               -item["mask_label_count"],
                                               item["bounds"][1], item["bounds"][0]))
    first = ranked[0]["bounds"]
    separate = [item["bounds"] for item in ranked[1:] if not overlaps(item["bounds"], first)]
    second = separate[0] if separate else next(
        (item["bounds"] for item in ranked[1:] if item["bounds"] != first), first)
    selected = [(first, "mask-rich"), (second, "mask-rich separate")]

    dapi_crops = HELPERS.tiff_windows(dapi_page, bounds)
    for item, crop in zip(scores, dapi_crops):
        item["dapi_mean"] = float(np.mean(crop))
        item["dapi_p99"] = float(np.quantile(crop, 0.99))
    dapi_ranked = sorted(scores, key=lambda item: (-item["dapi_mean"], -item["dapi_p99"],
                                                   item["bounds"][1], item["bounds"][0]))
    dapi_bound = next((item["bounds"] for item in dapi_ranked
                       if item["bounds"] not in [bounds for bounds, _ in selected]), None)
    if dapi_bound is not None:
        selected.append((dapi_bound, "DAPI-rich"))

    for legacy in LEGACY_WINDOWS.get(roi, []):
        if legacy not in [bounds for bounds, _ in selected]:
            selected.append((legacy, "previous gallery window"))
    return selected, [dict(item, bounds=list(item["bounds"])) for item in scores]


def overlaps(first, second):
    """Return whether two XY windows share pixels."""
    return not (first[2] <= second[0] or second[2] <= first[0]
                or first[3] <= second[1] or second[3] <= first[1])


def source_mask_match(source_page, source_mapping, bounds, current_mask):
    """Compare one current mask crop with the corresponding source mask pixels."""
    window = source_mapping["window"]
    pad_left, pad_top, _, _ = [int(value) for value in source_mapping["padding_ltrb"]]
    sx0, sy0, sx1, sy1 = [int(value) for value in window]
    valid_x0, valid_y0 = pad_left, pad_top
    valid_x1, valid_y1 = valid_x0 + sx1 - sx0, valid_y0 + sy1 - sy0
    x0, y0, x1, y1 = [int(value) for value in bounds]
    ix0, iy0 = max(x0, valid_x0), max(y0, valid_y0)
    ix1, iy1 = min(x1, valid_x1), min(y1, valid_y1)
    expected = np.zeros(current_mask.shape, dtype=current_mask.dtype)
    if ix0 < ix1 and iy0 < iy1:
        source_bounds = (sx0 + ix0 - valid_x0, sy0 + iy0 - valid_y0,
                         sx0 + ix1 - valid_x0, sy0 + iy1 - valid_y0)
        source_crop = HELPERS.tiff_windows(source_page, [source_bounds])[0]
        expected[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = source_crop.astype(current_mask.dtype)
    return bool(np.array_equal(expected, current_mask))


def channel_statistics(image_crop):
    """Summarize local intensities for interpreting the displayed fixed scale."""
    values = np.asarray(image_crop, dtype=np.float32)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "nonzero_fraction": float(np.count_nonzero(values) / values.size),
    }


def channel_rgb(values):
    """Display the normalized input channel on its shared 0-to-1 scale."""
    gray = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    pixels = (gray * 255).astype(np.uint8)
    return np.repeat(pixels[:, :, None], 3, axis=2)


def mask_boundary(mask):
    """Return local instance boundaries for signal-overlay inspection."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def render_roi(row, roi_dir, output_path):
    """Read matched local windows and render the selected input channels and mask."""
    metadata = json.loads((roi_dir / "metadata.json").read_text())
    channels = pd.read_parquet(roi_dir / "channels.parquet")
    channels["nuclear"] = channels["nuclear"].fillna(False).astype(bool)
    channels["body"] = channels["body"].fillna(False).astype(bool)
    channels["tiff_page"] = pd.to_numeric(channels["tiff_page"], errors="coerce")
    available = channels[channels["tiff_page"].notna() & channels["keep_channel"].fillna(False)]
    nuclear = available[available["nuclear"]].sort_values("tiff_page").head(1).to_dict("records")
    body = available[available["body"]].sort_values("tiff_page").head(3).to_dict("records")
    selected = nuclear + body

    mask_request = metadata["mask_request"]
    valid_bounds = [int(row.pad_left_px), int(row.pad_top_px),
                    int(row.width_px - row.pad_right_px),
                    int(row.height_px - row.pad_bottom_px)]
    image_tif = tifffile.TiffFile(str(roi_dir / "image.tiff"))
    mask_tif = tifffile.TiffFile(str(roi_dir / "mask.tiff"))
    height, width = mask_tif.pages[0].shape[-2:]
    dapi_page = int(nuclear[0]["tiff_page"])
    selections, candidate_scores = select_windows(
        mask_tif.pages[0], image_tif.pages[dapi_page], valid_bounds, height, width, row.image_id)
    bounds_list = [bounds for bounds, _ in selections]
    mask_crops = HELPERS.tiff_windows(mask_tif.pages[0], bounds_list)
    channel_crops = {}
    for channel in selected:
        page = int(channel["tiff_page"])
        channel_crops[page] = HELPERS.tiff_windows(image_tif.pages[page], bounds_list)

    source_mapping = mask_request.get("source") or {}
    source_path = Path((mask_request.get("file_identity") or {}).get("path", ""))
    source_tif = tifffile.TiffFile(str(source_path)) if source_path.is_file() else None
    source_shape = ([int(value) for value in source_tif.pages[0].shape[-2:]]
                    if source_tif is not None else None)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 22)
        section_font = ImageFont.truetype(font_path, 19)
        label_font = ImageFont.truetype(font_path, 15)
    except OSError:
        title_font = section_font = label_font = ImageFont.load_default()

    canvas_width = PANEL_COLUMNS * PANEL_SIZE + (PANEL_COLUMNS - 1) * 16 + 48
    window_records = []
    figure_paths = []
    for index, (bounds, mask_crop) in enumerate(zip(bounds_list, mask_crops)):
        x0, y0, x1, y1 = [int(value) for value in bounds]
        map_match = None
        if source_tif is not None and source_mapping.get("window") and source_mapping.get("padding_ltrb"):
            map_match = source_mask_match(source_tif.pages[0], source_mapping, bounds, mask_crop)
        label = "{} window {}".format(selections[index][1], index + 1)
        panel_count = len(selected) + 2
        rows = int(math.ceil(panel_count / float(PANEL_COLUMNS)))
        row_height = PANEL_SIZE + 40
        canvas_height = 58 + 42 + rows * row_height + 16
        canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 24, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), "{} | {} | origin={}".format(
            STUDY, row.image_id, metadata.get("mask_origin")),
            fill=(245, 248, 249), font=title_font)
        draw.text((24, 58),
                  "{} | x=[{}, {}) y=[{}, {}) | source mapping exact={}".format(
                      label, x0, x1, y0, y1, map_match),
                  fill=(247, 249, 250), font=section_font)

        panels = []
        intensity_stats = {}
        for channel in selected:
            page = int(channel["tiff_page"])
            crop = channel_crops[page][index]
            marker = str(channel.get("marker_id", "unknown"))
            panels.append((channel_rgb(crop), "{} | page {} | p99={:.3f}".format(
                marker, page, float(np.quantile(crop, 0.99)))) )
            intensity_stats[marker] = channel_statistics(crop)

        dapi = channel_crops[int(nuclear[0]["tiff_page"])][index]
        contour_overlay = channel_rgb(dapi)
        contour_overlay[mask_boundary(mask_crop)] = (250, 70, 60)
        panels.extend([
            (contour_overlay, "DAPI + mask boundary"),
            (HELPERS.colorize_mask(mask_crop), "Instance mask"),
        ])
        panel_y = 58 + 34
        for panel_index, (panel, name) in enumerate(panels):
            col, panel_row = panel_index % PANEL_COLUMNS, panel_index // PANEL_COLUMNS
            panel_x = 24 + col * (PANEL_SIZE + 16)
            y = panel_y + panel_row * row_height
            draw.text((panel_x, y), name, fill=(225, 233, 236), font=label_font)
            tile = Image.fromarray(np.asarray(panel, dtype=np.uint8), mode="RGB")
            tile = tile.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
            canvas.paste(tile, (panel_x, y + 23))

        figure_path = output_path.with_name(
            output_path.stem + "__w{}.jpg".format(index + 1))
        canvas.save(str(figure_path), format="JPEG", quality=94, optimize=True)
        figure_paths.append(figure_path.relative_to(OUTPUT_ROOT).as_posix())
        window_records.append({
            "selection_reason": selections[index][1],
            "bounds_xyxy": [x0, y0, x1, y1],
            "mask_nonzero_fraction": float(np.count_nonzero(mask_crop) / mask_crop.size),
            "mask_label_count": int(np.unique(mask_crop).size - (1 if np.any(mask_crop == 0) else 0)),
            "source_mapping_exact": map_match,
            "channel_statistics": intensity_stats,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_tif is not None:
        source_tif.close()
    image_tif.close()
    mask_tif.close()

    source_ref = (mask_request.get("file_identity") or {}).get("path")
    return {
        "study": STUDY,
        "roi": row.image_id,
        "panel": row.panel_id,
        "mask_origin": metadata.get("mask_origin"),
        "mask_type": (metadata.get("mask_check") or {}).get("mask_type"),
        "judgment": "待视觉复核",
        "findings": "",
        "likely_cause": "",
        "next_action": "",
        "source_mask_path": source_ref,
        "source_mask_shape_yx": source_shape,
        "source_window": source_mapping.get("window"),
        "padding_ltrb": source_mapping.get("padding_ltrb"),
        "valid_bounds_xyxy": valid_bounds,
        "channel_pages": [{"marker_id": item.get("marker_id"), "tiff_page": int(item["tiff_page"]),
                           "nuclear": bool(item.get("nuclear")), "body": bool(item.get("body"))}
                          for item in selected],
        "candidate_windows": candidate_scores,
        "windows": window_records,
        "figures": figure_paths,
    }


# Inventory the current processed ROI set; cells.parquet is not read.
image_index = pd.read_parquet(STUDY_ROOT / "images.parquet")
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
review_records = []
for row in image_index.itertuples(index=False):
    roi_dir = STUDY_ROOT / "images" / row.image_id
    figure = FIGURE_ROOT / (row.image_id.replace("/", "%2F") + ".jpg")
    review_records.append(render_roi(row, roi_dir, figure))

(OUTPUT_ROOT / "roi_review.json").write_text(
    json.dumps(review_records, ensure_ascii=False, indent=2), encoding="utf-8")

print("reviewed candidates", len(review_records), "figures", FIGURE_ROOT)
