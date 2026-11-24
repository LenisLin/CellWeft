"""Build a visual QC report from the current public experiment manifest.

Inputs: processed/_catalog/public/experiments.json and existing per-ROI image,
mask, channel, cell, image-index, and metadata assets.
Outputs: a static HTML report with one or two sampled ROIs per eligible study,
plus JPEG figures under processed/_catalog/public/evidence/.
Upstream: current public training and evaluation membership, version 20261119_v2.

The workflow inventories complete segmentation assets, selects representative
ROIs with a fixed seed, renders matched channel/mask views, and writes the report.
"""

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


NAS_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
MANIFEST_PATH = NAS_ROOT / "processed/_catalog/public/experiments.json"
OUTPUT_ROOT = NAS_ROOT / "processed/_catalog/public/evidence/segmentation_roi_review_20261119"
ASSET_FILES = ("image.tiff", "mask.tiff", "channels.parquet", "cells.parquet")
SEED = 20261119
CROP_SIZE = 512
PANEL_SIZE = 480
PANEL_COLUMNS = 3


def is_missing(value: object) -> bool:
    """Return whether a scalar dataframe value is null."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) == 0
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(pd.isna(value))


def text_value(value: object, default: str = "") -> str:
    """Convert a scalar source field to display text without pandas nulls."""
    return default if is_missing(value) else str(value)


def seed_for(study: str) -> int:
    """Derive a stable per-study random seed without process-randomized hashes."""
    return (SEED + sum((i + 1) * ord(char) for i, char in enumerate(study))) % (2 ** 32)


def source_description(row: pd.Series) -> str:
    """Describe the source batch using public source format and acquisition fields."""
    source_format = text_value(row.get("source_format"), "source unknown")
    acquisition = text_value(row.get("source_acquisition_id"))
    source_files = row.get("source_files")
    if not is_missing(source_files):
        if isinstance(source_files, np.ndarray):
            source_files = source_files.tolist()
        if isinstance(source_files, (list, tuple)) and source_files:
            source_name = Path(str(source_files[0])).name
        else:
            source_name = Path(str(source_files)).name
    else:
        source_name = ""
    values = [source_format]
    if source_name:
        values.append(source_name)
    if acquisition:
        values.append("acq " + acquisition)
    return " · ".join(values)


def specimen_key(row: pd.Series) -> Optional[str]:
    """Return an observed sample/core/slide key for distinct-specimen sampling."""
    for field in ("sample_id", "core_id", "slide_id"):
        value = text_value(row.get(field))
        if value:
            return field + ":" + value
    return None


def choose_random(candidates: List[dict], rng: np.random.RandomState,
                  excluded_specimens: Sequence[str] = ()) -> dict:
    """Select one reproducible candidate, preferring an unrepresented specimen."""
    excluded = set(excluded_specimens)
    distinct = [item for item in candidates
                if item["specimen_key"] and item["specimen_key"] not in excluded]
    pool = distinct if distinct else candidates
    index = int(rng.randint(0, len(pool)))
    return pool[index]


def inventory_study(study: str) -> Tuple[List[dict], Dict[str, int], int, bool]:
    """Load image membership and check complete segmentation-asset presence."""
    study_root = NAS_ROOT / "processed" / study
    image_table = study_root / "images.parquet"
    candidates: List[dict] = []
    missing_counts = {name: 0 for name in ASSET_FILES}
    indexed_count = 0
    if not image_table.is_file():
        return candidates, missing_counts, indexed_count, False

    images = pd.read_parquet(image_table)
    indexed_count = len(images)
    for _, row in images.iterrows():
        roi = text_value(row.get("image_id"))
        if not roi:
            continue
        roi_dir = study_root / "images" / roi
        missing = [name for name in ASSET_FILES if not (roi_dir / name).is_file()]
        for name in missing:
            missing_counts[name] += 1
        if missing:
            continue

        height = int(row["height_px"])
        width = int(row["width_px"])
        candidates.append({
            "study": study,
            "roi": roi,
            "row": row,
            "panel": text_value(row.get("panel_id"), "panel unknown"),
            "source_format": text_value(row.get("source_format"), "source unknown"),
            "source": source_description(row),
            "specimen_key": specimen_key(row),
            "height": height,
            "width": width,
            "pixels": height * width,
            "group_key": (text_value(row.get("panel_id"), "panel unknown"),
                          text_value(row.get("source_format"), "source unknown"),
                          text_value(row.get("source_acquisition_id"), "acquisition unknown")),
            "roi_dir": roi_dir,
        })
    return candidates, missing_counts, indexed_count, True


def select_rois(study: str, candidates: List[dict]) -> List[dict]:
    """Select a common stratum, plus one distinct source or clear metric outlier."""
    if study == "Aoki2024_IMC_RelapsedHodgkin":
        selected = []
        for roi, reason in (
            ("TMA_A1_1__acq_003", "指定优先样例：TMA_A1_1 acquisition 003"),
            ("TMA_A1_1__acq_006", "指定优先样例：TMA_A1_1 acquisition 006"),
        ):
            match = [item for item in candidates if item["roi"] == roi]
            if match:
                match[0]["sampling_reason"] = reason
                selected.append(match[0])
        return selected

    if not candidates:
        return []
    rng = np.random.RandomState(seed_for(study))
    group_counts: Dict[Tuple[str, str, str], int] = {}
    for item in candidates:
        group_counts[item["group_key"]] = group_counts.get(item["group_key"], 0) + 1
    common_group = sorted(group_counts, key=lambda key: (-group_counts[key], key))[0]
    primary_pool = [item for item in candidates if item["group_key"] == common_group]
    primary = choose_random(primary_pool, rng)
    primary["sampling_reason"] = "固定种子随机样例；来自最常见 panel/source/acquisition stratum"
    selected = [primary]

    alternate_groups = [key for key in group_counts if key != common_group]
    if alternate_groups:
        alternate_group = sorted(alternate_groups,
                                 key=lambda key: (-group_counts[key], key))[0]
        alternate_pool = [item for item in candidates if item["group_key"] == alternate_group]
        secondary = choose_random(alternate_pool, rng, [primary["specimen_key"] or ""])
        secondary["sampling_reason"] = "覆盖另一 panel/source/acquisition stratum"
        selected.append(secondary)
    return selected


def channel_selection(channels: pd.DataFrame, metadata: dict) -> List[dict]:
    """Choose up to five actual nuclear/body input channels for display."""
    table = channels.copy()
    table["nuclear"] = table["nuclear"].fillna(False).astype(bool)
    table["body"] = table["body"].fillna(False).astype(bool)
    table["tiff_page"] = pd.to_numeric(table["tiff_page"], errors="coerce")
    table = table[table["tiff_page"].notna() & table["keep_channel"].fillna(False).astype(bool)]

    signal = (metadata.get("segmentation_task") or {}).get("signal", {})
    groups = signal.get("groups", {}) if isinstance(signal, dict) else {}
    nuclear_pages = set(int(value) for value in groups.get("nuclear", []))
    body_pages = set(int(value) for value in groups.get("body", []))
    if nuclear_pages:
        table.loc[~table["tiff_page"].astype(int).isin(nuclear_pages), "nuclear"] = False
    if body_pages:
        table.loc[~table["tiff_page"].astype(int).isin(body_pages), "body"] = False

    nuclear = table[table["nuclear"]].sort_values("tiff_page").to_dict("records")
    body = table[table["body"]].sort_values("tiff_page").to_dict("records")
    chosen: List[dict] = []
    chosen.extend(nuclear[:2])
    body_limit = 5 - len(chosen)
    chosen.extend(body[:body_limit])
    if not chosen:
        chosen = table.sort_values("tiff_page").head(5).to_dict("records")
    return [dict(row, tiff_page=int(row["tiff_page"])) for row in chosen]


def channel_label(channel: dict) -> str:
    """Build a concise source/marker label for each channel panel."""
    marker = text_value(channel.get("marker_id"), "unmapped")
    source = text_value(channel.get("source_name"), "")
    page = int(channel["tiff_page"])
    role = ("DNA/nuclear" if bool(channel.get("nuclear")) else
            "membrane/body" if bool(channel.get("body")) else "other")
    if source and source != marker:
        marker += " · " + source
    return "{} | page {} | {}".format(marker, page, role)


def segment_plane(segment: np.ndarray) -> np.ndarray:
    """Convert one decoded grayscale TIFF tile/strip to a two-dimensional plane."""
    plane = np.squeeze(segment)
    if plane.ndim != 2:
        raise ValueError("Expected a grayscale image plane, got shape {}".format(segment.shape))
    return plane


def tiff_windows(page, bounds: Sequence[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """Decode only TIFF tiles or strips intersecting the requested windows."""
    page_height, page_width = page.shape[-2:]
    outputs = [np.zeros((y1 - y0, x1 - x0), dtype=page.dtype)
               for x0, y0, x1, y1 in bounds]
    if page.is_tiled:
        tile_height, tile_width = int(page.tilelength), int(page.tilewidth)
        tiles_per_row = int(math.ceil(page_width / float(tile_width)))
    else:
        tile_height, tile_width = int(page.rowsperstrip), page_width
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
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            out[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = tile[iy0-tile_y:iy1-tile_y,
                                                        ix0-tile_x:ix1-tile_x]
    return outputs


def display_scale(sample: np.ndarray, q995: object) -> float:
    """Choose one full-ROI display scale for normalized and unnormalized channels."""
    sample = np.asarray(sample, dtype=np.float32)
    q = float(np.quantile(sample, 0.995)) if sample.size else 1.0
    maximum = float(np.max(sample)) if sample.size else 1.0
    if maximum <= 1.05:
        return 1.0
    if not is_missing(q995) and float(q995) > 0:
        return float(q995)
    return max(q, 1e-6)


def scaled_channel(array: np.ndarray, scale: float) -> np.ndarray:
    """Convert a shared channel into an 8-bit grayscale panel."""
    values = np.asarray(array, dtype=np.float32)
    return np.clip(values / max(scale, 1e-6), 0, 1)


def random_windows(height: int, width: int, rng: np.random.RandomState) -> List[dict]:
    """Choose up to two reproducible, non-overlapping square windows."""
    size = min(CROP_SIZE, height, width)
    max_x, max_y = width - size, height - size
    first_x, first_y = int(rng.randint(max_x + 1)), int(rng.randint(max_y + 1))
    starts = [(first_x, first_y)]
    if max_x or max_y:
        for _ in range(64):
            x0, y0 = int(rng.randint(max_x + 1)), int(rng.randint(max_y + 1))
            if (x0, y0) == starts[0]:
                continue
            separated = (x0 + size <= first_x or first_x + size <= x0
                         or y0 + size <= first_y or first_y + size <= y0)
            if separated:
                starts.append((x0, y0))
                break
        if len(starts) == 1:
            for _ in range(64):
                x0, y0 = int(rng.randint(max_x + 1)), int(rng.randint(max_y + 1))
                if (x0, y0) != starts[0]:
                    starts.append((x0, y0))
                    break
        if len(starts) == 1:
            starts.append(starts[0])
    return [{"label": "Random window {}".format(index + 1),
             "bounds": (x0, y0, x0 + size, y0 + size),
             "location": "x=[{},{}) y=[{},{})".format(x0, x0 + size, y0, y0 + size)}
            for index, (x0, y0) in enumerate(starts)]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Assign stable instance colors while keeping the background black."""
    labels, inverse = np.unique(mask.astype(np.uint32), return_inverse=True)
    values = labels.astype(np.uint64)
    hashed = values * np.uint64(2654435761) + np.uint64(2246822519)
    colors = np.stack((72 + ((hashed >> 8) % 184),
                       72 + ((hashed >> 17) % 184),
                       72 + ((hashed >> 25) % 184)), axis=1).astype(np.uint8)
    colors[labels == 0] = 0
    return colors[inverse].reshape(mask.shape + (3,))


def draw_panel(canvas: Image.Image, x: int, y: int, panel: np.ndarray,
               label: str, font: ImageFont.ImageFont) -> None:
    """Paste one labeled channel or instance-mask panel into a figure."""
    if panel.ndim == 2:
        panel = np.repeat((np.clip(panel, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    tile = Image.fromarray(np.asarray(panel, dtype=np.uint8), mode="RGB")
    tile = tile.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    canvas.paste(tile, (x, y + 28))
    ImageDraw.Draw(canvas).text((x, y), label[:58], fill=(238, 242, 244), font=font)


def render_figure(study: str, candidate: dict, output_path: Path) -> dict:
    """Render random local channel and instance-mask views for one ROI."""
    roi_dir = candidate["roi_dir"]
    channels = pd.read_parquet(roi_dir / "channels.parquet")
    metadata = json.loads((roi_dir / "metadata.json").read_text())
    selected = channel_selection(channels, metadata)
    image_tif = tifffile.TiffFile(str(roi_dir / "image.tiff"))
    mask_tif = tifffile.TiffFile(str(roi_dir / "mask.tiff"))
    height, width = mask_tif.pages[0].shape[-2:]
    rng = np.random.RandomState(seed_for(study + "::" + candidate["roi"]))
    windows = random_windows(height, width, rng)
    bounds = [window["bounds"] for window in windows]
    local_planes: Dict[int, List[np.ndarray]] = {}
    scales: Dict[int, float] = {}
    for channel in selected:
        page = channel["tiff_page"]
        views = tiff_windows(image_tif.pages[page], bounds)
        local_planes[page] = views
        samples = np.concatenate([view.ravel() for view in views])
        scales[page] = display_scale(samples, channel.get("q995"))
    local_masks = tiff_windows(mask_tif.pages[0], bounds)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        title_font = ImageFont.truetype(font_path, 25)
        label_font = ImageFont.truetype(font_path, 17)
        window_font = ImageFont.truetype(font_path, 20)
    except OSError:
        title_font = label_font = window_font = ImageFont.load_default()

    panel_count = len(selected) + 1
    rows = int(math.ceil(panel_count / float(PANEL_COLUMNS)))
    panel_width = PANEL_COLUMNS * PANEL_SIZE + (PANEL_COLUMNS - 1) * 18
    canvas_width = panel_width + 72
    row_height = PANEL_SIZE + 48
    section_height = 48 + rows * row_height + 18
    canvas_height = 58 + len(windows) * section_height + 12
    canvas = Image.new("RGB", (canvas_width, canvas_height), (17, 23, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 15), "{} | {}".format(study, candidate["roi"]),
              fill=(246, 248, 249), font=title_font)

    for window_index, window in enumerate(windows):
        header_y = 58 + window_index * section_height
        draw.text((36, header_y), "{} | {}".format(window["label"], window["location"]),
                  fill=(247, 249, 250), font=window_font)
        panels = [(scaled_channel(local_planes[channel["tiff_page"]][window_index],
                                  scales[channel["tiff_page"]]), channel_label(channel))
                  for channel in selected]
        panels.append((colorize_mask(local_masks[window_index]), "Instance mask"))
        panel_y = header_y + 38
        for panel_index, (panel, label) in enumerate(panels):
            col, row = panel_index % PANEL_COLUMNS, panel_index // PANEL_COLUMNS
            panel_x = 36 + col * (PANEL_SIZE + 18)
            panel_top = panel_y + row * row_height
            draw_panel(canvas, panel_x, panel_top, panel, label, label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), format="JPEG", quality=92, optimize=True, progressive=True)
    image_tif.close()
    mask_tif.close()

    mask_origin = text_value(metadata.get("mask_origin"), "unknown")
    mask_check = metadata.get("mask_check") or metadata.get("source_mask_check") or {}
    mask_type = text_value(mask_check.get("mask_type"), "unknown") if isinstance(mask_check, dict) else "unknown"
    source_file = ""
    source_mask = metadata.get("source_mask_check") or {}
    if isinstance(source_mask, dict):
        source = source_mask.get("source") or {}
        source_file = Path(text_value(source.get("path"))).name if isinstance(source, dict) else ""
    return {
        "study": study,
        "roi": candidate["roi"],
        "panel": candidate["panel"],
        "source": candidate["source"],
        "mask_origin": mask_origin,
        "mask_type": mask_type,
        "mask_source_file": source_file,
        "channels": [channel_label(item) for item in selected],
        "has_nuclear_channel": any(bool(item.get("nuclear")) for item in selected),
        "has_body_channel": any(bool(item.get("body")) for item in selected),
        "windows": [window["location"] for window in windows],
        "sampling_reason": candidate["sampling_reason"],
    }


def html_table(rows: List[dict]) -> str:
    """Render the sample index with an explicit pending-review status."""
    parts = []
    for item in rows:
        parts.append("<tr data-study='{}'><td><a href='#{}'>{}</a></td><td>{}</td><td>{}</td>"
                     "<td>{}</td><td>{}</td></tr>".format(
                         html.escape(item["study"]), html.escape(item["anchor"]),
                         html.escape(item["study"]), html.escape(item["roi"]),
                         html.escape(item["panel"]), html.escape(item["mask_origin"]),
                         "待你复核"))
    return "\n".join(parts)


def make_html(manifest: dict, study_records: List[dict], roi_records: List[dict],
              missing_records: List[dict], output_path: Path) -> None:
    """Write a searchable, self-contained HTML page with relative figure assets."""
    summary_rows = []
    sections = []
    for study_record in study_records:
        study = study_record["study"]
        samples = [item for item in roi_records if item["study"] == study]
        if not samples:
            continue
        anchor = "study-" + quote(study, safe="")
        for sample in samples:
            summary_rows.append(dict(sample, anchor=anchor))
        roi_html = []
        for sample in samples:
            image_path = html.escape(sample["figure"])
            channel_list = "<br>".join(html.escape(value) for value in sample["channels"])
            window_list = "<br>".join(html.escape(value) for value in sample["windows"])
            channel_note = ("通道表标记了可用核/DNA通道。" if sample["has_nuclear_channel"]
                            else "当前通道表未标记可用核/DNA通道。")
            roi_html.append("""<article class="roi">
              <div class="roi-head"><h3>{roi}</h3><span class="tag">待你复核</span></div>
              <dl class="meta-grid">
                <div><dt>抽样理由</dt><dd>{sampling_reason}</dd></div>
                <div><dt>Panel / 来源批次</dt><dd>{panel}<br>{source}</dd></div>
                <div><dt>Mask 来源 / 类型</dt><dd>{mask_origin} / {mask_type}{source_file}</dd></div>
                <div><dt>通道</dt><dd>{channel_list}</dd></div>
                <div><dt>核通道记录</dt><dd>{channel_note}</dd></div>
                <div><dt>随机窗口坐标</dt><dd>{windows}</dd></div>
              </dl>
              <a class="figure-link" href="{image_path}" target="_blank" rel="noopener">
                <img loading="lazy" src="{image_path}" alt="{study} {roi} random channel crops and instance masks">
              </a>
              <p class="figure-path">图像：{image_path}</p>
            </article>""".format(
                roi=html.escape(sample["roi"]),
                sampling_reason=html.escape(sample["sampling_reason"]),
                panel=html.escape(sample["panel"]), source=html.escape(sample["source"]),
                mask_origin=html.escape(sample["mask_origin"]), mask_type=html.escape(sample["mask_type"]),
                source_file=("<br>" + html.escape(sample["mask_source_file"]) if sample["mask_source_file"] else ""),
                channel_list=channel_list, channel_note=html.escape(channel_note),
                windows=window_list, image_path=image_path,
                study=html.escape(study)))
        detail_attr = " open" if study == "Aoki2024_IMC_RelapsedHodgkin" else ""
        sections.append("<details id='{}'{}><summary><strong>{}</strong> <span>{} ROI</span></summary>{}</details>".format(
            html.escape(anchor), detail_attr, html.escape(study), len(samples), "".join(roi_html)))

    missing_rows = []
    for item in missing_records:
        missing_text = ", ".join("{}: {}".format(name, count)
                                  for name, count in item["missing_counts"].items() if count)
        if not missing_text:
            missing_text = "无可计数的ROI级缺失项"
        if not item["image_index_exists"]:
            index_status = "images.parquet 缺失"
        elif item["indexed_count"] == 0:
            index_status = "images.parquet 存在，0 条图像记录"
        else:
            index_status = "有图像记录，但没有完整 ROI"
        missing_rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(item["study"]), item["indexed_count"],
            html.escape(index_status), html.escape(missing_text)))

    overview_summary = ("<section class='callout'><strong>Aoki 样例</strong><p>展示指定的 "
                        "TMA_A1_1__acq_003 与 TMA_A1_1__acq_006。</p></section>")
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CellWeft 分割 ROI 抽样复核</title>
<style>
:root{color-scheme:light;--ink:#202b31;--muted:#68777e;--line:#d6dee1;--paper:#fff;--wash:#f3f6f6;--teal:#0b756f;--amber:#ad5c00;--red:#a93131}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:24px max(24px,calc((100vw - 1440px)/2)) 18px}
h1{font-size:25px;margin:0 0 6px}header p{margin:3px 0;color:var(--muted)}main{max-width:1440px;margin:20px auto;padding:0 20px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.stat{background:#fff;border:1px solid var(--line);padding:10px 14px;min-width:150px;border-radius:4px}.stat b{display:block;font-size:20px}
.controls{position:sticky;top:0;z-index:2;background:#fff;border:1px solid var(--line);padding:10px;display:flex;gap:10px;align-items:center;margin-bottom:16px}
input,select{font:inherit;border:1px solid #aebbc0;border-radius:3px;background:#fff;padding:8px;color:var(--ink)}#search{flex:1;min-width:220px}
section.block{background:#fff;border-top:1px solid var(--line);padding:18px 0;margin:20px 0}h2{font-size:20px;margin:0 0 12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{background:#f6f8f8;position:sticky;top:54px}
.callout{border-left:4px solid var(--teal);padding:10px 14px;background:#eff8f7;margin:14px 0}.callout p{margin:5px 0 0}
details{background:#fff;border:1px solid var(--line);border-radius:4px;margin:12px 0}summary{cursor:pointer;padding:13px 15px;display:flex;gap:10px;align-items:center}summary span{color:var(--muted);font-size:13px}.roi{padding:10px 16px 22px;border-top:1px solid var(--line)}
.roi-head{display:flex;align-items:center;gap:12px}.roi h3{font-size:17px;margin:10px 0}.tag{font-size:12px;padding:3px 8px;border:1px solid var(--line);border-radius:3px;background:#f5f7f7}
.meta-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 18px;margin:0 0 14px}.meta-grid div{padding:7px 0;border-top:1px solid #e6ebed;min-width:0}.meta-grid dt{font-size:12px;color:var(--muted)}.meta-grid dd{margin:2px 0 0;overflow-wrap:anywhere}.muted,.figure-path{color:var(--muted);font-size:12px}
.figure-link{display:block;background:#11171c;line-height:0}.figure-link img{display:block;width:100%;height:auto}.figure-path{margin:5px 0 0;overflow-wrap:anywhere}
.missing{border-left:4px solid var(--amber);padding-left:14px}.empty{padding:18px;color:var(--muted);text-align:center}
@media(max-width:760px){header{padding:18px}.meta-grid{grid-template-columns:1fr}.controls{position:static;flex-wrap:wrap}.controls input{width:100%}main{padding:0 10px}.stat{flex:1;min-width:120px}table{font-size:12px}}
</style></head><body>
<header><h1>公共训练与评价清单：分割 ROI 抽样复核</h1>
<p>当前 experiments.json · 抽样种子 {seed} · 生成于 {date}</p>
<p>每个有完整当前分割资产的 study 展示 1–2 个 ROI；每个 ROI 随机截取至多两个窗口，显示 3–5 个通道及实例 mask。</p>
<p>优先显示通道表标记可用的核/DNA与膜/胞质通道；未标记核通道的 ROI 会在记录中注明。</p>
<p>完整资产检查：image.tiff、mask.tiff、channels.parquet、cells.parquet 均存在。复核状态由你确认。</p></header>
<main><div class="stats"><div class="stat"><b>{study_count}</b>有完整分割资产的 study</div><div class="stat"><b>{roi_count}</b>展示 ROI</div><div class="stat"><b>{missing_count}</b>无完整资产 study</div><div class="stat"><b>{indexed_count:,}</b>清单图像成员</div></div>
<div class="controls"><input id="search" type="search" placeholder="筛选 study 或 ROI"></div>
{aoki_summary}
<section class="block"><h2>逐 ROI 汇总</h2><table id="summary"><thead><tr><th>Study</th><th>ROI</th><th>Panel</th><th>Mask 来源</th><th>复核状态</th></tr></thead><tbody>{summary_table}</tbody></table></section>
<section class="block missing"><h2>无完整当前分割资产的 study（{missing_count}）</h2><table><thead><tr><th>Study</th><th>清单图像成员</th><th>当前图像索引</th><th>ROI级缺失产物计数</th></tr></thead><tbody>{missing_rows}</tbody></table></section>
<section class="block"><h2>ROI 图像与记录</h2>{sections}</section></main>
<script>
const search=document.getElementById('search');
function update(){const q=search.value.toLowerCase();document.querySelectorAll('details[id^="study-"]').forEach(s=>{s.hidden=!s.innerText.toLowerCase().includes(q);});document.querySelectorAll('#summary tbody tr').forEach(r=>{r.hidden=!r.innerText.toLowerCase().includes(q);});}
search.addEventListener('input',update);
</script></body></html>"""
    replacements = {
        "{seed}": str(SEED),
        "{date}": "2026-11-19",
        "{study_count}": str(len(study_records) - len(missing_records)),
        "{roi_count}": str(len(roi_records)),
        "{missing_count}": str(len(missing_records)),
        "{indexed_count}": "{:,}".format(sum(item["indexed_count"] for item in study_records)),
        "{aoki_summary}": overview_summary,
        "{summary_table}": html_table(summary_rows),
        "{missing_rows}": "\n".join(missing_rows),
        "{sections}": "\n".join(sections),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def remove_stale_figures(output_root: Path, candidates: List[dict]) -> None:
    """Remove report JPEGs left by an earlier sample selection."""
    assets_root = output_root / "assets"
    if not assets_root.exists():
        return
    expected = {
        assets_root / candidate["study"] /
        (quote(candidate["roi"], safe="")[:180] + ".jpg")
        for candidate in candidates
    }
    for path in assets_root.rglob("*.jpg"):
        if path not in expected:
            path.unlink()


def main() -> None:
    """Inventory the manifest, select ROIs, render figures, and publish HTML."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", help="Generate one study while checking the renderer")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    studies = list(manifest["datasets"])
    if args.study:
        studies = [study for study in studies if study == args.study]
    study_records = []
    missing_records = []
    selected_records = []

    # Inventory current complete ROI assets and select a fixed, study-local sample.
    for study in studies:
        candidates, missing_counts, indexed_count, image_index_exists = inventory_study(study)
        study_record = {"study": study, "candidate_count": len(candidates),
                        "missing_counts": missing_counts, "indexed_count": indexed_count,
                        "image_index_exists": image_index_exists}
        study_records.append(study_record)
        if not candidates:
            missing_records.append(study_record)
            continue
        selected = select_rois(study, candidates)
        for candidate in selected:
            study_record["roi_count"] = study_record.get("roi_count", 0) + 1
            selected_records.append(candidate)

    remove_stale_figures(args.output_root, selected_records)

    # Render random channel and instance-mask windows for each selected ROI.
    roi_records = []
    for candidate in selected_records:
        relative_name = quote(candidate["roi"], safe="")[:180] + ".jpg"
        figure_path = args.output_root / "assets" / candidate["study"] / relative_name
        record = render_figure(candidate["study"], candidate, figure_path)
        record["figure"] = figure_path.relative_to(args.output_root).as_posix()
        record["sampling_reason"] = candidate["sampling_reason"]
        roi_records.append(record)
        print("rendered", candidate["study"], candidate["roi"], record["figure"])

    # Publish summary metadata beside the HTML so the visual review is traceable.
    output_path = args.output_root / "index.html"
    make_html(manifest, study_records, roi_records, missing_records, output_path)
    (args.output_root / "roi_review_summary.json").write_text(
        json.dumps({"seed": SEED, "complete_studies": len(study_records) - len(missing_records),
                    "missing_studies": missing_records, "rois": roi_records},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("report", output_path)


if __name__ == "__main__":
    main()
