"""Owner-only, bounded screen of three preregistered tissue recipes.

Inputs: frozen blind job/ROI snapshot and current D01-bound normalized TIFFs.
Outputs: candidate overlays, ROI coverage, diagnostic 224/112 window counts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
from tempfile import TemporaryDirectory
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_opening, uniform_filter

from cellweft.data.io.readers import open_source, tiff_source, windows
from cellweft.data.preparation.crops import grid_origins
from cellweft.data.preparation.io_coordination import formal_read_lease


MORPH = np.ones((3, 3), bool)
METHODS = ("baseline", "log_otsu", "log_mean15_otsu")
PRODUCTS = ("channels", "image", "mask", "cells", "neighbors", "expression")


def version(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def guard(owner: dict, frozen: dict) -> None:
    """Require the same D01 table, head, full request and six products throughout."""
    study_root = Path(owner["root"]) / "processed" / owner["study"]
    snapshot_path = study_root / "node_input_snapshot.json"
    if version(snapshot_path) != frozen["snapshot"] or version(study_root / "review_results.csv") != frozen["review_results"]:
        raise ValueError("D01 review or snapshot version changed")
    snapshot = json.loads(snapshot_path.read_text())
    for image_id in owner["images"]:
        entry = frozen["images"][image_id]
        image_root = study_root / "images" / image_id
        metadata_path = image_root / "metadata.json"
        delivery = Path(entry["head"]["delivery"])
        ready_path = delivery / "ready.json"
        if (version(metadata_path) != entry["metadata"] or version(ready_path) != entry["ready"]
                or version(image_root / "input.tiff") != entry["input"]):
            raise ValueError(f"Input/ready/metadata version changed: {image_id}")
        metadata = json.loads(metadata_path.read_text())
        ready = json.loads(ready_path.read_text())
        if (snapshot["image_deliveries"][image_id] != entry["head"]
                or ready["source_bindings"] not in snapshot["bindings"]
                or metadata["request"] != ready["requests"][image_id]
                or metadata["source_delivery"] != str(delivery)
                or metadata["sampling"]["status"] != "pending"):
            raise ValueError(f"Current request/binding/sampling changed: {image_id}")
        source = metadata["request"]["image"]["source"]
        if source["panel_id"] != owner["panel"] or source.get("valid_mask") is not None:
            raise ValueError(f"Panel or valid-mask geometry changed: {image_id}")
        if set(metadata["products"]) != set(PRODUCTS):
            raise ValueError(f"Product set changed: {image_id}")
        for name in PRODUCTS:
            path = Path(metadata["products"][name])
            if (metadata["product_status"][name] != "complete"
                    or version(path) != entry["products"][name]):
                raise ValueError(f"Product version changed: {image_id}/{name}")


def otsu(values: np.ndarray, valid: np.ndarray) -> float:
    histogram = np.histogram(values[valid], bins=256, range=(0, 1))[0]
    centers = (np.arange(256) + .5) / 256
    weights = np.cumsum(histogram).astype(np.float64)
    totals = np.cumsum(histogram * centers)
    other = weights[-1] - weights
    scores = np.full(256, -np.inf)
    splits = (weights > 0) & (other > 0)
    scores[splits] = ((totals[-1] * weights[splits] - totals[splits] * weights[-1]) ** 2
                      / (weights[splits] * other[splits]))
    return float(centers[np.argmax(scores)])


def integral(mask: np.ndarray) -> np.ndarray:
    summed = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    summed[1:, 1:] = mask.cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64)
    return summed


def count_region(summed: np.ndarray, x: int, y: int, size: int) -> int:
    return int(summed[y + size, x + size] - summed[y, x + size]
               - summed[y + size, x] + summed[y, x])


def overlay(projection: np.ndarray, valid: np.ndarray, masks: dict[str, np.ndarray],
            boxes: list[dict], output: Path) -> None:
    """Show the same grayscale projection under all three candidate overlays."""
    height, width = projection.shape
    size = 800
    canvas = Image.new("RGB", (size * 3, size + 35), (20, 20, 20))
    labels = ImageDraw.Draw(canvas)
    base = np.uint8(np.clip(np.sqrt(np.clip(projection, 0, 1)), 0, 1) * 220)
    rgb = np.stack([base] * 3, axis=-1)
    colors = {"definite_tissue": (67, 232, 104), "background_lumen": (0, 216, 255),
              "padding": (255, 108, 87), "edge_tissue_side": (248, 211, 60),
              "edge_background_side": (255, 145, 52)}
    for index, (name, mask) in enumerate(masks.items()):
        panel = rgb.copy()
        panel[mask] = (panel[mask].astype(np.uint16) * 2 // 3
                       + np.array([0, 72, 12], np.uint16)).astype(np.uint8)
        panel[~valid] = (110, 18, 18)
        view = Image.fromarray(panel).resize((size, size), Image.Resampling.BOX)
        draw = ImageDraw.Draw(view)
        for box in boxes:
            x0, y0, x1, y1 = box["xyxy"]
            draw.rectangle((x0 * size / width, y0 * size / height,
                            x1 * size / width, y1 * size / height),
                           outline=colors[box["label"]], width=2)
        canvas.paste(view, (index * size, 35))
        labels.text((index * size + 10, 10), name, fill="white")
    canvas.save(output)


def compare_image(owner: dict, image_id: str, frozen: dict, roi: dict,
                  lock_root: Path, output: Path) -> dict:
    image_root = Path(owner["root"]) / "processed" / owner["study"] / "images" / image_id
    metadata = json.loads((image_root / "metadata.json").read_text())
    with formal_read_lease(image_root / "channels.parquet", io_lock_root=lock_root):
        channels = pd.read_parquet(image_root / "channels.parquet")
    height, width = map(int, metadata["shape_yx"])
    bounds = frozen["valid_bounds"] or [0, 0, width, height]
    left, top, right, bottom = map(int, bounds)
    valid = np.zeros((height, width), bool)
    valid[top:bottom, left:right] = True
    candidate_pages = channels.loc[channels.tiff_page.notna()
                                   & (channels.nuclear | channels.role.eq("downstream")),
                                   "tiff_page"].astype(int).tolist()
    downstream_pages = channels.loc[channels.tiff_page.notna() & channels.role.eq("downstream"),
                                    "tiff_page"].astype(int).tolist()
    projection = np.zeros((height, width), np.float32)
    downstream_nonblack = np.zeros((height, width), bool)

    # Read the normalized owner publication through one-image local staging.
    with TemporaryDirectory(prefix="cellweft_tissue_screen_") as temporary:
        local_image = Path(temporary) / "image.tiff"
        with formal_read_lease(image_root / "image.tiff", io_lock_root=lock_root):
            shutil.copyfile(image_root / "image.tiff", local_image)
        with open_source(tiff_source(local_image, int(channels.tiff_page.notna().sum()))) as reader:
            if reader.shape[-2:] != (height, width):
                raise ValueError(f"Canvas changed: {image_id}")
            pages = []
            for page in candidate_pages:
                low, high = np.inf, -np.inf
                for x0, y0, x1, y1 in windows((height, width), 512):
                    selected = reader.read_window([page], (x0, y0, x1, y1))[0]
                    selected = selected[valid[y0:y1, x0:x1]]
                    if selected.size:
                        low, high = min(low, float(selected.min())), max(high, float(selected.max()))
                if low < high:
                    pages.append(page)
            if not pages or not downstream_pages:
                raise ValueError(f"No measurable candidate/downstream channels: {image_id}")
            union = sorted(set(pages) | set(downstream_pages))
            project_indices = [union.index(page) for page in pages]
            downstream_indices = [union.index(page) for page in downstream_pages]
            for x0, y0, x1, y1 in windows((height, width), 512):
                block = reader.read_window(union, (x0, y0, x1, y1))
                projection[y0:y1, x0:x1] = block[project_indices].max(axis=0)
                downstream_nonblack[y0:y1, x0:x1] = np.any(block[downstream_indices] != 0, axis=0)
    projection[~valid] = 0
    downstream_nonblack &= valid

    log_projection = np.log1p(9 * np.clip(projection, 0, 1)) / np.log(10)
    local_count = uniform_filter(valid.astype(np.float32), size=15, mode="reflect")
    spatial = np.divide(uniform_filter(log_projection * valid, size=15, mode="reflect"),
                        local_count, out=np.zeros_like(log_projection), where=local_count > 0)
    inputs = {"baseline": projection, "log_otsu": log_projection,
              "log_mean15_otsu": spatial}
    masks = {}
    diagnostics = {}
    downstream_integral = integral(downstream_nonblack)
    origins = grid_origins((height, width), 224, 112)
    for name, values in inputs.items():
        threshold = otsu(values, valid)
        binary = (values > threshold) & valid
        binary = binary_closing(binary_opening(binary, structure=MORPH), structure=MORPH) & valid
        masks[name] = binary
        tissue_integral = integral(binary)
        area_pass = eligible = 0
        for x, y in origins:
            if count_region(tissue_integral, x, y, 224) <= 25088:
                continue
            area_pass += 1
            eligible += count_region(downstream_integral, x, y, 224) > 0
        roi_rows = []
        for box in roi["boxes"]:
            x0, y0, x1, y1 = map(int, box["xyxy"])
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"ROI outside canvas: {image_id}")
            count = int(binary[y0:y1, x0:x1].sum())
            area = (x1 - x0) * (y1 - y0)
            coverage = count / area
            label = box["label"]
            limit = {"definite_tissue": (">=", .8), "background_lumen": ("<=", .1),
                     "padding": ("=", 0), "edge_tissue_side": (">=", .7),
                     "edge_background_side": ("<=", .2)}[label]
            passed = (count == 0 if label == "padding" else
                      coverage >= limit[1] if limit[0] == ">=" else coverage <= limit[1])
            roi_rows.append({"label": label, "xyxy": box["xyxy"], "selected_pixels": count,
                             "area_pixels": area, "tissue_coverage": coverage,
                             "research_screen": {"comparison": limit[0], "threshold": limit[1], "pass": bool(passed)}})
        diagnostics[name] = {"threshold": threshold, "tissue_pixels": int(binary.sum()),
                             "tissue_fraction_measured": float(binary.sum() / valid.sum()),
                             "outside_valid_selected_pixels": int((binary & ~valid).sum()),
                             "strict_area_windows_diagnostic": area_pass,
                             "strict_area_and_downstream_nonblack_windows_diagnostic": int(eligible),
                             "roi_coverage": roi_rows}
    image_output = output / image_id
    image_output.mkdir(parents=True, exist_ok=True)
    overlay(projection, valid, masks, roi["boxes"], image_output / "candidate_overlay.png")
    if image_id == "20240717_B2207344__acq_005":
        box = (680, 720, 744, 784)
        detail = Image.new("RGB", (3 * 256, 280), (20, 20, 20))
        draw = ImageDraw.Draw(detail)
        for index, (name, mask) in enumerate(masks.items()):
            crop = np.uint8(mask[box[1]:box[3], box[0]:box[2]]) * 255
            detail.paste(Image.fromarray(crop).resize((256, 256), Image.Resampling.NEAREST).convert("RGB"),
                         (index * 256, 24))
            draw.text((index * 256 + 4, 4), name, fill="white")
        detail.save(image_output / "lumen_candidate_detail.png")
    return {"study": owner["study"], "image_id": image_id, "panel_id": owner["panel"],
            "shape_yx": [height, width], "valid_bounds": list(bounds),
            "measured_pixels": int(valid.sum()), "nonconstant_projection_pages": pages,
            "actual_downstream_pages": downstream_pages, "grid_windows": len(origins),
            "overlay": str(image_output / "candidate_overlay.png"),
            "candidates": diagnostics, "blind_roi_ambiguities": roi["ambiguous"]}


parser = argparse.ArgumentParser()
parser.add_argument("--comparison-job", type=Path, required=True)
parser.add_argument("--owner", choices=("72", "73"), required=True)
parser.add_argument("--lock-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
snapshot = json.loads(args.comparison_job.read_text())
job, rois = snapshot["blind_job"], snapshot["blind_rois"]
if set(job["recipes_frozen_for_later_separate_comparison"]) != set(METHODS):
    raise ValueError("Comparison must contain exactly the three frozen recipes")
owner = job["owners"][args.owner]
frozen = job["frozen_versions"][args.owner]
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
guard(owner, frozen)
args.output.mkdir(parents=True, exist_ok=True)
receipt_path = args.output / "execution.json"
receipt = {"owner": args.owner, "pid": os.getpid(),
           "start_tick": int(Path("/proc/self/stat").read_text().split()[21]),
           "started_at_ns": time.time_ns(), "status": "running", "images": [],
           "script": str(Path(__file__).resolve()), "candidate_methods": list(METHODS)}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
try:
    for image_id in owner["images"]:
        key = owner["study"] + "/" + image_id
        receipt["images"].append(compare_image(owner, image_id, frozen["images"][image_id],
                                                rois["images"][key], args.lock_root, args.output))
        guard(owner, frozen)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["status"] = "complete"
except Exception as exc:
    receipt["status"] = "failed"
    receipt["error"] = str(exc)
    raise
finally:
    receipt["ended_at_ns"] = time.time_ns()
    receipt["exit_intent"] = 0 if receipt["status"] == "complete" else 1
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
