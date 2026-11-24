"""Render six preselected, owner-only tissue reference views before mask review.

Inputs: frozen job JSON, current D01-bound owner publications and ready packets.
Outputs: compact version preflight JSON or blind PNGs plus execution receipt.
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

from cellweft.data.io.readers import open_source, tiff_source, windows
from cellweft.data.preparation.io_coordination import formal_read_lease


PRODUCTS = ("channels", "image", "mask", "cells", "neighbors", "expression")


def version(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def current_versions(owner: dict) -> dict:
    """Verify current D01, image head, full request, and six formal products."""
    study_root = Path(owner["root"]) / "processed" / owner["study"]
    snapshot_path = study_root / "node_input_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    reviews = pd.read_csv(study_root / "review_results.csv", keep_default_na=False)
    latest = reviews.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
    d01 = latest[(latest.study == owner["study"]) & (latest.check_id == "D01")
                  & (latest.result == "pass")]
    result = {"snapshot": version(snapshot_path), "review_results": version(study_root / "review_results.csv"),
              "images": {}}
    for image_id in owner["images"]:
        image_root = study_root / "images" / image_id
        metadata_path = image_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        delivery = Path(snapshot["image_deliveries"][image_id]["delivery"])
        ready_path = delivery / "ready.json"
        ready = json.loads(ready_path.read_text())
        binding = ready["source_bindings"]
        accepted = any(json.loads(row.details).get("input_binding") == binding
                       for row in d01.itertuples() if row.details.lstrip().startswith("{"))
        if not accepted or binding not in snapshot["bindings"]:
            raise ValueError(f"D01/snapshot binding changed: {image_id}")
        if (ready["created_at_ns"] != snapshot["image_deliveries"][image_id]["created_at_ns"]
                or image_id not in ready["image_ids"] or metadata["source_delivery"] != str(delivery)
                or metadata["source_binding"] != binding or metadata["request"] != ready["requests"][image_id]):
            raise ValueError(f"Ready request or head changed: {image_id}")
        if metadata["image_id"] != image_id or metadata["request"]["image"]["source"]["panel_id"] != owner["panel"]:
            raise ValueError(f"Formal image identity changed: {image_id}")
        if metadata["sampling"]["status"] != "pending" or set(metadata["products"]) != set(PRODUCTS):
            raise ValueError(f"Formal sampling/products changed: {image_id}")
        if any(metadata["product_status"][name] != "complete" for name in PRODUCTS):
            raise ValueError(f"Incomplete formal product: {image_id}")
        paths = {name: Path(metadata["products"][name]) for name in PRODUCTS}
        if any(path.parent != image_root for path in paths.values()):
            raise ValueError(f"Unexpected formal product path: {image_id}")
        channels = pd.read_parquet(paths["channels"])
        selected = channels[(channels.role.eq("downstream") | channels.nuclear)
                            & channels.tiff_page.notna() & channels.q995.gt(0)]
        nuclear = selected[selected.nuclear].sort_values("source_channel_index")
        downstream = selected[selected.role.eq("downstream")].sort_values("source_channel_index")
        if nuclear.empty or len(downstream) < 2:
            raise ValueError(f"Missing reference channels: {image_id}")
        references = [nuclear.iloc[0], downstream.iloc[0], downstream.iloc[-1]]
        result["images"][image_id] = {
            "head": {"delivery": str(delivery), "created_at_ns": ready["created_at_ns"]},
            "ready": version(ready_path), "metadata": version(metadata_path),
            "products": {name: version(path) for name, path in paths.items()},
            "input": version(image_root / "input.tiff"),
            "valid_bounds": metadata["request"]["image"]["source"].get("valid_bounds"),
            "selected_pages": selected.tiff_page.astype(int).tolist(),
            "raw_pages": selected.output_channel_index.astype(int).tolist(),
            "reference_channels": [{"role": "nuclear" if idx == 0 else "downstream",
                                    "marker_id": str(row.marker_id),
                                    "source_channel_index": int(row.source_channel_index),
                                    "tiff_page": int(row.tiff_page)}
                                   for idx, row in enumerate(references)],
        }
    return result


def display(values: np.ndarray, valid: np.ndarray, *, raw: bool = False) -> tuple[np.ndarray, float]:
    """Apply a recorded display-only stretch without changing source pixels."""
    transformed = np.log1p(np.maximum(values, 0)) if raw else np.sqrt(np.clip(values, 0, 1))
    high = float(np.quantile(transformed[valid], .995))
    pixels = np.uint8(np.clip(transformed / max(high, 1e-8), 0, 1) * 255)
    pixels[~valid] = 0
    return pixels, high


def render_image(owner: dict, image_id: str, frozen: dict, output: Path, lock_root: Path) -> dict:
    image_root = Path(owner["root"]) / "processed" / owner["study"] / "images" / image_id
    shape = json.loads((image_root / "metadata.json").read_text())["shape_yx"]
    height, width = map(int, shape)
    bounds = frozen["valid_bounds"] or [0, 0, width, height]
    left, top, right, bottom = map(int, bounds)
    valid = np.zeros((height, width), bool)
    valid[top:bottom, left:right] = True
    pages = frozen["selected_pages"]
    raw_pages = frozen["raw_pages"]
    channel_pages = [row["tiff_page"] for row in frozen["reference_channels"]]
    raw_projection = np.zeros((height, width), np.float32)
    normalized_projection = np.zeros((height, width), np.float32)
    references = [np.zeros((height, width), np.float32) for _ in channel_pages]

    # Stage one image under the common formal read lease, then decode locally.
    with TemporaryDirectory(prefix="cellweft_blind_refs_") as temporary:
        local = Path(temporary)
        for name in ("input.tiff", "image.tiff"):
            with formal_read_lease(image_root / name, io_lock_root=lock_root):
                shutil.copyfile(image_root / name, local / name)
        with open_source(tiff_source(local / "input.tiff", max(raw_pages) + 1)) as raw_reader, \
             open_source(tiff_source(local / "image.tiff", max(pages) + 1)) as normalized_reader:
            if raw_reader.shape[-2:] != (height, width) or normalized_reader.shape[-2:] != (height, width):
                raise ValueError(f"Image canvas changed: {image_id}")
            for x0, y0, x1, y1 in windows((height, width), 512):
                raw_projection[y0:y1, x0:x1] = raw_reader.read_window(raw_pages, (x0, y0, x1, y1)).max(axis=0)
                normalized = normalized_reader.read_window(pages, (x0, y0, x1, y1))
                normalized_projection[y0:y1, x0:x1] = normalized.max(axis=0)
                for i, page in enumerate(channel_pages):
                    references[i][y0:y1, x0:x1] = normalized[pages.index(page)]

    destination = output / image_id
    destination.mkdir(parents=True, exist_ok=True)
    images = {"raw_projection": (raw_projection, True),
              "normalized_projection": (normalized_projection, False),
              "nuclear": (references[0], False),
              "downstream_first": (references[1], False),
              "downstream_last": (references[2], False)}
    display_limits = {}
    for name, (values, raw) in images.items():
        pixels, high = display(values, valid, raw=raw)
        Image.fromarray(pixels).save(destination / f"{name}.png")
        display_limits[name] = high
    Image.fromarray(np.uint8(valid) * 255).save(destination / "valid_domain.png")

    # Fixed native details make edge and interior signal inspectable before ROI selection.
    centers = [(max(left + 32, min(right - 32, (left + right) // 2)),
                max(top + 32, min(bottom - 32, (top + bottom) // 2))),
               (left + 32, top + 32), (right - 32, bottom - 32)]
    details = Image.new("RGB", (3 * 256, 2 * 280), "black")
    draw = ImageDraw.Draw(details)
    detail_boxes = []
    for i, (cx, cy) in enumerate(centers):
        box = (cx - 32, cy - 32, cx + 32, cy + 32)
        detail_boxes.append(list(box))
        for row, name in enumerate(("raw_projection", "normalized_projection")):
            pixels, _ = display(images[name][0], valid, raw=images[name][1])
            tile = Image.fromarray(pixels[box[1]:box[3], box[0]:box[2]]).resize((256, 256), Image.Resampling.NEAREST)
            details.paste(tile.convert("RGB"), (i * 256, row * 280 + 24))
            draw.text((i * 256 + 3, row * 280 + 3), f"{name} {box}", fill="white")
    details.save(destination / "fixed_64px_details.png")
    return {"image_id": image_id, "shape_yx": [height, width], "valid_bounds": list(bounds),
            "display_limits": display_limits, "detail_boxes_xyxy": detail_boxes,
            "png_dir": str(destination)}


parser = argparse.ArgumentParser()
parser.add_argument("--job", type=Path, required=True)
parser.add_argument("--owner", choices=("72", "73"), required=True)
parser.add_argument("--mode", choices=("preflight", "render"), required=True)
parser.add_argument("--lock-root", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
job = json.loads(args.job.read_text())
owner = job["owners"][args.owner]
if args.mode == "preflight":
    print(json.dumps(current_versions(owner), sort_keys=True))
else:
    if args.lock_root is None or args.output is None:
        parser.error("--lock-root and --output are required for render")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    frozen = job["frozen_versions"][args.owner]
    if current_versions(owner) != frozen:
        raise ValueError("Owner products changed after version freeze")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "execution.json"
    receipt = {"owner": args.owner, "pid": os.getpid(),
               "start_tick": int(Path("/proc/self/stat").read_text().split()[21]),
               "started_at_ns": time.time_ns(), "status": "running", "images": []}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for image_id in owner["images"]:
            receipt["images"].append(render_image(owner, image_id, frozen["images"][image_id],
                                                   args.output, args.lock_root))
            if current_versions(owner) != frozen:
                raise ValueError(f"Owner products changed during render: {image_id}")
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
