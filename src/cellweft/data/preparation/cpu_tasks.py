"""Task-local CPU products from a shared image and an already finalized mask."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import PreparationSettings
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.io.writers import json_value, write_json, write_table
from cellweft.data.preparation.crops import attach_crop_origins
from cellweft.data.preparation.indexing import build_cell_table
from cellweft.data.preparation.intensity import prepare_shared_image
from cellweft.data.preparation.neighbors import compute_neighbors
from cellweft.data.preparation.quantification import quantify_proteins
from cellweft.data.preparation.validation import validate_products


def execute_cpu_task(task: dict) -> dict:
    """Compute, validate and publish task-local tables; return the final receipt.

    Required keys are task_id, study, image_id, mask_identity, image_path,
    mask_path, channels_path and output_dir. Explicit paths resolve from the
    process working directory. The image is shared normalized float32 CYX TIFF;
    the aligned uint32 mask must already be cleared outside the processing domain.
    Optional valid_mask/sample_region TIFFs and half-open XYXY valid_bounds
    intersect that domain. block_size defaults to 512, cpu_workers to 4 (1..4).

    output_dir must be empty or owned by the same task identity. Retries recompute
    all tables, withdrawing result.json first. Only newly generated product paths
    appear in the receipt; result.json is installed last. Errors propagate and
    leave no successful receipt for the failed attempt. Inputs remain read-only.
    The controller owns process concurrency, transport and formal publication;
    the CLI configures native library thread limits before importing this module.
    """
    kind = task.get("kind", "quantification")
    if kind == "normalization":
        return _execute_normalization(task)
    if kind != "quantification":
        raise ValueError(f"Unsupported CPU task kind: {kind}")
    started = perf_counter()
    identity = {key: task[key] for key in ("task_id", "study", "image_id", "mask_identity")}
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("Task identity fields must be nonempty strings")
    if not isinstance(task["output_dir"], (str, Path)) or not str(task["output_dir"]).strip():
        raise ValueError("output_dir must be an explicit directory path")
    output = Path(task["output_dir"]).resolve()
    input_keys = ("image_path", "mask_path", "channels_path", "valid_mask", "sample_region")
    # Reject aliases before touching a lock, ownership marker or old receipt.
    for key in input_keys:
        value = task.get(key)
        if isinstance(value, (str, Path)) and str(value).strip() and Path(value).resolve().parent == output:
            raise ValueError("Inputs must be outside the task output directory")
    receipt_path = output / "result.json"
    owner_path = output / ".cpu-task.json"
    with resource_lock(output, name=".cpu-task.lock"):
        if owner_path.exists():
            if json.loads(owner_path.read_text()) != identity:
                raise ValueError("Output directory belongs to a different task identity")
        elif any(path.name != ".cpu-task.lock" for path in output.iterdir()):
            raise ValueError("Output directory contains products without task ownership")
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError("Existing result belongs to a different task identity")
        # Completion is withdrawn before validating or computing a new attempt.
        receipt_path.unlink(missing_ok=True)
        write_json(owner_path, identity)
        block_size = task.get("block_size", 512)
        cpu_workers = task.get("cpu_workers", 4)
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if type(cpu_workers) is not int or not 1 <= cpu_workers <= 4:
            raise ValueError("cpu_workers must be an integer between 1 and 4")
        paths = {}
        for key in input_keys:
            value = task.get(key) if key in {"valid_mask", "sample_region"} else task[key]
            if value is None and key in {"valid_mask", "sample_region"}:
                paths[key] = None
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{key} must be an explicit file path")
            paths[key] = Path(value).resolve(strict=True)
        bounds = task.get("valid_bounds")
        if bounds is not None:
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 4 or any(type(v) is not int for v in bounds):
                raise ValueError("valid_bounds must contain four XYXY integers")
            bounds = tuple(bounds)
        channels = pd.read_parquet(paths["channels_path"])
        source = replace(
            tiff_source(paths["image_path"], int(channels.tiff_page.notna().sum())),
            valid_mask=paths["valid_mask"], sample_region=paths["sample_region"],
            valid_bounds=bounds,
        )
        with TemporaryDirectory(prefix=".cpu-attempt-", dir=output) as temporary:
            staging = Path(temporary)
            with open_source(source) as image, open_source(tiff_source(paths["mask_path"], 1)) as mask:
                shape = image.shape[-2:]
                if mask.shape[-2:] != shape:
                    raise ValueError("Shared image and final mask must have equal dimensions")
                if image.dtype != np.dtype("float32") or mask.dtype != np.dtype("uint32"):
                    raise ValueError("Shared image must be float32 and final mask must be uint32")
                h, w = shape
                if bounds is not None and not (0 <= bounds[0] < bounds[2] <= w and 0 <= bounds[1] < bounds[3] <= h):
                    raise ValueError("valid_bounds must lie inside the image canvas")
                for key in ("valid_mask", "sample_region"):
                    if paths[key] is not None:
                        with tifffile.TiffFile(paths[key]) as tif:
                            if (len(tif.pages) != 1 or tif.pages[0].ndim != 2
                                    or tif.pages[0].dtype != np.dtype("uint8")):
                                raise ValueError(f"{key} must be one uint8 YX page")
                        with open_source(tiff_source(paths[key], 1)) as domain:
                            if domain.shape[-2:] != shape:
                                raise ValueError(f"{key} must match the image canvas")
                for block in mask.iter_blocks([0], block_size):
                    if np.any((block.values[0] > 0) & ~image.valid_window(block.window)):
                        raise ValueError("Final mask has foreground outside the processing domain")
                cells = attach_crop_origins(
                    mask, build_cell_table(mask, task["study"], task["image_id"], task["mask_identity"], block_size),
                )
                generated = {"cells": str(staging / "cells.parquet")}
                write_table(Path(generated["cells"]), cells)
                if ((channels.role == "downstream") & channels.tiff_page.notna()).any():
                    generated["expression"] = str(staging / "expression.parquet")
                    write_table(Path(generated["expression"]), quantify_proteins(image, mask, cells, channels, block_size))
                generated["neighbors"] = str(staging / "neighbors.parquet")
                write_table(Path(generated["neighbors"]), compute_neighbors(cells, workers=cpu_workers))
            validate_products({
                "image": str(paths["image_path"]), "channels": str(paths["channels_path"]),
                "mask": str(paths["mask_path"]), **generated,
            }, shape)
            products = {key: str(output / Path(path).name) for key, path in generated.items()}
            receipt = {**identity, "products": products, "elapsed_seconds": perf_counter() - started}
            write_json(staging / "result.json", receipt)
            for key, path in generated.items():
                os.replace(path, products[key])
            if "expression" not in generated:
                (output / "expression.parquet").unlink(missing_ok=True)
            os.replace(staging / "result.json", receipt_path)
    return receipt


def _execute_normalization(task: dict) -> dict:
    """Publish exact shared intensities and the complete source channel table."""
    from dataclasses import replace

    started = perf_counter()
    identity = {key: task[key] for key in ("task_id", "study", "image_id")}
    if any(not isinstance(value, str) or not value.strip() for value in identity.values()):
        raise ValueError("Task identity fields must be nonempty strings")
    image_request = task["image_request"]
    if not isinstance(image_request, dict) or not image_request:
        raise ValueError("image_request must be a complete JSON object")
    if any(key in task for key in ("mask_path", "mask_identity", "channels_path")):
        raise ValueError("Normalization task cannot carry mask or quantification inputs")
    if (not isinstance(task.get("image_path"), str) or Path(task["image_path"]).name != "input.tiff"
            or not isinstance(task.get("panel_path"), str) or Path(task["panel_path"]).name != "panel.parquet"):
        raise ValueError("Normalization requires input.tiff and panel.parquet")
    count = task["channel_count"]
    if type(count) is not int or count < 1:
        raise ValueError("channel_count must be a positive integer")
    tile_size, block_size = task["tile_size"], task["block_size"]
    if type(tile_size) is not int or tile_size < 16 or tile_size % 16:
        raise ValueError("tile_size must be a positive multiple of 16")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    workers = task["cpu_workers"]
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("cpu_workers must be an integer between 1 and 4")
    if not isinstance(task["compression"], str) or not task["compression"]:
        raise ValueError("compression must be nonempty")
    scratch_bytes = task["scratch_bytes"]
    if type(scratch_bytes) is not int or not 0 < scratch_bytes <= 16 * 1024**3:
        raise ValueError("normalization scratch_bytes must be within 1..16 GiB")
    output = Path(task["output_dir"]).resolve()
    paths = {}
    for key in ("image_path", "panel_path", "valid_mask", "sample_region"):
        value = task.get(key)
        if value is None and key in {"valid_mask", "sample_region"}:
            paths[key] = None
            continue
        if not isinstance(value, str) or not value or Path(value).name in {"", ".", ".."}:
            raise ValueError(f"{key} must be an input filename")
        candidate = Path(value)
        paths[key] = (candidate if candidate.is_absolute() else output.parent / candidate).resolve(strict=True)
        if paths[key].parent != output.parent:
            raise ValueError("Normalization inputs must be beside the task output directory")
    bounds = task.get("valid_bounds")
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or any(type(v) is not int for v in bounds):
            raise ValueError("valid_bounds must contain four XYXY integers")
        bounds = tuple(bounds)
    normalized_q995 = task.get("normalized_q995")
    if normalized_q995 is not None:
        if not isinstance(normalized_q995, (list, tuple)) or len(normalized_q995) != count:
            raise ValueError("normalized_q995 must cover every input channel")
        if any(q is not None and (type(q) not in (int, float) or not np.isfinite(q)) for q in normalized_q995):
            raise ValueError("normalized_q995 values must be finite or null")
        normalized_q995 = tuple(normalized_q995)
    owner_path = output / ".cpu-task.json"
    receipt_path = output / "result.json"
    owner = {**identity, "kind": "normalization", "image_request": image_request}
    with resource_lock(output, name=".cpu-task.lock"):
        if owner_path.exists():
            if json.loads(owner_path.read_text()) != owner:
                raise ValueError("Output directory belongs to a different task identity")
        elif any(path.name != ".cpu-task.lock" for path in output.iterdir()):
            raise ValueError("Output directory contains products without task ownership")
        receipt_path.unlink(missing_ok=True)
        write_json(owner_path, owner)
        mapping = pd.read_parquet(paths["panel_path"])
        scientific = image_request.get("panel")
        if not isinstance(scientific, list) or not scientific:
            raise ValueError("image_request must contain the complete scientific panel")
        fields = ("panel_id", "source_channel_index", "output_channel_index", "keep_channel",
                  "source_name", "marker_id", "role", "nuclear", "body")
        actual = json_value(mapping[[key for key in fields if key in mapping]].astype(object)
                            .where(pd.notna(mapping), None).to_dict("records"))
        if actual != scientific:
            raise ValueError("panel.parquet differs from image_request panel")
        with tifffile.TiffFile(paths["image_path"]) as tif:
            if len(tif.pages) != count or any(page.ndim != 2 or page.samplesperpixel != 1
                                              or page.shape != tif.pages[0].shape
                                              or page.dtype != tif.pages[0].dtype for page in tif.pages):
                raise ValueError("input.tiff pages differ from declared channel_count or canvas")
        settings = PreparationSettings(cache_dir=output, tile_size=tile_size,
                                       block_size=block_size, compression=task["compression"],
                                       cpu_workers=workers)
        source = replace(tiff_source(paths["image_path"], count),
                         valid_mask=paths["valid_mask"], sample_region=paths["sample_region"],
                         valid_bounds=bounds)
        with TemporaryDirectory(prefix=".cpu-attempt-", dir=output) as temporary:
            staging = Path(temporary)
            settings = replace(settings, cache_dir=staging)
            with open_source(source) as reader:
                shape = reader.shape[-2:]
                if bounds is not None and not (0 <= bounds[0] < bounds[2] <= shape[1]
                                               and 0 <= bounds[1] < bounds[3] <= shape[0]):
                    raise ValueError("valid_bounds must lie inside the image canvas")
                for key in ("valid_mask", "sample_region"):
                    if paths[key] is not None:
                        with open_source(tiff_source(paths[key], 1)) as domain:
                            if domain.shape[-2:] != shape:
                                raise ValueError(f"{key} must match the image canvas")
                shared = prepare_shared_image(reader, mapping, staging / "image.tiff",
                                              settings, normalized_q995)
            write_table(staging / "channels.parquet", shared.channels)
            staged = {"channels": str(staging / "channels.parquet")}
            if shared.path is not None:
                staged["image"] = str(shared.path)
            validate_products(staged, shape)
            products = {key: str(output / Path(path).name) for key, path in staged.items()}
            receipt = {**owner, "shape_yx": list(shape), "products": products,
                       "elapsed_seconds": perf_counter() - started}
            write_json(staging / "result.json", receipt)
            for key, path in staged.items():
                os.replace(path, products[key])
            if "image" not in staged:
                (output / "image.tiff").unlink(missing_ok=True)
            os.replace(staging / "result.json", receipt_path)
    return receipt
