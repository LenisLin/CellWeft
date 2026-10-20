"""The four agreed coarse mask checks, with an optional whole-image overview."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cellweft.data.contracts import ImageReader, MaskSource


def check_existing_mask(mask: ImageReader, image_shape: tuple[int, int],
                        source: MaskSource, block_size: int = 512) -> dict:
    """Check declared whole-cell type, bounds and integer indexing once.

    Type and alignment judgments come from source evidence or the single overview;
    the numerical scan does not infer biological segmentation accuracy.
    """
    bounds = mask.shape[-2:] == image_shape and not source.alignment_issue
    integer = mask.dtype.kind in "iu"
    nonempty = False
    if bounds and integer:
        for block in mask.iter_blocks([0], block_size):
            data = block.values[0]
            integer &= bool(data.min(initial=0) >= 0 and data.max(initial=0) <= np.iinfo(np.uint32).max)
            nonempty |= bool(np.any(data > 0))
    whole_cell = source.mask_type == "whole_cell"
    problems = []
    if source.overview_judgment is not None and type(source.overview_judgment) is not bool:
        raise ValueError("overview_judgment must be true, false or null")
    if source.overview_judgment is False:
        problems.append("whole-image overview rejected the mask")
    elif source.overview_required and source.overview_judgment is None:
        problems.append("whole-image overview judgment required")
    if not whole_cell:
        problems.append("whole-cell type not established")
    if not bounds:
        problems.append(source.alignment_issue or "mask/image dimensions differ")
    if not integer:
        problems.append("mask labels must be nonnegative integers within uint32")
    if bounds and integer and not nonempty:
        problems.append("mask is empty; whole-image judgment required")
    return {"mask_type": source.mask_type, "dimensions_and_range": bool(bounds),
            "integer_index": bool(integer), "overview": source.overview_judgment,
            "problems": problems, "usable": not problems}


def make_mask_overview(image: ImageReader, mask: ImageReader, output: Path,
                       channel: int, max_side: int = 1024) -> None:
    """Save one whole-image overlay using bounded source blocks; no cell crops."""
    from PIL import Image
    h, w = image.shape[-2:]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    values = np.zeros(((h+step-1)//step, (w+step-1)//step), dtype=np.float32)
    labels = np.zeros(values.shape, dtype=np.uint32)
    for block in image.iter_blocks([channel], 512):
        x0, y0, x1, y1 = block.window
        yy = np.arange(((y0+step-1)//step)*step, y1, step)
        xx = np.arange(((x0+step-1)//step)*step, x1, step)
        values[np.ix_(yy//step, xx//step)] = block.values[0][np.ix_(yy-y0, xx-x0)]
        current = mask.read_window([0], block.window)[0]
        labels[np.ix_(yy//step, xx//step)] = current[np.ix_(yy-y0, xx-x0)]
    scale = float(np.percentile(values, 99.5))
    gray = np.clip(values / scale, 0, 1) if scale > 0 else values
    rgb = np.repeat((gray[..., None] * 255).astype(np.uint8), 3, axis=2)
    edge = np.zeros(labels.shape, bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rgb[edge & (labels > 0)] = [255, 80, 40]
    Image.fromarray(rgb).save(output)
