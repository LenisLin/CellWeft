"""Native grid windows and exact cell membership derived from instance pixels."""
from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from cellweft.data.contracts import CropSample, ImageReader


def axis_starts(length: int, size: int = 224, stride: int = 112) -> list[int]:
    """Cover one axis, appending a final edge-aligned origin when needed."""
    if length < size:
        return [0]
    starts = list(range(0, length-size+1, stride))
    if starts[-1] != length-size:
        starts.append(length-size)
    return starts


def grid_origins(shape: tuple[int, int], size: int = 224, stride: int | None = None, *,
                 purpose: Literal["train", "val", "test", "s4"] = "train") -> list[tuple[int, int]]:
    """Return row-major XY origins for training or fixed evaluation windows.

    Training preserves the edge-aligned 224/112 grid. Evaluation uses fixed
    224-pixel starts and lets ``read_crop`` pad the final window.
    """
    h, w = shape
    if purpose == "train":
        stride = 112 if stride is None else stride
        x_starts, y_starts = axis_starts(w, size, stride), axis_starts(h, size, stride)
    elif purpose in {"val", "test", "s4"}:
        if size != 224 or stride not in {None, 224}:
            raise ValueError("Validation, test and S4 windows require size and stride 224")
        x_starts, y_starts = range(0, w, 224), range(0, h, 224)
    else:
        raise ValueError("purpose must be 'train', 'val', 'test' or 's4'")
    return [(x, y) for y, x in product(y_starts, x_starts)]


def scan_mask_geometry(mask: ImageReader, size: int = 224, stride: int = 112,
                       block_size: int = 512) -> tuple[dict[int, int], dict[int, set[tuple[int, int]]]]:
    """Collect label areas and exact crop memberships during one block scan.

    A positive pixel belongs to every native grid window containing that pixel.
    Testing the block/window intersection keeps this check bounded without
    reopening the mask for one read per crop.
    """
    if size < 1 or stride < 1 or block_size < 1:
        raise ValueError("Mask geometry sizes must be positive")
    h, w = mask.shape[-2:]
    x_starts, y_starts = axis_starts(w, size, stride), axis_starts(h, size, stride)
    counts: dict[int, int] = {}
    memberships: dict[int, set[tuple[int, int]]] = {}
    for block in mask.iter_blocks([0], block_size):
        values = block.values[0]
        labels, areas = np.unique(values, return_counts=True)
        for label, area in zip(labels, areas):
            if label:
                counts[int(label)] = counts.get(int(label), 0) + int(area)
        bx0, by0, bx1, by1 = block.window
        x_candidates = [x for x in x_starts if x < bx1 and x + size > bx0]
        y_candidates = [y for y in y_starts if y < by1 and y + size > by0]
        for y in y_candidates:
            ys, ye = max(by0, y) - by0, min(by1, y + size) - by0
            for x in x_candidates:
                xs, xe = max(bx0, x) - bx0, min(bx1, x + size) - bx0
                labels = np.unique(values[ys:ye, xs:xe])
                for label in labels[labels > 0]:
                    memberships.setdefault(int(label), set()).add((x, y))
    return counts, memberships


def attach_crop_origins(mask: ImageReader, cells: pd.DataFrame,
                        size: int = 224, stride: int = 112) -> pd.DataFrame:
    """Return CellTable with every actual intersecting window in crop_origins.

    Unique positive pixels directly establish membership, including partial
    instances; bbox-only intersections are omitted.
    """
    _, label_memberships = scan_mask_geometry(mask, size, stride)
    memberships = []
    for label in cells.mask_label:
        origins = sorted(label_memberships.get(int(label), set()), key=lambda origin: (origin[1], origin[0]))
        memberships.append([{"x0": x, "y0": y} for x, y in origins])
    result = cells.copy()
    result["crop_origins"] = memberships
    return result


def read_crop(image: ImageReader, mask: ImageReader | None, origin: tuple[int, int],
              channels: list[int], size: int = 224, *, mask_identity: str | None = None) -> CropSample:
    """Read paired windows, carrying measured holes/padding and final mask identity."""
    x, y = origin
    h, w = image.shape[-2:]
    if size < 1 or not (0 <= x < w and 0 <= y < h):
        raise ValueError("Crop origin must lie inside the input canvas")
    vh, vw = min(size, h-y), min(size, w-x)
    window = (x, y, x+vw, y+vh)
    pixels = np.zeros((len(channels), size, size), np.float32)
    pixels[:, :vh, :vw] = image.read_window(channels, window)
    labels = None
    if mask is not None:
        labels = np.zeros((size, size), np.uint32)
        labels[:vh, :vw] = mask.read_window([0], window)[0]
    valid = np.zeros((1,size,size),bool)
    valid[0,:vh,:vw] = image.valid_window(window)
    return CropSample(pixels,labels,valid,origin,mask_identity)
