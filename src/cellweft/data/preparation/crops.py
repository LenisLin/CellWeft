"""Native grid windows and exact cell membership derived from instance pixels."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader


def axis_starts(length: int, size: int = 224, stride: int = 112) -> list[int]:
    """Cover one axis, appending a final edge-aligned origin when needed."""
    if length < size:
        return [0]
    starts = list(range(0, length-size+1, stride))
    if starts[-1] != length-size:
        starts.append(length-size)
    return starts


def grid_origins(shape: tuple[int, int], size: int = 224, stride: int = 112) -> list[tuple[int, int]]:
    """Return XY origins in row-major order; short images retain valid bounds."""
    h, w = shape
    return [(x, y) for y, x in product(axis_starts(h, size, stride), axis_starts(w, size, stride))]


def attach_crop_origins(mask: ImageReader, cells: pd.DataFrame,
                        size: int = 224, stride: int = 112) -> pd.DataFrame:
    """Return CellTable with every actual intersecting window in crop_origins.

    Each grid window is read once. Unique positive pixels directly establish
    membership, including partial instances; bbox-only intersections are omitted.
    """
    memberships = [[] for _ in range(len(cells))]
    label_to_row = {int(label): i for i, label in enumerate(cells.mask_label)}
    h, w = mask.shape[-2:]
    for x, y in grid_origins((h, w), size, stride):
        labels = np.unique(mask.read_window([0], (x, y, min(x+size, w), min(y+size, h))))
        for label in labels[labels > 0]:
            memberships[label_to_row[int(label)]].append({"x0": x, "y0": y})
    result = cells.copy()
    result["crop_origins"] = memberships
    return result


def read_crop(image: ImageReader, mask: ImageReader | None, origin: tuple[int, int],
              channels: list[int], size: int = 224) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
    """Read paired native windows with right/bottom padding and return valid H/W."""
    x, y = origin
    h, w = image.shape[-2:]
    vh, vw = min(size, h-y), min(size, w-x)
    window = (x, y, x+vw, y+vh)
    pixels = np.zeros((len(channels), size, size), np.float32)
    pixels[:, :vh, :vw] = image.read_window(channels, window)
    labels = None
    if mask is not None:
        labels = np.zeros((size, size), np.uint32)
        labels[:vh, :vw] = mask.read_window([0], window)[0]
    return pixels, labels, (vh, vw)
