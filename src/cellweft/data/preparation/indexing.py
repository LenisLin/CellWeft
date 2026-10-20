"""Study records and stable cell geometry in independent-image coordinates."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader, ImageRecord, SourceSpec
from cellweft.data.io.writers import json_value


STUDY_COLUMNS = ["study", "image_id", "channel_count", "pixel_size_um",
                 "image_path", "marker_list_path", "output_path"]


def build_study_index(study: str, sources: dict[str, SourceSpec],
                      marker_list_path: Path, output_path: Path) -> pd.DataFrame:
    """Build exactly seven columns, using source image IDs and per-image scales."""
    rows = []
    for image_id, source in sources.items():
        for name in (study, image_id):
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("study and image_id must be single safe directory names")
        record = ImageRecord(study, image_id, len(source.channels), source.pixel_size_um,
                             source.path if source.path.is_dir() else source.path.parent,
                             marker_list_path, output_path)
        rows.append(json_value(asdict(record)))
    return pd.DataFrame(rows, columns=STUDY_COLUMNS)


def build_cell_table(mask: ImageReader, study: str, image_id: str, mask_identity: str,
                     block_size: int = 512) -> pd.DataFrame:
    """Freeze ascending positive labels to dense per-image indices and pixel geometry.

    Sparse labels never determine allocation size. Bboxes are half-open XYXY.
    """
    label_set = set()
    for block in mask.iter_blocks([0], block_size):
        label_set.update(np.unique(block.values[0]).tolist())
    label_set.discard(0)
    labels = np.array(sorted(label_set), dtype=np.uint32)
    n = len(labels)
    area = np.zeros(n, np.int64)
    sx = np.zeros(n, np.float64)
    sy = np.zeros(n, np.float64)
    xmin = np.full(n, mask.shape[2], np.int64)
    ymin = np.full(n, mask.shape[1], np.int64)
    xmax = np.zeros(n, np.int64)
    ymax = np.zeros(n, np.int64)
    for block in mask.iter_blocks([0], block_size):
        yy, xx = np.nonzero(block.values[0])
        ids = np.searchsorted(labels, block.values[0][yy, xx])
        xx, yy = xx + block.window[0], yy + block.window[1]
        area += np.bincount(ids, minlength=n)
        sx += np.bincount(ids, weights=xx, minlength=n)
        sy += np.bincount(ids, weights=yy, minlength=n)
        np.minimum.at(xmin, ids, xx); np.minimum.at(ymin, ids, yy)
        np.maximum.at(xmax, ids, xx+1); np.maximum.at(ymax, ids, yy+1)
    prefix = ":".join(quote(v, safe="") for v in (study, image_id, mask_identity))
    return pd.DataFrame({
        "cell_index": np.arange(n, dtype=np.int64), "mask_label": labels,
        "cell_id": [f"{prefix}:{int(label)}" for label in labels],
        "centroid_x": sx / area, "centroid_y": sy / area, "area": area,
        "bbox_x0": xmin, "bbox_y0": ymin, "bbox_x1": xmax, "bbox_y1": ymax,
    })
