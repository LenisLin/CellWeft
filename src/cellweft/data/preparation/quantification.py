"""Whole-cell means from shared intensities, before any training crop or resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader


def quantify_proteins(image: ImageReader, mask: ImageReader, cells: pd.DataFrame,
                      channels: pd.DataFrame, block_size: int = 512) -> pd.DataFrame:
    """Accumulate float64 sums for each cell, dividing by all instance pixels.

    Channel rows use nullable output TIFF pages; zero intensities contribute to
    the whole-cell denominator. Result columns are registry marker IDs.
    """
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError("Shared image and final mask must have equal dimensions")
    proteins = channels[(channels.role == "downstream") & channels.tiff_page.notna()]
    labels = cells.mask_label.to_numpy()
    result = cells[["cell_index", "mask_label"]].copy()
    for row in proteins.itertuples():
        sums = np.zeros(len(cells), np.float64)
        for block in mask.iter_blocks([0], block_size):
            foreground = block.values[0] > 0
            ids = np.searchsorted(labels, block.values[0][foreground])
            signal = image.read_window([int(row.tiff_page)], block.window)[0]
            sums += np.bincount(ids, weights=signal[foreground].astype(np.float64), minlength=len(cells))
        result[str(row.marker_id)] = (sums / cells.area.to_numpy()).astype(np.float32)
    return result
