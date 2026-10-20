"""Independent-image nearest neighbors in native pixel coordinates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def compute_neighbors(cells: pd.DataFrame, k: int = 20, workers: int = 1) -> pd.DataFrame:
    """Return nearest cell indices, excluding self by identity; absent slots are -1."""
    n = len(cells)
    indices = cells.cell_index.to_numpy()
    values = np.full((n, k), -1, np.int64)
    if n > 1:
        points = cells[["centroid_x", "centroid_y"]].to_numpy()
        distances, rows = cKDTree(points).query(points, k=min(n, k+1), workers=workers)
        for i in range(n):
            order = np.lexsort((indices[rows[i]], distances[i]))
            candidates = indices[rows[i][order]]
            selected = candidates[candidates != indices[i]][:k]
            values[i, :len(selected)] = selected
    result = pd.DataFrame(values, columns=[f"n{i:02d}" for i in range(1, k+1)])
    result.insert(0, "cell_index", indices)
    return result
