"""Measured-domain tissue estimation and strict training-window admission."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import tempfile

import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, binary_closing

from cellweft.data.contracts import ImageReader
from cellweft.data.io.readers import open_source, tiff_source, windows
from cellweft.data.io.writers import write_image
from cellweft.data.preparation.crops import grid_origins


@dataclass
class TissueEstimate:
    """Scoped tissue TIFF and its actual method; missing signal remains a gap."""

    path: Path | None
    metadata: dict
    gap: str | None = None


@dataclass
class SamplingResult:
    """Eligible origins, with None count until representative tissue review is complete."""

    origins: list[tuple[int, int]]
    count: int | None
    metadata: dict
    gap: str | None = None


@contextmanager
def estimate_tissue_region(image: ImageReader, channels: pd.DataFrame, cache_dir: Path,
                           *, bins: int = 256, block_size: int = 512) -> Iterator[TissueEstimate]:
    """Yield a temporary tissue map using global Otsu and exact block morphology.

    Candidate channels must vary on measured pixels. Four binary operations
    require a four-pixel halo, independent of TIFF storage tile boundaries.
    """
    if bins < 2 or block_size < 1:
        raise ValueError("Tissue histogram and block sizes must be positive")
    metadata = {"method":"max_otsu_open_close", "bins":bins, "kernel":3,
                "opening":1, "closing":1}
    candidates = channels[channels.tiff_page.notna() & (channels.nuclear | channels.role.eq('downstream'))]
    pages = []
    for page in candidates.tiff_page.astype(int):
        low, high = np.inf, -np.inf
        for block in image.iter_blocks([page],block_size):
            selected = block.values[0][block.valid]
            if selected.size:
                low,high = min(low,float(selected.min())),max(high,float(selected.max()))
        if low < high: pages.append(page)
    if not pages:
        yield TissueEstimate(None,metadata,"no nonconstant measured nuclear/protein signal")
        return
    cache_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tissue_',dir=cache_dir) as tmp:
        root = Path(tmp); projection_path = root/'projection.tiff'
        histogram = np.zeros(bins,np.int64)
        low, high = np.inf, -np.inf

        def project(_,window):
            nonlocal low,high
            values = image.read_window(pages,window).max(axis=0)
            valid = image.valid_window(window)
            selected = values[valid]
            if selected.size:
                low,high = min(low,float(selected.min())),max(high,float(selected.max()))
                histogram[:] += np.histogram(selected,bins=bins,range=(0,1))[0]
            return np.where(valid,values,0).astype(np.float32)

        write_image(projection_path,(1,*image.shape[-2:]),np.dtype('float32'),project)
        if low >= high or np.count_nonzero(histogram) < 2:
            yield TissueEstimate(None,metadata,"tissue projection has insufficient histogram variation")
            return
        centers = (np.arange(bins)+.5)/bins
        weights = np.cumsum(histogram).astype(np.float64)
        totals = np.cumsum(histogram*centers)
        other = weights[-1]-weights
        scores = np.full(bins,-np.inf)
        valid_splits = (weights > 0) & (other > 0)
        scores[valid_splits] = ((totals[-1]*weights[valid_splits]-totals[valid_splits]*weights[-1])**2
                                / (weights[valid_splits]*other[valid_splits]))
        threshold = float(centers[np.argmax(scores)])
        metadata.update(threshold=threshold,pages=pages)
        path = root/'tissue.tiff'
        h,w = image.shape[-2:]
        with open_source(tiff_source(projection_path,1)) as projection:
            def morphology(_,window):
                x0,y0,x1,y1 = window
                a,b,c,d = max(0,x0-4),max(0,y0-4),min(w,x1+4),min(h,y1+4)
                valid = image.valid_window((a,b,c,d))
                binary = (projection.read_window([0],(a,b,c,d))[0] > threshold) & valid
                binary = binary_closing(binary_opening(binary,structure=np.ones((3,3),bool)),
                                        structure=np.ones((3,3),bool)) & valid
                return binary[y0-b:y1-b,x0-a:x1-a].astype(np.uint8)
            write_image(path,(1,h,w),np.dtype('uint8'),morphology)
        yield TissueEstimate(path,metadata)


def select_training_origins(image: ImageReader, channels: pd.DataFrame, tissue: TissueEstimate,
                            *, reviewed: bool, size: int = 224, stride: int = 112,
                            min_tissue_fraction: float = .50) -> SamplingResult:
    """Select >50% tissue windows with actual nonzero downstream measurements.

    Unknown/unreviewed tissue is pending, distinct from a completed zero count.
    Full cell memberships are maintained separately and never filtered here.
    """
    if (size,stride,min_tissue_fraction) != (224,112,.50):
        raise ValueError("Training admission requires 224/112 and strictly >0.50")
    metadata = {**tissue.metadata,'reviewed':reviewed,'size':size,'stride':stride,
                'min_tissue_fraction':min_tissue_fraction,'comparison':'strictly_greater'}
    if not reviewed or tissue.path is None:
        gap = tissue.gap or "representative tissue review pending"
        return SamplingResult([],None,metadata,gap)
    pages = channels.loc[channels.role.eq('downstream') & channels.tiff_page.notna(),'tiff_page'].astype(int).tolist()
    accepted = []
    if pages:
        h,w = image.shape[-2:]
        with open_source(tiff_source(tissue.path,1)) as reader:
            for x,y in grid_origins((h,w),size,stride):
                window = (x,y,min(x+size,w),min(y+size,h))
                valid = image.valid_window(window)
                area = np.count_nonzero((reader.read_window([0],window)[0] > 0) & valid)
                if area <= size*size*min_tissue_fraction:
                    continue
                if any(np.any(image.read_window([page],window)[0][valid]) for page in pages):
                    accepted.append((x,y))
    return SamplingResult(accepted,len(accepted),metadata)
