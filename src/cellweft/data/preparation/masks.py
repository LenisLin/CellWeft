"""The four agreed coarse mask checks, with an optional whole-image overview."""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

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


class _AlignedMaskReader:
    """Windowed inverse orientation, declared crop and padding of a source label map."""

    def __init__(self, base: ImageReader, source: MaskSource):
        self.base, self.source, self.dtype = base, source, base.dtype
        h, w = base.shape[-2:]
        rh, rw = (w, h) if source.quarter_turns % 2 else (h, w)
        self.crop = source.window or (0, 0, rw, rh)
        x0, y0, x1, y1 = self.crop
        l, t, r, b = source.padding_ltrb
        if min(l,t,r,b) < 0 or not (0 <= x0 < x1 <= rw and 0 <= y0 < y1 <= rh):
            raise ValueError("Mask crop/padding exceeds its declared oriented source frame")
        self.shape = (1, y1-y0+t+b, x1-x0+l+r)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0,y0,x1,y1 = window
        if channels != [0] or not (0 <= x0 < x1 <= self.shape[2] and 0 <= y0 < y1 <= self.shape[1]):
            raise ValueError("Invalid aligned mask window/channel")
        l,t,r,b = self.source.padding_ltrb
        a,c = max(x0,l), min(x1,self.shape[2]-r)
        d,e = max(y0,t), min(y1,self.shape[1]-b)
        out = np.zeros((1,y1-y0,x1-x0), self.dtype)
        if a >= c or d >= e:
            return out
        v,u = np.mgrid[d-t+self.crop[1]:e-t+self.crop[1], a-l+self.crop[0]:c-l+self.crop[0]]
        h,w = self.base.shape[-2:]
        k = self.source.quarter_turns % 4
        ox,oy = [(u,v),(w-1-v,u),(w-1-u,h-1-v),(v,h-1-u)][k]
        if self.source.flip_x: ox = w-1-ox
        if self.source.flip_y: oy = h-1-oy
        left,top,right,bottom = int(ox.min()),int(oy.min()),int(ox.max())+1,int(oy.max())+1
        values = self.base.read_window([0],(left,top,right,bottom))[0]
        out[0,d-y0:e-y0,a-x0:c-x0] = values[oy-top,ox-left]
        return out

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        return np.ones((window[3]-window[1],window[2]-window[0]),bool)

    def iter_blocks(self, channels: list[int], block_size: int):
        from cellweft.data.contracts import ImageBlock
        from cellweft.data.io.readers import windows
        for window in windows(self.shape[-2:],block_size):
            yield ImageBlock(window,self.read_window(channels,window),self.valid_window(window))


@contextmanager
def open_aligned_mask(source: MaskSource, cache_dir: Path, *, decode_budget: int = 1_073_741_824) -> Iterator[ImageReader]:
    """Read TIFF or explicitly selected NPZ labels; orient, crop, then add padding.

    NPZ decoding is admitted by uncompressed member size before allocation.
    The original mask is never modified; all handles/cache files are scoped.
    """
    from dataclasses import replace
    import tempfile
    import zipfile
    from cellweft.data.io.readers import open_source, tiff_source
    from cellweft.data.io.writers import write_image

    cache_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mask_",dir=cache_dir) as temporary:
        path, page = source.path, source.page
        if source.format == "npz":
            if not source.npz_key:
                raise ValueError("NPZ mask requires an explicit array key")
            with zipfile.ZipFile(path) as archive:
                if archive.getinfo(source.npz_key+".npy").file_size > decode_budget:
                    raise ValueError("NPZ mask decode exceeds configured memory budget")
            with np.load(path,allow_pickle=False) as archive:
                values = archive[source.npz_key]
                if source.selection:
                    values = values[tuple(slice(None) if i is None else i for i in source.selection)]
                if values.ndim != 2 or values.dtype.kind not in "iu":
                    raise ValueError("NPZ selection must produce one integer YX label map")
                path = Path(temporary)/"source.tiff"
                write_image(path,(1,*values.shape),values.dtype,
                            lambda _,w: values[w[1]:w[3],w[0]:w[2]])
            page = 0
        elif source.format != "tiff":
            raise ValueError("Source mask format must be tiff or npz")
        descriptor = replace(tiff_source(path,1),channels=(page,))
        with open_source(descriptor) as base:
            yield _AlignedMaskReader(base,source)


def finalize_mask(mask: ImageReader, measured: ImageReader, output: Path, *,
                  tile_size: int = 256, compression: str = "deflate") -> bool:
    """Write uint32 labels only inside the measured domain; return whether any remain.

    Reject invalid labels before casting. Sparse positive identities remain intact.
    """
    from cellweft.data.io.writers import write_image
    if mask.shape[-2:] != measured.shape[-2:] or mask.dtype.kind not in "iu":
        raise ValueError("Final mask must be an integer map on the input canvas")
    nonempty = False

    def read_plane(_, window):
        nonlocal nonempty
        values = mask.read_window([0],window)[0]
        if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
            raise ValueError("Final mask labels exceed uint32")
        values = np.where(measured.valid_window(window),values,0).astype(np.uint32)
        nonempty |= bool(np.any(values))
        return values

    write_image(output,(1,*mask.shape[-2:]),np.dtype('uint32'),read_plane,
                tile_size=tile_size,compression=compression)
    if not nonempty: output.unlink()
    return nonempty
