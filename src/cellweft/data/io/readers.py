"""Explicit source decoding and local-coordinate window reads."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from cellweft.data.contracts import ImageBlock, SourceSpec, Window


def windows(shape: tuple[int, int], size: int) -> Iterator[Window]:
    """Yield disjoint XYXY blocks covering every image pixel exactly once."""
    h, w = shape
    for y in range(0, h, size):
        for x in range(0, w, size):
            yield x, y, min(x + size, w), min(y + size, h)


def read_tiff_window(tif: tifffile.TiffFile, page_index: int, window: Window,
                     *, page: tifffile.TiffPage | None = None,
                     segment_cache: dict | None = None) -> np.ndarray:
    """Decode only intersecting scalar TIFF tiles/strips into a YX window.

    Uncompressed contiguous pages map only the requested rows, avoiding repeated
    whole-strip copies. Other layouts decode intersecting segments.
    Color/sample-packed pages require explicit source adaptation.
    """
    if page is None:
        page = tif.pages[page_index].aspage()
    if page.ndim != 2 or page.samplesperpixel != 1:
        raise ValueError("Expected scalar YX TIFF page")
    x0, y0, x1, y1 = window
    h, w = page.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"Window {window} outside TIFF YX {page.shape}")
    if (page.is_final and tif.filehandle.is_file and page.dtype is not None
            and sum(page.databytecounts) == page.nbytes):
        if page.dataoffsets[0] + page.nbytes > tif.filehandle.size:
            raise ValueError("TIFF image data exceeds the logical file size")
        rows = tif.filehandle.memmap_array(
            tif.byteorder + page.dtype.char, (y1 - y0, w),
            offset=page.dataoffsets[0] + y0 * w * page.dtype.itemsize,
            mode="r",
        )
        return np.array(rows[:, x0:x1], dtype=page.dtype, copy=True)
    out = np.zeros((y1 - y0, x1 - x0), dtype=page.dtype)
    th, tw = (page.tilelength, page.tilewidth) if page.is_tiled else (page.rowsperstrip, w)
    columns = (w + tw - 1) // tw
    for ty in range(y0 // th, (y1 - 1) // th + 1):
        for tx in range(x0 // tw, (x1 - 1) // tw + 1):
            i = ty * columns + tx
            key = (tif, page_index, i)
            if segment_cache is not None and segment_cache["key"] == key:
                data = segment_cache["data"]
            else:
                if segment_cache is not None:
                    segment_cache["key"] = segment_cache["data"] = None
                fh = tif.filehandle
                fh.seek(page.dataoffsets[i])
                data, _, _ = page.decode(fh.read(page.databytecounts[i]), i,
                                          jpegtables=page.jpegtables)
                if data is None:
                    raise ValueError(f"Missing TIFF segment {i}; declare the acquisition validity explicitly")
                if segment_cache is not None and data.nbytes <= 128 * 1024**2:
                    segment_cache["key"], segment_cache["data"] = key, data
            segment = data[0, :, :, 0]
            sy, sx = ty * th, tx * tw
            a, b, c, d = max(x0, sx), max(y0, sy), min(x1, sx + tw), min(y1, sy + th)
            out[b-y0:d-y0, a-x0:c-x0] = segment[b-sy:d-sy, a-sx:c-sx]
    return out


class SourceReader:
    """Context-owned source reader; channel indices refer to SourceSpec.channels."""

    def __init__(self, source: SourceSpec, stack: ExitStack):
        self.source = source
        self._valid = None
        self._tiffs = []
        self._tiff_segment_cache = {"key": None, "data": None}
        stack.callback(self._tiff_segment_cache.clear)
        if source.kind in {"tiff", "tiff_planes"}:
            paths = source.paths if source.kind == "tiff_planes" else (source.path,)
            self._tiffs = [stack.enter_context(tifffile.TiffFile(p)) for p in paths]
            if source.kind == "tiff_planes" and len(paths) != len(source.channels):
                raise ValueError("TIFF plane paths and channel indices must have equal length")
            self._pages = [self._tiffs[i if source.kind == "tiff_planes" else 0].pages[c].aspage()
                           for i, c in enumerate(source.channels)]
            pages = self._pages
            native = pages[0].shape
            if len(native) != 2 or any(p.shape != native or p.dtype != pages[0].dtype for p in pages):
                raise ValueError("Selected TIFF channels must have matching scalar shape and dtype")
            self.dtype = pages[0].dtype
        elif source.kind == "zarr":
            import zarr
            self._array = zarr.open_array(str(source.path), mode="r")
            if self._array.ndim not in (3, 4) or (self._array.ndim == 4 and source.z_index is None):
                raise ValueError("Zarr source must be CYX, or CZYX with an explicit z_index")
            native, self.dtype = self._array.shape[-2:], self._array.dtype
            if hasattr(self._array.store, "close"):
                stack.callback(self._array.store.close)
        elif source.kind == "ims":
            import h5py
            self._h5 = stack.enter_context(h5py.File(source.path, "r"))
            self._datasets = [self._h5[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {i}/Data"]
                              for i in source.channels]
            if source.z_index is None:
                raise ValueError("IMS requires an explicit z_index")
            native = source.valid_shape or self._datasets[0].shape[-2:]
            self.dtype = self._datasets[0].dtype
        elif source.kind == "mcd":
            from readimc import MCDFile
            self._mcd = stack.enter_context(MCDFile(source.path))
            acquisitions = [a for slide in self._mcd.slides for a in slide.acquisitions
                            if a.id == source.acquisition_id]
            if len(acquisitions) != 1:
                raise ValueError("MCD acquisition_id must select exactly one acquisition")
            self._acquisition = acquisitions[0]
            meta = self._acquisition.metadata
            native = (int(meta["MaxY"]), int(meta["MaxX"]))
            self.dtype = np.dtype("float32")
            # MCD absence is not measured zero; use recorded acquisition coordinates.
            start, end = int(meta["DataStartOffset"]), int(meta["DataEndOffset"])
            nc = self._acquisition.num_channels + 3
            if int(meta["ValueBytes"]) != 4 or (end-start) % (nc*4):
                raise ValueError("MCD acquisition has incomplete pixel records")
            raw = np.memmap(source.path, dtype="float32", mode="r", offset=start,
                            shape=((end-start)//(nc*4), nc))
            stack.callback(raw._mmap.close)
            coords = raw[:, :2]
            if not np.isfinite(coords).all() or np.any(coords != np.floor(coords)):
                raise ValueError("MCD has invalid acquisition coordinates")
            x, y = coords[:, 0].astype(int), coords[:, 1].astype(int)
            if np.any((x < 0) | (x >= native[1]) | (y < 0) | (y >= native[0])):
                raise ValueError("MCD acquisition coordinates exceed declared image")
            self._valid = np.zeros(native, dtype=bool)
            self._valid[y, x] = True
            if int(self._valid.sum()) != len(x):
                raise ValueError("Duplicate MCD pixel coordinates need source-specific resolution")
            self._mcd_records = raw
            self._record_index = np.full(native, -1, dtype=np.int64)
            self._record_index[y, x] = np.arange(len(x))
        else:
            raise ValueError(f"Unsupported source kind: {source.kind}")
        h, w = native
        self.origin = source.window or (0, 0, w, h)
        x0, y0, x1, y1 = self.origin
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Source ROI exceeds its acquisition")
        self.shape = (len(source.channels), y1-y0, x1-x0)
        if source.valid_mask:
            self._valid_tif = stack.enter_context(tifffile.TiffFile(source.valid_mask))
            self._valid_page = self._valid_tif.pages[0].aspage()
        else:
            self._valid_tif = None
            self._valid_page = None
        self._sample_tif = (stack.enter_context(tifffile.TiffFile(source.sample_region))
                            if source.sample_region else None)
        self._sample_page = self._sample_tif.pages[0].aspage() if self._sample_tif else None

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        """Return independent CYX pixels in the local image coordinate frame."""
        x0, y0, x1, y1 = window
        if not (0 <= x0 < x1 <= self.shape[2] and 0 <= y0 < y1 <= self.shape[1]):
            raise ValueError("Read window exceeds local image")
        ox, oy = self.origin[:2]
        region = (x0+ox, y0+oy, x1+ox, y1+oy)
        native = [self.source.channels[i] for i in channels]
        if self.source.kind in {"tiff", "tiff_planes"}:
            return np.stack([read_tiff_window(self._tiffs[i if self.source.kind == "tiff_planes" else 0],
                                               self.source.channels[i], region, page=self._pages[i],
                                               segment_cache=self._tiff_segment_cache)
                             for i in channels])
        x0, y0, x1, y1 = region
        if self.source.kind == "zarr":
            selection = (native, slice(y0, y1), slice(x0, x1))
            if self._array.ndim == 4:
                selection = (native, self.source.z_index, slice(y0, y1), slice(x0, x1))
            return np.asarray(self._array.oindex[selection])
        if self.source.kind == "ims":
            return np.stack([np.asarray(self._datasets[i][self.source.z_index, y0:y1, x0:x1])
                             for i in channels])
        # Reuse the coordinate index across tiles; readimc otherwise scans and
        # copies all acquisition records for every channel/window request.
        indices = self._record_index[y0:y1, x0:x1]
        valid = indices >= 0
        result = np.zeros((len(native), y1-y0, x1-x0), dtype=self.dtype)
        for output, channel in enumerate(native):
            result[output, valid] = self._mcd_records[indices[valid], channel + 3]
        return result

    def valid_window(self, window: Window) -> np.ndarray:
        """Intersect measured pixels and an optional sample domain, independent of signal."""
        x0, y0, x1, y1 = window
        ox, oy = self.origin[:2]
        result = np.ones((y1-y0, x1-x0), dtype=bool)
        if self.source.valid_bounds is not None:
            a, b, c, d = self.source.valid_bounds
            yy, xx = np.ogrid[y0:y1, x0:x1]
            result &= (xx >= a) & (xx < c) & (yy >= b) & (yy < d)
        if self._valid is not None:
            result &= self._valid[y0+oy:y1+oy, x0+ox:x1+ox]
        for domain, page in ((self._valid_tif, self._valid_page),
                             (self._sample_tif, self._sample_page)):
            if domain is None:
                continue
            declared = read_tiff_window(domain, 0, (x0+ox,y0+oy,x1+ox,y1+oy), page=page)
            if not np.isin(declared,[0,1]).all():
                raise ValueError("Processing domain must contain only zero and one")
            result &= declared.astype(bool)
        return result

    def iter_blocks(self, channels: list[int], block_size: int) -> Iterator[ImageBlock]:
        """Yield disjoint measured-pixel blocks in row-major order."""
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


@contextmanager
def open_source(source: SourceSpec) -> Iterator[SourceReader]:
    """Open source resources and close all handles on normal exit or failure."""
    with ExitStack() as stack:
        yield SourceReader(source, stack)


def tiff_source(path: Path, channel_count: int, window: Window | None = None) -> SourceSpec:
    """Describe a prepared TIFF or scalar mask for shared window algorithms."""
    return SourceSpec(path.stem, "prepared", "tiff", path, tuple(range(channel_count)), window=window)
