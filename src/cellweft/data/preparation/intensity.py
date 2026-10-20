"""Exact whole-image P99.5 with bounded blocks and an SSD-backed order statistic."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader, PreparationSettings, SharedImageResult
from cellweft.data.io.writers import write_image


def channel_percentile(reader: ImageReader, channel: int, cache: Path, block_size: int) -> tuple[float | None, str]:
    """Compute exact linear P99.5 over all measured pixels, including zeros.

    The temporary float64 memmap is partitioned in place; blocks are not averaged.
    Nonfinite measured pixels fail only this channel.
    """
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="percentile_", dir=cache) as directory:
        capacity = reader.shape[1] * reader.shape[2]
        values = np.memmap(Path(directory) / "pixels.bin", mode="w+", dtype="float64", shape=(capacity,))
        count = 0
        try:
            for block in reader.iter_blocks([channel], block_size):
                selected = block.values[0][block.valid]
                if not np.isfinite(selected).all():
                    return None, "nonfinite measured pixels"
                values[count:count+len(selected)] = selected
                count += len(selected)
            if count == 0:
                return None, "no measured pixels"
            rank = (count - 1) * 0.995
            lo, hi = math.floor(rank), math.ceil(rank)
            sample = values[:count]
            sample.partition((lo, hi))
            q = float(sample[lo] + (sample[hi] - sample[lo]) * (rank - lo))
            return q, "" if q > 0 else "q995 <= 0"
        finally:
            values._mmap.close()


def prepare_shared_image(reader: ImageReader, mapping: pd.DataFrame, output: Path,
                         settings: PreparationSettings,
                         normalized_q995: tuple[float | None, ...] | None = None) -> SharedImageResult:
    """Compute shared float32 intensities and all-source-channel mappings.

    Supplied q995 values declare confirmed upstream use of the identical formula
    and statistical domain. Those pixels are copied, not normalized again.
    Excluded/failed channels keep rows with null output references.
    """
    if len(mapping) != reader.shape[0]:
        raise ValueError("Panel channel count does not match the source reader")
    if normalized_q995 is not None and len(normalized_q995) != reader.shape[0]:
        raise ValueError("Confirmed upstream q995 must cover every source channel")
    table = mapping.copy()
    table["q995"] = np.nan
    table["tiff_page"] = pd.Series([pd.NA] * len(table), dtype="Int64")
    table["expression_column"] = pd.Series([None] * len(table), dtype=object)
    table["numerical_reason"] = ""
    channels = []
    for row in table.itertuples():
        i = int(row.source_channel_index)
        if row.role == "exclude":
            table.at[i, "numerical_reason"] = "manual exclusion"
            continue
        if normalized_q995 is None:
            q, reason = channel_percentile(reader, i, settings.cache_dir, settings.block_size)
        else:
            q = normalized_q995[i]
            reason = "upstream channel unavailable" if q is None else ""
            if q is not None and not np.isfinite(q):
                raise ValueError("Confirmed upstream q995 must be finite or null")
            if q is not None and q <= 0:
                reason = "q995 <= 0"
            if not reason:
                measured = 0
                for block in reader.iter_blocks([i], settings.block_size):
                    values = block.values[0][block.valid]
                    measured += values.size
                    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                        raise ValueError("Confirmed normalized pixels must be finite and within [0,1]")
                if not measured:
                    reason = "no measured pixels"
        table.at[i, "q995"] = q
        table.at[i, "numerical_reason"] = reason
        if reason:
            continue
        table.at[i, "tiff_page"] = len(channels)
        if row.role == "downstream":
            table.at[i, "expression_column"] = row.marker_id
        channels.append(i)
    if not channels:
        return SharedImageResult(None, table, reader.shape[-2:])

    def normalized(page, window):
        source_channel = channels[page]
        pixels = reader.read_window([source_channel], window)[0].astype(np.float64)
        valid = reader.valid_window(window)
        pixels[~valid] = 0
        if normalized_q995 is not None:
            return pixels.astype(np.float32)
        return np.clip(pixels / table.at[source_channel, "q995"], 0, 1).astype(np.float32)

    write_image(output, (len(channels), *reader.shape[-2:]), np.dtype("float32"), normalized,
                tile_size=settings.tile_size, compression=settings.compression)
    return SharedImageResult(output, table, reader.shape[-2:])
