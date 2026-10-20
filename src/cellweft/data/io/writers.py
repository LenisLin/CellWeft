"""Stream scalar TIFF pages and publish closed products with metadata last."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

from cellweft.data.contracts import Window
from cellweft.data.io.readers import windows


def json_value(value):
    """Convert scientific records to JSON values without nonfinite substitutes."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: dict) -> None:
    """Atomically replace a small JSON record after closing its temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_image(path: Path, shape: tuple[int, int, int], dtype: np.dtype,
                read_plane: Callable[[int, Window], np.ndarray], *,
                tile_size: int = 256, compression: str = "deflate") -> None:
    """Stream CYX as scalar tiled BigTIFF pages; callback returns each native tile.

    Padding is storage-only. The declared TIFF page shape retains real H/W.
    """
    if tile_size <= 0 or tile_size % 16:
        raise ValueError("TIFF tile_size must be a positive multiple of 16")
    c, h, w = shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for channel in range(c):
            def tiles():
                for window in windows((h, w), tile_size):
                    values = read_plane(channel, window)
                    x0, y0, x1, y1 = window
                    if values.shape != (y1-y0, x1-x0):
                        raise ValueError("TIFF callback returned mismatched window shape")
                    tile = np.zeros((tile_size, tile_size), dtype=dtype)
                    tile[:y1-y0, :x1-x0] = values
                    yield tile
            writer.write(data=tiles(), shape=(h, w), dtype=dtype,
                         tile=(tile_size, tile_size), compression=compression,
                         photometric="minisblack", metadata=None)


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Write Parquet with explicit list-of-struct crop coordinates, including empty tables."""
    import pyarrow.parquet as pq
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    if "crop_origins" in table:
        typ = pa.list_(pa.struct([("x0", pa.int64()), ("y0", pa.int64())]))
        col = pa.array(table["crop_origins"].tolist(), type=typ)
        arrow = arrow.set_column(arrow.schema.get_field_index("crop_origins"), "crop_origins", col)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow, path, compression="zstd")


def publish_image(staging: Path, destination: Path, metadata: dict) -> None:
    """Publish actual closed files, then metadata; the study coordinator is sole writer.

    Removing the previous completion record first prevents interrupted updates
    from being mistaken for a complete image. Readers use metadata references.
    """
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metadata.json").unlink(missing_ok=True)
    for path in staging.iterdir():
        if path.is_file():
            os.replace(path, destination / path.name)
    write_json(destination / "metadata.json", metadata)


def publish_study(index: pd.DataFrame, output: Path, registry: Path) -> None:
    """Write the seven-field index and manifest referencing actual image metadata."""
    write_table(output / "images.parquet", index)
    images = [str(Path("images") / name / "metadata.json") for name in index.image_id
              if (output / "images" / name / "metadata.json").is_file()]
    write_json(output / "manifest.json", {
        "schema": "cellweft.preparation.v1", "study": str(index.study.iloc[0]) if len(index) else None,
        "registry": registry, "image_metadata": images,
        "normalization": {"percentile": 99.5, "method": "linear", "lower": 0},
        "crop": {"size": 224, "stride": 112, "edge": "cover"}, "protein_bank": None,
    })
