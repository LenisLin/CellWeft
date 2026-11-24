"""Resolve shared preparation settings once; dataset code owns source discovery."""
from __future__ import annotations

from pathlib import Path

import yaml

from cellweft.data.contracts import PreparationSettings


def load_settings(defaults: Path, cache_dir: Path, *, weights: Path | None = None,
                  batch_size: int | None = None, cpu_workers: int | None = None,
                  precision: str = "bf16", sampling_reviewed: bool = False,
                  allow_unknown_mask_segmentation: bool = False) -> PreparationSettings:
    """Load fixed scientific settings and the declared runtime overrides."""
    if precision not in {"bf16", "fp32"}:
        raise ValueError("Precision must be bf16 or fp32")
    values = yaml.safe_load(defaults.read_text())
    if values["normalization"] != {"percentile":99.5,"method":"linear","lower":0}:
        raise ValueError("Preparation requires the accepted P99.5 definition")
    if values["crop"] != {"size":224,"stride":112} or values["neighbors"] != {"k":20}:
        raise ValueError("Preparation requires native224/112 and top20 neighbors")
    parameters = values["segmentation"]
    if batch_size is not None:
        parameters["batch_size"] = batch_size
    return PreparationSettings(cache_dir=cache_dir,weights=weights,precision=precision,
        sampling_reviewed=sampling_reviewed,
        allow_unknown_mask_segmentation=allow_unknown_mask_segmentation,
        tile_size=values["io"]["tile_size"],block_size=values["io"]["block_size"],
        compression=values["io"]["compression"],
        cpu_workers=cpu_workers if cpu_workers is not None else values["runtime"]["cpu_workers"],
        segmentation_parameters=parameters)
