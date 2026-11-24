"""Typed boundaries; windows use local image pixels and half-open XYXY bounds."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
import pandas as pd

Window = tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskSource:
    """Supplied scalar TIFF mask, with source-provided type and coarse judgment."""

    path: Path
    mask_type: str
    page: int = 0
    window: Window | None = None
    alignment_issue: str | None = None
    overview_judgment: bool | None = None
    overview_required: bool = False
    format: str = "tiff"
    npz_key: str | None = None
    selection: tuple[int | None, ...] = ()
    padding_ltrb: tuple[int, int, int, int] = (0, 0, 0, 0)
    flip_x: bool = False
    flip_y: bool = False
    quarter_turns: int = 0


@dataclass(frozen=True)
class MaskRejection:
    """Documented reason that a paired author mask cannot be used."""

    reason: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Unusable mask requires a nonempty reason")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("Unusable mask requires traceable source evidence")


@dataclass(frozen=True)
class SourceSpec:
    """One independent image; source channels are indexed in their published order.

    TIFF pages/paths, MCD acquisition, Zarr CYX/CZYX, and IMS channel indices
    are explicit. Window is in the source frame; readers expose local (0, 0).
    Directory-backed sources require an explicit source revision for reuse.
    """

    image_id: str
    panel_id: str
    kind: str
    path: Path
    channels: tuple[int, ...]
    paths: tuple[Path, ...] = ()
    window: Window | None = None
    pixel_size_um: tuple[float, float] | None = None
    mask: MaskSource | None = None
    acquisition_id: int | None = None
    z_index: int | None = None
    valid_shape: tuple[int, int] | None = None
    valid_mask: Path | None = None
    source_revision: str | None = None
    intensity_description: str = "published intensities"
    normalized_q995: tuple[float | None, ...] | None = None
    valid_bounds: Window | None = None
    mask_status: str = "absent"
    sample_region: Path | None = None
    mask_rejection: MaskRejection | None = None


@dataclass(frozen=True)
class ImageRecord:
    """Exactly the seven study-index fields; input and output paths are directories."""

    study: str
    image_id: str
    channel_count: int
    pixel_size_um: tuple[float, float] | None
    image_path: Path
    marker_list_path: Path
    output_path: Path


@dataclass(frozen=True)
class ImageBlock:
    """CYX block and YX validity; measured zeros remain valid."""

    window: Window
    values: np.ndarray
    valid: np.ndarray


class ImageReader(Protocol):
    """Open reader owned by its context manager; reads return independent arrays."""

    shape: tuple[int, int, int]
    dtype: np.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray: ...
    def valid_window(self, window: Window) -> np.ndarray: ...
    def iter_blocks(self, channels: list[int], block_size: int) -> Iterator[ImageBlock]: ...


@dataclass(frozen=True)
class PreparationSettings:
    """Execution settings; accepted intensity, crop and neighbor rules stay fixed."""

    cache_dir: Path
    tile_size: int = 256
    block_size: int = 512
    compression: str = "deflate"
    cpu_workers: int = 1
    weights: Path | None = None
    precision: str = "bf16"
    sampling_reviewed: bool = False
    # An unknown author-mask binding normally keeps segmentation pending.  A
    # scoped runtime may explicitly permit cpsam_v2 while preserving the
    # unknown provenance in metadata; this never treats the author mask as
    # absent or validated.
    allow_unknown_mask_segmentation: bool = False
    tissue_bins: int = 256
    mask_decode_budget: int = 1_073_741_824
    segmentation_parameters: dict = field(default_factory=lambda: {
        "diameter": None, "bsize": 256, "batch_size": 4,
        "tile_overlap": 0.1, "flow_threshold": 0.4,
        "cellprob_threshold": 0.0, "min_size": 15,
        "max_size_fraction": 0.4, "resample": True,
        "augment": False, "niter": None, "do_3D": False,
    })


@dataclass
class SharedImageResult:
    """Persistent shared TIFF and the table of all source channels."""

    path: Path | None
    channels: pd.DataFrame
    shape: tuple[int, int]


@dataclass
class ImageResult:
    """Published image metadata, including only actual products and known gaps."""

    metadata_path: Path
    products: dict[str, str]
    gaps: list[str]
    reused: bool = False


@dataclass
class CropSample:
    """Paired input-local pixels and measured validity; labels keep their mask identity."""

    pixels: np.ndarray
    labels: np.ndarray | None
    spatial_valid: np.ndarray
    origin_xy: tuple[int, int]
    mask_identity: str | None = None


@dataclass(frozen=True)
class ViewGeometry:
    """Crop-local half-open window, square output and one shared discrete transform."""

    window: Window
    size: int
    flip_x: bool = False
    flip_y: bool = False
    quarter_turns: int = 0


@dataclass
class ViewSample:
    """Transformed arrays, input pixel-center affine and original visible cell indices."""

    sample: CropSample
    input_to_view: np.ndarray
    view_to_input: np.ndarray
    cell_indices: np.ndarray
