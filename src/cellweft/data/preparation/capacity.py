"""Local disk planning for one reviewed image and its active cache writes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import shutil
import threading


GIB = 1024**3
SSD_RESERVE = 64 * GIB


def occupied(path: Path) -> int:
    total = 0
    for asset in path.rglob("*"):
        try:
            if asset.is_file():
                total += asset.stat().st_size
        except FileNotFoundError:
            # A completed image may retire a temporary file during a scan.
            continue
    return total


def tiled_tiff_bytes(height: int, width: int, pages: int, bytes_per_pixel: int,
                     tile_size: int = 256) -> int:
    """Deflate payload bound plus a per-tile BigTIFF container allowance.

    The 32-byte allowance covers two 8-byte offset/count entries, alignment,
    and tile bookkeeping; 4096 bytes per page covers fixed IFD tags. This is
    an engineering planning bound for the current metadata=None writer.
    """
    tiles = math.ceil(height / tile_size) * math.ceil(width / tile_size)
    raw = tile_size * tile_size * bytes_per_pixel
    zlib_bound = raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + 13
    return pages * (tiles * (zlib_bound + 32) + 4096) + 4096 if pages else 0


def source_copy_bytes(source, destination: Path) -> int:
    """Count only growth of existing source copies; other products give no credit."""
    files = [(source.path, "path"), (source.valid_mask, "valid_mask"),
             (source.sample_region, "sample_region")]
    if source.mask and source.mask.path.is_file():
        files.append((source.mask.path, "source_mask"))
    total = 0
    for path, name in files:
        if path is None:
            continue
        target = destination / (name + path.suffix)
        current = target.stat().st_size if target.is_file() else 0
        total += max(0, path.stat().st_size - current)
    return total


def image_work_bytes(height: int, width: int, pages: int, *,
                     mask_unknown: bool, normalized: bool, reuse_shared: bool,
                     tile_size: int = 256) -> int:
    """Peak new writes for the actual sequential stages, with 48N scratch headroom.

    The 48N allowance is an engineering margin for variable table/scratch
    output, not a mathematical bound on arbitrary cell counts or mask files.
    """
    n = height * width
    shared = 0 if reuse_shared else tiled_tiff_bytes(height, width, pages, 4, tile_size)
    percentile = 0 if normalized or reuse_shared else 8 * n
    projection = tiled_tiff_bytes(height, width, 1, 4, tile_size) if pages else 0
    tissue = tiled_tiff_bytes(height, width, 1, 1, tile_size) if pages else 0
    phases = [percentile, shared, shared + projection + tissue]
    if not mask_unknown and pages:
        signal = tiled_tiff_bytes(height, width, 2, 4, tile_size)
        mask = tiled_tiff_bytes(height, width, 1, 4, tile_size)
        phases += [shared + signal + 4*n + mask,
                   shared + signal + 2*mask,
                   shared + signal + 2*mask + projection + tissue]
    return max(phases) + 48*n


@dataclass(frozen=True)
class CapacityDecision:
    required_bytes: int
    budget_bytes: int
    free_bytes: int
    existing_bytes: int
    reason: str


class CapacityGap(OSError):
    """This image cannot fit under the current policy even in isolation."""

    def __init__(self, decision: CapacityDecision):
        self.decision = decision
        super().__init__(decision.reason)


class CapacityReservations:
    """Atomically compare live files and remaining active growth reservations."""

    def __init__(self, cache: Path, budget_bytes: int):
        self.cache, self.budget_bytes = cache, budget_bytes
        self.lock = threading.Lock()
        self.active: dict[Path, tuple[int, int]] = {}

    def _remaining(self) -> int:
        return sum(max(0, planned - max(0, occupied(work) - baseline))
                   for work, (baseline, planned) in self.active.items())

    def acquire(self, work: Path, required_bytes: int) -> None:
        with self.lock:
            remaining = self._remaining()
            existing = occupied(self.cache)
            free = shutil.disk_usage(self.cache).free
            decision = CapacityDecision(required_bytes, self.budget_bytes, free, existing, "")
            if required_bytes > self.budget_bytes:
                raise CapacityGap(CapacityDecision(required_bytes, self.budget_bytes, free,
                                                   existing, "image_exceeds_cache_budget"))
            if occupied(work) + required_bytes > self.budget_bytes:
                raise CapacityGap(CapacityDecision(required_bytes, self.budget_bytes, free,
                                                   existing, "image_and_checkpoint_exceed_cache_budget"))
            if required_bytes > free + existing - SSD_RESERVE:
                raise CapacityGap(CapacityDecision(required_bytes, self.budget_bytes, free,
                                                   existing, "image_exceeds_current_ssd_capacity"))
            # Files already on disk, including unpublished checkpoints, count in
            # both occupied(cache) and disk free. Only future bytes are reserved.
            if existing + remaining + required_bytes > self.budget_bytes or free - remaining - required_bytes < SSD_RESERVE:
                # Other cache files or active tasks can retire and free capacity.
                raise OSError("Local staging capacity temporarily occupied")
            self.active[work] = (occupied(work), required_bytes)

    def release(self, work: Path) -> None:
        with self.lock:
            self.active.pop(work)
