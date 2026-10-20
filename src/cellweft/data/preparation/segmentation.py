"""Shared-signal composition and explicit local/file-based Cellpose-SAM execution."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader, PreparationSettings
from cellweft.data.io.readers import open_source, tiff_source, windows
from cellweft.data.io.writers import write_image, write_json


def content_identity(path: Path) -> dict:
    """Bind transported segmentation inputs/weights; do not hash raw datasets."""
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        size = handle.tell()
    return {"size": size, "sha256": digest}


def verify_task_inputs(task: dict, input_path: Path, weights: Path) -> dict:
    """Verify actual transported files and the two scalar float32 signal planes."""
    import tifffile
    actual = {"input": content_identity(input_path), "weights": content_identity(weights)}
    if actual != task["content_identity"]:
        raise ValueError("Segmentation input or weights content differs from the submitted task")
    signal = task["signal"]
    with tifffile.TiffFile(input_path) as tif:
        if (signal["axes"] != "CYX" or signal["dtype"] != "float32"
                or len(tif.pages) != 2 or signal["shape"][0] != 2
                or any(tuple(page.shape) != tuple(signal["shape"][1:])
                       or page.dtype != np.dtype("float32") for page in tif.pages)):
            raise ValueError("Segmentation signal shape, axes or dtype differs from the task")
    return actual


def build_segmentation_input(image: ImageReader, channels: pd.DataFrame, output: Path,
                             settings: PreparationSettings) -> dict:
    """Write two CYX pages [body, nucleus] using effective output pages directly.

    A varying nuclear composite is required. Zero body is recorded for the
    agreed once-only whole-image judgment of a newly generated mask.
    """
    groups = {}
    for group in ("body", "nuclear"):
        rows = channels[channels[group] & channels.tiff_page.notna()]
        groups[group] = rows.tiff_page.astype(int).tolist()
    if not groups["nuclear"]:
        return {"gap": "no effective nuclear channels"}
    bounds = {name: [np.inf, -np.inf] for name in groups}
    for block in image.iter_blocks(list(range(image.shape[0])), settings.block_size):
        for name, pages in groups.items():
            if pages:
                composite = block.values[pages].mean(axis=0, dtype=np.float32)
                bounds[name][0] = min(bounds[name][0], float(composite.min()))
                bounds[name][1] = max(bounds[name][1], float(composite.max()))
    if bounds["nuclear"][0] == bounds["nuclear"][1]:
        return {"gap": "nuclear composite has no spatial variation"}
    body_missing = not groups["body"] or bounds["body"][0] == bounds["body"][1]
    if body_missing:
        groups["body"] = []

    def composite(page, window):
        pages = groups["body" if page == 0 else "nuclear"]
        if not pages:
            return np.zeros((window[3]-window[1], window[2]-window[0]), np.float32)
        return image.read_window(pages, window).mean(axis=0, dtype=np.float32)

    write_image(output, (2, *image.shape[-2:]), np.dtype("float32"), composite,
                tile_size=settings.tile_size, compression=settings.compression)
    return {"gap": None, "groups": groups, "body_missing": body_missing,
            "shape": [2, *image.shape[-2:]], "axes": "CYX", "dtype": "float32"}


def stitch_instances(reader: ImageReader, predict: Callable[[np.ndarray], np.ndarray],
                     cache_dir: Path, output: Path, *, core_size: int = 1024,
                     halo: int = 128, match_iou: float = 0.5, tile_size: int = 256) -> None:
    """Predict bounded cores with halos and match labels on previously owned pixels.

    Overlap IoU and greedy one-to-one matching are provisional engineering rules.
    Only core pixels are written. Biological seam quality requires dataset validation.
    """
    h, w = reader.shape[-2:]
    with tempfile.TemporaryDirectory(prefix="segmentation_", dir=cache_dir) as directory:
        labels = np.memmap(Path(directory)/"labels.bin", mode="w+", dtype="uint32", shape=(h, w))
        labels[:] = 0
        next_label = 1
        try:
            for x0, y0, x1, y1 in windows((h, w), core_size):
                a, b, c, d = max(0,x0-halo), max(0,y0-halo), min(w,x1+halo), min(h,y1+halo)
                planes = reader.read_window([0, 1], (a,b,c,d))
                input_hwc = np.stack([planes[0], planes[1], np.zeros_like(planes[0])], axis=-1)
                local = np.asarray(predict(input_hwc))
                if local.shape != (d-b,c-a) or local.dtype.kind not in "iu" or np.any(local < 0):
                    raise ValueError("Segmenter must return nonnegative integer YX instances")
                ids = np.unique(local); ids = ids[ids > 0]
                old = labels[b:d,a:c]
                yy, xx = np.ogrid[b:d,a:c]
                owned = np.broadcast_to((yy < y0) | ((yy < y1) & (xx < x0)), local.shape)
                pairs = owned & (old > 0) & (local > 0)
                matches = []
                if pairs.any():
                    combinations, counts = np.unique(np.stack([local[pairs],old[pairs]],axis=1),
                                                       axis=0,return_counts=True)
                    lids, lcounts = np.unique(local[owned],return_counts=True)
                    gids, gcounts = np.unique(old[owned],return_counts=True)
                    lc, gc = dict(zip(lids,lcounts)), dict(zip(gids,gcounts))
                    for (lid,gid), intersection in zip(combinations,counts):
                        score = intersection / (lc[lid]+gc[gid]-intersection)
                        if score >= match_iou:
                            matches.append((float(score),int(lid),int(gid)))
                mapping, used = {}, set()
                for _, lid, gid in sorted(matches,reverse=True):
                    if lid not in mapping and gid not in used:
                        mapping[lid] = gid; used.add(gid)
                assigned = np.zeros(len(ids),np.uint32)
                for i, lid in enumerate(ids):
                    if int(lid) not in mapping:
                        if next_label > np.iinfo(np.uint32).max:
                            raise ValueError("Instance count exceeds uint32")
                        mapping[int(lid)] = next_label; next_label += 1
                    assigned[i] = mapping[int(lid)]
                center = local[y0-b:y1-b,x0-a:x1-a]
                target = labels[y0:y1,x0:x1]
                foreground = center > 0
                target[foreground] = assigned[np.searchsorted(ids,center[foreground])]
            write_image(output,(1,h,w),np.dtype("uint32"),
                        lambda _, win: labels[win[1]:win[3],win[0]:win[2]],tile_size=tile_size)
        finally:
            labels._mmap.close()


class CellposeExecutor:
    """One lazy local cpsam_v2 model per worker; task parameters determine inference."""

    def __init__(self, device: str = "cuda:0", core_size: int = 1024, halo: int = 128,
                 match_iou: float = 0.5):
        self.device, self.core_size, self.halo, self.match_iou = device, core_size, halo, match_iou
        self._model = None
        self._weights = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._model = None

    def execute(self, task: dict, *, input_path: Path | None = None,
                result_path: Path | None = None) -> Path:
        """Return a closed result JSON; remote workers may override transport paths."""
        weights = Path(task["weights"]).resolve(strict=True)
        input_path = input_path or Path(task["input_path"])
        actual = verify_task_inputs(task, input_path, weights)
        import importlib.metadata
        import torch
        from cellpose import models
        stitching = {"core_size":self.core_size,"halo":self.halo,"match_iou":self.match_iou}
        if stitching != task["stitching"]:
            raise ValueError("Worker stitching parameters must match the submitted task")
        if importlib.metadata.version("cellpose") != "4.2.1.1":
            raise ValueError("This adapter requires cellpose==4.2.1.1")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        model_identity = (weights, actual["weights"]["sha256"])
        if self._model is None or self._weights != model_identity:
            self._model = models.CellposeModel(gpu=self.device.startswith("cuda"),
                                               device=torch.device(self.device), pretrained_model=str(weights))
            if Path(self._model.pretrained_model).resolve() != weights:
                raise RuntimeError("Cellpose did not use the requested local checkpoint")
            self._weights = model_identity
        parameters = dict(task["parameters"])
        result_path = result_path or Path(task["result_path"])
        result_path.parent.mkdir(parents=True,exist_ok=True)
        mask_path = result_path.parent / "mask.tiff"
        partial = result_path.parent / "mask.partial.tiff"
        with open_source(tiff_source(input_path,2)) as reader:
            stitch_instances(reader,
                lambda hwc: self._model.eval(hwc,channel_axis=-1,normalize=False,**parameters)[0],
                result_path.parent,partial,core_size=self.core_size,halo=self.halo,
                match_iou=self.match_iou)
        partial.replace(mask_path)
        write_json(result_path, {
            "task_id":task["task_id"], "request":task["request"], "mask":"mask.tiff",
            "parameters":parameters, "model":"cpsam_v2", "normalize":False,
            "stitching":stitching,
            "content_identity":actual,
            "whole_cell_confirmed":not task["signal"]["body_missing"],
        })
        return result_path
