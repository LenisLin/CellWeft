"""Review indexed Feng IMC source masks, then stage reviewed source-frame bindings.

Inputs: intake index, original MCD/TXT, and steinbock images, masks and cell tables.
Outputs: mask review CSV, source assets and bindings under the selected data root.
Run with .venv-data. All per-image evidence stays in the private data root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from readimc import MCDFile, TXTFile
from skimage.measure import label


STUDY = "Feng2026_IMC_BladderCancer"
CACHE = Path("cache") / STUDY
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE = Path("/mnt/public/lyx/horizontalProject/IMC_20240725_63")
STEINBOCK = SOURCE / "steinbock"
SEED = 20261124


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def mcd_geometry(rows: list[dict], channel_names: list[str]) -> None:
    """Validate acquisition dimensions, channel order and data extent from each MCD."""
    containers = defaultdict(list)
    for row in rows:
        if row["format"] == "mcd":
            containers[row["raw_path"]].append(row)
    for path, members in containers.items():
        with MCDFile(path) as mcd:
            by_id = {acq.id: acq for slide in mcd.slides for acq in slide.acquisitions}
            for row in members:
                acq = by_id[int(row["acquisition_id"])]
                meta = acq.metadata
                width, height, start, stop, size = (
                    int(meta[key]) for key in ("MaxX", "MaxY", "DataStartOffset", "DataEndOffset", "ValueBytes")
                )
                stride = acq.num_channels + 3
                pixels, remainder = divmod(stop - start, stride * size)
                row.update(raw_pixels=pixels, start=start, stride=stride)
                row["raw_geometry_ok"] = (
                    (height, width) == row["shape"] and size == 4 and not remainder
                    and 0 < start < stop <= path.stat().st_size
                    and 0 < pixels <= width * height and acq.channel_names == channel_names
                )


def native_mcd(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the coordinates stored alongside each measured MCD pixel."""
    data = np.memmap(row["raw_path"], mode="r", dtype="<f4", offset=row["start"],
                     shape=(row["raw_pixels"], row["stride"]))
    xx = data[:, 0].astype(np.int32)
    yy = data[:, 1].astype(np.int32)
    if not (np.array_equal(xx, data[:, 0]) and np.array_equal(yy, data[:, 1])):
        raise ValueError("Noninteger MCD coordinates")
    return yy, xx, data[:, 3:]


def footprint(row: dict, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    height, width = row["shape"]
    if (xx.min() < 0 or yy.min() < 0 or xx.max() >= width or yy.max() >= height):
        raise ValueError("Raw coordinates exceed the source frame")
    measured = np.zeros((height, width), dtype=bool)
    measured[yy, xx] = True
    if np.count_nonzero(measured) != len(xx):
        raise ValueError("Raw coordinates are duplicated")
    return measured


def review_mask(row: dict) -> None:
    """Scan every label and compare the TIFF frame with the matching source images."""
    height, width = row["shape"]
    with tifffile.TiffFile(row["mask_path"]) as tif:
        page = tif.pages[0]
        orientation = page.tags.get("Orientation")
        scalar = len(tif.pages) == 1 and page.axes == "YX"
        row.update(mask_dtype=page.dtype.name,
                   mask_geometry_ok=scalar and page.shape == (height, width)
                   and (orientation is None or orientation.value == 1))
        mask = page.asarray()
    for folder in ("img", "img_pre_denoise"):
        with tifffile.TiffFile(STEINBOCK / folder / row["source_image"]) as tif:
            series = tif.series[0]
            orientation = tif.pages[0].tags.get("Orientation")
            row[f"{folder}_geometry_ok"] = (
                series.shape == (35, height, width) and series.axes in {"CYX", "QYX"}
                and len(tif.pages) == 35 and (orientation is None or orientation.value == 1)
            )
    row["labels_ok"] = (
        mask.dtype.kind in "iu" and mask.min() >= 0 and mask.max() <= np.iinfo(np.uint32).max
    )
    if not row["labels_ok"] or not row["mask_geometry_ok"]:
        return
    ids, counts = np.unique(mask, return_counts=True)
    instances = ids[ids > 0]
    row.update(instances=len(instances), background_px=int(counts[0]) if ids[0] == 0 else 0,
               label_min=int(instances[0]) if len(instances) else None,
               label_max=int(ids[-1]), labeled_px=int(counts[ids > 0].sum()),
               disconnected_extra_components=int(label(mask, connectivity=2).max()) - len(instances),
               outside_measured_px=0)
    props = pd.read_csv((STEINBOCK / "regionprops" / row["source_image"]).with_suffix(".csv"),
                        usecols=["Object", "area"]).sort_values("Object")
    row["regionprops_ok"] = np.array_equal(props.Object, instances) and np.array_equal(props.area, counts[ids > 0])
    if row["format"] == "mcd" and row["raw_geometry_ok"] and row["raw_pixels"] != height * width:
        yy, xx, _ = native_mcd(row)
        row["outside_measured_px"] = int(np.count_nonzero(mask[~footprint(row, yy, xx)]))


def transform_candidates(image: np.ndarray) -> dict[str, np.ndarray]:
    """Enumerate spatial flips and transpose before choosing the source orientation."""
    regular = {"identity": image, "flip_x": image[..., ::-1],
               "flip_y": image[..., ::-1, :], "flip_xy": image[..., ::-1, ::-1]}
    transposed = image.swapaxes(-1, -2)
    return regular | {"transpose": transposed, "rotate_90": transposed[..., ::-1],
                      "rotate_270": transposed[..., ::-1, :], "antitranspose": transposed[..., ::-1, ::-1]}


def sample_overlay(row: dict, panel: list[dict], keep: list[int]) -> None:
    """Evaluate raw orientation, mask-to-image placement and paired whole-cell means locally."""
    height, width = row["shape"]
    if row["format"] == "mcd":
        yy, xx, intensities = native_mcd(row)
    else:
        with TXTFile(row["raw_path"]) as txt:
            channel_order_ok = txt.channel_names == [entry["channel"] for entry in panel]
        data = pd.read_csv(row["raw_path"], sep="\t", dtype=np.float32)
        yy = data.Y.to_numpy().astype(np.int32)
        xx = data.X.to_numpy().astype(np.int32)
        row["raw_geometry_ok"] = (
            list(data.columns[3:6]) == ["X", "Y", "Z"] and np.array_equal(data.Z, yy * width + xx)
            and channel_order_ok and np.array_equal(yy, data.Y) and np.array_equal(xx, data.X)
            and (int(yy.max()) + 1, int(xx.max()) + 1) == (height, width)
        )
        intensities = data.iloc[:, 6:].to_numpy()
        row["raw_pixels"] = len(xx)
    measured = footprint(row, yy, xx)
    raw = np.zeros((len(keep), height, width), dtype=np.float32)
    for index, channel in enumerate(keep):
        raw[index, yy, xx] = intensities[:, channel]
    before = tifffile.memmap(STEINBOCK / "img_pre_denoise" / row["source_image"], mode="r")
    denoised = tifffile.memmap(STEINBOCK / "img" / row["source_image"], mode="r")
    mask = tifffile.imread(row["mask_path"])
    dna = [index for index, channel in enumerate(keep) if panel[channel]["channel"] in {"Ir191", "Ir193"}]
    marker_names = [panel[channel]["name"] for channel in keep]

    # Test whole-frame orientation on a fixed grid and compare every kept channel in two blocks.
    dna_raw = np.asarray(raw[dna])
    dna_before = np.asarray(before[dna, ::5, ::5])
    scores = {name: float(np.mean(candidate[:, ::5, ::5] == dna_before))
              for name, candidate in transform_candidates(dna_raw).items() if candidate.shape == dna_raw.shape}
    denoised_grid = np.log1p(np.maximum(denoised[dna, ::5, ::5], 0)).ravel()
    denoised_scores = {
        name: float(np.corrcoef(np.log1p(np.maximum(candidate[:, ::5, ::5], 0)).ravel(), denoised_grid)[0, 1])
        for name, candidate in transform_candidates(dna_raw).items() if candidate.shape == dna_raw.shape
    }
    size = min(64, height, width)
    center_y, center_x = np.unravel_index(np.argmax(raw[dna].max(axis=0)), (height, width))
    y0 = int(np.clip(center_y - size // 2, 0, height - size))
    x0 = int(np.clip(center_x - size // 2, 0, width - size))
    interior = (slice(None), slice(y0, y0 + size), slice(x0, x0 + size))
    boundary = (slice(None), slice(0, size), slice(width - size, width))
    row.update(orientation=max(scores, key=scores.get), orientation_identity_match=scores["identity"],
               orientation_next_match=max(score for name, score in scores.items() if name != "identity"),
               denoised_orientation=max(denoised_scores, key=denoised_scores.get),
               denoised_identity_correlation=denoised_scores["identity"],
               denoised_next_correlation=max(score for name, score in denoised_scores.items() if name != "identity"),
               raw_internal_equal=float(np.mean(raw[interior] == before[interior])),
               raw_boundary_equal=float(np.mean(raw[boundary] == before[boundary])),
               internal_x0=x0, internal_y0=y0, boundary_x0=width-size, block_size=size,
               outside_measured_px=int(np.count_nonzero(mask[~measured])))

    # Independently reconstruct the stored object-by-marker means from the overlaid pixels.
    objects = pd.read_csv((STEINBOCK / "intensities" / row["source_image"]).with_suffix(".csv"))
    flat_mask = mask.ravel()
    areas = np.bincount(flat_mask)
    positive = np.flatnonzero(areas[1:]) + 1
    objects = objects.sort_values("Object")
    matched = np.array_equal(objects.Object, positive)
    max_relative_residual = 0.0
    if matched and len(positive):
        for index, name in enumerate(marker_names):
            actual = np.bincount(flat_mask, weights=denoised[index].ravel())[positive] / areas[positive]
            stored = objects[name].to_numpy()
            max_relative_residual = max(max_relative_residual, float(np.max(np.abs(actual - stored) / (1 + np.abs(stored)))))
    nuclear = np.log1p(np.maximum(denoised[dna].max(axis=0), 0))
    positive_mask = (mask > 0) & measured
    background = (mask == 0) & measured
    inside = float(nuclear[positive_mask].mean()) if positive_mask.any() else 0.0
    outside = float(nuclear[background].mean()) if background.any() else 0.0
    row.update(overlay_log_dna_inside=inside, overlay_log_dna_outside=outside,
               overlay_mean_max_relative_error=max_relative_residual if matched else None,
               overlay_ok=bool(matched and len(positive) and max_relative_residual <= 1e-4 and inside > outside))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=NAS)
parser.add_argument("--workers", type=int, default=6)
args = parser.parse_args()
os.umask(0o077)

# Preserve intake order and use only synthetic identifiers for staged assets.
intake = read_rows(NAS / "cache/IMC_20240725_63_intake/indexed_sources.csv")
index = read_rows(STEINBOCK / "images.csv")
panel = read_rows(STEINBOCK / "panel.csv")
keep = [idx for idx, entry in enumerate(panel) if entry["keep"] == "1"]
assert len(intake) == len(index) == 520 and len(panel) == 39 and len(keep) == 35
assert len({row["source_image"] for row in intake}) == 520
group_files = []
for path in SOURCE.glob("*/*.csv"):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        if next(csv.reader(stream))[:2] == ["ID", "Group"]:
            group_files.append(path)
assert len(group_files) == 1
group_refs = {row["ID"].strip() for row in read_rows(group_files[0])[:63]}
assert len(group_refs) == 63
rows = []
for ordinal, (entry, exported) in enumerate(zip(intake, index, strict=True), 1):
    width, height = int(entry["width"]), int(entry["height"])
    assert (entry["source_image"], width, height, int(entry["indexed_channels"])) == (
        exported["image"], int(exported["width_px"]), int(exported["height_px"]), int(exported["num_channels"])
    )
    if entry["acquisition_id"]:
        assert int(entry["acquisition_id"]) == float(exported["acquisition_id"])
    parts = Path(entry["source_image"]).stem.split("_")
    if len(parts) == 2:
        group_key = parts[0]
    elif len(parts) == 4:
        group_key = "_".join(parts[1:3])
    else:
        group_key = parts[1] if len(parts) == 3 or len(parts) > 5 else ""
    raw_path = SOURCE / entry["raw_relative_to_study"]
    rows.append(dict(image_id=f"{STUDY}__roi{ordinal:04d}", width=width, height=height,
                     shape=(height, width), format=raw_path.suffix.lower()[1:], raw_path=raw_path,
                     acquisition_id=entry["acquisition_id"], source_image=entry["source_image"],
                     mask_path=STEINBOCK / "masks" / entry["source_image"],
                     group_join="matched" if group_key in group_refs else "unmatched",
                     filename_parts=len(parts), area=width * height))
assert Counter(row["format"] for row in rows) == {"mcd": 518, "txt": 2}
mcd_geometry(rows, [entry["channel"] for entry in panel])
print("MCD geometry checked", sum(row.get("raw_geometry_ok", False) for row in rows), flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    for ordinal, _ in enumerate(pool.map(review_mask, rows), 1):
        if ordinal % 100 == 0 or ordinal == len(rows):
            print("Masks read", ordinal, flush=True)

# Cover both TXT files, geometry and grouping anomalies, and the three baseline draws.
areas = np.array([row["area"] for row in rows])
largest = int(np.argmax(areas))
typical = sorted(range(len(rows)), key=lambda ordinal: (abs(areas[ordinal] - np.median(areas)), ordinal))
selected = {largest, next(ordinal for ordinal in typical if ordinal != largest)}
selected.add(int(np.random.default_rng(SEED).choice(sorted(set(range(len(rows))) - selected))))
strata = defaultdict(list)
for ordinal, row in enumerate(rows):
    group = (row["format"], row["shape"], row["group_join"], row["filename_parts"],
             row.get("raw_pixels", row["area"]) != row["area"])
    strata[group].append(ordinal)
    row["stratum"] = str(len(strata)) if len(strata[group]) == 1 else rows[strata[group][0]]["stratum"]
    if row["format"] == "txt" or row.get("outside_measured_px") or not row.get("regionprops_ok"):
        selected.add(ordinal)
worst_components = max(range(len(rows)), key=lambda ordinal: rows[ordinal].get("disconnected_extra_components", 0))
if rows[worst_components].get("disconnected_extra_components"):
    selected.add(worst_components)
for members in strata.values():
    if not selected.intersection(members):
        selected.add(members[0])
print("Overlay sample", len(selected), "strata", len(strata), flush=True)
with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
    samples = [pool.submit(sample_overlay, rows[ordinal], panel, keep) for ordinal in sorted(selected)]
    for count, sample in enumerate(samples, 1):
        sample.result()
        print("Overlays checked", count, flush=True)

# Publish only reviewed source-frame masks and leave per-image review status explicit.
root = args.data_root / CACHE
root.mkdir(parents=True, exist_ok=True)
bindings = {}
checks = ("raw_geometry_ok", "mask_geometry_ok", "img_geometry_ok", "img_pre_denoise_geometry_ok",
          "labels_ok", "background_px", "instances", "regionprops_ok")
for members in strata.values():
    representatives = [rows[ordinal] for ordinal in members if ordinal in selected]
    group_ok = all(row["orientation"] == row["denoised_orientation"] == "identity" and row["overlay_ok"]
                   and row["raw_geometry_ok"] for row in representatives)
    for ordinal in members:
        row = rows[ordinal]
        failures = [key for key in checks if not row.get(key)]
        row["sampled"] = ordinal in selected
        row["sample_evidence_ids"] = ";".join(rep["image_id"] for rep in representatives)
        row["mask_type"] = "whole_cell"
        row["mask_type_evidence"] = "source nuclear and cytoplasmic segmentation panel; paired object intensity overlay"
        row["overview_judgment"] = (
            row.get("orientation") == row.get("denoised_orientation") == "identity" and row.get("overlay_ok", False)
            if row["sampled"] else None
        )
        if row["sampled"] and row["overview_judgment"] is False:
            failures.append("source_orientation_or_overlay")
        if failures:
            binding = dict(status="unusable", reason=", ".join(failures), evidence=str(CACHE / "mask_review.csv"))
        elif not group_ok:
            binding = dict(status="unknown")
        else:
            path = Path("raw/private") / STUDY / "source" / row["mask_path"].relative_to(SOURCE)
            if not np.array_equal(tifffile.imread(args.data_root / path), tifffile.imread(row["mask_path"])):
                raise ValueError("Received mask pixels differ from source")
            binding = dict(status="bound", path=str(path), frame="source", mask_type="whole_cell",
                           page=0, flip_x=False, flip_y=False, quarter_turns=0,
                           overview_judgment=row["overview_judgment"])
        row.update(status=binding["status"], reason=binding.get("reason", ""), binding_path=binding.get("path", ""))
        bindings[row["image_id"]] = binding

private = {"raw_path", "mask_path", "source_image", "acquisition_id", "shape", "start", "stride", "area"}
fields = list(dict.fromkeys(key for row in rows for key in row if key not in private))
with (root / "mask_review.csv.part").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
(root / "mask_review.csv.part").replace(root / "mask_review.csv")
ordered_bindings = {row["image_id"]: bindings[row["image_id"]] for row in rows}
(root / "mask_bindings.json.part").write_text(json.dumps(ordered_bindings, indent=2) + "\n", encoding="utf-8")
(root / "mask_bindings.json.part").replace(root / "mask_bindings.json")
print("Binding statuses", dict(Counter(row["status"] for row in rows)), flush=True)
