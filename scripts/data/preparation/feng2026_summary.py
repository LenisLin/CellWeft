"""Verify the released Feng cohort and generate its standard processing summary."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile

from cellweft.data.io.writers import write_json
from cellweft.data.preparation.inputs import file_revision


STUDY = "Feng2026_IMC_BladderCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
ROOT = NAS / "processed" / STUDY
CACHE = NAS / "cache" / STUDY
TABLES = ("images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv", "review_results.csv")
PRODUCTS = {"image", "channels", "mask", "cells", "expression", "neighbors"}


def check_image(row: dict) -> tuple[int, str]:
    directory = ROOT / "images" / row["image_id"]
    data = json.loads((directory / "metadata.json").read_bytes())
    if data.get("publication_incomplete") or not PRODUCTS <= set(data["completed"]) or data["mask_origin"] != "source":
        raise ValueError("Incomplete image or unexpected mask origin")
    for key in PRODUCTS:
        product = Path(data["products"][key])
        if product.parent != directory or not product.is_file():
            raise ValueError("Missing formal product or nonportable path")
    counts = [pq.ParquetFile(directory / name).metadata.num_rows for name in
              ("cells.parquet", "expression.parquet", "neighbors.parquet")]
    if counts != [int(row["instances"])] * 3:
        raise ValueError("Source instances, expression and neighbors do not share a cell count")
    return counts[0], data["sampling"]["status"]


def check_measurements(row: dict, height: int, width: int, top: int, left: int) -> None:
    directory = ROOT / "images" / row["image_id"]
    channels = pd.read_parquet(directory / "channels.parquet").dropna(subset=["expression_column"])
    cells = pd.read_parquet(directory / "cells.parquet")
    with tifffile.TiffFile(directory / "input.tiff") as original, tifffile.TiffFile(directory / "image.tiff") as image:
        channel = channels.iloc[len(channels) // 2]
        raw_plane = original.pages[int(channel.output_channel_index)].asarray()
        scale = np.quantile(raw_plane[top:top + height, left:left + width].astype(np.float64), 0.995)
        np.testing.assert_allclose(scale, channel.q995, rtol=1e-7)
        expected = np.clip(raw_plane.astype(np.float64) / scale, 0, 1)
        plane = image.pages[int(channel.tiff_page)].asarray()
        np.testing.assert_allclose(plane, expected, rtol=1e-6, atol=1e-7)
    mask = tifffile.imread(directory / "mask.tiff")
    labels = cells.mask_label.to_numpy(dtype=int)
    counts = np.bincount(mask.ravel())[labels]
    np.testing.assert_array_equal(counts, cells.area)
    expression = pd.read_parquet(directory / "expression.parquet", columns=["cell_index", channel.expression_column])
    np.testing.assert_array_equal(expression.cell_index, cells.cell_index)
    means = np.bincount(mask.ravel(), weights=plane.ravel())[labels] / counts
    np.testing.assert_allclose(expression[channel.expression_column], means, rtol=1e-6, atol=1e-7)
    cell = cells.iloc[len(cells) // 2]
    distances = (cells.centroid_x - cell.centroid_x) ** 2 + (cells.centroid_y - cell.centroid_y) ** 2
    neighbors = pd.read_parquet(directory / "neighbors.parquet").set_index("cell_index").loc[cell.cell_index]
    chosen = neighbors[neighbors >= 0].to_numpy(dtype=int)
    np.testing.assert_allclose(distances.iloc[chosen], np.sort(distances.drop(cell.name))[:len(chosen)])
    if not len(cell.crop_origins) or any(not np.any(mask[origin["y0"]:origin["y0"] + 224,
                                                       origin["x0"]:origin["x0"] + 224] == cell.mask_label)
                                         for origin in cell.crop_origins):
        raise ValueError("Stored training windows must intersect their source instance")


images = pd.read_parquet(ROOT / "images.parquet").set_index("image_id")
review = pd.read_csv(CACHE / "mask_review.csv").set_index("image_id", drop=False)
acceptance = json.loads((CACHE.with_name(STUDY + "_review") / "progress.json").read_bytes())
plan_path = ROOT / "training_split.json"
plan = json.loads(plan_path.read_bytes()) if plan_path.is_file() else None
expected_ids = set(plan["retained_image_ids"]) if plan else set(review.index)
if set(images.index) != expected_ids or not expected_ids <= set(review.index) or acceptance["signed"] != len(images):
    raise ValueError("Study, mask and independent D01 scopes must agree")
review = review.loc[images.index]
records = review.loc[images.index].to_dict("records")
with ThreadPoolExecutor(max_workers=4) as readers:
    results = list(readers.map(check_image, records))
representatives = review[review.sampled]
for mask_row in representatives.to_dict("records"):
    source = images.loc[mask_row["image_id"]]
    check_measurements(mask_row, int(mask_row["height"]), int(mask_row["width"]),
                       int(source.pad_top_px), int(source.pad_left_px))
sampling = dict(Counter(status for _, status in results))
summary = dict(schema="cellweft.processing_summary.v1", study=STUDY, owner="Feng", registration_year=2026,
               diagnosis="Bladder cancer", source_alias="IMC_20240725_63", private_data=True,
               scope=dict(retained_images=len(images), technical_complete_images=len(results),
                          technical_incomplete_image_ids=[], image_keys="images.parquet:(study,image_id)"),
               processing=dict(normalization="Per-ROI per-channel measured-pixel P99.5", mask_origins=["source"],
                               cells=sum(count for count, _ in results), crop=dict(size=224, stride=112)),
               acceptance=dict(technical_status="complete", d01=acceptance["signed"], input_pixel_samples=acceptance["sampled"],
                               representative_images=representatives.image_id.tolist(), sampling=sampling,
                               numerical_review="P99.5, normalized plane, per-instance area and mean for one marker per ROI, one centroid neighbor list and cell/window intersections per ROI"),
               clinical_metadata=dict(patient_linked_images=int(images.source_patient_key.notna().sum()),
                                      patients=int(images.source_patient_key.nunique()), sample_rows=len(pd.read_csv(ROOT / "sample_metadata.csv"))),
               dependencies=dict(raw=f"raw/private/{STUDY}/manifest.json", qa=str(CACHE.relative_to(NAS)),
                                 tissue_estimate_review=str((CACHE / "tissue_estimate_samples.json").relative_to(NAS))),
               embedding_status="published", embedded_markers=int(pd.read_csv(ROOT / "panel_mapping.csv").embedding_row.nunique()),
               image_metadata_count=len(results), final_location=str(ROOT),
               nas_five_table_revisions={name: file_revision(ROOT / name) for name in TABLES})
if plan:
    summary["scope"].update(original_images=520, excluded_images=len(plan["excluded_image_ids"]),
                            exclusion_reason=plan["exclusion_reason"])
    summary["training_split"] = dict(path=str(plan_path.relative_to(NAS)), seed=plan["seed"],
                                    counts=plan["counts"], sampling_status=plan["sampling_status"])
destination = ROOT / "processing_summary.json"
write_json(destination, summary)
if json.loads(destination.read_bytes()) != summary:
    raise ValueError("Processing summary readback differs")
print(f"Verified {len(results)} ROIs and {summary['processing']['cells']} cells, numerical checks on {len(representatives)} ROIs, sampling {sampling}")
