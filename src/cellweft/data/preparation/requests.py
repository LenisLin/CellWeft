"""Versioned preparation request identities without compute pipeline imports."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cellweft.data.contracts import PreparationSettings, SourceSpec
from cellweft.data.io.writers import json_value


def file_identity(path: Path) -> dict:
    """Identify an input file by its resolved path, size and modification timestamp."""
    stat = path.stat()
    return {"path":str(path.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def source_identity(source: SourceSpec) -> dict:
    """Capture source selection separately from mask and panel-dependent processing."""
    value = asdict(source)
    value.pop("mask")
    value.pop("mask_status")
    value.pop("mask_rejection")
    files = [source.path, *source.paths]
    if source.valid_mask:
        files.append(source.valid_mask)
    if source.sample_region:
        files.append(source.sample_region)
    if source.kind == "zarr" and not source.source_revision:
        raise ValueError("Directory-backed source requires its explicit source_revision")
    value["file_identity"] = [file_identity(p) for p in dict.fromkeys(files)]
    return json_value(value)


def preparation_request(source: SourceSpec, panel_mapping: pd.DataFrame,
                        settings: PreparationSettings, shape_yx: tuple[int, int],
                        segmentation=None) -> dict:
    """Build the complete scientific request from reviewed identities and actual settings."""
    fields = ["panel_id","source_channel_index","output_channel_index","keep_channel",
              "source_name","marker_id","role","nuclear","body"]
    scientific_panel = panel_mapping[[key for key in fields if key in panel_mapping]]
    identity = source
    image_request = {"source":source_identity(identity),
                     "panel":json_value(scientific_panel.astype(object).where(pd.notna(scientific_panel),None).to_dict("records")),
                     "normalization":{"percentile":99.5,"method":"linear","lower":0}}
    mask_request = {"status": source.mask_status}
    # Legacy requests omit this field and therefore retain the historical
    # pending behavior.  A new packet that explicitly enables the fallback
    # carries the policy in its immutable request so every worker agrees.
    if source.mask_status == "unknown" and settings.allow_unknown_mask_segmentation:
        mask_request["unknown_status_policy"] = "cpsam_v2"
    if source.mask_status == "unusable":
        if source.mask is not None or source.mask_rejection is None:
            raise ValueError("Unusable mask requires rejection evidence without a pixel mask")
        mask_request["rejection"] = json_value(asdict(source.mask_rejection))
    elif source.mask_rejection is not None:
        raise ValueError("Mask rejection requires unusable status")
    if source.mask:
        frame_keys = ("kind", "path", "paths", "window", "acquisition_id", "z_index",
                      "source_revision", "valid_bounds", "valid_mask", "sample_region")
        mask_request = {"source":json_value(identity.mask),"file_identity":file_identity(identity.mask.path) if identity.mask.path.is_file() else {"missing":True},
                        "image_frame":{key:image_request["source"][key] for key in frame_keys},
                        "validity_identity":file_identity(identity.valid_mask) if identity.valid_mask else None,
                        "sample_region_identity":file_identity(identity.sample_region) if identity.sample_region else None,
                        "shape_yx":list(shape_yx)}
    segmentation_request = {"precision":settings.precision, "weights":file_identity(settings.weights) if settings.weights and settings.weights.is_file() else None,
                            "parameters":settings.segmentation_parameters,
                            "stitching":{"core_size":1024,"halo":128,"match_iou":0.5}}
    if segmentation:
        segmentation_request["stitching"] = {"core_size":segmentation.core_size,
            "halo":segmentation.halo,"match_iou":segmentation.match_iou}
    sampling_request = {"reviewed":settings.sampling_reviewed,"bins":settings.tissue_bins,"method":"max_otsu_open_close"}
    request = json_value({"schema":"cellweft.post_input.v2","image":image_request,"mask":mask_request,"segmentation":segmentation_request,
                          "sampling":sampling_request})
    return request
