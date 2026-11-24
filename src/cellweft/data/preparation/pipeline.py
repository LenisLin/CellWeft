"""Single-image dependency control, reuse, segmentation handoff and publication."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader, ImageRecord, ImageResult, MaskSource, PreparationSettings, SourceSpec, SharedImageResult
from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.io.writers import publish_image, write_image, write_json, write_table
from cellweft.data.preparation.crops import attach_crop_origins
from cellweft.data.preparation.indexing import build_cell_table
from cellweft.data.preparation.intensity import prepare_shared_image
from cellweft.data.preparation.masks import check_existing_mask, open_aligned_mask, finalize_mask
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.validation import validate_products
from cellweft.data.preparation.sampling import TissueEstimate, estimate_tissue_region, select_training_origins
from cellweft.data.preparation.neighbors import compute_neighbors
from cellweft.data.preparation.quantification import quantify_proteins
from cellweft.data.preparation.segmentation import CellposeExecutor, build_segmentation_input, content_identity
from cellweft.data.preparation.requests import file_identity, source_identity, preparation_request


def _completed_segmentation_result(task: dict) -> dict | None:
    """Accept a matching receipt only when its referenced mask is present."""
    path = Path(task["result_path"])
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt["task_id"] != task["task_id"] or receipt["request"] != task["request"]:
        raise ValueError("Segmentation result does not match the current image task")
    if receipt.get("parameters") != task["parameters"] or receipt.get("stitching") != task["stitching"]:
        raise ValueError("Segmentation result parameters differ from the submitted task")
    if receipt.get("precision") != task["precision"]:
        raise ValueError("Segmentation result precision differs from the submitted task")
    if receipt.get("content_identity") != task["content_identity"]:
        raise ValueError("Segmentation result used different input or weights content")
    return receipt if (path.parent / receipt["mask"]).is_file() else None


def _relocation_neutral(value):
    """Compare frozen science while permitting only copied asset locations to move."""
    if isinstance(value, dict):
        return {key: ("<relocated>" if key in {"path", "valid_mask", "sample_region"} and isinstance(item, str)
                      else _relocation_neutral(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocation_neutral(item) for item in value]
    return value


def checked_canonical_request(actual: dict, canonical: dict) -> dict:
    """Accept an externally verified frozen request only for path relocation."""
    if not isinstance(canonical, dict) or set(actual) != set(canonical):
        raise ValueError("Frozen preparation request has a different schema")
    if _relocation_neutral(actual) != _relocation_neutral(canonical):
        raise ValueError("Frozen preparation request differs from physical source or scientific settings")
    return canonical


def prepare_image(record: ImageRecord, source: SourceSpec, reader: ImageReader,
                  panel_mapping: pd.DataFrame, settings: PreparationSettings,
                  segmentation: CellposeExecutor | None = None, *,
                  identity_source: SourceSpec | None = None,
                  canonical_request: dict | None = None,
                  cpu_executor: Callable[..., dict[str, str]] | None = None,
                  shared_executor: Callable[..., SharedImageResult] | None = None) -> ImageResult:
    """Run one image under a cross-process writer lease; preserve reviewed study tables."""
    destination = record.output_path / "images" / record.image_id
    with resource_lock(destination, name=".preparation.lock"):
        return _prepare_image(record,source,reader,panel_mapping,settings,segmentation,
                              identity_source=identity_source,canonical_request=canonical_request,cpu_executor=cpu_executor,
                              shared_executor=shared_executor)


def _shared_source(path: Path, count: int, source: SourceSpec) -> SourceSpec:
    return replace(tiff_source(path,count),valid_mask=source.valid_mask,valid_bounds=source.valid_bounds,
                   sample_region=source.sample_region)


def _prepare_image(record: ImageRecord, source: SourceSpec, reader: ImageReader,
                  panel_mapping: pd.DataFrame, settings: PreparationSettings,
                  segmentation: CellposeExecutor | None = None, *,
                  identity_source: SourceSpec | None = None,
                  canonical_request: dict | None = None,
                  cpu_executor: Callable[..., dict[str, str]] | None = None,
                  shared_executor: Callable[..., SharedImageResult] | None = None) -> ImageResult:
    """Prepare or resume one independent image, publishing actual completed products.

    None as executor requests file handoff. A matching worker result is accepted
    on the next call. Existing valid masks never depend on segmentation signals.
    """
    destination = record.output_path / "images" / record.image_id
    destination.mkdir(parents=True,exist_ok=True)
    metadata_path = destination / "metadata.json"
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if old.get("publication_incomplete"):
        old_request = old.get("request",{})
        if "image" in old_request and "mask" in old_request:
            # Older interrupted checkpoints retained these identities only here.
            old.setdefault("image_request",old_request["image"])
            old.setdefault("mask_request",old_request["mask"])
    physical_request = preparation_request(identity_source or source,panel_mapping,settings,reader.shape[-2:],segmentation)
    request = checked_canonical_request(physical_request,canonical_request) if canonical_request is not None else physical_request
    image_request,mask_request = request['image'],request['mask']
    segmentation_request,sampling_request = request['segmentation'],request['sampling']
    old_products = old.get("products",{})
    available = {name for name, path in old_products.items() if Path(path).is_file()}
    products_exist = available == old_products.keys()
    task_old = old.get("segmentation_task")
    pending = task_old is not None and "mask" not in available
    overview_pending = bool(source.mask and source.mask.path.is_file() and source.mask.overview_required
                            and source.mask.overview_judgment is None)
    if (old.get("request") == request and products_exist and not pending
            and not (overview_pending and "mask" in available) and not old.get("publication_incomplete")):
        try:
            validate_products(old_products,reader.shape[-2:])
        except (ValueError, OSError):
            with resource_lock(destination):
                write_json(metadata_path,{**old,"completed":[],"publication_incomplete":True,
                                          "product_status":{k:"pending" for k in old_products}})
            raise
        return ImageResult(metadata_path,old_products,old.get("gaps",[]),True)
    # Persist dependency invalidation before computing or accepting a new attempt.
    image_changed = old.get("image_request") != image_request
    mask_changed = old.get("mask_request") != mask_request
    prediction_changed = old.get("request",{}).get("segmentation") != segmentation_request
    affected = {"image","channels","expression"} if image_changed else set()
    if mask_changed or (old.get("mask_origin") == "cpsam_v2" and (image_changed or prediction_changed)) or overview_pending:
        affected.update({"mask","cells","expression","neighbors"})
    missing = set(old_products)-available
    affected.update(missing)
    if "mask" in missing: affected.update({"cells","expression","neighbors"})
    if "cells" in missing: affected.update({"expression","neighbors"})
    if "channels" in missing: affected.update({"image","expression"})
    with resource_lock(destination):
        pending_metadata = {**old,"request":request,"image_request":image_request,
                            "mask_request":mask_request,"publication_incomplete":True,
                            "products":{k:v for k,v in old_products.items() if k not in affected},
                            "completed":[k for k in old.get("completed",[]) if k not in affected],
                            "segmentation_task":task_old if not (image_changed or mask_changed or prediction_changed) else None,
                            "product_status":{k:("pending" if k in affected else v) for k,v in old.get("product_status",{}).items()}}
        if image_changed or old.get("request",{}).get("sampling") != sampling_request:
            pending_metadata["sampling"] = {"status":"pending","count":None}
        write_json(metadata_path,pending_metadata)
    settings.cache_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare_",dir=destination.parent) as temporary:
        staging = Path(temporary)
        products = {}
        gaps = []
        # Reuse shared pixels independently of mask identity and downstream completion.
        reusable_channels = (pd.read_parquet(old_products["channels"])
                             if old.get("image_request") == image_request and "channels" in available else None)
        # Pending publication removes invalid products. Channel pages still tell
        # us whether an absent image must be rebuilt after an interrupted attempt.
        if (reusable_channels is not None
                and (not reusable_channels.tiff_page.notna().any() or "image" in available)):
            channels = reusable_channels
            shared_path = Path(old_products["image"]) if "image" in old_products else None
            products["channels"] = old_products["channels"]
            if shared_path:
                products["image"] = str(shared_path)
        else:
            shared = (shared_executor(record=record,source=source,reader=reader,
                        panel_mapping=panel_mapping,staging=staging,settings=settings,
                        image_request=image_request)
                      if shared_executor is not None else
                      prepare_shared_image(reader,panel_mapping,staging/"image.tiff",settings,source.normalized_q995))
            channels, shared_path = shared.channels, shared.path
            write_table(staging/"channels.parquet",channels)
            products["channels"] = str(destination/"channels.parquet")
            if shared_path:
                products["image"] = str(destination/"image.tiff")
        for row in channels.itertuples():
            if row.numerical_reason in {"nonfinite measured pixels","no measured pixels"}:
                gaps.append(f"channel {row.source_channel_index}: {row.numerical_reason}")
        mask_path, mask_identity, mask_check, mask_origin = None, None, None, None
        source_mask_check = old.get("source_mask_check") if old.get("mask_request") == mask_request else None
        if source.mask_status == "unusable":
            rejection = mask_request["rejection"]
            source_mask_check = {"usable":False, "reason":rejection["reason"],
                                 "evidence":rejection["evidence"],
                                 "problems":[rejection["reason"]],
                                 "mask_type":None, "dimensions_and_range":None,
                                 "integer_index":None, "overview":None}
        task = None
        # Predicted masks depend on the complete segmentation request; source masks
        # depend only on their own evidence and the selected image coordinate frame.
        if (old.get("mask_origin") == "cpsam_v2" and old.get("image_request") == image_request
                and old.get("request",{}).get("segmentation") == segmentation_request
                and old.get("mask_request") == mask_request
                and "mask" in available and not overview_pending):
            mask_path = Path(old_products["mask"])
            mask_identity, mask_check = old["mask_identity"], old["mask_check"]
            mask_origin, task = "cpsam_v2", task_old
        elif source.mask and source.mask.path.is_file():
            if (old.get("mask_origin") == "source" and old.get("mask_request") == mask_request
                    and "mask" in available):
                mask_path = Path(old_products["mask"])
                mask_identity, mask_check = old["mask_identity"], old["mask_check"]
                mask_origin = "source"
            elif source_mask_check is None or source_mask_check["usable"]:
                with open_aligned_mask(source.mask,settings.cache_dir,decode_budget=settings.mask_decode_budget) as supplied:
                    if source_mask_check is None:
                        source_mask_check = check_existing_mask(supplied,reader.shape[-2:],source.mask,settings.block_size)
                    mask_check = source_mask_check
                    if mask_check["usable"]:
                        candidate = staging/"mask.tiff"
                        if finalize_mask(supplied,reader,candidate,tile_size=settings.tile_size,compression=settings.compression):
                            mask_path = candidate
                            mask_identity, mask_origin = uuid.uuid4().hex, "source"
                        else:
                            gaps.append("source mask has no measured foreground")
            else:
                mask_check = source_mask_check

        if source.mask and not source.mask.path.is_file():
            source_mask_check = {"usable":False,"problems":["source mask file missing"]}
            gaps.append("source mask file missing")
        unknown_mask_fallback = (source.mask_status == "unknown"
                                 and settings.allow_unknown_mask_segmentation)
        if source.mask_status == "unknown":
            gaps.append("source mask binding unknown"
                        + ("; cpsam_v2 fallback enabled" if unknown_mask_fallback else ""))
        if overview_pending:
            gaps.append("source mask requires one whole-image overview judgment")
        if (mask_path is None and shared_path is not None and not overview_pending
                and (source.mask_status != "unknown" or unknown_mask_fallback)
                and (source.mask is None or source.mask.path.is_file())):
            # Reuse only a task tied to the same shared pixels, model and parameters.
            receipt = None
            if (old.get("image_request") == image_request and old.get("mask_request") == mask_request
                    and old.get("request",{}).get("segmentation") == segmentation_request and task_old):
                receipt = _completed_segmentation_result(task_old)
                if Path(task_old["input_path"]).is_file() or receipt is not None:
                    task = task_old
            if task is None and settings.weights and settings.weights.is_file():
                task_id = uuid.uuid4().hex
                task_dir = settings.cache_dir/"segmentation"/task_id
                task_dir.mkdir(parents=True)
                signal_path = task_dir/"signal.tiff"
                with open_source(_shared_source(shared_path,int(channels.tiff_page.notna().sum()),source)) as shared_reader:
                    signal = build_segmentation_input(shared_reader,channels,signal_path,settings)
                if signal["gap"]:
                    gaps.append(signal["gap"])
                else:
                    task = {"task_id":task_id,"study":record.study,"image_id":record.image_id,
                            "request":request,"weights":str(settings.weights.resolve()),
                            "parameters":settings.segmentation_parameters,"precision":settings.precision,
                            "stitching":segmentation_request["stitching"],"signal":signal,
                            "input_path":str(signal_path),"result_path":str(task_dir/"result.json"),
                            "content_identity":{"input":content_identity(signal_path),
                                                "weights":content_identity(settings.weights)}}
                    write_json(task_dir/"task.json",task)
            elif task is None:
                gaps.append("no local cpsam_v2 weights configured")
            if task:
                with resource_lock(destination):
                    active = json.loads(metadata_path.read_text())
                    active["segmentation_task"] = task
                    write_json(metadata_path,active)
                result_path = Path(task["result_path"])
                task_path = Path(task["input_path"]).parent / "task.json"
                if not task_path.is_file():
                    write_json(task_path, task)
                if receipt is None and segmentation:
                    segmentation.execute(task)
                    receipt = _completed_segmentation_result(task)
                if receipt is not None:
                    if receipt.get("whole_cell_confirmed"):
                        candidate = result_path.parent/receipt["mask"]
                        with open_source(tiff_source(candidate,1)) as predicted:
                            mask_check = check_existing_mask(predicted,reader.shape[-2:],
                                MaskSource(candidate,"whole_cell"),settings.block_size)
                            if mask_check["usable"]:
                                candidate_path = staging/"mask.tiff"
                                if finalize_mask(predicted,reader,candidate_path,tile_size=settings.tile_size):
                                    mask_path = candidate_path
                                    mask_identity, mask_origin = uuid.uuid4().hex, "cpsam_v2"
                                else:
                                    gaps.append("predicted mask has no measured foreground")
                            else:
                                gaps.extend(mask_check["problems"])
                    else:
                        gaps.append("new zero-body mask requires one whole-image whole-cell judgment")
                else:
                    gaps.append("segmentation result pending")
        if mask_path is None:
            gaps.append("whole-cell mask unavailable")
        else:
            products["mask"] = str(destination/"mask.tiff") if mask_path.parent == staging else str(mask_path)
            geometry_reusable = old.get("mask_identity") == mask_identity and {"cells","neighbors"}.issubset(available)
            remote_products = None
            if cpu_executor is not None and not geometry_reusable and shared_path is not None:
                remote_products = cpu_executor(record=record,source=source,image_path=shared_path,
                    mask_path=mask_path,mask_identity=mask_identity,channels=channels,
                    staging=staging,settings=settings)
            with open_source(tiff_source(mask_path,1)) as mask_reader:
                if remote_products is not None:
                    cells = pd.read_parquet(remote_products['cells'])
                    products.update({key:str(destination/Path(path).name) for key,path in remote_products.items()})
                elif geometry_reusable:
                    cells = pd.read_parquet(old_products["cells"])
                    products.update(cells=old_products["cells"],neighbors=old_products["neighbors"])
                else:
                    cells = attach_crop_origins(mask_reader,build_cell_table(mask_reader,record.study,record.image_id,mask_identity,settings.block_size))
                    write_table(staging/"cells.parquet",cells)
                    write_table(staging/"neighbors.parquet",compute_neighbors(cells,workers=settings.cpu_workers))
                    products.update(cells=str(destination/"cells.parquet"),neighbors=str(destination/"neighbors.parquet"))
                if shared_path is not None and channels.expression_column.notna().any():
                    if remote_products is not None:
                        if 'expression' not in remote_products:
                            raise ValueError('CPU result omitted required expression product')
                    elif geometry_reusable and not image_changed and "expression" in available:
                        products["expression"] = old_products["expression"]
                    else:
                        with open_source(_shared_source(shared_path,int(channels.tiff_page.notna().sum()),source)) as shared_reader:
                            expression = quantify_proteins(shared_reader,mask_reader,cells,channels,settings.block_size)
                        write_table(staging/"expression.parquet",expression)
                        products["expression"] = str(destination/"expression.parquet")
                else:
                    gaps.append("no effective protein channels")
        sampling = {"count":0,"status":"complete","size":224,"stride":112,"evaluated":True,"request":sampling_request}
        if not settings.sampling_reviewed:
            declared = TissueEstimate(None,{"method":sampling_request["method"],"bins":settings.tissue_bins,
                "kernel":3,"opening":1,"closing":1})
            chosen = select_training_origins(reader,channels,declared,reviewed=False)
            sampling = {**chosen.metadata,"evaluated":False,"request":sampling_request,
                "count":None,"status":"pending","gap":chosen.gap}
        if shared_path is not None:
            if not image_changed and old.get("request",{}).get("sampling") == sampling_request and old.get("sampling",{}).get("evaluated") and old["sampling"].get("request") == sampling_request:
                sampling = old["sampling"]
            elif settings.sampling_reviewed:
                with open_source(_shared_source(shared_path,int(channels.tiff_page.notna().sum()),source)) as shared_reader:
                    with estimate_tissue_region(shared_reader,channels,settings.cache_dir,bins=settings.tissue_bins,block_size=settings.block_size) as tissue:
                        chosen = select_training_origins(shared_reader,channels,tissue,reviewed=settings.sampling_reviewed)
                        sampling = {**chosen.metadata,"evaluated":True,"request":sampling_request,"count":chosen.count,"status":"pending" if chosen.count is None else "complete","gap":chosen.gap}
        elif settings.sampling_reviewed:
            declared = TissueEstimate(None,{"method":sampling_request["method"],"bins":settings.tissue_bins,
                "kernel":3,"opening":1,"closing":1},"no shared image for training tissue estimate")
            chosen = select_training_origins(reader,channels,declared,reviewed=True)
            sampling = {**chosen.metadata,"evaluated":False,"request":sampling_request,
                "count":None,"status":"pending","gap":chosen.gap}
        if sampling.get("gap"): gaps.append(sampling["gap"])
        failed = channels.numerical_reason.isin(["nonfinite measured pixels","no measured pixels"])
        incomplete = set()
        if failed.any():
            incomplete.add("image")
            if (failed & channels.role.eq("downstream")).any(): incomplete.add("expression")
        if mask_origin == "cpsam_v2" and (failed & (channels.nuclear | channels.body)).any():
            incomplete.update({"mask","cells","expression","neighbors"})
            gaps.append("prediction depends on failed segmentation channels")
        completed = sorted(set(products)-incomplete)
        metadata = {"schema":"cellweft.preparation.v2","study":record.study,"image_id":record.image_id,
                    "shape_yx":reader.shape[-2:],"pixel_size_um":record.pixel_size_um,
                    "source_window":source.window,"request":request,"image_request":image_request,
                    "mask_request":mask_request,"mask_identity":mask_identity,"mask_check":mask_check,
                    "mask_policy":{"unknown_status":
                                   "cpsam_v2" if settings.allow_unknown_mask_segmentation else "pending"},
                    "mask_origin":mask_origin,"source_mask_check":source_mask_check,
                    "segmentation_task":task,"products":products,"completed":completed,"gaps":gaps,
                    "product_status":{key:"complete" if key in completed else "partial" for key in products},
                    "sampling":sampling,
                    "crop":{"size":224,"stride":112,"edge":"cover","coordinates":"local_xy"}}
        publish_image(staging,destination,metadata)
        return ImageResult(metadata_path,products,gaps)
