"""Single-image dependency control, reuse, segmentation handoff and publication."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cellweft.data.contracts import ImageReader, ImageRecord, ImageResult, MaskSource, PreparationSettings, SourceSpec
from cellweft.data.io.readers import open_source, tiff_source
from cellweft.data.io.writers import json_value, publish_image, write_image, write_json, write_table
from cellweft.data.preparation.crops import attach_crop_origins
from cellweft.data.preparation.indexing import build_cell_table
from cellweft.data.preparation.intensity import prepare_shared_image
from cellweft.data.preparation.masks import check_existing_mask
from cellweft.data.preparation.neighbors import compute_neighbors
from cellweft.data.preparation.quantification import quantify_proteins
from cellweft.data.preparation.segmentation import CellposeExecutor, build_segmentation_input, content_identity


def file_identity(path: Path) -> dict:
    """Identify an input file by its resolved path, size and modification timestamp."""
    stat = path.stat()
    return {"path":str(path.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def source_identity(source: SourceSpec) -> dict:
    """Capture source selection separately from mask and panel-dependent processing."""
    value = asdict(source)
    value.pop("mask")
    files = [source.path, *source.paths]
    if source.valid_mask:
        files.append(source.valid_mask)
    if source.kind == "zarr" and not source.source_revision:
        raise ValueError("Directory-backed source requires its explicit source_revision")
    value["file_identity"] = [file_identity(p) for p in dict.fromkeys(files)]
    return json_value(value)


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
    if receipt.get("content_identity") != task["content_identity"]:
        raise ValueError("Segmentation result used different input or weights content")
    return receipt if (path.parent / receipt["mask"]).is_file() else None


def prepare_image(record: ImageRecord, source: SourceSpec, reader: ImageReader,
                  panel_mapping: pd.DataFrame, settings: PreparationSettings,
                  segmentation: CellposeExecutor | None = None) -> ImageResult:
    """Prepare or resume one independent image, publishing actual completed products.

    None as executor requests file handoff. A matching worker result is accepted
    on the next call. Existing valid masks never depend on segmentation signals.
    """
    destination = record.output_path / "images" / record.image_id
    destination.mkdir(parents=True,exist_ok=True)
    metadata_path = destination / "metadata.json"
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    image_request = {"source":source_identity(source),
                     "panel":json_value(panel_mapping.where(pd.notna(panel_mapping),None).to_dict("records")),
                     "normalization":{"percentile":99.5,"method":"linear","lower":0}}
    mask_request = None
    if source.mask:
        frame_keys = ("kind", "path", "paths", "window", "acquisition_id", "z_index",
                      "source_revision", "file_identity")
        mask_request = {"source":json_value(source.mask),"file_identity":file_identity(source.mask.path),
                        "image_frame":{key:image_request["source"][key] for key in frame_keys},
                        "shape_yx":list(reader.shape[-2:])}
    segmentation_request = {"weights":file_identity(settings.weights) if settings.weights and settings.weights.is_file() else None,
                            "parameters":settings.segmentation_parameters,
                            "stitching":{"core_size":1024,"halo":128,"match_iou":0.5}}
    if segmentation:
        segmentation_request["stitching"] = {"core_size":segmentation.core_size,
            "halo":segmentation.halo,"match_iou":segmentation.match_iou}
    request = json_value({"image":image_request,"mask":mask_request,"segmentation":segmentation_request})
    old_products = old.get("products",{})
    available = {name for name, path in old_products.items() if Path(path).is_file()}
    products_exist = available == old_products.keys()
    task_old = old.get("segmentation_task")
    pending = task_old is not None and "mask" not in available
    overview_pending = bool(source.mask and source.mask.overview_required
                            and source.mask.overview_judgment is None)
    if (old.get("request") == request and products_exist and not pending
            and not (overview_pending and "mask" in available)):
        return ImageResult(metadata_path,old_products,old.get("gaps",[]),True)
    settings.cache_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare_",dir=destination.parent) as temporary:
        staging = Path(temporary)
        products = {}
        gaps = []
        # Reuse shared pixels independently of mask identity and downstream completion.
        if (old.get("image_request") == image_request and "channels" in available
                and ("image" not in old_products or "image" in available)):
            channels = pd.read_parquet(old_products["channels"])
            shared_path = Path(old_products["image"]) if "image" in old_products else None
            products["channels"] = old_products["channels"]
            if shared_path:
                products["image"] = str(shared_path)
        else:
            shared = prepare_shared_image(reader,panel_mapping,staging/"image.tiff",settings,source.normalized_q995)
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
        task = None
        # Predicted masks depend on the complete segmentation request; source masks
        # depend only on their own evidence and the selected image coordinate frame.
        if (old.get("mask_origin") == "cpsam_v2" and old.get("request") == request
                and "mask" in available and not overview_pending):
            mask_path = Path(old_products["mask"])
            mask_identity, mask_check = old["mask_identity"], old["mask_check"]
            mask_origin, task = "cpsam_v2", task_old
        elif source.mask:
            if (old.get("mask_origin") == "source" and old.get("mask_request") == mask_request
                    and "mask" in available):
                mask_path = Path(old_products["mask"])
                mask_identity, mask_check = old["mask_identity"], old["mask_check"]
                mask_origin = "source"
            elif source_mask_check is None or source_mask_check["usable"]:
                descriptor = tiff_source(source.mask.path,1,source.mask.window)
                descriptor = SourceSpec(**{**asdict(descriptor),"channels":(source.mask.page,)})
                with open_source(descriptor) as supplied:
                    if source_mask_check is None:
                        source_mask_check = check_existing_mask(supplied,reader.shape[-2:],source.mask,settings.block_size)
                    mask_check = source_mask_check
                    if mask_check["usable"]:
                        mask_path = staging/"mask.tiff"
                        write_image(mask_path,(1,*reader.shape[-2:]),np.dtype("uint32"),
                                    lambda _, win: supplied.read_window([0],win)[0],
                                    tile_size=settings.tile_size,compression=settings.compression)
                        mask_identity, mask_origin = uuid.uuid4().hex, "source"
            else:
                mask_check = source_mask_check

        if overview_pending:
            gaps.append("source mask requires one whole-image overview judgment")
        if mask_path is None and shared_path is not None and not overview_pending:
            # Reuse only a task tied to the same shared pixels, model and parameters.
            receipt = None
            if old.get("request") == request and task_old:
                receipt = _completed_segmentation_result(task_old)
                if Path(task_old["input_path"]).is_file() or receipt is not None:
                    task = task_old
            if task is None and settings.weights and settings.weights.is_file():
                task_id = uuid.uuid4().hex
                task_dir = settings.cache_dir/"segmentation"/task_id
                task_dir.mkdir(parents=True)
                signal_path = task_dir/"signal.tiff"
                with open_source(tiff_source(shared_path,int(channels.tiff_page.notna().sum()))) as shared_reader:
                    signal = build_segmentation_input(shared_reader,channels,signal_path,settings)
                if signal["gap"]:
                    gaps.append(signal["gap"])
                else:
                    task = {"task_id":task_id,"study":record.study,"image_id":record.image_id,
                            "request":request,"weights":str(settings.weights.resolve()),
                            "parameters":settings.segmentation_parameters,
                            "stitching":segmentation_request["stitching"],"signal":signal,
                            "input_path":str(signal_path),"result_path":str(task_dir/"result.json"),
                            "content_identity":{"input":content_identity(signal_path),
                                                "weights":content_identity(settings.weights)}}
                    write_json(task_dir/"task.json",task)
            elif task is None:
                gaps.append("no local cpsam_v2 weights configured")
            if task:
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
                                mask_path = staging/"mask.tiff"
                                write_image(mask_path,(1,*reader.shape[-2:]),np.dtype("uint32"),
                                    lambda _, win: predicted.read_window([0],win)[0],tile_size=settings.tile_size)
                                mask_identity, mask_origin = uuid.uuid4().hex, "cpsam_v2"
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
            with open_source(tiff_source(mask_path,1)) as mask_reader:
                cells = build_cell_table(mask_reader,record.study,record.image_id,mask_identity,settings.block_size)
                if shared_path is not None and channels.expression_column.notna().any():
                    with open_source(tiff_source(shared_path,int(channels.tiff_page.notna().sum()))) as shared_reader:
                        expression = quantify_proteins(shared_reader,mask_reader,cells,channels,settings.block_size)
                    write_table(staging/"expression.parquet",expression)
                    products["expression"] = str(destination/"expression.parquet")
                else:
                    gaps.append("no effective protein channels")
                neighbors = compute_neighbors(cells,workers=settings.cpu_workers)
                cells = attach_crop_origins(mask_reader,cells)
                write_table(staging/"cells.parquet",cells)
                write_table(staging/"neighbors.parquet",neighbors)
                products.update(cells=str(destination/"cells.parquet"),neighbors=str(destination/"neighbors.parquet"))
        metadata = {"schema":"cellweft.preparation.v1","study":record.study,"image_id":record.image_id,
                    "shape_yx":reader.shape[-2:],"pixel_size_um":record.pixel_size_um,
                    "source_window":source.window,"request":request,"image_request":image_request,
                    "mask_request":mask_request,"mask_identity":mask_identity,"mask_check":mask_check,
                    "mask_origin":mask_origin,"source_mask_check":source_mask_check,
                    "segmentation_task":task,"products":products,"completed":list(products),"gaps":gaps,
                    "crop":{"size":224,"stride":112,"edge":"cover","coordinates":"local_xy"}}
        publish_image(staging,destination,metadata)
        return ImageResult(metadata_path,products,gaps)
