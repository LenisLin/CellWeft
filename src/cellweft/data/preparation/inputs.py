"""Validated, revision-bound per-study input delivery; no source-specific inference."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import json
from typing import Callable, ContextManager, Iterator, MutableMapping

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import ImageRecord, MaskRejection, MaskSource, SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.study_input import validate_image_metadata

# Vector row numbers and expressions cannot invalidate input pixels or quantification.
PANEL_INPUT_FIELDS = ["panel_id", "source_channel_index", "source_name", "marker_id",
                      "mapping_status", "role", "nuclear", "body", "keep_channel",
                      "output_channel_index", "s1_original_marker"]

# These are the image-table fields consumed by SourceSpec/preparation_request.
# Clinical and sample annotations remain in the full binding, but changing only
# those annotations must not invalidate already computed pixel products.
PROCESSING_IMAGE_FIELDS = [
    "image_id", "panel_id", "source_dtype", "source_x0_px", "source_y0_px",
    "source_x1_px", "source_y1_px", "height_px", "width_px", "source_channel_count",
    "channel_count", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px",
    "pixel_size_x_um", "pixel_size_y_um", "input_image_path", "valid_pixels_path",
    "sample_region_path", "source_revision", "intensity_description",
    "canvas_policy",
]
_HEADER_BATCH_SIZE = 16


def file_revision(path: Path) -> dict:
    """Identify a declared local asset by path, size and nanosecond modification time."""
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def processing_binding(binding: dict) -> dict:
    """Return the D01 binding components that affect scientific image products.

    The full binding remains the acceptance and provenance record.  This
    projection is only for deciding whether a metadata-only rebinding can
    reuse an image request and its existing products.
    """
    images = []
    for row in binding.get("images", []):
        images.append({key: row.get(key) for key in PROCESSING_IMAGE_FIELDS})
    return {
        "image_ids": list(binding.get("image_ids", [])),
        "images": images,
        "panel": binding.get("panel", []),
        "assets": binding.get("assets", []),
    }


def same_processing_binding(left: dict, right: dict) -> bool:
    """Compare only the binding fields that can change image computation."""
    return processing_binding(left) == processing_binding(right)


@contextmanager
def input_header_lease(path: Path, *, study_root: Path | None = None) -> Iterator[None]:
    """Hold the study and per-image input leases for one revision or header check."""
    study = resource_lock(study_root, shared=True) if study_root is not None else nullcontext()
    with study:
        with resource_lock(path.parent, shared=True, name=".input.lock"):
            yield


def _present(value) -> bool:
    if value is None:
        return False
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False
    return value != ""


def _header_cache_key(row, data_root: Path) -> str:
    """Build a stat-and-header expectation key for one selected image."""
    assets = {}
    for field in ("input_image_path", "valid_pixels_path", "sample_region_path"):
        value = getattr(row, field, None)
        if _present(value):
            assets[field] = file_revision(_path(data_root, value))
    expected = {
        field: row_value.item() if isinstance(row_value, np.generic) else row_value
        for field in ("image_id", "source_dtype", "height_px", "width_px",
                      "channel_count", "source_channel_count")
        if (row_value := getattr(row, field, None)) is not None
    }
    return json.dumps({"assets": assets, "expected": expected}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


def _header_cache_hit(header_cache: MutableMapping[str, dict] | None,
                      image_id: str, key: str) -> bool:
    if header_cache is None:
        return False
    value = header_cache.get(image_id)
    return isinstance(value, dict) and value.get("validated") is True and value.get("key") == key


def validate_panel_mapping(mapping: pd.DataFrame, *, input_channel_count: int) -> pd.DataFrame:
    """Preserve all source rows and validate their separate source/input positions."""
    rows = mapping.copy().sort_values("source_channel_index").reset_index(drop=True)
    if rows.panel_id.nunique() != 1 or rows.source_channel_index.tolist() != list(range(len(rows))):
        raise ValueError("Panel must contain every source channel once in order")
    for key in ("keep_channel", "nuclear", "body"):
        parsed = rows[key].astype(str).str.lower().map({"true": True, "false": False})
        if parsed.isna().any():
            raise ValueError(f"Panel {key} requires explicit booleans")
        rows[key] = parsed
    rows["output_channel_index"] = pd.to_numeric(rows.output_channel_index, errors="raise").astype("Int64")
    kept = rows[rows.keep_channel]
    if (sorted(kept.output_channel_index.dropna().tolist()) != list(range(input_channel_count))
            or len(kept) != input_channel_count or rows.loc[~rows.keep_channel, "output_channel_index"].notna().any()):
        raise ValueError("Kept channels must correspond exactly to input TIFF pages")
    if not rows.role.isin(["downstream", "segmentation-only", "exclude"]).all():
        raise ValueError("Unknown panel role")
    if ((rows.nuclear | rows.body) & ~rows.keep_channel).any():
        raise ValueError("Selected segmentation channel has no input pixels; repair input delivery")
    if kept.marker_id.isna().any() or kept.marker_id.eq("").any() or kept.role.eq("exclude").any():
        raise ValueError("Kept channels require a measurement identity and active role")
    if kept.loc[kept.role == "downstream", "marker_id"].duplicated().any():
        raise ValueError("Duplicate downstream marker requires explicit source selection")
    return rows


def _path(data_root: Path, value: str, *, resolved_root: Path | None = None) -> Path:
    root = data_root.resolve() if resolved_root is None else resolved_root
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input asset path is outside the declared data root")
    return path


@dataclass
class ReviewedStudyInputs:
    """Complete input tables plus the exact D01-bound selected image revision."""

    root: Path
    data_root: Path
    images: pd.DataFrame
    panel: pd.DataFrame
    samples: pd.DataFrame
    field_mapping: pd.DataFrame
    binding: dict

    def source(self, image_id: str, mask_binding: dict) -> tuple[ImageRecord, SourceSpec, pd.DataFrame]:
        """Resolve one reviewed input and its explicit mask status.

        Bound mask orientation is applied before its declared crop; input-frame
        bindings already include the input canvas and must not add padding.
        """
        row = self.images.set_index("image_id").loc[image_id]
        state = mask_binding["status"]
        if state not in {"unknown", "absent", "bound", "unusable"}:
            raise ValueError("Mask status must be unknown, absent, bound or unusable")
        rejection = None
        if state == "unusable":
            if set(mask_binding) != {"status", "reason", "evidence"}:
                raise ValueError("Unusable mask requires reason and source evidence only")
            rejection = MaskRejection(mask_binding["reason"], mask_binding["evidence"])
        panel = validate_panel_mapping(self.panel[self.panel.panel_id == row.panel_id],
                                       input_channel_count=int(row.channel_count))
        shape = (int(row.height_px), int(row.width_px))
        left, top, right, bottom = (int(row[f"pad_{side}_px"]) for side in ("left", "top", "right", "bottom"))
        scale = None if pd.isna(row.pixel_size_x_um) or pd.isna(row.pixel_size_y_um) else (float(row.pixel_size_x_um), float(row.pixel_size_y_um))
        mask = None
        if state == "bound":
            info = dict(mask_binding); info.pop("status")
            frame = info.pop("frame")
            info["path"] = _path(self.data_root, info["path"])
            if frame == "source":
                info["window"] = tuple(int(row[f"source_{name}_px"]) for name in ("x0", "y0", "x1", "y1"))
                info["padding_ltrb"] = (left, top, right, bottom)
            elif frame != "input":
                raise ValueError("Mask frame must be explicitly source or input")
            mask = MaskSource(**info)
        valid = None if pd.isna(row.valid_pixels_path) or row.valid_pixels_path == "" else _path(self.data_root, row.valid_pixels_path)
        region_value = row.get("sample_region_path")
        region = _path(self.data_root, region_value) if pd.notna(region_value) and region_value else None
        source = SourceSpec(image_id, row.panel_id, "tiff", _path(self.data_root, row.input_image_path),
                            tuple(range(int(row.channel_count))), pixel_size_um=scale, mask=mask,
                            valid_mask=valid, valid_bounds=(left, top, shape[1]-right, shape[0]-bottom),
                            source_revision=str(row.source_revision), intensity_description=row.intensity_description,
                            mask_status=state, sample_region=region, mask_rejection=rejection)
        record = ImageRecord(str(row.study), image_id, int(row.channel_count), scale,
                             source.path.parent, self.root / "panel_mapping.csv", self.root)
        return record, source, panel


def input_binding(study_root: Path, data_root: Path, image_ids: list[str], *,
                  asset_workers: int = 1) -> dict:
    """Build D01 evidence with bounded metadata concurrency and stable asset order.

    Every asset is resolved and stat'ed afresh; this function never grants approval.
    """
    if type(asset_workers) is not int or not 1 <= asset_workers <= 16:
        raise ValueError("asset_workers must be an integer in 1..16")
    images = pd.read_parquet(study_root / "images.parquet")
    if not image_ids or images.image_id.duplicated().any() or not set(image_ids).issubset(images.image_id):
        raise ValueError("Unknown or duplicate image identity in acceptance scope")
    selected = images.set_index("image_id").loc[sorted(image_ids)].reset_index()
    panel = pd.read_csv(study_root / "panel_mapping.csv", keep_default_na=False)
    panel = panel[panel.panel_id.isin(selected.panel_id)]
    panel = panel[[key for key in PANEL_INPUT_FIELDS if key in panel]].sort_values(["panel_id", "source_channel_index"])
    asset_values = []
    resolved_root = data_root.resolve()
    for row in selected.itertuples():
        for key in ("input_image_path", "valid_pixels_path", "sample_region_path"):
            value = getattr(row, key, None)
            if pd.notna(value) and value:
                asset_values.append(value)

    def asset_revision(value: str) -> dict:
        path = _path(data_root, value, resolved_root=resolved_root)
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    if asset_workers == 1:
        assets = [asset_revision(value) for value in asset_values]
    else:
        with ThreadPoolExecutor(max_workers=asset_workers) as pool:
            assets = list(pool.map(asset_revision, asset_values))
    return {"image_ids": sorted(image_ids), "images": json.loads(selected.to_json(orient="records")),
            "panel": json.loads(panel.to_json(orient="records")), "assets": assets,
            "samples": file_revision(study_root / "sample_metadata.csv"),
            "field_mapping": file_revision(study_root / "field_mapping.csv")}


def assert_reviewed_binding(study_root: Path, *, data_root: Path, binding: dict,
                            asset_workers: int = 1) -> None:
    """Recheck current acceptance/revocations under the caller's study lease.

    This is used immediately before consumption or publication, even if pixels
    did not change. Callers hold a shared or exclusive resource_lock throughout.
    """
    if input_binding(study_root,data_root,binding['image_ids'],
                     asset_workers=asset_workers) != binding:
        raise ValueError("Reviewed input revision changed before consumption")
    assert_reviewed_acceptance(pd.read_csv(study_root / "review_results.csv", keep_default_na=False), binding)


def assert_reviewed_acceptance(reviews: pd.DataFrame, binding: dict) -> None:
    """Check exact approval and blocking reviews separately from physical asset location."""
    current = reviews[(reviews.study == binding['images'][0]['study']) & (reviews.check_id == "D01")]
    latest = current.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
    checks = pd.read_csv(Path(__file__).resolve().parents[4]/"docs/data/templates/study_input/review_checks.csv")
    blocking = set(checks.loc[checks.blocking_scope == 'input','check_id'])
    all_latest = reviews.drop_duplicates(['scope','record_id','check_id'],keep='last')
    for issue in all_latest[(all_latest.study == binding['images'][0]['study']) & all_latest.check_id.isin(blocking) & all_latest.result.isin(['fail','unresolved'])].itertuples():
        detail = json.loads(issue.details) if issue.details.lstrip().startswith('{') else {}
        affected = detail.get('image_ids')
        if affected is None or set(affected).intersection(binding['image_ids']):
            raise ValueError(f"Unresolved input review: {issue.check_id}/{issue.record_id}")
    accepted = False
    for row in latest.itertuples():
        if row.result == "pass" and row.details.lstrip().startswith("{"):
            accepted |= json.loads(row.details).get("input_binding") == binding
    if not accepted:
        raise ValueError("No current D01 pass bound to this input revision and image scope")


def load_reviewed_study_inputs(study_root: Path, *, data_root: Path,
                               image_ids: list[str] | None = None,
                               accepted_scope_image_ids: list[str] | None = None,
                               header_cache: MutableMapping[str, dict] | None = None,
                               read_gate: Callable[[Path], ContextManager[object]] | None = None) -> ReviewedStudyInputs:
    """Require a current D01 pass bound to selected IDs and full input revision.

    Human-readable legacy scopes remain evidence for humans but are not parsed
    as permission to process unspecified images. Revisions need renewed binding.
    Cold header checks share one read gate per batch of at most 16 images.
    A delivery may select a subset of an explicitly supplied accepted scope.
    Its binding remains the full original scope and is rechecked in full;
    only selected TIFF headers and geometry are opened for consumption.
    """
    # Read and bind the tables under the study lease, then release it before
    # waiting for a caller-supplied NAS read slot.  The publisher acquires its
    # global write slot before requesting this same study lease.
    with resource_lock(study_root, shared=True):
        images = pd.read_parquet(study_root / "images.parquet")
        panel = pd.read_csv(study_root / "panel_mapping.csv", keep_default_na=False)
        samples = pd.read_csv(study_root / "sample_metadata.csv", keep_default_na=False)
        fields = pd.read_csv(study_root / "field_mapping.csv", keep_default_na=False)
        images = validate_image_metadata(images, samples, panel, fields)
        if images.study.nunique() != 1:
            raise ValueError("Input directory must contain one study")
        ids = sorted(image_ids if image_ids is not None else images.image_id.tolist())
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated selected image IDs")
        scope_ids = sorted(accepted_scope_image_ids if accepted_scope_image_ids is not None else ids)
        if len(scope_ids) != len(set(scope_ids)) or not set(ids).issubset(scope_ids):
            raise ValueError("Selected images must belong to the explicit accepted scope")
        binding = input_binding(study_root, data_root, scope_ids)
        assert_reviewed_binding(study_root,data_root=data_root,binding=binding)
        selected = images[images.image_id.isin(ids)].copy()
    cold_headers = []
    for row in selected.itertuples():
        if Path(row.image_id).name != row.image_id or row.image_id in {".", ".."}:
            raise ValueError("image_id must be one safe path component")
        shape = (int(row.height_px), int(row.width_px))
        policy = getattr(row, "canvas_policy", None)
        native_rectangle = isinstance(policy, str) and policy == "user_approved_native_rectangle"
        if min(shape) <= 0 or (shape[0] != shape[1] and not native_rectangle):
            raise ValueError("Input requires a positive square canvas or an explicitly approved native rectangle")
        l, t, r, b = (int(getattr(row, f"pad_{side}_px")) for side in ("left", "top", "right", "bottom"))
        if native_rectangle and (l or t or r or b):
            raise ValueError("Approved native rectangle must have no artificial padding")
        if min(l,t,r,b) < 0 or (row.source_x1_px-row.source_x0_px+l+r, row.source_y1_px-row.source_y0_px+t+b) != (shape[1],shape[0]):
            raise ValueError("Source window and padding do not match input geometry")
        selected_panel = panel[panel.panel_id == row.panel_id]
        if len(selected_panel) != int(row.source_channel_count):
            raise ValueError("source_channel_count differs from the complete panel")
        validate_panel_mapping(selected_panel, input_channel_count=int(row.channel_count))
        input_path = _path(data_root, row.input_image_path)
        with input_header_lease(input_path, study_root=study_root):
            cache_key = _header_cache_key(row, data_root)
            cached = _header_cache_hit(header_cache, row.image_id, cache_key)
        if not cached:
            cold_headers.append((row, input_path))
    for start in range(0, len(cold_headers), _HEADER_BATCH_SIZE):
        batch = cold_headers[start:start + _HEADER_BATCH_SIZE]
        gate = read_gate(batch[0][1]) if read_gate is not None else nullcontext()
        with gate:
            for row, input_path in batch:
                with input_header_lease(input_path, study_root=study_root):
                    shape = (int(row.height_px), int(row.width_px))
                    cache_key = _header_cache_key(row, data_root)
                    if _header_cache_hit(header_cache, row.image_id, cache_key):
                        continue
                    with tifffile.TiffFile(input_path) as tif:
                        if len(tif.pages) != row.channel_count or any(page.shape != shape or str(page.dtype) != row.source_dtype for page in tif.pages):
                            raise ValueError(f"{row.image_id}: input TIFF header differs from reviewed metadata")
                    for key in ("valid_pixels_path", "sample_region_path"):
                        value = getattr(row,key,None)
                        if not _present(value):
                            continue
                        with tifffile.TiffFile(_path(data_root, value)) as tif:
                            if len(tif.pages) != 1 or tif.pages[0].shape != shape or tif.pages[0].dtype != np.dtype('uint8'):
                                raise ValueError(f"{key}: domain TIFF must be uint8 YX on the input canvas")
                    if header_cache is not None:
                        header_cache[row.image_id] = {"key": cache_key, "validated": True}
    with resource_lock(study_root, shared=True):
        if input_binding(study_root, data_root, scope_ids) != binding:
            raise ValueError("Reviewed input revision changed while checking headers")
        assert_reviewed_binding(study_root, data_root=data_root, binding=binding)
    return ReviewedStudyInputs(study_root, data_root, selected, panel, samples, fields, binding)
