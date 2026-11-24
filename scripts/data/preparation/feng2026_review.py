"""Independently review the indexed Feng IMC sources and each published input batch.

The index and raw IMC files support source preflight before the author publishes
the four input tables and review CSV. Evidence stays in the study review cache;
only aggregate progress goes to stdout. --publish binds the verified scope.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import random
import sys
from tempfile import NamedTemporaryFile
import time
from typing import Any, Iterator
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from readimc import MCDFile, TXTFile
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.inputs import (
    PANEL_INPUT_FIELDS, assert_reviewed_acceptance, file_revision, input_binding,
    validate_panel_mapping,
)
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease, resolve_lock_root
from cellweft.data.preparation.study_input import TEMPLATES, panel_from_s1, validate_image_metadata

STUDY = "Feng2026_IMC_BladderCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE = Path("/mnt/public/lyx/horizontalProject/IMC_20240725_63")
S1_RELATIVE = Path("processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv")
INDEX_RELATIVE = Path("cache/IMC_20240725_63_intake/indexed_sources.csv")
PANEL_CONTRACT = Path(f"cache/{STUDY}/panel_contract.json")
PLANNED_INPUTS = Path(f"cache/{STUDY}/planned_images.parquet")
IMAGE_IDENTITY = Path(f"cache/{STUDY}/image_identity.csv")
DIRECT_MAP = {"Sm149": ("YAP1", "YAP1", "P46937"), "Yb174": ("ESR2", "ESR2", "Q92731")}
SEED = 20261124
BLOCK_SIZE = 64
REVIEWER = "feng2026_independent_d01"
INPUT_TABLES = ("images.parquet", "panel_mapping.csv", "sample_metadata.csv", "field_mapping.csv")
TECHNICAL_CHECKS = ("S01", "I01", "I02", "I03", "I04", "P01", "P02", "P03", "I05")
RESTRICTIONS = ["pretraining", "patient_level_split", "clinical_analysis", "patient_count_inference",
                "paired_analysis", "cross_image_biological_sample_merge", "biological_region_analysis"]


class ReviewIssue(ValueError):
    def __init__(self, check: str, reason: str):
        super().__init__(reason)
        self.check = check


def require(condition: bool, check: str, reason: str) -> None:
    if not condition:
        raise ReviewIssue(check, reason)


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def progress(stage: str, **counts: Any) -> None:
    print(json.dumps({"stage": stage, **counts}, ensure_ascii=False, sort_keys=True), flush=True)


def atomic_save(path: Path, content: str) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if path.exists():
                os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def present(value: Any) -> bool:
    return bool(pd.notna(value) and value != "")


def local_path(root: Path, value: str) -> Path:
    path = root / value
    require(path.resolve().is_relative_to(root.resolve()), "I01", "path_outside_declared_root")
    return path


@contextmanager
def study_read_lease(study_root: Path, data_root: Path) -> Iterator[None]:
    """Reuse the published lock inode for read-only snapshots in this sandbox."""
    io_root = resolve_lock_root(study_root, data_root=data_root)
    path = study_root / ".publication.lock"
    if io_root is not None and os.access(io_root if io_root.exists() else data_root, os.W_OK) and os.access(study_root, os.W_OK):
        with formal_read_lease(study_root, data_root=data_root), resource_lock(study_root, shared=True):
            yield
    else:
        with path.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            yield


def source_inventory(index_file: Path, source_root: Path) -> tuple[dict, dict, list[dict]]:
    """Reconcile every indexed acquisition with original headers and inventory non-indexed MCDs."""
    index_revision = file_revision(index_file)
    index = csv(index_file)
    require(not index.duplicated(["raw_relative_to_study", "acquisition_id"]).any()
            and index.source_image.is_unique, "S01", "duplicate_indexed_source")
    inventory, revisions, extra = {}, {"index": index_revision}, []
    for relative, group in index.groupby("raw_relative_to_study", sort=True):
        path = local_path(source_root, relative).resolve()
        revision = file_revision(path)
        revisions[str(path)] = revision
        kind = path.suffix.lower().lstrip(".")
        require(kind in {"mcd", "txt"}, "S01", "unsupported_indexed_source_format")
        with ExitStack() as stack:
            if kind == "mcd":
                container = stack.enter_context(MCDFile(path))
                acquisitions = {}
                indexed_ids = set(group.acquisition_id)
                for slide in container.slides:
                    for acquisition in slide.acquisitions:
                        key = str(acquisition.id)
                        require(key not in acquisitions, "S01", "ambiguous_mcd_acquisition")
                        acquisitions[key] = acquisition
                for key, acquisition in acquisitions.items():
                    meta = acquisition.metadata
                    if key not in indexed_ids and int(meta["DataEndOffset"]) > int(meta["DataStartOffset"]):
                        extra.append({"source": str(path), "acquisition": key,
                                      "panel": (tuple(acquisition.channel_labels), tuple(acquisition.channel_names))})
            else:
                acquisitions = {"": stack.enter_context(TXTFile(path))}
                coordinates = pd.read_csv(path, sep="\t", usecols=["X", "Y"], dtype=np.float64)
                require(np.isfinite(coordinates).all().all() and (coordinates >= 0).all().all()
                        and (coordinates % 1 == 0).all().all() and not coordinates.duplicated().any(),
                        "S01", "invalid_txt_coordinates")
                txt_shape = tuple(int(coordinates[axis].max()) + 1 for axis in ("Y", "X"))
            for entry in group.itertuples(index=False):
                identifier = entry.acquisition_id
                require(identifier in acquisitions, "S01", "indexed_acquisition_missing")
                native = acquisitions[identifier]
                scale = description = None
                if kind == "mcd":
                    meta = native.metadata
                    shape = (int(meta["MaxY"]), int(meta["MaxX"]))
                    scale = (native.pixel_size_x_um, native.pixel_size_y_um)
                    description = native.description
                    start, end = int(meta["DataStartOffset"]), int(meta["DataEndOffset"])
                    stride = 4 * (native.num_channels + 3)
                    require(int(meta["ValueBytes"]) == 4 and 0 <= start < end <= revision["size"]
                            and (end - start) % stride == 0, "S01", "invalid_mcd_record_span")
                    measured = (end - start) // stride
                else:
                    shape, measured = txt_shape, len(coordinates)
                require(shape == (int(entry.height), int(entry.width)) and 0 < measured <= np.prod(shape),
                        "S01", "indexed_source_shape_or_span_mismatch")
                names = tuple(native.channel_names)
                inventory[path, identifier] = dict(
                    kind=kind, shape=shape, labels=tuple(native.channel_labels), scale=scale, description=description,
                    names=names,
                    measured=measured, indexed_image=entry.source_image, revision=revision,
                    source_image_id=f"{relative}#Acquisition:{identifier}" if identifier else relative,
                    legacy_geometry=entry.headers_match.lower() == "true",
                )
        require(file_revision(path) == revision, "S01", "source_changed_during_inventory")
    panels = {(item["labels"], item["names"]) for item in inventory.values()}
    for entry in extra:
        entry["same_native_panel"] = entry.pop("panel") in panels
    require(file_revision(index_file) == index_revision, "S01", "index_changed_during_inventory")
    return inventory, revisions, extra


def published_paths(row: Any, data_root: Path, study_root: Path, *, header_ready: bool) -> tuple[Path, Path | None]:
    image_path = local_path(data_root, row.input_image_path)
    require(image_path.resolve().is_relative_to(study_root.resolve()) and (not header_ready or image_path.is_file()),
            "I01", "published_input_missing_or_outside_study")
    valid_path = local_path(data_root, row.valid_pixels_path) if present(row.valid_pixels_path) else None
    require(valid_path is None or (valid_path.resolve().is_relative_to(study_root.resolve())
                                   and (not header_ready or valid_path.is_file())),
            "I03", "published_domain_missing_or_outside_study")
    return image_path, valid_path


def check_image(row: Any, source: dict, panel: pd.DataFrame,
                image_path: Path, valid_path: Path | None, *, header_ready: bool) -> tuple:
    """Check acquisition geometry, channel positions, and image/domain TIFF headers."""
    height, width = source["shape"]
    require(present(row.source_axes) and row.source_axes == "CYX" and row.source_dtype == "float32"
            and not present(row.source_z_index) and not present(row.source_t_index), "I03", "native_axes_or_dtype")
    require((row.source_x0_px, row.source_y0_px, row.source_x1_px, row.source_y1_px) == (0, 0, width, height),
            "I03", "source_window_not_full_acquisition")
    require(row.split_method in {"already_independent", "source_acquisition"}
            and not present(getattr(row, "sample_region_path", None)) and present(row.split_evidence),
            "I04", "physical_roi_evidence_or_split_requires_review")
    if present(row.region_annotation):
        require(source["description"] is not None and row.region_annotation == source["description"],
                "I04", "non_native_region_annotation_requires_source_review")
    require(Path(row.image_id).name == row.image_id and row.image_id not in {".", ".."},
            "I01", "unsafe_image_identifier")
    shape = (row.height_px, row.width_px)
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    pads = (row.pad_left_px, row.pad_top_px, row.pad_right_px, row.pad_bottom_px)
    policy = getattr(row, "canvas_policy", None)
    if present(policy) and policy == "user_approved_native_rectangle":
        require(shape == (height, width) and pads == (0, 0, 0, 0), "I03", "invalid_native_rectangle")
    else:
        require(shape == (side, side) and pads == (left, top, side - width - left, side - height - top),
                "I03", "canvas_or_centered_padding_mismatch")
    scale = (row.pixel_size_x_um, row.pixel_size_y_um)
    scale_known = all(present(value) and np.isfinite(value) and value > 0 for value in scale)
    if source["kind"] == "mcd":
        require(scale_known and None not in source["scale"] and np.allclose(scale, source["scale"], rtol=1e-6, atol=0),
                "I03", "mcd_pixel_scale_mismatch")
    else:
        require(not any(present(value) for value in scale), "I03", "txt_scale_has_no_native_evidence")
    require(tuple(panel.source_name) == source["labels"] and row.source_channel_count == len(panel),
            "P01", "native_channel_sequence_mismatch")
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    require(row.channel_count == len(kept), "P03", "kept_channel_count_mismatch")
    irregular = source["measured"] != height * width
    require(not irregular or valid_path is not None, "I03", "missing_measured_domain")
    stratum = (row.source_format, row.source_axes, row.panel_id, row.split_method, row.source_dtype)
    if not header_ready:
        return stratum
    with tifffile.TiffFile(image_path) as tif:
        require(len(tif.pages) == len(kept) and all(page.shape == shape and page.dtype == np.dtype("float32")
                                                  for page in tif.pages), "P03", "input_tiff_page_mismatch")
        require(tif.ome_metadata is not None, "P03", "missing_ome_channel_metadata")
        require(tif.series[0].axes in {"CYX", "YX"} and (len(kept) == 1 or tif.series[0].axes == "CYX"),
                "P03", "ome_axes_mismatch")
        pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        require(pixels is not None and [item.get("Name") for item in pixels.findall("{*}Channel")] == kept.marker_id.tolist(),
                "P03", "ome_marker_order_mismatch")
        physical = tuple(pixels.get("PhysicalSize" + axis) for axis in ("X", "Y"))
        units = tuple(pixels.get("PhysicalSize" + axis + "Unit", "µm") for axis in ("X", "Y"))
        if scale_known:
            require(all(value is not None for value in physical) and units == ("µm", "µm")
                    and np.allclose(tuple(map(float, physical)), scale, rtol=1e-6, atol=0),
                    "I03", "ome_pixel_scale_mismatch")
        else:
            require(physical == (None, None), "I03", "unknown_scale_filled_in_ome")
    if valid_path is not None:
        with tifffile.TiffFile(valid_path) as tif:
            require(len(tif.pages) == 1 and tif.pages[0].shape == shape and tif.pages[0].dtype == np.dtype("uint8"),
                    "I03", "invalid_domain_tiff_header")
    return stratum


def check_panel(panel: pd.DataFrame, s1: Path) -> pd.DataFrame:
    panel = panel.copy()
    panel["source_channel_index"] = pd.to_numeric(panel.source_channel_index, errors="raise").astype("Int64")
    count = int(panel.keep_channel.str.lower().eq("true").sum())
    panel = validate_panel_mapping(panel, input_channel_count=count)
    downstream = set(panel.loc[panel.role.eq("downstream"), "source_channel_index"])
    require(bool(downstream), "P02", "no_retained_downstream_channel")
    require(panel.loc[~panel.keep_channel, "exclusion_reason"].str.strip().ne("").all(),
            "P02", "dropped_channel_without_evidence")
    keys = panel.set_index("source_name").s1_original_marker
    require(keys.index.is_unique and keys.ne("").all(), "P01", "incomplete_s1_channel_keys")
    require(keys.index.tolist() == keys.tolist(), "P01", "study_requires_exact_native_s1_keys")
    expected = panel_from_s1(STUDY, panel.panel_id.iloc[0], panel.source_name.tolist(), s1)
    semantic = [key for key in PANEL_INPUT_FIELDS if key != "output_channel_index"]
    require(panel[semantic].astype(str).equals(expected[semantic].astype(str)),
            "P02", "s1_identity_or_admission_mismatch")
    return panel


def study_s1(canonical: Path, contract: Path, cache: Path, inventory: dict) -> tuple[Path, list[Path], int]:
    """Resolve the canonical or published study-local S1 mapping and verify its lineage."""
    canonical_revision = file_revision(canonical)
    full = csv(canonical)
    official = int(full.Study.eq(STUDY).sum())
    if official or not contract.is_file():
        return canonical, [canonical], official
    contract_revision = file_revision(contract)
    mapping = json.loads(contract.read_text())
    require(mapping["study"] == STUDY and mapping["canonical_s1"] == canonical_revision,
            "P01", "study_s1_contract_version_mismatch")
    private = pd.DataFrame(mapping["study_s1_rows"], dtype=str).fillna("")
    native_sequences = {item["labels"] for item in inventory.values()}
    require(len(native_sequences) == 1 and set(private.Study) == {STUDY}
            and tuple(private["Original marker"]) in native_sequences,
            "P01", "study_s1_native_order_mismatch")
    columns = ["Unified marker", "MAP-KG mapping", "UniProt accession"]
    known = set(full[columns].itertuples(index=False, name=None))
    require(all(tuple(item["names"]) == tuple(channel["isotope"] for channel in mapping["channels"])
                for item in inventory.values()), "P01", "study_s1_isotope_order_mismatch")
    for row, channel in zip(private[columns].itertuples(index=False, name=None), mapping["channels"], strict=True):
        if row[1] or row[2]:
            require(row in known or DIRECT_MAP.get(channel["isotope"]) == row,
                    "P01", "study_s1_map_link_differs_from_canonical")
    require(file_revision(canonical) == canonical_revision and file_revision(contract) == contract_revision,
            "P01", "panel_evidence_changed_while_reading")
    path = cache / "study_s1_review.csv"
    atomic_save(path, private.to_csv(index=False))
    return path, [canonical, contract], len(private)


def choose_samples(records: dict, previous: set[str], *, baseline_required: bool = False) -> list[str]:
    ordered = sorted(records, key=lambda key: (np.prod(records[key]["source"]["shape"]), key))
    chosen = [key for key in ordered if key in previous]
    baseline = []
    if baseline_required or len(chosen) < min(3, len(ordered)):
        baseline = list(dict.fromkeys((ordered[len(ordered) // 2], ordered[-1])))
        rest = [key for key in ordered if key not in baseline]
        random.Random(SEED).shuffle(rest)
        baseline = (baseline + rest)[:3]
        chosen.extend(key for key in baseline if key not in chosen)
    covered = set()
    for key in chosen + ordered:
        record = records[key]
        group = record["stratum"], record["features"]
        if key in chosen or group not in covered:
            covered.add(group)
            if key not in chosen:
                chosen.append(key)
    return chosen


def inspect_pixels(record: dict) -> dict:
    """Match retained channels and measured domain to original X/Y coordinates."""
    row, source, (raw_path, identifier) = record["row"], record["source"], record["key"]
    kept = record["panel"].loc[lambda panel: panel.keep_channel].sort_values("output_channel_index")
    indices = kept.source_channel_index.astype(int).tolist()
    signal_channel = np.flatnonzero(kept.role.eq("downstream").to_numpy()).tolist()
    height, width = source["shape"]
    left, top = int(row.pad_left_px), int(row.pad_top_px)
    with ExitStack() as stack:
        tif = stack.enter_context(tifffile.TiffFile(record["input"]))
        domain = stack.enter_context(tifffile.TiffFile(record["valid"])) if record["valid"] else None
        if source["kind"] == "mcd":
            reader = stack.enter_context(open_source(SourceSpec(
                row.image_id, row.panel_id, "mcd", raw_path, tuple(range(len(source["labels"]))),
                acquisition_id=int(identifier))))
            read = lambda window: reader.read_window(indices, window)
            measured = reader.valid_window((0, 0, width, height))
        else:
            table = pd.read_csv(raw_path, sep="\t", dtype=np.float32)
            x_coord, y_coord = table.X.to_numpy(dtype=int), table.Y.to_numpy(dtype=int)
            measured = np.zeros((height, width), dtype=bool)
            measured[y_coord, x_coord] = True
            raster = np.zeros((len(indices), height, width), dtype=np.float32)
            raster[:, y_coord, x_coord] = table.iloc[:, [6 + index for index in indices]].to_numpy().T
            read = lambda window: raster[:, window[1]:window[3], window[0]:window[2]].copy()
        require(int(measured.sum()) == source["measured"], "I05", "source_coordinate_count_mismatch")
        expected_domain = np.zeros((row.height_px, row.width_px), dtype=bool)
        expected_domain[top:top+height, left:left+width] = measured
        if domain is not None:
            declared = domain.pages[0].asarray()
            require(np.array_equal(declared, expected_domain), "I05", "measured_domain_mismatch")
        for output_window in windows(expected_domain.shape, 512):
            out_x0, out_y0, out_x1, out_y1 = output_window
            invalid = ~expected_domain[out_y0:out_y1, out_x0:out_x1]
            if invalid.any():
                for output in range(len(indices)):
                    require(not read_tiff_window(tif, output, output_window)[invalid].any(),
                            "I05", "nonzero_pixels_outside_measurement")
        blocks = list(windows((height, width), BLOCK_SIZE))
        inner = [window for window in blocks if window[0] > 0 and window[1] > 0
                 and window[2] < width and window[3] < height]
        inner.sort(key=lambda window: abs(window[0] + window[2] - width) + abs(window[1] + window[3] - height))
        interior = None
        for window in inner or blocks:
            values = read(window)
            if np.any(values[signal_channel]):
                interior = window
                break
        require(interior is not None, "I05", "no_signal_in_sampled_interior")
        edge = blocks[-1] if blocks[-1] != interior else blocks[0]
        evidence = []
        for window in dict.fromkeys((interior, edge)):
            x0, y0, x1, y1 = window
            target = (x0 + left, y0 + top, x1 + left, y1 + top)
            expected = read(window)
            expected[:, ~measured[y0:y1, x0:x1]] = 0
            for output, expected_plane in enumerate(expected):
                observed = read_tiff_window(tif, output, target)
                require(expected_plane.dtype == observed.dtype and np.isfinite(expected_plane).all()
                        and np.array_equal(expected_plane, observed), "I05", "native_pixel_or_dtype_mismatch")
            evidence.append(dict(source_window=window, output_window=target))
    return dict(source_channel_indices=indices, output_channel_indices=list(range(len(indices))),
                windows=evidence, measured_pixels=source["measured"])


def review_batch(data_root: Path, study_root: Path, source_root: Path, inventory: dict, s1: Path,
                 previous: dict, image_table: Path, *, staged: bool = False,
                 baseline_required: bool = False) -> tuple[dict, dict, dict, dict]:
    samples = csv(study_root / "sample_metadata.csv")
    fields = csv(study_root / "field_mapping.csv")
    panels = csv(study_root / "panel_mapping.csv")
    images = validate_image_metadata(pd.read_parquet(image_table), samples, panels, fields)
    require(set(images.study) == {STUDY} and set(samples.study).issubset({STUDY})
            and set(fields.study).issubset({STUDY}), "I02", "foreign_study_in_input_tables")
    require(samples.sample_id.str.strip().ne("").all(), "I02", "empty_sample_primary_key")
    require(images.source_image_id.is_unique and images.input_image_path.is_unique,
            "I01", "duplicate_source_identity_or_input_path")
    errors, records, checked_panels = {}, {}, {}
    header_pending, geometry_pass = set(), set()
    indexed_paths = {path for path, _ in inventory}
    for panel_id, rows in panels.groupby("panel_id", sort=True):
        if panel_id in set(images.panel_id):
            try:
                checked_panels[panel_id] = check_panel(rows, s1)
            except ReviewIssue as error:
                checked_panels[panel_id] = error
            except (ValueError, KeyError):
                checked_panels[panel_id] = ReviewIssue("P01", "invalid_or_unmatched_s1_panel")
    selected_keys = defaultdict(list)
    for row in images.itertuples(index=False):
        try:
            require(len(row.source_files) == 1, "I01", "expected_single_original_source")
            declared = Path(row.source_files[0])
            path = (data_root / declared).resolve()
            if not declared.is_absolute() and path not in indexed_paths:
                path = (source_root / declared).resolve()
            acquisition = str(row.source_acquisition_id) if present(row.source_acquisition_id) else ""
            key = (path, acquisition)
            require(key in inventory, "I01", "original_source_or_acquisition_not_indexed")
            selected_keys[key].append(row.image_id)
            source = inventory[key]
            require(row.source_format == source["kind"] and row.source_image_id == source["source_image_id"],
                    "I01", "source_format_or_identity_mismatch")
            if source["kind"] == "mcd" and present(row.roi_id):
                require(row.roi_id == acquisition, "I01", "mcd_roi_identifier_mismatch")
            stat = source["revision"]
            require(row.source_revision == f"size_bytes={stat['size']};mtime_ns={stat['mtime_ns']}",
                    "I01", "original_source_revision_mismatch")
            panel = checked_panels[row.panel_id]
            if isinstance(panel, ReviewIssue):
                raise panel
            header_ready = not staged or local_path(data_root, row.input_image_path).is_file()
            if not header_ready:
                header_pending.add(row.image_id)
            image_path, valid_path = published_paths(row, data_root, study_root, header_ready=header_ready)
            input_revision = file_revision(image_path) if header_ready else None
            valid_revision = file_revision(valid_path) if header_ready and valid_path else None
            stratum = check_image(row, source, panel, image_path, valid_path, header_ready=header_ready)
            geometry_pass.add(row.image_id)
            if not header_ready:
                continue
            height, width = source["shape"]
            features = (height < width, height > width, source["measured"] != height * width,
                        not source["legacy_geometry"], valid_path is not None)
            revision = dict(source=source["revision"], acquisition=acquisition,
                            input=input_revision, domain=valid_revision,
                            pads=[int(row.pad_left_px), int(row.pad_top_px), int(row.pad_right_px), int(row.pad_bottom_px)],
                            channel_map=panel[PANEL_INPUT_FIELDS].to_json(orient="records"))
            records[row.image_id] = dict(row=row, source=source, key=key, panel=panel, input=image_path,
                                         valid=valid_path, stratum=stratum, features=features, revision=revision)
        except ReviewIssue as error:
            errors[row.image_id] = dict(check=error.check, reason=str(error))
        except (ValueError, OSError, ElementTree.ParseError):
            errors[row.image_id] = dict(check="P03", reason="invalid_input_file_or_header")
    for image_ids in selected_keys.values():
        if len(image_ids) > 1:
            for image_id in image_ids:
                records.pop(image_id, None)
                geometry_pass.discard(image_id)
                errors[image_id] = dict(check="I01", reason="repeated_original_acquisition")
    evidence, held = {}, set()
    explicit = set(previous) | {key for key, record in records.items() if record["source"]["kind"] == "txt"}
    sampled = choose_samples(records, explicit, baseline_required=baseline_required) if records else []
    for image_id in sampled:
        record = records[image_id]
        try:
            evidence[image_id] = (previous[image_id] if previous.get(image_id, {}).get("revision") == record["revision"]
                                  else dict(revision=record["revision"], **inspect_pixels(record)))
        except (ValueError, OSError) as error:
            errors[image_id] = dict(check="I05", reason=str(error) if isinstance(error, ReviewIssue) else "source_or_output_pixel_read_failed")
            held.add(record["stratum"])
    for stratum in sorted(held):
        extra = [key for key in sorted(records) if records[key]["stratum"] == stratum and key not in sampled][:3]
        for image_id in extra:
            record = records[image_id]
            try:
                evidence[image_id] = dict(revision=record["revision"], **inspect_pixels(record))
            except (ValueError, OSError) as error:
                errors[image_id] = dict(check="I05", reason=str(error) if isinstance(error, ReviewIssue) else "source_or_output_pixel_read_failed")
    accepted = {key: record for key, record in records.items() if record["stratum"] not in held}
    for key in records.keys() - accepted.keys():
        errors.setdefault(key, dict(check="I05", reason="stratum_held_for_pixel_anomaly"))
    declarations = fields[fields.target_table.eq("sample_metadata")]
    dictionary = csv(TEMPLATES / "data_dictionary.csv")
    sample_definitions = dictionary[dictionary.table_name.eq("sample_metadata")]
    vocab = csv(TEMPLATES / "value_sets.csv")
    clinical_gaps = Counter()
    for definition in sample_definitions.itertuples(index=False):
        key = definition.field_name
        if key not in samples:
            continue
        supplied = samples.loc[samples[key].ne(""), key]
        if supplied.empty:
            continue
        if definition.data_type in {"integer", "number"}:
            values = pd.to_numeric(supplied, errors="coerce")
            valid = np.isfinite(values)
            if definition.data_type == "integer":
                valid &= values.mod(1).eq(0)
            clinical_gaps["invalid_number"] += int((~valid).sum())
        elif definition.data_type == "date":
            clinical_gaps["invalid_date"] += int(pd.to_datetime(supplied, format="%Y-%m-%d", errors="coerce").isna().sum())
        if definition.value_set:
            allowed = vocab.loc[vocab.value_set.eq(definition.value_set), "allowed_value"]
            clinical_gaps["outside_vocabulary"] += int((~supplied.isin(allowed)).sum())
        if key not in {"study", "sample_id", "metadata_notes"}:
            clinical_gaps["unmapped_fields"] += int(not declarations.target_field.eq(key).any())
    for endpoint in ("last_followup", "os", "pfs", "dfs"):
        fields_required = [f"{endpoint}_{part}" for part in ("time", "time_unit", "origin")]
        if set(fields_required).issubset(samples):
            timing = samples[fields_required].ne("")
            clinical_gaps["incomplete_endpoint"] += int((timing.any(axis=1) & ~timing.all(axis=1)).sum())
    metadata = dict(sample_rows=len(samples), images_linked=int(images.sample_id.notna().sum()),
                    images_with_patient_bridge=int(images.sample_id.isin(samples.loc[samples.patient_key.ne(""), "sample_id"]).sum()),
                    clinical_fields_with_values=int(samples.drop(columns=["study", "sample_id", "metadata_notes", "patient_key"], errors="ignore").ne("").any().sum()),
                    sample_mapping_rules=len(declarations), clinical_structure_gaps=dict(+clinical_gaps),
                    unresolved_sample_patient_source_rules=True)
    identity_path = data_root / IMAGE_IDENTITY
    if identity_path.is_file():
        identity = csv(identity_path)
        require(identity.image_id.is_unique and set(identity.study) == {STUDY}, "I02", "invalid_author_identity_index")
        identity = identity.set_index("image_id")
        require(set(images.image_id).issubset(identity.index), "I02", "image_missing_from_author_identity_index")
        for source_key, identifiers in selected_keys.items():
            require(identity.loc[identifiers, "source_image"].eq(inventory[source_key]["indexed_image"]).all(),
                    "I01", "author_identity_index_source_mismatch")
        metadata.update(author_identity_rows=len(identity),
                        author_roi_links_reported_confirmed=int(identity.patient_linkage_status.eq("confirmed").sum()),
                        author_roi_links_reported_unresolved=int(identity.patient_linkage_status.eq("unresolved").sum()),
                        author_source_rules_declared=int(fields.target_table.eq("image_identity").sum()))
    return accepted, errors, evidence, dict(metadata=metadata, images=len(images) - len(header_pending),
                                            geometry_pass=len(geometry_pass), groups_held=len(held),
                                            indexed_linked=len(selected_keys), records=records)


def raw_samples(inventory: dict) -> list[dict]:
    """Exercise raw reads now, before TIFF publication makes comparison possible."""
    candidate = {str(number): dict(source=source, stratum=(source["kind"],),
                                  features=(source["shape"][0] != source["shape"][1],
                                            source["measured"] != np.prod(source["shape"]),
                                            not source["legacy_geometry"]))
                 for number, source in enumerate(inventory.values())}
    keys = list(inventory)
    chosen = set(choose_samples(candidate, set()))
    chosen.update(str(number) for number, (path, _) in enumerate(keys) if path.suffix.lower() == ".txt")
    results = []
    for index in sorted(map(int, chosen)):
        path, identifier = keys[index]
        source = inventory[path, identifier]
        height, width = source["shape"]
        sample_windows = [(0, 0, min(width, BLOCK_SIZE), min(height, BLOCK_SIZE)),
                          (width // 2, height // 2, min(width, width // 2 + BLOCK_SIZE),
                           min(height, height // 2 + BLOCK_SIZE))]
        if source["kind"] == "mcd":
            channels = list(range(len(source["labels"])))
            with open_source(SourceSpec("preflight", "preflight", "mcd", path, tuple(channels),
                                        acquisition_id=int(identifier))) as reader:
                measured = reader.valid_window((0, 0, width, height))
                blocks = [reader.read_window(channels, window) for window in sample_windows]
            require(measured.sum() == source["measured"], "I05", "mcd_coordinate_count_mismatch")
        else:
            with TXTFile(path) as reader:
                raw = reader.read_acquisition()
            blocks = [raw[:, y0:y1, x0:x1] for x0, y0, x1, y1 in sample_windows]
        require(all(block.dtype == np.float32 and np.isfinite(block).all() for block in blocks),
                "I05", "nonfinite_or_unexpected_source_dtype")
        results.append(dict(source=source["revision"], acquisition=identifier, windows=sample_windows, dtype="float32"))
    return results


def publication_rows(review: pd.DataFrame, preparer: str, binding: dict | None, qc: dict,
                     evidence_path: Path, revisions: dict, errors: dict, metadata: dict) -> pd.DataFrame:
    """Keep historical rows and supersede only this reviewer's decisions and author handoff."""
    rows = []
    restrictions = RESTRICTIONS
    training_authorization = None
    if binding:
        plan_path = Path(binding["field_mapping"]["path"]).parent / "training_split.json"
        if plan_path.is_file():
            plan = json.loads(plan_path.read_bytes())
            identity = csv(plan_path.parents[2] / IMAGE_IDENTITY).set_index("image_id")
            retained = binding["images"]
            require(plan["authority"] == "user_instruction_2026-11-24"
                    and plan["retained_image_ids"] == binding["image_ids"]
                    and len(retained) == 515
                    and all(present(row.get("source_patient_key"))
                            and identity.loc[row["image_id"], "patient_linkage_status"] == "confirmed"
                            and identity.loc[row["image_id"], "patient_key"] == row["source_patient_key"]
                            for row in retained),
                    "D01", "training_scope_or_confirmed_patient_connection_mismatch")
            restrictions = plan["remaining_restrictions"]
            metadata = dict(metadata, retained_patient_linked_images=len(retained),
                            retained_confirmed_patients=len({row["source_patient_key"] for row in retained}),
                            excluded_patient_unresolved_images=len(plan["excluded_image_ids"]))
            training_authorization = dict(authority=plan["authority"], plan=file_revision(plan_path),
                                          grouping="source_patient_key", counts=plan["counts"],
                                          fit_splits=["development_train"],
                                          validation_splits=["development_validation"],
                                          sampling_status="pending")

    def add(scope: str, record_id: str, check: str, result: str, detail: dict) -> None:
        rows.append(dict(study=STUDY, preparer=preparer, reviewer=REVIEWER, scope=scope,
                         record_id=record_id, check_id=check, result=result,
                         source_evidence=revisions["index"]["path"], output_evidence=str(evidence_path),
                         details=json.dumps(detail, ensure_ascii=False, sort_keys=True)))

    for check in TECHNICAL_CHECKS:
        affected = {key: error["reason"] for key, error in errors.items() if error["check"] == check}
        add("study", REVIEWER, check, "fail" if affected else ("pass" if binding else "unresolved"),
            dict(image_ids=sorted(affected) if affected else (binding or {}).get("image_ids", []),
                 problems=affected, study_qc=qc if check == "I05" else None))
    for check in ("M01", "M02"):
        add("sample", REVIEWER, check, "unresolved", dict(metadata=metadata, use_restrictions=restrictions,
            explanation="复核未确认本地或公开来源的逐条身份/临床连接及转换；保留实际覆盖及结构差异。"))
    latest = review.drop_duplicates(["scope", "record_id", "check_id"], keep="last")
    own_d01 = latest[latest.study.eq(STUDY) & latest.reviewer.eq(REVIEWER) & latest.check_id.eq("D01")]
    if binding:
        handoff = latest[(latest.study == STUDY) & (latest.preparer == preparer) & (latest.check_id == "D01")
                         & (latest.scope == "study") & (latest.result == "unresolved")
                         & latest.reviewer.isin(["", preparer])]
        require(len(handoff) <= 1, "D01", "multiple_author_handoffs_require_reconciliation")
        require(len(own_d01) <= 1, "D01", "multiple_prior_reviewer_scopes_require_reconciliation")
        record_id = handoff.record_id.iloc[0] if len(handoff) else own_d01.record_id.iloc[0] if len(own_d01) else REVIEWER
        preparer_failures = latest[latest.study.eq(STUDY) & latest.preparer.eq(preparer)
                                   & latest.reviewer.eq("") & latest.scope.eq("image")
                                   & latest.check_id.isin(TECHNICAL_CHECKS)
                                   & latest.result.isin(["fail", "unresolved"])]
        for row in preparer_failures.itertuples(index=False):
            if row.record_id not in binding["image_ids"] and not row.details.lstrip().startswith("{"):
                add(row.scope, row.record_id, row.check_id, row.result,
                    dict(image_ids=[row.record_id], preparer_observation=row.details))
        add("study", record_id, "D01", "pass", dict(input_binding=binding, study_qc=qc, metadata=metadata,
              technical_only=True, use_restrictions=restrictions, training_authorization=training_authorization,
              physical_scale_unavailable_image_ids=[row["image_id"] for row in binding["images"]
                                                    if row["source_format"] == "txt"],
              explanation="逐图结构与文件头已复核；独立源像素范围见 study_qc。仅批准明确绑定的技术处理范围。"))
    elif len(own_d01):
        for row in own_d01.itertuples(index=False):
            add(row.scope, row.record_id, "D01", "unresolved", dict(study_qc=qc, image_ids=qc["anomaly_scope"]))
    return pd.concat([review, pd.DataFrame(rows)], ignore_index=True)


def tables_ready(study_root: Path) -> bool:
    return all((study_root / name).is_file() for name in (*INPUT_TABLES, "review_results.csv"))


def acceptance_rows(study_root: Path) -> pd.DataFrame:
    reviews = csv(study_root / "review_results.csv")
    require(list(reviews.columns) == list(csv(TEMPLATES / "review_results.csv").columns),
            "D01", "invalid_review_csv_schema")
    return reviews


def asset_snapshot(study_root: Path, panel_sources: list[Path], image_table: Path) -> dict:
    paths = [image_table, *(study_root / name for name in INPUT_TABLES[1:]), *panel_sources]
    identity = study_root.parents[1] / IMAGE_IDENTITY
    if identity.is_file():
        paths.append(identity)
    return {str(path): file_revision(path) for path in paths}


def bindings_match_records(binding: dict, records: dict) -> bool:
    expected_assets = [revision for image_id in sorted(records)
                       for field in ("input", "domain") if (revision := records[image_id]["revision"][field]) is not None]
    return binding["assets"] == expected_assets


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=NAS)
parser.add_argument("--source-root", type=Path, default=SOURCE)
parser.add_argument("--cache", type=Path, help="Study review evidence directory")
parser.add_argument("--preparer", help="Initial author's task identity if the review CSV has more than one")
parser.add_argument("--publish", action="store_true", help="Append current verified review and bound D01 under the study lock")
parser.add_argument("--source-only", action="store_true", help="Check original sources even if TIFF delivery is pending")
parser.add_argument("--staged", action="store_true", help="Review available TIFFs against the author staging index without D01")
args = parser.parse_args()
if args.staged and (args.publish or args.source_only):
    parser.error("--staged is a read-only review mode")
data_root, source_root = args.data_root.resolve(), args.source_root.resolve()
study_root = data_root / "processed" / STUDY
cache = args.cache or data_root / "cache" / f"{STUDY}_review"
canonical_s1 = data_root / S1_RELATIVE
evidence_path = cache.resolve() / "review_evidence.json"
status_path = cache.resolve() / "progress.json"

# Inventory the original sources and preserve only file-local evidence.
try:
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    with resource_lock(cache), (cache / "diagnostics.log").open("w", encoding="utf-8") as diagnostics:
        with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
            inventory, revisions, extra = source_inventory(data_root / INDEX_RELATIVE, source_root)
            previous = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
            original_checks = (previous["raw_samples"] if previous.get("source_revisions") == revisions
                               and previous.get("raw_samples") else raw_samples(inventory))
            s1, panel_sources, s1_count = study_s1(canonical_s1, data_root / PANEL_CONTRACT, cache, inventory)
            panel_path = study_root / "panel_mapping.csv"
            panel_preflight = [check_panel(rows, s1) for _, rows in csv(panel_path).groupby("panel_id", sort=True)] if s1_count and panel_path.is_file() else []
        native_panels = {item["labels"] for item in inventory.values()}
        extra_same_panel = sum(item["same_native_panel"] for item in extra)
        summary = dict(indexed=len(inventory), formats=dict(Counter(item["kind"] for item in inventory.values())),
                       raw_sampled=len(original_checks), native_panels=len(native_panels),
                       rectangles=sum(item["shape"][0] != item["shape"][1] for item in inventory.values()),
                       irregular=sum(item["measured"] != item["shape"][0] * item["shape"][1] for item in inventory.values()),
                       indexed_geometry_alerts=sum(not item["legacy_geometry"] for item in inventory.values()),
                       extra_positive_span_same_panel_count=extra_same_panel,
                       s1_rows=s1_count,
                       panel_rows_checked=sum(len(panel) for panel in panel_preflight),
                       retained_channels=sum(int(panel.keep_channel.sum()) for panel in panel_preflight),
                       published_images=0, signed=0)
        current = dict(previous, source_revisions=revisions, raw_samples=original_checks,
                       index_outside_scope=extra)
        atomic_save(evidence_path, json.dumps(current, ensure_ascii=False))
        progress("source_preflight", **summary)
        stage = "waiting_for_initial_publication"

        # Review a stable batch under the same NAS/study lease order as publication.
        image_table = data_root / PLANNED_INPUTS if args.staged else study_root / "images.parquet"
        complete = tables_ready(study_root) if not args.staged else (
            image_table.is_file() and all((study_root / name).is_file() for name in INPUT_TABLES[1:]))
        if not args.source_only and complete and summary["s1_rows"]:
            with study_read_lease(study_root, data_root):
                if not args.staged:
                    review = acceptance_rows(study_root)
                    author_ids = set(review.loc[review.study.eq(STUDY) & review.reviewer.ne(REVIEWER), "preparer"]) - {"", REVIEWER}
                    preparer = args.preparer or (next(iter(author_ids)) if len(author_ids) == 1 else None)
                    require(preparer in author_ids, "D01", "initial_author_identity_missing_or_ambiguous")
                snapshot = asset_snapshot(study_root, panel_sources, image_table)
                with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
                    accepted, errors, pixels, counts = review_batch(data_root, study_root, source_root, inventory, s1,
                                                                     previous.get("pixel_samples", {}), image_table,
                                                                     staged=args.staged,
                                                                     baseline_required=not args.staged and not previous.get("formal_baseline_reviewed"))
                    binding = input_binding(study_root, data_root, list(accepted)) if accepted and not args.staged else None
                require(not binding or bindings_match_records(binding, accepted), "D01", "input_asset_changed_during_review")
                if args.staged:
                    require(all(file_revision(Path(revision["path"])) == revision for record in accepted.values()
                                for field in ("input", "domain") if (revision := record["revision"][field]) is not None),
                            "D01", "staged_asset_changed_during_review")
                require(all(file_revision(Path(revision["path"])) == revision for revision in revisions.values()),
                        "D01", "source_changed_during_review")
                require(snapshot == asset_snapshot(study_root, panel_sources, image_table), "D01", "table_or_s1_changed_during_review")
            strata = defaultdict(list)
            for key, record in counts["records"].items():
                strata[record["stratum"]].append(key)
            sampled_ids = sorted(pixels)
            qc = dict(policy_id="study_stratified_v1", stage="study_input", selection_seed=SEED,
                      result="pass" if len(accepted) == counts["images"] else ("partial" if accepted else "fail"),
                      strata=[dict(group=group, image_count=len(ids), passed_count=len(set(ids) & accepted.keys()),
                                   sample_ids=sorted(set(ids).intersection(sampled_ids))) for group, ids in sorted(strata.items())],
                      sample_ids=sampled_ids, evidence_paths=[str(evidence_path)],
                      indexed_source_count=len(inventory), outside_index_positive_span_same_panel_count=extra_same_panel,
                      anomaly_scope=sorted(set(counts["records"]) - accepted.keys() | errors.keys()))
            pending = sorted(set(inventory) - {record["key"] for record in counts["records"].values()})
            current.update(pixel_samples=pixels, errors=errors, study_qc=qc, metadata=counts["metadata"],
                           indexed_pending=[(str(path), acquisition) for path, acquisition in pending],
                           table_versions=snapshot,
                           formal_baseline_reviewed=bool(accepted and not args.staged) or previous.get("formal_baseline_reviewed", False))
            atomic_save(evidence_path, json.dumps(current, ensure_ascii=False))
            summary.update(published_images=counts["images"] if not args.staged else 0,
                           staged_images=counts["images"] if args.staged else 0, source_linked=counts["indexed_linked"],
                           geometry_pass=counts["geometry_pass"], structure_pass=len(counts["records"]),
                           reviewed_formats=dict(Counter(record["source"]["kind"] for record in counts["records"].values())),
                           sampled_formats=dict(Counter(counts["records"][key]["source"]["kind"] for key in sampled_ids)),
                           sampled=len(sampled_ids), passed=len(accepted),
                           failed_images=sum(error["reason"] != "stratum_held_for_pixel_anomaly" for error in errors.values()),
                           strata_held=counts["groups_held"], indexed_pending=len(pending),
                           metadata=counts["metadata"])
            stage = "staged_batch_reviewed" if args.staged else "batch_reviewed"

            # Bind exactly this pass set after rechecking the review and evidence versions.
            if args.publish:
                release_evidence = cache.resolve() / f"review_evidence_{time.time_ns()}.json"
                qc["evidence_paths"] = [str(release_evidence)]
                updated = publication_rows(review, preparer, binding, qc, release_evidence, revisions, errors,
                                           counts["metadata"])
                if binding:
                    try:
                        assert_reviewed_acceptance(updated, binding)
                    except ValueError as error:
                        raise ReviewIssue("D01", "existing_review_blocker_requires_reconciliation") from error
                summary["candidate_binding_checked"] = len(binding["image_ids"]) if binding else 0
                progress(stage, **summary)
                require(os.access(study_root, os.W_OK), "D01", "formal_study_not_writable_in_current_session")
                with formal_publish_lease(study_root / "review_results.csv", data_root=data_root):
                    require(snapshot == asset_snapshot(study_root, panel_sources, image_table)
                            and all(file_revision(Path(revision["path"])) == revision for revision in revisions.values()),
                            "D01", "source_or_table_changed_before_publication")
                    require(not binding or input_binding(study_root, data_root, binding["image_ids"]) == binding,
                            "D01", "asset_changed_before_publication")
                    review_now = acceptance_rows(study_root)
                    require(review_now.equals(review), "D01", "review_changed_before_publication")
                    atomic_save(release_evidence, json.dumps(current, ensure_ascii=False))
                    atomic_save(study_root / "review_results.csv", updated.to_csv(index=False))
                    if binding:
                        assert_reviewed_acceptance(acceptance_rows(study_root), binding)
                summary["signed"] = len(accepted)
                stage = "review_published"
        elif not args.source_only and tables_ready(study_root):
            stage = "waiting_for_study_s1"
        elif args.source_only:
            stage = "source_preflight_complete"
        atomic_save(status_path, json.dumps(dict(stage=stage, **summary), ensure_ascii=False))
        progress(stage, **summary)
except Exception as error:
    # Library diagnostics can contain private source paths and values.
    detail = {"stage": "blocked", "error_type": type(error).__name__}
    if isinstance(error, ReviewIssue):
        detail.update(check=error.check, reason=str(error))
    if "summary" in globals():
        detail.update(summary)
    if cache.is_dir() and os.access(cache, os.W_OK):
        with (cache / "diagnostics.log").open("a", encoding="utf-8") as diagnostics:
            import traceback
            traceback.print_exc(file=diagnostics)
        atomic_save(status_path, json.dumps(detail, ensure_ascii=False))
    print(json.dumps(detail, ensure_ascii=False), file=sys.stderr, flush=True)
    sys.exit(1)
