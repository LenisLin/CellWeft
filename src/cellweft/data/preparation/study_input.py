"""S1 channel selection and native-intensity, independent study input delivery."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
from dataclasses import asdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile

from cellweft.data.contracts import ImageReader, Window
from cellweft.data.io.readers import windows
from cellweft.data.io.writers import write_image


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEMPLATES = PROJECT_ROOT / "docs/data/templates/study_input"


def study_input_staging_root(path: Path | None = None) -> Path:
    """Return the SSD-only staging root for a study-input operation.

    ``CELLWEFT_INPUT_STAGING_ROOT`` remains an explicit deployment/test
    override.  The production default is project-local SSD runtime storage;
    when a processed study path is available, its study name is included so
    checkpoints from separate adapters cannot share a directory.
    """
    override = os.environ.get("CELLWEFT_INPUT_STAGING_ROOT")
    if override:
        return Path(override)
    root = PROJECT_ROOT / ".cache/cellweft/runtime/study_input"
    if path is not None:
        parts = Path(path).resolve().parts
        try:
            processed_index = parts.index("processed")
        except ValueError:
            processed_index = -1
        if processed_index >= 0 and processed_index + 1 < len(parts):
            return root / parts[processed_index + 1]
    return root


def _row_key(row: pd.Series, keys: Iterable[str]) -> tuple[str, ...]:
    values = []
    for key in keys:
        value = row[key]
        if isinstance(value, (list, dict, tuple)):
            values.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        elif value is None or pd.isna(value):
            values.append("<NA>")
        else:
            values.append(str(value))
    return tuple(values)


def _merge_table(existing: pd.DataFrame | None, incoming: pd.DataFrame,
                 keys: tuple[str, ...]) -> pd.DataFrame:
    """Merge a partial adapter table while retaining rows from the prior delivery."""
    incoming = incoming.copy()
    if existing is None or not len(existing):
        return incoming
    if not set(keys).issubset(existing.columns) or not set(keys).issubset(incoming.columns):
        return pd.concat([existing, incoming], ignore_index=True).drop_duplicates(keep="last")

    existing = existing.copy()
    for column in existing.columns:
        if column not in incoming.columns:
            values = {_row_key(row, keys): row[column] for _, row in existing.iterrows()}
            incoming[column] = incoming.apply(
                lambda row: values.get(_row_key(row, keys), pd.NA), axis=1)
    if "embedding_row" in existing.columns and "embedding_row" in incoming.columns:
        values = {_row_key(row, keys): row.embedding_row for _, row in existing.iterrows()}
        def carry_embedding(row: pd.Series):
            value = row.embedding_row
            missing = value is None or (isinstance(value, str) and not value)
            if not missing:
                missing = bool(pd.isna(value))
            if missing:
                return values.get(_row_key(row, keys), value)
            return value
        incoming["embedding_row"] = incoming.apply(carry_embedding, axis=1)
    incoming_keys = {_row_key(row, keys) for _, row in incoming.iterrows()}
    retained = existing.loc[
        ~existing.apply(lambda row: _row_key(row, keys) in incoming_keys, axis=1)
    ]
    columns = list(dict.fromkeys([*existing.columns, *incoming.columns]))
    frames = [frame.reindex(columns=columns) for frame in (retained, incoming) if len(frame)]
    return pd.concat(frames, ignore_index=True) if frames else incoming.reindex(columns=columns)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                      check_dtype=False, check_index_type=False,
                                      check_column_type=False)
    except AssertionError:
        return False
    return True


def _read_existing_table(output: Path, name: str) -> pd.DataFrame | None:
    path = output / name
    if not path.is_file():
        return None
    if name == "images.parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False)


_INPUT_TABLE_NAMES = (
    "images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv",
)


def _input_table_versions(output: Path) -> tuple[tuple[int, int] | None, ...]:
    """Return the publish versions used to detect a concurrent table update."""
    versions = []
    for name in _INPUT_TABLE_NAMES:
        path = output / name
        if not path.is_file():
            versions.append(None)
            continue
        stat = path.stat()
        versions.append((stat.st_size, stat.st_mtime_ns))
    return tuple(versions)


def _write_csv_if_changed(path: Path, frame: pd.DataFrame) -> bool:
    payload = _csv_bytes(frame)
    if path.is_file() and path.read_bytes() == payload:
        return False
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return True


def _write_images_if_changed(path: Path, images: pd.DataFrame) -> bool:
    current = pd.read_parquet(path) if path.is_file() else None
    if current is not None and _frames_equal(current, images):
        return False
    arrow = pa.Table.from_pandas(images, preserve_index=False)
    index = arrow.schema.get_field_index("source_files")
    arrow = arrow.set_column(index, "source_files",
                             pa.array(images.source_files.tolist(), type=pa.list_(pa.string())))
    temporary = path.with_name(path.name + ".partial")
    pq.write_table(arrow, temporary, compression="zstd")
    os.replace(temporary, path)
    return True


def panel_from_s1(study: str, panel_id: str, source_names: list[str], s1_path: Path,
                  *, aliases: dict[str, str] | None = None,
                  drop_indices: dict[int, str] | None = None,
                  downstream_source_indices: set[int] | None = None) -> pd.DataFrame:
    """Join ordered native labels to S1; explicit source-supported aliases only.

    S1 nuclear use overrides marker exclusion for segmentation input, while
    mapping_status preserves the original annotation. Duplicate measurements
    require an explicit downstream selection; selected segmentation rows retain pixels.
    When supplied, downstream_source_indices enumerates the entire downstream
    selection, including unique markers; drop_indices denotes physical removal.
    """
    s1 = pd.read_csv(s1_path, dtype=str, keep_default_na=False)
    s1 = s1[s1.Study == study].set_index("Original marker", verify_integrity=True)
    aliases, drop_indices = aliases or {}, drop_indices or {}
    rows = []
    for index, name in enumerate(source_names):
        key = aliases.get(name, name)
        if key not in s1.index:
            raise ValueError(f"{study}/{panel_id}: source label {name!r} has no S1 match")
        annotation = s1.loc[key]
        unified = annotation["Unified marker"]
        nuclear = annotation["Segmentation channel"] == "1"
        body = annotation["Segmentation channel"] == "2"
        if unified == "Excluded":
            status, role, marker = "excluded", "exclude", ""
        elif unified.endswith(" (segmentation only)"):
            status, role, marker = "segmentation_only", "segmentation-only", name
        elif unified.endswith(" (mapping pending)"):
            status, role = "mapping_pending", "downstream"
            marker = unified.removesuffix(" (mapping pending)")
        else:
            status, role, marker = "mapped", "downstream", unified
        if (nuclear or body) and role == "exclude":
            role, marker = "segmentation-only", name
        if (role == "downstream" and downstream_source_indices is not None
                and index not in downstream_source_indices):
            role = "segmentation-only" if nuclear or body else "exclude"
        keep = nuclear or body or role == "downstream"
        reason = "S1 selected nuclear input" if nuclear else "S1 admitted marker"
        exclusion = "" if keep else "S1 excluded or unselected segmentation channel"
        if index in drop_indices:
            if nuclear or body:
                raise ValueError(f"{panel_id}: cannot physically drop selected segmentation channel {index}")
            keep, role, exclusion = False, "exclude", drop_indices[index]
        rows.append(dict(panel_id=panel_id, source_channel_index=index,
                         source_name=name, marker_id=marker, mapping_status=status,
                         role=role, nuclear=nuclear, body=body, keep_channel=keep,
                         output_channel_index=None, keep_reason=reason if keep else "",
                         exclusion_reason=exclusion))
        rows[-1]["s1_original_marker"] = key
    panel = pd.DataFrame(rows)
    retained = panel.keep_channel
    panel.loc[retained, "output_channel_index"] = np.arange(retained.sum())
    panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
    downstream = panel[retained & (panel.role == "downstream")]
    if downstream.marker_id.duplicated().any():
        repeated = downstream.loc[downstream.marker_id.duplicated(False), "source_name"].tolist()
        raise ValueError(f"{panel_id}: specify source-supported duplicate selection for {repeated}")
    return panel


def _checkpoint_identity(reader: ImageReader, panel: pd.DataFrame, tile_size: int) -> dict | None:
    """Bind reusable pixels to the native reader and every declared source dependency."""
    from cellweft.data.io.readers import SourceReader
    from cellweft.data.preparation.inputs import file_revision

    if type(reader) is not SourceReader:
        return None
    source = reader.source
    if source.kind == "zarr" and not source.source_revision:
        return None
    native_paths = source.paths if source.kind == "tiff_planes" else (source.path,)
    assets = dict.fromkeys((*native_paths, source.valid_mask, source.sample_region))
    assets.pop(None, None)
    revisions = [file_revision(Path(path)) for path in assets]
    return {
        "reader": "SourceReader", "shape": list(reader.shape), "dtype": str(np.dtype(reader.dtype)),
        "source": json.loads(json.dumps(asdict(source), default=str, sort_keys=True)),
        "assets": revisions, "panel": panel.sort_values("source_channel_index")
        .sort_index(axis=1).to_json(orient="records"), "tile_size": tile_size,
    }


def _checkpoint_files_match(checkpoint: Path, record: dict) -> bool:
    from cellweft.data.preparation.inputs import file_revision

    result = record["result"]
    names = ["input.tiff"] + (["valid_pixels.tiff"] if result["valid_pixels"] else [])
    try:
        if any(file_revision(checkpoint / name) != record["files"][name] for name in names):
            return False
        with tifffile.TiffFile(checkpoint / "input.tiff") as tif:
            if (len(tif.pages) != result["channel_count"] or
                    tif.pages[0].shape != (result["height_px"], result["width_px"]) or
                    str(tif.pages[0].dtype) != result["source_dtype"]):
                return False
        if result["valid_pixels"]:
            with tifffile.TiffFile(checkpoint / "valid_pixels.tiff") as tif:
                if (len(tif.pages) != 1 or
                        tif.pages[0].shape != (result["height_px"], result["width_px"]) or
                        tif.pages[0].dtype != np.dtype("uint8")):
                    return False
    except (FileNotFoundError, KeyError, ValueError, tifffile.TiffFileError):
        return False
    return True


def _file_version(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _complete_partial_matches(staged: Path, partial: Path) -> tuple[tuple, tuple] | None:
    """Prove complete encoded-byte equality and capture both stable file versions."""
    if not partial.is_file():
        return None
    with staged.open("rb") as source, partial.open("rb") as target:
        staged_version = _file_version(os.fstat(source.fileno()))
        partial_version = _file_version(os.fstat(target.fileno()))
        if staged_version[2] != partial_version[2]:
            return None
        while source_block := source.read(8 * 1024 * 1024):
            if source_block != target.read(len(source_block)):
                return None
        if (target.read(1) or _file_version(os.fstat(source.fileno())) != staged_version or
                _file_version(os.fstat(target.fileno())) != partial_version or
                _file_version(staged.stat()) != staged_version or
                _file_version(partial.stat()) != partial_version):
            return None
    return staged_version, partial_version


def _proved_versions_unchanged(staged: Path, partial: Path, proof: tuple | None) -> bool:
    if proof is None:
        return False
    try:
        return _file_version(staged.stat()) == proof[0] and _file_version(partial.stat()) == proof[1]
    except FileNotFoundError:
        return False


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_partial_path(checkpoint: Path, name: str) -> Path:
    return checkpoint / (name + ".durable.json")


def _durable_partial_version(checkpoint: Path, name: str,
                             staged_version: tuple, partial: Path) -> tuple | None:
    try:
        record = json.loads(_durable_partial_path(checkpoint, name).read_text())
        if record["staged_version"] != list(staged_version):
            return None
        version = tuple(record["partial_version"])
        return version if _file_version(partial.stat()) == version else None
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def _copy_and_fsync(staged: Path, partial: Path) -> None:
    synced_bytes = 0
    stage = "source open"
    try:
        with staged.open("rb") as source:
            stage = "partial open"
            with partial.open("wb") as output:
                while True:
                    stage = "source read"
                    block = source.read(64 * 1024 * 1024)
                    if not block:
                        break
                    stage = "partial write"
                    if output.write(block) != len(block):
                        raise OSError("short partial write")
                    stage = "partial flush"
                    output.flush()
                    stage = "partial fsync"
                    os.fsync(output.fileno())
                    synced_bytes += len(block)
                if not synced_bytes:
                    stage = "partial flush"
                    output.flush()
                    stage = "partial fsync"
                    os.fsync(output.fileno())
                stage = "partial close"
            stage = "source close"
    except BaseException as error:
        error.add_note(f"{partial.name} {stage} failed; {synced_bytes} bytes previously fsynced")
        raise


def _copy_durable_partial(staged: Path, partial: Path, checkpoint: Path,
                          name: str, staged_version: tuple) -> None:
    proof_path = _durable_partial_path(checkpoint, name)
    if proof_path.exists():
        proof_path.unlink()
        _fsync_directory(checkpoint)
    _copy_and_fsync(staged, partial)
    if _file_version(staged.stat()) != staged_version:
        raise RuntimeError("Closed input checkpoint changed during publication copy")
    partial_version = _file_version(partial.stat())
    if partial_version[2] != staged_version[2]:
        raise RuntimeError(f"{name} partial byte count differs from closed checkpoint")
    temporary = proof_path.with_name(proof_path.name + ".partial")
    try:
        with temporary.open("w") as output:
            json.dump({"staged_version": staged_version, "partial_version": partial_version}, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, proof_path)
        _fsync_directory(checkpoint)
    except BaseException as error:
        error.add_note(f"{name} local durability proof write failed")
        raise
    if (_file_version(staged.stat()) != staged_version or
            _file_version(partial.stat()) != partial_version):
        raise RuntimeError(f"{name} changed during local durability proof write")


def export_input(reader: ImageReader, panel: pd.DataFrame, destination: Path,
                 *, tile_size: int = 512, io_lock_root: Path | None = None,
                 data_root: Path | None = None) -> dict:
    """Keep a closed local image across formal publication failures."""
    from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease, resource_lock
    from cellweft.data.preparation.inputs import file_revision

    staging_root = study_input_staging_root(destination)
    staging_root.mkdir(parents=True, exist_ok=True)
    destination = Path(destination)
    checkpoint = staging_root / "publish_checkpoints" / destination.resolve().relative_to("/")
    source = getattr(reader, "source", None)
    source_path = Path(getattr(source, "path", destination))
    with formal_read_lease(source_path, io_lock_root=io_lock_root, data_root=data_root):
        identity = _checkpoint_identity(reader, panel, tile_size)
        record_path = checkpoint / "complete.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
        reusable = (identity is not None and record is not None and
                    record["identity"] == identity and _checkpoint_files_match(checkpoint, record))
        if reusable:
            result = record["result"]
        else:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="cellweft-input-", dir=checkpoint.parent) as temporary_root:
                temporary = Path(temporary_root)
                staged = temporary / "input.tiff"
                result = _export_input(reader, panel, staged, tile_size=tile_size)
                if identity is not None and _checkpoint_identity(reader, panel, tile_size) != identity:
                    raise RuntimeError("Input source changed during local export")
                if result["admitted"]:
                    names = ["input.tiff"] + (["valid_pixels.tiff"] if result["valid_pixels"] else [])
                    files = {name: {**file_revision(temporary / name),
                                    "path": str((checkpoint / name).resolve())} for name in names}
                    # The marker is last; failed computation never creates a ready checkpoint.
                    (temporary / "complete.json").write_text(json.dumps(dict(
                        identity=identity, result=result, files=files), sort_keys=True))
                    if checkpoint.exists():
                        backup = Path(tempfile.mkdtemp(prefix="old-input-", dir=checkpoint.parent))
                        backup.rmdir()
                        checkpoint.rename(backup)
                        try:
                            temporary.rename(checkpoint)
                        except BaseException:
                            backup.rename(checkpoint)
                            raise
                        shutil.rmtree(backup)
                    else:
                        temporary.rename(checkpoint)
        checkpoint_versions = {}
        if result["admitted"]:
            names = ["input.tiff"] + (["valid_pixels.tiff"] if result["valid_pixels"] else [])
            checkpoint_versions = {
                name: _file_version((checkpoint / name).stat()) for name in names
            }
    staged = checkpoint / "input.tiff"
    publication_stage = "formal lease"
    try:
        proofs = {}
        if result["admitted"]:
            names = ["input.tiff"] + (["valid_pixels.tiff"] if result["valid_pixels"] else [])
            candidates = []
            for name in names:
                staged_file = checkpoint / name
                partial_file = destination.with_name(name + ".partial")
                durable_version = _durable_partial_version(
                    checkpoint, name, checkpoint_versions[name], partial_file)
                if (durable_version is not None and
                        partial_file.stat().st_size == staged_file.stat().st_size):
                    candidates.append((name, staged_file, partial_file, durable_version))
            if candidates:
                publication_stage = "partial read lease"
                with formal_read_lease(destination, io_lock_root=io_lock_root, data_root=data_root):
                    with resource_lock(destination.parent, shared=True, name=".input.lock"):
                        for name, staged_file, partial_file, durable_version in candidates:
                            publication_stage = f"{name.removesuffix('.tiff')} partial comparison"
                            proof = _complete_partial_matches(staged_file, partial_file)
                            proofs[name] = (proof if proof is not None and
                                            proof[0] == checkpoint_versions[name] and
                                            proof[1] == durable_version else None)
        publication_stage = "formal lease"
        with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock",
                                  io_lock_root=io_lock_root, data_root=data_root):
            publication_stage = "source version check"
            if result["admitted"] and identity is not None and _checkpoint_identity(reader, panel, tile_size) != identity:
                raise RuntimeError("Input source changed before formal publication")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if result["admitted"]:
                partial = destination.with_name(destination.name + ".partial")
                publication_stage = "input partial version check"
                input_reusable = _proved_versions_unchanged(staged, partial, proofs.get("input.tiff"))
                if not input_reusable:
                    publication_stage = "input copy"
                    _copy_durable_partial(staged, partial, checkpoint, "input.tiff",
                                          checkpoint_versions["input.tiff"])
                final_valid = destination.with_name("valid_pixels.tiff")
                staged_valid = checkpoint / "valid_pixels.tiff"
                if result["valid_pixels"]:
                    valid_partial = final_valid.with_name(final_valid.name + ".partial")
                    publication_stage = "valid_pixels partial version check"
                    valid_reusable = _proved_versions_unchanged(
                        staged_valid, valid_partial, proofs.get("valid_pixels.tiff"))
                    if not valid_reusable:
                        publication_stage = "valid_pixels copy"
                        _copy_durable_partial(staged_valid, valid_partial, checkpoint,
                                              "valid_pixels.tiff",
                                              checkpoint_versions["valid_pixels.tiff"])
                publication_stage = "partial durability version recheck"
                for name in names:
                    staged_file = checkpoint / name
                    partial_file = destination.with_name(name + ".partial")
                    if (_file_version(staged_file.stat()) != checkpoint_versions[name] or
                            _durable_partial_version(
                                checkpoint, name, checkpoint_versions[name], partial_file) is None):
                        raise RuntimeError(f"{name} durability proof or file version changed before replace")
                publication_stage = "input replace"
                os.replace(partial, destination)
                if result["valid_pixels"]:
                    publication_stage = "valid_pixels replace"
                    os.replace(valid_partial, final_valid)
                else:
                    final_valid.unlink(missing_ok=True)
                publication_stage = "input header reopen"
                with tifffile.TiffFile(destination) as tif:
                    if (len(tif.pages) != result["channel_count"] or
                            tif.pages[0].shape != (result["height_px"], result["width_px"]) or
                            str(tif.pages[0].dtype) != result["source_dtype"]):
                        raise ValueError("Published TIFF header differs from staged input")
                if result["valid_pixels"]:
                    publication_stage = "valid_pixels header reopen"
                    with tifffile.TiffFile(final_valid) as tif:
                        if (len(tif.pages) != 1 or
                                tif.pages[0].shape != (result["height_px"], result["width_px"]) or
                                tif.pages[0].dtype != np.dtype("uint8")):
                            raise ValueError("Published validity header differs from staged input")
                publication_stage = "source version recheck"
                if identity is not None and _checkpoint_identity(reader, panel, tile_size) != identity:
                    raise RuntimeError("Input source changed during formal publication")
                publication_stage = "destination directory fsync"
                _fsync_directory(destination.parent)
            else:
                destination.unlink(missing_ok=True)
                destination.with_name("valid_pixels.tiff").unlink(missing_ok=True)
                publication_stage = "destination directory fsync"
                _fsync_directory(destination.parent)
    except BaseException as error:
        if result["admitted"]:
            error.add_note(f"Input publication stage: {publication_stage}")
            error.add_note(f"Closed local input checkpoint for publication retry: {checkpoint}")
        raise
    if result["admitted"]:
        shutil.rmtree(checkpoint)
    return result


def _export_input(reader: ImageReader, panel: pd.DataFrame, destination: Path,
                 *, tile_size: int = 512) -> dict:
    """Stream retained native channels into a square OME BigTIFF, one tile at a time.

    The reader exposes one source ROI in local coordinates. All channels share
    centered padding. Acquisition validity is read independently of signal;
    image admission requires a nonzero retained downstream measurement.
    """
    panel = panel.sort_values("source_channel_index")
    if panel.source_channel_index.tolist() != list(range(reader.shape[0])):
        raise ValueError("Panel must describe every source channel in reader order")
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    if kept.output_channel_index.tolist() != list(range(len(kept))):
        raise ValueError("Output channel indices must be contiguous")
    if not len(kept) or not (kept.role == "downstream").any():
        raise ValueError("Input admission requires retained marker measurements")
    _, h, w = reader.shape
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    dtype = np.dtype(reader.dtype)
    measured = 0
    missing = False
    for window in windows((h, w), tile_size):
        valid = reader.valid_window(window)
        measured += int(valid.sum())
        missing |= not bool(valid.all())
    sums = np.zeros(len(kept), dtype=np.float64)
    nonzero = np.zeros(len(kept), dtype=np.int64)
    names = kept.marker_id.tolist()
    source_indices = kept.source_channel_index.tolist()

    def intersection(window: Window) -> tuple[Window, tuple[slice, slice]] | None:
        x0, y0, x1, y1 = window
        a, b, c, d = max(x0, left), max(y0, top), min(x1, left+w), min(y1, top+h)
        if a >= c or b >= d:
            return None
        return (a-left, b-top, c-left, d-top), (slice(b-y0, d-y0), slice(a-x0, c-x0))

    def tiles():
        for output, source_index in enumerate(source_indices):
            for window in windows((side, side), tile_size):
                tile = np.zeros((tile_size, tile_size), dtype=dtype)
                overlap = intersection(window)
                if overlap is not None:
                    region, slices = overlap
                    values = reader.read_window([source_index], region)[0]
                    valid = reader.valid_window(region)
                    if not np.isfinite(values[valid]).all():
                        raise ValueError(f"Nonfinite source pixels in channel {source_index}, window {region}")
                    tile[slices] = np.where(valid, values, 0)
                    sums[output] += values[valid].sum(dtype=np.float64)
                    nonzero[output] += np.count_nonzero(values[valid])
                yield tile

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    metadata = {"axes": "CYX", "Channel": {"Name": names}}
    source = getattr(reader, "source", None)
    if source is not None and source.pixel_size_um is not None:
        metadata.update(PhysicalSizeX=source.pixel_size_um[0], PhysicalSizeXUnit="µm",
                        PhysicalSizeY=source.pixel_size_um[1], PhysicalSizeYUnit="µm")
    with tifffile.TiffWriter(temporary, bigtiff=True, ome=True) as writer:
        writer.write(data=tiles(), shape=(len(kept), side, side), dtype=dtype,
                     tile=(tile_size, tile_size), compression="deflate",
                     photometric="minisblack", metadata=metadata, maxworkers=2)
    admitted = bool(nonzero[(kept.role == "downstream").to_numpy()].any())
    valid_path = None
    if admitted:
        with tifffile.TiffFile(temporary) as tif:
            if len(tif.pages) != len(kept) or tif.pages[0].shape != (side, side) or tif.pages[0].dtype != dtype:
                raise ValueError("Saved TIFF header differs from source/panel")
        if missing:
            valid_path = destination.with_name("valid_pixels.tiff")

            def read_valid(channel: int, window: Window) -> np.ndarray:
                x0, y0, x1, y1 = window
                values = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
                overlap = intersection(window)
                if overlap is not None:
                    region, slices = overlap
                    values[slices] = reader.valid_window(region)
                return values

            write_image(valid_path, (1, side, side), np.dtype("uint8"), read_valid,
                        tile_size=tile_size)
        else:
            destination.with_name("valid_pixels.tiff").unlink(missing_ok=True)
        os.replace(temporary, destination)
    else:
        temporary.unlink()
        destination.unlink(missing_ok=True)
        destination.with_name("valid_pixels.tiff").unlink(missing_ok=True)
    return dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                pad_right_px=side-w-left, pad_bottom_px=side-h-top,
                source_dtype=str(dtype), source_channel_count=reader.shape[0],
                channel_count=len(kept), admitted=admitted,
                channel_nonzero_pixels=nonzero.tolist(),
                channel_means=(sums / measured).tolist() if measured else [0.0]*len(kept),
                valid_pixels=valid_path.name if valid_path else None)


def validate_image_metadata(images: pd.DataFrame, samples: pd.DataFrame,
                            panel: pd.DataFrame, field_mapping: pd.DataFrame) -> pd.DataFrame:
    """Validate base fields and typed extensions without discarding source columns.

    Extension field_mapping notes contain JSON data_type and definition fields.
    Invalid required values, identities, foreign keys or extension types raise ValueError.
    """
    dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
    vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
    fields = dictionary[dictionary.table_name == "image_metadata"]
    missing = set(fields.field_name) - set(images.columns)
    if missing:
        raise ValueError(f"Missing image metadata columns: {sorted(missing)}")
    images = images.copy()
    files = [json.loads(value) if isinstance(value, str) else (value.tolist() if isinstance(value, np.ndarray) else value)
             for value in images.source_files]
    if any(not isinstance(value, list) or not value or
           not all(isinstance(path, str) for path in value) for value in files):
        raise ValueError("source_files must contain nonempty lists of source paths")
    images["source_files"] = files
    for field in fields.itertuples(index=False):
        key = field.field_name
        if field.data_type in {"string", "category"}:
            images[key] = images[key].astype("string").replace("", pd.NA)
        elif field.data_type == "integer":
            images[key] = pd.to_numeric(images[key], errors="raise").astype("Int64")
        elif field.data_type == "number":
            images[key] = pd.to_numeric(images[key], errors="raise").astype("Float64")
        if field.required and images[key].isna().any():
            raise ValueError(f"Missing required image field: {key}")
        if field.value_set:
            allowed = vocabulary.loc[vocabulary.value_set == field.value_set, "allowed_value"]
            if not images[key].dropna().isin(allowed).all():
                raise ValueError(f"Image field {key} contains values outside {field.value_set}")
    if images.duplicated(["study", "image_id"]).any():
        raise ValueError("Duplicate study/image identity")
    if samples.duplicated(["study", "sample_id"]).any():
        raise ValueError("Duplicate study/sample identity")
    if not set(zip(images.loc[images.sample_id.notna(), "study"], images.sample_id.dropna())).issubset(set(zip(samples.study, samples.sample_id))):
        raise ValueError("Image references an unknown biological sample")
    if not images.panel_id.isin(panel.panel_id).all():
        raise ValueError("Image references an unknown panel")
    extras = set(images.columns) - set(fields.field_name)
    for key in extras:
        definitions = field_mapping[(field_mapping.target_table == "image_metadata") &
                                    (field_mapping.target_field == key)]
        if definitions.empty:
            raise ValueError(f"Image extension {key} has no field_mapping definition")
        specifications = [json.loads(value) for value in definitions.notes]
        if any(spec != specifications[0] for spec in specifications):
            raise ValueError(f"Conflicting extension definitions: {key}")
        spec = specifications[0]
        if not spec.get("definition") or spec.get("data_type") not in {"string", "integer", "number", "boolean"}:
            raise ValueError(f"Extension {key} requires definition and supported data_type in notes JSON")
        kind = spec["data_type"]
        if kind in {"integer", "number"}:
            images[key] = pd.to_numeric(images[key], errors="raise").astype("Int64" if kind == "integer" else "Float64")
        elif kind == "boolean":
            values = images[key].dropna()
            if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"Extension {key} must contain booleans")
            images[key] = images[key].astype("boolean")
        else:
            if not images[key].dropna().map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"Extension {key} must contain strings")
            images[key] = images[key].astype("string")
    return images


def save_input_tables(output: Path, images: pd.DataFrame, samples: pd.DataFrame,
                      panel: pd.DataFrame, field_mapping: pd.DataFrame, *,
                      io_lock_root: Path | None = None,
                      data_root: Path | None = None,
                      priority: str = "normal",
                      replace_metadata_studies: set[str] | None = None) -> None:
    """Merge and publish the complete typed input tables.

    Existing image/sample/field rows remain available to partial adapters. A
    previously published ``embedding_row`` is carried forward when the input
    adapter supplies a panel without that vector-only column. Unchanged files
    are left untouched so their binding revisions remain stable. A four-table
    version check prevents an embedding update from being overwritten between
    merge and publication. ``priority`` uses the existing shared I/O queue;
    short input handoffs can precede another queued bulk result copy.
    ``replace_metadata_studies`` replaces prior sample and field-mapping rows
    for the named studies before merging their incoming rows.
    """
    # Source adapters may omit optional base fields; the delivered table is complete.
    dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
    optional = dictionary[(dictionary.table_name == "image_metadata") & ~dictionary.required]
    images = images.copy()
    for field in optional.field_name:
        if field not in images: images[field] = pd.NA
    incoming_images = images.copy()
    incoming_samples = samples.copy()
    incoming_panel = panel.copy()
    incoming_field_mapping = field_mapping.copy()
    from cellweft.data.io.publication import resource_lock
    from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease

    while True:
        # Read and merge outside the writer gate. The table versions are a CAS
        # token; a vector update between this block and publication forces a
        # fresh merge from the original caller frames.
        with formal_read_lease(output, io_lock_root=io_lock_root, data_root=data_root,
                               priority=priority):
            with resource_lock(output, shared=True):
                table_versions = _input_table_versions(output)
                prior_images = _read_existing_table(output, "images.parquet")
                prior_samples = _read_existing_table(output, "sample_metadata.csv")
                prior_panel = _read_existing_table(output, "panel_mapping.csv")
                prior_fields = _read_existing_table(output, "field_mapping.csv")
                if replace_metadata_studies:
                    if prior_samples is not None:
                        prior_samples = prior_samples[
                            ~prior_samples.study.isin(replace_metadata_studies)]
                    if prior_fields is not None:
                        prior_fields = prior_fields[
                            ~prior_fields.study.isin(replace_metadata_studies)]
                previous_payloads = {
                    name: (output / name).read_bytes() if (output / name).is_file() else None
                    for name in ("sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv")
                }
                merged_samples = _merge_table(prior_samples, incoming_samples, ("study", "sample_id"))
                merged_panel = _merge_table(
                    prior_panel, incoming_panel, ("panel_id", "source_channel_index"))
                merged_fields = _merge_table(
                    prior_fields, incoming_field_mapping,
                    ("study", "source_file", "source_sheet_or_table", "source_field",
                     "target_table", "target_field", "join_keys"))
                merged_images = _merge_table(prior_images, incoming_images, ("study", "image_id"))
                merged_images = validate_image_metadata(
                    merged_images, merged_samples, merged_panel, merged_fields)

        payloads = {
            "sample_metadata.csv": _csv_bytes(merged_samples),
            "panel_mapping.csv": _csv_bytes(merged_panel),
            "field_mapping.csv": _csv_bytes(merged_fields),
        }
        changed = (prior_images is None or not _frames_equal(prior_images, merged_images) or
                   any(previous_payloads[name] != payload for name, payload in payloads.items()))
        if not changed:
            return

        staging_root = study_input_staging_root(output)
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cellweft-tables-", dir=staging_root) as temporary_root:
            temporary_root = Path(temporary_root)
            staged_images = temporary_root / "images.parquet"
            arrow = pa.Table.from_pandas(merged_images, preserve_index=False)
            index = arrow.schema.get_field_index("source_files")
            arrow = arrow.set_column(index, "source_files",
                                     pa.array(merged_images.source_files.tolist(), type=pa.list_(pa.string())))
            pq.write_table(arrow, staged_images, compression="zstd")
            staged_payloads = {}
            for name, payload in payloads.items():
                staged = temporary_root / name
                staged.write_bytes(payload)
                staged_payloads[name] = staged

            # Recheck the CAS token while the writer and study publication
            # locks are held. No merge or serialization happens in this gate.
            with formal_publish_lease(output, directory=output,
                                      io_lock_root=io_lock_root, data_root=data_root,
                                      priority=priority):
                if _input_table_versions(output) != table_versions:
                    continue
                output.mkdir(parents=True, exist_ok=True)
                if prior_images is None or not _frames_equal(prior_images, merged_images):
                    temporary = output / "images.parquet.partial"
                    shutil.copyfile(staged_images, temporary)
                    os.replace(temporary, output / "images.parquet")
                for name, staged in staged_payloads.items():
                    if previous_payloads[name] == payloads[name]:
                        continue
                    temporary = output / f"{name}.partial"
                    shutil.copyfile(staged, temporary)
                    os.replace(temporary, output / name)
        return
