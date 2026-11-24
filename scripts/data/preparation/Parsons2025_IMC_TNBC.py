"""Prepare native S-BIAD2027 IMC ROIs from completed, proven local sources.

Inputs: canonical native manifest, recorded native acceptance, ordered channels, S1.
Outputs: native square input.tiff, complete input tables and preparer checkpoints.
Unresolved specimen/patient identities remain blank; independent review is pending.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from PIL import Image

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative
from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.writers import write_json
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

pa.set_cpu_count(3)
pa.set_io_thread_count(1)
STUDY = "Parsons2025_IMC_TNBC"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
RAW = ROOT / "raw/public" / STUDY / "S-BIAD2027"
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
NATIVE_FACTS = META / "native_source_acceptance.json"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TEMPLATES = PROJECT / "docs/data/templates/study_input"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", type=Path, required=True)
parser.add_argument("--io-timeout", type=int, default=180)
parser.add_argument("--candidate-output", type=Path, default=None,
                    help="Write a private candidate scope, preferably on local SSD")
parser.add_argument("--reuse-existing-formal", type=Path, default=None,
                    help="Exclude image IDs already present in this formal study output")
parser.add_argument("--exclude-calibration", action="store_true",
                    help="Explicitly exclude descriptions marked as test/calibration")
parser.add_argument("--full-pixel-check", action="store_true",
                    help="Run block-level source/output equality checks for every new image")
parser.add_argument("--direct-native-read", action="store_true",
                    help="Read only selected native acquisitions instead of copying whole MCD containers")
parser.add_argument("--source-cache", type=Path, default=None,
                    help="Share complete MCD staging across image shards on local SSD")
parser.add_argument("--shard-count", type=int, default=1,
                    help="Deterministic image shard count; requires --shard-index")
parser.add_argument("--shard-index", type=int, default=None,
                    help="Zero-based image shard index for private candidate output")
parser.add_argument("--aggregate-shards", action="store_true",
                    help="Merge closed shard image tables into the candidate parent root")
parser.add_argument("--aggregate-output", type=Path, default=None,
                    help="Explicit candidate parent root for --aggregate-shards")
parser.add_argument("--source030-candidate-manifest", type=Path, default=None,
                    help="Include the independently exported source030 candidate during aggregation")
parser.add_argument("--plan-only", action="store_true",
                    help="Build and checkpoint the selected scope without exporting images")
parser.add_argument("--stop-marker", type=Path, default=None,
                    help="Stop cleanly after the current image when this marker exists")
parser.add_argument("--repair-preservation-only", action="store_true",
                    help="Repair reviewed FFPE vocabulary and its two mappings without extending image scope")
args = parser.parse_args()
if args.shard_count < 1:
    raise ValueError("--shard-count must be positive")
if args.shard_index is not None and not 0 <= args.shard_index < args.shard_count:
    raise ValueError("--shard-index must be within --shard-count")
if args.aggregate_shards and args.shard_index is not None:
    raise ValueError("--aggregate-shards cannot be combined with --shard-index")
SOURCE_CACHE = args.source_cache.resolve() if args.source_cache is not None else None
if SOURCE_CACHE is not None:
    try:
        SOURCE_CACHE.relative_to(PROJECT.resolve())
    except ValueError:
        raise ValueError("--source-cache must be under the project SSD")
CANDIDATE_OUTPUT = None
if args.candidate_output is not None:
    candidate_output = args.candidate_output.resolve()
    if candidate_output == OUTPUT.resolve():
        raise ValueError("candidate output must be separate from formal processed output")
    OUTPUT = candidate_output
    CANDIDATE_OUTPUT = candidate_output
    OUTPUT.mkdir(parents=True, exist_ok=True)
args.job.mkdir(parents=True, exist_ok=True)
SHARD_MODE = args.shard_index is not None
SHARD_ROOT = OUTPUT / "_shards" / str(args.shard_index) if SHARD_MODE else None
TABLE_OUTPUT = SHARD_ROOT if SHARD_MODE else OUTPUT
IMAGE_OUTPUT = SHARD_ROOT if SHARD_MODE else OUTPUT
EXPORT_CACHE = args.job / "exports"
PIXEL_CHECK_CACHE = args.job / "pixel_checks"
OVERVIEW_CACHE = args.job / "overviews"
STOP_MARKER = args.stop_marker.resolve() if args.stop_marker is not None else None
if SHARD_MODE and not args.plan_only:
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
local = Path("/tmp") / STUDY / args.job.parent.name / args.job.name
local.mkdir(parents=True, exist_ok=True)
if SOURCE_CACHE is not None:
    SOURCE_preparer = args.job.name
previous_progress = json.loads((args.job / "progress.json").read_text()) if (args.job / "progress.json").exists() else {}
state = dict(study=STUDY, phase="repair" if args.repair_preservation_only else "prepare",
             scientific_repair_round=1 if args.repair_preservation_only else 0,
             completed_images=previous_progress.get("completed_images", 0),
             sample_rows=previous_progress.get("sample_rows", 0),
             sample_linked_images=previous_progress.get("sample_linked_images", 0),
             patient_linked_samples=0, independent_review_performed=False)


class SpaceUnavailable(RuntimeError):
    """The explicit NAS/local reserve cannot be preserved for this operation."""


def progress(stage: str, **details: object) -> None:
    """Persist the last closed scope and exact next operation on this study."""
    state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **details)
    write_json(args.job / "progress.json", state)
    print(json.dumps(dict(stage=stage, completed_images=state["completed_images"],
                         current_image=state.get("current_image"))), flush=True)


@contextmanager
def bounded_io(stage: str):
    """Close context-owned files on bounded I/O interruption and retain checkpoints."""
    def timeout(signum, frame):
        raise TimeoutError(f"I/O operation exceeded {args.io_timeout}s: {stage}")
    progress(stage)
    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.io_timeout)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def relative(path: Path) -> str:
    if path.is_relative_to(META):
        return str(metadata_source_relative(path, ROOT))
    if CANDIDATE_OUTPUT is not None:
        try:
            path.resolve().relative_to(CANDIDATE_OUTPUT.resolve())
        except ValueError:
            pass
        else:
            return str(path.resolve())
    for base in (ROOT, PROJECT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def template(name: str) -> pd.DataFrame:
    return pd.read_csv(TEMPLATES / f"{name}.csv", keep_default_na=False)


def native_preparation_sources(manifest: dict, facts: dict) -> list[tuple[str, dict]]:
    """Join recorded native acceptance to the canonical source metadata."""
    if (facts.get("schema") != "cellweft.parsons_native_source_acceptance.v1"
            or facts.get("study") != STUDY):
        raise ValueError("Expected the reviewed Parsons native source acceptance facts")
    sources = {entry["source_id"]: entry for entry in manifest["sources"]}
    result = []
    for record in facts["sources"]:
        source = sources[record["source_id"]]
        acquisitions = {entry["acquisition_id"]: entry for entry in source["acquisitions"]}
        accepted = []
        for acceptance in record["accepted_acquisitions"]:
            native = acquisitions[acceptance["id"]]
            accepted.append({
                "id": acceptance["id"], "canonical_roi_id": native["canonical_roi_id"],
                "native_metadata": native["native_metadata"],
                "ordered_channels": native["ordered_channels"],
                "all_finite": acceptance["all_finite"],
                "required_payload_complete": acceptance["required_payload_complete"],
            })
        result.append((record["source_id"] + ".json", {
            "source": source["source"], "file_stat": record["file_stat"],
            "acquisitions": accepted, "exclusions": record["exclusions"],
            "pending": record["pending"], "reused_proofs": record["reused_proofs"],
            "source_scope_complete": record["source_scope_complete"],
        }))
    return result


PRESERVATION_MAPPING = {"FFPE": "ffpe"}
PRESERVATION_RULE = (
    "Coalesce consistent nonblank source FORMAT (FFPE/FROZEN) values by specimen ID; "
    "map FFPE to vocabulary ffpe. Unknown or conflicting source categories remain blank; "
    "retain original attributes in the referenced official file lists."
)


def repair_preservation() -> None:
    """Repair only the independently identified clinical vocabulary difference."""
    targets = [OUTPUT / name for name in
               ("sample_metadata.csv", "field_mapping.csv", "review_results.csv")]
    capacity = dict(nas_free_bytes=shutil.disk_usage(ROOT).free,
                    local_free_bytes=shutil.disk_usage(local).free,
                    estimated_publication_bytes=sum(path.stat().st_size for path in targets) * 2)
    state["capacity"] = capacity
    if capacity["nas_free_bytes"] < 100 * 1024**3 + capacity["estimated_publication_bytes"]:
        raise SpaceUnavailable("Metadata publication would violate NAS reserve")
    if capacity["local_free_bytes"] < 64 * 1024**3 + capacity["estimated_publication_bytes"]:
        raise SpaceUnavailable("Metadata staging would violate local reserve")
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        with bounded_io("read_repair_scope"):
            before = {path.name: dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                      for path in targets}
            samples, fields, reviews = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in targets]
            images = pd.read_parquet(OUTPUT / "images.parquet")
            unchanged = {str(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                         for path in [OUTPUT / "images.parquet", OUTPUT / "panel_mapping.csv"]
                         + [ROOT / value for column in ("input_image_path", "valid_pixels_path")
                            for value in images[column].dropna() if value]}
            filelists = [json.loads((META / "identity_sources" / name).read_text())
                         for name in ("file_list1.json", "file_list2.json")]
    formats = {}
    for entry in filelists[0] + filelists[1]:
        attributes = {item["name"]: item.get("value", "") for item in entry["attributes"]}
        formats.setdefault(attributes["ID"], set()).add(attributes.get("FORMAT (FFPE/FROZEN)", ""))
    expected = []
    for identifier in samples.source_sample_id:
        values = formats[identifier] - {""}
        expected.append(PRESERVATION_MAPPING.get(next(iter(values)), "") if len(values) == 1 else "")
    assert set(samples.preservation_method) <= {"FFPE", "ffpe"}
    changed = int((samples.preservation_method != pd.Series(expected, index=samples.index)).sum())
    samples["preservation_method"] = expected
    affected = (fields.target_table == "sample_metadata") & (fields.target_field == "preservation_method")
    assert int(affected.sum()) == 2
    fields.loc[affected, "value_mapping"] = json.dumps(PRESERVATION_MAPPING)
    fields.loc[affected, "transformation_rule"] = PRESERVATION_RULE
    details = dict(scientific_repair_round=1, changed_sample_rows=changed,
                   mapping_rows=2, source_verified_sample_rows=len(samples),
                   independent_review_pending=True, input_pixels_reused=True,
                   prior_review="0862_review_0", patient_bridge="unresolved")
    for check, scope in (("M02", "sample"), ("D01", "study")):
        reviews.loc[len(reviews)] = [STUDY, preparer, "", scope, STUDY, check, "unresolved",
                                    relative(META / "identity_sources"), relative(OUTPUT), json.dumps(details)]
    staging = local / args.job.name
    staging.mkdir(exist_ok=True)
    for path, table in zip(targets, (samples, fields, reviews)):
        table.to_csv(staging / path.name, index=False, encoding="utf-8-sig")
    write_json(args.job / "repair_checkpoint.json", dict(details=details, before=before,
               unchanged_files=unchanged, local_staging=str(staging)))
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        with bounded_io("publish_preservation_repair"):
            for path in targets:
                assert dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) == before[path.name]
            for path in targets:
                temporary = path.with_suffix(".csv.partial")
                with (staging / path.name).open("rb") as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                os.replace(temporary, path)
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        with bounded_io("verify_preservation_repair"):
            for path, table in zip(targets, (samples, fields, reviews)):
                pd.testing.assert_frame_equal(pd.read_csv(path, dtype=str, keep_default_na=False), table)
            for name, version in unchanged.items():
                stat = Path(name).stat()
                assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == version
    state.update(expected_images=len(images), completed_images=len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.fillna("").ne("").sum()),
                 patient_linked_samples=int(samples.patient_key.ne("").sum()))
    write_json(args.job / "repair_checks.json", details)
    progress("prepared", status="prepared", independent_review_pending=True,
             remaining_image_ids=[], open_source_files=0)


def field(source: str, source_field: str, table: str, target: str, rule: str) -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=source_field, target_table=table, target_field=target,
                join_keys="canonical acquisition time/coordinates/dimensions + source ID + acquisition ID",
                value_mapping="{}", transformation_rule=rule, time_origin_rule="Unknown; left blank",
                conflict_rule="Unresolved identity remains blank; no inference from filename/ROI numbering",
                notes="Preparer evidence; independent review pending")


def header(path: Path, panel: pd.DataFrame, side: int) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (int(panel.keep_channel.sum()), side, side)
        assert tif.series[0].dtype == np.dtype("float32")
        names = [e.attrib["Name"] for e in ET.fromstring(tif.ome_metadata).iter()
                 if e.tag.endswith("}Channel")]
        expected = panel[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
        assert names == expected


def save_tables() -> None:
    # save_input_tables owns the read/CAS/write leases; queue admission must not
    # be charged against the adapter's bounded operation alarm.
    progress("publish_merged_tables")
    save_input_tables(TABLE_OUTPUT, pd.DataFrame(images, columns=template("image_metadata").columns),
                      samples, panels, fields)
    state["completed_images"] = len(images)
    state["indexed_images"] = len(images)
    write_json(args.job / "closed_pending_index.json", dict(images=[]))


def source_file_version(path: Path) -> dict:
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def shared_source_cache(item: dict) -> Path:
    """Stage one immutable MCD per source ID for all image shards."""
    if SOURCE_CACHE is None:
        raise RuntimeError("shared source cache is not configured")
    source = ROOT / item["source_file"]
    source_id = str(item["source_id"])
    if not source_id or Path(source_id).name != source_id:
        raise ValueError(f"invalid source_id for shared cache: {source_id!r}")
    expected = item["version"]
    entry = SOURCE_CACHE / "sources" / source_id
    staged = entry / source.name
    partial = staged.with_suffix(staged.suffix + ".partial")
    record = staged.with_suffix(staged.suffix + ".version.json")
    copy_record = staged.with_suffix(staged.suffix + ".copy.json")
    lock_directory = SOURCE_CACHE / ".locks" / source_id
    entry.mkdir(parents=True, exist_ok=True)

    def finish_complete() -> Path:
        if staged.exists():
            if partial.exists():
                raise RuntimeError(f"shared source cache has ambiguous complete/partial files: {staged}")
        else:
            if not partial.is_file():
                raise RuntimeError(f"shared source cache lost its partial source: {staged}")
            os.replace(partial, staged)
        local_version = source_file_version(staged)
        if local_version["size"] != expected["size"]:
            raise RuntimeError(f"shared source cache size mismatch: {staged}")
        write_json(record, dict(status="complete", source_id=source_id,
                                source_file=item["source_file"], source_version=expected,
                                local_version=local_version, copied_bytes=local_version["size"]))
        copy_record.unlink(missing_ok=True)
        return staged

    progress("shared_source_cache_wait", source_id=source_id)
    with resource_lock(lock_directory, name="source.lock"):
        progress("shared_source_cache_check", source_id=source_id)
        if record.exists():
            saved = json.loads(record.read_text())
            if saved.get("source_version") != expected:
                raise ValueError(f"shared source cache version conflict for {source_id}")
            if saved.get("status", "complete") != "complete":
                raise RuntimeError(f"shared source cache has an incomplete version record: {record}")
            if not staged.is_file():
                raise RuntimeError(f"shared source cache record has no complete source: {staged}")
            local_version = source_file_version(staged)
            if saved.get("local_version") != local_version or local_version["size"] != expected["size"]:
                raise RuntimeError(f"shared source cache changed after completion: {staged}")
            copy_record.unlink(missing_ok=True)
            progress("shared_source_cache_reused", source_id=source_id,
                     source_cache=str(staged))
            return staged

        if staged.exists() and not copy_record.exists():
            raise RuntimeError(f"shared source cache has an untracked staged file: {staged}")
        if staged.exists() and partial.exists():
            raise RuntimeError(f"shared source cache has ambiguous staged files: {staged}")
        if copy_record.exists() and not partial.exists():
            saved_copy = json.loads(copy_record.read_text())
            if saved_copy.get("source_version") != expected:
                raise ValueError(f"shared source cache version conflict for {source_id}")
            if not staged.is_file() or source_file_version(staged)["size"] != expected["size"]:
                raise RuntimeError(f"shared source cache has a receipt without complete bytes: {copy_record}")
            recorded_partial = saved_copy.get("partial_version")
            if recorded_partial and source_file_version(staged) != recorded_partial:
                raise RuntimeError(f"shared source cache complete rename changed bytes: {staged}")
            progress("shared_source_cache_finalize", source_id=source_id)
            return finish_complete()

        offset = 0
        if partial.exists():
            saved_copy = json.loads(copy_record.read_text())
            if saved_copy.get("source_version") != expected:
                raise ValueError(f"shared source cache partial version conflict for {source_id}")
            current_partial = source_file_version(partial)
            recorded_partial = saved_copy.get("partial_version", {})
            if current_partial["size"] < int(recorded_partial.get("size", 0)):
                raise RuntimeError(f"shared source cache partial was truncated: {partial}")
            offset = int(recorded_partial.get("size", 0))
            if current_partial["size"] > offset:
                # Bytes after the last fsync/checkpoint may be present in page
                # cache but are not a durable resume boundary.
                with partial.open("r+b") as output:
                    output.truncate(offset)
                    output.flush()
                    os.fsync(output.fileno())
            if offset > expected["size"]:
                raise RuntimeError(f"shared source cache partial exceeds source size: {partial}")
            if offset == expected["size"]:
                progress("shared_source_cache_finalize", source_id=source_id)
                return finish_complete()

        if shutil.disk_usage(SOURCE_CACHE).free <= 64 * 1024**3 + expected["size"]:
            raise SpaceUnavailable("Shared source cache would violate the local 64 GiB reserve")
        copied = offset
        chunk_size = 8 * 1024**2
        checkpoint_bytes = 64 * 1024**2
        if not partial.exists():
            partial.touch()
            write_json(copy_record, dict(status="partial", source_id=source_id,
                                         source_file=item["source_file"], source_version=expected,
                                         partial_version=source_file_version(partial),
                                         copied_bytes=offset,
                                         updated_at=datetime.now(timezone.utc).isoformat()))
        progress("stage_shared_native_source", source_id=source_id,
                 source_cache=str(staged), copied_bytes=copied,
                 source_size=expected["size"])
        with source.open("rb") as incoming, partial.open("ab" if offset else "wb") as outgoing:
            incoming.seek(offset)
            while copied < expected["size"]:
                segment_end = min(copied + checkpoint_bytes, expected["size"])
                # Keep the NAS read lease bounded to one durable checkpoint.
                # The source descriptor remains open so each segment resumes at
                # the existing offset without reopening the MCD over NFS.
                with formal_read_lease(source):
                    if source_file_version(source) != expected:
                        raise ValueError(f"source changed before shared staging segment: {source}")
                    while copied < segment_end:
                        chunk = incoming.read(min(chunk_size, segment_end - copied))
                        if not chunk:
                            raise OSError(f"short source read at {copied} bytes: {source}")
                        outgoing.write(chunk)
                        copied += len(chunk)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                    partial_version = source_file_version(partial)
                    write_json(copy_record, dict(status="partial", source_id=source_id,
                                                source_file=item["source_file"], source_version=expected,
                                                partial_version=partial_version,
                                                copied_bytes=copied,
                                                updated_at=datetime.now(timezone.utc).isoformat()))
                    if source_file_version(source) != expected:
                        raise ValueError(f"source changed during shared staging segment: {source}")
        if source_file_version(partial)["size"] != expected["size"]:
            raise OSError(f"shared source staging closed at the wrong size: {partial}")
        staged_path = finish_complete()
        progress("shared_source_cache_complete", source_id=source_id,
                 source_cache=str(staged_path), copied_bytes=expected["size"])
        return staged_path


def source_cache_complete(item: dict) -> bool:
    """Return whether this source can be reused without opening the NAS MCD."""
    if SOURCE_CACHE is None:
        return False
    source_id = str(item["source_id"])
    entry = SOURCE_CACHE / "sources" / source_id
    staged = entry / Path(item["source_file"]).name
    record = staged.with_suffix(staged.suffix + ".version.json")
    if not staged.is_file() or not record.is_file():
        return False
    saved = json.loads(record.read_text())
    return (saved.get("status") == "complete"
            and saved.get("source_version") == item["version"]
            and saved.get("local_version") == source_file_version(staged)
            and staged.stat().st_size == item["version"]["size"])


def rotate_unfinished_source_groups(assigned_by_shard: dict[int, list[dict]],
                                    completed_ids_by_shard: dict[int, set[str]],
                                    shard_index: int) -> tuple[list[dict], dict]:
    """Prioritize each uncached source on one stable shard without changing ownership."""
    groups_by_shard = {}
    pending_by_shard = {}
    source_items = {}
    representative = {}
    for index, shard_items in assigned_by_shard.items():
        groups = {}
        for item in shard_items:
            source_id = str(item["source_id"])
            groups.setdefault(source_id, []).append(item)
            source_items.setdefault(source_id, []).append(item)
        groups_by_shard[index] = groups
        completed = completed_ids_by_shard.get(index, set())
        pending = []
        for source_id, items_for_source in groups.items():
            if any(str(item["image_id"]) not in completed for item in items_for_source):
                pending.append(source_id)
                representative.setdefault(source_id, items_for_source[0])
        pending_by_shard[index] = pending

    uncached = {source_id for source_id, item in representative.items()
                if not source_cache_complete(item)}
    source_order = sorted(source_items,
                          key=lambda source_id: min(str(item["image_id"])
                                                     for item in source_items[source_id]))
    preferred_shards = {}
    shard_count = max(assigned_by_shard, default=-1) + 1
    for group_index, source_id in enumerate(source_order):
        if source_id not in uncached:
            continue
        owners = [index for index in sorted(pending_by_shard)
                  if source_id in pending_by_shard[index]]
        if not owners:
            continue
        target = group_index % shard_count
        preferred_shards[source_id] = target if target in owners else owners[0]

    current_groups = groups_by_shard[shard_index]
    pending_sources = set(pending_by_shard[shard_index])
    uncached_order = [source_id for source_id in source_order
                      if source_id in pending_sources and source_id in uncached]
    preferred_order = [source_id for source_id in uncached_order
                       if preferred_shards[source_id] == shard_index]
    other_uncached_order = [source_id for source_id in uncached_order
                            if preferred_shards[source_id] != shard_index]
    cached_order = [source_id for source_id in source_order
                    if source_id in pending_sources and source_id not in uncached]
    completed_only = [source_id for source_id in source_order
                      if source_id in current_groups and source_id not in pending_sources]
    group_order = cached_order + preferred_order + other_uncached_order + completed_only
    rotated = [item for source_id in group_order for item in current_groups[source_id]]
    starts = {}
    for index, pending in pending_by_shard.items():
        choices = [source_id for source_id in source_order
                   if source_id in pending and source_id in uncached
                   and preferred_shards.get(source_id) == index]
        if not choices:
            choices = [source_id for source_id in source_order
                       if source_id in pending and source_id in uncached]
        if not choices:
            choices = [source_id for source_id in source_order if source_id in pending]
        if choices:
            starts[index] = choices[0]
    return rotated, dict(starts=starts, source_group_order=source_order,
                         preferred_shards=preferred_shards,
                         pending_source_groups=pending_by_shard,
                         uncached_source_groups=sorted(uncached))


def cached_ready_source_groups(items: list[dict], completed: set[str]) -> Iterator[dict]:
    """Recheck shared cache readiness between groups, keeping each group intact."""
    groups = {}
    for item in items:
        groups.setdefault(str(item["source_id"]), []).append(item)
    pending = {}
    for source_id, rows in groups.items():
        if all(str(row["image_id"]) in completed for row in rows):
            yield from rows
        else:
            pending[source_id] = rows
    while pending:
        source_id = next((key for key, rows in pending.items()
                          if source_cache_complete(rows[0])), next(iter(pending)))
        yield from pending.pop(source_id)


def local_source(item: dict) -> Path:
    """Stage an unchanged complete MCD once, keeping decoding outside NAS slots."""
    source = ROOT / item["source_file"]
    if SOURCE_CACHE is not None:
        return shared_source_cache(item)
    if args.direct_native_read:
        stat = source.stat()
        assert dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino) == item["version"]
        return source
    staged = local / "sources" / source.name
    record = staged.with_suffix(".version.json")
    def version(path: Path) -> dict:
        stat = path.stat()
        return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    if record.exists() and staged.exists():
        saved = json.loads(record.read_text())
        if saved["source_version"] == item["version"] and saved["local_version"] == version(staged):
            return staged
    if shutil.disk_usage(local).free < 64 * 1024**3 + item["version"]["size"]:
        raise SpaceUnavailable("Local source staging would violate the 64 GiB reserve")
    staged.parent.mkdir(parents=True, exist_ok=True)
    partial = staged.with_suffix(".mcd.partial")
    copy_record = staged.with_suffix(".copy.json")
    offset = 0
    if partial.exists() and copy_record.exists():
        saved = json.loads(copy_record.read_text())
        if saved["source_version"] == item["version"] and saved["partial_version"] == version(partial):
            offset = partial.stat().st_size
    copy_started = False
    try:
        with formal_read_lease(source):
            with bounded_io("stage_complete_native_source"):
                assert version(source) == item["version"]
                with source.open("rb") as incoming, partial.open("ab" if offset else "wb") as outgoing:
                    copy_started = True
                    incoming.seek(offset)
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
                assert partial.stat().st_size == item["version"]["size"]
                assert version(source) == item["version"]
    finally:
        if copy_started and partial.exists():
            write_json(copy_record, dict(source_version=item["version"], partial_version=version(partial)))
    os.replace(partial, staged)
    copy_record.unlink()
    write_json(record, dict(source_version=item["version"], local_version=version(staged)))
    return staged


def verify_pixels(reader, panel: pd.DataFrame, destination: Path, image_id: str) -> dict:
    """Check all retained channels at native internal/boundary blocks and padding."""
    selected = panel[panel.keep_channel].sort_values("output_channel_index")
    channels = selected.source_channel_index.astype(int).tolist()
    _, h, w = reader.shape
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    blocks = [(0, 0, min(32, w), min(32, h)),
              (w//2, h//2, min(w//2+32, w), min(h//2+32, h))]
    with formal_read_lease(destination):
        with bounded_io("preparer_pixel_check"):
            with tifffile.TiffFile(destination) as tif:
                header(destination, panel, side)
                for window in blocks:
                    source = reader.read_window(channels, window)
                    a, b, c, d = window
                    target_window = (a+left, b+top, c+left, d+top)
                    output = np.stack([read_tiff_window(tif, i, target_window) for i in range(len(channels))])
                    assert source.dtype == output.dtype and np.array_equal(source, output)
                padding = []
                if left:
                    padding.append((0, top, left, min(top+32, side)))
                if top:
                    padding.append((left, 0, min(left+32, side), top))
                right, bottom = side-w-left, side-h-top
                if right:
                    padding.append((side-right, top, side, min(top+32, side)))
                if bottom:
                    padding.append((left, side-bottom, min(left+32, side), side))
                for window in padding:
                    for i in range(len(channels)):
                        assert not read_tiff_window(tif, i, window).any()
            valid_path = destination.with_name("valid_pixels.tiff")
            if valid_path.exists():
                with tifffile.TiffFile(valid_path) as valid:
                    assert valid.series[0].shape == (side, side) and valid.series[0].dtype == np.dtype("uint8")
                    for window in blocks:
                        a, b, c, d = window
                        actual = read_tiff_window(valid, 0, (a+left, b+top, c+left, d+top))
                        assert np.array_equal(actual, reader.valid_window(window).astype("uint8"))
            overview_path = OVERVIEW_CACHE / f"{panel.panel_id.iloc[0]}.png"
            overview = None
            if not overview_path.exists():
                indices = [int(selected[selected.nuclear].source_channel_index.iloc[0])]
                indices += selected[selected.role == "downstream"].source_channel_index.astype(int).tolist()[:3]
                overview = reader.read_window(indices, (0, 0, w, h))
    if overview is not None:
        tiles = []
        for plane in overview:
            positive = plane[plane > 0]
            scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
            tile = Image.fromarray((np.clip(plane / scale, 0, 1)*255).astype("uint8"))
            tile.thumbnail((384, 384))
            tiles.append(tile)
        canvas = Image.new("L", (384*len(tiles), 384))
        for i, tile in enumerate(tiles):
            canvas.paste(tile, (384*i, 0))
        overview_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(overview_path)
        write_json(overview_path.with_suffix(".json"), dict(image_id=image_id,
                   source_channel_indices=indices, display_only="positive P99.5 scaled; native input unchanged"))
    return dict(image_id=image_id, source_blocks=blocks, source_indices=channels,
                padding_blocks=padding, dtype="float32", comparison="all retained channels exactly equal",
                overview_path=relative(overview_path))


def verify_structure(panel: pd.DataFrame, destination: Path, image_id: str, side: int) -> dict:
    """Check the complete output header and validity geometry without rereading source blocks."""
    with formal_read_lease(destination):
        with bounded_io("preparer_structure_check"):
            header(destination, panel, side)
            valid_path = destination.with_name("valid_pixels.tiff")
            if valid_path.exists():
                with tifffile.TiffFile(valid_path) as valid:
                    assert valid.series[0].shape == (side, side)
                    assert valid.series[0].dtype == np.dtype("uint8")
    return dict(image_id=image_id, dtype="float32", side_px=side,
                comparison="header_shape_and_validity_only",
                pixel_equality="pending_stratified_review")


def panel_id_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(value))
    return (int(match.group(1)), str(value)) if match else (10**9, str(value))


def panel_signatures(table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    return {
        str(panel_id): tuple(group.sort_values("source_channel_index").source_name.astype(str))
        for panel_id, group in table.groupby("panel_id", sort=False)
    }


def remap_panel_ids(candidate_panel: pd.DataFrame, formal_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Bind panel IDs to formal source-label signatures without changing pixels."""
    formal_signatures = panel_signatures(formal_panel)
    candidate_signatures = panel_signatures(candidate_panel)
    signature_to_formal = {signature: panel_id for panel_id, signature in formal_signatures.items()}
    mapping = {}
    used = set(formal_signatures)
    next_id = max((panel_id_key(value)[0] for value in used), default=0) + 1
    for panel_id in sorted(candidate_signatures, key=panel_id_key):
        signature = candidate_signatures[panel_id]
        if signature in signature_to_formal:
            mapping[panel_id] = signature_to_formal[signature]
        else:
            while f"panel_{next_id:03d}" in used:
                next_id += 1
            mapping[panel_id] = f"panel_{next_id:03d}"
            used.add(mapping[panel_id])
            next_id += 1
    remapped = candidate_panel.copy()
    remapped["panel_id"] = remapped.panel_id.astype(str).map(mapping)
    return remapped, mapping


def aggregate_shards() -> None:
    """Merge closed shard snapshots incrementally with root-local panel binding."""
    if CANDIDATE_OUTPUT is None:
        raise ValueError("--aggregate-shards requires --candidate-output")
    if args.aggregate_output is not None and args.aggregate_output.resolve() != OUTPUT.resolve():
        raise ValueError("--aggregate-output must equal --candidate-output for an in-place handoff")
    parent_root = OUTPUT.resolve()
    def candidate_path(value: object) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        project_path = PROJECT / path
        return project_path if project_path.exists() else ROOT / path

    def root_snapshot(root: Path, *, required: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        images_path, panel_path = root / "images.parquet", root / "panel_mapping.csv"
        if not images_path.is_file() or not panel_path.is_file():
            if required:
                raise FileNotFoundError(f"closed candidate root is missing tables: {root}")
            return pd.DataFrame(), pd.DataFrame()
        with resource_lock(root, shared=True):
            images = pd.read_parquet(images_path)
            panel = pd.read_csv(panel_path, keep_default_na=False)
        return images, panel

    parent_table, parent_panel = root_snapshot(parent_root, required=True)
    root_snapshots = [("parent", parent_root, parent_table, parent_panel)]
    for index in range(args.shard_count):
        shard_root = parent_root / "_shards" / str(index)
        images, panel = root_snapshot(shard_root, required=False)
        if len(images) or len(panel):
            if not len(panel):
                raise ValueError(f"shard {index} has images but no panel index: {shard_root}")
            root_snapshots.append((str(index), shard_root, images, panel))
    shard_counts = {str(index): 0 for index in range(args.shard_count)}
    shard_counts.update({name: len(images) for name, _, images, _ in root_snapshots if name != "parent"})

    formal_root = args.reuse_existing_formal.resolve() if args.reuse_existing_formal else None
    if formal_root is None or not (formal_root / "panel_mapping.csv").is_file():
        raise FileNotFoundError("--reuse-existing-formal/panel_mapping.csv is required for panel binding")
    with formal_read_lease(formal_root), resource_lock(formal_root, shared=True):
        formal_panel = pd.read_csv(formal_root / "panel_mapping.csv", keep_default_na=False)
    formal_signatures = panel_signatures(formal_panel)
    signature_to_panel = {signature: panel_id for panel_id, signature in formal_signatures.items()}
    used_panel_ids = set(formal_signatures)
    next_panel_number = max((panel_id_key(value)[0] for value in used_panel_ids), default=0) + 1
    remapped_roots = []
    root_panel_maps = {}
    remapped_panel_frames = [formal_panel]
    for name, root, images, panel in root_snapshots:
        local_signatures = panel_signatures(panel)
        local_map = {}
        for local_id in sorted(local_signatures, key=panel_id_key):
            signature = local_signatures[local_id]
            if signature not in signature_to_panel:
                while f"panel_{next_panel_number:03d}" in used_panel_ids:
                    next_panel_number += 1
                signature_to_panel[signature] = f"panel_{next_panel_number:03d}"
                used_panel_ids.add(signature_to_panel[signature])
                next_panel_number += 1
            local_map[local_id] = signature_to_panel[signature]
        mapped_images = images.copy()
        if len(mapped_images):
            mapped_images["panel_id"] = mapped_images.panel_id.astype(str).map(local_map)
            if mapped_images.panel_id.isna().any():
                raise ValueError(f"root {name} contains an image with an unknown panel ID")
        mapped_panel = panel.copy()
        mapped_panel["panel_id"] = mapped_panel.panel_id.astype(str).map(local_map)
        remapped_roots.append((name, root, mapped_images, mapped_panel))
        root_panel_maps[name] = local_map
        remapped_panel_frames.append(mapped_panel)

    def input_revision(row: pd.Series) -> tuple:
        revision = []
        for column in ("input_image_path", "valid_pixels_path"):
            value = row.get(column)
            missing = value is None
            if not missing and not isinstance(value, (list, dict, tuple)):
                missing = bool(pd.isna(value))
            if missing or not str(value):
                revision.append(None)
                continue
            path = candidate_path(value)
            stat = path.stat()
            revision.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        return tuple(revision)

    def same_closed_input(left: pd.Series, right: pd.Series) -> bool:
        if input_revision(left) != input_revision(right):
            return False
        for column in ("source_revision", "source_acquisition_id", "height_px", "width_px", "source_files"):
            left_value, right_value = left.get(column), right.get(column)
            if hasattr(left_value, "tolist"):
                left_value = left_value.tolist()
            if hasattr(right_value, "tolist"):
                right_value = right_value.tolist()
            if column == "source_files" or isinstance(left_value, (list, tuple)) or isinstance(right_value, (list, tuple)):
                if json.dumps(left_value, ensure_ascii=False, sort_keys=True, default=str) != json.dumps(
                        right_value, ensure_ascii=False, sort_keys=True, default=str):
                    return False
            elif str(left_value) != str(right_value):
                return False
        return True

    merged_rows = {}
    duplicate_reused = []
    for name, _, images, _ in remapped_roots:
        if images.image_id.astype(str).duplicated().any():
            raise ValueError(f"duplicate image IDs inside root {name}")
        for row in images.to_dict("records"):
            image_id = str(row["image_id"])
            if image_id not in merged_rows:
                merged_rows[image_id] = row
                continue
            existing = pd.Series(merged_rows[image_id])
            incoming = pd.Series(row)
            if not same_closed_input(existing, incoming):
                raise ValueError(f"conflicting closed inputs for image {image_id}")
            duplicate_reused.append(image_id)
    all_images = pd.DataFrame(list(merged_rows.values()))
    all_images = all_images.sort_values("image_id").reset_index(drop=True)
    remapped_panel = pd.concat(remapped_panel_frames, ignore_index=True)
    remapped_panel = remapped_panel.drop_duplicates(["panel_id", "source_channel_index"], keep="first")
    for row in all_images.to_dict("records"):
        for value in (row["input_image_path"], row.get("valid_pixels_path")):
            if value and not candidate_path(value).is_file():
                raise FileNotFoundError(f"closed aggregate input is unavailable: {value}")

    samples_path = parent_root / "sample_metadata.csv"
    fields_path = parent_root / "field_mapping.csv"
    with resource_lock(parent_root, shared=True):
        samples = pd.read_csv(samples_path, keep_default_na=False)
        fields = pd.read_csv(fields_path, keep_default_na=False)
    source030_added = False
    source030_imported = False
    source030_import_paths = {}
    source030_plan_item = None
    if args.source030_candidate_manifest is not None:
        rescue_path = args.source030_candidate_manifest.resolve()
        rescue = json.loads(rescue_path.read_text())
        review_path = rescue_path.with_name("independent_review.json")
        review = json.loads(review_path.read_text())
        if rescue.get("image_id") != "source_030_acq_0003" or review.get("result") != "candidate_pass":
            raise ValueError("source030 candidate lacks the required independent candidate review")
        image_path = Path(rescue["image_path"])
        valid_path = Path(rescue["valid_pixels_path"])
        raw_path = Path(rescue["canonical_source"])
        raw_stat = raw_path.stat()
        expected_version = rescue["source_version"]
        if {"size": raw_stat.st_size, "mtime_ns": raw_stat.st_mtime_ns, "inode": raw_stat.st_ino} != expected_version:
            raise ValueError("source030 raw source changed since the bounded export")
        if not image_path.is_file() or not valid_path.is_file():
            raise FileNotFoundError("source030 candidate input or valid_pixels is unavailable")
        # Keep the helper export immutable, but place the published candidate assets
        # below this candidate root so downstream validation cannot cross workspaces.
        helper_versions = {
            "input.tiff": (image_path.stat().st_size, image_path.stat().st_mtime_ns),
            "valid_pixels.tiff": (valid_path.stat().st_size, valid_path.stat().st_mtime_ns),
        }
        import_root = parent_root / "_imports" / "source030" / "images" / rescue["image_id"]
        import_root.mkdir(parents=True, exist_ok=True)
        target_image = import_root / "input.tiff"
        target_valid = import_root / "valid_pixels.tiff"
        for source_path, target_path, name in (
                (image_path, target_image, "input.tiff"),
                (valid_path, target_valid, "valid_pixels.tiff")):
            if target_path.exists():
                target_stat = target_path.stat()
                if (target_stat.st_size, target_stat.st_mtime_ns) != helper_versions[name]:
                    raise ValueError(f"source030 imported {name} differs from helper export")
            else:
                shutil.copy2(source_path, target_path)
                target_stat = target_path.stat()
                if (target_stat.st_size, target_stat.st_mtime_ns) != helper_versions[name]:
                    raise ValueError(f"source030 imported {name} was not preserved byte/version-wise")
            source030_import_paths[name] = str(target_path.resolve())
        source030_imported = True
        native_manifest = json.loads((META / "native_manifest.json").read_text())
        source_entry = next(item for item in native_manifest["sources"] if item["source_id"] == "source_030")
        acquisition = next(item for item in source_entry["acquisitions"] if item["acquisition_id"] == 3)
        ordered_names = tuple(rescue["ordered_channel_names"])
        source030_panel_id = signature_to_panel.get(ordered_names)
        if source030_panel_id is None:
            raise ValueError("source030 ordered channels do not match a bound panel signature")
        specimen_records = []
        for name in ("file_list1.json", "file_list2.json"):
            entries = json.loads((META / "identity_sources" / name).read_text())
            specimen_records.extend(
                {attribute["name"]: attribute.get("value", "") for attribute in entry["attributes"]}
                for entry in entries if entry.get("path") == raw_path.name)
        identifiers = {record["ID"] for record in specimen_records}
        description = acquisition["native_metadata"]["Description"]
        matched = [identifier for identifier in identifiers
                   if re.search(r"(?<![A-Z0-9])" + re.escape(identifier) + r"(?![0-9])",
                                description.upper())]
        source030_sample_id = next(iter(identifiers)) if len(identifiers) == 1 else (matched[0] if len(matched) == 1 else "")
        if source030_sample_id and source030_sample_id not in set(samples.sample_id.astype(str)):
            raise ValueError(f"source030 sample {source030_sample_id} is absent from sample_metadata")
        export_result = rescue["export_result"]
        native_meta = acquisition["native_metadata"]
        source030_row = dict.fromkeys(template("image_metadata").columns, None)
        source030_row.update(
            study=STUDY, image_id=rescue["image_id"], sample_id=source030_sample_id or None,
            panel_id=source030_panel_id, image_kind="roi",
            source_image_id=acquisition["canonical_roi_id"], source_files=[relative(raw_path)],
            source_format="mcd", source_acquisition_id="3", source_axes="CYX", source_dtype="float32",
            source_revision="S-BIAD2027:native", slide_id="source_030",
            roi_id=str(native_meta["AcquisitionROIID"]), region_annotation=native_meta["Description"],
            source_x0_px=0, source_y0_px=0, source_x1_px=int(native_meta["MaxX"]),
            source_y1_px=int(native_meta["MaxY"]),
            pixel_size_x_um=float(native_meta["AblationDistanceBetweenShotsX"]),
            pixel_size_y_um=float(native_meta["AblationDistanceBetweenShotsY"]),
            input_image_path=str(target_image.resolve()), valid_pixels_path=str(target_valid.resolve()),
            intensity_description="Source-specific native float32 export; duplicate and missing coordinates zeroed by valid_pixels policy",
            split_method="source_acquisition", split_evidence=relative(rescue_path),
            metadata_notes=("Source030 bounded rescue candidate; only singleton native coordinates retained; "
                            "duplicate and missing coordinates are invalid; independent candidate review passed; "
                            "formal D01 remains pending."))
        for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                    "pad_bottom_px", "source_channel_count", "channel_count"):
            source030_row[key] = export_result[key]
        source030_plan_item = dict(
            image_id=rescue["image_id"], panel_id=source030_panel_id, source_id="source_030",
            source_file=relative(raw_path), proof_file=relative(rescue_path), version=expected_version,
            acquisition=acquisition, component="public", sample_id=source030_sample_id,
            identity_conflict=False, canonical_source_id=acquisition["canonical_roi_id"],
            source_selection_reason=("Independent bounded rescue export from source_030 acquisition 3; "
                                     "singleton native coordinates retained; D01 remains pending."),
            partial=True, required_payload_complete=False,
            partial_provenance={
                "classification": "native duplicate and missing coordinate records",
                "native_record_count": int(review["native_frame"]["raw_records"]),
                "valid_pixel_count": int(rescue["native_valid_pixels"]),
                "invalid_native_pixels": int(rescue["invalid_native_pixels"]),
                "duplicate_rows": int(rescue["duplicate_rows"]),
                "missing_coordinates": int(rescue["missing_coordinates"]),
                "required_payload_complete": False,
                "source_selection_reason": "bounded source030 rescue candidate",
            },
            input_image_path=str(target_image.resolve()),
            valid_pixels_path=str(target_valid.resolve()),
            export_receipt=relative(rescue_path),
        )
        existing_ids = set(all_images.image_id.astype(str))
        if rescue["image_id"] in existing_ids:
            existing_index = all_images.index[all_images.image_id.astype(str) == rescue["image_id"]][0]
            existing_row = all_images.loc[existing_index].copy()
            allowed_input_paths = {str(image_path.resolve()), str(target_image.resolve())}
            allowed_valid_paths = {str(valid_path.resolve()), str(target_valid.resolve())}
            if (str(Path(str(existing_row["input_image_path"])).resolve()) not in allowed_input_paths or
                    str(Path(str(existing_row["valid_pixels_path"])).resolve()) not in allowed_valid_paths):
                raise ValueError("source030 existing row uses an unapproved input or valid_pixels path")
            # Compare content/version metadata after normalizing the old helper path
            # to the candidate-root import path; paths alone are not a conflict here.
            normalized_existing = existing_row.copy()
            normalized_existing["input_image_path"] = str(target_image.resolve())
            normalized_existing["valid_pixels_path"] = str(target_valid.resolve())
            if not same_closed_input(normalized_existing, pd.Series(source030_row)):
                raise ValueError("source030 candidate conflicts with an existing closed image")
            all_images.loc[existing_index, :] = pd.Series(source030_row)
        else:
            all_images = pd.concat([all_images, pd.DataFrame([source030_row])], ignore_index=True)
            source030_added = True

    # The review controller consumes a single complete new-scope plan.  Start
    # from the frozen 575-image plan (not the 228 formal images), enrich it with
    # deterministic shard ownership and closed export paths, then append the
    # separately reviewed source030 candidate when requested.
    plan_root = args.job.parent.parent
    base_plan_path = plan_root / "prepare_ssd" / "source_plan.json"
    if not base_plan_path.is_file():
        raise FileNotFoundError(f"frozen Parsons source plan is unavailable: {base_plan_path}")
    base_plan = json.loads(base_plan_path.read_text())
    base_images = base_plan.get("images", [])
    if len(base_images) != 575:
        raise ValueError(f"expected 575 new Parsons images in frozen source plan, found {len(base_images)}")
    plan_images = {str(item["image_id"]): dict(item) for item in base_images}
    if len(plan_images) != len(base_images):
        raise ValueError("frozen Parsons source plan contains duplicate image IDs")
    shard_plan_paths = []
    shard_assignments = {}
    shard_plan_counts = {}
    for index in range(args.shard_count):
        shard_plan_path = args.job.parent / f"shard_{index}" / "source_plan.json"
        shard_plan_paths.append(relative(shard_plan_path))
        if not shard_plan_path.is_file():
            shard_plan_counts[str(index)] = 0
            continue
        shard_plan = json.loads(shard_plan_path.read_text())
        shard_plan_counts[str(index)] = len(shard_plan.get("images", []))
        for shard_item in shard_plan.get("images", []):
            image_id = str(shard_item["image_id"])
            if image_id not in plan_images:
                raise ValueError(f"shard {index} contains an image outside the frozen new scope: {image_id}")
            base_item = plan_images[image_id]
            if (base_item.get("version") != shard_item.get("version") or
                    base_item.get("source_file") != shard_item.get("source_file") or
                    base_item.get("partial") != shard_item.get("partial") or
                    base_item.get("required_payload_complete") != shard_item.get("required_payload_complete") or
                    base_item.get("acquisition", {}).get("canonical_roi_id") !=
                    shard_item.get("acquisition", {}).get("canonical_roi_id") or
                    base_item.get("acquisition", {}).get("ordered_channels") !=
                    shard_item.get("acquisition", {}).get("ordered_channels")):
                raise ValueError(f"shard {index} changes frozen source evidence for {image_id}")
            if image_id in shard_assignments and shard_assignments[image_id] != index:
                raise ValueError(f"image {image_id} is assigned to multiple shards")
            shard_assignments[image_id] = index
            plan_images[image_id]["assigned_shard"] = index
            plan_images[image_id]["shard_plan"] = relative(shard_plan_path)
            receipt = args.job.parent / f"shard_{index}" / "exports" / f"{image_id}.json"
            if receipt.is_file():
                plan_images[image_id]["export_receipt"] = relative(receipt)
    closed_by_id = {str(row["image_id"]): row for row in all_images.to_dict("records")}
    for image_id, item in plan_images.items():
        item.setdefault("assigned_shard", None)
        item.setdefault("shard_plan", None)
        item.setdefault("export_receipt", None)
        if not item["export_receipt"]:
            for receipt_root in (
                    plan_root / "prepare_ssd" / "exports",
                    plan_root / "prepare_candidate_797" / "exports"):
                receipt = receipt_root / f"{image_id}.json"
                if receipt.is_file():
                    item["export_receipt"] = relative(receipt)
                    break
        row = closed_by_id.get(image_id)
        if row is not None:
            item["input_image_path"] = row.get("input_image_path")
            item["valid_pixels_path"] = row.get("valid_pixels_path")
            item["candidate_state"] = "closed_in_parent_or_shard"
        else:
            item["candidate_state"] = "pending_export"
    if source030_plan_item is not None:
        source030_id = str(source030_plan_item["image_id"])
        if source030_id in plan_images:
            raise ValueError(f"source030 image already exists in the frozen new scope: {source030_id}")
        source030_plan_item["candidate_state"] = "closed_in_parent_or_shard"
        plan_images[source030_id] = source030_plan_item
    aggregate_plan = dict(base_plan)
    aggregate_plan.update(
        images=[plan_images[image_id] for image_id in sorted(plan_images)],
        aggregate_parent=str(parent_root),
        frozen_base_plan=relative(base_plan_path),
        frozen_base_image_count=len(base_images),
        shard_plan_paths=shard_plan_paths,
        shard_plan_counts=shard_plan_counts,
        shard_assignment_count=len(shard_assignments),
        source030_included=source030_plan_item is not None,
        aggregate_image_count=len(plan_images),
        aggregate_time=datetime.now(timezone.utc).isoformat(),
    )
    write_json(args.job / "source_plan.json", aggregate_plan)
    save_input_tables(parent_root, all_images, samples, remapped_panel, fields)
    receipt = dict(study=STUDY, parent_root=str(parent_root), shard_count=args.shard_count,
                   parent_image_count=len(parent_table), shard_image_counts=shard_counts,
                   aggregate_image_count=len(all_images), duplicate_reused_count=len(duplicate_reused),
                   duplicate_reused_image_ids=sorted(set(duplicate_reused)), root_panel_maps=root_panel_maps,
                   source030_added=source030_added, source030_imported=source030_imported,
                   source030_import_paths=source030_import_paths,
                   source_pixels_unchanged=True, closed_inputs_only=True,
                   aggregate_time=datetime.now(timezone.utc).isoformat())
    write_json(args.job / "aggregate_receipt.json", receipt)
    target = parent_root / "review_results.csv"
    reviews = pd.read_csv(target, keep_default_na=False) if target.is_file() else template("review_results")
    detail = (f"Shard boundary aggregate: parent={len(parent_table)}, "
              f"shards={sum(shard_counts.values())}, total={len(all_images)}; "
              "panel IDs remapped by formal source-label signatures; input pixels unchanged."
              + (" source030 candidate included; D01 remains pending." if source030_added else ""))
    review_mask = ((reviews.preparer.astype(str) == args.job.name) &
                   (reviews.check_id.astype(str) == "S01") &
                   reviews.details.astype(str).str.startswith("Shard boundary aggregate:"))
    review_row = [STUDY, args.job.name, "", "study", STUDY, "S01", "unresolved",
                  relative(args.job / "aggregate_receipt.json"), relative(parent_root), detail]
    if review_mask.any():
        reviews.loc[review_mask, :] = review_row
    else:
        reviews.loc[len(reviews)] = review_row
    temporary = target.with_suffix(".csv.partial")
    with formal_publish_lease(target, directory=parent_root):
        reviews.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)


try:
    if args.aggregate_shards:
        aggregate_shards()
        raise SystemExit(0)
    if args.repair_preservation_only:
        repair_preservation()
        raise SystemExit(0)
    # Freeze small metadata and accepted proofs under a shared read lease.
    with formal_read_lease(META):
        with bounded_io("read_source_metadata"):
            manifest = json.loads((META / "native_manifest.json").read_text())
            proofs = native_preparation_sources(manifest, json.loads(NATIVE_FACTS.read_text()))
            shutil.copyfile(S1, local / "s1.csv")
            source_metadata_version = {p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                       for p in (META / "native_manifest.json", S1)}
    canonical = manifest["canonical_source_by_roi"]
    proof_by_id = {Path(name).stem: proof for name, proof in proofs}
    manifest_by_source = {entry["source_id"]: entry for entry in manifest["sources"]}
    manifest_by_roi = {}
    for entry in manifest["sources"]:
        for acquisition in entry["acquisitions"]:
            manifest_by_roi.setdefault(acquisition["canonical_roi_id"], []).append(
                (entry["source_id"], acquisition))
    candidates, source_accounting = [], []
    formal_root = args.reuse_existing_formal.resolve() if args.reuse_existing_formal else None
    layouts = []
    if formal_root is not None and (formal_root / "panel_mapping.csv").is_file():
        with formal_read_lease(formal_root):
            with bounded_io("read_formal_panel_layouts"):
                formal_panel_seed = pd.read_csv(formal_root / "panel_mapping.csv", keep_default_na=False)
        for panel_id, group in sorted(formal_panel_seed.groupby("panel_id", sort=False),
                                      key=lambda item: panel_id_key(item[0])):
            layouts.append(group.sort_values("source_channel_index").source_name.astype(str).tolist())
    with formal_read_lease(META):
        with bounded_io("read_specimen_filelists"):
            filelists = [json.loads((META / "identity_sources" / name).read_text())
                         for name in ("file_list1.json", "file_list2.json")]
            identity_conflicts = json.loads((META / "identity_conflicts.json").read_text())
    specimens_by_file = {}
    for entry in filelists[0] + filelists[1]:
        attributes = {a["name"]: a.get("value", "") for a in entry["attributes"]}
        specimens_by_file.setdefault(entry["path"], []).append(attributes)
    specimen_records = {}
    for proof_name, proof in proofs:
        source_id = Path(proof_name).stem
        path = RAW / proof["source"]["name"]
        stat = path.stat()
        version = proof["file_stat"]
        assert stat.st_size == version["size"] == proof["source"]["size"]
        assert stat.st_mtime_ns == version["mtime_ns"] and stat.st_ino == version["inode"]
        source_accounting.append(dict(source_id=source_id, source=proof["source"],
                                      accepted=len(proof["acquisitions"]),
                                      exclusions=proof.get("exclusions", []), pending=proof.get("pending", []),
                                      preparation_exclusions=[], canonical_alternates=[],
                                      partial_included=[]))
        records = specimens_by_file[path.name]
        for record in records:
            specimen_records.setdefault(record["ID"], []).append(record)
        for acquisition in proof["acquisitions"]:
            canonical_source_id = canonical[acquisition["canonical_roi_id"]]
            selection_reason = "manifest canonical source proof"
            if canonical_source_id != source_id:
                owner_proof = proof_by_id.get(canonical_source_id)
                owner_has_roi = owner_proof is not None and any(
                    x["canonical_roi_id"] == acquisition["canonical_roi_id"]
                    for x in owner_proof.get("acquisitions", []))
                if owner_has_roi or not acquisition.get("required_payload_complete") or not acquisition.get("all_finite"):
                    continue
                selection_reason = (
                    f"accepted alternate proof source {source_id}; manifest canonical owner "
                    f"{canonical_source_id} has no matching proof acquisition")
                source_accounting[-1]["canonical_alternates"].append(dict(
                    canonical_roi_id=acquisition["canonical_roi_id"],
                    canonical_source_id=canonical_source_id,
                    selected_source_id=source_id,
                    reason=selection_reason))
            assert acquisition["required_payload_complete"] and acquisition["all_finite"]
            description = acquisition["native_metadata"]["Description"]
            if args.exclude_calibration and re.search(r"test|laser power|calibr|pap pen", description, re.IGNORECASE):
                source_accounting[-1]["preparation_exclusions"].append(dict(
                    acquisition_id=acquisition["id"], description=description,
                    reason="Explicit native test/laser-power calibration annotation"))
                continue
            identifiers = {record["ID"] for record in records}
            matched = [identifier for identifier in identifiers
                       if re.search(r"(?<![A-Z0-9])" + re.escape(identifier) + r"(?![0-9])",
                                    description.upper())]
            sample_id = next(iter(identifiers)) if len(identifiers) == 1 else (matched[0] if len(matched) == 1 else "")
            identity_conflict = path.name in identity_conflicts["uncertain_source_files"]
            if identity_conflict:
                sample_id = ""
            names = acquisition["ordered_channels"]
            if names not in layouts:
                layouts.append(names)
            candidates.append(dict(image_id=f"{source_id}_acq_{acquisition['id']:04d}",
                                   panel_id=f"panel_{layouts.index(names)+1:03d}",
                                   source_id=source_id, source_file=relative(path),
                                   proof_file=relative(NATIVE_FACTS), version=version,
                                   acquisition=acquisition, component=proof["source"]["component"],
                                   sample_id=sample_id, identity_conflict=identity_conflict,
                                   canonical_source_id=canonical_source_id,
                                   source_selection_reason=selection_reason,
                                   partial=False, required_payload_complete=True))
    complete_rois = {item["acquisition"]["canonical_roi_id"] for item in candidates}
    # Add only nonzero native-record exclusions whose canonical ROI has no complete source.
    partial_by_roi = {}
    for proof_name, proof in proofs:
        source_id = Path(proof_name).stem
        for exclusion in proof.get("exclusions", []):
            if int(exclusion.get("record_count", 0) or 0) <= 0:
                continue
            roi = exclusion["canonical_roi_id"]
            if roi in complete_rois or roi in partial_by_roi:
                continue
            canonical_source_id = canonical[roi]
            source_entry = manifest_by_source.get(canonical_source_id)
            if source_entry is None:
                raise ValueError(f"No manifest source for partial canonical ROI {roi}")
            matches = [x for x in source_entry["acquisitions"] if x["canonical_roi_id"] == roi]
            if len(matches) != 1:
                raise ValueError(f"Expected one manifest acquisition for partial canonical ROI {roi}")
            acquisition = dict(matches[0])
            acquisition["id"] = acquisition.get("id", acquisition.get("acquisition_id"))
            selected_proof_name = f"{canonical_source_id}.json"
            selected_proof = proof_by_id.get(canonical_source_id)
            if selected_proof is None:
                raise ValueError(f"No source proof for partial canonical source {canonical_source_id}")
            path = RAW / selected_proof["source"]["name"]
            version = selected_proof["file_stat"]
            stat = path.stat()
            assert stat.st_size == version["size"] == selected_proof["source"]["size"]
            assert stat.st_mtime_ns == version["mtime_ns"] and stat.st_ino == version["inode"]
            records = specimens_by_file[path.name]
            description = acquisition["native_metadata"]["Description"]
            identifiers = {record["ID"] for record in records}
            matched = [identifier for identifier in identifiers
                       if re.search(r"(?<![A-Z0-9])" + re.escape(identifier) + r"(?![0-9])",
                                    description.upper())]
            sample_id = next(iter(identifiers)) if len(identifiers) == 1 else (matched[0] if len(matched) == 1 else "")
            identity_conflict = path.name in identity_conflicts["uncertain_source_files"]
            if identity_conflict:
                sample_id = ""
            names = acquisition["ordered_channels"]
            if names not in layouts:
                layouts.append(names)
            partial_info = dict(
                classification=exclusion.get("classification", ""),
                native_record_count=int(exclusion["record_count"]),
                required_record_count=int(exclusion["required_records"]),
                data_start=int(exclusion["data_start"]), data_end=int(exclusion["data_end"]),
                stride=int(exclusion["stride"]), required_payload_complete=False,
                proof_source_id=source_id, canonical_source_id=canonical_source_id,
                source_selection_reason=(
                    "nonzero native-record partial source; no complete accepted canonical alternative"))
            item = dict(image_id=f"{canonical_source_id}_acq_{acquisition['id']:04d}",
                        panel_id=f"panel_{layouts.index(names)+1:03d}", source_id=canonical_source_id,
                        source_file=relative(path), proof_file=relative(NATIVE_FACTS),
                        version=version, acquisition=acquisition,
                        component=selected_proof["source"]["component"], sample_id=sample_id,
                        identity_conflict=identity_conflict, canonical_source_id=canonical_source_id,
                        source_selection_reason=partial_info["source_selection_reason"], partial=True,
                        required_payload_complete=False, partial_provenance=partial_info)
            candidates.append(item)
            partial_by_roi[roi] = item
            accounting = next(x for x in source_accounting if x["source_id"] == canonical_source_id)
            accounting["partial_included"].append(partial_info)
    assert not complete_rois.intersection(partial_by_roi)
    all_candidate_by_id = {str(item["image_id"]): item for item in candidates}
    if len(all_candidate_by_id) != len(candidates):
        raise ValueError("candidate scope contains duplicate image IDs")
    reused_formal_ids = set()
    if formal_root is not None:
        with formal_read_lease(formal_root):
            with bounded_io("read_existing_formal_scope"):
                formal_images = pd.read_parquet(formal_root / "images.parquet")
        reused_formal_ids = set(formal_images["image_id"].astype(str))
        if not SHARD_MODE:
            candidates = [item for item in candidates if item["image_id"] not in reused_formal_ids]
    parent_candidate_ids = set()
    assigned_candidate_count = 0
    parent_skipped_in_assignment = 0
    formal_skipped_in_assignment = 0
    if SHARD_MODE:
        with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
            with bounded_io("read_parent_candidate_scope"):
                parent_images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
        parent_candidate_ids = set(parent_images["image_id"].astype(str)) if len(parent_images) else set()
        # Assign the immutable full new scope first; parent closures are only skipped afterward.
        ordered = sorted(candidates, key=lambda item: item["image_id"])
        assigned_by_shard = {}
        for index in range(args.shard_count):
            raw_assigned = [item for position, item in enumerate(ordered)
                            if position % args.shard_count == index]
            assigned_by_shard[index] = [item for item in raw_assigned
                                        if item["image_id"] not in reused_formal_ids
                                        and item["image_id"] not in parent_candidate_ids]
            if index == args.shard_index:
                assigned = raw_assigned
        assigned_candidate_count = len(assigned)
        formal_skipped_in_assignment = sum(item["image_id"] in reused_formal_ids for item in assigned)
        parent_skipped_in_assignment = sum(item["image_id"] in parent_candidate_ids for item in assigned)
        candidates = assigned_by_shard[args.shard_index]
    panels = pd.concat([panel_from_s1(STUDY, f"panel_{i+1:03d}", names, local / "s1.csv")
                        for i, names in enumerate(layouts)], ignore_index=True)
    projected = sum(max(int(x["acquisition"]["native_metadata"]["MaxX"]),
                            int(x["acquisition"]["native_metadata"]["MaxY"])) ** 2
                    * int(panels.loc[panels.panel_id == x["panel_id"], "keep_channel"].sum()) * 5
                    for x in candidates)
    local_capacity_root = SOURCE_CACHE if SOURCE_CACHE is not None else local
    capacity = dict(estimated_output_and_valid_upper_bytes=projected,
                    nas_free_bytes=shutil.disk_usage(ROOT).free,
                    local_free_bytes=shutil.disk_usage(local_capacity_root).free,
                    nas_reserve_bytes=100 * 1024**3, local_reserve_bytes=64 * 1024**3)
    if capacity["nas_free_bytes"] <= projected + capacity["nas_reserve_bytes"]:
        raise SpaceUnavailable("Estimated output would violate the NAS 100 GiB reserve")
    if SOURCE_CACHE is not None:
        cache_scope = [item for image_id, item in all_candidate_by_id.items()
                       if image_id not in reused_formal_ids
                       and image_id not in parent_candidate_ids]
        source_cache_versions = {}
        for item in cache_scope:
            source_id = str(item["source_id"])
            version = (item["source_file"], item["version"])
            if source_id in source_cache_versions and source_cache_versions[source_id] != version:
                raise ValueError(f"source_id maps to multiple source versions: {source_id}")
            source_cache_versions[source_id] = version
        capacity["local_source_cache"] = str(SOURCE_CACHE)
        capacity["local_source_cache_scope_count"] = len(cache_scope)
        capacity["local_source_cache_source_count"] = len(source_cache_versions)
        capacity["local_source_staging_upper_bytes"] = sum(
            version["size"] for _, version in source_cache_versions.values())
    else:
        capacity["local_source_staging_upper_bytes"] = (0 if args.direct_native_read else
                                                         sum(proof["source"]["size"] for _, proof in proofs))
    if capacity["local_free_bytes"] <= capacity["local_reserve_bytes"] + capacity["local_source_staging_upper_bytes"] + 4 * 1024**3:
        raise SpaceUnavailable("Estimated native staging would violate the local 64 GiB reserve")
    exclusion_rows = [(Path(name).stem, entry)
                      for name, proof in proofs for entry in proof.get("exclusions", [])]
    exclusion_rois = {entry["canonical_roi_id"] for _, entry in exclusion_rows}
    nonzero_partial_rois = set(partial_by_roi)
    zero_record_rois = {roi for roi in exclusion_rois if roi not in nonzero_partial_rois}
    pending_rows = [(Path(name).stem, entry)
                    for name, proof in proofs for entry in proof.get("pending", [])]
    scope_gaps = dict(
        required_roi_count=len(manifest["required_roi_ids"]),
        complete_accepted_canonical_roi_count=len(complete_rois),
        complete_accepted_rois=sorted(complete_rois),
        nonzero_partial_roi_count=len(nonzero_partial_rois),
        nonzero_partial_rois=sorted(nonzero_partial_rois),
        zero_record_exclusion_roi_count=len(zero_record_rois),
        zero_record_exclusion_rois=sorted(zero_record_rois),
        all_exclusion_roi_count=len(exclusion_rois),
        pending_count=len(pending_rows), pending=pending_rows,
        canonical_alternate_count=sum(bool(x.get("canonical_alternates")) for x in source_accounting),
        canonical_alternates=[x for row in source_accounting for x in row["canonical_alternates"]],
        unresolved_source_scope=(
            "All proof exclusions and pending records are retained here. Zero-record exclusions "
            "are not exported as input; nonzero partial records are exported with valid_pixels and "
            "required_payload_complete=false. Pending proof records remain outside candidate scope."))
    scope_gaps_path = args.job / "source_scope_gaps.json"
    write_json(scope_gaps_path, scope_gaps)
    source_plan_path = args.job / "source_plan.json"
    write_json(source_plan_path, dict(images=candidates, source_accounting=source_accounting,
                                      metadata_version=source_metadata_version, capacity=capacity,
                                      reused_formal_images=sorted(reused_formal_ids),
                                      excluded_calibration=bool(args.exclude_calibration),
                                      scope_gaps_path=relative(scope_gaps_path),
                                      complete_accepted_canonical_roi_count=len(complete_rois),
                                      nonzero_partial_roi_count=len(nonzero_partial_rois)))
    state.update(expected_images=len(candidates), whole_study_required_rois=len(manifest["required_roi_ids"]),
                 reused_formal_images=len(reused_formal_ids), capacity=capacity,
                 candidate_output=str(OUTPUT), excluded_calibration=bool(args.exclude_calibration),
                 complete_accepted_canonical_roi_count=len(complete_rois),
                 nonzero_partial_roi_count=len(nonzero_partial_rois),
                 pending_source_count=len(pending_rows), all_exclusion_roi_count=len(exclusion_rois),
                 shard_count=args.shard_count, shard_index=args.shard_index,
                 parent_candidate_images=len(parent_candidate_ids),
                 assigned_candidate_count=assigned_candidate_count,
                 formal_skipped_in_assignment=formal_skipped_in_assignment,
                 parent_skipped_in_assignment=parent_skipped_in_assignment)
    if args.plan_only:
        progress("planned", status="planned", remaining_image_ids=[x["image_id"] for x in candidates],
                 no_image_exports=True)
        raise SystemExit(0)

    # File-list specimen IDs establish biological samples, independently of patient keys.
    sample_rows = []
    sample_columns = template("sample_metadata").columns
    clinical_fields = {"sample_type": "SAMPLE TYPE (CORE/RESECTION)",
                       "preservation_method": "FORMAT (FFPE/FROZEN)",
                       "nact_response_source": "OUTCOME (RESPONDER, NON-RESPONDER)",
                       "rcb_group_source": "RCB group", "source_matched_with": "Matched with?"}
    for identifier, records in sorted(specimen_records.items()):
        row = dict.fromkeys(sample_columns, "")
        row.update(study=STUDY, sample_id=identifier, source_sample_id=identifier)
        conflicts = []
        for target, source in clinical_fields.items():
            values = {record.get(source, "") for record in records} - {""}
            row[target] = next(iter(values)) if len(values) == 1 else ""
            if target == "preservation_method":
                row[target] = PRESERVATION_MAPPING.get(row[target], "")
            if len(values) > 1:
                conflicts.append(source)
        row["metadata_notes"] = "Official specimen ID/file attributes; patient identity and clinical time origins unresolved. Matched-with retained without patient-equivalence inference."
        if conflicts:
            row["metadata_notes"] += " Conflicting source fields left blank: " + "; ".join(conflicts)
        sample_rows.append(row)
    samples = pd.DataFrame(sample_rows)
    fields = pd.DataFrame([
        field(relative(META / "native_manifest.json"), "canonical_source_by_roi", "image_metadata",
              "image_id;source_image_id;source_files;source_revision",
              "One canonical exact native acquisition per ROI; source-qualified acquisition IDs; unchanged source proof versions required."),
        field(relative(NATIVE_FACTS), "sources.file_stat;accepted_acquisitions;exclusions;pending;reused_proofs", "image_metadata",
              "image_id;source_files;source_revision",
              "Reuse recorded native source versions and finite complete acquisition acceptance; preserve exclusion, pending and canonical reuse scope facts."),
        field(relative(META / "native_manifest.json"), "native_metadata;ordered_channels", "image_metadata",
              "source_acquisition_id;roi_id;region_annotation;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um",
              "Preserve exact acquisition Description, full native MaxX/MaxY frame, and recorded shot distances; no physical-coordinate unit inference."),
        field(relative(S1), "Original marker;Unified marker;Segmentation channel", "panel_mapping", "source_name;marker_id;role;keep_channel;output_channel_index",
              "Exact ordered native label joins to current study S1; preserve both selected DNA measurements and 35 protein measurements in source order."),
        field(relative(META / "native_manifest.json"), "native_metadata.MaxX;MaxY;DataStartOffset;DataEndOffset;ordered_channels", "image_metadata",
              "height_px;width_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;source_axes;source_dtype;source_channel_count;channel_count;input_image_path;valid_pixels_path",
              "Full native acquisition in float32 CYX with no intensity transformation. L=max(MaxX,MaxY); left=(L-MaxX)//2, right=L-MaxX-left, top=(L-MaxY)//2, bottom=L-MaxY-top. Shared exporter retains selected channels in native order; zero padding; optional uint8 validity marks acquisition coordinates, never intensity-derived. Output is per stable image ID."),
        field(relative(META / "identity_search.json"), "sources_checked;unresolved_bridge", "sample_metadata", "sample_id;patient_key",
              "Official filelist ID establishes specimen identity when a container has one ID; multi-specimen containers require exact specimen-ID match in native Description. Unresolved multi-specimen ROIs and all patient keys remain blank."),
        field(relative(META / "identity_conflicts.json"), "uncertain_source_files;source_evidence;finding", "image_metadata", "sample_id",
              "Leave the two containers with author processed-FOV label corrections unlinked pending an exact raw-acquisition/processed-FOV namespace bridge. Retain native images and all source annotations."),
    ])
    for target, source in {"sample_id": "ID", "source_sample_id": "ID", **clinical_fields}.items():
        for name in ("file_list1.json", "file_list2.json"):
            fields.loc[len(fields)] = field(relative(META / "identity_sources" / name), source,
                "sample_metadata", target, "Preserve source specimen attribute verbatim; consistent duplicate values coalesced; conflicting nonblank values left blank. Source R/NR is not RECIST best overall response; RCB is not a survival endpoint.")
            if target == "preservation_method":
                fields.loc[fields.index[-1], "value_mapping"] = json.dumps(PRESERVATION_MAPPING)
                fields.loc[fields.index[-1], "transformation_rule"] = PRESERVATION_RULE
            if target == "source_matched_with":
                fields.loc[fields.index[-1], "transformation_rule"] = (
                    "Preserve the exact source pointer to another sample; do not interpret it as patient equivalence. "
                    "Consistent duplicate values are coalesced; conflicting nonblank values remain blank."
                )
                fields.loc[fields.index[-1], "notes"] = json.dumps(dict(
                    data_type="string",
                    definition="Verbatim official file-list pointer to another sample; it does not identify the same patient."
                ))
            if target in {"nact_response_source", "rcb_group_source"}:
                fields.loc[fields.index[-1], "notes"] = json.dumps(dict(
                    data_type="string", definition=f"Verbatim official file-list categorical attribute: {source}; specimen-level source annotation; clinical index/time origin unknown"))
    with formal_read_lease(TABLE_OUTPUT):
        with bounded_io("read_existing_index"):
            images = pd.read_parquet(TABLE_OUTPUT / "images.parquet").to_dict("records") if (TABLE_OUTPUT / "images.parquet").exists() else []
            # Existing shard rows can already be present in the formal/parent snapshot
            # after an incremental aggregate.  The filtered `candidates` list controls
            # new work only; closed rows must resolve against the immutable full scope.
            by_id = all_candidate_by_id
            verified_existing = set()
            for row in images:
                item = by_id[row["image_id"]]
                # Rebind management metadata to the stable formal layout; the closed TIFF is untouched.
                row["panel_id"] = item["panel_id"]
                receipt = json.loads((EXPORT_CACHE / f"{item['image_id']}.json").read_text())
                selected_panel = panels[panels.panel_id == item["panel_id"]]
                selected = selected_panel[selected_panel.keep_channel].sort_values("output_channel_index")
                assert receipt["source_version"] == item["version"]
                assert receipt["source_indices"] == selected.source_channel_index.tolist()
                header(ROOT / row["input_image_path"], selected_panel, int(row["height_px"]))
                if (PIXEL_CHECK_CACHE / f"{item['image_id']}.json").exists():
                    verified_existing.add(item["image_id"])
    sample_by_image = {image_id: item["sample_id"] for image_id, item in all_candidate_by_id.items()}
    for row in images:
        row["sample_id"] = sample_by_image[row["image_id"]] or None
        item = by_id[row["image_id"]]
        row["metadata_notes"] = ("Source component " + item["component"] + "; "
                                  + item["source_selection_reason"] + "; patient bridge unresolved; "
                                  "specimen linked only by official file-list identity or exact Description match. "
                                  "Container-qualified slide record; physical slide/core equivalence unknown.")
        if item.get("partial"):
            row["metadata_notes"] += " Partial native records; valid_pixels marks the observed acquisition domain and required_payload_complete remains false."
    state.update(sample_rows=len(samples), sample_linked_images=sum(bool(row.get("sample_id")) for row in images))
    save_tables()
    completed = {x["image_id"]: x for x in images}
    if SHARD_MODE:
        completed_ids_by_shard = {args.shard_index: set(completed)}
        for index in range(args.shard_count):
            if index == args.shard_index:
                continue
            shard_root = OUTPUT / "_shards" / str(index)
            shard_table = shard_root / "images.parquet"
            if not shard_table.is_file():
                completed_ids_by_shard[index] = set()
                continue
            with formal_read_lease(shard_root):
                with bounded_io("read_existing_shard_completion"):
                    completed_ids_by_shard[index] = set(
                        pd.read_parquet(shard_table, columns=["image_id"]).image_id.astype(str))
        candidates, source_rotation = rotate_unfinished_source_groups(
            assigned_by_shard, completed_ids_by_shard, args.shard_index)
        state["source_rotation"] = source_rotation
    if not SHARD_MODE:
        # Publish a preparer handoff with all unresolved boundaries, never an independent D01 pass.
        reviews = template("review_results")
        for check, scope, details in [
            ("S01", "study", f"{len(candidates)} qualified canonical local acquisitions; {len(reused_formal_ids)} existing formal images reused; full required scope {len(manifest['required_roi_ids'])}; {len(partial_by_roi)} nonzero partial sources retained with required_payload_complete=false; pending and exclusions in source_scope_gaps.json"),
            ("M01", "sample", f"{len(samples)} known source specimens; {sum(bool(x['sample_id']) for x in candidates)} planned image/specimen links; patient keys unresolved. Multi-specimen containers and author corrected-label namespace conflicts lack direct ROI bridges. Search in identity_search.json and identity_conflicts.json."),
            ("D01", "study", "Preparer delivery only; independent review and final image count reconciliation pending.")]:
            reviews.loc[len(reviews)] = [STUDY, preparer, "", scope, STUDY, check, "unresolved",
                                        relative(source_plan_path), relative(OUTPUT), details]
        with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
            with bounded_io("publish_preparer_review"):
                target = OUTPUT / "review_results.csv"
                prior = pd.read_csv(target, keep_default_na=False) if target.exists() else template("review_results")
                temporary = target.with_suffix(".csv.partial")
                pd.concat([prior, reviews], ignore_index=True).to_csv(temporary, index=False, encoding="utf-8-sig")
                os.replace(temporary, target)

    ordered_candidates = (cached_ready_source_groups(candidates, set(completed))
                          if SHARD_MODE else candidates)
    state["dynamic_cached_source_priority"] = SHARD_MODE
    for item in ordered_candidates:
        acquisition = item["acquisition"]
        meta = acquisition["native_metadata"]
        side = max(int(meta["MaxX"]), int(meta["MaxY"]))
        panel = panels[panels.panel_id == item["panel_id"]]
        selected = panel[panel.keep_channel].sort_values("output_channel_index")
        destination = IMAGE_OUTPUT / "images" / item["image_id"] / "input.tiff"
        receipt = EXPORT_CACHE / f"{item['image_id']}.json"
        state["current_image"] = item["image_id"]
        if item["image_id"] in verified_existing:
            continue
        if item["image_id"] in completed:
            with formal_read_lease(destination):
                with bounded_io("verify_existing_input"):
                    saved = json.loads(receipt.read_text())
                    assert saved["source_version"] == item["version"]
                    assert saved["source_indices"] == selected.source_channel_index.tolist()
                    header(destination, panel, side)
            spec = SourceSpec(item["image_id"], item["panel_id"], "mcd", ROOT / item["source_file"],
                              tuple(range(len(acquisition["ordered_channels"]))),
                              acquisition_id=acquisition["id"], source_revision="S-BIAD2027:native")
            with ExitStack() as stack:
                with formal_read_lease(spec.path):
                    with bounded_io("open_existing_source"):
                        reader = stack.enter_context(open_source(spec))
                check = verify_pixels(reader, panel, destination, item["image_id"])
            write_json(PIXEL_CHECK_CACHE / f"{item['image_id']}.json", check)
            continue
        staged_source = local_source(item)
        image_upper_bytes = side * side * (len(selected) * 4 + 1)
        if shutil.disk_usage(ROOT).free < 100 * 1024**3 + image_upper_bytes:
            raise SpaceUnavailable("Image publication would violate the NAS 100 GiB reserve")
        if shutil.disk_usage(local).free < 64 * 1024**3 + image_upper_bytes:
            raise SpaceUnavailable("Image staging would violate the local 64 GiB reserve")
        spec = SourceSpec(item["image_id"], item["panel_id"], "mcd", staged_source,
                          tuple(range(len(acquisition["ordered_channels"]))),
                          acquisition_id=acquisition["id"], source_revision="S-BIAD2027:native")
        with ExitStack() as stack:
            with formal_read_lease(spec.path):
                with bounded_io("open_native_source"):
                    reader = stack.enter_context(open_source(spec))
                    assert reader._acquisition.channel_labels == acquisition["ordered_channels"]
                    assert reader.shape == (len(acquisition["ordered_channels"]), int(meta["MaxY"]), int(meta["MaxX"]))
            if destination.exists() and receipt.exists():
                with formal_read_lease(destination):
                    with bounded_io("resume_published_input"):
                        saved = json.loads(receipt.read_text())
                        assert saved["source_version"] == item["version"]
                        assert saved["source_indices"] == selected.source_channel_index.tolist()
                        header(destination, panel, side)
                        stats = saved["stats"]
            else:
                # export_input owns its NAS admission leases, including queue wait;
                # no outer alarm should turn a legal wait into a retryable failure.
                progress("export_native_input")
                stats = export_input(reader, panel, destination, tile_size=256)
                write_json(receipt, dict(source_version=item["version"],
                                         source_indices=selected.source_channel_index.tolist(), stats=stats))
            if stats["admitted"]:
                check = (verify_pixels(reader, panel, destination, item["image_id"])
                         if args.full_pixel_check else verify_structure(panel, destination,
                                                                         item["image_id"], side))
                write_json(PIXEL_CHECK_CACHE / f"{item['image_id']}.json", check)
        write_json(receipt, dict(source_version=item["version"],
                                 source_indices=selected.source_channel_index.tolist(), stats=stats))
        if not stats["admitted"]:
            progress("excluded_zero_marker", excluded_image=item["image_id"])
            continue
        row = dict.fromkeys(template("image_metadata").columns, None)
        row.update(study=STUDY, image_id=item["image_id"], sample_id=item["sample_id"] or None,
                   panel_id=item["panel_id"], image_kind="roi",
                   source_image_id=acquisition["canonical_roi_id"], source_files=[item["source_file"]],
                   source_format="mcd", source_acquisition_id=str(acquisition["id"]),
                   source_axes="CYX", source_dtype="float32", source_revision="S-BIAD2027:native",
                   slide_id=item["source_id"], roi_id=str(meta["AcquisitionROIID"]),
                   region_annotation=meta["Description"], source_x0_px=0, source_y0_px=0,
                   source_x1_px=int(meta["MaxX"]), source_y1_px=int(meta["MaxY"]),
                   pixel_size_x_um=float(meta["AblationDistanceBetweenShotsX"]),
                   pixel_size_y_um=float(meta["AblationDistanceBetweenShotsY"]),
                   input_image_path=relative(destination),
                   valid_pixels_path=relative(destination.with_name(stats["valid_pixels"])) if stats["valid_pixels"] else None,
                   intensity_description="Native float32 Dual acquisition signal; no normalization; measured zero retained",
                   split_method="source_acquisition", split_evidence=item["proof_file"],
                   metadata_notes=("Source component " + item["component"] + "; "
                                   + item["source_selection_reason"] + "; patient bridge unresolved; "
                                   "specimen linked only by official file-list identity or exact Description match. "
                                   "Container-qualified slide record; physical slide/core equivalence unknown."
                                   + (" Partial native records; valid_pixels marks the observed acquisition domain and required_payload_complete remains false."
                                      if item.get("partial") else "")))
        for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"):
            row[key] = stats[key]
        images.append(row)
        state["completed_images"] = len(images)
        write_json(args.job / "closed_pending_index.json", dict(images=images[state.get("indexed_images", 0):]))
        if len(images) % 3 == 0:
            save_tables()
        completed[item["image_id"]] = row
        state["sample_linked_images"] = sum(bool(row.get("sample_id")) for row in images)
        progress("image_closed_and_checkpointed", last_completed_image=item["image_id"])
        if STOP_MARKER is not None and STOP_MARKER.is_file():
            save_tables()
            write_json(args.job / "closed_pending_index.json", dict(images=[]))
            position = candidates.index(item)
            progress("stopped_at_image_boundary", status="stopped", stop_marker=str(STOP_MARKER),
                     remaining_image_ids=[x["image_id"] for x in candidates[position + 1:]])
            raise SystemExit(0)
    save_tables()
    write_json(args.job / "closed_pending_index.json", dict(images=[]))
    progress("prepared", status="prepared", remaining_image_ids=[], independent_review_pending=True)
except (TimeoutError, SpaceUnavailable) as error:
    progress("waiting_io", status="waiting_io", error=str(error), open_source_files=0,
             resume_operation=state["stage"], resume_only_unfinished=True,
             error_type=type(error).__name__, publication_checkpoint_notes=getattr(error, "__notes__", []))
    raise SystemExit(75)
