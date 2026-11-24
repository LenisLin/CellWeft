"""Prepare Feng2026 technical image products after independent D01 approval.

Inputs: processed study tables and input TIFFs, and the private study cache's
mask_review.csv and mask_bindings.json from feng2026_masks.py.
Outputs: standard per-image products and metadata. A private SSD work root
holds interrupted work, and --image-ids-json selects private representative IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from threading import Lock
from uuid import uuid4

STUDY = "Feng2026_IMC_BladderCancer"
SCOPE_SIZE = 520
PROJECT = Path(__file__).resolve().parents[3]
TABLES = ("images.parquet", "panel_mapping.csv", "sample_metadata.csv", "field_mapping.csv", "review_results.csv")
PRODUCTS = {"image": "image.tiff", "channels": "channels.parquet", "mask": "mask.tiff",
            "cells": "cells.parquet", "expression": "expression.parquet", "neighbors": "neighbors.parquet"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--work-root", type=Path, default=PROJECT / ".cache/cellweft/feng2026_downstream")
parser.add_argument("--mask-review", type=Path, help="Override mask_review.csv and its sibling mask_bindings.json")
parser.add_argument("--image-ids-json", type=Path, help="Private list of representative image IDs")
parser.add_argument("--tissue-reviewed-panels", type=Path, help="Private JSON list following representative tissue review")
parser.add_argument("--weights", type=Path, help="Enable local CPSAM v2 only for reviewed absent/unusable masks")
parser.add_argument("--cpu-cores", type=int, default=8)
args = parser.parse_args()
affinity = sorted(os.sched_getaffinity(0))
if not 1 <= args.cpu_cores <= min(8, len(affinity)):
    parser.error("--cpu-cores must be within the available eight-core budget")
os.sched_setaffinity(0, affinity[:args.cpu_cores])
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
import pyarrow as pa

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.execution import publish_staged_image, stage_source
from cellweft.data.preparation.inputs import assert_reviewed_binding, load_reviewed_study_inputs
from cellweft.data.preparation.io_coordination import formal_read_lease, resolve_lock_root
from cellweft.data.preparation.pipeline import prepare_image
from cellweft.data.preparation.requests import file_identity, preparation_request
from cellweft.data.preparation.scheduling import _latest_d01_scopes
from cellweft.data.preparation.segmentation import CellposeExecutor
from cellweft.data.preparation.validation import validate_products

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
data_root = args.data_root.resolve()
study_root = data_root / "processed" / STUDY
mask_review = (args.mask_review or data_root / "cache" / STUDY / "mask_review.csv").resolve()
mask_bindings = mask_review.with_name("mask_bindings.json")
work_root = args.work_root.resolve() / STUDY
io_root = resolve_lock_root(study_root, data_root=data_root)
table_paths = tuple(study_root / name for name in TABLES)
gpu_lock = Lock()
segmenter = None


def private_list(path: Path) -> set[str]:
    """Read a private explicit selection without reporting its members."""
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("Expected a JSON list of nonempty IDs")
    if len(values) != len(set(values)):
        raise ValueError("Repeated selection ID")
    return set(values)


def study_revision() -> tuple[tuple[int, int], ...]:
    """Use the study lease to detect changed tables without rescanning a D01 scope per image."""
    return tuple((stat.st_size, stat.st_mtime_ns) for stat in (path.stat() for path in table_paths))


def is_complete(metadata: dict, expected: dict, directory: Path, shape: tuple[int, int]) -> bool:
    """Trust a resumable result only after reopening products under the appropriate lease."""
    if (metadata.get("request") != expected or metadata.get("publication_incomplete")
            or not PRODUCTS.keys() <= set(metadata.get("completed", []))
            or not metadata.get("mask_identity") or metadata.get("mask_origin") not in {"source", "cpsam_v2"}
            or metadata.get("shape_yx") != list(shape)
            or (expected["sampling"]["reviewed"] and metadata.get("sampling", {}).get("status") != "complete")):
        return False
    products = metadata["products"]
    if any(products.get(key) != str(directory / name) or not (directory / name).is_file()
           for key, name in PRODUCTS.items()):
        return False
    validate_products(products, shape)
    return True


def compute(job: tuple) -> str:
    """Stage one approved source once, compute on SSD, and publish under the formal lease."""
    row, inputs, scope_state = job
    image_id = row.image_id
    mask_row = masks.get(image_id)
    binding = bindings.get(image_id)
    if mask_row is None or binding is None:
        return "mask_pending"
    phase = "mask_review"
    try:
        if not isinstance(binding, dict):
            raise ValueError("Mask binding must be an object")
        status = binding["status"]
        if (status != mask_row["status"] or str(row.source_format) != mask_row["format"]
                or int(mask_row["width"]) != row.source_x1_px - row.source_x0_px
                or int(mask_row["height"]) != row.source_y1_px - row.source_y0_px):
            return "mask_review_stale"
        if status == "unknown":
            return "mask_pending"
        record, source, panel = inputs.source(image_id, binding)
        mask_asset = None
        if status == "bound":
            overview = {"True": True, "False": False, "": None}[mask_row["overview_judgment"]]
            if (binding["path"] != mask_row["binding_path"] or binding["frame"] != "source"
                    or source.mask.mask_type != mask_row["mask_type"] or source.mask.overview_judgment is not overview):
                return "mask_review_stale"
            if source.mask.overview_required and source.mask.overview_judgment is None:
                return "mask_pending"
            if source.mask.overview_judgment is False or source.mask.mask_type != "whole_cell" or source.mask.alignment_issue:
                return "mask_needs_disposition"
            representatives = [masks.get(key) for key in mask_row["sample_evidence_ids"].split(";")]
            if any(rep is None or rep["stratum"] != mask_row["stratum"] or rep["sampled"] != "True"
                   or rep["overview_judgment"] != "True" for rep in representatives):
                return "mask_pending"
            if not source.mask.path.is_file():
                return "mask_asset_missing"
            mask_asset = file_identity(source.mask.path)
        elif status == "unusable" and mask_row["reason"] != binding["reason"]:
            return "mask_review_stale"
        elif args.weights is None:
            return "segmentation_required"

        phase = "stage"
        shape = (int(row.height_px), int(row.width_px))
        formal = study_root / "images" / image_id
        if source.path != formal / "input.tiff":
            raise ValueError("Unexpected formal input path")
        work = work_root / "work" / image_id
        gpu_required = status in {"absent", "unusable"}
        settings = load_settings(PROJECT / "configs/data/preparation/defaults.yaml", work / "cache",
            cpu_workers=1, sampling_reviewed=row.panel_id in tissue_panels,
            weights=args.weights if gpu_required else None, precision="bf16", batch_size=8)
        reviewed_assets = scope_state["assets"]
        receipt = {"source_revision": str(row.source_revision), "mask_file_identity": mask_asset, "review": mask_row}

        def check_binding() -> None:
            """Check current study acceptance and this image's physical assets under the caller's leases."""
            revision = study_revision()
            with scope_state["lock"]:
                if revision != scope_state["revision"]:
                    assert_reviewed_binding(study_root, data_root=data_root, binding=inputs.binding,
                                            asset_workers=args.cpu_cores)
                    scope_state["revision"] = revision
            if tuple(file_identity(path) for path in (mask_review, mask_bindings)) != mask_revision:
                raise ValueError("Mask reviews changed during this run")
            if tissue_revision is not None and file_identity(args.tissue_reviewed_panels) != tissue_revision:
                raise ValueError("Tissue reviews changed during this run")
            for path in (source.path, source.valid_mask, source.sample_region):
                if path is not None and file_identity(path) != reviewed_assets[str(path)]:
                    raise ValueError("D01 image asset changed")
            if mask_asset is not None and file_identity(source.mask.path) != mask_asset:
                raise ValueError("Reviewed mask asset changed")

        with formal_read_lease(formal, io_lock_root=io_root), resource_lock(study_root, shared=True), \
                resource_lock(formal, shared=True, name=".input.lock"):
            check_binding()
            desired = preparation_request(source, panel, settings, shape)
            old_path = formal / "metadata.json"
            with resource_lock(formal, shared=True):
                old_exists = old_path.is_file()
                old = json.loads(old_path.read_text()) if old_exists else {}
                try:
                    existing = is_complete(old, desired, formal, shape)
                except (ValueError, OSError, KeyError):
                    existing = False
            if (existing and old.get("input_binding") == inputs.binding and old.get("mask_binding") == binding
                    and old.get("mask_review") == receipt):
                if work.exists():
                    shutil.rmtree(work)
                return "reused" if old["sampling"]["status"] == "complete" else "reused_sampling_pending"
            local_source = stage_source(source, work / "input") if not existing else None

        phase = "compute"
        local_record = replace(record, output_path=work / "processed")
        local_dir = local_record.output_path / "images" / image_id
        local_metadata = local_dir / "metadata.json"
        if existing:
            metadata = old
        else:
            if local_metadata.is_file():
                cached = json.loads(local_metadata.read_text())
                try:
                    is_complete(cached, desired, local_dir, shape)
                except (ValueError, OSError, KeyError):
                    local_metadata.unlink()
            with open_source(local_source) as reader:
                if gpu_required:
                    with gpu_lock:
                        result = prepare_image(local_record, local_source, reader, panel, settings,
                            segmentation=segmenter, identity_source=source, canonical_request=desired)
                else:
                    result = prepare_image(local_record, local_source, reader, panel, settings,
                        identity_source=source, canonical_request=desired)
            metadata = json.loads(result.metadata_path.read_text())
            if not is_complete(metadata, desired, local_dir, shape):
                if status == "bound" and not metadata.get("mask_identity"):
                    return "mask_needs_disposition"
                return "products_incomplete"

        phase = "publication"
        metadata.update(mask_review=receipt)
        write_json(local_metadata, metadata)
        published = publish_staged_image(local_metadata, formal, uuid4().hex, io_root,
            check_binding, inputs.binding, binding, expected_request=desired, allow_new=not old_exists,
            previous_attempt=old.get("controller_attempt"), replace_previous=True)
        shutil.rmtree(work)
        return "published" if published["sampling"]["status"] == "complete" else "published_sampling_pending"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return f"{phase}_error_{type(error).__name__}"


if __name__ == "__main__":
    try:
        selected = private_list(args.image_ids_json) if args.image_ids_json else None
        target = SCOPE_SIZE if selected is None else len(selected)
        if not 1 <= target <= SCOPE_SIZE:
            raise ValueError("Invalid selection count")
        missing_tables = sum(not path.is_file() for path in table_paths)
        missing_review = sum(not path.is_file() for path in (mask_review, mask_bindings))
        if missing_tables or missing_review or not os.access(study_root, os.W_OK):
            print(json.dumps({"target": target, "ready": 0, "missing_study_tables": missing_tables,
                "missing_mask_review_files": missing_review, "formal_writable": os.access(study_root, os.W_OK)}), flush=True)
            raise SystemExit(2)

        work_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with resource_lock(work_root, name=".controller.lock"):
            with resource_lock(study_root, shared=True):
                design = pd.read_parquet(table_paths[0], columns=["image_id"])
                if len(design) > SCOPE_SIZE or design.image_id.duplicated().any():
                    raise ValueError("Study image index exceeds or duplicates the formal scope")
                table_revision = study_revision()
                mask_revision = tuple(file_identity(path) for path in (mask_review, mask_bindings))
                review = pd.read_csv(mask_review, dtype=str, keep_default_na=False)
                masks = review.set_index("image_id", verify_integrity=True).to_dict("index")
                bindings = json.loads(mask_bindings.read_text())
                if not isinstance(bindings, dict) or masks.keys() != bindings.keys():
                    raise ValueError("Mask review and binding scopes differ")
                tissue_revision = file_identity(args.tissue_reviewed_panels) if args.tissue_reviewed_panels else None
                tissue_panels = private_list(args.tissue_reviewed_panels) if args.tissue_reviewed_panels else set()

            scopes, review_gaps = _latest_d01_scopes(study_root)
            rejected_scopes = len(review_gaps)
            pending = set(design.image_id) if selected is None else selected.copy()
            jobs = []
            headers = {}
            for scope in sorted(scopes, key=lambda value: -len(value["binding"]["image_ids"])):
                scope_ids = scope["binding"]["image_ids"]
                candidates = pending.intersection(scope_ids)
                if not candidates:
                    continue
                try:
                    inputs = load_reviewed_study_inputs(study_root, data_root=data_root,
                        image_ids=sorted(candidates), accepted_scope_image_ids=scope_ids,
                        header_cache=headers, read_gate=lambda path: formal_read_lease(path, io_lock_root=io_root))
                except (OSError, ValueError, KeyError):
                    rejected_scopes += 1
                    continue
                scope_state = {"lock": Lock(), "revision": table_revision,
                    "assets": {asset["path"]: asset for asset in inputs.binding["assets"]}}
                jobs.extend((row, inputs, scope_state) for row in inputs.images.itertuples())
                pending.difference_update(candidates)

            if args.weights is not None:
                segmenter = CellposeExecutor("cuda:0", precision="bf16", memory_budget_bytes=20 * 1024**3)
            counts = Counter()
            with ThreadPoolExecutor(max_workers=args.cpu_cores) as pool:
                for result in pool.map(compute, jobs):
                    counts[result] += 1
            complete = sum(count for result, count in counts.items() if result.startswith(("published", "reused")))
            print(json.dumps({"target": target, "d01_ready": len(jobs), "d01_scope_gaps": rejected_scopes,
                "technical_complete": complete, "outcomes": dict(counts)}), flush=True)
            raise SystemExit(0 if complete == target else 2)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"startup_error": type(error).__name__}), flush=True)
        raise SystemExit(2) from None
