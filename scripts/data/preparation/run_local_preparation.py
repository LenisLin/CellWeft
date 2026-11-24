"""Prepare D01-approved TIFFs on SSD, sharing one local GPU and bounded NAS publication.

Inputs: reviewed study tables and immutable input.tiff; outputs: standard image products.
Use --image-ids-json for the representative batch before admitting the remaining scope.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback
import uuid

# Bound the complete process before importing numerical libraries or spawning workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--study", required=True)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--image-ids-json", type=Path)
parser.add_argument("--gpu-gib", type=int, default=20)
parser.add_argument("--cpu-cores", type=int, default=8)
parser.add_argument("--allow-unknown-mask-segmentation", action="store_true")
parser.add_argument("--sampling-reviewed", action="store_true",
                    help="Use only after representative tissue-estimation review passes")
args = parser.parse_args()
if not 1 <= args.gpu_gib <= 20:
    raise ValueError("GPU allocator budget must be within the approved 20 GiB")
allowed = sorted(os.sched_getaffinity(0))
if not 1 <= args.cpu_cores <= len(allowed):
    raise ValueError("CPU budget exceeds available affinity")
os.sched_setaffinity(0, allowed[:args.cpu_cores])
args.run_root.mkdir(parents=True, exist_ok=True)
project = Path(__file__).resolve().parents[3]
owner_path = project / ".cache/cellweft/runtime/local_preparation" / args.study / "controller.lock"
owner_path.parent.mkdir(parents=True, exist_ok=True)
owner = owner_path.open("a+")
fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

import torch
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.execution import CPUSlots, LocalSegmentation, stage_source, publish_staged_image
from cellweft.data.preparation.inputs import load_reviewed_study_inputs, assert_reviewed_binding
from cellweft.data.preparation.io_coordination import default_io_root
from cellweft.data.preparation.pipeline import prepare_image, preparation_request
from cellweft.data.preparation.remote_worker import limited_slots
from cellweft.data.preparation.validation import validate_products

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError("Local preparation requires GPU device access in this execution environment")
study_root = args.data_root / "processed" / args.study
locks = default_io_root(args.data_root)
required = {"image", "channels", "mask", "cells", "expression", "neighbors"}
selected = json.loads(args.image_ids_json.read_text()) if args.image_ids_json else None
import pandas as pd
scope = pd.read_parquet(study_root / "images.parquet").image_id.tolist()
inputs = load_reviewed_study_inputs(study_root, data_root=args.data_root,
    image_ids=selected, accepted_scope_image_ids=scope,
    read_gate=lambda _: limited_slots(locks, "nas_read", 2))
cpu_slots = CPUSlots(min(2, args.cpu_cores))
weights = args.data_root / "raw/shared/pretrained/cellpose/cpsam_v2/checkpoint"
segmenter = LocalSegmentation({"device": "cuda:0", "precision": "bf16", "batch_size": 8,
    "weights": str(weights), "memory_budget_bytes": args.gpu_gib * 1024**3}, cpu_slots, args.run_root)
attempt = uuid.uuid4().hex
state_path = args.run_root / "status.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"images": {}}
state.update(pid=os.getpid(), attempt=attempt, study=args.study, phase="running",
    cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_budget_gib=args.gpu_gib,
    selected_image_count=len(inputs.images), sampling_reviewed=args.sampling_reviewed)
state_lock = threading.Lock()
deliveries = queue.Queue(maxsize=2)
# Reserve before compute: completed producer-local packages also count toward backpressure.
package_slots = threading.Semaphore(2)
stop_admission = threading.Event()


def update(image_id: str, phase: str, **values) -> None:
    """Keep one queryable, atomically replaced controller receipt on SSD."""
    with state_lock:
        state["images"][image_id] = {"status": phase, "updated": time.time(), **values}
        state["updated"] = time.time()
        write_json(state_path, state)
    print(json.dumps({"image_id": image_id, "status": phase, **values}), flush=True)


def check_binding() -> None:
    assert_reviewed_binding(study_root, data_root=args.data_root, binding=inputs.binding)


def compute(image_id: str) -> None:
    """Read under input leases, compute on SSD, and release CPU before publication."""
    work = args.run_root / "work" / image_id
    mask_binding = {"status": "unknown"}
    package_reserved = False
    try:
        if stop_admission.is_set():
            update(image_id, "not_admitted", reason="publication failure; SSD packages retained")
            return
        record, canonical, panel = inputs.source(image_id, mask_binding=mask_binding)
        settings = load_settings(project / "configs/data/preparation/defaults.yaml", work / "cache",
            weights=weights, batch_size=8, cpu_workers=max(1, args.cpu_cores // 2),
            precision="bf16", sampling_reviewed=args.sampling_reviewed,
            allow_unknown_mask_segmentation=args.allow_unknown_mask_segmentation)
        formal = study_root / "images" / image_id
        update(image_id, "reading")
        with limited_slots(locks, "nas_read", 2), resource_lock(study_root, shared=True), resource_lock(formal, shared=True, name=".input.lock"):
            check_binding()
            with open_source(canonical) as reader:
                desired = preparation_request(canonical, panel, settings, reader.shape[-2:], segmenter)
            old_path = formal / "metadata.json"
            old = json.loads(old_path.read_text()) if old_path.exists() else {}
            if old.get("request") == desired and required.issubset(old.get("completed", [])) and not old.get("publication_incomplete"):
                validate_products(old["products"], tuple(old["shape_yx"]))
                update(image_id, "complete", reused=True, metadata=str(old_path))
                return
            local = stage_source(canonical, work / "input")
            # Resume changed sampling or partial delivery from existing scientific products.
            local_metadata = work / "processed" / record.study / "images" / image_id / "metadata.json"
            if old.get("products") and not local_metadata.exists():
                with resource_lock(formal, shared=True):
                    seed = {**old, "products": {}}
                    local_metadata.parent.mkdir(parents=True, exist_ok=True)
                    for name, path in old["products"].items():
                        target = local_metadata.parent / Path(path).name
                        if Path(path).is_file():
                            shutil.copy2(path, target)
                        seed["products"][name] = str(target)
                    write_json(local_metadata, seed)
            check_binding()
        update(image_id, "computing")
        package_slots.acquire()
        package_reserved = True
        if stop_admission.is_set():
            update(image_id, "not_admitted", reason="publication failure; SSD packages retained")
            return
        with cpu_slots, open_source(local) as reader:
            result = prepare_image(replace(record, output_path=work / "processed" / record.study),
                local, reader, panel, settings, segmenter,
                identity_source=canonical, canonical_request=desired)
        metadata = json.loads(result.metadata_path.read_text())
        if not required.issubset(metadata["completed"]):
            update(image_id, "incomplete", gaps=metadata["gaps"], metadata=str(result.metadata_path))
            return
        validate_products(metadata["products"], tuple(metadata["shape_yx"]))
        update(image_id, "waiting_publication", metadata=str(result.metadata_path))
        deliveries.put((image_id, work, result.metadata_path, formal, desired,
                        old.get("controller_attempt"), not old_path.exists(), mask_binding))
        package_reserved = False  # The publisher now owns this package reservation.
    except Exception as error:
        update(image_id, "failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if package_reserved:
            package_slots.release()


def publish() -> None:
    """Publish one complete package at a time; waiting uses no compute slot."""
    while (delivery := deliveries.get()) is not None:
        package_slots.release()  # This package is actively publishing, no longer waiting.
        image_id, work, metadata_path, formal, desired, previous, is_new, mask_binding = delivery
        try:
            update(image_id, "publishing")
            metadata = publish_staged_image(metadata_path, formal, attempt, locks, check_binding,
                inputs.binding, mask_binding, source_directory=formal, expected_request=desired,
                allow_new=is_new, previous_attempt=previous, replace_previous=True)
            update(image_id, "complete", metadata=str(formal / "metadata.json"),
                sampling=metadata["sampling"]["status"], gaps=metadata["gaps"])
            shutil.rmtree(work)
        except Exception as error:
            stop_admission.set()
            update(image_id, "publication_failed", error=f"{type(error).__name__}: {error}", metadata=str(metadata_path))
            traceback.print_exc()
        finally:
            deliveries.task_done()


# Two compute jobs overlap SSD/CPU work with the serialized persistent GPU.
publisher = threading.Thread(target=publish, name="nas-publisher")
publisher.start()
try:
    with ThreadPoolExecutor(max_workers=min(2, args.cpu_cores)) as pool:
        for _ in pool.map(compute, inputs.images.image_id.tolist()):
            pass
finally:
    deliveries.put(None)
    publisher.join()
    state["phase"] = "finished"
    state["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    state["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(state_path, state)
    owner.close()
failures = [key for key in inputs.images.image_id if state["images"].get(key, {}).get("status") != "complete"]
print(json.dumps({"complete": len(inputs.images) - len(failures), "not_complete": failures}), flush=True)
raise SystemExit(bool(failures))
