"""Resolve the excluded HuBMAP aggregate before any input publication.

Inputs: current catalogs, retained SPoRA metadata and local Zarr headers.
Outputs: job-local scope inventory and resumable progress; no aggregate input.
Prerequisite: docs/data/collection.md requires original-study attribution.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from cellweft.data.preparation.io_coordination import formal_read_lease


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path,
                    default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--job", type=Path, required=True)
args = parser.parse_args()
pa.set_cpu_count(4)
pa.set_io_thread_count(4)
root = args.data_root.resolve()
job = args.job.resolve()
study = "hubmap2019human"
source = root / "raw/public" / study / "spora"
metadata = metadata_source_directory(root, study)

# Confirm the current exclusion; a changed scope requires controller attribution.
with formal_read_lease(source, data_root=root):
    with (root / "processed/_catalog/public/studies.csv").open(encoding="utf-8-sig") as handle:
        registered = [row for row in csv.DictReader(handle)
                      if row["dataset_name"] == study]
    with (root / "processed/_catalog/public/candidates.csv").open(encoding="utf-8-sig") as handle:
        candidates = [row for row in csv.DictReader(handle)
                      if study in row.values()]
    if registered or len(candidates) != 1 or "NO (aggregate source" not in str(candidates[0]):
        raise RuntimeError("Scope evidence changed; controller must resolve original-study ownership")
    tissues = pq.read_table(source / "metadata/tissues.parquet").to_pandas()
    channels = pq.read_table(source / "codex/channels.parquet")
    availability = pq.read_table(metadata / "spora_inventory/codex/channels_per_tissue.parquet")
    prior_path = root / "processed/_catalog/public/evidence/hubmap2019human_followup_review.json"
    prior = json.loads(prior_path.read_text())

# Inspect object presence only: absent shards must never become fill-value images.
    inventory = []
    for image in sorted((source / "codex/0_3774mpp/images").glob("*.ome.zarr")):
        header_path = image / "0/zarr.json"
        if not header_path.is_file():
            inventory.append({
                "source_image_id": image.name.removesuffix(".ome.zarr"),
                "missing_array_header": True, "all_expected_shards_present": False,
                "pixel_decode_verified": False,
            })
            continue
        header = json.loads(header_path.read_text())
        shape = header["shape"]
        chunk = header["chunk_grid"]["configuration"]["chunk_shape"]
        grid = [range(math.ceil(size / step)) for size, step in zip(shape, chunk)]
        expected = list(itertools.product(*grid))
        present = sum((image / "0/c" / Path(*map(str, index))).is_file()
                      for index in expected)
        inventory.append({
            "source_image_id": image.name.removesuffix(".ome.zarr"),
            "shape": shape, "dtype": header["data_type"],
            "expected_shards": len(expected), "present_shards": present,
            "partial_files": len(list(image.rglob("*.partial"))),
            "all_expected_shards_present": present == len(expected),
            "pixel_decode_verified": False,
        })

# Source identifiers are evidence leads; original cohort and biological sample
# semantics remain unresolved and are not promoted to formal sample records.
sample_keys = tissues[["patient_id", "specimen_id"]].dropna().drop_duplicates()
identity_summary = {
    "source_metadata_rows": len(tissues),
    "rows_with_patient_id": int(tissues.patient_id.notna().sum()),
    "source_patient_id_count": int(tissues.patient_id.nunique()),
    "source_patient_specimen_pairs": len(sample_keys),
    "rows_with_original_patient_identifier": int(tissues.original_patient_identifier.notna().sum()),
    "distinct_original_filenames": int(tissues.original_file_name.nunique()),
    "clinical_source": "retained tissues.parquet identifiers only; no clinical table established",
    "limitation": "Source identifiers do not resolve original-study ownership or independent samples",
}
nas_free = shutil.disk_usage(root).free
local_free = shutil.disk_usage("/tmp").free
if nas_free < 100 * 2**30 or local_free < 64 * 2**30:
    raise RuntimeError("Insufficient reserve for the small scope checkpoint")
evidence = {
    "study": study, "checked_at": datetime.now(timezone.utc).isoformat(),
    "scope": "excluded_aggregate", "catalog_registered_rows": len(registered),
    "candidate_record": candidates[0], "prior_scope_review": prior,
    "source_identity_summary": identity_summary,
    "channel_metadata_rows": channels.num_rows,
    "panel_availability_rows": availability.num_rows,
    "local_image_groups": inventory,
    "space": {"nas_free_bytes": nas_free, "local_free_bytes": local_free,
              "estimated_checkpoint_bytes_upper_bound": 1048576,
              "planned_pixel_export_bytes": 0},
    "source_files_modified": False,
    "formal_output_directory_exists": (root / "processed" / study).exists(),
    "preparer": str(job), "independent_review_performed": False,
}
progress = {
    "study": study, "phase": "prepare", "scientific_repair_round": 0,
    "status": "blocked", "expected_images": 0, "exported_images": 0,
    "sample_rows": 0, "sample_linked_images": 0, "patient_linked_samples": 0,
    "completed_scope": "Current exclusion, local object presence and source identity leads checked",
    "resume_from": "Controller resolves original-study attribution and eligible nonduplicate scope",
    "issues": ["HuBMAP aggregate is excluded as a study; original-study attribution remains unresolved",
               "Retained Zarr groups are incomplete; no native pixel export is authorized"],
    "evidence_paths": [str(job / "scope_inventory.json"), str(prior_path)],
    "controller_requests": [
        "按现行聚合排除裁定协调本任务归属；不得恢复hubmap2019human资源行。",
        "若继续来源追查，请由主控确认原始study归属、重复观测及完整合格来源后派发；全局catalog和其他study保持未改。",
    ],
}
job.mkdir(parents=True, exist_ok=True)
for name, payload in [("scope_inventory.json", evidence), ("progress.json", progress)]:
    temporary = job / (name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(job / name)
print(json.dumps({**progress, "local_groups": len(inventory),
                  "groups_with_all_shards": sum(item["all_expected_shards_present"] for item in inventory)},
                 ensure_ascii=False))
