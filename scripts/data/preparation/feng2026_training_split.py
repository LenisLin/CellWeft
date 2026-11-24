"""Retire patient-unresolved Feng ROIs and publish a patient-isolated train/val split.

Inputs: accepted Feng inputs, source identity crosswalk, existing experiment catalog.
Outputs: 515 retained ROI directories, training_split.json and catalog membership.
Run --retire, renew independent D01 with feng2026_review.py, then run --publish.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

import pandas as pd

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.inputs import (
    assert_reviewed_acceptance, input_binding, same_processing_binding,
)


STUDY = "Feng2026_IMC_BladderCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
ROOT = NAS / "processed" / STUDY
CACHE = NAS / "cache" / STUDY
SEED = 20261124


def retained_binding(binding: dict, image_ids: set[str]) -> dict:
    """Project the historical approval onto retained, unchanged input assets."""
    result = deepcopy(binding)
    result["image_ids"] = sorted(image_ids)
    result["images"] = [row for row in result["images"] if row["image_id"] in image_ids]
    result["assets"] = [row for row in result["assets"] if Path(row["path"]).parent.name in image_ids]
    return result


parser = argparse.ArgumentParser(description=__doc__)
action = parser.add_mutually_exclusive_group(required=True)
action.add_argument("--retire", action="store_true")
action.add_argument("--publish", action="store_true")
args = parser.parse_args()

# Derive exclusions from the source crosswalk, without renumbering retained ROIs.
identity = pd.read_csv(CACHE / "image_identity.csv", keep_default_na=False)
excluded = sorted(identity.loc[identity.patient_linkage_status.eq("unresolved"), "image_id"])
if len(excluded) != 5 or len(identity) != 520:
    raise ValueError("Expected the user-approved five unresolved ROIs from the original 520")
with resource_lock(ROOT):
    images = pd.read_parquet(ROOT / "images.parquet")
    retained = images.loc[~images.image_id.isin(excluded)].copy()
    if len(retained) != 515 or retained.source_patient_key.isna().any():
        raise ValueError("Retained scope must contain 515 patient-linked ROIs")
    confirmed = identity.set_index("image_id").loc[retained.image_id]
    if not confirmed.patient_linkage_status.eq("confirmed").all() or not (
        confirmed.patient_key.to_numpy() == retained.source_patient_key.to_numpy()
    ).all():
        raise ValueError("Patient keys must agree with the confirmed source crosswalk")

    # Randomize patient keys once; the nearest integer to 20% gives 12/62 patients.
    patients = sorted(retained.source_patient_key.unique())
    if len(patients) != 62:
        raise ValueError("Expected 62 confirmed patients")
    random.Random(SEED).shuffle(patients)
    validation = set(patients[:round(len(patients) * 0.2)])
    groups = {}
    for patient, rows in retained.groupby("source_patient_key", sort=True):
        split = "development_validation" if patient in validation else "development_train"
        groups[f"{STUDY}::patient::{patient}"] = {
            "level": "patient", "members": {split: sorted(rows.image_id)},
        }
    counts = {
        split: {"patients": sum(split in group["members"] for group in groups.values()),
                "images": sum(len(group["members"].get(split, [])) for group in groups.values())}
        for split in ("development_train", "development_validation")
    }
    plan = dict(
        schema="cellweft.training_split.v1", study=STUDY, owner="Feng", year=2026,
        disease="Bladder cancer", private_data=True,
        authority="user_instruction_2026-11-24", grouping="source_patient_key",
        method="Python random.Random(seed).shuffle(sorted(unique patient keys)); first round(0.2*N) validate",
        seed=SEED, validation_fraction_patients=0.2, counts=counts,
        excluded_image_ids=excluded, exclusion_reason="patient_linkage_unresolved",
        retained_image_ids=sorted(retained.image_id), groups=groups,
        training_policy="Fit only development_train; use development_validation for internal validation and model selection.",
        sampling_status="pending", effective_markers="mapped, vector-bound downstream markers only",
        remaining_restrictions=["clinical_analysis", "paired_analysis",
                                "cross_image_biological_sample_merge", "biological_region_analysis"],
    )
    if args.retire:
        # Exact user-authorized deletion affects processed data; raw sources retain provenance.
        retained.to_parquet(ROOT / "images.parquet", index=False)
        write_json(ROOT / "training_split.json", plan)
        for image_id in excluded:
            directory = ROOT / "images" / image_id
            if directory.is_dir():
                shutil.rmtree(directory)
        if {path.name for path in (ROOT / "images").iterdir() if path.is_dir()} != set(retained.image_id):
            raise ValueError("Processed directories differ from retained image membership")
    else:
        if json.loads((ROOT / "training_split.json").read_bytes()) != plan:
            raise ValueError("Published split plan differs from the source-derived patient partition")
        binding = input_binding(ROOT, NAS, retained.image_id.tolist())
        reviews = pd.read_csv(ROOT / "review_results.csv", keep_default_na=False)
        assert_reviewed_acceptance(reviews, binding)
        latest = reviews[reviews.check_id.eq("D01")].iloc[-1]
        approval = json.loads(latest.details)
        if approval.get("training_authorization", {}).get("authority") != plan["authority"]:
            raise ValueError("Independent review has not reconciled the training authorization")
        # Scope-only changes preserve existing image, mask and quantitative products.
        for image_id in retained.image_id:
            path = ROOT / "images" / image_id / "metadata.json"
            metadata = json.loads(path.read_bytes())
            if not same_processing_binding(retained_binding(metadata["input_binding"], set(retained.image_id)), binding):
                raise ValueError("Retained input pixels or processing dependencies changed")
            metadata["input_binding"] = binding
            write_json(path, metadata)

        panel = pd.read_csv(ROOT / "panel_mapping.csv")
        markers = panel[panel.mapping_status.eq("mapped") & panel.role.eq("downstream") & panel.embedding_row.notna()]
        dataset = dict(role="source", private_data=True, owner="Feng", year=2026,
                       disease="Bladder cancer", groups=groups,
                       panels={key: sorted(rows.marker_id.unique()) for key, rows in markers.groupby("panel_id")},
                       split_source=f"{STUDY}/training_split.json",
                       training_authorization=approval["training_authorization"],
                       sampling_status="pending")

# Merge against the current catalog under its own lock, preserving concurrent changes.
if args.publish:
    catalog_root = NAS / "processed" / "_catalog" / "public"
    with resource_lock(catalog_root):
        path = catalog_root / "experiments.json"
        catalog = json.loads(path.read_bytes())
        catalog["datasets"][STUDY] = dataset
        pretraining = catalog["tasks"]["foundation_pretraining"]
        pretraining["description"] = "公共来源及用户批准的私有来源无标签预训练"
        pretraining["datasets"][STUDY] = {
            "members": [{"use": "development_images_without_task_labels", "splits": ["development_train"]}],
            "validation_members": [{"use": "internal_validation_without_task_labels", "splits": ["development_validation"]}],
            "fit_splits": ["development_train"], "validation_splits": ["development_validation"],
            "sampling_status": "pending",
        }
        write_json(path, catalog)
        reread = json.loads(path.read_bytes())
        if reread["datasets"][STUDY] != dataset:
            raise ValueError("Training catalog readback differs")

print(json.dumps(dict(stage="retired" if args.retire else "published", excluded=len(excluded),
                      retained=len(retained), counts=counts), ensure_ascii=False))
