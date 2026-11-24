"""Prepare native IDR0076 METABRIC IMC inputs and source-linked sample metadata.

Inputs: complete full-stack TIFFs, accepted native proofs, IDR annotation, S1,
and cached Ali 2020 / Rueda 2019 publication metadata. Outputs: per-image
input.tiff, study tables, resumable progress and preparer verification evidence.
Run --export with the shared formal write lease; --verify/--overview with the shared formal read lease.
Independent acceptance is performed by a separate reviewer after this script.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--export", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--overview", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
STUDY = "Ali2020_IMC_BreastCancer"
PANEL = "idr0076_fullstack_50"
root = args.data_root
raw = root / f"raw/public/{STUDY}/idr0076_20191219-ftp"
metadata = metadata_source_directory(root, STUDY)
intake = metadata / "idr0076_intake"
sources = metadata / "preparation_sources"
cache = Path(__file__).resolve().parents[3] / "cache/study_preparation" / STUDY
output = root / f"processed/{STUDY}"
progress_path = cache / "progress.json"
cache.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return path.as_posix()
    project_root = Path(__file__).resolve().parents[3]
    if path.is_relative_to(project_root):
        return path.relative_to(project_root).as_posix()
    return metadata_source_relative(path, root).as_posix()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, images: pd.DataFrame, gaps: list[str]) -> None:
    """Publish actual persisted image coverage after each independent image."""
    save_json(progress_path, dict(study=STUDY, stage=stage,
        updated_at=datetime.now(timezone.utc).isoformat(), expected_images=len(annotation),
        completed_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.ne("").sum()), gaps=gaps))


def save_tables(images: pd.DataFrame) -> None:
    save_input_tables(output, images, samples, panel, fields)


def check_sample_vocabulary(frame: pd.DataFrame) -> None:
    """Study-local contract check while shared publication validates only image fields.

    Native FFPE/breast-tumour descriptions need ffpe/tumor categorical values;
    an independent sample check prevents source text from bypassing that contract.
    """
    dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
    vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
    for field in dictionary[dictionary.table_name.eq("sample_metadata")].itertuples(index=False):
        if field.value_set:
            allowed = vocabulary.loc[vocabulary.value_set.eq(field.value_set), "allowed_value"]
            assert frame.loc[frame[field.field_name].ne(""), field.field_name].isin(allowed).all(), field.field_name


# Reuse accepted native evidence; partial downloads and range journals are untouched.
annotation_path = raw / "idr0076-experimentA-annotation.csv"
annotation = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
manifest = json.loads((intake / "source_manifest.json").read_text())
accepted = json.loads((intake / "accepted_index.json").read_text())
legacy = json.loads((intake / "legacy27_native_proof.json").read_text())
proofs = {row["file"]: row for row in legacy["images"]}
for name in accepted["new_proof_files"]:
    proof = json.loads((intake / name).read_text())
    proofs[proof["file_key"]] = proof
assert set(proofs) == set(annotation["Image File"]) == {p.name for p in (raw / "full_stacks").glob("*.tiff")}
assert len(proofs) == accepted["count"] == 548
ordered = json.loads((intake / "ordered_mapping_proof.json").read_text())
channel_map = pd.read_csv(metadata / "channel_map.csv", dtype=str, keep_default_na=False)
source_names = channel_map.source_channel_label.tolist()
assert annotation.Channels.map(lambda value: [x.strip() for x in value.split(",")]).map(lambda x: x == source_names).all()
assert source_names == ordered["ordered_labels"]
source_records = {row["file_key"]: row for row in manifest["sources"]}
inventory = []
for row in annotation.to_dict("records"):
    path = raw / "full_stacks" / row["Image File"]
    proof = proofs[path.name]
    stat = path.stat()
    identity = proof.get("file_identity")
    if identity:
        assert stat.st_size == identity["size"] and stat.st_mtime_ns == identity["mtime_ns"], path
    else:
        assert stat.st_size == proof["bytes"], path
    assert source_records[path.name]["specimen_join"]["metabric_id"] == row["Metabric ID"]
    shape = proof.get("native_shape", proof.get("shape"))
    assert shape[0] == 50
    inventory.append(dict(image_id=row["Image Name"], source_file=relative(path),
        shape=shape, source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
        native_proof=(relative(intake / "legacy27_native_proof.json") if not identity else
            relative(intake / f"proofs/{source_records[path.name]['source_order']:04d}.json"))))
output_estimate = sum(35 * max(x["shape"][1:]) ** 2 * 4 for x in inventory)
save_json(cache / "input_inventory.json", dict(study=STUDY, complete_images=548,
    ignored_source_suffixes=dict(Counter(p.suffix for p in (raw / "full_stacks").iterdir() if p.suffix != ".tiff")),
    uncompressed_output_estimate_bytes=output_estimate, images=inventory))

# Bind native isotope labels through the study-specific metal/marker map to S1.
# In particular Eu153 is published as p-Total HH3 but maps to p-HH3 in metal_to_marker.
metal_map = pd.read_csv(raw / "metal_to_marker.csv", dtype=str).set_index("metal").marker
assert channel_map.apply(lambda row: metal_map[row.metal] == row.source_marker, axis=1).all()
aliases = dict(zip(source_names, channel_map.source_marker))
panel = panel_from_s1(STUDY, PANEL, source_names,
    root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv", aliases=aliases)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 35 and panel.nuclear.sum() == 2

# METABRIC IDs identify original tumour specimens; independent cores and acquisitions
# share that biological sample. MBxxxx -> MB-xxxx is supported by all nonempty
# IDR cBioPortal URLs, and the Rueda tables use the same explicit METABRIC.ID key.
full_path = sources / "41586_2019_1007_MOESM7_ESM.txt"
molecular_path = sources / "41586_2019_1007_MOESM8_ESM.txt"
clinical = pd.read_csv(full_path, sep="\t", dtype=str, keep_default_na=False).replace("NA", "").set_index("METABRIC.ID", verify_integrity=True)
molecular = pd.read_csv(molecular_path, sep="\t", dtype=str, keep_default_na=False).replace("NA", "").set_index("METABRIC.ID", verify_integrity=True)
cbio_sample_path = sources / "cbioportal_data_clinical_sample.txt"
cbio_patient_path = sources / "cbioportal_data_clinical_patient.txt"
cbio_samples = pd.read_csv(cbio_sample_path, sep="\t", comment="#", dtype=str, keep_default_na=False).set_index("SAMPLE_ID", verify_integrity=True)
cbio_patients = pd.read_csv(cbio_patient_path, sep="\t", comment="#", dtype=str, keep_default_na=False).set_index("PATIENT_ID", verify_integrity=True)
full_source_fields = {
    "MATCHED.NORMAL.METABRIC.ID": "source_matched_normal_metabric_id",
    "Breast.Tumour.Laterality": "source_breast_tumour_laterality",
    "Date.Of.Diagnosis": "source_date_of_diagnosis",
    "Last.Followup.Status": "source_last_followup_status",
    "NPI": "source_npi",
    "ER.Status": "source_er_status",
    "Inferred.Menopausal.State": "source_inferred_menopausal_state",
    "Lymph.Nodes.Positive": "source_lymph_nodes_positive",
    "Breast.Surgery": "source_breast_surgery",
    "CT": "source_chemotherapy_regimen",
    "HT": "source_hormone_therapy_regimen",
    "RT": "source_radiotherapy_regimen",
    "Size": "source_tumor_size_mm",
    "DeathBreast": "source_death_breast",
    "Death": "source_death",
    "T": "source_time_to_last_followup_or_death_days",
    "TLR": "source_time_to_last_followup_or_locoregional_relapse_days",
    "LR": "source_locoregional_relapse",
    "TDR": "source_time_to_last_followup_or_distant_relapse_days",
    "DR": "source_distant_relapse",
    "Complete.Rec.History": "source_complete_recurrence_history",
}
molecular_source_fields = {
    "Cellularity": "source_cellularity",
    "HER2.SNP6": "source_her2_snp6",
    "iC10": "source_ic10",
    "ER.Expr": "source_er_expression",
    "Her2.Expr": "source_her2_expression",
    "PR.Expr": "source_pr_expression",
    "ClaudinSubtype": "source_claudin_subtype",
}
followup_status_to_vital_status = {
    "a": "alive",
    "d": "deceased",
    "d-d.s.": "deceased",
    "d-o.c.": "deceased",
}
assert set(clinical["Last.Followup.Status"].dropna().unique()) <= set(followup_status_to_vital_status)
def normalize_metabric_id(value: str) -> str:
    return "MB-" + value[2:] if value.startswith("MB") else value


annotation["patient_join"] = annotation["Metabric ID"].map(normalize_metabric_id)
assert annotation.groupby("Metabric ID")["patient_join"].nunique().max() == 1
assert annotation.groupby("Metabric ID")["patient_join"].first().is_unique
has_url = annotation["cBioPortal URL"].ne("")
assert all(url.endswith("caseId=" + key) for url, key in zip(annotation.loc[has_url, "cBioPortal URL"], annotation.loc[has_url, "patient_join"]))
assert annotation.groupby("Metabric ID")["Characteristics [Age at Diagnosis]"].nunique().max() == 1
samples = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str)
samples["source_tumor_stage"] = pd.Series(dtype=str)
for target_field in (*full_source_fields.values(), *molecular_source_fields.values()):
    samples[target_field] = pd.Series(dtype=str)
sample_rows = []
for native_id, group in annotation.groupby("Metabric ID", sort=False):
    native = group.iloc[0]
    key = native.patient_join
    sample = dict.fromkeys(samples.columns, "")
    # The explicit IDR identifier establishes the study-scoped patient key;
    # clinical tables are optional value sources and may lack a row.
    patient = key
    sample.update(study=STUDY, sample_id=native_id, source_sample_id=native_id,
        patient_key=patient, source_subject_id=patient, cohort_id="METABRIC",
        sample_type="tumour tissue", tissue_type="tumor", anatomical_site="breast",
        preservation_method="ffpe", primary_diagnosis=native["Characteristics [Diagnosis]"],
        primary_site=native["Characteristics [Organism Part]"],
        age_at_diagnosis_years=native["Characteristics [Age at Diagnosis]"],
        metadata_notes="IDR Metabric ID identifies one original tumour sample and supplies its study-scoped patient key; all its cores/acquisitions share sample_id. "
        "Age is age at diagnosis, not age at collection. Clinical variables absent in source are blank.")
    if key in clinical.index:
        c = clinical.loc[key]
        sample.update(histology=c["Histological.Type"], grade=c["Grade"],
                      source_tumor_stage=c["Stage"], center_id=c["Cohort"])
        if c["Age.At.Diagnosis"]:
            assert np.isclose(float(native["Characteristics [Age at Diagnosis]"]),
                              float(c["Age.At.Diagnosis"]), rtol=0, atol=1e-4)
            sample["age_at_diagnosis_years"] = c["Age.At.Diagnosis"]
        sample.update({target: c[source] for source, target in full_source_fields.items()})
        if c["Last.Followup.Status"]:
            sample["vital_status"] = followup_status_to_vital_status[c["Last.Followup.Status"]]
        sample["metadata_notes"] += (
            " Rueda Supplementary Table 5 matched by METABRIC.ID; stage source system/basis unspecified. "
            "Time-to-event fields remain in their published source form because their time origin is not defined.")
    else:
        sample["metadata_notes"] += " No row in Rueda Supplementary Table 5; fields sourced from that table remain blank."
    if key in molecular.index:
        m = molecular.loc[key]
        sample["molecular_subtype"] = m["Pam50Subtype"]
        sample.update({target: m[source] for source, target in molecular_source_fields.items()})
        sample["metadata_notes"] += " Rueda Supplementary Table 6 molecular fields matched by METABRIC.ID."
    if key in cbio_samples.index:
        cbio_patient = cbio_samples.loc[key, "PATIENT_ID"]
        sample["lesion_type"] = {"Primary": "primary"}[cbio_samples.loc[key, "SAMPLE_TYPE"]]
        sample["sex_reported"] = {"Female": "female"}[cbio_patients.loc[cbio_patient, "SEX"]]
        sample["metadata_notes"] += " cBioPortal SAMPLE_ID -> PATIENT_ID explicitly confirmed; lesion type and sex preserved from source."
    if key not in clinical.index and key not in cbio_samples.index:
        sample["metadata_notes"] += " No matching Rueda Supplementary Table 5 or cBioPortal clinical row; IDR identity key is retained and unavailable clinical fields remain blank."
    sample_rows.append(sample)
samples = pd.DataFrame(sample_rows, columns=samples.columns)
check_sample_vocabulary(samples)
patient_count = int(samples.patient_key.ne("").sum())
clinical_source_keys = set(clinical.index) | set(cbio_samples.index)
missing_clinical_samples = samples.loc[~samples.sample_id.map(normalize_metabric_id).isin(clinical_source_keys), "sample_id"].tolist()
missing_clinical_images = annotation.loc[annotation["Metabric ID"].isin(missing_clinical_samples), "Image Name"].tolist()
metadata_gaps = [f"{len(missing_clinical_samples)}份来源标本的{len(missing_clinical_images)}个image无匹配的Rueda Table 5/cBioPortal临床行；IDR身份键保留"] if missing_clinical_samples else []

fields = pd.read_csv(TEMPLATES / "field_mapping.csv", dtype=str)
field_rows = []


def mapping(source_file: Path, source_field: str, target: str, field: str, rule: str,
            *, keys: str = "Metabric ID -> METABRIC.ID", notes: str = "") -> None:
    record = dict.fromkeys(fields.columns, "")
    record.update(study=STUDY, source_file=relative(source_file), source_field=source_field,
        target_table=target, target_field=field, join_keys=keys, transformation_rule=rule,
        conflict_rule="source conflicts stop publication; optional unavailable values remain blank", notes=notes)
    field_rows.append(record)


mapping(Path(__file__).resolve(), "STUDY", "sample_metadata", "study",
    "Use the preparer's canonical study constant for every sample row.", keys="canonical preparer STUDY constant")
mapping(annotation_path, "Image Name;Image File;Metabric ID;Core ID;Image Number", "image_metadata", "image_id;source_image_id;source_files;sample_id;core_id;roi_id",
    "Image Name unchanged; Image File selects full source stack; sample_id=Metabric ID; core_id=Metabric ID + ':' + Core ID; roi_id=Image Number. Multiple views of one core share core_id.",
    notes="548 acquisitions; 504 distinct (Metabric ID, Core ID) pairs; 483 original tumour specimens; patient identity comes only from the explicit Metabric ID annotation, never from image filenames.")
mapping(annotation_path, "Metabric ID", "sample_metadata", "sample_id;source_sample_id",
    "Group acquisitions by the exact native Metabric ID; one original tumour specimen per unique value.",
    keys="Metabric ID -> sample_id", notes="483/483 unique native specimen identifiers define biological sample rows.")
mapping(annotation_path, "Metabric ID", "sample_metadata", "patient_key;source_subject_id",
    "Preserve the explicit IDR Metabric ID as the study-scoped patient identity; normalize its MBxxxx form to MB-xxxx. This identity join does not assert that a Rueda Table 5 or cBioPortal clinical row exists.",
    keys="sample_id -> unique IDR Metabric ID",
    notes=f"483/483 sample groups have one unique explicit IDR Metabric ID; nonempty IDR cBioPortal URLs corroborate the normalized format where provided. {len(missing_clinical_samples)} groups lack matched Rueda Table 5/cBioPortal clinical rows, with clinical fields left blank.")
mapping(cbio_sample_path, "SAMPLE_ID;SAMPLE_TYPE", "sample_metadata", "lesion_type",
    "Exact normalized IDR Metabric ID -> SAMPLE_ID; SAMPLE_TYPE Primary -> lesion_type primary; missing row leaves lesion_type blank.", keys="IDR Metabric ID (MBxxxx -> MB-xxxx) = SAMPLE_ID")
field_rows[-1]["value_mapping"] = json.dumps({"Primary": "primary"})
mapping(cbio_patient_path, "PATIENT_ID;SEX", "sample_metadata", "sex_reported",
    "Join through the explicit cBioPortal SAMPLE_ID -> PATIENT_ID mapping; SEX Female -> female.", keys="cBioPortal sample PATIENT_ID = clinical patient PATIENT_ID")
field_rows[-1]["value_mapping"] = json.dumps({"Female": "female"})
mapping(annotation_path, "Characteristics [Age at Diagnosis]", "sample_metadata", "age_at_diagnosis_years",
    "Cross-check diagnosis age against Rueda Supplementary Table 5 within 1e-4 years; use Table 5 precision when available.")
mapping(annotation_path, "Characteristics [Diagnosis];Characteristics [Organism Part]", "sample_metadata", "primary_diagnosis;primary_site;tissue_type;anatomical_site",
    "Preserve diagnosis and organ source values; tissue_type=tumor and anatomical_site=breast from original study tumour specimen definition.")
mapping(sources / "publication.html", "study specimen description", "sample_metadata", "sample_type;preservation_method;cohort_id",
    "Original breast tumour specimens in METABRIC; normalize FFPE -> ffpe per vocabulary. All cores from one tumour share sample_id.", notes="DOI 10.1038/s43018-020-0026-6; original study describes 483 formalin-fixed paraffin-embedded breast tumour samples.")
mapping(full_path, "Histological.Type;Grade", "sample_metadata", "histology;grade", "Exact METABRIC.ID join; preserve source category text; NA -> blank.")
mapping(full_path, "Age.At.Diagnosis", "sample_metadata", "age_at_diagnosis_years",
    "Exact METABRIC.ID join; use the published numeric years value when available; cross-check against the IDR annotation within 1e-4 years.")
mapping(full_path, "Cohort", "sample_metadata", "center_id",
    "Exact METABRIC.ID join; preserve the source hospital-of-origin code (1-5) without inferring named centers.")
mapping(full_path, "Stage", "sample_metadata", "source_tumor_stage",
    "Exact METABRIC.ID join; preserve raw source tumour stage; NA -> blank. Source dictionary specifies Tumor Stage without index origin/system, so stage_group_at_index remains blank.",
    notes=json.dumps({"data_type": "string", "definition": "Original Rueda 2019 tumour stage; no clinical index origin or staging system assigned"}))
for source_field, target_field in full_source_fields.items():
    mapping(full_path, source_field, "sample_metadata", target_field,
        "Exact METABRIC.ID join; preserve the published value verbatim; source NA is blank.",
        notes=("Source dictionary defines this field as days; time origin is unspecified."
               if source_field in {"T", "TLR", "TDR"} else ""))
mapping(full_path, "Last.Followup.Status", "sample_metadata", "vital_status",
    "Exact METABRIC.ID join; use the source dictionary's alive/death codes; preserve the original code separately.")
field_rows[-1]["value_mapping"] = json.dumps(followup_status_to_vital_status)
mapping(molecular_path, "Pam50Subtype", "sample_metadata", "molecular_subtype", "Exact METABRIC.ID join; preserve source category; NA -> blank.")
for source_field, target_field in molecular_source_fields.items():
    mapping(molecular_path, source_field, "sample_metadata", target_field,
        "Exact METABRIC.ID join; preserve the published value verbatim; source NA is blank.")
mapping(Path(__file__).resolve(), "STUDY;IDR Metabric ID;Rueda METABRIC.ID;cBioPortal clinical availability", "sample_metadata", "metadata_notes",
    "Record the direct IDR identity key separately from optional clinical-source joins; note unspecified clinical time origins without assigning derived endpoints.")
mapping(metadata / "channel_map.csv", "channel_index_zero_based;source_channel_label;metal;source_marker", "panel_mapping", "source_channel_index;source_name;s1_original_marker",
    "Native ordered full-stack labels verified against every IDR annotation row, existing ordered proof and metal_to_marker.csv; use source_marker as study-specific S1 key. Output indices enumerate retained channels in original source order.",
    notes="35 retained channels: 33 downstream markers and 2 DNA segmentation inputs. S1 excludes modified histone/kinase channels. All 50 native source rows retained in panel table.")
mapping(sources / "source_trace.json", "publication Figure 1b / microscopy method", "image_metadata", "pixel_size_x_um;pixel_size_y_um",
    "1 micrometre native IMC pixel size on both axes; no resampling.")
mapping(intake / "accepted_index.json", "native proofs;source shape", "image_metadata", "source_revision;source_dtype;source_axes;source_x0_px;source_y0_px;source_x1_px;source_y1_px;split_method;split_evidence",
    "Reuse 548 accepted native proofs (27 legacy + 521 per-image); verify current byte size and available mtime; full independent acquisition frame, CYX float32. Symmetric zero padding only; no coordinate rotation/cropping.",
    notes="No unmeasured pixels declared within full stacks; all-zero pixels remain measured. Input admission uses nonzero retained downstream marker anywhere in whole image.")
fields = pd.DataFrame(field_rows, columns=fields.columns)

images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists() else pd.read_csv(TEMPLATES / "image_metadata.csv")
images = images.set_index("image_id", drop=False)
receipts_path = cache / "export_receipts.json"
receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
if args.export or not (args.verify or args.overview):
    save_tables(images.reset_index(drop=True))
    pending = len(annotation) - len(receipts)
    progress("metadata_ready", images, metadata_gaps +
        ([f"{pending}个来源图像等待导出或接续"] if pending else []) + ["独立复核待执行"])

if args.export:
    assert shutil.disk_usage(root).free - output_estimate > 100 * 1024**3
    assert shutil.disk_usage("/tmp").free > 64 * 1024**3
    exported_this_run = 0
    for native, item in zip(annotation.to_dict("records"), inventory):
        image_id = native["Image Name"]
        destination = output / "images" / image_id / "input.tiff"
        if image_id in images.index and image_id in receipts:
            saved = receipts[image_id]
            assert destination.stat().st_size == saved["output_size"]
            assert destination.stat().st_mtime_ns == saved["output_mtime_ns"]
            assert saved["source_mtime_ns"] == item["source_mtime_ns"]
            assert saved["kept_source_indices"] == kept.source_channel_index.tolist()
            assert saved["channel_names"] == kept.marker_id.tolist()
            continue
        if args.limit is not None and exported_this_run >= args.limit:
            break
        try:
            source_path = root / item["source_file"]
            source = SourceSpec(image_id, PANEL, "tiff", source_path, tuple(range(50)), pixel_size_um=(1.0, 1.0))
            with open_source(source) as reader:
                assert list(reader.shape) == item["shape"] and reader.dtype == np.dtype("float32")
                report = export_input(reader, panel, destination)
            receipts[image_id] = dict(report, source_mtime_ns=item["source_mtime_ns"],
                source_bytes=item["source_bytes"], kept_source_indices=kept.source_channel_index.tolist(),
                channel_names=kept.marker_id.tolist())
            if report["admitted"]:
                height, width = item["shape"][1:]
                record = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
                record.update(study=STUDY, image_id=image_id, sample_id=native["Metabric ID"],
                    panel_id=PANEL, image_kind="roi", source_image_id=image_id,
                    source_files=[item["source_file"]], source_format="TIFF", source_axes="CYX",
                    source_dtype="float32", source_revision="IDR0076 20191219-ftp; accepted native proof",
                    core_id=native["Metabric ID"] + ":" + native["Core ID"], roi_id=native["Image Number"],
                    region_annotation=native["Characteristics [Diagnosis]"],
                    source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                    pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                    intensity_description=native["Comment [Image File Comments]"] + "; native float32 decoded values; no intensity normalization",
                    split_method="already_independent", split_evidence=relative(annotation_path),
                    metadata_notes="Full source acquisition, including measured zero/background pixels. IDR Metabric ID provides sample link; source Core ID is scoped by specimen.")
                record.update({k: report[k] for k in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]})
                images.loc[image_id] = pd.Series(record)
                receipts[image_id].update(output_size=destination.stat().st_size, output_mtime_ns=destination.stat().st_mtime_ns)
            save_json(receipts_path, receipts)
            save_tables(images.reset_index(drop=True))
            exported_this_run += 1
            progress("exporting", images, metadata_gaps + ["独立复核待执行"])
            print(json.dumps(dict(image_id=image_id, completed_images=len(images), admitted=report["admitted"])), flush=True)
        except Exception as error:
            progress("export_failed", images, [f"{image_id}: {type(error).__name__}: {error}"])
            raise
    progress("export_complete" if len(receipts) == len(annotation) else "export_partial", images,
             metadata_gaps + (["独立复核待执行"] if len(receipts) == len(annotation) else [f"尚余{len(annotation)-len(receipts)}个来源图像未处理", "独立复核待执行"]))

if args.overview:
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Representative full ROI views cover small/large fields and extreme aspect ratios.
    ranked = sorted(inventory, key=lambda item: np.prod(item["shape"][1:]))
    representatives = list(dict.fromkeys([inventory[0]["image_id"], inventory[len(inventory)//2]["image_id"],
        inventory[-1]["image_id"], ranked[0]["image_id"], ranked[-1]["image_id"],
        max(inventory, key=lambda item: max(item["shape"][1:])/min(item["shape"][1:]))["image_id"]]))
    fig, axes = plt.subplots(len(representatives), 3, figsize=(10, 3*len(representatives)), squeeze=False)
    for i, image_id in enumerate(representatives):
        path = raw / "full_stacks" / (image_id + "_fullstack.tiff")
        with tifffile.TiffFile(path) as tif:
            for j, (label, indices) in enumerate([("DNA1", [42]), ("DNA2", [43]), ("PanCK / CD45 / Vimentin", [40, 27, 15])]):
                values = [tif.pages[index].asarray(maxworkers=2) for index in indices]
                normalized = [np.clip(value/max(float(np.quantile(value[value>0], .995)) if np.any(value>0) else 1, 1e-12), 0, 1) for value in values]
                axes[i,j].imshow(normalized[0] if len(values)==1 else np.stack(normalized, axis=-1), cmap="gray", vmin=0, vmax=1)
                axes[i,j].set_title(image_id + "\n" + label, fontsize=8)
                axes[i,j].axis("off")
    fig.tight_layout()
    fig.savefig(cache / "representative_overview.jpg", dpi=120)
    plt.close(fig)
    save_json(cache / "overview_selection.json", dict(image_ids=representatives,
        source="original full-stack TIFF acquisition frame",
        display_only="P99.5 display scaling of positive pixels; input.tiff retains all native intensities"))
    if not (args.export or args.verify):
        raise SystemExit(0)

if args.verify:
    progress("verifying", images, metadata_gaps + ["正在核对全部头信息、几何、临床值及代表图像素；独立复核待执行"])
    # Reopen saved clinical values and validate every nonempty category against
    # the declared vocabulary and the original specimen/clinical join.
    saved_samples = pd.read_csv(output / "sample_metadata.csv", dtype=str, keep_default_na=False)
    check_sample_vocabulary(saved_samples)
    assert len(saved_samples) == annotation["Metabric ID"].nunique()
    assert saved_samples.sample_id.is_unique and saved_samples.patient_key.is_unique
    for row in saved_samples.to_dict("records"):
        native = annotation[annotation["Metabric ID"].eq(row["sample_id"])].iloc[0]
        key = native.patient_join
        assert row["source_sample_id"] == native["Metabric ID"]
        assert row["patient_key"] == row["source_subject_id"] == key
        expected_age = (clinical.loc[key, "Age.At.Diagnosis"]
                        if key in clinical.index and clinical.loc[key, "Age.At.Diagnosis"]
                        else native["Characteristics [Age at Diagnosis]"])
        assert row["age_at_diagnosis_years"] == expected_age
        assert np.isclose(float(row["age_at_diagnosis_years"]),
                          float(native["Characteristics [Age at Diagnosis]"]), rtol=0, atol=1e-4)
        for target, source in [("histology", "Histological.Type"), ("grade", "Grade"), ("source_tumor_stage", "Stage")]:
            assert row[target] == (clinical.loc[key, source] if key in clinical.index else "")
        assert row["molecular_subtype"] == (molecular.loc[key, "Pam50Subtype"] if key in molecular.index else "")
        cbio_patient = cbio_samples.loc[key, "PATIENT_ID"] if key in cbio_samples.index else ""
        assert row["sex_reported"] == ("female" if cbio_patient and cbio_patient in cbio_patients.index else "")
        assert row["lesion_type"] == ("primary" if key in cbio_samples.index else "")
        assert row["stage_group_at_index"] == ""
    # Verify every output header and two bounded pixel windows for every retained channel
    # in representative size/aspect cases, using direct source TIFF page reads.
    checked = []
    pixel_checks = []
    selection_path = cache / "overview_selection.json"
    selected = json.loads(selection_path.read_text())["image_ids"] if selection_path.exists() else [inventory[0]["image_id"]]
    inventory_by_id = {item["image_id"]: item for item in inventory}
    for record in images.to_dict("records"):
        image_id = record["image_id"]
        native_h, native_w = inventory_by_id[image_id]["shape"][1:]
        side = max(native_h, native_w)
        assert (record["source_x0_px"], record["source_y0_px"], record["source_x1_px"], record["source_y1_px"]) == (0, 0, native_w, native_h)
        assert record["pad_left_px"] == (side-native_w)//2 and record["pad_top_px"] == (side-native_h)//2
        assert record["pad_left_px"] + native_w + record["pad_right_px"] == side
        assert record["pad_top_px"] + native_h + record["pad_bottom_px"] == side
        assert record["height_px"] == record["width_px"] == side
        assert record["source_channel_count"] == 50 and record["channel_count"] == 35
        with tifffile.TiffFile(root / record["input_image_path"]) as tif:
            assert tif.series[0].shape == (35, record["height_px"], record["width_px"])
            assert tif.series[0].dtype == np.dtype("float32")
            ome = ET.fromstring(tif.ome_metadata)
            names = [x.attrib["Name"] for x in ome.iter() if x.tag.endswith("}Channel")]
            assert names == kept.marker_id.tolist()
            if image_id in selected:
                # OME series discovery caches TiffFrame objects, whereas the
                # shared bounded reader currently requires complete TiffPage tags.
                # Study-local adaptation pending shared TiffFrame support.
                tif.pages.cache = False
                tif.pages.useframes = False
                with tifffile.TiffFile(raw / "full_stacks" / (image_id + "_fullstack.tiff")) as source:
                    h, w = source.pages[0].shape
                    left, top = record["pad_left_px"], record["pad_top_px"]
                    dna = source.pages[42].asarray(maxworkers=2)
                    assert dna.max() > 0
                    peak_y, peak_x = np.unravel_index(dna.argmax(), dna.shape)
                    x = max(0, min(int(peak_x)-16, w-32))
                    y = max(0, min(int(peak_y)-16, h-32))
                    windows = [(x,y,min(w,x+32),min(h,y+32)), (max(0,w-32),max(0,h-32),w,h)]
                    for kind, window in zip(["signal", "boundary"], windows):
                        x0,y0,x1,y1 = window
                        for out, index in enumerate(kept.source_channel_index):
                            expected = read_tiff_window(source, index, window)
                            actual = read_tiff_window(tif, out, (x0+left,y0+top,x1+left,y1+top))
                            assert expected.dtype == actual.dtype
                            assert np.array_equal(expected, actual), (image_id,index,window)
                        pixel_checks.append(dict(image_id=image_id, block_type=kind,
                            source_window=window, all_retained_channels_equal=True))
                    for out in range(35):
                        if top: assert not read_tiff_window(tif,out,(0,0,record["width_px"],top)).any()
                        if left: assert not read_tiff_window(tif,out,(0,0,left,record["height_px"])).any()
                        bottom = record["pad_bottom_px"]
                        right = record["pad_right_px"]
                        if bottom: assert not read_tiff_window(tif,out,(0,record["height_px"]-bottom,record["width_px"],record["height_px"])).any()
                        if right: assert not read_tiff_window(tif,out,(record["width_px"]-right,0,record["width_px"],record["height_px"])).any()
        checked.append(image_id)
    save_json(cache / "preparer_verification.json", dict(study=STUDY, role="preparer_self_check",
        headers_checked=len(checked), image_ids=checked, pixel_checks=pixel_checks,
        sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=patient_count,
        saved_sample_rows_checked=len(saved_samples), missing_patient_samples=missing_patient_samples,
        missing_patient_images=missing_patient_images, independent_review="pending"))
    progress("prepared_self_checked", images, metadata_gaps + ["独立复核待执行"])

# Self-check evidence is labelled explicitly; D01 acceptance remains for the reviewer.
reviews = pd.read_csv(TEMPLATES / "review_results.csv", dtype=str)
review_rows = []
for check, scope, result, details in [
    ("S01", "study", "pass", f"整理者自检：548个完整原生TIFF；复用27+521份证明；.part及range记录未动。当前导出{len(images)}；全零排除{sum(not x['admitted'] for x in receipts.values())}。"),
    ("P01", "panel", "pass", "整理者自检：50个来源通道与IDR每行Channels、metal map、既有有序证明及本study S1一致。"),
    ("P02", "panel", "pass", "整理者自检：35个保留通道=33 downstream+2 DNA核输入；来源行及输出索引分别保存；准入marker无重复。"),
    ("M01", "sample", "unresolved" if missing_patient_samples else "pass", json.dumps(dict(
        note=f"整理者来源追查：548个image可连接483份原始肿瘤sample；{patient_count}份sample可确认患者连接。已查IDR显式Metabric ID/患者URL、作者发布仓库、论文补充表、Rueda Supplementary Tables5/6/8及cBioPortal官方sample/patient表。其余{len(missing_patient_samples)}份sample仍缺患者连接，保持来源标本身份，不从命名推断患者。", image_ids=missing_patient_images,
        sample_ids=missing_patient_samples), ensure_ascii=False)),
    ("M02", "sample", "pass", "整理者自检：年龄为诊断年龄；grade、histology、stage及PAM50通过公开原始临床附件匹配；源NA留空，未填不明确的终点、治疗时点、stage系统。"),
    ("D01", "study", "unresolved", f"已导出{len(images)}/548；本表为整理者工作记录。独立reviewer尚未执行校对，不能作为独立验收或程序消费绑定。"),
]:
    record = dict.fromkeys(reviews.columns, "")
    record.update(study=STUDY, preparer="Codex CLI Ali2020 0011_prepare_0", scope=scope,
        record_id=PANEL if scope=="panel" else STUDY, check_id=check, result=result,
        source_evidence=relative(annotation_path) + "; " + relative(sources / "source_trace.json"),
        output_evidence=relative(output), details=details)
    review_rows.append(record)
if args.verify:
    for check, details in [
        ("I01", "全部image主键及来源路径可定位；完整基础39字段经共享发布校验。"),
        ("I02", f"全部{len(images)}个image的sample/panel外键可连接；样本表483行。"),
        ("I03", "完整来源ROI框，原生float32、1um尺度；全部输出为方形，补边值在代表图全部通道核实。"),
        ("I04", "IDR显式Metabric ID/Core ID/Image Number界定独立来源ROI；保留完整采集范围和来源诊断注释；概览覆盖来源尺寸及长宽比极端。"),
        ("P03", f"全部{len(checked)}个TIFF重开检查CYX形状、float32、35个标准化名称及顺序。"),
        ("I05", json.dumps(dict(note="整理者自检；独立reviewer尚未复查", checks=pixel_checks), ensure_ascii=False)),
    ]:
        record = dict.fromkeys(reviews.columns, "")
        record.update(study=STUDY, preparer="Codex CLI Ali2020 0011_prepare_0", scope="study",
            record_id=STUDY, check_id=check, result="pass", source_evidence=relative(annotation_path),
            output_evidence=relative(cache / "preparer_verification.json"), details="整理者自检：" + details)
        review_rows.append(record)
review_path = output / "review_results.csv"
with formal_publish_lease(review_path, directory=output):
    pd.DataFrame(review_rows, columns=reviews.columns).to_csv(review_path, index=False, encoding="utf-8-sig")
print(json.dumps(dict(study=STUDY, expected_images=548, exported_images=len(images), sample_rows=len(samples),
    sample_linked_images=len(images), patient_linked_samples=patient_count, output_estimate_bytes=output_estimate)), flush=True)
