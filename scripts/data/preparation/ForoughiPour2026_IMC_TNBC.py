"""Deliver native IMC TMA inputs from Zenodo 17886494, without normalization.

Inputs: 12 complete MCD files, accepted native proofs, ROI/PID key workbooks,
clinical workbook, paper methods/antibody supplement and study-specific S1.
Outputs: independent square float32 CYX TIFFs, complete input tables and
preparer evidence. Run --prepare, then --export through batch shared formal write lease;
run --verify through shared formal read lease. Clinical identities are loaded only on NAS.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import io
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from PIL import Image, ImageDraw
from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

pa.set_cpu_count(3)
pa.set_io_thread_count(1)

STUDY = "ForoughiPour2026_IMC_TNBC"
PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "zenodo_17886494"
META = metadata_source_directory(ROOT, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0033_prepare_0"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PREPARER = "codex_cli_0033_prepare_0"


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, ROOT))


def template(name: str) -> pd.DataFrame:
    return pd.read_csv(TEMPLATES / f"{name}.csv", keep_default_na=False)


def progress(stage: str, **details: object) -> None:
    """Record actual resumable scope, without claiming independent acceptance."""
    current = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else template("image_metadata")
    record = dict(
        study=STUDY, stage=stage, expected_images=sum(x["eligible"] for x in plan),
        source_acquisitions=len(plan), completed_images=len(current),
        sample_rows=len(samples), sample_linked_images=int(current.sample_id.fillna("").ne("").sum()),
        patient_linked_samples=int(samples.patient_key.fillna("").ne("").sum()),
        gaps=["Two image-to-specimen associations unresolved: conflicting clinical collection records", "Independent input review pending"],
        updated_at=datetime.now(timezone.utc).isoformat())
    record.update(details)
    write_json(JOB / "progress.json", record)


def field(source: str, source_field: str, target: str, target_field: str,
          rule: str, *, sheet: str = "", notes: str = "", join: str = "MCD basename + ROI; key.PID = clinical.PID",
          mapping: dict | None = None) -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table=sheet,
                source_field=source_field, target_table=target, target_field=target_field,
                join_keys=join, value_mapping=json.dumps(mapping or {}, ensure_ascii=False),
                transformation_rule=rule, time_origin_rule="No inferred index, OS/DFS origin or follow-up origin.",
                conflict_rule="Consistent duplicate values coalesced; conflicting nonblank values retained in source evidence and target left blank.",
                notes=notes)


def source_spec(item: dict) -> SourceSpec:
    return SourceSpec(item["image_id"], item["panel_id"], "mcd", ROOT / item["source_file"],
                      tuple(range(40)), acquisition_id=item["acquisition_id"],
                      pixel_size_um=tuple(item["pixel_size_um"]), source_revision="zenodo:17886494:v1")


def header(path: Path, selected: pd.DataFrame, side: int) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (len(selected), side, side)
        assert all(p.dtype == np.dtype("float32") for p in tif.pages)
        channels = [x.attrib["Name"] for x in ET.fromstring(tif.ome_metadata).iter()
                    if x.tag.endswith("}Channel")]
        assert channels == selected.marker_id.tolist()


def save_tables() -> None:
    """Merge new images with the existing complete index after every image."""
    frame = pd.DataFrame(images) if images else template("image_metadata")
    save_input_tables(OUTPUT, frame, samples, panels, fields)


def reviews(stage: str) -> None:
    """Save preparer self-check evidence; D01 stays unresolved for another CLI."""
    rows = []
    checks = template("review_checks")
    for check in checks.itertuples(index=False):
        detail = {"stage": stage, "performed_by": "preparer", "independent_review": "pending"}
        result = "unresolved"
        if check.check_id == "S01":
            detail.update(source_acquisitions=len(plan), tissue_candidates=sum(x["eligible"] for x in plan),
                          cell_line_controls=sum(not x["eligible"] for x in plan),
                          exclusion_basis="Paper Data collection identifies cell-line spots; V1 input scope is human tissue.",
                          source_scope=relative(CACHE / "source_plan.json"),
                          control_identity_evidence=relative(META / "source_metadata/cell_line_controls.json"))
        elif check.check_id == "M01":
            detail.update(sample_rows=len(samples), sample_linked_images=sum(bool(x.get("sample_id")) for x in images),
                          patient_linked_samples=len(samples),
                          image_linked_samples=len({x["sample_id"] for x in images if x.get("sample_id")}),
                          clinical_only_samples=len(samples)-len({x["sample_id"] for x in plan if x["sample_id"]}),
                          image_ids=[x["image_id"] for x in plan if x["eligible"] and not x["sample_id"]],
                          unresolved_image_ids=[x["image_id"] for x in plan if x["eligible"] and not x["sample_id"]],
                          gap="One PID has two conflicting clinical specimen records without an ROI-to-specimen disambiguation key.",
                          evidence=relative(CACHE / "sample_identity_evidence.json"))
        elif check.check_id == "M02":
            detail.update(gap="Native Stage_2 and survival/follow-up values retained in source fields; OS/DFS cutoff units and all endpoint origins/index are not established.",
                          evidence=relative(CACHE / "sample_identity_evidence.json"))
        elif check.check_id == "I05":
            detail.update(evidence=relative(CACHE / "pixel_checks.json"))
        elif check.check_id == "D01":
            detail.update(exported_images=len(images), expected_images=sum(x["eligible"] for x in plan),
                          image_ids=[x["image_id"] for x in images],
                          message="Preparer delivery only. Independent reviewer must check and bind the accepted input scope.")
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                         record_id=STUDY, check_id=check.check_id, result=result,
                         source_evidence=relative(CACHE / "source_plan.json"), output_evidence=relative(OUTPUT),
                         details=json.dumps(detail, ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8-sig")


# Explicit modes keep metadata work outside the shared heavy-I/O slot.
parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--prepare", action="store_true")
mode.add_argument("--export", action="store_true")
mode.add_argument("--verify", action="store_true")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

if args.prepare:
    # Reuse accepted file proofs, checking size/mtime rather than hashing again.
    manifest = json.loads((META / "source_manifest.json").read_text())
    verification = json.loads((META / "acquisition_verification.json").read_text())
    channel_map = json.loads((META / "channel_mapping.json").read_text())
    native_map = json.loads((META / "mcd_source_mapping.json").read_text())
    verified_files = {x["source_file"]: x for x in verification["files"]}
    mapped_files = {x["source_file"]: x for x in channel_map["source_files"]}
    original_files = {x["file"]: x for x in native_map["files"]}
    assert {p.name for p in RAW.glob("*.mcd")} == set(verified_files)
    clinical = pd.read_excel(RAW / "TNBC_clin_data_upload.xlsx", dtype={"PID": str}, keep_default_na=False)
    clinical = clinical.map(lambda value: value.strip() if isinstance(value, str) else value)
    clinical["excel_row"] = np.arange(len(clinical)) + 2

    # Unique clinical tumor collections define samples; serial sections/cores
    # share that collection. Conflicting specimen records never select a row.
    clinical_by_pid = {}
    sample_rows, identity_evidence = [], []
    raw_extensions = {
        "Stage_2": ("source_stage_2", "string", "Unmodified Stage_2; index, staging system and basis unspecified"),
        "Followup_Days": ("source_followup_days", "number", "Source Followup_Days; time origin unspecified"),
        "Overall_survival_cut_off_from_last_communication_2": ("source_os_cutoff_value", "number", "Unmodified source overall-survival cutoff number; unit and origin unspecified"),
        "DFS_cut_off_from_last_clinic_visit_2": ("source_dfs_cutoff_value", "number", "Unmodified source DFS cutoff number; unit and origin unspecified"),
        "Neoadj_chemo_2": ("source_neoadjuvant_chemotherapy", "string", "Source neoadjuvant chemotherapy description"),
        "Primary_Recurrence_2": ("source_primary_recurrence", "string", "Source primary/recurrence/other tumor annotation"),
        "Surgery_2": ("source_surgery", "string", "Source surgery description"),
        "Race": ("source_race", "string", "Source reported race; no recoding"),
    }
    for pid, group in clinical.groupby("PID", sort=False):
        merged, conflicts = {}, {}
        for key in clinical.columns:
            if key in {"Unnamed: 0", "excel_row"}:
                continue
            values = list(dict.fromkeys(value for value in group[key].tolist() if value != ""))
            merged[key] = values[0] if len(values) == 1 else ""
            if len(values) > 1:
                conflicts[key] = values
        clinical_by_pid[pid] = dict(values=merged, conflicts=conflicts,
                                    excel_rows=group.excel_row.tolist(), sample_id="")
        identity_evidence.append(dict(source_subject_id=pid, excel_rows=group.excel_row.tolist(),
                                      conflicts=conflicts, source_records=group.to_dict("records")))

    # The two native panel IDs are historical local names and invert the paper
    # panel numbers. Match by ordered channels, never by the numeric suffix.
    panel_rows = []
    for source_panel in channel_map["panels"]:
        names = [x["native_label"] for x in source_panel["ordered_channels"]]
        # Supplementary Data 1 Sheet2: 165Ho uses the same commercial
        # EPR4877(2) reagent in both panels; 175Lu uses panel-dependent reagents.
        # Choose the common 165Ho measurement for cross-panel consistency.
        assert names[26] == names[36] == "PD-1"
        panel = panel_from_s1(STUDY, source_panel["panel_id"], names, S1,
                              downstream_source_indices=set(range(40)) - {36})
        panel["source_mass"] = [x["mass"] for x in source_panel["ordered_channels"]]
        panel["source_metal"] = [x["metal"] for x in source_panel["ordered_channels"]]
        panel.loc[panel.source_channel_index == 26, "keep_reason"] = "Common 165Ho EPR4877(2) StandardBiotools reagent across both panels; Supplementary Data 1 Sheet2"
        panel.loc[panel.source_channel_index == 36, "exclusion_reason"] = "Duplicate PD-1: retain the common 165Ho reagent; 175Lu reagent differs between source panels; not a selected segmentation input"
        panel_rows.append(panel)
    panels = pd.concat(panel_rows, ignore_index=True)

    # Join keys to actual MCD acquisition descriptions, labels and dimensions.
    plan = []
    with zipfile.ZipFile(RAW / "keys.zip") as archive:
        members = {n for n in archive.namelist() if n.endswith(".xlsx")}
        expected_members = set()
        for source_file in sorted(manifest["source_files"], key=lambda x: x["source_file"]):
            name = source_file["source_file"]
            path = ROOT / source_file["destination_rel"]
            proof = verified_files[name]
            stat = path.stat()
            assert proof["verification_complete"] and stat.st_size == proof["actual_bytes"] == source_file["expected_bytes"]
            assert stat.st_mtime_ns == proof["file_stat"]["mtime_ns"]
            key_name = re.sub(r"TMA_TNBC(2)?_(\d+)\.mcd", lambda m: f"keys/mcd{2 if m[1] else 1}_{m[2]}.xlsx", name)
            expected_members.add(key_name)
            key_path = CACHE / key_name
            if not key_path.exists():
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_bytes(archive.read(key_name))
            key = pd.read_excel(key_path, dtype=str, keep_default_na=False).set_index("ROI", verify_integrity=True)
            old_acquisitions = {x["acquisition_id"]: x for x in original_files[name]["acquisitions"]}
            acquisition_proof = json.loads((META / "native_verification" / Path(proof["proof_path"]).name).read_text())
            assert all(x["complete"] for x in acquisition_proof["acquisitions"])
            with MCDFile(path) as mcd:
                acquisitions = [(slide, acq) for slide in mcd.slides for acq in slide.acquisitions]
                assert len(acquisitions) == len(key) == proof["verified_acquisition_count"]
                assert {a.metadata["Description"] for _, a in acquisitions} == set(key.index)
                for slide, acq in acquisitions:
                    old = old_acquisitions[acq.id]
                    assert acq.channel_labels == old["channel_labels"]
                    assert acq.channel_masses == old["channel_masses"]
                    assigned_panel = panels[panels.panel_id == mapped_files[name]["panel_id"]].sort_values("source_channel_index")
                    assert acq.channel_labels == assigned_panel.source_name.tolist()
                    assert acq.channel_masses == assigned_panel.source_mass.tolist()
                    assert acq.channel_metals == assigned_panel.source_metal.tolist()
                    width, height = int(acq.metadata["MaxX"]), int(acq.metadata["MaxY"])
                    assert (width, height) == (old["width_px"], old["height_px"])
                    pid = key.loc[acq.metadata["Description"], "PID"]
                    clinical_info = clinical_by_pid.get(pid)
                    # The keys identify the five named cell-line controls, which
                    # the paper explicitly separates from tissue spots. A key
                    # absent from the clinical table must be author-confirmed.
                    cell_lines = {"BT474", "BT549", "MB435", "MB468", "SKBR3"}
                    if clinical_info is None:
                        assert pid in cell_lines
                    eligible = clinical_info is not None and clinical_info["values"]["Core Type"] == "Tumor"
                    image_id = f"{path.stem}__{acq.metadata['Description']}"
                    sample_id = f"tumor_specimen_{pid}" if eligible and not clinical_info["conflicts"] else ""
                    if sample_id:
                        clinical_info["sample_id"] = sample_id
                    plan.append(dict(image_id=image_id, source_file=relative(path), acquisition_id=acq.id,
                                     roi_id=acq.metadata["Description"], source_subject_id=pid,
                                     sample_id=sample_id, eligible=eligible,
                                     exclusion_reason="" if eligible else "Cell-line TMA control; outside human tissue input scope",
                                     panel_id=mapped_files[name]["panel_id"], width=width, height=height,
                                     slide_id=f"{path.stem}:slide{slide.id}", source_slide_name=slide.metadata["Name"],
                                     roi_points_um=acq.roi_points_um,
                                     pixel_size_um=[float(acq.metadata["AblationDistanceBetweenShotsX"]), float(acq.metadata["AblationDistanceBetweenShotsY"])],
                                     proof_path=proof["proof_path"], source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                     key_file=relative(key_path), region_annotation=acq.metadata["Description"]))
        assert members == expected_members
    assert len(plan) == verification["verified_acquisition_count"] == 299
    # Clinical-only confirmed tumor specimens are retained even without a local
    # ROI key; they add sample metadata, not images or inferred missing images.
    sample_pids = {pid for pid, info in clinical_by_pid.items()
                   if info["values"]["Core Type"] == "Tumor" and not info["conflicts"]}
    for pid in sorted(sample_pids):
        info = clinical_by_pid[pid]
        values = info["values"]
        info["sample_id"] = f"tumor_specimen_{pid}"
        local_images = [x["image_id"] for x in plan if x["source_subject_id"] == pid and x["eligible"]]
        row = dict.fromkeys(template("sample_metadata").columns, "")
        row.update(study=STUDY, sample_id=info["sample_id"], patient_key=f"{STUDY}:{pid}", source_subject_id=pid,
                   cohort_id="Yale_TNBC_TMA", center_id="Yale", sample_type="tissue", tissue_type="tumor",
                   anatomical_site=values["site_of_bx_2"],
                   processing_method="TMA spot; adjacent 5-micron sections for two IMC panels",
                   primary_diagnosis=values["Diagnosis"], age_at_diagnosis_years=values["Age_at_Diagnosis"],
                   histology=values["Histology"], grade=values["Histologic_Grade"],
                   vital_status={"living": "alive", "deceased": "deceased"}.get(values["living_deceased"], ""),
                   lesion_type={"primary": "primary", "recurrence": "recurrence", "NSCLC met in breast": "metastasis"}.get(values["Primary_Recurrence_2"], ""),
                   collection_method="resection" if values["Surgery_2"] in {"lumpectomy", "mastectomy"} else "",
                   metadata_notes=json.dumps(dict(
                       source_file=relative(RAW / "TNBC_clin_data_upload.xlsx"), sheet="Sheet1", excel_rows=info["excel_rows"],
                       biological_sample_rule="One unambiguous clinical tumor collection record per PID; all its source spots and adjacent sections share the biological sample. Paper Methods: annotated tumor block sampled for TMA. Original block accession and cross-section physical core correspondence unavailable.",
                       local_image_ids=local_images,
                       image_coverage="Source clinical specimen only; no local ROI key" if not local_images else "Source ROI keys linked",
                       index_and_endpoint_origin="not established; OS/DFS cutoff unit also unspecified; original values retained separately"), ensure_ascii=False))
        for original, (target, kind, _) in raw_extensions.items():
            value = values[original]
            row[target] = str(value) if kind == "string" and value != "" else value
        sample_rows.append(row)
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(CACHE / "samples_prepared.csv", index=False, encoding="utf-8-sig")
    panels.to_csv(CACHE / "panels_prepared.csv", index=False, encoding="utf-8-sig")
    write_json(CACHE / "source_plan.json", dict(study=STUDY, images=plan))
    write_json(CACHE / "sample_identity_evidence.json", dict(
        clinical_records=identity_evidence, sample_rows=len(samples),
        sample_linked_candidate_images=sum(bool(x["sample_id"]) for x in plan),
        image_linked_samples=len({x["sample_id"] for x in plan if x["sample_id"]}),
        clinical_only_samples=len(samples)-len({x["sample_id"] for x in plan if x["sample_id"]}),
        patient_linked_samples=len(samples), source_key_rows=len(plan),
        unresolved_images=[x["image_id"] for x in plan if x["eligible"] and not x["sample_id"]],
        author_repository="https://github.com/aforoughipour/SparTile",
        repository_files=json.loads((META / "source_metadata/author_tree.json").read_text())["tree"],
        repository_result="Inspected README, analysis module and 31-cell example notebook. Notebook uses one processed ROI and has no clinical or specimen/core map; repository tree has no additional map.",
        raw_identity_policy="PID is author-provided patient linkage; never inferred from image filenames.",
        gap="Conflicting duplicate specimen records have no image-to-record key. Original block accessions and matching physical core IDs across sections are not supplied."))
    write_json(CACHE / "public_source_trace.json", dict(
        checked_at=datetime.now(timezone.utc).isoformat(), sources=[
            dict(url="https://zenodo.org/records/17886494", cache=relative(META / "source_metadata/zenodo.html"), finding="Keys map IMC ROIs to corresponding patient; TNBC and TNBC2 are two panels."),
            dict(url="https://www.nature.com/articles/s43856-026-01400-4", cache=relative(META / "source_metadata/paper.html"), finding="Tumor blocks, multiple TMA spots and two adjacent sections; cell-line controls distinguished in Data collection."),
            dict(url="https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43856-026-01400-4/MediaObjects/43856_2026_1400_MOESM4_ESM.xlsx", cache=relative(META / "source_metadata/supplementary_data_1.xlsx"), finding="Sheet2 confirms PD-1 reagents at 165Ho and 175Lu; 165Ho reagent common across both panels."),
            dict(url="https://api.github.com/repos/aforoughipour/SparTile/git/trees/main?recursive=1", cache=relative(META / "source_metadata/author_tree.json"), finding="No further sample/core map.")],
        cell_line_control_identity_evidence=relative(META / "source_metadata/cell_line_controls.json"),
        panel_number_note="Historical foroughi_panel_1 = TNBC2 = paper panel 2; foroughi_panel_2 = TNBC = paper panel 1. Native ordered labels verified directly."))

    clinical_source = relative(RAW / "TNBC_clin_data_upload.xlsx")
    fields_list = [
        field(relative(META / "source_manifest.json"), "dataset_name", "image_metadata", "study", "Canonical study ID from the authorized task and accepted source manifest.", join="dataset_name = study"),
        field(relative(META / "source_manifest.json"), "dataset_name", "sample_metadata", "study", "Clinical records belong to the canonical source-manifest study.", join="dataset_name = study"),
        field(relative(META / "channel_mapping.json"), "source_files.source_file;panel_id;panels.ordered_channels", "image_metadata", "panel_id", "Assign the historical local panel ID only after matching native MCD labels, masses and metals to that panel's ordered channels. Historical panel_1 is paper panel 2; panel_2 is paper panel 1.", join="MCD basename + exact ordered native channels"),
        field(relative(RAW / "keys.zip"), "ROI;PID", "image_metadata", "sample_id", "mcd1_N maps TMA_TNBC_N; mcd2_N maps TMA_TNBC2_N. Match ROI to actual MCD Description. Join PID to unambiguous clinical tumor collection; share sample across cores/serial sections. Conflicting specimen identities leave image sample_id blank.", notes=f"{len(plan)} source ROIs; {len(samples)} sample rows; evidence in {relative(CACHE / 'sample_identity_evidence.json')}", sheet="12 keys/mcd*.xlsx"),
        field(clinical_source, "PID;Core Type", "sample_metadata", "sample_id;patient_key;source_subject_id", "Derive tumor_specimen_<PID> only for one unambiguous clinical tumor collection after consistent duplicate coalescing, including clinical-only specimens without a local ROI key. These rows add no image count. Patient identity is exact source PID; patient_key is study:PID, source_subject_id retains PID. Original specimen accession absent.", sheet="Sheet1"),
        field(relative(CACHE / "sample_identity_evidence.json"), "clinical_records;excel_rows;local ROI keys", "sample_metadata", "metadata_notes", "Serialize workbook/sheet/row provenance, the paper-based clinical-collection grouping rule, local image links and explicit endpoint/block/core identity gaps as JSON.", join="clinical.PID = author ROI key.PID"),
        field(relative(META / "source_metadata/paper.html"), "Methods/Data collection;IMC acquisition", "sample_metadata", "cohort_id;center_id;sample_type;tissue_type;processing_method", "Yale tissue TMA, tumor block collection, two adjacent 5-micron sections; multiple cores/sections of the same confirmed clinical collection share sample.", join="study methods + clinical Core Type=Tumor"),
        field(relative(META / "source_metadata/cell_line_controls.json"), "keys.PID;Cellosaurus identifiers;paper Data collection", "image_metadata", "image_kind", "Source keys contain 34 named cell-line control spots. Distinguish these from 265 human tissue acquisitions; cell-line objects remain in source_plan with explicit exclusion reason.", join="source ROI to author key label; known cell-line identities"),
        field(clinical_source, "site_of_bx_2;Diagnosis;Age_at_Diagnosis;Histology;Histologic_Grade", "sample_metadata", "anatomical_site;primary_diagnosis;age_at_diagnosis_years;histology;grade", "Whitespace-trimmed direct values; no inferred grade/stage/age conversion.", sheet="Sheet1"),
        field(clinical_source, "living_deceased", "sample_metadata", "vital_status", "Exact categories; blank remains blank.", sheet="Sheet1", mapping={"living": "alive", "deceased": "deceased"}),
        field(clinical_source, "Primary_Recurrence_2;Surgery_2", "sample_metadata", "lesion_type;collection_method", "Map explicit primary/recurrence/NSCLC met in breast; explicit lumpectomy/mastectomy to resection. Other values blank.", sheet="Sheet1"),
        field(relative(META / "mcd_source_mapping.json"), "MCD ID;Description;MaxX;MaxY;channel labels;AblationDistanceBetweenShotsX/Y", "image_metadata", "image_id;source_image_id;source_files;source_format;source_acquisition_id;source_dtype;source_axes;source_revision;slide_id;roi_id;region_annotation;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um", "Actual MCD header rechecked against accepted native metadata/proofs. Full acquisition [0,MaxX)x[0,MaxY); float32 CYX; measured step size; source Description retained. slide_id is MCD-basename-qualified native slide ID; shared generic Layout Name does not establish physical-slide equivalence. Cross-section physical core_id left blank.", join="MCD basename + acquisition ID"),
        field(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "source_channel_index;source_name;s1_original_marker;marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index;keep_reason;exclusion_reason", "Exact study/native-label lookup; retain nuclear inputs and admitted markers in native source order. Historical panel IDs invert paper numbering; compare actual ordered labels. The explicit PD-1 reagent rule refines duplicate keep/exclusion reasons.", join="Study + Original marker"),
        field(relative(META / "channel_mapping.json"), "panels.panel_id;ordered_channels.channel_index;mass;metal", "panel_mapping", "panel_id;source_mass;source_metal", "Keep native isotope identities and historical local panel IDs after the direct MCD channel-order check.", join="panel_id + source_channel_index"),
        field(relative(META / "source_metadata/supplementary_data_1.xlsx"), "Sheet2:PD-1;Metal;Clone;Vendor", "panel_mapping", "keep_channel;exclusion_reason;keep_reason", "Retain PD-1 165Ho source index26: EPR4877(2), StandardBiotools in both panels. Drop duplicate index36/175Lu, whose source reagent differs across panels. Selection is for reagent consistency, not an unmeasured signal-quality claim.", sheet="Sheet2", join="native mass + label + ordered panel"),
        field(relative(CACHE / "source_plan.json"), "width;height;pixel records", "image_metadata", "pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;source_channel_count;channel_count;input_image_path;valid_pixels_path;intensity_description;split_method;split_evidence;metadata_notes;image_kind", "Shared native export; complete source ROI, centered zero padding to max(H,W); missing records independent of intensity; all-zero admitted-marker image excluded. One full spot per acquisition per paper and source keys.", join="image_id"),
        field(relative(RAW / "keys.zip"), "PID", "image_metadata", "source_subject_id", "Preserve author patient key even where the clinical specimen join is ambiguous.", notes=json.dumps({"data_type": "string", "definition": "Author-provided ROI-to-patient key from keys.zip; specimen identity can remain unresolved"})),
    ]
    for original, (target, kind, definition) in raw_extensions.items():
        spec = dict(data_type=kind, definition=definition)
        if target == "source_followup_days":
            spec["unit"] = "days"
        fields_list.append(field(clinical_source, original, "sample_metadata", target,
                                 "Retain source value without assigning index or survival endpoint semantics.", sheet="Sheet1", notes=json.dumps(spec)))
    fields = pd.DataFrame(fields_list)
    fields.to_csv(CACHE / "fields_prepared.csv", index=False, encoding="utf-8-sig")
    images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
    save_tables()
    reviews("prepared")
    projected = sum(max(x["width"], x["height"]) ** 2 * int(panels.loc[(panels.panel_id == x["panel_id"]) & panels.keep_channel].shape[0]) * 4 for x in plan if x["eligible"])
    write_json(CACHE / "capacity.json", dict(uncompressed_output_bytes=projected, extraction_bytes=0,
                                             nas_free_bytes=shutil.disk_usage(ROOT).free, temporary_free_bytes=shutil.disk_usage("/tmp").free))
    progress("metadata_prepared", estimated_uncompressed_output_bytes=projected,
             gaps=["Conflicting duplicate clinical specimen PID: sample join blank for its two images", "Independent input review pending"])
    print(json.dumps(dict(stage="prepared", expected_images=sum(x["eligible"] for x in plan), samples=len(samples), panel_kept=panels.groupby("panel_id").keep_channel.sum().to_dict())))
    raise SystemExit(0)

# Reopen metadata artifacts; successful outputs are not decoded/re-exported on resume.
plan = json.loads((CACHE / "source_plan.json").read_text())["images"]
samples = pd.read_csv(CACHE / "samples_prepared.csv", keep_default_na=False)
panels = pd.read_csv(CACHE / "panels_prepared.csv", keep_default_na=False)
panels["output_channel_index"] = pd.to_numeric(panels.output_channel_index, errors="coerce").astype("Int64")
fields = pd.read_csv(CACHE / "fields_prepared.csv", keep_default_na=False)
images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records")

if args.export:
    completed = {x["image_id"] for x in images}
    progress("exporting", gaps=["Two candidate images have unresolved specimen association; patient keys retained", "Independent review pending"])
    for item in plan:
        if not item["eligible"]:
            continue
        panel = panels[panels.panel_id == item["panel_id"]]
        selected = panel[panel.keep_channel].sort_values("output_channel_index")
        destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
        receipt_path = CACHE / "exports" / f"{item['image_id']}.json"
        if item["image_id"] in completed:
            receipt = json.loads(receipt_path.read_text())
            assert receipt["source_size"] == item["source_size"] and receipt["source_mtime_ns"] == item["source_mtime_ns"]
            assert receipt["source_indices"] == selected.source_channel_index.tolist()
            header(destination, selected, max(item["width"], item["height"]))
            continue
        try:
            progress("exporting", current_image=item["image_id"])
            if shutil.disk_usage(ROOT).free < 100 * 1024 ** 3 + max(item["width"], item["height"]) ** 2 * len(selected) * 4:
                raise OSError("Insufficient NAS space with required 100 GiB reserve")
            if destination.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["source_size"] == item["source_size"] and receipt["source_mtime_ns"] == item["source_mtime_ns"]
                assert receipt["source_indices"] == selected.source_channel_index.tolist()
                stats = receipt["stats"]
            else:
                spec = source_spec(item)
                with open_source(spec) as reader:
                    stats = export_input(reader, panel, destination, tile_size=256)
                write_json(receipt_path, dict(source_size=item["source_size"], source_mtime_ns=item["source_mtime_ns"],
                                              source_indices=selected.source_channel_index.tolist(), stats=stats))
            if not stats["admitted"]:
                progress("exporting", excluded_zero_marker_image=item["image_id"])
                continue
            header(destination, selected, max(item["width"], item["height"]))
            row = dict.fromkeys(template("image_metadata").columns, None)
            row.update(study=STUDY, image_id=item["image_id"], sample_id=item["sample_id"], panel_id=item["panel_id"],
                       image_kind="tma_core", source_image_id=f"{Path(item['source_file']).name}:acquisition:{item['acquisition_id']}",
                       source_files=[item["source_file"]], source_format="mcd", source_acquisition_id=str(item["acquisition_id"]),
                       source_axes="CYX", source_dtype="float32", source_revision="zenodo:17886494:v1",
                       slide_id=item["slide_id"], roi_id=item["roi_id"], region_annotation=item["region_annotation"],
                       source_x0_px=0, source_y0_px=0, source_x1_px=item["width"], source_y1_px=item["height"],
                       pixel_size_x_um=item["pixel_size_um"][0], pixel_size_y_um=item["pixel_size_um"][1],
                       input_image_path=relative(destination), valid_pixels_path=relative(destination.with_name(stats["valid_pixels"])) if stats["valid_pixels"] else None,
                       intensity_description="Native float32 MCD Dual signal; no intensity transformation; native zero measurements retained; padding and absent acquisition records filled with zero",
                       split_method="source_acquisition", split_evidence=f"{item['key_file']};{item['proof_path']};paper Methods: each TMA spot is a separate ROI",
                       metadata_notes="Cross-section physical core map unavailable. slide_id identifies the container-qualified native slide record; physical slide equivalence across files is not established." + (" Conflicting clinical specimen records: sample_id unresolved; exact source patient key retained." if not item["sample_id"] else ""),
                       source_subject_id=item["source_subject_id"])
            for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"):
                row[key] = stats[key]
            images.append(row)
            save_tables()
            completed.add(item["image_id"])
            progress("exporting", last_completed_image=item["image_id"])
            print(f"exported {len(images)}/{sum(x['eligible'] for x in plan)} {item['image_id']}", flush=True)
        except Exception as error:
            reviews("interrupted")
            progress("error", current_image=item["image_id"], error=f"{type(error).__name__}: {error}")
            raise
    reviews("exported")
    progress("exported_pending_preparer_verification", gaps=["Two specimen associations unresolved", "Independent review pending"])

if args.verify:
    # All TIFF headers, per-image source signal receipts and geometry; representative
    # all-channel values use readimc directly, independent of the export reader.
    by_id = {x["image_id"]: x for x in plan}
    for row in images:
        item = by_id[row["image_id"]]
        selected = panels[(panels.panel_id == item["panel_id"]) & panels.keep_channel].sort_values("output_channel_index")
        header(ROOT / row["input_image_path"], selected, max(item["width"], item["height"]))
        if row.get("valid_pixels_path"):
            with tifffile.TiffFile(ROOT / row["valid_pixels_path"]) as valid:
                assert len(valid.pages) == 1
                assert valid.pages[0].shape == (row["height_px"], row["width_px"])
                assert valid.pages[0].dtype == np.dtype("uint8")
        receipt = json.loads((CACHE / "exports" / f"{item['image_id']}.json").read_text())
        assert receipt["source_size"] == item["source_size"] and receipt["source_mtime_ns"] == item["source_mtime_ns"]
        assert receipt["source_indices"] == selected.source_channel_index.tolist()
        assert receipt["stats"]["admitted"]
        assert any(n for n, role in zip(receipt["stats"]["channel_nonzero_pixels"], selected.role) if role == "downstream")
        assert row["source_x1_px"] + row["pad_left_px"] + row["pad_right_px"] == row["width_px"]
        assert row["source_y1_px"] + row["pad_top_px"] + row["pad_bottom_px"] == row["height_px"]
    selected_examples = []
    for panel_id in panels.panel_id.unique():
        subset = [x for x in images if x["panel_id"] == panel_id]
        selected_examples.append(subset[0])
        with_valid = [x for x in subset if pd.notna(x["valid_pixels_path"]) and x["valid_pixels_path"]]
        if with_valid and with_valid[0]["image_id"] != subset[0]["image_id"]:
            selected_examples.append(with_valid[0])
    comparisons, preview_records = [], []
    canvas = Image.new("RGB", (len(selected_examples) * 384, 430), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected_examples):
        item = by_id[row["image_id"]]
        selected = panels[(panels.panel_id == item["panel_id"]) & panels.keep_channel].sort_values("output_channel_index")
        w, h = item["width"], item["height"]
        regions = [(max(w//2-16, 0), max(h//2-16, 0), w//2+16, h//2+16), (w-32, h-32, w, h)]
        with MCDFile(ROOT / item["source_file"]) as mcd:
            acquisition = next(a for s in mcd.slides for a in s.acquisitions if a.id == item["acquisition_id"])
            native = mcd.read_acquisition(acquisition)
            acq_metadata = acquisition.metadata
        start, end = int(acq_metadata["DataStartOffset"]), int(acq_metadata["DataEndOffset"])
        records = np.memmap(ROOT / item["source_file"], dtype="float32", mode="r", offset=start,
                            shape=((end-start)//(43*4), 43))
        expected_valid = np.zeros((row["height_px"], row["width_px"]), dtype=bool)
        expected_valid[records[:, 1].astype(int)+row["pad_top_px"],
                       records[:, 0].astype(int)+row["pad_left_px"]] = True
        records._mmap.close()
        with tifffile.TiffFile(ROOT / row["input_image_path"]) as saved:
            for channel in selected.itertuples(index=False):
                for x0, y0, x1, y1 in regions:
                    actual = read_tiff_window(saved, channel.output_channel_index,
                                              (x0+row["pad_left_px"], y0+row["pad_top_px"], x1+row["pad_left_px"], y1+row["pad_top_px"]))
                    np.testing.assert_array_equal(actual, native[channel.source_channel_index, y0:y1, x0:x1])
                    assert actual.dtype == native.dtype == np.dtype("float32")
                # Check the full padding of each selected representative plane.
                plane = saved.pages[channel.output_channel_index].asarray(maxworkers=1)
                top, left = row["pad_top_px"], row["pad_left_px"]
                assert not plane[:top].any() and not plane[top+h:].any()
                assert not plane[:, :left].any() and not plane[:, left+w:].any()
            if row.get("valid_pixels_path"):
                measured = tifffile.imread(ROOT / row["valid_pixels_path"]).astype(bool)
                np.testing.assert_array_equal(measured, expected_valid)
            else:
                assert expected_valid.sum() == w*h
        x0, y0, x1, y1 = regions[0]
        interior_marker_nonzero = int(np.count_nonzero(
            native[selected[selected.role == "downstream"].source_channel_index.to_numpy(), y0:y1, x0:x1]))
        assert interior_marker_nonzero > 0
        comparisons.append(dict(image_id=row["image_id"], panel_id=row["panel_id"], regions=regions,
                                compared_channels=len(selected), exact=True, dtype="float32", full_padding_zero=True,
                                interior_marker_nonzero_channel_pixels=interior_marker_nonzero,
                                validity_matches_all_native_coordinates=True))
        # Display scaling is only for QC PNG, never applied to the input TIFF.
        groups = [selected[selected.nuclear].output_channel_index.tolist(),
                  selected[selected.marker_id.isin(["E-Cadherin/EpCAM", "PanCK"])].output_channel_index.tolist(),
                  selected[selected.marker_id == "CD45"].output_channel_index.tolist()]
        channels = []
        with tifffile.TiffFile(ROOT / row["input_image_path"]) as overview:
            for group in groups:
                signal = np.zeros((row["height_px"], row["width_px"]), dtype=np.float32)
                for channel in group:
                    signal += overview.pages[channel].asarray(maxworkers=1)
                limit = float(np.percentile(signal[signal > 0], 99)) if np.any(signal > 0) else 1.0
                channels.append(np.uint8(np.clip(signal/limit, 0, 1)*255))
        thumbnail = Image.fromarray(np.stack(channels, axis=-1)).resize((384, 384))
        canvas.paste(thumbnail, (index*384, 42))
        draw.text((index*384+4, 4), row["image_id"], fill="black")
        draw.text((index*384+4, 20), "R: DNA; G: epithelium; B: CD45", fill="black")
        preview_records.append(row["image_id"])
        del native
    canvas.save(CACHE / "representative_overviews.png")
    write_json(CACHE / "pixel_checks.json", dict(preparer=PREPARER, independent_review=False,
                                                 header_images=len(images), comparisons=comparisons,
                                                 overview_images=preview_records,
                                                 overview_source="Actual saved input.tiff planes on the square canvas, including padding",
                                                 overview_judgment="Pending visual inspection of representative_overviews.png"))
    reviews("preparer_verified")
    progress("prepared_for_independent_review", gaps=["Two image-to-specimen associations unresolved", "Independent review pending"],
             self_checked_headers=len(images), representative_pixel_checks=len(comparisons))
    print(json.dumps(dict(stage="preparer_verified", images=len(images), sample_linked_images=sum(bool(x.get("sample_id")) for x in images),
                          samples=len(samples), pixel_examples=len(comparisons))))
