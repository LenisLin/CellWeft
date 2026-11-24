"""Deliver native HTAN Duke MIBI inputs and evidence-based specimen metadata.

Inputs: 79 publication-linked and 46 accepted extension OME-TIFFs, their existing
native verification, official HTAN portal snapshot, author Table S1, and project S1.
Outputs: per-image input.tiff and complete study input tables on NAS.
Run export through the batch shared formal write lease wrapper; verify uses its read slot.
The repair-field-mapping stage only updates field rules after independent review.
The older lowercase entry runs later preparation stages and cannot be imported
without executing them. This task-specific entry reuses its OME/SourceSpec pattern
and the current shared input functions, stopping at native-intensity input delivery.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
import json
import os
import re
import shutil
import time
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

STUDY = "Risom2022_MIBI_BreastCancer"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--stage", choices=["prepare", "export", "verify", "repair-field-mapping"], default="export")
parser.add_argument("--image-id", action="append")
args = parser.parse_args()
root = args.data_root
output = root / "processed" / STUDY
cache = PROJECT / "cache/study_input" / STUDY
metadata = metadata_source_directory(root, STUDY)
job = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / (
    "0133_repair_1" if args.stage == "repair-field-mapping" else "0006_prepare_0")
cache.mkdir(parents=True, exist_ok=True)
job.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, root))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(stage: str, count: int, gaps: list[str], **details: object) -> None:
    write_json(job / "progress.json", dict(study=STUDY, stage=stage,
               completed_images=count, expected_images=125, gaps=gaps,
               updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details))


def template(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def known(value: object) -> str | None:
    """Treat the actual source missing-value spellings as missing, preserving zero."""
    if value is None or str(value).strip().lower() in {"", "na", "nc", "unknown", "not applicable", "not recorded"}:
        return None
    return str(value)


def mapping(source: Path, table: str, fields: str, target: str, targets: str,
            keys: str, rule: str, notes: str = "", values: dict | None = None) -> dict:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table=table,
                source_field=fields, target_table=target, target_field=targets,
                join_keys=keys, value_mapping=json.dumps(values) if values else None,
                transformation_rule=rule, time_origin_rule=None,
                conflict_rule="Preserve source namespaces; unresolved optional semantics remain blank.", notes=notes)


# 1. Bind all eligible source files to their existing native proofs and HTAN IDs.
portal_path = root / "raw/public" / STUDY / "manifest.json"
portal = json.loads(portal_path.read_text())["preparation_sources"]["portal"]
author_path = root / "raw/public" / STUDY / "metadata/Table_S1_Patient_Feature_Table.csv"
author = pd.read_csv(author_path, dtype=str, keep_default_na=False)
author = author.set_index("PointNumber", verify_integrity=True)
old_proof_path = portal_path
old = {r["filename"]: r for r in json.loads(old_proof_path.read_text())["preparation_sources"]["original_native"]["verified"]}
extension_path = metadata / "htan_duke_extension_ordered_markers_all46.json"
extension = {Path(r["source_filename"]).name: r for r in json.loads(extension_path.read_text())["files"]}
acceptance = json.loads((metadata / "htan_original_cohort_extension.json").read_text())
new_proofs = {name: (proof, portal_path) for name, proof in
              json.loads(portal_path.read_text())["preparation_sources"]["extension_native"].items()}
assert len(old) == 79 and len(extension) == 46 and len(new_proofs) == 46
assert not (set(old) & set(extension))
source_dirs = [root / "raw/public" / STUDY / name for name in
               ["htan_duke_release_2_4", "htan_duke_release_2_4_extension"]]
paths = {p.name: p for directory in source_dirs for p in directory.glob("*.ome.tif")}
assert set(paths) == set(old) | set(extension)
portal_files = {Path(r["Filename"]).name: r for r in portal["imaging"]}
assert set(paths) == set(portal_files)

units, panels, panel_ids = [], [], {}
for name, path in sorted(paths.items()):
    record = portal_files[name]
    point = re.fullmatch(r"Point(\d+)_.+\.ome\.tif", name)[1]
    if name in old:
        proof, proof_path = old[name], old_proof_path
        expected_names, expected_bytes = proof["marker_names"], proof["size"]
    else:
        proof, proof_path = new_proofs[name]
        expected_names = proof["ordered_markers"]
        expected_bytes = proof["source_stat"]["size"]
        assert proof["payload_bounds_complete"] and proof["all_required_proteins_decoded"]
        assert path.stat().st_mtime_ns == proof["source_stat"]["mtime_ns"]
        assert record["HTANDataFileID"] == extension[name]["htan_data_file_id"]
    assert path.stat().st_size == expected_bytes
    with tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        names = [c.get("Name") for c in pixels.findall("{*}Channel")]
        assert names == expected_names and len(tif.pages) == len(names)
        h, w = tif.pages[0].shape
        assert tif.series[0].axes == "CYX" and tif.pages[0].dtype == np.dtype("uint8")
        assert int(pixels.get("SizeZ")) == int(pixels.get("SizeT")) == 1
        if pixels.get("PhysicalSizeX") is not None and pixels.get("PhysicalSizeY") is not None:
            scale = [float(pixels.get(f"PhysicalSize{axis}")) for axis in "XY"]
            scale_evidence = "native OME PhysicalSizeX/Y"
        else:
            # Two actual source OME headers omit scale; their exact HTAN records supply it.
            assert record["PhysicalSizeXUnit"] == record["PhysicalSizeYUnit"] == "<U+00B5>m"
            scale = [float(record[f"PhysicalSize{axis}"]) for axis in "XY"]
            scale_evidence = "exact HTAN imaging record PhysicalSizeX/Y (native OME omits scale)"
        dtype = str(tif.pages[0].dtype)
    key = tuple(names)
    if key not in panel_ids:
        panel_id = f"htan_duke_mibi_{len(names)}ch"
        assert panel_id not in panel_ids.values()
        panel_ids[key] = panel_id
        panels.append(panel_from_s1(STUDY, panel_id, names,
                      root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"))
    sample_id, participant = record["HTANParentBiospecimenID"], record["HTANParticipantID"]
    biospecimen = portal["biospecimenByBiospecimenID"][sample_id]
    assert biospecimen["HTANParentID"] == participant
    clinical = author.loc[point].to_dict() if point in author.index else {}
    assert bool(clinical) == (name in old)
    unit = dict(image_id=name.removesuffix(".ome.tif"), filename=name, source_path=relative(path),
                point=point, sample_id=sample_id, participant=participant, panel_id=panel_ids[key],
                h=h, w=w, dtype=dtype, scale=scale, scale_evidence=scale_evidence, names=names,
                revision=record["releaseVersion"] + "; " + record["HTANDataFileID"],
                proof=relative(proof_path), portal=record, biospecimen=biospecimen, clinical=clinical)
    units.append(unit)
panel = pd.concat(panels, ignore_index=True)
assert len({u["sample_id"] for u in units}) == 125

# 2. Build biological specimen rows using explicit HTAN parents, then author Point map.
samples, links = [], []
for u in units:
    b, c = u["biospecimen"], u["clinical"]
    diagnosis = portal["diagnosisByParticipantID"].get(u["participant"], {})
    demographic = portal["demographicsByParticipantID"].get(u["participant"], {})
    # Author tissue annotation is specific to the imaged specimen. Two normal
    # specimens have generic HTAN Primary labels; retain both source values.
    normal = c["Tissue_Type"] == "normal" if c else b["TumorTissueType"] == "Normal"
    sample = dict.fromkeys(template("sample_metadata"))
    sample.update(study=STUDY, sample_id=u["sample_id"], patient_key=u["participant"],
                  source_subject_id=u["participant"], source_sample_id=u["sample_id"],
                  cohort_id="RAHBT" if b["ProtocolLink"] == "RAHBT" else None, center_id="HTA6",
                  sample_type="Tissue Biospecimen Type", tissue_type="normal" if normal else "tumor",
                  anatomical_site="Breast", primary_site=known(diagnosis.get("TissueorOrganofOrigin")),
                  lesion_type="non_malignant" if normal else {"Primary": "primary", "Additional Primary": "primary", "Recurrent": "recurrence"}[b["TumorTissueType"]],
                  timepoint_label="baseline" if b["TimepointLabel"] == "Baseline" else
                      "recurrence" if b["TimepointLabel"] == "Event" and b["TumorTissueType"] == "Recurrent" else
                      "other" if b["TimepointLabel"] == "Event" else None,
                  collection_method="resection", preservation_method="ffpe",
                  processing_method="Formalin; paraffin block",
                  sex_reported=known(demographic.get("Gender", "")) and str(demographic["Gender"]).lower(),
                  vital_status={"Alive": "alive", "Dead": "deceased", "Unknown": "unknown"}.get(demographic.get("VitalStatus")),
                  primary_diagnosis=known(diagnosis.get("PrimaryDiagnosis")),
                  histology=known(c.get("Tissue_Type")),
                  grade=known(diagnosis.get("TumorGrade")),
                  molecular_subtype=known(c.get("PAM50")))
    sample["metadata_notes"] = json.dumps(dict(
        evidence=relative(portal_path), HTAN_tissue_type=b["TumorTissueType"],
        HTAN_timepoint=b["TimepointLabel"],
        HTAN_collection_days_from_index=b.get("CollectionDaysfromIndex"),
        HTAN_AgeatDiagnosis_raw=diagnosis.get("AgeatDiagnosis"),
        tissue_conflict="Author specimen-level normal annotation takes precedence over generic HTAN Primary; original labels retained." if normal and b["TumorTissueType"] != "Normal" else None,
        time_semantics="CollectionDaysfromIndex resembles age in days; index origin not established, so no collection offset or survival endpoint is derived.",
        author_clinical={k: v for k, v in c.items() if k in author.columns[:35] and known(v) is not None},
        clinical_coverage="Author patient table joined by PointNumber" if c else
            "Extension not linked to the 79-resection author table; optional author clinical fields unavailable.",
        publication="10.1016/j.cell.2021.12.023" if c else "Additional acquisition publication attribution unknown"), ensure_ascii=False)
    samples.append(sample)
    links.append(dict(image_id=u["image_id"], source_filename=u["filename"],
                      HTANDataFileID=u["portal"]["HTANDataFileID"], sample_id=u["sample_id"],
                      patient_key=u["participant"], PointNumber=u["point"],
                      paper_patient_id=c.get("Patient_ID"), paper_TMAD_Patient=c.get("TMAD_Patient"),
                      paper_core_name=c.get("Full_Point_Name")))
samples = pd.DataFrame(samples, columns=template("sample_metadata"))
links = pd.DataFrame(links)
paper_links = links[links.paper_patient_id.notna()]
assert paper_links.groupby("paper_patient_id").patient_key.nunique().max() == 1
assert paper_links.groupby("patient_key").paper_patient_id.nunique().max() == 1
if args.stage != "repair-field-mapping":
    links.to_csv(cache / "image_sample_patient_links.csv", index=False, encoding="utf-8-sig")

field_rows = [
    mapping(portal_path, "imaging + biospecimenByBiospecimenID", "Filename;HTANDataFileID;HTANParentBiospecimenID;HTANParticipantID;HTANParentID",
            "image_metadata", "image_id;source_image_id;sample_id;split_evidence", "exact Filename basename -> HTANParentBiospecimenID -> HTANBiospecimenID; verify HTANParentID=HTANParticipantID",
            "One native independent acquisition per complete source TIFF; use the explicit parent specimen as sample, never infer a patient from a filename.",
            "125 images -> 125 biological biospecimen IDs -> 102 participant IDs; 79 author-linked images -> 70 participants; extension46 ->34 participants (2 shared with original)."),
    mapping(portal_path, "biospecimenByBiospecimenID + demographicsByParticipantID + diagnosisByParticipantID",
            "HTANBiospecimenID;HTANParentID;ProtocolLink;BiospecimenType;TumorTissueType;TimepointLabel;AcquisitionMethodType;PreservationMethod;FixativeType;StorageMethod;Gender;VitalStatus;PrimaryDiagnosis;TissueorOrganofOrigin;TumorGrade",
            "sample_metadata", "study;sample_id;patient_key;source_subject_id;source_sample_id;cohort_id;center_id;sample_type;tissue_type;anatomical_site;primary_site;lesion_type;timepoint_label;collection_method;preservation_method;processing_method;sex_reported;vital_status;primary_diagnosis;grade;metadata_notes",
            "explicit biospecimen and participant IDs",
            "Preserve IDs. Assign Breast from the confirmed source study. HTA6 is the center. ProtocolLink=RAHBT supplies cohort_id for118 specimens; seven external normal controls list only a paper DOI, so cohort_id stays empty. Excision->resection; FFPE->ffpe; gender lowercase; Alive/Dead->alive/deceased. TumorTissueType Normal->normal/non_malignant, Recurrent->tumor/recurrence, Primary or Additional Primary->tumor/primary. Baseline->baseline; Event+Recurrent->recurrence; other Event->other; not-applicable timepoint stays empty.",
            "All125 have explicit specimen and participant parents. Diagnosis/demographics cover118/125; seven normal external controls have no such source records. Unknown/NA/NC values stay empty. For Point2305 and Point2324, the author specimen-level normal annotation overrides generic HTAN Primary for tissue_type/lesion_type; both originals remain in notes. Total9 normal specimens."),
    mapping(author_path, "Table_S1_Patient_Feature_Table (Mendeley d87vg86zd8 v3)", "PointNumber;Patient_ID;TMAD_Patient;Full_Point_Name;TMAD_ID;Row;Column;Tissue_Type;PAM50",
            "sample_metadata", "histology;molecular_subtype;metadata_notes", "numeric Point acquisition identifier from exact published filename = PointNumber; compare patient grouping with explicit HTAN links",
            "Author Tissue_Type->histology and PAM50->molecular_subtype; preserve author patient/core IDs and raw optional clinical fields in notes and linkage evidence. No author ID replaces an HTAN patient ID.",
            "79/79 publication-linked acquisitions join uniquely;70 author patient groups match70 HTAN groups bijectively. Extension46 has no rows. TMAD_Patient denotes a core label, not patient identity."),
    mapping(author_path, "Table_S1_Patient_Feature_Table", "TMAD_ID;Row;Column;Full_Point_Name;Tissue_Type;Status",
            "image_metadata", "slide_id;core_id;region_annotation", "PointNumber derived solely as acquisition identity from exact published filename",
            "For the79 source-paper rows, core_id=TMAD_ID+row+column; no physical slide/section ID is supplied, so slide_id stays empty. region_annotation retains Full_Point_Name, Tissue_Type and Status as a JSON object. Extension region_annotation retains explicit HTAN TumorTissueType/TimepointLabel. Entire source FOV retained."),
    mapping(metadata / "ordered_markers.json", "native OME Pixels and accepted native verification", "Channel Name;SizeX;SizeY;Type;PhysicalSizeX;PhysicalSizeY",
            "image_metadata", "source_axes;source_dtype;source_revision;source_channel_count;pixel_size_x_um;pixel_size_y_um;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;input_image_path;intensity_description;image_kind;split_method",
            "exact source filename; embedded OME channel order",
            "Use actual native CYX uint8 TIFF headers and existing proofs, not stale portal SizeC=55/PixelType=int8. Preserve all pixels in full FOV; centered zero padding only if needed; no resampling. All files1024x1024 at0.488um/pixel.",
            "Point3105_31661 and Point4142_pt2008_Normal omit scale in OME; their exact HTAN imaging records explicitly supply0.488um/pixel. Other scales use native OME. HTAN Level2 published uint8 intensities are the native local source; original detector counts and upstream scaling are not reconstructed. Extension uses htan_duke_extension_ordered_markers_all46.json and its46 existing native proofs."),
    mapping(root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv", "S1", "Study;Original marker;Unified marker;Segmentation channel",
            "panel_mapping", "source_channel_index;source_name;marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index;keep_reason;exclusion_reason", "Study + exact native source label",
            "Use shared panel_from_s1; retain all admitted markers and Nuc nuclear input in source order. Source index and output index remain distinct; no study alias or duplicate measurement selection required."),
    mapping(portal_path, "biospecimen + diagnosis + demographics", "CollectionDaysfromIndex;AgeatDiagnosis;DaystoLastFollowup;TimepointLabel",
            "sample_metadata", "collection_day_from_index;age_at_collection_years;age_at_diagnosis_years;last_followup_time;os_time;pfs_time;dfs_time",
            "explicit HTAN specimen/participant IDs",
            "Leave derived ages, collection offsets and endpoint times empty where source time origin or units are not independently established. Preserve raw source values in metadata_notes; no survival endpoint derivation.",
            "Public paper was fetched as EuropePMC fullTextXML; Mendeley v3 author table and author MIBIAnalysis code archive were retrieved. Code archive contains general MIBI routines and supplies no additional specimen/participant map. Cached source locations are listed in downloaded_metadata.json."),
    mapping(metadata / "htan_original_cohort_extension.json", "accepted old_scope + new_scope", "dataset_name;canonical_id;old_scope;new_scope",
            "image_metadata", "study", "exact source filename in the accepted 79-file or 46-file scope",
            "Assign the task study constant Risom2022_MIBI_BreastCancer to every accepted source acquisition.",
            "The canonical cohort includes 79 publication-linked and 46 accepted extension acquisitions; this study label does not assign a paper DOI to the extension."),
    mapping(portal_path, "imaging + complete local source inventory", "Filename;HTANDataFileID",
            "image_metadata", "source_files", "exact Filename basename = accepted native-proof source filename = local source basename",
            "Resolve the complete local source in raw/public/Risom2022_MIBI_BreastCancer/htan_duke_release_2_4 or htan_duke_release_2_4_extension; store its path relative to the NAS root as a one-element string list in Parquet.",
            "Only complete accepted .ome.tif files are selected. Source identity and size are checked against the existing native proof; extension mtime_ns is also checked. No new payload hash is required."),
    mapping(portal_path, "imaging + native TIFF header", "Filename;FileFormat;OME Pixels",
            "image_metadata", "source_format", "exact source filename and its native TIFF header",
            "Set ome_tiff after opening the accepted .ome.tif with tifffile and parsing its embedded OME Pixels/Channel metadata; source axes and decoded dtype remain separate fields."),
    mapping(portal_path, "imaging", "Filename;FOVnumber",
            "image_metadata", "fov_id", "exact accepted source Filename basename",
            "Match the acquisition prefix with Point(\\d+)_.+\\.ome\\.tif and preserve Point plus the captured digits as fov_id. Cross-check each nonempty HTAN FOVnumber against those acquisition digits.",
            "All 125 filenames explicitly label a Point acquisition; 61 HTAN FOVnumber values agree and 64 are empty. This extracts acquisition identity only; sample and patient use explicit HTAN parent IDs."),
    mapping(metadata / "ordered_markers.json", "native OME channel order + existing native proofs", "ordered Channel Name sequence;source filename",
            "image_metadata", "panel_id", "exact source filename -> complete ordered native channel-name tuple -> panel_mapping.panel_id",
            "Group identical complete ordered native channel tuples and assign htan_duke_mibi_{source_channel_count}ch. Assert that different tuples do not reuse a panel_id. This scope has distinct 43-channel and 45-channel source panels.",
            "The extension's ordered labels are also recorded in raw/public/Risom2022_MIBI_BreastCancer/metadata.json#public/Risom2022_MIBI_BreastCancer/htan_duke_extension_ordered_markers_all46.json and its 46 native proofs. Source order, rather than the stale portal SizeC, defines the panel."),
    mapping(root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv", "S1 + derived panel_mapping + output TIFF header", "Study;Original marker;Unified marker;Segmentation channel;keep_channel;output_channel_index",
            "image_metadata", "channel_count", "image.panel_id = panel_mapping.panel_id; exact study and native source label for S1",
            "Count keep_channel=true rows in the actual panel, including selected nuclear inputs; require output_channel_index=0..K-1 and K equal the input.tiff channel axis and page count. Store K as channel_count.",
            "The 43-source-channel panel retains 37 channels; the 45-source-channel panel retains 39. source_channel_count and source_channel_index continue to describe the complete original panel."),
    mapping(portal_path, "imaging + explicit biospecimen parents + native scale provenance", "Filename;HTANParentBiospecimenID;HTANParticipantID;PhysicalSizeX;PhysicalSizeY",
            "image_metadata", "metadata_notes", "exact source filename -> explicit HTAN biospecimen and participant; filename PointNumber -> author table only for the 79 linked acquisitions",
            "Record the explicit source-file/specimen/participant chain and full native FOV. Append the actual scale source: native OME PhysicalSizeX/Y, or the exact HTAN imaging record when OME omits scale. Append publication-linked/author-Point-table status for the 79 original files, or accepted-extension/unknown per-file paper attribution for the 46 extension files.",
            "Native proofs and the accepted old/new scope identify the acquisition. No patient is inferred from Point labels, and extension notes do not borrow original-cohort clinical values."),
]
fields = pd.DataFrame(field_rows, columns=template("field_mapping"))
images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists() else pd.DataFrame(columns=template("image_metadata"))
images = images.reindex(columns=template("image_metadata"))
stats_path = cache / "export_stats.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
gaps = ["46个扩展采集的具体论文归属未知；未套用原79个样本的作者临床值。",
        "7个正常对照缺HTAN diagnosis/demographics；连接身份明确，临床可选字段留空。",
        "来源时间原点/部分临床字段语义未确认，相关派生字段留空。",
        "待主控组织不同CLI上下文独立复核。"]


# Apply the first review's field-rule repair without republishing accepted data.
if args.stage == "repair-field-mapping":
    from cellweft.data.io.publication import resource_lock

    progress("repairing_field_mapping", len(images), gaps, repair_round=1)
    with nullcontext():
        previous_fields = pd.read_csv(output / "field_mapping.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(fields.iloc[:len(previous_fields)].fillna("").reset_index(drop=True),
                                      previous_fields.reset_index(drop=True))
        coverage = {}
        for table, frame in [("image_metadata", images), ("sample_metadata", samples)]:
            mapped = {key for targets in fields.loc[fields.target_table == table, "target_field"]
                      for key in targets.split(";")}
            populated = {key for key in frame if frame[key].notna().any()}
            coverage[table] = sorted(populated - mapped)
            assert not coverage[table], f"Unmapped populated fields: {table}: {coverage[table]}"
        temporary = output / "field_mapping.csv.tmp"
        with formal_publish_lease(output / "field_mapping.csv", directory=output):
            fields.to_csv(temporary, index=False, encoding="utf-8-sig")
            pd.testing.assert_frame_equal(pd.read_csv(temporary, dtype=str, keep_default_na=False), fields.fillna(""))
            os.replace(temporary, output / "field_mapping.csv")
    repair = dict(study=STUDY, role="preparer_repair_self_check", repair_round=1,
                  expected_images=125, exported_images=len(images), sample_rows=len(samples),
                  sample_linked_images=int(images.sample_id.notna().sum()),
                  patient_linked_samples=int(samples.patient_key.notna().sum()),
                  distinct_patients=int(samples.patient_key.nunique()),
                  added_fields=fields.iloc[len(previous_fields):].target_field.tolist(),
                  unmapped_populated_fields=coverage, field_mapping_rows=len(fields),
                  changed_outputs=[relative(output / "field_mapping.csv")],
                  independent_review="M02/D01 recheck and acceptance binding pending; existing review_results preserved")
    write_json(job / "repair_self_check.json", repair)
    progress("field_mapping_repaired_pending_independent_review", len(images), gaps, repair_round=1,
             sample_rows=len(samples), sample_linked_images=repair["sample_linked_images"],
             patient_linked_samples=repair["patient_linked_samples"], distinct_patients=repair["distinct_patients"],
             repaired_fields=repair["added_fields"])
    print(json.dumps(repair, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)


def publish() -> None:
    """Checkpoint only actual completed images while preserving the full specimen table."""
    save_input_tables(output, images, samples, panel, fields)
    progress("exporting" if args.stage == "export" else args.stage, len(images), gaps,
             sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()))


# 3. Stream source-native tiles to independent TIFFs and checkpoint every image.
if args.stage == "export":
    for u in units:
        image_id = u["image_id"]
        if args.image_id and image_id not in args.image_id:
            continue
        current_panel = panel[panel.panel_id == u["panel_id"]]
        destination = output / "images" / image_id / "input.tiff"
        expected_names = current_panel.loc[current_panel.keep_channel, "marker_id"].tolist()
        if image_id in images.image_id.values and destination.exists():
            previous = images.set_index("image_id").loc[image_id]
            assert list(previous.source_files) == [u["source_path"]]
            with tifffile.TiffFile(destination) as tif:
                actual_names = [c.get("Name") for c in ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
                assert actual_names == expected_names and tif.series[0].shape == (len(expected_names), u["h"], u["w"])
            continue
        if shutil.disk_usage(root).free < 100 * 1024**3 + len(expected_names) * max(u["h"], u["w"])**2:
            raise RuntimeError("Insufficient NAS space including100GiB reserve")
        source = SourceSpec(image_id, u["panel_id"], "tiff", root / u["source_path"],
                            tuple(range(len(u["names"]))), pixel_size_um=tuple(u["scale"]))
        progress("exporting", len(images), gaps, current_image=image_id)
        with open_source(source) as reader:
            result = export_input(reader, current_panel, destination)
        stats[image_id] = result
        write_json(stats_path, stats)
        if not result["admitted"]:
            print(json.dumps(dict(image_id=image_id, excluded="all retained downstream channels are zero")), flush=True)
            continue
        c, b = u["clinical"], u["biospecimen"]
        row = dict.fromkeys(template("image_metadata"))
        row.update(study=STUDY, image_id=image_id, sample_id=u["sample_id"], panel_id=u["panel_id"],
                   image_kind="fov", source_image_id=u["portal"]["HTANDataFileID"],
                   source_files=[u["source_path"]], source_format="ome_tiff", source_axes="CYX",
                   source_dtype=u["dtype"], source_revision=u["revision"],
                   slide_id=None, core_id=f'{c["TMAD_ID"]}_R{c["Row"]}C{c["Column"]}' if c else None,
                   fov_id="Point" + u["point"], region_annotation=json.dumps(
                       {k: c[k] for k in ["Full_Point_Name", "Tissue_Type", "Status"]} if c else
                       {"TumorTissueType": b["TumorTissueType"], "TimepointLabel": b["TimepointLabel"]}),
                   source_x0_px=0, source_y0_px=0, source_x1_px=u["w"], source_y1_px=u["h"],
                   pixel_size_x_um=u["scale"][0], pixel_size_y_um=u["scale"][1],
                   input_image_path=relative(destination), valid_pixels_path=None,
                   intensity_description="Published HTAN MIBI Level2 uint8 intensities; preserve exact decoded values; upstream scaling not reconstructed.",
                   split_method="already_independent", split_evidence=relative(portal_path) + "; " + u["proof"],
                   metadata_notes="Explicit HTAN source file -> biological biospecimen -> participant; entire native FOV. Scale: " + u["scale_evidence"] + ". " +
                       ("Publication-linked source; author Point table joined." if c else "Accepted original-cohort extension; per-file paper attribution unknown."))
        for key in ["height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px", "pad_bottom_px", "channel_count", "source_channel_count"]:
            row[key] = result[key]
        images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
        images = images.sort_values("image_id").reset_index(drop=True)
        publish()
        print(json.dumps(dict(completed_images=len(images), image_id=image_id, shape=[row["channel_count"], row["height_px"], row["width_px"]])), flush=True)

publish()

# 4. Self-check every TIFF header; representative source/output pixels cover panels.
if args.stage == "verify":
    verification = dict(headers=[], pixel_checks=[], overviews=[])
    assert images.shape == (125, 39) and samples.shape == (125, 71)
    assert set(images.sample_id) == set(samples.sample_id)
    assert samples.patient_key.nunique() == 102
    assert samples.tissue_type.eq("normal").sum() == 9
    assert samples.cohort_id.eq("RAHBT").sum() == 118
    assert images.slide_id.isna().all()
    assert len(list((output / "images").glob("*/input.tiff"))) == 125
    assert not list((output / "images").glob("*/input.tiff.partial"))
    verification["metadata"] = dict(image_rows=125, image_fields=39, sample_rows=125,
        sample_fields=71, sample_linked_images=125, patient_linked_samples=125,
        distinct_patients=102, normal_samples=9, RAHBT_samples=118)
    by_id = {u["image_id"]: u for u in units}
    for row in images.itertuples(index=False):
        selected = panel[(panel.panel_id == row.panel_id) & panel.keep_channel].sort_values("output_channel_index")
        with tifffile.TiffFile(root / row.input_image_path) as tif:
            names = [c.get("Name") for c in ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
            assert names == selected.marker_id.tolist()
            assert tif.series[0].shape == (row.channel_count, row.height_px, row.width_px)
            assert tif.series[0].dtype == np.dtype(row.source_dtype)
            verification["headers"].append(row.image_id)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for panel_id, group in images.groupby("panel_id"):
        row = group.iloc[0]
        u = by_id[row.image_id]
        selected = panel[(panel.panel_id == panel_id) & panel.keep_channel].sort_values("output_channel_index")
        with tifffile.TiffFile(root / u["source_path"]) as source, tifffile.TiffFile(root / row.input_image_path) as target:
            for window in [(448, 448, 480, 480), (992, 992, 1024, 1024)]:
                source_nonzero = 0
                for channel in selected.itertuples(index=False):
                    before = read_tiff_window(source, int(channel.source_channel_index), window)
                    after = read_tiff_window(target, int(channel.output_channel_index), window)
                    assert before.dtype == after.dtype and np.array_equal(before, after)
                    source_nonzero += int(np.count_nonzero(before))
                assert source_nonzero > 0
                verification["pixel_checks"].append(dict(image_id=row.image_id, panel_id=panel_id,
                    window=list(window), channels=len(selected), equal=True,
                    source_nonzero_pixels=source_nonzero))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, label in zip(axes, ["Nuc", "PanKRT", "CD45"]):
                data = source.pages[u["names"].index(label)].asarray(maxworkers=1)
                ax.imshow(data, cmap="gray", vmin=0, vmax=max(1, float(np.percentile(data, 99.5))))
                ax.set_title(label); ax.axis("off")
            fig.suptitle(row.image_id + " | native full FOV; display only")
            fig.tight_layout()
            overview = cache / f"overview_{panel_id}.png"
            fig.savefig(overview, dpi=120); plt.close(fig)
            verification["overviews"].append(relative(overview))
    verification["role"] = "preparer self-check; independent review pending"
    write_json(cache / "self_check.json", verification)

# Review results explicitly identify preparer self-checks and leave independent acceptance pending.
review = []
checks = pd.read_csv(TEMPLATES / "review_checks.csv")
for check in checks.itertuples(index=False):
    detail = "整理者自检；独立复核待主控派发。"
    if check.check_id == "M01":
        detail += f" 已查本地HTAN官方snapshot、原79及扩展46核验、EuropePMC论文XML、Mendeley原作者特征表和代码归档；125个明确biospecimen/102位participant；已完成image连接{len(images)}/125，sample→patient125/125。7正常对照无诊断人口学资料，扩展46无原79临床行。"
    if check.check_id == "D01":
        detail += f" 已实际交付{len(images)}/125个input TIFF；{len(samples)}个sample，{samples.patient_key.nunique()}位participant；未进行独立验收。"
    review.append(dict(study=STUDY, preparer="codex_cli_0006_prepare_0", reviewer=None,
                       scope=check.scope, record_id=STUDY, check_id=check.check_id,
                       result="unresolved", source_evidence=";".join([relative(portal_path), relative(author_path), (cache / 'source_followup.json').as_posix()]),
                       output_evidence=(cache / 'self_check.json').as_posix() if (cache / "self_check.json").exists() else stats_path.as_posix(),
                       details=detail))
review_path = output / "review_results.csv"
with formal_publish_lease(review_path, directory=output):
    pd.DataFrame(review, columns=template("review_results")).to_csv(review_path, index=False, encoding="utf-8-sig")
summary = dict(study=STUDY, stage=args.stage, expected_images=125, exported_images=len(images),
               sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
               patient_linked_samples=int(samples.patient_key.notna().sum()), distinct_patients=int(samples.patient_key.nunique()),
               panels={pid: dict(source_channels=len(g), output_channels=int(g.keep_channel.sum())) for pid, g in panel.groupby("panel_id")},
               expected_uncompressed_output_bytes=sum(int(panel.loc[(panel.panel_id == u["panel_id"]) & panel.keep_channel].shape[0]) * max(u["h"], u["w"])**2 for u in units),
               source_bytes=sum(p.stat().st_size for p in paths.values()),
               output_tiff_bytes=sum(p.stat().st_size for p in (output / "images").glob("*/input.tiff")),
               excluded_images=[key for key, value in stats.items() if not value["admitted"]],
               evidence=[relative(portal_path), relative(author_path), (cache / 'image_sample_patient_links.csv').as_posix(), relative(old_proof_path), relative(metadata / "htan_original_cohort_extension.json")],
               issues=gaps, controller_requests=["请派发不同CLI上下文独立复核；本review_results仅标记整理者自检及待验收。",
                    "TASK指定的大写study脚本为本阶段入口；旧小写脚本属于后处理入口，后续请统一两入口的文档和阶段职责。"])
write_json(cache / "delivery_summary.json", summary)
progress("self_checked_pending_independent_review" if args.stage == "verify" else "export_complete" if len(images) == 125 else args.stage,
         len(images), gaps, sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
         patient_linked_samples=int(samples.patient_key.notna().sum()), distinct_patients=int(samples.patient_key.nunique()))
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
