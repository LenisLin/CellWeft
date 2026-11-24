"""Prepare native HTAN Release 2.0 MIBI inputs and evidence-based sample links.

Inputs: 58 localized OME-TIFFs, their accepted native verification, study S1,
HTAN MSK metadata, and the author's public MIBI clinical tables in the received source namespace.
Outputs: resumable per-FOV input.tiff, five input tables, and preparer checks.
Run NAS-heavy stages through the batch shared formal I/O coordination wrapper; no normalization.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.inputs import file_revision, input_binding
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)

STUDY = "Chan2021_MIBI_SCLC"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
OUTPUT = NAS / "processed" / STUDY
META = metadata_source_directory(NAS, STUDY)
SOURCE = META / "preparation_sources"
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0021_prepare_0/progress.json"
PROOF = META / "native_verification.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "htan_msk_mibi_29"
PREPARER = "Codex CLI Chan2021_MIBI_SCLC 0021_prepare_0"
OME = {"o": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def write_json(path: Path, value: object) -> None:
    """Publish a resumable study receipt without leaving a truncated JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage, expected_images=58,
                             completed_images=completed, updated_unix=time.time(), **details))


def read_table(name: str) -> pd.DataFrame:
    """Read the cached official HTAN row arrays with their explicit column order."""
    table = json.loads((SOURCE / f"htan_{name}.json").read_text())
    # Relevant biospecimen/clinical tables have matching row lengths. The source
    # FollowUp and unrelated ImagingLevel2 rows have known legacy schema drift.
    return pd.DataFrame([dict(zip(table["column_order"], row["values"], strict=True))
                         for row in table["record_list"]])


def native_header(path: Path) -> dict:
    """Read the actual full-resolution OME series; ignore reduced-resolution levels."""
    with tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find("o:Image/o:Pixels", OME)
        return dict(axes=tif.series[0].axes, shape=list(tif.series[0].shape),
                    dtype=str(tif.series[0].dtype),
                    names=[c.attrib["Name"] for c in pixels.findall("o:Channel", OME)],
                    pixel_size_x_um=float(pixels.attrib["PhysicalSizeX"]),
                    pixel_size_y_um=float(pixels.attrib["PhysicalSizeY"]))


def mapping(source: str, fields: str, target: str, *, table: str = "sample_metadata",
            transform: str = "Copy source value", keys: str = "", values: dict | None = None,
            notes: str = "", origin: str = "") -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=fields, target_table=table, target_field=target,
                join_keys=keys, value_mapping=json.dumps(values or {}, ensure_ascii=False),
                transformation_rule=transform, time_origin_rule=origin,
                conflict_rule="Use the specified source and preserve other source values in cache/notes",
                notes=notes)


def image_sample_mapping() -> dict:
    """Name the joined table containing every declared image/sample identity field."""
    return mapping(str(metadata_source_relative(SOURCE / 'htan_image_selected.csv', NAS)),
        "synapseId; HTANParentBiospecimenID; HTANParticipantID; PublicationID",
        "sample_id", table="image_metadata",
        keys="image_manifest.source_id=htan_image_selected.synapseId; "
             "htan_file_records.synapseId=htan_ImagingLevel2.entityId; "
             "HTANParentBiospecimenID=HTAN Biospecimen ID; "
             "HTAN Parent ID=HTANParticipantID; PublicationID=MIBI ID=MIBI name",
        transform="htan_image_selected.csv joins htan_file_records.json to "
                  "htan_ImagingLevel2.json on synapseId=entityId (one-to-one); "
                  "PublicationID comes from ImagingLevel2. Copy the explicit parent biospecimen ID.",
        notes=f"58/58 FOVs -> 37 biospecimens -> 37 participants. "
              "Sources: HTAN file records; "
              f"raw/public/{STUDY}/metadata.json#public/{STUDY}/preparation_sources/htan_ImagingLevel2.json. "
              "See recorded sample-linkage evidence.")


def metadata_notes_mapping() -> dict:
    """Separate source-backed note context from the preparer's missingness summary."""
    return mapping(str(metadata_source_relative(SOURCE / 'sample_linkage_evidence.json', NAS)),
        "source_urls; clinical_snapshot_conflicts; supplement_attempts", "metadata_notes",
        transform="Assemble source-linked notes using the preceding field mappings. "
                  "sample_tables generates optional_gaps as a preparer summary of "
                  "unconfirmed physical-core, clinical-cutoff and collection-date metadata "
                  "after the documented source search.",
        notes="optional_gaps is generated in scripts/data/preparation/Chan2021_MIBI_SCLC.py::sample_tables. "
              "Source-search provenance is retained with study preparation evidence; "
              "source_urls and supplement_attempts record the HTAN snapshot, paper/Table S14 "
              "attachment attempt and author clinical/sample tables consulted. "
              "All 71 sample template columns retained; unknown optional clinical, chronology "
              "and physical-core fields remain blank.")


def repair_provenance(images: pd.DataFrame) -> None:
    """Repair M02/C09 only; retain verified pixels and await independent rebinding."""
    manifest = pd.read_csv(META / "image_manifest.csv", dtype=str, keep_default_na=False)
    proofs = {row["source_id"]: row for row in json.loads(PROOF.read_text())}
    raw = NAS / "raw/public" / STUDY / "htan_release_2_0"
    assert {str(metadata_source_relative(path, NAS)) for path in raw.glob("*.ome.tiff")} == set(manifest.image_path)
    for source in manifest.itertuples(index=False):
        revision = (NAS / source.image_path).stat()
        assert revision.st_size == int(source.size) == proofs[source.source_id]["size"]
        assert revision.st_mtime_ns == proofs[source.source_id]["mtime_ns"]

    # Rebuild the precise source-ID join from cached official metadata, reusing
    # the prior native/pixel review rather than reopening all TIFF headers.
    official = json.loads((SOURCE / "htan_ImagingLevel2.json").read_text())
    entity_column = official["column_order"].index("entityId")
    source_ids = set(manifest.source_id)
    native = pd.DataFrame([dict(zip(official["column_order"], row["values"], strict=True))
        for row in official["record_list"] if row["values"][entity_column] in source_ids])
    files = pd.DataFrame(json.loads((SOURCE / "htan_file_records.json").read_text()))
    joined = files.merge(native[["entityId", "PublicationID"]],
                         left_on="synapseId", right_on="entityId", validate="one_to_one")
    selected = pd.read_csv(SOURCE / "htan_image_selected.csv", dtype=str, keep_default_na=False)
    identity_fields = ["synapseId", "HTANParentBiospecimenID", "HTANParticipantID", "PublicationID"]
    assert set(joined.synapseId) == set(selected.synapseId) == source_ids
    pd.testing.assert_frame_equal(
        joined[identity_fields].sort_values("synapseId").reset_index(drop=True),
        selected[identity_fields].sort_values("synapseId").reset_index(drop=True))
    assert len(images) == len(selected) == 58
    sources = selected.set_index("synapseId", verify_integrity=True)
    assert images.source_image_id.map(sources.HTANParentBiospecimenID).eq(images.sample_id).all()
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    by_sample = selected.drop_duplicates("HTANParentBiospecimenID").set_index("HTANParentBiospecimenID")
    assert len(samples) == len(by_sample) == 37
    assert samples.sample_id.map(by_sample.HTANParticipantID).eq(samples.patient_key).all()
    assert samples.sample_id.map(by_sample.PublicationID).eq(
        samples.metadata_notes.map(lambda value: json.loads(value)["author_mibi_id"])).all()
    biospecimens = read_table("Biospecimen").set_index("HTAN Biospecimen ID", verify_integrity=True)
    assert samples.sample_id.map(biospecimens["HTAN Parent ID"]).eq(samples.patient_key).all()
    for filename, key, separator in [("author_survival_metadata.tsv", "MIBI ID", "\t"),
                                       ("author_mibi_sample_table.csv", "MIBI name", ",")]:
        author = pd.read_csv(SOURCE / filename, sep=separator, dtype=str, keep_default_na=False)
        assert by_sample.PublicationID.isin(author[key]).all()

    progress("publishing_provenance_repair", len(images), gaps=["independent M02 recheck and D01 rebinding pending"])
    with nullcontext():
        fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
        assert len(fields) == 13 and fields.iloc[12].target_table == "sample_metadata"
        assert fields.iloc[12].target_field == "metadata_notes"
        before_fields = fields.copy()
        before = fields.iloc[12].to_dict()
        linkage = json.loads((SOURCE / "sample_linkage_evidence.json").read_text())
        correction = metadata_notes_mapping()
        declared_keys = [key.strip() for key in correction["source_field"].split(";")]
        assert set(declared_keys).issubset(linkage)
        assert samples.metadata_notes.map(lambda value: json.loads(value)["optional_gaps"]).eq(
            "physical TMA core map; clinical cutoff date; absolute collection date").all()
        protected_tables = {name: file_revision(OUTPUT / name)
                            for name in ["images.parquet", "sample_metadata.csv", "panel_mapping.csv"]}
        binding_before = input_binding(OUTPUT, NAS, images.image_id.tolist())
        fields.iloc[12] = pd.Series(correction)
        pd.testing.assert_frame_equal(before_fields.iloc[:12], fields.iloc[:12])
        changed_columns = [key for key in fields.columns if before[key] != correction[key]]
        assert set(changed_columns).issubset({"source_field", "transformation_rule", "notes"})
        temporary = OUTPUT / "field_mapping.csv.partial"
        with formal_publish_lease(OUTPUT / "field_mapping.csv", directory=OUTPUT):
            fields.to_csv(temporary, index=False, encoding="utf-8-sig")
            temporary.replace(OUTPUT / "field_mapping.csv")
        binding_after = input_binding(OUTPUT, NAS, images.image_id.tolist())
        assert {key: value for key, value in binding_before.items() if key != "field_mapping"} == {
            key: value for key, value in binding_after.items() if key != "field_mapping"}
        assert all(file_revision(OUTPUT / name) == revision for name, revision in protected_tables.items())
        evidence = dict(study=STUDY, repair_round=2, preparer_self_check=True,
            check_id="M02", rule_id="C09", field_mapping_data_row=13,
            before=before, after=fields.iloc[12].to_dict(), changed_columns=changed_columns,
            source_keys_checked=declared_keys, missing_declared_source_keys=[],
            generated_optional_gaps_sample_rows=len(samples),
            unaffected_field_mapping_rows_unchanged=True, protected_tables_unchanged=True,
            source_and_output_revisions_reused=True, protected_table_revisions=protected_tables,
            input_binding_changed_components=[key for key in binding_before
                                              if binding_before[key] != binding_after[key]],
            expected_images=58, exported_images=len(images), reused_tiffs=len(images),
            sample_rows=len(samples), sample_linked_images=int(images.sample_id.isin(samples.sample_id).sum()),
            patient_linked_samples=int(samples.patient_key.ne("").sum()),
            identity_join_reconstructed=True, native_verification_reused=str(metadata_source_relative(PROOF, NAS)),
            next_step="Independent reviewer must recheck M02/C09 and refresh D01 input_binding; verified TIFFs remain reusable")
        evidence_path = PROGRESS.parent / "repair_validation.json"
        write_json(evidence_path, evidence)
        reviews = pd.read_csv(OUTPUT / "review_results.csv", dtype=str, keep_default_na=False)
        records = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
            record_id=STUDY if scope == "study" else "all_exported", check_id=check,
            result="unresolved", source_evidence=str(evidence_path),
            output_evidence=str(metadata_source_relative(OUTPUT / 'field_mapping.csv', NAS)),
                details=json.dumps(dict(repair_round=2, preparer_self_check=True,
                repair_completed=True, findings=message), ensure_ascii=False))
            for check, scope, message in [
                ("M02", "sample", "C09 row 13 provenance corrected; optional_gaps identified as a preparer-generated summary, all declared JSON keys checked; 37 sample values retained; independent targeted recheck pending"),
                ("D01", "study", "Field mapping revision changed; prior input_binding is stale. Prior independent image/panel/identity findings retained; independent rebinding pending")]]
        temporary = OUTPUT / "review_results.csv.partial"
        with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
            pd.concat([reviews, pd.DataFrame(records)], ignore_index=True).to_csv(
                temporary, index=False, encoding="utf-8-sig")
            temporary.replace(OUTPUT / "review_results.csv")
    progress("prepared", len(images), sample_rows=len(samples), sample_linked_images=len(images),
             patient_linked_samples=int(samples.patient_key.ne("").sum()),
             reused_tiffs=len(images), newly_exported_images=0,
             gaps=["independent M02/C09 recheck and D01 rebinding pending"])
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


def sample_tables(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Connect FOV -> official parent biospecimen -> participant -> author MIBI ID.

    PublicationID is an explicit HTAN field, not a number inferred from filenames.
    Author survival values and censoring are kept at the author's endpoint;
    HTAN vital status from another snapshot must not change that censoring.
    """
    links = files[["HTANParentBiospecimenID", "HTANParticipantID", "PublicationID"]].drop_duplicates()
    assert not links.HTANParentBiospecimenID.duplicated().any()
    sources = {}
    for name, key, selected in [
        ("Biospecimen", "HTAN Biospecimen ID", links.HTANParentBiospecimenID),
        ("Demographics", "HTAN Participant ID", links.HTANParticipantID),
        ("Diagnosis", "HTAN Participant ID", links.HTANParticipantID),
        ("Exposure", "HTAN Participant ID", links.HTANParticipantID),
    ]:
        frame = read_table(name)
        frame = frame[frame[key].isin(selected)]
        assert len(frame) == len(links) and not frame[key].duplicated().any()
        frame.to_csv(CACHE / f"htan_{name}_selected.csv", index=False, encoding="utf-8-sig")
        sources[name] = frame.set_index(key)
    author = pd.read_csv(SOURCE / "author_survival_metadata.tsv", sep="\t", dtype=str,
                         keep_default_na=False).set_index("MIBI ID", verify_integrity=True)
    author_samples = pd.read_csv(SOURCE / "author_mibi_sample_table.csv", dtype=str,
                                 keep_default_na=False).set_index("MIBI name", verify_integrity=True)
    rows, conflicts = [], []
    for link in links.itertuples(index=False):
        sid, pid, publication_id = link
        bio = sources["Biospecimen"].loc[sid]
        demo = sources["Demographics"].loc[pid]
        diagnosis = sources["Diagnosis"].loc[pid]
        exposure = sources["Exposure"].loc[pid]
        clinical = author.loc[publication_id]
        measured = author_samples.loc[publication_id]
        assert bio["HTAN Parent ID"] == pid
        assert clinical["Vital Status"] == measured["Vital Status"]
        assert float(clinical["Overall Survival Following Biopsy (months)"]) == float(measured["OS tissue"])
        author_vital = {"1": "deceased", "0": "alive"}[clinical["Vital Status"]]
        htan_vital = {"Dead": "deceased", "Alive": "alive"}[demo["Vital Status"]]
        notes = dict(author_mibi_id=publication_id, htan_vital_status=demo["Vital Status"],
                     author_vital_status=clinical["Vital Status"],
                     survival_endpoint="author repository; cutoff date unspecified",
                     age_at_diagnosis_source=int(diagnosis["Age at Diagnosis"]) if pd.notna(diagnosis["Age at Diagnosis"]) else None,
                     age_is_obfuscated=bool(diagnosis["Age Is Obfuscated"]) if pd.notna(diagnosis["Age Is Obfuscated"]) else None,
                     source_collection_days_from_index=bio["Collection Days from Index"],
                     source_timepoint_label=bio["Timepoint Label"],
                     source_prior_therapies_at_collection=clinical["Prior therapies"],
                     collection_index_origin="not confirmed; no derived collection day/age",
                     optional_gaps="physical TMA core map; clinical cutoff date; absolute collection date")
        if author_vital != htan_vital:
            conflicts.append(dict(sample_id=sid, author_mibi_id=publication_id,
                                  author_status=author_vital, htan_status=htan_vital,
                                  resolution="Preserve author endpoint for author OS/censoring; do not combine follow-up snapshots"))
        row = dict(study=STUDY, sample_id=sid, patient_key=pid, source_subject_id=pid,
                   cohort_id="HTAN_MSK_MIBI", center_id="MSKCC", source_sample_id=sid,
                   sample_type=bio["Biospecimen Type"], tissue_type="tumor",
                   anatomical_site=bio["Site of Resection or Biopsy"],
                   lesion_type={"Primary": "primary", "Metastatic": "metastasis",
                                "Local recurrence": "recurrence", "Recurrent": "recurrence"}[bio["Tumor Tissue Type"]],
                   timepoint_label={"baseline": "baseline", "recurrence": "recurrence", "progression": "progression",
                                    "progression/end of treatment": "other"}[bio["Timepoint Label"]],
                   treatment_status_at_collection={"Yes": "treated", "No": "treatment_naive"}[clinical["Treated?"]],
                   collection_method={"Surgical Resection": "resection", "Biopsy": "biopsy"}[bio["Acquisition Method Type"]],
                   preservation_method="ffpe",
                   processing_method="TMA construction from archival FFPE tissue (paper methods)",
                   index_event="diagnosis",
                   index_definition_version="Chan2021 author Initial Stage; v1",
                   sex_reported={"Female": "female", "Male": "male"}[demo["Gender"]],
                   smoking_status="never" if exposure["Smoking Exposure"] == "No - Smoking Exposure" else "",
                   primary_diagnosis=diagnosis["Primary Diagnosis"],
                   primary_diagnosis_code=diagnosis["Morphology"], diagnosis_code_system="ICD-O-3 morphology",
                   primary_site=diagnosis["Tissue or Organ of Origin"], histology=diagnosis["Primary Diagnosis"],
                   grade=diagnosis["Tumor Grade"], stage_group_at_index=clinical["Initial Stage"],
                   stage_system="SCLC limited/extensive stage",
                   vital_status=author_vital,
                   os_time=float(clinical["Overall Survival Following Biopsy (months)"]),
                   os_time_unit="months", os_origin="biopsy", os_event=int(clinical["Vital Status"]),
                   os_event_definition="Death, as encoded by author MIBI survival table Vital Status: 1=death, 0=censored",
                   metadata_notes=json.dumps(notes, ensure_ascii=False))
        rows.append(row)
    columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns
    samples = pd.DataFrame(rows).reindex(columns=columns).fillna("")
    dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
    vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
    for field in dictionary[dictionary.table_name == "sample_metadata"].itertuples(index=False):
        populated = samples.loc[samples[field.field_name].ne(""), field.field_name]
        if field.value_set:
            allowed = vocabulary.loc[vocabulary.value_set == field.value_set, "allowed_value"]
            assert populated.astype(str).isin(allowed).all(), field.field_name
    provenance = dict(sample_rows=len(samples), sample_linked_images=len(files),
                      patient_linked_samples=int(samples.patient_key.ne("").sum()),
                      unique_patients=int(samples.patient_key.nunique()),
                      identity_join="source_id=synapseId -> HTANParentBiospecimenID=HTAN Biospecimen ID -> HTAN Parent ID=HTANParticipantID; PublicationID=MIBI ID=MIBI name",
                      clinical_snapshot_conflicts=conflicts,
                      missing_required_identity_links=[],
                      supplement_attempts="PMC Table S14 link identified; binary endpoint returned an HTML challenge, so author repository clinical table was used",
                      source_urls=["https://github.com/ncihtan/htan-portal", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8628860/", "https://github.com/joechanlab/SCLC_atlas-HTAN"])
    write_json(CACHE / "sample_linkage_evidence.json", provenance)
    rules = [
        image_sample_mapping(),
        mapping(str(metadata_source_relative(SOURCE / 'htan_Biospecimen_selected.csv', NAS)),
                "HTAN Biospecimen ID; HTAN Parent ID", "sample_id; source_sample_id; patient_key; source_subject_id",
                keys="HTANParentBiospecimenID=HTAN Biospecimen ID; HTAN Parent ID=HTANParticipantID"),
        mapping(str(metadata_source_relative(SOURCE / 'htan_Biospecimen_selected.csv', NAS)),
                "Biospecimen Type; Site of Resection or Biopsy; Tumor Tissue Type; Timepoint Label; Acquisition Method Type; Preservation Method",
                "sample_type; anatomical_site; lesion_type; timepoint_label; collection_method; preservation_method; tissue_type",
                values={"Primary": "primary", "Metastatic": "metastasis", "Local recurrence": "recurrence", "Recurrent": "recurrence",
                        "Surgical Resection": "resection", "Biopsy": "biopsy", "Formalin fixed paraffin embedded - FFPE": "ffpe",
                        "progression/end of treatment": "other"},
                notes="tissue_type=tumor from explicit tumor diagnoses and Tumor Tissue Type; source Collection Days from Index kept in metadata_notes because origin is unconfirmed"),
        mapping(str(metadata_source_relative(SOURCE / 'htan_Demographics_selected.csv', NAS)), "Gender; Vital Status", "sex_reported; metadata_notes",
                values={"Female": "female", "Male": "male"}, notes="HTAN Vital Status retained in notes; differs from author survival endpoint in 6 patients"),
        mapping(str(metadata_source_relative(SOURCE / 'htan_Diagnosis_selected.csv', NAS)),
                "Primary Diagnosis; Morphology; Tissue or Organ of Origin; Tumor Grade; Age at Diagnosis; Age Is Obfuscated",
                "primary_diagnosis; histology; primary_diagnosis_code; diagnosis_code_system; primary_site; grade; metadata_notes",
                notes="ICD-O-3 morphology from explicit morphology code; age source retained without unit conversion; one obfuscated age flagged"),
        mapping(str(metadata_source_relative(SOURCE / 'htan_Exposure_selected.csv', NAS)), "Smoking Exposure", "smoking_status",
                values={"No - Smoking Exposure": "never"}, notes="Yes does not resolve former/current; leave that target blank"),
        mapping(str(metadata_source_relative(SOURCE / 'author_survival_metadata.tsv', NAS)),
                "MIBI ID; Initial Stage; Prior therapies; Treated?; Overall Survival Following Biopsy (months); Vital Status",
                "stage_group_at_index; stage_system; index_event; index_definition_version; metadata_notes; treatment_status_at_collection; os_time; os_time_unit; os_origin; os_event; os_event_definition; vital_status",
                keys="HTAN ImagingLevel2.PublicationID=MIBI ID; author MIBI name crosscheck",
                values={"1": "deceased/death", "0": "alive/censored", "Yes": "treated", "No": "treatment_naive"},
                origin="Initial Stage at initial diagnosis; OS measured from biopsy, in months, at author follow-up endpoint",
                notes="All 37 OS durations and event codes agree with author_mibi_sample_table.csv. Six HTAN vital statuses differ; author durations/censoring remain paired. Prior therapies at tissue collection retained verbatim in metadata_notes, not assigned to the diagnosis-index boolean."),
        mapping(str(metadata_source_relative(SOURCE / 'paper.html', NAS)), "Patient cohorts; TMA construction", "processing_method; cohort_id; center_id",
                transform="Paper identifies MSKCC independent archival FFPE MIBI TMA cohort; no physical core ID inferred"),
        mapping(str(metadata_source_relative(META / 'image_manifest.csv', NAS)), "source_id; image_path", "study; image_id; source_image_id; source_files; source_format; source_revision; image_kind; fov_id; split_method; split_evidence; metadata_notes",
                table="image_metadata", transform="image_id/FOV ID from complete published filename; preserve source Synapse ID; already independent full FOV; Release 2.0"),
        mapping(str(metadata_source_relative(META / 'image_manifest.csv', NAS)), "OME Pixels attributes and channel order", "source_axes; source_dtype; source_channel_count; pixel_size_x_um; pixel_size_y_um; source_x0_px; source_y0_px; source_x1_px; source_y1_px; height_px; width_px",
                table="image_metadata", transform="Use actual OME full-resolution shape/dtype/order and exact 0.390625 um spacing; full [0,2048)x[0,2048) acquisition"),
        mapping(str(metadata_source_relative(S1, NAS)), "Study; Original marker; Unified marker; Segmentation channel", "panel_id; channel_count; input_image_path; pad_left_px; pad_top_px; pad_right_px; pad_bottom_px; valid_pixels_path; intensity_description",
                table="image_metadata", transform="Study+exact OME marker label -> panel_from_s1; retain 29 source channels in original order; all native pixels valid, square, zero padding"),
        mapping(str(metadata_source_relative(SOURCE / 'htan_file_records.json', NAS)), "Comment", "region_annotation", table="image_metadata", transform="Copy nonempty source Comment; otherwise blank"),
        metadata_notes_mapping(),
    ]
    return samples, pd.DataFrame(rules), provenance


def publish_tables(images: pd.DataFrame, samples: pd.DataFrame,
                   panel: pd.DataFrame, field_mapping: pd.DataFrame) -> None:
    """Merge and publish tables through the shared publication path."""
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)


def publish_checks(images: pd.DataFrame, samples: pd.DataFrame, validation: dict | None = None) -> None:
    """Record preparer findings, leaving independent acceptance explicitly pending."""
    records = []
    def add(check: str, scope: str, result: str, details: object) -> None:
        records.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                            record_id=STUDY if scope == "study" else "all_exported",
                            check_id=check, result=result,
                            source_evidence=str(metadata_source_relative(SOURCE, NAS)),
                            output_evidence=str(metadata_source_relative(OUTPUT, NAS)),
                            details=json.dumps(dict(preparer_self_check=True, findings=details), ensure_ascii=False)))
    add("S01", "study", "pass", dict(local_final_sources=58, indexed_images=len(images),
        zero_marker_exclusions=[k for k,v in outcomes.items() if not v["admitted"]],
        source_qc_annotation="No - Channels QC for 58/58 source files; reason unspecified; apply the frozen whole-image admitted-marker nonzero rule and preserve this source flag",
        untouched_partial_files=partial_files, original_verification=str(metadata_source_relative(PROOF, NAS))))
    add("M01", "sample", "pass", linkage)
    add("M02", "sample", "pass", "37/37 explicit author clinical joins; full template; endpoint conflicts preserved; optional unknowns blank")
    add("P01", "panel", "pass", "Exact study+source marker join; one ordered 29-channel native panel")
    add("P02", "panel", "pass", "28 downstream measurements plus dsDNA nuclear input; no duplicate downstream marker")
    if validation:
        for check in ["I01", "I02", "I03", "I04", "P03", "I05"]:
            add(check, "image", "pass", validation[check])
    add("D01", "study", "unresolved", dict(exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=len(samples),
        next_step="Different CLI context must independently review and bind D01; preparer cannot grant acceptance"))
    path = OUTPUT / "review_results.csv"
    # Retain review history on resumption; a new preparer check never invents a reviewer.
    frame = pd.DataFrame(records)
    if path.exists():
        frame = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False), frame], ignore_index=True)
    temporary = path.with_suffix(".csv.partial")
    with formal_publish_lease(path, directory=path.parent):
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=["metadata", "export", "verify", "repair"], default="metadata")
args = parser.parse_args()
if args.stage == "repair":
    PROGRESS = PROGRESS.parents[1] / "0159_repair_2/progress.json"
    PREPARER = "Codex CLI Chan2021_MIBI_SCLC 0159_repair_2"
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
# Metadata can be completed while the pixel-export command waits for its NAS
# slot. Serialize this study's stages before reading any existing index rows.
stage_lease = (CACHE / ".preparer.lock").open("a")
fcntl.flock(stage_lease, fcntl.LOCK_EX)
index_path = OUTPUT / "images.parquet"
images = pd.read_parquet(index_path) if index_path.exists() else pd.read_csv(TEMPLATES / "image_metadata.csv")
progress(args.stage, len(images), gaps=["independent review pending"])
if args.stage == "repair":
    repair_provenance(images)
    raise SystemExit(0)

# Reconcile final files against the accepted verification, never rehash sources.
manifest = pd.read_csv(META / "image_manifest.csv", dtype={"source_id": str})
proofs = {x["source_id"]: x for x in json.loads(PROOF.read_text())}
raw = NAS / "raw/public" / STUDY / "htan_release_2_0"
assert {str(metadata_source_relative(p, NAS)) for p in raw.glob("*.ome.tiff")} == set(manifest.image_path)
partial_files = [str(metadata_source_relative(p, NAS)) for p in sorted(raw.glob("*.part*"))]
headers = {}
for item in manifest.itertuples(index=False):
    path, proof = NAS / item.image_path, proofs[item.source_id]
    assert path.stat().st_size == item.size == proof["size"]
    assert path.stat().st_mtime_ns == proof["mtime_ns"], f"Source changed since accepted verification: {path}"
    header = native_header(path)
    assert header["shape"] == proof["shape"] == [29, 2048, 2048]
    assert header["names"] == proof["channel_markers"]
    assert header["axes"] == "CYX" and header["dtype"] == "float32"
    headers[item.source_id] = header
names = headers[manifest.iloc[0].source_id]["names"]
assert all(x["names"] == names for x in headers.values())
channel_manifest = pd.read_csv(META / "channel_markers.csv", keep_default_na=False)
for path, rows in channel_manifest.groupby("image_path", sort=False):
    assert rows.sort_values("channel_index").marker.tolist() == names
panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
write_json(CACHE / "header_inventory.json", headers)

# The raw official ImagingLevel2 table supplies PublicationID, absent in the
# processed portal file JSON. Only this study's 58 exact source IDs are joined.
raw_images = json.loads((SOURCE / "htan_ImagingLevel2.json").read_text())
image_metadata = pd.DataFrame([dict(zip(raw_images["column_order"], r["values"], strict=True))
    for r in raw_images["record_list"] if any(x in set(manifest.source_id) for x in r["values"] if isinstance(x, str))])
files = pd.DataFrame(json.loads((SOURCE / "htan_file_records.json").read_text()))
files = files.merge(image_metadata[["entityId", "PublicationID"]], left_on="synapseId", right_on="entityId", validate="one_to_one")
assert len(files) == len(manifest) == 58
files = files.set_index("synapseId", verify_integrity=True)
for item in manifest.itertuples(index=False):
    assert Path(files.loc[item.source_id, "Filename"]).name == Path(item.image_path).name
files.to_csv(CACHE / "htan_image_selected.csv", index=True, encoding="utf-8-sig")
samples, field_mapping, linkage = sample_tables(files.reset_index())
receipt = CACHE / "export_results.json"
outcomes = json.loads(receipt.read_text()) if receipt.exists() else {}
publish_tables(images, samples, panel, field_mapping)

if args.stage == "export":
    required_bytes = 58 * 29 * 2048 * 2048 * 4
    if shutil.disk_usage(NAS).free - required_bytes < 100 * 1024**3 or shutil.disk_usage("/tmp").free < 64 * 1024**3:
        progress("capacity_blocked", len(images), required_uncompressed_bytes=required_bytes,
                 gaps=["100 GiB NAS / 64 GiB local reserve unavailable"])
        raise SystemExit("Capacity reserve insufficient")
    for item in manifest.itertuples(index=False):
        image_id = Path(item.image_path).name.removesuffix(".ome.tiff")
        destination = OUTPUT / "images" / image_id / "input.tiff"
        existing = images[images.image_id == image_id]
        if not existing.empty:
            saved = native_header(destination)
            assert saved["shape"] == [29, 2048, 2048] and saved["dtype"] == "float32"
            assert saved["names"] == panel.loc[panel.keep_channel, "marker_id"].tolist()
            continue
        if image_id in outcomes and not outcomes[image_id]["admitted"]:
            continue
        progress("exporting", len(images), current_image=image_id, gaps=["independent review pending"])
        header = headers[item.source_id]
        source = SourceSpec(image_id, PANEL_ID, "tiff", NAS / item.image_path, tuple(range(29)),
                            pixel_size_um=(header["pixel_size_x_um"], header["pixel_size_y_um"]))
        if image_id in outcomes and destination.exists():
            # Recover the narrow interruption window after a complete TIFF and
            # its signal receipt were saved, before its index row was published.
            saved = native_header(destination)
            assert saved["shape"] == [29, 2048, 2048] and saved["dtype"] == "float32"
            assert saved["names"] == panel.loc[panel.keep_channel, "marker_id"].tolist()
            result = outcomes[image_id]
        else:
            with open_source(source) as reader:
                result = export_input(reader, panel, destination)
        outcomes[image_id] = result
        write_json(receipt, outcomes)
        if not result["admitted"]:
            publish_checks(images, samples)
            continue
        file_record = files.loc[item.source_id]
        record = dict(study=STUDY, image_id=image_id,
            sample_id=file_record.HTANParentBiospecimenID, panel_id=PANEL_ID, image_kind="fov",
            source_image_id=item.source_id, source_files=[item.image_path], source_format="ome_tiff",
            source_axes="CYX", source_dtype="float32", source_revision=file_record.releaseVersion,
            fov_id=Path(file_record.Filename).name, region_annotation=file_record.Comment or "",
            source_x0_px=0, source_y0_px=0, source_x1_px=2048, source_y1_px=2048,
            pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
            height_px=2048, width_px=2048, source_channel_count=29, channel_count=29,
            pixel_size_x_um=header["pixel_size_x_um"], pixel_size_y_um=header["pixel_size_y_um"],
            input_image_path=str(metadata_source_relative(destination, NAS)), valid_pixels_path="",
            intensity_description="Published HTAN MIBI ImagingLevel2 float32 intensities; original source processing retained; no CellWeft rescaling",
            split_method="already_independent", split_evidence=str(metadata_source_relative(SOURCE / 'htan_image_selected.csv', NAS)) + "; explicit parent biospecimen; full published FOV",
            metadata_notes=json.dumps(dict(author_mibi_id=file_record.PublicationID,
                htan_data_file_id=file_record.HTANDataFileID, source_qc=file_record.PassedQC,
                core_map="physical TMA core ID unavailable; image remains the published full FOV")))
        images = pd.concat([images, pd.DataFrame([record])], ignore_index=True)
        publish_tables(images, samples, panel, field_mapping)
        publish_checks(images, samples)
        progress("exporting", len(images), last_completed_image=image_id, gaps=["independent review pending"])
        print(json.dumps(dict(image_id=image_id, exported_images=len(images)), ensure_ascii=False), flush=True)

if args.stage == "verify":
    # Every delivered TIFF header, plus all 29 channels in two bounded windows
    # for the first, middle, and last FOV (one native-layout/panel group).
    expected_names = panel.loc[panel.keep_channel, "marker_id"].tolist()
    for image in images.itertuples(index=False):
        header = native_header(NAS / image.input_image_path)
        assert header["shape"] == [29, 2048, 2048] and header["dtype"] == "float32"
        assert header["axes"] == "CYX"
        assert header["names"] == expected_names
        assert header["pixel_size_x_um"] == header["pixel_size_y_um"] == 0.390625
    pixel_checks = []
    selection = sorted(set([0, len(images)//2, len(images)-1])) if len(images) else []
    for position in selection:
        image = images.iloc[position]
        with tifffile.TiffFile(NAS / image.source_files[0]) as source, tifffile.TiffFile(NAS / image.input_image_path) as output:
            for window in [(992, 992, 1056, 1056), (1984, 1984, 2048, 2048)]:
                source_nonzero = 0
                for channel in range(29):
                    before = read_tiff_window(source, channel, window)
                    after = read_tiff_window(output, channel, window)
                    assert before.dtype == after.dtype and np.array_equal(before, after)
                    source_nonzero += int(np.count_nonzero(before))
                if window[0] == 992:
                    assert source_nonzero > 0, "The internal pixel check must contain source signal"
                pixel_checks.append(dict(image_id=image.image_id, window=list(window), channels=list(range(29)),
                                         source_nonzero_values=source_nonzero, exact_equal=True))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(selection), 4, figsize=(12, 3.3 * len(selection)), squeeze=False)
    for row, position in enumerate(selection):
        image = images.iloc[position]
        with tifffile.TiffFile(NAS / image.input_image_path) as tif:
            for column, channel in enumerate([13, 17, 8, 24]):
                pixels = tif.pages[channel].asarray(maxworkers=2)
                reduced = pixels.reshape(512, 4, 512, 4).max(axis=(1, 3))
                nonzero = reduced[reduced > 0]
                axes[row, column].imshow(reduced, cmap="gray", vmin=0,
                    vmax=float(np.quantile(nonzero, 0.99)) if len(nonzero) else 1)
                axes[row, column].set_title(f"{image.image_id}\n{expected_names[channel]}", fontsize=8)
                axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(CACHE / "representative_overview.png", dpi=130)
    plt.close(fig)
    assert images.image_id.is_unique and images.sample_id.isin(samples.sample_id).all()
    validation = dict(I01=f"{len(images)} unique indexed FOVs; all input paths reopened",
        I02=f"{len(images)} FOVs -> {len(samples)} explicit biological samples -> 37 participants; one native panel",
        I03="All 2048x2048 full native acquisition boxes; zero padding; 0.390625 um; no axis change",
        I04="HTAN explicitly assigns one parent biospecimen per published FOV; no multi-sample split inferred; representative overview saved for visual inspection",
        P03=f"{len(images)} headers: CYX float32, 29 exact selected marker labels, exact native spacing",
        I05=pixel_checks, reviewer="", independent_acceptance="pending")
    write_json(CACHE / "preparer_validation.json", validation)
    publish_checks(images, samples, validation)

publish_checks(images, samples)
progress("prepared" if len(images) + sum(not x["admitted"] for x in outcomes.values()) == 58 else "metadata_ready",
         len(images), sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
         patient_linked_samples=int(samples.patient_key.ne("").sum()),
         gaps=["independent review pending"], clinical_snapshot_conflicts=len(linkage["clinical_snapshot_conflicts"]))
