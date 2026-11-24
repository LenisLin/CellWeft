"""Prepare the closed S-BIAD565 native-TIFF subset, preserving native intensity.

Inputs: accepted nine-file closeout, official image_key, supplementary tables, S1.
Outputs: per-core input.tiff, merged input tables, preparer evidence and progress.
Run with PYTHONPATH=src .venv-data/bin/python -B and the shared NAS lock root.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import openpyxl
import pandas as pd
from PIL import Image
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Roider2024_CODEX_Lymphoma"
NAS = Path(os.environ.get("CELLWEFT_DATA_ROOT", "/mnt/NAS_21T/ProjectData/CellWeft"))
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
CACHE = PROJECT / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "roider:panel1"
CACHE.mkdir(parents=True, exist_ok=True)
JOB.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def checkpoint(value: dict) -> None:
    temporary = JOB / "progress.json.partial"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(JOB / "progress.json")


def version(path: Path) -> dict:
    stat = path.stat()
    return dict(dev=stat.st_dev, ino=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


# Resolve physical cores to biological LN specimens and patients using primary keys.
closeout = json.loads((META / "selected_subset_closeout_20261025.json").read_text())
ordered = json.loads((META / "native_tiff_ordered_markers.json").read_text())
key_path = RAW_METADATA / "image_key.json"
official = {row["path"]: {item["name"]: item["value"] for item in row["attributes"]}
            for row in json.loads(key_path.read_text())}
workbook_path = RAW_METADATA / "41556_2024_1358_MOESM3_ESM.xlsx"
workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
rows = list(workbook["Supplementary Table 1"].values)
clinical = {str(row[0]): dict(zip(rows[0], row)) for row in rows[1:]}
workbook.close()
author_path = RAW_METADATA / "author_metaData.csv"
author = pd.read_csv(author_path, dtype=str, keep_default_na=False).set_index("PatientID", verify_integrity=True)
scope = closeout["acquisitions"]
assert len(scope) == 9 and closeout["qualified_usable_subset"]
for item in scope:
    attrs = official[item["source_filename"]]
    assert attrs["PatientID"] == item["patient_id"]
    assert attrs["Entity"] == item["entity"] and attrs["TMA"] == item["tma_id"]
    assert version(NAS / item["path"]) == item["file_stat"]

# S1 selects the final DRAQ5 nuclear measurement; repeated Hoechst is unselected.
panel = panel_from_s1(STUDY, PANEL_ID, ordered["ordered_channel_names"], S1)
assert len(panel) == 104 and int(panel.keep_channel.sum()) == 51
assert panel.loc[panel.nuclear, "source_channel_index"].tolist() == [103]
estimated_output = len(scope) * 51 * 5120 * 5120 * 2
assert shutil.disk_usage(NAS).free > 100 * 2**30 + estimated_output
assert shutil.disk_usage("/tmp").free > 64 * 2**30 + 51 * 5120 * 5120 * 2

sample_rows = []
for subject in dict.fromkeys(item["patient_id"] for item in scope):
    row = {name: "" for name in pd.read_csv(TEMPLATES / "sample_metadata.csv").columns}
    original = clinical[subject]
    extra = author.loc[subject]
    assert extra["MultiIF"] == "+"
    assert extra["Entity"] == original["Entity"].split(",")[0]
    row.update(study=STUDY, sample_id=f"specimen_{subject}", patient_key=subject,
               source_subject_id=subject, source_sample_id=subject,
               biospecimen_group_id=subject, sample_type="lymph node tissue",
               anatomical_site="lymph node", preservation_method="ffpe",
               primary_diagnosis=original["Entity"],
               index_event="specimen_collection", index_definition_version="author_LN_collection_v1",
               sex_reported={"F": "female", "M": "male"}[extra["Sex"]],
               histology=extra["Entity"] + ("; " + extra["Subtype"] if extra["Subtype"] != "NA" else ""),
               grade="FL grade " + extra["FL_Grade"] if extra["FL_Grade"] != "NA" else "",
               prior_systemic_therapy={"Yes": "true", "No": "false", "NA": ""}[extra["Pretreatment"]],
               metadata_notes="Official image_key PatientID identifies one LN specimen; repeated TMA cores share this sample. "
               + "Author Age=" + extra["Age"] + " (unit and time origin unspecified in CSV; age_at_* fields left blank). "
               + "Index denotes this LN collection; no survival outcomes or stage provided in examined cohort sources.")
    row["tissue_type"] = "normal" if original["Entity"] == "rLN" else "tumor"
    row["disease_status_at_collection"] = {"Initial diagnosis": "active", "Relapse": "relapsed"}.get(original["Status"], "")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)

fields = []
def mapping(source: Path, sheet: str, source_field: str, target: str, target_field: str,
            rule: str, notes: str = "") -> None:
    row = {name: "" for name in pd.read_csv(TEMPLATES / "field_mapping.csv").columns}
    row.update(study=STUDY, source_file=relative(source), source_sheet_or_table=sheet,
               source_field=source_field, target_table=target, target_field=target_field,
               join_keys="image_key.path = source_filename; PatientID = Supplementary Table 1.PatientID",
               transformation_rule=rule, notes=notes)
    fields.append(row)

mapping(key_path, "image_key", "path;TMA;PatientID;Entity", "image_metadata", "sample_id;core_id;slide_id;region_annotation",
        "One independent full native core per TIFF; sample_id=specimen_+PatientID; core_id=TMA+reg; preserve Entity region label.")
mapping(key_path, "image_key", "PatientID", "sample_metadata", "sample_id;patient_key;source_subject_id;source_sample_id;biospecimen_group_id",
        "One LN specimen per author PatientID; repeated physical TMA cores share sample and patient; no filename-derived patient identity.")
mapping(workbook_path, "Supplementary Table 1", "PatientID;Entity;Status", "sample_metadata", "primary_diagnosis;tissue_type;disease_status_at_collection;timepoint_label",
        "Preserve Entity; rLN=normal, lymphoma=tumor. Initial diagnosis=active; Relapse=relapsed. No study baseline/timepoint inferred from diagnosis status; missing Status stays blank.",
        "All 9 selected cores link to 7 LN samples and 7 patients. Additional sex, histology, grade, pretreatment and raw Age are obtained from author metaData.csv; stage and outcomes remain unknown.")
mapping(author_path, "metaData.csv", "PatientID;Entity;Subtype;FL_Grade;Sex;Pretreatment;Age;MultiIF", "sample_metadata",
        "sex_reported;histology;grade;prior_systemic_therapy;index_event;index_definition_version;metadata_notes",
        "Exact PatientID join, require MultiIF=+. F=female/M=male; preserve Entity/Subtype as histology and FL_Grade with explicit FL grade prefix. "
        "Pretreatment Yes=true/No=false/NA=blank; paper defines systemic treatment before LN collection. Index=specimen_collection, author_LN_collection_v1. "
        "Preserve raw Age in metadata_notes with unspecified unit/time origin; do not assign age_at_collection/index/diagnosis. No molecular-subtype conversion of non-GCB.",
        "Author repository data/metaData.csv downloaded and all 7 selected specimen keys matched. Clinical primary table and official image key retained separately.")
mapping(META / "native_tiff_ordered_markers.json", "ordered_channel_names;layout", "ordered_channel_names", "panel_mapping", "source_channel_index;source_name",
        "ImageJ ZCYX encodes 52 staining cycles x 2 readouts; flatten in native page order, no spatial Z or time selection.")
mapping(S1, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "marker_id;role;nuclear;body;keep_channel;output_channel_index",
        "Exact study+source label join; retain 50 protein measurements and selected final DRAQ5; exclude Blank and all unselected repeated Hoechst; retain native order.")
mapping(META / "selected_subset_closeout_20261025.json", "acquisitions;source_evidence", "physical_core_id;patient_id;eligibility", "sample_metadata", "sample_type;anatomical_site;preservation_method;metadata_notes",
        "Reuse verified primary methods: LN FFPE cores, two cores per specimen; preserve official specimen keys. Original 35-file cohort contains 19 people; only nine complete local files are prepared.",
        "Public BioStudies API and image_key re-read; local supplement read. Paper DOI 10.1038/s41556-024-01358-2 and author GitHub Huber-group-EMBL/CITEseqLN-Tcells inspected. GitHub tree API returned HTTP403 rate limit; official image_key already resolves all selected core/sample/patient links.")
field_mapping = pd.DataFrame(fields)

# A completed image receipt permits resumption without re-exporting correct pixels.
evidence_path = CACHE / "preparation_evidence.json"
evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {"images": {}}
image_rows = []
current = None
checkpoint(dict(status="preparing", expected_images=9, completed_images=[], next_image=scope[0]["physical_core_id"],
                estimated_uncompressed_output_bytes=estimated_output, local_reserve_bytes=64*2**30, nas_reserve_bytes=100*2**30))
try:
    # Whole-core nuclear/CD20/CD3 previews cover tumour and reactive-control sources.
    for item in (scope[0], scope[-1]):
        preview = CACHE / (item["physical_core_id"] + "_overview.png")
        if not preview.exists():
            with formal_read_lease(NAS / item["path"], data_root=NAS):
                with tifffile.TiffFile(NAS / item["path"]) as tif:
                    channels = [tif.pages[index].asarray(maxworkers=2)[::8, ::8] for index in (103, 57, 71)]
            scaled = [np.clip(values / max(float(np.percentile(values, 99.5)), 1), 0, 1) for values in channels]
            Image.fromarray((np.stack(scaled, axis=-1) * 255).astype("uint8")).save(preview)
    for item in scope:
        current = item["physical_core_id"]
        source = NAS / item["path"]
        destination = OUTPUT / "images" / current / "input.tiff"
        previous = evidence["images"].get(current)
        scientific_versions = {relative(path): version(path) for path in [source, S1, META / "native_tiff_ordered_markers.json"]}
        reusable = (previous is not None and previous["scientific_versions"] == scientific_versions
                    and destination.exists() and previous["output_version"] == version(destination))
        if reusable:
            result = previous["export"]
        else:
            checkpoint(dict(status="preparing", expected_images=9, completed_images=[r["image_id"] for r in image_rows], next_image=current, stage="export_input"))
            spec = SourceSpec(current, PANEL_ID, "tiff", source, tuple(range(104)),
                              source_revision="S-BIAD565:selected_subset_closeout_20261025")
            # export_input owns its read lease, local checkpoint and formal write lease.
            with open_source(spec) as reader:
                assert reader.shape == (104, 3750, 4998) and str(reader.dtype) == "uint16"
                result = export_input(reader, panel, destination, data_root=NAS)
            assert result["admitted"]
            with formal_read_lease(destination, data_root=NAS):
                with tifffile.TiffFile(destination) as tif:
                    assert tif.series[0].shape == (51, 4998, 4998)
                    assert str(tif.series[0].dtype) == "uint16"
                # Source/output interior and boundary values in every retained channel.
                with tifffile.TiffFile(source) as src, tifffile.TiffFile(destination) as out:
                    for channel in panel[panel.keep_channel].itertuples():
                        for x, y in [(2400, 1800), (0, 0)]:
                            native = read_tiff_window(src, channel.source_channel_index, (x,y,x+32,y+32))
                            saved = read_tiff_window(out, int(channel.output_channel_index), (x,y+624,x+32,y+656))
                            assert np.array_equal(native, saved)
                    assert not read_tiff_window(out, 0, (0,0,32,32)).any()
            evidence["images"][current] = dict(scientific_versions=scientific_versions,
                output_version=version(destination), export=result,
                pixel_self_check="all 51 retained channels, 32x32 source windows (2400,1800) and (0,0), y_output=y_source+624; upper padding zero")
            evidence_path.write_text(json.dumps(evidence, indent=2))
        row = {name: None for name in pd.read_csv(TEMPLATES / "image_metadata.csv").columns}
        row.update(study=STUDY, image_id=current, sample_id=f"specimen_{item['patient_id']}", panel_id=PANEL_ID,
            image_kind="tma_core", source_image_id=item["acquisition_id"], source_files=[item["path"]],
            source_format="tiff", source_axes="ZCYX", source_dtype="uint16",
            source_revision="S-BIAD565:selected_subset_closeout_20261025", slide_id=item["tma_id"], core_id=current,
            region_annotation=item["entity"], source_x0_px=0, source_y0_px=0, source_x1_px=4998, source_y1_px=3750,
            pad_left_px=0, pad_right_px=0, pad_top_px=624, pad_bottom_px=624,
            height_px=4998, width_px=4998, source_channel_count=104, channel_count=51,
            input_image_path=relative(destination), intensity_description="Published registered uint16 CODEX fluorescence; native decoded values retained without normalization.",
            split_method="already_independent", split_evidence=relative(key_path)+"; official path/TMA/PatientID/Entity; one core per source TIFF",
            metadata_notes="ImageJ Z axis stores staining cycles, flattened to native TIFF page order. Pixel scale not declared in native TIFF; left unknown.")
        image_rows.append(row)
        save_input_tables(OUTPUT, pd.DataFrame(image_rows), samples, panel, field_mapping, data_root=NAS)
        checkpoint(dict(status="preparing", expected_images=9, completed_images=[r["image_id"] for r in image_rows], next_image=None, stage="image_and_tables_closed"))
        print(f"closed {current}: {len(image_rows)}/9", flush=True)

    # Preparation facts are separate from independent review; no D01 pass is written.
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT, data_root=NAS):
        if not (OUTPUT / "review_results.csv").exists():
            pd.read_csv(TEMPLATES / "review_results.csv").to_csv(OUTPUT / "review_results.csv", index=False, encoding="utf-8-sig")
    checkpoint(dict(status="prepared", expected_images=9, exported_images=len(image_rows), sample_rows=len(samples),
                    sample_linked_images=len(image_rows), patient_linked_samples=len(samples), distinct_patients=samples.patient_key.nunique(), independent_review="pending"))
except BaseException as error:
    checkpoint(dict(status="waiting_io" if isinstance(error, (OSError, KeyboardInterrupt)) else "failed",
                    completed_images=[r["image_id"] for r in image_rows], next_image=current,
                    error=str(error), traceback=traceback.format_exc(), evidence_path=relative(evidence_path),
                    resume="Rerun this script; reuse version-bound image receipts and shared export publication checkpoints."))
    raise
