"""Prepare the accepted seven physical IMC cores from public S-BIAD1555.

Inputs: closed native proofs, physical-core review, public specimen table and S1.
Outputs: native float32 input TIFFs, five input tables and resumable receipts.
Run with PYTHONPATH=src .venv-data/bin/python -B and shared NAS lock root.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "VanOost2025_IMC_UPSMFS"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE_MANIFEST = NAS / "raw/public" / STUDY / "manifest.json"
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL = "vanoost:MFS_TMA_1:panel1"
LESION_TYPES = {"Primary": "primary", "Recurrence": "recurrence", "Metastasis": "metastasis"}
CACHE.mkdir(parents=True, exist_ok=True)
JOB.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return path.as_posix() if path.is_relative_to(CACHE.parent) else str(path.relative_to(NAS))


def version(path: Path) -> dict:
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                dev=stat.st_dev, ino=stat.st_ino)


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def checkpoint(**values: object) -> None:
    save_json(JOB / "progress.json", dict(study=STUDY, phase="prepare",
                                        scientific_repair_round=0, **values))


# Resolve accepted physical cores through ROI_ID -> L_ID -> Pat_ID, all on NAS.
manifest_path = SOURCE_MANIFEST
geometry_path = SOURCE_MANIFEST
with formal_read_lease(manifest_path, data_root=NAS):
    manifest = json.loads(manifest_path.read_text())["preparation_sources"]["core_selection"]
    geometry = json.loads(geometry_path.read_text())["preparation_sources"]["core_geometry"]
scope = manifest["selection"]
assert len(scope) == manifest["selected_images"] == 7
assert geometry["selected_acquisition_ids"] == [item["source_acquisition_id"] for item in scope]
for item in manifest["source_files"]:
    assert {key: version(Path(item["path"]))[key] for key in ("size", "mtime_ns")} == item["file_stat"]
clinical_path = Path(manifest["metadata_xlsx"])
with formal_read_lease(clinical_path, data_root=NAS):
    clinical = pd.read_excel(clinical_path, dtype=str, keep_default_na=False)
    proofs = [json.loads(SOURCE_MANIFEST.read_text())["preparation_sources"]["native_images"][item["physical_id"]] for item in scope]
    source_names = [channel["label"] for channel in proofs[0]["ordered_channels"]]
    panel = panel_from_s1(STUDY, PANEL, source_names, S1)
assert len(panel) == 49 and panel.keep_channel.sum() == 43
estimated = sum(max(p["payload_bounds"]["width"], p["payload_bounds"]["height"])**2 * 43 * 4 for p in proofs)
checkpoint(status="preparing", stage="space_check", expected_images=7, completed_images=[], estimated_output_bytes=estimated)
if shutil.disk_usage(NAS).free < 100*2**30 + estimated or shutil.disk_usage("/tmp").free < 64*2**30 + estimated:
    checkpoint(status="waiting_io", stage="space_check", expected_images=7, completed_images=[],
               reason="Need 100 GiB NAS and 64 GiB local reserve in addition to estimated export.")
    raise SystemExit(0)

sample_rows = {}
source_rows = []
for item, proof in zip(scope, proofs):
    assert proof["complete"] and proof["native_order_verified"] and proof["payload_bounds_complete"]
    assert [c["label"] for c in proof["ordered_channels"]] == source_names
    assert not item["split_roi"]
    identity = proof["metadata_identity"]
    matches = clinical[clinical.ROI_ID == identity["roi_id"]]
    assert len(matches) == 1
    original = matches.iloc[0]
    assert original.L_ID == identity["l_id"] and original.Pat_ID == item["patient_id"]
    assert original.Diagnosis in {"MFS", "USTS"} and original.Sample_Type in {"Primary", "Recurrence", "Metastasis"}
    assert original["Any_split_ROIs?"] == "False"
    specimen = clinical[clinical.L_ID == original.L_ID]
    for key in ["Pat_ID", "Diagnosis", "Sample_Type", "Treatment"]:
        assert specimen[key].nunique() == 1
    assert specimen.loc[specimen.Grade.isin(["1", "2", "3"]), "Grade"].nunique() <= 1
    source_rows.append(original)
    sample_id = "specimen_" + original.L_ID
    row = dict.fromkeys(pd.read_csv(TEMPLATES / "sample_metadata.csv").columns, "")
    row.update(study=STUDY, sample_id=sample_id, patient_key=original.Pat_ID,
               source_subject_id=original.Pat_ID, source_sample_id=original.L_ID,
               biospecimen_group_id=original.L_ID, sample_type="tissue",
               lesion_type=LESION_TYPES[original.Sample_Type],
               tissue_type="tumor", preservation_method="ffpe",
               primary_diagnosis=original.Diagnosis, histology=original.Diagnosis,
               grade=original.Grade if original.Grade in {"1", "2", "3"} else "",
               source_grade_code=original.Grade, source_treatment_code=original.Treatment,
               metadata_notes="ROI_ID links to source L_ID specimen and Pat_ID. Treatment code retained verbatim; timing and blank meaning not inferred. Grade scheme unspecified; code 0 not interpreted. Age, sex and individual endpoints unavailable in published ROI table.")
    sample_rows[sample_id] = row
samples = pd.DataFrame(sample_rows.values())

# Describe every nonempty source-derived clinical field, retaining ambiguous codes.
fields = []
def mapping(source_file: Path, sheet: str, source: str, table: str, target: str, rule: str, notes: str = "") -> None:
    row = dict.fromkeys(pd.read_csv(TEMPLATES / "field_mapping.csv").columns, "")
    row.update(study=STUDY, source_file=relative(source_file), source_sheet_or_table=sheet,
               source_field=source, target_table=table, target_field=target,
               join_keys="MCD filename/acquisition -> metadata ROI_ID -> L_ID -> Pat_ID", value_mapping="{}",
               transformation_rule=rule, conflict_rule="Require unique, consistent source join; do not infer missing values.", notes=notes)
    fields.append(row)
for source, target, rule in [
    ("L_ID", "sample_id", "Prefix specimen_ to original specimen identifier; no grouping by ROI location."),
    ("L_ID", "source_sample_id", "Verbatim string."), ("L_ID", "biospecimen_group_id", "Verbatim specimen identifier."),
    ("Pat_ID", "patient_key", "Verbatim string; preserve original deidentified patient identity."),
    ("Pat_ID", "source_subject_id", "Verbatim string."),
    ("Sample_Type", "lesion_type", "Primary -> primary; Recurrence -> recurrence; Metastasis -> metastasis. Lesion context does not establish treatment timing."),
    ("Diagnosis", "primary_diagnosis", "Verbatim; USTS is source wording for UPS."),
    ("Diagnosis", "histology", "Verbatim diagnostic code."),
    ("Grade", "grade", "Retain 1/2/3; blank and ambiguous 0 left empty; grading system unspecified."),
    ("Grade", "source_grade_code", "Verbatim source grade code.") ,
    ("Treatment", "source_treatment_code", "Verbatim source code; no treatment timing or blank-to-untreated conversion.")]:
    definition = json.dumps(dict(data_type="string", definition=rule, unit=None,
                                 time_origin="Not specified by source")) if target.startswith("source_") and target.endswith("_code") else ""
    mapping(clinical_path, "metadata_STS", source, "sample_metadata", target, rule, definition)
primary = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12222583/"
for target, rule in [("preservation_method", "IMC TMA methods explicitly use FFPE tissue; normalize to ffpe."),
                     ("tissue_type", "Tumor representative soft tissue sarcoma TMA specimens -> tumor."),
                     ("sample_type", "FFPE TMA tissue biospecimen -> tissue; lesion context is recorded separately in lesion_type.")]:
    mapping(primary, "Tissue samples and IMC methods", "Study methods", "sample_metadata", target, rule)
for row in fields:
    if row["target_field"] == "lesion_type":
        row["value_mapping"] = json.dumps(LESION_TYPES)
for target, rule in [("sample_id", "ROI_ID -> L_ID; prefix specimen_."), ("region_annotation", "Preserve ROI_location verbatim."), ("roi_id", "Preserve ROI_ID verbatim.")]:
    mapping(clinical_path, "metadata_STS", "ROI_ID/ROI_location/L_ID", "image_metadata", target, rule)
mapping(geometry_path, "native_geometry", "AcquisitionROI/Panorama", "image_metadata", "core_id", "Use accepted seven distinct physical core identities; retain full acquisition frame, not full 1.6 mm punched core extent.")
mapping(manifest_path, "selection/original_proof", "ordered_channels/native_metadata", "image_metadata", "source_image_id", "Original MCD slide 0 and acquisition; native 1 um raster and float32 Dual-count values; no scaling.")
mapping(SOURCE_MANIFEST, "public_search", "source search", "sample_metadata", "metadata_notes", "Checked local native annotations, public ROI workbook, paper methods/data availability and successful public BioStudies API. Author GitLab API request timed out; see public_search.json.")
supplement = primary + "#supplementary-material"
mapping(supplement, "Supplementary tables 1-4 and figure legends", "Supplemental information", "sample_metadata", "metadata_notes", "Panel, cell-type definitions and aggregate survival analyses; no patient-to-outcome lookup table. Author GitLab tree retrieval timed out; keep explicit clinical-source gap.")
field_mapping = pd.DataFrame(fields)

# Repair only the reviewed clinical categories, preserving the confirmed inputs.

# Version-bound per-image receipts permit exact resumption without rewriting inputs.
evidence_path = CACHE / "preparation_evidence.json"
evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {"images": {}}
image_rows = []
current = None
try:
    for item, proof, original in zip(scope, proofs, source_rows):
        current = item["physical_id"]
        source = Path(proof["source_path"])
        destination = OUTPUT / "images" / current / "input.tiff"
        dependencies = {relative(p): version(p) for p in [source, S1, manifest_path, geometry_path, clinical_path ]}
        prior = evidence["images"].get(current)
        reusable = (prior is not None and prior["dependencies"] == dependencies and destination.exists()
                    and version(destination) == prior["output_version"])
        checkpoint(status="preparing", stage="export_input", expected_images=7,
                   completed_images=[r["image_id"] for r in image_rows], next_image=current)
        spec = SourceSpec(current, PANEL, "mcd", source, tuple(range(49)), acquisition_id=item["source_acquisition_id"],
                          pixel_size_um=(1.0,1.0), intensity_description="Native IMC Dual count float32")
        with ExitStack() as stack:
            # Reader initialization reads coordinate records; exporter owns later leases.
            with formal_read_lease(source, data_root=NAS):
                reader = stack.enter_context(open_source(spec))
            if reusable:
                result = prior["export"]
            else:
                result = export_input(reader, panel, destination, data_root=NAS)
            assert result["admitted"]
            with formal_read_lease(destination, data_root=NAS):
                with tifffile.TiffFile(destination) as tif:
                    assert tif.series[0].shape == (43,1000,1000) and str(tif.series[0].dtype) == "float32"
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(tif.ome_metadata)
                    assert [c.attrib["Name"] for c in root.findall('.//{*}Channel')] == panel.loc[panel.keep_channel, "marker_id"].tolist()
                    for channel in panel[panel.keep_channel].itertuples():
                        for x,y in [(480,480),(0,0)]:
                            native = reader.read_window([channel.source_channel_index], (x,y,x+32,y+32))[0]
                            saved = read_tiff_window(tif, int(channel.output_channel_index), (x,y,x+32,y+32))
                            assert np.array_equal(native, saved)
                assert reader.valid_window((0,0,1000,1000)).all()
                if current == scope[0]["physical_id"]:
                    preview = reader.read_window([44,15,46], (0,0,1000,1000))[:,::4,::4]
                    scaled = [np.clip(a/max(float(np.percentile(a,99.5)),1),0,1) for a in preview]
                    preview_image = Image.fromarray((np.stack(scaled,axis=-1)*255).astype("uint8"))
            if current == scope[0]["physical_id"]:
                with formal_publish_lease(CACHE / "native_nuclear_CD45_Vimentin_overview.png", directory=CACHE, data_root=NAS):
                    preview_image.save(CACHE / "native_nuclear_CD45_Vimentin_overview.png")
        evidence["images"][current] = dict(dependencies=dependencies, output_version=version(destination), export=result,
            self_check="43 retained channels; exact float32 32x32 blocks at (480,480) and (0,0); OME numeric output order; full 1000x1000 measured validity.")
        save_json(evidence_path, evidence)
        row = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
        row.update(study=STUDY, image_id=current, sample_id="specimen_"+original.L_ID, panel_id=PANEL,
            image_kind="tma_core", source_image_id=proof["image_id"], source_files=[relative(source)], source_format="mcd",
            source_acquisition_id=str(item["source_acquisition_id"]), source_axes="CYX", source_dtype="float32",
            source_revision="original_public_S-BIAD1555", slide_id="MFS_TMA_1:slide0", core_id=current, roi_id=original.ROI_ID,
            region_annotation=original.ROI_location, source_x0_px=0, source_y0_px=0, source_x1_px=1000, source_y1_px=1000,
            pad_left_px=0,pad_top_px=0,pad_right_px=0,pad_bottom_px=0,height_px=1000,width_px=1000,
            source_channel_count=49,channel_count=43,pixel_size_x_um=1.0,pixel_size_y_um=1.0,
            input_image_path=relative(destination),intensity_description="Native float32 IMC Dual-count intensities; no normalization or resampling.",
            split_method="already_independent",split_evidence=relative(geometry_path),
            metadata_notes="Full native acquisition on one accepted physical core; source acquisition covers a 1 mm square within the punched core. Other acquisitions retain separate qualification gaps.")
        image_rows.append(row)
        save_input_tables(OUTPUT,pd.DataFrame(image_rows),samples,panel,field_mapping,data_root=NAS)
        checkpoint(status="preparing",stage="image_and_tables_closed",expected_images=7,
                   completed_images=[r["image_id"] for r in image_rows],next_image=None)
        print(f"closed acquisition {item['source_acquisition_id']}: {len(image_rows)}/7",flush=True)
    # Independent review is pending; preparer never writes D01 pass.
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT, data_root=NAS):
        if not (OUTPUT / "review_results.csv").exists():
            pd.read_csv(TEMPLATES / "review_results.csv").to_csv(OUTPUT / "review_results.csv",index=False,encoding="utf-8-sig")
    save_json(CACHE / "source_scope_and_coverage.json",dict(
        selected_images=7, other_acquisitions=len(manifest["excluded_members"]),
        deferred_members=manifest["excluded_members"], source_search=relative(CACHE / "public_search.json"),
        source_metadata=relative(clinical_path), physical_scope=relative(geometry_path),
        sample_rows=len(samples),sample_linked_images=len(image_rows),patient_linked_samples=len(samples),
        distinct_patients=int(samples.patient_key.nunique()),
        metadata_gaps="No per-patient age, sex or endpoint source in published ROI workbook. Treatment timing unresolved; verbatim code retained.",
        independent_review="pending"))
    checkpoint(status="prepared",expected_images=7,exported_images=len(image_rows),sample_rows=len(samples),
               sample_linked_images=len(image_rows),patient_linked_samples=len(samples),distinct_patients=int(samples.patient_key.nunique()),
               independent_review="pending", evidence_path=relative(evidence_path))
except BaseException as error:
    checkpoint(status="waiting_io" if isinstance(error,(OSError,KeyboardInterrupt)) else "needs_repair",
               completed_images=[r["image_id"] for r in image_rows],next_image=current,error=str(error),
               traceback=traceback.format_exc(),resume="Rerun script; reuse version-bound receipts and exporter local publication checkpoints.")
    raise
