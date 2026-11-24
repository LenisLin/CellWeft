"""Deliver published Cords 2024 lung IMC cores as native float32 CYX TIFFs.

Inputs: complete local SPORA Zarrs, accepted native verification, S1 and the
original Zenodo acquisition/core/clinical metadata cached for this study.
Outputs: input.tiff, all 39 image fields, sample/panel/field tables, preparer
checks and resumable per-image receipts. Independent D01 review is separate.
Run --export with the batch shared formal write lease wrapper, --verify with read.
Run --repair-clinical to repair reviewed clinical semantics, reusing all TIFFs.
The largest native ROI is <158 MB; decode each all-channel shard once in RAM.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
import io
import json
import os
import shutil
import struct
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import requests
import tifffile
import zarr
from PIL import Image
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.inputs import input_binding
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)
STUDY = "Cords2024_IMC_Lung"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
RAW = NAS / "raw/public" / STUDY / "spora"
OUTPUT = NAS / "processed" / STUDY
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
CP = META / "source_metadata" / "cp_metadata/cp_csv"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
SNAPSHOT = META / "file_evidence.json"
ACCEPTED = META / "accepted_images.json"
OBJECTS = META / "imc_source_objects.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0025_prepare_0/progress.json"
PANEL = "spora_IMC_43"
PREPARER = "0025_prepare_0"
REPAIR_JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0178_repair_1"
REPAIRER = "0178_repair_1"
CLINICAL_TIME_RULE = (
    "Clinical index event and staging edition/basis are not established. "
    "Index-time TNM, smoking, stage, age and survival targets remain null; "
    "paper smoking is at diagnosis. Source codes and author-decoded attributes "
    "are retained only as source clinical attributes, without an assigned index."
)
GAPS = ["91张控制core缺少供者/取样分组", "8份采样normal/metastase标注的组织/病灶解释缺口",
        "临床年龄/生存时间语义及一个未定义N编码", "待独立复核与D01绑定"]


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def clinical_context_notes(raw: dict, notes: dict) -> dict:
    """Preserve author codes/decoding without assigning an unsupported index."""
    notes["source_clinical_attributes"] = dict(
        tnm_t=dict(zip(map(str, range(1, 7)), ["T1a", "T1b", "T2a", "T2b", "T3", "T4"])).get(raw["T.new"]),
        tnm_n={"0": "N0", "1": "N1", "2": "N2", "3": "N3"}.get(raw["N"]),
        tnm_m={"0": "M0", "1": "M1a", "2": "M1b"}.get(raw["M.new"]),
        smoking_status={"0": "never", "1": "current", "2": "former", "3": "unknown"}.get(raw["Smok"]),
        smoking_timepoint="at diagnosis, per paper cohort description",
        index_event=None, stage_system=None, stage_basis=None,
    )
    notes["time_semantics"] = CLINICAL_TIME_RULE
    notes["grade_context"] = (
        "Author clinical Grade 1/2/3 retained verbatim; grading system/version "
        "and index time are not specified. Cross-study grade equivalence is unresolved."
    )
    return notes


def clinical_context_fields(fields: pd.DataFrame) -> pd.DataFrame:
    """Apply the same clinical interpretation to initial and repaired delivery."""
    rules = {
        "tnm_t": "T.new author decoding retained in metadata_notes.source_clinical_attributes.tnm_t; target null until index and staging edition/basis established",
        "tnm_n": "N author decoding retained in metadata_notes.source_clinical_attributes.tnm_n; undefined N=5 stays null; target null until index and staging edition/basis established",
        "tnm_m": "M.new author decoding retained in metadata_notes.source_clinical_attributes.tnm_m; target null until index and staging edition/basis established",
        "smoking_status": "Smok author decoding retained in metadata_notes.source_clinical_attributes.smoking_status at diagnosis; target null because diagnosis has not been defined as the clinical index",
        "index_event": "Null; no study clinical index defined; diagnosis-time source variables do not assign an index to other variables",
        "stage_system": "Null; exact staging system/version not established by consulted sources",
        "stage_basis": "Null; clinical/pathological staging basis not established by consulted sources",
        "grade": "Verbatim author Grade 1..3; NA -> null; grading system/version and index context unresolved, explicitly recorded in metadata_notes.grade_context",
        "metadata_notes": "Preserve clinical_raw and later growth label, source_stage_group and source_clinical_attributes; index-dependent targets null; grade context and other source gaps explicit",
    }
    fields = fields.copy()
    for target, rule in rules.items():
        mask = fields.target_table.eq("sample_metadata") & fields.target_field.eq(target)
        assert mask.sum() == 1, target
        fields.loc[mask, "transformation_rule"] = rule
        fields.loc[mask, "time_origin_rule"] = CLINICAL_TIME_RULE
        fields.loc[mask, "notes"] = json.dumps(dict(
            source_paper=relative(META / "source_metadata/paper.xml"),
            author_decoding=relative(next((META / "source_metadata/author_code").iterdir()) / "00_prepare_clinicaldata.Rmd"),
            limitation="Index context not assigned; grade system unresolved; original clinical values retained",
        ))
    return fields


def progress(phase: str, completed: int, **details: object) -> None:
    details["gaps"] = list(dict.fromkeys(GAPS + list(details.get("gaps", []))))
    write_json(PROGRESS, dict(study=STUDY, phase=phase, completed_images=completed,
                             expected_images=2040, updated_at=datetime.now(timezone.utc).isoformat(),
                             **details))


def fetch_metadata() -> None:
    """Confirm the already received source metadata without a network request."""
    assert CP.is_dir(), "Missing received clinical metadata"


def source_inventory() -> tuple[dict, dict]:
    """Reuse accepted image/order proofs, matching their current size/mtime receipts."""
    evidence = json.loads(SNAPSHOT.read_text())
    accepted = json.loads(ACCEPTED.read_text())
    assert set(accepted["accepted_image_ids"]) == set(evidence["images"])
    assert len(evidence["images"]) == 2040
    for name, expected in evidence["context"]:
        stat = (NAS / name).stat()
        assert [stat.st_size, stat.st_mtime_ns] == expected, name
    objects: dict[str, list[dict]] = {}
    for entry in json.loads(OBJECTS.read_text()):
        suffix = entry["key"].split("/imc/1_0mpp/images/", 1)[1]
        image_id = suffix.split(".ome.zarr/", 1)[0]
        objects.setdefault(image_id, []).append(entry)
    actual = {p.name.removesuffix(".ome.zarr") for p in (RAW / "imc/1_0mpp/images").iterdir()
              if p.is_dir() and p.name.endswith(".ome.zarr")}
    assert actual == set(evidence["images"])
    for image_id, item in evidence["images"].items():
        files = sorted(objects[image_id], key=lambda value: value["key"])
        assert len(files) == len(item["file_stats"])
        for entry, expected in zip(files, item["file_stats"], strict=True):
            suffix = entry["key"].split("spora-submissions/cords2024cancer/", 1)[1]
            stat = (RAW / suffix).stat()
            assert stat.st_size == entry["size"]
            assert [stat.st_size, stat.st_mtime_ns] == expected, (image_id, suffix)
    shapes = np.array([v["record"]["shape"] for v in evidence["images"].values()])
    estimate = int((shapes[:, 1:].max(axis=1) ** 2 * 42 * 4).sum())
    write_json(CACHE / "source_scope.json", dict(
        study=STUDY, accepted_images=len(actual), source_bytes=sum(e["size"] for es in objects.values() for e in es),
        uncompressed_square_output_bytes=estimate, largest_source_bytes=int(np.prod(shapes, axis=1).max() * 4),
        accepted_proof=relative(ACCEPTED), source_snapshot=relative(SNAPSHOT),
        source_inventory=relative(OBJECTS), verification="Existing native proof reused; current files size/mtime matched; no rehash",
        all_local_complete_images_included=True, source_modified=False))
    return evidence["images"], objects


def metadata(evidence: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe complete source frames; sample identity is joined separately below."""
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    tissues = tissues[tissues.modality == "imc"].copy()
    assert set(tissues.tissue_id) == set(evidence) and tissues.tissue_id.is_unique
    channels = pd.read_parquet(RAW / "imc/channels.parquet")
    available = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
    assert available.columns.tolist() == channels.channel_name.tolist()
    assert set(available.index) == set(tissues.tissue_id) and available.to_numpy().all()
    panel = panel_from_s1(STUDY, PANEL, channels.channel_name.tolist(), S1)
    panel["source_original_channel_name"] = channels.original_channel_name
    panel["source_description"] = channels.description
    panel["source_embedded_label"] = next(iter(evidence.values()))["record"]["embedded_labels"]
    panel["display_name"] = panel.marker_id
    rows = []
    for source in tissues.itertuples(index=False):
        proof = evidence[source.tissue_id]["record"]
        assert proof["canonical_channel_labels"] == channels.channel_name.tolist()
        assert proof["exact_canonical_order_checked"] and proof["dtype"] == "float32"
        nc, h, w = proof["shape"]
        side = max(h, w)
        root = NAS / proof["path"]
        group = json.loads((root / "zarr.json").read_text())["attributes"]["ome"]
        assert [x["label"] for x in group["omero"]["channels"]] == proof["embedded_labels"]
        assert proof["embedded_labels"] == panel.source_embedded_label.tolist()
        scale = group["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        assert scale["type"] == "scale" and scale["scale"] == [1.0, 1.0, 1.0]
        row = dict.fromkeys(columns("image_metadata"), None)
        row.update(study=STUDY, image_id=source.tissue_id, panel_id=PANEL,
                   image_kind="tma_core", source_image_id=source.tissue_id,
                   source_files=[relative(root), relative(RAW / "imc/channels.parquet")],
                   source_format="ome-zarr", source_axes="CYX", source_dtype="float32",
                   source_revision=relative(SNAPSHOT), core_id=source.original_file_name,
                   region_annotation=source.original_file_name,
                   source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                   pad_left_px=(side-w)//2, pad_right_px=side-w-(side-w)//2,
                   pad_top_px=(side-h)//2, pad_bottom_px=side-h-(side-h)//2,
                   height_px=side, width_px=side, source_channel_count=nc, channel_count=int(panel.keep_channel.sum()),
                   pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                   input_image_path=relative(OUTPUT / "images" / source.tissue_id / "input.tiff"),
                   intensity_description="Unchanged published SPORA float32 intensity; no normalization",
                   split_method="already_independent", split_evidence="One original TMA core per published complete ROI; original image_name retained",
                   metadata_notes="SPORA specimen_id enumerates ROIs; biological sampling requires original clinical/core map. "
                                  "Published rectangular raster retained, including measured zeros.")
        rows.append(row)
    images = pd.DataFrame(rows)
    samples = pd.DataFrame(columns=columns("sample_metadata"))
    fields = []
    for target in columns("image_metadata"):
        fields.append(dict(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
                           source_sheet_or_table="tissues + accepted image proofs + Zarr metadata + S1",
                           source_field=target, target_table="image_metadata", target_field=target,
                           join_keys="tissues.tissue_id=accepted.image_id=Zarr basename", value_mapping="{}",
                           transformation_rule="Exact source identity/full native CYX frame; centered square zero-padding; S1 source-order selection",
                           time_origin_rule="No time or clinical event semantics inferred",
                           conflict_rule="Require one-to-one image identity and exact accepted source order",
                           notes="Preparer source interpretation; sample linkage updated from original author metadata"))
    return images, samples, panel, pd.DataFrame(fields, columns=columns("field_mapping"))


def attach_clinical(images: pd.DataFrame, fields: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join explicit author RoiID -> primary-tumor sampling -> Patient_ID.

    The paper states paired cores are from the same tumor invasive front.
    Controls have only a pooled 'Control' placeholder, not donor/sample IDs.
    """
    source_path = CP / "clinical_data_ROI.csv"
    clinical = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    code_root = next((META / "source_metadata/author_code").iterdir())
    author = (code_root / "00_prepare_clinicaldata.Rmd").read_text()
    combine = (code_root / "combine_ROI_clinical-data.Rmd").read_text()
    assert 'clinical.data$Patient_ID <- paste(clinical.data$TMA, clinical.data$Patient_Nr,sep="_")' in author
    assert 'ac_clinical$Patient_ID[is.na(ac_clinical$Patient_ID)] <-"Control"' in combine
    paper = " ".join(ET.parse(META / "source_metadata/paper.xml").getroot().itertext())
    assert "all cores were sampled from the tumor invasive front" in paper
    assert "Samples were first deparaffinised using HistoClear" in paper
    # SPORA inserts a comma in one author control label; explicit TMA/acID join.
    typo = clinical.RoiID.eq("87_C613")
    assert typo.sum() == 1
    assert clinical.loc[typo, "Tma_ac"].eq("87C_98").all()
    acquisition = pd.read_csv(CP / "87_C_acquisition_metadata.csv", dtype=str, keep_default_na=False)
    acquisition = acquisition[acquisition.id.eq("98")].iloc[0]
    assert acquisition.description == "C613"
    image = images[images.core_id.eq("87_C6,13")].iloc[0]
    assert [image.source_y1_px, image.source_x1_px] == [int(acquisition.max_y), int(acquisition.max_x)]
    clinical["join_core"] = clinical.RoiID.mask(typo, "87_C6,13")
    # The two twice-acquired physical cores are outside this SPORA image scope.
    repeated = clinical[clinical.join_core.duplicated(False)]
    assert set(repeated.join_core).isdisjoint(set(images.core_id))
    clinical = clinical.drop_duplicates("join_core")
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    tissues = tissues[tissues.modality == "imc"]
    joined = tissues[["tissue_id", "patient_id", "original_patient_identifier", "original_file_name", "acID"]].rename(columns={"acID": "spora_acID"}).merge(
        clinical, left_on="original_file_name", right_on="join_core", how="left", validate="one_to_one", indicator=True)
    assert joined._merge.eq("both").all() and len(joined) == len(images)
    assert joined.original_patient_identifier.eq(joined.Patient_ID).all()
    assert joined.spora_acID.eq(pd.to_numeric(joined.acID)).all()
    growth = pd.read_csv(META / "source_metadata/v2_metadata/Growth patterns.csv", dtype=str, keep_default_na=False)
    # Same explicit TMA/Patient_Nr construction as the original author code;
    # matching the sub-block independently checks the later table's key.
    growth["Patient_ID"] = growth.ZTMA + "_" + growth["Nr."]
    assert growth.Patient_ID.is_unique
    joined = joined.drop(columns="_merge").merge(growth, on="Patient_ID", how="left", validate="many_to_one", indicator=True)
    controls = joined.Patient_ID.eq("Control")
    cases = joined[~controls]
    assert cases._merge.eq("both").all() and cases["TMA.x"].eq(cases.ZTMA + cases.Column).all()
    assert len(cases) == 1949 and cases.Patient_ID.nunique() == 1068 and controls.sum() == 91
    assert cases.groupby("Patient_ID").patient_id.nunique().eq(1).all()
    assert cases.groupby("patient_id").Patient_ID.nunique().eq(1).all()
    sample_rows = []
    raw_fields = ["Age", "Gender", "DX.name", "Grade", "Size", "Vessel", "Pleura", "T.new", "N", "M.new", "Stage",
                  "R", "Chemo", "Radio", "Chemo3", "Radio4", "Relapse", "Chemo5", "Radio6", "DFS", "Ev.O", "OS", "Smok", "Nikotin", "NeoAdj"]
    sex = {"1": "male", "2": "female"}
    stage = dict(zip(map(str, range(1, 8)), ["IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IV"]))
    for source_id, group in cases.groupby("Patient_ID", sort=True):
        assert group[raw_fields].nunique(dropna=False).le(1).all()
        source = group.iloc[0]
        row = dict.fromkeys(columns("sample_metadata"), None)
        row.update(study=STUDY, sample_id="sample_" + source_id,
                   patient_key=STUDY + ":" + source.patient_id, source_subject_id=source_id,
                   biospecimen_group_id=source_id,
                   cohort_id="cords2024cancer", center_id="USZ",
                   sample_type="lung-cohort tissue sampling, one or two paired TMA cores",
                   processing_method="TMA cores; HistoClear deparaffinisation, antigen retrieval and metal-conjugated antibody staining for IMC",
                   anatomical_site=None if source.GrowthPattern == "metastase" or source["DX.name"] == "Mesotheliom" else "lung",
                   primary_site=None if source["DX.name"] == "Mesotheliom" else "lung",
                   tissue_type=None if source.GrowthPattern == "normal" else "tumor",
                   lesion_type=None if source.GrowthPattern in {"normal", "metastase"} else "primary",
                   case_control_status="case", sex_reported=sex.get(source.Gender),
                   primary_diagnosis=None if source["DX.name"] == "NA" else source["DX.name"],
                   histology=None if source["DX.name"] == "NA" or source.GrowthPattern == "normal" else source["DX.name"],
                   grade=None if source.Grade == "NA" else source.Grade,
                   treatment_status_at_collection={"NoNeoAdjuvantTherapy": "no neoadjuvant therapy", "NeoAdjuvantTherapy": "received neoadjuvant therapy"}.get(source.NeoAdj),
                   vital_status={"0": "alive", "1": "deceased"}.get(source["Ev.O"]),
                   metadata_notes=json.dumps(clinical_context_notes({key: source[key] for key in raw_fields}, dict(
                       clinical_raw={key: source[key] for key in raw_fields},
                       source_growth_pattern=source.GrowthPattern,
                       source_stage_group=stage.get(source.Stage),
                       growth_pattern_source=relative(META / "source_metadata/v2_metadata/Growth patterns.csv"),
                       sample_evidence="Author clinical/core map + paired-core design + later growth table: one source patient and TMA sub-block define one paired-core sampling; native specimen ID not published",
                       growth_typing_gap="Later GrowthPattern is normal/metastase; conflicts with blanket primary-tumor cohort description. Related tissue/lesion/site/histology fields left null, raw annotation preserved" if source.GrowthPattern in {"normal", "metastase"} else "",
                       site_typing_gap="Clinical DX.name is Mesotheliom; exact specimen/primary site not published, so site fields null" if source["DX.name"] == "Mesotheliom" else "",
                       unmapped_code="N=5 is not defined in author code; tnm_n left null" if source.N == "5" else "",
                       missingness="NA retained in clinical_raw; target unknown values null")), ensure_ascii=False))
        sample_rows.append(row)
    samples = pd.DataFrame(sample_rows, columns=columns("sample_metadata"))
    lookup = joined.set_index("tissue_id")
    images = images.copy()
    images["sample_id"] = [None if lookup.loc[key, "Patient_ID"] == "Control" else "sample_" + lookup.loc[key, "Patient_ID"] for key in images.image_id]
    images["source_acquisition_id"] = images.image_id.map(lookup.acID)
    images["slide_id"] = images.image_id.map(lookup["TMA.x"])
    images["roi_id"] = images.image_id.map(lookup.ROI).replace("NA", None)
    images["region_annotation"] = [json.dumps({"ROI": lookup.loc[key, "ROI"], "DX.name": lookup.loc[key, "DX.name"],
                                              "GrowthPattern": None if lookup.loc[key, "Patient_ID"] == "Control" else lookup.loc[key, "GrowthPattern"]}, ensure_ascii=False) for key in images.image_id]
    images["metadata_notes"] = ["Control donor and biological sampling identity unavailable; pooled SPORA patient_id is not a patient; sample_id null" if lookup.loc[key, "Patient_ID"] == "Control" else "Paired-core sampling group from author core map and cohort design; native specimen ID absent; original clinical codes and later growth labels retained in sample metadata_notes" for key in images.image_id]
    rules = {
        "study": ("dataset", "Canonical CellWeft study for cords2024cancer"),
        "image_id": ("tissue_id", "Verbatim SPORA independent image ID"),
        "sample_id": ("RoiID;Patient_ID;paper paired-core design;Growth patterns", "sample_ + author Patient_ID for the explicit paired-core sampling group; Control remains null"),
        "panel_id": ("channels;channels_per_tissue;accepted canonical labels", "Same exact measured 43-channel order for all images -> spora_IMC_43"),
        "image_kind": ("paper TMA acquisition;RoiID", "Independent physical TMA core -> tma_core"),
        "source_image_id": ("tissue_id", "Verbatim SPORA tissue ID"),
        "source_files": ("accepted native path;channels.parquet", "NAS-relative original Zarr root and ordered channel table as list"),
        "source_format": ("Zarr zarr_format", "OME-Zarr v3 -> ome-zarr"),
        "source_acquisition_id": ("acID;spora.acID", "Exact author acquisition ID; numeric agreement checked"),
        "source_z_index": ("dimension_names", "No Z dimension -> null"),
        "source_t_index": ("dimension_names", "No time dimension -> null"),
        "source_axes": ("dimension_names", "c,y,x -> CYX; unchanged orientation"),
        "source_dtype": ("accepted dtype;Zarr data_type", "Native float32, unchanged"),
        "source_revision": ("reviewed_file_evidence.json", "Accepted complete source proof with unchanged size and mtime_ns"),
        "slide_id": ("TMA.x", "Verbatim author TMA slide identity, including A/B/C sub-block"),
        "core_id": ("original_file_name;RoiID;Tma_ac;acquisition dimensions", "Verbatim SPORA core ID; 87_C6,13 corresponds to author 87_C613 via explicit TMA=87C, acID=98 and native H/W"),
        "roi_id": ("ROI", "Verbatim within-TMA core coordinate; source NA -> null; full SPORA label retained in core_id"),
        "fov_id": ("source core layout", "No separate FOV identity -> null"),
        "region_annotation": ("ROI;DX.name;GrowthPattern", "Original per-core labels plus later explicit patient/TMA-linked growth label as JSON; NA string retained as source missing code"),
        "valid_pixels_path": ("complete published rectangular raster", "No unmeasured interior declared by SPORA; null; no pixelwise zero filtering"),
        "intensity_description": ("published float32 values", "Native published intensity with no rescaling or normalization"),
        "split_method": ("one independent published core", "already_independent"),
        "split_evidence": ("paper TMA design;RoiID;accepted source shape", "Full native source ROI; paired acquisitions outside scope; no sample split"),
        "metadata_notes": ("Patient_ID;author code Control handling", "Control identity gap vs case primary-tumor sampling evidence"),
        "input_image_path": ("tissue_id", "processed/<study>/images/<tissue_id>/input.tiff; NAS-relative"),
        "source_channel_count": ("accepted shape[0];channels", "43 exact source channels"),
        "channel_count": ("S1;panel.keep_channel", "42 retained source-order channels; PNAd excluded; both nuclear Iridium channels retained"),
    }
    for key in ["source_x0_px", "source_y0_px"]:
        rules[key] = ("full native ROI", "0")
    for key, axis in [("source_x1_px", 2), ("source_y1_px", 1)]:
        rules[key] = (f"accepted shape[{axis}]", "Exclusive native full-ROI bound")
    for key in ["height_px", "width_px"]:
        rules[key] = ("accepted H,W", "max(H,W)")
    for key in ["pad_left_px", "pad_right_px", "pad_top_px", "pad_bottom_px"]:
        rules[key] = ("accepted H,W", "Centered square padding: left/top=floor((L-W/H)/2); right/bottom=remainder")
    for key in ["pixel_size_x_um", "pixel_size_y_um"]:
        rules[key] = ("OME scale + paper Hyperion resolution", "1 micrometer; no resampling")
    assert set(rules) == set(columns("image_metadata"))
    for target, (source, rule) in rules.items():
        mask = fields.target_table.eq("image_metadata") & fields.target_field.eq(target)
        fields.loc[mask, "source_field"] = source
        fields.loc[mask, "transformation_rule"] = rule
        evidence_path = RAW / "metadata/tissues.parquet"
        if target in {"sample_id", "source_acquisition_id", "slide_id", "core_id", "roi_id", "region_annotation", "metadata_notes"}:
            evidence_path = source_path
        elif target in {"source_format", "source_axes", "source_dtype", "source_revision", "source_channel_count", "source_z_index", "source_t_index", "intensity_description", "valid_pixels_path"} or target.startswith(("pad_", "source_x", "source_y")) or target in {"height_px", "width_px"}:
            evidence_path = SNAPSHOT
        elif target in {"channel_count"}:
            evidence_path = S1
        elif target == "panel_id":
            evidence_path = RAW / "imc/channels.parquet"
        elif target in {"pixel_size_x_um", "pixel_size_y_um", "image_kind", "split_method", "split_evidence"}:
            evidence_path = META / "source_metadata/paper.xml"
        fields.loc[mask, "source_file"] = relative(evidence_path)
        fields.loc[mask, "notes"] = "Additional exact source paths: source_scope.json; sample_linkage.json; metadata_sources.json in study cache"
    mapping_rows = []
    mappings = {"study": ("dataset", "Canonical study"), "sample_id": ("Patient_ID", "sample_ prefix; cases only"),
                "patient_key": ("Patient_ID;spora.patient_id", "Explicit clinical Patient_ID joined to SPORA pseudonym, prefixed with study"),
                "source_subject_id": ("Patient_ID", "Verbatim author identity"), "source_sample_id": ("native specimen identifier", "Not published; null. sample_id denotes the source-supported paired-core sampling group"),
                "biospecimen_group_id": ("Patient_ID;TMA.x;Growth patterns ZTMA/Column/Nr.", "One author patient and TMA sub-block identify one paired-core sampling"),
                "cohort_id": ("dataset", "cords2024cancer"), "center_id": ("paper cohort", "USZ"),
                "sample_type": ("paper TMA design;author paired-core map", "One lung-cohort sampling with one or two paired cores"),
                "processing_method": ("paper Antibody panel, staining, and imaging", "Published HistoClear deparaffinisation, antigen retrieval and IMC antibody staining; fixation not inferred"),
                "anatomical_site": ("paper cohort;GrowthPattern;DX.name", "lung except metastase or Mesotheliom label, whose exact specimen site is unresolved -> null"), "primary_site": ("paper cohort;DX.name", "lung; Mesotheliom exact primary site unresolved -> null"),
                "tissue_type": ("paper core design;GrowthPattern", "tumor except later normal label -> null pending tissue-type interpretation"), "lesion_type": ("paper primary cancers;GrowthPattern", "primary except normal/metastase labels -> null pending interpretation"),
                "case_control_status": ("Patient_ID", "Non-Control clinical subject -> case"),
                "sex_reported": ("Gender;author code", json.dumps(sex)), "smoking_status": ("Smok;author code", "Index-time target null; source diagnosis-time decoding retained in metadata_notes"),
                "primary_diagnosis": ("DX.name", "Verbatim clinical diagnosis; NA -> null"), "histology": ("DX.name;GrowthPattern", "Verbatim; NA or later normal flag -> null"),
                "stage_group_at_index": ("Stage;author code", "Clinical index event not defined -> null; source_stage_group in metadata_notes stores the author-decoded stage: " + json.dumps(stage)), "tnm_t": ("T.new;author code", "Index-time target null; source decoding retained in metadata_notes"),
                "tnm_n": ("N;author code", "Index-time target null; source decoding/undefined N=5 retained in metadata_notes"),
                "tnm_m": ("M.new;author code", "Index-time target null; source decoding retained in metadata_notes"), "grade": ("Grade", "Verbatim 1..3; NA -> null"),
                "treatment_status_at_collection": ("NeoAdj;author code", "Preserve explicit neoadjuvant status; no post-relapse/adjuvant mixing"),
                "vital_status": ("Ev.O;author code", "0 -> alive; 1 -> deceased; NA -> null"),
                "metadata_notes": ("original clinical row;Growth patterns;author code;paper", "Preserve all raw clinical fields and later growth label; age/survival unit/time origin unknown -> target null; undefined N=5 preserved; growth typing conflicts explicit")}
    for target in columns("sample_metadata"):
        source, rule = mappings.get(target, ("not available or semantics unresolved", "Null; no default or inferred time/therapy coding"))
        mapping_rows.append(dict(study=STUDY, source_file=relative(META / "source_metadata/paper.xml" if target == "processing_method" else source_path), source_sheet_or_table="clinical_data_ROI.csv + author code v1.0 + paper",
                                 source_field=source, target_table="sample_metadata", target_field=target,
                                 join_keys="spora.original_file_name=clinical.RoiID; clinical.Patient_ID=TMA+Patient_Nr; paired tumor cores share sampling",
                                 value_mapping="{}", transformation_rule=rule, time_origin_rule="Do not infer clinical age/survival time origin or unit",
                                 conflict_rule="Require within-sampling consistency; Control excluded from patient/sample grouping", notes=relative(code_root)))
    fields = pd.concat([fields, pd.DataFrame(mapping_rows)], ignore_index=True)
    panel_rules = []
    for target, source, rule, source_file in [
        ("source_name", "channel_name;original_channel_name;description", "Preserve all 43 source rows in measured order; canonical descriptions recover three cocktail labels abbreviated in embedded metadata", RAW / "imc/channels.parquet"),
        ("marker_id", "Study;Original marker;Unified marker;Segmentation channel", "Exact Cords2024_IMC_Lung + source label join to S1; no cross-study aliases; retain both specifically named Iridium nuclear inputs", S1),
        ("keep_channel", "Unified marker;Segmentation channel", "42/43 kept; exclude PNAd only; other admitted markers and both selected nuclear channels retained; output_channel_index assigned consecutively in source order", S1),
    ]:
        panel_rules.append(dict(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_file.name,
                                source_field=source, target_table="panel_mapping", target_field=target,
                                join_keys="S1.Study=Cords2024_IMC_Lung and S1.Original marker=channels.channel_name",
                                value_mapping="{}", transformation_rule=rule, time_origin_rule="not applicable",
                                conflict_rule="Exact source order and single downstream measurement per unified marker", notes=""))
    fields = pd.concat([fields, pd.DataFrame(panel_rules)], ignore_index=True)
    write_json(CACHE / "sample_linkage.json", dict(
        expected_images=len(images), source_case_images=len(cases), sample_rows=len(samples), patient_linked_samples=len(samples),
        control_images=int(controls.sum()), control_image_ids=joined.loc[controls, "tissue_id"].tolist(),
        control_gap="Source Patient_ID=Control is an author-assigned placeholder for missing clinical linkage. Donor/specimen grouping unavailable; sample_id remains null.",
        join_coverage=2040, typo_rule="SPORA 87_C6,13 corresponds to clinical 87_C613 and original acquisition description C613; explicit Tma_ac=87C_98 and native H/W agree; source retained",
        double_acquired_cores="Two original duplicate RoiIDs are not in the current SPORA scope",
        source_files=[relative(source_path), relative(code_root / "00_prepare_clinicaldata.Rmd"), relative(code_root / "combine_ROI_clinical-data.Rmd"), relative(META / "source_metadata/paper.xml"), relative(META / "source_metadata/v2_metadata/Growth patterns.csv")],
        later_growth_flagged_sampling_groups=cases[cases.GrowthPattern.isin(["normal", "metastase"])].Patient_ID.nunique(),
        later_growth_flagged_image_ids=cases.loc[cases.GrowthPattern.isin(["normal", "metastase"]), "tissue_id"].tolist(),
        metadata_gaps=["91 control cores lack donor and biological sample grouping", "Exact age/survival unit and time origin not established", "One case has author-undefined N=5; target tnm_n null", "Later normal/metastase growth flags require tissue/lesion/site interpretation; related fields null"] ))
    return images, samples, clinical_context_fields(fields)


class NativeROI:
    """Bounded in-memory ROI adapter avoids repeated decoding of all-channel shards.

    All published cores are smaller than 158 MB (not whole-slide images).
    Pixel validity follows the published full rectangular raster, not signal.
    """
    def __init__(self, root: Path, image_id: str):
        array = zarr.open_array(str(root / "0"), mode="r")
        self.values = np.asarray(array[:])
        self.shape, self.dtype = self.values.shape, self.values.dtype
        self.source = SourceSpec(image_id, PANEL, "zarr", root / "0", tuple(range(self.shape[0])),
                                 pixel_size_um=(1.0, 1.0), source_revision=relative(SNAPSHOT))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def overview(reader: NativeROI, image_id: str, panel: pd.DataFrame) -> None:
    """Display nuclear, epithelial and stromal source signals over the entire ROI."""
    names = panel.source_name.tolist()
    composite = []
    for name in ["panCK", "Iridium (V2)", "VIM"]:
        plane = reader.values[names.index(name)]
        nonzero = plane[plane > 0]
        upper = np.percentile(nonzero, 99) if len(nonzero) else 1
        composite.append(np.minimum(np.maximum(plane, 0) / max(upper, 1e-9), 1))
    rgb = np.uint8(np.stack(composite, axis=-1) * 255)
    image = Image.fromarray(rgb)
    image.thumbnail((480, 480))
    image.save(CACHE / f"overview_{image_id}.png")


def preparer_checks(images: pd.DataFrame, samples: pd.DataFrame, panel: pd.DataFrame, phase: str) -> None:
    """Write only preparer observations; independent review and D01 stay unresolved."""
    checks = []
    for check_id, scope in [("S01", "study"), ("I01", "image"), ("I02", "image"),
                            ("I03", "image"), ("I04", "image"), ("P01", "panel"),
                            ("P02", "panel"), ("P03", "image"), ("I05", "image"),
                            ("M01", "sample"), ("M02", "sample"), ("D01", "study")]:
        checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                           record_id=PANEL if scope == "panel" else STUDY, check_id=check_id,
                           result="unresolved", source_evidence=relative(CACHE / "source_scope.json"),
                           output_evidence=relative(OUTPUT), details=json.dumps(dict(
                               phase=phase, preparer_only=True, exported_images=len(images),
                               processed_source_images=len(receipts),
                               excluded_all_marker_zero=[key for key, value in receipts.items() if not value["admitted"]],
                               exclusion_rule="Only exclude when every retained downstream marker is zero over the entire source ROI",
                               sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                               patient_linked_samples=int(samples.patient_key.notna().sum()),
                               metadata_evidence=relative(CACHE / "sample_linkage.json"),
                               later_growth_typing_gap="8 paired-core sampling groups have normal/metastase labels; related clinical typing fields null; identity and pixels retained",
                               join_keys="SPORA original_file_name -> author RoiID (one explicit TMA/acID label adaptation) -> author Patient_ID -> primary tumor sampling; explicit SPORA pseudonym connection",
                               consulted_sources=[relative(CP / "clinical_data_ROI.csv"), relative(META / "source_metadata/author_code"), relative(META / "source_metadata/paper.xml"), relative(CP / "87_C_acquisition_metadata.csv"), relative(META / "source_metadata/v2_metadata"), relative(CACHE / "source_followup.json")],
                               metadata_gap="91 control core donor/sampling identities unavailable; clinical time semantics unresolved",
                               verification=relative(CACHE / "preparer_verification.json") if (CACHE / "preparer_verification.json").exists() else "pending",
                               reason="Preparer self-check; pending independent review and D01 binding"), ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(checks, columns=columns("review_results")).to_csv(
            review_path, index=False, encoding="utf-8-sig")


def verify_outputs(images: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Reopen every actual TIFF, plus source/output blocks in each padding layout."""
    selected = panel[panel.keep_channel].sort_values("output_channel_index")
    expected_names = selected.marker_id.tolist()
    header_count = 0
    progress("verifying_tiff_headers", len(images), verified_tiff_headers=0)
    for row in images.itertuples(index=False):
        with tifffile.TiffFile(NAS / row.input_image_path) as tif:
            ome = ET.fromstring(tif.ome_metadata)
            names = [node.get("Name") for node in ome.iter() if node.tag.endswith("}Channel")]
            pixels = next(node for node in ome.iter() if node.tag.endswith("}Pixels"))
            assert float(pixels.get("PhysicalSizeX")) == row.pixel_size_x_um
            assert float(pixels.get("PhysicalSizeY")) == row.pixel_size_y_um
            assert pixels.get("PhysicalSizeXUnit") == pixels.get("PhysicalSizeYUnit") == "µm"
            assert len(tif.pages) == row.channel_count and names == expected_names
            assert tif.series[0].axes == "CYX"
            assert tif.series[0].shape == (row.channel_count, row.height_px, row.width_px)
            assert all(page.dtype == np.dtype(row.source_dtype) and page.shape == (row.height_px, row.width_px) for page in tif.pages)
            header_count += 1
            if header_count % 100 == 0 or header_count == len(images):
                progress("verifying_tiff_headers", len(images), verified_tiff_headers=header_count)
    sample_ids = []
    for subset in [images[images.pad_left_px > 0], images[images.pad_top_px > 0],
                   images[images.sample_id.isna()], images.sort_values("height_px"), images.sort_values("height_px", ascending=False)]:
        if not subset.empty:
            sample_ids.append(subset.iloc[0].image_id)
    checked = []
    for image_id in dict.fromkeys(sample_ids):
        progress("verifying_source_output_pixels", len(images), verified_tiff_headers=header_count,
                 pixel_samples_completed=len(checked), current_image=image_id)
        row = images.set_index("image_id").loc[image_id]
        reader = NativeROI(NAS / evidence[image_id]["record"]["path"], image_id)
        if pd.isna(row.sample_id):
            overview(reader, image_id, panel)
        _, h, w = reader.shape
        blocks = [(max(0, w//2-32), max(0, h//2-32), min(w, w//2+32), min(h, h//2+32)),
                  (0, 0, min(w, 64), min(h, 64))]
        with tifffile.TiffFile(NAS / row.input_image_path) as tif:
            for window in blocks:
                x0, y0, x1, y1 = window
                target = (x0+int(row.pad_left_px), y0+int(row.pad_top_px), x1+int(row.pad_left_px), y1+int(row.pad_top_px))
                for channel in selected.itertuples(index=False):
                    actual = read_tiff_window(tif, int(channel.output_channel_index), target)
                    wanted = reader.values[channel.source_channel_index, y0:y1, x0:x1]
                    assert actual.dtype == wanted.dtype and np.array_equal(actual, wanted)
            # Full four padding margins, in every retained channel.
            margins = [(0, 0, int(row.pad_left_px), row.height_px),
                       (row.width_px-int(row.pad_right_px), 0, row.width_px, row.height_px),
                       (0, 0, row.width_px, int(row.pad_top_px)),
                       (0, row.height_px-int(row.pad_bottom_px), row.width_px, row.height_px)]
            for margin in margins:
                if margin[2] > margin[0] and margin[3] > margin[1]:
                    for channel in selected.output_channel_index:
                        assert not read_tiff_window(tif, int(channel), margin).any()
        del reader
        checked.append(dict(image_id=image_id, source_xyxy_blocks=blocks, channels=len(selected), padding="all margins, all retained channels"))
    write_json(CACHE / "preparer_verification.json", dict(
        study=STUDY, preparer=PREPARER, independent_review=False,
        tiff_headers_checked=header_count, pixel_samples=checked,
        result="pass", limitations="Actual preparer checks only; independent reviewer and D01 binding pending"))


def repair_clinical() -> None:
    """Repair M02 target semantics only and prepare, never approve, a new D01.

    Existing independent technical checks and control identity gaps are retained.
    Source clinical rows are compared with every retained raw value before edits.
    The study lease covers metadata replacement and D01 revocation together.
    """
    progress("checking_clinical_repair_sources", 2040, repairer=REPAIRER)
    review_job = REPAIR_JOB.parent / "0174_review_0"
    prior_report = json.loads((review_job / "report.json").read_text())
    assert prior_report["exported_images"] == prior_report["expected_images"] == 2040
    with nullcontext():
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
        reviews = pd.read_csv(OUTPUT / "review_results.csv", dtype=str, keep_default_na=False)
        before_samples, before_fields = samples.copy(), fields.copy()
        clinical = pd.read_csv(CP / "clinical_data_ROI.csv", dtype=str, keep_default_na=False)
        assert len(images) == 2040 and len(samples) == 1068
        assert images.columns.tolist() == columns("image_metadata")
        assert samples.columns.tolist() == columns("sample_metadata")
        assert samples.sample_id.is_unique and samples.patient_key.ne("").all()
        case_ids = images.loc[images.sample_id.notna(), "image_id"].tolist()
        control_ids = images.loc[images.sample_id.isna(), "image_id"].tolist()
        assert len(case_ids) == 1949 and len(control_ids) == 91
        assert set(images.sample_id.dropna()) == set(samples.sample_id)
        immutable_paths = [OUTPUT / "images.parquet", OUTPUT / "panel_mapping.csv"]
        immutable_paths += [NAS / value for value in images.input_image_path]
        immutable_before = {relative(path): [path.stat().st_size, path.stat().st_mtime_ns]
                            for path in immutable_paths}
        assert set(json.loads((CACHE / "export_receipts.json").read_text())) == set(images.image_id)
        for name in ("sample_metadata", "field_mapping"):
            backup = REPAIR_JOB / f"before_{name}.csv"
            if not backup.exists():
                shutil.copy2(OUTPUT / f"{name}.csv", backup)

        # Exact source identifiers join clinical values; no patient-name inference.
        compared = 0
        for index, row in samples.iterrows():
            notes = json.loads(row.metadata_notes)
            source = clinical[clinical.Patient_ID.eq(row.source_subject_id)]
            assert not source.empty, row.sample_id
            raw = notes["clinical_raw"]
            assert source[list(raw)].nunique(dropna=False).eq(1).all(), row.sample_id
            assert source.iloc[0][list(raw)].to_dict() == raw, row.sample_id
            compared += len(raw)
            samples.loc[index, "metadata_notes"] = json.dumps(clinical_context_notes(raw, notes), ensure_ascii=False)
        targets = ["tnm_t", "tnm_n", "tnm_m", "smoking_status"]
        cleared_counts = {key: int(samples[key].ne("").sum()) for key in targets}
        affected_samples = before_samples.loc[before_samples[targets].ne("").any(axis=1), "sample_id"].tolist()
        affected_images = images.loc[images.sample_id.isin(affected_samples), "image_id"].tolist()
        samples[targets] = ""
        fields = clinical_context_fields(fields)
        pd.testing.assert_frame_equal(samples.drop(columns=targets + ["metadata_notes"]),
                                      before_samples.drop(columns=targets + ["metadata_notes"]))
        assert samples[targets + ["index_event", "stage_system", "stage_basis"]].eq("").all().all()
        for old, new in zip(before_samples.metadata_notes, samples.metadata_notes, strict=True):
            assert json.loads(old)["clinical_raw"] == json.loads(new)["clinical_raw"]

        changed = not samples.equals(before_samples) or not fields.equals(before_fields)
        if changed:
            progress("publishing_clinical_repair", 2040, cleared_nonempty_values=cleared_counts,
                     source_raw_values_compared=compared)
            details = dict(
                preparer_only=True, independent_review=False, repair_round=1,
                image_ids=case_ids, expected_images=2040, exported_images=2040,
                sample_rows=1068, sample_linked_images=1949, patient_linked_samples=1068,
                cleared_nonempty_values=cleared_counts, source_raw_values_compared=compared,
                changed_sample_ids=affected_samples, changed_image_ids=affected_images,
                action="Cleared index-time TNM/smoking targets; retained all raw codes and author-decoded source attributes; documented grade context limitation",
                technical_checks="Existing independent S01/I01-I05/P01-P03 and M01 retained; TIFFs, image index and panel unchanged",
                evidence=relative(REPAIR_JOB / "repair_verification.json"),
                reason="Repair prepared; independent M02 recheck and new D01 binding required",
            )
            additions = []
            for check_id, scope in [("M02", "sample"), ("D01", "study")]:
                additions.append(dict(study=STUDY, preparer=REPAIRER, reviewer="", scope=scope,
                                      record_id=STUDY, check_id=check_id, result="unresolved",
                                      source_evidence=relative(CP / "clinical_data_ROI.csv"),
                                      output_evidence=relative(OUTPUT / "sample_metadata.csv"),
                                      details=json.dumps(details, ensure_ascii=False)))
            updated_reviews = pd.concat([reviews, pd.DataFrame(additions)], ignore_index=True)
            # Revoke stale acceptance before replacing its metadata dependencies.
            for name, frame in [("review_results", updated_reviews), ("sample_metadata", samples),
                                ("field_mapping", fields)]:
                destination = OUTPUT / f"{name}.csv"
                temporary = destination.with_suffix(".csv.repair.tmp")
                with formal_publish_lease(destination, directory=OUTPUT):
                    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
                    temporary.replace(destination)

        actual_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        actual_fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(actual_samples, samples)
        pd.testing.assert_frame_equal(actual_fields, fields)
        actual_reviews = pd.read_csv(OUTPUT / "review_results.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(actual_reviews.iloc[:len(reviews)].reset_index(drop=True), reviews)
        for path in immutable_paths:
            assert [path.stat().st_size, path.stat().st_mtime_ns] == immutable_before[relative(path)], relative(path)
        candidate = input_binding(OUTPUT, NAS, case_ids)
        write_json(REPAIR_JOB / "candidate_input_binding.json", dict(
            preparer_only=True, independent_review=False, approved=False,
            instruction="Independent reviewer must recheck M02/dependencies before writing a D01 pass",
            input_binding=candidate))
        verification = dict(
            study=STUDY, repairer=REPAIRER, independent_review=False, repair_round=1,
            preparer_verification="pass", expected_images=2040, exported_images=2040,
            sample_rows=1068, sample_linked_images=1949, patient_linked_samples=1068,
            cleared_nonempty_values=cleared_counts, changed_sample_ids=affected_samples,
            changed_image_ids=affected_images, raw_source_values_compared=compared,
            raw_source_values_unchanged=True, unrelated_sample_fields_unchanged=True,
            source_clinical_attributes_preserved=True, all_four_target_fields_empty=True,
            unchanged_file_revisions=immutable_before, historical_review_rows_preserved=len(reviews),
            independent_technical_results_preserved=True, metadata_changed=changed,
            source_evidence=[relative(CP / "clinical_data_ROI.csv"), relative(META / "source_metadata/paper.xml"),
                             relative(next((META / "source_metadata/author_code").iterdir()) / "00_prepare_clinicaldata.Rmd"),
                             relative(review_job / "metadata_audit.json")],
            remaining_gaps=["91 control donor/sampling identities", "8 samples/14 images growth typing",
                            "Age/survival time semantics; undefined N=5", "Grade system and index context",
                            "Independent M02 recheck and D01 rebind for 1949 images"],
        )
        write_json(REPAIR_JOB / "repair_verification.json", verification)
    progress("clinical_repair_complete_pending_independent_review", 2040,
             sample_rows=1068, sample_linked_images=1949, patient_linked_samples=1068,
             cleared_nonempty_values=cleared_counts, source_raw_values_compared=compared,
             reused_tiffs=2040, newly_exported_images=0,
             verification=relative(REPAIR_JOB / "repair_verification.json"))
    print(json.dumps({key: value for key, value in verification.items()
                      if key not in {"changed_sample_ids", "changed_image_ids", "unchanged_file_revisions"}}, ensure_ascii=False), flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export", action="store_true")
parser.add_argument("--metadata", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--fetch", action="store_true")
parser.add_argument("--repair-clinical", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
if args.repair_clinical:
    PROGRESS = REPAIR_JOB / "progress.json"
    repair_clinical()
    raise SystemExit(0)
if args.fetch:
    fetch_metadata()
    if not (args.export or args.metadata or args.verify):
        raise SystemExit(0)
if args.verify and not (args.export or args.metadata):
    # Publication already bound the complete source inventory. Reopen all output
    # headers here and read source pixels only for the documented sample blocks.
    evidence = json.loads(SNAPSHOT.read_text())["images"]
    delivered = pd.read_parquet(OUTPUT / "images.parquet")
    panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
    panel["output_channel_index"] = pd.to_numeric(panel.output_channel_index.replace("", pd.NA)).astype("Int64")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    verify_outputs(delivered, panel)
    progress("preparer_verified", len(delivered), sample_rows=len(samples),
             sample_linked_images=int(delivered.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.ne("").sum()),
             gaps=["91张控制core缺少供者/取样分组", "临床时间语义缺口", "待独立复核"])
    raise SystemExit(0)
receipts_path = CACHE / "export_receipts.json"
receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
progress("checking_source_receipts", sum(r["admitted"] for r in receipts.values()),
         processed_sources=len(receipts), gaps=["91张控制core缺少供者/取样分组", "待独立复核"])
evidence, objects = source_inventory()
images, samples, panel, fields = metadata(evidence)
images, samples, fields = attach_clinical(images, fields)
completed = set(receipts)
if args.export:
    pending = images[~images.image_id.isin(completed)]
    if args.limit is not None:
        pending = pending.iloc[:args.limit]
    for row in pending.itertuples(index=False):
        if shutil.disk_usage(NAS).free < 100 * 1024**3 or shutil.disk_usage("/tmp").free < 64 * 1024**3:
            raise RuntimeError("Insufficient reserved disk capacity")
        progress("exporting", sum(r["admitted"] for r in receipts.values()), current_image=row.image_id,
                 gaps=["91张控制core缺少供者/取样分组", "待独立复核"])
        reader = NativeROI(NAS / evidence[row.image_id]["record"]["path"], row.image_id)
        assert list(reader.shape) == evidence[row.image_id]["record"]["shape"] and str(reader.dtype) == "float32"
        result = export_input(reader, panel, NAS / row.input_image_path)
        if len(receipts) in {0, 500, 1000, 1500, 2039}:
            overview(reader, row.image_id, panel)
        del reader
        result["image_id"] = row.image_id
        result["source_snapshot"] = relative(SNAPSHOT)
        receipts[row.image_id] = result
        write_json(receipts_path, receipts)
        admitted = {key for key, value in receipts.items() if value["admitted"]}
        delivered = images[images.image_id.isin(admitted)]
        save_input_tables(OUTPUT, delivered, samples, panel, fields)
        progress("exporting", len(delivered), processed_sources=len(receipts), gaps=["91张控制core缺少供者/取样分组", "待独立复核"])
        print(json.dumps(dict(completed=len(delivered), image_id=row.image_id, admitted=result["admitted"])), flush=True)
if args.export or args.metadata:
    admitted = {key for key, value in receipts.items() if value["admitted"]}
    delivered = images[images.image_id.isin(admitted)]
    save_input_tables(OUTPUT, delivered, samples, panel, fields)
    preparer_checks(delivered, samples, panel, "export_complete" if len(receipts) == 2040 else "partial_export")
    progress("export_complete" if len(receipts) == 2040 else "partial_export", len(delivered),
             processed_sources=len(receipts), sample_rows=len(samples),
             sample_linked_images=int(delivered.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()),
             gaps=["91张控制core缺少供者/取样分组", "临床时间语义缺口", "待独立复核"])
if args.verify:
    delivered = pd.read_parquet(OUTPUT / "images.parquet")
    verify_outputs(delivered, panel)
    progress("preparer_verified", len(delivered), processed_sources=len(receipts), sample_rows=len(samples),
             sample_linked_images=int(delivered.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()),
             gaps=["91张控制core缺少供者/取样分组", "临床时间语义缺口", "待独立复核"])
