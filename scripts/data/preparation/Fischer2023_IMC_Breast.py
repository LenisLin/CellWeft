"""Prepare Fischer breast IMC ROIs from the localized SPORA release.

Inputs: verified OME-Zarr level 0, source channel tables, S1, and author core metadata.
Outputs: native float32 square CYX input.tiff and the five study input tables on NAS.
Use plan for metadata-only scheduling; decode/export on the assigned remote CPU.
Formal publication uses a short per-image lease. Export is resumable.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import atexit
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["BLOSC_NTHREADS"] = "2"

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

zarr.config.set({"async.concurrency": 1, "threading.max_workers": 2})
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Fischer2023_IMC_Breast"
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = ROOT / "processed" / STUDY
RAW = ROOT / "raw/public" / STUDY / "spora"
META = metadata_source_directory(ROOT, STUDY)
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0032_prepare_0/progress.json"
PROOF = META / "native_verification.json"
AUTHOR = META / "source_metadata/article_mmc2.csv"
PANEL_ID = "imc_37"
PREPARER = "codex_cli_Fischer2023_IMC_Breast_0032_prepare_0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["plan", "prepare", "export", "verify", "run"])
parser.add_argument("--limit", type=int, default=25)
parser.add_argument("--s1", type=Path, default=ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv")
parser.add_argument("--plan-dir", type=Path)
args = parser.parse_args()
PROGRESS.parent.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / "receipts").mkdir(exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, ROOT))


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage, expected_images=len(eligible),
                             exported_images=completed, updated_at=datetime.now(timezone.utc).isoformat(),
                             **details))


def array_for(image_id: str) -> np.ndarray:
    """Decode one small source ROI once; largest verified ROI is < 1,000 px per side.

    Each source shard stores all 37 channels together. Repeated single-channel reads
    would decompress the same data repeatedly. This study-local adaptation reuses the
    shared TIFF exporter and bounds resident image data to one ROI (~121 MiB).
    """
    source_path = ROOT / proofs[image_id]["path"] / "0"
    with formal_read_lease(source_path):
        array = zarr.open_array(str(source_path), mode="r")
        values = np.asarray(array[:])
    assert values.shape == tuple(proofs[image_id]["shape"]) and values.dtype == np.dtype("float32")
    return values


class ROIReader:
    def __init__(self, values: np.ndarray, image_id: str):
        self.values, self.shape, self.dtype = values, values.shape, values.dtype
        self.source = SourceSpec(image_id, PANEL_ID, "zarr", ROOT / proofs[image_id]["path"] / "0",
                                 tuple(range(37)), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def check_header(path: Path, shape: tuple[int, int, int]) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == shape and tif.series[0].axes == "CYX"
        assert tif.series[0].dtype == np.dtype("float32")
        channels = ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")
        assert [channel.attrib["Name"] for channel in channels] == retained.display_name.tolist()


# Source order is checked against both the localized tables and every accepted OME header.
proof_data = json.loads(PROOF.read_text())
proofs = {record["image_id"]: record for record in proof_data["verified"]}
accepted = json.loads((META / "accepted_images.json").read_text())
assert set(proofs) == set(accepted["accepted_image_ids"])
native_order = json.loads((META / "imc_ordered_channels_by_tissue.json").read_text())
channels = pd.read_parquet(RAW / "imc/channels.parquet")
availability = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
source_names = channels.channel_name.tolist()
assert len(source_names) == 37
for image_id, proof in proofs.items():
    assert [item["channel_name"] for item in native_order[image_id]] == source_names
    assert proof["canonical_channel_labels"] == source_names and proof["exact_canonical_order_checked"]

# Source channel table explicitly recommends D1H2, not HTA28, for general Histone H3.
assert "Only use Histone H3" in str(channels.loc[channels.channel_name == "Histone H3 (HTA28)", "description"].iloc[0])
panel = panel_from_s1(STUDY, PANEL_ID, source_names, args.s1,
                      drop_indices={14: "Source channels.parquet recommends D1H2 for general Histone H3; duplicate HTA28 not selected"})
panel["display_name"] = panel.marker_id
panel.loc[panel.nuclear, "display_name"] = panel.loc[panel.nuclear, "source_name"]
retained = panel[panel.keep_channel].sort_values("output_channel_index")
neutral_marker = "Fischer_Yb176_target_unresolved"
target = panel.loc[panel.source_channel_index == 36].iloc[0]
identity_gap = not (target.marker_id == neutral_marker and target.mapping_status == "mapping_pending")
panel["source_identity_status"] = "confirmed"
panel.loc[panel.source_channel_index == 36, "source_identity_status"] = "unresolved" if identity_gap else "target_unresolved_user_approved_pending"
if identity_gap:
    panel.loc[panel.source_channel_index == 36, "keep_reason"] += "; source Table S2 also lists Cleaved Caspase3 at Yb176; export held for P01"
else:
    panel.loc[panel.source_channel_index == 36, "keep_reason"] += "; user-approved mapping_pending neutral measurement; target unresolved; no vector or training admission"
# The shared exporter currently labels from marker_id. Keep canonical marker_id in
# the mapping, but pass explicit measurement labels for nuclear TIFF channels.
export_panel = panel.copy()
export_panel["marker_id"] = export_panel.display_name

# Author R/00_CP_to_SCE.Rmd:234-235 joins physical cores by ArrayNr + UBTMA_location.
# SPORA specimen_id is a patient-local counter, so it is not used as a biological key.
tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
assert set(tissues.tissue_id) == set(proofs)
tissues["ArrayNr"] = tissues.original_file_name.str.split(".").str[0]
tissues["UBTMA_location"] = tissues.UBTMA_location.map(lambda value: str(int(value)) if pd.notna(value) else "")
author = pd.read_csv(AUTHOR, encoding="cp1252", dtype=str, keep_default_na=False)
author = author.replace({"#NULL!": "", "NA": "", "NaN": ""})
assert not author.duplicated(["ArrayNr", "UBTMA_location"]).any()
zenodo_author = pd.read_csv(META / "source_metadata/Patient_metadata.csv", encoding="cp1252", dtype=str, keep_default_na=False)
identity_comparison = author.merge(zenodo_author, on=["ArrayNr", "UBTMA_location"], suffixes=("_paper", "_zenodo"), validate="one_to_one")
assert len(identity_comparison) == len(author) == len(zenodo_author)
assert (identity_comparison.PID_paper == identity_comparison.PID_zenodo).all()
joined = tissues.merge(author, on=["ArrayNr", "UBTMA_location"], how="left", suffixes=("_spora", "_author"),
                       indicator=True, validate="many_to_one").set_index("tissue_id", verify_integrity=True)
joined["sample_id"] = pd.NA
known = joined._merge == "both"
joined.loc[known, "sample_id"] = joined.loc[known, "ArrayNr"] + "_core_" + joined.loc[known, "UBTMA_location"]
test_ids = set(joined.index[joined.acquisition_description.fillna("").str.contains("test__debris", regex=False)])
non_test = set(proofs) - test_ids
unlinked = sorted(non_test - set(joined.index[known]))
eligible = sorted(non_test & set(joined.index[known]))
assert len(eligible) == 1238 and len(unlinked) == 13

sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
samples = pd.DataFrame(columns=sample_columns)
sample_rows = []
for sample_id, group in joined[known].groupby("sample_id", sort=True):
    row = group.iloc[0]
    record = dict.fromkeys(sample_columns)
    record.update(study=STUDY, sample_id=sample_id, patient_key="Fischer2023_PID_" + row.PID,
                  source_subject_id=row.PID, cohort_id="ZTMA21_25_26", source_sample_id=row.ArrayNr + ":" + row.UBTMA_location,
                  sample_type="TMA core biopsy", processing_method="Tissue microarray; imaging mass cytometry")
    record["sex_reported"] = row.Sex_author or None
    record["histology"] = row.TumorSubtype_3gr_author or None
    record["tnm_t"], record["tnm_n"] = row.pT_author or None, row.pN_author or None
    record["grade"], record["molecular_subtype"] = row.Grade_author or None, row.Mol_Signature_author or None
    if row.OS_author and row.Livestatus1censored_author in {"0", "1"}:
        # Author R lines 154-174 define months since diagnosis and censoring code.
        record.update(os_time=float(row.OS_author), os_time_unit="month", os_origin="diagnosis",
                      os_event=1-int(row.Livestatus1censored_author),
                      os_event_definition="1 = death; 0 = censored (1 - author Livestatus1censored)")
    tissue_type = row.TissueType_author
    record["tissue_type"] = "normal" if tissue_type == "normal breast tissue" else ("tumor" if tissue_type else None)
    record["lesion_type"] = {"primary tumor": "primary", "DCIS": "primary", "lymph node mts": "metastasis",
                             "distant mts": "metastasis", "tumor recurrence": "recurrence", "normal breast tissue": "non_malignant"}.get(tissue_type)
    record["anatomical_site"] = "lymph node" if tissue_type == "lymph node mts" else ("breast" if tissue_type in {"primary tumor", "DCIS", "normal breast tissue"} else None)
    record["metadata_notes"] = ("Author physical core biopsy identity; repeated acquisitions of this core share sample_id. "
                                "Parent surgical specimen/block identity and same-parent-core grouping unavailable. "
                                "Age timing is unspecified and target age is empty. Author OS is months since diagnosis; updated times were rounded weeks/4. "
                                "Author ArrayNr+UBTMA_location overrides conflicting SPORA patient/clinical metadata.")
    sample_rows.append(record)
samples = pd.DataFrame(sample_rows, columns=sample_columns)

mapping_columns = pd.read_csv(TEMPLATES / "field_mapping.csv").columns.tolist()
mapping_rows = []


def mapping(source: str, source_field: str, table: str, target: str, rule: str, *, notes: str = "", values: dict | None = None) -> None:
    record = dict.fromkeys(mapping_columns, "")
    record.update(study=STUDY, source_file=source, source_sheet_or_table="source table", source_field=source_field,
                  target_table=table, target_field=target, join_keys="tissue_id -> (ArrayNr, UBTMA_location) -> author core row",
                  value_mapping=json.dumps(values or {}, ensure_ascii=False), transformation_rule=rule,
                  conflict_rule="Original paper Table S1 core row takes precedence; identity agrees with all Zenodo core rows. Source missing values retained empty; conflicts with SPORA retained in source_identity_conflicts.csv", notes=notes)
    mapping_rows.append(record)


author_path = relative(AUTHOR)
for target, source_field, rule in [
    ("study", "dataset", "Constant canonical study ID"),
    ("sample_id", "ArrayNr;UBTMA_location", "ArrayNr + '_core_' + UBTMA_location; one author-confirmed physical TMA core biopsy; repeated images share sample"),
    ("patient_key", "PID", "Prefix author PID with Fischer2023_PID_; no inference from SPORA image names"),
    ("source_subject_id", "PID", "Preserve source identifier as string"),
    ("cohort_id", "ArrayNr", "Study cohort ZTMA21_25_26"),
    ("source_sample_id", "ArrayNr;UBTMA_location", "ArrayNr + ':' + UBTMA_location"),
    ("sample_type", "ArrayNr;UBTMA_location", "Physical core biopsy as defined by author methods and R join; parent specimen/block unknown"),
    ("processing_method", "ArrayNr", "TMA IMC source provenance"),
    ("sex_reported", "Sex", "Copy author female/male; source missing tokens become empty"),
    ("histology", "TumorSubtype_3gr", "Copy source histology"),
    ("tnm_t", "pT", "Copy source pathological T code; no index time inferred"),
    ("tnm_n", "pN", "Copy source pathological N code; no index time inferred"),
    ("grade", "Grade", "Copy source grade"),
    ("molecular_subtype", "Mol_Signature", "Copy source molecular classification"),
    ("tissue_type", "TissueType", "normal breast tissue -> normal; other documented breast lesion types -> tumor"),
    ("lesion_type", "TissueType", "primary tumor/DCIS -> primary; lymph node/distant mts -> metastasis; recurrence -> recurrence; normal -> non_malignant"),
    ("anatomical_site", "TissueType", "lymph node mts -> lymph node; primary tumor/DCIS/normal breast tissue -> breast; distant and recurrence site unspecified"),
    ("metadata_notes", "ArrayNr;UBTMA_location;PID", "Record physical core granularity, unavailable parent specimen grouping and conservative clinical time handling"),
    ("os_time", "OS;Livestatus1censored", "Copy OS only when source censoring indicator is available; author R lines 154-174 define diagnosis-to-follow-up months, updated values round elapsed weeks/4"),
    ("os_time_unit", "OS", "month, as defined by author R survival calculation"),
    ("os_origin", "OS;author R code", "diagnosis; source R uses EDZP diagnosis date"),
    ("os_event", "Livestatus1censored", "1 - Livestatus1censored; 1=death, 0=censored; only with nonempty OS"),
    ("os_event_definition", "Livestatus1censored;Status_OS", "Author R maps censored=1, death=0; target event is inverted"),
]:
    mapping(author_path, source_field, "sample_metadata", target, rule)
    if target.startswith("os_"):
        mapping_rows[-1]["time_origin_rule"] = "Diagnosis; author R/00_CP_to_SCE.Rmd:154-174"
mapping(relative(RAW / "metadata/tissues.parquet"), "tissue_id;original_file_name;acquisition_description;UBTMA_location",
        "image_metadata", "sample_id", "Use published tissue_id and core coordinates; connect only author-confirmed ArrayNr+UBTMA_location; missing keys remain null",
        notes="Author repository: https://github.com/BodenmillerGroup/BC_LN_metastses/blob/main/R/00_CP_to_SCE.Rmd; lines 84-86,234-235. Original paper Table S1 obtained via Europe PMC supplementaryFiles (article_mmc2.csv); all 1720 ArrayNr+UBTMA_location/PID identities agree with Zenodo Patient_metadata.csv. Paper cleaned clinical entries take precedence over Zenodo sentinel codes. No author contact.")
mapping_rows[-1]["notes"] += f" Source linkage: {int(known.sum())}/1252 ROIs, {len(samples)} physical-core samples, {samples.patient_key.nunique()} patients; image export coverage reported separately in M01."
for target, fields, rule in [
    ("image_id", "tissue_id", "Preserve published SPORA tissue ID; names do not determine biological identity"),
    ("source_files", "tissue_id", "NAS-relative OME-Zarr source directory"),
    ("source_image_id", "tissue_id", "Published source ROI ID"),
    ("slide_id", "original_file_name", "Copy source TMA acquisition part"),
    ("core_id", "ArrayNr;UBTMA_location", "Author physical-core key when known; otherwise empty"),
    ("roi_id", "acquisition_description", "Copy source acquisition annotation; no inferred numeric acquisition ID"),
    ("region_annotation", "acquisition_description;TissueType", "JSON of literal SPORA acquisition description and authoritative author TissueType when connected"),
    ("source_revision", "source proof", "Localized SPORA OME-Zarr v3 release with reused source-size/native-read proof; no source hashing"),
    ("panel_id", "channel_name;channels_per_tissue", "All source ROIs share the same explicitly measured 37-channel order"),
    ("image_kind", "source ROI inventory", "Published independent ROI, including full physical core acquisition frame"),
    ("source_format", "zarr.json", "ome-zarr; OME 0.5 and native Zarr format 3"),
    ("source_axes", "dimension_names", "Canonical native c,y,x axis order -> CYX"),
    ("source_dtype", "data_type", "Copy native float32, without intensity transformation"),
    ("source_channel_count", "shape[0]", "37 measured source channels per accepted ROI"),
    ("channel_count", "panel_mapping.keep_channel", "35 selected channels; output indices follow retained source order"),
    ("source_x0_px", "source native frame", "0; use entire source acquisition width"),
    ("source_y0_px", "source native frame", "0; use entire source acquisition height"),
    ("source_x1_px", "shape[2]", "Native width; half-open source frame"),
    ("source_y1_px", "shape[1]", "Native height; half-open source frame"),
    ("height_px", "shape[1:3]", "L=max(native height,native width)"),
    ("width_px", "shape[1:3]", "L=max(native height,native width)"),
    ("pad_left_px", "native width;L", "floor((L-native width)/2)"),
    ("pad_right_px", "native width;L", "L-native width-pad_left_px"),
    ("pad_top_px", "native height;L", "floor((L-native height)/2)"),
    ("pad_bottom_px", "native height;L", "L-native height-pad_top_px"),
    ("pixel_size_x_um", "OME multiscales scale", "Copy native 1.0 micrometer scale"),
    ("pixel_size_y_um", "OME multiscales scale", "Copy native 1.0 micrometer scale"),
    ("input_image_path", "tissue_id", "NAS-relative processed/study/images/image_id/input.tiff"),
    ("intensity_description", "SPORA level 0 source array", "Unmodified native float32; source published scale"),
    ("split_method", "independent source ROI", "already_independent"),
    ("split_evidence", "source image geometry", "Full independently published ROI, no tissue cropping or resizing; centered square zero padding"),
    ("metadata_notes", "core metadata join", "Record author precedence, missing biological links and independent review status"),
]:
    mapping(relative(RAW / "metadata/tissues.parquet"), fields, "image_metadata", target, rule)
mapping(relative(META / "imc_channels.csv"), "channel_name;description", "panel_mapping", "source_name;marker_id;display_name",
        "Exact study+Original marker S1 join; D1H2 chosen over duplicate HTA28 by source recommendation; H3K27me3 excluded; nuclear display_name retains D1H2 clone",
        notes="Temporary study-local display_name adapter for shared exporter; canonical marker_id remains unchanged. Original paper Table S2 lists both Cleaved Caspase3 and cleaved PARP at Yb176; Panel.csv full selects Cleaved Caspase3. Native full.csv confirms Yb176 at source position 36. S1 PARP1-only identity is unresolved; export of this single panel is held. See channel_identity_evidence.json and article_mmc1.pdf.")
fields = pd.DataFrame(mapping_rows, columns=mapping_columns)
if not identity_gap:
    selected_field = fields.target_table.eq("panel_mapping") & fields.target_field.eq("source_name;marker_id;display_name")
    fields.loc[selected_field, "notes"] = ("User approved neutral Yb176 mapping_pending measurement from canonical S1; "
        "actual antibody remains unresolved. Retain raw output page34; mapped signal admission and vector use only the other34 confirmed markers. "
        "Fischer training and sampling remain pending; exact author-core-linked non-test ROI only.")

# Metadata-only verification may run while an export command is still queued for
# the global NAS slot. Serialize actual study mutations once either command starts.
if args.action not in {"run", "plan"}:
    preparation_lock = (CACHE / ".preparer.lock").open("a")
    fcntl.flock(preparation_lock, fcntl.LOCK_EX)
    atexit.register(preparation_lock.close)


def image_record(image_id: str, facts: dict) -> dict:
    row, proof = joined.loc[image_id], proofs[image_id]
    _, h, w = proof["shape"]
    record = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns)
    record.update(study=STUDY, image_id=image_id, sample_id=None if pd.isna(row.sample_id) else row.sample_id,
                  panel_id=PANEL_ID, image_kind="roi", source_image_id=image_id, source_files=[proof["path"]],
                  source_format="ome-zarr", source_axes="CYX", source_dtype=proof["dtype"],
                  source_revision="SPORA localized 2026-10-18; OME-Zarr v3; existing native/size proof", slide_id=row.original_file_name,
                  core_id=None if pd.isna(row.sample_id) else row.sample_id, roi_id=row.acquisition_description,
                  region_annotation=json.dumps({"acquisition_description": row.acquisition_description,
                                                "TissueType": row.TissueType_author if row._merge == "both" else row.TissueType_spora}, ensure_ascii=False),
                  source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                  pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(OUTPUT / "images" / image_id / "input.tiff"),
                  intensity_description="Unmodified published SPORA float32 intensities; no normalization, rescaling or transformation",
                  split_method="already_independent", split_evidence="Published independent ROI; full source frame retained; no multi-sample split",
                  metadata_notes="Author core-key biological linkage; SPORA original patient grouping may conflict. Unconnected ROIs retain null sample_id. Independent review pending.")
    for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]:
        record[key] = facts[key]
    return record


def receipts() -> dict:
    return {path.stem: json.loads(path.read_text()) for path in (CACHE / "receipts").glob("*.json")}


def publish() -> pd.DataFrame:
    records = [image_record(image_id, fact) for image_id, fact in sorted(receipts().items()) if fact["admitted"]]
    images = pd.DataFrame(records, columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns)
    if (OUTPUT / "images.parquet").exists():
        previous = pd.read_parquet(OUTPUT / "images.parquet")
        # Merge any existing rows outside this invocation's completed receipt range.
        images = pd.concat([previous[~previous.image_id.isin(images.image_id)], images], ignore_index=True)
    save_input_tables(OUTPUT, images, samples, panel, fields)
    return images


if args.action == "plan":
    assert args.plan_dir is not None and not identity_gap
    args.plan_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.plan_dir / "panel_mapping.csv", index=False)
    samples.to_csv(args.plan_dir / "sample_metadata.private.csv", index=False)
    fields.to_csv(args.plan_dir / "field_mapping.csv", index=False)
    confirmed = panel[panel.keep_channel & panel.role.eq("downstream") & panel.mapping_status.eq("mapped")]
    assert len(confirmed) == 34 and len(retained) == 35
    candidates = []
    for image_id in eligible:
        _, h, w = proofs[image_id]["shape"]
        side = max(h, w)
        facts = dict(height_px=side, width_px=side, pad_left_px=(side-w)//2, pad_top_px=(side-h)//2,
            pad_right_px=side-w-(side-w)//2, pad_bottom_px=side-h-(side-h)//2,
            source_channel_count=37, channel_count=35)
        candidates.append(dict(image_id=image_id, proof=proofs[image_id], image_row=image_record(image_id, facts)))
    write_json(args.plan_dir / "export_plan.private.json", dict(study=STUDY, panel_id=PANEL_ID,
        preparer="codex_cli_Fischer2023_IMC_Breast_scientific_repair_1", scientific_repair_round=1,
        candidates=candidates, pending_unlinked_image_ids=unlinked, excluded_test_image_ids=sorted(test_ids),
        confirmed_signal_source_indices=confirmed.source_channel_index.tolist(), retained_source_indices=retained.source_channel_index.tolist(),
        neutral_source_index=36, neutral_output_index=34, neutral_marker=neutral_marker,
        training_admission=False, panel=json.loads(panel.to_json(orient="records"))))
    print(json.dumps(dict(study=STUDY, candidate_upper_bound=len(candidates), pending_unlinked=len(unlinked),
        confirmed_signal_channels=len(confirmed), retained_pages=len(retained), host_arrays_read=False)))


if args.action == "prepare":
    # Reuse accepted native proofs; check complete object sizes, not pixel hashes.
    objects = json.loads((META / "imc_source_objects.json").read_text())
    inventory_prefix = "spora-submissions/fischer2023multiplex/"
    for item in objects:
        path = RAW / item["key"].removeprefix(inventory_prefix)
        assert path.stat().st_size == item["size"], str(path)
    actual = {path.name.removesuffix(".ome.zarr") for path in (RAW / "imc/1_0mpp/images").glob("*.ome.zarr")}
    assert actual == set(proofs)
    assert list(availability.columns) == source_names
    assert availability.to_numpy().all()
    conflict = joined[known & (pd.to_numeric(joined.original_patient_identifier, errors="coerce") != pd.to_numeric(joined.PID, errors="coerce"))]
    conflict.to_csv(CACHE / "source_identity_conflicts.csv", encoding="utf-8-sig")
    joined[joined._merge != "both"].to_csv(CACHE / "unlinked_source_rois.csv", encoding="utf-8-sig")
    # Literal acquisition descriptions can repeat; retain distinct observations unless pixels are equal.
    duplicate_evidence = []
    for description, group in joined[joined.acquisition_description.notna()].groupby("acquisition_description"):
        if len(group) < 2:
            continue
        ids = group.index.tolist()
        left, right = array_for(ids[0]), array_for(ids[1])
        duplicate_evidence.append(dict(description=description, image_ids=ids, shapes=[list(left.shape), list(right.shape)],
                                       exactly_equal=bool(np.array_equal(left, right)),
                                       decision="Distinct pixel observations of the same physical core; share sample_id"))
        assert not np.array_equal(left, right), "Exact duplicate requires explicit source selection before export"
    write_json(CACHE / "repeated_acquisition_annotations.json", duplicate_evidence)
    selections = list(joined.groupby("original_file_name").head(1).index)
    selections += list(joined.index[joined._merge != "both"])
    selections = list(dict.fromkeys(selections))
    thumbs = []
    for image_id in selections:
        values = array_for(image_id)
        rgb = np.zeros((*values.shape[1:], 3), dtype=np.float32)
        for out, channel in enumerate([35, 22, 0]):
            layer = values[channel]
            upper = np.quantile(layer[layer > 0], .99) if np.any(layer > 0) else 1
            rgb[:, :, out] = np.clip(layer / upper, 0, 1)
        thumb = Image.fromarray((rgb * 255).astype(np.uint8))
        thumb.thumbnail((235, 210))
        canvas = Image.new("RGB", (250, 255), "white")
        canvas.paste(thumb, ((250-thumb.width)//2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((3, 212), image_id.removeprefix("fischer2023multiplex_"), fill="black")
        draw.text((3, 228), str(joined.loc[image_id].acquisition_description)[:36], fill="black")
        thumbs.append(canvas)
    montage = Image.new("RGB", (250*5, 255*((len(thumbs)+4)//5)), "white")
    for i, thumb in enumerate(thumbs):
        montage.paste(thumb, (250*(i%5), 255*(i//5)))
    montage.save(CACHE / "source_overviews.png")
    upper_bytes = sum(len(retained) * max(proofs[image_id]["shape"][1:])**2 * 4 for image_id in eligible)
    write_json(CACHE / "source_scope.json", dict(source_images=len(proofs), eligible_before_signal=len(eligible),
               excluded_test_debris=sorted(test_ids), source_bytes=proof_data["source_bytes"],
               uncompressed_output_upper_bytes=upper_bytes, max_source_roi_bytes=max(np.prod(p["shape"]).item()*4 for p in proofs.values()),
               sample_rows=len(samples), linked_source_images=int(known.sum()), patient_linked_samples=int(samples.patient_key.notna().sum()),
               author_patients=int(samples.patient_key.nunique()), spora_author_patient_conflicts=len(conflict),
               metadata_evidence=[author_path, relative(META / "source_metadata/Patient_metadata.csv"), relative(META / "source_metadata/author_00_CP_to_SCE.Rmd"), relative(META / "source_metadata/author_tree.json"), relative(META / "source_metadata/article_fulltext.xml"), relative(META / "source_metadata/article_supplement_metadata_retrieval.json")],
               issues=["14 source ROIs lack author core metadata; one explicitly test/debris is excluded",
                       "Parent specimen/block grouping across different physical cores unavailable",
                       "Source Panel.csv full channel 37 says Cleaved Caspase3; SPORA OME/channel metadata says cleaved PARP",
                       "Independent review required; no D01 approval issued by preparer"]))
    publish()
    progress("metadata_prepared", 0, source_images=len(proofs), sample_rows=len(samples), gaps=json.loads((CACHE / "source_scope.json").read_text())["issues"])
    print(json.dumps(json.loads((CACHE / "source_scope.json").read_text()), ensure_ascii=False))

if args.action == "export":
    assert (CACHE / "source_scope.json").exists(), "Run prepare first"
    if identity_gap:
        progress("blocked_source_channel_identity", sum(f["admitted"] for f in receipts().values()),
                 blocked_image_count=len(eligible), blocked_source_channel_index=36,
                 gaps=["P01: Yb176 carries Cleaved Caspase3 and/or cleaved PARP; current S1 PARP1-only mapping requires controller resolution"])
        raise SystemExit("P01 blocked: resolve original Table S2 / Panel.csv Yb176 identity against study S1 before TIFF export")
    done = receipts()
    todo = [image_id for image_id in eligible if image_id not in done][:args.limit]
    for image_id in todo:
        if shutil.disk_usage(ROOT).free < 100*1024**3 or shutil.disk_usage("/tmp").free < 64*1024**3:
            progress("capacity_blocked", sum(f["admitted"] for f in done.values()), current_image=image_id)
            raise SystemExit("Insufficient reserved disk space")
        progress("exporting", sum(f["admitted"] for f in done.values()), current_image=image_id)
        values = array_for(image_id)
        confirmed = panel[panel.keep_channel & panel.role.eq("downstream") & panel.mapping_status.eq("mapped")]
        if not np.any(values[confirmed.source_channel_index.to_list()] != 0):
            write_json(CACHE / "receipts" / (image_id + ".json"), dict(admitted=False,
                signal_rule="confirmed_mapped_downstream_only", exclusion="no_confirmed_marker_signal"))
            done[image_id] = dict(admitted=False)
            del values
            continue
        reader = ROIReader(values, image_id)
        destination = OUTPUT / "images" / image_id / "input.tiff"
        facts = export_input(reader, export_panel, destination, tile_size=256)
        if facts["admitted"]:
            check_header(destination, (len(retained), facts["height_px"], facts["width_px"]))
            facts["output_size"] = destination.stat().st_size
        write_json(CACHE / "receipts" / (image_id + ".json"), facts)
        done[image_id] = facts
        # Real index after each image supports interruption without losing completed work.
        images = publish()
        progress("exporting" if set(eligible)-set(done) else "export_complete", len(images),
                 completed_source_images=len(done), all_zero_excluded=sum(not f["admitted"] for f in done.values()),
                 sample_linked_images=int(images.sample_id.notna().sum()), sample_rows=len(samples),
                 current_image=image_id, gaps=json.loads((CACHE / "source_scope.json").read_text())["issues"])
        print(json.dumps(dict(image_id=image_id, admitted=facts["admitted"], exported_images=len(images)), ensure_ascii=False), flush=True)
        del values, reader

if args.action == "verify":
    images = publish()
    all_receipts = receipts()
    source_scope = json.loads((CACHE / "source_scope.json").read_text())
    source_scope["metadata_evidence"] = [author_path, relative(META / "source_metadata/Patient_metadata.csv"),
        relative(META / "source_metadata/author_00_CP_to_SCE.Rmd"), relative(META / "source_metadata/author_tree.json"),
        relative(META / "source_metadata/article_fulltext.xml"), relative(META / "source_metadata/article_supplement_metadata_retrieval.json")]
    source_scope["paper_zenodo_identity_rows_checked"] = len(identity_comparison)
    source_scope["clinical_precedence"] = "Original paper Table S1 (article_mmc2.csv), with explicit author R endpoint semantics"
    source_scope["blocked_image_count"] = len(eligible) if identity_gap else 0
    source_scope["metadata_evidence"] += [relative(META / "source_metadata/article_mmc1.pdf"), relative(META / "source_metadata/channel_identity_evidence.json")]
    source_scope["issues"] = ["13 non-test source ROIs lack author core/sample identity",
        "Parent specimen/block grouping across different physical cores unavailable",
        "P01: original Table S2 lists both Cleaved Caspase3 and cleaved PARP at Yb176; S1 PARP1-only mapping unresolved; all 1251 candidate TIFF exports held",
        "No TIFF export or whole-image signal/pixel verification completed; independent review not performed"]
    write_json(CACHE / "source_scope.json", source_scope)
    headers, pixel_checks = [], []
    for row in images.itertuples(index=False):
        check_header(ROOT / row.input_image_path, (len(retained), int(row.height_px), int(row.width_px)))
        headers.append(row.image_id)
    # Representative groups include every original slide part plus both padding directions.
    selected = images.groupby("slide_id").head(1).image_id.tolist()
    for image_id in selected:
        row = images.set_index("image_id").loc[image_id]
        values = array_for(image_id)
        _, h, w = values.shape
        blocks = [(0, 0, min(32, w), min(32, h)), (w//2, h//2, min(w//2+32,w), min(h//2+32,h))]
        with formal_read_lease(ROOT / row.input_image_path), tifffile.TiffFile(ROOT / row.input_image_path) as tif:
            for block in blocks:
                x0,y0,x1,y1 = block
                output_block = (x0+int(row.pad_left_px),y0+int(row.pad_top_px),x1+int(row.pad_left_px),y1+int(row.pad_top_px))
                for channel in retained.itertuples(index=False):
                    actual = read_tiff_window(tif, int(channel.output_channel_index), output_block)
                    assert np.array_equal(actual, values[int(channel.source_channel_index),y0:y1,x0:x1])
                pixel_checks.append(dict(image_id=image_id, source_xyxy=block, output_xyxy=output_block,
                                         source_channels=retained.source_channel_index.tolist(), exact_equal=True))
            if row.pad_left_px:
                padding = (0, 0, int(row.pad_left_px), int(row.height_px))
            else:
                padding = (0, 0, int(row.width_px), int(row.pad_top_px)) if row.pad_top_px else None
            if padding:
                for channel in range(len(retained)):
                    assert not read_tiff_window(tif, channel, padding).any()
        del values
    write_json(CACHE / "preparer_verification.json", dict(header_image_ids=headers, exact_pixel_checks=pixel_checks,
               independent_review=False, source_count=len(proofs), eligible_count=len(eligible), completed_sources=len(all_receipts),
               exported_images=len(images), all_zero_excluded=[k for k,v in all_receipts.items() if not v["admitted"]],
               sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
               patient_linked_samples=int(samples.patient_key.notna().sum()), patient_count=int(samples.patient_key.nunique())))
    review_columns = pd.read_csv(TEMPLATES / "review_results.csv").columns
    review_rows = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples(index=False):
        detail = "Preparer self-check only; independent reviewer has not evaluated this delivery."
        result = "unresolved"
        if check.check_id == "M01":
            detail += f" Source linkage: {int(known.sum())}/1252 ROIs; delivered linkage: {images.sample_id.notna().sum()}/{len(images)} exported images; {len(samples)} physical-core samples; {samples.patient_key.nunique()} author patients. Unlinked ROI identities and parent specimen grouping remain unresolved; author core-key join overrides 225 SPORA patient-ID conflicts."
        elif check.check_id == "P01":
            detail += " 37-channel SPORA order and S1 exact join verified; 35 would be retained. Table S2 lists both Cleaved Caspase3 and cleaved PARP at Yb176; author Panel.csv full selects Cleaved Caspase3. Native full.csv confirms isotope at zero-based position 36. S1 PARP1-only identity is insufficiently established. All 1251 images share this panel and TIFF export is held pending controller resolution; see channel_identity_evidence.json."
        elif check.check_id == "D01":
            detail += f" {len(headers)} TIFF headers checked; {len(pixel_checks)} all-retained-channel source/output blocks exactly equal. No input_binding or D01 pass issued."
        elif check.check_id == "S01":
            detail += f" {len(proofs)} source ROIs; test/debris exclusions={sorted(test_ids)}; completed sources={len(all_receipts)}; all-zero count={sum(not v['admitted'] for v in all_receipts.values())}. See source_scope and receipts."
        review_rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope, record_id=STUDY,
                                check_id=check.check_id, result=result, source_evidence=relative(PROOF)+";"+author_path,
                                output_evidence=relative(CACHE / "preparer_verification.json"), details=detail))
    review_path = OUTPUT / "review_results.csv"
    if review_path.exists():
        previous = pd.read_csv(review_path, keep_default_na=False)
        previous = previous[previous.preparer != PREPARER]
        reviews = pd.concat([previous, pd.DataFrame(review_rows, columns=review_columns)], ignore_index=True)
    else:
        reviews = pd.DataFrame(review_rows, columns=review_columns)
    with formal_publish_lease(review_path, directory=review_path.parent):
        reviews.to_csv(review_path, index=False, encoding="utf-8-sig")
    progress("blocked_source_channel_identity" if identity_gap else ("prepared_pending_independent_review" if set(eligible).issubset(all_receipts) else "metadata_ready_export_queued"),
             len(images), completed_source_images=len(all_receipts),
             sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()), patient_count=int(samples.patient_key.nunique()),
             gaps=json.loads((CACHE / "source_scope.json").read_text())["issues"])
    print((CACHE / "preparer_verification.json").read_text())

if args.action == "run":
    # Release the global writer slot between small batches; keep every child managed.
    if identity_gap:
        raise SystemExit("P01 blocked: resolve Yb176 identity against study S1 before queued TIFF exports")
    while set(eligible) - set(receipts()):
        progress("waiting_for_write_slot", sum(f["admitted"] for f in receipts().values()),
                 gaps=json.loads((CACHE / "source_scope.json").read_text())["issues"])
        subprocess.run([sys.executable, "-B", __file__, "export", "--limit", str(args.limit)], check=True)
    subprocess.run([sys.executable, "-B", __file__, "verify"], check=True)
