"""Prepare native Colombo DLBCL ROI inputs from the complete Figshare v1 ZIP.

Inputs: accepted native proofs, ordered TIFF labels, S1, author clinical workbook,
and the paper's supplementary methods. Outputs: per-ROI uint16 CYX input.tiff,
complete image/sample/panel/field tables, and preparer evidence awaiting review.
Use --stage export under the batch shared formal write lease wrapper; --stage verify uses
its read wrapper. --stage repair updates reviewed metadata while preserving TIFFs
and existing image/panel tables. Clinical identifiers are read only from NAS.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative
from cellweft.data.raw_consolidation import file_identity

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import openpyxl
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables, validate_image_metadata,
)

STUDY = "Colombo2022_IMC_DLBCL"
PANEL = "colombo_figshare20010146_v1"
PREPARER = "Colombo2022_IMC_DLBCL_0023_prepare_0"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
SOURCE = META / "preparation_sources"
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
NATIVE = META / "native_runtime"
ARCHIVE = NAS / "raw/public" / STUDY / "figshare20010146_v1/raw-ablated-files.zip"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0023_prepare_0"
TEMP = Path("/tmp/study_input_batch") / STUDY
CLINICAL = SOURCE / "author/USC CGI DLBCL Data_Final.xlsx"
SUPPLEMENT = SOURCE / "advancesADV2022007493-suppl1.pdf"
AUTHOR_URL = "https://github.com/arcolombo/singleCell_DLBCL"


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE) or path.is_relative_to(JOB):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, images: pd.DataFrame, **details: object) -> None:
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, updated_at=datetime.now(timezone.utc).isoformat(),
        expected_images=len(accepted), completed_images=len(images),
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        gaps=metadata_gaps, **details))


def clinical_rows() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Expand the author's explicit comma-separated Run ID lists, not patient guesses.

    The final worksheet row repeats headers. TMA has merged cells and is filled
    within the table. Supplement p11 identifies these multi-ROI cases as replicates;
    grouping is retained at diagnostic case/TMA resolution. Exact biological
    specimen identity remains unresolved without donor block/accession evidence.
    """
    workbook = openpyxl.load_workbook(CLINICAL, read_only=True, data_only=True)
    sheet = workbook.active
    values = list(sheet.iter_rows(max_col=70, values_only=True))
    workbook.close()
    headers = values[0]
    rows, by_run = [], {}
    tma = None
    for row_number, values_row in enumerate(values[1:], 2):
        row = dict(zip(headers, values_row))
        if not isinstance(row.get("Patient ID"), (int, float)):
            continue
        tma = row["TMA"] if row["TMA"] else tma
        row["TMA"] = tma
        row["worksheet_row"] = row_number
        runs = [part.strip() for part in str(row["Run ID"]).split(",")]
        row["run_ids"] = runs
        row["source_subject_id"] = str(int(row["Patient ID"]))
        row["sample_id"] = "diagnostic_" + tma.replace(" ", "") + "_case_" + row["source_subject_id"]
        for run in runs:
            assert run not in by_run, f"Conflicting clinical Run ID: {run}"
            by_run[run] = row
        rows.append(row)
    return pd.DataFrame(rows), by_run


def verify_header(path: Path, height: int, width: int) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (len(kept), max(height, width), max(height, width))
        assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("uint16")
        root = ET.fromstring(tif.ome_metadata)
        names = [element.attrib["Name"] for element in root.iter() if element.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()


@contextmanager
def roi_reader(roi: dict):
    """Extract one ROI's planes to local disk and expose explicit scalar page paths.

    Source channel indices follow the existing case-sensitive filename order, not
    ZIP member order or an asserted instrument index. Shared tiff_planes requires
    channels=(0,...,0), as each source file contains one scalar page. NAS cache
    members are reused when already complete; only this task's local copies are
    removed. No production reader change is required.
    """
    directory = TEMP / roi["roi_id"]
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            entries = {Path(item["path"]).name: item for item in roi["files"]}
            for channel in channels:
                member = entries[channel["file_name"]]
                cached = CACHE / "extracted" / member["path"]
                if cached.exists() and cached.stat().st_size == member["size"]:
                    paths.append(cached)
                    continue
                path = directory / channel["file_name"]
                if not path.exists() or path.stat().st_size != member["size"]:
                    temporary = path.with_name(path.name + ".extracting")
                    with archive.open(member["path"]) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    assert temporary.stat().st_size == member["size"]
                    temporary.replace(path)
                paths.append(path)
        source = SourceSpec(roi["roi_id"], PANEL, "tiff_planes", paths[0],
                            tuple(0 for _ in paths), paths=tuple(paths),
                            pixel_size_um=(1.0, 1.0), source_revision=manifest["release_doi"])
        with open_source(source) as reader:
            assert reader.shape == (35, *proofs[roi["roi_id"]]["shape"])
            assert reader.dtype == np.dtype("uint16")
            yield reader
    finally:
        shutil.rmtree(directory)


def make_overview(reader, image_id: str) -> None:
    """Show the complete source acquisition with DNA, B/T-cell and stromal signals."""
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = reader.shape[-2:]
    names = ["DNA", "CD20", "CD3", "Vimentin"]
    figure, axes = plt.subplots(1, len(names), figsize=(14, 4))
    for name, axis in zip(names, axes):
        index = next(i for i, channel in enumerate(channels) if channel["source_label"] == name)
        values = reader.read_window([index], (0, 0, width, height))[0]
        stride = max(1, max(height, width) // 650)
        values = values[::stride, ::stride]
        positive = values[values > 0]
        vmax = float(np.percentile(positive, 99)) if len(positive) else 1.0
        axis.imshow(values, cmap="gray", vmin=0, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")
    figure.suptitle(image_id + f" | full source frame {width} x {height}")
    figure.tight_layout()
    path = CACHE / "overviews" / (image_id + ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


# Reuse the complete native inventory and its version-bound acceptance proofs.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("metadata", "export", "verify", "repair"), default="metadata")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
if args.stage == "repair":
    PREPARER = "Colombo2022_IMC_DLBCL_0168_repair_1"
    JOB = JOB.parent / "0168_repair_1"
manifest = json.loads((META / "native_manifest.json").read_text())
channels = json.loads((META / "ordered_markers.json").read_text())["channels"]
assert [c["file_name"] for c in channels] == sorted(c["file_name"] for c in channels)
assert len(channels) == 35
native_stat = ARCHIVE.stat()
current_versions = json.loads((META / "current_native_versions.json").read_text())["members"]
expected_identity = current_versions[str(ARCHIVE.relative_to(NAS))]["current"]
actual_identity = file_identity(ARCHIVE)
assert all(actual_identity[key] == expected_identity[key]
           for key in ("device", "inode", "size", "mtime_ns"))
stat_binding = dict(size=native_stat.st_size, mtime_ns=native_stat.st_mtime_ns,
                    inode=native_stat.st_ino)
inventory = json.loads((NATIVE / "archive_inventory.json").read_text())
# NFS device numbers vary across remounts; retain inode, size and mtime checks.
assert {key: inventory["native_stat"][key] for key in stat_binding} == stat_binding
proofs = {}
for roi in manifest["rois"]:
    proof = json.loads((NATIVE / "proofs" / (roi["roi_id"] + ".json")).read_text())
    assert proof["roi"] == roi
    assert {key: proof["native_stat"][key] for key in stat_binding} == stat_binding
    assert proof["classification"] == ("accepted" if roi["countable"] else "validated_identity_pending")
    proofs[roi["roi_id"]] = proof
accepted = [roi for roi in manifest["rois"] if roi["countable"]]
assert len(accepted) == 35
with zipfile.ZipFile(ARCHIVE) as archive:
    actual = {item.filename: (item.file_size, item.compress_size, item.header_offset)
              for item in archive.infolist() if not item.is_dir()}
expected = {item["path"]: (item["size"], item["compressed_size"], item["header_offset"])
            for roi in manifest["rois"] for item in roi["files"]}
assert actual == expected

# Each source label has an exact study-specific S1 row; no aliases are needed.
panel = panel_from_s1(STUDY, PANEL, [c["source_label"] for c in channels], S1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 33
assert set(panel.loc[~panel.keep_channel, "source_name"]) == {"Membrane", "p.Stat3"}
panel["source_file_name"] = [c["file_name"] for c in channels]
panel["source_isotope"] = [c["isotope"] for c in channels]

# Preserve explicit author case/TMA groups pending biological specimen resolution.
clinical, by_run = clinical_rows()
run_ids = {roi["roi_id"].split("_", 1)[0] for roi in accepted}
matched = clinical[clinical.run_ids.map(lambda values: bool(set(values) & run_ids))]
sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
sample_records = []
for row in matched.to_dict("records"):
    sample = {field: None for field in sample_columns}
    sample.update(study=STUDY, sample_id=row["sample_id"],
                  patient_key="colombo2022_case_" + row["source_subject_id"],
                  source_subject_id=row["source_subject_id"], cohort_id="USC_DLBCL_IMC",
                  center_id="Los Angeles County and USC Medical Centers",
                  sample_type="diagnostic tumor tissue represented in a TMA",
                  tissue_type="tumor", preservation_method="ffpe", case_control_status="case",
                  primary_diagnosis="diffuse large B-cell lymphoma", histology="DLBCL",
                  sex_reported={"M": "male", "F": "female"}[row["Gender"]],
                  metadata_notes=f"Author clinical workbook row {row['worksheet_row']}; source Run ID list: {row['Run ID']}. "
                  "Supplement pp6,11: diagnostic DLBCL tissues and replicate ROIs from the same case in a TMA. "
                  "Provisional sample_id denotes a source case/TMA group; exact biological specimen resolution is unresolved. "
                  "Donor block, accession, sampling date and anatomical site are unavailable; biospecimen_group_id remains empty. "
                  "Cell of origin (HANS) is retained as an IHC classification in cell_of_origin_hans_reported. "
                  "Age/stage in author table have no explicit measurement timepoint; retain as source extensions. Response is source reported, not asserted best response. "
                  "Clinical workbook survival fields are not transferred because endpoint coding conflicts exist.")
    sample.update(source_run_ids=str(row["Run ID"]), author_worksheet_row=row["worksheet_row"],
                  cell_of_origin_hans_reported=row["Cell of origin (HANS)"],
                  age_reported_years=row["Age"], stage_reported=row["Stage"],
                  treatment_category_reported=row["Treatment Category"], treatment_regimen_reported=row["chemo"],
                  response_reported=row["Response"], response_category_reported=row["Response category"])
    sample_records.append(sample)
samples = pd.DataFrame(sample_records, columns=sample_columns + ["source_run_ids", "author_worksheet_row", "age_reported_years",
                       "stage_reported", "treatment_category_reported", "treatment_regimen_reported", "response_reported", "response_category_reported",
                       "cell_of_origin_hans_reported"])
unmatched = sorted(run_ids - set(by_run))
metadata_gaps = ["No clinical/sample identity map for source Run IDs: " + ", ".join(unmatched),
                 "26 provisional diagnostic case/TMA groups have confirmed patient links; exact biological specimen resolution remains unresolved without donor block/accession and collection times",
                 "Three repeat/second-part acquisition identities remain withheld under the existing scope decision",
                 "Independent review pending"]

# Record sources, connection keys, field semantics and unresolved identity coverage.
mapping_rows = []


def mapping(source: Path, table: str, fields: str, target_table: str, target: str,
            keys: str, rule: str, notes: str = "", values: str = "") -> None:
    mapping_rows.append(dict(study=STUDY, source_file=relative(source), source_sheet_or_table=table,
                             source_field=fields, target_table=target_table, target_field=target,
                             join_keys=keys, value_mapping=values, transformation_rule=rule,
                             time_origin_rule="No unstated timepoint or index-event inference",
                             conflict_rule="Keep unresolved identities empty; preserve source values without clinical recoding",
                             notes=notes))


for target in ("sample_id", "patient_key", "source_subject_id"):
    mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "TMA;Run ID;Patient ID", "sample_metadata", target,
            "Source ROI directory prefix == explicit Run ID token; split author comma lists; forward-fill merged TMA cells",
            "Retain provisional case/TMA sample_id groups and the declared source Patient ID with namespaced patient_key; exact biological specimen identity unresolved",
            f"{len(run_ids)-len(unmatched)}/35 image joins; {len(samples)} case/TMA groups; {samples.patient_key.nunique()} patients. Unmapped runs: {','.join(unmatched)}. Supplement pp6,11 supports replicate grouping but does not identify donor specimens.")
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "TMA;Run ID;Patient ID", "sample_metadata", "biospecimen_group_id",
        "Source case/TMA group", "Leave empty: no donor specimen/block/accession map establishes the same biological specimen",
        "Case/TMA grouping and replicate ROI design do not establish donor specimen identity; M01 remains unresolved for all 26 groups.")
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "Run ID;TMA;Patient ID", "image_metadata", "sample_id",
        "Source ROI prefix == Run ID token", "Join explicitly listed runs only; do not match internal 408x tokens in 418x source directory names",
        "Five unmatched runs are retained as images with sample_id empty. No patient identity inferred from filenames.")
mapping(SUPPLEMENT, "pp6,11;Tissue Microarray;TMA and replicate divergence analysis", "FFPE DLBCL tissues;replicate ROIs;diagnostic cohort", "sample_metadata",
        "sample_type;tissue_type;preservation_method;primary_diagnosis;histology;center_id;case_control_status",
        "Matched author clinical case and TMA", "Use diagnostic DLBCL tumor tissue group; multiple declared replicate ROIs share one sample_id")
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "Gender", "sample_metadata", "sex_reported",
        "Run ID -> Patient ID", "M -> male, F -> female", values=json.dumps({"M": "male", "F": "female"}))
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "Cell of origin (HANS)", "sample_metadata", "molecular_subtype",
        "Run ID -> Patient ID", "Leave empty: the source HANS IHC classification does not establish a molecular subtype")
for target, source, kind, definition in [
    ("source_run_ids", "Run ID", "string", "Author's verbatim list of replicate acquisition run identifiers"),
    ("author_worksheet_row", "worksheet row", "integer", "One-based source clinical workbook row"),
    ("cell_of_origin_hans_reported", "Cell of origin (HANS)", "string", "Verbatim author-reported cell-of-origin classification by the HANS immunohistochemistry algorithm; source GCB/NGCB labels retained"),
    ("age_reported_years", "Age", "number", "Author-reported age in years; measurement timepoint unspecified"),
    ("stage_reported", "Stage", "string", "Verbatim author-reported stage; staging system and timepoint unspecified"),
    ("treatment_category_reported", "Treatment Category", "string", "Verbatim treatment category without a collection/index timepoint"),
    ("treatment_regimen_reported", "chemo", "string", "Verbatim reported regimen/history, not an asserted index therapy"),
    ("response_reported", "Response", "string", "Verbatim response, without asserted best-response/timepoint semantics"),
    ("response_category_reported", "Response category", "string", "Verbatim author response category; not converted to CR/PR/PD")]:
    mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", source, "sample_metadata", target,
            "Run ID -> Patient ID", "Preserve source value; source unknown remains empty",
            json.dumps(dict(data_type=kind, definition=definition)))
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "TMA;Run ID;Patient ID", "sample_metadata", "cohort_id",
        "Only source rows joined to accepted image Run IDs", "Assign study-local constant USC_DLBCL_IMC to this author clinical cohort",
        "Synthetic cohort namespace for source inclusion, not a source patient or biospecimen identifier.")
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "worksheet row;Run ID;TMA;clinical field headers", "sample_metadata", "metadata_notes",
        "Explicit Run ID list -> source clinical worksheet row", "Compose notes from source row number/run list, cited supplement design, unresolved specimen resolution, HANS method, unspecified clinical timepoints and withheld conflicting survival endpoints",
        "Supplement source: " + relative(SUPPLEMENT) + "; notes are provenance and uncertainty statements, not additional clinical observations.")
mapping(META / "native_manifest.json", "rois", "roi_id;directory;countable", "image_metadata", "image_kind",
        "One accepted source ROI directory per image", "Assign roi to each complete independent published source ROI frame")
mapping(META / "ordered_markers.json", "channels", "file_name;source_label;isotope", "image_metadata", "panel_id",
        "All 35 accepted source ROIs share the same ordered measurement planes", "Assign " + PANEL + " for the Figshare 20010146 v1 ordered panel",
        "Panel namespace is synthetic; plane order and identity are independently reviewed against native files and S1.")
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "Run ID;TMA", "image_metadata", "slide_id",
        "ROI directory prefix == explicit author Run ID token", "Forward-fill merged TMA cells within the source clinical table; preserve matched TMA label verbatim; leave unmatched ROI slide_id empty",
        "TMA denotes the author slide group, not the donor specimen or block.")
for table in ("sample_metadata", "image_metadata", "panel_mapping"):
    mapping(META / "native_manifest.json", "study release", "study;release_doi", table, "study",
            "This study and accepted Figshare release", "Assign stable study namespace " + STUDY)
mapping(CLINICAL, "DLBCL TMA-our cases (cleaned)", "outcome;os;os_d;dates", "sample_metadata", "vital_status;os_time;os_event;age_at_collection_years;stage_group_at_index",
        "Run ID -> Patient ID", "Leave unfilled: outcome/OS event disagreement and unstated baseline timepoints",
        "An author row reports Diseased with os=censored; no survival endpoints or diagnosis/collection dates inferred.")
mapping(SOURCE / "author/scripts/Figure1_noFiltering_12-11_windows.Rmd", "lines 443-447,910-914", "missingSurv;ROIID;39722 rename comment", "image_metadata", "metadata_notes",
        "Exact ROI run identifiers", "Preserve source missing-clinical and first-pass provenance; source Run ID 3972 has a direct clinical workbook join",
        "Author states 3978/3979 lack clinical information, and the first pass 3972 was omitted in favor of renamed 39722. This local release contains 3972, not 39722. No substitution or image deletion.")
mapping(SOURCE / "author/scripts/original-submission-version/Erik_preprocessing_cluster_cleanup.R", "lines 27-31", "ROIID exclusions", "image_metadata", "metadata_notes",
        "Exact ROI run identifiers", "Record original analysis exclusions 4051/4054/4058 without inferring their sample identity or importing cell-level thresholds",
        "Original script gives no reason for these three ROI exclusions; clinical workbook also lacks these run identifiers. Retain published source pixels under the existing 35-ROI input scope.")
mapping(META / "native_manifest.json", "rois and files", "roi_id;directory;countable;exclusion_reason;file path", "image_metadata",
        "image_id;source_image_id;source_files;source_format;roi_id;core_id;region_annotation;split_method;split_evidence;metadata_notes",
        "Exact native ROI directory identity", "Retain full accepted acquisition frame; image_id uses unique explicit run prefix; no sample-region split",
        "38 directories: 35 original accepted images, 3 identity-pending withheld; per-ROI ilastik derived masks are not intensity channels. No author downstream cell threshold applied.")
mapping(S1, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
        "source_channel_index;marker_id;role;keep_channel;output_channel_index", "Study == Colombo2022_IMC_DLBCL; exact source labels",
        "35 ordered native measurement planes; retain 33 in source order including both DNA channels; exclude Membrane and p.Stat3; no aliases",
        "Native order is case-sensitive source TIFF basename lexical order from ordered_markers.json, not ZIP member order. Historical instrument order is not asserted.")
mapping(SUPPLEMENT, "p7;Tissue imaging and ablation", "1-µm2 spot size", "image_metadata", "pixel_size_x_um;pixel_size_y_um",
        "All accepted source ROIs", "Use 1 µm by 1 µm source ablation grid; no spatial resampling",
        "Source scalar TIFF resolution tags are not used as physical units")
mapping(META / "ordered_markers.json", "channels", "file_name;source_label;isotope", "panel_mapping", "source_name;source_file_name;source_isotope",
        "Exact file basename within each ROI directory", "One scalar page per measurement; shared SourceSpec tiff_planes channels are all 0 while panel indices enumerate source paths")
mapping(NATIVE / "archive_inventory.json", "native_stat;source;files", "source version;stat;native ROI shape;uint16 plane intensities", "image_metadata",
        "source_axes;source_dtype;source_revision;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_right_px;pad_top_px;pad_bottom_px;height_px;width_px;source_channel_count;channel_count;input_image_path;intensity_description;valid_pixels_path",
        "Exact source archive and ROI directory", "Reuse accepted native proof and current stat; preserve all decoded uint16 intensities and full ROI; center zero padding to square CYX TIFF",
        "A zero source pixel remains measured. Source masks are optional. No intensity transform, cell filtering or tissue crop.")
field_mapping = pd.DataFrame(mapping_rows)
image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=image_columns)

# Apply only the first review's metadata repairs; retain independent image evidence.
if args.stage == "repair":
    progress("repairing_metadata", images, repair_round=1, independent_review=False)
    review_job = META / "independent_review"
    review_summary = json.loads((review_job / "verification_summary.json").read_text())
    prior_stat = review_summary["source_snapshot"]
    unchanged_paths = [OUTPUT / "images.parquet", OUTPUT / "panel_mapping.csv"]
    unchanged_paths += [NAS / path for path in images.input_image_path]
    before_stat = {relative(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                   for path in unchanged_paths}
    for path in images.input_image_path:
        assert before_stat[path] == prior_stat[path], f"TIFF changed since independent review: {path}"

    # Keep the original reviewed small tables for exact repair-diff verification.
    for name in ("sample_metadata.csv", "field_mapping.csv", "review_results.csv"):
        backup = JOB / (name.removesuffix(".csv") + ".before.csv")
        if not backup.exists():
            shutil.copyfile(OUTPUT / name, backup)
    before_samples = pd.read_csv(JOB / "sample_metadata.before.csv", dtype=str, keep_default_na=False)
    repaired_samples = pd.read_csv(StringIO(samples.to_csv(index=False)), dtype=str, keep_default_na=False)
    changed_fields = {"molecular_subtype", "biospecimen_group_id", "metadata_notes"}
    unchanged_fields = [field for field in before_samples if field not in changed_fields]
    pd.testing.assert_frame_equal(before_samples[unchanged_fields], repaired_samples[unchanged_fields])
    assert repaired_samples.molecular_subtype.eq("").all()
    assert repaired_samples.biospecimen_group_id.eq("").all()
    assert repaired_samples.cell_of_origin_hans_reported.equals(before_samples.molecular_subtype)
    for sample in samples.itertuples(index=False):
        source = matched.loc[matched.sample_id.eq(sample.sample_id)].iloc[0]
        assert sample.cell_of_origin_hans_reported == source["Cell of origin (HANS)"]
        assert sample.source_run_ids == str(source["Run ID"])
    hans_rule = field_mapping.loc[field_mapping.target_field.eq("cell_of_origin_hans_reported")].iloc[0]
    hans_definition = json.loads(hans_rule.notes)
    assert hans_definition["data_type"] == "string" and "HANS" in hans_definition["definition"]
    assert hans_rule.source_field == "Cell of origin (HANS)"
    required_rules = [("sample_metadata", "cohort_id"), ("sample_metadata", "metadata_notes"),
                      ("image_metadata", "image_kind"), ("image_metadata", "panel_id"), ("image_metadata", "slide_id")]
    for table, field in required_rules:
        rule = field_mapping.loc[field_mapping.target_table.eq(table) & field_mapping.target_field.eq(field)]
        assert len(rule) == 1 and rule.source_file.ne("").all() and rule.transformation_rule.ne("").all()
    assert set(images.loc[images.sample_id.isna(), "image_id"]) == {"ROI_" + run for run in unmatched}
    validate_image_metadata(images, samples, panel, field_mapping)

    sources = json.loads((SOURCE / "metadata_sources.json").read_text())
    sources.update(sample_basis="Provisional source case/TMA groups only; exact biological specimen resolution remains unresolved without donor specimen/block/accession and sampling dates.",
                   biospecimen_group_id_policy="Empty: same donor specimen identity unconfirmed",
                   hans_field="cell_of_origin_hans_reported; verbatim HANS IHC cell-of-origin classification",
                   gaps=metadata_gaps, repair_evidence=relative(JOB / "repair_verification.json"))
    scope = json.loads((SOURCE / "scope.json").read_text())
    scope["metadata_gaps"] = metadata_gaps
    reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
    reviews = reviews[~(reviews.preparer.eq(PREPARER) & reviews.reviewer.eq(""))]
    repaired_actions = [
        dict(target="sample_metadata.cell_of_origin_hans_reported", rows=len(samples), action="Preserve HANS source values in a typed method-specific extension; molecular_subtype empty"),
        dict(target="sample_metadata.biospecimen_group_id", rows=len(samples), action="Clear unconfirmed donor-specimen grouping; retain provisional case/TMA sample_id and patient links"),
        dict(target="field_mapping", fields=[field for _, field in required_rules], action="Add exact source and naming/connection/composition rules"),
    ]
    repair_rows = []
    for check_id, table_scope, record in [("M01", "sample", "all_delivered"), ("M02", "sample", "all_delivered"), ("D01", "study", STUDY)]:
        details = dict(repair_round=1, independent_review=False, repaired_at=datetime.now(timezone.utc).isoformat(),
                       expected_images=35, exported_images=len(images), sample_rows=len(samples),
                       sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum()),
                       image_ids=images.image_id.tolist(), metadata_gaps=metadata_gaps,
                       message="Preparer metadata repair completed; independent M01/M02/D01 recheck pending. Existing independent image/panel passes preserved.")
        if check_id == "M01":
            details.update(unmatched_image_ids=["ROI_" + run for run in unmatched],
                           sample_resolution="26 provisional case/TMA groups; biological specimen identity unresolved",
                           biospecimen_group_id="Empty until donor specimen identity is established",
                           connection_keys="Source ROI prefix == explicit workbook Run ID token -> Patient ID; same source case/TMA list shares provisional sample_id",
                           sources_reviewed=[relative(CLINICAL), relative(SUPPLEMENT), relative(SOURCE / "advancesADV2022007493-suppl2.xlsx"),
                                             relative(SOURCE / "author_tree.json"), relative(review_job / "semantic_review.json")])
        else:
            details["repair_actions"] = repaired_actions
        repair_rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=table_scope, record_id=record,
                                check_id=check_id, result="unresolved", source_evidence=relative(CACHE / "metadata_sources.json"),
                                output_evidence=relative(OUTPUT / "sample_metadata.csv"), details=json.dumps(details, ensure_ascii=False)))

    # Shared save_input_tables republishes all four tables. This study-local
    # metadata repair replaces only the changed CSVs under the same study lock.
    with nullcontext():
        for name, frame in [("sample_metadata", samples), ("field_mapping", field_mapping),
                            ("review_results", pd.concat([reviews, pd.DataFrame(repair_rows)], ignore_index=True))]:
            destination = OUTPUT / (name + ".csv")
            temporary = destination.with_suffix(".csv.tmp")
            with formal_publish_lease(destination, directory=OUTPUT):
                frame.to_csv(temporary, index=False, encoding="utf-8-sig")
                temporary.replace(destination)
        write_json(CACHE / "metadata_sources.json", sources)
        write_json(CACHE / "scope.json", scope)

    after_stat = {relative(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                  for path in unchanged_paths}
    assert before_stat == after_stat
    written = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(written, repaired_samples)
    written_mapping = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(written_mapping, field_mapping.fillna(""))
    verification = dict(study=STUDY, preparer=PREPARER, independent_review=False, repair_round=1,
                        checked_sample_rows=len(samples), source_hans_values_preserved=len(samples), molecular_subtype_empty=True,
                        biospecimen_group_id_empty=True, other_sample_values_unchanged=True,
                        required_mapping_rules_verified=required_rules, repair_actions=repaired_actions,
                        image_index_and_panel_unchanged=True, tiffs_unchanged_since_independent_review=len(images),
                        preserved_file_stat=after_stat, new_pixel_or_header_checks=False,
                        publication_adapter="Study-local atomic CSV replacement under shared resource_lock; migrate if shared selective metadata publication becomes available",
                        reused_independent_review=relative(review_job / "report.json"),
                        native_proofs_reused=len(proofs), archive_inventory_members_checked=len(actual), metadata_gaps=metadata_gaps)
    write_json(JOB / "repair_verification.json", verification)
    report = dict(study=STUDY, script="scripts/data/preparation/Colombo2022_IMC_DLBCL.py", status="partial",
                  expected_images=35, exported_images=len(images), sample_rows=len(samples),
                  sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum()),
                  summary="已实际执行第一轮metadata修补：26行HANS原值迁至已定义扩展字段，molecular_subtype及无标本依据的biospecimen_group_id留空；补齐5个字段映射。35张既有TIFF、image索引及panel原样复用，自检通过，待独立复查。",
                  issues=["ROI_3978、3979、4051、4054、4058仍缺sample身份映射。", "26行仅为有患者依据的暂定病例/TMA组；确切生物学取样粒度未证实，M01继续unresolved。", "本轮整理者自检已完成，独立M01/M02/D01复查尚未执行。"],
                  controller_requests=["安排独立上下文定向复查M01、M02及D01，复用此前9项image/panel通过结论。", "核定26个暂定病例/TMA组的取样粒度处理结论；缺供体标本号/组织块/日期，当前保留M01 unresolved。",
                                       "记录共享实现迁移：save_input_tables会重写四表，本轮在study脚本内用同一resource_lock原子替换受影响CSV；后续若提供局部metadata发布接口，再迁回共享模块。"],
                  evidence_paths=[relative(OUTPUT / "sample_metadata.csv"), relative(OUTPUT / "field_mapping.csv"),
                                  relative(OUTPUT / "review_results.csv"), relative(JOB / "repair_verification.json"),
                                  relative(CACHE / "metadata_sources.json"), relative(CLINICAL), relative(review_job / "report.json")])
    write_json(JOB / "report.json", report)
    progress("metadata_repair_complete_pending_independent_review", images, repair_round=1, independent_review=False,
             tiffs_reused=len(images), repair_verification=relative(JOB / "repair_verification.json"))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(0)

save_input_tables(OUTPUT, images, samples, panel, field_mapping)
write_json(CACHE / "metadata_sources.json", dict(
    paper_doi="10.1182/bloodadvances.2022007493", fulltext=relative(META / "evidence/worker/query/q01_fulltext_xml.body"),
    raw_release_url="https://figshare.com/articles/dataset/Raw_images_for_DLBCL_samples/20010146",
    supplementary_endpoint="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9631676/supplementaryFiles",
    supplementary_pdf=relative(SUPPLEMENT), supplementary_workbook=relative(SOURCE / "advancesADV2022007493-suppl2.xlsx"),
    supplementary_workbook_result="Association/interaction result tables, not a patient/core map",
    author_repository=AUTHOR_URL, author_commit=json.loads((SOURCE / "author_tree.json").read_text())["sha"],
    author_clinical=relative(CLINICAL), native_proof_root=relative(NATIVE),
    sample_rows=len(samples), sample_linked_images=len(run_ids)-len(unmatched), patient_linked_samples=int(samples.patient_key.notna().sum()),
    unmatched_run_ids=unmatched, sample_basis="Provisional source case/TMA groups; replicate ROI design supports patient grouping, but exact biological specimen resolution remains unresolved without donor specimen/block/accession and sampling dates.",
    source_analysis_caveats="Author excludes unmapped 4051/4054/4058; author notes 3978/3979 have no clinical information and first-pass 3972 was omitted from final analysis. These do not change the original accepted input scope.",
    gaps=metadata_gaps))
output_bytes = sum(len(kept) * max(proofs[roi["roi_id"]]["shape"]) ** 2 * 2 for roi in accepted)
write_json(CACHE / "scope.json", dict(
    expected_images=35, source_directories=38, source_measurement_planes=35, retained_channels=33,
    uncompressed_output_bytes=output_bytes, archive_bytes=ARCHIVE.stat().st_size,
    max_local_roi_bytes=max(sum(item["size"] for item in roi["files"]) for roi in accepted),
    native_stat=stat_binding, accepted_roi_ids=[r["roi_id"] for r in accepted],
    withheld=[dict(roi_id=r["roi_id"], reason=r["exclusion_reason"]) for r in manifest["rois"] if not r["countable"]],
    metadata_gaps=metadata_gaps))
progress(args.stage, images, expected_output_bytes=output_bytes)

# Export and commit an index row after each admitted ROI; correct rows are reused.
if args.stage == "export":
    assert shutil.disk_usage(NAS).free > 100 * 1024 ** 3 + output_bytes
    assert shutil.disk_usage("/tmp").free > 64 * 1024 ** 3 + 200 * 1024 ** 2
    count = 0
    for roi in accepted:
        run = roi["roi_id"].split("_", 1)[0]
        image_id = "ROI_" + run
        height, width = proofs[roi["roi_id"]]["shape"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt = CACHE / "receipts" / (image_id + ".json")
        if image_id in set(images.image_id):
            verify_header(destination, height, width)
            continue
        progress("exporting", images, current_image=image_id)
        if destination.exists() and receipt.exists():
            result = json.loads(receipt.read_text())
            verify_header(destination, height, width)
        else:
            with roi_reader(roi) as reader:
                result = export_input(reader, panel, destination, tile_size=512)
                # Overview all unusually short ROIs and a representative per TMA;
                # unmatched clinical rows are also inspected for acquisition identity.
                if run in {"3964", "4044", "4182"} or height < 0.9 * width or run in unmatched:
                    make_overview(reader, image_id)
            write_json(receipt, result)
        if result["admitted"]:
            verify_header(destination, height, width)
            clinical_row = by_run.get(run)
            record = {key: None for key in image_columns}
            record.update(study=STUDY, image_id=image_id,
                          sample_id=clinical_row["sample_id"] if clinical_row else None, panel_id=PANEL,
                          image_kind="roi", source_image_id=roi["directory"],
                          source_files=[relative(ARCHIVE)], source_format="zip/tiff_planes", source_axes="CYX",
                          source_dtype="uint16", source_revision=manifest["release_doi"],
                          slide_id=clinical_row["TMA"] if clinical_row else None,
                          roi_id=run, region_annotation=roi["roi_id"],
                          source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                          pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                          input_image_path=relative(destination),
                          intensity_description="Native published ablated uint16 TIFF intensities; no added normalization or transformation",
                          split_method="already_independent", split_evidence=relative(META / "native_manifest.json"),
                          metadata_notes="Complete published source ROI retained, including all background and measured zeros. "
                          + (f"Explicit Run ID join to author workbook row {clinical_row['worksheet_row']}; replicate diagnostic tumor group." if clinical_row
                             else "No source clinical/sample mapping. Tissue/case identity unconfirmed; image retained within existing accepted scope."))
            if run == "3972":
                record["metadata_notes"] += " Author Figure1 source comment: first pass 3972 omitted from final cell analysis in favor of renamed 39722; 39722 is absent from this raw release."
            if run in {"4051", "4054", "4058"}:
                record["metadata_notes"] += " Original author preprocessing script excludes this ROI without a stated reason; original accepted input scope retained."
            for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"):
                record[key] = result[key]
            if result["valid_pixels"]:
                record["valid_pixels_path"] = relative(destination.with_name(result["valid_pixels"]))
            images = pd.DataFrame([record]) if images.empty else pd.concat([images, pd.DataFrame([record])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, field_mapping)
        progress("exporting", images, last_completed_image=image_id)
        print(json.dumps(dict(image_id=image_id, admitted=result["admitted"], completed_images=len(images))), flush=True)
        count += 1
        if args.limit and count >= args.limit:
            break
    if TEMP.exists() and not any(TEMP.iterdir()):
        TEMP.rmdir()

# Preparer verification covers all TIFF headers and declared representative pixels.
if args.stage == "verify":
    pixel_checks = []
    padding_checks = []
    by_image = images.set_index("image_id")
    for row in images.itertuples(index=False):
        verify_header(NAS / row.input_image_path, row.source_y1_px, row.source_x1_px)
    for roi in accepted:
        run = roi["roi_id"].split("_", 1)[0]
        image_id = "ROI_" + run
        if image_id not in by_image.index or run not in {"3964", "3972", "4044", "4057", "4062", "4182"}:
            continue
        row = by_image.loc[image_id]
        height, width = proofs[roi["roi_id"]]["shape"]
        # Read ZIP-member TIFF windows directly: verification does not extract
        # another set of source images or retain whole-image arrays in memory.
        check_windows = [(width // 2, height // 2, width // 2 + 32, height // 2 + 32), (0, 0, 32, 32)]
        with zipfile.ZipFile(ARCHIVE) as archive, tifffile.TiffFile(NAS / row.input_image_path) as tif:
            for channel in kept.itertuples(index=False):
                member = roi["directory"] + "/" + channels[channel.source_channel_index]["file_name"]
                with archive.open(member) as handle, tifffile.TiffFile(handle) as source_tif:
                    for window in check_windows:
                        native = read_tiff_window(source_tif, 0, window)
                        x0, y0, x1, y1 = window
                        saved_window = (x0 + row.pad_left_px, y0 + row.pad_top_px, x1 + row.pad_left_px, y1 + row.pad_top_px)
                        saved = read_tiff_window(tif, channel.output_channel_index, saved_window)
                        assert saved.dtype == native.dtype and np.array_equal(saved, native), (image_id, channel.source_name)
            for window in check_windows:
                pixel_checks.append(dict(image_id=image_id, source_window=list(window), all_33_channels_equal=True))
            for start, end in [(0, int(row.pad_top_px)), (int(row.pad_top_px + height), int(row.height_px))]:
                if start < end:
                    assert not any(read_tiff_window(tif, i, (0, start, int(row.width_px), end)).any() for i in range(len(kept)))
            for start, end in [(0, int(row.pad_left_px)), (int(row.pad_left_px + width), int(row.width_px))]:
                if start < end:
                    assert not any(read_tiff_window(tif, i, (start, 0, end, int(row.height_px))).any() for i in range(len(kept)))
            padding_checks.append(dict(image_id=image_id, all_retained_channels_padding_zero=True,
                                       padding_ltrb=[int(row.pad_left_px), int(row.pad_top_px), int(row.pad_right_px), int(row.pad_bottom_px)]))
    write_json(CACHE / "preparer_verification.json", dict(
        preparer=PREPARER, independent_review=False, headers_checked=len(images), image_fields=len(images.columns),
        native_proofs_reused=len(proofs), archive_inventory_members_checked=len(actual),
        sample_linked_images=int(images.sample_id.notna().sum()), sample_rows=len(samples),
        patient_linked_samples=int(samples.patient_key.notna().sum()), unique_patients=int(samples.patient_key.nunique()),
        sample_rows_rebuilt_from_author_workbook=True, pixel_checks=pixel_checks, padding_checks=padding_checks,
        overview_files=[relative(path) for path in sorted((CACHE / "overviews").glob("*.jpg"))],
        metadata_gaps=metadata_gaps))
    if TEMP.exists() and not any(TEMP.iterdir()):
        TEMP.rmdir()

# Only preparer statements are written here; independent D01 acceptance stays pending.
review_path = OUTPUT / "review_results.csv"
reviews = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.DataFrame(columns=pd.read_csv(TEMPLATES / "review_results.csv").columns)
reviews = reviews[~(reviews.preparer.eq(PREPARER) & reviews.reviewer.eq(""))]
review_rows = []
for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples(index=False):
    details = dict(preparer_delivery=True, independent_review=False, exported_images=len(images), expected_images=35,
                   sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                   patient_linked_samples=int(samples.patient_key.notna().sum()),
                   message="Preparer delivery; independent verification and acceptance pending")
    if check.check_id == "M01":
        details.update(unmatched_run_ids=unmatched, diagnostic_group_basis="Explicit author run list + supplement replicate ROI design; donor blocks unavailable", metadata_gaps=metadata_gaps)
    if check.check_id == "S01":
        details.update(source_directories=38, withheld=3, no_new_image_exclusions="Only all-zero admitted markers would exclude an accepted image")
    review_rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                            record_id=STUDY if check.scope == "study" else "all_delivered",
                            check_id=check.check_id, result="unresolved", source_evidence=relative(CACHE / "metadata_sources.json"),
                            output_evidence=relative(OUTPUT / "images.parquet"), details=json.dumps(details)))
with formal_publish_lease(review_path, directory=review_path.parent):
    pd.concat([reviews, pd.DataFrame(review_rows)], ignore_index=True).to_csv(review_path, index=False, encoding="utf-8-sig")
progress("prepared_pending_independent_review" if len(images) == len(accepted) else args.stage, images)
print(json.dumps(dict(study=STUDY, expected_images=35, exported_images=len(images), sample_rows=len(samples),
                      sample_linked_images=int(images.sample_id.notna().sum()),
                      patient_linked_samples=int(samples.patient_key.notna().sum())), ensure_ascii=False), flush=True)
