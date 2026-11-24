"""Prepare the 78 native Liu FL ROIs from Zenodo 7086346 v1.

Inputs: complete RAR, accepted native proofs, per-ROI summaries, S1, and the
paper's supplementary clinical table. Outputs: native uint16 CYX input TIFFs,
39-field image index, specimen/clinical metadata, and preparer self-checks.
Shared helpers stage each image locally and acquire the formal NAS lease only
for publication. Run verify while extracted sources exist; it removes own cache.
Patient rows are read from NAS; unresolved paired-sample identities stay empty.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Liu2022_IMC_FollicularLymphoma"
PANEL = "liu_zenodo7086346_v1"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
PROCESSED_EVIDENCE = OUTPUT / "preparation_evidence"
SOURCE_INVESTIGATION = PROCESSED_EVIDENCE / "source_investigation.json"
PROJECT = Path(__file__).resolve().parents[3]
CACHE = PROCESSED_EVIDENCE
JOB = PROJECT / "cache/study_preparation" / STUDY / "jobs/0217_prepare_0"
TEMP = Path("/tmp/study_input_batch") / STUDY
ARCHIVE = NAS / "raw/public" / STUDY / "public_original/raw_data.rar"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
ARCHIVE_INVENTORY = META / "rar_member_inventory.json"
NATIVE_METADATA = META / "native_metadata"
SOURCE_EVIDENCE = META / "source_evidence"
SUPPLEMENT = SOURCE_EVIDENCE / "fl_supplement.docx"
PAPER = SOURCE_EVIDENCE / "fl_epmc_full.xml"
POD24_METHODS_NOTE = "Methods describes 13 paired FL samples at diagnosis and POD24; this is not a count of positive clinical POD24 flags."


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, **details: object) -> None:
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, updated_at=datetime.now(timezone.utc).isoformat(),
        expected_images=len(records), completed_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        patient_linked_images=int(images.sample_id.isin(samples.loc[samples.patient_key.notna(), "sample_id"]).sum()),
        image_index=relative(OUTPUT / "images.parquet"),
        export_checkpoint=relative(CACHE / "export_state.json"),
        phase="repair" if args.stage == "repair" else "prepare",
        scientific_repair_round=1 if args.stage == "repair" else 0,
        gaps=gaps, **details))


def clinical_metadata() -> tuple[pd.DataFrame, list[dict]]:
    """Join the explicit clinical Number token to the source specimen directory.

    Each of 39 directories is one released lymph-node section with two ROIs.
    Only exact Number matches establish patient links. Neither numeric suffix
    matching nor replacing the A/B prefix establishes a paired patient.
    """
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with formal_read_lease(SUPPLEMENT), zipfile.ZipFile(SUPPLEMENT) as package:
        root = ET.fromstring(package.read("word/document.xml"))
    table = root.findall(".//w:tbl", ns)[1]
    rows = [["".join(cell.itertext()).strip() for cell in row.findall("w:tc", ns)]
            for row in table.findall("w:tr", ns)]
    clinical = pd.DataFrame([row for row in rows[2:] if len(row) == 9], columns=rows[1])
    assert len(clinical) == 26 and clinical.iloc[:, 0].is_unique
    if args.stage != "repair":
        clinical.to_csv(CACHE / "source_clinical_table.csv", index=False, encoding="utf-8-sig")
    clinical_by_number = {row[0]: row for row in clinical.itertuples(index=False, name=None)}
    clinical_pod24_count = int(clinical.iloc[:, 8].astype(int).sum())
    columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
    result = []
    for specimen in sorted({item["physical_image_id"].split("::")[1].split("/")[0] for item in records}):
        row = dict.fromkeys(columns, None)
        number = specimen.split("_", 1)[0]
        row.update(study=STUDY, sample_id=specimen, source_sample_id=specimen,
                   sample_type="lymph node tissue section", tissue_type="tumor",
                   anatomical_site="lymph node", preservation_method="ffpe",
                   primary_diagnosis="Follicular lymphoma")
        if number in clinical_by_number:
            values = clinical_by_number[number]
            row.update(patient_key=STUDY + ":" + number, source_subject_id=number,
                       age_at_diagnosis_years=int(values[2]), sex_reported=values[1],
                       index_event="diagnosis", grade={"0": "1-2", "1": "3A", "2": "3B"}[values[3]],
                       stage_group_at_index=values[4].replace("Ⅲ", "III").replace(" ", ""),
                       stage_system="Lugano", index_regimen=values[6],
                       flipi2_at_diagnosis=int(values[5]), induction_response_reported=values[7],
                       pod24_reported=int(values[8]), source_lugano_stage=values[4],
                       metadata_notes="Exact source directory Number token = supplementary clinical Number. "
                       "Clinical fields refer to diagnosis/induction; collection timepoint and paired specimen map unresolved. "
                       f"Source clinical POD24 flags total {clinical_pod24_count}. {POD24_METHODS_NOTE} Values preserved.")
        else:
            row.update(metadata_notes="Released specimen section confirmed by source directory; no exact clinical Number match. "
                       "Patient identity, A/B pairing, collection timepoint and clinical values unresolved.")
        result.append(row)
    frame = pd.DataFrame(result)
    mappings = []

    def mapping(source: str, target: str, rule: str, *, target_table: str = "sample_metadata",
                source_file: str = relative(SUPPLEMENT), notes: str = "", value_mapping: str = "") -> None:
        mappings.append(dict(study=STUDY, source_file=source_file,
            source_sheet_or_table=("Source archive directory / per-ROI summary" if source_file == relative(ARCHIVE)
                                   else "Patient characteristics and sample preparation" if source == "Patient characteristics and sample preparation"
                                   else "Source investigation record" if source_file == relative(SOURCE_INVESTIGATION)
                                   else "word/document.xml table[1], clinical Number table"),
            source_field=source, target_table=target_table, target_field=target,
            join_keys="source specimen directory token before first underscore = Number; exact, case-sensitive",
            value_mapping=value_mapping, transformation_rule=rule,
            time_origin_rule="Clinical table diagnosis/first-line induction; specimen collection timepoint unassigned",
            conflict_rule="Unmatched directories remain unlinked; never infer A/B pairing", notes=notes))

    for target in ("sample_id", "source_sample_id"):
        mapping("raw_data/<specimen>/<ROI>", target,
                "One biological section per source specimen directory; both original ROIs share this ID. "
                "39 directories agree with the author's 39 lymph-node sections.", source_file=relative(ARCHIVE))
    for target in ("patient_key", "source_subject_id"):
        mapping("Number", target, "Use only exact clinical Number matches; patient_key is study:Number.",
                notes="26/39 sample rows, 52/78 images linked to a clinical patient; 13 specimens/26 images unlinked.")
    fields = [(clinical.columns[1], "sex_reported", "Preserve female/male values"),
              (clinical.columns[2], "age_at_diagnosis_years", "Integer years at diagnosis; not age at collection"),
              (clinical.columns[3], "grade", 'Map 0 to 1-2, 1 to 3A, 2 to 3B'),
              (clinical.columns[4], "stage_group_at_index", "Replace Roman numeral Ⅲ with III and remove spaces"),
              (clinical.columns[6], "index_regimen", "Preserve source regimen spelling"),
              (clinical.columns[5], "flipi2_at_diagnosis", "Integer source clinical risk score"),
              (clinical.columns[7], "induction_response_reported", "Preserve first-line response; not relabeled best overall response"),
              (clinical.columns[8], "pod24_reported", "Preserve source binary report; not a survival event. "
               "Clinical table has 14 positive flags. " + POD24_METHODS_NOTE + " No source value corrected"),
              (clinical.columns[4], "source_lugano_stage", "Preserve unmodified source text")]
    extensions = {"flipi2_at_diagnosis": ("integer", "Source FLIPI-2 clinical score at diagnosis"),
                  "induction_response_reported": ("string", "Source response to first-line induction"),
                  "pod24_reported": ("integer", "Source POD24 indicator, 1=yes and 0=no"),
                  "source_lugano_stage": ("string", "Unmodified source Lugano stage")}
    for source, target, rule in fields:
        notes = json.dumps(dict(zip(("data_type", "definition"), extensions[target]))) if target in extensions else ""
        mapping(source, target, rule, notes=notes)
    for target, rule in {
        "index_event": "Diagnosis is the index for patient-table grade, stage and induction fields",
        "stage_system": "Source column is Lugano stage",
    }.items():
        mapping("Clinical Number table", target, rule)
    for target in ("sample_type", "tissue_type", "anatomical_site", "preservation_method", "primary_diagnosis"):
        mapping("Patient characteristics and sample preparation", target,
                "Source methods: 39 paraffin-embedded lymph-node sections from FL patients; selected malignant follicles.")
    mapping("raw_data/<specimen>/<ROI>", "sample_id", "Both ROIs map to their parent biological section directory",
            target_table="image_metadata", source_file=relative(ARCHIVE))
    mapping("Per-ROI summary Page/Channel/Label and scalar TIFF filename", "panel_id",
            "Preserve all 41 source channels in own summary Page order; physical scalar TIFF page is always zero.",
            target_table="image_metadata", source_file=relative(ARCHIVE))
    mapping("TIFF XResolution, ResolutionUnit", "pixel_size_x_um", "25400 pixels/inch => 25400/25400 = 1 micrometre/pixel; verify every source plane",
            target_table="image_metadata", source_file=relative(ARCHIVE))
    mapping("TIFF YResolution, ResolutionUnit", "pixel_size_y_um", "25400 pixels/inch => 25400/25400 = 1 micrometre/pixel; verify every source plane",
            target_table="image_metadata", source_file=relative(ARCHIVE))
    mapping("Local inventory, per-ROI summaries, Zenodo, paper supplement, author repository search", "metadata_notes",
            "Source investigation: Zenodo has only raw_data.rar; archive contains scalar TIFFs and summaries only. "
            "Supplement contains clinical Number table but no B-directory patient map or sample timepoint map. "
            "Paper references FlowSOM package; no study author sample/core repository identified in DOI/release searches.",
            source_file=relative(SOURCE_INVESTIGATION),
            notes="Coverage is recorded per sample and in source_investigation.json. No author contact.")
    return frame, mappings


def verify_header(path: Path, height: int, width: int) -> None:
    with formal_read_lease(path), tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (len(kept), max(height, width), max(height, width))
        assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("uint16")
        root = ET.fromstring(tif.ome_metadata)
        names = [node.attrib["Name"] for node in root.iter() if node.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()
        pixels = next(node for node in root.iter() if node.tag.endswith("}Pixels"))
        assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0


def extract_sources() -> None:
    """Extract the accepted, source-bound complete archive once to own local cache."""
    TEMP.mkdir(parents=True, exist_ok=True)
    members = inventory["members"]
    needed = [member for member in members if not (TEMP / member["name"]).exists()
              or (TEMP / member["name"]).stat().st_size != member["size"]]
    if not needed:
        return
    local_bytes = sum(member["size"] for member in needed) + ARCHIVE.stat().st_size
    output_bytes = sum(max(geometry(item).values()) ** 2 * len(kept) * 2 for item in records)
    assert shutil.disk_usage(TEMP).free >= local_bytes + 64 * 1024**3, "Local reserve below 64 GiB"
    assert shutil.disk_usage(NAS).free >= output_bytes + 100 * 1024**3, "NAS reserve below 100 GiB"
    progress("extracting", extraction_bytes=local_bytes, uncompressed_output_estimate=output_bytes)
    local_archive = TEMP / ARCHIVE.name
    if not local_archive.exists() or local_archive.stat().st_size != ARCHIVE.stat().st_size:
        with formal_read_lease(ARCHIVE):
            shutil.copyfile(ARCHIVE, local_archive)
    subprocess.run(["unrar", "x", "-mt4", "-idq", "-y", "-o+", str(local_archive), str(TEMP) + "/"], check=True)
    for member in members:
        assert (TEMP / member["name"]).stat().st_size == member["size"], member["name"]


def geometry(item: dict) -> dict[str, int]:
    shape = item["native_proof"]["native_read_evidence"]["channel_evidence"][0]["read"]["geometry"]
    return {"height": shape["height"], "width": shape["width"]}


@contextmanager
def roi_reader(item: dict):
    """Use existing summary-bound file order; scalar planes have physical page 0."""
    identity = item["physical_image_id"].split("::")[1]
    metadata_path = NATIVE_METADATA / (identity.replace("/", "__") + ".json")
    with formal_read_lease(metadata_path):
        metadata = json.loads(metadata_path.read_text())
    channels = metadata["ordered_source_channels"]
    paths = tuple(TEMP / channel["name"] for channel in channels)
    summary_path = TEMP / item["native_proof"]["binding"]["source_summary"]["name"]
    summary = pd.read_csv(summary_path, sep="\t", dtype=str)
    assert summary.Page.astype(int).tolist() == list(range(41))
    assert summary.Label.tolist() == panel.source_name.tolist() == [c["native_label"] for c in channels]
    for channel, path in zip(channels, paths):
        assert path.stat().st_size == channel["size"]
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            assert int(tags["ResolutionUnit"].value) == 2, "Source TIFF resolution unit must be inch"
            for key in ("XResolution", "YResolution"):
                numerator, denominator = tags[key].value
                assert 25400.0 * denominator / numerator == 1.0, f"Source calibration differs: {path}"
    source = SourceSpec(identity, PANEL, "tiff_planes", paths[0], tuple(0 for _ in paths),
                        paths=paths, pixel_size_um=(1.0, 1.0), source_revision="Zenodo7086346_v1")
    with open_source(source) as reader:
        size = geometry(item)
        assert reader.shape == (41, size["height"], size["width"])
        assert reader.dtype == np.dtype("uint16")
        yield reader


def overview(reader, image_id: str) -> None:
    """Display full acquisition DNA, B/T cells and stroma without changing inputs."""
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    h, w = reader.shape[-2:]
    for axis, label in zip(axes, ("DNA2", "142Nd_CD19", "170Er_CD3", "143Nd_Vimentin")):
        index = panel.source_name.tolist().index(label)
        plane = reader.read_window([index], (0, 0, w, h))[0]
        stride = max(1, max(h, w) // 700)
        view = plane[::stride, ::stride]
        positive = view[view > 0]
        axis.imshow(view, cmap="gray", vmin=0, vmax=np.percentile(positive, 99) if positive.size else 1)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(f"{image_id} | complete source frame {w} x {h}")
    figure.tight_layout()
    path = CACHE / "overviews" / (image_id + ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    plt.close(figure)


def save_reviews() -> None:
    """Save preparer evidence; independent reviewer and D01 acceptance stay pending."""
    counts = dict(expected_images=len(records), exported_images=len(images), sample_rows=len(samples),
                  sample_linked_images=int(images.sample_id.notna().sum()),
                  patient_linked_samples=int(samples.patient_key.notna().sum()))
    evidence = [relative(META / "native_verification.json"), relative(SOURCE_INVESTIGATION)]
    evidence.extend(relative(CACHE / name) for name in
                    ("public_source_recheck_0217.json", "clinical_self_check.json",
                     "visual_self_check.json", "pixel_self_check.json")
                    if (CACHE / name).exists())
    rows = []
    for check, result, detail in (
        ("S01", "pass", "Preparer: complete source-bound archive inventory; 78 native ROIs in 39 sections; all admitted marker images retained."),
        ("I01", "pass", "Preparer: unique source/image paths, complete 39-field index, per-image checkpoint."),
        ("I02", "pass", "Preparer: all image sample/panel foreign keys resolve without row multiplication."),
        ("I03", "pass", "Preparer: full source frame, centered zero padding, native uint16; TIFF calibration 25400 pixels/inch = 1 micrometre/pixel."),
        ("I04", "pass", "Preparer: original independent non-TMA ROIs; parent directory specifies one section; no multi-sample split."),
        ("P01", "pass", "Preparer: own source summary order and exact study S1 labels checked for each ROI."),
        ("P02", "pass", "Preparer: 41 source channels, 37 kept = 35 admitted markers + DNA1/DNA2; 4 technical exclusions."),
        ("P03", "pass", "Preparer: every completed TIFF reopened for uint16 CYX shape and all output channel names."),
        ("I05", "pass" if (CACHE / "pixel_self_check.json").exists() else "unresolved", "Preparer pixel self-check only; details in pixel_self_check.json when present. Independent review pending."),
        ("M01", "unresolved", "39 specimen rows; 78/78 image-to-sample links; 26/39 exact clinical patient links. "
         "13 specimens (26 images) lack patient map. Collection timepoints and A/B paired map unresolved."),
        ("M02", "unresolved", "Preparer: nonempty clinical fields rebuilt from original DOCX table. "
         "Source POD24 flags have 14 positives; Methods describes 13 paired FL samples. Independent clinical review pending."),
        ("D01", "unresolved", "Prepared image delivery and metadata scope are separate; independent reviewer/controller acceptance pending."),
    ):
        if len(images) < len(records) and (check.startswith("I") or check in {"S01", "P03"}):
            result = "unresolved"
            detail = "Image preparation incomplete; completed scope only. " + detail
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study" if check in {"S01", "D01"} else "sample" if check.startswith("M") else "panel" if check in {"P01", "P02"} else "image",
                         record_id=STUDY, check_id=check, result=result,
                         source_evidence=json.dumps(evidence), output_evidence=relative(OUTPUT),
                         details=json.dumps(dict(preparer_self_check=True, independent_review_pending=True,
                                                 counts=counts, description=detail), ensure_ascii=False)))
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        pd.DataFrame(rows).to_csv(OUTPUT / "review_results.csv", index=False, encoding="utf-8-sig")


def record_repair() -> None:
    """Record preparer changes and revoke the stale metadata-bound D01 result."""
    checks = (("M02", "sample", "Clinical values retained; POD24 interpretation corrected in 26 linked sample notes, "
               "one field mapping row, and source evidence. Independent clinical recheck pending."),
              ("D01", "study", "Sample and field mapping versions changed; prior input binding is stale. "
               "Independent reviewer must recheck M02 and bind the confirmed 52-image scope again."))
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                 record_id=STUDY, check_id=check, result="unresolved",
                 source_evidence=json.dumps([relative(SUPPLEMENT), relative(PAPER),
                                             relative(SOURCE_INVESTIGATION)]),
                 output_evidence=json.dumps([relative(OUTPUT / "sample_metadata.csv"),
                                             relative(OUTPUT / "field_mapping.csv")]),
                 details=json.dumps(dict(description=description, scientific_repair_round=1,
                                         independent_review_pending=True), ensure_ascii=False))
            for check, scope, description in checks]
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        path = OUTPUT / "review_results.csv"
        previous = pd.read_csv(path, keep_default_na=False)
        if ((previous.preparer == PREPARER) & (previous.check_id == "D01") &
                (previous.result == "unresolved")).any():
            return
        temporary = path.with_name(path.name + ".partial")
        pd.concat([previous, pd.DataFrame(rows)], ignore_index=True).to_csv(
            temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


# Reuse original acceptance proofs and source statistics without rehashing raw data.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("metadata", "export", "verify", "repair"), default="metadata")
parser.add_argument("--keep-temporary", action="store_true")
parser.add_argument("--job", type=Path, default=JOB)
args = parser.parse_args()
JOB = args.job
PREPARER = STUDY + "_" + JOB.name
CACHE.mkdir(parents=True, exist_ok=True)
with formal_read_lease(META):
    native = json.loads((META / "native_verification.json").read_text())
records = native["records"]
assert native["scope_complete"] and len(records) == 78 and not native["gaps"]
with formal_read_lease(ARCHIVE_INVENTORY):
    inventory = json.loads(ARCHIVE_INVENTORY.read_text())
source_stat = ARCHIVE.stat()
for item in records:
    proof = item["native_proof"]
    assert proof["complete"] and proof["source_stat_bound"] and proof["native_order_verified"]
    assert proof["source_file_stat"]["size"] == source_stat.st_size
    assert proof["source_file_stat"]["mtime_ns"] == source_stat.st_mtime_ns
source_names = [channel["source_label"] for channel in records[0]["source_channel"]]
assert all([c["source_label"] for c in item["source_channel"]] == source_names for item in records)
panel = panel_from_s1(STUDY, PANEL, source_names, S1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(panel) == 41 and len(kept) == 37
samples, mappings = clinical_metadata()
field_mapping = pd.DataFrame(mappings, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)
gaps = ["13 specimen rows / 26 images lack patient mapping; A/B paired map absent",
        "Collection timepoint map unresolved",
        "Independent input review pending"]
investigation = dict(
    study=STUDY, release="https://zenodo.org/records/7086346", paper="https://pmc.ncbi.nlm.nih.gov/articles/PMC9396877/",
    supplement_url="https://static-content.springer.com/esm/art%3A10.1186%2Fs13045-022-01326-z/MediaObjects/13045_2022_1326_MOESM1_ESM.docx",
    supplement_source=relative(SUPPLEMENT), clinical_table="word/document.xml second table, Number clinical table",
    source_inventory=relative(ARCHIVE_INVENTORY), member_count=len(inventory["members"]),
    uncompressed_source_bytes=sum(member["size"] for member in inventory["members"]),
    native_proofs_reused=len(records), archive_size=source_stat.st_size, archive_mtime_ns=source_stat.st_mtime_ns,
    pixel_size_um=[1.0, 1.0], pixel_calibration_basis="Source TIFF X/YResolution=25400 pixels/inch, ResolutionUnit=2; every plane checked at export",
    specimen_basis="39 released section directories, two original independent ROIs each; matches author Methods 39 LN sections",
    join="source directory token before first underscore exactly equals clinical table Number; no A/B inference",
    author_repository_search=dict(queries=['"s13045-022-01326-z" "github"', '"Liu" "7086346" "sample"'],
                                  outcome="No study-specific author sample/core map found; linked FlowSOM is a method package"),
    sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
    source_clinical_pod24_positive_count=int(samples.pod24_reported.fillna(0).sum()),
    methods_pod24_paired_samples=13,
    sample_linked_images=78, patient_linked_images=52, gaps=gaps)
with formal_publish_lease(SOURCE_INVESTIGATION):
    write_json(SOURCE_INVESTIGATION, investigation)
with formal_read_lease(OUTPUT):
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns)
state_path = CACHE / "export_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
if args.stage == "repair":
    assert len(images) == len(records), "Repair requires the existing complete image index"
    progress("repairing_metadata", completed_scope="78 existing image records; 39 sample rows")
save_input_tables(OUTPUT, images, samples, panel, field_mapping)
if args.stage == "repair":
    check_path = CACHE / "clinical_self_check.json"
    with formal_read_lease(check_path):
        check = json.loads(check_path.read_text())
    check["result"] = "nonempty clinical values reproduce source; POD24 paired-sample wording corrected; patient and timepoint map unresolved"
    with formal_publish_lease(check_path):
        write_json(check_path, check)
    record_repair()
    progress("repair_complete", repaired_sample_notes=26, repaired_field_mapping_rows=1,
             reused_tiffs=len(images), independent_review_pending=True)
else:
    progress("metadata_ready")

if args.stage == "export":
    try:
        extract_sources()
        # Every completion is published with its real index before proceeding.
        for position, item in enumerate(records):
            identity = item["physical_image_id"].split("::")[1]
            image_id = identity.replace("/", "__")
            size = geometry(item)
            destination = OUTPUT / "images" / image_id / "input.tiff"
            if image_id in set(images.image_id):
                verify_header(destination, **size)
                continue
            progress("exporting", current_image=image_id)
            with roi_reader(item) as reader:
                result = export_input(reader, panel, destination)
                if position in {0, 26, 52}:
                    overview(reader, image_id)
            state[image_id] = result
            write_json(state_path, state)
            if result["admitted"]:
                verify_header(destination, **size)
                row = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
                row.update(study=STUDY, image_id=image_id, sample_id=identity.split("/")[0], panel_id=PANEL,
                    image_kind="roi", source_image_id=item["physical_image_id"], source_files=[relative(ARCHIVE)],
                    source_format="tiff_planes", source_axes="YX",
                    source_dtype="uint16", source_revision="Zenodo7086346_v1",
                    pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                    roi_id=identity.split("/")[1], source_x0_px=0, source_y0_px=0,
                    source_x1_px=size["width"], source_y1_px=size["height"],
                    input_image_path=relative(destination),
                    intensity_description="Native uint16 author-exported IMC TIFF intensities, unchanged; source viewer thresholding described in supplement",
                    split_method="already_independent", split_evidence=relative(META / "native_verification.json"),
                    metadata_notes="Source member prefix raw_data/" + identity + "; no specimen timepoint/paired map; TIFF source calibration 1 micrometre/pixel")
                row.update({key: result[key] for key in ("height_px", "width_px", "pad_left_px", "pad_top_px",
                    "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count")})
                images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
                save_input_tables(OUTPUT, images, samples, panel, field_mapping)
            progress("exporting", last_completed_image=image_id,
                     excluded_images=[key for key, value in state.items() if not value["admitted"]])
            print(json.dumps(dict(image_id=image_id, admitted=result["admitted"], completed_images=len(images))), flush=True)
        save_reviews()
        progress("export_complete", temporary_cache= str(TEMP))
    except Exception as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise

if args.stage == "verify":
    try:
        assert all((TEMP / member["name"]).exists() and (TEMP / member["name"]).stat().st_size == member["size"]
                   for member in inventory["members"]), "Run export to restore local source planes before verify"
        for position, item in enumerate(records):
            image_id = item["physical_image_id"].split("::")[1].replace("/", "__")
            if image_id in set(images.image_id):
                verify_header(OUTPUT / "images" / image_id / "input.tiff", **geometry(item))
                if (position + 1) % 13 == 0:
                    progress("checking_headers", tiff_headers_checked=position + 1)
        checks = []
        for position in (0, 26, 52):
            item = records[position]
            image_id = item["physical_image_id"].split("::")[1].replace("/", "__")
            row = images[images.image_id == image_id].iloc[0]
            size = geometry(item)
            with roi_reader(item) as reader, formal_read_lease(NAS / row.input_image_path), tifffile.TiffFile(NAS / row.input_image_path) as tif:
                for window in ((0, 0, 32, 32), (size["width"] // 2, size["height"] // 2,
                                size["width"] // 2 + 32, size["height"] // 2 + 32)):
                    expected = reader.read_window(kept.source_channel_index.tolist(), window)
                    x0, y0, x1, y1 = window
                    shifted = (x0 + row.pad_left_px, y0 + row.pad_top_px, x1 + row.pad_left_px, y1 + row.pad_top_px)
                    actual = np.stack([read_tiff_window(tif, i, shifted) for i in range(len(kept))])
                    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
                    checks.append(dict(image_id=image_id, source_xyxy=list(window), output_xyxy=[int(value) for value in shifted],
                                       channel_count=len(kept), dtype=str(actual.dtype),
                                       source_nonzero_pixels=int(np.count_nonzero(expected)), equal=True))
                for index in range(len(kept)):
                    if row.pad_top_px:
                        assert not read_tiff_window(tif, index, (0, 0, int(row.width_px), int(row.pad_top_px))).any()
                    if row.pad_bottom_px:
                        assert not read_tiff_window(tif, index, (0, int(row.height_px-row.pad_bottom_px), int(row.width_px), int(row.height_px))).any()
                    if row.pad_left_px:
                        assert not read_tiff_window(tif, index, (0, 0, int(row.pad_left_px), int(row.height_px))).any()
                    if row.pad_right_px:
                        assert not read_tiff_window(tif, index, (int(row.width_px-row.pad_right_px), 0, int(row.width_px), int(row.height_px))).any()
        write_json(CACHE / "pixel_self_check.json", dict(preparer=PREPARER, independent_review=False,
            tiff_headers_checked=len(images), blocks=checks, padding_all_channels_checked=3,
            exported_images=len(images), sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
            patient_linked_samples=int(samples.patient_key.notna().sum())))
        save_reviews()
        progress("prepared", actual_tiffs=len(list((OUTPUT / "images").glob("*/input.tiff"))),
                 temporary_cache=str(TEMP), temporary_retained=args.keep_temporary)
        if not args.keep_temporary:
            shutil.rmtree(TEMP)
    except Exception as error:
        progress("failed_self_check", error_type=type(error).__name__, error=str(error))
        raise

if args.stage == "metadata":
    save_reviews()
    print(json.dumps(dict(study=STUDY, expected_images=len(records), completed_images=len(images),
                         sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()))))
