"""Prepare the seven qualified IDR IBEX acquisitions at native intensity.

Inputs: accepted source closeout, exact IMS proofs, IDR annotation and current S1.
Outputs: per-image input TIFF, five metadata tables, resumable receipts and JOB progress.
The lower central native Z plane is selected explicitly; no projection is applied.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
from pathlib import Path
import shutil
import signal
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease, formal_publish_lease
from cellweft.data.preparation.study_input import panel_from_s1, export_input, save_input_tables

STUDY = "Radtke2024_CellDive_FL"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = ROOT / "docs/data/templates/study_input"
CACHE = ROOT / "cache/study_input_runtime" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("JOB", str(CACHE / "0803_prepare_0")))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
CLOSEOUT = metadata_source_directory(NAS, STUDY) / 'selected_subset_closeout_20261025.json'
SOURCE_METADATA = metadata_source_directory(NAS, STUDY)
ANNOTATION = SOURCE_METADATA / "input_facts/current_idr_annotation.csv"
SOURCE_MAPPING = SOURCE_METADATA / "idr0158_ibex_channels.json"
PREPARER = "0803_prepare_0"
for path in (CACHE, OUTPUT, JOB, CACHE / "receipts"):
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, **extra: object) -> None:
    write_json(JOB / "progress.json", dict(study=STUDY, phase="prepare", stage=stage,
               expected_images=7, exported_images=len(image_rows), sample_rows=len(samples),
               sample_linked_images=len(image_rows), patient_linked_samples=int(samples.patient_key.ne("").sum()),
               completed_image_ids=[r["image_id"] for r in image_rows], **extra))


def decode(value: np.ndarray) -> str:
    return b"".join(value).decode()


class SourceDecodeError(RuntimeError):
    """A native source block fails decoding on two bounded reads."""


def export_native(spec: SourceSpec, panel: pd.DataFrame, destination: Path, item: dict) -> dict:
    """Attach exact coordinates to the observed HDF5 filter failure, with one explicit re-read."""
    with open_source(spec) as reader:
        native_read = reader.read_window
        def contextual_read(channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
            try:
                return native_read(channels, window)
            except OSError as first:
                if "filter returned failure" not in str(first):
                    raise
                record = dict(image_id=spec.image_id, source_path=item["path"], source_stat=item["source_stat"],
                              z_index=spec.z_index, source_channels=channels, window=list(window), first_error=str(first))
                try:
                    values = native_read(channels, window)
                except OSError as second:
                    record.update(repeat_error=str(second), repeated_failure=True)
                    write_json(CACHE / "receipts" / (spec.image_id + "_source_failure.json"), record)
                    raise SourceDecodeError(json.dumps(record)) from second
                record.update(repeated_failure=False, recovery="One immediate bounded source re-read succeeded")
                write_json(CACHE / "receipts" / (spec.image_id + "_transient_read.json"), record)
                return values
        reader.read_window = contextual_read
        return export_input(reader, panel, destination, tile_size=512)


def stop_for_io(signum: int, frame: object) -> None:
    if "--repair-metadata" in sys.argv:
        write_json(JOB / "progress.json", dict(study=STUDY, phase="repair", repair_round=1,
                   status="waiting_io", stage="metadata_repair_resume_required",
                   resume_command="scripts/data/preparation/Radtke2024_CellDive_FL.py --repair-metadata",
                   checkpoint="Merge is idempotent; reopen current tables and publish only missing metadata changes"))
        raise SystemExit(0)
    raise InterruptedError("Controller requested a closed-checkpoint I/O handoff")


signal.signal(signal.SIGUSR1, stop_for_io)
closeout = json.loads(CLOSEOUT.read_text())
source_records = json.loads((SOURCE_MAPPING).read_text())["records"]
source_records = {r["file_name"]: r for r in source_records}
annotation = pd.read_csv(ANNOTATION, keep_default_na=False, dtype=str)
s1_keys = set(pd.read_csv(S1, keep_default_na=False).query("Study == @STUDY")["Original marker"])

# Exact public sample identity and clinical annotation; donor correspondence remains evidence-bounded.
sample_rows = []
for item in closeout["acquisitions"]:
    original = annotation.loc[annotation["Image Name"].eq(item["acquisition_id"])].iloc[0]
    row = {key: "" for key in pd.read_csv(TEMPLATES / "sample_metadata.csv").columns}
    row.update(study=STUDY, sample_id=item["sample_id"], source_sample_id=original["Characteristics [Sample ID]"],
               tissue_type="tumor" if item["sample_id"].startswith("FL") else "normal",
               sample_type="tissue", anatomical_site="Lymph node",
               source_age_reported=original["Characteristics [Age]"],
               source_ethnic_or_racial_group=original["Characteristics [Ethnic or Racial Group]"],
               sex_reported=original["Characteristics [Sex]"].lower(),
               case_control_status="case" if item["sample_id"].startswith("FL") else "control",
               primary_diagnosis="Follicular lymphoma" if item["sample_id"].startswith("FL") else "Histologically normal lymph node",
               clinical_group=original["Characteristics [Clinical Group]"],
               source_quality_note=item["source_quality_note"],
               metadata_notes="Exact IDR image-to-sample match. Patient linkage unresolved pending explicit donor crosswalk; clinical group is not a time-to-event value. IDR Age is retained verbatim because unit and reference event are not established.")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)

mapping_rows = []

def mapping(source: str, table: str, field: str, rule: str, notes: str = "",
            source_file: Path | None = None, source_table: str = "IDR experimentA annotation") -> None:
    row = {key: "" for key in pd.read_csv(TEMPLATES / "field_mapping.csv").columns}
    row.update(study=STUDY, source_file=str(metadata_source_relative(source_file or ANNOTATION, NAS)), source_sheet_or_table=source_table,
               source_field=source, target_table=table, target_field=field, join_keys="Image Name = acquisition_id; Characteristics [Sample ID]",
               transformation_rule=rule, conflict_rule="Retain exact original annotation; unresolved identity stays empty", notes=notes)
    mapping_rows.append(row)

for source, target, rule in [
    ("Characteristics [Sample ID]", "source_sample_id", "Preserve original ID; stable sample_id uses accepted source folder identity"),
    ("Characteristics [Age]", "source_age_reported", "Preserve original IDR Age text; unit and reference event unspecified; all anchored age fields remain empty"),
    ("Characteristics [Sex]", "sex_reported", "Male -> male; Female -> female"),
    ("Characteristics [Ethnic or Racial Group]", "source_ethnic_or_racial_group", "Preserve original IDR annotation; no category inference"),
    ("Characteristics [Clinical Group]", "clinical_group", "Preserve original clinical group; do not infer PFS or OS event/time"),
    ("Comment [Image Quality]", "source_quality_note", "Preserve accepted source quality statement"),
    ("Characteristics [Sample ID]", "patient_key", "Leave empty: source sample identity alone does not establish a donor crosswalk"),
]:
    mapping(source, "sample_metadata", target, rule)
mapping("Image Name", "image_metadata", "sample_id", "Exact one original IMS per accepted source sample; seven image/sample connections")
for item in closeout["acquisitions"]:
    mapping("DataSetInfo/Image; ImageSizeX/Y/Z", "image_metadata", "source_z_index", "Lower central actual plane: (ImageSizeZ - 1) // 2; TimePoint 0; no Z projection; crop array storage padding to ImageSizeX/Y", source_file=NAS / item["path"], source_table="IMS DataSetInfo and ResolutionLevel 0 attributes")
    mapping("DataSetInfo/Image ExtMin/ExtMax and X/Y", "image_metadata", "pixel_size_x_um", "(ExtMax0 - ExtMin0)/X and (ExtMax1 - ExtMin1)/Y; units supported by paper microscopy methods (micrometres)", source_file=NAS / item["path"], source_table="IMS DataSetInfo/Image attributes")
mapping("annotation_channels.source_label", "panel_mapping", "s1_original_marker", "IDR per-channel antibody label is an explicit alias to generic IBEX S1 marker; exact Cycle1/2 Hoechst rows retain current S1 nuclear use; other cycles alias Nuclei and remain unselected", notes="Current S1: " + str(metadata_source_relative(S1, NAS)), source_file=SOURCE_MAPPING, source_table="records.annotation_channels")
for field, definition in [("source_ethnic_or_racial_group", "Original IDR ethnic or racial group annotation, without recoding"), ("source_age_reported", "Original IDR Characteristics [Age] text; unit and reference event unspecified"), ("clinical_group", "Original IDR clinical group; no survival duration/event inference"), ("source_quality_note", "Accepted exact source-specific image quality statement")]:
    row = next(row for row in mapping_rows if row["target_field"] == field)
    row["notes"] = json.dumps(dict(data_type="string", definition=definition))
    if field == "source_quality_note":
        row.update(source_file=str(metadata_source_relative(CLOSEOUT, NAS)), source_sheet_or_table="acquisitions", source_field="source_quality_note")
for field, rule in [("sample_id", "Accepted source folder ID; original annotation ID retained separately"), ("case_control_status", "Accepted FL acquisitions -> case; histologically normal nLN acquisitions -> control"), ("primary_diagnosis", "Accepted FL acquisitions -> Follicular lymphoma; accepted nLN acquisitions -> Histologically normal lymph node")]:
    mapping("sample_id; clinical_group; accepted pathology classification", "sample_metadata", field, rule, source_file=CLOSEOUT, source_table="acquisitions; source_evidence")
next(row for row in mapping_rows if row["target_field"] == "patient_key")["notes"] = json.dumps(dict(
    sample_rows=7, patient_linked_samples=0,
    searched_evidence=[str(metadata_source_relative(SOURCE_METADATA / "input_facts" / name, NAS)) for name in
                      ("public_source_search.json", "additional_source_search.json")],
    saved_primary_article=f"raw/public/{STUDY}/metadata/primary_paper.html",
    definition="Patient key remains empty; no donor identity is inferred from a source sample identifier or demographics"))
mapping("Characteristics [Organism Part 1]; Characteristics [Organism Part 2]", "sample_metadata", "sample_type",
        "IDR lymph-node tissue acquisition -> tissue; no blood or cell-suspension inference")
mapping("Characteristics [Organism Part 1]; Characteristics [Organism Part 2]", "sample_metadata", "anatomical_site",
        "Preserve confirmed lymph-node anatomical site as Lymph node")
mapping("accepted pathology classification; sample_id; clinical_group", "sample_metadata", "tissue_type",
        "Accepted follicular lymphoma specimens -> tumor; histologically normal nLN specimens -> normal",
        source_file=CLOSEOUT, source_table="acquisitions; source_evidence")
field_mapping = pd.DataFrame(mapping_rows)

# The first repair changes clinical metadata only, retaining the independent pixel evidence.
if "--repair-metadata" in sys.argv:
    repair_id = JOB.name
    write_json(JOB / "progress.json", dict(study=STUDY, phase="repair", repair_round=1,
               status="waiting_io", stage="awaiting_metadata_read_lease",
               resume_command="scripts/data/preparation/Radtke2024_CellDive_FL.py --repair-metadata"))
    with formal_read_lease(OUTPUT):
        with resource_lock(OUTPUT, shared=True):
            images = pd.read_parquet(OUTPUT / "images.parquet")
            panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
            prior_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False, dtype=str)
            prior_fields = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False, dtype=str)
            unchanged_paths = [OUTPUT / "images.parquet", OUTPUT / "panel_mapping.csv"]
            unchanged_paths += [NAS / value for value in images.input_image_path]
            unchanged_versions = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
                                  for path in unchanged_paths}
    expected_tissue = samples.set_index("sample_id").tissue_type
    assert set(prior_samples.sample_id) == set(expected_tissue.index)
    repaired_samples = prior_samples.copy()
    repaired_samples["tissue_type"] = repaired_samples.sample_id.map(expected_tissue)
    additions = field_mapping.loc[field_mapping.target_field.isin(
        ["sample_type", "tissue_type", "anatomical_site"])]
    capacity = dict(nas_free_bytes=shutil.disk_usage(NAS).free,
                    local_free_bytes=shutil.disk_usage("/tmp").free,
                    estimated_additional_bytes=1024**2, nas_reserve_bytes=100*2**30,
                    local_reserve_bytes=64*2**30)
    image_rows = images.to_dict("records")
    write_json(JOB / "capacity_estimate.json", capacity)
    assert capacity["nas_free_bytes"] > capacity["nas_reserve_bytes"] + capacity["estimated_additional_bytes"]
    assert capacity["local_free_bytes"] > capacity["local_reserve_bytes"] + capacity["estimated_additional_bytes"]
    write_json(JOB / "progress.json", dict(study=STUDY, phase="repair", repair_round=1,
               status="prepared", stage="metadata_publication_pending", exported_images=len(images)))
    save_input_tables(OUTPUT, images, repaired_samples, panel, additions)
    with formal_read_lease(OUTPUT):
        with resource_lock(OUTPUT, shared=True):
            actual_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False, dtype=str)
            actual_fields = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False, dtype=str)
            pd.testing.assert_frame_equal(actual_samples, repaired_samples)
            assert set(additions.target_field).issubset(set(actual_fields.target_field))
            assert unchanged_versions == {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
                                          for path in unchanged_paths}
    evidence = dict(study=STUDY, repair_round=1, preparer=repair_id, independent_review_pending=True,
                    changed_field="sample_metadata.tissue_type", sample_rows=len(actual_samples),
                    tissue_counts=actual_samples.tissue_type.value_counts().to_dict(),
                    added_mapping_targets=additions.target_field.tolist(),
                    previous_mapping_rows=len(prior_fields), current_mapping_rows=len(actual_fields),
                    exported_images=len(images), sample_linked_images=int(images.sample_id.isin(actual_samples.sample_id).sum()),
                    patient_linked_samples=int(actual_samples.patient_key.ne("").sum()),
                    unchanged_versions=unchanged_versions)
    write_json(JOB / "metadata_repair.json", evidence)
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
        reviews = pd.concat([reviews, pd.DataFrame([dict(study=STUDY, preparer=repair_id, reviewer="",
            scope="study", record_id=STUDY, check_id="M02", result="unresolved",
            source_evidence=str(metadata_source_relative(CLOSEOUT, NAS)), output_evidence=str(metadata_source_relative(JOB / 'metadata_repair.json', NAS)),
            details=json.dumps(dict(repair_round=1, action="Corrected seven tissue classifications and added three field mappings",
                                    independent_review_pending=True, clinical_attachment_gap_retained=True)))])], ignore_index=True)
        temporary = OUTPUT / "review_results.csv.partial"
        reviews.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(OUTPUT / "review_results.csv")
    write_json(JOB / "progress.json", dict(evidence, phase="repair", status="prepared", stage="metadata_repair_closed"))
    print(json.dumps({key: evidence[key] for key in ("sample_rows", "exported_images", "sample_linked_images", "patient_linked_samples")}))
    raise SystemExit(0)

# Bind each actual ordered panel and true native acquisition shape before estimating capacity.
panels, inventory = [], []
for item in closeout["acquisitions"]:
    channels = source_records[item["acquisition_id"]]["annotation_channels"]
    names = [c["source_channel"] for c in channels]
    aliases = {c["source_channel"]: c["source_label"] for c in channels}
    for name in names:
        if name in {"Hoechst:Cycle1_Nuclei", "Hoechst:Cycle2_Nuclei"}:
            aliases[name] = name
    panel_id = "IBEX_" + item["sample_id"]
    panel = panel_from_s1(STUDY, panel_id, names, S1, aliases=aliases)
    panels.append(panel)
    path = NAS / item["path"]
    proof = json.loads((SOURCE_METADATA / "input_facts/native_proofs" / Path(item["proof"]).name).read_text())
    stat = path.stat()
    assert (stat.st_size, stat.st_mtime_ns) == (proof["file_stat"]["size"], proof["file_stat"]["mtime_ns"])
    assert proof["annotation_embedded_order_equal"]
    with formal_read_lease(path):
        with h5py.File(path, "r") as source:
            group = source["DataSet/ResolutionLevel 0/TimePoint 0"]
            dims = {key: decode(value) for key, value in source["DataSetInfo/Image"].attrs.items()}
            native = {key: decode(value) for key, value in group["Channel 0"].attrs.items()}
            h, w, z = (int(native[key]) for key in ("ImageSizeY", "ImageSizeX", "ImageSizeZ"))
            assert len(group) == len(names)
            assert [decode(source[f"DataSetInfo/Channel {i}"].attrs["Name"]) for i in range(len(names))] == names
            dtype = str(group["Channel 0/Data"].dtype)
            scale = [(float(dims[f"ExtMax{i}"]) - float(dims[f"ExtMin{i}"]))/int(dims[axis]) for i, axis in enumerate(("X", "Y"))]
    inventory.append(dict(**item, panel_id=panel_id, shape=[len(names), h, w], z_count=z, z_index=(z-1)//2,
                          pixel_size_um=scale, dtype=dtype, source_stat=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                          maximum_output_bytes=int(panel.keep_channel.sum()) * max(h,w)**2 * np.dtype(dtype).itemsize))
panel_table = pd.concat(panels, ignore_index=True)
write_json(CACHE / "source_inventory.json", dict(acquisitions=inventory, excluded_members=closeout["excluded_members"]))
maximum_output = sum(item["maximum_output_bytes"] for item in inventory)
local_peak = max(item["maximum_output_bytes"] for item in inventory) * 2
capacity = dict(maximum_uncompressed_output_bytes=maximum_output, conservative_local_peak_bytes=local_peak,
                nas_free_bytes=shutil.disk_usage(NAS).free, local_free_bytes=shutil.disk_usage("/tmp").free,
                nas_reserve_bytes=100*2**30, local_reserve_bytes=64*2**30)
write_json(CACHE / "capacity_estimate.json", capacity)
assert capacity["nas_free_bytes"] > maximum_output + capacity["nas_reserve_bytes"]
assert capacity["local_free_bytes"] > local_peak + capacity["local_reserve_bytes"]
image_rows = []
source_failures = []
progress("inventory_complete", capacity=capacity)

def save_reviews(*, final: bool = False) -> None:
    """Publish actual preparer scope without issuing independent acceptance."""
    rows=[]
    identifiers=[row["image_id"] for row in image_rows]
    checks=[
        ("S01", "fail" if final and source_failures else ("pass" if final else "unresolved"),
         dict(image_ids=[r["image_id"] for r in source_failures] if source_failures else identifiers,
              expected_sources=7, exported_images=len(image_rows), source_failures=source_failures,
              note="Prior bounded source proofs reused; selected-plane decoding verified by export. Three historical exclusions remain unchanged.")),
        ("M01", "unresolved", dict(sample_rows=len(samples), sample_linked_images=len(image_rows), patient_linked_samples=0,
              note="Exact image-to-sample IDR links confirmed; explicit donor crosswalk not established. See public_source_search.json and additional_source_search.json.")),
        ("M02", "unresolved", dict(note="IDR Age text (unit/reference event unspecified), sex and original clinical groups retained. Supplementary clinical table endpoints did not deliver a usable workbook; no survival duration/event inferred.")),
        ("D01", "unresolved", dict(image_ids=identifiers, note="Preparer self-check only. Independent reviewer must inspect per-image evidence and issue current input_binding; independent acceptance remains pending.")),
    ]
    for check, result, details in checks:
        rows.append(dict(study=STUDY,preparer=PREPARER,reviewer="",scope="study",record_id=STUDY,
                         check_id=check,result=result,source_evidence=str(metadata_source_relative(CLOSEOUT, NAS)),
                         output_evidence=str(metadata_source_relative(OUTPUT, NAS)),details=json.dumps(details)))
    for image_id in identifiers:
        if image_id in {"IBEX_FL4", "IBEX_FL6"}:
            rows.append(dict(study=STUDY,preparer=PREPARER,reviewer="",scope="image",record_id=image_id,
                check_id="I05",result="unresolved",source_evidence=str(metadata_source_relative(SOURCE_METADATA / 'input_facts/spatial_metadata_search.json', NAS)),
                output_evidence=str(metadata_source_relative(CACHE / 'receipts' / (image_id + '.json'), NAS)),
                details=json.dumps(dict(image_ids=[image_id], pixel_identity="Two blocks, all retained channels equal; see receipt",
                    unresolved="Staircase/rectangular source black areas require authoritative acquisition-footprint evidence; no intensity-derived valid mask is invented. FL4 also has source-reported masked nuclear corruption without an available exact artifact mask."))))
    with formal_publish_lease(OUTPUT,directory=OUTPUT):
        temporary=OUTPUT/"review_results.csv.partial"
        pd.DataFrame(rows).to_csv(temporary,index=False,encoding="utf-8-sig")
        temporary.replace(OUTPUT/"review_results.csv")


save_reviews()

# Publish metadata before pixels so an I/O handoff retains the confirmed sample coverage.
save_input_tables(OUTPUT, pd.DataFrame(columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns), samples, panel_table, field_mapping)

# The shared exporter leases source reads and formal writes independently, and closes every image.
try:
    for item in inventory:
        image_id = "IBEX_" + item["sample_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = CACHE / "receipts" / (image_id + ".json")
        panel = panel_table.loc[panel_table.panel_id.eq(item["panel_id"])]
        names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
        old = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if old and old["source_stat"] == item["source_stat"] and old["panel_names"] == names and old["image_metadata"]["source_z_index"] == item["z_index"] and destination.is_file():
            stat = destination.stat()
            assert [stat.st_size, stat.st_mtime_ns] == old["output_stat"]
            reused_row = old["image_metadata"].copy()
            if image_id in {"IBEX_FL4", "IBEX_FL6"} and "Acquisition-footprint evidence" not in reused_row["metadata_notes"]:
                reused_row["metadata_notes"] += " Acquisition-footprint evidence and exact author artifact masks unavailable; spatial validity pending; no zero-intensity-derived mask is supplied."
            image_rows.append(reused_row)
            progress("reused", current_image=image_id)
            continue
        failure_path = CACHE / "receipts" / (image_id + "_source_failure.json")
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
        if (failure and failure["source_stat"] == item["source_stat"] and failure["z_index"] == item["z_index"]
                and set(failure["source_channels"]).issubset(set(panel.loc[panel.keep_channel].source_channel_index))):
            source_failures.append(failure)
            progress("reused_source_failure", current_image=image_id)
            continue
        _, h, w = item["shape"]
        spec = SourceSpec(image_id, item["panel_id"], "ims", NAS / item["path"], tuple(range(len(panel))),
                          z_index=item["z_index"], valid_shape=(h,w), pixel_size_um=tuple(item["pixel_size_um"]))
        progress("exporting", current_image=image_id, next_source=item["path"], z_index=item["z_index"])
        try:
            result = export_native(spec, panel, destination, item)
        except SourceDecodeError as error:
            source_failures.append(json.loads(str(error)))
            progress("source_decode_failed", current_image=image_id, source_failures=source_failures)
            continue
        assert result["admitted"], "Accepted source has no retained marker signal in selected native plane"
        row = dict(study=STUDY, image_id=image_id, sample_id=item["sample_id"], panel_id=item["panel_id"],
                   image_kind="roi", source_image_id=item["acquisition_id"], source_files=[item["path"]],
                   source_format="ims", source_z_index=item["z_index"], source_t_index=0, source_axes="CZYX",
                   source_dtype=result["source_dtype"], source_revision=json.dumps(item["source_stat"],sort_keys=True),
                   region_annotation=item["source_quality_note"], source_x0_px=0, source_y0_px=0,
                   source_x1_px=w, source_y1_px=h, height_px=result["height_px"], width_px=result["width_px"],
                   source_channel_count=len(panel), channel_count=result["channel_count"],
                   pixel_size_x_um=item["pixel_size_um"][0], pixel_size_y_um=item["pixel_size_um"][1],
                   input_image_path=str(metadata_source_relative(destination, NAS)), valid_pixels_path=None,
                   intensity_description="Native published uint8 IMS intensities; no scaling, thresholding or projection applied during preparation",
                   split_method="source_acquisition", split_evidence="One exact original IBEX IMS for one IDR sample; complete ImageSizeX/Y acquisition rectangle",
                   metadata_notes=f"Lower central native Z plane {item['z_index']} of {item['z_count']}; other Z planes are not additional images. Patient linkage unresolved.")
        if image_id in {"IBEX_FL4", "IBEX_FL6"}:
            row["metadata_notes"] += " Acquisition-footprint evidence and exact author artifact masks unavailable; spatial validity pending; no zero-intensity-derived mask is supplied."
        row.update({key: result[key] for key in ("pad_left_px","pad_top_px","pad_right_px","pad_bottom_px")})
        # Reopen native source and published TIFF under one shared read lease for bounded self-check.
        blocks = [(w//2, h//2, min(w,w//2+32), min(h,h//2+32)), (w-32,h-32,w,h)]
        with formal_read_lease(destination):
            with open_source(spec) as reader, tifffile.TiffFile(destination) as tif:
                assert tif.series[0].shape == (len(names),max(h,w),max(h,w))
                assert str(tif.pages[0].dtype) == item["dtype"]
                ome_names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter() if node.tag.endswith("}Channel")]
                assert ome_names == names
                for block in blocks:
                    values = reader.read_window(panel.loc[panel.keep_channel].source_channel_index.tolist(), block)
                    x0,y0,x1,y1 = block
                    target = (x0+row["pad_left_px"],y0+row["pad_top_px"],x1+row["pad_left_px"],y1+row["pad_top_px"])
                    for i in range(len(names)):
                        assert np.array_equal(values[i],read_tiff_window(tif,i,target))
                if row["pad_top_px"]:
                    for i in range(len(names)):
                        assert not read_tiff_window(tif,i,(0,0,32,min(32,row["pad_top_px"]))).any()
                # Display only: lower-resolution native plane, sampled without changing saved intensity.
                with h5py.File(spec.path,"r") as source:
                    level = min(3, max(int(k.split()[-1]) for k in source["DataSet"].keys()))
                    group = source[f"DataSet/ResolutionLevel {level}/TimePoint 0"]
                    selection = [int(panel.loc[panel.nuclear].source_channel_index.iloc[0]),int(panel.loc[panel.role.eq("downstream")].source_channel_index.iloc[0])]
                    planes=[]
                    for index in selection:
                        ds=group[f"Channel {index}/Data"]
                        plane=np.asarray(ds[min(item["z_index"]//(2**level),ds.shape[0]-1)])
                        step=max(1,max(plane.shape)//1000); plane=plane[::step,::step].astype(float)
                        high=np.percentile(plane,99.5); planes.append(np.clip(plane/max(high,1),0,1))
                    rgb=np.stack([planes[1],planes[1],planes[0]],axis=-1)
                    Image.fromarray((rgb*255).astype(np.uint8)).save(CACHE/(image_id+"_overview.png"))
        stat = destination.stat()
        image_rows.append(row)
        write_json(receipt_path,dict(image_metadata=row,source_stat=item["source_stat"],panel_names=names,
                   output_stat=[stat.st_size,stat.st_mtime_ns],export=result,selfcheck=dict(blocks=blocks,all_retained_channels=True,headers=True,padding=True)))
        save_input_tables(OUTPUT,pd.DataFrame(image_rows),samples,panel_table,field_mapping)
        progress("image_closed",current_image=image_id)
except (InterruptedError, OSError) as error:
    # Preserve any complete local publication checkpoint for a subsequent I/O-only resume.
    progress("waiting_io",status="waiting_io",error=str(error),
             resume_script=str(metadata_source_relative(Path(__file__), ROOT)),
             local_checkpoint_root=os.environ.get("CELLWEFT_INPUT_STAGING_ROOT","/tmp/study_input_batch"))
    raise SystemExit(0)

save_input_tables(OUTPUT, pd.DataFrame(image_rows), samples, panel_table, field_mapping)
save_reviews(final=True)
progress("prepared" if not source_failures else "source_decode_partial",status="prepared" if not source_failures else "partial",independently_reviewed=False, source_failures=source_failures)
print(json.dumps(dict(status="prepared" if not source_failures else "partial",exported_images=len(image_rows),sample_rows=len(samples),patient_linked_samples=0)))
