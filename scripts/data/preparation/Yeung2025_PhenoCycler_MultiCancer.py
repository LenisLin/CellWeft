"""Export eight physical TMA cores from the published Yeung QPTIFF.

Inputs: the verified Zenodo ZIP, native core/IFD proofs, public core crosswalk,
and current S1. Outputs: native-intensity CYX TIFFs and per-study input tables.
Each completed core is indexed before proceeding to the next.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Yeung2025_PhenoCycler_MultiCancer"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
ARCHIVE = ROOT / "raw/public" / STUDY / "public_original/Bwb_SpatialProteomics_TMA_rawfiles.zip"
MEMBER = "Bwb_SpatialProteomics_TMA_rawfiles/240508_VET_QCTMA2_A2_55p_Scan1.qptiff"
GEOMETRY = META / "collection_evidence/source_geometry_manifest.json"
AXIS = GEOMETRY.with_name("source_native_axis_verified.json")
CROSSWALK = META / "preparation_evidence/public_core_crosswalk.json"
OUTPUT = ROOT / "processed" / STUDY
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY / "jobs/0653_prepare_0")))
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PREPARER = "codex_yeung_" + JOB.name


def relative(path: Path) -> str:
    return metadata_source_relative(path, ROOT).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def publish_json(path: Path, value: dict) -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / (path.name + ".new")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, partial)
        os.replace(partial, path)


def progress(stage: str, **details: object) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").is_file() else pd.DataFrame()
    linked = set(samples.loc[samples.patient_key.notna(), "sample_id"]) if len(samples) else set()
    repairing = JOB.name.endswith("repair_1")
    state = dict(study=STUDY, phase="repair" if repairing else "prepare",
                 scientific_repair_round=1 if repairing else 0,
                 stage=stage, expected_images=8, exported_images=len(images),
                 sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=len(linked),
                 patient_linked_images=int(images.sample_id.isin(linked).sum()) if len(images) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    publish_json(JOB / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> tuple[list[dict], list[str], dict]:
    with formal_read_lease(META):
        native = json.loads((META / "native_verification.json").read_text())
        manifest = json.loads((META / "source_manifest.json").read_text())
        geometry = json.loads(GEOMETRY.read_text())
        axis = json.loads(AXIS.read_text())
        crosswalk = json.loads(CROSSWALK.read_text())
    stat = ARCHIVE.stat()
    if (not native["scope_complete"] or native["gaps"] or native["source_defects"]
            or len(native["records"]) != 8 or len(geometry["cores"]) != 8
            or len(crosswalk["cores"]) != 8 or stat.st_size != manifest["source_bytes"]
            or (stat.st_size, stat.st_mtime_ns) !=
               (geometry["source_stat"]["size"], geometry["source_stat"]["mtime_ns"])):
        raise ValueError("Archive and eight-core native proof do not match")
    names = ["DAPI"] + [entry["source_label"] for entry in native["records"][0]["source_channel"]]
    full = [row for row in axis["records"] if row["image_type"] == "FullResolution"]
    if (len(full) != 59 or [row["ifd_index"] for row in full] != list(range(59))
            or [row["biomarker"] for row in full] != names
            or any(row["shape"] != [20160, 9600] or row["dtype"] != "uint8" for row in full)):
        raise ValueError("Full-resolution native IFD order differs from proof")
    with formal_read_lease(ARCHIVE), zipfile.ZipFile(ARCHIVE) as archive:
        member = archive.getinfo(MEMBER)
    if member.file_size != 1205290891:
        raise ValueError("QPTIFF member length differs from native proof")
    scope = []
    for number, (record, core, clinical) in enumerate(
            zip(native["records"], geometry["cores"], crosswalk["cores"]), 1):
        image_id = core["geometry_id"]
        if (record["physical_image_id"] != image_id or clinical["number"] != number
                or len(record["source_channel"]) != 58
                or [entry["source_label"] for entry in record["source_channel"]] != names[1:]
                or not record["native_proof"]["complete"]):
            raise ValueError(f"Native core proof differs for core {number}")
        scope.append(dict(image_id=image_id, number=number,
                          box=core["native_envelope_xyxy"], clinical=clinical,
                          revision=f"zip_size={stat.st_size};zip_mtime_ns={stat.st_mtime_ns};member_crc={member.CRC}"))
    return scope, names, dict(size=member.file_size, crc=member.CRC)


def source_stage(member: dict) -> Path:
    TEMP.mkdir(parents=True, exist_ok=True)
    path = TEMP / Path(MEMBER).name
    binding = TEMP / (path.name + ".source.json")
    identity = dict(archive=relative(ARCHIVE), member=MEMBER, **member)
    if (path.is_file() and binding.is_file() and path.stat().st_size == member["size"]
            and json.loads(binding.read_text()) == identity):
        return path
    # The 64 GiB reserve includes the ZIP extraction and two local TIFF copies.
    largest_side = 3520
    local_need = member["size"] + 2 * 56 * largest_side**2
    if shutil.disk_usage(TEMP).free < 64 * 1024**3 + local_need:
        raise OSError(28, "Local input staging would consume the 64 GiB reserve")
    partial = TEMP / (path.name + ".partial")
    with formal_read_lease(ARCHIVE), zipfile.ZipFile(ARCHIVE) as archive:
        info = archive.getinfo(MEMBER)
        if (info.file_size, info.CRC) != (member["size"], member["crc"]):
            raise ValueError("ZIP member differs from native proof")
        with archive.open(info) as source, partial.open("wb") as staged:
            shutil.copyfileobj(source, staged, length=8 * 1024**2)
    if partial.stat().st_size != member["size"]:
        raise ValueError("Incomplete local source stage")
    os.replace(partial, path)
    binding.write_text(json.dumps(identity, sort_keys=True))
    return path


def verify_source_header(path: Path, names: list[str]) -> None:
    with tifffile.TiffFile(path) as tif:
        for index, name in enumerate(names):
            page = tif.pages[index].aspage()
            if page.shape != (20160, 9600) or page.dtype != np.dtype("uint8"):
                raise ValueError(f"Unexpected QPTIFF full-resolution page {index}: {name}")


def metadata_tables(scope: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = []
    for item in scope:
        clinical = item["clinical"]
        paper_id = clinical["paper_sample_id"]
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=item["image_id"],
                   patient_key=f"paper_tma_{paper_id}" if paper_id else None,
                   source_sample_id=paper_id,
                   sample_type="tissue core",
                   tissue_type="tumor" if paper_id else "normal",
                   anatomical_site=clinical["tissue"],
                   primary_diagnosis=clinical["diagnosis"], sex_reported=clinical["sex"],
                   metadata_notes=("Unique tumor diagnosis joins numbered author core to paper Table 1; "
                                   "paper Sample ID is a specimen identifier, not a source patient ID; "
                                   "patient_key is a study-local surrogate; patient age has no known "
                                   "collection/index timepoint."
                                   if paper_id else
                                   "Normal tonsil core: direct specimen/person link is unresolved; "
                                   "paper Table 1 lists one tonsil sample but does not map both cores."))
        samples.append(row)
    fields = []

    def add(source: str, source_field: str, table: str, target: str, keys: str, rule: str) -> None:
        row = dict.fromkeys(columns("field_mapping"), "")
        row.update(study=STUDY, source_file=source, source_field=source_field,
                   target_table=table, target_field=target, join_keys=keys,
                   value_mapping="{}", transformation_rule=rule)
        fields.append(row)

    add(relative(GEOMETRY), "cores.geometry_id;native_envelope_xyxy", "image_metadata",
        "image_id;source_x0_px;source_y0_px;source_x1_px;source_y1_px", "geometry_id",
        "Use reviewed separated physical-core envelopes in native half-open coordinates.")
    add(relative(AXIS), "records.ifd_index;biomarker", "panel_mapping",
        "source_channel_index;source_name", "FullResolution IFD index",
        "IFD 0 is DAPI; full-resolution IFDs 1-58 follow measured native protein order.")
    add(relative(S1), "Study;Original marker;Segmentation channel", "panel_mapping",
        "marker_id;keep_channel;output_channel_index", "Study;Original marker",
        "Current S1 controls downstream and segmentation use; retain all source rows.")
    add(relative(CROSSWALK), "cores.number;tissue;diagnosis", "image_metadata",
        "region_annotation;core_id", "numbered overview spatial order=geometry_id order",
        "Author numbered overview identifies each physical core by tissue/disease.")
    add(relative(CROSSWALK), "cores.paper_sample_id;diagnosis;sex", "sample_metadata",
        "patient_key;source_sample_id;primary_diagnosis;sex_reported", "unique tumor diagnosis to paper Table 1",
        "Six uniquely named tumor cores join to paper Sample IDs; patient_key is a study-local "
        "surrogate, not a source patient ID. Two tonsil patient keys remain null.")
    add(relative(GEOMETRY), "cores.geometry_id", "sample_metadata", "sample_id",
        "geometry_id", "Each separated physical core is one biological sample; use its image ID.")
    add(relative(CROSSWALK), "cores.number", "sample_metadata", "sample_type",
        "numbered physical core", "Each numbered physical TMA core is a tissue core.")
    add(relative(CROSSWALK), "cores.diagnosis;tissue", "sample_metadata", "tissue_type",
        "numbered overview spatial order=geometry_id order",
        "Map six carcinoma cores to tumor and two normal tonsil cores to normal; "
        "preserve the anatomic label in anatomical_site.")
    add(relative(CROSSWALK), "cores.tissue", "sample_metadata", "anatomical_site",
        "numbered overview spatial order=geometry_id order",
        "Copy the reviewed anatomic tissue label for each numbered core.")
    add(relative(CROSSWALK), "patient_link_rule", "review_results", "M01", "core number",
        "Local ZIP/QuPath metadata, author numbered overview, Zenodo and paper Table 1 inspected; "
        "8 core samples, 6 patient-linked samples and 2 unresolved tonsil links.")
    return (pd.DataFrame(samples, columns=columns("sample_metadata")),
            pd.DataFrame(fields, columns=columns("field_mapping")))


def image_row(item: dict, result: dict, destination: Path) -> dict:
    x0, y0, x1, y1 = item["box"]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"], sample_id=item["image_id"],
               panel_id="PhenoCycler_P01", image_kind="tma_core",
               source_image_id=MEMBER, source_files=[relative(ARCHIVE)], source_format="qptiff",
               source_axes="CYX", source_dtype="uint8", source_revision=item["revision"],
               slide_id="240508_VET_QCTMA2_A2_55p", core_id=str(item["number"]),
               roi_id=item["image_id"], region_annotation=item["clinical"]["diagnosis"],
               source_x0_px=x0, source_y0_px=y0, source_x1_px=x1, source_y1_px=y1,
               pixel_size_x_um=0.5055777729734726, pixel_size_y_um=0.5055777729734726,
               input_image_path=relative(destination),
               intensity_description="Native QPTIFF full-resolution uint8 fluorescence intensity; no normalization",
               split_method="reviewed_dearray",
               split_evidence=f"Reviewed native envelope and author numbered TMA overview, core {item['number']}.",
               metadata_notes="Native source verification and clinical linkage rules are cited in field_mapping.")
    for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = result[key]
    return row


def check_pixels(item: dict, source: Path, destination: Path, result: dict, panel: pd.DataFrame) -> None:
    x0, y0, x1, y1 = item["box"]
    left, top = result["pad_left_px"], result["pad_top_px"]
    width, height = x1-x0, y1-y0
    blocks = [(0, 0, min(32, width), min(32, height)),
              (width//2, height//2, min(width//2+32, width), min(height//2+32, height))]
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    with formal_read_lease(destination), tifffile.TiffFile(source) as native, tifffile.TiffFile(destination) as output:
        for output_index, source_index in enumerate(kept.source_channel_index):
            for a, b, c, d in blocks:
                original = read_tiff_window(native, int(source_index), (x0+a, y0+b, x0+c, y0+d))
                delivered = read_tiff_window(output, output_index, (left+a, top+b, left+c, top+d))
                if original.dtype != delivered.dtype or not np.array_equal(original, delivered):
                    raise ValueError(f"Pixel mismatch in {item['image_id']} channel {source_index}")


def existing_input_result(item: dict, destination: Path, panel: pd.DataFrame) -> dict:
    """Recover the closed first TIFF after an index-only validation failure."""
    x0, y0, x1, y1 = item["box"]
    width, height = x1-x0, y1-y0
    side = max(width, height)
    names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
    with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        saved_names = [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")]
        if (len(tif.pages) != len(names) or saved_names != names
                or any(page.shape != (side, side) or page.dtype != np.dtype("uint8")
                       for page in tif.pages)):
            raise ValueError(f"Unindexed TIFF differs from current panel: {destination}")
    left, top = (side-width)//2, (side-height)//2
    return dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                source_channel_count=59, channel_count=len(names), admitted=True)


def write_review(scope: list[dict]) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
    evidence = ";".join((relative(META / "native_verification.json"), relative(GEOMETRY),
                         relative(AXIS), relative(CROSSWALK), relative(S1)))
    details = {
        "S01": f"Eight complete physical-core proofs; {len(images)}/8 delivered inputs.",
        "I01": "Each image identifies source ZIP member, physical core and native envelope.",
        "I02": f"{len(images)} image/sample links; eight core sample rows.",
        "I03": "Native core envelopes with centered square padding and QuPath pixel calibration.",
        "I04": "Author numbered overview joined spatially to reviewed core geometry.",
        "P01": "Full-resolution IFD 0 DAPI and 58 ordered measured labels joined to current S1.",
        "P02": "S1 channel uses retained; excluded source rows kept in panel table.",
        "P03": f"{len(images)} published TIFF headers reopened by shared exporter.",
        "I05": f"{len(images)} source/output per-channel boundary and interior blocks compared.",
        "M01": "Eight physical core sample rows; six unique tumor diagnoses join paper Table 1; "
               "two tonsil patient links unresolved after local/public source search.",
        "M02": "Public Table 1 supplies tumor diagnosis and sex; patient age has no collection timepoint.",
        "D01": f"Preparer delivery {len(images)}/8; independent review and current input binding pending.",
    }
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                 record_id=STUDY, check_id=check, result="unresolved",
                 source_evidence=evidence, output_evidence=relative(OUTPUT),
                 details="Preparer self-check: " + detail)
            for check, detail in details.items()]
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        partial = path.with_name(path.name + ".partial")
        pd.DataFrame(rows, columns=columns("review_results")).to_csv(partial, index=False, encoding="utf-8-sig")
        os.replace(partial, path)


def run() -> None:
    scope, names, member = source_scope()
    panel = panel_from_s1(STUDY, "PhenoCycler_P01", names, S1)
    if len(panel) != 59 or int(panel.keep_channel.sum()) != 56 or int(panel.nuclear.sum()) != 1:
        raise ValueError("Current S1 no longer matches verified Yeung panel")
    samples, fields = metadata_tables(scope)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_input_tables(OUTPUT, pd.DataFrame(columns=columns("image_metadata")), samples, panel, fields)
    progress("metadata_ready", next_image=scope[0]["image_id"])
    source = source_stage(member)
    verify_source_header(source, names)
    for item in scope:
        destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
        with formal_read_lease(OUTPUT):
            current = pd.read_parquet(OUTPUT / "images.parquet")
        if item["image_id"] in set(current.image_id):
            if not destination.is_file() or not destination.stat().st_size:
                raise ValueError(f"Indexed image has no TIFF: {item['image_id']}")
            continue
        if shutil.disk_usage(ROOT).free < 100 * 1024**3 + 56 * max(
                item["box"][2]-item["box"][0], item["box"][3]-item["box"][1])**2:
            progress("waiting_io", next_image=item["image_id"], issue="NAS 100 GiB reserve")
            write_review(scope)
            return
        progress("exporting", current_image=item["image_id"], next_image=item["image_id"])
        try:
            if destination.is_file() and item["image_id"].endswith("_r01"):
                result = existing_input_result(item, destination, panel)
            else:
                spec = SourceSpec(item["image_id"], "PhenoCycler_P01", "tiff", source,
                                  tuple(range(59)), window=tuple(item["box"]),
                                  pixel_size_um=(0.5055777729734726, 0.5055777729734726),
                                  source_revision=item["revision"])
                with open_source(spec) as reader:
                    result = export_input(reader, panel, destination)
            if not result["admitted"]:
                raise ValueError(f"No admitted marker signal: {item['image_id']}")
            check_pixels(item, source, destination, result, panel)
            save_input_tables(OUTPUT, pd.DataFrame([image_row(item, result, destination)]), samples, panel, fields)
            progress("image_completed", last_image=item["image_id"],
                     next_image=next((x["image_id"] for x in scope if x["image_id"] not in
                                      set(pd.read_parquet(OUTPUT / "images.parquet").image_id)), None))
        except OSError as error:
            if error.errno not in {5, 28, 110, 116}:
                raise
            progress("waiting_io", next_image=item["image_id"],
                     issue=f"I/O or capacity error: errno {error.errno}: {error.strerror}",
                     local_source_stage=str(source))
            write_review(scope)
            return
    write_review(scope)
    progress("prepared_pending_independent_review", next_image=None)
    source.unlink()
    source.with_name(source.name + ".source.json").unlink()


def repair() -> None:
    scope, _, _ = source_scope()
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        if set(images.image_id) != {item["image_id"] for item in scope}:
            raise ValueError("Current eight-image index differs from the reviewed scope")
        if any(not (OUTPUT / "images" / item["image_id"] / "input.tiff").is_file()
               for item in scope):
            raise ValueError("Reviewed image is missing its native input TIFF")
    samples, fields = metadata_tables(scope)
    save_input_tables(OUTPUT, pd.DataFrame(columns=columns("image_metadata")), samples,
                      pd.DataFrame(columns=columns("panel_mapping")), fields)
    progress("metadata_repaired_pending_independent_review", next_image=None,
             repaired_tables=["sample_metadata.csv", "field_mapping.csv"],
             affected_checks=["M02", "M01", "D01"],
             preserved_image_count=len(images))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    elif args.repair:
        repair()
