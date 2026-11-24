"""Prepare native CODEX inputs from the nine local HTAN WUSTL acquisitions.

Inputs: HTAN Release 4.0 TIFFs, ordered panel audit, study S1, and the
publication's CODEX specimen lookup and clinical supplement. Outputs: CYX
input.tiff and per-study input tables. Run with --run; one image is closed and
indexed before the next begins. Identity conflicts remain unresolved metadata.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import pandas as pd
import tifffile
import numpy as np
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Mo2024_CODEX_MultiCancer"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0253_prepare_0")))
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
LOOKUP = META / "source_metadata" / "Sample_ID_Lookup_table_v1.xlsx"
CLINICAL = META / "source_metadata" / "41586_2024_8087_MOESM3_ESM.xlsx"
PREPARER = "codex_mo2024_" + JOB.name


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return metadata_source_relative(path, ROOT).as_posix()


def template(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def publish_local(source: Path, destination: Path) -> None:
    with formal_publish_lease(destination, directory=destination.parent):
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, partial)
        os.replace(partial, destination)


def checkpoint(stage: str, **details: object) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    state = dict(study=STUDY, phase="prepare", scientific_repair_round=0,
                 stage=stage, expected_images=9, exported_images=len(images),
                 sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / "progress.json"
    staged.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    publish_local(staged, JOB / "progress.json")
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> list[dict]:
    with formal_read_lease(META):
        manifest = json.loads((META / "source_manifest.json").read_text())
        ordered = json.loads((META / "source_ordered_channel_map.json").read_text())
    by_name = {name: panel for panel in ordered["panels"]
               for name in panel["accepted_filename_ids"]}
    assert manifest["accepted_file_count"] == len(manifest["accepted_files"]) == 9
    scope = []
    for entry in manifest["accepted_files"]:
        path = ROOT / entry["relative_path"]
        stat = path.stat()
        if stat.st_size != entry["expected_size_bytes"]:
            raise ValueError(f"Source size differs from qualified manifest: {path.name}")
        panel = by_name[entry["filename"]]
        if panel["panel_id"] != entry["panel_id"] or not panel["accepted"]:
            raise ValueError(f"Panel assignment differs from qualified manifest: {path.name}")
        scope.append(dict(entry=entry, source=path, panel=panel,
                          image_id=path.name.removesuffix(".ome.tiff"),
                          source_revision=f"HTAN_Release_4.0;size={stat.st_size};mtime_ns={stat.st_mtime_ns}"))
    return scope


def identity_tables(scope: list[dict]) -> tuple[pd.DataFrame, dict[str, dict], list[str], list[str]]:
    """Join by both HTAN biospecimen and exact published CODEX acquisition name."""
    with formal_read_lease(ROOT / "raw/public" / STUDY / "metadata.json"):
        lookup = pd.read_excel(LOOKUP, sheet_name="CODEX", dtype=str)
        clinical = pd.read_excel(CLINICAL, sheet_name="4_clinical_data", dtype=str)
    lookup = lookup.set_index("HTAN Biospecimen ID", verify_integrity=True)
    clinical = clinical.set_index(["Case_ID", "Sample Type"], verify_integrity=True)
    samples, links, unresolved, clinical_conflicts = {}, {}, [], []
    for item in scope:
        entry = item["entry"]
        biospecimen = entry["portal_parent_biospecimen_id"]
        specimen = lookup.loc[biospecimen] if biospecimen in lookup.index else None
        expected_name = item["image_id"].replace("__", "_")
        if (specimen is None or specimen["WUSTL Biospecimen ID"] != expected_name or
                specimen["HTAN Participant ID"] != entry["portal_participant_id"]):
            unresolved.append(item["image_id"])
            links[item["image_id"]] = dict(sample_id=None, region_annotation=None,
                                           metadata_notes="Sample link unresolved: HTAN manifest and author CODEX lookup do not agree on exact acquisition identity.")
            continue
        clinical_type = {"Primary": "Tumor", "Metastasis": "Met"}[specimen["Primary_metastasis"]]
        clinical_key = (specimen["WUSTL Participant ID"], clinical_type)
        case = clinical.loc[clinical_key, :] if clinical_key in clinical.index else None
        if case is None:
            clinical_conflicts.append(item["image_id"])
        sample = dict.fromkeys(template("sample_metadata"))
        sample.update(study=STUDY, sample_id=biospecimen,
                      patient_key=entry["portal_participant_id"],
                      source_subject_id=specimen["WUSTL Participant ID"],
                      source_sample_id=specimen["WUSTL Biospecimen ID"],
                      sample_type="tissue section", primary_diagnosis=specimen["Cancer Type"],
                      lesion_type={"Primary": "primary", "Metastasis": "metastasis"}.get(
                          specimen["Primary_metastasis"]),
                      age_at_diagnosis_years=case["age at diagnosis"] if case is not None else None,
                      age_at_collection_years=case["age at collection date"] if case is not None else None,
                      sex_reported={"F": "female", "M": "male"}.get(case["sex"]) if case is not None else None,
                      metadata_notes=("HTAN parent biospecimen and participant agree with exact author CODEX lookup name; "
                                      "clinical values joined by WUSTL Participant ID and lesion type."
                                      if case is not None else
                                      "HTAN and author specimen identity agree; publication clinical lesion type conflicts, so clinical values are withheld."))
        samples[biospecimen] = sample
        links[item["image_id"]] = dict(sample_id=biospecimen,
                                       region_annotation=specimen["Primary_metastasis"],
                                       metadata_notes="Exact author CODEX lookup and HTAN biospecimen/participant agreement.")
    return (pd.DataFrame(samples.values(), columns=template("sample_metadata")),
            links, unresolved, clinical_conflicts)


def field_rows() -> pd.DataFrame:
    common = dict(study=STUDY, source_sheet_or_table="", value_mapping="{}",
                  time_origin_rule="", conflict_rule="Conflicting identity remains unlinked")
    definitions = [
        (META / "source_manifest.json", "portal_parent_biospecimen_id; portal_participant_id; filename",
         "image_metadata", "sample_id", "HTAN biospecimen + exact CODEX acquisition name + participant",
         "Link only rows agreeing with the author CODEX lookup"),
        (LOOKUP, "HTAN Biospecimen ID; WUSTL Biospecimen ID; HTAN Participant ID",
         "sample_metadata", "sample_id; patient_key; source_sample_id; source_subject_id",
         "HTAN Biospecimen ID + exact WUSTL Biospecimen ID + HTAN Participant ID",
         "Preserve the author specimen and patient identities after exact agreement"),
        (LOOKUP, "Cancer Type; Primary_metastasis", "sample_metadata",
         "primary_diagnosis; lesion_type", "HTAN Biospecimen ID",
         "Copy cancer code; Primary -> primary and Metastasis -> metastasis"),
        (CLINICAL, "age at diagnosis; age at collection date; sex", "sample_metadata",
         "age_at_diagnosis_years; age_at_collection_years; sex_reported",
         "WUSTL Participant ID = Case_ID + Primary_metastasis = Sample Type",
         "Copy reported ages; F -> female and M -> male"),
        (LOOKUP, "Primary_metastasis", "image_metadata", "region_annotation",
         "HTAN Biospecimen ID + exact acquisition name", "Copy source category without reclassification"),
        (META / "source_ordered_channel_map.json", "native_channel_order_from_audit",
         "panel_mapping", "source_name", "panel_id + source_channel_index",
         "Preserve qualified OME channel order and join study S1"),
    ]
    rows = []
    for source, source_field, target_table, target_field, join_keys, rule in definitions:
        rows.append(dict(**common, source_file=relative(source), source_field=source_field,
                         target_table=target_table, target_field=target_field,
                         join_keys=join_keys, transformation_rule=rule,
                         notes="9 source images; 7 exact image-to-sample matches; 7 sample-to-patient links; 2 identity gaps retained."))
    return pd.DataFrame(rows, columns=template("field_mapping"))


def source_header(path: Path, panel: pd.DataFrame) -> tuple[int, int, str, tuple[float, float] | None]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "CYX" or len(tif.pages) != len(panel) or series.shape[0] != len(panel):
            raise ValueError(f"Unexpected source TIFF axes/pages: {path.name}")
        ome = ET.fromstring(tif.ome_metadata)
        pixels = ome.find(".//{*}Pixels")
        names = [node.attrib["Name"] for node in ome.findall(".//{*}Channel")]
        if names != panel.sort_values("source_channel_index").source_name.tolist():
            raise ValueError(f"OME channel order differs from qualified panel: {path.name}")
        scale = None
        if "PhysicalSizeX" in pixels.attrib and "PhysicalSizeY" in pixels.attrib:
            scale = (float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"]))
        return series.shape[1], series.shape[2], str(series.dtype), scale


def image_record(item: dict, geometry: dict, shape: tuple[int, int, str, tuple[float, float] | None],
                 link: dict, destination: Path) -> dict:
    h, w, dtype, scale = shape
    row = dict.fromkeys(template("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"], panel_id=item["panel"]["panel_id"],
               image_kind="roi", source_image_id=item["entry"]["source_uuid"],
               source_files=[relative(item["source"])], source_format="ome_tiff",
               source_axes="CYX", source_dtype=dtype, source_revision=item["source_revision"],
               roi_id=item["image_id"], source_x0_px=0, source_y0_px=0,
               source_x1_px=w, source_y1_px=h,
               pixel_size_x_um=scale[0] if scale else None,
               pixel_size_y_um=scale[1] if scale else None,
               input_image_path=relative(destination),
               intensity_description="Published CODEX uint8 native intensity, no added normalization",
               split_method="already_independent",
               split_evidence="HTAN Release 4.0 independent CODEX OME-TIFF acquisition",
               **link)
    for key in ("height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = geometry[key]
    return row


def review_rows(unresolved: list[str], clinical_conflicts: list[str], images: pd.DataFrame) -> pd.DataFrame:
    sources = [META / "source_manifest.json", META / "source_ordered_channel_map.json",
               LOOKUP, CLINICAL, S1]
    outputs = [OUTPUT / name for name in ("images.parquet", "sample_metadata.csv",
               "panel_mapping.csv", "field_mapping.csv")]
    pending = [("study", STUDY, "D01", images.image_id.tolist(),
                "Independent review and input binding pending")]
    pending += [("image", image_id, "M01", [image_id],
                 "Source manifest and author CODEX lookup lack a consistent exact specimen identity")
                for image_id in unresolved]
    pending += [("image", image_id, "M02", [image_id],
                 "Author CODEX lesion type conflicts with publication clinical table; clinical values withheld")
                for image_id in clinical_conflicts]
    rows = []
    for scope, record_id, check_id, image_ids, reason in pending:
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                         record_id=record_id, check_id=check_id, result="unresolved",
                         source_evidence=json.dumps([relative(path) for path in sources]),
                         output_evidence=json.dumps([relative(path) for path in outputs]),
                         details=json.dumps(dict(preparer_only=True, independent_review="pending",
                                                 image_ids=image_ids, reason=reason))))
    return pd.DataFrame(rows, columns=template("review_results"))


def publish_preparer_review(unresolved: list[str], clinical_conflicts: list[str]) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
    reviews = review_rows(unresolved, clinical_conflicts, images)
    staged = TEMP / "review_results.csv"
    reviews.to_csv(staged, index=False, encoding="utf-8-sig")
    publish_local(staged, OUTPUT / "review_results.csv")


def source_overview(scope: list[dict]) -> None:
    """Inspect DAPI, epithelial and immune signal across published source pyramids."""
    sheet = Image.new("RGB", (3 * 360, 3 * 390), "white")
    draw = ImageDraw.Draw(sheet)
    for position, item in enumerate(scope):
        labels = item["panel"]["native_channel_order_from_audit"]
        selected = [labels.index("DAPI"),
                    next(i for i, name in enumerate(labels) if name.startswith("Pan") and "Cytokeratin" in name),
                    next(i for i, name in enumerate(labels) if name.startswith("CD68"))]
        with formal_read_lease(item["source"]):
            with tifffile.TiffFile(item["source"]) as tif:
                level = tif.series[0].levels[-1]
                channels = np.stack([level.pages[i].asarray(maxworkers=2) for i in selected])
        channels = channels.astype("float32")
        scale = np.maximum(np.percentile(channels, 99.5, axis=(1, 2)), 1)
        rgb = np.moveaxis(np.clip(channels / scale[:, None, None] * 255, 0, 255).astype("uint8"), 0, -1)
        picture = Image.fromarray(rgb)
        picture.thumbnail((350, 350))
        x, y = (position % 3) * 360, (position // 3) * 390
        sheet.paste(picture, (x, y + 30))
        draw.text((x + 2, y + 4), item["image_id"][:37], fill="black")
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / "source_overview.png"
    sheet.save(staged)
    publish_local(staged, CACHE / "source_overview.png")


def run() -> None:
    progress_path = Path(os.environ.get("CELLWEFT_STUDY_INPUT_RESUME", str(JOB / "progress.json")))
    resume = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    scope = source_scope()
    samples, links, unresolved, clinical_conflicts = identity_tables(scope)
    with formal_read_lease(S1):
        panels = pd.concat([panel_from_s1(STUDY, group["panel_id"],
                             group["native_channel_order_from_audit"], S1).drop(columns="s1_original_marker")
                            for group in {item["panel"]["panel_id"]: item["panel"] for item in scope}.values()],
                           ignore_index=True)
    fields = field_rows()
    resume_details = {key: resume[key] for key in
                      ("next_image", "staged_source_revision", "staged_bytes") if key in resume}
    checkpoint("sources_and_identity_checked", unresolved_image_count=len(unresolved),
               clinical_conflict_image_count=len(clinical_conflicts), **resume_details)
    reused_images = 0
    with formal_read_lease(OUTPUT):
        prior = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
    for item in sorted(scope, key=lambda entry: entry["entry"]["expected_size_bytes"]):
        image_id = item["image_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        existing = prior[prior.image_id.eq(image_id)] if len(prior) else pd.DataFrame()
        if len(existing) and existing.iloc[0].source_revision == item["source_revision"] and destination.is_file():
            reused_images += 1
            continue
        TEMP.mkdir(parents=True, exist_ok=True)
        local = TEMP / item["source"].name
        source_bytes = item["source"].stat().st_size
        resume_staged_source = (
            resume.get("next_image", resume.get("current_image")) == image_id
            and resume.get("staged_source_revision") == item["source_revision"]
            and local.is_file() and local.stat().st_size <= source_bytes
        )
        if shutil.disk_usage(TEMP).free < 64 * 1024**3 + source_bytes:
            checkpoint("waiting_io", next_image=image_id,
                       staged_source_revision=item["source_revision"],
                       staged_bytes=local.stat().st_size if resume_staged_source else 0,
                       reason="Local staging reserve below 64 GiB")
            return
        staged_bytes = local.stat().st_size if resume_staged_source else 0
        checkpoint("source_staging", next_image=image_id,
                   staged_source_revision=item["source_revision"], staged_bytes=staged_bytes,
                   reused_images=reused_images)
        if staged_bytes < source_bytes:
            try:
                with formal_read_lease(item["source"]):
                    with item["source"].open("rb") as source_file, local.open("ab" if staged_bytes else "wb") as staged_file:
                        if staged_bytes:
                            boundary = min(staged_bytes, 1024 * 1024)
                            source_file.seek(staged_bytes - boundary)
                            with local.open("rb") as existing_file:
                                existing_file.seek(staged_bytes - boundary)
                                if existing_file.read(boundary) != source_file.read(boundary):
                                    raise ValueError(f"Staged source prefix differs: {image_id}")
                        source_file.seek(staged_bytes)
                        shutil.copyfileobj(source_file, staged_file, length=8 * 1024 * 1024)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                checkpoint("waiting_io", next_image=image_id,
                           staged_source_revision=item["source_revision"],
                           staged_bytes=local.stat().st_size,
                           reason="NAS source staging returned EIO; resume local partial source")
                return
        panel = panels[panels.panel_id.eq(item["panel"]["panel_id"])]
        keep_staged_source = False
        try:
            shape = source_header(local, panel)
            h, w, dtype, scale = shape
            upper_output = len(panel[panel.keep_channel]) * max(h, w)**2
            if (shutil.disk_usage(TEMP).free < 64 * 1024**3 + upper_output or
                    shutil.disk_usage(ROOT).free < 100 * 1024**3 + upper_output):
                keep_staged_source = True
                checkpoint("waiting_io", next_image=image_id,
                           staged_source_revision=item["source_revision"],
                           staged_bytes=local.stat().st_size,
                           reason="Input output estimate exceeds local 64 GiB or NAS 100 GiB reserve")
                return
            source = SourceSpec(image_id, item["panel"]["panel_id"], "tiff", local,
                                tuple(range(len(panel))), pixel_size_um=scale)
            checkpoint("image_exporting", current_image=image_id)
            with open_source(source) as reader:
                try:
                    geometry = export_input(reader, panel, destination, tile_size=512)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    keep_staged_source = True
                    checkpoint("waiting_io", next_image=image_id,
                               staged_source_revision=item["source_revision"],
                               staged_bytes=local.stat().st_size,
                               reason="NAS input.tiff publication returned EIO; formal input absent; retry this image")
                    return
            if not geometry["admitted"]:
                checkpoint("image_excluded_no_marker_signal", current_image=image_id)
                continue
            row = image_record(item, geometry, shape, links[image_id], destination)
            save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panels, fields)
            previous = prior[~prior.image_id.eq(image_id)] if len(prior) else prior
            prior = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
            publish_preparer_review(unresolved, clinical_conflicts)
            checkpoint("image_completed", current_image=image_id)
        finally:
            if not keep_staged_source:
                local.unlink(missing_ok=True)
    publish_preparer_review(unresolved, clinical_conflicts)
    checkpoint("prepared_pending_independent_review", unresolved_image_count=len(unresolved),
               clinical_conflict_image_count=len(clinical_conflicts), reused_images=reused_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--overview", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    elif args.overview:
        source_overview(source_scope())
    else:
        parser.print_help()
