"""Prepare native IMC inputs from Nawrocki's complete Hyperion TXT ROIs.

Inputs: accepted native TXT verification, original coordinate rows, study S1.
Outputs: square float32 CYX input TIFFs and study input tables. The publication
supplies HistoCAT cell measurements that resolve ROI-to-specimen links.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr

from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import panel_from_s1, save_input_tables


STUDY = "Nawrocki2023_IMC_MultipleMyeloma"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--job", type=Path, default=None)
args = parser.parse_args()
root = args.data_root
output = root / "processed" / STUDY
metadata = metadata_source_directory(root, STUDY)
supplement = root / "raw/public" / STUDY / "metadata/PMC10722139_supplementaryFiles.zip"
histocat = supplement.with_name("HistoCAT_Exported.zip")
crosswalk_path = root / "raw/public" / STUDY / "manifest.json"
extra_proof_path = root / "raw/public" / STUDY / "manifest.json"
discrepancy_path = root / "raw/public" / STUDY / "manifest.json"
job = args.job or Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
temporary = Path("/tmp/study_input_batch") / STUDY
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
preparer = job.name
temporary.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return metadata_source_relative(path, root).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def linked_images() -> int:
    return sum(isinstance(row.get("sample_id"), str) and bool(row["sample_id"])
               for row in images.values())


def write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def publish_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    with formal_publish_lease(path, directory=path.parent):
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(payload)
        os.replace(partial, path)


def progress(stage: str, image_id: str | None = None) -> None:
    write_json(job / "progress.json", dict(study=STUDY, stage=stage,
        expected_images=len(accepted), completed_images=len(images),
        next_image_id=image_id, sample_rows=len(samples),
        sample_linked_images=linked_images(),
        patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
        updated_at_utc=datetime.now(timezone.utc).isoformat()))


def field_rule(source_file: str, source_field: str, target_table: str,
               target_field: str, rule: str, *, notes: str = "") -> dict:
    return dict(study=STUDY, source_file=source_file, source_sheet_or_table="",
        source_field=source_field, target_table=target_table, target_field=target_field,
        join_keys="", value_mapping="{}", transformation_rule=rule,
        time_origin_rule="", conflict_rule="Unverified joins remain empty", notes=notes)


def check_tiff(path: Path, side: int, names: list[str], valid_path: Path | None) -> None:
    with tifffile.TiffFile(path) as image:
        series = image.series[0]
        assert series.shape == (len(names), side, side)
        assert series.dtype == np.dtype("float32")
        assert image.ome_metadata is not None
        assert all(f'Name="{name}"' in image.ome_metadata for name in names)
    if valid_path is not None:
        assert valid_path.is_file()
        with tifffile.TiffFile(valid_path) as image:
            assert image.pages[0].shape == (side, side)
            assert image.pages[0].dtype == np.dtype("uint8")


def stage_image(source: Path, record: dict, panel: pd.DataFrame,
                staged: Path) -> tuple[dict, Path | None]:
    """Decode one complete source stream into its original XY grid and validity."""
    bounds = record["physical_identity"]["coordinate_ranges"]
    width, height = bounds["width_px"], bounds["height_px"]
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    names = kept.display_name.tolist()
    pixels = np.zeros((len(kept), side, side), dtype=np.float32)
    seen = np.zeros((height, width), dtype=bool)
    selected = kept.source_name.tolist()
    types = {"X": "int32", "Y": "int32", **{name: "float32" for name in selected}}
    rows = 0
    with formal_read_lease(source):
        for chunk in pd.read_csv(source, sep="\t", usecols=["X", "Y", *selected],
                                 dtype=types, chunksize=50_000):
            x, y = chunk.X.to_numpy(), chunk.Y.to_numpy()
            linear = y.astype(np.int64) * width + x
            if (np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or
                    np.any(y >= height) or len(np.unique(linear)) != len(linear) or
                    seen[y, x].any()):
                raise ValueError(f"Invalid or repeated source XY coordinates: {source.name}")
            values = chunk[selected].to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite source measurement: {source.name}")
            pixels[:, y + top, x + left] = values.T
            seen[y, x] = True
            rows += len(chunk)
    if rows != record["physical_identity"]["native_rows"] or not seen.any():
        raise ValueError(f"Source row count differs from accepted proof: {source.name}")
    downstream = kept.role.eq("downstream").to_numpy()
    if not np.any(pixels[downstream]):
        raise ValueError(f"All admitted markers have zero signal: {source.name}")

    tifffile.imwrite(staged, pixels, bigtiff=True, ome=True, compression="zlib",
                     metadata={"axes": "CYX", "Channel": {"Name": names}},
                     photometric="minisblack", maxworkers=2)
    valid = None
    if not seen.all():
        valid = staged.with_name("valid_pixels.tiff")
        mask = np.zeros((side, side), dtype=np.uint8)
        mask[top:top + height, left:left + width] = seen
        tifffile.imwrite(valid, mask, compression="zlib", photometric="minisblack")
    check_tiff(staged, side, names, valid)
    return dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                source_x0_px=0, source_y0_px=0, source_x1_px=width,
                source_y1_px=height, source_channel_count=len(record["channels"]),
                channel_count=len(kept), source_dtype="float32", measured_pixels=rows), valid


def publish_image(staged: Path, valid: Path | None, destination: Path) -> None:
    with formal_publish_lease(destination, directory=destination.parent,
                              lock_name=".input.lock"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name("input.tiff.partial")
        shutil.copyfile(staged, pending)
        os.replace(pending, destination)
        if valid is not None:
            target = destination.with_name("valid_pixels.tiff")
            pending = destination.with_name("valid_pixels.tiff.partial")
            shutil.copyfile(valid, pending)
            os.replace(pending, target)


def verify_distinct_source(verification: dict) -> dict:
    """Resolve the held public TXT by full native coordinates and channels."""
    alias = verification["native_result"]["resolved_source_aliases"][0]
    canonical = next(record for record in verification["native_result"]["images"]
                     if record["source_member"] == alias["canonical_source_member"])
    source = root / "raw/public" / STUDY / "public_original" / alias["source_member"]
    official = next(item for item in verification["source_manifest"]["sources"]
                    if item["name"] == source.name)
    stat = source.stat()
    assert stat.st_size == official["size"] == alias["binding"]["source_file_stat"]["size"]
    assert stat.st_mtime_ns == alias["binding"]["source_file_stat"]["mtime_ns"]
    with formal_read_lease(discrepancy_path):
        discrepancy = {"source_coordinate_bounds": json.loads(discrepancy_path.read_text())["preparation_sources"]["source_coordinate_bounds"]}
    bounds = discrepancy["source_coordinate_bounds"]
    width, height = bounds["xmax"] + 1, bounds["ymax"] + 1
    assert bounds["xmin"] == bounds["ymin"] == 0
    assert (width, height) != (canonical["physical_identity"]["coordinate_ranges"]["width_px"],
                               canonical["physical_identity"]["coordinate_ranges"]["height_px"])
    with formal_read_lease(extra_proof_path):
        proof = json.loads(extra_proof_path.read_text())["preparation_sources"]["extra_native"]
    assert proof["source_file_stat"] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    assert proof["channels"] == canonical["channels"]
    assert proof["width_px"] == width and proof["height_px"] == height
    assert proof["full_grid"] and proof["native_rows"] == width * height
    record = copy.deepcopy(canonical)
    record["source_member"] = source.name
    record["source_file_path"] = str(source)
    record["source_file_stat"] = proof["source_file_stat"]
    record["proof_path"] = str(extra_proof_path)
    record["accepted_proof_reused"] = False
    identity = record["physical_identity"]
    identity["id"] = source.name + "::distinct_native_frame"
    identity["source_name"] = source.name
    identity["identity_basis"] = proof["identity_basis"]
    identity["native_rows"] = proof["native_rows"]
    identity["coordinate_ranges"].update(
        x_min=0, x_max=width - 1, x_count=width, y_min=0,
        y_max=height - 1, y_count=height, z_min=0,
        z_max=proof["native_rows"] - 1, z_count=proof["native_rows"],
        width_px=width, height_px=height)
    return record


def resolve_samples() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Reuse the accepted specimen/member relation for this fixed source scope."""
    with formal_read_lease(crosswalk_path):
        links = json.loads(crosswalk_path.read_text())["preparation_sources"]["specimen_links"]
    if set(links) != set(images):
        raise ValueError("Current images differ from the accepted specimen scope")
    for image_id, link in links.items():
        if images[image_id]["sample_id"] != link["sample_id"]:
            raise ValueError("Current specimen link differs from the accepted binding")
    return samples.copy(), {image_id: {"member": item["member"]} for image_id, item in links.items()}


def save_reviews() -> None:
    details = dict(source_files=32, admitted_images=len(images),
        source_scope_resolution="The held public TXT has a distinct full native frame and a separate published HistoCAT spatial match.",
        sample_rows=len(samples), sample_linked_images=linked_images(),
        patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
        native_proof=relative(extra_proof_path), crosswalk=relative(crosswalk_path),
        gap="New image and sample await targeted independent review and D01 rebinding.")
    rows = [dict(study=STUDY, preparer=preparer, reviewer="", scope="study",
        record_id=STUDY, check_id="S01", result="unresolved",
        source_evidence=relative(extra_proof_path),
        output_evidence=relative(output / "images.parquet"),
        details=json.dumps(details, ensure_ascii=False)),
        dict(study=STUDY, preparer=preparer, reviewer="", scope="sample",
        record_id=STUDY, check_id="M01", result="unresolved",
        source_evidence="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10722139/fullTextXML; " + relative(supplement) + "; https://api.figshare.com/v2/articles/24175404",
        output_evidence=relative(output / "sample_metadata.csv"),
        details=json.dumps(details, ensure_ascii=False)),
        dict(study=STUDY, preparer=preparer, reviewer="", scope="study",
        record_id=STUDY, check_id="D01", result="unresolved",
        source_evidence=relative(metadata / "native_verification.json"),
        output_evidence=relative(output),
        details="Repair round 1, preparer self-check only; independent reviewer and input_binding pending. " +
                json.dumps(details, ensure_ascii=False))]
    path = output / "review_results.csv"
    current = pd.read_csv(path, keep_default_na=False) if path.is_file() else pd.DataFrame(columns=columns("review_results"))
    new = pd.DataFrame(rows, columns=columns("review_results"))
    keys = ["preparer", "scope", "record_id", "check_id"]
    current = current[~current[keys].apply(tuple, axis=1).isin(new[keys].apply(tuple, axis=1))]
    payload = pd.concat([current, new], ignore_index=True).to_csv(index=False).encode("utf-8-sig")
    with formal_publish_lease(path, directory=output):
        pending = path.with_name("review_results.csv.partial")
        pending.write_bytes(payload)
        os.replace(pending, path)


with formal_read_lease(metadata / "native_verification.json"):
    verification = json.loads((metadata / "native_verification.json").read_text())
accepted = verification["native_result"]["images"]
assert verification["scope_complete"] and verification["native_result"]["status"] == "passed"
assert len(accepted) == 31 and len(verification["source_manifest"]["sources"]) == 32
assert len(verification["native_result"]["resolved_source_aliases"]) == 1
assert supplement.is_file()
assert histocat.is_file()
write_json(job / "progress.json", dict(study=STUDY, stage="verifying_distinct_source",
    expected_images=32, completed_images=31,
    next_image_id="20190215_ROI 01_1 (1).txt",
    updated_at_utc=datetime.now(timezone.utc).isoformat()))
accepted = [*accepted, verify_distinct_source(verification)]

# S1 resolves biological measurements; six acquisition fields are retained as
# excluded panel rows, and the two Ir channels keep their segmentation role.
panels = {}
for record in accepted:
    native = record["channels"]
    panel_id = STUDY + ("__original47" if len(native) == 47 else "__standard39")
    if panel_id in panels:
        assert panels[panel_id].source_name.tolist() == native
        continue
    with formal_read_lease(s1_path):
        panel = panel_from_s1(STUDY, panel_id, native[6:], s1_path)
    panel.source_channel_index += 6
    technical = [dict(panel_id=panel_id, source_channel_index=index, source_name=name,
        marker_id="", mapping_status="excluded", role="exclude", nuclear=False,
        body=False, keep_channel=False, output_channel_index=pd.NA, keep_reason="",
        exclusion_reason="Native acquisition field, not a measured marker",
        s1_original_marker="", display_name="") for index, name in enumerate(native[:6])]
    panel["display_name"] = panel.marker_id
    panel.loc[panel.role.eq("segmentation-only"), "display_name"] = panel.source_name
    panel = pd.concat([pd.DataFrame(technical), panel.astype(object)], ignore_index=True)
    assert panel.source_name.tolist() == native
    panels[panel_id] = panel
panel_table = pd.concat(panels.values(), ignore_index=True)
with formal_read_lease(output / "sample_metadata.csv"):
    samples = pd.read_csv(output / "sample_metadata.csv")
source_note = "Article confirms six anonymous IMC patients and pre/post marrow samples. HistoCAT cell tables identify specimens when their spatial measurements uniquely match source pixels; supplementary clinical Table S2 supplies no per-patient clinical values."
fields = pd.DataFrame([
    field_rule(relative(metadata / "native_verification.json"), "physical_image_id", "image_metadata", "image_id",
               "Preserve the 31 previously accepted ROI identities"),
    field_rule(relative(extra_proof_path), "distinct native coordinate frame", "image_metadata", "image_id",
               "Add the 32nd public ROI after full-channel, full-grid source verification"),
    field_rule(relative(metadata / "native_verification.json"), "source_channel", "panel_mapping", "source_name",
               "Preserve ordered native TXT header; match measured labels to study S1"),
    field_rule("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10722139/fullTextXML",
               "IMC methods", "sample_metadata", "sample_id", "No per-ROI join is published", notes=source_note),
    field_rule(relative(crosswalk_path), "source image to HistoCAT cell table",
               "image_metadata", "sample_id", "Unique ROI number, coordinate coverage and spatial CD138 measurement correlation",
               notes=source_note),
    field_rule(relative(histocat), "HistoCAT member patient/timepoint label",
               "sample_metadata", "patient_key", "Parse published anonymous patient code after spatial match", notes=source_note),
    field_rule(relative(histocat), "HistoCAT member patient label",
               "sample_metadata", "source_subject_id", "Preserve published anonymous patient code", notes=source_note),
    field_rule(relative(histocat), "HistoCAT member patient and PRE/POST labels",
               "sample_metadata", "source_sample_id", "Group all ROIs of one patient and timepoint", notes=source_note),
    field_rule(relative(histocat), "HistoCAT member PRE/POST label",
               "sample_metadata", "timepoint_label", "PRE=pretreatment; POST=posttreatment", notes=source_note),
    field_rule(relative(supplement),
               "Supplementary Methods and Table S2", "sample_metadata", "patient_key",
               "No published TXT ROI or acquisition date key in these attachments", notes=source_note),
], columns=columns("field_mapping"))
for target, source_field in {
    "sample_type": "bone marrow specimens", "anatomical_site": "bone marrow specimens",
    "collection_method": "bone marrow biopsy samples", "preservation_method": "FFPE sections",
    "primary_diagnosis": "multiple myeloma", "primary_site": "bone marrow",
    "case_control_status": "patients with multiple myeloma",
}.items():
    fields.loc[len(fields)] = field_rule(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10722139/fullTextXML",
        source_field, "sample_metadata", target, "Apply article Methods to matched IMC specimens")

output.mkdir(parents=True, exist_ok=True)
with formal_read_lease(output / "images.parquet"):
    existing = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").is_file() else pd.DataFrame(columns=columns("image_metadata"))
images = {row.image_id: row._asdict() for row in existing.itertuples(index=False)}
progress("source_checked", accepted[0]["source_member"])
for record in accepted:
    source = root / "raw/public" / STUDY / "public_original" / record["source_member"]
    stat = source.stat()
    proof = record["source_file_stat"]
    if (stat.st_size, stat.st_mtime_ns) != (proof["size"], proof["mtime_ns"]):
        raise ValueError(f"Source changed after accepted native verification: {source.name}")
    native = record["channels"]
    panel_id = STUDY + ("__original47" if len(native) == 47 else "__standard39")
    panel = panels[panel_id]
    image_id = STUDY + "__" + re.sub(r"[^A-Za-z0-9]+", "_", source.stem).strip("_")
    if image_id in images and source.name != accepted[-1]["source_member"]:
        continue
    destination = output / "images" / image_id / "input.tiff"
    valid_destination = destination.with_name("valid_pixels.tiff")
    bounds = record["physical_identity"]["coordinate_ranges"]
    side = max(bounds["height_px"], bounds["width_px"])
    names = panel.loc[panel.keep_channel].sort_values("output_channel_index").display_name.tolist()
    expected_valid = (bounds["width_px"] * bounds["height_px"] !=
                      record["physical_identity"]["native_rows"])
    if destination.is_file() and (not expected_valid or valid_destination.is_file()):
        with formal_read_lease(destination):
            check_tiff(destination, side, names, valid_destination if expected_valid else None)
        valid = valid_destination if expected_valid else None
        geometry = dict(height_px=side, width_px=side,
            pad_left_px=(side - bounds["width_px"]) // 2,
            pad_top_px=(side - bounds["height_px"]) // 2,
            source_x0_px=0, source_y0_px=0, source_x1_px=bounds["width_px"],
            source_y1_px=bounds["height_px"],
            source_channel_count=len(native), channel_count=len(names), source_dtype="float32")
        geometry["pad_right_px"] = side - bounds["width_px"] - geometry["pad_left_px"]
        geometry["pad_bottom_px"] = side - bounds["height_px"] - geometry["pad_top_px"]
        if image_id in images:
            progress("reused_image", image_id)
            continue
    else:
        progress("reading_source", image_id)
        with tempfile.TemporaryDirectory(prefix="nawrocki-", dir=temporary) as local:
            staged = Path(local) / "input.tiff"
            geometry, valid = stage_image(source, record, panel, staged)
            progress("publishing_image", image_id)
            publish_image(staged, valid, destination)
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=image_id, sample_id=None, panel_id=panel_id,
        image_kind="roi", source_image_id=record["physical_identity"]["id"],
        source_files=[relative(source)], source_format="Hyperion TXT coordinate table",
        source_acquisition_id=source.stem, source_axes="row,channel; X/Y pixel coordinates",
        source_revision="10.6084/m9.figshare.24175404.v1", roi_id=source.stem,
        region_annotation=source.stem, input_image_path=relative(destination),
        valid_pixels_path=relative(valid_destination) if valid else None,
        intensity_description="Original TXT marker values decoded as float32; no normalization or clipping.",
        split_method="already_independent", split_evidence=relative(
            extra_proof_path if source.name == accepted[-1]["source_member"]
            else metadata / "native_verification.json"),
        metadata_notes=source_note + " Native TXT XY rows reconstructed without resampling; centered zero padding.",
        **{key: value for key, value in geometry.items() if key != "measured_pixels"})
    images[image_id] = row
    save_input_tables(output, pd.DataFrame(list(images.values())), samples, panel_table, fields)
    progress("image_committed", image_id)

samples, linked = resolve_samples()
save_input_tables(output, pd.DataFrame(list(images.values())), samples, panel_table, fields)
save_reviews()
progress("repair_exported_awaiting_independent_review")
print(json.dumps(dict(study=STUDY, exported_images=len(images), expected_images=len(accepted),
    sample_rows=len(samples), sample_linked_images=len(linked),
    patient_linked_samples=int(samples.patient_key.notna().sum())), ensure_ascii=False))
