"""Prepare the eight admitted SIMPLI normal-colon IMC acquisitions.

Inputs: complete Zenodo 5545882 data.zip, accepted native verification, S1,
and study-cached publication/repository metadata. Outputs: native input TIFFs,
39-field image index, sample/panel/field tables, preparation checks and receipts.
Run exports through the batch shared formal write lease wrapper. CRC_4's absent source
file and the unproven CRC/CLN alias remain explicit metadata gaps.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile
from xml.etree import ElementTree

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Bortolomeazzi2022_IMC_NormalColon_SIMPLI/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Bortolomeazzi2022_IMC_NormalColon_SIMPLI"
PREPARER = "0017_prepare_0"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--verify-existing-only", action="store_true",
                    help="Recover completed TIFFs after interrupted self-checks; no extraction or TIFF writes")
parser.add_argument("--progress", type=Path, default=PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0017_prepare_0/progress.json")
args = parser.parse_args()
args.progress.parent.mkdir(parents=True, exist_ok=True)
root = args.data_root
metadata = metadata_source_directory(root, STUDY)
cache = PROJECT / ".cache/cellweft/runtime/study_input" / STUDY
evidence = metadata / "source_metadata"
output = root / "processed" / STUDY
receipts = cache / "image_receipts"
temporary = Path("/tmp/study_input_batch") / STUDY
for directory in (evidence, output, receipts, temporary, cache / "qc"):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def save_json(path: Path, value: object) -> None:
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def progress(stage: str, **details: object) -> None:
    save_json(args.progress, dict(study=STUDY, stage=stage, expected_images=len(selected),
              completed_images=len(image_rows), updated_utc=datetime.now(timezone.utc).isoformat(),
              gaps=gaps, controller_requests=controller_requests, **details))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def image_key(record: dict) -> str:
    return f"{Path(record['source_member_name']).stem}__slide{record['slide_id']}__acquisition{record['source_acquisition_id']}"


def binding(record: dict) -> dict:
    return dict(source_archive=relative(archive), source_stat=source_stat,
                native_image_id=record["image_id"], panel=panel.astype(object).where(panel.notna(), None).to_dict("records"))


def checked_receipt(record: dict) -> dict | None:
    """Reuse only an already checked TIFF with the same source and channel selection."""
    path = receipts / f"{image_key(record)}.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt["input_binding"] != binding(record):
        raise ValueError(f"Changed input binding requires a targeted repair: {path}")
    product = root / receipt["image_row"]["input_image_path"]
    stat = product.stat()
    if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != receipt["output_stat"]:
        raise ValueError(f"Changed completed TIFF requires a targeted repair: {product}")
    return receipt


def source_sample(record: dict) -> str | None:
    """Join both exact MCD member basename and exact acquisition description."""
    rows = raw_samples[(raw_samples.file_name.map(lambda value: Path(value).name) ==
                        Path(record["source_member_name"]).name) &
                       (raw_samples.roi_name == record["description"])]
    if len(rows) > 1:
        raise ValueError("Ambiguous source specimen crosswalk")
    return rows.iloc[0].sample_name if len(rows) else None


def field_rule(target_table: str, target_field: str, source_file: str, source_field: str,
               rule: str, *, join: str = "", notes: str = "") -> dict:
    return dict(study=STUDY, source_file=source_file, source_sheet_or_table="",
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join, value_mapping="{}", transformation_rule=rule,
                time_origin_rule="", conflict_rule="Unproven joins remain empty", notes=notes)


def preparation_checks() -> pd.DataFrame:
    """Record performed preparation checks; independent D01 acceptance stays pending."""
    checks = []
    visual_path = cache / "qc/visual_checks.json"
    visual_images = json.loads(visual_path.read_text())["images"] if visual_path.exists() else {}

    def add(scope: str, record_id: str, check_id: str, result: str, details: object) -> None:
        checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                      record_id=record_id, check_id=check_id, result=result,
                      source_evidence=relative(metadata / "native_verification.json"),
                      output_evidence=relative(output), details=json.dumps(dict(
                          assessment="preparer_self_check; independent review pending", evidence=details),
                          ensure_ascii=False)))

    add("study", STUDY, "S01", "pass", dict(archive_members=len(members),
        native_acquisitions=len(selected), exported=len(image_rows),
        excluded_images=excluded, original_verification_reused=True))
    for check_id in ("P01", "P02"):
        add("panel", panel_id, check_id, "pass", dict(source_channels=len(panel),
            retained_channels=len(kept), nuclear=panel.loc[panel.nuclear, "source_name"].tolist(),
            s1=relative(s1_path), omitted_from_analysis_csv_but_native_admitted=["144Nd-CD14", "149Sm-CCR4"]))
    for record in selected:
        key = image_key(record)
        if key not in image_rows:
            continue
        receipt = json.loads((receipts / f"{key}.json").read_text())
        for check_id in ("I01", "I02", "I03", "I04", "P03", "I05"):
            result = "unresolved" if check_id == "I04" and key not in visual_images else "pass"
            add("image", key, check_id, result, dict(receipt=relative(receipts / f"{key}.json"),
                sample_id=image_rows[key]["sample_id"], source_region=record["description"],
                visual_check=relative(visual_path) if check_id == "I04" and key in visual_images else None,
                pixel_checks=receipt["pixel_checks"] if check_id == "I05" else None))
    linked = sum(bool(row["sample_id"]) for row in image_rows.values())
    add("sample", STUDY, "M01", "unresolved", dict(sample_rows=len(samples),
        sample_linked_images=linked, total_images=len(image_rows), patient_linked_samples=0,
        lookup_evidence=relative(evidence / "sample_trace.json"), gaps=gaps))
    add("sample", STUDY, "M02", "pass", dict(filled_fields=samples.loc[:, samples.notna().any()].columns.tolist(),
        rule=relative(output / "field_mapping.csv"),
        missing_clinical_values="CLN-specific clinical values await an explicit CRC-to-CLN crosswalk; patient fields are empty"))
    add("study", STUDY, "D01", "unresolved", dict(exported_images=len(image_rows),
        expected_images=len(selected), sample_rows=len(samples), sample_linked_images=linked,
        patient_linked_samples=0, independent_reviewer_pending=True, gaps=gaps))
    return pd.DataFrame(checks, columns=columns("review_results"))


def save_tables() -> None:
    """Merge completed acquisitions and publish after each image."""
    if image_rows:
        images = pd.DataFrame(image_rows.values()).reindex(columns=columns("image_metadata"))
        save_input_tables(output, images, samples, panel, mapping)
    else:
        with formal_publish_lease(output, directory=output):
            samples.to_csv(output / "sample_metadata.csv", index=False, encoding="utf-8-sig")
            panel.to_csv(output / "panel_mapping.csv", index=False, encoding="utf-8-sig")
            mapping.to_csv(output / "field_mapping.csv", index=False, encoding="utf-8-sig")
    review = preparation_checks()
    with formal_publish_lease(output, directory=output):
        review.to_csv(output / "review_results.csv", index=False, encoding="utf-8-sig")


def scalar_pages(tif: tifffile.TiffFile) -> SimpleNamespace:
    """Adapt OME TiffFrames for the shared scalar-page window reader.

    Reading tif.series materializes TiffFrames, which lack samplesperpixel in
    the installed tifffile. aspage() loads only page directories, not pixels.
    This observed shared-reader issue should migrate to src via the controller.
    """
    return SimpleNamespace(pages=[page.aspage() for page in tif.pages], filehandle=tif.filehandle)


def recover_geometry(reader: object, destination: Path) -> dict:
    """Recover native statistics after the observed post-write TiffFrame failure.

    export_input had already atomically closed/renamed the TIFF. Reconstruct
    geometry and admission statistics, then run header, block and padding checks
    below; preserve the existing TIFF bytes and mtime.
    """
    h, w = reader.shape[-2:]
    side = max(h, w)
    sums = np.zeros(len(kept), dtype=np.float64)
    nonzero = np.zeros(len(kept), dtype=np.int64)
    measured, missing = 0, False
    for box in windows((h, w), 512):
        valid = reader.valid_window(box)
        measured += int(valid.sum())
        missing |= not bool(valid.all())
        values = reader.read_window(kept.source_channel_index.tolist(), box)
        assert np.isfinite(values[:, valid]).all()
        sums += values[:, valid].sum(axis=1, dtype=np.float64)
        nonzero += np.count_nonzero(values[:, valid], axis=1)
    if missing:
        assert destination.with_name("valid_pixels.tiff").is_file()
    return dict(height_px=side, width_px=side, pad_left_px=(side-w)//2, pad_top_px=(side-h)//2,
                pad_right_px=side-w-(side-w)//2, pad_bottom_px=side-h-(side-h)//2,
                source_dtype=str(reader.dtype), source_channel_count=reader.shape[0], channel_count=len(kept),
                admitted=bool(nonzero[(kept.role == "downstream").to_numpy()].any()),
                channel_nonzero_pixels=nonzero.tolist(), channel_means=(sums/measured).tolist(),
                valid_pixels="valid_pixels.tiff" if missing else None)


def verify_pixels(reader: object, destination: Path, geometry: dict) -> list[dict]:
    """Compare all saved channels at an interior and an acquired boundary block."""
    h, w = reader.shape[-2:]
    signal_row = next(row for row, count in zip(kept.itertuples(index=False), geometry["channel_nonzero_pixels"])
                      if row.role == "downstream" and count > 0)
    signal = reader.read_window([signal_row.source_channel_index], (0, 0, w, h))[0]
    signal = np.where(reader.valid_window((0, 0, w, h)), signal, 0)
    sy, sx = np.unravel_index(np.argmax(signal[16:-16, 16:-16]), (h-32, w-32))
    sy, sx = int(sy) + 16, int(sx) + 16
    assert signal[sy, sx] != 0
    boxes = [(sx-16, sy-16, sx+16, sy+16),
             (0, 0, min(32, w), min(32, h))]
    results = []
    valid_file = destination.with_name("valid_pixels.tiff")
    with tifffile.TiffFile(destination) as tif, (
            tifffile.TiffFile(valid_file) if geometry["valid_pixels"] else nullcontext(None)) as valid_tif:
        assert tif.series[0].shape == (len(kept), geometry["height_px"], geometry["width_px"])
        assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("float32")
        names = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()
        scalar = scalar_pages(tif)
        valid_scalar = scalar_pages(valid_tif) if valid_tif is not None else None
        for box in boxes:
            valid = reader.valid_window(box)
            shifted = tuple(value + (geometry["pad_left_px"] if index % 2 == 0 else geometry["pad_top_px"])
                            for index, value in enumerate(box))
            for row in kept.itertuples(index=False):
                original = reader.read_window([row.source_channel_index], box)[0]
                saved = read_tiff_window(scalar, int(row.output_channel_index), shifted)
                assert saved.dtype == original.dtype
                assert np.array_equal(saved, np.where(valid, original, 0))
            if valid_tif is not None:
                assert valid_tif.pages[0].dtype == np.dtype("uint8")
                assert np.array_equal(read_tiff_window(valid_scalar, 0, shifted), valid.astype("uint8"))
            results.append(dict(source_xyxy=box, output_xyxy=shifted,
                                source_channels=kept.source_channel_index.tolist(), exact_equal=True,
                                signal_anchor_source_channel=int(signal_row.source_channel_index),
                                source_validity_equal=True))
        side = geometry["height_px"]
        padding = [(0, 0, geometry["pad_left_px"], side),
                   (0, 0, side, geometry["pad_top_px"]),
                   (side-geometry["pad_right_px"], 0, side, side),
                   (0, side-geometry["pad_bottom_px"], side, side)]
        for box in padding:
            if box[0] < box[2] and box[1] < box[3]:
                for page in range(len(kept)):
                    assert not read_tiff_window(scalar, page, box).any()
                if valid_tif is not None:
                    assert not read_tiff_window(valid_scalar, 0, box).any()
    return results


def overview(reader: object, key: str) -> None:
    """Show DNA and epithelial/stromal/immune channels using display-only scaling."""
    h, w = reader.shape[-2:]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    step = max(1, (max(h, w) + 599)//600)
    for axis, channel in zip(axes, [52, 54, 21, 16, 25]):
        values = reader.read_window([channel], (0, 0, w, h))[0]
        values = values[::step, ::step]
        limit = float(np.percentile(values, 99.5))
        axis.imshow(values, cmap="gray", vmin=0, vmax=limit if limit > 0 else 1)
        axis.set_title(source_names[channel]); axis.axis("off")
    fig.suptitle(key + " | full native ROI; display P99.5 only")
    fig.tight_layout()
    fig.savefig(cache / "qc" / f"{key}.png", dpi=110)
    plt.close(fig)


# Reuse the completed source identity and original native proofs without hashing.
verification = json.loads((metadata / "native_verification.json").read_text())
native = verification["native_result"]
selected = native["verified_native_acquisitions"]
assert len(selected) == 8 and native["scope_complete"]
archive = root / "raw/public" / STUDY / "public_original/data.zip"
stat = archive.stat()
source_stat = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
assert source_stat == {key: native["source_file_stat"][key] for key in source_stat}
markers = json.loads((metadata / "native_markers.json").read_text())["panels"]
source_names = markers[0]["ordered_source_labels"]
assert all(row["ordered_source_labels"] == source_names for row in markers)
assert all(row["complete"] and row["counted"] for row in selected)
panel_id = f"{STUDY}__native62"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(panel) == 62 and len(kept) == 30
assert panel.loc[panel.nuclear, "source_channel_index"].tolist() == [52, 54]
with zipfile.ZipFile(archive) as source_zip:
    members = [dict(name=item.filename, size=item.file_size, compressed_size=item.compress_size)
               for item in source_zip.infolist()]
    for name in ("raw_file_metadata.csv", "channel_metadata.csv"):
        (evidence / name).write_bytes(source_zip.read("data/" + name))
raw_samples = pd.read_csv(evidence / "raw_file_metadata.csv", dtype=str, keep_default_na=False)
assert len(raw_samples) == 6 and raw_samples.sample_name.is_unique
gaps = ["8个image中3个缺少可证实的生物学sample连接；来源CSV中CRC_4引用缺失的s0129 MCD，不能替换为s0130",
        "归档CRC_1–CRC_6与论文CLN1–CLN6未发现明确别名表，补充表的分部位/临床值未并入",
        "全部6个来源sample的patient标识未公开对应；patient_key与source_subject_id留空",
        "独立复核及主控验收待执行"]
controller_requests = ["共享io/readers.py的read_tiff_window不支持OME series产生的TiffFrame（缺samplesperpixel）；"
                       "本study通过TiffFrame.aspage()构造页视图临时适配，请主控迁移修复到共享模块。"]

# The archive explicitly calls these six rows samples. Retain the unlinked row.
samples = pd.DataFrame(index=range(len(raw_samples)), columns=columns("sample_metadata"))
samples["study"] = STUDY
samples["sample_id"] = raw_samples.sample_name
samples["source_sample_id"] = raw_samples.sample_name
for field, value in dict(sample_type="FFPE tissue section", tissue_type="normal", anatomical_site="Colon",
                         collection_method="resection", preservation_method="ffpe",
                         primary_diagnosis="colorectal cancer").items():
    samples[field] = value
samples["metadata_notes"] = (
    "Biological sample identity: archive raw_file_metadata.csv; normal-colon mucosa scope, "
    "FFPE and resection: article Methods/Sample description. CRC_4 has no exact native ROI match. "
    "CRC-to-CLN and patient correspondence unresolved; no individual clinical values inferred.")
trace_path = evidence / "sample_trace.json"
assert trace_path.is_file(), "Save the actual public/local metadata search record before exporting"

# Record every base field, missing field and sample provenance in the delivered mapping.
mapping_rows = []
image_fields = columns("image_metadata")
optional_empty = {"source_z_index", "source_t_index", "core_id", "fov_id"}
for field in image_fields:
    rule = "Original native MCD acquisition/proof plus deterministic square export; see study script and per-image receipt"
    if field in optional_empty:
        rule = "Empty: no source-defined value or applicable Z/time/core/FOV selection"
    if field == "sample_id":
        rule = "Exact join on basename(file_name)=MCD archive member AND roi_name=native Description; no filename-only or row-number alias"
    mapping_rows.append(field_rule("image_metadata", field,
        relative(evidence / "raw_file_metadata.csv") if field == "sample_id" else relative(metadata / "native_verification.json"),
        "file_name + roi_name + sample_name" if field == "sample_id" else "native_result.verified_native_acquisitions",
        rule, join="source_member_name + description" if field == "sample_id" else "native image_id",
        notes="5/8 images connect to 5 of 6 explicit source samples; patient coverage 0/6" if field == "sample_id" else ""))
for field in columns("sample_metadata"):
    filled = field in {"study", "sample_id", "source_sample_id", "sample_type", "tissue_type", "anatomical_site",
                       "collection_method", "preservation_method", "primary_diagnosis", "metadata_notes"}
    direct = field in {"sample_id", "source_sample_id"}
    mapping_rows.append(field_rule("sample_metadata", field,
        relative(evidence / ("raw_file_metadata.csv" if direct else "paper.html")),
        "sample_name" if direct else "Methods: Sample description; Staining and IMC ablation",
        "Copy explicit source specimen ID" if direct else (
            "Study-wide normal colon mucosa; resection, FFPE and colorectal cancer donor context" if filled else
            "Empty: individual value or identity not established for archive sample; CLN clinical rows lack proven alias"),
        join="sample_name" if direct else "study scope", notes="patient identity unresolved" if field in {"patient_key", "source_subject_id"} else ""))
mapping_rows.append(field_rule("panel_mapping", "source_name", relative(metadata / "native_markers.json"),
    "ordered_source_labels", "Preserve all 62 native labels; S1 exact study+label join retains 30 channels",
    join="Study + Original marker", notes="Native CD14 and CCR4 are S1-admitted although absent from the 28-channel analysis CSV"))
mapping_rows.append(field_rule("sample_metadata", "metadata_notes", relative(trace_path), "sources + gaps",
    "Active sample/patient tracing; exact keys, source coverage and unjoined clinical source are recorded here",
    notes="6 source sample rows; 5/8 image connections; 0/6 patient connections; CRC_4 absent member; 3 unlinked native acquisitions"))
mapping = pd.DataFrame(mapping_rows, columns=columns("field_mapping"))
image_rows = {}
if (output / "images.parquet").exists():
    existing = pd.read_parquet(output / "images.parquet").astype(object)
    existing = existing.where(existing.notna(), None)
    image_rows = {row["image_id"]: row for row in existing.to_dict("records")}
excluded = []
estimate = sum(max(item["data_boundary"]["width"], item["data_boundary"]["height"])**2 * len(kept) * 4
               for item in selected)
save_json(cache / "source_inventory.json", dict(archive=relative(archive), source_stat=source_stat,
          members=members, expected_native_images=8, maximum_uncompressed_tiff_bytes=estimate,
          maximum_local_mcd_bytes=max(item["expected_member_bytes"] for item in selected),
          native_verification=relative(metadata / "native_verification.json"),
          accepted_scope=relative(metadata / "primary_scope_review.json"),
          source_defects=native["source_defects"], no_full_hash_repeated=True))
save_tables()
progress("metadata_ready", estimated_uncompressed_output_bytes=estimate)

# Extract one complete MCD at a time and export each acquisition in source order.
if not args.metadata_only:
    if shutil.disk_usage(root).free - estimate < 100*1024**3 or shutil.disk_usage(temporary).free - max(
            item["expected_member_bytes"] for item in selected) < 64*1024**3:
        raise RuntimeError("Insufficient reserved capacity for this study")
    for member in dict.fromkeys(item["source_member_name"] for item in selected):
        records = [item for item in selected if item["source_member_name"] == member]
        pending = []
        for record in records:
            completed = checked_receipt(record)
            if completed:
                image_rows[image_key(record)] = completed["image_row"]
                image_rows[image_key(record)]["sample_id"] = source_sample(record)
            else:
                destination = output / "images" / image_key(record) / "input.tiff"
                if not args.verify_existing_only or destination.exists():
                    pending.append(record)
        if not pending:
            save_tables()
            continue
        progress("extracting", member=member)
        local = temporary / Path(member).name
        expected_size = records[0]["expected_member_bytes"]
        if not (local.exists() and local.stat().st_size == expected_size):
            if args.verify_existing_only:
                raise ValueError("Existing TIFF recovery needs the retained local MCD; run the normal entry via write slot")
            partial = local.with_suffix(".mcd.extracting")
            with zipfile.ZipFile(archive) as source_zip, source_zip.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8*1024*1024)
            assert partial.stat().st_size == expected_size
            partial.replace(local)
        for record in pending:
            key = image_key(record)
            progress("exporting", current_image=key, member=member)
            source = SourceSpec(key, panel_id, "mcd", local, tuple(range(len(panel))),
                pixel_size_um=(float(record["native_metadata"]["AblationDistanceBetweenShotsX"]),
                               float(record["native_metadata"]["AblationDistanceBetweenShotsY"])),
                acquisition_id=record["source_acquisition_id"], source_revision="10.5281/zenodo.5545882",
                intensity_description="MCD Dual float32 native counts; no intensity normalization")
            destination = output / "images" / key / "input.tiff"
            try:
                with open_source(source) as reader:
                    actual = list(reader._acquisition.channel_labels)
                    assert actual == source_names, (key, actual)
                    h, w = reader.shape[-2:]
                    assert (w, h) == (record["data_boundary"]["width"], record["data_boundary"]["height"])
                    geometry = (recover_geometry(reader, destination) if destination.exists()
                                else export_input(reader, panel, destination))
                    if not geometry["admitted"]:
                        excluded.append(dict(image_id=key, reason="All retained downstream markers are zero over the entire ROI"))
                        save_json(cache / "excluded_images.json", excluded)
                        continue
                    pixel_checks = verify_pixels(reader, destination, geometry)
                    overview(reader, key)
                row = dict.fromkeys(image_fields)
                row.update(study=STUDY, image_id=key, sample_id=source_sample(record), panel_id=panel_id,
                    image_kind="roi", source_image_id=record["image_id"], source_files=[relative(archive)],
                    source_format="mcd", source_acquisition_id=str(record["source_acquisition_id"]),
                    source_axes="CYX", source_dtype="float32", source_revision="10.5281/zenodo.5545882",
                    slide_id=f"{Path(member).stem}__slide{record['slide_id']}", roi_id=record["acquisition_roi_id"],
                    region_annotation=record["description"], source_x0_px=0, source_y0_px=0,
                    source_x1_px=w, source_y1_px=h, height_px=geometry["height_px"], width_px=geometry["width_px"],
                    source_channel_count=len(panel), channel_count=len(kept), pixel_size_x_um=source.pixel_size_um[0],
                    pixel_size_y_um=source.pixel_size_um[1], input_image_path=relative(destination),
                    valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if geometry["valid_pixels"] else None,
                    intensity_description=source.intensity_description, split_method="source_acquisition",
                    split_evidence=relative(metadata / "native_verification.json"),
                    metadata_notes="Complete native ROI; channel-major CYX view of MCD pixel records. "
                        "Sample joined by exact native Description and source MCD basename where available. "
                        "No patient or CRC-to-CLN identity inferred.")
                for field in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"):
                    row[field] = geometry[field]
                stat = destination.stat()
                save_json(receipts / f"{key}.json", dict(input_binding=binding(record), image_row=row,
                    output_stat=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns), geometry=geometry,
                    pixel_checks=pixel_checks, qc=relative(cache / "qc" / f"{key}.png")))
                image_rows[key] = row
                save_tables()
                progress("image_completed", current_image=key)
                print(json.dumps(dict(image_id=key, completed_images=len(image_rows), sample_id=row["sample_id"])), flush=True)
            except Exception as error:
                progress("failed", current_image=key, error=f"{type(error).__name__}: {error}")
                raise
        if not args.verify_existing_only:
            local.unlink()
    save_tables()
    progress("existing_images_recovered" if args.verify_existing_only else "prepared",
             sample_rows=len(samples), sample_linked_images=sum(bool(row["sample_id"]) for row in image_rows.values()),
             patient_linked_samples=0, excluded_images=excluded)
