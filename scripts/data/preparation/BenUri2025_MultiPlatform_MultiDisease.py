"""Prepare the locally eligible CombPlex large-scale CODEX acquisitions.

Inputs: S-BIAD873 Raw single-plane TIFFs, existing native verification, and S1.
Outputs: published uint16 CYX input.tiff, the five study input tables, and resumable
preparer evidence. No intensity normalization or independent acceptance is run.
Multi-file channels have no stack order: source indices explicitly follow sorted
source filenames, and each index is bound to that file, never to inventory order.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
from typing import Iterator

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "4"

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.contracts import ImageBlock
from cellweft.data.io.readers import read_tiff_window, windows
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.source_metadata import (
    metadata_source_directory, metadata_source_relative, read_metadata_member,
)

STUDY = "BenUri2025_MultiPlatform_MultiDisease"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RELEASE = NAS / "raw/public" / STUDY / "S-BIAD873"
EXPERIMENT = "CODEX_large_scale_compresseion_experiment"
RAW = RELEASE / "CombPlex_data" / EXPERIMENT / "Raw"
SOURCES = metadata_source_directory(NAS, STUDY) / "preparation_sources"
CACHE = Path(__file__).resolve().parents[3] / "cache/study_preparation" / STUDY
OUTPUT = NAS / "processed" / STUDY
PROGRESS = CACHE / "progress.json"
ELIGIBLE_BUNDLE = NAS / "raw/public" / STUDY / "metadata.json"
ELIGIBLE_MEMBER = f"public/{STUDY}/measured_channels_by_fov.json"
ELIGIBLE_REFERENCE = f"{ELIGIBLE_BUNDLE.relative_to(NAS)}#{ELIGIBLE_MEMBER}"
NATIVE_MEMBER = f"public/{STUDY}/native_channel_verification.json"
NATIVE_REFERENCE = f"{ELIGIBLE_BUNDLE.relative_to(NAS)}#{NATIVE_MEMBER}"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "CODEX_large_scale_28planes"
PREPARER = "0015_prepare_0"


def read_eligible_channels(bundle: Path, member: str) -> dict[str, list[dict]]:
    """Read the consolidated inventory; local raw FOVs still define the scope."""
    with formal_read_lease(bundle):
        return json.loads(read_metadata_member(bundle, member))["fovs"]


def read_native_proofs(bundle: Path, member: str) -> dict[str, dict]:
    """Read the accepted native proofs from their permanent source bundle."""
    with formal_read_lease(bundle):
        return json.loads(read_metadata_member(bundle, member))


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return path.as_posix()
    return metadata_source_relative(path, NAS).as_posix()


class SinglePlaneReader:
    """Decode one 2962-square channel once, bounding memory to one native plane.

    Shared TIFF window reads repeatedly decode the same full-width strips for
    adjacent output tiles. This study-local adaptation caches only the current
    channel (17 MiB); migration to a shared bounded strip/plane cache is pending.
    """

    def __init__(self, paths: list[Path], shape: tuple[int, int], dtype: str) -> None:
        self.paths = paths
        self.shape = (len(paths), *shape)
        self.dtype = np.dtype(dtype)
        self.channel = None
        self.plane = None

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        values = []
        for channel in channels:
            if channel != self.channel:
                self.plane = None
                with tifffile.TiffFile(self.paths[channel]) as tif:
                    self.plane = tif.pages[0].asarray(maxworkers=2)
                self.channel = channel
            values.append(self.plane[y0:y1, x0:x1])
        return np.stack(values)

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def iter_blocks(self, channels: list[int], block_size: int) -> Iterator[ImageBlock]:
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


def update_progress(stage: str, count: int, **details) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage, expected_images=len(fovs),
                             completed_images=count, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             metadata_gaps=metadata_gaps, **details))


def mapping(source: str, field: str, table: str, target: str,
            rule: str, notes: str = "", join: str = "") -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=field, target_table=table, target_field=target,
                join_keys=join, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="Preserve source identity; unknown remains empty",
                notes=notes)


def save_tables() -> None:
    frame = pd.DataFrame(rows, columns=image_columns)
    # Merge previous completed batches. The saved row is published after its TIFF
    # and per-image receipt close, so metadata-only reruns reuse completed pixels.
    save_input_tables(OUTPUT, frame, samples, panel, field_mapping)


def check_saved(path: Path, channels: list[str], shape: tuple[int, int], dtype: str) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (len(channels), *shape)
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].dtype == np.dtype(dtype)
        assert len(tif.pages) == len(channels)
        names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        assert names == channels


def compare_pixels(source_paths: list[Path], output: Path, kept: pd.DataFrame) -> list[dict]:
    """Compare internal and edge blocks for every retained channel, directly."""
    blocks = [(1024, 1024, 1056, 1056), (2930, 2930, 2962, 2962)]
    with tifffile.TiffFile(output) as target:
        for row in kept.itertuples(index=False):
            with tifffile.TiffFile(source_paths[row.source_channel_index]) as source:
                for block in blocks:
                    a = read_tiff_window(source, 0, block)
                    b = read_tiff_window(target, int(row.output_channel_index), block)
                    assert a.dtype == b.dtype and np.array_equal(a, b)
    return [dict(xyxy=list(block), channels="all retained", equal=True) for block in blocks]


def overview(path: Path, image_id: str, kept: pd.DataFrame) -> None:
    """Save source-intensity views with display-only percentile scaling."""
    labels = ["DAPI", "Pan-Keratin", "CD8", "SMA"]
    montage = Image.new("RGB", (512*len(labels), 544), "white")
    for index, label in enumerate(labels):
        channel = int(kept.loc[kept.source_name.eq(label), "output_channel_index"].iloc[0])
        with tifffile.TiffFile(path / (label+".tif") if path.is_dir() else path) as tif:
            data = tif.pages[0 if path.is_dir() else channel].asarray(maxworkers=2)[::6, ::6]
        low, high = np.percentile(data, [1, 99.5])
        scaled = np.clip((data.astype(np.float32)-low)/max(high-low, 1), 0, 1)
        view = Image.fromarray((scaled*255).astype(np.uint8)).resize((512, 512))
        montage.paste(view.convert("RGB"), (index*512, 32))
        ImageDraw.Draw(montage).text((index*512+5, 7), f"{image_id}: {label}", fill="black")
    montage.save(CACHE / f"overview_{image_id}.png")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--limit", type=int, help="Bound newly exported images; existing results are merged")
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--source-overviews", action="store_true", help="Inspect three representative native FOVs under the read I/O slot")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
preparer_lease = (OUTPUT / ".preparer.lock").open("a")
fcntl.flock(preparer_lease, fcntl.LOCK_EX)

# Resolve the actual local scope against the accepted source list and native proofs.
native = read_native_proofs(ELIGIBLE_BUNDLE, NATIVE_MEMBER)
eligible = read_eligible_channels(ELIGIBLE_BUNDLE, ELIGIBLE_MEMBER)
fovs = sorted(path for path in RAW.iterdir() if path.is_dir())
image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
names = sorted(path.stem for path in (RAW / "A-1").glob("*.tif"))
panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
panel["source_filename"] = [name + ".tif" for name in panel.source_name]
scope = []
for fov in fovs:
    source_key = f"{EXPERIMENT}/{fov.name}"
    inventory = {row["filename"]: row for row in eligible[source_key]}
    assert sorted(inventory) == [name+".tif" for name in names]
    assert sorted(path.name for path in fov.iterdir()) == sorted(inventory)
    for filename, source in inventory.items():
        path = fov / filename
        proof = native[source["source_path"]]
        assert path.stat().st_size == source["size"] == proof["source_size"]
        assert proof["complete"] and proof["shape"] == [2962, 2962]
        assert proof["dtype"] == "uint16" and proof["axes"] == "YX"
    scope.append(dict(image_id=f"CODEX22_{fov.name}", source_image_id=source_key,
                      source_files=[relative(fov / (name+".tif")) for name in names],
                      complete=True, verification=NATIVE_REFERENCE))
assert len(fovs) == 143 and len(names) == 28 and len(kept) == 22
write_json(CACHE / "local_scope.json", dict(study=STUDY, images=scope,
           source_file_count=4004, source_bytes=sum((NAS/p).stat().st_size for s in scope for p in s["source_files"]),
           excluded_scope="Other CODEX/MIBI branches were removed by existing eligibility selection; none are present locally",
           source_order="Sorted exact plane filenames; source_channel_index binds to source_files in this order",
           nominal_output_bytes=143*22*2962*2962*2,
           native_proofs_reused=True, hashes_recomputed=False))
if args.source_overviews:
    for fov_name in ("A-1", "E-8", "I-16"):
        overview(RAW / fov_name, f"CODEX22_{fov_name}", kept)
    print("Three native-source overviews saved", flush=True)
    raise SystemExit(0)

# Sample/core-to-case identity is filled only from a demonstrated source map.
# BR1503f is the paper's commercial TMA, but a product version or an assumed
# adjacency of cores is not enough to infer sample or patient IDs.
samples = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str)
metadata_gaps = ["Missing release-specific FOV/core to biological specimen map; sample and patient identities remain unresolved",
                 "Scalar TIFF has only 72-dpi print resolution; author CODEXSaveTifs casts uint16 and resizes to 2962-square; physical sampling of distributed pixels is unverified"]
trace_path = SOURCES / "sample_source_trace.json"
field_mapping = pd.DataFrame([
    mapping(ELIGIBLE_REFERENCE, "fovs/source_path/filename", "image_metadata", "source_files",
            "Bind every raw single-plane TIFF to its exact FOV; sort filenames for the explicit multi-file source axis",
            "28 source files; no intrinsic scalar TIFF channel axis", "experiment/FOV/filename"),
    mapping(ELIGIBLE_REFERENCE, "experiment/FOV", "image_metadata", "source_image_id",
            "Keep published experiment/FOV identifier; prefix image_id with CODEX22", join="experiment/FOV"),
    mapping(relative(RELEASE / "ReadMe.rtf"), "Raw/FOV_name", "image_metadata", "split_method",
            "already_independent; preserve complete 2962-square acquisition frame",
            "No geometric crop or sample boundary inferred"),
    mapping(relative(S1), "Study + Original marker", "panel_mapping", "marker_id",
            "Exact study and source filename stem join; keep DAPI and 21 admitted protein measurements",
            "Five measured mixture planes and gH2AX are excluded by this study's S1"),
    mapping(relative(trace_path), "FOV/core -> biological specimen -> patient", "image_metadata", "sample_id",
            "No inferred ROI=sample conversion; leave sample_id empty until release-specific map is confirmed",
            "M01: sample rows=0; image->sample=0/143; sample->patient=0/0. See cached primary sources and explicit missing key.",
            "Required missing key: released FOV name to manufacturer core to source specimen/case"),
    mapping(relative(SOURCES / "supplementary_table1.xlsx"), "Key_Resources_Table/Biological Samples", "sample_metadata", "metadata_notes",
            "Paper identifies TissueArray BR1503f, a breast carcinoma TMA; cannot assign core-level clinical values without its matching map",
            "Different product replacement or batch is not used as a clinical substitute"),
    mapping(relative(RAW / "A-1/DAPI.tif"), "XResolution/ResolutionUnit", "image_metadata", "pixel_size_x_um",
            "Leave biological scale empty: 72 dpi is generic print-resolution metadata", "Published pixel grid is retained"),
    mapping(relative(SOURCES / "CODEXSaveTifs.m"), "uint16 / imresize to 2962x2962", "image_metadata", "intensity_description",
            "Preserve distributed Raw TIFF values; record the author's upstream conversion code without inferring the original microscope pixel size",
            "Source code is evidence of possible upstream casting/resizing; no instrument-level intensity or resolution reconstruction is attempted"),
    mapping(relative(SOURCES / "attachment_contents.json"), "settings/activeImageSet; empty selections and features", "image_metadata", "metadata_notes",
            "Six source Mantis databases were inspected; two Raw attachments identify BreastCaTMA_Stain8_Run3, without tissue specimen or case keys",
            "Viewer scales are display transforms, not physical pixel size or biological identity"),
], columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)
field_mapping.loc[field_mapping.source_file.eq(relative(SOURCES / "supplementary_table1.xlsx")),
                  "source_sheet_or_table"] = "Key_Resources_Table"
rows = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
receipts_path = CACHE / "export_receipts.json"
receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
save_tables()
update_progress("metadata_prepared" if args.metadata_only else "exporting", len(rows))

# Export each FOV under the controller's single NAS writer slot, then persist
# the real index immediately. Resume checks receipts and headers before reuse.
new_count = 0
for item in scope:
    if args.metadata_only:
        break
    image_id = item["image_id"]
    paths = [NAS / path for path in item["source_files"]]
    destination = OUTPUT / "images" / image_id / "input.tiff"
    if image_id in receipts:
        if receipts[image_id]["admitted"]:
            check_saved(destination, kept.marker_id.tolist(), (2962, 2962), "uint16")
            if not any(row["image_id"] == image_id for row in rows):
                rows.append(receipts[image_id]["image_record"])
                save_tables()
        continue
    if args.limit is not None and new_count >= args.limit:
        break
    update_progress("exporting", len(rows), current_image=image_id)
    if shutil.disk_usage(NAS).free < 100*1024**3 + 22*2962*2962*2:
        update_progress("capacity_blocked", len(rows), current_image=image_id)
        raise RuntimeError("Insufficient NAS capacity above 100 GiB reservation")
    reader = SinglePlaneReader(paths, (2962, 2962), "uint16")
    result = export_input(reader, panel, destination)
    reader.plane = None
    if result["admitted"]:
        check_saved(destination, kept.marker_id.tolist(), (2962, 2962), "uint16")
        record = dict.fromkeys(image_columns)
        record.update(study=STUDY, image_id=image_id, panel_id=PANEL_ID,
                      image_kind="fov", source_image_id=item["source_image_id"],
                      source_files=item["source_files"], source_format="tiff_planes",
                      source_axes="YX per source plane; CYX explicit filename order",
                      source_dtype="uint16", source_revision="S-BIAD873 release 2023-09-07; original measured Raw",
                      fov_id=image_id.removeprefix("CODEX22_"),
                      source_x0_px=0, source_y0_px=0, source_x1_px=2962, source_y1_px=2962,
                      pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                      height_px=2962, width_px=2962, source_channel_count=28, channel_count=22,
                      input_image_path=relative(destination),
                      intensity_description="Published Raw uint16 measured CODEX values, unchanged by CellWeft. Author CODEXSaveTifs uses uint16/imresize to 2962-square; original instrument scale unverified.",
                      split_method="already_independent", split_evidence=relative(RELEASE / "ReadMe.rtf"),
                      metadata_notes="Sample/patient identity unresolved; retain exact source FOV, without inferring patients from names. No source region annotation or verified physical pixel calibration supplied.")
        rows = [row for row in rows if row["image_id"] != image_id] + [record]
        result["image_record"] = record
        if image_id in {"CODEX22_A-1", "CODEX22_E-8", "CODEX22_I-16"}:
            result["pixel_blocks"] = compare_pixels(paths, destination, kept)
            overview(destination, image_id, kept)
    receipts[image_id] = result
    write_json(receipts_path, receipts)
    save_tables()
    new_count += 1
    update_progress("exporting", len(rows), last_completed_image=image_id,
                    excluded_zero_images=[key for key, value in receipts.items() if not value["admitted"]])
    print(json.dumps(dict(image_id=image_id, completed_images=len(rows), admitted=result["admitted"])), flush=True)

# Preparer findings remain distinct from independent reviewer acceptance.
review_columns = pd.read_csv(TEMPLATES / "review_results.csv").columns
review = pd.DataFrame([
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="unresolved", source_evidence=relative(CACHE / "local_scope.json"),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details=f"Preparer self-check: {len(rows)}/143 FOVs exported; all 4004 local source planes match existing native proofs and accepted size. Independent review pending."),
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample", record_id=STUDY,
         check_id="M01", result="unresolved", source_evidence=relative(trace_path),
         output_evidence=relative(OUTPUT / "sample_metadata.csv"),
         details="Preparer trace: 0 sample rows; 0/143 images linked; 0 patients. Release-specific core-to-specimen-to-patient map not confirmed; primary sources retained in cache."),
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
         check_id="D01", result="unresolved", source_evidence=relative(CACHE), output_evidence=relative(OUTPUT),
         details=f"Preparer delivery {len(rows)} TIFFs; independent review pending; M01 biological identity and source pixel scale remain explicit gaps. No independent pass or input_binding asserted."),
], columns=review_columns)
review_path = OUTPUT / "review_results.csv"
with formal_publish_lease(review_path, directory=OUTPUT):
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
update_progress("prepared_for_independent_review" if len(rows) == len(fovs) else "partial",
                len(rows), exported_images=len(rows), sample_rows=len(samples),
                sample_linked_images=0, patient_linked_samples=0,
                controller_requests=["Arrange independent review; M01 requires release-specific BR1503f FOV/core-to-specimen map",
                                     "Consider migrating bounded single-plane/strip caching to the shared TIFF reader; study-local adaptation retained here"])
print(json.dumps(dict(study=STUDY, exported_images=len(rows), expected_images=len(fovs),
                      sample_rows=len(samples), sample_linked_images=0, patient_linked_samples=0)), flush=True)
