"""Prepare 11 distinct lymph-node MIBI FOVs from 12 published Point directories.

Inputs: Zenodo 8096953 v1 single-channel TIFFs, original native verification,
source channel_mapping.csv, and this study's rows in the unique S1 table.
Outputs: resumable input.tiff files, complete input tables, and preparer checks.
The shared helpers stage locally and acquire short per-image publication leases.
Sample identities require an explicit source map; Point numbers identify FOVs.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import ExitStack
import filecmp
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)

STUDY = "Liu2023_MIBI_LymphNode"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_8096953"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB",
                          str(PROJECT / "cache/study_preparation" / STUDY / "jobs/prepare")))
PROGRESS = JOB / "progress.json"
PROOF = CACHE / "native_verification.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "native_MIBI_panel"
PREPARER = f"Codex CLI {STUDY} {JOB.name} preparer"
REVISION = "Zenodo 8096953 v1 (2023-06-29); original extracted TIFF members"
PIXEL_UM = 500 / 1024
PAPER_URL = "https://www.nature.com/articles/s41467-023-40068-5"
GAP = "No explicit Point/FOV-to-biospecimen or patient map found in the inspected sources; six individuals reported only at cohort level."
DUPLICATE = ("Point4/Point5 have byte-identical intensity TIFFs across all 45 channels. "
             "Point4 is the provisional first-label representative of this pixel set; "
             "their distinct segmentation masks leave the true Point identity unresolved.")
# The author's 1_imageToMatrix.py defines CSV `sample` as the Point index.
# Supplementary Information contains figures and antibody Table 1, not a
# specimen/clinical map. The `cimac_points.csv` reference in the author's
# 12_pixel_cluster_correlations.R belongs to another serial-section TMA cohort.
# Full release member listings and repository/source snapshots are in CACHE;
# sample_linkage_evidence.json records the exact inspected sources and gaps.
# Both published source cell tables were also extracted with bounded HTTP ZIP
# ranges: they have `fov`/cell/expression fields, without specimen/patient keys.
os.environ["MPLCONFIGDIR"] = str(JOB / "matplotlib")


def relative(path: Path) -> str:
    return metadata_source_relative(path, NAS).as_posix()


def write_json(path: Path, value: object) -> None:
    """Keep each per-image receipt and progress file resumable."""
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with formal_publish_lease(path, directory=path.parent):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(payload)
        temporary.replace(path)


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage, source_fovs=12, expected_images=12,
        distinct_local_intensity_images=11,
        completed_images=completed, sample_rows=0, sample_linked_images=0,
        patient_linked_samples=0, phase="repair" if args.stage == "repair" else "prepare",
        scientific_repair_round=1 if args.stage == "repair" else 0,
        gaps=[GAP], updated_unix=time.time(), **details))


def source_inventory() -> tuple[pd.DataFrame, dict]:
    """Reconcile original published file identities, archived members and old proof.

    These are independent single-channel files, without an encoded C axis.
    source_channel_index is the explicit file order in channel_mapping.csv,
    itself the lexicographic filename order used by the original verifier.
    It does not assert a mass-spectrometer acquisition order absent from TIFFs.
    """
    mapping = pd.read_csv(META / "channel_mapping.csv", dtype=str, keep_default_na=False)
    proof = json.loads(PROOF.read_text())
    prior_path = CACHE / "source_inventory.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    proven_headers = {file["path"]: file
                      for row in prior.get("acquisitions", []) for file in row["source_files"]}
    assert proof["all_verified"] and proof["n_fovs"] == 12
    proven = {row["fov"]: row for row in proof["fovs"]}
    with zipfile.ZipFile(RAW / "mibi_single_channel_tifs.zip") as archive:
        members = {row.filename: row for row in archive.infolist() if not row.is_dir()}
    assert len(members) == 540
    assert set(members) == set(mapping.source_relative_path)
    actual = {relative(p) for p in (RAW / "mibi_single_channel_tifs").glob("*/*.tif")}
    assert actual == {relative(RAW / name) for name in members}
    records = []
    for fov, group in mapping.groupby("fov", sort=False):
        names = group.marker_or_acquisition_channel.tolist()
        assert names == proven[fov]["markers"] and len(names) == 45
        assert proven[fov]["shapes"] == [[1024, 1024]] and proven[fov]["dtypes"] == ["uint16"]
        files = []
        for row in group.itertuples(index=False):
            path = RAW / row.source_relative_path
            assert path.name == row.channel_file and path.stem == row.marker_or_acquisition_channel
            stat = path.stat()
            assert stat.st_size == members[row.source_relative_path].file_size
            identity = dict(path=relative(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if proven_headers.get(relative(path)) != identity:
                with formal_read_lease(path), tifffile.TiffFile(path) as tif:
                    assert len(tif.pages) == 1 and tif.pages[0].shape == (1024, 1024)
                    assert tif.pages[0].dtype == np.dtype("uint16")
            files.append(identity)
        records.append(dict(fov=fov, source_files=files))
    assert {row["fov"] for row in records} == set(proven)
    inventory = dict(study=STUDY, source_archive=relative(RAW / "mibi_single_channel_tifs.zip"),
        archive_members=540, source_fovs=12, qualified_fovs=11, source_channel_count=45,
        source_order="Explicit channel_mapping.csv file order; scalar YX TIFF per channel",
        reused_native_verification=relative(PROOF), raw_source_modified=False,
        excluded_fovs=[dict(fov="Point5", reason=DUPLICATE)], partial_files_used=[], acquisitions=records,
        estimated_uncompressed_output_bytes=11 * 34 * 1024 * 1024 * 2,
        free_bytes={"NAS": shutil.disk_usage(NAS).free, "tmp": shutil.disk_usage("/tmp").free})
    write_json(CACHE / "source_inventory.json", inventory)
    return mapping, inventory


def field_map() -> pd.DataFrame:
    """Explain each source interpretation, including the unresolved identity join."""
    columns = pd.read_csv(TEMPLATES / "field_mapping.csv").columns
    rows = []

    def add(source: str, field: str, target: str, rule: str, notes: str = "",
            table: str = "image_metadata", keys: str = "fov=Point directory") -> None:
        rows.append(dict(study=STUDY, source_file=source, source_sheet_or_table="",
            source_field=field, target_table=table, target_field=target,
            join_keys=keys, value_mapping="{}", transformation_rule=rule,
            time_origin_rule="", conflict_rule="Preserve source identity; leave unconfirmed values blank", notes=notes))

    add(relative(META / "channel_mapping.csv"), "fov; source_relative_path", "image_id; source_image_id; fov_id; source_files",
        "Use the published Point directory ID for each retained FOV; list all 45 source paths in mapping row order.",
        "540 archive members and 12 Point directories yield 11 distinct observations after excluding the Point5 duplicate.")
    add(relative(CACHE / "duplicate_point4_point5.json"), "Point4 and Point5 source TIFF bytes", "image_id; split_evidence; metadata_notes",
        "Keep Point4 provisionally as the first published image label; exclude the redundant Point5 pixels from images.parquet and retain its prior export in study cache.",
        DUPLICATE + " The original archive and extracted source files are unchanged.", keys="Point4=Point5 across all 45 channel filenames")
    add(relative(RAW / "segmentation/Point4_feature_0.tif"), "Point4/Point5 published instance masks", "split_evidence; metadata_notes",
        "Record the distinct instance masks as a source identity conflict; do not assign either mask to the retained input until reviewed.",
        "The two 1024x1024 source masks differ despite identical published image intensities; true FOV identity remains unresolved.",
        keys="Point4_feature_0.tif; Point5_feature_0.tif")
    add(relative(CACHE / "source_inventory.json"), "scalar YX TIFF shape and dtype", "source_axes; source_dtype; source_channel_count; source_x0_px; source_y0_px; source_x1_px; source_y1_px",
        "Each source file is YX uint16, 1024x1024. Stack explicitly ordered files as C; preserve the complete [0,1024)x[0,1024) FOV.")
    add("https://zenodo.org/records/8096953", "version; description", "source_revision; image_kind; split_method; split_evidence; intensity_description",
        "Zenodo 8096953 v1 single-channel image release; already-independent FOVs. Preserve decoded published uint16 intensities.",
        "No additional normalization, segmentation, background subtraction or intensity transform applied. Upstream preprocessing is not fully specified for these released files.")
    add(PAPER_URL, "Methods: Lymph node MIBI-TOF dataset", "pixel_size_x_um; pixel_size_y_um",
        "500 micrometers / 1024 pixels = 0.48828125 micrometers/pixel in both axes.",
        "Confirmed also by the Zenodo image description; TIFF 72-dpi display resolution is not a tissue pixel scale.", keys="study-level field size; all 12 Point FOVs")
    add(relative(S1), "Study; Original marker; Unified marker; Segmentation channel", "source_name; marker_id; role; nuclear; body; keep_channel; source_channel_index; output_channel_index",
        "Exact Study+Original marker join, no cross-study aliases. Keep S1 admitted measurements and selected segmentation inputs; output positions are contiguous among kept rows.",
        "45 source rows; 34 retained, 11 excluded. HH3 -> Histone H3 is retained as the selected nuclear input.", table="panel_mapping", keys="Study=Liu2023_MIBI_LymphNode; Original marker=TIFF basename")
    add(relative(CACHE / "sample_linkage_evidence.json"), "local metadata; release attachments; source cell tables; paper; supplementary tables; author repository", "sample_id",
        "No verified join key. Keep image.sample_id empty until an explicit FOV-to-biospecimen map is available.",
        "11 distinct images; 0 confirmed sample rows; 0/11 image-to-sample links. Point IDs are imaging regions. " + GAP, keys="No source-supported Point→biospecimen key found")
    add(relative(CACHE / "sample_linkage_evidence.json"), "Methods cohort size and inspected identity sources", "patient_key; source_subject_id; source_sample_id",
        "Cohort-level six individuals does not identify six specimens or assign FOVs. No patient or specimen rows are synthesized.",
        "0 sample-to-patient links. Demographics and diagnosis cannot be assigned at FOV level.", table="sample_metadata", keys="No source-supported biospecimen→subject key found")
    add(relative(CACHE / "sample_linkage_evidence.json"), "source search outcome", "metadata_notes",
        "Record the concrete missing FOV-to-biospecimen/patient correspondence and scope of the inspected evidence.", GAP)
    return pd.DataFrame(rows, columns=columns)


def review_row(check: str, result: str, details: object, scope: str = "study", record: str = STUDY) -> dict:
    return dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
        record_id=record, check_id=check, result=result,
        source_evidence=json.dumps([relative(PROOF), relative(CACHE / "source_inventory.json"),
            relative(CACHE / "sample_linkage_evidence.json"), relative(S1)]),
        output_evidence=relative(OUTPUT),
        details=json.dumps(dict(preparer_self_check=True, independent_review_pending=True,
            evidence=details), ensure_ascii=False))


def save_reviews(rows: list[dict]) -> None:
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        frame = pd.DataFrame(rows, columns=pd.read_csv(TEMPLATES / "review_results.csv").columns)
        temporary = OUTPUT / "review_results.csv.partial"
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(OUTPUT / "review_results.csv")


def verify(images: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Check all TIFF headers and two independent pixel blocks per sampled FOV."""
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    checks = []
    for row in images.itertuples(index=False):
        with formal_read_lease(NAS / row.input_image_path), tifffile.TiffFile(NAS / row.input_image_path) as tif:
            assert tif.series[0].shape == (34, 1024, 1024) and tif.series[0].axes == "CYX"
            assert tif.series[0].dtype == np.dtype("uint16")
            pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
            assert [x.attrib["Name"] for x in pixels.findall("{*}Channel")] == kept.marker_id.tolist()
            assert float(pixels.attrib["PhysicalSizeX"]) == PIXEL_UM
            assert float(pixels.attrib["PhysicalSizeY"]) == PIXEL_UM
        checks.append(dict(image_id=row.image_id, header_pass=True))
    for image_id in ("Point1", "Point12"):
        row = images.set_index("image_id").loc[image_id]
        paths = list(row.source_files)
        for window in ((480, 480, 512, 512), (992, 992, 1024, 1024)):
            # Keep source and output reads sequential; never nest shared slots.
            x0, y0, x1, y1 = window
            expected = []
            with formal_read_lease(NAS / paths[0]):
                for channel in kept.itertuples(index=False):
                    values = tifffile.imread(NAS / paths[channel.source_channel_index], maxworkers=2)
                    expected.append(values[y0:y1, x0:x1].copy())
            with formal_read_lease(NAS / row.input_image_path), tifffile.TiffFile(NAS / row.input_image_path) as tif:
                for channel, values in zip(kept.itertuples(index=False), expected, strict=True):
                    saved = tif.pages[int(channel.output_channel_index)].asarray(maxworkers=2)[y0:y1, x0:x1]
                    assert values.dtype == saved.dtype
                    np.testing.assert_array_equal(values, saved)
            checks.append(dict(image_id=image_id, window_xyxy=list(window),
                               output_channels=list(range(34)), pixel_equality=True))
    write_json(CACHE / "preparer_verification.json", dict(study=STUDY,
        all_image_headers_checked=len(images), sampled_image_ids=["Point1", "Point12"],
        checks=checks, padding="No padding: all complete source FOVs are square 1024x1024",
        independent_review=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    channel_positions = [int(kept.set_index("source_name").loc[name, "output_channel_index"])
                         for name in ("CD20", "CD3", "HH3")]
    for axis, row in zip(axes.flat, images.sort_values("image_id", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)).itertuples(index=False)):
        with formal_read_lease(NAS / row.input_image_path), tifffile.TiffFile(NAS / row.input_image_path) as tif:
            planes = [tif.pages[i].asarray(maxworkers=2)[::4, ::4] for i in channel_positions]
        rgb = np.stack([np.clip(p / max(float(np.percentile(p[p > 0], 99.5)), 1), 0, 1)
                        for p in planes], axis=-1)
        axis.imshow(rgb); axis.set_title(row.image_id); axis.axis("off")
    axes.flat[-1].axis("off")
    fig.suptitle("Native input overview: CD20 red / CD3 green / Histone H3 blue\nDisplay scaling only; TIFF intensities unchanged")
    fig.tight_layout()
    fig.savefig(CACHE / "input_overview.png", dpi=140)
    plt.close(fig)


def repair_duplicate(images: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Remove one repeated published observation while preserving its export."""
    left = RAW / "mibi_single_channel_tifs/Point4"
    right = RAW / "mibi_single_channel_tifs/Point5"
    channels = sorted(path.name for path in left.glob("*.tif"))
    assert len(channels) == 45
    with formal_read_lease(left):
        assert channels == sorted(path.name for path in right.glob("*.tif"))
        assert all(filecmp.cmp(left / name, right / name, shallow=False) for name in channels)
    mask4 = RAW / "segmentation/Point4_feature_0.tif"
    mask5 = RAW / "segmentation/Point5_feature_0.tif"
    with formal_read_lease(mask4):
        first_mask = tifffile.imread(mask4, maxworkers=2)
        second_mask = tifffile.imread(mask5, maxworkers=2)
    assert first_mask.shape == second_mask.shape == (1, 1024, 1024)
    differing_mask_pixels = int(np.count_nonzero(first_mask != second_mask))
    assert differing_mask_pixels > 0
    assert {"Point4", "Point5"}.issubset(set(images.image_id)) or (
        "Point4" in set(images.image_id) and "Point5" not in set(images.image_id))
    assert len(images) in (11, 12)
    progress("duplicate_confirmed", 10, duplicate_pair=["Point4", "Point5"],
             identical_source_channels=45, resume_at="field_mapping_and_image_index")
    evidence = dict(study=STUDY, source_fovs=12, distinct_intensity_pixel_sets=11,
                    identical_source_channels=channels, comparison="full file bytes",
                    provisional_retained_image_id="Point4", excluded_intensity_fov="Point5",
                    source_instance_masks=dict(paths=[relative(mask4), relative(mask5)],
                                               shape=list(first_mask.shape),
                                               differing_pixels=differing_mask_pixels,
                                               identity_assignment="unresolved"),
                    reason=DUPLICATE, original_source_modified=False,
                    prior_independent_review="Earlier independent report; original review session has closed.")
    write_json(CACHE / "duplicate_point4_point5.json", evidence)

    for evidence_path in (CACHE / "sample_linkage_evidence.json",
                          META / "input_source_evidence.json"):
        with formal_read_lease(evidence_path):
            record = json.loads(evidence_path.read_text())
        if evidence_path.name == "sample_linkage_evidence.json":
            record["local_qualified_images"] = 11
            record["published_point_directories"] = 12
            record["excluded_duplicate_fov"] = "Point5"
            record["recommended_next_step"] = (
                "Independent repair review of Point4/Point5 and refreshed D01 binding; "
                "M01 remains unresolved until a source-supported biological map is localized. "
                "No author contact performed.")
        else:
            record["scope"]["qualified_local_fovs"] = 11
            record["scope"]["published_point_directories"] = 12
            record["scope"]["excluded_duplicate_fov"] = "Point5"
            record["duplicate_evidence"] = relative(CACHE / "duplicate_point4_point5.json")
        write_json(evidence_path, record)

    with formal_read_lease(OUTPUT):
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    assert samples.empty
    save_input_tables(OUTPUT, images, samples, panel, field_map())

    while True:
        with formal_read_lease(OUTPUT):
            with resource_lock(OUTPUT, shared=True):
                current = OUTPUT / "images.parquet"
                version = (current.stat().st_size, current.stat().st_mtime_ns)
                frame = pd.read_parquet(current)
        assert len(frame) in (11, 12) and "Point4" in set(frame.image_id)
        retained = frame.loc[frame.image_id != "Point5"].copy()
        point4 = retained.image_id == "Point4"
        if len(frame) == 11 and retained.loc[point4, "split_evidence"].iloc[0] == DUPLICATE:
            break
        retained.loc[point4, "split_evidence"] = DUPLICATE
        retained.loc[point4, "metadata_notes"] = GAP + " " + DUPLICATE
        with tempfile.TemporaryDirectory(prefix="liu-input-repair-", dir="/tmp") as staging:
            staged = Path(staging) / "images.parquet"
            arrow = pa.Table.from_pandas(retained, preserve_index=False)
            index = arrow.schema.get_field_index("source_files")
            arrow = arrow.set_column(index, "source_files",
                                     pa.array([list(paths) for paths in retained.source_files],
                                              type=pa.list_(pa.string())))
            pq.write_table(arrow, staged, compression="zstd")
            with formal_publish_lease(current, directory=OUTPUT):
                if (current.stat().st_size, current.stat().st_mtime_ns) != version:
                    continue
                temporary = current.with_name(current.name + ".partial")
                shutil.copyfile(staged, temporary)
                os.replace(temporary, current)
        break

    prior_export = OUTPUT / "images/Point5/input.tiff"
    archived_export = CACHE / "excluded/Point5/input.tiff"
    with formal_publish_lease(prior_export, directory=OUTPUT):
        if prior_export.exists():
            assert not archived_export.exists()
            archived_export.parent.mkdir(parents=True, exist_ok=True)
            os.replace(prior_export, archived_export)
        else:
            assert archived_export.exists()
    remaining_ids = sorted(frame.image_id[frame.image_id != "Point5"].tolist())
    progress("index_repaired", 11, retained_image_ids=remaining_ids,
             excluded_source_fov="Point5", archived_export=relative(archived_export),
             resume_at="preparer_review_record")

    reviews_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(reviews_path, directory=OUTPUT):
        reviews = pd.read_csv(reviews_path, dtype=str, keep_default_na=False)
        own_repair = (reviews.preparer == PREPARER) & reviews.check_id.isin(["S01", "D01"])
        reviews = reviews.loc[~own_repair]
        repairs = [
            review_row("S01", "unresolved", DUPLICATE),
            review_row("D01", "unresolved", "Repaired 11-image index; source identity and refreshed D01 binding pending independent review."),
        ]
        for row in repairs:
            row["source_evidence"] = json.dumps([relative(CACHE / "duplicate_point4_point5.json"),
                                                  relative(PROOF), relative(mask4), relative(mask5)])
            row["details"] = json.dumps(dict(scientific_repair_round=1,
                preparer_self_check=True, independent_review_pending=True,
                image_ids=["Point4"] if row["check_id"] == "S01" else remaining_ids,
                action="Point5 redundant intensity excluded; Point4 provisional; source masks differ; recheck S01 and D01",
                differing_mask_pixels=differing_mask_pixels,
                source_fovs=12, exported_images=11, sample_rows=0,
                sample_linked_images=0, patient_linked_samples=0), ensure_ascii=False)
        updated = pd.concat([reviews, pd.DataFrame(repairs)], ignore_index=True)
        temporary = reviews_path.with_name(reviews_path.name + ".partial")
        updated.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, reviews_path)
    report_path = CACHE / "preparation_report.json"
    with formal_read_lease(report_path):
        report = json.loads(report_path.read_text())
    report.update(expected_images=12, exported_images=11, source_fovs=12,
                  distinct_local_intensity_images=11,
                  excluded_duplicate_fov="Point5", scientific_repair_round=1,
                  duplicate_evidence=relative(CACHE / "duplicate_point4_point5.json"),
                  issues=[GAP, "Point4/Point5 intensities are identical but published instance masks differ; true FOV identity and one expected FOV intensity are unresolved.",
                          "Independent repair review and refreshed D01 binding pending."],
                  controller_requests=[
                      "Review Point4/Point5 source-mask conflict and true image identity; keep Point4 held until resolved, then recheck S01 and refresh D01 binding for the confirmed scope.",
                      "Keep M01 unresolved until a source-supported FOV-to-biospecimen-to-patient map is localized.",
                      "Repair shared read_tiff_window for OME TiffFrame pages; this study's direct page verification works."],
                  shared_module_adaptation_required=True)
    write_json(report_path, report)
    progress("repair_complete_pending_independent_review", 11,
             retained_image_ids=remaining_ids,
             archived_export=relative(archived_export),
             review_results=relative(reviews_path),
             report=relative(report_path),
             next_step="Independent reviewer assesses Point4/Point5 identity and rechecks S01/D01; M01 remains unresolved")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("inventory", "export", "verify", "finalize", "repair"))
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
mapping, inventory = source_inventory()
names = mapping[mapping.fov == "Point1"].marker_or_acquisition_channel.tolist()
assert all(group.marker_or_acquisition_channel.tolist() == names for _, group in mapping.groupby("fov"))
panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
assert int(panel.keep_channel.sum()) == 34
if args.stage == "inventory":
    print(json.dumps({k: v for k, v in inventory.items() if k != "acquisitions"}, ensure_ascii=False))
else:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.read_csv(TEMPLATES / "image_metadata.csv")
    if args.stage == "repair":
        repair_duplicate(images, panel)
    elif args.stage == "export":
        assert (CACHE / "sample_linkage_evidence.json").exists(), "Record actual metadata-source investigation before export"
        with formal_read_lease(OUTPUT):
            samples = (pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
                       if (OUTPUT / "sample_metadata.csv").exists() else pd.read_csv(TEMPLATES / "sample_metadata.csv"))
        assert samples.empty, "New sample evidence requires updating this study's explicit linkage rules"
        fields = field_map()
        rows = {row["image_id"]: row for row in images.to_dict("records")}
        try:
            for source in inventory["acquisitions"]:
                fov = source["fov"]
                if fov == "Point5":
                    continue
                destination = OUTPUT / "images" / fov / "input.tiff"
                receipt_path = CACHE / "exports" / f"{fov}.json"
                binding = dict(source=source, panel=panel.astype(str).to_dict("records"), revision=REVISION)
                if fov in rows and receipt_path.exists() and destination.exists():
                    receipt = json.loads(receipt_path.read_text())
                    stat = destination.stat()
                    assert receipt["binding"] == binding
                    assert receipt["output_stat"] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                    print(f"reuse {fov}", flush=True)
                    continue
                progress("exporting", len(rows), current_image=fov)
                required_bytes = 34 * 1024 * 1024 * 2
                free_bytes = dict(NAS=shutil.disk_usage(NAS).free, tmp=shutil.disk_usage("/tmp").free)
                if free_bytes["NAS"] < 100 * 2**30 + required_bytes or free_bytes["tmp"] < 64 * 2**30 + required_bytes:
                    progress("waiting_io", len(rows), current_image=fov,
                             reason="Free space would cross NAS 100 GiB or local 64 GiB reserve",
                             required_image_bytes=required_bytes, free_bytes=free_bytes)
                    raise SystemExit(75)
                paths = tuple(NAS / row["path"] for row in source["source_files"])
                spec = SourceSpec(fov, PANEL_ID, "tiff_planes", paths[0], (0,) * 45,
                    paths=paths, pixel_size_um=(PIXEL_UM, PIXEL_UM), source_revision=REVISION)
                # Opening TIFF readers touches source headers; the helper owns
                # the subsequent source-read and formal publication leases.
                with ExitStack() as stack:
                    with formal_read_lease(paths[0]):
                        reader = stack.enter_context(open_source(spec))
                    result = export_input(reader, panel, destination)
                assert result["admitted"], f"Unexpected all-marker-zero FOV {fov}; retain receipt and reconcile admission"
                row = {field: None for field in pd.read_csv(TEMPLATES / "image_metadata.csv").columns}
                row.update(study=STUDY, image_id=fov, panel_id=PANEL_ID, image_kind="fov",
                    source_image_id=fov, source_files=[relative(path) for path in paths],
                    source_format="tiff_planes", source_axes="YX", source_dtype="uint16",
                    source_revision=REVISION, fov_id=fov, source_x0_px=0, source_y0_px=0,
                    source_x1_px=1024, source_y1_px=1024, pad_left_px=0, pad_top_px=0,
                    pad_right_px=0, pad_bottom_px=0, height_px=1024, width_px=1024,
                    source_channel_count=45, channel_count=34, pixel_size_x_um=PIXEL_UM,
                    pixel_size_y_um=PIXEL_UM, input_image_path=relative(destination),
                    intensity_description="Published single-channel MIBI uint16 intensities, preserved exactly; upstream preprocessing not fully specified; no intensity transform in this stage",
                    split_method="already_independent",
                    split_evidence="Zenodo 8096953 description: Point directories are independent FOVs; complete original 1024x1024 frame retained",
                    metadata_notes=GAP)
                rows[fov] = row
                images = pd.DataFrame(rows.values())
                save_input_tables(OUTPUT, images, samples, panel, fields)
                stat = destination.stat()
                write_json(receipt_path, dict(binding=binding, export=result,
                    output_stat=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)))
                save_reviews([review_row("S01", "pass", "12 published Point directories; Point5 excluded as a byte-identical duplicate of Point4; 11 distinct inputs"),
                    review_row("M01", "unresolved", GAP, "sample"),
                    review_row("D01", "unresolved", f"Exported {len(rows)}/11 images. Sample linkage unresolved. Independent review pending.")])
                progress("exporting", len(rows), last_completed_image=fov)
                print(f"exported {fov}: {len(rows)}/11", flush=True)
            progress("exported_pending_self_check", len(rows))
        except Exception as error:
            progress("export_error", len(rows), error=f"{type(error).__name__}: {error}")
            raise
    elif args.stage == "verify":
        assert len(images) == 11
        verify(images, panel)
        progress("self_checked_pending_overview_review", len(images))
        print(json.dumps(dict(verified_images=len(images), pixel_sampled_images=2,
            sample_rows=0, sample_linked_images=0, patient_linked_samples=0,
            overview=relative(CACHE / "input_overview.png")), ensure_ascii=False))
    else:
        # This receipt is written only after the preparer has actually viewed
        # input_overview.png; it is not an independent acceptance record.
        overview = json.loads((CACHE / "overview_review.json").read_text())
        verified = json.loads((CACHE / "preparer_verification.json").read_text())
        assert set(images.image_id).issubset(set(overview["inspected_image_ids"]))
        assert verified["all_image_headers_checked"] == len(images) == 11
        reviews = [
            review_row("S01", "pass", "12 Point directories and 540 source files; Point5 duplicates Point4 across 45 channels, leaving 11 distinct observations."),
            review_row("I01", "pass", "11 unique Point image identities; source path lists follow explicit source-channel file order; each independent TIFF is indexed."),
            review_row("I02", "pass", "All 11 panel foreign keys resolve. sample_id is empty with the explicit M01 identity gap; no fabricated sample links."),
            review_row("I03", "pass", "Full [0,1024)x[0,1024) FOV, uint16, 0.48828125 micrometers/pixel, square, no padding or spatial transform."),
            review_row("I04", "pass", overview),
            review_row("P01", "pass", "All 45 source file labels match this study's S1 rows exactly; no aliases. File order is explicit; no mass-acquisition C-axis order is inferred.", "panel", PANEL_ID),
            review_row("P02", "pass", "34 kept measurements, 11 S1 exclusions; all selected nuclear/body channels retained; unique downstream markers; source and output positions are distinct.", "panel", PANEL_ID),
            review_row("P03", "pass", "All 11 retained OME BigTIFFs have CYX 34x1024x1024 uint16, S1 names/order, and native pixel size; all 39 base image fields published."),
            review_row("I05", "pass", verified),
            review_row("M01", "unresolved", dict(sample_rows=0, sample_linked_images=0,
                patient_linked_samples=0, reported_cohort_individuals=6, gap=GAP,
                evidence=relative(CACHE / "sample_linkage_evidence.json")), "sample"),
            review_row("M02", "not_applicable", "No confirmed biological sample/clinical records to populate; full 71-column sample template remains empty. Source search and missing joins are documented under M01.", "sample"),
            review_row("D01", "unresolved", dict(exported_images=11, expected_images=12,
                sample_rows=0, sample_linked_images=0, patient_linked_samples=0,
                pending="Independent CLI review and unresolved source identity joins; preparer does not grant D01 acceptance.")),
        ]
        save_reviews(reviews)
        report = dict(study=STUDY, status="partial", expected_images=12,
            exported_images=11, sample_rows=0, sample_linked_images=0,
            patient_linked_samples=0, output_root=relative(OUTPUT),
            script="scripts/data/preparation/Liu2023_MIBI_LymphNode.py",
            verification=relative(CACHE / "preparer_verification.json"),
            metadata_evidence=relative(CACHE / "sample_linkage_evidence.json"),
            issues=[GAP, "Independent review pending; D01 remains unresolved."],
            controller_requests=["Arrange independent review of the 11 distinct native input TIFFs and tables.",
                "Retain M01 unresolved until a source-supported Point-to-biospecimen-to-patient map is localized; preserve the completed image products."],
            shared_module_adaptation_required=False)
        write_json(CACHE / "preparation_report.json", report)
        progress("prepared_partial_metadata_independent_review_pending", len(images),
                 report=relative(CACHE / "preparation_report.json"))
        print(json.dumps(report, ensure_ascii=False))
