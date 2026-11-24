"""Prepare the ten released CellTune melanoma MIBI FOVs at published intensity.

Inputs are the verified Depot v0.2 marker TIFFs, ordered MarkerTable, and S1.
Outputs are independent input.tiff files and the five study input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative, read_metadata_member

import json
import os
from pathlib import Path
import shutil
import signal
import xml.etree.ElementTree as ET

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


STUDY = "celltune_melanoma_oct2522"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "celltunedepot_v0.2"
META = metadata_source_directory(NAS, STUDY)
SOURCE_BUNDLE = NAS / "raw/public" / STUDY / "metadata.json"
PROOF_MEMBER = "public/source_followups/20261108/celltune/melanoma_per_image_verification.json"
PROOF = Path(str(SOURCE_BUNDLE) + "#" + PROOF_MEMBER)
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0697_prepare_0"
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY / "sources"
PANEL_ID = "Melanoma_Oct2522"
REVISION = "CellTuneDepot-v0.2/ManualLabels/Melanoma_Oct2522"


def relative(path: Path) -> str:
    if path.is_relative_to(JOB):
        return str(path)
    return metadata_source_relative(path, NAS).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def source_scope() -> tuple[list[dict], pd.DataFrame]:
    mapping = json.loads((META / "channel_mapping.json").read_text())
    names = pd.read_csv(RAW / "Tables/MarkerTable.csv").Marker.tolist()
    assert names == mapping["measured_marker_names"]
    records = sorted(mapping["images"], key=lambda row: row["image_id"])
    with formal_read_lease(SOURCE_BUNDLE, data_root=NAS):
        proofs = {row["image_id"]: row for row in json.loads(read_metadata_member(SOURCE_BUNDLE, PROOF_MEMBER))}
    assert len(records) == len(proofs) == 10
    assert {path.name for path in (RAW / "Images").iterdir() if path.is_dir()} == set(proofs)
    for record in records:
        image_id = record["image_id"]
        channels = record["channels"]
        assert len(channels) == len(proofs[image_id]["channels"]) == 40
        assert [row["marker"] for row in channels] == names
        for channel, proof in zip(channels, proofs[image_id]["channels"], strict=True):
            assert channel["index"] == proof["panel_index_0based"]
            assert channel["filename"] == proof["file"]
            assert proof["shape"] == [1024, 1024] and proof["dtype"] == "uint8"
        record["source_paths"] = [RAW / "Images" / image_id / name for name in (row["filename"] for row in channels)]
        record["source_bytes"] = [row["bytes"] for row in proofs[image_id]["channels"]]
    panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
    assert len(panel) == 40 and panel.keep_channel.sum() == 38
    assert panel.loc[~panel.keep_channel, "source_name"].tolist() == ["Fe56", "MPO_Calp"]
    return records, panel


def stage_source(record: dict) -> list[Path]:
    image_id = record["image_id"]
    directory = STAGING / image_id
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    def timed_out(signum: int, frame: object) -> None:
        raise TimeoutError(f"Source read lease or staging exceeded 180 seconds for {image_id}")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(180)
    try:
        with formal_read_lease(RAW):
            for source, expected_bytes in zip(record["source_paths"], record["source_bytes"], strict=True):
                target = directory / source.name
                source_stat = source.stat()
                assert source_stat.st_size == expected_bytes
                if not target.is_file() or (target.stat().st_size, target.stat().st_mtime_ns) != (
                    source_stat.st_size, source_stat.st_mtime_ns
                ):
                    temporary = target.with_name(target.name + ".partial")
                    shutil.copy2(source, temporary)
                    assert (temporary.stat().st_size, temporary.stat().st_mtime_ns) == (
                        source_stat.st_size, source_stat.st_mtime_ns
                    )
                    temporary.replace(target)
                staged.append(target)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return staged


def field_rules() -> pd.DataFrame:
    rows = []

    def add(source: str, source_field: str, target_table: str, target_field: str,
            join_keys: str, rule: str, notes: str) -> None:
        rows.append(dict(study=STUDY, source_file=source, source_sheet_or_table="",
            source_field=source_field, target_table=target_table, target_field=target_field,
            join_keys=join_keys, value_mapping="{}", transformation_rule=rule,
            time_origin_rule="", conflict_rule="Unconfirmed biological or clinical values remain empty",
            notes=notes))

    add(relative(META / "channel_mapping.json"), "images.image_id;channels.filename",
        "image_metadata", "image_id;source_files;source_image_id", "image_id=FOV directory name",
        "Use each of the ten verified whole FOVs as an independent image; source_files lists all 40 published marker TIFFs.",
        "Source proof: " + relative(PROOF))
    add(relative(RAW / "Tables/MarkerTable.csv"), "Marker", "panel_mapping",
        "source_name;source_channel_index", "Marker=TIFF stem",
        "Use MarkerTable row order; select named single-channel TIFFs in that order.",
        "40/40 filenames per FOV matched the prior full-decode proof.")
    add(relative(S1), "Study;Original marker;Unified marker;Segmentation channel",
        "panel_mapping", "marker_id;role;keep_channel;nuclear;body;output_channel_index",
        "Study=celltune_melanoma_oct2522; Original marker=Marker",
        "Keep 38 S1-admitted channels in native source order; exclude Fe56 and MPO_Calp.",
        "HH3 remains the documented histone H3 nuclear signal.")
    add(relative(RAW / "Images"), "Slide/Point;TMA tag", "image_metadata",
        "slide_id;core_id;fov_id;sample_id", "FOV directory name",
        "Preserve source slide and TMA region tags; leave sample_id empty until a biological specimen map is verified.",
        "Ten FOVs; 0 confirmed image-to-sample links, 0 sample rows, 0 patient links. TMA tags identify regions only. "
        "Local Labels.csv, CellTypes.csv, MarkerTable.csv, README_source.md, Depot attribution and public Depot page "
        "contain no FOV-to-specimen or patient key; original cohort publication remains unidentified. "
        "Public source: https://celltune.org/celltunedepot.html")
    add(relative(RAW / "Tables"), "Labels.image;cellID;class", "sample_metadata",
        "sample_id;patient_key", "Labels.image=FOV name",
        "Cell labels establish FOV identity only and cannot assign a biological sample or patient.",
        "Clinical fields, anatomical site per FOV, treatment and timepoint are not supplied; leave empty.")
    return pd.DataFrame(rows).reindex(columns=columns("field_mapping"))


def image_row(record: dict, destination: Path, result: dict) -> dict:
    image_id = record["image_id"]
    parts = image_id.split("_")
    slide = parts[0]
    core = next((part for part in parts if part.startswith("TMA")), None)
    return dict(study=STUDY, image_id=image_id, sample_id=None, panel_id=PANEL_ID,
        image_kind="fov", source_image_id=image_id,
        source_files=[relative(path) for path in record["source_paths"]],
        source_format="TIFF planes", source_axes="YX", source_dtype="uint8",
        source_revision=REVISION, slide_id=slide, core_id=core, fov_id=image_id,
        region_annotation=None, source_x0_px=0, source_y0_px=0,
        source_x1_px=1024, source_y1_px=1024, pad_left_px=result["pad_left_px"],
        pad_top_px=result["pad_top_px"], pad_right_px=result["pad_right_px"],
        pad_bottom_px=result["pad_bottom_px"], height_px=result["height_px"],
        width_px=result["width_px"], source_channel_count=40, channel_count=38,
        input_image_path=relative(destination), valid_pixels_path=None,
        intensity_description="Published uint8 per-marker MIBI TIFF intensities; no intensity transform",
        split_method="already_independent",
        split_evidence=relative(META / "panel_qualification.json"),
        metadata_notes="Source FOV directory is complete; specimen and patient association unresolved")


def verify_header(destination: Path, panel: pd.DataFrame) -> None:
    names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
    with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
        series = tif.series[0]
        assert series.axes == "CYX" and series.shape == (38, 1024, 1024)
        assert series.dtype == np.dtype("uint8")
        xml = ET.fromstring(tif.ome_metadata)
        observed = [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")]
        assert observed == names


def reviews(n: int, *, pixels_verified: bool = False) -> pd.DataFrame:
    source = ";".join(map(relative, (RAW / "Tables", META / "channel_mapping.json", PROOF, S1)))
    descriptions = {
        "S01": f"Source scope is 10 released complete FOVs; {n} exported; original cohort 302 FOVs are outside local scope.",
        "I01": f"{n} image IDs map to verified named FOV directories and original TIFF paths.",
        "I02": f"{n} image-to-panel links; no confirmed image-to-sample connection.",
        "I03": "Whole 1024x1024 native FOVs; no crop, padding or resampling.",
        "I04": "Source TMA labels retained as region IDs; no specimen boundary split required.",
        "P01": "40 ordered MarkerTable rows match named source TIFFs and study S1.",
        "P02": "38 S1-admitted channels retained; Fe56 and MPO_Calp excluded.",
        "P03": f"{n} published OME-TIFF headers reopened as CYX uint8 with 38 ordered names.",
        "M01": "Local source tables, attribution, prior proof and public Depot page checked; 0 sample rows, "
                 "0 image-to-sample links and 0 patient links. Biological identity map unresolved.",
        "M02": "No per-FOV clinical table or specimen/patient bridge; clinical values remain empty.",
        "I05": ("Preparer checked both 32x32 interior and boundary windows for all 38 retained "
                "channels of Slide01_Point001; independent review pending."
                if pixels_verified else "Preparer pixel spot check pending completion of source scope."),
        "D01": f"Preparer delivered {n}/10 images; independent review and current input_binding pending; "
                 "sample and patient links unresolved.",
    }
    rows = [dict(study=STUDY, preparer="0697_prepare_0", reviewer="", scope="study",
        record_id=STUDY, check_id=check,
        result="unresolved" if check in {"M01", "M02", "D01"} or check == "I05" and not pixels_verified else "pass",
        source_evidence=source, output_evidence=relative(OUTPUT),
        details="Preparer self-check; independent reviewer pending. " + details)
        for check, details in descriptions.items()]
    return pd.DataFrame(rows).reindex(columns=columns("review_results"))


def save_reviews(frame: pd.DataFrame) -> None:
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


def main() -> None:
    records, panel = source_scope()
    samples = pd.DataFrame(columns=columns("sample_metadata"))
    fields = field_rules()
    image_path = OUTPUT / "images.parquet"
    with formal_read_lease(OUTPUT):
        existing = pd.read_parquet(image_path) if image_path.is_file() else pd.DataFrame()
    rows = {row["image_id"]: row for row in existing.to_dict("records")}
    estimate = 10 * 38 * 1024 * 1024
    nas_free = shutil.disk_usage(NAS).free
    local_free = shutil.disk_usage(STAGING.parent).free if STAGING.parent.exists() else shutil.disk_usage("/tmp").free

    def progress(stage: str, **extra: object) -> None:
        remaining = [record["image_id"] for record in records if record["image_id"] not in rows]
        publish_json(JOB / "progress.json", dict(study=STUDY, phase="prepare",
            scientific_repair_round=0, stage=stage, expected_images=10, completed_images=len(rows),
            sample_rows=0, sample_linked_images=0, patient_linked_samples=0,
            next_image=remaining[0] if remaining else None, remaining_images=len(remaining), **extra))

    progress("source_checked", estimated_uncompressed_output_bytes=estimate,
             nas_free_bytes=nas_free, local_free_bytes=local_free)
    if nas_free - estimate < 100 * 1024**3 or local_free - 2 * estimate < 64 * 1024**3:
        progress("waiting_io", waiting_reason="NAS 100 GiB or local 64 GiB reserve unavailable")
        return

    for record in records:
        image_id = record["image_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in rows:
            assert destination.stat().st_size > 0
            continue
        try:
            staged = stage_source(record)
            source = SourceSpec(image_id, PANEL_ID, "tiff_planes", staged[0],
                tuple(0 for _ in staged), paths=tuple(staged), source_revision=REVISION,
                intensity_description="Published uint8 MIBI marker TIFF intensities")
            with open_source(source) as reader:
                result = export_input(reader, panel, destination, tile_size=512)
            assert result["admitted"] and result["source_dtype"] == "uint8"
            verify_header(destination, panel)
            rows[image_id] = image_row(record, destination, result)
            save_input_tables(OUTPUT, pd.DataFrame(rows.values()), samples, panel, fields)
            shutil.rmtree(STAGING / image_id)
            progress("exporting", last_completed_image=image_id)
            print(f"exported {len(rows)}/10 {image_id}", flush=True)
        except (OSError, TimeoutError) as error:
            progress("waiting_io", waiting_reason=f"{type(error).__name__}: {error}",
                     current_image=image_id, resume_scope=f"{len(rows)}/10 committed; retry {image_id}")
            return

    # Two windows in one representative FOV check all retained source channels.
    first = records[0]
    destination = OUTPUT / "images" / first["image_id"] / "input.tiff"
    selected = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    windows = [(448, 448, 480, 480), (992, 992, 1024, 1024)]
    with formal_read_lease(RAW), formal_read_lease(destination):
        with tifffile.TiffFile(destination) as output:
            for channel in selected.itertuples(index=False):
                source_path = first["source_paths"][channel.source_channel_index]
                with tifffile.TiffFile(source_path) as native:
                    for window in windows:
                        expected = read_tiff_window(native, 0, window)
                        observed = read_tiff_window(output, int(channel.output_channel_index), window)
                        np.testing.assert_array_equal(expected, observed)
    save_reviews(reviews(len(rows), pixels_verified=True))
    progress("self_checked_pending_independent_review", pixel_spotcheck_image=first["image_id"],
             pixel_spotcheck_windows=windows, pixel_spotcheck_channels=len(selected))


if __name__ == "__main__":
    main()
