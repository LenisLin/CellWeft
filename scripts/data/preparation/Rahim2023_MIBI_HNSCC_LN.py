"""Deliver native MIBI input images for 96 distinct HNSCC lymph-node cores.

Inputs: original Mendeley ZIP, accepted native proofs, embedded core metadata,
and the study's S1 rows. Outputs: resumable native input TIFFs and input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Rahim2023_MIBI_HNSCC_LN"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
ARCHIVE = NAS / "raw/public" / STUDY / "public_original/MIBI_images.zip"
META = metadata_source_directory(NAS, STUDY)
PROOF = META / "native_verification.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
OUTPUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(CACHE)))
PANEL_ID = "Mendeley_MIBI_41"
REVISION = "Mendeley 2zgppyr2rr v1 original MIBI_images.zip"
PREPARER = f"Codex {STUDY} {JOB.name} preparer"


def relative(path: Path) -> str:
    return metadata_source_relative(path, NAS).as_posix()


def publish_json(path: Path, content: dict) -> None:
    payload = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(payload)
        temporary.replace(path)


def checkpoint(stage: str, rows: dict, **details: object) -> None:
    publish_json(JOB / "progress.json", dict(study=STUDY, phase="prepare",
        scientific_repair_round=0, stage=stage, expected_images=96,
        exported_images=len(rows), sample_rows=len(rows),
        sample_linked_images=len(rows), patient_linked_samples=len(rows),
        updated_unix=time.time(), **details))


class LocalPlanes:
    """Expose only selected, uniformly uint16 source planes in source-index order."""

    def __init__(self, paths: dict[int, Path], shape: tuple[int, int]):
        self.paths = paths
        self.source = type("Source", (), {"path": next(iter(paths.values())), "pixel_size_um": None})()
        self.shape = (41, *shape)
        self.dtype = np.dtype("uint16")
        self.tiffs = {index: tifffile.TiffFile(path) for index, path in paths.items()}
        for tif in self.tiffs.values():
            page = tif.pages[0]
            if page.shape != shape or page.dtype != self.dtype or len(tif.pages) != 1:
                raise ValueError("Selected native TIFF plane differs from accepted uint16 YX layout")

    def close(self) -> None:
        for tif in self.tiffs.values():
            tif.close()

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        return np.stack([read_tiff_window(self.tiffs[index], 0, window) for index in channels])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def field_rules() -> pd.DataFrame:
    columns = pd.read_csv(TEMPLATES / "field_mapping.csv").columns
    rows = []

    def add(source: str, field: str, table: str, target: str, keys: str, rule: str, notes: str = "") -> None:
        rows.append(dict(study=STUDY, source_file=source, source_sheet_or_table="",
            source_field=field, target_table=table, target_field=target, join_keys=keys,
            value_mapping="{}", transformation_rule=rule, time_origin_rule="",
            conflict_rule="Preserve source values; leave unsupported clinical fields empty", notes=notes))

    add(relative(PROOF), "counted;physical_identity;source_folder;source_fov", "image_metadata",
        "image_id;source_image_id;core_id;source_files", "source_fov=embedded MIBI_metadata.fov",
        "Use the 96 complete counted physical-core representatives; preserve ZIP and member folder identity."
        " The other source folders are repeats or held by original intake scope.")
    add(relative(ARCHIVE), "MIBI_images/MIBI_metadata.csv: surgicalpathid_block_core;patient_id", "sample_metadata",
        "sample_id;patient_key;source_sample_id;source_subject_id", "source_fov=fov;physical_core_id=surgicalpathid_block_core",
        "Each distinct physical core is one biological sample; link its source patient_id."
        " All 96 counted cores link to 32 source patients.")
    add(relative(ARCHIVE), "MIBI_metadata.csv: surgical_path_id;core_status;uninvolved_LN;uninvolved_core;treated;days_from_atezo1_to_surgery;LN_number", "sample_metadata",
        "biospecimen_group_id;metadata_notes", "source_fov=fov",
        "Preserve source clinical values in metadata_notes; surgical_path_id groups specimen provenance."
        " No index-date or treatment interpretation is inferred from flags.")
    add(relative(ARCHIVE), "MIBI_metadata.csv: TMA_core_ROI;LN_number;core_status", "image_metadata",
        "region_annotation;roi_id;metadata_notes", "source_fov=fov",
        "Preserve ROI and LN annotations verbatim with core status; retain full source FOV.")
    add(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
        "source_name;marker_id;role;nuclear;body;keep_channel;output_channel_index",
        "Study=Rahim2023_MIBI_HNSCC_LN;Original marker=source TIFF basename",
        "41 ordered native labels, 36 selected uint16 channels. Excluded Au is uint32; no intensity conversion is applied to selected channels.")
    add(relative(PROOF), "native_common_shape;native_read_evidence;source_member_stats", "image_metadata",
        "source_axes;source_dtype;source_channel_count;source_x1_px;source_y1_px",
        "source_fov=accepted native image proof",
        "Use each accepted native shape and selected-channel dtype; source channel order comes from native_markers.json."
        " Pixel size is unconfirmed and remains blank.")
    return pd.DataFrame(rows, columns=columns)


def review_rows(count: int) -> pd.DataFrame:
    columns = pd.read_csv(TEMPLATES / "review_results.csv").columns
    evidence = json.dumps([relative(PROOF), relative(META / "native_markers.json"), relative(ARCHIVE), relative(S1)])
    checks = [
        ("S01", "unresolved", f"{count}/96 distinct source cores exported; source scope requires independent review"),
        ("M01", "unresolved", f"{count} sample rows, {count} image-sample and {count} sample-patient links from embedded metadata; independent source review pending"),
        ("D01", "unresolved", f"{count}/96 input TIFFs and tables; independent acceptance pending"),
    ]
    return pd.DataFrame([dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
        record_id=STUDY, check_id=check, result=result, source_evidence=evidence,
        output_evidence=relative(OUTPUT), details=detail) for check, result, detail in checks], columns=columns)


def publish_reviews(count: int) -> None:
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        temporary = path.with_name(path.name + ".partial")
        review_rows(count).to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


def repair_clinical_field_mapping() -> None:
    path = OUTPUT / "field_mapping.csv"
    with formal_read_lease(path):
        current = pd.read_csv(path, keep_default_na=False)
        version = (path.stat().st_size, path.stat().st_mtime_ns)
    target = (current.target_table == "sample_metadata") & (
        current.target_field == "biospecimen_group_id;metadata_notes")
    assert target.sum() == 1
    rule = field_rules().loc[lambda frame: (
        (frame.target_table == "sample_metadata") &
        (frame.target_field == "biospecimen_group_id;metadata_notes"))].iloc[0]
    if current.loc[target, "source_field"].iloc[0] != rule.source_field:
        current.loc[target, "source_field"] = rule.source_field
        payload = current.to_csv(index=False).encode("utf-8-sig")
        with formal_publish_lease(path, directory=OUTPUT):
            assert (path.stat().st_size, path.stat().st_mtime_ns) == version
            temporary = path.with_name(path.name + ".partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
    publish_json(JOB / "progress.json", dict(
        study=STUDY, phase="repair", scientific_repair_round=1,
        stage="clinical_mapping_repaired_pending_independent_review",
        expected_images=96, exported_images=96, sample_rows=96,
        sample_linked_images=96, patient_linked_samples=96,
        changed_file=relative(path), next_step="Independent M02 review and D01 rebind",
        updated_unix=time.time()))


def sample_row(proof: dict, columns: list[str]) -> dict:
    meta = proof["source_metadata"]
    row = dict.fromkeys(columns)
    row.update(study=STUDY, sample_id=proof["physical_identity"],
        patient_key=meta["patient_id"], source_subject_id=meta["patient_id"],
        source_sample_id=proof["physical_identity"], biospecimen_group_id=meta["surgical_path_id"],
        sample_type="tissue", metadata_notes=json.dumps({key: meta[key] for key in (
            "core_status", "uninvolved_LN", "uninvolved_core", "treated",
            "days_from_atezo1_to_surgery", "LN_number")}, ensure_ascii=False))
    return row


def image_row(proof: dict, result: dict, destination: Path, columns: list[str]) -> dict:
    meta = proof["source_metadata"]
    h, w = proof["native_common_shape"]
    row = dict.fromkeys(columns)
    row.update(study=STUDY, image_id=proof["source_fov"], sample_id=proof["physical_identity"],
        panel_id=PANEL_ID, image_kind="tma_core", source_image_id=proof["source_folder"],
        source_files=[relative(ARCHIVE)], source_format="tiff_planes", source_axes="YX",
        source_dtype="uint16", source_revision=REVISION, core_id=proof["physical_identity"],
        roi_id=meta["TMA_core_ROI"], fov_id=proof["source_fov"],
        region_annotation=json.dumps({"TMA_core_ROI": meta["TMA_core_ROI"],
            "LN_number": meta["LN_number"], "core_status": meta["core_status"]}),
        source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=result["height_px"], width_px=result["width_px"],
        source_channel_count=41, channel_count=result["channel_count"],
        input_image_path=relative(destination),
        intensity_description="Original decoded uint16 MIBI TIFF intensities; no normalization in input stage",
        split_method="already_independent",
        split_evidence="Accepted native proof selects one complete FOV per physical core",
        metadata_notes="Pixel size not established from inspected source; source contains repeated views of some cores")
    return row


def main() -> None:
    with formal_read_lease(PROOF):
        proofs = json.loads(PROOF.read_text())["native_result"]["images"]
    with formal_read_lease(META / "native_markers.json"):
        markers = json.loads((META / "native_markers.json").read_text())["panels"]
    assert len(proofs) == len(markers) == 96
    names = markers[0]["ordered_source_labels"]
    assert len(names) == 41 and all(item["ordered_source_labels"] == names for item in markers)
    assert all(proof["complete"] and proof["counted"] for proof in proofs)
    assert len({proof["physical_identity"] for proof in proofs}) == 96
    panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
    assert int(panel.keep_channel.sum()) == 36
    kept = panel.loc[panel.keep_channel, ["source_channel_index", "source_name", "marker_id"]]
    for proof in proofs:
        native = {item["marker"]: item for item in proof["native_read_evidence"]}
        assert all(native[name]["dtype"] == "uint16" for name in kept.source_name
                   if name in native)
    stat = ARCHIVE.stat()
    assert all(proof["source_file_stat"]["size"] == stat.st_size and
               proof["source_file_stat"]["mtime_ns"] == stat.st_mtime_ns for proof in proofs)
    expected_names = {proof["source_fov"] for proof in proofs}

    image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
    sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
    with formal_read_lease(OUTPUT):
        existing = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else None
    rows = {} if existing is None else existing.set_index("image_id", drop=False).to_dict("index")
    assert set(rows).issubset(expected_names)
    samples = {proof["physical_identity"]: sample_row(proof, sample_columns) for proof in proofs}
    field_mapping = field_rules()
    checkpoint("source_checked", rows, next_image=next((proof["source_fov"] for proof in proofs
        if proof["source_fov"] not in rows), None), source_bytes=stat.st_size,
        estimated_native_output_bytes=sum(72 * int(np.prod(proof["native_common_shape"])) for proof in proofs),
        nas_free_bytes=shutil.disk_usage(NAS).free, tmp_free_bytes=shutil.disk_usage("/tmp").free)

    for proof in proofs:
        image_id = proof["source_fov"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in rows:
            with formal_read_lease(destination):
                assert destination.is_file() and destination.stat().st_size > 0
            continue
        h, w = proof["native_common_shape"]
        estimate = 36 * h * w * 2
        nas_free, tmp_free = shutil.disk_usage(NAS).free, shutil.disk_usage("/tmp").free
        if nas_free < 100 * 2**30 + estimate or tmp_free < 64 * 2**30 + 2 * estimate:
            checkpoint("waiting_io", rows, next_image=image_id, reason="Space reserve would be crossed",
                nas_free_bytes=nas_free, tmp_free_bytes=tmp_free, estimated_image_bytes=estimate)
            raise SystemExit(75)
        checkpoint("extracting", rows, next_image=image_id)
        with tempfile.TemporaryDirectory(prefix="rahim-source-", dir="/tmp") as directory:
            local = Path(directory)
            paths = {}
            with formal_read_lease(ARCHIVE), zipfile.ZipFile(ARCHIVE) as archive:
                for item in kept.itertuples(index=False):
                    name = f"{proof['source_folder']}/{item.source_name}.tiff"
                    info = archive.getinfo(name)
                    assert info.file_size == proof["source_member_stats"][item.source_name]["file_size"]
                    path = local / f"{item.source_channel_index}.tiff"
                    with archive.open(info) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    paths[int(item.source_channel_index)] = path
            reader = LocalPlanes(paths, (h, w))
            try:
                result = export_input(reader, panel, destination, tile_size=max(h, w))
            finally:
                reader.close()
        assert result["admitted"] and result["channel_count"] == 36
        with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
            assert tif.series[0].shape == (36, max(h, w), max(h, w))
            assert tif.series[0].dtype == np.dtype("uint16")
        rows[image_id] = image_row(proof, result, destination, image_columns)
        current = pd.DataFrame(rows.values(), columns=image_columns)
        linked = pd.DataFrame([samples[identity] for identity in current.sample_id], columns=sample_columns)
        save_input_tables(OUTPUT, current, linked, panel, field_mapping)
        publish_reviews(len(rows))
        checkpoint("exporting", rows, last_completed_image=image_id,
            next_image=next((p["source_fov"] for p in proofs if p["source_fov"] not in rows), None))
        print(f"exported {len(rows)}/96 {image_id}", flush=True)
    checkpoint("prepared_pending_independent_review", rows, next_image=None)


if __name__ == "__main__":
    if sys.argv[1:] == ["--repair-field-mapping"]:
        repair_clinical_field_mapping()
    else:
        main()
