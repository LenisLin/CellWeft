"""Prepare the 48 source-bound native CODEX regions from the official release."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, _export_input, panel_from_s1, save_input_tables,
)


STUDY = "Matusiak2024_CODEX_MultiDisease"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE = ROOT / "raw/public" / STUDY / "official_gcs_2023"
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
JOB = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_JOB",
    str(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0239_prepare_0"),
))
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
PROOFS = META / "native_verification.json"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
CLINICAL_URL = "https://storage.googleapis.com/spatial-mac-data/Analysis_code_and_tab_data/CODEX/sample_clinical_data_Spatial_Mac_Map.xlsx"
REGION_URL = "https://storage.googleapis.com/spatial-mac-data/Analysis_code_and_tab_data/CODEX/tissue_region_meta_data_CODEX_spatial_MAC_Map.csv"
CLINICAL = META / "source_metadata" / "sample_clinical_data_Spatial_Mac_Map.xlsx"
REGIONS = META / "source_metadata" / "tissue_region_meta_data_CODEX_spatial_MAC_Map.csv"
HEIGHT, WIDTH, PLANE_BYTES, PAYLOAD_START = 3468, 4620, 32044320, 349
MAX_OUTPUT_BYTES = 48 * 40 * WIDTH * WIDTH * 2
CURRENT: dict[str, object] = {}


class WaitingIO(RuntimeError):
    pass


@contextmanager
def acquisition_timeout(label: str):
    def expired(_signum: int, _frame: object) -> None:
        raise WaitingIO(label)

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def bounded_read(path: Path):
    with ExitStack() as stack:
        with acquisition_timeout(f"read lease: {path.name}"):
            stack.enter_context(formal_read_lease(path))
        yield


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return metadata_source_relative(path, ROOT).as_posix()


def publish_file(source: Path, destination: Path, *, priority: str = "normal") -> None:
    """Stage outside the NAS gate and publish under NAS, study, input locks."""
    with ExitStack() as stack:
        with acquisition_timeout(f"publication lease: {destination.name}"):
            stack.enter_context(formal_publish_lease(destination, directory=OUTPUT, priority=priority))
            lock_directory = destination.parent if destination.name == "input.tiff" else OUTPUT
            stack.enter_context(resource_lock(lock_directory, name=".input.lock"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, partial)
        if partial.stat().st_size != source.stat().st_size:
            raise ValueError(f"Incomplete formal publication: {destination}")
        os.replace(partial, destination)


def checkpoint(stage: str, *, from_local: bool = False, **details: object) -> None:
    if from_local:
        state = json.loads((TEMP / "progress.json").read_text())
        indexed = CURRENT.get("indexed_images", 0)
        if isinstance(indexed, int) and indexed > state["exported_images"]:
            state["exported_images"] = indexed
            state["sample_linked_images"] = indexed
    else:
        with bounded_read(OUTPUT):
            images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
            samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
        state = {
            "study": STUDY, "phase": "prepare", "scientific_repair_round": 0,
            "expected_images": 48, "exported_images": len(images),
            "sample_rows": len(samples),
            "sample_linked_images": int(images.sample_id.notna().sum()) if len(images) else 0,
            "patient_linked_samples": int(samples.patient_key.notna().sum()) if len(samples) else 0,
        }
    state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / "progress.json"
    staged.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    publish_file(staged, JOB / "progress.json", priority="high" if stage == "waiting_io" else "normal")
    print(json.dumps(state, ensure_ascii=False), flush=True)


def stage_official_metadata() -> None:
    for staged, destination in (
        (Path("/tmp/Matusiak2024_sample_clinical_data.xlsx"), CLINICAL),
        (Path("/tmp/Matusiak2024_tissue_region_metadata.csv"), REGIONS),
    ):
        if not destination.exists():
            if not staged.exists():
                raise FileNotFoundError(f"Download the official small metadata attachment: {destination.name}")
            publish_file(staged, destination)


def source_scope() -> tuple[list[dict], dict[str, list[str]]]:
    publication = json.loads(PROOFS.read_text())
    proofs = publication["native_proofs"]
    if len(proofs) != 48 or publication["blockers"]:
        raise ValueError("Official native qualification is not complete")
    panels = json.loads((META / "native_ordered_markers.json").read_text())["panels"]
    scope = []
    seen = set()
    for proof in proofs:
        cohort, filename = proof["acquisition_id"].split("/", 1)
        source = SOURCE / cohort / filename
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (proof["bytes"], proof["mtime_ns"]):
            raise ValueError(f"Source changed since qualification: {cohort}/{filename}")
        if (not proof["source_archive"].endswith(
                ("BC_stacked_TIFs_andChannel_names.zip" if cohort == "BC" else
                 "CRC_stacked_TIFs_with_channel_names.zip")) or proof["source_member"] != filename):
            raise ValueError(f"Source member identity mismatch: {cohort}/{filename}")
        native_labels = proof.get("source_ordered_labels", proof["ordered_labels"])
        if native_labels != panels[cohort]["ordered_labels"]:
            raise ValueError(f"Panel identity mismatch: {cohort}/{filename}")
        available = 79 if cohort == "BC" else 76
        if stat.st_size < PAYLOAD_START + available * PLANE_BYTES:
            raise ValueError(f"Required native planes incomplete: {cohort}/{filename}")
        if proof.get("all_segment_bounds_checked") is not True:
            raise ValueError(f"Native qualification missing: {cohort}/{filename}")
        key = (cohort, filename.removesuffix("_montage.tif"))
        if key in seen:
            raise ValueError(f"Duplicate acquisition: {key}")
        seen.add(key)
        scope.append({"cohort": cohort, "reg": key[1], "source": source, "proof": proof})
    return sorted(scope, key=lambda x: (x["cohort"], x["reg"])), {
        cohort: panel["ordered_labels"] for cohort, panel in panels.items()
    }


def source_metadata(scope: list[dict]) -> pd.DataFrame:
    regions = pd.read_csv(REGIONS, dtype=str)
    clinical = pd.read_excel(CLINICAL, dtype=str)
    clinical["reg"] = clinical.reg.str.zfill(3).radd("reg")
    regions["cohort"] = regions.tissue.map({"breast": "BC", "colon": "CRC"})
    clinical["cohort"] = clinical.tissue.map(
        {"breast": "BC", "colon": "CRC", "ILEUM": "CRC", "duodenum": "CRC"})
    if regions.cohort.isna().any() or clinical.cohort.isna().any():
        raise ValueError("Unrecognized clinical tissue category")
    joined = regions.merge(clinical, on=["cohort", "reg"], how="left",
                           suffixes=("_region", "_clinical"), validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Source region has no clinical record")
    for name in ("donor_number", "patient_number"):
        if not joined[f"{name}_region"].eq(joined[f"{name}_clinical"]).all():
            raise ValueError(f"Conflicting source identity: {name}")
    expected = {(item["cohort"], item["reg"]) for item in scope}
    if len(joined) != 48 or set(zip(joined.cohort, joined.reg)) != expected:
        raise ValueError("Official region table does not match qualified images")
    return joined.set_index(["cohort", "reg"], verify_integrity=True)


def build_panels(labels: dict[str, list[str]]) -> pd.DataFrame:
    parts = []
    for cohort, names in labels.items():
        drops = {79: "BC source-bound proof: trailing DRAQ5 plane incomplete"} if cohort == "BC" else {}
        panel = panel_from_s1(STUDY, cohort, names, S1, drop_indices=drops)
        if set(panel.loc[panel.nuclear, "source_name"]) != {"Hoechst2", "Hoechst3"}:
            raise ValueError(f"Unexpected S1 nuclear selection: {cohort}")
        parts.append(panel.drop(columns="s1_original_marker"))
    return pd.concat(parts, ignore_index=True)


def sample_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
    rows = []
    for (cohort, reg), source in metadata.iterrows():
        row = dict.fromkeys(columns)
        row.update(study=STUDY, sample_id=f"{cohort}_donor_{source.donor_number_region}",
                   patient_key=f"{cohort}_patient_{source.patient_number_region}",
                   source_subject_id=source.patient_number_region,
                   source_sample_id=source.donor_number_region,
                   sample_type="tissue region", anatomical_site=source.tissue_clinical,
                   region_type=source.type, clinical_region=source.region,
                   diagnostic_subtype=(None if source.diagnostic_subtype == "UNKNOWN" else
                                       source.diagnostic_subtype),
                   neoadjuvant_treatment=source.NEOAdjuvant_chemo_or_radiation,
                   her2_status=source.HER2, er_status=source.ER, pr_status=source.PR,
                   ki67_percent=source["ki67 %"], grade=source.grade,
                   metadata_notes=("Official tissue-region CSV and clinical workbook joined by "
                                   "cohort + reg; donor and patient identifiers agree. "
                                   "Cohort is retained in patient_key because cross-cohort identity is unverified."))
        rows.append(row)
    return pd.DataFrame(rows).where(lambda x: x.notna(), None)


def field_rows() -> pd.DataFrame:
    columns = pd.read_csv(TEMPLATES / "field_mapping.csv").columns.tolist()
    common = {"study": STUDY, "source_sheet_or_table": "", "join_keys": "cohort + reg",
              "value_mapping": "{}", "time_origin_rule": "", "conflict_rule": "Reject discordant donor/patient identifiers"}
    rows = []
    definitions = [
        (REGIONS, "reg; tissue; donor_number", "image_metadata", "image_id; sample_id",
         "One qualified source region per (cohort, reg); sample_id uses cohort + donor_number"),
        (REGIONS, "type", "image_metadata", "region_annotation",
         "Copy the official region type without reclassification"),
        (REGIONS, "patient_number", "sample_metadata", "patient_key",
         "Namespace official patient identifier by cohort; preserve source_subject_id"),
        (CLINICAL, "donor_number; patient_number", "sample_metadata", "source_sample_id; source_subject_id",
         "Verify exact identity against region CSV; copy source identifiers"),
        (CLINICAL, "region; diagnostic_subtype; NEOAdjuvant_chemo_or_radiation; grade; HER2; ER; PR; ki67 %",
         "sample_metadata", "clinical_region; diagnostic_subtype; neoadjuvant_treatment; grade; her2_status; er_status; pr_status; ki67_percent",
         "Copy available clinical values as strings; unknown values remain empty"),
        (META / "native_ordered_markers.json", "ordered_labels", "panel_mapping", "source_name",
         "Preserve native flattened ImageJ czt order; S1 maps admitted measurements"),
        (PROOFS, "native_proofs", "image_metadata", "source_revision",
         "Require qualified source size and mtime; first 79 BC or 76 CRC planes are contiguous"),
    ]
    for source, source_field, target_table, target_field, rule in definitions:
        notes = (json.dumps({"data_type": "string", "definition":
                             "Study-local patient key namespaced by cohort from the official region patient number."})
                 if target_table == "sample_metadata" and target_field == "patient_key" else
                 "48 qualified regions; 48 region-to-sample links; 48 sample-to-patient links. "
                 "Clinical values originate from the official CODEX release."
                 if source in (REGIONS, CLINICAL) else "")
        rows.append({**common, "source_file": relative(source), "source_field": source_field,
                     "target_table": target_table, "target_field": target_field,
                     "transformation_rule": rule, "notes": notes})
    sample_fields = [
        (REGIONS, "donor_number; tissue", "sample_id", "string",
         "Study-local sample key from region donor number, namespaced by mapped cohort.", ""),
        (REGIONS, "patient_number; tissue", "patient_key", "string",
         "Study-local patient key from region patient number, namespaced by mapped cohort; cross-cohort identity is unverified.", ""),
        (CLINICAL, "donor_number", "source_sample_id", "string",
         "Official donor number, checked against the matched region row.", ""),
        (CLINICAL, "patient_number", "source_subject_id", "string",
         "Official patient number, checked against the matched region row.", ""),
        (REGIONS, "reg", "sample_type", "string",
         "Each qualified source row is one imaged tissue region; assigned the constant tissue region.", ""),
        (CLINICAL, "tissue", "anatomical_site", "string",
         "Verbatim source tissue label for the matched region, including ileum and duodenum.", ""),
        (REGIONS, "type", "region_type", "string",
         "Verbatim tissue-region type from the official region table.", ""),
        (CLINICAL, "region", "clinical_region", "string",
         "Verbatim clinical workbook region designation.", ""),
        (CLINICAL, "diagnostic_subtype", "diagnostic_subtype", "string",
         "Verbatim source diagnostic subtype; UNKNOWN and blank are missing values.", ""),
        (CLINICAL, "NEOAdjuvant_chemo_or_radiation", "neoadjuvant_treatment", "string",
         "Verbatim source neoadjuvant chemotherapy or radiation field; collection timing is unspecified.", "source timing unspecified"),
        (CLINICAL, "grade", "grade", "string",
         "Verbatim source grade; grading system and assessment time are unspecified.", "source timing unspecified"),
        (CLINICAL, "HER2", "her2_status", "string",
         "Verbatim source HER2 category; assay and assessment time are unspecified.", "source timing unspecified"),
        (CLINICAL, "ER", "er_status", "string",
         "Verbatim source ER category; assay and assessment time are unspecified.", "source timing unspecified"),
        (CLINICAL, "PR", "pr_status", "string",
         "Verbatim source PR category; assay and assessment time are unspecified.", "source timing unspecified"),
        (CLINICAL, "ki67 %", "ki67_percent", "number",
         "Source Ki-67 percentage; assay and assessment time are unspecified.", "source timing unspecified"),
    ]
    for source, source_field, target, kind, definition, origin in sample_fields:
        notes = {"data_type": kind, "definition": definition}
        if target == "ki67_percent":
            notes["unit"] = "%"
        if origin:
            notes["time_origin"] = origin
        rows.append({**common, "source_file": relative(source), "source_field": source_field,
                     "target_table": "sample_metadata", "target_field": target,
                     "value_mapping": ('{"UNKNOWN": null}' if target == "diagnostic_subtype" else "{}"),
                     "transformation_rule": ("Use cohort + reg; preserve source value except UNKNOWN -> blank"
                                             if target == "diagnostic_subtype" else
                                             "Use cohort + reg; preserve the stated source value or key rule"),
                     "time_origin_rule": origin, "notes": json.dumps(notes)})
    return pd.DataFrame(rows, columns=columns)


def stage_selected_planes(source: Path, staged: Path, panel: pd.DataFrame, cohort: str) -> None:
    """Copy retained native planes in large sequential reads; excluded planes stay sparse."""
    available = 79 if cohort == "BC" else 76
    indices = panel.loc[panel.keep_channel, "source_channel_index"].astype(int).tolist()
    if any(index >= available for index in indices):
        raise ValueError(f"Selected plane exceeds qualified source: {cohort}")
    with source.open("rb") as input_file, staged.open("wb") as output_file:
        output_file.truncate(PAYLOAD_START + available * PLANE_BYTES)
        for index in indices:
            offset = PAYLOAD_START + index * PLANE_BYTES
            input_file.seek(offset)
            output_file.seek(offset)
            remaining = PLANE_BYTES
            while remaining:
                chunk = input_file.read(min(8 * 1024**2, remaining))
                if not chunk:
                    raise ValueError(f"Native plane {index} ended early: {source}")
                output_file.write(chunk)
                remaining -= len(chunk)


class NativeReader:
    """Read physical contiguous ImageJ planes, including malformed-directory BC files."""

    def __init__(self, path: Path, cohort: str, staged_source: Path):
        self.source = SimpleNamespace(path=path, pixel_size_um=None)
        self.shape = (80 if cohort == "BC" else 76, HEIGHT, WIDTH)
        self.dtype = np.dtype("uint16")
        available = 79 if cohort == "BC" else 76
        self.pixels = np.memmap(staged_source, mode="r", dtype=">u2", offset=PAYLOAD_START,
                                shape=(available, HEIGHT, WIDTH))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.stack([self.pixels[channel, y0:y1, x0:x1] for channel in channels]).astype("uint16")

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


class FastTiffWriter(tifffile.TiffWriter):
    def write(self, *args: object, **kwargs: object) -> object:
        if kwargs.get("compression") == "deflate":
            kwargs["compression"] = "zstd"
            kwargs["compressionargs"] = {"level": 1}
        return super().write(*args, **kwargs)


def image_row(item: dict, sample: pd.Series, panel: pd.DataFrame, result: dict) -> dict:
    cohort, reg, path, proof = item["cohort"], item["reg"], item["source"], item["proof"]
    side = max(HEIGHT, WIDTH)
    return {
        "study": STUDY, "image_id": f"{cohort}_{reg}", "sample_id": sample.sample_id,
        "panel_id": cohort, "image_kind": "roi", "source_image_id": f"{cohort}/{path.name}",
        "source_files": [relative(path)], "source_format": "imagej_tiff_contiguous_planes",
        "source_acquisition_id": f"{cohort}/{path.name}", "source_axes": "ZCYX",
        "source_dtype": "uint16", "source_revision": f"official_gcs:{cohort};size={proof['bytes']};mtime_ns={proof['mtime_ns']}",
        "roi_id": reg, "region_annotation": sample.region_type,
        "source_x0_px": 0, "source_y0_px": 0, "source_x1_px": WIDTH, "source_y1_px": HEIGHT,
        "pad_left_px": (side-WIDTH)//2, "pad_right_px": side-WIDTH-(side-WIDTH)//2,
        "pad_top_px": (side-HEIGHT)//2, "pad_bottom_px": side-HEIGHT-(side-HEIGHT)//2,
        "height_px": side, "width_px": side, "source_channel_count": len(panel),
        "channel_count": int(panel.keep_channel.sum()),
        "input_image_path": relative(OUTPUT / "images" / f"{cohort}_{reg}" / "input.tiff"),
        "valid_pixels_path": None,
        "intensity_description": "Original unsigned 16-bit contiguous ImageJ plane intensities; lossless TIFF, no scaling",
        "split_method": "source_acquisition",
        "split_evidence": f"Official tissue-region row {cohort}/{reg}; one region per source acquisition",
        "metadata_notes": ("BC trailing DRAQ5 is unavailable and excluded by source-bound proof; "
                           "physical scale absent in supplied TIFF header" if cohort == "BC" else
                           "Physical scale absent in supplied TIFF header"),
    }


def verify_output(path: Path, panel: pd.DataFrame, *, read_lease_held: bool = False) -> None:
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    lease = nullcontext() if read_lease_held else bounded_read(path)
    with lease, tifffile.TiffFile(path) as tif:
        if (tif.series[0].shape != (len(kept), WIDTH, WIDTH) or
                tif.pages[0].dtype != np.dtype("uint16") or len(tif.pages) != len(kept)):
            raise ValueError(f"Output TIFF header mismatch: {path}")
        names = [channel.attrib["Name"] for channel in
                 ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
        if names != kept.marker_id.tolist():
            raise ValueError(f"Output channel names mismatch: {path}")


def run(max_images: int | None) -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    nas_available = shutil.disk_usage(ROOT).free
    local_available = shutil.disk_usage(TEMP).free
    if nas_available - MAX_OUTPUT_BYTES < 100 * 1024**3 or local_available - 4 * 1024**3 < 64 * 1024**3:
        checkpoint("waiting_io", waiting_reason="NAS or local staging space below reserved threshold",
                   nas_free_bytes=nas_available, local_free_bytes=local_available,
                   estimated_max_output_bytes=MAX_OUTPUT_BYTES)
        return
    stage_official_metadata()
    scope, labels = source_scope()
    metadata = source_metadata(scope)
    panel = build_panels(labels)
    samples = sample_rows(metadata)
    fields = field_rows()
    if not (len(samples) == 48 and samples.sample_id.is_unique):
        raise ValueError("Sample identity is not one-to-one with qualified region")
    with bounded_read(OUTPUT):
        completed = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
    known = set(completed.image_id) if len(completed) else set()
    existing_unindexed = []
    for item in scope:
        image_id = f"{item['cohort']}_{item['reg']}"
        if image_id not in known and (OUTPUT / "images" / image_id / "input.tiff").exists():
            existing_unindexed.append(item)
    CURRENT.update(resume_image_index=len(known)+1,
                   existing_unindexed_files=len(existing_unindexed))
    if existing_unindexed:
        with bounded_read(OUTPUT):
            for item in existing_unindexed:
                image_id = f"{item['cohort']}_{item['reg']}"
                verify_output(OUTPUT / "images" / image_id / "input.tiff",
                              panel[panel.panel_id == item["cohort"]], read_lease_held=True)
    verified_existing = {f"{item['cohort']}_{item['reg']}" for item in existing_unindexed}
    produced = 0
    pending_rows = []
    last_index = 0
    last_image = ""
    for index, item in enumerate(scope, start=1):
        cohort, reg, source = item["cohort"], item["reg"], item["source"]
        image_id = f"{cohort}_{reg}"
        CURRENT.update(attempted_image_index=index, attempted_image_id=image_id)
        destination = OUTPUT / "images" / image_id / "input.tiff"
        current_panel = panel[panel.panel_id == cohort]
        sample = samples.loc[samples.sample_id == f"{cohort}_donor_{metadata.loc[(cohort, reg), 'donor_number_region']}"].iloc[0]
        if image_id in known and destination.exists():
            continue
        if max_images is not None and produced >= max_images:
            break
        if not destination.exists():
            local_directory = TEMP / image_id
            local_directory.mkdir(parents=True, exist_ok=True)
            local_source = local_directory / "selected_planes.bin"
            local = local_directory / "input.tiff"
            if local.exists():
                result = {"admitted": True}
            else:
                with bounded_read(source):
                    stage_selected_planes(source, local_source, current_panel, cohort)
                available = 79 if cohort == "BC" else 76
                if local_source.stat().st_size != PAYLOAD_START + available * PLANE_BYTES:
                    raise ValueError(f"Local source staging incomplete: {image_id}")
                reader = NativeReader(source, cohort, local_source)
                with patch("tifffile.TiffWriter", FastTiffWriter):
                    result = _export_input(reader, current_panel, local)
                del reader
                local_source.unlink()
            if not result["admitted"]:
                raise ValueError(f"No nonzero admitted marker in qualified source: {image_id}")
            verify_output(local, current_panel)
            publish_file(local, destination)
            local.unlink()
            CURRENT.update(last_published_image=image_id, last_published_index=index,
                           existing_unindexed_files=index-int(CURRENT.get("indexed_images", len(known))))
        else:
            result = {"admitted": True}
            if image_id not in verified_existing:
                raise ValueError(f"Existing file was not preverified: {image_id}")
            CURRENT.update(last_published_image=image_id, last_published_index=index,
                           existing_unindexed_files=index-int(CURRENT.get("indexed_images", len(known))))
        row = image_row(item, sample, current_panel, result)
        pending_rows.append(row)
        produced += 1
        last_index, last_image = index, image_id
        if len(pending_rows) == 4:
            with acquisition_timeout("metadata publication lease"):
                save_input_tables(OUTPUT, pd.DataFrame(pending_rows), samples, panel, fields)
            pending_rows.clear()
            CURRENT.update(resume_image_index=last_index+1, existing_unindexed_files=0,
                           indexed_images=last_index)
            checkpoint("running", next_image_index=last_index, last_completed_image=last_image,
                       source_scope_images=48, estimated_max_output_bytes=MAX_OUTPUT_BYTES)
    if pending_rows:
        with acquisition_timeout("metadata publication lease"):
            save_input_tables(OUTPUT, pd.DataFrame(pending_rows), samples, panel, fields)
        CURRENT.update(resume_image_index=last_index+1, existing_unindexed_files=0,
                       indexed_images=last_index)
        checkpoint("running", next_image_index=last_index, last_completed_image=last_image,
                   source_scope_images=48, estimated_max_output_bytes=MAX_OUTPUT_BYTES)
    with bounded_read(OUTPUT):
        final_count = len(pd.read_parquet(OUTPUT / "images.parquet"))
    if final_count < 48:
        checkpoint("batch_complete", next_image_index=last_index or len(known))
        return
    review = OUTPUT / "review_results.csv"
    if not review.exists():
        staged = TEMP / "review_results.csv"
        pd.DataFrame(columns=pd.read_csv(TEMPLATES / "review_results.csv").columns).to_csv(
            staged, index=False, encoding="utf-8-sig")
        publish_file(staged, review)
    checkpoint("prepared", next_image_index=48, independent_review_pending=True,
               source_scope_images=48, estimated_max_output_bytes=MAX_OUTPUT_BYTES)


def repair_clinical() -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    nas_available = shutil.disk_usage(ROOT).free
    local_available = shutil.disk_usage(TEMP).free
    if nas_available - 4 * 1024**2 < 100 * 1024**3 or local_available - 4 * 1024**2 < 64 * 1024**3:
        checkpoint("waiting_io", phase="repair", scientific_repair_round=1,
                   waiting_reason="NAS or local staging space below reserved threshold",
                   nas_free_bytes=nas_available, local_free_bytes=local_available,
                   resume="Retry --repair-clinical after space is available")
        return
    with bounded_read(SOURCE):
        scope, _ = source_scope()
        metadata = source_metadata(scope)
    samples = sample_rows(metadata)
    fields = field_rows()
    with bounded_read(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    if (len(images) != 48 or len(samples) != 48 or
            set(images.sample_id) != set(samples.sample_id)):
        raise ValueError("Existing image and sample scope no longer matches 48 qualified regions")
    with acquisition_timeout("clinical metadata publication lease"):
        save_input_tables(OUTPUT, images, samples, panel, fields)
    CURRENT.update(resume_image_index=49, clinical_unknown_values_cleared=6)
    checkpoint("repair_complete", phase="repair", scientific_repair_round=1,
               resume_image_index=49, affected_scope="sample_metadata and field_mapping",
               independent_review_pending=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repair-clinical", action="store_true")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    if args.run == args.repair_clinical:
        parser.error("Specify exactly one of --run and --repair-clinical")
    try:
        if args.repair_clinical:
            repair_clinical()
        else:
            run(args.max_images)
    except WaitingIO as error:
        checkpoint("waiting_io", from_local=(TEMP / "progress.json").exists(),
                   waiting_reason=str(error), **CURRENT,
                   phase="repair" if args.repair_clinical else "prepare",
                   scientific_repair_round=1 if args.repair_clinical else 0,
                   resume="Run the same command; indexed rows and published files are reused")


if __name__ == "__main__":
    main()
