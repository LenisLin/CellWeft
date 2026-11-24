"""Prepare all eleven source-verified thymus CODEX regions.

The source ZIPs contain one scalar TIFF per cycle/channel. Each image is
extracted to local scratch, exported with the shared input helper, and then
indexed before the next source is opened. Independent review follows export.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)


STUDY = "Stankiewicz2023_CODEX_Thymus"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_11585722"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PAPER_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12643260/"
AUTHOR_CODE = NAS / "raw/public" / STUDY / "metadata/gitlab_thymus_integration.Rmd"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0471_prepare_0")))
PROGRESS = JOB / "progress.json"
GIB = 1024 ** 3


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def save_json(path: Path, value: dict) -> None:
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)


def image_rows() -> pd.DataFrame:
    path = OUTPUT / "images.parquet"
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=columns("image_metadata"))


def progress(stage: str, *, current: str = "", issue: str = "") -> None:
    images = image_rows()
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").is_file() else pd.DataFrame()
    save_json(PROGRESS, dict(
        study=STUDY, phase="repair", repair_round=2, stage=stage,
        expected_images=11, exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
        current_image=current, issue=issue,
        next_image=next((item["image_id"] for item in source_images()
                         if item["image_id"] not in set(images.image_id)), ""),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    ))


def source_images() -> list[dict]:
    proof = json.loads((META / "native_verification.json").read_text())
    native = proof["native_result"]
    if not proof["scope_complete"] or native["accepted_physical_image_count"] != 7:
        raise ValueError("Seven complete physical source images are required")
    images = [dict(item, scope_id=item["representative_region_id"].split(":", 1)[0])
              for item in native["images"]]
    if len(images) != 7 or any(not item["complete"] or not item["counted"]
                               or not item["native_order_verified"] for item in images):
        raise ValueError("Native source evidence is incomplete")
    selected = {item["representative_region_id"] for item in images}
    additions = []
    for proof in native["region_proofs"]:
        if proof["region_id"] in selected:
            continue
        if not proof["complete"] or not proof["native_order_verified"] or not proof["payload_bounds_complete"]:
            raise ValueError(f"Incomplete source region: {proof['region_id']}")
        item = dict(proof)
        item["representative_region_id"] = proof["region_id"]
        suffix = ("T087_architecture_region1" if proof["scope_id"] == "T087_architecture" else
                  proof["scope_id"])
        item["image_id"] = f"{STUDY}:{suffix}"
        additions.append(item)
    if len(additions) != 4:
        raise ValueError("Expected four additional complete source regions")
    return images + additions


def write_source_scope(items: list[dict]) -> None:
    source = json.loads((META / "ordered_markers_and_source_members.json").read_text())
    selected = {item["representative_region_id"] for item in items}
    native = json.loads((META / "native_verification.json").read_text())["native_result"]
    representatives = {item["representative_region_id"] for item in native["images"]}
    regions = []
    for container in source["source_containers"]:
        for region in container["measured_regions"]:
            region_id = f"{container['scope_id']}:{region['region_group']}"
            chosen = region_id in selected
            regions.append(dict(
                region_id=region_id, source_archive=relative(RAW / container["archive_key"]),
                source_channel_count=len(region["ordered_file_marker_bindings"]),
                representative_input=region_id in representatives,
                selected_input=chosen,
                disposition="independent source region retained as one image" if chosen else
                            "unresolved source region",
            ))
    if (len(regions) != 11 or sum(row["representative_input"] for row in regions) != 7
            or sum(row["selected_input"] for row in regions) != 11):
        raise ValueError("The source-region account differs from the native proof")
    path = CACHE / "source_scope.json"
    save_json(path, dict(study=STUDY, source_archive_count=10,
                         measured_region_count=11, representative_input_count=7,
                         selected_input_count=11,
                         regions=regions,
                         source_evidence=relative(META / "ordered_markers_and_source_members.json")))


def source_revision(item: dict) -> str:
    stat = (RAW / item["source_file_name"]).stat()
    proof_stat = item["source_file_stat"]
    if stat.st_size != proof_stat["size"] or stat.st_mtime_ns != proof_stat["mtime_ns"]:
        raise ValueError(f"Source archive changed: {item['source_file_name']}")
    return f"zenodo_11585722:{stat.st_size}:{stat.st_mtime_ns}"


def panel_id(item: dict) -> str:
    if item["image_id"].endswith(":T087_architecture_region1"):
        return "architecture_T087_region1"
    if item["scope_id"].endswith(("_treg", "_ecm")):
        return f"{item['scope_id'].split('_', 1)[1]}_{item['sample_id']}"
    return f"architecture_{item['sample_id']}"


def panel_tables(items: list[dict]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panels = {}
    for item in items:
        names = item["channels"]
        if len(names) not in (76, 80) or len(item["source_member_bindings"]) != len(names):
            raise ValueError(f"Unexpected channel count: {item['image_id']}")
        selections = None
        if item["sample_id"] == "T100":
            if item["scope_id"] == "T100_architecture":
                selections = set(range(76)) - {13, 59}
        if item["scope_id"] == "T097_ecm":
            selections = set(range(80)) - {17}
        panel = panel_from_s1(STUDY, panel_id(item), names, S1,
                              downstream_source_indices=selections)
        if selections is not None:
            if item["scope_id"] == "T100_architecture":
                panel.loc[panel.source_channel_index == 13, "exclusion_reason"] = "Repeated CD45RA; select cycle 7 (index 25), higher native P99 and nonzero coverage"
                panel.loc[panel.source_channel_index == 59, "exclusion_reason"] = "Repeated DLL4; select cycle 17 (index 66), higher native P99 and nonzero coverage"
            else:
                panel.loc[panel.source_channel_index == 17, "exclusion_reason"] = "Repeated CD4; select CD4 (Laura), index 55, higher native P99 and nonzero coverage"
        panels[item["image_id"]] = panel
    return panels, pd.concat(panels.values(), ignore_index=True)


def sample_and_field_tables(items: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = AUTHOR_CODE.read_text()
    samples = []
    for item in {item["sample_id"]: item for item in items}.values():
        label = item["sample_id"]
        if f'patient[sample_name == ' not in code or f'<- "{label}"' not in code:
            raise ValueError(f"No author patient-label evidence for {label}")
        sample = dict.fromkeys(columns("sample_metadata"), "")
        sample.update(study=STUDY, sample_id=label, patient_key=label,
                      source_subject_id=label, source_sample_id=label,
                      cohort_id="published_postnatal_thymus",
                      sample_type="published thymus tissue section",
                      tissue_type="normal", anatomical_site="thymus",
                      metadata_notes="Author code declares this T label as a patient label. "
                      "The paper's six age-matched CODEX donors are a subset; "
                      "an exact seven-label-to-six-donor subset table is not established here.")
        samples.append(sample)
    rows = []

    def mapping(source: Path | str, table: str, field: str, rule: str,
                *, source_field: str, sheet: str = "", join: str = "") -> None:
        row = dict.fromkeys(columns("field_mapping"), "")
        row.update(study=STUDY, source_file=relative(source) if isinstance(source, Path) else source, source_sheet_or_table=sheet,
                   source_field=source_field, target_table=table, target_field=field,
                   join_keys=join, transformation_rule=rule)
        rows.append(row)

    mapping(META / "native_verification.json", "image_metadata", "sample_id",
            "Use native proof physical_identity/sample_id; seven representative images link to seven source tissues.",
            source_field="native_result.images[].sample_id", join="sample_id")
    mapping(AUTHOR_CODE, "sample_metadata", "patient_key",
            "Keep the explicit T patient label in author integration code; unique biological donor count beyond the labels is not independently inferred.",
            source_field="patient[sample_name]", sheet="Integration and Pre-Processing", join="source T label")
    mapping(AUTHOR_CODE, "sample_metadata", "source_subject_id",
            "Copy the author's explicit T patient label for the matching source tissue; do not infer membership in the paper's six-donor CODEX subset.",
            source_field="patient[sample_name]", sheet="Integration and Pre-Processing", join="source T label")
    mapping(META / "source_manifest.json", "sample_metadata", "source_sample_id",
            "Use source T tissue label as the sample identifier; alternate panels share this sample.",
            source_field="patient_evidence.public_source_patient_labels", join="source T label")
    mapping(PAPER_SOURCE, "sample_metadata", "tissue_type",
            "Methods identify the released representative CODEX tissue as normal postnatal thymus; no donor-specific clinical attributes inferred.",
            source_field="study design and data availability", sheet="Methods")
    mapping(PAPER_SOURCE, "sample_metadata", "anatomical_site",
            "The published tissue is thymus; this is a specimen site, not a patient diagnosis.",
            source_field="study design and data availability", sheet="Methods")
    mapping(PAPER_SOURCE, "sample_metadata", "sample_type",
            "Released CODEX regions are thymus tissue sections.",
            source_field="study design and data availability", sheet="Methods")
    mapping(META / "source_manifest.json", "sample_metadata", "cohort_id",
            "Group the published postnatal thymus source tissues; no donor-specific diagnosis is implied.",
            source_field="disease; tissue", join="source T label")
    mapping(META / "ordered_markers_and_source_members.json", "image_metadata", "source_image_id",
            "Keep the full archive-internal region prefix; each cycle/channel TIFF binds by native marker_index_0based.",
            source_field="source_containers[].measured_regions[].region_group", join="sample_id; panel")
    mapping(META / "ordered_markers_and_source_members.json", "panel_mapping", "source_channel_index",
            "Use official experiment channelNamesArray and ordered archive members; join exact Study + Original marker to S1.",
            source_field="ordered_markers; ordered_file_marker_bindings", join="scope_id; marker_index_0based")
    rows[-1]["source_file"] = relative(S1)
    rows[-1]["source_sheet_or_table"] = f"S1[Study={STUDY}] and native ordered marker manifest"
    mapping(META / "official_native_config/T100_architecture.experiment.json", "panel_mapping", "keep_channel",
            "For T100 repeated CD45RA retain index 25 over 13 (native P99 7821 vs 6103; nonzero 2082502 vs 2018304); "
            "repeated DLL4 retain index 66 over 59 (P99 19120 vs 7554; nonzero 3756273 vs 3477181). "
            "CD45RA exposures match at 100; DLL4 exposures are 200 at index 59 and 165 at index 66. "
            "Selection uses measured signal and coverage, not equal DLL4 exposure; alternative raw rows remain in panel mapping.",
            source_field="channelNamesArray; exposureTimesArray", sheet="T100 architecture experiment")
    mapping(META / "official_native_config/T097_ecm.experiment.json", "panel_mapping", "keep_channel",
            "For repeated CD4 retain CD4 (Laura), index 55, over index 17: native P99 3606 vs 1651 "
            "and nonzero pixels 9293978 vs 7202907; keep both source rows in panel mapping.",
            source_field="channelNamesArray; ordered source TIFFs", sheet="T097 ECM experiment")
    mapping(META / "source_manifest.json", "sample_metadata", "metadata_notes",
            "Reviewed 10 local Zenodo ZIPs, seven representative native proofs, source code and paper XML. "
            "Eleven image/sample and seven source-labeled patient connections; paper reports six age-matched CODEX donors, "
            "but the exact subset crosswalk and further donor clinical values remain unresolved.",
            source_field="patient_evidence; source_evidence", join="source T label")
    return pd.DataFrame(samples, columns=columns("sample_metadata")), pd.DataFrame(rows, columns=columns("field_mapping"))


class LocalPlanes:
    """Expose staged retained scalar TIFFs under the full source channel indices."""

    def __init__(self, files: dict[int, Path], channel_count: int, stack: ExitStack):
        self.tiffs = {index: stack.enter_context(tifffile.TiffFile(path))
                      for index, path in files.items()}
        pages = [tif.pages[0] for tif in self.tiffs.values()]
        if not pages or any(page.shape != pages[0].shape or page.dtype != pages[0].dtype
                            or page.ndim != 2 for page in pages):
            raise ValueError("Staged scalar source TIFFs disagree on shape or dtype")
        self.shape = (channel_count, *pages[0].shape)
        self.dtype = pages[0].dtype
        self.source = SimpleNamespace(path=next(iter(files.values())), pixel_size_um=None)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        return np.stack([read_tiff_window(self.tiffs[index], 0, window) for index in channels])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def stage_planes(item: dict, panel: pd.DataFrame, directory: Path) -> dict[int, Path]:
    archive = RAW / item["source_file_name"]
    bindings = item["source_member_bindings"]
    keep = panel.loc[panel.keep_channel, "source_channel_index"].astype(int).tolist()
    staged = {}
    with formal_read_lease(archive):
        with zipfile.ZipFile(archive) as source:
            for index in keep:
                binding = bindings[index]
                if binding["marker_index_0based"] != index or binding["marker"] != item["channels"][index]:
                    raise ValueError(f"Source member order mismatch at channel {index}")
                info = source.getinfo(binding["member"])
                if info.file_size != binding["bytes"]:
                    raise ValueError(f"Source member size mismatch at channel {index}")
                path = directory / f"channel_{index:03d}.tif"
                with source.open(info) as inp, path.open("wb") as out:
                    shutil.copyfileobj(inp, out, length=1024 * 1024)
                staged[index] = path
    return staged


def image_record(item: dict, result: dict, revision: str, destination: Path) -> dict:
    h = result["height_px"] - result["pad_top_px"] - result["pad_bottom_px"]
    w = result["width_px"] - result["pad_left_px"] - result["pad_right_px"]
    row = dict.fromkeys(columns("image_metadata"), "")
    row.update(study=STUDY, image_id=item["image_id"], sample_id=item["sample_id"],
               panel_id=panel_id(item), image_kind="roi", source_image_id=item["representative_region_id"],
               source_files=[relative(RAW / item["source_file_name"])],
               source_format="zip_tiff_planes", source_z_index=0, source_axes="channel files: YX; z00",
               source_dtype=result["source_dtype"], source_revision=revision,
               roi_id=item["representative_region_id"].split(":", 1)[1],
               source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=result["height_px"], width_px=result["width_px"],
               source_channel_count=result["source_channel_count"], channel_count=result["channel_count"],
               input_image_path=relative(destination),
               intensity_description="Published stitched CODEX uint16 intensities with author correction; no CellWeft normalization",
               split_method="already_independent",
               split_evidence="Native region proof and archive-internal stitched-region channel filenames",
               metadata_notes="Complete source region; other regions of the same tissue have separate image IDs. "
               "Pixel-size unit not independently confirmed from the experiment config.")
    if result["valid_pixels"]:
        row["valid_pixels_path"] = relative(destination.with_name(result["valid_pixels"]))
    return row


def verify_headers(images: pd.DataFrame, panel: pd.DataFrame) -> None:
    for row in images.itertuples():
        selected = panel[panel.panel_id == row.panel_id]
        names = selected.loc[selected.keep_channel].sort_values("output_channel_index").marker_id.tolist()
        with formal_read_lease(NAS / row.input_image_path):
            with tifffile.TiffFile(NAS / row.input_image_path) as tif:
                if tif.series[0].shape != (len(names), row.height_px, row.width_px) or tif.series[0].dtype != np.dtype("uint16"):
                    raise ValueError(f"Output TIFF shape/dtype mismatch: {row.image_id}")
                pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
                if [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")] != names:
                    raise ValueError(f"Output TIFF channel order mismatch: {row.image_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Export all remaining verified images")
    parser.add_argument("--limit", type=int, help="Limit new exports for a bounded run")
    args = parser.parse_args()
    items = source_images()
    per_image_panel, panel = panel_tables(items)
    samples, fields = sample_and_field_tables(items)
    if not args.run:
        print(json.dumps(dict(expected_images=len(items), source_archives=10,
                              source_regions=11,
                              kept_channels={item["sample_id"]: int(per_image_panel[item["image_id"]].keep_channel.sum())
                                             for item in items})))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    write_source_scope(items)
    scratch = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
    scratch.mkdir(parents=True, exist_ok=True)
    images = image_rows()
    save_input_tables(OUTPUT, images, samples, panel, fields)
    completed = set(images.image_id)
    todo = [item for item in items if item["image_id"] not in completed]
    if args.limit is not None:
        todo = todo[:args.limit]
    progress("ready")
    for item in todo:
        identity = item["image_id"]
        revision = source_revision(item)
        progress("staging", current=identity)
        # The full source archive, local extracted planes and one compressed
        # output coexist during this image; the fixed reserve remains free.
        source_bytes = (RAW / item["source_file_name"]).stat().st_size
        if shutil.disk_usage(NAS).free < 100 * GIB + source_bytes:
            progress("waiting_io", current=identity, issue="NAS 100 GiB reserve")
            return
        if shutil.disk_usage(scratch).free < 64 * GIB + source_bytes * 2:
            progress("waiting_io", current=identity, issue="Local 64 GiB reserve")
            return
        with tempfile.TemporaryDirectory(prefix="source-", dir=scratch) as directory:
            selected_panel = per_image_panel[identity]
            files = stage_planes(item, selected_panel, Path(directory))
            with ExitStack() as stack:
                reader = LocalPlanes(files, len(selected_panel), stack)
                destination = OUTPUT / "images" / identity / "input.tiff"
                result = export_input(reader, selected_panel, destination)
        save_json(CACHE / "image_results" / f"{identity.rsplit(':', 1)[1]}.json",
                  dict(image_id=identity, source_revision=revision, result=result,
                       source_proof=relative(META / "native_verification.json")))
        if result["admitted"]:
            images = pd.concat([images, pd.DataFrame([image_record(item, result, revision, destination)])],
                               ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, fields)
        else:
            save_json(CACHE / "excluded" / f"{identity.rsplit(':', 1)[1]}.json",
                      dict(image_id=identity, reason="No nonzero admitted marker signal", result=result))
        progress("exporting", current=identity)
        print(json.dumps(dict(image_id=identity, admitted=result["admitted"],
                              exported_images=len(images))), flush=True)
    if len(images):
        verify_headers(images, panel)
    progress("prepared_awaiting_independent_review" if len(images) == 11 else "partial_prepared")


if __name__ == "__main__":
    main()
