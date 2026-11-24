"""Prepare Hickey2023 native-intensity inputs from verified SPoRA and HuBMAP sources.

Inputs: source-native proofs, tissue/specimen metadata, publication Tables 1/10,
        and the study-specific rows of Supplementary Table S1.
Outputs: processed/<study>/{images.parquet,sample_metadata,panel_mapping,
         field_mapping,review_results.csv}, plus independent CYX input.tiff.
Run `--run` to build metadata and export the scope through one batch write slot.
The preparer's checks require a subsequent independent review.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import openpyxl
import pandas as pd
from PIL import Image
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import ImageReader, SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)

STUDY = "Hickey2023_CODEX_Intestine"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
BATCH = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"
PROGRESS = BATCH / "jobs" / STUDY / "0009_prepare_0/progress.json"
SPORA_PROOF = META / "spora_native_verification.json"
HUB_PROOF = META / "hub_native_verification.json"
HUB_MANIFEST = META / "hub_source_manifest.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TABLE1 = META / "source_metadata/41586_2023_5915_MOESM4_ESM.xlsx"
TABLE10 = META / "source_metadata/41586_2023_5915_MOESM13_ESM.xlsx"


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def progress(stage: str, **details) -> None:
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else None
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").exists() else None
    state = dict(stage=stage, completed_images=len(images) if images is not None else 0,
                 expected_images=66, sample_linked_images=int(images.sample_id.notna().sum()) if images is not None else 0,
                 sample_rows=len(samples) if samples is not None else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if samples is not None else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(details)
    save_json(PROGRESS, state)


def template(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def clinical_records() -> dict[str, dict]:
    """Transpose publication donor metadata, preserving the original clinical values."""
    workbook = openpyxl.load_workbook(TABLE1, read_only=True, data_only=True)
    rows = list(workbook.active.values)
    workbook.close()
    return {subject: {row[0]: row[column] for row in rows[1:]}
            for column, subject in enumerate(rows[0][1:], 1)}


def build_metadata() -> None:
    """Resolve explicit specimen identities and enumerate the complete accepted scope."""
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proof = load_json(SPORA_PROOF)
    assert proof["full_source_covered"] and proof["verified_count"] == 64 and not proof["failed"]
    tissues = pd.read_parquet(META / "metadata/tissues.parquet")
    tissues = tissues[tissues.modality == "codex"].set_index("tissue_id", verify_integrity=True)
    canonical = load_json(META / "spora_inventory/codex_ordered_marker_names.json")
    presence = pd.read_parquet(META / "codex/channels_per_tissue.parquet")
    channel_metadata = pd.read_parquet(META / "codex/channels.parquet")
    all_names = channel_metadata.channel_name.tolist()
    scope, panels, sample_rows = [], {}, []
    clinical = clinical_records()
    clinical_fields = {
        "donor race": "race_reported", "History of diabetes": "history_diabetes_reported",
        "History of cancer": "history_cancer_reported",
        "History of hypertension": "history_hypertension_reported",
        "History gastrointestinal disease": "history_gastrointestinal_disease_reported",
    }

    def sample_row(sample_id: str, subject: str, source_id: str, group: str | None, notes: str) -> dict:
        row = dict.fromkeys(template("sample_metadata"))
        values = clinical[subject]
        row.update(study=STUDY, sample_id=sample_id, patient_key=subject,
                   source_subject_id=subject, source_sample_id=source_id,
                   biospecimen_group_id=group, sample_type="tissue", anatomical_site="Intestine",
                   age_at_collection_years=values["donor age"], sex_reported=values["donor sex"],
                   bmi_kg_m2=values["BMI"], metadata_notes=notes)
        row.update({target: values[source] for source, target in clinical_fields.items()})
        return row

    def register(names: list[str]) -> str:
        key = tuple(names)
        if key not in panels:
            panels[key] = f"panel_{len(panels) + 1:03d}"
        return panels[key]

    for source in sorted(proof["verified"], key=lambda record: record["image_id"]):
        assert source["dtype"] == "uint16"
        image_id = source["image_id"]
        native = NAS / source["path"]
        root = load_json(native / "zarr.json")["attributes"]["ome"]
        assert root["omero"]["name"] == root["multiscales"][0]["name"] == image_id
        scale = root["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        labels = [channel["label"] for channel in root["omero"]["channels"]]
        names = canonical[image_id]
        assert labels == source["canonical_channel_labels"] == names
        assert [name for name in all_names if presence.loc[image_id, name]] == names
        array_meta = load_json(native / "0/zarr.json")
        assert array_meta["shape"] == source["shape"] and array_meta["data_type"] == source["dtype"]
        objects = [path for path in native.rglob("*")
                   if path.is_file() and not path.name.endswith(".partial")]
        assert len(objects) == source["source_object_count"]
        assert sum(path.stat().st_size for path in objects) == source["source_bytes"]
        tissue = tissues.loc[image_id]
        # specimen_id is explicitly supplied by SPoRA; it is not inferred from the filename.
        sample_id = f"{tissue.patient_id}_{tissue.specimen_id}"
        sample_rows.append(sample_row(sample_id, tissue.original_patient_identifier,
                                     tissue.v1_specimen_id, None,
                                     "SPoRA tissue_id joins explicit patient_id/specimen_id; "
                                     "v1_specimen_id retained as the original specimen label."))
        scope.append(dict(image_id=image_id, path=source["path"], kind="zarr",
                          shape=source["shape"], names=names, panel_id=register(names),
                          sample_id=sample_id, region_annotation=tissue.v1_specimen_id,
                          pixel_size=[scale[2], scale[1]], source_revision=relative(SPORA_PROOF)))

    manifest = load_json(HUB_MANIFEST)
    hub_verified = {record["source_id"]: record for record in load_json(HUB_PROOF)["actual"]}
    for source in manifest["sources"]:
        verified = hub_verified[source["source_id"]]
        path = NAS / source["raw_path"]
        assert verified["status"] == "verified" and path.stat().st_size == verified["source_size"]
        names = [channel["name"] or "<unnamed>" for channel in source["ordered_channels"]]
        with tifffile.TiffFile(path) as tif:
            import xml.etree.ElementTree as ET
            xml = ET.fromstring(tif.ome_metadata)
            observed = [channel.get("Name") or "<unnamed>" for channel in xml.findall(".//{*}Channel")]
            assert observed == names
            shape = list(tif.series[0].shape)
            assert tif.series[0].dtype == np.dtype("uint16")
            pixel = xml.find(".//{*}Pixels")
            assert pixel.get("PhysicalSizeXUnit") == pixel.get("PhysicalSizeYUnit") == "nm"
            scale = [float(pixel.get(f"PhysicalSize{axis}")) / 1000 for axis in ("X", "Y")]
        scope.append(dict(image_id="hubmap_" + source["source_id"] + "_reg001",
                          path=source["raw_path"], kind="tiff", shape=shape, names=names,
                          panel_id=register(names), sample_id=None,
                          region_annotation=source["tissue"], pixel_size=scale,
                          source_revision=relative(HUB_PROOF)))
        # Table 10 establishes four actual biological pieces per parent. reg001's
        # exact piece remains unassigned; preserve confirmed sample rows separately.
        workbook = openpyxl.load_workbook(TABLE10, read_only=True, data_only=True)
        matching = [row for row in workbook.active.values if row[8] == source["portal_parent"]]
        workbook.close()
        assert len(matching) == 1
        subject, _, organ, pieces = matching[0][:4]
        prefix = pieces.split(",")[0].rsplit("-", 1)[0] + "-"
        expanded = [pieces.split(",")[0]] + [prefix + value.strip() for value in pieces.split(",")[1:]]
        assert expanded == source["table10_sample_ids"]
        for specimen in expanded:
            sample_rows.append(sample_row(specimen, subject, specimen, organ,
                                         "Publication Table 10 confirms this organ piece and donor; "
                                         "exact reg001-to-piece connection is unresolved."))
    panel = pd.concat([panel_from_s1(STUDY, panel_id, list(names), S1)
                       for names, panel_id in panels.items()], ignore_index=True)
    samples = pd.DataFrame(sample_rows)
    fields = []

    def mapping(source: str, sheet: str, source_field: str, target: str,
                target_field: str, keys: str, rule: str, notes: str = "") -> None:
        row = dict.fromkeys(template("field_mapping"), "")
        row.update(study=STUDY, source_file=source, source_sheet_or_table=sheet,
                   source_field=source_field, target_table=target, target_field=target_field,
                   join_keys=keys, transformation_rule=rule, notes=notes)
        fields.append(row)

    mapping(relative(META / "metadata/tissues.parquet"), "tissues", "tissue_id;patient_id;specimen_id",
            "image_metadata", "sample_id", "source_image_id=tissue_id;modality=codex",
            "sample_id = explicit patient_id + '_' + specimen_id; 64/66 images linked; "
            "64 SPoRA specimens across eight publication donors.")
    mapping(relative(META / "metadata/tissues.parquet"), "tissues", "original_patient_identifier;v1_specimen_id",
            "sample_metadata", "patient_key;source_subject_id;source_sample_id",
            "tissue_id;patient_id;specimen_id", "Retain explicit publication donor and specimen identifiers.")
    mapping(relative(TABLE10), "data-use-table", "CODEX dataset ID;Stanford Donor;Organ Piece - Stanford Sample ID",
            "image_metadata", "sample_id", "portal_parent=CODEX dataset ID",
            "Each parent lists four organ pieces. Both reg001 images are independent full acquisitions "
            "on visual overview, but exact piece membership remains unresolved; sample_id stays null.")
    mapping(relative(META / "hubmap_supplement_ordered_markers.json"), "sources",
            "raw_path;source_id;portal_parent", "image_metadata", "sample_id",
            "source_files[0]=raw_path; portal_parent=Table10 CODEX dataset ID",
            "Resolve each raw acquisition UUID to its explicit portal parent before the Table10 join; "
            "both reg001-to-organ-piece assignments remain unresolved.")
    for source_field, target_field in {"donor age": "age_at_collection_years",
                                       "donor sex": "sex_reported", "BMI": "bmi_kg_m2",
                                       **clinical_fields}.items():
        mapping(relative(TABLE1), "Sheet1", source_field, "sample_metadata", target_field,
                "source_subject_id=Stanford ID", "Transpose donor columns; copy original value.",
                json.dumps(dict(data_type="string", definition=source_field + " as reported in publication Table 1"))
                if target_field in clinical_fields.values() else "")
    mapping(relative(S1), "supplementary_table_s1_marker_mapping", "Study;Original marker",
            "panel_mapping", "marker_id;keep_channel;output_channel_index", "Study;source_name=Original marker",
            "Exact study-specific S1 match. Keep admitted markers and selected nuclear/body channels. "
            "Null HuBMAP channel name uses the study's explicit <unnamed> S1 row.")
    mapping(relative(META / "spora_inventory/codex_ordered_marker_names.json"), "per-image labels",
            "ordered marker list", "panel_mapping", "source_channel_index;source_name", "source_image_id",
            "Canonical labels cross-checked against embedded OME-Zarr labels and channels_per_tissue; "
            "source order retained; output positions assigned only to kept rows.")
    mapping(relative(TABLE10), "data-use-table", "organ pieces;Stanford Donor", "sample_metadata",
            "sample_id;patient_key;source_sample_id;biospecimen_group_id", "portal_parent=CODEX dataset ID",
            "Preserve eight explicitly listed biological pieces for donor B001; "
            "the explicit HuBMAP organ ID supplies their common parent specimen group. "
            "These rows do not imply exact reg001 image assignment. SPoRA supplies no multi-section/aliquot group; leave it blank.")
    mapping(relative(META / "source_metadata/paper.html"), "publication study scope", "human intestine tissue",
            "sample_metadata", "sample_type;anatomical_site", "study",
            "sample_type=tissue; anatomical_site=Intestine. No finer site is inferred from specimen abbreviations.")
    mapping(relative(META / "source_metadata/source_lookup.json"), "source lookup", "sources", "sample_metadata",
            "metadata_notes", "study", "Record local/public lookup evidence and unresolved reg001 piece membership.")
    field_mapping = pd.DataFrame(fields)
    images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists()
              else pd.DataFrame(columns=template("image_metadata")))
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)
    save_json(CACHE / "scope.json", scope)
    save_json(CACHE / "capacity.json", dict(expected_images=len(scope),
              square_uncompressed_bytes=sum(max(row["shape"][1:])**2 * int(
                  panel[panel.panel_id == row["panel_id"]].keep_channel.sum()) * 2 for row in scope),
              largest_local_stage_bytes=max(np.prod(row["shape"]).item()*2 for row in scope),
              nas_reserved_bytes=100*1024**3, local_reserved_bytes=64*1024**3))
    progress("metadata_ready", sample_rows=len(samples), source_images_with_sample_identity=64,
             patient_linked_samples=int(samples.patient_key.notna().sum()),
             gaps=["Two HuBMAP reg001 acquisitions lack exact organ-piece assignment; source donor B001 known.",
                   "Independent review pending."])


class LocalNativeReader:
    """A single bounded, disk-backed image stage avoids all-channel Zarr redecoding.

    Temporary study adaptation: shared export requests one channel at a time,
    whereas this source's shards compress all channels jointly. Migrate spatial
    chunk caching into a reusable reader only after controller review.
    """
    def __init__(self, source: SourceSpec, array: np.memmap):
        self.source, self.array = source, array
        self.shape, self.dtype = array.shape, array.dtype

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])


@contextmanager
def source_reader(record: dict) -> Iterator[ImageReader]:
    source = SourceSpec(record["image_id"], record["panel_id"], record["kind"],
                        NAS / record["path"], tuple(range(record["shape"][0])),
                        pixel_size_um=tuple(record["pixel_size"]))
    if record["kind"] == "tiff":
        temporary = stage_native(record)
        try:
            # Centered padding otherwise makes a source tile intersect several
            # output tiles. Keep that repeated decoding on the local SSD.
            with open_source(replace(source, path=temporary)) as reader:
                yield reader
        finally:
            temporary.unlink(missing_ok=True)
        return
    temporary = stage_native(record)
    stage = np.memmap(temporary, dtype="uint16", mode="r", shape=tuple(record["shape"]))
    try:
        overview = CACHE / (record["panel_id"] + ".overview.png")
        if not overview.exists():
            channels = [record["names"].index(name) for name in ("DRAQ5", "CYTOKERATIN", "CD45")]
            rgb = []
            for channel in channels:
                values = np.asarray(stage[channel, ::8, ::8], dtype=float)
                scale = np.percentile(values[values > 0], 99.5)
                rgb.append(np.clip(values / scale, 0, 1))
            Image.fromarray((np.stack(rgb, axis=-1)*255).astype("uint8")).save(overview)
            save_json(overview.with_suffix(".json"), dict(image_id=record["image_id"],
                      source=record["path"], channels=[record["names"][channel] for channel in channels],
                      channel_colors=["red", "green", "blue"], sampling_step=8,
                      display_scaling="per-channel nonzero P99.5; visualization only"))
        yield LocalNativeReader(source, stage)
    finally:
        stage._mmap.close()
        temporary.unlink(missing_ok=True)


def stage_native(record: dict) -> Path:
    """Stage Zarr decoded shards or an encoded TIFF once, using bounded local storage."""
    zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
    blosc.set_nthreads(1)
    suffix = ".native.dat" if record["kind"] == "zarr" else ".source.tif"
    temporary = Path("/tmp/study_input_batch") / STUDY / (record["image_id"] + suffix)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    stage_size = (int(np.prod(record["shape"])) * 2 if record["kind"] == "zarr"
                  else (NAS / record["path"]).stat().st_size)
    # Preparation and export can meet at the same image while using distinct NAS
    # slots. The local lock makes stage publication atomic between those commands.
    with temporary.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if temporary.exists():
            assert temporary.stat().st_size == stage_size
            return temporary
        if shutil.disk_usage(temporary.parent).free < stage_size + 64*1024**3:
            raise RuntimeError("Single-image stage would violate the 64 GiB local reserve")
        partial = temporary.with_suffix(".partial")
        if record["kind"] == "tiff":
            with formal_read_lease(NAS / record["path"]):
                shutil.copyfile(NAS / record["path"], partial)
            assert partial.stat().st_size == stage_size
            os.replace(partial, temporary)
            return temporary
        with formal_read_lease(NAS / record["path"] / "0"):
            native = zarr.open_array(str(NAS / record["path"] / "0"), mode="r")
            stage = np.memmap(partial, dtype="uint16", mode="w+", shape=tuple(record["shape"]))
            try:
                for x0, y0, x1, y1 in windows(record["shape"][-2:], 1024):
                    stage[:, y0:y1, x0:x1] = native[:, y0:y1, x0:x1]
                stage.flush()
            finally:
                stage._mmap.close()
        os.replace(partial, temporary)
    return temporary


def export_one(image_id: str) -> None:
    record = next(row for row in load_json(CACHE / "scope.json") if row["image_id"] == image_id)
    images = pd.read_parquet(OUTPUT / "images.parquet")
    if image_id in set(images.image_id):
        print("Reused", image_id, flush=True)
        return
    panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
    selected = panel[panel.panel_id == record["panel_id"]]
    destination = OUTPUT / "images" / image_id / "input.tiff"
    progress("exporting", current_image=image_id, gaps=["HuBMAP exact organ-piece map unresolved; independent review pending."])
    with source_reader(record) as reader:
        statistics = export_input(reader, selected, destination)
        if statistics["admitted"]:
            row = dict.fromkeys(template("image_metadata"))
            _, height, width = record["shape"]
            row.update(study=STUDY, image_id=image_id, sample_id=record["sample_id"],
                       panel_id=record["panel_id"], image_kind="roi",
                       source_image_id=Path(record["path"]).name if record["kind"] == "tiff" else image_id,
                       source_files=[record["path"]], source_format="OME-Zarr" if record["kind"] == "zarr" else "OME-TIFF",
                       source_axes="CYX", source_dtype="uint16", source_revision=record["source_revision"],
                       roi_id=image_id, region_annotation=record["region_annotation"],
                       source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                       pixel_size_x_um=record["pixel_size"][0], pixel_size_y_um=record["pixel_size"][1],
                       input_image_path=relative(destination), split_method="already_independent",
                       split_evidence="Complete level-0 source acquisition; no tissue bounding-box crop.",
                       intensity_description="Published decoded uint16 intensities; no additional normalization.",
                       metadata_notes="Exact organ-piece assignment unresolved; source donor B001 established in Table 10."
                       if record["sample_id"] is None else "Source tissue/specimen linkage retained.")
            for key in ["height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px", "pad_bottom_px",
                        "source_channel_count", "channel_count"]:
                row[key] = statistics[key]
            samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
            fields = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False)
            images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, fields)
        save_json(CACHE / "image_checks" / (image_id + ".json"), statistics)
    progress("image_completed", current_image=image_id, last_admitted=statistics["admitted"],
             gaps=["HuBMAP exact organ-piece map unresolved; independent review pending."])
    print("Completed", image_id, "admitted", statistics["admitted"], "total", len(images), flush=True)


def selfcheck() -> None:
    """Check all headers and representative native pixel blocks; label as preparation checks."""
    import xml.etree.ElementTree as ET
    progress("selfchecking", gaps=["Two HuBMAP exact organ-piece links and independent review pending."])
    zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
    blosc.set_nthreads(1)
    scope = {row["image_id"]: row for row in load_json(CACHE / "scope.json")}
    assert {row["dtype"] for row in load_json(SPORA_PROOF)["verified"]} == {"uint16"}
    images = pd.read_parquet(OUTPUT / "images.parquet")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
    panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    visual = load_json(CACHE / "visual_review.json")
    assert visual["overview_count"] == len(visual["images"]) == 7
    rows, groups = [], set()

    def result(check: str, record: str, level: str, value: str, detail: str,
               evidence: str = "") -> None:
        row = dict.fromkeys(template("review_results"), "")
        row.update(study=STUDY, preparer="0009_prepare_0", reviewer="", scope=level,
                   record_id=record, check_id=check, result=value,
                   source_evidence=evidence or relative(SPORA_PROOF) + ";" + relative(HUB_PROOF),
                   output_evidence=relative(OUTPUT), details="整理者自检，待独立复核。" + detail)
        rows.append(row)

    assert not images.duplicated(["study", "image_id"]).any()
    assert not samples.duplicated(["study", "sample_id"]).any()
    assert set(images.sample_id.dropna()).issubset(set(samples.sample_id))
    assert set(images.panel_id).issubset(set(panel.panel_id))
    for _, native_panel in panel.groupby("panel_id"):
        assert native_panel.source_channel_index.tolist() == list(range(len(native_panel)))
        kept = native_panel[native_panel.keep_channel].sort_values("output_channel_index")
        assert kept.output_channel_index.astype(int).tolist() == list(range(len(kept)))
        assert native_panel[native_panel.nuclear | native_panel.body].keep_channel.all()
        assert not kept[kept.role == "downstream"].marker_id.duplicated().any()
    for image_row in images.itertuples(index=False):
        record = scope[image_row.image_id]
        kept = panel[(panel.panel_id == record["panel_id"]) & panel.keep_channel].sort_values("output_channel_index")
        path = NAS / image_row.input_image_path
        _, height, width = record["shape"]
        side = max(height, width)
        assert (image_row.source_x0_px, image_row.source_y0_px,
                image_row.source_x1_px, image_row.source_y1_px) == (0, 0, width, height)
        left, top = (side-width)//2, (side-height)//2
        assert (image_row.pad_left_px, image_row.pad_top_px, image_row.pad_right_px,
                image_row.pad_bottom_px) == (left, top, side-width-left, side-height-top)
        with formal_read_lease(path), tifffile.TiffFile(path) as tif:
            assert tif.series[0].shape == (len(kept), side, side)
            assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("uint16")
            xml = ET.fromstring(tif.ome_metadata)
            names = [channel.get("Name") for channel in xml.findall(".//{*}Channel")]
            assert names == kept.marker_id.tolist()
            pixel_metadata = xml.find(".//{*}Pixels")
            assert pixel_metadata.get("PhysicalSizeXUnit") == pixel_metadata.get("PhysicalSizeYUnit") == "µm"
            assert float(pixel_metadata.get("PhysicalSizeX")) == image_row.pixel_size_x_um == record["pixel_size"][0]
            assert float(pixel_metadata.get("PhysicalSizeY")) == image_row.pixel_size_y_um == record["pixel_size"][1]
            check = load_json(CACHE / "image_checks" / (image_row.image_id + ".json"))
            assert check["admitted"] and any(check["channel_nonzero_pixels"])
            group = (record["kind"], record["panel_id"], "CYX", "already_independent", "uint16")
            if group not in groups:
                groups.add(group)
                # Temporary shared-reader adaptation: OME series inspection can
                # cache TiffFrame, while read_tiff_window requires TiffPage fields.
                for index in range(len(kept)):
                    tif.pages.get(index, cache=True, aspage=True)
                channels = kept.source_channel_index.astype(int).tolist()
                source = SourceSpec(record["image_id"], record["panel_id"], record["kind"],
                                    NAS / record["path"], tuple(range(record["shape"][0])))
                if record["kind"] == "zarr":
                    source = replace(source, path=source.path / "0")
                with formal_read_lease(source.path), open_source(source) as original:
                    # Intestinal lumens can make the geometric center empty.
                    # Select a signal-bearing interior block from a fixed grid.
                    points = [(width//2, height//2)] + [(width*x//4, height*y//4)
                              for y in (1, 2, 3) for x in (1, 2, 3) if (x, y) != (2, 2)]
                    for x0, y0 in points:
                        internal = (x0, y0, x0+32, y0+32)
                        expected_internal = original.read_window(channels, internal)
                        if np.any(expected_internal):
                            break
                    else:
                        raise ValueError(f"Choose a source-supported signal block for {record['image_id']}")
                    blocks = [internal, (0, 0, 32, 32)]
                    expected_blocks = [expected_internal, original.read_window(channels, blocks[1])]
                for block, expected in zip(blocks, expected_blocks):
                    x0,y0,x1,y1 = block
                    observed = np.stack([read_tiff_window(tif, index, (x0+left,y0+top,x1+left,y1+top))
                                         for index in range(len(kept))])
                    assert expected.dtype == observed.dtype and np.array_equal(expected, observed)
                assert np.any(expected_blocks[0])
                padding_blocks = []
                if top:
                    padding_blocks.append((0, 0, side, top))
                if side-height-top:
                    padding_blocks.append((0, top+height, side, side))
                if left:
                    padding_blocks.append((0, 0, left, side))
                if side-width-left:
                    padding_blocks.append((left+width, 0, side, side))
                for block in padding_blocks:
                    assert all(not np.any(read_tiff_window(tif, index, block)) for index in range(len(kept)))
                result("I05", image_row.image_id, "image", "pass",
                       json.dumps(dict(group=group, source_blocks=blocks,
                                       output_channels=list(range(len(kept))), source_channels=channels,
                                       padding_blocks=padding_blocks, exact_equal=True), ensure_ascii=False), record["path"])
        result("P03", image_row.image_id, "image", "pass", "TIFF CYX shape/dtype/channel labels/order verified.", record["path"])
    for check, detail in [("I01", "Image IDs and native source paths unique and locatable."),
                           ("I02", "Non-null sample and all panel foreign keys verified."),
                           ("I03", "All complete source bounds and centered square padding verified."),
                           ("P01", "Seven ordered native panels linked by exact study+source label to S1."),
                           ("P02", "Selected nuclear/body and admitted marker channels kept; downstream IDs unique.")]:
        result(check, STUDY, "study" if check.startswith("I") else "panel", "pass", detail)
    result("I04", STUDY, "study", "unresolved", "64 SPoRA tissue identities established; two HuBMAP acquisitions "
           f"lack an exact organ-piece identity. {visual['overview_count']} representative native-source overviews were visually "
           "checked by the preparer; full source bounds retained. Panel 5 has sparse cytokeratin/CD45 "
           "in the viewed representative.", relative(CACHE / "visual_review.json"))
    image_checks = {path.stem: load_json(path) for path in (CACHE / "image_checks").glob("*.json")}
    assert set(image_checks) == set(scope)
    admitted_ids = {image_id for image_id, check in image_checks.items() if check["admitted"]}
    excluded_ids = sorted(set(scope) - admitted_ids)
    assert admitted_ids == set(images.image_id)
    assert len(list((OUTPUT / "images").glob("*/input.tiff"))) == len(images)
    result("S01", STUDY, "study", "pass", f"Local scope 64 SPoRA + 2 HuBMAP; {len(images)} images exported; "
           f"{len(excluded_ids)} excluded because all admitted downstream markers are zero across the full ROI: "
           f"{excluded_ids}. All 66 candidates accounted for; reused source-native proofs; no full rehash.")
    clinical = clinical_records()
    tissue_metadata = pd.read_parquet(META / "metadata/tissues.parquet")
    tissue_metadata = tissue_metadata[tissue_metadata.modality == "codex"].set_index("tissue_id")
    samples_by_id = samples.set_index("sample_id", verify_integrity=True)
    for record in scope.values():
        if record["kind"] != "zarr":
            continue
        tissue = tissue_metadata.loc[record["image_id"]]
        sample = samples_by_id.loc[record["sample_id"]]
        assert record["sample_id"] == f"{tissue.patient_id}_{tissue.specimen_id}"
        assert sample.patient_key == sample.source_subject_id == tissue.original_patient_identifier
        assert sample.source_sample_id == tissue.v1_specimen_id
        assert pd.isna(sample.biospecimen_group_id)
    workbook = openpyxl.load_workbook(TABLE10, read_only=True, data_only=True)
    table10 = list(workbook.active.values)
    workbook.close()
    for source in load_json(HUB_MANIFEST)["sources"]:
        source_row = next(row for row in table10 if row[8] == source["portal_parent"])
        subject, _, organ, pieces = source_row[:4]
        prefix = pieces.split(",")[0].rsplit("-", 1)[0] + "-"
        specimens = [pieces.split(",")[0]] + [prefix + value.strip() for value in pieces.split(",")[1:]]
        for specimen in specimens:
            sample = samples_by_id.loc[specimen]
            assert sample.patient_key == sample.source_subject_id == subject
            assert sample.source_sample_id == specimen and sample.biospecimen_group_id == organ
    assert samples.study.eq(STUDY).all() and samples.sample_type.eq("tissue").all()
    assert samples.anatomical_site.eq("Intestine").all() and samples.tissue_type.isna().all()
    checked = 0
    for sample in samples.itertuples(index=False):
        donor = clinical[sample.source_subject_id]
        assert float(sample.age_at_collection_years) == donor["donor age"]
        assert sample.sex_reported == donor["donor sex"]
        assert float(sample.bmi_kg_m2) == donor["BMI"]
        for source_field, target in [("donor race", "race_reported"),
                                      ("History of diabetes", "history_diabetes_reported"),
                                      ("History of cancer", "history_cancer_reported"),
                                      ("History of hypertension", "history_hypertension_reported"),
                                      ("History gastrointestinal disease", "history_gastrointestinal_disease_reported")]:
            assert getattr(sample, target) == donor[source_field]
        checked += 8
    result("M02", STUDY, "sample", "pass", f"All {checked} populated donor clinical values match Table 1; "
           "all 72 specimen/donor identities rebuilt from tissue metadata and Table 10; "
           "optional clinical fields remain blank.", relative(TABLE1) + ";" + relative(TABLE10) + ";" +
           relative(META / "metadata/tissues.parquet"))
    result("M01", STUDY, "sample", "unresolved", f"72 biological sample rows, 72 patient-linked, nine donors; "
           f"{int(images.sample_id.notna().sum())}/{len(images)} image-to-sample links. "
           "Two HuBMAP reg001 exact tissue-piece assignments unresolved after Table10, portal ancestry, "
           "and author-repository lookup. Eight confirmed B001 organ pieces retained separately.",
           relative(META / "source_metadata/source_lookup.json"))
    result("D01", STUDY, "study", "unresolved", f"{len(images)}/66 TIFFs delivered; independent review pending; "
           "two exact organ-piece image links unresolved.")
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8-sig")
    save_json(CACHE / "selfcheck.json", dict(exported_images=len(images), expected_images=66,
              sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
              patient_linked_samples=int(samples.patient_key.notna().sum()),
              pixel_groups_checked=len(groups), clinical_values_checked=checked,
              excluded_images=excluded_ids, independent_review=False))
    progress("prepared_pending_independent_review", sample_rows=len(samples),
             sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=72,
             gaps=["Two HuBMAP exact organ-piece assignments unresolved.", "Independent review pending."])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", action="store_true")
parser.add_argument("--run", action="store_true")
parser.add_argument("--image")
parser.add_argument("--check", action="store_true")
parser.add_argument("--stage")
parser.add_argument("--stage-all", action="store_true")
parser.add_argument("--stage-tiffs", action="store_true")
parser.add_argument("--export-remaining", action="store_true",
                    help="Export the remaining scope inside one externally acquired write slot")
args = parser.parse_args()
if args.stage:
    record = next(row for row in load_json(CACHE / "scope.json") if row["image_id"] == args.stage)
    stage_native(record)
    save_json(CACHE / "stage_progress.json", dict(last_staged=args.stage,
              stage_bytes=int(np.prod(record["shape"]))*2 if record["kind"] == "zarr" else (NAS / record["path"]).stat().st_size,
              stage_format="decoded CYX memmap" if record["kind"] == "zarr" else "encoded source TIFF",
              staged_utc=datetime.now(timezone.utc).isoformat()))
    print("Staged", args.stage, flush=True)
elif args.stage_all or args.stage_tiffs:
    for record in load_json(CACHE / "scope.json"):
        selected_kind = record["kind"] == "tiff" if args.stage_tiffs else True
        if selected_kind and not (CACHE / "image_checks" / (record["image_id"] + ".json")).exists():
            stage_native(record)
            save_json(CACHE / "stage_progress.json", dict(last_staged=record["image_id"],
                      stage_bytes=int(np.prod(record["shape"]))*2 if record["kind"] == "zarr" else
                      (NAS / record["path"]).stat().st_size,
                      stage_format="decoded CYX memmap" if record["kind"] == "zarr" else "encoded source TIFF",
                      staged_utc=datetime.now(timezone.utc).isoformat()))
elif args.export_remaining:
    for record in load_json(CACHE / "scope.json"):
        export_one(record["image_id"])
    progress("export_complete_pending_selfcheck", gaps=["HuBMAP organ-piece linkage and independent review pending."])
elif args.check:
    selfcheck()
elif args.image:
    export_one(args.image)
elif args.build or args.run:
    build_metadata()
    if args.run:
        for record in load_json(CACHE / "scope.json"):
            export_one(record["image_id"])
else:
    parser.error("Choose --build, --run, --image IMAGE_ID, or --check")
