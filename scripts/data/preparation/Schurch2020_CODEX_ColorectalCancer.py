"""Prepare native CODEX input images from the verified Schurch SPoRA release.

Inputs: verified level-0 OME-Zarr, tissue metadata, TCIA clinical attachments,
        ordered source channels, and the study-specific Supplementary Table S1.
Outputs: one CYX input.tiff per admitted ROI and the study input tables.
The preparer records checks; independent review follows after the files close.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import openpyxl
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Schurch2020_CODEX_ColorectalCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY) / 'spora_inventory'
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_JOB",
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0574_prepare_0",
))
PROOF = NAS / "raw/public" / STUDY / "manifest.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
RESERVE_NAS = 100 * 1024**3
RESERVE_LOCAL = 64 * 1024**3


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    with formal_publish_lease(path, directory=path.parent):
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(content)
        os.replace(temporary, path)


def progress(stage: str, **detail: object) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else None
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").is_file() else None
    state = dict(study=STUDY, stage=stage, expected_images=251,
                 exported_images=len(images) if images is not None else 0,
                 sample_linked_images=int(images.sample_id.notna().sum()) if images is not None else 0,
                 sample_rows=len(samples) if samples is not None else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if samples is not None else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(detail)
    write_json(JOB / "progress.json", state)


def clinical_sources() -> tuple[dict[int, dict], dict[int, dict]]:
    """Read source clinical sheets without changing their patient/region codes."""
    patient_file = NAS / "raw/public" / STUDY / "metadata/CRC_TMAs_patient_annotations.xlsx"
    tumor_file = NAS / "raw/public" / STUDY / "metadata/Multi-tumor_TMA_composition.xlsx"
    patient_book = openpyxl.load_workbook(patient_file, read_only=True, data_only=True)
    patient_rows = list(patient_book.active.values)
    patient_book.close()
    patient = {int(row[0]): dict(zip(patient_rows[0], row)) for row in patient_rows[1:]
               if isinstance(row[0], (int, float))}
    tumor_book = openpyxl.load_workbook(tumor_file, read_only=True, data_only=True)
    tumor_rows = list(tumor_book.active.values)
    tumor_book.close()
    tumor = {int(str(row[4])[3:]): dict(zip(tumor_rows[0], row)) for row in tumor_rows[1:]
             if str(row[4]).startswith("reg")}
    if len(patient) != 35 or len(tumor) != 66:
        raise ValueError("TCIA clinical attachments have an unexpected patient or region scope")
    return patient, tumor


def metadata() -> list[dict]:
    """Join complete source objects to explicit tissue and patient identities."""
    proof = json.loads(PROOF.read_text())["preparation_sources"]["native_images"]
    if not proof["full_source_covered"] or proof["verified_count"] != 251 or proof["failed"]:
        raise ValueError("SPoRA native source verification is incomplete")
    tissue = pd.read_parquet(META / "metadata/tissues.parquet")
    tissue = tissue[tissue.modality == "codex"].set_index("tissue_id", verify_integrity=True)
    canonical = json.loads((META / "codex_ordered_marker_names.json").read_text())
    presence = pd.read_parquet(META / "codex/channels_per_tissue.parquet")
    all_names = pd.read_parquet(META / "codex/channels.parquet").channel_name.tolist()
    patient_clinical, tumor_clinical = clinical_sources()
    panels: dict[tuple[str, ...], str] = {}
    panel_frames = []
    sample_rows: dict[str, dict] = {}
    scope = []
    for item in sorted(proof["verified"], key=lambda value: value["image_id"]):
        image_id = item["image_id"]
        source = NAS / item["path"]
        if not item["all_source_sizes_match"] or item["dtype"] != "float64":
            raise ValueError(f"Unverified source: {image_id}")
        names = item["canonical_channel_labels"]
        if names != canonical[image_id] or names != [n for n in all_names if presence.loc[image_id, n]]:
            raise ValueError(f"Channel order differs from native inventory: {image_id}")
        if image_id not in tissue.index:
            raise ValueError(f"No tissue metadata for {image_id}")
        with formal_read_lease(source):
            root = json.loads((source / "zarr.json").read_text())["attributes"]["ome"]
            array = json.loads((source / "0/zarr.json").read_text())
        if ([entry["label"] for entry in root["omero"]["channels"]] != names
                or array["shape"] != item["shape"] or array["data_type"] != item["dtype"]):
            raise ValueError(f"Native header differs from verified inventory: {image_id}")
        if tuple(names) not in panels:
            panel_id = f"panel_{len(panels) + 1:03d}"
            panels[tuple(names)] = panel_id
            panel_frames.append(panel_from_s1(STUDY, panel_id, names, S1))
        panel_id = panels[tuple(names)]
        row = tissue.loc[image_id]
        if row.original_mpp != 0.37744:
            raise ValueError(f"Unexpected native pixel size for {image_id}")
        origin = str(row.original_patient_identifier)
        kind = ("crc" if origin.startswith("CRC_TMA_A_") else
                "multi_tumor" if origin.startswith("Multi-tumor_TMA_") else "tonsil")
        patient_number = int(row.Patient) if kind == "crc" else None
        if kind == "crc" and patient_number not in patient_clinical:
            raise ValueError(f"CRC patient number absent from TCIA clinical attachment: {image_id}")
        if kind == "multi_tumor":
            region = int(origin.rsplit("_", 1)[1])
            if region not in tumor_clinical:
                raise ValueError(f"Multi-tumor region absent from TCIA composition: {image_id}")
            source_tissue = tumor_clinical[region]["tissue"]
            source_diagnosis = tumor_clinical[region]["diagnosis"]
            if row.tissue != source_tissue:
                raise ValueError(f"Multi-tumor tissue conflicts with TCIA attachment: {image_id}")
        sample_id = (f"crc_{row.patient_id}" if kind == "crc" else
                     f"multitumor_reg{region:03d}" if kind == "multi_tumor" else "tonsil_1")
        sample = dict.fromkeys(columns("sample_metadata"))
        sample.update(study=STUDY, sample_id=sample_id,
                      patient_key=str(row.patient_id) if kind == "crc" else None,
                      source_subject_id=str(patient_number) if kind == "crc" else None,
                      source_sample_id=origin, sample_type="tissue",
                      tissue_type=("colorectal tumor invasive front" if kind == "crc" else
                                   source_tissue if kind == "multi_tumor" else "tonsil"),
                      anatomical_site=("colon" if kind == "crc" else
                                       source_tissue if kind == "multi_tumor" else "tonsil"),
                      sex_reported=str(row.Sex).lower() if kind == "crc" and pd.notna(row.Sex) else None,
                      primary_diagnosis=("colorectal cancer" if kind == "crc" else
                                         source_diagnosis if kind == "multi_tumor" else None),
                      metadata_notes=("CRC patient link verified against TCIA patient annotations; "
                                      "two TMA spots per patient lack an image-level core map. "
                                      "Age and survival unit/origin remain unspecified." if kind == "crc" else
                                      "TCIA composition identifies this individual region; donor unverified."
                                      if kind == "multi_tumor" else
                                      "One source tonsil label spans 45 independent fields; donor unverified."))
        if sample_id in sample_rows and sample_rows[sample_id] != sample:
            raise ValueError(f"Conflicting repeated sample metadata: {sample_id}")
        sample_rows[sample_id] = sample
        scope.append(dict(image_id=image_id, path=item["path"], shape=item["shape"],
                          names=names, panel_id=panel_id, sample_id=sample_id,
                          patient_key=str(row.patient_id) if kind == "crc" else None,
                          source_label=origin,
                          category=kind, embedded_name=root["omero"]["name"],
                          source_revision=relative(PROOF)))
    if (len(scope) != 251 or len(sample_rows) != 102
            or len({row["patient_key"] for row in scope if row["patient_key"]}) != 35):
        raise ValueError("Source image and patient coverage differs from inventory")
    panel = pd.concat(panel_frames, ignore_index=True)
    samples = pd.DataFrame(sample_rows.values())
    mapping = []
    def field(source: Path, source_field: str, target: str, target_field: str,
              join: str, rule: str) -> None:
        record = dict.fromkeys(columns("field_mapping"), "")
        record.update(study=STUDY, source_file=relative(source), source_sheet_or_table=source.suffix,
                      source_field=source_field, target_table=target, target_field=target_field,
                      join_keys=join, transformation_rule=rule)
        mapping.append(record)
    field(META / "metadata/tissues.parquet", "tissue_id;patient_id;specimen_id;Patient",
          "image_metadata", "sample_id", "source_image_id=tissue_id",
          "Exact tissue_id join; 140 CRC images share 35 patient-level tumor sample labels, "
          "66 multi-tumor regions have distinct samples, 45 tonsil fields share one sample. "
          "All 251 images have sample IDs; only 35 CRC samples link to patients.")
    field(NAS / "raw/public" / STUDY / "metadata/CRC_TMAs_patient_annotations.xlsx", "Patient;Sex;Age;OS;DFS",
          "sample_metadata", "source_subject_id;sex_reported", "tissues.Patient=Patient",
          "35 CRC patient numbers checked; sex copied only to CRC samples. Age and survival are not mapped because the "
          "collection time and endpoint time unit/origin are not established by these sources.")
    field(NAS / "raw/public" / STUDY / "metadata/Multi-tumor_TMA_composition.xlsx", "region #;tissue;diagnosis",
          "sample_metadata", "tissue_type;anatomical_site;primary_diagnosis",
          "original_patient_identifier=Multi-tumor_TMA_<regNNN> and attachment region #=regNNN",
          "66 multi-tumor regions retain attachment-specific tissue and diagnosis; source "
          "patient_id for these and tonsil is not accepted as a donor link.")
    field(META / "codex_ordered_marker_names.json", "ordered channel labels",
          "panel_mapping", "source_channel_index;source_name", "source_image_id",
          "Native Zarr labels, ordered inventory and per-tissue presence agree; three panel variants.")
    field(S1, "Study;Original marker;Unified marker;Segmentation channel",
          "panel_mapping", "marker_id;keep_channel;output_channel_index",
          "Study;source_name=Original marker", "Exact S1 match and source order; retain selected nuclear and body inputs.")
    field(PROOF, "verified;shape;dtype;source_bytes", "image_metadata",
          "source_files;source_dtype;source_revision", "source_image_id=image_id",
          "251 complete level-0 source groups. Embedded OME name differs from directory/tissue_id "
          "for every image; preserve both and request independent identity review.")
    field(META / "metadata/tissues.parquet", "original_mpp", "image_metadata",
          "pixel_size_x_um;pixel_size_y_um", "source_image_id=tissue_id",
          "Use 0.37744 um/px from native inventory and TCIA acquisition description; "
          "embedded multiscale numeric scale 1.0 is inconsistent and is recorded as a gap.")
    prior = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file()
             else pd.DataFrame(columns=columns("image_metadata")))
    for index, old in prior.iterrows():
        record = next(item for item in scope if item["image_id"] == old.image_id)
        prior.at[index, "sample_id"] = record["sample_id"]
    # Earlier preparer-only metadata used per-image sample IDs; replace those
    # rows before merging corrected image keys. The output is still unrevised.
    sample_path = OUTPUT / "sample_metadata.csv"
    sample_payload = samples.to_csv(index=False).encode("utf-8-sig")
    if sample_path.is_file() and sample_path.read_bytes() != sample_payload:
        with formal_publish_lease(sample_path, directory=OUTPUT):
            temporary = sample_path.with_suffix(".csv.partial")
            temporary.write_bytes(sample_payload)
            os.replace(temporary, sample_path)
    field_path = OUTPUT / "field_mapping.csv"
    field_payload = pd.DataFrame(mapping).to_csv(index=False).encode("utf-8-sig")
    if field_path.is_file() and field_path.read_bytes() != field_payload:
        with formal_publish_lease(field_path, directory=OUTPUT):
            temporary = field_path.with_suffix(".csv.partial")
            temporary.write_bytes(field_payload)
            os.replace(temporary, field_path)
    save_input_tables(OUTPUT, prior, samples, panel, pd.DataFrame(mapping))
    write_json(CACHE / "scope.json", {"images": scope})
    kept = {panel_id: int(frame.keep_channel.sum()) for panel_id, frame in
            ((f"panel_{i+1:03d}", frame) for i, frame in enumerate(panel_frames))}
    write_json(CACHE / "capacity.json", dict(expected_images=len(scope),
        square_uncompressed_bytes=sum(max(row["shape"][1:])**2 * kept[row["panel_id"]] * 8 for row in scope),
        largest_native_stage_bytes=max(int(np.prod(row["shape"])) * 8 for row in scope),
        nas_reserved_bytes=RESERVE_NAS, local_reserved_bytes=RESERVE_LOCAL))
    completed = set(pd.read_parquet(OUTPUT / "images.parquet").image_id)
    next_image = next((row["image_id"] for row in scope if row["image_id"] not in completed), None)
    progress("metadata_ready", next_image=next_image,
             gaps=["Embedded OME names differ from all source directory/tissue IDs; identity review pending.",
                   "Embedded scale 1.0 differs from original_mpp 0.37744; TCIA states 377.44 nm/px."])
    return scope


class StagedReader:
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
def staged_reader(record: dict):
    """Decode channel-joint Zarr shards once on local SSD for bounded export."""
    source = NAS / record["path"] / "0"
    STAGING.mkdir(parents=True, exist_ok=True)
    stage_bytes = int(np.prod(record["shape"])) * 8
    if shutil.disk_usage(STAGING).free < stage_bytes * 2 + RESERVE_LOCAL:
        raise RuntimeError("Local temporary space would cross the 64 GiB reserve")
    with tempfile.TemporaryDirectory(prefix="schurch-native-", dir=STAGING) as directory:
        stage_path = Path(directory) / "native.dat"
        zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
        blosc.set_nthreads(1)
        with formal_read_lease(source):
            native = zarr.open_array(str(source), mode="r")
            stage = np.memmap(stage_path, mode="w+", dtype="float64", shape=tuple(record["shape"]))
            try:
                for x0, y0, x1, y1 in windows(record["shape"][-2:], 1024):
                    stage[:, y0:y1, x0:x1] = native[:, y0:y1, x0:x1]
                stage.flush()
            finally:
                stage._mmap.close()
        stage = np.memmap(stage_path, mode="r", dtype="float64", shape=tuple(record["shape"]))
        spec = SourceSpec(record["image_id"], record["panel_id"], "zarr", stage_path,
                          tuple(range(record["shape"][0])), pixel_size_um=(0.37744, 0.37744),
                          source_revision=record["source_revision"])
        try:
            yield StagedReader(spec, stage)
        finally:
            stage._mmap.close()


def export_one(record: dict) -> None:
    image_id = record["image_id"]
    current = pd.read_parquet(OUTPUT / "images.parquet")
    if image_id in set(current.image_id):
        return
    panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
    selected = panel[panel.panel_id == record["panel_id"]]
    destination = OUTPUT / "images" / image_id / "input.tiff"
    progress("staging", current_image=image_id, next_image=image_id)
    with staged_reader(record) as reader:
        statistics = export_input(reader, selected, destination)
    if statistics["admitted"]:
        image = dict.fromkeys(columns("image_metadata"))
        _, height, width = record["shape"]
        image.update(study=STUDY, image_id=image_id, sample_id=record["sample_id"],
                     panel_id=record["panel_id"], image_kind="roi", source_image_id=image_id,
                     source_files=[record["path"]], source_format="OME-Zarr", source_axes="CYX",
                     source_dtype="float64", source_revision=record["source_revision"],
                     roi_id=image_id, region_annotation=record["source_label"],
                     source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
                     pixel_size_x_um=0.37744, pixel_size_y_um=0.37744,
                     input_image_path=relative(destination), split_method="already_independent",
                     split_evidence="Complete independent level-0 source ROI; no tissue crop.",
                     intensity_description="Published decoded float64 intensities; no new normalization.",
                     metadata_notes=f"Embedded OME name: {record['embedded_name']}; "
                                    "source directory/tissue_id used for identity. OME scale is 1.0; "
                                    "native acquisition and tissue table state 0.37744 um/px.")
        for key in ("height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px",
                    "pad_bottom_px", "source_channel_count", "channel_count"):
            image[key] = statistics[key]
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
        fields = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False)
        save_input_tables(OUTPUT, pd.DataFrame([image]), samples, panel, fields)
    write_json(CACHE / "image_checks" / f"{image_id}.json", statistics)
    completed = set(pd.read_parquet(OUTPUT / "images.parquet").image_id)
    scope = json.loads((CACHE / "scope.json").read_text())["images"]
    next_image = next((row["image_id"] for row in scope if row["image_id"] not in completed), None)
    progress("image_completed", current_image=image_id, last_admitted=statistics["admitted"],
             next_image=next_image, scientific_repair_round=0)
    print(f"Completed {image_id}: admitted={statistics['admitted']}", flush=True)


def preparer_review(scope: list[dict]) -> None:
    """Record bounded preparer checks without assigning an independent pass."""
    images = pd.read_parquet(OUTPUT / "images.parquet")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
    panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    if images.image_id.duplicated().any() or not set(images.sample_id).issubset(set(samples.sample_id)):
        raise ValueError("Image/sample identity check failed")
    for start in range(0, len(images), 16):
        batch = images.iloc[start:start + 16]
        with formal_read_lease(OUTPUT):
            for row in batch.itertuples(index=False):
                with tifffile.TiffFile(NAS / row.input_image_path) as tif:
                    kept = panel[(panel.panel_id == row.panel_id) & panel.keep_channel].sort_values("output_channel_index")
                    if tif.series[0].shape != (len(kept), row.height_px, row.width_px):
                        raise ValueError(f"Saved TIFF geometry/channel count mismatch: {row.image_id}")
                    if tif.series[0].dtype != np.dtype("float64"):
                        raise ValueError(f"Saved TIFF dtype mismatch: {row.image_id}")
                    xml = ET.fromstring(tif.ome_metadata)
                    names = [item.get("Name") for item in xml.findall(".//{*}Channel")]
                    if names != kept.marker_id.tolist():
                        raise ValueError(f"Saved TIFF channel labels differ from panel: {row.image_id}")
                    pixels = xml.find(".//{*}Pixels")
                    if (float(pixels.get("PhysicalSizeX")) != 0.37744
                            or float(pixels.get("PhysicalSizeY")) != 0.37744):
                        raise ValueError(f"Saved TIFF pixel size differs from native acquisition: {row.image_id}")
        progress("self_check", checked_images=min(start + 16, len(images)), next_image=None,
                 scientific_repair_round=0)
    patient_samples = set(samples.loc[samples.patient_key.notna(), "sample_id"])
    patient_images = int(images.sample_id.isin(patient_samples).sum())
    reviews = []
    for check, result, detail in (
        ("S01", "unresolved", f"Preparer: {len(images)}/251 images exported from 251 verified source objects; embedded OME names differ from directory/tissue_id for all images, and embedded scale differs from acquisition scale. Independent identity and scale review required."),
        ("M01", "unresolved", f"Preparer: {len(images)} indexed images connect to {images.sample_id.nunique()} represented samples among {len(samples)} source sample labels; {patient_images} images connect to {len(patient_samples)} CRC samples with recorded patient keys. Multi-tumor and tonsil donors, CRC core assignment, and age/survival semantics remain unresolved; the public source inquiry found no additional verified per-image bridge."),
        ("D01", "unresolved", f"Preparer checked headers for {len(images)}/251 exported files; independent reviewer and controller acceptance remain pending."),
    ):
        row = dict.fromkeys(columns("review_results"), "")
        row.update(study=STUDY, preparer=JOB.name, reviewer="", scope="study",
                   record_id=STUDY, check_id=check, result=result,
                   source_evidence=relative(PROOF) + ";" + relative(META / "metadata/tissues.parquet") + ";" + relative(CACHE / "public_source_inquiry_0647.json"),
                   output_evidence=relative(OUTPUT), details=detail)
        reviews.append(row)
    target = OUTPUT / "review_results.csv"
    with formal_publish_lease(target, directory=OUTPUT):
        temporary = target.with_suffix(".csv.partial")
        pd.DataFrame(reviews).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)


def repair_metadata() -> None:
    """Reconcile source core labels and CRC anatomical sites after review round 0."""
    with formal_read_lease(META):
        tissue = pd.read_parquet(META / "metadata/tissues.parquet")
    crc = tissue[(tissue.modality == "codex") &
                 tissue.original_patient_identifier.astype(str).str.startswith("CRC_TMA_A_")]
    crc = crc.set_index("tissue_id", verify_integrity=True)
    with formal_read_lease(NAS / "raw/public" / STUDY):
        clinical, _ = clinical_sources()
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
        panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
        fields = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False)
        reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
        prior_review_bytes = (OUTPUT / "review_results.csv").read_bytes()
    if len(images) != 251 or len(samples) != 102 or len(crc) != 140:
        raise ValueError("Published image, sample or CRC source scope changed")
    crc_images = images[images.source_image_id.isin(crc.index)]
    if len(crc_images) != 140:
        raise ValueError("Published CRC image scope differs from source")
    core_labels = crc.tissue_id_old.astype(str)
    core_suffix = core_labels.str.extract(r"(TMA[AB][12])$")[0]
    if (core_suffix.isna().any() or core_labels.duplicated().any() or
            crc.assign(core_suffix=core_suffix.values).groupby("patient_id").core_suffix
            .apply(lambda values: set(values) == {"TMAA1", "TMAA2", "TMAB1", "TMAB2"}).eq(False).any()):
        raise ValueError("Source core labels are not four distinct TMA cores per CRC patient")
    site_map = {"Rectum": "rectum", "Cecum": "colon", "Ascendens": "colon",
                "Transversum": "colon", "Descendens": "colon", "Sigma": "colon"}
    sample_sites = {}
    for image in crc_images.itertuples(index=False):
        source = crc.loc[image.source_image_id]
        patient = clinical[int(source.Patient)]
        if (source.Full_Tumor_Location != patient["Full_Tumor_Location"] or
                source.Full_Tumor_Location not in site_map or
                str(source.patient_id) != image.sample_id.removeprefix("crc_")):
            raise ValueError("CRC source, workbook and published sample join disagree")
        spots = str(patient["TMA spot / region"]).split(",")
        if len(spots) != 2 or not all(part.strip().isdigit() for part in spots):
            raise ValueError("CRC workbook spot pair is incomplete")
        previous = sample_sites.setdefault(image.sample_id, site_map[source.Full_Tumor_Location])
        if previous != site_map[source.Full_Tumor_Location]:
            raise ValueError("One CRC sample has conflicting source sites")
        images.loc[images.image_id == image.image_id, "core_id"] = core_labels.loc[image.source_image_id]
    if len(sample_sites) != 35 or sum(site == "rectum" for site in sample_sites.values()) != 6:
        raise ValueError("CRC sample site coverage differs from source")
    for sample_id, site in sample_sites.items():
        selected = samples.sample_id == sample_id
        if selected.sum() != 1:
            raise ValueError("CRC sample row is missing or duplicated")
        samples.loc[selected, "anatomical_site"] = site
        samples.loc[selected, "metadata_notes"] = (
            "CRC patient link verified against TCIA patient annotations; four source "
            "TMA core labels per patient retained in image core_id. Workbook lists a "
            "two-number spot pair without a one-to-one numeric spot assignment. "
            "Age and survival unit/origin remain unspecified.")

    source_file = relative(META / "metadata/tissues.parquet")
    core = dict.fromkeys(fields.columns, "")
    core.update(study=STUDY, source_file=source_file, source_sheet_or_table=".parquet",
                source_field="tissue_id_old", target_table="image_metadata",
                target_field="core_id", join_keys="source_image_id=tissue_id",
                transformation_rule="For 140 CRC images preserve the complete source tissue_id_old "
                "core label. Each of 35 patients has TMAA1, TMAA2, TMAB1 and TMAB2; "
                "the workbook gives a two-number spot pair but does not assign one "
                "numeric spot to each source label.",
                notes="Source tissue metadata and CRC_TMAs_patient_annotations.xlsx; "
                "https://www.cancerimagingarchive.net/collection/crc_ffpe-codex_cellneighs/")
    site = dict.fromkeys(fields.columns, "")
    site.update(study=STUDY, source_file=source_file, source_sheet_or_table=".parquet",
                source_field="Full_Tumor_Location", target_table="sample_metadata",
                target_field="anatomical_site", join_keys="tissues.patient_id=sample_id without crc_ prefix",
                value_mapping=json.dumps(site_map),
                transformation_rule="Map the CRC source full tumor location to colon or rectum; "
                "confirm the same location in the TCIA patient attachment. Six of 35 "
                "CRC samples are rectum (24 images); the other 29 are colon (116 images).")
    for mapping in (core, site):
        matched = ((fields.source_file == mapping["source_file"]) &
                   (fields.source_field == mapping["source_field"]) &
                   (fields.target_table == mapping["target_table"]) &
                   (fields.target_field == mapping["target_field"]))
        if matched.sum() > 1:
            raise ValueError("Duplicate CRC field mapping")
        if matched.any():
            for column, value in mapping.items():
                fields.loc[matched, column] = value
        else:
            fields = pd.concat([fields, pd.DataFrame([mapping])], ignore_index=True)
    inquiry = fields.source_file == relative(CACHE / "public_source_inquiry_0647.json")
    if inquiry.sum() == 1:
        fields.loc[inquiry, "transformation_rule"] = (
            "Public inquiry did not establish additional donor links. Local tissue_id_old "
            "provides 140 CRC image-level core labels; 66 multi-tumor ROI and 45 tonsil "
            "FOV retain unverified donor identity.")
        fields.loc[inquiry, "value_mapping"] = ""
    save_input_tables(OUTPUT, images, samples, panel, fields)

    preparer = (reviews.scope == "study") & (reviews.record_id == STUDY) & (
        reviews.check_id == "M01") & (reviews.reviewer == "")
    if preparer.sum() != 1:
        raise ValueError("Preparer M01 record is missing or duplicated")
    reviews.loc[preparer, "details"] = (
        "Preparer repair round 1: 251 images connect to 102 source-labelled samples; "
        "140 CRC images have distinct source core_id labels and connect to 35 "
        "patient-keyed samples. The workbook gives spot pairs without an individual "
        "numeric spot assignment. Donor bridges remain unverified for 66 multi-tumor "
        "ROI and 45 tonsil FOV; age/survival time semantics remain unresolved. "
        "Independent recheck and D01 binding are pending.")
    target = OUTPUT / "review_results.csv"
    payload = reviews.to_csv(index=False).encode("utf-8-sig")
    if payload != prior_review_bytes:
        with formal_publish_lease(target, directory=OUTPUT):
            temporary = target.with_suffix(".csv.partial")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
    progress("repair_1_prepared", scientific_repair_round=1, next_image=None,
             repaired_core_images=140, repaired_rectum_samples=6,
             independent_review="pending")


def run() -> None:
    scope_path = CACHE / "scope.json"
    if scope_path.is_file() and (CACHE / "capacity.json").is_file() and all((OUTPUT / name).is_file() for name in
                                    ("images.parquet", "sample_metadata.csv",
                                     "panel_mapping.csv", "field_mapping.csv")):
        scope = json.loads(scope_path.read_text())["images"]
        proof = json.loads(PROOF.read_text())["preparation_sources"]["native_images"]
        if (len(scope) != 251 or len({row["image_id"] for row in scope}) != 251
                or not proof["full_source_covered"] or proof["verified_count"] != 251
                or proof["failed"] or
                {row["image_id"]: row["path"] for row in scope} !=
                {row["image_id"]: row["path"] for row in proof["verified"]}):
            raise ValueError("Cached scope differs from complete source verification")
        completed = set(pd.read_parquet(OUTPUT / "images.parquet").image_id)
        next_image = next((row["image_id"] for row in scope if row["image_id"] not in completed), None)
        progress("metadata_ready", next_image=next_image, reused_verified_scope=True,
                 scientific_repair_round=0)
    else:
        scope = metadata()
    capacity = json.loads((CACHE / "capacity.json").read_text())
    if (shutil.disk_usage(NAS).free < capacity["square_uncompressed_bytes"] + RESERVE_NAS
            or shutil.disk_usage(STAGING).free < capacity["largest_native_stage_bytes"] * 2 + RESERVE_LOCAL):
        completed = set(pd.read_parquet(OUTPUT / "images.parquet").image_id)
        next_image = next((row["image_id"] for row in scope if row["image_id"] not in completed), None)
        progress("waiting_io", reason="NAS or local capacity reserve not available", next_image=next_image)
        return
    for record in scope:
        export_one(record)
    preparer_review(scope)
    progress("prepared", next_image=None, independent_review="pending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Prepare and export the verified local source")
    parser.add_argument("--repair", action="store_true", help="Repair reviewed CRC metadata only")
    arguments = parser.parse_args()
    if arguments.repair:
        repair_metadata()
    elif arguments.run:
        run()
    else:
        parser.print_help()
