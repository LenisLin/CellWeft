"""Prepare native MAPS cHL MIBI and CODEX FOVs as independent input images.

Inputs: Zenodo 10067010 channel TIFFs/ZIP, original case metadata,
        reviewed channel inventories, and the study S1 mapping.
Outputs: resumable input.tiff files and the four study input tables plus
         preparer checks; independent review follows after publication.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (TEMPLATES, export_input, panel_from_s1,
                                                   save_input_tables, validate_image_metadata)


STUDY = "Shaban2024_MultiPlatform_cHL"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_10067010"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0500_prepare_0")))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
ZIP = RAW / "cHL_1_MIBI.zip"
TABLE = NAS / "raw/public" / STUDY / "metadata/tableS2_MIBI.parsed.json"
PAPER = "https://doi.org/10.1038/s41467-023-44188-w"
STAGING = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
RESERVE_NAS = 100 * 1024**3
RESERVE_LOCAL = 64 * 1024**3


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def publish_bytes(path: Path, payload: bytes) -> None:
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def progress(stage: str, completed: list[dict], samples: pd.DataFrame, **detail: object) -> None:
    state = dict(study=STUDY, phase="prepare", repair_round=0, stage=stage,
                 expected_images=39, exported_images=len(completed),
                 sample_linked_images=sum(bool(row.get("sample_id")) for row in completed),
                 sample_rows=len(samples),
                 patient_linked_samples=int(samples.patient_key.notna().sum()),
                 updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(detail)
    publish_bytes(JOB / "progress.json", json.dumps(state, indent=2).encode())


def chl2_region_annotations() -> dict[str, str]:
    status = pd.read_csv(RAW / "cHL_2_MIBI/cHL2_metadata.csv")
    if set(status.image_id) != set(range(1, 7)) or not set(status.control_sample) <= {"Yes", "No"}:
        raise ValueError("The cHL_2 case/control table does not cover its six FOVs")
    return {f"chl2_fov_{int(row.image_id):02d}":
            "reactive lymph node control" if row.control_sample == "Yes" else "classical Hodgkin lymphoma"
            for row in status.itertuples(index=False)}


def source_scope() -> list[dict]:
    """Reconcile 32 archived and seven expanded original FOVs with inventories."""
    archived = pd.read_csv(META / "chl1_archive_channel_marker_files.csv")
    local = pd.read_csv(META / "channel_marker_files.csv")
    manifest = json.loads((META / "chl1_archive_member_manifest.json").read_text())
    if ZIP.stat().st_size != manifest["archive_bytes"] or len(manifest["members"]) != 1504:
        raise ValueError("The cHL_1 archive differs from its reviewed member inventory")
    with formal_read_lease(ZIP), zipfile.ZipFile(ZIP) as archive:
        members = set(archive.namelist())
    if not {row["member"] for row in manifest["members"]}.issubset(members):
        raise ValueError("The cHL_1 archive lacks an inventoried member")
    scope = []
    for name, frame in sorted(archived.groupby("acquisition_id"), key=lambda pair: int(pair[0].rsplit("/", 1)[1])):
        frame = frame[frame.marker != "Segmentation"].sort_values("marker")
        if len(frame) != 46:
            raise ValueError(f"Incomplete archived channel inventory: {name}")
        scope.append(dict(source_id=name, image_id=f"chl1_fov_{int(name.rsplit('/', 1)[1]):02d}",
                          names=frame.marker.tolist(), members=frame.member.tolist(),
                          source_files=[relative(ZIP)], archive=True, platform="MIBI"))
    for name, frame in sorted(local.groupby("image_id")):
        frame = frame.sort_values("marker_name")
        paths = [NAS / value for value in frame.channel_file]
        if len(paths) != (49 if name.startswith("cHL_CODEX") else 46) or any(not path.is_file() for path in paths):
            raise ValueError(f"Incomplete expanded channel inventory: {name}")
        prefix = "codex" if name.startswith("cHL_CODEX") else "chl2"
        suffix = "full" if prefix == "codex" else f"fov_{int(name.rsplit('/', 1)[1]):02d}"
        scope.append(dict(source_id=name, image_id=f"{prefix}_{suffix}", names=frame.marker_name.tolist(),
                          paths=paths, source_files=[relative(path) for path in paths],
                          archive=False, platform="CODEX" if prefix == "codex" else "MIBI"))
    region_annotations = chl2_region_annotations()
    for item in scope:
        if item["image_id"] in region_annotations:
            item["region_annotation"] = region_annotations[item["image_id"]]
    if len(scope) != 39:
        raise ValueError(f"Expected 39 original MAPS acquisitions; found {len(scope)}")
    return scope


def source_samples() -> tuple[dict[int, str], pd.DataFrame]:
    """Use the original MAPS case grouping, without EBV release FOV joins."""
    with formal_read_lease(ZIP), zipfile.ZipFile(ZIP) as archive:
        with archive.open("cHL_1_MIBI/cHL1_metadata.csv") as source:
            rows = pd.read_csv(source, dtype=str, keep_default_na=False)
    if set(rows.columns) != {"case_id", "image_id", "control_sample"} or len(rows) != 32:
        raise ValueError("The original cHL1 case table does not describe 32 FOVs")
    if set(rows.image_id) != {str(number) for number in range(1, 33)}:
        raise ValueError("The original cHL1 case table has missing or duplicate FOVs")
    if not rows.case_id.str.fullmatch(r"\d+").all() or not set(rows.control_sample) <= {"Yes", "No"}:
        raise ValueError("Invalid original cHL1 case or control labels")
    if not rows.groupby("case_id").control_sample.nunique().eq(1).all():
        raise ValueError("Original cHL1 case has conflicting control labels")
    if (rows.case_id.nunique(), rows.control_sample.value_counts().to_dict()) != (20, {"No": 30, "Yes": 2}):
        raise ValueError("Original cHL1 cohort differs from the reviewed scope")
    if rows.loc[rows.control_sample.eq("No"), "case_id"].nunique() != 18:
        raise ValueError("Original cHL1 tumor case count differs from the paper")
    samples = {}
    image_samples = {}
    for row in rows.itertuples(index=False):
        sample_id = f"maps_chl1_case_{row.case_id}"
        image_samples[int(row.image_id)] = sample_id
        if sample_id in samples:
            continue
        control = row.control_sample == "Yes"
        samples[sample_id] = dict(
            study=STUDY, sample_id=sample_id, source_sample_id=row.case_id,
            source_subject_id=None if control else row.case_id,
            patient_key=None if control else f"maps_chl1_case_{row.case_id}",
            sample_type="tissue", tissue_type="normal" if control else "tumor",
            case_control_status="control" if control else "case",
            primary_diagnosis="reactive lymph node" if control else "classical Hodgkin lymphoma",
            anatomical_site="lymph node",
            metadata_notes=("Original MAPS case_id groups FOVs. The paper reports one control rLN, "
                            "but its two control case IDs cannot yet be resolved as one specimen."
                            if control else
                            "Patient key uses original MAPS case_id to group FOVs. Its interpretation "
                            "as a patient is supported by 18 tumor case IDs and the paper's 18 patients, "
                            "pending independent review. EBV clinical fields await a cross-release FOV map."))
    samples["codex_tissue_1"] = dict(
        study=STUDY, sample_id="codex_tissue_1", source_sample_id="cHL_CODEX",
        sample_type="tissue", tissue_type="tumor", case_control_status="case",
        primary_diagnosis="classical Hodgkin lymphoma", anatomical_site="lymph node",
        metadata_notes="MAPS article reports one large cHL CODEX tissue image from one patient; the patient identity and cross-platform overlap are unknown.")
    frame = pd.DataFrame([dict.fromkeys(columns("sample_metadata"), None) | row for row in samples.values()])
    return image_samples, frame


def field_mapping() -> pd.DataFrame:
    entries = []

    def add(source: Path | str, field: str, target_table: str, target_field: str, keys: str, rule: str) -> None:
        entries.append(dict(study=STUDY, source_file=relative(source) if isinstance(source, Path) else source,
                            source_sheet_or_table="public source inventory", source_field=field,
                            target_table=target_table, target_field=target_field, join_keys=keys,
                            value_mapping="", transformation_rule=rule, time_origin_rule="",
                            conflict_rule="retain source value; unresolved identity remains null",
                            notes=""))

    add(META / "chl1_archive_channel_marker_files.csv", "member;marker;acquisition_id",
        "panel_mapping", "source_name;source_channel_index", "acquisition_id",
        "The ZIP member basename supplies each native channel; Segmentation.tiff is a label mask, not an intensity channel. Alphabetical marker order is the explicit source-file order.")
    add(META / "channel_marker_files.csv", "channel_file;marker_name;image_id",
        "panel_mapping", "source_name;source_channel_index", "image_id",
        "Each scalar source TIFF supplies one channel; alphabetical marker order is explicit.")
    add(S1, "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
        "marker_id;mapping_status;role;nuclear;body;keep_channel", "Study + Original marker",
        "Join exact source basename to accepted S1 and retain its selected segmentation channels.")
    add(ZIP, "cHL_1_MIBI/cHL1_metadata.csv: image_id;case_id;control_sample",
        "sample_metadata", "sample_id;source_sample_id;patient_key;case_control_status",
        "image_id -> case_id", "Group original FOVs by case_id; interpret 18 tumor case IDs as within-cohort patient keys based on the paper's 18-patient count, pending independent review. Keep two control case IDs separate pending specimen identity evidence.")
    add(PAPER, "cHL 1 cohort description", "sample_metadata",
        "patient_key;primary_diagnosis;anatomical_site", "original cHL1 case grouping",
        "The paper reports 18 cHL patients and one control reactive lymph node. It supplies no per-case histology, sex, timepoint or EBV status.")
    add(TABLE, "A;B;C;D;E;F", "sample_metadata", "metadata_notes",
        "cross-release FOV map unresolved", "The later EBV cohort shares original source ownership, but same-number FOV case groups conflict with cHL1_metadata.csv. Do not transfer patient or clinical fields without an exact cross-release correspondence.")
    add(PAPER, "cHL CODEX dataset description", "sample_metadata", "sample_id",
        "single full CODEX FOV", "One large tissue image is explicitly reported from one patient; create one sample row and leave patient_key blank because its identity is not published.")
    add(ZIP, "cHL_1_MIBI/raw_image/<FOV>/<marker>.tiff", "image_metadata",
        "source_files;source_image_id;source_dtype;source_x1_px;source_y1_px",
        "archive member acquisition path", "Use complete source FOV; decode original float32 TIFF pixels without scaling. ZIP member location is in source_image_id and panel inventory.")
    add(RAW / "cHL_2_MIBI/raw_image", "<FOV>/<marker>.tiff", "image_metadata",
        "source_files;source_image_id;source_dtype;source_x1_px;source_y1_px", "FOV directory",
        "Use complete source FOV and native float32 intensity; sample crosswalk is unavailable.")
    add(RAW / "cHL_2_MIBI/cHL2_metadata.csv", "image_id;control_sample", "image_metadata",
        "region_annotation", "image_id = cHL_2 FOV directory number",
        "Yes denotes the reactive lymph node control; No denotes cHL case by the MAPS cohort description. The table does not identify samples or patients.")
    add(RAW / "cHL_CODEX/raw_image", "<marker>.tiff", "image_metadata",
        "source_files;source_image_id;source_dtype;source_x1_px;source_y1_px", "full image",
        "Use complete source image and native uint16 intensity; no source sample crosswalk established.")
    return pd.DataFrame(entries, columns=columns("field_mapping"))


class PlaneReader:
    """Expose same-size native TIFF memmaps in source-channel order."""

    def __init__(self, source_path: Path, planes: list[Path | None]):
        self.source = type("Source", (), {"path": source_path, "pixel_size_um": None})()
        self.arrays = [tifffile.memmap(path, mode="r") if path else None for path in planes]
        present = [array for array in self.arrays if array is not None]
        if not present or any(array.ndim != 2 or array.shape != present[0].shape or
                              array.dtype != present[0].dtype for array in present):
            raise ValueError(f"Native planes differ in dimensions or dtype: {source_path}")
        self.dtype = present[0].dtype
        self.shape = (len(planes), *present[0].shape)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.stack([np.asarray(self.arrays[index][y0:y1, x0:x1]) for index in channels])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def close(self) -> None:
        for array in self.arrays:
            if array is not None:
                array._mmap.close()


def image_row(item: dict, panel: pd.DataFrame, result: dict, sample_id: str | None) -> dict:
    number = int(item["source_id"].rsplit("/", 1)[1]) if item["platform"] == "MIBI" else None
    h, w = item["shape"]
    return dict(study=STUDY, image_id=item["image_id"], sample_id=sample_id,
                panel_id=item["panel_id"], image_kind="fov" if number else "roi",
                source_image_id=item["source_id"], source_files=item["source_files"],
                source_format="zip_tiff_planes" if item["archive"] else "tiff_planes",
                source_axes="YX scalar planes", source_dtype=result["source_dtype"],
                source_revision="zenodo:10067010:v1", fov_id=str(number) if number else None,
                roi_id=item["source_id"] if number is None else None,
                region_annotation=item.get("region_annotation"), source_x0_px=0, source_y0_px=0,
                source_x1_px=w, source_y1_px=h,
                pad_left_px=result["pad_left_px"], pad_right_px=result["pad_right_px"],
                pad_top_px=result["pad_top_px"], pad_bottom_px=result["pad_bottom_px"],
                height_px=result["height_px"], width_px=result["width_px"],
                source_channel_count=result["source_channel_count"],
                channel_count=result["channel_count"],
                input_image_path=relative(OUTPUT / "images" / item["image_id"] / "input.tiff"),
                valid_pixels_path=None, intensity_description="Published native channel TIFF intensities; no additional scaling",
                split_method="already_independent", split_evidence=item["source_id"],
                metadata_notes=None if sample_id else "Source FOV to biological sample/patient crosswalk unresolved.")


def prepare_one(item: dict, panel: pd.DataFrame) -> dict:
    kept = panel.loc[panel.keep_channel, "source_channel_index"].tolist()
    with tempfile.TemporaryDirectory(prefix="shaban-fov-", dir=STAGING) as directory:
        paths: list[Path | None] = [None] * len(item["names"])
        if item["archive"]:
            with formal_read_lease(ZIP), zipfile.ZipFile(ZIP) as archive:
                for index in kept:
                    path = Path(directory) / f"{index:03d}.tiff"
                    with archive.open(item["members"][index]) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    paths[index] = path
        else:
            with formal_read_lease(item["paths"][0]):
                for index in kept:
                    path = Path(directory) / f"{index:03d}.tiff"
                    shutil.copyfile(item["paths"][index], path)
                    paths[index] = path
        reader = PlaneReader(Path(directory), paths)
        try:
            item["shape"] = reader.shape[1:]
            side = max(reader.shape[1:])
            estimate = len(kept) * side * side * reader.dtype.itemsize
            if shutil.disk_usage(STAGING).free < RESERVE_LOCAL + estimate * 2:
                raise RuntimeError("waiting_io: local 64 GiB reserve plus one FOV staging is unavailable")
            if shutil.disk_usage(NAS).free < RESERVE_NAS + estimate:
                raise RuntimeError("waiting_io: NAS 100 GiB reserve plus one FOV output is unavailable")
            destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
            result = export_input(reader, panel, destination, data_root=NAS)
        finally:
            reader.close()
    return result


def reviews(completed: list[dict], samples: pd.DataFrame) -> None:
    path = OUTPUT / "review_results.csv"
    if path.is_file():
        with formal_read_lease(path):
            prior = pd.read_csv(path, dtype=str, keep_default_na=False)
        if prior.reviewer.ne("").any():
            return
    rows = []
    for check, result, detail in (
        ("S01", "pass" if len(completed) == 39 else "unresolved", f"Original MAPS scope: 32 cHL_1 MIBI, 6 cHL_2 MIBI, 1 CODEX; exported {len(completed)}; later EBV FOVs await crosswalk."),
        ("I01", "pass", "Published rows use original FOV paths and native full acquisition frames."),
        ("I02", "pass", "Nonempty sample and panel foreign keys were checked by the shared table publisher."),
        ("I03", "pass", "Shared exporter recorded full native bounds and centered square padding."),
        ("P01", "pass", "All source intensity channels use the reviewed file inventories and exact S1 labels."),
        ("P02", "pass", "S1 selection and output positions generated by shared panel helper."),
        ("P03", "pass", "Each input TIFF header was reopened by the shared exporter."),
        ("M01", "unresolved", f"Original MAPS case metadata and article connect {sum(bool(x.get('sample_id')) for x in completed)} exported images to {len(samples)} sample rows; {int(samples.patient_key.notna().sum())} tumor sample rows have patient keys. Two control case IDs may refer to one reported rLN specimen. cHL_2 sample/patient identities and later EBV FOV correspondence remain unresolved."),
        ("D01", "unresolved", "Preparer delivery complete for current scope; independent reviewer has not yet run."),
    ):
        rows.append(dict(study=STUDY, preparer=JOB.name, reviewer="",
                         scope="study", record_id=STUDY, check_id=check, result=result,
                         source_evidence=relative(META), output_evidence=relative(OUTPUT), details=detail))
    frame = pd.DataFrame(rows, columns=columns("review_results"))
    publish_bytes(OUTPUT / "review_results.csv", frame.to_csv(index=False).encode("utf-8-sig"))


def repair_metadata() -> None:
    """Correct original case identity while retaining reviewed native TIFFs."""
    image_samples, samples = source_samples()
    fields = field_mapping()
    names = ("images.parquet", "sample_metadata.csv", "field_mapping.csv")
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        versions = {name: ((OUTPUT / name).stat().st_size, (OUTPUT / name).stat().st_mtime_ns)
                    for name in names}
        table = pq.read_table(OUTPUT / "images.parquet")
        panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    image_ids = table.column("image_id").to_pylist()
    if len(image_ids) != 39 or len(set(image_ids)) != 39:
        raise ValueError("Repair requires the existing 39-image index")
    for image_id in image_ids:
        if image_id.startswith("chl1_fov_") and int(image_id.rsplit("_", 1)[1]) not in image_samples:
            raise ValueError(f"Original case metadata missing {image_id}")
    if sum(image_id.startswith("chl1_fov_") for image_id in image_ids) != 32:
        raise ValueError("Repair requires all 32 original cHL1 images")
    sample_ids = [image_samples[int(image_id.rsplit("_", 1)[1])]
                  if image_id.startswith("chl1_fov_") else original
                  for image_id, original in zip(image_ids, table.column("sample_id").to_pylist())]
    old_notes = table.column("metadata_notes").to_pylist()
    notes = ["Original MAPS case_id supplies the tissue sample; later EBV FOV identity is unresolved."
             if image_id.startswith("chl1_fov_") else old
             for image_id, old in zip(image_ids, old_notes)]
    for name, values in (("sample_id", sample_ids), ("metadata_notes", notes)):
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))
    validate_image_metadata(table.to_pandas(), samples, panel, fields)
    if shutil.disk_usage(NAS).free < RESERVE_NAS + 1024**2:
        raise RuntimeError("waiting_io: NAS 100 GiB reserve plus metadata publication is unavailable")
    if shutil.disk_usage(STAGING).free < RESERVE_LOCAL + 1024**2:
        raise RuntimeError("waiting_io: local 64 GiB reserve plus metadata staging is unavailable")
    progress("repair_staging", table.to_pandas().to_dict("records"), samples,
             phase="repair", repair_round=1, next_scope="cHL1 metadata publication")
    with tempfile.TemporaryDirectory(prefix="shaban-repair-", dir=STAGING) as directory:
        staged = Path(directory)
        pq.write_table(table, staged / "images.parquet", compression="zstd")
        (staged / "sample_metadata.csv").write_bytes(samples.to_csv(index=False).encode("utf-8-sig"))
        (staged / "field_mapping.csv").write_bytes(fields.to_csv(index=False).encode("utf-8-sig"))
        with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
            changed_names = [name for name in names
                             if (OUTPUT / name).read_bytes() != (staged / name).read_bytes()]
        with formal_publish_lease(OUTPUT, directory=OUTPUT):
            current = {name: ((OUTPUT / name).stat().st_size, (OUTPUT / name).stat().st_mtime_ns)
                       for name in names}
            if current != versions:
                raise RuntimeError("Input tables changed during metadata repair")
            for name in changed_names:
                temporary = OUTPUT / (name + ".repair_1.partial")
                shutil.copyfile(staged / name, temporary)
                os.replace(temporary, OUTPUT / name)
    record_repair()
    progress("prepared_pending_independent_review", table.to_pandas().to_dict("records"), samples,
             phase="repair", repair_round=1, next_scope="I04/M01/M02 cHL1; D01 binding")


def record_repair() -> None:
    path = OUTPUT / "review_results.csv"
    source = f"{relative(ZIP)}#cHL_1_MIBI/cHL1_metadata.csv"
    affected = [f"chl1_fov_{number:02d}" for number in range(1, 33)]
    items = (
        ("image", "chl1_32", "I04", affected,
         "Repair 1: replaced the later EBV FOV-number assignment with original MAPS case_id; independent review pending."),
        ("sample", "chl1_32", "M01", affected,
         "Repair 1: original case grouping connects 32 images to 20 source cases; two control case IDs remain separate pending specimen evidence."),
        ("sample", "chl1_32", "M02", affected,
         "Repair 1: removed EBV patient, sex, histology, timepoint and status assignments without a cross-release FOV map."),
        ("study", "chl1_32", "D01", affected,
         "Repair 1 metadata published; independent review and current input binding pending."),
        ("study", STUDY, "D01", ["codex_full"],
         "Shared sample and field table revisions invalidate the earlier CODEX binding; independent rebinding pending."),
    )
    with formal_read_lease(path), resource_lock(OUTPUT, shared=True):
        prior = pd.read_csv(path, dtype=str, keep_default_na=False)
        version = (path.stat().st_size, path.stat().st_mtime_ns)
    prior = prior[~((prior.preparer == JOB.name) & prior.reviewer.eq(""))]
    rows = [dict(study=STUDY, preparer=JOB.name, reviewer="", scope=scope,
                 record_id=record_id, check_id=check_id, result="unresolved",
                 source_evidence=source, output_evidence=relative(OUTPUT),
                 details=json.dumps({"repair_round": 1, "image_ids": image_ids,
                                     "summary": summary}, separators=(",", ":")))
            for scope, record_id, check_id, image_ids, summary in items]
    updated = pd.concat([prior, pd.DataFrame(rows, columns=prior.columns)], ignore_index=True)
    payload = updated.to_csv(index=False).encode("utf-8-sig")
    with formal_publish_lease(path, directory=OUTPUT):
        if (path.stat().st_size, path.stat().st_mtime_ns) != version:
            raise RuntimeError("Review results changed during repair publication")
        temporary = path.with_name(path.name + ".repair_1.partial")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


if os.environ.get("CELLWEFT_INPUT_PHASE") == "repair":
    STAGING.mkdir(parents=True, exist_ok=True)
    repair_metadata()
    raise SystemExit(0)


waiting_reason = os.environ.get("CELLWEFT_INPUT_WAITING_IO_REASON")
if waiting_reason:
    with formal_read_lease(OUTPUT):
        paused_images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records")
        paused_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
        paused_panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
    region_annotations = chl2_region_annotations()
    for row in paused_images:
        if row["image_id"] in region_annotations:
            row["region_annotation"] = region_annotations[row["image_id"]]
    save_input_tables(OUTPUT, pd.DataFrame(paused_images), paused_samples, paused_panel,
                      field_mapping(), data_root=NAS)
    reviews(paused_images, paused_samples)
    progress("waiting_io", paused_images, paused_samples, next_image="codex_full",
             reason=waiting_reason)
    raise SystemExit(75)


# Reconcile the source scope and original case grouping.
STAGING.mkdir(parents=True, exist_ok=True)
scope = source_scope()
patient_rows, samples = source_samples()
fields = field_mapping()
panels = {}
for item in scope:
    key = tuple(item["names"])
    if key not in panels:
        panels[key] = panel_from_s1(STUDY, f"panel_{len(panels) + 1:02d}", item["names"], S1)
    item["panel_id"] = panels[key].panel_id.iloc[0]
panel_table = pd.concat(panels.values(), ignore_index=True)

# Continue from the published image index; each new FOV is closed before indexing.
with formal_read_lease(OUTPUT):
    existing = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
completed = existing.to_dict("records")
known = {row["image_id"] for row in completed}
progress("source_scope_checked", completed, samples, next_image=next((x["image_id"] for x in scope if x["image_id"] not in known), None))
for item in scope:
    if item["image_id"] in known:
        continue
    image_id = item["image_id"]
    progress("staging", completed, samples, next_image=image_id)
    try:
        result = prepare_one(item, panels[tuple(item["names"])])
        if not result["admitted"]:
            raise ValueError(f"No retained downstream marker signal: {image_id}")
        number = int(item["source_id"].rsplit("/", 1)[1]) if item["source_id"].startswith("cHL_1_MIBI") else None
        sample_id = (patient_rows[number] if number else
                     "codex_tissue_1" if item["platform"] == "CODEX" else None)
        row = image_row(item, panels[tuple(item["names"])], result, sample_id)
        save_input_tables(OUTPUT, pd.DataFrame([row]), samples, panel_table, fields, data_root=NAS)
        completed.append(row)
        known.add(image_id)
        progress("image_completed", completed, samples, last_completed_image=image_id,
                 next_image=next((x["image_id"] for x in scope if x["image_id"] not in known), None))
    except RuntimeError as error:
        if str(error).startswith("waiting_io:"):
            progress("waiting_io", completed, samples, next_image=image_id, reason=str(error))
            raise SystemExit(75) from error
        raise

# Refresh clinical source fields when resuming a fully exported scope.
if completed:
    region_by_image = {item["image_id"]: item["region_annotation"] for item in scope
                       if "region_annotation" in item}
    for row in completed:
        if row["image_id"] == "codex_full":
            row["sample_id"] = "codex_tissue_1"
        if row["image_id"] in region_by_image:
            row["region_annotation"] = region_by_image[row["image_id"]]
    save_input_tables(OUTPUT, pd.DataFrame(completed), samples, panel_table, fields, data_root=NAS)
reviews(completed, samples)
progress("prepared_pending_independent_review", completed, samples, next_image=None)
