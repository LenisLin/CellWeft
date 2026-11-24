"""Prepare the locally complete SPORA human lung IMC ROIs."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Zhuhuman2025_IMC_Lung"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
RAW = NAS / "raw/public" / STUDY / "spora"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
JOB = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_JOB",
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0693_prepare_0",
))
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12151776/"
SUPPLEMENT_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S153561082500162X-mmc2.xlsx"
SUPPLEMENT = NAS / "raw/public" / STUDY / "metadata/table_s1.xlsx"
PREPARER = JOB.name


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, NAS))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    with formal_publish_lease(path, directory=path.parent, data_root=NAS):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def progress(stage: str, completed: int, **details: object) -> None:
    publish_json(JOB / "progress.json", dict(study=STUDY, phase="prepare", stage=stage,
                 expected_images=1649, completed_images=completed,
                 updated_at=datetime.now(timezone.utc).isoformat(), **details))


class ArrayReader:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.source = SimpleNamespace(path=Path("/tmp/study_input_batch"), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.array[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def source_record(image_id: str, row: pd.Series, ordered: dict, channel_table: pd.DataFrame,
                  availability: pd.DataFrame) -> dict:
    group = RAW / "imc/1_0mpp/images" / f"{image_id}.ome.zarr"
    with formal_read_lease(group, data_root=NAS):
        root_meta = json.loads((group / "zarr.json").read_text())
        array_meta = json.loads((group / "0/zarr.json").read_text())
        shape = array_meta["shape"]
        names = channel_table.loc[availability.loc[image_id].to_numpy(bool), "channel_name"].tolist()
        source_names = [item["channel_name"] for item in ordered[image_id]]
        labels = [item["label"] for item in root_meta["attributes"]["ome"]["omero"]["channels"]]
        if names != source_names or names != labels or shape != [len(names), int(row.height), int(row.width)]:
            raise ValueError(f"{image_id}: source panel or shape differs from published metadata")
        if array_meta["data_type"] != "float32" or array_meta["dimension_names"] != ["c", "y", "x"]:
            raise ValueError(f"{image_id}: unexpected native dtype or axes")
        axes = root_meta["attributes"]["ome"]["multiscales"][0]
        if [item["name"] for item in axes["axes"]] != ["c", "y", "x"] or axes["datasets"][0]["coordinateTransformations"][0]["scale"] != [1.0, 1.0, 1.0]:
            raise ValueError(f"{image_id}: unexpected native axis or pixel scale")
        chunk = array_meta["chunk_grid"]["configuration"]["chunk_shape"]
        files = [group / "zarr.json", group / "0/zarr.json"]
        for y in range((shape[1] + chunk[1] - 1) // chunk[1]):
            for x in range((shape[2] + chunk[2] - 1) // chunk[2]):
                files.append(group / "0/c/0" / str(y) / str(x))
        versions = []
        for path in files:
            stat = path.stat()
            if stat.st_size == 0:
                raise ValueError(f"{image_id}: empty source shard {path.name}")
            versions.append(dict(path=relative(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return dict(image_id=image_id, group=relative(group), shape=shape, names=names,
                files=versions, source_revision=json.dumps(versions, separators=(",", ":")))


def attach_supplement(tissues: pd.DataFrame, cases: pd.DataFrame,
                      rois: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    tissues = tissues.copy()
    tail = tissues.old_tissue_id.astype(str).str.rsplit("_", n=1).str[-1]
    parsed = tail.str.extract(r"^(.*)ROI([0-9]+)$")
    if parsed.isna().any().any():
        raise ValueError("A source ROI lacks a lesion and ROI index")
    tissues["source_group_id"] = tissues.original_patient_identifier.astype(str) + "-" + parsed[0]
    tissues["sample_id"] = (STUDY + "__" + tissues.patient_id.astype(str) + "__" + parsed[0])
    paper_roi = tissues.source_group_id + "-ROI" + parsed[1].astype(int).astype(str).str.zfill(3)
    if rois.ROI_ID.dropna().duplicated().any():
        raise ValueError("Published Table S1B repeats an ROI identifier")
    tissues["region_annotation"] = paper_roi.map(rois.set_index("ROI_ID").ROI_Diag)

    cases = cases.copy()
    cases["Case"] = cases.Case.ffill()
    cases["source_group_id"] = cases["Path Acc"].str.replace(
        r"^[A-Za-z]+(?=[0-9]{4}-)", "", regex=True)
    groups = tissues[["patient_id", "source_group_id"]].drop_duplicates()
    joined = groups.merge(cases[["source_group_id", "Case"]], on="source_group_id", how="left")
    if joined.groupby("patient_id").Case.nunique().gt(1).any():
        raise ValueError("Table S1A gives conflicting cases for a SPORA patient")
    case_for_patient = joined.dropna(subset=["Case"]).drop_duplicates("patient_id").set_index("patient_id").Case
    clinical = cases.groupby("Case").first()
    patient_fields = {}
    for patient_id, case_id in case_for_patient.items():
        source = clinical.loc[case_id]
        patient_fields[patient_id] = dict(
            sex_reported={"F": "female", "M": "male"}.get(source["Gender"]),
            smoking_status={"NEVER": "never", "FORMER": "former", "CURRENT": "current"}.get(
                source["Type of smoker (Current, Former, Never)"]),
            vital_status={"A": "alive", "D": "deceased"}.get(
                source["Current VITAL STATUS (Alive;A or Deceased;D)"]),
        )
    if (tissues.sample_id.nunique(), tissues.region_annotation.notna().sum(),
            len(case_for_patient)) != (123, 1502, 56):
        raise ValueError("Published supplement does not reproduce the verified lesion, ROI, and case joins")
    return tissues, patient_fields


def sample_table(tissues: pd.DataFrame, patient_fields: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, group in tissues.groupby("sample_id", sort=False):
        tissue = group.iloc[0]
        diagnoses = group.region_annotation.dropna().unique().tolist()
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=tissue.sample_id,
                   patient_key=f"{STUDY}__{tissue.patient_id}",
                   source_subject_id=tissue.original_patient_identifier,
                   source_sample_id=tissue.source_group_id, sample_type="tissue",
                   anatomical_site="lung", preservation_method="ffpe",
                   histology=diagnoses[0] if len(diagnoses) == 1 else None,
                   metadata_notes=("Source old_tissue_id lesion prefix groups ROIs by tissue section. "
                                   "Table S1B histology is left blank when matched ROIs disagree; "
                                   "unmatched ROI annotations and incomplete clinical linkage remain open."))
        row.update(patient_fields.get(tissue.patient_id, {}))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns("sample_metadata"))


def field_table(tissues: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(source: str, source_field: str, target_table: str, target_field: str,
            join: str, rule: str, notes: str) -> None:
        row = dict.fromkeys(columns("field_mapping"), "")
        row.update(study=STUDY, source_file=source, source_field=source_field,
                   target_table=target_table, target_field=target_field,
                   join_keys=join, transformation_rule=rule, notes=notes)
        rows.append(row)
    tissue = relative(RAW / "metadata/tissues.parquet")
    add(tissue, "tissue_id;patient_id;specimen_id;old_tissue_id", "image_metadata",
        "image_id;sample_id;source_image_id;roi_id", "tissue_id",
        "Use one complete SPORA tissue ROI per input; group old_tissue_id lesion prefixes within patient as tissue sections.",
        f"{len(tissues)} public ROI records link to {len(samples)} source-supported tissue sections.")
    add(tissue, "patient_id;original_patient_identifier;tissue_id", "sample_metadata",
        "patient_key;source_subject_id;sample_id", "tissue_id",
        "Keep the published pseudonymous patient hash; confirm one-to-one mapping with original_patient_identifier.",
        "The local SPORA tissue table identifies 62 patient groups.")
    add(relative(SUPPLEMENT), "Table S1B: ROI_ID;ROI_Diag", "image_metadata",
        "region_annotation", "original_patient_identifier + old_tissue_id lesion/ROI suffix",
        "Match the exact paper ROI identifier after zero-padding the source ROI number to three digits.",
        f"{tissues.region_annotation.notna().sum()} of {len(tissues)} source ROIs match; unmatched annotations remain empty.")
    add(relative(SUPPLEMENT), "Table S1B: ROI_Diag", "sample_metadata",
        "histology", "source tissue-section group",
        "Fill only when all matched ROIs within a section have one diagnosis.",
        f"{samples.histology.notna().sum()} of {len(samples)} sections have consistent matched ROI diagnoses.")
    add(relative(SUPPLEMENT), "Table S1A: Path Acc;Case;Gender;Type of smoker (Current, Former, Never);Current VITAL STATUS (Alive;A or Deceased;D)",
        "sample_metadata", "sex_reported;smoking_status;vital_status",
        "unique normalized Path Acc tissue-section key -> Case -> source patient",
        "Remove the letter prefix only from four-digit accession keys; require one case per SPORA patient and use the case-level source values.",
        f"Case linkage supports {samples.sex_reported.notna().sum()} of {len(samples)} sections; unmatched patients remain blank. Source: {SUPPLEMENT_URL}")
    add(SOURCE_URL, "Methods: FFPE slides; human lung IMC", "sample_metadata",
        "sample_type;anatomical_site;preservation_method", "study-level scope",
        "Record tissue, lung, and FFPE only; leave ROI-specific stage and diagnosis empty.",
        "Paper describes 1618 analyzed ROIs; the public SPORA release lists 1649. Exact paper exclusion crosswalk is unresolved.")
    add(relative(RAW / "imc/channels.parquet") + ";" + relative(RAW / "imc/channels_per_tissue.parquet"),
        "channel_name;per-tissue availability", "panel_mapping", "source_name;source_channel_index",
        "tissue_id; canonical channel row order", "Filter channel rows by per-tissue availability; compare OME labels and prior ordered mapping.",
        "Three source panels (43, 44, 45 channels).")
    add(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
        "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
        "Study + exact source channel name", "Apply the current S1 mapper independently to each source panel.",
        "NAKATPASE is S1-excluded; NAK-ATPASE is selected where measured.")
    add(relative(RAW / "imc/1_0mpp/images"), "0/zarr.json;zarr.json;0/c/0 shards", "image_metadata",
        "source_files;source_revision;source_axes;source_dtype;geometry;input_image_path",
        "tissue_id", "Read native float32 CYX ROI, retain source pixels, center-pad to square and publish OME BigTIFF.",
        "Source revision records expected shard size and mtime; unmeasured tissue grouping and source-to-paper exclusions remain open.")
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def image_row(image_id: str, tissue: pd.Series, record: dict, panel_id: str,
              result: dict, destination: Path) -> dict:
    height, width = record["shape"][1:]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=image_id, sample_id=tissue.sample_id,
               panel_id=panel_id, image_kind="roi", source_image_id=image_id,
               source_files=[record["group"]], source_format="ome_zarr",
               source_axes="CYX", source_dtype="float32",
               source_revision=record["source_revision"], roi_id=tissue.old_tissue_id,
               region_annotation=tissue.region_annotation,
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=result["height_px"], width_px=result["width_px"],
               source_channel_count=result["source_channel_count"], channel_count=result["channel_count"],
               pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
               intensity_description="SPORA 1_um decoded float32 native intensities; no input normalization",
               split_method="already_independent", split_evidence=relative(RAW / "metadata/tissues.parquet"),
               metadata_notes="One source tissue ROI; Table S1B annotation is present only for an exact ROI identifier match.")
    return row


def migrate_previous_tables(rows: list[dict], tissues: pd.DataFrame,
                            samples: pd.DataFrame, prior_samples: pd.DataFrame,
                            versions: tuple[tuple[int, int] | None, ...]) -> None:
    lookup = tissues.set_index("tissue_id")
    changed = False
    for row in rows:
        tissue = lookup.loc[row["image_id"]]
        annotation = tissue.region_annotation if pd.notna(tissue.region_annotation) else None
        old_annotation = row.get("region_annotation")
        if row["sample_id"] != tissue.sample_id or (None if pd.isna(old_annotation) else old_annotation) != annotation:
            row["sample_id"] = tissue.sample_id
            row["region_annotation"] = annotation
            changed = True
    if not changed and set(prior_samples.sample_id) == set(samples.sample_id):
        return
    if not {row["sample_id"] for row in rows}.issubset(set(samples.sample_id)):
        raise ValueError("Existing images cannot be linked to source tissue sections")
    staging_root = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zhu-section-metadata-", dir=staging_root) as directory:
        staged = Path(directory)
        images = pd.DataFrame(rows)
        table = pa.Table.from_pandas(images, preserve_index=False)
        position = table.schema.get_field_index("source_files")
        table = table.set_column(position, "source_files",
                                 pa.array(images.source_files.tolist(), type=pa.list_(pa.string())))
        pq.write_table(table, staged / "images.parquet", compression="zstd")
        samples.to_csv(staged / "sample_metadata.csv", index=False, encoding="utf-8-sig")
        targets = (OUTPUT / "images.parquet", OUTPUT / "sample_metadata.csv")
        with formal_publish_lease(OUTPUT, directory=OUTPUT, data_root=NAS):
            current = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else None for p in targets)
            if current != versions:
                raise RuntimeError("Existing input metadata changed during section migration")
            for target in targets:
                temporary = target.with_name(target.name + ".partial")
                shutil.copyfile(staged / target.name, temporary)
                os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=None, help="Bound newly processed images for a managed batch")
    args = parser.parse_args()
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
    blosc.set_nthreads(1)
    with formal_read_lease(RAW / "metadata/tissues.parquet", data_root=NAS):
        tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
        channels = pd.read_parquet(RAW / "imc/channels.parquet")
        availability = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
        ordered = json.loads((META / "imc_ordered_channels_by_tissue.json").read_text())
    if len(tissues) != 1649 or set(tissues.tissue_id) != set(availability.index) != set(ordered):
        raise ValueError("SPORA metadata has an incomplete or duplicated tissue scope")
    if availability.columns.tolist() != channels.channel_name.tolist():
        raise ValueError("Source channel availability order differs from canonical channel order")
    if tissues.groupby("patient_id").original_patient_identifier.nunique().max() != 1 or tissues.groupby("original_patient_identifier").patient_id.nunique().max() != 1:
        raise ValueError("SPORA patient identifiers do not form a one-to-one crosswalk")
    with formal_read_lease(SUPPLEMENT, data_root=NAS):
        cases = pd.read_excel(SUPPLEMENT, sheet_name="Table S1A", dtype=str)
        rois = pd.read_excel(SUPPLEMENT, sheet_name="Table S1B", dtype=str)
    tissues, patient_fields = attach_supplement(tissues, cases, rois)
    samples = sample_table(tissues, patient_fields)
    fields = field_table(tissues, samples)
    panels = {}
    with formal_read_lease(S1, data_root=NAS):
        for tissue_id in tissues.tissue_id:
            names = channels.loc[availability.loc[tissue_id].to_numpy(bool), "channel_name"].tolist()
            signature = tuple(names)
            if signature not in panels:
                panel_id = f"{STUDY}__native_{len(names)}"
                panels[signature] = panel_from_s1(STUDY, panel_id, names, S1)
    panel = pd.concat(panels.values(), ignore_index=True)
    panel_by_name = {names: frame for names, frame in panels.items()}
    if len(panels) != 3:
        raise ValueError("Unexpected number of native source panels")
    prior = OUTPUT / "images.parquet"
    prior_sample_path = OUTPUT / "sample_metadata.csv"
    with formal_read_lease(OUTPUT, data_root=NAS):
        rows = pd.read_parquet(prior).to_dict("records") if prior.is_file() else []
        prior_samples = pd.read_csv(prior_sample_path, dtype=str) if prior_sample_path.is_file() else pd.DataFrame()
        versions = tuple((path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else None
                         for path in (prior, prior_sample_path))
    if rows and not prior_samples.empty:
        migrate_previous_tables(rows, tissues, samples, prior_samples, versions)
    indexed = {row["image_id"] for row in rows}
    prior_rows = {row["image_id"]: row for row in rows}
    receipts = CACHE / "image_receipts"
    processed = 0
    excluded = []
    for tissue in tissues.set_index("tissue_id").itertuples():
        image_id = tissue.Index
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = receipts / f"{image_id}.json"
        if image_id in indexed and destination.is_file() and destination.stat().st_size > 0:
            continue
        row = tissues.loc[tissues.tissue_id == image_id].iloc[0]
        record = source_record(image_id, row, ordered, channels, availability)
        current_panel = panel_by_name[tuple(record["names"])]
        receipt = None
        if receipt_path.is_file():
            with formal_read_lease(receipt_path, data_root=NAS):
                receipt = json.loads(receipt_path.read_text())
        matching_receipt = (receipt is not None
                            and receipt["source_revision"] == record["source_revision"]
                            and receipt["panel_id"] == current_panel.panel_id.iloc[0])
        if matching_receipt and not receipt["exported"]["admitted"]:
            if image_id in indexed:
                raise ValueError(f"{image_id}: an excluded source remains in the published image index")
            excluded.append(dict(image_id=image_id, reason="all retained downstream measurements zero"))
            continue
        reusable = matching_receipt and destination.is_file()
        if reusable and image_id in prior_rows:
            prior_row = prior_rows[image_id]
            exported = receipt["exported"]
            reusable = (destination.stat().st_size > 0
                        and prior_row["panel_id"] == current_panel.panel_id.iloc[0]
                        and prior_row["input_image_path"] == relative(destination)
                        and prior_row["channel_count"] == exported["channel_count"]
                        and exported["channel_count"] == int(current_panel.keep_channel.sum())
                        and exported["height_px"] == exported["width_px"] == max(record["shape"][1:]))
        elif reusable:
            with formal_read_lease(destination, data_root=NAS):
                with tifffile.TiffFile(destination) as tif:
                    reusable = (len(tif.pages) == int(current_panel.keep_channel.sum())
                                and tif.pages[0].shape == (max(record["shape"][1:]),) * 2
                                and tif.pages[0].dtype == np.dtype("float32"))
        if reusable:
            if image_id not in indexed:
                rows.append(image_row(image_id, row, record, current_panel.panel_id.iloc[0],
                                      receipt["exported"], destination))
                indexed.add(image_id)
            continue
        if args.max_images is not None and processed >= args.max_images:
            break
        side = max(record["shape"][1:])
        output_need = side * side * int(current_panel.keep_channel.sum()) * 4
        local_need = side * side * int(current_panel.keep_channel.sum()) * 4
        if shutil.disk_usage(NAS).free < 100 * 1024**3 + output_need:
            save_input_tables(OUTPUT, pd.DataFrame(rows, columns=columns("image_metadata")),
                              samples, panel, fields, data_root=NAS)
            progress("waiting_io", len(indexed), reason="NAS reserve below 100 GiB plus next input", next_image=image_id)
            return
        if shutil.disk_usage("/tmp").free < 64 * 1024**3 + local_need:
            save_input_tables(OUTPUT, pd.DataFrame(rows, columns=columns("image_metadata")),
                              samples, panel, fields, data_root=NAS)
            progress("waiting_io", len(indexed), reason="Local reserve below 64 GiB plus next staging TIFF", next_image=image_id)
            return
        if processed % 5 == 0:
            progress("processing_image_batch", len(indexed), next_image=image_id)
        with formal_read_lease(NAS / record["group"], data_root=NAS):
            array = np.asarray(zarr.open_array(str(NAS / record["group"] / "0"), mode="r")[:])
        if tuple(array.shape) != tuple(record["shape"]) or array.dtype != np.dtype("float32"):
            raise ValueError(f"{image_id}: decoded native array differs from verified source header")
        result = export_input(ArrayReader(array), current_panel, destination, tile_size=256, data_root=NAS)
        if not result["admitted"]:
            publish_json(receipt_path, dict(image_id=image_id, source_revision=record["source_revision"],
                                            panel_id=current_panel.panel_id.iloc[0], exported=result))
            if image_id in indexed:
                raise ValueError(f"{image_id}: source changed to zero-signal after prior publication")
            excluded.append(dict(image_id=image_id, reason="all retained downstream measurements zero"))
            processed += 1
            if processed % 5 == 0:
                save_input_tables(OUTPUT, pd.DataFrame(rows, columns=columns("image_metadata")),
                                  samples, panel, fields, data_root=NAS)
                progress("batch_saved", len(indexed), last_image=image_id, excluded_count=len(excluded))
            continue
        kept = current_panel[current_panel.keep_channel].sort_values("output_channel_index")
        with formal_read_lease(destination, data_root=NAS):
            with tifffile.TiffFile(destination) as tif:
                if len(tif.pages) != len(kept) or tif.pages[0].shape != (side, side) or tif.pages[0].dtype != np.dtype("float32"):
                    raise ValueError(f"{image_id}: published TIFF header mismatch")
                for channel in kept.itertuples(index=False):
                    window = (0, 0, min(32, record["shape"][2]), min(32, record["shape"][1]))
                    x0, y0, x1, y1 = window
                    output_window = (x0 + result["pad_left_px"], y0 + result["pad_top_px"],
                                     x1 + result["pad_left_px"], y1 + result["pad_top_px"])
                    actual = read_tiff_window(tif, int(channel.output_channel_index), output_window)
                    expected = array[channel.source_channel_index, y0:y1, x0:x1]
                    if not np.array_equal(actual, expected):
                        raise ValueError(f"{image_id}: native pixel check differs at channel {channel.source_channel_index}")
        receipt = dict(image_id=image_id, source_revision=record["source_revision"],
                       panel_id=current_panel.panel_id.iloc[0], exported=result,
                       preparer_pixel_check="all retained channels, first 32x32 native block")
        publish_json(receipt_path, receipt)
        if image_id in indexed:
            rows = [prior_row for prior_row in rows if prior_row["image_id"] != image_id]
        rows.append(image_row(image_id, row, record, current_panel.panel_id.iloc[0], result, destination))
        indexed.add(image_id)
        processed += 1
        if processed % 5 == 0:
            save_input_tables(OUTPUT, pd.DataFrame(rows, columns=columns("image_metadata")), samples,
                              panel, fields, data_root=NAS)
            progress("batch_saved", len(indexed), last_image=image_id, excluded_count=len(excluded))
    save_input_tables(OUTPUT, pd.DataFrame(rows, columns=columns("image_metadata")), samples, panel, fields,
                      data_root=NAS)
    publish_json(CACHE / "scope_summary.json", dict(study=STUDY, source_count=len(tissues),
                 exported_images=len(indexed), excluded=excluded, panel_count=len(panels),
                 sample_rows=len(samples), patient_groups=samples.patient_key.nunique(),
                 exact_paper_roi_annotations=int(tissues.region_annotation.notna().sum()),
                 patient_clinical_sample_rows=int(samples.sex_reported.notna().sum()),
                 source_provenance=[relative(RAW / "metadata/tissues.parquet"),
                                    relative(META / "imc_ordered_channels_by_tissue.json"),
                                    relative(SUPPLEMENT), SOURCE_URL, SUPPLEMENT_URL],
                 paper_scope_note="Published analyzed ROI count 1618 versus 1649 local SPORA tissue rows; exact crosswalk unresolved."))
    reviews = []
    for check, details in [
        ("S01", "1649 SPORA ROI entries; 1502 exact Table S1B ROI annotations; published analyzed count 1618; remaining correspondence unresolved."),
        ("M01", f"{len(samples)} tissue-section samples link to 62 pseudonymous patients. "
         f"{samples.sex_reported.notna().sum()} samples have a Table S1A patient-level clinical crosswalk; "
         "unmatched sections and ROI annotations remain unresolved."),
        ("D01", f"Preparer closed {len(indexed)} inputs; independent review and controller acceptance pending."),
    ]:
        reviews.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                            record_id=STUDY, check_id=check, result="unresolved",
                            source_evidence=relative(RAW / "metadata/tissues.parquet") + ";" + relative(SUPPLEMENT) + ";" + SOURCE_URL,
                            output_evidence=relative(OUTPUT / "images.parquet") + ";" + relative(CACHE / "scope_summary.json"),
                            details=details))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT, data_root=NAS):
        reviews_frame = pd.DataFrame(reviews, columns=columns("review_results"))
        temporary = review_path.with_name(review_path.name + ".partial")
        reviews_frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, review_path)
    progress("prepared_pending_independent_review" if len(indexed) == len(tissues) - len(excluded)
             else "batch_complete", len(indexed), excluded_count=len(excluded),
             sample_rows=len(samples), patient_linked_samples=len(samples))
    print(json.dumps(dict(study=STUDY, source_count=len(tissues), exported_images=len(indexed),
                          excluded_count=len(excluded), sample_rows=len(samples),
                          patient_groups=samples.patient_key.nunique()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
