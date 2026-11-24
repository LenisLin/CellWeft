"""Prepare published Salie HCC SPORA IMC images and specimen metadata.

The SPORA float32 values are copied exactly; their upstream normalization
history is described by the source publication, not reversed here.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
import zarr
from numcodecs import blosc
from PIL import Image

from cellweft.data.contracts import ImageBlock, SourceSpec, Window
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, windows
from cellweft.data.preparation.io_coordination import default_io_root, formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Salie2025_IMC_HCC"
PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "spora"
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0451_prepare_0")))
PROOF = ROOT / "raw/public" / STUDY / "manifest.json"
ORDER = metadata_source_directory(ROOT, STUDY) / 'spora_inventory' / 'imc_ordered_marker_names.json'
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TISSUES = RAW / "metadata/tissues.parquet"
SOURCE_REVISION = "SPORA salie2025spatial local verified 194-image release"
PREPARER = "Salie2025_preparer_" + JOB.name
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11874287/"
IMAGE_COLUMNS = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
CLINICAL_COLUMNS = (
    "age", "sex", "etiology", "immune_type", "overall_survival_months", "censored",
    "cirrhosis", "grading_tma", "tu_max_size_cm", "nodules", "vascular_invasion",
    "uicc_stage", "r_status", "biopsy_or_resectate", "bclc_stage",
    "child_pugh_score", "ecog_baseline", "therapy_regimen", "drugs", "therapy_line",
    "best_response_recist", "response", "pfs_months", "afp_ng_dl",
)
zarr.config.set({"async.concurrency": 4, "threading.max_workers": 4})
blosc.set_nthreads(2)
pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def checkpoint(phase: str, completed: int, **details: object) -> None:
    atomic_json(JOB / "progress.json", dict(study=STUDY, phase=phase,
                expected_images=194, completed_images=completed,
                updated_at=pd.Timestamp.now(tz="UTC").isoformat(), **details))


def source_path(image_id: str) -> Path:
    return RAW / "imc/1_0mpp/images" / f"{image_id}.ome.zarr"


def panel_id(names: list[str]) -> str:
    return f"spora_imc_{len(names)}"


class DecodedROI:
    """Serve one decoded source image to the shared tiled TIFF publisher."""

    def __init__(self, values: np.ndarray):
        self.values = values
        self.shape, self.dtype = values.shape, values.dtype
        self.source = SimpleNamespace(path=Path("/tmp/study_input_batch"), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def iter_blocks(self, channels: list[int], block_size: int):
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


def source_inventory(tissues: pd.DataFrame, orders: dict[str, list[str]]) -> dict[str, dict]:
    proof = json.loads(PROOF.read_text())["preparation_sources"]["native_images"]
    verified = {item["image_id"]: item for item in proof["verified"]}
    identities = set(tissues.tissue_id)
    if not proof["full_source_covered"] or identities != set(verified) or identities != set(orders):
        raise ValueError("The tissue, ordered-panel and verified image scopes differ")
    for image_id, item in verified.items():
        path = source_path(image_id)
        metadata = json.loads((path / "0/zarr.json").read_text())
        ome = json.loads((path / "zarr.json").read_text())
        names = [channel["label"] for channel in ome["attributes"]["ome"]["omero"]["channels"]]
        if (not item["all_source_sizes_match"] or not path.is_dir()
                or metadata["shape"] != item["shape"] or metadata["data_type"] != item["dtype"]
                or names != orders[image_id] or len(names) != item["shape"][0]):
            raise ValueError(f"Current source differs from native proof: {image_id}")
        shards = list((path / "0/c").rglob("*"))
        files = [member for member in shards if member.is_file() and not member.name.endswith(".partial")]
        if len(files) != item["source_object_count"] - 2:
            raise ValueError(f"Incomplete source shards: {image_id}")
        current_bytes = sum(member.stat().st_size for member in files)
        current_bytes += (path / "zarr.json").stat().st_size + (path / "0/zarr.json").stat().st_size
        if current_bytes != item["source_bytes"]:
            raise ValueError(f"Changed source size: {image_id}")
    return verified


def make_metadata(tissues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_template = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str)
    for field in CLINICAL_COLUMNS:
        different = tissues.groupby("patient_id")[field].nunique(dropna=False).gt(1)
        if different.any():
            raise ValueError(f"Source clinical field varies within one specimen: {field}")
    samples = []
    for _, group in tissues.groupby("patient_id", sort=True):
        source = group.iloc[0]
        item = {field: pd.NA for field in sample_template.columns}
        item.update(study=STUDY, sample_id=f"specimen_{source.patient_id}",
                    patient_key=f"{STUDY}:{source.patient_id}",
                    source_subject_id=source.original_patient_id, cohort_id=source.cohort,
                    sample_type="HCC liver tissue", tissue_type="tumor",
                    anatomical_site="liver", primary_diagnosis="hepatocellular carcinoma",
                    case_control_status="case", preservation_method="ffpe",
                    collection_method=("resection" if source.cohort == "Discovery" or source.biopsy_or_resectate == "R"
                                       else "biopsy" if source.biopsy_or_resectate == "B" else pd.NA),
                    age_at_collection_years=source.age,
                    sex_reported={"m": "male", "f": "female"}.get(str(source.sex).lower(), pd.NA),
                    response_criteria="RECIST" if pd.notna(source.best_response_recist) else pd.NA,
                    best_overall_response=source.best_response_recist,
                    metadata_notes=("One published FFPE tissue sample per patient; region identity is on image rows. "
                                    "Source clinical fields retained verbatim with source_ prefix. "
                                    "OS/PFS origins and censoring code are not established by this local source."))
        for field in CLINICAL_COLUMNS:
            value = source[field]
            item[f"source_{field}"] = value if pd.notna(value) else pd.NA
        samples.append(item)
    samples = pd.DataFrame(samples)
    mappings = []

    def mapping(source_file: Path, field: str, table: str, target: str, rule: str, *,
                notes: str = "", join: str = "") -> None:
        if not join:
            join = ("sample.sample_id = 'specimen_' + tissues.patient_id"
                    if table == "sample_metadata" else "tissues.patient_id = sample.patient_id")
        mappings.append(dict(study=STUDY, source_file=relative(source_file),
                             source_sheet_or_table=source_file.stem, source_field=field,
                             target_table=table, target_field=target, join_keys=join,
                             value_mapping="{}", transformation_rule=rule,
                             time_origin_rule="", conflict_rule="source rows must agree within patient",
                             notes=notes))

    mapping(TISSUES, "tissue_id", "image_metadata", "image_id", "exact source identity", join="tissues.tissue_id = image.image_id")
    mapping(TISSUES, "patient_id", "image_metadata", "sample_id", "specimen_ + source patient_id",
            join="tissues.tissue_id = image.image_id; image.sample_id = 'specimen_' + tissues.patient_id")
    mapping(TISSUES, "location", "image_metadata", "region_annotation", "preserve source ROI region",
            join="tissues.tissue_id = image.image_id")
    mapping(TISSUES, "original_patient_id", "sample_metadata", "source_subject_id", "preserve source value")
    mapping(TISSUES, "patient_id", "sample_metadata", "patient_key", "study + source patient_id")
    mapping(TISSUES, "cohort", "sample_metadata", "cohort_id", "preserve source value")
    mapping(TISSUES, "age", "sample_metadata", "age_at_collection_years", "preserve reported age")
    mapping(TISSUES, "sex", "sample_metadata", "sex_reported", "m=male; f=female")
    mapping(TISSUES, "biopsy_or_resectate", "sample_metadata", "collection_method",
            "Discovery surgical samples=resection; ICI B=biopsy, R=resection",
            notes=PAPER)
    mapping(TISSUES, "best_response_recist", "sample_metadata", "best_overall_response", "preserve RECIST code")
    for source_field, target, rule, values, evidence in (
        ("cohort", "anatomical_site", "HCC specimen site is liver", {}, "Methods: Study cohort"),
        ("cohort", "case_control_status", "published HCC cohorts are cases", {}, "Methods: Study cohort"),
        ("cohort", "preservation_method", "both IMC cohorts use FFPE sections", {}, "Methods: Study cohort"),
        ("cohort", "primary_diagnosis", "published cohort diagnosis is HCC", {}, "Methods: Study cohort"),
        ("best_response_recist", "response_criteria",
         "RECIST when best_response_recist is present; otherwise empty", {},
         "Results: checkpoint therapy outcome, RECIST v1.1"),
        ("cohort", "sample_type", "HCC liver tissue specimen", {}, "Methods: Study cohort"),
        ("cohort", "tissue_type", "HCC tumor specimen", {}, "Methods: Study cohort"),
    ):
        mapping(TISSUES, source_field, "sample_metadata", target, rule,
                notes=f"{PAPER}; {evidence}")
    for field in CLINICAL_COLUMNS:
        kind = "number" if pd.api.types.is_numeric_dtype(tissues[field]) else "string"
        mapping(TISSUES, field, "sample_metadata", f"source_{field}", "preserve source value without reinterpretation",
                notes=json.dumps(dict(data_type=kind, definition=f"Original SPORA tissues.{field}; "
                      "clinical time origin and coding follow source metadata")))
    mapping(ORDER, "ordered_marker_names", "panel_mapping", "source_name", "OME channel labels equal ordered inventory",
            join="images.image_id = imc_ordered_marker_names key; images.panel_id = panel_mapping.panel_id; ordered_marker_names[source_channel_index] = panel_mapping.source_name")
    mapping(S1, "Original marker", "panel_mapping", "marker_id", "exact study-specific S1 match",
            join="existing Salie study selection in S1; S1 Original marker = panel_mapping.s1_original_marker; retain the existing marker identity mapping")
    return samples, pd.DataFrame(mappings)


def repair_m02() -> None:
    """Add the seven source-supported mappings without changing reviewed image assets."""
    fields_to_repair = {
        "anatomical_site", "case_control_status", "preservation_method",
        "primary_diagnosis", "response_criteria", "sample_type", "tissue_type",
    }
    with formal_read_lease(TISSUES):
        tissues = pd.read_parquet(TISSUES)
    expected_samples, expected_mappings = make_metadata(tissues)
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False, dtype=str)
        current = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False, dtype=str)
    if len(samples) != 96 or len(expected_samples) != 96:
        raise ValueError("Sample scope changed since independent review")
    actual = samples.set_index("sample_id").fillna("")
    expected = expected_samples.set_index("sample_id").fillna("")
    if set(actual.index) != set(expected.index) or not actual[sorted(fields_to_repair)].equals(
            expected.loc[actual.index, sorted(fields_to_repair)].astype(str)):
        raise ValueError("Reviewed sample values differ from current source reconstruction")
    additions = expected_mappings.loc[
        expected_mappings.target_table.eq("sample_metadata") &
        expected_mappings.target_field.isin(fields_to_repair)
    ].copy()
    if len(additions) != len(fields_to_repair):
        raise ValueError("Incomplete M02 mapping repair")
    old_targets = set(current.loc[current.target_table.eq("sample_metadata"), "target_field"])
    additions = additions.loc[~additions.target_field.isin(old_targets)]
    if len(additions):
        target = OUTPUT / "field_mapping.csv"
        with formal_publish_lease(target, directory=OUTPUT):
            latest = pd.read_csv(target, keep_default_na=False, dtype=str)
            if set(latest.target_field) != set(current.target_field) or len(latest) != len(current):
                raise ValueError("Field mapping changed during repair")
            temporary = target.with_name(target.name + ".partial")
            pd.concat([latest, additions], ignore_index=True).to_csv(
                temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, target)
    checkpoint("repair_m02_prepared", 194, scientific_repair_round=1,
               repaired_fields=sorted(fields_to_repair), added_mapping_rows=len(additions),
               sample_rows=96, sample_linked_images=194, patient_linked_samples=96,
               next_step="independent M02 review and D01 rebinding")


def repair_m02_join() -> None:
    """Replace the 38 invalid sample joins identified by independent review."""
    with formal_read_lease(TISSUES):
        tissues = pd.read_parquet(TISSUES)
    expected_samples, expected_mappings = make_metadata(tissues)
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False, dtype=str)
        current = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False, dtype=str)

    sample_join = "sample.sample_id = 'specimen_' + tissues.patient_id"
    old_join = "tissues.patient_id = sample.patient_id"
    actual_ids = samples.set_index("sample_id")
    expected_ids = expected_samples.set_index("sample_id")
    if (len(samples) != 96 or len(tissues) != 194 or set(actual_ids.index) != set(expected_ids.index)
            or not actual_ids.loc[expected_ids.index, "patient_key"].equals(
                expected_ids["patient_key"].astype(str))):
        raise ValueError("Current sample identity differs from source reconstruction")

    affected = current.target_table.eq("sample_metadata")
    expected = expected_mappings.loc[expected_mappings.target_table.eq("sample_metadata")]
    actual_fields = current.loc[affected].set_index("target_field")
    expected_fields = expected.set_index("target_field")
    if (int(affected.sum()) != 38 or len(expected) != 38
            or set(actual_fields.index) != set(expected_fields.index)
            or not actual_fields[["source_field", "transformation_rule"]].equals(
                expected_fields.loc[actual_fields.index, ["source_field", "transformation_rule"]])
            or not current.loc[affected, "join_keys"].isin([old_join, sample_join]).all()):
        raise ValueError("Reviewed M02 mapping scope changed")

    pending = affected & current.join_keys.eq(old_join)
    checkpoint("repair_m02_join_ready", 194, scientific_repair_round=2,
               affected_mapping_rows=int(pending.sum()), sample_rows=96,
               sample_linked_images=194, patient_linked_samples=96)
    if pending.any():
        target = OUTPUT / "field_mapping.csv"
        with formal_publish_lease(target, directory=OUTPUT):
            latest = pd.read_csv(target, keep_default_na=False, dtype=str)
            if not latest.equals(current):
                raise ValueError("Field mapping changed during repair")
            latest.loc[pending, "join_keys"] = sample_join
            temporary = target.with_name(target.name + ".partial")
            latest.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, target)
    checkpoint("repair_m02_join_prepared", 194, scientific_repair_round=2,
               repaired_mapping_rows=int(pending.sum()), sample_rows=96,
               sample_linked_images=194, patient_linked_samples=96,
               next_step="independent M02 recheck and D01 rebinding")


def image_record(row: pd.Series, names: list[str], result: dict) -> dict:
    identity = row.tissue_id
    path = source_path(identity)
    destination = OUTPUT / "images" / identity / "input.tiff"
    record = {field: None for field in IMAGE_COLUMNS}
    record.update(study=STUDY, image_id=identity, sample_id=f"specimen_{row.patient_id}",
                  panel_id=panel_id(names), image_kind="roi", source_image_id=identity,
                  source_files=[relative(path)], source_format="zarr", source_axes="CYX",
                  source_dtype="float32", source_revision=SOURCE_REVISION,
                  roi_id=row.specimen_id, region_annotation=row.location,
                  source_x0_px=0, source_y0_px=0, source_x1_px=result["source_width"],
                  source_y1_px=result["source_height"], pixel_size_x_um=1.0,
                  pixel_size_y_um=1.0, input_image_path=relative(destination),
                  intensity_description=("Published SPORA float32 values preserved exactly, without this script "
                                         "rescaling or clipping; upstream original filepath identifies a normalized TIFF."),
                  split_method="already_independent",
                  split_evidence=f"{relative(TISSUES)}:tissue_id, location, filepath; {PAPER}",
                  metadata_notes=("SPORA has distinct ROI paths; source specimen_id is an ROI ordinal. "
                                  "Original instrument counts are unavailable in this local SPORA scope."))
    for field in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                  "pad_bottom_px", "source_channel_count", "channel_count"):
        record[field] = result[field]
    return record


def check_header(path: Path, panel: pd.DataFrame, side: int) -> None:
    expected = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
    with formal_read_lease(path), resource_lock(OUTPUT, shared=True), resource_lock(
            path.parent, shared=True, name=".input.lock"):
        with tifffile.TiffFile(path) as tif:
            if tif.series[0].shape != (len(expected), side, side) or tif.series[0].dtype != np.dtype("float32"):
                raise ValueError(f"Unexpected TIFF header: {path}")
            ome = ElementTree.fromstring(tif.ome_metadata)
            actual = [node.attrib["Name"] for node in ome.iter() if node.tag.endswith("}Channel")]
            if actual != expected:
                raise ValueError(f"Unexpected TIFF channel order: {path}")


def review(rows: list[dict], samples: pd.DataFrame, complete: bool) -> None:
    details = dict(stage="preparer_self_check", expected_images=194, exported_images=len(rows),
                   sample_rows=len(samples), sample_linked_images=len(rows),
                   patient_linked_samples=int(samples.patient_key.notna().sum()),
                   scientific_repair_round=0, independent_review="pending",
                   upstream_normalized_source=True)
    checks = {
        "S01": ("study", PROOF, "Verified 194 local SPORA images; whole-image marker admission follows export."),
        "M01": ("sample", TISSUES, "Original patient links 96 published FFPE tissue samples and 194 ROI records."),
        "M02": ("sample", TISSUES, "Clinical source columns are preserved; OS/PFS origins remain unspecified."),
        "P01": ("panel", ORDER, "Two ordered OME channel sets agree with source inventory."),
        "P02": ("panel", S1, "Exact study-specific S1 mapping and source-order retention."),
        "P03": ("panel", OUTPUT / "panel_mapping.csv", "Preparer TIFF header and channel order check."),
        "I01": ("study", PROOF, "Source identities and local byte sizes match verified inventory."),
        "I02": ("study", TISSUES, "Every delivered ROI links one specimen and panel."),
        "I03": ("study", PROOF, "Full native source ROI and centered square padding."),
        "I04": ("study", TISSUES, "Published tissue_id identifies independent ROI."),
        "D01": ("study", OUTPUT, "Independent review and acceptance binding pending."),
    }
    additions = []
    for check, (scope, source, note) in checks.items():
        result = "unresolved" if check == "D01" or (check == "S01" and not complete) else "pass"
        additions.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                              record_id=f"{STUDY}:prepare_0", check_id=check, result=result,
                              source_evidence=relative(source), output_evidence=relative(OUTPUT),
                              details=json.dumps(dict(details, note=note), ensure_ascii=False)))
    target = OUTPUT / "review_results.csv"
    with formal_publish_lease(target, directory=OUTPUT):
        existing = pd.read_csv(target, keep_default_na=False) if target.exists() else pd.DataFrame(columns=additions[0].keys())
        existing = existing.loc[~((existing.preparer == PREPARER) & (existing.reviewer == "") &
                                  (existing.record_id == f"{STUDY}:prepare_0"))]
        temporary = target.with_name(target.name + ".partial")
        pd.concat([existing, pd.DataFrame(additions)], ignore_index=True).to_csv(
            temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, target)


def main() -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    tissues = pd.read_parquet(TISSUES).sort_values("tissue_id").reset_index(drop=True)
    orders = json.loads(ORDER.read_text())
    proof = source_inventory(tissues, orders)
    panels = {size: panel_from_s1(STUDY, f"spora_imc_{size}",
                                  next(names for names in orders.values() if len(names) == size), S1)
              for size in {len(names) for names in orders.values()}}
    if len(panels) != 2 or set(panels) != {42, 45}:
        raise ValueError("Unexpected source panel scope")
    for image_id, names in orders.items():
        if names != panels[len(names)].source_name.tolist():
            raise ValueError(f"Source channel order differs within panel: {image_id}")
    samples, mappings = make_metadata(tissues)
    panel = pd.concat([panels[size] for size in sorted(panels)], ignore_index=True)
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        images_path = OUTPUT / "images.parquet"
        existing = pd.read_parquet(images_path) if images_path.exists() else pd.DataFrame(columns=IMAGE_COLUMNS)
    rows = existing.to_dict("records")
    completed = set(existing.image_id)
    excluded_path = JOB / "excluded_images.json"
    excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else {}
    checkpoint("source_checked", len(rows), source_images=194, source_objects=602,
               expected_uncompressed_square_bytes=sum(max(v["shape"][1:])**2 *
               (v["shape"][0] - int(v["shape"][0] == 42 and "pERK12" in orders[k])) * 4
               for k, v in proof.items()), sample_rows=len(samples), patient_linked_samples=len(samples))
    for row in tissues.itertuples(index=False):
        image_id = row.tissue_id
        if image_id in completed or image_id in excluded:
            continue
        source = source_path(image_id)
        destination = OUTPUT / "images" / image_id / "input.tiff"
        side = max(proof[image_id]["shape"][1:])
        expected_bytes = side * side * len(panels[len(orders[image_id])].loc[
            panels[len(orders[image_id])].keep_channel]) * 4
        if (shutil.disk_usage(ROOT).free - expected_bytes < 100 * 1024**3 or
                shutil.disk_usage("/tmp").free - expected_bytes < 64 * 1024**3):
            checkpoint("waiting_io", len(rows), next_image=image_id, reason="reserved disk space threshold")
            return
        checkpoint("exporting", len(rows), current_image=image_id,
                   next_image=image_id, sample_rows=len(samples), patient_linked_samples=len(samples))
        spec = SourceSpec(image_id, panel_id(orders[image_id]), "zarr", source / "0",
                          tuple(range(len(orders[image_id]))), pixel_size_um=(1.0, 1.0),
                          source_revision=SOURCE_REVISION)
        with formal_read_lease(spec.path):
            with open_source(spec) as reader:
                height, width = reader.shape[1:]
                values = reader.read_window(list(range(reader.shape[0])), (0, 0, width, height))
        if values.shape != tuple(proof[image_id]["shape"]) or values.dtype != np.dtype("float32"):
            raise ValueError(f"Decoded source changed: {image_id}")
        selected = panels[len(orders[image_id])]
        if destination.exists():
            with formal_read_lease(destination), resource_lock(OUTPUT, shared=True), resource_lock(
                    destination.parent, shared=True, name=".input.lock"):
                actual = tifffile.imread(destination, maxworkers=2)
            kept = selected.loc[selected.keep_channel].sort_values("output_channel_index")
            top, left = (side-height)//2, (side-width)//2
            if not np.array_equal(actual[:, top:top+height, left:left+width],
                                  values[kept.source_channel_index.tolist()]) or not np.isfinite(actual).all():
                raise ValueError(f"Unindexed existing TIFF needs independent inspection: {image_id}")
            result = dict(admitted=True, height_px=side, width_px=side,
                          pad_left_px=left, pad_top_px=top,
                          pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                          source_channel_count=len(orders[image_id]), channel_count=len(kept),
                          source_width=width, source_height=height)
        else:
            result = export_input(DecodedROI(values), selected, destination)
            result.update(source_width=width, source_height=height)
        del values
        if result["admitted"]:
            check_header(destination, selected, side)
            rows.append(image_record(pd.Series(row._asdict()), orders[image_id], result))
            save_input_tables(OUTPUT, pd.DataFrame([rows[-1]], columns=IMAGE_COLUMNS), samples, panel, mappings)
            completed.add(image_id)
        else:
            excluded[image_id] = "All retained downstream marker channels have zero signal"
            atomic_json(excluded_path, excluded)
        checkpoint("exporting", len(rows), last_completed_image=image_id,
                   next_image=next((x for x in tissues.tissue_id if x not in completed and x not in excluded), None),
                   excluded_images=len(excluded), sample_rows=len(samples),
                   sample_linked_images=len(rows), patient_linked_samples=len(samples))
        print(json.dumps(dict(completed=len(rows), excluded=len(excluded), image_id=image_id)), flush=True)
    save_input_tables(OUTPUT, pd.DataFrame(rows, columns=IMAGE_COLUMNS), samples, panel, mappings)
    review(rows, samples, len(rows) + len(excluded) == len(tissues))
    checkpoint("prepared", len(rows), excluded_images=len(excluded), sample_rows=len(samples),
               sample_linked_images=len(rows), patient_linked_samples=len(samples),
               independent_review="pending")


def verify() -> None:
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
        panels = pd.read_csv(OUTPUT / "panel_mapping.csv")
        fields = pd.read_csv(OUTPUT / "field_mapping.csv")
        reviews = pd.read_csv(OUTPUT / "review_results.csv")
    tissues = pd.read_parquet(TISSUES)
    assert len(images) == images.image_id.nunique() == len(tissues) == 194
    assert set(images.image_id) == set(tissues.tissue_id)
    assert len(samples) == samples.sample_id.nunique() == 96
    assert set(images.sample_id).issubset(set(samples.sample_id))
    assert samples.patient_key.ne("").all() and samples.patient_key.nunique() == 96
    assert set(images.panel_id) == set(panels.panel_id) == {"spora_imc_42", "spora_imc_45"}
    assert len(fields) >= len(CLINICAL_COLUMNS)
    assert (reviews.loc[reviews.check_id == "D01", "result"] == "unresolved").any()
    assert all((ROOT / path).is_file() for path in images.input_image_path)
    assert not (JOB / "excluded_images.json").exists()

    comparisons = []
    for identity, group in images.groupby("panel_id"):
        padded = group.loc[(group.pad_left_px + group.pad_top_px +
                            group.pad_right_px + group.pad_bottom_px) > 0]
        choices = padded if len(padded) else group
        image = choices.loc[(choices.source_x1_px * choices.source_y1_px).idxmin()]
        selected = panels.loc[(panels.panel_id == identity) & panels.keep_channel].sort_values("output_channel_index")
        indices = selected.source_channel_index.astype(int).tolist()
        source = source_path(image.image_id)
        width, height = int(image.source_x1_px), int(image.source_y1_px)
        spec = SourceSpec(image.image_id, identity, "zarr", source / "0",
                          tuple(range(int(image.source_channel_count))),
                          pixel_size_um=(1.0, 1.0), source_revision=SOURCE_REVISION)
        with formal_read_lease(spec.path):
            with open_source(spec) as reader:
                expected = reader.read_window(indices, (0, 0, width, height))
        destination = ROOT / image.input_image_path
        with formal_read_lease(destination), resource_lock(OUTPUT, shared=True), resource_lock(
                destination.parent, shared=True, name=".input.lock"):
            with tifffile.TiffFile(destination) as tif:
                actual = tif.asarray(maxworkers=2)
        top, left = int(image.pad_top_px), int(image.pad_left_px)
        assert actual.shape == (len(indices), int(image.height_px), int(image.width_px))
        assert actual.dtype == expected.dtype == np.dtype("float32")
        assert np.array_equal(actual[:, top:top+height, left:left+width], expected)
        assert not actual[:, :top].any() and not actual[:, top+height:].any()
        assert not actual[:, :, :left].any() and not actual[:, :, left+width:].any()
        step = max(1, int(np.ceil(actual.shape[-1] / 768)))
        names = selected.source_name.tolist()

        def preview(name: str) -> np.ndarray:
            values = actual[names.index(name), ::step, ::step]
            positive = values[values > 0]
            scale = np.percentile(positive, 99) if len(positive) else 1.0
            return np.clip(values / max(float(scale), 1e-12) * 255, 0, 255).astype(np.uint8)

        overview = np.stack([preview("CK7"), preview("CD45"), preview("DNA1")], axis=-1)
        overview_path = JOB / f"overview_{identity}.png"
        Image.fromarray(overview).save(overview_path)
        comparisons.append(dict(panel_id=identity, image_id=image.image_id,
                                source_shape=list(expected.shape), output_shape=list(actual.shape),
                                compared_channels=indices, nonzero_source_pixels=int(np.count_nonzero(expected)),
                                full_pixels_equal=True, padding_zero=True,
                                overview_path=relative(overview_path)))
    atomic_json(JOB / "self_check.json", dict(study=STUDY, identity="preparer_self_check",
               source_images=len(tissues), exported_images=len(images), sample_rows=len(samples),
               sample_linked_images=int(images.sample_id.notna().sum()),
               patient_linked_samples=int(samples.patient_key.ne("").sum()),
               image_files=len(images), panel_count=panels.panel_id.nunique(),
               field_mapping_rows=len(fields), review_rows=len(reviews), comparisons=comparisons,
               independent_review="pending"))
    print(json.dumps(dict(verified_images=len(images), pixel_comparisons=len(comparisons))), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Run preparer-only source and output self-check")
    parser.add_argument("--repair-m02", action="store_true", help="Repair seven reviewed sample field mappings")
    parser.add_argument("--repair-m02-join", action="store_true", help="Repair reviewed sample join rules")
    arguments = parser.parse_args()
    if arguments.repair_m02_join:
        repair_m02_join()
    elif arguments.repair_m02:
        repair_m02()
    elif arguments.verify:
        verify()
    else:
        main()
