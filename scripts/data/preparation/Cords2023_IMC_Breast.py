"""Deliver Cords 2023's 109 published IMC ROIs as native float32 CYX TIFFs.

Inputs: local spora OME-Zarr, accepted native proofs, S1, and the original
Zenodo Image/clinical/acquisition tables cached from cp-output_config.zip.
The author read script explicitly defines SampleID separately from ROI/acID.
Outputs: independent input.tiff, 39-field image index, biological sample table,
panel/field mappings, preparer checks and resumable per-image receipts.
Use --export under the batch shared formal write lease wrapper; --verify under read.
The small native ROIs (at most 156 MB) are read once to avoid repeatedly
decoding the all-channel Zarr shards in the channel-major shared TIFF writer.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import io
import json
import os
import shutil
import struct
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import tifffile
import zarr
from numcodecs import blosc
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
STUDY = "Cords2023_IMC_Breast"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
RAW = NAS / "raw/public" / STUDY / "spora"
OUTPUT = NAS / "processed" / STUDY
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
CP = META / "source_metadata" / "cp_metadata/cp_30_new"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PROOF = META / "native_verification.json"
SNAPSHOT = META / "file_evidence.json"
OBJECTS = META / "imc_source_objects.json"
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0024_prepare_0/progress.json"
PANEL = "spora_IMC_39"
PREPARER = "0024_prepare_0"
REQUESTS = [
    "请由不同CLI上下文独立复核109张输入及两级metadata，完成D01绑定；本记录仅为整理者自检。",
    "本study临时缓存不超过156MB的单ROI以复用Zarr打包全部39通道的原生块；来源解释保留在脚本，后续可按实际需求将有界缓存迁入共享读取层。",
]


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress(phase: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, completed_images=completed,
                             expected_images=109, gaps=["待独立复核与D01验收绑定"],
                             updated_at=datetime.now(timezone.utc).isoformat(), **details))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def fetch_metadata() -> None:
    """Confirm the already received source metadata without a network request."""
    assert CP.is_dir(), "Missing received clinical metadata"


def build_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Connect explicit original SampleID/ROI and clinical tables to spora IDs."""
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    cp_images = pd.read_csv(CP / "Image.csv", keep_default_na=False)
    clinical = pd.read_csv(CP / "clinical_data.csv", dtype=str, keep_default_na=False)
    cp_images["original_file_name"] = cp_images.FileName_FullStack.str.replace(
        "_ac_full.tiff", "_ac.ome.tiff", regex=False)
    joined = tissues.merge(cp_images, on="original_file_name", validate="one_to_one", how="left",
                           indicator=True)
    assert len(joined) == 109 and joined._merge.eq("both").all()
    # This exact fourth-token rule is the author's SampleID assignment, not a patient guess.
    joined["source_sample_id"] = joined.FileName_FullStack.str.split("_").str[3]
    joined = joined.merge(clinical, left_on="source_sample_id", right_on="SampleID",
                          validate="many_to_one", how="left", indicator="clinical_join")
    assert joined.clinical_join.eq("both").all()
    assert (joined.original_patient_identifier == joined.Metadata_acname).all()
    assert (joined.source_sample_id == "T" + joined["Patient ID"]).all()
    assert joined.groupby("source_sample_id").patient_id.nunique().eq(1).all()
    assert joined.groupby("patient_id").source_sample_id.nunique().eq(1).all()
    joined["sample_id"] = "sample_" + joined.source_sample_id
    # Public author supplement confirms FFPE primary breast specimens and separate ROIs.
    author = (META / "source_metadata/author_read_imc.Rmd").read_text()
    assert 'cell_meta$SampleID <- ac_info[cell_meta$ImageNumber,4]' in author
    paper = " ".join(ET.parse(META / "source_metadata/paper.xml").getroot().itertext())
    assert "FFPE" in paper and "primary breast cancer samples" in paper
    for name in ["Gender", "Histopathology", "Clinical Subtype", "Grade_x", "T Staging",
                 "N Staging", "M Staging", "Neoadjuvant Therapy Received", "Treatment", "Grade_y"]:
        assert joined.groupby("sample_id")[name].nunique(dropna=False).eq(1).all()
    raw_extensions = {
        "source_age_range_at_surgery": "Age range at Surgery",
        "source_health_status": "Health Status",
        "source_clinical_subtype": "Clinical Subtype",
        "source_er_status_by_ihc": "ER Status by IHC",
        "source_er_percent_positive_cells_by_ihc": "ER percent positive cells by IHC",
        "source_pr_status_by_ihc": "PR Status by IHC",
        "source_pr_percent_positive_cells_by_ihc": "PR percent positive cells by IHC",
        "source_her2_status_by_ihc": "HER2 Status by IHC",
        "source_her2_status_by_fish_or_sish": "HER2 Status by FISH or SISH",
        "source_her2_ihc_score": "HER2 IHC Score",
        "source_menopause_status": "Menopause Status",
        "source_neoadjuvant_therapy_received": "Neoadjuvant Therapy Received",
        "source_previous_cancer_incidences": "Previous Cancer Incidences",
        "source_resection": "Resection",
        "source_er_irs_score": "ER IRS Score",
        "source_pr_irs_score": "PR IRS Score",
        "source_grade_comment": "Grade comment",
        "source_ki67_percent_positive_cells_by_ihc": "Ki-67 percent positive cells by IHC",
        "source_lymph_node_status_l": "Lymph Node Status (L)",
        "source_date": "Date",
    }
    sample_rows = []
    for _, group in joined.groupby("sample_id", sort=True):
        source = group.iloc[0]
        assert str(source.Grade_y) == str(source.Grade_x).removeprefix("G")
        no_treatment = source["Neoadjuvant Therapy Received"] == "no"
        assert (source.Treatment == "No") == no_treatment
        # Preserve specific source subtype wording, including the author's ER-negative qualifier.
        row = dict.fromkeys(columns("sample_metadata"), None)
        row.update(study=STUDY, sample_id=source.sample_id,
                   patient_key=f"{STUDY}:{source.patient_id}", source_subject_id=source["Patient ID"],
                   cohort_id="cords2023cancer", source_sample_id=source.source_sample_id,
                   biospecimen_group_id=source.source_sample_id, sample_type="primary breast tumor FFPE section",
                   tissue_type="tumor", anatomical_site="breast", lesion_type="primary",
                   preservation_method="ffpe", case_control_status="case", sex_reported="female",
                   primary_diagnosis="breast cancer", primary_site="breast", histology=source.Histopathology,
                   tnm_t=source["T Staging"], tnm_n=source["N Staging"],
                   tnm_m=None if source["M Staging"] == "Mx" else source["M Staging"],
                   grade=source.Grade_x, molecular_subtype=source.Subtype,
                   prior_systemic_therapy=not no_treatment,
                   treatment_status_at_collection="no neoadjuvant therapy" if no_treatment else source.Treatment,
                   metadata_notes=json.dumps({"age_range_at_surgery": source["Age range at Surgery"],
                       "grade_comment": source["Grade comment"], "spora_clinical_subtype": source["Clinical Subtype"],
                       "source_M_staging": source["M Staging"],
                       "time_semantics": "Grade may be preneoadjuvant; no baseline/collection date inferred",
                       "sample_evidence": "Author SampleID rule + Image.csv + clinical_data.csv; spora specimen_id is an ROI enumeration"},
                       ensure_ascii=False))
        for target, source_field in raw_extensions.items():
            values = group[source_field].dropna()
            if pd.api.types.is_string_dtype(values.dtype):
                values = values[values.astype(str).str.strip().ne("")]
            unique_values = values.map(str).unique()
            assert len(unique_values) <= 1, (source.sample_id, source_field)
            row[target] = unique_values[0] if len(unique_values) else None
        assert group.Gender.eq("F").all()
        sample_rows.append(row)
    samples = pd.DataFrame(sample_rows)
    channel_table = pd.read_parquet(RAW / "imc/channels.parquet")
    names = channel_table.channel_name.tolist()
    duplicate_index = names.index("CD11b2")
    assert channel_table.iloc[duplicate_index].description == "CD11b is measured on two channels. We only keep the first one."
    panel = panel_from_s1(STUDY, PANEL, names, S1,
                          drop_indices={duplicate_index: "Source channels.parquet explicitly selects first CD11b measurement"})
    panel["source_original_channel_name"] = channel_table.original_channel_name
    panel["source_description"] = channel_table.description
    availability = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
    assert set(availability.index) == set(joined.tissue_id)
    assert list(availability.columns) == names and availability.to_numpy().all()
    fields = []

    def mapping(source_file: Path, source_field: str, table: str, target: str, rule: str,
                *, values: dict | None = None, notes: str = "") -> None:
        row = dict.fromkeys(columns("field_mapping"), "")
        row.update(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_file.name,
                   source_field=source_field, target_table=table, target_field=target,
                   join_keys="tissues.original_file_name=Image.FileName_FullStack after _ac suffix replacement; author SampleID=clinical.SampleID",
                   value_mapping=json.dumps(values or {}, ensure_ascii=False), transformation_rule=rule,
                   time_origin_rule="No event origin or exact age inferred", conflict_rule="Require unique consistent joins; preserve author clinical subtype qualifier",
                   notes=notes)
        fields.append(row)

    image_fields = {
        "study": ("dataset", "Canonical study ID for cords2023cancer"),
        "image_id": ("tissue_id", "Verbatim spora tissue ID; one independent ROI"),
        "sample_id": ("FileName_FullStack; clinical.SampleID", "Author fourth-token SampleID; prefix sample_; multiple ROIs share one sample"),
        "panel_id": ("channels.parquet; embedded omero.channels", "All images have the same exact ordered 39-channel source panel"),
        "image_kind": ("FileName_FullStack; Metadata_acid", "One source ROI -> roi"),
        "source_image_id": ("tissue_id", "Verbatim source tissue ID"),
        "source_files": ("tissue_id; imc/channels.parquet", "NAS-relative original Zarr root plus original panel table; JSON/list encoding"),
        "source_format": ("zarr.json", "Published OME-Zarr v3 -> ome-zarr"),
        "source_acquisition_id": ("Metadata_acid", "Exact original acquisition ID, joined by full original file name"),
        "source_z_index": ("dimension_names", "No Z axis in native CYX -> null"),
        "source_t_index": ("dimension_names", "No T axis in native CYX -> null"),
        "source_axes": ("dimension_names", "Native c,y,x -> CYX, no transpose or rotation"),
        "source_dtype": ("data_type", "Native float32, unchanged"),
        "source_revision": ("accepted native verification; reviewed_file_evidence", "NAS-relative accepted proof; current source size/mtime equals its accepted snapshot"),
        "slide_id": ("Metadata_acname; Metadata_slideid", "Full original acquisition session + _s + slide ID"),
        "core_id": ("image_kind", "ROI imaging; no TMA core identity -> null"),
        "roi_id": ("Metadata_roiid", "Verbatim original ROI ID as string"),
        "fov_id": ("image_kind", "No separate source FOV identity -> null"),
        "region_annotation": ("Metadata_Description", "Original ROI/TLS description, unchanged"),
        "source_x0_px": ("Width_FullStack", "Full native ROI -> 0"),
        "source_y0_px": ("Height_FullStack", "Full native ROI -> 0"),
        "source_x1_px": ("Width_FullStack; Metadata_MaxX; Zarr.shape[2]", "All three source widths agree; exclusive right bound"),
        "source_y1_px": ("Height_FullStack; Metadata_MaxY; Zarr.shape[1]", "All three source heights agree; exclusive bottom bound"),
        "pad_left_px": ("Width/Height_FullStack", "floor((max(H,W)-W)/2)"),
        "pad_top_px": ("Width/Height_FullStack", "floor((max(H,W)-H)/2)"),
        "pad_right_px": ("Width/Height_FullStack", "max(H,W)-W-pad_left_px"),
        "pad_bottom_px": ("Width/Height_FullStack", "max(H,W)-H-pad_top_px"),
        "height_px": ("Width/Height_FullStack", "max(H,W)"),
        "width_px": ("Width/Height_FullStack", "max(H,W)"),
        "source_channel_count": ("channels.parquet; Zarr.shape[0]", "39 source channels, exact embedded order"),
        "channel_count": ("S1; channels.description[CD11b2]", "37 retained channels; exclude PNAd and explicitly unselected second CD11b"),
        "pixel_size_x_um": ("ome.multiscales.datasets.coordinateTransformations; Metadata_AblationDistanceBetweenShotsX", "1 micrometer; original and published scale agree"),
        "pixel_size_y_um": ("ome.multiscales.datasets.coordinateTransformations; Metadata_AblationDistanceBetweenShotsY", "1 micrometer; original and published scale agree"),
        "input_image_path": ("tissue_id", "processed/<study>/images/<tissue_id>/input.tiff, relative to NAS"),
        "valid_pixels_path": ("published native raster", "Complete published rectangular ROI; native measured zeros preserved; padding recorded separately; null"),
        "intensity_description": ("data_type; published source pixels", "Unchanged published float32 values; no normalization"),
        "split_method": ("tissue_id; FileName_FullStack", "Already independent original ROI -> already_independent"),
        "split_evidence": ("FileName_FullStack; Metadata_acname; Metadata_acid", "Original Image.csv gives full ROI and unique sample/acquisition crosswalk"),
        "metadata_notes": ("original_file_name; specimen_id; channels.original_channel_name", "Preserve source provenance and explain spora per-ROI specimen_id semantics"),
    }
    for target, (source_field, rule) in image_fields.items():
        source_file = CP / "Image.csv"
        if target in {"study", "image_id", "source_image_id", "metadata_notes"}:
            source_file = RAW / "metadata/tissues.parquet"
        elif target in {"panel_id", "source_channel_count"}:
            source_file = RAW / "imc/channels.parquet"
        elif target == "channel_count":
            source_file = S1
        elif target == "source_revision":
            source_file = SNAPSHOT
        elif target in {"source_format", "source_z_index", "source_t_index", "source_axes", "source_dtype",
                        "source_files", "valid_pixels_path", "pixel_size_x_um", "pixel_size_y_um", "intensity_description"}:
            source_file = RAW / "imc/1_0mpp/images"
        mapping(source_file, source_field, "image_metadata", target, rule,
                notes="Spora identities/axes/dtype/scale also verified against tissues.parquet and each Zarr zarr.json; accepted scope in source_inventory.json")
    mapping(RAW / "metadata/tissues.parquet", "tissue_id; original_file_name; patient_id; Patient ID", "sample_metadata", "patient_key",
            "Explicit spora patient crosswalk, confirmed against original SampleID. 109/109 images -> 12 samples -> 12 patients.")
    mapping(CP / "Image.csv", "FileName_FullStack; Metadata_acname; Metadata_acid", "sample_metadata", "sample_id",
            "Author read script defines fourth underscore token as SampleID; merge clinical_data.SampleID; group all ROIs of same confirmed biological specimen. Do not use spora specimen_id as sample.")
    sample_sources = {"source_subject_id": "Patient ID", "histology": "Histopathology", "tnm_t": "T Staging",
                      "tnm_n": "N Staging", "tnm_m": "M Staging", "grade": "Grade",
                      "metadata_notes": "Age range at Surgery; Grade comment; M Staging; Clinical Subtype"}
    for target, source in sample_sources.items():
        mapping(RAW / "metadata/tissues.parquet", source, "sample_metadata", target,
                "Preserve source strings; Mx becomes null with raw value retained in notes; age interval is not converted to a point age; grade timing retained in notes.")
    for target, source in raw_extensions.items():
        mapping(RAW / "metadata/tissues.parquet", source, "sample_metadata", target,
                "Preserve the exact source value as text without conversion or clinical interpretation.",
                notes="Only a unique nonempty value within the exact image-to-sample group is retained.")
        if target == "source_date":
            fields[-1]["time_origin_rule"] = (
                "Source Date event/time origin is not established; do not interpret as collection, diagnosis, or surgery date."
            )
    mapping(CP / "clinical_data.csv", "Subtype", "sample_metadata", "molecular_subtype",
            "Preserve author's exact subtype; spora's coarser subtype retained in notes.")
    mapping(CP / "clinical_data.csv", "Treatment; Grade", "sample_metadata", "prior_systemic_therapy",
            "No -> false; yes/Yes (ECPaclitaxel) -> true. Require agreement with spora neoadjuvant flag and Grade. No treatment-origin or baseline endpoint inferred.")
    mapping(RAW / "metadata/tissues.parquet", "Gender", "sample_metadata", "sex_reported", "Source F -> female", values={"F": "female"})
    mapping(META / "source_metadata/paper.xml", "Methods: patient cohort; Tissue preparation and staining", "sample_metadata", "preservation_method",
            "Primary breast cancer tumor sections are FFPE; cohort has 12 IMC patients. Set tumor/breast/primary/ffpe/case; no stage group or collection date inferred.")
    for target in ["study", "cohort_id", "source_sample_id", "biospecimen_group_id", "sample_type",
                   "tissue_type", "anatomical_site", "lesion_type", "case_control_status", "primary_diagnosis", "primary_site"]:
        mapping(META / "source_metadata/author_read_imc.Rmd", "author SampleID assignment; article patient cohort and tissue preparation", "sample_metadata", target,
                "Source SampleID identifies one primary breast tumor specimen, shared by all its ROIs. Constants: canonical study/cohort, primary breast tumor FFPE section, tumor, breast, primary, case, breast cancer. source_sample_id and biospecimen_group_id retain exact author SampleID.")
    mapping(CP / "clinical_data.csv", "Treatment", "sample_metadata", "treatment_status_at_collection",
            "No -> no neoadjuvant therapy; other source strings retained verbatim; corroborated by spora neoadjuvant flag")
    mapping(CP / "Image.csv", "Metadata_Description", "image_metadata", "region_annotation", "Retain verbatim original ROI/TLS annotation")
    mapping(RAW / "imc/channels.parquet", "channel_name; original_channel_name; description", "panel_mapping", "source_name",
            "Embedded spora channel_name defines actual stored order; original_channel_name separately preserves original CD3e/vWF-CD31 identity. S1 exact study-specific spora labels; no cross-study aliases.")
    mapping(RAW / "imc/channels.parquet", "description[CD11b2]", "panel_mapping", "keep_channel",
            "Drop second CD11b as explicitly recommended by source; exclude S1 PNAd; retain 37/39 channels including DNA1/DNA2.")
    return joined, samples, panel, pd.DataFrame(fields)


def inventory(joined: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Reuse accepted native proofs by matching file size/mtime and actual metadata."""
    proof = json.loads(PROOF.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    objects = json.loads(OBJECTS.read_text())
    assert proof["full_source_covered"] and not proof["failed"] and proof["verified_count"] == 109
    for name, old in snapshot["context"]:
        stat = (NAS / name).stat()
        assert [stat.st_size, stat.st_mtime_ns] == old, name
    accepted = {item["image_id"]: item for item in proof["verified"]}
    roots = RAW / "imc/1_0mpp/images"
    assert {p.name.removesuffix(".ome.zarr") for p in roots.glob("*.ome.zarr")} == set(accepted) == set(joined.tissue_id)
    output_bytes = 0
    for source in joined.itertuples():
        item = accepted[source.tissue_id]
        native = roots / f"{source.tissue_id}.ome.zarr"
        matching = sorted((x for x in objects if f"/{source.tissue_id}.ome.zarr/" in x["key"]), key=lambda x: x["key"])
        stats = []
        for member in matching:
            suffix = member["key"].split("/imc/1_0mpp/images/", 1)[1]
            stat = (roots / suffix).stat()
            assert stat.st_size == member["size"]
            stats.append([stat.st_size, stat.st_mtime_ns])
        assert stats == snapshot["images"][source.tissue_id]["file_stats"], source.tissue_id
        metadata = json.loads((native / "zarr.json").read_text())["attributes"]["ome"]
        labels = [channel["label"] for channel in metadata["omero"]["channels"]]
        assert labels == panel.source_name.tolist() == item["canonical_channel_labels"]
        assert metadata["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
        shape = json.loads((native / "0/zarr.json").read_text())["shape"]
        assert shape == item["shape"] == [39, source.Height_FullStack, source.Width_FullStack]
        assert shape[1:] == [source.Metadata_MaxY, source.Metadata_MaxX]
        output_bytes += int(panel.keep_channel.sum()) * max(shape[1:]) ** 2 * 4
    report = dict(images=109, source_objects=len(objects), native_proof=relative(PROOF),
                  proof_reuse="all source size/mtime and context unchanged; no full hash",
                  source_bytes=sum(x["size"] for x in objects), maximum_uncompressed_output_bytes=output_bytes,
                  sample_count=int(joined.sample_id.nunique()), patient_count=int(joined.patient_id.nunique()),
                  biological_sample_basis="original author SampleID + clinical_data.csv; multiple ROIs share sample_id",
                  additional_original_acquisition_rows=len(pd.read_csv(CP / "acquisition_metadata.csv")),
                  scope="All 109 published local spora ROIs; original acquisition list is supporting metadata, not extra local pixels")
    original = pd.read_csv(CP / "acquisition_metadata.csv")
    keys = joined[["Metadata_acname", "Metadata_acid"]]
    comparison = original.merge(keys, left_on=["AcSession", "AcquisitionID"],
                                right_on=["Metadata_acname", "Metadata_acid"], how="left", indicator=True)
    report["original_acquisitions_without_published_local_image"] = comparison.loc[
        comparison._merge == "left_only", ["AcSession", "AcquisitionID", "Description", "MaxX", "MaxY"]].to_dict("records")
    report["supplementary_followup"] = relative(CACHE / "source_followup.json")
    write_json(CACHE / "source_inventory.json", report)
    return accepted


class CachedROI:
    """A single small ROI; all-channel native shard decode is performed once."""

    def __init__(self, source: SourceSpec):
        self.source = source
        with open_source(source) as reader:
            self.shape, self.dtype = reader.shape, reader.dtype
            self.values = reader.read_window(list(range(self.shape[0])), (0, 0, self.shape[2], self.shape[1]))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def thumbnail(reader: CachedROI, name: str) -> None:
    """Display only: RGB=PanCK/Vimentin/DNA1, separately scaled at P99."""
    planes = reader.values[[34, 22, 35], ::4, ::4]
    rgb = np.zeros((*planes.shape[1:], 3), dtype=np.uint8)
    for index, plane in enumerate(planes):
        maximum = np.percentile(plane, 99)
        if maximum > 0:
            rgb[:, :, index] = (np.sqrt(np.clip(plane / maximum, 0, 1)) * 255).astype(np.uint8)
    destination = CACHE / "overviews" / f"{name}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(destination)


def make_sheets(image_ids: list[str]) -> None:
    for start in range(0, len(image_ids), 30):
        sheet = Image.new("RGB", (1100, 1440), "#181818")
        draw = ImageDraw.Draw(sheet)
        for index, image_id in enumerate(image_ids[start:start+30]):
            x, y = (index % 5)*220, (index//5)*240
            with Image.open(CACHE / "overviews" / f"{image_id}.png") as picture:
                picture.thumbnail((210, 210))
                sheet.paste(picture, (x, y))
            draw.text((x+2, y+212), image_id.removeprefix("cords2023cancer_"), fill="white")
        sheet.save(CACHE / "overviews" / f"sheet_{start//30+1}.jpg", quality=90)


def checks(images: pd.DataFrame, samples: pd.DataFrame, verification: dict | None = None) -> None:
    rows = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples():
        result = "pass"
        detail = {"preparer_self_check": True, "independent_review": "pending",
                  "exported_images": len(images), "sample_rows": len(samples),
                  "sample_linked_images": int(images.sample_id.notna().sum()),
                  "patient_linked_samples": int(samples.patient_key.notna().sum()),
                  "image_ids": images.image_id.tolist()}
        if check.check_id == "M01":
            detail["join_keys"] = "spora.original_file_name=Image.FileName_FullStack after suffix replacement; author fourth-token SampleID=clinical.SampleID; spora.patient_id and Patient ID provide explicit patient crosswalk"
            detail["identity_sources"] = [relative(CP / "Image.csv"), relative(CP / "clinical_data.csv"),
                                          relative(META / "source_metadata/author_read_imc.Rmd"), relative(META / "source_metadata/paper.xml"),
                                          relative(CACHE / "source_followup.json")]
            detail["sample_ids"] = samples.sample_id.tolist()
            detail["identity_gaps"] = []
        if check.check_id in {"S01", "I04"} and (CACHE / "overview_inspection.json").exists():
            detail["overview_inspection"] = relative(CACHE / "overview_inspection.json")
        if check.check_id == "D01" or (check.check_id == "I05" and verification is None):
            result = "unresolved"
            detail["reason"] = "待独立复核与D01绑定" if check.check_id == "D01" else "像素抽查尚未执行"
        if check.check_id == "M02":
            detail["limitations"] = "年龄仅有区间；无确切采样时点、生存终点或stage group；Grade可能为新辅助治疗前评估，已保留原注释。"
        if check.check_id == "I05" and verification:
            detail["pixel_verification"] = verification
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                         record_id=STUDY if check.scope != "panel" else PANEL, check_id=check.check_id,
                         result=result, source_evidence=relative(CACHE / "source_inventory.json") + ";" + relative(CP / "Image.csv"),
                         output_evidence=relative(OUTPUT / "images.parquet"),
                         details=json.dumps(detail, ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(rows, columns=columns("review_results")).to_csv(review_path, index=False, encoding="utf-8-sig")


# Resolve original metadata and bind the exact local, previously accepted scope.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fetch-metadata", action="store_true")
parser.add_argument("--export", action="store_true")
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
if args.fetch_metadata:
    fetch_metadata()
joined, samples, panel, field_mapping = build_metadata()
accepted = inventory(joined, panel)
OUTPUT.mkdir(parents=True, exist_ok=True)
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
progress("source_and_sample_linkage_confirmed", len(images), sample_rows=len(samples), patient_linked_samples=len(samples))
if not args.export and not args.verify:
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        for name, table in [("sample_metadata", samples), ("panel_mapping", panel), ("field_mapping", field_mapping)]:
            table.to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")

# Incremental publication: each successful ROI has a TIFF, receipt and real index.
if args.export:
    required = json.loads((CACHE / "source_inventory.json").read_text())["maximum_uncompressed_output_bytes"]
    assert shutil.disk_usage(NAS).free > required + 100*1024**3
    receipts = CACHE / "export_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    panel_binding = panel.loc[panel.keep_channel, ["source_channel_index", "output_channel_index", "marker_id"]].to_dict("records")
    for source in joined.sort_values("tissue_id").to_dict("records"):
        image_id = source["tissue_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = receipts / f"{image_id}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt["panel_binding"] == panel_binding, "Stored TIFF channel selection changed; targeted repair required"
            if receipt["admitted"]:
                assert destination.exists() and destination.stat().st_size == receipt["output_size"]
                row = receipt["image_metadata"]
                images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
            continue
        progress("exporting", len(images), current_image=image_id)
        path = RAW / "imc/1_0mpp/images" / f"{image_id}.ome.zarr/0"
        spec = SourceSpec(image_id, PANEL, "zarr", path, tuple(range(39)), pixel_size_um=(1., 1.))
        reader = CachedROI(spec)
        thumbnail(reader, image_id)
        result = export_input(reader, panel, destination, tile_size=256)
        row = dict.fromkeys(columns("image_metadata"), None)
        signal_xy = None
        if result["admitted"]:
            c, h, w = reader.shape
            row.update(study=STUDY, image_id=image_id, sample_id=source["sample_id"], panel_id=PANEL,
                       image_kind="roi", source_image_id=image_id,
                       source_files=[relative(path.parent), relative(RAW / "imc/channels.parquet")],
                       source_format="ome-zarr", source_acquisition_id=int(source["Metadata_acid"]),
                       source_axes="CYX", source_dtype="float32", source_revision=relative(PROOF),
                       slide_id=source["Metadata_acname"] + "_s" + str(source["Metadata_slideid"]),
                       roi_id=str(source["Metadata_roiid"]), region_annotation=source["Metadata_Description"],
                       source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                       **{key: result[key] for key in ["pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "height_px", "width_px", "source_channel_count", "channel_count"]},
                       pixel_size_x_um=1., pixel_size_y_um=1., input_image_path=relative(destination),
                       intensity_description="Published spora IMC float32 intensities; unchanged decoded values; no normalization",
                       split_method="already_independent", split_evidence=relative(CP / "Image.csv"),
                       metadata_notes=json.dumps({"original_file_name": source["original_file_name"],
                           "source_split": source["split"],
                           "spora_specimen_id": int(source["specimen_id"]), "spora_specimen_semantics": "per-ROI ordinal, not biological sample",
                           "source_panel_identity": "spora uses CD3/CD31 labels; original source CD3e/vWF-CD31 retained in panel",
                           "acquisition_validity": "complete published raster; native measured zeros preserved"}, ensure_ascii=False))
            images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
            kept = panel[panel.keep_channel].sort_values("output_channel_index")
            measured = next(channel for channel in kept.itertuples()
                            if channel.role == "downstream" and result["channel_nonzero_pixels"][channel.output_channel_index] > 0)
            positions = np.flatnonzero(reader.values[measured.source_channel_index])
            y, x = divmod(int(positions[len(positions)//2]), w)
            signal_xy = [x, y]
        write_json(receipt_path, dict(**result, image_metadata=row if result["admitted"] else None,
                                     panel_binding=panel_binding,
                                     signal_xy=signal_xy,
                                     output_size=destination.stat().st_size if result["admitted"] else 0,
                                     exclusion_reason=None if result["admitted"] else "All admitted marker measurements are zero over the entire ROI"))
        save_input_tables(OUTPUT, images, samples, panel, field_mapping)
        checks(images, samples)
        progress("exporting", len(images), last_completed_image=image_id)
        print(f"{len(images)}/109 {image_id}", flush=True)
        del reader
    make_sheets(sorted(joined.tissue_id))

# Check every TIFF header and representative retained-channel pixels, including all padding cases.
verification = None
if args.verify:
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    pixel_checks = []
    selected = set(images.loc[(images.pad_left_px > 0) | (images.pad_top_px > 0), "image_id"])
    selected.add(sorted(images.image_id)[0])
    for source in images.to_dict("records"):
        image_id = source["image_id"]
        with tifffile.TiffFile(NAS / source["input_image_path"]) as tif:
            assert len(tif.pages) == len(kept)
            assert all(page.dtype == np.dtype("float32") and page.shape == (source["height_px"], source["width_px"]) for page in tif.pages)
            root = ET.fromstring(tif.ome_metadata)
            assert [x.attrib["Name"] for x in root.iter() if x.tag.endswith("Channel")] == kept.marker_id.tolist()
            if image_id not in selected:
                continue
            path = RAW / "imc/1_0mpp/images" / f"{image_id}.ome.zarr/0"
            with open_source(SourceSpec(image_id, PANEL, "zarr", path, tuple(range(39)))) as reader:
                h, w = reader.shape[-2:]
                sx, sy = json.loads((CACHE / "export_receipts" / f"{image_id}.json").read_text())["signal_xy"]
                sx, sy = min(max(0, sx-16), w-32), min(max(0, sy-16), h-32)
                blocks = [(0, 0, 32, 32), (sx, sy, sx+32, sy+32)]
                for number, block in enumerate(blocks):
                    native = reader.read_window(kept.source_channel_index.tolist(), block)
                    if number == 1:
                        assert np.any(native), "The interior comparison block must contain native signal"
                    x0, y0, x1, y1 = block
                    pl, pt = source["pad_left_px"], source["pad_top_px"]
                    output_block = (x0+pl, y0+pt, x1+pl, y1+pt)
                    for index in range(len(kept)):
                        assert np.array_equal(native[index], read_tiff_window(tif, index, output_block))
                    pixel_checks.append(dict(image_id=image_id, source_block=block, output_block=output_block, channels=len(kept)))
                side = source["height_px"]
                strips = [(0, 0, side, source["pad_top_px"]), (0, side-source["pad_bottom_px"], side, side),
                          (0, 0, source["pad_left_px"], side), (side-source["pad_right_px"], 0, side, side)]
                for block in strips:
                    if block[0] < block[2] and block[1] < block[3]:
                        for index in range(len(kept)):
                            assert not np.any(read_tiff_window(tif, index, block))
    verification = dict(header_images=len(images), pixel_images=len(selected), pixel_blocks=pixel_checks,
                        native_dtype="float32", retained_channels=len(kept), padding="all padding strips checked in every padded image")
    write_json(CACHE / "preparer_verification.json", verification)

if len(images):
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)
    checks(images, samples, verification)
    summary = dict(study=STUDY, status="prepared" if len(images) == 109 else "partial", script=str(Path(__file__).relative_to(PROJECT)),
                   expected_images=109, exported_images=len(images), sample_rows=len(samples),
                   sample_linked_images=int(images.sample_id.notna().sum()),
                   patient_linked_samples=int(samples.patient_key.notna().sum()),
                   issues=["独立复核与D01绑定待主控安排；临床年龄区间、Grade时点及未知Mx已在metadata中保留。"],
                   controller_requests=REQUESTS, evidence_paths=[relative(CACHE / "source_inventory.json"), relative(CACHE / "preparer_verification.json"), relative(OUTPUT / "review_results.csv")])
    write_json(CACHE / "summary.json", summary)
    progress("prepared_pending_independent_review" if len(images) == 109 else "partial", len(images),
             sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=len(samples))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
