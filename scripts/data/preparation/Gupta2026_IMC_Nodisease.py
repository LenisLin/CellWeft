"""Deliver native SPORA IMC ROIs and specimen-linked metadata for Gupta 2026.

Inputs: the complete local gupta2026single release, its final native verification,
the study's S1 rows, and small original Zenodo clinical/guide attachments.
Outputs: processed/Gupta2026_IMC_Nodisease/{images.parquet,*_mapping.csv,
sample_metadata.csv,review_results.csv,images/*/input.tiff}; resumable NAS cache.
Set CELLWEFT_STUDY_INPUT_JOB to the NAS checkpoint directory. Run prepare,
export, then verify; shared helpers acquire a lease for each formal publication.
The recover action compares completed TIFFs and publishes missing index rows.
Independent review is assigned by the controller after these files are closed.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import Iterator
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Gupta2026_IMC_Nodisease/matplotlib"

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import tifffile
import zarr
from numcodecs import blosc
from xml.etree import ElementTree

from cellweft.data.contracts import ImageBlock, SourceSpec, Window
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Gupta2026_IMC_Nodisease"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "spora"
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
PREPARER = f"Gupta2026_preparer_{JOB.name}"
REPAIR_RECORD = f"{STUDY}:repair_1"
META = metadata_source_directory(ROOT, STUDY)
PROOF = META / "native_verification.json"
OBJECTS = META / "imc_source_objects.json"
CLINICAL = META / "source_metadata/zenodo_metadata/IMCBreastAgeingAtlasPublic/data/derived/clinical.parquet"
GUIDE = META / "source_metadata/zenodo_metadata/IMCBreastAgeingAtlasPublic/DataGuide.pdf"
ORIGINAL_PANEL = META / "source_metadata/zenodo_metadata/IMCBreastAgeingAtlasPublic/data/raw/panel/panel.csv"
READ_ORDER = ORIGINAL_PANEL.with_name("metalReadOrder.csv")
ARTICLE = "https://www.nature.com/articles/s43587-026-01104-3"
SPORA_SOURCE = "https://livingobjects.ebi.ac.uk/bioimaging-01-pub/spora-submissions/gupta2026single/"
ZENODO = "https://zenodo.org/records/18418221"
ARCHIVE = "https://zenodo.org/api/records/18418221/files/IMCBreastAgeingAtlasPublic.zip/content"
REPOSITORY = "https://github.com/HRazaAliLab/IMCBreastAgeingAtlas"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "imc_41_spora"
REVISION = "SPORA BIA-SPORA-010 gupta2026single, localized 2026-10-18"
zarr.config.set({"async.concurrency": 4, "threading.max_workers": 4})
blosc.set_nthreads(1)
pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, ROOT))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def progress(phase: str, completed: int, **details: object) -> None:
    write_json(JOB / "progress.json", dict(study=STUDY, phase=phase,
               completed_images=completed, expected_images=1492,
               updated_at=pd.Timestamp.now(tz="UTC").isoformat(), **details))


def publish_preparer_checks(additions: pd.DataFrame) -> None:
    """Update only this repair's self-check rows, preserving independent history."""
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=path.parent):
        review = pd.read_csv(path, keep_default_na=False) if path.exists() else additions.iloc[:0]
        replaced = (review.preparer.eq(PREPARER) & review.reviewer.eq("")
                    & review.record_id.eq(REPAIR_RECORD)
                    & review.check_id.isin(additions.check_id))
        result = pd.concat([review.loc[~replaced], additions], ignore_index=True)
        temporary = path.with_name(path.name + ".partial")
        result.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


class RemoteArchive(io.RawIOBase):
    """Read only bounded ZIP metadata members using the published archive size."""

    def __init__(self, size: int):
        self.position, self.size = 0, size

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = min(self.size, self.position + size) if size >= 0 else self.size
        if end <= self.position:
            return b""
        with requests.get(ARCHIVE, headers={"Range": f"bytes={self.position}-{end-1}"},
                          params={"range_start": self.position}, timeout=60, stream=True) as response:
            if response.status_code != 206:
                raise ValueError(f"Metadata extraction requires HTTP 206; got {response.status_code}")
            data = response.content
        if len(data) != end - self.position:
            raise ValueError("Incomplete bounded archive response")
        self.position = end
        return data


def source_metadata() -> None:
    """Require the accepted clinical, guide and original panel members."""
    for source in (CLINICAL, GUIDE, ORIGINAL_PANEL, READ_ORDER):
        if not source.is_file():
            raise FileNotFoundError(source)


class DecodedROI:
    """Adapt one small CYX ROI decoded once to the shared tiled TIFF exporter.

    SPORA shards pack all 41 channels together. Decoding them separately for every
    output plane repeats source I/O and decompression. The verified largest ROI
    occupies 179.13 MiB, so one ROI is held at a time; whole-slide loading is absent.
    This source-layout optimization can later move to the shared reader cache.
    """

    def __init__(self, values: np.ndarray, source: SourceSpec):
        self.values = values
        # The source was read under its own lease; the exporter now uses only this memory array.
        self.source = SimpleNamespace(path=Path("/tmp/study_input_batch"),
                                      pixel_size_um=source.pixel_size_um)
        self.shape, self.dtype = values.shape, values.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def iter_blocks(self, channels: list[int], block_size: int) -> Iterator[ImageBlock]:
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


def source_spec(row: pd.Series) -> SourceSpec:
    return SourceSpec(row.tissue_id, PANEL_ID, "zarr",
                      RAW / "imc/1_0mpp/images" / f"{row.tissue_id}.ome.zarr/0",
                      tuple(range(41)), pixel_size_um=(1.0, 1.0), source_revision=REVISION)


def inspect_sources(tissues: pd.DataFrame, panel_names: list[str]) -> pd.DataFrame:
    """Reuse final native proofs and compare current source object sizes, without hashes."""
    progress("checking_source_objects", 0, gaps=["正在按最终清单核对完整成员大小；既有核验复用，不重算hash"])
    proof = json.loads(PROOF.read_text())
    verified = {x["image_id"]: x for x in proof["verified"]}
    assert set(verified) == set(tissues.tissue_id)
    prefix = "spora-submissions/gupta2026single/imc/1_0mpp/images/"
    objects: dict[str, list[dict]] = {}
    for item in json.loads(OBJECTS.read_text()):
        suffix = item["key"].removeprefix(prefix)
        objects.setdefault(suffix.split("/")[0].removesuffix(".ome.zarr"), []).append(
            dict(path=RAW / "imc/1_0mpp/images" / suffix, size=item["size"]))
    rows = []
    for row in tissues.itertuples(index=False):
        record = verified[row.tissue_id]
        assert record["canonical_channel_labels"] == panel_names
        assert record["shape"] == [41, row.image_height, row.image_width]
        assert record["all_source_sizes_match"] and record["exact_canonical_order_checked"]
        missing = [relative(x["path"]) for x in objects[row.tissue_id]
                   if not x["path"].is_file() or x["path"].stat().st_size != x["size"]]
        rows.append(dict(image_id=row.tissue_id, complete=not missing, missing=missing,
                         shape=record["shape"], dtype=record["dtype"],
                         source_object_count=len(objects[row.tissue_id])))
        if len(rows) % 100 == 0:
            progress("checking_source_objects", 0, source_images_checked=len(rows),
                     gaps=["尚待逐image导出；metadata连接已核实"])
    inventory = pd.DataFrame(rows)
    write_json(CACHE / "source_inventory.json", dict(
        final_native_verification=relative(PROOF), source_objects=relative(OBJECTS),
        object_count=int(inventory.source_object_count.sum()), images=rows,
        expected_uncompressed_square_bytes=sum(max(x["shape"][1:])**2*41*2 for x in rows),
        largest_decoded_roi_bytes=max(np.prod(x["shape"])*2 for x in rows).item(),
        checked_at=pd.Timestamp.now(tz="UTC").isoformat()))
    return inventory


def metadata_tables(tissues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confirm the original core map and build one row per donor mammoplasty specimen."""
    clinical = pd.read_parquet(CLINICAL)
    joined = tissues.merge(clinical, left_on="original_file_name", right_on="ImageID",
                           validate="one_to_one", how="left", indicator=True)
    assert joined._merge.eq("both").all()
    assert joined.original_patient_identifier.eq(joined.PatientID).all()
    assert (joined.age.eq(joined.Age) | (joined.age.isna() & joined.Age.isna())).all()
    assert joined.image_width.eq(joined.ImageWidth).all() and joined.image_height.eq(joined.ImageHeight).all()
    assert joined.groupby("patient_id").PatientID.nunique().eq(1).all()
    assert joined.groupby("patient_id").Age.nunique().le(1).all()
    samples = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str)
    rows = []
    for patient, group in joined.groupby("patient_id", sort=True):
        row = {name: pd.NA for name in samples.columns}
        age = group.Age.dropna()
        row.update(study=STUDY, sample_id=f"mammoplasty_{patient}",
                   patient_key=f"{STUDY}:{patient}", source_subject_id=group.PatientID.iloc[0],
                   cohort_id="normal_breast_reduction_mammoplasty", sample_type="reduction mammoplasty tissue",
                   tissue_type="normal", anatomical_site="Breast",
                   disease_status_at_collection="no_evidence_of_disease", collection_method="resection",
                   preservation_method="ffpe", processing_method="1.5-mm TMA cores; IMC tissue section",
                   age_at_collection_years=float(age.iloc[0]) if len(age) else pd.NA,
                   case_control_status="reference", sex_reported="female",
                   metadata_notes="Same donor mammoplasty specimen across TMA cores, supported by paper Methods and author README. "
                   "No separate surgical specimen accession, side or within-breast location is published. "
                   "Clinical age is age at tissue acquisition. " + ("Source age is missing." if age.empty else ""))
        rows.append(row)
    samples = pd.DataFrame(rows, columns=samples.columns)
    mappings = []

    def mapping(source: str, target_table: str, target: str, rule: str,
                file: Path = CLINICAL, notes: str = "",
                join_keys: str = "tissues.original_file_name = clinical.ImageID; tissues.original_patient_identifier = clinical.PatientID") -> None:
        mappings.append(dict(study=STUDY, source_file=relative(file), source_sheet_or_table=file.stem,
                        source_field=source, target_table=target_table, target_field=target,
                        join_keys=join_keys,
                        value_mapping="{}", transformation_rule=rule,
                        time_origin_rule="Age is at donor sample acquisition (DataGuide p5)" if target == "age_at_collection_years" else "",
                        conflict_rule="Require exact image/patient/dimension/nonmissing age agreement, isotope/SPORA order and S1 marker identity; fail on conflicts",
                        notes=notes))

    mapping("ImageID;PatientID;patient_id", "image_metadata", "sample_id",
            "ImageID one-to-one joins all 1492 tissue IDs. mammoplasty_<SPORA patient_id> groups cores from one donor specimen; "
            "SPORA specimen_id is a per-image ordinal and is not used as the biological sample ID.",
            notes="527 donor samples; 1492 image links; 527 patient links. Paper Methods and author README identify 527 mammoplasty samples.")
    for target, source, rule in [
        ("sample_id", "PatientID;patient_id", "mammoplasty_<SPORA patient_id>; one donor surgical specimen with multiple TMA cores"),
        ("patient_key", "PatientID;patient_id", "Study-scoped SPORA patient_id, cross-checked to original clinical.PatientID"),
        ("source_subject_id", "PatientID", "Preserve original public PatientID as a string, including leading zeros"),
        ("age_at_collection_years", "Age", "Preserve age at specimen acquisition; require within-patient consistency; 1 sample unknown"),
    ]:
        mapping(source, "sample_metadata", target, rule)
    for target, rule in {
        "cohort_id": "normal_breast_reduction_mammoplasty",
        "sample_type": "reduction mammoplasty tissue",
        "tissue_type": "normal", "anatomical_site": "Breast; subsite and side unavailable",
        "lesion_type": "Leave empty: individual donor lesion-type annotations are not provided",
        "disease_status_at_collection": "no_evidence_of_disease; refers to breast cancer-free donor cohort",
        "collection_method": "resection (reduction mammoplasty)", "preservation_method": "ffpe",
        "processing_method": "1.5-mm TMA cores; IMC tissue section", "case_control_status": "reference",
        "sex_reported": "female; cohort explicitly consists of women",
    }.items():
        mapping("Methods: Study design, TMA construction and metadata", "sample_metadata", target,
                rule, META / "source_metadata/article.html", join_keys="Study cohort; all published donor mammoplasty specimens")
    mapping("ImageID;DataGuide p3", "image_metadata", "core_id",
            "Remove only terminal _split from original_file_name: interrupted/resumed acquisitions share one physical core. "
            "Keep each released acquisition as a distinct image, preserving its full local rectangle.", GUIDE)
    mapping("tissue_id;original_file_name;specimen_id;split", "image_metadata", "metadata_notes",
            "Preserve source ROI ordinal, exact original ImageID and publisher train/test split in JSON; no new split assignment.",
            RAW / "metadata/tissues.parquet", join_keys="tissue_id = image_id")
    mapping("channel_name;channels_per_tissue", "panel_mapping", "source_channel_index",
            "Row order of channels.parquet filtered by measured mask; all 1492 images have the same 41-channel measured order.",
            RAW / "imc/channels.parquet", join_keys="channel table row order; channels_per_tissue.tissue_id = tissue_id")
    mapping("Metal;Full;FormattedName;metalReadOrder.csv", "panel_mapping", "source_channel_index,source_metal,source_name",
            "Join the original 41-line metal read order to panel.csv rows with Full=1; require FormattedName order exactly equal "
            "to all SPORA channels. Additional antibody/cocktail component rows share an isotope and are not separate image channels. "
            "FormattedName gives the intended marker, including amplification channels, rather than the secondary-antibody Marker label.",
            ORIGINAL_PANEL, join_keys="metalReadOrder.Metal = panel.Metal where Full=1; FormattedName = SPORA channel_name")
    mapping("Study;Original marker", "panel_mapping", "marker_id",
            "Exact study-specific S1 join; all 41 channels retained, DNA1/DNA2 preserve distinct nuclear identities.", S1,
            join_keys="S1.Study = study; S1.Original marker = panel.source_name")
    mapping("zarr /0 dimension_names and scale", "image_metadata", "source_x0_px,source_y0_px,source_x1_px,source_y1_px",
            "Use complete native rectangle; centered zero padding to max(height,width); retain each source dtype "
            "(1481 uint16 and 11 uint8 acquisitions) and native integer values at 1 um/pixel.",
            PROOF, join_keys="verification.verified.image_id = tissues.tissue_id")
    mapping("TissueArea", "image_metadata", "region_annotation",
            "Leave region_annotation empty: no categorical ROI annotation published. TissueArea is a cell-point convex hull estimate "
            "(DataGuide p5), and is not used to define a tissue mask or sample region.", GUIDE)
    mapping("clinical;DataGuide;publication;author repository", "sample_metadata", "metadata_notes",
            "Trace local tissues and original clinical attachments, paper supplementary Tables 1-5, DataGuide p3/p5 and author core-count code. "
            "Clinical fields beyond age, exact anatomical subsite, collection date and specimen accession are unavailable.",
            notes=f"{ARTICLE}; {ZENODO}; {REPOSITORY}; cache/ study attachments. All 1492/527/527 identity links confirmed.")
    write_json(CACHE / "metadata_trace.json", dict(study=STUDY, source_urls=[SPORA_SOURCE, ARTICLE, ZENODO, REPOSITORY],
        clinical_path=relative(CLINICAL), guide_path=relative(GUIDE),
        image_join="tissues.original_file_name = clinical.ImageID", patient_join="original_patient_identifier = PatientID",
        image_count=len(joined), sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
        source_split_acquisitions=int(joined.original_file_name.str.endswith("_split").sum()),
        missing_age_samples=int(samples.age_at_collection_years.isna().sum()),
        original_panel_order_check=dict(matched_channels=41, panel_path=relative(ORIGINAL_PANEL),
                                        read_order_path=relative(READ_ORDER), rule="Full=1 per isotope; intended marker is FormattedName"),
        sample_rule="One donor reduction-mammoplasty specimen per published donor; cores share sample_id; source specimen_id is an image ordinal",
        supplementary_tables="Tables 1 antibodies, 2 rationales, 3 proliferation, 4 patient counts, 5 phenotypes; no additional clinical/core map",
        unknown_optional_fields="No separate surgical accession, side, anatomical subsite, date, treatment or outcomes. One donor age missing."))
    return samples, pd.DataFrame(mappings)


def review_results(images: pd.DataFrame, samples: pd.DataFrame, complete: bool) -> None:
    """Record preparer checks without issuing an independent reviewer acceptance."""
    n = len(images)
    linked = int(images.sample_id.notna().sum()) if n else 0
    details = {"stage": "preparer_self_check", "expected_images": len(tissues), "exported_images": n,
               "sample_linked_images": linked, "sample_rows": len(samples),
               "patient_linked_samples": int(samples.patient_key.notna().sum()),
               "export_scope_complete": complete, "independent_review": "pending",
               "image_ids": images.loc[images.image_id.isin(repair_image_ids), "image_id"].tolist(),
               "scientific_repair_round": 1}
    rows = []
    checks = {
        "S01": ("study", relative(CACHE / "source_inventory.json"), "Full local 1492-image SPORA scope; source sizes and final native proofs reused; admission from full retained-marker signal scan."),
        "M01": ("sample", relative(CACHE / "metadata_trace.json"), "Original clinical ImageID->PatientID validates every image; 527 published donor mammoplasty specimens group multiple cores; see field_mapping."),
        "M02": ("sample", relative(CLINICAL), "All populated clinical fields reconstructed from clinical table and published Methods; one source age absent and optional fields remain empty."),
        "P01": ("panel", relative(S1), "Exact 41-channel source-specific S1 matches; no aliases."),
        "P02": ("panel", relative(S1), "41 channels retained in source order; DNA1 and DNA2 retain nuclear identity; downstream measurements unique."),
        "I01": ("study", relative(PROOF), "Unique source IDs and indexed file paths checked at every save."),
        "I02": ("study", relative(CLINICAL), "Every nonempty image sample_id resolves to sample_metadata; panel foreign keys checked by shared validator."),
        "I03": ("study", relative(PROOF), "Whole source rectangle, native 1 um/pixel and per-image uint16/uint8 intensity; centered square padding."),
        "I04": ("study", relative(GUIDE), "Already independent source core acquisitions; _split is a resumed part of the same physical core, not a new specimen."),
    }
    for check, (scope, evidence, message) in checks.items():
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                         record_id=REPAIR_RECORD, check_id=check,
                         result="unresolved" if check == "S01" and not complete else "pass", source_evidence=evidence,
                         output_evidence=relative(OUTPUT), details=json.dumps(dict(details, note=message), ensure_ascii=False)))
    rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                     record_id=REPAIR_RECORD, check_id="D01", result="unresolved", source_evidence=relative(PROOF),
                     output_evidence=relative(OUTPUT), details=json.dumps(dict(details,
                     note="整理者自检；导出范围见计数。待不同CLI上下文独立复核、绑定及主控验收。"), ensure_ascii=False)))
    publish_preparer_checks(pd.DataFrame(rows))


def save_tables(rows: list[dict]) -> pd.DataFrame:
    images = pd.DataFrame(rows, columns=image_columns)
    save_input_tables(OUTPUT, images, samples, panel, mappings)
    return images


def make_image(row: pd.Series, result: dict, destination: Path) -> dict:
    record = {name: None for name in image_columns}
    record.update(study=STUDY, image_id=row.tissue_id, sample_id=f"mammoplasty_{row.patient_id}",
                  panel_id=PANEL_ID, image_kind="tma_core", source_image_id=row.tissue_id,
                  source_files=[relative(destination_source(row))], source_format="zarr", source_axes="CYX",
                  source_dtype="uint16", source_revision=REVISION,
                  core_id=row.original_file_name.removesuffix("_split"), roi_id=row.original_file_name,
                  source_x0_px=0, source_y0_px=0, source_x1_px=int(row.image_width), source_y1_px=int(row.image_height),
                  pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                  intensity_description=f"Published SPORA /imc/1_0mpp measured-channel {result['source_dtype']} intensities preserved exactly; "
                  "no rescaling, clipping, spillover correction or normalization applied by this script. "
                  "DataGuide describes supplied 41-layer IMC stacks; five low-SNR channels had been removed upstream.",
                  split_method="already_independent",
                  split_evidence=f"{relative(RAW / 'metadata/tissues.parquet')}:tissue_id; "
                  f"{relative(CLINICAL)}:ImageID; DataGuide p3 (resumed _split acquisition identity)",
                  metadata_notes=json.dumps(dict(original_file_name=row.original_file_name,
                       source_specimen_id=int(row.specimen_id), source_split=row.split,
                       resumed_same_core=row.original_file_name.endswith("_split")), ensure_ascii=False))
    for key in ["height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px", "pad_bottom_px",
                "source_channel_count", "channel_count", "source_dtype"]:
        record[key] = result[key]
    return record


def destination_source(row: pd.Series) -> Path:
    return RAW / "imc/1_0mpp/images" / f"{row.tissue_id}.ome.zarr"


def check_header(path: Path, height: int, names: list[str], dtype: str | np.dtype) -> None:
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX" and tif.series[0].shape == (len(names), height, height)
        assert tif.series[0].dtype == np.dtype(dtype)
        ome = ElementTree.fromstring(tif.ome_metadata)
        assert [x.attrib["Name"] for x in ome.iter() if x.tag.endswith("}Channel")] == names
        pixels = next(x for x in ome.iter() if x.tag.endswith("}Pixels"))
        assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0


def reuse_unindexed_output(values: np.ndarray, destination: Path) -> dict:
    """Recover a completed native TIFF after a subsequent index/header check failed.

    The first uint8 acquisition exposed an incorrect uint16-only preparer check.
    Shared export had already saved the correct uint8 TIFF. Compare its complete
    pixel array, including padding, before adopting it without another export.
    """
    _, height, width = values.shape
    side = max(height, width)
    left, top = (side-width)//2, (side-height)//2
    with formal_read_lease(destination), resource_lock(OUTPUT, shared=True), resource_lock(
            destination.parent, shared=True, name=".input.lock"):
        check_header(destination, side, panel.marker_id.tolist(), values.dtype)
        actual = tifffile.imread(destination, maxworkers=2)
    assert np.array_equal(actual[:, top:top+height, left:left+width], values)
    assert not actual[:, :top].any() and not actual[:, top+height:].any()
    assert not actual[:, :, :left].any() and not actual[:, :, left+width:].any()
    nonzero = np.count_nonzero(values, axis=(1, 2))
    admitted = bool(nonzero[panel.role.eq("downstream").to_numpy()].any())
    assert admitted
    return dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                source_dtype=str(values.dtype), source_channel_count=values.shape[0],
                channel_count=len(panel), admitted=admitted,
                channel_nonzero_pixels=nonzero.tolist(),
                channel_means=values.mean(axis=(1, 2), dtype=np.float64).tolist(),
                valid_pixels=None, reused_after_full_pixel_comparison=True)


def verify_images(images: pd.DataFrame) -> None:
    """Check this job's new images without re-reading previously indexed images."""
    repaired = images[images.image_id.isin(repair_image_ids)]
    kept = panel.loc[panel.keep_channel]
    names = kept.marker_id.tolist()
    native_dtypes = inventory.set_index("image_id")["dtype"]
    assert repaired.source_dtype.eq(repaired.image_id.map(native_dtypes)).all()
    for checked, image in enumerate(repaired.itertuples(index=False), start=1):
        path = ROOT / image.input_image_path
        with formal_read_lease(path), resource_lock(OUTPUT, shared=True), resource_lock(
                path.parent, shared=True, name=".input.lock"):
            check_header(path, int(image.height_px), names, image.source_dtype)
        if checked % 100 == 0:
            progress("checking_repaired_headers", len(images), checked_headers=checked,
                     repaired_images=len(repaired))

    source_rows = tissues[tissues.tissue_id.isin(repaired.image_id)]
    selected = []
    if len(source_rows):
        selected = source_rows.iloc[[0, -1]].tissue_id.tolist()
        selected += source_rows.loc[source_rows.original_file_name.str.endswith("_split"), "tissue_id"].tolist()
        selected += source_rows.loc[source_rows.tissue_id.isin(
            inventory.loc[inventory.dtype.eq("uint8"), "image_id"]), "tissue_id"].tolist()
        selected += source_rows.loc[[source_rows.image_height.idxmax(), source_rows.image_width.idxmax(),
                                    (source_rows.image_width*source_rows.image_height).idxmin()], "tissue_id"].tolist()
    lookup = tissues.set_index("tissue_id", drop=False)
    delivered = repaired.set_index("image_id")
    comparisons, padding_checks = [], []
    for identity in dict.fromkeys(selected):
        row, image = lookup.loc[identity], delivered.loc[identity]
        spec = source_spec(row)
        width, height = int(row.image_width), int(row.image_height)
        path = ROOT / image.input_image_path
        with formal_read_lease(spec.path), resource_lock(OUTPUT, shared=True), resource_lock(
                path.parent, shared=True, name=".input.lock"):
            with open_source(spec) as source, tifffile.TiffFile(path) as tif:
                decoded = source.read_window(list(range(41)), (0, 0, width, height))
                signal_coordinates = np.argwhere(np.any(decoded, axis=0))
                cy, cx = signal_coordinates[len(signal_coordinates)//2]
                block_width, block_height = min(32, width), min(32, height)
                x0 = min(max(0, int(cx)-block_width//2), width-block_width)
                y0 = min(max(0, int(cy)-block_height//2), height-block_height)
                blocks = [(x0, y0, x0+block_width, y0+block_height),
                          (width-block_width, height-block_height, width, height)]
                scalar_tif = SimpleNamespace(filehandle=tif.filehandle,
                    pages=[page.aspage() if isinstance(page, tifffile.TiffFrame) else page for page in tif.pages])
                for block in blocks:
                    a, b, c, d = block
                    expected = decoded[:, b:d, a:c]
                    output_block = (a+int(image.pad_left_px), b+int(image.pad_top_px),
                                    c+int(image.pad_left_px), d+int(image.pad_top_px))
                    actual = np.stack([read_tiff_window(scalar_tif, channel, output_block) for channel in range(41)])
                    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
                    comparisons.append(dict(image_id=identity, source_window=list(block),
                        output_window=list(output_block), channels=list(range(41)), equal=True,
                        source_signal_nonzero=bool(expected.any())))
                side = int(image.width_px)
                middle = side//2
                pad_blocks = []
                if image.pad_top_px:
                    pad_blocks.append(("top", (middle-16, 0, middle+16, min(32, int(image.pad_top_px)))))
                if image.pad_bottom_px:
                    pad_blocks.append(("bottom", (middle-16, side-min(32, int(image.pad_bottom_px)), middle+16, side)))
                if image.pad_left_px:
                    pad_blocks.append(("left", (0, middle-16, min(32, int(image.pad_left_px)), middle+16)))
                if image.pad_right_px:
                    pad_blocks.append(("right", (side-min(32, int(image.pad_right_px)), middle-16, side, middle+16)))
                for edge, block in pad_blocks:
                    for channel in range(41):
                        assert not read_tiff_window(scalar_tif, channel, block).any()
                    padding_checks.append(dict(image_id=identity, edge=edge, output_window=list(block),
                                               channels=list(range(41)), zero=True))
        progress("checking_repaired_pixels", len(images), current_image=identity,
                 checked_source_output_blocks=len(comparisons))
    evidence = JOB / "verification_self_check.json"
    write_json(evidence, dict(header_images=len(repaired), image_ids=repaired.image_id.tolist(),
        previously_indexed_images=len(images)-len(repaired), source_dtype_counts=repaired.source_dtype.value_counts().to_dict(),
        pixel_comparisons=comparisons, padding_checks=padding_checks,
        original_visual_evidence=relative(CACHE / "source_overviews_preparer.json"),
        identity="preparer_self_check; independent review pending", scientific_repair_round=1))
    additions = []
    for check in ["P03", "I05"]:
        additions.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=REPAIR_RECORD,
            check_id=check, result="pass", source_evidence=relative(PROOF), output_evidence=relative(OUTPUT),
            details=json.dumps(dict(header_images=len(repaired), source_output_blocks=len(comparisons),
                image_ids=repaired.image_id.tolist(), evidence=relative(evidence), stage="preparer_self_check"))))
    publish_preparer_checks(pd.DataFrame(additions))


# Resolve source identity and native panel before any pixel export.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["prepare", "metadata", "recover", "export", "verify"])
parser.add_argument("--limit", type=int, default=32, help="Maximum newly exported images; one formal lease per publication")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
source_metadata()
write_json(CACHE / "controller_requests.json", [
    "完成后请安排不同CLI上下文独立复核及D01绑定；整理者不签发独立验收。",
    "export_input/save_input_tables已改为共享formal lease，仅正式逐image发布阶段占用write槽；"
    "controller应直接调度study脚本并传入统一CELLWEFT_NAS_LOCK_ROOT，保留study/image级发布锁。",
])
tissues = pd.read_parquet(RAW / "metadata/tissues.parquet").sort_values("tissue_id").reset_index(drop=True)
assert tissues.imc_mpp.eq(1.0).all() and tissues.imc_mpp_preprocessed.eq(1.0).all()
channels = pd.read_parquet(RAW / "imc/channels.parquet")
availability = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
assert channels.channel_name.tolist() == availability.columns.tolist()
assert availability.to_numpy().all() and set(availability.index) == set(tissues.tissue_id)
# The original stack is ordered by isotope, with component antibody rows sharing
# a channel; Full=1 identifies its display/measurement identity exactly once.
original_panel = pd.read_csv(ORIGINAL_PANEL)
read_order = pd.read_csv(READ_ORDER, header=None, names=["Metal"], dtype=str).Metal.tolist()
original_order = original_panel[original_panel.Full.eq(1)].set_index("Metal", verify_integrity=True)
assert original_order.loc[read_order, "FormattedName"].tolist() == channels.channel_name.tolist()
panel = panel_from_s1(STUDY, PANEL_ID, channels.channel_name.tolist(), S1)
panel["source_metal"] = read_order
assert panel.keep_channel.all() and panel.loc[panel.nuclear, "source_name"].tolist() == ["DNA1", "DNA2"]
samples, mappings = metadata_tables(tissues)
image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
excluded_path = CACHE / "excluded_images.json"
excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else {}
if args.mode == "prepare" or not (CACHE / "source_inventory.json").exists():
    inventory = inspect_sources(tissues, channels.channel_name.tolist())
else:
    inventory = pd.DataFrame(json.loads((CACHE / "source_inventory.json").read_text())["images"])
with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
    existing = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=image_columns)
rows = existing.to_dict("records")
JOB.mkdir(parents=True, exist_ok=True)
if not (JOB / "before_images.parquet").exists():
    existing.to_parquet(JOB / "before_images.parquet", index=False)
baseline_images = pd.read_parquet(JOB / "before_images.parquet")
repair_image_ids = tissues.loc[~tissues.tissue_id.isin(baseline_images.image_id), "tissue_id"].tolist()

if args.mode == "metadata":
    # Refresh source/clinical rules without rewriting the resumable image index.
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        for name, frame in [("sample_metadata", samples), ("panel_mapping", panel), ("field_mapping", mappings)]:
            frame.to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(dict(phase="metadata_refreshed", sample_rows=len(samples), original_panel_channels=len(read_order))), flush=True)
elif args.mode == "prepare":
    images = save_tables(rows)
    review_results(images, samples, len(rows) + len(excluded) == int(inventory.complete.sum()))
    estimate = json.loads((CACHE / "source_inventory.json").read_text())["expected_uncompressed_square_bytes"]
    progress("prepared_metadata", len(images), eligible_images=int(inventory.complete.sum()),
             sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
             expected_uncompressed_square_bytes=estimate, gaps=["待逐图导出及独立复核"],
             optional_metadata_gaps=["1位供者年龄缺失；标本独立登记号、侧别、解剖亚部位、治疗及结局未公开"])
elif args.mode in {"export", "recover"}:
    completed = set(existing.image_id) | set(excluded)
    pending = tissues[tissues.tissue_id.isin(inventory.loc[inventory.complete, "image_id"]) & ~tissues.tissue_id.isin(completed)]
    for _, row in pending.head(args.limit).iterrows():
        destination = OUTPUT / "images" / row.tissue_id / "input.tiff"
        if args.mode == "recover" and not destination.exists():
            continue
        if shutil.disk_usage(ROOT).free < 100*1024**3 + 256*1024**2 or shutil.disk_usage("/tmp").free < 64*1024**3:
            progress("capacity_blocked", len(rows), gaps=["必须保留NAS 100GiB和本地64GiB"])
            raise SystemExit("Insufficient reserved disk space")
        started = time.monotonic()
        progress("recovering_existing" if args.mode == "recover" else "exporting", len(rows),
                 current_image=row.tissue_id, eligible_images=int(inventory.complete.sum()),
                 gaps=["导出进行中；独立复核待执行"])
        source = source_spec(row)
        with formal_read_lease(source.path):
            with open_source(source) as reader:
                values = reader.read_window(list(range(41)), (0, 0, int(row.image_width), int(row.image_height)))
        result = (reuse_unindexed_output(values, destination) if destination.exists()
                  else export_input(DecodedROI(values, source_spec(row)), panel, destination))
        if result["admitted"]:
            with formal_read_lease(destination), resource_lock(OUTPUT, shared=True), resource_lock(
                    destination.parent, shared=True, name=".input.lock"):
                check_header(destination, result["height_px"], panel.marker_id.tolist(), values.dtype)
            rows.append(make_image(row, result, destination))
        else:
            excluded[row.tissue_id] = "All retained downstream marker channels are zero over the whole source ROI"
            write_json(excluded_path, excluded)
        del values
        images = save_tables(rows)
        with (CACHE / "image_export_records.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(image_id=row.tissue_id, **result, elapsed_seconds=time.monotonic()-started)) + "\n")
        progress("exporting", len(images), last_completed_image=row.tissue_id, excluded_images=len(excluded),
                 sample_linked_images=int(images.sample_id.notna().sum()), sample_rows=len(samples),
                 patient_linked_samples=int(samples.patient_key.notna().sum()),
                 gaps=["独立复核待执行"])
        print(json.dumps(dict(image_id=row.tissue_id, exported=len(images), excluded=len(excluded),
                              seconds=round(time.monotonic()-started, 2))), flush=True)
    remaining = int(inventory.complete.sum()) - len(rows) - len(excluded)
    review_results(pd.DataFrame(rows, columns=image_columns), samples, remaining == 0)
    phase = "export_complete" if remaining == 0 else ("recovered_existing" if args.mode == "recover" else "export_batch_complete")
    progress(phase, len(rows), remaining_images=remaining,
             excluded_images=len(excluded), sample_linked_images=len(rows), sample_rows=len(samples),
             patient_linked_samples=int(samples.patient_key.notna().sum()),
             gaps=([f"剩余{remaining}张待正式导出"] if remaining else []) + ["待全量头信息、自检像素及独立复核"])
else:
    progress("checking_exported_headers", len(existing), checked_headers=0,
             gaps=["全量头信息及代表性像素自检待完成；独立复核待执行"])
    verify_images(existing)
    remaining = int(inventory.complete.sum()) - len(existing) - len(excluded)
    review_results(existing, samples, remaining == 0)
    progress("self_check_complete" if remaining == 0 else "partial_self_check_complete", len(existing),
             remaining_images=remaining, excluded_images=len(excluded), sample_linked_images=len(existing),
             sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
             gaps=([f"剩余{remaining}张待正式导出"] if remaining else []) + ["待不同CLI上下文独立复核和主控验收"],
             optional_metadata_gaps=["1位供者年龄缺失；独立标本登记号、侧别、解剖亚部位、治疗及结局未公开"])
