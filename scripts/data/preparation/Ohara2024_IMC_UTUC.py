"""Prepare Ohara UTUC IMC TIFFs as native-intensity study inputs.

Inputs: 58 qualified Zenodo 5719188 TIFFs, ordered channels, S1, and the
author's sample table. Outputs: per-ROI CYX TIFFs and image, sample, panel,
field, preparer-review, and resumable decision tables on the study NAS path.

Source follow-up: processed/Ohara2024_IMC_UTUC/storage_reception_20261110/
ohara_released_scope_acceptance_20261028T2140Z.json records the independently
checked author master-sample/block hierarchy, literal
patient-code rule, current D01 and 58 author whole-cell mask bindings. Raw table
identities are retained. Article-cohort membership, optional clinical fields,
fine mask boundaries, tissue and training quality remain pending. The existing
input export is reused; acceptance is recorded by the independent reviewer.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Ohara2024_IMC_UTUC"
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10948878/"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
OUTPUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
JOB = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0257_prepare_0"
PROOF = NAS / "raw/public" / STUDY / "manifest.json"
ORDER = metadata_source_directory(NAS, STUDY) / 'native_tiff_ordered_markers.json'
AUTHOR = NAS / "raw/public" / STUDY / "metadata/author_samples.csv"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
LOCAL = Path("/tmp/study_input_batch") / STUDY
PANEL_ID = "ohara:panel1"
PREPARER = "Ohara2024_IMC_UTUC_0257_prepare_0"


class NativeTiffReader:
    """Window-access the released uncompressed, contiguous scalar TIFF pages."""

    def __init__(self, source: SourceSpec):
        self.source = source
        self.pixels = tifffile.memmap(source.path)
        self.shape = self.pixels.shape
        self.dtype = self.pixels.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.pixels[channels, y0:y1, x0:x1])

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1 - y0, x1 - x0), dtype=bool)

    def close(self) -> None:
        self.pixels._mmap.close()


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, NAS))


def publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with formal_publish_lease(path, directory=path.parent):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)


def field(source: Path, source_field: str, target_table: str, target_field: str,
          join_keys: str, rule: str, notes: str) -> dict:
    return dict(study=STUDY, source_file=relative(source),
                source_sheet_or_table="public source", source_field=source_field,
                target_table=target_table, target_field=target_field,
                join_keys=join_keys, value_mapping="", transformation_rule=rule,
                time_origin_rule="No time interval inferred",
                conflict_rule="Keep unresolved without a direct source key", notes=notes)


def checkpoint(stage: str, images: pd.DataFrame, decisions: dict, **extra: object) -> None:
    joined = images.sample_id.notna() if len(images) else pd.Series(dtype=bool)
    publish_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(qualified),
        completed_images=len(images), processed_sources=len(decisions),
        excluded_zero_signal=sum(not item["admitted"] for item in decisions.values()),
        sample_rows=len(samples), sample_linked_images=int(joined.sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        next_source=next((item["native_path"] for item in sorted(qualified, key=lambda value: value["native_path"])
                          if item["native_path"] not in decisions), None),
        updated_at=datetime.now(timezone.utc).isoformat(), **extra))


# Source qualification, image identity, and author sample links.
qualified = json.loads(PROOF.read_text())["preparation_sources"]["native_images"]["verified_images"]
ordered = json.loads(ORDER.read_text())["ordered_channel_names"]
author = pd.read_csv(AUTHOR, dtype=str, keep_default_na=False).set_index("sample_name", verify_integrity=True)
assert len(qualified) == 58 and len(ordered) == 43 and len(author) == 14
assert {Path(item["native_path"]).parts[-3] for item in qualified} == set(author.index)
assert all((NAS / item["native_path"]).is_file() and
           (NAS / item["native_path"]).stat().st_size == item["file_stat"]["size"] and
           (NAS / item["native_path"]).stat().st_mtime_ns == item["file_stat"]["mtime_ns"]
           for item in qualified)

panel = panel_from_s1(STUDY, PANEL_ID, ordered, S1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 28 and int((kept.role == "downstream").sum()) == 26
image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()

sample_rows = []
for sample_name, source in author.iterrows():
    row = {column: pd.NA for column in sample_columns}
    row.update(study=STUDY, sample_id=sample_name, patient_key=source.PMID,
               source_subject_id=source.PMID, source_sample_id=source.PMID_block,
               sample_type="tumor tissue", primary_diagnosis="upper tract urothelial carcinoma",
               lesion_type=source["Primary/Metastasis"].lower(),
               molecular_subtype=source["Basal/Luminal"] or pd.NA,
               metadata_notes=("Author sample_name matches the image directory; PMID links specimens to patients. "
                               f"Source qc={source.qc or 'unreported'}; segmentation={source.segmentation}. "
                               "Specific anatomical site and optional clinical outcomes are not provided by this table."))
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=sample_columns)
assert samples.patient_key.nunique() == 7

fields = pd.DataFrame([
    dict(study=STUDY, source_file=relative(PROOF), source_sheet_or_table="collection manifest",
         source_field="study", target_table="sample_metadata", target_field="study",
         join_keys="study-scoped raw source manifest", value_mapping=STUDY,
         transformation_rule="copy canonical study identifier from the source manifest",
         time_origin_rule="", conflict_rule="none",
         notes="The public source manifest records the study identity for this author sample table."),
    dict(study=STUDY, source_file=relative(AUTHOR), source_sheet_or_table="CSV",
         source_field="sample_name", target_table="sample_metadata", target_field="sample_id",
         join_keys="exact author sample_name", value_mapping="copy sample_name",
         transformation_rule="preserve the author sample key as the row identifier",
         time_origin_rule="", conflict_rule="do not merge distinct sample_name rows",
         notes="All 14 author sample_name values match one source directory and each of the 58 TIFFs maps to a source sample row."),
    dict(study=STUDY, source_file=PAPER_URL, source_sheet_or_table="Abstract; Methods: Imaging Mass Cytometry",
         source_field="upper tract urothelial carcinoma; 14 tumor samples", target_table="sample_metadata",
         target_field="primary_diagnosis;sample_type", join_keys="reported IMC cohort",
         value_mapping="upper tract urothelial carcinoma; tumor tissue",
         transformation_rule="copy the explicit cohort disease and tissue description to the reported IMC sample rows",
         time_origin_rule="", conflict_rule="retain the article 6-patient versus author-table 7-PMID discrepancy",
         notes="The article reports 14 IMC tumor samples from 6 patients; the author table has 14 rows and 7 distinct PMID values. No patient identities are merged or inferred."),
    field(PROOF, "native_path;file_stat;series_shape;series_axes", "image_metadata",
          "source_files;source_image_id;source_axes;source_dtype", "native_path",
          "Use 58 accepted, unchanged original TIFFs and their full scalar page order",
          "Collection proof contains 54 all-channel bounded reads and four previously accepted sampled-page proofs; 46 CYX and 12 QYX series, all with 43 scalar planes."),
    field(ORDER, "ordered_channel_names", "panel_mapping",
          "source_channel_index;source_name", "ordered native page index",
          "Bind all 43 TIFF pages to the native ordered labels",
          "The Q axis on 12 TIFF series is the same 43 ordered scalar pages, not a biological Z or time axis."),
    field(S1, "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
          "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
          "study + exact native source_name", "Retain 26 admitted measurements and two DNA segmentation channels",
          "Source order is preserved; excluded and empty channels have no output index."),
    field(AUTHOR, "sample_name;PMID_block", "image_metadata", "sample_id",
          "TIFF parent sample directory = sample_name", "Join exact directory name to the author sample table",
          "All 58 accepted TIFFs connect to one of 14 sample_name rows; each source ROI remains a distinct image."),
    field(AUTHOR, "PMID;PMID_block", "sample_metadata",
          "patient_key;source_subject_id;source_sample_id", "sample_name",
          "Use source PMID as the deidentified patient key and PMID_block as specimen identity",
          "Fourteen author samples map to seven distinct patient keys; PMID is never inferred from TIFF filename tokens."),
    field(AUTHOR, "Primary/Metastasis;Basal/Luminal", "sample_metadata",
          "lesion_type;molecular_subtype", "sample_name",
          "Map Primary/Metastasis to primary/metastasis and retain the published subtype value",
          "Subtype is unfilled when the author table leaves it blank; no site, stage, treatment or outcome is inferred."),
    field(AUTHOR, "qc;segmentation", "sample_metadata", "metadata_notes", "sample_name",
          "Preserve the original author flags per sample without treating them as an input exclusion rule",
          "The author table contains TRUE, FALSE and blank qc values; independent review should interpret their analysis scope."),
    field(AUTHOR, "Tcell phenotype;Tcell Inflamation Signature ssGSEA ES;Tcell Inflamation Signature (z-score)",
          "sample_metadata", "metadata_notes", "sample_name",
          "Clinical source examined; do not substitute derived signatures for clinical outcomes",
          "Author table includes optional derived immune summaries. Source values remain available in the cached author table."),
    field(NAS / "raw/public" / STUDY / "metadata/zenodo_5719188_metadata.csv", "filename;checksum;filesize;id",
          "image_metadata", "source_revision", "Zenodo file path",
          "Use public record 5719188 as source revision", "This Zenodo metadata.csv is a file manifest, not a clinical table."),
    field(NAS / "raw/public" / STUDY / "metadata/zenodo_10721591_metadata.csv", "filename;checksum;filesize;id",
          "sample_metadata", "metadata_notes", "Zenodo record",
          "Check the later public release for an identity bridge",
          "This record also contains a file manifest; its WCM zip filenames do not directly identify the 58 TIFF ROI paths."),
])

LOCAL.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
index_path = OUTPUT / "images.parquet"
images = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame(columns=image_columns)
decisions_path = CACHE / "source_decisions.json"
decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}

# Reserve space for the full conservative output bound and one local source/image.
sides = [max((item.get("series_shape") or item.get("header_shape"))[-2:]) for item in qualified]
output_bound = sum(len(kept) * side * side * 4 for side in sides)
largest_source = max(item["file_stat"]["size"] for item in qualified)
largest_image = len(kept) * max(sides) ** 2 * 4
assert shutil.disk_usage(NAS).free > (100 << 30) + output_bound
assert shutil.disk_usage(LOCAL).free > (64 << 30) + largest_source + largest_image
save_input_tables(OUTPUT, pd.DataFrame(columns=image_columns), samples, panel, fields)
checkpoint("preflight", images, decisions, output_bound_bytes=output_bound,
           largest_local_source_bytes=largest_source, largest_staged_image_bytes=largest_image)

# Each source is copied and closed before the helper stages and publishes its ROI.
batch_rows = []
batch_count = 0
for item in sorted(qualified, key=lambda value: value["native_path"]):
    key = item["native_path"]
    source = NAS / key
    group = source.parts[-3]
    image_id = "ohara_" + source.stem.removesuffix("_full")
    destination = OUTPUT / "images" / image_id / "input.tiff"
    if key in decisions and (not decisions[key]["admitted"] or
                             (image_id in set(images.image_id) and destination.is_file())):
        continue
    local_source = LOCAL / source.name
    partial = local_source.with_name(local_source.name + ".partial")
    try:
        if not local_source.is_file() or local_source.stat().st_size != item["file_stat"]["size"]:
            with formal_read_lease(source):
                shutil.copyfile(source, partial)
            os.replace(partial, local_source)
        assert local_source.stat().st_size == item["file_stat"]["size"]
        specimen = SourceSpec(image_id, PANEL_ID, "tiff", local_source, tuple(range(43)),
                              source_revision="zenodo:5719188",
                              intensity_description="Original released TIFF float32 intensities; no normalization")
        reader = NativeTiffReader(specimen)
        try:
            shape = item.get("series_shape") or item.get("header_shape")
            assert reader.shape == tuple(shape) and reader.dtype.name == "float32"
            result = export_input(reader, panel, destination)
        finally:
            reader.close()
        if result["admitted"]:
            with formal_read_lease(destination):
                with tifffile.TiffFile(destination) as tif:
                    assert tif.series[0].shape == (len(kept), result["height_px"], result["width_px"])
                    assert tif.series[0].dtype.name == "float32" and tif.series[0].axes == "CYX"
            row = {column: pd.NA for column in image_columns}
            row.update(study=STUDY, image_id=image_id, sample_id=group, panel_id=PANEL_ID,
                       image_kind="roi", source_image_id=relative(source), source_files=[relative(source)],
                       source_format="tiff", source_axes=item.get("series_axes", item.get("header_axes")),
                       source_dtype="float32", source_revision="zenodo:5719188",
                       slide_id=group, roi_id=source.stem, source_x0_px=0, source_y0_px=0,
                       source_x1_px=reader.shape[2], source_y1_px=reader.shape[1],
                       source_channel_count=43, channel_count=len(kept),
                       input_image_path=relative(destination),
                       valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                          if result["valid_pixels"] else pd.NA),
                       intensity_description=specimen.intensity_description,
                       split_method="already_independent", split_evidence="Complete released TIFF ROI frame",
                       metadata_notes="Exact author sample_name link; source TIFF has no calibrated physical pixel size.",
                       **{name: result[name] for name in ("height_px", "width_px", "pad_left_px",
                                                            "pad_top_px", "pad_right_px", "pad_bottom_px")})
            batch_rows.append(row)
            images = pd.concat([images.loc[images.image_id.ne(image_id)], pd.DataFrame([row])],
                               ignore_index=True)
        decisions[key] = dict(admitted=bool(result["admitted"]), image_id=image_id,
                              reason="retained downstream signal" if result["admitted"] else
                              "all retained downstream channels zero", source_file=relative(source))
        batch_count += 1
        if batch_count == 3:
            save_input_tables(OUTPUT, pd.DataFrame(batch_rows, columns=image_columns), samples, panel, fields)
            publish_json(decisions_path, decisions)
            checkpoint("exporting", images, decisions, last_source=key)
            print(f"{len(decisions)}/{len(qualified)} sources checkpointed", flush=True)
            batch_rows.clear()
            batch_count = 0
    finally:
        partial.unlink(missing_ok=True)
        local_source.unlink(missing_ok=True)

if batch_count:
    save_input_tables(OUTPUT, pd.DataFrame(batch_rows, columns=image_columns), samples, panel, fields)
    publish_json(decisions_path, decisions)
    checkpoint("exporting", images, decisions, last_source=key)
    print(f"{len(decisions)}/{len(qualified)} sources checkpointed", flush=True)

# The preparer records coverage and leaves independent checks to the reviewer.
review_rows = [
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
         check_id="S01", result="unresolved", source_evidence=relative(PROOF),
         output_evidence=relative(decisions_path),
         details=json.dumps({"qualified_sources": len(qualified), "processed_sources": len(decisions),
                             "admitted_images": len(images),
                             "excluded_zero_signal": sum(not x["admitted"] for x in decisions.values()),
                             "pending": "independent source-scope and tissue overview review"})),
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample", record_id=STUDY,
         check_id="M01", result="unresolved", source_evidence=relative(AUTHOR),
         output_evidence=relative(OUTPUT / "sample_metadata.csv"),
         details=json.dumps({"sample_rows": len(samples), "sample_linked_images": int(images.sample_id.notna().sum()),
                             "patient_linked_samples": int(samples.patient_key.notna().sum()),
                             "patient_groups": int(samples.patient_key.nunique()),
                             "join": "TIFF directory = author sample_name; author PMID = patient key",
                             "pending": "independent identity and clinical source review"})),
    dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
         check_id="D01", result="unresolved", source_evidence=relative(PROOF),
         output_evidence=relative(index_path),
         details="Preparer files closed; independent review and input binding pending."),
]
review_path = OUTPUT / "review_results.csv"
with formal_publish_lease(review_path, directory=OUTPUT):
    temporary = review_path.with_name(review_path.name + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pd.read_csv(TEMPLATES / "review_results.csv").columns)
        writer.writeheader()
        writer.writerows(review_rows)
    os.replace(temporary, review_path)
checkpoint("prepared_pending_independent_review", images, decisions)
