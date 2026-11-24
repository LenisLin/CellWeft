"""Prepare native IMC TXT acquisitions from the Xi 2026 public release."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Xi2026_IMC_NSCLC_LymphNode"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
RAW = NAS / "raw/public" / STUDY / "public_original/Raw_IMC_TXT.zip"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0607_prepare_0"
REPAIR_JOB = Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0640_repair_1"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
AUTHOR_CODE = NAS / "raw/public" / STUDY / "metadata/Publication_Code_IMC_Analysis.Rmd"
CLINICAL = NAS / "raw/public" / STUDY / "metadata/paper_supplementary_tables.xlsx"
PANEL_ID = "native_TXT_41"
PREPARER = "0607_prepare_0"
REVISION = "10.5281/zenodo.18265339"


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def publish_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode()
    with formal_publish_lease(path, directory=path.parent, data_root=NAS):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def progress(phase: str, expected: int, completed: int, *, job: Path = JOB,
             **details: object) -> None:
    publish_json(job / "progress.json", dict(study=STUDY, phase=phase,
                 expected_images=expected, completed_images=completed,
                 updated_at=datetime.now(timezone.utc).isoformat(), **details))


def source_identity(member: str) -> tuple[str, str, str]:
    name = Path(member).stem
    if "_ROI_" not in name or not name.startswith("Patient"):
        raise ValueError(f"Unexpected original acquisition name: {member}")
    sample = name.split("_", 1)[0]
    patient = sample.split("-", 1)[0]
    if not re.fullmatch(r"Patient\d+-[\w]+", sample) or not re.fullmatch(r"Patient\d+", patient):
        raise ValueError(f"Unrecognized sample/patient key: {member}")
    return name, sample, patient


def excluded_ln() -> str:
    text = AUTHOR_CODE.read_text(encoding="utf-8")
    match = re.search(r'Removing [^\n]*metastatic LN[^\n]*\n[^\n]*spe\$ln_id != "([^"]+)"', text)
    if match is None:
        raise ValueError("Author exclusion of metastatic LN could not be verified")
    return match.group(1)


class TxtReader:
    def __init__(self, records: np.memmap):
        xy = records[:, :2]
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy != np.floor(xy)):
            raise ValueError("Native TXT has invalid XY coordinates")
        width, height = xy.max(axis=0).astype(np.int64) + 1
        if width * height > 50_000_000:
            raise ValueError("Native TXT frame exceeds the verified local geometry limit")
        self.index = np.full((height, width), -1, dtype=np.int32)
        for start in range(0, len(records), 65536):
            end = min(start + 65536, len(records))
            x, y = records[start:end, :2].astype(np.int64).T
            self.index[y, x] = np.arange(start, end, dtype=np.int32)
        if np.count_nonzero(self.index >= 0) != len(records):
            raise ValueError("Duplicate source XY records")
        self.records = records
        self.shape = (41, int(height), int(width))
        self.dtype = np.dtype("float32")
        self.source = SimpleNamespace(path=Path("/tmp/study_input_batch"), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        index = self.index[y0:y1, x0:x1]
        valid = index >= 0
        result = np.zeros((len(channels), y1-y0, x1-x0), dtype=np.float32)
        for output, channel in enumerate(channels):
            result[output, valid] = self.records[index[valid], channel + 2]
        return result

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.index[y0:y1, x0:x1] >= 0


def decode_member(member: str, proof: dict, scratch: Path) -> tuple[np.memmap, int]:
    count = 0
    with formal_read_lease(RAW, data_root=NAS):
        with zipfile.ZipFile(RAW) as archive:
            info = archive.getinfo(member)
            expected = proof["source_member_stat"]
            if info.file_size != expected["file_size"] or info.CRC != expected["CRC"]:
                raise ValueError("Original ZIP member differs from accepted native proof")
            with archive.open(info) as stream, scratch.open("wb") as output:
                header = stream.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
                if header != proof["channels"] or len(header) != 47:
                    raise ValueError("Native header differs from accepted ordered channels")
                for chunk in pd.read_csv(stream, sep="\t", names=header,
                                         usecols=[3, 4, *range(6, 47)],
                                         dtype=np.float32, chunksize=32768):
                    block = chunk.to_numpy(dtype=np.float32)
                    block.tofile(output)
                    count += len(block)
    if not count or scratch.stat().st_size != count * 43 * 4:
        raise ValueError("Native TXT decode is incomplete")
    return np.memmap(scratch, mode="r", dtype=np.float32, shape=(count, 43)), count


def field_mapping() -> pd.DataFrame:
    rows = []
    for table, field, source, rule in [
        ("image_metadata", "source_image_id", "ZIP member", "Exact accepted native TXT member"),
        ("image_metadata", "sample_id", "sample name before ROI suffix", "Use LN prefix before first underscore, as in author IMC analysis"),
        ("image_metadata", "region_annotation", "LN prefix and ROI token", "Retain native acquisition identity; no inferred LN station"),
        ("image_metadata", "source_x1_px", "X", "Full decoded max(X)+1, half-open"),
        ("image_metadata", "source_y1_px", "Y", "Full decoded max(Y)+1, half-open"),
        ("image_metadata", "input_image_path", "native TXT measurement columns", "Float32 native values in S1-selected source order"),
        ("image_metadata", "pixel_size_x_um", "Zenodo dataset description", "1 µm Hyperion acquisition pixel spacing"),
        ("image_metadata", "pixel_size_y_um", "Zenodo dataset description", "1 µm Hyperion acquisition pixel spacing"),
        ("sample_metadata", "sample_id", "spe$ln_id", "Source LN prefix shared by its ROIs"),
        ("sample_metadata", "patient_key", "spe$patient_id", "Source LN key before hyphen, as in author code"),
        ("sample_metadata", "primary_diagnosis", "study inclusion", "NSCLC resection cohort; exact histology unavailable"),
        ("sample_metadata", "anatomical_site", "study description", "Non-metastatic regional lymph node"),
        ("sample_metadata", "sex_reported", "Supplementary Table 1: Patient ID", "Leave blank: no verified crosswalk to native IMC identifiers"),
        ("sample_metadata", "smoking_status", "Supplementary Table 1: Patient ID", "Leave blank: no verified crosswalk to native IMC identifiers"),
        ("sample_metadata", "stage_group_at_index", "Supplementary Table 1: Patient ID", "Leave blank: no verified crosswalk to native IMC identifiers"),
        ("sample_metadata", "age_at_collection_years", "Supplementary Table 1: age range", "Leave blank: source has intervals and no verified native IMC crosswalk"),
        ("panel_mapping", "source_channel_index", "native TXT header", "Measurement columns after first six coordinate/acquisition columns; S1 exact label join"),
    ]:
        source_file = AUTHOR_CODE if "spe$" in source or field == "sample_id" else META / "native_verification.json"
        if field in {"primary_diagnosis", "anatomical_site"}:
            source_file = NAS / "raw/public" / STUDY / "metadata/README.txt"
        if source.startswith("Supplementary Table"):
            source_file = CLINICAL
        rows.append(dict(study=STUDY, source_file=relative(source_file),
                         source_sheet_or_table="native TXT / author IMC analysis / release README",
                         source_field=source, target_table=table, target_field=field,
                         join_keys="ZIP member -> LN prefix -> patient prefix",
                         value_mapping="", transformation_rule=rule, time_origin_rule="",
                         conflict_rule="Author metastatic LN exclusion applies to non-metastatic input scope",
                         notes="Native image→LN→patient links are confirmed by author IMC code. Supplementary Tables 1/2 use a different anonymous patient numbering system without an exact IMC crosswalk; overlapping numbers do not establish identity."))
    release = "https://zenodo.org/records/18265339"
    for field, source_file, source_field, mapping, rule in [
        ("source_subject_id", relative(AUTHOR_CODE), "spe$patient_id", "",
         "Use the patient prefix of the native LN key, as in the author IMC analysis"),
        ("source_sample_id", relative(AUTHOR_CODE), "spe$ln_id", "",
         "Use the LN prefix of the native ROI name, as in the author IMC analysis"),
        ("sample_type", release, "Description: Lymph node tissue samples", "",
         "Record lymph node tissue for each source LN"),
        ("tissue_type", release, "Description: lymph node tissue; non-metastatic study scope",
         json.dumps({"non-metastatic lymph node": "normal"}),
         "Map pathology-confirmed non-metastatic LN tissue to the non-tumor normal category; patients have NSCLC"),
        ("collection_method", release, "Description: surgical specimens from curative-intent resection",
         json.dumps({"surgical resection": "resection"}),
         "Record source surgical collection method"),
        ("preservation_method", release, "Description: FFPE tissue sections",
         json.dumps({"FFPE": "ffpe"}),
         "Record source tissue preservation method"),
        ("case_control_status", release, "Description: patients undergoing NSCLC resection",
         json.dumps({"NSCLC patient": "case"}),
         "Cancer patient case status; non-metastatic LN is the sampled tissue, not a healthy control patient"),
    ]:
        rows.append(dict(study=STUDY, source_file=source_file,
                         source_sheet_or_table="author IMC analysis" if source_file != release else "Zenodo dataset description",
                         source_field=source_field, target_table="sample_metadata", target_field=field,
                         join_keys="native ROI -> LN prefix -> patient prefix",
                         value_mapping=mapping, transformation_rule=rule, time_origin_rule="",
                         conflict_rule="Only pathology-confirmed non-metastatic LN acquisitions are included",
                         notes="29 source LN samples link 106 included ROI to 11 patient keys; published clinical tables lack a verified native IMC crosswalk."))
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def sample_table(proofs: list[dict]) -> pd.DataFrame:
    entries = {}
    for proof in proofs:
        _, sample, patient = source_identity(proof["source_member"])
        if sample in entries:
            continue
        row = dict.fromkeys(columns("sample_metadata"), "")
        row.update(study=STUDY, sample_id=sample, patient_key=patient,
                   source_subject_id=patient, source_sample_id=sample,
                   sample_type="lymph node tissue", tissue_type="normal",
                   anatomical_site="non-metastatic regional lymph node",
                   case_control_status="case", primary_diagnosis="non-small cell lung cancer",
                   collection_method="resection", preservation_method="ffpe",
                   metadata_notes="Author LN/patient identity from IMC code. Published clinical Table 1 and tissue Table 2 have no verified crosswalk to native IMC identifiers; individual clinical values remain blank."
                   )
        entries[sample] = row
    return pd.DataFrame(entries.values(), columns=columns("sample_metadata"))


def image_row(proof: dict, result: dict, destination: Path) -> dict:
    image_id, sample, _ = source_identity(proof["source_member"])
    row = dict.fromkeys(columns("image_metadata"), None)
    row.update(study=STUDY, image_id=image_id, sample_id=sample, panel_id=PANEL_ID,
               image_kind="roi", source_image_id=proof["source_member"],
               source_files=[relative(RAW)], source_format="txt", source_acquisition_id=image_id,
               source_axes="XY records; 41 measurement columns", source_dtype="float32",
               source_revision=REVISION, roi_id=image_id.split("_ROI_", 1)[1].split("_", 1)[0],
               region_annotation=image_id, source_x0_px=0, source_y0_px=0,
               source_x1_px=result["source_width"], source_y1_px=result["source_height"],
               pixel_size_x_um=1.0, pixel_size_y_um=1.0,
               input_image_path=relative(destination),
               valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
               intensity_description="Original IMC TXT measurements decoded as float32; no normalization or transformation",
               split_method="source_acquisition", split_evidence="One complete original TXT acquisition per ROI",
               metadata_notes="Native ZIP member and channel order verified; author analysis LN/patient parsing. Source slide ID unavailable. Independent review pending.")
    for field in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                  "pad_bottom_px", "source_channel_count", "channel_count"):
        row[field] = result[field]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--waiting-io", action="store_true")
    parser.add_argument("--repair-review", action="store_true")
    args = parser.parse_args()
    if args.waiting_io and args.max_images != 0:
        parser.error("--waiting-io requires --max-images 0")
    if args.repair_review and (args.waiting_io or args.max_images is not None):
        parser.error("--repair-review cannot be combined with preparation options")
    with formal_read_lease(META / "native_verification.json", data_root=NAS):
        verification = json.loads((META / "native_verification.json").read_text())
    proofs = verification["native_result"]["images"]
    if verification["native_result"]["status"] != "passed" or len(proofs) != 107:
        raise ValueError("Accepted native proof set changed")
    with formal_read_lease(RAW, data_root=NAS):
        archive_size = RAW.stat().st_size
    if archive_size != proofs[0]["source_file_stat"]["size"]:
        raise ValueError("Original archive size changed since native verification")
    with formal_read_lease(AUTHOR_CODE, data_root=NAS):
        excluded = excluded_ln()
    selected = [proof for proof in proofs if source_identity(proof["source_member"])[1] != excluded]
    expected = len(selected)
    names = proofs[0]["channels"][6:]
    if any(proof["channels"][6:] != names or not proof["complete"] for proof in selected):
        raise ValueError("Source channels or native completeness differ across selected acquisitions")
    with formal_read_lease(S1, data_root=NAS):
        panel = panel_from_s1(STUDY, PANEL_ID, names, S1,
                              downstream_source_indices=set(range(41)))
    samples = sample_table(selected)
    fields = field_mapping()
    if args.repair_review:
        with formal_read_lease(OUTPUT / "images.parquet", data_root=NAS):
            existing = pd.read_parquet(OUTPUT / "images.parquet")
        if len(existing) != expected or existing.image_id.nunique() != expected:
            raise ValueError("Existing image table does not cover the accepted source scope")
        with formal_read_lease(OUTPUT, data_root=NAS):
            present = sum((NAS / path).is_file() for path in existing.input_image_path)
        if present != expected:
            raise ValueError(f"Only {present}/{expected} listed input TIFFs exist")
        free_nas = shutil.disk_usage(NAS).free
        free_local = shutil.disk_usage("/tmp").free
        if free_nas < 100 * 1024**3 or free_local < 64 * 1024**3:
            progress("waiting_io", expected, len(existing), job=REPAIR_JOB,
                     reason="Storage reserve check failed before metadata publication",
                     next_image_index=len(existing), free_nas_bytes=free_nas,
                     free_local_bytes=free_local)
            return
        save_input_tables(OUTPUT, existing, samples, panel, fields, data_root=NAS)
        progress("repaired_pending_review", expected, len(existing), job=REPAIR_JOB,
                 next_image_index=len(existing), sample_rows=len(samples),
                 sample_linked_images=int(existing.sample_id.notna().sum()),
                 patient_linked_samples=int(samples.patient_key.ne("").sum()),
                 repairs=["P02: restored CD45RO downstream role; retained TIFF channel 31",
                          "M02: added source and transformation rows for seven populated sample fields"],
                 independent_review="pending")
        return
    images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
    for row in images:
        name, _, _ = source_identity(row["source_image_id"])
        row["slide_id"] = None
        row["roi_id"] = name.split("_ROI_", 1)[1].split("_", 1)[0]
        row["metadata_notes"] = ("Native ZIP member and channel order verified; author analysis LN/patient "
                                 "parsing. Source slide ID unavailable. Independent review pending.")
    complete_ids = {row["image_id"] for row in images if (NAS / row["input_image_path"]).is_file()}
    free_nas = shutil.disk_usage(NAS).free
    free_local = shutil.disk_usage("/tmp").free
    max_scratch = max(proof["source_member_stat"]["file_size"] for proof in selected)
    output_bound = expected * 41 * 4096 * 4096 * 4
    if free_nas < output_bound + 100 * 1024**3 or free_local < max_scratch + 64 * 1024**3:
        progress("waiting_io", expected, len(complete_ids), reason="Storage reserve check failed",
                 free_nas_bytes=free_nas, free_local_bytes=free_local,
                 next_image_index=len(complete_ids))
        return
    progress("sources_checked", expected, len(complete_ids), source_members=len(proofs),
             excluded_members=len(proofs)-expected, sample_rows=len(samples),
             patient_linked_samples=int(samples.patient_key.ne("").sum()),
             free_nas_bytes=free_nas, free_local_bytes=free_local,
             conservative_output_bound_bytes=output_bound)
    if images:
        save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel, fields, data_root=NAS)
    produced = 0
    for index, proof in enumerate(selected):
        image_id, _, _ = source_identity(proof["source_member"])
        if image_id in complete_ids:
            continue
        if args.max_images is not None and produced >= args.max_images:
            break
        destination = OUTPUT / "images" / image_id / "input.tiff"
        with tempfile.TemporaryDirectory(prefix="xi-native-", dir="/tmp") as temporary:
            scratch = Path(temporary) / "measurements.bin"
            records, count = decode_member(proof["source_member"], proof, scratch)
            try:
                reader = TxtReader(records)
                source_height, source_width = reader.shape[-2:]
                estimated = len(panel.loc[panel.keep_channel]) * max(source_height, source_width)**2 * 4
                if shutil.disk_usage(NAS).free < estimated + 100*1024**3 or shutil.disk_usage("/tmp").free < estimated + 64*1024**3:
                    progress("waiting_io", expected, len(complete_ids), reason="Per-image storage reserve check failed",
                             next_image_index=index, estimated_output_bytes=estimated)
                    return
                result = export_input(reader, panel, destination, data_root=NAS)
                result.update(source_height=source_height, source_width=source_width, decoded_records=count)
                if not result["admitted"]:
                    progress("source_excluded", expected, len(complete_ids), next_image_index=index,
                             reason="No retained downstream measurement has signal")
                    continue
                with tifffile.TiffFile(destination) as tif:
                    if tif.series[0].shape != (len(panel.loc[panel.keep_channel]), result["height_px"], result["width_px"]):
                        raise ValueError("Published TIFF header differs from panel/geometry")
                row = image_row(proof, result, destination)
                images = [old for old in images if old["image_id"] != image_id] + [row]
                save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel, fields, data_root=NAS)
                complete_ids.add(image_id)
                produced += 1
                progress("image_complete", expected, len(complete_ids), latest_image_index=index,
                         next_image_index=index+1, sample_rows=len(samples),
                         sample_linked_images=len(complete_ids),
                         patient_linked_samples=int(samples.patient_key.ne("").sum()))
                print(json.dumps(dict(completed_images=len(complete_ids), expected_images=expected,
                                      source_shape=reader.shape, output_bytes=destination.stat().st_size)), flush=True)
            finally:
                del records
    review_path = OUTPUT / "review_results.csv"
    checks = pd.read_csv(TEMPLATES / "review_checks.csv", keep_default_na=False)
    review = pd.DataFrame([dict(study=STUDY, preparer=PREPARER, reviewer="", scope=item.scope,
                                record_id=PANEL_ID if item.scope == "panel" else STUDY,
                                check_id=item.check_id, result="unresolved",
                                source_evidence=relative(META / "native_verification.json") + "; " + relative(AUTHOR_CODE),
                                output_evidence=relative(OUTPUT),
                                details=f"Preparer checkpoint: {len(complete_ids)}/{expected} inputs published; independent reviewer pending. "
                                        f"{len(samples)} LN samples and {int(samples.patient_key.ne('').sum())} patient-linked samples. "
                                        "Author metastatic LN excluded; published clinical table lacks a verified native IMC patient crosswalk.")
                           for item in checks.itertuples(index=False)], columns=columns("review_results"))
    if review_path.is_file():
        prior = pd.read_csv(review_path, keep_default_na=False)
        review = pd.concat([prior.loc[prior.preparer != PREPARER], review], ignore_index=True)
    with formal_publish_lease(review_path, directory=OUTPUT, data_root=NAS):
        review_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = review_path.with_name(review_path.name + ".partial")
        review.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, review_path)
    phase = "waiting_io" if args.waiting_io else ("prepared" if len(complete_ids) == expected else "partial")
    progress(phase, expected, len(complete_ids),
             next_image_index=len(complete_ids), sample_rows=len(samples),
             sample_linked_images=len(complete_ids),
             patient_linked_samples=int(samples.patient_key.ne("").sum()),
             independent_review="pending",
             reason="Shared NAS read/write admission had multi-minute waits; resume from next_image_index" if args.waiting_io else "")


if __name__ == "__main__":
    main()
