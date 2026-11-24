"""Prepare the 352 released Van Krimpen IMC ROIs as native-intensity inputs.

The large RDS is copied to local scratch under a shared NAS read lease. Its
R-serialized image vectors are decoded one at a time; the shared exporter stages
and publishes each TIFF under the project NAS write lease. Four-image table
batches and the job checkpoint make a stopped run resumable.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "VanKrimpen2026_IMC_MelanomaTDLN"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB",
                          PROJECT / "cache/study_preparation" / STUDY / "jobs/0605_prepare_0"))
SOURCE = NAS / "raw/public" / STUDY / "public_original/images_spe.rds"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "zenodo20038486_revision6"
SCRATCH = Path("/tmp/study_input_batch") / STUDY
LOCAL_SOURCE = SCRATCH / "images_spe.rds"
RDATA_VENDOR = Path(__file__).resolve().parents[3] / "reference/vendor/rdata-1.1.0"

sys.path.insert(0, str(RDATA_VENDOR))
from rdata.parser._parser import DEFAULT_ALTREP_MAP  # noqa: E402
from rdata.parser._xdr import ParserXDR  # noqa: E402

pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def publish_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with formal_read_lease(path):
        unchanged = path.is_file() and path.read_text() == payload
    if unchanged:
        return
    with formal_publish_lease(path, directory=path.parent, lock_name=".checkpoint.lock"):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(payload)
        os.replace(temporary, path)


def stage_source() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    source_stat = SOURCE.stat()
    if LOCAL_SOURCE.is_file():
        local_stat = LOCAL_SOURCE.stat()
        if (local_stat.st_size, local_stat.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns):
            return
    temporary = LOCAL_SOURCE.with_name(LOCAL_SOURCE.name + ".partial")
    offset = temporary.stat().st_size if temporary.is_file() else 0
    if offset > source_stat.st_size:
        raise ValueError("Local source staging is longer than the accepted RDS")
    with formal_read_lease(SOURCE):
        with SOURCE.open("rb") as source, temporary.open("ab") as target:
            source.seek(offset)
            shutil.copyfileobj(source, target, length=1024*1024)
    shutil.copystat(SOURCE, temporary)
    os.replace(temporary, LOCAL_SOURCE)


class NativeReader:
    def __init__(self, values: np.ndarray, image_id: str):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.source = SimpleNamespace(path=LOCAL_SOURCE, pixel_size_um=None, image_id=image_id)

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def field(source_file: Path, source_table: str, source_field: str, target_table: str,
          target_field: str, rule: str, *, join: str = "", notes: str = "") -> dict[str, str]:
    return dict(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_table,
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join, value_mapping="", transformation_rule=rule, time_origin_rule="",
                conflict_rule="Leave unconfirmed sample or clinical associations empty.", notes=notes)


def source_tables(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = sorted(records, key=lambda item: item["native_proof"]["source_binding"]["image_index"])
    names = [item["source_label"] for item in ordered[0]["source_channel"]]
    if len(ordered) != 352 or [item["native_proof"]["source_binding"]["image_index"] for item in ordered] != list(range(1, 353)):
        raise ValueError("Accepted source image indices are not 1..352")
    if any([part["source_label"] for part in item["source_channel"]] != names for item in ordered):
        raise ValueError("Source panel order differs between images")
    panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
    sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv", nrows=0).columns.tolist()
    groups: dict[str, set[str]] = {}
    for item in ordered:
        identity = item["native_proof"]["physical_identity"]
        groups.setdefault(identity["patient_id"], set()).add(identity["group"])
    samples = []
    for source_id, values in sorted(groups.items()):
        if len(values) != 1:
            raise ValueError("One source Patient.ID has conflicting cohort groups")
        group = next(iter(values))
        row = dict.fromkeys(sample_columns, None)
        row.update(study=STUDY, sample_id=source_id, source_sample_id=source_id,
                   source_subject_id=source_id, patient_key=None if source_id.endswith("_Ctrl") else f"{STUDY}:{source_id}",
                   cohort_id=group, sample_type="lymph node tissue section",
                   anatomical_site="lymph node", preservation_method="FFPE",
                   case_control_status="control" if group == "HC" else "case",
                   primary_diagnosis=None if group == "HC" else "melanoma",
                   metadata_notes=("Source Patient.ID has a control suffix; its relationship to the primary source ID is unconfirmed. "
                                   if source_id.endswith("_Ctrl") else "")
                   + "The published Supplementary Table 1 has cohort aggregates, no individual clinical rows."
                   + (" HC is a cancer-free control LN cohort." if group == "HC" else ""))
        samples.append(row)
    fields = [
        field(SOURCE, "CytoImageList", "names(images)", "image_metadata", "source_image_id;source_files;source_axes;source_dtype",
              "The exact RDS list member identifies one complete source ROI; decode R column-major Y,X,C double values without intensity transformation."),
        field(META / "native_verification.json", "records[].native_proof", "source_binding.image_index;physical_identity.roi_id",
              "image_metadata", "image_id;roi_id;source_x0_px;source_y0_px;source_x1_px;source_y1_px",
              "Bind the accepted member ordinal and ROI.ID to its native dimensions; the entire ROI starts at (0,0).",
              join="names(images) = source_binding.image_name = SPE Image.name"),
        field(META / "native_verification.json", "records[].native_proof.physical_identity", "patient_id;group",
              "image_metadata", "sample_id;region_annotation", "Use exact source Patient.ID as source sample key; preserve the ROI.ID region code without expanding its meaning.",
              join="image ROI.ID = source proof ROI.ID = SPE ROI.ID"),
        field(META / "native_verification.json", "records[].native_proof.physical_identity", "patient_id;group",
              "sample_metadata", "sample_id;source_sample_id;source_subject_id;patient_key;cohort_id;case_control_status",
              "One released LN source sample per exact Patient.ID; unique source IDs connect to patient_key except the unresolved control-suffixed ID.",
              join="image sample_id = exact source Patient.ID",
              notes=(f"{len(ordered)} accepted image records connect to {len(samples)} distinct source sample tokens; "
                     f"{sum(row['patient_key'] is not None for row in samples)}/{len(samples)} samples "
                     "have a verified patient key. One control-suffixed token has no confirmed patient bridge.")),
        field(META / "supplementary_tables.pdf", "Supplementary Table 1", "cohort aggregates",
              "sample_metadata", "metadata_notes", "No person-level values can be joined from cohort aggregates.",
              notes="Individual age, sex, primary-tumour features and recurrence times remain empty."),
        field(CACHE / "paper_methods_excerpt.json", "published Methods", "FFPE LN; ROI region definitions",
              "sample_metadata", "sample_type;anatomical_site;preservation_method;primary_diagnosis",
              "Use study-level tissue processing and source Group to distinguish melanoma TDLN from cancer-free cLN."),
        field(S1, "Study rows", "Original marker;Unified marker;Segmentation channel",
              "panel_mapping", "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
              "Match every original source channel in its RDS order; keep S1 admitted measurement and segmentation channels."),
    ]
    return panel, pd.DataFrame(samples), pd.DataFrame(fields)


def image_row(item: dict, result: dict) -> dict:
    proof = item["native_proof"]
    identity = proof["physical_identity"]
    height, width, channels = proof["actual_image_dimensions"]
    image_id = item["physical_image_id"]
    path = OUTPUT / "images" / image_id / "input.tiff"
    row = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv", nrows=0).columns.tolist(), None)
    row.update(study=STUDY, image_id=image_id, sample_id=identity["patient_id"], panel_id=PANEL_ID,
               image_kind="roi", source_image_id=identity["image_name"], source_files=[relative(SOURCE)],
               source_format="rds", source_axes="R array YXC, column-major", source_dtype="float64",
               source_revision="Zenodo 20038486 revision 6", roi_id=identity["roi_id"],
               region_annotation=identity["roi_id"].split("_")[1],
               source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
               pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
               pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
               height_px=result["height_px"], width_px=result["width_px"],
               source_channel_count=channels, channel_count=result["channel_count"],
               input_image_path=relative(path),
               intensity_description="Released images_spe.rds double intensities; CellWeft preserves decoded values without normalization.",
               split_method="already_independent",
               split_evidence=relative(META / "native_verification.json") + ":records[].native_proof.source_binding",
               metadata_notes="Source ROI has no independently supplied pixel size. "
               + ("Control-suffixed source sample has unresolved patient linkage." if identity["patient_id"].endswith("_Ctrl") else ""))
    return row


def preparer_results(images: pd.DataFrame, samples: pd.DataFrame, *, pixel_checks: list[str]) -> None:
    checks = pd.read_csv(TEMPLATES / "review_checks.csv", keep_default_na=False)
    linked_images = int(images.sample_id.isin(samples.sample_id).sum())
    patient_images = int(images.merge(samples[["sample_id", "patient_key"]], on="sample_id", how="left")
                         .patient_key.notna().sum())
    patient_samples = int(samples.patient_key.notna().sum())
    rows = []
    for check in checks.itertuples(index=False):
        if check.check_id == "D01":
            result, details = "unresolved", "Preparer delivery closed; independent review and D01 binding remain pending."
        elif check.check_id == "M01":
            result = "unresolved"
            details = (f"{linked_images}/{len(images)} exported images join {len(samples)} source samples; "
                       f"{patient_images}/{len(images)} images and {patient_samples}/{len(samples)} samples "
                       "have a verified patient key. One control-suffixed source sample has no verified "
                       "patient bridge. The published clinical table gives cohort aggregates only.")
        elif check.check_id == "I05":
            result = "pass" if pixel_checks else "unresolved"
            details = "Preparer block comparisons: " + "; ".join(pixel_checks)
        else:
            result, details = "pass", "Preparer self-check for the delivered scope; independent review pending."
        if check.check_id == "S01":
            details = "352 released RDS ROIs; 20 further paper-reported ROIs not localized in the selected release. " + details
        rows.append(dict(study=STUDY, preparer="VanKrimpen_preparer_0605", reviewer="", scope=check.scope,
                         record_id=STUDY, check_id=check.check_id, result=result,
                         source_evidence=relative(META / "native_verification.json") + ";" + relative(META / "supplementary_tables.pdf"),
                         output_evidence=relative(OUTPUT), details=details))
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        existing = pd.read_csv(path, keep_default_na=False) if path.is_file() else pd.DataFrame(columns=rows[0].keys())
        current = existing.loc[~((existing.preparer == "VanKrimpen_preparer_0605") & (existing.reviewer == ""))]
        merged = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        temporary = path.with_name(path.name + ".partial")
        merged.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


def progress(stage: str, *, next_index: int, images: pd.DataFrame, samples: pd.DataFrame, issues: list[str]) -> None:
    value = dict(study=STUDY, phase="prepare", scientific_repair_round=0,
                 stage=stage, updated_at=datetime.now(timezone.utc).isoformat(),
                 expected_images=352, completed_images=len(images), next_source_image_index=next_index,
                 sample_rows=len(samples) if len(images) else 0,
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(images) else 0,
                 image_index=relative(OUTPUT / "images.parquet"), issues=issues)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with formal_publish_lease(JOB / "progress.json", directory=JOB, lock_name=".checkpoint.lock"):
        for path in (JOB / "progress.json", CACHE / "export_state.json"):
            temporary = path.with_name(path.name + ".partial")
            temporary.write_text(payload)
            os.replace(temporary, path)


def pixel_check(image_id: str, values: np.ndarray, panel: pd.DataFrame, result: dict) -> str:
    path = OUTPUT / "images" / image_id / "input.tiff"
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    block = min(32, values.shape[1], values.shape[2])
    positions = [(0, 0, block, block),
                 (values.shape[2]-block, values.shape[1]-block, values.shape[2], values.shape[1])]
    with formal_read_lease(path), tifffile.TiffFile(path) as tif:
        expected_names = kept.marker_id.tolist()
        observed_names = [node.attrib.get("Name") for node in ET.fromstring(tif.ome_metadata).iter()
                          if node.tag.endswith("Channel")]
        side = max(values.shape[1:])
        if (len(tif.pages) != len(kept) or observed_names != expected_names
                or any(page.shape != (side, side) or page.dtype != values.dtype for page in tif.pages)):
            raise ValueError(f"TIFF header or channel order differs from source: {image_id}")
        for output_index, source_index in enumerate(kept.source_channel_index):
            saved = tif.pages[output_index].asarray()
            for x0, y0, x1, y1 in positions:
                expected = values[source_index, y0:y1, x0:x1]
                left, top = result["pad_left_px"], result["pad_top_px"]
                actual = saved[y0+top:y1+top, x0+left:x1+left]
                if not np.array_equal(actual, expected):
                    raise ValueError(f"Pixel mismatch: {image_id}, channel {source_index}, block {(x0,y0,x1,y1)}")
    return f"{image_id}: all {len(kept)} channels, top-left and bottom-right edge {block}x{block} blocks"


def main() -> None:
    with formal_read_lease(META / "native_verification.json"):
        verification = json.loads((META / "native_verification.json").read_text())
    records = sorted(verification["records"], key=lambda item: item["native_proof"]["source_binding"]["image_index"])
    if not verification["scope_complete"] or len(records) != 352:
        raise ValueError("Accepted source scope is incomplete")
    proof_stat = records[0]["native_proof"]["source_file_stat"]
    source_stat = SOURCE.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (proof_stat["size"], proof_stat["mtime_ns"]):
        raise ValueError("Source RDS version differs from accepted native proof")
    with formal_read_lease(S1):
        panel, samples, fields = source_tables(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    publish_json(CACHE / "paper_methods_excerpt.json", dict(
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13645638/",
        source_sections=["Image acquisition", "Cell segmentation", "Supplementary Table 1"],
        findings=["FFPE lymph-node material was sectioned for IMC.",
                  "The paper reports 372 analysed ROIs, while its released RDS has 352 accepted ROIs.",
                  "Supplementary Table 1 reports cohort aggregates, not patient-level clinical rows."],
        sample_linkage="The RDS source Patient.ID and ROI.ID provide the exact local image-to-sample keys."))
    with formal_read_lease(OUTPUT / "images.parquet"):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
    already = {row.image_id for row in images.itertuples(index=False)
               if (NAS / row.input_image_path).is_file()} if len(images) else set()
    indexed = {row["image_id"]: row for row in images.to_dict("records")} if len(images) else {}
    next_missing = next((index for index, item in enumerate(records, 1)
                         if item["physical_image_id"] not in already), len(records)+1)
    progress("staging_source", next_index=next_missing, images=images, samples=samples, issues=[])
    stage_source()
    pixel_checks: list[str] = []
    pending: list[dict] = []

    class ImageParser(ParserXDR):
        def parse_double_array(self) -> np.ndarray:
            length = self.parse_int()
            if length < 500_000:
                return self._parse_array_values(np.float64, length)
            position = self.image_count
            if position >= len(records):
                raise ValueError("More image arrays than accepted source records")
            item = records[position]
            h, w, c = item["native_proof"]["actual_image_dimensions"]
            if length != h*w*c:
                raise ValueError(f"Image array {position+1} has {length} values, expected {h*w*c}")
            image_id = item["physical_image_id"]
            raw = self.file.read(length*8)
            if len(raw) != length*8:
                raise EOFError(f"Truncated RDS image array {position+1}")
            self.image_count += 1
            if image_id in already:
                if position in (0, 89):
                    values = np.frombuffer(raw, dtype=">f8").astype(np.float64).reshape((h, w, c), order="F").transpose(2, 0, 1)
                    pixel_checks.append(pixel_check(image_id, values, panel, indexed[image_id]))
                return np.empty(0, dtype=np.float64)
            values = np.frombuffer(raw, dtype=">f8").astype(np.float64).reshape((h, w, c), order="F").transpose(2, 0, 1)
            destination = OUTPUT / "images" / image_id / "input.tiff"
            if destination.is_file():
                side = max(h, w)
                result = dict(pad_left_px=(side-w)//2, pad_right_px=side-w-(side-w)//2,
                              pad_top_px=(side-h)//2, pad_bottom_px=side-h-(side-h)//2,
                              height_px=side, width_px=side, channel_count=int(panel.keep_channel.sum()))
                pixel_checks.append(pixel_check(image_id, values, panel, result))
            else:
                reader = NativeReader(values, image_id)
                result = export_input(reader, panel, destination)
                if not result["admitted"]:
                    raise ValueError(f"Accepted source ROI has no retained downstream signal: {image_id}")
                if position in (0, 89):
                    pixel_checks.append(pixel_check(image_id, values, panel, result))
            pending.append(image_row(item, result))
            if len(pending) == 4:
                flush(position+2)
            print(json.dumps(dict(image_index=position+1, exported_images=len(already)+len(pending))), flush=True)
            return np.empty(0, dtype=np.float64)

    def flush(next_index: int) -> None:
        nonlocal images
        if not pending:
            return
        incoming = pd.DataFrame(pending)
        print(json.dumps(dict(stage="publishing_tables", next_source_image_index=next_index,
                              batch_images=len(incoming))), flush=True)
        save_input_tables(OUTPUT, incoming, samples, panel, fields)
        already.update(incoming.image_id)
        with formal_read_lease(OUTPUT / "images.parquet"):
            images = pd.read_parquet(OUTPUT / "images.parquet")
        pending.clear()
        progress("exporting", next_index=next_index, images=images, samples=samples, issues=[])
        print(json.dumps(dict(stage="tables_published", completed_images=len(images))), flush=True)

    with gzip.open(LOCAL_SOURCE, "rb") as stream:
        if stream.read(2) != b"X\n":
            raise ValueError("Expected XDR RDS serialization")
        parser = ImageParser(memoryview(b""), expand_altrep=True, altrep_constructor_dict=DEFAULT_ALTREP_MAP)
        parser.file = stream
        parser.image_count = 0
        parser.parse_all()
        if parser.image_count != len(records) or stream.read(1):
            raise ValueError("RDS image count or trailing bytes differ from accepted scope")
    flush(353)
    if len(images) != 352:
        raise ValueError(f"Only {len(images)} of 352 image rows were published")
    save_input_tables(OUTPUT, images.iloc[0:0], samples, panel, fields)
    preparer_results(images, samples, pixel_checks=pixel_checks)
    progress("prepared_for_independent_review", next_index=353, images=images, samples=samples,
             issues=["One control-suffixed sample has unresolved patient linkage.",
                     "Published clinical table supplies cohort aggregates only.",
                     "20 paper-reported ROIs are absent from selected public RDS release."])
    print(json.dumps(dict(study=STUDY, expected_images=352, exported_images=len(images),
                          sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                          patient_linked_samples=int(samples.patient_key.notna().sum()),
                          pixel_checks=pixel_checks)), flush=True)


if __name__ == "__main__":
    main()
