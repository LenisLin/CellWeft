"""Deliver the 12 admitted Allam SPORA IMC cores as native-intensity inputs.

Inputs: SPORA Zarr/Parquet, accepted panel scope, original verification, S1,
and the article's Supplementary Table 2 (downloaded into this study's cache).
Outputs: per-core input.tiff, study input tables, resumable receipts and QC.
The public table's explicit metastatic-of relation groups paired specimens.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Iterator
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Allam2022_IMC_Lung/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.raw_consolidation import file_identity
from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Allam2022_IMC_Lung"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path)
parser.add_argument("--metadata-only", action="store_true", help="Prepare metadata, or refresh completed input receipts")
parser.add_argument("--repair-field-mapping", action="store_true",
                    help="Repair M02 provenance only; preserve TIFFs and existing image/sample/panel tables")
args = parser.parse_args()
root = args.data_root
raw = root / "raw/public" / STUDY / "spora"
RAW_METADATA = root / "raw/public" / STUDY / "metadata"
output = root / "processed" / STUDY
cache = output / "preparation_evidence"
evidence = cache / "source_metadata"
receipts = cache / "image_receipts"
temporary = Path("/tmp/study_input_batch") / STUDY
for directory in (cache, evidence, output, receipts, temporary):
    directory.mkdir(parents=True, exist_ok=True)
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)


@contextmanager
def supplementary_member(filename: str) -> Iterator[Path]:
    """Restore one accepted original ZIP attachment to scoped SSD scratch."""
    manifest = RAW_METADATA.parent / "manifest.json"
    archive = RAW_METADATA / "supplementary_files.zip"
    with tempfile.TemporaryDirectory(prefix=f"{STUDY}_supplement_", dir=temporary) as directory:
        destination = Path(directory) / filename
        with formal_read_lease(archive):
            proof = json.loads(manifest.read_text())["accepted_archive_members"][filename]
            expected = proof["archive_identity"]
            if any(file_identity(archive)[key] != expected[key] for key in
                   ("device", "inode", "size", "mtime_ns")):
                raise ValueError("Accepted source ZIP version changed")
            with zipfile.ZipFile(archive) as source_zip:
                member = source_zip.getinfo(proof["member"])
                if (member.file_size, member.compress_size, member.CRC, member.header_offset) != (
                        proof["size"], proof["compressed_size"], proof["crc"], proof["header_offset"]):
                    raise ValueError("Accepted source ZIP member changed")
                with source_zip.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.stat().st_size != proof["size"]:
                raise ValueError("Incomplete source ZIP member restoration")
        yield destination


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, root))


def save_json(path: Path, value: dict | list) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def progress(stage: str, count: int, **details: object) -> None:
    if args.progress:
        save_json(args.progress, dict(study=STUDY, stage=stage, completed_images=count,
                                     expected_images=12, updated_utc=datetime.now(timezone.utc).isoformat(),
                                     gaps=["独立复核及主控验收待执行"], **details))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


class StagedReader:
    """Read a disk-backed native CYX array after one bounded Zarr decode.

    SPORA packs all channels in each spatial chunk. Shared channel-major TIFF
    export would decode/read that same source chunk 25 times. Stage one core in
    256-pixel blocks to local disk; no complete CYX image is allocated in RAM.
    This source-specific adaptation can later migrate to a shared staging reader.
    """

    def __init__(self, array: np.memmap, source: SourceSpec):
        self.array, self.source = array, source
        self.shape, self.dtype = array.shape, array.dtype

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


# Reuse the accepted 12-image scope and original source-size verification.
scope_path = cache / "reviewed_scope_ledger.json"
scope_document = json.loads(scope_path.read_text())
scope = scope_document.get("studies", scope_document)[STUDY]
verification_path = cache / "native_verification.json"
verified = {item["image_id"]: item for item in json.loads(verification_path.read_text())["verified"]}
selected = sorted(scope["eligible_acquisition_ids"])
assert len(selected) == 12
tissues_path = raw / "metadata/tissues.parquet"
tissues = pd.read_parquet(tissues_path).set_index("tissue_id", verify_integrity=True)
channel_table = pd.read_parquet(raw / "imc/channels.parquet")
availability = pd.read_parquet(raw / "imc/channels_per_tissue.parquet")
ordered_path = metadata_source_directory(root, STUDY) / 'imc_ordered_channels_by_tissue.json'
ordered = json.loads(ordered_path.read_text())
assert availability.columns.tolist() == channel_table.channel_name.tolist()
source_names = channel_table.loc[availability.loc[selected[0]].to_numpy(bool), "channel_name"].tolist()
panel_id = f"{STUDY}__spora_native26"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(panel) == 26 and len(kept) == 25
assert panel.loc[panel.nuclear, "source_name"].tolist() == ["Histone 3"]

inventory = []
for image_id in selected:
    proof = verified[image_id]
    group = root / proof["path"]
    files = sorted(p for p in group.rglob("*") if p.is_file() and not p.name.endswith(".partial"))
    group_metadata = json.loads((group / "zarr.json").read_text())["attributes"]["ome"]
    array_metadata = json.loads((group / "0/zarr.json").read_text())
    actual_names = channel_table.loc[availability.loc[image_id].to_numpy(bool), "channel_name"].tolist()
    assert actual_names == source_names == [item["channel_name"] for item in ordered[image_id]]
    assert source_names == [item["label"] for item in group_metadata["omero"]["channels"]]
    assert source_names == [item["channel_name"] for item in proof["marker_mapping"]]
    assert array_metadata["shape"] == proof["shape"] == [26, 1500, 1500]
    assert array_metadata["data_type"] == proof["dtype"] == "float64"
    assert len(files) == proof["source_object_count"] == 6
    assert sum(path.stat().st_size for path in files) == proof["source_bytes"]
    assert all((group / f"0/c/0/{y}/{x}").is_file() for y in range(2) for x in range(2))
    multiscale = group_metadata["multiscales"][0]
    assert [item["name"] for item in multiscale["axes"]] == ["c", "y", "x"]
    scale = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [1.0, 1.0, 1.0]
    inventory.append(dict(image_id=image_id, source_path=relative(group), source_bytes=proof["source_bytes"],
                          files=[dict(path=relative(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                 for p in files], shape=proof["shape"], dtype=proof["dtype"],
                          ignored_partial_files=[relative(p) for p in group.rglob("*.partial")]))
if not args.repair_field_mapping:
    save_json(cache / "source_inventory.json", dict(selected=inventory, scope_evidence=relative(scope_path),
          reused_verification=relative(verification_path), excluded_source_images=[item for item in ordered if item not in selected],
          exclusion_reason=scope["reason"], source_bytes=sum(row["source_bytes"] for row in inventory),
          maximum_uncompressed_output_bytes=12 * 25 * 1500 * 1500 * 8,
          maximum_local_staging_bytes=26 * 1500 * 1500 * 8))

# Recover source clinical rows without embedding a patient/core list in code.
pdf_reference = relative(RAW_METADATA / "supplementary_files.zip") + "#41698_2022_305_MOESM1_ESM.pdf"
text_path = evidence / "supplementary_materials.txt"
if not args.repair_field_mapping:
    with supplementary_member("41698_2022_305_MOESM1_ESM.pdf") as pdf:
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True)
pages = text_path.read_text().split("\f")
page_number, clinical_page = next((i, text) for i, text in enumerate(pages, 1) if "TMA: LC814a" in text)
section = clinical_page.split("TMA: LC814a (Cohort 1)", 1)[1].split("TMA: LC1002a", 1)[0]
clinical_rows = []
for block in re.split(r"\n(?=[A-H]\d+\s+\d+\s+[MF]\s+)", section):
    row_text = " ".join(block.split())
    if not row_text:
        continue
    match = re.fullmatch(r"([A-H]\d+)\s+(\d+)\s+([MF])\s+(Lymph Node|Lung)\s+(.+?)\s+(T\d+N\d+M\d+|–)\s+(\d+|–)\s+([IV]+[ABC]?|–)\s+Malignant\s*(.*)", row_text)
    if match is None:
        raise ValueError(f"Unparsed Supplementary Table 2 row: {row_text}")
    core, age, sex, site, diagnosis_start, tnm, grade, stage, diagnosis_end = match.groups()
    diagnosis = (diagnosis_start + " " + diagnosis_end).strip()
    metastatic_of = re.search(r" of ([A-H]\d+)$", diagnosis)
    primary_core = metastatic_of.group(1) if metastatic_of else core
    assert (site == "Lymph Node") == (metastatic_of is not None)
    clinical_rows.append(dict(tma_id="LC814a", cohort="Cohort 1", core=core, age=age, sex=sex,
                              anatomical_site=site, diagnosis=diagnosis, tnm=None if tnm == "–" else tnm,
                              grade=None if grade == "–" else grade, stage=None if stage == "–" else stage,
                              primary_core=primary_core, source_page=page_number, source_row_text=row_text))
clinical = pd.DataFrame(clinical_rows).set_index("core", verify_integrity=True)
assert len(clinical) == 12 and clinical.primary_core.nunique() == 6
if not args.repair_field_mapping:
    clinical.reset_index().to_csv(evidence / "supplementary_table2_cohort1.csv", index=False, encoding="utf-8-sig")
article = ElementTree.parse(RAW_METADATA / "article.xml")
methods = " ".join("".join(item.itertext()) for item in article.iter("sec")
                   if item.findtext("title") == "FFPE tissues")
assert "formalin-fixed paraffin-embedded" in methods

sample_rows, image_samples = [], {}
for image_id in selected:
    tissue = tissues.loc[image_id]
    core = str(tissue.original_patient_identifier).removeprefix("IMC_cohort_1")
    assert str(tissue.original_patient_identifier) == "IMC_cohort_1" + core
    assert str(tissue.original_file_name).endswith("_" + core)
    clinical_row = clinical.loc[core]
    primary = clinical.loc[clinical_row.primary_core]
    is_node = clinical_row.anatomical_site == "Lymph Node"
    site_id = "lymph_node" if is_node else "lung"
    sample_id = f"LC814a__{clinical_row.primary_core}__{site_id}"
    image_samples[image_id] = (sample_id, core)
    row = dict.fromkeys(columns("sample_metadata"))
    row.update(study=STUDY, sample_id=sample_id, patient_key=f"{STUDY}__LC814a__{clinical_row.primary_core}",
               cohort_id="LC814a_Cohort1", source_sample_id=f"LC814a:{core}",
               biospecimen_group_id=sample_id, sample_type="tissue", tissue_type="tumor",
               anatomical_site=clinical_row.anatomical_site, lesion_type="metastasis" if is_node else "primary",
               matched_sample_id=f"LC814a__{clinical_row.primary_core}__{'lung' if is_node else 'lymph_node'}",
               preservation_method="ffpe", processing_method="TMA core; metal-tagged IMC staining; source-published 1_um Zarr",
               case_control_status="case", sex_reported={"M": "male", "F": "female"}[clinical_row.sex],
               primary_diagnosis=primary.diagnosis, primary_site="lung", histology=primary.diagnosis,
               grade=clinical_row.grade, age_reported_years=int(clinical_row.age),
               stage_reported=clinical_row.stage, tnm_reported=clinical_row.tnm,
               pathology_diagnosis_reported=clinical_row.diagnosis,
               metadata_notes=("Supplementary Table 2, PDF page 73: matched primary/node identities use the explicit 'of' relation. "
                               "SPORA patient_id identifies individual cores, not independent patients. "
                               "Age/stage/TNM timing and staging system are unspecified; retain reported values in extensions, "
                               "leave collection/diagnosis/index ages and index staging blank. Treatment, collection dates and survival unavailable. "
                               "Table 2 and Fig.1 identify this 12-core TMA as LC814a; Methods has contradictory cohort/TMA wording."))
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows).drop_duplicates(["study", "sample_id"])
assert len(samples) == 12 and samples.patient_key.nunique() == 6
dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
for field in dictionary[dictionary.table_name == "sample_metadata"].itertuples(index=False):
    if field.value_set:
        allowed = vocabulary.loc[vocabulary.value_set == field.value_set, "allowed_value"]
        assert samples[field.field_name].dropna().isin(allowed).all(), field.field_name


def mapping(source_file: str, source_table: str, source_field: str, target_table: str,
            target_field: str, join_keys: str, rule: str, notes: str,
            conflict: str = "Preserve source values; do not infer absent values.") -> dict:
    row = dict.fromkeys(columns("field_mapping"))
    row.update(study=STUDY, source_file=source_file, source_sheet_or_table=source_table,
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join_keys, transformation_rule=rule, conflict_rule=conflict, notes=notes)
    return row


field_rows = [
    mapping(relative(tissues_path), "tissues", "tissue_id;original_patient_identifier;original_file_name;specimen_id;patient_id",
            "image_metadata", "image_id;sample_id;source_image_id;core_id;roi_id;region_annotation", "tissue_id; original_patient_identifier core label + LC814a",
            "Join each published tissue ID to its full independent Zarr and explicit IMC_cohort_1 core label; connect to Supplementary Table 2. "
            "One biological specimen per primary-reference plus anatomical site; repeated views of that specimen share sample_id.",
            "12/12 admitted images connect to 12 biological specimens; patient groups require the supplementary metastatic-of relation.",
            "Do not equate the 12 distinct SPORA patient_id strings with 12 patients; specimen_id=0 is only local to that source identifier."),
    mapping(pdf_reference, "Supplementary Table 2, PDF page 73", "Patient Tissue;Pathology Diagnosis ('of <primary core>')",
            "sample_metadata", "sample_id;patient_key;source_sample_id;matched_sample_id;cohort_id;biospecimen_group_id",
            "LC814a + explicit core code; metastatic diagnosis -> referenced primary core",
            "Create six patient groups from explicit metastatic-of links; lung and lymph-node specimens remain separate sample rows.",
            "Local Parquet, primary article, Supplementary Table 2 and author repository metadata/ROI-processing notebooks inspected. "
            "12 image->sample joins, 12 sample->patient joins, six distinct patients. No shared collection date is inferred."),
    mapping(pdf_reference, "Supplementary Table 2, PDF page 73", "Sex;Organ/Anatomic Site;Pathology Diagnosis;Grade;Type",
            "sample_metadata", "sex_reported;anatomical_site;lesion_type;primary_diagnosis;primary_site;histology;grade;case_control_status;tissue_type",
            "LC814a + core; explicit primary_core relation for primary diagnosis",
            "Map M/F to male/female; Type=Malignant to case_control_status=case and tissue_type=tumor; primary versus metastatic by organ and explicit diagnosis. "
            "Primary diagnosis/histology come from the linked lung specimen; grade is the source row's direct value.",
            "All actual nonempty clinical values are reconstructed from the PDF table; node grades remain blank. Staging system and temporal origin unavailable.",
            "Prefer original Table 2 (IIIA/IIA) over SPORA's coarsened stage (III/II); retain raw TNM without re-staging."),
    mapping(relative(RAW_METADATA / "article.xml"), "Methods: FFPE tissues; Fig.1; Data; Images", "FFPE;TMA;ROI dimensions;published processing",
            "sample_metadata", "preservation_method;processing_method;sample_type",
            "public study; admitted 12-core cohort confirmed by Table 2",
            "Map FFPE/TMA tissue material to sample_type=tissue; record FFPE preservation and IMC TMA handling. "
            "Treat age/stage and treatment timing as unspecified; leave timepoint and index fields blank.",
            "Paper Methods reverses TMA/cohort descriptions; Table 2, Fig.1 and paired core identities identify the admitted cohort as LC814a."),
    mapping(relative(raw / "imc/channels.parquet") + ";" + relative(raw / "imc/channels_per_tissue.parquet"),
            "channels;channels_per_tissue", "channel_name;original_channel_name;availability",
            "panel_mapping", "panel_id;source_channel_index;source_name", "tissue_id + canonical row order",
            "Filter canonical channel rows by each tissue availability vector, preserving row order; verify all Zarr OMERO labels and prior ordered proof.",
            "26 ordered native channels agree for all 12 sources; official spora_io/datasets/multiplex.py reading rule reused."),
    mapping(relative(s1_path), "S1", "Study;Original marker;Unified marker;Segmentation channel",
            "panel_mapping", "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
            "Study + exact native Original marker", "Use the shared S1 mapper; retain 25 channels in source order; exclude H3K9me3; preserve Histone 3 nuclear input.",
            "No aliases or duplicate downstream marker selection required. source_channel_index and output_channel_index are distinct."),
    mapping(relative(cache / "source_inventory.json"), "selected", "source_path;shape;dtype;files;OME scale",
            "image_metadata", "source_files;source_revision;source_axes;source_dtype;source_format;geometry;pixel_size_x_um;pixel_size_y_um;input_image_path;intensity_description;split_method;split_evidence",
            "source image ID", "Keep the full published 1500x1500 independent ROI, float64 intensities and 1_um scale; zero square padding; lossless CYX OME BigTIFF.",
            "SPORA-native float64 is preserved without normalization. Original acquisition export/preprocessing may differ; no raw MCD-count claim. "
            "All four source shards must exist. Whole-image admission tests all retained downstream markers, independently of individual zero pixels."),
]
extension_definitions = {
    "age_reported_years": ("integer", "Age reported in Supplementary Table 2 in years; collection/diagnosis/index origin unspecified"),
    "stage_reported": ("string", "Source row stage in Supplementary Table 2; staging system and temporal origin unspecified"),
    "tnm_reported": ("string", "Source row TNM in Supplementary Table 2; staging system and temporal origin unspecified"),
    "pathology_diagnosis_reported": ("string", "Verbatim source row pathology diagnosis in Supplementary Table 2"),
}
for key, (kind, definition) in extension_definitions.items():
    source_field = {"age_reported_years": "Age", "stage_reported": "Stage", "tnm_reported": "TNM",
                    "pathology_diagnosis_reported": "Pathology Diagnosis"}[key]
    field_rows.append(mapping(pdf_reference, "Supplementary Table 2, PDF page 73", source_field, "sample_metadata", key,
                              "LC814a + core", "Preserve the reported value without assigning an unsupported clinical time origin.",
                              json.dumps(dict(data_type=kind, definition=definition))))
fields = pd.DataFrame(field_rows, columns=columns("field_mapping"))

# Round-1 M02 repair reuses the reviewed images and keeps independent failures
# open until a different reviewer confirms the repaired mapping and D01 binding.
if args.repair_field_mapping:
    from cellweft.data.preparation.inputs import input_binding

    progress("checking_M02_mapping_repair", 12)
    repair_path = cache / "field_mapping_repair_1.json"
    with nullcontext():
        images = pd.read_parquet(output / "images.parquet")
        delivered_samples = pd.read_csv(output / "sample_metadata.csv", keep_default_na=False)
        assert set(images.image_id) == set(selected)
        assert set(images.sample_id) == set(delivered_samples.sample_id) == set(samples.sample_id)
        expected = samples.set_index("sample_id")[["sample_type", "tissue_type"]]
        actual = delivered_samples.set_index("sample_id").loc[expected.index, expected.columns]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        previous_fields = pd.read_csv(output / "field_mapping.csv", keep_default_na=False)
        repaired_fields = fields.fillna("")
        changed_rows = [2, 3]
        pd.testing.assert_frame_equal(previous_fields.drop(index=changed_rows),
                                      repaired_fields.drop(index=changed_rows))
        pending = output / "field_mapping.csv.tmp"
        with formal_publish_lease(output / "field_mapping.csv", directory=output):
            repaired_fields.to_csv(pending, index=False, encoding="utf-8-sig")
            pending.replace(output / "field_mapping.csv")
        repair = dict(study=STUDY, repair_round=1, status="prepared", self_check="pass",
                      affected_check_id="M02", affected_sample_ids=delivered_samples.sample_id.tolist(),
                      source_evidence=[pdf_reference, relative(RAW_METADATA / "article.xml")],
                      changed_target_fields=["sample_type", "tissue_type"], added_source_fields=["Type"],
                      summary="补齐组织/TMA材料→sample_type=tissue及Type=Malignant→tissue_type=tumor；12份sample现有值与来源一致。",
                      expected_images=12, exported_images=len(images), sample_rows=len(delivered_samples),
                      sample_linked_images=int(images.sample_id.isin(delivered_samples.sample_id).sum()),
                      patient_linked_samples=int(delivered_samples.patient_key.ne("").sum()),
                      unique_patients=int(delivered_samples.patient_key.nunique()),
                      reused_tiffs=len(images), newly_exported_images=0,
                      candidate_input_binding=input_binding(output, root, selected),
                      independent_recheck="M02 and D01 pending; previous independent rows retained")
        save_json(repair_path, repair)
        review_path = output / "review_results.csv"
        reviews = pd.read_csv(review_path, keep_default_na=False)
        details = json.dumps(dict(repair_round=1, summary="字段映射修补完成，整理者自检通过；待独立复查M02并刷新D01绑定。",
                                  image_ids=selected, candidate_binding_evidence=relative(repair_path)), ensure_ascii=False)
        reviews.loc[len(reviews)] = dict(study=STUDY, preparer="codex_Allam2022_IMC_Lung_0129_repair_1", reviewer="",
                                        scope="study", record_id=STUDY, check_id="D01", result="unresolved",
                                        source_evidence=pdf_reference + ";" + relative(RAW_METADATA / "article.xml"),
                                        output_evidence=relative(output / "field_mapping.csv") + ";" + relative(repair_path),
                                        details=details)
        pending_review = output / "review_results.csv.tmp"
        with formal_publish_lease(review_path, directory=output):
            reviews.to_csv(pending_review, index=False, encoding="utf-8-sig")
            pending_review.replace(review_path)
    progress("repaired_pending_independent_recheck", len(images), sample_rows=len(delivered_samples),
             sample_linked_images=repair["sample_linked_images"], patient_linked_samples=repair["patient_linked_samples"],
             unique_patients=repair["unique_patients"], newly_exported_images=0,
             repair_evidence=relative(repair_path))
    print(json.dumps({key: value for key, value in repair.items() if key != "candidate_input_binding"},
                     ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)

# Publish available identity evidence while the managed TIFF writer is busy.
if args.metadata_only and not any(receipts.glob("*.json")):
    prior_index = output / "images.parquet"
    images = pd.read_parquet(prior_index) if prior_index.exists() else pd.DataFrame(columns=columns("image_metadata"))
    save_input_tables(output, images, samples, panel, fields)
    progress("metadata_prepared_waiting_for_managed_NAS_write_slot", len(images),
             sample_rows=len(samples), patient_linked_samples=12, unique_patients=6)
    print(json.dumps(dict(study=STUDY, stage="metadata_prepared", exported_images=len(images),
                          sample_rows=12, unique_patients=6), ensure_ascii=False), flush=True)
    raise SystemExit(0)

# Each completed image is published immediately; previous batch rows are merged.
index_path = output / "images.parquet"
image_rows = pd.read_parquet(index_path).to_dict("records") if index_path.exists() else []
existing = {row["image_id"]: row for row in image_rows}
all_receipts = {}
for item in inventory:
    image_id = item["image_id"]
    destination = output / "images" / image_id / "input.tiff"
    receipt_path = receipts / f"{image_id}.json"
    revision = json.dumps(dict(spora_scale="imc/1_0mpp", files=item["files"]), separators=(",", ":"))
    sample_id, core = image_samples[image_id]
    reusable = receipt_path.exists() and destination.is_file()
    if reusable:
        receipt = json.loads(receipt_path.read_text())
        reusable = receipt["source_revision"] == revision and receipt["channel_names"] == kept.marker_id.tolist()
    if not reusable:
        if args.metadata_only:
            raise ValueError(f"Completed unchanged input receipt unavailable for {image_id}")
        if shutil.disk_usage(root).free < 100 * 1024**3 + 25 * 1500 * 1500 * 8:
            raise RuntimeError("Insufficient NAS reserve for next image")
        if shutil.disk_usage(temporary).free < 64 * 1024**3 + 26 * 1500 * 1500 * 8:
            raise RuntimeError("Insufficient local reserve for next image")
        progress("decoding_native_source", len(all_receipts), current_image=image_id)
        source = SourceSpec(image_id, panel_id, "zarr", root / item["source_path"] / "0", tuple(range(26)),
                            pixel_size_um=(1.0, 1.0), source_revision=revision)
        stage_path = temporary / f"{image_id}.npy"
        array = np.lib.format.open_memmap(stage_path, mode="w+", dtype="float64", shape=(26, 1500, 1500))
        try:
            with open_source(source) as reader:
                for window in windows((1500, 1500), 256):
                    x0, y0, x1, y1 = window
                    array[:, y0:y1, x0:x1] = reader.read_window(list(range(26)), window)
            array.flush()
            progress("exporting_input", len(all_receipts), current_image=image_id)
            exported = export_input(StagedReader(array, source), panel, destination, tile_size=256)
            assert exported["admitted"], f"All retained markers are zero for {image_id}"
            assert exported["valid_pixels"] is None
            # QC overview scaling is visualization-only; input pixels remain native.
            overview = []
            for source_name in ("Histone 3", "PanKeratin", "CD68"):
                plane = np.asarray(array[source_names.index(source_name), ::5, ::5]).copy()
                high = np.quantile(plane, 0.99)
                overview.append(np.clip(plane / high, 0, 1) if high > 0 else plane)
            plt.imsave(cache / f"overview_{image_id}.png", np.stack(overview, axis=-1))
            receipt = dict(image_id=image_id, source_revision=revision, channel_names=kept.marker_id.tolist(),
                           exported=exported, preparer_pixel_checks=[])
            with tifffile.TiffFile(destination) as tif:
                for window in [(0, 0, 32, 32), (734, 734, 766, 766), (1468, 1468, 1500, 1500)]:
                    x0, y0, x1, y1 = window
                    for channel in kept.itertuples(index=False):
                        actual = read_tiff_window(tif, int(channel.output_channel_index), window)
                        assert actual.dtype == array.dtype
                        assert np.array_equal(actual, array[channel.source_channel_index, y0:y1, x0:x1])
                    receipt["preparer_pixel_checks"].append(dict(window_xyxy=window, channel_count=25, exact=True))
            save_json(receipt_path, receipt)
        finally:
            array._mmap.close()
            stage_path.unlink(missing_ok=True)
    with tifffile.TiffFile(destination) as tif:
        names = [element.attrib["Name"] for element in ElementTree.fromstring(tif.ome_metadata).iter()
                 if element.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()
        assert tif.series[0].shape == (25, 1500, 1500) and tif.series[0].dtype == np.dtype("float64")
    exported = receipt["exported"]
    values = dict.fromkeys(columns("image_metadata"))
    values.update(study=STUDY, image_id=image_id, sample_id=sample_id, panel_id=panel_id, image_kind="tma_core",
                  source_image_id=image_id, source_files=[item["source_path"]], source_format="ome_zarr",
                  source_axes="CYX", source_dtype="float64", source_revision=revision,
                  core_id=f"LC814a:{core}", roi_id=tissues.loc[image_id].original_file_name,
                  region_annotation=tissues.loc[image_id].original_file_name,
                  source_x0_px=0, source_y0_px=0, source_x1_px=1500, source_y1_px=1500,
                  pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                  height_px=1500, width_px=1500, source_channel_count=26, channel_count=25,
                  pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                  intensity_description="Unmodified SPORA-published float64 intensity values at 1_um scale; lossless TIFF, no normalization; not asserted raw MCD counts.",
                  split_method="already_independent",
                  split_evidence=f"Published tissue_id={image_id}; original_file_name={tissues.loc[image_id].original_file_name}; "
                                 f"LC814a core {core} in Supplementary Table 2 PDF page 73; full source ROI retained.",
                  metadata_notes="Explicit Table 2 metastatic-of patient linkage; SPORA per-core patient_id is not the patient grouping. "
                                 "All source pixels retained; no sample split or inferred tissue mask.")
    existing[image_id] = values
    images = pd.DataFrame(existing.values(), columns=columns("image_metadata"))
    save_input_tables(output, images, samples, panel, fields)
    all_receipts[image_id] = receipt
    progress("image_saved", len(all_receipts), current_image=image_id, sample_rows=len(samples), patient_groups=6)
    print(f"saved {len(all_receipts)}/12 {image_id}", flush=True)

# Preparer checks are recorded separately from the independent review identity.
review_rows = []
check_scopes = pd.read_csv(TEMPLATES / "review_checks.csv").set_index("check_id")["scope"]
for check in ["S01", "I01", "I02", "I03", "I04", "P01", "P02", "P03", "I05", "M01", "M02", "D01"]:
    details = {
        "S01": "12 admitted sources exported; 14 source records excluded by accepted 19-channel panel scope. All source object counts/bytes agree with reused proof; one stale raw partial ignored.",
        "I01": "12 unique image identities, source groups and TIFF paths.",
        "I02": "12/12 images connect to 12 samples and one panel.",
        "I03": "Full 1500x1500 published frame; native 1_um pixels; no padding or resampling.",
        "I04": "Source files are independent core ROIs; core labels match Table 2. Preparer visual confirmation of all overviews is recorded in overview_review.json.",
        "P01": "26 channels: Parquet availability-filtered row order, metadata JSON, OMERO labels and original proof agree.",
        "P02": "25 retained channels; H3K9me3 excluded by S1; Histone 3 nuclear and five body channels retained; no duplicate downstream marker.",
        "P03": "All 12 TIFF headers checked: CYX 25x1500x1500 float64, exact OME channel names/order.",
        "I05": "Preparer exact comparison for all 12 images, all 25 retained channels, three 32x32 blocks (origin, center, lower-right); no padding. Receipt gives coordinates. Independent sampling pending.",
        "M01": "Local Parquet, original paper, Supplementary Table 2 (page 73), Zenodo record and author repository inspected. 12 specimens and six explicit primary/node patient groups: 12/12 image joins and 12/12 patient-linked sample rows.",
        "M02": "Clinical rows parsed from original PDF and visually checked. SPORA stage is coarsened; Table 2 preserves IIIA/IIA. Unknown clinical time origins retained via defined reported-value extensions. No treatment/survival data available.",
        "D01": "Input preparation completed for 12 admitted sources; different-context independent review and controller acceptance pending.",
    }[check]
    scope_name = check_scopes[check]
    record_ids = {"study": [STUDY], "image": selected, "panel": [panel_id], "sample": samples.sample_id.tolist()}[scope_name]
    for record_id in record_ids:
        review_rows.append(dict(study=STUDY, preparer="codex_Allam2022_IMC_Lung_prepare_0", reviewer=None,
                                scope=scope_name, record_id=record_id, check_id=check,
                                result="unresolved" if check == "D01" or
                                (check == "I04" and not (cache / "overview_review.json").exists()) else "pass",
                                source_evidence=relative(cache / "source_inventory.json") + ";" + pdf_reference,
                                output_evidence=relative(output / "images.parquet") + ";" + relative(receipts),
                                details="Preparer self-check only; independent reviewer not assigned here. " + details))
review_path = output / "review_results.csv"
reviews = pd.DataFrame(review_rows, columns=columns("review_results"))
if review_path.exists():
    prior = pd.read_csv(review_path)
    independent = prior[prior.reviewer.notna()]
    reviews = pd.concat([independent, reviews], ignore_index=True)
with formal_publish_lease(review_path, directory=review_path.parent):
    reviews.to_csv(review_path, index=False, encoding="utf-8-sig")
summary = dict(study=STUDY, status="prepared", expected_images=12, exported_images=len(all_receipts),
               source_metadata_records=len(tissues), excluded_19_channel_records=14,
               sample_rows=len(samples), sample_linked_images=len(images), patient_linked_samples=int(samples.patient_key.notna().sum()),
               unique_patients=int(samples.patient_key.nunique()), source_channel_count=26, output_channel_count=25,
               native_dtype="float64", output_shape=[25, 1500, 1500],
               output_file_bytes=sum((root / path).stat().st_size for path in images.input_image_path),
               preparer_exact_blocks=36, independent_review="pending",
               clinical_gaps=["Treatment/survival/collection dates absent; age/stage temporal origin and staging system unspecified"],
               controller_requests=["安排不同CLI上下文独立复核及主控验收。",
                                    "共享export_input对SPORA多通道chunk会重复解码；本study采用分块本地memmap暂存，后续可迁移至共享staging读取器。"],
               source_urls=["https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9437065/supplementaryFiles",
                            "https://doi.org/10.1038/s41698-022-00305-4", "https://zenodo.org/records/6784253",
                            "https://github.com/coskunlab/SpatialVizScore", "https://github.com/bunnelab/spora-io/blob/main/spora_io/datasets/multiplex.py"],
               evidence_paths=[relative(cache / "source_inventory.json"), pdf_reference,
                               relative(evidence / "supplementary_table2_cohort1.csv"), relative(review_path)])
save_json(cache / "input_delivery_summary.json", summary)
progress("prepared_pending_independent_review", len(all_receipts), sample_rows=len(samples), patient_linked_samples=12, unique_patients=6)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
