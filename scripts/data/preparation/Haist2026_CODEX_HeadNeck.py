"""Prepare 389 native Haist CODEX regions and source-declared specimen metadata.

Inputs: formal raw SPORA Zarr metadata, current processed image scope,
embedded channel order, study-specific S1, publication Table S3 and author metadata.
Outputs: square uint16 CYX input.tiff and the complete study input tables.
Run --inventory first; --run exports resumably with per-file I/O leases.
The preparer's checks are evidence for a separate reviewer, not acceptance.
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
from types import SimpleNamespace
import time
import tempfile
import zipfile
from typing import Iterator
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile
import zarr

from cellweft.data.raw_consolidation import file_identity
from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)

zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
STUDY = "Haist2026_CODEX_HeadNeck"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW_ROOT = NAS / "raw/public" / STUDY
RAW = NAS / "raw/public" / STUDY / "spora"
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
TEMP = Path("/tmp/study_input_batch") / STUDY
PROGRESS = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_PROGRESS",
    str(TEMP / "progress.json"),
))
REGION_FACTS = RAW_METADATA / "author_region_facts.csv"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "codex_59"
PREPARER = "codex_haist_0038_prepare_0"


@contextmanager
def supplementary_member(filename: str) -> Iterator[Path]:
    """Restore one accepted original ZIP attachment to scoped SSD scratch."""
    manifest = RAW_METADATA.parent / "manifest.json"
    archive = RAW_METADATA / "supplementary_files.zip"
    with tempfile.TemporaryDirectory(prefix=f"{STUDY}_supplement_", dir=TEMP) as directory:
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


def read_supplement_table(filename: str) -> pd.DataFrame:
    TEMP.mkdir(parents=True, exist_ok=True)
    with supplementary_member(filename) as source:
        return pd.read_excel(source)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def source_revision_records(value: str) -> list[tuple[str, int, int]]:
    """Decode object records already published in each image source revision."""
    records = json.loads(value)
    if not records or any(not isinstance(record, list) or len(record) != 3 for record in records):
        raise ValueError("Processed source revision has an unexpected object record")
    return [(str(path), int(size), int(mtime_ns)) for path, size, mtime_ns in records]


def processed_region_facts(images: pd.DataFrame, samples: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Recover exact author region labels retained in the formal input tables."""
    subjects = samples.set_index("sample_id").source_subject_id
    facts = {}
    for row in images.itertuples():
        if pd.isna(row.roi_id) or pd.isna(row.region_annotation) or pd.isna(row.sample_id):
            continue
        patient = subjects.get(row.sample_id)
        if pd.isna(patient):
            continue
        fact = (str(patient), str(row.region_annotation))
        if row.roi_id in facts and facts[row.roi_id] != fact:
            raise ValueError(f"Conflicting processed region facts: {row.roi_id}")
        facts[row.roi_id] = fact
    return facts


def merge_region_facts(facts: dict[str, tuple[str, str]], path: Path) -> dict[str, tuple[str, str]]:
    """Add only exact raw author facts not already present in processed tables."""
    if not path.exists():
        return facts
    result = dict(facts)
    for row in pd.read_csv(path, dtype=str).itertuples(index=False):
        key, fact = row.unique_region, (row.Pat_ID, row.Tissue_type_broad)
        if key in result and result[key] != fact:
            raise ValueError(f"Raw author region fact conflicts with processed tables: {key}")
        result[key] = fact
    return result


def progress(stage: str, **extra) -> None:
    path = OUTPUT / "images.parquet"
    images = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    path = OUTPUT / "sample_metadata.csv"
    samples = pd.read_csv(path) if path.exists() else pd.DataFrame()
    state = dict(study=STUDY, stage=stage, expected_images=389,
                 completed_images=len(images), exported_images=len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat())
    state.update(extra)
    save_json(PROGRESS, state)


def source_inventory() -> tuple[pd.DataFrame, list[dict], pd.DataFrame, dict[str, tuple[str, str]], dict, pd.DataFrame]:
    """Bind raw Zarr metadata and the current processed source revisions."""
    with formal_read_lease(RAW_ROOT / "metadata.json", data_root=NAS):
        with resource_lock(RAW_ROOT, shared=True), resource_lock(OUTPUT, shared=True):
            tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
            tissues = tissues[tissues.modality == "codex"].copy()
            images = pd.read_parquet(OUTPUT / "images.parquet")
            samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
            assert len(tissues) == len(images) == 389 and tissues.tissue_id.is_unique
            ordered = load_json(META / "spora_inventory/codex_ordered_marker_names.json")
            channel_names = pd.read_parquet(RAW / "codex/channels.parquet").channel_name.tolist()
            available = pd.read_parquet(RAW / "codex/channels_per_tissue.parquet")
            assert set(images.image_id) == set(tissues.tissue_id) == set(ordered)
            root = RAW / "codex/0_37744mpp/images"
            assert {p.name.removesuffix(".ome.zarr") for p in root.glob("*.ome.zarr")} == set(images.image_id)
            revisions = images.set_index("image_id").source_revision.to_dict()
            source_objects = source_bytes = 0
            for row in images.itertuples():
                records = source_revision_records(row.source_revision)
                source_objects += len(records)
                source_bytes += sum(size for _, size, _ in records)
            scope = []
            for row in tissues.sort_values("tissue_id").itertuples():
                identity = row.tissue_id
                path = root / f"{identity}.ome.zarr"
                group = load_json(path / "zarr.json")["attributes"]["ome"]
                array = load_json(path / "0/zarr.json")
                labels = [c["label"] for c in group["omero"]["channels"]]
                declared = [n for n, keep in zip(channel_names, available.loc[identity], strict=True) if keep]
                processed = images.loc[images.image_id == identity].iloc[0]
                assert labels == ordered[identity] == declared
                assert array["shape"] == [59, 1440, 1920]
                assert array["data_type"] == processed.source_dtype == "uint16"
                assert array["shape"][0] == processed.source_channel_count == 59
                assert array["shape"][2] == processed.source_x1_px and array["shape"][1] == processed.source_y1_px
                assert array["dimension_names"] == ["c", "y", "x"]
                scale = group["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
                scope.append(dict(image_id=identity, source_path=relative(path), shape=array["shape"],
                                  labels=labels, original_image_name=row.original_image_name,
                                  source_revision=revisions[identity],
                                  embedded_scale=scale, declared_mpp=0.37744))
            region_facts = merge_region_facts(processed_region_facts(images, samples), REGION_FACTS)
            panel = panel_from_s1(STUDY, PANEL_ID, scope[0]["labels"], S1)
            assert all(item["labels"] == scope[0]["labels"] for item in scope)
            assert len(panel) == int(panel.keep_channel.sum()) == 59
            panel["source_qc_pass"] = pd.read_parquet(RAW / "codex/channels.parquet").qc_pass.tolist()
            source_stats = dict(source_objects=source_objects, source_bytes=source_bytes)
    save_json(CACHE / "source_scope.json", scope)
    save_json(CACHE / "scope_reconciliation.json", dict(
        expected_images=389, complete_images=len(scope), **source_stats,
        processed_scope=relative(OUTPUT / "images.parquet"),
        estimated_uncompressed_square_output_bytes=389 * 59 * 1920 * 1920 * 2,
        metadata_modality_rows={"codex": 389, "he": 389},
        excluded_images=[],
        scale_conflict="Published directory: 0.37744 micrometers/pixel; embedded OME transform: 1.0. See scale_evidence.json.",
    ))
    return tissues, scope, panel, region_facts, source_stats, images


def prepare_metadata(tissues: pd.DataFrame, region_facts: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use explicit SPORA specimen identities; do not infer shared parent blocks.

    SPORA declares a patient-local specimen_id for every released region. Those
    source-declared specimens are retained as traceable sampling records. Their
    relation to common original tissue blocks is unresolved, and M01 says so.
    Clinical encodings without documented time units/event semantics stay raw.
    """
    author = pd.read_csv(RAW_METADATA / "HNSCCdiscovery_clinmetadata.csv")
    publication = read_supplement_table("NIHMS2186495-supplement-Table_S3.xlsx")
    source_fields = author.columns.tolist()
    assert set(tissues.unique_region).issubset(set(publication.unique_region))
    assert set(author.unique_region).issubset(set(publication.unique_region))
    pd.testing.assert_frame_equal(author.sort_values("unique_region").reset_index(drop=True),
                                  publication[publication.unique_region.isin(author.unique_region)]
                                  .sort_values("unique_region").reset_index(drop=True), check_dtype=False)
    clinical = tissues[source_fields].sort_values("unique_region").reset_index(drop=True)
    clinical = clinical.mask(clinical.isna(), np.nan)
    pd.testing.assert_frame_equal(clinical, publication[publication.unique_region.isin(tissues.unique_region)]
                                  .sort_values("unique_region").reset_index(drop=True), check_dtype=False)
    reconciliation = dict(
        spora_regions=len(tissues), publication_table_s3_regions=len(publication), author_csv_regions=len(author),
        spora_publication_exact_rows=len(tissues), author_publication_exact_rows=len(author),
        spora_author_common_regions=len(set(tissues.unique_region) & set(author.unique_region)),
        publication_regions_without_spora=sorted(set(publication.unique_region) - set(tissues.unique_region)),
        spora_regions_without_author_csv=sorted(set(tissues.unique_region) - set(author.unique_region)),
        source_patients=int(tissues.patient_id.nunique()),
        patients_with_age_or_sex=int(tissues.loc[tissues.Age.notna() | tissues.Gender_female.notna(), "patient_id"].nunique()),
        source_patient_ids_without_clinical_age_or_sex=sorted(tissues.loc[tissues.Age.isna() & tissues.Gender_female.isna(), "patient_id"].unique()),
    )
    assert (tissues.original_patient_identifier.astype(str) == tissues.CaseID.astype(str)).all()
    assert not tissues.duplicated(["patient_id", "specimen_id"]).any()
    tissues = tissues.copy()
    tissues["sample_id"] = [f"{patient}_specimen_{specimen:04d}" for patient, specimen
                             in zip(tissues.patient_id, tissues.specimen_id, strict=True)]
    annotations = {}
    clinical_keys = publication.set_index("unique_region").CaseID.astype(str)
    for region, (patient, compartment) in region_facts.items():
        if region not in clinical_keys.index or patient != clinical_keys.loc[region]:
            raise ValueError(f"Author region fact does not match Table S3: {region}")
        annotations[region] = compartment
    tissues["region_annotation"] = tissues.unique_region.map(annotations)
    annotated_count = int(tissues.region_annotation.notna().sum())
    region_sources = [relative(OUTPUT / "images.parquet"), relative(OUTPUT / "sample_metadata.csv")]
    if REGION_FACTS.exists():
        region_sources.append(relative(REGION_FACTS))
    region_source = ";".join(region_sources)
    reconciliation.update(source_declared_specimen_rows=len(tissues),
                          source_declared_specimens_with_patient=len(tissues),
                          compartment_linked_images=annotated_count,
                          compartment_unresolved_regions=tissues.loc[tissues.region_annotation.isna(), "unique_region"].tolist(),
                          parent_biological_specimen_map_available=False)
    save_json(CACHE / "clinical_source_reconciliation.json", reconciliation)
    rows = []
    for row in tissues.itertuples():
        sample = dict.fromkeys(columns("sample_metadata"))
        sample.update(study=STUDY, sample_id=row.sample_id, patient_key=row.patient_id,
                      source_subject_id=str(row.original_patient_identifier), cohort_id="HNSCC discovery",
                      center_id="Stanford Department of Pathology",
                      source_sample_id=f"{row.patient_id}:{row.specimen_id}",
                      sample_type="source-declared FFPE specimen",
                      preservation_method="ffpe",
                      primary_diagnosis="head and neck squamous cell carcinoma",
                      histology="squamous cell carcinoma",
                      metadata_notes="SPORA patient_id + specimen_id is the source-declared specimen key; "
                      "unique_region joins publication Table S3 exactly; author CSV agrees for all shared regions. "
                      "Parent tissue block correspondence across cores is not supplied. "
                      f"Author compartment labels are linked for {annotated_count} of {len(tissues)} released regions; "
                      f"{len(tissues) - annotated_count} region compartments remain unresolved. "
                      "source specimen records do not assert distinct parent biological resections. "
                      "Age, survival time, status and stage source values are preserved in source_* fields "
                      "where timing or coding is insufficient for standardized clinical endpoints.")
        compartment = annotations.get(row.unique_region)
        if annotations:
            sample["source_tissue_type_broad"] = compartment
        if compartment:
            sample["anatomical_site"] = "primary head and neck tumor" if compartment == "PT" else "lymph node"
            sample["tissue_type"] = {"PT": "tumor", "LN_met": "tumor", "LN_met_adjacent": "adjacent_normal",
                                   "LN_benign": "normal"}[compartment]
            sample["lesion_type"] = {"PT": "primary", "LN_met": "metastasis",
                                   "LN_met_adjacent": "non_malignant", "LN_benign": "non_malignant"}[compartment]
            sample["sample_type"] = f"source-declared FFPE specimen ({compartment})"
            if compartment.startswith("LN_") and row.Neck_dissection == 1:
                sample["collection_method"] = "resection"
        if pd.notna(row.Gender_female):
            sample["sex_reported"] = {0.0: "male", 1.0: "female"}[row.Gender_female]
        for field in source_fields:
            value = getattr(row, field)
            sample[f"source_{field.lower()}"] = (str(value) if field == "CaseID" else value) if pd.notna(value) else None
        sample["source_split"] = row.split
        rows.append(sample)
    samples = pd.DataFrame(rows)
    mappings = []

    def mapping(source: str, table: str, field: str, rule: str, notes: str = "") -> None:
        record = dict.fromkeys(columns("field_mapping"), "")
        record.update(study=STUDY, source_file=relative(RAW / "metadata/tissues.parquet"),
                      source_sheet_or_table="tissues[modality=codex]", source_field=source,
                      target_table=table, target_field=field,
                      join_keys="SPORA tissue_id -> unique_region -> Table S3/author CSV unique_region; CaseID -> original_patient_identifier",
                      transformation_rule=rule, conflict_rule="Raise on any joined source discrepancy", notes=notes)
        mappings.append(record)

    for source, field, rule in [
        ("patient_id;specimen_id", "sample_id", "Source-declared specimen: patient_id + '_specimen_' + zero-padded specimen_id. Shared parent blocks unresolved."),
        ("patient_id", "patient_key", "Preserve explicit source patient_id, verified against CaseID and Table S3."),
        ("original_patient_identifier", "source_subject_id", "Preserve as string; exact equality to author CaseID."),
        ("patient_id;specimen_id", "source_sample_id", "Preserve explicit composite source specimen identity, patient_id:specimen_id."),
        ("parent specimen", "biospecimen_group_id", "No parent tissue block / common specimen key supplied; leave blank."),
        ("Gender_female", "sex_reported", "0 -> male; 1 -> female; missing stays blank."),
        ("Current_smoker;Former_smoker", "smoking_status", "Leave blank: source flags have no verified index event or collection time. Preserve both raw source fields."),
    ]:
        mapping(source, "sample_metadata", field, rule)
    for field in source_fields:
        data_type = "string" if field in {"CaseID", "unique_region", "Nodal_disease", "Distant_recurrence", "OS_status"} else "number"
        mapping(field, "sample_metadata", f"source_{field.lower()}", "Preserve original value without clinical recoding.",
                json.dumps(dict(data_type=data_type, definition=f"Unmodified source clinical column {field}",
                                evidence="SPORA equals Table S3 for all 389 regions; author CSV agrees for 388 shared regions",
                                units="Unconfirmed where not explicitly documented; no unit conversion",
                                time_origin="Unconfirmed; no endpoint reinterpretation")))
    mapping("split", "sample_metadata", "source_split", "Retain source split as provenance; no new study split is assigned.",
            json.dumps(dict(data_type="string", definition="Unmodified SPORA source split label")))
    article = relative(RAW_METADATA / "article.xml")
    for field, rule in {"cohort_id": "Publication discovery cohort", "center_id": "Methods: Stanford Department of Pathology",
                        "sample_type": "Explicit SPORA specimen key; original FFPE collection",
                        "preservation_method": "Methods: formalin-fixed paraffin-embedded tissue blocks",
                        "primary_diagnosis": "Methods: HNSCC discovery cohort",
                        "histology": "Methods: head and neck squamous cell carcinoma"}.items():
        mapping("Study design for HNSCC discovery cohort", "sample_metadata", field, rule,
                "Publication reports 78 discovery patients; the released SPORA inventory contains 79 patient IDs. "
                "No release row is deleted to enforce the publication cohort count.")
        mappings[-1]["source_file"] = article
        mappings[-1]["source_sheet_or_table"] = "Methods: Study design for HNSCC discovery cohort"
    mapping("Neck_dissection;Tissue_type_broad", "sample_metadata", "collection_method",
            "For author-confirmed LN compartments with Neck_dissection=1, use resection; otherwise leave original specimen collection procedure blank.")
    mapping("tissue_id;patient_id;specimen_id", "image_metadata", "sample_id", "Join explicit source specimen key; 389/389 source regions linked to 389 declared specimens and 79 patient IDs.")
    mapping("original_image_name;unique_region", "image_metadata", "roi_id", "Preserve explicit independent source region ID. Original BIA file list confirms one TIFF per region.")
    mapping("physical core identity", "image_metadata", "core_id", "Physical core map across views/regions not supplied; leave blank rather than equating a region label with a biological core key.")
    mapping("original_image_name", "image_metadata", "slide_id", "Physical slide identity not established; leave blank. Acquisition prefixes remain in original core/region labels.")
    mapping("unique_region", "image_metadata", "region_annotation", "Region identity is retained in roi_id; join verified author Tissue_type_broad where available, otherwise leave compartment annotation blank.")
    if annotations:
        mapping("unique_region;Pat_ID;Tissue_type_broad", "image_metadata", "region_annotation",
                "Join exact author cell-table unique_region, require Pat_ID=CaseID; preserve Tissue_type_broad verbatim.",
                "Use exact facts retained in formal processed tables, plus any received missing-region source facts; no spatial boundary inferred.")
        mappings[-1]["source_file"] = region_source
        mappings[-1]["source_sheet_or_table"] = "Author HNSCCdiscovery_CODEX.csv: unique_region,Pat_ID,Tissue_type_broad"
        for field, rule in {
            "source_tissue_type_broad": "Preserve explicit author Tissue_type_broad value",
            "anatomical_site": "PT -> primary head and neck tumor; LN_* -> lymph node",
            "tissue_type": "PT/LN_met -> tumor; LN_benign -> normal; LN_met_adjacent -> adjacent_normal (microscopically non-tumor-involved LN region)",
            "lesion_type": "PT -> primary; LN_met -> metastasis; other supplied LN regions -> non_malignant",
            "sample_type": "Source-declared FFPE specimen with original author compartment label",
        }.items():
            mapping("Tissue_type_broad", "sample_metadata", field, rule,
                    json.dumps(dict(data_type="string", definition=rule)))
            mappings[-1]["source_file"] = region_source
            mappings[-1]["source_sheet_or_table"] = "Author HNSCCdiscovery_CODEX.csv: unique_region,Pat_ID,Tissue_type_broad"
    mapping("source_directory;OME coordinateTransformations", "image_metadata", "pixel_size_x_um", "Use 0.37744 um from the published resolution directory, as defined by the official SPORA format contract; embedded OME scale 1.0 conflicts and is retained in scale_evidence.json.")
    mapping("source_directory;OME coordinateTransformations", "image_metadata", "pixel_size_y_um", "Same scale rule as pixel_size_x_um.")
    mapping("channels.parquet;embedded OME channels;S1", "panel_mapping", "source_channel_index;output_channel_index",
            "All 59 channels have exact study-specific S1 matches; retain 56 mapped downstream markers, one mapping-pending channel and both selected nuclear measurements. "
            "Output order equals source order; MCT remains mapping_pending. No duplicate downstream marker.")
    mappings[-1]["source_file"] = relative(S1)
    mappings[-1]["source_sheet_or_table"] = f"S1[Study={STUDY}]"
    mappings[-1]["join_keys"] = "S1 Study + Original Marker -> embedded OME channel label; source channel position -> output channel position"
    mapping("qc_pass", "panel_mapping", "source_qc_pass", "Retain original SPORA channel QC flag as evidence; input retention follows the study-specific S1.",
            json.dumps(dict(data_type="boolean", definition="Unmodified source channel QC flag")))
    mappings[-1]["source_file"] = relative(RAW / "codex/channels.parquet")
    mappings[-1]["source_sheet_or_table"] = "codex/channels.parquet"
    mappings[-1]["join_keys"] = "channels.parquet row position -> source_channel_index; channel_name -> source_name"
    mapping("Source traceability and publication attachments", "sample_metadata", "metadata_notes",
            "Read local SPORA metadata, original BIA file list, author clinical CSV, publication XML and Table S3. "
            "No original study-specific code repository is identified (publication says no original code). "
            "Parent block/core consolidation remains unresolved; 389 declared specimen rows, 389 image links and 389 patient-linked specimen rows.",
            "Portal Dryad link points to Hickey intestine, excluded from Haist evidence. "
            "Original Haist data release is Mendeley 10.17632/t4k5kj3cxr.1 / BIA S-BIAD1638. "
            "See metadata_retrieval.json for successful and failed attachment URLs.")
    samples.to_csv(CACHE / "source_declared_samples.csv", index=False, encoding="utf-8-sig")
    return tissues, samples, pd.DataFrame(mappings, columns=columns("field_mapping"))


class StagedReader:
    """Bounded local native staging avoids decoding 59-channel Zarr chunks 59 times.

    Shared export_input streams channel-major TIFF tiles, while the native Zarr
    encodes all channels in each chunk. Stage every all-channel spatial shard
    once into a local memmap; max native block is 59x1024x1024 uint16 (118 MiB).
    This source-specific adapter can later move into the shared staging path.
    """

    def __init__(self, values: np.memmap, scale: float | None):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.source = SimpleNamespace(pixel_size_um=(scale, scale) if scale else None)

    def read_window(self, channels: list[int], window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.values[channels, y0:y1, x0:x1])

    def valid_window(self, window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1 - y0, x1 - x0), dtype=bool)


def overview(values: np.memmap, item: dict) -> None:
    """Save a native-frame display composite for representative slide QC only."""
    label = item["original_image_name"]
    slide = label.split("reg")[0]
    path = CACHE / "overviews" / f"{slide}.png"
    if path.exists():
        return
    names = item["labels"]
    components = [values[names.index(name), ::4, ::4].astype(float)
                  for name in ["Cytokeratin", "CD45", "Hoechst2"]]
    rgb = np.stack([np.clip(v / max(float(np.percentile(v, 99.5)), 1), 0, 1)
                    for v in components], axis=-1)
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    canvas = Image.new("RGB", (image.width, image.height + 30))
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((5, 5), f"{label} | CK / CD45 / Hoechst2 | display only", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_reviews(images: pd.DataFrame, samples: pd.DataFrame, scope: list[dict]) -> None:
    records = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples():
        sources = [relative(CACHE / "source_scope.json")]
        if check.check_id in {"M01", "M02"}:
            sources += [relative(RAW / "metadata/tissues.parquet"),
                        relative(RAW_METADATA / "supplementary_files.zip") + "#NIHMS2186495-supplement-Table_S3.xlsx",
                        relative(RAW_METADATA / "HNSCCdiscovery_clinmetadata.csv"),
                        relative(OUTPUT / "images.parquet"), relative(OUTPUT / "sample_metadata.csv")]
            if REGION_FACTS.exists():
                sources.append(relative(REGION_FACTS))
        elif check.check_id in {"P01", "P02", "P03"}:
            sources.append(relative(S1))
        elif check.check_id == "I04":
            sources += [relative(CACHE / "preparer_visual_qc.json"),
                        relative(OUTPUT / "images.parquet"), relative(OUTPUT / "sample_metadata.csv")]
            if REGION_FACTS.exists():
                sources.append(relative(REGION_FACTS))
        detail = dict(preparer_self_check=True, independent_review_pending=True,
                      image_count=len(images), expected_images=len(scope), sample_rows=len(samples),
                      source_patient_count=int(samples.patient_key.nunique()),
                      sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                      patient_linked_samples=int(samples.patient_key.notna().sum()))
        detail["compartment_linked_source_specimens"] = int(samples.source_tissue_type_broad.notna().sum()) if "source_tissue_type_broad" in samples else 0
        if check.check_id == "M01":
            unresolved_regions = samples.loc[samples.source_tissue_type_broad.isna(), "unique_region"].tolist()
            detail["gap"] = (f"389 explicit source-declared specimens; common parent tissue blocks are not supplied. "
                             f"Biological consolidation across cores remains unresolved. Author compartment facts are "
                             f"available for {389 - len(unresolved_regions)} regions; unresolved: {unresolved_regions}. "
                             "Publication 78 patients versus release 79 patient IDs.")
        elif check.check_id == "M02":
            detail["gap"] = "Survival time units/origins, OS +/- and HPV coding unconfirmed; raw source values retained without endpoint conversion."
        elif check.check_id in {"I03", "P03"}:
            detail["scale_evidence"] = relative(CACHE / "scale_evidence.json")
        elif check.check_id == "D01":
            detail["gap"] = "Prepared inputs require independent CLI review and controller acceptance. No preparer D01 pass is emitted."
        records.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                            record_id=STUDY if check.scope != "panel" else PANEL_ID,
                            check_id=check.check_id, result="unresolved",
                            source_evidence=json.dumps(sources),
                            output_evidence=relative(OUTPUT), details=json.dumps(detail, ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=review_path.parent):
        pd.DataFrame(records, columns=columns("review_results")).to_csv(
            review_path, index=False, encoding="utf-8-sig")


def verify_outputs(images: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Check all headers and every channel in two source-linked representative blocks."""
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    pixel_checks = []
    for row in images.itertuples():
        with formal_read_lease(NAS / row.input_image_path):
            with tifffile.TiffFile(NAS / row.input_image_path) as tif:
                assert tif.series[0].shape == (59, 1920, 1920)
                assert tif.series[0].dtype == np.dtype("uint16")
                pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
                assert [c.attrib["Name"] for c in pixels.findall("{*}Channel")] == kept.marker_id.tolist()
                if pd.notna(row.pixel_size_x_um):
                    assert float(pixels.attrib["PhysicalSizeX"]) == float(row.pixel_size_x_um)
                    assert float(pixels.attrib["PhysicalSizeY"]) == float(row.pixel_size_y_um)
    if len(images):
        row = images.iloc[0]
        specification = SourceSpec(row.image_id, PANEL_ID, "zarr", NAS / row.source_files[0] / "0", tuple(range(59)))
        source_path = NAS / row.source_files[0] / "0"
        with formal_read_lease(source_path):
            with open_source(specification) as reader, tifffile.TiffFile(NAS / row.input_image_path) as tif:
                for region in [(960, 720, 992, 752), (0, 0, 32, 32)]:
                    expected = reader.read_window(list(range(59)), region)
                    a, b, c, d = region
                    observed = np.stack([read_tiff_window(tif, i, (a, b + 240, c, d + 240)) for i in range(59)])
                    assert np.array_equal(expected, observed) and expected.dtype == observed.dtype
                    pixel_checks.append(dict(image_id=row.image_id, source_xyxy=list(region), output_y_offset=240,
                                             source_channels=list(range(59)), equal=True))
                for region in [(0, 0, 32, 32), (0, 1680, 32, 1712)]:
                    assert not any(read_tiff_window(tif, i, region).any() for i in range(59))
    evidence = dict(headers_checked=len(images), expected_shape=[59, 1920, 1920], dtype="uint16",
                    all_channel_pixel_checks=pixel_checks, padding_checks_passed=bool(len(images)),
                    independent_review=False)
    save_json(CACHE / "preparer_verification.json", evidence)
    return evidence


# Resolve the complete local source before exporting any new image.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inventory", action="store_true")
parser.add_argument("--run", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--source-overviews", action="store_true")
parser.add_argument("--repair-metadata", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
if not any((args.inventory, args.run, args.verify, args.source_overviews, args.repair_metadata)):
    parser.error("Choose --inventory, --run, --verify, --source-overviews or --repair-metadata")
CACHE.mkdir(parents=True, exist_ok=True)
if args.repair_metadata:
    progress("repair_metadata_started", scientific_repair_round=1)
    tissues, _, _, region_facts, _, images = source_inventory()
    tissues, samples, field_mapping = prepare_metadata(tissues, region_facts)
    images_path = OUTPUT / "images.parquet"
    panel_path = OUTPUT / "panel_mapping.csv"
    panel = pd.read_csv(panel_path)
    assert len(images) == 389 and set(images.sample_id) == set(samples.sample_id)
    unchanged_versions = {path: (path.stat().st_size, path.stat().st_mtime_ns)
                          for path in (images_path, panel_path)}
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)
    assert unchanged_versions == {path: (path.stat().st_size, path.stat().st_mtime_ns)
                                  for path in unchanged_versions}
    progress("repair_metadata_published", scientific_repair_round=1,
             smoking_status_nonempty=int(samples.smoking_status.notna().sum()),
             compartment_linked_samples=int(samples.source_tissue_type_broad.notna().sum()),
             independent_review_pending=True)
    raise SystemExit(0)
if args.source_overviews:
    tissues, source_scope, panel, region_facts, _, _ = source_inventory()
    examples = {}
    unresolved = set(tissues.unique_region) - set(region_facts)
    for item in source_scope:
        label = item["original_image_name"]
        examples.setdefault(label.split("reg")[0], item)
        if label in unresolved:
            examples[label] = item
    directory = CACHE / "source_preflight_overviews"
    directory.mkdir(parents=True, exist_ok=True)
    for item in examples.values():
        labels = ["Cytokeratin", "CD45", "Hoechst2"]
        specification = SourceSpec(item["image_id"], PANEL_ID, "zarr", NAS / item["source_path"] / "0",
                                   tuple(item["labels"].index(name) for name in labels))
        with formal_read_lease(NAS / item["source_path"] / "0"):
            with open_source(specification) as reader:
                values = reader.read_window([0, 1, 2], (0, 0, 1920, 1440))[:, ::4, ::4].astype(float)
        rgb = np.stack([np.clip(value / max(float(np.percentile(value, 99.5)), 1), 0, 1)
                        for value in values], axis=-1)
        view = Image.fromarray((rgb * 255).astype(np.uint8))
        canvas = Image.new("RGB", (view.width, view.height + 30))
        canvas.paste(view, (0, 30))
        ImageDraw.Draw(canvas).text((5, 5), item["original_image_name"] + " | CK / CD45 / Hoechst2", fill="white")
        canvas.save(directory / (item["original_image_name"] + ".png"))
        print(json.dumps(dict(source_overview=item["original_image_name"])), flush=True)
    raise SystemExit(0)
progress("source_inventory")
tissues, scope, panel, region_facts, _, images = source_inventory()
tissues, samples, field_mapping = prepare_metadata(tissues, region_facts)
OUTPUT.mkdir(parents=True, exist_ok=True)
if len(images):
    images["region_annotation"] = images.image_id.map(tissues.set_index("tissue_id").region_annotation)
source_revisions = {item["image_id"]: item["source_revision"] for item in scope}
assert all(row.source_revision == source_revisions[row.image_id] for row in images.itertuples())
save_input_tables(OUTPUT, images, samples, panel, field_mapping)
write_reviews(images, samples, scope)
clinical_reconciliation = load_json(CACHE / "clinical_source_reconciliation.json")
gaps = ["Parent tissue block/core consolidation unresolved; 389 declared specimen keys do not establish 389 distinct biological samplings.",
        "Author compartment labels unresolved for " + ", ".join(clinical_reconciliation["compartment_unresolved_regions"]) + ".",
        "Publication reports 78 discovery patients; released metadata contains 79 patient IDs.",
        "Survival units/origins and event coding not verified; raw source clinical fields retained.",
        "Independent CLI review pending."]
save_json(CACHE / "controller_requests.json", dict(requests=[
    "Arrange independent input review; preparer has not issued D01 acceptance.",
    "Resolve source-declared specimen versus parent biological tissue block mapping before treating repeated cores as independent samples.",
    "Reconcile the released 79 patient IDs with the publication's 78 discovery patients.",
    ("Receive exact author compartment facts for " + ", ".join(clinical_reconciliation["compartment_unresolved_regions"]) + "."
     if clinical_reconciliation["compartment_unresolved_regions"] else "All current source-region compartment facts are linked."),
    "Review recorded scale conflict: output uses 0.37744 um according to the explicit SPORA resolution-directory format contract; embedded OME says 1.0.",
    "Source portal Dryad DOI 10.5061/dryad.gmsbcc2sq belongs to Hickey human intestine; correct that external provenance in owning catalog only after review.",
    "Review the catalog tissue scope: verified author annotations include primary head/neck tumor (PT) as well as lymph node; retain all qualified local CODEX regions.",
    "Consider migrating the all-channel Zarr-to-local-memmap staging adapter into shared preparation; channel-major export otherwise repeatedly decodes all-channel chunks.",
]))
progress("metadata_prepared", gaps=gaps)

# Stage bounded all-channel spatial blocks once; stream the original uint16 values.
if args.run:
    TEMP.mkdir(parents=True, exist_ok=True)
    scale_evidence = load_json(CACHE / "scale_evidence.json")
    scale = scale_evidence.get("resolved_pixel_size_um")
    completed = set(images.image_id)
    todo = [item for item in scope if item["image_id"] not in completed]
    if args.limit is not None:
        todo = todo[:args.limit]
    for item in todo:
        identity = item["image_id"]
        progress("exporting", current_image=identity, gaps=gaps)
        if shutil.disk_usage(NAS).free < 100 * 1024**3 + 59 * 1920 * 1920 * 2:
            raise RuntimeError("NAS 100 GiB reserve would be violated by this image")
        if shutil.disk_usage(TEMP).free < 64 * 1024**3 + 59 * 1440 * 1920 * 2:
            raise RuntimeError("Local 64 GiB reserve would be violated by staging")
        start = time.monotonic()
        staging = TEMP / "native.dat"
        values = np.memmap(staging, mode="w+", dtype="uint16", shape=tuple(item["shape"]))
        specification = SourceSpec(identity, PANEL_ID, "zarr", NAS / item["source_path"] / "0", tuple(range(59)))
        with formal_read_lease(NAS / item["source_path"] / "0"):
            with open_source(specification) as reader:
                assert reader.shape == tuple(item["shape"]) and reader.dtype == np.dtype("uint16")
                for window in windows(reader.shape[-2:], 1024):
                    x0, y0, x1, y1 = window
                    values[:, y0:y1, x0:x1] = reader.read_window(list(range(59)), window)
        values.flush()
        staged_seconds = time.monotonic() - start
        overview(values, item)
        destination = OUTPUT / "images" / identity / "input.tiff"
        result = export_input(StagedReader(values, scale), panel, destination)
        values._mmap.close()
        del values
        staging.unlink()
        save_json(CACHE / "image_results" / f"{identity}.json",
                  dict(image_id=identity, source_revision=item["source_revision"], result=result,
                       staging_seconds=staged_seconds, total_seconds=time.monotonic() - start))
        if result["admitted"]:
            source_row = tissues[tissues.tissue_id == identity].iloc[0]
            row = dict.fromkeys(columns("image_metadata"))
            row.update(study=STUDY, image_id=identity, sample_id=source_row.sample_id,
                       panel_id=PANEL_ID, image_kind="roi", source_image_id=identity,
                       source_files=[item["source_path"]], source_format="ome-zarr-v3", source_axes="CYX",
                       source_dtype="uint16", source_revision=item["source_revision"],
                       roi_id=source_row.unique_region,
                       region_annotation=source_row.region_annotation,
                       source_x0_px=0, source_y0_px=0, source_x1_px=1920, source_y1_px=1440,
                       pad_left_px=0, pad_right_px=0, pad_top_px=240, pad_bottom_px=240,
                       height_px=1920, width_px=1920, source_channel_count=59, channel_count=59,
                       pixel_size_x_um=scale, pixel_size_y_um=scale,
                       input_image_path=relative(destination),
                       intensity_description="Published native uint16 CODEX intensities; no normalization or rescaling",
                       split_method="already_independent",
                       split_evidence="One independent SPORA tissue_id / original BIA raw_TIFF unique_region; preserve full 1920x1440 acquisition frame",
                       metadata_notes="Source-declared specimen linkage retained; parent tissue block consolidation unresolved. "
                       "Pixel scale evidence and the embedded SPORA conflict are recorded in cache scale_evidence.json.")
            images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, field_mapping)
            if len(images) == 1:
                verify_outputs(images, panel)
        write_reviews(images, samples, scope)
        progress("exporting", last_completed_image=identity, last_image_seconds=time.monotonic() - start, gaps=gaps)
        print(json.dumps(dict(image_id=identity, completed_images=len(images), admitted=result["admitted"],
                              staging_seconds=round(staged_seconds, 2), seconds=round(time.monotonic() - start, 2))), flush=True)
    if (TEMP / "native.dat").exists():
        (TEMP / "native.dat").unlink()
    TEMP.rmdir()

if args.verify or args.run:
    progress("verifying_outputs", gaps=gaps)
    verify_outputs(images, panel)
    write_reviews(images, samples, scope)
    progress("prepared_awaiting_independent_review" if len(images) == len(scope) else "partial_prepared",
             gaps=gaps, verification=relative(CACHE / "preparer_verification.json"))
