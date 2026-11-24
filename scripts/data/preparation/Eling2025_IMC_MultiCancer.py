"""Deliver native SPORA IMMUcan ROIs and evidence-based sample metadata.

Inputs: 179 verified level-0 OME-Zarr ROIs, source channel/tissue tables, S1,
and cached author metadata. Outputs: square uint16 CYX input.tiff, complete
study input tables, per-image export receipts and preparer checks. Run using
the batch shared formal write lease wrapper. Independent acceptance remains separate.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative, publish_metadata_sources

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Iterator
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile
import zarr

from cellweft.data.contracts import ImageBlock, SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease

STUDY = "Eling2025_IMC_MultiCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "spora"
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0030_prepare_0"
NATIVE = META / "native_verification.json"
MANIFEST = META / "imc_source_objects.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL = "IMC_panel1_40"
PREPARER = "0030_prepare_0"
INTENSITY = "Published SPORA level-0 uint16 intensities; no additional intensity transformation"


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(native), completed_images=completed,
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details))


def mapping(source: Path, field: str, table: str, target: str, rule: str,
            *, join: str = "", notes: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table="",
                source_field=field, target_table=table, target_field=target,
                join_keys=join, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="Unknown values remain empty; do not infer identity from ROI",
                notes=notes)


class CachedRoiReader:
    """Decode this study's small ROI once to avoid re-reading 40-channel shards.

    The largest native ROI is 40 x 600 x 600 uint16 (<28 MiB).
    Shared SourceReader provides the explicit level-0 decode; export windows
    then read the bounded in-memory ROI. No source pixels are transformed.
    Migration target: a shared bounded Zarr shard/ROI cache, coordinated by the
    controller; this study-local adapter does not alter shared source modules.
    """

    def __init__(self, source: SourceSpec) -> None:
        self.source = source
        with formal_read_lease(source.path):
            with open_source(source) as reader:
                self.shape, self.dtype = reader.shape, reader.dtype
                _, h, w = self.shape
                self.values = reader.read_window(list(range(self.shape[0])), (0, 0, w, h))

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1].copy()

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def iter_blocks(self, channels: list[int], block_size: int) -> Iterator[ImageBlock]:
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


def source_spec(image_id: str) -> SourceSpec:
    return SourceSpec(image_id, PANEL, "zarr", NAS / native[image_id]["path"] / "0",
                      tuple(range(len(names))), pixel_size_um=(1.0, 1.0),
                      source_revision=revisions[image_id], intensity_description=INTENSITY)


def check_header(path: Path, height: int, width: int) -> None:
    with formal_read_lease(path):
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].shape == (len(kept), height, width)
            assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("uint16")
            assert len(tif.pages) == len(kept)
            ome = ET.fromstring(tif.ome_metadata)
            channel_names = [e.attrib["Name"] for e in ome.iter() if e.tag.endswith("}Channel")]
            assert channel_names == kept.marker_id.tolist()
            pixels = next(e for e in ome.iter() if e.tag.endswith("}Pixels"))
            assert float(pixels.attrib["PhysicalSizeX"]) == 1.0
            assert float(pixels.attrib["PhysicalSizeY"]) == 1.0


def make_overviews(records: list[dict]) -> None:
    """Show nuclear and marker structure for all geometries and shared slides."""
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for start in range(0, len(records), 4):
        batch = records[start:start+4]
        fig, axes = plt.subplots(len(batch), 2, figsize=(9, 4*len(batch)), squeeze=False)
        for ax, row in zip(axes, batch):
            reader = CachedRoiReader(source_spec(row["image_id"]))
            planes = reader.values

            def scaled(name: str) -> np.ndarray:
                data = planes[names.index(name)].astype(float)
                high = np.percentile(data, 99.5)
                return np.clip(data / high, 0, 1) if high else data

            ax[0].imshow(scaled("DNA1"), cmap="gray")
            ax[1].imshow(np.stack([scaled("Ecad"), scaled("CD3"), scaled("MPO")], axis=-1))
            ax[0].set_title(row["image_id"] + "\nDNA1", fontsize=9)
            ax[1].set_title(f"Ecad / CD3 / MPO; native {reader.shape[-2:]}")
            for axis in ax:
                axis.axis("off")
            del reader
        fig.tight_layout()
        fig.savefig(CACHE / f"overview_{start//4+1}.png", dpi=120)
        plt.close(fig)
    write_json(CACHE / "overview_manifest.json", dict(
        image_ids=[row["image_id"] for row in records],
        source_revisions={row["image_id"]: revisions[row["image_id"]] for row in records},
        display="DNA1 grayscale and Ecad/CD3/MPO RGB, per-channel P99.5 display scaling only",
        visual_inspection="pending"))


def save_tables() -> None:
    frame = pd.DataFrame(rows, columns=image_columns)
    save_input_tables(OUTPUT, frame, samples, panel, field_mapping)


def save_pending_review() -> None:
    """Register self-check evidence without claiming independent review/D01 pass."""
    checks = pd.read_csv(TEMPLATES / "review_checks.csv", keep_default_na=False)
    result = []
    for check in checks.itertuples(index=False):
        details = dict(preparer_self_check=True, independent_review="pending",
                       exported_images=len(rows), expected_images=len(native))
        if check.check_id == "M01":
            details.update(sample_rows=len(samples),
                           sample_linked_images=sum(bool(image_samples.get(row["image_id"], "")) for row in rows),
                           source_image_sample_links=len(image_samples),
                           patient_linked_samples=int(samples.patient_key.ne("").sum()),
                           metadata_gaps=metadata_gaps,
                           source_trace=relative(META / "source_metadata/source_trace.json"))
        result.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                           record_id=STUDY if check.scope == "study" else "all",
                           check_id=check.check_id, result="unresolved",
                           source_evidence=relative(CACHE / "inventory.json"),
                           output_evidence=relative(CACHE / "preparer_checks.json"),
                           details=json.dumps(details, ensure_ascii=False)))
    path = OUTPUT / "review_results.csv"
    # A later preparer rerun must not erase independent review history.
    with formal_read_lease(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old.reviewer.ne("")]
        if not old.empty:
            return
    with formal_publish_lease(path, directory=path.parent):
        pd.DataFrame(result).to_csv(path, index=False, encoding="utf-8-sig")


def extract_author_tables() -> None:
    """Extract complete author colData from a bounded 32 MiB RDS gzip prefix.

    The source is a 297 MB SCE container; its complete colData precedes assays.
    Stop parsing after colData and intern repeated serialized strings to bound
    memory. The cached pure-Python rdata parser is isolated to this extraction;
    the project environment and source package are unchanged. Image-level
    metadata is required to be constant across all cells of each author ROI.
    """
    import importlib
    import sys
    import types
    import zlib

    cache = META / "source_metadata"
    package = types.ModuleType("rdata")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/dependencies/Eling2025_IMC_MultiCancer/rdata-1.0.0-py3-none-any.whl") + "/rdata"]
    sys.modules["rdata"] = package
    rparser = importlib.import_module("rdata.parser")
    implementation = importlib.import_module("rdata.parser._parser")
    kind = rparser.RObjectType

    def unref(obj):
        while obj is not None and obj.info.type == kind.REF:
            obj = obj.referenced_object
        return obj

    def strings(obj):
        obj = unref(obj)
        if obj.info.type == kind.CHAR:
            return obj.value.decode() if obj.value is not None else None
        if obj.info.type == kind.SYM:
            return strings(obj.value)
        if obj.info.type == kind.STR:
            return [strings(value) for value in obj.value]

    def attributes(obj):
        node, values = unref(obj).attributes, {}
        while node is not None and unref(node).info.type == kind.LIST:
            node = unref(node)
            values[strings(node.tag)] = node.value[0]
            node = node.value[1]
        return values

    class AuthorTableFound(Exception):
        pass

    original = implementation.Parser.parse_R_object
    interned = {}

    def capture(parser, *args, **kwargs):
        obj = original(parser, *args, **kwargs)
        if obj.info.type == kind.CHAR:
            return interned.setdefault(obj.value, obj)
        if obj.info.type == kind.S4:
            slots = attributes(obj)
            if "listData" in slots:
                columns = strings(attributes(slots["listData"])["names"])
                if "sample_id" in columns and "SampleId" in columns:
                    raise AuthorTableFound(obj)
        return obj

    prefix = (cache / "sce_raw.prefix").read_bytes() + (cache / "sce_raw.range_8_32").read_bytes()
    assert len(prefix) == 32 * 1024**2
    serialized = zlib.decompressobj(31).decompress(prefix)
    implementation.Parser.parse_R_object = capture
    try:
        try:
            rparser.parse_data(serialized, extension=".rds")
        except AuthorTableFound as found:
            data = attributes(found.args[0])["listData"]
    finally:
        implementation.Parser.parse_R_object = original
    columns = strings(attributes(data)["names"])
    image_fields = ["image", "sample_id", "width_px", "height_px", "acquisition_id",
                    "SlideId", "Study", "Box.Description", "Position", "SampleId",
                    "Indication", "BatchId", "SubBatchId", "ROI", "ROIonSlide"]
    values = {}
    for name, obj in zip(columns, data.value):
        if name not in image_fields:
            continue
        obj = unref(obj)
        if obj.info.type == kind.STR:
            values[name] = strings(obj)
        elif "levels" in attributes(obj):
            levels = strings(attributes(obj)["levels"])
            values[name] = [levels[int(i)-1] if not np.ma.is_masked(i) else None for i in obj.value]
        else:
            values[name] = obj.value
    cells = pd.DataFrame(values)
    assert cells.groupby("sample_id")[image_fields].nunique(dropna=False).le(1).all().all()
    author = cells.drop_duplicates().copy()
    assert len(author) == len(tissues) and author.sample_id.is_unique
    assert set(author.sample_id) == set(tissues.original_patient_identifier)
    author.to_csv(cache / "author_image_metadata.csv", index=False, encoding="utf-8-sig")
    write_json(cache / "author_colData_receipt.json", dict(
        source_cell_rows=len(cells), image_rows=len(author), complete_colData=True,
        image_metadata_fields=len(image_fields), all_fields_constant_within_each_roi=True,
        source_prefix_bytes=len(prefix), source="https://zenodo.org/records/12912567/files/sce_raw.rds"))
    build_author_sample_tables(author, len(cells))


def build_author_sample_tables(author: pd.DataFrame, source_cell_rows: int) -> None:
    """Connect explicit author SampleId to published SPORA patient identities."""
    cache = META / "source_metadata"

    # Five of the ten published workflow patients overlap this classifier cohort.
    # Their explicit author patient regex validates GU/HN/GI. The author README
    # lists all five cancer types; BREAS and THOR correspond to breast and NSCLC
    # (also supported by source Study codes containing BREAS/LUNG/THOR).
    clinical = pd.read_csv(cache / "workflow_sample_metadata.csv", dtype=str, keep_default_na=False)
    author["clinical_join_key"] = author.sample_id.str.extract(r"_(\d{8})")[0]
    cross = author.merge(clinical, left_on="clinical_join_key", right_on="patient_id")
    assert len(cross) == 5 and len(clinical) == 10
    assert cross.groupby("Indication").indication.nunique().eq(1).all()
    indication_map = dict(GU="RCC", HN="SCCHN", GI="CRC", BREAS="BCC", THOR="NSCLC")
    assert all(indication_map[row.Indication] == row.indication for row in cross.itertuples())
    assert set(indication_map) == set(author.Indication)
    diagnoses = dict(CRC="Colorectal cancer", SCCHN="Squamous cell carcinoma of head and neck",
                     NSCLC="Non-small cell lung cancer", BCC="Breast cancer", RCC="Renal cell carcinoma")
    sites = dict(CRC="Colorectum", SCCHN="Head and neck", NSCLC="Lung", BCC="Breast", RCC="Kidney")
    joined = tissues.reset_index().merge(author, left_on="original_patient_identifier", right_on="sample_id",
                                         validate="one_to_one", suffixes=("_spora", "_author"))
    assert joined.height.eq(pd.to_numeric(joined.height_px)).all()
    assert joined.width.eq(pd.to_numeric(joined.width_px)).all()
    assert joined.SampleId.nunique() == len(joined) and joined.patient_id.nunique() == len(joined)
    paper_text = " ".join(ET.parse(cache / "paper.xml").getroot().itertext())
    metastatic_ids = set(re.findall(r"Patient sample\s+([0-9]+)/[A-Z0-9-]+\s+was a distant metastasis", paper_text))
    metastatic_samples = set(cross.loc[cross.patient_id.isin(metastatic_ids), "SampleId"])
    sample_rows, connection_rows = [], []
    for record in joined.to_dict("records"):
        sample = {key: "" for key in sample_columns}
        diagnosis = indication_map[record["Indication"]]
        sample.update(study=STUDY, sample_id=record["SampleId"], source_sample_id=record["SampleId"],
                      patient_key=f"{STUDY}__{record['patient_id']}", source_subject_id=record["patient_id"],
                      cohort_id=record["Study"], sample_type="tissue", preservation_method="ffpe",
                      primary_diagnosis=diagnoses[diagnosis], primary_site=sites[diagnosis],
                      lesion_type="metastasis" if record["SampleId"] in metastatic_samples else "",
                      metadata_notes=json.dumps(dict(author_indication=record["Indication"],
                                                     author_SampleId=record["SampleId"],
                                                     patient_identity="Explicit published SPORA patient_id; no filename-derived patient identity",
                                                     clinical_scope="Indication, FFPE and explicit published lesion annotation only; sampled anatomical site, individual age, stage, treatment and outcomes unavailable")))
        sample_rows.append(sample)
        connection_rows.append(dict(tissue_id=record["tissue_id"], sample_id=record["SampleId"],
                                    source_image_name=record["original_patient_identifier"],
                                    source_acquisition_id=int(float(record["acquisition_id"])) if pd.notna(record["acquisition_id"]) and record["acquisition_id"] != "" else None,
                                    slide_id=record["SlideId"], roi_id=record["ROI"], author_position=record["Position"]))
    pd.DataFrame(sample_rows, columns=sample_columns).to_csv(cache / "biological_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(connection_rows).to_csv(cache / "image_sample_map.csv", index=False, encoding="utf-8-sig")
    extra = [
        mapping(RAW / "metadata/tissues.parquet", "patient_id", "sample_metadata", "patient_key;source_subject_id", "Preserve explicit SPORA patient ID; study-prefix patient_key", join="original_patient_identifier = author sample_id; author SampleId defines biological sample"),
        mapping(cache / "author_image_metadata.csv", "Study", "sample_metadata", "cohort_id", "Preserve author study/cohort code"),
        mapping(cache / "author_image_metadata.csv", "Indication", "sample_metadata", "primary_diagnosis;primary_site", "Five overlapping author workflow patients validate GU/HN/GI; source BREAS/LUNG/THOR cohort codes and author classifier README define breast/NSCLC; expand diagnosis and primary-site terms; sampled anatomical_site remains unknown", notes=json.dumps(indication_map)),
        mapping(cache / "author_image_metadata.csv", "SampleId", "sample_metadata", "sample_type", "Author SampleId identifies a tissue specimen; source study images cancer tissue sections"),
        mapping(cache / "paper.xml", "Human subjects", "sample_metadata", "preservation_method", "Published IMMUcan patient material is FFPE blocks sectioned at 4um; map to ffpe"),
        mapping(cache / "paper.xml", "Human subjects: named patient sample described as distant metastasis", "sample_metadata", "lesion_type", "Map only the explicitly reported sample to metastasis; other samples remain unknown", join="paper named patient ID = workflow clinical patient_id = author image patient regex; then explicit author SampleId", notes="No clinical cohort averages are assigned to individuals"),
        mapping(cache / "author_image_metadata.csv", "acquisition_id;SlideId;ROI", "image_metadata", "source_acquisition_id;slide_id;roi_id", "Use explicit author acquisition/slide/within-sample ROI IDs; all source values are constant within each ROI", join="original_patient_identifier = author sample_id"),
        mapping(cache / "workflow_sample_metadata.csv", "patient_id;indication", "sample_metadata", "metadata_notes", "Exact author pipeline patient regex matches five of ten workflow patients and validates their indications; individual clinical values are not extended to other subjects"),
    ]
    pd.DataFrame(extra).to_csv(cache / "author_field_mapping.csv", index=False, encoding="utf-8-sig")
    write_json(cache / "author_metadata_extraction.json", dict(
        source="https://zenodo.org/records/12912567/files/sce_raw.rds", source_bytes=297421587,
        fetched_ranges=[[0, 8388607], [8388608, 33554431]], fetched_bytes=32 * 1024**2,
        complete_colData=True, expression_assays_read=False, source_cell_rows=source_cell_rows,
        image_rows=len(author), biological_samples=len(sample_rows), patient_linked_samples=len(sample_rows),
        explicit_patient_source=relative(RAW / "metadata/tissues.parquet"),
        field_constancy="All 15 image metadata fields constant within all 179 author ROIs",
        indication_crosswalk=indication_map, clinical_validation_rows=len(cross),
        shared_slide_images=int(author.SlideId.duplicated(False).sum()),
        explicit_metastasis_samples=len(metastatic_samples),
        shared_slide_resolution="Author SampleId and Position distinguish the two biological samples; no filename-prefix merging"))
    print(json.dumps(dict(source_cell_rows=source_cell_rows, samples=len(sample_rows), connected_images=len(joined))))


# Confirm source scope and exact channel order, reusing existing native proof.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["inventory", "author-metadata", "link-metadata", "overview", "prepare", "metadata", "verify"])
args = parser.parse_args()
zarr.config.set({"async.concurrency": 4, "threading.max_workers": 4})
from numcodecs import blosc
blosc.set_nthreads(1)
CACHE.mkdir(parents=True, exist_ok=True)
proof = json.loads(NATIVE.read_text())
assert proof["full_source_covered"] and not proof["failed"]
native = {v["image_id"]: v for v in proof["verified"]}
tissues = pd.read_parquet(RAW / "metadata/tissues.parquet").set_index("tissue_id", verify_integrity=True)
channels = pd.read_parquet(RAW / "imc/channels.parquet")
availability = pd.read_parquet(META / "spora_inventory/imc/channels_per_tissue.parquet")
ordered = json.loads((META / "spora_inventory/imc_ordered_marker_names.json").read_text())
names = channels.channel_name.tolist()
assert set(tissues.index) == set(native) == set(ordered) == set(availability.index)
assert names == availability.columns.tolist() and availability.to_numpy().all()
assert all(labels == names for labels in ordered.values())
actual_ids = {p.name.removesuffix(".ome.zarr") for p in (RAW / "imc/1_0mpp/images").glob("*.ome.zarr")}
assert actual_ids == set(native)
objects = json.loads(MANIFEST.read_text())
object_sizes = {o["key"].split("schulz2024immucan/", 1)[1]: o["size"] for o in objects}
revisions, inventory = {}, []
for image_id, item in native.items():
    root = NAS / item["path"]
    header = json.loads((root / "zarr.json").read_text())["attributes"]["ome"]
    array = json.loads((root / "0/zarr.json").read_text())
    assert [x["label"] for x in header["omero"]["channels"]] == names
    assert array["shape"] == item["shape"] == [len(names), int(tissues.loc[image_id, "height"]), int(tissues.loc[image_id, "width"])]
    assert array["data_type"] == item["dtype"] == "uint16"
    assert item["exact_canonical_order_checked"] and not item["embedded_label_differences"]
    snapshot = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.endswith(".partial"):
            continue
        key = str(metadata_source_relative(p, RAW))
        assert p.stat().st_size == object_sizes[key], key
        snapshot.append(dict(path=relative(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns))
    assert len(snapshot) == item["source_object_count"] == 3
    revisions[image_id] = json.dumps(snapshot, separators=(",", ":"))
    inventory.append(dict(image_id=image_id, source_path=item["path"], shape=item["shape"],
                          dtype=item["dtype"], native_proof=relative(NATIVE), files=snapshot))
panel = panel_from_s1(STUDY, PANEL, names, S1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
estimated_tiff_bytes = sum(len(kept) * max(i["shape"][-2:]) ** 2 * 2 for i in native.values())
write_json(CACHE / "inventory.json", dict(
    source_images=len(native), source_objects=len(objects), source_bytes=proof["source_bytes"],
    expected_images=len(native), source_channels=len(names), retained_channels=len(kept),
    estimated_uncompressed_tiff_bytes=estimated_tiff_bytes, existing_native_proof=relative(NATIVE),
    source_scope="All 179 locally complete level-0 1um SPORA IMMUcan ROIs; no mirror images added",
    images=inventory))

# Biological samples use author SampleId, never image-level author sample_id.
sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns.tolist()
image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
sample_path = META / "source_metadata/biological_samples.csv"
connection_path = META / "source_metadata/image_sample_map.csv"
samples = pd.read_csv(sample_path, dtype=str, keep_default_na=False) if sample_path.exists() else pd.DataFrame(columns=sample_columns)
connections = pd.read_csv(connection_path, dtype=str, keep_default_na=False) if connection_path.exists() else pd.DataFrame(columns=["tissue_id", "sample_id"])
image_samples = connections.set_index("tissue_id").sample_id.to_dict()
source_details = connections.set_index("tissue_id").to_dict("index")
metadata_gaps = [] if len(image_samples) == len(native) else ["Author biological SampleId mapping remains incomplete; no ROI-to-sample assumption"]
if not samples.empty and samples.patient_key.eq("").any():
    metadata_gaps.append("Some biological samples lack an explicit author patient identifier")
if not samples.empty:
    metadata_gaps.append("Optional sampled anatomical site, age, sex, stage, treatment and outcomes are unavailable in checked sources; supplementary Methods PDF returned HTTP 403")
if "source_acquisition_id" in connections and connections.source_acquisition_id.eq("").any():
    metadata_gaps.append(f"Source acquisition_id is unavailable for {int(connections.source_acquisition_id.eq('').sum())} ROI; the image/sample connection is complete")
field_mapping = pd.DataFrame([
    mapping(RAW / "metadata/tissues.parquet", "tissue_id", "image_metadata", "image_id", "Preserve published tissue_id"),
    mapping(RAW / "metadata/tissues.parquet", "original_patient_identifier", "image_metadata", "source_image_id", "Preserve original author image name; SPORA tissue_id locates the source container"),
    mapping(RAW / "metadata/tissues.parquet", "height;width;imc_mpp", "image_metadata", "source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um", "Full native ROI; published 1um spacing; centered zero padding only"),
    mapping(RAW / "imc/channels.parquet", "channel_name", "panel_mapping", "source_name;source_channel_index", "Preserve source row order; cross-check embedded OME labels and per-tissue availability"),
    mapping(S1, "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index", "Exact study and source-label match; no study aliases needed"),
    mapping(connection_path, "tissue_id;sample_id", "image_metadata", "sample_id", "Join source original image name to explicit author SampleId; multiple ROIs from one biological sample share sample_id", join="tissues.original_patient_identifier = author image name"),
    mapping(META / "source_metadata/author_image_metadata.csv", "SampleId", "sample_metadata", "sample_id;source_sample_id", "Source SampleId defines biological sample; explicit SPORA patient_id is linked through original image identity", notes="See source_metadata/source_trace.json for downloaded publication, supplements, author code and actual coverage"),
    mapping(META / "source_metadata/imc_metadata_README.md", "sample_id;SampleId;ROI", "sample_metadata", "metadata_notes", "Author lower-case sample_id is image name; capitalized SampleId is sample identity; no inference from ROI filenames"),
])
extra_mapping = META / "source_metadata/author_field_mapping.csv"
if extra_mapping.exists():
    field_mapping = pd.concat([field_mapping, pd.read_csv(extra_mapping, dtype=str, keep_default_na=False)], ignore_index=True)
with formal_read_lease(OUTPUT / "images.parquet"):
    existing = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=image_columns)
rows = existing.to_dict("records")
progress("inventory_complete", len(rows), metadata_gaps=metadata_gaps,
         estimated_uncompressed_tiff_bytes=estimated_tiff_bytes)

if args.action == "inventory":
    print(json.dumps(dict(expected_images=len(native), source_channels=len(names),
                          retained_channels=len(kept), estimated_tiff_bytes=estimated_tiff_bytes)))

if args.action == "author-metadata":
    extract_author_tables()
    publish_metadata_sources(META)

if args.action == "link-metadata":
    author_table = pd.read_csv(META / "source_metadata/author_image_metadata.csv", dtype=str, keep_default_na=False)
    extraction = json.loads((META / "source_metadata/author_colData_receipt.json").read_text())
    assert extraction["complete_colData"] and extraction["image_rows"] == len(author_table)
    build_author_sample_tables(author_table, extraction["source_cell_rows"])
    publish_metadata_sources(META)

if args.action == "overview":
    source_author = pd.read_csv(META / "source_metadata/author_image_metadata.csv", dtype=str, keep_default_na=False)
    shared_slides = set(source_author.loc[source_author.SlideId.duplicated(False), "sample_id"])
    selected = {}
    for image_id in sorted(native):
        shape = tuple(native[image_id]["shape"][-2:])
        selected.setdefault(shape, dict(image_id=image_id))
        if tissues.loc[image_id, "original_patient_identifier"] in shared_slides:
            selected[image_id] = dict(image_id=image_id)
    overview_records = list({row["image_id"]: row for row in selected.values()}.values())
    make_overviews(overview_records)
    progress("native_overviews_ready", len(rows), overview_images=len(overview_records),
             sample_rows=len(samples), sample_linked_images=len(image_samples),
             patient_linked_samples=int(samples.patient_key.ne("").sum()), metadata_gaps=metadata_gaps)
    print(json.dumps(dict(overview_images=len(overview_records))))

if args.action == "metadata":
    for row in rows:
        row["sample_id"] = image_samples.get(row["image_id"], "")
        for key in ("source_acquisition_id", "slide_id", "roi_id"):
            row[key] = source_details.get(row["image_id"], {}).get(key)
    save_tables()
    save_pending_review()
    progress("metadata_updated", len(rows), sample_rows=len(samples),
             sample_linked_images=sum(bool(r["sample_id"]) for r in rows),
             patient_linked_samples=int(samples.patient_key.ne("").sum()), metadata_gaps=metadata_gaps)

if args.action == "prepare":
    assert shutil.disk_usage(NAS).free - estimated_tiff_bytes >= 100 * 1024**3
    assert shutil.disk_usage("/tmp").free >= 64 * 1024**3
    completed = {row["image_id"]: row for row in rows}
    for image_id in sorted(native):
        receipt_path = CACHE / "receipts" / f"{image_id}.json"
        destination = OUTPUT / "images" / image_id / "input.tiff"
        item = native[image_id]
        _, h, w = item["shape"]
        if image_id in completed:
            assert completed[image_id]["source_revision"] == revisions[image_id]
            check_header(destination, max(h, w), max(h, w))
            completed[image_id]["sample_id"] = image_samples.get(image_id, "")
            continue
        progress("exporting", len(completed), current_image=image_id, metadata_gaps=metadata_gaps)
        try:
            # A closed TIFF with matching source/panel receipt survives an index-write interruption.
            if destination.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["source_revision"] == revisions[image_id]
                assert receipt["channel_names"] == kept.marker_id.tolist()
                geometry = receipt["geometry"]
                check_header(destination, max(h, w), max(h, w))
            else:
                reader = CachedRoiReader(source_spec(image_id))
                geometry = export_input(reader, panel, destination, tile_size=256)
                del reader
                write_json(receipt_path, dict(image_id=image_id, source_revision=revisions[image_id],
                                             channel_names=kept.marker_id.tolist(), geometry=geometry))
            if not geometry["admitted"]:
                progress("all_marker_zero_excluded", len(completed), excluded_image=image_id,
                         metadata_gaps=metadata_gaps)
                continue
            record = {field: None for field in image_columns}
            record.update(study=STUDY, image_id=image_id, sample_id=image_samples.get(image_id, ""),
                          panel_id=PANEL, image_kind="roi",
                          source_image_id=str(tissues.loc[image_id, "original_patient_identifier"]),
                          source_files=[item["path"]], source_format="zarr", source_axes="CYX",
                          source_dtype="uint16", source_revision=revisions[image_id],
                          source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                          pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                          input_image_path=relative(destination), intensity_description=INTENSITY,
                          split_method="already_independent",
                          split_evidence="SPORA level-0 tissue_id = one published IMC ROI; full source frame preserved",
                          metadata_notes=json.dumps(dict(spora_patient_id=str(tissues.loc[image_id, "patient_id"]),
                                                         spora_specimen_id=str(tissues.loc[image_id, "specimen_id"]),
                                                         spora_split=str(tissues.loc[image_id, "split"]),
                                                         identity_policy="Source author biological metadata takes precedence over per-ROI SPORA IDs")))
            record.update({key: geometry[key] for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]})
            for key in ("source_acquisition_id", "slide_id", "roi_id"):
                record[key] = source_details.get(image_id, {}).get(key)
            completed[image_id] = record
            rows = list(completed.values())
            save_tables()
            progress("exporting", len(rows), last_completed=image_id, metadata_gaps=metadata_gaps)
            print(f"exported {len(rows)}/{len(native)} {image_id}", flush=True)
        except Exception as error:
            progress("failed", len(completed), failed_image=image_id,
                     error=f"{type(error).__name__}: {error}", metadata_gaps=metadata_gaps)
            raise
    rows = list(completed.values())
    save_tables()
    save_pending_review()
    progress("export_complete", len(rows), sample_rows=len(samples), sample_linked_images=len(image_samples),
             patient_linked_samples=int(samples.patient_key.ne("").sum()), metadata_gaps=metadata_gaps)

if args.action == "verify":
    # Header checks cover every image. Pixel comparisons cover each native geometry.
    checks, representatives = [], {}
    for row in rows:
        image_id = row["image_id"]
        check_header(NAS / row["input_image_path"], row["height_px"], row["width_px"])
        representatives.setdefault(tuple(native[image_id]["shape"][-2:]), row)
    for row in representatives.values():
        image_id = row["image_id"]
        _, h, w = native[image_id]["shape"]
        left, top = row["pad_left_px"], row["pad_top_px"]
        reader = CachedRoiReader(source_spec(image_id))
        source_positions = kept.source_channel_index.tolist()
        with tifffile.TiffFile(NAS / row["input_image_path"]) as tif:
            for x0, y0 in [(w//2-16, h//2-16), (0, 0)]:
                window = (x0, y0, x0+32, y0+32)
                values = reader.read_window(source_positions, window)
                out_window = (x0+left, y0+top, x0+left+32, y0+top+32)
                saved = np.stack([read_tiff_window(tif, i, out_window) for i in range(len(kept))])
                assert values.dtype == saved.dtype and np.array_equal(values, saved)
                checks.append(dict(image_id=image_id, source_window=window, output_window=out_window,
                                   channels=len(kept), equal=True, dtype=str(values.dtype)))
            side = max(h, w)
            padding = [(0, 0, left, side), (0, 0, side, top),
                       (left+w, 0, side, side), (0, top+h, side, side)]
            for win in padding:
                if win[2] > win[0] and win[3] > win[1]:
                    assert all(not read_tiff_window(tif, i, win).any() for i in range(len(kept)))
        del reader
    # Reuse already inspected native overviews when source bindings are unchanged.
    source_author = pd.read_csv(META / "source_metadata/author_image_metadata.csv", dtype=str, keep_default_na=False)
    shared_slides = set(source_author.loc[source_author.SlideId.duplicated(False), "sample_id"])
    overview_rows = {row["image_id"]: row for row in representatives.values()}
    overview_rows.update({row["image_id"]: row for row in rows if row["source_image_id"] in shared_slides})
    representative_rows = list(overview_rows.values())
    overview_path = CACHE / "overview_manifest.json"
    overview_binding = json.loads(overview_path.read_text()) if overview_path.exists() else {}
    requested = {row["image_id"] for row in representative_rows}
    if not requested.issubset(set(overview_binding.get("image_ids", []))):
        make_overviews(representative_rows)
    else:
        assert all(overview_binding["source_revisions"][image_id] == revisions[image_id] for image_id in requested)
    frame = pd.DataFrame(rows)
    assert len(frame) == len(native) and frame.image_id.is_unique
    assert set(frame.image_id) == set(native)
    assert set(frame.sample_id.dropna().replace("", np.nan).dropna()).issubset(set(samples.sample_id))
    write_json(CACHE / "preparer_checks.json", dict(
        independent_review=False, header_checked_images=len(rows), native_geometry_groups=len(representatives),
        pixel_checks=checks, all_padding_channels_checked=True,
        sample_rows=len(samples), sample_linked_images=int(frame.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.ne("").sum()), metadata_gaps=metadata_gaps,
        overview_files=[relative(p) for p in sorted(CACHE.glob("overview_*.png"))],
        overview_visual_inspection=json.loads((CACHE / "overview_manifest.json").read_text())["visual_inspection"]))
    save_pending_review()
    progress("self_checks_complete", len(rows), sample_rows=len(samples),
             sample_linked_images=int(frame.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.ne("").sum()), metadata_gaps=metadata_gaps,
             independent_review="pending")
    print(json.dumps(dict(headers=len(rows), pixel_blocks=len(checks), geometry_groups=len(representatives))))
