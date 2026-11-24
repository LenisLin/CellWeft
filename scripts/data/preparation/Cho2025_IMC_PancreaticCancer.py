"""Deliver the 58 accepted Cho IMC ROIs at published native intensity.

Inputs: complete 1 µm Spora Zarr v3 stores, existing native verification, S1,
        cached paper XML, supplementary clinical table, and author core map.
Outputs: independent CYX input.tiff, 39-field image index, biological sample,
         panel, field mapping, preparer checks, and resumable per-image receipts.
Run export via the batch shared formal write lease wrapper and verify via read mode.
The repair-cohort stage only patches cohort provenance and records repair round 1.
The paper's 18 patient/site autopsy specimens and author core map define sample
groups; the accepted 58 ROIs cover 16 groups. Spora specimen_id is a core ordinal.

Current source follow-up (2026-10-29): 56 inputs retain their prior D01 approval.
The two published core18/core22 regions match original B6 continuation data in
35 retained channels, while their patient/site bridge remains unresolved; CD4
also differs from that original MCD range. Preserve the original tables and
the explicit review hold. This exporter has not been rerun for that hold.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Cho2025_IMC_PancreaticCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "spora/imc/1_0mpp/images"
SOURCE = metadata_source_directory(NAS, STUDY)
META = SOURCE / 'spora_inventory'
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0022_prepare_0"
PROOF = SOURCE / "native_verification.json"
MANIFEST = SOURCE / "imc_source_objects.json"
ACCEPTED = SOURCE / "accepted_images.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PREPARER = "codex_cli:0022_prepare_0"
PANEL_ID = "cho2025modeling_imc_37"
PAPER_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12262348/fullTextXML"
SUPPLEMENT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12262348/supplementaryFiles"
GAPS = ["待不同CLI上下文独立复核及D01输入验收",
        "来源未提供精确年龄、原始标本block标识、采样日期或生存终点；相关可选字段留空"]
CONTROLLER_REQUESTS = [
    "请安排不同CLI上下文独立复核58张输入，并完成与当前产物绑定的D01验收。",
    "本study catalog的paper_doi为空；已核实10.1101/2025.06.12.659314，请主控通过唯一汇总补齐。",
    "可评估将ChannelChunkCache迁入共享Zarr读取器，避免逐输出通道反复解压全通道shard；本study已用16窗口有界缓存适配，未修改src。",
]


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(stage: str, images: pd.DataFrame, **extra: object) -> None:
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(tissues), completed_images=len(images),
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        unique_patients=int(samples.patient_key.nunique()), gaps=GAPS,
        controller_requests=CONTROLLER_REQUESTS,
        updated_at=datetime.now(timezone.utc).isoformat(), **extra))


class ChannelChunkCache:
    """Bound decoded windows to 16 × 37 × 256² float32 values (148 MiB).

    Spora shards span all 37 channels, while the shared exporter writes channel
    first. Cache the exact native windows per ROI to avoid decoding each shard
    36 times. This study-local performance adapter is a candidate for migration.
    It changes neither pixels nor acquisition validity.
    """
    def __init__(self, reader):
        self.reader = reader
        self.source, self.shape, self.dtype = reader.source, reader.shape, reader.dtype
        self.cached = lru_cache(maxsize=16)(self.read_all)

    def read_all(self, window: Window) -> np.ndarray:
        return self.reader.read_window(list(range(self.shape[0])), window)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        return self.cached(tuple(window))[channels]

    def valid_window(self, window: Window) -> np.ndarray:
        return self.reader.valid_window(window)


def verify_header(path: Path, height: int, width: int) -> None:
    with tifffile.TiffFile(path) as tif:
        side = max(height, width)
        assert tif.series[0].shape == (len(kept), side, side), path
        assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("float32"), path
        xml = ET.fromstring(tif.ome_metadata)
        assert [x.attrib["Name"] for x in xml.findall(".//{*}Channel")] == kept.marker_id.tolist()
        pixels = xml.find(".//{*}Pixels")
        assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0


def make_overview(reader, image_id: str) -> None:
    """Render complete ROI nuclear, epithelial and stromal views for human QC."""
    _, height, width = reader.shape
    values = reader.read_window([17, 28, 26], (0, 0, width, height))
    canvas = Image.new("RGB", (900, 330), "black")
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(("DNA1", "CK", "SMAVIM")):
        plane = values[i]
        positive = plane[plane > 0]
        scale = np.percentile(positive, 99) if len(positive) else 1
        rendered = Image.fromarray((np.clip(plane / scale, 0, 1) * 255).astype("uint8"))
        rendered.thumbnail((300, 300))
        canvas.paste(rendered.convert("RGB"), (i * 300, 25))
        draw.text((i * 300 + 4, 6), label, fill="white")
    draw.text((4, 315), image_id, fill="white")
    (CACHE / "overviews").mkdir(exist_ok=True)
    canvas.save(CACHE / "overviews" / f"{image_id}.jpg")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("metadata", "export", "verify", "repair-cohort"), required=True)
parser.add_argument("--limit", type=int)
args = parser.parse_args()
if args.stage == "repair-cohort":
    JOB = JOB.parent / "0164_repair_1"
    GAPS = ["cohort_id映射修补后待独立定向复查M02并完成D01绑定",
            "来源未提供精确年龄、原始标本block标识、采样日期或生存终点；相关可选字段留空"]
    CONTROLLER_REQUESTS = ["请由不同CLI上下文复查新增cohort_id映射及16行来源重建，复用已通过的58张TIFF核验，完成M02和D01绑定。"]
zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# Bind current complete objects to accepted native proofs without rehashing data.
proof = json.loads(PROOF.read_text())
accepted = set(json.loads(ACCEPTED.read_text())["accepted_image_ids"])
proofs = {record["image_id"]: record for record in proof["verified"]}
tissues = pd.read_parquet(META / "metadata/tissues.parquet").sort_values("tissue_id")
assert set(tissues.tissue_id) == accepted == set(proofs)
assert accepted == {path.name.removesuffix(".ome.zarr") for path in RAW.glob("*.ome.zarr")}
source_objects = json.loads(MANIFEST.read_text())
prefix = "spora-submissions/cho2025modeling/imc/1_0mpp/images/"
for obj in source_objects:
    path = RAW / obj["key"].removeprefix(prefix)
    assert path.is_file() and path.stat().st_size == obj["size"], path
ordered = json.loads((META / "imc_ordered_marker_names.json").read_text())
source_names = pd.read_parquet(META / "imc/channels.parquet").channel_name.tolist()
channel_presence = pd.read_parquet(META / "imc/channels_per_tissue.parquet")
assert list(channel_presence.columns) == source_names and channel_presence.to_numpy().all()
for row in tissues.itertuples(index=False):
    group = json.loads((RAW / f"{row.tissue_id}.ome.zarr/zarr.json").read_text())
    ome = group["attributes"]["ome"]
    assert [x["label"] for x in ome["omero"]["channels"]] == ordered[row.tissue_id] == source_names
    assert [a["name"] for a in ome["multiscales"][0]["axes"]] == ["c", "y", "x"]
    assert ome["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(panel) == 37 and len(kept) == 36

# Connect source core numbers to the original clinical table and author map.
author = pd.read_excel(SOURCE / "source_metadata/Specimen designation.xlsx", dtype=str).rename(columns={"Name": "core_number"})
author["core_number"] = author.core_number.astype(int)
clinical = pd.read_excel(SOURCE / "source_metadata/media-2.xlsx", dtype=str)
clinical[["ID", "Sex", "Age", "Treatments"]] = clinical[["ID", "Sex", "Age", "Treatments"]].ffill()
clinical["core_number"] = clinical.core.astype(int)
clinical["Patient"] = clinical.ID.str.replace("-", "", regex=False)
joined = tissues.merge(author, on="core_number", validate="one_to_one").merge(
    clinical, on="core_number", suffixes=("_author", "_clinical"), validate="one_to_one")
assert len(joined) == len(tissues)
assert joined.raw_patient_id.eq(joined.Patient_author).all()
assert joined.raw_patient_id.eq(joined.Patient_clinical).all()
assert joined.location.eq(joined.Site_author).all() and joined.location.eq(joined.Site_clinical).all()
assert joined.sex.eq(joined.Sex).all() and joined.treatments.eq(joined.Treatments).all()
assert joined.age_decade.astype(int).astype(str).eq(joined.Age.str.split("-").str[0]).all()
assert len(author.groupby(["Patient", "Site"])) == 18
tissues["sample_id"] = "cho2025modeling_" + tissues.patient_id + "_" + tissues.location.str.lower() + "_autopsy"
samples_list = []
for sample_id, group in tissues.groupby("sample_id", sort=True):
    assert group.dataset.notna().all() and group.dataset.nunique() == 1, sample_id
    first = group.iloc[0]
    original = joined.loc[joined.tissue_id.eq(first.tissue_id)].iloc[0]
    sample = {name: "" for name in pd.read_csv(TEMPLATES / "sample_metadata.csv").columns}
    sample.update(study=STUDY, sample_id=sample_id, patient_key="cho2025modeling_" + first.patient_id,
                  source_subject_id=first.raw_patient_id, cohort_id=first.dataset,
                  center_id="Johns Hopkins Research Autopsy Program", sample_type="autopsy tumor tissue",
                  tissue_type="tumor", anatomical_site=first.location,
                  lesion_type="primary" if first.location == "Pancreas" else "metastasis",
                  collection_method="other", preservation_method="ffpe", case_control_status="case",
                  sex_reported={"M": "male", "F": "female"}[first.sex],
                  primary_diagnosis="pancreatic ductal adenocarcinoma", primary_site="Pancreas",
                  histology="ductal adenocarcinoma", age_range_reported=original.Age,
                  treatment_history_reported=original.Treatments,
                  metadata_notes="Patient/site autopsy specimen group supported by paper's 18 specimens (8 pancreas, 10 liver), clinical table and author core map; source block identifiers unavailable. Cores: "
                  + ",".join(group.core_number.astype(str)) + ". Spora specimen_id is a per-patient core ordinal. Age is a reported range; timing relative to diagnosis is unspecified. Treatment history does not define an index regimen or timepoint.")
    samples_list.append(sample)
samples = pd.DataFrame(samples_list)
assert len(samples) == 16 and samples.patient_key.nunique() == 12

# Document each join, transformation, optional absence, and provenance once.
mapping_rows = []


def mapping(file: Path, sheet: str, source: str, target_table: str, target: str,
            keys: str, transformation: str, notes: str = "", value_mapping: str = "") -> None:
    mapping_rows.append(dict(study=STUDY, source_file=relative(file), source_sheet_or_table=sheet,
                             source_field=source, target_table=target_table, target_field=target,
                             join_keys=keys, value_mapping=value_mapping, transformation_rule=transformation,
                             time_origin_rule="Not inferred unless explicitly supplied", conflict_rule="Assert agreement across source core map, clinical table and Spora", notes=notes))


for target in ("sample_id", "patient_key", "source_subject_id", "anatomical_site"):
    mapping(META / "metadata/tissues.parquet", "tissues", "tissue_id;patient_id;raw_patient_id;location;core_number", "sample_metadata", target,
            "tissue_id -> core_number -> author.Name / clinical.core; raw_patient_id == remove_hyphen(clinical.ID)",
            "Group cores by source patient_id + location at the research autopsy; stable sample_id uses pseudonymous patient and site; source_subject_id preserves raw_patient_id",
            "58/58 images -> 16 biological patient/site specimen groups -> 12 patients; original 64-core author map spans 18 groups. Local accepted scope omits six source-author cores; not additional local images.")
mapping(META / "metadata/tissues.parquet", "tissues", "dataset", "sample_metadata", "cohort_id",
        "images.source_image_id == tissues.tissue_id; images.sample_id == sample_metadata.sample_id; sample group = tissues.patient_id + tissues.location at research autopsy",
        "For each confirmed sample_id, require one nonempty tissues.dataset value across all linked cores and preserve that value verbatim as cohort_id",
        "58/58 accepted image/core records reconstruct all 16/16 sample cohort_id values; no cohort inferred from another study or from patient identity.")
mapping_rows[-1]["conflict_rule"] = "Stop if dataset is missing or differs among cores of the same confirmed sample; do not substitute a default cohort"
mapping(SOURCE / "source_metadata/Specimen designation.xlsx", "Sheet1", "Name;Patient;Site;Designation", "image_metadata", "sample_id",
        "tissues.core_number == Name", "Verify patient/site identity and Tumor designation; join all 58 source cores", "Original author core map covers 64 cores. Spora specimen_id does not identify the underlying biological specimen.")
mapping(SOURCE / "source_metadata/article.xml", "Human samples;Figure 1A", "research autopsy;FFPE;18 samples (8 pancreas,10 liver)", "sample_metadata", "sample_id",
        "Patient + Site", "Use the original study's patient/site specimen grouping, replicated across its TMA cores", "Source block IDs are absent; sample resolution is the published autopsy patient/site specimen group.")
for target, source, transformation in [
    ("sex_reported", "Sex", "M -> male; F -> female"),
    ("age_range_reported", "Age", "Preserve the reported interval verbatim; exact age fields remain empty"),
    ("treatment_history_reported", "Treatments", "Forward-fill within merged patient cells and preserve verbatim; do not infer index therapy")]:
    notes = json.dumps({"data_type": "string", "definition": "Published patient age interval; measurement timepoint unspecified" if target == "age_range_reported" else "Published treatment history; no index-regimen or timepoint inference"}) if target != "sex_reported" else "Clinical Table 1 is cached as media-2.xlsx; patient columns are merged cells."
    mapping(SOURCE / "source_metadata/media-2.xlsx", "Sheet1", source, "sample_metadata", target,
            "clinical.core == tissues.core_number; clinical.ID without hyphen == raw_patient_id", transformation, notes)
mapping(SOURCE / "source_metadata/article.xml", "Human samples", "PDAC;autopsy;formalin fixed paraffin embedded", "sample_metadata",
        "primary_diagnosis;primary_site;histology;collection_method;preservation_method;center_id;sample_type;case_control_status", "entire cohort",
        "Use the stated cohort and collection method; autopsy maps to collection_method=other with explicit sample_type", PAPER_URL)
mapping(SOURCE / "source_metadata/Specimen designation.xlsx", "Sheet1", "Designation;Site", "sample_metadata", "tissue_type;lesion_type", "Patient + Site",
        "Tumor -> tumor; Pancreas -> primary, Liver -> metastasis, supported by original paper")
mapping(META / "metadata/tissues.parquet", "tissues", "tissue_id;core_number;location;split", "image_metadata", "image_id;source_image_id;core_id;region_annotation;metadata_notes",
        "tissue_id == Zarr group name", "Preserve full ROI identity, source core number, site and published source split; source split is provenance, not a new CellWeft split assignment")
mapping(S1, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "source_channel_index;marker_id;role;keep_channel;output_channel_index",
        "Study == Cho2025_IMC_PancreaticCancer; Original marker == embedded channel label", "37 native source rows; retain 36 in source order, including DNA1/2; exclude pSTAT; no aliases")
mapping(RAW / "<tissue_id>.ome.zarr", "OME-Zarr v3 0", "axes;shape;dtype;scale;pixels", "image_metadata", "source_axes;source_dtype;pixel_size_x_um;pixel_size_y_um;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;intensity_description",
        "tissue_id -> store/0", "Use full native CYX ROI frame, unchanged float32, centered zero padding, 1 µm scale; published Spora intensities, no additional transformation",
        "Shared exporter uses a study-local bounded all-channel shard window cache; candidate for migration to shared readers. Source block/collection dates and exact ages are optional absent values.")
mapping(SOURCE / "source_metadata/geometry_review.json", "preparer source overview", "full published ROI;source core identity", "image_metadata", "split_method;split_evidence",
        "tissue_id == source image group", "Keep the complete published independent ROI; center-pad the two non-square source frames",
        "Five representative views cover both sites and both non-square shapes. Source frames are retained; tissue beyond them is not reconstructed. Independent review pending.")
field_mapping = pd.DataFrame(mapping_rows)

# Repair the reviewed mapping omission without republishing correct input tables.
if args.stage == "repair-cohort":
    with nullcontext():
        images = pd.read_parquet(OUTPUT / "images.parquet")
        saved_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        preserved_paths = [OUTPUT / name for name in ("images.parquet", "sample_metadata.csv", "panel_mapping.csv")]
        preserved_paths += [NAS / path for path in images.input_image_path]
        before = {relative(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in preserved_paths}
        progress("repairing_cohort_mapping", images)
        linked = images[["source_image_id", "sample_id"]].merge(
            tissues[["tissue_id", "sample_id", "dataset"]], left_on="source_image_id", right_on="tissue_id",
            suffixes=("_image", "_source"), validate="one_to_one")
        assert len(linked) == len(images) == len(tissues) == 58
        assert linked.sample_id_image.eq(linked.sample_id_source).all()
        assert linked.groupby("sample_id_image").dataset.nunique().eq(1).all()
        reconstructed = linked.groupby("sample_id_image").dataset.first()
        actual = saved_samples.set_index("sample_id").cohort_id
        assert actual.to_dict() == reconstructed.to_dict()
        assert len(actual) == 16
        mapping_path = OUTPUT / "field_mapping.csv"
        current = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
        repaired = field_mapping[field_mapping.target_table.eq("sample_metadata") & field_mapping.target_field.eq("cohort_id")]
        previous = current.target_table.eq("sample_metadata") & current.target_field.eq("cohort_id")
        updated = pd.concat([current.loc[~previous], repaired], ignore_index=True)
        temporary = mapping_path.with_suffix(".csv.tmp")
        with formal_publish_lease(mapping_path, directory=OUTPUT):
            updated.to_csv(temporary, index=False, encoding="utf-8-sig")
            pd.testing.assert_frame_equal(pd.read_csv(temporary, dtype=str, keep_default_na=False), updated)
            os.replace(temporary, mapping_path)
        after = {relative(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in preserved_paths}
        assert before == after
        evidence_path = JOB / "cohort_mapping_repair.json"
        write_json(evidence_path, dict(
            study=STUDY, preparer="codex_cli:0164_repair_1", independent_review=False, repair_round=1,
            source_file=relative(META / "metadata/tissues.parquet"), source_field="dataset",
            mapping=repaired.to_dict(orient="records"), source_cores_checked=len(linked),
            sample_values_checked=len(actual), all_cohort_values_match=True,
            sample_cohort_reconstruction=[dict(sample_id=key, source_core_count=int(linked.sample_id_image.eq(key).sum()),
                                              source_dataset=value, saved_cohort_id=actual.loc[key])
                                         for key, value in reconstructed.items()],
            mapping_rows_before=len(current), mapping_rows_after=len(updated),
            unrelated_mapping_rows_preserved=True, preserved_assets_size_mtime_ns=after,
            input_tables_and_58_tiffs_unchanged=True, expected_images=58, exported_images=len(images),
            sample_linked_images=int(images.sample_id.notna().sum()), sample_rows=len(saved_samples),
            patient_linked_samples=int(saved_samples.patient_key.ne("").sum()),
            unique_patients=int(saved_samples.patient_key.nunique()),
            pending="Independent targeted M02 review and D01 input binding"))
        review_path = OUTPUT / "review_results.csv"
        reviews = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        repair_rows = []
        for check, scope, record_id in [("M02", "sample", "all_delivered"), ("D01", "study", STUDY)]:
            repair_rows.append(dict(study=STUDY, preparer="codex_cli:0164_repair_1", reviewer="", scope=scope,
                                    record_id=record_id, check_id=check, result="unresolved",
                                    source_evidence=relative(META / "metadata/tissues.parquet"),
                                    output_evidence=relative(mapping_path),
                                    details=json.dumps(dict(repair_round=1, independent_review=False,
                                        action="补记tissues.dataset到cohort_id的来源、逐core连接键、组内唯一值保留及冲突规则；16/16来源值一致",
                                        preserved="58张TIFF、images/sample/panel及已有独立复核记录保持",
                                        evidence=relative(evidence_path),
                                        pending="待不同CLI上下文复查M02及完成D01绑定；本条为整理者修补记录"), ensure_ascii=False)))
        # Replace only this repair attempt's own notes when rerun.
        reviews = reviews[~(reviews.preparer.eq("codex_cli:0164_repair_1") & reviews.reviewer.eq(""))]
        temporary = review_path.with_suffix(".csv.tmp")
        with formal_publish_lease(review_path, directory=OUTPUT):
            pd.concat([reviews, pd.DataFrame(repair_rows)], ignore_index=True).to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, review_path)
    progress("repaired_pending_independent_review", images, repair_round=1,
             cohort_values_reconstructed=16, input_tiffs_reused=58, evidence=relative(evidence_path))
    print(json.dumps(dict(study=STUDY, stage=args.stage, exported_images=len(images),
                          sample_rows=len(saved_samples), sample_linked_images=len(linked),
                          patient_linked_samples=int(saved_samples.patient_key.ne("").sum()),
                          unique_patients=int(saved_samples.patient_key.nunique()),
                          cohort_values_reconstructed=16, input_tiffs_reused=58), ensure_ascii=False), flush=True)
    raise SystemExit(0)

write_json(CACHE / "metadata_sources.json", dict(
    paper_doi="10.1101/2025.06.12.659314", paper_url=PAPER_URL, supplementary_url=SUPPLEMENT_URL,
    supplementary_clinical_file=relative(SOURCE / "source_metadata/media-2.xlsx"), supplementary_antibody_file=relative(SOURCE / "source_metadata/media-3.xlsx"),
    author_repository="https://github.com/DeshpandeLab/Spatial_Influence",
    author_commit=json.loads((SOURCE / "source_metadata/author_repository_tree.json").read_text())["sha"],
    author_core_file=relative(SOURCE / "source_metadata/Specimen designation.xlsx"),
    original_raw_release="https://zenodo.org/records/15601931", original_processed_release="https://zenodo.org/records/15596960",
    source_proof=relative(PROOF), source_manifest=relative(MANIFEST),
    geometry_review=relative(SOURCE / "source_metadata/geometry_review.json"),
    local_images=len(tissues), sample_rows=len(samples), unique_patients=int(samples.patient_key.nunique()),
    author_cores_not_in_selected_spora_scope=sorted(set(author.core_number) - set(tissues.core_number)),
    sample_basis="Published 18 patient/site autopsy specimens; local accepted 58 cores represent 16 groups; block-level identifiers unavailable"))

image_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns.tolist()
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=image_columns)
save_input_tables(OUTPUT, images, samples, panel, field_mapping)
expected_bytes = sum(len(kept) * max(proofs[i]["shape"][-2:]) ** 2 * 4 for i in accepted)
write_json(CACHE / "scope.json", dict(expected_images=len(tissues), native_source_objects=len(source_objects),
                                   native_source_bytes=sum(x["size"] for x in source_objects),
                                   uncompressed_output_bytes=expected_bytes, source_proof=relative(PROOF),
                                   native_shapes={str(shape): sum(tuple(r["shape"]) == shape for r in proofs.values()) for shape in {tuple(r["shape"]) for r in proofs.values()}},
                                   exclusion_rule="Only an entire ROI with all admitted downstream measurements zero is excluded"))
progress(args.stage, images, expected_output_bytes=expected_bytes)

if args.stage == "export":
    assert shutil.disk_usage(NAS).free > 100 * 1024 ** 3 + expected_bytes
    assert shutil.disk_usage("/tmp").free > 64 * 1024 ** 3
    completed_now = 0
    for row in tissues.itertuples(index=False):
        destination = OUTPUT / "images" / row.tissue_id / "input.tiff"
        receipt_path = CACHE / "receipts" / f"{row.tissue_id}.json"
        height, width = proofs[row.tissue_id]["shape"][-2:]
        if row.tissue_id in set(images.image_id):
            verify_header(destination, height, width)
            continue
        source = SourceSpec(row.tissue_id, PANEL_ID, "zarr", RAW / f"{row.tissue_id}.ome.zarr/0",
                            tuple(range(len(source_names))), pixel_size_um=(1.0, 1.0),
                            source_revision="Spora accepted native proof: " + relative(PROOF))
        progress("exporting", images, current_image=row.tissue_id)
        if destination.exists() and receipt_path.exists():
            result = json.loads(receipt_path.read_text())
            verify_header(destination, height, width)
        else:
            with open_source(source) as reader:
                result = export_input(ChannelChunkCache(reader), panel, destination, tile_size=256)
                if (row.core_number in (4, 5, 60) or height != width) and not (CACHE / "overviews" / f"{row.tissue_id}.jpg").exists():
                    make_overview(reader, row.tissue_id)
            write_json(receipt_path, result)
        if result["admitted"]:
            verify_header(destination, height, width)
            record = {name: None for name in image_columns}
            record.update(study=STUDY, image_id=row.tissue_id, sample_id=row.sample_id, panel_id=PANEL_ID,
                          image_kind="tma_core", source_image_id=row.tissue_id,
                          source_files=[relative(RAW / f"{row.tissue_id}.ome.zarr")], source_format="ome-zarr",
                          source_axes="CYX", source_dtype="float32", source_revision=source.source_revision,
                          core_id=str(row.core_number), roi_id=row.tissue_id,
                          region_annotation="Tumor; " + row.location, source_x0_px=0, source_y0_px=0,
                          source_x1_px=width, source_y1_px=height, pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                          input_image_path=relative(destination),
                          intensity_description="Published Spora full-resolution float32 intensities; unchanged values, no additional normalization",
                          split_method="already_independent", split_evidence=relative(SOURCE / "source_metadata/geometry_review.json"),
                          metadata_notes=f"Original core {row.core_number}; Spora specimen_id={row.specimen_id} is a core ordinal. Published split={row.split}; no new split assignment. Complete source ROI frame retained.")
            for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"):
                record[key] = result[key]
            if result["valid_pixels"]:
                record["valid_pixels_path"] = relative(destination.with_name(result["valid_pixels"]))
            images = pd.concat([images, pd.DataFrame([record])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, field_mapping)
        progress("exporting", images, last_completed_image=row.tissue_id)
        completed_now += 1
        print(json.dumps(dict(image_id=row.tissue_id, admitted=result["admitted"], completed_images=len(images)), ensure_ascii=False), flush=True)
        if args.limit and completed_now >= args.limit:
            break

if args.stage == "verify":
    checks = []
    for row in images.itertuples(index=False):
        verify_header(NAS / row.input_image_path, row.source_y1_px, row.source_x1_px)
    # Full source-to-output channel comparison on internal and boundary blocks;
    # include both non-square source shapes and both biological sites.
    selected = tissues[tissues.core_number.isin([4, 5, 60]) | tissues.tissue_id.map(lambda i: proofs[i]["shape"][-2] != proofs[i]["shape"][-1])]
    by_id = images.set_index("image_id")
    for source_row in selected.itertuples(index=False):
        if source_row.tissue_id not in by_id.index:
            continue
        row = by_id.loc[source_row.tissue_id]
        height, width = proofs[source_row.tissue_id]["shape"][-2:]
        source = SourceSpec(source_row.tissue_id, PANEL_ID, "zarr", RAW / f"{source_row.tissue_id}.ome.zarr/0", tuple(range(37)))
        with open_source(source) as reader, tifffile.TiffFile(NAS / row.input_image_path) as tif:
            for window in [(width // 2, height // 2, width // 2 + 32, height // 2 + 32), (0, 0, 32, 32)]:
                native = reader.read_window(kept.source_channel_index.tolist(), window)
                x0, y0, x1, y1 = window
                output_window = (x0 + row.pad_left_px, y0 + row.pad_top_px, x1 + row.pad_left_px, y1 + row.pad_top_px)
                saved = np.stack([read_tiff_window(tif, k, output_window) for k in range(len(kept))])
                assert saved.dtype == native.dtype and np.array_equal(saved, native), source_row.tissue_id
                checks.append(dict(image_id=source_row.tissue_id, source_window=list(window), all_retained_channels_equal=True))
            if row.pad_top_px:
                for k in range(len(kept)):
                    assert not read_tiff_window(tif, k, (0, 0, int(row.width_px), int(row.pad_top_px))).any()
                    assert not read_tiff_window(tif, k, (0, int(row.pad_top_px + height), int(row.width_px), int(row.height_px))).any()
                checks.append(dict(image_id=source_row.tissue_id, all_retained_padding_zero=True))
    write_json(CACHE / "preparer_verification.json", dict(
        preparer=PREPARER, independent_review=False, image_headers_checked=len(images),
        source_object_sizes_checked=len(source_objects), sample_linked_images=int(images.sample_id.notna().sum()),
        sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
        unique_patients=int(samples.patient_key.nunique()), pixel_checks=checks,
        source_geometry_review=relative(SOURCE / "source_metadata/geometry_review.json"),
        source_sample_clinical_agreement=True, overview_files=[relative(p) for p in sorted((CACHE / "overviews").glob("*.jpg"))]))

# Preparer statements remain explicitly separate from independent acceptance.
review_path = OUTPUT / "review_results.csv"
if not review_path.exists():
    rows = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples(index=False):
        detail = f"Preparer delivery: {len(images)}/58 images, 16 sample groups, 12 patients. Independent review pending."
        if check.check_id == "M01":
            detail += " All 58 image/core->sample joins and all 16 sample->patient joins trace to clinical supplement media-2.xlsx, author Specimen designation.xlsx, and original 18-sample patient/site study design. Source block IDs are not supplied."
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                         record_id=STUDY if check.scope == "study" else "all_delivered",
                         check_id=check.check_id, result="unresolved",
                         source_evidence=relative(CACHE / "metadata_sources.json"),
                         output_evidence=relative(OUTPUT / "images.parquet"), details=detail))
    with formal_publish_lease(review_path, directory=review_path.parent):
        pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8-sig")
else:
    reviews = pd.read_csv(review_path, keep_default_na=False)
    # Update only this preparer's unreviewed rows; preserve any later reviewer work.
    own = reviews.preparer.eq(PREPARER) & reviews.reviewer.eq("")
    reviews.loc[own, "details"] = reviews.loc[own, "details"].str.replace(r"Preparer delivery: \d+/58", f"Preparer delivery: {len(images)}/58", regex=True)
    with formal_publish_lease(review_path, directory=review_path.parent):
        reviews.to_csv(review_path, index=False, encoding="utf-8-sig")
progress("prepared_pending_independent_review" if len(images) == len(tissues) else args.stage, images)
print(json.dumps(dict(study=STUDY, stage=args.stage, exported_images=len(images), sample_rows=len(samples),
                      sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=len(samples),
                      unique_patients=int(samples.patient_key.nunique())), ensure_ascii=False), flush=True)
