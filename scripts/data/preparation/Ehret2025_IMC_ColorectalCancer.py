"""Deliver Ehret's 108 SPORA IMC ROIs at their published intensity.

Inputs: raw Zarr objects and their source-object manifest, current processed
image scope, study S1 rows, and original Zenodo metadata plus bioRxiv sources.
Outputs: independent input.tiff files and the complete input metadata tables.
Run pixel work through the batch shared formal write lease wrapper. The first reduction
of the original cell metadata requires a read slot; later runs reuse its exact
108-row study-local reduction. No normalization.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import requests
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

blosc.set_nthreads(1)
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
STUDY = "Ehret2025_IMC_ColorectalCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
RAW = NAS / "raw/public" / STUDY / "spora/imc/1_0mpp/images"
RAW_ROOT = NAS / "raw/public" / STUDY
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
SOURCE_OBJECTS = RAW_METADATA / "imc_source_objects.json"
OBJECT_PREFIX = "spora-submissions/ehret2025shaking/imc/1_0mpp/images/"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
TEMP = Path("/tmp/study_input_batch") / STUDY
PROGRESS = Path(os.environ.get("CELLWEFT_STUDY_INPUT_PROGRESS", str(TEMP / "progress.json")))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PREPARER = "Codex CLI Ehret2025 0028_prepare_0"
REVISION = "SPORA ehret2025shaking; native verification complete; Zenodo 15198804 v1"


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def progress(stage: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage,
        completed_images=completed, expected_images=108,
        updated_at=datetime.now(timezone.utc).isoformat(), **details))


def source_object_records(objects: list[dict]) -> list[tuple[str, int]]:
    """Return source-relative object paths and accepted byte sizes."""
    records = []
    for item in objects:
        key = item["key"]
        if not key.startswith(OBJECT_PREFIX):
            raise ValueError(f"Unexpected source-object key: {key}")
        records.append((key.removeprefix(OBJECT_PREFIX), int(item["size"])))
    if len({key for key, _ in records}) != len(records):
        raise ValueError("Source-object manifest contains duplicate keys")
    return records


class RemoteZip(io.RawIOBase):
    """Read only ZIP directory bytes and the two necessary metadata members."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if not size:
            return b""
        end = min(self.size - 1, self.position + size - 1)
        with requests.get(self.url, headers={"Range": f"bytes={self.position}-{end}"},
                          stream=True, timeout=35) as response:
            if response.status_code != 206:
                raise ValueError(f"Metadata ZIP requires bounded HTTP 206: {response.status_code}")
            data = response.content
        if len(data) != end - self.position + 1:
            raise ValueError("Incomplete metadata ZIP range")
        self.position += len(data)
        return data


def fetch_metadata() -> None:
    """Cache original source evidence; no image archive is downloaded."""
    urls = {
        "zenodo_record.json": "https://zenodo.org/api/records/15198804",
        "github_tree.json": "https://api.github.com/repos/behret/shoco_assembloid_imc_analysis/git/trees/main?recursive=1",
        "load_data.R": "https://raw.githubusercontent.com/behret/shoco_assembloid_imc_analysis/main/load_data.R",
        "main_cell_types.Rmd": "https://raw.githubusercontent.com/behret/shoco_assembloid_imc_analysis/main/main_cell_types.Rmd",
        "biorxiv_api.json": "https://api.biorxiv.org/details/biorxiv/10.64898/2026.06.02.729459",
        "biorxiv_v1.xml": "https://www.biorxiv.org/content/early/2026/06/04/2026.06.02.729459.source.xml",
        "biorxiv_v2.xml": "https://www.biorxiv.org/content/early/2026/09/23/2026.06.02.729459.source.xml",
        "figure_s2.jpg": "https://www.biorxiv.org/content/biorxiv/early/2026/06/04/2026.06.02.729459/F7.large.jpg",
        "supplemental_table1.docx": "https://www.biorxiv.org/content/early/2026/06/04/2026.06.02.729459/DC1/embed/media-1.docx?download=true",
        "v2_table_s2.xlsx": "https://www.biorxiv.org/content/early/2026/09/23/2026.06.02.729459/DC1/embed/media-1.xlsx?download=true",
    }
    RAW_METADATA.mkdir(parents=True, exist_ok=True)
    for filename, url in urls.items():
        path = RAW_METADATA / filename
        if not path.exists():
            response = requests.get(url, timeout=35)
            response.raise_for_status()
            path.write_bytes(response.content)
    record = json.loads((RAW_METADATA / "zenodo_record.json").read_text())
    for archive in record["files"]:
        needed = [RAW_METADATA / name for name in ("rca_meta_data.csv", "rca_tma_meta_data.csv")]
        manifest = RAW_METADATA / (archive["key"] + ".members.json")
        if manifest.exists() and (archive["key"] != "processed_data.zip" or all(p.exists() for p in needed)):
            continue
        with zipfile.ZipFile(RemoteZip(archive["links"]["self"], archive["size"])) as source:
            write_json(manifest, [dict(name=i.filename, size=i.file_size,
                compressed_size=i.compress_size) for i in source.infolist()])
            if archive["key"] == "processed_data.zip":
                for path in needed:
                    if not path.exists():
                        path.write_bytes(source.read("processed/" + path.name))
    write_json(CACHE / "public_metadata_sources.json", dict(urls=urls,
        archives="Only central directories and original metadata members read by HTTP 206",
        paper_version="v1 resource table and S2B; v2 IMC methods, Figure S4B and Table S2 Donors checked",
        newer_alias_boundary="v2 tumor Donor aliases have no explicit old Pat_ID crosswalk; retain v1 clinical values keyed by original Pat_ID",
        failed_attempts={"biorxiv_v1.full": "429", "web full page": "403"}))


def original_metadata() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Recover original patient/ROI links and clinical values without hardcoded IDs."""
    reduction = CACHE / "original_roi_metadata.csv"
    if reduction.exists():
        # Zenodo 15198804 v1 inputs are immutable and cached without replacement.
        original = pd.read_csv(reduction, dtype=str)
    else:
        core = pd.read_csv(RAW_METADATA / "rca_meta_data.csv", dtype=str,
            usecols=["ROI_ID", "Experiment_ID", "Sample_ID", "Pat_ID", "Sample_type"]).drop_duplicates()
        columns = pd.read_csv(RAW_METADATA / "rca_tma_meta_data.csv", nrows=0).columns
        fields = [c for c in columns if c not in {"Unnamed: 0", "Cell_ID", "area", "eccentricity",
            "perimeter", "centroid_X", "centroid_Y", "uniqueID", "main_cell_type", "cell_type"}]
        tma = pd.read_csv(RAW_METADATA / "rca_tma_meta_data.csv", dtype=str, usecols=fields).drop_duplicates()
        original = pd.concat([core, tma], ignore_index=True)
        original.to_csv(reduction, index=False, encoding="utf-8-sig")
    if len(original) != 108 or original.Sample_ID.duplicated().any():
        raise ValueError("Original metadata is not one consistent row per source ROI")
    members = json.loads((RAW_METADATA / "raw_data.zip.members.json").read_text())
    native_names = {Path(item["name"]).name.removesuffix(".ome.tiff") for item in members
                    if item["name"].endswith(".ome.tiff")}
    if native_names != set(original.Sample_ID):
        raise ValueError("Original raw ZIP members and published metadata identities differ")
    with zipfile.ZipFile(RAW_METADATA / "supplemental_table1.docx") as source:
        root = ET.fromstring(source.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    patients = {}
    for cell in root.findall(".//w:tc", ns):
        text = "".join(cell.itertext())
        match = re.fullmatch(r"(P\d+) \((.*), Age: (\d+), Sex: (Male|Female)\)", text)
        if match:
            identity, clinical, age, sex = match.groups()
            values = clinical.split(", ")
            patients[identity] = dict(source_age_years=age, sex_reported=sex.lower(),
                primary_site=", ".join(values[:-2]), histology=values[-2], molecular_subtype=values[-1])
    if not patients:
        raise ValueError("No biological samples parsed from original supplementary resource table")
    return original, patients


def culture_pbmc_donor() -> str:
    """Read the shared IMC PBMC donor from v2's explicit sampling-strategy caption."""
    root = ET.parse(RAW_METADATA / "biorxiv_v2.xml").getroot()
    figure = next(figure for figure in root.iter("fig") if figure.attrib.get("id") == "figs4")
    text = " ".join(figure.itertext())
    match = re.search(r"Sampling strategy[^.]*allogeneic PBMC donor (IC\d+)", text)
    if match is None:
        raise ValueError("The native IMC comparison's PBMC donor cannot be located in v2 Figure S4B")
    return match.group(1)


def source_inventory() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Bind raw Zarr metadata to the formal source manifest and image scope."""
    with formal_read_lease(SOURCE_OBJECTS, data_root=NAS):
        with resource_lock(RAW_ROOT, shared=True), resource_lock(OUTPUT, shared=True):
            objects = json.loads(SOURCE_OBJECTS.read_text())
            images = pd.read_parquet(OUTPUT / "images.parquet", columns=[
                "image_id", "source_revision", "source_dtype", "source_channel_count",
                "source_x1_px", "source_y1_px",
            ])
            tissues = pd.read_parquet(META / "spora_inventory/metadata/tissues.parquet")
            ordered = json.loads((META / "spora_inventory/imc_ordered_marker_names.json").read_text())
            names = pd.read_parquet(META / "spora_inventory/imc/channels.parquet").channel_name.tolist()
            available = pd.read_parquet(META / "spora_inventory/imc/channels_per_tissue.parquet")
            object_records = source_object_records(objects)
            accepted = set(images.image_id)
            if images.image_id.duplicated().any():
                raise ValueError("Processed image scope contains duplicate image IDs")
            if not images.source_revision.eq(REVISION).all():
                raise ValueError("Processed image scope has an unexpected source version")
            manifest_ids = {key.split("/", 1)[0].removesuffix(".ome.zarr")
                            for key, _ in object_records}
            actual = {p.name.removesuffix(".ome.zarr") for p in RAW.glob("*.ome.zarr")}
            if manifest_ids != accepted or accepted != set(tissues.tissue_id) or actual != accepted:
                raise ValueError("Raw source, processed image and tissue inventories differ")
            processed = images.set_index("image_id")
            native_metadata = {}
            panels = {}
            for index, row in tissues.iterrows():
                identity = row.tissue_id
                group = json.loads((RAW / (identity + ".ome.zarr") / "zarr.json").read_text())
                array = json.loads((RAW / (identity + ".ome.zarr") / "0/zarr.json").read_text())
                scale = group["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
                saved = processed.loc[identity]
                if (array["data_type"] != saved.source_dtype or array["shape"][0] != saved.source_channel_count
                        or array["shape"][2] != saved.source_x1_px or array["shape"][1] != saved.source_y1_px
                        or array["dimension_names"] != ["c", "y", "x"] or scale != [1.0, 1.0, 1.0]):
                    raise ValueError(f"Raw Zarr metadata and current processed image differ: {identity}")
                labels = [c["label"] for c in group["attributes"]["ome"]["omero"]["channels"]]
                selected = [name for name, keep in zip(names, available.loc[identity].tolist(), strict=True) if keep]
                if labels != ordered[identity] or labels != selected:
                    raise ValueError(f"Ordered panel mismatch: {identity}")
                native_metadata[identity] = dict(shape=array["shape"], dtype=array["data_type"], embedded_labels=labels)
                panel_id = "imc_tma_44" if len(labels) == 44 else "imc_models_42"
                tissues.loc[index, "panel_id"] = panel_id
                if panel_id not in panels:
                    panels[panel_id] = panel_from_s1(STUDY, panel_id, labels, S1)
            panel = pd.concat(panels.values(), ignore_index=True)
            source_stats = dict(source_objects=len(object_records),
                                source_bytes=sum(size for _, size in object_records))
    return tissues, native_metadata, panel, source_stats


def prepare_samples(tissues: pd.DataFrame, original: pd.DataFrame,
                    patients: dict[str, dict], pbmc_donor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map sampling units with original labels, S2B and individual culture embedding.

    PT: four ROIs of one primary specimen share the source Experiment_ID.
    Tumor TMA: one sampled primary specimen per explicit source patient.
    PDA/CoC: numbered individual culture structures, positioned in Histogel cavities,
    are distinct specimens; the trailing TMA coordinate is not the sampling unit.
    Patient keys are explicit Pat_ID -> SPORA patient_id links, never name guesses.
    Cultured samples' patient keys identify their tumor source. V2 Figure S4B
    separately supplies the common allogeneic PBMC donor for this IMC comparison.
    """
    linked = tissues.merge(original, left_on="raw_sample_id", right_on="Sample_ID", validate="one_to_one")
    if not (linked.raw_patient_id == linked.Pat_ID).all() or not (linked.roi_id == linked.ROI_ID).all():
        raise ValueError("SPORA-to-original metadata identity conflict")
    linked["biological_source_id"] = [
        row.Pat_ID if row.sample_group == "TMA" else row.Experiment_ID if row.sample_group == "PT"
        else row.ROI_ID.rsplit("_", 1)[0] for row in linked.itertuples()]
    linked["sample_id"] = [f"{row.sample_group.lower()}_{row.biological_source_id}" for row in linked.itertuples()]
    samples = []
    for sample_id, group in linked.groupby("sample_id", sort=True):
        row = group.iloc[0]
        if group.patient_id.nunique() != 1:
            raise ValueError("One biological specimen refers to multiple source patients")
        sample = dict.fromkeys(pd.read_csv(TEMPLATES / "sample_metadata.csv", nrows=0).columns, "")
        sample.update(study=STUDY, sample_id=sample_id, patient_key=row.patient_id,
            source_subject_id=row.Pat_ID, source_sample_id=row.biological_source_id,
            cohort_id=row.sample_group, biospecimen_group_id=row.patient_id,
            sample_type={"TMA": "primary tumor tissue", "PT": "primary tumor tissue",
                "PDA": "SHOCO patient-derived assembloid", "CoC": "conventional PDO-PBMC coculture"}[row.sample_group],
            tissue_type="tumor" if row.sample_group in {"TMA", "PT"} else "other",
            preservation_method="ffpe", primary_diagnosis="colorectal cancer",
            source_age_years="", source_stage_group="", source_tnm_t="", source_tnm_n="",
            source_tnm_m="", source_colon_region="",
            allogeneic_pbmc_donor_id=pbmc_donor if row.sample_group in {"PDA", "CoC"} else "")
        if row.sample_group == "TMA":
            sample.update(sex_reported=row.Gender.lower(), source_age_years=row.Age,
                anatomical_site=row.Tissue_type, primary_site=row.Tissue_type,
                source_stage_group=row.Stage, source_tnm_t=row.TNM_T,
                source_tnm_n=row.TNM_N, source_tnm_m=row.TNM_M,
                source_colon_region=row.Colon_region)
        else:
            sample.update(patients[row.Pat_ID])
            sample["anatomical_site"] = sample["primary_site"] if row.sample_group == "PT" else "in vitro culture"
        sample["metadata_notes"] = (
            "原始Zenodo ROI表与SPORA tissues按完整Sample_ID精确连接；"
            "年龄/分期时点未定义，保存source_*，不填index或collection年龄。"
            + "临床按v1原始Pat_ID连接；v2改用Donor编号，缺少明确新旧对照，不按编号顺序强连。"
            + ("培养物patient_key为肿瘤/PDO来源患者；v2图S4B明确的共有异体PBMC供者见allogeneic_pbmc_donor_id。"
               if row.sample_group in {"PDA", "CoC"} else "")
        )
        samples.append(sample)
    linked.to_csv(CACHE / "image_sample_source_links.csv", index=False, encoding="utf-8-sig")
    return linked, pd.DataFrame(samples)


def field_mappings() -> pd.DataFrame:
    """Record single-field derivations, source joins and typed image extensions."""
    rows = []

    def add(source: str, target: str, rule: str, *, table: str = "sample_metadata",
            filename: str = "image_sample_source_links.csv", notes: str = "",
            source_file: Path | None = None, source_table: str = "joined SPORA and original ROI metadata",
            join_keys: str = "tissues.raw_sample_id=Sample_ID; tissues.raw_patient_id=Pat_ID; tissue_id=image_id") -> None:
        rows.append(dict(study=STUDY, source_file=relative(source_file or CACHE / filename),
            source_sheet_or_table=source_table, source_field=source,
            target_table=table, target_field=target,
            join_keys=join_keys,
            value_mapping="", transformation_rule=rule, time_origin_rule="来源未明确时点的变量保留source_*",
            conflict_rule="原始metadata与SPORA不一致则停止对应对象；不从文件名推测患者", notes=notes))

    add("Pat_ID; patient_id", "patient_key", "显式Pat_ID连接SPORA匿名patient_id；55个肿瘤来源患者；PBMC供者另列")
    add("Pat_ID", "source_subject_id", "保留来源字符串")
    add("Experiment_ID; ROI_ID; Sample_type", "sample_id", "TMA按来源患者的一份肿瘤取样；PT同一Experiment_ID四个ROI共用sample；PDA/CoC按编号独立培养物合并其位置标签。见原论文S2B及Histogel逐个样本嵌入方法。")
    add("Experiment_ID; ROI_ID; Sample_type", "source_sample_id", "PT保留Experiment_ID；TMA保留Pat_ID；培养物保留去掉位置后缀的完整结构编号")
    add("sample_group", "cohort_id", "保留TMA/PT/PDA/CoC来源组")
    add("Pat_ID; patient_id", "biospecimen_group_id", "同一患者原发组织和培养衍生物共享来源患者分组")
    add("sample_group", "sample_type", "TMA/PT=primary tumor tissue; PDA=SHOCO patient-derived assembloid; CoC=conventional PDO-PBMC coculture")
    add("sample_group", "tissue_type", "TMA/PT=tumor; PDA/CoC=other")
    add("Gender; Biological samples Sex", "sex_reported", "Male=male; Female=female；非TMA以补充资源表Pat_ID精确连接")
    add("Figure S4B sampling strategy: allogeneic PBMC donor", "allogeneic_pbmc_donor_id",
        "v2图S4B明确本IMC比较的SHOCO和rBM-Dome共用一位异体PBMC供者；填写全部44个培养物sample；原发肿瘤sample留空。patient_key继续表示肿瘤来源患者。后续严格供者隔离需要同时考虑两种供者键。",
        source_file=RAW_METADATA / "biorxiv_v2.xml", source_table="Figure S4B sampling strategy",
        join_keys="study native IMC comparator experiment; sample_group in {PDA,CoC}",
        notes=json.dumps(dict(data_type="string",
            definition="Published allogeneic PBMC donor label for the native IMC comparator experiment, separate from the tumor-origin patient_key")))
    add("Age; Biological samples Age", "source_age_years", "来源年龄（年）；时点未定义，保留原值", notes='{"data_type":"number","unit":"years","definition":"Source-reported patient age; reference time not specified"}')
    for source, target in [("Stage", "source_stage_group"), ("TNM_T", "source_tnm_t"),
                           ("TNM_N", "source_tnm_n"), ("TNM_M", "source_tnm_m"),
                           ("Colon_region", "source_colon_region")]:
        add(source, target, "保留原始TMA变量；分期系统和时点未提供，不赋予index语义",
            notes=json.dumps(dict(data_type="string", definition=f"Original TMA {source}; source reference time and staging system unspecified")))
    for target, rule in [
        ("primary_site", "TMA Tissue_type；其余使用补充资源表中显式患者肿瘤解剖部位"),
        ("anatomical_site", "原发组织使用来源部位；培养物为in vitro culture"),
        ("histology", "补充资源表Biological samples中的患者组织学原值"),
        ("molecular_subtype", "补充资源表Biological samples中的MSI/MSS原值"),
        ("preservation_method", "原论文IMC sample staining明确所有比较样本为FFPE"),
        ("primary_diagnosis", "来源研究和原始临床表确认colorectal cancer"),
    ]:
        add("Biological samples; IMC methods; Tissue_type", target, rule,
            source_file=RAW_METADATA / ("supplemental_table1.docx" if target not in {"preservation_method"} else "biorxiv_v1.xml"))
    add("metadata gaps and provenance", "metadata_notes", "记录时点缺口、培养物异体供者边界和来源追查范围")
    for target, source, rule in [
        ("source_patient_id", "Pat_ID", "复制原始显式患者键（培养物为肿瘤来源患者）"),
        ("source_sample_group", "sample_group", "保留SPORA来源类别"),
        ("source_split", "split", "保留SPORA发布划分标签，仅为来源注释"),
    ]:
        add(source, target, rule, table="image_metadata",
            source_file=(META / "spora_inventory/metadata/tissues.parquet") if target != "source_patient_id" else None,
            notes=json.dumps(dict(data_type="string", definition=rule), ensure_ascii=False))
    for target, source, rule in [
        ("study", "study", "稳定study标识Ehret2025_IMC_ColorectalCancer；保留原2025来源别名"),
        ("image_id", "tissue_id", "tissue_id与原生Zarr名称精确相同"),
        ("source_image_id", "tissue_id", "原生Zarr名称；实际数组在0"),
        ("source_files", "tissue_id", "原始OME-Zarr目录，路径相对NAS"),
        ("panel_id", "panel_id", "原始嵌入标签=ordered_map=channels筛选availability；44或42通道"),
        ("sample_id", "sample_id", "image_sample_source_links.csv中已确认取样连接"),
        ("slide_id", "Experiment_ID", "原始Experiment_ID"),
        ("core_id", "TMA_coordinate; ROI_ID", "TMA_coordinate或培养物ROI的明确位置后缀；PT留空"),
        ("roi_id", "ROI_ID", "原始ROI_ID"),
        ("region_annotation", "Sample_type; ROI_ID", "原始Sample_type、ROI_ID及已发布位置注释"),
        ("source_revision", "raw object manifest; current processed image scope", "保留已发布study版本；对象大小与原始Zarr元数据按正式来源读取"),
        ("intensity_description", "data_type; native array values", "原生发布float32，无本轮归一化；发布前补偿/处理以原论文为准"),
        ("split_method", "Sample_ID; native independent ROI", "already_independent；完整来源ROI，不进一步裁组织"),
        ("split_evidence", "Sample_ID; Experiment_ID; ROI_ID", "原始ROI表、SPORA数组框、论文S2B及逐培养物Histogel嵌入方法"),
    ]:
        source_file = (OUTPUT / "images.parquet" if target == "study" else
                       SOURCE_OBJECTS if target == "source_revision" else
                       RAW if target == "intensity_description" else None)
        add(source, target, rule, table="image_metadata", source_file=source_file)
    for target, source, rule in [
        ("image_kind", "Sample_type; TMA_coordinate", "PT=roi；TMA与独立培养结构所在孔位=tma_core"),
        ("source_format", "zarr_format; ome.version", "原生OME-Zarr v3"),
        ("source_axes", "dimension_names", "原生c,y,x对应CYX"),
        ("source_dtype", "data_type", "原生float32；TIFF保存相同dtype"),
        ("source_x0_px", "shape", "完整来源ROI左边界0"),
        ("source_y0_px", "shape", "完整来源ROI上边界0"),
        ("source_x1_px", "shape[2]", "完整来源宽度w，半开右边界"),
        ("source_y1_px", "shape[1]", "完整来源高度h，半开下边界"),
        ("pad_left_px", "shape", "L=max(h,w); left=(L-w)//2"),
        ("pad_right_px", "shape", "right=L-w-left"),
        ("pad_top_px", "shape", "top=(L-h)//2"),
        ("pad_bottom_px", "shape", "bottom=L-h-top"),
        ("height_px", "shape", "L=max(h,w)"),
        ("width_px", "shape", "L=max(h,w)"),
        ("source_channel_count", "shape[0]; embedded labels", "保留原始42/44通道位置"),
        ("channel_count", "S1 and retained panel rows", "本study全部42/44来源通道准入；DNA1/2保留"),
        ("pixel_size_x_um", "multiscales coordinateTransformations.scale[2]", "保留来源1微米像素尺度"),
        ("pixel_size_y_um", "multiscales coordinateTransformations.scale[1]", "保留来源1微米像素尺度"),
        ("input_image_path", "image_id", "processed/<study>/images/<image_id>/input.tiff；NAS相对路径"),
        ("valid_pixels_path", "native dense array", "完整稠密来源ROI无未采集位置声明；padding由四侧字段表达；留空"),
        ("source_acquisition_id", "native independent Zarr", "来源已为独立ROI，不是MCD采集；留空"),
        ("source_z_index", "dimension_names", "来源CYX无Z轴；留空"),
        ("source_t_index", "dimension_names", "来源CYX无时间轴；留空"),
        ("fov_id", "native source annotations", "无独立FOV标识；保留roi_id；留空"),
        ("metadata_notes", "source identity and sampling rules", "记录完整来源ROI、来源split及生物学取样语义"),
    ]:
        add(source, target, rule, table="image_metadata", source_file=RAW,
            source_table="<tissue_id>.ome.zarr/zarr.json and 0/zarr.json; source_scope.json; retained panel rows")
    add("public metadata and source investigation", "metadata_notes",
        "已读正式processed image scope、raw tissues/通道表与Zarr元数据，Zenodo两个metadata成员及ZIP成员目录，作者GitHub目录与R加载代码，原论文v1/v2 XML、S2B/S4B、v1补充资源表及v2 Table S2 Donors。108 image→99 sampling specimens→55 tumor-source patients；44培养物另接同一位PBMC供者。v2肿瘤Donor别名没有明确旧Pat_ID对照，临床沿v1直接来源；年龄/分期时点及随访未发布。S2B/S4B示意每患者PT/SHOCO/CoC为4/10/4 ROI；实际发布为12/29/15，按完整原始metadata及已准入108张对账。TMA来源52张与v2明确52例一致，v1写作32例。",
        filename="public_metadata_sources.json")
    for row in rows:
        if row["target_table"] != "sample_metadata":
            continue
        target = row["target_field"]
        if target in {"sex_reported", "source_age_years", "primary_site", "anatomical_site"}:
            row["source_file"] = ";".join([relative(CACHE / "original_roi_metadata.csv"),
                                               relative(RAW_METADATA / "supplemental_table1.docx")])
            row["source_sheet_or_table"] = "TMA original metadata; v1 key resources Biological samples"
            row["join_keys"] = "TMA: Sample_ID=raw_sample_id; PT/PDA/CoC: Pat_ID=explicit key-resources patient label"
        elif target in {"histology", "molecular_subtype"}:
            row["source_sheet_or_table"] = "v1 key resources Biological samples"
            row["join_keys"] = "Pat_ID=explicit key-resources patient label"
        elif target == "preservation_method":
            row["source_sheet_or_table"] = "IMC methods: sample staining"
            row["join_keys"] = "study native IMC specimen preparation"
    return pd.DataFrame(rows)


class StagedReader:
    """Read native channels from a per-image local memmap without NAS re-decoding.

    SPORA shards pack every channel together. Stage in 256x256 all-channel blocks
    once, then let the shared channel-major TIFF writer read the local copy.
    This study-local I/O adaptation is a candidate for shared-module migration.
    """

    def __init__(self, values: np.ndarray, source: SourceSpec):
        self.values, self.source = values, source
        self.shape, self.dtype = values.shape, values.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.values[channels, y0:y1, x0:x1])

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


@contextmanager
def staged_reader(source: SourceSpec):
    TEMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=source.image_id + "_", dir=TEMP) as temporary:
        with open_source(source) as native:
            array = np.lib.format.open_memmap(Path(temporary) / "native.npy", mode="w+",
                dtype=native.dtype, shape=native.shape)
            for x0, y0, x1, y1 in windows(native.shape[-2:], 256):
                array[:, y0:y1, x0:x1] = native.read_window(list(range(native.shape[0])), (x0, y0, x1, y1))
            array.flush()
        try:
            yield StagedReader(array, source)
        finally:
            array._mmap.close()


def thumbnail(reader: StagedReader, panel: pd.DataFrame, identity: str) -> None:
    """Save full-ROI DNA/epithelial/immune RGB previews for visual provenance."""
    names = panel.source_name.tolist()
    planes = []
    for name in ["panK", "CD45", "DNA1"]:
        values = np.asarray(reader.values[names.index(name), ::3, ::3])
        positive = values[values > 0]
        scale = np.percentile(positive, 99) if len(positive) else 1
        planes.append(np.clip(values / scale, 0, 1))
    image = Image.fromarray((np.stack(planes, axis=-1) * 255).astype("uint8"))
    (CACHE / "overviews").mkdir(exist_ok=True)
    image.save(CACHE / "overviews" / (identity + ".png"))


def image_record(row: pd.Series, shape: list[int], stats: dict) -> dict:
    identity, _, height, width = row.tissue_id, *shape
    record = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv", nrows=0).columns, None)
    record.update(study=STUDY, image_id=identity, sample_id=row.sample_id,
        panel_id=row.panel_id, image_kind="roi" if row.sample_group == "PT" else "tma_core",
        source_image_id=identity, source_files=[relative(RAW / (identity + ".ome.zarr"))],
        source_format="ome-zarr-v3", source_axes="CYX", source_dtype="float32",
        source_revision=REVISION, slide_id=row.Experiment_ID,
        core_id=(row.TMA_coordinate if row.sample_group == "TMA" else
                 row.ROI_ID.rsplit("_", 1)[1] if row.sample_group != "PT" else None),
        roi_id=row.ROI_ID, region_annotation=f"{row.Sample_type}; ROI_ID={row.ROI_ID}",
        source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
        pixel_size_x_um=1.0, pixel_size_y_um=1.0,
        input_image_path=relative(OUTPUT / "images" / identity / "input.tiff"),
        intensity_description="Published SPORA float32 IMC intensities; unchanged. No normalization in input preparation.",
        split_method="already_independent",
        split_evidence="Original Zenodo ROI metadata; SPORA native CYX ROI; manuscript S2B and individual Histogel embedding; full acquisition frame retained",
        metadata_notes="原始sample标签包含ROI；使用有来源的生物学取样规则；原始split仅保留为注释。",
        source_patient_id=row.Pat_ID, source_sample_group=row.sample_group, source_split=row.split)
    for name in ["height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px", "pad_bottom_px",
                 "source_dtype", "source_channel_count", "channel_count"]:
        record[name] = stats[name]
    return record


def review_results(images: pd.DataFrame, samples: pd.DataFrame, excluded: list) -> None:
    """Publish preparer checks explicitly; independent D01 remains unresolved."""
    details = dict(preparer_selfcheck=True, independent_review=False,
        exported_images=len(images), expected_images=108, excluded_images=excluded,
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.ne("").sum()),
        distinct_tumor_source_patients=int(samples.patient_key.nunique()),
        culture_pbmc_donors="V2 Figure S4B supplies one common allogeneic PBMC donor for all 44 culture specimens; patient_key identifies tumor origin")
    rows = []
    for check, scope in [("S01", "study"), ("I01", "image"), ("I02", "image"), ("I03", "image"),
                         ("I04", "image"), ("P01", "panel"), ("P02", "panel"), ("P03", "image"),
                         ("I05", "image"), ("M01", "sample"), ("M02", "sample"), ("D01", "study")]:
        evidence = [OUTPUT / "images.parquet", SOURCE_OBJECTS]
        if check.startswith("P"):
            evidence += [S1, META / "spora_inventory/imc_ordered_marker_names.json"]
        if check in {"I02", "I04", "M01", "M02", "D01"}:
            evidence += [CACHE / "image_sample_source_links.csv", CACHE / "original_roi_metadata.csv",
                         RAW_METADATA / "biorxiv_v1.xml", RAW_METADATA / "biorxiv_v2.xml", RAW_METADATA / "supplemental_table1.docx",
                         RAW_METADATA / "v2_table_s2.xlsx", RAW_METADATA / "figure_s2.jpg"]
        if check in {"I03", "I04", "I05", "P03", "D01"}:
            evidence += [CACHE / "selfcheck.json", CACHE / "visual_review.json"]
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
            record_id="all_prepared", check_id=check, result="unresolved",
            source_evidence=";".join(relative(path) for path in evidence),
            output_evidence=relative(OUTPUT / "images.parquet"),
            details=json.dumps({**details, "reason": "整理者提交；独立复核尚未执行",
                "selfcheck_evidence": relative(CACHE / "selfcheck.json")}, ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(rows).to_csv(review_path, index=False, encoding="utf-8-sig")


def contact_sheets() -> None:
    files = sorted((CACHE / "overviews").glob("*.png"))
    for first in range(0, len(files), 18):
        canvas = Image.new("RGB", (1080, 1500), "white")
        draw = ImageDraw.Draw(canvas)
        for index, file in enumerate(files[first:first + 18]):
            with Image.open(file) as image:
                image.thumbnail((350, 215))
                x, y = (index % 3) * 360, (index // 3) * 250
                canvas.paste(image, (x, y))
                draw.text((x + 2, y + 218), file.stem.removeprefix("ehret2025shaking_"), fill="black")
        canvas.save(CACHE / f"overview_{first // 18 + 1:02d}.jpg")


def selfcheck(images: pd.DataFrame, panel: pd.DataFrame, samples: pd.DataFrame,
              source_metadata: dict) -> dict:
    """Reopen all TIFF headers and compare all-channel interior/boundary blocks per layout."""
    checks, pixels = [], []
    seen = set()
    for row in images.itertuples():
        selected = panel[panel.panel_id == row.panel_id].sort_values("output_channel_index")
        _, h, w = source_metadata[row.image_id]["shape"]
        side = max(h, w)
        assert (row.source_x0_px, row.source_y0_px, row.source_x1_px, row.source_y1_px) == (0, 0, w, h)
        assert (row.pad_left_px, row.pad_right_px, row.pad_top_px, row.pad_bottom_px) == (
            (side-w)//2, side-w-(side-w)//2, (side-h)//2, side-h-(side-h)//2)
        with tifffile.TiffFile(NAS / row.input_image_path) as tif:
            ome = ET.fromstring(tif.ome_metadata)
            names = [element.attrib["Name"] for element in ome.iter()
                     if element.tag.endswith("}Channel")]
            physical = next(element for element in ome.iter() if element.tag.endswith("}Pixels"))
            assert float(physical.attrib["PhysicalSizeX"]) == row.pixel_size_x_um == 1.0
            assert float(physical.attrib["PhysicalSizeY"]) == row.pixel_size_y_um == 1.0
            assert physical.attrib["PhysicalSizeXUnit"] == physical.attrib["PhysicalSizeYUnit"] == "µm"
            assert tif.series[0].axes == "CYX"
            assert tif.series[0].shape == (row.channel_count, row.height_px, row.width_px)
            assert tif.pages[0].dtype == np.dtype(row.source_dtype)
            assert names == selected.marker_id.tolist()
            checks.append(row.image_id)
            key = (row.panel_id, row.source_sample_group, bool(row.pad_left_px or row.pad_top_px or row.pad_right_px or row.pad_bottom_px))
            if key in seen:
                continue
            seen.add(key)
            # Series discovery caches TiffFrame objects in this tifffile version.
            # Shared read_tiff_window expects scalar TiffPage attributes. Use the
            # public API to cache full pages locally; migrate this fix upstream.
            for channel in range(row.channel_count):
                tif.pages.get(channel, aspage=True, cache=True)
            source = SourceSpec(row.image_id, row.panel_id, "zarr",
                RAW / (row.image_id + ".ome.zarr") / "0", tuple(range(row.source_channel_count)))
            with open_source(source) as native:
                h, w = native.shape[-2:]
                # Cultured structures can have empty central lumens. Locate a
                # source-positive interior pixel in the unresized 3x preview.
                with Image.open(CACHE / "overviews" / (row.image_id + ".png")) as overview:
                    ys, xs = np.nonzero(np.asarray(overview).any(axis=2))
                ys, xs = ys * 3, xs * 3
                interior = (xs >= 16) & (xs <= w-16) & (ys >= 16) & (ys <= h-16)
                xs, ys = xs[interior], ys[interior]
                assert len(xs), f"No displayed interior signal: {row.image_id}"
                nearest = np.argmin((xs-w/2)**2 + (ys-h/2)**2)
                cx, cy = int(xs[nearest]), int(ys[nearest])
                for block_kind, window in [("signal_interior", (cx-16, cy-16, cx+16, cy+16)),
                                           ("boundary", (0, 0, min(w, 32), min(h, 32)))]:
                    expected = native.read_window(selected.source_channel_index.tolist(), window)
                    if block_kind == "signal_interior":
                        assert np.count_nonzero(expected), f"Interior block contains no signal: {row.image_id}"
                    x0, y0, x1, y1 = window
                    output_window = (x0 + row.pad_left_px, y0 + row.pad_top_px,
                                     x1 + row.pad_left_px, y1 + row.pad_top_px)
                    actual = np.stack([read_tiff_window(tif, channel, output_window) for channel in range(row.channel_count)])
                    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
                    pixels.append(dict(image_id=row.image_id, panel_id=row.panel_id, block_kind=block_kind,
                        sample_group=row.source_sample_group,
                        source_window=[int(value) for value in window],
                        output_window=[int(value) for value in output_window],
                        channels=int(row.channel_count), equal=True))
                for window in [(0, 0, row.width_px, row.pad_top_px),
                               (0, row.height_px-row.pad_bottom_px, row.width_px, row.height_px),
                               (0, 0, row.pad_left_px, row.height_px),
                               (row.width_px-row.pad_right_px, 0, row.width_px, row.height_px)]:
                    if window[0] < window[2] and window[1] < window[3]:
                        assert all(not read_tiff_window(tif, channel, window).any() for channel in range(row.channel_count))
    result = dict(headers_checked=len(checks), pixel_blocks=pixels,
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.ne("").sum()),
        independent_review=False, visual_review="See visual_review.json; preparer observation only")
    write_json(CACHE / "selfcheck.json", result)
    return result


# Run the local scientific workflow in source -> sample -> panel -> image order.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=["metadata", "export", "selfcheck"], default="export")
parser.add_argument("--fetch-metadata", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
if args.fetch_metadata:
    fetch_metadata()
progress("metadata", len(pd.read_parquet(OUTPUT / "images.parquet")) if (OUTPUT / "images.parquet").exists() else 0)
tissues, native_metadata, panel, source_stats = source_inventory()
original, patient_values = original_metadata()
linked, samples = prepare_samples(tissues, original, patient_values, culture_pbmc_donor())
mapping = field_mappings()
write_json(CACHE / "source_scope.json", dict(study=STUDY, expected_images=len(linked),
    source_objects=source_stats["source_objects"], source_bytes=source_stats["source_bytes"],
    source_object_manifest=relative(SOURCE_OBJECTS),
    panel_image_counts=linked.panel_id.value_counts().to_dict(),
    sample_rows=len(samples), sample_linked_images=len(linked), patient_linked_samples=len(samples),
    distinct_tumor_source_patients=samples.patient_key.nunique(),
    source_group_image_counts=linked.sample_group.value_counts().to_dict(),
    publication_scope_note="S2B/S4B schematic 4/10/4 ROIs per patient differs from released 12 PT/29 PDA/15 CoC total. Released 52 TMA ROIs agree with v2's explicit 52 tumors; v1 stated 32. Source scope follows explicit 108-image admission and original ZIP/metadata, not schematic counts.",
    culture_pbmc_donors=samples.loc[samples.allogeneic_pbmc_donor_id.ne(""), "allogeneic_pbmc_donor_id"].unique().tolist(),
    newer_tumor_alias_crosswalk="Not supplied; v1 direct Pat_ID clinical values retained, not inferred from v2 Donor numbering",
    planned_uncompressed_square_bytes=sum(v["shape"][0] * max(v["shape"][-2:]) ** 2 * 4 for v in native_metadata.values()),
    maximum_local_stage_bytes=max(np.prod(v["shape"]) * 4 for v in native_metadata.values()).item(),
    metadata_gaps=["age/stage reference time and follow-up outcomes absent", "explicit v1 Pat_ID to v2 tumor Donor alias crosswalk absent"]))
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
excluded = json.loads((CACHE / "excluded_images.json").read_text()) if (CACHE / "excluded_images.json").exists() else []

if args.stage == "metadata":
    samples.to_csv(CACHE / "sample_metadata_prepared.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(CACHE / "panel_mapping_prepared.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(CACHE / "field_mapping_prepared.csv", index=False, encoding="utf-8-sig")
    if len(images):
        save_input_tables(OUTPUT, images, samples, panel, mapping)
        review_results(images, samples, excluded)
    progress("metadata_prepared", len(images), sample_rows=len(samples),
        gaps=["Pixel selfcheck and independent review pending" if len(images) == 108
              else "Remaining TIFF export and independent review pending"])
elif args.stage == "export":
    if shutil.disk_usage(NAS).free < 100 * 1024**3 + 18 * 1024**3 or shutil.disk_usage("/tmp").free < 65 * 1024**3:
        raise RuntimeError("Insufficient reserved storage for this study's native square TIFFs")
    completed = set(images.image_id) if len(images) else set()
    pending = linked[~linked.tissue_id.isin(completed | {x["image_id"] for x in excluded})]
    if args.limit:
        pending = pending.iloc[:args.limit]
    for _, row in pending.iterrows():
        identity = row.tissue_id
        progress("exporting", len(images), current_image=identity, sample_rows=len(samples))
        selected = panel[panel.panel_id == row.panel_id]
        source = SourceSpec(identity, row.panel_id, "zarr", RAW / (identity + ".ome.zarr") / "0",
            tuple(range(native_metadata[identity]["shape"][0])), pixel_size_um=(1.0, 1.0), source_revision=REVISION)
        try:
            with staged_reader(source) as reader:
                thumbnail(reader, selected, identity)
                stats = export_input(reader, selected, OUTPUT / "images" / identity / "input.tiff", tile_size=256)
            write_json(CACHE / "image_stats" / (identity + ".json"), stats)
            if stats["admitted"]:
                record = image_record(row, native_metadata[identity]["shape"], stats)
                images = pd.concat([images, pd.DataFrame([record])], ignore_index=True)
                save_input_tables(OUTPUT, images, samples, panel, mapping)
            else:
                excluded.append(dict(image_id=identity, reason="All retained downstream markers zero across entire native ROI"))
                write_json(CACHE / "excluded_images.json", excluded)
            progress("exporting", len(images), last_completed_image=identity, excluded_images=len(excluded), sample_rows=len(samples))
            print(f"{len(images)}/108 {identity} admitted={stats['admitted']}", flush=True)
        except Exception as error:
            progress("export_error", len(images), current_image=identity, gaps=[f"{type(error).__name__}: {error}"])
            raise
    contact_sheets()
    # Republish source-derived metadata on resume without rewriting correct TIFFs.
    if len(images):
        save_input_tables(OUTPUT, images, samples, panel, mapping)
        review_results(images, samples, excluded)
    progress("prepared_for_selfcheck", len(images), sample_rows=len(samples), excluded_images=len(excluded),
        gaps=["Selfcheck and independent review pending"])
else:
    progress("selfchecking", len(images), sample_rows=len(samples),
        gaps=["TIFF header and stratified pixel selfcheck running; independent review pending"])
    save_input_tables(OUTPUT, images, samples, panel, mapping)
    try:
        result = selfcheck(images, panel, samples, native_metadata)
    except Exception as error:
        progress("selfcheck_error", len(images), sample_rows=len(samples),
            gaps=[f"{type(error).__name__}: {error}", "Independent review pending"])
        raise
    review_results(images, samples, excluded)
    progress("prepared_awaiting_independent_review", len(images), sample_rows=len(samples),
        sample_linked_images=result["sample_linked_images"], patient_linked_samples=result["patient_linked_samples"],
        gaps=["Independent reviewer/controller D01 acceptance pending", "Explicit v1/v2 tumor donor alias crosswalk absent; source-keyed v1 clinical values retained", "Clinical age/stage time origin and follow-up absent"])
    print(json.dumps(result, ensure_ascii=False), flush=True)
