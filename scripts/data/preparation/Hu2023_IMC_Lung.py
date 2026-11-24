"""Prepare the seven admitted Hu2023 SPORA lung IMC cores.

Inputs: localized OME-Zarr, original verification, tissue/channel tables, S1,
and the paper's cached Supplementary Table 4. Outputs: native float64 CYX
input.tiff, the five study input tables, source evidence and image receipts.
Shared helpers acquire per-image formal write leases; verification uses read leases.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Hu2023_IMC_Lung"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "spora"
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
EVIDENCE = META / "source_metadata"
OUTPUT = NAS / "processed" / STUDY
LOCAL = Path("/tmp/study_input_batch") / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0209_repair_2"
PROOF = META / "native_verification.json"
ACCEPTED = META / "accepted_images.json"
FILE_EVIDENCE = META / "file_evidence.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "Hu2023_lung_imc20"
PREPARER = "codex_cli_Hu2023_IMC_Lung_0209_repair_2"
PAPER_URL = "https://www.nature.com/articles/s41467-023-43917-5"
SUPPLEMENT_URL = ("https://media.springernature.com/original/springer-static/esm/"
                  "art%3A10.1038%2Fs41467-023-43917-5/MediaObjects/41467_2023_43917_MOESM1_ESM.pdf")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["prepare", "stage", "export", "verify", "run"], default="run", nargs="?")
args = parser.parse_args()
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)
for directory in (CACHE / "image_receipts", EVIDENCE, OUTPUT, LOCAL, JOB):
    directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE) or path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def save_json(path: Path, value: dict | list) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def progress(stage: str, **details: object) -> None:
    save_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(eligible),
        exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        independently_confirmed_patient_linked_samples=0,
        updated_at=datetime.now(timezone.utc).isoformat(),
        gaps=([f"正式TIFF及image索引仅完成{len(images)}/{len(eligible)}"] if len(images) < len(eligible) else [])
             + ["独立复查及主控验收待执行；本批独立生物学患者数未知，patient_key保留SPORA声明身份"],
        **details))


class StagedReader:
    """Serve a disk-backed native core to the shared channel-major TIFF exporter.

    SPORA shards contain all channels together. Decode each 256-pixel source
    block once into a local memmap; this avoids decoding the same shard twenty
    times and never allocates a whole CYX image. Migrate this staging pattern
    to shared code only through the controller; scientific interpretation stays here.
    """

    def __init__(self, values: np.memmap, source: SourceSpec):
        self.values, self.source = values, source
        self.shape, self.dtype = values.shape, values.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.values[channels, y0:y1, x0:x1])

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def source_binding(image_id: str) -> dict:
    revision = json.dumps(file_evidence["images"][image_id]["file_stats"], separators=(",", ":"))
    return dict(source_files=source_objects[image_id], source_revision=revision,
                source_channels=source_names, output_names=retained.marker_id.tolist())


def stage_native(image_id: str) -> tuple[np.memmap, Path]:
    """Stage one verified core in bounded native blocks, reusable across I/O leases."""
    record = proofs[image_id]
    shape = tuple(record["shape"])
    path = LOCAL / f"{image_id}.npy"
    receipt = LOCAL / f"{image_id}.json"
    binding = source_binding(image_id)
    if receipt.exists() and path.exists() and json.loads(receipt.read_text()) == binding:
        values = np.load(path, mmap_mode="r")
        assert values.shape == shape and values.dtype == np.dtype("float64")
        return values, path
    assert shutil.disk_usage(LOCAL).free - int(np.prod(shape))*8 >= 64*1024**3
    progress("staging_native", image_id=image_id)
    values = np.lib.format.open_memmap(path, mode="w+", dtype="float64", shape=shape)
    try:
        with formal_read_lease(NAS / record["path"]):
            source = zarr.open_array(str(NAS / record["path"] / "0"), mode="r")
            try:
                assert source.shape == shape and source.dtype == np.dtype("float64")
                for x0, y0, x1, y1 in windows(shape[-2:], 256):
                    values[:, y0:y1, x0:x1] = source[:, y0:y1, x0:x1]
            finally:
                source.store.close()
        values.flush()
        save_json(receipt, binding)
    except BaseException:
        values._mmap.close()
        raise
    return values, path


def check_pixels(path: Path, source: object, shape: tuple[int, ...]) -> list[dict]:
    """Self-check each TIFF header and all retained channels in two native blocks."""
    _, h, w = shape
    checks = []
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == shape and tif.series[0].axes == "CYX"
        assert tif.series[0].dtype == np.dtype("float64")
        ome = ET.fromstring(tif.ome_metadata)
        assert [row.attrib["Name"] for row in ome.findall(".//{*}Channel")] == retained.marker_id.tolist()
        physical = ome.find(".//{*}Pixels").attrib
        assert float(physical["PhysicalSizeX"]) == float(physical["PhysicalSizeY"]) == 1.0
        # OME series inspection can cache TiffFrame objects; the shared reader uses full pages.
        tif.pages.useframes = False
        for channel in range(len(retained)):
            tif.pages.get(channel, aspage=True, cache=True)
        for box in [(w//2, h//2, w//2+16, h//2+16), (w-16, h-16, w, h)]:
            x0, y0, x1, y1 = box
            native = np.asarray(source[:, y0:y1, x0:x1])
            saved = np.stack([read_tiff_window(tif, c, box) for c in range(len(retained))])
            assert saved.dtype == native.dtype and np.array_equal(saved, native)
            checks.append(dict(window_xyxy=list(box), source_channels=list(range(len(retained))),
                               exact_equal=True, nonzero_pixels=int(np.count_nonzero(native))))
    return checks


def publish() -> None:
    coverage = dict(source_linkable_images=len(eligible), exported_images=len(images),
                    sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                    patient_linked_samples=int(samples.patient_key.notna().sum()),
                    independently_confirmed_patient_linked_samples=0)
    identity_fields = (fields.target_table == "sample_metadata") & fields.target_field.isin(
        ["sample_id", "patient_key", "source_subject_id", "source_sample_id"])
    fields.loc[identity_fields, "notes"] = (
        f"来源可连接{len(eligible)}/{len(eligible)}；实际交付image→sample={coverage['sample_linked_images']}/{len(eligible)}；"
        f"sample→SPORA声明patient={coverage['patient_linked_samples']}/{len(samples)}；独立病例关系未核实")
    save_input_tables(OUTPUT, images, samples, panel, fields)
    source_trace["coverage"] = coverage
    save_json(CACHE / "source_trace.json", source_trace)


def overview(image_id: str, values: np.memmap) -> None:
    """Save display-only nuclear/marker overview; input intensities are untouched."""
    nuclear = values[source_names.index("Histone 3"), ::6, ::6]
    epithelial = values[source_names.index("PanKeratin"), ::6, ::6]
    immune = values[source_names.index("CD45RO"), ::6, ::6]
    layers = []
    for values2d in (epithelial, immune, nuclear):
        positive = values2d[values2d > 0]
        scale = float(np.percentile(positive, 99)) if len(positive) else 1.0
        layers.append((np.clip(values2d / scale, 0, 1) * 255).astype("uint8"))
    composite = Image.fromarray(np.stack(layers, axis=-1)).resize((320, 320))
    canvas = Image.new("RGB", (320, 350), "white")
    canvas.paste(composite, (0, 30))
    label = str(tissues.loc[image_id, "original_file_name"])
    ImageDraw.Draw(canvas).text((8, 8), label + " | PanCK/CD45RO/H3", fill="black")
    destination = CACHE / "overviews" / f"{image_id}.png"
    destination.parent.mkdir(exist_ok=True)
    canvas.save(destination)


# Reuse native verification and its file-stat binding; no new source hashes.
eligible = json.loads(ACCEPTED.read_text())["accepted_image_ids"]
proofs = {item["image_id"]: item for item in json.loads(PROOF.read_text())["verified"]}
file_evidence = json.loads(FILE_EVIDENCE.read_text())
assert set(eligible) == set(proofs)
manifest = json.loads((META / "imc_source_objects.json").read_text())
for key, stat in file_evidence["context"]:
    current = (NAS / key).stat()
    assert [current.st_size, current.st_mtime_ns] == stat, key
source_objects = {}
for image_id in eligible:
    records = [row for row in manifest if f"/{image_id}.ome.zarr/" in row["key"]]
    paths = [RAW / row["key"].split("spora-submissions/hu2023single/", 1)[1] for row in records]
    assert [[p.stat().st_size, p.stat().st_mtime_ns] for p in paths] == file_evidence["images"][image_id]["file_stats"]
    assert all(p.stat().st_size == row["size"] for p, row in zip(paths, records))
    assert len(records) == proofs[image_id]["source_object_count"]
    source_objects[image_id] = [relative(p) for p in paths]
actual = {p.name.removesuffix(".ome.zarr") for p in (RAW / "imc/1_0mpp/images").glob("*.ome.zarr")}
assert actual == set(eligible)

# Source table order, the embedded OME labels and S1 must agree for every core.
tissues = pd.read_parquet(RAW / "metadata/tissues.parquet").set_index("tissue_id", verify_integrity=True)
channels = pd.read_parquet(RAW / "imc/channels.parquet")
availability = pd.read_parquet(RAW / "imc/channels_per_tissue.parquet")
ordered = json.loads((metadata_source_directory(NAS, STUDY) / 'spora_inventory' / 'imc_ordered_marker_names.json').read_text())
source_names = ordered[eligible[0]]
assert len(source_names) == 20
for image_id in eligible:
    assert ordered[image_id] == source_names
    assert channels.loc[availability.loc[image_id].to_numpy(dtype=bool), "channel_name"].tolist() == source_names
    group = json.loads((NAS / proofs[image_id]["path"] / "zarr.json").read_text())
    assert [row["label"] for row in group["attributes"]["ome"]["omero"]["channels"]] == source_names
    assert proofs[image_id]["canonical_channel_labels"] == source_names
    assert proofs[image_id]["exact_canonical_order_checked"]
    assert [axis["name"] for axis in group["attributes"]["ome"]["multiscales"][0]["axes"]] == ["c", "y", "x"]
    scale = group["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [1.0, 1.0, 1.0]
panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1)
panel["display_name"] = panel.marker_id
retained = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(retained) == 20 and panel.nuclear.sum() == 1

# Original release notes identify 27 instrument pages. SPORA publishes the 20
# protein channels in its own order; original DNA/background pages are not local
# SPORA channels. Keep these two positions distinct in the source evidence.
original_aliases = {"ECadherin": "E-cadherin", "CD8": "CD8a", "GranzymeB": "Granzyme B",
                    "Ki67": "Ki-67", "COL1": "Collagen Ty1", "HistoneH3": "Histone 3"}
acquisition_rows = []
for summary in sorted((EVIDENCE / "original_imc_summaries").glob("*.txt")):
    original = pd.read_csv(summary, sep="\t")
    matched = []
    for row in original.itertuples(index=False):
        label = str(row.Label).split("_", 1)[-1]
        spora_name = original_aliases.get(label, label)
        index = source_names.index(spora_name) if spora_name in source_names else None
        if index is not None:
            matched.append(spora_name)
        acquisition_rows.append(dict(source_file=relative(summary), original_page_index=row.Page,
                                     original_channel=row.Channel, original_label=row.Label,
                                     spora_source_name=spora_name if index is not None else "",
                                     spora_source_channel_index=index))
    assert set(matched) == set(source_names) and len(matched) == 20
pd.DataFrame(acquisition_rows).to_csv(CACHE / "original_acquisition_channel_map.csv", index=False, encoding="utf-8-sig")

# Supplementary Table 4 explicitly lists the selected core labels and clinical attributes.
# The paper's seven patients refer to the whole 63-core TMA, without a selected
# core-to-case map. Keep SPORA identities and the unresolved biological relationship.
text = (EVIDENCE / "supplementary_information.txt").read_text()
table4 = text.split("426", 1)[-1].split("Supplementary Table 4.", 1)[0]
clinical = {}
for match in re.finditer(r"((?:[A-Z]\d+[, ]*)+)\s+M\s+Adenocarcino\s+3\s+(IIIA|IIA|IB)\s+Malignant", table4):
    for core in re.findall(r"[A-Z]\d+", match[1]):
        clinical[core] = dict(sex_reported="male", histology="Adenocarcinoma", grade="3", source_stage=match[2])
assert len(clinical) == len(eligible)
sample_rows = []
image_samples = {}
for image_id in eligible:
    source_row = tissues.loc[image_id]
    core = str(source_row.original_file_name).removeprefix("Lung ")
    source_patient = str(source_row.patient_id)
    source_specimen = str(source_row.specimen_id)
    sample_id = f"{STUDY}_{source_patient}_{source_specimen}"
    image_samples[image_id] = sample_id
    sample_rows.append(dict(
        study=STUDY, sample_id=sample_id, patient_key=f"{STUDY}:{source_patient}",
        source_subject_id=source_patient, cohort_id="BS04081a", source_sample_id=str(source_row.original_file_name),
        sample_type="FFPE lung tumor specimen", tissue_type="tumor", anatomical_site="lung",
        preservation_method="ffpe", primary_diagnosis="Lung adenocarcinoma", primary_site="lung",
        **{key: value for key, value in clinical[core].items() if key != "source_stage"}, metadata_notes=(
            "取样键按SPORA tissue_id→patient_id/specimen_id建立，7行对应来源声明取样；"
            "source_sample_id和临床连接键为original_file_name中的core标签。补充表4提供sex/histology/grade/stage。"
            f"Supplementary Table 4来源Stage={clinical[core]['source_stage']}；时点未定义，stage_group_at_index留空。"
            "论文整张63-core TMA来自7位患者，未提供本批7个core各自的病例对应；独立生物学患者数未知。"
            "patient_key为SPORA声明身份；厂家case ID导出返回403，未取得独立病例对照。"
            "未取得年龄、TNM、治疗或生存资料；未推定分期起算时点及体系。")))
samples = pd.DataFrame(sample_rows).reindex(columns=pd.read_csv(TEMPLATES / "sample_metadata.csv").columns)
samples = samples.drop_duplicates(["study", "sample_id"])
pd.DataFrame([dict(core=key, **value) for key, value in clinical.items()]).to_csv(
    CACHE / "supplementary_table4.csv", index=False, encoding="utf-8-sig")

field_rows = []
def mapping(source_file: Path, source_table: str, source_field: str, target_table: str,
            target_field: str, join: str, rule: str, notes: str = "", value_mapping: str = "") -> None:
    field_rows.append(dict(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_table,
                           source_field=source_field, target_table=target_table, target_field=target_field,
                           join_keys=join, value_mapping=value_mapping, transformation_rule=rule,
                           time_origin_rule="来源未定义时点，时间字段保持空值", conflict_rule="保持来源值；不推断缺失",
                           notes=notes))

for target, source, rule in [
    ("sample_id", "patient_id;specimen_id", "study+来源patient_id+specimen_id确定生物学取样；同键多图共享sample_id"),
    ("patient_key", "patient_id", "study命名空间+来源声明patient_id；本批core的独立病例身份未确认"),
    ("source_subject_id", "patient_id", "原值复制"),
    ("source_sample_id", "original_file_name", "原值复制；明确Lung core身份"),
]:
    mapping(RAW / "metadata/tissues.parquet", "tissues", source, "sample_metadata", target,
            "tissue_id=image_id", rule)
for target in ("sample_id", "core_id", "region_annotation"):
    mapping(RAW / "metadata/tissues.parquet", "tissues", "tissue_id;patient_id;specimen_id;original_file_name",
            "image_metadata", target, "tissue_id=source_image_id", "sample用来源取样键；core_id及region_annotation保留original_file_name")
for target, source, rule in [
    ("sex_reported", "Sex", "M→male"), ("histology", "Pathology Diagnosis", "合并排版换行：Adenocarcinoma"),
    ("primary_diagnosis", "Pathology Diagnosis;paper lung methods", "器官lung+来源Adenocarcinoma"),
    ("grade", "Grade", "原值字符串3"),
    ("stage_group_at_index", "Stage", "来源时点未定义，目标留空；Stage原值保存在metadata_notes"),
    ("metadata_notes", "Stage", "记录Supplementary Table 4来源Stage原值及未定义时点，保留IB/IIA/IIIA"),
    ("tissue_type", "Type", "Malignant→tumor"),
]:
    mapping(EVIDENCE / "supplementary_information.pdf", "Supplementary Table 4", source,
            "sample_metadata", target, "Core number=original_file_name.removeprefix('Lung ')", rule,
            "缓存解析表：" + relative(CACHE / "supplementary_table4.csv"))
for target, rule in [
    ("cohort_id", "论文methods明确阵列BS04081a"), ("sample_type", "论文methods明确FFPE lung tumor取样"),
    ("preservation_method", "论文FFPE→ffpe"), ("anatomical_site", "论文及来源Lung→lung"),
    ("primary_site", "论文原发lung adenocarcinoma→lung"),
]:
    mapping(EVIDENCE / "article.html", "Tissue preparation and isotope-conjugates antibody labeling",
            "lung samples", "sample_metadata", target, "论文lung cohort→tissues.original_file_name", rule)
mapping(RAW / "imc/channels.parquet", "channels + channels_per_tissue", "channel_name;is_nuclear_marker",
        "panel_mapping", "source_channel_index", "tissue_id+availability True按channels原顺序", "逐图与OME及既有native verification核对20通道顺序")
mapping(S1, "supplementary_table_s1_marker_mapping", "Study;Original marker;Unified marker;Segmentation channel",
        "panel_mapping", "marker_id", "Study+Original marker=study+source_name", "精确study对接；20通道全保留，CD45RA/CD45RO按S1分别保留")
mapping(CACHE / "original_acquisition_channel_map.csv", "Zenodo original IMC summary notes", "original_page_index;original_label",
        "panel_mapping", "source_name", "原始采集标签→本study明确别名→SPORA channel_name",
        "5份原始采集注释各27页均确认同一20蛋白身份；原始page与SPORA source_channel_index分别记录",
        "E6/F7无单独summary附件，但7个原始core目录、论文表4和SPORA元数据一致；本次输入来源为已核验SPORA20通道")
mapping(EVIDENCE / "github_tree.json", "author repository + notebooks", "notebooks/07_vae_patient_comparison_lung.ipynb",
        "sample_metadata", "metadata_notes", "作者仓库公开study资料", "已查作者仓库及patient notebook；未找到独立vendor case/core map",
        "https://github.com/coskunlab/scSpaMet；Zenodo原始发布页https://zenodo.org/records/6784251；厂家导出尝试403")
field_rows.append(dict(
    study=STUDY, source_file="scripts/data/preparation/Hu2023_IMC_Lung.py",
    source_sheet_or_table="study configuration", source_field="STUDY",
    target_table="sample_metadata", target_field="study",
    join_keys="One canonical study declaration for this output table",
    value_mapping="", transformation_rule="Copy the stable CellWeft study identifier declared by the dataset preparer to every row.",
    time_origin_rule="", conflict_rule="All sample rows use the same declared study identifier.",
    notes="Internal canonical identifier; it labels the prepared study and does not assert patient identity."))
fields = pd.DataFrame(field_rows).reindex(columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)

source_trace = dict(
    paper_url=PAPER_URL, supplement_url=SUPPLEMENT_URL,
    source_release="SPORA hu2023single / OME-Zarr 0.5 / highest published 1.0 mpp",
    native_verification=relative(PROOF), accepted_scope=relative(ACCEPTED), file_stat_binding=relative(FILE_EVIDENCE),
    original_publish_url="https://zenodo.org/records/6784251",
    original_publish_checked="Zenodo页面为2.7GB图像归档，本次不重复下载图像；关联metadata从论文及作者仓库追查",
    author_repository="https://github.com/coskunlab/scSpaMet",
    author_files=["github_tree.json", "github_README.md", "01_IMC_processing_large.ipynb", "07_vae_patient_comparison_lung.ipynb"],
    original_archive_members=relative(EVIDENCE / "zenodo_zip_members.json"),
    original_acquisition_notes=relative(CACHE / "original_acquisition_channel_map.csv"),
    original_archive_metadata_check="远端Range读取ZIP中央目录及5个KB级采集summary；7个原始Lung core目录一致，未发现临床表附件",
    clinical_table="Supplementary Table 4", sample_join="tissue_id→original_file_name core→Table 4 Core number",
    patient_join="SPORA tissue_id→patient_id；仅保留来源声明身份",
    paper_patient_scope="论文整张63-core TMA来自7位患者，研究选择7个core的21个ROI；未给出所选core各自的病例对应",
    patient_limit="厂家BS04081a规格导出403；本批7行声明patient_key不代表已确认7名独立生物学患者，独立病例连接为0/7",
    inventory_images=len(tissues), admitted_local_images=len(eligible),
    other_inventory_images=len(tissues)-len(eligible), other_scope="保留既有panel准入；其他44个inventory对象不在本地合格范围",
    output_uncompressed_bytes=sum(int(np.prod(proofs[i]["shape"]))*8 for i in eligible),
    peak_local_staging_bytes=max(int(np.prod(proofs[i]["shape"]))*8 for i in eligible),
    missing_fields="年龄、TNM、治疗、随访及生存信息未取得，留空；未从core名字推断患者",
    independent_review="保留全部既有独立复核记录；本轮TIFF交付及复核行保留修补仅为整理者自检，待独立CLI复查")

# Merge previously published images; reuse a correct file only with its receipt binding.
images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists()
          else pd.DataFrame(columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns))
publish()
progress("metadata_prepared")

if args.action == "stage":
    staged_reports = []
    for image_id in eligible:
        values, local_path = stage_native(image_id)
        nonzero = np.zeros(len(source_names), dtype=np.int64)
        for x0, y0, x1, y1 in windows(values.shape[-2:], 256):
            block = values[:, y0:y1, x0:x1]
            assert np.isfinite(block).all()
            nonzero += np.count_nonzero(block, axis=(1, 2))
        admitted = bool(nonzero[(panel.role == "downstream").to_numpy()].any())
        assert admitted
        overview(image_id, values)
        staged_reports.append(dict(image_id=image_id, source_binding=source_binding(image_id),
                                   shape=list(values.shape), dtype=str(values.dtype), all_finite=True,
                                   channel_nonzero_pixels=nonzero.tolist(), admitted=admitted,
                                   native_staging_path=str(local_path)))
        values._mmap.close()
        progress("native_staged", image_id=image_id,
                 staged_images=len(list(LOCAL.glob("*.npy"))), temporary_storage=str(LOCAL))
        print(f"staged {image_id}", flush=True)
    save_json(CACHE / "source_admission.json", staged_reports)

if args.action in {"run", "export"}:
    for image_id in eligible:
        record = proofs[image_id]
        shape = tuple(record["shape"])
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = CACHE / "image_receipts" / f"{image_id}.json"
        binding = source_binding(image_id)
        revision = binding["source_revision"]
        if destination.exists() and receipt_path.exists() and image_id in set(images.image_id):
            receipt = json.loads(receipt_path.read_text())
            current = destination.stat()
            assert receipt["input_binding"] == binding
            assert receipt["output_stat"] == [current.st_size, current.st_mtime_ns]
            progress("reused", image_id=image_id)
            continue
        required = int(np.prod(shape)) * 8
        assert shutil.disk_usage(NAS).free - required >= 100*1024**3
        assert shutil.disk_usage(LOCAL).free - required >= 64*1024**3
        values, local_path = stage_native(image_id)
        completed = False
        try:
            # The exporter reads the bound local memmap; NAS is used only for publication.
            spec = SourceSpec(image_id, PANEL_ID, "npy", local_path, tuple(range(20)),
                              pixel_size_um=(1.0, 1.0), source_revision=revision)
            if destination.exists():
                # A completed TIFF may precede its index when read-back or table publication stops.
                # Establish full native equality before reusing this unindexed derived file.
                progress("checking_unindexed_output", image_id=image_id)
                with formal_read_lease(destination):
                    with resource_lock(OUTPUT, shared=True):
                        with resource_lock(destination.parent, shared=True, name=".input.lock"):
                            with tifffile.TiffFile(destination) as tif:
                                assert tif.series[0].shape == shape
                                assert tif.series[0].dtype == values.dtype
                                for channel in range(len(retained)):
                                    decoded = tif.pages.get(channel, aspage=True).asarray(maxworkers=2)
                                    assert np.array_equal(decoded, values[channel])
                result = dict(admitted=True, recovered_existing_output=True,
                              full_native_equality_channels=len(retained))
            else:
                progress("exporting_native", image_id=image_id)
                result = export_input(StagedReader(values, spec), panel, destination, tile_size=256)
            assert result["admitted"], "All admitted markers are zero; scope needs an exclusion record"
            progress("checking_saved_image", image_id=image_id)
            with formal_read_lease(destination):
                with resource_lock(OUTPUT, shared=True):
                    with resource_lock(destination.parent, shared=True, name=".input.lock"):
                        source = zarr.open_array(str(NAS / record["path"] / "0"), mode="r")
                        try:
                            assert source.shape == shape and source.dtype == np.dtype("float64")
                            pixel_checks = check_pixels(destination, source, shape)
                        finally:
                            source.store.close()
            overview(image_id, values)
            row = dict(study=STUDY, image_id=image_id, sample_id=image_samples[image_id], panel_id=PANEL_ID,
                       image_kind="tma_core", source_image_id=image_id, source_files=[record["path"]],
                       source_format="ome-zarr", source_axes="CYX", source_dtype="float64", source_revision=revision,
                       slide_id="BS04081a", core_id=str(tissues.loc[image_id, "original_file_name"]),
                       region_annotation=str(tissues.loc[image_id, "original_file_name"]),
                       source_x0_px=0, source_y0_px=0, source_x1_px=shape[2], source_y1_px=shape[1],
                       pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                       height_px=shape[1], width_px=shape[2], source_channel_count=20, channel_count=20,
                       pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                       intensity_description="SPORA published native float64 intensities; no rescaling or normalization",
                       split_method="already_independent", split_evidence=relative(RAW / "metadata/tissues.parquet"),
                       metadata_notes="完整来源TMA core采集框；来源已方形，padding为0。sample/patient连接及身份限度见field_mapping与source_trace。")
            images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
            publish()
            stat = destination.stat()
            save_json(receipt_path, dict(image_id=image_id, input_binding=binding, output_stat=[stat.st_size, stat.st_mtime_ns],
                                         export=result, pixel_checks=pixel_checks, self_check_only=True))
            completed = True
            progress("image_saved", image_id=image_id)
            print(f"saved {image_id}: {len(images)}/{len(eligible)}", flush=True)
        finally:
            values._mmap.close()
            if completed:
                local_path.unlink(missing_ok=True)
                local_path.with_suffix(".json").unlink(missing_ok=True)

if args.action in {"run", "verify"}:
    progress("verifying_outputs")
    summaries = []
    for row in images.itertuples(index=False):
        record = proofs[row.image_id]
        destination = NAS / row.input_image_path
        with formal_read_lease(destination):
            with resource_lock(OUTPUT, shared=True):
                with resource_lock(destination.parent, shared=True, name=".input.lock"):
                    source = zarr.open_array(str(NAS / record["path"] / "0"), mode="r")
                    try:
                        checks = check_pixels(destination, source, tuple(record["shape"]))
                        summaries.append(dict(image_id=row.image_id, header_pass=True, pixel_checks=checks))
                    finally:
                        source.store.close()
    save_json(CACHE / "self_check.json", dict(study=STUDY, preparer=PREPARER, independent_reviewer=None,
        source_images=len(eligible), exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        source_declared_patient_count=int(samples.patient_key.nunique()), checks=summaries,
        patient_identity_limit="保留来源声明patient_id；厂家病例ID无法独立认证", independent_review="pending"))

# Preserve prior independent findings; append only affected checks for re-review.
review_path = OUTPUT / "review_results.csv"
review = pd.read_csv(review_path) if review_path.exists() else pd.read_csv(TEMPLATES / "review_results.csv")
checks = pd.read_csv(TEMPLATES / "review_checks.csv")
rows = []
for check in checks[~checks.check_id.isin(["S01", "P01", "P02", "M02"])].itertuples(index=False):
    details = dict(repair_round=2, review_type="preparer_repair_self_check",
                   preparer_self_check=("completed" if args.action in {"run", "verify"} else "metadata_only"),
                   independent_review="pending", exported_images=len(images), expected_images=len(eligible),
                   sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                   source_linkable_images=len(eligible),
                   patient_linked_samples=int(samples.patient_key.notna().sum()),
                   independently_confirmed_patient_linked_samples=0,
                   patient_identity="SPORA声明身份；本批独立生物学患者数未知",
                   repair_actions=f"仅替换本整理者reviewer为空的自检行，保留全部独立复核记录；复用原生暂存，正式TIFF和39字段索引实际交付{len(images)}/{len(eligible)}；M01独立病例关系缺口保持",
                   prior_review=relative(JOB.parent / "0190_review_2/report.json"))
    rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                     record_id=PANEL_ID if check.scope == "panel" else STUDY,
                     check_id=check.check_id, result="unresolved", source_evidence=relative(CACHE / "source_trace.json"),
                     output_evidence=relative(CACHE / "self_check.json") if args.action in {"run", "verify"} else relative(OUTPUT),
                     details=json.dumps(details, ensure_ascii=False)))
with formal_publish_lease(OUTPUT, directory=OUTPUT):
    own_self_checks = (review.preparer == PREPARER) & review.reviewer.fillna("").str.strip().eq("")
    review = pd.concat([review[~own_self_checks], pd.DataFrame(rows, columns=review.columns)], ignore_index=True)
    pending = OUTPUT / "review_results.csv.tmp"
    review.to_csv(pending, index=False, encoding="utf-8-sig")
    pending.replace(OUTPUT / "review_results.csv")
progress("prepared_waiting_independent_review" if len(images) == len(eligible)
         else "native_staged" if args.action == "stage" else "metadata_prepared",
         staged_images=len(list(LOCAL.glob("*.npy"))))
print(json.dumps(dict(study=STUDY, expected_images=len(eligible), exported_images=len(images),
                      sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum())), ensure_ascii=False))
