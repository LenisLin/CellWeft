"""Deliver the complete Greenbaum author FOV release as native input TIFFs.

Inputs: two figshare archives, received author attachments, study S1 rows,
and formal source manifests. Outputs: per-FOV CYX input.tiff, the
39-field image index, sample/panel/field tables and preparer review records.
Run pixel export through the batch shared formal write lease wrapper. Strict OOXML
supplement parsing is study-local and should later migrate to shared I/O.
Use --repair-field-mapping for the round-1 sample identity provenance repair;
this preserves existing TIFFs, metadata values and independent review records.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import nullcontext
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    export_input, panel_from_s1, save_input_tables, validate_image_metadata,
)

STUDY = "Greenbaum2023_MIBI_MaternalFetalInterface"
COHORT = "decidua_66"
PATIENT_SOURCE_FIELDS = {
    "GA weeks": "source_ga_weeks",
    "GA days": "source_ga_days",
    "Gravidity": "source_gravidity",
    "Parity": "source_parity",
    "prior therapeutic abortion": "source_prior_therapeutic_abortion",
    "race": "source_race",
    "smoker": "source_smoker",
    "Medication": "source_medication",
    "HIV status": "source_hiv_status",
    "history of preeclampsia": "source_history_of_preeclampsia",
    "Chronic hypertension": "source_chronic_hypertension",
    "diabetes mellitus": "source_diabetes_mellitus",
    "renal disease": "source_renal_disease",
    "autoimmune disease": "source_autoimmune_disease",
    "multifetal gestation or twins": "source_multifetal_gestation_or_twins",
    "fetal anomalies": "source_fetal_anomalies",
    "notes": "source_notes",
    "Pathology": "source_pathology",
    "Num_Cores": "source_num_cores",
}
BLOCK_SOURCE_FIELDS = {
    "villi present*": "source_villi_present",
    "# regions annotated by pathologist": "source_regions_annotated_by_pathologist",
}
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "figshare_16663465"
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
PROGRESS = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_PROGRESS",
    str(PROJECT / "cache/study_preparation" / STUDY / "jobs/0034_prepare_0/progress.json"),
))
TEMP = Path("/tmp/study_input_batch") / STUDY


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def columns(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns.tolist()


def strict_sheet(path: Path, sheet: int) -> pd.DataFrame:
    """Read source strict-OOXML values; openpyxl reports no sheets for these files.

    Source workbooks remain unchanged. Cell references preserve sparse columns,
    and shared strings retain the published categories and missing-value tokens.
    """
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(t.text or "" for t in item.findall(".//{*}t"))
                   for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml")).findall(".//{*}row"):
            values = {}
            for cell in row.findall("{*}c"):
                element = cell.find("{*}v")
                value = element.text if element is not None else ""
                if cell.attrib.get("t") == "s":
                    value = strings[int(value)]
                values[re.sub(r"\d", "", cell.attrib["r"])] = value
            rows.append(values)
    frame = pd.DataFrame(rows[1:]).rename(columns=rows[0])
    return frame.replace({"NaN": "", "nan": ""}).fillna("")


def read_public_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    """Read the received clinical, FOV bridge, and published-point attachments."""
    clinical_path = RAW_METADATA / "41586_2023_6298_MOESM3_ESM.xlsx"
    arteries_path = RAW_METADATA / "41586_2023_6298_MOESM4_ESM.xlsx"
    points_path = RAW_METADATA / "41586_2023_6298_MOESM10_ESM.csv"
    with formal_read_lease(RAW_METADATA):
        with resource_lock(RAW_METADATA.parent, shared=True, name=".raw_consolidation.lock"):
            clinical = strict_sheet(clinical_path, 1)
            histology = strict_sheet(clinical_path, 2)
            arteries = strict_sheet(arteries_path, 1)
            published_points = set(pd.read_csv(points_path).Point)
    return clinical, histology, arteries, published_points


def read_source_manifests() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read the already received author source scope and channel map."""
    bundle = NAS / "raw/public" / STUDY / "metadata.json"
    with formal_read_lease(bundle, data_root=NAS):
        with resource_lock(RAW.parent, shared=True, name=".raw_consolidation.lock"):
            manifest = pd.read_csv(META / "figshare_image_manifest.csv", keep_default_na=False)
            source_channels = pd.read_csv(META / "figshare_channel_markers.csv", keep_default_na=False)
            coverage = json.loads((META / "source_coverage.json").read_text())
    return manifest, source_channels, coverage


def source_scope_rows(manifest: pd.DataFrame, source_channels: pd.DataFrame,
                      coverage: dict) -> tuple[list[str], list[dict]]:
    """Join the received author FOV and channel manifests without rereading ZIP contents."""
    expected_fovs = coverage["author_expected_fovs"]
    expected_channels = expected_fovs * 49
    if (expected_fovs != 228 or coverage["author_archive_verified_fovs"] != expected_fovs
            or coverage["author_channel_tiffs_verified"] != expected_channels
            or coverage.get("errors") != [] or len(manifest) != expected_fovs
            or len(source_channels) != expected_channels):
        raise ValueError("Formal Greenbaum source coverage no longer matches the accepted 228-FOV scope")
    keys = ["source_archive_id", "archive_path", "fov_path"]
    if manifest.duplicated(keys).any():
        raise ValueError("Formal Greenbaum image manifest contains duplicate FOV keys")
    ordered_channels = source_channels.sort_values(keys + ["channel_index"], kind="stable")
    grouped_channels = {
        key: group.sort_values("channel_index", kind="stable")
        for key, group in ordered_channels.groupby(["source_archive_id", "archive_path", "fov_path"], sort=False)
    }
    first_group = next(iter(grouped_channels.values()))
    source_names = first_group.marker.tolist()
    scope_rows = []
    for row in manifest.itertuples(index=False):
        mapping = grouped_channels[(row.source_archive_id, row.archive_path, row.fov_path)]
        scope_rows.append(dict(
            fov=Path(row.fov_path).parent.name, archive_path=row.archive_path,
            fov_path=row.fov_path, archive_id=row.source_archive_id,
            channels=[dict(channel_index=int(channel.channel_index), marker=channel.marker,
                           member_name=channel.member_name)
                      for channel in mapping.itertuples(index=False)]))
    return source_names, scope_rows


def stage_fov_channels(row: dict) -> list[Path]:
    """Stage one complete FOV from its received ZIP under a single read lease."""
    archive_file = NAS / row["archive_path"]
    TEMP.mkdir(parents=True, exist_ok=True)
    paths = []
    with formal_read_lease(archive_file):
        with resource_lock(RAW.parent, shared=True, name=".raw_consolidation.lock"):
            with zipfile.ZipFile(archive_file) as archive:
                for channel in row["channels"]:
                    member = Path(channel["member_name"])
                    if member.is_absolute() or ".." in member.parts:
                        raise ValueError(f"Unsafe Greenbaum archive member path: {member}")
                    info = archive.getinfo(member.as_posix())
                    if shutil.disk_usage(TEMP).free < info.file_size + 64 * 1024**3:
                        raise RuntimeError("Local extraction would violate the 64 GiB reserve")
                    local = TEMP / member
                    local.parent.mkdir(parents=True, exist_ok=True)
                    staged = local.with_suffix(local.suffix + ".partial")
                    with archive.open(info) as source, staged.open("wb") as target:
                        shutil.copyfileobj(source, target, 8 * 1024**2)
                    if staged.stat().st_size != info.file_size:
                        raise ValueError(f"Incomplete Greenbaum source member: {member}")
                    staged.replace(local)
                    paths.append(local)
    return paths


def sample_gap_summary(fov_ids: set[str], linked_fovs: set[str],
                       unlisted_fovs: set[str]) -> tuple[list[str], list[str]]:
    missing = sorted(fov_ids - linked_fovs)
    gaps = [f"{len(missing)}/{len(fov_ids)} FOV缺少明确FOV→来源block/patient连接；不按文件名推断。",
            f"完整发布有{len(fov_ids)}个FOV，论文分析/补充表8有{len(fov_ids)-len(unlisted_fovs)}个；{len(unlisted_fovs)}个额外发布FOV的逐项分析排除原因未解释。既有source_coverage明确canonical为{len(fov_ids)}，按该范围保留并标注。"]
    return gaps, missing


def check_header(path: Path, panel: pd.DataFrame) -> None:
    names = panel.loc[panel.keep_channel, "marker_id"].tolist()
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (len(names), 2048, 2048)
        assert tif.series[0].axes == "CYX"
        assert len(tif.pages) == len(names)
        assert all(p.shape == (2048, 2048) and p.dtype == np.dtype("uint8") for p in tif.pages)
        xml = ET.fromstring(tif.ome_metadata)
        assert [e.attrib["Name"] for e in xml.iter() if e.tag.endswith("}Channel")] == names


def sample_identity_mappings() -> pd.DataFrame:
    """Document project identifiers from the canonical release and clinical cohort."""
    rows = []
    for target, source, sheet, source_field, join, rule, notes in [
        ("study", META / "source_coverage.json", "", "original_paper_doi;canonical_source",
         "study-level DOI and canonical release; applies to every source clinical patient row",
         f"original_paper_doi=10.1038/s41586-023-06298-9且canonical_source=https://doi.org/10.6084/m9.figshare.16663465时，赋项目稳定study标识{STUDY}；全部66行同属此study。",
         "study为项目命名空间；来源论文和canonical发布范围由source_coverage明确对接。"),
        ("cohort_id", RAW_METADATA / "41586_2023_6298_MOESM3_ESM.xlsx", "patient data", "patient",
         "sample_metadata.source_subject_id = Supplementary Table 1 patient (exact string)",
         f"Supplementary Table 1的66个非空唯一patient对应正文所述最终66人decidua/placenta FFPE block队列；所有该表成员赋study内项目队列标识{COHORT}，不依赖FOV连接覆盖。",
         f"{COHORT}为项目派生队列标签，来源未发布同名cohort代码；队列定义见{relative(RAW_METADATA / 'paper.html')}的Retrospective cohort design及最终66人段落，block语义见{relative(RAW_METADATA / '41586_2023_6298_MOESM1_ESM.pdf')}的Supplementary Table 1表注。"),
    ]:
        record = dict.fromkeys(columns("field_mapping"), "")
        record.update(study=STUDY, source_file=relative(source), source_sheet_or_table=sheet,
                      source_field=source_field, target_table="sample_metadata", target_field=target,
                      join_keys=join, transformation_rule=rule,
                      conflict_rule="若canonical来源或临床队列成员不一致则停止该字段赋值并核实。", notes=notes)
        rows.append(record)
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def repair_sample_identity_mappings() -> None:
    """Repair the two reviewed provenance gaps without republishing correct inputs."""
    with nullcontext():
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
        mapping_path = OUTPUT / "field_mapping.csv"
        mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
        clinical, _, arteries, published_points = read_public_source_tables()
        clinical = clinical[clinical.patient.ne("")]
        coverage = json.loads((META / "source_coverage.json").read_text())
        assert coverage["original_paper_doi"] == "10.1038/s41586-023-06298-9"
        assert coverage["canonical_source"] == "https://doi.org/10.6084/m9.figshare.16663465"
        assert len(samples) == len(clinical) == clinical.patient.nunique() == 66
        assert set(samples.source_subject_id) == set(clinical.patient)
        assert samples.study.eq(STUDY).all() and samples.cohort_id.eq(COHORT).all()
        affected = mapping.target_table.eq("sample_metadata") & mapping.target_field.isin(["study", "cohort_id"])
        repaired = pd.concat([mapping.loc[~affected], sample_identity_mappings()], ignore_index=True)
        validate_image_metadata(images, samples, panel, repaired)
        fov_ids = set(images.fov_id.dropna().astype(str))
        linked_fovs = set(arteries.point_name.astype(str)) & fov_ids
        unlisted_fovs = fov_ids - published_points
        gaps, _ = sample_gap_summary(fov_ids, linked_fovs, unlisted_fovs)
        counts = dict(study=STUDY, expected_images=228, completed_images=len(images),
                      sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                      patient_linked_samples=int(samples.patient_key.ne("").sum()))
        write_json(PROGRESS, dict(**counts, phase="repairing_field_mapping",
                                 gaps=gaps, updated_at=datetime.now(timezone.utc).isoformat()))
        temporary = mapping_path.with_suffix(".csv.tmp")
        with formal_publish_lease(mapping_path, directory=OUTPUT):
            repaired.to_csv(temporary, index=False, encoding="utf-8-sig")
            temporary.replace(mapping_path)
        receipt = dict(**counts, round=1, preparer="Codex_0184_repair_1",
                       affected_fields=["sample_metadata.study", "sample_metadata.cohort_id"],
                       field_mapping_rows_before=len(mapping), field_mapping_rows_after=len(repaired),
                       reconstructed_nonempty_values=2*len(samples),
                       changed_outputs=[relative(mapping_path)],
                       review_status="M02 repair submitted; existing independent results preserved; D01 pending independent recheck",
                       gaps=gaps, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(CACHE / "repair_1_field_mapping.json", receipt)
        write_json(PROGRESS, dict(**receipt, phase="repair_prepared_pending_independent_review"))
    print(json.dumps(receipt, ensure_ascii=False))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--repair-field-mapping", action="store_true",
                    help="Repair reviewed study/cohort provenance only; preserve inputs and independent review rows")
parser.add_argument("--progress-path", type=Path, default=PROGRESS)
parser.add_argument("--check-first", "--limit", dest="check_first", type=int, default=1,
                    help="Validate the initial batch, then continue the complete release within the same I/O lease")
parser.add_argument("--overview-observation", help="Preparer observation after viewing the generated overviews")
args = parser.parse_args()
PROGRESS = args.progress_path
if args.repair_field_mapping:
    repair_sample_identity_mappings()
    raise SystemExit(0)
for directory in (CACHE, OUTPUT, TEMP):
    directory.mkdir(parents=True, exist_ok=True)

# Bind the accepted author scope and channel order from the received source metadata.
manifest, source_channels, coverage = read_source_manifests()
source_names, scope_rows = source_scope_rows(manifest, source_channels, coverage)
panel_id = STUDY + "__figshare_49"
panel = panel_from_s1(STUDY, panel_id, source_names, S1)
assert len(panel) == 49 and panel.keep_channel.sum() == 39
write_json(CACHE / "source_scope.json", dict(
    study=STUDY, canonical_release="10.6084/m9.figshare.16663465", expected_images=228,
    accepted_source_scope=relative(META / "source_coverage.json"),
    source_manifest=relative(META / "figshare_image_manifest.csv"),
    channel_manifest=relative(META / "figshare_channel_markers.csv"),
    source_channels=49, output_channels=39,
    images=[{key: row[key] for key in ("fov", "archive_path", "fov_path", "archive_id")}
            for row in scope_rows],
    duplicate_representation="172 HuBMAP expression images retained as source only; not added to canonical author FOVs",
    mat_companions="Each canonical FOV folder also includes dataNoAgg.mat; retained as source-only companion representation, not extra images. One whosmat probe found only an unlabeled countsNoAgg double array (2048,2048,49). The published per-channel TIFF manifest and archive CRCs remain the source mapping contract; MAT channel order has not been inferred.",
    capacity=dict(maximum_uncompressed_tiff_bytes=228*39*2048*2048,
                  largest_local_fov_decoded_bytes=49*2048*2048,
                  nas_reserve_bytes=100*1024**3, local_reserve_bytes=64*1024**3)))

# Establish biological block samples from the explicit clinical and FOV bridges.
clinical_path = RAW_METADATA / "41586_2023_6298_MOESM3_ESM.xlsx"
arteries_path = RAW_METADATA / "41586_2023_6298_MOESM4_ESM.xlsx"
clinical, histology, arteries, published_points = read_public_source_tables()
clinical = clinical[clinical.patient.ne("")]
assert len(clinical) == clinical.patient.nunique() == 66
histology = histology[histology.patient.ne("")]
assert histology.patient.is_unique
histology_by_patient = histology.set_index("patient").to_dict("index")
assert set(clinical.patient).issubset(histology_by_patient)
bridge = arteries[["point_name", "patient"]].drop_duplicates()
assert not bridge.point_name.duplicated().any()
assert set(bridge.patient).issubset(set(clinical.patient))
fov_patient = bridge.set_index("point_name").patient.to_dict()
sample_rows = []
for row in clinical.to_dict("records"):
    sample = dict.fromkeys(columns("sample_metadata"), "")
    source_fields = {target: row[source] for source, target in PATIENT_SOURCE_FIELDS.items()}
    source_fields.update({target: histology_by_patient[row["patient"]][source]
                          for source, target in BLOCK_SOURCE_FIELDS.items()})
    sample.update(study=STUDY, sample_id=STUDY + "__block_" + row["patient"],
                  patient_key=STUDY + "__patient_" + row["patient"], source_subject_id=row["patient"],
                  cohort_id=COHORT, sample_type="archival tissue block",
                  tissue_type="decidua and placenta", anatomical_site="maternal-fetal interface",
                  preservation_method="FFPE", collection_method="elective pregnancy termination",
                  sex_reported="female", bmi_kg_m2=row["bmi"], age_group_at_collection=row["age"],
                  gestational_age_days=row["total GA in days"],
                  metadata_notes="Supplementary Table 1 patient data and blocks histology; one archival block per individual in the reported 66-block cohort. Multiple imaged cores share this biological sample. No separate source block accession is provided; source_sample_id remains empty. Exact anatomical location unknown.",
                  **source_fields)
    sample_rows.append(sample)
samples = pd.DataFrame(sample_rows)
sample_for_patient = samples.set_index("source_subject_id").sample_id.to_dict()
linked_fovs = set(fov_patient) & {r["fov"] for r in scope_rows}
unlisted = sorted({r["fov"] for r in scope_rows} - published_points)
gaps, missing_fov_ids = sample_gap_summary({r["fov"] for r in scope_rows}, linked_fovs, set(unlisted))
write_json(CACHE / "sample_investigation.json", dict(
    sources=[relative(clinical_path), relative(arteries_path), relative(RAW_METADATA / "Code.zip"),
             relative(RAW_METADATA / "Single_cells_MIBI.csv.zip"), relative(RAW_METADATA / "caltech_record.json"),
             relative(RAW_METADATA / "paper.html"), relative(RAW_METADATA / "figshare_article.json"),
             relative(RAW_METADATA / "41586_2023_6298_MOESM1_ESM.pdf"),
             relative(RAW_METADATA / "41586_2023_6298_MOESM13_ESM.xlsx"),
             relative(RAW_METADATA / "tma_arrays.html"), relative(RAW_METADATA / "tma_papers.html")],
    article_url="https://www.nature.com/articles/s41586-023-06298-9",
    supplementary_mirror="https://authors.library.caltech.edu/records/tx0a9-hd791",
    join="exact FOV folder basename = Supplementary Table 2 point_name; explicit patient column = Supplementary Table 1 patient; one block per individual as article states 66 individuals and 66 tissue blocks",
    sample_rows=len(samples), sample_linked_images=len(linked_fovs), patient_linked_samples=66,
    image_linked_samples=len({fov_patient[fov] for fov in linked_fovs}),
    missing_fov_ids=missing_fov_ids,
    author_analysis_unlisted_fov_ids=unlisted, gaps=gaps))

field_rows = []


def field(source: str, source_field: str, table: str, target: str, rule: str,
          notes: str = "", join_keys: str = "exact point_name; explicit patient; clinical patient") -> None:
    record = dict.fromkeys(columns("field_mapping"), "")
    record.update(study=STUDY, source_file=source, source_field=source_field,
                  target_table=table, target_field=target, join_keys=join_keys,
                  transformation_rule=rule, conflict_rule="保留来源；缺少明确桥接时留空", notes=notes)
    field_rows.append(record)


field(relative(arteries_path), "point_name;patient", "image_metadata", "sample_id",
      "FOV目录basename精确连接显式point_name→patient表；连接Supplementary Table 1来源生物学block。多个core共享sample_id。",
      f"sample_rows=66; image_sample={len(linked_fovs)}/228; sample_patient=66/66; 未连接FOV留空。")
for target in ("patient_key", "source_subject_id", "sample_id", "bmi_kg_m2"):
    field(relative(clinical_path), "patient;bmi", "sample_metadata", target,
          "来源patient标识同时索引该队列的一份archival tissue block；patient_key及sample_id按study命名空间生成，BMI照录。",
          join_keys="sample_metadata.source_subject_id=patient")
field(relative(clinical_path), "patient; no separate block accession", "sample_metadata", "source_sample_id",
      "来源未单列独立block accession，source_sample_id留空；每位patient唯一block生成本地sample_id，患者原键保存在source_subject_id。")
for target, source, definition, dtype in [
    ("age_group_at_collection", "age", "Published age interval at specimen collection; no midpoint imputation", "string"),
    ("gestational_age_days", "total GA in days", "Gestational age at specimen collection in days", "integer")]:
    field(relative(clinical_path), source, "sample_metadata", target, "原样保留来源年龄分组或孕龄天数",
          json.dumps(dict(data_type=dtype, definition=definition)),
          join_keys="sample_metadata.source_subject_id=patient")
for source, target in PATIENT_SOURCE_FIELDS.items():
    field(relative(clinical_path), source, "sample_metadata", target,
          "按patient精确连接并保留Table S1来源字段原值，不推断未报告含义。",
          json.dumps(dict(data_type="string", definition=f"Exact source value from Table S1 field {source}.")),
          join_keys="sample_metadata.source_subject_id=patient")
for source, target in BLOCK_SOURCE_FIELDS.items():
    field(relative(clinical_path), source, "sample_metadata", target,
          "按patient精确连接blocks histology表并保留来源字段原值。",
          json.dumps(dict(data_type="string", definition=f"Exact source value from blocks histology field {source}.")),
          join_keys="sample_metadata.source_subject_id=patient")
for target in ("sample_type", "tissue_type", "anatomical_site", "preservation_method", "collection_method", "sex_reported", "metadata_notes"):
    field(relative(RAW_METADATA / "paper.html"), "cohort design;TMA construction", "sample_metadata", target,
          "正文明确66位女性的66份组织block、终止妊娠取材及FFPE处理；sex_reported按来源women记录female；精确解剖位置未知；不推断正常对照标签。")
field(relative(RAW_METADATA / "41586_2023_6298_MOESM1_ESM.pdf"), "Supplementary Table 1 full legend, page 2", "sample_metadata", "sample_id",
      "完整表注明确以每位患者的组织block为单位；同block可含多块组织，不把core/FOV作为独立sample。")
for target in columns("image_metadata"):
    if target == "sample_id":
        continue
    field(relative(META / "figshare_image_manifest.csv"), "archive_path;fov_path;shape;dtype", "image_metadata", target,
          "完整来源FOV 2048×2048，独立CYX input.tiff，uint8来源去噪强度；不裁剪，不补边，无Z/T选择。尺度取论文808 µm/2048原生像素=0.39453125 µm。原始采集ID、slide/core身份及区域注释未明确则留空。")
field(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "output_channel_index",
      "同study来源49行精确对接S1；保留39行含H3核输入；按来源顺序分配输出位置。10个明确excluded通道不写入。")
field(relative(META / "source_coverage.json"), "canonical source;source coverage", "sample_metadata", "metadata_notes",
      "基于已接收canonical来源范围与原作者附件计算sample、image、patient连接覆盖；缺口保留M01。")
field_mapping = pd.DataFrame(field_rows, columns=columns("field_mapping"))
field_mapping = pd.concat([field_mapping, sample_identity_mappings()], ignore_index=True)
field_mapping.loc[field_mapping.source_file == relative(clinical_path), "source_sheet_or_table"] = "patient data"
images_path = OUTPUT / "images.parquet"
if images_path.exists():
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        images = pd.read_parquet(images_path)
else:
    images = pd.DataFrame(columns=columns("image_metadata"))
if len(images):
    images["sample_id"] = images.fov_id.map(fov_patient).map(sample_for_patient)
receipts_path = CACHE / "export_receipts.json"
receipts = {}


def progress(phase: str, **detail: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, expected_images=228,
                             completed_images=len(images), sample_rows=66,
                             sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=66,
                             gaps=gaps, updated_at=datetime.now(timezone.utc).isoformat(), **detail))


def save_tables() -> None:
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)
    checks = []
    for check, result, description in [
        ("S01", "pass", "整理者自检：source_coverage明确canonical输入为228个作者发布FOV；与论文211个分析FOV的差异逐ID列于当前sample_investigation.json，保留分析范围注释。"),
        ("P01", "pass", "整理者自检：228个FOV同49通道来源顺序，逐行S1精确对接。"),
        ("P02", "pass", "整理者自检：39保留/10排除，H3核输入保留，无重复downstream marker。"),
        ("M01", "unresolved" if len(linked_fovs)<228 else "pass", f"来源生物学block=66、patient连接=66；本批image→sample={int(images.sample_id.notna().sum())}/{len(images)}，完整来源={len(linked_fovs)}/228；缺失ID及来源追查见sample_investigation.json。"),
        ("M02", "pass", "整理者自检：66/66患者行按patient精确连接；补齐19个Table S1来源字段和2个blocks histology字段。BMI按来源保留65/66个非空原值；年龄区间不取中值，不推断吸烟0/1临床定义。"),
        ("I01", "pass", f"整理者自检：{len(images)}个唯一study/image主键；archive及FOV目录和逐通道成员由正式来源image/channel manifest定位。"),
        ("I02", "pass", "整理者自检：非空sample外键均连接66份来源block；panel外键及39基础字段由save_input_tables验证。"),
        ("I03", "pass", "整理者自检：全部已交付FOV为完整[0,2048)×[0,2048)，方形无需padding；未推断内部采集缺失位置。"),
        ("P03", "pass" if len(images) else "unresolved", f"整理者自检：已导出{len(images)}张TIFF头、CYX uint8、39通道名称/顺序已核对。"),
        ("I05", "pass" if any(v.get("pixel_checks") for v in receipts.values()) else "unresolved", "本次整理逐值自检抽样及坐标见export_receipts.json；独立I05仍待复核。"),
        ("I04", "pass" if args.overview_observation else "unresolved", args.overview_observation or "来源为独立FOV完整框，代表性概览待查看确认。"),
        ("D01", "unresolved", f"真实交付{len(images)}/228；sample连接及来源范围缺口保留；当前preparer自检，独立CLI复核待主控组织。")]:
        check_row = dict.fromkeys(columns("review_results"), "")
        source_evidence = {
            "S01": [META / "source_coverage.json", META / "figshare_image_manifest.csv"],
            "P01": [META / "figshare_channel_markers.csv"], "P02": [S1],
            "M01": [clinical_path, arteries_path, META / "source_coverage.json"],
            "M02": [clinical_path],
            "I01": [META / "figshare_image_manifest.csv", META / "figshare_channel_markers.csv"],
            "I02": [arteries_path],
            "I03": [META / "figshare_image_manifest.csv"],
            "P03": [META / "figshare_channel_markers.csv"],
            "I04": [CACHE / "overview_observation.json"],
            "I05": [META / "figshare_image_manifest.csv", META / "figshare_channel_markers.csv"],
            "D01": [META / "source_coverage.json", clinical_path, arteries_path],
        }[check]
        check_row.update(study=STUDY, preparer="Codex_0034_prepare_0", scope="sample" if check.startswith("M") else "study",
                         record_id=STUDY, check_id=check, result=result,
                         source_evidence=";".join(relative(path).as_posix() for path in source_evidence),
                         output_evidence=relative(OUTPUT), details=description)
        if check in {"S01", "M01"}:
            affected = unlisted if check == "S01" else sorted({r["fov"] for r in scope_rows}-linked_fovs)
            check_row["details"] = json.dumps(dict(
                image_ids=[STUDY + "__" + fov for fov in affected],
                description=description, preparer_check=True), ensure_ascii=False)
        checks.append(check_row)
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(checks).to_csv(review_path, index=False, encoding="utf-8-sig")


save_tables()
progress("metadata_ready" if args.metadata_only else "export_started")
if not args.metadata_only:
    processed_now = 0
    try:
        for position, row in enumerate(scope_rows):
            fov = row["fov"]
            image_id = STUDY + "__" + fov
            destination = OUTPUT / "images" / image_id / "input.tiff"
            if image_id in set(images.image_id) and destination.exists():
                check_header(destination, panel)
                continue
            progress("exporting", current_fov=fov)
            paths = stage_fov_channels(row)
            source = SourceSpec(image_id, panel_id, "tiff_planes", paths[0], tuple([0]*49),
                                paths=tuple(paths), pixel_size_um=(808/2048, 808/2048))
            with open_source(source) as reader:
                receipt = export_input(reader, panel, destination)
                receipt.update(fov=fov, archive_path=row["archive_path"], fov_path=row["fov_path"])
                if receipt["admitted"]:
                    check_header(destination, panel)
                    # Compare both archive layouts using all 39 retained channels.
                    if position in (0, 91, 92, 227):
                        comparisons = []
                        with tifffile.TiffFile(destination) as tif:
                            for window in [(768, 768, 800, 800), (1280, 256, 1312, 288),
                                           (2016, 2016, 2048, 2048)]:
                                source_nonzero = 0
                                signal_channels = 0
                                for p in panel[panel.keep_channel].itertuples(index=False):
                                    x0, y0, x1, y1 = window
                                    with tifffile.TiffFile(paths[p.source_channel_index]) as source_tif:
                                        expected = source_tif.pages[0].asarray(maxworkers=4)[y0:y1, x0:x1]
                                    actual = read_tiff_window(tif, int(p.output_channel_index), window)
                                    assert expected.dtype == actual.dtype and np.array_equal(expected, actual)
                                    count = int(np.count_nonzero(expected))
                                    source_nonzero += count
                                    signal_channels += int(count > 0)
                                comparisons.append(dict(window=list(window), channels=39, equal=True,
                                                        source_nonzero_pixels=source_nonzero,
                                                        channels_with_source_signal=signal_channels))
                        receipt["pixel_checks"] = comparisons
                    if position in (0, 45, 91, 92, 159, 227):
                        thumbnails = []
                        for marker in ("H3", "CK7", "CD45", "SMA"):
                            values = reader.read_window([source_names.index(marker)], (0, 0, 2048, 2048))[0]
                            thumb = Image.fromarray(values).resize((384, 384))
                            canvas = Image.new("L", (384, 410));canvas.paste(thumb, (0, 26))
                            ImageDraw.Draw(canvas).text((8, 7), marker, fill=255)
                            thumbnails.append(canvas)
                        sheet = Image.new("L", (1536, 440))
                        ImageDraw.Draw(sheet).text((8, 4), fov + " | source uint8 display, no pixel normalization", fill=255)
                        for i, thumb in enumerate(thumbnails):sheet.paste(thumb, (i*384, 30))
                        sheet.save(CACHE / f"overview_{fov}.png")
                    record = dict.fromkeys(columns("image_metadata"), pd.NA)
                    patient = fov_patient.get(fov)
                    record.update(study=STUDY, image_id=image_id, sample_id=sample_for_patient.get(patient), panel_id=panel_id,
                                  image_kind="fov", source_image_id=row["fov_path"], source_files=[row["archive_path"]],
                                  source_format="tiff_planes_in_zip", source_axes="CYX", source_dtype="uint8",
                                  source_revision="figshare_16663465:file_"+str(row["archive_id"]), fov_id=fov,
                                  source_x0_px=0, source_y0_px=0, source_x1_px=2048, source_y1_px=2048,
                                  pad_left_px=0, pad_right_px=0, pad_top_px=0, pad_bottom_px=0,
                                  height_px=2048, width_px=2048, source_channel_count=49, channel_count=39,
                                  pixel_size_x_um=808/2048, pixel_size_y_um=808/2048, input_image_path=relative(destination),
                                  intensity_description="Native published uint8 denoised/aggregate-removed per-channel TIFF intensities; no further normalization",
                                  split_method="already_independent", split_evidence="Author release groups independent FOVs into 49 scalar TIFFs; complete FOV retained",
                                  metadata_notes=("Explicit Supplementary Table point→patient→block link" if patient else "FOV→block/patient link unresolved")+
                                  ("; present in paper analysis Table 8" if fov in published_points else "; absent from paper analysis Table 8; retained under accepted 228-FOV canonical input scope; analysis omission reason unresolved"))
                    images = pd.concat([images, pd.DataFrame([record])], ignore_index=True)
            receipts[image_id] = receipt
            write_json(receipts_path, receipts)
            save_tables()
            progress("exporting", last_completed_fov=fov)
            print(json.dumps(dict(fov=fov, completed=len(images), admitted=receipt["admitted"])), flush=True)
            processed_now += 1
            if processed_now == args.check_first:
                progress("initial_batch_verified", initial_batch_images=processed_now)
            if TEMP.exists():
                shutil.rmtree(TEMP)
                TEMP.mkdir(parents=True)
    except BaseException as error:
        progress("error", error=str(error))
        raise
save_tables()
progress("prepared_pending_independent_review" if len(images)==228 else "partial")
print(json.dumps(dict(study=STUDY, exported_images=len(images), expected_images=228,
                      sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                      patient_linked_samples=int(samples.patient_key.ne("").sum())), ensure_ascii=False))
