"""Prepare the 78 locally accepted WUSTL IMC ROIs at published native intensity.

Inputs: 57 verified IDC DICOM plane sets, 21 verified original OME-TIFF ROIs,
S1, HTAN portal specimen/participant records, and paper supplementary tables.
Outputs: square CYX input.tiff and complete image/sample/panel/field tables.
Run metadata/verify with shared formal read lease; run export with shared formal write lease.
Run public-metadata-refresh to publish only the official Table 1 sample fields.
The additional 21 original scans belong to the admitted source cohort; their
publication attribution remains unknown. No intensity conversion is performed.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import openpyxl
import pandas as pd
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.preparation.io_coordination import formal_publish_lease

STUDY = "CuiZhou2022_IMC_PDAC"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
RAW = NAS / "raw/public" / STUDY
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0026_prepare_0"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
OLD_PROOF = META / "native_verification.json"
PORTAL = META / "source_metadata" / "htan_wustl_source_metadata.json"
SUPPLEMENT = META / "source_metadata" / "41588_2022_1157_MOESM3_ESM.xlsx"
TABLE1_SHEET = "Table1_Clinical_data"
TABLE1_TIME_FIELD = "Time_from_diagnosis_to_surgery_(weeks)"
PREPARER = "codex_cli:0026_prepare_0"
PAPER_URL = "https://www.nature.com/articles/s41588-022-01157-1"
SUPPLEMENT_URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41588-022-01157-1/MediaObjects/41588_2022_1157_MOESM3_ESM.xlsx"
PORTAL_URL = "https://github.com/ncihtan/htan-portal/blob/master/public/processed_syn_data.json.gz"
REQUESTS = [
    "将本study单帧未压缩DICOM plane reader评估迁入共享io/readers；当前仅在本study脚本适配，按单通道缓存并保留uint16。",
    "独立复核本批78个image及27份biospecimen/9位患者的来源连接；整理者未签署D01独立验收。",
    "source_description.json的旧57图范围说明早于已批准21图扩展；请在主控来源说明中保留两批归属差别，新增21图不归因于原论文。",
    "核实source_identity_conflicts.json中7个ROI的来源PA/PB及H1/H2标本标签冲突；当前sample_id留空，未按文件名推断更正。",
]
GAPS = [
    "新增21个原生TIFF的论文DOI归属未知；仅按已准入HTAN来源队列交付。",
    "HTAN OME PhysicalSizeX/Y及IDC PixelSpacing存在转换异常；采用逐图官方portal 1 µm字段并保留冲突证据。",
    "HTAN原TIFF的OME Type/portal PixelType写float，但实际TIFF页与原生解码为uint16；按实际存储dtype保留，未转换为float。",
    "7个IDC ROI的portal SizeX/SizeY与实际源图不同；DICOM原始文件名/采集ID/parent均明确，采用实际DICOM尺寸并逐项记录。",
    "7个ROI的原始采集标本标签与HTAN parent标本别名冲突，sample_id留空；其患者身份明确，图像继续保留。",
    "年龄、绝对时间及生存终点缺少可直接采用的统一时点语义，相应基础字段留空；公开临床原值保存在本study缓存。",
    "独立reviewer及D01验收待主控安排。",
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
        study=STUDY, stage=stage, expected_images=len(candidates), completed_images=len(images),
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        unique_patients=int(samples.patient_key.nunique()), gaps=GAPS,
        controller_requests=REQUESTS, updated_at=datetime.now(timezone.utc).isoformat(), **extra))


def source_binding(candidate: dict) -> list[dict]:
    return [dict(path=p, size=(NAS / p).stat().st_size, mtime_ns=(NAS / p).stat().st_mtime_ns)
            for p in candidate["source_files"]]


def source_table1_field(source_field: str) -> str:
    slug = source_field.replace("%", "pct")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", slug).strip("_").lower()
    return f"source_table1_{slug}"


def attach_table1_clinical(samples: pd.DataFrame, fields: pd.DataFrame,
                           table1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_keys = table1["Patient_ID"].dropna().map(str)
    if source_keys.duplicated().any():
        raise ValueError("Table1_Clinical_data.Patient_ID must be unique")
    clinical = {str(row["Patient_ID"]): row for row in table1.to_dict(orient="records")
                if pd.notna(row["Patient_ID"])}
    sample_keys = [None if pd.isna(value) or str(value) == "" else str(value)
                   for value in samples["source_subject_id"]]
    if any(key is None or key not in clinical for key in sample_keys):
        raise ValueError("Every current sample must resolve to exactly one Table1 Patient_ID")

    source_fields = [name for name in table1.columns if name != "Patient_ID"]
    extension_targets = {name: source_table1_field(name) for name in source_fields}
    if len(set(extension_targets.values())) != len(extension_targets):
        raise ValueError("Table1 source fields collide after extension-field normalization")

    def raw_text(value: object) -> str | None:
        return None if pd.isna(value) else str(value)

    for source_field, target_field in extension_targets.items():
        values = [raw_text(clinical[key][source_field]) for key in sample_keys]
        if target_field in samples:
            for current, expected in zip(samples[target_field], values):
                if pd.notna(current) and str(current) != "" and str(current) != str(expected):
                    raise ValueError(f"Existing {target_field} differs from Table1 source")
        samples[target_field] = values

    day_values = []
    for key in sample_keys:
        weeks = clinical[key][TABLE1_TIME_FIELD]
        if pd.isna(weeks):
            raise ValueError("Table1 diagnosis-to-surgery interval is missing for a current sample")
        days = Decimal(str(weeks)) * Decimal(7)
        if days != days.to_integral_value():
            raise ValueError("Table1 diagnosis-to-surgery interval does not resolve to whole days")
        day_values.append(int(days))
    if not samples["collection_method"].eq("resection").all():
        raise ValueError("Table1 diagnosis-to-surgery interval requires surgical-resection samples")
    for target_field, values in (("index_event", ["diagnosis"] * len(samples)),
                                 ("collection_day_from_index", day_values)):
        if target_field in samples:
            for current, expected in zip(samples[target_field], values):
                if pd.notna(current) and str(current) != "" and str(current) != str(expected):
                    raise ValueError(f"Existing {target_field} differs from Table1 source")
        samples[target_field] = values

    source_file = relative(SUPPLEMENT)
    portal_source = relative(PORTAL)
    portal_mask = fields.source_file.str.endswith("htan_wustl_source_metadata.json")
    fields.loc[portal_mask, "source_file"] = portal_source
    table1_mask = fields.source_sheet_or_table.eq(TABLE1_SHEET)
    fields.loc[table1_mask, "source_file"] = source_file
    table1_join = "sample_metadata.source_subject_id = Table1_Clinical_data.Patient_ID"
    sample_join = "sample_metadata.sample_id = biospecimenByBiospecimenID.HTANBiospecimenID"
    mapping_rules = [
        ("biospecimenByBiospecimenID", "HTANBiospecimenID", "sample_id", None,
         "HTAN biospecimen identifier精确复制为sample键。", sample_join, "不适用。",
         "来源标本键未命中或重复时停止写入。"),
        ("biospecimenByBiospecimenID", "HTANParentID", "patient_key", None,
         "由HTAN biospecimen记录读取显式HTANParentID作为患者键。", sample_join,
         "不适用；该字段是患者标识。", "不按sample名称推断患者。"),
        ("biospecimenByBiospecimenID", "SourceHTANBiospecimenID", "source_sample_id", None,
         "经HTANBiospecimenID连接后保留来源biospecimen别名原值。", sample_join,
         "不适用。", "别名冲突保留原值并另记身份冲突。"),
        ("biospecimenByBiospecimenID", "HTANBiospecimenID", "biospecimen_group_id", None,
         "同一HTAN biospecimen的ROI共用该组键。", sample_join, "不适用。",
         "不按ROI名称重建标本组。"),
        ("biospecimenByBiospecimenID", "BiospecimenType", "sample_type", None,
         "经HTAN biospecimen键连接后保留BiospecimenType原值。", sample_join, "不适用。",
         "保留HTAN来源类别原值。"),
        ("biospecimenByBiospecimenID", "AcquisitionMethodType", "collection_method",
         {"Surgical Resection": "resection"}, "Surgical Resection精确映射为resection。",
         sample_join, "不适用；采集方式字段无时间含义。", "其他来源类别不作推断。"),
        ("biospecimenByBiospecimenID", "TumorTissueType", "tissue_type",
         {"Primary": "tumor"}, "TumorTissueType的Primary精确映射为tumor；其他来源值不作推断。",
         sample_join, "不适用；组织类别字段无时间含义。", "其他来源类别不作推断。"),
        ("biospecimenByBiospecimenID", "TumorTissueType", "lesion_type",
         {"Primary": "primary"}, "TumorTissueType的Primary精确映射为primary；其他来源值不作推断。",
         sample_join, "不适用；病灶类别字段无时间含义。", "其他来源类别不作推断。"),
        ("biospecimenByBiospecimenID", "TimepointLabel", "timepoint_label", {
            "Baseline": "baseline", "baseline": "baseline",
            "End of Treatment": "post_treatment", "End of treatment": "post_treatment",
        }, "HTAN Baseline映射为baseline；End of Treatment大小写变体映射为post_treatment。",
         sample_join, "仅保留HTAN时间点标签；不转换为诊断或治疗间隔。",
         "未列出的时间点类别不作推断。"),
        ("files", "ParticipantID", "source_subject_id", None,
         "通过files.biospecimenIds精确连接sample，再保留ParticipantID原值。",
         "sample_metadata.sample_id = files.biospecimenIds; source_subject_id = files.ParticipantID",
         "不适用；该字段是来源患者别名。", "每个biospecimen必须解析到唯一ParticipantID。"),
        ("demographicsByParticipantID", "Gender", "sex_reported", None,
         "按patient_key连接demographicsByParticipantID并保留Gender原值。",
         "sample_metadata.patient_key = demographicsByParticipantID.HTANParticipantID",
         "不适用；保留HTAN该版本的来源性别类别。", "不从其他来源重编码。"),
        ("demographicsByParticipantID", "VitalStatus", "vital_status",
         {"Dead": "deceased", "Alive": "alive", "unknown": "unknown"},
         "按patient_key连接HTAN VitalStatus；Dead/Alive/unknown按词表精确映射。",
         "sample_metadata.patient_key = demographicsByParticipantID.HTANParticipantID",
         "保留HTAN后续公开版本状态；不推断采集或随访日期。",
         "Table1的Vital_status保存在独立source_table1扩展字段，不合并来源版本。"),
        ("diagnosisByParticipantID", "SiteofResectionorBiopsy", "anatomical_site", None,
         "按patient_key连接diagnosisByParticipantID并保留来源原值。",
         "sample_metadata.patient_key = diagnosisByParticipantID.HTANParticipantID",
         "不适用；解剖部位字段无时间含义。", "不与Table1患者级Location作无键合并。"),
        ("diagnosisByParticipantID", "PrimaryDiagnosis", "primary_diagnosis", None,
         "按patient_key连接diagnosisByParticipantID并保留来源原值。",
         "sample_metadata.patient_key = diagnosisByParticipantID.HTANParticipantID",
         "不适用；诊断文本不提供具体日期。", "不推断诊断日期或分期时点。"),
        ("diagnosisByParticipantID", "TissueorOrganofOrigin", "primary_site", None,
         "按patient_key连接diagnosisByParticipantID并保留来源原值。",
         "sample_metadata.patient_key = diagnosisByParticipantID.HTANParticipantID",
         "不适用；原发部位字段无时间含义。", "不与其他部位字段近似合并。"),
        (TABLE1_SHEET, "Treatment", "treatment_status_at_collection", None,
         "按source_subject_id与Patient_ID精确连接并保留Treatment原值。", table1_join,
         "保留原表取样语境；不推断index regimen。", "原始值同时保留于source_table1_treatment。"),
        (TABLE1_SHEET, "Grade", "grade", None,
         "按source_subject_id与Patient_ID精确连接并将Grade原值转为文本。", table1_join,
         "不把无明确时点的Grade转换为index stage。", "原始值同时保留于source_table1_grade。"),
        (TABLE1_SHEET, "Smoker", "smoking_status", {"Former": "former"},
         "仅将Smoker的Former精确映射为former；Yes/No缺少ever/current定义，标准字段留空。",
         table1_join, "来源未说明状态时点，不视为index时吸烟状态。",
         "Yes/No原值保留于source_table1_smoker，标准字段留空。"),
    ]
    for sheet, source_field, target_field, value_mapping, rule, join_keys, time_rule, conflict_rule in mapping_rules:
        mask = (fields.source_sheet_or_table.eq(sheet)
                & fields.source_field.eq(source_field)
                & fields.target_field.eq(target_field))
        if mask.sum() != 1:
            raise ValueError(f"Expected exactly one source mapping for {target_field}")
        fields.loc[mask, "value_mapping"] = (
            json.dumps(value_mapping, ensure_ascii=False) if value_mapping else ""
        )
        fields.loc[mask, "transformation_rule"] = rule
        fields.loc[mask, "join_keys"] = join_keys
        fields.loc[mask, "time_origin_rule"] = time_rule
        fields.loc[mask, "conflict_rule"] = conflict_rule
    fields = fields.loc[~(table1_mask & fields.target_field.isin(
        set(extension_targets.values()) | {"source_subject_id"}))].copy()
    fields = fields.loc[~fields.target_field.isin({"index_event", "collection_day_from_index"})].copy()

    join = table1_join
    mappings = [dict(
        study=STUDY, source_file=source_file, source_sheet_or_table=TABLE1_SHEET,
        source_field="Patient_ID", target_table="sample_metadata", target_field="source_subject_id",
        join_keys=join, value_mapping="",
        transformation_rule="按公开Patient_ID与现有source_subject_id精确等值连接；一位患者可对应多个HTAN标本行。",
        time_origin_rule="不适用；本字段为来源患者键。",
        conflict_rule="不进行近似匹配；未命中或重复来源键时停止写入。",
        notes=json.dumps({"data_type": "string", "definition": "Table1 patient key used by the explicit ParticipantID crosswalk."}, ensure_ascii=False)
    )]
    for source_field, target_field in extension_targets.items():
        unit_match = re.search(r"\(([^()]*)\)$", source_field)
        unit = "%" if source_field.endswith("_%") else (unit_match.group(1) if unit_match else None)
        time_rule = "保留原表字段，不推断采样或诊断时点。"
        conflict_rule = "只按source_subject_id与Patient_ID精确连接；空单元格保持空白。"
        if source_field == TABLE1_TIME_FIELD:
            time_rule = "原字段明示诊断至手术的间隔，单位为周；标准采样日另按7天/周换算。"
        elif source_field == "Survival_post_surgery_(weeks)":
            time_rule = "原字段明示以手术为起点、单位为周；事件与删失定义不完整，不映射为标准生存终点。"
        elif source_field == "Vital_status":
            conflict_rule = "保留2022 Table1原值；后续HTAN版本的vital_status单独保留，不合并来源版本。"
        mappings.append(dict(
            study=STUDY, source_file=source_file, source_sheet_or_table=TABLE1_SHEET,
            source_field=source_field, target_table="sample_metadata", target_field=target_field,
            join_keys=join, value_mapping="",
            transformation_rule="保留官方工作簿单元格值的文本表示；不作类别重编码或单位换算。",
            time_origin_rule=time_rule, conflict_rule=conflict_rule,
            notes=json.dumps({"data_type": "string", "definition": f"Raw Table1_Clinical_data value for {source_field}.", "unit": unit}, ensure_ascii=False)
        ))
    for target_field, rule, time_rule in (
        ("index_event", "Table1字段明示诊断至手术间隔，index_event填为diagnosis；HTAN样本记录确认collection_method=resection。",
         "index_event来自Time_from_diagnosis_to_surgery_(weeks)的明确起点。"),
        ("collection_day_from_index", "将Table1诊断至手术间隔的周数乘以7，转换为整数天；仅应用于精确Patient_ID匹配且采集方法为resection的当前样本。",
         "起点为diagnosis，终点为surgical resection；原始周数另存source_table1_time_from_diagnosis_to_surgery_weeks。"),
    ):
        mappings.append(dict(
            study=STUDY, source_file=source_file, source_sheet_or_table=TABLE1_SHEET,
            source_field=TABLE1_TIME_FIELD, target_table="sample_metadata", target_field=target_field,
            join_keys=join, value_mapping="", transformation_rule=rule,
            time_origin_rule=time_rule,
            conflict_rule="只按source_subject_id与Patient_ID精确连接；仅对明确的手术标本填写。",
            notes=""
        ))
    fields = pd.concat([fields, pd.DataFrame(mappings, columns=fields.columns)], ignore_index=True)
    return samples, fields


def dicom_parent(dataset) -> str:
    parents = {str(item.TextValue)
               for specimen in dataset.SpecimenDescriptionSequence
               for step in specimen.SpecimenPreparationSequence
               for item in step.SpecimenPreparationStepContentItemSequence
               if item.ConceptNameCodeSequence[0].CodeMeaning == "Parent Specimen Identifier"}
    if len(parents) != 1:
        raise ValueError(f"DICOM specimen has ambiguous biological parents: {parents}")
    return parents.pop()


def source_identity_audit(portal: dict) -> dict[str, dict]:
    """Record contradictions between explicit parent aliases and physical labels.

    The source alias contains PA/PB or S1H1/S1H2. The original acquisition's
    filename carries a conflicting physical label for seven ROIs. These rules
    detect the conflict; they do not choose a corrected biological parent.
    DICOM original filename (0009,1001) independently preserves that acquisition
    label. Paper Table2/5 corroborate the distinct biological sample labels.
    """
    conflicts = []
    for source in portal["files"]:
        bio = portal["biospecimenByBiospecimenID"][source["biospecimenIds"][0]]
        alias = bio["SourceHTANBiospecimenID"]
        name = Path(source["Filename"]).name
        reason = None
        if "-PBA1,PBA4" in alias and "_PA_A1_A4_" in name:
            reason = "HTAN parent alias denotes PB but original acquisition filename denotes PA"
        if "-S1H1L1,S1H1L4" in alias and "S1H2_L1_L4_" in name:
            reason = "HTAN parent alias denotes H1 but original acquisition filename denotes H2"
        if "-S1H2L1,S1H2L4" in alias and "S1H1_L1_L4_" in name:
            reason = "HTAN parent alias denotes H2 but original acquisition filename denotes H1"
        if reason:
            conflicts.append(dict(image_id=source["DataFileID"], declared_biospecimen_id=bio["HTANBiospecimenID"],
                                  source_biospecimen_alias=alias, source_filename=source["Filename"],
                                  patient_key=bio["HTANParentID"], reason=reason,
                                  resolution="unresolved; image sample_id empty; do not infer a corrected parent",
                                  evidence=[relative(PORTAL), relative(SUPPLEMENT) + ":Table2_Metadata;Table5_Pathology_review",
                                            "IDC original filename (0009,1001), SpecimenIdentifier and parent"] ))
    write_json(CACHE / "source_identity_conflicts.json", dict(
        scope="source declared biological parent versus acquisition physical sample label",
        affected_images=len(conflicts), records=conflicts))
    return {r["image_id"]: r for r in conflicts}


def read_dicom(path: Path, **kwargs):
    import pydicom

    return pydicom.dcmread(path, **kwargs)


class DicomPlanesReader:
    """Study-local adapter for accepted single-frame, uncompressed IDC planes.

    Keep only the current scalar channel decoded (largest source plane <4 MiB).
    Preserve stored uint16 values without applying rescale or display transforms.
    Plane order is the numeric original optical-path ID, also recorded in panel.
    This layout is absent from the shared reader and is reported for migration.
    """
    def __init__(self, candidate: dict):
        self.paths = [NAS / p for p in candidate["source_files"]]
        self.shape = (len(self.paths), candidate["height"], candidate["width"])
        self.dtype = np.dtype("uint16")
        self.source = SimpleNamespace(pixel_size_um=(1.0, 1.0))
        self.channel = None
        self.plane = None

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        result = []
        for channel in channels:
            if self.channel != channel:
                dataset = read_dicom(self.paths[channel])
                self.plane = dataset.pixel_array
                if self.plane.shape != self.shape[-2:] or self.plane.dtype != self.dtype:
                    raise ValueError(f"Unexpected IDC scalar plane: {self.paths[channel]}")
                self.channel = channel
            result.append(self.plane[y0:y1, x0:x1].copy())
        return np.stack(result)

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


@contextmanager
def source_reader(candidate: dict):
    if candidate["source_format"] == "dicom_planes":
        yield DicomPlanesReader(candidate)
    else:
        from cellweft.data.io.readers import open_source

        spec = SourceSpec(candidate["image_id"], candidate["panel_id"], "tiff",
                          NAS / candidate["source_files"][0], tuple(range(len(candidate["names"]))),
                          pixel_size_um=(1.0, 1.0))
        with open_source(spec) as reader:
            yield reader


def load_sources() -> tuple[list[dict], pd.DataFrame, dict]:
    """Reconcile all accepted payloads with prior proofs and live source headers."""
    import tifffile
    from cellweft.data.preparation.study_input import panel_from_s1

    portal = json.loads(PORTAL.read_text())
    files = {f["synapseId"]: f for f in portal["files"]}
    manifest = json.loads((META / "source_manifest.json").read_text())
    proofs = {p["acquisition_id"]: p for p in json.loads(OLD_PROOF.read_text())}
    mapping = pd.read_csv(META / "channel_mapping.csv", dtype=str, keep_default_na=False)
    extension = json.loads((META / "htan_wustl_extension_ordered_markers.json").read_text())["files"]
    extension_scope = json.loads((META / "htan_original_cohort_extension.json").read_text())
    extension_proofs = {json.loads(Path(p).read_text())["physical_identity"]["independent_acquisition_id"]:
                        json.loads(Path(p).read_text()) for p in extension_scope["new_scope"]["proofs"]}
    records = []
    live_paths = set()
    for entry in manifest:
        image_id = entry["data_file_id"]
        proof = proofs[image_id]
        assert proof["complete"]
        annotation = files[entry["synapse_id"]]
        assert annotation["DataFileID"] == image_id and annotation["Filename"] == entry["source_filename"]
        channels = sorted(proof["channels"], key=lambda x: int(x["optical_path_id"]))
        assert len(channels) == len(entry["objects"]) == proof["channel_count"]
        expected = {x["key"]: x["size"] for x in entry["objects"]}
        lookup = mapping[mapping.acquisition_id == image_id].set_index("image_path")
        for channel in channels:
            path = NAS / channel["path"]
            live_paths.add(path)
            assert path.stat().st_size == channel["local_size"] == channel["source_size"]
            assert path.stat().st_size == expected[str(metadata_source_relative(path, RAW / 'idc_portal_release'))]
            row = lookup.loc[channel["path"]]
            header = read_dicom(path, stop_before_pixels=True)
            optical = header.OpticalPathSequence[0]
            assert str(optical.OpticalPathIdentifier) == channel["optical_path_id"] == row.optical_path_id
            assert str(optical.OpticalPathDescription) == channel["marker"] == row.marker
            assert str(header.SpecimenDescriptionSequence[0].SpecimenIdentifier) == image_id
            assert Path(str(header[0x00091001].value)).name == Path(entry["source_filename"]).name
            assert dicom_parent(header) == annotation["biospecimenIds"][0]
            assert str(header.PatientID) == annotation["diagnosisIds"][0]
            assert [int(header.Rows), int(header.Columns)] == proof["shape"]
            assert int(header.NumberOfFrames) == 1 and int(header.SamplesPerPixel) == 1
            assert int(header.BitsAllocated) == 16 and int(header.PixelRepresentation) == 0
            assert str(header.file_meta.TransferSyntaxUID) == "1.2.840.10008.1.2.1"
        records.append(dict(image_id=image_id, source_files=[x["path"] for x in channels],
                            source_format="dicom_planes", source_axes="YX", dtype="uint16",
                            height=proof["shape"][0], width=proof["shape"][1],
                            names=[x["marker"] for x in channels],
                            optical_ids=[int(x["optical_path_id"]) for x in channels],
                            source_filename=entry["source_filename"], synapse_id=entry["synapse_id"],
                            source_revision="IDC portal release; HTAN batch_1_03302022_v2",
                            publication_scope="paper-linked", native_proof=relative(OLD_PROOF)))
    assert live_paths == set((RAW / "idc_portal_release").rglob("*.dcm"))
    for entry in extension:
        image_id = entry["htan_data_file_id"]
        proof = extension_proofs[image_id]
        path = RAW / "htan_wustl_batch1_extension" / Path(entry["source_filename"]).name
        stat = path.stat()
        assert proof["complete"] and stat.st_size == entry["source_bytes"] == proof["source_stat"]["size"]
        assert stat.st_mtime_ns == proof["source_stat"]["mtime_ns"]
        annotation = files[entry["synapse_id"]]
        assert annotation["DataFileID"] == image_id and annotation["Filename"] == entry["source_filename"]
        with tifffile.TiffFile(path) as tif:
            xml = ET.fromstring(tif.ome_metadata)
            names = [c.attrib["Name"] for c in xml.findall(".//{*}Channel")]
            assert names == proof["ordered_channels"] == [x["name"] for x in entry["ordered_channels"]]
            assert tif.series[0].axes == "CYX"
            dtype = str(tif.series[0].dtype)
            count, height, width = tif.series[0].shape
            assert count == len(names) == len(tif.pages)
            assert all(page.dtype == np.dtype(dtype) and page.shape == (height, width) for page in tif.pages)
        records.append(dict(image_id=image_id, source_files=[relative(path)], source_format="ome_tiff",
                            source_axes="CYX", dtype=dtype, height=height, width=width, names=names,
                            optical_ids=[int(x["id"].split(":")[-1]) for x in entry["ordered_channels"]],
                            source_filename=entry["source_filename"], synapse_id=entry["synapse_id"],
                            source_revision="HTAN batch_1_03302022_v2; source revision " + entry["source_revision"],
                            publication_scope="admitted source-cohort extension; publication DOI unknown",
                            native_proof=relative(Path(next(p for p in extension_scope["new_scope"]["proofs"]
                                                           if Path(p).stem.endswith(image_id))))))
    assert {NAS / r["source_files"][0] for r in records if r["source_format"] == "ome_tiff"} == set(
        (RAW / "htan_wustl_batch1_extension").glob("*.tiff"))
    panels = {}
    for record in records:
        annotation = files[record["synapse_id"]]
        assert len(annotation["biospecimenIds"]) == len(annotation["diagnosisIds"]) == 1
        assert annotation["PhysicalSizeX"] == annotation["PhysicalSizeY"] == 1
        assert annotation["PhysicalSizeXUnit"] == annotation["PhysicalSizeYUnit"] == "µm"
        record["portal_geometry"] = [annotation["SizeY"], annotation["SizeX"]]
        record["portal_geometry_agrees"] = record["portal_geometry"] == [record["height"], record["width"]]
        # IDC original-file private tag, acquisition ID and parent ID agree.
        # Seven portal SizeX/SizeY values are permuted; geometry follows pixels.
        if record["source_format"] == "ome_tiff":
            assert record["portal_geometry_agrees"]
        assert annotation["SizeC"] == len(record["names"])
        record.update(sample_id=annotation["biospecimenIds"][0],
                      declared_sample_id=annotation["biospecimenIds"][0], patient_key=annotation["diagnosisIds"][0])
        signature = tuple(zip(record["optical_ids"], record["names"]))
        if signature not in panels:
            panel_id = f"wustl_imc_{len(record['names'])}ch"
            panel = panel_from_s1(STUDY, panel_id, record["names"], S1)
            panel["source_optical_path_id"] = record["optical_ids"]
            panel.loc[panel.nuclear & panel.source_name.eq("DNA"), "keep_reason"] = (
                "S1选定DNA核输入；原生191Ir/193Ir测量均保留为segmentation-only"
            )
            panel.loc[panel.nuclear & panel.source_name.eq("Histone H3"), "keep_reason"] = (
                "S1 selected DNA and Histone H3 nuclear inputs; native DNA remains segmentation-only"
            )
            panels[signature] = panel
        record["panel_id"] = panels[signature].panel_id.iloc[0]
    # The explicit HTAN parent and original physical sample label conflict for
    # 7 ROIs (PA/PB or H1/H2). Keep the declaration as evidence, not as a confirmed
    # sample assignment. The cached audit lists source IDs, never patient lists in code.
    conflicts = source_identity_audit(portal)
    for record in records:
        record["sample_identity_conflict"] = conflicts.get(record["image_id"])
        if record["sample_identity_conflict"]:
            assert record["declared_sample_id"] == record["sample_identity_conflict"]["declared_biospecimen_id"]
            record["sample_id"] = None
    assert len({r["image_id"] for r in records}) == len(records) == 78
    return sorted(records, key=lambda x: x["image_id"]), pd.concat(panels.values(), ignore_index=True), portal


def sample_tables(portal: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use explicit HTAN parent IDs, then the paper's exact participant key.

    Do not derive patients or specimens from ROI names. Clinical age and dates
    remain raw in the cached sources where time semantics differ. Treatment is
    copied as collection-context annotation, not interpreted as an index regimen.
    """
    workbook = openpyxl.load_workbook(SUPPLEMENT, read_only=True, data_only=True)
    values = list(workbook["Table1_Clinical_data"].values)
    table1 = pd.DataFrame(values[1:], columns=values[0])
    clinical = {str(row["Patient_ID"]): row for row in table1.to_dict(orient="records")
                if pd.notna(row["Patient_ID"])}
    rows = []
    for sample_id in sorted({r["declared_sample_id"] for r in candidates}):
        bio = portal["biospecimenByBiospecimenID"][sample_id]
        images = [r for r in candidates if r["declared_sample_id"] == sample_id]
        patient = bio["HTANParentID"]
        assert {r["patient_key"] for r in images} == {patient}
        participants = {f["ParticipantID"] for f in portal["files"] if f["biospecimenIds"] == [sample_id]}
        assert len(participants) == 1
        subject = participants.pop()
        source = clinical[subject]
        demo = portal["demographicsByParticipantID"][patient]
        diagnosis = portal["diagnosisByParticipantID"][patient]
        assert source["Sex"] == demo["Gender"]
        row = dict.fromkeys(pd.read_csv(TEMPLATES / "sample_metadata.csv").columns, None)
        row.update(study=STUDY, sample_id=sample_id, patient_key=patient, source_subject_id=subject,
                   cohort_id="HTAN_WUSTL_IMC_batch_1_03302022_v2", center_id="HTAN WUSTL",
                   source_sample_id=bio["SourceHTANBiospecimenID"], biospecimen_group_id=sample_id,
                   sample_type=bio["BiospecimenType"],
                   anatomical_site=diagnosis["SiteofResectionorBiopsy"],
                   primary_diagnosis=diagnosis["PrimaryDiagnosis"], primary_site=diagnosis["TissueorOrganofOrigin"],
                   sex_reported=demo["Gender"], treatment_status_at_collection=source["Treatment"],
                   grade=str(source["Grade"]), collection_method="resection",
                   timepoint_label={"baseline": "baseline", "end of treatment": "post_treatment"}[bio["TimepointLabel"].lower()],
                   metadata_notes="HTANParentBiospecimenID→HTANBiospecimenID→HTANParentID；同一标本多ROI共用sample。"
                   "PreservationMethod=Frozen/StorageMethod=Frozen at -80C仅保留原值于缓存，未推断为fresh_frozen。"
                   "Supplement Age缺精确时点；Smoker Yes/No缺ever/current定义；HTAN时间字段起点混合，均未填入对应标准字段。"
                   "HTAN clinical为后来公开版本，Supplement为2022版；不合并生存终点。")
        if any(r["sample_identity_conflict"] for r in images):
            row["metadata_notes"] += " 此标本来源parent别名与采集PA/PB或H1/H2标签冲突，关联ROI暂未填sample_id；见source_identity_conflicts.json。"
        assert bio["AcquisitionMethodType"] == "Surgical Resection"
        if bio["TumorTissueType"] == "Primary":
            row.update(tissue_type="tumor", lesion_type="primary")
        if source["Smoker"] == "Former":
            row["smoking_status"] = "former"
        if demo["VitalStatus"] in {"Dead", "Alive"}:
            row["vital_status"] = {"Dead": "deceased", "Alive": "alive"}[demo["VitalStatus"]]
        elif demo["VitalStatus"] == "unknown":
            row["vital_status"] = "unknown"
        rows.append(row)
    workbook.close()
    samples = pd.DataFrame(rows)
    fields = []

    def field(source_file: Path, sheet: str, source: str, target: str, rule: str,
              target_table: str = "sample_metadata", notes: str = "", mapping: str = "") -> None:
        fields.append(dict(study=STUDY, source_file=relative(source_file), source_sheet_or_table=sheet,
                           source_field=source, target_table=target_table, target_field=target,
                           join_keys="synapseId/HTANDataFileID→HTANParentBiospecimenID→HTANBiospecimenID→HTANParentID; ParticipantID→Patient_ID",
                           value_mapping=mapping, transformation_rule=rule, time_origin_rule="未知时点不转换为index或生存时间",
                           conflict_rule="身份以HTAN显式连接及DICOM parent交叉核对；版本临床来源分别保留",
                           notes=notes))

    for source, target in [("HTANParentBiospecimenID", "sample_id")]:
        field(PORTAL, "files", source, target, "逐文件显式标本ID；7个标签冲突ROI留空，不按ROI或文件名猜测", "image_metadata")
    biospecimen_fields = [
        ("HTANBiospecimenID", "sample_id", "", "HTAN biospecimen identifier精确复制为sample键。"),
        ("HTANParentID", "patient_key", "", "HTAN parent participant identifier精确复制为patient键。"),
        ("SourceHTANBiospecimenID", "source_sample_id", "", "保留来源biospecimen别名原值。"),
        ("HTANBiospecimenID", "biospecimen_group_id", "", "同一HTAN biospecimen的ROI共用该组键。"),
        ("BiospecimenType", "sample_type", "", "保留HTAN biospecimen类型原值。"),
        ("AcquisitionMethodType", "collection_method", {"Surgical Resection": "resection"},
         "Surgical Resection精确映射为resection。"),
        ("TumorTissueType", "tissue_type", {"Primary": "tumor"},
         "Primary精确映射为tumor；其他来源值不作推断。"),
        ("TumorTissueType", "lesion_type", {"Primary": "primary"},
         "Primary精确映射为primary；其他来源值不作推断。"),
        ("TimepointLabel", "timepoint_label", {
            "Baseline": "baseline", "End of Treatment": "post_treatment",
            "End of treatment": "post_treatment",
        }, "Baseline及End of Treatment按精确来源类别映射。"),
    ]
    for source, target, value_mapping, rule in biospecimen_fields:
        mapping = json.dumps(value_mapping, ensure_ascii=False) if value_mapping else ""
        field(PORTAL, "biospecimenByBiospecimenID", source, target, rule, mapping=mapping)
    field(PORTAL, "files", "ParticipantID", "source_subject_id", "公开受试者别名原值；以明确ParticipantID连接补充临床表")
    for source, target, table in [("Gender", "sex_reported", "demographicsByParticipantID"),
                                   ("VitalStatus", "vital_status", "demographicsByParticipantID"),
                                   ("SiteofResectionorBiopsy", "anatomical_site", "diagnosisByParticipantID"),
                                   ("PrimaryDiagnosis", "primary_diagnosis", "diagnosisByParticipantID"),
                                   ("TissueorOrganofOrigin", "primary_site", "diagnosisByParticipantID")]:
        mapping = {"Dead": "deceased", "Alive": "alive"} if target == "vital_status" else ""
        field(PORTAL, table, source, target, "经patient_key精确连接；除vital_status外保留来源原值",
              mapping=json.dumps(mapping, ensure_ascii=False) if mapping else "")
    for source, target, rule in [("Treatment", "treatment_status_at_collection", "保留原值Naive/FOLFIRINOX/Gem_abrax，不推断index regimen"),
                                 ("Grade", "grade", "临床病理grade原值转字符串；不填入未知时间的index stage"),
                                 ("Smoker", "smoking_status", "仅Former→former；Yes/No缺ever/current定义故留空")]:
        mapping = json.dumps({"Former": "former"}, ensure_ascii=False) if target == "smoking_status" else ""
        field(SUPPLEMENT, "Table1_Clinical_data", source, target, rule, mapping=mapping)
    for target in ("cohort_id", "center_id", "study"):
        field(PORTAL, "files", "atlas_name/assayName/Filename", target, "由本批明确HTAN WUSTL IMC batch_1来源固定study/center/cohort标识")
    field(PORTAL, "files", "PhysicalSizeX/PhysicalSizeY", "pixel_size_x_um,pixel_size_y_um",
          "采用逐图portal 1 µm；原OME PhysicalSize等于ROI宽高、DICOM转换尺度异常不采用", "image_metadata")
    field(META / "channel_mapping.csv", "ordered channels", "optical_path_id/marker/image_path",
          "source_channel_index,output_channel_index", "DICOM按数值optical_path_id排序；原TIFF按OME page/channel顺序；S1逐study+原名连接", "panel_mapping")
    field(S1, "CuiZhou2022_IMC_PDAC", "Original marker/Unified marker/Segmentation channel", "marker_id,role,nuclear,body,keep_channel",
          "All admitted measurements plus the two native DNA and one Histone H3 nuclear inputs are retained; DNA remains segmentation-only and H3 remains downstream plus nuclear; publish each actual panel separately", "panel_mapping")
    field(CACHE / "source_scope.json", "candidates", "height/width/portal_geometry/portal_geometry_agrees",
          "source_x1_px,source_y1_px,height_px,width_px", "采用实际DICOM/OME TIFF像素尺寸；7个portal维度错配以DICOM原始文件名、采集ID、parent及原核验证明校正", "image_metadata")
    field(PORTAL, "files; biospecimenByBiospecimenID", "HTANDataFileID/biospecimenIds/HTANParentID", "metadata_notes",
          "确认71/78 image→25份biospecimen；7个ROI的PA/PB或H1/H2标本标签冲突；全部27份来源biospecimen均有明确patient，共9位patient",
          notes="Supplement Tables 1–5已查阅；Table1临床按显式ParticipantID对接；Table2/3/5取样信息交叉参照，未以名字扩展连接。")
    # Every base field is reviewed: missing values retain a concrete missingness rule.
    dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
    # Explain every base image field, including deliberately empty source fields.
    image_sources = {
        "study": "canonical study from accepted source manifests",
        "image_id": "HTANDataFileID / DICOM SpecimenIdentifier",
        "panel_id": "ordered original optical_path_id + marker signature",
        "image_kind": "independent published IMC ROI",
        "source_image_id": "HTANDataFileID / DICOM SpecimenIdentifier",
        "source_files": "prior native proof channel paths sorted by optical_path_id, or original TIFF path",
        "source_format": "actual DICOM scalar plane set / actual OME-TIFF pages",
        "source_axes": "scalar DICOM YX with channel order in source_files / TIFF CYX",
        "source_dtype": "actual native TIFF page dtype / DICOM stored scalar uint16, not OME Type declaration",
        "source_revision": "accepted IDC release or original HTAN batch and source revision",
        "roi_id": "HTANDataFileID, also identified by original source_filename in metadata_notes",
        "source_x0_px": "0: complete source ROI left boundary",
        "source_y0_px": "0: complete source ROI top boundary",
        "source_x1_px": "actual decoded source width, not stale portal SizeX",
        "source_y1_px": "actual decoded source height, not stale portal SizeY",
        "pad_left_px": "(max(height,width)-width)//2",
        "pad_top_px": "(max(height,width)-height)//2",
        "pad_right_px": "max(height,width)-width-pad_left_px",
        "pad_bottom_px": "max(height,width)-height-pad_top_px",
        "height_px": "max(source height,source width)",
        "width_px": "max(source height,source width)",
        "source_channel_count": "all original channels in actual panel",
        "channel_count": "kept panel rows in output_channel_index order",
        "pixel_size_x_um": "files.PhysicalSizeX=1 µm; OME/DICOM conversion-scale conflict recorded",
        "pixel_size_y_um": "files.PhysicalSizeY=1 µm; OME/DICOM conversion-scale conflict recorded",
        "input_image_path": "processed/<study>/images/<HTANDataFileID>/input.tiff relative to NAS",
        "intensity_description": "stored native intensity; upstream IDC conversion explicitly distinguished",
        "split_method": "already_independent: full original ROI",
        "split_evidence": "HTAN source record + prior native proof + actual acquisition header",
        "metadata_notes": "original filename, publication scope, native geometry and identity contradictions",
    }
    covered_images = {f["target_field"] for f in fields if f["target_table"] == "image_metadata"}
    for spec in dictionary[dictionary.table_name == "image_metadata"].itertuples(index=False):
        if spec.field_name not in covered_images:
            source = image_sources.get(spec.field_name, "not supplied/not applicable for these independent scalar ROIs")
            field(CACHE / "source_scope.json", "candidates + native source headers + export receipts", source,
                  spec.field_name, "按所列来源或明确公式填写；无来源的可选字段保持空值。", "image_metadata")
    covered = {f["target_field"] for f in fields if f["target_table"] == "sample_metadata"}
    for spec in dictionary[dictionary.table_name == "sample_metadata"].itertuples(index=False):
        if spec.field_name not in covered:
            rule = "来源未提供可直接采用的值或时间语义，保留空；不从ROI名称推断临床。"
            if spec.field_name == "metadata_notes":
                rule = "说明显式身份连接、原始Frozen、年龄/吸烟/时间语义及公开版本差异。"
            field(PORTAL, "reviewed source clinical/biospecimen records", spec.field_name, spec.field_name, rule)
    field_mapping = pd.DataFrame(fields, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)
    return attach_table1_clinical(samples, field_mapping, table1)


def header_check(candidate: dict, path: Path, panel: pd.DataFrame) -> None:
    import tifffile

    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    side = max(candidate["height"], candidate["width"])
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (len(kept), side, side)
        assert tif.series[0].dtype == np.dtype(candidate["dtype"])
        xml = ET.fromstring(tif.ome_metadata)
        assert [c.attrib["Name"] for c in xml.findall(".//{*}Channel")] == kept.marker_id.tolist()
        pixels = xml.find(".//{*}Pixels")
        assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0


def image_row(candidate: dict, stats: dict) -> dict:
    row = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
    path = OUTPUT / "images" / candidate["image_id"] / "input.tiff"
    row.update(study=STUDY, image_id=candidate["image_id"], sample_id=candidate["sample_id"],
               panel_id=candidate["panel_id"], image_kind="roi", source_image_id=candidate["image_id"],
               source_files=candidate["source_files"], source_format=candidate["source_format"],
               source_axes=candidate["source_axes"], source_dtype=candidate["dtype"],
               source_revision=candidate["source_revision"], roi_id=candidate["image_id"],
               source_x0_px=0, source_y0_px=0, source_x1_px=candidate["width"], source_y1_px=candidate["height"],
               pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(path),
               intensity_description=("IDC published stored uint16; original IMC float-to-DICOM conversion is upstream; no rescale/normalization here"
                                      if candidate["source_format"] == "dicom_planes" else
                                      "HTAN published level-2 IMC stored uint16 (OME Type/portal PixelType incorrectly declare float); no intensity conversion or normalization here"),
               split_method="already_independent", split_evidence=f"{relative(PORTAL)}:{candidate['synapse_id']}; {candidate['native_proof']}; complete source ROI",
               metadata_notes=f"source_filename={candidate['source_filename']}; publication_scope={candidate['publication_scope']}; "
               "sample来自HTAN显式parent；完整来源ROI，无sample混入证据；1µm来自逐图portal字段，旧OME/DICOM尺度冲突已记录。")
    if not candidate["portal_geometry_agrees"]:
        row["metadata_notes"] += f" portal声明YX={candidate['portal_geometry']}与像素不同，按DICOM实际YX={[candidate['height'], candidate['width']]}保存；DICOM原始文件名/采集ID/parent与标本连接一致。"
    if candidate["sample_identity_conflict"]:
        row["metadata_notes"] += f" 来源声明sample={candidate['declared_sample_id']}，但采集物理标签与标本别名矛盾；sample_id留空，patient={candidate['patient_key']}由明确来源确认；见{relative(CACHE / 'source_identity_conflicts.json')}。"
    for key in ("height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px", "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = stats[key]
    if stats["valid_pixels"]:
        row["valid_pixels_path"] = relative(path.with_name(stats["valid_pixels"]))
    return row


def write_reviews(images: pd.DataFrame, verified: dict | None = None) -> None:
    details = dict(preparer_self_check=True, independent_review=False, exported_images=len(images),
                   expected_images=len(candidates), sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                   patient_linked_samples=int(samples.patient_key.notna().sum()), unique_patients=int(samples.patient_key.nunique()))
    checks = {"S01": "78个准入独立ROI，57个IDC加21个来源队列扩展；原核验复用、live size/header/panel重核；未重hash。",
              "I01": "image_id与HTANDataFileID唯一；source_files相对NAS，完整39基础字段。",
              "I02": "非空image→sample→patient显式连接；7图sample留空，panel外键有效。",
              "I03": "完整ROI框；原生分辨率；居中方形补边。",
              "I04": "71图标本标签与来源显式连接一致；7图标本声明冲突保留空关联；未进行多sample拆分。",
              "P01": "原生光路/OME page顺序对接本study S1；两个实际panel共55行。",
              "P02": "27/28个来源通道均保留；191Ir/193Ir两条DNA为segmentation-only，其余downstream。",
              "P03": "已输出图像逐一重开核对CYX/dtype/名称/顺序/1µm。",
              "M01": "71/78 image确认sample连接；27份明确biospecimen→9位patient。7图PA/PB或H1/H2来源标签冲突，未按文件名推断；见source_identity_conflicts.json。",
              "M02": "已填临床值逐sample由明确来源重建；未知时点/age/smoking Yes-No及生存终点保留空值。",
              "D01": "整理者交付；独立reviewer尚未核对，D01不签署pass。"}
    rows = []
    for check, explanation in checks.items():
        check_details = dict(details, explanation=explanation)
        if check in {"M01", "I04"}:
            check_details["image_ids"] = [r["image_id"] for r in candidates if r["sample_identity_conflict"]]
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
                         check_id=check, result="unresolved" if check in {"D01", "M01", "I04"} else "pass",
                         source_evidence=f"{relative(PORTAL)};{relative(SUPPLEMENT)};{relative(CACHE / 'source_scope.json')}",
                         output_evidence=relative(OUTPUT / "images.parquet"),
                         details=json.dumps(check_details, ensure_ascii=False)))
    if verified:
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
                         check_id="I05", result="pass", source_evidence=relative(CACHE / "source_scope.json"),
                         output_evidence=relative(CACHE / "preparer_verification.json"),
                         details=json.dumps(dict(preparer_self_check=True, **verified), ensure_ascii=False)))
    result = pd.DataFrame(rows, columns=pd.read_csv(TEMPLATES / "review_results.csv").columns)
    path = OUTPUT / "review_results.csv"
    if path.exists():
        previous = pd.read_csv(path, dtype=str, keep_default_na=False)
        previous = previous[~((previous.preparer == PREPARER) & (previous.reviewer == ""))]
        result = pd.concat([previous, result], ignore_index=True)
    with formal_publish_lease(path, directory=path.parent):
        result.to_csv(path, index=False, encoding="utf-8-sig")


def publish_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with formal_publish_lease(path, directory=path.parent, data_root=NAS):
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


def record_table1_review_pending(samples: pd.DataFrame, table1: pd.DataFrame) -> None:
    path = OUTPUT / "review_results.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    marker = '"public_table1_metadata_refresh": true'
    if frame.details.str.contains(marker, regex=False).any():
        return
    row = dict.fromkeys(frame.columns, "")
    row.update(
        study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
        check_id="M02", result="unresolved",
        source_evidence=f"{relative(SUPPLEMENT)};{relative(PORTAL)}",
        output_evidence=f"{relative(OUTPUT / 'sample_metadata.csv')};{relative(OUTPUT / 'field_mapping.csv')}",
        details=json.dumps(dict(
            public_table1_metadata_refresh=True, independent_review=False,
            sample_rows=len(samples), source_patient_rows=len(table1),
            mapped_patients=int(samples.source_subject_id.nunique()),
            explanation="Official Table 1 source fields and diagnosis-to-surgery timing added; independent M02 review is pending."
        ), ensure_ascii=False)
    )
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    publish_csv(path, frame)


def verify_pixels(candidate: dict, panel: pd.DataFrame) -> dict:
    """Compare all kept channels in a signal-bearing and a boundary source block."""
    import tifffile
    from cellweft.data.io.readers import read_tiff_window

    h, w = candidate["height"], candidate["width"]
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    blocks = [(max(0, w//2-16), max(0, h//2-16), min(w, w//2+16), min(h, h//2+16)),
              (0, 0, min(w, 32), min(h, 32))]
    path = OUTPUT / "images" / candidate["image_id"] / "input.tiff"
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    with source_reader(candidate) as reader, tifffile.TiffFile(path) as tif:
        # Locate a known nonzero downstream pixel, rather than assuming center signal.
        for index in kept[kept.role == "downstream"].source_channel_index:
            plane = reader.read_window([int(index)], (0, 0, w, h))[0]
            yx = np.argwhere(plane > 0)
            if len(yx):
                y, x = yx[len(yx)//2]
                x0, y0 = max(0, int(x)-16), max(0, int(y)-16)
                blocks[0] = (x0, y0, min(w, x0+32), min(h, y0+32))
                break
        else:
            raise ValueError("Representative image has no downstream signal")
        for entry in kept.itertuples(index=False):
            for block in blocks:
                source = reader.read_window([int(entry.source_channel_index)], block)[0]
                x0, y0, x1, y1 = block
                target = read_tiff_window(tif, int(entry.output_channel_index), (x0+left, y0+top, x1+left, y1+top))
                assert source.dtype == target.dtype and np.array_equal(source, target)
            if left:
                assert not read_tiff_window(tif, int(entry.output_channel_index), (0, 0, left, side)).any()
            if side-w-left:
                assert not read_tiff_window(tif, int(entry.output_channel_index), (left+w, 0, side, side)).any()
            if top:
                assert not read_tiff_window(tif, int(entry.output_channel_index), (0, 0, side, top)).any()
            if side-h-top:
                assert not read_tiff_window(tif, int(entry.output_channel_index), (0, top+h, side, side)).any()
    return dict(image_id=candidate["image_id"], panel_id=candidate["panel_id"], source_format=candidate["source_format"],
                dtype=candidate["dtype"], source_blocks=blocks, compared_channels=len(kept), padding_checked=True)


def make_overview(candidate: dict) -> Path:
    """Render full ROI nuclear, epithelial and stromal signal for visual review."""
    names = candidate["names"]
    selected = [names.index("DNA"), names.index("Pan Keratin"), names.index("SMA")]
    with source_reader(candidate) as reader:
        values = reader.read_window(selected, (0, 0, candidate["width"], candidate["height"]))
    canvas = Image.new("RGB", (900, 345), "black")
    draw = ImageDraw.Draw(canvas)
    for i, plane in enumerate(values):
        positive = plane[plane > 0]
        scale = np.percentile(positive, 99) if len(positive) else 1
        rendered = Image.fromarray((np.clip(plane/scale, 0, 1)*255).astype("uint8"))
        rendered.thumbnail((294, 294))
        canvas.paste(rendered.convert("RGB"), (i*300, 25))
        draw.text((i*300+5, 5), names[selected[i]], fill="white")
    draw.text((5, 322), candidate["image_id"] + " " + candidate["source_format"] + " " + candidate["panel_id"], fill="white")
    path = CACHE / "overviews" / f"{candidate['image_id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("metadata", "export", "verify", "public-metadata-refresh"), required=True)
parser.add_argument("--limit", type=int)
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

if args.stage == "public-metadata-refresh":
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
    workbook = openpyxl.load_workbook(SUPPLEMENT, read_only=True, data_only=True)
    values = list(workbook[TABLE1_SHEET].values)
    table1 = pd.DataFrame(values[1:], columns=values[0])
    workbook.close()
    samples, fields = attach_table1_clinical(samples, fields, table1)
    record_table1_review_pending(samples, table1)
    publish_csv(OUTPUT / "sample_metadata.csv", samples)
    publish_csv(OUTPUT / "field_mapping.csv", fields)
    print(json.dumps(dict(
        stage=args.stage, sample_rows=len(samples),
        patients=int(samples.patient_key.nunique()),
        source_table1_rows=len(table1), source_fields=len(table1.columns) - 1,
        mapped_extensions=len(table1.columns) - 1,
    ), ensure_ascii=False), flush=True)
    raise SystemExit(0)

from cellweft.data.preparation.study_input import export_input, save_input_tables

# Resolve actual source channels and biological identities before pixel export.
candidates, panels, portal = load_sources()
samples, fields = sample_tables(portal)
images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists()
          else pd.DataFrame(columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns))
estimated_bytes = sum(max(r["height"], r["width"])**2 * len(r["names"]) * np.dtype(r["dtype"]).itemsize for r in candidates)
write_json(CACHE / "source_scope.json", dict(study=STUDY, expected_images=len(candidates),
           original_idc_images=57, admitted_original_tiff_extension_images=21,
           untouched_incomplete_files=[dict(path=relative(p), size=p.stat().st_size)
                                       for p in sorted(RAW.rglob("*.part"))],
           estimated_uncompressed_output_bytes=estimated_bytes, candidates=candidates,
           source_evidence=[relative(OLD_PROOF), relative(META / "htan_original_cohort_extension.json"),
                            relative(PORTAL), relative(SUPPLEMENT)],
           consulted_public_urls=[PAPER_URL, SUPPLEMENT_URL, PORTAL_URL,
                                  "https://portal.imaging.datacommons.cancer.gov/collections/htan_wustl/"],
           metadata_gaps=GAPS, controller_requests=REQUESTS))
save_input_tables(OUTPUT, images, samples, panels, fields)
progress(args.stage, images, estimated_uncompressed_output_bytes=estimated_bytes)

if args.stage == "metadata":
    print(json.dumps(dict(stage="metadata", expected_images=len(candidates), samples=len(samples),
                         patients=int(samples.patient_key.nunique()), estimated_output_bytes=estimated_bytes), ensure_ascii=False), flush=True)

if args.stage == "export":
    if shutil.disk_usage(NAS).free < estimated_bytes + 100*1024**3 or shutil.disk_usage('/tmp').free < 64*1024**3:
        raise SystemExit("Insufficient reserved space; source metadata preserved")
    errors = []
    excluded = []
    for candidate in candidates[:args.limit]:
        image_id = candidate["image_id"]
        panel = panels[panels.panel_id == candidate["panel_id"]]
        path = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = CACHE / "export_receipts" / f"{image_id}.json"
        try:
            binding = source_binding(candidate)
            if path.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["source_binding"] == binding
                assert receipt["output_marker_names"] == panel[panel.keep_channel].marker_id.tolist()
                header_check(candidate, path, panel)
                stats = receipt["export_stats"]
                reused = True
            else:
                with source_reader(candidate) as reader:
                    stats = export_input(reader, panel, path)
                if not stats["admitted"]:
                    excluded.append(dict(image_id=image_id, reason="all admitted downstream channels zero"))
                    write_json(CACHE / "exclusions.json", excluded)
                    continue
                header_check(candidate, path, panel)
                write_json(receipt_path, dict(image_id=image_id, source_binding=binding,
                           output_marker_names=panel[panel.keep_channel].marker_id.tolist(), export_stats=stats))
                reused = False
            row = image_row(candidate, stats)
            images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panels, fields)
            progress("export", images, last_completed_image=image_id, reused=reused, errors=errors)
            print(json.dumps(dict(image_id=image_id, completed=len(images), reused=reused), ensure_ascii=False), flush=True)
        except Exception as error:
            errors.append(dict(image_id=image_id, error=repr(error)))
            write_json(CACHE / "export_errors.json", errors)
            progress("export_error", images, errors=errors)
            print(json.dumps(errors[-1], ensure_ascii=False), flush=True)
    write_reviews(images)
    progress("export_complete" if not errors else "export_partial", images, errors=errors, excluded=excluded)
    if errors:
        raise SystemExit(1)

if args.stage == "verify":
    exported = {str(x) for x in images.image_id}
    groups = {}
    for candidate in candidates:
        if candidate["image_id"] not in exported:
            continue
        panel = panels[panels.panel_id == candidate["panel_id"]]
        header_check(candidate, OUTPUT / "images" / candidate["image_id"] / "input.tiff", panel)
        key = (candidate["source_format"], candidate["panel_id"], candidate["dtype"])
        groups.setdefault(key, candidate)
    representatives = {r["image_id"]: r for r in groups.values()}
    mismatched_geometry = next((r for r in candidates if not r["portal_geometry_agrees"] and r["image_id"] in exported), None)
    if mismatched_geometry:
        representatives[mismatched_geometry["image_id"]] = mismatched_geometry
    smallest = min((r for r in candidates if r["image_id"] in exported), key=lambda r: r["height"] * r["width"])
    representatives[smallest["image_id"]] = smallest
    comparisons = []
    for candidate in representatives.values():
        panel = panels[panels.panel_id == candidate["panel_id"]]
        comparisons.append(verify_pixels(candidate, panel))
        make_overview(candidate)
    verification = dict(header_checked_images=len(exported), pixel_comparisons=comparisons,
                        independent_review=False, source_hashes_recomputed=False)
    write_json(CACHE / "preparer_verification.json", verification)
    write_reviews(images, verification)
    progress("prepared", images, verification=verification)
    print(json.dumps(verification, ensure_ascii=False), flush=True)
