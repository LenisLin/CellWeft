"""Prepare native serial-section IMC inputs for the four Kuett breast models.

Inputs: Zenodo 4752030 complete archives, accepted acquisition proofs, study S1,
paper methods, supplementary antibody tables and pinned author notebooks.
Outputs: native uint16 square CYX input.tiff, image/sample/panel/field tables,
preparer checks and resumable progress. Run exports through batch shared formal write lease.
The main and second punches share one biological specimen, as stated in Methods.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
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
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import openpyxl
import pandas as pd
import requests
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

STUDY = "Kuett2021_IMC_BreastCancer"
REVISION = "82f43d46713a461b5a1cec3e70b5bd5eb81830ca"
PROJECT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--panel", action="append", choices=["main_201710", "second_201709", "lvi_blood", "lvi_lymph"])
parser.add_argument("--resume-image", help="First image ID to process after a verified checkpoint")
parser.add_argument("--max-new-images", type=int, help="Finish after this many new image receipts")
args = parser.parse_args()
if args.max_new_images is not None and args.max_new_images < 1:
    parser.error("--max-new-images must be positive")
root = args.data_root
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
output = root / "processed" / STUDY
metadata = metadata_source_directory(root, STUDY)
job = Path(os.environ["CELLWEFT_STUDY_INPUT_JOB"])
PREPARER = "codex_cli_" + job.name
temporary = Path("/tmp/study_input_batch") / STUDY
templates = PROJECT / "docs/data/templates/study_input"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
for directory in (cache, output, job, temporary, cache / "receipts", cache / "overviews"):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(temporary / "matplotlib")


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, root))


def write_json(path: Path, value: object) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(staging, path)


def progress(stage: str, **details: object) -> None:
    write_json(job / "progress.json", dict(study=STUDY, stage=stage,
        completed_images=len(images), expected_images=312, sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        updated_at_utc=datetime.now(timezone.utc).isoformat(), **details))


def fetch(name: str, url: str) -> Path:
    """Read the formally received public source metadata."""
    path = metadata / "source_metadata" / name
    assert path.is_file(), f"Missing received source: {name}"
    return path


# Reuse accepted proofs, binding size/mtime to the original complete archives.
proof_path = metadata / "acquisition_verification.json"
with formal_read_lease(proof_path):
    verification = json.loads(proof_path.read_text())
assert verification["complete"] and verification["verified_acquisitions"] == 312
archives = verification["archives"]
assert sum(len(a["images"]) for a in archives) == 312
inventory = []
for archive in archives:
    first = archive["images"][0]
    path = root / first["source_identity"]["raw_path"]
    stat = path.stat()
    assert archive["complete"] and not archive["failures"]
    assert len({json.dumps(p["native_channel_map"], sort_keys=True) for p in archive["images"]}) == 1
    for proof in archive["images"]:
        assert proof["complete"]
        assert (stat.st_size, stat.st_mtime_ns) == (proof["file_stat"]["size"], proof["file_stat"]["mtime_ns"])
    with formal_read_lease(path), zipfile.ZipFile(path) as outer:
        for member in outer.infolist():
            inventory.append(dict(archive=relative(path), member=member.filename, bytes=member.file_size,
                                  native_container=member.filename == first["source_identity"]["nested_source_member"]))
            if 0 < member.file_size < 15000 and member.filename.endswith(".csv"):
                destination = cache / "archive_metadata" / member.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(outer.read(member))
write_json(cache / "outer_inventory.json", inventory)
record_path = metadata / "source_metadata/official_zenodo_record.json"
with formal_read_lease(record_path):
    record = json.loads(record_path.read_text())
assert int(record["id"]) == 4752030
assert {item["key"]: item["size"] for item in record["files"]} == {
    a["archive"]: a["images"][0]["file_stat"]["size"] for a in archives}
write_json(cache / "source_scope.json", dict(accepted_images=312,
    native_tiff_members=312 + sum(len(a["unselected_native_members"]) for a in archives),
    additional_members_outside_accepted_scope={a["panel"]: a["unselected_native_members"] for a in archives if a["unselected_native_members"]},
    scope_rule="复用既有准入：2017模型image_order，LVI模型final_3D_stack_order；其余未列入已核验manifest的TIFF在本批范围外，完整来源保留。",
    excluded_derivatives="SIMILARITY aligned stacks, 3D masks, single-cell tables; duplicate representations of source acquisitions",
    source_archive_sizes={a["archive"]: a["images"][0]["file_stat"]["size"] for a in archives}))

# The primary paper resolves specimen grouping; no per-patient clinical table is published.
article_path = fetch("article.xml", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7613779/fullTextXML")
with formal_read_lease(article_path):
    article = ET.parse(article_path).getroot()
paragraphs = [" ".join(p.itertext()) for p in article.iter("p")]
tissue_text = next(p for p in paragraphs if "The main and the second breast carcinoma 3D models were generated from one patient sample" in p)
assert "The blood and the lymph vessel lymphovascular invasion models were generated from two different patient samples" in tissue_text
scale_text = next(p for p in paragraphs if "Laser ablation was performed at a resolution of approximately 1" in p)
supplement_url = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43018-021-00301-w/MediaObjects/43018_2021_301_MOESM3_ESM.xlsx"
supplement = fetch("Supplementary_tables.xlsx", supplement_url)
with formal_read_lease(supplement):
    workbook = openpyxl.load_workbook(supplement, read_only=True, data_only=True)
    try:
        supplement_rows = {sheet.title: [list(row) for row in sheet.iter_rows(max_row=150, values_only=True)
                                        if any(value is not None for value in row)] for sheet in workbook}
    finally:
        workbook.close()
assert len(supplement_rows) == 4
assert any(row[:3] == ["Gd156", "HER2", "29D8"] for row in supplement_rows["Supplementary_Tabel_3"])
assert any(row[:4] == ["Nd148", "S100A4", "NJ-4F3-D1", "LVI lymph"] for row in supplement_rows["Supplementary_Tabel_4"])
tree_path = fetch("github_tree.json", f"https://api.github.com/repos/BodenmillerGroup/3D_IMC_publication/git/trees/{REVISION}?recursive=1")
with formal_read_lease(tree_path):
    tree = json.loads(tree_path.read_text())
repository_files = [node["path"] for node in tree["tree"] if node["type"] == "blob"]
notebook_path = "Python/scripts/single_cell_analysis/model2_model201709_model_analyse_single_cells.ipynb"
notebook = fetch("author_repository/" + notebook_path,
    f"https://raw.githubusercontent.com/BodenmillerGroup/3D_IMC_publication/{REVISION}/{notebook_path}")
with formal_read_lease(notebook):
    notebook_cells = ["".join(cell.get("source", [])) for cell in json.loads(notebook.read_text())["cells"]]
assert any("HER2(bis)" in cell and "sc.pl.violin" in cell for cell in notebook_cells)

model_samples = {"main_201710": "her2_main_second_specimen", "second_201709": "her2_main_second_specimen",
                 "lvi_blood": "blood_lvi_specimen", "lvi_lymph": "lymph_lvi_specimen"}
sample_rows = []
for specimen in dict.fromkeys(model_samples.values()):
    row = {name: None for name in pd.read_csv(templates / "sample_metadata.csv").columns}
    row.update(study=STUDY, sample_id=STUDY + "__" + specimen,
        patient_key=STUDY + "__patient_" + specimen.removesuffix("_specimen"),
        sample_type="archived breast carcinoma tissue specimen", tissue_type="tumor",
        anatomical_site="breast", primary_site="breast", preservation_method="ffpe",
        center_id="University Hospital Zurich tissue biobank", primary_diagnosis="breast carcinoma",
        molecular_subtype="HER2-positive", case_control_status="case",
        metadata_notes="患者键是依据论文Tissue samples段明确关系建立的本study分组键；无公开个体患者编号。"
                       "主/第二模型为同一标本的不同punch，共享sample_id；两个LVI模型为另两份患者标本。"
                       "标本原始ID、年龄、性别、分期、治疗和随访无对应公开表，留空。")
    if specimen == "her2_main_second_specimen":
        row["histology"] = "ductal breast carcinoma"
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)

# Resolve all physical pages to S1; the six acquisition fields have no protein identity.
# Temporary study adapter: panel_from_s1 currently requires an S1 row for every
# source name. Map annotated channels through it, then restore explicitly excluded
# native acquisition fields. A shared technical-row API can replace this adapter.
with formal_read_lease(s1_path):
    s1 = pd.read_csv(s1_path, dtype=str, keep_default_na=False)
s1_names = set(s1.loc[s1.Study == STUDY, "Original marker"])
panels = []
for archive in archives:
    proof = archive["images"][0]
    native = proof["native_channel_map"]
    annotated = [channel for channel in native if channel["channel_type"] != "acquisition_field"]
    names = [channel["label"] or channel["isotope"] for channel in annotated]
    assert set(names) <= s1_names
    selected = set(range(len(annotated)))
    if archive["panel"] == "second_201709":
        selected.remove(names.index("HER2"))  # Table 3 and author figure use Gd156/29D8.
    if archive["panel"] == "lvi_lymph":
        selected.remove(names.index("FSP1"))  # Table 4 explicitly lists Nd148/NJ-4F3-D1 for lymph.
    with formal_read_lease(s1_path):
        panel = panel_from_s1(STUDY, STUDY + "__" + archive["panel"], names, s1_path,
                              downstream_source_indices=selected)
    panel["source_channel_index"] = [channel["index"] for channel in annotated]
    panel["isotope"] = [channel["isotope"] for channel in annotated]
    panel["source_channel_name"] = [channel["source_channel"] for channel in annotated]
    for index, channel in enumerate(annotated):
        if channel["label"] == "Iridium":
            panel.loc[index, "marker_id"] = channel["isotope"]
        if index not in selected:
            panel.loc[index, "exclusion_reason"] = ("Duplicate CD340: select Gd156/29D8, Supplementary Table 3 and author notebook cell 23"
                if archive["panel"] == "second_201709" else
                "Duplicate S100A4: select Nd148/NJ-4F3-D1 explicitly listed for LVI lymph in Supplementary Table 4")
    fields = [dict(panel_id=STUDY + "__" + archive["panel"], source_channel_index=channel["index"],
        source_name=channel["source_channel"], marker_id="", mapping_status="excluded", role="exclude",
        nuclear=False, body=False, keep_channel=False, output_channel_index=None, keep_reason="",
        exclusion_reason="Native acquisition metadata page, not an antibody or nuclear measurement",
        s1_original_marker="", isotope="", source_channel_name=channel["source_channel"])
        for channel in native if channel["channel_type"] == "acquisition_field"]
    panel = pd.concat([panel, pd.DataFrame(fields)], ignore_index=True).sort_values("source_channel_index")
    panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
    panel["display_name"] = panel.marker_id
    assert panel.source_channel_index.tolist() == list(range(len(native)))
    panels.append(panel)
panel = pd.concat(panels, ignore_index=True)

field_rows = []


def mapping(source: Path, source_field: str, table: str, target: str, rule: str, notes: str = "") -> None:
    field_rows.append(dict(study=STUDY, source_file=relative(source), source_sheet_or_table="",
        source_field=source_field, target_table=table, target_field=target,
        join_keys="source archive model + exact native member; Methods model-to-specimen relation",
        value_mapping="{}", transformation_rule=rule, time_origin_rule="",
        conflict_rule="按原始证据；不根据切片文件名猜患者；未知留空", notes=notes))


mapping(article_path, "Methods / Tissue samples", "sample_metadata", "sample_id",
        "main_201710与second_201709共享一份生物学取样；lvi_blood、lvi_lymph各为另一份取样，共3行。")
mapping(article_path, "Methods / Tissue samples", "sample_metadata", "patient_key",
        "依据同一患者及两份不同患者标本的明确描述，生成3个study内关系分组键；无公开原始患者ID。")
for name in ("sample_type", "tissue_type", "anatomical_site", "primary_site", "primary_diagnosis", "preservation_method", "center_id", "case_control_status"):
    mapping(article_path, "Methods / Tissue samples", "sample_metadata", name, "按论文明确的存档乳腺癌FFPE标本及组织库来源填写。")
    if name in {"tissue_type", "preservation_method"}:
        values = {"breast carcinoma": "tumor"} if name == "tissue_type" else {"formalin-fixed paraffin-embedded": "ffpe"}
        field_rows[-1]["value_mapping"] = json.dumps(values)
        field_rows[-1]["transformation_rule"] = "论文来源原值按模板value_sets.csv转换为标准分类值；诊断原文保留于primary_diagnosis。"
for name in ("histology", "molecular_subtype"):
    mapping(article_path, "Results / 3D IMC; lymphovascular invasion", "sample_metadata", name,
            "主/第二模型为HER2+ ductal breast carcinoma；两份LVI为HER2+ breast carcinoma，不将ductal外推至LVI。")
mapping(proof_path, "source_identity.raw_path + nested_source_member + source_member", "image_metadata", "source_image_id",
        "源图身份为嵌套ZIP成员；source_files引用完整外层ZIP，读取未配准原生TIFF完整ROI。")
mapping(proof_path, "panel/native_channel_map", "image_metadata", "sample_id",
        "归档模型→论文明确标本关系；来源候选312张对应3份标本，正式image连接覆盖随实际索引更新。")
mapping(article_path, "Methods / Tissue samples; source model", "image_metadata", "core_id",
        "模型标识定位独立组织punch；主/第二punch共享sample_id，切片ROI不新建sample。")
mapping(article_path, "Methods / Data acquisition", "image_metadata", "pixel_size_x_um", "论文声明laser ablation约1 µm；记录1 µm，精度为approximately。")
mapping(article_path, "Methods / Data acquisition", "image_metadata", "pixel_size_y_um", "同上，原生XY无缩放。")
mapping(proof_path, "panel/model identity", "image_metadata", "region_annotation",
        "保留来源模型语义：MainHer2BreastCancerModel / SecondHer2BreastCancerModel / LVIBloodBreastCancerModel / LVILymphBreastCancerModel。")
mapping(s1_path, "Study + Original marker", "panel_mapping", "marker_id",
        "按native isotope/page和来源panel clean_target连接本study S1；不按panel CSV行序猜像素顺序。")
mapping(supplement, "Tables 3/4; Metal, Target, Antibody Clone, Antibody used in", "panel_mapping", "keep_channel",
        "第二模型HER2保留Gd156/29D8（作者notebook cell23实际使用）；lymph的S100A4保留Nd148/NJ-4F3-D1；重复项排除。")
mapping(proof_path, "native_channel_map.channel_type", "panel_mapping", "source_channel_index",
        "保留全部原生页面；六个acquisition_field无S1行，明确排除；Iridium核通道按原isotope显示Ir191/Ir193。")
mapping(supplement, "All four supplementary tables; pinned GitHub repository tree", "sample_metadata", "metadata_notes",
        "已查论文、四个ZIP全部小型metadata、补充表全部四sheet及作者仓库树；补充表为技术/抗体资料，无个体临床表或sample map。")
field_mapping = pd.DataFrame(field_rows)
write_json(cache / "sample_provenance.json", dict(tissue_samples_paragraph=tissue_text,
    data_acquisition_paragraph=scale_text, model_to_sample=model_samples,
    sample_rows=3, expected_image_links=312, patient_groups=3,
    patient_identifier_semantics="Methods-derived study-local patient grouping; no published individual ID",
    supplementary_sheets=list(supplement_rows), author_repository_revision=REVISION,
    repository_files=repository_files, clinical_gap="无公开个体临床ID、年龄、性别、分期、治疗或结局表。",
    sources=[relative(record_path), relative(article_path), relative(supplement), relative(tree_path), relative(notebook), relative(cache / "outer_inventory.json")]))
controller_requests = ["安排不同CLI上下文独立复核；D01未写独立验收pass。",
    "四个原生ZIP共319张TIFF；本批按原准入manifest处理312张，另main 5张、second 1张、lymph 1张位于作者选定范围外，见source_scope.json。若要扩展范围，需单独裁定并核验这7张。",
    "共享panel_from_s1暂不能表达无S1项的采集元数据页面；本study适配先映射抗体/isotope行，再补回六个明确排除的technical行。后续可迁移该通用能力。"]
write_json(cache / "controller_requests.json", controller_requests)
with formal_read_lease(output), resource_lock(output, shared=True):
    images = (pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists()
              else pd.read_csv(templates / "image_metadata.csv"))


def save_tables() -> None:
    """Publish actual image linkage coverage with every resumable image checkpoint."""
    linked = int(images.sample_id.notna().sum())
    selected = (field_mapping.target_table == "image_metadata") & (field_mapping.target_field == "sample_id")
    field_mapping.loc[selected, "transformation_rule"] = (
        f"归档模型→论文明确标本关系；来源候选312张对应3份标本；正式索引已发布{len(images)}/312张，"
        f"其中{linked}/{len(images)}张已连接sample；3/3份标本连接论文确认的患者分组。")
    save_input_tables(output, images, samples, panel, field_mapping)
    write_json(cache / "metadata_validation.json", dict(
        image_base_fields=39, actual_image_rows=len(images), sample_fields=len(samples.columns),
        sample_rows=len(samples), sample_linked_images=linked,
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        corrected_vocabulary_cells=6, tissue_type=samples.tissue_type.tolist(),
        preservation_method=samples.preservation_method.tolist(),
        panel_rows=len(panel), field_mapping_rows=len(field_mapping),
        validation_stage="preparer_repair", independent_review="pending"))


save_tables()
failures = []


def save_checks() -> None:
    """Append repair evidence, preserving prior independent reviewer conclusions."""
    rows = []
    for check in pd.read_csv(templates / "review_checks.csv").itertuples(index=False):
        if check.check_id in {"P01", "P02", "M02"}:
            continue  # Preserve the unaffected independent panel/clinical passes.
        details = dict(stage="preparer_repair", independent_review=False, repair_round=2,
            exported_images=len(images), expected_images=312, sample_rows=3,
            sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=3,
            note=f"第2轮修补：实际索引{len(images)}/312张；field_mapping同步实际sample连接覆盖。整理者自检证据见receipt，等待不同CLI独立复查及D01绑定；沿用0199复核的P01/P02/M02结论。",
            failures=failures, per_image_evidence=relative(cache / "receipts"))
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
            record_id=STUDY, check_id=check.check_id, result="unresolved",
            source_evidence=relative(proof_path) + ";" + relative(cache / "sample_provenance.json"),
            output_evidence=relative(output), details=json.dumps(details, ensure_ascii=False)))
    with formal_read_lease(output), resource_lock(output, shared=True):
        previous = pd.read_csv(output / "review_results.csv", keep_default_na=False)
    previous = previous[~((previous.preparer == PREPARER) & (previous.reviewer == ""))]
    staging = temporary / "review_results.csv"
    pd.concat([previous, pd.DataFrame(rows)], ignore_index=True).to_csv(staging, index=False, encoding="utf-8-sig")
    with formal_publish_lease(output / "review_results.csv", directory=output):
        partial = output / "review_results.csv.tmp"
        shutil.copyfile(staging, partial)
        os.replace(partial, output / "review_results.csv")


save_checks()
progress("metadata_prepared", gaps=["等待原生TIFF导出和不同CLI独立复核"])
if args.metadata_only:
    write_json(cache / "delivery_summary.json", dict(study=STUDY, expected_images=312,
        exported_images=len(images), sample_rows=3,
        sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=3,
        per_panel=images.groupby("panel_id").size().to_dict(), pixel_checked_images=0,
        stage="metadata_prepared_pixels_pending", independent_review="pending",
        controller_requests=controller_requests))
    raise SystemExit(0)


def check_pixels(reader: object, destination: Path, selected: pd.DataFrame, info: dict) -> list[dict]:
    """Compare all retained channels at source-interior and boundary windows."""
    h, w = reader.shape[-2:]
    regions = [(max(0, w // 2 - 16), max(0, h // 2 - 16), min(w, w // 2 + 16), min(h, h // 2 + 16)),
               (0, 0, min(32, w), min(32, h))]
    evidence = []
    with formal_read_lease(destination), resource_lock(destination.parent, shared=True, name=".input.lock"), tifffile.TiffFile(destination) as tif:
        for x0, y0, x1, y1 in regions:
            values = reader.read_window(selected.source_channel_index.tolist(), (x0, y0, x1, y1))
            target = (x0 + info["pad_left_px"], y0 + info["pad_top_px"], x1 + info["pad_left_px"], y1 + info["pad_top_px"])
            actual = np.stack([read_tiff_window(tif, i, target) for i in range(len(selected))])
            assert actual.dtype == values.dtype and np.array_equal(actual, values)
            evidence.append(dict(source_window=[x0, y0, x1, y1], output_window=list(target),
                                 retained_channels=len(selected), equal=True, dtype=str(values.dtype)))
        side = info["height_px"]
        padding = [(0, 0, side, info["pad_top_px"]), (0, side-info["pad_bottom_px"], side, side),
                   (0, 0, info["pad_left_px"], side), (side-info["pad_right_px"], 0, side, side)]
        for x0, y0, x1, y1 in padding:
            if x1 > x0 and y1 > y0:
                assert all(not read_tiff_window(tif, i, (x0, y0, x1, y1)).any() for i in range(len(selected)))
    return evidence


def overview(reader: object, chosen: pd.DataFrame, destination: Path, title: str) -> None:
    """Render native nuclear/epithelial measurements for provenance and ROI context."""
    import matplotlib.pyplot as plt
    nuclear = chosen[chosen.nuclear].iloc[-1]
    marker = chosen[chosen.source_name == "panCK"].iloc[0]
    h, w = reader.shape[-2:]
    values = reader.read_window([int(nuclear.source_channel_index), int(marker.source_channel_index)], (0, 0, w, h))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, image, name in zip(axes, values, [nuclear.marker_id, marker.marker_id]):
        high = np.percentile(image[image > 0], 99.5) if (image > 0).any() else 1
        ax.imshow(image, cmap="gray", vmin=0, vmax=high)
        ax.set_title(str(name))
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


# Export one native acquisition at a time, checkpointing after each image.
resume_pending = args.resume_image is not None
new_images = 0
stop_batch = False
for archive in archives:
    pid = archive["panel"]
    if args.panel and pid not in args.panel:
        continue
    proofs = archive["images"]
    first = proofs[0]
    source = root / first["source_identity"]["raw_path"]
    nested_member = first["source_identity"]["nested_source_member"]
    nested = temporary / "native_archives" / Path(nested_member).name
    current_panel = panel[panel.panel_id == STUDY + "__" + pid].copy()
    kept = current_panel[current_panel.keep_channel].sort_values("output_channel_index")
    estimated_output = sum(max(p["bounded_native_reads"]["channel_reads"][0]["shape"]) ** 2 * len(kept) * 2 for p in proofs)
    assert shutil.disk_usage(root).free > 100 * 1024**3 + estimated_output
    assert shutil.disk_usage(temporary).free > 64 * 1024**3 + first["source_identity"]["nested_source_member_bytes"] + max(p["source_member_bytes"] for p in proofs)
    progress("extracting_native_archive", panel=pid, estimated_uncompressed_output_bytes=estimated_output,
             gaps=["本批其余原生TIFF待导出", "等待独立复核"])
    with formal_read_lease(source), zipfile.ZipFile(source) as outer:
        nested_info = outer.getinfo(nested_member)
        assert nested_info.file_size == first["source_identity"]["nested_source_member_bytes"]
        if not nested.exists():
            nested.parent.mkdir(exist_ok=True)
            staging = nested.with_suffix(".extracting")
            with outer.open(nested_info) as src, staging.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
            assert staging.stat().st_size == nested_info.file_size
            os.replace(staging, nested)
        assert nested.stat().st_size == nested_info.file_size
    with zipfile.ZipFile(nested) as native_zip:
        native_members = {m.filename for m in native_zip.infolist()
                          if m.filename.lower().endswith((".tiff", ".tif"))
                          and not m.filename.startswith("__MACOSX/")
                          and not Path(m.filename).name.startswith("._")}
        expected_members = {p["source_member"] for p in proofs}
        assert native_members == expected_members | set(archive["unselected_native_members"])
        write_json(cache / f"native_inventory_{pid}.json", dict(native_container=relative(source) + "!" + nested_member,
            local_staging_path=str(nested),
            members=[dict(member=m.filename, bytes=m.file_size,
                          selected=m.filename in expected_members,
                          reason="accepted source acquisition" if m.filename in expected_members else "outside accepted source image-order/stack-order manifest")
                     for m in native_zip.infolist() if not m.is_dir()]))
        minimum_area = min(np.prod(p["bounded_native_reads"]["channel_reads"][0]["shape"]) for p in proofs)
        for number, proof in enumerate(proofs):
            image_id = pid + "__" + re.sub(r"[^A-Za-z0-9_.-]+", "_", proof["image_id"].removesuffix(".tiff").removesuffix(".tif"))
            if resume_pending:
                if image_id != args.resume_image:
                    continue
                resume_pending = False
            destination = output / "images" / image_id / "input.tiff"
            receipt_path = cache / "receipts" / (image_id + ".json")
            member = native_zip.getinfo(proof["source_member"])
            assert member.file_size == proof["source_member_bytes"]
            revision = f"Zenodo4752030;archive_size={proof['file_stat']['size']};archive_mtime_ns={proof['file_stat']['mtime_ns']};member={proof['source_member']};bytes={member.file_size};CRC={member.CRC}"
            if receipt_path.exists():
                with formal_read_lease(receipt_path):
                    receipt = json.loads(receipt_path.read_text())
                assert receipt["row"]["source_revision"] == revision
                assert receipt["panel_selection"] == kept[["source_channel_index", "marker_id"]].to_dict("records")
                if receipt["result"]["admitted"]:
                    assert destination.exists()
                    assert receipt["output_stat"] == dict(size=destination.stat().st_size, mtime_ns=destination.stat().st_mtime_ns)
                    with formal_read_lease(destination), resource_lock(destination.parent, shared=True, name=".input.lock"), tifffile.TiffFile(destination) as tif:
                        assert len(tif.pages) == len(kept) and tif.pages[0].shape == (receipt["result"]["height_px"],) * 2
                    indexed = images.loc[images.image_id == image_id]
                    if indexed.empty:
                        images = pd.concat([images, pd.DataFrame([receipt["row"]])], ignore_index=True)
                        save_tables()
                    else:
                        assert len(indexed) == 1
                        assert all(indexed.iloc[0][key] == receipt["row"][key] for key in
                                   ("source_image_id", "source_revision", "panel_id", "sample_id", "input_image_path"))
                progress("exporting_native_images", current_image=image_id, panel=pid, reused=True, gaps=["等待剩余导出及独立复核"])
                continue
            local_tiff = temporary / "current_source.tiff"
            progress("exporting_native_images", current_image=image_id, panel=pid, gaps=["等待剩余导出及独立复核"])
            try:
                with native_zip.open(member) as src, local_tiff.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024**2)
                assert local_tiff.stat().st_size == member.file_size
                with tifffile.TiffFile(local_tiff) as tif:
                    assert len(tif.pages) == len(current_panel)
                    shape = tif.pages[0].shape
                    axes = tif.series[0].axes
                    assert list(shape) == proof["bounded_native_reads"]["channel_reads"][0]["shape"]
                    assert all(page.shape == shape and page.dtype == np.dtype("uint16") for page in tif.pages)
                    if tif.ome_metadata:
                        channels = [e for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.rsplit("}", 1)[-1] == "Channel"]
                        assert [e.attrib["Fluor"] for e in channels] == proof["channel_order"]
                spec = SourceSpec(image_id, STUDY + "__" + pid, "tiff", local_tiff,
                                  tuple(range(len(current_panel))), pixel_size_um=(1.0, 1.0))
                with open_source(spec) as reader:
                    info = export_input(reader, current_panel, destination)
                    pixels = []
                    if info["admitted"]:
                        with formal_read_lease(destination), resource_lock(destination.parent, shared=True, name=".input.lock"), tifffile.TiffFile(destination) as tif:
                            names = [e.attrib["Name"] for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.rsplit("}", 1)[-1] == "Channel"]
                            assert names == kept.marker_id.tolist()
                            assert tif.series[0].axes == "CYX"
                            assert tif.series[0].shape == (len(kept), info["height_px"], info["width_px"])
                            assert tif.series[0].dtype == np.dtype(info["source_dtype"])
                        # Every first source layout and each unusually small ROI gets explicit pixel/overview checks.
                        if number in {0, len(proofs)//2, len(proofs)-1} or min(shape) < 300 or np.prod(shape) == minimum_area:
                            pixels = check_pixels(reader, destination, kept, info)
                            overview(reader, kept, cache / "overviews" / (image_id + ".png"), pid + " / " + proof["image_id"])
                row = dict(study=STUDY, image_id=image_id, sample_id=STUDY + "__" + model_samples[pid],
                    panel_id=STUDY + "__" + pid, image_kind="roi", source_image_id=nested_member + "!" + proof["source_member"],
                    source_files=[relative(source)], source_format="ome_tiff" if pid.startswith("lvi") else "tiff",
                    source_axes=axes, source_dtype=info["source_dtype"], source_revision=revision,
                    core_id=Path(archive["archive"]).stem, roi_id=proof["image_id"], region_annotation=Path(archive["archive"]).stem,
                    source_x0_px=0, source_y0_px=0, source_x1_px=shape[1], source_y1_px=shape[0],
                    **{key: info[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "height_px", "width_px", "source_channel_count", "channel_count")},
                    pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                    valid_pixels_path=relative(destination.with_name(info["valid_pixels"])) if info["valid_pixels"] else None,
                    intensity_description="Published unaligned native uint16 ion-count TIFF intensities; unchanged decoded values; no registration, compensation or intensity normalization applied here",
                    split_method="already_independent", split_evidence=relative(proof_path),
                    metadata_notes="完整来源采集框；一张TIFF为一份标本的独立连续切片ROI；约1 µm XY像素尺度来自论文Methods；未把切片顺序写作原生Z轴。")
                receipt = dict(row=row, result=info, pixel_checks=pixels,
                    panel_selection=kept[["source_channel_index", "marker_id"]].to_dict("records"),
                    output_stat=dict(size=destination.stat().st_size, mtime_ns=destination.stat().st_mtime_ns) if info["admitted"] else None)
                write_json(receipt_path, receipt)
                if info["admitted"]:
                    images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
                    save_tables()
                save_checks()
                progress("exporting_native_images", current_image=image_id, panel=pid,
                         admitted=info["admitted"], gaps=["等待剩余导出及独立复核"])
                print(json.dumps(dict(image=image_id, completed=len(images), admitted=info["admitted"])), flush=True)
                new_images += 1
                if args.max_new_images is not None and new_images >= args.max_new_images:
                    stop_batch = True
                    break
            except (OSError, ValueError, AssertionError, zipfile.BadZipFile) as error:
                failures.append(dict(image_id=image_id, error=repr(error)))
                write_json(cache / "export_failures.json", failures)
                progress("image_error_continuing", failures=failures, gaps=["错误图像保留具体失败；继续其他完整成员", "等待独立复核"])
                print(json.dumps(failures[-1]), flush=True)
            finally:
                local_tiff.unlink(missing_ok=True)
    if stop_batch:
        break

if resume_pending:
    raise ValueError(f"Resume image was not found in selected panels: {args.resume_image}")
save_checks()
with formal_read_lease(root / "raw/public" / STUDY / "metadata.json", data_root=root):
    receipts = [json.loads(path.read_text()) for path in (cache / "receipts").glob("*.json")]
excluded = [r["row"]["image_id"] for r in receipts if not r["result"]["admitted"]]
write_json(cache / "delivery_summary.json", dict(study=STUDY, expected_images=312,
    exported_images=len(images), excluded_all_marker_zero=excluded, failures=failures,
    sample_rows=3, sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=3,
    per_panel=images.groupby("panel_id").size().to_dict(), pixel_checked_images=sum(bool(r["pixel_checks"]) for r in receipts),
    independent_review="pending", controller_requests=controller_requests))
progress("prepared_waiting_independent_review" if len(images) + len(excluded) == 312 and not failures else "partial_batch_complete",
    failures=failures, excluded_all_marker_zero=excluded, gaps=["不同CLI独立复核及D01绑定待主控安排"] if len(images) == 312 else ["未输出范围见receipt及失败记录", "等待独立复核"])
print(json.dumps(dict(exported_images=len(images), expected_images=312, failures=failures, excluded=excluded)), flush=True)
