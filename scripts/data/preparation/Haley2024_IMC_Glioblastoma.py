"""Deliver Haley GBM's 24 admitted physical cores at published float32 intensity.

Inputs: complete images.zip, native verification, ordered source panel, S1,
Cases.xls (ROIs/Cases), README and article XML in this study's metadata cache.
Excel decoding uses the project-local xlrd 2.0.2 runtime dependency.
Outputs: independent CYX input.tiff files and the five study input tables.
Cases.xls supplies ROI-to-case links; three cores from one case share a sample.
The published Core/Edge disagreement is retained explicitly for adjudication.
Run pixel export through the batch shared formal write lease wrapper.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import csv
import io
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Haley2024_IMC_Glioblastoma/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import ImageReader, SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import default_io_root, formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables, validate_image_metadata


STUDY = "Haley2024_IMC_Glioblastoma"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--repair-region-conflict", action="store_true")
parser.add_argument("--repair-evidence-dir", type=Path)
parser.add_argument("--repair-before-versions", type=Path)
parser.add_argument("--source-overviews-only", action="store_true", help="Read three native planes per core into small QC overviews; use the read slot")
parser.add_argument("--recover-existing-only", action="store_true", help="Validate and index already-written TIFFs without a receipt; use the read slot")
parser.add_argument("--limit", type=int, help="Maximum new images in this invocation")
parser.add_argument("--progress", type=Path, default=PROJECT / ".cache/cellweft/runtime/jobs/Haley2024_IMC_Glioblastoma/0039_prepare_0/progress.json")
args = parser.parse_args()


def repair_region_conflict() -> None:
    if args.repair_evidence_dir is None or args.repair_before_versions is None:
        parser.error("region repair requires --repair-evidence-dir and --repair-before-versions")
    data = args.data_root
    study_output = data / "processed" / STUDY
    source = metadata_source_directory(data, STUDY) / "source_metadata"
    sys.path.insert(0, str(PROJECT / ".cache/cellweft/runtime/dependencies/Haley2024_IMC_Glioblastoma/python_deps"))
    import xlrd
    private = args.repair_evidence_dir
    private.mkdir(parents=True, exist_ok=True)
    image_path = study_output / "images.parquet"
    field_path = study_output / "field_mapping.csv"
    quality_path = study_output / "annotation_evidence/region_annotation_quality.json"
    evidence_path = f"processed/{STUDY}/annotation_evidence/region_annotation_quality.json"
    baseline = json.loads(args.repair_before_versions.read_text(encoding="utf-8"))

    def version(path: Path) -> dict:
        info = path.stat()
        return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}

    def check_untouched() -> None:
        assert [version(Path(item["path"])) for item in baseline["unmodified"]] == baseline["unmodified"]

    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    check_untouched()
    assert not baseline["quality_record_existed"]
    current_images = pd.read_parquet(image_path)
    with field_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        field_columns = reader.fieldnames
        current_rules = list(reader)
    assert field_columns is not None
    if quality_path.exists():
        assert len(current_rules) == 18
        assert [row["target_field"] for row in current_rules[16:]] == [
            "region_annotation_status", "region_annotation_evidence_path"]
        assert len(current_images) == 24
        assert current_images.region_annotation_status.eq("source_conflict").all()
        assert current_images.region_annotation_evidence_path.eq(evidence_path).all()
        assert json.loads(quality_path.read_text())["affected_image_ids"] == current_images.image_id.tolist()
        check_untouched()
        print(json.dumps({"affected": 24, "state": "already_published_no_rewrite"}))
        return
    assert version(image_path) == baseline["images"]
    assert version(field_path) == baseline["field_mapping"]
    assert len(current_images) == 24 and current_images.image_id.is_unique
    assert len(current_rules) == 16
    assert "region_annotation_status" not in current_images and "region_annotation_evidence_path" not in current_images

    book = xlrd.open_workbook(str(source / "Cases.xls"))
    sheet = book.sheet_by_name("ROIs")
    headers = sheet.row_values(0)
    rois = [dict(zip(headers, sheet.row_values(index))) for index in range(1, sheet.nrows)]
    assert len(rois) == 24
    roi_classes = {str(row["ROI"]): str(row["Histology_class"]) for row in rois}
    assert len(roi_classes) == 24 and set(roi_classes.values()) == {"Core", "Edge"}
    workbook_cases = {}
    for row in rois:
        workbook_cases.setdefault(str(int(row["Case"])), Counter())[str(row["Histology_class"])] += 1
    article = ElementTree.parse(source / "article.xml")
    table = next(item for item in article.iter("table-wrap") if item.attrib.get("id") == "T1")
    article_cases = {}
    for row in table.findall(".//tbody/tr"):
        cells = ["".join(cell.itertext()).strip() for cell in row]
        article_cases[cells[0]] = {"Edge": int(cells[7]), "Core": int(cells[8])}
    assert len(workbook_cases) == 8 and set(workbook_cases) == set(article_cases)
    assert all(dict(workbook_cases[key]) != article_cases[key] for key in workbook_cases)
    workbook_counts = Counter()
    article_counts = Counter()
    for key in workbook_cases:
        workbook_counts.update(workbook_cases[key])
        article_counts.update(article_cases[key])
    assert dict(workbook_counts) == {"Core": 15, "Edge": 9}
    assert dict(article_counts) == {"Core": 9, "Edge": 15}
    assert set(current_images.core_id.astype(str)) == set(roi_classes)
    assert all(str(row.region_annotation) == roi_classes[str(row.core_id)] for row in current_images.itertuples())

    staged_images = current_images.copy()
    staged_images["region_annotation_status"] = pd.Series("source_conflict", index=staged_images.index, dtype="string")
    staged_images["region_annotation_evidence_path"] = pd.Series(evidence_path, index=staged_images.index, dtype="string")
    source_file = ";".join(str(metadata_source_relative(source / name, data)) for name in ("Cases.xls", "article.xml"))

    def rule(field: str, definition: str, transformation: str) -> dict:
        row = dict.fromkeys(field_columns, "")
        row.update(study=STUDY, source_file=source_file,
                   source_sheet_or_table="Cases.xls:ROIs;article.xml:Table1",
                   source_field="ROIs.Histology_class;Table1.Edge;Table1.Core",
                   target_table="image_metadata", target_field=field,
                   join_keys="images.core_id=Cases.xls.ROIs.ROI;ROIs.Case=article.xml.Table1.case",
                   transformation_rule=transformation,
                   conflict_rule="Preserve ROIs.Histology_class verbatim in region_annotation; do not adjudicate Core/Edge biological class from contradictory aggregates.",
                   notes=json.dumps({"data_type": "string", "definition": definition}))
        return row

    new_rules = [
        rule("region_annotation_status", "Source classification conflict status for this image; source_conflict means the source ROI label is transcribed but biological Core/Edge classification remains unresolved.", "Set source_conflict for each exact ROIs.ROI match; Table1 case-level Core/Edge counts disagree for every matched case."),
        rule("region_annotation_evidence_path", "Data-root-relative path to the source classification conflict quality record for this image.", f"Set {evidence_path} for each exact ROIs.ROI match."),
    ]
    quality = {
        "study": STUDY, "status": "unresolved_source_conflict",
        "source_versions": [version(source / "Cases.xls"), version(source / "article.xml")],
        "source_claims": {"Cases.xls ROIs.Histology_class": dict(workbook_counts),
                          "article.xml Table1": dict(article_counts),
                          "conflicting_case_count": 8, "source_annotation_transcription_errors": 0},
        "affected_image_ids": staged_images.image_id.tolist(), "affected_count": 24,
        "scope": "Optional biological Core/Edge classification only; independently documented ROI identity, sample/patient linkage and native geometry are unaffected by the label discrepancy.",
        "use_restrictions": ["Do not treat either source's Core/Edge label as a verified biological class.",
                             "Exclude these labels from Core/Edge stratification, class-conditioned sampling and class-specific biological conclusions until independent resolution."],
        "independent_review": "I04 and D01 remain pending independent post-repair review and current binding.",
    }
    image_stage = private / "staged_images.parquet"
    field_stage = private / "staged_field_mapping.csv"
    quality_stage = private / "staged_region_annotation_quality.json"
    staged_images.to_parquet(image_stage, index=False)
    with field_stage.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_columns)
        writer.writeheader()
        writer.writerows(current_rules + new_rules)
    write_json(quality_stage, quality)
    verified_images = pd.read_parquet(image_stage)
    pd.testing.assert_frame_equal(verified_images[current_images.columns], current_images, check_exact=True)
    assert verified_images.columns.tolist() == current_images.columns.tolist() + ["region_annotation_status", "region_annotation_evidence_path"]
    assert all(str(verified_images[key].dtype) == "string" for key in ("region_annotation_status", "region_annotation_evidence_path"))
    with field_stage.open(newline="", encoding="utf-8-sig") as handle:
        verified_rules = list(csv.DictReader(handle))
    assert verified_rules[:16] == current_rules and len(verified_rules) == 18
    assert all(json.loads(row["notes"])["data_type"] == "string" for row in verified_rules[16:])
    validated = validate_image_metadata(verified_images, pd.read_csv(study_output / "sample_metadata.csv"),
                                        pd.read_csv(study_output / "panel_mapping.csv"), pd.read_csv(field_stage))
    assert len(validated) == 24
    assert json.loads(quality_stage.read_text()) == quality
    check_untouched()
    assert version(image_path) == baseline["images"] and version(field_path) == baseline["field_mapping"]
    assert not quality_path.exists()

    from cellweft.data.io.publication import resource_lock
    from cellweft.data.preparation.io_coordination import formal_write_lease

    def publish(stage: Path, target: Path, expected: dict | None) -> None:
        with formal_write_lease(target, io_lock_root=default_io_root(data)):
            with resource_lock(target.parent):
                check_untouched()
                if expected is None:
                    assert not target.exists()
                else:
                    assert version(target) == expected
                pending = target.with_name(target.name + ".region_repair_pending")
                with stage.open("rb") as reader, pending.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(pending, target)
                descriptor = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

    publish(quality_stage, quality_path, None)
    publish(field_stage, field_path, baseline["field_mapping"])
    publish(image_stage, image_path, baseline["images"])
    check_untouched()
    assert pd.read_parquet(image_path).equals(verified_images)
    with field_path.open(newline="", encoding="utf-8-sig") as handle:
        assert list(csv.DictReader(handle)) == verified_rules
    assert json.loads(quality_path.read_text()) == quality
    result = {"24affected": 24, "old_values_unchanged": True, "2columns_added": True,
              "2field_rules_added": True, "TIFF_and_unaffected_tables_unchanged": True,
              "round1": "published_pending_independent_review", "D01_written": False,
              "tool_sessions_closed": True}
    write_json(private / "validation.json", {"source_roi_matches": 24, "source_conflicting_cases": 8,
              "source_aggregate": {"workbook": dict(workbook_counts), "paper": dict(article_counts)},
              "old_values_unchanged": True, "old_16_field_rules_unchanged": True,
              "validate_image_metadata_accepted": True, "unaffected_versions_unchanged": True})
    write_json(private / "repair_receipt.json", {"study": STUDY, "round": 1,
              "state": result["round1"], "before_versions": str(args.repair_before_versions),
              "after_versions": {"images": version(image_path), "field_mapping": version(field_path),
                                 "quality": version(quality_path)}, "D01_written": False})
    write_json(private / "final_result.json", result)
    print(json.dumps({"affected": 24, "state": result["round1"]}))


if args.repair_region_conflict:
    repair_region_conflict()
    raise SystemExit(0)

root = args.data_root
metadata = metadata_source_directory(root, STUDY)
archive = root / "raw/public" / STUDY / "public_original/images.zip"
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
evidence = metadata / "source_metadata"
staging_root = os.environ.get("CELLWEFT_INPUT_STAGING_ROOT")
planes = Path(staging_root) / "haley_native_planes" if staging_root else cache / "native_planes"
receipts = cache / "image_receipts"
output = root / "processed" / STUDY
for directory in (cache, planes, receipts, output):
    directory.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT / ".cache/cellweft/runtime/dependencies/Haley2024_IMC_Glioblastoma/python_deps"))
import xlrd


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def save_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


GAPS = ["Cases.xls Histology_class 的 Core/Edge 计数与论文表1相反；新版edge_core.csv下载返回401/403，区域注释待主控裁定。",
        "整理者自检已记录；独立复核及D01验收待不同CLI上下文完成。"]
CONTROLLER_REQUESTS = [
    "按annotation_conflicts.json裁定Core/Edge注释；若获取新版edge_core.csv可仅修订metadata。",
    "由不同CLI上下文独立复核后填写D01绑定；当前未签发独立验收。",
    "请修复共享read_tiff_window对OME TiffFrame的支持（缺samplesperpixel）；本study先用aspage()适配，后续迁回共享读取。"]


def progress(stage: str, count: int, **details: object) -> None:
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.progress, dict(study=STUDY, stage=stage, completed_images=count,
                                 expected_images=24, updated_utc=datetime.now(timezone.utc).isoformat(),
                                 gaps=GAPS, controller_requests=CONTROLLER_REQUESTS, **details))


# Reconcile all archive members with the accepted physical scope and old proofs.
manifest = json.loads((metadata / "source_manifest.json").read_text())
verification = json.loads((metadata / "native_verification.json").read_text())
ordered = json.loads((metadata / "source_marker_order_evidence.json").read_text())["ordered_channels_by_folder"]
native = json.loads((metadata / "native_markers.json").read_text())["panels"]
selected = manifest["scope_selector"]["selected_folders"]
proofs = {row["physical_image_id"].split("=", 1)[1]: row["native_proof"] for row in verification["records"]}
native_names = {row["image_id"].split("=", 1)[1]: row["ordered_source_labels"] for row in native}
stat = archive.stat()
assert len(selected) == 24 and set(selected) == set(proofs)
assert verification["scope_complete"] and verification["native_result"]["actual_n_images"] == 24
source_names = native_names[selected[0]]
panel_id = f"{STUDY}__native40"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(source_names) == 40 and len(kept) == 39
assert panel.loc[panel.nuclear, "source_name"].tolist() == ["DNA1", "DNA3"]
assert panel.loc[~panel.keep_channel, "source_name"].tolist() == ["pERK"]
inventory = []
with zipfile.ZipFile(archive) as source_zip:
    members = {item.filename: item for item in source_zip.infolist()}
    for core in selected:
        proof = proofs[core]
        assert stat.st_size == proof["source_file_stat"]["size"]
        assert stat.st_mtime_ns == proof["source_file_stat"]["mtime_ns"]
        assert proof["complete"] and proof["native_order_verified"]
        assert source_names == native_names[core] == [item["target"] for item in ordered[core]]
        names = [item["archive_member"] for item in ordered[core]]
        assert names == proof["source_member_names"]
        for item in proof["source_member_stats"]:
            actual = members[item["name"]]
            assert (actual.file_size, actual.compress_size, actual.CRC, actual.header_offset) == (
                item["file_size"], item["compress_size"], item["CRC"], item["header_offset"])
        assert set(names) == {name for name in members if name.startswith(core + "/") and not members[name].is_dir()}
        records = proof["native_read_evidence"]["channel_records"]
        assert all(item["shape"] == [1500, 1500] and item["dtype"] == "float32" for item in records)
        inventory.append(dict(core=core, image_id=proof["image_id"], members=names,
                              uncompressed_bytes=sum(members[name].file_size for name in names),
                              shape=[40, 1500, 1500], dtype="float32", native_proof=relative(Path(proof["proof_path"]))))
    excluded = [name for name in members if name.startswith("Tonsil/") and not members[name].is_dir()]
    assert len(excluded) == 40 and len(members) == 1025
source_inventory = dict(
    source_archive=relative(archive), archive_bytes=stat.st_size, archive_mtime_ns=stat.st_mtime_ns,
    reused_verification=relative(metadata / "native_verification.json"), selected=inventory,
    archive_member_count=len(members), selected_file_count=960, excluded_file_count=len(excluded),
    exclusion_reason="Existing admission: Tonsil diagnostic context unresolved; no new scope decision.",
    extraction_bytes=sum(item["uncompressed_bytes"] for item in inventory),
    maximum_uncompressed_output_bytes=24 * 39 * 1500 * 1500 * 4,
    peak_per_image_ram_bytes_estimate=4 * 1536 * 1536 * 8,
    verification_reuse="Archive size/mtime and ZIP member metadata agree with accepted native proof; no full hash repeated.")
if not args.source_overviews_only:
    save_json(cache / "source_inventory.json", source_inventory)


def save_overview(preview: np.ndarray, destination: Path) -> None:
    """Display DNA1/GFAP/CD45 without changing delivered source intensities."""
    rgb = np.moveaxis(preview[:, ::5, ::5], 0, -1).astype(np.float32)
    for channel in range(3):
        low, high = np.percentile(rgb[:, :, channel], [1, 99.5])
        rgb[:, :, channel] = np.clip((rgb[:, :, channel] - low) / (high - low), 0, 1)
    destination.parent.mkdir(exist_ok=True)
    plt.imsave(destination, rgb)


def read_output_window(tif: tifffile.TiffFile, index: int, window: tuple[int, int, int, int]) -> np.ndarray:
    """Temporarily materialize OME TiffFrame metadata for the shared decoder.

    The first export exposed shared read_tiff_window's TiffPage-only attribute
    access. aspage() supplies that metadata while preserving bounded tile reads.
    Controller request: migrate this handling into the shared TIFF window reader.
    """
    page = tif.pages[index]
    if isinstance(page, tifffile.TiffFrame):
        page = page.aspage()
    view = SimpleNamespace(pages={index: page}, filehandle=tif.filehandle)
    return read_tiff_window(view, index, window)


def recover_geometry(reader: ImageReader, destination: Path) -> dict:
    """Recover the first complete TIFF after output-QC reader failure, without rewriting it.

    Only one1500x1500plane is decoded at a time. Header and source-pixel checks
    below still run before this existing TIFF is added to the image index.
    """
    _, height, width = reader.shape
    nonzero, sums = [], []
    with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
        assert tif.series[0].shape == (len(kept), height, width)
        assert tif.series[0].dtype == reader.dtype
        for index in range(len(kept)):
            values = tif.pages[index].asarray(maxworkers=1)
            assert np.isfinite(values).all()
            nonzero.append(int(np.count_nonzero(values)))
            sums.append(float(values.sum(dtype=np.float64)))
    return dict(height_px=height, width_px=width, pad_left_px=0, pad_top_px=0,
                pad_right_px=0, pad_bottom_px=0, source_dtype=str(reader.dtype),
                source_channel_count=reader.shape[0], channel_count=len(kept),
                admitted=bool(np.asarray(nonzero)[(kept.role == "downstream").to_numpy()].any()),
                channel_nonzero_pixels=nonzero, channel_means=(np.asarray(sums)/(height*width)).tolist(),
                valid_pixels=None, reused_existing_tiff=True)


if args.source_overviews_only:
    # Each scalar native plane is only9MB. No TIFF member is extracted to disk;
    # the read-slot operation emits300px display-only PNGs for visual review.
    with zipfile.ZipFile(archive) as source_zip:
        for core in selected:
            overview_path = cache / "source_overviews" / f"{core}.png"
            if overview_path.exists():
                continue
            preview = []
            for name in ("DNA1", "GFAP", "CD45"):
                member = ordered[core][source_names.index(name)]["archive_member"]
                with formal_read_lease(archive):
                    with tifffile.TiffFile(io.BytesIO(source_zip.read(member))) as source_tif:
                        preview.append(source_tif.asarray())
            save_overview(np.stack(preview), overview_path)
    figure, axes = plt.subplots(4, 6, figsize=(12, 8))
    for axis, core in zip(axes.flat, selected):
        axis.imshow(plt.imread(cache / "source_overviews" / f"{core}.png"))
        axis.set_title(core, fontsize=9)
        axis.axis("off")
    figure.suptitle("Source RGB: DNA1 / GFAP / CD45; display scaling only", fontsize=11)
    figure.tight_layout()
    figure.savefig(cache / "source_overviews/contact_sheet.png", dpi=150)
    plt.close(figure)
    print("24 source overviews available; contact sheet saved; no input TIFF exported.", flush=True)
    raise SystemExit(0)

# Recover explicit ROI→case and case→clinical joins from the public workbook.
workbook = xlrd.open_workbook(str(evidence / "Cases.xls"))


def sheet_frame(name: str) -> pd.DataFrame:
    sheet = workbook.sheet_by_name(name)
    return pd.DataFrame([sheet.row_values(i) for i in range(1, sheet.nrows)], columns=sheet.row_values(0))


rois = sheet_frame("ROIs")
cases = sheet_frame("Cases")
rois["Case"] = rois.Case.astype(int).astype(str)
cases["Case"] = cases.Case.astype(int).astype(str)
assert set(rois.ROI) == set(selected) and rois.ROI.is_unique and cases.Case.is_unique
assert len(cases) == 8 and rois.groupby("Case").size().eq(3).all()
article = ElementTree.parse(evidence / "article.xml")
primary_paragraph = next(" ".join("".join(item.itertext()).split()) for item in article.iter("p")
                         if "three cores taken per case" in "".join(item.itertext()))
assert "Eight primary" in primary_paragraph
table = next(item for item in article.iter("table-wrap") if item.attrib.get("id") == "T1")
article_rows = [["".join(cell.itertext()).strip() for cell in row] for row in table.findall(".//tbody/tr")]
article_cases = {row[0]: row for row in article_rows}
conflicts = []
for case, subset in rois.groupby("Case"):
    row = article_cases[case]
    source_case = cases.set_index("Case").loc[case]
    assert (str(int(source_case["Age at surgery"])), source_case.Sex, source_case.Site.strip(), source_case.MGMT) == (
        row[2], row[3], row[1], row[6])
    workbook_counts = {key: int(subset.Histology_class.eq(key).sum()) for key in ("Edge", "Core")}
    paper_counts = {"Edge": int(row[7]), "Core": int(row[8])}
    if workbook_counts != paper_counts:
        conflicts.append(dict(case=case, cases_xls=workbook_counts, paper_table1=paper_counts))
assert len(conflicts) == 8
save_json(cache / "annotation_conflicts.json", dict(
    source_workbook=relative(evidence / "Cases.xls"), source_paper=relative(evidence / "article.xml"),
    conflicts=conflicts, decision="Retain source Histology_class exactly; biological edge/core interpretation unresolved.",
    newer_attachment="https://datadryad.org/downloads/file_stream/3364336",
    access_attempts={"public_file_stream":403,"api_file_download":401},
    impact="Region annotation only; ROI→case and clinical joins agree independently."))
rois.to_csv(cache / "source_rois.csv", index=False, encoding="utf-8-sig")
cases.to_csv(cache / "source_cases.csv", index=False, encoding="utf-8-sig")

sample_rows = []
for case in cases.to_dict("records"):
    case_id = case["Case"]
    row = dict.fromkeys(columns("sample_metadata"))
    row.update(study=STUDY, sample_id=f"case_{case_id}_primary_GBM", patient_key=f"{STUDY}__case_{case_id}",
               source_subject_id=case_id, source_sample_id=f"Case {case_id}",
               biospecimen_group_id=f"case_{case_id}_primary_GBM", cohort_id="Salford_primary_GBM_IMC",
               center_id="Salford Royal Hospital", sample_type="tissue", tissue_type="tumor",
               anatomical_site=case["Site"].strip(), primary_site="brain", lesion_type="primary",
               preservation_method="ffpe", case_control_status="case", primary_diagnosis="Glioblastoma",
               histology="Glioblastoma", sex_reported={"M":"male","F":"female"}[case["Sex"]],
               processing_method="5 um FFPE section; three TMA cores per case; lanthanide-antibody IMC",
               age_at_surgery_years=int(case["Age at surgery"]), idh1_status_reported=case["IDH1"],
               atrx_status_reported=case["ATRX"], mgmt_methylation_status_reported=case["MGMT"],
               metadata_notes="Cases.xls ROIs.Case→Cases.Case; paper Methods explicitly three TMA cores per primary case. "
               "One case-level primary GBM tissue specimen is the biological sample. Age is explicitly at surgery, retained "
               "as an extension; collection/index dates, prior therapy and outcome were not supplied. "
               "Core/Edge labels conflict with paper Table1; this affects image annotation, not case identity.")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)
roi_lookup = rois.set_index("ROI")


def mapping(source_file: Path, source_table: str, source_field: str, target_table: str,
            target_field: str, rule: str, *, notes: str = "", join: str = "") -> dict:
    row = dict.fromkeys(columns("field_mapping"))
    row.update(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_table,
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join, transformation_rule=rule, notes=notes,
               conflict_rule="Preserve explicit source value; missing optional clinical fields remain empty.")
    return row


fields = [
    mapping(evidence / "Cases.xls", "ROIs + Cases", "ROI;Case", "image_metadata", "sample_id",
            "Match archive folder exactly to ROIs.ROI; join ROIs.Case=Cases.Case; case_{Case}_primary_GBM.",
            join="archive folder=ROIs.ROI; ROIs.Case=Cases.Case",
            notes="24/24 images connect to 8 biological samples and 8 source cases. No case inferred from ROI text."),
    mapping(evidence / "article.xml", "Methods: GBM samples; Table1", "three cores taken per case", "sample_metadata", "sample_id",
            "One primary GBM specimen per reported case; all three physical cores share one biological sample.",
            notes=primary_paragraph),
    mapping(evidence / "Cases.xls", "Cases", "Case", "sample_metadata", "patient_key",
            "Prefix explicit public case ID with study; same case shares patient key.", join="ROIs.Case=Cases.Case"),
    mapping(evidence / "Cases.xls", "Cases", "Case", "sample_metadata", "source_subject_id;source_sample_id;biospecimen_group_id",
            "Keep explicit Case as source_subject_id; retain source specimen label Case {Case}; group the three cores under case_{Case}_primary_GBM.",
            join="ROIs.Case=Cases.Case", notes="The case-level specimen identity is supported by the paper's three-cores-per-case sampling description."),
    mapping(evidence / "Cases.xls", "ROIs", "Histology_class", "image_metadata", "region_annotation",
            "Copy exact source string; biological interpretation unresolved because all 8 case-level counts disagree with Table1.",
            join="archive folder=ROIs.ROI", notes="See source_metadata/annotation_conflicts.json; newer edge_core.csv currently returns401/403."),
    mapping(evidence / "Cases.xls", "Cases", "Site", "sample_metadata", "anatomical_site", "Trim trailing whitespace; retain laterality and site."),
    mapping(evidence / "Cases.xls", "Cases", "Sex", "sample_metadata", "sex_reported", "M→male; F→female."),
    mapping(evidence / "Cases.xls", "Cases", "Age at surgery", "sample_metadata", "age_at_surgery_years", "Retain explicit surgical age.",
            notes=json.dumps(dict(data_type="number", definition="Age explicitly reported at surgery", unit="years"))),
    mapping(evidence / "Cases.xls", "Cases", "IDH1", "sample_metadata", "idh1_status_reported", "Retain source status without additional subtype inference.",
            notes=json.dumps(dict(data_type="string", definition="Source-reported IDH1 status"))),
    mapping(evidence / "Cases.xls", "Cases", "ATRX", "sample_metadata", "atrx_status_reported", "Retain source status.",
            notes=json.dumps(dict(data_type="string", definition="Source-reported ATRX status"))),
    mapping(evidence / "Cases.xls", "Cases", "MGMT", "sample_metadata", "mgmt_methylation_status_reported", "Retain source status including Not performed.",
            notes=json.dumps(dict(data_type="string", definition="Source-reported MGMT methylation status"))),
    mapping(evidence / "article.xml", "Methods: GBM samples + IMC tissue staining", "primary GBM; FFPE; three cores; 5 um", "sample_metadata",
            "primary_diagnosis;histology;lesion_type;preservation_method;processing_method;center_id;cohort_id;sample_type;tissue_type;primary_site;case_control_status",
            "All 8 source cases are primary GBM bank specimens; FFPE TMA sections and institution follow Methods."),
    mapping(evidence / "README_v1.md", "Description", "1 um2; denoised 32-bit TIFF total counts", "image_metadata",
            "pixel_size_x_um;pixel_size_y_um;intensity_description", "Use 1 um per axis and preserve published denoised float32 intensity."),
    mapping(metadata / "source_marker_order_evidence.json", "ordered_channels_by_folder", "archive_member;target", "panel_mapping",
            "source_channel_index;source_name", "Use exact per-folder native order, sequential0-based indices; filename ordinal remains in source inventory."),
    mapping(s1_path, "Study=Haley2024_IMC_Glioblastoma", "Original marker", "panel_mapping",
            "marker_id;role;keep_channel;output_channel_index", "Exact study+source label join; retain37 downstream proteins + DNA1/DNA3; exclude pERK."),
    mapping(evidence / "article.xml", "Data and materials availability; Supplementary materials", "Dryad repository; Figs S1-S7", "sample_metadata",
            "metadata_notes", "Source tracing: read local archive/proofs, public Dryad and Zenodo pages, Cases.xls, README, article Methods/Table1/Table2 and repository links. "
            "Supplement is figures S1-S7; no additional clinical table listed. Paper links general processing repositories, no study-specific sample map. "
            "Case/sample identity is complete; collection dates, therapy and survival remain absent optional metadata.")
]
field_mapping = pd.DataFrame(fields, columns=columns("field_mapping"))
field_mapping.loc[field_mapping.target_field.eq("sex_reported"), "value_mapping"] = json.dumps({"M":"male", "F":"female"})
region_quality_path = output / "annotation_evidence/region_annotation_quality.json"
if region_quality_path.exists():
    # Keep the independently documented conflict visible on ordinary I/O resumes.
    repaired_fields = pd.read_csv(output / "field_mapping.csv", keep_default_na=False)
    extension_fields = repaired_fields.loc[repaired_fields.target_field.isin(
        ["region_annotation_status", "region_annotation_evidence_path"])]
    assert len(extension_fields) == 2
    field_mapping = pd.concat([field_mapping, extension_fields], ignore_index=True)


def publish_checks(images: pd.DataFrame) -> None:
    """Publish preparer checks with an empty independent-reviewer field."""
    checks = []
    evidence_path = relative(cache / "source_inventory.json")
    for check, result, details in [
        ("S01", "pass", "Preparer self-check: 960 selected TIFF members in24GBM cores;40Tonsil members excluded by existing admission. Native proofs reused."),
        ("I01", "pass", "Preparer self-check: unique exact archive folder identities; each source/output path is recorded."),
        ("I02", "pass", f"Preparer self-check: {len(images)}/{len(images)} indexed images link to8defined samples; panel foreign keys resolve."),
        ("I03", "pass", "Preparer self-check: full1500x1500source frame, native1um pixels, square without added padding."),
        ("I04", "unresolved", "All24 ROI→case identities confirmed. Original Histology_class copied verbatim; Edge/Core biological assignment conflicts with paper Table1. See annotation_conflicts.json."),
        ("P01", "pass", "Preparer self-check:40ordered source rows match each folder, native proofs and exact study-specific S1 labels."),
        ("P02", "pass", "Preparer self-check:37unique downstream measurements + DNA1/DNA3 retained; pERK excluded per S1."),
        ("P03", "pass", f"Preparer self-check: all{len(images)}completed TIFFs reopened;39x1500x1500float32; OME names match panel order."),
        ("I05", "pass", "Preparer self-check: every exported image compared at internal(700,700,732,732) and boundary(1468,1468,1500,1500) blocks; all39channels and dtype equal source. Image receipts contain actual results."),
        ("M01", "pass", f"Preparer self-check: Cases.xls ROIs.ROI→ROIs.Case→Cases.Case; paper explicitly three cores per case. {len(images)}image links;8sample rows;8/8patient-linked samples. No identity inferred from filenames."),
        ("M02", "pass", "Preparer self-check: all8clinical records joined on explicit Case; age/sex/site/MGMT independently agree with paper Table1; source IDH1/ATRX retained. Surgical age is an extension. Missing dates/therapy/outcomes remain empty."),
        ("D01", "unresolved", f"Preparer handoff: {len(images)}/24native images. Independent reviewer and controller acceptance are pending; no input_binding pass is issued. Annotation conflict remains explicit.")
    ]:
        if check == "S01":
            visual_path = cache / "source_visual_review.json"
            if visual_path.exists():
                visual = json.loads(visual_path.read_text())
                details += " Actual preparer visual review: " + ",".join(visual["reviewed_cores"]) + "; " + relative(visual_path)
            else:
                result, details = "unresolved", details + " Representative source overview inspection is pending."
        if check in {"S01", "D01"}:
            scope, identities = "study", [STUDY]
        elif check.startswith("M"):
            scope, identities = "sample", samples.sample_id.tolist()
        elif check in {"P01", "P02"}:
            scope, identities = "panel", [panel_id]
        else:
            scope, identities = "image", images.image_id.tolist()
        for identity in identities:
            record_details = details
            if check == "M01":
                linked = int(images.sample_id.eq(identity).sum())
                record_details += f" Current sample {identity}: {linked} indexed images; explicit case/patient identity confirmed."
            checks.append(dict(study=STUDY, preparer="Codex CLI 0039_prepare_0", reviewer="", scope=scope,
                               record_id=identity, check_id=check, result=result,
                               source_evidence=evidence_path + ";" + relative(evidence / "Cases.xls"),
                               output_evidence=relative(output / "images.parquet"), details=record_details))
    review_path = output / "review_results.csv"
    with formal_publish_lease(review_path, directory=review_path.parent):
        pd.DataFrame(checks, columns=columns("review_results")).to_csv(review_path, index=False, encoding="utf-8-sig")


images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
if region_quality_path.exists():
    images["region_annotation_status"] = pd.Series("source_conflict", index=images.index, dtype="string")
    images["region_annotation_evidence_path"] = pd.Series(relative(region_quality_path), index=images.index, dtype="string")
save_input_tables(output, images, samples, panel, field_mapping)
publish_checks(images)
progress("metadata_ready" if args.metadata_only else "exporting", len(images), sample_rows=8, patient_linked_samples=8)

# Export one physical source ROI at a time and publish its resumable index.
new_images = 0
if not args.metadata_only and not args.recover_existing_only:
    missing_planes = sum(members[name].file_size for item in inventory for name in item["members"]
                         if not (planes / name).exists())
    pending_output = sum(39 * 1500 * 1500 * 4 for item in inventory
                         if not (receipts / f"{item['core']}.json").exists())
    if shutil.disk_usage(root).free < 100 * 1024**3 + missing_planes + pending_output:
        progress("insufficient_capacity", len(images), required_new_bytes=missing_planes + pending_output)
        raise SystemExit("Native planes and TIFF outputs would breach the100GiB NAS reserve.")
if not args.metadata_only:
    with zipfile.ZipFile(archive) as source_zip:
        for item in inventory:
            core = item["core"]
            image_id = item["image_id"]
            destination = output / "images" / image_id / "input.tiff"
            if args.recover_existing_only and not destination.exists():
                continue
            receipt_path = receipts / f"{core}.json"
            if receipt_path.exists() and destination.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["source_archive_size"] == stat.st_size and receipt["source_archive_mtime_ns"] == stat.st_mtime_ns
                assert receipt["retained_names"] == kept.marker_id.tolist()
                assert receipt["output_file_stat"] == dict(size=destination.stat().st_size, mtime_ns=destination.stat().st_mtime_ns)
                image_row = receipt["image_metadata"]
            else:
                if args.limit is not None and new_images >= args.limit:
                    break
                progress("extracting", len(images), current_image=image_id)
                for member in item["members"]:
                    path = planes / member
                    expected = members[member].file_size
                    if path.exists() and path.stat().st_size == expected:
                        continue
                    if args.recover_existing_only:
                        raise ValueError("Recover-existing mode requires complete cached source planes; use the write slot to extract missing planes.")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending = path.with_suffix(path.suffix + ".partial")
                    with formal_read_lease(archive):
                        with source_zip.open(member) as source, pending.open("wb") as target:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
                    assert pending.stat().st_size == expected
                    pending.replace(path)
                paths = tuple(planes / name for name in item["members"])
                spec = SourceSpec(image_id=image_id, panel_id=panel_id, kind="tiff_planes", path=paths[0],
                                  paths=paths, channels=(0,) * 40, pixel_size_um=(1.,1.),
                                  source_revision="Zenodo11073042 v1 / Dryad10.5061/dryad.2z34tmprw")
                progress("exporting", len(images), current_image=image_id)
                with open_source(spec) as reader:
                    assert reader.shape == (40,1500,1500) and reader.dtype == np.dtype("float32")
                    # Native TIFF is one uncompressed strip per plane. A1536tile
                    # reads each source plane once and bounds RAM to one channel.
                    geometry = recover_geometry(reader, destination) if destination.exists() else export_input(reader, panel, destination, tile_size=1536)
                    assert geometry["admitted"]
                    comparisons = []
                    with formal_read_lease(destination), tifffile.TiffFile(destination) as saved:
                        assert saved.series[0].shape == (39,1500,1500) and saved.series[0].dtype == np.dtype("float32")
                        xml = ElementTree.fromstring(saved.ome_metadata)
                        saved_names = [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")]
                        assert saved_names == kept.marker_id.tolist()
                        for window in [(700,700,732,732),(1468,1468,1500,1500)]:
                            original = reader.read_window(kept.source_channel_index.tolist(), window)
                            result = np.stack([read_output_window(saved, i, window) for i in range(39)])
                            assert original.dtype == result.dtype and np.array_equal(original,result)
                            marker_values = original[(kept.role == "downstream").to_numpy()]
                            comparisons.append(dict(window=list(window), all_retained_channels=39, exact_equal=True,
                                                    downstream_nonzero_values=int(np.count_nonzero(marker_values))))
                    # Display-only scaling; saved input intensities are unchanged.
                    overview_channels = [source_names.index(name) for name in ("DNA1","GFAP","CD45")]
                    preview = reader.read_window(overview_channels,(0,0,1500,1500))
                    overview_path=cache / "overviews" / f"{core}.png"
                    save_overview(preview,overview_path)
                source_roi=roi_lookup.loc[core]
                case_id=source_roi.Case
                image_row=dict.fromkeys(columns("image_metadata"))
                image_row.update(study=STUDY,image_id=image_id,sample_id=f"case_{case_id}_primary_GBM",panel_id=panel_id,
                                 image_kind="tma_core",source_image_id=core,source_files=[relative(archive)],
                                 source_format="TIFF planes in ZIP",source_axes="CYX",source_dtype="float32",
                                 source_revision="Zenodo11073042 v1; Dryad10.5061/dryad.2z34tmprw",
                                 core_id=core,roi_id=core,region_annotation=source_roi.Histology_class,
                                 source_x0_px=0,source_y0_px=0,source_x1_px=1500,source_y1_px=1500,
                                 pad_left_px=0,pad_top_px=0,pad_right_px=0,pad_bottom_px=0,height_px=1500,width_px=1500,
                                 source_channel_count=40,channel_count=39,pixel_size_x_um=1.,pixel_size_y_um=1.,
                                 input_image_path=relative(destination),split_method="already_independent",
                                 intensity_description="Published IMC-Denoise float32 metal counts per1um2pixel; values preserved without normalization.",
                                 split_evidence=relative(evidence / "Cases.xls") + "#ROIs;" + relative(evidence / "article.xml") + "#IMC tissue staining: each TMA core imaged as separate ROI",
                                 metadata_notes="Full published ROI frame. Ordered native member names in cache source_inventory.json. "
                                 "region_annotation is exact Cases.xls Histology_class; conflicting Edge/Core paper counts unresolved.")
                save_json(receipt_path,dict(study=STUDY,core=core,image_metadata=image_row,geometry=geometry,
                                           source_archive_size=stat.st_size,source_archive_mtime_ns=stat.st_mtime_ns,
                                           output_file_stat=dict(size=destination.stat().st_size,mtime_ns=destination.stat().st_mtime_ns),
                                           retained_names=kept.marker_id.tolist(),pixel_comparisons=comparisons,
                                           overview=relative(overview_path)))
                new_images+=1
            if region_quality_path.exists():
                image_row = {**image_row, "region_annotation_status": "source_conflict",
                             "region_annotation_evidence_path": relative(region_quality_path)}
            images=images.loc[images.image_id != image_id]
            images=pd.concat([images,pd.DataFrame([image_row])],ignore_index=True)
            save_input_tables(output,images,samples,panel,field_mapping)
            publish_checks(images)
            progress("exporting",len(images),current_image=image_id,sample_rows=8,
                     sample_linked_images=int(images.sample_id.notna().sum()),patient_linked_samples=8)
            print(json.dumps(dict(completed_images=len(images),image_id=image_id),ensure_ascii=False),flush=True)
            if staging_root and (planes / core).is_dir():
                shutil.rmtree(planes / core)

progress("prepared" if len(images)==24 else "partial",len(images),sample_rows=8,
         sample_linked_images=int(images.sample_id.notna().sum()),patient_linked_samples=8,
         output_root=relative(output))
save_json(cache / "preparation_summary.json",dict(study=STUDY,expected_images=24,exported_images=len(images),
    sample_rows=8,sample_linked_images=int(images.sample_id.notna().sum()),patient_linked_samples=8,
    status="prepared" if len(images)==24 else "partial",gaps=GAPS,
    controller_requests=CONTROLLER_REQUESTS,
    shared_module_adaptations=["read_output_window materializes TiffFrame.aspage() metadata before invoking the shared bounded decoder; migrate to shared read_tiff_window."]))
