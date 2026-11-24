"""Prepare the accepted Zenodo 15242478 native IMC subset, without scaling.

Inputs: 18 unchanged MCD files, 56 accepted acquisition/core proofs and S1.
Outputs: square native float32 inputs, five metadata tables, per-image receipts
and preparer checks. Patient identities are never inferred from ROI numbers.
Run with PYTHONPATH=src and project .venv-data/bin/python -B; --job records
the active controller checkpoint. Re-running merges tables and reuses receipts.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

pa.set_cpu_count(3)
pa.set_io_thread_count(1)
STUDY = "Zhu2025_IMC_TNBC_Disparity"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
OUTPUT = ROOT / "processed" / STUDY
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", type=Path, required=True)
parser.add_argument("--verify-only", action="store_true")
parser.add_argument("--index-receipts-only", action="store_true",
                    help="Recover already closed image receipts and verify; do not export new images")
args = parser.parse_args()
PREPARER = args.job.name
for directory in (CACHE, CACHE / "receipts", OUTPUT, args.job):
    directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / "cache"):
        return path.as_posix()
    return str(metadata_source_relative(path, ROOT))


def progress(status: str, **values: object) -> None:
    write_json(args.job / "progress.json", dict(study=STUDY, phase="prepare", scientific_repair_round=0,
               status=status, exported_images=len(images), expected_images=len(plan),
               sample_rows=len(samples), sample_linked_images=sum(bool(x.get("sample_id")) for x in images),
               patient_linked_samples=0, **values))


def interrupted(signum: int, frame: object) -> None:
    """Close active contexts and retain per-image publication checkpoints."""
    progress("waiting_io", reason=f"Execution interrupted by signal {signum}",
             resume="Rerun the same command; only images without completed receipts require export",
             publication_checkpoint_root="/tmp/study_input_batch/publish_checkpoints/" + str(OUTPUT).lstrip("/"))
    raise InterruptedError(f"Signal {signum}; resumable progress saved")


signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)


def review() -> None:
    checks = pd.read_csv(TEMPLATES / "review_checks.csv", keep_default_na=False)
    rows = []
    for check in checks.itertuples():
        detail = "整理者自检证据已保存；等待不同任务身份独立复核。"
        if check.check_id == "M01":
            detail = "BSW2 Validation十份整切片sample按官方MCD basename建立sample；Zenodo明确AA/EA为群体标签，但未定义数字后缀为患者ID，故patient_key和source_subject_id留空。57例clinical_attributes属于BSW-Discovery，Supplementary Tables 1-2分别为Discovery和Roswell Park汇总，均未映射到BSW2。Discovery五个物理core仍无sample/患者crosswalk，study范围M01保持未决。"
        if check.check_id == "M02":
            detail = "仅填写有来源支持的样本类型、平台、疾病、FFPE、队列和文件名群体标签；年龄、性别、分期、治疗和生存值留空。不得将Discovery或Roswell患者级/汇总值转填至BSW2。"
        if check.check_id == "D01":
            detail = f"实际导出{len(images)}/{len(plan)}；sample连接{sum(bool(x.get('sample_id')) for x in images)}图；10份BSW2 sample中患者连接0份。Discovery五个core的sample关联仍未决；独立复核和验收待办。"
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                         record_id=STUDY, check_id=check.check_id, result="unresolved",
                         source_evidence=relative(META / "selected_subset_closeout_20261024.json"),
                         output_evidence=relative(CACHE), details=detail))
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        temporary = OUTPUT / "review_results.csv.partial"
        pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, OUTPUT / "review_results.csv")


# Reuse the accepted bounded native proofs; source XYZ are coordinate fields,
# while readimc and SourceReader expose only the subsequent measured channels.
subset = json.loads((META / "selected_subset_closeout_20261024.json").read_text())
assert subset["qualified_usable_subset"] and subset["n_images"] == 56
plan, panels, layouts = [], [], {}
for selected in subset["selected_images"]:
    proof = json.loads((ROOT / "raw/public" / STUDY / "manifest.json").read_text())["preparation_sources"]["native_images"][selected["acquisition_id"]]
    path = ROOT / selected["source_file"]
    stat = path.stat()
    assert proof["complete"] and proof["native_order_verified"] and proof["all_required_proteins_decoded"]
    assert stat.st_size == proof["source_file_stat"]["size"]
    assert stat.st_mtime_ns == proof["source_file_stat"]["mtime_ns"]
    raw_names = [x["source_label"] for x in selected["source_channel"]]
    assert raw_names[:3] == ["X", "Y", "Z"]
    names = raw_names[3:]
    layout = tuple(names)
    if layout not in layouts:
        panel_id = f"native_{len(layouts)+1:02d}"
        layouts[layout] = panel_id
        panel = panel_from_s1(STUDY, panel_id, names, S1, aliases={"": "<Unnamed channel>"})
        panel["mcd_record_field_index"] = panel.source_channel_index + 3
        panels.append(panel)
    identity = selected["acquisition_id"].split(":", 1)[1]
    specimen = path.stem if proof["source_group"] == "BSW-Validation" else ""
    boundary = proof["data_boundary"]
    plan.append(dict(image_id=identity, panel_id=layouts[layout], source_file=selected["source_file"],
                     acquisition_id=proof["acquisition_id"], width=boundary["width"], height=boundary["height"],
                     proof=str(ROOT / "raw/public" / STUDY / "manifest.json") + "#preparation_sources/native_images/" + selected["acquisition_id"], source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                     sample_id=specimen, source_group=proof["source_group"],
                     core_id=proof["physical_identity"].get("physical_id") or "",
                     description=proof["description"],
                     step_x=float(proof["native_metadata"]["AblationDistanceBetweenShotsX"]),
                     step_y=float(proof["native_metadata"]["AblationDistanceBetweenShotsY"])))
panel = pd.concat(panels, ignore_index=True)
samples = pd.read_csv(TEMPLATES / "sample_metadata.csv")
sample_rows = []
for specimen in dict.fromkeys(x["sample_id"] for x in plan if x["sample_id"]):
    row = dict.fromkeys(samples.columns, "")
    source_group = specimen.rsplit("-", 1)[1][:2]
    if source_group not in {"AA", "EA"}:
        raise ValueError(f"Unexpected BSW-Validation group token: {source_group}")
    group_label = {"AA": "Black American", "EA": "White American"}[source_group]
    row.update(study=STUDY, sample_id=specimen, source_sample_id=specimen,
               cohort_id="BSW-Validation", sample_type="tumor tissue section", tissue_type="breast",
               anatomical_site="breast", preservation_method="FFPE", primary_diagnosis="triple-negative breast cancer",
               molecular_subtype="TNBC", metadata_notes=f"Official Zenodo filename group token {source_group} maps to {group_label}; the numeric suffix is not defined as a patient ID. The paper describes ten whole-section BSW2 patients (five per group). The 57-patient clinical_attributes table is BSW-Discovery; the checked supplement has no individual BSW2 clinical table. No BSW2 patient key, age, sex, stage, treatment, or outcome value is assigned.")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=samples.columns)
fields = []
for source, origin, target, columns, rule in [
    (META / "selected_subset_closeout_20261024.json", "selected_images; native_proof", "image_metadata", "image_id;source_files;source_acquisition_id;core_id;roi_id;split_evidence", "Use only 56 accepted native units; reuse physical-core identity and selected original acquisition; all other acquisitions remain outside this subset."),
    (META / "native_verification.json", "MCD native metadata;ordered channels", "image_metadata", "source_channel_count;source_axes;source_dtype;source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um", "MCD records start with XYZ coordinate fields. Reader/panel index starts at the first measured field (record index 3). Preserve full native acquisition, actual MaxX/MaxY and ablation step; source float32 Dual signal unchanged."),
    (META / "source_manifest.json", "source_manifest.dataset_name;canonical_id", "sample_metadata", "study", "Use the exact public dataset name as the stable study identifier."),
    (META / "official_zenodo_record.json", "BSW-Validation filenames; cohort description", "sample_metadata", "sample_id;source_sample_id;cohort_id;metadata_notes", "Use the exact MCD basename without extension as sample_id and source_sample_id. The public record maps AA/EA filename tokens to Black/White American groups; the numeric suffix is not defined as a patient ID. All ROIs from one file share the source-labelled whole-section sample."),
    (META / "source_attachments/clinical_attributes.txt", "clinical_attributes.txt;57 rows", "sample_metadata", "metadata_notes", "This 57-patient clinical source belongs to BSW-Discovery and is not joined to BSW2 Validation. Do not copy its clinical values into BSW2 rows; Discovery core-to-patient mapping remains unresolved."),
    (META / "source_attachments/article.xml", "Methods: BSW-2 whole sections;FFPE", "sample_metadata", "sample_type;tissue_type;anatomical_site;preservation_method;primary_diagnosis;molecular_subtype", "Apply only cohort-supported FFPE TNBC breast whole-section specimen attributes; no individual clinical values copied."),
    (META / "source_attachments/article.xml", "Supplementary Tables 1-2 and Figures 14-15; PMC file 41467_2025_61034_MOESM1_ESM.pptx", "supplement cohort captions and BSW2 tissue labels", "metadata_notes", "Supplementary Table 1 summarizes BSW-Discovery and Table 2 summarizes Roswell Park; BSW2 Figures 14-15 show five BA and five WA tissues without individual clinical values or patient crosswalk. Source: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-61034-3/MediaObjects/41467_2025_61034_MOESM1_ESM.pptx"),
    (S1, "Original marker;Unified marker;Segmentation channel", "panel_mapping", "source_channel_index;source_name;marker_id;keep_channel;output_channel_index;nuclear;body", "Exact study-specific S1 join; empty native labels map to <Unnamed channel>, original source_name retained. Exclude XYZ from measurement axis; all measured fields retained in panel. Preserve both S1-selected Ir channels and admitted markers in source order."),
]:
    join_keys = "source basename + native acquisition ID / accepted physical core ID"
    value_mapping = "{}"
    conflict_rule = "No inferred patient or Discovery specimen join"
    if source.name == "source_manifest.json":
        join_keys = "source_manifest.dataset_name"
        conflict_rule = "Use only the manifest dataset name matching the canonical study"
    elif source.name == "official_zenodo_record.json":
        join_keys = "exact source MCD basename; BSW-Validation cohort"
        value_mapping = "{\"AA\":\"Black American\",\"EA\":\"White American\"}"
        conflict_rule = "AA/EA denote group only; numeric filename suffix is not a patient identifier"
    elif source.name == "clinical_attributes.txt":
        join_keys = "clinical patient label; BSW-Discovery cohort"
        conflict_rule = "Never join BSW-Discovery rows to BSW2 Validation samples"
    fields.append(dict(study=STUDY, source_file=relative(source), source_sheet_or_table=origin,
                       source_field=origin, target_table=target, target_field=columns,
                       join_keys=join_keys, value_mapping=value_mapping, transformation_rule=rule,
                       time_origin_rule="Unknown clinical origins remain empty",
                       conflict_rule=conflict_rule, notes=""))
fields = pd.DataFrame(fields, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)
images = pd.read_parquet(OUTPUT / "images.parquet").to_dict("records") if (OUTPUT / "images.parquet").exists() else []
write_json(CACHE / "source_plan.json", plan)
estimate = sum(max(x["width"], x["height"])**2 * (int(panel[panel.panel_id == x["panel_id"]].keep_channel.sum())*4+1) for x in plan)
space = dict(estimated_output_bytes=estimate, nas_free_bytes=shutil.disk_usage(ROOT).free,
             estimated_local_source_bytes=sum({x["source_file"]: x["source_size"] for x in plan}.values()),
             local_free_bytes=shutil.disk_usage("/tmp").free, nas_reserve_bytes=100*1024**3,
             local_reserve_bytes=64*1024**3)
write_json(args.job / "space_estimate.json", space)
if space["nas_free_bytes"] - estimate < space["nas_reserve_bytes"] or space["local_free_bytes"] - space["estimated_local_source_bytes"] - estimate < space["local_reserve_bytes"]:
    progress("waiting_io", reason="Insufficient free space above required reserves", resume="Rerun same command; existing image receipts are retained")
    raise SystemExit(0)

# Export and close one image before publishing its index and progress.
if not args.verify_only:
    progress("exporting")
    for item in plan:
        source_path = ROOT / item["source_file"]
        destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
        receipt_path = CACHE / "receipts" / (item["image_id"] + ".json")
        selected = panel[panel.panel_id == item["panel_id"]].copy()
        signature = dict(source_size=item["source_size"], source_mtime_ns=item["source_mtime_ns"],
                         acquisition_id=item["acquisition_id"], panel=selected.to_json(orient="records"))
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if args.index_receipts_only and not (prior and prior["signature"] == signature and destination.exists()):
            continue
        if prior and prior["signature"] == signature and destination.exists():
            if any(x["image_id"] == item["image_id"] for x in images):
                continue
            result = prior["result"]
        else:
            progress("exporting", current_image=item["image_id"])
            # Stage each unchanged MCD once, so strided native-channel reads
            # operate on local storage rather than repeatedly traversing NAS.
            local = Path("/tmp/cellweft-zhu2025-sources") / source_path.name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or (local.stat().st_size, local.stat().st_mtime_ns) != (item["source_size"], item["source_mtime_ns"]):
                progress("exporting", current_image=item["image_id"], stage="source_staging", local_source=str(local))
                with formal_read_lease(source_path):
                    temporary = local.with_suffix(".mcd.partial")
                    shutil.copy2(source_path, temporary)
                    assert (source_path.stat().st_size, source_path.stat().st_mtime_ns) == (item["source_size"], item["source_mtime_ns"])
                    assert temporary.stat().st_size == item["source_size"]
                    os.replace(temporary, local)
            spec = SourceSpec(image_id=item["image_id"], panel_id=item["panel_id"], kind="mcd",
                              path=local, acquisition_id=item["acquisition_id"],
                              source_revision=json.dumps([str(source_path),item["source_size"],item["source_mtime_ns"]]),
                              channels=tuple(range(len(selected))))
            with ExitStack() as stack:
                with formal_read_lease(local):
                    reader = stack.enter_context(open_source(spec))
                result = export_input(reader, selected, destination, tile_size=256)
            write_json(receipt_path, dict(signature=signature, result=result))
        if not result["admitted"]:
            progress("exporting", excluded_zero_marker_image=item["image_id"])
            continue
        row = dict.fromkeys(pd.read_csv(TEMPLATES / "image_metadata.csv").columns, None)
        row.update(study=STUDY, image_id=item["image_id"], sample_id=item["sample_id"] or None,
                   panel_id=item["panel_id"], image_kind="tma_core" if item["core_id"] else "roi",
                   source_image_id=f"{source_path.name}::acquisition_{item['acquisition_id']}",
                   source_files=[item["source_file"]], source_format="mcd", source_acquisition_id=str(item["acquisition_id"]),
                   source_axes="CYX", source_revision="zenodo:15242478:revision4", slide_id=source_path.stem+":slide0",
                   core_id=item["core_id"] or None, roi_id=str(item["acquisition_id"]), region_annotation=item["description"],
                   source_x0_px=0, source_y0_px=0, source_x1_px=item["width"], source_y1_px=item["height"],
                   pixel_size_x_um=item["step_x"], pixel_size_y_um=item["step_y"], input_image_path=relative(destination),
                   valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
                   intensity_description="Native MCD float32 Dual signal; no normalization. XYZ are acquisition coordinates, not measured channels.",
                   split_method="source_acquisition", split_evidence=item["proof"],
                   metadata_notes="Patient linkage unresolved. " + ("Discovery physical core identified; biological specimen crosswalk unresolved." if not item["sample_id"] else "All ROIs share source-labelled whole-section specimen."))
        for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_dtype", "source_channel_count", "channel_count"):
            row[key] = result[key]
        images = [x for x in images if x["image_id"] != item["image_id"]] + [row]
        save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel, fields)
        progress("exporting", last_completed_image=item["image_id"])
        print(f"exported {len(images)}/{len(plan)}", flush=True)
    save_input_tables(OUTPUT, pd.DataFrame(images), samples, panel, fields)
    review()

# Preparer self-check: all TIFF headers; all channels in two <=64px windows
# per actual panel. Read leases cover initialization, source and output reads.
comparisons, previews = [], []
checked_panels = set()
for item in plan:
    row = next((x for x in images if x["image_id"] == item["image_id"]), None)
    if row is None:
        continue
    selected = panel[(panel.panel_id == item["panel_id"]) & panel.keep_channel].sort_values("output_channel_index")
    path = ROOT / row["input_image_path"]
    with formal_read_lease(path):
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == len(selected)
            assert tif.pages[0].shape == (row["height_px"], row["width_px"])
            assert tif.pages[0].dtype == np.dtype("float32")
            labels = [e.attrib["Name"] for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.endswith("}Channel")]
            assert labels == selected.marker_id.tolist()
            rgb = []
            for group in (selected[selected.nuclear], selected[selected.marker_id == "PanCK"], selected[selected.marker_id == "CD45"]):
                signal = sum((tif.pages[int(i)].asarray(maxworkers=1) for i in group.output_channel_index), np.zeros(tif.pages[0].shape, np.float32))
                scale = np.percentile(signal[signal > 0], 99) if np.any(signal > 0) else 1
                rgb.append(np.uint8(np.clip(signal / scale, 0, 1)*255))
            previews.append((item["image_id"], Image.fromarray(np.stack(rgb, axis=-1)).resize((192, 192))))
    if item["panel_id"] in checked_panels:
        continue
    spec = SourceSpec(image_id=item["image_id"], panel_id=item["panel_id"], kind="mcd", path=ROOT/item["source_file"],
                      acquisition_id=item["acquisition_id"], channels=tuple(range(int(row["source_channel_count"]))))
    w, h = item["width"], item["height"]
    blocks = [(w//2, h//2, min(w//2+64,w), min(h//2+64,h)), (0,0,min(64,w),min(64,h))]
    with formal_read_lease(path):
        with open_source(spec) as reader, tifffile.TiffFile(path) as tif:
            for block in blocks:
                x0,y0,x1,y1 = block
                target = (x0+row["pad_left_px"],y0+row["pad_top_px"],x1+row["pad_left_px"],y1+row["pad_top_px"])
                native = reader.read_window(selected.source_channel_index.astype(int).tolist(), block)
                for index, output_index in enumerate(selected.output_channel_index):
                    np.testing.assert_array_equal(native[index], read_tiff_window(tif,int(output_index),target))
            for page in tif.pages:
                if row["pad_left_px"]:
                    assert not read_tiff_window(tif,page.index,(0,0,row["pad_left_px"],row["height_px"])).any()
                if row["pad_top_px"]:
                    assert not read_tiff_window(tif,page.index,(0,0,row["width_px"],row["pad_top_px"])).any()
    comparisons.append(dict(image_id=item["image_id"],panel_id=item["panel_id"],all_retained_channels=len(selected),blocks=blocks,exact=True))
    checked_panels.add(item["panel_id"])
for offset in range(0,len(previews),16):
    group = previews[offset:offset+16]
    canvas = Image.new("RGB",(4*256,4*224),"white")
    draw = ImageDraw.Draw(canvas)
    for index,(identity,preview) in enumerate(group):
        x,y = index%4*256,index//4*224
        canvas.paste(preview,(x,y+26));draw.text((x,y),identity[:36],fill="black")
    canvas.save(CACHE / f"overview_{offset//16:02d}.png")
write_json(CACHE / "preparer_checks.json",dict(independent_review=False,header_images=len(images),pixel_checks=comparisons,
           overview_images=len(previews),overview_channels="R: Ir191+Ir193; G: PanCK; B: CD45",visual_judgment="Pending actual inspection"))
if args.verify_only or args.index_receipts_only:
    review()
progress("prepared" if len(images) == len(plan) else "waiting_io", independent_review=False,
         next_image=next((x["image_id"] for x in plan if x["image_id"] not in {r["image_id"] for r in images}), None),
         resume="Rerun without --verify-only; reuse completed receipts and resume at the first missing image",
         patient_linkage_gap=True, discovery_sample_linkage_gap=5,
         header_images=len(images), pixel_checked_images=len(comparisons))
print(json.dumps(dict(status="prepared" if len(images) == len(plan) else "waiting_io",images=len(images),samples=len(samples),patient_linked_samples=0)))
