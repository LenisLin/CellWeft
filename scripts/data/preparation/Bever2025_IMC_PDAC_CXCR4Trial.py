"""Inventory and prepare native Bever 2025 PDAC IMC acquisitions.

Inputs: unchanged Zenodo 11122889 MCD files and the read-only S1 table.
Outputs: private inventory, candidate native TIFF inputs and five study tables.
The first inventory stage can run independently with --inventory-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

from readimc import MCDFile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Bever2025_IMC_PDAC_CXCR4Trial"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY
IMAGE_ROOT = RAW / "images"
METADATA_ROOT = RAW / "metadata"
PROJECT = Path(__file__).resolve().parents[3]
JOB = PROJECT / "cache/bever_hiltbrunner_20261108" / STUDY
OUTPUT = JOB / "candidate_processed" / STUDY
MANIFEST = RAW / "manifest.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inventory-only", action="store_true")
parser.add_argument("--limit", type=int, help="Export only the first N native acquisitions for a pilot")
parser.add_argument("--pilot", action="store_true", help="Export one representative acquisition per native panel")
parser.add_argument("--verify-only", action="store_true", help="Check complete candidate without exporting source pixels")
args = parser.parse_args()
JOB.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def progress(phase: str, **values: object) -> None:
    save_json(JOB / "progress.json", dict(study=STUDY, phase=phase,
              updated_at=datetime.now(timezone.utc).isoformat(), **values))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    if path.is_file() and temporary.read_bytes() == path.read_bytes():
        temporary.unlink()
        return
    os.replace(temporary, path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / name, nrows=0).columns.tolist()


def canonical_source_file(value: str) -> str:
    return f"raw/public/{STUDY}/images/{Path(value).name}"


expected = {
    Path(item["target"]).name: {"size": item["target_identity"]["size"]}
    for item in json.loads(MANIFEST.read_text())["members"]
    if Path(item["target"]).parent == IMAGE_ROOT and Path(item["target"]).suffix.lower() == ".mcd"
}
actual = {path.name: path for path in IMAGE_ROOT.glob("*.mcd")}
assert set(actual) == set(expected), (set(expected) - set(actual), set(actual) - set(expected))
cached_inventory = (json.loads((JOB / "inventory.json").read_text())
                    if (JOB / "inventory.json").is_file() else None)
reuse_inventory = bool(cached_inventory and cached_inventory.get("actual_mcd") == len(expected)
    and set(cached_inventory["files"]) == set(actual)
    and all((actual[Path(item["source_file"]).name].stat().st_size,
             actual[Path(item["source_file"]).name].stat().st_mtime_ns)
            == (item["source_size"], item["source_mtime_ns"])
            for item in cached_inventory["acquisitions"]))
acquisitions = ([dict(item, source_file=canonical_source_file(item["source_file"]))
                 for item in cached_inventory["acquisitions"]] if reuse_inventory else [])
layouts: dict[tuple[tuple[str, str, int, str], ...], str] = {}
for item in acquisitions:
    layout = tuple((channel["label"], channel["name"], channel["mass"], channel["metal"])
                   for channel in item["channels"])
    layouts[layout] = item["panel_id"]
progress("inventory_reused" if reuse_inventory else "inventory", expected_mcd=len(expected),
         inspected_mcd=len(expected) if reuse_inventory else 0,
         acquisition_count=len(acquisitions), panel_count=len(layouts))
for number, (name, path) in enumerate([] if reuse_inventory else sorted(actual.items()), 1):
    stat = path.stat()
    assert stat.st_size == expected[name]["size"], name
    with formal_read_lease(path):
        with MCDFile(path) as mcd:
            for slide in mcd.slides:
                for acquisition in slide.acquisitions:
                    meta = acquisition.metadata
                    channels = [dict(index=index, label=label, name=channel_name,
                                 metal=metal, mass=mass)
                            for index, (label, channel_name, metal, mass) in enumerate(zip(
                                acquisition.channel_labels, acquisition.channel_names,
                                acquisition.channel_metals, acquisition.channel_masses))]
                    layout = tuple((item["label"], item["name"], item["mass"], item["metal"])
                               for item in channels)
                    panel_id = layouts.setdefault(layout, f"native_{len(layouts)+1:02d}")
                    acquisitions.append(dict(source_file=str(path.relative_to(ROOT)),
                    source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                    slide_id=slide.id, slide_filename=slide.metadata.get("Filename", ""),
                    acquisition_id=acquisition.id, description=meta.get("Description", ""),
                    width=int(meta["MaxX"]), height=int(meta["MaxY"]),
                    data_start=int(meta["DataStartOffset"]), data_end=int(meta["DataEndOffset"]),
                    pixel_size_x_um=meta.get("AblationDistanceBetweenShotsX"),
                    pixel_size_y_um=meta.get("AblationDistanceBetweenShotsY"),
                    signal_type=meta.get("SignalType"), value_bytes=meta.get("ValueBytes"),
                    panel_id=panel_id, channels=channels))
    progress("inventory", expected_mcd=len(expected), inspected_mcd=number,
             acquisition_count=len(acquisitions), panel_count=len(layouts))

inventory = dict(study=STUDY, release="Zenodo 11122889 revision 4",
    expected_mcd=len(expected), actual_mcd=len(actual), acquisition_count=len(acquisitions),
    panel_count=len(layouts), files=sorted(actual), acquisitions=acquisitions,
    annotation_release_files=["backup_output.rds", "backup_output_P010.rds"],
annotation_files_local=[name for name in ("backup_output.rds", "backup_output_P010.rds")
                            if (METADATA_ROOT / name).is_file()],
    sample_patient_gap="ROI-to-biological-sample and patient crosswalk pending source review",
    author_mask_gap="Annotated RDS files not present locally; mask type and acquisition pairing pending")
save_json(JOB / "inventory.json", inventory)
progress("inventory_complete", expected_mcd=len(expected), inspected_mcd=len(actual),
         acquisition_count=len(acquisitions), panel_count=len(layouts),
         candidate_images=0, missing_annotation_files=inventory["annotation_release_files"])
print(json.dumps({key: inventory[key] for key in
                  ("actual_mcd", "acquisition_count", "panel_count", "annotation_files_local")}))
if args.inventory_only:
    raise SystemExit(0)

# Author workbooks and both RDS metadata lists identify analytic FCS rows.
author = METADATA_ROOT
full_metadata = pd.read_excel(author / "J19113_metadata.xlsx", dtype=str)
p010_metadata = pd.read_excel(author / "J19113_P010_metadata.xlsx", dtype=str)
assert len(full_metadata) == 63 and len(p010_metadata) == 6
clinical = {}
for table, origin in ((full_metadata, "full cohort"), (p010_metadata, "P010 supplement")):
    for row in table.to_dict("records"):
        key = row["file_name"].removesuffix(".fcs")
        clinical.setdefault(key, row | {"source_table": origin})
        if origin == "P010 supplement" and key.startswith("P010_"):
            clinical[key] = row | {"source_table": origin}

scope = []
for item in acquisitions:
    stem = Path(item["source_file"]).stem
    match = re.fullmatch(r"ROI_(\d+)", item["description"])
    ordinal = item["acquisition_id"] if stem == "P06_t3" else (int(match[1]) if match else item["acquisition_id"])
    fcs_id = f"{stem}_{ordinal}"
    evidence = clinical.get(fcs_id)
    scope.append(dict(image_id=f"{stem}__a{item['acquisition_id']:02d}",
        source_file=item["source_file"], acquisition_id=item["acquisition_id"],
        native_description=item["description"], width=item["width"], height=item["height"],
        candidate_fcs_id=fcs_id, author_fcs_match=bool(evidence),
        author_source=evidence["source_table"] if evidence else "",
        case=evidence["Case"] if evidence else "", timepoint=evidence["Timepoint"] if evidence else "",
        tumor=evidence["Tumor"] if evidence else "", time_group=evidence["TimeGroup"] if evidence else "",
        decision="include_native_acquisition", reason=(
            "Author FCS/RDS metadata row matches native ROI in the same MCD"
            if evidence else "Independent native acquisition; no author FCS row; sample linkage pending")))
assert len(scope) == 79 and len({x["image_id"] for x in scope}) == 79
write_csv(JOB / "source_scope.csv", scope, list(scope[0]))

# Proposal uses the formal S1 as a read-only marker vocabulary. Generic or
# state-specific measurements remain pending when source epitope is unresolved.
s1_path = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
s1 = pd.read_csv(s1_path, dtype=str, keep_default_na=False)
assert not (s1.Study == STUDY).any()
s1_columns = list(s1.columns)
assert s1_columns == ["Study", "Original marker", "Unified marker", "MAP-KG mapping",
                      "UniProt accession", "Segmentation channel"]
all_channels = {channel["label"]: channel for item in acquisitions for channel in item["channels"]}
overrides = {
    "Collagen": ("Collagen (mapping pending)", "", "", ""),
    "CK": ("PanCK (mapping pending)", "", "", "2"),
    "TOX2": ("TOX (mapping pending)", "", "", ""),
    "Tox-Tox2": ("TOX (mapping pending)", "", "", ""),
    "pSTAT3": ("pSTAT3 (mapping pending)", "", "", ""),
    "191Ir": ("191Ir (segmentation only)", "", "", "1"),
    "193Ir": ("193Ir (segmentation only)", "", "", "1"),
    "DNA1": ("DNA (segmentation only)", "", "", "1"),
    "DNA2": ("DNA (segmentation only)", "", "", "1"),
}
aliases = {"CXCLl12": "CXCL12"}
proposal, proposal_evidence = [], []
for label, native in all_channels.items():
    match = s1[s1["Original marker"] == aliases.get(label, label)]
    if label in overrides:
        unified, gene, accession, segmentation = overrides[label]
        basis = "Source-specific measurement or nuclear identity; see evidence note"
    elif not match.empty:
        alternatives = match[match["Unified marker"] != "Excluded"]
        chosen = (alternatives if not alternatives.empty else match).iloc[0]
        unified, gene, accession, segmentation = (chosen[key] for key in s1_columns[2:])
        basis = f"Formal S1 analogue: {chosen['Study']} / {chosen['Original marker']}"
        if label in {"PM2", "PM3", "PM4"}:
            segmentation = ""
    else:
        unified, gene, accession, segmentation = f"{label} (mapping pending)", "", "", ""
        basis = "No exact marker in formal S1"
    if label == "CXCLl12":
        basis += "; source Tm169 matches author panel CXCL12"
    if label in {"191Ir", "193Ir"}:
        basis += "; native Ir slot has observed nuclear morphology; dye identity unconfirmed"
    if label in {"DNA1", "DNA2"}:
        basis += "; source explicitly labels this Ir channel DNA"
    if label == "CK":
        basis += "; author panel calls Sm148 PanCK without clone"
    if label == "Collagen":
        basis += "; author panel calls In113 COL without collagen type"
    if label in {"TOX2", "Tox-Tox2"}:
        basis += "; native labels differ and author panel calls Eu153 TOX"
    proposal.append(dict(zip(s1_columns, [STUDY, label, unified, gene, accession, segmentation])))
    proposal_evidence.append(dict(study=STUDY, source_marker=label,
        source_mass=native["mass"], source_metal=native["metal"],
        candidate_mapping=unified, mapping_basis=basis,
        source_panel=f"Zenodo 11122889 native MCD channels; {author / 'J19113_panel.xlsx'}",
        existing_s1_reuse=str(s1_path),
        status="mapping_pending" if "(mapping pending)" in unified else "proposed"))
write_csv(JOB / "candidate_s1_rows.csv", proposal, s1_columns)
write_csv(JOB / "candidate_s1_evidence.csv", proposal_evidence, list(proposal_evidence[0]))
panels = []
for panel_id in dict.fromkeys(item["panel_id"] for item in acquisitions):
    native = next(item for item in acquisitions if item["panel_id"] == panel_id)
    panel = panel_from_s1(STUDY, panel_id, [c["label"] for c in native["channels"]],
                          JOB / "candidate_s1_rows.csv")
    panel["source_mass"] = [c["mass"] for c in native["channels"]]
    panel["source_metal"] = [c["metal"] for c in native["channels"]]
    panels.append(panel)
panel = pd.concat(panels, ignore_index=True)

samples = pd.DataFrame(columns=columns("sample_metadata.csv"))
site_conflicts = [dict(image_id=item["image_id"], candidate_fcs_id=item["candidate_fcs_id"],
    full_cohort_site="OTHER", p010_specific_site="ADRENAL",
    resolution="unresolved; do not use site for biological stratification")
    for item in scope if item["case"] == "P010" and item["author_fcs_match"]]
write_csv(JOB / "source_site_conflicts.csv", site_conflicts, list(site_conflicts[0]))

field_columns = columns("field_mapping.csv")
field_values = [
    dict(source_file=str(JOB / "inventory.json"), source_sheet_or_table="acquisitions",
         source_field="source_file;acquisition_id;Description;MaxX;MaxY;channels", target_table="image_metadata",
         target_field="source_image_id;source_acquisition_id;source_x1_px;source_y1_px;source_channel_count",
         join_keys="MCD basename + native acquisition ID", transformation_rule="One complete native acquisition per image; centered square padding; native float32 intensities and measured validity."),
    dict(source_file=str(author / "J19113_metadata.xlsx"), source_sheet_or_table="Sheet2",
         source_field="file_name;sample_id;Case;Tumor;Timepoint;TimeGroup", target_table="sample_metadata",
         target_field="source_case_id;source_timepoint;source_time_group;source_tumor_site_label",
         join_keys="MCD stem + native ROI ordinal = FCS basename",
         transformation_rule="Transcribe image-level author Case/Timepoint only. FCS sample_id denotes ROI, not biological sample; all image sample_id values remain empty."),
    dict(source_file=str(author / "J19113_P010_metadata.xlsx"), source_sheet_or_table="Sheet1",
         source_field="file_name;Case;Tumor;Timepoint;TimeGroup", target_table="sample_metadata",
         target_field="source_case_id;source_timepoint;source_tumor_site_label",
         join_keys="P010 native acquisition to P010-specific FCS basename",
         transformation_rule="P010-specific rows supplement full cohort; OTHER versus ADRENAL site classification is unresolved and image site label remains blank."),
    dict(source_file=str(JOB / "candidate_s1_rows.csv"), source_sheet_or_table="six-column proposal",
         source_field="Original marker;Unified marker;Segmentation channel", target_table="panel_mapping",
         target_field="source_name;marker_id;nuclear;body;keep_channel;output_channel_index",
         join_keys="actual ordered native label + proposed study S1 row",
         transformation_rule="Retain admitted markers and source-supported Ir nuclear channels; generic epitopes remain mapping pending; exclude technical metals."),
    dict(source_file=str(JOB / "source_scope.csv"), source_sheet_or_table="all 79 acquisitions",
         source_field="author_fcs_match;decision;reason", target_table="image_metadata",
         target_field="sample_id;split_method;metadata_notes",
         join_keys="MCD basename + native acquisition ID to author FCS basename",
         transformation_rule="All 79 measured native ROIs kept. No biological specimen/block crosswalk exists; image sample_id remains blank even for author FCS rows."),
]
for target, definition in {
    "source_case_id": "Deidentified author Case for an exactly matched FCS ROI; image-level subject evidence only.",
    "source_timepoint": "Author Timepoint for an exactly matched FCS ROI, as reported.",
    "source_time_group": "Author TimeGroup for an exactly matched FCS ROI, as reported.",
    "source_tumor_site_label": "Author Tumor site for a matched ROI; blank for P010 due to OTHER/ADRENAL discrepancy.",
    "author_fcs_id": "Exact matching FCS ROI basename in the author configuration; blank when unlisted.",
    "author_analysis_scope": "Whether this native acquisition is listed in author FCS analysis metadata.",
}.items():
    field_values.append(dict(source_file=str(JOB / "source_scope.csv"), source_sheet_or_table="ROI linkage",
        source_field=target, target_table="image_metadata", target_field=target,
        join_keys="MCD basename + native ROI ordinal to FCS basename",
        transformation_rule="Retain only source-supported image-level labels; no biological sample inference.",
        notes=json.dumps({"data_type": "string", "definition": definition})))
fields = pd.DataFrame([{**dict.fromkeys(field_columns, ""), **dict(study=STUDY, value_mapping="{}",
    time_origin_rule="Use explicit author Timepoint only", conflict_rule="Do not assign P010 site or unlisted ROI sample identity") , **entry}
    for entry in field_values], columns=field_columns)
save_json(JOB / "mask_binding_proposal.json", dict(study=STUDY,
    status="no_pixel_mask_identified", bindings={},
    evidence=[str(author / "backup_output.rds"),
              str(author / "backup_output_P010.rds"),
              f"{author / 'J19113IMC-main.zip'}#J19113IMC-main/Rscripts/"],
    reason="Both RDS objects contain flowCore single-cell FCS lists, sample IDs and metadata, without an image-sized instance mask field; source MCD contains measured channels only.",
    proposal="Use cpsam_v2 for eligible nuclear inputs after independent input review."))

save_json(JOB / "plan.json", dict(study=STUDY, source_acquisitions=79,
    candidate_images=79, author_fcs_linked=sum(x["author_fcs_match"] for x in scope),
    sample_rows=len(samples), author_case_keys=len({x["case"] for x in scope if x["case"]}),
    panel_channels={p: len(panel[panel.panel_id == p]) for p in panel.panel_id.unique()},
    source_scope=str(JOB / "source_scope.csv")))
print(json.dumps(dict(stage="planned", images=len(scope), author_linked=sum(x["author_fcs_match"] for x in scope),
                      samples=len(samples), panels=len(panels))), flush=True)
progress("planned", expected_images=len(scope), exported_images=0,
         author_fcs_linked=sum(x["author_fcs_match"] for x in scope),
         sample_linked_images=0, source_case_linked_images=sum(x["author_fcs_match"] for x in scope),
         source_rds_downloaded=["backup_output.rds", "backup_output_P010.rds"],
         candidate_s1=str(JOB / "candidate_s1_rows.csv"))

# Stage each unchanged MCD once on SSD. The exporter then reads only local
# source bytes while publishing candidate images into the private SSD job.
OUTPUT.mkdir(parents=True, exist_ok=True)
(JOB / "source_cache").mkdir(exist_ok=True)
receipts = JOB / "receipts"
receipts.mkdir(exist_ok=True)
selected = list(zip(acquisitions, scope))
if args.pilot:
    seen = set()
    selected = [(native, link) for native, link in selected
                if not (native["panel_id"] in seen or seen.add(native["panel_id"]))]
elif args.limit is not None:
    selected = selected[:args.limit]
if args.verify_only:
    selected = []
rows_by_id = {}
if (OUTPUT / "images.parquet").is_file():
    rows_by_id = {row["image_id"]: row for row in pd.read_parquet(OUTPUT / "images.parquet").to_dict("records")}
start_ticks = int(Path("/proc/self/stat").read_text().split()[21])
for native, link in selected:
    image_id = link["image_id"]
    original = ROOT / native["source_file"]
    local = JOB / "source_cache" / original.name
    original_version = (native["source_size"], native["source_mtime_ns"])
    if (not local.is_file() or
            (local.stat().st_size, local.stat().st_mtime_ns) != original_version):
        progress("staging_source", expected_images=79, exported_images=len(rows_by_id),
                 current_image=image_id, source_file=native["source_file"],
                 pid=os.getpid(), start_ticks=start_ticks)
        temporary = local.with_name(local.name + ".partial")
        with formal_read_lease(original):
            shutil.copy2(original, temporary)
            assert (original.stat().st_size, original.stat().st_mtime_ns) == original_version
            assert (temporary.stat().st_size, temporary.stat().st_mtime_ns) == original_version
        os.replace(temporary, local)
    destination = OUTPUT / "images" / image_id / "input.tiff"
    receipt = receipts / f"{image_id}.json"
    panel_stat = (JOB / "candidate_s1_rows.csv").stat()
    signature = dict(source_file=native["source_file"], source_size=native["source_size"],
                     source_mtime_ns=native["source_mtime_ns"], acquisition_id=native["acquisition_id"],
                     panel_id=native["panel_id"], panel_revision=(panel_stat.st_size, panel_stat.st_mtime_ns))
    prior = json.loads(receipt.read_text()) if receipt.is_file() else None
    if prior and prior["signature"] == signature and destination.is_file():
        result = prior["result"]
    else:
        progress("exporting", expected_images=79, exported_images=len(rows_by_id),
                 current_image=image_id, source_file=native["source_file"],
                 pid=os.getpid(), start_ticks=start_ticks)
        source = SourceSpec(image_id=image_id, panel_id=native["panel_id"], kind="mcd",
                            path=local, acquisition_id=native["acquisition_id"],
                            channels=tuple(range(len(native["channels"]))),
                            source_revision=f"Zenodo11122889-r4:{native['source_size']}:{native['source_mtime_ns']}",
                            pixel_size_um=(float(native["pixel_size_x_um"]), float(native["pixel_size_y_um"])),
                            intensity_description="Native float32 MCD Dual signal; unnormalized")
        selected_panel = panel[panel.panel_id == native["panel_id"]]
        with open_source(source) as reader:
            result = export_input(reader, selected_panel, destination, tile_size=256, data_root=ROOT)
        assert result["admitted"] and result["source_dtype"] == "float32", image_id
        save_json(receipt, dict(signature=signature, result=result))
    row = dict.fromkeys(columns("image_metadata.csv"), None)
    site = link["tumor"] if link["case"] != "P010" else ""
    row.update(study=STUDY, image_id=image_id, panel_id=native["panel_id"], image_kind="roi",
        source_image_id=f"{original.name}::acquisition_{native['acquisition_id']}",
        source_files=[native["source_file"]], source_format="mcd",
        source_acquisition_id=str(native["acquisition_id"]), source_axes="CYX",
        source_dtype=result["source_dtype"], source_revision="zenodo:11122889:r4",
        slide_id=f"{original.stem}:slide_{native['slide_id']}",
        roi_id=native["description"], region_annotation=native["description"],
        source_x0_px=0, source_y0_px=0, source_x1_px=native["width"], source_y1_px=native["height"],
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=result["height_px"], width_px=result["width_px"],
        source_channel_count=len(native["channels"]), channel_count=result["channel_count"],
        pixel_size_x_um=float(native["pixel_size_x_um"]), pixel_size_y_um=float(native["pixel_size_y_um"]),
        input_image_path=str(destination),
        valid_pixels_path=str(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
        intensity_description="Native MCD float32 Dual signal; no normalization or thresholding. Measured zeros remain valid.",
        split_method="source_acquisition", split_evidence=str(JOB / "inventory.json"),
        metadata_notes=("Author FCS ROI and image-level Case/Timepoint linked; biological specimen ID unresolved."
                        if link["author_fcs_match"] else
                        "Author FCS row absent; native acquisition and measured signal retained; sample and patient linkage unresolved."),
        source_case_id=link["case"] or None, source_timepoint=link["timepoint"] or None,
        source_time_group=link["time_group"] or None, source_tumor_site_label=site or None,
        author_fcs_id=link["candidate_fcs_id"] if link["author_fcs_match"] else None,
        author_analysis_scope="listed" if link["author_fcs_match"] else "unlisted")
    rows_by_id[image_id] = row
    save_input_tables(OUTPUT, pd.DataFrame(rows_by_id.values()), samples, panel, fields, data_root=ROOT)
    progress("exporting", expected_images=79, exported_images=len(rows_by_id),
             last_completed_image=image_id, current_source=original.name,
             author_fcs_linked=sum(bool(r.get("author_fcs_id")) for r in rows_by_id.values()),
             sample_linked_images=0, pid=os.getpid(), start_ticks=start_ticks)
    print(json.dumps(dict(stage="candidate_image", image_id=image_id, exported=len(rows_by_id),
                          expected=79)), flush=True)

# Whole-study structure checks, with exact native/output windows on one image
# from each actual ordered panel. The selection is stable across pilot and full runs.
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
pixel_checks = []
headers_checked = 0
for row in images.to_dict("records"):
    path = Path(row["input_image_path"])
    native = next(item for item in acquisitions if
                  f"{Path(item['source_file']).stem}__a{item['acquisition_id']:02d}" == row["image_id"])
    source_stat = (ROOT / native["source_file"]).stat()
    assert (source_stat.st_size, source_stat.st_mtime_ns) == (native["source_size"], native["source_mtime_ns"])
    local_stat = (JOB / "source_cache" / Path(native["source_file"]).name).stat()
    assert (local_stat.st_size, local_stat.st_mtime_ns) == (native["source_size"], native["source_mtime_ns"])
    kept = panel[(panel.panel_id == row["panel_id"]) & panel.keep_channel].sort_values("output_channel_index")
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == len(kept), row["image_id"]
        assert all(page.shape == (row["height_px"], row["width_px"]) and
                   page.dtype == np.dtype("float32") for page in tif.pages), row["image_id"]
        names = [node.attrib.get("Name") for node in ET.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist(), row["image_id"]
    if pd.notna(row["valid_pixels_path"]):
        with tifffile.TiffFile(row["valid_pixels_path"]) as tif:
            assert len(tif.pages) == 1 and tif.pages[0].dtype == np.dtype("uint8")
            assert tif.pages[0].shape == (row["height_px"], row["width_px"])
    headers_checked += 1
for panel_id in panel.panel_id.unique():
    representative = next((item for item in images.to_dict("records") if item["panel_id"] == panel_id), None)
    if representative is None:
        continue
    native = next(item for item in acquisitions if
                  f"{Path(item['source_file']).stem}__a{item['acquisition_id']:02d}" == representative["image_id"])
    local = JOB / "source_cache" / Path(native["source_file"]).name
    source = SourceSpec(image_id=representative["image_id"], panel_id=panel_id, kind="mcd",
                        path=local, acquisition_id=native["acquisition_id"],
                        channels=tuple(range(len(native["channels"]))))
    kept = panel[(panel.panel_id == panel_id) & panel.keep_channel].sort_values("output_channel_index")
    w, h = native["width"], native["height"]
    windows = [(0, 0, min(w, 48), min(h, 48)),
               (max(0, w//2-24), max(0, h//2-24), min(w, w//2+24), min(h, h//2+24))]
    with open_source(source) as reader, tifffile.TiffFile(representative["input_image_path"]) as tif:
        for source_window in windows:
            x0, y0, x1, y1 = source_window
            target_window = (x0+representative["pad_left_px"], y0+representative["pad_top_px"],
                             x1+representative["pad_left_px"], y1+representative["pad_top_px"])
            native_values = reader.read_window(kept.source_channel_index.astype(int).tolist(), source_window)
            for i, output_index in enumerate(kept.output_channel_index):
                np.testing.assert_array_equal(native_values[i], read_tiff_window(tif, int(output_index), target_window))
    pixel_checks.append(dict(image_id=representative["image_id"], panel_id=panel_id,
                             checked_channels=len(kept), windows=windows, exact=True))
save_json(JOB / "self_check.json", dict(study=STUDY, headers_checked=headers_checked,
    pixel_checks=pixel_checks, independent_review="pending", d01="pending"))

review_rows = [dict(study=STUDY, preparer="/root/bever_prepare", reviewer="", scope="image",
    record_id=row["image_id"], check_id="P03", result="preparer_checked",
    source_evidence=str(JOB / "inventory.json"), output_evidence=row["input_image_path"],
    details="Native float32 TIFF header and square frame checked; independent review pending.")
    for row in images.to_dict("records")]
review_rows.extend([
    dict(study=STUDY, preparer="/root/bever_prepare", reviewer="", scope="study",
         record_id=STUDY, check_id="M01", result="unresolved",
         source_evidence=str(JOB / "source_scope.csv"), output_evidence=str(OUTPUT / "images.parquet"),
         details="63 author FCS ROIs have Case/Timepoint; no biological specimen/block crosswalk; sample_id and sample table remain empty."),
    dict(study=STUDY, preparer="/root/bever_prepare", reviewer="", scope="study",
         record_id=STUDY, check_id="D01", result="pending",
         source_evidence=str(JOB / "self_check.json"), output_evidence=str(OUTPUT / "images.parquet"),
         details="Candidate inputs require independent reviewer and controller acceptance."),
])
write_csv(OUTPUT / "review_results.csv", review_rows, columns("review_results.csv"))
status = "candidate_complete" if len(images) == 79 else "candidate_partial"
save_json(JOB / "preparation_receipt.json", dict(status=status, preparer="/root/bever_prepare",
    study=STUDY, images=len(images), source_acquisitions=79, author_fcs_linked=int(images.author_fcs_id.notna().sum()),
    sample_linked_images=0, sample_rows=0, author_case_keys=14, independent_review="pending",
    candidate_output=str(OUTPUT), self_check=str(JOB / "self_check.json"),
    candidate_s1=str(JOB / "candidate_s1_rows.csv")))
progress(status, expected_images=79, exported_images=len(images), headers_checked=headers_checked,
         pixel_checked_panels=len(pixel_checks), author_fcs_linked=int(images.author_fcs_id.notna().sum()),
         sample_linked_images=0, pid=os.getpid(), start_ticks=start_ticks,
         candidate_output=str(OUTPUT), independent_review="pending")
print(json.dumps(dict(status=status, images=len(images), author_fcs_linked=int(images.author_fcs_id.notna().sum()),
                      headers_checked=headers_checked, pixel_checked_panels=len(pixel_checks))), flush=True)
