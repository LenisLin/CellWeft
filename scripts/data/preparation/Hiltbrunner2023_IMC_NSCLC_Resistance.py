"""Inventory and prepare native IMC acquisitions from Hiltbrunner 2023.

Inputs: Zenodo 8041882 MCDs, AK_List.csv, and the current formal S1.
Outputs: private source inventory, S1 proposal, candidate study input tables and TIFFs.
Native MCD intensities and acquisition geometry are retained for independent review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from readimc import MCDFile
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, save_input_tables


STUDY = "Hiltbrunner2023_IMC_NSCLC_Resistance"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY
IMAGE_ROOT = RAW / "images"
METADATA_ROOT = RAW / "metadata"
SSD = Path(__file__).resolve().parents[3]
JOB = SSD / "cache/bever_hiltbrunner_20261108" / STUDY
OUTPUT = JOB / "candidate_processed" / STUDY
LOCAL = JOB / "source_mcd"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inventory-only", action="store_true")
parser.add_argument("--verify-only", action="store_true")
parser.add_argument("--rewrite", action="store_true", help="Re-export all candidate TIFFs after a panel identity change")
parser.add_argument("--remap-ome", action="store_true", help="Update the Yb172 mixed measurement name in closed candidate TIFF headers")
args = parser.parse_args()
assert not args.remap_ome or (args.verify_only and not args.rewrite)
JOB.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_receipt(value: dict) -> None:
    write_json(JOB / "prep_receipt.json", value)
    write_json(JOB / "preparation_receipt.json", value)


def progress(stage: str, **extra: object) -> None:
    write_json(JOB / "progress.json", dict(study=STUDY, stage=stage,
        updated_utc=datetime.now(timezone.utc).isoformat(), **extra))


def template(name: str) -> pd.DataFrame:
    return pd.read_csv(TEMPLATES / (name + ".csv"), nrows=0)


def field(source_file: str, source_field: str, target_table: str, target_field: str,
          rule: str, join: str = "MCD basename + acquisition ID") -> dict:
    return dict(study=STUDY, source_file=source_file, source_sheet_or_table="native header / published release",
        source_field=source_field, target_table=target_table, target_field=target_field,
        join_keys=join, value_mapping="", transformation_rule=rule,
        time_origin_rule="", conflict_rule="", notes="")


if args.remap_ome:
    write_receipt(dict(study=STUDY, status="remap_in_progress",
        preparer="/root/hiltbrunner_prepare"))


# Inventory all native acquisitions and retain detector identities in source order.
source = json.loads((RAW / "manifest.json").read_text())
expected = {Path(item["target"]).name: item["target_identity"]["size"]
    for item in source["members"]
    if item["role"] == "original_author_image" and item["target"].endswith(".mcd")}
assert {path.name for path in IMAGE_ROOT.glob("*.mcd")} == set(expected)
files = []
acquisitions = []
for path in sorted(IMAGE_ROOT.glob("*.mcd")):
    stat = path.stat()
    assert stat.st_size == expected[path.name]
    with MCDFile(path) as mcd:
        slides = mcd.slides
        file_rows = []
        for slide in slides:
            for acq in slide.acquisitions:
                row = dict(source_file=str(path.relative_to(NAS)), source_bytes=stat.st_size,
                    source_mtime_ns=stat.st_mtime_ns, slide_id=slide.metadata["ID"],
                    slide_filename=slide.metadata.get("Filename"), acquisition_id=acq.id,
                    description=acq.description, width=acq.width_px, height=acq.height_px,
                    pixel_size_x_um=acq.pixel_size_x_um, pixel_size_y_um=acq.pixel_size_y_um,
                    channel_names=acq.channel_names, channel_labels=acq.channel_labels,
                    channel_masses=acq.channel_masses, channel_metals=acq.channel_metals,
                    data_start=acq.metadata.get("DataStartOffset"),
                    data_end=acq.metadata.get("DataEndOffset"))
                acquisitions.append(row)
                file_rows.append(row)
        files.append(dict(source_file=str(path.relative_to(NAS)), bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, slides=len(slides), acquisitions=len(file_rows),
            descriptions=[row["description"] for row in file_rows]))
    progress("inventory_scan", scanned_mcd=len(files), total_mcd=len(expected),
        acquisitions=len(acquisitions))

write_json(JOB / "inventory.json", dict(study=STUDY, zenodo_record=8041882,
    source_mcd=len(files), source_acquisitions=len(acquisitions),
    descriptions=dict(Counter(row["description"] for row in acquisitions)),
    panel_variants=len({tuple(zip(row["channel_names"], row["channel_labels"])) for row in acquisitions}),
    source_mask_note="Zenodo lists ilastik Probabilities_mask.tiff members; local selected raw files contain MCD and AK_List.csv only. Mask type and instance semantics unverified.",
    sample_patient_note="MCD filenames include P1-P7 and response/resistance; native acquisition description labels are ROI phenotypes, not a clinical crosswalk.",
    files=files, acquisitions=acquisitions))
progress("inventory_complete", source_mcd=len(files), source_acquisitions=len(acquisitions))
if args.inventory_only:
    print(json.dumps(dict(source_mcd=len(files), source_acquisitions=len(acquisitions),
        descriptions=Counter(row["description"] for row in acquisitions),
        panel_variants=len({tuple(zip(row["channel_names"], row["channel_labels"])) for row in acquisitions}))))
    raise SystemExit(0)


# Resolve the 39 labeled antibody measurements against the current S1 vocabulary.
layout = acquisitions[0]
assert all((row["channel_names"], row["channel_labels"]) ==
           (layout["channel_names"], layout["channel_labels"]) for row in acquisitions)
antibodies = pd.read_csv(METADATA_ROOT / "AK_List.csv", sep=";", dtype=str, keep_default_na=False)
formal = pd.read_csv(S1, dtype=str, keep_default_na=False)
assert not (formal.Study == STUDY).any()
targets = antibodies.groupby("Metal Tag")["Target"].agg(list).to_dict()
aliases = {
    "SMA": "αSMA", "CCL21 / 6Ckine": "CCL21", "CD206 (MMR)": "CD206",
    "CD274 (B7-H1- PD-L1)": "CD274", "CD278 (ICOS)": "CD278",
    "CD279 (PD-1)": "CD279", "GITR / TNFRSF18": "CD357",
    "Indoleamine 2- 3-dioxygenase (IDO)": "IDO1",
    "Siglec-1 / CD169": "CD169", "pan Cytokeratin": "PanCK [AE1/AE3]",
}
body_markers = {"PanCK [AE1/AE3]", "αSMA", "CD3e", "CD45", "CD68"}
proposal = []
evidence = []
panel_rows = []
for index, (mass, label) in enumerate(zip(layout["channel_names"], layout["channel_labels"])):
    labeled = not bool(re.fullmatch(r"\d+[A-Za-z]+", label))
    source_targets = targets.get(mass, [])
    assert labeled == bool(source_targets), (mass, label, source_targets)
    if labeled:
        if mass == "Yb172":
            choices = formal[formal["Unified marker"].eq("vWF/CD31")][
                ["Unified marker", "MAP-KG mapping", "UniProt accession"]].drop_duplicates()
            assert len(choices) == 1
            unified, gene, accession = choices.iloc[0].tolist()
            basis = ("AK_List.csv lists CD31 EPR3094 and vWF poly vwf on Yb172, both at 4 ug/mL; "
                     "36 MCD acquisitions expose one Yb172 detector slot labeled vWF. Reuse current S1 mixed vWF/CD31 group; do not split into two protein signals.")
        elif mass == "Yb176":
            unified, gene, accession = "Cleaved Caspase3 (mapping pending)", "", ""
            basis = "Cleaved Caspase3 is a cleavage-specific measurement; current S1 total Caspase-3 mapping is not equivalent."
        else:
            target = source_targets[0]
            wanted = aliases.get(target)
            candidates = formal[formal["Unified marker"].eq(wanted)] if wanted else formal[formal["Original marker"].eq(target)]
            choices = candidates[["Unified marker", "MAP-KG mapping", "UniProt accession"]].drop_duplicates()
            if len(choices) != 1:
                raise ValueError(f"Unresolved S1 identity for {mass}: {target}; {len(choices)} choices")
            unified, gene, accession = choices.iloc[0].tolist()
            basis = f"AK_List.csv {mass} target {target}; S1 {candidates.iloc[0]['Study']} / {candidates.iloc[0]['Original marker']} -> {unified}."
            if mass == "Pr141":
                basis += " AE1 and AE3 antibodies share the Pr141 signal; S1 PanCK [AE1/AE3] represents their combined measurement."
        nuclear = unified == "Histone H3"
        body = unified in body_markers
        proposal.append(dict(zip(formal.columns, [STUDY, label, unified, gene, accession,
            "1" if nuclear else "2" if body else ""])))
        evidence.append(dict(source_channel_index=index, mass=mass, source_label=label,
            antibody_targets=" | ".join(source_targets), proposed_unified=unified,
            proposed_segmentation="nuclear" if nuclear else "body" if body else "",
            identity_evidence=basis,
            segmentation_evidence=("AK_List.csv In113 Histone H3; native nuclear protein measurement" if nuclear else
                "Current S1 marks this marker class as body input in other studies; proposed for review" if body else "")))
        marker = unified.removesuffix(" (mapping pending)")
        status = "mapping_pending" if unified.endswith(" (mapping pending)") else "mapped"
        role, keep, reason = "downstream", True, "antibody-tagged native measurement"
    else:
        marker, status, role, keep, reason = "", "excluded", "exclude", False, ""
        nuclear = body = False
    panel_rows.append(dict(panel_id="zenodo8041882_native133", source_channel_index=index,
        source_name=label, marker_id=marker, mapping_status=status, role=role,
        nuclear=nuclear, body=body, keep_channel=keep,
        output_channel_index=None, keep_reason=reason,
        exclusion_reason="" if keep else "Unlabeled detector mass; no antibody assignment in AK_List.csv",
        source_mass=mass, source_metal=re.match(r"[A-Za-z]+", mass).group(0)))
assert len(proposal) == 39 and len(panel_rows) == 133
panel = pd.DataFrame(panel_rows)
panel.loc[panel.keep_channel, "output_channel_index"] = range(39)
panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
pd.DataFrame(proposal, columns=formal.columns).to_csv(JOB / "candidate_s1_rows.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(evidence).to_csv(JOB / "candidate_s1_evidence.csv", index=False, encoding="utf-8-sig")

# Each shared P code names one patient-level response/resistance pair in the public release.
samples = []
sample_by_file = {}
for item in files:
    basename = Path(item["source_file"]).name
    match = re.search(r"_P([1-7])[_ ](response|resistance)", basename)
    assert match, basename
    subject, group = match.groups()
    sample_id = f"P{subject}_{group}"
    patient_key = f"{STUDY}__patient__P{subject}"
    sample_by_file[item["source_file"]] = sample_id
    row = dict.fromkeys(template("sample_metadata").columns, None)
    row.update(study=STUDY, sample_id=sample_id, patient_key=patient_key,
        source_subject_id=f"P{subject}",
        source_sample_id=Path(basename).stem,
        timepoint_label=group,
        tissue_type="tumor", primary_diagnosis="non-small cell lung cancer",
        metadata_notes=("The public MCD basename P code links the response and resistance specimens "
            "for one study patient. Supplementary Table 3 reports sex and histology by numeric "
            "patient number, but the CellProfiler Image.csv crosswalk to MCD P codes is absent; "
            "those individual values are not assigned. Age is reported only as a cohort summary."))
    samples.append(row)
samples_frame = pd.DataFrame(samples)

fields = pd.DataFrame([
    field("https://zenodo.org/records/8041882", "study accession and title", "sample_metadata",
        "study", "Assign the stable CellWeft study identifier to every row.", "Zenodo record 8041882"),
    field("raw/public/" + STUDY + "/images/*.mcd", "MCD basename P# and response/resistance", "sample_metadata",
        "sample_id;source_sample_id;timepoint_label", "Preserve one source-named specimen per MCD basename; derive sample ID from P# and response/resistance, and preserve the response/resistance label as the timepoint."),
    field("raw/public/" + STUDY + "/images/*.mcd; https://pmc.ncbi.nlm.nih.gov/articles/PMC10449840/",
        "MCD basename P#; article reports seven patients with paired response/resistance tumors", "sample_metadata",
        "patient_key;source_subject_id",
        "Use the shared P# prefix as the study-internal patient key; the same prefix labels one response and one resistance specimen. Namespace the key by study; do not infer the separate numeric PatientID from the absent Image.csv.",
        "exact shared P# prefix across paired MCD basenames"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC10449840/; raw/public/" + STUDY + "/metadata/Analysis IMC.Rmd",
        "Supplementary Table 3 Patient/Sex/Histology; author Rmd uses numeric PatientID", "sample_metadata",
        "sex_reported;histology",
        "Individual sex and histology are published by numeric patient number, but the CellProfiler Image.csv crosswalk from those numbers to MCD P# filenames is absent; leave both fields unassigned.",
        "no verified numeric PatientID to MCD P# crosswalk"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC10449840/", "Table 1 age range and median", "sample_metadata",
        "age_at_collection_years", "Age is reported only as a cohort-level median and range; do not assign an individual value.",
        "study cohort"),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC10449840/", "Methods and patient history", "sample_metadata",
        "tissue_type;primary_diagnosis", "The study reports paired NSCLC tumor samples for all seven patients."),
    field("https://pmc.ncbi.nlm.nih.gov/articles/PMC10449840/", "paper and Zenodo public release", "sample_metadata",
        "metadata_notes", "Record the source-level patient pairing and the absent acquisition-level clinical table."),
    field("raw/public/" + STUDY + "/metadata/Read IMC RAW data.Rmd",
        "Image.csv/acname PatientID, TherapyStatus, SampleID", "sample_metadata",
        "disease_status_at_collection;treatment_status_at_collection",
        "The author's R script reads these fields from CellProfiler Image.csv, which is absent from the local public files; leave unsupported per-image values blank.",
        "MCD acquisition name to missing CellProfiler Image.csv/acname"),
    field("raw/public/" + STUDY + "/images/*.mcd", "slide ID; acquisition ID; Description; MaxX; MaxY; ablation step", "image_metadata",
        "source_image_id;roi_id;region_annotation;source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um",
        "One independent full native acquisition per image; no multi-sample split evidence. Use exact coordinate-derived validity, not intensity.",
        "MCD basename + slide ID + acquisition ID"),
    field("raw/public/" + STUDY + "/metadata/AK_List.csv", "Metal Tag;Target", "panel_mapping",
        "source_name;marker_id;mapping_status;role;nuclear;body",
        "Match metal tag to MCD channel label and preserve detector order. Candidate S1 and per-row evidence kept in private jobdir.",
        "metal tag = MCD channel mass"),
    field("raw/public/" + STUDY + "/manifest.json", "Native MCD source members", "image_metadata",
        "source_files;source_revision", "Use published Zenodo 8041882 MCD filenames and immutable local source file versions."),
    field("raw/public/" + STUDY + "/images/*.mcd", "native pixel coordinates", "image_metadata",
        "valid_pixels_path", "MCD coordinate records identify measured pixels; absent records require valid_pixels.tiff."),
], columns=template("field_mapping").columns)

progress("candidate_metadata_ready", expected_images=len(acquisitions), source_mcd=len(files),
    source_channels=133, labeled_channels=39, samples=len(samples))

images = []
prior = OUTPUT / "images.parquet"
if prior.exists() and not args.rewrite:
    images = pd.read_parquet(prior).to_dict("records")
image_ids = {row["image_id"] for row in images}
if not args.verify_only:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    for source_item in files:
        native = NAS / source_item["source_file"]
        cached = LOCAL / native.name
        if not cached.is_file() or cached.stat().st_size != source_item["bytes"]:
            progress("source_staging", source_file=native.name, exported_images=len(images), expected_images=len(acquisitions))
            temporary = cached.with_suffix(".mcd.partial")
            with formal_read_lease(native):
                shutil.copy2(native, temporary)
                assert (native.stat().st_size, native.stat().st_mtime_ns) == (
                    source_item["bytes"], source_item["mtime_ns"])
            assert temporary.stat().st_size == source_item["bytes"]
            os.replace(temporary, cached)
        for item in (row for row in acquisitions if row["source_file"] == source_item["source_file"]):
            image_id = re.sub(r"[^A-Za-z0-9_]+", "_", native.stem) + f"_a{item['acquisition_id']}"
            if image_id in image_ids:
                continue
            progress("exporting", current_image=image_id, exported_images=len(images), expected_images=len(acquisitions))
            source_spec = SourceSpec(image_id=image_id, panel_id="zenodo8041882_native133", kind="mcd",
                path=cached, acquisition_id=item["acquisition_id"], channels=tuple(range(133)),
                source_revision=json.dumps([source_item["source_file"], source_item["bytes"], source_item["mtime_ns"]]))
            destination = OUTPUT / "images" / image_id / "input.tiff"
            with open_source(source_spec) as reader:
                result = export_input(reader, panel, destination, tile_size=512)
            if not result["admitted"]:
                raise ValueError(f"No downstream signal in {image_id}; retain and review source acquisition")
            row = dict.fromkeys(template("image_metadata").columns, None)
            row.update(study=STUDY, image_id=image_id, sample_id=sample_by_file[item["source_file"]],
                panel_id="zenodo8041882_native133", image_kind="roi",
                source_image_id=f"{native.name}::slide_{item['slide_id']}::acquisition_{item['acquisition_id']}",
                source_files=[item["source_file"]], source_format="mcd",
                source_acquisition_id=str(item["acquisition_id"]), source_axes="CYX",
                source_revision="zenodo:8041882:revision5", slide_id=Path(native.name).stem + "::slide_" + item["slide_id"],
                roi_id=str(item["acquisition_id"]), region_annotation=item["description"],
                source_x0_px=0, source_y0_px=0, source_x1_px=item["width"], source_y1_px=item["height"],
                pixel_size_x_um=item["pixel_size_x_um"], pixel_size_y_um=item["pixel_size_y_um"],
                input_image_path=str(destination),
                valid_pixels_path=str(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
                intensity_description="Native decoded MCD float32 Dual signal; no intensity normalization; 39 antibody-labeled channels retained.",
                split_method="source_acquisition", split_evidence="One MCD acquisition per source ROI; no spatial multi-sample annotation.",
                metadata_notes="Source filename group and specimen P code retained. Patient linkage unresolved; author whole-cell instance mask unverified.")
            for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px",
                        "pad_bottom_px", "source_dtype", "source_channel_count", "channel_count"):
                row[key] = result[key]
            images.append(row)
            image_ids.add(image_id)
            save_input_tables(OUTPUT, pd.DataFrame(images), samples_frame, panel, fields)
            progress("exporting", last_completed_image=image_id, exported_images=len(images), expected_images=len(acquisitions))
            print(f"candidate {len(images)}/{len(acquisitions)} {image_id}", flush=True)

if args.remap_ome:
    assert len(images) == len(acquisitions)
    write_receipt(dict(study=STUDY, status="remap_in_progress",
        preparer="/root/hiltbrunner_prepare", images=len(images)))
    remap = []
    for row in images:
        path = Path(row["input_image_path"])
        before_stat = path.stat()
        with tifffile.TiffFile(path) as tif:
            offsets = [(page.dataoffsets, page.databytecounts) for page in tif.pages]
            old = tif.pages[0].description
            assert len(tif.pages) == 39
        if 'Name="Yb172_CD31_vWF"' in old:
            assert old.count('Name="Yb172_CD31_vWF"') == 1
            new = old.replace('Name="Yb172_CD31_vWF"', 'Name="vWF/CD31"')
            with tifffile.TiffFile(path, mode="r+b") as tif:
                tif.pages[0].tags["ImageDescription"].overwrite(new.encode("utf-8"))
                tif.filehandle.flush()
                os.fsync(tif.filehandle.fileno())
        else:
            assert old.count('Name="vWF/CD31"') == 1
            new = old
        with tifffile.TiffFile(path) as tif:
            assert [(page.dataoffsets, page.databytecounts) for page in tif.pages] == offsets
            assert tif.pages[0].description == new
        after_stat = path.stat()
        remap.append(dict(image_id=row["image_id"], pixel_segments_unchanged=True,
            before_bytes=before_stat.st_size, after_bytes=after_stat.st_size,
            before_mtime_ns=before_stat.st_mtime_ns, after_mtime_ns=after_stat.st_mtime_ns))
        progress("ome_remap", remapped_images=len(remap), expected_images=len(acquisitions))
    write_json(JOB / "ome_remap_evidence.json", dict(old_marker_id="Yb172_CD31_vWF",
        new_marker_id="vWF/CD31", images=remap, all_pixel_segment_tables_unchanged=True))
    save_input_tables(OUTPUT, pd.DataFrame(images), samples_frame, panel, fields)

# Reopen every output header and compare three source-paired native pixel windows.
assert len(images) == len(acquisitions), (len(images), len(acquisitions))
kept = panel[panel.keep_channel].sort_values("output_channel_index")
checks = []
output_versions = []
sample_images = []
for group in ("response", "resistance"):
    sample_images.extend(next(row for row in images if f"_{group}" in row["sample_id"] and row["region_annotation"] in
                              ("Tcell_high", "Tcell_low")) for _ in range(1))
sample_images.append(next(row for row in images if row["region_annotation"] not in ("Tcell_high", "Tcell_low")))
sample_ids = {row["image_id"] for row in sample_images}
for row in images:
    path = Path(row["input_image_path"])
    stat = path.stat()
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 39 and all(p.dtype == np.dtype("float32") for p in tif.pages)
        assert all(p.shape == (row["height_px"], row["width_px"]) for p in tif.pages)
        names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter() if node.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()
        h3_index = int(kept.loc[kept.marker_id == "Histone H3", "output_channel_index"].iloc[0])
        h3_nonzero = int(np.count_nonzero(tif.pages[h3_index].asarray()))
        assert h3_nonzero > 0, row["image_id"]
    valid_version = None
    if row["valid_pixels_path"] is not None and not pd.isna(row["valid_pixels_path"]):
        valid_path = Path(row["valid_pixels_path"])
        with tifffile.TiffFile(valid_path) as valid_tif:
            assert len(valid_tif.pages) == 1
            assert valid_tif.pages[0].shape == (row["height_px"], row["width_px"])
            assert valid_tif.pages[0].dtype == np.dtype("uint8")
            assert np.isin(valid_tif.pages[0].asarray(), [0, 1]).all()
        valid_stat = valid_path.stat()
        valid_version = dict(path=str(valid_path), bytes=valid_stat.st_size,
                             mtime_ns=valid_stat.st_mtime_ns)
    output_versions.append(dict(image_id=row["image_id"], input_bytes=stat.st_size,
        input_mtime_ns=stat.st_mtime_ns, valid_pixels=valid_version, h3_nonzero=h3_nonzero))
    if row["image_id"] not in sample_ids:
        continue
    src = NAS / row["source_files"][0]
    cached = LOCAL / src.name
    spec = SourceSpec(row["image_id"], row["panel_id"], "mcd", cached, tuple(range(133)),
                      acquisition_id=int(row["source_acquisition_id"]))
    width, height = int(row["source_x1_px"]), int(row["source_y1_px"])
    windows = [(0, 0, min(64, width), min(64, height)),
               (max(0, width//2-32), max(0, height//2-32), min(width, width//2+32), min(height, height//2+32))]
    with open_source(spec) as reader, tifffile.TiffFile(path) as tif:
        for window in windows:
            native = reader.read_window(kept.source_channel_index.tolist(), window)
            x0,y0,x1,y1 = window
            padded = (x0+row["pad_left_px"], y0+row["pad_top_px"],
                      x1+row["pad_left_px"], y1+row["pad_top_px"])
            for i in range(39):
                np.testing.assert_array_equal(native[i], read_tiff_window(tif, i, padded))
    checks.append(dict(image_id=row["image_id"], panel_id=row["panel_id"], windows=windows,
                       checked_channels=39, exact=True))
source_versions = []
for item in files:
    stat = (NAS / item["source_file"]).stat()
    assert (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"])
    source_versions.append(dict(source_file=item["source_file"], bytes=stat.st_size,
                                mtime_ns=stat.st_mtime_ns))
write_json(JOB / "self_check.json", dict(all_headers=len(images), native_window_images=checks,
    source_versions=source_versions, output_versions=output_versions, status="pass"))
patient_key_by_sample = {row["sample_id"]: row["patient_key"] for row in samples}
patient_linked_images = sum(row.get("sample_id") in patient_key_by_sample for row in images)
write_json(JOB / "identity_evidence.json", dict(study=STUDY,
    source="MCD basename, slide ID and acquisition ID; Zenodo Read IMC RAW data.Rmd",
    image_to_sample=[dict(image_id=row["image_id"], source_file=row["source_files"][0],
        acquisition_id=row["source_acquisition_id"], sample_id=row["sample_id"])
        for row in images], sample_to_patient=[dict(sample_id=row["sample_id"],
        source_subject_id=row["source_subject_id"], patient_key=row["patient_key"])
        for row in samples], image_sample_coverage=len(images),
    samples=len(samples), patient_linked_samples=len(patient_key_by_sample),
    patient_linked_images=patient_linked_images,
    gap="Public MCD P1-P7 prefixes link each response/resistance pair and are used as study-scoped patient keys. The paper's individual sex/histology table uses numeric patient numbers, but the CellProfiler Image.csv/acname crosswalk to P# filenames is absent; age is available only as a cohort summary."))

reviews = []
for check in pd.read_csv(TEMPLATES / "review_checks.csv", keep_default_na=False).itertuples():
    reviews.append(dict(study=STUDY, preparer="/root/hiltbrunner_prepare", reviewer="", scope=check.scope,
        record_id=STUDY, check_id=check.check_id, result="unresolved",
        source_evidence=str(JOB / "inventory.json"), output_evidence=str(JOB / "self_check.json"),
        details="Candidate prepared; independent review and D01 pending. Public P# response/resistance patient pairing is recorded; individual sex/histology values are not assigned because the numeric PatientID-to-P# crosswalk is absent. Age is cohort-level only. Mixed Yb172 mapping and author mask semantics remain unresolved."))
pd.DataFrame(reviews).to_csv(OUTPUT / "review_results.csv", index=False, encoding="utf-8-sig")
write_receipt(dict(study=STUDY, status="candidate_complete",
    preparer="/root/hiltbrunner_prepare", images=len(images), samples=len(samples),
    patient_linked_images=patient_linked_images, panel_rows=len(panel), retained_channels=39,
    candidate_root=str(OUTPUT), self_check=str(JOB / "self_check.json")))
progress("candidate_complete", exported_images=len(images), expected_images=len(acquisitions),
    sample_linked_images=len(images), patient_linked_images=patient_linked_images,
    panel_rows=len(panel), retained_channels=39)
