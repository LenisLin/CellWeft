"""Prepare six native Allam 2024 SLO IMC sections as candidate study inputs.

Inputs: the 156 selected Zenodo TIFF members, source member plan, formal S1.
Outputs: private inventory, S1 proposal, five study tables and native input TIFFs.
Donor labels are matched by tissue and letter; matching letters across tissues are not merged.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)

STUDY = "Allam2024_IMC_NormalSLO"
PANEL = "zenodo8303732_native26"
PANEL_ORDER = (
    "CD38", "Vimentin", "BCL6", "ICOS1", "CD11b", "CD11c", "FoxP3", "CD4",
    "CD20", "CD8", "C-Myc", "PD1", "Ki67", "COL1", "CD3", "CD27",
    "CD138", "CXCR5", "EZH2", "CD21", "CD86", "CD83", "CXCR4",
    "H3K27me3", "DNA1", "DNA2",
)
SUPPLEMENT = "https://academic.oup.com/pnasnexus/article/3/8/pgae285/7717711#supplementary-data"
PLAN = Path("processed") / STUDY / "preparation_evidence" / "ready_for_download_allam_slo_selected.json"
PAPER_PROOF = Path("processed") / STUDY / "preparation_evidence" / "imc_b_primary_main_review.json"
DEFAULT_JOB = Path("/home/lenislin/Experiment/projects/CellWeft/cache/new_studies_20261107") / STUDY

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--output-root", type=Path, help="Parent of <study>; defaults to formal processed")
parser.add_argument("--jobdir", type=Path, default=DEFAULT_JOB)
parser.add_argument("--s1", type=Path, help="Use a reviewed private candidate S1 for export")
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--verify-only", action="store_true", help="Refresh candidate tables and check existing inputs")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
root = args.data_root
raw = root / "raw/public" / STUDY
output = (args.output_root or root / "processed") / STUDY
job = args.jobdir
job.mkdir(parents=True, exist_ok=True)
manifest = json.loads((root / PLAN).read_text())
publication = json.loads((root / PAPER_PROOF).read_text())["allam"]
members = manifest["source_files"]
groups = manifest["source_groups"]
assert len(members) == 156 and len(groups) == 6


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def progress(stage: str, completed: int, issues: list[str]) -> None:
    save_json(job / "progress.json", dict(study=STUDY, stage=stage,
        updated_at=datetime.now(timezone.utc).isoformat(), source_files=156,
        total_images=6, candidate_exported_images=completed, issues=issues,
        log=str(job / "prepare.log")))


# Check every downloaded member against the selected ZIP member and native TIFF header.
inventory = []
for group in groups:
    selected = [item for item in members if item["name"].startswith(group + "/")]
    selected = sorted(selected, key=lambda item: PANEL_ORDER.index(Path(item["name"]).stem))
    actual = {path.name for path in (raw / group).iterdir() if path.is_file()}
    assert actual == {Path(item["name"]).name for item in selected}, group
    files = []
    for item in selected:
        path = raw / item["name"]
        stat = path.stat()
        assert stat.st_size == item["uncompressed_bytes"], path
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            assert len(tif.pages) == 1 and page.ndim == 2 and page.samplesperpixel == 1
            assert page.dtype == np.dtype("uint8")
            shape = tuple(page.shape)
            description = page.description
            assert json.loads(description)["shape"] == list(shape)
            assert page.resolutionunit == 1
        files.append(dict(path=path.relative_to(root).as_posix(),
                          bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                          label=path.stem, shape=list(shape), dtype="uint8"))
    assert len(files) == 26 and len({tuple(x["shape"]) for x in files}) == 1
    assert tuple(x["label"] for x in files) == PANEL_ORDER
    inventory.append(dict(source_group=group, image_id=group.replace(" ", "_"),
                          shape=files[0]["shape"], source_axes="YX per TIFF; CYX across files",
                          ordered_labels=[x["label"] for x in files], files=files,
                          split_decision="one complete published group frame; no core/ROI map within group",
                          mask_status="published image ZIP selection has no paired mask member"))
assert len({tuple(x["ordered_labels"]) for x in inventory}) == 1
labels = inventory[0]["ordered_labels"]
assert set(labels) == set(publication["all_original_measurements"])
save_json(job / "inventory.json", dict(study=STUDY, archive_url=manifest["source_archive_url"],
    selected_member_plan=str(root / PLAN), source_proof=str(root / PAPER_PROOF),
    paper_doi=manifest["paper_doi"], source_file_count=156, total_images=6,
    native_dtype="uint8", pixel_size_um=None, ordered_panel_basis=SUPPLEMENT + " (Supplementary Table 1)",
    pixel_scale_note="TIFF ResolutionUnit=1 and X/YResolution=1/1 are not a calibrated micrometre scale",
    sample_evidence="six published tissue-specific donor labels; exact Tissue+Donor labels establish six distinct study-local donor keys",
    vendor_tissue_ids_unbound=["SU1", "SM2", "lly06N022A9", "lly50N001A4", "S09-10377", "S09-11278"],
    groups=inventory))

# Propose study-specific S1 rows from existing exact/synonymous marker identities.
formal_s1 = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
existing = pd.read_csv(formal_s1, dtype=str, keep_default_na=False)
has_formal_s1 = existing.Study.eq(STUDY).any()
overrides = {"ICOS1": "ICOS", "COL1": "COL1", "C-Myc": "C-Myc"}
body = {"CD20", "CD3", "CD8", "CD11c", "Vimentin"}
proposal, rationale = [], []
for label in labels:
    lookup = overrides.get(label, label)
    matches = existing[existing["Original marker"].str.casefold() == lookup.casefold()]
    assert len(matches), label
    if label == "H3K27me3":
        mapped = ("H3K27me3 (mapping pending)", "", "")
        note = "Current preparation specification explicitly retains H3K27me3; existing S1 rows exclude it. No canonical protein mapping or nuclear use proposed."
    else:
        candidates = matches[matches["Unified marker"].ne("Excluded")]
        counts = candidates.groupby(["Unified marker", "MAP-KG mapping", "UniProt accession"]).size().sort_values(ascending=False)
        mapped = counts.index[0]
        note = "Reuse existing S1 exact label/established synonym; majority non-excluded mapping."
    segmentation = "1" if label in {"DNA1", "DNA2"} else ("2" if label in body else "")
    proposal.append(dict(Study=STUDY, **{"Original marker": label,
        "Unified marker": mapped[0], "MAP-KG mapping": mapped[1],
        "UniProt accession": mapped[2], "Segmentation channel": segmentation}))
    support = matches[(matches["Unified marker"] == mapped[0]) &
                      (matches["MAP-KG mapping"] == mapped[1]) &
                      (matches["UniProt accession"] == mapped[2])]
    rationale.append(dict(source_label=label, source_member=inventory[0]["files"][labels.index(label)]["path"],
        source_measurement="191Ir/193Ir intercalator" if label.startswith("DNA") else "published protein-channel filename",
        reused_s1_studies="|".join(support.Study.unique()[:8]),
        reused_s1_original_marker=lookup if len(support) else "",
        segmentation_basis=("published 191Ir/193Ir intercalator and existing DNA S1 nuclear rule" if label.startswith("DNA")
                            else "study-specific candidate body coverage; image intensity review pending" if label in body else ""),
        note=note))
save_csv(job / "candidate_s1_rows.csv", proposal, list(existing.columns))
save_csv(job / "candidate_s1_evidence.csv", rationale, list(rationale[0]))
progress("inventory_and_candidate_s1_ready", 0, [
    "vendor tissue-ID-to-folder mapping unresolved; published tissue-specific donor keys are retained",
    "author whole-cell mask described in paper but no paired mask in selected image source",
    "H3K27me3 retention conflicts with historical S1 exclusion; candidate mapping pending",
    "physical pixel scale not encoded in downloaded TIFFs",
])
if args.metadata_only:
    raise SystemExit(0)

# The private S1 is the sole authority for the candidate panel and export.
s1 = args.s1 or (formal_s1 if has_formal_s1 else job / "candidate_s1_rows.csv")
panel = panel_from_s1(STUDY, PANEL, labels, s1)
assert panel.loc[panel.nuclear, "source_name"].tolist() == ["DNA1", "DNA2"]
kept = panel[panel.keep_channel].sort_values("output_channel_index")
samples_template = pd.read_csv(TEMPLATES / "sample_metadata.csv", dtype=str, keep_default_na=False)
samples_template["source_reported_age_years"] = pd.Series(dtype=str)
published_demographics = {
    "TS Donor A": ("M", "20"), "TS Donor E": ("M", "35"),
    "LN Donor A": ("F", "21"), "LN Donor E": ("F", "64"),
}
tissue_names = {"INT": "Intestinal Tissue", "LN": "Lymph Node", "TS": "Tonsil"}
sample_rows = []
for group in groups:
    row = dict.fromkeys(samples_template.columns, "")
    tissue = group.split()[0]
    donor_label = group.split(maxsplit=1)[1]
    source_subject_id = f"{tissue_names[tissue]} {donor_label}"
    row.update(study=STUDY, sample_id=group.replace(" ", "_"),
        source_sample_id=group, sample_type="de-identified vendor tissue section",
        tissue_type="normal", anatomical_site={"INT": "small intestine", "LN": "lymph node", "TS": "tonsil"}[tissue],
        patient_key=f"{STUDY}:{source_subject_id}", source_subject_id=source_subject_id,
        preservation_method="ffpe",
        metadata_notes="The folder label matches the tissue-specific donor label in Supplementary Table 2. "
        "Patient key uses the exact tissue-specific donor label; repeated letters across tissues remain separate. "
        "vendor tissue IDs are not linked to these folders.")
    if group in published_demographics:
        row["sex_reported"] = {"M": "male", "F": "female"}[published_demographics[group][0]]
        row["source_reported_age_years"] = published_demographics[group][1]
        row["metadata_notes"] += " Sex and age from publication Supplementary Table 2."
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows, columns=samples_template.columns)
field_template = pd.read_csv(TEMPLATES / "field_mapping.csv", dtype=str, keep_default_na=False)
field_rows = [dict(study=STUDY, source_file=str(root / PLAN), source_sheet_or_table="study registry",
    source_field="study identifier", target_table="sample_metadata", target_field="study",
    join_keys="study", value_mapping="",
    transformation_rule="assign the stable CellWeft study identifier to every row", time_origin_rule="",
    conflict_rule="", notes="Identity is the public study accession/title mapped to the local study id."),
    dict(study=STUDY, source_file=str(root / PLAN), source_sheet_or_table="source_files",
    source_field="exact group directory", target_table="sample_metadata", target_field="sample_id",
    join_keys="exact tissue-group folder label", value_mapping="spaces -> underscores",
    transformation_rule="normalize the exact group label into a unique sample key", time_origin_rule="",
    conflict_rule="do not merge matching donor letters across tissues", notes="Six rows; one per public source group."),
    dict(study=STUDY, source_file=SUPPLEMENT, source_sheet_or_table="Supplementary Table 2",
    source_field="Tissue; Donor", target_table="sample_metadata", target_field="source_subject_id",
    join_keys="exact tissue label + donor letter", value_mapping="",
    transformation_rule="preserve the tissue-specific public donor label; do not merge repeated letters across tissues",
    time_origin_rule="", conflict_rule="retain tissue prefix; do not merge matching letters across tissues",
    notes="All six tissue-specific donor labels match the published group labels."),
    dict(study=STUDY, source_file=str(root / PLAN), source_sheet_or_table="source_files",
    source_field="exact group directory", target_table="sample_metadata", target_field="source_sample_id",
    join_keys="exact tissue-group folder label", value_mapping="",
    transformation_rule="retain the exact published source group label", time_origin_rule="",
    conflict_rule="", notes="One row per public tissue-section group."),
    dict(study=STUDY, source_file=str(root / PAPER_PROOF), source_sheet_or_table="paper scope and selected source groups",
    source_field="normal tissue group and tissue label", target_table="sample_metadata",
    target_field="sample_type;tissue_type;anatomical_site;preservation_method",
    join_keys="exact tissue-group folder label", value_mapping="normal tissue; FFPE",
    transformation_rule="map the source tissue code to its expanded anatomical site",
    time_origin_rule="", conflict_rule="", notes="Study describes normal tissue sections; site is encoded in source group."),
    dict(study=STUDY, source_file=SUPPLEMENT, source_sheet_or_table="Supplementary Table 2",
    source_field="Tissue; Donor; Sex; Age", target_table="sample_metadata",
    target_field="sex_reported;source_reported_age_years", join_keys="exact tissue label + donor letter",
    value_mapping="M -> male; F -> female; age in years; n/a -> blank",
    transformation_rule="exact tissue-specific donor-row match; retain age as reported without assigning collection-time semantics",
    time_origin_rule="age as reported; collection timing not separately confirmed",
    conflict_rule="INT donor demographics remain blank", notes="Four of six rows report age and sex; INT rows are n/a."),
    dict(study=STUDY, source_file=SUPPLEMENT, source_sheet_or_table="Supplementary Table 2",
    source_field="Tissue; Donor", target_table="sample_metadata", target_field="patient_key",
    join_keys="exact tissue label + donor letter", value_mapping=f"{STUDY}:<exact source_subject_id>",
    transformation_rule="namespace the exact tissue-specific donor label; retain the tissue prefix in the key",
    time_origin_rule="", conflict_rule="do not merge matching donor letters across tissues",
    notes="Supplementary Table 2 lists six tissue-qualified donor records; known same-letter tonsil/lymph-node rows have differing reported demographics."),
    dict(study=STUDY, source_file="; ".join((str(root / PAPER_PROOF), SUPPLEMENT)),
    source_sheet_or_table="paper and Supplementary Table 2", source_field="source group and publication metadata",
    target_table="sample_metadata", target_field="metadata_notes", join_keys="exact tissue label + donor letter",
    value_mapping="", transformation_rule="record source coverage and unresolved vendor-ID-to-folder link",
    time_origin_rule="", conflict_rule="", notes="Supplementary Table 2 reports donor metadata; vendor tissue IDs remain unlinked."),
    dict(study=STUDY, source_file=str(root / PLAN), source_sheet_or_table="source_files",
    source_field="name", target_table="image_metadata", target_field="source_files",
    join_keys="exact group directory and marker filename", value_mapping="",
    transformation_rule="ZIP central-directory order; 26 scalar TIFFs form one CYX acquisition",
    time_origin_rule="", conflict_rule="", notes="One complete frame per group; no physical multi-sample split evidence.")]
fields = pd.DataFrame(field_rows, columns=field_template.columns)

if args.verify_only:
    images = pd.read_parquet(output / "images.parquet")
    save_input_tables(output, images, samples, panel, fields, data_root=root)
    images = pd.read_parquet(output / "images.parquet")
    assert len(images) == 6 and set(images.image_id) == {x["image_id"] for x in inventory}
    by_id = images.set_index("image_id", verify_integrity=True)
    checked = []
    for item in inventory:
        image_id = item["image_id"]
        row = by_id.loc[image_id]
        h, w = item["shape"]
        side = max(h, w)
        path = output / "images" / image_id / "input.tiff"
        assert path.is_file() and row.height_px == row.width_px == side
        assert row.source_channel_count == 26 and row.channel_count == len(kept)
        assert row.pad_left_px + w + row.pad_right_px == side
        assert row.pad_top_px + h + row.pad_bottom_px == side
        assert row.sample_id == image_id and row.panel_id == PANEL
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == len(kept)
            assert all(page.shape == (side, side) and page.dtype == np.dtype("uint8") for page in tif.pages)
            channels = ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")
            assert [channel.attrib["Name"] for channel in channels] == kept.marker_id.tolist()
        checked.append(dict(image_id=image_id, source_shape=item["shape"], output_side=side,
                            source_files=26, retained_channels=len(kept), header="pass"))
    pixel_checks = []
    for image_id in ("INT_Donor_B", "LN_Donor_E", "TS_Donor_A"):
        item = next(x for x in inventory if x["image_id"] == image_id)
        row = by_id.loc[image_id]
        h, w = item["shape"]
        windows = [(max(0, w // 2 - 32), max(0, h // 2 - 32),
                    min(w, w // 2 + 32), min(h, h // 2 + 32)),
                   (0, 0, min(w, 64), min(h, 64))]
        with tifffile.TiffFile(output / "images" / image_id / "input.tiff") as destination:
            for channel in kept.itertuples(index=False):
                source_path = root / item["files"][channel.source_channel_index]["path"]
                with tifffile.TiffFile(source_path) as source_tif:
                    for window in windows:
                        x0, y0, x1, y1 = window
                        shifted = (x0 + row.pad_left_px, y0 + row.pad_top_px,
                                   x1 + row.pad_left_px, y1 + row.pad_top_px)
                        original = read_tiff_window(source_tif, 0, window)
                        candidate = read_tiff_window(destination, int(channel.output_channel_index), shifted)
                        assert np.array_equal(original, candidate), (image_id, channel.source_name, window)
                pixel_checks.append(dict(image_id=image_id, channel=channel.source_name,
                                         windows=[list(x) for x in windows], dtype="uint8", result="equal"))
            if row.pad_left_px or row.pad_top_px:
                assert not read_tiff_window(destination, 0, (0, 0, 64, 64)).any()
    save_json(job / "self_check.json", dict(study=STUDY, complete_source_members=156,
        all_image_headers=checked, sampled_pixel_windows=pixel_checks,
        sample_rows=len(samples), patient_linked_rows=int(samples.patient_key.ne("").sum()),
        reviewer="", independent_review="pending", d01="pending"))
    review_columns = pd.read_csv(TEMPLATES / "review_results.csv", dtype=str).columns.tolist()
    review_rows = []
    for check_id in ("S01", "I01", "I02", "I03", "I04", "P01", "P02", "P03", "I05", "M01", "M02", "D01"):
        result = "pending" if check_id == "D01" else "preparer_checked"
        detail = (
            "Supplementary Table 2 donor labels match all six groups by tissue and letter; "
            "the complete tissue-qualified donor label forms a study-local patient key; vendor-ID-to-folder mapping remains unknown."
            if check_id == "M01" else
            "All populated sample fields map to publication sources; the two intestinal rows report n/a for age and sex."
            if check_id == "M02" else
            "Independent reviewer has not signed this candidate result."
        )
        review_rows.append(dict(study=STUDY, preparer="Allam2024_study_preparer", reviewer="",
            scope="study", record_id=STUDY, check_id=check_id, result=result,
            source_evidence=str(job / "inventory.json"),
            output_evidence=str(job / "self_check.json"), details=detail))
    save_csv(output / "review_results.csv", review_rows, review_columns)
    progress("candidate_verified_by_preparer", 6,
             ["independent review and D01 pending", "vendor tissue-ID-to-folder mapping unresolved", "author mask binding unresolved"])
    print("verified 6 headers and 3 grouped pixel samples", flush=True)
    raise SystemExit(0)

image_rows = []
selected_images = inventory[:args.limit] if args.limit is not None else inventory
for item in selected_images:
    group, image_id = item["source_group"], item["image_id"]
    h, w = item["shape"]
    side = max(h, w)
    paths = tuple(root / f["path"] for f in item["files"])
    source = SourceSpec(image_id=image_id, panel_id=PANEL, kind="tiff_planes", path=paths[0],
        paths=paths, channels=(0,) * len(paths), source_revision=None,
        intensity_description="published uint8 TIFF intensities; no normalization")
    destination = output / "images" / image_id / "input.tiff"
    with open_source(source) as reader:
        result = export_input(reader, panel, destination, data_root=root)
    assert result["admitted"] and result["source_dtype"] == "uint8"
    row = dict(study=STUDY, image_id=image_id, sample_id=image_id, panel_id=PANEL,
        image_kind="roi", source_image_id=group,
        source_files=[f["path"] for f in item["files"]], source_format="tiff_planes",
        source_axes="CYX across 26 scalar YX TIFFs", source_dtype="uint8",
        source_revision="Zenodo record 8303732 IMC_images.zip selected ZIP members; local size/mtime in inventory",
        region_annotation=group.split()[0], source_x0_px=0, source_y0_px=0,
        source_x1_px=w, source_y1_px=h,
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=side, width_px=side, source_channel_count=26,
        channel_count=len(kept), input_image_path=str(destination),
        intensity_description="published uint8 TIFF intensities; no normalization",
        split_method="already_independent", split_evidence=str(job / "inventory.json"),
        metadata_notes="One native published group frame; no sub-ROI/core map. Physical scale and vendor tissue-ID-to-folder mapping unresolved.")
    image_rows.append(row)
    # Save only the current candidate range, allowing a bounded --limit run to resume.
    save_input_tables(output, pd.DataFrame(image_rows), samples, panel, fields, data_root=root)
    progress("candidate_export_in_progress", len(pd.read_parquet(output / "images.parquet")), [
        "independent review and D01 pending", "vendor tissue-ID-to-folder mapping unresolved", "author mask binding unresolved"])
    print(f"exported {image_id} {result['channel_count']}x{side}x{side}", flush=True)

review = pd.read_csv(TEMPLATES / "review_results.csv", dtype=str, keep_default_na=False)
review_rows = []
for item in inventory:
    if not (output / "images" / item["image_id"] / "input.tiff").is_file():
        continue
    review_rows.append(dict(study=STUDY, preparer="Allam2024_study_preparer", reviewer="",
        scope="image", record_id=item["image_id"], check_id="P03", result="preparer_checked",
        source_evidence=str(job / "inventory.json"),
        output_evidence=str(output / "images" / item["image_id"] / "input.tiff"),
        details="Native source header, 26 ordered labels, uint8 and square output header checked; independent review pending."))
review_rows.append(dict(study=STUDY, preparer="Allam2024_study_preparer", reviewer="",
    scope="study", record_id=STUDY, check_id="M01", result="preparer_checked",
    source_evidence=str(root / PAPER_PROOF), output_evidence=str(output / "sample_metadata.csv"),
    details="Supplementary Table 2 donor labels match all six groups by tissue and letter; tissue-qualified labels are namespaced as six distinct study-local donor keys. Vendor-ID-to-folder mapping remains unknown."))
review_rows.append(dict(study=STUDY, preparer="Allam2024_study_preparer", reviewer="",
    scope="study", record_id=STUDY, check_id="D01", result="pending",
    source_evidence=str(job / "inventory.json"), output_evidence=str(output / "images.parquet"),
    details="Independent reviewer and controller admission required; candidate inputs only."))
save_csv(output / "review_results.csv", review_rows, list(review.columns))
progress("candidate_export_complete" if len(image_rows) == 6 else "candidate_export_partial",
         len(pd.read_parquet(output / "images.parquet")),
         ["independent review and D01 pending", "vendor tissue-ID-to-folder mapping unresolved", "author mask binding unresolved"])
