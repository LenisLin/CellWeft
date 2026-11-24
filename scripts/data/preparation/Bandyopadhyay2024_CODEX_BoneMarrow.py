"""Prepare native CODEX inputs and evidence-based specimen links for bone marrow.

Inputs: 20 Figshare v1 QPTIFFs, verified_channel_map, study S1 marker rows,
publisher clinical Table S1 (mmc1), and cached author R analysis scripts.
Outputs: independent CYX input.tiff, the four input tables, preparer checks,
source overviews and resumable per-image receipts. Run exports through shared formal I/O coordination.
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
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)
from cellweft.data.preparation.io_coordination import formal_publish_lease

STUDY = "Bandyopadhyay2024_CODEX_BoneMarrow"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
PROGRESS = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0013_prepare_0/progress.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
REPOSITORY = META / "source_metadata/author_repository/CODEX_Analysis"
CLINICAL = META / "source_metadata/publisher_mmc1.xlsx"
PREPARER = "Codex CLI study preparer 0013_prepare_0"


def write_json(path: Path, payload: dict) -> None:
    """Replace progress/receipts only after their complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def progress(stage: str, images: pd.DataFrame, **details) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage, expected_images=20,
        completed_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        updated_at=datetime.now(timezone.utc).isoformat(), **details))


def source_inventory() -> list[dict]:
    """Reuse native proofs; check current file lengths and all full-resolution labels."""
    verified = json.loads((META / "verified_channel_map.json").read_text())
    for panel in verified["panels"].values():
        assert (NAS / panel["source_marker_path"]).read_text().splitlines() == [row["marker"] for row in panel["channels"]]
    found = []
    for item in verified["images"]:
        path = NAS / item["path"]
        proof = json.loads((META / "native_verification" / Path(item["verification_path"]).name).read_text())
        assert proof["readable_complete"] and proof["exact_marker_match"]
        assert path.stat().st_size == proof["bytes"]
        channels = proof["full_resolution_channels"]
        names = [row["marker"] for row in channels]
        assert names == [row["marker"] for row in verified["panels"][item["panel"]]["channels"]]
        with tifffile.TiffFile(path) as tif:
            descriptions = [ET.fromstring(tif.pages[row["page_index"]].description) for row in channels]
            assert [xml.findtext("Biomarker") for xml in descriptions] == names
            assert all(xml.findtext("ImageType") == "FullResolution" for xml in descriptions)
            page = tif.pages[channels[0]["page_index"]]
            assert list(page.shape) == channels[0]["shape"] and str(page.dtype) == channels[0]["dtype"]
            scale = []
            for tag in ("XResolution", "YResolution"):
                a, b = page.tags[tag].value
                assert page.tags["ResolutionUnit"].value == 3
                scale.append(10000 * b / a)
        found.append(dict(**item, image_id=path.stem, names=names,
            pages=[row["page_index"] for row in channels], height=page.shape[0], width=page.shape[1],
            dtype=str(page.dtype), pixel_size_um=scale, slide_id=descriptions[0].findtext("SlideID"),
            source_study_name=descriptions[0].findtext("StudyName"), bytes=proof["bytes"],
            mtime_ns=path.stat().st_mtime_ns))
    actual = {relative(p) for p in (NAS / "raw/public" / STUDY).rglob("*.qptiff")}
    assert actual == {item["path"] for item in found}
    return found


def biological_samples(inventory: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Connect donor/specimen IDs using Table S1 and author project assignments.

    The final AML script explicitly corrects patient numbering for specimens 183
    and 191. NSM clinical rows are joined by unique pathology suffix. If an AML
    specimen suffix conflicts, the clinical row is used only when the corrected
    subject and explicit author timepoint identify exactly one Table S1 row.
    """
    clinical = pd.read_excel(CLINICAL, sheet_name="Table S1", header=None).where(lambda x: x.notna(), None)
    normal = clinical[clinical[0].astype(str).str.fullmatch(r"H\d+")]
    diseased = clinical[clinical[0].astype(str).str.match(r"(?:AML|NSM)\d+_")]
    final_path = REPOSITORY / "Step6_AML_ReferenceMapping/RefMap_AML_ForManuscript_Final.R"
    final = final_path.read_text()
    corrections = dict(re.findall(r'(AML\d+_\d+)\$orig.ident <- "([^"]+)"', final))
    levels = re.search(r'levels = c\(("SB67_NBM[^\n]+)\)\)', final).group(1)
    projects = re.findall(r'"([^"]+)"', levels)
    labels = re.findall(r'"([^"]+)"', re.search(r'orig.ident_annotations <- c\(([^\n]+)\)', final).group(1))
    timepoints = dict(zip(projects, labels, strict=True))
    rows, links = [], []
    for item in inventory:
        raw_id = item["image_id"].split("_Scan")[0]
        candidates = list(REPOSITORY.rglob(f"*{raw_id}_Create_SeuratObj.R"))
        assert len(candidates) == 1
        author_file = candidates[0]
        project = re.search(r'project\s*=\s*"([^"]+)"', author_file.read_text()).group(1)
        specimen = re.search(r"_(NBM\d+)_", project).group(1)
        assert re.search(r"NBM\d+", item["source_study_name"]).group() == specimen
        notes = []
        clinical_row = None
        if raw_id.startswith("H"):
            clinical_row = normal[normal[0] == raw_id].iloc[0]
            subject = raw_id
            timepoint = None
        elif raw_id.startswith("AML"):
            suffix = raw_id.rsplit("_", 1)[1]
            corrected = [value for value in corrections.values() if f"_{suffix}_CODEX" in value]
            project = corrected[0] if corrected else project
            identity = timepoints[project]
            subject, timepoint = identity.split("_")
            exact = diseased[diseased[0].astype(str).str.match(fr"{subject}_{suffix}(?: |$)")]
            if len(exact) == 1:
                clinical_row = exact.iloc[0]
            elif len(exact) == 0:
                source_timepoint = {"Dx": "Dx", "MRD": "PostTx"}.get(timepoint)
                subject_rows = diseased[diseased[0].astype(str).str.match(fr"{re.escape(subject)}_\d+ ")]
                if source_timepoint is not None:
                    labeled_timepoints = subject_rows[0].astype(str).str.extract(r"\(([^()]*)\)\s*$")[0]
                    candidates = subject_rows[labeled_timepoints == source_timepoint]
                    if len(candidates) == 1:
                        clinical_row = candidates.iloc[0]
                        notes.append("Published specimen suffix conflicts; clinical row matched by the author-corrected patient and unique explicit timepoint.")
                if clinical_row is None:
                    notes.append("Clinical Table S1 specimen suffix conflicts with final author code; no unique patient/timepoint row was available, so specimen-level clinical fields remain blank.")
            else:
                notes.append("Clinical Table S1 has multiple exact specimen rows; specimen-level clinical fields remain blank.")
            if raw_id.split("_")[0] != subject:
                notes.append("Final author code explicitly corrects the patient prefix used in the published QPTIFF label.")
        else:
            suffix = raw_id.rsplit("_", 1)[1]
            exact = diseased[diseased[0].astype(str).str.fullmatch(fr"NSM\d+_{suffix}")]
            assert len(exact) == 1
            clinical_row = exact.iloc[0]
            subject = str(clinical_row[0]).split("_")[0]
            timepoint = None
            if subject != raw_id.split("_")[0]:
                notes.append("NSM source prefixes differ between image and Table S1; matched by the unique pathology specimen suffix.")
        row = dict(study=STUDY, sample_id=specimen, source_sample_id=specimen,
            patient_key=f"{STUDY}:{subject}", source_subject_id=subject,
            cohort_id=item["panel"], sample_type="bone marrow tissue", preservation_method="ffpe",
            anatomical_site="femoral head" if raw_id.startswith("H") else "iliac crest",
            collection_method="resection" if raw_id.startswith("H") else "biopsy",
            tissue_type="normal" if not raw_id.startswith("AML") else "tumor",
            case_control_status="reference" if raw_id.startswith("H") else ("case" if raw_id.startswith("AML") else "control"),
            metadata_notes=" ".join(notes))
        if raw_id.startswith("H"):
            row.update(age_at_collection_years=clinical_row[5], sex_reported=str(clinical_row[4]).lower())
        else:
            if clinical_row is not None:
                row.update(age_at_collection_years=clinical_row[7], sex_reported=str(clinical_row[6]).lower())
                if clinical_row[9] is not None:
                    row["vital_status"] = {"Living":"alive", "Deceased":"deceased"}[clinical_row[9]]
            if raw_id.startswith("AML"):
                row.update(primary_diagnosis="acute myeloid leukemia", timepoint_label="pre_treatment" if timepoint == "Dx" else "post_treatment")
                if clinical_row is not None:
                    row["collection_day_from_index"] = clinical_row[1]
                    row["index_event"] = "treatment_start"
                    if clinical_row[2] is not None:
                        row["index_regimen"] = clinical_row[2]
                else:
                    paired = diseased[diseased[0].astype(str).str.startswith(subject + "_")]
                    assert paired[6].nunique() == 1
                    row["sex_reported"] = str(paired.iloc[0][6]).lower()
        rows.append(row)
        demographic_rows = ([int(clinical_row.name)+1] if clinical_row is not None else
            [int(index)+1 for index in diseased[diseased[0].astype(str).str.startswith(subject + "_")].index])
        links.append(dict(image_id=item["image_id"], sample_id=specimen,
            raw_slide_id=item["slide_id"], source_raw_id=raw_id, source_project=project,
            embedded_study_name=item["source_study_name"],
            source_subject_id=subject, clinical_excel_row=int(clinical_row.name)+1 if clinical_row is not None else None,
            demographics_excel_rows=json.dumps(demographic_rows),
            author_file=relative(author_file), final_author_file=relative(final_path),
            source_timepoint=timepoint, notes=" ".join(notes)))
    samples = pd.DataFrame(rows).reindex(columns=pd.read_csv(TEMPLATES / "sample_metadata.csv").columns)
    assert not samples.sample_id.duplicated().any()
    return samples, pd.DataFrame(links)


def field_rules() -> pd.DataFrame:
    common = dict(study=STUDY, source_sheet_or_table="Table S1 / author CODEX analysis / QPI XML",
        value_mapping="{}", time_origin_rule="AML collection days are relative to treatment start as labelled in Table S1.",
        conflict_rule="Final author AML numbering overrides old image prefixes; NSM uses unique pathology suffix; AML clinical rows use an exact specimen suffix or a unique corrected subject plus explicit timepoint. Retain any source suffix conflict in metadata_notes.")
    rules = [
        (relative(CACHE / "sample_image_crosswalk.csv"), "embedded_study_name;source_project;source_subject_id;sample_id;clinical_excel_row", "sample_metadata", "sample_id;source_sample_id;patient_key;source_subject_id;cohort_id", "QPTIFF embedded NBM StudyName matches author NBM project; unique pathology suffix -> clinical subject", "Crosswalk records original author files and clinical row numbers. Public pseudonymous subject IDs are study-qualified; repeat AML timepoints share the corrected patient."),
        ("study configuration", "STUDY", "sample_metadata", "study", "Registered preparation study identifier", "Use the fixed public study identifier for this preparation; it is a dataset key, not a patient attribute."),
        (relative(CLINICAL), "Age;Sex", "sample_metadata", "age_at_collection_years;sex_reported", "Exact specimen, or unique corrected subject plus explicit author timepoint; normal donor", "Copy source values and lowercase sex. If specimen suffix conflicts, use only a unique patient/timepoint row and record the conflict; otherwise leave specimen clinical fields blank. Source row numbers are in clinical_excel_row / demographics_excel_rows of sample_image_crosswalk.csv."),
        (relative(CLINICAL), "Status", "sample_metadata", "vital_status", "Exact specimen, or unique corrected subject plus explicit author timepoint", "Living -> alive; Deceased -> deceased. Follow-up cutoff is unspecified; no survival times or events derived."),
        (relative(CLINICAL), "Treatment Timepoint (days);Treatment", "sample_metadata", "collection_day_from_index;index_event;index_regimen", "Exact specimen, or unique corrected subject plus explicit author timepoint", "Copy source days/regimen; treatment timepoint defines treatment-start origin. Record the published suffix conflict when the patient/timepoint bridge is unique; otherwise leave specimen clinical fields blank."),
        (relative(CACHE / "sample_image_crosswalk.csv"), "source_timepoint", "sample_metadata", "timepoint_label", "Author final corrected project -> Sample_Name", "Dx -> pre_treatment, MRD -> post_treatment; do not interpret sample histology as formal treatment response."),
        (relative(META / "primary_paper.html"), "STAR Methods: human samples; CODEX imaging; AML section", "sample_metadata", "sample_type;preservation_method;anatomical_site;collection_method;tissue_type;case_control_status;primary_diagnosis", "Figshare panel and source specimen group", "CODEX FFPE tissue; healthy femoral heads and AML/NSM iliac crest biopsies; reference/case/control reflect study design."),
        (relative(CACHE / "sample_image_crosswalk.csv"), "notes", "sample_metadata", "metadata_notes", "Source conflict or qualified join recorded per specimen", "Retain source-label discrepancies and explain when public patient/timepoint evidence uniquely supports a clinical-row match."),
        (relative(CACHE / "source_inventory.json"), "path;image_id;slide_id;source_file_id;source_study_name;pixel_size_um", "image_metadata", "source_files;source_image_id;slide_id;source_revision;region_annotation;pixel_size_x_um;pixel_size_y_um", "Exact Figshare file ID and full-resolution page index", "Inventory comes from native proof and QPI XML/TIFF tags. Copy source StudyName to region_annotation; centimetre TIFF resolution converts to um/pixel."),
        (relative(CACHE / "sample_image_crosswalk.csv"), "image_id;sample_id", "image_metadata", "sample_id", "image_id", "20 independent scans correspond to 20 biological specimens, 18 subjects; no patient inferred from ROI numbering."),
        (relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "source_channel_index;output_channel_index;marker_id;role;keep_channel", "Exact study + embedded Biomarker", "Normal 54/54 and AML/NSM 53/54 retained; only NPM1C is S1 excluded. DAPI retained for nuclear input."),
    ]
    return pd.DataFrame([dict(common, source_file=f, source_field=s, target_table=t,
        target_field=target, join_keys=join, transformation_rule=rule,
        notes="Clinical retrieval and source conflicts recorded in metadata_search.json; preparer checks are not independent acceptance.")
        for f,s,t,target,join,rule in rules]).reindex(columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)


def metadata_evidence() -> None:
    """Keep exact public metadata locations and the observed source-label conflicts."""
    tree = json.loads((META / "source_metadata/github_tree.json").read_text())
    files = [{"local_path":relative(path),
        "url":"https://raw.githubusercontent.com/tanlabcode/spatial-bonemarrow-atlas/main/" + str(metadata_source_relative(path, META / 'source_metadata/author_repository'))}
        for path in (META / "source_metadata/author_repository").rglob("*.R")]
    write_json(CACHE / "metadata_search.json", dict(
        local_sources=[relative(META / name) for name in ("source_manifest.json", "verified_channel_map.json", "primary_paper.html", "figshare_25122311.json", "figshare_25123538.json")],
        release_urls=["https://plus.figshare.com/articles/dataset/25122311", "https://plus.figshare.com/articles/dataset/25123538"],
        collection_url="https://api.figshare.com/v2/collections/7174914/articles",
        ancillary_metadata_urls=["https://api.figshare.com/v2/articles/25127579", "https://api.figshare.com/v2/articles/25127657"],
        ancillary_findings="Published mask filenames retain the same old AML/NSM prefixes as the QPTIFFs; the Seurat release lists three RDS payloads. Only the small listing metadata was fetched.",
        clinical_url="https://ars.els-cdn.com/content/image/1-s2.0-S0092867424004082-mmc1.xlsx",
        clinical_local=relative(CLINICAL), clinical_bytes=CLINICAL.stat().st_size,
        clinical_sheet="Table S1", paper_doi="10.1016/j.cell.2024.04.013",
        repository_url="https://github.com/tanlabcode/spatial-bonemarrow-atlas",
        repository_tree_sha=tree["sha"], author_files=files,
        retrieval_notes="PMC supplemental link returned a browser-check HTML page; the publisher mmc1 Excel was retrieved and opened successfully. No image payload or mask was downloaded.",
        connection_coverage=dict(images=20, biological_specimens=20, patient_linked_samples=20, distinct_patients=18),
        conflicts=["Final AML analysis explicitly corrects old patient prefixes for pathology specimens 183 and 191.",
            "Table S1 NSM prefixes for 1086 and 1720 differ from the QPTIFF release; unique pathology suffix joins are preserved.",
            "Table S1 AML1 diagnostic specimen is labelled 191, while final author analysis assigns the QPTIFF specimen ending 183 to AML1_Dx. The unique corrected patient plus explicit Dx timepoint selects the Table S1 row; the source suffix discrepancy remains in sample metadata notes."],
        controller_requests=["Independent review of native inputs and source identity joins.",
            "Independently review the unique corrected patient/timepoint clinical-row bridge and confirm the suffix conflict remains documented."] ))


def overview(item: dict) -> None:
    """Display native reduced pyramid DAPI/CD45/CD34; no full WSI decoding."""
    from PIL import Image
    path = CACHE / "overviews" / f"{item['image_id']}.png"
    if path.exists():
        return
    with tifffile.TiffFile(NAS / item["path"]) as tif:
        pages = tif.series[0].levels[-1].pages
        rgb = np.stack([pages[item["names"].index(name)].asarray(maxworkers=2)
                        for name in ("CD45", "CD34", "DAPI")], axis=-1)
    for channel in range(3):
        values = rgb[..., channel]
        positive = values[values > 0]
        if len(positive):
            rgb[..., channel] = np.clip(values.astype(float) * 255 / np.percentile(positive, 99.5), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    image.thumbnail((600, 600))
    temporary = path.with_name(path.stem + ".tmp.png")
    image.save(temporary)
    os.replace(temporary, path)


def image_record(item: dict, receipt: dict) -> dict:
    h, w = item["height"], item["width"]
    geometry = {key:receipt[key] for key in ("height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_dtype", "source_channel_count", "channel_count")}
    return dict(study=STUDY, image_id=item["image_id"], sample_id=links.loc[item["image_id"], "sample_id"],
        panel_id=item["panel"], image_kind="roi", source_image_id=Path(item["path"]).name,
        source_files=[item["path"]], source_format="qptiff", source_axes="CYX",
        source_revision=f"figshare_v1:file_id={item['source_file_id']}", slide_id=item["slide_id"],
        region_annotation=item["source_study_name"], source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
        pixel_size_x_um=item["pixel_size_um"][0], pixel_size_y_um=item["pixel_size_um"][1],
        input_image_path=relative(OUTPUT / "images" / item["image_id"] / "input.tiff"),
        intensity_description="Native uint8 PhenoCycler-Fusion release intensities after source stitching, registration and background subtraction; no downstream rescaling.",
        split_method="already_independent", split_evidence=relative(CACHE / "sample_image_crosswalk.csv"),
        metadata_notes=links.loc[item["image_id"], "notes"], **geometry)


def preparer_results(images: pd.DataFrame, complete: bool) -> None:
    """Record the actual clinical issue and handoff; self-checks remain separate."""
    rows = []
    for affected in crosswalk[crosswalk.clinical_excel_row.isna()].itertuples(index=False):
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample",
            record_id=affected.sample_id, check_id="M02", result="unresolved",
            source_evidence=relative(CLINICAL) + ";" + affected.final_author_file,
            output_evidence=relative(OUTPUT / "sample_metadata.csv"),
            details=json.dumps(dict(image_ids=[affected.image_id],
                issue="Table S1 AML1 Dx uses suffix 191 but final author code explicitly assigns corrected AML1 to specimen 183.",
                action="Specimen clinical values and collection age remain blank; author-supported patient/Dx identity and concordant paired-patient sex are retained.",
                preparer_finding=True), ensure_ascii=False)))
    details = dict(independent_review_pending=True, image_export_complete=complete,
        exported_images=len(images), expected_images=20, sample_rows=len(samples),
        source_images_with_sample_connection=len(crosswalk),
        exported_sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()), distinct_patients=samples.patient_key.nunique(),
        metadata_trace=relative(CACHE / "metadata_search.json"),
        identity_crosswalk=relative(CACHE / "sample_image_crosswalk.csv"),
        note="Preparer handoff only; no independent input_binding/pass generated. Self-check evidence is in preparer_validation.json when completed.")
    identity_details = dict(details, image_ids=crosswalk.image_id.tolist(),
        checked_join_keys="Embedded QPI NBM StudyName -> author project -> corrected AML subject / exact clinical suffix or unique explicit patient timepoint; healthy donor SAMPLE ID.",
        preparer_identity_connection_gaps=[],
        remaining_gap="Independent M01 review pending; the diagnostic clinical suffix conflict is separately scoped under M02.")
    rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample",
        record_id=STUDY, check_id="M01", result="unresolved",
        source_evidence=relative(CACHE / "metadata_search.json") + ";" + relative(CACHE / "sample_image_crosswalk.csv"),
        output_evidence=relative(OUTPUT / "sample_metadata.csv"), details=json.dumps(identity_details, ensure_ascii=False)))
    rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
        record_id=STUDY, check_id="D01", result="unresolved",
        source_evidence=relative(CACHE / "source_inventory.json") + ";" + relative(CLINICAL),
        output_evidence=relative(OUTPUT / "images.parquet"), details=json.dumps(details, ensure_ascii=False)))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        reviewed = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.DataFrame()
        if len(reviewed):
            reviewed = reviewed[reviewed.reviewer.astype(str).str.strip().ne("")]
        pd.concat([reviewed, pd.DataFrame(rows)], ignore_index=True).to_csv(
            review_path, index=False, encoding="utf-8-sig")


# Source and specimen identity evidence is small and can be rebuilt without pixels.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=["metadata", "overview", "export", "validate"], default="export")
parser.add_argument("--image-id", action="append", help="Optional bounded export selection; existing indices are merged.")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
inventory = source_inventory()
write_json(CACHE / "source_inventory.json", dict(images=inventory, expected_images=20,
    uncompressed_square_output_bytes=sum(max(x["height"], x["width"])**2 * (54 if x["panel"] == "normal_marrow" else 53) for x in inventory),
    native_verification_reused=True, source_files_read_only=True))
if args.phase == "overview":
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (5*420, 4*450), "#181818")
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(inventory):
        overview(item)
        with Image.open(CACHE / "overviews" / f"{item['image_id']}.png") as picture:
            picture.thumbnail((410, 410))
            x, y = index % 5 * 420, index // 5 * 450
            canvas.paste(picture, (x+(410-picture.width)//2,y+25))
            draw.text((x+5,y+5),item["image_id"],fill="white")
    canvas.save(CACHE / "source_overview_montage.png")
    raise SystemExit(0)
panels = pd.concat([panel_from_s1(STUDY, key, rows["names"], S1)
    for key, rows in {item["panel"]:item for item in inventory}.items()], ignore_index=True)
samples, crosswalk = biological_samples(inventory)
metadata_evidence()
crosswalk.to_csv(CACHE / "sample_image_crosswalk.csv", index=False, encoding="utf-8-sig")
links = crosswalk.set_index("image_id", verify_integrity=True)
fields = field_rules()
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns)
if len(images):
    images["sample_id"] = images.image_id.map(links.sample_id)
    images["metadata_notes"] = images.image_id.map(links.notes)
save_input_tables(OUTPUT, images, samples, panels, fields,
    replace_metadata_studies={STUDY})
preparer_results(images, len(images) == 20)
progress("metadata_complete", images, gaps=["Independent review pending", "Clinical Table S1 AML1 diagnostic specimen suffix conflict"])

if args.phase == "export":
    # Smallest scans first; publish a real closed index and receipt after each scan.
    for item in sorted(inventory, key=lambda x:max(x["height"],x["width"])**2):
        if args.image_id and item["image_id"] not in args.image_id:
            continue
        image_id = item["image_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = CACHE / "receipts" / f"{image_id}.json"
        panel = panels[panels.panel_id == item["panel"]]
        identity = dict(source_path=item["path"], bytes=item["bytes"], mtime_ns=item["mtime_ns"],
            source_indices=panel.loc[panel.keep_channel, "source_channel_index"].tolist(),
            output_names=panel.loc[panel.keep_channel, "marker_id"].tolist())
        if receipt_path.exists() and destination.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt["identity"] == identity
            assert destination.stat().st_size == receipt["output_bytes"]
            assert destination.stat().st_mtime_ns == receipt["output_mtime_ns"]
        else:
            side = max(item["height"],item["width"])
            needed = side**2 * int(panel.keep_channel.sum())
            assert shutil.disk_usage(NAS).free > needed + 100*1024**3
            assert shutil.disk_usage("/tmp").free > 64*1024**3
            progress("exporting", images, current_image=image_id, gaps=["Independent review pending", "One clinical specimen suffix conflict"])
            overview(item)
            source = SourceSpec(image_id, item["panel"], "tiff", NAS / item["path"],
                tuple(item["pages"]), pixel_size_um=tuple(item["pixel_size_um"]))
            with open_source(source) as reader:
                receipt = export_input(reader, panel, destination, tile_size=512)
            assert receipt["admitted"], "All admitted marker channels were zero; record exclusion before continuing."
            receipt.update(identity=identity, output_bytes=destination.stat().st_size,
                output_mtime_ns=destination.stat().st_mtime_ns)
            write_json(receipt_path, receipt)
        record = image_record(item, receipt)
        images = pd.concat([images[images.image_id != image_id], pd.DataFrame([record])], ignore_index=True)
        save_input_tables(OUTPUT, images, samples, panels, fields)
        preparer_results(images, len(images) == 20)
        progress("image_completed", images, last_completed_image=image_id,
            gaps=["Independent review pending", "Clinical AML1 Dx suffix conflict"])
        print(json.dumps(dict(image_id=image_id, completed_images=len(images), output_bytes=receipt["output_bytes"])), flush=True)
    progress("export_complete" if len(images) == 20 else "batch_complete", images,
        gaps=["Independent review pending", "Clinical AML1 Dx suffix conflict"])

if args.phase == "validate":
    # All headers; signal/boundary blocks per panel and padding orientation.
    results = []
    sampled = set()
    for item in inventory:
        selected = images[images.image_id == item["image_id"]]
        if selected.empty:
            continue
        row = selected.iloc[0]
        panel = panels[(panels.panel_id == item["panel"]) & panels.keep_channel]
        destination = NAS / row.input_image_path
        with tifffile.TiffFile(destination) as tif:
            assert len(tif.pages) == int(row.channel_count)
            assert all(p.shape == (int(row.height_px),int(row.width_px)) and p.dtype == np.uint8 for p in tif.pages)
            ome = ET.fromstring(tif.ome_metadata)
            names = [node.attrib["Name"] for node in ome.iter() if node.tag.endswith("}Channel")]
            assert names == panel.marker_id.tolist()
            pixels = next(node for node in ome.iter() if node.tag.endswith("}Pixels")).attrib
            assert int(pixels["SizeC"]) == int(row.channel_count)
            assert int(pixels["SizeZ"]) == int(pixels["SizeT"]) == 1
            assert np.isclose(float(pixels["PhysicalSizeX"]),float(row.pixel_size_x_um),rtol=1e-12,atol=0)
            assert np.isclose(float(pixels["PhysicalSizeY"]),float(row.pixel_size_y_um),rtol=1e-12,atol=0)
            assert pixels["PhysicalSizeXUnit"] == pixels["PhysicalSizeYUnit"] == "µm"
            evidence = dict(image_id=item["image_id"], panel_id=item["panel"],
                source_format=row.source_format, source_axes=row.source_axes,
                split_method=row.split_method, header=True, pixel_windows=[])
            group = (item["panel"], int(row.pad_left_px) > 0, int(row.pad_top_px) > 0)
            if group not in sampled:
                evidence["padding_orientation"] = "left_right" if group[1] else "top_bottom"
                evidence["source_channel_indices"] = panel.source_channel_index.tolist()
                evidence["output_channel_indices"] = panel.output_channel_index.tolist()
                source = SourceSpec(item["image_id"], item["panel"], "tiff", NAS / item["path"], tuple(item["pages"]))
                proof = json.loads((META / "native_verification" / Path(item["verification_path"]).name).read_text())
                channel = proof["full_resolution_channels"][15]
                segment = max(channel["readable_regions"], key=lambda x:x["pixel_max"])["segment_index"]
                columns = (item["width"] + 511)//512
                sx, sy = segment % columns * 512, segment // columns * 512
                with open_source(source) as reader:
                    probe = reader.read_window([15], (sx,sy,min(sx+512,item["width"]),min(sy+512,item["height"])))[0]
                assert probe.any()
                py, px = np.unravel_index(probe.argmax(),probe.shape)
                sx = max(0,min(sx+int(px)-16,item["width"]-32))
                sy = max(0,min(sy+int(py)-16,item["height"]-32))
                for window in [(sx,sy,sx+32,sy+32), (0,0,32,32)]:
                    x0,y0,x1,y1 = window
                    out_window = (x0+int(row.pad_left_px),y0+int(row.pad_top_px),x1+int(row.pad_left_px),y1+int(row.pad_top_px))
                    with open_source(source) as reader:
                        values = reader.read_window(panel.source_channel_index.tolist(), window)
                    saved = np.stack([read_tiff_window(tif, int(index),out_window) for index in panel.output_channel_index])
                    assert saved.dtype == values.dtype and np.array_equal(values,saved)
                    evidence["pixel_windows"].append(dict(source_window=window, all_channels=int(row.channel_count), equal=True, contains_signal=bool(values.any())))
                side = int(row.width_px)
                middle = side//2
                margins = [(0,middle,int(row.pad_left_px),middle+32),
                    (side-int(row.pad_right_px),middle,side,middle+32),
                    (middle,0,middle+32,int(row.pad_top_px)),
                    (middle,side-int(row.pad_bottom_px),middle+32,side)]
                evidence["padding_windows"] = []
                for window in margins:
                    if window[0] < window[2] and window[1] < window[3]:
                        assert all(not read_tiff_window(tif,int(i),window).any() for i in panel.output_channel_index)
                        evidence["padding_windows"].append(dict(window=window,all_channels=int(row.channel_count),all_zero=True))
                sampled.add(group)
            results.append(evidence)
    write_json(CACHE / "preparer_validation.json", dict(images=results, checked_headers=len(results),
        sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
        distinct_patients=samples.patient_key.nunique(), image_sample_links=int(images.sample_id.notna().sum()),
        independent_review=False))
    progress("prepared_pending_independent_review", images, checked_headers=len(results),
        sampled_panels=len({key[0] for key in sampled}), sampled_layouts=len(sampled),
        pixel_blocks=sum(len(x["pixel_windows"]) for x in results),
        gaps=["Independent review pending", "Clinical AML1 Dx suffix conflict; specimen clinical fields withheld"])
    print(json.dumps(dict(checked_headers=len(results),sample_linked_images=int(images.sample_id.notna().sum()),sample_rows=len(samples),distinct_patients=samples.patient_key.nunique())),flush=True)
