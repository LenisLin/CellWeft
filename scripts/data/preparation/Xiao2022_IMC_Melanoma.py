"""Deliver the locally verified Xiao melanoma MCD and published TIFF ROIs."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Xiao2022_IMC_Melanoma"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(ROOT, STUDY) / 'local_intake_v1'
RAW = ROOT / "raw/public" / STUDY / "zenodo_6838169"
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    ROOT / "cache/study_input_batch/20261023_224251/jobs" / STUDY / "0741_repair_1")))
TEMPLATES = PROJECT / "docs/data/templates/study_input"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
SUPPLEMENT = ROOT / "raw/public" / STUDY / "metadata/43856_2022_197_MOESM2_ESM.pdf"
ROI_COUNT_AUDIT = OUTPUT / "cohort_roi_count_audit.json"
TIFF_DOMAIN_AUDIT = OUTPUT / "published_roi20_native_domain.json"
STAGE_ROOT = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / "mcd_source_files"
AUTHOR_IDENTITY = META.parent / "author_code_v2_identity_20261027T0800Z"


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, completed: int, next_image: str | None, **details: object) -> None:
    payload = dict(study=STUDY, phase="repair", repair=1, stage=stage,
                   expected_images=219, exported_images=completed,
                   excluded_images=len(excluded), next_image=next_image,
                   updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    JOB.mkdir(parents=True, exist_ok=True)
    target = JOB / "progress.json"
    with formal_publish_lease(target, directory=JOB, data_root=ROOT):
        temporary = target.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(target)


def mapping(path: Path, source_field: str, target_table: str, target_field: str,
            rule: str, notes: str, join_keys: str = "") -> dict[str, str]:
    return dict(study=STUDY, source_file=relative(path), source_sheet_or_table="",
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join_keys, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


def source_spec(record: object, image_id: str, source: Path) -> SourceSpec:
    kind = "mcd" if record.source_kind == "mcd" else "tiff"
    return SourceSpec(image_id=image_id, panel_id=str(record.panel_id), kind=kind,
                      path=source, channels=tuple(range(41)),
                      acquisition_id=int(record.acquisition_id) if kind == "mcd" else None,
                      pixel_size_um=(1.0, 1.0),
                      intensity_description="Native unnormalized float32 IMC intensities")


def checked_header(path: Path, shape: tuple[int, int], names: list[str]) -> None:
    with formal_read_lease(path, data_root=ROOT):
        with tifffile.TiffFile(path) as tif:
            assert len(tif.pages) == len(names)
            assert all(page.shape == shape and page.dtype.name == "float32" for page in tif.pages)
            actual = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter()
                      if node.tag.endswith("}Channel")]
            assert actual == names


def staged_mcd(source: Path, image_id: str) -> Path:
    """Copy one unchanged native file for all acquisitions in its source group."""
    original = source.stat()
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    staged = STAGE_ROOT / source.name
    if staged.is_file():
        current = staged.stat()
        if (current.st_size, current.st_mtime_ns) == (original.st_size, original.st_mtime_ns):
            return staged
    if shutil.disk_usage(STAGE_ROOT).free < 64 * 1024**3 + original.st_size:
        raise OSError("local 64 GiB reserve for MCD staging")
    partial = staged.with_suffix(staged.suffix + ".partial")
    with formal_read_lease(source, data_root=ROOT):
        shutil.copy2(source, partial)
        latest = source.stat()
    if (latest.st_size, latest.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
        raise RuntimeError("Native MCD changed during local staging")
    partial.replace(staged)
    return staged


def repair_author_cohort_metadata() -> None:
    """Apply the evidenced 158-ROI identity rule without exporting image pixels."""
    import argparse
    import zlib

    from cellweft.data.io.publication import resource_lock
    from cellweft.data.io.writers import write_json
    from cellweft.data.preparation.inputs import file_revision
    from cellweft.data.preparation.io_coordination import formal_write_lease

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-cohort-metadata-only", action="store_true")
    parser.add_argument("--author-evidence", type=Path, required=True)
    parser.add_argument("--metadata-job", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    evidence = args.author_evidence.resolve()
    job = args.metadata_job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    selected = evidence / "official_selected"
    members = ("data/input/meta/ROI_meta_afterQC.csv", "data/input/meta/IMC_QC2.csv",
               "data/input/clidata/mel_puch_cli_IMC26.csv")
    source_receipt = json.loads((evidence / "official_selected_receipt.json").read_text())
    declared = {row["name"]: row for row in source_receipt["selected_members"]}
    for name in members:
        data = (selected / name).read_bytes()
        assert len(data) == declared[name]["size"]
        assert f"{zlib.crc32(data):08x}" == declared[name]["crc32"]
    evidence_paths = [selected / name for name in members]
    evidence_paths += [META / "source_overlap.json", META / "rois.csv", SUPPLEMENT,
                       evidence / "official_selected_receipt.json"]
    evidence_versions = {str(path): file_revision(path) for path in evidence_paths}

    paths = [OUTPUT / name for name in ("images.parquet", "sample_metadata.csv",
                                        "field_mapping.csv", "panel_mapping.csv", "review_results.csv")]
    before = {path.name: file_revision(path) for path in paths}
    images = pd.read_parquet(paths[0])
    old_samples = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    fields = pd.read_csv(paths[2], dtype=str, keep_default_na=False)
    assert len(images) == 217 and images.image_id.is_unique
    assert len(old_samples) == 32
    inputs_before = {str(row.image_id): file_revision(ROOT / row.input_image_path)
                     for row in images.itertuples(index=False)}
    for path in paths:
        assert file_revision(path) == before[path.name]
        shutil.copyfile(path, job / ("before_" + path.name))

    def label(value: str) -> str:
        match = re.fullmatch(r"(?:PD-)?(\d+)F?", str(value))
        assert match is not None, "Unexpected author label syntax"
        return f"{int(match.group(1))}F"

    def released_key(value: str) -> str:
        match = re.search(r"_(\d+F?)_ROI(\d+)", Path(value).name)
        assert match is not None, "Unexpected released ROI filename"
        return f"{label(match.group(1))}_ROI{int(match.group(2))}"

    after = pd.read_csv(selected / members[0], dtype=str, keep_default_na=False)
    qc = pd.read_csv(selected / members[1], dtype=str, keep_default_na=False)
    author_clinical = pd.read_csv(selected / members[2], dtype=str, keep_default_na=False)
    assert len(after) == 158 and after.sampleROI2.is_unique and after.QC.eq("1.0").all()
    qc_keys = {f"{label(row['sample'])}_ROI{int(row['ROI'])}": row
               for row in qc.to_dict("records")}
    assert len(qc_keys) == len(qc) == 206
    author_labels = {label(value) for value in author_clinical.Sample_ID}
    assert len(author_labels) == len(author_clinical) == 26
    after_by_key = after.set_index("sampleROI2").to_dict("index")
    assert {key.split("_ROI")[0] for key in after_by_key} == author_labels
    assert all(qc_keys[key]["QC"] == "1" for key in after_by_key)
    overlap = json.loads((META / "source_overlap.json").read_text())
    native = pd.read_csv(META / "rois.csv", dtype=str, keep_default_na=False)
    native_by_key = {(Path(row["mcd_path"]).name, int(row["acquisition_id"])): row["image_id"]
                     for row in native.to_dict("records")}
    assert len(native_by_key) == len(native)
    assignments = {}
    for row in overlap:
        key = released_key(row["public_source"])
        image_id = native_by_key[(Path(row["local_source"]).name, int(row["acquisition_id"]))]
        assert key in after_by_key and image_id not in assignments
        assignments[image_id] = key
    assert len(assignments) == len(set(assignments.values())) == 158
    assert set(assignments.values()) == set(after_by_key)
    assert set(assignments) <= set(images.image_id)
    pending = sorted(set(images.image_id) - set(assignments))
    assert len(pending) == 59
    old_by_id = images.set_index("image_id")
    assert old_by_id.loc[list(assignments)].source_format.value_counts().to_dict() == {"mcd": 157, "tiff": 1}
    old_sample_by_id = old_samples.set_index("sample_id")
    sample_rows = []
    for author_label in sorted(author_labels):
        image_ids = [image_id for image_id, key in assignments.items()
                     if key.split("_ROI")[0] == author_label]
        old_ids = set(old_by_id.loc[image_ids].sample_id)
        rows = old_sample_by_id.loc[sorted(old_ids)].reset_index()
        assert rows.patient_key.eq(f"{STUDY}:{author_label[:-1]}").all()
        invariant_columns = [name for name in rows if name not in
                             ("sample_id", "source_sample_id", "metadata_notes")]
        assert len(rows[invariant_columns].drop_duplicates()) == 1
        row = rows.iloc[0].to_dict()
        row.update(sample_id=f"{STUDY}__FFPE_{author_label}", source_sample_id=author_label,
                   metadata_notes="Author final FFPE sample/H&E panel label; membership restricted to "
                   "158 released after-QC ROIs. Clinical values retained from original Supplementary Table 2.")
        sample_rows.append(row)
    samples = pd.DataFrame(sample_rows, columns=old_samples.columns)
    assert len(samples) == samples.patient_key.nunique() == 26
    updated = images.copy()
    updated["sample_id"] = None
    updated["region_annotation"] = updated.region_annotation.astype(object)
    for index, row in updated.iterrows():
        image_id = row.image_id
        if image_id in assignments:
            key = assignments[image_id]
            updated.at[index, "sample_id"] = f"{STUDY}__FFPE_{key.split('_ROI')[0]}"
            updated.at[index, "region_annotation"] = after_by_key[key]["region"]
            updated.at[index, "metadata_notes"] = (
                "Author after-QC ROI membership joined by released TIFF -> native source/acquisition; "
                "biological sample is final FFPE/H&E label, not an MCD file group.")
        else:
            updated.at[index, "region_annotation"] = None
            updated.at[index, "metadata_notes"] = (
                "Biological sample and patient identity pending; outside author after-QC ROI scope. "
                f"Original native source group: {images.at[index, 'sample_id']}")
    assert updated.sample_id.notna().sum() == 158
    assert set(updated.sample_id.dropna()) == set(samples.sample_id)
    unchanged = [name for name in images if name not in ("sample_id", "region_annotation", "metadata_notes")]
    pd.testing.assert_frame_equal(images[unchanged], updated[unchanged])

    superseded = {relative(META / name) for name in ("rois.csv", "samples.csv", "summary.json")}
    fields = fields.loc[~fields.source_file.isin(superseded)].copy()
    clinical_rows = fields.source_file.eq(relative(SUPPLEMENT)) & fields.target_table.eq("sample_metadata")
    fields.loc[clinical_rows, "join_keys"] = "Author final FFPE/H&E sample label numeric ID = Supplementary Table 2 ID"
    fields.loc[clinical_rows, "transformation_rule"] = (
        "Join only the 26 evidenced final FFPE sample labels to the published clinical ID; "
        "preserve original PDF clinical values and blanks. Code-release clinical values are not replacements.")
    new_fields = [
        mapping(AUTHOR_IDENTITY / members[0], "sampleROI2;QC;region", "image_metadata",
                "sample_id;region_annotation", "Only QC=1 author after-QC keys joined through exact "
                "released TIFF/native source and acquisition correspondence; retain author region text.",
                "158 linked images; all other 59 inputs retain null biological sample and patient admission.",
                "sampleROI2 = released TIFF ROI key -> source_overlap -> rois(image_id)"),
        mapping(AUTHOR_IDENTITY / members[2], "Sample_ID", "sample_metadata", "sample_id;patient_key;source_sample_id",
                "One biological sample per author final FFPE/H&E panel label; documented PD- prefix/F "
                "suffix normalization connects the clinical ID. Never infer extra native acquisition identity.",
                "26 final FFPE samples from Methods and Supplementary Figure 1; one label spans two MCD files."),
        mapping(SUPPLEMENT, "Methods;Supplementary Figure 1;Table 2", "sample_metadata", "sample_id;patient_key",
                "The final 26 FFPE samples and 26 patient-labeled H&E panels define the bounded biological sample.",
                "H&E rectangles lack ROI numbers; exact ROI membership comes from author ROI table, not rectangle order."),
    ]
    fields = pd.concat([fields, pd.DataFrame(new_fields, columns=fields.columns)], ignore_index=True)
    updated.to_parquet(job / "images.parquet", index=False)
    samples.to_csv(job / "sample_metadata.csv", index=False)
    fields.to_csv(job / "field_mapping.csv", index=False)
    write_json(job / "identity_scope.private.json", {"assignments": assignments, "pending_image_ids": pending,
               "original_image_sample_ids": dict(zip(images.image_id, images.sample_id))})
    write_json(job / "candidate_validation.json", {"expected_images": 219, "exported_images": 217,
               "sample_linked_images": 158, "sample_rows": 26, "patient_linked_samples": 26,
               "pending_images": 59, "inputs_before": inputs_before, "metadata_before": before,
               "source_versions": evidence_versions,
               "scientific_repair": 2, "pixel_compute": False})
    if not args.apply:
        print(json.dumps({"status": "candidate_validated", "images": 217, "admitted_identity": 158, "pending": 59}))
        return

    with formal_write_lease(OUTPUT, data_root=ROOT, priority="high"):
        with resource_lock(OUTPUT):
            assert all(file_revision(path) == before[path.name] for path in paths)
            assert all(file_revision(Path(path)) == version for path, version in evidence_versions.items())
            assert all(file_revision(ROOT / row.input_image_path) == inputs_before[str(row.image_id)]
                       for row in images.itertuples(index=False))
            assert not AUTHOR_IDENTITY.exists(), "Immutable author evidence directory already exists"
            for name in members:
                destination = AUTHOR_IDENTITY / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(selected / name, destination)
            shutil.copyfile(evidence / "official_selected_receipt.json", AUTHOR_IDENTITY / "official_selected_receipt.json")
            for path in paths[:3]:
                temporary = path.with_name(f".{path.name}.author_cohort_repair2.{os.getpid()}.partial")
                shutil.copyfile(job / path.name, temporary)
                os.chmod(temporary, path.stat().st_mode & 0o777)
                temporary.replace(path)
            assert all(file_revision(path) == before[path.name] for path in paths[3:])
    write_json(job / "report.json", {"study": STUDY, "phase": "repair", "repair": 2,
               "status": "partial", "expected_images": 219, "exported_images": 217,
               "sample_linked_images": 158, "sample_rows": 26, "patient_linked_samples": 26,
               "patient_linked_images": 158, "pending_images": 59,
               "script": "scripts/data/preparation/Xiao2022_IMC_Melanoma.py",
               "independent_review_pending": True, "D01_changed": False,
               "unchanged_input_versions": inputs_before,
               "metadata_before": before, "metadata_after": {path.name: file_revision(path) for path in paths},
               "source_versions": evidence_versions,
               "source_evidence": str(AUTHOR_IDENTITY), "closed_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "repair2_metadata_published_pending_independent_review", "images": 217,
                      "sample_linked_images": 158, "sample_rows": 26, "pending": 59}))


if "--author-cohort-metadata-only" in sys.argv:
    repair_author_cohort_metadata()
    raise SystemExit(0)
if AUTHOR_IDENTITY.exists():
    raise SystemExit("Author-cohort metadata supersedes the provisional MCD-file grouping; use --author-cohort-metadata-only.")


with formal_read_lease(META, data_root=ROOT):
    inventory = pd.read_csv(META / "images.csv", dtype={"acquisition_id": int})
    rois = pd.read_csv(META / "rois.csv", keep_default_na=False)
    local_samples = pd.read_csv(META / "samples.csv", keep_default_na=False)
    native = json.loads((META / "native_verification.json").read_text())
    overlaps = json.loads((META / "source_overlap.json").read_text())
    ordered = pd.read_csv(META / "panels.csv")
assert len(inventory) == 219 and inventory.image_id.is_unique
assert inventory.source_kind.value_counts().to_dict() == {"mcd": 218, "published_tiff": 1}
assert len(local_samples) == 32 and len(rois.query("passed")) == 218
assert len(overlaps) == 158 and all(row["sampled_pixels_equal"] for row in overlaps)
proofs = {(Path(row["source"]).name, int(row["acquisition_id"])): row
          for row in native["acquisitions"] if row["passed"]}
assert len(proofs) == 218
sizes = {Path(row["source"]).name: int(row["size"]) for row in native["files"]}

with formal_read_lease(S1, data_root=ROOT):
    names = ordered.query("panel_id == 'panel_03'").sort_values("channel_index").marker_name.tolist()
    panel03 = panel_from_s1(STUDY, "panel_03", names, S1)
    public_panel = panel_from_s1(STUDY, "published_41_channels", names, S1)
panel02 = panel03.copy()
panel02["panel_id"] = "panel_02"
# The earlier MCD panel has an unlabeled technical 176Yb channel at index 37.
panel02.loc[37, ["source_name", "marker_id", "mapping_status", "role",
                 "keep_channel", "keep_reason", "exclusion_reason", "s1_original_marker",
                 "output_channel_index"]] = ["176Yb", "", "excluded", "exclude", False,
                                           "", "Unlabeled technical channel in embedded panel", "", pd.NA]
panel02.loc[38:39, "output_channel_index"] -= 1
panel = pd.concat([panel02, panel03, public_panel], ignore_index=True)
assert panel.groupby("panel_id").keep_channel.sum().to_dict() == {
    "panel_02": 36, "panel_03": 37, "published_41_channels": 37}

# Supplementary Table 2 gives the published patient identifiers. Two rows
# have a blank IM ROI column, while their progression indicators are present.
with formal_read_lease(SUPPLEMENT, data_root=ROOT):
    supplement_text = subprocess.check_output([
        "pdftotext", "-f", "3", "-l", "3", "-layout", str(SUPPLEMENT), "-"
    ], text=True)
clinical: dict[str, dict[str, str]] = {}
for line in supplement_text.splitlines():
    if not re.match(r"^\s*\d+\s+[FM]\s+", line):
        continue
    parts = re.split(r"\s{2,}", line.strip())
    if len(parts) == 12:
        assert not line[61:69].strip()
        parts.insert(7, "")
    assert len(parts) == 13 and re.fullmatch(r"\d+", parts[0])
    assert re.fullmatch(r"(?:CR|PR|SD|PD)(?:\(\d+\))?", parts[3])
    assert parts[9] in {"", "0", "1"} and parts[11] in {"0", "1"}
    clinical[parts[0]] = dict(zip((
        "source_subject_id", "sex", "age_range", "response", "tumor_thickness",
        "tnm", "site", "im_roi_count", "ct_roi_count", "progressed", "pfs_months",
        "deceased", "os_months"), parts))
assert len(clinical) == 26
published_ids = {re.match(r"\d+", path.stem.split("_ROI")[0].split("_")[-1]).group()
                 for path in (RAW / "melanoma_IMC_data/raw_tiff").glob("*.tiff")}
assert published_ids == set(clinical)
assert sum(int(row["im_roi_count"] or 0) + int(row["ct_roi_count"])
           for row in clinical.values()) == 158

sample_rows = []
for row in local_samples.itertuples(index=False):
    subject_number = re.fullmatch(r"\d+F?", row.sample_id.split("_")[-1])
    patient_id = re.match(r"\d+", subject_number.group()).group() if subject_number else None
    detail = clinical.get(patient_id)
    sample = dict(study=STUDY, sample_id=row.sample_id, source_sample_id=row.sample_id,
                  sample_type="tissue", tissue_type="melanoma tumor",
                  primary_diagnosis="melanoma",
                  metadata_notes=("Source MCD file group; patient number matched to published"
                                  " supplementary clinical table. Biological specimen grouping"
                                  " within a patient remains provisional." if detail else
                                  "Source MCD file group; biological specimen and patient identity unresolved."))
    if detail:
        sample.update(patient_key=f"{STUDY}:{patient_id}", source_subject_id=patient_id,
                      sex_reported={"F": "female", "M": "male"}[detail["sex"]],
                      best_overall_response=re.match(r"CR|PR|SD|PD", detail["response"]).group(),
                      index_event="anti-PD-1 treatment start",
                      os_time=detail["os_months"], os_time_unit="months",
                      os_origin="anti-PD-1 treatment start", os_event=detail["deceased"],
                      os_event_definition="Death versus censoring in Supplementary Table 2",
                      pfs_time=detail["pfs_months"], pfs_time_unit="months",
                      pfs_origin="anti-PD-1 treatment start",
                      pfs_event=detail["progressed"] or None,
                      pfs_event_definition="Progression versus censoring in Supplementary Table 2",
                      source_age_range=detail["age_range"],
                      source_response=detail["response"],
                      source_tumor_thickness=detail["tumor_thickness"],
                      source_tnm=detail["tnm"], source_site=detail["site"],
                      source_im_roi_count=detail["im_roi_count"],
                      source_ct_roi_count=detail["ct_roi_count"])
    sample_rows.append(sample)
samples = pd.DataFrame(sample_rows)
for name in columns("sample_metadata"):
    if name not in samples:
        samples[name] = None
assert len(samples) == 32 and samples.patient_key.notna().sum() == 27
PATIENT_LINKED_SAMPLES = int(samples.patient_key.notna().sum())
patient_linked_sample_ids = set(samples.loc[samples.patient_key.notna(), "sample_id"])
PATIENT_LINKED_IMAGES = int(inventory.image_id.str.split("__acq_").str[0]
                            .isin(samples.loc[samples.patient_key.notna(), "sample_id"]).sum())
assert PATIENT_LINKED_IMAGES == 180
fields = pd.DataFrame([
    mapping(META / "images.csv", "image_id;source_kind;image_path;acquisition_id;panel_id",
            "image_metadata", "image_id;source_files;source_acquisition_id;panel_id",
            "Use the accepted 218 native MCD acquisitions and one complete published TIFF only.",
            "219 independent candidates; 158 published TIFFs overlap native acquisitions."
            " The published TIFF for one incomplete MCD acquisition is the sole TIFF-only input.",
            "images.image_id = accepted ROI image_id; source basename and acquisition_id identify MCD"),
    mapping(META / "rois.csv", "image_id;sample_id;description;slide_id;pixel_size_x_um;pixel_size_y_um",
            "image_metadata", "sample_id;roi_id;slide_id;pixel_size_x_um;pixel_size_y_um",
            "Join complete MCD acquisition to its source file group and retain the source ROI description.",
            "Local intake has no confirmed biological specimen or patient crosswalk."
            " Source file grouping is retained as a provisional sample identity.",
            "rois.image_id = images.image_id; rois.sample_id = MCD filename stem"),
    mapping(META / "samples.csv", "sample_id;mapping_status;patient_key",
            "sample_metadata", "sample_id;patient_key",
            "Create one source file group row per MCD file; match the numeric filename suffix"
            " to a published patient number only when present in Supplementary Table 2.",
            "32 source file groups; 27 groups map to 26 published patients; 5 remain unmatched."
            " Biological specimen grouping remains provisional.",
            "samples.sample_id = MCD filename stem = image_id prefix"),
    mapping(SUPPLEMENT, "Supplementary Table 2: ID;Sex;Age;Response;Tumor thickness;TNM;Site",
            "sample_metadata", "patient_key;source_subject_id;sex_reported;best_overall_response",
            "Join the published numeric patient ID to the MCD filename suffix; retain"
            " parenthetical response annotation separately while mapping CR/PR/SD/PD.",
            "All 26 supplement IDs exactly match the 26 numeric suffixes in 158 published TIFFs."
            " The table lists 158 ROI counts, including two blank IM ROI cells;"
            " blanks remain missing. Counts are not used to admit images.",
            "numeric suffix of source MCD filename = Supplementary Table 2 ID"),
    mapping(SUPPLEMENT, "Supplementary Table 2: PFS;isProgressed;OS;isDeceased",
            "sample_metadata", "pfs_time;pfs_event;os_time;os_event",
            "Use published months and explicit event indicators; the two blank cells are IM ROI counts.",
            "Paper Methods defines survival time from treatment to progression/death or last contact."
            " Clinical follow-up was updated October 2021.",
            "Supplementary Table 2 ID = source_subject_id"),
    mapping(ROI_COUNT_AUDIT, "supplement_total_roi_count;published_tiff_count",
            "sample_metadata", "source_im_roi_count;source_ct_roi_count",
            "Preserve published patient-level source counts; do not use counts as individual image IDs.",
            "All 26 patient ROI totals match the released TIFF counts;"
            " two blank IM ROI cells remain missing in sample metadata. Aggregate 158.",
            "Supplementary Table 2 ID = published TIFF filename numeric suffix"),
    mapping(META / "source_overlap.json", "public_source;local_source;acquisition_id;status",
            "image_metadata", "source_files;source_image_id;split_evidence",
            "Retain one native source for each same acquisition; do not treat the cropped duplicate as independent.",
            "157 same-acquisition and one cropped overlap; all sampled comparisons matched."
            " Full pixel equality remains for independent review."),
    mapping(META / "native_verification.json", "acquisitions;files;panels",
            "image_metadata", "source_dtype;source_channel_count;source_x1_px;source_y1_px",
            "Require recorded complete acquisition and unchanged source file size.",
            "Native evidence checks metadata, payload length and first/middle/last records, not every pixel."),
    mapping(TIFF_DOMAIN_AUDIT, "records_inside_published_frame;unique_coordinates_inside_published_frame",
            "image_metadata", "valid_pixels_path;source_x1_px;source_y1_px",
            "Published TIFF-only frame is 2000 by 1448; all 2,896,000 positions were"
            " acquired in the corresponding native MCD, so no validity companion is required.",
            "Full coordinate-domain audit, separate from sampled native channel verification.",
            "published TIFF ROI20 = native acquisition 20 in the corresponding MCD"),
    mapping(META / "panels.csv", "panel_id;channel_index;marker_name;role",
            "panel_mapping", "source_name;mapping_status;keep_channel;output_channel_index",
            "Preserve actual MCD order; exclude unlabeled technical 176Yb on panel_02.",
            "panel_02 retains 36 channels; panel_03 and published TIFF retain 37."),
    mapping(S1, "Study;Original marker;Unified marker;Segmentation channel",
            "panel_mapping", "marker_id;role;nuclear;body",
            "Join source marker names to current study S1 and retain selected nuclei and admitted proteins.",
            "DNA1 and DNA2 are segmentation-only; source intensity and marker identity retained."),
    mapping(RAW / "Melanoma_panel.csv", "FullStack;marker;Metal",
            "panel_mapping", "source_channel_index;source_name",
            "Published TIFF 41 pages follow the source FullStack order.",
            "Zenodo record 6838169, DOI 10.5281/zenodo.6838169."),
    mapping(META / "summary.json", "mapped_samples;patient_count_status",
            "sample_metadata", "patient_key;metadata_notes",
            "Keep five source file groups without a published patient number unresolved.",
            "Public paper DOI 10.1038/s43856-022-00197-2 describes 26 patients and 158"
            " analyzed ROIs; 27 source file groups and 180 candidate images match published IDs."),
    mapping(OUTPUT / "source_exclusions.json",
            "image_id;source_file;reason;category", "image_metadata", "input_image_path",
            "Do not export MCD acquisitions rejected by full coordinate validation or images"
            " without retained downstream marker signal.",
            "These exclusions are discovered after the intake's sampled native checks."),
], columns=columns("field_mapping"))
for target, definition in {
    "source_age_range": "Published patient age bracket; exact age is not supplied.",
    "source_response": "Verbatim published treatment response with parenthetical annotation.",
    "source_tumor_thickness": "Verbatim published tumor thickness category.",
    "source_tnm": "Verbatim published TNM stage annotation.",
    "source_site": "Verbatim published melanoma site category.",
    "source_im_roi_count": "Published patient-level invasive-margin ROI count before final image reconciliation.",
    "source_ct_roi_count": "Published patient-level core-tumor ROI count before final image reconciliation.",
}.items():
    fields.loc[len(fields)] = mapping(
        SUPPLEMENT, target, "sample_metadata", target,
        "Preserve Supplementary Table 2 source text without inferring an exact age or image identity.",
        json.dumps({"data_type": "string", "definition": definition}),
        "Supplementary Table 2 ID = source_subject_id")

existing: dict[str, dict] = {}
if (OUTPUT / "images.parquet").is_file():
    with formal_read_lease(OUTPUT / "images.parquet", data_root=ROOT):
        prior = pd.read_parquet(OUTPUT / "images.parquet")
    existing = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
    assert set(existing) <= set(inventory.image_id)
    for row in existing.values():
        row["metadata_notes"] = (
            "Source file group is provisional; biological specimen grouping unresolved."
            if row["sample_id"] in patient_linked_sample_ids else
            "Source file group is provisional; biological specimen and patient link unresolved.")
defect_path = OUTPUT / "source_exclusions.json"
with formal_read_lease(defect_path, data_root=ROOT):
    excluded: dict[str, dict] = json.loads(defect_path.read_text()) if defect_path.is_file() else {}
assert set(excluded) <= set(inventory.image_id)


def save_exclusion(image_id: str, source: Path, reason: str, category: str) -> None:
    excluded[image_id] = dict(source_file=relative(source), reason=reason, category=category,
                              prior_native_verification="sampled_native_records_only")
    defect_path.parent.mkdir(parents=True, exist_ok=True)
    with formal_publish_lease(defect_path, directory=defect_path.parent, data_root=ROOT):
        temporary = defect_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(defect_path)


SOURCE_DEFECT_PREFIXES = (
    "MCD has invalid acquisition coordinates",
    "MCD acquisition coordinates exceed declared image",
    "Duplicate MCD pixel coordinates",
    "Nonfinite source pixels",
)


pending_images = 0


def save_progress(stage: str, next_image: str | None, **details: object) -> None:
    global pending_images
    save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields, data_root=ROOT)
    checkpoint(stage, len(existing), next_image, sample_rows=len(samples),
               sample_linked_images=len(existing), patient_linked_samples=PATIENT_LINKED_SAMPLES,
               patient_linked_images=sum(row["sample_id"] in patient_linked_sample_ids
                                         for row in existing.values()), **details)
    pending_images = 0


def next_pending() -> str | None:
    return next((name for name in inventory.image_id
                 if name not in existing and name not in excluded), None)


checkpoint("source_verified", len(existing), next((name for name in inventory.image_id
                                                   if name not in existing and name not in excluded), None),
           sample_rows=len(samples), patient_linked_samples=PATIENT_LINKED_SAMPLES)
for record in inventory.itertuples(index=False):
    image_id = str(record.image_id)
    source = ROOT / record.image_path
    destination = OUTPUT / "images" / image_id / "input.tiff"
    source_group = image_id.split("__acq_")[0]
    assert source_group in set(local_samples.sample_id)
    if image_id in excluded or (image_id in existing and destination.is_file()):
        continue
    if record.source_kind == "mcd":
        proof = proofs[(source.name, int(record.acquisition_id))]
        assert proof["width"] == record.width and proof["height"] == record.height
        assert source.stat().st_size == sizes[source.name]
    else:
        assert source.is_file() and source.stat().st_size > 0
    selected = panel[panel.panel_id == record.panel_id]
    output_names = selected[selected.keep_channel].sort_values("output_channel_index").marker_id.tolist()
    next_bytes = len(output_names) * max(record.width, record.height) ** 2 * 4
    if shutil.disk_usage(ROOT).free < 100 * 1024**3 + next_bytes:
        save_progress("waiting_io", image_id, reason="NAS 100 GiB reserve")
        raise SystemExit(75)
    if shutil.disk_usage("/tmp").free < 64 * 1024**3 + 2 * next_bytes:
        save_progress("waiting_io", image_id, reason="local 64 GiB reserve")
        raise SystemExit(75)
    try:
        reader_path = staged_mcd(source, image_id) if record.source_kind == "mcd" else source
    except OSError as error:
        save_progress("waiting_io", image_id, reason=str(error),
                      local_checkpoint=str(STAGE_ROOT / source.name))
        raise SystemExit(75) from error
    spec = source_spec(record, image_id, reader_path)
    reader_context = open_source(spec)
    try:
        with formal_read_lease(reader_path, data_root=ROOT):
            reader = reader_context.__enter__()
        try:
            assert reader.shape == (41, record.height, record.width)
            assert reader.dtype.name == "float32"
            result = export_input(reader, selected, destination, tile_size=256, data_root=ROOT)
        finally:
            reader_context.__exit__(None, None, None)
    except OSError as error:
        save_progress("waiting_io", image_id, reason=str(error),
                      local_checkpoint=os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch"))
        raise SystemExit(75) from error
    except ValueError as error:
        if spec.kind != "mcd" or not str(error).startswith(SOURCE_DEFECT_PREFIXES):
            raise
        save_exclusion(image_id, source, str(error), "invalid_native_coordinates")
        save_progress("excluded_source_defect", next_pending(), reason=str(error))
        continue
    if not result["admitted"]:
        save_exclusion(image_id, source, "No retained downstream marker signal",
                       "zero_marker_signal")
        save_progress("excluded_no_marker_signal", next_pending())
        continue
    checked_header(destination, (result["height_px"], result["width_px"]), output_names)
    if result["valid_pixels"]:
        with formal_read_lease(destination.with_name("valid_pixels.tiff"), data_root=ROOT):
            with tifffile.TiffFile(destination.with_name("valid_pixels.tiff")) as tif:
                assert len(tif.pages) == 1 and tif.pages[0].shape == (
                    result["height_px"], result["width_px"]) and tif.pages[0].dtype.name == "uint8"
    roi = rois.loc[rois.image_id == image_id]
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=image_id, sample_id=source_group,
               panel_id=str(record.panel_id), image_kind="roi",
               source_image_id=(f"{source.name}:acquisition:{record.acquisition_id}"
                                if record.source_kind == "mcd" else source.name),
               source_files=[relative(source)], source_format=spec.kind,
               source_acquisition_id=(str(record.acquisition_id) if spec.kind == "mcd" else None),
               source_axes="CYX", source_dtype="float32",
               source_revision="local_mcd_v1" if spec.kind == "mcd" else "zenodo_6838169",
               slide_id=str(roi.iloc[0].slide_id) if len(roi) else None,
               roi_id=str(roi.iloc[0].description) if len(roi) else source.stem,
               source_x0_px=0, source_y0_px=0,
               source_x1_px=int(record.width), source_y1_px=int(record.height),
               pixel_size_x_um=1.0, pixel_size_y_um=1.0,
               input_image_path=relative(destination),
               valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                  if result["valid_pixels"] else None),
               intensity_description="Native unnormalized float32 IMC intensity",
               split_method="source_acquisition" if spec.kind == "mcd" else "already_independent",
               split_evidence=relative(META / "source_overlap.json"),
               metadata_notes=("Source file group is provisional; biological specimen grouping unresolved."
                               if source_group in patient_linked_sample_ids
                               else "Source file group is provisional; biological specimen and patient link unresolved."),
               **{key: result[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px",
                                                 "pad_bottom_px", "height_px", "width_px",
                                                 "source_channel_count", "channel_count")})
    existing[image_id] = row
    pending_images += 1
    if pending_images == 4:
        save_progress("image_saved", next_pending())
    print(f"saved {len(existing)}/219", flush=True)

save_progress("images_closed", None)
reviews = pd.DataFrame([
    dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study",
         record_id=STUDY, check_id="S01", result="unresolved",
         source_evidence=relative(META / "native_verification.json"),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details=f"Preparer exported {len(existing)} of 219 candidates; 218 previously accepted native acquisitions"
                 " and one published TIFF from an incomplete native acquisition."
                 f" {len(excluded)} additional source or zero-signal exclusions found on full read."
                 f" Exclusion evidence: {relative(defect_path)}. Original 158 published TIFFs"
                 " deduplicated against MCD; independent review pending."),
    dict(study=STUDY, preparer=JOB.name, reviewer="", scope="sample",
         record_id=STUDY, check_id="M01", result="unresolved",
         source_evidence=relative(SUPPLEMENT),
         output_evidence=relative(OUTPUT / "sample_metadata.csv"),
         details=f"{len(samples)} provisional source file groups; {len(existing)} images linked to"
                 f" groups, {PATIENT_LINKED_SAMPLES} source groups linked to 26 published patient"
                 " IDs. Five source groups and biological specimen crosswalk remain unresolved."
                 f" ROI count audit: {relative(ROI_COUNT_AUDIT)}."),
    dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study",
         record_id=STUDY, check_id="D01", result="unresolved",
         source_evidence=relative(META / "source_overlap.json"),
         output_evidence=relative(OUTPUT / "images.parquet"),
         details="Preparer output closed; independent reviewer and controller acceptance pending."),
], columns=columns("review_results"))
review_path = OUTPUT / "review_results.csv"
with formal_read_lease(review_path, data_root=ROOT):
    previous = pd.read_csv(review_path) if review_path.is_file() else None
if previous is None or (previous.reviewer.fillna("").eq("").all()
                            and previous.preparer.fillna("").isin({"0634_prepare_0", JOB.name}).all()):
    with formal_publish_lease(review_path, directory=OUTPUT, data_root=ROOT):
        partial = review_path.with_suffix(".csv.partial")
        reviews.to_csv(partial, index=False, encoding="utf-8-sig")
        partial.replace(review_path)
checkpoint("repair_closed_pending_independent_review", len(existing), None,
           sample_rows=len(samples), sample_linked_images=len(existing),
           patient_linked_samples=PATIENT_LINKED_SAMPLES,
           patient_linked_images=sum(row["sample_id"] in patient_linked_sample_ids
                                     for row in existing.values()))
