"""Deliver native TiRank IMC acquisitions and source-linked study input tables.

Inputs: the 24 complete local MCDs, existing native verification, private source
specimen/clinical/ROI tables, and this study's S1 rows. Public MCDs are aliases.
Outputs: independent CYX input.tiff files and image/sample/panel/mapping tables.
Clinical identities are read at runtime from NAS; no patient list is embedded.
Run exports through the batch shared formal write lease wrapper. No normalization is run.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "2"

import numpy as np
import pandas as pd
from readimc import MCDFile
import tifffile

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Lin2026_IMC_GastricCancer"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PRIVATE = metadata_source_directory(NAS, STUDY, scope='private') / 'local_intake_v1'
PUBLIC = metadata_source_directory(NAS, STUDY)
RAW = NAS / "raw/private" / STUDY / "source_tables_v1"
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
PROGRESS = Path(os.environ.get(
    "CELLWEFT_STUDY_INPUT_PROGRESS",
    str(PROJECT / "cache/study_preparation" / STUDY / "jobs/0002_prepare_0/progress.json"),
))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "Lin2026_IMC_GastricCancer__native41"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["prepare", "export", "verify", "overview", "stage-region-conflict", "public-metadata"], default="export")
parser.add_argument("--limit", type=int, help="Maximum new acquisitions in this invocation")
parser.add_argument("--repair-evidence-dir", type=Path, help="Private staging directory for the bounded metadata repair")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def progress(stage: str, completed: int, **details: object) -> None:
    details.setdefault("gaps", ["Two conflicting source ROI annotations retained as null",
                               "Clinical codebook/endpoint time reference unconfirmed",
                               "Formal hospital identity and split pending", "Independent review pending"])
    save_json(PROGRESS, dict(study=STUDY, stage=stage, completed_images=completed,
                           expected_images=187, updated_at=datetime.now(timezone.utc).isoformat(), **details))


def mapping(source_file: Path, source_field: str, target_table: str, target_field: str,
            rule: str, *, keys: str = "", notes: str = "", conflict: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_file.stem,
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=keys, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule=conflict, notes=notes)


def region_conflict_rules() -> list[dict]:
    """Describe unresolved annotation joins without selecting a biological label."""
    definitions = {
        "region_annotation_status": "Optional annotation provenance status; source_conflict means the original ROI identifiers do not determine a unique biological region annotation. Null for unaffected images.",
        "region_annotation_evidence_path": "Data-root-relative private evidence retaining the original conflicting source rows and excluded classification uses. Null for unaffected images.",
    }
    return [mapping(RAW / "roi.csv", "sample_id;roi_id;roi;tissue", "image_metadata", name,
        "Set only for the two acquisitions implicated by the disagreeing source ROI identifiers; preserve both null region_annotation values and every other image value.",
        keys="images.sample_id=roi.sample_id;source_acquisition_id compared separately to roi_id and the roi suffix",
        conflict="Retain all original candidate rows in the quality record; do not assign either candidate tissue label or use the affected labels for classification.",
        notes=json.dumps(dict(data_type="string", definition=definition)))
        for name, definition in definitions.items()]


if args.mode == "stage-region-conflict":
    from cellweft.data.io.publication import resource_lock
    from cellweft.data.preparation.inputs import assert_reviewed_binding
    from cellweft.data.preparation.study_input import validate_image_metadata

    if args.repair_evidence_dir is None:
        parser.error("stage-region-conflict requires --repair-evidence-dir")
    stage = args.repair_evidence_dir
    stage.mkdir(parents=True, exist_ok=False, mode=0o700)

    def version(path: Path) -> dict:
        stat = path.stat()
        return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)

    with resource_lock(OUTPUT, shared=True):
        original = pd.read_parquet(OUTPUT / "images.parquet")
        rules = table(OUTPUT / "field_mapping.csv")
        reviews = table(OUTPUT / "review_results.csv")
        accepted = reviews[(reviews.check_id == "D01") & (reviews.result == "pass")].iloc[-1]
        binding = json.loads(accepted.details)["input_binding"]
        assert len(binding["image_ids"]) == 185
        assert_reviewed_binding(OUTPUT, data_root=NAS, binding=binding)
        intake = table(PRIVATE / "rois.csv")
        raw = table(RAW / "roi.csv")
        affected = intake[intake.roi_annotation_status == "ambiguous_source_identifiers"]
        ids = sorted(affected.image_id)
        assert len(original) == 187 and len(ids) == 2
        assert set(original.image_id) - set(binding["image_ids"]) == set(ids)
        assert original.loc[original.image_id.isin(ids), "region_annotation"].isna().all()
        assert "region_annotation_status" not in original
        assert len(rules) == 38 and not (CACHE / "region_annotation_quality.json").exists()
        native_files = table(PRIVATE / "files.csv")
        paths = [OUTPUT / name for name in ("images.parquet", "panel_mapping.csv", "sample_metadata.csv", "field_mapping.csv", "review_results.csv")]
        paths += list(PRIVATE.glob("*.csv")) + [PRIVATE / "native_verification.json", RAW / "roi.csv", RAW / "clinical.csv"]
        paths += [NAS / p for p in native_files.logical_path]
        paths += [NAS / p for p in original.input_image_path]
        before = {str(path): version(path) for path in paths}
        for name in ("images.parquet", "panel_mapping.csv", "sample_metadata.csv", "field_mapping.csv", "review_results.csv"):
            shutil.copyfile(OUTPUT / name, stage / ("before_" + name))
        staged = original.copy()
        evidence_path = relative(CACHE / "region_annotation_quality.json")
        for name in ("region_annotation_status", "region_annotation_evidence_path"):
            staged[name] = pd.Series(pd.NA, index=staged.index, dtype="string")
        selected = staged.image_id.isin(ids)
        staged.loc[selected, "region_annotation_status"] = "source_conflict"
        staged.loc[selected, "region_annotation_evidence_path"] = evidence_path
        quality_rows = []
        first = raw.roi_id.str.extract(r"(\d+)$", expand=False).astype(int)
        second = raw.roi.str.extract(r"(\d+)$", expand=False).astype(int)
        image_index = original.set_index("image_id", verify_integrity=True)
        for image in ids:
            row = image_index.loc[image]
            candidates = raw[raw.sample_id.eq(row.sample_id) &
                (first.eq(int(row.source_acquisition_id)) | second.eq(int(row.source_acquisition_id)))]
            assert len(candidates) in (1, 2)
            quality_rows.append(dict(image_id=image, source_acquisition_id=int(row.source_acquisition_id),
                candidate_source_row_indices=candidates.index.tolist(), candidate_source_rows=candidates.to_dict("records"),
                region_annotation=None, annotation_assignment="unresolved; source keys disagree"))
        quality = dict(study=STUDY, status="unresolved_source_conflict", round=1,
            source_versions=[before[str(RAW / "roi.csv")], before[str(PRIVATE / "rois.csv")]],
            affected_image_ids=ids, affected_count=2, original_candidate_rows=quality_rows,
            physical_input_basis="Existing independent review confirms native acquisition identity, full frame, pixels and sample/patient linkage; the optional tissue-label join remains ambiguous.",
            use_restrictions=["Exclude the two unresolved region labels from biological classification, region-stratified analyses, class-conditioned sampling and class-specific conclusions.",
                "Case-study only; pretraining remains disabled and clinical/hospital split confirmation remains pending."],
            independent_review="Staged round1 metadata; physical-scope admission requires independent post-repair review and an exact new D01 binding.")
        staged_rules = pd.concat([rules, pd.DataFrame(region_conflict_rules())], ignore_index=True)
        staged.to_parquet(stage / "images.parquet", index=False)
        staged_rules.to_csv(stage / "field_mapping.csv", index=False, encoding="utf-8-sig")
        save_json(stage / "region_annotation_quality.json", quality)
        for name in ("sample_metadata.csv", "panel_mapping.csv"):
            shutil.copyfile(OUTPUT / name, stage / name)
        pd.testing.assert_frame_equal(staged[original.columns], original, check_exact=True)
        assert validate_image_metadata(staged, table(stage / "sample_metadata.csv"),
            table(stage / "panel_mapping.csv"), table(stage / "field_mapping.csv")).shape[0] == 187
        assert all(version(Path(path)) == value for path, value in before.items())
        save_json(stage / "before_versions.private.json", dict(guards=before, binding=binding,
            affected_image_ids=ids, proposed_round=1, formal_writes=False))
    print(json.dumps(dict(state="round1_staged_pending_independent_review", affected_images=2,
        image_rows=187, old_values_unchanged=True, new_extension_columns=2, formal_writes=False)))
    raise SystemExit(0)


def source_spec(row: dict) -> SourceSpec:
    return SourceSpec(image_id=row["image_id"], panel_id=PANEL_ID, kind="mcd",
                      path=NAS / row["source_files"][0], channels=tuple(range(41)),
                      acquisition_id=int(row["source_acquisition_id"]),
                      pixel_size_um=(float(row["pixel_size_x_um"]), float(row["pixel_size_y_um"])),
                      source_revision=row["source_revision"])


def read_output_window(tif: tifffile.TiffFile, channel: int, window: Window) -> np.ndarray:
    """Materialize OME TiffFrame IFD metadata before the shared tiled reader.

    Temporary study adapter for read_tiff_window requiring TiffPage attributes;
    neither the TIFF nor the shared module is changed.
    """
    page = tif.pages[channel]
    if isinstance(page, tifffile.TiffFrame):
        page = page.aspage()
    view = SimpleNamespace(pages=[page], filehandle=tif.filehandle)
    return read_tiff_window(view, 0, window)


# Resolve canonical source identity, native order and specimen/patient linkage.
source_samples = table(PRIVATE / "samples.csv")
rois = table(PRIVATE / "rois.csv")
files = table(PRIVATE / "files.csv")
clinical = table(PRIVATE / "clinical.csv").set_index("PathNum", drop=False, verify_integrity=True)
patient_map = table(PRIVATE / "patient_id_mapping.csv").set_index("patient_key", verify_integrity=True)
original_clinical = table(RAW / "clinical.csv").set_index("PathNum", drop=False, verify_integrity=True)
PUBLIC_CLINICAL_FILE = PUBLIC / "13073_2026_1604_MOESM2_ESM.xlsx"
public_clinical = pd.read_excel(PUBLIC_CLINICAL_FILE, sheet_name="Table S4", header=1, dtype=str).fillna("")
public_clinical = public_clinical.set_index("PathNum", drop=False, verify_integrity=True)
assert public_clinical.PatientID.nunique() == len(public_clinical) == 10
for specimen, clin in clinical.iterrows():
    original = original_clinical.loc[specimen]
    assert all(original[key] == clin[key] for key in original.index if key != "PatientID")
    assert original.PatientID == patient_map.loc[clin.patient_key, "source_patient_id"]
native = json.loads((PRIVATE / "native_verification.json").read_text())
overlap = json.loads((PUBLIC / "local_intake_v1/source_overlap.json").read_text())
source_panel = table(PRIVATE / "panels.csv").sort_values("channel_index", key=lambda s: s.astype(int))
source_names = source_panel.source_label.tolist()
panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1)
assert len(panel) == 41 and int(panel.keep_channel.sum()) == 37
assert int(panel.nuclear.sum()) == 2 and not panel[panel.role == "downstream"].marker_id.duplicated().any()
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert source_samples.source_specimen_label.nunique() == 24
assert source_samples.patient_key.nunique() == 12
assert (source_samples.patient_key != "").all()
assert len(rois[rois.passed == "True"]) == 187

image_columns = list(pd.read_csv(TEMPLATES / "image_metadata.csv", nrows=0))
sample_columns = list(pd.read_csv(TEMPLATES / "sample_metadata.csv", nrows=0))
field_rows = []
clinical_fields = {
    "Surgery": "source_surgery_status", "Age": "source_age_years", "BMI": "source_bmi_kg_m2",
    "DiffGrade": "source_differentiation_grade_code", "LaurenType": "source_lauren_type_code",
    "TRG": "source_trg_code", "Sex": "source_sex_code", "TumorSize": "source_tumor_size",
    "cT": "source_ct_code", "cN": "source_cn_code", "VI": "source_vi_code", "NI": "source_ni_code",
    "OS": "source_os_code", "OSTime": "source_os_time", "RFS": "source_rfs_code",
    "RFSTime": "source_rfs_time", "PathNum": "source_pathology_id", "RegionInfo": "source_region_info",
}
for specimen, published in public_clinical.iterrows():
    assert specimen in clinical.index
    local = clinical.loc[specimen]
    assert all(str(local[source]) == published[source] for source in clinical_fields)
for source_field, target_field in clinical_fields.items():
    field_rows.append(mapping(PRIVATE / "clinical.csv", source_field, "sample_metadata", target_field,
                              "Preserve source text exactly after exact pathology-ID join; no clinical code or time-origin inference.",
                              keys="samples.source_specimen_label = clinical.PathNum",
                              notes=json.dumps(dict(data_type="string", definition=f"Native clinical {source_field}; source coding/time reference retained without reinterpretation"))))
sample_rows = []
for item in source_samples.to_dict("records"):
    specimen = item["source_specimen_label"]
    clin = clinical.loc[specimen]
    published = public_clinical.loc[specimen] if specimen in public_clinical.index else None
    assert clin.patient_key == item["patient_key"]
    row = dict.fromkeys(sample_columns)
    row.update(study=STUDY, sample_id=specimen, patient_key=item["patient_key"],
               source_subject_id=(published.PatientID if published is not None else
                                  patient_map.loc[item["patient_key"], "source_patient_id"]),
               source_hospital_id=(published.HospitalID if published is not None else ""),
               cohort_id=STUDY, source_sample_id=specimen, biospecimen_group_id=specimen,
               matched_sample_id=source_samples.loc[source_samples.patient_key.eq(item["patient_key"]) &
                                                    source_samples.source_specimen_label.ne(specimen), "source_specimen_label"].item(),
               anatomical_site="Stomach", primary_site="Stomach", primary_diagnosis="Gastric cancer",
               case_control_status="case", timepoint_label="other",
               metadata_notes=(
                   "Public Table S4 row joined by exact PathNum; source_subject_id is the published PatientID. "
                   "source_hospital_id is preserved as named without treating it as a center code. "
                   "Project patient_key remains the local private crosswalk. "
                   if published is not None else
                   "No public Table S4 row; source clinical values and source_subject_id come from local private intake, outside the public release scope. "
               ) + "Surgery Before/After is retained without equating it to treatment timing; age reference, categorical codebook, endpoint units/origin and formal hospital identity remain unconfirmed. All ROIs of this specimen share sample_id. Case study; pretraining disabled; split pending clinical/hospital confirmation.")
    source_clinical = published if published is not None else clin
    row.update({target: source_clinical[source] for source, target in clinical_fields.items()})
    row.update(data_use_role="case_study", pretraining_allowed=False,
               split_group_id=item["split_group_id"], patient_split_status="pending_clinical_hospital_confirmation")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)
assert len(samples) == 24 and samples.patient_key.nunique() == 12

field_rows.extend([
    mapping(PRIVATE / "samples.csv", "source_specimen_label;patient_key;split_group_id", "sample_metadata",
            "study;sample_id;source_sample_id;biospecimen_group_id;patient_key;cohort_id;matched_sample_id;split_group_id",
            "Use source clinical pathology specimen as biological sample; all same-specimen ROIs share sample_id; paired specimens join the same source patient.",
            keys="rois.sample_id = samples.sample_id; samples.source_specimen_label = clinical.PathNum",
            notes="24 source-confirmed specimens, 12 patients; 187/187 accepted images linked to sample and patient. Physical slide count is not inferred from MCD count."),
    mapping(PRIVATE / "patient_id_mapping.csv", "source_patient_id;patient_key", "sample_metadata", "source_subject_id",
            "Use the private source ID only for rows without a public Table S4 match; matched public rows use Table S4.PatientID below.", keys="patient_key"),
    mapping(RAW / "clinical.csv", "PathNum;PatientID;Surgery;clinical fields", "sample_metadata", "source_subject_id;patient_key;source_sample_id",
            "Check all 24 original clinical records against local_intake_v1 clinical values and exact patient crosswalk.", keys="PathNum; PatientID = patient_id_mapping.source_patient_id"),
    mapping(PRIVATE / "clinical.csv", "Surgery", "sample_metadata", "timepoint_label",
            "Use other; preserve Before/After surgery text in source_surgery_status without equating surgery timing to treatment timing."),
    mapping(PUBLIC / "zenodo_record.json", "metadata.description", "sample_metadata",
            "anatomical_site;primary_site;primary_diagnosis;case_control_status;metadata_notes",
            "Apply gastric cancer and stomach study context to source-confirmed specimens; optional unsupported clinical fields remain null."),
    mapping(PRIVATE / "rois.csv", "sample_id;image_id;acquisition_id;mcd_path;slide_id;description;panel_id;width;height;pixel_size_x_um;pixel_size_y_um", "image_metadata",
            "study;image_id;sample_id;panel_id;image_kind;source_image_id;source_files;source_format;source_acquisition_id;source_z_index;source_t_index;source_axes;source_dtype;source_revision;slide_id;core_id;roi_id;fov_id;source_x0_px;source_y0_px;source_x1_px;source_y1_px;source_channel_count;pixel_size_x_um;pixel_size_y_um;split_method;split_evidence;metadata_notes",
            "One complete native acquisition per image, whole [0,width)x[0,height) box. No Z/time layer or core split. Image identity preserves canonical source index.",
            keys="(source MCD, acquisition_id); rois.sample_id = samples.sample_id"),
    mapping(PRIVATE / "rois.csv", "tissue_annotation;roi_annotation_status;source_roi_id", "image_metadata", "region_annotation",
            "Copy original tissue_annotation only when source ROI keys agree; keep two ambiguous annotations null.",
            keys="source specimen and acquisition description/ROI ID",
            conflict="roi_mapping_issues.csv identifies one source-row conflict affecting two acquisitions; acquisition pixels and specimen/patient link remain valid."),
    mapping(S1, "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
            "source_channel_index;source_name;marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index;keep_reason;exclusion_reason",
            "Exact study and embedded source-label join; retain 35 distinct downstream measurements and both Ir DNA inputs, exclude 4 technical channels.",
            keys="Study + Original marker", notes="The unused 131Xe S1 row belongs to the calibration panel, not tissue native41."),
    mapping(PRIVATE / "native_verification.json", "acquisitions;files", "image_metadata",
            "pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;channel_count;input_image_path;valid_pixels_path;intensity_description",
            "Shared export_input: centered square zero padding, float32 native records, lossless tiled OME BigTIFF. Retained downstream nonzero test covers entire image; measured validity comes from source coordinates."),
    mapping(PUBLIC / "local_intake_v1/source_overlap.json", "public_source;local_source;basis", "image_metadata", "source_files",
            "Use one canonical local MCD acquisition; 10 public-release files and public/private local links are aliases, not additional images.",
            notes="Reuses native verification and overlap evidence; validates current size/mtime and acquisition metadata without rehashing."),
    mapping(CACHE / "source_metadata/search_log.json", "source;url;status", "sample_metadata", "metadata_notes",
            "Public publication, release and author repository supplement local specimen/patient evidence; public 10-patient subset does not replace the complete local 12-patient cohort."),
])
public_clinical_fields = {
    "HospitalID": "source_hospital_id", "PatientID": "source_subject_id",
    **clinical_fields,
}
for source_field, target_field in public_clinical_fields.items():
    field_rows.append(mapping(PUBLIC_CLINICAL_FILE, source_field, "sample_metadata", target_field,
        "Preserve the public Table S4 source text on exact PathNum join; applies only to the 10 published IMC sample rows and does not reinterpret clinical codes or time origins.",
        keys="Table S4.PathNum = sample_metadata.sample_id",
        notes=json.dumps(dict(data_type="string", definition=f"Published Table S4 {source_field}; retained as source text"))))
for target_table in ["image_metadata", "sample_metadata"]:
    for field, kind, definition in [
        ("data_use_role", "string", "Source-authorized data-use role; case_study"),
        ("pretraining_allowed", "boolean", "Whether this source may enter pretraining; false for this cohort"),
        ("split_group_id", "string", "Existing source patient grouping shared across linked specimens and cohorts"),
        ("patient_split_status", "string", "Formal hospital identity and patient holdout confirmation status"),
    ]:
        source_file = PRIVATE / ("samples.csv" if field == "split_group_id" else "README.md" if field == "patient_split_status" else "summary.json")
        source_field = "split_group_id" if field == "split_group_id" else "formal hospital holdout status" if field == "patient_split_status" else "role;pretraining_allowed"
        field_rows.append(mapping(source_file, source_field, target_table, field,
                                  "Preserve existing private case-study role and split restriction.",
                                  notes=json.dumps(dict(data_type=kind, definition=definition))))
fields = pd.DataFrame(field_rows, columns=list(pd.read_csv(TEMPLATES / "field_mapping.csv", nrows=0)))
fields.loc[fields.source_file.str.endswith("13073_2026_1604_MOESM2_ESM.xlsx"), "source_sheet_or_table"] = "Table S4"
if (CACHE / "region_annotation_quality.json").exists():
    fields = pd.concat([fields, pd.DataFrame(region_conflict_rules())], ignore_index=True)

images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=image_columns + ["data_use_role", "pretraining_allowed", "split_group_id", "patient_split_status"])
if (OUTPUT / "field_mapping.csv").exists():
    prior_fields = table(OUTPUT / "field_mapping.csv")
    image_extensions = set(images.columns) - set(image_columns)
    prior_image_extensions = prior_fields[
        prior_fields.target_table.eq("image_metadata") &
        prior_fields.target_field.isin(image_extensions)
    ]
    fields = pd.concat([fields, prior_image_extensions], ignore_index=True).drop_duplicates(
        subset=["study", "source_file", "source_sheet_or_table", "source_field",
                "target_table", "target_field", "join_keys"], keep="last")
# Temporary study adapter: pandas empty StringDtype.map(...).all() fails in
# validate_image_metadata's extension check. Keep extension inputs as object;
# the shared validator still converts published columns to their declared type.
for extension in ("data_use_role", "split_group_id", "patient_split_status"):
    images[extension] = images[extension].astype(object)
if args.mode == "public-metadata":
    save_input_tables(OUTPUT, images, samples, panel, fields, replace_metadata_studies={STUDY})
    print(json.dumps(dict(status="public_sample_metadata_published", public_samples=len(public_clinical),
                          total_local_samples=len(samples), public_clinical_fields=len(public_clinical_fields))))
    raise SystemExit(0)

progress("source_metadata", len(images), sample_rows=24, patient_linked_samples=24, patient_count=12)

if args.mode == "prepare":
    # Source proof reuse is conditional on current native file identity and order.
    inventory = []
    for item in files.to_dict("records"):
        path = NAS / item["logical_path"]
        stat = path.stat()
        assert stat.st_size == int(item["size_bytes"]) and stat.st_mtime_ns == int(item["mtime_ns"])
        public_alias = NAS / item["logical_path"].replace("raw/private/", "raw/public/", 1)
        assert public_alias.samefile(path)
        expected = rois[rois.mcd_path == relative(path)]
        with MCDFile(path) as mcd:
            actual = {a.id: a for slide in mcd.slides for a in slide.acquisitions}
            for rec in expected.to_dict("records"):
                a = actual[int(rec["acquisition_id"])]
                assert list(a.channel_labels) == source_names and a.width_px == int(rec["width"]) and a.height_px == int(rec["height"])
                assert int(a.metadata["DataStartOffset"]) == int(rec["data_start"])
                assert int(a.metadata["DataEndOffset"]) == int(rec["data_end"])
        inventory.append(dict(path=relative(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                              tissue_acquisitions=len(expected), admitted_native=(expected.passed == "True").sum().item()))
    archive = NAS / "raw/public" / STUDY / "zenodo_14864800/UnionH3-IMC.zip"
    with zipfile.ZipFile(archive) as z:
        members = [dict(member=i.filename, bytes=i.file_size) for i in z.infolist()]
    assert len(members) == 10 and len(overlap) == 10
    for alias in overlap:
        public_path, local_path = Path(alias["public_source"]), Path(alias["local_source"])
        assert public_path.stat().st_size == local_path.stat().st_size
        assert next(i["bytes"] for i in members if i["member"] == public_path.name) == public_path.stat().st_size
    planned = []
    for rec in rois[rois.passed == "True"].to_dict("records"):
        item = source_samples.set_index("sample_id").loc[rec["sample_id"]]
        stat = (NAS / rec["mcd_path"]).stat()
        w, h = int(rec["width"]), int(rec["height"])
        side = max(w, h)
        row = dict.fromkeys(image_columns)
        row.update(study=STUDY, image_id=rec["image_id"], sample_id=item.source_specimen_label,
                   panel_id=PANEL_ID, image_kind="roi", source_image_id=rec["image_id"],
                   source_files=[rec["mcd_path"]], source_format="mcd", source_acquisition_id=int(rec["acquisition_id"]),
                   source_axes="CYX", source_dtype="float32", source_revision=f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}",
                   slide_id=f"{Path(rec['mcd_path']).stem}__slide_{rec['slide_id']}", roi_id=rec["description"],
                   region_annotation=rec["tissue_annotation"] or None,
                   source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                   pad_left_px=(side-w)//2, pad_top_px=(side-h)//2,
                   pad_right_px=side-w-(side-w)//2, pad_bottom_px=side-h-(side-h)//2,
                   height_px=side, width_px=side, source_channel_count=41, channel_count=37,
                   pixel_size_x_um=float(rec["pixel_size_x_um"]), pixel_size_y_um=float(rec["pixel_size_y_um"]),
                   input_image_path=f"processed/{STUDY}/images/{rec['image_id']}/input.tiff",
                   intensity_description="Native decoded MCD float32 intensities; unnormalized; no denoising",
                   split_method="source_acquisition", split_evidence="MCD acquisition and source pathology specimen; one specimen per native ROI",
                   metadata_notes=f"Source ROI annotation status: {rec['roi_annotation_status']}. Canonical local observation; public aliases deduplicated. Case-study restrictions retained.",
                   data_use_role="case_study", pretraining_allowed=False, split_group_id=rec["split_group_id"],
                   patient_split_status="pending_clinical_hospital_confirmation")
        planned.append(row)
    plan = pd.DataFrame(planned)
    canonical = table(PUBLIC / "local_intake_v1/images.csv")
    assert set(canonical.image_id) == set(plan.image_id)
    plan.to_parquet(CACHE / "image_plan.parquet", index=False)
    save_json(CACHE / "source_inventory.json", dict(files=inventory, archive=relative(archive), archive_members=members,
              public_alias_files=10, local_tissue_acquisitions=188, native_accepted_acquisitions=187,
              calibration_exclusions=65, incomplete_exclusions=1, source_only_duplicates=True,
              native_proof=relative(PRIVATE / "native_verification.json"),
              output_uncompressed_bytes=int(sum(int(r["width_px"])**2*37*4 for r in planned))))
    save_input_tables(OUTPUT, images, samples, panel, fields)
    if not (OUTPUT / "review_results.csv").exists():
        with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
            pd.DataFrame([dict(study=STUDY, preparer="Codex CLI study preparer", reviewer="", scope="study",
                           record_id=STUDY, check_id="D01", result="unresolved", source_evidence=relative(PRIVATE),
                           output_evidence=relative(OUTPUT), details="整理者进度记录：来源和metadata已准备；TIFF导出、自检和独立复核待执行。")]).to_csv(
                               OUTPUT / "review_results.csv", index=False, encoding="utf-8-sig")
    progress("metadata_prepared", len(images), expected_uncompressed_bytes=int((plan.width_px**2*37*4).sum()),
             sample_rows=24, patient_linked_samples=24, patient_count=12,
             gaps=["Two source ROI annotations conflict", "Clinical codebook/endpoint reference and formal hospital split unconfirmed"])

if args.mode == "export":
    plan = pd.read_parquet(CACHE / "image_plan.parquet")
    receipts_path = CACHE / "export_receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    completed = set(images.image_id)
    new_count = 0
    for row in plan.to_dict("records"):
        if row["image_id"] in completed:
            assert (NAS / row["input_image_path"]).is_file()
            continue
        if row["image_id"] in receipts and not receipts[row["image_id"]]["admitted"]:
            continue
        if args.limit is not None and new_count >= args.limit:
            break
        if shutil.disk_usage(NAS).free < 100*1024**3 + row["width_px"]**2*37*4 or shutil.disk_usage("/tmp").free < 64*1024**3:
            progress("capacity_blocked", len(images), gaps=["Required reserved free space unavailable"])
            raise RuntimeError("Insufficient reserved output space")
        progress("exporting", len(images), current_image_id=row["image_id"], sample_rows=24, patient_linked_samples=24)
        try:
            if row["image_id"] in receipts and (NAS / row["input_image_path"]).is_file():
                result = receipts[row["image_id"]]
                with tifffile.TiffFile(NAS / row["input_image_path"]) as tif:
                    assert tif.series[0].shape == (37, result["height_px"], result["width_px"])
                    assert tif.series[0].dtype == np.dtype("float32")
            else:
                with open_source(source_spec(row)) as reader:
                    result = export_input(reader, panel, NAS / row["input_image_path"])
        except Exception as exc:
            progress("export_error", len(images), current_image_id=row["image_id"], gaps=[str(exc)])
            raise
        receipts[row["image_id"]] = result
        save_json(receipts_path, receipts)
        if result["admitted"]:
            row.update({k: result[k] for k in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_dtype", "source_channel_count", "channel_count"]})
            row["valid_pixels_path"] = str(Path(row["input_image_path"]).with_name(result["valid_pixels"])) if result["valid_pixels"] else None
            images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, fields)
        new_count += 1
        progress("exporting", len(images), last_completed_image_id=row["image_id"], sample_rows=24, patient_linked_samples=24)
        print(json.dumps(dict(completed_images=len(images), expected_images=187)), flush=True)
    progress("export_complete" if len(receipts) == len(plan) else "batch_complete", len(images),
             native_eligible_images=len(plan), signal_excluded_images=sum(not r["admitted"] for r in receipts.values()),
             sample_rows=24, patient_linked_samples=24, patient_count=12)

if args.mode == "verify":
    checks = []
    pixel_checks = []
    padding_checks = []
    names = kept.marker_id.tolist()
    representatives = set(images.iloc[[0, -1]].image_id) if len(images) else set()
    representatives.update(images.loc[images.pad_left_px.gt(0), "image_id"].head(1))
    representatives.update(images.loc[images.pad_top_px.gt(0), "image_id"].head(1))
    representatives.update(images.loc[images.width_px.eq(images.width_px.max()), "image_id"].head(1))
    representatives.update(images.loc[images.region_annotation.isna(), "image_id"])
    for row in images.to_dict("records"):
        with tifffile.TiffFile(NAS / row["input_image_path"]) as tif:
            assert len(tif.pages) == 37 and tif.series[0].shape == (37, row["height_px"], row["width_px"])
            assert tif.series[0].axes == "CYX" and row["height_px"] == row["width_px"]
            assert row["source_x1_px"] - row["source_x0_px"] + row["pad_left_px"] + row["pad_right_px"] == row["width_px"]
            assert row["source_y1_px"] - row["source_y0_px"] + row["pad_top_px"] + row["pad_bottom_px"] == row["height_px"]
            assert tif.series[0].dtype == np.dtype("float32")
            root = ET.fromstring(tif.ome_metadata)
            pixels = next(e for e in root.iter() if e.tag.endswith("}Pixels"))
            assert float(pixels.attrib["PhysicalSizeX"]) == row["pixel_size_x_um"]
            assert float(pixels.attrib["PhysicalSizeY"]) == row["pixel_size_y_um"]
            assert pixels.attrib["PhysicalSizeXUnit"] == pixels.attrib["PhysicalSizeYUnit"] == "µm"
            channel_names = [e.attrib["Name"] for e in root.iter() if e.tag.endswith("}Channel")]
            assert channel_names == names
            checks.append(dict(image_id=row["image_id"], shape=list(tif.series[0].shape), dtype="float32", ordered_channel_names=True))
            if row["image_id"] in representatives:
                with MCDFile(NAS / row["source_files"][0]) as mcd:
                    acquisition = next(a for slide in mcd.slides for a in slide.acquisitions if a.id == int(row["source_acquisition_id"]))
                    w, h = int(row["source_x1_px"]), int(row["source_y1_px"])
                    for x, y in [(w//2-16, h//2-16), (max(0, w-32), max(0, h-32))]:
                        window = (x, y, x+32, y+32)
                        src = mcd.read_acquisition(acquisition, channels=kept.source_channel_index.tolist(), region=window)
                        out_window = (x+row["pad_left_px"], y+row["pad_top_px"], x+32+row["pad_left_px"], y+32+row["pad_top_px"])
                        dst = np.stack([read_output_window(tif, i, out_window) for i in range(37)])
                        assert np.array_equal(src, dst)
                        pixel_checks.append(dict(image_id=row["image_id"], source_window=list(window), output_window=list(out_window), channels=37, exact_equal=True))
                    side = int(row["width_px"])
                    padding_strips = [
                        (0, 0, row["pad_left_px"], side),
                        (side-row["pad_right_px"], 0, side, side),
                        (0, 0, side, row["pad_top_px"]),
                        (0, side-row["pad_bottom_px"], side, side),
                    ]
                    for strip in padding_strips:
                        if strip[0] < strip[2] and strip[1] < strip[3]:
                            assert all(not read_output_window(tif, i, strip).any() for i in range(37))
                            padding_checks.append(dict(image_id=row["image_id"], output_window=list(strip), channels=37, all_zero=True))
    save_json(CACHE / "preparer_verification.json", dict(headers=checks, pixel_blocks=pixel_checks, padding_strip_checks=padding_checks, independent_review=False))
    review_rows = []
    for check_id in table(TEMPLATES / "review_checks.csv").check_id:
        result = "unresolved" if check_id in {"I04", "M02", "D01"} else "pass"
        details = {
            "S01": "253 local acquisitions: 65 calibration exclusions, 188 tissue acquisitions, 1 incomplete excluded, 187 native eligible. Ten public MCD/ZIP members are aliases, not extra observations.",
            "I01": f"{len(images)} unique image records point to canonical native MCD acquisitions and actual TIFFs.",
            "I02": f"{len(images)}/{len(images)} images link to 24 source pathology specimens; all 24 specimens link to 12 source-confirmed patients.",
            "I03": "Full native ROI, source resolution/orientation, centered square padding; no multi-sample split.",
            "I04": "185/187 eligible ROI tissue annotations match; 2 source-row identifiers conflict and annotations stay null. Specimen/patient joins unaffected. Representative overview is separately recorded after visual inspection.",
            "P01": "All 41 native tissue channel labels exactly joined to this study's S1; calibration-only 131Xe is not part of native tissue panel.",
            "P02": "35 distinct downstream markers and 2 selected DNA nuclear inputs retained; 4 technical channels excluded.",
            "P03": f"All {len(checks)} TIFF headers checked: float32, 37 CYX channels and ordered OME names match panel.",
            "I05": f"Preparer exact comparisons: {len(pixel_checks)} 32x32 native/output blocks, all 37 channels, square and non-square source geometry; padding examined. See preparer_verification.json.",
            "M01": "Local source specimen list, clinical pathology IDs, patient crosswalk and ROI index reviewed: 24 specimens, 12 patients, complete image/sample/patient linkage. Public source checks and cached metadata recorded; published subset is not full local cohort.",
            "M02": "Original clinical text preserved in defined source_* extensions; unverified categorical coding, age time reference and endpoint units/origin remain explicit. Standardized clinical endpoint fields stay null; hospital identity unconfirmed.",
            "D01": f"{len(images)}/187 native images delivered with panel/sample/field tables. These are preparer self-checks; separate CLI independent review and controller acceptance remain pending.",
        }[check_id]
        check_scope = table(TEMPLATES / "review_checks.csv").set_index("check_id").loc[check_id, "scope"]
        review_rows.append(dict(study=STUDY, preparer="Codex CLI study preparer", reviewer="", scope=check_scope,
                                record_id=PANEL_ID if check_scope == "panel" else STUDY, check_id=check_id, result=result,
                                source_evidence=relative(PRIVATE), output_evidence=relative(OUTPUT),
                                details="整理者自检（非独立复核）。" + details))
    for row in images.loc[images.region_annotation.isna()].to_dict("records"):
        review_rows.append(dict(study=STUDY, preparer="Codex CLI study preparer", reviewer="", scope="image",
                                record_id=row["image_id"], check_id="I04", result="unresolved",
                                source_evidence=relative(PRIVATE / "roi_mapping_issues.csv"), output_evidence=relative(OUTPUT / "images.parquet"),
                                details="来源ROI编号与原始行后缀冲突；region_annotation留空。完整来源采集及sample/patient连接保留；待主控核实来源注释。"))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding="utf-8-sig")
    progress("prepared_pending_independent_review", len(images), sample_linked_images=int(images.sample_id.notna().sum()),
             sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()), patient_count=12,
             headers_checked=len(checks), exact_pixel_blocks=len(pixel_checks),
             gaps=["2 conflicting source ROI annotations retained as null", "Clinical codebook/endpoint time reference missing", "Formal hospital identity and split pending", "Independent review pending"])

if args.mode == "overview":
    # Preview-only display scaling leaves the delivered native TIFF untouched.
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = set(images.iloc[[0, -1]].image_id)
    selected.update(images.loc[images.pad_left_px.gt(0), "image_id"].head(1))
    selected.update(images.loc[images.pad_top_px.gt(0), "image_id"].head(1))
    selected.update(images.loc[images.width_px.eq(images.width_px.max()), "image_id"].head(1))
    selected.update(images.loc[images.region_annotation.isna(), "image_id"])
    subset = images[images.image_id.isin(selected)]
    fig, axes = plt.subplots(len(subset), 2, figsize=(8, 4*len(subset)), squeeze=False)
    for axis_row, row in zip(axes, subset.to_dict("records")):
        with tifffile.TiffFile(NAS / row["input_image_path"] ) as tif:
            planes = []
            for name in ["CD45", "PanCK", "DNA1"]:
                index = kept.loc[kept.marker_id.eq(name), "output_channel_index"].item()
                values = tif.pages[index].asarray(maxworkers=2)[::3, ::3]
                high = np.percentile(values[values > 0], 99.5) if (values > 0).any() else 1
                planes.append(np.clip(values / high, 0, 1))
        axis_row[0].imshow(planes[2], cmap="gray")
        axis_row[1].imshow(np.stack(planes, axis=-1))
        axis_row[0].set_title(row["image_id"] + "\nDNA1")
        axis_row[1].set_title("R=CD45 G=PanCK B=DNA1\n" + str(row["region_annotation"]))
        for axis in axis_row:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(CACHE / "representative_overview.png", dpi=100)
    plt.close(fig)
    save_json(CACHE / "overview_scope.json", dict(image_ids=list(subset.image_id),
              display="Preview-only positive-pixel P99.5 scaling; native TIFF not changed", visual_inspection="pending"))
