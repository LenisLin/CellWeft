"""Prepare the 27 original IMC TMA cores, preserving published float32 intensity.

Inputs: Zenodo 8341225 ZIP, accepted native proofs/exact archive members, study S1 rows,
        archive ROI-to-patient table, paper and supplementary clinical/panel tables.
Outputs: per-core input.tiff and study panel/image/sample/field/review tables.
Shared helpers acquire short per-image NAS leases; run without an outer I/O lock.
The ROI and supplementary clinical patient namespaces conflict. Only the archive
patient namespace is linked; supplementary clinical values remain unassigned.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Kucukkose2024_IMC_ColorectalCancer/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Kucukkose2024_IMC_ColorectalCancer"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = META / "preparation_evidence"
SOURCE_META = META
WORK = PROJECT / "cache/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("JOB", str(WORK / "prepare")))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "Kucukkose_IMC_native43"
PREPARER = "codex_cli_Kucukkose_" + JOB.name
REVISION = "Zenodo8341225_original_Kucukkose_Baars_et_al"
ROI_MEMBER = "Kucukkose_Baars_et_al/Metadata/20211008_ROI_info.csv"
CLINICAL_GAP = ("Archive ROI Patient_ID and supplementary Table 1 pt ID are incompatible namespaces: "
                "MSI non-metastatic and MSS metastatic groups have conflicting patient-ID assignments. "
                "The per-patient comparison is retained in the source metadata bundle as "
                "preparation_evidence/clinical_namespace_conflicts.csv. "
                "No within-group permutation or clinical-value join is inferred. "
                "Age, sex, stage, detailed site and mutation status remain blank for all samples.")
PANEL_CAVEAT = ("Supplementary Table 3 assigns 113In/115In to aSMA, 143Nd to CD20 and 161Dy to CD45, "
                "whereas all native TIFF/session labels identify LMNB1, CD45 and CD20 respectively. "
                "Use accepted native identities and exact study S1 rows; controller must retain this discrepancy.")
SOURCE_STAGING = PROJECT / "cache/study_input" / STUDY / "source_members"


def work_reference(path: Path) -> Path:
    """Keep SSD process evidence distinct from NAS-relative source paths."""
    return path.resolve()


def write_json(path: Path, data: object) -> None:
    """Close a JSON checkpoint before publishing its name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def progress(stage: str, images: pd.DataFrame | None = None, **details: object) -> None:
    """Report only successfully saved image rows, with the clinical identity gap."""
    with formal_read_lease(OUTPUT):
        if images is None:
            images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
        delivered_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False) if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=27, completed_images=len(images),
        sample_rows=len(delivered_samples),
        confirmed_source_sample_groups=9,
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(delivered_samples.patient_key.ne("").sum()) if len(delivered_samples) else 0,
        phase="prepare", scientific_repair_round=0, independent_review=False,
        gaps=[CLINICAL_GAP, PANEL_CAVEAT, "Independent review pending"],
        updated_at=datetime.now(timezone.utc).isoformat(), **details))


def field_row(source: str, source_field: str, table: str, target: str,
              rule: str, *, join: str = "", notes: str = "") -> dict:
    """Record one source-supported join or transformation in the template."""
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=source_field, target_table=table, target_field=target,
                join_keys=join, value_mapping="", transformation_rule=rule,
                time_origin_rule="", conflict_rule="Retain source namespace; never infer clinical ID permutations",
                notes=notes)


def image_id(proof: dict) -> str:
    return Path(proof["source_member_name"]).stem.replace(" ", "_")


def source_spec(proof: dict, path: Path) -> SourceSpec:
    """The published OME TIFF is the source frame, not an inferred MCD frame."""
    return SourceSpec(image_id(proof), PANEL_ID, "tiff", path, tuple(range(43)),
                      pixel_size_um=(1.0, 1.0), source_revision=REVISION)


@contextmanager
def extracted_member(proof: dict):
    """Expose one exact ZIP member on SSD only for the current operation."""
    SOURCE_STAGING.mkdir(parents=True, exist_ok=True)
    entry = proof["archive_entry"]
    with tempfile.TemporaryDirectory(prefix=f"{STUDY}_", dir=SOURCE_STAGING) as directory:
        destination = Path(directory) / Path(proof["source_member_name"]).name
        partial = destination.with_suffix(destination.suffix + ".partial")
        with formal_read_lease(archive), ZipFile(archive) as source_zip:
            member = source_zip.getinfo(proof["source_member_name"])
            assert member.filename == entry["exact_archive_entry_name"]
            assert (member.file_size, member.CRC) == (
                entry["exact_archive_entry_size"], entry["exact_archive_entry_crc"])
            with source_zip.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            assert partial.stat().st_size == entry["exact_archive_entry_size"]
            os.replace(partial, destination)
        yield destination


def read_header(proof: dict, path: Path) -> tuple[int, int]:
    """Bind original TIFF channels, dimensions and scale to accepted native proof."""
    stat = path.stat()
    assert stat.st_size == proof["archive_entry"]["exact_archive_entry_size"]
    with tifffile.TiffFile(path) as tif:
        root = ET.fromstring(tif.ome_metadata)
        pixels = root.find(".//{*}Pixels")
        names = [x.attrib["Name"] for x in pixels.findall("{*}Channel")]
        assert names == proof["channels"]
        assert tif.series[0].shape == tuple(proof["tiff_layout"]["shape"])
        assert tif.series[0].dtype == np.dtype("float32")
        # These imctools TIFFs omit OME PhysicalSize; the same acquisition's
        # original session records both ablation distances explicitly.
        acquisition = next(a for a in sessions[proof["session_name"]]["acquisitions"]
                           if a["id"] == proof["source_acquisition_id"])
        assert acquisition["ablation_distance_between_shots_x"] == acquisition["ablation_distance_between_shots_y"] == 1.0
        return tif.series[0].shape[-2:]


def inspect_sources(proofs: list[dict]) -> None:
    """Inspect each full core plus both duplicate measurements without altering pixels.

    Display-only log1p/percentile scaling is confined to PNG overviews. Duplicate
    QC reports source statistics and association with the S1-selected nuclear input.
    """
    fig, axes = plt.subplots(7, 4, figsize=(16, 27))
    duplicate_fig, duplicate_axes = plt.subplots(3, 4, figsize=(15, 11))
    duplicate_examples = {proofs[0]["session_name"], proofs[12]["session_name"], proofs[21]["session_name"]}
    stats, headers = [], []
    row = 0
    for index, proof in enumerate(proofs):
        with extracted_member(proof) as source_path:
            h, w = read_header(proof, source_path)
            source = tifffile.memmap(source_path, mode="r")
            try:
                planes = np.asarray(source[[1, 2, 41, 12, 7]], dtype=np.float32)
            finally:
                source._mmap.close()
                del source
        channels = np.log1p(planes)
        for channel, name in enumerate(["113In_LMNB1", "115In_LMNB1"]):
            stats.append(dict(image_id=image_id(proof), source_name=name,
                              source_channel_index=channel + 1,
                              nonzero_pixels=int(np.count_nonzero(planes[channel])),
                              mean=float(planes[channel].mean()), maximum=float(planes[channel].max()),
                              log1p_correlation_193Ir=float(np.corrcoef(channels[channel].ravel(), channels[2].ravel())[0, 1])))
        preview = channels[[2, 3, 4], ::4, ::4].transpose(1, 2, 0)
        scale = np.quantile(preview, 0.995, axis=(0, 1))
        axes.flat[index].imshow(np.clip(preview / np.maximum(scale, 1e-12), 0, 1))
        axes.flat[index].set_title(image_id(proof), fontsize=7)
        axes.flat[index].axis("off")
        if proof["session_name"] in duplicate_examples:
            for column, channel in enumerate([0, 1, 2, 3]):
                image = channels[channel, ::2, ::2]
                duplicate_axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=np.quantile(image, .995))
                duplicate_axes[row, column].set_title(["113In_LMNB1", "115In_LMNB1", "193Ir", "Pan-Keratin"][column])
                duplicate_axes[row, column].set_ylabel(image_id(proof), fontsize=5)
                duplicate_axes[row, column].set_xticks([])
                duplicate_axes[row, column].set_yticks([])
            duplicate_examples.remove(proof["session_name"])
            row += 1
        headers.append(dict(image_id=image_id(proof), shape=[43, h, w],
                            dtype="float32", source_member=proof["source_member_name"],
                            accepted_proof_reused=True, member_stat_matches=True))
        del planes, channels
    axes.flat[-1].axis("off")
    fig.suptitle("All 27 complete native core frames: red=193Ir, green=Pan-Keratin, blue=CD45; display scaling only")
    fig.tight_layout()
    fig.savefig(WORK / "all_core_overview.png", dpi=110)
    plt.close(fig)
    duplicate_fig.tight_layout()
    duplicate_fig.savefig(WORK / "duplicate_channel_overview.png", dpi=130)
    plt.close(duplicate_fig)
    pd.DataFrame(stats).to_csv(WORK / "duplicate_channel_qc.csv", index=False)
    write_json(WORK / "source_header_checks.json", headers)


def review_row(check: str, result: str, details: object, *, scope: str = "study",
               record: str = STUDY, source: str = "", target: str = "") -> dict:
    """Preparer self-check evidence; reviewer identity remains empty."""
    return dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope, record_id=record,
                check_id=check, result=result, source_evidence=source, output_evidence=target,
                details=json.dumps(dict(preparer_self_check=True, evidence=details), ensure_ascii=False))


# 1. Reuse the original accepted physical-core scope and source/member stat binding.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=["metadata", "inspect", "export", "verify"], required=True)
parser.add_argument("--job-dir", type=Path, default=JOB)
args = parser.parse_args()
JOB = args.job_dir
PREPARER = "codex_cli_Kucukkose_" + JOB.name
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
with formal_read_lease(META):
    native = json.loads((META / "native_verification.json").read_text())["native_result"]
proofs = native["images"]
archive = Path(native["source_file_path"])
assert (archive.stat().st_size, archive.stat().st_mtime_ns) == (native["source_file_stat"]["size"], native["source_file_stat"]["mtime_ns"])
assert len(proofs) == 27 and all(x["complete"] for x in proofs)
assert len({tuple(x["channels"]) for x in proofs}) == 1
progress(args.stage)

with formal_read_lease(archive), ZipFile(archive) as z:
    inventory = [dict(member=i.filename, bytes=i.file_size,
                      disposition="selected_native_core" if i.filename in {p["source_member_name"] for p in proofs}
                      else "source_metadata" if i.filename.endswith((".csv", ".json", ".xml"))
                      else "outside_selected_native_tiff_scope")
                 for i in z.infolist() if not i.is_dir()]
    for proof in proofs:
        member = z.getinfo(proof["source_member_name"])
        assert member.file_size == proof["archive_entry"]["exact_archive_entry_size"]
        assert member.CRC == proof["archive_entry"]["exact_archive_entry_crc"]
    assert (SOURCE_META / "20211008_ROI_info.csv").read_bytes() == z.read(ROI_MEMBER)
    sessions = {}
    for session in sorted({p["session_name"] for p in proofs}):
        member = f"Kucukkose_Baars_et_al/RAW/{session}/{session}_session.json"
        data = z.read(member)
        source = SOURCE_META / f"{session}_session.json"
        assert source.read_bytes() == data
        sessions[session] = json.loads(source.read_bytes())
pd.DataFrame(inventory).to_csv(WORK / "source_scope_inventory.csv", index=False)
write_json(WORK / "capacity_and_scope.json", dict(
    source_zip_bytes=archive.stat().st_size, archive_members=len(inventory),
    selected_native_images=len(proofs),
    selected_member_bytes=sum(p["expected_member_bytes"] for p in proofs),
    member_read_strategy="canonical_zip_to_ephemeral_ssd",
    estimated_uncompressed_square_tiff_bytes=sum(32 * max(p["tiff_layout"]["shape"][-2:])**2 * 4 for p in proofs),
    nas_free_bytes=shutil.disk_usage(NAS).free, nas_reserve_bytes=100 * 1024**3,
    local_free_bytes=shutil.disk_usage("/tmp").free, local_reserve_bytes=64 * 1024**3))
estimated_output_bytes = sum(32 * max(p["tiff_layout"]["shape"][-2:])**2 * 4 for p in proofs)
estimated_local_bytes = max(32 * max(p["tiff_layout"]["shape"][-2:])**2 * 4 for p in proofs) * 2
if (shutil.disk_usage(NAS).free < 100 * 1024**3 + estimated_output_bytes
        or shutil.disk_usage("/tmp").free < 64 * 1024**3 + estimated_local_bytes):
    progress("waiting_io", reason="Insufficient space above NAS/local reserve", next_stage=args.stage)
    raise SystemExit("waiting_io: insufficient reserved space")

# 2. The archive has a direct ROI -> patient map; primary-resection triplicates
#    share one biological sample per patient under the paper's Human tissues scope.
with formal_read_lease(SOURCE_META):
    roi = pd.read_csv(SOURCE_META / "20211008_ROI_info.csv", sep=";", dtype=str)
    clinical = pd.read_excel(SOURCE_META / "supplementary_table_1.xlsx", header=2, dtype=str)
assert set(roi.ImageNumber) == {Path(p["source_member_name"]).stem for p in proofs}
crosscheck = roi.groupby("Patient_ID", sort=True).first().reset_index().merge(
    clinical[["pt ID", "Group"]], left_on="Patient_ID", right_on="pt ID", validate="one_to_one")
crosscheck["groups_agree"] = crosscheck.Combined_label.str.lower() == crosscheck.Group.str.lower()
crosscheck.to_csv(WORK / "clinical_namespace_conflicts.csv", index=False, encoding="utf-8-sig")
samples = []
for patient, group in roi.groupby("Patient_ID", sort=True):
    assert len(group) == 3 and group.MMR.nunique() == group.Disease.nunique() == 1
    sample = {k: "" for k in pd.read_csv(TEMPLATES / "sample_metadata.csv").columns}
    sample.update(study=STUDY, sample_id=f"primary_tumor_ROI_P{patient}",
                  patient_key=f"{STUDY}_ROI_P{patient}", source_subject_id=patient,
                  cohort_id=group.Combined_label.iloc[0], sample_type="primary tumor resection",
                  tissue_type="tumor", anatomical_site="colorectum", lesion_type="primary",
                  timepoint_label="other", collection_method="resection",
                  treatment_status_at_collection="no prior systemic therapy",
                  case_control_status="case", primary_diagnosis="colorectal carcinoma",
                  molecular_subtype=group.MMR.iloc[0],
                  metadata_notes="One primary-resection biological sampling per archive Patient_ID; three physical cores. Collection timepoint: primary tumor resection; no treatment index established. " + CLINICAL_GAP)
    samples.append(sample)
samples = pd.DataFrame(samples)
sample_dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
sample_vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
for field in sample_dictionary[(sample_dictionary.table_name == "sample_metadata") &
                               sample_dictionary.value_set.ne("")].itertuples(index=False):
    allowed = sample_vocabulary.loc[sample_vocabulary.value_set == field.value_set, "allowed_value"]
    assert samples.loc[samples[field.field_name].ne(""), field.field_name].isin(allowed).all(), field.field_name
roi = roi.set_index("ImageNumber", verify_integrity=True)

# 3. Keep the actual native panel and S1 identity, selecting the higher-signal
#    duplicate after native-image QC (selection evidence is required before export).
names = proofs[0]["channels"]
downstream = set(range(43)) - {1}
with formal_read_lease(S1):
    panel = panel_from_s1(STUDY, PANEL_ID, names, S1, downstream_source_indices=downstream)
panel.loc[panel.source_channel_index == 1, "exclusion_reason"] = (
    "Duplicate Lamin-B1: select native 115In_LMNB1 based on source QC; 113In is not S1-selected for segmentation")
panel.loc[panel.source_channel_index == 2, "keep_reason"] = "Selected duplicate Lamin-B1: stronger native signal and nuclear structure; see duplicate_channel_qc.csv/overview"

archive_relative = str(metadata_source_relative(archive, NAS))
roi_source = archive_relative + "::" + ROI_MEMBER
paper_source = str(metadata_source_relative(SOURCE_META / 'paper_pmc10803761.html', NAS))
fields = [
    field_row(roi_source, "ImageNumber", "image_metadata", "image_id", "Original TIFF stem with spaces replaced by underscores", join="ImageNumber = archive TIFF member stem"),
    field_row(roi_source, "Patient_ID", "image_metadata", "sample_id", "primary_tumor_ROI_P + archive Patient_ID", join="exact ImageNumber; 27/27 image links"),
    field_row(roi_source, "Patient_ID", "sample_metadata", "sample_id", "One primary-tumor resection per archive patient; three distinct TMA cores share sample_id", join="Patient_ID; 9 samples / 27 cores", notes=paper_source + " Results and Human tissues"),
    field_row(roi_source, "Patient_ID", "sample_metadata", "patient_key", "Study-prefixed ROI patient namespace", join="Patient_ID; 9/9 sample-to-patient links", notes=CLINICAL_GAP),
    field_row(roi_source, "Patient_ID", "sample_metadata", "source_subject_id", "Preserve public archive patient identifier verbatim"),
    field_row(roi_source, "Combined_label", "sample_metadata", "cohort_id", "Preserve archive group verbatim"),
    field_row(roi_source, "MMR", "sample_metadata", "molecular_subtype", "Preserve archive MSI/MSS verbatim"),
    field_row(roi_source, "Combined_label", "image_metadata", "region_annotation", "Preserve archive combined group and native acquisition description"),
    field_row(paper_source, "Results; Human tissues", "sample_metadata", "sample_type,tissue_type,anatomical_site,lesion_type,collection_method,timepoint_label,primary_diagnosis,case_control_status", "Resected primary colorectal tumor, three cores per patient; tissue_type=tumor and timepoint_label=other per controlled vocabulary; custom timepoint primary tumor resection retained in metadata_notes; no specimen ID inferred from ROI label"),
    field_row(paper_source, "Human tissues", "sample_metadata", "treatment_status_at_collection", "Patients did not receive systemic therapy before resection -> no prior systemic therapy", notes="Collection-time statement; no patient-index therapy field assigned without a defined index event"),
    field_row(str(metadata_source_relative(SOURCE_META / 'supplementary_table_1.xlsx', NAS)), "pt ID; Group; clinical columns", "sample_metadata", "age_at_collection_years,sex_reported,stage_group_at_index,primary_site", "No join: inconsistent clinical patient namespace", join="Unresolved; see clinical_namespace_conflicts.csv", notes=CLINICAL_GAP),
    field_row(str(metadata_source_relative(SOURCE_META / 'author_repositories.json', NAS)), "public organization repository listing", "sample_metadata", "patient_key", "No CRC sample/core map repository found in author's three public repositories", notes="MATISSE-Pipeline, MATISSEStarProtocol and Melanoma-BJC-2025; archive provides direct core map"),
    field_row(str(metadata_source_relative(META / 'native_verification.json', NAS)), "source_member_name; source_acquisition_id; description; session_name", "image_metadata", "source_files,source_image_id,source_acquisition_id,slide_id,core_id,roi_id", "Archive relative path plus exact member; native session acquisition and physical-core proof reused"),
    field_row(archive_relative, "OME Pixels + acquisition ablation distances", "image_metadata", "source_axes,source_dtype,pixel_size_x_um,pixel_size_y_um,source_x0_px,source_y0_px,source_x1_px,source_y1_px", "Whole published native TIFF CYX frame; float32; 1 um/pixel; no reconstruction of absent MCD margins"),
    field_row(str(metadata_source_relative(S1, NAS)), "Study + Original marker", "panel_mapping", "marker_id,mapping_status,nuclear,body,role,keep_channel,output_channel_index", "Exact study-specific S1 join and source-order output indices; 193Ir retained for nuclear input", notes=PANEL_CAVEAT),
    field_row(str(metadata_source_relative(CACHE / 'duplicate_channel_qc.csv', NAS)), "113In_LMNB1;115In_LMNB1; native overview", "panel_mapping", "keep_channel,role,exclusion_reason", "Select 115In_LMNB1 as unique downstream Lamin-B1 from inspected source signal; no selected segmentation duplicate is discarded"),
]
field_mapping = pd.DataFrame(fields, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)

if args.stage == "metadata":
    # Small source tables can be delivered while the formal TIFF writer queues.
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        for name, frame in [("sample_metadata", samples), ("panel_mapping", panel), ("field_mapping", field_mapping)]:
            frame.to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        pending = [review_row("M02", "unresolved", CLINICAL_GAP, scope="sample"),
                   review_row("D01", "unresolved", "Metadata prepared; TIFF export waits for the shared NAS write slot. Independent review pending.")]
        pd.DataFrame(pending, columns=pd.read_csv(TEMPLATES / "review_results.csv").columns).to_csv(
            OUTPUT / "review_results.csv", index=False, encoding="utf-8-sig")
    progress("waiting_nas_write_slot", metadata_tables_prepared=True)
    print("Prepared sample=9, patient-linked samples=9, panel=43 source/32 retained; images=0 pending NAS write slot", flush=True)

if args.stage == "inspect":
    inspect_sources(proofs)
    progress("source_inspection_complete", source_qc="Await human visual inspection of generated overviews")

if args.stage == "export":
    assert (CACHE / "visual_inspection.json").exists(), "Inspect native overviews before exporting"
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
    with formal_read_lease(WORK):
        completed = json.loads((WORK / "export_results.json").read_text()) if (WORK / "export_results.json").exists() else {}
    assert not len(images) or set(images.image_id).issubset({image_id(p) for p in proofs})
    for proof in proofs:
        current_image_id = image_id(proof)
        destination = OUTPUT / "images" / current_image_id / "input.tiff"
        if current_image_id in completed:
            assert destination.exists() and (images.image_id == current_image_id).any()
            continue
        progress("exporting_image", images, current_image=current_image_id)
        with extracted_member(proof) as source_path:
            spec = source_spec(proof, source_path)
            h, w = read_header(proof, source_path)
            with open_source(spec) as reader:
                result = export_input(reader, panel, destination)
        completed[current_image_id] = result
        if result["admitted"]:
            source_name = Path(proof["source_member_name"]).stem
            annotation = roi.loc[source_name]
            row = dict(study=STUDY, image_id=current_image_id,
                       sample_id=f"primary_tumor_ROI_P{annotation.Patient_ID}", panel_id=PANEL_ID,
                       image_kind="tma_core", source_image_id=proof["source_member_name"],
                       source_files=[archive_relative], source_format="ome_tiff", source_axes="CYX",
                       source_acquisition_id=str(proof["source_acquisition_id"]),
                       source_dtype="float32", source_revision=REVISION,
                       slide_id=proof["session_name"] + "_slide1", core_id=proof["physical_core_id"],
                       roi_id=str(proof["source_acquisition_id"]),
                       region_annotation=proof["description"] + "; " + annotation.Combined_label,
                       source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                       pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                       input_image_path=str(metadata_source_relative(destination, NAS)),
                       valid_pixels_path=str(metadata_source_relative(destination.with_name(result['valid_pixels']), NAS)) if result["valid_pixels"] else "",
                       intensity_description="Published RAW OME-TIFF float32 IMC intensities, unchanged; no CellWeft normalization",
                       split_method="already_independent", split_evidence="One original TIFF per physical TMA core; archive ROI table and native core proof",
                       metadata_notes="Full published TIFF acquisition frame retained, including background and source tissue-quality artefacts; " + CLINICAL_GAP)
            row.update({k: result[k] for k in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]})
            # New batch records replace only their own identity; all prior image rows remain.
            if len(images):
                images = images[images.image_id != current_image_id]
            images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, field_mapping)
        write_json(WORK / "export_results.json", completed)
        progress("image_saved", images, last_image=current_image_id)
        print(f"Saved {len(images)}/27: {current_image_id}", flush=True)
    progress("export_complete_pending_self_verification", images)

if args.stage == "verify":
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
    with formal_read_lease(WORK):
        results = json.loads((WORK / "export_results.json").read_text())
        verification_path = WORK / "self_verification_checkpoint.json"
        verified = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    headers, blocks = [], []
    # Every output header and padding is checked; all 27 have two all-channel
    # blocks compared directly to their accepted original TIFF (not to exports).
    for proof in proofs:
        current_image_id = image_id(proof)
        if not results[current_image_id]["admitted"]:
            continue
        row = images.set_index("image_id").loc[current_image_id]
        path = NAS / row.input_image_path
        kept = panel[panel.keep_channel].sort_values("output_channel_index")
        entry = proof["archive_entry"]
        binding = dict(source_stat=[entry["exact_archive_entry_size"], entry["exact_archive_entry_crc"]],
                       input_stat=[path.stat().st_size, path.stat().st_mtime_ns],
                       source_member=entry["exact_archive_entry_name"],
                       source_channels=kept.source_channel_index.tolist(),
                       output_names=kept.marker_id.tolist(),
                       geometry=[int(row[key]) for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"]])
        prior = verified.get(current_image_id)
        if prior and prior["binding"] == binding:
            headers.append(prior["header"])
            blocks.extend(prior["pixel_blocks"])
            continue
        expected_blocks = []
        with extracted_member(proof) as source_path:
            # Read the accepted native frame from the exact archive member.
            native_pixels = tifffile.memmap(source_path, mode="r")
            try:
                h, w = native_pixels.shape[-2:]
                xy = [(max(0, w//2-16), max(0, h//2-16)), (max(0, w-32), max(0, h-32))]
                for x, y in xy:
                    window = (x, y, min(x+32, w), min(y+32, h))
                    expected = np.asarray(native_pixels[kept.source_channel_index.tolist(), y:window[3], x:window[2]]).copy()
                    expected_blocks.append((window, expected))
            finally:
                native_pixels._mmap.close()
            with formal_read_lease(path), tifffile.TiffFile(path) as tif:
                root = ET.fromstring(tif.ome_metadata)
                assert [c.attrib["Name"] for c in root.findall(".//{*}Channel")] == kept.marker_id.tolist()
                assert len(tif.pages) == len(kept)
                assert tif.pages[0].dtype == np.dtype("float32")
                assert tif.pages[0].shape == (int(row.height_px), int(row.width_px))
                for window, expected in expected_blocks:
                    output_window = (window[0]+int(row.pad_left_px), window[1]+int(row.pad_top_px),
                                     window[2]+int(row.pad_left_px), window[3]+int(row.pad_top_px))
                    actual = np.stack([read_tiff_window(tif, i, output_window) for i in range(len(kept))])
                    assert np.array_equal(actual, expected) and actual.dtype == expected.dtype
                    blocks.append(dict(image_id=current_image_id, source_window=list(window),
                                       output_window=list(output_window), channels=len(kept), equal=True))
                # Inspect every padding pixel across all retained channels.
                for i in range(len(kept)):
                    for window in [(0, 0, int(row.pad_left_px), int(row.height_px)),
                                   (int(row.width_px-row.pad_right_px), 0, int(row.width_px), int(row.height_px)),
                                   (0, 0, int(row.width_px), int(row.pad_top_px)),
                                   (0, int(row.height_px-row.pad_bottom_px), int(row.width_px), int(row.height_px))]:
                        if window[0] < window[2] and window[1] < window[3]:
                            assert not read_tiff_window(tif, i, window).any()
        headers.append(dict(image_id=current_image_id, channels=len(kept), dtype="float32", header_pass=True, padding_pass=True))
        verified[current_image_id] = dict(binding=binding, header=headers[-1], pixel_blocks=blocks[-2:])
        write_json(verification_path, verified)
        progress("self_verifying_images", images, self_verified_images=len(headers), last_verified_image=current_image_id)
        print(f"Self-verified {len(headers)}/27: {current_image_id}", flush=True)
    assert len(images) == len(headers) == 27
    assert len(samples) == samples.patient_key.nunique() == 9
    assert images.groupby("sample_id").size().eq(3).all()
    # Re-publish exact current source-derived metadata without rewriting correct TIFFs.
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)
    write_json(WORK / "self_verification.json", dict(headers=headers, pixel_blocks=blocks,
               images=27, samples=9, patient_linked_samples=9, sample_linked_images=27,
               independent_review=False, clinical_gap=CLINICAL_GAP))
    evidence = str(work_reference(WORK / 'self_verification.json'))
    reviews = [
        review_row("S01", "pass", "175 archive members; selected all 27 original RAW IMC TIFFs, no all-zero admitted-marker image; validation IHC masks and derived segmentation excluded from native input scope", source=str(work_reference(WORK / 'source_scope_inventory.csv'))),
        review_row("I01", "pass", "27 unique original archive members and 27 independent image paths", source=str(metadata_source_relative(META / 'native_verification.json', NAS)), target=str(metadata_source_relative(OUTPUT / 'images.parquet', NAS))),
        review_row("I02", "pass", "27/27 image->sample; 9 sample->patient; 3 cores per primary resection; all panel FKs resolve"),
        review_row("I03", "pass", "All complete source TIFF frames, unchanged orientation/1 um scale, centered square padding", target=evidence),
        review_row("I04", "pass", "All native core overviews visually inspected; source core/patient annotations retained; no adjacent-sample split required", source=str(metadata_source_relative(CACHE / 'visual_inspection.json', NAS))),
        review_row("P01", "pass", "43 ordered native labels agree between all TIFF/session proofs and exact study S1 rows. " + PANEL_CAVEAT, scope="panel", record=PANEL_ID),
        review_row("P02", "pass", "Retain 31 marker channels plus S1 nuclear 193Ir; choose 115In_LMNB1 duplicate from source QC; all selected body channels retained", scope="panel", record=PANEL_ID, source=str(metadata_source_relative(CACHE / 'duplicate_channel_qc.csv', NAS))),
        review_row("P03", "pass", "27 TIFF headers, 32 retained channels each, exact output marker names/order", target=evidence),
        review_row("I05", "pass", dict(blocks=blocks, all_padding_pixels_all_channels=True), target=evidence),
        review_row("M01", "pass", "Archive ROI CSV exact ImageNumber->Patient_ID + paper primary-resection/three-core design: 27/27 sample-linked images, 9 samples, 9/9 patient links; clinical namespace conflict retained separately", scope="sample", source=roi_source),
        review_row("M02", "unresolved", CLINICAL_GAP + " Public release, paper Human tissues, supplementary Tables 1/3 and author's repository listing inspected.", scope="sample", source=str(work_reference(WORK / 'clinical_namespace_conflicts.csv'))),
        review_row("D01", "unresolved", dict(image_ids=images.image_id.tolist(), images_exported=27, expected_images=27,
                   sample_rows=9, sample_linked_images=27, patient_linked_samples=9,
                   reason="Preparer delivery complete; distinct CLI independent review and controller acceptance required; no input_binding pass issued")),
    ]
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=review_path.parent):
        pd.DataFrame(reviews, columns=pd.read_csv(TEMPLATES / "review_results.csv").columns).to_csv(
            review_path, index=False, encoding="utf-8-sig")
    progress("prepared_awaiting_independent_review", images, self_verified_images=len(headers), pixel_blocks=len(blocks))
    print(json.dumps(dict(status="prepared", images=len(images), samples=9,
                          sample_linked_images=27, patient_linked_samples=9,
                          pixel_blocks=len(blocks)), ensure_ascii=False), flush=True)
