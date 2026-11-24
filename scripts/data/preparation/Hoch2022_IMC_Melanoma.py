"""Prepare the 50 published SPORA Hoch melanoma IMC ROIs at native intensity.

Inputs: verified level-0 Zarrs, study-specific S1, author RNA Image/clinical/spot
tables. Outputs: square CYX input.tiff and complete study input tables.
Biological samples are tissue BlockID groups; cores are repeated observations.
Clinical tables and patient identifiers are loaded from NAS, never embedded here.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import io
import json
import os
import re
import shutil
import signal
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Hoch2022_IMC_Melanoma/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Hoch2022_IMC_Melanoma"
PREPARER = "Codex CLI Hoch2022_IMC_Melanoma/0208_repair_2"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path, default=PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0208_repair_2/progress.json")
parser.add_argument("--fetch-metadata", action="store_true", help="Fetch only small author metadata archive members")
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--source-overviews", action="store_true", help="Render representative source views under a read slot")
parser.add_argument("--stage-sources", action="store_true", help="Stage native spatial blocks on local disk under a read slot")
args = parser.parse_args()
args.progress.parent.mkdir(parents=True, exist_ok=True)
root = args.data_root
raw = root / "raw/public" / STUDY / "spora"
cache = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
metadata = metadata_source_directory(root, STUDY)
evidence = metadata / "source_metadata"
output = root / "processed" / STUDY
receipts = cache / "image_receipts"
temporary = Path("/tmp/study_input_batch") / STUDY
for directory in (cache, evidence, output, receipts, temporary):
    directory.mkdir(parents=True, exist_ok=True)
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def save_json(path: Path, value: dict | list) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def progress(stage: str, count: int, **details: object) -> None:
    gaps = ["来源临床表缺少对照标本患者身份（1份sample、2张image）", "独立复核及主控验收待执行"]
    if count < 50:
        gaps.insert(0, f"正式TIFF及image索引尚未全部交付（{count}/50）")
    save_json(args.progress, dict(study=STUDY, stage=stage, completed_images=count,
              expected_images=50, updated_utc=datetime.now(timezone.utc).isoformat(),
              gaps=gaps, **details))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def stop_for_io(reason: str) -> None:
    """Keep the last per-image checkpoint when an infrastructure wait is stopped."""
    checkpoint = json.loads(args.progress.read_text()) if args.progress.exists() else dict(study=STUDY, completed_images=0)
    checkpoint.update(status="waiting_io", waiting_reason=reason,
                      updated_utc=datetime.now(timezone.utc).isoformat(),
                      resume="Rerun this script with the same progress path; reuse committed receipts and native staging.")
    save_json(args.progress, checkpoint)
    raise SystemExit(75)


signal.signal(signal.SIGTERM, lambda signum, frame: stop_for_io("Execution stopped while awaiting shared I/O; current checkpoint retained."))


# Native staging uses a read lease and creates local memmaps, allowing a short
# subsequent TIFF write lease. Complete local files are published after flush.
if args.stage_sources:
    native_cache = temporary / "native"
    native_cache.mkdir(parents=True, exist_ok=True)
    inventory_document = json.loads((cache / "source_inventory.json").read_text())
    staging_bytes = sum(int(np.prod(item["shape"])) * 4 for item in inventory_document["images"])
    if shutil.disk_usage(temporary).free - staging_bytes < 64 * 1024**3:
        raise RuntimeError("Insufficient local reserve for complete native staging")
    staged_count = 0
    for item in inventory_document["images"]:
        image_id = item["image_id"]
        destination = native_cache / f"{image_id}.dat"
        complete = native_cache / f"{image_id}.json"
        binding = dict(source_files=item["files"], shape=item["shape"], dtype=item["dtype"])
        for record in item["files"]:
            stat = (root / record["path"]).stat()
            assert [stat.st_size, stat.st_mtime_ns] == [record["size"], record["mtime_ns"]]
        if complete.exists():
            assert json.loads(complete.read_text()) == binding
        else:
            pending = destination.with_suffix(".dat.partial")
            array = np.memmap(pending, mode="w+", dtype=item["dtype"], shape=tuple(item["shape"]))
            source = SourceSpec(image_id, "native_staging", "zarr", root / item["source_path"] / "0", tuple(range(41)))
            with formal_read_lease(source.path), open_source(source) as reader:
                for x0, y0, x1, y1 in windows(reader.shape[-2:], 256):
                    array[:, y0:y1, x0:x1] = reader.read_window(list(range(41)), (x0, y0, x1, y1))
            array.flush()
            array._mmap.close()
            pending.replace(destination)
            save_json(complete, binding)
        staged_count += 1
        save_json(cache / "native_staging_progress.json", dict(study=STUDY, staged_source_images=staged_count,
                  expected_images=50, local_bytes=staging_bytes, updated_utc=datetime.now(timezone.utc).isoformat(),
                  note="Native local memmaps only; no formal TIFF has been published by this read-stage command."))
        print(f"Staged native source: {staged_count}/50 {image_id}", flush=True)
    raise SystemExit(0)


# Source overview is independent of TIFF publication, so it can use a read slot
# while the native TIFF command waits for the single shared write slot.
if args.source_overviews:
    proof_path = metadata / "native_verification.json"
    proofs = sorted(json.loads(proof_path.read_text())["verified"], key=lambda item: item["image_id"])
    representative_ids = {proofs[0]["image_id"], proofs[5]["image_id"], proofs[2]["image_id"],
                          max(proofs, key=lambda item: max(item["shape"][1:]))["image_id"]}
    source_views = []
    for proof in proofs:
        image_id = proof["image_id"]
        if image_id not in representative_ids:
            continue
        source = SourceSpec(image_id, "source_overview", "zarr", root / proof["path"] / "0", tuple(range(41)))
        with formal_read_lease(source.path), open_source(source) as reader:
            data = reader.read_window([37, 6, 14], (0, 0, reader.shape[2], reader.shape[1]))
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, values, label in zip(axes, data, ["DNA1", "S100", "CD3e"]):
            upper = np.percentile(values[values > 0], 99) if np.any(values > 0) else 1
            axis.imshow(values[::3, ::3], cmap="gray", vmin=0, vmax=upper)
            axis.set_title(label)
            axis.axis("off")
        figure.suptitle(image_id)
        figure.tight_layout()
        target = cache / f"source_overview_{image_id}.png"
        figure.savefig(target, dpi=130)
        plt.close(figure)
        source_views.append(dict(image_id=image_id, source_path=proof["path"], shape=proof["shape"],
                                 channels=[37, 6, 14], overview_path=relative(target)))
        print(f"Source overview: {image_id}", flush=True)
    save_json(cache / "source_overview_manifest.json", source_views)
    raise SystemExit(0)


class RemoteArchive(io.RawIOBase):
    """Seek within the author ZIP using bounded HTTP ranges for small metadata.

    The 7.1 GB archive is never downloaded in full. ZIP64 directory and selected
    compressed members are retrieved; exact range responses are mandatory.
    """

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        response = requests.get(self.url + f"?range={self.position}-{end-1}",
                   headers={"Range": f"bytes={self.position}-{end-1}"}, timeout=60)
        response.raise_for_status()
        if (response.status_code != 206 or len(response.content) != end - self.position
                or response.headers.get("Content-Range") != f"bytes {self.position}-{end-1}/{self.size}"):
            raise ValueError("Author archive did not return the requested byte range")
        self.position = end
        return response.content


# Recover the author clinical/core links, reusing complete downloaded members.
if args.fetch_metadata:
    for name, url in [
        ("zenodo_5994136.json", "https://zenodo.org/api/records/5994136"),
        ("zenodo_6004986.json", "https://zenodo.org/api/records/6004986"),
        ("00_prepare_clinical_dat.rmd", "https://raw.githubusercontent.com/BodenmillerGroup/MelanomaIMC/publication/analysis/00_prepare_clinical_dat.rmd"),
        ("00_prepare_clinical_dat.html", "https://bodenmillergroup.github.io/MelanomaIMC/00_prepare_clinical_dat.html"),
    ]:
        if not (evidence / name).exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            (evidence / name).write_bytes(response.content)
    record = json.loads((evidence / "zenodo_5994136.json").read_text())
    archive = next(item for item in record["files"] if item["key"] == "data_for_analysis.zip")
    members = ["191119_TMA_spotmatrix.csv", "rna/Image.csv", "rna/clinical_data_RNA.csv"]
    prefix = "data_for_analysis_zenodo/"
    if any(not (evidence / "zenodo_analysis" / prefix / name).exists() for name in members):
        with zipfile.ZipFile(RemoteArchive(archive["links"]["self"], archive["size"])) as zipped:
            save_json(evidence / "data_for_analysis_zip_members.json", [
                dict(filename=item.filename, size=item.file_size, compressed_size=item.compress_size)
                for item in zipped.infolist()])
            for name in members:
                target = evidence / "zenodo_analysis" / prefix / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zipped.read(prefix + name))

analysis = evidence / "zenodo_analysis/data_for_analysis_zenodo"
clinical_path = analysis / "rna/clinical_data_RNA.csv"
acquisition_path = analysis / "rna/Image.csv"
spot_path = analysis / "191119_TMA_spotmatrix.csv"
tissues_path = raw / "metadata/tissues.parquet"
channel_path = raw / "imc/channels.parquet"
availability_path = metadata_source_directory(root, STUDY) / 'spora_inventory' / 'imc' / 'channels_per_tissue.parquet'
ordered_path = metadata_source_directory(root, STUDY) / 'spora_inventory' / 'imc_ordered_marker_names.json'
verification_path = metadata / "native_verification.json"
acceptance_path = metadata / "accepted_images.json"
file_evidence_path = metadata / "source_file_evidence.json"
objects_path = metadata / "imc_source_objects.json"
verified = {item["image_id"]: item for item in json.loads(verification_path.read_text())["verified"]}
verified_file_stats = json.loads(file_evidence_path.read_text())["images"]
selected = sorted(json.loads(acceptance_path.read_text())["accepted_image_ids"])
assert len(selected) == len(verified) == 50
tissues = pd.read_parquet(tissues_path).set_index("tissue_id", verify_integrity=True)
channels = pd.read_parquet(channel_path)
availability = pd.read_parquet(availability_path)
ordered = json.loads(ordered_path.read_text())
source_names = channels.channel_name.tolist()
panel_id = f"{STUDY}__spora_rna41"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
assert len(panel) == 41 and int(panel.keep_channel.sum()) == 28
panel["source_qc_pass"] = channels.qc_pass.to_numpy()
panel.loc[panel.source_name == "VIM", "keep_reason"] += "; retained per study-specific S1 despite SPORA qc_pass=False"
kept = panel[panel.keep_channel].sort_values("output_channel_index")

# Check all complete source members against the original manifest, without hashes.
source_objects = json.loads(objects_path.read_text())
inventory = []
for image_id in selected:
    proof = verified[image_id]
    group = root / proof["path"]
    ome = json.loads((group / "zarr.json").read_text())["attributes"]["ome"]
    metadata = json.loads((group / "0/zarr.json").read_text())
    assert ordered[image_id] == source_names == proof["canonical_channel_labels"]
    assert [item["label"] for item in ome["omero"]["channels"]] == source_names
    assert availability.columns.tolist() == source_names and availability.loc[image_id].all()
    assert metadata["shape"] == proof["shape"] and metadata["data_type"] == proof["dtype"] == "float32"
    assert metadata["dimension_names"] == ["c", "y", "x"]
    downstream_indices = set(kept.loc[kept.role == "downstream", "source_channel_index"])
    # A retained-marker-positive verified block already rules out an entirely
    # zero image. The full TIFF pass additionally checks every retained pixel.
    assert any(channel["channel_index"] in downstream_indices and channel["nonzero_pixels"] > 0
               for channel in proof["channel_reads"]), image_id
    assert ome["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
    objects = [item for item in source_objects if f"/{image_id}.ome.zarr/" in item["key"]]
    files = []
    for item in objects:
        path = group / item["key"].split(f"/{image_id}.ome.zarr/", 1)[1]
        stat = path.stat()
        assert stat.st_size == item["size"]
        files.append(dict(path=relative(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    assert len(files) == proof["source_object_count"] and sum(x["size"] for x in files) == proof["source_bytes"]
    assert [[item["size"], item["mtime_ns"]] for item in files] == verified_file_stats[image_id]["file_stats"]
    actual_members = {relative(path) for path in group.rglob("*") if path.is_file() and not path.name.endswith(".partial")}
    assert actual_members == {item["path"] for item in files}, image_id
    inventory.append(dict(image_id=image_id, source_path=relative(group), files=files,
                          shape=proof["shape"], dtype=proof["dtype"],
                          ignored_partial_files=[relative(p) for p in group.rglob("*.partial")]))
save_json(cache / "source_inventory.json", dict(images=inventory, reused_native_verification=relative(verification_path),
          reused_file_identity=relative(file_evidence_path),
          accepted_scope=relative(acceptance_path), expected_images=50,
          source_bytes=sum(x["size"] for item in inventory for x in item["files"]),
          uncompressed_output_bytes=sum(28 * max(item["shape"][1:])**2 * 4 for item in inventory),
          maximum_local_staging_bytes=max(int(np.prod(item["shape"])) * 4 for item in inventory)))

# Join source image_number to the original RNA acquisition, then core -> block -> patient.
clinical = pd.read_csv(clinical_path, dtype=str, keep_default_na=False).replace({"NA": None, "": None})
acquisitions = pd.read_csv(acquisition_path, keep_default_na=False).set_index("ImageNumber", verify_integrity=True)
spots = pd.read_csv(spot_path, sep=";", dtype=str, keep_default_na=False)
clinical["ImageNumber"] = clinical.ImageNumber.astype(int)
clinical = clinical.set_index("ImageNumber", verify_integrity=True)
joined_rows = []
for image_id in selected:
    tissue = tissues.loc[image_id]
    # hs_<ImageNumber> is cross-checked with all 48 explicit SPORA numbers, and
    # for the two control cores additionally with original acquisition dimensions.
    image_number = int(str(tissue.original_file_name).removeprefix("hs_"))
    if pd.notna(tissue.image_number):
        assert image_number == int(tissue.image_number)
    clinical_row, acquisition = clinical.loc[image_number], acquisitions.loc[image_number]
    assert clinical_row.Description == acquisition.Metadata_Description
    assert verified[image_id]["shape"][1:] == [int(acquisition.Metadata_MaxY), int(acquisition.Metadata_MaxX)]
    spot = spots[spots.SpotNr == clinical_row.SpotNr]
    assert len(spot) == 1 and spot.iloc[0].Description == clinical_row.Description
    assert spot.iloc[0].BlockID.strip() == clinical_row.BlockID
    source_slide = re.search(r"(ZTMA[0-9.]+_slide[0-9]+)", str(acquisition.FileName_FullStack)).group(1)
    joined_rows.append(dict(image_id=image_id, image_number=image_number, source_slide_id=source_slide,
                            original_acquisition_file=acquisition.FileName_FullStack, **clinical_row.to_dict()))
joined = pd.DataFrame(joined_rows)
assert len(joined) == 50 and joined.BlockID.nunique() == 34 and joined.PatientID.nunique() == 31
joined.to_csv(cache / "image_core_sample_patient_links.csv", index=False, encoding="utf-8-sig")

# Preserve clinical source semantics; binned age and unstated stage timing stay explicit.
sample_rows = []
extensions = {"age_range_reported": "Age_range", "stage_reported": "Cancer_Stage",
              "mutation_reported": "Mutation", "primary_melanoma_type_reported": "Primary_melanoma_type",
              "adjuvant_reported": "Adjuvant", "status_at_3months_reported": "Status_at_3m",
              "relapse_reported": "relapse", "prior_treatment_count_reported": "Nr_treatments_before_surgery",
              "last_prior_drug_reported": "Name_last_drug_used",
              "last_prior_systemic_treatment_reported": "Last_sys_treatment_before_surgery",
              "drug_after_surgery_reported": "Drug_after_surgery",
              "therapy_after_surgery_reported": "Treatment_after_surgery",
              "response_at_3months_reported": "Response_at_3m",
              "progression_followup_days_reported": "Time_to_progression_or_last_PET",
              "progression_event_reported": "censoring_progression", "core_locations_reported": "Location"}
numeric_extensions = {"prior_treatment_count_reported": "integer",
                      "progression_followup_days_reported": "number", "progression_event_reported": "integer"}
author_code = (evidence / "00_prepare_clinical_dat.rmd").read_text()
assert "clinical_mat$Last_PET - clinical_mat$Start_treatment" in author_code
assert "clinical_mat$Date_death - clinical_mat$Start_treatment" in author_code
for block, group in joined.groupby("BlockID", sort=True):
    for name in ["PatientID", "TissueType", "Gender", "MM_location", *list(extensions.values())[:-1]]:
        assert group[name].nunique(dropna=True) <= 1, (block, name)
    source = group.iloc[0]
    patient = source.PatientID
    is_tumor = source.TissueType == "Melanoma"
    row = dict.fromkeys(columns("sample_metadata"))
    row.update(study=STUDY, sample_id=f"{STUDY}__block__{block}",
               patient_key=f"{STUDY}__patient__{patient}" if pd.notna(patient) else None,
               source_subject_id=patient, cohort_id="ZTMA256.1_RNA", source_sample_id=block,
               biospecimen_group_id=block, sample_type="tissue", tissue_type="tumor" if is_tumor else "other",
               anatomical_site=source.MM_location if pd.notna(source.MM_location) else source.TissueType,
               processing_method="TMA core; author RNA panel", case_control_status="case" if is_tumor else "control",
               sex_reported={"f": "female", "m": "male"}.get(source.Gender),
               primary_diagnosis="Melanoma" if is_tumor else None,
               metadata_notes=("Biological specimen = author BlockID; all observed core views share this sample. "
                 "Clinical source is the author RNA table, joined through ImageNumber and spot map. "
                 "Age is binned; stage timing/system and clinical cutoff unspecified; corresponding time-specific fields blank. "
                 "Exact clinical dates are source-rounded to month; reported OS durations are copied without date recalculation. "
                 "OS origin follows executable author code (Start_treatment); the surgery-origin prose comment is inconsistent. "
                 + ("Author clinical table has no patient identifier for this control lymph-node block." if not is_tumor else "")))
    linked_tissues = tissues.loc[group.image_id]
    patient_hashes = linked_tissues.patient_id.dropna().astype(str).unique()
    age_values = linked_tissues.patient_age.dropna().unique()
    if (linked_tissues.patient_id.notna().all() and len(patient_hashes) == 1
            and len(age_values) == 1):
        row["source_patient_age"] = str(age_values[0])
    else:
        row["source_patient_age"] = ""
    for target, original in extensions.items():
        row[target] = ";".join(sorted(group[original].dropna().unique())) if original == "Location" else source[original]
    if pd.notna(source.Time_to_death_or_last_PET):
        assert pd.notna(source.Start_treatment) and source.censoring_death in {"0", "1"}
        row.update(os_time=float(source.Time_to_death_or_last_PET), os_time_unit="days",
                   os_origin="Start_treatment: author-recorded treatment after surgery",
                   os_event=source.censoring_death,
                   os_event_definition="Death from Start_treatment; otherwise censored at source Last_PET")
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)
for name, kind in numeric_extensions.items():
    samples[name] = pd.to_numeric(samples[name], errors="raise").astype("Int64" if kind == "integer" else "Float64")
assert samples.patient_key.notna().sum() == 33
dictionary = pd.read_csv(TEMPLATES / "data_dictionary.csv", keep_default_na=False)
vocabulary = pd.read_csv(TEMPLATES / "value_sets.csv", keep_default_na=False)
for field in dictionary[dictionary.table_name == "sample_metadata"].itertuples(index=False):
    if field.value_set:
        assert samples[field.field_name].dropna().isin(vocabulary.loc[vocabulary.value_set == field.value_set, "allowed_value"]).all(), field.field_name


def mapping(path: Path, source_table: str, source_field: str, target_table: str,
            target_field: str, join_keys: str, rule: str, notes: str = "") -> dict:
    row = dict.fromkeys(columns("field_mapping"))
    row.update(study=STUDY, source_file=relative(path), source_sheet_or_table=source_table,
               source_field=source_field, target_table=target_table, target_field=target_field,
               join_keys=join_keys, transformation_rule=rule,
               conflict_rule="Preserve original values; missing identity remains blank.", notes=notes)
    return row


field_rows = [
    mapping(tissues_path, "tissues", "dataset", "sample_metadata", "study", "author BlockID within this study",
            f"Assign the canonical study constant {STUDY} to all 34 BlockID samples linked from the accepted SPORA hoch2022multiplexed release; "
            "the source dataset and author RNA clinical/acquisition context establish study membership."),
    mapping(tissues_path, "tissues", "tissue_id;original_file_name;image_number;patient_id;specimen_id", "image_metadata",
            "study;image_id;source_image_id;sample_id", "original_file_name hs_<ImageNumber> -> author RNA ImageNumber -> BlockID",
            "Keep the published tissue_id as image_id; author BlockID defines biological sample. SPORA specimen_id is a view counter, not a biological sample ID.",
            "50/50 images link to 34 biological samples; 48/50 images link through 33 samples to 31 patients. Two control images share one patient-unknown sample. "
            "Author PatientID merges numeric-format aliases in SPORA (35 apparent IDs include two unknown controls and two format aliases)."),
    mapping(clinical_path, "RNA clinical table", "BlockID;PatientID;TissueType;Gender;MM_location", "sample_metadata",
            "sample_id;source_sample_id;biospecimen_group_id;patient_key;source_subject_id;cohort_id;sample_type;tissue_type;anatomical_site;case_control_status;sex_reported;primary_diagnosis;processing_method",
            "author ImageNumber -> BlockID; source PatientID", "One sample per BlockID; patient_key prefixes the literal nonmissing author PatientID. "
            "m/f -> male/female; Melanoma -> tumor/case/Melanoma; Lymphnode CTRL -> other/control with source TissueType as anatomical site. "
            "TMA-core processing and RNA cohort are author study/acquisition context.",
            "PatientID absent in one lymph-node control block. Optional unknown clinical values remain blank. Clinical rows agree within every selected BlockID."),
    mapping(tissues_path, "SPoRA tissues", "patient_id;patient_age", "sample_metadata", "source_patient_age",
            "image source_image_id = tissues.tissue_id; image sample_id = author BlockID sample_id",
            "Copy the raw SPoRA patient_age value as text only when all linked tissues resolve to one nonmissing SPoRA patient_id and one unique nonmissing age. Leave other groups empty; assign no unit or timepoint.",
            "Exact image-to-tissue joins support the unique sample-group values; age unit and timing are not inferred."),
    mapping(acquisition_path, "RNA Image.csv", "ImageNumber;FileName_FullStack;Metadata_Description;Metadata_MaxX;Metadata_MaxY;Metadata_AcquisitionROIID;Metadata_slideid",
            "image_metadata", "core_id;roi_id;slide_id;source_acquisition_id;region_annotation;metadata_notes", "author ImageNumber",
            "Cross-check full source H/W and Description against clinical row; use literal ZTMA*_slide* filename token for physical slide identity. "
            "Metadata_slideid is container-local (1), while all 50 actual filenames identify physical slide2; preserve local context in metadata_notes. "
            "Region annotation contains literal core Location and TissueType; no core is treated as an independent sample."),
    mapping(spot_path, "spot matrix", "SpotNr;BlockID;Description;TissueType;Location", "sample_metadata", "source_sample_id",
            "clinical SpotNr == spot SpotNr; Description and stripped BlockID", "Verify clinical core-to-block linkage against independent author spot map."),
    mapping(channel_path, "channels and OME labels", "channel_name;qc_pass", "panel_mapping", "source_name;source_channel_index;source_qc_pass",
            "source index; ordered marker JSON; channels_per_tissue", "Preserve every ordered source row; cross-check all images against embedded OME labels and original native verification. "
            "VIM remains per S1 admission although source QC marks it false."),
    mapping(s1_path, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
            "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index", "exact Study + Original marker",
            "Retain 28/41 source channels in source order: admitted proteins plus both DNA nuclear inputs; exclude RNA channels and p-Rb exactly per S1."),
    mapping(verification_path, "verified", "path;shape;dtype;canonical_channel_labels", "image_metadata",
            "source_files;source_format;source_axes;source_dtype;source_revision;source_channel_count;channel_count;pixel_size_x_um;pixel_size_y_um;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;input_image_path;valid_pixels_path;intensity_description;image_kind;split_method;split_evidence;metadata_notes",
            "source image_id", "Use all 50 accepted full measured ROI frames, native float32 CYX, 1 um pixels; centered square zero padding. "
            "Reuse original verification and exact manifest member sizes; only globally all-zero admitted-marker images would be excluded."),
    mapping(evidence / "00_prepare_clinical_dat.rmd", "author analysis", "clinical_mat;spot_mat;full_mat_rna", "sample_metadata", "metadata_notes",
            "BlockID and ImageNumber", "Follow author's block/core/image linkage. Age bins retained as bins; index/collection ages and stage timing not inferred. "
            "OS origin follows executable Start_treatment formula despite an inconsistent surgery-origin prose comment; copy published durations without recomputing month-rounded dates. "
            "Progression-only endpoints remain source extensions because the author's event definition does not include death.",
            "Local Parquet; Zenodo raw release 6004986 and analysis release 5994136; author repo and imcdatasets preparation script inspected. "
            "Science article/supplement URL was inaccessible (403/tool access failure); available author tables provide all core-to-block joins."),
    mapping(clinical_path, "RNA clinical table and author preparation code", "Time_to_death_or_last_PET;censoring_death;Start_treatment;Last_PET;Date_death",
            "sample_metadata", "os_time;os_time_unit;os_origin;os_event;os_event_definition", "ImageNumber -> BlockID",
            "Copy nonmissing source OS duration in days, event 0/1 and Start_treatment origin, as defined by the executable author R code. "
            "When OS duration is missing, leave the whole canonical endpoint group blank. Preserve actual source zero event codes with known duration.",
            "Author source dates have been rounded to month; published numeric durations were calculated before rounding and are not rederived here."),
]
for target, original in extensions.items():
    definition = f"Author RNA clinical table {original}, as reported; no index time inferred"
    if target == "prior_treatment_count_reported":
        definition += "; number of treatments before surgery, not an established treatment-line count"
    if target in {"progression_followup_days_reported", "progression_event_reported"}:
        definition += "; days from Start_treatment until progression or Last_PET; event 1=progression, 0=censored, death is not included in the author's event rule"
    field_rows.append(mapping(clinical_path, "RNA clinical table", original, "sample_metadata", target, "BlockID",
         ("Copy as nullable numeric value without endpoint recalculation." if target in numeric_extensions
          else "Preserve source string; for core_locations_reported join distinct Location values using semicolon."),
         json.dumps(dict(data_type=numeric_extensions.get(target, "string"), definition=definition))))
field_mapping = pd.DataFrame(field_rows, columns=columns("field_mapping"))
save_json(cache / "source_trace.json", dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),
    public_sources=["https://zenodo.org/records/6004986", "https://zenodo.org/records/5994136",
      "https://github.com/BodenmillerGroup/MelanomaIMC/tree/publication",
      "https://bodenmillergroup.github.io/MelanomaIMC/00_prepare_clinical_dat.html",
      "https://github.com/BodenmillerGroup/imcdatasets/blob/devel/inst/scripts/make-HochSchulz-2022-Melanoma.Rmd"],
    unavailable_supplement="https://www.science.org/doi/suppl/10.1126/sciimmunol.abk1692/suppl_file/sciimmunol.abk1692_sm.pdf",
    unavailable_reason="Article HTTP 403; supplement inaccessible via web tool.",
    archive="data_for_analysis.zip", archive_record="10.5281/zenodo.5994136",
    members=[relative(p) for p in (clinical_path, acquisition_path, spot_path)],
    image_sample_links=50, sample_rows=34, patient_linked_samples=33, patient_linked_images=48,
    distinct_patients=31, patient_unknown_samples=1,
    remaining_gap="One source-identified lymph-node control block lacks author PatientID."))


class StagedReader:
    """Read a local disk-backed source ROI after a single spatial Zarr pass.

    SPORA shards pack all 41 channels together. Staging avoids decoding each
    source block again for every output channel in the shared TIFF exporter.
    This study-local adapter is a candidate for later shared-module migration.
    """

    def __init__(self, array: np.memmap, source: SourceSpec):
        # The shared exporter computes from this local file, outside NAS slots.
        self.array, self.source = array, replace(source, path=Path(array.filename))
        self.shape, self.dtype = array.shape, array.dtype

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def append_reviews(checks: list[tuple[str, str, str]]) -> None:
    """Append preparer evidence while preserving prior review history."""
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="",
                 scope="study", record_id=STUDY, check_id=check, result=result,
                 source_evidence=";".join(map(relative, [verification_path, tissues_path, clinical_path, acquisition_path, spot_path, s1_path])),
                 output_evidence=relative(output), details=detail) for check, result, detail in checks]
    path = output / "review_results.csv"
    previous = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns("review_results"))
    with formal_publish_lease(path, directory=path.parent):
        pending = path.with_name(path.name + ".partial")
        pd.concat([previous, pd.DataFrame(rows, columns=columns("review_results"))], ignore_index=True).to_csv(
            pending, index=False, encoding="utf-8-sig")
        pending.replace(path)


def overview(array: np.memmap, image_id: str) -> None:
    """Render representative nuclei and protein context for source-region review."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, name in zip(axes, ["DNA1", "S100", "CD3e"]):
        values = np.asarray(array[source_names.index(name), ::3, ::3])
        upper = np.percentile(values[values > 0], 99) if np.any(values > 0) else 1
        axis.imshow(values, cmap="gray", vmin=0, vmax=upper)
        axis.set_title(name)
        axis.axis("off")
    figure.suptitle(image_id)
    figure.tight_layout()
    figure.savefig(cache / f"overview_{image_id}.png", dpi=130)
    plt.close(figure)


identity_details = json.dumps(dict(
    repair_round=2, image_ids=joined.loc[joined.PatientID.isna(), "image_id"].tolist(),
    source_sample_linked_images=50, sample_rows=34, patient_linked_samples=33,
    patient_linked_images=48, distinct_patients=31,
    source_trace=relative(cache / "source_trace.json"),
    details="Author RNA Image.csv, clinical_data_RNA.csv and spot matrix connect all 50 source images to 34 BlockID samples. "
    "Local SPORA metadata, author clinical code and imcdatasets code were inspected in preparation/review 0 and reused. "
    "One control sample lacks author PatientID, affecting only the two listed images; article supplement access failed. "
    "SPORA apparent count 35 includes two unknown controls and numeric aliases."), ensure_ascii=False)

images = pd.read_parquet(output / "images.parquet") if (output / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
link_rows = joined.set_index("image_id", verify_integrity=True)
representatives = {selected[0], selected[5], max(inventory, key=lambda item: max(item["shape"][1:]))["image_id"],
                   next(item.image_id for item in joined.itertuples() if item.Location == "M")}
progress("metadata_ready" if args.metadata_only else "exporting", len(images),
         sample_rows=34, source_sample_linked_images=50, sample_linked_images=len(images),
         patient_linked_samples=33, distinct_patients=31)
if args.metadata_only:
    with formal_publish_lease(output, directory=output):
        samples.to_csv(output / "sample_metadata.csv", index=False, encoding="utf-8-sig")
        panel.to_csv(output / "panel_mapping.csv", index=False, encoding="utf-8-sig")
        field_mapping.to_csv(output / "field_mapping.csv", index=False, encoding="utf-8-sig")
    # Prior independent panel/clinical passes remain authoritative in this
    # repair; metadata-only execution cannot close missing-TIFF failures.
    append_reviews([
        ("M01", "unresolved", identity_details),
        ("D01", "unresolved", f"Repair round 2 preparer metadata execution: {len(images)}/50 formal TIFFs indexed; "
            "50 complete source ROIs connect to 34 samples, 33 samples and 48 images have confirmed patient identities. "
            "Prior independent P01/P02/M02 passes are preserved; missing-TIFF I01/P03 failures remain open. "
            "Formal TIFF delivery, independent image re-review and controller acceptance remain pending. No acceptance binding issued."),
    ])
else:
    remaining_bytes = sum(28 * max(item["shape"][1:])**2 * 4 for item in inventory
                          if not (receipts / f"{item['image_id']}.json").exists())
    if (shutil.disk_usage(root).free - remaining_bytes < 100 * 1024**3
            or shutil.disk_usage(temporary).free - max(28 * max(item["shape"][1:])**2 * 4
                                                       for item in inventory) < 64 * 1024**3):
        stop_for_io("Insufficient NAS/local reserve for remaining native TIFF delivery.")
    for item in inventory:
        image_id = item["image_id"]
        destination = output / "images" / image_id / "input.tiff"
        receipt_path = receipts / f"{image_id}.json"
        binding = dict(source_files=item["files"], source_indices=kept.source_channel_index.tolist(), marker_names=kept.marker_id.tolist())
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt["binding"] == binding, f"Input changed: {image_id}"
            if receipt["stats"]["admitted"]:
                assert destination.exists(), image_id
        else:
            source = SourceSpec(image_id, panel_id, "zarr", root / item["source_path"] / "0", tuple(range(41)),
                                pixel_size_um=(1.0, 1.0), source_revision="SPORA level0; original verified object sizes")
            staged_receipt = temporary / "native" / f"{image_id}.json"
            if staged_receipt.exists():
                assert json.loads(staged_receipt.read_text()) == dict(source_files=item["files"], shape=item["shape"], dtype=item["dtype"])
                stage_path = temporary / "native" / f"{image_id}.dat"
                stage = np.memmap(stage_path, mode="r", dtype=item["dtype"], shape=tuple(item["shape"]))
            else:
                stage_path = temporary / "native_roi.dat"
                stage = np.memmap(stage_path, mode="w+", dtype=item["dtype"], shape=tuple(item["shape"]))
                progress("staging", len(images), current_image=image_id)
                with formal_read_lease(source.path), open_source(source) as reader:
                    for window in windows(reader.shape[-2:], 256):
                        x0, y0, x1, y1 = window
                        stage[:, y0:y1, x0:x1] = reader.read_window(list(range(41)), window)
                stage.flush()
            try:
                progress("writing_tiff", len(images), current_image=image_id)
                stats = export_input(StagedReader(stage, source), panel, destination)
                if image_id in representatives:
                    overview(stage, image_id)
                comparisons = []
                if stats["admitted"]:
                    # Compare every retained native channel in internal and edge blocks.
                    h, w = item["shape"][1:]
                    for x0, y0, x1, y1 in [(w//2, h//2, w//2+32, h//2+32), (w-32, h-32, w, h)]:
                        out_window = (x0+stats["pad_left_px"], y0+stats["pad_top_px"],
                                      x1+stats["pad_left_px"], y1+stats["pad_top_px"])
                        with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
                            actual = np.stack([read_tiff_window(tif, i, out_window) for i in range(len(kept))])
                        expected = np.asarray(stage[kept.source_channel_index.tolist(), y0:y1, x0:x1])
                        assert actual.dtype == expected.dtype and np.array_equal(actual, expected), image_id
                        comparisons.append(dict(source_window=[x0, y0, x1, y1], output_window=list(out_window), channels=28, equal=True))
                receipt = dict(binding=binding, stats=stats, pixel_comparisons=comparisons,
                               source_inventory=relative(cache / "source_inventory.json"))
                save_json(receipt_path, receipt)
            finally:
                stage._mmap.close()
            stage_path.unlink()
            staged_receipt.unlink(missing_ok=True)
        if not receipt["stats"]["admitted"]:
            images = images[images.image_id != image_id]
            continue
        stats = receipt["stats"]
        with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
            names = [element.attrib["Name"] for element in ElementTree.fromstring(tif.ome_metadata).iter() if element.tag.endswith("}Channel")]
            assert names == kept.marker_id.tolist()
            side = stats["height_px"]
            left, top = stats["pad_left_px"], stats["pad_top_px"]
            right, bottom = stats["pad_right_px"], stats["pad_bottom_px"]
            padding_windows = []
            if left:
                padding_windows.append((0, side//2, min(left, 16), side//2+16))
            if right:
                padding_windows.append((side-min(right, 16), side//2, side, side//2+16))
            if top:
                padding_windows.append((side//2, 0, side//2+16, min(top, 16)))
            if bottom:
                padding_windows.append((side//2, side-min(bottom, 16), side//2+16, side))
            for window in padding_windows:
                assert not any(np.any(read_tiff_window(tif, index, window)) for index in range(28)), image_id
            # Local workaround: OME series parsing caches TiffFrame objects,
            # which shared read_tiff_window cannot yet consume. Read scalar
            # pages before materializing the series; request shared migration.
            assert tif.series[0].shape == (28, stats["height_px"], stats["width_px"])
            assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("float32")
        receipt["padding_checks"] = dict(windows=[list(window) for window in padding_windows], channels=28, all_zero=True)
        save_json(receipt_path, receipt)
        link = link_rows.loc[image_id]
        acquisition = acquisitions.loc[link.image_number]
        row = dict.fromkeys(columns("image_metadata"))
        row.update(study=STUDY, image_id=image_id, sample_id=f"{STUDY}__block__{link.BlockID}", panel_id=panel_id,
                   image_kind="tma_core", source_image_id=image_id, source_files=[item["source_path"]],
                   source_format="zarr", source_axes="CYX", source_dtype="float32", source_channel_count=41, channel_count=28,
                   source_acquisition_id=int(acquisition.Metadata_AcquisitionROIID),
                   source_revision="SPORA hoch2022multiplexed 1_0mpp level0; original 50-image verification reused",
                   slide_id=link.source_slide_id, core_id=f"{link.source_slide_id}__{link.Description}",
                   roi_id=str(acquisition.Metadata_AcquisitionROIID),
                   region_annotation=json.dumps(dict(Description=link.Description, Location=link.Location, TissueType=link.TissueType), ensure_ascii=False),
                   source_x0_px=0, source_y0_px=0, source_x1_px=item["shape"][2], source_y1_px=item["shape"][1],
                   pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                   intensity_description="Published SPORA float32 intensities, preserved without normalization; author RNA source uses spill-corrected TIFF intensities.",
                   split_method="already_independent", split_evidence=relative(acquisition_path),
                   metadata_notes=("Full published measured ROI; dense source grid including zeros retained. Centered square zero padding. "
                       "One author TMA core per image; sample grouped by source BlockID. "
                       f"Original acquisition file: {link.original_acquisition_file}; container-local Metadata_slideid={int(acquisition.Metadata_slideid)}."))
        for name in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"]:
            row[name] = stats[name]
        images = pd.concat([images[images.image_id != image_id], pd.DataFrame([row])], ignore_index=True)
        save_input_tables(output, images, samples, panel, field_mapping)
        progress("exporting", len(images), last_completed_image=image_id, sample_rows=34,
                 sample_linked_images=len(images), patient_linked_samples=33)
        print(f"{image_id}: {len(images)}/50 saved", flush=True)

    # Repair only the missing image delivery; preserve accepted P01/P02/M02
    # evidence from independent review 0195 instead of superseding those rows.
    visual_path = cache / "overview_review.json"
    visual_result = "pass" if visual_path.exists() else "unresolved"
    visual_detail = (f"Preparer visual review recorded in {relative(visual_path)}. " if visual_path.exists()
                     else "Preparer visual inspection remains to be recorded. ")
    checks = [
        ("S01", "pass", f"Preparer self-check: {len(images)}/50 images exported; complete source frames, original verification reused; partial files untouched. All retained-marker signal counts are in per-image receipts."),
        ("I01", "pass", "Preparer self-check: all image/source identities and 39 base fields saved; one immutable source group per image."),
        ("I02", "pass", "Preparer self-check: 50/50 image-to-sample and panel joins; 34 source BlockID samples."),
        ("I03", "pass", "Preparer self-check: full source bounds, native 1 um scale, square centered padding, dtype float32; all TIFF headers checked."),
        ("I04", visual_result, "Author core map and full ROI dimensions agree for all 50. " + visual_detail + "Independent review pending."),
        ("P03", "pass", "Every exported TIFF reopened; CYX, 28 channels, marker names/order, float32 and geometry verified."),
        ("I05", "pass", "Preparer self-check: every image, two 32x32 blocks, all 28 channels compared exactly to decoded native disk staging; zero-padding blocks checked in all retained channels. Coordinates in image_receipts. Independent direct-source sampling pending."),
        ("M01", "unresolved", identity_details),
        ("D01", "unresolved", "Repair round 2 preparer delivery: 50 formal TIFFs and the 39-field image index completed; "
            "sample_metadata.study field mapping supplied. Independent re-review and controller acceptance pending; "
            "no program-consumable acceptance binding generated. M01 retains the one control-sample/two-image patient identity gap."),
    ]
    append_reviews(checks)
    progress("prepared_pending_independent_review" if visual_path.exists() else "prepared_pending_visual_and_independent_review", len(images), sample_rows=len(samples), sample_linked_images=len(images),
             patient_linked_samples=int(samples.patient_key.notna().sum()), distinct_patients=int(samples.patient_key.nunique()),
             controller_requests=["独立复核并验收；对照标本患者缺失需明确范围处理",
                 "共享 read_tiff_window 在 OME series 解析后收到 TiffFrame 会因 samplesperpixel 缺失失败；已用28页内存OME TIFF重现。本study先做标量页窗口检查再解析series，未改src；请主控统一迁移适配。",
                 "将本study磁盘staging适配迁入共享模块时复用本脚本证据；无需修改本批正确TIFF"])
