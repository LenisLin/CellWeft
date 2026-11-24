"""Prepare the verified Karimi brain core TIFFs at published uint8 intensity.

Inputs: complete BrainData.zip, extracted core TIFFs, clinical workbook, ordered
Zenodo panel, per-image qualification, and the current S1 mapping.
Outputs: square CYX input.tiff per qualified core and five study input tables.
The shared exporter stages locally and leases each formal NAS publication.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import zipfile
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import openpyxl
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease, resource_lock
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables, validate_image_metadata


STUDY = "Karimi2023_IMC_BrainTumours"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / "raw/public" / STUDY / "zenodo_7884599"
ARCHIVE = RAW / "BrainData.zip"
EXTRACTED = RAW / "Brain_Data2/BRAIN_IMC_MaskTif"
CLINICAL = RAW / "Brain_Data2/Clinical Data GBM BrM.xlsx"
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0781_prepare_0")))
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
LOCAL = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
PREPARER = JOB.name


def relative(path: Path) -> str:
    if path.is_relative_to(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime"):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def columns(table: str) -> list[str]:
    return list(pd.read_csv(TEMPLATES / f"{table}.csv", nrows=0).columns)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(status: str, stage: str, images: pd.DataFrame, *, next_image: str | None,
             issue: str | None = None) -> None:
    linked = images.merge(samples[["sample_id", "patient_key"]], on="sample_id", how="left") if len(images) else images
    atomic_json(JOB / "progress.json", dict(
        study=STUDY, status=status, phase="prepare", repair_round=0, stage=stage,
        updated_at=datetime.now(timezone.utc).isoformat(),
        expected_images=len(qualified), exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        unique_patient_keys=int(samples.patient_key.nunique()),
        patient_linked_images=int(linked.patient_key.notna().sum()) if len(images) else 0,
        next_image=next_image, issue=issue,
    ))


def review_rows(images: pd.DataFrame, stage: str) -> None:
    if not len(images):
        return
    source = ";".join(map(relative, (ARCHIVE, CLINICAL, META / "per_image_verification.json")))
    output = relative(OUTPUT)
    count = len(images)
    rows = []
    assessments = [
        ("S01", "study", STUDY, "unresolved",
         f"Source proof has 393 readable named views of 389 physical cores; 392 exact panel mappings. Exported {count}/392; BrM_032C2 exact 20-plane order remains unresolved."),
        ("M01", "study", STUDY, "unresolved",
         f"Clinical workbook joins all 393 source image IDs. Exported {count} images connect to physical-core samples; full qualified set has 388 samples and 185 recorded patient keys. Four half-core A/B pairs share sample/core identities."),
        ("M02", "study", STUDY, "unresolved",
         "Source clinical columns are retained in sample metadata notes. Age has no verified event anchor; treatment and survival strings retain source semantics. Independent clinical reconstruction pending."),
        ("P01", "study", STUDY, "pass", "Two ordered 21-plane source panels match the qualified per-image proofs and current S1 labels."),
        ("P02", "study", STUDY, "pass", "Both panels retain 20 protein measurements and the specific DNA1 nuclear measurement in native order."),
        ("D01", "study", STUDY, "unresolved",
         f"Preparer handoff: {count}/392 qualified images exported; full sample table has {len(samples)} patient-linked samples. Independent review and controller acceptance pending."),
    ]
    for check, scope, record, result, detail in assessments:
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                         record_id=record, check_id=check, result=result,
                         source_evidence=source, output_evidence=output,
                         details=json.dumps(dict(assessment="preparer_self_check", stage=stage, text=detail))))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        prior = (pd.read_csv(review_path, keep_default_na=False)
                 if review_path.is_file() else pd.DataFrame(columns=columns("review_results")))
        prior = prior.loc[prior.preparer != PREPARER]
        merged = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True)
        temporary = review_path.with_suffix(".csv.partial")
        merged.to_csv(temporary, index=False, columns=columns("review_results"), encoding="utf-8-sig")
        os.replace(temporary, review_path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--waiting-io-reason", default=None)
parser.add_argument("--repair-half-cores", action="store_true",
                    help="Repair the four reviewed A/B core identities without rewriting TIFFs")
args = parser.parse_args()

# Resolve the already qualified source subset and its ordered panels.
proof = json.loads((META / "per_image_verification.json").read_text())["images"]
qualified = sorted((item for item in proof if item["channel_mapping_valid"]),
                   key=lambda item: item["archive_member"])
assert len(proof) == 393 and len(qualified) == 392
markers = pd.read_csv(META / "channel_markers.csv", keep_default_na=False)
panels = {}
for name in ("BrM", "Glioma"):
    source = markers.loc[markers.panel == name].sort_values("channel_index_1based")
    assert source.channel_index_1based.tolist() == list(range(1, 22))
    panels[name] = panel_from_s1(STUDY, f"karimi2023_{name}", source.marker.tolist(), S1)
    assert int(panels[name].keep_channel.sum()) == 21
panel = pd.concat(panels.values(), ignore_index=True)

# Keep the source's patient connection on NAS, without embedding identities here.
book = openpyxl.load_workbook(CLINICAL, read_only=True, data_only=True)
sheet = book.active
records = list(sheet.values)
book.close()
clinical_columns = list(records[0])
clinical = {str(row[0]): dict(zip(clinical_columns, row)) for row in records[1:]}
assert len(clinical) == 393
assert {Path(item["archive_member"]).stem for item in proof} == set(clinical)
# Replicate Cores=0.5 identifies two source views of one physical core.
half_views = {key: key[:-1] for key, row in clinical.items() if row["Replicate Cores"] == 0.5}
assert len(half_views) == 8 and len(set(half_views.values())) == 4
for core in set(half_views.values()):
    assert {key for key, base in half_views.items() if base == core} == {core + "A", core + "B"}
    assert all(clinical[core + "A"][key] == clinical[core + "B"][key]
               for key in clinical_columns if key != "Image ID")

sample_rows = []
seen_cores = set()
for item in qualified:
    image_id = Path(item["archive_member"]).stem
    core = half_views.get(image_id, image_id)
    if core in seen_cores:
        continue
    seen_cores.add(core)
    record = clinical[image_id]
    patient = str(record["Patient ID"])
    assert patient and patient != "None"
    region = str(record["Tissue Region"])
    sex = {"F": "female", "M": "male", "Unknown": "unknown"}[record["Sex"]]
    row = dict.fromkeys(columns("sample_metadata"), None)
    row.update(study=STUDY, sample_id=f"core_{core}", patient_key=patient,
               source_subject_id=patient, source_sample_id=core,
               sample_type="source-published tissue core",
               tissue_type="adjacent_normal" if region == "Adjacent Normal" else "tumor",
               anatomical_site="Brain", sex_reported=sex,
               primary_diagnosis=record["Tumor Type"],
               primary_site=(record["Primary Site of Brain Metastasis"]
                             if record["Tumor Type"] == "Brain Metastasis" else "Brain"),
               case_control_status="case",
               metadata_notes=json.dumps({key: value for key, value in record.items()
                                          if key not in {"Image ID", "Patient ID"}},
                                         ensure_ascii=False, default=str))
    sample_rows.append(row)
samples = pd.DataFrame(sample_rows)

field_rows = []
def field(source_file: Path, source_field: str, target_table: str,
          target_field: str, rule: str, notes: str) -> None:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(source_file),
               source_sheet_or_table="Sheet1" if source_file == CLINICAL else "source evidence",
               source_field=source_field, target_table=target_table,
               target_field=target_field, join_keys="Image ID = TIFF stem",
               transformation_rule=rule, notes=notes)
    field_rows.append(row)

field(CLINICAL, "Image ID;Patient ID", "sample_metadata", "sample_id;patient_key;source_subject_id",
      "one sample per physical core; merge A/B views only when Replicate Cores=0.5 and all other clinical fields agree; copy source patient key",
      "393 source views represent 389 physical cores; qualified 392 views connect to 388 samples and 185 recorded patient keys. Four reviewed A/B pairs share one sample/core each.")
field(CLINICAL, "Image ID;Tissue Region", "sample_metadata", "source_sample_id;sample_type;anatomical_site;case_control_status",
      "source_sample_id is Image ID except paired half-core A/B suffixes are removed; source-published tissue core; Brain; case",
      "Paired identities use Replicate Cores=0.5 and equality of every source field except Image ID. Per-view clinical notes retain the original 0.5 weight; adjacent-normal cores retain case cohort membership.")
field(CLINICAL, "Tumor Type;Primary Site of Brain Metastasis;Tissue Region;Sex", "sample_metadata",
      "primary_diagnosis;primary_site;tissue_type;sex_reported",
      "copy diagnosis/site, map adjacent normal and M/F/Unknown", "Only source-confirmed clinical categories are filled.")
field(CLINICAL, ";".join(map(str, clinical_columns[1:])), "sample_metadata", "metadata_notes",
      "store source reported fields as JSON without reinterpreting endpoint/time anchors",
      "Age, surgical stage, IDH/MGMT, treatment and endpoint values retain original wording; normalized timing is pending.")
field(META / "per_image_verification.json", "archive_member;shape;axes;dtype;panel;markers",
      "image_metadata", "source_image_id;source_axes;source_dtype;panel_id;source_channel_count",
      "join exact archive member and 21-plane ordered marker proof", "392 qualified; BrM_032C2 unresolved and excluded from output.")
field(META / "channel_markers.csv", "panel;channel_index_1based;marker", "panel_mapping",
      "source_name;source_channel_index", "convert one-based source order to zero-based",
      "Source published ordered labels checked against each qualified proof and current S1.")
field(META / "clinical_identity_review_20261023.json", "source_image_join_matches;unique_people;physical_core_note",
      "sample_metadata", "sample_id;patient_key",
      "reuse source identity review, exact workbook Image ID join and source Patient ID cardinality",
      "393/393 source Image IDs join; selected 392 views connect to 388 sample rows with patient keys (185 unique). "
      "Public-source search evidence is retained with the study preparation evidence: "
      "Zenodo 7884599 ordered panels, archive clinical workbook and paired MAT, official IMC_Brain README/root, "
      "and bounded superseded-release probe. Exact 20-plane exception map remains absent. Independent review 0819_review_0 reconciles 389 physical cores using four 0.5 A/B pairs. "
      "No author contact or new source search in this preparation run.")
field(CLINICAL, "Image ID;Replicate Cores", "image_metadata", "sample_id;core_id",
      "A/B views with Replicate Cores=0.5 and equal remaining clinical fields share suffix-stripped core_id and core_<core_id> sample_id; retain image_id and source member",
      "Four pairs confirmed by independent 0819_review_0; 392 images, 388 samples, 185 recorded patient keys. No image geometry or pixel change.")
fields = pd.DataFrame(field_rows)

# Metadata-only repair: replace obsolete sample keys rather than merge them back.
if args.repair_half_cores:
    JOB.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    repair_stage = LOCAL / "half_core_repair"
    repair_stage.mkdir(exist_ok=True)
    review_job = META / "source_metadata"
    evidence = json.loads((review_job / "half_core_source_identity.json").read_text())
    assert set(evidence["affected_image_ids"]) == set(half_views)
    names = ("images.parquet", "sample_metadata.csv", "field_mapping.csv", "review_results.csv")
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        versions = {name: (OUTPUT / name).stat().st_mtime_ns for name in names}
        original_images = pd.read_parquet(OUTPUT / "images.parquet")
        original_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
        original_fields = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False)
        reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
        prior_panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
        file_versions = {item["image_id"]: dict(size=(NAS / item["input_image_path"]).stat().st_size,
                         mtime_ns=(NAS / item["input_image_path"]).stat().st_mtime_ns)
                         for item in original_images.to_dict("records")}
    assert len(original_images) == 392
    for header in evidence["headers"]:
        assert file_versions[header["image_id"]] == {key: header["revision"][key] for key in ("size", "mtime_ns")}
    images = original_images.copy()
    affected = images.image_id.isin(half_views)
    for index in images.index[affected]:
        core = half_views[images.at[index, "image_id"]]
        images.at[index, "sample_id"] = f"core_{core}"
        images.at[index, "core_id"] = core
        images.at[index, "metadata_notes"] = "Half-core A/B views share physical core/sample; source Replicate Cores=0.5 and all other clinical fields agree. See 0819_review_0."
    # Preserve every existing non-identity value and unrelated sample row.
    obsolete = {f"core_{view}" for view in half_views}
    replacements = samples.loc[samples.sample_id.isin({f"core_{base}" for base in half_views.values()})]
    replaced_keys = obsolete | set(replacements.sample_id)
    samples = pd.concat([original_samples.loc[~original_samples.sample_id.isin(replaced_keys)], replacements], ignore_index=True)
    assert len(samples) == 388 and samples.sample_id.is_unique
    for _, replacement in replacements.iterrows():
        prior_key = replacement.sample_id if replacement.sample_id in set(original_samples.sample_id) else replacement.sample_id + "A"
        old = original_samples.loc[original_samples.sample_id == prior_key].iloc[0]
        for name in samples.columns.difference(["sample_id", "source_sample_id"]):
            assert ("" if pd.isna(replacement[name]) else replacement[name]) == old[name]
    # Change only the three identity/search mapping records; add the image rule.
    fields = original_fields.copy()
    for candidate in field_rows:
        match = (fields.source_file == candidate["source_file"]) & (fields.target_field == candidate["target_field"])
        if candidate["target_field"] in {"sample_id;patient_key;source_subject_id", "source_sample_id;sample_type;anatomical_site;case_control_status", "sample_id;patient_key"}:
            assert match.sum() == 1
            fields.loc[match, ["transformation_rule", "notes"]] = [candidate["transformation_rule"], candidate["notes"]]
    fields = pd.concat([fields, pd.DataFrame([field_rows[-1]])], ignore_index=True).drop_duplicates(
        subset=["study", "source_file", "source_field", "target_table", "target_field"], keep="last")
    images = validate_image_metadata(images, samples, prior_panel, fields)
    prior_progress = JOB / "progress.json"
    if prior_progress.is_file() and json.loads(prior_progress.read_text()).get("status") == "prepared":
        pd.testing.assert_frame_equal(images, original_images)
        pd.testing.assert_frame_equal(samples.fillna("").reset_index(drop=True), original_samples.reset_index(drop=True), check_dtype=False)
        pd.testing.assert_frame_equal(fields.reset_index(drop=True), original_fields.reset_index(drop=True), check_dtype=False)
        print(json.dumps(dict(status="prepared", resume="existing closed repair verified; no publication", images=392, samples=388)), flush=True)
        raise SystemExit(0)
    details = dict(assessment="preparer_self_check", phase="repair", science_repair_round=1,
                   image_ids=sorted(half_views), text="Four half-core pairs now share sample/core identity; 392 TIFF versions unchanged. 388 samples and 185 recorded patient keys. Independent metadata re-review and renewed D01 binding required.")
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY + ":repair_1",
                 check_id=check, result="unresolved", source_evidence=relative(CLINICAL),
                 output_evidence=relative(OUTPUT), details=json.dumps(details)) for check in ("I01", "I04", "M01", "D01")]
    reviews = pd.concat([reviews.loc[reviews.preparer != PREPARER], pd.DataFrame(rows)], ignore_index=True)
    images.to_parquet(repair_stage / "images.parquet", index=False)
    for name, table in (("sample_metadata.csv", samples), ("field_mapping.csv", fields), ("review_results.csv", reviews)):
        table.to_csv(repair_stage / name, index=False, encoding="utf-8-sig")
    estimated_bytes = sum((repair_stage / name).stat().st_size for name in names)
    checkpoint = dict(study=STUDY, phase="repair", science_repair_round=1, status="partial",
                      stage="metadata_staged", local_stage=str(repair_stage), expected_images=392,
                      exported_images=392, sample_rows=388, sample_linked_images=392,
                      patient_linked_samples=388, unique_patient_keys=185,
                      estimated_publication_bytes=estimated_bytes, next_image=None)
    atomic_json(JOB / "progress.json", checkpoint)
    if shutil.disk_usage(NAS).free < 100 * 2**30 + estimated_bytes or shutil.disk_usage(LOCAL).free < 64 * 2**30:
        checkpoint.update(status="waiting_io", issue="Insufficient reserved capacity; closed metadata staging retained")
        atomic_json(JOB / "progress.json", checkpoint)
        raise SystemExit(75)
    def lease_timeout(signum: int, frame: object) -> None:
        raise TimeoutError("Shared NAS publication lease unavailable within 45 seconds")
    signal.signal(signal.SIGALRM, lease_timeout)
    try:
        signal.alarm(45)
        with formal_publish_lease(OUTPUT, directory=OUTPUT):
            signal.alarm(0)
            assert versions == {name: (OUTPUT / name).stat().st_mtime_ns for name in names}
            for name in names:
                temporary = OUTPUT / (name + ".partial")
                with (repair_stage / name).open("rb") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, OUTPUT / name)
            descriptor = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError as error:
        signal.alarm(0)
        checkpoint.update(status="waiting_io", issue=str(error))
        atomic_json(JOB / "progress.json", checkpoint)
        raise SystemExit(75) from error
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        saved = pd.read_parquet(OUTPUT / "images.parquet")
        saved_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
        assert len(saved) == 392 and len(saved_samples) == 388
        assert saved.merge(saved_samples, on=["study", "sample_id"], validate="many_to_one").shape[0] == 392
        assert saved_samples.patient_key.ne("").all() and saved_samples.patient_key.nunique() == 185
        pd.testing.assert_frame_equal(saved.loc[~affected].reset_index(drop=True), original_images.loc[~affected].reset_index(drop=True))
        for item in saved.to_dict("records"):
            stat = (NAS / item["input_image_path"]).stat()
            assert file_versions[item["image_id"]] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    atomic_json(JOB / "repair_evidence.json", dict(**details, image_file_versions=file_versions,
                clinical_pair_fields_equal=True, unaffected_image_records_equal=True,
                unchanged_panel=True, publication_bytes=estimated_bytes))
    checkpoint.update(status="prepared", stage="repair_closed_awaiting_independent_review", issue=None)
    atomic_json(JOB / "progress.json", checkpoint)
    print(json.dumps(checkpoint), flush=True)
    raise SystemExit(0)

images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file()
          else pd.DataFrame(columns=columns("image_metadata")))
completed = set(images.image_id) if len(images) else set()
remaining = [item for item in qualified if Path(item["archive_member"]).stem not in completed]
estimated_bytes = sum(21 * max(item["shape"][1:]) ** 2 for item in remaining)
nas_free = shutil.disk_usage(NAS).free
local_free = shutil.disk_usage(LOCAL.parent).free
if nas_free < 100 * 2**30 + estimated_bytes or local_free < 64 * 2**30 + 512 * 2**20:
    progress("waiting_io", "capacity", images,
             next_image=Path(remaining[0]["archive_member"]).stem if remaining else None,
             issue=f"capacity: estimated_output={estimated_bytes}; nas_free={nas_free}; local_free={local_free}")
    raise SystemExit(75)

LOCAL.mkdir(parents=True, exist_ok=True)
JOB.mkdir(parents=True, exist_ok=True)
if args.limit is not None:
    remaining = remaining[:args.limit]
progress("partial", "exporting", images,
         next_image=Path(remaining[0]["archive_member"]).stem if remaining else None)

# Publish closed images and persist the typed index in small resumable batches.
pending_rows = []
for position, item in enumerate(remaining, start=1):
    member = item["archive_member"]
    image_id = Path(member).stem
    source_path = RAW / member
    from_archive = not source_path.is_file()
    if from_archive:
        source_path = LOCAL / f"{image_id}.tif"
        if not source_path.is_file():
            temporary = source_path.with_suffix(".tif.partial")
            with formal_read_lease(ARCHIVE), zipfile.ZipFile(ARCHIVE) as archive:
                info = archive.getinfo(member)
                with archive.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            assert temporary.stat().st_size == info.file_size
            os.replace(temporary, source_path)
    source = SourceSpec(image_id=image_id, panel_id=f"karimi2023_{item['panel']}",
                        kind="tiff", path=source_path, channels=tuple(range(21)),
                        source_revision="Zenodo 7884599 BrainData.zip",
                        intensity_description="Source-published masked IMC core uint8 intensity; no CellWeft normalization")
    destination = OUTPUT / "images" / image_id / "input.tiff"
    try:
        with open_source(source) as reader:
            assert reader.shape == tuple(item["shape"]) and reader.dtype == np.dtype("uint8")
            if destination.is_file():
                # Recover a closed publication whose previous index batch did not commit.
                _, h, w = reader.shape
                side = max(h, w)
                left, top = (side - w) // 2, (side - h) // 2
                with formal_read_lease(destination), resource_lock(OUTPUT, shared=True), \
                        resource_lock(destination.parent, name=".input.lock", shared=True), \
                        tifffile.TiffFile(destination) as saved:
                    assert saved.series[0].shape == (21, side, side)
                    assert saved.series[0].dtype == np.dtype("uint8")
                    names = [node.attrib["Name"] for node in ET.fromstring(saved.ome_metadata)
                             .findall(".//{*}Channel")]
                    expected = panels[item["panel"]].sort_values("output_channel_index").marker_id.tolist()
                    assert names == expected
                    signal = False
                    for channel in range(21):
                        original = reader.read_window([channel], (0, 0, w, h))[0]
                        padded = np.pad(original, ((top, side - h - top), (left, side - w - left)))
                        assert np.array_equal(padded, saved.series[0].pages[channel].asarray())
                        if panels[item["panel"]].iloc[channel].role == "downstream":
                            signal = signal or bool(original.any())
                    assert signal
                result = dict(admitted=True, channel_count=21, height_px=side,
                              pad_left_px=left, pad_top_px=top,
                              pad_right_px=side - w - left, pad_bottom_px=side - h - top)
            else:
                result = export_input(reader, panels[item["panel"]], destination, tile_size=512)
            assert result["admitted"] and result["channel_count"] == 21
            side = result["height_px"]
            left, top = result["pad_left_px"], result["pad_top_px"]
            _, h, w = reader.shape
            boxes = [(0, 0, min(32, w), min(32, h)),
                     (max(0, w // 2 - 16), max(0, h // 2 - 16),
                      min(w, w // 2 + 16), min(h, h // 2 + 16))]
            with formal_read_lease(destination), tifffile.TiffFile(destination) as saved:
                assert saved.series[0].shape == (21, side, side)
                assert saved.series[0].dtype == np.dtype("uint8")
                names = [node.attrib["Name"] for node in ET.fromstring(saved.ome_metadata)
                         .findall(".//{*}Channel")]
                assert names == panels[item["panel"]].sort_values("output_channel_index").marker_id.tolist()
                for box in boxes:
                    x0, y0, x1, y1 = box
                    translated = (x0 + left, y0 + top, x1 + left, y1 + top)
                    for channel in range(21):
                        original = reader.read_window([channel], box)[0]
                        output = read_tiff_window(saved, channel, translated)
                        assert np.array_equal(original, output)
        row = dict.fromkeys(columns("image_metadata"), None)
        core = half_views.get(image_id, image_id)
        row.update(study=STUDY, image_id=image_id, sample_id=f"core_{core}",
                   panel_id=source.panel_id, image_kind="tma_core", source_image_id=member,
                   source_files=[relative(ARCHIVE)], source_format="tiff",
                   source_axes=item["axes"], source_dtype="uint8", source_revision=source.source_revision,
                   core_id=core, region_annotation=clinical[image_id]["Tissue Region"],
                   source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                   pad_left_px=left, pad_top_px=top,
                   pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
                   height_px=side, width_px=side, source_channel_count=21, channel_count=21,
                   input_image_path=relative(destination), valid_pixels_path=None,
                   intensity_description=source.intensity_description,
                   split_method="already_independent",
                   split_evidence=relative(META / "per_image_verification.json") + "#images.archive_member",
                   metadata_notes="393 source views represent 389 physical cores; four reviewed A/B half-core pairs share physical core/sample identity.")
        pending_rows.append(row)
        if len(pending_rows) == 8 or position == len(remaining):
            updated = pd.concat([images, pd.DataFrame(pending_rows)], ignore_index=True)
            save_input_tables(OUTPUT, updated, samples, panel, fields)
            images = updated
            completed.update(pending["image_id"] for pending in pending_rows)
            pending_rows.clear()
            next_image = next((Path(candidate["archive_member"]).stem for candidate in qualified
                               if Path(candidate["archive_member"]).stem not in completed), None)
            progress("partial", "exporting", images, next_image=next_image)
        if from_archive:
            source_path.unlink()
    except OSError as error:
        progress("waiting_io", "source_or_publication_io", images,
                 next_image=pending_rows[0]["image_id"] if pending_rows else image_id,
                 issue=f"{type(error).__name__}: {error}; closed publication checkpoint remains at {LOCAL.parent / 'publish_checkpoints' / metadata_source_relative(destination, '/')}")
        raise SystemExit(75) from error

stage = ("prepared_awaiting_independent_review" if len(images) == len(qualified)
         else "waiting_io" if args.waiting_io_reason else "partial_export")
review_rows(images, stage)
next_image = next((Path(item["archive_member"]).stem for item in qualified
                   if Path(item["archive_member"]).stem not in completed), None)
progress("prepared" if next_image is None else "waiting_io" if args.waiting_io_reason else "partial",
         stage, images, next_image=next_image, issue=args.waiting_io_reason)
print(json.dumps(dict(study=STUDY, exported_images=len(images), expected_images=len(qualified),
                      sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
                      next_image=next_image)), flush=True)
