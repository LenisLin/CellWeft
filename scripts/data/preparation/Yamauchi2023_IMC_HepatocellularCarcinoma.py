"""Export verified Yamauchi HCC MCD acquisitions as native study inputs."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import os
from pathlib import Path
import re
import shutil
import signal
import csv
import io
import tempfile
from collections import Counter
from datetime import datetime, timezone
from zipfile import ZipFile
from xml.etree import ElementTree

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import pandas as pd
import numpy as np
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Yamauchi2023_IMC_HepatocellularCarcinoma"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
ARCHIVE = ROOT / "raw/public" / STUDY / "public_original/Images.zip"
OUTPUT = ROOT / "processed" / STUDY
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0650_prepare_0")))
LOCAL = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY / "members"
PROOF = META / "native_verification.json"
SUPPLEMENT = ROOT / "raw/public" / STUDY / "metadata/Datasets_and_codes.zip"
PANEL_ID = STUDY + "__native_28"
SOURCE_REVISION = "figshare_22215760_v3_Images.zip"
PUBLIC_METHODS = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10600115/"
QC_EXCLUSIONS = {
    7: "Author reports rare-earth PD-L1 antibody did not work in this setting",
    9: "Author reports rare-earth CD11c antibody did not work in this setting",
    22: "Author reports rare-earth PD-L1 antibody did not work in this setting",
}


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def image_id_for(item: dict, description_counts: Counter[str]) -> str:
    description = item["description"]
    if description_counts[description] == 1:
        return description
    member = Path(item["source_member_name"]).stem
    return f"{member}_ROI_{int(item['source_acquisition_id']):03d}"


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, completed: int, next_image: str | None, **details: object) -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    payload = dict(study=STUDY, stage=stage, expected_images=121,
                   completed_images=completed, next_image=next_image,
                   updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    pending = JOB / "progress.json.partial"
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(pending, JOB / "progress.json")


def field(source: Path, source_field: str, table: str, target: str, rule: str,
          join: str, notes: str = "") -> dict[str, str]:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table="",
                source_field=source_field, target_table=table, target_field=target,
                join_keys=join, value_mapping="{}", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


def local_member(archive: ZipFile, name: str, expected: int, source_size: int) -> Path:
    info = archive.getinfo(name)
    if info.file_size != expected:
        raise ValueError(f"MCD member size differs from native proof: {name}")
    target = LOCAL / Path(name).name
    marker = target.with_suffix(".complete.json")
    identity = dict(archive_size=source_size, member=name, size=expected, crc=info.CRC)
    if target.is_file() and marker.is_file() and target.stat().st_size == expected:
        if json.loads(marker.read_text()) == identity:
            return target
    LOCAL.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".mcd.partial")
    with formal_read_lease(ARCHIVE, data_root=ROOT):
        with archive.open(name) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if partial.stat().st_size != expected:
        raise ValueError(f"Extracted MCD size differs from proof: {name}")
    os.replace(partial, target)
    marker_partial = marker.with_suffix(".json.partial")
    marker_partial.write_text(json.dumps(identity, sort_keys=True))
    os.replace(marker_partial, marker)
    return target


def publish_preparer_review(completed: int, *, waiting: bool) -> None:
    review_path = OUTPUT / "review_results.csv"
    reviews = pd.DataFrame([
        dict(study=STUDY, preparer=JOB.name, reviewer="", scope="sample", record_id=STUDY,
             check_id="M01", result="unresolved", source_evidence=relative(PROOF),
             output_evidence=relative(OUTPUT / "sample_metadata.csv"),
             details=f"Preparer trace: {completed}/121 image-to-sample links; 30 biopsy-member samples "
                     "link to 15 public FL_ID subjects. Clinical table covers 28/30 samples; one "
                     "conflicting RECIST value remains blank. Independent review pending."),
        dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study", record_id=STUDY,
             check_id="D01", result="unresolved", source_evidence=relative(PROOF),
             output_evidence=relative(OUTPUT / "images.parquet"),
             details=("Preparer export paused for shared I/O; independent review pending." if waiting else
                      "Preparer export complete; independent review and input binding pending.")),
    ], columns=columns("review_results"))
    with formal_read_lease(review_path, data_root=ROOT):
        prior = (pd.read_csv(review_path, keep_default_na=False) if review_path.is_file() else
                 pd.DataFrame(columns=reviews.columns))
    replace = (prior.preparer.eq(JOB.name) & prior.reviewer.eq("") &
               prior.check_id.isin(["M01", "D01"]))
    reviews = pd.concat([prior.loc[~replace], reviews], ignore_index=True)
    with formal_publish_lease(review_path, directory=OUTPUT, data_root=ROOT):
        temporary = review_path.with_suffix(".csv.partial")
        reviews.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, review_path)


def recover_published(reader: object, panel: pd.DataFrame, destination: Path) -> dict:
    """Recover an image published just before an interrupted table update."""
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    _, height, width = reader.shape
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    valid_path = destination.with_name("valid_pixels.tiff")
    with formal_read_lease(destination, data_root=ROOT):
        with tifffile.TiffFile(destination) as tif:
            names = [node.attrib["Name"] for node in ElementTree.fromstring(tif.ome_metadata).iter()
                     if node.tag.endswith("}Channel")]
            if (len(tif.pages) != len(kept) or names != kept.marker_id.tolist() or
                    any(page.shape != (side, side) or page.dtype != np.dtype("float32")
                        for page in tif.pages)):
                raise ValueError(f"Unindexed published image header differs: {destination}")
            for x0, y0 in ((0, 0), (max(0, width // 2 - 8), max(0, height // 2 - 8))):
                x1, y1 = min(width, x0 + 16), min(height, y0 + 16)
                valid = reader.valid_window((x0, y0, x1, y1))
                for output, source_index in enumerate(kept.source_channel_index):
                    expected = np.where(valid, reader.read_window([int(source_index)],
                                          (x0, y0, x1, y1))[0], 0)
                    observed = tif.pages[output].asarray()[top+y0:top+y1, left+x0:left+x1]
                    if not np.array_equal(observed, expected):
                        raise ValueError(f"Unindexed published image pixel mismatch: {destination}")
        if valid_path.is_file():
            with tifffile.TiffFile(valid_path) as tif:
                if (len(tif.pages) != 1 or tif.pages[0].shape != (side, side) or
                        tif.pages[0].dtype != np.dtype("uint8")):
                    raise ValueError(f"Unindexed validity mask differs: {valid_path}")
    return dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                source_channel_count=reader.shape[0], channel_count=len(kept),
                admitted=True, valid_pixels=valid_path.name if valid_path.is_file() else None)


def record_waiting_io() -> None:
    with formal_read_lease(OUTPUT, data_root=ROOT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False)
        proof = json.loads(PROOF.read_text())
    completed = len(images)
    records = [item["native_proof"] for item in proof["records"]]
    counts = Counter(item["description"] for item in records)
    remaining = sorted(image_id_for(item, counts) for item in records
                       if image_id_for(item, counts) not in set(images.image_id))
    next_image = remaining[0] if remaining else None
    local_checkpoint = (LOCAL.parent.parent / "publish_checkpoints" /
                        metadata_source_relative((OUTPUT / 'images' / next_image / 'input.tiff').resolve(), '/')) if next_image else None
    checkpoint("waiting_io", completed, next_image, reason="shared NAS write queue",
               sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
               patient_linked_samples=int(samples.patient_key.astype(bool).sum()),
               local_checkpoint=str(local_checkpoint) if local_checkpoint and local_checkpoint.exists() else None,
               next_step="resume script from next unindexed acquisition")


def repair_stale_patient_rule() -> None:
    """Remove the superseded patient rule retained by the table merge."""
    path = OUTPUT / "field_mapping.csv"
    with formal_read_lease(OUTPUT, data_root=ROOT):
        with resource_lock(OUTPUT, shared=True):
            previous = path.read_bytes()
            fields = pd.read_csv(io.BytesIO(previous), dtype=str, keep_default_na=False)
            images = pd.read_parquet(OUTPUT / "images.parquet")
            samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str,
                                  keep_default_na=False)
    obsolete = (fields.source_field.eq("qualification.people") &
                fields.target_table.eq("sample_metadata") &
                fields.target_field.eq("patient_key"))
    current = (fields.source_field.str.contains("FL_ID", regex=False) &
               fields.target_table.eq("sample_metadata") &
               fields.target_field.str.contains("patient_key", regex=False))
    if obsolete.sum() != 1 or current.sum() != 1:
        raise ValueError("Expected one obsolete and one current patient mapping rule")
    if (len(images) != 121 or len(samples) != 30 or
            not images.sample_id.isin(samples.sample_id).all() or
            not samples.patient_key.astype(bool).all()):
        raise ValueError("Published image, sample, or patient coverage changed")

    staging = LOCAL.parent / "table_repairs"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yamauchi-fields-", dir=staging) as temporary:
        staged = Path(temporary) / "field_mapping.csv"
        fields.loc[~obsolete].to_csv(staged, index=False, encoding="utf-8-sig")
        with formal_publish_lease(path, directory=OUTPUT, data_root=ROOT):
            if path.read_bytes() != previous:
                raise RuntimeError("Field mapping changed during repair; rerun against current table")
            pending = path.with_suffix(".csv.partial")
            shutil.copyfile(staged, pending)
            os.replace(pending, path)
    checkpoint("repair_applied", len(images), None, repair_round=1,
               sample_rows=len(samples), sample_linked_images=len(images),
               patient_linked_samples=int(samples.patient_key.astype(bool).sum()),
               repaired_field="obsolete qualification.people patient_key rule",
               next_step="independent M01 recheck and controller P02 ruling before D01")


def repair_antibody_qc() -> None:
    """Remove author-failed antibody measurements from verified native inputs."""
    def write_wait_expired(_signum: int, _frame: object) -> None:
        raise TimeoutError("shared NAS write lease unavailable")

    with formal_read_lease(OUTPUT, data_root=ROOT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        old_panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
        fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
    if len(images) != 121 or len(samples) != 30 or len(old_panel) != 28:
        raise ValueError("Published Yamauchi scope changed before QC repair")
    names = old_panel.sort_values("source_channel_index").source_name.tolist()
    if (names[7] != "PD-L1-28-8" or names[9] != "CD11c-EP1347Y" or
            names[22] != "PD-L1-poly"):
        raise ValueError("QC antibody source positions changed")
    with formal_read_lease(S1, data_root=ROOT):
        panel = panel_from_s1(STUDY, PANEL_ID, names, S1,
                              downstream_source_indices=set(range(28)) - {7, 9, 22},
                              drop_indices=QC_EXCLUSIONS)
    old_kept = old_panel.loc[old_panel.keep_channel].sort_values("output_channel_index")
    new_kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    old_output_by_source = dict(zip(old_kept.source_channel_index,
                                    old_kept.output_channel_index, strict=True))
    positions = [int(old_output_by_source[source]) for source in new_kept.source_channel_index]
    labels = new_kept.marker_id.tolist()
    if len(positions) != 21 or len(set(positions)) != 21:
        raise ValueError("Unexpected QC-retained panel width")
    qc_rule = fields.source_field.eq("Study;Original marker") & fields.target_table.eq("panel_mapping")
    if qc_rule.sum() != 1:
        raise ValueError("Expected one S1 panel mapping rule")
    fields.loc[qc_rule, "transformation_rule"] = (
        "Join exact study-specific S1 labels; exclude both PD-L1 measurements and CD11c "
        "from downstream and saved pixels because the paper reports their antibodies "
        "did not work in this experiment. Preserve all 28 source rows and S1 identities."
    )
    fields.loc[qc_rule, "notes"] = PUBLIC_METHODS + " (Discussion, limitations; antibody QC)"

    staging = LOCAL.parent / "antibody_qc_repair"
    staging.mkdir(parents=True, exist_ok=True)
    repaired_ids: list[str] = []
    for index, row in images.iterrows():
        image_id = row.image_id
        destination = ROOT / row.input_image_path
        stage = staging / f"{image_id}.tiff"
        next_count = int((images.channel_count == 21).sum())
        checkpoint("repair2_exporting", next_count, image_id, repair_round=2,
                   staged_file=str(stage), next_step="repack and publish this image")
        with formal_read_lease(destination, data_root=ROOT):
            with resource_lock(destination.parent, shared=True, name=".input.lock"):
                source_stat = destination.stat()
                with tifffile.TiffFile(destination) as tif:
                    if len(tif.pages) == 21:
                        if (any(page.shape != (int(row.height_px), int(row.width_px)) or
                                page.dtype != np.dtype("float32") for page in tif.pages) or
                                [channel.get("Name") for channel in
                                 ElementTree.fromstring(tif.ome_metadata).iter()
                                 if channel.tag.endswith("}Channel")] != labels):
                            raise ValueError(f"Already repaired TIFF differs: {image_id}")
                        images.at[index, "channel_count"] = 21
                        repaired_ids.append(image_id)
                        continue
                    if len(tif.pages) != 23 or int(row.channel_count) != 23:
                        raise ValueError(f"Source TIFF width differs: {image_id}")
                    values = np.stack([tif.pages[position].asarray() for position in positions])
        if (shutil.disk_usage(ROOT).free < 100 * 1024**3 + 2 * values.nbytes or
                shutil.disk_usage("/tmp").free < 64 * 1024**3 + 2 * values.nbytes):
            checkpoint("waiting_io", next_count, image_id, repair_round=2,
                       reason="NAS 100 GiB or local 64 GiB reserve", staged_file=str(stage))
            return
        reuse_stage = False
        if stage.is_file():
            with tifffile.TiffFile(stage) as tif:
                reuse_stage = (len(tif.pages) == 21 and
                               [channel.get("Name") for channel in
                                ElementTree.fromstring(tif.ome_metadata).iter()
                                if channel.tag.endswith("}Channel")] == labels and
                               all(np.array_equal(page.asarray(), values[position])
                                   for position, page in enumerate(tif.pages)))
        if not reuse_stage:
            tifffile.imwrite(stage, values, bigtiff=True, ome=True, tile=(256, 256),
                             compression="deflate", photometric="minisblack",
                             metadata={"axes": "CYX", "Channel": {"Name": labels},
                                       "PhysicalSizeX": 1.0, "PhysicalSizeXUnit": "µm",
                                       "PhysicalSizeY": 1.0, "PhysicalSizeYUnit": "µm"},
                             maxworkers=2)
        del values
        with tifffile.TiffFile(stage) as tif:
            if (len(tif.pages) != 21 or tif.pages[0].shape !=
                    (int(row.height_px), int(row.width_px)) or
                    tif.pages[0].dtype != np.dtype("float32")):
                raise ValueError(f"Staged TIFF header differs: {image_id}")
        previous_handler = signal.signal(signal.SIGALRM, write_wait_expired)
        signal.alarm(120)
        try:
            with formal_publish_lease(destination, directory=destination.parent,
                                      lock_name=".input.lock", data_root=ROOT):
                signal.alarm(0)
                if (destination.stat().st_size != source_stat.st_size or
                        destination.stat().st_mtime_ns != source_stat.st_mtime_ns):
                    raise RuntimeError(f"Input changed during QC repair: {image_id}")
                partial = destination.with_name(destination.name + ".partial")
                shutil.copyfile(stage, partial)
                os.replace(partial, destination)
                with tifffile.TiffFile(destination) as tif:
                    if len(tif.pages) != 21 or tif.pages[0].dtype != np.dtype("float32"):
                        raise ValueError(f"Published QC TIFF differs: {image_id}")
        except TimeoutError:
            checkpoint("waiting_io", next_count, image_id, repair_round=2,
                       reason="shared NAS write lease unavailable for 120 seconds",
                       staged_file=str(stage), repaired_image_ids=repaired_ids,
                       next_step="resume QC repair at this staged image")
            return
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        stage.unlink()
        images.at[index, "channel_count"] = 21
        repaired_ids.append(image_id)
        next_image = images.iloc[index + 1].image_id if index + 1 < len(images) else None
        checkpoint("repair2_published", next_count + 1, next_image, repair_round=2,
                   repaired_image_ids=repaired_ids,
                   next_step="resume at next 23-channel image")
        print(f"QC repacked {next_count + 1}/121 {image_id}", flush=True)

    previous_handler = signal.signal(signal.SIGALRM, write_wait_expired)
    signal.alarm(120)
    try:
        save_input_tables(OUTPUT, images, samples, panel, fields, data_root=ROOT)
    except TimeoutError:
        checkpoint("waiting_io", 121, None, repair_round=2,
                   reason="shared NAS table publication lease unavailable for 120 seconds",
                   repaired_image_ids=repaired_ids,
                   next_step="resume QC repair to publish panel, image and field tables")
        return
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    checkpoint("repair2_prepared", 121, None, repair_round=2, sample_rows=30,
               sample_linked_images=121, patient_linked_samples=30,
               next_step="independent P02/P03/I05 recheck and D01 input binding")


def public_subject_tables() -> tuple[set[str], dict[str, dict[str, str]], dict[str, list[str]]]:
    """Read public subject keys and clinical annotations without changing source files."""
    with formal_read_lease(SUPPLEMENT, data_root=ROOT):
        with ZipFile(SUPPLEMENT) as archive:
            def rows(suffix: str) -> list[list[str]]:
                name = next(name for name in archive.namelist()
                            if name.startswith("Datasets and codes/") and name.endswith(suffix))
                return list(csv.reader(io.StringIO(archive.read(name).decode("utf-8-sig")),
                                       delimiter="\t"))

            blood = rows("Dataset/Blood/Luminex_Results.txt")
            annotation = rows("Code/Fig1/Fig1b/Annotation_Fig1b.txt")
            clinical = rows("Dataset/Clinical Data/Patient_Suvival&Response.txt")
    if blood[0][0] != "FL_ID" or annotation[0] != [
        "FL_ID", "PFS", "censor", "mRECIST", "RECIST", "Sex", "DNA_input"
    ] or clinical[0][:8] != [
        "FL_ID", "Group", "Subtype", "Subtype", "Etiology", "PFS",
        "Response_mRECIST", "Response_RECIST"
    ]:
        raise ValueError("Unexpected public subject table columns")
    blood_ids = {row[0] for row in blood[1:]}
    annotation_rows = {row[0]: dict(zip(annotation[0], row)) for row in annotation[1:]}
    clinical_rows = {row[0]: row for row in clinical[1:]}
    if len(annotation_rows) != len(annotation) - 1 or len(clinical_rows) != len(clinical) - 1:
        raise ValueError("Public FL_ID keys are not unique")
    return blood_ids, annotation_rows, clinical_rows


def migrate_duplicate_descriptions(existing: dict[str, dict], records: list[dict],
                                   counts: Counter[str], samples: pd.DataFrame,
                                   panel: pd.DataFrame, fields: pd.DataFrame) -> None:
    """Give already published duplicate descriptions their MCD-qualified identities."""
    migrated = 0
    for item in records:
        old_id = item["description"]
        new_id = image_id_for(item, counts)
        member = Path(item["source_member_name"]).stem
        if (new_id == old_id or old_id not in existing or
                existing[old_id]["sample_id"] != member):
            continue
        old_dir = OUTPUT / "images" / old_id
        new_dir = OUTPUT / "images" / new_id
        with formal_publish_lease(new_dir / "input.tiff", directory=OUTPUT, data_root=ROOT):
            if old_dir.is_dir():
                if new_dir.exists():
                    raise ValueError(f"Duplicate image migration destination exists: {new_id}")
                with resource_lock(old_dir, name=".input.lock"):
                    old_dir.rename(new_dir)
            if not (new_dir / "input.tiff").is_file():
                raise ValueError(f"Duplicate image migration source missing: {old_id}")
        row = existing.pop(old_id)
        row["image_id"] = new_id
        row["input_image_path"] = relative(new_dir / "input.tiff")
        if row["valid_pixels_path"] is not None and not pd.isna(row["valid_pixels_path"]):
            row["valid_pixels_path"] = relative(new_dir / "valid_pixels.tiff")
        row["metadata_notes"] = (row["metadata_notes"] or "") + (
            " Duplicate source description; image_id uses MCD member and acquisition.")
        existing[new_id] = row
        migrated += 1
    if migrated:
        save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples,
                          panel, fields, data_root=ROOT)
        checkpoint("identity_migrated", len(existing), None, migrated_images=migrated,
                   next_step="resume remaining MCD acquisitions")


def main() -> None:
    with formal_read_lease(META, data_root=ROOT):
        proof = json.loads(PROOF.read_text())
    records = [item["native_proof"] for item in proof["records"]]
    description_counts = Counter(item["description"] for item in records)
    if len({image_id_for(item, description_counts) for item in records}) != len(records):
        raise ValueError("MCD-qualified acquisition IDs are not unique")
    if not proof["scope_complete"] or len(records) != 121 or not all(
        item["complete"] and item["counted"] for item in records
    ):
        raise ValueError("Native proof does not cover 121 counted acquisitions")
    if (OUTPUT / "images.parquet").is_file():
        with formal_read_lease(OUTPUT, data_root=ROOT):
            completed = pd.read_parquet(OUTPUT / "images.parquet")
        if len(completed) == 121:
            repair_antibody_qc()
            return
    names = records[0]["channels"]
    if len(names) != 28 or any(item["channels"] != names for item in records):
        raise ValueError("Source acquisitions do not share the verified 28-channel order")
    archive_size = ARCHIVE.stat().st_size
    if archive_size != records[0]["source_file_stat"]["size"]:
        raise ValueError("Source archive size differs from native proof")
    with formal_read_lease(S1, data_root=ROOT):
        # The paper reports both PD-L1 measurements and CD11c failed antibody QC.
        panel = panel_from_s1(STUDY, PANEL_ID, names, S1,
                              downstream_source_indices=set(range(28)) - {7, 9, 22},
                              drop_indices=QC_EXCLUSIONS)
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    if len(kept) != 21:
        raise ValueError(f"Unexpected retained panel width: {len(kept)}")

    groups: dict[str, list[dict]] = {}
    for item in records:
        groups.setdefault(item["source_member_name"], []).append(item)
    if len(groups) != 30:
        raise ValueError("Unexpected number of verified MCD biopsy members")
    blood_ids, annotation_rows, clinical_rows = public_subject_tables()
    samples = []
    for member in sorted(groups):
        stem = Path(member).stem
        match = re.fullmatch(r"FL\d+_(?:T|POST|T-sai-|POST_sai)", stem)
        if not match:
            raise ValueError(f"Unrecognized source specimen label: {stem}")
        subject = stem.split("_", 1)[0]
        source_timepoint = stem.split("_", 1)[1]
        member_phase = "POST" if source_timepoint.startswith("POST") else "T"
        described_phases = {phase.group(1) for item in groups[member]
                            if (phase := re.match(r"FL-?\d+_(T|POST)", item["description"]))}
        timepoint_conflict = any(phase != member_phase for phase in described_phases)
        if subject not in blood_ids or subject not in annotation_rows:
            raise ValueError(f"Public FL_ID evidence missing for source member: {stem}")
        annotation = annotation_rows[subject]
        clinical = clinical_rows.get(subject)
        if annotation["Sex"] not in {"Male", "Female"}:
            raise ValueError(f"Unexpected public sex category for {stem}")
        response_agrees = clinical is None or clinical[7] == annotation["RECIST"]
        response = annotation["RECIST"] if response_agrees else ""
        if response and response not in {"CR", "PR", "SD", "PD", "NE"}:
            raise ValueError(f"Unexpected public RECIST category for {stem}")
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=stem, source_sample_id=stem,
                   patient_key=f"{STUDY}__{subject}", source_subject_id=subject,
                   sample_type="tumor biopsy", tissue_type="tumor",
                   anatomical_site="liver", primary_diagnosis="hepatocellular carcinoma",
                   case_control_status="case", collection_method="percutaneous biopsy",
                   preservation_method="ffpe",
                   timepoint_label=None if timepoint_conflict else source_timepoint,
                   index_regimen="lenvatinib", sex_reported=annotation["Sex"].lower(),
                   molecular_subtype=clinical[2] if clinical else None,
                   response_criteria="RECIST 1.1" if response else None,
                   best_overall_response=response or None,
                   metadata_notes=("One MCD member groups ROIs from one biopsy section; member FL_ID "
                                   "matches public blood and Fig1b annotation keys. "
                                   + ("Clinical table has no matching FL_ID. " if clinical is None else "")
                                   + ("Clinical and Fig1b RECIST disagree; response left blank. "
                                      if not response_agrees else "")
                                   + ("MCD member and ROI descriptions disagree on T/POST; "
                                      "timepoint left blank. " if timepoint_conflict else "")
                                   + "PFS left blank pending harmonized origin, units and event semantics."))
        samples.append(row)
    sample_frame = pd.DataFrame(samples, columns=columns("sample_metadata"))
    join = "native_proof.source_member_name = MCD archive member; all acquisitions in one member share sample_id"
    fields = pd.DataFrame([
        field(PROOF, "description;source_member_name;source_acquisition_id", "image_metadata", "image_id",
              "Use source description when unique; for three duplicated descriptions, qualify by "
              "MCD member stem and acquisition number. Preserve the original description as "
              "region_annotation.", "MCD member + acquisition = unique physical image",
              "121 counted acquisitions yield 121 unique image IDs; three duplicate-description pairs."),
        field(PROOF, "records.native_proof", "image_metadata", "source_image_id;source_acquisition_id;roi_id",
              "Preserve verified native slide/acquisition/ROI identity and full source rectangle.", join),
        field(ARCHIVE, "Images/MCDfiles/*.mcd", "image_metadata", "sample_id",
              "MCD member stem groups ROI acquisitions from one biopsy section.", join,
              "121/121 expected images join 30 source biopsy members."),
        field(ARCHIVE, "Images/MCDfiles/*.mcd", "sample_metadata", "sample_id;source_sample_id;timepoint_label",
              "Use exact MCD member stem; retain its source timepoint label only when ROI descriptions "
              "do not disagree on T/POST. Do not expand timing semantics.",
              "distinct MCD member stem = sample_id", "One member has conflicting T/POST descriptions; "
              "its timepoint_label is blank. Other description spelling variants retain the member label."),
        field(META / "primary_scope_review.json", "scope_selector", "sample_metadata",
              "sample_type;tissue_type;anatomical_site;primary_diagnosis;case_control_status;collection_method;preservation_method",
              "Use published HCC tumor biopsy methods and FFPE sections for all source members.",
              "study = source manifest dataset_name", PUBLIC_METHODS),
        field(SUPPLEMENT, "Dataset/Blood/Luminex_Results.txt:FL_ID; Code/Fig1/Fig1b/Annotation_Fig1b.txt:FL_ID",
              "sample_metadata", "patient_key;source_subject_id;sex_reported",
              "Exact MCD member FL_ID joins both public FL_ID tables; qualify patient_key by study. "
              "Map figure Sex Male/Female to male/female.", "MCD member stem prefix = FL_ID",
              "30/30 biopsy-member samples link to 15 public subject IDs; 121/121 expected images link via member."),
        field(SUPPLEMENT, "Dataset/Clinical Data/Patient_Suvival&Response.txt:FL_ID;Subtype;Response_RECIST; "
              "Code/Fig1/Fig1b/Annotation_Fig1b.txt:RECIST", "sample_metadata",
              "molecular_subtype;response_criteria;best_overall_response",
              "Copy RNA immune subtype for exact clinical FL_ID; retain RECIST only when clinical "
              "and figure annotations agree, or only the figure annotation covers the subject.",
              "MCD member stem prefix = FL_ID", "Clinical table covers 28/30 samples (112/121 images); "
              "one subject has conflicting RECIST values and remains blank. PFS is not transferred."),
        field(META / "primary_scope_review.json", "published cohort treatment", "sample_metadata",
              "index_regimen", "All enrolled patients received first-line lenvatinib monotherapy.",
              "study scope = HCC cohort", PUBLIC_METHODS),
        field(PROOF, "records.source_channel", "panel_mapping", "source_name",
              "Use verified native MCD order for all 121 acquisitions.", "native channel order = source_channel_index"),
        field(S1, "Study;Original marker", "panel_mapping", "marker_id;role;keep_channel",
              "Join exact study-specific S1 labels; exclude both PD-L1 measurements and CD11c "
              "from downstream and saved pixels because the paper reports their antibodies "
              "did not work in this experiment. Preserve all 28 source rows and S1 identities.",
              "S1.Study = study; S1.Original marker = source_name",
              PUBLIC_METHODS + " (Discussion, limitations; antibody QC)"),
    ], columns=columns("field_mapping"))

    existing: dict[str, dict] = {}
    if (OUTPUT / "images.parquet").is_file():
        with formal_read_lease(OUTPUT, data_root=ROOT):
            prior = pd.read_parquet(OUTPUT / "images.parquet")
        existing = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
    migrate_duplicate_descriptions(existing, records, description_counts,
                                   sample_frame, panel, fields)
    with ZipFile(ARCHIVE) as archive:
        for member, items in sorted(groups.items()):
            pending = [item for item in items
                       if image_id_for(item, description_counts) not in existing or not
                       (OUTPUT / "images" / image_id_for(item, description_counts) / "input.tiff").is_file()]
            if not pending:
                continue
            member_bytes = items[0]["archive_entry"]["exact_archive_entry_size"]
            max_side = max(max(int(item["native_metadata"]["MaxX"]),
                               int(item["native_metadata"]["MaxY"])) for item in pending)
            next_bytes = len(kept) * max_side * max_side * 4
            next_id = image_id_for(pending[0], description_counts)
            if shutil.disk_usage(ROOT).free < 100 * 1024**3 + 2 * next_bytes:
                checkpoint("waiting_io", len(existing), next_id, reason="NAS 100 GiB reserve")
                return
            if shutil.disk_usage("/tmp").free < 64 * 1024**3 + member_bytes + 2 * next_bytes:
                checkpoint("waiting_io", len(existing), next_id, reason="local 64 GiB reserve")
                return
            source = local_member(archive, member, member_bytes, archive_size)
            indexed_before_batch = len(existing)
            for item in sorted(pending, key=lambda value: value["source_acquisition_id"]):
                image_id = image_id_for(item, description_counts)
                destination = OUTPUT / "images" / image_id / "input.tiff"
                spec = SourceSpec(image_id=image_id, panel_id=PANEL_ID, kind="mcd", path=source,
                                  channels=tuple(range(28)), acquisition_id=int(item["source_acquisition_id"]),
                                  pixel_size_um=(1.0, 1.0), source_revision=SOURCE_REVISION)
                checkpoint("exporting", indexed_before_batch, image_id, member=member,
                           batch_published=len(existing) - indexed_before_batch)
                with open_source(spec) as reader:
                    result = (recover_published(reader, panel, destination) if destination.is_file() else
                              export_input(reader, panel, destination, tile_size=256, data_root=ROOT))
                if not result["admitted"]:
                    checkpoint("excluded_no_marker_signal", indexed_before_batch, image_id,
                               member=member, batch_published=len(existing) - indexed_before_batch)
                    continue
                with formal_read_lease(destination, data_root=ROOT):
                    with tifffile.TiffFile(destination) as tif:
                        if (len(tif.pages) != len(kept) or tif.pages[0].dtype.name != "float32" or
                                tif.pages[0].shape != (result["height_px"], result["width_px"])):
                            raise ValueError(f"Published TIFF header differs: {image_id}")
                row = dict.fromkeys(columns("image_metadata"))
                row.update(study=STUDY, image_id=image_id, sample_id=Path(member).stem,
                           panel_id=PANEL_ID, image_kind="roi", source_image_id=item["image_id"],
                           source_files=[relative(ARCHIVE)], source_format="mcd",
                           source_acquisition_id=str(item["source_acquisition_id"]),
                           source_axes="CYX", source_dtype="float32", source_revision=SOURCE_REVISION,
                           slide_id=str(item["slide_id"]), roi_id=str(item["acquisition_roi_id"]),
                           region_annotation=item["description"], source_x0_px=0, source_y0_px=0,
                           source_x1_px=int(item["native_metadata"]["MaxX"]),
                           source_y1_px=int(item["native_metadata"]["MaxY"]),
                           pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                           input_image_path=relative(destination),
                           valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                              if result["valid_pixels"] else None),
                           intensity_description="Native unnormalized float32 MCD acquisition intensity",
                           split_method="source_acquisition", split_evidence=relative(PROOF),
                           metadata_notes=("Native acquisition and ROI identity from complete local proof. "
                                           + ("Duplicate source description; image_id uses MCD member "
                                              "and acquisition." if description_counts[item["description"]] > 1
                                              else "")),
                           **{key: result[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px",
                                                           "pad_bottom_px", "height_px", "width_px",
                                                           "source_channel_count", "channel_count")})
                existing[image_id] = row
                checkpoint("published_unindexed", indexed_before_batch, image_id,
                           member=member, batch_published=len(existing) - indexed_before_batch,
                           next_step="reopen published TIFFs and index this member on resume")
                print(f"published {len(existing)}/121 {image_id}", flush=True)
            if len(existing) > indexed_before_batch:
                save_input_tables(OUTPUT, pd.DataFrame(existing.values()), sample_frame,
                                  panel, fields, data_root=ROOT)
                next_unindexed = next((image_id_for(item, description_counts) for item in records
                                       if image_id_for(item, description_counts) not in existing), None)
                checkpoint("batch_saved", len(existing), next_unindexed,
                           member=member, sample_rows=len(sample_frame),
                           sample_linked_images=len(existing),
                           patient_linked_samples=len(sample_frame))
                print(f"indexed {len(existing)}/121 from {member}", flush=True)
                if os.environ.get("CELLWEFT_STUDY_INPUT_RECOVER_ONLY") == "1":
                    record_waiting_io()
                    return

    save_input_tables(OUTPUT, pd.DataFrame(existing.values()), sample_frame, panel, fields, data_root=ROOT)
    publish_preparer_review(len(existing), waiting=False)
    checkpoint("prepared", len(existing), None, sample_rows=len(sample_frame),
               sample_linked_images=len(existing), patient_linked_samples=len(sample_frame),
               next_step="independent review of source, panel, pixels, and M01/D01")


if __name__ == "__main__":
    if os.environ.get("CELLWEFT_STUDY_INPUT_REPAIR_ANTIBODY_QC") == "1":
        repair_antibody_qc()
    elif os.environ.get("CELLWEFT_STUDY_INPUT_REPAIR_PATIENT_RULE") == "1":
        repair_stale_patient_rule()
    elif os.environ.get("CELLWEFT_STUDY_INPUT_WAITING_IO") == "1":
        record_waiting_io()
    else:
        main()
