"""Prepare six proven PhenoCycler liver acquisitions as native-intensity inputs.

Inputs: Zenodo 17735558 ZIP, native_verification.json, study S1, and the
published cohort description. Outputs: per-image CYX TIFF and input tables.
Run with --run; every completed image is indexed before the next ZIP member.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import pandas as pd
import tifffile
import numpy as np

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "Yakubovsky2026_PhenoCycler_Liver"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
ARCHIVE = ROOT / "raw/public" / STUDY / "zenodo_17735558/PhenoCyclerLiverPanel.zip"
OUTPUT = ROOT / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / "cache/study_input" / STUDY
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_input_jobs" / STUDY / "0649_prepare_0")))
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PUBLIC_EVIDENCE = ROOT / "raw/public" / STUDY / "manifest.json"
CLINICAL = ROOT / "raw/public" / STUDY / "metadata/figure1b_clinical.csv"
PREPARER = "codex_yakubovsky_" + JOB.name


def relative(path: Path) -> str:
    return metadata_source_relative(path, ROOT).as_posix()


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def publish_json(path: Path, value: dict) -> None:
    TEMP.mkdir(parents=True, exist_ok=True)
    staged = TEMP / (path.name + ".new")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    with formal_publish_lease(path, directory=path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, target)
        os.replace(target, path)


def progress(stage: str, **details: object) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str) if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    linked = set(samples.loc[samples.patient_key.notna(), "sample_id"]) if len(samples) else set()
    repairing = "repair_" in JOB.name
    state = dict(study=STUDY, phase="repair" if repairing else "prepare",
                 scientific_repair_round=1 if repairing else 0,
                 stage=stage, expected_images=6, exported_images=len(images),
                 sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=len(linked),
                 patient_linked_images=int(images.sample_id.isin(linked).sum()) if len(images) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    publish_json(JOB / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> list[dict]:
    with formal_read_lease(META):
        proof = json.loads((META / "native_verification.json").read_text())
        manifest = json.loads((META / "source_manifest.json").read_text())
    stat = ARCHIVE.stat()
    if (not proof["scope_complete"] or proof["gaps"] or len(proof["native_result"]["images"]) != 6
            or stat.st_size != manifest["source"]["size"]):
        raise ValueError("Local ZIP does not match the complete six-image native proof")
    by_member = {entry["source_member"]: entry for entry in proof["native_result"]["images"]}
    scope = []
    for item in manifest["images"]:
        member = item["image_member"]["name"]
        native = by_member[member]
        if (not native["complete"] or not native["native_read_evidence"]["all_required_protein_channels_decoded"]
                or native["native_read_evidence"]["channel_count"] != 49
                or native["geometry"] != dict(width=item["width"], height=item["height"])):
            raise ValueError(f"Incomplete native evidence: {member}")
        prior = native["source_stat"]
        if (stat.st_size, stat.st_mtime_ns) != (prior["size"], prior["mtime_ns"]):
            raise ValueError(f"ZIP version differs from native evidence: {member}")
        scope.append(dict(member=member, image_id=Path(member).name.removesuffix(".ome.tif"),
                          file_size=item["image_member"]["size"],
                          crc=native["member_stat"]["CRC"], native=native,
                          height=item["height"], width=item["width"],
                          scale=item["pixel_calibration"]["pixelWidth"]["value"],
                          source_revision=f"zenodo_17735558;zip_size={stat.st_size};zip_mtime_ns={stat.st_mtime_ns};member_crc={native['member_stat']['CRC']}"))
    return sorted(scope, key=lambda item: item["file_size"])


def source_stage(item: dict) -> Path:
    TEMP.mkdir(parents=True, exist_ok=True)
    path = TEMP / Path(item["member"]).name
    binding = path.with_suffix(path.suffix + ".source.json")
    identity = dict(archive=relative(ARCHIVE), member=item["member"],
                    size=item["file_size"], crc=item["crc"], revision=item["source_revision"])
    if (path.exists() and binding.exists() and path.stat().st_size == item["file_size"]
            and json.loads(binding.read_text()) == identity):
        return path
    if shutil.disk_usage(TEMP).free < 64 * 1024**3 + item["file_size"] * 2:
        raise OSError(28, "Local source and output staging would consume the 64 GiB reserve")
    partial = path.with_suffix(path.suffix + ".partial")
    with formal_read_lease(ARCHIVE), zipfile.ZipFile(ARCHIVE) as archive:
        member = archive.getinfo(item["member"])
        if (member.file_size, member.CRC) != (item["file_size"], item["crc"]):
            raise ValueError(f"ZIP member differs from native proof: {item['member']}")
        with archive.open(member) as source, partial.open("wb") as staged:
            shutil.copyfileobj(source, staged, length=8 * 1024**2)
    if partial.stat().st_size != item["file_size"]:
        raise ValueError(f"Incomplete local source stage: {item['member']}")
    os.replace(partial, path)
    binding.write_text(json.dumps(identity, sort_keys=True))
    return path


def verify_header(path: Path, item: dict) -> None:
    with tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        names = [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")]
        if (len(tif.pages) != 49 or names != item["native"]["channels"]
                or any(page.shape != (item["height"], item["width"]) or page.dtype.name != "float32"
                       for page in tif.pages)
                or int(pixels.attrib["SizeZ"]) != 1 or int(pixels.attrib["SizeT"]) != 1):
            raise ValueError(f"Staged TIFF differs from native proof: {item['member']}")


def sample_tables(scope: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(CLINICAL, dtype=str, keep_default_na=False).set_index("cohort_id", verify_integrity=True)
    rows = []
    for item in scope:
        source_id = item["image_id"].removesuffix("_Liver")
        # One acquisition adds an E section suffix to its published cohort ID.
        patient = None if source_id.endswith("E") else source_id
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=source_id, patient_key=patient,
                   source_subject_id=patient, source_sample_id=source_id,
                   sample_type="tissue", tissue_type="liver",
                   anatomical_site="liver",
                   metadata_notes=("Published PhenoCycler cohort: living liver donor" if source_id.startswith("M")
                                   else "Published PhenoCycler cohort: adjacent normal liver")
                   + ("; section-suffixed acquisition has no confirmed patient mapping."
                      if source_id.endswith("E") else "."))
        if patient is not None and patient in clinical.index:
            record = clinical.loc[patient]
            for field in ("sex_reported", "age_band_years", "bmi_kg_m2", "smoking_reported",
                          "chemotherapy_reported", "resection_indication", "surgery_description", "histology"):
                row[field] = record[field]
            row["collection_method"] = ("biopsy" if record["surgery_description"].lower().endswith("biopsy")
                                        else "resection")
            row["metadata_notes"] += " Paper Fig. 1b provides the recorded clinical attributes."
        else:
            row["metadata_notes"] += (" Paper Fig. 1b cannot be joined without a patient mapping."
                                      if patient is None else " Paper Fig. 1b has no row for this cohort member.")
        rows.append(row)
    samples = pd.DataFrame(rows, columns=columns("sample_metadata") + [
        "age_band_years", "smoking_reported", "chemotherapy_reported",
        "resection_indication", "surgery_description"])
    fields = []

    def add(source: Path, field: str, table: str, target: str, keys: str, rule: str,
            notes: dict | None = None) -> None:
        row = dict.fromkeys(columns("field_mapping"), "")
        row.update(study=STUDY, source_file=relative(source), source_field=field,
                   target_table=table, target_field=target, join_keys=keys,
                   value_mapping="{}", transformation_rule=rule,
                   notes=json.dumps(notes) if notes else "")
        fields.append(row)

    add(META / "native_verification.json", "native_result.images;source_member;channels;geometry",
        "image_metadata", "source_image_id;source_files;source_dtype",
        "source_member", "Use six complete native proofs; retain ZIP member as source_image_id.")
    add(META / "source_manifest.json", "images.image_member.name", "sample_metadata",
        "sample_id;source_sample_id", "ZIP member stem=sample_id",
        "Each independent TIFF stem identifies its sampled section; retain any section suffix.")
    add(PUBLIC_EVIDENCE, "publication_sources.zenodo_url;publication_sources.result", "sample_metadata",
        "sample_type;tissue_type;anatomical_site;metadata_notes", "study DOI",
        "Published PhenoCycler material is liver tissue; Zenodo describes living donor and adjacent-normal cohorts.")
    add(PUBLIC_EVIDENCE, "publication_sources.paper_url;publication_sources.section_suffix_link", "sample_metadata",
        "patient_key;source_subject_id", "published cohort ID=unsuffixed member stem",
        "Use direct cohort IDs for unsuffixed members. The section-suffixed member has only an inferred prefix link, so patient fields remain blank.")
    add(CLINICAL, "Figure 1b;cohort_id", "sample_metadata", "patient_key",
        "cohort_id=patient_key", "Join only confirmed patient keys. Figure 1b does not cover every member; the section-suffixed acquisition has no confirmed patient key.")
    for source_field, target_field, definition in (
        ("sex_reported", "sex_reported", "Reported sex in paper Figure 1b"),
        ("age_band_years", "age_band_years", "Published age range in years; no exact age inferred"),
        ("bmi_kg_m2", "bmi_kg_m2", "Published BMI in kg/m2"),
        ("smoking_reported", "smoking_reported", "Published yes/no smoking entry; timing unspecified"),
        ("chemotherapy_reported", "chemotherapy_reported", "Published chemotherapy entry; timing unspecified"),
        ("resection_indication", "resection_indication", "Published resection indication"),
        ("surgery_description", "surgery_description", "Published surgery description"),
        ("histology", "histology", "Published histology description"),
    ):
        notes = ({"data_type": "string", "definition": definition}
                 if target_field in {"age_band_years", "smoking_reported", "chemotherapy_reported",
                                     "resection_indication", "surgery_description"} else None)
        add(CLINICAL, source_field, "sample_metadata", target_field, "cohort_id=patient_key",
            "Transcribed from paper Figure 1b; blank where that cohort ID is absent.", notes)
    add(CLINICAL, "surgery_description", "sample_metadata", "collection_method",
        "cohort_id=patient_key", "Open biopsy maps to biopsy; other reported operations map to resection.")
    add(S1, "Study;Original marker;Segmentation channel", "panel_mapping",
        "marker_id;keep_channel;output_channel_index", "Study;Original marker",
        "Exact 49-channel native order joined to current S1; retain DAPI nuclear input and 48 admitted proteins.")
    return samples, pd.DataFrame(fields, columns=columns("field_mapping"))


def image_row(item: dict, result: dict, destination: Path) -> dict:
    source_id = item["image_id"].removesuffix("_Liver")
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"], sample_id=source_id,
               panel_id="PhenoCycler_P01", image_kind="roi", source_image_id=item["member"],
               source_files=[relative(ARCHIVE)], source_format="ome_tiff",
               source_axes="CYX", source_dtype="float32", source_revision=item["source_revision"],
               roi_id=source_id, region_annotation=("living donor liver" if source_id.startswith("M")
                                                    else "adjacent normal liver"),
               source_x0_px=0, source_y0_px=0, source_x1_px=item["width"], source_y1_px=item["height"],
               pixel_size_x_um=item["scale"], pixel_size_y_um=item["scale"],
               input_image_path=relative(destination),
               intensity_description="Published PhenoCycler OME-TIFF float32 intensity; no CellWeft normalization",
               split_method="already_independent",
               split_evidence="One complete OME-TIFF per QuPath project acquisition; native proof covers all 49 channels.",
               metadata_notes="ZIP member retained in source_image_id; native source proof and public cohort cited in field_mapping.")
    for key in ("height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = result[key]
    return row


def check_pixels(item: dict, result: dict, source: Path, destination: Path) -> None:
    """Compare every retained channel in source-interior and boundary blocks."""
    h, w = item["height"], item["width"]
    left, top = result["pad_left_px"], result["pad_top_px"]
    windows = [(w // 2, h // 2, min(w // 2 + 32, w), min(h // 2 + 32, h)),
               (0, 0, min(32, w), min(32, h))]
    with formal_read_lease(destination), tifffile.TiffFile(source) as native, tifffile.TiffFile(destination) as output:
        for channel in range(result["channel_count"]):
            for x0, y0, x1, y1 in windows:
                original = read_tiff_window(native, channel, (x0, y0, x1, y1))
                delivered = read_tiff_window(output, channel,
                                             (x0 + left, y0 + top, x1 + left, y1 + top))
                if original.dtype != delivered.dtype or not np.array_equal(original, delivered):
                    raise ValueError(f"Native pixel mismatch in {item['image_id']}, channel {channel}")
            if left or top:
                padding = read_tiff_window(output, channel, (0, 0, 32, 32))
                if np.any(padding):
                    raise ValueError(f"Nonzero square padding in {item['image_id']}, channel {channel}")
    publish_json(CACHE / "pixel_checks" / f"{item['image_id']}.json",
                 dict(image_id=item["image_id"], checked_channels=result["channel_count"],
                      source_windows=windows, padding_corner_checked=bool(left or top),
                      source=relative(ARCHIVE), member=item["member"], output=relative(destination),
                      result="exact"))


def write_review(scope: list[dict], complete: bool) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
    pixel_checked = sum((CACHE / "pixel_checks" / f"{image_id}.json").is_file()
                        for image_id in images.image_id)
    evidence = ";".join(map(relative, (META / "native_verification.json", PUBLIC_EVIDENCE, S1)))
    rows = []
    for check, detail in (
        ("S01", f"Six complete native ZIP member proofs; {len(images)}/6 published inputs; no excluded member."),
        ("I01", "Unique QuPath member and delivered image path per input."),
        ("I02", f"{len(images)} image/sample links; six sample rows and five confirmed patient keys."),
        ("I03", "Full source frames and centered square padding; source scale from QuPath metadata."),
        ("I04", "One published acquisition per member; the section-suffix cohort link needs independent confirmation."),
        ("P01", "Native proof and S1 agree on 49 ordered source labels."),
        ("P02", "DAPI nuclear input and 48 distinct protein channels retained."),
        ("P03", f"{len(images)} published TIFF headers reopened by shared exporter."),
        ("I05", f"{pixel_checked}/{len(images)} per-image 49-channel source/output block checks; see cache pixel_checks."),
        ("M01", "Six image/sample links; five patient links confirmed, one section-suffixed acquisition unresolved."),
        ("M02", "Paper Figure 1b clinical values transcribed only where the patient link is confirmed; one confirmed cohort member has no Figure 1b row."),
        ("D01", f"Preparer delivery {len(images)}/6; independent review and current input binding pending."),
    ):
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                         record_id=STUDY, check_id=check, result="unresolved",
                         source_evidence=evidence, output_evidence=relative(OUTPUT),
                         details="Preparer self-check: " + detail))
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        temporary = path.with_name(path.name + ".partial")
        pd.DataFrame(rows, columns=columns("review_results")).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    if complete:
        progress("prepared_pending_independent_review")


def check_existing(scope: list[dict]) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet").set_index("image_id", verify_integrity=True)
    for item in scope:
        image_id = item["image_id"]
        if image_id not in images.index or (CACHE / "pixel_checks" / f"{image_id}.json").is_file():
            continue
        source = TEMP / (Path(item["member"]).name + ".for_check")
        if not source.is_file():
            source = source_stage(item)
        verify_header(source, item)
        row = images.loc[image_id]
        check_pixels(item, row.to_dict(), source, ROOT / row.input_image_path)
        if source.name.endswith(".for_check"):
            source.unlink()
        progress("pixel_checked", last_image=image_id)


def repair_metadata() -> None:
    scope = source_scope()
    corrected_samples, corrected_fields = sample_tables(scope)
    with formal_read_lease(OUTPUT):
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
    affected = corrected_samples.sample_id.str.endswith("E")
    if int(affected.sum()) != 1 or set(samples.sample_id) != set(corrected_samples.sample_id):
        raise ValueError("Published samples differ from the reviewed six-sample scope")
    corrected = corrected_samples.loc[affected].iloc[0].fillna("")
    samples.loc[samples.sample_id == corrected.sample_id, corrected.index] = corrected.values

    obsolete = (fields.source_file == relative(PUBLIC_EVIDENCE)) & fields.source_field.isin(
        ["publication_pheno_cycler_cohort", "supplementary_table_5"])
    fields = fields.loc[~obsolete].copy()
    keys = ("source_file", "source_field", "target_table", "target_field")
    for _, row in corrected_fields.fillna("").iterrows():
        match = pd.Series(True, index=fields.index)
        for key in keys:
            match &= fields[key] == row[key]
        if match.any():
            fields.loc[match, row.index] = row.values
        else:
            fields = pd.concat([fields, row.to_frame().T], ignore_index=True)

    TEMP.mkdir(parents=True, exist_ok=True)
    staged = {}
    for name, table in (("sample_metadata.csv", samples), ("field_mapping.csv", fields)):
        path = TEMP / (name + ".repair_1")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        staged[name] = path
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        for name, source in staged.items():
            target = OUTPUT / name
            partial = target.with_name(name + ".partial")
            shutil.copyfile(source, partial)
            os.replace(partial, target)
    progress("metadata_repaired_pending_independent_review",
             repaired_scope="one patient link and dependent clinical values; field mappings",
             next_step="independent M01/M02/D01 re-review")


def run(*, metadata_only: bool = False) -> None:
    scope = source_scope()
    panel = panel_from_s1(STUDY, "PhenoCycler_P01", scope[0]["native"]["channels"], S1)
    if any(item["native"]["channels"] != scope[0]["native"]["channels"] for item in scope):
        raise ValueError("Source channels differ between proven members")
    if (len(panel) != 49 or int(panel.keep_channel.sum()) != 49
            or int(panel.nuclear.sum()) != 1 or int((panel.role == "downstream").sum()) != 48):
        raise ValueError("Study S1 no longer selects the proven 49-channel panel")
    samples, fields = sample_tables(scope)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_input_tables(OUTPUT, pd.DataFrame(columns=columns("image_metadata")), samples, panel, fields)
    progress("metadata_ready", next_image=scope[0]["image_id"])
    if metadata_only:
        write_review(scope, complete=False)
        pending = next((item["image_id"] for item in scope if item["image_id"] not in
                        set(pd.read_parquet(OUTPUT / "images.parquet").image_id)), None)
        checkpoint = (Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) /
                      "publish_checkpoints" / metadata_source_relative(OUTPUT / 'images' / pending / 'input.tiff', '/'))
        progress("waiting_io", next_image=pending,
                 issue="Formal image publication returned errno 5; closed local checkpoint retained",
                 local_source_stage=str(TEMP / f"{pending}.ome.tif"),
                 local_output_checkpoint=str(checkpoint / "complete.json"),
                 local_closed_input=str(checkpoint / "input.tiff"),
                 formal_partial=str(OUTPUT / "images" / pending / "input.tiff.partial"))
        return
    for item in scope:
        destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
        with formal_read_lease(OUTPUT):
            current = pd.read_parquet(OUTPUT / "images.parquet")
        if item["image_id"] in set(current.image_id):
            if not destination.is_file() or not destination.stat().st_size:
                raise ValueError(f"Existing index has no completed TIFF: {item['image_id']}")
            continue
        progress("staging_source", current_image=item["image_id"], next_image=item["image_id"])
        stage = source_stage(item)
        verify_header(stage, item)
        source = SourceSpec(item["image_id"], "PhenoCycler_P01", "tiff", stage,
                            tuple(range(49)), pixel_size_um=(item["scale"], item["scale"]),
                            source_revision=item["source_revision"])
        progress("exporting", current_image=item["image_id"], next_image=item["image_id"])
        try:
            with open_source(source) as reader:
                result = export_input(reader, panel, destination)
            if not result["admitted"]:
                raise ValueError(f"All retained protein measurements are zero: {item['image_id']}")
            check_pixels(item, result, stage, destination)
            save_input_tables(OUTPUT, pd.DataFrame([image_row(item, result, destination)]), samples, panel, fields)
            stage.unlink()
            stage.with_suffix(stage.suffix + ".source.json").unlink()
            progress("image_completed", last_image=item["image_id"],
                     next_image=next((x["image_id"] for x in scope if x["image_id"] not in
                                      set(pd.read_parquet(OUTPUT / "images.parquet").image_id)), None))
        except OSError as error:
            if error.errno not in {5, 28, 110, 116}:
                raise
            checkpoint = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / \
                "publish_checkpoints" / metadata_source_relative(destination.resolve(), '/')
            progress("waiting_io", next_image=item["image_id"],
                     issue=f"I/O or capacity error: errno {error.errno}: {error.strerror}",
                     local_source_stage=str(stage),
                     local_output_checkpoint=str(checkpoint / "complete.json"),
                     local_closed_input=str(checkpoint / "input.tiff"),
                     formal_partial=str(destination.with_name("input.tiff.partial")))
            write_review(scope, complete=False)
            return
    check_existing(scope)
    write_review(scope, complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--repair-metadata", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    elif args.metadata_only:
        run(metadata_only=True)
    elif args.check_existing:
        scope = source_scope()
        check_existing(scope)
        write_review(scope, complete=len(pd.read_parquet(OUTPUT / "images.parquet")) == len(scope))
    elif args.repair_metadata:
        repair_metadata()
