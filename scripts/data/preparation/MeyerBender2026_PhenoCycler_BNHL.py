"""Prepare proven S-BIAD2100 TMA cores from native Zarr ZIP intensities."""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile
import zarr

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "MeyerBender2026_PhenoCycler_BNHL"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
TEMP = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / "cache/study_preparation" / STUDY / "jobs/0771_prepare_0")))
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PREPARER = "codex_meyerbender_" + JOB.name
PANEL_ID = "PhenoCycler_TMA"
EXPECTED_IMAGES = 255  # 250 TMA cores and five distinct WSI sections in the source manifests.


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
        partial = path.with_name(path.name + ".partial")
        shutil.copyfile(staged, partial)
        os.replace(partial, path)


def progress(stage: str, **details: object) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
        samples = (pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
                   if (OUTPUT / "sample_metadata.csv").is_file() else pd.DataFrame())
    linked = set(samples.loc[samples.patient_key.ne(""), "sample_id"]) if len(samples) else set()
    state = dict(study=STUDY, phase="prepare", scientific_repair_round=0,
                 stage=stage, expected_images=EXPECTED_IMAGES, exported_images=len(images),
                 sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=len(linked),
                 patient_linked_images=int(images.sample_id.isin(linked).sum()) if len(images) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    publish_json(JOB / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def source_scope() -> tuple[list[dict], list[str]]:
    with formal_read_lease(META):
        manifest = json.loads((META / "tma_source_manifest.json").read_text())
        ordered = json.loads((META / "tma_ordered_markers.json").read_text())["ordered_channels"]
        proofs = {core["core_id"]: json.loads(
            (META / "tma_native_proofs" / f"{core['core_id']}.json").read_text())
            for core in manifest["cores"]}
        stats = {core["core_id"]: (ROOT / core["archive_rel"]).stat()
                 if (ROOT / core["archive_rel"]).is_file() else None
                 for core in manifest["cores"]}
    if len(manifest["cores"]) != 250 or len(ordered) != 56 or manifest["ordered_channels"] != ordered:
        raise ValueError("TMA coverage or ordered panel differs from the source manifest")
    scope = []
    for core in manifest["cores"]:
        image_id = core["core_id"]
        archive = ROOT / core["archive_rel"]
        proof_path = META / "tma_native_proofs" / f"{image_id}.json"
        proof = proofs[image_id]
        stat = stats[image_id]
        if (not proof["complete"] or proof["fixture"] or proof["pixel_array"] != "_image_raw"
                or proof["core_id"] != image_id or proof["shape"] != [56, 3000, 3000]
                or proof["ordered_channels"] != ordered or core["ordered_channels"] != ordered
                or core["raw_shape"] != [56, 3000, 3000] or core["raw_dtype"] != "|u1"
                or stat is None or (stat.st_size, stat.st_mtime_ns) !=
                (proof["file_stat"]["size"], proof["file_stat"]["mtime_ns"])):
            raise ValueError(f"Complete source or native proof differs: {image_id}")
        attrs = core["source_attributes"]
        if not core["identity_confirmed"] or attrs["SampleID"] != image_id or not attrs["PatientID"]:
            raise ValueError(f"Source identity is unresolved: {image_id}")
        scope.append(dict(image_id=image_id, archive=archive, proof=proof_path, attrs=attrs,
                          names=ordered,
                          revision=f"S-BIAD2100;size={stat.st_size};mtime_ns={stat.st_mtime_ns}"))
    if len({item["image_id"] for item in scope}) != 250:
        raise ValueError("Duplicate source core identity")
    return sorted(scope, key=lambda item: item["image_id"]), ordered


def stage_source(item: dict) -> Path:
    TEMP.mkdir(parents=True, exist_ok=True)
    stage = TEMP / item["image_id"]
    binding = dict(archive=relative(item["archive"]), revision=item["revision"])
    marker = stage / "source.json"
    if marker.is_file() and json.loads(marker.read_text()) == binding:
        return stage
    if shutil.disk_usage(TEMP).free < 64 * 1024**3 + 2 * 56 * 3000**2 + 2 * 1024**3:
        raise OSError(28, "Local staging would consume the 64 GiB reserve")
    partial = TEMP / (item["image_id"] + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    with formal_read_lease(item["archive"]), zipfile.ZipFile(item["archive"]) as source:
        members = [info for info in source.infolist()
                   if info.filename.startswith(("_image_raw/", "channels/")) and not info.is_dir()]
        raw = [info.filename for info in members if info.filename.startswith("_image_raw/")]
        if len([name for name in raw if Path(name).name.count(".") == 2
                and Path(name).name[0].isdigit()]) != 256:
            raise ValueError(f"Incomplete native Zarr chunk inventory: {item['image_id']}")
        for info in members:
            target = partial / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
    array = zarr.open_array(str(partial / "_image_raw"), mode="r")
    names = zarr.open_array(str(partial / "channels"), mode="r")[:].tolist()
    if array.shape != (56, 3000, 3000) or array.dtype != np.dtype("uint8") or names != item["names"]:
        raise ValueError(f"Staged native Zarr differs from manifest: {item['image_id']}")
    marker_in_partial = partial / "source.json"
    marker_in_partial.write_text(json.dumps(binding, sort_keys=True))
    if stage.exists():
        shutil.rmtree(stage)
    partial.rename(stage)
    return stage


def metadata_tables(scope: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = []
    for item in scope:
        attrs = item["attrs"]
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=item["image_id"], patient_key=attrs["PatientID"],
                   source_subject_id=attrs["PatientID"], source_sample_id=attrs["SampleID"],
                   sample_type="tissue core",
                   preservation_method="ffpe", primary_diagnosis=attrs["Entity"],
                   metadata_notes="Published TMA Zarr SampleID and PatientID directly identify this core and source patient; exact anatomical site and other clinical attributes are not present in the local core manifest.")
        samples.append(row)
    fields = []

    def add(source: Path, source_field: str, table: str, target: str, keys: str, rule: str) -> None:
        row = dict.fromkeys(columns("field_mapping"), "")
        row.update(study=STUDY, source_file=relative(source), source_field=source_field,
                   target_table=table, target_field=target, join_keys=keys,
                   value_mapping="{}", transformation_rule=rule)
        fields.append(row)

    manifest = META / "tma_source_manifest.json"
    add(manifest, "cores.core_id;archive_rel;raw_shape;raw_dtype", "image_metadata",
        "image_id;source_files;source_dtype", "core_id", "One released TMA core ZIP is one independent CYX image; retain full 3000x3000 frame.")
    add(manifest, "cores.source_attributes.SampleID;PatientID", "sample_metadata",
        "sample_id;source_sample_id;patient_key;source_subject_id", "core_id=SampleID",
        "Direct published core-to-patient link; distinct PatientID values define patient coverage.")
    add(manifest, "cores.source_attributes.Entity", "sample_metadata", "primary_diagnosis",
        "core_id=SampleID", "Retain source Entity label verbatim; no stage or diagnosis-code inference.")
    add(META / "source_evidence.json", "exact_original_source_statements;paper.tissue",
        "sample_metadata", "sample_type", "core_id=SampleID;250 TMA cores only",
        "Published Results identify 250 independent 1-mm TMA core crops; set sample_type to tissue core for these released cores.")
    add(META / "source_evidence.json", "paper.tissue", "sample_metadata", "preservation_method",
        "core_id=SampleID;250 TMA cores only",
        "Published FFPE tissue context, corroborated by Methods Par45 deparaffinization in raw/public/MeyerBender2026_PhenoCycler_BNHL/metadata.json#public/MeyerBender2026_PhenoCycler_BNHL/81_MeyerBender2026_PhenoCycler_BNHL.xml; map FFPE to controlled vocabulary ffpe. Do not infer other individual clinical values.")
    fields[-1]["value_mapping"] = '{"FFPE":"ffpe"}'
    for field in fields:
        if field["target_field"] in {"sample_id;source_sample_id;patient_key;source_subject_id", "primary_diagnosis"}:
            field["notes"] = ("Official S-BIAD2100 spatialproteomics_biostudies_filelist_3.json attributes "
                "SampleID/PatientID/Entity are retained as cores.source_attributes in tma_source_manifest.json; "
                "join the ZIP/core_id to SampleID. Current official file-list and all 250 identity/Entity rows "
                "were independently checked using the recorded study-level source evidence.")
    add(META / "tma_native_proofs", "complete;pixel_array;shape;file_stat", "image_metadata",
        "source_revision;source_axes;source_channel_count", "core_id", "Use complete stat-bound _image_raw native proof for each source core.")
    add(META / "tma_ordered_markers.json", "ordered_channels", "panel_mapping", "source_name",
        "source_channel_index", "Use TMA source order; do not use WSI panel or processed _image intensities.")
    add(S1, "Study;Original marker;Segmentation channel", "panel_mapping",
        "marker_id;keep_channel;output_channel_index", "Study;Original marker",
        "Current S1 selects DAPI nuclear input and 55 distinct protein measurements.")
    add(META / "tma_source_manifest.json", "cores.source_attributes", "review_results", "M01",
        "core_id=SampleID", "Local core manifest and 250 native proofs give 250 core/sample links and 132 source patient keys; WSI sections are separate and still downloading.")
    return pd.DataFrame(samples, columns=columns("sample_metadata")), \
        pd.DataFrame(fields, columns=columns("field_mapping"))


def image_row(item: dict, result: dict, destination: Path) -> dict:
    row = dict.fromkeys(columns("image_metadata"))
    row.update(study=STUDY, image_id=item["image_id"], sample_id=item["image_id"],
               panel_id=PANEL_ID, image_kind="tma_core", source_image_id="_image_raw",
               source_files=[relative(item["archive"])], source_format="zarr_zip",
               source_axes="CYX", source_dtype="uint8", source_revision=item["revision"],
               core_id=item["image_id"], roi_id=item["image_id"],
               region_annotation=item["attrs"]["Entity"],
               source_x0_px=0, source_y0_px=0, source_x1_px=3000, source_y1_px=3000,
               input_image_path=relative(destination),
               intensity_description="Released _image_raw uint8 PhenoCycler values; no CellWeft normalization",
               split_method="already_independent",
               split_evidence="One complete published TMA core Zarr ZIP per SampleID; native proof covers all 56 raw channels.",
               metadata_notes="Native Zarr pixel scale is not recorded in the verified core manifest; source proof and identity are cited in field_mapping.")
    for key in ("height_px", "width_px", "pad_left_px", "pad_right_px", "pad_top_px",
                "pad_bottom_px", "source_channel_count", "channel_count"):
        row[key] = result[key]
    return row


def check_pixels(item: dict, destination: Path, panel: pd.DataFrame, stage: Path) -> None:
    source = zarr.open_array(str(stage / "_image_raw"), mode="r")
    windows = [(1450, 1450, 1482, 1482), (0, 0, 32, 32)]
    with formal_read_lease(destination), tifffile.TiffFile(destination) as output:
        pixels = ET.fromstring(output.ome_metadata).find(".//{*}Pixels")
        names = [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")]
        kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
        if (len(output.pages) != len(kept) or names != kept.marker_id.tolist()
                or any(page.shape != (3000, 3000) or page.dtype != np.dtype("uint8")
                       for page in output.pages)):
            raise ValueError(f"Published TIFF header differs: {item['image_id']}")
        for output_index, source_index in enumerate(kept.source_channel_index):
            for x0, y0, x1, y1 in windows:
                native = source[int(source_index), y0:y1, x0:x1]
                published = read_tiff_window(output, output_index, (x0, y0, x1, y1))
                if not np.array_equal(native, published):
                    raise ValueError(f"Native pixel mismatch: {item['image_id']}/{source_index}")
    publish_json(OUTPUT / "pixel_checks" / f"{item['image_id']}.json",
                 dict(image_id=item["image_id"], source=relative(item["archive"]),
                      output=relative(destination), channels=len(kept), windows=windows,
                      result="exact"))


def write_review(complete_tma: bool, source_failures: list[dict] | None = None) -> None:
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").is_file() else pd.DataFrame()
    evidence = ";".join(map(relative, (META / "tma_source_manifest.json",
                                      META / "tma_ordered_markers.json", S1)))
    details = {
        "S01": f"Preparer: {len(images)}/250 source TMA cores exported; five WSI archives remain incomplete. "
               f"Archive read failures: {json.dumps(source_failures or [], ensure_ascii=False)}",
        "I01": "Preparer: each indexed core has one source ZIP and one input.tiff.",
        "I02": f"Preparer: {len(images)} indexed images join the source SampleID core rows.",
        "I03": "Preparer: full 3000x3000 native frames; pixel scale unverified and left blank.",
        "I04": "Preparer: released core SampleID is the physical core identity; Entity retained verbatim.",
        "P01": "Preparer: 56 native source channels joined in order to current S1.",
        "P02": "Preparer: DAPI nuclear input and 55 distinct downstream proteins retained.",
        "P03": "Preparer: shared exporter reopened each published header.",
        "I05": "Preparer: per-image pixel checks are in pixel_checks/; independent sampling pending.",
        "M01": "Preparer: local manifest gives 250 core/sample links and 132 source patient keys; WSI is pending.",
        "M02": "Preparer: only source Entity and FFPE tissue context recorded; other clinical fields remain blank.",
        "D01": f"Preparer delivery {len(images)}/250 TMA cores; independent review and binding pending.",
    }
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                 record_id=STUDY, check_id=key, result="unresolved", source_evidence=evidence,
                 output_evidence=relative(OUTPUT), details=value) for key, value in details.items()]
    path = OUTPUT / "review_results.csv"
    with formal_publish_lease(path, directory=OUTPUT):
        partial = path.with_name(path.name + ".partial")
        pd.DataFrame(rows, columns=columns("review_results")).to_csv(partial, index=False, encoding="utf-8-sig")
        os.replace(partial, path)
    if complete_tma:
        progress("prepared_tma_pending_wsi_and_independent_review")


def repair_metadata() -> None:
    """Repair review-0 M02 only; preserve pixels and defer renewed D01 to the reviewer."""
    TEMP.mkdir(parents=True, exist_ok=True)
    estimate = 16 * 1024**2  # CSVs, before/after evidence and checkpoints; no pixel staging.
    free = {"nas": shutil.disk_usage(ROOT).free, "local": shutil.disk_usage(TEMP).free}
    if free["nas"] < 100 * 1024**3 + estimate or free["local"] < 64 * 1024**3 + estimate:
        publish_json(JOB / "progress.json", dict(study=STUDY, phase="repair",
            scientific_repair_round=1, stage="waiting_io", next_step="repair_metadata",
            estimated_bytes=estimate, free_bytes=free, issue="Required disk reserve unavailable"))
        return
    names = ("sample_metadata.csv", "field_mapping.csv", "review_results.csv")
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        original = {name: (OUTPUT / name).read_bytes() for name in names}
        samples = pd.read_csv(OUTPUT / names[0], dtype=str, keep_default_na=False)
        fields = pd.read_csv(OUTPUT / names[1], dtype=str, keep_default_na=False)
        review = pd.read_csv(OUTPUT / names[2], dtype=str, keep_default_na=False)
        images = pd.read_parquet(OUTPUT / "images.parquet")
        manifest = json.loads((META / "tma_source_manifest.json").read_text())
        protected = [OUTPUT / "images.parquet", OUTPUT / "panel_mapping.csv"] + [ROOT / p for p in images.input_image_path]
        versions = {relative(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in protected}
    scope = [dict(image_id=core["core_id"], attrs=core["source_attributes"]) for core in manifest["cores"]]
    expected, generated = metadata_tables(scope)
    expected = expected.fillna("").set_index("sample_id").loc[samples.sample_id]
    for field in ("patient_key", "source_subject_id", "source_sample_id", "primary_diagnosis", "sample_type"):
        if samples[field].tolist() != expected[field].tolist():
            raise ValueError(f"Repair scope differs from reviewed source: {field}")
    if len(samples) != 250 or not samples.preservation_method.isin(["FFPE", "ffpe"]).all():
        raise ValueError("Unexpected preservation values or sample coverage")
    samples["preservation_method"] = "ffpe"
    for target in ("sample_type", "preservation_method"):
        fields = fields[~(fields.target_table.eq("sample_metadata") & fields.target_field.eq(target))]
        fields = pd.concat([fields, generated[generated.target_field.eq(target)]], ignore_index=True)
    for target in ("sample_id;source_sample_id;patient_key;source_subject_id", "primary_diagnosis"):
        fields.loc[fields.target_field.eq(target), "notes"] = generated.loc[generated.target_field.eq(target), "notes"].iloc[0]
    note = dict(scientific_repair_round=1, role="preparer", image_ids=images.image_id.tolist(),
        note="M02 repair: 250 FFPE→ffpe; added TMA/FFPE source rules and official file-list→manifest identity/Entity lineage. Preparer self-check complete; independent metadata review and renewed D01 pending.",
        evidence_paths=[relative(JOB / "repair_verification.json")])
    additions = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
        check_id=check, result="unresolved", source_evidence=relative(META / "source_evidence.json"),
        output_evidence=relative(OUTPUT), details=json.dumps(note, ensure_ascii=False)) for check in ("M02", "D01")]
    review = pd.concat([review, pd.DataFrame(additions)], ignore_index=True)
    for name, table in zip(names, (samples, fields, review)):
        table.to_csv(TEMP / (name + ".repair"), index=False, encoding="utf-8-sig")
    publish_json(JOB / "progress.json", dict(study=STUDY, phase="repair", scientific_repair_round=1,
        stage="metadata_staged", exported_images=len(images), sample_rows=len(samples),
        next_step="Publish sample_metadata.csv, field_mapping.csv and pending review records",
        local_closed_tables=[str(TEMP / (name + ".repair")) for name in names]))
    # Publish only the affected tables after comparing their read versions under the study lock.
    with formal_publish_lease(OUTPUT / names[0], directory=OUTPUT):
        if any((OUTPUT / name).read_bytes() != original[name] for name in names):
            raise ValueError("Metadata changed during repair; reread before retry")
        JOB.mkdir(parents=True, exist_ok=True)
        for name in names:
            backup = JOB / (name + ".before")
            if not backup.exists():
                backup.write_bytes(original[name])
        for name in names:
            partial = OUTPUT / (name + ".partial")
            with (TEMP / (name + ".repair")).open("rb") as incoming, partial.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            os.replace(partial, OUTPUT / name)
        directory_fd = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
        actual = pd.read_csv(OUTPUT / names[0], dtype=str, keep_default_na=False)
        actual_fields = pd.read_csv(OUTPUT / names[1], dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(samples, actual)
        pd.testing.assert_frame_equal(fields, actual_fields)
        assert all([p.stat().st_size, p.stat().st_mtime_ns] == versions[relative(p)] for p in protected)
        assert not actual.sample_id.duplicated().any() and images.sample_id.isin(actual.sample_id).all()
        assert actual.preservation_method.eq("ffpe").all()
    state = dict(study=STUDY, phase="repair", scientific_repair_round=1,
        stage="metadata_repaired_pending_independent_review", expected_images=EXPECTED_IMAGES,
        exported_images=len(images), sample_rows=len(actual), sample_linked_images=len(images),
        patient_linked_samples=int(actual.patient_key.ne("").sum()), patient_linked_images=len(images),
        distinct_patient_keys=int(actual.patient_key.nunique()), protected_versions=versions,
        estimated_bytes=estimate, free_bytes=free, updated_utc=datetime.now(timezone.utc).isoformat(),
        next_step="Independent M02/metadata dependency review and renewed D01; no pixel re-export")
    publish_json(JOB / "repair_verification.json", state)
    publish_json(JOB / "progress.json", state)
    print(json.dumps({k: v for k, v in state.items() if k != "protected_versions"}, ensure_ascii=False), flush=True)


def run(max_images: int | None) -> None:
    scope, names = source_scope()
    panel = panel_from_s1(STUDY, PANEL_ID, names, S1)
    if len(panel) != 56 or int(panel.keep_channel.sum()) != 56 or int(panel.nuclear.sum()) != 1:
        raise ValueError("Current S1 no longer selects the proven TMA panel")
    samples, fields = metadata_tables(scope)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_input_tables(OUTPUT, pd.DataFrame(columns=columns("image_metadata")), samples, panel, fields)
    with formal_read_lease(OUTPUT):
        current = pd.read_parquet(OUTPUT / "images.parquet")
        for path in current.input_image_path:
            published = ROOT / path
            if not published.is_file() or published.stat().st_size == 0:
                raise ValueError(f"Indexed image has no complete TIFF: {path}")
    done = set(current.image_id)
    count = 0
    source_failures = []
    for item in scope:
        image_id = item["image_id"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in done:
            continue
        if max_images is not None and count >= max_images:
            break
        try:
            stage = stage_source(item)
            source = SourceSpec(image_id, PANEL_ID, "zarr", stage / "_image_raw",
                                tuple(range(56)), source_revision=item["revision"])
            with formal_read_lease(destination):
                published = destination.is_file()
            if published:
                result = dict(height_px=3000, width_px=3000, pad_left_px=0,
                              pad_right_px=0, pad_top_px=0, pad_bottom_px=0,
                              source_channel_count=56, channel_count=56)
            else:
                with open_source(source) as reader:
                    result = export_input(reader, panel, destination)
                if not result["admitted"]:
                    raise ValueError(f"All retained protein channels are zero: {image_id}")
            check_pixels(item, destination, panel, stage)
            save_input_tables(OUTPUT, pd.DataFrame([image_row(item, result, destination)]), samples, panel, fields)
            done.add(image_id)
            count += 1
            shutil.rmtree(stage)
            progress("image_completed", last_image=image_id,
                     next_image=next((x["image_id"] for x in scope if x["image_id"] not in done), None))
        except zipfile.BadZipFile as error:
            # A failed ZIP CRC invalidates this source despite its earlier native proof.
            failure = dict(image_id=image_id, source=relative(item["archive"]),
                           source_revision=item["revision"], result="failed_archive_read",
                           error=str(error),
                           local_source_partial=str(TEMP / (image_id + ".partial")))
            source_failures.append(failure)
            publish_json(OUTPUT / "source_checks" / f"{image_id}.json", failure)
            progress("source_not_qualified", failed_image=image_id, source_failure=failure)
        except OSError as error:
            if error.errno not in {5, 28, 110, 116}:
                raise
            checkpoint = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / \
                "publish_checkpoints" / metadata_source_relative(destination.resolve(), '/')
            progress("waiting_io", next_image=image_id, issue=f"errno {error.errno}: {error.strerror}",
                     local_source_stage=str(TEMP / image_id),
                     local_output_checkpoint=str(checkpoint / "complete.json"),
                     local_closed_input=str(checkpoint / "input.tiff"),
                     formal_partial=str(destination.with_name("input.tiff.partial")))
            write_review(False, source_failures)
            return
    write_review(len(done) == len(scope), source_failures)
    if len(done) != len(scope):
        progress("partial_batch", source_failures=source_failures,
                 next_image=next(x["image_id"] for x in scope if x["image_id"] not in done))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--repair-metadata", action="store_true")
    arguments = parser.parse_args()
    if arguments.repair_metadata:
        repair_metadata()
    else:
        run(arguments.max_images)
