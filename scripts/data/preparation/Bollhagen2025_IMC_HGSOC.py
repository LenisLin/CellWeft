"""Prepare the 15 admitted public HR-IMC acquisitions at published intensity.

Inputs: Zenodo 17077712 native TIFFs, accepted native proofs, source panels,
        study-specific S1 rows, cached paper and author analysis repository.
Outputs: independent CYX input.tiff files and complete study input tables.
Run exports through the batch shared formal write lease wrapper; verification uses read.
The paper establishes one ex vivo specimen/donor. In situ specimen joins remain
unresolved. Treatment labels describe ex vivo culture conditions, not therapy.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables, validate_image_metadata,
)

STUDY = "Bollhagen2025_IMC_HGSOC"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0016_prepare_0"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
AUTHOR = META / "source_metadata/author_repository/HR_IMC-main"
PAPER_URL = "https://www.nature.com/articles/s41592-025-02889-8"
REPO_URL = "https://github.com/BodenmillerGroup/HR_IMC"
GAPS = ["3 in situ images lack image-to-biological-specimen and patient correspondence",
        "Only one ex vivo donor is confirmed; full-cohort patient eligibility requires controller assessment",
        "Independent review and D01 acceptance are pending"]
PREPARER = "codex_cli:0016_prepare_0"
CONDITION_CODES = {"Carboplatin": "C", "control": "CP", "DMSO": "DMSO"}


def extension_value_mapping(field: str) -> str:
    """Express the three reviewed categorical conversions as JSON objects."""
    mapping = (CONDITION_CODES if field == "author_condition_code" else
               {value: value for value in CONDITION_CODES} if field == "source_condition" else
               {f"{number:03d}": f"{number:03d}" for number in range(1, 5)})
    return json.dumps(mapping)


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    """Use the NAS-relative path convention for every delivered file reference."""
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def progress(stage: str, images: pd.DataFrame, **extra: object) -> None:
    """Persist actual index coverage after each completed acquisition."""
    write_json(JOB / "progress.json", dict(
        study=STUDY, stage=stage, expected_images=len(acquisitions),
        completed_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.notna().sum()),
        gaps=GAPS, updated_at=datetime.now(timezone.utc).isoformat(), **extra))


def verify_header(path: Path, kept: pd.DataFrame, shape: tuple[int, int],
                  dtype: np.dtype) -> None:
    """Reopen every native-intensity output and check the actual OME channel order."""
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].shape == (len(kept), *shape), path
        assert tif.series[0].axes == "CYX" and tif.series[0].dtype == dtype, path
        root = ET.fromstring(tif.ome_metadata)
        channels = root.findall(".//{*}Channel")
        assert [channel.attrib["Name"] for channel in channels] == kept.marker_id.tolist(), path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("metadata", "export", "verify", "finalize", "trace-rds", "repair"), required=True)
parser.add_argument("--job-dir", type=Path)
parser.add_argument(
    "--scope-file", type=Path,
    help="JSON scope manifest containing the exact acquisition IDs to prepare; leaves the full study output unchanged.",
)
parser.add_argument(
    "--output-root", type=Path,
    help="Optional separate output directory for a scoped rerun; required with --scope-file to avoid replacing the full study.",
)
args = parser.parse_args()
if args.job_dir is not None:
    JOB = args.job_dir
elif args.stage in {"trace-rds", "repair"}:
    JOB = JOB.parent / "0148_repair_1"
if args.scope_file is not None and args.output_root is None:
    raise ValueError("--scope-file requires --output-root so the full-study output cannot be overwritten")
if args.output_root is not None:
    OUTPUT = args.output_root
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def repair_progress(stage: str, **extra: object) -> None:
    """Keep repair progress separate from the original preparation job."""
    state = json.loads((JOB / "progress.json").read_text()) if (JOB / "progress.json").exists() else {}
    state.update(study=STUDY, stage=stage, expected_images=15, completed_images=15,
                 sample_rows=1, sample_linked_images=12, patient_linked_samples=1,
                 gaps=GAPS,
                 updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    write_json(JOB / "progress.json", state)


# Inspect serialized SCE slots directly, retaining metadata and streaming assays.
# rdata 1.1.0 and its xarray dependency are cached only within this study.
# This source-specific adapter can migrate to shared code after controller review.
if args.stage == "trace-rds":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/dependencies/Bollhagen2025_IMC_HGSOC/rds_parser_package"))
    import rdata
    from rdata.parser._xdr import ParserXDR
    from rdata.parser import RObjectType

    class MetadataParser(ParserXDR):
        """Parse R structure from gzip/ZIP without materializing assay matrices."""

        def __init__(self, stream):
            super().__init__(memoryview(b""), expand_altrep=False, altrep_constructor_dict={})
            self.file = stream
            self.skipped_arrays = []

        def _parse_array_values(self, dtype, length):
            dtype = np.dtype(dtype)
            # The per-pixel 42-channel assays exceed metadata vector lengths.
            # Drain these arrays in 8 MiB pieces; retain every slot and attribute.
            if length > 10_000_000:
                remaining = length * dtype.itemsize
                self.skipped_arrays.append(dict(length=length, dtype=str(dtype)))
                repair_progress("rds_streaming_assay", skipped_arrays=self.skipped_arrays)
                while remaining:
                    block = self.file.read(min(remaining, 8 * 1024**2))
                    if not block:
                        raise EOFError("Truncated serialized assay")
                    remaining -= len(block)
                return np.empty(0, dtype=dtype)
            return super()._parse_array_values(dtype, length)

        def _parse_vector_value(self, reference_list, bytecode_rep_list):
            length = self.parse_int()
            if length <= 4096:
                return [self.parse_R_object(reference_list, bytecode_rep_list) for _ in range(length)]
            counts = Counter()
            for _ in range(length):
                item = self.parse_R_object(reference_list, bytecode_rep_list)
                assert item.info.type == RObjectType.CHAR, item.info.type
                counts[None if item.value is None else item.value.decode("utf-8")] += 1
            summary = dict(length=length, unique_count=len(counts), examples=list(counts)[:8])
            if len(counts) <= 100:
                summary["value_counts"] = dict(counts)
            repair_progress("rds_string_metadata", last_string_vector=summary)
            return summary

    def attributes(node):
        """Decode the S4 slot/attribute pairlist, preserving exact source keys."""
        result = {}
        while node is not None and node.info.type == RObjectType.LIST:
            tag = node.tag
            while tag.info.type == RObjectType.REF:
                tag = tag.referenced_object
            result[tag.value.value.decode("utf-8")] = node.value[0]
            node = node.value[1]
        return result

    def summarize(node):
        """Inventory all metadata slots and summarize repeated per-pixel values."""
        if node.info.type == RObjectType.REF:
            return summarize(node.referenced_object)
        attrs = {key: summarize(value) for key, value in attributes(node.attributes).items()}
        if isinstance(node.value, np.ndarray):
            value = node.value.tolist() if node.value.size <= 20 else dict(
                length=node.value.size, examples=node.value[:8].tolist())
        elif node.info.type == RObjectType.CHAR:
            value = node.value.decode("utf-8") if node.value is not None else None
        elif node.info.type in {RObjectType.STR, RObjectType.VEC, RObjectType.EXPR, RObjectType.ALTREP}:
            value = node.value if isinstance(node.value, dict) else [summarize(item) for item in node.value]
        elif node.info.type == RObjectType.SYM:
            value = summarize(node.value)
        elif node.value is None:
            value = None
        else:
            raise ValueError(f"Uninventoried R type: {node.info.type}")
        return dict(r_type=node.info.type.name, attributes=attrs, value=value)

    manifest = json.loads((META / "source_manifest.json").read_text())
    archive = NAS / manifest["archive_rel"]
    archive_proof = json.loads((META / "archive_directory.json").read_text())
    assert archive.stat().st_size == archive_proof["archive_file_stat"]["size"]
    assert archive.stat().st_mtime_ns == archive_proof["archive_file_stat"]["mtime_ns"]
    member = "Zenodo/Figure_4/sce.RDS"
    repair_progress("rds_read_started", archive_member=member)
    with zipfile.ZipFile(archive) as source_zip, source_zip.open(member) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
        assert stream.read(2) == b"X\n"
        reader = MetadataParser(stream)
        parsed = reader.parse_all()
        reader.check_complete()
        summary = summarize(parsed.object)
        result = dict(archive=relative(archive), archive_member=member,
            member_bytes=source_zip.getinfo(member).file_size, parser=f"rdata {rdata.__version__}",
            method="Complete gzip/XDR structure parse; numeric assay vectors >10M drained; repeated strings counted",
            format_version=parsed.versions.format, complete_serialization=True,
            skipped_numeric_arrays=reader.skipped_arrays, sce=summary)
    write_json(CACHE / "figure4_sce_metadata.json", result)
    repair_progress("rds_metadata_parsed", rds_evidence=relative(CACHE / "figure4_sce_metadata.json"))
    print(json.dumps(dict(stage="rds_metadata_parsed", evidence=relative(CACHE / "figure4_sce_metadata.json"))))
    raise SystemExit(0)


# Repair reviewed mappings in place; retain the independent review history and
# all unrelated table/file revisions. The next CLI context performs re-review.
if args.stage == "repair":
    repair_path = JOB / "repair_evidence.json"
    previous_repair = json.loads(repair_path.read_text()) if repair_path.exists() else {}
    images = pd.read_parquet(OUTPUT / "images.parquet")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str)
    panels = pd.read_csv(OUTPUT / "panel_mapping.csv")
    mappings = pd.read_csv(OUTPUT / "field_mapping.csv", keep_default_na=False)
    protected = [OUTPUT / name for name in ("images.parquet", "sample_metadata.csv", "panel_mapping.csv")]
    protected += [NAS / path for path in images.input_image_path]
    revisions = {relative(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) for path in protected}
    changes = []
    for field in ("source_condition", "author_condition_code", "source_replicate_label"):
        selected = (mappings.target_table == "image_metadata") & (mappings.target_field == field)
        assert selected.sum() == 1
        original = mappings.loc[selected, "value_mapping"].iloc[0]
        converted = extension_value_mapping(field)
        mappings.loc[selected, "value_mapping"] = converted
        if field != "author_condition_code":
            mappings.loc[selected, "transformation_rule"] = "Preserve source categorical value verbatim; in situ fields empty"
        conversion = json.loads(converted)
        source_field = "source_condition" if field == "author_condition_code" else field
        joined = images[[source_field, field]] if source_field != field else images[[field]]
        for _, row in joined.dropna().iterrows():
            assert conversion[row[source_field]] == row[field]
        changes.append(dict(target_field=field, previous_value_mapping=original, value_mapping=conversion))
    for value in mappings.value_mapping:
        if value:
            assert isinstance(json.loads(value), dict)

    # The archived SCE contains per-pixel image/cell identity and morphology.
    # Its fully parsed metadata adds no biological specimen or patient join.
    rds_path = CACHE / "figure4_sce_metadata.json"
    rds = json.loads(rds_path.read_text())
    assert rds["complete_serialization"]
    slots = rds["sce"]["attributes"]
    col_data = slots["colData"]["attributes"]["listData"]
    columns = [item["value"] for item in col_data["attributes"]["names"]["value"]]
    assert columns == ["Center_X", "Center_Y", "image", "cell", "Area", "major_axis", "minor_axis", "eccentricity"]
    values = dict(zip(columns, col_data["value"]))
    missing_images = images.loc[images.sample_id.isna(), "image_id"].tolist()
    assert set(values["image"]["value"]["value_counts"]) == set(missing_images)
    internal_columns = slots["int_colData"]["attributes"]["listData"]
    internal_names = [item["value"] for item in internal_columns["attributes"]["names"]["value"]]
    assert internal_names == ["reducedDims", "altExps", "colPairs"]
    assert all(item["attributes"]["listData"]["value"] == [] for item in internal_columns["value"])
    assert slots["metadata"]["value"] == []
    internal_metadata = [item["value"] for item in slots["int_metadata"]["attributes"]["names"]["value"]]
    assert set(internal_metadata) == {"version", "cofactor"}
    findings_path = CACHE / "figure4_sce_identity_findings.json"
    findings = dict(source_archive=rds["archive"], archive_member=rds["archive_member"],
        parsed_structure=relative(rds_path), complete_serialization=True,
        colData_fields=columns, image_value_counts=values["image"]["value"]["value_counts"],
        pixel_rows=slots["colData"]["attributes"]["nrows"]["value"][0],
        cell_label_summary=values["cell"]["value"],
        internal_colData_fields=internal_names, internal_colData_payloads_empty=True,
        metadata_empty=True, internal_metadata_fields=internal_metadata,
        join_keys="sce.colData.image -> exact acquisition_id; cell is a segmented object label",
        specimen_patient_join="No biological specimen or patient correspondence provided in the inspected SCE metadata",
        retained_identity_gap_image_ids=missing_images,
        sample_linked_images=12, sample_rows=1, patient_linked_samples=1,
        interpretation="ROI labels and cell labels do not establish biological sampling or patient identity")
    write_json(findings_path, findings)
    evidence_path = CACHE / "source_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["figure4_sce_metadata_inspection"] = dict(evidence=relative(findings_path), **findings)
    evidence["unresolved"]["gap"] = (
        "Three in situ ROIs have no established biological specimen/patient correspondence. "
        "The entire Figure 4 SCE serialization was parsed; colData has image/cell/coordinate/morphology "
        "fields, metadata is empty, and internal metadata has package versions/cofactor only.")
    write_json(evidence_path, evidence)
    for table, field in [("image_metadata", "sample_id"), ("sample_metadata", "patient_key")]:
        selected = (mappings.target_table == table) & (mappings.target_field == field)
        mappings.loc[selected, "source_file"] = relative(evidence_path)
        mappings.loc[selected, "notes"] = (
            "12/15 images -> one biological specimen -> one documented donor. "
            "Three in situ ROI joins remain unresolved after full SCE metadata inspection; "
            "exact sce.colData.image -> acquisition_id identifies ROIs only. Sources and coverage: "
            + relative(findings_path) + "; " + relative(evidence_path))
    validate_image_metadata(images, samples, panels, mappings)
    repair_progress("mapping_repair_validated")
    temporary = OUTPUT / "field_mapping.csv.tmp"
    with nullcontext():
        with formal_publish_lease(OUTPUT / "field_mapping.csv", directory=OUTPUT):
            mappings.to_csv(temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, OUTPUT / "field_mapping.csv")
        reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
        own_rows = (reviews.preparer == "codex_cli:0148_repair_1") & (reviews.reviewer == "")
        existing_reviews = reviews.loc[~own_rows].copy()
        repair_rows = []
        summaries = {
            "M01": "已完整解析Figure_4/sce.RDS结构及metadata；8列colData仅含坐标、ROI、cell和形态，metadata为空，内部metadata为版本/cofactor。3个in situ标本/患者连接仍缺；12/15图连接1标本及1供者。",
            "M02": "已将3条value_mapping修为JSON对象，逐值重建分类转换及完整image metadata校验通过；待不同CLI上下文独立复查。",
            "D01": "修补轮次1实际完成：15/15 TIFF复用，image/sample/panel索引未改；3条映射及2条身份来源说明更新。3图身份缺口及队列人数准入交主控；未签发input_binding，待独立复查。",
        }
        for check_id, summary in summaries.items():
            repair_rows.append(dict(study=STUDY, preparer="codex_cli:0148_repair_1", reviewer="",
                scope="study", record_id=STUDY, check_id=check_id, result="unresolved",
                source_evidence=relative(findings_path), output_evidence=relative(repair_path),
                details=json.dumps(dict(review_round=1, preparer_self_check=True,
                    independent_review_pending=True, summary=summary,
                    image_ids=missing_images if check_id == "M01" else images.image_id.tolist(),
                    sample_linked_images=12, sample_rows=1, patient_linked_samples=1), ensure_ascii=False)))
        reviews = pd.concat([existing_reviews, pd.DataFrame(repair_rows)], ignore_index=True)
        review_temporary = OUTPUT / "review_results.csv.tmp"
        with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
            reviews.to_csv(review_temporary, index=False, encoding="utf-8-sig")
            os.replace(review_temporary, OUTPUT / "review_results.csv")
        published_reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(published_reviews.iloc[:len(existing_reviews)], existing_reviews.reset_index(drop=True))
    assert revisions == {relative(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns) for path in protected}
    write_json(repair_path, dict(study=STUDY, repair_round=1,
        preparer="codex_cli:0148_repair_1", independent_review=False,
        mapping_changes=previous_repair.get("mapping_changes", changes),
        rds_identity_findings=relative(findings_path),
        all_nonempty_value_mappings_are_json_objects=True,
        actual_image_values_reconstructed=True, protected_revisions_unchanged=revisions,
        independent_review_history_preserved=True, retained_review_rows=len(existing_reviews),
        preparer_repair_rows_added=len(repair_rows), input_binding_issued=False))
    repair_progress("repaired_pending_independent_review", mapping_rows_repaired=3,
                    identity_field_notes_updated=2, rds_identity_findings=relative(findings_path))
    print(json.dumps(dict(stage="mapping_repaired", exported_images=len(images), tiff_files_rewritten=0)))
    raise SystemExit(0)


# Bind the accepted acquisition scope to the complete archive and existing proofs.
manifest = json.loads((META / "source_manifest.json").read_text())
native_verification = json.loads((META / "acquisition_verification.json").read_text())
channel_mapping = json.loads((META / "channel_mapping.json").read_text())
acquisitions = manifest["acquisitions"]
assert {a["acquisition_id"] for a in acquisitions} == set(native_verification["verified_acquisition_ids"])
if args.scope_file is not None:
    scope = json.loads(args.scope_file.read_text())
    scope_ids = scope.get("image_ids")
    if not isinstance(scope_ids, list) or not scope_ids or len(scope_ids) != len(set(scope_ids)):
        raise ValueError("Scope manifest must contain a non-empty unique image_ids list")
    known_ids = {a["acquisition_id"] for a in acquisitions}
    unknown_ids = set(scope_ids) - known_ids
    if unknown_ids:
        raise ValueError(f"Scope contains unknown acquisition IDs: {sorted(unknown_ids)}")
    acquisitions = [a for a in acquisitions if a["acquisition_id"] in set(scope_ids)]
    if {a["acquisition_id"] for a in acquisitions} != set(scope_ids):
        raise ValueError("Scope manifest did not resolve every requested acquisition")
archive = NAS / manifest["archive_rel"]
archive_proof_path = META / "archive_directory.json"
archive_proof = json.loads(archive_proof_path.read_text())
assert archive.stat().st_size == archive_proof["archive_file_stat"]["size"]
assert archive.stat().st_mtime_ns == archive_proof["archive_file_stat"]["mtime_ns"]
headers = {}
inventory = []
metadata_headers = []
with zipfile.ZipFile(archive) as source_zip:
    accepted_members = {a["archive_member"] for a in acquisitions}
    for member in source_zip.infolist():
        name = member.filename
        if name in accepted_members:
            decision = "admitted independent HGSOC intensity acquisition"
        elif member.is_dir():
            decision = "directory"
        elif name.startswith(("Zenodo/Figure_3/", "Zenodo/Figure_4/")):
            decision = ("duplicate/downsampled display variant" if "/img/" in name or "/example_img_" in name
                        else "associated mask, panel or derived analysis; no additional acquisition")
        else:
            decision = "outside accepted HGSOC cohort: other tissue or technical method assessment"
        inventory.append(dict(archive_member=name, uncompressed_bytes=member.file_size,
                              compressed_bytes=member.compress_size, decision=decision))
        if name.startswith(("Zenodo/Figure_3/", "Zenodo/Figure_4/")) and name.endswith(".csv"):
            with source_zip.open(member) as stream:
                header = stream.readline().decode("utf-8-sig").strip()
            metadata_headers.append(dict(archive_member=name, header=header))
    assert len(inventory) == archive_proof["zip64_entry_count"]
    for acquisition in acquisitions:
        path = NAS / acquisition["native_rel"]
        proof = json.loads((NAS / acquisition["proof_rel"]).read_text())
        stat = path.stat()
        assert proof["complete"] and proof["native_order_verified"] and proof["source_order_verified"]
        assert stat.st_size == proof["native_file_stat"]["size"]
        assert stat.st_mtime_ns == proof["native_file_stat"]["mtime_ns"]
        assert acquisition["archive_member"] in source_zip.namelist()
        with tifffile.TiffFile(path) as tif:
            shape = tif.series[0].shape
            assert len(tif.pages) == shape[0] == proof["native_channel_count"] == 42
            headers[acquisition["acquisition_id"]] = dict(shape=list(shape),
                axes=tif.series[0].axes, dtype=str(tif.series[0].dtype))
    panel_frames = []
    for source_panel in channel_mapping["panels"]:
        raw_panel = pd.read_csv(NAS / source_panel["source_panel_path"], keep_default_na=False)
        # The accepted source comparison ignores CR/LF differences only.
        archived_text = source_zip.read(source_panel["source_panel_member"]).decode("utf-8")
        assert archived_text.splitlines() == (NAS / source_panel["source_panel_path"]).read_text().splitlines()
        assert raw_panel.name.tolist() == [c["marker"] for c in source_panel["ordered_channels"]]
        current = panel_from_s1(STUDY, source_panel["panel_id"], raw_panel.name.tolist(), S1)
        panel_frames.append(current)
panels = pd.concat(panel_frames, ignore_index=True)
if args.scope_file is not None:
    panels = panels[panels.panel_id.isin({a["panel_id"] for a in acquisitions})].reset_index(drop=True)
pd.DataFrame(inventory).to_csv(CACHE / "archive_inventory.csv", index=False)
write_json(CACHE / "archive_metadata_headers.json", metadata_headers)

# Trace biological sampling through the paper, reporting summary and author code.
# These cached sources were fetched from the public publisher/repository endpoints.
required_sources = [META / "source_metadata/article.txt", META / "source_metadata/reporting_summary.pdf",
                    META / "source_metadata/paper_antibody_table.csv",
                    AUTHOR / "Scripts/Figure_3/ex_vivo_analysis.Rmd",
                    AUTHOR / "Scripts/Figure_4/In_situ_sce_generation.Rmd"]
for path in required_sources:
    assert path.exists(), path
paper = (META / "source_metadata/article.txt").read_text()
assert "The tissue sample from a patient with high-grade serous ovarian cancer" in paper
ex_vivo = [a for a in acquisitions if a["panel_id"] == "figure3_high_res"]
source_specimens = {re.fullmatch(r".+_ExVivo_(.+)_(Carboplatin|control|DMSO)_\d+",
                               a["acquisition_id"]).group(1) for a in ex_vivo}
assert len(source_specimens) == 1
sample_id = "ex_vivo_specimen_01"
sample = {column: pd.NA for column in pd.read_csv(TEMPLATES / "sample_metadata.csv").columns}
sample.update(study=STUDY, sample_id=sample_id, patient_key="ex_vivo_donor_01",
              source_sample_id=next(iter(source_specimens)),
              sample_type="patient-derived ovarian cancer 3D culture",
              tissue_type="tumor", processing_method="3D ex vivo culture",
              primary_diagnosis="high-grade serous ovarian cancer",
              primary_site="ovary", histology="high-grade serous carcinoma",
              case_control_status="case", sex_reported="female",
              metadata_notes=("One biological surgical specimen and one donor established by paper Methods; "
                "12 images are technical replicates across culture conditions (Reporting Summary p2). "
                "Generated donor key expresses the single documented donor; no public donor ID. "
                "Source specimen token joins the Figure 3 acquisition series, not a patient-name inference. "
                "Age, stage, outcome and in situ correspondence are unavailable. Culture treatment is not clinical therapy."))
samples = pd.DataFrame([sample])
if (OUTPUT / "sample_metadata.csv").exists():
    previous = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str).replace("", pd.NA)
    samples = pd.concat([previous[previous.sample_id != sample_id], samples], ignore_index=True)

image_fields = list(pd.read_csv(TEMPLATES / "image_metadata.csv").columns)
images = (pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists()
          else pd.DataFrame(columns=image_fields))
evidence = dict(
    study=STUDY, paper_url=PAPER_URL, record_url="https://zenodo.org/records/17077712",
    author_repository=REPO_URL, expected_images=15, existing_verification=relative(META / "acquisition_verification.json"),
    s1_mapping=relative(S1), ordered_panel_metadata=relative(META / "channel_mapping.json"),
    proof_reuse="size and mtime_ns match accepted native proofs; no rehash",
    archive_inventory=relative(CACHE / "archive_inventory.csv"),
    archive_metadata_headers=relative(CACHE / "archive_metadata_headers.json"),
    sample_join=dict(image_ids=[a["acquisition_id"] for a in ex_vivo], sample_id=sample_id,
        source_sample_id=next(iter(source_specimens)), patient_key="ex_vivo_donor_01",
        basis=["Paper Methods: one surgical tissue sample from one HGSOC patient",
               "Reporting Summary p2: four technical replicates per ex vivo condition; intra-patient heterogeneity",
               "Author Figure 3 ex_vivo_analysis.Rmd lines 53-61 binds exact image IDs to culture conditions"],
        coverage=dict(sample_linked_images=12, sample_rows=1, patient_linked_samples=1)),
    unresolved=dict(image_ids=[a["acquisition_id"] for a in acquisitions if a["panel_id"] == "figure4_insitu"],
        gap="Paper Figure 4 confirms 3 ROIs but gives no biological specimen IDs or patient correspondence; "
            "archive attachments and author in situ SCE generation supply image/cell identities only"),
    patient_eligibility="Only one ex vivo donor is established. In situ donor identity is unavailable; "
        "the source does not substantiate a >=5-patient claim. Preserve the authorized 15-image scope and refer study-level eligibility to the controller.",
    reporting_summary=dict(url="https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fs41592-025-02889-8/MediaObjects/41592_2025_2889_MOESM1_ESM.pdf",
        path=relative(META / "source_metadata/reporting_summary.pdf"),
        inspection="PDF p2 and p3 rendered and visually read by preparer; text layer is empty",
        facts=["ovarian cancer samples from female individuals", "no population characteristic information collected",
               "recruitment within a larger unpublished ex vivo study", "clinical data marked not applicable"]),
    supplementary_table=dict(path=relative(META / "source_metadata/paper_antibody_table.csv"),
        content="antibody identities and staining concentrations; no clinical or sample map"),
    repository_access=dict(api_attempt="HTTP 403 rate limit", recovery="Public codeload ZIP downloaded; full repository tree checked and relevant Figure 3/4 analysis inspected; no separate clinical/sample table"),
    pixel_size_um=0.333, pixel_size_basis="reported 333 nm pitch in paper Figure 3/4 and author HR-IMC workflow; TIFF resolution tags have no physical unit",
    condition_semantics=dict(Carboplatin="author code C", control="author code CP", DMSO="author code DMSO",
        note="Keep source labels; paper describes carboplatin/paclitaxel combination. Do not interpret control as untreated."),
    source_preprocessing="Published uint16 HR-IMC TIFFs; article describes hot-pixel filtering and RL deconvolution. "
        "File-specific processing lineage not supplied; preserve published values.",
    controller_requests=["Resolve the three in situ image-to-specimen/patient joins if additional local source correspondence exists; preserve 15 image outputs.",
                         "Reassess study-level patient eligibility: ex vivo is one documented donor, in situ patient mapping is absent; no >=5-patient claim is supported.",
                         "Arrange independent review; preparer does not issue D01 acceptance."])
write_json(CACHE / "source_evidence.json", evidence)

# Document source, join and missingness for every template field.
mapping_rows = []
for table, columns in [("image_metadata", image_fields), ("sample_metadata", list(samples.columns))]:
    for field in columns:
        mapping_rows.append(dict(study=STUDY, source_file=relative(META / "source_manifest.json"),
            source_sheet_or_table="acquisitions; accepted native proof; paper and author metadata",
            source_field=field, target_table=table, target_field=field,
            join_keys="acquisition_id -> panel_id; Figure 3 acquisition series -> documented single specimen -> single donor",
            value_mapping="", transformation_rule="Preserve source value or leave unavailable optional field empty",
            time_origin_rule="No clinical timeline inferred from acquisition date or culture treatment",
            conflict_rule="Explicit author code condition overrides informal interpretation; preserve original label",
            notes="Detailed source and join coverage: " + relative(CACHE / "source_evidence.json")))
field_mapping = pd.DataFrame(mapping_rows)
image_rules = {
    "study": ("dataset_name", "Use canonical study identity from accepted source manifest"),
    "image_id": ("acquisition_id", "Preserve exact accepted acquisition identity"),
    "sample_id": ("Figure 3 acquisition series + paper single-specimen statement", "12 ex vivo ROI images share one specimen; 3 in situ values empty"),
    "panel_id": ("acquisitions.panel_id", "Exact manifest panel key; Figure 3 and Figure 4 remain separate"),
    "image_kind": ("archive intensity ROI member; paper Figure 4 ROI statement", "roi"),
    "source_image_id": ("archive_member", "Exact full archive member identifies the source object"),
    "source_files": ("native_rel; archive_rel", "Ordered NAS-relative native TIFF and original archive paths"),
    "source_format": ("native_rel TIFF header", "tiff"),
    "source_axes": ("native TIFF series.axes", "QYX: source page/Q axis is the ordered 42-channel axis"),
    "source_dtype": ("native TIFF series.dtype", "uint16; preserve decoded dtype and values"),
    "source_revision": ("source.record_id; acquisition_verification.json", "Zenodo record 17077712 with accepted archive/native proof binding"),
    "roi_id": ("roi_label; acquisition_id", "Preserve source in situ ROI label; ex vivo uses full acquisition ID"),
    "region_annotation": ("condition", "Preserve source ex vivo condition exactly; in situ annotation absent"),
    "source_x0_px": ("native TIFF image coordinate origin", "0; full native acquisition"),
    "source_y0_px": ("native TIFF image coordinate origin", "0; full native acquisition"),
    "source_x1_px": ("native TIFF series.shape[-1]", "Full source width; half-open right edge"),
    "source_y1_px": ("native TIFF series.shape[-2]", "Full source height; half-open bottom edge"),
    "height_px": ("source height and width", "max(source height, source width)"),
    "width_px": ("source height and width", "max(source height, source width)"),
    "source_channel_count": ("native TIFF pages; accepted ordered source panel", "42"),
    "channel_count": ("panel_mapping.keep_channel", "Count S1-retained rows; 34 per panel"),
    "input_image_path": ("acquisition_id", "processed/<study>/images/<acquisition_id>/input.tiff; NAS relative"),
    "valid_pixels_path": ("complete rectangular TIFF frame", "Empty: entire published source rectangle retained; no unmeasured domain supplied"),
    "intensity_description": ("paper Methods image processing; published TIFF dtype", evidence["source_preprocessing"]),
    "split_method": ("independent archive ROI member", "already_independent"),
    "split_evidence": ("source_manifest and acquisition proof_rel", "NAS-relative source manifest and exact acquisition proof"),
    "metadata_notes": ("source_evidence.json", "Explicit channel axis, reported pixel pitch, and unresolved in situ specimen joins"),
}
for field in ("source_acquisition_id", "source_z_index", "source_t_index", "slide_id", "core_id", "fov_id"):
    image_rules[field] = ("not supplied/applicable in the published independent TIFF", "Leave empty; do not infer physical slide/core or Z/T identities")
for field in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px"):
    image_rules[field] = ("source TIFF dimensions; centered square padding", "All admitted sources are square, so this padding is 0")
for field in ("pixel_size_x_um", "pixel_size_y_um"):
    image_rules[field] = ("paper Figure 3/4 captions; HR-IMC workflow", "0.333 um/pixel, the reported 333 nm pitch; no physical TIFF resolution tag")
for field, (locator, rule) in image_rules.items():
    selected = (field_mapping.target_table == "image_metadata") & (field_mapping.target_field == field)
    field_mapping.loc[selected, ["source_field", "transformation_rule"]] = [locator, rule]
for field in ("source_condition", "author_condition_code", "source_replicate_label"):
    field_mapping.loc[len(field_mapping)] = dict(study=STUDY,
        source_file=relative(AUTHOR / "Scripts/Figure_3/ex_vivo_analysis.Rmd"),
        source_sheet_or_table="lines 53-61; source_manifest acquisitions", source_field=field,
        target_table="image_metadata", target_field=field, join_keys="exact acquisition_id",
        value_mapping=extension_value_mapping(field),
        transformation_rule=("Ex vivo acquisition condition; in situ fields empty" if field == "author_condition_code"
                             else "Preserve source categorical value verbatim; in situ fields empty"),
        time_origin_rule="Ex vivo only; no patient therapy inference", conflict_rule="retain source labels",
        notes=json.dumps(dict(data_type="string", definition={
            "source_condition":"Original ex vivo acquisition condition label",
            "author_condition_code":"Ex vivo condition code assigned by the author analysis script",
            "source_replicate_label":"Source technical replicate label, not biological sample identity"}[field])))

sample_rules = {
    "sample_id": ("paper Methods + Figure 3 acquisition series", "One biological specimen; all 12 technical ROI images share this key"),
    "patient_key": ("paper Methods: singular patient; Reporting Summary p2", "Generated study-local key for the documented single ex vivo donor; 1/1 sample linked; no in situ donor link"),
    "source_sample_id": ("acquisition_id specimen token + paper single-specimen description", "Extract the common source specimen label; anchor grouping to biological statement, not ROI numbering"),
    "sex_reported": ("Reporting Summary PDF p2", "Explicit female participants for ovarian cancer samples"),
    "sample_type": ("paper Methods: Three-dimensional patient-derived ovarian cancer culture", "patient-derived ovarian cancer 3D culture"),
    "processing_method": ("paper Methods: Three-dimensional patient-derived ovarian cancer culture", "3D ex vivo culture"),
    "tissue_type": ("paper Methods and Figure 3 HGSOC culture", "tumor"),
    "primary_diagnosis": ("paper Methods: high-grade serous ovarian cancer", "high-grade serous ovarian cancer"),
    "primary_site": ("paper disease identity", "ovary; collection anatomical site remains empty"),
    "histology": ("paper high-grade serous disease identity", "high-grade serous carcinoma"),
    "case_control_status": ("paper patient-derived HGSOC specimen", "case; distinct from ex vivo control condition"),
}
for field, (locator, rule) in sample_rules.items():
    selected = (field_mapping.target_table == "sample_metadata") & (field_mapping.target_field == field)
    field_mapping.loc[selected, ["source_file", "source_field", "transformation_rule"]] = [
        relative(CACHE / "source_evidence.json"), locator, rule]
for field in samples.columns:
    if pd.isna(sample[field]):
        selected = (field_mapping.target_table == "sample_metadata") & (field_mapping.target_field == field)
        field_mapping.loc[selected, "notes"] = "Source provides no value or identity; optional field remains empty. See source_evidence.json and M01."

stats_path = CACHE / "export_stats.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
progress("metadata_ready", images)
if args.stage == "metadata":
    with formal_publish_lease(OUTPUT, directory=OUTPUT):
        for name, frame in [("sample_metadata", samples), ("panel_mapping", panels), ("field_mapping", field_mapping)]:
            frame.to_csv(OUTPUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(dict(stage="metadata_ready", expected_images=len(acquisitions), samples=len(samples),
                         retained_channels=panels.groupby("panel_id").keep_channel.sum().to_dict())))

# Export one complete native ROI at a time, checkpointing actual published rows.
if args.stage == "export":
    expected_bytes = sum(int(np.prod(headers[a["acquisition_id"]]["shape"][1:])) *
                         int(panels[panels.panel_id == a["panel_id"]].keep_channel.sum()) * 2 for a in acquisitions)
    assert shutil.disk_usage(NAS).free - expected_bytes >= 100 * 1024**3
    assert shutil.disk_usage("/tmp").free >= 64 * 1024**3
    for acquisition in acquisitions:
        image_id = acquisition["acquisition_id"]
        source = SourceSpec(image_id, acquisition["panel_id"], "tiff", NAS / acquisition["native_rel"],
                            tuple(range(42)), pixel_size_um=(0.333, 0.333))
        panel = panels[panels.panel_id == source.panel_id]
        kept = panel[panel.keep_channel].sort_values("output_channel_index")
        _, h, w = headers[image_id]["shape"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in stats and stats[image_id]["admitted"] and destination.exists():
            verify_header(destination, kept, (max(h, w), max(h, w)), np.dtype(headers[image_id]["dtype"]))
            result = stats[image_id]
        else:
            progress("exporting", images, current_image=image_id, estimated_output_bytes=expected_bytes)
            with open_source(source) as reader:
                # All accepted planes fit within 1660x1660; one 2048 tile per
                # channel avoids repeatedly decoding the source's single strip.
                result = export_input(reader, panel, destination, tile_size=2048)
            stats[image_id] = result
            write_json(stats_path, stats)
        if result["admitted"]:
            is_ex_vivo = acquisition["panel_id"] == "figure3_high_res"
            condition = acquisition.get("condition") if is_ex_vivo else pd.NA
            record = {column: pd.NA for column in image_fields}
            record.update(study=STUDY, image_id=image_id, sample_id=sample_id if is_ex_vivo else pd.NA,
                panel_id=source.panel_id, image_kind="roi", source_image_id=acquisition["archive_member"],
                source_files=[acquisition["native_rel"], manifest["archive_rel"]], source_format="tiff",
                source_axes=headers[image_id]["axes"], source_dtype=result["source_dtype"],
                source_revision="Zenodo record 17077712; accepted archive-directory/native proofs",
                roi_id=acquisition.get("roi_label", image_id), region_annotation=condition,
                source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                **{key: result[key] for key in ("pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px",
                                               "height_px", "width_px", "source_channel_count", "channel_count")},
                pixel_size_x_um=0.333, pixel_size_y_um=0.333, input_image_path=relative(destination),
                intensity_description=evidence["source_preprocessing"], split_method="already_independent",
                split_evidence=relative(META / "source_manifest.json") + ";" + acquisition["proof_rel"],
                metadata_notes=("QYX source axis Q is the ordered channel/page axis; scale is the reported 333 nm pitch. " +
                    ("12 technical ROI images share one documented ex vivo specimen/donor." if is_ex_vivo else
                     "Biological specimen and patient correspondence unresolved for 3 in situ ROIs.")))
            record.update(source_condition=condition,
                author_condition_code={"Carboplatin":"C", "control":"CP", "DMSO":"DMSO"}[condition] if is_ex_vivo else pd.NA,
                source_replicate_label=acquisition.get("replicate_label") if is_ex_vivo else pd.NA)
            images = pd.concat([images[images.image_id != image_id], pd.DataFrame([record])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panels, field_mapping)
        progress("export_checkpoint", images, current_image=image_id)
        print(f"{image_id}: admitted={result['admitted']}; indexed={len(images)}", flush=True)
    progress("export_complete", images)

# Verify actual files, exact native pixel blocks, full marker admission, and overview.
if args.stage == "verify":
    checks = []
    overview = Image.new("RGB", (5 * 350, 3 * 380), "white")
    draw = ImageDraw.Draw(overview)
    for number, acquisition in enumerate(acquisitions):
        image_id = acquisition["acquisition_id"]
        result = stats[image_id]
        assert result["admitted"], "All-zero exclusion requires explicit source reconciliation"
        record = images.set_index("image_id").loc[image_id]
        panel = panels[panels.panel_id == acquisition["panel_id"]]
        kept = panel[panel.keep_channel].sort_values("output_channel_index")
        _, h, w = headers[image_id]["shape"]
        verify_header(NAS / record.input_image_path, kept, (h, w), np.dtype(headers[image_id]["dtype"]))
        with tifffile.TiffFile(NAS / acquisition["native_rel"]) as native, tifffile.TiffFile(NAS / record.input_image_path) as prepared:
            for signal_channel in kept.loc[kept.role == "downstream", "source_channel_index"]:
                signal = native.pages[signal_channel].asarray(maxworkers=2)[64:-64, 64:-64]
                if np.any(signal):
                    y, x = np.unravel_index(int(np.argmax(signal)), signal.shape)
                    x, y = int(x)+64, int(y)+64
                    break
            else:
                raise ValueError(f"{image_id}: no interior marker signal for the required pixel comparison")
            blocks = [(x-32, y-32, x+32, y+32), (0, 0, 64, 64)]
            for channel in kept.itertuples():
                plane = native.pages[channel.source_channel_index].asarray(maxworkers=2)
                for window in blocks:
                    x0, y0, x1, y1 = window
                    actual = read_tiff_window(prepared, int(channel.output_channel_index), window)
                    assert actual.dtype == plane.dtype and np.array_equal(actual, plane[y0:y1, x0:x1])
            display = []
            for channel in [2, 4, 37]:
                plane = native.pages[channel].asarray(maxworkers=2).astype(np.float32)
                high = float(np.percentile(plane, 99.5))
                display.append(np.clip(plane / (high or 1), 0, 1))
            rgb = np.stack(display, axis=-1)
            thumbnail = Image.fromarray((rgb * 255).astype(np.uint8)).resize((330, 330))
            x, y = (number % 5)*350, (number // 5)*380
            overview.paste(thumbnail, (x, y))
            draw.text((x+3, y+335), image_id.replace("20240923_JWAB_", ""), fill="black")
        checks.append(dict(image_id=image_id, header="pass", pixels="exact_equal",
            channels_compared=len(kept), windows=[list(b) for b in blocks],
            signal_source_channel_index=int(signal_channel),
            nonzero_downstream_channels=int(np.count_nonzero(np.asarray(result["channel_nonzero_pixels"])[(kept.role == "downstream").to_numpy()])),
            original_shape=[42, h, w], output_shape=[len(kept), h, w], dtype="uint16"))
        progress("verifying", images, verified_images=len(checks), current_image=image_id)
    overview.save(CACHE / "native_overview.png")
    write_json(CACHE / "verification.json", dict(preparer=PREPARER, independent_review=False,
        checked_images=len(checks), checks=checks, preview=relative(CACHE / "native_overview.png"),
        preview_channels="R: epithelial keratin; G: vimentin; B: Ir191; display-only P99.5 scaling",
        sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum())))

    review_rows = []
    rules = pd.read_csv(TEMPLATES / "review_checks.csv")
    scope_count = len(acquisitions)
    scope_sample_linked = int(images.sample_id.notna().sum())
    scope_patient_linked = int(samples.patient_key.notna().sum())
    scope_unresolved = images.loc[images.sample_id.isna(), "image_id"].tolist()
    for rule in rules.itertuples():
        details = {"S01":f"388 archive entries reconciled; {scope_count} scoped admitted images; every retained downstream panel has whole-image nonzero signal",
            "I01":f"{scope_count} unique image identities; exact archive member/native proof and output paths",
            "I02":f"{scope_sample_linked}/{scope_count} scoped images link to the documented biological specimen/donor; {len(scope_unresolved)} scoped image sample links remain empty; panel key valid",
            "I03":f"{scope_count} complete source frames; all padding=0; 0.333 um reported pitch",
            "I04":"Already independent source ROIs; no multi-sample dearray or sample-region masking performed; visual inspection recorded separately",
            "P01":"Both 42-row source panels checked against archive and native proofs; exact study-specific S1 join",
            "P02":"34 retained channels per panel, including both Ir nuclear inputs; downstream marker IDs unique",
            "P03":f"All {scope_count} OME TIFFs reopened; uint16/CYX, channel count/order/names verified",
            "I05":f"Every scoped image, all 34 retained channels: two 64x64 native/output blocks exactly equal with matching dtype; details in verification.json",
            "M01":f"{scope_sample_linked}/{scope_count} scoped images -> 1 biological specimen -> {scope_patient_linked} documented donor link(s). Clinical fields beyond this identity are not required for technical input; independent scope review remains pending",
            "M02":"All filled sample fields reconstructed from cited paper and Reporting Summary p2; unavailable optional template fields remain empty as documented; no clinical therapy or timing inferred",
            "D01":f"{scope_count}/{scope_count} native-intensity files and metadata prepared; {scope_sample_linked} image/sample links and {scope_patient_linked} sample/patient link(s); awaiting independent reviewer"}[rule.check_id]
        if rule.check_id == "M01":
            details = json.dumps(dict(preparer_self_check=True, independent_review_pending=True,
                image_ids=scope_unresolved,
                sample_linked_images=scope_sample_linked, sample_rows=len(samples), patient_linked_samples=scope_patient_linked,
                note=details), ensure_ascii=False)
        else:
            details = "Preparer self-check only; independent review pending. " + details
        review_rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
            record_id=STUDY, check_id=rule.check_id,
            result="unresolved" if rule.check_id in {"I04", "M01", "D01"} else "pass",
            source_evidence=relative(CACHE / "source_evidence.json"), output_evidence=relative(CACHE / "verification.json"),
            details=details))
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding="utf-8-sig")
    save_input_tables(OUTPUT, images, samples, panels, field_mapping)
    progress("prepared_pending_visual_and_independent_review", images, verified_images=len(checks))
    print(json.dumps(dict(stage="prepared", exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum()))))

# Invoke only after the preparer has actually viewed native_overview.png.
if args.stage == "finalize":
    verification = json.loads((CACHE / "verification.json").read_text())
    assert verification["checked_images"] == len(images) == len(acquisitions)
    verification["visual_inspection"] = dict(preparer=PREPARER,
        scope=f"All {len(acquisitions)} complete ROI frames in native_overview.png",
        result="Ex vivo dispersed cells and in situ contiguous tissue support source acquisition scope; "
               "no evidence requiring sample dearray or removal of the full source frame",
        independent_review=False)
    write_json(CACHE / "verification.json", verification)
    reviews = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
    selected = reviews.check_id == "I04"
    reviews.loc[selected, "result"] = "pass"
    reviews.loc[selected, "details"] = (
        "Preparer visual self-check of all 15 full-frame previews: ex vivo cells and in situ tissue; "
        "source identity and complete ROI preserved; no sample-region split required. "
        "Three in situ specimen/patient joins remain unresolved in M01. Independent review pending.")
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        reviews.to_csv(review_path, index=False, encoding="utf-8-sig")
    progress("prepared", images, verified_images=len(images), overview_reviewed=True,
             controller_requests=evidence["controller_requests"])
    print(json.dumps(dict(status="partial", expected_images=len(acquisitions), exported_images=len(images),
        sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(samples.patient_key.notna().sum()))))
