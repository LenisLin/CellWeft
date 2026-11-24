"""Prepare verified HCC TLS MCD acquisitions as native IMC study inputs.

Inputs are the local Zenodo MCD files, their acquisition/channel proof, the
study S1 rows, and public ROI annotations. Outputs are one CYX input per
eligible acquisition plus the study input tables and preparer review trace.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import json
import argparse
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"

import pandas as pd
import numpy as np
import pyarrow as pa
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)

STUDY = "hcc_tls_imc_2024"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "zenodo_10020271"
META = metadata_source_directory(ROOT, STUDY)
OUTPUT = ROOT / "processed" / STUDY
CACHE = META / "source_annotations"
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(
    Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0740_prepare_0")))
PROOF = META / "acquisition_verification.json"
CHANNELS = META / "channel_mapping.json"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12042221/"
ZENODO = "https://zenodo.org/records/10020271"
REVISION = "zenodo_10020271_v1"
EXPECTED = 44

pa.set_cpu_count(2)
pa.set_io_thread_count(2)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def checkpoint(stage: str, indexed: int, next_image: str | None, **details: object) -> None:
    JOB.mkdir(parents=True, exist_ok=True)
    record = dict(study=STUDY, phase="prepare", repair_round=0, stage=stage,
                  expected_images=EXPECTED, indexed_images=indexed, next_image=next_image,
                  updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    partial = JOB / "progress.json.partial"
    partial.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    os.replace(partial, JOB / "progress.json")


def field(source: str, source_field: str, table: str, target: str,
          rule: str, join: str, notes: str = "") -> dict[str, str]:
    return dict(study=STUDY, source_file=source, source_sheet_or_table="",
                source_field=source_field, target_table=table, target_field=target,
                join_keys=join, value_mapping="{}", transformation_rule=rule,
                time_origin_rule="", conflict_rule="", notes=notes)


def patient_from_member(member: str) -> str:
    match = re.fullmatch(r"(?:J\d+_)?(P\d+|OT\d+)[A-Z]?", Path(member).stem)
    if match is None:
        raise ValueError(f"Unrecognized public HCC specimen label: {member}")
    return match.group(1)


def image_id_for(item: dict) -> str:
    return f"{Path(item['source_file']).stem}__acquisition_{int(item['acquisition_id']):03d}"


def source_exclusion(item: dict) -> str | None:
    description = item["native_metadata"]["Description"]
    if description == "ROI_DB" and item["width"] == item["height"] == 50:
        return "50x50 ROI_DB has no tissue ROI annotation; test-sized acquisition pending S01 review"
    if item["width"] == 800 and item["height"] == 36 and description == "ROI_5_1":
        return "36-pixel acquisition fragment; a separate 764-pixel split acquisition is present"
    return None


def public_samples(files: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in files:
        member = entry["source_file"]
        specimen = Path(member).stem
        subject = patient_from_member(member)
        row = dict.fromkeys(columns("sample_metadata"))
        row.update(study=STUDY, sample_id=specimen,
                   source_sample_id=specimen, source_subject_id=subject,
                   patient_key=f"{STUDY}__{subject}", sample_type="resected tumor section",
                   tissue_type="tumor", anatomical_site="liver",
                   primary_diagnosis="hepatocellular carcinoma",
                   case_control_status="case", preservation_method="ffpe",
                   collection_method="surgical resection",
                   metadata_notes=("Source MCD member identifies a tumor section. The public paper "
                                   "identifies the nine IMC subjects as neoadjuvant ICB patients; "
                                   "member suffix distinguishes sections from one subject. "
                                   "Per-subject clinical values require exact supplemental joins."))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns("sample_metadata"))


def fields() -> pd.DataFrame:
    proof = relative(PROOF)
    catalogue = relative(CACHE / "tls_catalogue.xlsx")
    histology = relative(CACHE / "metadata_histology_pub.txt")
    supplement = relative(CACHE / "paper_supplement.pdf")
    rows = [
        field(proof, "source_file;acquisition_id", "image_metadata",
              "image_id;source_image_id;source_acquisition_id;roi_id",
              "Use exact MCD member and acquisition ID; retain native ROI description.",
              "source_file + acquisition_id", "47 entries; 44 counted by source proof."),
        field(proof, "width;height;native_metadata.Description", "image_metadata",
              "source_x1_px;source_y1_px;region_annotation",
              "Retain the complete native acquisition frame and original description.",
              "source_file + acquisition_id"),
        field(relative(RAW), "MCD member stem", "image_metadata", "sample_id",
              "One MCD specimen section groups its ROI acquisitions.",
              "MCD member stem = sample_id"),
        field(relative(RAW), "MCD member stem", "sample_metadata",
              "sample_id;source_sample_id;patient_key;source_subject_id",
              "Retain member stem for specimen; remove acquisition suffix and section letter "
              "from public subject label, then prefix patient key with study ID.",
              "MCD member stem -> public subject label",
              "Ten MCD specimen sections map to nine public subjects; source zero padding "
              "is normalized for comparison with paper Table 5. Section suffix does not "
              "create another patient. Names remain only in NAS metadata."),
        field(supplement, "Supplementary Table 5: Patient", "sample_metadata",
              "patient_key;source_subject_id",
              "Confirm all nine public subject labels after removing numeric zero padding; "
              "retain the source-spelled label in source_subject_id.",
              "MCD member subject label -> Table 5 Patient",
              "Table 5 gives patient-level TLS counts, not individual treatment or survival values."),
        field(PAPER, "IMC staining and acquisition; nine-patient cohort", "sample_metadata",
              "sample_type;tissue_type;anatomical_site;primary_diagnosis;case_control_status;"
              "preservation_method;collection_method",
              "Use publication-level FFPE resected HCC liver tumor and cohort statements.",
              "study DOI = publication DOI",
              "Individual clinical endpoints and treatment response remain unfilled."),
        field(catalogue, "patient_id_ROI;Name;ImageJ_ROI", "image_metadata",
              "region_annotation", "Check published ROI/TLS identity; retain native MCD "
              "description without inferring a TLS type for unmatched acquisitions.",
              "public specimen prefix + ROI name", "38 TLS rows cover 31 published ROIs."),
        field(histology, "TLS;TLS_type;Location", "image_metadata", "metadata_notes",
              "Review public TLS histology labels; do not assign an individual TLS "
              "label to a multi-TLS acquisition without an exact ROI join.",
              "published TLS label to catalogue ImageJ_ROI"),
        field(relative(CHANNELS), "acquisitions.*.channels", "panel_mapping",
              "source_name;source_channel_index", "Use actual ordered native channel "
              "labels for each of the two verified MCD layouts.",
              "source_file + acquisition_id -> channel vector"),
        field(relative(RAW), "AblationDistanceBetweenShotsX;AblationDistanceBetweenShotsY",
              "image_metadata", "pixel_size_x_um;pixel_size_y_um",
              "Read the native MCD acquisition step distances in micrometers; all 47 "
              "local acquisition headers report 1 by 1.",
              "source_file + acquisition_id"),
        field(relative(S1), "Study;Original marker", "panel_mapping",
              "marker_id;role;nuclear;body;keep_channel",
              "Join exact study-specific native label to current S1 and retain all 50 source rows.",
              "S1.Study = study; S1.Original marker = native_label"),
    ]
    return pd.DataFrame(rows, columns=columns("field_mapping"))


def preparer_review(indexed: int, exclusions: list[dict]) -> None:
    rows = [
        dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study",
             record_id=STUDY, check_id="S01", result="unresolved",
             source_evidence=relative(PROOF), output_evidence=relative(OUTPUT / "images.parquet"),
             details=json.dumps(dict(counted_native_acquisitions=EXPECTED,
                                     source_exclusions=exclusions,
                                     published_images=indexed,
                                     independent_review="pending"))),
        dict(study=STUDY, preparer=JOB.name, reviewer="", scope="sample",
             record_id=STUDY, check_id="M01", result="unresolved",
             source_evidence=relative(CACHE / "tls_catalogue.xlsx"),
             output_evidence=relative(OUTPUT / "sample_metadata.csv"),
             details=("Preparer trace: source-member specimen and public subject labels "
                      "joined; publication and catalogue reviewed. Individual clinical "
                      "endpoints remain empty; independent review pending.")),
        dict(study=STUDY, preparer=JOB.name, reviewer="", scope="study",
             record_id=STUDY, check_id="D01", result="unresolved",
             source_evidence=relative(PROOF), output_evidence=relative(OUTPUT / "images.parquet"),
             details="Preparer export closed; independent review and input binding pending."),
    ]
    path = OUTPUT / "review_results.csv"
    with formal_read_lease(path, data_root=ROOT):
        old = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns("review_results"))
    own = old.preparer.eq(JOB.name) & old.reviewer.eq("") & old.check_id.isin(["S01", "M01", "D01"])
    reviews = pd.concat([old.loc[~own], pd.DataFrame(rows, columns=columns("review_results"))], ignore_index=True)
    with formal_publish_lease(path, directory=OUTPUT, data_root=ROOT):
        temporary = path.with_suffix(".csv.partial")
        reviews.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


def main() -> None:
    with formal_read_lease(META, data_root=ROOT):
        proof = json.loads(PROOF.read_text())
        channel_map = json.loads(CHANNELS.read_text())
    files = proof["files"]
    records = [item for entry in files for item in entry["acquisitions"] if item["counted"]]
    if len(files) != 10 or len(records) != EXPECTED or not all(
        entry["verification_complete"] for entry in files
    ):
        raise ValueError("Local proof does not establish the 10-file, 44-counted source scope")
    for entry in files:
        source = RAW / entry["source_file"]
        stat = source.stat()
        if (stat.st_size != entry["file_stat"]["size"] or
                stat.st_mtime_ns != entry["file_stat"]["mtime_ns"]):
            raise ValueError(f"Source version differs from completed proof: {source}")
    if len({image_id_for(item) for item in records}) != EXPECTED:
        raise ValueError("Native acquisition identities are not unique")
    for name in ("tls_catalogue.xlsx", "metadata_histology_pub.txt"):
        if not (CACHE / name).is_file():
            raise FileNotFoundError(CACHE / name)

    panels = {}
    panel_ids = {}
    for item in records:
        key = f"{item['source_file']}#acquisition_id={item['acquisition_id']}"
        mapped = channel_map["acquisitions"][key]["channels"]
        native = item["channels"]
        if (len(mapped) != 50 or len(native) != 50 or
                [(x["index"], x["native_label"]) for x in native] !=
                [(x["index"], x["native_label"]) for x in mapped]):
            raise ValueError(f"Channel proof mismatch: {key}")
        names = tuple(channel["native_label"] for channel in native)
        panel_id = f"{STUDY}__{names[0]}_native50"
        panel_ids[key] = panel_id
        if panel_id in panels and panels[panel_id].source_name.tolist() != list(names):
            raise ValueError(f"Distinct native vectors share panel ID: {panel_id}")
        if panel_id not in panels:
            panels[panel_id] = panel_from_s1(STUDY, panel_id, list(names), S1)
    panel = pd.concat(panels.values(), ignore_index=True)
    samples = public_samples(files)
    if len(samples) != 10 or samples.patient_key.nunique() != 9:
        raise ValueError("Public specimen-to-patient scope differs from paper")
    mapping = fields()
    excluded = [dict(image_id=image_id_for(item), reason=reason) for item in records
                if (reason := source_exclusion(item))]
    eligible = [item for item in records if source_exclusion(item) is None]
    if len(excluded) != 4:
        raise ValueError("Debug or fragment acquisitions differ from local proof")
    existing = {}
    index = OUTPUT / "images.parquet"
    if index.is_file():
        with formal_read_lease(index, data_root=ROOT):
            prior = pd.read_parquet(index)
        existing = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
    checkpoint("source_checked", len(existing),
               next((image_id_for(item) for item in eligible if image_id_for(item) not in existing), None),
               eligible_images=len(eligible), source_exclusions=excluded,
               sample_rows=len(samples), patient_linked_samples=len(samples))

    for item in eligible:
        image_id = image_id_for(item)
        destination = OUTPUT / "images" / image_id / "input.tiff"
        if image_id in existing and destination.is_file():
            continue
        source = RAW / item["source_file"]
        panel_id = panel_ids[f"{item['source_file']}#acquisition_id={item['acquisition_id']}"]
        selected_panel = panels[panel_id]
        side = max(item["width"], item["height"])
        peak = 39 * side * side * 4
        if shutil.disk_usage(ROOT).free < 100 * 1024**3 + 2 * peak:
            checkpoint("waiting_io", len(existing), image_id,
                       reason="NAS free space below 100 GiB reserve plus two image copies")
            return
        if shutil.disk_usage("/tmp").free < 64 * 1024**3 + 2 * peak:
            checkpoint("waiting_io", len(existing), image_id,
                       reason="local free space below 64 GiB reserve plus two image copies")
            return
        spec = SourceSpec(image_id=image_id, panel_id=panel_id, kind="mcd", path=source,
                          channels=tuple(range(50)), acquisition_id=int(item["acquisition_id"]),
                          pixel_size_um=(1.0, 1.0), source_revision=REVISION,
                          intensity_description="Native unnormalized float32 MCD intensity")
        checkpoint("exporting", len(existing), image_id, source_file=item["source_file"])
        with open_source(spec) as reader:
            result = export_input(reader, selected_panel, destination, tile_size=256, data_root=ROOT)
        if not result["admitted"]:
            excluded.append(dict(image_id=image_id, reason="No nonzero admitted marker pixels"))
            checkpoint("excluded_no_marker_signal", len(existing), image_id,
                       source_exclusions=excluded)
            continue
        with formal_read_lease(destination, data_root=ROOT):
            with tifffile.TiffFile(destination) as tif:
                if (len(tif.pages) != result["channel_count"] or
                        tif.pages[0].shape != (side, side) or
                        tif.pages[0].dtype.name != "float32"):
                    raise ValueError(f"Published input header differs: {image_id}")
        row = dict.fromkeys(columns("image_metadata"))
        row.update(study=STUDY, image_id=image_id,
                   sample_id=Path(item["source_file"]).stem, panel_id=panel_id,
                   image_kind="roi", source_image_id=item["native_acquisition_id"],
                   source_files=[relative(source)], source_format="mcd",
                   source_acquisition_id=str(item["acquisition_id"]), source_axes="CYX",
                   source_dtype="float32", source_revision=REVISION,
                   roi_id=item["native_metadata"]["Description"],
                   region_annotation=item["native_metadata"]["Description"],
                   source_x0_px=0, source_y0_px=0,
                   source_x1_px=item["width"], source_y1_px=item["height"],
                   pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                   input_image_path=relative(destination),
                   valid_pixels_path=(relative(destination.with_name("valid_pixels.tiff"))
                                      if result["valid_pixels"] else None),
                   intensity_description="Native unnormalized float32 MCD acquisition intensity",
                   split_method="source_acquisition", split_evidence=relative(PROOF),
                   metadata_notes="Native frame and channel order from completed local acquisition proof.",
                   **{key: result[key] for key in (
                       "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px",
                       "height_px", "width_px", "source_channel_count", "channel_count")})
        existing[image_id] = row
        checkpoint("published_unindexed", len(existing) - 1, image_id,
                   next_step="reopen published TIFF and save image metadata")
        save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, mapping,
                          data_root=ROOT)
        remaining = next((image_id_for(value) for value in eligible
                          if image_id_for(value) not in existing), None)
        checkpoint("indexed", len(existing), remaining,
                   sample_rows=len(samples), sample_linked_images=len(existing),
                   patient_linked_samples=len(samples), source_exclusions=excluded)
        print(f"indexed {len(existing)}/{len(eligible)} {image_id}", flush=True)

    if existing:
        save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, mapping,
                          data_root=ROOT)
    preparer_review(len(existing), excluded)
    checkpoint("prepared", len(existing), None, source_exclusions=excluded,
               sample_rows=len(samples), sample_linked_images=len(existing),
               patient_linked_samples=len(samples),
               next_step="independent source, panel, pixel and clinical review")


def repair_ot7_split_candidate(request_path: Path) -> None:
    """Export the two exactly adjacent native acquisitions to a review candidate."""
    from contextlib import ExitStack
    from decimal import Decimal
    from readimc import MCDFile
    from cellweft.data.contracts import ImageBlock, Window
    from cellweft.data.io.readers import windows

    request = json.loads(request_path.read_text())
    source = Path(request["source_mcd"])
    destination = Path(request["destination"])
    stat = source.stat()
    before = dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    with MCDFile(source) as mcd:
        acquisitions = {a.id: a for slide in mcd.slides for a in slide.acquisitions}
        upper, lower = acquisitions[7], acquisitions[9]
        assert (upper.height_px, lower.height_px, upper.width_px, lower.width_px) == (36, 764, 800, 800)
        assert upper.num_channels == lower.num_channels == 50
        for key in ("channel_labels", "channel_metals", "channel_masses"):
            assert list(getattr(upper, key)) == list(getattr(lower, key))
        points = []
        for acquisition in (upper, lower):
            assert acquisition.pixel_size_x_um == acquisition.pixel_size_y_um == 1.0
            polygon = [[Decimal(str(x)), Decimal(str(y))] for x, y in acquisition.roi_points_um]
            x, y = polygon[0]
            width, height = Decimal(acquisition.width_px), Decimal(acquisition.height_px)
            assert polygon == [[x, y], [x+width, y], [x+width, y-height], [x, y-height]]
            points.append(polygon)
        assert points[0][2] == points[1][1] and points[0][3] == points[1][0]

    panel = pd.read_csv(request["panel_csv"])
    panel = panel[panel.panel_id == request["panel_id"]].copy()
    panel["output_channel_index"] = panel.output_channel_index.astype("Int64")
    assert len(panel) == 50 and int(panel.keep_channel.sum()) == 39

    class NativeSplitReader:
        def __init__(self, readers: list) -> None:
            self.readers = readers
            self.shape = (50, 800, 800)
            self.dtype = np.dtype("float32")
            self.source = SourceSpec("OT7__acquisition_009", request["panel_id"],
                "mcd_acquisition_mosaic", source, tuple(range(50)), pixel_size_um=(1.0, 1.0),
                source_revision="zenodo_10020271_v1_OT7_acquisitions_7_9_exact_native_grid")

        def read_window(self, channels: list[int], window: Window) -> np.ndarray:
            x0, y0, x1, y1 = window
            result = np.zeros((len(channels), y1-y0, x1-x0), dtype=self.dtype)
            for start, end, reader in zip((0, 36), (36, 800), self.readers):
                a, b = max(y0, start), min(y1, end)
                if a < b:
                    result[:, a-y0:b-y0] = reader.read_window(channels, (x0, a-start, x1, b-start))
            return result

        def valid_window(self, window: Window) -> np.ndarray:
            x0, y0, x1, y1 = window
            result = np.zeros((y1-y0, x1-x0), dtype=bool)
            for start, end, reader in zip((0, 36), (36, 800), self.readers):
                a, b = max(y0, start), min(y1, end)
                if a < b:
                    result[a-y0:b-y0] = reader.valid_window((x0, a-start, x1, b-start))
            return result

        def iter_blocks(self, channels: list[int], block_size: int):
            for window in windows((800, 800), block_size):
                yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))

    with ExitStack() as stack:
        readers = [stack.enter_context(open_source(SourceSpec(
            "OT7__acquisition_"+str(aid), request["panel_id"], "mcd", source,
            tuple(range(50)), acquisition_id=aid))) for aid in (7, 9)]
        reader = NativeSplitReader(readers)
        valid = int(reader.valid_window((0, 0, 800, 800)).sum())
        result = export_input(reader, panel, destination, tile_size=256,
                              data_root=Path(request["data_root"]))
    stat = source.stat()
    assert before == dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    assert result["admitted"]
    report = dict(study=STUDY, source=before, source_acquisition_ids=[7, 9],
        placement_yx=[[0, 0], [36, 0]], shape_yx=[800, 800], measured_pixels=valid,
        seam_y=36, interpolation=False, source_pixels_changed=False,
        original_roi_points_um=[[[str(x), str(y)] for x, y in polygon] for polygon in points],
        candidate=str(destination), export=result, formal_NAS_inputs_changed=False)
    Path(request["report_path"]).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(candidate=str(destination), measured_pixels=valid, admitted=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-ot7-split-candidate", type=Path)
    arguments = parser.parse_args()
    if arguments.repair_ot7_split_candidate:
        repair_ot7_split_candidate(arguments.repair_ot7_split_candidate)
    else:
        main()
