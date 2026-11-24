"""Prepare native COMET core inputs from Figshare 28861841 v3.

Inputs: 16 verified OME TIFFs, ordered channel map, S1, official coldata.csv,
and the article XML. Outputs: native uint8 CYX inputs and study metadata.
Each published image is indexed before the next source is opened.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import (TEMPLATES, export_input, panel_from_s1,
                                                   save_input_tables, validate_image_metadata)

STUDY = "Ly2025_COMET_AML"
PANEL = "figshare_28861841_v3_comet"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
RAW = NAS / "raw/public" / STUDY / "figshare_28861841_v3"
PROOFS = META / "native_verification"
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
DEFAULT_JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0227_prepare_0"
ATTACHMENTS = {
    "coldata.csv": (Path("/tmp/Ly2025_COMET_AML_coldata.csv"), 28513199),
    "paper.xml": (Path("/tmp/Ly2025_COMET_AML_paper.xml"), None),
    "supplement.pdf": (Path("/tmp/Ly2025_COMET_AML_supplement.pdf"), 9806842),
}


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def publish_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n"
    with formal_publish_lease(path, directory=path.parent):
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def source_table(table: str) -> list[str]:
    return pd.read_csv(TEMPLATES / table, nrows=0).columns.tolist()


def clinical_table(path: Path) -> pd.DataFrame:
    """Transcribe the public paper's Table 1 without assigning unreported time origins."""
    with formal_read_lease(path):
        root = ET.parse(path).getroot()
    table = next(item for item in root.iter("table-wrap")
                 if item.find("label") is not None and "Table 1" in "".join(item.find("label").itertext()))
    rows = [["".join(cell.itertext()).strip() for cell in row if cell.tag in {"td", "th"}]
            for row in table.iter("tr")]
    patients = rows[0][1:]
    values = {row[0]: row[1:] for row in rows[1:] if len(row) == len(patients) + 1}
    return pd.DataFrame({"patient_id": patients,
                         "sex": values["Sex/gender"],
                         "age_source_years": values["Age (years)"],
                         "treatment": values["Pre/post treatment"],
                         "vital": values["Status"],
                         "aml_subtype": values["AML subtype*"],
                         "survival_source_months": values["Survival time (months)"]})


def sample_and_field_tables(coldata_path: Path, paper_path: Path, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Use one donor-block biopsy per released core and link its public patient ID."""
    with formal_read_lease(coldata_path):
        cells = pd.read_csv(coldata_path, usecols=["tma_name", "patient_id"], dtype=str)
    pairs = cells.drop_duplicates().sort_values("tma_name")
    if len(pairs) != 16 or pairs.tma_name.duplicated().any():
        raise ValueError("Official coldata does not give one patient per released core")
    expected = {Path(item["name"]).stem.replace("_", "-") for item in manifest["files"]}
    if set(pairs.tma_name) != expected:
        raise ValueError("Official coldata core names differ from TIFF inventory")
    clinical = clinical_table(paper_path).set_index("patient_id")
    if set(pairs.patient_id) != set(clinical.index) or len(clinical) != 7:
        raise ValueError("Published core patients differ from article Table 1")
    samples = []
    for core, patient in pairs.itertuples(index=False, name=None):
        value = clinical.loc[patient]
        row = dict.fromkeys(source_table("sample_metadata.csv"), None)
        row.update(study=STUDY, sample_id="biopsy_" + core.replace("-", "_"),
                   patient_key=STUDY + ":" + patient, source_subject_id=patient,
                   sample_type="bone marrow trephine biopsy", tissue_type="tumor",
                   anatomical_site="bone marrow", preservation_method="ffpe",
                   primary_diagnosis="acute myeloid leukemia",
                   sex_reported=value.sex.lower(),
                   vital_status=value.vital.lower(),
                   metadata_notes="One core from each donor block per article Methods (16 biopsies, 7 patients); core identity is the released tma_name, not an original biopsy ID. "
                                  "Table 1 treatment category is patient-level and cannot be assigned to this biopsy at collection: " + value.treatment + ". "
                                  "Age and survival time in Table 1 lack explicit origin and are not assigned to time-specific base fields. "
                                  "Source AML subtype: " + value.aml_subtype + "; source age years: " + value.age_source_years +
                                  "; source survival months: " + value.survival_source_months)
        samples.append(row)
    mappings = []

    def mapping(source_file: Path, source_field: str, target_table: str, target_field: str,
                join_keys: str, rule: str, notes: str = "") -> None:
        mappings.append(dict(study=STUDY, source_file=relative(source_file),
                             source_sheet_or_table="coldata.csv" if source_file == coldata_path else
                             "Methods" if source_file == paper_path and source_field.startswith("Methods") else
                             "Table 1" if source_file == paper_path else
                             "Supplementary Table 1" if source_file == META / "source_metadata/supplement.pdf" else
                             "OME TIFF / source manifest",
                             source_field=source_field, target_table=target_table,
                             target_field=target_field, join_keys=join_keys, value_mapping="{}",
                             transformation_rule=rule, time_origin_rule="Unspecified in Table 1" if source_file == paper_path else "",
                             conflict_rule="Require one patient_id per tma_name; do not merge distinct cores by patient",
                             notes=notes))

    mapping(coldata_path, "tma_name", "image_metadata", "sample_id", "TIFF stem underscore to hyphen = tma_name",
            "Each of 16 unique released cores represents one donor-block biopsy; sample_id = biopsy_ + TIFF stem.",
            "Paper Methods: 16 biopsies from 7 patients, one core from each donor block; Results says seven biopsies. Source core identity resolves the released 16-core range.")
    mapping(paper_path, "Methods: 16 biopsies; one core from each donor block", "sample_metadata", "sample_id",
            "Methods donor block -> released core tma_name; TIFF stem underscore to hyphen = tma_name",
            "Create a distinct sample per released core; original biopsy IDs are not published.")
    mapping(coldata_path, "patient_id", "sample_metadata", "patient_key", "tma_name -> patient_id",
            "Study-scoped key from public pseudonym; 16 core samples join to seven patients.")
    mapping(coldata_path, "patient_id", "sample_metadata", "source_subject_id", "tma_name -> patient_id",
            "Preserve the public patient pseudonym as released in the official cell metadata.")
    for source, target, rule in (("Sex/gender", "sex_reported", "Lowercase published category"),
                                 ("Status", "vital_status", "Lowercase published follow-up category")):
        mapping(paper_path, source, "sample_metadata", target, "patient_id = Table 1 PT column", rule)
    for target in ("sample_type", "tissue_type", "anatomical_site", "preservation_method", "primary_diagnosis"):
        mapping(paper_path, "Methods: bone marrow biopsy / FFPE AML", "sample_metadata", target,
                "16 donor blocks -> released tma_name core", "AML-affected bone marrow biopsy; tumor is a cohort-level tissue classification, not an estimate of tumor fraction")
    mapping(paper_path, "Pre/post treatment; Age (years); Survival time (months); AML subtype", "sample_metadata", "metadata_notes",
            "patient_id = Table 1 PT column", "Preserve patient-level source values in notes; do not assign treatment category to individual biopsy collection or assert age/survival origin")
    mapping(META / "source_ordered_channel_map.json", "OME channel order", "image_metadata", "panel_id", "TIFF file = manifest name",
            "31 native channels; DAPI retained, generic fluorophores and repeated applications excluded from saved channels")
    mapping(META / "source_metadata/supplement.pdf", "Supplementary Table 1: COMET Antibodies Quality", "panel_mapping", "keep_channel", "antigen + cycle + source OME channel order",
            "CD34 cycle 1 quality *** exceeds cycle 11 **; FOXP3 cycle 8 ** exceeds cycle 14 *. Retain higher-rated measurement for each duplicate marker.")
    return pd.DataFrame(samples), pd.DataFrame(mappings, columns=source_table("field_mapping.csv")), dict(zip(pairs.tma_name, pairs.patient_id))


def publish_repaired_metadata(images: pd.DataFrame, samples: pd.DataFrame,
                              field_mapping: pd.DataFrame, panel: pd.DataFrame,
                              versions: dict[str, tuple[int, int]]) -> None:
    """Replace the corrected study tables together under the study publication lock."""
    images = validate_image_metadata(images, samples, panel, field_mapping)
    with tempfile.TemporaryDirectory(prefix="ly-comet-repair-") as temporary_root:
        staged = Path(temporary_root)
        arrow = pa.Table.from_pandas(images, preserve_index=False)
        index = arrow.schema.get_field_index("source_files")
        arrow = arrow.set_column(index, "source_files",
                                 pa.array(images.source_files.tolist(), type=pa.list_(pa.string())))
        pq.write_table(arrow, staged / "images.parquet", compression="zstd")
        samples.to_csv(staged / "sample_metadata.csv", index=False, encoding="utf-8-sig")
        field_mapping.to_csv(staged / "field_mapping.csv", index=False, encoding="utf-8-sig")
        with formal_publish_lease(OUTPUT, directory=OUTPUT):
            current = {name: ((OUTPUT / name).stat().st_size, (OUTPUT / name).stat().st_mtime_ns)
                       for name in versions}
            if current != versions:
                raise RuntimeError("Input tables changed during Ly metadata repair; retry from current tables")
            for name in ("sample_metadata.csv", "field_mapping.csv", "images.parquet"):
                temporary = OUTPUT / (name + ".partial")
                shutil.copyfile(staged / name, temporary)
                os.replace(temporary, OUTPUT / name)


def source_spec(path: Path, proof: dict) -> tuple[SourceSpec, list[str]]:
    with formal_read_lease(path), tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 31 or tif.series[0].axes != "CYX" or tif.series[0].dtype.name != "uint8":
            raise ValueError(f"Native OME geometry differs: {path}")
        root = ET.fromstring(tif.ome_metadata)
        names = [node.attrib["Name"] for node in root.iter() if node.tag.endswith("}Channel")]
        pixels = next(node for node in root.iter() if node.tag.endswith("}Pixels"))
        width, height = int(pixels.attrib["SizeX"]), int(pixels.attrib["SizeY"])
        scale = float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])
        if scale[0] != scale[1] or list(tif.series[0].shape) != [31, height, width]:
            raise ValueError(f"Native OME scale/shape differs: {path}")
    if names != proof["ome_channel_names"] or [height, width] != proof["shape_yx"]:
        raise ValueError(f"Native proof/channel order differs: {path}")
    source = SourceSpec(path.stem, PANEL, "tiff", path, tuple(range(31)), pixel_size_um=scale,
                        source_revision="Figshare28861841_v3",
                        intensity_description="Native uint8 COMET source intensity, no additional normalization")
    return source, names


def verify_output(destination: Path, side: int, panel: pd.DataFrame) -> None:
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    with formal_read_lease(destination), tifffile.TiffFile(destination) as tif:
        if tif.series[0].shape != (len(kept), side, side) or tif.series[0].dtype.name != "uint8":
            raise ValueError(f"Published image header differs: {destination}")
        names = [node.attrib["Name"] for node in ET.fromstring(tif.ome_metadata).iter()
                 if node.tag.endswith("}Channel")]
        if names != kept.marker_id.tolist():
            raise ValueError(f"Published image names differ: {destination}")


def resume_published(source: SourceSpec, proof: dict, destination: Path, panel: pd.DataFrame) -> dict:
    """Index an output left between atomic TIFF publication and table publication."""
    height, width = proof["shape_yx"]
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    verify_output(destination, side, panel)
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    source_window = (0, 0, min(32, width), min(32, height))
    output_window = (left, top, left + source_window[2], top + source_window[3])
    with formal_read_lease(source.path), formal_read_lease(destination), \
            tifffile.TiffFile(source.path) as native, tifffile.TiffFile(destination) as saved:
        for output_index, source_index in enumerate(kept.source_channel_index):
            if not (read_tiff_window(native, int(source_index), source_window) ==
                    read_tiff_window(saved, output_index, output_window)).all():
                raise ValueError(f"Published image pixels differ: {destination}")
    return dict(admitted=True, height_px=side, width_px=side, pad_left_px=left,
                pad_top_px=top, pad_right_px=side-width-left, pad_bottom_px=side-height-top,
                source_channel_count=31, channel_count=len(kept), valid_pixels=None)


def review_results(images: pd.DataFrame, samples: pd.DataFrame, expected: int, job: Path) -> None:
    counts = dict(expected_images=expected, exported_images=len(images), sample_rows=len(samples),
                  sample_linked_images=int(images.sample_id.notna().sum()),
                  patient_linked_samples=int(samples.patient_key.notna().sum()))
    rows = []
    for check in ("S01", "I01", "I02", "I03", "I04", "P01", "P02", "P03", "I05", "M01", "M02", "D01"):
        rows.append(dict(study=STUDY, preparer=STUDY + "_" + job.name, reviewer="",
                         scope="study" if check in {"S01", "D01"} else "sample" if check.startswith("M") else
                         "panel" if check in {"P01", "P02"} else "image", record_id=STUDY, check_id=check,
                         result="unresolved" if check in {"I05", "D01"} or len(images) != expected else "pass",
                         source_evidence=json.dumps([relative(META / "source_manifest.json"), relative(META / "source_metadata/coldata.csv"),
                                                     relative(META / "source_metadata/paper.xml"), relative(META / "source_metadata/supplement.pdf")]),
                         output_evidence=relative(OUTPUT),
                         details=json.dumps(dict(preparer_self_check=True, independent_review_pending=True,
                                                 counts=counts, description="Preparer source and output checks; independent reviewer pending"))))
    result = pd.DataFrame(rows, columns=source_table("review_results.csv"))
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        temporary = OUTPUT / "review_results.csv.partial"
        result.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, OUTPUT / "review_results.csv")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
parser.add_argument("--stage", choices=("metadata", "export", "repair"), default="export")
args = parser.parse_args()
job = args.job

with formal_read_lease(META):
    manifest = json.loads((META / "source_manifest.json").read_text())
    channel_map = json.loads((META / "source_ordered_channel_map.json").read_text())
assert len(manifest["files"]) == 16 and len(channel_map["channels"]) == 31
source_names = [channel["ome_name"] for channel in channel_map["channels"]]
panel = panel_from_s1(STUDY, PANEL, source_names, S1,
                      drop_indices={23: "Supplementary Table 1 CD34 cycle 11 quality **; cycle 1 quality *** retained",
                                    30: "Supplementary Table 1 FOXP3 cycle 14 quality *; cycle 8 quality ** retained"})
assert len(panel) == 31 and panel.keep_channel.sum() == 27

# Publication attachments are restored from the accepted source namespace.
samples, field_mapping, core_patient = sample_and_field_tables(META / "source_metadata/coldata.csv", META / "source_metadata/paper.xml", manifest)
with formal_read_lease(OUTPUT), resource_lock(OUTPUT, shared=True):
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=source_table("image_metadata.csv"))
    if args.stage == "repair":
        published_panel = pd.read_csv(OUTPUT / "panel_mapping.csv")
        table_names = ("images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv")
        table_versions = {name: ((OUTPUT / name).stat().st_size, (OUTPUT / name).stat().st_mtime_ns)
                          for name in table_names}
completed = set(images.image_id)


def progress(stage: str, **details: object) -> None:
    publish_json(job / "progress.json", dict(study=STUDY, phase="repair" if args.stage == "repair" else "prepare",
                 scientific_repair_round=1 if args.stage == "repair" else 0,
                 stage=stage, updated_at=datetime.now(timezone.utc).isoformat(),
                 expected_images=len(manifest["files"]), exported_images=len(images),
                 sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                 patient_linked_samples=int(samples.patient_key.notna().sum()),
                 next_image=next((Path(item["name"]).stem for item in manifest["files"]
                                  if Path(item["name"]).stem not in completed), None), **details))


if args.stage == "repair":
    if len(images) != len(manifest["files"]) or set(images.image_id) != {
            Path(item["name"]).stem for item in manifest["files"]}:
        raise ValueError("Repair requires all 16 already indexed native inputs")
    images = images.copy()
    images["sample_id"] = "biopsy_" + images.image_id.astype(str)
    progress("repair_staged", affected_tables=["images.parquet", "sample_metadata.csv", "field_mapping.csv"])
    publish_repaired_metadata(images, samples, field_mapping, published_panel, table_versions)
    progress("repair_published", independent_review_pending=True)
else:
    save_input_tables(OUTPUT, images, samples, panel, field_mapping)
    progress("metadata_ready")

if args.stage == "export":
    for item in manifest["files"]:
        image_id = Path(item["name"]).stem
        source_path = RAW / item["name"]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        with formal_read_lease(PROOFS / (image_id + ".json")):
            proof = json.loads((PROOFS / (image_id + ".json")).read_text())
        stat = source_path.stat()
        if (stat.st_size != item["size_bytes"] or stat.st_size != proof["bytes_actual"] or
                stat.st_mtime_ns != proof["file_stat"]["mtime_ns"] or
                not proof["all_native_channels_bounded_read"]):
            raise ValueError(f"Complete native proof no longer binds {item['name']}")
        source, names = source_spec(source_path, proof)
        if names != source_names:
            raise ValueError(f"Panel differs for {item['name']}")
        side = max(proof["shape_yx"])
        if image_id in completed:
            verify_output(destination, side, panel)
            continue
        progress("exporting", current_image=image_id)
        if destination.is_file():
            result = resume_published(source, proof, destination, panel)
        else:
            with open_source(source) as reader:
                result = export_input(reader, panel, destination, tile_size=512)
        if not result["admitted"]:
            progress("source_excluded", excluded_image=image_id, reason="all admitted marker pixels zero")
            continue
        verify_output(destination, side, panel)
        row = dict.fromkeys(source_table("image_metadata.csv"), None)
        row.update(study=STUDY, image_id=image_id, sample_id="biopsy_" + image_id,
                   panel_id=PANEL, image_kind="tma_core", source_image_id=item["name"],
                   source_files=[relative(source_path)], source_format="tiff", source_axes="CYX",
                   source_dtype="uint8", source_revision="Figshare28861841_v3",
                   slide_id=image_id.split("_")[0], core_id=image_id, roi_id=image_id,
                   source_x0_px=0, source_y0_px=0, source_x1_px=proof["shape_yx"][1], source_y1_px=proof["shape_yx"][0],
                   pixel_size_x_um=source.pixel_size_um[0], pixel_size_y_um=source.pixel_size_um[1],
                   input_image_path=relative(destination),
                   valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
                   intensity_description=source.intensity_description,
                   split_method="already_independent", split_evidence=relative(META / "source_metadata/coldata.csv"),
                   metadata_notes="One complete original core frame; official cell metadata joins tma_name to patient_id")
        row.update({key: result[key] for key in ("height_px", "width_px", "pad_left_px", "pad_top_px",
                    "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count")})
        images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
        save_input_tables(OUTPUT, images, samples, panel, field_mapping)
        completed.add(image_id)
        progress("exporting", last_completed_image=image_id)
        print(json.dumps(dict(image_id=image_id, exported_images=len(images))), flush=True)
    review_results(images, samples, len(manifest["files"]), job)
    progress("prepared")
