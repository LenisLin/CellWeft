"""Prepare the 58 admitted Bowen IMC ROIs without changing published intensity.

Inputs: local SPORA Zarr/Parquet, original native verification, study-specific S1,
Zenodo 7655637 metadata and publisher Supplemental Table 1 in this study's cache.
Outputs: independent CYX input.tiff, panel/image/sample/field/review tables,
resumable per-image receipts, native-source inventory and preparer checks.
Patient identity comes from the author's ROI map, never from image-name parsing.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/study_input_batch/Bowen2023_IMC_Colorectal/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables
from cellweft.data.preparation.io_coordination import formal_publish_lease


STUDY = "Bowen2023_IMC_Colorectal"
PROJECT = Path(__file__).resolve().parents[3]
TEMPLATES = PROJECT / "docs/data/templates/study_input"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path("/mnt/NAS_21T/ProjectData/CellWeft"))
parser.add_argument("--progress", type=Path, default=PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0018_prepare_0/progress.json")
parser.add_argument("--metadata-only", action="store_true")
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
for directory in (cache, output, receipts, temporary):
    directory.mkdir(parents=True, exist_ok=True)
zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)


def relative(path: Path) -> str:
    if path.is_relative_to(cache):
        return str(path)
    return str(metadata_source_relative(path, root))


def save_json(path: Path, value: dict | list) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def progress(stage: str, count: int, **details: object) -> None:
    save_json(args.progress, dict(study=STUDY, stage=stage, completed_images=count,
                                 expected_images=58, updated_utc=datetime.now(timezone.utc).isoformat(),
                                 gaps=["独立复核与主控验收待执行", "Supplemental Table 1与作者ROI表的Subject编号无crosswalk；个体剂量保持空值"],
                                 **details))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


class StagedReader:
    """Expose a local disk-backed ROI after one spatially blocked Zarr decode.

    These sources pack all 39 channels in each spatial chunk. Stage 256x256
    blocks to local disk so channel-major TIFF export does not repeatedly
    decode the same NAS shard. This study adapter can later move to shared IO.
    """

    def __init__(self, array: np.memmap, source: SourceSpec):
        self.array, self.source = array, source
        self.shape, self.dtype = array.shape, array.dtype

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.asarray(self.array[channels, y0:y1, x0:x1])

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


# Reconcile every local source with original verification and the admitted scope.
progress("reconciling_sources", 0)
verification_path = metadata / "native_verification.json"
verification = json.loads(verification_path.read_text())
verified = {item["image_id"]: item for item in verification["verified"]}
assert verification["verified_count"] == len(verified) == 58 and not verification["failed"]
scope_path = metadata / "source_scope_decision.json"
assert json.loads(scope_path.read_text())["root_decision"] == "retain_existing_qualified_normal_tissue_scope"
tissues_path = raw / "metadata/tissues.parquet"
tissues = pd.read_parquet(tissues_path).set_index("tissue_id", verify_integrity=True)
assert set(tissues.index) == set(verified)
selected = sorted(verified)
assert {p.name.removesuffix(".ome.zarr") for p in (raw / "imc/1_0mpp/images").glob("*.ome.zarr")} == set(selected)
channel_path = raw / "imc/channels.parquet"
availability_path = raw / "imc/channels_per_tissue.parquet"
channels = pd.read_parquet(channel_path)
availability = pd.read_parquet(availability_path)
ordered_path = metadata_source_directory(root, STUDY) / 'imc_ordered_channels_by_tissue.json'
ordered = json.loads(ordered_path.read_text())
source_names = channels.channel_name.tolist()
assert availability.columns.tolist() == source_names and len(source_names) == 39
author_channels = pd.read_csv(evidence / "metadata_nf.csv")
assert author_channels.metal.tolist() == source_names
panel_id = f"{STUDY}__spora_TMA23_39"
s1_path = root / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
panel = panel_from_s1(STUDY, panel_id, source_names, s1_path)
kept = panel[panel.keep_channel].sort_values("output_channel_index")
assert len(kept) == 38 and panel.loc[~panel.keep_channel, "source_name"].tolist() == ["101Ru"]
assert panel.loc[panel.nuclear, "source_name"].tolist() == ["191Ir", "193Ir"]

source_objects_path = metadata / "source_objects.json"
source_objects = json.loads(source_objects_path.read_text())
inventory = []
for image_id in selected:
    proof = verified[image_id]
    group = root / proof["path"]
    group_metadata = json.loads((group / "zarr.json").read_text())["attributes"]["ome"]
    array_metadata = json.loads((group / "0/zarr.json").read_text())
    assert channels.loc[availability.loc[image_id].to_numpy(bool), "channel_name"].tolist() == source_names
    assert source_names == [item["channel_name"] for item in ordered[image_id]]
    assert source_names == [item["channel_name"] for item in proof["marker_mapping"]]
    assert source_names == [item["label"] for item in group_metadata["omero"]["channels"]]
    assert array_metadata["shape"] == proof["shape"] and proof["shape"][0] == 39
    assert array_metadata["data_type"] == proof["dtype"] == "float32"
    objects = [item for item in source_objects if f"/{image_id}.ome.zarr/" in item["key"]]
    files = []
    for item in objects:
        suffix = item["key"].split(f"/{image_id}.ome.zarr/", 1)[1]
        path = group / suffix
        assert path.is_file() and path.stat().st_size == item["size"]
        files.append(dict(path=relative(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
    assert len(files) == proof["source_object_count"]
    assert sum(item["size"] for item in files) == proof["source_bytes"]
    multiscale = group_metadata["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["c", "y", "x"]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]
    inventory.append(dict(image_id=image_id, source_path=relative(group), files=files,
                          shape=proof["shape"], dtype=proof["dtype"],
                          ignored_partial_files=[relative(p) for p in group.rglob("*.partial")]))
estimated_output = sum(38 * max(item["shape"][-2:])**2 * 4 for item in inventory)
save_json(cache / "source_inventory.json", dict(study=STUDY, selected=inventory,
          reused_native_verification=relative(verification_path), accepted_scope=relative(scope_path),
          source_objects=len(source_objects), source_bytes=sum(item["size"] for item in source_objects),
          estimated_uncompressed_output_bytes=estimated_output, expected_images=58,
          resolution_note="Highest published SPORA 1_um scale; full published frame, no original-MCD coordinate claim"))

# Rebuild ROI -> biopsy visit -> patient using the original author's table.
author_sample_path = evidence / "meta_sample.tsv"
author = pd.read_csv(author_sample_path, sep="\t", dtype=str)
assert len(author) == 58 and author.roi.is_unique
joined = tissues.reset_index().merge(author, on="roi", how="left", validate="one_to_one", suffixes=("", "_author"))
for key in ("Subject ID", "Treatment Arm", "Visit"):
    assert (joined[key].astype(str) == joined[f"{key}_author"].astype(str)).all()
assert (joined["Subject ID"].astype(str) == joined.original_patient_identifier.astype(str)).all()
assert joined.groupby("Subject ID").patient_id.nunique().eq(1).all()
assert joined.groupby("patient_id")["Subject ID"].nunique().eq(1).all()
assert set(joined.Visit) == {"Pre", "Post"}
assert joined.groupby("Subject ID")["Treatment Arm"].nunique().eq(1).all()
# The publisher supplement confirms paired specimens but uses another numbering.
# Equal subject numerals cannot bridge the conflicting arm/ROI-count records.
supplement_path = evidence / "Table_1.docx"
with zipfile.ZipFile(supplement_path) as document:
    document_xml = ET.fromstring(document.read("word/document.xml"))
word = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
table = document_xml.find(".//w:tbl", word)
supplement_rows = [["".join(text.text or "" for text in cell.findall(".//w:t", word)).strip()
                    for cell in row.findall("w:tc", word)] for row in table.findall("w:tr", word)]
supplement = pd.DataFrame(supplement_rows[1:], columns=["Subject", "Treatment group", "Pre_ROIs", "Post_ROIs"])
supplement[["Subject", "Pre_ROIs", "Post_ROIs"]] = supplement[["Subject", "Pre_ROIs", "Post_ROIs"]].astype(int)
assert len(supplement) == 18 and int(supplement.Pre_ROIs.sum() + supplement.Post_ROIs.sum()) == 58
author_counts = author.pivot_table(index=["Subject ID", "Treatment Arm"], columns="Visit", values="roi", aggfunc="size").reset_index()
comparison = author_counts.merge(supplement, left_on=author_counts["Subject ID"].astype(int), right_on="Subject", validate="one_to_one")
comparison["arm_agrees_if_equal_numeral"] = comparison["Treatment Arm"] == comparison["Treatment group"].map(
    lambda value: "Placebo" if value == "Placebo" else "Naproxen")
comparison["counts_agree_if_equal_numeral"] = (comparison.Pre == comparison.Pre_ROIs) & (comparison.Post == comparison.Post_ROIs)
assert not comparison.arm_agrees_if_equal_numeral.all()
comparison.to_csv(cache / "subject_numbering_comparison.csv", index=False, encoding="utf-8-sig")
workspace_path = evidence / "flowjo.wsp"
workspace = ET.parse(workspace_path).getroot()
assert len(workspace.findall(".//Sample")) == 58
clinical_attributes = [element.attrib for element in workspace.iter()
                       if any(term in str(element.attrib).lower()
                              for term in ("naprox", "placebo", "high dose", "low dose"))]
assert not clinical_attributes
save_json(cache / "clinical_numbering_gap.json", dict(
    source=relative(supplement_path), roi_map=relative(author_sample_path),
    caption="Each subject (N=18) had a pre- and post-treatment biopsy assessed (N=36 samples).",
    supplement_total_ROIs=58, author_total_ROIs=58,
    comparison=relative(cache / "subject_numbering_comparison.csv"),
    conclusion="Supplemental Subject and author Subject ID are incompatible identifiers without a published crosswalk; "
               "do not join on equal numerals or infer identities/dose from ROI-count patterns.",
    affected_fields=["individual naproxen dose", "Supplemental Subject crosswalk"],
    additional_search=dict(source=relative(workspace_path), result="58 samples; no Naproxen/placebo/high-low-dose annotations or crosswalk. "
                           "Workspace default unrelated treatment keyword is not used."),
    publisher_additional_attachment=dict(source=relative(evidence / "DataSheet_1.pdf"), pages=3,
                                         result="Preparer inspected Supplemental Figures 1-3; no further clinical table or subject crosswalk."),
    preserved_connections="58 images -> 36 author-mapped biopsy visits -> 18 source patients"))
joined["sample_id"] = STUDY + "__" + joined.patient_id + "__" + joined.Visit
joined = joined.set_index("tissue_id", verify_integrity=True)
samples_rows = []
for sample_id, group in joined.groupby("sample_id", sort=True):
    row = group.iloc[0]
    patient = f"{STUDY}__{row.patient_id}"
    paired = f"{STUDY}__{row.patient_id}__{'Post' if row.Visit == 'Pre' else 'Pre'}"
    sample = dict.fromkeys(columns("sample_metadata"))
    sample.update(study=STUDY, sample_id=sample_id, patient_key=patient,
                  source_subject_id=str(row["Subject ID"]), cohort_id="NCT02052908_IMC_subcohort",
                  center_id="MD_Anderson", source_sample_id=f"Subject {row['Subject ID']}:{row.Visit}",
                  sample_type="tissue", tissue_type="normal", anatomical_site="colorectal mucosa",
                  lesion_type="non_malignant", matched_sample_id=paired,
                  timepoint_label={"Pre": "pre_treatment", "Post": "post_treatment"}[row.Visit],
                  treatment_status_at_collection=f"{row.Visit}; randomized arm={row['Treatment Arm']}",
                  collection_method="biopsy", preservation_method="ffpe",
                  processing_method="TMA23 IMC tissue section",
                  primary_diagnosis="Lynch syndrome", primary_site="colorectal mucosa",
                  treatment_arm_reported=row["Treatment Arm"], visit_reported=row.Visit,
                  source_split_reported=row.split,
                  metadata_notes="Author meta_sample.tsv maps each ROI to Subject ID and Visit. "
                  "Subject+Visit denotes one study biopsy specimen; multiple cores/ROIs share this sample. "
                  "Physical biopsy accession/core map unavailable. Normal mucosa from predisposed carriers; "
                  "Individual age, sex, disease history and precise collection dates absent from supplied metadata. "
                  "Supplemental Table 1 Subject differs from author Subject ID; crosswalk absent, so dose is not assigned. "
                  "Post is the study post-intervention visit; no exact duration imputed per individual.")
    samples_rows.append(sample)
samples = pd.DataFrame(samples_rows, columns=columns("sample_metadata") +
                       ["treatment_arm_reported", "visit_reported", "source_split_reported"])
assert len(samples) == 36 and samples.patient_key.nunique() == 18
save_json(cache / "sample_linkage.json", dict(image_rows=58, sample_rows=36, patient_linked_samples=36,
          unique_patients=18, source_key="tissues.roi -> author.meta_sample.roi; Subject ID + Visit -> biopsy sample",
          evidence=[relative(tissues_path), relative(author_sample_path), relative(evidence / "article.html"), relative(supplement_path)],
          roi_connections=joined.reset_index()[["tissue_id", "roi", "sample_id", "patient_id", "Subject ID", "Visit"]].to_dict("records"),
          gaps=["Physical biopsy accessions/core identifiers absent; original published ROI frames retained",
                "Supplemental Subject differs from author Subject ID; dose crosswalk absent; optional clinical values remain null"] ))


def mapping(source: Path, table: str, source_field: str, target_table: str,
            target_field: str, rule: str, *, notes: str = "", join: str = "") -> dict:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table=table,
                source_field=source_field, target_table=target_table, target_field=target_field,
                join_keys=join, value_mapping="", transformation_rule=rule, time_origin_rule="",
                conflict_rule="Require source agreement; preserve unknown optional values as empty", notes=notes)


article = evidence / "article.html"
source_inventory_mapping = mapping(
    source_objects_path, "source_objects;native_verification;SPoRA tissues;OME-Zarr metadata",
    "object keys;shape;dtype;pixel size;scale", "image_metadata",
    "source_revision;source_format;source_axes;source_dtype;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;source_channel_count;channel_count;pixel_size_x_um;pixel_size_y_um;input_image_path;valid_pixels_path;intensity_description;split_method",
    "Use the canonical source object inventory, native verification and published OME-Zarr metadata. "
    "Retain the full published CYX frame and float32 values, centered square zero padding, and published scale; "
    "keep all zero-valued source pixels. Only a whole ROI with all admitted downstream markers zero is excluded.",
    join="image_id", notes="SPoRA 1 um scale is retained without resampling; original MCD-level intensities and coordinates are not asserted.")
source_inventory_mapping["source_file"] = ";".join(
    relative(source) for source in (source_objects_path, verification_path, tissues_path,
                                    raw / "imc/1_0mpp/images"))
field_rows = [
    mapping(author_sample_path, "meta_sample", "roi;Subject ID;Visit;Treatment Arm", "image_metadata", "sample_id;roi_id;region_annotation",
            "Join exact ROI with tissues.parquet; verify Subject ID/Visit/Treatment Arm equality for 58/58; "
            "construct one biopsy sample per patient_id+Visit (36), shared by all corresponding ROI views.",
            join="tissues.roi = meta_sample.roi", notes="58/58 images -> 36 biopsy-visit samples -> 18 patients; see sample_linkage.json"),
    mapping(tissues_path, "tissues", "tissue_id;original_file_name;roi;patient_id;original_patient_identifier;Subject ID",
            "image_metadata", "image_id;source_image_id;source_files;slide_id;roi_id;split_evidence",
            "Use exact tissue_id for image identity, preserve complete published ROI frame; TMA23 from original_file_name. "
            "Do not infer a physical core ID from ROI number.", join="tissue_id = source group name", notes="All original files are imctools/TMA23/<roi>/full_stack.ome.tiff; 58 unique ROI"),
    mapping(tissues_path, "tissues", "patient_id;Subject ID;original_patient_identifier;Visit", "sample_metadata",
            "sample_id;patient_key;source_subject_id;source_sample_id;matched_sample_id",
            "Verify one-to-one correspondence of source subject identifiers and patient_id. "
            "Namespace patient_id; join Subject ID+Visit to study-defined biopsy specimen; pair opposite visits.",
            join="Subject ID + Visit", notes="36/36 sample rows have patient_key; 18 unique patients. Source sample IDs are composite visit labels, not biopsy accessions."),
    mapping(article, "Materials and methods: Study design/Tissue micro array", "participants;FFPE colon biopsies;pre/post samples;TMA sections",
            "sample_metadata", "cohort_id;center_id;sample_type;tissue_type;anatomical_site;lesion_type;collection_method;preservation_method;processing_method;primary_diagnosis;primary_site",
            "Study-specific NCT02052908 IMC cohort at MD Anderson; tissue/normal/colorectal mucosa/non_malignant/biopsy/ffpe. "
            "Lynch syndrome retained as predisposition diagnosis; TMA23 processing label from original_file_name.",
            notes="Article describes 18 paired pre/post participants (36 biopsy specimens), with two blocks excluded, without public block IDs. "
            "Published source map still contains 36 subject-visit groups and 58 ROI; no additional images invented or discarded."),
    mapping(author_sample_path, "meta_sample", "Visit;Treatment Arm", "sample_metadata", "timepoint_label;treatment_status_at_collection",
            "Pre -> pre_treatment; Post -> post_treatment; preserve treatment arm and visit verbatim in status. "
            "Do not assign exact collection day or dose from cohort-level duration.", join="Subject ID + Visit"),
    mapping(channel_path, "channels + channels_per_tissue", "channel_name;original_channel_name;availability", "panel_mapping", "panel_id;source_channel_index;source_name",
            "Preserve canonical channel table order filtered by per-tissue availability; require agreement with "
            "all embedded labels, ordered metadata, native proofs and author metadata_nf.csv.",
            notes="Source contains one 39-channel panel; official spora_io/datasets/multiplex.py rule already localized."),
    mapping(s1_path, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
            "marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
            "Exact study+source-name join; discard 101Ru and preserve other 38 source rows in order, "
            "including nuclear 191Ir/193Ir. No source aliases or duplicate downstream markers.", join="Study + Original marker"),
    source_inventory_mapping,
    mapping(supplement_path, "Supplemental Table 1", "Subject;Treatment group;Number of ROIs(pre-treatment);Number of ROIs(post-treatment)",
            "sample_metadata", "metadata_notes",
            "Supplement caption confirms 18 paired subjects/36 biopsy specimens and 58 ROI. Subject numerals conflict with "
            "author meta_sample.tsv arms/counts. Retain exact author ROI mapping; no per-patient dose or supplemental Subject assigned.",
            notes="See sample-scope review_results.csv record_id=public_clinical_crosswalk. Individual age/sex/stage/survival absent; "
                  "dose is a specific unlinked source field, not a presumed unavailable measurement."),
    mapping(workspace_path, "Workspace", "Sample;Group;Keyword", "sample_metadata", "metadata_notes",
            "Inspect original author's workspace annotations for dose/subject crosswalk: 58 sample nodes, "
            "no Naproxen/placebo/high-low-dose attributes. No clinical value imported from default workspace keywords.",
            notes="Sample-scope review_results.csv record_id=public_clinical_crosswalk records the workspace inspection."),
    mapping(evidence / "DataSheet_1.pdf", "Supplemental Figures 1-3, pages 1-3", "figures and group labels",
            "sample_metadata", "metadata_notes",
            "Preparer inspected all three publisher supplement pages: dose-group summaries and tSNE figures; "
            "no per-subject identity crosswalk or second clinical table. No individual clinical value imported."),
]
for target, source, definition in [
    ("treatment_arm_reported", "Treatment Arm", "Verbatim randomized arm from the author's meta_sample.tsv"),
    ("visit_reported", "Visit", "Verbatim pre/post visit label from the author's meta_sample.tsv"),
    ("source_split_reported", "split", "SPORA source split label; retained as source metadata, not a new CellWeft split decision"),
]:
    field_rows.append(mapping(tissues_path if source == "split" else author_sample_path,
                             "tissues" if source == "split" else "meta_sample", source,
                             "sample_metadata", target, "Preserve source value verbatim per sample.",
                             join="Subject ID + Visit", notes=json.dumps(dict(data_type="string", definition=definition))))
fields = pd.DataFrame(field_rows, columns=columns("field_mapping"))

# Publish metadata first; each completed TIFF then extends the resumable index.
index_path = output / "images.parquet"
existing = {row["image_id"]: row for row in pd.read_parquet(index_path).to_dict("records")} if index_path.exists() else {}
if args.metadata_only:
    save_input_tables(output, pd.DataFrame(existing.values(), columns=columns("image_metadata")), samples, panel, fields)
    progress("metadata_ready_waiting_for_managed_write_slot", len(existing), sample_rows=36,
             sample_linked_images=len(existing), patient_linked_samples=36, unique_patients=18,
             estimated_uncompressed_output_bytes=estimated_output)
    print(json.dumps(dict(study=STUDY, stage="metadata_ready", exported_images=len(existing),
                          sample_rows=36, patient_linked_samples=36, unique_patients=18)), flush=True)
    raise SystemExit(0)

all_receipts = {}
excluded = []
for item in inventory:
    image_id = item["image_id"]
    _, h, w = item["shape"]
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    destination = output / "images" / image_id / "input.tiff"
    receipt_path = receipts / f"{image_id}.json"
    revision = json.dumps(dict(spora_scale="imc/1_0mpp", files=item["files"]), separators=(",", ":"))
    reusable = receipt_path.exists() and destination.is_file()
    if reusable:
        receipt = json.loads(receipt_path.read_text())
        reusable = receipt["source_revision"] == revision and receipt["channel_names"] == kept.marker_id.tolist()
    if not reusable:
        if shutil.disk_usage(root).free < 100 * 1024**3 + 38 * side * side * 4:
            progress("capacity_blocked_NAS", len(all_receipts), current_image=image_id)
            raise RuntimeError("Insufficient NAS reserve for next image")
        if shutil.disk_usage(temporary).free < 64 * 1024**3 + 39 * h * w * 4:
            progress("capacity_blocked_local", len(all_receipts), current_image=image_id)
            raise RuntimeError("Insufficient local reserve for staging")
        progress("decoding_native_source", len(all_receipts), current_image=image_id)
        source = SourceSpec(image_id, panel_id, "zarr", root / item["source_path"] / "0", tuple(range(39)),
                            pixel_size_um=(1.0, 1.0), source_revision=revision)
        stage_path = temporary / f"{image_id}.npy"
        array = np.lib.format.open_memmap(stage_path, mode="w+", dtype="float32", shape=tuple(item["shape"]))
        try:
            with open_source(source) as reader:
                for window in windows((h, w), 256):
                    x0, y0, x1, y1 = window
                    array[:, y0:y1, x0:x1] = reader.read_window(list(range(39)), window)
            array.flush()
            progress("exporting_input", len(all_receipts), current_image=image_id)
            exported = export_input(StagedReader(array, source), panel, destination, tile_size=256)
            receipt = dict(image_id=image_id, source_revision=revision, channel_names=kept.marker_id.tolist(),
                           exported=exported, preparer_pixel_checks=[])
            if exported["admitted"]:
                probe = int(np.flatnonzero((np.asarray(exported["channel_nonzero_pixels"]) > 0) &
                                           (kept.role.to_numpy() == "downstream"))[0])
                probe_channel = int(kept.iloc[probe].source_channel_index)
                sy, sx = np.unravel_index(np.argmax(np.abs(array[probe_channel])), (h, w))
                bw, bh = min(w, 32), min(h, 32)
                bx, by = max(0, min(w-bw, int(sx)-16)), max(0, min(h-bh, int(sy)-16))
                signal_window = (bx, by, bx+bw, by+bh)
                receipt["signal_probe"] = dict(source_channel_index=probe_channel,
                                                source_name=source_names[probe_channel], xy=[int(sx), int(sy)])
                with tifffile.TiffFile(destination) as tif:
                    blocks = [(0, 0, min(w, 32), min(h, 32)),
                              signal_window,
                              (max(0, w-32), max(0, h-32), w, h)]
                    for x0, y0, x1, y1 in blocks:
                        dest_window = (x0+left, y0+top, x1+left, y1+top)
                        for channel in kept.itertuples(index=False):
                            values = read_tiff_window(tif, int(channel.output_channel_index), dest_window)
                            assert values.dtype == array.dtype
                            assert np.array_equal(values, array[channel.source_channel_index, y0:y1, x0:x1])
                        receipt["preparer_pixel_checks"].append(dict(source_window=[x0,y0,x1,y1],
                                output_window=dest_window, all_channels=38, exact=True))
                    for pad_window in [(0,0,left,side), (0,0,side,top),
                                       (left+w,0,side,side), (0,top+h,side,side)]:
                        if pad_window[2] <= pad_window[0] or pad_window[3] <= pad_window[1]:
                            continue
                        for channel in range(38):
                            assert not read_tiff_window(tif, channel, pad_window).any()
                overview = []
                for name in ("191Ir", "CK", "CD45"):
                    plane = np.asarray(array[source_names.index(name), ::3, ::3]).copy()
                    high = np.quantile(plane, 0.99)
                    overview.append(np.clip(plane/high, 0, 1) if high > 0 else plane)
                plt.imsave(cache / f"overview_{image_id}.png", np.stack(overview, axis=-1))
            save_json(receipt_path, receipt)
        finally:
            array._mmap.close()
            stage_path.unlink(missing_ok=True)
    if not receipt["exported"]["admitted"]:
        excluded.append(dict(image_id=image_id, reason="All retained downstream markers zero in complete published ROI"))
        existing.pop(image_id, None)
        continue
    with tifffile.TiffFile(destination) as tif:
        names = [e.attrib["Name"] for e in ET.fromstring(tif.ome_metadata).iter() if e.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()
        assert tif.series[0].shape == (38, side, side) and tif.series[0].dtype == np.dtype("float32")
    row = joined.loc[image_id]
    values = dict.fromkeys(columns("image_metadata"))
    values.update(study=STUDY, image_id=image_id, sample_id=row.sample_id, panel_id=panel_id, image_kind="roi",
                  source_image_id=image_id, source_files=[item["source_path"]], source_format="ome_zarr",
                  source_axes="CYX", source_dtype="float32", source_revision=revision, slide_id="TMA23", roi_id=row.roi,
                  region_annotation=json.dumps(dict(roi=row.roi, Subject_ID=str(row["Subject ID"]),
                                        Treatment_Arm=row["Treatment Arm"], Visit=row.Visit,
                                        original_file_name=row.original_file_name), ensure_ascii=False),
                  source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                  pad_left_px=left, pad_top_px=top, pad_right_px=side-w-left, pad_bottom_px=side-h-top,
                  height_px=side, width_px=side, source_channel_count=39, channel_count=38,
                  pixel_size_x_um=1.0, pixel_size_y_um=1.0, input_image_path=relative(destination),
                  intensity_description="Unmodified SPORA-published float32 intensities, highest published 1_um scale; "
                                        "lossless TIFF and zero padding only; no normalization; not asserted raw MCD counts.",
                  split_method="already_independent", split_evidence=f"Author meta_sample.tsv {row.roi}; "
                  f"SPORA tissue_id={image_id}, original_file_name={row.original_file_name}; full published frame.",
                  metadata_notes="Author ROI/Subject ID/Visit map confirmed. Physical core and biopsy accession absent; "
                                 "multiple ROI from one biopsy visit share sample_id. Published dense pixels kept; "
                                 "zero-valued pixels do not imply an acquisition gap.")
    existing[image_id] = values
    images = pd.DataFrame(existing.values(), columns=columns("image_metadata"))
    save_input_tables(output, images, samples, panel, fields)
    all_receipts[image_id] = receipt
    progress("image_saved", len(all_receipts), current_image=image_id,
             sample_rows=36, sample_linked_images=len(images), patient_linked_samples=36, unique_patients=18)
    print(f"saved {len(all_receipts)}/58 {image_id}", flush=True)

# Prepare source overviews and clearly distinguish self-checks from acceptance.
for page, start in enumerate(range(0, len(selected), 15), 1):
    figure, axes = plt.subplots(3, 5, figsize=(13, 8))
    for ax, image_id in zip(axes.flat, selected[start:start+15]):
        path = cache / f"overview_{image_id}.png"
        if path.exists():
            ax.imshow(plt.imread(path))
        ax.set_title(f"{image_id.removeprefix('bowen2023naproxen_')}\n{joined.loc[image_id].roi}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    figure.suptitle("Preparer overview: red=191Ir, green=CK, blue=CD45 (display-only P99)")
    figure.tight_layout()
    figure.savefig(cache / f"overview_sheet_{page}.png", dpi=130)
    plt.close(figure)

save_json(cache / "excluded_images.json", excluded)
scopes = pd.read_csv(TEMPLATES / "review_checks.csv").set_index("check_id").scope
details = {
    "S01": "58 original verified ROI reconciled against 177 original objects and accepted normal-mucosa scope; "
           f"{len(all_receipts)} exported, {len(excluded)} excluded; see original proofs and source_inventory.json; no source hashes repeated.",
    "I01": "All published image identities, full source paths and corresponding TIFFs are unique and locatable.",
    "I02": f"{len(all_receipts)} images link to 36 sample rows, 18 patients and the actual 39-channel panel.",
    "I03": "Entire published ROI frame retained; native float32, 1_um published scale, CYX square, centered zero padding.",
    "I04": "Author meta_sample.tsv confirms ROI/Subject ID/Visit. Independent published ROI frames preserved. "
           "Physical core accession unavailable. Preparer visual check recorded separately in overview_review.json.",
    "P01": "All 39-channel orders agree among author metadata_nf.csv, Parquet availability-filtered order, "
           "localized ordered JSON, source OMERO labels and reused native proofs.",
    "P02": "S1 exact study+marker mapping; 101Ru excluded; 38 channels retained, including 191Ir/193Ir nuclear inputs; "
           "source_channel_index differs from output_channel_index; downstream markers unique.",
    "P03": "All completed TIFF headers reopened: 38-channel float32 square CYX, exact channel names/order and source scale.",
    "I05": "Preparer compares all 38 output channels in source-origin, confirmed nonzero-marker and lower-right <=32px blocks per image, "
           "and all padding pixels; coordinates in receipts. Independent review sampling still pending.",
    "M01": "Local metadata, original paper, publisher Supplemental Table 1, original Zenodo publication and author ROI/channel attachments inspected. "
           "58/58 images connect to 36 biopsy-visit samples; 36/36 samples connect to 18 patients. "
           "Grouping uses original Subject ID+Visit and paper's paired-biopsy design, never image-name inference. "
           "Physical biopsy accessions unavailable; no unresolved image/patient identity linkage in this published scope.",
    "M02": "All populated values derive from original ROI map or explicitly study-wide tissue/method facts. "
           "Supplemental Table 1 has conflicting Subject numbering versus the author ROI table; dose crosswalk remains unresolved "
           "and no dose is assigned. Comparison and clinical_numbering_gap.json retain specific conflicts. "
           "Individual age/sex/history/survival absent. Source split retained as reported metadata only.",
    "D01": "Preparation and preparer self-check only; different CLI context independent review and controller acceptance pending.",
}
review_rows = []
for check, message in details.items():
    scope = scopes[check]
    ids = {"study": [STUDY], "image": list(all_receipts), "panel": [panel_id], "sample": samples.sample_id.tolist()}[scope]
    for identity in ids:
        review_rows.append(dict(study=STUDY, preparer="codex_Bowen2023_IMC_Colorectal_0018_prepare_0", reviewer=None,
                               scope=scope, record_id=identity, check_id=check,
                               result="unresolved" if check in {"D01", "M02"} or (check == "I04" and not (cache / "overview_review.json").exists()) else "pass",
                               source_evidence=relative(cache / "source_inventory.json") + ";" + relative(author_sample_path) + ";" + relative(supplement_path),
                               output_evidence=relative(index_path) + ";" + relative(receipts),
                               details="Preparer self-check; independent reviewer unassigned. " + message))
review_path = output / "review_results.csv"
reviews = pd.DataFrame(review_rows, columns=columns("review_results"))
if review_path.exists():
    previous = pd.read_csv(review_path)
    reviews = pd.concat([previous[previous.reviewer.notna()], reviews], ignore_index=True)
with formal_publish_lease(review_path, directory=review_path.parent):
    reviews.to_csv(review_path, index=False, encoding="utf-8-sig")
summary = dict(study=STUDY, status="prepared", expected_images=58, exported_images=len(all_receipts),
               sample_rows=36, sample_linked_images=len(all_receipts), patient_linked_samples=36, unique_patients=18,
               source_channel_count=39, output_channel_count=38, native_dtype="float32",
               output_file_bytes=sum((root / p).stat().st_size for p in images.input_image_path),
               preparer_exact_blocks=sum(len(r["preparer_pixel_checks"]) for r in all_receipts.values()),
               independent_review="pending", excluded_images=excluded,
               issues=["独立复核与主控验收待执行", "补充表Subject与作者ROI表编号不一致；剂量crosswalk缺失，M02保留unresolved；ROI/sample/patient连接完整"],
               controller_requests=["安排不同CLI上下文独立复核58张图、36份sample及18位patient的输入交付。",
                                    "确认Supplemental Table 1与meta_sample.tsv编号冲突的临床处理结论；没有crosswalk前不补入个体剂量。",
                                    "共享export_input按通道读取会反复解码SPORA全通道chunk；本study分块暂存local memmap，后续可统一迁移。"],
               source_urls=["https://doi.org/10.3389/fimmu.2023.1162669", "https://zenodo.org/records/7655637",
                            "https://zenodo.org/api/records/7655637/files/meta_sample.tsv/content", "https://doi.org/10.3389/fimmu.2023.1162669.s002",
                            "https://livingobjects.ebi.ac.uk/bioimaging-01-pub/spora-submissions/bowen2023naproxen/"],
               evidence_paths=[relative(cache / "source_inventory.json"), relative(author_sample_path),
                               relative(cache / "sample_linkage.json"), relative(cache / "clinical_numbering_gap.json"), relative(review_path)])
save_json(cache / "input_delivery_summary.json", summary)
progress("prepared_pending_independent_review", len(all_receipts), sample_rows=36,
         sample_linked_images=len(all_receipts), patient_linked_samples=36, unique_patients=18)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
