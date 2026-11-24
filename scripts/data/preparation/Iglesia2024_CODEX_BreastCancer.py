"""Prepare native CODEX inputs from the 48 qualified HTAN WUSTL source images.

Inputs: original OME-TIFFs, qualified-intake proofs, HTAN portal metadata, and S1.
Outputs: independent square input.tiff files and study input/clinical tables on NAS.
Stage and validate locally; publish each closed image under the shared NAS lease.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative
from cellweft.data.raw_consolidation import file_identity

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

from cellweft.data.preparation.study_input import (
    TEMPLATES, export_input, panel_from_s1, save_input_tables,
)
from cellweft.data.io.readers import read_tiff_window
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease


STUDY = "Iglesia2024_CODEX_BreastCancer"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(ROOT, STUDY)
SOURCE = META / "preparation_sources"
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = ROOT / "processed" / STUDY
INTAKE = META / "qualified_intake"
JOB = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/jobs" / STUDY / "0213_prepare_0"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TMP = Path("/tmp/study_input_batch") / STUDY
STATE: dict = {}
PUBLISHED_COUNT = 0


class WaitingIO(RuntimeError):
    """A closed, resumable task is waiting for shared I/O or reserved space."""


@contextmanager
def bounded_io(operation: str):
    """Bound contention waits; completed local images remain available on retry."""
    def timeout(signum: int, frame: object) -> None:
        raise WaitingIO(operation)
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def image_publication(destination: Path):
    """Acquire NAS, study, then input locks; the timer covers acquisition only."""
    with ExitStack() as stack:
        with bounded_io("Waiting for NAS/study/input publication lease"):
            stack.enter_context(formal_publish_lease(destination, directory=OUTPUT))
            stack.enter_context(resource_lock(destination.parent, name=".input.lock"))
        yield


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE) or path.is_relative_to(JOB):
        return str(path)
    return metadata_source_relative(path, ROOT).as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def progress(stage: str, completed: int, **details: object) -> None:
    STATE.update(details)
    state = dict(study=STUDY, stage=stage, status="waiting_io" if stage == "waiting_io" else stage,
                 phase="prepare", scientific_repair_round=0, expected_images=48,
                 completed_images=completed, updated_at=datetime.now(timezone.utc).isoformat(),
                 gaps=["HTAN registers imaging specimens separately; common biological sampling across sections/views unresolved",
                       "Paper reports 47 slides/13 cases; release selection has 48 images/14 participants",
                       "Independent review pending"], **STATE)
    write_json(JOB / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def report_interruption(kind: type, error: BaseException, traceback: object) -> None:
    if isinstance(error, WaitingIO):
        progress("waiting_io", PUBLISHED_COUNT, waiting_reason=str(error),
                 checkpoint="Re-run the same command; reuse computed image and completed export receipts")
    else:
        sys.__excepthook__(kind, error, traceback)


sys.excepthook = report_interruption


def publish_tables(images: pd.DataFrame, samples: pd.DataFrame, panel: pd.DataFrame,
                   fields: pd.DataFrame) -> None:
    """Merge and publish study tables through the shared publication path."""
    global PUBLISHED_COUNT
    with bounded_io("Waiting for metadata publication lease"):
        save_input_tables(OUTPUT, images, samples, panel, fields)
    PUBLISHED_COUNT = len(images)


class PlaneReader:
    """Decode one native channel at a time, avoiding repeated 3-row strip reads.

    This study-local adaptation of the shared TIFF reader is needed because
    horizontal tile requests otherwise decode every full-width source strip
    repeatedly. Only one YX plane is cached; no full CYX WSI is materialized.
    It should later migrate to a bounded strip/plane cache in the shared reader.
    """

    def __init__(self, tif: tifffile.TiffFile, pixel_size: tuple[float, float], preview_channels: list[int]):
        self.tif = tif
        self.shape = (len(tif.pages), *tif.pages[0].shape)
        self.dtype = tif.pages[0].dtype
        self.source = SimpleNamespace(pixel_size_um=pixel_size, path=Path(tif.filehandle.path))
        self.channel = None
        self.plane = None
        self.preview_channels = preview_channels
        self.previews = {}

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        assert len(channels) == 1
        channel = channels[0]
        if channel != self.channel:
            self.plane = None
            self.plane = self.tif.pages[channel].asarray(maxworkers=2)
            self.channel = channel
            if channel in self.preview_channels:
                stride = max(1, max(self.shape[1:]) // 640)
                self.previews[channel] = self.plane[::stride, ::stride].copy()
        x0, y0, x1, y1 = window
        return self.plane[None, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


class NativeInputWriter(tifffile.TiffWriter):
    """Use lossless Zstd for this 312-GiB uncompressed WSI delivery.

    The shared exporter fixes deflate. A real 8192-square native uint8 block
    took 2.77 s with zlib6 versus 0.127 s with Zstd1, both exact round-trips;
    Zstd was 11% larger. See cache codec_benchmark.json. This process-local
    adapter should migrate to an explicit shared exporter codec parameter.
    """

    def write(self, *args: object, **kwargs: object) -> object:
        if kwargs.get("compression") == "deflate":
            kwargs["compression"] = "zstd"
            kwargs["compressionargs"] = {"level": 1}
        return super().write(*args, **kwargs)


def save_overview(reader: PlaneReader, selected: pd.DataFrame, image_id: str, destination: Path) -> None:
    """Render representative native fields; scaling is for display only."""
    channel_images = []
    for row in selected.itertuples():
        values = reader.previews[row.source_channel_index]
        high = np.percentile(values[values > 0], 99) if (values > 0).any() else 1
        scaled = np.uint8(np.clip(values.astype(float) / max(high, 1), 0, 1)*255)
        tile = Image.new("RGB", (360, 400), "white")
        thumbnail = Image.fromarray(scaled).convert("RGB")
        thumbnail.thumbnail((350, 350))
        tile.paste(thumbnail, ((360-thumbnail.width)//2, 40))
        ImageDraw.Draw(tile).text((5, 10), row.source_name, fill="black")
        channel_images.append(tile)
    canvas = Image.new("RGB", (360*len(channel_images), 440), "white")
    ImageDraw.Draw(canvas).text((5, 5), image_id + " | display P99 only", fill="black")
    for index, tile in enumerate(channel_images):
        canvas.paste(tile, (360*index, 30))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def header(path: Path) -> dict:
    with bounded_io("Waiting for source header read lease"), formal_read_lease(path), tifffile.TiffFile(path) as tif:
        pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        names = [x.attrib["Name"] for x in pixels.findall("{*}Channel")]
        h, w = tif.pages[0].shape
        assert len(tif.pages) == int(pixels.attrib["SizeC"]) == len(names)
        assert int(pixels.attrib["SizeZ"]) == int(pixels.attrib["SizeT"]) == 1
        assert all(p.shape == (h, w) and p.dtype == tif.pages[0].dtype for p in tif.pages)
        return dict(names=names, height=h, width=w, dtype=str(tif.pages[0].dtype),
                    pixel_size=[float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])],
                    tiled=tif.pages[0].is_tiled, rows_per_strip=tif.pages[0].rowsperstrip)


def verify_pixels(source: Path, destination: Path, panel: pd.DataFrame, info: dict) -> list[dict]:
    """Preparer self-check: all retained channels in an internal and edge block."""
    h, w = info["height"], info["width"]
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    blocks = [(w//2, h//2, min(w//2+32, w), min(h//2+32, h)),
              (0, 0, min(32, w), min(32, h))]
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    with formal_read_lease(destination), tifffile.TiffFile(source) as src, tifffile.TiffFile(destination) as dst:
        assert len(dst.pages) == len(kept)
        assert dst.pages[0].shape == (side, side) and str(dst.pages[0].dtype) == info["dtype"]
        names = [x.attrib["Name"] for x in ET.fromstring(dst.ome_metadata).findall(".//{*}Channel")]
        assert names == kept.marker_id.tolist()
        for row in kept.itertuples():
            for block in blocks:
                x0, y0, x1, y1 = block
                a = read_tiff_window(src, row.source_channel_index, block)
                b = read_tiff_window(dst, row.output_channel_index,
                                     (x0+left, y0+top, x1+left, y1+top))
                assert np.array_equal(a, b) and a.dtype == b.dtype
            if left:
                assert not read_tiff_window(dst, row.output_channel_index, (0, 0, left, min(32, side))).any()
            if top:
                assert not read_tiff_window(dst, row.output_channel_index, (0, 0, min(32, side), top)).any()
    return [dict(source_window=list(b), all_retained_channels=True, equal=True) for b in blocks]


# Resolve original sources and reuse their existing native verification identity.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--action", choices=["inspect", "stage", "overview", "export"], default="export")
parser.add_argument("--image-id", help="Process one qualified image; merge existing index")
parser.add_argument("--job", type=Path, default=JOB, help="Current controller job directory")
args = parser.parse_args()
JOB = args.job
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest = read_json(META / "source_manifest.json")
source_panels = {p["panel_id"]: p for p in read_json(META / "source_ordered_channel_map.json")["panels"]}
sources = manifest["accepted_files"]
current_versions = read_json(META / "current_native_versions.json")["members"]
if args.action == "stage":
    # Byte-copy original sources to the local temporary disk under a read slot.
    # TIFF generation remains exclusively under the formal write-slot command.
    staged_directory = TMP / "staged_sources"
    staged_directory.mkdir(parents=True, exist_ok=True)
    for entry in sources:
        source = ROOT / entry["relative_path"]
        staged = staged_directory / source.name
        binding = staged.with_suffix(staged.suffix + ".json")
        st = source.stat()
        identity = dict(source_size=st.st_size, source_mtime_ns=st.st_mtime_ns)
        assert st.st_size == entry["expected_size_bytes"]
        if staged.exists() and binding.exists() and read_json(binding) == identity and staged.stat().st_size == st.st_size:
            continue
        if shutil.disk_usage(TMP).free < 64*1024**3 + st.st_size:
            raise RuntimeError("Local staging would consume the 64 GiB reserve")
        partial = staged.with_suffix(staged.suffix + ".copying")
        with formal_read_lease(source):
            shutil.copyfile(source, partial)
        assert partial.stat().st_size == st.st_size
        partial.replace(staged)
        write_json(binding, identity)
        print(json.dumps(dict(staged_source=entry["source_id"], size=st.st_size)), flush=True)
    raise SystemExit(0)
if args.action == "overview":
    # Render only staged local source pixels; no image index or TIFF is changed.
    for panel_id, native in source_panels.items():
        entry = next(e for e in sources if e["panel_id"] == panel_id)
        staged = TMP / "staged_sources" / Path(entry["filename"]).name
        binding = staged.with_suffix(staged.suffix + ".json")
        destination = CACHE / "overviews" / (panel_id.split("#")[-1] + ".png")
        if destination.exists() or not binding.exists():
            continue
        assert staged.stat().st_size == read_json(binding)["source_size"]
        info = header(staged)
        assert info["names"] == native["native_channel_order_from_audit"]
        panel = panel_from_s1(STUDY, panel_id, info["names"], S1)
        selected = panel[panel.marker_id.isin(["DAPI", "PanCK", "CD3e"]) & panel.keep_channel]
        with tifffile.TiffFile(staged) as tif:
            reader = PlaneReader(tif, tuple(info["pixel_size"]), selected.source_channel_index.tolist())
            for channel in selected.source_channel_index:
                reader.read_window([int(channel)], (0, 0, 1, 1))
            save_overview(reader, selected, staged.name.removesuffix(".ome.tiff"), destination)
        print(str(destination), flush=True)
    raise SystemExit(0)
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.read_csv(TEMPLATES / "image_metadata.csv")
PUBLISHED_COUNT = len(images)
progress("source_inspection", len(images))
panels = pd.concat([panel_from_s1(STUDY, p["panel_id"], p["native_channel_order_from_audit"], S1)
                    for p in source_panels.values()], ignore_index=True)
headers = {}
source_records = []
previous_inspection = {row["source_id"]: row for row in read_json(SOURCE / "source_inspection.json")["included"]}
portal_images = read_json(SOURCE / "portal_images.json")
for entry in sources:
    path = ROOT / entry["relative_path"]
    proof_path = INTAKE / "proofs" / (entry["filename"].replace("/", "__") + ".json")
    proof = read_json(proof_path)
    portal = portal_images[entry["filename"]]
    assert portal["DataFileID"] == entry["portal_data_file_id"]
    assert portal["ParentBiospecimenID"] == entry["portal_parent_biospecimen_id"]
    assert portal["ParticipantID"] == entry["portal_participant_id"]
    assert portal["PassedQC"] == "yes"
    st = path.stat()
    expected_identity = current_versions[entry["relative_path"]]["current"]
    actual_identity = file_identity(path)
    assert all(actual_identity[key] == expected_identity[key]
               for key in ("device", "inode", "size", "mtime_ns"))
    # NFS device numbers vary across remounts; retain all file-specific checks.
    identity_fields = ("size", "mtime_ns", "ctime_ns", "ino")
    assert {k: proof["file_stat"][k] for k in identity_fields} == {
        k: getattr(st, "st_"+k) for k in identity_fields}
    assert st.st_size == entry["expected_size_bytes"]
    assert proof["source_identity"] == {k: entry[k] for k in ("filename", "source_uuid", "drs_uri", "did", "expected_size_bytes", "relative_path", "panel_id")}
    cached_header = previous_inspection.get(entry["source_id"])
    staged_header = TMP / "staged_sources" / path.name
    staged_identity = staged_header.with_suffix(staged_header.suffix + ".json")
    if (cached_header and cached_header["source_file"] == entry["relative_path"]
            and cached_header["proof"] == relative(proof_path)
            and cached_header["names"] == proof["channel_names"]):
        # The intake proof above still matches the full native file identity.
        info = {key: cached_header[key] for key in
                ("names", "height", "width", "dtype", "pixel_size", "tiled", "rows_per_strip")}
    elif (staged_header.exists() and staged_identity.exists()
            and read_json(staged_identity) == dict(source_size=st.st_size, source_mtime_ns=st.st_mtime_ns)
            and staged_header.stat().st_size == st.st_size):
        info = header(staged_header)
    else:
        info = header(path)
    assert info["names"] == proof["channel_names"] == source_panels[entry["panel_id"]]["native_channel_order_from_audit"]
    headers[entry["source_id"]] = info
    source_records.append(dict(source_id=entry["source_id"], source_file=entry["relative_path"],
                               proof=relative(proof_path), portal_passed_qc=portal["PassedQC"],
                               portal_biospecimen_id=portal["ParentBiospecimenID"], **info))
source_directory = ROOT / "raw/public" / STUDY / "HTAN_Release_4.0"
local_inventory = [relative(p) for p in sorted(source_directory.rglob("*")) if p.is_file()]
write_json(CACHE / "source_inspection.json", dict(included=source_records, excluded=manifest["excluded_files"],
           local_inventory=local_inventory,
           existing_proofs_reused=True, full_hash_repeated=False,
           estimated_uncompressed_output_bytes=sum(max(x["height"], x["width"])**2 *
               int(panels[panels.panel_id.eq(e["panel_id"])].keep_channel.sum()) * np.dtype(x["dtype"]).itemsize
               for e in sources for x in [headers[e["source_id"]]])))

# HTAN explicitly identifies specimens and participants. It does not supply a
# common sampling hierarchy for imaging dates/left/right sections; retain each
# registered specimen pending evidence for collapsing biological samples.
biospecimens = read_json(SOURCE / "biospecimenByBiospecimenID.json")
diagnoses = read_json(SOURCE / "diagnosisByParticipantID.json")
demographics = read_json(SOURCE / "demographicsByParticipantID.json")
sample_rows = []
for specimen_id in sorted({e["portal_parent_biospecimen_id"] for e in sources}):
    bio = biospecimens[specimen_id]
    patient = bio["HTANParentID"]
    diagnosis, demographic = diagnoses[patient], demographics[patient]
    assert patient in {e["portal_participant_id"] for e in sources if e["portal_parent_biospecimen_id"] == specimen_id}
    sample_rows.append(dict(study=STUDY, sample_id=specimen_id, patient_key=patient,
        source_subject_id=patient, source_sample_id=bio["SourceHTANBiospecimenID"],
        cohort_id="HTAN_WUSTL_BRCA23", center_id="HTAN WUSTL", sample_type="tissue",
        tissue_type={"Primary": "tumor"}.get(bio["TumorTissueType"], ""),
        anatomical_site=diagnosis["SiteofResectionorBiopsy"],
        lesion_type={"Primary": "primary"}.get(bio["TumorTissueType"], ""),
        collection_method={"Surgical Resection": "resection"}[bio["AcquisitionMethodType"]],
        preservation_method={"Formalin fixed paraffin embedded - FFPE": "ffpe"}[bio["PreservationMethod"]],
        timepoint_label={"Baseline": "baseline", "End of Treatment": "post_treatment"}[bio["TimepointLabel"]],
        primary_diagnosis=diagnosis["PrimaryDiagnosis"], primary_site=diagnosis["TissueorOrganofOrigin"],
        sex_reported=demographic["Gender"].lower(),
        vital_status={"Alive": "alive", "Dead": "deceased"}[demographic["VitalStatus"]],
        metadata_notes="Confirmed HTAN imaging biospecimen; common biological sampling across sections/views unresolved. "
                       "CollectionDaysfromIndex not transferred: index origin unresolved. Paper clinical sample IDs not joined by name inference. "
                       f"Source TumorTissueType={bio['TumorTissueType']}; Post therapy neoadjuvant does not define residual tissue/lesion category; those values remain empty."))
samples = pd.DataFrame(sample_rows).reindex(columns=pd.read_csv(TEMPLATES / "sample_metadata.csv").columns)

mapping_rows = []
def field(source_file: str, table: str, source_field: str, target_table: str, target_field: str,
          join_keys: str, rule: str, notes: str = "", value_mapping: str = "") -> None:
    mapping_rows.append(dict(study=STUDY, source_file=source_file, source_sheet_or_table=table,
        source_field=source_field, target_table=target_table, target_field=target_field,
        join_keys=join_keys, value_mapping=value_mapping, transformation_rule=rule,
        time_origin_rule="", conflict_rule="Preserve original values; unresolved sampling hierarchy is not inferred", notes=notes))

field(relative(META / "source_manifest.json"), "accepted_files", "portal_parent_biospecimen_id;portal_participant_id",
      "image_metadata", "sample_id", "filename -> portal_parent_biospecimen_id -> HTANBiospecimenID",
      "Use registered HTAN biospecimen ID; confirm HTANParentID matches portal participant ID",
      "48/48 images linked; 48 registered specimens, 14 participants. Common biological sampling across sections/views unresolved.")
sample_fields = {
    "source_sample_id": ("Biospecimen", "SourceHTANBiospecimenID"),
    "patient_key": ("Biospecimen", "HTANParentID"),
    "source_subject_id": ("Biospecimen", "HTANParentID"),
    "anatomical_site": ("Diagnosis", "SiteofResectionorBiopsy"),
    "lesion_type": ("Biospecimen", "TumorTissueType"),
    "tissue_type": ("Biospecimen", "TumorTissueType"),
    "collection_method": ("Biospecimen", "AcquisitionMethodType"),
    "preservation_method": ("Biospecimen", "PreservationMethod"),
    "timepoint_label": ("Biospecimen", "TimepointLabel"),
    "primary_diagnosis": ("Diagnosis", "PrimaryDiagnosis"),
    "primary_site": ("Diagnosis", "TissueorOrganofOrigin"),
    "sex_reported": ("Demographics", "Gender"),
    "vital_status": ("Demographics", "VitalStatus"),
}
category_mappings = {
    "timepoint_label": {"Baseline": "baseline", "End of Treatment": "post_treatment"},
    "lesion_type": {"Primary": "primary", "Post therapy neoadjuvant": ""},
    "tissue_type": {"Primary": "tumor", "Post therapy neoadjuvant": ""},
    "collection_method": {"Surgical Resection": "resection"},
    "preservation_method": {"Formalin fixed paraffin embedded - FFPE": "ffpe"},
    "vital_status": {"Alive": "alive", "Dead": "deceased"},
}
for target, (table, source_field) in sample_fields.items():
    filename = {"Biospecimen": "biospecimenByBiospecimenID.json", "Diagnosis": "diagnosisByParticipantID.json",
                "Demographics": "demographicsByParticipantID.json"}[table]
    field(relative(SOURCE / filename), table, source_field, "sample_metadata", target,
          "HTANBiospecimenID; HTANParentID=HTANParticipantID",
          "Apply explicit category mapping; lowercase sex; otherwise copy source",
          notes="Post-therapy label alone does not define residual tissue or lesion category; keep these cells empty." if target in {"tissue_type", "lesion_type"} else "",
          value_mapping=json.dumps(category_mappings[target]) if target in category_mappings else "")
field(relative(META / "source_manifest.json"), "study selection", "dataset_name;cohort;atlas", "sample_metadata",
      "study;sample_id;cohort_id;center_id;sample_type;tissue_type", "HTANBiospecimenID",
      "Study constants from qualified breast cohort; sample_id is the explicit registered biospecimen ID; tissue_type uses the separate TumorTissueType mapping")
field(relative(SOURCE / "43018_2024_773_MOESM2_ESM.xlsx"), "S1 Cohort_information", "Sample_ID;Case",
      "sample_metadata", "metadata_notes", "No authoritative cross-assay sample linkage available",
      "Supplement downloaded and inspected; no clinical values joined by parsing image names",
      "Paper: 47 CODEX slides/13 cases; qualified release: 48 images/14 participants. Author repository tree inspected; no CODEX sample map located.")
field(relative(SOURCE / "metadata_recheck_0213.json"), "identity recheck", "exact_supplement_sample_matches;available_section_number;available_adjacent_specimens",
      "sample_metadata", "metadata_notes", "Exact HTAN identifier matches against supplementary Sample_ID",
      "No exact supplementary sample match or section/adjacency link; retain registered specimens and explicit participant links",
      "48 registered specimen rows; 48 source-image links; 48 participant-linked rows across 14 participants. Common biological sampling remains unresolved.")
field(relative(S1), "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
      "source_channel_index;output_channel_index;marker_id;role;keep_channel", "Study + exact native OME Channel Name",
      "Use panel_from_s1 with exact study-specific source names; retain selected DAPI and body channels")
field(relative(CACHE / "source_inspection.json"), "included", "OME Pixels;Channel;proof", "image_metadata",
      "source_image_id;source_files;source_format;source_axes;source_dtype;source_revision;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um",
      "qualified source_id=filename", "Full native acquisition box; OME SizeZ=SizeT=1; CYX pages; HTAN_Release_4.0")
field(relative(CACHE / "source_inspection.json"), "included", "height;width;names", "image_metadata",
      "study;image_id;panel_id;image_kind;roi_id;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;source_channel_count;channel_count;input_image_path;intensity_description;split_method;split_evidence;metadata_notes",
      "source_id", "Independent source ROI; image_id is full native filename stem; centered zero padding; source uint8 retained without normalization")
field(relative(SOURCE / "portal_images.json"), "ImagingLevel2", "Comment;FOVnumber;FOVX;FOVY", "image_metadata",
      "region_annotation;fov_id", "Filename=qualified source filename",
      "All 48 source Comment and FOV fields are null; leave optional region annotation/FOV identity empty",
      "All 48 portal data-file/specimen/participant identities match the source manifest; PassedQC=yes for all images. OME provides no additional sample/plate annotations.")
fields = pd.DataFrame(mapping_rows, columns=pd.read_csv(TEMPLATES / "field_mapping.csv").columns)
progress("source_inspection_complete", len(images), sample_rows=len(samples),
         sample_linked_images=int(images.sample_id.notna().sum()),
         patient_linked_samples=int(samples.patient_key.notna().sum()),
         unique_patients=int(samples.patient_key.nunique()), qualified_source_images=len(sources),
         estimated_uncompressed_output_bytes=read_json(CACHE / "source_inspection.json")["estimated_uncompressed_output_bytes"])

# Reserve sufficient space for the largest square image and one temporary source.
if args.action == "export":
    TMP.mkdir(parents=True, exist_ok=True)
    for entry in sources:
        image_id = Path(entry["filename"]).name.removesuffix(".ome.tiff")
        if args.image_id and image_id != args.image_id:
            continue
        source = ROOT / entry["relative_path"]
        info = headers[entry["source_id"]]
        panel = panels[panels.panel_id.eq(entry["panel_id"])]
        destination = OUTPUT / "images" / image_id / "input.tiff"
        receipt_path = CACHE / "exports" / (image_id + ".json")
        published_matches = False
        panel_identity = panel[["source_channel_index", "source_name", "marker_id", "role", "nuclear", "body", "keep_channel", "output_channel_index"]].to_json(orient="records")
        if receipt_path.exists() and read_json(receipt_path).get("excluded"):
            receipt = read_json(receipt_path)
            assert receipt["source_size"] == source.stat().st_size and receipt["source_mtime_ns"] == source.stat().st_mtime_ns
            assert receipt["panel_identity"] == panel_identity
            continue
        if receipt_path.exists() and destination.exists():
            receipt = read_json(receipt_path)
            assert receipt["source_size"] == source.stat().st_size and receipt["source_mtime_ns"] == source.stat().st_mtime_ns
            assert receipt["panel_identity"] == panel_identity
            assert receipt["output_size"] == destination.stat().st_size and receipt["output_mtime_ns"] == destination.stat().st_mtime_ns
            published_matches = True
            if image_id in set(images.image_id):
                continue
        progress("exporting", len(images), current_image=image_id, local_checkpoint="", local_output="")
        raw_output = max(info["height"], info["width"])**2 * int(panel.keep_channel.sum()) * np.dtype(info["dtype"]).itemsize
        if shutil.disk_usage(ROOT).free < 100*1024**3 + raw_output or shutil.disk_usage(TMP).free < 64*1024**3 + source.stat().st_size:
            raise WaitingIO(f"Insufficient reserved space for {image_id}: output {raw_output} bytes")
        temporary_source = TMP / source.name
        staged = TMP / "staged_sources" / source.name
        staged_binding = staged.with_suffix(staged.suffix + ".json")
        if staged.exists() and staged_binding.exists() and read_json(staged_binding) == dict(source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns):
            assert staged.stat().st_size == source.stat().st_size
            temporary_source = staged
        else:
            with formal_read_lease(source):
                shutil.copyfile(source, temporary_source)
        # The persistent local result is closed and self-checked before NAS admission.
        computed_directory = TMP / "computed" / image_id
        local_output = computed_directory / "input.tiff"
        computed_receipt = computed_directory / "computed.json"
        identity = dict(source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                        panel_identity=panel_identity)
        cached = read_json(computed_receipt) if computed_receipt.exists() else {}
        if (cached.get("identity") == identity and local_output.exists()
                and cached.get("local_size") == local_output.stat().st_size
                and cached.get("local_mtime_ns") == local_output.stat().st_mtime_ns):
            exported, pixel_checks = cached["exported"], cached["pixel_self_checks"]
        else:
            if shutil.disk_usage(TMP).free < 64*1024**3 + raw_output:
                raise WaitingIO(f"Local output staging would consume the 64 GiB reserve: {image_id}")
            overview = CACHE / "overviews" / (entry["panel_id"].split("#")[-1] + ".png")
            selected = panel[panel.marker_id.isin(["DAPI", "PanCK", "CD3e"]) & panel.keep_channel]
            with tifffile.TiffFile(temporary_source) as tif:
                reader = PlaneReader(tif, tuple(info["pixel_size"]), selected.source_channel_index.tolist() if not overview.exists() else [])
                with patch.object(tifffile, "TiffWriter", NativeInputWriter):
                    exported = export_input(reader, panel, local_output)
                if not overview.exists():
                    save_overview(reader, selected, image_id, overview)
            del reader
            pixel_checks = verify_pixels(temporary_source, local_output, panel, info) if exported["admitted"] else []
            if exported["admitted"]:
                write_json(computed_receipt, dict(identity=identity, exported=exported,
                    pixel_self_checks=pixel_checks, local_size=local_output.stat().st_size,
                    local_mtime_ns=local_output.stat().st_mtime_ns))
        if not exported["admitted"]:
            write_json(receipt_path, dict(image_id=image_id, excluded="all admitted markers zero",
                       source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
                       panel_identity=panel_identity, **exported))
            temporary_source.unlink()
            if temporary_source == staged:
                staged_binding.unlink()
            continue
        progress("computed_waiting_publish", len(images), current_image=image_id,
                 local_checkpoint=str(computed_receipt), local_output=str(local_output))
        if not published_matches:
            with image_publication(destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name("input.tiff.partial")
                shutil.copyfile(local_output, partial)
                partial.replace(destination)
        # Formal rereads account for a shared read slot; local self-checks already cover pixels.
        output_info = header(destination)
        assert output_info["height"] == output_info["width"] == exported["height_px"]
        assert output_info["dtype"] == info["dtype"]
        assert output_info["names"] == panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
        record = dict(study=STUDY, image_id=image_id, sample_id=entry["portal_parent_biospecimen_id"], panel_id=entry["panel_id"],
            image_kind="roi", source_image_id=entry["source_id"], source_files=[entry["relative_path"]], source_format="tiff",
            source_axes="CYX", source_dtype=info["dtype"], source_revision="HTAN_Release_4.0", roi_id=image_id,
            source_x0_px=0, source_y0_px=0, source_x1_px=info["width"], source_y1_px=info["height"],
            pixel_size_x_um=info["pixel_size"][0], pixel_size_y_um=info["pixel_size"][1],
            input_image_path=relative(destination), intensity_description="Native published Level 2 uint8 intensities; no additional normalization",
            split_method="already_independent", split_evidence=relative(META / "source_manifest.json"),
            metadata_notes="Full independent source frame; HTAN biospecimen identity confirmed; shared sampling hierarchy unresolved")
        record.update({k: exported[k] for k in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]})
        images = pd.concat([images[~images.image_id.eq(image_id)], pd.DataFrame([record])], ignore_index=True)
        write_json(receipt_path, dict(image_id=image_id, source_size=source.stat().st_size,
                   panel_identity=panel_identity,
                   lossless_compression="zstd level 1",
                   source_mtime_ns=source.stat().st_mtime_ns, output_size=destination.stat().st_size,
                   output_mtime_ns=destination.stat().st_mtime_ns, pixel_self_checks=pixel_checks, **exported))
        publish_tables(images, samples, panels, fields)
        STATE["sample_linked_images"] = int(images.sample_id.notna().sum())
        temporary_source.unlink()
        if temporary_source == staged:
            staged_binding.unlink()
        local_output.unlink()
        computed_receipt.unlink()
        progress("exporting", len(images), last_completed_image=image_id)

# Preparing and independently reviewing are distinct identities. D01 remains
# unresolved until the controller assigns an independent reviewer and binding.
publish_tables(images, samples, panels, fields)
review = []
for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples():
    detail = {"preparer_self_checks": True, "independent_review_pending": True,
              "exported_images": len(images), "expected_images": 48,
              "sample_rows": len(samples), "sample_linked_images": int(images.sample_id.notna().sum()),
              "patient_linked_samples": int(samples.patient_key.notna().sum()), "unique_patients": samples.patient_key.nunique()}
    if check.check_id == "M01":
        detail["unresolved"] = "Common biological sampling across image-specific HTAN specimens; no authoritative grouping map. Paper 47/13 versus qualified release 48/14 retained explicitly."
    if check.check_id == "I05":
        detail["self_check_scope"] = "Every exported image, all retained channels, 32x32 center and edge blocks; dtype/OME names/header and padding. See per-image receipts."
    review.append(dict(study=STUDY, preparer="codex_Iglesia2024_prepare_0213", reviewer="", scope=check.scope,
        record_id=STUDY, check_id=check.check_id, result="unresolved", source_evidence=relative(CACHE / ("metadata_search_evidence.json" if check.check_id in {"M01", "M02"} else "source_inspection.json")),
        output_evidence=relative(OUTPUT / "images.parquet"), details=json.dumps(detail, ensure_ascii=False)))
with bounded_io("Waiting for review metadata publication lease"):
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        review_path = OUTPUT / "review_results.csv"
        existing = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.DataFrame()
        # Preserve separately authored independent review history.
        independent = existing[existing.reviewer.ne("")] if len(existing) else existing
        updated = pd.concat([pd.DataFrame(review), independent], ignore_index=True)
        temporary_review = review_path.with_suffix(".csv.partial")
        updated.to_csv(temporary_review, index=False, encoding="utf-8-sig")
        temporary_review.replace(review_path)
progress("prepared" if len(images) == 48 else "partial", len(images), sample_rows=len(samples),
         sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=int(samples.patient_key.notna().sum()),
         unique_patients=int(samples.patient_key.nunique()), controller_requests=[
             "Arrange independent review and D01 binding after closure",
             "Resolve common biological sampling across registered imaging biospecimens and release 48/14 versus paper 47/13",
             "Migrate single-channel caching of short-strip WSI TIFFs to shared readers when coordinating shared code",
             "Expose lossless codec selection in shared export_input; native block benchmark supports Zstd1 for this 312-GiB WSI scope"])
