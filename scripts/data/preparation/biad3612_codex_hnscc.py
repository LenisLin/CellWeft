"""Prepare S-BIAD3612 native core intensities and source-linked metadata.

Inputs: complete official core OME-TIFFs, core_metadata.tsv, source README,
77-channel native labels, and current study-specific S1 annotations.
Outputs: square native input.tiff and five study metadata/review tables.
Source QC selects the published 31-marker subset; S1 controls admission and
additional segmentation channels. Patient identities remain unassigned.
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
import time
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import imagecodecs
from PIL import Image
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.inputs import PANEL_INPUT_FIELDS, file_revision
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "biad3612_codex_hnscc"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
RAW = NAS / "raw/public" / STUDY / "S-BIAD3612"
OUT = NAS / "processed" / STUDY
CACHE = Path(__file__).resolve().parents[3] / ".cache/cellweft/runtime/study_input" / STUDY
LOCAL = Path("/tmp/cellweft-study-input") / STUDY
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL = "biad3612_native77_qc31"


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def strong_version(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def progress(status: str, **details: object) -> None:
    value = dict(study=STUDY, phase="prepare", scientific_repair_round=0,
                 status=status, updated_at=datetime.now(timezone.utc).isoformat(), **details)
    write_json(CACHE / "progress.json", value)
    write_json(args.job / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


class WaitingIO(RuntimeError):
    """A shared admission wait exceeded this invocation's bounded wait."""


def check_wait(signum: int, frame: object) -> None:
    # Only interrupt shared admission before a lease is held. Computation,
    # copying, fsync, and lease release always finish normally.
    while frame is not None:
        if (frame.f_code.co_name == "_slots" and
                frame.f_globals.get("__name__") == "cellweft.data.preparation.io_coordination" and
                "entered" not in frame.f_locals and
                time.monotonic() - frame.f_locals.get("started", time.monotonic()) > args.wait_seconds):
            raise WaitingIO(f"Shared NAS admission unavailable for {args.wait_seconds}s")
        frame = frame.f_back


def field(source: Path, source_field: str, table: str, target: str, rule: str,
          notes: str = "", join: str = "core_id") -> dict:
    row = dict.fromkeys(columns("field_mapping"), "")
    row.update(study=STUDY, source_file=relative(source), source_sheet_or_table=source.name,
               source_field=source_field, target_table=table, target_field=target,
               join_keys=join, transformation_rule=rule, notes=notes)
    return row


def reviews(images: pd.DataFrame, samples: pd.DataFrame) -> None:
    rows = []
    for check, result, details in [
        ("S01", "unresolved", "28 source-retained cores; only complete and decodable local files processed. See inventory.json, image_results and source_decode_failure_*.json for unqualified versions. Full TMA overlaps these cores; 16 source-QC-excluded cores not added."),
        ("P01", "unresolved", "Preparer: native77 order and exact S1 mapping; independent review pending."),
        ("P02", "unresolved", "Source README explicitly restricts high-quality downstream measurements to marker_panel.tsv; retain extra S1 segmentation measurements only for segmentation. Independent review pending."),
        ("I01", "unresolved", "Preparer image/source identities recorded per closed image; independent review pending."),
        ("I02", "unresolved", "Exact official image_file/core_id specimen link; independent foreign-key review pending."),
        ("I03", "unresolved", "Full native core frame and centered square padding; independent geometry review pending."),
        ("I04", "unresolved", "Author supplies individual physical core images and core-level site/HPV annotation; independent region review pending."),
        ("I05", "unresolved", "Preparer pixel blocks and padding recorded in preparer_pixel_checks.json when available; independent pixel review pending."),
        ("P03", "unresolved", "Shared exporter reopens every published TIFF header; independent channel/name/order review pending."),
        ("M01", "unresolved", f"{len(samples)} source-confirmed core specimens; {len(images)} exported images linked to samples; 0 patient-linked samples. Official core metadata lacks patient keys; original cohort-level 44 independent patients does not establish retained-core patient mapping. Title search found DOI 10.1038/s41588-026-02723-7 and DSimkin/Spatial.HNSCC. Supplementary Table1 has clinical patient IDs without CODEX core keys; Table3 locations repeat across two TMAs and four slides. No exact CODEX-to-patient map established; see source_search.json."),
        ("M02", "unresolved", "Only source-supported sample/core/site/HPV and cohort protocol fields filled. Independent clinical-value reconstruction pending."),
        ("D01", "unresolved", f"Preparer only: {len(images)}/28 native images published. Independent review and patient linkage pending; no input_binding pass claimed."),
    ]:
        row = dict.fromkeys(columns("review_results"), "")
        row.update(study=STUDY, preparer=args.job.name, scope="study", record_id=STUDY,
                   check_id=check, result=result, source_evidence=relative(CACHE),
                   output_evidence=relative(OUT), details=details)
        rows.append(row)
    with formal_publish_lease(OUT / "review_results.csv", directory=OUT):
        path = OUT / "review_results.csv"
        # Preserve any independently authored records on a resumed invocation.
        old = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old.reviewer != ""]
        frame = pd.concat([pd.DataFrame(rows), old], ignore_index=True)
        temporary = path.with_suffix(".csv.partial")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


def overview_and_pixels(local: Path, destination: Path, row: dict, panel: pd.DataFrame) -> None:
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    h, w = row["source_y1_px"], row["source_x1_px"]
    l, t = row["pad_left_px"], row["pad_top_px"]
    evidence = []
    with formal_read_lease(destination):
        with tifffile.TiffFile(local) as source, tifffile.TiffFile(destination) as output:
            for x, y in [(w // 2, h // 2), (0, 0)]:
                for channel in kept.itertuples():
                    a = read_tiff_window(source, channel.source_channel_index, (x, y, x+32, y+32))
                    b = read_tiff_window(output, int(channel.output_channel_index), (x+l, y+t, x+l+32, y+t+32))
                    assert a.dtype == b.dtype and np.array_equal(a, b)
                evidence.append(dict(source_xy=[x, y], size=[32, 32], channels=len(kept), equal=True))
            if t:
                for i in range(len(kept)):
                    assert not read_tiff_window(output, i, (0, 0, 32, min(t, 32))).any()
            if l:
                for i in range(len(kept)):
                    assert not read_tiff_window(output, i, (0, 0, min(l, 32), 32)).any()
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for name, color in [("DAPI", 2), ("PanCK", 0), ("CD45", 1)]:
                index = int(panel.loc[panel.source_name == name, "source_channel_index"].iloc[0])
                values = source.pages[index].asarray(maxworkers=2)
                upper = float(np.percentile(values[values > 0], 99)) if values.any() else 1.0
                rgb[:, :, color] = np.clip(values / upper * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(rgb)
            image.thumbnail((1000, 1000))
            image.save(CACHE / f"{row['image_id']}_overview.png")
    write_json(CACHE / "preparer_pixel_checks.json", dict(image_id=row["image_id"], blocks=evidence,
               padding_checked=True, overview_channels=["DAPI", "PanCK", "CD45"],
               identity="preparer self-check; independent review pending"))


def source_decode_failure(local: Path, original: Path, panel: pd.DataFrame) -> dict:
    """Locate a failed LZW tile and verify the same encoded bytes on NAS.

    A source with a reproducibly undecodable retained tile is unqualified for
    input export. Preserve its original file and record its exact version.
    """
    failure = None
    with tifffile.TiffFile(local) as tif:
        for channel in panel[panel.keep_channel].itertuples():
            page = tif.pages[channel.source_channel_index].aspage()
            for i, (offset, count) in enumerate(zip(page.dataoffsets, page.databytecounts)):
                tif.filehandle.seek(offset)
                encoded = tif.filehandle.read(count)
                try:
                    page.decode(encoded, i, jpegtables=page.jpegtables)
                except imagecodecs.LzwError as error:
                    failure = dict(image_id=original.name.removesuffix(".ome.tif"),
                                   source_channel_index=int(channel.source_channel_index), source_name=channel.source_name,
                                   tile_index=i, offset=offset, byte_count=count, decode_error=repr(error),
                                   source_revision=file_revision(original), local_revision=file_revision(local))
                    break
            if failure:
                break
    assert failure is not None, "Failed tile was not reproducible; source diagnosis required"
    with formal_read_lease(original):
        before = strong_version(original)
        with original.open("rb") as source:
            source.seek(failure["offset"])
            original_encoded = source.read(failure["byte_count"])
        assert before == strong_version(original)
        assert original_encoded == encoded, "Source/staged encoded bytes differ; do not exclude source"
        try:
            imagecodecs.lzw_decode(original_encoded)
        except imagecodecs.LzwError as error:
            failure["nas_direct_decode_error"] = repr(error)
        else:
            raise AssertionError("Original NAS tile unexpectedly decodes")
    failure.update(qualified=False, nas_encoded_tile_equal_local=True, source_strong_version=before,
                   action="Preserve raw source and local staging. Original download owner must repair/revalidate this source version.")
    write_json(CACHE / f"source_decode_failure_{failure['image_id']}.json", failure)
    return failure


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", type=Path, required=True)
parser.add_argument("--wait-seconds", type=int, default=90)
parser.add_argument("--metadata-only", action="store_true", help="Refresh source tracing and tables without starting another image")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
LOCAL.mkdir(parents=True, exist_ok=True)
signal.signal(signal.SIGALRM, check_wait)
signal.setitimer(signal.ITIMER_REAL, 1, 1)

# Resolve the source QC panel without substituting its order for image metadata.
manifest = json.loads((META / "source_manifest.json").read_text())
names = [item["Name"] for item in json.loads((META / "ordered_markers.json").read_text())["native_order"]]
quality = pd.read_csv(META / "source_analysis_marker_panel.tsv", sep="\t", dtype=str).analysis_column_name.tolist()
assert "Only the 31 high-quality markers" in (META / "source_metadata/README.txt").read_text()
panel = panel_from_s1(STUDY, PANEL, names, S1,
                      downstream_source_indices={i for i, name in enumerate(names) if name in quality})
for i, channel in panel.iterrows():
    if channel.source_name not in quality:
        if channel.keep_channel:
            panel.loc[i, "keep_reason"] = "Source QC excludes downstream use; retain S1-selected segmentation input"
        else:
            panel.loc[i, "exclusion_reason"] = "Not in source README high-quality downstream panel; no S1 segmentation use"
    elif channel.keep_channel and channel.role == "downstream":
        panel.loc[i, "keep_reason"] = "Source QC high-quality marker and S1 admitted"

# Each published physical core supplies a specimen ID; no patient ID is inferred.
cores = pd.read_csv(META / "source_metadata/core_metadata.tsv", sep="\t", dtype=str, keep_default_na=False)
assert len(cores) == 28 and cores.core_id.is_unique
rds_probe = META / "source_metadata/author_metadata_semantic_probe.json"
rds_rule = "Downloaded small author RDS; gzip/exact retained-core identifier byte probe found no Core_ IDs. Semantic RDS decoding unavailable because pyreadr/rdata/Rscript are absent; no patient linkage inferred. See author_metadata_probe.json."
if rds_probe.exists():
    decoded = json.loads(rds_probe.read_text())
    for name, key in (("samples_metadata.rds", "source_revision"), ("core_metadata.tsv", "core_metadata_revision")):
        # Relocation changes the pathname; the admitted source retains its size and timestamp.
        actual = file_revision(META / "source_metadata" / name)
        assert all(actual[k] == decoded[key][k] for k in ("size", "mtime_ns")), "RDS semantic evidence source changed"
    assert not decoded["core_to_patient_bridge"] and not any(decoded["exact_retained_core_matches_by_column"].values())
    assert decoded["core_matches_in_index"] == 0
    rds_rule = (f"Author RDS semantically decoded with {decoded['reader']} {decoded['reader_version']}: "
                f"{decoded['rows']} rows/{len(decoded['columns'])} columns; no retained core IDs in any column or index. "
                "No core-to-patient bridge established. See author_metadata_semantic_probe.json.")
samples = pd.DataFrame([{**dict.fromkeys(columns("sample_metadata"), ""),
    "study": STUDY, "sample_id": r.core_id, "source_sample_id": r.core_id,
    "sample_type": "tissue", "tissue_type": "tumor", "anatomical_site": r.tumor_site,
    "preservation_method": "ffpe", "primary_diagnosis": "laryngeal squamous cell carcinoma",
    "histology": "conventional keratinizing squamous cell carcinoma", "hpv_status_source": r.HPV_status,
    "metadata_notes": "Source-confirmed physical TMA core specimen; original cohort 44 independent patients; retained-core patient linkage unavailable. Paper supplementary Table1 clinical IDs lack CODEX core keys; Table3 repeated locations lack exact CODEX slide binding. Clinical and survival values remain unjoined."}
    for r in cores.itertuples()])
mapping = pd.DataFrame([
    field(META / "source_metadata/core_metadata.tsv", "core_id;image_file", "image_metadata", "sample_id", "Exact image_file and core_id join to one physical core specimen; retain source core_id as sample_id"),
    field(META / "source_metadata/core_metadata.tsv", "core_id", "sample_metadata", "sample_id", "Preserve exact physical core specimen ID; no patient inference"),
    field(META / "source_metadata/core_metadata.tsv", "core_id", "sample_metadata", "source_sample_id", "Preserve source core ID; patient_key remains empty"),
    field(META / "source_metadata/core_metadata.tsv", "tumor_site", "sample_metadata", "anatomical_site", "Preserve source site"),
    field(META / "source_metadata/core_metadata.tsv", "HPV_status", "sample_metadata", "hpv_status_source", "Preserve source HPV classification", json.dumps(dict(data_type="string", definition="Source core HPV status"))),
    *[field(META / "source_metadata/biostudies.json", "Specimen.Sample preparation protocol;Study.Description", "sample_metadata", target, rule, join="study")
      for target, rule in [
          ("sample_type", "Physical FFPE tissue core; sample_type=tissue"),
          ("tissue_type", "Source tumor cores; controlled tissue_type=tumor"),
          ("preservation_method", "Source FFPE mapped to controlled preservation_method=ffpe"),
          ("primary_diagnosis", "Cohort explicitly laryngeal squamous cell carcinoma"),
          ("histology", "Specimen protocol explicitly conventional keratinizing laryngeal squamous cell carcinoma"),
      ]],
    field(META / "source_metadata/biostudies.json", "Specimen;Biosample;Publication", "sample_metadata", "metadata_notes", "Cohort protocol supplies FFPE, laryngeal SCC and conventional keratinizing histology. Patient keys and linked publication absent; no inferred patient linkage.", join="study"),
    field(META / "source_metadata/README.txt", "Marker quality-control note", "panel_mapping", "keep_channel", "Use marker_panel.tsv for downstream QC membership, native image XML for all77 source positions; S1 for marker admission and segmentation. No v2 alternative appears in the source QC subset.", join="source_name;analysis_column_name"),
    field(S1, "Study;Original marker", "panel_mapping", "marker_id;mapping_status;role;nuclear;body", "Exact study/source label mapping; source QC and S1 admission intersect for downstream measurements", join="Study;Original marker"),
    field(META / "source_metadata/supplementary_tables_1_10.xlsx", "Table1 Patient ID;Table3 Location,ID,TMA,Slide", "sample_metadata", "patient_key", "Leave unassigned: Table1 lacks CODEX core IDs; Table3 has repeated locations across two TMAs and four slides without an exact binding to this independent CODEX cohort.", join="unresolved"),
    field(META / "source_metadata/CODEX_TMA_Final.R", "core_id;sample metadata references", "sample_metadata", "metadata_notes", "Author analysis preserves core IDs; transcriptomic sample metadata references do not establish this CODEX-to-patient identity. No coordinate-only or tissue-only join.", join="unresolved"),
    field(META / "source_metadata/samples_metadata.rds", "author sample metadata", "sample_metadata", "patient_key", rds_rule, join="unresolved"),
])
images = pd.read_parquet(OUT / "images.parquet") if (OUT / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
if len(images):
    previous_panel = pd.read_csv(OUT / "panel_mapping.csv")
    identity_fields = [key for key in PANEL_INPUT_FIELDS if key in panel]
    previous_identity = previous_panel[identity_fields].astype(object).where(previous_panel[identity_fields].notna(), "")
    current_identity = panel[identity_fields].astype(object).where(panel[identity_fields].notna(), "")
    assert json.loads(previous_identity.to_json(orient="records")) == json.loads(current_identity.to_json(orient="records")), "Existing images require targeted review before changing the scientific panel"
inventory = []
for item in manifest["sources"]:
    if not item["name"].endswith(".ome.tif"):
        continue
    path = RAW / item["name"]
    available = path.is_file() and path.stat().st_size == item["size"]
    revision = file_revision(path) if available else None
    bad_path = CACHE / f"source_decode_failure_{path.name.removesuffix('.ome.tif')}.json"
    known_bad = json.loads(bad_path.read_text()) if bad_path.exists() else None
    # Persisted device numbers are mount-local; retain inode and all file times.
    qualified = available and not (known_bad and known_bad["source_revision"] == revision
                                  and known_bad.get("source_strong_version") is not None
                                  and known_bad["source_strong_version"][1:] == strong_version(path)[1:])
    inventory.append(dict(**item, complete_local=available, qualified_local=qualified,
                          revision=revision, decode_failure=relative(bad_path) if known_bad else None))
write_json(CACHE / "inventory.json", inventory)
write_json(CACHE / "source_search.json", dict(
    study_url=manifest["source_url"], api_url="https://www.ebi.ac.uk/biostudies/api/v1/studies/S-BIAD3612",
    successful_small_sources=["https://www.ebi.ac.uk/biostudies/files/S-BIAD3612/metadata/"+n for n in ["core_metadata.tsv", "README.txt"]],
    failed_endpoint="ftp.ebi.ac.uk HTTPS metadata requests timed out; BioStudies files endpoint succeeded",
    paper_doi="10.1038/s41588-026-02723-7", paper_url="https://www.nature.com/articles/s41588-026-02723-7",
    supplement_url="https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41588-026-02723-7/MediaObjects/41588_2026_2723_MOESM4_ESM.xlsx",
    repository_url="https://github.com/DSimkin/Spatial.HNSCC", author_aux_url="https://api.figshare.com/v2/articles/27967107",
    examined_sources=["biostudies.json", "core_metadata.tsv", "README.txt", "paper.html", "supplementary_tables_1_10.xlsx", "CODEX_TMA_Final.R", "TMA_Analysis.R", "get_supp_files.R", "figshare_aux_metadata.json", "samples_metadata.rds", "author_metadata_probe.json"] + ([rds_probe.name] if rds_probe.exists() else []),
    source_search_limitations="GitHub tree API returned403/rate limit; named author source files retrieved via raw URLs. " + rds_rule,
    patient_linkage="No patient IDs in core_metadata. Publication subsection empty, but exact-title public search found paper and repository. Supplementary Table1 clinical IDs lack CODEX core keys; Table3 has440 rows with8 occurrences of A2 across2TMAs/4slides, without exact CODEX slide binding. Author CODEX analysis and aux inventory do not establish patient keys. Clinical values remain unjoined.",
    sample_rows=len(samples), patient_linked_samples=0))
progress("prepared_metadata", complete_local=sum(i["complete_local"] for i in inventory),
         qualified_local=sum(i["qualified_local"] for i in inventory),
         expected_images=28, exported_images=len(images), sample_rows=len(samples), patient_linked_samples=0)

current = None
try:
    save_input_tables(OUT, images, samples, panel, mapping)
    reviews(images, samples)
    for item in ([] if args.metadata_only else inventory):
        if not item["qualified_local"]:
            continue
        original = RAW / item["name"]
        current = original.name.removesuffix(".ome.tif")
        assert cores.loc[cores.core_id == current, "image_file"].tolist() == [item["name"]]
        result_path = CACHE / "image_results" / f"{current}.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            destination = NAS / previous["row"]["input_image_path"]
            if previous["source_revision"] == file_revision(original) and destination.is_file() and previous["output_revision"] == file_revision(destination):
                if current not in set(images.image_id):
                    images = pd.concat([images, pd.DataFrame([previous["row"]])], ignore_index=True)
                    save_input_tables(OUT, images, samples, panel, mapping)
                continue
            raise ValueError(f"{current}: existing image proof changed; targeted review required")
        local = LOCAL / original.name
        revision = file_revision(original)
        if shutil.disk_usage(LOCAL).free < 64 * 1024**3 + item["size"]:
            raise WaitingIO("Insufficient free space after 64GiB local reserve for source staging")
        progress("staging_source", image_id=current, source_revision=revision, exported_images=len(images), source_staging_bytes=item["size"])
        with formal_read_lease(original):
            if not local.is_file() or file_revision(local).get("size") != revision["size"] or local.stat().st_mtime_ns != original.stat().st_mtime_ns:
                partial = local.with_suffix(local.suffix + ".partial")
                shutil.copy2(original, partial)
                partial.replace(local)
            assert file_revision(original) == revision
        with tifffile.TiffFile(local) as tif:
            pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
            assert [c.attrib["Name"] for c in pixels.findall("{*}Channel")] == names
            h, w = int(pixels.attrib["SizeY"]), int(pixels.attrib["SizeX"])
            assert int(pixels.attrib["SizeC"]) == 77 and int(pixels.attrib["SizeZ"]) == int(pixels.attrib["SizeT"]) == 1
            assert len(tif.pages) == 77 and tif.pages[0].dtype == np.dtype("uint16")
            assert all(max(o+b for o,b in zip(p.dataoffsets, p.databytecounts)) <= item["size"] for p in tif.pages)
            scale = tuple(float(pixels.attrib[k]) for k in ("PhysicalSizeX", "PhysicalSizeY"))
            assert all(pixels.attrib[k] == "µm" for k in ("PhysicalSizeXUnit", "PhysicalSizeYUnit"))
        # Deflate's encoded worst-case bound includes one extra percent plus
        # TIFF tile/OME overhead. Source staging has already been accounted for.
        tiled_side = ((max(h, w) + 511) // 512) * 512
        estimate = int(panel.keep_channel.sum()) * tiled_side**2 * 2 * 1.01 + 1024**2
        if shutil.disk_usage(LOCAL).free < 64 * 1024**3 + estimate or shutil.disk_usage(NAS).free < 100 * 1024**3 + estimate:
            raise WaitingIO("Insufficient free space after 64GiB local / 100GiB NAS reserve for exact image geometry")
        spec = SourceSpec(current, PANEL, "tiff", local, tuple(range(77)), pixel_size_um=scale,
                          source_revision=json.dumps(revision, sort_keys=True))
        destination = OUT / "images" / current / "input.tiff"
        progress("exporting_image", image_id=current, source_revision=revision, exported_images=len(images))
        try:
            with open_source(spec) as reader:
                result = export_input(reader, panel, destination)
        except imagecodecs.LzwError:
            failure = source_decode_failure(local, original, panel)
            item["qualified_local"] = False
            item["decode_failure"] = relative(CACHE / f"source_decode_failure_{current}.json")
            write_json(CACHE / "inventory.json", inventory)
            progress("source_excluded", image_id=current, exported_images=len(images), failure=failure)
            continue
        assert file_revision(original) == revision
        if not result["admitted"]:
            write_json(result_path, dict(source_revision=revision, result=result, exclusion="all admitted markers zero"))
            local.unlink()
            continue
        row = dict.fromkeys(columns("image_metadata"))
        row.update(study=STUDY, image_id=current, sample_id=current, panel_id=PANEL,
                   image_kind="tma_core", source_image_id=item["name"], source_files=[relative(original)],
                   source_format="ome-tiff", source_axes="CYX", source_dtype="uint16", source_revision=json.dumps(revision, sort_keys=True),
                   core_id=current, region_annotation="Larynx; HPV-negative; retained after source QC",
                   source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                   pixel_size_x_um=scale[0], pixel_size_y_um=scale[1], input_image_path=relative(destination),
                   intensity_description="Native decoded uint16 CODEX intensity; no rescaling or normalization",
                   split_method="already_independent", split_evidence="Source individual physical core OME-TIFF; complete native frame",
                   metadata_notes="Core specimen linked by exact official metadata image_file; patient linkage unresolved")
        for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]:
            row[key] = result[key]
        assert not result["valid_pixels"]
        write_json(result_path, dict(source_revision=revision, output_revision=file_revision(destination), result=result, row=row))
        images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
        save_input_tables(OUT, images, samples, panel, mapping)
        if not (CACHE / "preparer_pixel_checks.json").exists():
            overview_and_pixels(local, destination, row, panel)
        local.unlink()
        reviews(images, samples)
        progress("image_closed", image_id=current, exported_images=len(images), sample_linked_images=int(images.sample_id.notna().sum()), sample_rows=len(samples), patient_linked_samples=0)
    reviews(images, samples)
    progress("prepared_awaiting_independent_review" if len(images) == sum(i["qualified_local"] for i in inventory) else "partial_prepared", expected_images=28, exported_images=len(images),
             sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=0)
except (WaitingIO, OSError) as error:
    progress("waiting_io", image_id=current, exported_images=len(images), expected_images=28,
             sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=0,
             error=repr(error), notes=getattr(error, "__notes__", []),
             resume="Rerun identical command; reuse image_results and closed helper publication checkpoints; continue only incomplete images",
             local_staging=str(LOCAL), helper_checkpoints=os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch") + "/publish_checkpoints")
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
