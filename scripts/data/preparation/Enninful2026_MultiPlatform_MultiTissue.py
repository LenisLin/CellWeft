"""Deliver six locally verified DBiTplus human mxIF acquisitions as native inputs.

Inputs: author workbook, native-v2 manifest/proofs, complete QPTIFF/OME-TIFF, S1.
Outputs: independent square CYX input.tiff, 39-field index, sample/panel/mapping
tables and preparer checks. Public source attachments are cached study-locally.
The author workbook identifies biological sections; it supplies no patient map.
Native SlideID conflicts are retained without inventing a patient or diagnosis.
Run inventory/verify under shared formal read lease, and export under shared formal write lease.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Enninful2026_MultiPlatform_MultiTissue"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
META = metadata_source_directory(NAS, STUDY)
RAW_METADATA = NAS / "raw/public" / STUDY / "metadata"
CACHE = PROJECT / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
PROGRESS = PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0031_prepare_0/progress.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PREPARER = "0031_prepare_0"
GAPS = ["Patient identifiers and donor clinical tables are absent in the checked workbook/article/supplementary tables/author repository.",
        "Independent CLI review and D01 acceptance are pending."]
CONTROLLER_REQUESTS = [
    "Repair shared read_tiff_window for TiffFrame after OME series discovery; this study materializes selected IFD headers with pages.get(aspage=True, cache=True).",
    "Migrate the bounded three-native-row TIFF cache to the shared reader to avoid repeated decoding and NAS reads with non-aligned padding.",
    "Independently review six images / five biological samplings, including C3/C4 sharing S3; adjudicate the verified missing patient identifiers in M01 before D01 acceptance.",
]


def relative(path: Path) -> str:
    try:
        return str(metadata_source_relative(path, NAS))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def progress(stage: str, **extra) -> None:
    write_json(PROGRESS, dict(study=STUDY, stage=stage, expected_images=6,
                             exported_images=len(images), completed_images=len(images),
                             sample_rows=len(samples), sample_linked_images=len(images),
                             patient_linked_samples=0, gaps=GAPS,
                             controller_requests=CONTROLLER_REQUESTS,
                             updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra))


class RowCachedReader:
    """Keep only three native tile rows of the current channel in RAM (<50 MiB).

    Shared window reading redecodes neighboring tiles when centered padding is
    not tile-aligned. These 20k--50k WSI widths make that repeated NAS access
    material. Batch contiguous native tile-row reads and reuse overlap. This
    local performance adapter should migrate to the common bounded TIFF reader.
    Pixels still use tifffile's native segment decoder, with no intensity change.
    """

    def __init__(self, reader, image_id: str) -> None:
        self.reader = reader
        self.source, self.shape, self.dtype = reader.source, reader.shape, reader.dtype
        self.image_id = image_id
        self.rows = OrderedDict()
        self.channel = None
        self.last_report = 0.0

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        result = np.empty((len(channels), y1-y0, x1-x0), dtype=self.dtype)
        for output, channel in enumerate(channels):
            if channel != self.channel:
                self.rows.clear()
                self.channel = channel
                print(f"{self.image_id}: source channel {channel+1}/{self.shape[0]}", flush=True)
            page = self.reader._tiffs[0].pages[self.source.channels[channel]]
            th, tw = page.tilelength, page.tilewidth
            columns = (self.shape[2] + tw - 1) // tw
            for ty in range(y0//th, (y1-1)//th + 1):
                if ty not in self.rows:
                    row = np.zeros((th, self.shape[2]), dtype=self.dtype)
                    start, stop = ty*columns, (ty+1)*columns
                    for data, index in page.parent.filehandle.read_segments(
                            page.dataoffsets[start:stop], page.databytecounts[start:stop],
                            indices=list(range(start, stop))):
                        decoded, _, _ = page.decode(data, index, jpegtables=page.jpegtables)
                        if decoded is None:
                            raise ValueError(f"Missing native tile: channel={channel}, tile={index}")
                        tx = index % columns
                        width = min(tw, self.shape[2]-tx*tw)
                        row[:, tx*tw:tx*tw+width] = decoded[0, :, :width, 0]
                    self.rows[ty] = row
                    if len(self.rows) > 3:
                        self.rows.popitem(last=False)
                    if time.monotonic()-self.last_report > 25:
                        progress("exporting", current_image=self.image_id,
                                 source_channel_index=channel, native_y_px=ty*th)
                        self.last_report = time.monotonic()
                self.rows.move_to_end(ty)
                a, b = max(y0, ty*th), min(y1, (ty+1)*th)
                result[output, a-y0:b-y0] = self.rows[ty][a-ty*th:b-ty*th, x0:x1]
        return result

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        return self.reader.valid_window(window)


def field(source: str, sheet: str, source_field: str, table: str, target: str,
          rule: str, notes: str = "") -> dict:
    return dict(study=STUDY, source_file=source, source_sheet_or_table=sheet,
                source_field=source_field, target_table=table, target_field=target,
                join_keys="record_id + exact filename/native archive member -> source_id -> image_id/sample_id",
                value_mapping="", transformation_rule=rule, time_origin_rule="",
                conflict_rule="Retain author workbook and native header labels separately; unknown stays empty",
                notes=notes)


def save_tables() -> None:
    save_input_tables(OUTPUT, pd.DataFrame(images, columns=image_columns), samples, panels, mappings)


def headers_and_pixels(source: dict, proof: dict, image_id: str, panel: pd.DataFrame) -> list[dict]:
    """Check all output channels against 32-square native internal/edge blocks."""
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    _, h, w = proof["shape"]
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    evidence = []
    with tifffile.TiffFile(NAS / proof["native_path"]) as native, tifffile.TiffFile(OUTPUT/image_id/"input.tiff") as target:
        assert target.series[0].shape == (len(kept), side, side)
        assert target.series[0].axes == "CYX" and target.series[0].dtype == np.dtype(proof["dtype"])
        names = [node.attrib["Name"] for node in ET.fromstring(target.ome_metadata).iter() if node.tag.endswith("}Channel")]
        assert names == kept.marker_id.tolist()
        # OME series discovery caches lightweight TiffFrames. The shared window
        # reader requires samplesperpixel and jpegtables from complete TiffPages.
        # Materialize only selected IFD headers; common-reader repair is pending.
        for index in kept.source_channel_index:
            native.pages.get(int(index), aspage=True, cache=True)
        for index in kept.output_channel_index:
            target.pages.get(int(index), aspage=True, cache=True)
        candidates = [(w//2, h//2), (int(.6*w), int(.4*h)), (int(.4*w), int(.4*h)), (int(.65*w), int(.65*h))]
        internal = next((x, y, x+32, y+32) for x, y in candidates
                        if read_tiff_window(native, 0, (x, y, x+32, y+32)).any())
        blocks = [internal, (w-32, h-32, w, h), (0, 0, 32, 32)]
        for channel in kept.itertuples(index=False):
            for block in blocks:
                x0, y0, x1, y1 = block
                actual = read_tiff_window(target, int(channel.output_channel_index), (x0+left, y0+top, x1+left, y1+top))
                expected = read_tiff_window(native, int(channel.source_channel_index), block)
                assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
            for block in [(0, 0, 32, min(top, 32))] if top else [(0, 0, min(left, 32), 32)]:
                if block[2] and block[3]:
                    assert not read_tiff_window(target, int(channel.output_channel_index), block).any()
            evidence.append(dict(source_channel_index=int(channel.source_channel_index),
                                 output_channel_index=int(channel.output_channel_index),
                                 source_windows=blocks, padding_ltrb=[left, top, side-w-left, side-h-top]))
    return evidence


def make_overview(proof: dict, image_id: str) -> None:
    """Use QPTIFF pyramids, or stream only DNA tile rows for CellScape."""
    destination = CACHE / "overviews" / f"{image_id}.png"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(NAS/proof["native_path"]) as tif:
        if proof["format"] == "QPTIFF":
            level = tif.series[0].levels[-1]
            # The reduced CYX level preserves the full native acquisition frame.
            values = level.pages[0].asarray(maxworkers=2)
        else:
            page = tif.pages[0]
            h, w = page.shape
            values = np.zeros(((h+31)//32, (w+31)//32), dtype=page.dtype)
            for data, index in tif.filehandle.read_segments(page.dataoffsets, page.databytecounts):
                decoded, position, _ = page.decode(data, index)
                y, x = position[2:4]
                tile = decoded[0, :, :, 0]
                yy, xx = min(tile.shape[0], h-y), min(tile.shape[1], w-x)
                values[y//32:(y+yy+31)//32, x//32:(x+xx+31)//32] = tile[:yy:32, :xx:32]
        scale = float(np.percentile(values[values > 0], 99.5)) if values.any() else 1
        display = Image.fromarray(np.uint8(np.clip(values/scale, 0, 1)*255)).convert("RGB")
        display.thumbnail((650, 650))
        canvas = Image.new("RGB", (680, 710), "black")
        canvas.paste(display, (15, 35))
        ImageDraw.Draw(canvas).text((15, 10), f"{image_id} full frame; nuclear overview only", fill="white")
        canvas.save(destination)


# Resolve complete source objects using the authoritative native-v2 order.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["inventory", "export", "verify"], default="export")
parser.add_argument("--images", nargs="*", help="Optional exact C1..C6 subset; merge existing index")
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((META/"source_manifest.v2.json").read_text())
sources = manifest["sources"]
image_columns = pd.read_csv(TEMPLATES/"image_metadata.csv").columns.tolist()
sample_columns = pd.read_csv(TEMPLATES/"sample_metadata.csv").columns.tolist()
images = pd.read_parquet(OUTPUT/"images.parquet").to_dict("records") if (OUTPUT/"images.parquet").exists() else []
sample_rows, panel_rows, inventory = [], [], []
proofs = {}
for index, source in enumerate(sources, 1):
    image_id, source_id = f"C{index}", source["source_id"]
    proof_path = META/("proofs_v2" if index in (3, 6) else "proofs")/(source_id+".json")
    proof = json.loads(proof_path.read_text())
    path = NAS/proof["native_path"]
    stat = path.stat()
    assert proof["status"] == "verified" and proof["native_order_verified"]
    assert proof["file_stat"] == [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    assert proof["ordered_labels"] == source["native_channel_order"]
    assert proof["shape"] == source["shape"]
    with tifffile.TiffFile(path) as tif:
        pages = [tif.pages[i] for i in range(len(proof["ordered_labels"]))]
        assert all(list(p.shape) == proof["shape"][1:] and p.dtype == np.dtype(proof["dtype"]) for p in pages)
        if proof["format"] == "QPTIFF":
            labels = [ET.fromstring(p.description).findtext("Biomarker") for p in pages]
            assert labels == proof["ordered_labels"]
            slide_id = ET.fromstring(pages[0].description).findtext("SlideID")
            resolution = [float(pages[0].tags[k].value[0])/float(pages[0].tags[k].value[1]) for k in ("XResolution", "YResolution")]
            unit = int(pages[0].tags["ResolutionUnit"].value)
            pixel_size = [10000/r for r in resolution] if unit == 3 else ([25400/r for r in resolution] if unit == 2 else [None, None])
        else:
            xml = ET.fromstring(tif.ome_metadata)
            pixel_node = next(n for n in xml.iter() if n.tag.endswith("}Pixels"))
            labels = [n.attrib["Name"] for n in xml.iter() if n.tag.endswith("}Channel")]
            assert labels == proof["ordered_labels"]
            slide_id = ""
            pixel_size = [float(pixel_node.attrib[k]) for k in ("PhysicalSizeX", "PhysicalSizeY")]
    archive = NAS/"raw/public"/STUDY/f"zenodo_{source['record_id']}"/source["filename"]
    members = []
    if source["member"]:
        with zipfile.ZipFile(archive) as zipped:
            members = [{"name": entry.filename, "bytes": entry.file_size} for entry in zipped.infolist()]
            assert zipped.getinfo(source["member"]["name"]).file_size == path.stat().st_size
    panel = panel_from_s1(STUDY, source["panel_id"], proof["ordered_labels"], S1)
    panel_rows.append(panel)
    sample = dict.fromkeys(sample_columns, "")
    sample.update(study=STUDY, sample_id=f"S{index}", source_sample_id=source["workbook_sample"],
                  sample_type="Paraffin tissue section", preservation_method="ffpe",
                  anatomical_site="Cerebellum" if index == 2 else ("Lymph node" if index in (1, 3, 4) else ""),
                  primary_diagnosis="CLL with transformation to DLBCL" if index == 5 else ("Marginal zone lymphoma" if index == 6 else ""),
                  metadata_notes="Author workbook Metadata filename->sample identifies this source biological section. "
                  "No patient ID or cross-section donor/block map available; different section labels are not assumed to imply different patients. "
                  + " ".join(source["identity_caveats"]))
    if index == 3:
        sample["source_sample_id"] = slide_id.split("-01", 1)[0]
        sample["metadata_notes"] += (
            " C3 native specimen code matches the author repository biological LYMPH_NODE directory used for "
            "CODEXfeature_labeled_aligned_50um.csv; workbook maps that table to the C4 section. "
            "Supplementary Table 3 independently labels the 49-marker control and 35-marker 50um section as Lymph node 2. "
            "C3 comparison and C4 DBiTplus sections share this biological sampling (S3). "
            "C3 overview has three tissue pieces, all retained under the single source specimen label.")
    if index != 4:
        sample_rows.append(sample)
    proofs[image_id] = proof
    inventory.append(dict(image_id=image_id, source_id=source_id, proof=relative(proof_path),
                          native_path=proof["native_path"], archive_path=relative(archive),
                          archive_members=members, shape=proof["shape"], dtype=proof["dtype"],
                          slide_id=slide_id, pixel_size_um=pixel_size, source_sample_id=source["workbook_sample"],
                          retained_channels=int(panel.keep_channel.sum()), identity_caveats=source["identity_caveats"]))
# GEO sample accessions identify sequencing sections, not patients. Exact titles
# connect the 50um, MZL and CLL sections; preserve accession evidence in notes.
geo_path = RAW_METADATA/"GSE308167_family.soft"
if geo_path.exists():
    accession, geo_samples = "", {}
    for line in geo_path.read_text().splitlines():
        if line.startswith("^SAMPLE = "):
            accession = line.split(" = ", 1)[1]
        if line.startswith("!Sample_title = "):
            geo_samples[line.split(" = ", 1)[1]] = accession
    for index in (4, 5, 6):
        label = sources[index-1]["workbook_sample"]
        if label in geo_samples:
            sample_id = "S3" if index == 4 else f"S{index}"
            sample = next(row for row in sample_rows if row["sample_id"] == sample_id)
            sample["anatomical_site"] = "Lymph node"
            sample["metadata_notes"] += f" Exact source section title joins GEO {geo_samples[label]}; GEO reports lymph node tissue, without donor identity."
samples = pd.DataFrame(sample_rows, columns=sample_columns)
panels = pd.concat(panel_rows, ignore_index=True)

# Keep source keys, all target field semantics and optional clinical gaps explicit.
mapping_rows = [field(relative(RAW_METADATA/"Multiplexed_IF_Metadata.xlsx"), "Metadata", "File Name + Sample",
                      "sample_metadata", key,
                      "Match exact workbook filename; documented Lymphnode_ prefix alias for C1 and CLL archive/native member alias. "
                      "One author-described section is one sample; no ROI-derived patient key. "
                      "Source sample context retained; patient and shared-block identities unresolved.")
                for key in ["sample_id", "source_sample_id", "sample_type", "preservation_method", "anatomical_site"]]
for key in ["patient_key", "source_subject_id"]:
    mapping_rows.append(field(relative(META/"sample_provenance.json"), "Zenodo records; article Methods; supplementary materials; author repository; GEO SOFT",
                              "No donor ID or specimen-to-patient map found", "sample_metadata", key,
                              "Leave empty. All 6 images link to 5 biological samplings; C3/C4 share S3; 0 samples link to patients."))
mapping_rows.append(field(relative(RAW_METADATA/"author_repository/Data_Lymph_node/Gene_Protein_Comparison.Rmd"),
                          "read.csv statement under LYMPH_NODE specimen directory", "native C3 SlideID specimen code + CODEXfeature_labeled_aligned_50um.csv",
                          "sample_metadata", "sample_id",
                          "C3 native SlideID specimen prefix equals the author repository LYMPH_NODE specimen directory that contains "
                          "the 50um feature table. Workbook Metadata maps that exact table to C4. Thus C3 and C4 share S3, "
                          "without a patient inference. Source acquisition IDs and workbook section labels remain separate."))
mapping_rows.append(field(relative(RAW_METADATA/"41592_2025_2948_MOESM1_ESM.pdf"),
                          "Supplementary Tables 3 and 4; Supplementary Fig. 6", "Lymph node 2 (Control); Lymph node 2 (50um); FFPE block Age (in years)",
                          "sample_metadata", "sample_id",
                          "Table 3 names both 49-marker Control (C3) and 35-marker 50um (C4) as lymph node 2; Fig.6 describes adjacent sections. "
                          "This confirms shared sampling S3. Table 4 reports FFPE BLOCK age and RNA QC, so its ages are not patient ages. "
                          "Do not populate age_at_collection_years, age_at_index_years or age_at_diagnosis_years from block storage age. "
                          "Attachments were obtained from Europe PMC supplementaryFiles after Springer TLS/PMC browser-check failures."))
mapping_rows.append(field(relative(RAW_METADATA/"41592_2025_2948_MOESM2_ESM.pdf"),
                          "Reporting Summary pages 2-3 (rendered scan)", "Human participants; ethics oversight; Clinical data", "sample_metadata", "metadata_notes",
                          "Archived de-identified FFPE reactive lymph node and lymphoma blocks from Yale Pathology Tissue Services; "
                          "demographics marked not relevant and clinical data marked not applicable. No donor identifiers or patient table supplied. "
                          "The reporting PDF is scanned; page2/page3 PNGs were visually examined after pdftotext yielded no text."))
for key in ("anatomical_site", "metadata_notes"):
    mapping_rows.append(field(relative(geo_path), "GSE308167 SOFT samples", "Sample_title; Sample_source_name_ch1; Sample_characteristics_ch1",
                              "sample_metadata", key,
                              "Exact author workbook section labels join GEO titles for C4, C5 and C6. The latter two explicitly report lymph node (lymphoma). "
                              "Keep GSM sample accessions in notes; no patient identifier or demographics are present. C4 contributes to shared sample S3."))
mapping_rows.append(field(relative(META/"paper.html"), "Lymphoma results; author workbook Metadata",
                          "MZL_Paraffin_section; CLL_to_DLBCL_Lymphoma_Paraffin_section", "sample_metadata", "primary_diagnosis",
                          "Retain author specimen disease interpretation. MZL native SlideID DLBCL_rerun is preserved as an acquisition label, not a diagnosis override."))
image_rules = {
    "study": ("canonical authorized study", "Use the task canonical study identifier."),
    "image_id": ("sources.source_id", "C1..C6 correspond exactly to ordered source_001..source_006, independently of export order."),
    "sample_id": ("workbook_sample; native SlideID; SI Table3; author 50um table path", "C3/C4 share S3 (lymph node2 biological sampling); other acquisitions join their author-described sampling. No donor inference."),
    "panel_id": ("sources.panel_id; native channel order", "Use the per-file native panel ID; preserve each distinct source order."),
    "image_kind": ("native acquisition geometry", "roi denotes the complete independent native acquisition."),
    "source_image_id": ("member.name or filename", "Exact native ZIP member name, or direct QPTIFF filename."),
    "source_files": ("filename; record_id; proof.native_path", "NAS-relative release container and reused native extraction, deduplicated for the direct file."),
    "source_format": ("proof.format", "QPTIFF or OME-TIFF, verified from native header."),
    "source_acquisition_id": ("native file layout", "Empty: each source file is already one acquisition; there is no MCD acquisition ID."),
    "source_z_index": ("OME Pixels.SizeZ", "Zero for singleton OME Z; empty for QPTIFF with no declared Z axis."),
    "source_t_index": ("OME Pixels.SizeT", "Zero for singleton OME T; empty for QPTIFF with no declared T axis."),
    "source_axes": ("native full-resolution scalar pages", "CYX: channel-ordered full-resolution scalar pages; pyramids and RGB overviews excluded."),
    "source_dtype": ("native page.dtype", "Preserve native uint8 for QPTIFF and uint16 for CellScape OME-TIFF."),
    "source_revision": ("record_id; native-v2 proof", "Bind exact Zenodo release record and native verified order; reuse matching file_stat proof without full hash."),
    "slide_id": ("QPTIFF XML SlideID", "Retain native SlideID verbatim, including documented filename conflicts; OME generic Image0 is not a slide identifier."),
    "core_id": ("source specimen annotation", "Empty: no TMA core ID is supplied."),
    "roi_id": ("source region annotation", "Empty: no additional source ROI identifier is supplied for the complete native frame."),
    "fov_id": ("source field annotation", "Empty: no additional independent FOV identifier is supplied."),
    "region_annotation": ("workbook Metadata.Sample", "Retain author sample/section context verbatim; C3/C4 section labels remain distinct within shared sample S3."),
    "source_x0_px": ("native full-resolution frame", "Zero; retain the complete source frame."),
    "source_y0_px": ("native full-resolution frame", "Zero; retain the complete source frame."),
    "source_x1_px": ("native page.shape[1]", "Full native width W; half-open XYXY bounds."),
    "source_y1_px": ("native page.shape[0]", "Full native height H; half-open XYXY bounds."),
    "pad_left_px": ("native W,H", "floor((max(H,W)-W)/2)."),
    "pad_top_px": ("native W,H", "floor((max(H,W)-H)/2)."),
    "pad_right_px": ("native W,H; pad_left_px", "max(H,W)-W-pad_left_px."),
    "pad_bottom_px": ("native W,H; pad_top_px", "max(H,W)-H-pad_top_px."),
    "height_px": ("native W,H", "max(H,W), with zero padding and no resize."),
    "width_px": ("native W,H", "max(H,W), with zero padding and no resize."),
    "source_channel_count": ("native full-resolution page labels", "Count every source channel, including nuclear and excluded rows."),
    "channel_count": ("panel_mapping.keep_channel", "Count retained nuclear/body/admitted-marker rows; output positions are contiguous."),
    "pixel_size_x_um": ("XResolution + ResolutionUnit or OME PhysicalSizeX", "Convert TIFF cm/inch units to micrometers; retain OME 0.182 micrometers. Never infer from 25um/50um filename labels."),
    "pixel_size_y_um": ("YResolution + ResolutionUnit or OME PhysicalSizeY", "Convert TIFF cm/inch units to micrometers; retain OME 0.182 micrometers. Never infer from 25um/50um filename labels."),
    "input_image_path": ("published per-image destination", "NAS-relative processed/<study>/<image_id>/input.tiff."),
    "valid_pixels_path": ("native TIFF rectangular measurement domain", "Empty: complete rectangular TIFF input, with padding represented by explicit geometric fields; measured zeros remain valid."),
    "intensity_description": ("native published pixel dtype", "Native published intensities; no normalization, rescaling or per-pixel zero filtering."),
    "split_method": ("workbook/native independent acquisition", "already_independent; no disease-region split within the single CLL/DLBCL specimen."),
    "split_evidence": ("workbook Metadata filename-to-section; source overview", "Retain complete native frames. C3 separated pieces share the source specimen; SI Table3 confirms control/50um sampling identity."),
    "metadata_notes": ("identity_caveats; sample provenance", "Preserve filename/native SlideID conflicts and patient identity gap without overriding author specimen interpretation."),
}
assert set(image_rules) == set(image_columns)
for key, (source_field, rule) in image_rules.items():
    mapping_rows.append(field(relative(META/"source_manifest.v2.json"), "sources; native QPTIFF XML / OME XML",
                              source_field, "image_metadata", key, rule))
mapping_rows.append(field(relative(S1), "S1", "Study + Original marker", "panel_mapping", "source_channel_index",
                          "Exact study-specific native labels; no aliases needed. Full-resolution native IFD order -> source_channel_index; "
                          "kept rows -> contiguous output_channel_index. rH2AX excluded only in C3; cTIP/GAP67/CD11e mapping_pending retained. "
                          "All S1-selected nuclear/body channels retained; no duplicate downstream marker in any panel."))
mappings = pd.DataFrame(mapping_rows)
write_json(CACHE/"source_inventory.json", {"sources": inventory, "scope": "Six complete human acquisitions; no additional complete local image objects.",
           "excluded": "Mouse embryo, H&E, workbook-only unavailable 25um, ZIP AppleDouble, RGB overviews and pyramids are outside six-source scope.",
           "output_upper_bound_bytes": sum(i["retained_channels"]*max(i["shape"][1:])**2*np.dtype(i["dtype"]).itemsize for i in inventory),
           "adapter": "Three native tile-row cache plus full TiffPage header materialization for shared scalar window reads after OME series discovery.",
           "controller_requests": CONTROLLER_REQUESTS})
save_tables()
progress("inventory_complete")

if args.mode == "inventory":
    for item in inventory:
        make_overview(proofs[item["image_id"]], item["image_id"])
    progress("overviews_complete")

if args.mode == "export":
    for item, source in zip(inventory, sources):
        image_id = item["image_id"]
        if args.images and image_id not in args.images:
            continue
        panel = panels[panels.panel_id == source["panel_id"]]
        proof = proofs[image_id]
        destination = OUTPUT/image_id/"input.tiff"
        receipt_path = CACHE/"export_receipts"/(image_id+".json")
        if any(row["image_id"] == image_id for row in images) and destination.exists() and receipt_path.exists():
            print(f"Reuse completed {image_id}", flush=True)
            continue
        if destination.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
        else:
            upper = item["retained_channels"]*max(proof["shape"][1:])**2*np.dtype(proof["dtype"]).itemsize
            if shutil.disk_usage(NAS).free < upper+100*1024**3 or shutil.disk_usage("/tmp").free < 64*1024**3:
                progress("capacity_blocked", current_image=image_id, required_upper_bytes=upper)
                raise RuntimeError(f"Insufficient reserved space for {image_id}")
            progress("exporting", current_image=image_id)
            spec = SourceSpec(image_id, source["panel_id"], "tiff", NAS/proof["native_path"],
                              tuple(range(proof["shape"][0])), pixel_size_um=tuple(item["pixel_size_um"]))
            with open_source(spec) as native:
                receipt = export_input(RowCachedReader(native, image_id), panel, destination)
            write_json(receipt_path, receipt)
        if not receipt["admitted"]:
            progress("all_marker_zero_excluded", current_image=image_id)
            continue
        _, h, w = proof["shape"]
        row = dict.fromkeys(image_columns, None)
        row.update(study=STUDY, image_id=image_id, sample_id="S3" if image_id == "C4" else f"S{sources.index(source)+1}", panel_id=source["panel_id"],
                   image_kind="roi", source_image_id=source["member"]["name"] if source["member"] else source["filename"],
                   source_files=list(dict.fromkeys([item["archive_path"], proof["native_path"]])),
                   source_format=proof["format"], source_axes="CYX", source_dtype=proof["dtype"],
                   source_revision=f"Zenodo record {source['record_id']}; native-v2 verified order", slide_id=item["slide_id"] or None,
                   source_z_index=0 if proof["format"] == "OME-TIFF" else None,
                   source_t_index=0 if proof["format"] == "OME-TIFF" else None,
                   region_annotation=source["workbook_sample"], source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                   pixel_size_x_um=item["pixel_size_um"][0], pixel_size_y_um=item["pixel_size_um"][1],
                   input_image_path=relative(destination), intensity_description="Native published " + proof["dtype"] + "; no intensity normalization",
                   split_method="already_independent", split_evidence=relative(RAW_METADATA/"Multiplexed_IF_Metadata.xlsx")+":Metadata filename-to-section",
                   metadata_notes="Complete native acquisition retained, including measured zeros; no patient inference. " + " ".join(source["identity_caveats"]))
        row.update({key:receipt[key] for key in ["height_px", "width_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px", "source_channel_count", "channel_count"]})
        images = [previous for previous in images if previous["image_id"] != image_id] + [row]
        save_tables()
        progress("image_complete", latest_image=image_id)

# Preparer evidence is explicit; these records do not impersonate independent review.
checks = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY,
               check_id="S01", result="pass", source_evidence=relative(CACHE/"source_inventory.json"),
               output_evidence=relative(OUTPUT/"panel_mapping.csv"),
               details=json.dumps({"performed_by": "preparer", "independent_review": False,
                                   "complete_native_sources": 6, "prior_proof_file_stat_reused": True,
                                   "full_hash_rerun": False, "source_channels": len(panels),
                                   "retained_channels_across_panels": int(panels.keep_channel.sum()),
                                   "archive_members": "Native members located; AppleDouble excluded; overviews/pyramids not new images"})),
          dict(study=STUDY, preparer=PREPARER, reviewer="", scope="panel", record_id="all",
               check_id="P01", result="pass", source_evidence=relative(S1)+";"+relative(META/"source_manifest.v2.json"),
               output_evidence=relative(OUTPUT/"panel_mapping.csv"),
               details=json.dumps({"performed_by": "preparer", "independent_review": False,
                                   "exact_native_order_vs_header_and_proof": True,
                                   "s1_join": "Exact Study + Original marker; no aliases needed",
                                   "mapping_pending_retained": ["cTIP", "GAP67", "CD11e"]})),
          dict(study=STUDY, preparer=PREPARER, reviewer="", scope="panel", record_id="all",
               check_id="P02", result="pass", source_evidence=relative(S1), output_evidence=relative(OUTPUT/"panel_mapping.csv"),
               details=json.dumps({"performed_by": "preparer", "independent_review": False,
                                   "nuclear_and_body_preserved": True, "downstream_markers_unique_per_panel": True,
                                   "only_removed_channel": "C3 source index9 rH2AX, S1 Excluded"}))]
if args.mode == "verify":
    for item, source in zip(inventory, sources):
        image_id = item["image_id"]
        if not any(row["image_id"] == image_id for row in images):
            continue
        details = headers_and_pixels(source, proofs[image_id], image_id, panels[panels.panel_id == source["panel_id"]])
        write_json(CACHE/"pixel_checks"/(image_id+".json"), {"image_id": image_id, "preparer_checks": details})
        checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope="image", record_id=image_id,
                           check_id="I05", result="pass", source_evidence=item["native_path"],
                           output_evidence=relative(OUTPUT/image_id/"input.tiff"),
                           details=json.dumps({"performed_by": "preparer", "independent_review": False,
                                               "all_retained_channels": details}, ensure_ascii=False)))
        progress("verifying", latest_image=image_id)
review_path = OUTPUT/"review_results.csv"
existing = pd.read_csv(review_path, keep_default_na=False).to_dict("records") if review_path.exists() else []
checks += [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample", record_id="all", check_id="M01", result="unresolved",
                source_evidence=relative(META/"sample_provenance.json")+";"+relative(RAW_METADATA/"Multiplexed_IF_Metadata.xlsx"),
                output_evidence=relative(OUTPUT/"sample_metadata.csv"),
                details=json.dumps({"performed_by": "preparer", "sample_rows": len(samples), "sample_linked_images": len(images),
                                    "patient_linked_samples": 0, "join": "author Metadata filename -> section; C3/C4 share specimen via native ID and author 50um table path -> S3",
                                    "gap": GAPS[0], "image_ids": [row["image_id"] for row in images]}, ensure_ascii=False)),
           dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study", record_id=STUDY, check_id="D01", result="unresolved",
                source_evidence=relative(CACHE/"source_inventory.json"), output_evidence=relative(OUTPUT),
                details=json.dumps({"performed_by": "preparer", "expected_images": 6, "exported_images": len(images),
                                    "independent_review": "pending", "gap": GAPS}, ensure_ascii=False))]
keys = {(r["scope"],r["record_id"],r["check_id"]) for r in checks}
existing = [r for r in existing if r.get("reviewer") or (r["scope"],r["record_id"],r["check_id"]) not in keys]
with formal_publish_lease(review_path, directory=review_path.parent):
    pd.DataFrame(existing+checks, columns=pd.read_csv(TEMPLATES/"review_results.csv").columns).to_csv(review_path, index=False, encoding="utf-8-sig")
progress("prepared" if len(images) == 6 else "partial", mode=args.mode)
