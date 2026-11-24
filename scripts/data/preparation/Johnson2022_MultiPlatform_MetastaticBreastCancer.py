"""Prepare the nine admitted Johnson acquisitions at native intensity.

Inputs: qualified manifest/native proofs, HTAN source metadata, and study S1 rows.
Outputs: independent square CYX TIFFs and the complete study input tables.
The local DICOM/tile adapters are study-scoped pending controller migration.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import ast
from collections import OrderedDict
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4
import xml.etree.ElementTree as ET
import zipfile

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import pydicom
import tifffile
import openpyxl

from cellweft.data.contracts import SourceSpec, Window
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import windows, read_tiff_window
from cellweft.data.preparation.io_coordination import default_io_root, formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

STUDY = "Johnson2022_MultiPlatform_MetastaticBreastCancer"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = META / "preparation_evidence"
OUT = NAS / "processed" / STUDY
IO_LOCK_ROOT = default_io_root(NAS)
JOB = Path(os.environ.get("CELLWEFT_STUDY_JOB", PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0044_prepare_0"))
TMP = Path("/tmp/study_input_batch") / STUDY
TEMPLATES = Path("docs/data/templates/study_input")
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
SUPPLEMENT = NAS / "raw/public" / STUDY / "metadata/europepmc_PMC8861971/supplementaryFiles.zip"


def clinical_workbook():
    """Read the original Clinical IHC attachment from the received release ZIP."""
    with formal_read_lease(SUPPLEMENT), zipfile.ZipFile(SUPPLEMENT) as archive:
        payload = archive.read("mmc2.xlsx")
    return openpyxl.load_workbook(io.BytesIO(payload), data_only=True, read_only=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def progress(stage: str, **details: object) -> None:
    completed = details.get("completed_image_ids")
    if completed is None:
        with formal_read_lease(OUT), resource_lock(OUT, shared=True):
            completed = (pd.read_parquet(OUT / "images.parquet").image_id.tolist()
                         if (OUT / "images.parquet").exists() else [])
    details.setdefault("gaps",["独立复核待执行","未核实临床时间原点/年龄/生存字段留空",
                              "未提供独立采集有效域注释；原生镶嵌边缘零值原样保留"])
    details.setdefault("completed_image_ids", completed)
    details.setdefault("remaining_image_ids", [source["data_file_id"] for source in sources
                                                if source["data_file_id"] not in completed])
    write_json(JOB / "progress.json", dict(study=STUDY, phase="prepare",
               scientific_repair_round=0, stage=stage,
               expected_images=len(sources), completed_images=len(completed),
               updated_at=datetime.now(timezone.utc).isoformat(), **details))


def records(atlas: dict, component: str) -> list[dict]:
    table = atlas["bts:" + component]
    return [dict(zip(table["column_order"], row["values"])) for row in table["record_list"]]


def prepare_metadata(*, write_lineage: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Follow published biospecimen parents to each biological collection.

    Serial assay sections share their highest biospecimen ancestor immediately
    below the participant. Identifiers are read from NAS, never hard-coded here.
    """
    snapshot = CACHE / "htan_source_subset.json"
    evidence = json.loads(snapshot.read_text())
    by_id = {r["HTAN Biospecimen ID"]: r for r in evidence["biospecimens"]}
    links, sample_rows = {}, {}
    for row in evidence["imaging"]:
        key, chain = row["HTAN Parent Biospecimen ID"], []
        while key in by_id:
            chain.append(key)
            key = by_id[key]["HTAN Parent ID"]
        sample, leaf = by_id[chain[-1]], by_id[chain[0]]
        sid = sample["HTAN Biospecimen ID"]
        links[row["HTAN Data File ID"]] = dict(sample_id=sid, patient_key=key,
              lineage=chain, source_biospecimen_id=chain[0], source_metadata=row)
        if sid not in sample_rows:
            diagnosis = next(r for r in evidence["diagnosis"] if r["HTAN Participant ID"] == key)
            demographic = next(r for r in evidence["demographics"] if r["HTAN Participant ID"] == key)
            sample_rows[sid] = dict(study=STUDY, sample_id=sid, patient_key=key,
                source_subject_id=key, source_sample_id=sid, biospecimen_group_id=sid,
                center_id="HTAN OHSU", sample_type="Tissue", tissue_type="tumor",
                anatomical_site=sample["Site of Resection or Biopsy"], lesion_type={"Metastatic":"metastasis"}[sample["Tumor Tissue Type"]],
                timepoint_label="other", source_timepoint_label=sample["Timepoint Label"],
                collection_method={"Biopsy":"biopsy"}[sample["Acquisition Method Type"]],
                preservation_method={"Formalin fixed paraffin embedded - FFPE":"ffpe"}[leaf["Preservation Method"]], case_control_status="case",
                sex_reported=demographic["Gender"].lower(), primary_diagnosis=diagnosis["Primary Diagnosis"],
                primary_diagnosis_code=diagnosis["Morphology"],
                primary_site=diagnosis["Tissue or Organ of Origin"], histology=diagnosis["Primary Diagnosis"],
                vital_status={"Dead":"deceased"}[demographic["Vital Status"]],
                metadata_notes="HTAN explicit ancestry; serial assay sections share collection ancestor. "
                  "Collection Days from Index and age/survival left blank: reference semantics not verified. "
                  "No treatment or response inferred from image filenames.")
    sample_columns = pd.read_csv(TEMPLATES / "sample_metadata.csv").columns
    samples = pd.DataFrame(sample_rows.values()).reindex(columns=list(sample_columns)+["source_timepoint_label"])
    # Preserve uniquely linked biopsy IHC verbatim. Bx1 lists two lesions with
    # no lesion→biospecimen crosswalk, so its sample-level IHC remains empty.
    clinical = SUPPLEMENT
    if clinical.exists():
        workbook = clinical_workbook()
        rows = list(workbook["Clinical IHC"].values)
        for index, sample in samples.iterrows():
            columns = [i for i, name in enumerate(rows[0]) if i and str(name).startswith(sample.source_timepoint_label)]
            if len(columns)!=1:
                samples.loc[index,"metadata_notes"] += " Clinical IHC has multiple lesion columns without a lesion-to-biospecimen crosswalk; not assigned to this sample."
                continue
            observations = {rows[0][i]: {r[0]: r[i] for r in rows[1:] if r[0] and r[i] not in (None, "N/A")} for i in columns}
            samples.loc[index, "clinical_ihc_source_json"] = json.dumps(observations, ensure_ascii=False)
        workbook.close()
    dictionary=pd.read_csv(TEMPLATES / "data_dictionary.csv",keep_default_na=False)
    vocabulary=pd.read_csv(TEMPLATES / "value_sets.csv",keep_default_na=False)
    for field in dictionary[(dictionary.table_name=="sample_metadata") & (dictionary.value_set!="")].itertuples(index=False):
        if not samples[field.field_name].dropna().isin(vocabulary.loc[vocabulary.value_set==field.value_set,"allowed_value"]).all():
            raise ValueError("Sample category outside template vocabulary: "+field.field_name)
    if write_lineage:
        write_json(CACHE / "image_sample_lineage.json", links)

    # Retain the earliest original target measurement; repeated P/R antibody
    # measurements are assessed in the overview/QC record before export.
    repeated = {"CK5P", "CD8R", "CD4R", "CD20P"}
    panels = []
    for pid in dict.fromkeys(s["panel_id"] for s in sources):
        labels = next(s["ordered_labels"] for s in sources if s["panel_id"] == pid)
        selected = {i for i, label in enumerate(labels) if label not in repeated}
        panel = panel_from_s1(STUDY, pid, labels, S1, downstream_source_indices=selected)
        panel.loc[panel.source_name.isin(repeated), "exclusion_reason"] = (
            "Repeated antibody measurement: retain original earlier round; see duplicate_qc.json and source protocol")
        panels.append(panel)
    panel = pd.concat(panels, ignore_index=True)

    mappings = []
    def mapping(table: str, field: str, source_field: str, rule: str, source: Path = snapshot,
                source_table: str = "HTAN OHSU", notes: str = "") -> None:
        mappings.append(dict(study=STUDY, source_file=relative(source), source_sheet_or_table=source_table,
             source_field=source_field, target_table=table, target_field=field,
             join_keys="HTAN Data File ID → HTAN Parent Biospecimen ID → HTAN Parent ID → HTAN Participant ID",
             value_mapping="", transformation_rule=rule, time_origin_rule="", conflict_rule="Native header owns geometry/channel order", notes=notes))
    for field, source_field in {
        "sample_id":"HTAN Parent Biospecimen ID; HTAN Parent ID", "patient_key":"HTAN Participant ID",
        "source_subject_id":"HTAN Participant ID", "source_sample_id":"HTAN Biospecimen ID",
        "biospecimen_group_id":"HTAN Biospecimen ID", "anatomical_site":"Site of Resection or Biopsy",
        "lesion_type":"Tumor Tissue Type", "timepoint_label":"Timepoint Label",
        "collection_method":"Acquisition Method Type", "preservation_method":"Preservation Method",
        "sex_reported":"Gender", "primary_diagnosis":"Primary Diagnosis", "histology":"Primary Diagnosis",
        "primary_diagnosis_code":"Morphology", "primary_site":"Tissue or Organ of Origin", "vital_status":"Vital Status",
        "center_id":"htan_name", "sample_type":"Biospecimen Type", "tissue_type":"Tumor Tissue Type",
        "case_control_status":"Tumor Tissue Type",
        "metadata_notes":"Source omissions and unverified time semantics"}.items():
        mapping("sample_metadata", field, source_field,
                "Follow explicit ancestry to collection below participant; direct source value or stated categorical recode; no filename-based patient inference")
    recodes={"tissue_type":{"Metastatic":"tumor"},"lesion_type":{"Metastatic":"metastasis"},
             "timepoint_label":{"Biopsy 1":"other","Biopsy 2":"other","Biopsy 3":"other"},
             "collection_method":{"Biopsy":"biopsy"},"preservation_method":{"Formalin fixed paraffin embedded - FFPE":"ffpe"},
             "vital_status":{"Dead":"deceased"},"sex_reported":{"Female":"female"},"case_control_status":{"Metastatic":"case"}}
    for row in mappings:
        if row["target_field"] in recodes:
            row["value_mapping"]=json.dumps(recodes[row["target_field"]],ensure_ascii=False)
    mapping("sample_metadata","source_timepoint_label","Timepoint Label","Preserve original biopsy designation; template timepoint_label=other",notes=json.dumps({"data_type":"string","definition":"Original HTAN biopsy timepoint label"}))
    mapping("image_metadata", "sample_id", "HTAN Data File ID; HTAN Parent Biospecimen ID; HTAN Parent ID",
            "Exact acquisition join and parent traversal: 9/9 image links, 3 samples, 3/3 samples linked to 1 participant")
    for field in ["image_id", "source_image_id", "source_files", "source_format", "source_axes", "source_dtype",
                  "source_revision", "slide_id", "roi_id", "region_annotation", "source_x0_px", "source_y0_px",
                  "source_x1_px", "source_y1_px", "pad_left_px", "pad_top_px", "pad_right_px", "pad_bottom_px",
                  "height_px", "width_px", "source_channel_count", "channel_count", "input_image_path",
                  "valid_pixels_path", "intensity_description", "split_method", "split_evidence", "metadata_notes",
                  "image_kind", "pixel_size_x_um", "pixel_size_y_um", "panel_id"]:
        mapping("image_metadata", field, "Native proof/header; source filename; HTAN PhysicalSizeX/Y",
                "Complete native acquisition; centered zero padding; source marker order retained after S1 selection",
                META / "qualified_source_manifest.json")
    for row in mappings:
        if row["target_field"] in {"pixel_size_x_um","pixel_size_y_um"}:
            row["source_field"]="Native OME Pixels PhysicalSizeX/Y; DICOM PixelSpacing"
            row["transformation_rule"]="Native OME um copied directly; DICOM mm multiplied by 1000. CyCIF 0.325 um; mIHC native 0.5022 um supersedes portal 0.5002 um."
            row["conflict_rule"]="Native source header owns pixel scale"
    mapping("panel_mapping", "all_source_rows", "ordered_labels; Study; Original marker",
            "Exact study/name S1 join; no cross-study aliases. Repeated target: original earlier round after source QC; selected nuclear/body inputs remain.", S1)
    mapping("sample_metadata", "clinical_ihc_source_json", "Clinical IHC Assay; Biopsy 1/2/3 columns",
            "Exact Timepoint Label → unique matching Clinical IHC column prefix. Two Biopsy 1 lesion columns lack lesion→biospecimen identity and are not sample-assigned. Omit N/A; retain source values for Biopsy 2/3.",
            clinical, "mmc2.xlsx / Clinical IHC", notes=json.dumps({"data_type":"string","definition":"JSON of uniquely sample-linked source clinical IHC observations; ambiguous lesion assignments remain empty"}))
    mapping("sample_metadata", "clinical_missingness", "Table S1; author repository",
            "Local HTAN metadata, paper, and EuropePMC supplementary Clinical IHC/RECIST/Treatments tables inspected; unverified index-linked treatment/response/age/survival summaries remain empty",
            CACHE / "paper.xml", notes="https://pmc.ncbi.nlm.nih.gov/articles/PMC8861971/ ; https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8861971/supplementaryFiles ; author repository returned 404")
    return samples, panel, pd.DataFrame(mappings), links


class NativeReader:
    """Bounded native TIFF/DICOM tile reads with at most two tile rows cached.

    DICOM here is Explicit VR Little Endian, scalar uint16, TILED_FULL with
    row-major frames. No full WSI channel or multichannel array is allocated.
    """
    def __init__(self, source: dict, proof: dict, paths: list[Path]):
        self.shape = tuple(proof["shape"])
        self.dtype = np.dtype(proof["dtype"])
        self._gate_path = paths[0]
        self.tiffs = []
        self.handles = []
        self.pages = []
        self.cache: OrderedDict = OrderedDict()
        self.dicom = proof.get("representation") == "idc_dicom_mirror"
        if self.dicom:
            for path in paths:
                with path.open("rb") as fh:
                    header = pydicom.dcmread(fh, stop_before_pixels=True)
                    pixel_tag = fh.read(12)
                    if pixel_tag[:4] != b"\xe0\x7f\x10\x00" or pixel_tag[4:6] != b"OW":
                        raise ValueError("Unexpected native DICOM PixelData encoding")
                    offset = fh.tell()
                if (str(header.file_meta.TransferSyntaxUID) != "1.2.840.10008.1.2.1" or
                    header.DimensionOrganizationType != "TILED_FULL" or header.SamplesPerPixel != 1 or
                    header.BitsAllocated != 16 or header.PixelRepresentation != 0 or
                    int(header.TotalPixelMatrixRows) != self.shape[1] or int(header.TotalPixelMatrixColumns) != self.shape[2]):
                    raise ValueError("DICOM differs from the admitted full-resolution scalar layout")
                self.pages.append((offset, int(header.Rows), int(header.Columns), int(header.NumberOfFrames)))
                self.handles.append(path.open("rb"))
            sy,sx=header.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
            scale=(float(sx)*1000,float(sy)*1000)
        else:
            tif = tifffile.TiffFile(paths[0]); self.tiffs.append(tif)
            pixels=ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
            names = [c.get("Name") for c in pixels.findall("{*}Channel")]
            if names != source["ordered_labels"] or tuple(tif.series[0].shape) != self.shape:
                raise ValueError("Native TIFF shape/channel identity differs")
            if pixels.get("PhysicalSizeXUnit")!="µm" or pixels.get("PhysicalSizeYUnit")!="µm":
                raise ValueError("Native physical-size unit differs from reviewed source layout")
            scale=(float(pixels.get("PhysicalSizeX")),float(pixels.get("PhysicalSizeY")))
            self.pages = [p.aspage() if isinstance(p, tifffile.TiffFrame) else p for p in tif.pages]
        self.source = SourceSpec(source["data_file_id"], source["panel_id"],
                                 "dicom" if self.dicom else "tiff", paths[0],
                                 tuple(range(self.shape[0])), pixel_size_um=scale)
        self.active_channel = None

    def close(self) -> None:
        for resource in self.tiffs + self.handles:
            resource.close()

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        if self._gate_path.is_relative_to(NAS):
            with formal_read_lease(self._gate_path):
                return self._read_window_unlocked(channels, window)
        return self._read_window_unlocked(channels, window)

    def _read_window_unlocked(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        result = np.zeros((len(channels), y1-y0, x1-x0), dtype=self.dtype)
        for output, channel in enumerate(channels):
            if self.active_channel != channel:
                self.cache.clear(); self.active_channel = channel
            if self.dicom:
                offset, th, tw, frames = self.pages[channel]
            else:
                page = self.pages[channel]; th, tw = page.tilelength, page.tilewidth
            columns = (self.shape[2] + tw - 1) // tw
            for ty in range(y0//th, (y1-1)//th + 1):
                for tx in range(x0//tw, (x1-1)//tw + 1):
                    index = ty * columns + tx
                    if index not in self.cache:
                        if self.dicom:
                            fh = self.handles[channel]; fh.seek(offset + index*th*tw*2)
                            body = fh.read(th*tw*2)
                            tile = np.frombuffer(body, dtype="<u2").reshape(th, tw)
                        else:
                            fh = self.tiffs[0].filehandle; fh.seek(page.dataoffsets[index])
                            tile = page.decode(fh.read(page.databytecounts[index]), index)[0][0, :, :, 0]
                        self.cache[index] = tile
                        while len(self.cache) > 2*columns + 4:
                            self.cache.popitem(last=False)
                    tile = self.cache[index]
                    sx, sy = tx*tw, ty*th
                    a, b, c, d = max(x0,sx), max(y0,sy), min(x1,sx+tw), min(y1,sy+th)
                    result[output,b-y0:d-y0,a-x0:c-x0] = tile[b-sy:d-sy,a-sx:c-sx]
        return result


def source_paths(source: dict, proof: dict) -> list[Path]:
    """Resolve the admitted representation from its own verified native proof.

    The alternate BEMS350173 Scene-002 CRDC TIFF fragment stays inventoried as
    incomplete in raw manifest.json. IDC DICOM verification applies to the
    admitted DICOM files; equivalence of the preserved TIFF fragment is unresolved.
    """
    if proof["status"] != "verified" or proof["ordered_labels"] != source["ordered_labels"]:
        raise ValueError("Admitted panel order differs from its native proof")
    if proof.get("representation") == "idc_dicom_mirror":
        manifest = json.loads((META / "idc_representation_manifest.json").read_text())
        if [c["marker"] for c in manifest["channels"]] != source["ordered_labels"]:
            raise ValueError("DICOM representation channel order differs from original acquisition")
        paths = [NAS / manifest["raw_path"] / c["source"]["name"] for c in manifest["channels"]]
        for index, path in enumerate(paths):
            p = json.loads((META / "idc_proofs" / source["synapse_id"] / f"channel_{index:02}.json").read_text())
            if [path.stat().st_size, path.stat().st_mtime_ns] != p["file_stat"][2:]:
                raise ValueError("Native DICOM proof file binding changed")
    else:
        paths = [NAS / proof["native_path"]]
        if [paths[0].stat().st_size, paths[0].stat().st_mtime_ns] != proof["file_stat"][2:]:
            raise ValueError("Native TIFF proof file binding changed")
    return paths


def stage_sources(paths: list[Path]) -> list[Path]:
    """Stage complete native files atomically on SSD, reusing finished copies."""
    TMP.mkdir(parents=True,exist_ok=True)
    local=[]
    for path in paths:
        target=TMP/path.name
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            if shutil.disk_usage('/tmp').free < path.stat().st_size + 64*1024**3:
                raise RuntimeError("Local staging would violate the 64 GiB reserve")
            temporary=target.with_name(target.name+f".staging-{os.getpid()}")
            with formal_read_lease(path):
                shutil.copyfile(path,temporary)
            os.replace(temporary,target)
        local.append(target)
    return local


def recover_complete_partial(reader: NativeReader, selected_panel: pd.DataFrame,
                             destination: Path, *, reuse_existing_partial: bool = True) -> dict | None:
    """Validate and publish a complete image retained after an I/O interruption."""
    partial = destination.with_name(destination.name + ".partial")
    local_complete = TMP / (destination.parent.name + ".complete.tiff")
    if not local_complete.exists():
        return None
    candidate = local_complete
    complete_size = local_complete.stat().st_size
    saved_candidates = (sorted(destination.parent.glob(destination.name + ".recovery-*.partial"),
                               key=lambda path: path.stat().st_mtime_ns, reverse=True)
                        if reuse_existing_partial else [])
    if reuse_existing_partial:
        saved_candidates.append(partial)
    for saved_candidate in saved_candidates:
        if not saved_candidate.exists() or saved_candidate.stat().st_size != complete_size:
            continue
        comparison_path = CACHE / (destination.parent.name + "_partial_comparison.json" if saved_candidate == partial
                                   else destination.parent.name + "_" + saved_candidate.name.replace(".", "_") + "_comparison.json")
        identity = dict(local_stat=[complete_size, local_complete.stat().st_mtime_ns],
                        partial_stat=[saved_candidate.stat().st_size, saved_candidate.stat().st_mtime_ns])
        if saved_candidate != partial:
            identity["partial_path"] = relative(saved_candidate)
        comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
        compared = int(comparison.get("compared_bytes", 0)) if comparison.get("identity") == identity else 0
        with formal_read_lease(saved_candidate), local_complete.open("rb") as local, saved_candidate.open("rb") as saved:
            tail = max(0, complete_size - 8 * 1024 * 1024)
            local.seek(tail)
            saved.seek(tail)
            if local.read(complete_size - tail) != saved.read(complete_size - tail):
                progress("partial_rejected", current_image=destination.parent.name,
                         candidate=relative(saved_candidate), reason="tail differs from local complete TIFF",
                         completed_image_ids=images.image_id.tolist())
                continue
            local.seek(compared)
            saved.seek(compared)
            while True:
                block = local.read(8 * 1024 * 1024)
                if saved.read(len(block)) != block:
                    break
                if not block:
                    write_json(comparison_path, dict(identity=identity, compared_bytes=compared))
                    progress("partial_compared", current_image=destination.parent.name,
                             compared_bytes=compared, comparison_path=relative(comparison_path),
                             completed_image_ids=images.image_id.tolist())
                    candidate = saved_candidate
                    break
                compared += len(block)
                if compared % (256 * 1024 * 1024) == 0:
                    write_json(comparison_path, dict(identity=identity, compared_bytes=compared))
                    progress("comparing_partial", current_image=destination.parent.name,
                             compared_bytes=compared, comparison_path=relative(comparison_path),
                             completed_image_ids=images.image_id.tolist())
        if candidate == saved_candidate:
            break
    kept = selected_panel[selected_panel.keep_channel].sort_values("output_channel_index")
    side = max(reader.shape[-2:])
    h, w = reader.shape[-2:]
    left, top = (side-w)//2, (side-h)//2
    with formal_read_lease(candidate), tifffile.TiffFile(candidate) as tif:
        if (len(tif.pages) != len(kept) or tif.series[0].shape != (len(kept), side, side)
                or tif.series[0].dtype != reader.dtype):
            raise ValueError("Interrupted TIFF has incomplete pages or incompatible geometry")
        pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
        names = [item.get("Name") for item in pixels.findall("{*}Channel")]
        if (names != kept.marker_id.tolist() or
                (float(pixels.get("PhysicalSizeX")), float(pixels.get("PhysicalSizeY")))
                != reader.source.pixel_size_um):
            raise ValueError("Interrupted TIFF channel or scale metadata differs")
        size = candidate.stat().st_size
        pages = [page.aspage() if isinstance(page, tifffile.TiffFrame) else page for page in tif.pages]
        for page in pages:
            if (page.shape != (side, side) or page.dtype != reader.dtype or
                    len(page.dataoffsets) != ((side+1023)//1024)**2 or
                    any(offset + count > size for offset, count in zip(page.dataoffsets, page.databytecounts))):
                raise ValueError("Interrupted TIFF has missing or truncated tile data")
        scalar = SimpleNamespace(pages=pages, filehandle=tif.filehandle)
        for x, y in ((0, 0), (max(0, w//2-16), max(0, h//2-16)),
                     (max(0, w-32), max(0, h-32))):
            source_window = (x, y, min(x+32, w), min(y+32, h))
            output_window = (x+left, y+top, source_window[2]+left, source_window[3]+top)
            native = reader.read_window(kept.source_channel_index.tolist(), source_window)
            saved = np.stack([read_tiff_window(scalar, index, output_window)
                              for index in range(len(kept))])
            if not np.array_equal(native, saved):
                raise ValueError("Interrupted TIFF differs from native pixels")
    progress("publishing_local_complete", current_image=destination.parent.name,
             local_complete_bytes=complete_size, completed_image_ids=images.image_id.tolist())
    with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
        if candidate == local_complete:
            recovery = destination.with_name(destination.name + f".recovery-{uuid4().hex}.partial")
            shutil.copyfile(local_complete, recovery)
            os.replace(recovery, destination)
        else:
            os.replace(candidate, destination)
        with tifffile.TiffFile(destination) as tif:
            if (len(tif.pages) != len(kept) or tif.series[0].shape != (len(kept), side, side)
                    or tif.series[0].dtype != reader.dtype):
                raise ValueError("Published TIFF header differs from validated local image")
    return dict(height_px=side, width_px=side, pad_left_px=left, pad_top_px=top,
                pad_right_px=side-w-left, pad_bottom_px=side-h-top,
                source_dtype=str(reader.dtype), source_channel_count=reader.shape[0],
                channel_count=len(kept), admitted=True, valid_pixels=None,
                recovered_complete_partial=True)


def overview(source: dict, proof: dict, paths: list[Path]) -> None:
    """Create bounded full-extent overview and repeated-antibody QC samples."""
    from PIL import Image, ImageDraw
    link = links[source["data_file_id"]]
    row = link["source_metadata"]
    reader = NativeReader(source, proof, paths)
    h,w = reader.shape[-2:]; labels=source["ordered_labels"]
    qc_names = (["DAPI1", "CK19", "CD45"] + [x for x in ["CD20", "CD20P", "CD8", "CD8R", "CD4", "CD4R", "CK5", "CK5P"] if x in labels]
                if source["assay"] == "CyCIF" else ["DSDNA", "PANCK", "CD45"])
    qc_names = list(dict.fromkeys(qc_names))
    step = max(1, int(np.ceil(max(h,w)/700)))
    arrays = []
    stats = {}
    try:
        for name in qc_names:
            channel=labels.index(name); thumb=np.zeros(((h+step-1)//step,(w+step-1)//step),dtype=reader.dtype)
            for region in windows((h,w),1024):
                x0,y0,x1,y1=region; ay,ax=((y0+step-1)//step)*step,((x0+step-1)//step)*step
                if ay>=y1 or ax>=x1:continue
                values=reader.read_window([channel],region)[0]
                thumb[ay//step:(y1+step-1)//step,ax//step:(x1+step-1)//step]=values[ay-y0::step,ax-x0::step]
            stats[name]=dict(sampled_nonzero=int(np.count_nonzero(thumb)),sampled_pixels=int(thumb.size),maximum=int(thumb.max()))
            high=np.percentile(thumb[thumb>0],99) if np.any(thumb) else 1
            arrays.append(Image.fromarray(np.uint8(np.clip(thumb.astype(float)/high,0,1)*255)).convert("RGB"))
        cell_w=max(a.width for a in arrays);cell_h=max(a.height for a in arrays)+25
        canvas=Image.new("RGB",(cell_w*min(4,len(arrays)),cell_h*((len(arrays)+3)//4)),"white")
        draw=ImageDraw.Draw(canvas)
        for index,(name,array) in enumerate(zip(qc_names,arrays)):
            x,y=(index%4)*cell_w,(index//4)*cell_h;canvas.paste(array,(x,y+25));draw.text((x+5,y+5),name,fill="black")
        canvas.save(CACHE / f"{source['synapse_id']}_overview.jpg")
        write_json(CACHE / f"{source['synapse_id']}_overview.json",dict(source_shape=reader.shape,step=step,channels=stats,
                   display_only="Each channel scaled to its sampled nonzero P99; input pixels unchanged"))
    finally:
        reader.close()


def write_review(images: pd.DataFrame, *, io_lock_root: Path | None = None) -> None:
    review=[]
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples(index=False):
        review.append(dict(study=STUDY,preparer="Codex study preparer",reviewer="",scope=check.scope,
             record_id=STUDY,check_id=check.check_id,result="unresolved",
             source_evidence=relative(META / "qualified_source_manifest.json")+";"+relative(CACHE / "htan_source_subset.json"),
             output_evidence=relative(OUT),details=json.dumps(dict(preparer_self_check=True,independent_review="pending",
                completed_images=len(images),expected_images=len(sources),sample_rows=len(samples),
                sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                source_image_sample_links=len(links),
                patient_linked_samples=int(samples.patient_key.notna().sum()),
                native_reader_evidence=relative(CACHE / "dicom_reader_all_channels_validation.json"),
                source_overview_evidence=relative(CACHE / "overview_review.json"),
                exported_pixel_validation="not performed" if not len(images) else "see preparer_verification.json when generated",
                note="Source lineage and native-reader checks are preparer evidence; independent acceptance required. "
                  "Bx1 clinical IHC lesion→biospecimen crosswalk and unverified time semantics remain explicit clinical gaps."),ensure_ascii=False)))
    review_path = OUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUT, io_lock_root=io_lock_root):
        pd.DataFrame(review).to_csv(review_path,index=False,encoding="utf-8-sig")


def verify_images(images: pd.DataFrame, all_images: pd.DataFrame) -> None:
    """Preparer self-check: every header plus two native pixel blocks/channel.

    This records concrete comparison coordinates and zero-padding probes;
    independent D01 acceptance is deliberately left pending.
    """
    from PIL import Image
    checks=[]
    for image in images.itertuples(index=False):
        source=next(s for s in sources if s["data_file_id"]==image.image_id)
        proof=json.loads((META / "proofs" / (source["synapse_id"]+".json")).read_text())
        paths=source_paths(source,proof)
        kept=panel[(panel.panel_id==image.panel_id) & panel.keep_channel].sort_values("output_channel_index")
        reader=NativeReader(source,proof,stage_sources(paths))
        h,w=reader.shape[-2:];blocks=[];padding=[]
        qc=json.loads((CACHE / f"{source['synapse_id']}_overview.json").read_text())
        step=qc["step"];thumb_h,thumb_w=(h+step-1)//step,(w+step-1)//step
        # The second overview panel is CK19/PANCK. Use its source-aligned
        # bright location only to choose a signal-containing comparison block.
        with Image.open(CACHE / f"{source['synapse_id']}_overview.jpg") as picture:
            locator=np.asarray(picture.convert("L"))[25:25+thumb_h,thumb_w:2*thumb_w]
        yy,xx=np.unravel_index(np.argmax(locator),locator.shape)
        signal_xy=(int(min(max(xx*step-16,0),w-32)),int(min(max(yy*step-16,0),h-32)))
        try:
            with formal_read_lease(NAS/image.input_image_path), tifffile.TiffFile(NAS/image.input_image_path) as tif:
                names=[c.get("Name") for c in ET.fromstring(tif.ome_metadata).findall("{*}Image/{*}Pixels/{*}Channel")]
                assert tif.series[0].shape==(len(kept),max(h,w),max(h,w))
                assert tif.series[0].dtype==reader.dtype and names==kept.marker_id.tolist()
                pixels=ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
                assert (float(pixels.get("PhysicalSizeX")),float(pixels.get("PhysicalSizeY"))) == reader.source.pixel_size_um
                assert (image.pixel_size_x_um,image.pixel_size_y_um) == reader.source.pixel_size_um
                # Shared read_tiff_window currently expects scalar TiffPage
                # attributes absent on OME TiffFrame objects. Adapt this study
                # locally; keep the common-reader correction with controller.
                scalar=SimpleNamespace(pages=[p.aspage() if isinstance(p,tifffile.TiffFrame) else p for p in tif.pages],
                                       filehandle=tif.filehandle)
                for index,(x,y) in enumerate([signal_xy,(0,max(0,h-32))]):
                    region=(x,y,min(x+32,w),min(y+32,h))
                    native=reader.read_window(kept.source_channel_index.tolist(),region)
                    target=tuple(int(value) for value in
                                 (region[0]+image.pad_left_px,region[1]+image.pad_top_px,
                                  region[2]+image.pad_left_px,region[3]+image.pad_top_px))
                    saved=np.stack([read_tiff_window(scalar,i,target) for i in range(len(kept))])
                    assert saved.dtype==native.dtype and np.array_equal(native,saved)
                    if index==0:
                        assert np.any(native[(kept.role=="downstream").to_numpy()])
                    blocks.append(dict(source_xyxy=region,output_xyxy=target,channels=len(kept),equal=True))
                side=max(h,w)
                probes=[]
                if image.pad_left_px:probes.append((0,0,min(16,image.pad_left_px),16))
                if image.pad_top_px:probes.append((0,0,16,min(16,image.pad_top_px)))
                if image.pad_right_px:probes.append((side-min(16,image.pad_right_px),side-16,side,side))
                if image.pad_bottom_px:probes.append((side-16,side-min(16,image.pad_bottom_px),side,side))
                for region in probes:
                    assert all(not np.any(read_tiff_window(scalar,i,region)) for i in range(len(kept)))
                    padding.append(tuple(int(value) for value in region))
            checks.append(dict(image_id=image.image_id,header_pass=True,pixel_blocks=blocks,padding_probes=padding))
        finally:
            reader.close()
    verification_path = CACHE / "preparer_verification.json"
    previous = json.loads(verification_path.read_text()) if verification_path.exists() else {"images": []}
    verified = {item["image_id"]: item for item in previous["images"]}
    verified.update({item["image_id"]: item for item in checks})
    write_json(verification_path,dict(study=STUDY,independent_review=False,images=list(verified.values()),
               sample_rows=len(samples),sample_linked_images=int(all_images.sample_id.notna().sum()),patient_linked_samples=int(samples.patient_key.notna().sum())))
    print("preparer verification",len(verified),"images",flush=True)


def strong_version(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path), dev=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def require_version(expected: dict) -> None:
    path = Path(expected["path"])
    if strong_version(path) != expected:
        raise ValueError(f"Strong version changed: {path}")


def receive_preflight(receipt_path: Path, plan_path: Path, root_path: Path) -> tuple[dict, dict]:
    for required in (CACHE / "htan_source_subset.json", CACHE / "image_sample_lineage.json",
                     CACHE / "paper.xml", SUPPLEMENT):
        if not required.is_file():
            raise ValueError(f"Required closed metadata source missing: {required}")
    plan = json.loads(plan_path.read_text())
    root = json.loads(root_path.read_text())
    service_path = Path(root["service_receipt"]["path"])
    require_version(root["service_receipt"])
    service = json.loads(service_path.read_text())
    require_version(root["native_plan_version"])
    require_version(service["plan_version"])
    if Path(service["plan_version"]["path"]) != plan_path:
        raise ValueError("Byte service used a different native plan")
    if (root["service"] != dict(MainPID="0", NRestarts="0", ExecMainStatus="0", ActiveState="inactive")
            or service["exit_code"] != 0 or service["stage"] != "byte_published"
            or service["pid"] != 3574712 or service["start_ticks"] != 60208070
            or not all(root[key] for key in ("complete_byte_equality", "successful_same_handle_fsync_close",
                                             "successful_destination_directory_fsync", "all_native_plan_versions_current",
                                             "closed_native_CLI_reopened", "formal_ninth_input_published"))):
        raise ValueError("Closed successful byte publication is not established")
    if (root["formal_version"] != service["formal_version"] or
            root["service_progress"]["formal_version"] != service["formal_version"]):
        raise ValueError("Byte publication receipts disagree")
    require_version(root["formal_version"])
    if (plan["study"] != STUDY or plan["image_id"] != "HTA9_1_19404" or
            root["formal_version"]["path"] != plan["expected_formal_path"] or
            plan["current_images"]["count"] != 8 or len(plan["current_images"]["original_eight_input_versions"]) != 8):
        raise ValueError("Native plan does not describe the missing ninth input")
    # The old scientific reader and pixel-verification code remains byte-for-byte in AST form.
    before = ast.parse((JOB / "per_study_script.before.py").read_text())
    current = ast.parse(Path(__file__).read_text())
    scientific = {"NativeReader", "source_paths", "stage_sources", "recover_complete_partial",
                  "overview", "verify_images"}
    def science_nodes(tree: ast.Module) -> dict:
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in scientific}
    if science_nodes(before) != science_nodes(current) or len(science_nodes(before)) != len(scientific):
        raise ValueError("Native scientific code differs from frozen script")
    old_script = plan["input_versions_after"][str(Path(__file__).resolve())]
    before_stat = strong_version(JOB / "per_study_script.before.py")
    if (before_stat["size"] != old_script["size"] or
            before_stat["mtime_ns"] != old_script["mtime_ns"]):
        raise ValueError("Frozen script copy is unavailable")
    # The original path version is superseded only by this audited metadata/IO delta.
    for path, expected in plan["input_versions_after"].items():
        if path not in {str(Path(__file__).resolve()), str(IO_LOCK_ROOT / "control.json")}:
            require_version(expected)
    for expected in plan["source_files"] + plan["native_cache_files"]:
        require_version(expected)
    require_version(plan["local_complete"])
    source = next(s for s in sources if s["data_file_id"] == plan["image_id"])
    if source != plan["source_manifest_record"]:
        raise ValueError("Current source manifest record changed")
    proof_path = META / "proofs" / (source["synapse_id"] + ".json")
    proof = json.loads(proof_path.read_text())
    proof_fields = {key: proof[key] for key in ("status", "shape", "axes", "dtype", "ordered_labels", "file_stat")}
    expected_proof = {key: value for key, value in plan["native_proof"].items() if key != "path"}
    if proof_fields != expected_proof or source_paths(source, proof) != [Path(v["path"]) for v in plan["source_files"]]:
        raise ValueError("Current native proof or source binding changed")
    if (plan["source_spec"]["path"] != plan["native_cache_files"][0]["path"] or
            plan["source_spec"]["pixel_size_um"] != root["root_header_reopen"]["physical_size_um"] or
            plan["expected"] != root["root_header_reopen"] or
            len(plan["checks"]["pixel_windows"]) != 3 or
            not all(item["all_selected_channels_equal"] and item["channels"] == 39
                    for item in plan["checks"]["pixel_windows"]) or
            len(plan["checks"]["zero_padding_probes"]) != 2 or
            not all(value is True for key, value in plan["checks"].items() if key not in {"pixel_windows", "zero_padding_probes"})):
        raise ValueError("Closed native scientific evidence is incomplete")
    if receipt_path.exists():
        raise ValueError("Ninth image export receipt already exists")
    return plan, root


def image_row(source: dict, proof: dict, paths: list[Path], result: dict,
              pixel_size_um: tuple[float, float]) -> dict:
    image_id = source["data_file_id"]
    destination = OUT / "images" / image_id / "input.tiff"
    metadata = links[image_id]["source_metadata"]
    h,w=proof["shape"][-2:]
    row=dict(study=STUDY,image_id=image_id,sample_id=links[image_id]["sample_id"],panel_id=source["panel_id"],
        image_kind="roi",source_image_id=source["filename"],source_files=[relative(p) for p in paths],
        source_format="dicom" if len(paths)>1 else "ome_tiff",source_axes="CYX",source_dtype=proof["dtype"],
        source_revision="HTAN Release4 CRDC v22.6.2.img / source native proof" if len(paths)==1 else "IDC native full-resolution representation / source native proof",
        slide_id=source["filename"].split('_Scene-')[0] if source["assay"]=="CyCIF" else None,
        roi_id=source["filename"].removesuffix('.ome.tif'),region_annotation=metadata.get("Comment"),
        source_x0_px=0,source_y0_px=0,source_x1_px=w,source_y1_px=h,
        pixel_size_x_um=pixel_size_um[0],pixel_size_y_um=pixel_size_um[1],
        input_image_path=relative(destination),valid_pixels_path=None,intensity_description="Native published level-2 intensities; lossless dtype, no normalization",
        split_method="already_independent",split_evidence=relative(CACHE / "image_sample_lineage.json"),
        metadata_notes="Full admitted source acquisition. Portal SizeC/SizeX/SizeY can be stale; native proof/header owns geometry. "
            "55 DICOM files constitute one acquisition." if len(paths)>1 else
            "Full admitted source acquisition; native header owns geometry/channel count/scale. "
            "For mIHC native OME scale 0.5022 um supersedes portal 0.5002 um.")
    row.update({k:result[k] for k in ['height_px','width_px','pad_left_px','pad_top_px','pad_right_px','pad_bottom_px','source_channel_count','channel_count']})
    return row


# Prepare metadata before acquiring the shared large-I/O slot.
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("action",choices=["metadata","overview","stage","export","verify"])
parser.add_argument("--image",help="Exact HTAN data file ID; omission processes all admitted acquisitions")
parser.add_argument("--recover-complete-partial", action="store_true",
                    help="Validate and publish a complete TIFF left by an interrupted NAS operation")
parser.add_argument("--publish-local-complete", action="store_true",
                    help="Publish the validated closed local TIFF without comparing old NAS partials")
args=parser.parse_args()
CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
sources=json.loads((META / "qualified_source_manifest.json").read_text())["sources"]
samples,panel,field_mapping,links=prepare_metadata()
image_columns=pd.read_csv(TEMPLATES / "image_metadata.csv").columns
with formal_read_lease(OUT), resource_lock(OUT, shared=True):
    images=pd.read_parquet(OUT / "images.parquet") if (OUT / "images.parquet").exists() else pd.DataFrame(columns=image_columns)
if args.action in {"metadata","export"}:
    save_input_tables(OUT,images,samples,panel,field_mapping)
    write_review(images)
    progress("metadata_ready",sample_rows=len(samples),patient_linked_samples=int(samples.patient_key.notna().sum()),
             gaps=["独立复核待执行","未核实时点/年龄/生存字段留空"])

if args.action == "verify":
    verify_images(images if args.image is None else images[images.image_id==args.image], images)

if args.action in {"overview","stage","export"}:
    selected=[s for s in sources if args.image is None or s["data_file_id"] == args.image]
    if not selected:raise ValueError("No admitted image matches the requested ID")
    # Small acquisitions first; every completed image is published immediately.
    selected.sort(key=lambda s: np.prod(json.loads((META / "proofs" / (s["synapse_id"]+".json")).read_text())["shape"]))
    for source in selected:
        image_id=source["data_file_id"]
        proof=json.loads((META / "proofs" / (source["synapse_id"]+".json")).read_text())
        paths=source_paths(source,proof)
        if args.action == "overview":
            progress("overview",current_image=image_id)
            overview(source,proof,paths)
            continue
        if args.action == "stage":
            progress("staging",current_image=image_id)
            stage_sources(paths)
            print(image_id,"staged",flush=True)
            continue
        destination=OUT / "images" / image_id / "input.tiff"
        receipt=CACHE / f"{image_id}_export.json"
        if image_id in images.image_id.values and receipt.exists() and destination.exists():
            saved=json.loads(receipt.read_text())
            if [destination.stat().st_size,destination.stat().st_mtime_ns] != saved["output_stat"]:
                raise ValueError("Existing output changed; targeted review required")
            print(image_id,"reuse",flush=True);continue
        if source["panel_id"] in {"CyCIF_1","CyCIF_3"} and not (CACHE / "duplicate_qc.json").exists():
            raise ValueError("Repeated antibody overview decision must precede this panel export")
        retained=panel[(panel.panel_id==source["panel_id"]) & panel.keep_channel]
        side=max(proof["shape"][-2:]);estimate=len(retained)*side*side*np.dtype(proof["dtype"]).itemsize
        staging=sum(p.stat().st_size for p in paths)
        if shutil.disk_usage(NAS).free < estimate + 100*1024**3 or shutil.disk_usage('/tmp').free < staging + 64*1024**3:
            progress("capacity_gap",current_image=image_id,required_output_upper_bytes=estimate,staging_bytes=staging)
            raise RuntimeError("Capacity reserve would be violated")
        progress("staging",current_image=image_id,required_output_upper_bytes=estimate)
        local=stage_sources(paths)
        reader=NativeReader(source,proof,local)
        progress("exporting",current_image=image_id)
        export_error = None
        try:
            selected_panel = panel[panel.panel_id==source["panel_id"]]
            result = (recover_complete_partial(reader, selected_panel, destination,
                                               reuse_existing_partial=not args.publish_local_complete)
                      if args.recover_complete_partial else None)
            if result is None:
                local_complete = TMP / (image_id + ".complete.tiff")
                staged_result = export_input(reader,selected_panel,local_complete,tile_size=1024)
                if staged_result["admitted"]:
                    progress("local_complete", current_image=image_id,
                             local_complete_bytes=local_complete.stat().st_size,
                             completed_image_ids=images.image_id.tolist())
                    result = recover_complete_partial(reader, selected_panel, destination)
                else:
                    result = staged_result
        except OSError as exc:
            export_error = exc
        finally:
            reader.close()
        if export_error is not None:
            local_complete = TMP / (image_id + ".complete.tiff")
            progress("waiting_io",current_image=image_id,error=str(export_error),
                     operation="publish validated local TIFF to formal input.tiff",
                     local_complete_path=str(local_complete),
                     local_complete_bytes=local_complete.stat().st_size if local_complete.exists() else None,
                     resume_action="export --image " + image_id +
                                   " --recover-complete-partial --publish-local-complete",
                     completed_image_ids=images.image_id.tolist(),
                     remaining_image_ids=[s["data_file_id"] for s in sources if s["data_file_id"] not in images.image_id.values],
                     sample_rows=len(samples),sample_linked_images=int(images.sample_id.notna().sum()),
                     patient_linked_samples=int(samples.patient_key.notna().sum()),
                     gaps=["NAS input.tiff publication interrupted; resume at current_image",
                           "Independent review pending", "Unverified clinical time semantics remain empty"])
            raise export_error
        if not result["admitted"]:
            write_json(receipt,dict(image_id=image_id,result=result,exclusion="All retained downstream channels are zero"))
            continue
        row=image_row(source,proof,paths,result,reader.source.pixel_size_um)
        images=pd.concat([images[images.image_id!=image_id],pd.DataFrame([row]).reindex(columns=image_columns)],ignore_index=True)
        save_input_tables(OUT,images,samples,panel,field_mapping)
        write_json(receipt,dict(image_id=image_id,result=result,source_proof=relative(META / "proofs" / (source['synapse_id']+'.json')),
                    output_stat=[destination.stat().st_size,destination.stat().st_mtime_ns]))
        write_review(images)
        progress("image_completed",current_image=image_id,sample_rows=len(samples),sample_linked_images=int(images.sample_id.notna().sum()),
                 patient_linked_samples=int(samples.patient_key.notna().sum()),gaps=["独立复核待执行"])
        print(image_id,"completed",result["channel_count"],result["height_px"],flush=True)
    progress("prepared" if args.action=="export" and len(images)==len(sources) else "ready_for_next_step",
             sample_rows=len(samples),sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
             patient_linked_samples=int(samples.patient_key.notna().sum()),gaps=["独立复核待执行","未核实临床时点语义保持空值"])
