"""Prepare the published Greenwald TONIC MIBI ROIs at native SPORA intensities.

Inputs: SPORA 1_0mpp OME-Zarr, accepted native proofs, S1, small author
metadata members from Zenodo TONIC_Cohort.zip (17065908), and the cached
BioStudies core/sample map. Source retrieval evidence is in metadata_trace.json.
Outputs: independent CYX input.tiff and the five per-study input tables.
Run metadata first, then export with the shared formal write lease.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import signal
import time
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Iterator

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import blosc

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative
from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, panel_from_s1, save_input_tables


STUDY = "Greenwald2026_MIBI_Breast"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = NAS / f"raw/public/{STUDY}/images/spora_release"
SOURCE = metadata_source_directory(NAS, STUDY) / "source_attachments"
OUTPUT = NAS / f"processed/{STUDY}"
CACHE = Path(__file__).resolve().parents[3] / "cache/study_preparation" / STUDY
EVIDENCE = OUTPUT / "preparation_evidence"
RECEIPTS = EVIDENCE / "receipts"
LOCAL = Path("/tmp/study_input_batch") / STUDY
JOB = CACHE / "jobs/prepare"
PROOF = SOURCE / "source_evidence/verification.json"
INVENTORY = SOURCE / "source_evidence/mibi_source_objects.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "Greenwald2026_MIBI_Breast_mibi39"
PREPARER = "codex_0212_prepare_0"
REVISION = "SPORA-greenwald2026temporal-1_0mpp-native-accepted-1258"
NUCLEAR_EVIDENCE = "https://github.com/angelolab/publications/blob/main/2024-Greenwald_Nederlof_etal_TONIC/single_cell_analysis/Segment_Image_Data_working.ipynb"
AUTHOR_ZIP = "https://zenodo.org/records/17065908"
ISSUES = ["309个ROI在SPORA及作者core表均无生物学取样/患者对应；其占位patient_id不作为真实患者",
          "H3K27me3/H3K9ac在当前S1为excluded且无核输入标记；按当前规则从input.tiff排除，核输入缺口待主控协调",
          "28个取样的BioStudies图像清单分类为Liver，但上层Biosample注释为Human Lung Cells；保留图像清单分类并记录冲突供独立复核",
          "独立复核尚待主控安排；整理者不写D01 pass"]
REQUESTS = ["从本轮JOB/progress.json接续未完成图像；共享NAS读取有超时记录，科学repair轮保持0",
            "安排本study已关闭的实际图像范围及两级metadata独立复核；D01当前为unresolved，309张身份未明ROI保留局部M01缺口，另复核28份取样的Liver/Lung来源注释冲突",
            "请主控将本轮实际输入数量、覆盖及I/O等待记录到docs/project/status.md，并协调后续批次",
            "请主控裁定S1本study H3K27me3/H3K9ac核输入缺口：作者使用二者，但当前S1 excluded且核用途留空；本次按S1排除，若决策变更需重导受影响TIFF及复核",
            "共享export_input对全通道共同分块的Zarr逐通道读取会重复解码；本study采用单ROI有界解码及Zstd无损TIFF，后续迁移可配置codec及chunk-aware读取到共享模块"]


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


class IOWaitTimeout(BaseException):
    """The current publication could not finish within its I/O budget."""


@contextmanager
def io_deadline() -> Iterator[None]:
    """Bound an I/O operation and release its nested leases on timeout."""
    def timed_out(signum, frame):
        raise IOWaitTimeout(f"Shared NAS I/O exceeded {args.io_wait_seconds}s")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, args.io_wait_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def progress(stage: str, images: pd.DataFrame, **extra) -> None:
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False) if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    represented = samples[samples.sample_id.isin(images.sample_id)] if len(images) else samples.iloc[:0]
    save_json(JOB / "progress.json", dict(study=STUDY, stage=stage, expected_images=1258,
        exported_images=len(images), sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
        patient_linked_samples=int(samples.patient_key.ne("").sum()) if len(samples) else 0,
        distinct_patients=int(samples.patient_key[samples.patient_key.ne("")].nunique()) if len(samples) else 0,
        exported_sample_rows=len(represented), exported_patients=int(represented.patient_key[represented.patient_key.ne("")].nunique()) if len(represented) else 0,
        issues=ISSUES, controller_requests=REQUESTS, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra))


def publish_tables(images: pd.DataFrame, samples: pd.DataFrame, panel: pd.DataFrame, fields: pd.DataFrame) -> None:
    """Merge and publish tables through the shared short publication path."""
    with io_deadline():
        save_input_tables(OUTPUT, images, samples, panel, fields)


def source_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Connect exact acquisition IDs to author Tissue_ID, never the ROI ordinal.

    The 949 matches agree with every nonempty SPORA tissue, timepoint and
    patient field. The remaining 309 use a synthetic SPORA patient bucket.
    Author Tissue_ID is the biological specimen; repeated cores share one row.
    """
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    cores = pd.read_csv(SOURCE / "author_metadata/TONIC_data_per_core.csv", dtype=str)
    timepoints = pd.read_csv(SOURCE / "author_metadata/TONIC_data_per_timepoint.csv", dtype=str)
    patients = pd.read_csv(SOURCE / "author_metadata/TONIC_data_per_patient.csv", dtype=str)
    original_map = pd.read_csv(SOURCE / "biostudies_core_sample_map.csv", dtype=str)
    comparison = original_map.merge(cores, on="fov", how="left", validate="one_to_one", suffixes=("_bia", "_author"))
    for key in ["Tissue_ID", "Patient_ID", "Timepoint"]:
        assert (comparison[key+"_bia"] == comparison[key+"_author"]).all()
    assert original_map.groupby("Tissue_ID").anatomical_site.nunique().max() == 1
    sites = original_map.drop_duplicates("Tissue_ID").set_index("Tissue_ID").anatomical_site
    assert cores.fov.is_unique and timepoints.Tissue_ID.is_unique and patients.Patient_ID.is_unique
    joined = tissues.merge(cores, left_on="original_id", right_on="fov", how="left", validate="one_to_one", suffixes=("_spora", "_author"))
    known = joined.Tissue_ID.notna()
    assert (joined.loc[known, "original_tissue_identifier"] == joined.loc[known, "Tissue_ID"]).all()
    assert (joined.loc[known, "Timepoint_spora"] == joined.loc[known, "Timepoint_author"]).all()
    assert (pd.to_numeric(joined.loc[known, "original_patient_identifier"]) == pd.to_numeric(joined.loc[known, "Patient_ID"])).all()
    assert (joined.loc[~known, "original_patient_identifier"] == "Unknown").all()
    joined["sample_id"] = joined.Tissue_ID.map(lambda value: f"TONIC_{value}" if pd.notna(value) else None)
    selected = joined[known].drop_duplicates("Tissue_ID").set_index("Tissue_ID")
    assert joined[known].groupby("Tissue_ID").Patient_ID.nunique().max() == 1
    assert joined[known].groupby("Tissue_ID").Timepoint_author.nunique().max() == 1
    assert selected.index.isin(timepoints.Tissue_ID).all() and selected.Patient_ID.isin(patients.Patient_ID).all()
    samples = pd.DataFrame(columns=pd.read_csv(TEMPLATES / "sample_metadata.csv").columns)
    time_map = {"baseline": "baseline", "pre_nivo": "pre_treatment", "on_nivo": "on_treatment", "on_nivo_1_cycle": "on_treatment"}
    rows = []
    for tissue, record in selected.iterrows():
        tp = record.Timepoint_author
        row = dict.fromkeys(samples.columns, "")
        row.update(study=STUDY, sample_id=record.sample_id, patient_key=f"TONIC_{record.Patient_ID}",
            source_subject_id=record.Patient_ID, cohort_id="TONIC", source_sample_id=tissue,
            sample_type="tumor specimen", tissue_type="tumor", primary_site="breast",
            primary_diagnosis="triple-negative breast cancer", preservation_method="ffpe",
            timepoint_label=time_map.get(tp, "other"),
            metadata_notes=json.dumps(dict(source_timepoint=tp, source_tissue_with_blocks=sorted(joined.loc[joined.Tissue_ID == tissue, "Tissue_ID_with_block"].dropna().unique()),
                evidence="author TONIC_data_per_core/timepoint/patient; biological specimen Tissue_ID; source timepoint relative to nivolumab where applicable; no individual response/survival released in these tables"), ensure_ascii=False))
        if tp in {"primary", "primary_other"}:
            row["lesion_type"] = "primary"
        elif tp == "local_recurrence":
            row["lesion_type"] = "recurrence"
        elif tp in {"baseline", "pre_nivo", "on_nivo", "on_nivo_1_cycle", "metastasis_1", "metastasis_2"}:
            row["lesion_type"] = "metastasis"
        if tissue in sites.index:
            row["anatomical_site"] = sites.loc[tissue]
            notes = json.loads(row["metadata_notes"])
            notes["source_anatomical_site_category"] = sites.loc[tissue]
            if sites.loc[tissue] == "liver":
                notes["source_annotation_conflict"] = "S-BIAD1288 image-specific Liver file list and component name are retained; the component Biosample association instead says Human Lung Cells"
            row["metadata_notes"] = json.dumps(notes, ensure_ascii=False)
        rows.append(row)
    samples = pd.DataFrame(rows, columns=samples.columns)

    channels = pd.read_parquet(RAW / "mibi/channels.parquet")
    available = pd.read_parquet(RAW / "mibi/channels_per_tissue.parquet")
    assert list(available.columns) == channels.channel_name.tolist() and available.to_numpy().all()
    assert set(available.index) == set(tissues.tissue_id)
    panel = panel_from_s1(STUDY, PANEL_ID, channels.channel_name.tolist(), S1)
    # Author evidence is retained in field_mapping; current S1 leaves both
    # histone-modification nuclear uses unselected.
    nuclear = channels.is_nuclear_marker.to_numpy()
    assert channels.loc[nuclear, "channel_name"].tolist() == ["H3K27me3", "H3K9ac"]
    assert (panel.loc[nuclear, "mapping_status"] == "excluded").all()
    assert not panel.loc[nuclear, ["nuclear", "keep_channel"]].to_numpy().any()
    assert int(panel.keep_channel.sum()) == 37

    fields = []
    def mapping(source_file: str, source_field: str, target: str, target_field: str, rule: str, notes: str = "") -> None:
        fields.append(dict(study=STUDY, source_file=source_file, source_sheet_or_table="", source_field=source_field,
            target_table=target, target_field=target_field, join_keys="SPORA original_id=author fov; author Tissue_ID; author Patient_ID",
            value_mapping="", transformation_rule=rule, time_origin_rule="", conflict_rule="Assert all overlapping SPORA and author fields agree", notes=notes))
    core_path = relative(SOURCE / "author_metadata/TONIC_data_per_core.csv")
    mapping(core_path, "fov;Tissue_ID", "image_metadata", "sample_id", "Exact original_id=fov join; TONIC_ + Tissue_ID; missing author identities remain null", "949/1258 images linked to 356 biological specimens; 309 unknown acquisition identities remain unresolved")
    for target in ["sample_id", "source_sample_id"]:
        mapping(core_path, "Tissue_ID", "sample_metadata", target, "One specimen per source Tissue_ID, shared by all cores; sample_id uses TONIC_ namespace")
    for target in ["patient_key", "source_subject_id"]:
        mapping(core_path, "Patient_ID", "sample_metadata", target, "Exact author patient identifier; patient_key uses TONIC_ namespace", "356/356 specimens link to 112 patients; SPORA's synthetic Unknown bucket is not a patient")
    mapping(relative(SOURCE / "author_metadata/TONIC_data_per_timepoint.csv"), "Timepoint", "sample_metadata", "timepoint_label", "baseline→baseline; pre_nivo→pre_treatment; on_nivo/on_nivo_1_cycle→on_treatment; other source timepoints→other; preserve exact values in metadata_notes")
    mapping(core_path, "Timepoint", "sample_metadata", "lesion_type", "primary/primary_other→primary; local_recurrence→recurrence; baseline/pre_nivo/on_nivo/on_nivo_1_cycle/metastasis_1/metastasis_2→metastasis")
    mapping(relative(SOURCE / "biostudies_core_sample_map.csv"), "Tissue_ID;anatomical_site", "sample_metadata", "anatomical_site", "Preserve original BioStudies image-specific file-list category; exact specimen-key propagation to all cores of that specimen; retain the more specific file-list/component-name assignment when a general Biosample association disagrees", "678 FOVs / 294 specimens; all tissue/patient/timepoint keys agree. The Liver component Biosample association says Human Lung Cells; original Liver file-list classification and this conflict are preserved in sample notes. 309 unknown ROI absent.")
    mapping(relative(SOURCE / "biostudies_record.json"), "Sample preparation protocol;Description", "sample_metadata", "sample_type;tissue_type;preservation_method;primary_diagnosis;primary_site;cohort_id", "Study-level tumor specimens, FFPE, TNBC, breast, TONIC; no patient-specific clinical values inferred")
    mapping(core_path, "Tissue_ID_with_block;Timepoint", "sample_metadata", "metadata_notes", "Preserve source timepoint and distinct source block labels; no grouping by block or by ROI ordinal")
    mapping(relative(RAW / "metadata/tissues.parquet"), "tissue_id;original_id;wrong_tissue_id", "image_metadata", "image_id;source_image_id;core_id;fov_id;roi_id;slide_id;metadata_notes", "Corrected canonical tissue_id is image_id; original_id is acquisition/core/FOV identity; slide parsed from the explicit TONIC_TMA{number}_R{row}C{column} key; embedded OME names equal explicit wrong_tissue_id legacy aliases and are provenance only")
    mapping(core_path, "Pathologist_assessment_HE;Timepoint;MIBI_DATA_ACQUIRED", "image_metadata", "region_annotation", "Preserve exact source values as JSON, including No Tissue/No Tumor; image admission depends only on full-image marker signal")
    mapping(relative(RAW / "mibi/1_0mpp/images"), "OME-Zarr multiscales;array metadata;pixels", "image_metadata", "source_files;source_format;source_axes;source_dtype;source_revision;source_channel_count;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;channel_count;pixel_size_x_um;pixel_size_y_um;input_image_path;valid_pixels_path;intensity_description;split_method;split_evidence", "Complete independent published ROI, native float32, no intensity operation; OME scale=1um (original_mpp=0.6 refers to prior acquisition); no missing chunks by source inventory; full square ROI; no split or pixelwise zero filtering")
    mapping(relative(RAW / "mibi/channels.parquet"), "channel_name;is_nuclear_marker", "panel_mapping", "source_name;source_channel_index;nuclear;role;keep_channel;output_channel_index", "Canonical measured order equals availability columns and every embedded OME channel list; author nuclear labels retained as source evidence only", NUCLEAR_EVIDENCE)
    mapping(relative(S1), "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "marker_id;mapping_status;role;body;keep_channel", "Exact study/name lookup; 37 admitted expression markers retained; both histone modifications excluded without S1 nuclear selection")
    mapping(relative(SOURCE / "supplementary_table.xlsx"), "Supplementary tables 1,3,4", "sample_metadata", "metadata_notes", "Reviewed clinical aggregate and patient/timepoint summary; no unique core→specimen map for the 309 missing ROI and no individual outcome values", "Publication data availability restricts individual response/sequencing data; do not infer these from cohort summaries")
    mapping(relative(SOURCE / "metadata_trace.json"), "public_sources;join_coverage;older_metadata_exact_comparisons", "image_metadata", "metadata_notes", "Record specimen/patient gap after local SPORA, current and earlier author release, seven BioStudies file lists, author repository and paper supplement tracing")
    return joined, samples, panel, pd.DataFrame(fields)


def inspect_sources(tissues: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Reuse accepted native proofs and recheck manifest member sizes, without hashes."""
    proof = json.loads(PROOF.read_text())
    by_id = {entry["image_id"]: entry for entry in proof["verified"]}
    assert set(by_id) == set(tissues.tissue_id) and not proof["failed"]
    official_split = SOURCE / "source_evidence/official_nimbus_tonic_split.json"
    official_ids = {name for group in json.loads(official_split.read_text()).values() for name in group}
    assert set(tissues.original_id) == official_ids
    manifest = json.loads(INVENTORY.read_text())
    member_paths = set()
    for entry in manifest:
        suffix = entry["key"].removeprefix("spora-submissions/greenwald2026temporal/")
        path = RAW / suffix
        assert path.is_file() and path.stat().st_size == entry["size"], path
        member_paths.add(path)
    actual = set((RAW / "mibi/1_0mpp/images").rglob("*"))
    actual_files = {path for path in actual if path.is_file() and not path.name.endswith(".partial")}
    assert actual_files == member_paths
    uncompressed = 0
    shapes = Counter()
    for row in tissues.itertuples():
        path = RAW / f"mibi/1_0mpp/images/{row.tissue_id}.ome.zarr"
        group = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
        array = json.loads((path / "0/zarr.json").read_text())
        assert group["omero"]["name"] == row.wrong_tissue_id
        assert [c["label"] for c in group["omero"]["channels"]] == panel.source_name.tolist()
        assert array["shape"] == by_id[row.tissue_id]["shape"] and array["data_type"] == "float32"
        assert array["dimension_names"] == ["c", "y", "x"]
        scale = group["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale == [1.0, 1.0, 1.0]
        uncompressed += int(np.prod(array["shape"])) * 4
        shapes[str(array["shape"])] += 1
    result = dict(expected_images=len(tissues), source_objects=len(manifest), source_bytes=sum(x["size"] for x in manifest),
        native_uncompressed_output_bytes=uncompressed, shapes=dict(shapes), prior_native_proof=relative(PROOF),
        original_acquisition_id_evidence=relative(official_split), original_acquisition_ids_checked=len(official_ids),
        embedded_legacy_alias_check=dict(images_checked=len(tissues), source_key="tissues.wrong_tissue_id = embedded OME omero.name",
            decision="Keep corrected SPORA tissue_id as image_id; use exact original_id for author acquisition/specimen joins; retain legacy alias as provenance only"),
        verification="Reused accepted source-specific proofs; actual member sizes, OME channels, array shapes and scale checked; no full rehash",
        patient_linked_images=int(tissues.sample_id.notna().sum()), samples=int(tissues.sample_id.nunique()), patients=int(tissues.Patient_ID.nunique()),
        unresolved_image_ids=tissues.loc[tissues.sample_id.isna(), "tissue_id"].tolist(), issues=ISSUES, controller_requests=REQUESTS,
        source_urls=["https://www.nature.com/articles/s43018-026-01114-5", AUTHOR_ZIP, NUCLEAR_EVIDENCE],
        author_metadata_download="HTTP 206 ZIP central-directory and selected-member reads; five metadata CSVs extracted, no full 5.8GB archive download",
        portal_link_issue="SPORA portal Zenodo 18418221 refers to Gupta ageing breast, not TONIC; actual author dataset is 17065908, publication DOI 14112852 resolves to that version")
    save_json(EVIDENCE / "source_inventory.json", result)
    return result


def image_row(record: pd.Series, shape: tuple[int, ...]) -> dict:
    _, h, w = shape
    side = max(h, w)
    notes = "Author specimen/patient verified" if pd.notna(record.sample_id) else "No biological specimen/patient mapping after SPORA, author archive, repository and paper supplement tracing; synthetic SPORA patient not used"
    notes += f"; embedded OME legacy alias from source wrong_tissue_id={record.wrong_tissue_id}; corrected canonical tissue_id retained"
    return dict(study=STUDY, image_id=record.tissue_id, sample_id=record.sample_id, panel_id=PANEL_ID,
        image_kind="roi", source_image_id=record.original_id,
        source_files=[relative(RAW / f"mibi/1_0mpp/images/{record.tissue_id}.ome.zarr")],
        source_format="ome-zarr", source_axes="CYX", source_dtype="float32", source_revision=REVISION,
        slide_id=record.original_id.rsplit("_", 1)[0], core_id=record.original_id, roi_id=record.original_id, fov_id=record.original_id,
        region_annotation=json.dumps({key: str(record[key]) for key in ["Timepoint_author", "Pathologist_assessment_HE_author", "MIBI_DATA_ACQUIRED_author"] if pd.notna(record[key])}, ensure_ascii=False),
        source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
        pad_left_px=(side-w)//2, pad_top_px=(side-h)//2, pad_right_px=side-w-(side-w)//2, pad_bottom_px=side-h-(side-h)//2,
        height_px=side, width_px=side, source_channel_count=39, channel_count=37,
        pixel_size_x_um=1.0, pixel_size_y_um=1.0,
        input_image_path=relative(OUTPUT / f"images/{record.tissue_id}/input.tiff"),
        intensity_description="Published SPORA 1_0mpp float32 intensities preserved bit-for-bit; no additional scaling, normalization or filtering",
        split_method="already_independent", split_evidence="SPORA independent ROI; exact original_id match to author acquisition/core identity; complete 1_0mpp bounding frame",
        metadata_notes=notes)


def write_native_tiff(staged: Path, values: np.ndarray, panel: pd.DataFrame) -> None:
    kept = panel[panel.keep_channel].sort_values("output_channel_index")
    assert values.shape[0] == len(kept) == 37
    staged.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(staged, values, bigtiff=True, ome=True, photometric="minisblack", tile=(512, 512),
            compression="zstd", compressionargs={"level": 1}, predictor=True, maxworkers=2,
            metadata=dict(axes="CYX", Channel={"Name": kept.marker_id.tolist()},
                PhysicalSizeX=1.0, PhysicalSizeXUnit="µm", PhysicalSizeY=1.0, PhysicalSizeYUnit="µm"))
    with tifffile.TiffFile(staged) as tif:
        assert tif.series[0].shape == values.shape and tif.series[0].dtype == values.dtype
        assert len(tif.pages) == 37
        tif.pages.useframes = False
        tif.pages.cache = False
        side = values.shape[-1]
        for x0, y0, x1, y1 in [(side//2-16, side//2-16, side//2+16, side//2+16), (0, 0, 32, 32)]:
            for channel in range(37):
                assert np.array_equal(read_tiff_window(tif, channel, (x0, y0, x1, y1)), values[channel, y0:y1, x0:x1])


def publish_closed_tiff(staged: Path, destination: Path) -> None:
    """Publish one checked local TIFF after durable, bounded NAS copying."""
    with io_deadline(), formal_publish_lease(destination, directory=OUTPUT):
        with resource_lock(destination.parent, name=".input.lock"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name("input.tiff.partial")
            with staged.open("rb") as source, temporary.open("wb") as target:
                pending = 0
                while block := source.read(8 * 1024**2):
                    target.write(block)
                    pending += len(block)
                    if pending >= 64 * 1024**2:
                        target.flush()
                        os.fsync(target.fileno())
                        pending = 0
                target.flush()
                os.fsync(target.fileno())
            assert temporary.stat().st_size == staged.stat().st_size
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def rewrite_native_tiles(path: Path, staged: Path, panel: pd.DataFrame) -> None:
    """Copy retained compressed source tiles and check decoded output blocks."""
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    indices = kept.source_channel_index.tolist()
    with tifffile.TiffFile(path) as source:
        source.pages.useframes = False
        source.pages.cache = False
        names = [channel.attrib["Name"] for channel in ET.fromstring(source.ome_metadata).findall(".//{*}Channel")]
        assert names == panel.marker_id.where(panel.keep_channel, panel.source_name).tolist()
        assert len(source.pages) == 39 and source.series[0].dtype == np.dtype("float32")
        shape = source.series[0].shape[-2:]
        source.pages.useframes = False
        source.pages.cache = False
        assert all((source.pages[index].compression == 50000 and source.pages[index].predictor == 3
            and source.pages[index].tilewidth == source.pages[index].tilelength == 512) for index in indices)
        staged.parent.mkdir(parents=True, exist_ok=True)

        def tiles():
            with path.open("rb") as raw:
                for index in indices:
                    page = source.pages[index]
                    for offset, count in zip(page.dataoffsets, page.databytecounts):
                        raw.seek(offset)
                        payload = raw.read(count)
                        assert len(payload) == count
                        yield payload

        tifffile.imwrite(staged, data=tiles(), shape=(37, *shape), dtype=np.float32,
            bigtiff=True, ome=True, photometric="minisblack", tile=(512, 512),
            compression="zstd", predictor=True,
            metadata=dict(axes="CYX", Channel={"Name": kept.marker_id.tolist()},
                PhysicalSizeX=1.0, PhysicalSizeXUnit="µm", PhysicalSizeY=1.0, PhysicalSizeYUnit="µm"))
        with tifffile.TiffFile(staged) as target:
            assert target.series[0].shape == (37, *shape) and target.series[0].dtype == np.dtype("float32")
            target.pages.useframes = False
            target.pages.cache = False
            side = shape[-1]
            for x0, y0, x1, y1 in [(side//2-16, side//2-16, side//2+16, side//2+16), (0, 0, 32, 32)]:
                for output, index in enumerate(indices):
                    assert np.array_equal(read_tiff_window(source, index, (x0, y0, x1, y1)),
                        read_tiff_window(target, output, (x0, y0, x1, y1)))


def export_roi(record: pd.Series, panel: pd.DataFrame) -> tuple[dict, dict]:
    """Decode a bounded small ROI once, avoiding repeated shared-C shard decoding.

    Actual ROI sizes are checked from the accepted source; no whole-slide
    decoder is used. Source pixels are float32; Zstd+float predictor is lossless.
    This local adapter should migrate to configurable shared export behavior.
    """
    path = RAW / f"mibi/1_0mpp/images/{record.tissue_id}.ome.zarr/0"
    cached = LOCAL / f"{record.tissue_id}.ome.zarr"
    # A ready directory is published only after every compressed source member
    # has been copied; partial prefetches are never used as image sources.
    read_path = cached / "0" if cached.exists() else path
    spec = SourceSpec(record.tissue_id, PANEL_ID, "zarr", read_path, tuple(range(39)), pixel_size_um=(1.0, 1.0), source_revision=REVISION)
    with formal_read_lease(read_path):
        with open_source(spec) as reader:
            shape = reader.shape
            assert int(np.prod(shape)) * 4 <= 256 * 1024**2
            values = reader.read_window(list(range(39)), (0, 0, shape[2], shape[1]))
    assert values.dtype == np.dtype("float32") and np.isfinite(values).all()
    nonzero = np.count_nonzero(values, axis=(1, 2))
    admitted = bool(nonzero[(panel.role == "downstream").to_numpy()].any())
    row = image_row(record, shape)
    receipt = dict(image_id=record.tissue_id, admitted=admitted, channel_nonzero_pixels=nonzero.tolist(),
        source_shape=list(shape), source_revision=REVISION, source_objects=[dict(path=relative(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(path.parent.rglob("*")) if p.is_file()])
    if not admitted:
        return row, receipt
    side = max(shape[-2:])
    if shape[1] != shape[2]:
        padded = np.zeros((39, side, side), dtype=values.dtype)
        padded[:, row["pad_top_px"]:row["pad_top_px"]+shape[1], row["pad_left_px"]:row["pad_left_px"]+shape[2]] = values
        values = padded
    destination = NAS / row["input_image_path"]
    staged = LOCAL / "output_staging" / record.tissue_id / "input.tiff"
    kept_indices = panel.loc[panel.keep_channel].sort_values("output_channel_index").source_channel_index.to_numpy()
    write_native_tiff(staged, values[kept_indices], panel)
    receipt["output_bytes"] = staged.stat().st_size
    receipt["output_channel_count"] = 37
    receipt["decoded_from"] = str(read_path)
    receipt["pixel_checks"] = "37 retained channels: center 32x32 and top-left 32x32 exactly equal to decoded source; dtype float32"
    # This closed, checked local TIFF can be published after a lease timeout.
    save_json(LOCAL / f"receipts/{record.tissue_id}.json", receipt)
    publish_closed_tiff(staged, destination)
    return row, receipt


def prefetch_sources(tissues: pd.DataFrame) -> None:
    """Stage compressed ROI members while queued for the single NAS writer.

    Run under shared formal read lease; source inventory is already size-verified.
    Each ready directory is atomic, and only this study's temporary copies
    are removed after delivery. The source and formal TIFFs stay on NAS.
    """
    LOCAL.mkdir(parents=True, exist_ok=True)
    copied = 0
    for _, record in tissues.iterrows():
        target = LOCAL / f"{record.tissue_id}.ome.zarr"
        if target.exists() or (OUTPUT / f"images/{record.tissue_id}/input.tiff").exists():
            continue
        if shutil.disk_usage("/tmp").free < 64*1024**3 + 256*1024**2:
            raise RuntimeError("Prefetch would consume reserved local capacity")
        source = RAW / f"mibi/1_0mpp/images/{record.tissue_id}.ome.zarr"
        staging = target.with_name(target.name + ".copying")
        for original in source.rglob("*"):
            if not original.is_file():
                continue
            destination = staging / original.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_name(destination.name + ".copying")
                with formal_read_lease(original):
                    shutil.copyfile(original, temporary)
                assert temporary.stat().st_size == original.stat().st_size
                os.replace(temporary, destination)
        os.replace(staging, target)
        copied += 1
        save_json(CACHE / "prefetch_progress.json", dict(stage="prefetching", copied_rois=copied, last_image=record.tissue_id, local_root=str(LOCAL)))
        if copied % 50 == 0:
            print(json.dumps(dict(prefetched=copied, last_image=record.tissue_id)), flush=True)
    save_json(CACHE / "prefetch_progress.json", dict(stage="prefetch_complete", copied_rois=copied, local_root=str(LOCAL)))
    print(json.dumps(dict(prefetch_complete=copied)), flush=True)


def recover_unindexed_export(record: pd.Series, panel: pd.DataFrame) -> tuple[dict, dict] | None:
    """Recover a checked NAS or local TIFF after an interrupted publication.

    Only a matching source version, output size and explicit pixel-check
    receipt permit reuse. Unconfirmed partial files are not candidates.
    """
    receipt_path = RECEIPTS / f"{record.tissue_id}.json"
    path = OUTPUT / f"images/{record.tissue_id}/input.tiff"
    local_receipt = not receipt_path.exists()
    if not receipt_path.exists():
        receipt_path = LOCAL / f"receipts/{record.tissue_id}.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    candidate = path
    if not candidate.exists() and receipt.get("confirmed_partial", False):
        candidate = path.with_name("input.tiff.partial")
    if not candidate.exists():
        candidate = LOCAL / "output_staging" / record.tissue_id / "input.tiff"
    if not candidate.exists():
        return None
    if (not receipt["admitted"] or receipt["source_revision"] != REVISION
            or receipt.get("output_channel_count") != 37 or receipt["output_bytes"] != candidate.stat().st_size):
        return None
    for member in receipt["source_objects"]:
        stat = (NAS / member["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (member["size"], member["mtime_ns"]):
            return None
    row = image_row(record, tuple(receipt["source_shape"]))
    with io_deadline(), ExitStack() as stack:
        stack.enter_context(formal_read_lease(candidate))
        if candidate.is_relative_to(NAS):
            stack.enter_context(resource_lock(OUTPUT, shared=True))
            stack.enter_context(resource_lock(path.parent, name=".input.lock", shared=True))
        tif = stack.enter_context(tifffile.TiffFile(candidate))
        names = [channel.attrib["Name"] for channel in ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
        kept_names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
        if names != kept_names or tif.series[0].dtype != np.dtype("float32") or tif.series[0].shape != (37, row["height_px"], row["width_px"]):
            return None
    if candidate == path and local_receipt:
        staged = LOCAL / "output_staging" / record.tissue_id / "input.tiff"
        if not staged.exists() or staged.stat().st_size != receipt["output_bytes"]:
            return None
        publish_closed_tiff(staged, path)
    elif candidate != path:
        if candidate == path.with_name("input.tiff.partial"):
            raise ValueError("Unproven NAS partial cannot be published")
        publish_closed_tiff(candidate, path)
    receipt["recovered_without_reexport"] = True
    return row, receipt


def reconcile_existing_panel(images: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Remove the two S1-excluded channels from closed native TIFFs.

    Prior TIFFs had all 39 native source channels and per-image pixel receipts.
    Each corrected file is checked locally and atomically replaces its prior
    version under the formal publication lease; interruptions resume by receipt.
    """
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    source_indices = kept.source_channel_index.to_numpy()
    completed = 0
    if args.resume_progress is not None:
        prior = json.loads(args.resume_progress.read_text())
        completed = int(prior["corrected_images"])
        assert 0 < completed <= len(images)
        last_id = images.iloc[completed - 1].image_id
        assert prior.get("last_completed_image", last_id) == last_id
        if completed < len(images) and "current_image" in prior:
            assert prior["current_image"] == images.iloc[completed].image_id
        last_receipt = json.loads((RECEIPTS / f"{last_id}.json").read_text())
        assert last_receipt["output_channel_count"] == 37
        assert last_receipt["retained_source_indices"] == source_indices.tolist()
        assert last_receipt["output_bytes"] == (OUTPUT / f"images/{last_id}/input.tiff").stat().st_size
        progress("correcting_panel", images, corrected_images=completed,
            last_completed_image=last_id, resumed_from=str(args.resume_progress),
            scientific_repair_round=0)
    for record in images.iloc[completed:].itertuples():
        image_id = record.image_id
        path = OUTPUT / f"images/{image_id}/input.tiff"
        receipt_path = RECEIPTS / f"{image_id}.json"
        local_receipt_path = LOCAL / f"panel_correction/{image_id}.json"
        receipt = json.loads(receipt_path.read_text())
        staged = LOCAL / "panel_correction" / image_id / "input.tiff"
        local_source = staged.with_name("source_39.tiff")
        try:
            if receipt.get("output_channel_count") != 37:
                publish_staged = False
                if local_receipt_path.exists():
                    candidate = json.loads(local_receipt_path.read_text())
                    if (candidate.get("output_channel_count") == 37
                            and candidate["source_revision"] == REVISION
                            and candidate["retained_source_indices"] == source_indices.tolist()):
                        if candidate["output_bytes"] == path.stat().st_size:
                            receipt = candidate
                        elif staged.exists() and candidate["output_bytes"] == staged.stat().st_size:
                            with tifffile.TiffFile(staged) as tif:
                                names = [channel.attrib["Name"] for channel in ET.fromstring(tif.ome_metadata).findall(".//{*}Channel")]
                                assert names == kept.marker_id.tolist() and tif.series[0].shape == (37, record.height_px, record.width_px)
                            receipt = candidate
                            publish_staged = True
                if receipt.get("output_channel_count") != 37:
                    assert receipt["source_revision"] == REVISION and receipt["output_bytes"] == path.stat().st_size
                    local_source.parent.mkdir(parents=True, exist_ok=True)
                    with io_deadline(), formal_read_lease(path), resource_lock(OUTPUT, shared=True), resource_lock(path.parent, name=".input.lock", shared=True):
                        shutil.copyfile(path, local_source)
                        assert local_source.stat().st_size == receipt["output_bytes"]
                    rewrite_native_tiles(local_source, staged, panel)
                    local_source.unlink()
                    receipt = dict(receipt, output_bytes=staged.stat().st_size,
                        output_channel_count=37, retained_source_indices=source_indices.tolist(),
                        pixel_checks="Prior 39-channel native receipt; retained compressed tiles copied; all 37-channel center/boundary pixels exactly checked")
                    save_json(local_receipt_path, receipt)
                    publish_staged = True
                if publish_staged:
                    publish_closed_tiff(staged, path)
            assert receipt["output_bytes"] == path.stat().st_size
            save_json(receipt_path, receipt)
            if staged.exists():
                staged.unlink()
            if local_receipt_path.exists():
                local_receipt_path.unlink()
            completed += 1
            progress("correcting_panel", images, corrected_images=completed,
                last_completed_image=image_id, scientific_repair_round=0)
            if completed % 25 == 0:
                print(json.dumps(dict(corrected_images=completed, last_image=image_id)), flush=True)
            if args.limit is not None and completed >= args.limit and completed < len(images):
                progress("correcting_panel_partial", images, corrected_images=completed,
                    last_completed_image=image_id, scientific_repair_round=0)
                return
        except IOWaitTimeout as error:
            progress("waiting_io", images, status="waiting_io", corrected_images=completed,
                last_completed_image=images.iloc[completed - 1].image_id,
                current_image=image_id, reason=str(error),
                closed_staged_input=str(staged) if staged.exists() and local_receipt_path.exists() else "",
                staged_receipt=str(local_receipt_path) if local_receipt_path.exists() else "",
                resume="Run reconcile again; completed receipts and closed local TIFF are reused",
                scientific_repair_round=0)
            print(json.dumps(dict(status="waiting_io", corrected_images=completed,
                current_image=image_id, reason=str(error))), flush=True)
            return
    corrected = images.copy()
    corrected["channel_count"] = 37
    try:
        publish_tables(corrected, samples, panel, field_mapping)
        write_reviews(corrected, json.loads((EVIDENCE / "source_inventory.json").read_text()), complete=False)
    except IOWaitTimeout as error:
        progress("waiting_io", images, status="waiting_io", corrected_images=completed,
            current_image="study_tables", reason=str(error), resume="Run reconcile again to publish corrected tables",
            scientific_repair_round=0)
        return
    progress("panel_reconciled", corrected, corrected_images=completed, scientific_repair_round=0)


def write_reviews(images: pd.DataFrame, inventory: dict, *, complete: bool) -> None:
    rows = []
    def add(check: str, result: str, details: str | dict, scope: str = "study", evidence: str = "") -> None:
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope, record_id=STUDY, check_id=check, result=result,
            source_evidence=evidence or relative(EVIDENCE / "source_inventory.json"), output_evidence=relative(OUTPUT),
            details=json.dumps(details, ensure_ascii=False) if isinstance(details, dict) else details))
    add("S01", "pass" if complete else "unresolved", f"Preparer self-check; {len(images)}/1258 exported; native proofs reused; no identity-based image deletion; full-image downstream nonzero admission")
    add("M01", "unresolved", dict(image_ids=inventory["unresolved_image_ids"], image_sample_connections=int(images.sample_id.notna().sum()),
        source_sample_rows=356, patient_linked_samples=356, distinct_patients=112,
        evidence="SPORA tissues + exact author fov→Tissue_ID→Patient_ID join; five author metadata CSVs in both Zenodo releases, seven original BioStudies file lists, author repository, Nature supplementary tables 1/3/4 reviewed; see metadata_trace.json",
        gap="309 ROI lack specimen and patient identity in all reviewed public maps; SPORA Unknown bucket is not a patient"), "sample")
    add("M02", "pass", "Preparer source reconstruction; all 356 specimen identities and 112 patients match author core/timepoint/patient tables and overlapping SPORA fields; individual clinical outcomes not provided", "sample")
    if complete:
        for check in ["I01", "I02", "I03", "P01", "P02", "P03", "I05"]:
            add(check, "pass", "Preparer self-check only; see source_inventory.json, per-image receipts and self_validation.json; independent review pending")
        add("I04", "pass", "Independent published acquisition ROI, no multi-sample splitting; author identity mapped when present; source region annotations retained; overview montage inspected by preparer")
    add("D01", "unresolved", dict(image_ids=images.image_id.tolist(), note="Preparer delivery; independent reviewer/controller acceptance pending; image and metadata coverage reported separately"))
    staged = LOCAL / "review_results.csv"
    staged.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(staged, index=False, encoding="utf-8-sig")
    temporary = OUTPUT / "review_results.csv.tmp"
    with io_deadline(), formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        shutil.copyfile(staged, temporary)
        os.replace(temporary, OUTPUT / "review_results.csv")


def source_overview(tissues: pd.DataFrame) -> None:
    """Inspect both actual ROI sizes and the separate unknown-identity scope."""
    from PIL import Image, ImageDraw
    selected = []
    for size in [1229, 614]:
        for known in [True, False]:
            for _, record in tissues[tissues.sample_id.notna() == known].iterrows():
                path = RAW / f"mibi/1_0mpp/images/{record.tissue_id}.ome.zarr/0"
                if json.loads((path / "zarr.json").read_text())["shape"][-1] == size:
                    selected.append((record.tissue_id, known, size))
                    break
    canvas = Image.new("RGB", (320*len(selected), 375), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (image_id, known, size) in enumerate(selected):
        path = RAW / f"mibi/1_0mpp/images/{image_id}.ome.zarr/0"
        spec = SourceSpec(image_id, PANEL_ID, "zarr", path, tuple(range(39)))
        with formal_read_lease(path):
            with open_source(spec) as reader:
                planes = reader.read_window([8, 20, 25, 26], (0, 0, size, size))
        rgb = []
        for values in [planes[0], planes[1], planes[2]+planes[3]]:
            positive = values[values > 0]
            scale = np.percentile(positive, 99.5) if len(positive) else 1.0
            rgb.append(np.clip(values/scale, 0, 1))
        tile = Image.fromarray((np.stack(rgb, axis=-1)*255).astype("uint8")).resize((320, 320))
        canvas.paste(tile, (320*column, 0))
        draw.text((320*column+4, 325), image_id.replace("greenwald2026temporal_", "")+f"\n{size}px specimen {'known' if known else 'unknown'}", fill="black")
    canvas.save(CACHE / "native_source_overviews.png")
    save_json(CACHE / "overview_selection.json", dict(selected=selected, display_only="R=CD45, G=ECAD, B=sum(H3K27me3,H3K9ac); each display scaled by positive-pixelP99.5; no source modification"))
    print(json.dumps(dict(overview=str(CACHE / "native_source_overviews.png"), selected=selected)), flush=True)


def check_formal_header(record: dict, previous: dict | None, names: list[str]) -> tuple[str, dict]:
    """Check one TIFF/source version with its own shared NAS read admission."""
    image_id = record["image_id"]
    path = NAS / record["input_image_path"]
    with formal_read_lease(path), resource_lock(OUTPUT, shared=True), resource_lock(path.parent, name=".input.lock", shared=True):
        receipt = json.loads((RECEIPTS / f"{image_id}.json").read_text())
        assert receipt["admitted"] and receipt["source_revision"] == REVISION
        for member in receipt["source_objects"]:
            stat = (NAS / member["path"]).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (member["size"], member["mtime_ns"]), f"Source changed: {member['path']}"
        stat = path.stat()
        version = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns, inode=stat.st_ino)
        assert version["size"] == receipt["output_bytes"]
        shape = [37, record["height_px"], record["width_px"]]
        if (previous is not None and previous["version"] == version
                and previous.get("shape") == shape and previous.get("channel_names") == names):
            return image_id, previous
        with tifffile.TiffFile(path) as tif:
            assert list(tif.series[0].shape) == shape
            assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("float32")
            pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
            assert [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")] == names
            assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0
        after = path.stat()
        assert version == dict(size=after.st_size, mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns, inode=after.st_ino)
    return image_id, dict(version=version, shape=shape, channel_names=names,
        source_objects_checked=len(receipt["source_objects"]), channels=37, dtype="float32")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=["metadata", "overview", "prefetch", "reconcile", "export", "verify", "checkpoint", "headers"])
parser.add_argument("--limit", type=int)
parser.add_argument("--job-dir", type=Path, default=JOB)
parser.add_argument("--resume-progress", type=Path)
parser.add_argument("--io-wait-seconds", type=float, default=60.0)
args = parser.parse_args()
if args.io_wait_seconds <= 0:
    parser.error("--io-wait-seconds must be positive")
JOB = args.job_dir
PREPARER = f"codex_{JOB.name}"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)
RECEIPTS.mkdir(parents=True, exist_ok=True)
zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
blosc.set_nthreads(1)
try:
    with io_deadline(), formal_read_lease(OUTPUT):
        tissues, samples, panel, field_mapping = source_metadata()
        base_columns = pd.read_csv(TEMPLATES / "image_metadata.csv").columns
        images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=base_columns)
except IOWaitTimeout as error:
    checkpoint = json.loads((JOB / "progress.json").read_text())
    checkpoint.update(stage="waiting_io", status="waiting_io", reason=str(error),
        current_image="study_metadata_read", resume="Repeat the same mode and job directory; existing image checkpoints remain valid",
        scientific_repair_round=0)
    save_json(JOB / "progress.json", checkpoint)
    print(json.dumps(dict(status="waiting_io", phase="study_metadata_read")), flush=True)
    raise SystemExit(0)

if args.mode == "metadata":
    progress("checking_source_inventory", images)
    inventory = inspect_sources(tissues, panel)
    publish_tables(images, samples, panel, field_mapping)
    write_reviews(images, inventory, complete=False)
    progress("metadata_ready", images, source_inventory=relative(EVIDENCE / "source_inventory.json"))
    print(json.dumps({k: v for k, v in inventory.items() if k != "unresolved_image_ids"}, ensure_ascii=False), flush=True)

elif args.mode == "overview":
    source_overview(tissues)

elif args.mode == "prefetch":
    prefetch_sources(tissues)

elif args.mode == "reconcile":
    reconcile_existing_panel(images, panel)

elif args.mode == "checkpoint":
    checkpoint = json.loads((JOB / "progress.json").read_text())
    inventory = json.loads((EVIDENCE / "source_inventory.json").read_text())
    try:
        with io_deadline(), formal_read_lease(OUTPUT):
            images = pd.read_parquet(OUTPUT / "images.parquet")
            delivered_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
            tiffs = {path.parent.name for path in (OUTPUT / "images").glob("*/input.tiff")}
            receipts = {path.stem for path in RECEIPTS.glob("*.json")}
            local_receipts = {path.stem for path in (LOCAL / "receipts").glob("*.json")}
            indexed = set(images.image_id)
            unindexed = tiffs - indexed
            assert receipts <= tiffs and indexed <= receipts
            assert unindexed <= {checkpoint.get("current_image")} & local_receipts
            assert images.image_id.is_unique and len(images.columns) == 39
            assert set(images.sample_id.dropna()) <= set(delivered_samples.sample_id)
    except IOWaitTimeout as error:
        checkpoint.update(stage="waiting_io", status="waiting_io", checkpoint_read_wait=str(error),
            review_results_refresh="deferred until shared I/O recovers")
        save_json(JOB / "progress.json", checkpoint)
        print(json.dumps(checkpoint, ensure_ascii=False), flush=True)
        raise SystemExit(0)
    coverage = dict(exported_images=len(images), tiff_files=len(tiffs), receipt_files=len(receipts),
        recoverable_unindexed_images=sorted(unindexed),
        sample_rows=len(delivered_samples), sample_linked_images=int(images.sample_id.notna().sum()),
        patient_linked_samples=int(delivered_samples.patient_key.ne("").sum()),
        patients=int(delivered_samples.patient_key.nunique()),
        exported_sample_rows=int(images.sample_id.nunique()),
        exported_patients=int(delivered_samples.loc[delivered_samples.sample_id.isin(images.sample_id), "patient_key"].nunique()),
        validation="Source-derived identity assertions and file/index/receipt/foreign-key counts; per-export local TIFF headers and all-channel center/boundary checks retained in receipts; additional full NAS-header reread interrupted in shared lease exit; independent review pending")
    save_json(JOB / "closed_delivery.json", coverage)
    try:
        write_reviews(images, inventory, complete=len(images) == 1258)
        coverage["review_results_refreshed"] = True
    except IOWaitTimeout as error:
        coverage["review_results_refreshed"] = False
        coverage["review_publication_wait"] = str(error)
        coverage["staged_review_results"] = str(LOCAL / "review_results.csv")
    save_json(JOB / "closed_delivery.json", coverage)
    waiting = {}
    if checkpoint.get("status") == "waiting_io" and len(images) < 1258:
        waiting = {key: checkpoint[key] for key in (
            "status", "current_image", "reason", "closed_staged_input", "staged_receipt",
            "resume", "scientific_repair_round", "corrected_images",
            "last_completed_image") if key in checkpoint}
        if (waiting.get("current_image") in indexed
                and waiting.get("resume", "").startswith("Run export again")):
            waiting["last_completed_image"] = waiting["current_image"]
            waiting["current_image"] = tissues.loc[~tissues.tissue_id.isin(indexed), "tissue_id"].iloc[0]
            waiting.pop("closed_staged_input", None)
            waiting.pop("staged_receipt", None)
    stage = "waiting_io" if waiting else "export_complete" if len(images) == 1258 else "export_partial"
    progress(stage, images, evidence=relative(JOB / "closed_delivery.json"), **waiting)
    print(json.dumps(coverage, ensure_ascii=False), flush=True)

elif args.mode == "export":
    inventory = json.loads((EVIDENCE / "source_inventory.json").read_text())
    assert not len(images) or (images.channel_count == 37).all(), "Run reconcile before further export"
    pending = tissues[~tissues.tissue_id.isin(images.image_id)]
    if args.limit is not None:
        pending = pending.iloc[:args.limit]
    progress("exporting", images, pending_images=len(pending))
    waiting_io = False
    for _, record in pending.iterrows():
        if shutil.disk_usage(NAS).free < 100*1024**3 + 256*1024**2 or shutil.disk_usage("/tmp").free < 64*1024**3:
            progress("waiting_io", images, status="waiting_io", current_image=record.tissue_id,
                reason="Insufficient reserved capacity for next ROI")
            waiting_io = True
            break
        started = time.monotonic()
        try:
            recovered = recover_unindexed_export(record, panel)
            row, receipt = recovered if recovered is not None else export_roi(record, panel)
            save_json(RECEIPTS / f"{record.tissue_id}.json", receipt)
            if receipt["admitted"]:
                updated_images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
                publish_tables(updated_images, samples, panel, field_mapping)
                images = updated_images
            cached = LOCAL / f"{record.tissue_id}.ome.zarr"
            if cached.exists():
                shutil.rmtree(cached)
            staged = LOCAL / "output_staging" / record.tissue_id
            if staged.exists():
                shutil.rmtree(staged)
            local_receipt = LOCAL / f"receipts/{record.tissue_id}.json"
            if local_receipt.exists():
                local_receipt.unlink()
            progress("exporting", images, last_completed_image=record.tissue_id,
                last_image_seconds=round(time.monotonic()-started, 2))
            print(json.dumps(dict(image_id=record.tissue_id, exported=len(images), admitted=receipt["admitted"], seconds=round(time.monotonic()-started, 2))), flush=True)
        except IOWaitTimeout as error:
            # Reconcile the atomic index if the timeout occurred during its
            # helper publication, after a TIFF had already been committed.
            reconciliation_pending = False
            try:
                with io_deadline(), formal_read_lease(OUTPUT):
                    images = pd.read_parquet(OUTPUT / "images.parquet")
            except IOWaitTimeout:
                reconciliation_pending = True
            progress("waiting_io", images, status="waiting_io", current_image=record.tissue_id,
                reason=str(error), closed_staged_input=str(LOCAL / "output_staging" / record.tissue_id / "input.tiff"),
                staged_receipt=str(LOCAL / f"receipts/{record.tissue_id}.json"),
                resume="Run export again; indexed images and closed TIFF receipts are reused", scientific_repair_round=0,
                last_completed_image=images.iloc[-1].image_id if len(images) else "",
                index_reconciliation_pending=reconciliation_pending)
            waiting_io = True
            print(json.dumps(dict(status="waiting_io", exported_images=len(images), current_image=record.tissue_id, reason=str(error))), flush=True)
            break
        except Exception as error:
            progress("export_error", images, current_image=record.tissue_id, error=f"{type(error).__name__}: {error}")
            raise
    if not waiting_io:
        write_reviews(images, inventory, complete=len(images) == 1258)
        progress("export_complete" if len(images) == 1258 else "export_partial", images)

elif args.mode == "headers":
    # Resume formal-header/source-version checks without decoding or rewriting
    # the 1258 already delivered native TIFFs. A timeout must escape tifffile's
    # Exception handlers so an interrupted parse cannot become a false failure.
    checkpoint_path = JOB / "header_checks.json"
    checked = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    expected_names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
    progress("checking_formal_headers", images, headers_checked=len(checked), scientific_repair_round=0)
    for offset in range(0, len(images), 4):
        batch = images.iloc[offset:offset+4].to_dict("records")
        try:
            # Each worker acquires its own NAS read lease before study/input
            # locks. The parent joins/terminates all workers before checkpointing.
            with io_deadline():
                with multiprocessing.get_context("fork").Pool(processes=4) as workers:
                    results = workers.starmap(check_formal_header,
                        [(row, checked.get(row["image_id"]), expected_names) for row in batch])
            checked.update(results)
            save_json(checkpoint_path, checked)
            progress("checking_formal_headers", images, headers_checked=len(checked),
                last_completed_image=batch[-1]["image_id"], scientific_repair_round=0)
            print(json.dumps(dict(headers_checked=len(checked), total=len(images))), flush=True)
        except IOWaitTimeout as error:
            progress("waiting_io", images, status="waiting_io", headers_checked=len(checked), current_image=batch[0]["image_id"],
                pending_header_image_ids=[row["image_id"] for row in batch],
                reason=str(error), resume="Run headers with this job directory; reuse only unchanged checked TIFF versions; no export remains", scientific_repair_round=0)
            print(json.dumps(dict(status="waiting_io", headers_checked=len(checked), current_image=batch[0]["image_id"])), flush=True)
            break
    else:
        assert set(checked) == set(images.image_id)
        progress("prepared_awaiting_independent_review", images, headers_checked=len(checked), scientific_repair_round=0,
            evidence=relative(checkpoint_path), independent_review="pending")
        print(json.dumps(dict(status="prepared", headers_checked=len(checked), independent_review="pending")), flush=True)

elif args.mode == "verify":
    inventory = json.loads((EVIDENCE / "source_inventory.json").read_text())
    checkpoint = json.loads((JOB / "progress.json").read_text()) if (JOB / "progress.json").exists() else {}
    waiting_checkpoint = {key: checkpoint[key] for key in (
        "status", "current_image", "reason", "closed_staged_input", "staged_receipt", "resume", "scientific_repair_round") if key in checkpoint} if checkpoint.get("status") == "waiting_io" else {}
    progress("verifying", images, **waiting_checkpoint)
    from PIL import Image, ImageDraw
    overview_ids = []
    shapes = set()
    for _, record in images.iterrows():
        path = NAS / record.input_image_path
        with formal_read_lease(path), resource_lock(OUTPUT, shared=True), resource_lock(path.parent, name=".input.lock", shared=True), tifffile.TiffFile(path) as tif:
            assert tif.series[0].shape == (37, record.height_px, record.width_px)
            assert tif.series[0].axes == "CYX" and tif.series[0].dtype == np.dtype("float32")
            ome = ET.fromstring(tif.ome_metadata)
            pixels = ome.find(".//{*}Pixels")
            assert [channel.attrib["Name"] for channel in pixels.findall("{*}Channel")] == panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
            assert float(pixels.attrib["PhysicalSizeX"]) == float(pixels.attrib["PhysicalSizeY"]) == 1.0
            receipt = json.loads((RECEIPTS / f"{record.image_id}.json").read_text())
            assert receipt["admitted"] and receipt["output_bytes"] == path.stat().st_size
            assert receipt["source_revision"] == REVISION
            shape = (record.height_px, record.width_px)
            if shape not in shapes or (pd.isna(record.sample_id) and not any(x[1] == "unknown" for x in overview_ids)):
                overview_ids.append((record.image_id, "unknown" if pd.isna(record.sample_id) else "known"))
                shapes.add(shape)
    # A compact overview uses the source-selected histones and two body markers.
    canvas = Image.new("RGB", (320*len(overview_ids), 370), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (image_id, state) in enumerate(overview_ids):
        path = OUTPUT / f"images/{image_id}/input.tiff"
        with formal_read_lease(path), resource_lock(OUTPUT, shared=True), resource_lock(path.parent, name=".input.lock", shared=True), tifffile.TiffFile(path) as tif:
            planes = [tif.pages[channel].asarray(maxworkers=2) for channel in [8, 20, 28]]
        groups = planes
        rgb = []
        for values in groups:
            positive = values[values > 0]
            scale = np.percentile(positive, 99.5) if len(positive) else 1.0
            rgb.append(np.clip(values/scale, 0, 1))
        tile = Image.fromarray((np.stack(rgb, axis=-1)*255).astype("uint8")).resize((320, 320))
        canvas.paste(tile, (column*320, 0))
        draw.text((column*320+4, 324), image_id.replace("greenwald2026temporal_", "")+"\n"+state, fill="black")
    canvas.save(CACHE / "source_overviews.png")
    result = dict(images=len(images), headers_checked=len(images), channels=37, dtype="float32", sample_rows=len(samples),
        sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=len(samples), patients=samples.patient_key.nunique(),
        exported_sample_rows=int(images.sample_id.nunique()),
        exported_patients=int(samples.loc[samples.sample_id.isin(images.sample_id), "patient_key"].nunique()),
        pixel_checks="Every exported ROI: all37 retained channels, center32x32 and top-left32x32 at export or panel reconciliation; no full-output pixel scan",
        source_shapes=inventory["shapes"], overview_image_ids=overview_ids, preparer=PREPARER, independent_review="pending")
    # Refresh source-derived clinical annotations after export without rewriting
    # any TIFF; all identities and actual channel headers have been checked above.
    publish_tables(images, samples, panel, field_mapping)
    save_json(EVIDENCE / "self_validation.json", result)
    write_reviews(images, inventory, complete=len(images) == 1258)
    progress("waiting_io" if waiting_checkpoint else "prepared_awaiting_independent_review" if len(images) == 1258 else "partial_awaiting_independent_review", images,
        evidence=relative(EVIDENCE / "self_validation.json"), **waiting_checkpoint)
    print(json.dumps(result, ensure_ascii=False), flush=True)
