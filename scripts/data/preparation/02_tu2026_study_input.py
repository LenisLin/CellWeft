"""Prepare the confirmed CODEX cores from the Tu 2026 PhenoCycler study.

The public release contains three QPTIFF TMA scans.  The author measurement
tables provide the Parent label and native micrometre coordinates for each
core.  The clinical crosswalk selects the 29 CODEX=Y cores; the four CODEX=N
Parent labels remain in the source evidence and are not exported.

This adapter keeps the QPTIFFs untouched, reads only the selected core windows
from full-resolution pages, and publishes native uint8 input TIFFs plus the
study input tables.  It does not perform normalization, segmentation,
quantification, or embedding lookup.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative, publish_metadata_sources

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from xml.etree import ElementTree

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/study_input_batch/Tu2026_PhenoCycler_Neuroblastoma/matplotlib")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables, validate_image_metadata


STUDY = "Tu2026_PhenoCycler_Neuroblastoma"
NAS_ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE_ROOT = NAS_ROOT / "raw/public" / STUDY / "images"
SOURCE_METADATA = metadata_source_directory(NAS_ROOT, STUDY)
RAW_METADATA = NAS_ROOT / "raw/public" / STUDY / "metadata"
OUTPUT_ROOT = NAS_ROOT / "processed" / STUDY
FACT_ROOT = SOURCE_METADATA / "input_facts"
EVIDENCE_ROOT = OUTPUT_ROOT / "preparation_evidence"
S1_PATH = NAS_ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
PANEL_ID = "Tu2026_PhenoCycler_native54"
SOURCE_REVISION = "Zenodo11444751v1"
# All 162 full-resolution source IFDs report this upstream processing flag.
INTENSITY_DESCRIPTION = (
    "Native uint8 QPTIFF full-resolution fluorescence; source AutofluorescenceSubtracted=True "
    "(upstream autofluorescence subtraction); CellWeft preserves decoded source intensities "
    "without additional normalization"
)
TEMP_ROOT = Path("/tmp/study_input_batch") / STUDY
REGION_ROOT = TEMP_ROOT / "sample_regions"
PROGRESS_PATH = (Path("/home/lenislin/Experiment/projects/CellWeft/.cache/cellweft/runtime/study_input") /
                 STUDY / "progress.json")
PREPARER = "codex_cli_tu2026_0001_prepare_0"


def region_path(parent: str) -> Path:
    """Resolve the formal published region or the current local candidate."""
    published = OUTPUT_ROOT / "images" / parent / "sample_region.tiff"
    candidate = REGION_ROOT / f"{parent}.tiff"
    if published.is_file():
        return published
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No published or staged sample region for {parent}")


def visual_gap_region(parent: str, window: tuple[int, int, int, int]) -> np.ndarray:
    """Keep the target side of gaps seen in 1/16-resolution native DAPI.

    D3 includes an unlabelled inter-core fragment. E3 and E4 include tissue
    from unlabelled neighbouring cores. These fragments are not assigned to a
    biological specimen without a source label. The piecewise lines lie in visible
    inter-core background; they are not tissue contours. Coordinates and the
    DAPI/author-point overlays are retained in sample_regions/*native_detail.png.
    """
    x0,y0,x1,y1 = window
    yy,xx = np.ogrid[y0:y1,x0:x1]
    if parent == "TMA1_D3":
        return yy >= 18800 - (xx-21484) * (1660/2416)
    if parent == "TMA5_E3":
        threshold = np.interp(yy, [26408,27300,27750,27800,27950,30111], [30900,30850,30550,30433,30233,30233])
        return xx >= threshold
    if parent == "TMA5_E4":
        threshold = np.interp(yy, [32371,35500,36000,36600,37400,38000,39359],
                              [29346,29346,29400,29500,29750,30000,30000])
        return xx >= threshold
    return np.ones((y1-y0,x1-x0),dtype=bool)


def extend_zero_margin_frames(bounds: pd.DataFrame) -> pd.DataFrame:
    """Restore background margins now that neighbouring samples have a separate region.

    Earlier rectangle-only proposals used zero margins at eight neighbouring
    edges. Restore 100 µm there; keep the confirmed C1 pixels and frame intact.
    The E3 sparse tail and the E4 physical scan edge are source-specific fixes.
    """
    bounds = bounds.copy()
    for parent in bounds.index:
        filename = bounds.loc[parent,"qptiff"]
        shape,scale,_ = qptiff_metadata(SOURCE_ROOT/filename)
        for edge,direction in [("left","x0"),("top","y0"),("right","x1"),("bottom","y1")]:
            if float(bounds.loc[parent,f"margin_{edge}_um"]) == 0:
                key = f"source_{direction}_px"
                offset = int(np.ceil(100*scale))
                value = int(bounds.loc[parent,key]) + (-offset if direction.endswith("0") else offset)
                limit = shape[1] if direction.startswith("x") else shape[0]
                bounds.loc[parent,key] = min(limit,max(0,value))
                bounds.loc[parent,f"margin_{edge}_um"] = 100.0
                bounds.loc[parent,"notes"] += f" Restored {edge} background margin with sample_region separating neighbours."
        if parent == "TMA5_E3":
            bounds.loc[parent,"source_y1_px"] = 30111
            bounds.loc[parent,"margin_bottom_um"] = 300.0
        if parent == "TMA5_E4":
            bounds.loc[parent,"source_x1_px"] = shape[1]
        for coordinate in ("x0","y0","x1","y1"):
            bounds.loc[parent,f"source_{coordinate}_um"] = int(bounds.loc[parent,f"source_{coordinate}_px"])/scale
    FACT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = FACT_ROOT / "core_bounds.csv.partial"
    with formal_publish_lease(FACT_ROOT / "core_bounds.csv", directory=FACT_ROOT):
        bounds.to_csv(temporary,index=False,encoding="utf-8-sig")
        os.replace(temporary, FACT_ROOT / "core_bounds.csv")
    return bounds


def progress(stage: str, images: pd.DataFrame, *, progress_path: Path = PROGRESS_PATH, **details) -> None:
    """Record only closed images that are actually present in the study index."""
    write_json(progress_path, dict(study=STUDY, stage=stage, expected_images=29,
               completed_images=len(images), sample_rows=29,
               sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
               updated_at=datetime.now(timezone.utc).isoformat(), **details))


def author_coordinates() -> pd.DataFrame:
    """Keep the original Parent/coordinate evidence; these are cell centroids, not tissue polygons."""
    path = FACT_ROOT / "author_parent_coordinates.parquet"
    if not path.exists():
        columns = ["Image", "Parent", "Centroid X µm", "Centroid Y µm"]
        points = pd.concat([pd.read_csv(RAW_METADATA / name,
                            usecols=columns) for name in
                            ("TMA1_3_measurements.csv", "TMA5_measurements.csv")], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        with formal_publish_lease(path, directory=path.parent):
            points.rename(columns={"Centroid X µm": "x_um", "Centroid Y µm": "y_um"}).to_parquet(temporary, index=False)
            os.replace(temporary, path)
    return pd.read_parquet(path)


def fetch_identity_metadata() -> None:
    """Read just public H5AD identity columns by HTTP Range, never its intensity matrix.

    Repeated author IDs suggest pre/post groups, but the release has no explicit
    patient-ID dictionary and disagrees with the paper's 27-patient count if all
    three repeated groups are treated as patients. Preserve them as evidence;
    do not populate patient_key from an ambiguous ID column.
    """
    import fsspec
    import h5py
    url = "https://zenodo.org/api/records/11444751/files/NB_metadata_merged_CN.h5ad/content"
    with fsspec.open(url,mode="rb",block_size=131072) as remote:
        with h5py.File(remote,"r") as handle:
            values = {}
            fields = list(handle["obs"])
            for key in ("Parent","ID","Treatment","CODEX","Image"):
                value = handle["obs"][key]
                if isinstance(value,h5py.Group):
                    categories,codes = value["categories"].asstr()[:],value["codes"][:]
                    values[key] = np.where(codes>=0,categories[np.maximum(codes,0)],"")
                else:
                    values[key] = value[:]
    frame = pd.DataFrame(values)
    crosswalk = FACT_ROOT / "author_h5ad_identity_crosswalk.csv"
    crosswalk.parent.mkdir(parents=True, exist_ok=True)
    temporary = crosswalk.with_name(crosswalk.name + ".partial")
    with formal_publish_lease(crosswalk, directory=crosswalk.parent):
        frame.value_counts(list(values)).rename("cell_count").reset_index().to_csv(
            temporary,index=False,encoding="utf-8-sig")
        os.replace(temporary, crosswalk)
    write_json(FACT_ROOT/"public_metadata_search.json",dict(
        time=datetime.now(timezone.utc).isoformat(),source_url="https://zenodo.org/records/11444751",
        metadata_url=url,retrieval="HTTP Range; obs identity columns only; matrix not downloaded",
        source_rows=len(frame),obs_fields=fields,
        paper_url="https://doi.org/10.1186/s13073-026-01622-0",
        paper_fulltext_local=f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/pmc13041199_fulltext.xml",
        repository_search_queries=['"Spatial multi-omics characterization of neuroblastoma" github',
                                   '"Spatial multiomics reveals the interplay between neuroblastoma" github'],
        repository_search_result="No author repository linked in paper fulltext; exact-title searches identified no study repository",
        patient_definition="H5AD ID is not explicitly documented as a patient identifier; candidate groups retained, patient_key unresolved"))
    publish_metadata_sources(FACT_ROOT)


def prepare_sample_regions(bounds: pd.DataFrame) -> None:
    """Propose sample processing regions from all 33 original Parent labels.

    Nearest labelled centroid assigns the entire extraction rectangle, including
    background. It is a sample partition, never an inferred tissue boundary or a
    centroid hull. An 8-native-pixel grid is refined at every conflicting labelled
    centroid. The preparer must inspect each DAPI/region overlay before export.
    Source pixels and the independently reviewed C1 frame remain unchanged.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REGION_ROOT.mkdir(parents=True, exist_ok=True)
    points = author_coordinates()
    evidence = []
    for filename, group in bounds.groupby("qptiff"):
        native_shape, scale, _ = qptiff_metadata(SOURCE_ROOT / filename)
        scan = points[points.Image.eq(filename + " - resolution #1")]
        xy = scan[["x_um", "y_um"]].to_numpy() * scale
        labels = scan.Parent.to_numpy()
        tree = cKDTree(xy)
        # The full-resolution channel order is followed by the thumbnail, then
        # 54 pages per reduced level. Use actual level dimensions for XY mapping.
        with tifffile.TiffFile(SOURCE_ROOT / filename) as tif:
            overview = tif.pages[271].asarray(maxworkers=2)
        ratio_y, ratio_x = native_shape[0] / overview.shape[0], native_shape[1] / overview.shape[1]
        figure, axes = plt.subplots((len(group)+2)//3, 3, figsize=(15, 5*((len(group)+2)//3)), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, group.itertuples(index=False)):
            parent = row.parent
            x0, y0, x1, y1 = (int(getattr(row, key)) for key in
                              ("source_x0_px", "source_y0_px", "source_x1_px", "source_y1_px"))
            height, width = y1-y0, x1-x0
            side = max(height, width)
            left, top = (side-width)//2, (side-height)//2
            step = 8
            yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
            _, nearest = tree.query(np.column_stack((xx.ravel()+step/2, yy.ravel()+step/2)), workers=4)
            coarse = (labels[nearest] == parent).reshape(yy.shape)
            region = np.repeat(np.repeat(coarse, step, axis=0), step, axis=1)[:height, :width]
            inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            local = np.floor(xy[inside] - [x0, y0]).astype(int)
            target = labels[inside] == parent
            conflicts = region[local[:, 1], local[:, 0]] != target
            for px, py in local[conflicts]:
                a, b, c, d = max(0,px-step), max(0,py-step), min(width,px+step+1), min(height,py+step+1)
                ry, rx = np.mgrid[b:d, a:c]
                _, nearest = tree.query(np.column_stack((rx.ravel()+x0+.5, ry.ravel()+y0+.5)), workers=4)
                region[b:d, a:c] = (labels[nearest] == parent).reshape(ry.shape)
            region &= visual_gap_region(parent, (x0,y0,x1,y1))
            included = region[local[:, 1], local[:, 0]]
            own_excluded = int(np.count_nonzero(target & ~included))
            neighbour_included = int(np.count_nonzero(~target & included))
            if own_excluded or neighbour_included:
                raise ValueError(f"Centroid partition conflict for {parent}: {own_excluded}, {neighbour_included}")
            padded = np.zeros((side, side), dtype=np.uint8)
            padded[top:top+height, left:left+width] = region
            tifffile.imwrite(REGION_ROOT / f"{parent}.tiff", padded, compression="deflate", maxworkers=2,
                             photometric="minisblack", metadata={"axes":"YX"})
            # Include context outside the extraction frame so visible tissue at
            # the edge and every neighbouring core can be assessed visually.
            a,b = max(0,int(x0/ratio_x)-22),max(0,int(y0/ratio_y)-22)
            c,d = min(overview.shape[1],int(np.ceil(x1/ratio_x))+22),min(overview.shape[0],int(np.ceil(y1/ratio_y))+22)
            context = overview[b:d,a:c]
            ax.imshow(context, cmap="gray", vmin=0, vmax=max(1,float(np.percentile(context,99.5))),
                      extent=(a*ratio_x,c*ratio_x,d*ratio_y,b*ratio_y))
            ax.contour(region[::8,::8], levels=[.5], colors=["lime"], linewidths=1,
                       extent=(x0,x1,y1,y0), origin="upper") if not region.all() else None
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((x0,y0),width,height,fill=False,edgecolor="cyan",linewidth=1))
            ax.set_title(f"{parent}: cyan=frame; green=sample partition")
            evidence.append(dict(parent=parent, qptiff=filename, own_centroids_in_frame=int(target.sum()),
                                 other_centroids_in_frame=int((~target).sum()), own_centroids_excluded=own_excluded,
                                 other_centroids_allowed=neighbour_included,
                                 allowed_pixels=int(region.sum()), frame_pixels=height*width,
                                 candidate_path=metadata_source_relative(OUTPUT_ROOT / 'images' / parent / 'sample_region.tiff', NAS_ROOT).as_posix(),
                                 review_status="pending_visual_review"))
            print(f"region candidate {parent}: own={target.sum()}, neighbours={(~target).sum()}",flush=True)
        figure.tight_layout()
        figure.savefig(REGION_ROOT / f"{Path(filename).stem}_region_review.png",dpi=130)
        plt.close(figure)
    write_json(REGION_ROOT / "region_evidence.json", dict(
        method="nearest-author-Parent processing partition; 8-native-pixel grid with local refinement; three visually reviewed gap constraints for unlabelled fragments; not a tissue boundary",
        source="raw/public/"+STUDY+"/metadata", records=evidence))


def text_element(root: ElementTree.Element, name: str) -> str:
    for element in root.iter():
        if element.tag.split("}")[-1] == name:
            return (element.text or "").strip()
    return ""


def qptiff_metadata(path: Path) -> tuple[tuple[int, int], float, str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "CYX" or series.shape[0] != 54 or series.dtype != np.dtype("uint8"):
            raise ValueError(f"Unexpected full-resolution layout for {path.name}: {series.axes} {series.shape} {series.dtype}")
        description = tif.pages[0].description
    root = ElementTree.fromstring(description.encode("utf-16"))
    scale = float(text_element(root, "ScaleFactor"))
    slide_id = text_element(root, "SlideID")
    if scale <= 0 or not slide_id:
        raise ValueError(f"Missing QPTIFF scale or slide ID: {path.name}")
    return tuple(int(x) for x in series.shape[-2:]), scale, slide_id


def native_channel_names() -> list[str]:
    records = json.loads((SOURCE_METADATA / "native_markers.json").read_text())
    panels = records["source_scan_panels"]
    names = panels[0]["ordered_source_labels"]
    if len(names) != 54 or any(panel["ordered_source_labels"] != names for panel in panels):
        raise ValueError("The three QPTIFFs do not share the confirmed 54-channel order")
    return names


def core_records() -> dict[str, dict]:
    summary = json.loads((SOURCE_METADATA / "original_author_core_label_summary.json").read_text())
    records = {}
    for scan in summary["source_qptiff_records"]:
        for record in scan["core_records"]:
            if record["parent"] in records:
                raise ValueError(f"Duplicate Parent label: {record['parent']}")
            records[record["parent"]] = {**record, "qptiff": scan["qptiff"]}
    if len(records) != 33:
        raise ValueError(f"Expected 33 exact Parent labels, found {len(records)}")
    return records


def column(frame: pd.DataFrame, *names: str) -> str:
    normalized = {str(name).strip().lower(): name for name in frame.columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    raise ValueError(f"Clinical crosswalk needs one of {names}; found {list(frame.columns)}")


def clinical_crosswalk() -> tuple[pd.DataFrame, dict[str, str]]:
    path = FACT_ROOT / "clinical_source_table.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing controller-confirmed clinical crosswalk: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    parent_key = column(frame, "Parent", "parent", "core_id")
    codex_key = column(frame, "CODEX", "codex", "codex_status")
    sample_key = column(frame, "sample_id", "sample")
    eligible = frame[frame[codex_key].str.strip().str.upper().eq("Y")].copy()
    if eligible[parent_key].duplicated().any() or eligible[sample_key].duplicated().any():
        raise ValueError("CODEX=Y crosswalk must have unique Parent and sample_id")
    if eligible[sample_key].eq("").any():
        raise ValueError("Every CODEX=Y core must have a biological sample_id")
    mapping = dict(zip(eligible[parent_key], eligible[sample_key]))
    expected = {parent for parent, record in core_records().items() if parent in mapping}
    if len(mapping) != 29:
        raise ValueError(f"Expected 29 CODEX=Y core rows, found {len(mapping)}")
    if len(expected) != 29:
        raise ValueError("Clinical crosswalk contains Parent labels absent from author core metadata")
    return frame, mapping


def sample_table() -> pd.DataFrame:
    path = FACT_ROOT / "sample_metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing controller-confirmed sample table: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "sample_id" not in frame.columns or frame.sample_id.duplicated().any():
        raise ValueError("sample_metadata.csv needs unique sample_id values")
    return frame


def reviewed_core_bounds() -> pd.DataFrame:
    path = FACT_ROOT / "core_bounds.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reviewed core bounds: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "parent", "qptiff", "source_x0_px", "source_y0_px",
        "source_x1_px", "source_y1_px", "review_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"core_bounds.csv is missing columns: {sorted(missing)}")
    if frame["parent"].duplicated().any():
        raise ValueError("core_bounds.csv must have one row per Parent")
    expected = set(clinical_crosswalk()[1])
    if set(frame["parent"]) != expected:
        raise ValueError("core_bounds.csv must contain exactly the 29 CODEX=Y Parent labels")
    records = core_records()
    if any(records[parent]["qptiff"] != qptiff for parent, qptiff in zip(frame["parent"], frame["qptiff"])):
        raise ValueError("core_bounds.csv qptiff bindings disagree with author core metadata")
    for name in ("source_x0_px", "source_y0_px", "source_x1_px", "source_y1_px"):
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} must contain integer native pixels")
        frame[name] = values.astype("int64")
    if frame["review_status"].eq("").any():
        raise ValueError("Every core_bounds row needs a review_status")
    return frame.set_index("parent", drop=False)


def field_mapping() -> pd.DataFrame:
    template = pd.read_csv(Path(__file__).resolve().parents[3] / "docs/data/templates/study_input/field_mapping.csv")
    rows = [
        dict(study=STUDY, source_file=";".join(f"raw/public/{STUDY}/images/{name}" for name in
                                              ("Neuro_c1.qptiff","Neuro_c3.qptiff","Neuro_c5.qptiff")),
             source_sheet_or_table="QPTIFF full-resolution pages", source_field="54 scalar pages; ImageDescription",
             target_table="image_metadata", target_field="source_files;source_axes;source_dtype;slide_id;pixel_size_x_um;pixel_size_y_um",
             join_keys="qptiff filename", value_mapping="", transformation_rule="Read full-resolution CYX pages 0-53; retain source uint8 and ScaleFactor-derived micrometre scale",
             time_origin_rule="", conflict_rule="source metadata", notes="Three files are complete according to source_manifest.json"),
        dict(study=STUDY, source_file=";".join(f"raw/public/{STUDY}/images/{name}" for name in
                                              ("Neuro_c1.qptiff","Neuro_c3.qptiff","Neuro_c5.qptiff")),
             source_sheet_or_table="QPTIFF full-resolution IFDs 0-53", source_field="ImageDescription/AutofluorescenceSubtracted",
             target_table="image_metadata", target_field="intensity_description",
             join_keys="source_files -> qptiff filename; all 54 retained source channels", value_mapping='{"True":"upstream autofluorescence subtraction"}',
             transformation_rule="Record source AutofluorescenceSubtracted=True; preserve decoded native uint8 intensities without additional normalization during CellWeft input preparation",
             time_origin_rule="", conflict_rule="Original QPTIFF ImageDescription metadata",
             notes="The source flag describes upstream processing; it does not imply that CellWeft performed autofluorescence subtraction"),
        dict(study=STUDY, source_file=f"raw/public/{STUDY}/metadata.json#public/{STUDY}/original_author_core_label_summary.json",
             source_sheet_or_table="core_records", source_field="qptiff;parent;centroid_x_um;centroid_y_um",
             target_table="image_metadata", target_field="image_id;core_id;region_annotation;source_x0_px;source_y0_px;source_x1_px;source_y1_px",
             join_keys="qptiff + Parent", value_mapping="", transformation_rule="Use the manually reviewed native-pixel frame in core_bounds.csv; export one core window and center-pad to square",
             time_origin_rule="", conflict_rule="source coordinate evidence", notes="Envelope is reviewed against reduced full-frame QPTIFF overview; it is not a cell-derived valid-pixel mask"),
        dict(study=STUDY, source_file=f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/core_bounds.csv",
             source_sheet_or_table="core_bounds", source_field="Parent;source_x0_px;source_y0_px;source_x1_px;source_y1_px;review_status",
             target_table="image_metadata", target_field="source_x0_px;source_y0_px;source_x1_px;source_y1_px;split_method",
             join_keys="Parent", value_mapping="Export review_status=confirmed or preparer_confirmed", transformation_rule="Read visually confirmed native-pixel extraction bounds; retain prior independent C1 verification and send new cores for independent review",
             time_origin_rule="", conflict_rule="reviewed core boundary table", notes="Bounds are the extraction frame, not a signal-derived valid-pixel mask"),
        dict(study=STUDY, source_file="processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv",
             source_sheet_or_table="S1", source_field="Study + Original marker",
             target_table="panel_mapping", target_field="marker_id;mapping_status;role;nuclear;body;keep_channel;output_channel_index",
             join_keys="Study + source_name", value_mapping="", transformation_rule="Map the confirmed native 54-channel order through S1; retain S1-selected DAPI nuclear input and admitted marker channels",
             time_origin_rule="", conflict_rule="S1 source mapping", notes="No embedding lookup is performed in this step"),
        dict(study=STUDY, source_file=f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/clinical_source_table.csv",
             source_sheet_or_table="TMAMap", source_field="Parent;CODEX;sample_id",
             target_table="image_metadata", target_field="sample_id;metadata_notes",
             join_keys="Parent", value_mapping="CODEX=Y only", transformation_rule="Use the controller-confirmed 29-row CODEX crosswalk; retain CODEX=N labels in review evidence only",
             time_origin_rule="", conflict_rule="source clinical crosswalk", notes="The sample table is biological-sampling level; no patient inference from Parent label"),
        dict(study=STUDY, source_file=f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/region_evidence.json",
             source_sheet_or_table="per-core processing region", source_field="Parent;native XY;DAPI overview and detailed gap review",
             target_table="image_metadata", target_field="sample_region_path",
             join_keys="Parent; source window; centered padding", value_mapping="1=target sample allowed region; 0=neighbour or padding",
             transformation_rule="Nearest original Parent-labelled coordinate partition, visually checked on all cores; manual background gap constraints for TMA1_D3/TMA5_E3/TMA5_E4; preserve input.tiff pixels",
             time_origin_rule="", conflict_rule="Keep labelled target cells and complete visible target tissue; exclude adjacent samples",
             notes=json.dumps({"data_type":"string", "definition":"NAS-relative path to the sample processing region"})),
        dict(study=STUDY, source_file=f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/public_metadata_search.json",
             source_sheet_or_table="local author CSVs, paper Supplementary Table S1, Zenodo H5AD obs, repository search",
             source_field="TMAMap;Patient ID;Parent;H5AD ID", target_table="sample_metadata", target_field="patient_key;metadata_notes",
             join_keys="TMAMap=Parent; H5AD Parent", value_mapping="29 confirmed specimen rows; patient_key remains unconfirmed",
             transformation_rule="Preserve published specimen identifiers; save H5AD ID group suggestions separately because its ID field has no explicit patient definition",
             time_origin_rule="Pre/post treatment are distinct biological samplings",
             conflict_rule="Do not reconcile 27 reported patients by guessing pairs from clinical similarities or core names",
             notes="Image/sample coverage is reported separately from unresolved cross-specimen patient identity in M01"),
    ]
    extra = FACT_ROOT / "clinical_field_mapping.csv"
    if extra.is_file():
        extension = pd.read_csv(extra, dtype=str, keep_default_na=False)
        extension = extension.reindex(columns=template.columns, fill_value="")
        rows.extend(extension.to_dict(orient="records"))
    return pd.DataFrame(rows, columns=template.columns)


def build_image_row(parent: str, record: dict, source_path: Path, result: dict,
                    source_shape: tuple[int, int], slide_id: str, scale: float,
                    sample_id: str, clinical_path: str) -> dict:
    source_relative = metadata_source_relative(source_path, NAS_ROOT).as_posix()
    x0, y0, x1, y1 = (int(result.pop(key)) for key in ("source_x0_px", "source_y0_px", "source_x1_px", "source_y1_px"))
    image_id = parent
    return {
        "study": STUDY, "image_id": image_id, "sample_id": sample_id,
        "panel_id": PANEL_ID, "image_kind": "tma_core",
        "source_image_id": f"{source_path.name}:{parent}", "source_files": [source_relative],
        "source_format": "qptiff", "source_acquisition_id": pd.NA,
        "source_z_index": pd.NA, "source_t_index": pd.NA, "source_axes": "CYX",
        "source_dtype": "uint8", "source_revision": SOURCE_REVISION,
        "slide_id": slide_id, "core_id": parent, "roi_id": pd.NA, "fov_id": pd.NA,
        "region_annotation": parent, "source_x0_px": x0, "source_y0_px": y0,
        "source_x1_px": x1, "source_y1_px": y1,
        "pad_left_px": int(result["pad_left_px"]), "pad_top_px": int(result["pad_top_px"]),
        "pad_right_px": int(result["pad_right_px"]), "pad_bottom_px": int(result["pad_bottom_px"]),
        "height_px": int(result["height_px"]), "width_px": int(result["width_px"]),
        "source_channel_count": 54, "channel_count": int(result["channel_count"]),
        "pixel_size_x_um": 1.0 / scale, "pixel_size_y_um": 1.0 / scale,
        "input_image_path": metadata_source_relative(OUTPUT_ROOT / 'images' / image_id / 'input.tiff', NAS_ROOT).as_posix(),
        "valid_pixels_path": pd.NA,
        "intensity_description": INTENSITY_DESCRIPTION,
        "split_method": "reviewed_dearray",
        "split_evidence": f"Original author Parent={parent}; CODEX=Y in {clinical_path}; native bounds recorded in raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/core_bounds.csv",
        "metadata_notes": "Exact native-pixel core frame from reviewed inter-core boundaries; tissue and background are retained within the square output; sample_id joined from controller-confirmed clinical table",
    }


def verify_pixels(source: SourceSpec, path: Path, row: dict, points: pd.DataFrame) -> dict:
    """Check two independent blocks across all 54 channels before publishing a new image."""
    own = points[points.Parent.eq(row["image_id"])][["x_um", "y_um"]].to_numpy()
    scale = 1 / row["pixel_size_x_um"]
    xy = own * scale
    point = xy[np.argmin(np.square(xy-np.median(xy,axis=0)).sum(axis=1))]
    width = row["source_x1_px"]-row["source_x0_px"]
    height = row["source_y1_px"]-row["source_y0_px"]
    px = min(width-32,max(0,int(point[0])-row["source_x0_px"]-16))
    py = min(height-32,max(0,int(point[1])-row["source_y0_px"]-16))
    blocks = [(px,py,px+32,py+32),(0,max(0,height-32),32,height)]
    with open_source(source) as reader, tifffile.TiffFile(path) as tif:
        for index, (a,b,c,d) in enumerate(blocks):
            expected = reader.read_window(list(range(54)),(a,b,c,d))
            actual = np.stack([read_tiff_window(tif,channel,(a+row["pad_left_px"],b+row["pad_top_px"],
                               c+row["pad_left_px"],d+row["pad_top_px"])) for channel in range(54)])
            if expected.dtype != actual.dtype or not np.array_equal(expected,actual):
                raise ValueError(f"Native pixel mismatch: {row['image_id']} block {index}")
            if index == 0 and not np.any(expected[1:]):
                raise ValueError(f"No downstream signal in target-labelled tissue block: {row['image_id']}")
        for channel in range(54):
            if row["pad_top_px"] and np.any(read_tiff_window(tif,channel,(0,0,32,1))):
                raise ValueError("Nonzero top padding")
            if row["pad_left_px"] and np.any(read_tiff_window(tif,channel,(0,0,1,32))):
                raise ValueError("Nonzero left padding")
    return dict(blocks_local_xyxy=blocks, channels=list(range(54)), exact_equal=True,
                target_marker_signal=True, padding_checked=True)


def self_review(images: pd.DataFrame, samples: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Append preparer checks without replacing earlier independent C1 results."""
    columns = ["study","preparer","reviewer","scope","record_id","check_id","result",
               "source_evidence","output_evidence","details"]
    evidence = f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts"
    output = f"processed/{STUDY}"
    rows = []
    def check(scope: str, record: str, key: str, result: str, details: str) -> None:
        rows.append(dict(study=STUDY,preparer=PREPARER,reviewer="",scope=scope,record_id=record,
                         check_id=key,result=result,source_evidence=evidence,output_evidence=output,
                         details="整理者自检；独立复核待主控另行派发。"+details))
    check("study",STUDY,"S01","pass" if len(images)==29 else "unresolved",
          f"3份完整QPTIFF；作者33个Parent中29个CODEX=Y纳入、4个CODEX=N不导出；临床表38行中其余9行CODEX=N。实际闭合索引{len(images)}/29。原native_verification仅完整来源证明，本批新增独立core输入；无重新全量hash。")
    for row in images.to_dict(orient="records"):
        parent = row["image_id"]
        for key, detail in [
            ("I01","唯一image身份、原始QPTIFF/Parent与已关闭TIFF对应。"),
            ("I02","sample_id及panel_id外键可连接，39基础字段及sample_region_path扩展保存。"),
            ("I03","来源框、原生uint8和像素尺度保留，中心方形补边；TMA5_E4框止于实际扫描X边缘。"),
            ("I04","全部来源Parent坐标及DAPI概览逐图核对；sample_region为允许处理分区，包含背景，不作为组织轮廓；三个无标注邻组织碎片以可见间隙约束；padding为0。"),
            ("P03","54页uint8 CYX，OME通道名和输出位置与panel一致。"),
            ("I05","新导出图的目标组织内部块及边界块各32×32，全部54通道逐值相等；坐标详见export_evidence。C1沿用先前独立5块逐值验证，TIFF未重写。")]:
            check("image",parent,key,"pass",detail)
    check("panel",PANEL_ID,"P01","pass","全部54个来源标签按Study+Original marker精确连接当前唯一S1，source/output位置均为0..53。")
    check("panel",PANEL_ID,"P02","pass","DAPI为segmentation-only；其余53个downstream marker保留且唯一，所有S1核及膜/胞质分割通道像素均保留。")
    check("sample","29 specimens","M01","unresolved",
          f"29行有临床TMAMap=作者Parent标本连接，已导出{len(images)}图全部可连接。已查本地作者表、公开Zenodo、论文临床补充表和H5AD obs及作者仓库检索。H5AD ID对26个合格Parent给出23个不同分组及3对pre/post建议，但未明确定义patient身份，与论文29core/27patient计数仍需澄清；patient_key=空，确认patient连接0/29。建议见author_h5ad_identity_crosswalk.csv。")
    check("sample","29 specimens","M02","pass","复用既有独立逐值核对的临床整理，来源Patient ID保存source_subject_id，不用来源行ID或相似临床值推定患者；报告OS和follow-up保留原值，未造时间起点。")
    check("study",STUDY,"D01","unresolved",
          f"实际交付{len(images)}/29张input.tiff及相应sample_region、panel和sample表。sample连接{len(images)}/{len(images)}；患者连接0/29未解决。新图及区域掩码等待独立复核；C1原TIFF与历史独立检查保持。")
    path = OUTPUT_ROOT/"review_results.csv"
    previous = pd.read_csv(path,dtype=str,keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)
    previous = previous[~previous.preparer.eq(PREPARER)]
    with formal_publish_lease(path, directory=path.parent):
        pd.concat([previous,pd.DataFrame(rows,columns=columns)],ignore_index=True).to_csv(path,index=False,encoding="utf-8-sig")


def audit_delivery(images: pd.DataFrame, samples: pd.DataFrame, panel: pd.DataFrame,
                   points: pd.DataFrame) -> dict:
    """Reopen every NAS TIFF header and every region; check source-labelled sample separation."""
    names = panel.loc[panel.keep_channel].sort_values("output_channel_index").marker_id.tolist()
    records = []
    for row in images.to_dict(orient="records"):
        path = NAS_ROOT/row["input_image_path"]
        side = int(row["height_px"])
        with tifffile.TiffFile(path) as tif:
            if tif.series[0].axes!="CYX" or tif.series[0].shape!=(54,side,side) or tif.series[0].dtype!=np.dtype("uint8"):
                raise ValueError(f"Published TIFF layout differs: {row['image_id']}")
            xml = ElementTree.fromstring(tif.ome_metadata)
            actual = [e.attrib["Name"] for e in xml.iter() if e.tag.split("}")[-1]=="Channel"]
            if actual!=names:
                raise ValueError(f"Published channel names differ: {row['image_id']}")
        # Shared read_tiff_window currently requires TiffPage; OME series
        # discovery above installs TiffFrame objects. Keep pixel reads on a
        # fresh handle until the controller repairs the shared reader contract.
        with tifffile.TiffFile(path) as tif:
            middle = side//2
            padding_blocks = []
            for edge,window in [("left",(0,middle,1,middle+32)),
                                ("right",(side-1,middle,side,middle+32)),
                                ("top",(middle,0,middle+32,1)),
                                ("bottom",(middle,side-1,middle+32,side))]:
                if int(row[f"pad_{edge}_px"]):
                    for channel in range(54):
                        if read_tiff_window(tif,channel,window).any():
                            raise ValueError(f"Nonzero input padding: {row['image_id']}/{edge}")
                    padding_blocks.append(dict(edge=edge,window_xyxy=window,channels=54))
        region = tifffile.imread(NAS_ROOT/row["sample_region_path"],maxworkers=2)
        if region.shape!=(side,side) or region.dtype!=np.dtype("uint8") or not np.isin(region,[0,1]).all():
            raise ValueError("Published sample region shape/type/values differ")
        left,top,right,bottom = [int(row["pad_"+edge+"_px"]) for edge in ("left","top","right","bottom")]
        for strip in (region[:top],region[side-bottom:] if bottom else region[:0],
                      region[:,:left],region[:,side-right:] if right else region[:,:0]):
            if strip.any():
                raise ValueError("Sample region permits padding")
        filename = Path(row["source_files"][0]).name
        scan = points[points.Image.eq(filename+" - resolution #1")]
        xy = scan[["x_um","y_um"]].to_numpy()/row["pixel_size_x_um"]
        x0,y0,x1,y1 = [int(row["source_"+key+"_px"]) for key in ("x0","y0","x1","y1")]
        inside = (xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
        local = np.floor(xy[inside]-[x0,y0]).astype(int)
        own = scan.Parent.eq(row["image_id"]).to_numpy()[inside]
        allowed = region[local[:,1]+top,local[:,0]+left].astype(bool)
        if np.any(own&~allowed) or np.any(~own&allowed):
            raise ValueError(f"Published sample partition fails source labels: {row['image_id']}")
        records.append(dict(image_id=row["image_id"],shape_cyx=[54,side,side],dtype="uint8",header_names_equal=True,
                            target_centroids_allowed=int(own.sum()),neighbour_centroids_excluded=int((~own).sum()),
                            region_binary=True,padding_region_zero=True,input_padding_blocks=padding_blocks,
                            input_bytes=path.stat().st_size))
    result = dict(preparer=PREPARER,independent_review=False,expected_images=29,exported_images=len(images),
                  sample_rows=len(samples),sample_linked_images=int(images.sample_id.notna().sum()),
                  patient_linked_samples=int(samples.patient_key.ne("").sum()),
                  total_input_bytes=sum(row["input_bytes"] for row in records),records=records,
                  remaining=["patient ID semantics and cross-specimen linkage", "independent review of new images and all sample regions"])
    validation_path = EVIDENCE_ROOT / "preparer_validation.json"
    with formal_publish_lease(validation_path, directory=EVIDENCE_ROOT):
        write_json(validation_path,result)
    return result


def repair_intensity_description(job_root: Path) -> None:
    """Repair review-0 intensity metadata while retaining confirmed pixels and tables.

    Recheck only the source processing flag on each full-resolution IFD. The
    independent pixel, geometry, panel and clinical checks remain historical
    evidence; this preparer records the repair without signing acceptance.
    """
    preparer = f"codex_cli_tu2026_{job_root.name}"
    with nullcontext():
        images = pd.read_parquet(OUTPUT_ROOT / "images.parquet")
        progress("checking_source_intensity_metadata", images, progress_path=job_root / "progress.json",
                 patient_linked_samples=0, gaps=["Patient identity unresolved; targeted independent metadata review pending"])
        source_evidence = []
        for relative in sorted({str(path) for paths in images.source_files for path in paths}):
            with tifffile.TiffFile(NAS_ROOT / relative) as tif:
                flags = [text_element(ElementTree.fromstring(tif.pages[index].description.encode("utf-16")),
                                      "AutofluorescenceSubtracted") for index in range(54)]
            if flags != ["True"] * 54:
                raise ValueError(f"Source processing flags differ from review-0 evidence: {relative}")
            source_evidence.append(dict(source_file=relative, source_channel_indices=list(range(54)),
                                        AutofluorescenceSubtracted=flags))
        previous_descriptions = images.intensity_description.unique().tolist()
        images["intensity_description"] = INTENSITY_DESCRIPTION
        samples = pd.read_csv(OUTPUT_ROOT / "sample_metadata.csv", dtype=str, keep_default_na=False)
        panel = pd.read_csv(OUTPUT_ROOT / "panel_mapping.csv", dtype=str, keep_default_na=False)
        mappings = pd.read_csv(OUTPUT_ROOT / "field_mapping.csv", dtype=str, keep_default_na=False)
        intensity_rule = field_mapping().query("target_table == 'image_metadata' and target_field == 'intensity_description'")
        is_intensity = mappings.target_table.eq("image_metadata") & mappings.target_field.eq("intensity_description")
        mappings = pd.concat([mappings.loc[~is_intensity], intensity_rule], ignore_index=True)
        images = validate_image_metadata(images, samples, panel, mappings)

        # Replace only the two affected metadata files under the study lease.
        image_temporary = OUTPUT_ROOT / "images.parquet.partial"
        mapping_temporary = OUTPUT_ROOT / "field_mapping.csv.partial"
        with formal_publish_lease(OUTPUT_ROOT / "images.parquet", directory=OUTPUT_ROOT):
            images.to_parquet(image_temporary, index=False, compression="zstd")
            os.replace(image_temporary, OUTPUT_ROOT / "images.parquet")
        with formal_publish_lease(OUTPUT_ROOT / "field_mapping.csv", directory=OUTPUT_ROOT):
            mappings.to_csv(mapping_temporary, index=False, encoding="utf-8-sig")
            os.replace(mapping_temporary, OUTPUT_ROOT / "field_mapping.csv")

        evidence_path = job_root / "intensity_metadata_repair.json"
        evidence = dict(preparer=preparer, independent_review=False, repair_round=1,
                        image_ids=images.image_id.tolist(), source_evidence=source_evidence,
                        previous_descriptions=previous_descriptions, intensity_description=INTENSITY_DESCRIPTION,
                        changed_fields=["image_metadata.intensity_description", "field_mapping intensity_description rule"],
                        expected_images=29, exported_images=len(images), sample_rows=len(samples),
                        sample_linked_images=int(images.sample_id.notna().sum()),
                        patient_linked_samples=int(samples.patient_key.ne("").sum()))
        write_json(evidence_path, evidence)
        review_path = OUTPUT_ROOT / "review_results.csv"
        reviews = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        repair = dict(study=STUDY, preparer=preparer, reviewer="", scope="study",
                      record_id=STUDY, check_id="D01", result="unresolved",
                      source_evidence=metadata_source_relative(evidence_path, NAS_ROOT).as_posix(),
                      output_evidence=f"processed/{STUDY}/images.parquet;processed/{STUDY}/field_mapping.csv",
                      details="第一轮整理者定向修补：逐一重读3×54个原始全分辨率IFD的AutofluorescenceSubtracted=True，补齐已发布29行强度说明及精确字段来源规则。既有TIFF、sample_region及sample/panel表复用，保留全部历史独立复核记录。患者连接仍0/29；本轮metadata修补待不同CLI上下文独立复查，不签发D01 pass。")
        temporary = review_path.with_suffix(".csv.partial")
        with formal_publish_lease(review_path, directory=review_path.parent):
            pd.concat([reviews, pd.DataFrame([repair], columns=reviews.columns)], ignore_index=True).to_csv(
                temporary, index=False, encoding="utf-8-sig")
            os.replace(temporary, review_path)
    progress("metadata_repair_prepared", images, progress_path=job_root / "progress.json",
             patient_linked_samples=evidence["patient_linked_samples"],
             gaps=["Patient identity unresolved; targeted independent metadata review pending"])
    print(f"Repaired intensity metadata for {len(images)} images; reused every input and sample-region TIFF", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", help="Process one Parent core, e.g. TMA1_C1")
    parser.add_argument("--skip-tables", action="store_true", help="Publish only the requested input TIFF without replacing study tables")
    parser.add_argument("--prepare-regions", action="store_true", help="Build candidate sample partitions and DAPI overlays for visual review")
    parser.add_argument("--confirm-regions", action="store_true", help="Record the preparer's completed visual review of the region overlays")
    parser.add_argument("--tables-only", action="store_true", help="Refresh metadata for the already published image index")
    parser.add_argument("--repair-intensity-description", type=Path, metavar="JOB_DIR",
                        help="Repair review-0 upstream intensity metadata only; record preparer evidence in JOB_DIR")
    parser.add_argument("--fetch-identity", action="store_true", help="Retrieve public H5AD identity metadata only")
    parser.add_argument("--audit-only", action="store_true", help="Reopen all published headers, regions and four-side input padding blocks")
    args = parser.parse_args()

    if args.repair_intensity_description:
        repair_intensity_description(args.repair_intensity_description)
        return

    if args.fetch_identity:
        fetch_identity_metadata()
        return

    bounds = reviewed_core_bounds()
    if args.prepare_regions:
        prepare_sample_regions(extend_zero_margin_frames(bounds))
        return
    if args.confirm_regions:
        evidence = json.loads((REGION_ROOT / "region_evidence.json").read_text())
        review_destination = EVIDENCE_ROOT / "region_review"
        review_destination.mkdir(parents=True, exist_ok=True)
        for record in evidence["records"]:
            source_review = REGION_ROOT / f"{Path(record['qptiff']).stem}_region_review.png"
            destination_review = review_destination / source_review.name
            temporary_review = destination_review.with_name(destination_review.name + ".partial")
            with formal_publish_lease(destination_review, directory=review_destination):
                shutil.copyfile(source_review, temporary_review)
                os.replace(temporary_review, destination_review)
            record.update(review_status="preparer_confirmed", reviewer=PREPARER,
                          visual_evidence=metadata_source_relative(destination_review, NAS_ROOT).as_posix())
        fact_path = FACT_ROOT / "region_evidence.json"
        with formal_publish_lease(fact_path, directory=FACT_ROOT):
            write_json(fact_path, evidence)
        update = bounds.copy()
        pending = ~update.review_status.eq("confirmed")
        update.loc[pending,"review_status"] = "preparer_confirmed"
        update.loc[pending,"reviewer"] = PREPARER
        bounds_path = FACT_ROOT / "core_bounds.csv"
        temporary_bounds = bounds_path.with_name(bounds_path.name + ".partial")
        with formal_publish_lease(bounds_path, directory=FACT_ROOT):
            update.to_csv(temporary_bounds,index=False,encoding="utf-8-sig")
            os.replace(temporary_bounds, bounds_path)
        return

    source_names = native_channel_names()
    panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1_PATH)
    if len(panel) != 54 or int(panel.keep_channel.sum()) != 54:
        raise ValueError("The current Tu S1 panel must retain all 54 source channels, including DAPI")
    _, sample_by_parent = clinical_crosswalk()
    samples = sample_table()
    if not set(sample_by_parent.values()).issubset(set(samples.sample_id)):
        raise ValueError("Clinical core crosswalk references sample IDs absent from sample_metadata.csv")
    records = core_records()
    parents = sorted(bounds.index[bounds.review_status.isin(["confirmed", "preparer_confirmed"])])
    if args.parent:
        requested = args.parent.removesuffix(".tiff")
        parents = [parent for parent in parents if parent == requested]
        if not parents:
            raise ValueError(f"Requested Parent has no confirmed CODEX core boundary: {args.parent}")
    if not parents:
        raise ValueError("Confirm the core boundaries in core_bounds.csv before export")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_ROOT / "images.parquet"
    existing = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()
    mappings = field_mapping()
    if args.audit_only:
        result = audit_delivery(existing,samples,panel,author_coordinates())
        print(f"preparer audit: images={result['exported_images']}, sample links={result['sample_linked_images']}, patient links={result['patient_linked_samples']}",flush=True)
        return
    if args.tables_only:
        existing["intensity_description"] = INTENSITY_DESCRIPTION
        save_input_tables(OUTPUT_ROOT, existing, samples, panel, mappings)
        self_review(existing,samples,panel)
        progress("metadata_refreshed",existing,patient_linked_samples=0,
                 gaps=["patient mapping and independent review of new cores/regions remain pending"])
        return
    if args.skip_tables:
        raise ValueError("Batch delivery requires incremental study indices; --skip-tables is no longer supported")
    images = existing.to_dict(orient="records")
    region_evidence = json.loads((FACT_ROOT/"region_evidence.json").read_text())
    if any(record["review_status"] != "preparer_confirmed" for record in region_evidence["records"]):
        raise ValueError("Visually confirm all sample regions before exporting")
    points = author_coordinates()
    TEMP_ROOT.mkdir(parents=True,exist_ok=True)
    evidence_directory = EVIDENCE_ROOT / "export_evidence"
    evidence_directory.mkdir(parents=True,exist_ok=True)
    required_bytes = sum(54*max(int(r.source_y1_px-r.source_y0_px),int(r.source_x1_px-r.source_x0_px))**2
                         for r in bounds.itertuples() if r.parent in parents)
    if shutil.disk_usage(NAS_ROOT).free < required_bytes + 100*1024**3:
        raise ValueError(f"NAS reserve would be exceeded by uncompressed upper bound {required_bytes} bytes")
    progress("export_queued",existing,uncompressed_output_upper_bound_bytes=required_bytes,patient_linked_samples=0,
             gaps=["patient identity and independent review pending"])
    qptiff_cache: dict[str, tuple[tuple[int, int], float, str]] = {}
    staged_filename = None
    for parent in parents:
        record = records[parent]
        source_path = SOURCE_ROOT / record["qptiff"]
        if record["qptiff"] not in qptiff_cache:
            qptiff_cache[record["qptiff"]] = qptiff_metadata(source_path)
        source_shape, scale, slide_id = qptiff_cache[record["qptiff"]]
        bound = bounds.loc[parent]
        x0 = int(bound["source_x0_px"])
        y0 = int(bound["source_y0_px"])
        x1 = int(bound["source_x1_px"])
        y1 = int(bound["source_y1_px"])
        if not (0 <= x0 < x1 <= source_shape[1] and 0 <= y0 < y1 <= source_shape[0]):
            raise ValueError(f"Invalid native core window for {parent}: {(x0,y0,x1,y1)}")
        destination = OUTPUT_ROOT / "images" / parent / "input.tiff"
        previous = [row for row in images if row["image_id"]==parent]
        if previous:
            row = previous[0]
            if tuple(int(row[key]) for key in ("source_x0_px","source_y0_px","source_x1_px","source_y1_px")) != (x0,y0,x1,y1):
                raise ValueError(f"Existing image bounds changed: {parent}")
            with tifffile.TiffFile(destination) as tif:
                if len(tif.pages)!=54 or tif.pages[0].shape!=(int(row["height_px"]),int(row["width_px"])) or tif.pages[0].dtype!=np.dtype("uint8"):
                    raise ValueError(f"Existing closed TIFF header mismatch: {parent}")
            action = "reused"
        else:
            if staged_filename != source_path.name:
                if staged_filename is not None:
                    (TEMP_ROOT/staged_filename).unlink()
                local_source = TEMP_ROOT/source_path.name
                if shutil.disk_usage(TEMP_ROOT).free < source_path.stat().st_size + 64*1024**3 + 8*1024**3:
                    raise ValueError("Local source staging would exceed the 64 GiB reserve")
                progress("staging_source",pd.DataFrame(images),current_image=parent,current_source=source_path.name,
                         patient_linked_samples=0,gaps=["patient identity and independent review pending"])
                with formal_read_lease(source_path):
                    shutil.copyfile(source_path,local_source)
                if local_source.stat().st_size!=source_path.stat().st_size:
                    raise ValueError("Incomplete local QPTIFF staging")
                staged_filename = source_path.name
            source = SourceSpec(image_id=parent, panel_id=PANEL_ID, kind="tiff", path=TEMP_ROOT/source_path.name,
                            channels=tuple(range(54)), window=(x0, y0, x1, y1),
                            pixel_size_um=(1.0 / scale, 1.0 / scale), source_revision=SOURCE_REVISION,
                            intensity_description=INTENSITY_DESCRIPTION)
            local_output = TEMP_ROOT/parent/"input.tiff"
            progress("exporting_image",pd.DataFrame(images),current_image=parent,patient_linked_samples=0,
                     gaps=["patient identity and independent review pending"])
            with open_source(source) as reader:
                result = export_input(reader,panel,local_output,tile_size=512)
            if result["admitted"] is not True:
                raise ValueError(f"Image-level admission failed for {parent}")
            source_statistics = dict(result)
            result.update(source_x0_px=x0,source_y0_px=y0,source_x1_px=x1,source_y1_px=y1)
            row = build_image_row(parent,record,source_path,result,source_shape,slide_id,scale,
                                  sample_by_parent[parent],f"raw/public/{STUDY}/metadata.json#public/{STUDY}/input_facts/clinical_source_table.csv")
            pixel_check = verify_pixels(source,local_output,row,points)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary = destination.with_name("input.tiff.partial")
            with formal_publish_lease(destination, directory=destination.parent, lock_name=".input.lock"):
                shutil.copyfile(local_output,temporary)
                os.replace(temporary,destination)
            evidence_path = evidence_directory / f"{parent}.json"
            with formal_publish_lease(evidence_path, directory=evidence_directory):
                write_json(evidence_path,dict(
                           image_id=parent,source=metadata_source_relative(source_path, NAS_ROOT).as_posix(),
                           native_statistics=source_statistics,pixel_check=pixel_check,
                           output_bytes=destination.stat().st_size,preparer=PREPARER))
            shutil.rmtree(local_output.parent)
            images.append(row)
            action = "published"
        region_destination = destination.with_name("sample_region.tiff")
        if not region_destination.exists():
            with formal_publish_lease(region_destination, directory=destination.parent, lock_name=".input.lock"):
                temporary_region = region_destination.with_name("sample_region.tiff.partial")
                shutil.copyfile(region_path(parent),temporary_region)
                os.replace(temporary_region,region_destination)
        row["sample_region_path"] = metadata_source_relative(region_destination, NAS_ROOT).as_posix()
        row["intensity_description"] = INTENSITY_DESCRIPTION
        row["metadata_notes"] = ("Published specimen joined by TMAMap=Parent; complete visible core frame with native scan limits. "
            "sample_region defines target processing domain from author labels and visually checked inter-core gaps; "
            "it is not a tissue boundary. input.tiff retains neighbour pixels outside that region. Patient linkage remains unresolved.")
        row["split_evidence"] += "; sample processing partition is recorded in the accepted preparation evidence."
        # Persist after every closed image; previous batches stay in the index.
        frame = pd.DataFrame(images)
        save_input_tables(OUTPUT_ROOT,frame,samples,panel,mappings)
        self_review(frame,samples,panel)
        progress("exporting",frame,last_completed_image=parent,patient_linked_samples=0,
                 gaps=["patient identity and independent review pending"])
        print(f"{action} {parent}; closed index={len(images)}/29",flush=True)
    if staged_filename is not None:
        (TEMP_ROOT/staged_filename).unlink()
    audit_delivery(pd.DataFrame(images),samples,panel,points)
    progress("prepared",pd.DataFrame(images),patient_linked_samples=0,
             gaps=["cross-specimen patient identity remains unresolved; independent review of new cores and all sample regions pending"])
    print(f"prepared {len(images)} closed core images, {len(samples)} biological specimens; patient linkage unresolved",flush=True)


if __name__ == "__main__":
    main()
