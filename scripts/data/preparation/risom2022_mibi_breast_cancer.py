"""Prepare the collected Risom2022 MIBI study using shared scientific operations.

Inputs: published Point*.ome.tif, confirmed global registry and source-panel CSV.
Outputs: processed/Risom2022_MIBI_BreastCancer/ with seven-field study index and images.
Prerequisite: 00_prepare_marker_registry.py and manual panel mapping are complete.
"""
import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import tifffile
import yaml

from cellweft.data.contracts import ImageRecord, SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.io.writers import publish_study
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.indexing import build_study_index
from cellweft.data.preparation.markers import build_marker_registry, build_panel_mapping
from cellweft.data.preparation.pipeline import prepare_image
from cellweft.data.preparation.segmentation import CellposeExecutor

PROJECT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("CELLWEFT_DATA_ROOT", "/mnt/NAS_21T/ProjectData/CellWeft")))
parser.add_argument("--config", type=Path, default=PROJECT/"configs/data/preparation/risom2022_mibi_breast_cancer.yaml")
parser.add_argument("--image-id", action="append", help="Process selected IDs; preserve the complete study index")
parser.add_argument("--device", help="Local device, e.g. cuda:0; omitted means file-based segmentation handoff")
parser.add_argument("--batch-size", type=int)
args = parser.parse_args()


def discover_images(config: dict, data_root: Path) -> dict[str, SourceSpec]:
    """Bind published scalar TIFF pages and OME micrometer scales to image IDs."""
    sources = {}
    for path in sorted((data_root/config["source_directory"]).glob(config["image_glob"])):
        with tifffile.TiffFile(path) as tif:
            pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
            scale = None
            if "PhysicalSizeX" in pixels.attrib and "PhysicalSizeY" in pixels.attrib:
                factors = {"µm":1.,"um":1.,"nm":0.001,"mm":1000.}
                scale = tuple(float(pixels.attrib[f"PhysicalSize{axis}"]) *
                              factors[pixels.attrib.get(f"PhysicalSize{axis}Unit","µm")] for axis in "XY")
            image_id = path.name.removesuffix(".ome.tif")
            sources[image_id] = SourceSpec(image_id,config["panel_id"],"tiff",path,
                tuple(range(len(tif.pages))),pixel_size_um=scale,
                intensity_description=config["intensity_description"])
    if not sources:
        raise ValueError("No published images matched this dataset configuration")
    return sources


# 1. Load confirmed semantics and resolve all source images before processing.
config = yaml.safe_load(args.config.read_text())
registry_path = args.data_root/config["registry"]
registry = build_marker_registry(pd.read_csv(registry_path))
panels = build_panel_mapping(pd.read_csv(args.data_root/config["confirmed_mapping"]),registry)
sources = discover_images(config,args.data_root)
output = args.data_root/"processed"/config["study"]
output.mkdir(parents=True,exist_ok=True)
panel_path = output/"panel_mapping.csv"
pd.concat(panels.values(),ignore_index=True).to_csv(panel_path,index=False)
index = build_study_index(config["study"],sources,panel_path,output)
publish_study(index,output,registry_path)
settings = load_settings(PROJECT/"configs/data/preparation/defaults.yaml",PROJECT/".cache/preparation",
    weights=args.data_root/"resources/pretrained/cellpose/cpsam_v2/checkpoint",batch_size=args.batch_size)

# 2. Apply the shared per-image chain; one local executor reuses its loaded model.
executor = CellposeExecutor(args.device) if args.device else None
with executor if executor is not None else nullcontext() as active_executor:
    for row in index.itertuples(index=False):
        if args.image_id and row.image_id not in args.image_id:
            continue
        record = ImageRecord(row.study,row.image_id,row.channel_count,row.pixel_size_um,
                             Path(row.image_path),Path(row.marker_list_path),Path(row.output_path))
        source = sources[row.image_id]
        with open_source(source) as reader:
            result = prepare_image(record,source,reader,panels[source.panel_id],settings,active_executor)
        print(row.image_id,result.metadata_path,", ".join(result.gaps))

# 3. Publish references to actual completed per-image metadata.
publish_study(index,output,registry_path)
