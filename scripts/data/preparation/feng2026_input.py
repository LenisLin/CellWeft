"""Export the 520 indexed Feng2026 ROIs from original MCD and TXT records.

Check native coordinates first, then preview representatives and the planned
index. The full run publishes only verified TIFFs and preparer review records.
Publication uses the separately prepared native39 panel and image identity map.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from readimc import MCDFile
import tifffile

from cellweft.data.contracts import ImageBlock, ImageReader, SourceSpec, Window
from cellweft.data.io.readers import open_source, read_tiff_window, windows
from cellweft.data.preparation.inputs import file_revision, validate_panel_mapping
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, validate_image_metadata


STUDY = "Feng2026_IMC_BladderCancer"
PANEL = f"{STUDY}__native39"
PREPARER = "feng2026_input"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE = Path("/mnt/public/lyx/horizontalProject/IMC_20240725_63")
INTAKE = NAS / "cache/IMC_20240725_63_intake/indexed_sources.csv"
OUTPUT = NAS / "processed" / STUDY
CACHE = NAS / "cache" / STUDY
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
NATIVE_COUNT = 39
TXT_COORDINATES = ["Start_push", "End_push", "Pushes_duration", "X", "Y", "Z"]


def revision(path: Path) -> str:
    stat = file_revision(path)
    return f"size_bytes={stat['size']};mtime_ns={stat['mtime_ns']}"


def inventory(index: pd.DataFrame) -> tuple[list[dict], list[str], list[str]]:
    """Bind the intake order to original acquisition metadata and TXT headers."""
    if len(index) != 520 or not index.source_image.is_unique or not index.headers_match.eq("True").all():
        raise ValueError("The intake must contain 520 unique, matched entries")
    rows = []
    for number, entry in enumerate(index.itertuples(index=False), 1):
        relative = Path(entry.raw_relative_to_study)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Intake path must be relative to the original study directory")
        path = SOURCE / relative
        kind = path.suffix.lower().removeprefix(".")
        width, height = int(entry.width), int(entry.height)
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        image_id = f"{STUDY}__roi{number:04d}"
        acquisition_id = int(entry.acquisition_id) if kind == "mcd" else None
        rows.append(dict(
            study=STUDY, image_id=image_id, panel_id=PANEL, image_kind="roi",
            source_files=[str(path)], source_format=kind,
            source_acquisition_id=str(acquisition_id) if acquisition_id is not None else None,
            source_image_id=f"{relative}#Acquisition:{acquisition_id}" if kind == "mcd" else str(relative),
            source_axes="CYX", source_dtype="float32", source_channel_count=NATIVE_COUNT,
            source_x0_px=0, source_y0_px=0, source_x1_px=width, source_y1_px=height,
            pad_left_px=left, pad_top_px=top, pad_right_px=side-width-left,
            pad_bottom_px=side-height-top, height_px=side, width_px=side,
            input_image_path=str((OUTPUT / "images" / image_id / "input.tiff").relative_to(NAS)),
            intensity_description="Original acquisition intensities decoded as float32",
            split_method="source_acquisition" if kind == "mcd" else "already_independent",
            split_evidence=str(INTAKE.relative_to(NAS)),
        ))
    if (Counter(row["source_format"] for row in rows) != {"mcd": 518, "txt": 2}
            or len({row["source_image_id"] for row in rows}) != len(rows)):
        raise ValueError("Intake must bind 518 distinct MCD acquisitions and two distinct TXT files")

    by_path: dict[Path, list[dict]] = {}
    for row in rows:
        by_path.setdefault(Path(row["source_files"][0]), []).append(row)
    labels, isotopes = None, None
    for path, group in by_path.items():
        if group[0]["source_format"] != "mcd":
            continue
        source_revision = revision(path)
        with MCDFile(path) as container:
            acquisitions: dict[int, list] = {}
            for slide in container.slides:
                for acq in slide.acquisitions:
                    acquisitions.setdefault(acq.id, []).append(acq)
            for row in group:
                acq, = acquisitions[int(row["source_acquisition_id"])]
                channel_labels, channel_names = list(acq.channel_labels), list(acq.channel_names)
                if labels is None:
                    labels, isotopes = channel_labels, channel_names
                if (channel_labels != labels or channel_names != isotopes
                        or acq.num_channels != NATIVE_COUNT
                        or (acq.width_px, acq.height_px) != (row["source_x1_px"], row["source_y1_px"])):
                    raise ValueError("MCD geometry or ordered native panel disagrees with intake")
                meta = acq.metadata
                start, end = int(meta["DataStartOffset"]), int(meta["DataEndOffset"])
                count, remainder = divmod(end-start, (NATIVE_COUNT+3)*4)
                if int(meta["ValueBytes"]) != 4 or start < 0 or end > path.stat().st_size or remainder or not (
                        0 < count <= acq.height_px*acq.width_px):
                    raise ValueError("MCD records disagree with the declared acquisition")
                row.update(source_revision=source_revision, region_annotation=acq.description,
                           roi_id=str(acq.id), pixel_size_x_um=acq.pixel_size_x_um,
                           pixel_size_y_um=acq.pixel_size_y_um, recorded_pixels=count)
        if revision(path) != source_revision:
            raise ValueError("MCD version changed during inventory")

    txt_headers = [f"{label}({isotope}Di)" for label, isotope in zip(labels, isotopes, strict=True)]
    for row in rows:
        path = Path(row["source_files"][0])
        if row["source_format"] == "txt":
            source_revision = revision(path)
            with path.open(encoding="utf-8-sig") as handle:
                if next(handle).rstrip("\r\n").split("\t") != TXT_COORDINATES + txt_headers:
                    raise ValueError("TXT header disagrees with the ordered native panel")
            with open_roi(row | {"source_revision": source_revision}) as reader:
                height, width = reader.shape[-2:]
                count = int(reader.valid_window((0, 0, width, height)).sum())
            row.update(source_revision=source_revision, recorded_pixels=count,
                       metadata_notes="TXT records contain XY coordinates and no physical pixel size.")
        row["valid_pixels_path"] = (
            str(Path(row["input_image_path"]).with_name("valid_pixels.tiff"))
            if row["recorded_pixels"] < row["source_x1_px"]*row["source_y1_px"] else None
        )
    return rows, labels, isotopes


class TxtReader:
    """Decode the original TXT raster by explicit XY, including measured zeros."""

    dtype = np.dtype("float32")

    def __init__(self, path: Path, width: int, height: int) -> None:
        frame = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=np.float32)
        coords = frame[["X", "Y"]].to_numpy()
        if not (np.isfinite(coords).all() and np.equal(coords, np.floor(coords)).all()):
            raise ValueError("TXT coordinates must be finite integers")
        xcoord, ycoord = coords.astype(np.intp).T
        if np.any((xcoord < 0) | (xcoord >= width) | (ycoord < 0) | (ycoord >= height)):
            raise ValueError("TXT coordinates exceed the indexed frame")
        self.valid = np.zeros((height, width), dtype=bool)
        self.valid[ycoord, xcoord] = True
        if int(self.valid.sum()) != len(frame) or frame.empty:
            raise ValueError("TXT must contain unique measured pixel coordinates")
        self.shape = (NATIVE_COUNT, height, width)
        self.pixels = np.zeros(self.shape, dtype=self.dtype)
        self.pixels[:, ycoord, xcoord] = frame.iloc[:, len(TXT_COORDINATES):].to_numpy().T

    def read_window(self, channels: list[int], window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.pixels[channels, y0:y1, x0:x1]

    def valid_window(self, window: Window) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.valid[y0:y1, x0:x1].copy()

    def iter_blocks(self, channels: list[int], block_size: int) -> Iterator[ImageBlock]:
        for window in windows(self.shape[-2:], block_size):
            yield ImageBlock(window, self.read_window(channels, window), self.valid_window(window))


@contextmanager
def open_roi(row: dict) -> Iterator[ImageReader]:
    path = Path(row["source_files"][0])
    if revision(path) != row["source_revision"]:
        raise ValueError("Source version changed since inventory")
    if row["source_format"] == "txt":
        yield TxtReader(path, row["source_x1_px"], row["source_y1_px"])
    else:
        scale = row["pixel_size_x_um"], row["pixel_size_y_um"]
        spec = SourceSpec(row["image_id"], PANEL, "mcd", path, tuple(range(NATIVE_COUNT)),
                          acquisition_id=int(row["source_acquisition_id"]),
                          pixel_size_um=scale if None not in scale else None,
                          source_revision=row["source_revision"])
        with open_source(spec) as reader:
            yield reader
    if revision(path) != row["source_revision"]:
        raise ValueError("Source version changed during ROI read")


def representatives(rows: list[dict]) -> set[str]:
    mcd = [row for row in rows if row["source_format"] == "mcd"]
    selected = [mcd[0], *[row for row in rows if row["source_format"] == "txt"]]
    for predicate in (lambda row: row["source_x1_px"] != row["source_y1_px"],
                      lambda row: row["valid_pixels_path"] is not None):
        example = next((row for row in mcd if predicate(row)), None)
        if example is not None:
            selected.append(example)
    return {row["image_id"] for row in selected}


def verify_pixels(reader: ImageReader, tif: tifffile.TiffFile, row: dict,
                  source_indices: list[int], valid: np.ndarray) -> None:
    """Compare the center and opposite corners across every retained plane."""
    _, height, width = reader.shape
    left, top = row["pad_left_px"], row["pad_top_px"]
    for x0, y0 in ((0, 0), ((width-32)//2, (height-32)//2), (width-32, height-32)):
        window = x0, y0, x0+32, y0+32
        target = x0+left, y0+top, x0+32+left, y0+32+top
        expected = np.where(valid[y0:y0+32, x0:x0+32], reader.read_window(source_indices, window), 0)
        actual = np.stack([read_tiff_window(tif, page, target) for page in range(len(source_indices))])
        if not np.array_equal(expected, actual):
            raise ValueError("Exported pixels disagree with the original ROI")
    for x0, y0, x1, y1 in ((0, 0, left, row["height_px"]),
                           (left+width, 0, row["width_px"], row["height_px"]),
                           (left, 0, left+width, top),
                           (left, top+height, left+width, row["height_px"])):
        if x1 > x0 and y1 > y0 and any(read_tiff_window(tif, page, (x0, y0, x1, y1)).any()
                                       for page in range(len(source_indices))):
            raise ValueError("Input padding must be zero")


def publish_image(row: dict, panel: pd.DataFrame, sampled: bool) -> str | None:
    """Return a privacy-safe failure reason or verify the published TIFF pair."""
    try:
        kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
        path = NAS / row["input_image_path"]
        with open_roi(row) as reader:
            _, height, width = reader.shape
            valid = reader.valid_window((0, 0, width, height))
            if int(valid.sum()) != row["recorded_pixels"]:
                return "Measured pixel count differs from the source inventory"
            result = export_input(reader, panel, path, data_root=NAS)
            if not result["admitted"]:
                return "No nonzero downstream marker in the measured domain"
            if bool(result["valid_pixels"]) != bool(row["valid_pixels_path"]):
                return "Exported valid domain disagrees with the source inventory"
            square = row["height_px"], row["width_px"]
            with tifffile.TiffFile(path) as tif:
                pixels = ElementTree.fromstring(tif.ome_metadata).find(".//{*}Pixels")
                channels = pixels.findall("{*}Channel")
                if (tif.series[0].axes != "CYX" or len(tif.pages) != len(kept)
                        or any(page.shape != square or page.dtype != reader.dtype for page in tif.pages)
                        or [node.get("Name") for node in channels] != kept.marker_id.tolist()):
                    return "Published OME TIFF header disagrees with native geometry or panel order"
                for axis in ("X", "Y"):
                    scale = row.get(f"pixel_size_{axis.lower()}_um")
                    stored = pixels.get(f"PhysicalSize{axis}")
                    if scale != (float(stored) if stored is not None else None):
                        return "Published OME TIFF pixel size disagrees with the original ROI"
                if sampled:
                    verify_pixels(reader, tif, row, kept.source_channel_index.tolist(), valid)
            if row["valid_pixels_path"]:
                expected = np.pad(valid.astype(np.uint8), (
                    (row["pad_top_px"], row["pad_bottom_px"]),
                    (row["pad_left_px"], row["pad_right_px"]),
                ))
                actual = tifffile.imread(NAS / row["valid_pixels_path"])
                if actual.dtype != np.uint8 or not np.array_equal(actual, expected):
                    return "Published valid mask disagrees with the measured domain and padding"
    except Exception as exc:
        return type(exc).__name__ + (f" errno={exc.errno}" if isinstance(exc, OSError) else "")
    return None


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    partial = path.with_suffix(".parquet.partial")
    frame.to_parquet(partial, index=False)
    if pd.read_parquet(partial, columns=["image_id"]).image_id.tolist() != frame.image_id.tolist():
        raise ValueError("Serialized image order differs from the intake order")
    partial.replace(path)


def review_row(row: dict, check: str, result: str, details: str) -> dict:
    return dict(study=STUDY, preparer=PREPARER, reviewer="", scope="image",
                record_id=row["image_id"], check_id=check, result=result,
                source_evidence=str(INTAKE.relative_to(NAS)),
                output_evidence=row["input_image_path"] if result == "pass" else "",
                details=details)


def abort_private(_kind: type[BaseException], error: BaseException, _traceback: object) -> None:
    errno = f" errno={error.errno}" if isinstance(error, OSError) else ""
    print(f"Feng2026 input stopped: {type(error).__name__}{errno}", file=sys.stderr)


sys.excepthook = abort_private
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("check", "preview", "full"))
parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
args = parser.parse_args()
panel_path = OUTPUT / "panel_mapping.csv"
identity_path = CACHE / "image_identity.csv"
if args.stage != "check":
    missing = [path.name for path in (panel_path, identity_path) if not path.is_file()]
    if missing:
        sys.exit("Missing study dependencies: " + ", ".join(missing))

# Validate the intake before using any study-specific delivery metadata.
index = pd.read_csv(INTAKE, dtype=str, keep_default_na=False)
rows, labels, isotopes = inventory(index)
sampled_ids = representatives(rows)
print(f"Indexed {len(rows)} ROIs (518 MCD, 2 TXT), "
      f"{sum(bool(row['valid_pixels_path']) for row in rows)} with missing coordinates", flush=True)

if args.stage == "check":
    for row in rows:
        if row["image_id"] not in sampled_ids:
            continue
        with open_roi(row) as reader:
            _, height, width = reader.shape
            domain = reader.valid_window((0, 0, width, height))
            if int(domain.sum()) != row["recorded_pixels"]:
                raise ValueError("Representative ROI measured count differs from inventory")
            for x0, y0 in ((0, 0), ((width-32)//2, (height-32)//2), (width-32, height-32)):
                region = x0, y0, x0+32, y0+32
                if not np.isfinite(reader.read_window(list(range(NATIVE_COUNT)), region)).all():
                    raise ValueError("Representative ROI contains nonfinite source pixels")
    print(f"Checked native pixels and valid coordinates for {len(sampled_ids)} representative ROIs")
    sys.exit(0)

panel_version, identity_version = revision(panel_path), revision(identity_path)
panel = pd.read_csv(panel_path, keep_default_na=False)
panel = validate_panel_mapping(panel, input_channel_count=int(panel.keep_channel.astype(str).str.lower().eq("true").sum()))
if (len(panel) != NATIVE_COUNT or not panel.panel_id.eq(PANEL).all()
        or panel.source_name.tolist() not in (labels, isotopes)
        or not panel.loc[panel.keep_channel].role.eq("downstream").any()):
    sys.exit("native39 panel does not match the original ordered channels or lacks a retained marker")
identities = pd.read_csv(identity_path, dtype=str, keep_default_na=False).set_index("image_id", verify_integrity=True)
if set(identities.index) != {row["image_id"] for row in rows}:
    sys.exit("Image identity table must cover the exact indexed ROI range")
samples_path = OUTPUT / "sample_metadata.csv"
samples_version = revision(samples_path) if samples_path.is_file() else None
samples = (pd.read_csv(samples_path, dtype=str, keep_default_na=False) if samples_path.is_file()
           else pd.read_csv(TEMPLATES / "sample_metadata.csv"))
columns = pd.read_csv(TEMPLATES / "image_metadata.csv", nrows=0).columns
for row in rows:
    row["channel_count"] = int(panel.keep_channel.sum())
    row["sample_id"] = identities.at[row["image_id"], "sample_id"]
    if "region_annotation" in identities.columns:
        annotation = identities.at[row["image_id"], "region_annotation"]
        if annotation:
            row["region_annotation"] = annotation
planned = validate_image_metadata(pd.DataFrame(rows, columns=columns), samples, panel, pd.DataFrame())

# Publish the full planned index early for review. Formal rows require real assets.
planned_path = CACHE / "planned_images.parquet"
review_path = OUTPUT / "review_results.csv"
images_path = OUTPUT / "images.parquet"
review_columns = pd.read_csv(TEMPLATES / "review_results.csv", nrows=0).columns
with formal_publish_lease(images_path, directory=OUTPUT, data_root=NAS):
    existing_review = (pd.read_csv(review_path, keep_default_na=False) if review_path.is_file()
                       else pd.DataFrame(columns=review_columns))
    if (existing_review.check_id.eq("D01") & existing_review.result.eq("pass")).any():
        sys.exit("Existing D01 requires coordination before replacing bound inputs")
    if args.stage == "preview" and images_path.is_file():
        existing_ids = pd.read_parquet(images_path, columns=["image_id"]).image_id
        if not set(existing_ids).issubset(sampled_ids):
            sys.exit("Formal image index already extends beyond the preview")
with formal_publish_lease(planned_path, directory=CACHE, data_root=NAS):
    write_parquet(planned_path, planned)
print(f"Published planned index for {len(planned)} ROIs", flush=True)

jobs = [row for row in rows if args.stage == "full" or row["image_id"] in sampled_ids]
reviews, completed = [], []
failures: Counter[str] = Counter()
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    results = executor.map(lambda row: publish_image(row, panel, row["image_id"] in sampled_ids), jobs)
    for position, (row, error) in enumerate(zip(jobs, results, strict=True), 1):
        image_id = row["image_id"]
        sampled = image_id in sampled_ids
        if error is not None:
            failures[error] += 1
            reviews.append(review_row(row, "S01", "unresolved", error))
        else:
            completed.append(image_id)
            for check, detail in (
                ("S01", "Indexed ROI admitted by nonzero downstream marker in the measured domain"),
                ("I01", "Indexed source object and source version verified against the published input"),
                ("I02", "Panel and all populated sample references checked against current input tables"),
                ("I03", "Native ROI frame, square padding and measured domain checked"),
                ("P03", "Formal CYX TIFF reopened: dtype, dimensions, channel names and order checked"),
            ):
                reviews.append(review_row(row, check, "pass", detail))
            if sampled:
                reviews.append(review_row(row, "I05", "pass", "All retained planes matched the original at "
                                          "the center and opposite 32 px corners, and square padding was zero"))
        if position % 25 == 0:
            print(f"Publication checked {position}/{len(jobs)}, verified {len(completed)}", flush=True)

with formal_publish_lease(images_path, directory=OUTPUT, data_root=NAS):
    if (revision(panel_path) != panel_version or revision(identity_path) != identity_version
            or (revision(samples_path) if samples_path.is_file() else None) != samples_version):
        sys.exit("Study dependencies changed during export; published TIFFs require recheck")
    current = (pd.read_csv(review_path, keep_default_na=False) if review_path.is_file()
               else pd.DataFrame(columns=review_columns))
    if (current.check_id.eq("D01") & current.result.eq("pass")).any():
        sys.exit("D01 was added during export; coordinate an input binding recheck")
    own = current.preparer.eq(PREPARER) & current.reviewer.eq("")
    current = current.loc[~own]
    review_partial = review_path.with_suffix(".csv.partial")
    pd.concat([current, pd.DataFrame(reviews, columns=review_columns)], ignore_index=True).to_csv(
        review_partial, index=False, encoding="utf-8-sig")
    write_parquet(images_path, planned.loc[planned.image_id.isin(completed)])
    review_partial.replace(review_path)
print(f"Formal inputs verified {len(completed)}/{len(jobs)} attempted, {len(rows)} indexed", flush=True)
if failures:
    sys.exit(f"Preparation failures by category: {dict(failures)}")
