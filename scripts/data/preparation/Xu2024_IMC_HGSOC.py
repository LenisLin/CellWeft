"""Prepare Xu2024 HGSOC native MCD acquisitions as private input candidates.

The public archive is a ZIP64 Deflate64 file. This adapter decodes one MCD at
a time, uses the shared MCD SourceReader/exporter for validity and square
padding, and writes only the private candidate directory. It does not run
normalization, segmentation, quantification, crops or embeddings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


PROJECT = Path(__file__).resolve().parents[3]
DATA = Path("/mnt/NAS_21T/ProjectData/CellWeft")
STUDY = "Xu2024_IMC_HGSOC"
PREPARER_ID = "/root"
JOB = PROJECT / "cache/chakraborty_xu_20261107" / STUDY
ARCHIVE = DATA / "raw/public" / STUDY / "Ovarian_SciAdv_IMC_GeoMx.zip"
DEPENDENCIES = PROJECT / "cache/chakraborty_xu_20261107/deflate64_dependencies"
LOCAL_ARCHIVE = Path("/tmp/cellweft_xu_20261107/source_archive.zip")
TEMPLATES = PROJECT / "docs/data/templates/study_input"
PANEL_ID = f"{STUDY}__mcd47"
SAMPLE_SOURCE_NOTES = (
    "Native MCD member/acquisition is a provisional ROI-level key, not a verified "
    "biological-sample or patient ID. The paper reports patient-level clinical and "
    "demographic data for 42 patients in Supplement Table S1 and 110 tumor tissue "
    "samples, but no public key linking those records to current MCD cores/ROIs was "
    "verified. patient_key, anatomical_site, and lesion phase remain empty; per-ROI "
    "histology and preservation are not established by the acquisition description."
)
SOURCE_MEMBERS = [
    ("OVTP1_Bottom_080520.mcd", 13801233409),
    ("OVTP1_Top_080520.mcd", 8937042569),
    ("OVTP2_Bottom.mcd", 13293004874),
    ("OVTP2_Top_080520.mcd", 13317522932),
]
BACKGROUND = {
    "104Ru", "120Sn", "127I", "131Xe", "133Cs", "138Ba", "169Tm",
    "191Ir", "193Ir", "195Pt", "208Pb", "141Pr",
}

# Segmentation channel is a use role. A channel marked 1 or 2 remains a normal
# downstream measurement as well.
PANEL_MAP: dict[str, tuple[str, str, str, str]] = {
    "CD45": ("CD45", "PTPRC", "P08575", "2"),
    "FAP": ("FAP", "FAP", "Q12884", ""),
    "HLA-ABC": ("HLA-ABC", "mean(HLA-A, HLA-B, HLA-C)", "P04439 | P01889 | P10321", ""),
    "CD38": ("CD38", "CD38", "P28907", ""),
    "CD11c": ("CD11c", "ITGAX", "P20702", ""),
    "NKG2A": ("CD159a", "KLRC1", "P26715", ""),
    "SMA": ("alphaSMA", "ACTA2", "P62736", "2"),
    "T-bet": ("T-bet", "TBX21", "Q9UL17", ""),
    "CD16": ("CD16", "mean(FCGR3A, FCGR3B)", "P08637 | O75015", ""),
    "BCL6": ("BCL6", "BCL6", "P41182", ""),
    "Keratin": ("Cytokeratin (mapping pending)", "", "", "2"),
    "CCR4": ("CD194", "CCR4", "P51679", ""),
    "CD68": ("CD68", "CD68", "P34810", "2"),
    "CD138": ("CD138", "SDC1", "P18827", ""),
    "TGF-Beta": ("TGF-Beta (mapping pending)", "", "", ""),
    "LAG3": ("CD223", "LAG3", "P18627", ""),
    "CXCL13": ("CXCL13", "CXCL13", "O43927", ""),
    "CD11b": ("CD11b", "ITGAM", "P11215", ""),
    "CD4": ("CD4", "CD4", "P01730", ""),
    "HLA-DPDQDR": ("HLA-II (mapping pending)", "", "", ""),
    "PD-L1": ("CD274", "CD274", "Q9NZQ7", ""),
    "CD20": ("CD20", "MS4A1", "P11836", ""),
    "CD8a": ("CD8a", "CD8A", "P01732", ""),
    "Foxp3": ("FoxP3", "FOXP3", "Q9BZS1", ""),
    "RORgT": ("RORgammaT", "RORC", "P51449", ""),
    "PD-1": ("CD279", "PDCD1", "Q15116", ""),
    "CD19": ("CD19", "CD19", "P15391", ""),
    "Granzyme-B": ("Granzyme B", "GZMB", "P10144", ""),
    "KI-67": ("Ki-67", "MKI67", "P46013", ""),
    "CD3": ("CD3e", "CD3E", "P07766", ""),
    "CD27": ("CD27", "CD27", "P26842", ""),
    "PDL-2": ("CD273", "PDCD1LG2", "Q9BQ51", ""),
    "IDO": ("IDO1", "IDO1", "P14902", ""),
    "Podoplanin": ("Podoplanin", "PDPN", "Q86YL7", ""),
    "IL-10": ("IL-10", "IL10", "P22301", ""),
    "Histon-H3": ("Histone H3", "H3C1", "P68431", "1"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, default=JOB / "candidate_processed")
parser.add_argument("--limit", type=int, default=None,
                    help="Export only the first N acquisitions after reading all metadata")
parser.add_argument("--only-member", type=str, default=None,
                    help="Bounded smoke test: process one already-staged MCD member")
parser.add_argument("--metadata-only", action="store_true")
args = parser.parse_args()

OUTPUT = args.output_root / STUDY
JOB.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def candidate_path(path: Path) -> str:
    for root in (DATA, PROJECT):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.to_csv(temporary, index=False, columns=fieldnames, encoding="utf-8-sig")
    temporary.replace(path)


def update_progress(**values: Any) -> None:
    path = JOB / "progress.json"
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, current)


def zip_backend():
    import sys
    sys.path.insert(0, str(DEPENDENCIES))
    import zipfile64.zipfile as zipfile  # type: ignore
    return zipfile


def ensure_local_archive() -> Path:
    """Copy the immutable source once so Deflate64 reads stay off the NAS."""
    expected = ARCHIVE.stat()
    if LOCAL_ARCHIVE.is_file() and LOCAL_ARCHIVE.stat().st_size == expected.st_size:
        return LOCAL_ARCHIVE
    LOCAL_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = LOCAL_ARCHIVE.with_suffix(".zip.partial")
    with formal_read_lease(ARCHIVE, data_root=DATA):
        before = ARCHIVE.stat()
        with ARCHIVE.open("rb") as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=64 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        after = ARCHIVE.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        partial.unlink(missing_ok=True)
        raise RuntimeError("Source archive changed while being copied")
    if partial.stat().st_size != after.st_size:
        raise RuntimeError("Local source archive copy has an unexpected size")
    partial.replace(LOCAL_ARCHIVE)
    return LOCAL_ARCHIVE


def stage_member(member: str, stage_root: Path) -> Path:
    """Reuse the verified first MCD and decode only one later member."""
    first = Path("/tmp/cellweft_xu_20261107/OVTP1_Bottom_080520.mcd")
    expected = dict(SOURCE_MEMBERS)[member]
    if member == SOURCE_MEMBERS[0][0] and first.is_file() and first.stat().st_size == expected:
        return first
    target = stage_root / member
    if target.is_file() and target.stat().st_size == expected:
        return target
    partial = target.with_suffix(".mcd.extracting")
    local_archive = ensure_local_archive()
    with (zip_backend().ZipFile(local_archive) as archive,
          archive.open(member) as source,
          partial.open("wb") as output):
        while block := source.read(8 * 1024 * 1024):
            output.write(block)
    if partial.stat().st_size != expected:
        raise RuntimeError(f"{member}: decoded byte count differs from source member")
    partial.replace(target)
    return target


def native_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def source_revision(member: str, acquisition: Any) -> str:
    metadata = acquisition.metadata
    return ":".join([
        "zenodo7618945", str(ARCHIVE.stat().st_size), member,
        str(dict(SOURCE_MEMBERS)[member]), str(acquisition.id),
        str(metadata.get("DataStartOffset")), str(metadata.get("DataEndOffset")),
    ])


def has_acquired_pixels(acquisition: Any) -> bool:
    metadata = acquisition.metadata
    start = metadata.get("DataStartOffset")
    end = metadata.get("DataEndOffset")
    if start is None or end is None:
        return True
    return int(end) > int(start)


def save_native_windows(member: str, path: Path, acquisitions: list[Any],
                        source_names: list[str], evidence_dir: Path) -> list[dict[str, Any]]:
    """Keep small raw evidence before a staged MCD is released."""
    eligible = [acquisition for acquisition in acquisitions if has_acquired_pixels(acquisition)]
    selected: list[Any] = []
    for index in (0, len(eligible) // 2, len(eligible) - 1):
        if eligible and eligible[index] not in selected:
            selected.append(eligible[index])
    output_dir = evidence_dir / "native_windows" / Path(member).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for acquisition in selected:
        metadata = acquisition.metadata
        height = int(metadata["MaxY"])
        width = int(metadata["MaxX"])
        window_width = min(64, width)
        window_height = min(64, height)
        x0 = max(0, min(width - window_width, width // 2 - window_width // 2))
        y0 = max(0, min(height - window_height, height // 2 - window_height // 2))
        window = (x0, y0, x0 + window_width, y0 + window_height)
        scale = (native_float(metadata, "AblationDistanceBetweenShotsX"),
                 native_float(metadata, "AblationDistanceBetweenShotsY"))
        pixel_size = scale if all(value is not None for value in scale) else None
        source = SourceSpec(
            image_id=f"{Path(member).stem}__acquisition_{acquisition.id}__native_window",
            panel_id=PANEL_ID, kind="mcd", path=path,
            channels=tuple(range(len(source_names))), acquisition_id=int(acquisition.id),
            pixel_size_um=pixel_size, source_revision=source_revision(member, acquisition),
            intensity_description="Native MCD raw evidence window; no normalization",
        )
        with open_source(source) as reader:
            values = reader.read_window(list(range(len(source_names))), window)
            valid = reader.valid_window(window)
        evidence_path = output_dir / f"acquisition_{acquisition.id}.npz"
        np.savez_compressed(
            evidence_path, values=np.asarray(values, dtype=np.float32),
            valid=np.asarray(valid, dtype=np.uint8), x0=x0, y0=y0,
            x1=x0 + window_width, y1=y0 + window_height,
        )
        records.append({
            "member": member, "acquisition_id": int(acquisition.id),
            "source_channel_labels": list(acquisition.channel_labels),
            "source_revision": source_revision(member, acquisition),
            "source_path": str(path), "window_xyxy": list(window),
            "shape_cyx": list(values.shape), "dtype": str(values.dtype),
            "valid_fraction": float(valid.mean()),
            "finite": bool(np.isfinite(values).all()),
            "nonzero_by_channel": np.count_nonzero(values, axis=(1, 2)).tolist(),
            "evidence_path": candidate_path(evidence_path),
        })
    write_json(evidence_dir / "native_windows" / f"{Path(member).stem}.json", records)
    return records


def candidate_s1(source_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    six = ["Study", "Original marker", "Unified marker", "MAP-KG mapping",
           "UniProt accession", "Segmentation channel"]
    for index, name in enumerate(source_names):
        if name in BACKGROUND:
            unified, mapping, accession, segmentation = "Excluded", "", "", ""
            status = "excluded_background"
        else:
            unified, mapping, accession, segmentation = PANEL_MAP.get(
                name, (f"{name} (mapping pending)", "", "", ""))
            status = "mapping_pending" if "mapping pending" in unified else "mapped"
        rows.append({
            "Study": STUDY, "Original marker": name, "Unified marker": unified,
            "MAP-KG mapping": mapping, "UniProt accession": accession,
            "Segmentation channel": segmentation,
        })
        evidence.append({
            "study": STUDY, "source_channel_index": index, "source_marker": name,
            "candidate_unified_marker": unified, "mapping_status": status,
            "source_basis": "native per-acquisition MCD channel_labels; panel variants retain their actual ordered labels",
            "mapping_basis": "existing CellWeft S1 alias when available; unresolved label remains explicit",
            "segmentation_basis": "S1 role only; segmentation channel remains in the downstream input",
        })
    return pd.DataFrame(rows, columns=six), pd.DataFrame(evidence)


def field_mapping() -> pd.DataFrame:
    fields = columns("field_mapping")
    rows = []
    for source_field, target_table, target_field, rule, join, notes in [
        ("channel_labels", "panel_mapping", "source_name",
         "Preserve every native channel label and its zero-based order; background rows remain in the mapping table.",
         "native MCD member + channel index", "47 source channels; retained output is derived by panel_from_s1."),
        ("MaxX;MaxY;AblationDistanceBetweenShotsX/Y;ValueBytes", "image_metadata",
         "source_x1_px;source_y1_px;pixel_size_x_um;pixel_size_y_um;source_dtype",
         "Read actual acquisition metadata; leave scale empty if the source does not provide it.",
         "MCD member + acquisition id", "Full native acquisition frame; no resampling."),
        ("Description", "image_metadata", "roi_id;region_annotation;sample_id",
         "Retain the exact native ROI description and use a unique native-acquisition sample key; no patient inference.",
         "MCD member + acquisition id", "Table S1 contains patient-level data, but no shared key to MCD cores/ROIs was verified."),
        ("DataStartOffset;DataEndOffset", "image_metadata", "source_revision",
         "Bind the candidate to the archive revision, MCD member, acquisition and native data offsets.",
         "archive + member + acquisition", "Used by the SourceSpec/export checkpoint."),
        ("Zenodo archive and paper Supplement Table S1", "sample_metadata", "study",
         "Assign the stable CellWeft study identifier to every sample row.",
         "Zenodo record 7618945 / DOI 10.1126/sciadv.adk8805",
         "Patient-level supplement data exist, but no shared key to MCD cores/ROIs was verified."),
        ("MCD member basename; acquisition ID", "sample_metadata", "sample_id",
         "Construct a unique provisional acquisition-level ROI key; do not treat it as a patient or confirmed biological-sample ID.",
         "exact MCD member + acquisition ID",
         "The provisional key identifies the processed ROI/acquisition only."),
        ("MCD Description; member/acquisition fallback", "sample_metadata", "source_sample_id",
         "Preserve the native Description; if absent, retain the exact native ROI/acquisition identifier.",
         "exact MCD member + acquisition ID",
         "No public key links these ROI/acquisition identifiers to Supplement Table S1."),
        ("Article Methods and MCD acquisition description", "sample_metadata", "sample_type",
         "Retain the current TMA ROI category; do not promote it to a confirmed biological-sample identity.",
         "MCD member + acquisition ID",
         "The article distinguishes tumor tissue samples from cores/ROIs."),
        ("MCD archive / paper Supplement Table S1", "sample_metadata", "metadata_notes",
         "Record the existence of patient-level supplement data and the absence of a verified row-level core/ROI-to-patient key.",
         "MCD member/acquisition; no verified join to Table S1 patient rows",
         "Keep unsupported patient and biological-sample attributes blank."),
        ("Paper Supplement Table S1", "sample_metadata", "patient_key;source_subject_id",
         "Leave patient fields empty because no authoritative key joins Table S1 patient rows to current MCD cores/ROIs.",
         "No shared patient/sample/core/ROI key verified",
         "The paper reports 42 patient records; cohort totals are not a patient crosswalk."),
        ("archive inventory", "image_metadata", "metadata_notes",
         "Record the missing patient/core bridge and author whole-cell mask; technical processing remains candidate-only pending independent review.",
         "study", "Patient-level clinical/demographic data are in Supplement Table S1, but no image/core-to-patient key was verified."),
    ]:
        rows.append(dict(study=STUDY, source_file=str(ARCHIVE), source_sheet_or_table="",
                         source_field=source_field, target_table=target_table,
                         target_field=target_field, join_keys=join,
                         value_mapping="{}", transformation_rule=rule,
                         time_origin_rule="", conflict_rule="Unproven joins remain empty",
                         notes=notes))
    return pd.DataFrame(rows, columns=fields)


def sample_row(image_id: str, description: str) -> dict[str, Any]:
    row = {field: None for field in columns("sample_metadata")}
    row.update(
        study=STUDY, sample_id=f"{STUDY}__sample__{image_id}",
        source_sample_id=description or image_id,
        sample_type="TMA ROI", tissue_type=None,
        preservation_method=None,
        metadata_notes=SAMPLE_SOURCE_NOTES,
    )
    return row


def image_row(member: str, acquisition: Any, result: dict[str, Any], destination: Path,
              all_names: list[str], sample_id: str, evidence_path: Path,
              panel_id: str) -> dict[str, Any]:
    metadata = acquisition.metadata
    scale_x = native_float(metadata, "AblationDistanceBetweenShotsX")
    scale_y = native_float(metadata, "AblationDistanceBetweenShotsY")
    image_id = f"{Path(member).stem}__acquisition_{acquisition.id}"
    valid_name = result.get("valid_pixels")
    row = {field: None for field in columns("image_metadata")}
    row.update(
        study=STUDY, image_id=image_id, sample_id=sample_id, panel_id=panel_id,
        image_kind="roi", source_image_id=f"{member}::acquisition_{acquisition.id}",
        source_files=[f"raw/public/{STUDY}/{ARCHIVE.name}"], source_format="mcd",
        source_acquisition_id=str(acquisition.id), source_axes="CYX",
        source_dtype=result["source_dtype"], source_revision=source_revision(member, acquisition),
        slide_id=f"{Path(member).stem}__slide_{acquisition.slide.id}", core_id="",
        roi_id=str(acquisition.id), fov_id="", region_annotation=str(acquisition.description or ""),
        source_x0_px=0, source_y0_px=0,
        source_x1_px=int(metadata["MaxX"]), source_y1_px=int(metadata["MaxY"]),
        pad_left_px=result["pad_left_px"], pad_top_px=result["pad_top_px"],
        pad_right_px=result["pad_right_px"], pad_bottom_px=result["pad_bottom_px"],
        height_px=result["height_px"], width_px=result["width_px"],
        source_channel_count=len(all_names), channel_count=result["channel_count"],
        pixel_size_x_um=scale_x, pixel_size_y_um=scale_y,
        input_image_path=candidate_path(destination),
        valid_pixels_path=candidate_path(destination.with_name(valid_name)) if valid_name else None,
        intensity_description="Native MCD Dual float32 ion counts; no CellWeft normalization applied",
        split_method="source_acquisition", split_evidence=candidate_path(evidence_path),
        metadata_notes=(
            "Native full acquisition frame with shared square padding. "
            "Patient-level clinical/demographic data exist in Supplement Table S1, but no public "
            "core/ROI-to-patient key was verified; the author whole-cell mask is absent from the archive. "
            "technical input remains candidate-only pending independent review."
        ),
    )
    return row


def review_table(image_rows: list[dict[str, Any]], all_acquisitions: list[dict[str, Any]],
                 evidence: Path, output: Path) -> pd.DataFrame:
    fields = columns("review_results")
    rows = []
    for row in image_rows:
        rows.append(dict(study=STUDY, preparer=PREPARER_ID, reviewer="",
                         scope="image", record_id=row["image_id"], check_id="P03",
                         result="preparer_checked", source_evidence=str(evidence),
                         output_evidence=str(output / "images" / row["image_id"] / "input.tiff"),
                         details="Shared SourceReader/export_input header and geometry written; independent review pending."))
    for check_id, result, details in [
        ("S01", "preparer_checked", f"Four ZIP members and {len(all_acquisitions)} native acquisitions enumerated; source versions retained."),
        ("I01", "preparer_checked", f"{len(image_rows)} admitted candidate images have unique member/acquisition identities and candidate paths."),
        ("I02", "preparer_checked", f"{len(image_rows)} images have provisional native-ROI sample rows; patient links remain 0."),
        ("I03", "preparer_checked", "Shared exporter preserved full native frame and square padding without resampling."),
        ("I04", "unresolved", "No authoritative core/sample/region crosswalk in the archive; native ROI description retained."),
        ("P01", "preparer_checked", "Each acquisition uses its actual ordered panel; isotope-only rows, including variant raw9=141Pr, are explicitly excluded rather than relabelled CD38."),
        ("P02", "preparer_checked", "S1-derived nuclear/body roles remain downstream measurements and Histone H3 is the nuclear proposal."),
        ("P03", "preparer_checked", "Shared exporter wrote OME channel names in panel order; independent review pending."),
        ("I05", "unresolved", "Root independently checked three native source windows; full candidate pixel review remains pending."),
        ("M01", "unresolved", "Supplement Table S1 has patient-level records, but no public key links those records to current MCD cores/ROIs."),
        ("M02", "unresolved", "Patient and biological-sample attributes remain blank; the nonempty acquisition-level sample fields are explicitly mapped, pending independent review."),
        ("D01", "unresolved", "Candidate export complete; independent input review and downstream handover remain pending."),
    ]:
        rows.append(dict(study=STUDY, preparer=PREPARER_ID, reviewer="", scope="study",
                         record_id=STUDY, check_id=check_id, result=result,
                         source_evidence=str(evidence), output_evidence=str(output), details=details))
    return pd.DataFrame(rows, columns=fields)


def main() -> None:
    from readimc import MCDFile

    image_columns = columns("image_metadata")
    sample_columns = columns("sample_metadata")
    evidence_dir = JOB / "source_metadata"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    all_acquisitions: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    source_names: list[str] | None = None
    panel: pd.DataFrame | None = None
    panels: dict[tuple[str, ...], pd.DataFrame] = {}
    fields = field_mapping()
    existing_s1 = JOB / "candidate_s1_rows.csv"
    if existing_s1.is_file():
        existing_frame = pd.read_csv(existing_s1, keep_default_na=False)
        source_names = existing_frame["Original marker"].tolist()[:47]
        panel = panel_from_s1(STUDY, PANEL_ID, source_names, existing_s1)
        panels[tuple(source_names)] = panel
    existing_panel = OUTPUT / "panel_mapping.csv"
    if existing_panel.is_file():
        for _, frame in pd.read_csv(existing_panel, keep_default_na=False).groupby("panel_id", sort=False):
            names = frame.sort_values("source_channel_index").source_name.tolist()
            panels[tuple(names)] = panel_from_s1(STUDY, str(frame.iloc[0].panel_id), names, existing_s1)
    existing_images = OUTPUT / "images.parquet"
    if existing_images.is_file():
        image_rows = pd.read_parquet(existing_images).to_dict("records")
    existing_samples = OUTPUT / "sample_metadata.csv"
    if existing_samples.is_file():
        sample_rows = pd.read_csv(existing_samples, keep_default_na=False).to_dict("records")
        for row in sample_rows:
            row["sample_type"] = "TMA ROI"
            row["tissue_type"] = None
            row["preservation_method"] = None
            row["metadata_notes"] = SAMPLE_SOURCE_NOTES
    existing_acquisitions = evidence_dir / "all_acquisitions.json"
    if existing_acquisitions.is_file():
        all_acquisitions = json.loads(existing_acquisitions.read_text())
    existing_inventory = JOB / "inventory.json"
    if existing_inventory.is_file():
        old_excluded = json.loads(existing_inventory.read_text()).get("excluded_images", [])
        if isinstance(old_excluded, list):
            excluded = old_excluded
    existing_image_ids = {str(row["image_id"]) for row in image_rows}
    metadata_keys = {
        (str(row.get("member")), int(row.get("acquisition_id")))
        for row in all_acquisitions
    }
    processed = len(image_rows) + len(excluded)
    decoded_members = 0
    stage_root = Path("/tmp/cellweft_xu_20261107/mcd_members")
    stage_root.mkdir(parents=True, exist_ok=True)
    members = [item for item in SOURCE_MEMBERS
               if args.only_member is None or item[0] == args.only_member]
    if not members:
        raise ValueError(f"Unknown MCD member for --only-member: {args.only_member}")
    update_progress(phase="source_read", status="running", source_archive=str(ARCHIVE),
                    source_members=len(SOURCE_MEMBERS), decoded_members=0,
                    acquisitions=len(all_acquisitions),
                    candidate_images=len(image_rows), exported_images=len(image_rows),
                    sample_linked_images=len(sample_rows),
                    patient_linked_images=0, downstream_run=False, independent_review="pending",
                    gaps=["No verified public key from Supplement Table S1 to MCD cores/ROIs",
                          "No author whole-cell mask in archive", "Independent review pending"])

    for member_index, (member, expected_bytes) in enumerate(members):
        path = (Path("/tmp/cellweft_xu_20261107/OVTP1_Bottom_080520.mcd")
                if member == SOURCE_MEMBERS[0][0] else stage_root / member)
        if not path.is_file() or path.stat().st_size != expected_bytes:
            path = stage_member(member, stage_root)
        with MCDFile(path) as mcd:
            acquisitions = [acq for slide in mcd.slides for acq in slide.acquisitions]
            if not acquisitions:
                raise RuntimeError(f"{member}: no acquisitions found")
            for acquisition in acquisitions:
                labels = list(acquisition.channel_labels)
                if len(labels) != 47:
                    raise ValueError(f"{member}/{acquisition.id}: expected 47 native channels")
                if source_names is None:
                    source_names = labels
                key = tuple(labels)
                if key not in panels:
                    known = list(dict.fromkeys(name for names in [tuple(source_names), *panels, key] for name in names))
                    s1, s1_evidence = candidate_s1(known)
                    write_csv(existing_s1, s1, list(s1.columns))
                    write_csv(JOB / "candidate_s1_evidence.csv", s1_evidence)
                    panel_id = PANEL_ID if labels == source_names else f"{PANEL_ID}_variant{len(panels) + 1}"
                    panels[key] = panel_from_s1(STUDY, panel_id, labels, existing_s1)
                active_panel = panels[key]
                panel_id = str(active_panel.iloc[0].panel_id)
                panel = pd.concat(panels.values(), ignore_index=True)
                metadata = dict(member=member, slide_id=acquisition.slide.id,
                                acquisition_id=acquisition.id, description=acquisition.description,
                                channel_labels=list(acquisition.channel_labels),
                                metadata=acquisition.metadata)
                metadata_key = (member, int(acquisition.id))
                if metadata_key not in metadata_keys:
                    all_acquisitions.append(metadata)
                    metadata_keys.add(metadata_key)
                update_progress(phase="source_read", status="running",
                                decoded_members=decoded_members, acquisitions=len(all_acquisitions),
                                candidate_images=len(image_rows), exported_images=len(image_rows),
                                active_source=f"{member}::{acquisition.id}")
                if args.metadata_only or (args.limit is not None and processed >= args.limit):
                    continue
                image_id = f"{Path(member).stem}__acquisition_{acquisition.id}"
                if image_id in existing_image_ids:
                    continue
                if not has_acquired_pixels(acquisition):
                    if image_id in {item["image_id"] for item in excluded}:
                        continue
                    excluded.append({
                        "image_id": image_id, "member": member,
                        "acquisition_id": int(acquisition.id),
                        "reason": "source_no_acquired_image_data",
                        "data_start_offset": acquisition.metadata.get("DataStartOffset"),
                        "data_end_offset": acquisition.metadata.get("DataEndOffset"),
                    })
                    processed += 1
                    continue
                sample_id = f"{STUDY}__sample__{image_id}"
                scale = (native_float(acquisition.metadata, "AblationDistanceBetweenShotsX"),
                         native_float(acquisition.metadata, "AblationDistanceBetweenShotsY"))
                pixel_size = scale if all(value is not None for value in scale) else None
                source = SourceSpec(
                    image_id=image_id, panel_id=panel_id, kind="mcd", path=path,
                    channels=tuple(range(len(labels))), acquisition_id=int(acquisition.id),
                    pixel_size_um=pixel_size, source_revision=source_revision(member, acquisition),
                    intensity_description="Native MCD Dual float32 ion counts; no normalization",
                )
                destination = OUTPUT / "images" / image_id / "input.tiff"
                update_progress(phase="exporting", status="running", source_member=member,
                                acquisitions=len(all_acquisitions), active_image=image_id,
                                exported_images=len(image_rows))
                with open_source(source) as reader:
                    result = export_input(reader, active_panel, destination, tile_size=256)
                processed += 1
                if not result["admitted"]:
                    excluded.append({"image_id": image_id, "member": member,
                                     "acquisition_id": acquisition.id,
                                     "reason": "No retained downstream marker signal"})
                    continue
                row = image_row(member, acquisition, result, destination, labels,
                                sample_id, evidence_dir / "all_acquisitions.json", panel_id)
                image_rows.append(row)
                sample_rows.append(sample_row(image_id, str(acquisition.description or image_id)))
                existing_image_ids.add(image_id)
                save_input_tables(
                    OUTPUT,
                    pd.DataFrame(image_rows, columns=image_columns),
                    pd.DataFrame(sample_rows, columns=sample_columns),
                    panel,
                    fields,
                )
                update_progress(phase="exporting", status="running", source_member=member,
                                acquisitions=len(all_acquisitions), candidate_images=len(image_rows),
                                exported_images=len(image_rows), sample_linked_images=len(image_rows),
                                patient_linked_images=0, active_image=image_id)
                print(json.dumps({"stage": "candidate_input_exported", "study": STUDY,
                                  "image_id": image_id, "completed": len(image_rows)},
                                 ensure_ascii=False), flush=True)
            save_native_windows(member, path, acquisitions, source_names, evidence_dir)
        decoded_members += 1
        # The independent reviewer still needs this native member. Reclaim the
        # generated cache only after its source-to-input review has closed.
        update_progress(phase="source_read", status="running", decoded_members=decoded_members,
                        acquisitions=len(all_acquisitions), candidate_images=len(image_rows),
                        exported_images=len(image_rows), sample_linked_images=len(image_rows))

    if source_names is None or panel is None:
        raise RuntimeError("No source panel was read")
    write_json(evidence_dir / "all_acquisitions.json", all_acquisitions)
    save_input_tables(
        OUTPUT,
        pd.DataFrame(image_rows, columns=image_columns),
        pd.DataFrame(sample_rows, columns=sample_columns),
        panel,
        fields,
    )
    write_json(JOB / "inventory.json", {
        "schema": "cellweft.xu2024_hgsoc_source_inventory.v2",
        "study": STUDY, "status": "metadata_verified" if args.metadata_only else "candidate_exported",
        "source_archive": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
        "archive_revision": {"size": ARCHIVE.stat().st_size,
                             "mtime_ns": ARCHIVE.stat().st_mtime_ns},
        "archive_compression": "ZIP64 Deflate64 method 9; member bytes checked during staging",
        "archive_members": [{"name": name, "bytes": size} for name, size in SOURCE_MEMBERS],
        "acquisitions": len(all_acquisitions), "candidate_images": len(image_rows),
        "excluded_images": excluded, "channel_count": len(source_names),
        "source_channel_labels": source_names,
        "panels": [{"panel_id": str(frame.iloc[0].panel_id), "source_channel_labels": list(names),
                    "retained_channel_count": int(frame.keep_channel.sum())} for names, frame in panels.items()],
        "sample_linked_images": len(sample_rows), "patient_linked_images": 0,
        "mask": {"status": "absent_from_archive_inventory",
                 "action": "cpsam_v2 after independent D01 using reviewed nuclear/body inputs"},
        "source_metadata": str(evidence_dir / "all_acquisitions.json"),
        "gaps": ["No verified public key from Supplement Table S1 to MCD cores/ROIs",
                 "No author whole-cell mask", "Independent review pending"],
    })
    if not args.metadata_only:
        review = review_table(image_rows, all_acquisitions, evidence_dir / "all_acquisitions.json", OUTPUT)
        write_csv(OUTPUT / "review_results.csv", review, list(review.columns))
    bounded = args.limit is not None or args.only_member is not None
    status = "metadata_complete" if args.metadata_only else ("candidate_partial" if bounded else "candidate_complete")
    update_progress(phase=status, status=status, decoded_members=decoded_members,
                    acquisitions=len(all_acquisitions), candidate_images=len(image_rows),
                    complete_candidate_images=len(image_rows), acquisition_count=len(all_acquisitions),
                    source_stage="All selected MCD members decoded and enumerated",
                    exported_images=len(image_rows), sample_linked_images=len(sample_rows),
                    patient_linked_images=0, active_image=None, independent_review="pending",
                    excluded_images=len(excluded))
    write_json(JOB / "preparation_receipt.json", {
        "status": status, "preparer": PREPARER_ID, "study": STUDY,
        "preparation_contributors": ["/root/xu_prepare", "/root"],
        "runtime_controller": "/root",
        "images": len(image_rows), "acquisitions": len(all_acquisitions),
        "sample_linked_images": len(sample_rows), "patient_linked_images": 0,
        "candidate_root": str(OUTPUT), "source_retained": True,
        "downstream_complete": False, "independent_review": "pending",
        "excluded_images": excluded,
    })


if __name__ == "__main__":
    with (JOB / ".preparation.worker.lock").open("a+b") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
