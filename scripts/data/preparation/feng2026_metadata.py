"""Register the private Feng IMC source and cross-check ROI patient grouping.

Inputs are the ordered NAS intake index, the author R workflow, its two saved
annotation states, and the grouping table. Outputs are the private raw manifest
and metadata bundle, two study CSVs, and the image identity CSV for input workers.
Run with PYTHONPATH=src. Use --check-only for aggregate verification without writes.
"""

import argparse
import csv
import errno
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from cellweft.data.raw_consolidation import file_identity
from cellweft.data.source_metadata import build_metadata_bundle
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import _write_images_if_changed, validate_image_metadata


STUDY = "Feng2026_IMC_BladderCancer"
ALIAS = "IMC_20240725_63"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE = Path("/mnt/public/lyx/horizontalProject") / ALIAS
TEMPLATES = Path(__file__).resolve().parents[3] / "docs/data/templates/study_input"
INDEX = NAS / "cache" / f"{ALIAS}_intake/indexed_sources.csv"
RAW = NAS / "raw/private" / STUDY
PROCESSED = NAS / "processed" / STUDY
CACHE = NAS / "cache" / STUDY
R_EXTRACT = """
dataset <- readRDS(commandArgs(TRUE)[1])
columns <- attr(attr(dataset, "colData"), "listData")
required <- c("sample_id", "roi_id", "patient_id", "width_px", "height_px")
stopifnot(all(required %in% names(columns)))
fields <- c(required, intersect("group", names(columns)))
values <- lapply(columns[fields], function(column) {
    if (inherits(column, "Rle")) column <- rep(attr(column, "values"), attr(column, "lengths"))
    as.character(column)
})
write.csv(unique(as.data.frame(values)), row.names=FALSE, na="")
"""


def stop_safely(error_type: type, error: BaseException, traceback: object) -> None:
    """Keep private source paths, labels and subprocess output out of diagnostics."""
    if type(error) is RuntimeError:
        print(str(error), file=sys.stderr)
    elif isinstance(error, OSError):
        print(f"Metadata I/O failed ({errno.errorcode.get(error.errno, error_type.__name__)})", file=sys.stderr)
    else:
        print(f"Metadata verification failed ({error_type.__name__})", file=sys.stderr)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def roi_key(value: str) -> str:
    """Match export and RDS ROI strings after stripping the extension and separators."""
    return re.sub(r"[-._ ]", "", value.removesuffix(".tiff").removesuffix(".tif")).lower()


def rds_rows(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    """Read unique per-ROI colData rows in memory, including geometry and grouping."""
    result = subprocess.run(["Rscript", "--vanilla", "-", str(path)],
                            input=R_EXTRACT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"RDS ROI extraction failed (exit {result.returncode})")
    reader = csv.DictReader(io.StringIO(result.stdout))
    rows = list(reader)
    keyed = {roi_key(row["sample_id"]): row for row in rows}
    if len(rows) != 520 or len(keyed) != 520 or "" in keyed:
        raise RuntimeError("An RDS does not contain 520 unique, internally consistent ROI rows")
    return set(reader.fieldnames or ()), keyed


def write_verified(path: Path, data: bytes) -> None:
    """Publish one result atomically after checking its staged bytes."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            if pending.read_bytes() != data:
                raise RuntimeError("Staged output byte verification failed")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    if path.read_bytes() != data:
        raise RuntimeError("Published output byte verification failed")


def csv_bytes(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    value = stream.getvalue()
    reader = csv.DictReader(io.StringIO(value))
    if reader.fieldnames != fields or list(reader) != [dict.fromkeys(fields, "") | row for row in rows]:
        raise RuntimeError("CSV header or row round trip failed")
    return value.encode("utf-8-sig")


def source_file(relative: str) -> Path:
    path = (SOURCE / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(SOURCE) or not path.is_file():
        raise RuntimeError("An indexed source file is outside the study or unavailable")
    return path


sys.excepthook = stop_safely
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check-only", action="store_true")
parser.add_argument("--tables-only", action="store_true", help="Publish identity tables during reception of source pixels")
args = parser.parse_args()

if not args.check_only:
    for directory in (RAW, PROCESSED, CACHE):
        parent = directory if directory.exists() else directory.parent
        if os.statvfs(parent).f_flag & os.ST_RDONLY:
            sys.exit("NAS publication blocked (EROFS)")

_, index = read_csv(INDEX)
index = index[:520]
keys = [roi_key(row["source_image"]) for row in index]
if len(index) != 520 or len(set(keys)) != 520 or any(not key for key in keys):
    raise RuntimeError("The ordered intake scope does not have 520 unique ROI keys")
native = {relative: source_file(relative) for relative in dict.fromkeys(
    row["raw_relative_to_study"] for row in index)}

author_scripts = [path for path in SOURCE.glob("*/*") if path.suffix.lower() == ".r"]
[(group_script, group_code)] = [(path, text) for path in author_scripts
                               if "spe_pid$Patient_ID" in (text := path.read_text())]
group_path_parts = re.search(
    r'group_info\s*<-\s*read\.csv\(file\.path\(workDir,\s*"([^"\n]+)",\s*"([^"\n]+)"\)\)',
    group_code,
).groups()
group_table = source_file(str(Path(*group_path_parts)))
_, group_rows = read_csv(group_table)
group_rows = group_rows[:63]
groups = {row["ID"]: row["Group"] for row in group_rows}
if len(groups) != 63 or any(not subject.strip() or not group.strip() for subject, group in groups.items()):
    raise RuntimeError("The author grouping scope does not contain 63 unique nonempty entries")
patient_keys = {subject: f"{STUDY}__patient{ordinal:04d}"
                for ordinal, subject in enumerate(groups, 1)}
alias_pairs = re.findall(r'pID\[pID\s*%in%\s*"([^"]+)"\]\s*<-\s*"([^"]+)"', group_code)
aliases = dict(alias_pairs)
if len(alias_pairs) != 4 or len(aliases) != 4 or not set(aliases.values()) <= groups.keys():
    raise RuntimeError("The four author alias rules could not be tied to the grouping table")

rds_name = re.search(r'(?m)^spe\s*<-\s*readRDS\(file\.path\(saveDir,\s*"([^"\n]+)"\)\)',
                     group_code).group(1)
grouped_source = next(path for path in SOURCE.glob("*/*") if path.name == rds_name and path.is_file())
rds_sources = sorted(path for path in grouped_source.parent.glob("*.rds") if rds_name.endswith(path.name))
if len(rds_sources) != 2:
    raise RuntimeError("Expected two saved states of the author RDS used by the grouping workflow")
states = [(path, *rds_rows(path)) for path in rds_sources]
[early] = [rows for _, fields, rows in states if "group" not in fields]
[(grouped_path, grouped)] = [(path, rows) for path, fields, rows in states if "group" in fields]
if early.keys() != grouped.keys() or grouped.keys() != set(keys):
    raise RuntimeError("The RDS ROI sets do not reconcile one-to-one with the intake scope")

identities = []
alias_confirmed = 0
changed_confirmed = 0
geometry_matches = 0
for ordinal, (entry, key) in enumerate(zip(index, keys, strict=True), 1):
    previous, current = early[key], grouped[key]
    dimensions = (int(entry["width"]), int(entry["height"]))
    rds_dimensions = (int(current["width_px"]), int(current["height_px"]))
    previous_dimensions = (int(previous["width_px"]), int(previous["height_px"]))
    if dimensions != rds_dimensions or dimensions != previous_dimensions:
        raise RuntimeError("RDS ROI geometry does not match the ordered intake index")
    geometry_matches += 1
    subject, group = current["patient_id"], current["group"]
    if subject in groups and group == groups[subject]:
        status = "confirmed"
        changed_confirmed += subject != previous["patient_id"]
        alias_confirmed += aliases.get(previous["patient_id"]) == subject
    else:
        status = "unresolved"
    identities.append(dict(study=STUDY, image_id=f"{STUDY}__roi{ordinal:04d}", sample_id="",
                           patient_key=patient_keys[subject] if status == "confirmed" else "",
                           source_subject_id=subject if status == "confirmed" else "",
                           source_roi_id=current["sample_id"], source_image=entry["source_image"],
                           source_group=group if status == "confirmed" else "", patient_linkage_status=status,
                           sample_linkage_status="unresolved"))

confirmed = sum(row["patient_linkage_status"] == "confirmed" for row in identities)
patients = len({row["patient_key"] for row in identities if row["patient_key"]})
print(f"ROI identities={len(identities)} geometry_matches={geometry_matches} raw_files={len(native)} "
      f"patient_linked_rois={confirmed} patients={patients} "
      f"unresolved_patient_rois={len(identities) - confirmed} prior_id_changes_confirmed={changed_confirmed} "
      f"explicit_alias_matches={alias_confirmed} biological_samples=0", flush=True)

bundle_path = RAW / "metadata.json"
prefix = f"private/{STUDY}/"
metadata_sources = {prefix + "intake/indexed_sources.csv": INDEX,
                    prefix + group_table.relative_to(SOURCE).as_posix(): group_table,
                    prefix + group_script.relative_to(SOURCE).as_posix(): group_script}
for name in ("steinbock/images.csv", "steinbock/panel.csv"):
    metadata_sources[prefix + name] = SOURCE / name
group_reference = f"{bundle_path.relative_to(NAS)}#{prefix}{group_table.relative_to(SOURCE)}"
member_paths = list(native.values()) + rds_sources + [SOURCE / "steinbock/masks" / row["source_image"] for row in index]
source_members = {Path("source") / path.relative_to(SOURCE): path for path in member_paths}
rds_reference = (RAW / "source" / grouped_path.relative_to(SOURCE)).relative_to(NAS).as_posix()
intake_reference = f"{bundle_path.relative_to(NAS)}#{prefix}intake/indexed_sources.csv"
mapping = [
    dict(study=STUDY, source_file=intake_reference, source_sheet_or_table="indexed_sources",
         source_field="source_image; width; height", target_table="image_identity", target_field="image_id",
         join_keys="Input row order 1–520", value_mapping="{}",
         transformation_rule="Assign study__roiNNNN in input order",
         notes="Require a unique image identifier and a unique normalized ROI key."),
    dict(study=STUDY, source_file=rds_reference, source_sheet_or_table="SpatialExperiment colData",
         source_field="sample_id; width_px; height_px; patient_id; group",
         target_table="image_identity", target_field="source_roi_id; patient_linkage_status",
         join_keys="RDS sample_id = indexed source_image after case and separator normalization",
         value_mapping="{}", transformation_rule="Require one ROI per normalized key and matching width and height",
         conflict_rule="Use grouped RDS patient_id only when its group agrees with the author grouping table",
         notes=f"Both RDS states covered 520 ROIs. {changed_confirmed} changed patient labels were corroborated. "
               f"{alias_confirmed} corroborated rows matched an explicit author alias from the earlier state."),
    dict(study=STUDY, source_file=group_reference, source_sheet_or_table="Author grouping table rows 1–63",
         source_field="ID; Group", target_table="image_identity",
         target_field="patient_key; source_subject_id; source_group", join_keys="Grouped RDS patient_id = ID",
         value_mapping="{}", transformation_rule="Assign study__patientNNNN from the grouping table's 1-based row order",
         conflict_rule="Leave the three patient fields empty unless ID and Group both agree with the grouped RDS",
         notes=f"{confirmed} verified ROI links across {patients} patients. {len(index) - confirmed} ROI links unresolved."),
    dict(study=STUDY, source_file=rds_reference, source_sheet_or_table="SpatialExperiment colData",
         source_field="sample_id; roi_id", target_table="sample_metadata", target_field="sample_id",
         join_keys="", value_mapping="{}", transformation_rule="Require a source link from ROI to biological specimen",
         conflict_rule="Keep sample_id empty when specimen identity is unresolved",
         notes="The author RDS sample_id is ROI-level. The patient grouping table does not supply a biospecimen key."),
]
mapping.append(dict(study=STUDY, source_file=rds_reference, source_sheet_or_table="SpatialExperiment colData",
                    source_field="patient_id;group", target_table="image_metadata", target_field="source_patient_key",
                    join_keys="Study image_id and source ROI evidence in derived/image_identity.csv", value_mapping="{}",
                    transformation_rule="Link only source-confirmed ROI identities corroborated by the grouping table",
                    notes=json.dumps({"data_type": "string", "definition": "Deidentified study patient key confirmed directly for this ROI"})))
tables = {
    PROCESSED / "sample_metadata.csv": csv_bytes(read_csv(TEMPLATES / "sample_metadata.csv")[0], []),
    PROCESSED / "field_mapping.csv": csv_bytes(read_csv(TEMPLATES / "field_mapping.csv")[0], mapping),
    CACHE / "image_identity.csv": csv_bytes(list(identities[0]), identities),
}
owner = dict(scope="private", study=STUDY, namespace=prefix.rstrip("/"), registered_study=True,
             raw_directory_preexisting=RAW.exists(), source_state="received",
             owner="Feng", registration_year=2026,
             diagnosis="Bladder cancer", source_alias=ALIAS)
members = []
for target, source in source_members.items():
    identity = file_identity(source)
    members.append(dict(target=(RAW / target).relative_to(NAS).as_posix(), original_source=str(source),
                        source_identity=identity, reception="copy"))
manifest = dict(schema="cellweft.raw_source_manifest.v1", study=STUDY, scope="private",
                raw_directory=RAW.relative_to(NAS).as_posix(), owner="Feng", registration_year=2026,
                diagnosis="Bladder cancer", source_alias=ALIAS, source_root=str(SOURCE),
                member_scope="indexed MCD/TXT containers, source instance masks and ROI grouping RDS evidence",
                roi_count=520, index_reference=intake_reference, members=members,
                external_sources_preserved=True, self_contained=True)

if args.check_only:
    print(f"In-memory CSV round trips={len(tables)} planned_raw_members={len(members)} "
          f"metadata_bundle_members={len(metadata_sources)}")
    sys.exit(0)

for directory in (RAW, PROCESSED, CACHE):
    directory.mkdir(parents=True, exist_ok=True)

if args.tables_only:
    for path, content in tables.items():
        write_verified(path, content)
    print(f"Published and verified {len(tables)} identity tables")
    sys.exit(0)

for member in members:
    received = NAS / member["target"]
    identity = file_identity(received)
    size = identity["size"]
    if size != member["source_identity"]["size"]:
        raise RuntimeError("Received source size differs")
    with received.open("rb") as target_stream, Path(member["original_source"]).open("rb") as upstream:
        for offset in sorted({0, size // 2, max(0, size - 65536)}):
            target_stream.seek(offset)
            upstream.seek(offset)
            if target_stream.read(65536) != upstream.read(65536):
                raise RuntimeError("Received source byte readback differs")
    member.update(target_identity=identity, byte_readback="head_middle_tail_64KiB")

for path, content in tables.items():
    if not path.is_file() or path.read_bytes() != content:
        write_verified(path, content)
metadata_sources[prefix + "derived/image_identity.csv"] = CACHE / "image_identity.csv"
build_metadata_bundle(bundle_path, metadata_sources, owner=owner)
manifest["published_ns"] = time.time_ns()
write_verified(RAW / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
print(f"Published and verified outputs={len(tables) + 2} received_sources={len(members)}")

image_table = PROCESSED / "images.parquet"
with formal_publish_lease(image_table, directory=PROCESSED, data_root=NAS):
    images = pd.read_parquet(image_table)
    expected_ids = [row["image_id"] for row in identities]
    if images.image_id.tolist() != expected_ids:
        raise RuntimeError("Portable input registration requires all 520 exported ROI rows in intake order")
    relative_sources = [(RAW / "source" / row["raw_relative_to_study"]).relative_to(NAS) for row in index]
    old_sources = [Path(paths[0]) for paths in images.source_files]
    for original, canonical, entry in zip(old_sources, relative_sources, index, strict=True):
        if original not in (SOURCE / entry["raw_relative_to_study"], canonical):
            raise RuntimeError("Image index disagrees with the received source identity")
    reviews = pd.read_csv(PROCESSED / "review_results.csv", keep_default_na=False)
    if reviews.check_id.eq("D01").any():
        if not all(old == new for old, new in zip(old_sources, relative_sources, strict=True)):
            raise RuntimeError("Portable input registration must precede independent D01 acceptance")
    else:
        images["source_files"] = [[str(path)] for path in relative_sources]
        revisions = {path: (NAS / path).stat() for path in set(relative_sources)}
        images["source_revision"] = [f"size_bytes={revisions[path].st_size};mtime_ns={revisions[path].st_mtime_ns}" for path in relative_sources]
        images["source_patient_key"] = [row["patient_key"] or pd.NA for row in identities]
        images = validate_image_metadata(images, pd.read_csv(PROCESSED / "sample_metadata.csv"),
                                         pd.read_csv(PROCESSED / "panel_mapping.csv"), pd.read_csv(PROCESSED / "field_mapping.csv"))
        _write_images_if_changed(image_table, images)
        reopened = pd.read_parquet(image_table)
        if reopened.source_patient_key.notna().sum() != confirmed or [paths[0] for paths in reopened.source_files] != list(map(str, relative_sources)):
            raise RuntimeError("Portable input registration readback failed")
print(f"Portable source references={len(images)} direct patient links={confirmed}")
