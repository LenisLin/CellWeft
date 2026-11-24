"""Deliver native-intensity Danenberg 2022 breast IMC cores and source metadata.

Inputs: 794 complete SPORA Zarr ROIs, accepted native verification, S1, the
original Zenodo stack order/analysis code and the paper's sampling description.
Outputs: per-core input.tiff, full image/sample/panel/field tables and resumable
preparer observations. Run --assess-panel/--verify under shared formal read lease;
run --export under shared formal write lease. Independent review remains pending.
Repair: --repair-download fetches only compressed author metadata; run
--repair-sources under shared formal read lease to decode its metadata in memory.
--survival-sources
fetches small clinical attachments; --repair-metadata updates affected tables
and compares all unchanged image fields and TIFF file versions.

Source-specific adaptation: one bounded ROI is decoded once because SPORA
shards store all 39 channels together (largest ROI <249 MB). Move this adapter
to shared code only through the controller; source modules are unchanged.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import ctypes
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from numcodecs import blosc
import pandas as pd
import requests
import tifffile
import zarr
from PIL import Image, ImageDraw

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables

zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
blosc.set_nthreads(1)
STUDY = "Danenberg2022_IMC_BreastCancer"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
PROJECT = Path(__file__).resolve().parents[3]
RAW = NAS / "raw/public" / STUDY / "spora"
OUTPUT = NAS / "processed" / STUDY
META = metadata_source_directory(NAS, STUDY)
CACHE = META / "preparation_evidence"
AUTHOR = CACHE / "author_metadata/MBTMEIMCPublic"
COMPRESSED_CELLS = NAS / "raw/public" / STUDY / "metadata/zenodo_6036188/MBTMEIMCPublic/SingleCells.fst.zip-deflate"
COLLECTION = META / "collection_evidence"
VERIFICATION = COLLECTION / "spora_remaining/danenberg2022breast/verification.json"
OBJECTS = COLLECTION / "spora_inventory/danenberg2022breast/imc_source_objects.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
PREPARER = "0171_repair_1"
PROGRESS = PROJECT / "cache/study_input" / STUDY / "jobs" / PREPARER / "progress.json"
PANEL = "spora_imc_39"
RSCRIPT = Path("/home/lenislin/miniconda3/bin/Rscript")


def fetch_repair_sources(*, include_cells: bool = True, download_only: bool = False) -> None:
    """Recover the exact-version author identity/clinical members, without images.

    SingleCells.fst is needed because the release has no separate core map.
    Its complete compressed payload is read under a NAS read slot and decoded
    into an anonymous RAM file; only identity/tissue columns are read by R.
    No large uncompressed member is saved to disk. Only the small core summary
    is published. This bounds NAS activity to one read and avoids occupying
    the TIFF writer throughout a metadata network transfer.
    R packages are installed in this study cache, leaving global environments
    unchanged. This is a study-local FST adapter for later controller migration.
    """
    progress("fetching_author_metadata", 794,
             gaps=["逐core组织注释及作者临床附件待解析", "待独立复查"])
    archive = json.loads((CACHE / "author_archive_inventory.json").read_text())
    size = json.loads((CACHE / "code_record.json").read_text())["files"][0]["size"]
    url = "https://zenodo.org/records/6036188/files/MBTMEStrIMCPublic.zip?download=1"

    def byte_range(start: int, end: int) -> bytes:
        for attempt in range(3):
            try:
                response = requests.get(url + f"&repair_member_offset={start}",
                                        headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
                assert len(response.content) == end-start+1
                return response.content
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2)
        raise AssertionError("Unreachable range request")

    recovered = []
    wanted = ["IMCClinical.fst", "SingleCellsAnnotation.xlsx"]
    if include_cells:
        wanted.append("SingleCells.fst")
    for name in wanted:
        entry = next(e for e in archive if e["name"] == f"MBTMEIMCPublic/{name}")
        path = AUTHOR / name
        if not path.exists():
            header_bytes = byte_range(entry["offset"], entry["offset"]+1023)
            header = struct.unpack("<4s5H3I2H", header_bytes[:30])
            assert header[0] == b"PK\x03\x04" and header[3] == 8
            payload_start = entry["offset"] + 30 + header[-2] + header[-1]
            compressed = (COMPRESSED_CELLS if name == "SingleCells.fst"
                          else path.with_suffix(path.suffix + ".zip-deflate"))
            completed = compressed.stat().st_size if compressed.exists() else 0
            assert completed <= entry["compressed"]
            with compressed.open("ab") as handle:
                while completed < entry["compressed"]:
                    end = min(completed + 16*1024**2, entry["compressed"])
                    handle.write(byte_range(payload_start+completed, payload_start+end-1))
                    handle.flush()
                    completed = end
                    progress("fetching_author_metadata", 794, current_member=name,
                             compressed_bytes=completed, expected_compressed_bytes=entry["compressed"],
                             gaps=["逐core组织注释及作者临床附件待解析", "待独立复查"])
            if download_only:
                continue
            if name == "SingleCells.fst":
                recovered.append(dict(**entry, recovered_path=relative(compressed),
                                      metadata_decode="Complete member decoded in RAM; selected metadata columns read"))
                continue
            temporary = path.with_suffix(path.suffix + ".tmp")
            decoder = zlib.decompressobj(-15)
            with compressed.open("rb") as source, temporary.open("wb") as target:
                while block := source.read(1024**2):
                    target.write(decoder.decompress(block))
                target.write(decoder.flush())
            assert decoder.eof and temporary.stat().st_size == entry["size"]
            temporary.replace(path)
        assert path.stat().st_size == entry["size"]
        recovered.append(dict(**entry, recovered_path=relative(path)))
    if download_only:
        return

    # Keep the public release description with its explicit filename join rule.
    page = CACHE / "author_release.html"
    if not page.exists():
        response = requests.get("https://zenodo.org/records/6036188", timeout=60)
        response.raise_for_status()
        page.write_text(response.text)
    library = CACHE / "r_library"
    library.mkdir(exist_ok=True)
    temporary_root = Path("/tmp/study_input_batch") / STUDY
    temporary_root.mkdir(parents=True, exist_ok=True)
    # The installed R Makeconf is empty on this host; use its matching cached
    # package configuration through a study-local override, without repairing R.
    original_makeconf = RSCRIPT.parents[1] / "pkgs/r-base-4.2.3-h32f4cee_16/lib/R/etc/Makeconf"
    config = original_makeconf.read_text()
    build_prefix = config.splitlines()[0].removeprefix("R_HOME=").removesuffix("/lib/R")
    r_home = CACHE / "r_home"
    r_home.mkdir(exist_ok=True)
    installed_home = RSCRIPT.parents[1] / "lib/R"
    for path in installed_home.iterdir():
        if path.name not in {"etc", "bin"} and not (r_home / path.name).exists():
            (r_home / path.name).symlink_to(path)
    for directory in ["etc", "bin"]:
        (r_home / directory).mkdir(exist_ok=True)
        for path in (installed_home / directory).iterdir():
            if path.name not in {"Makeconf", "R"} and not (r_home / directory / path.name).exists():
                (r_home / directory / path.name).symlink_to(path)
    (r_home / "etc/Makeconf").write_text(config.replace(build_prefix, str(RSCRIPT.parents[1])))
    r_command = r_home / "bin/R"
    launcher = RSCRIPT.with_name("R").read_text()
    launcher = launcher.replace(f"R_HOME_DIR={installed_home}", f"R_HOME_DIR={r_home}", 1)
    r_command.write_text(launcher)
    r_command.chmod(0o755)
    environment = {**os.environ, "R_LIBS_USER": str(library), "MAKEFLAGS": "-j4",
                   "TMPDIR": str(temporary_root),
                   "PATH": str(RSCRIPT.parent) + os.pathsep + os.environ["PATH"],
                   "BUILD_INTERNAL_LIBS": "true"}
    if not (library / "fst").exists():
        for package, version in [("fstcore", "0.10.0"), ("fst", "0.9.8")]:
            tarball = CACHE / f"{package}_{version}.tar.gz"
            if not tarball.exists():
                response = requests.get(f"https://cran.r-project.org/src/contrib/{tarball.name}", timeout=60)
                response.raise_for_status()
                tarball.write_bytes(response.content)
            with (CACHE / f"{package}_install.log").open("w") as log:
                subprocess.run([str(r_command), "CMD", "INSTALL",
                                f"--library={library}", str(tarball)],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    progress("parsing_author_metadata", 794, gaps=["临床覆盖与core连接待核对", "待独立复查"])
    cell_descriptor = None
    cell_path = AUTHOR / "SingleCells.fst"
    if include_cells and not cell_path.exists():
        # R/fst needs a seekable file; a RAM-backed descriptor avoids a second
        # NAS copy of all 39 unused intensity columns in the cell table.
        cell_descriptor = ctypes.CDLL(None, use_errno=True).memfd_create(b"danenberg_core_metadata", 0)
        if cell_descriptor < 0:
            raise OSError(ctypes.get_errno(), "Unable to allocate metadata RAM descriptor")
        decoder = zlib.decompressobj(-15)
        with COMPRESSED_CELLS.open("rb") as source, os.fdopen(os.dup(cell_descriptor), "wb") as target:
            while block := source.read(1024**2):
                target.write(decoder.decompress(block))
            target.write(decoder.flush())
        expected = next(e["size"] for e in archive if e["name"] == "MBTMEIMCPublic/SingleCells.fst")
        assert decoder.eof and os.fstat(cell_descriptor).st_size == expected
        cell_path = Path(f"/proc/self/fd/{cell_descriptor}")
    r_code = '''
args <- commandArgs(TRUE)
library(fst)
threads_fst(4)
clinical <- read_fst(file.path(args[1], "IMCClinical.fst"))
write.csv(clinical, file.path(args[1], "IMCClinical.csv"), row.names=FALSE, na="")
writeLines(capture.output(str(clinical)), file.path(args[1], "IMCClinical_types.txt"))
if (args[2] == "True") {
writeLines(metadata_fst(args[3])$columnNames, file.path(args[1], "SingleCells_columns.txt"))
cells <- read_fst(args[3],
    columns=c("ImageNumber", "metabric_id", "isTumour", "is_normal", "is_dcis"))
writeLines(capture.output(str(cells)), file.path(args[1], "SingleCells_identity_types.txt"))
cells[] <- lapply(cells, function(x) { x <- as.character(x); x[is.na(x)] <- ""; x })
cores <- aggregate(rep(1L,nrow(cells)), cells, length)
names(cores)[ncol(cores)] <- "source_cell_rows"
write.csv(cores, file.path(args[1], "core_annotations.csv"), row.names=FALSE, na="")
}
'''
    try:
        subprocess.run([str(r_command), "--vanilla", "--slave", "-e", r_code, "--args", str(AUTHOR), str(include_cells), str(cell_path)],
                       pass_fds=() if cell_descriptor is None else (cell_descriptor,), env=environment, check=True)
    finally:
        if cell_descriptor is not None:
            os.close(cell_descriptor)
    annotation = pd.read_excel(AUTHOR / "SingleCellsAnnotation.xlsx")
    identity_definitions = annotation[annotation.iloc[:, 0].isin(
        ["ImageNumber", "metabric_id", "is_tumour", "is_normal", "is_dcis"])]
    write_json(CACHE / "repair_metadata_sources.json", dict(
        release="https://zenodo.org/records/6036188", archive=url,
        recovered_members=recovered, fst_versions=dict(fst="0.9.8", fstcore="0.10.0"),
        clinical=relative(AUTHOR / "IMCClinical.csv"),
        core_annotations=relative(AUTHOR / "core_annotations.csv"),
        r_library=relative(library),
        identity_annotation_definitions=identity_definitions.fillna("").to_dict("records"),
        fst_to_csv_field_names={"isTumour": "is_tumour"},
        join_rule="Author release explicitly defines MB####_ImageNumber_FullStack.tiff; normalize MB-#### to MB#### only after checking filename and all source identities",
        adaptation="Study-local R/fst metadata reader; propose controller migration if reused"))


def fetch_survival_sources() -> None:
    """Cache upstream clinical values and their variable definitions.

    Danenberg cites Rueda 2019 for updated METABRIC clinical data. Tables 5/8
    allow a direct value/units/endpoint comparison; exact T origin is considered
    separately because the upstream narrative uses both diagnosis and surgery.
    """
    publisher = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-019-1007-8/MediaObjects/"
    urls = {
        "rueda2019_paper.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6647838/fullTextXML",
        "rueda2019_supp_table5.txt": publisher + "41586_2019_1007_MOESM7_ESM.txt",
        "rueda2019_supp_table8.xls": publisher + "41586_2019_1007_MOESM10_ESM.xls",
        "rueda2019_supp_methods.pdf": publisher + "41586_2019_1007_MOESM1_ESM.pdf",
        "rueda2019_CreateModelER.R": "https://raw.githubusercontent.com/cclab-brca/brcarepred/master/Pipeline/1.CreateModelER.R"}
    for name, url in urls.items():
        path = CACHE / name
        if not path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
    dependencies = CACHE / "python_deps"
    if not (dependencies / "xlrd").exists():
        subprocess.run([sys.executable, "-B", "-m", "pip", "install", "--no-cache-dir", "--no-deps",
                        "--target", str(dependencies), "xlrd==2.0.2"], check=True)
    sys.path.insert(0, str(dependencies))
    definitions = pd.read_excel(CACHE / "rueda2019_supp_table8.xls", sheet_name="full_dataset")
    write_json(CACHE / "survival_source_definitions.json", dict(
        source_urls=urls, table8_definitions=definitions.fillna("").to_dict("records"),
        failed_optional_endpoints=[
            "PMC direct Table5 URL returned an HTML download interstitial",
            "EuropePMC supplementaryFiles contained only a zero-byte GIF; publisher files obtained"],
        exact_origin_limitation="Table8 defines T as time until last follow-up or death in days, without explicit baseline. Upstream methods/code refer to both diagnosis and post-surgery; no exact diagnosis-versus-surgery date conversion is established."))


def repair_metadata() -> None:
    """Repair only source annotations/clinical provenance and verify unchanged inputs.

    Use author ImageNumber plus metabric_id and the documented original filename
    rule to reconstruct the full core map. Clinical missingness is checked against
    both the exact-release FST and its cited upstream clinical release. Raw DSS
    values remain separate from OS/PFS/DFS; no clinical values are imputed.
    """
    progress("repairing_metadata", 794, gaps=["逐core连接与临床来源逐值核对中", "待独立复查"])
    images = pd.read_parquet(OUTPUT / "images.parquet")
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
    fields = pd.read_csv(OUTPUT / "field_mapping.csv", dtype=str, keep_default_na=False)
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    assert len(images) == 794 and len(samples) == 718
    before_images, before_samples = images.copy(deep=True), samples.copy(deep=True)
    before_panel = (OUTPUT / "panel_mapping.csv").read_bytes()
    input_versions = {r.image_id: dict(size=(NAS / r.input_image_path).stat().st_size,
                                     mtime_ns=(NAS / r.input_image_path).stat().st_mtime_ns)
                      for r in images.itertuples(index=False)}
    backup = PROGRESS.parent / "metadata_before_repair"
    backup.mkdir(exist_ok=True)
    for name in ["images.parquet", "sample_metadata.csv", "panel_mapping.csv", "field_mapping.csv", "review_results.csv"]:
        if not (backup / name).exists():
            shutil.copyfile(OUTPUT / name, backup / name)

    cores = pd.read_csv(AUTHOR / "core_annotations.csv", dtype=str, keep_default_na=False)
    assert cores.ImageNumber.is_unique and cores.metabric_id.ne("").all()
    cores["source_subject_id"] = cores.metabric_id.str.replace(r"^MB-", "MB", regex=True)
    cores["original_file_name"] = cores.source_subject_id + "_" + cores.ImageNumber + "_FullStack.tiff"
    assert cores.original_file_name.is_unique
    assert set(cores.original_file_name) == set(tissues.original_file_name)
    core_map = tissues.merge(cores, on="original_file_name", validate="one_to_one", suffixes=("", "_author"))
    assert (core_map.original_patient_identifier == core_map.source_subject_id).all()
    assert (core_map.metabric_id.isna() | core_map.metabric_id.eq(core_map.source_subject_id)).all()
    annotations = ["isTumour", "is_normal", "is_dcis"]
    assert core_map.isTumour.isin(["TRUE", "FALSE", ""]).all()
    assert all(core_map[key].isin(["0", "1", ""]).all() for key in ["is_normal", "is_dcis"])
    annotation_by_image = core_map.set_index("tissue_id")
    for index, row in images.iterrows():
        source = annotation_by_image.loc[row.image_id]
        assert row.core_id + "_FullStack.tiff" == source.original_file_name
        assert row.sample_id == "primary_breast_" + source.original_patient_identifier
        images.at[index, "region_annotation"] = json.dumps({k: (source[k] == "TRUE" if k == "isTumour" else int(source[k])) if source[k] != "" else None
                                                           for k in annotations})
        notes = json.loads(row.metadata_notes)
        notes["region_annotation"] = "Author SingleCells.fst core-level flags, checked constant across all source cell rows; null preserves an unknown source flag"
        notes["author_image_number"] = source.ImageNumber
        notes["author_metabric_id"] = source.metabric_id_author
        images.at[index, "metadata_notes"] = json.dumps(notes)
    core_map.to_csv(CACHE / "repaired_core_map.csv", index=False)

    clinical = pd.read_csv(AUTHOR / "IMCClinical.csv", dtype=str, keep_default_na=False)
    clinical["source_subject_id"] = clinical.metabric_id.str.replace(r"^MB-", "MB", regex=True)
    assert clinical.source_subject_id.is_unique
    upstream = pd.read_csv(CACHE / "rueda2019_supp_table5.txt", sep="\t", dtype=str, keep_default_na=False)
    assert upstream["METABRIC.ID"].is_unique
    upstream["source_subject_id"] = upstream["METABRIC.ID"].str.replace(r"^MB-", "MB", regex=True)
    matched = clinical.merge(upstream, on="source_subject_id", validate="one_to_one", suffixes=("", "_upstream"))
    assert len(matched) == len(clinical) == 709
    assert np.allclose(matched.yearsToStatus.astype(float), matched["T"].astype(float)/365.25,
                       rtol=1e-12, atol=1e-12)
    assert (matched.DeathBreast == matched.DeathBreast_upstream).all()
    clinical_by_id = clinical.set_index("source_subject_id")
    clinical_fields = {"Grade": "grade", "PAM50": "molecular_subtype", "ERStatus": "source_er_status",
                       "ERBB2_pos": "source_erbb2_positive", "LymphNodesOrdinal": "source_lymph_nodes_ordinal",
                       "sizeOrdinal": "source_size_ordinal", "IntClust": "source_intclust",
                       "yearsToStatus": "source_years_to_status", "DeathBreast": "source_death_breast",
                       "isValidation": "source_is_validation"}
    missing = []
    compared = 0
    for index, row in samples.iterrows():
        subject = row.source_subject_id
        assert row.sample_id == f"primary_breast_{subject}" and row.patient_key == f"METABRIC:{subject}"
        if subject in clinical_by_id.index:
            for source, target in clinical_fields.items():
                expected, actual = clinical_by_id.loc[subject, source], row[target]
                if expected != "" and source in {"Grade", "yearsToStatus", "DeathBreast"}:
                    assert np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-12)
                else:
                    assert actual.lower() == expected.lower(), (subject, source)
                compared += 1
        else:
            assert not upstream.source_subject_id.eq(subject).any()
            assert not upstream["MATCHED.NORMAL.METABRIC.ID"].str.replace("-", "").eq(subject).any()
            assert all(row[target] == "" for target in clinical_fields.values())
            missing.append(dict(sample_id=row.sample_id, source_subject_id=subject,
                                image_ids=images.loc[images.sample_id.eq(row.sample_id), "image_id"].tolist(),
                                resolution="Absent in exact-release IMCClinical.fst and cited upstream Table5, including matched-normal identifier; preserve clinical blanks"))
        notes = json.loads(row.metadata_notes)
        notes["clinical_missing"] = subject not in clinical_by_id.index
        notes["clinical_source_check"] = relative(CACHE / "clinical_source_reconciliation.json")
        notes["survival"] = "yearsToStatus = upstream Table5 T/365.25; DeathBreast = disease-specific death (1 yes, 0 no); DSS endpoint confirmed by author survival code. Exact T baseline is not explicit in Table8 and diagnosis/surgery terminology differs in upstream text/code. Raw fields retained; OS/PFS/DFS remain blank."
        samples.at[index, "metadata_notes"] = json.dumps(notes)
    assert len(missing) == 11 and sum(len(r["image_ids"]) for r in missing) == 13
    write_json(CACHE / "clinical_source_reconciliation.json", dict(
        author_clinical_rows=len(clinical), source_linked_samples=len(samples)-len(missing),
        missing_samples=missing, clinical_fields_compared=compared,
        upstream_rows=len(upstream), author_upstream_rows_compared=len(matched),
        all_years_equal_upstream_days_divided_by_365_25=True,
        all_disease_specific_events_equal=True, endpoint="disease-specific survival",
        origin="Not explicitly defined for T; diagnosis versus surgery unresolved in source wording",
        source_evidence=[relative(AUTHOR / "IMCClinical.fst"), relative(AUTHOR / "IMCClinical.csv"),
                         relative(CACHE / "rueda2019_supp_table5.txt"), relative(CACHE / "rueda2019_supp_table8.xls"),
                         relative(CACHE / "rueda2019_supp_methods.pdf"), relative(CACHE / "rueda2019_CreateModelER.R"),
                         relative(AUTHOR / "Code/Figure6/SurvivalPlots.R")]))

    # Replace obsolete source-search limitations; record source rules once.
    fields.loc[fields.source_file.eq(relative(CACHE / "metadata_sources.json")), "notes"] = (
        "Repair source search completed: full author core map, exact-release clinical FST, Rueda2019 clinical Tables5/8, methods and author code; 11 samples/13 images are absent from both clinical releases; exact survival baseline remains source-ambiguous")
    fields.loc[fields.source_file.eq(relative(CACHE / "paper.xml")), "notes"] = (
        "Cohort specimen provenance remains primary excision; core composition is now retained per-image from author flags, including normal/DCIS, without changing image admission or source sample grouping")
    for key in ["source_years_to_status", "source_death_breast"]:
        selected = fields.target_field.eq(key)
        for index in fields.index[selected]:
            specification = json.loads(fields.at[index, "notes"])
            specification["definition"] = ("Source yearsToStatus, equal to Rueda2019 Table5 T/365.25, used for disease-specific survival" if key == "source_years_to_status"
                                           else "Source disease-specific death event: 1=yes, 0=no; confirmed against Rueda2019 Table5/Table8 and author Cox model")
            specification["time_origin"] = "T baseline not explicitly defined; upstream diagnosis/post-surgery wording ambiguous; standardized endpoints remain blank"
            fields.at[index, "notes"] = json.dumps(specification)

    def mapping(path: Path, source: str, table: str, target: str, join: str, rule: str, notes: str) -> None:
        fields.loc[len(fields)] = dict(study=STUDY, source_file=relative(path), source_sheet_or_table=path.name,
                                      source_field=source, target_table=table, target_field=target, join_keys=join,
                                      value_mapping="", transformation_rule=rule, time_origin_rule="",
                                      conflict_rule="Conflicting core flags or clinical values halt publication", notes=notes)

    fields = fields[~fields.source_file.isin([relative(COMPRESSED_CELLS), relative(AUTHOR / "SingleCells.fst.zip-deflate"),
                                            relative(CACHE / "clinical_source_reconciliation.json")])].reset_index(drop=True)
    mapping(COMPRESSED_CELLS, "ImageNumber; metabric_id; isTumour; is_normal; is_dcis",
            "image_metadata", "region_annotation", "author MB####_ImageNumber_FullStack.tiff = tissues.original_file_name; verify original_patient_identifier and nonmissing metabric_id",
            "Read all source rows in identity/annotation columns; require constant flags and one subject per ImageNumber; preserve FST boolean isTumour and numeric is_normal/is_dcis/null as JSON without inferring histology from publication flag",
            "All 794 images connected to author core annotations. FST isTumour is the logical name used by author code; the CSV dictionary calls it is_tumour (0/1). See SingleCellsAnnotation.xlsx, author Code/ExtFig4 and repaired_core_map.csv")
    mapping(CACHE / "clinical_source_reconciliation.json", "exact-release FST and cited upstream clinical Tables5/8",
            "sample_metadata", "metadata_notes", "explicit source_subject_id = normalized author/upstream METABRIC identifier",
            "Compare every delivered clinical value to exact-release FST; check missing subjects in primary and matched-normal upstream IDs; confirm DSS event and days-to-years equality; preserve verified source missingness",
            "794 images -> 718 biological samples -> 718 patient keys; 707 samples have exact-release clinical rows; 11 samples/13 images absent in both clinical sources. DSS endpoint confirmed; exact baseline unresolved, raw source fields retained")

    save_input_tables(OUTPUT, images, samples, panel, fields)
    saved_images = pd.read_parquet(OUTPUT / "images.parquet")
    saved_samples = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(saved_images.drop(columns=["region_annotation", "metadata_notes"]),
                                  before_images.drop(columns=["region_annotation", "metadata_notes"]))
    pd.testing.assert_frame_equal(saved_samples.drop(columns="metadata_notes"), before_samples.drop(columns="metadata_notes"))
    assert (OUTPUT / "panel_mapping.csv").read_bytes() == before_panel
    for row in saved_images.itertuples(index=False):
        stat = (NAS / row.input_image_path).stat()
        assert input_versions[row.image_id] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        source = annotation_by_image.loc[row.image_id]
        assert json.loads(row.region_annotation) == {k: (source[k] == "TRUE" if k == "isTumour" else int(source[k])) if source[k] != "" else None for k in annotations}
    write_json(CACHE / "repaired_sample_linkage.json", dict(
        sample_rows=718, sample_linked_images=794, patient_linked_samples=718,
        author_core_annotations=794, keys=["ImageNumber", "metabric_id", "original_file_name", "tissue_id", "original_patient_identifier"],
        grouping="Unchanged biological primary breast excision; multiple cores share sample; flags describe core histology",
        evidence=[relative(CACHE / "repaired_core_map.csv"), relative(AUTHOR / "SingleCellsAnnotation.xlsx"),
                  relative(CACHE / "author_release.html"), relative(CACHE / "clinical_source_reconciliation.json")]))
    linkage = json.loads((CACHE / "sample_linkage.json").read_text())
    linkage.update(author_core_annotations=794, repair_round=1,
                   latest_evidence=relative(CACHE / "repaired_sample_linkage.json"),
                   clinical_source_reconciliation=relative(CACHE / "clinical_source_reconciliation.json"),
                   gaps=["11 samples/13 images absent from exact-release and cited upstream clinical tables",
                         "DSS endpoint/units confirmed; exact T baseline remains source-ambiguous"])
    write_json(CACHE / "sample_linkage.json", linkage)
    sources = json.loads((CACHE / "metadata_sources.json").read_text())
    sources.update(repair_metadata_sources=relative(CACHE / "repair_metadata_sources.json"),
                   upstream_clinical_sources=relative(CACHE / "survival_source_definitions.json"),
                   latest_sample_linkage=relative(CACHE / "repaired_sample_linkage.json"))
    write_json(CACHE / "metadata_sources.json", sources)
    summary = dict(study=STUDY, preparer=PREPARER, independent_review=False,
                   repair_round=1, expected_images=794, exported_images=794, sample_rows=718,
                   sample_linked_images=794, patient_linked_samples=718,
                   annotated_images=int(saved_images.region_annotation.notna().sum()),
                   annotation_flag_counts={k: core_map[k].value_counts(dropna=False).to_dict() for k in annotations},
                   all_input_file_size_mtime_unchanged=True, panel_bytes_unchanged=True,
                   unchanged_image_fields="All fields except region_annotation and metadata_notes",
                   unchanged_sample_fields="All fields except metadata_notes; no clinical values imputed",
                   clinical_source_reconciliation=relative(CACHE / "clinical_source_reconciliation.json"),
                   sample_linkage=relative(CACHE / "repaired_sample_linkage.json"),
                   input_versions=input_versions,
                   gaps=["11份sample/13个core在作者及上游临床表均缺失，临床字段保留空值",
                         "DSS事件/单位已确认；来源未明确T的精确起点，OS/PFS/DFS保持空值", "待独立复查I04及受影响metadata并完成D01绑定"])
    write_json(PROGRESS.parent / "metadata_repair_verification.json", summary)
    review = pd.read_csv(OUTPUT / "review_results.csv", dtype=str, keep_default_na=False)
    review = review[~(review.preparer.eq(PREPARER) & review.reviewer.eq(""))]
    checks = pd.read_csv(TEMPLATES / "review_checks.csv")
    for check in checks[checks.check_id.isin(["I04", "M01", "M02", "D01"])].itertuples(index=False):
        review.loc[len(review)] = dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                                       record_id=STUDY, check_id=check.check_id, result="unresolved",
                                       source_evidence=relative(CACHE / "repaired_core_map.csv"),
                                       output_evidence=relative(OUTPUT), details=json.dumps(dict(
                                           preparer_only=True, independent_review_pending=True, repair_round=1,
                                           repair_evidence=relative(PROGRESS.parent / "metadata_repair_verification.json"),
                                           sample_linked_images=794, sample_rows=718, patient_linked_samples=718,
                                           reused_image_review="0170_review_0 P03/I05; all TIFF size/mtime unchanged",
                                           clinical_gaps=summary["gaps"])))
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT):
        temporary = OUTPUT / "review_results.csv.tmp"
        review.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(OUTPUT / "review_results.csv")
    progress("metadata_repaired_pending_independent_review", 794, sample_rows=718,
             sample_linked_images=794, patient_linked_samples=718, annotated_images=794,
             gaps=summary["gaps"])
    print(json.dumps({k: v for k, v in summary.items() if k != "input_versions"}, ensure_ascii=False), flush=True)


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def progress(phase: str, completed: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, completed_images=completed,
                             expected_images=794, updated_at=datetime.now(timezone.utc).isoformat(),
                             **details))


def fetch_metadata() -> None:
    """Fetch small source records and selected original ZIP metadata only."""
    urls = {"data_record.json": "https://zenodo.org/api/records/5850952",
            "code_record.json": "https://zenodo.org/api/records/6036188",
            "paper.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7612730/fullTextXML"}
    for name, url in urls.items():
        if not (CACHE / name).exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            (CACHE / name).write_bytes(response.content)
    size = json.loads((CACHE / "code_record.json").read_text())["files"][0]["size"]
    url = "https://zenodo.org/records/6036188/files/MBTMEStrIMCPublic.zip?download=1"

    def byte_range(start: int, end: int) -> bytes:
        response = requests.get(url + f"&member_offset={start}",
                                headers={"Range": f"bytes={start}-{end}"}, timeout=60)
        response.raise_for_status()
        assert response.status_code == 206
        assert response.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
        return response.content

    inventory = CACHE / "author_archive_inventory.json"
    if not inventory.exists():
        start = size - 1048576
        with zipfile.ZipFile(io.BytesIO(byte_range(start, size-1))) as zipped:
            write_json(inventory, [dict(name=e.filename, size=e.file_size, compressed=e.compress_size,
                                       offset=e.header_offset+start) for e in zipped.infolist()])
    entries = json.loads(inventory.read_text())
    wanted = [e for e in entries if e["size"] and
              (e["name"].endswith("/markerStackOrder.csv") or "/Code/" in e["name"])]
    for entry in wanted:
        path = CACHE / "author_metadata" / entry["name"]
        if path.exists():
            continue
        data = byte_range(entry["offset"], entry["offset"]+entry["compressed"]+1023)
        header = struct.unpack("<4s5H3I2H", data[:30])
        assert header[0] == b"PK\x03\x04"
        start = 30 + header[-2] + header[-1]
        compressed = data[start:start+entry["compressed"]]
        values = zlib.decompress(compressed, -15) if header[3] == 8 else compressed
        assert len(values) == entry["size"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(values)
    write_json(CACHE / "metadata_sources.json", dict(
        urls=urls, author_archive=url, fetched_members=[e["name"] for e in wanted],
        sample_grouping="paper Methods + author ExtFig4 group multi-image tumours by metabric_id",
        clinical="SPORA tissues.parquet carries explicit original_patient_identifier and clinical fields",
        paper_supplements="Reporting summary and antibody/pixel-classifier Tables 1-2; no per-core clinical supplement listed",
        request_failure="One optional AbPanel.csv range returned HTTP 504 during discovery; stack order and author code obtained",
        large_images_downloaded=False))


def source_inventory() -> tuple[dict, pd.DataFrame, list[str]]:
    """Reuse native pixel/order proofs, checking all currently complete objects."""
    proof = json.loads(VERIFICATION.read_text())
    assert proof["full_source_covered"] and not proof["failed"] and proof["verified_count"] == 794
    evidence = {r["image_id"]: r for r in proof["verified"]}
    tissues = pd.read_parquet(RAW / "metadata/tissues.parquet")
    channels = pd.read_parquet(RAW / "imc/channels.parquet")
    names = channels.channel_name.tolist()
    stack = pd.read_csv(AUTHOR / "markerStackOrder.csv")
    assert stack.iloc[:, 1].tolist() == channels.original_channel_name.tolist()
    available = pd.read_parquet(metadata_source_directory(NAS, STUDY) / 'spora_inventory' / 'imc' / 'channels_per_tissue.parquet')
    assert available.columns.tolist() == names and bool(available.to_numpy().all())
    assert set(tissues.tissue_id) == set(evidence) == set(available.index)
    assert tissues.tissue_id.is_unique and tissues.original_file_name.is_unique
    archive = json.loads((CACHE / "author_archive_inventory.json").read_text())
    original_stacks = [e for e in archive if e["name"].endswith("_FullStack.tiff")]
    assert len(original_stacks) == 794
    assert {Path(e["name"]).name for e in original_stacks} == set(tissues.original_file_name)
    actual = {p.name.removesuffix(".ome.zarr") for p in (RAW / "imc/1_0mpp/images").iterdir()
              if p.is_dir() and p.name.endswith(".ome.zarr")}
    assert actual == set(evidence)
    objects = json.loads(OBJECTS.read_text())
    stats = []
    for entry in objects:
        suffix = entry["key"].split("spora-submissions/danenberg2022breast/", 1)[1]
        path = RAW / suffix
        stat = path.stat()
        assert stat.st_size == entry["size"], path
        stats.append(dict(path=relative(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    differences = {}
    for image_id, item in evidence.items():
        assert item["exact_canonical_order_checked"] and item["canonical_channel_labels"] == names
        group = json.loads((NAS / item["path"] / "zarr.json").read_text())["attributes"]["ome"]
        labels = [c["label"] for c in group["omero"]["channels"]]
        assert labels == item["embedded_labels"]
        assert group["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [1, 1, 1]
        array = json.loads((NAS / item["path"] / "0/zarr.json").read_text())
        assert array["shape"] == item["shape"] and array["data_type"] == item["dtype"] == "float32"
        differences[image_id] = item["embedded_label_differences"]
    write_json(CACHE / "source_scope.json", dict(
        study=STUDY, expected_images=794, source_objects=len(objects),
        complete_objects=stats, reused_verification=relative(VERIFICATION),
        canonical_order=names, author_stack_order=relative(AUTHOR / "markerStackOrder.csv"),
        author_archive_stacks=794, all_original_filenames_match_author_archive=True,
        embedded_label_differences=differences, all_source_objects_size_match=True,
        source_covered="All 794 complete local native 1 µm IMC cores; no source hash repeated",
        upper_uncompressed_output_bytes=sum(max(r["shape"][1:])**2 * 38 * 4 for r in evidence.values()),
        largest_decoded_roi_bytes=max(np.prod(r["shape"]).item()*4 for r in evidence.values())))
    return evidence, tissues, names


class NativeROI:
    """Decode one source core once; the full rectangular raster is measured."""
    def __init__(self, item: dict):
        array = zarr.open_array(str(NAS / item["path"] / "0"), mode="r")
        self.values = np.asarray(array[:])
        self.shape, self.dtype = self.values.shape, self.values.dtype
        assert list(self.shape) == item["shape"] and str(self.dtype) == item["dtype"]
        self.source = SourceSpec(item["image_id"], STUDY, "zarr", NAS / item["path"] / "0",
                                 tuple(range(self.shape[0])), pixel_size_um=(1.0, 1.0))

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.values[channels, y0:y1, x0:x1]

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return np.ones((y1-y0, x1-x0), dtype=bool)


def overview(reader: NativeROI, image_id: str) -> None:
    """Show the full ROI and both measured HER2 clones; scaling is display only."""
    layers = []
    for index, label in [(37, "DNA1"), (35, "panCK"), (12, "HER2 3B5"), (19, "HER2 D8F12")]:
        plane = reader.values[index]
        positive = plane[plane > 0]
        upper = float(np.percentile(positive, 99)) if len(positive) else 1
        rendered = Image.fromarray(np.uint8(np.clip(plane / max(upper, 1e-9), 0, 1)*255))
        rendered.thumbnail((420, 420))
        canvas = Image.new("RGB", (440, 455), "black")
        canvas.paste(rendered, ((440-rendered.width)//2, 30))
        ImageDraw.Draw(canvas).text((10, 8), label, fill="white")
        layers.append(canvas)
    page = Image.new("RGB", (880, 940), "black")
    for i, canvas in enumerate(layers):
        page.paste(canvas, ((i % 2)*440, 20+(i//2)*455))
    ImageDraw.Draw(page).text((10, 4), image_id, fill="white")
    page.save(CACHE / f"overview_{image_id}.png")


def assess_panel(evidence: dict, tissues: pd.DataFrame) -> None:
    """Select one duplicate measurement using source-linked native ROI signals.

    Both clones pass source QC and are used by the authors. Compare raw ROI
    means against source ERBB2_pos in 16 positive and 16 negative distinct
    patients, selected by tissue_id order, one ROI per patient. Choose larger
    rank AUC. This is panel QC, not clinical validation or input normalization.
    """
    path = CACHE / "duplicate_channel_assessment.json"
    if path.exists():
        return
    rows = []
    for status in [False, True]:
        subset = tissues[tissues.ERBB2_pos == status].sort_values("tissue_id").drop_duplicates("patient_id").head(16)
        for row in subset.itertuples(index=False):
            progress("assessing_duplicate_channels", 0, qc_completed=len(rows), current_image=row.tissue_id,
                     gaps=["待HER2重复测量选择", "待独立复核"])
            reader = NativeROI(evidence[row.tissue_id])
            rows.append(dict(image_id=row.tissue_id, patient_id=row.patient_id, erbb2_positive=status,
                             mean_3B5=float(reader.values[12].mean(dtype=np.float64)),
                             mean_D8F12=float(reader.values[19].mean(dtype=np.float64))))
            if len(rows) in {1, 17}:
                overview(reader, row.tissue_id)
            del reader
    measurements = pd.DataFrame(rows)
    auc = {}
    for name in ["3B5", "D8F12"]:
        positives = measurements.loc[measurements.erbb2_positive, f"mean_{name}"].to_numpy()
        negatives = measurements.loc[~measurements.erbb2_positive, f"mean_{name}"].to_numpy()
        auc[name] = float(((positives[:, None] > negatives).sum() + 0.5*(positives[:, None] == negatives).sum()) /
                          (len(positives)*len(negatives)))
    assert auc["3B5"] != auc["D8F12"], "Duplicate measurement choice needs further source QC"
    selected = max(auc, key=auc.get)
    write_json(path, dict(measurements=rows, raw_roi_mean_auc=auc, selected_clone=selected,
                         selected_source_index=12 if selected == "3B5" else 19,
                         source="channels.parquet describes clones; tissues.parquet ERBB2_pos",
                         rule="larger rank AUC of untransformed whole-ROI mean in 32 distinct patients",
                         limitation="preparer panel QC only; not a validated clinical classifier"))


def metadata(evidence: dict, tissues: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    choice = json.loads((CACHE / "duplicate_channel_assessment.json").read_text())
    selected_index = choice["selected_source_index"]
    panel = panel_from_s1(STUDY, PANEL, names, S1,
                          downstream_source_indices=set(range(39)) - ({12, 19} - {selected_index}))
    omitted = 19 if selected_index == 12 else 12
    panel.loc[omitted, "exclusion_reason"] = "Duplicate CD340 measurement; source-linked raw signal QC in duplicate_channel_assessment.json"
    panel.loc[selected_index, "keep_reason"] = "S1 admitted HER2 measurement; selected clone by source-linked raw signal QC"
    images = []
    for row in tissues.sort_values("tissue_id").itertuples(index=False):
        item = evidence[row.tissue_id]
        _, h, w = item["shape"]
        side = max(h, w)
        left, top = (side-w)//2, (side-h)//2
        record = dict.fromkeys(columns("image_metadata"))
        record.update(study=STUDY, image_id=row.tissue_id,
                      sample_id=f"primary_breast_{row.original_patient_identifier}", panel_id=PANEL,
                      image_kind="tma_core", source_image_id=row.tissue_id, source_files=[item["path"]],
                      source_format="zarr", source_axes="CYX", source_dtype="float32",
                      source_revision="SPORA 1_0mpp; source_scope.json and native verification 2026-10-21",
                      core_id=row.original_file_name.removesuffix("_FullStack.tiff"),
                      source_x0_px=0, source_y0_px=0, source_x1_px=w, source_y1_px=h,
                      pad_left_px=left, pad_top_px=top, pad_right_px=side-w-left, pad_bottom_px=side-h-top,
                      height_px=side, width_px=side, source_channel_count=39, channel_count=int(panel.keep_channel.sum()),
                      pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                      input_image_path=relative(OUTPUT / "images" / row.tissue_id / "input.tiff"),
                      intensity_description="Native SPORA decoded float32 IMC ion-count intensity; no added normalization, clipping, spillover correction or interpolation",
                      split_method="already_independent", split_evidence=relative(CACHE / "sample_linkage.json"),
                      metadata_notes=json.dumps(dict(original_file_name=row.original_file_name,
                          source_patient_id=row.patient_id, source_specimen_id=row.specimen_id,
                          source_split=row.split, used_in_publication=row.used_in_publication,
                          region_annotation="No per-core normal/DCIS label in local tissues table; used_in_publication is not interpreted as histology")))
        images.append(record)

    # The original primary tumour is the biological sample. SPORA specimen_id
    # enumerates cores and must not turn multiple cores into independent samples.
    assert (tissues.groupby("original_patient_identifier").patient_id.nunique() == 1).all()
    clinical = {"ERStatus": "source_er_status", "ERBB2_pos": "source_erbb2_positive",
                "LymphNodesOrdinal": "source_lymph_nodes_ordinal", "sizeOrdinal": "source_size_ordinal",
                "IntClust": "source_intclust", "yearsToStatus": "source_years_to_status",
                "DeathBreast": "source_death_breast", "isValidation": "source_is_validation"}
    samples = []
    for subject, group in tissues.groupby("original_patient_identifier", sort=True):
        record = dict.fromkeys(columns("sample_metadata"))
        record.update(study=STUDY, sample_id=f"primary_breast_{subject}", patient_key=f"METABRIC:{subject}",
                      source_subject_id=subject, cohort_id="METABRIC", source_sample_id=subject,
                      biospecimen_group_id=subject, sample_type="tissue", anatomical_site="Breast",
                      lesion_type="primary", treatment_status_at_collection="treatment-naive",
                      collection_method="resection", preservation_method="ffpe", processing_method="TMA; 4 µm section; IMC",
                      sex_reported="female", primary_diagnosis="Breast cancer", primary_site="Breast")
        for source, target in {"Grade": "grade", "PAM50": "molecular_subtype", **clinical}.items():
            values = group[source].dropna().unique()
            assert len(values) <= 1, (subject, source)
            record[target] = values[0].item() if len(values) and isinstance(values[0], np.generic) else (values[0] if len(values) else None)
        record["metadata_notes"] = json.dumps(dict(
            sample_basis="one treatment-naive primary breast excision; multiple TMA cores share sample",
            source_patient_id=group.patient_id.iloc[0], source_core_count=len(group),
            clinical_missing=group.metabric_id.isna().all().item(),
            survival="yearsToStatus/DeathBreast retained as source fields; OS/PFS/DFS left blank because endpoint/time-origin mapping is not established"))
        samples.append(record)
    samples = pd.DataFrame(samples)
    fields = []

    def mapping(source_file: Path, source_field: str, target_table: str, target_field: str,
                rule: str, notes: str = "", join: str = "tissue_id") -> None:
        record = dict.fromkeys(columns("field_mapping"), "")
        record.update(study=STUDY, source_file=relative(source_file), source_sheet_or_table=source_file.name,
                      source_field=source_field, target_table=target_table, target_field=target_field,
                      join_keys=join, transformation_rule=rule, conflict_rule="Unique nonmissing source value per biological sample; conflicting values halt metadata publication", notes=notes)
        fields.append(record)

    tissue_path = RAW / "metadata/tissues.parquet"
    mapping(tissue_path, "tissue_id; original_patient_identifier; specimen_id; original_file_name", "image_metadata", "sample_id",
            "Explicit tissue_id join; primary_breast_ + original_patient_identifier; repeated cores share primary tumour sample",
            "794 images -> 718 biological samples -> 718 source-identified patients. 11 patients/13 images lack imported clinical linkage; identity is still explicit.")
    mapping(tissue_path, "original_patient_identifier", "sample_metadata", "sample_id; patient_key; source_subject_id; source_sample_id; biospecimen_group_id",
            "Same original primary tumour across cores per paper Methods and author ExtFig4; patient_key is METABRIC: + explicit original_patient_identifier",
            "Do not infer identities from filenames; specimen_id is a core enumeration in this release", "original_patient_identifier")
    mapping(tissue_path, "Grade; PAM50", "sample_metadata", "grade; molecular_subtype",
            "Unique nonmissing Grade/PAM50 per original_patient_identifier; preserve source units/categories", join="original_patient_identifier")
    for source, target in clinical.items():
        mapping(tissue_path, source, "sample_metadata", target,
                "Preserve unique nonmissing source value without endpoint conversion", json.dumps(dict(
                    data_type="number" if source in {"yearsToStatus", "DeathBreast"} else ("boolean" if source in {"ERBB2_pos", "isValidation"} else "string"),
                    definition=f"Original source {source}; clinical annotation carried by SPORA",
                    unit="years" if source == "yearsToStatus" else "",
                    time_origin="not established for yearsToStatus; no OS/PFS/DFS substitution" if source == "yearsToStatus" else "source primary diagnosis")), join="original_patient_identifier")
    mapping(CACHE / "paper.xml", "Methods / Study design, TMA production and metadata; Tissue labeling and IMC",
            "sample_metadata", "cohort_id; sample_type; anatomical_site; lesion_type; treatment_status_at_collection; collection_method; preservation_method; processing_method; sex_reported; primary_diagnosis; primary_site",
            "Cohort-level source facts: women; primary breast excision before systemic therapy; FFPE; 4 µm TMA sections",
            "Core-level histological normal/DCIS classification is unavailable in local tissue table; sample tissue_type and image region_annotation remain null", "study")
    mapping(AUTHOR / "Code/ExtFig4/CellPhenotypeCompositionByMultiImageTumours.R", "metabric_id; ImageNumber; totalCores; tumourID",
            "sample_metadata", "metadata_notes", "Authors group multiple core images by original tumour/metabric_id; supports one primary specimen per source subject", join="original_patient_identifier / metabric_id")
    mapping(tissue_path, "original_file_name; patient_id; specimen_id; split; used_in_publication", "image_metadata", "core_id; metadata_notes",
            "core_id removes exact _FullStack.tiff suffix; other source values preserved in metadata_notes; split/publication flag do not filter image admission")
    mapping(RAW / "imc/channels.parquet", "channel_name; original_channel_name; description; qc_pass", "panel_mapping", "source_name; source_channel_index; keep_channel",
            "Official channel table row order confirmed against original markerStackOrder; embedded Zarr label differences retained in source_scope.json",
            "Both HER2 clones pass source QC; one downstream clone chosen using original measurements; omitted duplicate has no selected segmentation use", "source_channel_index")
    mapping(CACHE / "duplicate_channel_assessment.json", "selected_source_index; raw_roi_mean_auc", "panel_mapping", "keep_reason; exclusion_reason",
            "Keep the clone with greater raw ROI-mean rank AUC against source ERBB2_pos in the documented 32-patient QC subset", join="source_channel_index")
    mapping(S1, "Study; Original marker; Unified marker; Segmentation channel", "panel_mapping", "marker_id; mapping_status; role; nuclear; body; output_channel_index",
            "Exact study + official source name; preserve selected DNA1/DNA2 and body channels; output indices follow retained source order", join="Study + Original marker")
    mapping(CACHE / "source_scope.json", "native proof; array shape; scale", "image_metadata", "source_files; source_image_id; source_format; source_axes; source_dtype; source_revision; source_x0_px; source_y0_px; source_x1_px; source_y1_px; height_px; width_px; pad_left_px; pad_right_px; pad_top_px; pad_bottom_px; pixel_size_x_um; pixel_size_y_um; source_channel_count; channel_count; input_image_path; valid_pixels_path; intensity_description; image_kind; split_method; split_evidence",
            "Full measured rectangular core; no splitting; center-pad shorter axis to square; retain decoded dtype and intensity; 1 µm source scale; validity follows rectangular acquisition, not signal")
    mapping(CACHE / "metadata_sources.json", "public source search and attachment inventory", "sample_metadata", "metadata_notes",
            "Local metadata -> original publication -> author archive stack order/code; cache selected small metadata and complete archive member list",
            "11 samples lack imported clinical fields; survival endpoint origin and per-core normal/DCIS annotations remain unassigned", "study")
    write_json(CACHE / "sample_linkage.json", dict(
        sample_rows=len(samples), source_images=len(images), sample_linked_images=len(images),
        patient_linked_samples=int(samples.patient_key.notna().sum()), unique_patients=samples.patient_key.nunique(),
        grouping="Explicit original_patient_identifier -> one excised primary tumour; cores share sample, supported by Methods and author ExtFig4",
        keys=["tissue_id", "original_patient_identifier", "patient_id", "original_file_name"],
        clinical_missing_samples=int(tissues.groupby("original_patient_identifier").metabric_id.count().eq(0).sum()),
        evidence=[relative(tissue_path), relative(CACHE / "paper.xml"), relative(CACHE / "metadata_sources.json"),
                  relative(AUTHOR / "Code/ExtFig4/CellPhenotypeCompositionByMultiImageTumours.R")],
        gaps=["11 patients/13 cores lack imported clinical linkage", "No per-core normal/DCIS map in current tissue metadata", "Survival time origin/endpoint mapping not established; raw source values retained"] ))
    return pd.DataFrame(images), samples, panel, pd.DataFrame(fields, columns=columns("field_mapping"))


def preparer_results(images: pd.DataFrame, samples: pd.DataFrame) -> None:
    """Record handoff facts without impersonating an independent reviewer."""
    rows = []
    for check in pd.read_csv(TEMPLATES / "review_checks.csv").itertuples(index=False):
        rows.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=check.scope,
                         record_id=PANEL if check.scope == "panel" else STUDY, check_id=check.check_id,
                         result="unresolved", source_evidence=relative(CACHE / "source_scope.json"),
                         output_evidence=relative(OUTPUT), details=json.dumps(dict(
                             preparer_only=True, independent_review_pending=True,
                             exported_images=len(images), expected_images=794,
                             sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                             patient_linked_samples=int(samples.patient_key.notna().sum()),
                             sample_evidence=relative(CACHE / "sample_linkage.json"),
                             pixel_evidence=relative(CACHE / "preparer_verification.json"),
                             clinical_gaps="11 samples lack imported clinical fields; survival origin and per-core normal/DCIS annotations unresolved"))))
    frame = pd.DataFrame(rows, columns=columns("review_results"))
    path = OUTPUT / "review_results.csv"
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        old = old[~((old.preparer == PREPARER) & (old.reviewer == ""))]
        frame = pd.concat([old, frame], ignore_index=True)
    with formal_publish_lease(path, directory=path.parent):
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def verify_outputs(images: pd.DataFrame, panel: pd.DataFrame, evidence: dict) -> None:
    """Reopen all TIFF headers and compare all kept channels in source blocks."""
    selected = panel[panel.keep_channel].sort_values("output_channel_index")
    headers = []
    for row in images.itertuples(index=False):
        _, h, w = evidence[row.image_id]["shape"]
        side = max(h, w)
        assert (row.source_x0_px, row.source_y0_px, row.source_x1_px, row.source_y1_px) == (0, 0, w, h)
        assert (row.pad_left_px, row.pad_top_px, row.pad_right_px, row.pad_bottom_px) == (
            (side-w)//2, (side-h)//2, side-w-(side-w)//2, side-h-(side-h)//2)
        assert row.height_px == row.width_px == side
        with tifffile.TiffFile(NAS / row.input_image_path) as tif:
            names = [c.get("Name") for c in ET.fromstring(tif.ome_metadata).iter() if c.tag.endswith("}Channel")]
            assert names == selected.marker_id.tolist()
            assert tif.series[0].shape == (len(selected), row.height_px, row.width_px)
            assert len(tif.pages) == len(selected) and all(p.dtype == np.dtype(row.source_dtype) for p in tif.pages)
            pixels = next(c for c in ET.fromstring(tif.ome_metadata).iter() if c.tag.endswith("}Pixels"))
            assert float(pixels.get("PhysicalSizeX")) == float(pixels.get("PhysicalSizeY")) == 1.0
        headers.append(row.image_id)
    subsets = [images[images.pad_left_px > 0], images[images.pad_top_px > 0],
               images.sort_values("height_px"), images.sort_values("height_px", ascending=False)]
    ids = list(dict.fromkeys(frame.iloc[0].image_id for frame in subsets if len(frame)))
    checked = []
    for image_id in ids:
        row = images.set_index("image_id").loc[image_id]
        reader = NativeROI(evidence[image_id])
        _, h, w = reader.shape
        windows = [(max(0, w//2-32), max(0, h//2-32), min(w, w//2+32), min(h, h//2+32)),
                   (0, 0, min(w, 64), min(h, 64))]
        with tifffile.TiffFile(NAS / row.input_image_path) as tif:
            for x0, y0, x1, y1 in windows:
                target = (x0+int(row.pad_left_px), y0+int(row.pad_top_px), x1+int(row.pad_left_px), y1+int(row.pad_top_px))
                for channel in selected.itertuples(index=False):
                    actual = read_tiff_window(tif, int(channel.output_channel_index), target)
                    expected = reader.values[channel.source_channel_index, y0:y1, x0:x1]
                    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
            margins = [(0, 0, int(row.pad_left_px), row.height_px),
                       (row.width_px-int(row.pad_right_px), 0, row.width_px, row.height_px),
                       (0, 0, row.width_px, int(row.pad_top_px)),
                       (0, row.height_px-int(row.pad_bottom_px), row.width_px, row.height_px)]
            for margin in margins:
                if margin[2] > margin[0] and margin[3] > margin[1]:
                    for channel in selected.output_channel_index:
                        assert not read_tiff_window(tif, int(channel), margin).any()
        overview(reader, image_id)
        del reader
        checked.append(dict(image_id=image_id, source_xyxy_blocks=windows,
                            channels=len(selected), exact_pixels=True, padding="all four margins/all kept channels"))
    write_json(CACHE / "preparer_verification.json", dict(
        study=STUDY, preparer=PREPARER, independent_review=False, headers_checked=len(headers),
        pixel_samples=checked, result="pass", sample_linkage=relative(CACHE / "sample_linkage.json"),
        limitation="Preparer observations; independent review/D01 binding pending"))


def verify_saved_metadata(images: pd.DataFrame, tissues: pd.DataFrame) -> None:
    """Compare delivered sample identities and clinical values to source rows."""
    saved = pd.read_csv(OUTPUT / "sample_metadata.csv", dtype=str, keep_default_na=False)
    assert saved.sample_id.is_unique and len(saved) == 718
    by_sample = saved.set_index("sample_id")
    sources = tissues.set_index("tissue_id")
    for row in images.itertuples(index=False):
        subject = sources.loc[row.image_id, "original_patient_identifier"]
        assert row.sample_id == f"primary_breast_{subject}"
        assert by_sample.loc[row.sample_id, "patient_key"] == f"METABRIC:{subject}"
    clinical = {"Grade": "grade", "PAM50": "molecular_subtype", "ERStatus": "source_er_status",
                "ERBB2_pos": "source_erbb2_positive", "LymphNodesOrdinal": "source_lymph_nodes_ordinal",
                "sizeOrdinal": "source_size_ordinal", "IntClust": "source_intclust",
                "yearsToStatus": "source_years_to_status", "DeathBreast": "source_death_breast",
                "isValidation": "source_is_validation"}
    compared = 0
    for subject, group in tissues.groupby("original_patient_identifier"):
        row = by_sample.loc[f"primary_breast_{subject}"]
        assert row.source_subject_id == row.source_sample_id == row.biospecimen_group_id == subject
        for source, target in clinical.items():
            values = group[source].dropna().unique()
            if len(values):
                expected = values[0]
                if source in {"Grade", "yearsToStatus", "DeathBreast"}:
                    assert float(row[target]) == float(expected), (subject, source)
                else:
                    assert row[target] == str(expected), (subject, source)
            else:
                assert row[target] == "", (subject, source)
            compared += 1
        assert all(row[field] == "" for field in ["os_time", "os_event", "pfs_time", "pfs_event", "dfs_time", "dfs_event"])
    write_json(CACHE / "preparer_metadata_verification.json", dict(
        preparer=PREPARER, independent_review=False, sample_rows=len(saved),
        patient_linked_samples=int(saved.patient_key.ne("").sum()),
        sample_linked_images=len(images), clinical_values_and_missingness_compared=compared,
        result="pass", source=relative(RAW / "metadata/tissues.parquet")))


# Prepare sources and identities first; publish only images with completed receipts.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fetch", action="store_true")
parser.add_argument("--repair-sources", action="store_true")
parser.add_argument("--repair-sources-light", action="store_true")
parser.add_argument("--repair-download", action="store_true")
parser.add_argument("--survival-sources", action="store_true")
parser.add_argument("--repair-metadata", action="store_true")
parser.add_argument("--assess-panel", action="store_true")
parser.add_argument("--metadata", action="store_true")
parser.add_argument("--export", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--limit", type=int)
args = parser.parse_args()
CACHE.mkdir(parents=True, exist_ok=True)
if args.repair_download:
    fetch_repair_sources(download_only=True)
    raise SystemExit(0)
if args.survival_sources:
    fetch_survival_sources()
    if not args.repair_metadata:
        raise SystemExit(0)
if args.repair_metadata:
    repair_metadata()
    raise SystemExit(0)
if args.repair_sources:
    fetch_repair_sources()
    if not (args.fetch or args.assess_panel or args.metadata or args.export or args.verify):
        raise SystemExit(0)
if args.repair_sources_light:
    fetch_repair_sources(include_cells=False)
    raise SystemExit(0)
if args.fetch:
    fetch_metadata()
    if not (args.assess_panel or args.metadata or args.export or args.verify):
        raise SystemExit(0)
evidence, tissues, names = source_inventory()
if args.assess_panel:
    assess_panel(evidence, tissues)
    if not (args.metadata or args.export or args.verify):
        raise SystemExit(0)
images, samples, panel, fields = metadata(evidence, tissues, names)
receipt_path = CACHE / "export_receipts.json"
receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
gaps = ["11份sample缺少导入临床字段", "逐core正常/DCIS注释缺失", "生存起点及终点语义未映射", "待独立复核"]
if args.export:
    pending = images[~images.image_id.isin(receipts)]
    if args.limit is not None:
        pending = pending.iloc[:args.limit]
    for row in pending.itertuples(index=False):
        if shutil.disk_usage(NAS).free < 100*1024**3 or shutil.disk_usage("/tmp").free < 64*1024**3:
            raise RuntimeError("Insufficient reserved disk capacity")
        progress("exporting", sum(r["admitted"] for r in receipts.values()), current_image=row.image_id, gaps=gaps)
        reader = NativeROI(evidence[row.image_id])
        result = export_input(reader, panel, NAS / row.input_image_path)
        if len(receipts) in {0, 200, 400, 600, 793}:
            overview(reader, row.image_id)
        del reader
        result["image_id"] = row.image_id
        result["source_path"] = evidence[row.image_id]["path"]
        receipts[row.image_id] = result
        write_json(receipt_path, receipts)
        admitted = {key for key, value in receipts.items() if value["admitted"]}
        delivered = images[images.image_id.isin(admitted)]
        if (OUTPUT / "images.parquet").exists():
            old = pd.read_parquet(OUTPUT / "images.parquet")
            additional = old[~old.image_id.isin(delivered.image_id)]
            if not additional.empty:
                delivered = pd.concat([additional, delivered], ignore_index=True)
        save_input_tables(OUTPUT, delivered, samples, panel, fields)
        progress("exporting", len(delivered), processed_sources=len(receipts), sample_rows=len(samples),
                 sample_linked_images=int(delivered.sample_id.notna().sum()),
                 patient_linked_samples=int(samples.patient_key.notna().sum()), gaps=gaps)
        print(json.dumps(dict(exported_images=len(delivered), image_id=row.image_id, admitted=result["admitted"])), flush=True)
if args.export or args.metadata:
    admitted = {key for key, value in receipts.items() if value["admitted"]}
    delivered = images[images.image_id.isin(admitted)]
    if (OUTPUT / "images.parquet").exists():
        old = pd.read_parquet(OUTPUT / "images.parquet")
        additional = old[~old.image_id.isin(images.image_id)]
        if not additional.empty:
            delivered = pd.concat([additional, delivered], ignore_index=True)
    save_input_tables(OUTPUT, delivered, samples, panel, fields)
    preparer_results(delivered, samples)
    progress("export_complete" if len(receipts) == 794 else "partial_export", len(delivered),
             processed_sources=len(receipts), sample_rows=len(samples),
             sample_linked_images=int(delivered.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()), gaps=gaps)
if args.verify:
    delivered = pd.read_parquet(OUTPUT / "images.parquet")
    progress("preparer_verification", len(delivered), sample_rows=len(samples), gaps=gaps)
    verify_saved_metadata(delivered, tissues)
    verify_outputs(delivered, panel, evidence)
    progress("preparer_verified", len(delivered), processed_sources=len(receipts), sample_rows=len(samples),
             sample_linked_images=int(delivered.sample_id.notna().sum()),
             patient_linked_samples=int(samples.patient_key.notna().sum()), gaps=gaps)
