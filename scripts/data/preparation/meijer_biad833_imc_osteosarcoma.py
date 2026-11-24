"""Export the 135 verified native S-BIAD833 MCD acquisitions.

Inputs: five MCDs, preserved acceptance lineage, public sample workbook and S1.
Outputs: native intensity TIFFs and the five per-study input tables.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

from contextlib import contextmanager
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from xml.etree import ElementTree

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables


STUDY = "meijer_biad833_imc_osteosarcoma"
ROOT = Path("/mnt/NAS_21T/ProjectData/CellWeft")
RAW = ROOT / "raw/public" / STUDY / "S-BIAD833"
RAW_METADATA = ROOT / "raw/public" / STUDY / "metadata"
META = metadata_source_directory(ROOT, STUDY)
PROOFS = META / "input_facts/native_snapshots"
CLOSEOUT = META / "input_facts/native_closeout_134.json"
FOLLOWUP = META / "input_facts/public_source_followup.json"
AUDIT = META / "input_facts/audit_0816.json"
OUTPUT = ROOT / "processed" / STUDY
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(Path(__file__).resolve().parents[3] / "cache/study_input_runtime" / STUDY / "0746_prepare_0")))
WORKBOOK = META / "evidence/worker/query/q03_metadata_xlsx.body"
S1 = ROOT / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
REPAIR = META / "source_record_corrections.json"
PANEL_ID = "imc_49"
PREPARER = JOB.name
SCIENTIFIC_REPAIR_ROUND = int(os.environ.get("CELLWEFT_SCIENTIFIC_REPAIR_ROUND", "0"))


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, ROOT))


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def progress(stage: str, indexed: int, next_image: str | None, *,
             priority: str = "normal", **details: object) -> None:
    payload = dict(study=STUDY, stage=stage, expected_images=135, completed_images=indexed,
                   next_image=next_image, scientific_repair_round=SCIENTIFIC_REPAIR_ROUND,
                   updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    JOB.mkdir(parents=True, exist_ok=True)
    path = JOB / "progress.json"
    with formal_publish_lease(path, directory=JOB, data_root=ROOT, priority=priority):
        temporary = path.with_name("progress.json.partial")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)


def mapping(source: Path, source_field: str, table: str, target: str,
            rule: str, join: str = "", notes: str = "") -> dict[str, str]:
    return dict(study=STUDY, source_file=relative(source), source_sheet_or_table="Sheet1" if source == WORKBOOK else "",
                source_field=source_field, target_table=table, target_field=target,
                join_keys=join, value_mapping="{}", transformation_rule=rule,
                time_origin_rule="", conflict_rule="Ambiguous identity remains empty",
                notes=notes)


def inventory() -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = META / "native_manifest.json"
    with formal_read_lease(manifest_path, data_root=ROOT):
        manifest = json.loads(manifest_path.read_text())
        proofs = [json.loads((PROOFS / f"{entry['source_id']}.json").read_text())
                  for entry in manifest["sources"]]
        workbook = pd.read_excel(WORKBOOK, sheet_name="Sheet1")
    # The preserved snapshots cover 117 acquisitions and retain their original
    # incomplete scope. Later acceptance admitted 17, followed by the one-record
    # ROI_057 correction. Reuse those facts without creating new payload proofs.
    with formal_read_lease(CLOSEOUT, data_root=ROOT):
        closeout = json.loads(CLOSEOUT.read_text())
        repair = json.loads((META / "native_view_repairs.json").read_text())
    accepted = {item["canonical_roi_id"]: item for proof in proofs for item in proof["acquisitions"]}
    if (len(accepted) != 117 or len(closeout["new_accepted"]) != 17 or
            closeout["accepted"] != 134 or closeout["accepted_proof_entries_preserved"] != 117 or
            repair["accepted_proofs_preserved"] != 134 or repair["new_image_count"] != 135):
        raise ValueError("Unexpected preserved acceptance lineage")
    later = set(closeout["new_accepted"])
    corrected = repair["applied_acquisition"]
    if later & accepted.keys() or corrected["canonical_roi_id"] in (later | accepted.keys()):
        raise ValueError("Overlapping acceptance lineage")
    accepted[corrected["canonical_roi_id"]] = corrected
    scope = set(accepted) | later
    rows = []
    source_names = None
    with formal_read_lease(RAW, data_root=ROOT):
        for entry, proof in zip(manifest["sources"], proofs):
            path = RAW / entry["source"]["name"]
            stat = path.stat()
            if (stat.st_size != entry["source"]["size"] or proof["source"] != entry["source"] or
                    {key: getattr(stat, key) for key in ("st_size", "st_mtime_ns", "st_ino")} !=
                    {"st_size": proof["native_stat"]["size"], "st_mtime_ns": proof["native_stat"]["mtime_ns"],
                     "st_ino": proof["native_stat"]["inode"]}):
                raise ValueError(f"Native MCD changed: {path}")
            for native in proof["native_inventory"]:
                key = native["canonical_roi_id"]
                if key not in scope:
                    continue
                fact = accepted.get(key)
                if native["derived_split"] or (fact is not None and
                        (not fact["all_finite"] or not fact["required_payload_complete"] or
                         fact["native_metadata"] != native["native_metadata"])):
                    raise ValueError(f"Invalid accepted source metadata: {key}")
                names = native["ordered_channels"]
                if source_names is None:
                    source_names = names
                if names != source_names or len(names) != 49:
                    raise ValueError(f"Unexpected ordered panel: {key}")
                rows.append(dict(source=path, source_id=entry["source_id"],
                                 proof=fact if fact is not None else {"acceptance_record": relative(CLOSEOUT)},
                                 native=native, image_id=f"{entry['source_id']}__{native['native_metadata']['Description']}"))
    with formal_read_lease(OUTPUT / "images.parquet", data_root=ROOT):
        current_ids = set(pd.read_parquet(OUTPUT / "images.parquet", columns=["image_id"]).image_id)
    if len(rows) != 135 or len(scope) != 135 or {row["image_id"] for row in rows} != current_ids:
        raise ValueError("Acceptance lineage differs from the current 135-image scope")
    with formal_read_lease(S1, data_root=ROOT):
        panel = panel_from_s1(STUDY, PANEL_ID, source_names, S1)
    named = workbook.loc[workbook.ROI.notna()].copy()
    named["source_image_id"] = named["MCD file"] + "_ROI_" + named.ROI.astype(int).astype(str).str.zfill(3)
    if named.source_image_id.duplicated().any() or workbook.groupby("L_ID").pat_ID.nunique().gt(1).any():
        raise ValueError("Ambiguous public workbook identity")
    return rows, panel, workbook, named


class CorrectedMCDReader:
    """Read ROI_057 with the accepted one-record official correction in memory."""

    def __init__(self, path: Path, native: dict, correction: dict):
        self.source = SourceSpec(native["canonical_roi_id"], PANEL_ID, "mcd", path,
                                 tuple(range(49)), acquisition_id=native["acquisition_id"],
                                 pixel_size_um=(1.0, 1.0), source_revision="official single-record view")
        self.dtype = np.dtype("float32")
        self.shape = (49, native["height"], native["width"])
        metadata = native["native_metadata"]
        start, end = int(metadata["DataStartOffset"]), int(metadata["DataEndOffset"])
        self.records = np.memmap(path, dtype="<f4", mode="r", offset=start,
                                 shape=((end-start)//(52*4), 52))
        index = int(correction["record_index"])
        if (start + index*208 != correction["byte_offset"] or
                self.records[index].tobytes().hex() != correction["original_record_hex"]):
            raise ValueError("Original ROI_057 record differs from accepted correction")
        self.corrected = np.frombuffer(bytes.fromhex(correction["official_record_hex"]),
                                       dtype="<f4").copy()
        x = self.records[:, 0].astype(np.int64)
        y = self.records[:, 1].astype(np.int64)
        x[index], y[index] = int(self.corrected[0]), int(self.corrected[1])
        if (np.any(x < 0) or np.any(y < 0) or np.any(x >= self.shape[2]) or
                np.any(y >= self.shape[1])):
            raise ValueError("Corrected ROI_057 coordinates outside source image")
        self.indices = np.full(self.shape[1:], -1, dtype=np.int64)
        self.indices[y, x] = np.arange(len(x))
        if (int((self.indices >= 0).sum()) != len(x) or len(x) != self.shape[1]*self.shape[2]):
            raise ValueError("Corrected ROI_057 has missing or duplicate pixels")
        self.corrected_index = index

    def read_window(self, channels: list[int], window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        indices = self.indices[y0:y1, x0:x1]
        result = np.stack([self.records[indices, channel+3] for channel in channels])
        if np.any(indices == self.corrected_index):
            for output, channel in enumerate(channels):
                result[output, indices == self.corrected_index] = self.corrected[channel+3]
        return result

    def valid_window(self, window: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = window
        return self.indices[y0:y1, x0:x1] >= 0

    def close(self) -> None:
        self.records._mmap.close()


@contextmanager
def source_reader(item: dict):
    if item["image_id"] == "LCCO_OS_1__ROI_057":
        with formal_read_lease(REPAIR, data_root=ROOT):
            correction = json.loads(REPAIR.read_text())
        reader = CorrectedMCDReader(item["source"], item["native"], correction)
        try:
            yield reader
        finally:
            reader.close()
    else:
        spec = SourceSpec(item["image_id"], PANEL_ID, "mcd", item["source"],
                          tuple(range(49)), acquisition_id=item["native"]["acquisition_id"],
                          pixel_size_um=(1.0, 1.0), source_revision="S-BIAD833 accepted native proof")
        with open_source(spec) as reader:
            yield reader


def sample_tables(workbook: pd.DataFrame, named: pd.DataFrame, items: list[dict]):
    samples = []
    lesion_types = {"Primary": "primary", "Metastasis": "metastasis",
                    "Local recurrence": "recurrence"}
    for _, group in workbook.groupby("L_ID", sort=False):
        first = group.iloc[0]
        row = dict.fromkeys(columns("sample_metadata"), "")
        icr = group.ICR.dropna()
        row.update(study=STUDY, sample_id=str(first.L_ID), patient_key=str(first.pat_ID),
                   source_subject_id=str(first.pat_ID), source_sample_id=str(first.L_ID),
                   sample_type="tissue", tissue_type="tumor",
                   lesion_type=lesion_types[str(first.Sample_type)], anatomical_site=str(first.Location),
                   primary_diagnosis="Osteosarcoma", preservation_method="ffpe",
                   source_icr=str(icr.iloc[0]) if len(icr) else "",
                   metadata_notes="Public workbook Sheet1; source Sample_type mapped to lesion_type. Optional clinical values absent.")
        samples.append(row)
    by_image = dict(zip(named.source_image_id, named.L_ID.astype(str)))
    slide_rows = workbook.loc[workbook.ROI.isna()].set_index("MCD file")
    links = {}
    for item in items:
        slide = item["source_id"]
        source_key = slide + "_" + item["native"]["description"]
        sample_id = by_image.get(source_key)
        if sample_id is None and slide in slide_rows.index:
            sample_id = str(slide_rows.loc[slide, "L_ID"])
        links[item["image_id"]] = sample_id
    fields = [
        mapping(PROOFS / "LCCO_OS_1.json", "canonical_roi_id;native_metadata;ordered_channels", "image_metadata", "image_id;source_acquisition_id;source_x0_px;source_y0_px;source_x1_px;source_y1_px", "Preserved native inventory plus 134-image closeout and ROI057 correction; current scope is checked against images.parquet", notes="Acceptance lineage: " + relative(CLOSEOUT) + ";" + relative(META / "native_view_repairs.json")),
        mapping(S1, "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping", "marker_id;role;nuclear;body;keep_channel", "Exact study-specific ordered S1 selection"),
        mapping(WORKBOOK, "MCD file;ROI;L_ID", "image_metadata", "sample_id", "Exact MCD basename plus ROI number; for three slide-level rows ROI is blank and all native ROI of that MCD share its specimen", "MCD file + ROI", "80 exact ROI links and 7 slide-level ROI links; remaining native ROI lack workbook sample mapping."),
        mapping(WORKBOOK, "L_ID;pat_ID", "sample_metadata", "sample_id;patient_key;source_subject_id;source_sample_id", "Preserve published specimen and patient keys; do not infer unlisted patients", "L_ID"),
        mapping(WORKBOOK, "Sample_type;Location", "sample_metadata", "lesion_type;anatomical_site", "Map Primary/Metastasis/Local recurrence to primary/metastasis/recurrence; preserve literal location", "L_ID"),
        mapping(WORKBOOK, "ICR", "sample_metadata", "source_icr", "Preserve the published immunologic constant of rejection category as a source value; blank where unpublished", "L_ID",
                json.dumps({"data_type": "string", "definition": "Published ICR category for the biological specimen"})),
        mapping(META / "evidence/study_metadata.body", "Description;Biosample", "sample_metadata", "primary_diagnosis;preservation_method", "Study-level osteosarcoma diagnosis; map source FFPE to vocabulary ffpe; no patient-specific clinical inference"),
        mapping(META / "evidence/study_metadata.body", "Description;Biosample", "sample_metadata", "sample_type;tissue_type", "Archival osteosarcoma tissue specimens: sample_type=tissue; tissue_type=tumor. Preserve specific diagnosis in primary_diagnosis."),
    ]
    return pd.DataFrame(samples, columns=[*columns("sample_metadata"), "source_icr"]), pd.DataFrame(fields, columns=columns("field_mapping")), links


def header_matches(path: Path, panel: pd.DataFrame, side: int) -> bool:
    kept = panel.loc[panel.keep_channel].sort_values("output_channel_index")
    with tifffile.TiffFile(path) as tif:
        names = [element.attrib["Name"] for element in ElementTree.fromstring(tif.ome_metadata).iter()
                 if element.tag.endswith("}Channel")]
        return (len(tif.pages) == len(kept) and names == kept.marker_id.tolist() and
                all(page.shape == (side, side) and page.dtype == np.dtype("float32") for page in tif.pages))


def review(indexed: int, linked: int, sample_count: int, excluded: list[str]) -> None:
    path = OUTPUT / "review_results.csv"
    rows = [dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                 record_id=STUDY, check_id="S01", result="unresolved",
                 source_evidence=";".join(relative(PROOFS / f"{source}.json") for source in
                                          ("LCCO_OS_1", "LCCO_OS_2", "20210811_L7211",
                                           "20211008_L7177", "20211011_L3676")) + ";" + relative(CLOSEOUT) + ";" + relative(META / "native_view_repairs.json"),
                 output_evidence=relative(OUTPUT / "images.parquet"),
                 details=f"Preparer: 135 qualified native acquisitions; {indexed} indexed; "
                         f"{len(excluded)} no-marker-signal exclusions: {', '.join(excluded) or 'none'}. "
                         "Independent source-scope review pending."),
            dict(study=STUDY, preparer=PREPARER, reviewer="", scope="sample",
                 record_id=STUDY, check_id="M01", result="unresolved",
                 source_evidence=relative(WORKBOOK) + ";" + relative(META / "native_manifest.json"),
                 output_evidence=relative(OUTPUT / "sample_metadata.csv"),
                 details=f"Preparer: {indexed}/135 images indexed; {linked} image-to-sample links, {sample_count} sample-to-patient links. Unlisted ROI identities remain empty; independent review pending."),
            dict(study=STUDY, preparer=PREPARER, reviewer="", scope="study",
                 record_id=STUDY, check_id="D01", result="unresolved",
                 source_evidence=relative(CLOSEOUT) + ";" + relative(META / "native_view_repairs.json"),
                 output_evidence=relative(OUTPUT / "images.parquet"),
                 details=f"Preparer export {indexed}/135; independent S01/I01-I05/P01-P03/M01-M02 review and binding pending.")]
    incoming = pd.DataFrame(rows, columns=columns("review_results"))
    with formal_read_lease(path, data_root=ROOT):
        prior = pd.read_csv(path, keep_default_na=False) if path.is_file() else incoming.iloc[:0]
    prior = prior.loc[~((prior.preparer == PREPARER) & (prior.reviewer == "") &
                       prior.check_id.isin(["S01", "M01", "D01"]))]
    merged = pd.concat([prior, incoming], ignore_index=True)
    with formal_publish_lease(path, directory=OUTPUT, data_root=ROOT):
        temporary = path.with_name("review_results.csv.partial")
        merged.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


def remove_obsolete_mapping() -> None:
    path = OUTPUT / "field_mapping.csv"
    with formal_read_lease(path, data_root=ROOT):
        prior = pd.read_csv(path, keep_default_na=False)
        version = (path.stat().st_size, path.stat().st_mtime_ns)
    obsolete = ((prior.study == STUDY) & (prior.source_file == relative(WORKBOOK)) &
                (prior.source_field == "Sample_type;Location") &
                (prior.target_field == "sample_type;anatomical_site"))
    if not obsolete.any():
        return
    cleaned = prior.loc[~obsolete]
    with formal_publish_lease(path, directory=OUTPUT, data_root=ROOT):
        if (path.stat().st_size, path.stat().st_mtime_ns) != version:
            raise RuntimeError("Field mapping changed during correction")
        temporary = path.with_name("field_mapping.csv.partial")
        cleaned.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)


def repair_metadata() -> None:
    """Repair the reviewed vocabulary and provenance without rewriting pixels."""
    if shutil.disk_usage(ROOT).free < 100*1024**3 + 100*1024**2 or shutil.disk_usage("/tmp").free < 64*1024**3 + 100*1024**2:
        progress("waiting_io", 135, None, next_step="repair_metadata", reason="Metadata repair needs 100 MiB above configured reserves")
        return
    items, _, workbook, named = inventory()
    rebuilt, fields, links = sample_tables(workbook, named, items)
    followup = FOLLOWUP
    with formal_read_lease(followup, data_root=ROOT):
        followup_evidence = json.loads(followup.read_text())
    fields = pd.concat([fields, pd.DataFrame([
        mapping(followup, "released workbook;article;PDF S1/S2;author repository checks", "image_metadata", "sample_id",
                "Retain 80 exact ROI and 7 slide-level links. PDF lacks native MCD/ROI bridge; preserve 48 empty sample links.",
                "MCD file + ROI", "27 sample rows; 27 patient-linked samples; 7 distinct published patient keys. Repository access failures remain documented gaps."),
        mapping(RAW_METADATA / "supplement.zip#bmjonc-3-1-s001.pdf", "Table S1;Table S2", "sample_metadata", "metadata_notes",
                "S1 clinical values are cohort aggregates; S2 uses publication sample labels without an exact native ROI bridge. Do not assign aggregate clinical values or infer cross-namespace specimen identity.",
                notes="PDF pages 3-4; optional individual clinical fields remain empty. BioImage Archive and paper cohort descriptions have different stated scopes.")
    ])], ignore_index=True)
    with formal_read_lease(OUTPUT, data_root=ROOT):
        images = pd.read_parquet(OUTPUT / "images.parquet")
        samples = pd.read_csv(OUTPUT / "sample_metadata.csv", keep_default_na=False, dtype=str)
        panel = pd.read_csv(OUTPUT / "panel_mapping.csv", keep_default_na=False)
        prior = pd.read_csv(OUTPUT / "review_results.csv", keep_default_na=False)
        audit = json.loads(AUDIT.read_text())
        versions = {entry["image_id"]: entry["revision"] for entry in audit["headers"]}
        for row in images.itertuples():
            stat = (ROOT / row.input_image_path).stat()
            expected = versions[row.image_id]
            if (stat.st_size, stat.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
                raise ValueError(f"Reviewed input version changed: {row.image_id}")
            if (row.sample_id if pd.notna(row.sample_id) else None) != links[row.image_id]:
                raise ValueError(f"Reviewed sample link changed: {row.image_id}")
    # Verify every unaffected sample value against the source-derived table.
    unchanged = [name for name in samples if name not in ("tissue_type", "preservation_method")]
    left = samples.set_index("sample_id").sort_index()
    right = rebuilt.set_index("sample_id").sort_index()
    pd.testing.assert_frame_equal(left[[name for name in unchanged if name != "sample_id"]],
                                  right[[name for name in unchanged if name != "sample_id"]], check_dtype=False)
    samples["tissue_type"] = "tumor"
    samples["preservation_method"] = "ffpe"
    save_input_tables(OUTPUT, images, samples, panel, fields, data_root=ROOT)

    details = dict(scientific_repair_round=SCIENTIFIC_REPAIR_ROUND,
                   action="Corrected 27 tissue_type and preservation_method values; added sample_type/tissue_type provenance and explicit FFPE-to-ffpe rule",
                   sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()),
                   patient_linked_samples=int(samples.patient_key.ne("").sum()),
                   unique_patient_keys=int(samples.patient_key.nunique()),
                   reused_reviewed_input_images=len(versions),
                   public_followup=relative(FOLLOWUP),
                   new_sample_links_proven=followup_evidence["new_sample_links_proven"],
                   conclusion="Preparer repair complete; independent metadata recheck and new D01 binding pending. 48 unlinked ROI remain unresolved.")
    incoming = []
    for check, scope in (("M01", "sample"), ("M02", "sample"), ("D01", "study")):
        incoming.append(dict(study=STUDY, preparer=PREPARER, reviewer="", scope=scope,
                             record_id=STUDY, check_id=check, result="unresolved",
                             source_evidence=relative(FOLLOWUP),
                             output_evidence=relative(OUTPUT / "sample_metadata.csv") + ";" + relative(OUTPUT / "field_mapping.csv"),
                             details=json.dumps(details)))
    # Append preparer evidence; independent reviewer records remain intact.
    prior = prior.loc[~((prior.preparer == PREPARER) & (prior.reviewer == "") &
                        prior.check_id.isin(["M01", "M02", "D01"]))]
    with formal_publish_lease(OUTPUT / "review_results.csv", directory=OUTPUT, data_root=ROOT):
        temporary = OUTPUT / "review_results.csv.partial"
        pd.concat([prior, pd.DataFrame(incoming)], ignore_index=True).to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, OUTPUT / "review_results.csv")
    progress("prepared", len(images), None, **{key: value for key, value in details.items() if key != "scientific_repair_round"},
             next_step="independent metadata review and D01 rebinding")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-metadata", action="store_true")
    if parser.parse_args().repair_metadata:
        repair_metadata()
        return
    items, panel, workbook, named = inventory()
    samples, fields, links = sample_tables(workbook, named, items)
    index = OUTPUT / "images.parquet"
    with formal_read_lease(index, data_root=ROOT):
        prior = pd.read_parquet(index) if index.is_file() else pd.DataFrame()
    existing = {row["image_id"]: row.to_dict() for _, row in prior.iterrows()}
    if set(existing) - {item["image_id"] for item in items}:
        raise ValueError("Existing index contains an image outside accepted native scope")
    with formal_read_lease(OUTPUT, data_root=ROOT):
        for item in items:
            if item["image_id"] not in existing:
                continue
            destination = OUTPUT / "images" / item["image_id"] / "input.tiff"
            side = max(item["native"]["height"], item["native"]["width"])
            if not destination.is_file() or not header_matches(destination, panel, side):
                raise ValueError(f"Previously indexed TIFF changed: {item['image_id']}")
    pending_rows = []

    for position, item in enumerate(items):
        key = item["image_id"]
        native = item["native"]
        side = max(native["height"], native["width"])
        destination = OUTPUT / "images" / key / "input.tiff"
        if key in existing:
            continue
        predicted = side*side*int(panel.keep_channel.sum())*4
        if shutil.disk_usage(ROOT).free < 100*1024**3 + 2*predicted or shutil.disk_usage("/tmp").free < 64*1024**3 + 2*predicted:
            if pending_rows:
                save_input_tables(OUTPUT, pd.DataFrame(pending_rows), samples, panel, fields, data_root=ROOT)
            progress("waiting_io", len(existing), key, reason="NAS 100 GiB or local 64 GiB reserve", sample_rows=len(samples),
                     sample_linked_images=sum(bool(row["sample_id"]) for row in existing.values()), patient_linked_samples=len(samples))
            return
        checkpoint = Path(os.environ.get("CELLWEFT_INPUT_STAGING_ROOT", "/tmp/study_input_batch")) / "publish_checkpoints" / metadata_source_relative(destination.resolve(), '/')
        with formal_read_lease(destination, data_root=ROOT):
            if destination.is_file() and not header_matches(destination, panel, side):
                raise ValueError(f"Unindexed published TIFF header differs: {key}")
            published_unindexed = destination.is_file() and not checkpoint.exists()
            valid_published = destination.with_name("valid_pixels.tiff").is_file() if published_unindexed else False
            if valid_published:
                with tifffile.TiffFile(destination.with_name("valid_pixels.tiff")) as tif:
                    if len(tif.pages) != 1 or tif.pages[0].shape != (side, side) or tif.pages[0].dtype != np.dtype("uint8"):
                        raise ValueError(f"Unindexed valid-pixel TIFF header differs: {key}")
        if published_unindexed:
            left, top = (side-native["width"])//2, (side-native["height"])//2
            result = dict(admitted=True, valid_pixels="valid_pixels.tiff" if valid_published else None,
                          pad_left_px=left, pad_top_px=top,
                          pad_right_px=side-native["width"]-left,
                          pad_bottom_px=side-native["height"]-top,
                          height_px=side, width_px=side, source_channel_count=49,
                          channel_count=int(panel.keep_channel.sum()))
        else:
            with source_reader(item) as reader:
                result = export_input(reader, panel, destination, tile_size=256, data_root=ROOT)
        if not result["admitted"]:
            progress("excluded_no_marker_signal", len(existing), key)
            continue
        with formal_read_lease(destination, data_root=ROOT):
            if not header_matches(destination, panel, side):
                raise ValueError(f"Published TIFF header differs: {key}")
        row = dict.fromkeys(columns("image_metadata"))
        row.update(study=STUDY, image_id=key, sample_id=links[key], panel_id=PANEL_ID,
                   image_kind="roi", source_image_id=native["description"],
                   source_files=[relative(item["source"])], source_format="mcd",
                   source_acquisition_id=str(native["acquisition_id"]), source_axes="CYX",
                   source_dtype="float32", source_revision="S-BIAD833 accepted native proof",
                   slide_id=item["source_id"], roi_id=native["description"],
                   region_annotation=native["description"], source_x0_px=0, source_y0_px=0,
                   source_x1_px=native["width"], source_y1_px=native["height"],
                   pixel_size_x_um=1.0, pixel_size_y_um=1.0,
                   input_image_path=relative(destination),
                   valid_pixels_path=relative(destination.with_name("valid_pixels.tiff")) if result["valid_pixels"] else None,
                   intensity_description="Native float32 MCD Dual measurement values; no normalization",
                   split_method="already_independent", split_evidence=relative(PROOFS / f"{item['source_id']}.json"),
                   metadata_notes=("Public workbook specimen linked by exact ROI or slide-level file." if links[key] else
                                   "No public MCD/ROI-to-biospecimen row; sample and patient unknown.") +
                   (" ROI_057 uses accepted read-only official single-record correction." if key == "LCCO_OS_1__ROI_057" else ""),
                   **{name: result[name] for name in ("pad_left_px", "pad_top_px", "pad_right_px",
                                                       "pad_bottom_px", "height_px", "width_px",
                                                       "source_channel_count", "channel_count")})
        existing[key] = row
        pending_rows.append(row)
        progress("published_unindexed", len(existing)-len(pending_rows), key, next_step="merge published images into index")
        if len(pending_rows) == 4:
            save_input_tables(OUTPUT, pd.DataFrame(pending_rows), samples, panel, fields, data_root=ROOT)
            pending_rows.clear()
            progress("batch_saved", len(existing), items[position+1]["image_id"] if position+1 < len(items) else None,
                     sample_rows=len(samples), sample_linked_images=sum(bool(existing[k]["sample_id"]) for k in existing),
                     patient_linked_samples=len(samples))
            print(f"indexed {len(existing)}/135 through {key}", flush=True)

    if not existing:
        raise ValueError("No admitted native ROI")
    if pending_rows:
        save_input_tables(OUTPUT, pd.DataFrame(pending_rows), samples, panel, fields, data_root=ROOT)
    save_input_tables(OUTPUT, pd.DataFrame(existing.values()), samples, panel, fields, data_root=ROOT)
    remove_obsolete_mapping()
    excluded = [item["image_id"] for item in items if item["image_id"] not in existing]
    review(len(existing), sum(bool(row["sample_id"]) for row in existing.values()),
           len(samples), excluded)
    progress("prepared", len(existing), None, sample_rows=len(samples),
             sample_linked_images=sum(bool(row["sample_id"]) for row in existing.values()),
             patient_linked_samples=len(samples), next_step="independent review")


if __name__ == "__main__":
    main()
