"""Prepare native CODEX inputs from the 12 verified human Zenodo acquisitions.

Inputs: existing native proofs, QuPath channel order, publication clinical table,
author CODEX/Visium key and sample/region table, and study-specific S1 rows. Outputs: independent CYX
input.tiff and complete study metadata. Clinical records remain on NAS.
Run --run; --overview creates source-pyramid contact sheets under a read slot.
The preparer's evidence does not constitute independent acceptance.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import imagecodecs
import tifffile

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import TEMPLATES, export_input, panel_from_s1, save_input_tables

STUDY = "Greenwald2024_CODEX_Glioblastoma"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
OUTPUT = NAS / "processed" / STUDY
CACHE = OUTPUT / "preparation_evidence"
TEMP = Path("/tmp/study_input_batch") / STUDY
BATCH = CACHE
JOB = Path(os.environ.get("CELLWEFT_STUDY_INPUT_JOB", str(BATCH / "jobs" / STUDY / "0211_prepare_0")))
PROGRESS = JOB / "progress.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
SCRIPT = Path(__file__).resolve()
PREPARER = "codex_greenwald_" + JOB.name
RAW_STUDY = NAS / "raw/public" / STUDY
RAW_MANIFEST = RAW_STUDY / "manifest.json"
AUTHOR_INPUTS = META / "author_inputs/Inputs/Codex"
AUTHOR_REPOSITORY = META / "author_repository"
PUBLICATION = META / "publication"
GEO = META / "geo"
AUTHOR_IDENTITY = META / "source_facts/author_cells_identity_fields.csv"
PUBLICATION_TEXT = META / "source_facts/mmc1.txt"
SOURCE_SCOPE = META / "source_facts/source_scope.json"
SOURCE_FAILURES = META / "source_facts/source_failures"
SOURCE_SCOPE_OUTPUT = CACHE / "source_scope.json"
SOURCE_FAILURE_OUTPUT = CACHE / "source_failures"
CHANNEL_DECISIONS = META / "source_facts/channel_decisions.json"


def relative(path: Path) -> str:
    return str(metadata_source_relative(path, NAS))


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def source_failure_records() -> list[tuple[Path, dict]]:
    directories = (SOURCE_FAILURES, SOURCE_FAILURE_OUTPUT)
    paths = {path.name: path for directory in directories for path in directory.glob("*.json")}
    return [(path, json.loads(read_source_bytes(path))) for path in sorted(paths.values())]


def read_source_bytes(path: Path) -> bytes:
    if path.is_relative_to(META):
        return path.read_bytes()
    if not path.is_relative_to(NAS):
        raise ValueError(f"Greenwald2024 reader input is outside the NAS root: {path}")
    if path.is_relative_to(RAW_STUDY):
        with raw_source_read(path):
            return path.read_bytes()
    with formal_read_lease(path, data_root=NAS):
        return path.read_bytes()


@contextmanager
def raw_source_read(path: Path):
    if not path.is_relative_to(RAW_STUDY):
        raise ValueError(f"Greenwald2024 raw source is outside the study root: {path}")
    with formal_read_lease(path, data_root=NAS):
        with resource_lock(RAW_STUDY, shared=True, name=".raw_consolidation.lock"):
            yield path


def template(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv").columns.tolist()


def progress(stage: str, **extra) -> None:
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame()
    samples = pd.read_csv(OUTPUT / "sample_metadata.csv") if (OUTPUT / "sample_metadata.csv").exists() else pd.DataFrame()
    state = dict(study=STUDY, stage=stage, expected_images=12, completed_images=len(images),
                 exported_images=len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()) if len(images) else 0,
                 patient_linked_samples=int(samples.patient_key.notna().sum()) if len(samples) else 0,
                 updated_utc=datetime.now(timezone.utc).isoformat())
    if len(samples):
        state['unresolved_patient_samples']=samples.loc[samples.patient_key.isna(),'sample_id'].tolist()
    if (CACHE/'scope_reconciliation.json').exists():
        state['panel_deferred_images']=[x['image_id'] for x in load_json(CACHE/'scope_reconciliation.json')['sources'] if x['status']=='deferred_panel_measurement_identity']
    state['source_decode_deferred_images']=sorted({record['image_id'] for _, record in source_failure_records()})
    state.update(extra)
    save_json(PROGRESS, state)


def source_scope() -> list[dict]:
    """Reuse bound native proofs; streaming export still verifies full decodability.

    ZH811_1_B exposed invalid, zero-filled LZW segments despite the earlier proof.
    Record such failures separately; neither original files nor journals are repaired.
    """
    proof = load_json(META / "acquisition_verification.json")
    mapping = {x["file_key"]: x for x in load_json(META / "ordered_mapping.json")["images"]}
    manifest = {x["file_key"]: x for x in load_json(META / "source_manifest.json")["sources"]}
    cached={x['image_id']:x for x in json.loads(read_source_bytes(SOURCE_SCOPE))} if SOURCE_SCOPE.exists() else {}
    scope = []
    assert proof["scope_complete"] and proof["actual_validated_count"] == 12
    for item in proof["verified_images"]:
        path = NAS / item["source_rel_path"]
        with raw_source_read(path):
            stat = path.stat()
            assert stat.st_size == item["file_stat"]["size"] and stat.st_mtime_ns == item["file_stat"]["mtime_ns"]
            assert item["ordered_labels"] == mapping[item["file_key"]]["ordered_labels"]
            image_id=item['file_key'].removesuffix('.ome.tif')
            revision=f"zenodo:12624860/{item['file_key']};size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
            if image_id in cached and cached[image_id]['source_revision']==revision and cached[image_id]['labels']==item['ordered_labels']:
                scope.append(cached[image_id])
                continue
            with tifffile.TiffFile(path) as tif:
                assert list(tif.series[0].shape) == item["native_shape"]
                pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
                scale = [float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])]
        scope.append(dict(image_id=item["file_key"].removesuffix(".ome.tif"),
                          source_path=item["source_rel_path"], labels=item["ordered_labels"],
                          project_name=mapping[item["file_key"]]["project_image_name"],
                          shape=item["native_shape"], pixel_size_um=scale,
                          marker_order_basis=item["marker_order_basis"],
                          source_revision=revision,
                          source_order=item["source_order"], source_url=manifest[item["file_key"]]["source_url"]))
    if not cached:
        save_json(SOURCE_SCOPE_OUTPUT, scope)
    return scope


def build_samples(scope: list[dict]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Join author specimens to Table S1, retaining patient conflicts.

    Author cells_table.RDS has sample/region fields but no patient identifier;
    its extracted identity fields confirm all 12 QuPath specimen labels.
    """
    author=pd.read_csv(BytesIO(read_source_bytes(AUTHOR_IDENTITY)),dtype=str)
    assert author['sample'].is_unique and set(author['sample'])=={x['project_name'] for x in scope}
    author_regions=dict(zip(author['sample'],author['region']))
    text=read_source_bytes(PUBLICATION_TEXT).decode("utf-8")
    cohort = text.split("This cohort", 1)[1].split("External cohort", 1)[0]
    header = next(line for line in cohort.splitlines() if line.startswith("Sample-ID"))
    records = []
    for line in cohort.splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        if len(fields) >= 9 and fields[0] not in {"Sample-ID"}:
            entry = dict(zip(["Sample-ID", "Patient-ID", "Region", "Tumor Histology", "Grade", "MGMT-status", "Location", "Sex", "Age"], fields[:9]))
            entry['Visium'] = line[header.index('Visium'):header.index('CODEX')].strip()
            entry['CODEX'] = line[header.index('CODEX'):].strip()
            records.append(entry)
    clinical = pd.DataFrame(records)
    clinical.to_csv(CACHE / "publication_table_s1.csv", index=False, encoding="utf-8-sig")
    lookup = {re.sub(r"[^a-z0-9]", "", row["Sample-ID"].lower()): row for row in records}
    r = read_source_bytes(AUTHOR_REPOSITORY / "Module6_CODEX.R").decode("utf-8").split("key_sample <-", 1)[1].split("# Import data", 1)[0]
    left, right = r.split("visium = c(", 1)
    pairs = dict(zip(re.findall(r'"([^\"]+)"', left), re.findall(r'"([^\"]+)"', right)))
    rows, links = [], {}
    for item in scope:
        name = item["project_name"]
        key = pairs.get(name)
        row_source = lookup.get(re.sub(r"[^a-z0-9]", "", key.lower())) if key else None
        if key and row_source is None and key.lower().endswith('bulk'):
            row_source = lookup.get(re.sub(r"[^a-z0-9]", "", key[:-4].lower()))
        if row_source is None:
            # Only exact section suffixes are stripped; no patient prefix is inferred.
            base = re.sub(r"_B$|_v6$", "", name)
            row_source = lookup.get(re.sub(r"[^a-z0-9]", "", base.lower()))
        sample = dict.fromkeys(template("sample_metadata"))
        sample.update(study=STUDY, sample_id=name, source_sample_id=name,
                      sample_type="fresh frozen CODEX tissue section", tissue_type="tumor",
                      collection_method="resection", preservation_method="fresh_frozen",
                      primary_diagnosis="Glioblastoma", primary_site="Brain",
                      case_control_status="case")
        if name.startswith('ZH'):
            sample['center_id']='University Hospital Zurich'
        elif name.startswith('MGH'):
            sample['center_id']='Massachusetts General Hospital'
        notes = ["Independent CODEX specimen in author Module6_CODEX.R levels_samples and Zenodo 12624860.",
                 "Author cells_table.RDS sample/region identity confirmed; no patient/donor/subject identifier is present in that table."]
        region = None
        if row_source:
            region = row_source["Region"]
            patient = row_source["Patient-ID"]
            sample.update(source_sample_id=row_source["Sample-ID"], histology=row_source["Tumor Histology"],
                          grade=row_source["Grade"], anatomical_site=row_source["Location"],
                          sex_reported={"M": "male", "F": "female"}[row_source["Sex"]],
                          reported_age_years=int(row_source["Age"]),
                          molecular_subtype=row_source["Tumor Histology"].split(" GBM")[0])
            notes.append(f"Publication Table S1 Sample-ID={row_source['Sample-ID']}; MGMT-status={row_source['MGMT-status']}.")
            notes.append("Table S1 Age is reported in years without a defined collection time point; age_at_collection_years is withheld.")
            if not row_source["Sample-ID"].startswith(patient):
                notes.append("Patient-ID conflicts with specimen identity and article Figure 1 / participant details; patient_key and source_subject_id withheld pending resolution.")
            else:
                sample.update(patient_key=patient, source_subject_id=patient)
            notes.append("Connected using explicit Module6_CODEX.R key_sample" if key else "Connected by exact QuPath specimen label after terminal view suffix removal")
        else:
            notes.append("No verified publication clinical-row correspondence; patient and clinical values withheld. Table S1 CODEX bulk entry conflicts with the released source label.")
        if region is None or region in {'NA',''}:
            region=author_regions[name]
            notes.append('Source region label from author cells_table.RDS: '+region+'.')
        sample["metadata_notes"] = " ".join(notes)
        rows.append(sample)
        links[item["image_id"]] = dict(sample_id=name, region_annotation=region, metadata_notes=sample["metadata_notes"])
    return pd.DataFrame(rows, columns=template("sample_metadata") + ["reported_age_years"]), links


def build_panels(scope: list[dict]) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    decisions = json.loads(read_source_bytes(CHANNEL_DECISIONS)) if CHANNEL_DECISIONS.exists() else {}
    frames, identities, blocked = {}, {}, []
    variants={}
    for item in scope:
        signature=tuple(item['labels'])
        if signature not in variants:variants[signature]=len(variants)+1
    for item in scope:
        names = item["labels"]
        repeated = [i for i, name in enumerate(names) if name.startswith("S100B")]
        drops = {}
        if len(repeated) > 1:
            # Table S3 lists S100B at cycle 13, without per-image channel indices.
            # Native S100B positions differ between source panels; no cross-panel reassignment.
            choice = decisions.get(item["image_id"])
            if choice is None:
                blocked.append(item["image_id"])
                continue
            drops = {i: choice["evidence"] for i in repeated if i != choice["selected_source_channel_index"]}
        identity = (tuple(names), tuple(drops))
        if identity not in frames:
            panel_id = f"CODEX_P{variants[tuple(names)]:02d}"
            if drops:
                panel_id+=f"_S{choice['selected_source_channel_index']+1:02d}"
            frames[identity] = panel_from_s1(STUDY, panel_id, names, S1, drop_indices=drops)
        identities[item["image_id"]] = frames[identity].panel_id.iloc[0]
    return pd.concat(frames.values(), ignore_index=True), identities, blocked


def field_mapping(images: pd.DataFrame | None = None, samples: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    def add(source, source_field, table, target, keys, rule, notes):
        rows.append(dict(study=STUDY, source_file=source, source_sheet_or_table="", source_field=source_field,
                         target_table=table, target_field=target, join_keys=keys, value_mapping="{}",
                         transformation_rule=rule, time_origin_rule="", conflict_rule="Retain source conflict in metadata_notes; withhold unresolved identity", notes=notes))
    add(relative(META / "acquisition_verification.json"), "verified_images", "image_metadata", "source_files", "file_key", "Reuse size/mtime-bound native proof; full source ROI; no source pixel or partial modification", "12 human acquisitions; organoid and H&E outside accepted CODEX scope")
    add(relative(META / "ordered_mapping.json"), "ordered_labels; marker_order_provenance", "panel_mapping", "source_name", "file_key; source_channel_index", "Use corrected verified order; native OME labels take precedence over QuPath when explicitly proven", "ZH811_INF C40 is P2Y12, not legacy QuPath S100B; source and output indices remain separate")
    add(relative(S1), "Study; Original marker", "panel_mapping", "marker_id", "Study; Original marker", "Exact study-specific S1 join; selected nuclear/body channels retained", "Duplicate choices require per-source measurement evidence in channel_decisions.json")
    add(relative(AUTHOR_REPOSITORY / "Module6_CODEX.R"), "levels_samples; key_sample", "sample_metadata", "sample_id", "QuPath project_image_name=key_sample.codex", "Connect explicit CODEX/Visium pair to publication Sample-ID; retain original CODEX identity as sample_id", "Author source code; Zenodo release confirms 12 fresh frozen CODEX GBM specimens")
    add(relative(AUTHOR_INPUTS / "Inputs/Codex/cells_table.RDS"), "sample; region; complete column set", "sample_metadata", "sample_id", "Author sample=QuPath project_image_name", "Cross-check all 12 explicit CODEX specimen identities using the retained identity-only fact", "Original author attachment is retained in raw; no patient/donor/subject identifier is assigned from this table")
    add(relative(PUBLICATION_TEXT), "Table S1: Sample-ID; Patient-ID; Region; Tumor Histology; Grade; MGMT-status; Location; Sex; Age", "sample_metadata", "patient_key", "key_sample.visium=normalized Table S1 Sample-ID", "Copy supported clinical values; map F/M to female/male; no survival or treatment inference", "Exact text extraction from the retained original supplementary PDF; existing patient conflicts and missing matches are preserved")
    add(relative(PUBLICATION / "mmc1.pdf"), "Table S1", "sample_metadata", "patient_key", "mmc1.txt extraction", "Original publication supplement for the processed text fact", "Original supplementary attachment retained under raw")
    add(relative(PUBLICATION_TEXT), "Table S1 Region", "image_metadata", "region_annotation", "sample_id", "Copy original region label for explicitly connected specimens", "bulk/T1/inf/nec are source collection-region labels; no annotation inferred from image filename")
    for source_field,target,rule in [
        ('Sample-ID','source_sample_id','Use explicit author CODEX/Visium correspondence; unmatched specimen retains its CODEX source identity'),
        ('Patient-ID','source_subject_id','Copy only consistent publication Patient-ID; conflicting rows remain empty'),
        ('Tumor Histology','histology','Copy original tumor/histology description'),
        ('Tumor Histology','molecular_subtype','Retain the reported IDH status'),
        ('Grade','grade','Copy source grade without assigning a grading-system version'),
        ('Location','anatomical_site','Copy original anatomical location'),
        ('Sex','sex_reported','F -> female; M -> male'),
        ('Age','age_at_collection_years','Leave empty: Table S1 does not define the age time point as collection'),
        ('Age','reported_age_years','Copy reported integer age in years without assigning a collection, index, or diagnosis time point'),
        ('MGMT-status','metadata_notes','Retain original methylation category without collapsing partially methylated'),
    ]:
        notes = 'Only explicitly linked clinical rows are populated; no clinical value assigned to the unmatched source specimen'
        if target == 'reported_age_years':
            notes = json.dumps(dict(data_type='number', definition='Age in years as reported in Table S1; measurement time point unspecified', unit='years', time_origin='unknown'), ensure_ascii=False)
        add(relative(PUBLICATION_TEXT),f'Table S1 {source_field}','sample_metadata',target,'Explicit linked Table S1 Sample-ID',rule,notes)
        if source_field == 'Age':
            rows[-1]['time_origin_rule'] = 'Source age time point is unspecified; do not infer collection age'
    add(relative(PUBLICATION / "article.html"), "Experimental model and study participant details; Sample Preparation", "sample_metadata", "collection_method", "study", "All original human CODEX samples underwent tumor resection and fresh freezing", "Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11088502/; article attachment retained under raw")
    for target,value in [
        ('sample_type','fresh frozen CODEX tissue section'),
        ('tissue_type','tumor'),
        ('preservation_method','fresh_frozen'),
        ('primary_diagnosis','Glioblastoma'),
        ('primary_site','Brain'),
        ('case_control_status','case'),
    ]:
        add(relative(PUBLICATION/'article.html'),'Human GBM cohort and CODEX sample preparation',
            'sample_metadata',target,'study; released human CODEX specimen',
            'Assign cohort value supported by the original human GBM release: '+value,
            'The 12 human CODEX specimens are fresh-frozen resected GBM brain tissue sections; organoid is outside this input scope')
    add(relative(PUBLICATION/'article.html'), 'Experimental model: explicit ZH/MGH source-center definitions', 'sample_metadata', 'center_id', 'Source specimen label and article center definition', 'ZH samples -> University Hospital Zurich; MGH samples -> Massachusetts General Hospital', 'Center-prefix correspondence is explicitly defined by the original article; it is not used to derive patient identity.')
    add(relative(GEO/'GSE237183_family.soft'), 'GSM titles and characteristics', 'sample_metadata', 'metadata_notes', 'Author key_sample paired Visium specimen', 'Inspect original study GEO specimen annotations; do not derive patient identity from a specimen title', 'GEO confirms distinct ZH1019 regional specimen titles but supplies no explicit patient key; ZH916 bulk is not explicitly paired to released CODEX ZH916_2_B. Identity conflicts remain unresolved.')
    add(relative(AUTHOR_INPUTS/'Inputs/Codex/cells_table.RDS'),'region','image_metadata','region_annotation','Author sample=QuPath project_image_name','Copy the unique author region label when the publication region is unannotated or its clinical row is unmatched','Preserve existing publication region labels for the other specimens; region identity does not establish patient identity')
    for failure_path, _ in source_failure_records():
        add(relative(failure_path),'native retained-channel decode failure','image_metadata','source_files','image_id; source_revision','Pause only this image; no zero substitution or repair of original source','Existing size/mtime-bound native proof did not establish full-resolution decodability; see recorded channel/window and native segment evidence')
    if images is not None and samples is not None:
        rows[3]['notes'] += f"; actual coverage: {len(samples)} specimen rows; {len(images)} exported images; {int(images.sample_id.notna().sum())} image-to-sample links; {int(samples.patient_key.notna().sum())} sample-to-patient links. Unresolved specimens: "+','.join(samples.loc[samples.patient_key.isna(),'sample_id'])
    return pd.DataFrame(rows, columns=template("field_mapping"))


def current_field_mapping(images: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Carry the current reviewed mapping rows through source-only migrations."""
    path = OUTPUT / "field_mapping.csv"
    if not path.exists():
        return field_mapping(images, samples)
    with formal_read_lease(path, data_root=NAS):
        with resource_lock(OUTPUT, shared=True):
            return pd.read_csv(path, dtype=str, keep_default_na=False)


def publish_tables(scope: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict, list]:
    samples, links = build_samples(scope)
    panels, identities, blocked = build_panels(scope)
    images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=template("image_metadata"))
    if len(images):
        for index, image in images.iterrows():
            for key, value in links[image.image_id].items():
                images.at[index, key] = value
            images.at[index,'panel_id']=identities[image.image_id]
    save_input_tables(OUTPUT, images, samples, panels, current_field_mapping(images, samples))
    return images, panels, identities, blocked


def repair_age_metadata() -> None:
    """Move the reviewed Table S1 age values to an untimed source field."""
    with formal_read_lease(OUTPUT):
        images = pd.read_parquet(OUTPUT / 'images.parquet')
        samples = pd.read_csv(OUTPUT / 'sample_metadata.csv')
        panels = pd.read_csv(OUTPUT / 'panel_mapping.csv')
    if 'reported_age_years' not in samples:
        samples['reported_age_years'] = pd.NA
    affected = samples.age_at_collection_years.notna()
    if int(affected.sum()) == 11 and samples.reported_age_years.isna().all():
        samples['reported_age_years'] = pd.to_numeric(samples.age_at_collection_years).astype('Int64')
        samples.loc[affected, 'age_at_collection_years'] = pd.NA
        samples.loc[affected, 'metadata_notes'] += ' Table S1 Age is reported in years without a defined collection time point; age_at_collection_years is withheld.'
    else:
        assert not affected.any() and int(samples.reported_age_years.notna().sum()) == 11
        assert int(samples.metadata_notes.str.contains('time point; age_at_collection_years is withheld', regex=False).sum()) == 11
    samples['reported_age_years'] = pd.to_numeric(samples.reported_age_years).astype('Int64')
    save_input_tables(OUTPUT, images, samples, panels, current_field_mapping(images, samples))
    progress('age_metadata_repaired', repaired_sample_rows=11,
             next_step='independent M02 review and D01 rebinding')


def overviews(scope: list[dict]) -> None:
    """Inspect low-resolution source pyramids without decoding whole WSI channels."""
    sheet = Image.new("RGB", (4*420, 3*490), "white")
    stats = []
    for k, item in enumerate(scope):
        with raw_source_read(NAS / item["source_path"]) as source_path:
            with tifffile.TiffFile(source_path) as tif:
                level = tif.series[0].levels[-1]
                selected = [0, 17, 18]
                array = np.stack([level.pages[i].asarray(maxworkers=2) for i in selected])
                rgb = np.moveaxis(array[[2, 1, 0]], 0, -1)
                rgb = np.clip(rgb.astype(float) * (255 / np.maximum(np.percentile(rgb, 99, axis=(0, 1)), 1)), 0, 255).astype('uint8')
                picture = Image.fromarray(rgb)
                picture.thumbnail((410, 440))
                x,y=(k%4)*420,(k//4)*490
                sheet.paste(picture,(x,y+35));ImageDraw.Draw(sheet).text((x+5,y+5),item['image_id'],fill='black')
                duplicate = []
                for i, name in enumerate(item['labels']):
                    if name.startswith('S100B'):
                        data=level.pages[i].asarray(maxworkers=2)
                        duplicate.append(dict(source_channel_index=i, source_name=name, nonzero=int(np.count_nonzero(data)),mean=float(data.mean()),p99=float(np.percentile(data,99)),max=int(data.max())))
                stats.append(dict(image_id=item['image_id'],pyramid_shape=list(level.shape),S100B=duplicate))
    sheet.save(CACHE/'source_overviews.png')
    save_json(CACHE/'source_pyramid_measurements.json',stats)
    duplicate_sheet = Image.new('RGB', (1200, 3*450), 'white')
    group = 0
    for item in scope:
        indices=[i for i,n in enumerate(item['labels']) if n.startswith('S100B')]
        if len(indices)!=2:continue
        with raw_source_read(NAS/item['source_path']) as source_path:
            with tifffile.TiffFile(source_path) as tif:
                level=tif.series[0].levels[-1]
                for column,index in enumerate([0]+indices):
                    array=level.pages[index].asarray(maxworkers=2)
                    picture=Image.fromarray(np.clip(array.astype(float)*255/max(float(np.percentile(array,99.9)),1),0,255).astype('uint8')).convert('RGB')
                    picture.thumbnail((390,400))
                    duplicate_sheet.paste(picture,(column*400,group*450+35))
                    ImageDraw.Draw(duplicate_sheet).text((column*400+5,group*450+5),f"{item['image_id']} C{index+1}: {item['labels'][index]}",fill='black')
        group+=1
    duplicate_sheet.save(CACHE/'duplicate_channel_overviews.png')


def export_one(item: dict, panel: pd.DataFrame, links: dict) -> None:
    """Stage one source, stream bounded native reads, and publish one resumable index."""
    image_id = item['image_id']
    destination = OUTPUT/'images'/image_id/'input.tiff'
    receipt_path = CACHE/'exports'/f'{image_id}.json'
    _, h, w=item['shape'];side=max(h,w)
    selected=panel[panel.keep_channel].sort_values('output_channel_index')
    selection=dict(source_indices=selected.source_channel_index.tolist(),marker_names=selected.marker_id.tolist())
    reuse=destination.exists() and receipt_path.exists()
    if reuse:
        geometry=load_json(receipt_path)
        assert geometry['source_revision']==item['source_revision'] and geometry['selection']==selection
        print(f'reuse {image_id}',flush=True)
    source_path=NAS/item['source_path']
    with raw_source_read(source_path):
        source_size = source_path.stat().st_size
    required=item['shape'][0]*side*side+source_size
    if not reuse and (shutil.disk_usage(NAS).free < 100*1024**3+required or shutil.disk_usage(TEMP).free <64*1024**3+required):
        raise RuntimeError(f'Insufficient reserved capacity for {image_id}: {required} bytes upper estimate')
    if not reuse:
        progress('exporting',current_image=image_id,estimated_uncompressed_bytes=required)
        local=TEMP/(image_id+'.ome.tif');TEMP.mkdir(parents=True,exist_ok=True)
        with raw_source_read(source_path):
            shutil.copyfile(source_path,local)
        source=SourceSpec(image_id,panel.panel_id.iloc[0],'tiff',local,tuple(range(item['shape'][0])),pixel_size_um=tuple(item['pixel_size_um']))
        with open_source(source) as reader:
            native_read=reader.read_window
            def read_with_context(channels, window):
                try:
                    return native_read(channels,window)
                except imagecodecs.LzwError as error:
                    failure_path=SOURCE_FAILURE_OUTPUT/f'{image_id}.json'
                    failure_record=dict(image_id=image_id,source_path=item['source_path'],
                        source_revision=item['source_revision'],source_channel_indices=channels,
                        source_xyxy=list(window),error_type=type(error).__name__,error=str(error),
                        disposition='deferred_source_decode; original unchanged; no zero substitution')
                    SOURCE_FAILURE_OUTPUT.mkdir(parents=True, exist_ok=True)
                    with formal_publish_lease(failure_path,directory=SOURCE_FAILURE_OUTPUT,data_root=NAS):
                        if failure_path.exists():
                            if load_json(failure_path)!=failure_record:
                                raise ValueError(f'Job failure record conflicts with prior evidence: {failure_path}')
                        else:
                            save_json(failure_path,failure_record)
                    raise
            reader.read_window=read_with_context
            try:
                geometry=export_input(reader,panel,destination,tile_size=512)
            except imagecodecs.LzwError:
                destination.with_name(destination.name+'.partial').unlink(missing_ok=True)
                local.unlink()
                progress('image_deferred_source_decode',current_image=image_id)
                print(f'deferred {image_id}: native retained-channel decode failure',flush=True)
                return
        local.unlink()
        save_json(receipt_path,dict(image_id=image_id,source_revision=item['source_revision'],selection=selection,**geometry))
    if not geometry['admitted']:
        return
    record=dict.fromkeys(template('image_metadata'))
    record.update(study=STUDY,image_id=image_id,panel_id=panel.panel_id.iloc[0],image_kind='roi',
                  source_image_id=Path(item['source_path']).name,source_files=[item['source_path']],
                  source_format='ome_tiff',source_axes='CYX',source_dtype=geometry['source_dtype'],source_revision=item['source_revision'],
                  roi_id=item['project_name'],source_x0_px=0,source_y0_px=0,source_x1_px=w,source_y1_px=h,
                  pixel_size_x_um=item['pixel_size_um'][0],pixel_size_y_um=item['pixel_size_um'][1],
                  input_image_path=relative(destination),intensity_description='Published uint8 CODEX intensity; unchanged native values; no additional normalization',
                  split_method='already_independent',split_evidence='Zenodo original human CODEX specimen and QuPath full-image acquisition; author Module6 specimen list',
                  **links[image_id])
    for key in ['height_px','width_px','pad_left_px','pad_right_px','pad_top_px','pad_bottom_px','source_channel_count','channel_count']:
        record[key]=geometry[key]
    images,panels,_,_=publish_tables(source_scope())
    images=images[images.image_id!=image_id]
    images=pd.concat([images,pd.DataFrame([record])],ignore_index=True)
    samples,_=build_samples(source_scope())
    save_input_tables(OUTPUT,images,samples,panels,current_field_mapping(images,samples))
    progress('image_completed',current_image=image_id)
    print(f'completed {image_id}: {geometry["channel_count"]} x {side} x {side}',flush=True)


def self_check(scope: list[dict]) -> None:
    """Check all output headers and bounded native pixels per actual panel."""
    images=pd.read_parquet(OUTPUT/'images.parquet')
    panels=pd.read_csv(OUTPUT/'panel_mapping.csv')
    sources={x['image_id']:x for x in scope};results=[]
    chosen=set()
    for row in images.itertuples(index=False):
        _,h,w=sources[row.image_id]['shape'];side=max(h,w)
        left,top=(side-w)//2,(side-h)//2
        assert (row.source_x0_px,row.source_y0_px,row.source_x1_px,row.source_y1_px)==(0,0,w,h)
        assert (row.pad_left_px,row.pad_top_px,row.pad_right_px,row.pad_bottom_px)==(left,top,side-w-left,side-h-top)
        assert (row.height_px,row.width_px)==(side,side)
        panel=panels[(panels.panel_id==row.panel_id)&panels.keep_channel].sort_values('output_channel_index')
        with formal_read_lease(NAS/row.input_image_path), tifffile.TiffFile(NAS/row.input_image_path) as output:
            assert output.series[0].shape==(row.channel_count,row.height_px,row.width_px)
            assert output.pages[0].dtype==np.dtype(row.source_dtype)
            ome=ET.fromstring(output.ome_metadata)
            pixels=ome.find('.//{*}Pixels')
            assert float(pixels.attrib['PhysicalSizeX'])==row.pixel_size_x_um
            assert float(pixels.attrib['PhysicalSizeY'])==row.pixel_size_y_um
            names=[c.attrib['Name'] for c in ome.findall('.//{*}Channel')]
            assert names==panel.marker_id.tolist()
            group=(row.panel_id,bool(row.pad_left_px or row.pad_right_px),bool(row.pad_top_px or row.pad_bottom_px))
            evidence=dict(image_id=row.image_id,pixel_group=list(group),header_pass=True,blocks=[])
            if group not in chosen:
                chosen.add(group);item=sources[row.image_id];_,h,w=item['shape']
                with raw_source_read(NAS/item['source_path']) as source_path:
                    with tifffile.TiffFile(source_path) as original:
                        for x,y in [(w//2,h//2),(0,0),(max(0,w-32),max(0,h-32))]:
                            region=(x,y,min(w,x+32),min(h,y+32));out_region=(x+row.pad_left_px,y+row.pad_top_px,region[2]+row.pad_left_px,region[3]+row.pad_top_px)
                            for p in panel.itertuples(index=False):
                                a=stable_window(original,int(p.source_channel_index),region)
                                b=stable_window(output,int(p.output_channel_index),out_region)
                                assert np.array_equal(a,b)
                            evidence['blocks'].append(dict(source_xyxy=[int(v) for v in region],output_xyxy=[int(v) for v in out_region],channels=len(panel)))
            # Padding is tested on every output and every retained channel.
            padding=[]
            if row.pad_left_px:padding.append((0,row.pad_top_px, min(32,row.pad_left_px),min(row.height_px,row.pad_top_px+32)))
            if row.pad_top_px:padding.append((row.pad_left_px,0,min(row.width_px,row.pad_left_px+32),min(32,row.pad_top_px)))
            if row.pad_right_px:padding.append((row.pad_left_px+w,row.pad_top_px,min(row.width_px,row.pad_left_px+w+32),min(row.height_px,row.pad_top_px+32)))
            if row.pad_bottom_px:padding.append((row.pad_left_px,row.pad_top_px+h,min(row.width_px,row.pad_left_px+32),min(row.height_px,row.pad_top_px+h+32)))
            for region in padding:
                for index in range(row.channel_count):assert not stable_window(output,index,region).any()
            evidence['padding_xyxy']=[[int(v) for v in region] for region in padding]
            results.append(evidence)
    save_json(CACHE/'preparer_self_checks.json',results)


def stable_window(tif: tifffile.TiffFile, index: int, region: tuple[int,int,int,int]) -> np.ndarray:
    """Local TiffFrame adaptation; migrate aspage support to the shared reader later."""
    page=tif.pages[index]
    if isinstance(page,tifffile.TiffFrame):
        page=page.aspage()
    return read_tiff_window(SimpleNamespace(pages=[page],filehandle=tif.filehandle),0,region)


def finish(scope: list[dict], blocked: list[str]) -> None:
    images=pd.read_parquet(OUTPUT/'images.parquet');samples=pd.read_csv(OUTPUT/'sample_metadata.csv')
    gaps=[]
    source_failures=source_failure_records()
    current_revisions={item['image_id']:item['source_revision'] for item in scope}
    failed=sorted({record['image_id'] for _, record in source_failures
                   if current_revisions.get(record['image_id'])==record['source_revision']})
    pending_images=[x['image_id'] for x in scope if x['image_id'] not in set(images.image_id)|set(blocked)|set(failed)]
    if pending_images:gaps.append('Eligible sources awaiting export: '+','.join(pending_images))
    if failed:gaps.append('Native retained-channel LZW decode failed; source unchanged: '+','.join(failed))
    if blocked:gaps.append('Duplicate S100B channel selection lacks source-specific measurement decision: '+','.join(blocked))
    if samples.patient_key.isna().any():gaps.append('Patient identity/clinical-row conflicts remain for '+','.join(samples.loc[samples.patient_key.isna(),'sample_id']))
    summary=dict(study=STUDY,expected_images=12,exported_images=len(images),sample_rows=len(samples),sample_linked_images=int(images.sample_id.notna().sum()),patient_linked_samples=int(samples.patient_key.notna().sum()),blocked_images=blocked,source_decode_deferred_images=failed,pending_export_images=pending_images,issues=gaps,independent_review='pending')
    save_json(CACHE/'preparation_summary.json',summary)
    reconciliation=load_json(CACHE/'scope_reconciliation.json')
    reconciliation.pop('complete_verified_images',None)
    reconciliation.update(existing_native_proof_images=12,initial_panel_eligible_images=12-len(blocked),
                          eligible_this_batch=12-len(set(blocked+failed)),exported_this_batch=len(images),
                          deferred_images=len(set(blocked+failed)))
    for source in reconciliation['sources']:
        if source['image_id'] in failed:
            failure_path=next(path for path, record in source_failures
                              if record['image_id']==source['image_id']
                              and record['source_revision']==current_revisions[source['image_id']])
            source.update(status='deferred_source_decode',reason='Full-resolution native LZW segment decode failed; existing native proof insufficient; see '+relative(failure_path))
        elif source['image_id'] in set(images.image_id):
            source.update(status='exported_native_input',reason='Full retained-channel streaming decode and native input export completed')
    save_json(CACHE/'scope_reconciliation.json',reconciliation)
    rows=[]
    pending=[('study',STUDY,'D01',images.image_id.tolist(),'Independent review and input binding are pending')]
    if failed:
        pending.append(('study',STUDY,'S01',failed,'Retained native channels cannot be decoded; source acquisition completeness requires controller repair/reacquisition'))
    for image_id in blocked:
        pending.append(('panel',image_id,'P02',[image_id],'C35/C40 duplicate S100B measurement identity/selection unresolved'))
    for sample_id in samples.loc[samples.patient_key.isna(),'sample_id']:
        affected=[x['image_id'] for x in scope if x['project_name']==sample_id]
        pending.append(('sample',sample_id,'M01',affected,'Publication Patient-ID conflict or absent explicit clinical crosswalk; specimen identity is preserved'))
        pending.append(('sample',sample_id,'M02',affected,'Conflicting patient identity or unmatched clinical row; affected values withheld'))
    sources=[META/'acquisition_verification.json',META/'ordered_mapping.json',S1,SOURCE_SCOPE,
             AUTHOR_INPUTS/'cells_table.RDS',AUTHOR_REPOSITORY/'Module6_CODEX.R',
             AUTHOR_IDENTITY,PUBLICATION_TEXT,PUBLICATION/'mmc1.pdf',PUBLICATION/'article.html',
             GEO/'GSE237183_family.soft',RAW_MANIFEST]
    sources += [path for path, _ in source_failures]
    outputs=[OUTPUT/name for name in ['images.parquet','sample_metadata.csv','panel_mapping.csv','field_mapping.csv']]
    outputs += [SOURCE_SCOPE_OUTPUT]
    outputs += [CACHE/name for name in ['preparation_summary.json','preparer_self_checks.json','scope_reconciliation.json']]
    for row_scope,record_id,check,image_ids,reason in pending:
        details=dict(preparer_only=True,independent_review='pending',summary=summary,image_ids=image_ids,reason=reason)
        rows.append(dict(study=STUDY,preparer=PREPARER,reviewer='',scope=row_scope,record_id=record_id,check_id=check,result='unresolved',
                         source_evidence=json.dumps([relative(p) for p in sources]),
                         output_evidence=json.dumps([relative(p) for p in outputs]),details=json.dumps(details,ensure_ascii=False)))
    reviews=pd.DataFrame(rows,columns=template('review_results'))
    if (OUTPUT/'review_results.csv').exists():
        previous=pd.read_csv(OUTPUT/'review_results.csv',dtype=str,keep_default_na=False)
        previous=previous[~((previous.preparer==PREPARER)&(previous.reviewer==''))]
        reviews=pd.concat([previous,reviews],ignore_index=True)
    review_path = OUTPUT / 'review_results.csv'
    with formal_publish_lease(review_path, directory=review_path.parent):
        reviews.to_csv(review_path,index=False,encoding='utf-8-sig')
    progress('partial_outputs_pending_export' if pending_images else 'prepared_pending_independent_review',
             issues=gaps,blocked_images=blocked,pending_export_images=pending_images)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',action='store_true')
parser.add_argument('--overview',action='store_true')
parser.add_argument('--image')
parser.add_argument('--check',action='store_true')
parser.add_argument('--metadata',action='store_true')
parser.add_argument('--repair-age',action='store_true')
parser.add_argument('--batch-export',action='store_true')
parser.add_argument('--finalize',action='store_true')
args=parser.parse_args()
CACHE.mkdir(parents=True,exist_ok=True);TEMP.mkdir(parents=True,exist_ok=True)
scope=source_scope()
if args.overview:
    overviews(scope)
elif args.image:
    _,panels,identities,_=publish_tables(scope)
    item=next(x for x in scope if x['image_id']==args.image)
    _,links=build_samples(scope)
    export_one(item,panels[panels.panel_id==identities[args.image]],links)
elif args.run:
    _,_,blocked=build_panels(scope)
    progress('queued_for_export',blocked_images=blocked)
    subprocess.run([sys.executable,'-B',str(SCRIPT),'--batch-export'],check=True)
    subprocess.run([sys.executable,'-B',str(SCRIPT),'--check'],check=True)
    finish(scope,blocked)
elif args.check:
    self_check(scope)
elif args.metadata:
    publish_tables(scope)
elif args.repair_age:
    repair_age_metadata()
elif args.batch_export:
    _,panels,identities,blocked=publish_tables(scope)
    _,links=build_samples(scope)
    source_failures=source_failure_records()
    for item in scope:
        if item['image_id'] not in blocked:
            if any(record['image_id']==item['image_id']
                   and record['source_revision']==item['source_revision']
                   for _,record in source_failures):
                print(f"deferred {item['image_id']}: unchanged source has recorded decode failure",flush=True)
                continue
            export_one(item,panels[panels.panel_id==identities[item['image_id']]],links)
elif args.finalize:
    checked=load_json(CACHE/'preparer_self_checks.json')
    images=pd.read_parquet(OUTPUT/'images.parquet')
    assert {x['image_id'] for x in checked}==set(images.image_id)
    _,_,blocked=build_panels(scope)
    finish(scope,blocked)
else:
    parser.print_help()
