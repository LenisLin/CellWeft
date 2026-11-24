"""Deliver the 46 publication-associated HTAPP MBC MIBI FOVs at native intensity.

Inputs: release-3.0 OME-TIFFs, accepted acquisition manifest/native verification,
HTAN biospecimen ancestry, paper Table S2 and this study's S1 marker rows.
Outputs: independent CYX input TIFFs, image/sample/panel/field tables and preparer
checks. Shared helpers stage locally and acquire a formal lease for each publication.
No study-local reader adaptation is needed: the shared scalar TIFF reader applies.
"""
from __future__ import annotations

from cellweft.data.source_metadata import metadata_source_directory, metadata_source_relative

import argparse
import gzip
import json
import os
import shutil
import signal
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import pandas as pd
import pyarrow as pa
import tifffile

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

from cellweft.data.contracts import SourceSpec
from cellweft.data.io.readers import open_source, read_tiff_window
from cellweft.data.preparation.io_coordination import formal_publish_lease, formal_read_lease
from cellweft.data.preparation.study_input import export_input, panel_from_s1, save_input_tables


STUDY = "Klughammer2024_MIBI_MetastaticBreastCancer"
PROJECT = Path(__file__).resolve().parents[3]
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
META = metadata_source_directory(NAS, STUDY)
CACHE = PROJECT / ".cache/cellweft/runtime/study_input" / STUDY
OUTPUT = NAS / "processed" / STUDY
MANIFEST = META / "native_manifest.json"
NATIVE = META / "native_verification.json"
PORTAL = META / "source_metadata/htan_portal_selected.json"
DATABASE = META / "source_metadata/htan_selected_metadata.json"
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
TEMPLATES = PROJECT / "docs/data/templates/study_input"
JOB = PROJECT / ".cache/cellweft/runtime/jobs" / STUDY / "0214_prepare_0"
PROGRESS = JOB / "progress.json"
PREPARER = "codex_cli_0214_prepare_0"
PANEL_ID = "HTAPP_MBC_panel18"
PUBLICATION = "hta1_2024_nature-medicine_johanna-klughammer"
PAPER_URL = "https://www.nature.com/articles/s41591-024-03215-z"
SUPPLEMENT_URL = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-024-03215-z/MediaObjects/41591_2024_3215_MOESM3_ESM.xlsx"
AUTHOR_URL = "https://github.com/klarman-cell-observatory/HTAPP-Pipelines/tree/master/HTAPP_MBC"
GAPS = [
    "像素尺度来源冲突：OME为1 µm，HTAN表为0.48828125 µm；未选定实际尺度，输出尺度留空。",
    "论文未报告MIBI分析；本批46张的关联依据为HTAN publicationIds，另38张不在已准入范围。",
    "整理者自检；独立CLI复核及D01验收由主控组织。",
]


def relative(path: Path) -> str:
    if path.is_relative_to(CACHE) or path.is_relative_to(JOB):
        return str(path)
    return str(metadata_source_relative(path, NAS))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def columns(name: str) -> list[str]:
    return pd.read_csv(TEMPLATES / f"{name}.csv", nrows=0).columns.tolist()


def progress(phase: str, count: int, **details: object) -> None:
    write_json(PROGRESS, dict(study=STUDY, phase=phase, expected_images=46,
                             completed_images=count, gaps=GAPS,
                             updated_at=datetime.now(timezone.utc).isoformat(), **details))


class LeaseWait(TimeoutError):
    """A formal NAS lease remained unavailable at the checkpoint deadline."""


@contextmanager
def io_checkpoint():
    """Close active files before recording a bounded, resumable lease wait."""
    def alarm_handler(signum, frame):
        current = frame
        while current is not None:
            module = current.f_globals.get("__name__", "")
            name = current.f_code.co_name
            if module == "cellweft.data.preparation.io_coordination" and name == "_slot":
                raise LeaseWait(f"{current.f_locals['prefix']} shared slot unavailable")
            if module == "cellweft.data.io.publication" and name == "resource_lock":
                raise LeaseWait(str(current.f_locals["directory"] / current.f_locals["name"]))
            current = current.f_back
        signal.alarm(args.io_wait_seconds)

    previous = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.io_wait_seconds)
    try:
        yield
    except LeaseWait as error:
        pending = [record["source_id"] for record in records
                   if record["source_id"] not in committed_ids]
        progress("waiting_io", len(committed_ids), status="waiting_io", waiting_reason=str(error),
                 pending_image_ids=pending, resume_image=pending[0] if pending else None,
                 sample_rows=len(samples), patient_linked_samples=int(samples.patient_key.notna().sum()),
                 sample_linked_images=len(committed_ids), checkpoint=relative(stats_path),
                 resume_command=f"PYTHONPATH=src .venv-data/bin/python -B {relative_script} --job-dir {args.job_dir}" +
                                (" --verify" if args.verify else " --metadata-only" if args.metadata_only else ""))
        print(json.dumps(dict(status="waiting_io", completed_images=len(committed_ids),
                              reason=str(error), progress=str(PROGRESS)), ensure_ascii=False), flush=True)
        raise SystemExit(0)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def selected_metadata(records: list[dict]) -> dict:
    """Read the received study-specific HTAN parent and clinical snapshot."""
    return read_json(DATABASE)


def check_header(path: Path, panel: pd.DataFrame) -> None:
    """Reopen the delivered scalar planes and verify native type and named order."""
    names = panel.loc[panel.keep_channel, "marker_id"].tolist()
    with formal_read_lease(path), tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == "CYX" and tif.series[0].shape == (len(names), 1024, 1024)
        assert len(tif.pages) == len(names)
        assert all(page.shape == (1024, 1024) and page.dtype == np.dtype("uint16") for page in tif.pages)
        xml = ET.fromstring(tif.ome_metadata)
        assert [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")] == names


def build_samples(records: list[dict], metadata: dict) -> pd.DataFrame:
    """Join paper biopsy annotations by the exact terminal HTAN biospecimen ID.

    Paper S2 repeats a biopsy across assays; only agreeing clinical annotations
    are collapsed. HTAN age/day fields have obfuscation/semantic concerns and are
    left empty. Original receptor status remains a biopsy-level source note.
    """
    table = pd.read_excel(META / "source_metadata/supplementary_tables_1_5.xlsx", sheet_name="Tabe S2", keep_default_na=False)
    clinical_fields = ["Biopsy site", "Biopsy receptor status", "Biopsy receptor status (simplified)",
                       "Metastatic Presentation", "Biopsy days after metastasis", "Histology (breast)",
                       "Treatment status", "Treatment most recent class"]
    samples = []
    for sample_id in sorted({metadata["chains"][record["source_id"]][-1] for record in records}):
        linked = [record for record in records if metadata["chains"][record["source_id"]][-1] == sample_id]
        patients = {record["participant_id"] for record in linked}
        assert len(patients) == 1
        patient = patients.pop()
        original = metadata["biospecimens"][sample_id]
        assert original["ParentID"] == patient
        assert original["PortionWeight"] == "original tumor (tissue removed by procedure)"
        rows = table.loc[table["HTAN ID"] == sample_id, clinical_fields].drop_duplicates()
        assert len(rows) == 1, f"Missing/conflicting paper biopsy metadata: {sample_id}"
        clinical = rows.iloc[0].to_dict()
        demographics = metadata["demographicsByParticipantID"][patient]
        diagnosis = metadata["diagnosisByParticipantID"][patient]
        ancestry = [metadata["biospecimens"][key] for key in metadata["chains"][linked[0]["source_id"]]]
        assert any(row["StorageMethod"] == "Paraffin block" for row in ancestry)
        assert any(row["FixativeType"] == "Formalin" for row in ancestry)
        assert original["AcquisitionMethodType"] == "Biopsy"
        row = dict(study=STUDY, sample_id=sample_id, patient_key=patient, source_subject_id=patient,
                   source_sample_id=sample_id, biospecimen_group_id=sample_id, cohort_id="HTAN_HTAPP_MBC",
                   center_id=original["atlasid"], sample_type=original["BiospecimenType"], tissue_type="tumor",
                   anatomical_site=clinical["Biopsy site"], collection_method="biopsy", preservation_method="ffpe",
                   processing_method="MIBI FFPE section", case_control_status="case",
                   treatment_status_at_collection=clinical["Treatment status"],
                   sex_reported=demographics["Gender"].lower(),
                   primary_diagnosis=diagnosis["PrimaryDiagnosis"], primary_site=diagnosis["TissueorOrganofOrigin"])
        # Histology is described as breast histology in S2; unknown stays empty.
        if clinical["Histology (breast)"] != "Unknown":
            row["histology"] = clinical["Histology (breast)"]
        row["metadata_notes"] = json.dumps(dict(
            identity_evidence="HTAN explicit biospecimen parent chain terminates at participant",
            clinical_source="Supplementary Tables 1–5 / Tabe S2, exact HTAN ID",
            source_biopsy_annotations=clinical,
            missing="No clinical index/cutoff selected; obfuscated HTAN age/days and survival not imported; lesion type not inferred from metastatic cohort label.",
            image_count=len(linked)), ensure_ascii=False)
        samples.append(row)
    return pd.DataFrame(samples).reindex(columns=columns("sample_metadata"))


def field_rules() -> pd.DataFrame:
    """Record source keys and transformations once per coherent source rule."""
    rules = []

    def add(source: Path, table: str, field: str, target: str, target_field: str,
            keys: str, transformation: str, notes: str = "") -> None:
        rules.append(dict(study=STUDY, source_file=relative(source), source_sheet_or_table=table,
                          source_field=field, target_table=target, target_field=target_field,
                          join_keys=keys, value_mapping="", transformation_rule=transformation,
                          time_origin_rule="", conflict_rule="Conflicts remain documented; no inferred substitution", notes=notes))

    add(MANIFEST, "files", "source_id;filename;source_path;participant_id;parent_biospecimen_id", "image_metadata",
        "image_id;source_image_id;source_files;source_acquisition_id", "source_id=HTANDataFileID; filename=Filename",
        "Use accepted acquisition IDs and exact source files; require HTAN MIBI Level2 publication association and existing verified source stat.")
    add(META / "source_metadata/htan_selected_metadata.json", "biospecimens;chains", "HTANBiospecimenID;HTANParentID;ParentID;PortionWeight",
        "image_metadata", "sample_id;slide_id;fov_id;region_annotation", "manifest parent_biospecimen_id → explicit ParentID chain → participant_id",
        "sample_id=terminal biological collection; slide_id=section parent of FOV; fov_id=actual FOV biospecimen; region_annotation=source PortionWeight.",
        "46 FOVs → 14 original tumor biopsies → 14 participants; FOV and section remain separate from biological sample.")
    add(META / "source_metadata/htan_selected_metadata.json", "biospecimens;demographicsByParticipantID;diagnosisByParticipantID",
        "ParentID;Gender;PrimaryDiagnosis;TissueorOrganofOrigin;AcquisitionMethodType;StorageMethod;FixativeType;BiospecimenType;PortionWeight;atlasid;atlas_name",
        "sample_metadata", "sample_id;patient_key;source_subject_id;source_sample_id;biospecimen_group_id;cohort_id;center_id;sample_type;tissue_type;case_control_status;sex_reported;primary_diagnosis;primary_site;collection_method;preservation_method;processing_method",
        "terminal biospecimen ParentID=participant; clinical dictionaries keyed by participant",
        "Retain public IDs and BiospecimenType; center_id=atlasid; define cohort_id=HTAN_HTAPP_MBC for the accepted HTAPP MBC subset. Original tumor PortionWeight→tissue_type=tumor and case_control_status=case. Lower-case Gender; Biopsy→biopsy; formalin-fixed/paraffin-block ancestry→ffpe; processed FOVs are MIBI FFPE sections.",
        "Day/age fields are not imported: HTAN AgeIsObfuscated and ambiguous index semantics. Preservation and processing describe imaged material downstream of the original biopsy.")
    add(META / "source_metadata/supplementary_tables_1_5.xlsx", "Tabe S2", "HTAN ID;Biopsy site;Histology (breast);Treatment status;biopsy annotations",
        "sample_metadata", "anatomical_site;histology;treatment_status_at_collection;metadata_notes", "HTAN ID=terminal biospecimen sample_id",
        "Collapse repeated assay rows only after all eight biopsy annotations agree. Keep biopsy site and treatment status verbatim; Unknown histology→empty; other annotations in source_biopsy_annotations JSON.",
        "14/14 samples matched. Receptor values remain biopsy notes, without assigning patient-index molecular subtype or survival origin.")
    add(S1, "S1", "Study;Original marker;Unified marker;Segmentation channel", "panel_mapping",
        "source_name;source_channel_index;marker_id;role;keep_channel;output_channel_index;nuclear;body",
        "Study=study; Original marker=OME Channel Name", "Exact study-specific labels; retain source order; keep 37 downstream markers and dsDNA; exclude 11 source channels.")
    add(MANIFEST, "files → referenced native OME-TIFF Image/Pixels/Channel and scalar page headers",
        "source_path;size;Image Name;SizeX;SizeY;SizeZ;SizeT;Type;DimensionOrder;Channel Name;release",
        "image_metadata", "panel_id;image_kind;source_format;source_z_index;source_t_index;source_axes;source_dtype;source_revision;roi_id;source_x0_px;source_y0_px;source_x1_px;source_y1_px;pad_left_px;pad_top_px;pad_right_px;pad_bottom_px;height_px;width_px;source_channel_count;channel_count;input_image_path;valid_pixels_path;intensity_description;split_method;split_evidence",
        "source_id=image_id; source_path=source_files[0]",
        "Each native file is one independent FOV: CYX, uint16, 49×1024×1024, singleton Z/T at index 0. Compare MBC Point identity in filename and native OME Image Name; assign the verified panel18. Full half-open frame [0,1024)×[0,1024), square, all four pads zero. Retain 38 channels in output order, no intensity transform. Save NAS-relative processed/<study>/images/<image_id>/input.tiff. No source missing-pixel domain is declared, so valid_pixels_path is empty; split_method=already_independent and split_evidence=accepted manifest.",
        "All 46 source headers were enumerated in source_header_inventory.json; output headers are checked after each export. Physical size is handled separately because source OME and portal values conflict.")
    add(PORTAL, "ImagingLevel2", "PhysicalSizeX;PhysicalSizeY;ImageID;FOVnumber", "image_metadata",
        "pixel_size_x_um;pixel_size_y_um;metadata_notes", "HTANDataFileID=source_id; Filename=filename",
        "Pixel size empty: portal 0.48828125 µm conflicts with OME 1 µm. Portal ImageID has NB typo; identity follows filename+synapse+DRS+HTAN ID and native MBC Image Name.",
        "Native resolution and intensities unchanged. Both scale values recorded; no resampling. FOV identifiers use biospecimen keys, not the erroneous portal ImageID.")
    add(META / "htan_release_3.0_source_files.json", "source files", "filename;publication_ids", "image_metadata", "metadata_notes",
        "filename", "84 indexed MBC files; only 46 linked/accepted files are complete and exported; 38 unassociated files excluded; all .part files untouched.",
        "The paper's analyzed spatial assays omit MIBI. HTAN associates these 46 MIBI acquisitions with this publication; do not claim that the paper analyzed them.")
    add(META / "source_metadata/author_CODEX_MIBI_panel.tsv", "author repository / HTAPP_MBC/00_meta", "method;gene;gene_orig", "panel_mapping",
        "source_name", "method=mibi", "Author repository corroborates MIBI panel context; actual ordered TIFF names and this study's S1 determine export.",
        "Repository tree inspected for sample/core metadata. Explicit identity ancestry and exact paper S2 HTAN IDs provide the sample connection.")
    return pd.DataFrame(rules).reindex(columns=columns("field_mapping"))


def save_review(images: pd.DataFrame, samples: pd.DataFrame, verified: bool) -> None:
    """Write preparer self-checks, keeping independent D01 acceptance unresolved."""
    checks = []

    def add(check: str, result: str, details: str) -> None:
        checks.append(dict(study=STUDY, preparer=PREPARER, reviewer="",
                           scope="sample" if check.startswith("M") else "study", record_id=STUDY,
                           check_id=check, result=result,
                           source_evidence=";".join(map(relative, [MANIFEST, NATIVE, META / "source_metadata/htan_selected_metadata.json", S1])),
                           output_evidence=relative(OUTPUT), details="整理者自检。" + details))

    for check, details in {
        "S01": f"当前已导出{len(images)}/46；84条来源中的38条不在已准入论文关联范围；整图marker信号依据见export_records.json；完整文件与.part不改。",
        "I01": "所有已发布图像保留HTAN采集ID及原始NAS相对路径。",
        "I02": f"{len(images)}张已发布图像均连接14行sample表中的取样；panel外键可连接。",
        "I03": "完整1024×1024来源框，无裁切、补边或重采样；uint16强度保持；两份尺度证据冲突，尺度字段留空。",
        "I04": "来源独立FOV；HTAN明确FOV→section→FFPE block→original tumor biopsy→participant；region_annotation保留PortionWeight。",
        "P01": "逐来源OME核对49个有序标签，精确连接本study S1。",
        "P02": "保留37个downstream与1个dsDNA核通道，排除11通道；无重复downstream。",
        "P03": "已逐图重开头信息，确认38×1024×1024、CYX、uint16及所有标准化通道名/顺序。",
        "M01": f"已查本地HTAN原始发布快照、原始取样父子链、论文补充Tabe S2及作者仓库。sample {len(samples)}行，patient 14个；已导出image→sample {len(images)}/{len(images)}，sample→patient 14/14。",
        "M02": "14份取样按精确HTAN ID连接S2；重复assay临床值一致；已填字段规则见field_mapping；未定义index/cutoff、混淆年龄/天数及生存字段留空。",
    }.items():
        add(check, "pass", details)
    add("I05", "pass" if verified else "unresolved",
        "全部38保留通道在代表图的32×32边界/内部块逐值比较；概览图见cache/verification.json。" if verified else "等待完成后执行--verify像素块及代表图概览自检。")
    add("D01", "unresolved", f"图像交付{len(images)}/46，sample14行且patient全部连接。整理者不能作独立验收；独立CLI复核及input_binding待主控组织。")
    temporary = OUTPUT / "review_results.csv.tmp"
    review_path = OUTPUT / "review_results.csv"
    with formal_publish_lease(review_path, directory=OUTPUT):
        pd.DataFrame(checks).reindex(columns=columns("review_results")).to_csv(
            temporary, index=False, encoding="utf-8-sig")
        temporary.replace(review_path)


def verify(images: pd.DataFrame, panel: pd.DataFrame, records: list[dict]) -> dict:
    """Check all headers; compare all retained channels and overview three FOVs."""
    os.environ["MPLCONFIGDIR"] = str(CACHE / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for row in images.itertuples(index=False):
        check_header(NAS / row.input_image_path, panel)
    by_id = {record["source_id"]: record for record in records}
    ids = images.image_id.tolist()
    representatives = list(dict.fromkeys([ids[0], ids[len(ids)//2], ids[-1]]))
    kept = panel[panel.keep_channel]
    plot_channels = ["Target:dsDNA", "Target:E-Cadherin", "Target:CD45"]
    figure, axes = plt.subplots(len(representatives), 3, figsize=(12, 4*len(representatives)), squeeze=False)
    comparisons = []
    for index, image_id in enumerate(representatives):
        record = by_id[image_id]
        destination = NAS / images.set_index("image_id").loc[image_id, "input_image_path"]
        with formal_read_lease(destination), tifffile.TiffFile(NAS / record["source_path"]) as source, tifffile.TiffFile(destination) as target:
            for window in [(0, 0, 32, 32), (496, 496, 528, 528)]:
                marker_nonzero = 0
                for channel in kept.itertuples(index=False):
                    actual = read_tiff_window(source, channel.source_channel_index, window)
                    exported = read_tiff_window(target, int(channel.output_channel_index), window)
                    assert actual.dtype == exported.dtype and np.array_equal(actual, exported)
                    if channel.role == "downstream":
                        marker_nonzero += int(np.count_nonzero(actual))
                if window[0] == 496:
                    assert marker_nonzero > 0
                comparisons.append(dict(image_id=image_id, window=list(window), channels=len(kept),
                                        downstream_nonzero_values=marker_nonzero, exact=True))
            for col, name in enumerate(plot_channels):
                channel = panel.loc[panel.source_name == name].iloc[0]
                plane = source.pages[int(channel.source_channel_index)].asarray(maxworkers=1)
                display = np.log1p(plane.astype(np.float32))
                nonzero = display[plane > 0]
                vmax = float(np.quantile(nonzero, .995)) if len(nonzero) else 1.0
                axes[index, col].imshow(display, cmap="gray", vmin=0, vmax=vmax)
                axes[index, col].set_title(f"{image_id}\n{name}", fontsize=9)
                axes[index, col].axis("off")
    figure.tight_layout()
    figure.savefig(CACHE / "representative_overviews.png", dpi=120)
    plt.close(figure)
    return dict(header_count=len(images), representatives=representatives, comparisons=comparisons,
                overview=relative(CACHE / "representative_overviews.png"),
                display_transform="log1p and nonzero P99.5 for overview display only; input TIFF unchanged")


# Resolve source scope and biological identities before acquiring any new pixels.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--limit", type=int, default=46, help="Maximum new images in this invocation")
parser.add_argument("--metadata-only", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--job-dir", type=Path, default=JOB)
parser.add_argument("--io-wait-seconds", type=int, default=120)
args = parser.parse_args()
PROGRESS = args.job_dir / "progress.json"
PREPARER = "codex_cli_" + args.job_dir.name
relative_script = str(metadata_source_relative(Path(__file__), PROJECT))
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
records = [record for record in read_json(MANIFEST)["files"] if record["study"] == STUDY]
assert len(records) == 46
native = read_json(NATIVE)["files"]
portal = {row["HTANDataFileID"]: row for row in read_json(PORTAL)
          if str(row.get("Filename", "")).startswith("mibi_level_2/MBC/") and PUBLICATION in (row.get("publicationIds") or [])}
assert set(portal) == {record["source_id"] for record in records}
metadata = selected_metadata(records)
samples = build_samples(records, metadata)
panel = panel_from_s1(STUDY, PANEL_ID,
                      read_json(META / "htapp_native_channel_mapping.json")["ordered_target_labels"], S1)
assert len(panel) == 49 and panel.keep_channel.sum() == 38
mapping = field_rules()
images = pd.read_parquet(OUTPUT / "images.parquet") if (OUTPUT / "images.parquet").exists() else pd.DataFrame(columns=columns("image_metadata"))
committed_ids = set(images.image_id)
stats_path = CACHE / "export_records.json"
statistics = read_json(stats_path) if stats_path.exists() else {}
progress("metadata_ready", len(images), sample_rows=len(samples), patient_linked_samples=14)

# Reserve enough space for the remaining native outputs and one local staged image.
estimated_output_bytes = (46-len(images))*38*1024*1024*2
space = dict(remaining_output_bytes=estimated_output_bytes, local_staging_bytes=38*1024*1024*2,
             nas_available_bytes=shutil.disk_usage(NAS).free, local_available_bytes=shutil.disk_usage("/tmp").free,
             nas_reserve_bytes=100*1024**3, local_reserve_bytes=64*1024**3)
write_json(args.job_dir / "space_estimate.json", space)
if (space["nas_available_bytes"]-estimated_output_bytes < space["nas_reserve_bytes"] or
        space["local_available_bytes"]-space["local_staging_bytes"] < space["local_reserve_bytes"]):
    progress("waiting_io", len(images), status="waiting_io", waiting_reason="required free-space reserve unavailable", **space)
    raise SystemExit(0)

# Reuse earlier native verification through source size+mtime, without a new hash.
inventory = []
accepted_names = {record["filename"] for record in records}
for entry in read_json(META / "htan_release_3.0_source_files.json"):
    path = NAS / "raw/public" / STUDY / "htan_release_3.0" / entry["filename"]
    inventory.append(dict(filename=entry["filename"], accepted=entry["filename"] in accepted_names,
                          complete_file=path.is_file(), partial_exists=Path(str(path)+".part").is_file(),
                          reason="accepted HTAN publication-associated acquisition" if entry["filename"] in accepted_names else "outside accepted publication-associated scope"))
write_json(CACHE / "source_inventory.json", inventory)
write_json(CACHE / "metadata_sources.json", dict(
    htan_snapshot=relative(DATABASE), htan_portal="https://data.humantumoratlas.org/",
    manifest=relative(MANIFEST), native_verification=relative(NATIVE), paper=PAPER_URL,
    supplementary_table=dict(url=SUPPLEMENT_URL, local=relative(META / "source_metadata/supplementary_tables_1_5.xlsx"), sheet="Tabe S2", exact_sample_matches=14),
    author_repository=AUTHOR_URL, author_tree=relative(META / "source_metadata/author_repository_tree.json"),
    sample_identity="explicit HTAN FOV→section→block→original collection→participant chain",
    scale_conflict=dict(ome_um=1.0, htan_um=0.48828125, action="leave physical scale unset; no resampling")))

with io_checkpoint():
    if args.verify:
        verification = verify(images, panel, records)
        write_json(CACHE / "verification.json", verification)
        save_review(images, samples, True)
        progress("self_checks_complete_pending_independent_review", len(images), sample_rows=len(samples),
                 sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=14)
        print(json.dumps(verification, ensure_ascii=False), flush=True)
    elif args.metadata_only:
        save_input_tables(OUTPUT, images, samples, panel, mapping)
        save_review(images, samples, False)
        progress("metadata_ready", len(images), sample_rows=len(samples), patient_linked_samples=14)
    else:
        new_images = 0
        for record in records:
            signal.alarm(args.io_wait_seconds)
            image_id = record["source_id"]
            source_path = NAS / record["source_path"]
            proof = native[image_id]
            stat = source_path.stat()
            assert stat.st_size == record["expected_bytes"] == proof["file_stat"]["size"]
            assert stat.st_mtime_ns == proof["file_stat"]["mtime_ns"]
            assert proof["verification_status"] == "verified" and proof["mapping_verified"]
            assert record["filename"] == portal[image_id]["Filename"]
            assert record["parent_biospecimen_id"] == portal[image_id]["HTANParentBiospecimenID"]
            assert record["participant_id"] == portal[image_id]["HTANParticipantID"]
            destination = OUTPUT / "images" / image_id / "input.tiff"
            if image_id in set(images.image_id):
                assert destination.exists() and statistics[image_id]["admitted"]
                check_header(destination, panel)
                continue
            if new_images >= args.limit:
                break
            with formal_read_lease(source_path), tifffile.TiffFile(source_path) as tif:
                xml = ET.fromstring(tif.ome_metadata)
                names = [node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Channel")]
                assert names == panel.source_name.tolist() == proof["observed_channel_labels"]
                assert tif.series[0].axes == "CYX" and tif.series[0].shape == (49, 1024, 1024)
                image_name = next(node.attrib["Name"] for node in xml.iter() if node.tag.endswith("}Image"))
                # The portal's NB ImageID typo must not replace the native MBC point identity.
                point = source_path.name.removeprefix("MIBITIFF_MBC_").removesuffix("_combined.ome.tiff")
                assert image_name == f"HTAPP_MBC_{point}.tiff SIMS"
            source = SourceSpec(image_id, STUDY, "tiff", source_path, tuple(range(49)), source_revision="HTAN Release 3.0")
            progress("exporting", len(images), current_image=image_id, estimated_uncompressed_total_bytes=46*38*1024*1024*2)
            # A publication can precede its index checkpoint when the next lease waits.
            # Reuse that closed TIFF only with the matching source revision and saved QC.
            saved = statistics.get(image_id)
            if (saved and saved["admitted"] and destination.is_file() and
                    saved["source_size"] == stat.st_size and saved["source_mtime_ns"] == stat.st_mtime_ns):
                check_header(destination, panel)
                stats = saved
            else:
                with ExitStack() as source_stack:
                    with formal_read_lease(source_path):
                        reader = source_stack.enter_context(open_source(source))
                    stats = export_input(reader, panel, destination)
                statistics[image_id] = dict(stats, source_path=record["source_path"],
                                            source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
                write_json(stats_path, statistics)
            new_images += 1
            if not stats["admitted"]:
                progress("excluded_zero_markers", len(images), current_image=image_id)
                continue
            check_header(destination, panel)
            chain = metadata["chains"][image_id]
            row = dict(study=STUDY, image_id=image_id, sample_id=chain[-1], panel_id=PANEL_ID,
                       image_kind="fov", source_image_id=image_id, source_files=[record["source_path"]],
                       source_format="OME-TIFF", source_acquisition_id=image_id, source_z_index=0, source_t_index=0,
                       source_axes="CYX", source_dtype="uint16", source_revision="HTAN Release 3.0",
                       slide_id=chain[1], fov_id=chain[0], roi_id=image_name,
                       region_annotation=metadata["biospecimens"][chain[0]]["PortionWeight"],
                       source_x0_px=0, source_y0_px=0, source_x1_px=1024, source_y1_px=1024,
                       pad_left_px=0, pad_top_px=0, pad_right_px=0, pad_bottom_px=0,
                       height_px=1024, width_px=1024, source_channel_count=49, channel_count=38,
                       input_image_path=relative(destination),
                       intensity_description="HTAN Level-2 published uint16 MIBI intensities, unchanged; no normalization",
                       split_method="already_independent", split_evidence=relative(MANIFEST),
                       metadata_notes=json.dumps(dict(ome_image_name=image_name, synapse_id=record["synapse_id"],
                           drs_uri=record["drs_uri"], biological_chain=chain,
                           physical_scale_conflict="OME=1 µm versus HTAN=0.48828125 µm; unset",
                           portal_image_name=portal[image_id]["ImageID"],
                           publication_scope="HTAN-associated MIBI; paper analysis does not describe MIBI"), ensure_ascii=False))
            images = pd.concat([images, pd.DataFrame([row])], ignore_index=True)
            save_input_tables(OUTPUT, images, samples, panel, mapping)
            committed_ids = set(images.image_id)
            save_review(images, samples, False)
            progress("exporting", len(images), last_completed_image=image_id, sample_rows=len(samples),
                     sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=14)
            print(json.dumps(dict(completed_images=len(images), image_id=image_id), ensure_ascii=False), flush=True)
        progress("export_complete" if len(images) == 46 else "export_batch_complete", len(images),
                 sample_rows=len(samples), sample_linked_images=int(images.sample_id.notna().sum()), patient_linked_samples=14)
