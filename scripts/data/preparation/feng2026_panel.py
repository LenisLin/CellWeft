"""Build the Feng2026 native39 input panel and fixed MAP vectors on CPU.

Inputs: Steinbock raw headers and panel, indexed intake, canonical S1, fixed MAP bank.
Outputs: study panel_mapping.csv, marker_embeddings.pt, and private panel_contract.json.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import pandas as pd
from readimc import MCDFile
import torch

from cellweft.data.preparation.embeddings import _validate, build_study_embeddings
from cellweft.data.preparation.inputs import file_revision, validate_panel_mapping
from cellweft.data.preparation.study_input import panel_from_s1


STUDY = "Feng2026_IMC_BladderCancer"
PANEL_ID = f"{STUDY}__native39"
NAS = Path("/mnt/NAS_21T/ProjectData/CellWeft")
SOURCE = Path("/mnt/public/lyx/horizontalProject/IMC_20240725_63")
S1 = NAS / "processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv"
BANK = NAS / "processed/_resources/protein_banks/map_gene_projector_v1"
INTAKE = NAS / "cache/IMC_20240725_63_intake/indexed_sources.csv"
NUCLEAR = {"Ir191", "Ir193"}
BODY = {"Pr141", "Nd142", "Sm152", "Gd156", "Gd160"}
TXT_METADATA = ["Start_push", "End_push", "Pushes_duration", "X", "Y", "Z"]
S1_FIELDS = ["Unified marker", "MAP-KG mapping", "UniProt accession"]
S1_MARKERS = dict(zip(
    ("Pr141", "Nd142", "Nd143", "Nd144", "Nd145", "Sm147", "Nd150", "Sm152", "Eu153",
     "Gd154", "Gd155", "Gd156", "Gd158", "Tb159", "Gd160", "Dy161", "Dy162", "Dy163",
     "Dy164", "Ho165", "Er166", "Er167", "Er168", "Tm169", "Er170", "Yb171", "Yb172",
     "Yb173", "Lu175"),
    ("CD68", "αSMA", "Vimentin", "CD47", "CD274", "FAP", "EPCAM", "CD45", "CD44",
     "CD279", "FoxP3", "CD31", "E-Cadherin", "CD4", "PanCK", "CD20", "CD8a", "CD90",
     "CK5", "β-Catenin", "AR", "TGFB1", "Ki-67", "COLLAGEN-I", "CD3e", "CD66a", "CD133",
     "CD152", "CD309"),
    strict=True,
))
PENDING = {
    "Nd146": ("ALDH", "Source label lacks isoform specificity."),
    "Eu151": ("Snail/Slug", "Combined label has no equivalent S1 measurement or verified reagent specificity."),
}
DIRECT = {
    "Sm149": ("YAP1", "P46937", "Source labels total target YAP1 without an active-state qualifier."),
    "Yb174": ("ESR2", "Q92731", "Source labels ER beta, the ESR2 protein (UniProt Q92731)."),
}


def raw_headers(path: Path) -> list[tuple[int | None, tuple[tuple[str, str], ...]]]:
    """Read channel metadata without logging paths or acquisition identifiers."""
    kind = path.suffix.lower()
    try:
        if kind == ".mcd":
            with MCDFile(path) as source:
                return [(acquisition.id, tuple(zip(acquisition.channel_names, acquisition.channel_labels, strict=True)))
                        for slide in source.slides for acquisition in slide.acquisitions]
        with path.open(encoding="utf-8-sig") as source:
            fields = source.readline().rstrip("\r\n").split("\t")
        if fields[:len(TXT_METADATA)] != TXT_METADATA:
            raise ValueError("TXT metadata columns differ")
        channels = []
        for field in fields[len(TXT_METADATA):]:
            match = re.fullmatch(r"(.+)\(([A-Z][a-z]?\d+)Di\)", field)
            if match is None:
                raise ValueError("Invalid TXT intensity field")
            label, isotope = match.groups()
            channels.append((isotope, label))
        return [(None, tuple(channels))]
    except Exception:
        raise ValueError(f"Unable to parse one raw {kind.upper().lstrip('.')} channel header") from None


parser = ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, default=NAS, help="Root for processed/ and cache/ study artifacts")
args = parser.parse_args()
processed = args.output_root / "processed" / STUDY
cache = args.output_root / "cache" / STUDY
try:
    processed.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
except OSError as error:
    raise SystemExit(f"Cannot create study output directories: {error.strerror}") from None

source_panel = pd.read_csv(SOURCE / "steinbock/panel.csv", dtype=str, keep_default_na=False)
native = tuple(zip(source_panel.channel, source_panel.name, strict=True))
if len(native) != 39 or not source_panel.channel.is_unique or not source_panel.name.is_unique:
    raise ValueError("Expected 39 unique native isotope and label pairs")
if (set(source_panel.loc[source_panel.deepcell == "1", "channel"]) & NUCLEAR != NUCLEAR
        or set(source_panel.loc[source_panel.deepcell == "2", "channel"]) & BODY != BODY):
    raise ValueError("Segmentation selection disagrees with the source panel")

indexed = pd.read_csv(INTAKE, dtype=str, keep_default_na=False)
selected = defaultdict(Counter)
for row in indexed.itertuples():
    path = SOURCE / row.raw_relative_to_study
    acquisition = int(row.acquisition_id) if path.suffix.lower() == ".mcd" else None
    selected[path][acquisition] += 1
if any(count != 1 for acquisitions in selected.values() for count in acquisitions.values()):
    raise ValueError("Indexed intake contains repeated raw references")
paths = sorted(path for path in (SOURCE / "steinbock/raw").rglob("*")
               if path.suffix.lower() in {".mcd", ".txt"} and path.is_file())
raw_counts = Counter()
indexed_counts = Counter()
with ThreadPoolExecutor(max_workers=4) as workers:
    for path, headers in zip(paths, workers.map(raw_headers, paths), strict=True):
        kind = path.suffix.lower().lstrip(".")
        for acquisition, channels in headers:
            if channels == native:
                layout = "native39_same_order"
            elif Counter(channels) == Counter(native):
                layout = "native39_reordered"
            else:
                layout = f"different_{len(channels)}_channels"
            raw_counts[f"{kind}:{layout}"] += 1
            if acquisition in selected.get(path, {}):
                indexed_counts[f"{kind}:{layout}"] += 1
if (sum(indexed_counts.values()) != len(indexed)
        or set(indexed_counts) != {"mcd:native39_same_order", "txt:native39_same_order"}):
    raise ValueError(f"Selected source layouts do not share native39 order: {dict(indexed_counts)}")

canonical_s1 = pd.read_csv(S1, dtype=str, keep_default_na=False)
if canonical_s1.Study.eq(STUDY).any():
    raise ValueError("Canonical S1 now has study rows. Review them before replacing the private study mapping.")
private_rows = []
mapping_basis = {}
for row in source_panel.itertuples(index=False):
    isotope, name = row.channel, row.name
    expression, accession = "", ""
    if row.keep == "0":
        marker, basis = "Excluded", "Source panel excludes this technical channel."
    elif isotope in NUCLEAR:
        marker, basis = "DNA (segmentation only)", "DNA intercalator selected by the source panel and S1 nuclear rule."
    elif isotope in PENDING:
        label, basis = PENDING[isotope]
        marker = f"{label} (mapping pending)"
    elif isotope in DIRECT:
        marker, accession, basis = DIRECT[isotope]
        expression = marker
    else:
        rows = canonical_s1.loc[canonical_s1["Unified marker"].eq(S1_MARKERS[isotope]), S1_FIELDS].drop_duplicates()
        if len(rows) != 1:
            raise ValueError("An expected canonical S1 marker has an absent or ambiguous mapping")
        marker, expression, accession = rows.iloc[0]
        basis = "Reuses the canonical S1 unified marker and its MAP expression."
    private_rows.append({"Study": STUDY, "Original marker": name, "Unified marker": marker,
                         "MAP-KG mapping": expression, "UniProt accession": accession,
                         "Segmentation channel": "1" if isotope in NUCLEAR else "2" if isotope in BODY else ""})
    mapping_basis[name] = basis

with TemporaryDirectory(prefix=".feng-panel-", dir=processed) as staged_processed, \
        TemporaryDirectory(prefix=".feng-panel-", dir=cache) as staged_cache:
    stage = Path(staged_processed)
    contract_stage = Path(staged_cache) / "panel_contract.json"
    private_s1 = Path(staged_cache) / "study_s1.csv"
    pd.DataFrame(private_rows).to_csv(private_s1, index=False, encoding="utf-8-sig")
    panel = panel_from_s1(STUDY, PANEL_ID, list(source_panel.name), private_s1)
    panel = validate_panel_mapping(panel, input_channel_count=35)
    if panel.keep_channel.tolist() != source_panel.keep.eq("1").tolist():
        raise ValueError("Retained channels disagree with the source panel")
    bundle, panel = build_study_embeddings(STUDY, panel, private_s1, BANK)

    channels = json.loads(panel.to_json(orient="records"))
    for channel, (isotope, name) in zip(channels, native, strict=True):
        channel.update(isotope=isotope, mapping_basis=mapping_basis[name])
    input_indices = panel.loc[panel.keep_channel, "source_channel_index"].tolist()
    contract = {
        "study": STUDY, "panel_id": PANEL_ID, "index_base": 0,
        "source_channel_count": len(native), "input_channel_count": len(input_indices),
        "input_source_channel_indices": input_indices,
        "nuclear_input_channel_indices": panel.loc[panel.nuclear, "output_channel_index"].tolist(),
        "body_input_channel_indices": panel.loc[panel.body, "output_channel_index"].tolist(),
        "segmentation_basis": "S1 DNA preference and five source body candidates covering tumor, immune, stromal and endothelial cells. Signal quality is unverified.",
        "channels": channels, "indexed_layout_counts": dict(indexed_counts), "raw_layout_counts": dict(raw_counts),
        "txt_metadata_columns": TXT_METADATA, "canonical_s1": file_revision(S1), "study_s1_rows": private_rows,
    }
    contract_stage.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle.payload["metadata"]["s1"] = {**file_revision(contract_stage),
                                       "path": str((cache / contract_stage.name).resolve()),
                                       "key": "study_s1_rows"}
    bundle.payload["metadata"]["canonical_s1"] = contract["canonical_s1"]
    panel_stage = stage / "panel_mapping.csv"
    vectors_stage = stage / "marker_embeddings.pt"
    panel.to_csv(panel_stage, index=False, encoding="utf-8-sig")
    torch.save(bundle.payload, vectors_stage)
    reopened = pd.read_csv(panel_stage, keep_default_na=False)
    reopened["embedding_row"] = pd.to_numeric(reopened.embedding_row, errors="raise").astype("Int64")
    _validate(torch.load(vectors_stage, map_location="cpu", weights_only=True), reopened)
    panel_stage.replace(processed / panel_stage.name)
    contract_stage.replace(cache / contract_stage.name)
    vectors_stage.replace(processed / vectors_stage.name)

print(json.dumps({"panel_id": PANEL_ID, "indexed_layout_counts": dict(indexed_counts),
                  "raw_layout_counts": dict(raw_counts), "mapping_status_counts": panel.mapping_status.value_counts().to_dict(),
                  "input_channels": len(input_indices), "nuclear_channels": int(panel.nuclear.sum()),
                  "body_channels": int(panel.body.sum()), "embedding_shape": list(bundle.payload["embeddings"].shape)},
                 ensure_ascii=False, sort_keys=True))
