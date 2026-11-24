"""CPU-only S1 lookup and coherent study-local marker embedding publication."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json
import os
import re
import tempfile

import pandas as pd

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision, assert_reviewed_binding, load_reviewed_study_inputs


def _panel_identity(panel: pd.DataFrame) -> str:
    value = panel.drop(columns=['embedding_row'],errors='ignore').copy()
    if 'output_channel_index' in value:
        value['output_channel_index'] = pd.to_numeric(value.output_channel_index.replace('',pd.NA),errors='raise').astype('Int64')
    return value.replace('',None).to_json(orient='records')


def _same_panel_identity(left: str, right: str) -> bool:
    if left == right:
        return True

    def ordered(identity: str) -> list[dict]:
        def key(row: dict) -> tuple:
            raw = row.get('source_channel_index')
            output = row.get('output_channel_index')
            return (row['panel_id'], -1 if raw is None else int(raw),
                    output is None, -1 if output is None else int(output))

        return sorted(json.loads(identity), key=key)

    return ordered(left) == ordered(right)


def _expression_genes(expression: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r'(mean|proxy)\(([^()]*)\)',expression)
    mode, genes = (match[1],[gene.strip() for gene in match[2].split(',')]) if match else ('direct',[expression])
    if not genes or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',gene) for gene in genes):
        raise ValueError(f"Unsupported S1 mapping expression: {expression}")
    if mode in {'proxy','direct'} and len(genes) != 1:
        raise ValueError("Direct/proxy mappings require exactly one gene")
    if len(set(genes)) != len(genes):
        raise ValueError("Repeated mapping members require explicit S1 correction")
    return mode,genes


@dataclass
class StudyEmbeddings:
    """CPU tensor payload and the panel snapshot from which it was derived."""

    payload: dict
    panel_identity: str
    acceptance: dict | None = None
    data_root: Path | None = None
    acceptances: list[dict] | None = None


def build_study_embeddings(study: str, panel_mapping: pd.DataFrame, s1_path: Path,
                           bank_dir: Path, *, panel_ids: set[str] | None = None) -> tuple[StudyEmbeddings, pd.DataFrame]:
    """Resolve kept mapped downstream markers; missing/conflicting members raise ValueError."""
    import torch
    panel = panel_mapping.copy()
    s1 = pd.read_csv(s1_path,dtype=str,keep_default_na=False)
    s1 = s1[s1.Study == study].set_index('Original marker',verify_integrity=True)
    manifest = json.loads((bank_dir/'manifest.json').read_text())
    genes = json.loads((bank_dir/'gene_symbols.json').read_text())
    vectors = torch.load(bank_dir/'embeddings.pt',map_location='cpu',weights_only=True)
    if manifest['rows'] != len(genes) or len(set(genes)) != len(genes) or vectors.shape != (len(genes),manifest['dimension']) or vectors.dtype != torch.float32 or not torch.isfinite(vectors).all():
        raise ValueError("Protein bank shape, keys, dtype or finite-value contract failed")
    lookup = {gene:i for i,gene in enumerate(genes)}
    keep = panel.keep_channel.astype(str).str.lower().map({'true':True,'false':False})
    if keep.isna().any(): raise ValueError("keep_channel must be boolean")
    selected_panels = set(panel.panel_id) if panel_ids is None else panel_ids
    if not selected_panels.issubset(set(panel.panel_id)):
        raise ValueError("Embedding scope references an unknown panel")
    active = keep & panel.role.eq('downstream') & panel.mapping_status.eq('mapped') & panel.panel_id.isin(selected_panels)
    expressions = {}
    for row in panel[active].itertuples():
        key = getattr(row,'s1_original_marker',row.source_name)
        if key not in s1.index:
            raise ValueError(f"{study}/{row.panel_id}/{row.source_name}: missing explicit S1 source identity")
        entry = s1.loc[key]
        expression = entry['MAP-KG mapping'].strip()
        if entry['Unified marker'] != row.marker_id or not expression:
            raise ValueError(f"{key}: reviewed panel/S1 identity or mapping mismatch")
        if row.marker_id in expressions and expressions[row.marker_id] != expression:
            raise ValueError(f"Conflicting within-study mappings for {row.marker_id}")
        expressions[row.marker_id] = expression
    marker_ids = sorted(expressions)
    outputs = []
    for marker in marker_ids:
        mode,members = _expression_genes(expressions[marker])
        missing = [member for member in members if member not in lookup]
        if missing: raise ValueError(f"{marker}: missing bank members {missing}")
        selected = vectors[[lookup[member] for member in members]]
        outputs.append(selected.mean(dim=0) if mode == 'mean' else selected[0])
    embeddings = torch.stack(outputs) if outputs else torch.empty((0,manifest['dimension']),dtype=torch.float32)
    rows = {marker:i for i,marker in enumerate(marker_ids)}
    panel['embedding_row'] = pd.Series(pd.NA,index=panel.index,dtype='Int64')
    panel.loc[active,'embedding_row'] = panel.loc[active,'marker_id'].map(rows).astype('Int64')
    payload = {'embeddings':embeddings,'marker_ids':marker_ids,
               'mapping_expressions':[expressions[m] for m in marker_ids],
               'metadata':{'study':study,'panel_ids':sorted(selected_panels),'panel_identity':_panel_identity(panel_mapping),'bank':file_revision(bank_dir/'embeddings.pt'),
                           'bank_manifest':manifest,'gene_keys':file_revision(bank_dir/'gene_symbols.json'),
                           'manifest_file':file_revision(bank_dir/'manifest.json'),'s1':file_revision(s1_path),
                           'normalization':'raw projector output; no L2 normalization'}}
    return StudyEmbeddings(payload,_panel_identity(panel_mapping)),panel


def _validate(payload: dict, panel: pd.DataFrame) -> None:
    import torch
    if not _same_panel_identity(_panel_identity(panel), payload['metadata']['panel_identity']):
        raise ValueError('Current panel input identity differs from the embedding asset')
    ids = payload['marker_ids']; values = payload['embeddings']
    if len(set(ids)) != len(ids) or len(payload['mapping_expressions']) != len(ids) or values.ndim != 2 or values.shape[0] != len(ids) or values.dtype != torch.float32 or not torch.isfinite(values).all():
        raise ValueError("Study embedding payload contract failed")
    expected = panel.keep_channel.astype(str).str.lower().eq('true') & panel.role.eq('downstream') & panel.mapping_status.eq('mapped') & panel.panel_id.isin(payload['metadata']['panel_ids'])
    if panel.loc[expected,'embedding_row'].isna().any() or panel.loc[~expected,'embedding_row'].notna().any():
        raise ValueError("Embedding row assignment differs from actual channel use")
    for row in panel[panel.embedding_row.notna()].itertuples():
        index = int(row.embedding_row)
        if index != row.embedding_row or not 0 <= index < len(ids) or ids[index] != row.marker_id:
            raise ValueError("Panel and embedding payload row identity mismatch")


def publish_study_embeddings(bundle: StudyEmbeddings, panel_mapping: pd.DataFrame,
                             study_root: Path, *, nas_lock_root: Path | None = None) -> None:
    """Stage/reopen, pause cooperating consumers and publish vectors last."""
    import torch
    study_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.embeddings_',dir=study_root) as temporary:
        temp = Path(temporary)
        torch.save(bundle.payload,temp/'marker_embeddings.pt')
        panel_mapping.to_csv(temp/'panel_mapping.csv',index=False,encoding='utf-8-sig')
        reloaded = torch.load(temp/'marker_embeddings.pt',map_location='cpu',weights_only=True)
        mapped = pd.read_csv(temp/'panel_mapping.csv',keep_default_na=False)
        mapped['embedding_row'] = pd.to_numeric(mapped.embedding_row,errors='coerce').astype('Int64')
        _validate(reloaded,mapped)
        def publish_current() -> None:
            with resource_lock(study_root):
                acceptances = list(bundle.acceptances or [])
                if bundle.acceptance is not None:
                    acceptances.append(bundle.acceptance)
                if acceptances:
                    if bundle.data_root is None:
                        raise ValueError("Bound embedding publication requires data_root")
                    for acceptance in acceptances:
                        assert_reviewed_binding(study_root,data_root=bundle.data_root,binding=acceptance)
                current = pd.read_csv(study_root/'panel_mapping.csv',keep_default_na=False)
                if not _same_panel_identity(_panel_identity(current), bundle.panel_identity):
                    raise ValueError("Panel input changed while constructing study embeddings")
                (study_root/'marker_embeddings.pt').unlink(missing_ok=True)
                os.replace(temp/'panel_mapping.csv',study_root/'panel_mapping.csv')
                os.replace(temp/'marker_embeddings.pt',study_root/'marker_embeddings.pt')
                _validate(torch.load(study_root/'marker_embeddings.pt',map_location='cpu',weights_only=True),mapped)

        if nas_lock_root is None:
            publish_current()
        else:
            from cellweft.data.preparation.io_coordination import formal_write_lease
            with formal_write_lease(study_root,io_lock_root=nas_lock_root,priority='high'):
                publish_current()


def _vector_read_gate(nas_lock_root: Path | None):
    if nas_lock_root is None:
        return None
    from cellweft.data.preparation.remote_worker import limited_slots

    def gate(_path: Path):
        return limited_slots(nas_lock_root, 'nas_read', 2)

    return gate


def export_study_embedding_task(task: dict, *, data_root: Path, s1_path: Path,
                                bank_dir: Path, nas_lock_root: Path | None = None,
                                header_cache: dict | None = None,
                                read_gate=None) -> dict:
    """Build and publish one current study vector from exact D01 scopes.

    ``task['scopes']`` contains exact accepted bindings.  Their image sets are
    never replaced by a synthetic union for D01 validation; only their panel
    IDs are unioned for the one study-level vector payload.
    """
    study = str(task['study'])
    root = Path(data_root) / 'processed' / study
    scopes = task.get('scopes') or [{
        'scope_key': task.get('scope_key', ''),
        'image_ids': list(task.get('image_ids', [])),
        'input_binding': task.get('input_binding'),
    }]
    gate = read_gate or _vector_read_gate(nas_lock_root)
    panel = None
    panel_identity = None
    panel_ids = set(task.get('panel_ids', []))
    acceptances = []
    for scope in scopes:
        image_ids = list(scope.get('image_ids', []))
        if not image_ids:
            raise ValueError(f'{study}: vector scope has no image IDs')
        inputs = load_reviewed_study_inputs(root, data_root=Path(data_root), image_ids=image_ids,
                                            header_cache=header_cache, read_gate=gate)
        expected = scope.get('input_binding')
        if expected is not None and inputs.binding != expected:
            raise ValueError(f'{study}: D01 binding changed for vector scope')
        current_identity = _panel_identity(inputs.panel)
        if panel_identity is None:
            panel_identity = current_identity
            panel = inputs.panel
        elif not _same_panel_identity(panel_identity, current_identity):
            raise ValueError(f'{study}: panel input changed between vector scopes')
        panel_ids.update(inputs.images.panel_id.astype(str))
        acceptances.append(inputs.binding)
    if panel is None:
        raise ValueError(f'{study}: no vector scope')
    build_context = gate(Path(s1_path)) if gate is not None else nullcontext()
    with build_context:
        bundle, mapped = build_study_embeddings(study, panel, Path(s1_path), Path(bank_dir),
                                                panel_ids=panel_ids)
    bundle.data_root = Path(data_root)
    bundle.acceptances = acceptances
    publish_study_embeddings(bundle, mapped, root, nas_lock_root=nas_lock_root)
    return {'study': study, 'status': 'complete',
            'markers': len(bundle.payload['marker_ids']),
            'image_ids': sorted({image_id for scope in scopes
                                 for image_id in scope.get('image_ids', [])}),
            'panel_ids': sorted(panel_ids),
            'scopes': [scope.get('scope_key') for scope in scopes]}


@contextmanager
def published_study_embeddings(study_root: Path, *, expected_study: str) -> Iterator[tuple[dict, pd.DataFrame]]:
    """Yield matching current panel and CPU vectors while holding their read lease."""
    import torch
    with resource_lock(study_root,shared=True):
        panel = pd.read_csv(study_root/'panel_mapping.csv',keep_default_na=False)
        panel['embedding_row'] = pd.to_numeric(panel.embedding_row,errors='coerce').astype('Int64')
        payload = torch.load(study_root/'marker_embeddings.pt',map_location='cpu',weights_only=True)
        if payload["metadata"]["study"] != expected_study:
            raise ValueError("Embedding asset belongs to a different study")
        _validate(payload,panel)
        yield payload,panel
