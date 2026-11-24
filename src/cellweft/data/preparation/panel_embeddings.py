"""Independent panel-only marker vector handover and owner CPU execution."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.embeddings import (
    _expression_genes, _panel_identity, build_study_embeddings,
    publish_study_embeddings, published_study_embeddings,
)
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_read_lease


_ASSETS = ('panel_mapping.csv', 's1.csv', 'bank/manifest.json',
           'bank/gene_symbols.json', 'bank/embeddings.pt')


def _reviewed_panels(study: str, study_root: Path, *, historical_scope: dict | None = None
                     ) -> tuple[pd.DataFrame, list[str], int]:
    reviews = pd.read_csv(study_root / 'review_results.csv', keep_default_na=False)
    images = pd.read_parquet(study_root / 'images.parquet', columns=['image_id', 'panel_id'])
    panel = pd.read_csv(study_root / 'panel_mapping.csv', keep_default_na=False)
    selected = sorted(set(images.panel_id.astype(str)))
    if not selected or not set(selected).issubset(set(panel.panel_id.astype(str))):
        raise ValueError(f'{study}: actual image panels are missing from panel mapping')
    if historical_scope is None:
        current = reviews[reviews.record_id == study]
        for check in ('S01', 'P01', 'P02', 'P03'):
            rows = current[current.check_id == check]
            if rows.empty or rows.iloc[-1].result != 'pass':
                raise ValueError(f'{study}: current {check} review is not pass')
        for check in ('P01', 'P02'):
            details = json.loads(current[current.check_id == check].iloc[-1].details)
            declared = details.get('panel_ids', [details['panel_id']] if 'panel_id' in details else None)
            if declared is not None and not set(selected).issubset(set(declared)):
                raise ValueError(f'{study}: {check} does not cover actual panels')
            if declared is None and len(selected) != 1:
                raise ValueError(f'{study}: {check} lacks explicit multi-panel coverage')
        p03 = json.loads(current[current.check_id == 'P03'].iloc[-1].details)
        if not set(images.image_id.astype(str)).issubset(set(p03.get('image_ids', []))):
            raise ValueError(f'{study}: P03 does not cover actual images')
    return panel, selected, len(images)


def _load_historical_scope(path: Path, *, study: str, node_id: str,
                           approved_source_revision: dict | None = None) -> dict:
    if approved_source_revision is not None and file_revision(path) != approved_source_revision:
        raise ValueError(f'{study}: panel source adjudication lacks approved receipt version')
    receipt = json.loads(path.read_text())
    if receipt.get('kind') == 'independent_panel_only_source_adjudication':
        if approved_source_revision is None or file_revision(path) != approved_source_revision:
            raise ValueError(f'{study}: panel source adjudication lacks approved receipt version')
        if (receipt.get('study') != study or receipt.get('owner_node') != node_id
                or receipt.get('decision') != 'source_and_marker_provenance_sufficient_for_panel_only_vector'
                or receipt.get('formal_review_or_vector_writes') != 0):
            raise ValueError(f'{study}: panel source adjudication is absent or unaudited')
        return receipt
    if receipt.get('kind') != 'explicit_historical_panel_only_vector_scope_receipt':
        raise ValueError('Historical review scope receipt has the wrong kind')
    matching = [entry for entry in receipt['studies'] if entry['study'] == study]
    if len(matching) != 1 or matching[0]['owner_node'] != node_id:
        raise ValueError(f'{study}: historical review scope is absent or belongs to another owner')
    entry = matching[0]
    if any(entry['independent_pass_rows'].get(check, 0) < 1
           for check in ('S01', 'P01', 'P02', 'P03')):
        raise ValueError(f'{study}: historical review scope lacks an independent pass')
    return entry


def _assert_source_adjudication(entry: dict, *, source_versions: dict,
                                panel_ids: list[str], image_count: int,
                                panel: pd.DataFrame, data_root: Path) -> None:
    study = entry['study']
    if (entry['actual_panel_ids'] != panel_ids or len(panel_ids) != 1
            or entry['current_formal_image_context'] != image_count):
        raise ValueError(f'{study}: panel source adjudication image/panel scope changed')
    frozen = entry['source_versions']
    names = {'images.parquet', 'panel_mapping.csv', 'review_results.csv',
             'native_tiff_ordered_markers.json', 'channel_labels.csv',
             'native_tiff_source_proof', 'independent_review_report',
             'official_source_reception', 's1', 'bank_manifest',
             'bank_genes', 'bank_embeddings'}
    if set(frozen) != names:
        raise ValueError(f'{study}: panel source adjudication proof set changed')
    expected = {name: frozen[name] for name in
                ('images.parquet', 'panel_mapping.csv', 'review_results.csv')}
    expected.update({'s1.csv': frozen['s1'],
                     'bank/manifest.json': frozen['bank_manifest'],
                     'bank/gene_symbols.json': frozen['bank_genes'],
                     'bank/embeddings.pt': frozen['bank_embeddings']})
    if source_versions != expected:
        raise ValueError(f'{study}: panel source adjudication source version changed')
    for name in names - set(('images.parquet', 'panel_mapping.csv', 'review_results.csv',
                             's1', 'bank_manifest', 'bank_genes', 'bank_embeddings')):
        path = Path(frozen[name]['path'])
        if not path.resolve().is_relative_to(data_root.resolve()) or file_revision(path) != frozen[name]:
            raise ValueError(f'{study}: panel source adjudication proof version changed: {name}')

    evidence = entry['panel_source_evidence']
    if (evidence['current_image_to_native_source_memberships'] != image_count
            or evidence['source_stats_matching_prior_proofs'] != image_count
            or evidence['reused_independent_header_name_checks'] != image_count
            or evidence['all_channel_bounded_proofs'] + evidence['prior_sampled_page_proofs'] != image_count
            or evidence['mapped_downstream_expression_members_missing_from_bank'] != 0):
        raise ValueError(f'{study}: panel source adjudication lacks image/source identity proof')
    native = json.loads(Path(frozen['native_tiff_ordered_markers.json']['path']).read_text())
    labels = pd.read_csv(frozen['channel_labels.csv']['path'], keep_default_na=False)
    ordered = panel.sort_values('source_channel_index')
    count = len(ordered)
    if (native['panel_id'] != panel_ids[0] or len(native['mapping_rows']) != count
            or len(native['ordered_channel_names']) != count
            or evidence['ordered_native_channels'] != count
            or evidence['current_panel_rows'] != count
            or evidence['exact_study_S1_source_keys'] != count
            or labels['index'].tolist() != list(range(count))
            or labels.channel.tolist() != native['ordered_channel_names']
            or ordered.source_channel_index.tolist() != list(range(count))
            or ordered.source_name.tolist() != native['ordered_channel_names']):
        raise ValueError(f'{study}: panel source adjudication ordered channel identity changed')
    s1 = pd.read_csv(frozen['s1']['path'], keep_default_na=False)
    s1 = s1[s1.Study == study]
    if (len(s1) != count or s1['Original marker'].duplicated().any()
            or set(s1['Original marker']) != set(ordered.s1_original_marker)):
        raise ValueError(f'{study}: panel source adjudication S1 source keys changed')
    mapping = s1.set_index('Original marker')
    downstream = ordered[(ordered.role == 'downstream') & (ordered.mapping_status == 'mapped')]
    genes = set(json.loads(Path(frozen['bank_genes']['path']).read_text()))
    for row in downstream.itertuples():
        source = mapping.loc[row.s1_original_marker]
        if source['Unified marker'] != row.marker_id or not source['MAP-KG mapping']:
            raise ValueError(f'{study}: panel source adjudication S1 marker identity changed')
        if not set(_expression_genes(source['MAP-KG mapping'])[1]).issubset(genes):
            raise ValueError(f'{study}: panel source adjudication bank expression changed')
    if (len(downstream.marker_id.unique()) != evidence['mapped_downstream_markers']
            or int(ordered.keep_channel.sum()) != evidence['kept_channels']
            or int((ordered.role == 'segmentation-only').sum()) != evidence['nuclear_segmentation_only_channels']
            or int((ordered.role == 'exclude').sum()) != evidence['excluded_channels']):
        raise ValueError(f'{study}: panel source adjudication marker counts changed')
    proof = json.loads(Path(frozen['native_tiff_source_proof']['path']).read_text())
    if (proof['expected_images'] != image_count
            or proof['actual_validated_count'] != image_count
            or len(proof['verified_images']) != image_count):
        raise ValueError(f'{study}: panel source adjudication native proof count changed')
    images = pd.read_parquet(frozen['images.parquet']['path'],
                             columns=['source_files', 'source_channel_count', 'panel_id'])
    by_name = {Path(item['native_path']).name: item for item in proof['verified_images']}
    current_names = [Path(files[0]).name for files in images.source_files if len(files) == 1]
    if (len(current_names) != image_count or len(set(current_names)) != image_count
            or len(by_name) != image_count or set(by_name) != set(current_names)
            or set(images.panel_id.astype(str)) != set(panel_ids)
            or set(images.source_channel_count) != {count}):
        raise ValueError(f'{study}: panel source adjudication image membership changed')
    bounded = sampled = 0
    for files in images.source_files:
        current = data_root / files[0]
        item = by_name[current.name]
        native_path = data_root / item['native_path']
        if (not current.resolve().is_relative_to(data_root.resolve())
                or not native_path.resolve().is_relative_to(data_root.resolve())):
            raise ValueError(f'{study}: panel source adjudication native source escapes data root')
        source_stat = current.stat()
        native_stat = native_path.stat()
        expected_stat = item['file_stat']
        expected_pair = (expected_stat['size'], expected_stat['mtime_ns'])
        if ((source_stat.st_size, source_stat.st_mtime_ns) != expected_pair
                or (native_stat.st_size, native_stat.st_mtime_ns) != expected_pair):
            raise ValueError(f'{study}: panel source adjudication native source version changed')
        if 'channels' in item:
            bounded += 1
            if (item['status'] != 'verified' or item['panel_id'] != panel_ids[0]
                    or item['ordered_channel_names'] != native['ordered_channel_names']
                    or len(item['channels']) != count
                    or [channel['label'] for channel in item['channels']] != native['ordered_channel_names']):
                raise ValueError(f'{study}: panel source adjudication bounded proof changed')
        else:
            sampled += 1
            if (not item['legacy_proof']['valid']
                    or item['legacy_proof']['channels'] != count
                    or item['header_shape'][0] != count):
                raise ValueError(f'{study}: panel source adjudication sampled proof changed')
    if (bounded != evidence['all_channel_bounded_proofs']
            or sampled != evidence['prior_sampled_page_proofs']):
        raise ValueError(f'{study}: panel source adjudication proof stratum changed')
    report = json.loads(Path(frozen['independent_review_report']['path']).read_text())
    official = json.loads(Path(frozen['official_source_reception']['path']).read_text())
    if (report['study'] != study or report['status'] != 'partial'
            or report['expected_images'] != image_count
            or report['exported_images'] != image_count
            or official['study'] != study or official['owner'] != entry['owner_node']
            or official['aggregate_counts']['exported_roi'] != image_count
            or official['aggregate_counts']['historical_review']['headers_and_names_checked'] != image_count
            or official['new_D01'] != 0 or official['formal_changes'] is not False
            or 'unresolved' not in official['identity_decision']
            or 'pending' not in official['QC_admission_rule']):
        raise ValueError(f'{study}: panel source adjudication reports do not support source-only scope')
    reviews = pd.read_csv(frozen['review_results.csv']['path'], keep_default_na=False)
    for check, declared in entry['current_review_scopes'].items():
        rows = reviews[reviews.check_id == check]
        if rows.empty or any(str(rows.iloc[-1][key]) != declared[key]
                             for key in ('scope', 'record_id', 'result')):
            raise ValueError(f'{study}: panel source adjudication review scope changed: {check}')
    if any(entry['current_review_scopes'][check]['result'] != 'pass'
           for check in ('P01', 'P02', 'P03')):
        raise ValueError(f'{study}: panel source adjudication lacks panel review pass')


def _assert_historical_scope(entry: dict, *, source_versions: dict,
                             panel_ids: list[str], image_count: int,
                             panel: pd.DataFrame, data_root: Path) -> None:
    if entry.get('kind') == 'independent_panel_only_source_adjudication':
        _assert_source_adjudication(entry, source_versions=source_versions,
                                    panel_ids=panel_ids, image_count=image_count,
                                    panel=panel, data_root=data_root)
        return
    if entry['image_count'] != image_count or entry['actual_panel_ids'] != panel_ids:
        raise ValueError(f"{entry['study']}: historical review scope image/panel set changed")
    expected = {**entry['input_review_versions'],
                's1.csv': entry['s1_bank_versions']['s1'],
                'bank/manifest.json': entry['s1_bank_versions']['bank_manifest'],
                'bank/gene_symbols.json': entry['s1_bank_versions']['bank_genes'],
                'bank/embeddings.pt': entry['s1_bank_versions']['bank_embeddings']}
    if source_versions != expected:
        raise ValueError(f"{entry['study']}: historical review scope source version changed")
    reviews = pd.read_csv(entry['input_review_versions']['review_results.csv']['path'],
                          keep_default_na=False)
    for check in ('S01', 'P01', 'P02', 'P03'):
        if not ((reviews.check_id == check) & (reviews.result == 'pass')).any():
            raise ValueError(f"{entry['study']}: current historical {check} pass is absent")
    for frozen in entry['independent_review_report_versions']:
        path = Path(frozen['path'])
        if not path.resolve().is_relative_to(data_root.resolve()) or file_revision(path) != frozen:
            raise ValueError(f"{entry['study']}: independent review report version changed")


def build_panel_embedding_packet(study: str, *, data_root: Path, s1_path: Path,
                                 bank_dir: Path, node_id: str, packet_root: Path,
                                 io_lock_root: Path | None = None,
                                 review_scope_receipt: Path | None = None,
                                 approved_source_revision: dict | None = None) -> dict:
    """Freeze reviewed panel and S1/bank assets; publish ready.json last."""
    if node_id not in {'72', '73'} or not study or Path(study).name != study:
        raise ValueError('Study and owner node must be explicit')
    if packet_root.exists():
        raise FileExistsError(packet_root)
    study_root = data_root / 'processed' / study
    sources = {'panel_mapping.csv': study_root / 'panel_mapping.csv',
               's1.csv': s1_path,
               **{f'bank/{name}': bank_dir / name for name in
                  ('manifest.json', 'gene_symbols.json', 'embeddings.pt')}}
    scope = None
    if review_scope_receipt is not None:
        scope = {'receipt': file_revision(review_scope_receipt),
                 'entry': _load_historical_scope(review_scope_receipt,
                                                 study=study, node_id=node_id,
                                                 approved_source_revision=approved_source_revision)}
        if scope['entry'].get('kind') == 'independent_panel_only_source_adjudication':
            scope['approved_source_revision'] = approved_source_revision
    packet_root.mkdir(parents=True)
    try:
        with formal_read_lease(study_root, io_lock_root=io_lock_root), resource_lock(study_root, shared=True):
            panel, panel_ids, image_count = _reviewed_panels(
                study, study_root, historical_scope=scope['entry'] if scope else None)
            source_versions = {name: file_revision(path) for name, path in sources.items()}
            source_versions['images.parquet'] = file_revision(study_root / 'images.parquet')
            source_versions['review_results.csv'] = file_revision(study_root / 'review_results.csv')
            if scope is not None:
                _assert_historical_scope(scope['entry'], source_versions=source_versions,
                                         panel_ids=panel_ids, image_count=image_count,
                                         panel=panel, data_root=data_root)
            for name in _ASSETS:
                target = packet_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sources[name], target)
                if file_revision(sources[name]) != source_versions[name]:
                    raise ValueError(f'{study}: source changed while freezing {name}')
            if (file_revision(study_root / 'images.parquet') != source_versions['images.parquet']
                    or file_revision(study_root / 'review_results.csv') != source_versions['review_results.csv']):
                raise ValueError(f'{study}: panel review changed while freezing')
        ready = {'study': study, 'node_id': node_id, 'panel_ids': panel_ids,
                 'panel_identity': _panel_identity(panel),
                 'source_versions': source_versions,
                 'assets': {name: {key: file_revision(packet_root / name)[key]
                                   for key in ('size', 'mtime_ns')} for name in _ASSETS}}
        if scope is not None:
            if file_revision(review_scope_receipt) != scope['receipt']:
                raise ValueError(f'{study}: historical review receipt changed while freezing')
            ready['review_scope'] = scope
        write_json(packet_root / 'ready.json', ready)
        return ready
    except BaseException:
        shutil.rmtree(packet_root)
        raise


def validate_panel_embedding_packet(packet_root: Path, *, node_id: str) -> dict:
    ready = json.loads((packet_root / 'ready.json').read_text())
    if ready['node_id'] != node_id or not ready['study'] or Path(ready['study']).name != ready['study']:
        raise ValueError('Panel vector packet belongs to another owner or invalid study')
    if set(ready['assets']) != set(_ASSETS):
        raise ValueError('Panel vector packet assets differ from contract')
    for name in _ASSETS:
        current = file_revision(packet_root / name)
        if {key: current[key] for key in ('size', 'mtime_ns')} != ready['assets'][name]:
            raise ValueError(f'Panel vector packet asset changed: {name}')
    panel = pd.read_csv(packet_root / 'panel_mapping.csv', keep_default_na=False)
    if _panel_identity(panel) != ready['panel_identity']:
        raise ValueError('Panel vector packet identity changed')
    if not set(ready['panel_ids']).issubset(set(panel.panel_id.astype(str))):
        raise ValueError('Panel vector packet selects missing panels')
    return ready


def stage_owner_panel_packet(packet_root: Path, *, reused_resources: Path,
                             node_id: str) -> dict:
    """Link exact owner-local S1/bank replicas, then publish ready.json last."""
    pending = packet_root / 'ready.pending.json'
    ready = json.loads(pending.read_text())
    transferred_bytes = pending.stat().st_size + (packet_root / 'panel_mapping.csv').stat().st_size
    if ready['node_id'] != node_id:
        raise ValueError('Panel packet belongs to a different owner')
    reused_bytes = 0
    for name in _ASSETS[1:]:
        source = reused_resources / name
        target = packet_root / name
        current = file_revision(source)
        if {key: current[key] for key in ('size', 'mtime_ns')} != ready['assets'][name]:
            raise ValueError(f'Owner-local resource version differs: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        reused_bytes += current['size']
    os.replace(pending, packet_root / 'ready.json')
    validate_panel_embedding_packet(packet_root, node_id=node_id)
    return {'study': ready['study'], 'status': 'ready',
            'reused_owner_bytes': reused_bytes,
            'transferred_data_bytes': transferred_bytes}


def verify_panel_embedding_source(packet_root: Path, *, data_root: Path,
                                  io_lock_root: Path | None = None,
                                  approved_source_revision: dict | None = None) -> dict:
    """Host-side preflight/receipt check against the authoritative source head."""
    ready = json.loads((packet_root / 'ready.json').read_text())
    study = ready['study']
    study_root = data_root / 'processed' / study
    scope = ready.get('review_scope')
    if scope is not None:
        receipt_path = Path(scope['receipt']['path'])
        if file_revision(receipt_path) != scope['receipt']:
            raise ValueError(f'{study}: historical review scope receipt changed')
        if (scope['entry'].get('kind') == 'independent_panel_only_source_adjudication'
                and (approved_source_revision is None
                     or approved_source_revision != scope.get('approved_source_revision'))):
            raise ValueError(f'{study}: panel source adjudication lacks approved receipt version')
        current_entry = _load_historical_scope(receipt_path, study=study,
                                               node_id=ready['node_id'],
                                               approved_source_revision=approved_source_revision)
        if file_revision(receipt_path) != scope['receipt'] or current_entry != scope['entry']:
            raise ValueError(f'{study}: historical review scope receipt changed')
    for name in ('panel_mapping.csv', 'images.parquet', 'review_results.csv'):
        if ready['source_versions'][name]['path'] != str((study_root / name).resolve()):
            raise ValueError(f'{study}: {name} is not the authoritative study source')
    with formal_read_lease(study_root, io_lock_root=io_lock_root), resource_lock(study_root, shared=True):
        panel, selected, image_count = _reviewed_panels(
            study, study_root, historical_scope=scope['entry'] if scope else None)
        if selected != ready['panel_ids'] or _panel_identity(panel) != ready['panel_identity']:
            raise ValueError(f'{study}: current reviewed panel differs from packet')
        for name, frozen in ready['source_versions'].items():
            source = Path(frozen['path'])
            if not source.resolve().is_relative_to(data_root.resolve()):
                raise ValueError(f'{study}: source escapes authoritative data root: {name}')
            if file_revision(source) != frozen:
                raise ValueError(f'{study}: source version changed: {name}')
        if scope is not None:
            _assert_historical_scope(scope['entry'], source_versions=ready['source_versions'],
                                     panel_ids=selected, image_count=image_count,
                                     panel=panel, data_root=data_root)
    if scope is not None and file_revision(Path(scope['receipt']['path'])) != scope['receipt']:
        raise ValueError(f'{study}: historical review scope receipt changed')
    return {'study': study, 'status': 'current', 'panel_ids': selected,
            'source_versions': ready['source_versions']}


def audit_panel_embeddings(study_root: Path, *, study: str,
                           bank_dir: Path, source_versions: dict) -> dict:
    """Independently recompute every published marker from frozen bank rows."""
    import torch

    genes = json.loads((bank_dir / 'gene_symbols.json').read_text())
    source = torch.load(bank_dir / 'embeddings.pt', map_location='cpu', weights_only=True)
    lookup = {gene: row for row, gene in enumerate(genes)}
    with published_study_embeddings(study_root, expected_study=study) as (payload, panel):
        if payload['metadata'].get('panel_only_source_versions') != source_versions:
            raise ValueError(f'{study}: published vector was replaced before numeric audit')
        for row, expression in enumerate(payload['mapping_expressions']):
            mode, members = _expression_genes(expression)
            selected = torch.stack([source[lookup[gene]] for gene in members])
            expected = selected.mean(dim=0) if mode == 'mean' else selected[0]
            torch.testing.assert_close(payload['embeddings'][row], expected, rtol=0, atol=0)
        return {'study': study, 'markers': len(payload['marker_ids']),
                'dimension': int(payload['embeddings'].shape[1]),
                'panel_rows': len(panel), 'numeric_rows_checked': len(payload['marker_ids']),
                'vector_revision': file_revision(study_root / 'marker_embeddings.pt'),
                'panel_revision': file_revision(study_root / 'panel_mapping.csv')}


def execute_panel_embedding_packet(packet_root: Path, *, node_id: str,
                                   node_root: Path, io_lock_root: Path) -> dict:
    """Build on the owner CPU and publish against the packet's exact panel."""
    from cellweft.data.preparation.io_coordination import formal_write_lease
    from cellweft.data.preparation.study_node import _claim

    with formal_read_lease(packet_root, io_lock_root=io_lock_root):
        ready = validate_panel_embedding_packet(packet_root, node_id=node_id)
        study = ready['study']
        panel = pd.read_csv(packet_root / 'panel_mapping.csv', keep_default_na=False)
        bundle, mapped = build_study_embeddings(
            study, panel, packet_root / 's1.csv', packet_root / 'bank',
            panel_ids=set(ready['panel_ids']))
        validate_panel_embedding_packet(packet_root, node_id=node_id)
    bundle.payload['metadata']['panel_only_source_versions'] = ready['source_versions']
    study_root = node_root / 'processed' / study
    with formal_write_lease(study_root, io_lock_root=io_lock_root, priority='high'):
        with _claim(study_root / '.embedding_worker.lock') as owned:
            if not owned:
                raise RuntimeError(f'{study}: embedding worker is active; retry packet')
            with resource_lock(study_root):
                existing = study_root / 'panel_mapping.csv'
                if existing.is_file():
                    current = pd.read_csv(existing, keep_default_na=False)
                    if _panel_identity(current) != ready['panel_identity']:
                        raise ValueError(f'{study}: owner panel differs from frozen panel')
                else:
                    temporary = study_root / '.panel_mapping.csv.pending'
                    shutil.copy2(packet_root / 'panel_mapping.csv', temporary)
                    os.replace(temporary, existing)
            # The shared claim stays held across the short publication lock.
            publish_study_embeddings(bundle, mapped, study_root)
    with formal_read_lease(packet_root, io_lock_root=io_lock_root):
        result = audit_panel_embeddings(study_root, study=study,
                                        bank_dir=packet_root / 'bank',
                                        source_versions=ready['source_versions'])
    import socket
    start_ticks = int(Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19])
    return {**result, 'source_versions': ready['source_versions'],
            'panel_ids': ready['panel_ids'], 'status': 'complete',
            'physical_owner': socket.gethostname(), 'pid': os.getpid(),
            'start_ticks': start_ticks}
