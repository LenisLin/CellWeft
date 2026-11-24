"""Return only approved, frozen accepted products to NAS; never delete owner data."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from functools import partial
import signal
import traceback
from types import ModuleType

import pandas as pd

from cellweft.data.io.writers import write_json
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.storage_reception_relocation import (
    Relocation, activation_record, assert_completion_writable, relocation_for,
)
from cellweft.data.preparation.storage_reception_runtime import (
    load_storage_return_runner, select_storage_return_roots,
    storage_return_runner_lock,
)
from cellweft.data.preparation.storage_reception_dependencies import DependencyMap
from cellweft.data.preparation.storage_return import return_image
from cellweft.data.preparation.storage_return_study import finalize_study
from cellweft.data.preparation.storage_return_inputs import retain_image_inputs
from cellweft.data.preparation.remote_worker import shared_disk_reservation
from cellweft.data.preparation.io_coordination import default_io_root, formal_write_lease

PROJECT = Path('/home/lenislin/Experiment/projects/CellWeft')
NAS = Path('/mnt/NAS_21T/ProjectData/CellWeft')
SCRATCH_RUNTIME = PROJECT / '.cache/cellweft/runtime'
LOCK_ROOT = default_io_root(NAS)
MAX_COMPLETION_GENERATIONS = 32
_SUMMARY_TABLES = (
    'images.parquet', 'sample_metadata.csv', 'field_mapping.csv',
    'review_results.csv', 'panel_mapping.csv',
)
RECEPTION_ROOT = NAS / 'processed/_shared/storage_reception/completed_studies'
ROOT, COMPLETED, RUNTIME_MODE = select_storage_return_roots(
    PROJECT, NAS)


def switch_activated_reader(record: dict) -> ModuleType:
    """Fresh-import the durable activated root for the activation callback."""
    actual, _, _ = activation_record(RECEPTION_ROOT)
    if actual != record:
        raise ValueError('persisted activation differs from callback record')
    current = load_storage_return_runner(Path(__file__))
    if (current.RUNTIME_MODE != 'activated' or current.COMPLETED != RECEPTION_ROOT
            or current.ROOT != RECEPTION_ROOT / 'execution'
            or current.RECEPTION_ROOT != RECEPTION_ROOT
            or current.LOCK_ROOT != LOCK_ROOT):
        raise ValueError('fresh runner did not select the activated roots and old lock domain')
    return current


def assert_runner_root_active() -> None:
    if RUNTIME_MODE != 'activated' or COMPLETED != RECEPTION_ROOT:
        raise RuntimeError('storage-return runner is not bound to the activated processed root')
    activation_record(RECEPTION_ROOT)
    assert_completion_writable(COMPLETED, RECEPTION_ROOT)


@contextmanager
def runner_guard():
    """Hold the worker lifecycle lock and reject a retired completion root."""
    with storage_return_runner_lock(ROOT):
        assert_runner_root_active()
        yield


def reception_dependencies() -> DependencyMap | None:
    if RUNTIME_MODE != 'activated' or COMPLETED != RECEPTION_ROOT:
        return None
    activated = activation_record(COMPLETED)
    activation = activated[0]
    dependencies = DependencyMap(COMPLETED)
    if dependencies.revision != activation.get('dependencies_revision'):
        raise ValueError('activated dependency version changed')
    if activation_record(COMPLETED)[1:] != activated[1:]:
        raise ValueError('activation changed during dependency read')
    return dependencies


def _relocation(study: str) -> Relocation | None:
    if RUNTIME_MODE != 'activated' or COMPLETED != RECEPTION_ROOT:
        return None
    return relocation_for(COMPLETED, study)


def reception_scope_path(study: str, path: Path) -> Path:
    relocation = _relocation(study)
    if relocation is not None:
        old = relocation.record['entries'][-1]['old']['path']
        if str(path.absolute()) == old:
            return Path(relocation.scope['path'])
    return path


def completion_binding_path(study: str, generation: int = 1) -> Path:
    """Return an exact immutable sidecar path, retaining the legacy first name."""
    if not 1 <= generation <= MAX_COMPLETION_GENERATIONS:
        raise ValueError('completion binding generation is outside the bounded range')
    suffix = '' if generation == 1 else f'.{generation}'
    return COMPLETED / f'{study}.successor{suffix}.json'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_scope(frozen: dict, approved: dict, images: pd.DataFrame) -> None:
    """Reject substituted image keys before any target mutation."""
    study = approved['study']
    records = frozen['images']
    if (frozen.get('schema') != 'cellweft.storage_return_study_scope.v1'
            or frozen.get('study') != study or frozen.get('owner') != approved['owner']
            or frozen.get('deletion_authorized') is not False
            or frozen.get('source') != approved['owner_path']
            or Path(frozen['target']) != NAS / 'processed' / study):
        raise ValueError(f'{study}: frozen identity differs from approval')
    keys = [(record['study'], record['image_id']) for record in records]
    current = list(zip(images.study.astype(str), images.image_id.astype(str)))
    if (len(keys) != approved['current_images'] or len(keys) != len(set(keys))
            or len(current) != len(set(current)) or set(keys) != set(current)
            or any(key[0] != study or Path(key[1]).name != key[1]
                   or key[1] in {'.', '..'} for key in keys)
            or any(str(record['node']) != approved['owner'] for record in records)):
        raise ValueError(f'{study}: exact approved (study,image_id) keys differ')
    for record in records:
        if not record.get('request_digest') or not record.get('source_input_identity'):
            raise ValueError(f'{study}/{record["image_id"]}: missing accepted evidence')
    study_products = frozen['study_product_revisions']
    if (set(study_products) != {'panel_mapping.csv', 'marker_embeddings.pt'}
            or any(Path(revision['path']) != Path(frozen['source']) / name
                   for name, revision in study_products.items())):
        raise ValueError(f'{study}: frozen study product paths differ from approved owner')
    dependencies = reception_dependencies()
    for revision in [frozen['accepted_index'], *frozen['study_product_revisions'].values()]:
        expected = dependencies.translate(revision) if dependencies is not None else revision
        actual = file_revision(Path(expected['path']))
        if any(actual[key] != expected[key] for key in ('path', 'size', 'mtime_ns')):
            raise ValueError(f'{study}: frozen evidence version changed: {revision["path"]}')


def _completion_chain(study: str) -> list[tuple[Path, dict, dict]]:
    """Read only this study's bounded chain and verify every immutable link."""
    relocation = _relocation(study)
    original_path = COMPLETED / f'{study}.json'
    original = json.loads(original_path.read_text())
    chain = [(original_path, original, file_revision(original_path))]
    missing = False
    for generation in range(1, MAX_COMPLETION_GENERATIONS + 1):
        path = completion_binding_path(study, generation)
        if not path.exists():
            missing = True
            continue
        if missing:
            raise ValueError(f'{study}: successor completion generation gap')
        binding = json.loads(path.read_text())
        expected_schema = ('cellweft.storage_return_completion_binding.v1' if generation == 1
                           else 'cellweft.storage_return_completion_binding.v2')
        if binding.get('schema') != expected_schema:
            raise ValueError(f'{study}: successor completion binding schema is invalid')
        if binding.get('study') != study:
            raise ValueError(f'{study}: successor completion binding study is invalid')
        previous_path, previous, previous_revision = chain[-1]
        previous_state = previous if generation == 1 else previous['successor']
        predecessor = binding.get('predecessor')
        if not isinstance(predecessor, Mapping):
            raise ValueError(f'{study}: successor completion binding lacks predecessor')
        predecessor_revision = predecessor.get('revision')
        if not isinstance(predecessor_revision, dict):
            raise ValueError(f'{study}: predecessor lacks exact revision')
        physical_revision = (relocation.translate(predecessor_revision)
                             if relocation is not None else predecessor_revision)
        if (predecessor.get('path') != predecessor_revision.get('path')
                or Path(physical_revision.get('path', '')).resolve() != previous_path.resolve()
                or physical_revision != previous_revision
                or predecessor.get('frozen_scope_revision') != previous_state.get('frozen_scope_revision')
                or predecessor.get('summary_revision') != previous_state.get('summary_revision')):
            raise ValueError(f'{study}: successor completion predecessor changed')
        if generation > 1 and binding.get('generation') != generation:
            raise ValueError(f'{study}: successor completion generation is invalid')
        successor = binding.get('successor')
        if not isinstance(successor, Mapping):
            raise ValueError(f'{study}: successor completion binding lacks successor')
        image_ids = original.get('image_ids')
        if image_ids is not None and successor.get('image_ids') != image_ids:
            raise ValueError(f'{study}: successor completion image scope differs from predecessor')
        migration = binding.get('migration')
        if not isinstance(migration, Mapping) or migration.get('status') != 'metadata_only':
            raise ValueError(f'{study}: successor completion migration proof is invalid')
        chain.append((path, binding, file_revision(path)))
    if relocation is not None:
        relocation.check()
    return chain


def validate_summary_table_revisions(study: str, target: Path) -> None:
    """Reject a versioned summary whose published table snapshot is stale.

    Historical summaries without ``nas_five_table_revisions`` retain their
    existing completion/scope/summary binding contract.
    """
    summary_path = Path(target) / 'processing_summary.json'
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'{study}: processing summary is not valid JSON') from error
    if not isinstance(summary, Mapping):
        raise ValueError(f'{study}: processing summary must be a JSON object')
    if 'nas_five_table_revisions' not in summary:
        return
    revisions = summary['nas_five_table_revisions']
    if not isinstance(revisions, Mapping) or not isinstance(revisions.get('after'), Mapping):
        raise ValueError(f'{study}: processing summary table revisions are malformed')
    after = revisions['after']
    for name in _SUMMARY_TABLES:
        expected = after.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(f'{study}: processing summary lacks table revision for {name}')
        try:
            actual = file_revision(Path(target) / name)
        except FileNotFoundError as error:
            raise ValueError(f'{study}: processing summary table is missing: {name}') from error
        if dict(expected) != actual:
            raise ValueError(f'{study}: current {name} differs from processing summary table revisions')


def received_study(study: str, frozen_path: Path, target: Path) -> bool:
    """Require the latest validated generation to bind the current scope/summary."""
    _relocation(study)  # Check required activated members before treating absence as new.
    path = COMPLETED / f'{study}.json'
    if not path.exists():
        return False
    frozen_path = reception_scope_path(study, frozen_path)
    relocation = _relocation(study)
    chain = _completion_chain(study)
    record = chain[-1][1]
    current = record if len(chain) == 1 else record['successor']
    scope_revision = current.get('frozen_scope_revision')
    if relocation is not None:
        scope_revision = relocation.translate(scope_revision)
    if (scope_revision != file_revision(frozen_path)
            or current.get('summary_revision') != file_revision(target / 'processing_summary.json')):
        if len(chain) == 1:
            raise ValueError(f'{study}: completed return proof changed; reconcile before retry')
        raise ValueError(f'{study}: successor completion binding does not match current scope/summary')
    validate_summary_table_revisions(study, target)
    if len(chain) > 2:
        frozen = json.loads(frozen_path.read_text())
        if [image['image_id'] for image in frozen['images']] != chain[0][1]['image_ids']:
            raise ValueError(f'{study}: successor completion image scope differs from predecessor')
    if relocation is not None:
        relocation.check()
    return True


def _sync_completion(path: Path) -> None:
    """Finish durability on both publication and interrupted-publication retry."""
    with path.open('rb') as handle:
        os.fsync(handle.fileno())
    descriptor = os.open(COMPLETED, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def complete_study(study: str, frozen_path: Path, frozen: dict, receipt: dict,
                   *, completion_path: Path | None = None,
                   migration: Mapping | None = None,
                   expected_previous_revision: Mapping | None = None,
                   prepublish_check: Callable[[], None] | None = None) -> None:
    frozen_path = reception_scope_path(study, frozen_path)
    generation = None
    if completion_path is not None:
        generation = next((number for number in range(1, MAX_COMPLETION_GENERATIONS + 1)
                           if Path(completion_path).resolve()
                           == completion_binding_path(study, number).resolve()), None)
        if generation is None:
            raise ValueError(f'{study}: successor binding path is not the controlled sidecar')
        if generation > 1 and expected_previous_revision is None:
            raise ValueError(f'{study}: successor binding requires explicit previous revision')
    target = Path(frozen['target'])
    expected_scope = file_revision(frozen_path)
    expected_summary = file_revision(target / 'processing_summary.json')
    predecessor_path = COMPLETED / f'{study}.json'
    expected_predecessor = (file_revision(predecessor_path)
                            if completion_path is not None else None)
    with formal_write_lease(COMPLETED, io_lock_root=LOCK_ROOT):
        with resource_lock(target, shared=True):
            with resource_lock(COMPLETED):
                assert_runner_root_active()
                current_scope = file_revision(frozen_path)
                current_summary = file_revision(target / 'processing_summary.json')
                if (json.loads(frozen_path.read_text()) != frozen
                        or current_scope != expected_scope or current_summary != expected_summary):
                    raise ValueError(f'{study}: completion inputs changed before publication')
                validate_summary_table_revisions(study, target)
                image_ids = [image['image_id'] for image in frozen['images']]
                if completion_path is None:
                    path = predecessor_path
                    record = {'schema': 'cellweft.storage_return_completed.v1', 'study': study,
                              'received_at': now(), 'frozen_scope_revision': current_scope,
                              'summary_revision': current_summary, 'image_ids': image_ids,
                              'source_retained': True, 'deletion_executed': False,
                              'receipt': receipt}
                else:
                    path = Path(completion_path)
                    if not isinstance(migration, Mapping) or migration.get('status') != 'metadata_only':
                        raise ValueError(f'{study}: successor binding requires metadata-only migration proof')
                    if file_revision(predecessor_path) != expected_predecessor:
                        raise ValueError(f'{study}: completion predecessor changed before publication')
                    chain = _completion_chain(study)
                    if chain[0][1].get('image_ids') != image_ids:
                        raise ValueError(f'{study}: successor image order differs from predecessor')
                    # A published generation can be retried after a receipt-writing failure.
                    if path.exists():
                        if len(chain) != generation + 1:
                            raise ValueError(f'{study}: unknown later completion generation exists')
                        existing = chain[-1][1]
                        if (existing['successor'] != {
                                'frozen_scope_revision': current_scope,
                                'summary_revision': current_summary, 'image_ids': image_ids}
                                or existing.get('migration') != dict(migration)
                                or existing.get('receipt') != receipt
                                or (generation > 1 and existing['predecessor']['revision']
                                    != expected_previous_revision)):
                            raise ValueError(f'{study}: immutable completion binding already exists')
                        if prepublish_check is not None:
                            prepublish_check()
                        if (file_revision(frozen_path) != current_scope
                                or file_revision(target / 'processing_summary.json') != current_summary):
                            raise ValueError(f'{study}: completion inputs changed before publication')
                        validate_summary_table_revisions(study, target)
                        if any(file_revision(item_path) != revision
                               for item_path, _, revision in chain):
                            raise ValueError(f'{study}: completion predecessor changed before publication')
                        _sync_completion(path)
                        return
                    if len(chain) != generation:
                        raise ValueError(f'{study}: successor must extend the current completion generation')
                    previous_path, previous, predecessor_revision = chain[-1]
                    if (generation > 1 and predecessor_revision != expected_previous_revision):
                        raise ValueError(f'{study}: completion predecessor changed before publication')
                    predecessor = previous if generation == 1 else previous['successor']
                    record = {
                        'schema': ('cellweft.storage_return_completion_binding.v1' if generation == 1
                                   else 'cellweft.storage_return_completion_binding.v2'),
                        'study': study,
                        'bound_at': now(),
                        'predecessor': {
                            'path': str(previous_path.resolve()),
                            'revision': predecessor_revision,
                            'frozen_scope_revision': predecessor['frozen_scope_revision'],
                            'summary_revision': predecessor['summary_revision'],
                        },
                        'successor': {
                            'frozen_scope_revision': current_scope,
                            'summary_revision': current_summary,
                            'image_ids': image_ids,
                        },
                        'migration': dict(migration),
                        'receipt': receipt,
                    }
                    if generation > 1:
                        record['generation'] = generation
                if path.exists():
                    raise ValueError(f'{study}: immutable completion binding already exists')
                if prepublish_check is not None:
                    prepublish_check()
                if (file_revision(frozen_path) != current_scope
                        or file_revision(target / 'processing_summary.json') != current_summary):
                    raise ValueError(f'{study}: completion inputs changed before publication')
                validate_summary_table_revisions(study, target)
                if completion_path is not None:
                    if any(file_revision(item_path) != revision
                           for item_path, _, revision in chain):
                        raise ValueError(f'{study}: completion predecessor changed before publication')
                if path.exists():
                    raise ValueError(f'{study}: immutable completion binding already exists')
                temporary = path.with_name(path.name + '.partial')
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                allow_nan=False) + '\n')
                with temporary.open('rb') as handle:
                    os.fsync(handle.fileno())
                # Link the closed file without replacement, including for uncooperative writers.
                try:
                    os.link(temporary, path)
                except FileExistsError as error:
                    raise ValueError(f'{study}: immutable completion binding already exists') from error
                temporary.unlink()
                _sync_completion(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', action='append', default=[])
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    dependencies = reception_dependencies()
    approval = (dependencies.read_json('approval') if dependencies is not None
                else json.loads((ROOT / 'approval.json').read_text()))
    if approval['user_answer'] != '批准按既定门槛回传，删除另审' or approval['deletion_authorized']:
        raise ValueError('The exact non-destructive return approval is required')
    scope = (dependencies.read_json('approved_scope') if dependencies is not None
             else json.loads((PROJECT / approval['scope_file']).read_text()))
    approved = {row['study']: row for row in scope['studies']}
    if not set(args.study).issubset(approved):
        parser.error('Selected study is outside the approved migration scope')
    plan = (dependencies.read_json('disposition') if dependencies is not None else json.loads((PROJECT / 'cache/data_return_cleanup_planning_20261106/disposition_plan.json').read_text()))
    estimates = {row['study']: row.get('owner_tree_logical_bytes') for row in plan['studies']}
    data_inventory = (dependencies.read_json('inventory') if dependencies is not None else json.loads((PROJECT / 'cache/data_return_cleanup_planning_20261106/inventory_data_volume/inventory.json').read_text()))
    estimates.update({row['study']: row['formal_required_bytes']
                      for row in data_inventory['owner72_formal_products']['records']})
    names = args.study or sorted(approved, key=lambda name: estimates.get(name) or 10**30)
    ROOT.mkdir(exist_ok=True)
    with runner_guard():
        state = {'schema': 'cellweft.storage_return_execution.v1', 'status': 'running',
                 'pid': os.getpid(), 'started_at': now(), 'approved_studies': 103,
                 'approved_images': 19931, 'selected_studies': names, 'completed_studies': [],
                 'held_studies': [], 'current': None, 'processed_images_this_run': 0,
                 'deletion_executed': False}
        stop_requested = False
        def stop_after_image(signum, frame) -> None:
            nonlocal stop_requested
            stop_requested = True
        signal.signal(signal.SIGTERM, stop_after_image)
        signal.signal(signal.SIGINT, stop_after_image)
        def report(event: dict) -> None:
            state['updated_at'] = now()
            write_json(ROOT / 'execution_state.json', state)
            with (ROOT / 'events.jsonl').open('a') as log:
                log.write(json.dumps({'at': now(), **event}, ensure_ascii=False) + '\n')
                log.flush()
            print(json.dumps(event, ensure_ascii=False), flush=True)
        report({'status': 'started', 'selected_studies': len(names)})
        for study in names:
            relocation = _relocation(study)
            frozen_path = (Path(relocation.scope['path']) if relocation is not None
                           else ROOT / 'studies' / f'{study}.json')
            if not frozen_path.is_file():
                state['held_studies'].append({'study': study, 'reason': 'current accepted source scope not frozen'})
                report({'study': study, 'status': 'hold', 'reason': 'scope_not_frozen'})
                continue
            frozen = json.loads(frozen_path.read_text())
            source, target = Path(frozen['source']), Path(frozen['target'])
            state['current'] = {'study': study, 'stage': 'study_preflight'}
            try:
                if file_revision(target / 'images.parquet') != frozen['images_table_revision']:
                    raise ValueError('NAS approved image table changed after freezing')
                if received_study(study, frozen_path, target):
                    state['completed_studies'].append(study)
                    report({'study': study, 'status': 'previously_received'})
                    continue
                images = pd.read_parquet(target / 'images.parquet')
                validate_scope(frozen, approved[study], images)
                image_ids = [row['image_id'] for row in frozen['images']]
                input_rows = images.set_index('image_id')
                for accepted in frozen['images']:
                    image_id = accepted['image_id']
                    if stop_requested:
                        state['status'] = 'stopped_at_image_boundary'
                        state['current'] = None
                        report({'status': state['status']})
                        return
                    if args.max_images is not None and state['processed_images_this_run'] >= args.max_images:
                        state['status'] = 'bounded_batch_complete'
                        state['current'] = None
                        report({'status': state['status']})
                        return
                    state['current'] = {'study': study, 'image_id': image_id, 'stage': 'input_retention'}
                    report({'study': study, 'image_id': image_id, 'status': 'returning'})
                    accepted['accepted_index'] = frozen['accepted_index']
                    checkpoint = ROOT / 'images' / study / f'{image_id}.json'
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    started = time.monotonic()
                    assert_runner_root_active()
                    input_assets = retain_image_inputs(
                        target, image_id, input_rows.loc[image_id].to_dict(), accepted,
                        data_root=NAS, lock_root=LOCK_ROOT,
                        prepublish_check=assert_runner_root_active)
                    if any(asset['new_hardlink'] for asset in input_assets):
                        report({'study': study, 'image_id': image_id, 'status': 'input_links_published'})
                    state['current']['stage'] = 'return'
                    report({'study': study, 'image_id': image_id, 'status': 'returning_products'})
                    assert_runner_root_active()
                    receipt = return_image(source / 'images' / image_id,
                                           target / 'images' / image_id, accepted,
                                           checkpoint=checkpoint, lock_root=LOCK_ROOT,
                                           shared_disk_reservation=partial(
                                               shared_disk_reservation, LOCK_ROOT,
                                               owner=f'storage_return:{study}/{image_id}'),
                                           prepublish_check=assert_runner_root_active)
                    state['processed_images_this_run'] += 1
                    report({'study': study, 'image_id': image_id, 'status': 'received',
                            'elapsed_seconds': round(time.monotonic() - started, 2),
                            'receipt': str(checkpoint), 'result': receipt.get('status')})
                evidence = {'user_confirmed': True,
                            'technical_reception': frozen['accepted_index']['path'],
                            'sampling_policy': 'study_stratified_v1',
                            'accepted_index': frozen['accepted_index'],
                            'historical_exclusions': frozen['historical_exclusions'],
                            'existing_evidence': frozen['existing_evidence'],
                            'images_table_revision': frozen['images_table_revision'],
                            'study_product_revisions': frozen['study_product_revisions'],
                            'approved': str(ROOT / 'approval.json'),
                            'frozen_scope': str(frozen_path)}
                state['current'] = {'study': study, 'stage': 'study_reception'}
                report({'study': study, 'status': 'finalizing_study'})
                assert_runner_root_active()
                receipt = finalize_study(target, source, image_ids=image_ids, evidence=evidence,
                                         raw=frozen['raw'], lock_root=LOCK_ROOT,
                                         prepublish_check=assert_runner_root_active)
                assert_runner_root_active()
                write_json(ROOT / f'{study}.receipt.json', receipt)
                complete_study(study, frozen_path, frozen, receipt)
                state['completed_studies'].append(study)
                report({'study': study, 'status': 'study_received', 'images': len(image_ids)})
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                failure = {'study': study, 'reason': str(error),
                           'image_id': state['current'].get('image_id'),
                           'stage': state['current']['stage'],
                           'error_type': type(error).__name__,
                           'errno': getattr(error, 'errno', None),
                           'filename': getattr(error, 'filename', None),
                           'filename2': getattr(error, 'filename2', None),
                           'traceback': traceback.format_exc()}
                state['held_studies'].append(failure)
                report({**failure, 'status': 'hold'})
        state['status'] = 'completed' if not state['held_studies'] else 'completed_with_holds'
        state['current'] = None
        report({'status': state['status'], 'completed_studies': len(state['completed_studies']),
                'held_studies': len(state['held_studies'])})


if __name__ == '__main__':
    main()
