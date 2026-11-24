"""Relocate immutable storage-reception evidence without reading scientific arrays."""
from __future__ import annotations

import json
import fcntl
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_write_lease

SCHEMA = 'cellweft.storage_reception_relocation.v1'
MAX_GENERATIONS = 32


@contextmanager
def storage_return_runner_lock(root: Path) -> Iterator[None]:
    """Exclude worker lifecycles while a legacy completion root is activated."""
    with (Path(root) / '.runner.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _study(study: str) -> None:
    if not study or Path(study).name != study or study in {'.', '..'} or '\\' in study:
        raise ValueError('invalid study component')


def _canonical(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f'noncanonical or symlink path: {path}')
    return path


def _revision(value: dict) -> dict:
    if (not isinstance(value, dict) or set(value) != {'path', 'size', 'mtime_ns'}
            or type(value['size']) is not int or value['size'] < 0
            or type(value['mtime_ns']) is not int):
        raise ValueError('invalid exact revision')
    _canonical(Path(value['path']))
    return value


def _name(study: str, generation: int) -> str:
    if generation == 0:
        return f'{study}.json'
    suffix = '' if generation == 1 else f'.{generation}'
    return f'{study}.successor{suffix}.json'


def manifest_path(completed: Path, study: str) -> Path:
    _study(study)
    return completed / 'relocations' / f'{study}.json'


def activation_record(completed: Path, legacy_completed: Path | None = None) -> tuple[dict, dict, dict]:
    path, fence_path = completed / 'relocation_activation.json', completed / 'legacy_root_retirement.json'
    revision, fence_revision = file_revision(path), file_revision(fence_path)
    record, fence = json.loads(_read(path)), json.loads(_read(fence_path))
    recorded_legacy = record.get('legacy_completed')
    if not isinstance(recorded_legacy, str):
        raise ValueError('activation legacy-root identity is missing')
    retired_root = _canonical(Path(recorded_legacy))
    if legacy_completed is not None and _canonical(legacy_completed) != retired_root:
        raise ValueError('activation legacy-root identity differs')
    if (record.get('schema') != 'cellweft.storage_reception_activation.v1'
            or record.get('legacy_completed') != str(retired_root)
            or fence.get('schema') != 'cellweft.storage_reception_retired.v1'
            or fence.get('legacy_completed') != str(retired_root)
            or fence.get('destination') != str(completed) or fence.get('activation') != record):
        raise ValueError('activation differs from the immutable retirement plan')
    if file_revision(path) != revision or file_revision(fence_path) != fence_revision:
        raise ValueError('reception activation changed during read')
    return record, revision, fence_revision


def relocation_for(completed: Path, study: str, *,
                   legacy_completed: Path | None = None) -> Relocation | None:
    """The activation roster makes missing migration manifests fail closed."""
    _study(study)
    activation = completed / 'relocation_activation.json'
    record, activation_revision, fence_revision = activation_record(completed, legacy_completed)
    retired_root = Path(record['legacy_completed'])
    if completed == retired_root:
        return None
    members = record.get('studies')
    if not isinstance(members, dict):
        raise ValueError('invalid relocation activation members')
    path = manifest_path(completed, study)
    if study not in members:
        if path.exists():
            raise ValueError('unregistered relocation manifest')
        if (file_revision(activation) != activation_revision
                or file_revision(completed / 'legacy_root_retirement.json') != fence_revision):
            raise ValueError('relocation activation changed during read')
        return None  # New native receptions can be published after activation.
    if _revision(members[study]) != file_revision(path):
        raise ValueError('activated relocation manifest changed')
    relocation = Relocation(completed, study, legacy_completed=retired_root)
    relocation.activation_revision = activation_revision
    relocation.fence_revision = fence_revision
    relocation.check()
    return relocation


def assert_completion_writable(completed: Path, retirement_root: Path) -> None:
    """Call under COMPLETED's existing exclusive lock before any publication."""
    path = retirement_root / 'legacy_root_retirement.json'
    if path.exists():
        record = json.loads(_read(path))
        if (record.get('schema') != 'cellweft.storage_reception_retired.v1'
                or record.get('destination') != str(retirement_root)):
            raise ValueError('invalid completion-root retirement fence')
        if record.get('legacy_completed') == str(completed):
            raise ValueError('completion root retired for relocation; use the activated root')


def _validate_sources(runner: ModuleType, relocation: Relocation) -> None:
    """CAS the complete old chain, including absence of any later generation."""
    record = relocation.record
    study = record['study']
    chain = runner._completion_chain(study)
    expected = record['entries'][:record['completion_count']]
    if ([revision for _, _, revision in chain] != [entry['old'] for entry in expected]
            or len(chain) != record['completion_count']):
        raise ValueError('source completion head changed after relocation staging')
    for entry, identity in zip(record['entries'], record['source_identities'], strict=True):
        source, destination = Path(entry['old']['path']), Path(entry['new']['path'])
        current_identity = list(_stamp(source))
        identity_matches = current_identity == identity
        # NFS remounts can change st_dev while preserving the file ID. Keep
        # inode, revision and exact bytes bound for every relocated member.
        if current_identity[1:] == identity[1:]:
            identity_matches = True
        if (file_revision(source) != entry['old'] or not identity_matches
                or _read(source) != _read(destination)):
            raise ValueError('old evidence changed after relocation staging')
    summary = Path(record['summary_revision']['path'])
    summary_identity = list(_stamp(summary))
    if (file_revision(summary) != record['summary_revision']
            or (summary_identity != record['summary_identity']
                and summary_identity[1:] != record['summary_identity'][1:])):
        raise ValueError('source summary changed after relocation staging')
    scope = Path(record['entries'][-1]['old']['path'])
    if not runner.received_study(study, scope, summary.parent):
        raise ValueError('old study is no longer received')
    relocation.check()


def _activate_locked(completed: Path, studies: list[str], *, legacy_completed: Path,
                     runner: ModuleType, switch: Callable[[dict], ModuleType],
                     dependencies_required: bool = True) -> dict:
    """CAS, fence old writers, activate and switch while holding the old locks.

    Deploy fence-aware writers and drain pre-upgrade processes first. `switch`
    must atomically persist the new root and return a freshly configured reader.
    After fence publication failures recover forward by repeating this call.
    """
    if len(studies) != len(set(studies)):
        raise ValueError('duplicate activation study')
    completed, legacy_completed = _canonical(completed), _canonical(legacy_completed)
    if runner.COMPLETED != legacy_completed:
        raise ValueError('activation requires a legacy-root validator')
    if runner.RECEPTION_ROOT != completed:
        raise ValueError('legacy writer retirement fence must target the exact new root')
    relocations = [Relocation(completed, study, legacy_completed=legacy_completed)
                   for study in sorted(studies)]
    targets = sorted({Path(item.summary['path']).parent for item in relocations})
    with formal_write_lease(legacy_completed, io_lock_root=runner.LOCK_ROOT), ExitStack() as stack:
        for target in targets:
            stack.enter_context(resource_lock(target, shared=True))
        stack.enter_context(resource_lock(legacy_completed))
        stack.enter_context(resource_lock(completed))
        dependency_revision = None
        if dependencies_required or (completed / 'dependencies.json').exists():
            from cellweft.data.preparation.storage_reception_dependencies import DependencyMap
            dependencies = DependencyMap(completed)
            dependencies.validate_sources()
            dependency_revision = dependencies.revision
        members = {}
        # Refuse an omitted legacy completion, including new studies received
        # after the caller built its migration list. Only exact controlled names.
        originals = {path.stem for path in legacy_completed.glob('*.json')
                     if '.successor' not in path.stem and path.name != 'retired_for_relocation.json'}
        if originals != set(studies):
            raise ValueError('activation roster differs from complete legacy study set')
        for relocation in relocations:
            _validate_sources(runner, relocation)
            members[relocation.record['study']] = relocation.revision
        record = {'schema': 'cellweft.storage_reception_activation.v1',
                  'legacy_completed': str(legacy_completed), 'studies': members,
                  'dependencies_revision': dependency_revision}
        fence = {'schema': 'cellweft.storage_reception_retired.v1',
                 'legacy_completed': str(legacy_completed),
                 'destination': str(completed), 'activation': record}
        # Fence first. A crash here leaves old writes disabled and the exact
        # target roster recoverable; retry publishes the same activation.
        _publish(completed / 'legacy_root_retirement.json',
                 (json.dumps(fence, indent=2) + '\n').encode())
        _publish(completed / 'relocation_activation.json',
                 (json.dumps(record, indent=2) + '\n').encode())
        current = switch(record)
        if current.COMPLETED != completed or current.LEGACY_COMPLETED != legacy_completed:
            raise ValueError('switch did not return the activated reception reader')
        for relocation in relocations:
            study = relocation.record['study']
            if not current.received_study(study, Path(relocation.scope['path']),
                                          Path(relocation.summary['path']).parent):
                raise ValueError('switched reception reader did not accept the staged study')
        return record


def activate(completed: Path, studies: list[str], *, legacy_completed: Path,
             runner: ModuleType, switch: Callable[[dict], ModuleType],
             dependencies_required: bool = True) -> dict:
    """Activate under the old runner lifecycle lock, before taking NAS leases."""
    runner.ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with storage_return_runner_lock(runner.ROOT):
            return _activate_locked(
                completed, studies, legacy_completed=legacy_completed,
                runner=runner, switch=switch,
                dependencies_required=dependencies_required)
    except BlockingIOError as error:
        raise RuntimeError('legacy storage-return runner is still active') from error


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _read(path: Path) -> bytes:
    _canonical(path)
    before = _stamp(path)
    with path.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        if before != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ValueError('evidence replaced while opening')
        data = handle.read()
        after = os.fstat(handle.fileno())
    if before != _stamp(path) or before != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('evidence changed during read')
    return data


def _sync(path: Path) -> None:
    with path.open('rb') as handle:
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_private_mode(path: Path, label: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f'{label} mode differs from private publication mode')


def _publish(path: Path, data: bytes) -> dict:
    """Publish one private, no-clobber evidence file with crash-safe retry."""
    _canonical(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require_private_mode(path, 'immutable relocation destination')
        if _read(path) != data:
            raise ValueError('immutable relocation destination already differs')
        _sync(path)
        return file_revision(path)
    temporary = path.with_name(f'{path.name}.partial.{uuid.uuid4().hex}')
    _canonical(temporary)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _require_private_mode(temporary, 'immutable relocation partial')
    temporary_identity = _stamp(temporary)
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require_private_mode(path, 'immutable relocation destination')
        if _read(path) != data:
            raise ValueError('immutable relocation destination already differs')
        _sync(path)
        if temporary.exists() and _stamp(temporary) == temporary_identity:
            temporary.unlink()
            _sync_directory(path.parent)
        return file_revision(path)
    # Persist the published name before removing the recoverable temporary name.
    if _stamp(path)[:2] != temporary_identity[:2]:
        raise ValueError('relocation destination changed during publication')
    _sync(path)
    if _stamp(temporary) != temporary_identity:
        raise ValueError('relocation partial changed before cleanup')
    temporary.unlink()
    _sync_directory(path.parent)
    return file_revision(path)


class Relocation:
    """Strict old-to-current translation, never a basename search or fallback."""

    def __init__(self, completed: Path, study: str, *,
                 legacy_completed: Path | None = None) -> None:
        _study(study)
        self.completed = _canonical(completed)
        self.activation_revision: dict | None = None
        self.fence_revision: dict | None = None
        path = manifest_path(completed, study)
        self.revision = file_revision(path)
        self.record = json.loads(_read(path))
        record = self.record
        recorded_legacy = record.get('legacy_completed')
        if not isinstance(recorded_legacy, str):
            raise ValueError('relocation legacy-root identity is missing')
        self.legacy = _canonical(Path(recorded_legacy))
        if legacy_completed is not None and _canonical(legacy_completed) != self.legacy:
            raise ValueError('relocation legacy-root identity differs')
        if (record.get('schema') != SCHEMA or record.get('study') != study
                or record.get('legacy_completed') != str(self.legacy)):
            raise ValueError('invalid relocation identity')
        entries = record.get('entries')
        count = record.get('completion_count')
        if (type(count) is not int or not 1 <= count <= MAX_GENERATIONS + 1
                or not isinstance(entries, list) or len(entries) != count + 1):
            raise ValueError('invalid relocation closed member set')
        identities = record.get('source_identities')
        if (not isinstance(identities, list) or len(identities) != len(entries)
                or any(not isinstance(item, list) or len(item) != 4
                       or any(type(value) is not int for value in item) for item in identities)
                or not isinstance(record.get('summary_identity'), list)
                or len(record['summary_identity']) != 4):
            raise ValueError('invalid relocation source identities')
        self.summary = _revision(record['summary_revision'])
        target = Path(self.summary['path']).parent
        if Path(self.summary['path']).name != 'processing_summary.json':
            raise ValueError('invalid relocation summary path')
        self.translations: dict[str, tuple[dict, dict]] = {}
        for index, entry in enumerate(entries):
            old, new = _revision(entry['old']), _revision(entry['new'])
            if index < count:
                expected_old = self.legacy / _name(study, index)
                expected_new = self.completed / _name(study, index)
                if old['path'] != str(expected_old):
                    raise ValueError('uncontrolled historical completion path')
            else:
                old_scope = Path(old['path'])
                expected_new = (old_scope if old_scope.is_relative_to(target)
                                else self.completed / 'scopes' / f'{study}.json')
            if new['path'] != str(expected_new) or new['size'] != old['size']:
                raise ValueError('uncontrolled relocated evidence path or size')
            if old['path'] in self.translations:
                raise ValueError('duplicate relocation member')
            if file_revision(expected_new) != new:
                raise ValueError('relocated evidence version changed')
            self.translations[old['path']] = old, new
        self.scope = entries[-1]['new']
        # The mapped scope is exactly the state the old chain last received.
        old_head = json.loads(_read(completed / _name(study, count - 1)))
        state = old_head if count == 1 else old_head['successor']
        if (state['frozen_scope_revision'] != entries[-1]['old']
                or state['summary_revision'] != self.summary):
            raise ValueError('relocation head state differs from old binding')
        frozen = json.loads(_read(Path(self.scope['path'])))
        original = json.loads(_read(completed / _name(study, 0)))
        image_ids = [image['image_id'] for image in frozen['images']]
        if (frozen.get('study') != study or frozen.get('target') != str(target)
                or image_ids != original['image_ids']
                or image_ids != record.get('image_ids')
                or len(image_ids) != len(set(image_ids))):
            raise ValueError('relocation frozen image scope differs')
        self.check()

    def translate(self, revision: dict) -> dict:
        revision = _revision(revision)
        entry = self.translations.get(revision['path'])
        if entry is None:
            return revision
        old, new = entry
        if revision != old:
            raise ValueError('historical evidence version differs from relocation')
        return new

    def check(self) -> None:
        """Reject concurrent changes to the manifest or any mapped evidence."""
        if file_revision(manifest_path(self.completed, self.record['study'])) != self.revision:
            raise ValueError('relocation manifest changed during read')
        if any(file_revision(Path(new['path'])) != new
               for _, new in self.translations.values()):
            raise ValueError('relocated evidence version changed')
        if (self.activation_revision is not None
                and file_revision(self.completed / 'relocation_activation.json')
                != self.activation_revision):
            raise ValueError('relocation activation changed during read')
        if (self.fence_revision is not None
                and file_revision(self.completed / 'legacy_root_retirement.json') != self.fence_revision):
            raise ValueError('retirement plan changed during read')


def relocate_study(runner: ModuleType, study: str, frozen_path: Path, target: Path,
                   destination: Path, *, prepublish_check: Callable[[], None] | None = None) -> dict:
    """Stage exact evidence; caller switches COMPLETED only after all studies pass.

    Existing writer lock order is retained. The additional destination lock is
    acquired last. No source unlink, mtime setting, array read or scientific QA.
    """
    _study(study)
    old = _canonical(runner.COMPLETED)
    destination = _canonical(destination)
    frozen_path, target = _canonical(frozen_path), _canonical(target)
    if old == destination:
        raise ValueError('relocation requires distinct evidence roots')
    path = manifest_path(destination, study)
    with formal_write_lease(old, io_lock_root=runner.LOCK_ROOT):
        with resource_lock(target, shared=True):
            with resource_lock(old):
                with resource_lock(destination):
                    if path.exists():
                        relocation = Relocation(destination, study, legacy_completed=old)
                        _validate_sources(runner, relocation)
                        if relocation.summary != file_revision(target / 'processing_summary.json'):
                            raise ValueError('published relocation no longer matches current summary')
                        # A published retry also rejects a changed surviving old chain/scope.
                        for entry in relocation.record['entries']:
                            source = Path(entry['old']['path'])
                            if source.exists() and (file_revision(source) != entry['old']
                                                   or _read(source) != _read(Path(entry['new']['path']))):
                                raise ValueError('old evidence changed after relocation publication')
                        _sync(path)
                        return relocation.record
                    if not runner.received_study(study, frozen_path, target):
                        raise ValueError('study is not received')
                    chain = runner._completion_chain(study)
                    frozen = json.loads(_read(frozen_path))
                    ids = [image['image_id'] for image in frozen['images']]
                    if (frozen.get('study') != study or frozen.get('target') != str(target)
                            or ids != chain[0][1]['image_ids'] or len(ids) != len(set(ids))):
                        raise ValueError('current frozen identity differs from reception')
                    sources = [item[0] for item in chain] + [frozen_path]
                    snapshots = [(source, file_revision(source), _stamp(source), _read(source))
                                 for source in sources]
                    summary = file_revision(target / 'processing_summary.json')
                    summary_stamp = _stamp(target / 'processing_summary.json')
                    entries = []
                    for index, (_, revision, _, data) in enumerate(snapshots):
                        new = (destination / _name(study, index) if index < len(chain)
                               else (frozen_path if frozen_path.is_relative_to(target)
                                     else destination / 'scopes' / f'{study}.json'))
                        actual = file_revision(new) if new == frozen_path else _publish(new, data)
                        entries.append({'old': revision, 'new': actual})
                    if prepublish_check is not None:
                        prepublish_check()
                    if (runner._completion_chain(study) != chain
                            or not runner.received_study(study, frozen_path, target)
                            or file_revision(target / 'processing_summary.json') != summary
                            or _stamp(target / 'processing_summary.json') != summary_stamp
                            or any(file_revision(source) != revision or _stamp(source) != stamp
                                   or _read(source) != data
                                   for source, revision, stamp, data in snapshots)):
                        raise ValueError('reception inputs changed before relocation publication')
                    record = {'schema': SCHEMA, 'study': study, 'legacy_completed': str(old),
                              'completion_count': len(chain), 'entries': entries,
                              'summary_revision': summary, 'image_ids': ids,
                              'source_identities': [list(item[2]) for item in snapshots],
                              'summary_identity': list(summary_stamp)}
                    data = (json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode()
                    _publish(path, data)
                    Relocation(destination, study, legacy_completed=old)
                    return record


def relocate_native_scope(runner: ModuleType, study: str, frozen_path: Path,
                           target: Path, scope_destination: Path) -> Path:
    """Keep an already-final processed completion root; extend its normal chain.

    This covers Allam/Sorin roots already moved by the results owner. Their
    current physical chain must validate first; a broken pre-existing chain is
    never repaired by silently translating unknown historical paths here.
    """
    _study(study)
    completed = _canonical(runner.COMPLETED)
    frozen_path, target = _canonical(frozen_path), _canonical(target)
    scope_destination = _canonical(scope_destination)
    if not completed.is_relative_to(target.parent):
        raise ValueError('native completion root must already be inside processed')
    if not scope_destination.is_relative_to(target) or frozen_path == scope_destination:
        raise ValueError('native scope requires a new per-study processed destination')
    if completed == runner.RECEPTION_ROOT:
        raise ValueError('primary relocated root uses its registered scope mapping')
    if scope_destination.exists():
        current_chain = runner._completion_chain(study)
        current_record = current_chain[-1][1]
        current_state = current_record if len(current_chain) == 1 else current_record['successor']
        if current_state.get('frozen_scope_revision') == file_revision(scope_destination):
            if runner.received_study(study, scope_destination, target):
                return scope_destination
    if not runner.received_study(study, frozen_path, target):
        raise ValueError('native processed completion is not currently received')
    chain = runner._completion_chain(study)
    generation = len(chain)
    path = runner.completion_binding_path(study, generation)
    frozen_data = _read(frozen_path)
    frozen = json.loads(frozen_data)
    identity, old_scope = _stamp(frozen_path), file_revision(frozen_path)
    summary = file_revision(target / 'processing_summary.json')
    with formal_write_lease(completed, io_lock_root=runner.LOCK_ROOT):
        with resource_lock(target, shared=True), resource_lock(completed):
            assert_completion_writable(completed, runner.RECEPTION_ROOT)
            if (runner._completion_chain(study) != chain
                    or not runner.received_study(study, frozen_path, target)
                    or file_revision(frozen_path) != old_scope or _stamp(frozen_path) != identity):
                raise ValueError('native completion head changed before scope staging')
            new_scope = _publish(scope_destination, frozen_data)
    def check() -> None:
        if (file_revision(frozen_path) != old_scope or _stamp(frozen_path) != identity
                or _read(frozen_path) != frozen_data or _read(scope_destination) != frozen_data
                or file_revision(scope_destination) != new_scope
                or file_revision(target / 'processing_summary.json') != summary):
            raise ValueError('native scope identity changed before successor publication')
    migration = {'status': 'metadata_only', 'kind': 'scope_evidence_relocation',
                 'old_scope': old_scope, 'new_scope': new_scope,
                 'byte_equivalent': True, 'source_identity': list(identity)}
    runner.complete_study(study, scope_destination, frozen, {'scope_relocation': migration},
        completion_path=path, migration=migration, expected_previous_revision=chain[-1][2],
        prepublish_check=check)
    if not runner.received_study(study, scope_destination, target):
        raise ValueError('native scope successor did not read back')
    return scope_destination
