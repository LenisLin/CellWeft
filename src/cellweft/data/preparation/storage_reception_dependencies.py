"""Retain only the runner's actual bootstrap inputs and bound index versions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import ModuleType

from cellweft.data.preparation.storage_reception_relocation import _canonical, _publish, _read, _revision, _stamp, _sync
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.io_coordination import formal_write_lease

ROLES = {'approval': 'approval.json', 'approved_scope': 'approved_scope.json',
         'disposition': 'disposition_plan.json', 'inventory': 'inventory.json'}


def _base_revision(record: dict) -> dict:
    """Frozen index identities include inode; the historical record stays intact."""
    return _revision({key: record[key] for key in ('path', 'size', 'mtime_ns')})


class DependencyMap:
    def __init__(self, completed: Path) -> None:
        self.completed = _canonical(completed)
        self.path = completed / 'dependencies.json'
        self.revision = file_revision(self.path)
        self.record = json.loads(_read(self.path))
        if self.record.get('schema') != 'cellweft.storage_reception_dependencies.v1':
            raise ValueError('invalid reception dependency schema')
        self.roles = self.record.get('roles')
        if not isinstance(self.roles, dict) or set(self.roles) != set(ROLES):
            raise ValueError('invalid reception bootstrap role set')
        indexes = self.record.get('accepted_indexes')
        if not isinstance(indexes, list) or not indexes:
            raise ValueError('missing frozen accepted index versions')
        self.entries = [*self.roles.values(), *indexes]
        self.mapping = {}
        for role, entry in self.roles.items():
            if entry['new']['path'] != str(completed / 'execution' / ROLES[role]):
                raise ValueError('uncontrolled bootstrap destination')
        for entry in indexes:
            old = _revision(entry['old'])
            if entry['new']['path'] != str(completed / 'accepted_indexes' / Path(old['path']).name):
                raise ValueError('uncontrolled accepted index destination')
            if entry['method'] != 'same_inode_hardlink':
                raise ValueError('accepted indexes require same-inode retention')
        for entry in self.entries:
            old, new = _revision(entry['old']), _revision(entry['new'])
            if old['path'] in self.mapping or old['size'] != new['size']:
                raise ValueError('duplicate or mismatched dependency version')
            identity = entry['source_identity']
            if (not isinstance(identity, list) or len(identity) != 4
                    or identity[2:] != [old['size'], old['mtime_ns']]):
                raise ValueError('invalid dependency source identity')
            if file_revision(Path(new['path'])) != new:
                raise ValueError('retained dependency version changed')
            if (entry['method'] == 'same_inode_hardlink'
                    and list(_stamp(Path(new['path']))) != identity):
                raise ValueError('retained index is not the exact source inode')
            self.mapping[old['path']] = entry
        self.check()

    def check(self) -> None:
        if (file_revision(self.path) != self.revision
                or any(file_revision(Path(entry['new']['path'])) != entry['new']
                       for entry in self.entries)):
            raise ValueError('reception dependencies changed during read')

    def validate_sources(self) -> None:
        """Run inside the final activation's existing lease and publication locks."""
        for entry in self.entries:
            source, new = Path(entry['old']['path']), Path(entry['new']['path'])
            if (file_revision(source) != entry['old']
                    or list(_stamp(source)) != entry['source_identity']):
                raise ValueError('source dependency changed before activation')
            if entry['method'] == 'same_inode_hardlink':
                if _stamp(source) != _stamp(new):
                    raise ValueError('accepted index no longer shares the retained inode')
            elif _read(source) != _read(new):
                raise ValueError('bootstrap dependency bytes differ')
        self.check()

    def translate(self, revision: dict) -> dict:
        base = _base_revision(revision)
        entry = self.mapping.get(base['path'])
        if entry is None:
            return base
        if base != entry['old']:
            raise ValueError('wrong historical dependency version')
        if 'inode' in revision and revision['inode'] != entry['source_identity'][1]:
            raise ValueError('wrong historical dependency inode')
        return entry['new']

    def read_json(self, role: str) -> dict:
        entry = self.roles[role]
        value = json.loads(_read(Path(entry['new']['path'])))
        self.check()
        return value


def retain_runner_dependencies(runner: ModuleType, completed: Path,
                               frozen_paths: list[Path]) -> dict:
    """Stage small startup JSON and hardlink exactly the indexes frozen scopes bind.

    Call after study staging, before final activation; source versions are CAS
    checked again by activation. Index bodies are never read or parsed here.
    """
    _canonical(completed)
    with formal_write_lease(runner.COMPLETED, io_lock_root=runner.LOCK_ROOT):
        with resource_lock(runner.COMPLETED), resource_lock(completed):
            return _retain_runner_dependencies(runner, completed, frozen_paths)


def _retain_runner_dependencies(runner: ModuleType, completed: Path,
                                frozen_paths: list[Path]) -> dict:
    approval_path = runner.ROOT / 'approval.json'
    approval = json.loads(_read(approval_path))
    sources = {'approval': approval_path,
               'approved_scope': runner.PROJECT / approval['scope_file'],
               'disposition': runner.PROJECT / 'cache/data_return_cleanup_planning_20261106/disposition_plan.json',
               'inventory': runner.PROJECT / 'cache/data_return_cleanup_planning_20261106/inventory_data_volume/inventory.json'}
    if any(not _canonical(source).is_relative_to(_canonical(runner.PROJECT))
           for source in sources.values()):
        raise ValueError('bootstrap source escapes the configured project')
    roles = {}
    for role, source in sources.items():
        _canonical(source)
        identity, revision, data = list(_stamp(source)), file_revision(source), _read(source)
        new = completed / 'execution' / ROLES[role]
        current = _publish(new, data)
        if list(_stamp(source)) != identity or file_revision(source) != revision:
            raise ValueError('bootstrap changed during retention')
        roles[role] = {'old': revision, 'new': current, 'source_identity': identity,
                       'method': 'exact_bytes_copy'}
    indexes = {}
    approved = json.loads(_read(completed / 'execution' / ROLES['approved_scope']))
    approved_studies = {row['study'] for row in approved['studies']}
    seen_studies = set()
    for frozen_path in frozen_paths:
        frozen = json.loads(_read(frozen_path))
        if frozen['study'] in seen_studies or frozen['study'] not in approved_studies:
            raise ValueError('frozen index scope differs from the approved startup batch')
        seen_studies.add(frozen['study'])
        declared = frozen['accepted_index']
        old = _base_revision(declared)
        source = Path(old['path'])
        _canonical(source)
        identity = list(_stamp(source))
        if (file_revision(source) != old
                or ('inode' in declared and declared['inode'] != identity[1])):
            raise ValueError('frozen accepted index no longer has its declared identity')
        if old['path'] in indexes:
            if indexes[old['path']]['old'] != old:
                raise ValueError('conflicting frozen accepted index versions')
            continue
        new = completed / 'accepted_indexes' / source.name
        _canonical(new)
        new.parent.mkdir(parents=True, exist_ok=True)
        if not new.exists():
            os.link(source, new)
        if list(_stamp(new)) != identity or list(_stamp(source)) != identity:
            raise ValueError('accepted index retention did not preserve exact inode')
        _sync(new)
        indexes[old['path']] = {'old': old, 'new': file_revision(new),
                                 'source_identity': identity, 'method': 'same_inode_hardlink'}
    if seen_studies != approved_studies:
        raise ValueError('frozen index scope does not cover the approved startup batch')
    record = {'schema': 'cellweft.storage_reception_dependencies.v1', 'roles': roles,
              'accepted_indexes': [indexes[key] for key in sorted(indexes)]}
    _publish(completed / 'dependencies.json', (json.dumps(record, indent=2) + '\n').encode())
    DependencyMap(completed).validate_sources()
    return record
