"""Publish an index-path successor to an existing independent reception."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_write_lease
from cellweft.data.preparation.independent_reception import (
    CONTINUATION_SCHEMA, EVIDENCE_SCHEMA, SUMMARY_SNAPSHOT_SCHEMA, MAX_CONTINUATIONS,
    _check_index_move, _check_raw_source, _continuations, _predecessor_summary,
    _identity, _read, _read_basis, _read_independent_reception, _require_revision,
    relocate_index_fields, scientific_summary,
)
from cellweft.data.preparation.storage_reception_relocation import _canonical
from cellweft.data.source_intake import _save_journal, _sync_directory


def _sync_evidence(path: Path) -> None:
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _immutable(path: Path, value: dict) -> None:
    """No-clobber small evidence with the NAS caller's normal creation mode."""
    _canonical(path)
    payload = (json.dumps(value, indent=2) + '\n').encode()
    if path.exists():
        if not path.is_file():
            raise ValueError('Independent immutable evidence is not a regular file')
        if path.read_bytes() != payload:
            raise ValueError('Independent immutable evidence conflicts')
        _sync_evidence(path)
        return
    pending = path.with_name(path.name + '.receiving_' + uuid4().hex)
    with pending.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if pending.read_bytes() != payload:
        raise ValueError('Independent evidence readback differs')
    os.link(pending, path)
    _sync_directory(path.parent)
    pending.unlink()
    _sync_directory(path.parent)


def relocate_independent_index(study_root: Path, *, data_root: Path,
                               reception_root: Path, predecessor_revision: dict,
                               index_destination: Path) -> dict:
    """Publish only after source closure and the retained index are readable.

    Hold the existing formal lease and study lock throughout summary CAS and
    immutable successor publication. A prepared journal owns one candidate;
    interruptions before or after summary replacement recover forward. Unknown
    candidates and conflicting immutable records are preserved and rejected.
    The index is stat-checked, never read or copied here.
    """
    study_root, data_root, reception_root, index_destination = map(
        _canonical, (study_root, data_root, reception_root, index_destination))
    if reception_root.parent != study_root / 'storage_reception':
        raise ValueError('Independent reception basis is outside this study')
    root = reception_root.parent
    path = root / 'accepted_index_relocation_20261110.json'
    journal_path = root / 'accepted_index_relocation_journal.json'
    summary_path = study_root / 'processing_summary.json'
    candidate = root / 'accepted_index_summary.candidate.json'
    predecessor_path = _canonical(Path(predecessor_revision['path']))
    if predecessor_path.parent != root or predecessor_path == path:
        raise ValueError('Independent predecessor path differs')
    with formal_write_lease(summary_path, data_root=data_root), resource_lock(study_root):
        _require_revision(predecessor_revision)
        if path.exists():
            record = _read(path)
            if (record['predecessor_revision'] != predecessor_revision
                    or record['index_move']['new']['path'] != str(index_destination)):
                raise ValueError('Independent published successor belongs to another plan')
            result = _read_independent_reception(study_root, data_root=data_root,
                reception_root=reception_root, continuation_path=path)
            # A previous link/replace may have survived an unsuccessful fsync.
            _sync_evidence(summary_path)
            _sync_evidence(path)
            return result
        if journal_path.exists():
            _canonical(journal_path)
            journal = _read(journal_path)
            record = journal['record']
            if (record['predecessor_revision'] != predecessor_revision
                    or record['index_move']['new']['path'] != str(index_destination)):
                raise ValueError('Independent prepared successor belongs to another plan')
        else:
            before = _read_independent_reception(study_root, data_root=data_root,
                reception_root=reception_root, continuation_path=predecessor_path)
            if len(before['continuation_chain']) >= MAX_CONTINUATIONS:
                raise ValueError('Independent continuation chain bound reached')
            declared = before['scientific_summary']['acceptance']['accepted_index']
            old = {key: declared[key] for key in ('path', 'size', 'mtime_ns')}
            old_path = _canonical(Path(old['path']))
            _require_revision(old)
            move = {'old': old, 'new': file_revision(index_destination),
                    'original_identity': _identity(old_path),
                    'received_identity': _identity(index_destination)}
            previous = _read(predecessor_path)
            _check_index_move(move, previous, data_root)
            summary_anchor = None
            if previous['schema'] == CONTINUATION_SCHEMA:
                anchor_path = root / 'source_summary_snapshot_20261110.json'
                anchor = {'schema': SUMMARY_SNAPSHOT_SCHEMA, 'study': study_root.name,
                          'predecessor_revision': predecessor_revision,
                          'summary_revision': before['summary_revision'],
                          'summary_identity': _identity(summary_path),
                          'summary': before['summary']}
                _immutable(anchor_path, anchor)
                summary_anchor = file_revision(anchor_path)
            updated = relocate_index_fields(before['summary'], move['old'], move['new'])
            if candidate.exists() or candidate.is_symlink():
                raise ValueError('Unjournaled independent summary candidate preserved')
            with candidate.open('xb') as stream:
                stream.write((json.dumps(updated, indent=2) + '\n').encode())
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(root)
            if _read(candidate) != updated:
                raise ValueError('Independent summary candidate readback differs')
            record = {'schema': EVIDENCE_SCHEMA, 'study': study_root.name,
                      'status': 'complete', 'migration': 'accepted_index_path_relocation',
                      'predecessor_revision': predecessor_revision, 'index_move': move,
                      'old_summary_revision': before['summary_revision'],
                      'summary_revision': {**file_revision(candidate), 'path': str(summary_path)},
                      'summary_snapshot': updated,
                      'scientific_summary': scientific_summary(updated),
                      **{key: before[key] for key in ('basis', 'protected_versions')}}
            if summary_anchor is not None:
                record['predecessor_summary_snapshot_revision'] = summary_anchor
            journal = {'record': record, 'old_summary_identity': _identity(summary_path),
                       'candidate_identity': _identity(candidate), 'updated_summary': updated,
                       'chain_revisions': before['continuation_chain'],
                       'raw_source': before['raw_source']}
            _save_journal(journal_path, journal)
        # Recheck all bound inputs on every retry, including after replacement.
        actual_chain = _continuations(predecessor_path, root)
        if (journal['chain_revisions'] != [revision for _, _, revision in actual_chain]
                or len(actual_chain) >= MAX_CONTINUATIONS):
            raise ValueError('Independent prepared predecessor chain differs')
        previous = _read(predecessor_path)
        _check_index_move(record['index_move'], previous, data_root)
        full_before = _predecessor_summary(previous, predecessor_revision, record, root)
        expected_full = relocate_index_fields(full_before, record['index_move']['old'],
                                              record['index_move']['new'])
        expected_science = relocate_index_fields(previous['scientific_summary'],
            record['index_move']['old'], record['index_move']['new'])
        if (record.get('schema') != EVIDENCE_SCHEMA or record.get('study') != study_root.name
                or record.get('status') != 'complete'
                or record.get('migration') != 'accepted_index_path_relocation'
                or record['old_summary_revision'] != previous['summary_revision']
                or record['scientific_summary'] != expected_science
                or scientific_summary(journal['updated_summary']) != expected_science
                or record.get('summary_snapshot') != expected_full
                or journal['updated_summary'] != expected_full
                or journal['chain_revisions'][-1] != predecessor_revision
                or record['summary_revision'] != {
                    'path': str(summary_path), 'size': journal['candidate_identity']['size'],
                    'mtime_ns': journal['candidate_identity']['mtime_ns']}):
            raise ValueError('Independent prepared successor binding differs')
        _check_raw_source(journal['raw_source'], study_root.name, data_root)
        current = _read_basis(study_root, reception_root)
        if any(current[key] != record[key] for key in ('basis', 'protected_versions')):
            raise ValueError('Independent protected reception changed before publication')
        if _identity(summary_path) == journal['old_summary_identity']:
            old_snapshot = _read_independent_reception(study_root, data_root=data_root,
                reception_root=reception_root, continuation_path=predecessor_path)
            if old_snapshot['summary'] != full_before:
                raise ValueError('Independent original full summary differs from adopted snapshot')
            _canonical(candidate)
            if (_identity(candidate) != journal['candidate_identity']
                    or _read(candidate) != journal['updated_summary']):
                raise ValueError('Independent summary candidate CAS differs')
            os.replace(candidate, summary_path)
            _sync_directory(study_root)
        elif _identity(summary_path) != journal['candidate_identity']:
            raise ValueError('Independent summary publication CAS differs')
        _sync_evidence(summary_path)
        if (_read(summary_path) != journal['updated_summary']
                or file_revision(summary_path) != record['summary_revision']):
            raise ValueError('Independent summary differs from prepared successor')
        _check_raw_source(journal['raw_source'], study_root.name, data_root)
        for revision in journal['chain_revisions']:
            _require_revision(revision)
        _check_index_move(record['index_move'], _read(predecessor_path), data_root)
        _immutable(path, record)
        return _read_independent_reception(study_root, data_root=data_root,
            reception_root=reception_root, continuation_path=path)
