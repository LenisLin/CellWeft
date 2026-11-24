"""Parsons' independent input-closure receipt, separate from storage-return chains."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cellweft.data.io.publication import resource_lock
from cellweft.data.preparation.inputs import file_revision
from cellweft.data.preparation.io_coordination import formal_write_lease
from cellweft.data.preparation.storage_reception_relocation import _canonical, _publish, _read, _revision, _stamp

STUDY = 'Parsons2025_IMC_TNBC'
NAMES = {'completion': 'completion_receipt.json', 'input': 'formal_input_receipt.json',
         'closure': 'input_scope_closed.json'}
SOURCE_FACT_NAMES = {
    'source_reader_closure': 'native_source_reader_closure_20261110.json',
    'source_archive': 'author_source_archive_20261110.json',
    'source_facts': 'native_source_facts_20261110.json',
}
EVIDENCE_NAMES = {**NAMES, **SOURCE_FACT_NAMES}


def _base(record: dict) -> dict:
    return _revision({key: record[key] for key in ('path', 'size', 'mtime_ns')})


def _validate_records(records: dict[str, dict], entries: dict[str, dict], target: Path) -> list[str]:
    completion, formal, closure = records['completion'], records['input'], records['closure']
    if (completion.get('schema') != 'cellweft.parsons_completion_receipt.v1'
            or completion.get('study') != STUDY
            or completion.get('status') != 'technical_complete_formally_received'):
        raise ValueError('invalid independent Parsons completion')
    for role, key in [('input', 'formal_input_receipt'), ('closure', 'input_scope_closure')]:
        binding = completion[key]
        if binding.get('path') != entries[role]['old']['path'] or binding.get('revision') != entries[role]['old']:
            raise ValueError('Parsons completion input-closure binding differs')
    if closure.get('input_receipt_revision') != entries['input']['old']:
        raise ValueError('Parsons closure input receipt binding differs')
    ids = closure.get('image_ids')
    if (formal.get('study') != STUDY or closure.get('study') != STUDY
            or closure.get('status') != 'closed' or not isinstance(ids, list)
            or len(ids) != len(set(ids)) or len(ids) != completion.get('formal_images')
            or len(ids) != formal.get('input_images')):
        raise ValueError('Parsons closed image scope differs')
    image_revision = formal['images_parquet_revision']
    images_path = target / 'images.parquet'
    if (Path(image_revision['path']) != images_path or file_revision(images_path) != _base(image_revision)
            or ('inode' in image_revision and images_path.stat().st_ino != image_revision['inode'])):
        raise ValueError('Parsons formally bound image table version differs')
    # Read only the original key table, never arrays or numerical QA outputs.
    images = pd.read_parquet(images_path, columns=['study', 'image_id'])
    keys = list(zip(images.study.astype(str), images.image_id.astype(str)))
    if len(keys) != len(set(keys)) or set(keys) != {(STUDY, image_id) for image_id in ids}:
        raise ValueError('Parsons current image keys differ from closed scope')
    # Existing receipt already binds this review version; only stat it.
    review = completion['input_scope_closure']['final_review']
    if (review['path'] != str(target / 'review_results.csv')
            or review['revision'] != file_revision(target / 'review_results.csv')
            or closure['review_results_revision'] != review['revision']):
        raise ValueError('Parsons original bound review version differs')
    if completion.get('processing_summary') != str(target / 'processing_summary.json'):
        raise ValueError('Parsons summary target differs')
    return ids


def _validate_source_facts(records: dict[str, dict]) -> None:
    if any(records[role].get('study') != STUDY or records[role].get('status') != 'complete'
           for role in SOURCE_FACT_NAMES):
        raise ValueError('Parsons necessary source closure facts differ')


class ParsonsReception:
    def __init__(self, target: Path) -> None:
        self.target = _canonical(target)
        self.root = target / 'storage_reception_20261110'
        self.path = self.root / 'parsons_completion_relocation.json'
        self.revision = file_revision(self.path)
        self.record = json.loads(_read(self.path))
        record = self.record
        if (record.get('schema') != 'cellweft.parsons_reception_relocation.v2'
                or record.get('study') != STUDY or target.name != STUDY):
            raise ValueError('invalid Parsons reception relocation')
        entries = record.get('entries')
        if not isinstance(entries, dict) or set(entries) != set(EVIDENCE_NAMES):
            raise ValueError('invalid Parsons reception evidence set')
        historical_completion = _revision(entries['completion']['old'])['path']
        historical_root = Path(historical_completion).parent
        if Path(historical_completion).name != EVIDENCE_NAMES['completion']:
            raise ValueError('uncontrolled historical Parsons completion path')
        for role, entry in entries.items():
            old, new = _revision(entry['old']), _revision(entry['new'])
            if (new['path'] != str(self.root / EVIDENCE_NAMES[role]) or old['size'] != new['size']
                    or file_revision(Path(new['path'])) != new):
                raise ValueError('Parsons reception path or version differs')
            if role in SOURCE_FACT_NAMES and old['path'] != str(
                    historical_root / SOURCE_FACT_NAMES[role]):
                raise ValueError('uncontrolled historical Parsons source fact path')
        records = {role: json.loads(_read(Path(entry['new']['path']))) for role, entry in entries.items()}
        _validate_source_facts(records)
        ids = _validate_records(records, entries, target)
        self.images_revision = _base(records['input']['images_parquet_revision'])
        self.review_revision = records['closure']['review_results_revision']
        if ids != record.get('image_ids'):
            raise ValueError('Parsons reception closed image keys differ')
        # This new manifest explicitly binds the current summary; the historical
        # completion only records its path, and is never represented otherwise.
        if record.get('summary_revision') != file_revision(target / 'processing_summary.json'):
            raise ValueError('Parsons current summary version differs')
        self.check()

    def check(self) -> None:
        if (file_revision(self.path) != self.revision
                or any(file_revision(Path(entry['new']['path'])) != entry['new']
                       for entry in self.record['entries'].values())
                or file_revision(self.target / 'images.parquet') != self.images_revision
                or file_revision(self.target / 'review_results.csv') != self.review_revision
                or file_revision(self.target / 'processing_summary.json') != self.record['summary_revision']):
            raise ValueError('Parsons reception changed during read')


def received_parsons(target: Path) -> bool:
    ParsonsReception(target).check()
    return True


def relocate_parsons(target: Path, *, io_lock_root: Path,
                      legacy_completion: Path) -> dict:
    """Use only after the Parsons owner is drained; preserve all original bodies."""
    target, legacy_completion = _canonical(target), _canonical(legacy_completion)
    root = target / 'storage_reception_20261110'
    path = root / 'parsons_completion_relocation.json'
    with formal_write_lease(target, io_lock_root=io_lock_root):
        with resource_lock(target), resource_lock(legacy_completion.parent):
            if path.exists():
                return ParsonsReception(target).record
            completion = json.loads(_read(legacy_completion))
            sources = {'completion': legacy_completion,
                       'input': Path(completion['formal_input_receipt']['path']),
                       'closure': Path(completion['input_scope_closure']['path'])}
            sources.update({role: legacy_completion.parent / name
                            for role, name in SOURCE_FACT_NAMES.items()})
            snapshots = {role: (file_revision(source), list(_stamp(source)), _read(source))
                         for role, source in sources.items()}
            entries = {role: {'old': snapshot[0],
                             'new': _publish(root / EVIDENCE_NAMES[role], snapshot[2])}
                       for role, snapshot in snapshots.items()}
            records = {role: json.loads(snapshot[2]) for role, snapshot in snapshots.items()}
            _validate_source_facts(records)
            ids = _validate_records(records, entries, target)
            summary = file_revision(target / 'processing_summary.json')
            summary_identity = _stamp(target / 'processing_summary.json')
            if any(file_revision(sources[role]) != revision
                   or list(_stamp(sources[role])) != identity or _read(sources[role]) != data
                   for role, (revision, identity, data) in snapshots.items()):
                raise ValueError('Parsons source evidence changed before publication')
            if (file_revision(target / 'processing_summary.json') != summary
                    or _stamp(target / 'processing_summary.json') != summary_identity):
                raise ValueError('Parsons summary changed before publication')
            record = {'schema': 'cellweft.parsons_reception_relocation.v2', 'study': STUDY,
                      'entries': entries, 'summary_revision': summary, 'image_ids': ids,
                      'summary_binding': 'new_relocation_manifest; historical_completion_records_path_only'}
            _publish(path, (json.dumps(record, indent=2) + '\n').encode())
            return ParsonsReception(target).record
