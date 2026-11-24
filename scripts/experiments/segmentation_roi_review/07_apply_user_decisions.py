"""Apply the user's ROI decisions to the public experiment entry.

Inputs: independently checked proposed_decisions.json and experiments.json.
Outputs: active experiment membership, ROI-specific marker exclusions, and receipts.
Original scientific assets and fixed splits of retained ROIs are preserved.
"""

import copy
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from cellweft.data.io.publication import resource_lock


CATALOG = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public')
OUTPUT = CATALOG / 'evidence/segmentation_followup_20261119/user_decisions_20261119'
proposal = json.loads((OUTPUT / 'proposed_decisions.json').read_text())
decisions = proposal['decisions']
keys = {(r['study'], r['roi']) for r in decisions}
assert len(keys) == len(decisions) == 50
assert Counter(r['action'] for r in decisions) == {
    'accept': 34, 'discard': 11, 'exclude_channels': 4, 'resegment': 1}
blocked = {(r['study'], r['roi']) for r in decisions
           if r['action'] in {'discard', 'resegment'}}


def membership(manifest):
    """Return each ROI's fixed group/split assignment for a before/after check."""
    members = {}
    for study, dataset in manifest['datasets'].items():
        for group, value in dataset.get('groups', {}).items():
            for split, rois in value['members'].items():
                for roi in rois:
                    assert (study, roi) not in members
                    members[study, roi] = (group, split)
        for split, rois in dataset.get('ungrouped', {}).items():
            for roi in rois:
                assert (study, roi) not in members
                members[study, roi] = (None, split)
    return members


with resource_lock(CATALOG):
    manifest_path = CATALOG / 'experiments.json'
    original = json.loads(manifest_path.read_text())
    assert original['version'] == '20261119_v2'
    before = membership(original)
    assert keys <= before.keys()
    manifest = copy.deepcopy(original)
    receipt = {'before_version': original['version'], 'removed_members': [],
               'explicit_task_removals': [], 'channel_exclusions': []}
    for row in decisions:
        study, roi, action = row['study'], row['roi'], row['action']
        if action in {'discard', 'resegment'}:
            group, split = before[study, roi]
            dataset = manifest['datasets'][study]
            members = dataset['groups'][group]['members'] if group else dataset['ungrouped']
            members[split].remove(roi)
            receipt['removed_members'].append({**row, 'original_group': group,
                                               'original_split': split})
        if action == 'exclude_channels':
            root = CATALOG.parent.parent / study / 'images' / roi
            channels = pd.read_parquet(root / 'channels.parquet')
            available = channels.loc[channels.tiff_page.notna(), 'marker_id'].tolist()
            assert set(row['channels']) <= set(available)
            manifest['datasets'][study].setdefault('excluded_markers_by_image', {})[roi] = row['channels']
            receipt['channel_exclusions'].append({**row,
                'mask_retained': True,
                'historical_mask_uses_excluded_markers': True})

    for task_name, task in manifest['tasks'].items():
        for study, dataset in task.get('datasets', {}).items():
            for member in dataset.get('members', []):
                if 'images' in member:
                    removed = [roi for roi in member['images'] if (study, roi) in blocked]
                    member['images'] = [roi for roi in member['images'] if (study, roi) not in blocked]
                    if removed:
                        receipt['explicit_task_removals'].append({
                            'task': task_name, 'study': study, 'rois': removed})
            if 'valid_targets' in dataset:
                dataset['valid_targets'] = {roi: markers for roi, markers in dataset['valid_targets'].items()
                                            if (study, roi) not in blocked}

    after = membership(manifest)
    assert before.keys() - after.keys() == blocked
    assert all(value == before[key] for key, value in after.items())
    assert all((study, roi) in after for study, roi in keys - blocked)
    manifest['version'] = '20261119_v3'
    manifest['selection']['input_markers'] += (
        ' Subtract datasets[study].excluded_markers_by_image[image_id] from all '
        'effective image inputs and cell-expression measurements; stored TIFF pages '
        'and original measurement columns remain unchanged.')
    manifest['selection']['targets'] += (
        ' Subtract the same per-image excluded markers from target candidates, '
        'valid_targets, hidden signals, losses, and scoring before task execution.')
    manifest['quality_control'] = {
        'decisions': str((OUTPUT / 'decisions.json').relative_to(CATALOG)),
        'inactive_rois': receipt['removed_members'],
        'resegmentation_policy': 'INT_Donor_E remains inactive until a reviewed replacement '
            'mask and matching cells/expression/neighbors/crops are published.',
        'retained_mask_history': 'The four marker-exclusion ROIs retain their current masks, '
            'which were generated using the now-excluded body markers. The exclusions '
            'control current measurement use and do not remove historical mask dependence.'}
    receipt.update(after_version=manifest['version'], active_before=len(before), active_after=len(after),
                   retained_assignments_unchanged=True, action_counts=dict(Counter(r['action'] for r in decisions)))
    registry = {**proposal, 'state': 'applied', 'experiment_version': manifest['version'],
                'discard_semantics': 'Excluded from active experiments; original assets retained.',
                'resegmentation_state': 'pending_candidate_review',
                'channel_semantics': 'Current inputs/measurements/targets excluded per ROI; existing masks retained.'}
    (OUTPUT / 'experiments_before.json').write_text(json.dumps(original, ensure_ascii=False, indent=2) + '\n')
    (OUTPUT / 'decisions.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n')
    temporary = manifest_path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assert membership(json.loads(temporary.read_text())) == after
    temporary.replace(manifest_path)
    (OUTPUT / 'application_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
