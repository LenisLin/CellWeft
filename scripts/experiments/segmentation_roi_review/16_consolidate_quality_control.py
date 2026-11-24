"""Keep one current QC registry and remove superseded ROI-decision process files.

Inputs: current experiment entry, final ROI decisions and small-image checks.
Outputs: processed/_catalog/public/quality_control.json and stable QC figure paths.
Preview prints the exact cleanup scope; --apply --review-file uses reviewed scope.
Scientific study assets, fixed membership and historical HTML remain unchanged.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from cellweft.data.io.publication import resource_lock


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft')
CATALOG = ROOT/'processed/_catalog/public'
OLD = CATALOG/'evidence/segmentation_followup_20261119/user_decisions_20261119'
SMALL = CATALOG/'evidence/small_fov_exclusions_20261120'
FOLLOWUP = CATALOG/'evidence/small_fov_followup_20261120'
RESULTS = ROOT/'results/small_fov_crop_review_20261120'
FIGURES = CATALOG/'quality_control/small_fov'
REGISTRY = CATALOG/'quality_control.json'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--apply',action='store_true')
parser.add_argument('--review-file',type=Path)
parser.add_argument('--plan-file',type=Path)
args = parser.parse_args()


def membership(manifest):
    """Resolve fixed assignments to prove cleanup cannot change selection."""
    result = {}
    for study,dataset in manifest['datasets'].items():
        for group,value in dataset.get('groups',{}).items():
            for split,ids in value['members'].items():
                result.update({(study,roi):(group,split) for roi in ids})
        for split,ids in dataset.get('ungrouped',{}).items():
            result.update({(study,roi):(None,split) for roi in ids})
    return result


# Merge the final decisions; preserve every existing action and inactive assignment.
original = json.loads((CATALOG/'experiments.json').read_text())
assert original['version'] == '20261120_v2'
decisions = {}
for source in [OLD/'decisions.json',SMALL/'decisions.json',FOLLOWUP/'decisions.json']:
    for row in json.loads(source.read_text())['decisions']:
        key = (row['study'],row['roi'])
        assert key not in decisions
        decisions[key] = row
assert len(decisions) == 162
assert Counter(r['action'] for r in decisions.values()) == {'discard':116,'accept':42,'exclude_channels':4}
active = membership(original)
assert len(active) == 22391
inactive = {(r['study'],r['roi']) for r in original['quality_control']['inactive_rois']}
assert inactive == {key for key,r in decisions.items() if r['action']=='discard'}
assert all(key in active for key,r in decisions.items() if r['action']!='discard')
small_rows = json.loads((RESULTS/'roi_checks.json').read_text())
for row in small_rows:
    row['final_user_action'] = decisions[row['study'],row['roi']]['action']
    row['figure'] = str((FIGURES/row['figure']).relative_to(CATALOG))
assert Counter(r['final_user_action'] for r in small_rows) == {'discard':104,'accept':8}
padding = json.loads((FOLLOWUP/'padding_read_verification.json').read_text())
assert len(padding) == 8
size_groups = Counter(tuple(r['source_shape_yx']) for r in small_rows
                      if r['max_tissue_fraction_224'] <= .5)

# Keep unique PNG evidence; remove snapshots, proposals, logs and superseded receipts.
source_images = sorted(RESULTS.rglob('*.png'))
assert len(source_images) == 119
moves = [(source,FIGURES/source.relative_to(RESULTS)) for source in source_images]
assert all(not target.exists() for _,target in moves)
remove = sorted([p for directory in [SMALL,FOLLOWUP] for p in directory.iterdir() if p.is_file()])
remove += sorted(p for p in RESULTS.iterdir() if p.is_file() and p.suffix!='.png')
old_process_names = [
    'annotation_verification.json','application_receipt.json','decisions.json',
    'experiments_before.json','experiments_before_int_e_discard.json',
    'final_annotations_independent_review.md','final_annotations_review_events.jsonl',
    'final_annotations_review_prompt.md','final_annotations_review_resolution.json',
    'final_annotations_review_stderr.log','html_verification.json',
    'implementation_independent_review.md','implementation_review_events.jsonl',
    'implementation_review_prompt.md','implementation_review_stderr.log',
    'int_e_discard_confirmation.json','plan.md','processed_file_changes.json',
    'processed_file_changes.md','proposed_decisions.json','review_events.jsonl',
    'review_prompt.md','review_stderr.log','small_fov_display_facts.json']
remove += [OLD/name for name in old_process_names]
assert all(p.is_file() and not p.is_symlink() for p in remove)
assert not any(p.suffix in {'.tiff','.parquet','.html','.png'} for p in remove)
# The historical HTML links these two texts; keep their existing targets.
assert OLD/'summary.md' not in remove and OLD/'independent_review.md' not in remove
sizes = {p:p.stat().st_size for p in remove}
directories = [RESULTS/'figures',RESULTS/'scratch',RESULTS,SMALL,FOLLOWUP]
allowed_entries = set(remove) | set(source_images) | set(directories)
directory_entries = {str(directory):[str(p) for p in sorted(directory.iterdir())]
                     for directory in directories}
unexpected = [entry for entries in directory_entries.values() for entry in entries
              if Path(entry) not in allowed_entries]
if unexpected:
    raise RuntimeError(f'Unplanned files in retiring directories: {unexpected}')
plan = {'registry':str(REGISTRY),'manifest':str(CATALOG/'experiments.json'),
        'active_members':len(active),'decision_counts':dict(Counter(r['action'] for r in decisions.values())),
        'remove_files':[str(p) for p in remove],'removed_file_count':len(remove),
        'remove_file_sizes':{str(p):sizes[p] for p in remove},
        'removed_file_bytes':sum(sizes.values()),'move_png_count':len(moves),
        'png_moves':[{'source':str(source),'target':str(target),'size_bytes':source.stat().st_size}
                     for source,target in moves],
        'figure_destination':str(FIGURES),'removed_directories':[str(p) for p in directories],
        'directory_entries':directory_entries,
        'scientific_assets_modified':False,'html_modified':False,'new_version_or_snapshot':False}
if not args.apply:
    print(json.dumps(plan,ensure_ascii=False,indent=2))
else:
    assert args.review_file is not None and args.review_file.is_file()
    assert args.plan_file is not None and args.plan_file.is_file()
    if json.loads(args.plan_file.read_text()) != plan:
        raise RuntimeError('Cleanup scope changed since review.')
    updated = copy.deepcopy(original)
    updated['quality_control']['decisions'] = 'quality_control.json'
    del updated['quality_control']['small_fov_decisions']
    del updated['quality_control']['small_fov_followup_decisions']
    registry = {
        'experiment_version':original['version'],
        'decisions':[decisions[key] for key in sorted(decisions)],
        'policies':{'discard':'Excluded from active experiments; original scientific assets retained.',
                    'padding':'Eight accepted small ROIs preserve native pixels and permit zero padding to 224x224 with padding marked invalid; no resizing.',
                    'sampling':'ROI acceptance does not approve the automatic tissue estimator; existing per-ROI sampling metadata remains unchanged.',
                    'marker_exclusions':'Consumers must apply manifest marker exclusions; training-runtime enforcement has not been verified.'},
        'small_images':{'checked':112,'area_limited_discarded':102,'additional_user_discards':2,
                        'accepted_with_padding':8,'rois':small_rows,'padding_verification':padding,
                        'area_limited_size_groups':[{'source_shape_yx':list(shape),'count':size_groups[shape]}
                            for shape in sorted(size_groups,key=lambda shape:(shape[0]*shape[1],shape))],
                        'tissue_estimation_note':'候选前景之外仍可见结构化核或蛋白信号，不能将候选前景直接等同真实组织范围。真实组织比例未核验；用户已对ROI作最终取舍，ROI接受与组织估计算法验收分别记录。',
                        'contact_sheets':[str((FIGURES/p.name).relative_to(CATALOG)) for p in sorted(RESULTS.glob('contact_*.png'))]},
        'segmentation_review':{
            'observations':str((OLD/'current_review_summary.json').relative_to(CATALOG)),
            'remaining_priority_rois':str((OLD/'current_priority_rois.json').relative_to(CATALOG))},
        'independent_review':{'model':'gpt-6.2-sol','reasoning_effort':'medium','exit_code':0,
                              'text':args.review_file.read_text(),
                              'resolution':'Directory entries, deletion sizes and PNG moves are rechecked after acquiring the catalog lock, before any write. The same lock covers migration, publication and cleanup; retired producers are no longer run. The tissue-estimation observation and current document paths are preserved.'},
        'cleanup':{k:v for k,v in plan.items() if k not in {'remove_files','remove_file_sizes','png_moves','removed_directories','directory_entries'}},
    }
    assert membership(updated) == active
    with resource_lock(CATALOG):
        assert json.loads((CATALOG/'experiments.json').read_text()) == original
        # Recheck after waiting for the lock; keep it through migration and retirement.
        locked_entries = {str(directory):[str(p) for p in sorted(directory.iterdir())]
                          for directory in directories}
        if locked_entries != plan['directory_entries']:
            raise RuntimeError('Retiring directory contents changed while waiting for publication.')
        if any(path.stat().st_size != sizes[path] for path in remove):
            raise RuntimeError('A retiring file changed while waiting for publication.')
        if any(Path(row['source']).stat().st_size != row['size_bytes'] or Path(row['target']).exists()
               for row in plan['png_moves']):
            raise RuntimeError('A figure move changed while waiting for publication.')
        for source,target in moves:
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
        temporary = REGISTRY.with_suffix('.partial.json')
        temporary.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(REGISTRY)
        temporary = CATALOG/'experiments.partial.json'
        temporary.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(CATALOG/'experiments.json')
        assert json.loads(REGISTRY.read_text()) == registry
        assert json.loads((CATALOG/'experiments.json').read_text()) == updated
        assert all((CATALOG/r['figure']).is_file() for r in small_rows)
        for path in remove:
            assert path.stat().st_size == sizes[path]
            path.unlink()
        for directory in directories:
            directory.rmdir()
        assert not any(p.exists() for p in remove)
        assert all(target.is_file() and not source.exists() for source,target in moves)
        assert (OLD/'summary.md').is_file() and (OLD/'independent_review.md').is_file()
        registry = json.loads(REGISTRY.read_text())
        registry['cleanup']['completed'] = True
        temporary = REGISTRY.with_suffix('.partial.json')
        temporary.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(REGISTRY)
    print(json.dumps({k:v for k,v in plan.items() if k not in {'remove_files','remove_file_sizes','png_moves','removed_directories','directory_entries'}},ensure_ascii=False,indent=2))
