"""Apply the user's remaining small-ROI exclusions and eight padding acceptances.

Inputs: script 13 audit, current experiment manifest and explicit user decisions.
Outputs: 63 exclusion decisions, eight acceptances, staged manifest and receipt.
Default stages; --publish atomically applies the independently reviewed proposal.
Native scientific assets and sampling metadata remain unchanged.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import tifffile

from cellweft.data.io.publication import resource_lock


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft')
CATALOG = ROOT/'processed/_catalog/public'
OUTPUT = CATALOG/'evidence/small_fov_followup_20261120'
OUTPUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--publish', action='store_true')
args = parser.parse_args()
EXPLICIT_DISCARD = {('Ali2020_IMC_BreastCancer', 'MB0471_1_104'),
                    ('Cords2024_IMC_Lung', 'cords2024cancer_hasivhty_0000')}


def membership(manifest):
    """Resolve each active ROI to its fixed biological group and split."""
    result = {}
    for study, dataset in manifest['datasets'].items():
        for group, value in dataset.get('groups', {}).items():
            for split, ids in value['members'].items():
                for roi in ids:
                    assert (study, roi) not in result
                    result[study, roi] = (group, split)
        for split, ids in dataset.get('ungrouped', {}).items():
            for roi in ids:
                assert (study, roi) not in result
                result[study, roi] = (None, split)
    return result


def write_json(name, value):
    (OUTPUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


if not args.publish:
    audit = json.loads((ROOT/'results/small_fov_crop_review_20261120/roi_checks.json').read_text())
    before = json.loads((CATALOG/'experiments.json').read_text())
    assert before['version'] == '20261120_v1'
    assignments = membership(before)
    area_limited = {(r['study'],r['roi']) for r in audit if r['max_tissue_fraction_224'] <= .5}
    remaining_small = area_limited & assignments.keys()
    assert len(area_limited) == 102 and len(remaining_small) == 61
    blocked = remaining_small | EXPLICIT_DISCARD
    accepted = {(r['study'],r['roi']) for r in audit if r['max_tissue_fraction_224'] > .5} - EXPLICIT_DISCARD
    assert len(blocked) == 63 and len(accepted) == 8
    assert blocked | accepted <= assignments.keys()
    selected = [r for r in audit if (r['study'],r['roi']) in blocked | accepted]
    indices = {study: {r['image_id']:r for r in pq.read_table(
        ROOT/'processed'/study/'images.parquet').to_pylist()}
        for study in {r['study'] for r in selected}}
    decisions = []
    proposal = copy.deepcopy(before)

    # Bind the selected IDs to current dimensions before modifying membership.
    for row in selected:
        study, roi = row['study'], row['roi']
        directory = ROOT/'processed'/study/'images'/roi
        indexed = indices[study][roi]
        assert [indexed['height_px'], indexed['width_px']] == row['canvas_shape_yx']
        assert [indexed['source_y1_px']-indexed['source_y0_px'],
                indexed['source_x1_px']-indexed['source_x0_px']] == row['source_shape_yx']
        metadata = json.loads((directory/'metadata.json').read_text())
        assert metadata['request']['image']['source']['valid_bounds'] == row['valid_bounds']
        for name in ['image.tiff','mask.tiff']:
            with tifffile.TiffFile(directory/name) as tif:
                assert all(list(page.shape) == row['canvas_shape_yx'] for page in tif.pages)
        group, split = assignments[study,roi]
        discard = (study,roi) in blocked
        decision = {'study':study, 'roi':roi, 'action':'discard' if discard else 'accept',
                    'channels':[], 'authority':'explicit_user_followup_20261120',
                    'original_group':group, 'original_split':split,
                    'source_shape_yx':row['source_shape_yx'], 'canvas_shape_yx':row['canvas_shape_yx']}
        if discard:
            decision['reason'] = ('User approved exclusion of all 102 area-limited ROIs.'
                                  if (study,roi) in remaining_small else 'User explicitly discarded this ROI after viewing.')
            dataset = proposal['datasets'][study]
            members = dataset['groups'][group]['members'] if group is not None else dataset['ungrouped']
            members[split].remove(roi)
        else:
            decision.update(reason='User visually accepted this ROI and permitted padding to 224x224.',
                            roi_review_status='accepted', padding_to_yx=[224,224],
                            preserve_native_pixels=True, rescale=False,
                            sampling_metadata_changed=False,
                            automated_tissue_estimator_approved=False)
        decisions.append(decision)

    task_removals = []
    for name,task in proposal['tasks'].items():
        for study,dataset in task.get('datasets',{}).items():
            for entry in dataset.get('members',[]):
                if 'images' in entry:
                    removed = [roi for roi in entry['images'] if (study,roi) in blocked]
                    entry['images'] = [roi for roi in entry['images'] if (study,roi) not in blocked]
                    if removed:
                        task_removals.append({'task':name,'study':study,'rois':removed})
            if 'valid_targets' in dataset:
                dataset['valid_targets'] = {roi:markers for roi,markers in dataset['valid_targets'].items()
                                           if (study,roi) not in blocked}
    proposal['version'] = '20261120_v2'
    proposal['quality_control']['inactive_rois'].extend(r for r in decisions if r['action']=='discard')
    proposal['quality_control']['small_fov_followup_decisions'] = str((OUTPUT/'decisions.json').relative_to(CATALOG))
    after = membership(proposal)
    assert assignments.keys()-after.keys() == blocked
    assert all(value == assignments[key] for key,value in after.items())
    assert not area_limited & after.keys()
    assert accepted <= after.keys()
    assert len(proposal['quality_control']['inactive_rois']) == 116
    write_json('experiments_before.json',before)
    write_json('experiments_proposed.json',proposal)
    write_json('decisions.json',{'experiment_version':proposal['version'],'decisions':decisions,
        'semantics':'Discard removes formal membership; original scientific assets retained.',
        'padding_semantics':'Eight accepted ROIs retain native pixels and allow runtime padding to 224x224, without rescaling or rewriting TIFFs.',
        'sampling_semantics':'Human ROI acceptance is recorded; existing sampling metadata and automatic tissue-estimator approval are not changed.',
        'supersedes':'The prior pending-human-review status of all ten geometry-capable ROIs is replaced by two discards and eight acceptances.',
        'shorthand_resolution':{'104':'Ali2020_IMC_BreastCancer/MB0471_1_104',
                                '0000':'Cords2024_IMC_Lung/cords2024cancer_hasivhty_0000'}})
    summary = {'state':'staged','before_version':before['version'],'after_version':proposal['version'],
        'active_before':len(assignments),'active_after':len(after),'removed_this_round':63,
        'remaining_area_limited_removed':61,'additional_user_discards':2,'padding_accepted':8,
        'all_102_area_limited_inactive':True,'inactive_total':116,
        'removed_by_study':dict(Counter(r['study'] for r in decisions if r['action']=='discard')),
        'removed_by_split':dict(Counter(r['original_split'] for r in decisions if r['action']=='discard')),
        'explicit_task_removals':task_removals,'retained_assignments_unchanged':True,
        'original_scientific_assets_modified':False,'sampling_metadata_modified':False,'html_modified':False}
    write_json('summary.json',summary)
    report = ['# 小图后续用户决定','',
              '本次新增剔除63幅：剩余61幅面积不足者，以及用户明确指定的MB0471_1_104和Cords的cords2024cancer_hasivhty_0000。全部102幅面积不足者均退出正式成员。','',
              '原先10幅几何可行者改为2幅弃用、8幅接受；8幅保留原生像素，允许读取时补齐224×224，不放大、不重写TIFF，不再列为待人工复核。ROI接受与自动组织估计方法验收分别记录，既有sampling元数据本轮不改。','',
              f"正式清单发布计划：{before['version']} → {proposal['version']}，成员{len(assignments):,} → {len(after):,}。",'',
              '| Study | ROI | 用户决定 | 原始H×W |','|---|---|---|---|']
    for r in decisions:
        h,w = r['source_shape_yx']
        report.append(f"| {r['study']} | {r['roi']} | {'剔除' if r['action']=='discard' else '接受，允许padding至224'} | {h}×{w} |")
    report += ['', '发布时仅改正式processed/_catalog/public/experiments.json；本目录保留快照、决定、独立复核和发布回执。原始图像、mask、细胞、表达及study索引保持。HTML保持。','']
    (OUTPUT/'report.md').write_text('\n'.join(report))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
else:
    assert (OUTPUT/'independent_review.md').is_file()
    before = json.loads((OUTPUT/'experiments_before.json').read_text())
    proposal = json.loads((OUTPUT/'experiments_proposed.json').read_text())
    with resource_lock(CATALOG):
        assert json.loads((CATALOG/'experiments.json').read_text()) == before
        temporary = CATALOG/'experiments.partial.json'
        temporary.write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(CATALOG/'experiments.json')
        assert json.loads((CATALOG/'experiments.json').read_text()) == proposal
    summary = json.loads((OUTPUT/'summary.json').read_text())
    summary.update(state='applied',modified_formal_files=[str(CATALOG/'experiments.json')])
    write_json('summary.json',summary)
    write_json('application_receipt.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
