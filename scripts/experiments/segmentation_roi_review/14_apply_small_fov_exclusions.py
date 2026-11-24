"""Confirm small-image dimensions and exclude the 41 user-approved 50x50 ROIs.

Inputs: script 13's 112-ROI audit, current TIFF headers/metadata, experiment entry.
Outputs: size groups, every ROI's dimensions, proposed membership and receipt.
Run without flags to stage; --publish applies the independently reviewed proposal.
Original study indices and scientific assets remain available for recovery.
"""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import tifffile
import pyarrow.parquet as pq

from cellweft.data.io.publication import resource_lock


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft')
CATALOG = ROOT / 'processed/_catalog/public'
AUDIT = ROOT / 'results/small_fov_crop_review_20261120'
OUTPUT = CATALOG / 'evidence/small_fov_exclusions_20261120'
OUTPUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--publish', action='store_true')
args = parser.parse_args()


def membership(manifest):
    """Resolve each active image's fixed group and split."""
    result = {}
    for study, dataset in manifest['datasets'].items():
        for group, value in dataset.get('groups', {}).items():
            for split, rois in value['members'].items():
                for roi in rois:
                    assert (study, roi) not in result
                    result[study, roi] = (group, split)
        for split, rois in dataset.get('ungrouped', {}).items():
            for roi in rois:
                assert (study, roi) not in result
                result[study, roi] = (None, split)
    return result


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


if not args.publish:
    rows = json.loads((AUDIT / 'roi_checks.json').read_text())
    before = json.loads((CATALOG / 'experiments.json').read_text())
    assert before['version'] == '20261119_v4'
    assignments = membership(before)
    indices = {study: {r['image_id']: r for r in pq.read_table(
        ROOT/'processed'/study/'images.parquet').to_pylist()}
        for study in {r['study'] for r in rows}}

    # Reopen current image/mask headers and metadata; distinguish padding from acquisition.
    for row in rows:
        directory = ROOT / 'processed' / row['study'] / 'images' / row['roi']
        metadata = json.loads((directory / 'metadata.json').read_text())
        source = metadata['request']['image']['source']
        indexed = indices[row['study']][row['roi']]
        assert [indexed['height_px'], indexed['width_px']] == row['canvas_shape_yx']
        assert [indexed['source_y1_px']-indexed['source_y0_px'],
                indexed['source_x1_px']-indexed['source_x0_px']] == row['source_shape_yx']
        assert [indexed[k] for k in ['pad_left_px','pad_top_px','pad_right_px','pad_bottom_px']] == row['padding_ltrb']
        for name in ['image.tiff', 'mask.tiff']:
            with tifffile.TiffFile(directory / name) as tif:
                assert all(list(page.shape) == row['canvas_shape_yx'] for page in tif.pages)
        assert source['valid_bounds'] == row['valid_bounds']
        height, width = row['canvas_shape_yx']
        left, top, right, bottom = row['padding_ltrb']
        assert [height-top-bottom, width-left-right] == row['source_shape_yx']
        row['size_action'] = ('discard' if row['source_shape_yx'] == [50, 50]
                              else 'retain_pending_review' if row['max_tissue_fraction_224'] > .5
                              else 'retain_current_window_rule_ineligible')
    removed = [r for r in rows if r['size_action'] == 'discard']
    blocked = {(r['study'], r['roi']) for r in removed}
    assert len(rows) == 112 and len(blocked) == len(removed) == 41
    assert all(r['canvas_shape_yx'] == [50, 50] for r in removed)
    assert blocked <= assignments.keys()

    proposal = copy.deepcopy(before)
    decisions = []
    for row in removed:
        study, roi = row['study'], row['roi']
        group, split = assignments[study, roi]
        dataset = proposal['datasets'][study]
        members = dataset['groups'][group]['members'] if group is not None else dataset['ungrouped']
        members[split].remove(roi)
        decisions.append({'study': study, 'roi': roi, 'action': 'discard', 'channels': [],
                          'authority': 'explicit_user_approval_20261120_native_50x50',
                          'reason': 'User approved exclusion of all 41 native 50x50 ROIs.',
                          'original_group': group, 'original_split': split})
    task_removals = []
    for name, task in proposal['tasks'].items():
        for study, dataset in task.get('datasets', {}).items():
            for entry in dataset.get('members', []):
                if 'images' in entry:
                    discarded = [roi for roi in entry['images'] if (study, roi) in blocked]
                    entry['images'] = [roi for roi in entry['images'] if (study, roi) not in blocked]
                    if discarded:
                        task_removals.append({'task': name, 'study': study, 'rois': discarded})
            if 'valid_targets' in dataset:
                dataset['valid_targets'] = {roi: markers for roi, markers in dataset['valid_targets'].items()
                                           if (study, roi) not in blocked}
    proposal['version'] = '20261120_v1'
    proposal['quality_control']['inactive_rois'].extend(decisions)
    proposal['quality_control']['small_fov_decisions'] = str((OUTPUT/'decisions.json').relative_to(CATALOG))
    remaining = membership(proposal)
    assert assignments.keys()-remaining.keys() == blocked
    assert all(value == assignments[key] for key, value in remaining.items())
    assert before['split_version'] == proposal['split_version']
    write_json('experiments_before.json', before)
    write_json('experiments_proposed.json', proposal)
    write_json('decisions.json', {'experiment_version': proposal['version'], 'decisions': decisions,
                                'semantics': 'Excluded from formal experiment membership; original assets retained.',
                                'supersedes': 'Earlier retention of these eight Lyman small FOVs is superseded by this user approval.'})
    write_json('all_112_image_sizes.json', rows)

    # Group the 102 area-limited images by exact native height/width, with every ID listed.
    groups = defaultdict(list)
    for row in rows:
        if row['max_tissue_fraction_224'] <= .5:
            groups[tuple(row['source_shape_yx'])].append(row)
    ordered = sorted(groups, key=lambda shape: (shape[0]*shape[1], shape))
    report = ['# 小图尺寸确认与50×50 ROI剔除', '',
              '尺寸均为像素，按H×W（高×宽）记录。当前TIFF头信息及metadata已重新读取核对。原始采集尺寸与补边后的保存画布分别列出。', '',
              '## 102幅实测面积不足者：按原始采集尺寸统计', '',
              '| 原始H×W | 数量 | 采集框面积 / 224² | 本次决定 |', '|---|---:|---:|---|']
    for shape in ordered:
        count = len(groups[shape])
        action = '正式入口剔除' if shape == (50, 50) else '保留；当前固定窗口规则不适用'
        report.append(f'| {shape[0]}×{shape[1]} | {count} | {shape[0]*shape[1]/50176:.2%} | {action} |')
    report += ['', '共102幅、21种尺寸。41幅50×50剔除后，其余61幅不自动删除。采集框面积是几何上限；逐图实测有效面积见下表。', '',
               '## 102幅逐image尺寸', '']
    for shape in ordered:
        report += [f'### 原始{shape[0]}×{shape[1]}：{len(groups[shape])}幅', '',
                   '| Study | Image ID | 保存画布H×W | 实测有效像素数 | 完整224窗口占比上限 |', '|---|---|---|---:|---:|']
        for r in groups[shape]:
            h, w = r['canvas_shape_yx']
            report.append(f"| {r['study']} | {r['roi']} | {h}×{w} | {r['measured_pixels']} | {r['max_tissue_fraction_224']:.2%} |")
        report.append('')
    report += ['## 另外10幅：为什么待核实', '',
               '完整224×224窗口共50,176像素，现行准入要求组织像素严格超过25,088。',
               '这10幅的实测有效像素数超过25,088，所以图像尺寸本身不排除通过；但实测有效像素不全等于组织像素。既有max-projection/Otsu候选仅估出3.64%–21.52%，且部分结构化核或蛋白信号位于候选前景之外。尚未核实真实组织范围，不能据此宣布组织太少，也不能宣布符合门槛。', '',
               '例：一幅200×200图有40,000个采集像素，占224窗口79.72%；要通过，仍需其中至少25,089个像素确认为组织。', '',
               '| Study | Image ID | 原始H×W | 实测面积上限 | 候选组织占比（未核实） |', '|---|---|---|---:|---:|']
    for r in rows:
        if r['max_tissue_fraction_224'] > .5:
            h,w = r['source_shape_yx']
            report.append(f"| {r['study']} | {r['roi']} | {h}×{w} | {r['max_tissue_fraction_224']:.2%} | {r['diagnostic_tissue_fraction_224']:.2%} |")
    report += ['', '## 剔除范围与文件变更', '',
               '按用户批准，Bever 6幅、Lyman 8幅、Parsons 27幅，共41幅原生50×50从正式公共训练/评价成员剔除。其余ROI原固定group/split和marker排除规则保持；41幅的原始图像、mask、cells及study图像索引保留，用于恢复。此前Lyman小FOV保留决定由本次批准取代。', '',
               f"正式入口版本：{before['version']} → {proposal['version']}；公共入口成员数：{len(assignments):,} → {len(remaining):,}（含pending与外部评价成员）。", '',
               '发布时仅更新正式文件processed/_catalog/public/experiments.json；新增快照、决定及回执在本目录。按split引用的任务继承排除，显式列表及valid_targets也已检查。未运行训练消费者验证，未修改HTML。', '']
    (OUTPUT/'size_report.md').write_text('\n'.join(report))
    summary = {'before_version': before['version'], 'after_version': proposal['version'],
               'active_before': len(assignments), 'active_after': len(remaining),
               'removed_count': len(blocked), 'removed_by_study': dict(Counter(r['study'] for r in removed)),
               'removed_by_split': dict(Counter(assignments[key][1] for key in blocked)),
               'size_groups': [{'height_px': h, 'width_px': w, 'count':len(groups[h,w])} for h,w in ordered],
               'area_limited_total':102, 'area_limited_retained':61, 'tissue_review_pending':10,
               'explicit_task_removals':task_removals, 'retained_assignments_unchanged':True,
               'scientific_assets_modified':False, 'html_modified':False, 'state':'staged'}
    write_json('summary.json',summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
else:
    # Publish only the reviewed staged proposal, under the catalog publication lock.
    review = OUTPUT/'independent_review.md'
    assert review.is_file() and review.stat().st_size > 0
    before = json.loads((OUTPUT/'experiments_before.json').read_text())
    proposal = json.loads((OUTPUT/'experiments_proposed.json').read_text())
    decisions = json.loads((OUTPUT/'decisions.json').read_text())['decisions']
    blocked = {(r['study'],r['roi']) for r in decisions}
    assert len(blocked) == 41
    assert membership(before).keys()-membership(proposal).keys() == blocked
    with resource_lock(CATALOG):
        current = json.loads((CATALOG/'experiments.json').read_text())
        assert current == before, 'Catalog changed since the reviewed proposal was staged.'
        temporary = CATALOG/'experiments.partial.json'
        temporary.write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(CATALOG/'experiments.json')
        assert json.loads((CATALOG/'experiments.json').read_text()) == proposal
    summary = json.loads((OUTPUT/'summary.json').read_text())
    summary['state'] = 'applied'
    summary['modified_formal_files'] = [str(CATALOG/'experiments.json')]
    summary['independent_review'] = str(review)
    write_json('summary.json',summary)
    write_json('application_receipt.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
