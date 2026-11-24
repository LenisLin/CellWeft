"""Verify final ROI annotations and report the actual processed-file changes.

Inputs: experiments v2 snapshot, current v4, and final user decisions.
Outputs: annotation_verification.json and processed_file_changes.md/json.
This audit reads existing assets without modifying scientific files or HTML.
"""

import json
from collections import Counter
from pathlib import Path


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed')
CATALOG = ROOT / '_catalog/public'
OUTPUT = CATALOG / 'evidence/segmentation_followup_20261119/user_decisions_20261119'
before = json.loads((OUTPUT / 'experiments_before.json').read_text())
current = json.loads((CATALOG / 'experiments.json').read_text())
registry = json.loads((OUTPUT / 'decisions.json').read_text())
ALLAM = 'Allam2024_IMC_NormalSLO'
BANDY = 'Bandyopadhyay2024_CODEX_BoneMarrow'
IGLESIA = 'Iglesia2024_CODEX_BreastCancer'
LY = 'Ly2025_COMET_AML'
LYMAN = 'Lyman2025_IMC_PanIN'

# Encode the user's final annotations independently of the applied registry.
discard = {
    ALLAM: ['INT_Donor_E'],
    BANDY: ['AML2_183_Scan1', 'H33_Scan1', 'H39_Scan1', 'H41_Scan1',
            'NSM2_1086_Scan1', 'NSM1_1720_Scan2', 'AML1_382_Scan2'],
    IGLESIA: ['HT171B1-S1H9A1-A4_right__20230526', 'HT297B1-S1H1_right__20230701',
              'HT565B1-S1H2_right__20230701'],
    LYMAN: ['lyman_12_a001'],
}
accept = {
    BANDY: ['AML3_1443_Scan1', 'H14_Scan1', 'H26_Scan1', 'H37_Scan1'],
    IGLESIA: ['HT397B1-S1H3A1-U2__20230315', 'HT243B1-S1H4A4__20230420',
              'HT243B1-S1H4A4_left__20230512', 'HT243B1-S1H4A4_right__20230512',
              'HT271B1-S1H6A5_left__20230512', 'HT308B1-S1H5A4_left__20230512',
              'HT308B1-S1H5A4_right__20230512', 'HT565B1-H2__20230427',
              'HT339B2-H2_right__20230701', 'HT323B1-H1__20220804'],
    LYMAN: ['lyman_13_a003', 'lyman_01_a006', 'lyman_03_a004', 'lyman_06_a001'],
}
excluded = {
    'HT171B1-S1H9A1-4_left__20230512': ['CD68'],
    'HT271B1-S1H6A5_right__20230512': ['CD68'],
    'HT171B1-S1H9A1-A4_left__20230526': ['CD68'],
    'HT171B1-S1H9A1-4_right__20230512': ['CD68', 'CD3e'],
}


def members(manifest):
    """Resolve the shared fixed split of every active image."""
    result = {}
    for study, dataset in manifest['datasets'].items():
        for group, data in dataset.get('groups', {}).items():
            for split, rois in data['members'].items():
                for roi in rois:
                    assert (study, roi) not in result
                    result[study, roi] = [group, split]
        for split, rois in dataset.get('ungrouped', {}).items():
            for roi in rois:
                assert (study, roi) not in result
                result[study, roi] = [None, split]
    return result


def differences(old, new, path=''):
    """Report changed manifest fields, retaining exact member-list changes."""
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            result.extend(differences(old.get(key), new.get(key), path + '/' + key))
        return result
    return [{'field': path, 'before': old, 'after': new}]


old_members, new_members = members(before), members(current)
ly_rois = sorted(roi for study, roi in old_members if study == LY)
assert len(ly_rois) == 16 and 'TMA8_2D' in ly_rois
accept[LY] = ly_rois
expected = {(study, roi): ('discard', []) for study, rois in discard.items() for roi in rois}
expected.update({(study, roi): ('accept', []) for study, rois in accept.items() for roi in rois})
expected.update({(IGLESIA, roi): ('exclude_channels', markers) for roi, markers in excluded.items()})
actual = {(r['study'], r['roi']): (r['action'], r['channels']) for r in registry['decisions']}
assert len(registry['decisions']) == len(actual) == len(expected) == 50
assert actual == expected
blocked = {key for key, (action, _) in expected.items() if action == 'discard'}
assert old_members.keys() - new_members.keys() == blocked
assert not new_members.keys() - old_members.keys()
assert all(split == old_members[key] for key, split in new_members.items())
assert all(key in new_members for key in expected.keys() - blocked)
assert current['split_version'] == before['split_version']
assert current['datasets'][IGLESIA]['excluded_markers_by_image'] == excluded
assert all('excluded_markers_by_image' not in dataset for study, dataset
           in current['datasets'].items() if study != IGLESIA)
assert Counter(r['action'] for r in current['quality_control']['inactive_rois']) == {'discard': 12}
assert {(r['study'], r['roi']) for r in current['quality_control']['inactive_rois']} == blocked

# Check both explicit task membership and split-reference expansion boundaries.
task_removals = []
for task_name, task in current['tasks'].items():
    for study, dataset in task.get('datasets', {}).items():
        previous = before['tasks'][task_name]['datasets'][study]
        for old, new in zip(previous.get('members', []), dataset.get('members', []), strict=True):
            if 'images' in old:
                assert new['images'] == [roi for roi in old['images'] if (study, roi) not in blocked]
                removed = [roi for roi in old['images'] if roi not in new['images']]
                if removed:
                    task_removals.append({'task': task_name, 'study': study, 'removed': removed})
                assert all((study, roi) in new_members for roi in new['images'])
            else:
                assert new == old
        assert all((study, roi) not in blocked for roi in dataset.get('valid_targets', {}))

small = [('Lyman2025_IMC_PanIN', 'lyman_06_a%03d' % n) for n in range(8, 12)]
small += [('Lyman2025_IMC_PanIN', 'lyman_10_a%03d' % n) for n in range(1, 5)]
assert all(key in new_members for key in small)
summary = json.loads((OUTPUT / 'current_review_summary.json').read_text())
assert {(r['study'], r['roi']): (r['user_decision']['action'], r['user_decision']['channels'])
        for r in summary['rois'] if 'user_decision' in r} == expected
priority = json.loads((OUTPUT / 'current_priority_rois.json').read_text())
assert all((r['study'], r['roi']) not in expected for r in priority['rois'])

verification = {
    'experiment_version': current['version'], 'split_version': current['split_version'],
    'user_decisions_checked': 50, 'action_counts': dict(Counter(r['action'] for r in registry['decisions'])),
    'active_members_before': len(old_members), 'active_members_after': len(new_members),
    'removed_rois': [{'study': study, 'roi': roi} for study, roi in sorted(blocked)],
    'retained_fixed_assignments_unchanged': True, 'explicit_task_removals': task_removals,
    'excluded_channels': excluded, 'ly_all_16_retained': True, 'small_fovs_all_8_retained': True,
    'current_summary_and_priority_match': True,
    'channel_exclusion_scope': 'Experiment consumption rules; original TIFF pages and Parquet columns retained.',
    'training_runtime_exclusion_tested': False, 'html_modified_this_audit': False,
}
(OUTPUT / 'annotation_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n')

changes = differences(before, current)
inventory = {
    'scope': 'ROI annotation handling from experiments v2 through final v4, plus this verification report.',
    'base_directory': str(ROOT),
    'modified_formal_files': [{'path': str(CATALOG / 'experiments.json'), 'changes': changes}],
    'updated_current_evidence': [str(OUTPUT / name) for name in ['decisions.json',
        'current_review_summary.json', 'current_priority_rois.json', 'display_assets.json', 'summary.md']],
    'new_snapshots_and_receipts': [str(OUTPUT / name) for name in ['experiments_before.json',
        'experiments_before_int_e_discard.json', 'application_receipt.json',
        'int_e_discard_confirmation.json', 'annotation_verification.json']],
    'new_candidate_files': [str(path) for path in sorted((OUTPUT / 'INT_Donor_E_candidate').iterdir())
                            if path.is_file()],
    'new_figure_files': [str(path) for path in sorted((OUTPUT / 'figures').rglob('*')) if path.is_file()],
    'reports_created_this_audit': [str(OUTPUT / name) for name in [
        'annotation_verification.json', 'processed_file_changes.json', 'processed_file_changes.md',
        'final_annotations_review_prompt.md', 'final_annotations_review_events.jsonl',
        'final_annotations_review_stderr.log', 'final_annotations_independent_review.md',
        'final_annotations_review_resolution.json']],
    'all_annotation_evidence_files': [str(path) for path in sorted(OUTPUT.rglob('*')) if path.is_file()],
    'historical_html_updates_before_stop': [str(OUTPUT.parent / name) for name in [
        'index.html', 'readable.html', 'independent_codex_review/report.html']],
    'historical_html_verification': str(OUTPUT / 'html_verification.json'),
    'publication_runtime': {
        'lock_file': str(CATALOG / '.publication.lock'),
        'lock_file_exists': (CATALOG / '.publication.lock').is_file(),
        'lock_note': 'Existing publication lock retained; creation ownership is not inferred.',
        'temporary_file': str(CATALOG / 'experiments.partial.json'),
        'temporary_file_exists': (CATALOG / 'experiments.partial.json').exists(),
        'temporary_note': 'Used for atomic publication and replaced by experiments.json; not retained.'},
    'unchanged_scientific_files': ['<study>/images.parquet', '<study>/panel_mapping.csv',
        '<study>/images/<ROI>/input.tiff', '<study>/images/<ROI>/image.tiff',
        '<study>/images/<ROI>/mask.tiff', '<study>/images/<ROI>/channels.parquet',
        '<study>/images/<ROI>/cells.parquet', '<study>/images/<ROI>/expression.parquet',
        '<study>/images/<ROI>/metadata.json'],
    'html': 'Not modified by this verification or following the user stop instruction; existing HTML is historical.',
}
(OUTPUT / 'processed_file_changes.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n')
report = [
    '# ROI 批注处置与 processed 文件变更', '',
    '范围：本轮逐 ROI 批注从实验入口 v2 到最终 v4 的实际改动，以及本次核对新增的报告。HTML 已停止处理。', '',
    '最终 50 个唯一 ROI：12 弃用、34 接受（含 Ly 全部 16 个）、4 个 ROI 排除 5 条通道，0 个待重分割。', '',
    '## 正式文件', '',
    f'`{CATALOG / "experiments.json"}`', '',
    '- 版本 20261119_v2 → 20261119_v4；固定划分版本保持 20261118_v1。',
    '- 共同实验成员删除 12 个弃用 ROI：22,507 → 22,495；其余成员的 group／split 全部保持。',
    '- Bandyopadhyay 的 cross_dataset_marker_prediction、unseen_marker_prediction、spatial_recovery 三项任务各移除 6 个显式图像成员；其他按 split 引用的任务从共同成员自动继承排除。',
    '- Iglesia 的 excluded_markers_by_image 只登记指定 4 个 ROI 的 4 条 CD68 与 1 条 CD3e；实验入口要求消费者在输入、表达测量、目标、隐藏信号、损失和评分中扣除，尚未验证训练运行时屏蔽。',
    '- quality_control 记录 12 个弃用项、原划分和决定入口；INT_Donor_E 不再是重分割待办。', '',
    '## 当前记录文件', '',
    '以下路径均相对本报告目录：', '',
    '| 文件 | 实际调整 |', '| --- | --- |',
    '| decisions.json | 50 个 ROI 的用户最终处置；INT_Donor_E 改为弃用，候选不接纳。 |',
    '| current_review_summary.json | 在原图像观察外附逐 ROI 用户决定；最终计数 12／34／4。experiment_version 为当前 v4，manifest_version 记录原复核所用 v2。 |',
    '| current_priority_rois.json | 用户接受／弃用／已登记通道排除的 ROI 退出待核实列表；保留本次未点名的 9 个历史疑点。 |',
    '| display_assets.json | 关联此前导出的通道图、小视野图与候选对照；INT_Donor_E 候选标记为用户拒绝、ROI 弃用。 |',
    '| summary.md | 中文逐 ROI 决定汇总，最终更新为 12 个弃用、0 个待重分割。 |', '',
    '## 新增且保留的证据', '',
    '- experiments_before.json：v2 原清单快照；experiments_before_int_e_discard.json：INT_Donor_E 改弃用前的 v3 快照。',
    '- application_receipt.json：最初 v2→v3 处置回执，保留当时重分割决定；int_e_discard_confirmation.json：随后改弃用的最终回执。',
    '- independent_review.md、implementation_independent_review.md 及各自 prompt／events／stderr：之前实际 gpt-6.2-sol/medium CLI 的独立核对记录，不改写历史结论。',
    f'- figures/：此前新增 {len(inventory["new_figure_files"])} 个 PNG，包含 4 个去指定通道的局部图、8 个小视野各 7 个原生 PNG。',
    '- INT_Donor_E_candidate/：signal.tiff、task.json、mask.tiff、result.json、comparison_1.png、comparison_2.png、comparison_3.png、candidate_display.png；均为新增候选／展示资产，未发布到原 ROI，现随弃用决定保留。',
    '- annotation_verification.json、processed_file_changes.json、processed_file_changes.md：本次新增的最终核对与文件变更清单；JSON内的all_annotation_evidence_files逐文件列出本目录的完整证据资产。',
    '- final_annotations_review_prompt.md、final_annotations_review_events.jsonl、final_annotations_review_stderr.log、final_annotations_independent_review.md、final_annotations_review_resolution.json：本次实际gpt-6.2-sol/medium独立核对的提示词、调用事件／错误日志、最终意见与措辞修订回执。', '',
    '## 历史 HTML 与发布运行文件', '',
    '- 在用户停止处理HTML之前，evidence/segmentation_followup_20261119/index.html、readable.html及independent_codex_review/report.html曾导出v3处置展示，html_verification.json记录当时验证。停止指令之后（包括v4及本次）均未改写HTML；这些旧页面不代表最终决定。',
    '- `_catalog/public/.publication.lock`为既有发布锁，留存供互斥使用，不能据此认定其首次创建来自本轮；它不是科学资产。',
    '- `_catalog/public/experiments.partial.json`曾作为原子发布中间文件，已替换为正式experiments.json，当前不留存。', '',
    '## 原 study 科学资产', '',
    '各 study 的 images.parquet、panel_mapping.csv，以及原 ROI 目录中的 input.tiff、image.tiff、mask.tiff、channels.parquet、cells.parquet、expression.parquet、metadata.json 未被这轮批注处置改写或删除。接受的 ROI 不重算；丢弃通过当前实验成员退出落实，原文件保留。', '',
    '通道排除目前是正式实验入口的逐图使用规则：原 TIFF 页与表达列仍存在，当前 channels.parquet 的来源／像素位置不被改写。训练消费者必须应用该规则；本次未运行训练，因此不声称已验证训练运行时的通道屏蔽。4 个保留 mask 的历史分割通道依赖也仍保留。', '',
    '本次未更新 HTML；现有 HTML 可能保留 INT_Donor_E 曾为待重分割的历史文字，最终状态以 experiments.json 和 decisions.json 为准。', '',
    '## 核对结果', '',
    '50 个 ROI 与用户批注逐项一致；共同成员仅减 12 个；全部保留图的原固定划分一致；显式任务清单同步；Ly 16/16 和 8 个小视野保留；当前汇总与优先列表一致。', '',
    '最终核对经实际codex-cli gpt-6.2-sol/medium独立复核通过。报告已按其意见区分通道排除规则与训练运行验证，并补齐历史HTML及发布运行文件的说明。', '',
]
(OUTPUT / 'processed_file_changes.md').write_text('\n'.join(report))
print(json.dumps(verification, ensure_ascii=False, indent=2))
