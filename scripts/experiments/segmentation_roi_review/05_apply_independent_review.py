"""Apply the completed independent review to the existing evidence reports.

Preserve frozen conclusions and original figures; update only QC artifacts.
Run before 02_collect_followup.py and 04_publish_independent_review.py.
"""

import json
from collections import Counter
from pathlib import Path


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence')
OUTPUT = ROOT / 'segmentation_followup_20261119'
REVIEW = OUTPUT / 'independent_codex_review'
response = json.loads((REVIEW / 'response.json').read_text())
frozen = json.loads((REVIEW / 'frozen_review_summary.json').read_text())
independent = {(r['study'], r['roi']): r for r in response['roi_reviews']}
assert set(independent) == {(r['study'], r['roi']) for r in frozen['rois']}
assert len(independent) == 165 and all(r['reviewed_figures'] for r in independent.values())
assert all(r['verdict'] != 'not_reviewed' for r in independent.values())

# Conservative categories retain uncertainty about bright fused structures.
changed_judgments = {
    'lyman_01_a006': '局部问题',
    'lyman_03_a004': '局部问题',
    'lyman_06_a001': '局部问题',
    'lyman_13_a003': '局部问题',
    'HT397B1-S1H3A1-U2__20230315': '局部问题',
    'HT271B1-S1H6A5_right__20230512': '局部问题',
    'HT339B2-H2_right__20230701': '证据不足',
}
resolutions = []
for study in response['study_reviews']:
    name = study['study']
    path = OUTPUT / name / 'roi_review.json'
    rows = json.loads(path.read_text())
    for row in rows:
        opinion = independent[name, row['roi']]
        row['independent_review'] = opinion
        roi = row['roi']
        if roi in changed_judgments:
            row['judgment'] = changed_judgments[roi]
            row['findings'] = [opinion['assessment']]
            row['likely_cause'] = '独立复核与主控看图确认局部疑点；亮团内细胞身份及精细边界仍待验证。'
            row['next_action'] = opinion['required_change'] + '保留现有mask，确认后决定是否局部修复或重分割。'
        if roi == 'HT323B1-H1__20220804':
            row['findings'] = ['修正后的PanCK input对照与image一致；撤回旧图所称PanCK共同条带归因。',
                               'CD3e、CD68可见横向信号层次变化；DAPI与多数实例同位，未确认整体XY错位。']
            row['likely_cause'] = '项目input至image的PanCK归一化一致；CD3e/CD68带状变化的上游成因未定。'
            row['next_action'] = '按命名通道复核条带位置；旧PanCK对照作失效证据保存，不据此重分割。'
        if roi == 'H26_Scan1':
            row['findings'] = ['组织窗核与mask大体对应；第二窗三枚实例缺少DAPI及胞体标记支持。',
                               '独立原生配对核对确认实例47286、47299、47326来自作者mask；四通道source→input像素一致。']
            row['likely_cause'] = '高置信疑似作者mask背景实例；该窗像素配对已核实，未支持padding或全局平移解释。'
            row['next_action'] = '最高优先核实三枚完整实例及作者mask定义，再决定局部排除或修复。'
        if name.startswith('Bandy'):
            for window in row['windows']:
                if window['selection_reason'] == 'DAPI-rich':
                    window['selection_reason'] = 'DAPI候选定位窗；实际信号强弱以图像为准'
            if roi in {'H41_Scan1', 'NSM2_1086_Scan1', 'H37_Scan1'}:
                row['independent_review_followup'] = opinion['assessment']
                row['next_action'] += '另单列无核支持长形实例或未覆盖核样点，核实有效背景与padding。'
        if roi == 'lyman_12_a001':
            row['likely_cause'] = 'input及成品均有内部暗带，原生视野无padding；原始MCD像素成因尚未核实。'
            row['next_action'] = '同XY追查MCD原始像素；在此之前不把暗带判作无细胞区。'
        if row['judgment'] != '样例可用':
            row['priority_level'] = 1 if roi in {'H26_Scan1', 'lyman_13_a003'} else 2
            row['issue_category'] = ('疑似背景实例' if name.startswith('Bandy') else
                                     '证据不足' if row['judgment'] == '证据不足' else
                                     '疑似局部欠覆盖' if name.startswith('Lyman') and roi != 'lyman_12_a001' else
                                     '通道质量或局部边界问题')
        if opinion['verdict'] == 'revision_required':
            resolutions.append({'study': name, 'roi': roi, 'judgment': row['judgment'],
                                'action': row['next_action'], 'remaining': '保留局部不确定性；未执行重分割。'})
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    report = OUTPUT / name / 'report.md'
    archive = OUTPUT / name / 'report_before_independent_review.md'
    if not archive.exists():
        archive.write_text(report.read_text())
    counts = dict(Counter(r['judgment'] for r in rows))
    lines = [f'# {name} 独立复核后报告', '',
             f'共{len(rows)}个ROI；当前判断：{counts}。结论仅适用于所示窗口。', '',
             '独立CLI：gpt-6.2-sol / medium；完整独立意见见 ../independent_codex_review/report.html。', '',
             '独立study意见：' + study['assessment'], '',
             '旧报告保留为 report_before_independent_review.md，旧计数及被撤回归因不作为当前结论。', '',
             '| ROI | 当前判断 | 发现 | 下一步 |', '|---|---|---|---|']
    for row in rows:
        findings = json.dumps(row['findings'], ensure_ascii=False) if isinstance(row['findings'], list) else row['findings']
        lines.append(f"| {row['roi']} | {row['judgment']} | {findings.replace('|', '／')} | {row['next_action']} |")
    lines += ['', '图、窗口坐标、来源核对及逐项独立意见见 roi_review.json；原始分割资产保持。']
    report.write_text('\n'.join(lines) + '\n')

# Missing or empty indexes provide no denominator for an ROI missing-rate claim.
path = ROOT / 'segmentation_roi_review_20261119/roi_review_summary.json'
gallery = json.loads(path.read_text())
for row in gallery['missing_studies']:
    row['index_status'] = 'empty_index' if row['image_index_exists'] else 'missing_index'
    row['indexed_count'] = 0 if row['image_index_exists'] else None
    row['candidate_count'] = None
    row['missing_counts'] = {name: None for name in row['missing_counts']}
    row['expected_roi_count'] = None
    row['missing_fraction'] = None
    row['asset_completeness'] = 'unconfirmed'
path.write_text(json.dumps(gallery, ensure_ascii=False, indent=2))
(REVIEW / 'resolution.json').write_text(json.dumps({
    'status': 'review_completed_revisions_recorded',
    'roi_resolutions': resolutions,
    'global_changes': ['Iglesia当前图使用正确input页和XY标题；撤回HT323的PanCK共条带归因，旧图保留。',
                       '50×50解释为原生小视野，边界判断依所见信号；五项资产缺口计数改为不可计算。',
                       '所有165ROI保留独立意见；两个最高优先项为H26_Scan1与lyman_13_a003。'],
    'unresolved': ['核样亮团身份、whole-cell精细边界、暗带的上游原因及部分作者mask语义仍待补证。',
                   '未执行局部修复或重分割。'],
}, ensure_ascii=False, indent=2))
print('Independent review revisions applied:', len(resolutions))
