"""Collect five study-specific ROI reviews and the user's visual decisions.

Inputs: the original gallery summary and five subagent roi_review.json reports.
Outputs: follow-up HTML/JSON, user decision register, updated original gallery labels.
Original image, mask, and other scientific assets are read-only in this workflow.
"""

import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pandas as pd

EVIDENCE = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence')
ORIGINAL = EVIDENCE / 'segmentation_roi_review_20261119'
OUTPUT = EVIDENCE / 'segmentation_followup_20261119'
STUDIES = {
    'Allam2024_IMC_NormalSLO': 6,
    'Bandyopadhyay2024_CODEX_BoneMarrow': 20,
    'Iglesia2024_CODEX_BreastCancer': 48,
    'Ly2025_COMET_AML': 16,
    'Lyman2025_IMC_PanIN': 75,
}


def display(value):
    """Escape scalar or structured review evidence for an HTML text field."""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(str(value))


def figure_path(study, value):
    """Resolve a study-owned evidence figure to its path inside this report."""
    path = Path(value)
    if not path.is_absolute():
        path = OUTPUT / study / path
    relative = path.relative_to(OUTPUT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return relative.as_posix()


# Require every indexed ROI in the five requested studies to have a result.
original = json.loads((ORIGINAL / 'roi_review_summary.json').read_text())
records = []
study_summaries = []
for study, expected_count in STUDIES.items():
    rows = json.loads((OUTPUT / study / 'roi_review.json').read_text())
    if len(rows) != expected_count or len({row['roi'] for row in rows}) != expected_count:
        raise ValueError('Incomplete ROI review: ' + study)
    indexed = pd.read_parquet(EVIDENCE.parents[2] / study / 'images.parquet', columns=['image_id'])
    if {row['roi'] for row in rows} != set(indexed['image_id']):
        raise ValueError('ROI review membership differs from current image index: ' + study)
    for row in rows:
        if row['judgment'] not in {'样例可用', '局部问题', '疑似系统性问题', '证据不足'}:
            raise ValueError('Unfinished review: ' + study + '/' + row['roi'])
        row['study'] = study
        row['figures'] = [figure_path(study, path) for path in row['figures']]
    records.extend(rows)
    study_summaries.append({
        'study': study, 'roi_count': len(rows),
        'judgments': dict(Counter(row['judgment'] for row in rows)),
        'report': study + '/report.md',
    })

# Preserve the distinction between user decisions on the sampled gallery and this audit.
missing = {row['study'] for row in original['missing_studies']}
samples_by_study = {}
for row in original['rois']:
    samples_by_study.setdefault(row['study'], []).append(row)
decisions = []
for study in sorted(set(samples_by_study) | missing):
    if study in STUDIES:
        status = '专项复核完成，见逐ROI结论'
        authority = '用户点名；subagent逐ROI复核'
    elif study in missing:
        status = '无完整当前分割资产'
        authority = '资产清单'
    elif all(row['mask_origin'] == 'source' for row in samples_by_study[study]):
        status = '作者原始cell mask，免额外复核'
        authority = '用户指定豁免；范围为原展示样例'
    else:
        status = '用户视觉复核通过'
        authority = '用户反馈；范围为原展示样例'
    decisions.append({'study': study, 'status': status, 'authority': authority})
decision_map = {row['study']: row['status'] for row in decisions}
result = {'date': '2026-11-19', 'manifest_version': original['manifest_version'],
          'studies': study_summaries, 'rois': records, 'user_decisions': decisions,
          'missing_studies': original['missing_studies']}
(OUTPUT / 'review_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
(OUTPUT / 'user_review_decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2))
priorities = [row for row in records if row['judgment'] != '样例可用']
priorities.sort(key=lambda row: row.get('priority_level', 2))
(OUTPUT / 'priority_rois.json').write_text(json.dumps(priorities, ensure_ascii=False, indent=2))

# Build a single entry page with study summaries and all per-ROI evidence.
parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CellWeft 分割专项复核</title><style>
body{font:15px/1.55 system-ui;color:#233039;background:#f4f6f7;margin:24px auto;max-width:1500px;padding:0 20px}
table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d8e0e3;padding:9px;text-align:left;vertical-align:top}
details{background:white;margin:16px 0;padding:15px;border:1px solid #d8e0e3}summary{cursor:pointer;font-weight:bold}
article{padding:16px 0;border-bottom:1px solid #d8e0e3}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}
input{padding:12px;font:inherit;width:90%;margin:18px 0}a{color:#066a83}.note{background:#e6f0f3;padding:15px}
</style><h1>分割专项复核：5 个 study / 165 个 ROI</h1>
<p class="note">根据用户逐图反馈开展。其他 study 的“通过”表示用户对上一版抽样图的视觉复核；作者原始 cell mask 按用户要求免额外复核。Bandyopadhyay 为明确点名的例外，检查输入、mask 配对与映射。每个 ROI 的具体发现、判断与限制见下方记录。</p>
<p><a href="../segmentation_roi_review_20261119/index.html">返回原抽样图像库</a> · <a href="review_summary.json">结构化复核记录</a> · <a href="priority_rois.json">需优先处理或补证的ROI</a></p>
<input id="search" placeholder="搜索 study、ROI、结论或异常"><table><tr><th>Study</th><th>ROI数</th><th>逐ROI判断计数</th><th>报告</th></tr>''']
for study in study_summaries:
    name = study['study']
    parts.append('<tr><td><a href="#{}">{}</a></td><td>{}</td><td>{}</td><td><a href="{}">report.md</a></td></tr>'.format(
        quote(name), display(name), study['roi_count'], display(study['judgments']), quote(study['report'], safe='/')))
parts.append('</table>')
parts.append('<details open><summary>需优先处理或补证的 ROI（{}）</summary><table><tr><th>Study / ROI</th><th>判断</th><th>建议后续</th></tr>'.format(len(priorities)))
for row in priorities:
    parts.append('<tr><td><a href="#{}">{}</a><br>{}</td><td>{}<br>{}</td><td>{}</td></tr>'.format(
        quote(row['study']), display(row['study']), display(row['roi']),
        display(row['judgment']), display(row.get('issue_category', '')), display(row['next_action'])))
parts.append('</table></details>')
for study in STUDIES:
    parts.append('<details open id="{}"><summary>{}</summary>'.format(quote(study), display(study)))
    report = (OUTPUT / study / 'report.md').read_text()
    parts.append('<details><summary>Study 完整报告</summary><pre>' + display(report) + '</pre></details>')
    for row in records:
        if row['study'] != study:
            continue
        parts.append('<article><h3>{} · {}</h3>'.format(display(row['roi']), display(row['judgment'])))
        for key, title in [('mask_origin','Mask来源'), ('findings','发现'), ('likely_cause','原因及证据边界'), ('next_action','建议后续')]:
            parts.append('<p><strong>{}</strong></p><pre>{}</pre>'.format(title, display(row[key])))
        for path in row['figures']:
            uri = quote(path, safe='/')
            parts.append('<a href="{0}" target="_blank"><img loading="lazy" src="{0}"></a>'.format(uri))
        parts.append('</article>')
    parts.append('</details>')
parts.append('<details><summary>其他 study 的用户复核记录与资产缺口</summary><table><tr><th>Study</th><th>状态</th><th>依据/范围</th></tr>')
for row in decisions:
    parts.append('<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(*(display(row[key]) for key in ['study','status','authority'])))
parts.append('</table><p>五项资产缺口：四项缺images.parquet；IMC_GC索引存在但0行。均无有效ROI分母，预期ROI数、缺失数和比例不可计算，完整性未确认。</p><pre>' + display(original['missing_studies']) + '</pre></details>')
parts.append('''<script>
document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('details[id]').forEach(d=>{d.hidden=!d.textContent.toLowerCase().includes(q);});});
</script></html>''')
(OUTPUT / 'index.html').write_text('\n'.join(parts))

# Refresh labels in the original gallery without re-reading or re-rendering pixels.
page = (ORIGINAL / 'index.html').read_text()
roi_decisions = {(row['study'], row['roi']): row['judgment'] for row in records}
for study, samples in samples_by_study.items():
    detail_pattern = r'(<details id=\x27study-' + re.escape(quote(study, safe='')) + r'\x27[^>]*>)(.*?)(</details>)'
    detail = re.search(detail_pattern, page, flags=re.DOTALL)
    body = detail[2]
    for sample in samples:
        sample['review_status'] = decision_map[study]
        if study in STUDIES:
            sample['review_status'] = '专项复核：' + roi_decisions[(study, sample['roi'])]
        elif sample['mask_origin'] == 'source':
            sample['review_status'] = '作者原始cell mask，免额外复核'
        head = '<h3>' + html.escape(sample['roi']) + '</h3><span class="tag">'
        body = re.sub(re.escape(head) + '[^<]*', lambda m: head + sample['review_status'], body)
    page = page[:detail.start(2)] + body + page[detail.end(2):]
    pattern = r'(<tr data-study=\x27' + re.escape(html.escape(study)) + r'\x27>.*?<td>)([^<]*)(</td></tr>)'
    page = re.sub(pattern, lambda m: m[1] + decision_map[study] + m[3], page)
banner = '<p id="followup-link"><a href="../segmentation_followup_20261119/index.html">查看5个study的逐ROI专项复核与用户确认记录</a></p>'
if 'id="followup-link"' not in page:
    page = page.replace('<main>', '<main>' + banner, 1)
page = page.replace('复核状态由你确认。', '用户确认与专项复核状态已记录；详见专项复核入口。')
for sample in original['rois']:
    old = html.escape(sample['figure'])
    new = html.escape(quote(sample['figure'], safe='/'))
    page = page.replace('src="' + old + '"', 'src="' + new + '"').replace('href="' + old + '"', 'href="' + new + '"')
(ORIGINAL / 'index.html').write_text(page)
(ORIGINAL / 'roi_review_summary.json').write_text(json.dumps(original, ensure_ascii=False, indent=2))
print('reviewed ROI:', len(records), 'study decisions:', len(decisions))
print(OUTPUT / 'index.html')
