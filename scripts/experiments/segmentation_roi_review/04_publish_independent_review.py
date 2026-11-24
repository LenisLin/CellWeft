"""Publish the actual gpt-6.2-sol/medium independent review.

Inputs: frozen ROI conclusions, Codex CLI response, and the reviewer's event log.
Outputs: an independent-review HTML linked from both existing galleries.
Scientific data and the reviewer's original response are not modified.
"""

import html
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import quote


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence')
REVIEW = ROOT / 'segmentation_followup_20261119/independent_codex_review'
response = json.loads((REVIEW / 'response.json').read_text())
frozen = json.loads((REVIEW / 'frozen_review_summary.json').read_text())
expected = {(row['study'], row['roi']) for row in frozen['rois']}
received = {(row['study'], row['roi']) for row in response['roi_reviews']}
if received != expected or len(response['roi_reviews']) != len(expected):
    raise ValueError('Independent review must account for all 165 unique ROI conclusions')


def escaped(value):
    """Escape plain or structured evidence for the static review report."""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(str(value))


counts = dict(Counter(row['verdict'] for row in response['roi_reviews']))
viewed = sum(bool(row['reviewed_figures']) and row['verdict'] != 'not_reviewed'
             for row in response['roi_reviews'])
parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codex CLI 独立复核</title><style>
body{font:15px/1.6 system-ui;color:#24333d;background:#f4f6f7;margin:24px auto;max-width:1500px;padding:0 20px}
table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d8e0e3;padding:9px;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}section{background:white;padding:15px;margin:15px 0}a{color:#066a83}h1{font-size:26px}
</style><h1>Codex CLI 独立复核：gpt-6.2-sol / medium</h1>
<p>调用端显式指定模型与推理档位，正式科学资产保持只读。独立意见与主控修订处理分别保留。</p>
<p><a href="../index.html">返回专项复核</a> · <a href="response.json">原始独立意见 JSON</a> · <a href="request.json">调用记录</a> · <a href="events.jsonl">CLI 事件日志</a></p>''']
parts.append('<p>总体意见：{}；记录 {}/165；报告已看图 {}/165。</p><pre>{}</pre><pre>{}</pre>'.format(
    escaped(response['verdict']), len(received), viewed, escaped(counts), escaped(response['summary'])))
resolution = REVIEW / 'resolution.json'
if resolution.exists():
    parts.append('<section><h2>主控修订处理</h2><pre>{}</pre></section>'.format(escaped(json.loads(resolution.read_text()))))
parts.append('<section><h2>独立复核发现</h2>')
for issue in response['issues']:
    parts.append('<h3>{} · {} / {}</h3><pre>{}</pre>'.format(
        escaped(issue['severity']), escaped(issue['study']), escaped(issue['roi']), escaped(issue)))
parts.append('</section><section><h2>Study 总结与状态归属</h2><pre>{}</pre><pre>{}</pre></section>'.format(
    escaped(response['study_reviews']), escaped(response['status_attribution_review'])))
parts.append('<section><h2>逐 ROI 独立意见</h2><table><tr><th>Study / ROI</th><th>复核意见</th><th>证据与处理建议</th></tr>')
for row in response['roi_reviews']:
    links = []
    for figure in row['reviewed_figures']:
        path = Path(figure)
        if not path.is_absolute():
            path = ROOT / 'segmentation_followup_20261119' / path
            if not path.is_file():
                path = ROOT / 'segmentation_followup_20261119' / row['study'] / figure
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = os.path.relpath(path, REVIEW)
        links.append('<a href="{}">{}</a>'.format(quote(relative, safe='/'), escaped(path.name)))
    parts.append('<tr><td>{}<br>{}</td><td>{}<br>{}</td><td><pre>{}</pre><pre>{}</pre>{}</td></tr>'.format(
        escaped(row['study']), escaped(row['roi']), escaped(row['verdict']), escaped(row['independent_judgment']),
        escaped(row['assessment']), escaped(row['required_change']), '<br>'.join(links)))
parts.append('</table></section><h2>独立复核限制</h2><pre>{}</pre></html>'.format(escaped(response['limitations'])))
(REVIEW / 'report.html').write_text('\n'.join(parts))
(REVIEW / 'coverage.json').write_text(json.dumps({
    'unique_roi_count': len(received), 'viewed_roi_count_reported': viewed,
    'verdicts': counts, 'overall_verdict': response['verdict'],
}, ensure_ascii=False, indent=2))
for folder in ['segmentation_followup_20261119', 'segmentation_roi_review_20261119']:
    page_path = ROOT / folder / 'index.html'
    page = page_path.read_text()
    if 'id="independent-review-link"' not in page:
        link = '<p id="independent-review-link"><a href="../segmentation_followup_20261119/independent_codex_review/report.html">查看 gpt-6.2-sol / medium 独立复核意见与修订记录</a></p>'
        page = page.replace('<body>', '<body>' + link, 1) if '<body>' in page else page.replace('<h1>', link + '<h1>', 1)
        page_path.write_text(page)
print('Independent review:', response['verdict'], counts, 'viewed reported:', viewed)
