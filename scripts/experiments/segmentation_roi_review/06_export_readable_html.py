"""Export readable Chinese QC reports using the actual CLI editorial output.

Inputs: current review_summary.json, original independent opinions, editorial.json.
Outputs: renewed main and independent HTML pages; scientific assets stay intact.
"""

import html
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import quote


ROOT = Path('/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence')
REPORT = ROOT / 'segmentation_followup_20261119'
EDITION = REPORT / 'readable_edition'
data = json.loads((REPORT / 'review_summary.json').read_text())
edited = json.loads((EDITION / 'editorial.json').read_text())
review = json.loads((REPORT / 'independent_codex_review/response.json').read_text())
copy_by_roi = {(r['study'], r['roi']): r for r in edited['rois']}
expected = {(r['study'], r['roi']) for r in data['rois']}
if len(edited['rois']) != 165 or set(copy_by_roi) != expected:
    raise ValueError('Editorial output must cover exactly the current 165 ROI')
copy_by_study = {r['study']: r for r in edited['studies']}
if len(edited['studies']) != 5 or set(copy_by_study) != {r['study'] for r in data['studies']}:
    raise ValueError('Editorial output must cover exactly the current five studies')
opinions = {(r['study'], r['roi']): r for r in review['roi_reviews']}
ACTIONS = REPORT / 'user_decisions_20261119'
confirmed = json.loads((ACTIONS / 'decisions.json').read_text())
decisions = {(r['study'], r['roi']): r for r in confirmed['decisions']}
display_assets = json.loads((ACTIONS / 'display_assets.json').read_text())
action_labels = {'accept': '用户确认接受', 'discard': '已弃用',
                 'exclude_channels': '指定通道已排除', 'resegment': '需要重新分割'}
short_names = {
    'Allam2024_IMC_NormalSLO': 'Allam · 正常淋巴器官 · IMC',
    'Bandyopadhyay2024_CODEX_BoneMarrow': 'Bandyopadhyay · 骨髓 · CODEX',
    'Iglesia2024_CODEX_BreastCancer': 'Iglesia · 乳腺癌 · CODEX',
    'Ly2025_COMET_AML': 'Ly · 急性髓系白血病 · COMET',
    'Lyman2025_IMC_PanIN': 'Lyman · 胰腺上皮内瘤变 · IMC',
}
verdicts = {'supported': '支持原结论', 'revision_required': '建议修订原结论',
            'insufficient_evidence': '现有证据仍不足'}


def esc(value):
    """Escape text without exposing structured machine records in the page."""
    return html.escape(str(value))


def uri(path, parent):
    """Keep evidence links relative to the exported HTML location."""
    return quote(os.path.relpath(path, parent), safe='/')


def anchor(row):
    return 'roi-' + row['study'] + '-' + row['roi']


def badge(judgment):
    color = {'样例可用': 'good', '局部问题': 'problem', '证据不足': 'uncertain',
             '用户确认接受': 'good', '已弃用': 'uncertain',
             '指定通道已排除': 'good', '需要重新分割': 'problem'}[judgment]
    return f'<span class="badge {color}">{judgment}</span>'


def decision_text(row):
    """Make the user's decision the current usage instruction for this ROI."""
    decision = decisions.get((row['study'], row['roi']))
    if not decision:
        return row['judgment'], copy_by_roi[row['study'], row['roi']]['next_step']
    action = decision['action']
    if action == 'accept':
        text = '按你的确认保留当前结果；固定划分保持。' + (
            '此 ROI 属于你确认整体保留的 Ly 数据集。' if row['study'] == 'Ly2025_COMET_AML' else '')
    elif action == 'discard':
        text = '按你的确认整张 ROI 弃用，已退出当前实验成员；原图、原 mask 及来源记录保留。'
    elif action == 'exclude_channels':
        text = ('按你的确认仅在本 ROI 排除 ' + '、'.join(decision['channels']) +
                ' 的当前输入、表达测量及任务目标；保留其他通道和现有 mask。'
                '现有 mask 曾使用被排除通道生成，这一历史依赖仍保留。')
    else:
        text = ('按你的确认需要重新分割，旧分割已退出当前实验使用。'
                '已另存一份重分割候选；候选尚未验收，也尚未替换正式 mask。')
    return action_labels[action], text


style = '''
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#202c35;font:18px/1.8 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
main{max-width:1280px;margin:auto;padding:36px 26px 80px}h1{font-size:32px;line-height:1.4;margin:8px 0 20px}h2{font-size:25px;margin:0 0 16px}h3{font-size:21px;margin:0 0 12px}p{margin:10px 0 18px;max-width:88ch}a{color:#056b87;text-underline-offset:4px}a:hover{color:#003d50}
section,.box{background:white;border:1px solid #dce3e8;border-radius:12px;padding:25px;margin:22px 0}.eyebrow,.muted{color:#566875}.eyebrow{font-size:15px;letter-spacing:.08em}.counts{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}.count{padding:14px 22px;background:white;border:1px solid #dce3e8;border-radius:10px}.count strong{font-size:29px;display:block;line-height:1.3}
.badge{display:inline-block;padding:2px 11px;border-radius:20px;font-size:15px;font-weight:600;white-space:nowrap}.good{background:#e5f4eb;color:#185f38}.problem{background:#fff0e5;color:#914200}.uncertain{background:#eef0f4;color:#4a5768}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:16px}td,th{text-align:left;vertical-align:top;border-bottom:1px solid #e0e6ea;padding:13px 10px}th{background:#f3f6f8}td p{margin:0}td.id{min-width:190px;overflow-wrap:anywhere}td.action{min-width:260px}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:20px 0}input[type=search]{font:inherit;padding:11px 14px;border:1px solid #afbdc7;border-radius:7px;flex:1;min-width:240px}button,select{font:inherit;padding:10px;border:1px solid #afbdc7;background:white;border-radius:7px;cursor:pointer}
nav{display:flex;gap:10px;flex-wrap:wrap}nav a{padding:5px 10px;background:white;border:1px solid #cdd8df;border-radius:6px;font-size:16px}.roi{margin:16px 0;border:1px solid #d8e0e6;border-radius:9px;background:white}.roi>summary{padding:17px 20px;cursor:pointer;line-height:1.6}.roi>summary .roi-id{overflow-wrap:anywhere;font-size:17px;margin-right:12px}.takeaway{display:block;color:#344856;margin:7px 0 0 20px;font-size:18px}.roi-body{padding:0 22px 22px}.label{font-weight:700;margin-bottom:3px}.limitation{color:#566875;border-left:3px solid #c5d1d9;padding-left:14px;font-size:16px}.figure img{width:100%;height:auto;display:block;border-radius:6px;background:#162027}.figure{margin:15px 0}.figure figcaption{font-size:15px;color:#566875;margin-top:8px}.support>summary,.technical>summary{cursor:pointer;color:#466170;padding:8px 0}.technical{font-size:16px}.study-id{font-size:14px;color:#657783;overflow-wrap:anywhere}.priority-one{border-left:5px solid #bb651b;padding-left:16px}.notice{background:#eef5f8;padding:14px 18px;border-radius:8px}.legend p{font-size:16px;margin:8px 0}.study{scroll-margin-top:20px}.source{font-size:15px;color:#5f7280}.empty{text-align:center;padding:20px;color:#657783}[hidden]{display:none!important}
@media(max-width:650px){main{padding:20px 12px}section,.box{padding:17px}body{font-size:17px}h1{font-size:26px}.roi-body{padding:0 12px 15px}.counts{gap:8px}.count{padding:10px 15px}button{font-size:15px}}
'''
style += '.native-grid{display:flex;flex-wrap:wrap;gap:22px;align-items:start}.native-grid figure{margin:10px 0;min-width:110px}.native-grid img{display:block;width:50px;height:50px;image-rendering:pixelated;background:#12181d}.native-grid.zoom img{width:200px;height:200px}.native-grid figcaption{font-size:15px;max-width:200px}.history{border-left:3px solid #d6dfe5;padding-left:15px;color:#566875}'
priorities = sorted((r for r in data['rois'] if r['judgment'] != '样例可用'
                     and (r['study'], r['roi']) not in decisions),
                    key=lambda r: r.get('priority_level', 2))
counts = Counter(r['judgment'] for r in data['rois'])

# Both existing entry links lead to a readable account of the final conclusions.
for path, independent_page in [(REPORT / 'index.html', False),
                               (REPORT / 'independent_codex_review/report.html', True)]:
    parent = path.parent
    archive = EDITION / ('previous_independent_html.txt' if independent_page else 'previous_main_html.txt')
    if not archive.exists():
        archive.write_text(path.read_text())
    title = '独立复核后的分割质控报告' if independent_page else '细胞分割质控报告'
    parts = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{style}</style></head><body><main>',
             '<div class="eyebrow">CELLWEFT · 2026年11月19日</div>', f'<h1>{title}</h1>',
             '<p>本页按你逐 ROI 的最新确认更新使用状态，并保留之前的图像观察供追溯。Ly 的 16 个 ROI 全部保留。</p>',
             '<p class="source">文字与决定映射均经过实际 gpt-6.2-sol / medium CLI 复核。使用清单版本：20261119_v3。</p>']
    parts.append('<div class="counts"><div class="count"><strong>34</strong>用户接受</div><div class="count"><strong>11</strong>整张弃用</div><div class="count"><strong>4</strong>仅排除指定通道</div><div class="count"><strong>1</strong>需要重新分割</div></div>')
    parts.append('<div class="notice">50 个 ROI 有你的明确处置；页面保留全部 165 个历史复核记录。点击 ROI 查看图片。接受的 ROI 已采用你的确认；旧模型观察放在历史记录中。</div>')
    parts.append('<section><h2>你的确认与实施结果</h2><p>“弃用”已落实到当前实验成员清单，原始资产保留。INT_Donor_E 暂停使用旧分割，另存候选结果。通道排除仅针对指定 ROI，使用时同时排除输入、表达测量及任务目标；当前 mask 保留。</p><p><b>8 个小视野为何看起来很大：</b>lyman_06_a008–a011、lyman_10_a001–a004 的原始 image、input 和 mask 均为 50×50 像素，像素尺寸为 1 µm；整个视野为 50×50 µm。旧 HTML 将小图放入大画布后铺满页面。下面已提供原尺寸图和明确标注的 4 倍像素放大图。</p><details class="support"><summary>展开全部 50 个 ROI 的处置表</summary><div class="scroll"><table><tr><th>数据集 / ROI</th><th>当前处置</th><th>实施内容</th></tr>')
    for item in confirmed['decisions']:
        row = next(r for r in data['rois'] if (r['study'], r['roi']) == (item['study'], item['roi']))
        label, instruction = decision_text(row)
        parts.append(f'<tr><td class="id"><a href="#{quote(anchor(row))}">{esc(row["roi"])}</a><br>{esc(short_names[row["study"]].split(" · ")[0])}</td><td>{badge(label)}</td><td>{esc(instruction)}</td></tr>')
    parts.append('</table></div></details><p><a href="' + uri(ACTIONS / 'summary.md', parent) + '">完整中文处置清单</a> · <a href="' + uri(ACTIONS / 'independent_review.md', parent) + '">CLI 独立核对记录</a></p></section>')
    parts.append('<details class="box legend"><summary>如何理解这些判断和图片？</summary><p><b>样例可用：</b>抽查窗口中未发现明显异常，不能据此保证整张 ROI 都没有问题。</p><p><b>局部问题：</b>图中存在通道异常或分割疑点，需要核实具体部位。</p><p><b>证据不足：</b>目前的信号或细节不足以判断细胞边界，需要补充依据。</p><p><b>ROI：</b>本次查看的组织区域。<b>DAPI / DNA：</b>用于观察细胞核的信号。<b>实例分割图：</b>不同颜色表示不同的细胞编号，叠加图上的线条表示分割轮廓。</p></details>')
    parts.append('<nav>')
    for study in data['studies']:
        parts.append(f'<a href="#{quote(study["study"])}">{esc(short_names[study["study"]].split(" · ")[0])}</a>')
    parts.append(f'</nav><section><h2>其余历史待核实项 · {len(priorities)} 个 ROI</h2><p>下表只保留本次未点名、且此前仍有疑点的记录。INT_Donor_E 的重新分割单独按你的明确决定执行。</p><details class="support"><summary>展开尚未更新处置的历史记录</summary><div class="scroll"><table><tr><th>数据集 / ROI</th><th>主要发现</th><th>此前建议</th></tr>')
    for row in priorities:
        text = copy_by_roi[row['study'], row['roi']]
        parts.append(f'<tr><td class="id"><a href="#{quote(anchor(row))}">{esc(row["roi"])}</a><br><span class="muted">{esc(short_names[row["study"]].split(" · ")[0])}</span><br>{badge(row["judgment"])}</td><td>{esc(text["takeaway"])}</td><td class="action">{esc(text["next_step"])}</td></tr>')
    parts.append('</table></div></details></section><h2>按数据集看图</h2><div class="toolbar"><input type="search" id="search" aria-label="搜索ROI" placeholder="搜索 ROI、数据集或发现"><select id="status" aria-label="筛选判断"><option value="">全部状态</option><option>用户确认接受</option><option>已弃用</option><option>指定通道已排除</option><option>需要重新分割</option><option>样例可用</option><option>局部问题</option><option>证据不足</option></select><button id="expand">展开显示的 ROI</button><button id="collapse">收起全部 ROI</button></div><p id="matches" class="muted" aria-live="polite"></p>')
    for study in data['studies']:
        name = study['study']; text = copy_by_study[name]
        current = Counter(action_labels[r['action']] for r in confirmed['decisions'] if r['study'] == name)
        parts.append(f'<section class="study" id="{quote(name)}"><h2>{esc(short_names[name])}</h2><div class="study-id">{esc(name)}</div><p><b>本次确认：</b>' + '；'.join(f'{label} {number} 个' for label, number in current.items()) + '。</p>')
        parts.append(f'<details class="support"><summary>此前独立审阅的研究概述</summary><p>{esc(text["summary"])}</p><p>{esc(text["next_step"])}</p></details>')
        for row in data['rois']:
            if row['study'] != name:
                continue
            text = copy_by_roi[name, row['roi']]
            label, instruction = decision_text(row)
            decision = decisions.get((name, row['roi']))
            search = esc(' '.join([name, short_names[name], row['roi'], label, instruction, *[text[k] for k in ['takeaway', 'observation', 'next_step']]]).lower())
            opened = ' open' if decision and decision['action'] == 'resegment' else ''
            takeaway = instruction if decision else text['takeaway']
            parts.append(f'<details class="roi" id="{esc(anchor(row))}" data-status="{label}" data-search="{search}"{opened}><summary><span class="roi-id">{esc(row["roi"])}</span>{badge(label)}<span class="takeaway">{esc(takeaway)}</span></summary><div class="roi-body">')
            parts.append(f'<p class="label">当前处置</p><p>{esc(instruction)}</p>')
            if decision:
                parts.append('<details class="support history"><summary>查看此前模型的图像观察与建议</summary>')
            parts.append(f'<p class="label">此前图像观察</p><p>{esc(text["observation"])}</p><p class="label">此前建议</p><p>{esc(text["next_step"])}</p>')
            if text['limitation']:
                parts.append(f'<p class="limitation">{esc(text["limitation"])}</p>')
            if decision:
                parts.append('</details>')
            asset = display_assets.get(row['roi'])
            if asset and asset['kind'] == 'native_small_fov':
                parts.append('<p><b>原尺寸：50×50 像素；1 µm/px；视野 50×50 µm。</b>以下每张图以 50×50 CSS 像素显示。</p>')
                for zoom in [False, True]:
                    if zoom:
                        parts.append('<details class="support"><summary>查看 4 倍像素放大（200×200，不增加原始细节）</summary>')
                    parts.append('<div class="native-grid' + (' zoom' if zoom else '') + '">')
                    for panel in asset['panels']:
                        link = uri(REPORT / panel['path'], parent)
                        parts.append(f'<figure><a href="{link}"><img loading="lazy" src="{link}" width="50" height="50" alt="{esc(panel["label"])} 原始50×50像素"></a><figcaption>{esc(panel["label"])}</figcaption></figure>')
                    parts.append('</div>')
                    if zoom:
                        parts.append('</details>')
            elif asset:
                new_figures = asset.get('figures', [asset.get('figure')])
                for figure in new_figures:
                    link = uri(REPORT / figure, parent)
                    parts.append(f'<figure class="figure"><a href="{link}"><img loading="lazy" src="{link}" alt="{esc(row["roi"])} 本次调整对照"></a><figcaption>本次调整对照 · 点击查看原图</figcaption></figure>')
            if asset:
                parts.append('<details class="support"><summary>查看调整前的历史质控图</summary>')
            for index, figure in enumerate(row['figures']):
                if index == 1:
                    parts.append('<details class="support"><summary>查看其他窗口和来源对照图</summary>')
                link = uri(REPORT / figure, parent)
                parts.append(f'<figure class="figure"><a href="{link}" target="_blank"><img loading="lazy" src="{link}" alt="{esc(row["roi"])} 质控图 {index+1}"></a><figcaption>质控图 {index+1} · 点击按原图大小查看</figcaption></figure>')
            if len(row['figures']) > 1:
                parts.append('</details>')
            if asset:
                parts.append('</details>')
            opinion = opinions[name, row['roi']]
            parts.append(f'<details class="technical"><summary>复核记录与技术依据</summary><p>此前独立审阅对模型原结论的意见：{verdicts[opinion["verdict"]]}。本次使用处置以你的逐 ROI 决定为准。</p><p><a href="{uri(REPORT / name / "roi_review.json", parent)}">逐 ROI 原始记录</a> · <a href="{uri(REPORT / "independent_codex_review/response.json", parent)}">完整独立意见</a></p></details></div></details>')
        parts.append('</section>')
    parts.append('<section><h2>还有 5 个数据集暂时无法看图</h2><p>以下数据集缺少可用的当前图像清单，因此无法计算应有多少 ROI、缺失多少文件或缺失比例。它们尚未完成本轮图像质量复核。</p><div class="scroll"><table><tr><th>数据集</th><th>当前缺口</th></tr>')
    for row in data['missing_studies']:
        reason = '图像清单存在，但没有图像记录。' if row['image_index_exists'] else '缺少图像清单。'
        parts.append(f'<tr><td>{esc(row["study"])}</td><td>{reason}</td></tr>')
    parts.append('</table></div></section><details class="box technical"><summary>其他数据集与原始记录</summary><p>其他 107 个有资产的数据集沿用你的确认：65 个已由你视觉复核通过，42 个使用作者原始分割结果，按你的要求免额外复核。本次独立看图覆盖上方 5 个数据集的 165 个 ROI。</p><ul>')
    links = [('原抽样图像库', ROOT / 'segmentation_roi_review_20261119/index.html'),
             ('结构化最终结论', REPORT / 'review_summary.json'),
             ('优先复核清单', REPORT / 'priority_rois.json'),
             ('独立审阅原文', REPORT / 'independent_codex_review/response.json'),
             ('独立审阅后的逐项处理', REPORT / 'independent_codex_review/resolution.json'),
             ('本次文字整理的调用记录', EDITION / 'request.json')]
    for label, target in links:
        parts.append(f'<li><a href="{uri(target, parent)}">{label}</a></li>')
    parts.append('</ul></details></main><script>const rois=[...document.querySelectorAll(".roi")];function filter(){const q=document.getElementById("search").value.trim().toLowerCase(),status=document.getElementById("status").value;let n=0;rois.forEach(r=>{r.hidden=!(r.dataset.search.includes(q)&&(!status||r.dataset.status===status));if(!r.hidden)n++;});document.querySelectorAll(".study").forEach(s=>s.hidden=![...s.querySelectorAll(".roi")].some(r=>!r.hidden));document.getElementById("matches").textContent=`显示 ${n} / 165 个 ROI`;}document.getElementById("search").addEventListener("input",filter);document.getElementById("status").addEventListener("change",filter);document.getElementById("expand").onclick=()=>rois.filter(r=>!r.hidden).forEach(r=>r.open=true);document.getElementById("collapse").onclick=()=>rois.forEach(r=>r.open=false);filter();function reveal(){const r=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(r&&r.classList.contains("roi")){r.open=true;r.scrollIntoView({block:"start"});}}window.addEventListener("hashchange",reveal);reveal();</script></body></html>')
    path.write_text('\n'.join(parts))
    print(path)

# A new filename gives the reader an unambiguous entry to the renewed edition.
(REPORT / 'readable.html').write_text((REPORT / 'index.html').read_text())
