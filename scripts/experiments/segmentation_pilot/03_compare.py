"""Describe A/B instance agreement and aligned source-reference concordance.
Inputs: manifest, prepared inputs, saved masks/runtime from 01/02.
Outputs: comparison/*.csv/json, full-ROI and fixed-center PNGs, offline HTML report.
Source masks are references, not human gold standards. Each source image is one unit.
"""
import os
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "8"

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pilot_io import arguments, load_samples, labels, instance_metrics, cell_statistics, write_json


def save_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def boundaries(mask: np.ndarray) -> np.ndarray:
    """Instance outlines depend on label equality, not numeric label magnitude."""
    edge = np.zeros(mask.shape, bool)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:-1] |= mask[:-1] != mask[1:]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return edge & (mask > 0)


def preview(sample_id: str, a: np.ndarray, b: np.ndarray, masks: dict,
            reference: np.ndarray | None, output: Path, zoom: bool) -> None:
    """Same B background in every mask panel; center zoom fixed before predictions."""
    h, w = b.shape[:2]
    if zoom:
        y, x = max(0, (h - 384)//2), max(0, (w - 384)//2)
        crop = (slice(y, min(h, y+384)), slice(x, min(w, x+384)))
    else:
        crop = (slice(None), slice(None))
    panels = [('input_A', None), ('input_B', None)] + [(s, masks.get(s)) for s in SCHEMES] + [('source reference', reference)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout='constrained')
    base = np.stack([b[..., 0], b[..., 0]*0.45, b[..., 1]], axis=-1)
    for ax, (name, mask) in zip(axes.ravel(), panels):
        view = a if name == 'input_A' else b
        rgb = np.stack([view[..., 0], view[..., 0]*0.45, view[..., 1]], axis=-1) if name.startswith('input') else base
        rgb = rgb[crop].copy()
        if mask is not None:
            rgb[boundaries(mask)[crop]] = [0, 1, 0]
        ax.imshow(np.clip(rgb, 0, 1), interpolation='nearest')
        suffix = ' (nuclei)' if name == 'SD_B' else ''
        if mask is None and not name.startswith('input'):
            suffix = ' (not available)' if name == 'source reference' else ' (missing / failed)'
        ax.set_title(name + suffix)
        ax.axis('off')
    extent = f'center 384px maximum, native origin x={x},y={y}' if zoom else f'full fixed ROI {w}x{h}'
    fig.suptitle(f'{sample_id}: {extent}; red=membrane, blue=nucleus, green=instance boundary')
    fig.savefig(output, dpi=150)
    plt.close(fig)


def html_table(rows: list[dict]) -> str:
    if not rows:
        return '<p>无可用指标。</p>'
    columns = list(dict.fromkeys(k for row in rows for k in row))
    header = ''.join(f'<th>{html.escape(c)}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(row.get(c, "")))}</td>' for c in columns) + '</tr>' for row in rows)
    return f'<div class="table"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


args = arguments(__doc__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
samples = load_samples(args.manifest)
SCHEMES = ['CP_A', 'CP_B', 'IS_A', 'IS_B', 'SD_B']
PAIRS = [('CP_A', 'CP_B'), ('IS_A', 'IS_B'), ('CP_B', 'IS_B'), ('CP_A', 'IS_A')]
out = args.output / 'comparison'
(out / 'figures').mkdir(parents=True, exist_ok=True)
cell_rows, metric_rows, availability, sections, differences = [], [], [], [], []

# Retain every manifest image, including missing/failed combinations.
for sample in samples:
    sid = sample['sample_id']
    identity = {k: sample[k] for k in ['sample_id', 'study', 'technology']}
    prep = args.output / 'prepared' / sid
    available = {}
    image_rows, image_metrics = [], []
    a_path, b_path = prep / 'input_A.npy', prep / 'input_B.npy'
    prep_runtime_path = args.output / 'runtime' / 'preparation' / f'{sid}.json'
    prep_runtime = json.loads(prep_runtime_path.read_text()) if prep_runtime_path.exists() else {}
    if not a_path.exists() or not b_path.exists() or prep_runtime.get('status') != 'ok':
        availability.append({**identity, 'scheme': 'preparation', 'status': prep_runtime.get('status', 'missing'),
                             'error': prep_runtime.get('error', '')})
        for scheme in SCHEMES:
            runtime_path = args.output / 'runtime' / scheme / f'{sid}.json'
            runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
            availability.append({**identity, 'scheme': scheme,
                                 'status': runtime.get('status', 'blocked_preparation'),
                                 'error': runtime.get('error', ''), 'reason': runtime.get('reason', '')})
        context = html.escape(json.dumps({k: sample.get(k, 'unknown') for k in
                              ['source_intensity_phase', 'fit_scope', 'mpp']}, ensure_ascii=False))
        sections.append(f'<section id="{html.escape(sid)}"><h2>{html.escape(sid)}</h2>'
                        f'<p>准备未成功，未生成图像比较。</p><pre>{context}</pre></section>')
        continue
    a, b = np.load(a_path, allow_pickle=False), np.load(b_path, allow_pickle=False)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError(f'{sid}: A/B shapes must be identical H,W,2')
    for scheme in SCHEMES:
        path = args.output / 'masks' / scheme / f'{sid}.npy'
        runtime_path = args.output / 'runtime' / scheme / f'{sid}.json'
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        ok = path.exists() and runtime.get('status') == 'ok'
        status = 'ok' if ok else ('missing' if runtime.get('status') == 'ok' else runtime.get('status', 'missing'))
        availability.append({**identity, 'scheme': scheme, 'status': status,
                             'error': runtime.get('error', ''), 'reason': runtime.get('reason', '')})
        if not ok:
            continue
        mask = labels(np.load(path, allow_pickle=False), b.shape[:2])
        available[scheme] = mask
        row = {**identity, 'scheme': scheme, 'target': 'nuclei' if scheme == 'SD_B' else 'whole_cell',
               **cell_statistics(mask, sample.get('mpp')),
               **{k: runtime.get(k) for k in ['inference_seconds', 'rss_bytes', 'process_peak_rss_bytes',
                                             'gpu_max_allocated_bytes', 'gpu_max_reserved_bytes']}}
        cell_rows.append(row)
        image_rows.append(row)
    for left, right in PAIRS:
        if left in available and right in available:
            row = {**identity, 'comparison': f'{left}_vs_{right}',
                   'interpretation': 'within_model_A_B_agreement' if left[:2] == right[:2] else 'cross_model_agreement',
                   'reference_side': left, 'prediction_side': right, 'target': 'whole_cell',
                   **instance_metrics(available[left], available[right])}
            metric_rows.append(row)
            image_metrics.append(row)
    for model in ['CP', 'IS']:
        paired = {r['scheme']: r for r in image_rows}
        if f'{model}_A' in paired and f'{model}_B' in paired:
            differences.append({**identity, 'comparison': f'{model}_B_minus_A',
                **{key: paired[f'{model}_B'][key] - paired[f'{model}_A'][key]
                   if paired[f'{model}_B'][key] is not None and paired[f'{model}_A'][key] is not None else None
                   for key in ['count', 'coverage', 'border_fraction', 'area_px_median', 'area_um2_median']}})
    # Saved IS nuclei share native coordinates; never compare them to whole-cell labels.
    if 'SD_B' in available:
        for scheme in ['IS_A', 'IS_B']:
            nuclei_path = args.output / 'masks' / scheme / f'{sid}_nuclei.npy'
            if scheme in available and nuclei_path.exists():
                nuclei = labels(np.load(nuclei_path, allow_pickle=False), b.shape[:2])
                row = {**identity, 'comparison': f'{scheme}_nuclei_vs_SD_B',
                       'interpretation': 'auxiliary_nuclear_agreement', 'target': 'nuclei',
                       'reference_side': scheme + '_nuclei', 'prediction_side': 'SD_B',
                       **instance_metrics(nuclei, available['SD_B'])}
                metric_rows.append(row)
                image_metrics.append(row)
    reference = None
    if sample['reference'] is not None:
        reference = labels(np.load(prep / 'reference.npy', allow_pickle=False), b.shape[:2])
        for scheme, mask in available.items():
            if sample['reference']['whole_cell'] != (scheme != 'SD_B'):
                continue
            row = {**identity, 'comparison': f'source_vs_{scheme}', 'interpretation': 'source_reference_concordance',
                   'reference_side': 'source', 'prediction_side': scheme,
                   'target': 'whole_cell' if sample['reference']['whole_cell'] else 'nuclei',
                   'source_method': sample['reference']['method'],
                   'source_acquisition': sample['reference']['acquisition'],
                   **instance_metrics(reference, mask)}
            metric_rows.append(row)
            image_metrics.append(row)
    for zoom, suffix in [(False, 'full'), (True, 'center')]:
        preview(sid, a, b, available, reference, out / 'figures' / f'{sid}_{suffix}.png', zoom)
    notes = html.escape(sample.get('notes', ''))
    stats = json.loads((prep / 'stats.json').read_text())
    context = html.escape(json.dumps({k: stats.get(k) for k in [
        'source_intensity_phase', 'source_phase_interpretation', 'fit_scope', 'fit_region_count',
        'statistics_domain', 'statistics_status', 'actual_domain', 'physical_fov_um', 'roi_edges',
        'gaussian']}, ensure_ascii=False, indent=2))
    diagnostics = html.escape(json.dumps(stats['input_diagnostics'], ensure_ascii=False, indent=2))
    sections.append(f'<section id="{html.escape(sid)}"><h2>{html.escape(sid)} — {html.escape(sample["technology"])}</h2>'
        f'<p>{html.escape(sample["study"])}；{notes}</p>'
        f'<a href="../prepared/{sid}/stats.json">输入域/拟合统计</a>'
        f'<h3>来源阶段、拟合范围与边缘</h3><pre>{context}</pre>'
        f'<details><summary>所选来源通道与 A/B 输入诊断</summary><pre>{diagnostics}</pre></details>'
        f'<div class="panzoom"><img src="figures/{sid}_full.png" alt="full ROI"></div>'
        f'<details><summary>固定中心局部图</summary><div class="panzoom"><img src="figures/{sid}_center.png" alt="fixed center"></div></details>'
        + html_table(image_rows) + html_table(image_metrics) + html_table([r for r in differences if r['sample_id'] == sid]) + '</section>')

# Descriptive technology summaries keep actual n and the intended n=3 visible.
groups = defaultdict(list)
for sample in samples:
    groups[sample['technology']].append(sample['sample_id'])
summary_rows = []
for technology, sample_ids in groups.items():
    for scheme in SCHEMES:
        rows = [r for r in cell_rows if r['technology'] == technology and r['scheme'] == scheme]
        result = {'technology': technology, 'series': scheme, 'intended_n': 3,
                  'manifest_n': len(sample_ids), 'n_available': len(rows), 'unit': 'source image / fixed ROI'}
        for status in ['ok', 'failed', 'skipped', 'missing', 'blocked_preparation']:
            result['n_' + status] = sum(r['technology'] == technology and r['scheme'] == scheme
                                        and r['status'] == status for r in availability)
        for key in ['count', 'coverage', 'area_um2_median', 'border_fraction', 'inference_seconds']:
            values = [r[key] for r in rows if r[key] is not None]
            result[key + '_image_mean'] = float(np.mean(values)) if values else None
            result[key + '_n_defined'] = len(values)
        summary_rows.append(result)
    for comparison in sorted({f'{a}_vs_{b}' for a, b in PAIRS} | {r['comparison'] for r in metric_rows}):
        rows = [r for r in metric_rows if r['technology'] == technology and r['comparison'] == comparison]
        result = {'technology': technology, 'series': comparison, 'intended_n': 3,
                  'manifest_n': len(sample_ids), 'n_available': len(rows), 'unit': 'source image / fixed ROI'}
        for key in ['precision', 'recall', 'f1', 'foreground_iou']:
            values = [r[key] for r in rows if r[key] is not None]
            result[key + '_image_mean'] = float(np.mean(values)) if values else None
            result[key + '_n_defined'] = len(values)
        summary_rows.append(result)

save_csv(out / 'cell_statistics.csv', cell_rows)
save_csv(out / 'paired_differences.csv', differences)
save_csv(out / 'instance_comparisons.csv', metric_rows)
save_csv(out / 'availability.csv', availability)
save_csv(out / 'technology_summary.csv', summary_rows)
write_json(out / 'results.json', {'manifest_n': len(samples), 'intended_n': 12,
    'paired_differences': differences,
    'primary_ok': sum(r['scheme'] in ['CP_A', 'CP_B'] and r['status'] == 'ok' for r in availability),
    'completion': 'complete' if len(samples) == 12 and len({s['study'] for s in samples}) == 12
        and all(len(v) == 3 for v in groups.values()) and len(groups) == 4
        and sum(r['scheme'] in ['CP_A', 'CP_B'] and r['status'] == 'ok' for r in availability) == 24 else 'partial',
    'technology_counts': {k: len(v) for k, v in groups.items()}, 'availability': availability,
    'cell_statistics': cell_rows, 'instance_comparisons': metric_rows, 'technology_summary': summary_rows,
    'interpretation': 'A/B agreement and source-reference concordance; no significance or accuracy ranking',
    'matching': 'sparse maximum-cardinality one-to-one IoU>=0.5; background=0; ID permutation invariant',
    'empty_convention': 'undefined precision/recall/F1=null; both-empty foreground IoU=1'})
nav = ' · '.join(f'<a href="#{html.escape(s["sample_id"])}">{html.escape(s["sample_id"])}</a>' for s in samples)
report = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CellWeft segmentation pilot</title>
<style>body{font-family:system-ui;margin:24px;color:#222}section{border-top:1px solid #bbb;margin-top:32px}th,td{padding:5px;border:1px solid #ddd;white-space:nowrap}table{border-collapse:collapse}.table,.panzoom{overflow:auto;max-width:100%}img{max-width:100%;cursor:zoom-in}img.expanded{max-width:none;width:3000px;cursor:zoom-out}summary{cursor:pointer}nav{position:sticky;top:0;background:white;padding:8px}</style>
<h1>12 源图分割快速比较</h1><p>每技术计划 n=3，逐图为分析单位；大图只比较预先固定 ROI。
A 的统计为最多3个读取域的像素加权小样本估计；IBEX可用明确标注的同源不相交窗口，不代表独立图像或全cohort。A/B 的 F1 衡量方案一致性；来源 mask 为来源参考，非人工金标准。
CP/IS主表为whole-cell；SD_B及保存的IS核另列辅助一致性。未知mpp不猜测物理尺度，IS跳过，CP/SD保留原生像素面积。所有ROI边缘截断实例均纳入；触边比例指ROI外沿，不是内部tile边缘。P/R是方向性覆盖，F1是方案一致性；边界叠加仅视觉检查。数量更多不表示更准确；无显著性或生物学验证结论。点击图像放大、滚动查看。</p>'''
report += f'<p>当前 manifest 图像数：{len(samples)} / 12。</p><nav>{nav}</nav>'
report += '<h2>按技术描述性汇总</h2>' + html_table(summary_rows)
report += '<h2>输出与失败</h2>' + html_table(availability) + ''.join(sections)
report += '<script>document.querySelectorAll("img").forEach(i=>i.onclick=()=>i.classList.toggle("expanded"));</script></html>'
(out / 'report.html').write_text(report)
print(f'Comparison written: {out / "report.html"}')
