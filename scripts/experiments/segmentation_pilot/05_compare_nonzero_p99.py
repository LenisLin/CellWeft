"""Compare all-pixel and positive-pixel P99 scaling on fixed CP_B instances.

Inputs: corrected protein-channel manifest, original pilot ROIs and CP_B masks.
Outputs: two expression tables per case, spatial diagnostics and whole-ROI figures.
Both methods divide without subtracting P1; zeros remain in every cell mean.
Spatial diagnostics describe signal support, not ground-truth signal/noise labels.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import time

for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "8"
os.environ.setdefault("MPLCONFIGDIR", "/mnt/NAS_21T/ProjectData/CellWeft/cache/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.stats import spearmanr
from skimage.measure import block_reduce
from pilot_io import read_source, labels, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
output = Path(config["output"])
output.mkdir(parents=True, exist_ok=False)
write_json(output / "manifest.json", config)
pilot = Path(config["pilot_root"])
previous = Path(config["previous_quantification_root"])
old_stats = pd.read_csv(previous / "channel_statistics.csv")
summary, channel_records, checks, sections = [], [], [], []
started_all = time.perf_counter()
kernel = np.ones((3, 3), np.uint8)
kernel[1, 1] = 0

for sample in config["samples"]:
    start = time.perf_counter()
    sid = sample["sample_id"]
    target = output / sid
    target.mkdir()
    mask = labels(np.load(pilot / "masks/CP_B" / f"{sid}.npy"),
                  (sample["roi"][3], sample["roi"][2]))
    flat = mask.ravel().astype(np.int64)
    counts = np.bincount(flat)
    ids = np.flatnonzero(counts)
    ids = ids[ids > 0]
    cells = pd.read_parquet(previous / sid / "cells.parquet")
    np.testing.assert_array_equal(cells.mask_label, ids)
    cells.to_parquet(target / "cells.parquet", index=False)
    tables = {method: cells[["cell_index", "mask_label"]].copy()
              for method in ["all_p99", "nonzero_p99"]}
    old_expression = pd.read_parquet(previous / sid / "expression.parquet")
    selected = [c for c in sample["quantification_channels"] if c["include"]]
    old_case = old_stats[old_stats.sample_id.eq(sid)]
    # Overview selection is reproducible: sparsest/densest previously undefined
    # nonempty channels, then a restored marker; common markers fill spare slots.
    sparse = old_case[old_case.quantification_status.eq("undefined_p1_p99_range") & (old_case.raw_max > 0)].sort_values("raw_nonzero_fraction")
    visual_names = []
    if len(sparse):
        visual_names += [sparse.iloc[0].source_name, sparse.iloc[-1].source_name]
    visual_names += [c["source_name"] for c in selected if c["restored_protein_measurement"]][:1]
    visual_names += [c["source_name"] for c in selected if c["marker_label"] in {"CD45", "CD20", "Ki67", "B2M", "CD3"}]
    visual_names = list(dict.fromkeys(visual_names))[:3]
    check_rows = np.unique(np.linspace(0, len(ids)-1, 3, dtype=int))
    check_pixels = [np.flatnonzero(flat == ids[r]) for r in check_rows]
    check_count, max_error = 0, 0.0
    records, figures = [], []
    for channel in selected:
        name = channel["source_name"]
        source = dict(sample["source"], indices=[channel["source_index"]])
        if source["kind"] == "tiff_planes":
            source["paths"] = [channel["path"]]
        raw = read_source(source, sample["roi"])[0][0]
        positive = raw > 0
        positive_values = raw[positive]
        q1, q99, q998 = np.percentile(raw, [1, 99, 99.8], method="linear")
        # Vector quantiles retain float64 interpolation, matching the all-pixel call.
        qp = float(np.percentile(positive_values, [99], method="linear")[0]) if len(positive_values) else 0.0
        assert qp >= q99
        rec = dict(sample_id=sid, technology=sample["technology"], **channel,
                   positive_pixels=int(positive.sum()), positive_fraction=float(positive.mean()),
                   raw_max=float(raw.max()), q1=float(q1), q99_all=float(q99), q998_all=float(q998),
                   q99_nonzero=qp, scale_ratio_nonzero_to_all=qp/q99 if q99 > 0 else None,
                   all_zero=not bool(len(positive_values)), recovered=bool(q99 == 0 and qp > 0))
        if len(positive_values):
            neighbor_counts = convolve(positive.astype(np.uint8), kernel, mode="constant", cval=0)
            rec["isolated_positive_fraction"] = float(np.mean(neighbor_counts[positive] == 0))
            rec["positive_inside_mask_fraction"] = float(np.mean(mask[positive] > 0))
        planes, values_by_method = {}, {}
        for method, scale in [("all_p99", q99), ("nonzero_p99", qp)]:
            if scale == 0 and len(positive_values):
                tables[method][name] = np.full(len(ids), np.nan, np.float32)
                continue
            plane = np.clip(raw/scale, 0, 1).astype(np.float32) if scale > 0 else np.zeros_like(raw)
            means = (np.bincount(flat, weights=plane.ravel(), minlength=len(counts))[ids] / counts[ids]).astype(np.float32)
            assert np.isfinite(means).all() and means.min() >= 0 and means.max() <= 1
            direct = np.array([plane.ravel()[pixels].mean(dtype=np.float64) for pixels in check_pixels])
            np.testing.assert_allclose(means[check_rows], direct, atol=1e-7, rtol=1e-6)
            max_error = max(max_error, float(np.max(np.abs(means[check_rows]-direct))))
            check_count += len(check_rows)
            tables[method][name] = means
            values_by_method[method] = means
            rec.update({f"{method}_cell_mean": float(means.mean()),
                        f"{method}_cell_p50": float(np.median(means)),
                        f"{method}_cell_p99": float(np.percentile(means,99)),
                        f"{method}_nonzero_cells": int(np.count_nonzero(means)),
                        f"{method}_pixel_saturated_fraction": float(np.mean(plane == 1)),
                        f"{method}_mask_inside_mean": float(plane[mask>0].mean()),
                        f"{method}_mask_outside_mean": float(plane[mask==0].mean()) if np.any(mask==0) else None})
            if name in visual_names:
                planes[method] = plane
        if q99 > 0:
            a, b = values_by_method["all_p99"], values_by_method["nonzero_p99"]
            assert np.all(b <= a + 1e-7)
            np.testing.assert_array_equal(a>0, b>0)
            rec["cell_mean_ratio_nonzero_to_all"] = float(b.mean()/a.mean()) if a.mean()>0 else None
            rec["cell_spearman"] = float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else None
        if name in old_expression:
            old = old_expression[name].to_numpy()
            if np.isfinite(old).all():
                rec["cell_mean_change_from_old_p1p99"] = float(values_by_method["nonzero_p99"].mean()-old.mean())
                if q1 == 0:
                    np.testing.assert_allclose(tables["all_p99"][name], old, rtol=1e-6, atol=1e-7)
        records.append(rec)

        if name in visual_names:
            # Max pooling preserves sparse foci in the display; quantification is native-pixel.
            step = max(1, int(np.ceil(max(mask.shape)/768)))
            source_display = np.log1p(raw)/np.log1p(raw.max()) if raw.max()>0 else raw
            lut = np.zeros(len(counts), np.float32)
            lut[ids] = values_by_method["nonzero_p99"]
            displays = [("Source: log1p / log1p(max)\nDISPLAY ONLY",source_display),
                        ("All-pixel P99",planes.get("all_p99")),
                        ("Nonzero-pixel P99",planes["nonzero_p99"]),
                        ("Cell mean: nonzero P99",lut[mask])]
            fig,axes = plt.subplots(1,4,figsize=(16,4.8))
            for ax,(title,plane) in zip(axes,displays):
                if plane is None:
                    ax.text(.5,.5,"Undefined: P99 = 0",ha="center",va="center",transform=ax.transAxes)
                else:
                    reduced = block_reduce(plane,(step,step),np.max)
                    ax.imshow(reduced,cmap="viridis",vmin=0,vmax=1,interpolation="nearest")
                ax.set_title(title,fontsize=10);ax.axis("off")
            fig.suptitle(f'{sid} | {name} | nonzero={positive.mean()*100:.4g}% | Q99 all={q99:.5g}, nonzero={qp:.5g}\n'
                         f'Fixed whole ROI; display max pooling {step}x{step}; normalized panels use 0-1',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.88))
            filename=f'channel_{channel["source_index"]:03d}.png'
            fig.savefig(target/filename,dpi=130,bbox_inches="tight");plt.close(fig)
            figures.append((name,filename))
    for method,table in tables.items():
        table.to_parquet(target/f"expression_{method}.parquet",index=False)
        pd.testing.assert_frame_equal(table,pd.read_parquet(target/f"expression_{method}.parquet"))
    frame=pd.DataFrame(records)
    frame.to_csv(target/"channels.csv",index=False)
    row=dict(sample_id=sid,technology=sample["technology"],cells=len(ids),channels=len(selected),
             restored_channels=int(frame.restored_protein_measurement.sum()),
             all_p99_undefined=int(frame.recovered.sum()),nonzero_p99_undefined=0,
             all_zero_channels=int(frame.all_zero.sum()),elapsed_seconds=time.perf_counter()-start)
    summary.append(row);channel_records.extend(records)
    checks.append(dict(sample_id=sid,cell_rows_match=True,direct_checks=check_count,
                       max_absolute_error=max_error,nonzero_method_never_exceeds_valid_all_method=True,
                       parquet_roundtrip=True))
    image_html=''.join(f'<h3>{name}</h3><a href="{sid}/{filename}"><img src="{sid}/{filename}" loading="lazy"></a>' for name,filename in figures)
    sections.append(f'<section><h2>{sid}</h2><p>{len(ids):,} cells × {len(selected)} channels. '
                    f'<a href="{sid}/expression_nonzero_p99.parquet">Nonzero-P99 expression</a> · '
                    f'<a href="{sid}/expression_all_p99.parquet">All-P99 expression</a> · '
                    f'<a href="{sid}/channels.csv">Channel statistics</a></p>{image_html}</section>')
    print(f'{sid}: {row}',flush=True)

summary_table=pd.DataFrame(summary)
channel_table=pd.DataFrame(channel_records)
summary_table.to_csv(output/"summary.csv",index=False)
channel_table.to_csv(output/"channel_comparison.csv",index=False)
channel_table[channel_table.recovered].to_csv(output/"recovered_channels.csv",index=False)
write_json(output/"verification.json",checks)
runtime=dict(elapsed_seconds=time.perf_counter()-started_all,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             gpu_used=False,cpu_thread_limit=8,cases=len(summary),cells=int(summary_table.cells.sum()),
             channels=int(summary_table.channels.sum()),restored_channels=int(summary_table.restored_channels.sum()),
             recovered_channels=int(summary_table.all_p99_undefined.sum()),all_zero_channels=int(summary_table.all_zero_channels.sum()))
write_json(output/"runtime.json",runtime)
(output/"code").mkdir()
for path in [Path(__file__),Path(__file__).with_name("pilot_io.py")]:shutil.copy2(path,output/"code"/path.name)
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Nonzero P99 comparison</title>
<style>body{font:16px system-ui;margin:30px;max-width:1500px;color:#172638}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}img{width:100%}section{margin-top:50px}p{line-height:1.7}</style>
<h1>非零像素 P99：12 个固定 CP_B case</h1>
<p>全部像素 P99 与非零像素 P99 均仅作除法及 [0,1] 截断，不扣 P1。mask、ROI、Z 层和细胞均值定义固定；均值包含零值。全零通道保留为零。恢复此前误排除的蛋白家族/组合测量。</p>
<p>每行图依次为来源图像的 log 显示、全部像素 P99、非零像素 P99、后者的实例均值。来源图的 log 仅供观察，未用于定量；其余统一 0–1 色标。缩略图按标注倍率作 max pooling，计算保留原生像素。</p>
<p>选图规则：此前未定义通道中非零像素最少与最多的各一项、一个恢复通道，空余用常见 marker 补齐。所有通道数值均在 CSV。无阴性对照/人工真值，mask 外也可能含真实蛋白信号；孤立像素比例只作空间诊断，不能当作噪声率。</p>
<p><a href="channel_comparison.csv">全部通道比较</a> · <a href="recovered_channels.csv">恢复计算的稀疏通道</a> · <a href="manifest.json">配置</a> · <a href="verification.json">核验</a></p>'''
page+=summary_table.to_html(index=False,float_format=lambda x:f'{x:.3g}')+''.join(sections)+'</html>'
(output/"report.html").write_text(page)
print(json.dumps(runtime),flush=True)
