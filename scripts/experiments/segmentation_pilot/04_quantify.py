"""Quantify protein-marker signals in the 12 frozen CP_B segmentation cases.

Inputs: reviewed channel manifest, original fixed ROIs and existing CP_B masks.
Outputs: per-case expression/cell Parquet, channel statistics, previews and report.
Each channel uses exact ROI P1/P99; all instance pixels contribute to its mean.
This is an isolated pilot, not the full-slide production preparation pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "8"
os.environ.setdefault("MPLCONFIGDIR", "/mnt/NAS_21T/ProjectData/CellWeft/cache/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot_io import labels, read_source, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
pilot = Path(config["pilot_root"])
output = Path(config["output"])
output.mkdir(parents=True, exist_ok=False)
write_json(output / "manifest.json", config)
start_all = time.perf_counter()
summaries, all_channels, checks, report_sections = [], [], [], []
original_counts = pd.read_csv(pilot / "comparison" / "cell_statistics.csv").query("scheme == 'CP_B'").set_index("sample_id")["count"]

# Reuse the selected masks; quantify channels serially to bound memory and NAS I/O.
for sample in config["samples"]:
    started = time.perf_counter()
    sid = sample["sample_id"]
    destination = output / sid
    destination.mkdir()
    mask_path = pilot / "masks" / config["mask_scheme"] / f"{sid}.npy"
    mask = labels(np.load(mask_path), (sample["roi"][3], sample["roi"][2]))
    flat = mask.ravel().astype(np.int64)
    counts = np.bincount(flat)
    ids = np.flatnonzero(counts)
    ids = ids[ids > 0]
    n_cells = len(ids)
    assert n_cells == original_counts[sid]
    x0, y0, width, height = sample["roi"]
    yy, xx = np.indices(mask.shape)
    x = np.bincount(flat, weights=xx.ravel(), minlength=len(counts))[ids] / counts[ids]
    y = np.bincount(flat, weights=yy.ravel(), minlength=len(counts))[ids] / counts[ids]
    edge_ids = np.unique(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]))
    cells = pd.DataFrame({"cell_index": np.arange(n_cells), "mask_label": ids,
                          "area_px": counts[ids], "centroid_x_roi": x, "centroid_y_roi": y,
                          "centroid_x_source": x + x0, "centroid_y_source": y + y0,
                          "touches_roi_border": np.isin(ids, edge_ids)})
    cells.to_parquet(destination / "cells.parquet", index=False)
    expression = cells[["cell_index", "mask_label"]].copy()
    selected = [c for c in sample["quantification_channels"] if c["include"]]
    source_names = [c["source_name"] for c in selected]
    assert len(source_names) == len(set(source_names)), f"{sid}: repeated expression column"
    # Fixed direct checks include sparse/nonconsecutive mask labels without relabelling.
    check_ids = ids[np.unique(np.linspace(0, n_cells - 1, min(3, n_cells), dtype=int))]
    check_pixels = [np.flatnonzero(flat == label) for label in check_ids]
    check_rows = np.searchsorted(ids, check_ids)
    old_stats = json.loads((pilot / "prepared" / sid / "stats.json").read_text())
    old_raw = np.load(pilot / "prepared" / sid / "raw_selected.npz")["image"]
    prior_positions = dict(zip(sample["source"]["indices"], range(len(sample["source"]["indices"]))))
    records, previews = [], []
    preferred = {"CD45", "CD45Rb", "CD45RO", "CD20", "CD20Ms", "Ki67", "Ki-67", "KI67", "KI67Rb", "B2M"}
    visual_indices = [c["source_index"] for c in selected
                      if c["marker_label"] in preferred or c["source_name"] in preferred][:3]
    read_seconds = normalize_seconds = aggregate_seconds = 0.0
    reused_channel_checks = direct_checks = 0
    max_direct_error = 0.0

    for channel in sample["quantification_channels"]:
        record = dict(sample_id=sid, **channel)
        if not channel["include"]:
            record["quantification_status"] = "excluded_by_marker_review"
            records.append(record)
            continue
        tick = time.perf_counter()
        source = dict(sample["source"], indices=[channel["source_index"]])
        if source["kind"] == "tiff_planes":
            source["paths"] = [channel["path"]]
        raw = read_source(source, sample["roi"])[0][0]
        read_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        q1, q99 = np.percentile(raw, [1, 99], method="linear")
        record.update(q1=float(q1), q99=float(q99), raw_min=float(raw.min()),
                      raw_max=float(raw.max()), raw_nonzero_fraction=float(np.mean(raw != 0)))
        if channel["source_index"] in prior_positions:
            pos = prior_positions[channel["source_index"]]
            np.testing.assert_array_equal(raw, old_raw[pos])
            np.testing.assert_allclose([q1, q99], [old_stats["q1"][pos], old_stats["q99"][pos]], rtol=0, atol=0)
            reused_channel_checks += 1
        name = channel["source_name"]
        if q99 <= q1:
            expression[name] = np.full(n_cells, np.nan, dtype=np.float32)
            record["quantification_status"] = "undefined_p1_p99_range"
            normalize_seconds += time.perf_counter() - tick
            records.append(record)
            continue
        normalized = np.clip((raw - q1) / (q99 - q1), 0, 1).astype(np.float32)
        normalize_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        sums = np.bincount(flat, weights=normalized.ravel(), minlength=len(counts))
        values = (sums[ids] / counts[ids]).astype(np.float32)
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        expected = np.array([normalized.ravel()[pixels].mean(dtype=np.float64) for pixels in check_pixels])
        np.testing.assert_allclose(values[check_rows], expected, rtol=1e-6, atol=1e-7)
        max_direct_error = max(max_direct_error, float(np.max(np.abs(values[check_rows] - expected))))
        direct_checks += len(check_ids)
        expression[name] = values
        aggregate_seconds += time.perf_counter() - tick
        quantiles = np.quantile(values, [0, .1, .5, .9, 1])
        record.update(quantification_status="quantified", cell_mean=float(values.mean()),
                      zero_cell_fraction=float(np.mean(values == 0)),
                      **dict(zip(["cell_min", "cell_p10", "cell_median", "cell_p90", "cell_max"], quantiles.tolist())))
        if channel["source_index"] in visual_indices:
            lookup = np.full(len(counts), np.nan, dtype=np.float32)
            lookup[ids] = values
            step = max(1, int(np.ceil(max(mask.shape) / 768)))
            previews.append((name, normalized[::step, ::step].copy(), lookup[mask[::step, ::step]]))
        records.append(record)

    # Save expression values and exact normalization/selection definitions together.
    expression.to_parquet(destination / "expression.parquet", index=False)
    channel_table = pd.DataFrame(records)
    channel_table.to_csv(destination / "channels.csv", index=False)
    reloaded = pd.read_parquet(destination / "expression.parquet")
    pd.testing.assert_frame_equal(expression, reloaded)
    np.testing.assert_array_equal(reloaded.mask_label.to_numpy(), ids)
    values = expression[source_names].to_numpy()
    valid_columns = channel_table.quantification_status.eq("quantified")
    undefined = channel_table.loc[channel_table.quantification_status.eq("undefined_p1_p99_range"), "source_name"].tolist()

    # One whole-ROI overview: pixel intensities above, instance means below.
    if previews:
        fig, axes = plt.subplots(2, len(previews), figsize=(4 * len(previews), 8), squeeze=False)
        for j, (name, pixels, means) in enumerate(previews):
            for i, (title, plane) in enumerate([("Normalized pixels", pixels), ("Cell mean", means)]):
                artist = axes[i, j].imshow(plane, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
                axes[i, j].set_title(f"{name} | {title}", fontsize=10)
                axes[i, j].axis("off")
        fig.suptitle(f"{sid} | CP_B | same fixed ROI | scale 0-1", fontsize=12)
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.65, label="Normalized signal")
        fig.savefig(destination / "overview.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    summary = dict(sample_id=sid, technology=sample["technology"], cells=n_cells,
                   source_channels=len(records), selected_channels=len(selected),
                   quantified_channels=int(valid_columns.sum()), undefined_channels=len(undefined),
                   excluded_channels=len(records)-len(selected), roi_width=width, roi_height=height,
                   border_cells=int(cells.touches_roi_border.sum()),
                   finite_values=int(np.isfinite(values).sum()), nan_values=int(np.isnan(values).sum()),
                   expression_min=float(np.nanmin(values)), expression_max=float(np.nanmax(values)),
                   read_seconds=read_seconds, normalization_seconds=normalize_seconds,
                   aggregation_and_checks_seconds=aggregate_seconds,
                   total_seconds=time.perf_counter()-started)
    summaries.append(summary)
    all_channels.extend(records)
    checks.append(dict(sample_id=sid, mask_shape=list(mask.shape), mask_label_rows_match=True,
                       original_cp_b_count_match=True,
                       prior_selected_channels_equal=reused_channel_checks, direct_means_checked=direct_checks,
                       direct_max_absolute_error=max_direct_error, parquet_roundtrip=True,
                       undefined_channels=undefined))
    record_columns = ["source_name", "quantification_status", "q1", "q99", "cell_median", "cell_p90", "zero_cell_fraction", "exclusion_reason"]
    report_sections.append(f'<section id="{sid}"><h2>{sid}</h2><p>{n_cells:,} cells × {len(selected)} selected channels; '
                           f'{len(undefined)} undefined channels. ROI: {sample["roi"]}; border cells: {summary["border_cells"]}.</p>'
                           f'<p><a href="{sid}/expression.parquet">Expression Parquet</a> · '
                           f'<a href="{sid}/cells.parquet">Cell coordinates</a> · '
                           f'<a href="{sid}/channels.csv">Channels and normalization</a></p>'
                           f'<img src="{sid}/overview.png" alt="Whole ROI pixel intensity and cell mean" loading="lazy">'
                           f'<details><summary>Channel results and exclusions</summary>{channel_table[record_columns].to_html(index=False, float_format=lambda x: f"{x:.5g}")}</details></section>')
    print(f'{sid}: {n_cells} cells, {int(valid_columns.sum())}/{len(selected)} channels; undefined={undefined}; {summary["total_seconds"]:.1f}s', flush=True)

summary_table = pd.DataFrame(summaries)
summary_table.to_csv(output / "summary.csv", index=False)
pd.DataFrame(all_channels).to_csv(output / "channel_statistics.csv", index=False)
write_json(output / "verification.json", checks)
runtime = dict(elapsed_seconds=time.perf_counter()-start_all,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               cpu_thread_limit=8, gpu_used=False, mask_scheme=config["mask_scheme"],
               cases=len(summaries), total_cells=int(summary_table.cells.sum()),
               quantified_case_channels=int(summary_table.quantified_channels.sum()),
               undefined_case_channels=int(summary_table.undefined_channels.sum()),
               finite_values=int(summary_table.finite_values.sum()), nan_values=int(summary_table.nan_values.sum()))
write_json(output / "runtime.json", runtime)
intro = ('<h1>CP_B 蛋白通道定量：12 个固定 case</h1>'
         '<p>逐通道固定 ROI P1/P99 → [0,1] → mask 内全部像素算术均值，包含零值；无 arcsinh 或其他缩放。'
         '复用原 CP_B mask，未重新分割。大图为原试验 ROI，分位数并非整片统计。边缘实例保留，定量范围仅为 ROI 内可见部分。</p>'
         '<p>排除非蛋白、重复及本轮未解决的混合靶标通道。列为来源 marker 测量，尚未完成统一 protein bank 身份映射。'
         'P99≤P1 的通道保留为 NaN，不代表零表达。Nirmal 原分割已观察到部分范围贴近细胞核；本轮沿用，未进行生物学准确度验证。</p>'
         '<p><a href="summary.csv">逐图汇总 CSV</a> · <a href="channel_statistics.csv">全部通道统计 CSV</a> · '
         '<a href="manifest.json">输入与方法</a> · <a href="verification.json">数值核验</a></p>')
undefined_table = pd.DataFrame(all_channels).query("quantification_status == 'undefined_p1_p99_range'")
undefined_table = undefined_table.copy()
undefined_table["nonzero_pixel_percent"] = undefined_table.raw_nonzero_fraction * 100
undefined_table[["sample_id", "source_name", "q1", "q99", "raw_min", "raw_max", "nonzero_pixel_percent"]].to_csv(output / "undefined_channels.csv", index=False)
sparse_count = int((undefined_table.raw_max > 0).sum())
zero_count = int((undefined_table.raw_max == 0).sum())
intro += (f'<p><strong>实际结果：{runtime["total_cells"]:,} 个实例，'
          f'{runtime["quantified_case_channels"]}/{int(summary_table.selected_channels.sum())} 个 case×通道可定量；'
          f'{runtime["undefined_case_channels"]} 个通道归一化未定义（{sparse_count} 个仍有非零信号，{zero_count} 个全零）。'
          '缺失列保存 NaN，当前结果不是完整 training-ready 矩阵。</strong></p>')
intro += '<p><a href="undefined_channels.csv">归一化未定义通道的具体数值</a>：当 P1=P99 时分母为零。原图仍有非零信号的通道不能按零表达处理。</p>'
summary_html = summary_table[["sample_id", "technology", "cells", "selected_channels", "quantified_channels", "undefined_channels", "expression_min", "expression_max"]].to_html(index=False, float_format=lambda x: f"{x:.5g}")
page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>CellWeft protein quantification pilot</title><style>body{font:16px system-ui;margin:30px;max-width:1400px;color:#172638}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}section{margin-top:45px}p{line-height:1.7}details{overflow:auto}</style>'
page += intro + summary_html + ''.join(report_sections) + '</html>'
(output / "report.html").write_text(page)
print(json.dumps(runtime), flush=True)
