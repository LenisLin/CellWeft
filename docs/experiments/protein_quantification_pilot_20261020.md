# CP_B 固定区域蛋白定量实测

2026-10-20。复用分割 pilot 的 12 份 CP_B（Cellpose-SAM cpsam_v2）实例 mask，覆盖 **29,535 个实例**。纳入 430 个 case×蛋白 marker 通道，379 个可按当前归一化规则定量，51 个未定义；6 个 case 的纳入通道齐全、6 个存在缺口。未重新分割，未修改来源文件。

[完整图文报告](/mnt/NAS_21T/ProjectData/CellWeft/results/protein_quantification_pilot_20261020/run02/report.html) · [逐图汇总](/mnt/NAS_21T/ProjectData/CellWeft/results/protein_quantification_pilot_20261020/run02/summary.csv) · [全部通道统计](/mnt/NAS_21T/ProjectData/CellWeft/results/protein_quantification_pilot_20261020/run02/channel_statistics.csv) · [51 个未定义通道](/mnt/NAS_21T/ProjectData/CellWeft/results/protein_quantification_pilot_20261020/run02/undefined_channels.csv) · [输入清单](/mnt/NAS_21T/ProjectData/CellWeft/results/protein_quantification_pilot_20261020/run02/manifest.json) · [数值核验](/mnt/NAS_21T/ProjectData/CellWeft/results/protein_quantification_pilot_20261020/run02/verification.json)

## 方法与范围

- 全部沿用原试验固定来源、原生 ROI、Z 层（IBEX 为 Z=1）和 CP_B mask；小图全图、大图为冻结 ROI，未扩展到完整切片。
- 原试验仅保存分割选定通道。本轮逐通道读取同一 ROI，按全 ROI 像素（含背景）计算精确 linear P1/P99，生成与 B 路一致的逐通道 [0,1] 表示。31 个重叠通道的来源像素和分位数与旧准备记录完全一致。大图的 ROI 分位数不是正式整片统计。
- 每个正整数 mask label 内，按全部实例像素求逐通道算术均值，包含零值；背景 0 不建行。不追加 arcsinh、z-score 或其他缩放。每个 ROI 实例计算一次。
- 527 个来源通道中，97 个按本轮清单排除：非蛋白核染/元素/背景/RNA、重复及未解决的混合靶标。列保留来源 marker 名称和通道位置；尚未完成全套 canonical protein、measured form 和 protein bank 映射。这是通道级定量实验，不作为完整 training-ready 数据。
- 触及 ROI 边界的实例保留，标记 `touches_roi_border`；其均值只覆盖 ROI 内可见像素。Nirmal 的部分 CP_B 实例范围贴近核，沿用先前分割报告的限制，本轮不新增逐细胞审查或生物学准确性结论。

## 逐图结果

|Case|实例数|纳入通道|有效定量通道|未定义通道|
|---|---:|---:|---:|---:|
|Danenberg2022_imc|1,360|31|31|0|
|Hoch2022_imc|6,480|26|26|0|
|Jackson2020_imc|864|30|30|0|
|Schurch2020_codex|2,349|55|54|1|
|Monkman2024_codex|2,973|35|35|0|
|Phillips2021_codex|2,286|53|52|1|
|Risom2022_mibi|679|37|18|19|
|Azulay2025_mibi|1,537|33|20|13|
|Liu2023_mibi|3,969|36|20|16|
|Ly2025_comet|542|25|25|0|
|Nirmal2022_cycif|2,400|30|30|0|
|Radtke2024_ibex|4,096|39|38|1|

## P1/P99 未定义的实际含义

这 51 个通道都满足 P1=P99=0，因此 `(x-P1)/(P99-P1)` 的分母为零。50 个通道仍有非零信号，非零像素占所选 ROI 的约 0.00334%–0.99545%；另 1 个为 Azulay PD1，在本 ROI 内全零。51 是 case×通道数，不是不同蛋白种类数。

例如 Schurch CD20 只有约 0.275% 像素非零，最大值 49,194，但全 ROI 的 P99 仍为 0。不能将该通道判作零表达。本轮保留其表达列为 NaN，记录来源最大值与非零比例，未改成非零像素分位数、提高百分位或重新缩放。问题出现在共享归一化阶段，不是细胞内均值算法。

按技术：MIBI 48 个、CODEX 2 个、IBEX 1 个；具体名单与数值见上方 CSV。稀疏通道归一化规则是下一步需确认的生产边界。

## 数值验收与资源

- 12 个矩阵的实例数全部与原 CP_B 报告一致；表达行与原 mask_label 一致，Parquet 写回重读一致。
- 379 个有效通道共有 949,873 个有限表达值，全部位于 [0,1]；51 个未定义通道产生 105,117 个 NaN 单元格，未填零。
- 对每个有效通道的 3 个固定实例作独立像素切片均值检查，共 1,137 项，最大绝对误差 2.96×10⁻⁸（float32 保存误差）。
- 成功运行耗时 115.27 秒，CPU 线程上限 8、单路 NAS 读取，峰值 RSS 约 2.46 GiB，未使用 GPU。包括读取、归一化、汇总、数值检查、表格与图件写入；不是全队列吞吐基准。
- 首次沙箱运行停在 Zarr 数组打开，SIGINT 中断且无表达输出，保留在 run01。宿主有界读取检查通过后，run02 完成；上述耗时仅指成功运行。

## 产物与重现

每个 case 保存 `expression.parquet`（cell_index、mask_label、marker 列，float32），`cells.parquet`（实例像素数、ROI/来源 centroid、触边标记），`channels.csv`（选择依据、来源位置、P1/P99、缺口和表达分布），以及整 ROI 像素强度/实例均值对照图。背景在实例均值图中留白，所有强度图使用统一 0–1 色标。

实现为 [04_quantify.py](/home/lenislin/Experiment/projects/CellWeft/scripts/experiments/segmentation_pilot/04_quantify.py)，参数为 [实验配置](/home/lenislin/Experiment/projects/CellWeft/configs/experiments/protein_quantification_pilot_20261020.json)。使用项目 `.venv-cellpose-sam/bin/python` 执行，传 `--config`；重跑须指定新输出目录。生产文件格式和分块布局仍待确定，本轮 Parquet 是隔离实验产物格式。
