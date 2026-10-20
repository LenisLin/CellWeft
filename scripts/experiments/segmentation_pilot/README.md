# 12 源图预处理与分割探索

已完成5组合、60次真实预测。设计、独立复核、逐图结果及限制见[实验记录](../../../docs/experiments/segmentation_pilot_20261019.md)。本目录是隔离实验实现，`pilot_io.py`只在本实验内共享，不是正式训练数据API。

固定输入清单及结果：`/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/`。运行参数见 `configs/experiments/segmentation_pilot_20261019.json`。保持原始来源只读；重跑应使用新的输出目录，执行器拒绝覆盖已有组合。

在项目根目录执行，以下目录名为重跑示例：

```bash
PILOT_RUN=/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run02
PILOT_MANIFEST=/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/manifest.json
PILOT_SCRIPT=scripts/experiments/segmentation_pilot
PILOT_MODELS=/mnt/NAS_21T/ProjectData/CellWeft/resources/pretrained
export NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/mnt/NAS_21T/ProjectData/CellWeft/cache/matplotlib
scripts/resources/cellpose-sam.sh "$PILOT_SCRIPT/01_prepare.py" --manifest "$PILOT_MANIFEST" --output "$PILOT_RUN"
for scheme in CP_A CP_B; do
  scripts/resources/cellpose-sam.sh "$PILOT_SCRIPT/02_segment.py" --manifest "$PILOT_MANIFEST" --output "$PILOT_RUN" --scheme "$scheme" --weights "$PILOT_MODELS/cellpose/cpsam_v2/checkpoint"
done
for scheme in IS_A IS_B; do
  .venv-instanseg/bin/python "$PILOT_SCRIPT/02_segment.py" --manifest "$PILOT_MANIFEST" --output "$PILOT_RUN" --scheme "$scheme" --weights "$PILOT_MODELS/instanseg/fluorescence_nuclei_and_cells/0.1.1/instanseg.pt"
done
.venv-stardist/bin/python "$PILOT_SCRIPT/02_segment.py" --manifest "$PILOT_MANIFEST" --output "$PILOT_RUN" --scheme SD_B --weights "$PILOT_MODELS/stardist/2D_versatile_fluo"
scripts/resources/cellpose-sam.sh "$PILOT_SCRIPT/03_compare.py" --manifest "$PILOT_MANIFEST" --output "$PILOT_RUN"
```

`01_prepare.py`按固定ROI逐选定通道读取，拟合最多3个域，保存raw_selected（CYX）、A/B双通道（HWC，膜/核）、统计及运行状态。一般fit_scope为independent_files；IBEX显式same_source_disjoint_windows例外要求同文件/通道/Z，三个拟合域与目标及彼此不相交。所有分位数为读取域精确值，统计不是全cohort统计；非有限或负来源值报错，退化通道两路共同省略并记录。

`02_segment.py`每次只加载一个模型，逐图保存背景0的uint32二维mask。CP/IS使用GPU，SD只用CPU；IS按实际mpp适配0.5µm/px，未知mpp明确跳过。标签均恢复到同一原生ROI坐标；使用manifest的ROI原点对应来源图，不把label编号当跨图全局cell ID。推理失败记录于runtime并返回非零状态。

`03_compare.py`生成逐图及技术汇总CSV/JSON、固定全ROI/中心局部PNG、离线可放大HTML。IoU≥0.5最大基数一对一匹配使用稀疏重叠图；A/B与跨模型F1都表示一致性。SD及IS核辅助比较单列；无确认来源mask时不报告准确度。独立合成合同检查入口：`python scripts/experiments/segmentation_pilot/check_synthetic.py`。

## CP_B 蛋白定量

`04_quantify.py` 复用固定 CP_B mask，以原 ROI 的逐通道 P1/P99 图像计算实例内全部像素均值（含零值），不追加 arcsinh/缩放。配置为 `configs/experiments/protein_quantification_pilot_20261020.json`；用 `.venv-cellpose-sam/bin/python scripts/experiments/segmentation_pilot/04_quantify.py --config configs/experiments/protein_quantification_pilot_20261020.json` 执行，重跑前指定新的输出目录。该环境在沙箱中打开 Zarr 出现停滞，成功运行使用宿主执行环境。

[本轮报告](../../../docs/experiments/protein_quantification_pilot_20261020.md)记录 12 个 case、29,535 个实例，以及 51 个 P1/P99 范围未定义通道。缺失列为 NaN，不自动换阈值或填零；这不是完整 production/training-ready 数据。全部矩阵、通道统计与图件位于 NAS `results/protein_quantification_pilot_20261020/run02/`。

## 全部/非零像素 P99 比较

`05_compare_nonzero_p99.py` 固定原 CP_B mask，比较全部像素与正像素的 P99 除法归一化；两者均不扣 P1，细胞均值均包含零值。恢复 28 个蛋白家族/组合测量，共 458 个通道。执行命令：

```bash
.venv-cellpose-sam/bin/python scripts/experiments/segmentation_pilot/05_compare_nonzero_p99.py --config configs/experiments/nonzero_p99_pilot_20261020.json
```

重跑需先指定新输出目录；脚本拒绝覆盖。成功结果为 NAS `results/nonzero_p99_pilot_20261020/run02/`，含两套表达矩阵、统计与整 ROI 图件。[结果解释](../../../docs/experiments/nonzero_p99_pilot_20261020.md)说明 51 个非空通道恢复计算与信号可信度的区别；本轮没有重新分割或确认生产方案。
