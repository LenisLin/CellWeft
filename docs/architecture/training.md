# 训练目标与生命周期

结构见[model.md](model.md)，输入见[data_contract.md](data_contract.md)，遮盖/删除模式见[editing.md](editing.md)。本文维护正式训练约束，不表示训练已实现。

2026-10-19 图像基座与输入规格以[模型规范](model.md)开头的最新决定及[数据准备](../data/preparation.md)为准。本页 §1 基于旧 DINOv3 的初始化、crop 尺寸和图像 recipe 表述待独立 DINOv2 迁移统一更新，不作为当前图像准备或训练配置依据。

## 1. 图像独立预训练

接入官方 DINOv3 工程，锁定 commit、环境、基础配置并保存覆盖差异；使用 B 架构从头进行领域预训练，不把官方发布权重的训练历史当成本项目训练历史。
明确覆盖：ViT-B/16、global crop 256、protein-aware PatchEmbed、数据/metadata、领域适用增强及硬件 batch 设置。
原包记录的 2026-10-17 基础 YAML 为 global 224、local 96、8 local views；DINO/iBOT 两 heads 各 65,536 prototypes、hidden 2,048、bottleneck 256，Gram=false。实际接入时核对锁定版本，差异需要显式记录和决策。
保留所选官方 local views、heads、teacher EMA、调度与 DINO/iBOT；Gram 由明确 recipe 控制，不要求训练 7B teacher、权重蒸馏或全部高分辨率后训练。
增强与几何同步规则见[数据准备](../data/preparation.md)。iBOT 在位置内 marker fusion 后、跨空间主干前执行。
仅显式 smoke 配置可缩减测试尺寸；不能作为正式默认配置。

## 2. 细胞独立预训练

Intrinsic 和 Neighborhood 均参与预训练。L_cell = L_marker + lambda_context * L_cell_latent。
Marker loss 保留 identity、遮盖 value，仅监督实际测量 targets；至少一条 decoder 路径经过 pooled intrinsic embedding。

1. 先划分 visible/target cells；在所有 neighborhoods、overlapping context 和聚合统计路径移除 student targets。
2. Student 只编码可见细胞；EMA teacher 编码完整邻域并提供 masked cells 的 contextual latent，stop-gradient。
3. Predictor 使用可见 context、目标 location/rank query 和可选有限 marker hints。
4. Hints 仅进 predictor；完整 target profile 或 teacher latent 不作为 student 输入。
5. Latent loss 在 normalized MSE/cosine 中配置一种默认目标；不堆叠多个目标，不增加整条表达谱恢复 decoder。

监测均值、std、norm 与 teacher/student 方差，零向量塌缩导致的低 loss 不代表成功。

## 3. Joint 冻结与梯度

| 模块 | 初始策略 |
| --- | --- |
| Protein bank、image PatchEmbed | 固定 |
| Image ViT、Intrinsic、Neighborhood | 固定；必要时按明确配置只解冻末端 |
| Fusion、masked-cell predictor、alignment/projection heads | 训练 |

Neighborhood 已独立预训练；冻结后仍对变化输入重新前向。
L_joint = L_joint_cell_latent + lambda_align * L_alignment。
Fusion 的主要梯度来自融合后的 masked-cell latent prediction，两方向 cross-attention 均需有通向 loss 的梯度。
仅使用 ST-Align-inspired Neighborhood–Region 欧氏投影、归一化和 multi-positive 对比目标。相同 region 的多个 neighborhoods 可为正样本；cross-rank negatives 按明确 patient/crop/overlap 规则组织。
若 alignment 位于融合前且基座全冻结，它仅更新 projection heads，不能声称单独训练 Fusion。不叠加 HyperST/hyperbolic/entailment/hierarchy 目标。

## 4. Joint 双视图与 EMA

1. 读取同一区域的空间一致 image/cell 输入；student 同步遮盖目标 cells 后再运行基座和 Fusion。
2. Teacher 读取完整视图，通过基座与 EMA Fusion 提取对应 cell latent，停止梯度。
3. Predictor 在目标位置学习该 latent；联合阶段不每步重复完整 DINO multi-crop，也不必每步计算冻结基座原生 losses。
4. 冻结基座可共享权重对象，full/edited 输入分别 forward；不维护未使用的大型副本或 DINO heads/optimizer states。
5. EMA 覆盖 trainable Fusion 和后续解冻的基座部分；Fusion 用预训练表示 residual 路径初始化。
6. 监测 latent variance/norm、teacher/student dispersion 和梯度；不自动增加新损失。

## 5. Autograd 与模式

requires_grad=False 与 eval() 分别管理，顶层 train() 不得改变冻结基座的预期状态。
冻结基座仅作前端时可用 no_grad，普通输出张量进入可训练 Fusion。若可训练输入继续经过冻结 block，需保留对输入的梯度。
不默认用 inference_mode 生成后续 backward 需要保存的张量；若使用，转换为普通 tensor 并验证。
末端解冻保留对应 autograd，优化器只纳入指定参数。每阶段保存冻结表和实际 trainable parameter report。

## 6. 未锁定训练参数

Marker mask / cell target / hint ratio 均配置化；0.3 / 0.5 / 0.1 仅为 smoke 候选，不是正式确认值。
Hint 的具体实现与消融需明确记录；loss 权重、negative 规则、末端解冻时机与正式 batch 在实施和实验配置中明确。本次文档拆分不代替这些研究决策。
恢复、资源统计和工程验收见[工程规范](../development/engineering.md)。
