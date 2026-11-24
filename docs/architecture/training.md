# 预训练工作流、目标与梯度

本页维护预训练流程、目标和首版资源配置。结构见[模型规范](model.md)，字段与几何见[数据合同](data_contract.md)，推理编辑见[编辑规范](editing.md)，实施进度见[当前状态](../project/status.md)。

预训练联合 DINO、iBOT、whole-cell 表达 MSE 和 DINOv2 配套 KoLeo。像素 decoder 按 §5 另行训练。

## 1. 数据准备与训练流程

1. 读取完整 image 上生成的图像、实例、whole-cell 表及 study MAP 资产。按[准备规范§7](../data/preparation.md#7-crop-与训练视图)采样基础窗口，按[数据合同§4](data_contract.md#4-空间对应)组装和变换空间字段。
2. 按 §2 构建 teacher 和 student 多视图，按 §3 计算 DINO、iBOT 与 KoLeo。复用未施加 iBOT 遮盖的 student global view 做细胞监督。若没有可复用视图，则另做一次完整 global view 的 student 前向。
3. 按[数据合同§4](data_contract.md#4-空间对应)选择视图中的实例，以其完整 image 的离线 whole-cell 均值为目标。记录截断、无可见像素及可读细胞覆盖。
4. 按 §3 归约加权目标并反向传播。每次累积完成后执行一次 optimizer step，并按 §4 更新 EMA 与调度。
5. 按 §4 保存训练状态并导出配套 EMA。下游读出见[表征评价](../evaluation/representation.md)。

## 2. DINOv2 多视图与遮盖

采用 DINOv2 的 global／local 多视图组织和 iBOT 遮盖方式。本地参考默认每个基础样本生成两个 224×224 global views 和八个 96×96 local views；global 采样面积比例为 0.32–1.0，local 为 0.05–0.32。以上作为首版参考配置，训练配置记录实际尺寸、数量、缩放范围和随机种子。8×8 patch 时 global view 是 28×28 网格、784 个 patch，local view 是 12×12 网格、144 个 patch；每个 view 均有 CLS。DINO 的随机裁剪／缩放属于训练视图变换，不改变原始 image 或基础 crop 坐标。

Teacher 读取两个完整 global views；student 读取 global 和 local views。DINO 使用 teacher global CLS 与不同 student view 的 CLS，排除相同 global view 的自身配对；iBOT 在同一 global view 的空间位置比较 teacher 和 student patch 输出。配置、view 数和 loss 归约参照 DINOv2 实现。

iBOT 在部分 student global views 中对空间 patch token 采用块状遮盖，teacher 对同一 global view 保持完整；遮盖仅作用 student 视图，不修改原始图像、实例表或 whole-cell 真值。本地 DINOv2 默认 `mask_sample_probability=0.5`、`mask_ratio_min_max=[0.1,0.5]` 可作为首版起点。计算 iBOT 时只纳入被遮盖且对应有效实测域的 patch。图像 padding 保持为空间无效，不把 padding 当作训练真值。

图像、实例 mask、centroid 和 spatial-valid 随视图使用同一裁剪、缩放、旋转或翻转；坐标与有效性变换必须可追溯。蛋白图像采用适用于多通道数值图像的数据增强和归一化，不将 RGB 色相、饱和度、灰度及太阳化操作直接套到蛋白通道。DINO 的视图几何实现与具体预处理服从[数据准备](../data/preparation.md)和[数据合同](data_contract.md)。

Protein-aware tokenizer 在 patch 内汇总 marker；iBOT 对汇总后的空间 token 做遮盖。蛋白重建阶段的逐 marker 局部遮盖必须在 tokenizer 汇总 marker 之前作用于对应图像通道和观测状态，隐藏值不能进入空间 token。Figure 2 的局部／整通道遮盖与 Figure 3 的全 panel 同位置遮盖均在原 ROI 坐标生成，再提取 crop，见§5。

## 3. 联合训练目标

\[
\mathcal L=\mathcal L_{\mathrm{DINO}}+\lambda_P\mathcal L_{\mathrm{iBOT}}+\lambda_C\mathcal L_{\mathrm{cell}}+\lambda_K\mathcal L_{\mathrm{KoLeo}}.
\]

| 目标 | 监督与输入 | 更新的 student 模块 |
| --- | --- | --- |
| DINO | teacher global CLS 对其他 global／local student CLS 的分布蒸馏 | protein-aware tokenizer、ViT、DINO projection head |
| iBOT | 同一 global view 中被遮盖有效 patch 的 teacher／student 原型分布蒸馏 | tokenizer、ViT、iBOT projection head |
| Whole-cell MSE | 未施加 iBOT 遮盖的 student global view 上，实例读出预测 whole-cell marker 值 | tokenizer、ViT、cell adapter、表达预测头及可训练 marker 投影 |
| KoLeo | student global CLS 的表示分布正则 | tokenizer、ViT |

DINO／iBOT 沿用 DINOv2 的 teacher centering、温度和 cross-entropy 形式，使用独立的 DINO／iBOT projection heads；DINOv2 的 ViT-L/G 正式训练配置同样启用 `ibot.separate_head=true`。CLS 与 patch targets 按对应位置分别构造和归约。权重、温度与其他配置按实际实验记录；不将尚未运行的数值当作性能结论。

设第 b 个未遮盖 student global view 的细胞集合为 \(C_b\)，细胞 i 的有效实测 marker 集合为 \(S_{bi}\)，预测和离线 whole-cell 定量值分别为 \(\widehat x_{bim}\) 与 \(x_{im}\)：

\[
\mathcal L_{\mathrm{cell}}=
\operatorname{mean}_{b}\operatorname{mean}_{i\in C_b}
\operatorname{mean}_{m\in S_{bi}}(\widehat x_{bim}-x_{im})^2.
\]

先对每个细胞的有效 marker、再对 view 内有目标的细胞、最后对有目标的 views 求均值；真实零表达参与。没有有效细胞或 marker 的 view 不纳入该项均值，但仍可参与图像自监督。该预测目标检验模型输出与离线 whole-cell 定量的一致性。一个 view 编码一次，多个实例共享图像 features 并可分块通过 adapter；mask 是固定输入，梯度更新参数，不更新分割轮廓。训练重叠视图中同一细胞可出现多次；验证／测试按 centroid 归属固定不重叠 crop。

训练 loss 权重按各项数值、梯度及实际实验调整；DINOv2 的 iBOT=1、KoLeo=0.1 和 cell MSE=1 可作为可修改起点。首版运行记录四项未加权和加权 loss、有效目标数量、student／teacher 表征统计及实际权重；不把调权写成重新选择训练目标的前置决策。

## 4. 梯度、EMA 与导出

1. Teacher 参数从 student 对应模块初始化，teacher 使用 `no_grad`／停止梯度。DINO／iBOT 的 teacher target 由 EMA 图像主干及配套 projection head 产生；固定 MAP bank 不更新。
2. 每次 optimizer step 后，对 teacher 对应参数执行 \(\theta_T\leftarrow m\theta_T+(1-m)\theta_S\)。DINOv2 本地默认 EMA momentum 从 0.992 按 cosine 调度到 1.0，可作为首版起点；实际参数和调度随实验记录。
3. Cell adapter 与 marker 表达头由 student cell MSE 训练，同时按相同 EMA 更新规则维护对应副本，以便与 EMA 主干配套导出。Cell adapter／表达头的 EMA 是针对新增模块的扩展；不增加独立 cell teacher loss。
4. 基座导出使用同一个训练步的 EMA tokenizer、ViT、cell adapter 及表达头；若只需 CLS，不执行 cell adapter。细胞注释和编辑比较使用配套 EMA 主干及 cell adapter，冻结后按各任务训练或使用读出。
5. 保存 optimizer、scheduler、teacher／student、center、梯度缩放器、optimizer update count、RNG、sampler 状态、解析后的训练配置及来源版本。恢复后保持采样、多视图遮盖与 EMA／scheduler 的连续性；梯度累积中间不更新 EMA。

## 5. 预训练后的像素 decoder

基座完成预训练后冻结 EMA 图像编码器，使用开发训练分组训练 marker 条件化像素 decoder，开发验证分组选参；Figure 2 M1–M4 与 Figure 3 S1 复用同一 decoder 结构和同一符合各任务数据留出规则的权重。该 decoder 不属于上述基座联合预训练 loss。

所有目标 marker 共享 decoder 参数。输入输出、条件接口和首版结构配置见[模型规范§7](model.md#7-图像-marker-重建读出)。

训练视图覆盖逐 marker 独立局部遮盖、整目标通道遮盖以及同一位置全 panel 遮盖，按各任务 ROI 坐标先生成遮盖后切窗。参照本地 VirTues，重建 MSE 对完整有效重建图计算；padding 和无效实测域排除。评价按对应协议只在指定的隐藏有效像素计分；训练配置不改变预先确定的数据划分和目标集。M4 的全局未见 marker 在基座预训练和 decoder 训练中都不能提供测量或预测监督；用同一合规 checkpoint 查询目标身份。

## 6. 批处理、优化与资源

1. 一次基础 crop 可生成多个不同大小 views；每 view 的 patch token 数、teacher 前向、adapter 逐实例计算及反向激活都计入显存和吞吐；以 ViT-B/8 各视图的实际 token 数估算资源。
2. Data loader 批量读取图像、同步实例标签、有效域和 marker 资产；同一 view 的全部实例共享一次图像编码，adapter 可按实例数分块。参数更新后重新计算 trainable features。
3. 首轮优先采用 BF16、SDPA、按需窗口 I/O 和梯度累积。按实测显存决定是否启用 activation checkpointing。分布式运行采用 PyTorch DDP。将各类视图的数量和计算量纳入首轮 profile。

| 项目 | 首轮 provisional 配置 |
| --- | --- |
| 硬件 | 4 或 8×H20 预训练；下游冻结特征导出和编辑推理可少卡 |
| Effective crop batch | 512，按基础 crop 计；`microbatch×world_size×accumulation_steps`，不按派生 view／cell／patch 加计 |
| 4／8 卡 batch 示例 | 4×每卡32×累积4 或 8×每卡32×累积2；均需按 ViT-B/8、多视图、teacher 和 adapter 实测显存后调整 |
| Optimizer | AdamW，betas=(0.9,0.999)，eps=1e-8；bias 与 normalization 参数不做 weight decay |
| Peak LR／weight decay | 2e-4／0.04 作为既有 bootstrap 起点，按实际 batch 和运行记录调整 |
| LR schedule | warm-up 后 cosine decay；总更新数、warm-up 与末端学习率随实验记录 |
| Precision／gradient clipping | 硬件支持时 BF16；全局 gradient norm 1.0 起点 |
| EMA | 参考 DINOv2 的 momentum 0.992→1.0 cosine，按 optimizer update 计数 |

采样应跨来源、患者和 image 分散，并记录实际采样与 batch 组成。短程训练先检查同步几何、两级 mask 可见性、梯度流、EMA 配对、训练恢复和真实显存，再进行正式训练。训练进展和验证结果记录在[项目状态](../project/status.md)。
