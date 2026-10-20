# CellWeft：空间蛋白基础模型研究计划

工作名：CellWeft（细胞织域）。原始计划日期：2026-10-17；本文合并 2026-10-18 已确认的研究目标。
英文题名：CellWeft: A Cell-Centric, Knowledge-Informed Spatial Proteomics Foundation Model for Multiscale In Silico Perturbation。
项目名为工作名。实现进度见[当前状态](status.md)，技术合同由[文档入口](../README.md)链接的专项规范维护。

## 1. 研究目标

V1 数据范围限定为人类肿瘤及其配对癌旁、正常对照，以及独立正常组织；不纳入以其他非肿瘤疾病为主体的研究。逐 study 准入和混合队列处理见[收集规范](../data/collection.md)。

以细胞为可索引、可编辑单位，结合蛋白身份/功能先验、细胞 marker 测量与完整 multiplex images，学习 marker、intrinsic cell、contextual cell、neighborhood、region/niche 和 ROI/tissue 表示。
V1 研究删除 marker、单个细胞或细胞集合后，保留 target cell 与微环境的模型响应；优先研究细胞及细胞群删除，关注影响强度、参考状态方向、空间背景依赖和组合效应。
保留细胞自身测量固定时，其 intrinsic 表示不因邻居删除改变；主要响应读出为 neighbor-contextual、fused-contextual 和固定区域表示。区域整体变化与固定保留细胞集合的变化分别评价。
表示 delta 是扰动预测读出，其生物学解释需要独立证据。移除 marker 观测与模拟蛋白功能丧失采用不同操作定义；蛋白级空间响应读出尚未锁定。

## 2. 动机与取舍

相同自身状态的细胞可能处于不同组织环境。以肿瘤免疫微环境为例，研究周围 macrophages、CAF 等细胞的存在及空间组织如何影响目标 T cell 的表示。
图像保留形态、亚细胞分布、组织边界和胞外信号；细胞视图提供稳定实体、分子测量、邻近关系和编辑对象。两条分支共同编码同一区域。
借鉴 Geneformer 对模型输入对象编辑后重新编码的范式；niche 删除解析为 cell IDs，marker 删除同步作用于细胞输入和图像。参考资料见[文献索引](../references.md)。

| 问题演化 | 保留的方向 |
| --- | --- |
| 泛癌种与多医院 | 公共多平台数据支撑基座，两家医院承担独立应用，临床任务不决定唯一预训练目标。 |
| Virtual cell / tissue 与药物 | 保留共享 molecular/cell 接口；任意药物组织生成和药物条件训练不进入 V1。 |
| Drug–protein graph | 知识进入固定身份/功能表示；不把任意 KG delta 直接加到空间 hidden state。 |
| 可编辑输入 | Marker/cell/niche 编辑后重新编码，不要求先生成完整治疗后表达矩阵。 |
| 双分支 | 保留完整图像和细胞实体，不将图像限定为胞外特征或强制 composable-image teacher。 |
| 多尺度交互 | 参考 TERRA 邻域组织、ST-Align 双向交互和单一欧氏对齐；HyperST 保留为背景。 |
| 自监督 | Marker measurement prediction 与 masked-cell latent prediction 并存。 |
| 计算 | 图像主干采用 B；联合阶段冻结或少量解冻基座，复用 checkpoints 做消融。 |

## 3. 科学问题

1. 固定 PLM+regulatory embeddings 相比 PLM-only，是否改善跨 marker/panel/cohort 迁移？
2. 邻域与图像是否为 intrinsic cell 表示提供独立信息？
3. 相同基座 checkpoints 下，递归交互是否优于后期拼接和单轮交互？
4. 输入删除是否产生稳定、可追踪、具有选择性和背景依赖的目标细胞与微环境响应？
5. 公共预训练能否迁移到未参与预训练与适配的医院？

表征与消融方案见[表征评价](../evaluation/representation.md)，删除响应与验证见[扰动评价](../evaluation/perturbation.md)。

## 4. 能力与训练阶段

Protein knowledge embeddings 是固定、跨样本共享的前置资产，可独立生成。Spatial FM 是 V1 主体；药物条件 intrinsic-cell delta 是 V2。
优化流程为预训练与下游微调：预训练先独立训练图像/细胞基座，再联合训练；下游包括医院适配和临床任务。
V1 保留未来接口，不引入 CITE-seq、CyTOF、4i 扰动训练、drug-conditioned loss 或额外扰动数据下载。模型接口见[模型规范](../architecture/model.md)，程序见[训练规范](../architecture/training.md)。

## 5. 里程碑

| 阶段 | 交付与验收重点 |
| --- | --- |
| M0 | 研究/架构/config/sources/decision ledger；锁定正式设计与来源版本。 |
| M1 | Manifests、marker registry、cell/kNN/crop indices、合成 fixtures；核对坐标、实体、patient split 与去重。 |
| M2 | 双基座前向；验证可变 panel、动态 grids 和 validity masks。 |
| M3 | 官方图像训练适配、marker/latent 训练；验证 EMA、遮盖、loss 和梯度。 |
| M4 | Frozen encoders、Fusion、predictor、alignment；验证冻结和无目标泄漏。 |
| M5 | 三尺度编辑与表征导出；验证双视图、overlap/halo 和归一化一致性。 |
| M6 | Profiler、八卡启动、checkpoint/resume；验证吞吐、显存和分布式恢复。 |
| M7 | 基线、消融与医院应用；保留完整 split 和可复现实验证据。 |

实施步骤、工程验收及预算见[工程规范](../development/engineering.md)。文档完成不代表 M0 全部验收或模型实现完成。

## 6. 尚待锁定

实际 marker panel 到 MAP 表示库的实体映射与覆盖；Marker/Neighborhood/Fusion 宽度、层数、轮数和共享方式；masking/hint ratios；医院分工；全图统计估计与退化通道处置、大图分割拼接、protein quantification、原生物理 FOV/重采样、基础 crop 保存分辨率/步长/边界及存储布局；末端解冻时机。已确认的统一图像预处理、来源 mask 优先复用、cpsam_v2 新分割、邻域和准备流程见[数据准备](../data/preparation.md)。
候选数值在所属模型/训练规范中明确标为 provisional。正式 DINO heads、prototypes 和 local views 的改变需要显式决策。
