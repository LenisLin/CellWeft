# CellWeft：空间蛋白基础模型研究计划

工作名：CellWeft（细胞织域）。
英文题名：CellWeft: A Cell-Centric, Knowledge-Informed Spatial Proteomics Foundation Model for Multiscale In Silico Perturbation。
项目名为工作名。实现进度见[当前状态](status.md)，技术合同由[文档入口](../README.md)链接的专项规范维护。

## 1. 研究目标

V1 数据范围限定为人类肿瘤（含原位癌）及其配对癌旁、正常对照，以及独立正常组织；其他癌前病变暂缓，不纳入以其他非肿瘤疾病为主体的研究。逐 study 准入和混合队列处理见[收集规范](../data/collection.md)。

以细胞为可索引、可编辑单位，结合固定蛋白知识、实例 mask 与完整 multiplex images，学习 cell、spatial patch、crop 和组织表示，并研究细胞上下文与区域读出。Whole-cell expression head 以离线 whole-cell 定量为监督，验证模型预测与该定量的一致性；评价模型能否改善原始定量时，以独立测量或重复测量为参照。
V1 使用单一空间蛋白图像主干输出 patch 与 crop 表征，以主干后的实例适配器读取细胞表征。结构见[模型规范](../architecture/model.md)。
以移除 marker 观测、单个细胞或细胞集合为基础，研究保留细胞与固定微环境区域的表示变化；Figure 5 同时探索细胞替换及 marker 数值／状态编辑，关注参考状态方向、空间背景依赖和组合效应。
Cell-removal 的读出反映编辑输入后的模型变化；有方向的生物学假设由独立参考状态及证据支持。另以外部单细胞蛋白扰动数据提供条件化蛋白变化，指导空间图像中的目标细胞编辑，作为 Figure 6 或补充材料候选。

## 2. 动机与取舍

相同自身状态的细胞可能处于不同组织环境。以肿瘤免疫微环境为例，研究周围 macrophages、CAF 等细胞的存在及空间组织如何影响目标 T cell 的表示。
图像保留形态、亚细胞分布、组织边界和胞外信号；实例 mask 提供稳定细胞身份、读出区域和编辑对象。图像主干输出空间表示，实例读出结合分割信息获得细胞表示。
借鉴 Geneformer 对模型输入对象编辑后重新编码的范式；niche 删除解析为 cell IDs，在图像及实例视图上执行编辑。参考资料见[文献索引](../references.md)。

## 3. 科学问题

1. 利用共享分子身份和图像学习的表征能否迁移到不同 marker panels 和 cohorts？
2. 该模型的 spatial features 能否表达细胞邻域、组织界面与可靠标记支持的胞外模式？
3. 同一 checkpoint 的 cell、patch、crop 和组织读出能否支持对应尺度的任务？
4. Marker/cell/niche 删除、细胞替换及 marker 状态编辑能否产生具有选择性、背景依赖且跨患者复现的表示方向变化，并由真实组织／细胞检索解释？
5. 公共预训练能否迁移到未参与预训练与适配的医院？
6. 双中心胃癌治疗前后的空间表征变化，能否关联病理响应与随访结局，比较纯化疗与化免，并返回到可解释的组织区域和细胞状态？

固定单模型的读出与通用下游评价见[表征评价](../evaluation/representation.md)，胃癌双中心临床应用的队列规则、Figure 4 图版和分析流程见[Figure 4 协议](../paper/figure4_gastric_cancer.md)，临床相关微环境的分型编辑、方向和真实检索见[Figure 5 协议](../paper/figure5_spatial_editing.md)，输入编辑及外部蛋白扰动指导的实验见[扰动评价](../evaluation/perturbation.md)。

### 3.1 Figure 1–5 研究主线与输入输出

Figure 1 图形化呈现数据全景、模型架构、任务纵览与下游应用。Figure 2／3 检验分子及组织表征能力，Figure 4 连接胃癌临床应用，Figure 5 研究同一模型的人为指定空间编辑响应。外部蛋白扰动指导的细胞编辑单独考虑 Figure 6 或补充材料。下表维护研究主线；成员、标注、指标与具体分析由任务规范维护。

| 图 | 输入与任务 | 输出及所属规范 |
| --- | --- | --- |
| Figure 1 | 多来源空间蛋白数据、固定 MAP 知识、完整图像与细胞实例 mask | 数据全景、图像主干与实例细胞读出、任务与应用关系示意；结构见[模型规范](../architecture/model.md)。 |
| Figure 2 | 四种蛋白图像恢复条件及内部／跨 dataset 细胞注释 | 目标 marker 像素恢复及逐细胞标签预测，含不同标注预算的比较。[任务协议](../evaluation/representation.md#3-figure-2-蛋白恢复与细胞类型注释) |
| Figure 3 | 空间回补、局部组织识别、患者表型预测及相似组织检索 | 像素恢复、image／ROI 分类、患者预测和真实组织排序。[任务协议](../evaluation/representation.md#4-figure-3-组织结构理解与表型应用) |
| Figure 4 | 两院胃癌治疗前后图像及临床标注 | 折外病理响应预测、组织解释及后续生存分析。[临床协议](../paper/figure4_gastric_cancer.md) |
| Figure 5 | CellWeft 人工删除／减少、CAF→T 替换及指定 marker 数值编辑，复用 Figure 4 固定患者 MIL | 编辑后表征、响应概率、真实检索及匹配 R 参考方向。[编辑实验协议](../paper/figure5_spatial_editing.md) |
| Figure 6 或补充材料（候选） | 外部实测单细胞或作者聚合的蛋白扰动响应，指导空间图像中匹配细胞的编辑 | 编辑后细胞、区域及患者读出。[扰动评价](../evaluation/perturbation.md#5-外部扰动来源的响应解释) |

## 4. 能力与训练流程

以固定 MAP 表示和空间图像学习基座表示，经联合预训练、冻结下游适配及固定模型评价检验跨尺度能力。训练阶段由[训练规范](../architecture/training.md)维护。数据准备、划分与私有来源的使用范围见[数据准备](../data/preparation.md)和[数据目录](../data/catalog.md)。
## 5. 里程碑

| 阶段 | 交付与验收重点 |
| --- | --- |
| M0 | 研究/架构/config/sources/decision ledger；锁定正式设计与来源版本。 |
| M1 | Manifests、marker registry、cell/kNN/crop indices、合成 fixtures；核对坐标、实体、patient split 与去重。 |
| M2 | 实现图像 token、Spatial ViT、CLS、实例 mask 细胞读出和像素恢复接口；核对 panel、观测状态和输出维度。 |
| M3 | 接通 DINO／iBOT、KoLeo 与 whole-cell MSE，完成短程训练接线验收；核对未遮盖 student global view 的 cell loss、梯度流、配套 EMA 和恢复。 |
| M4 | 联合训练恢复、批处理与单配置资源验收；确定实际 microbatch、累积与训练预算。 |
| M5 | Marker/cell/niche 删除及计划中的替换／状态编辑与按需导出；核对稳定 IDs、编辑来源、crop 依赖、固定 readout 和归一化。 |
| M6 | 正式联合预训练与 checkpoint 交付；登记真实训练范围和配置。 |
| M7 | 冻结编码器后的像素 decoder、细胞／图像／患者下游任务、输入编辑分析及医院应用；按患者划分和汇总。 |

实施与工程验收见[工程规范](../development/engineering.md)，实际实施与验证见[当前状态](status.md)。
