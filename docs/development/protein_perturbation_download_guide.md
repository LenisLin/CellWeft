# 三项 T 细胞蛋白扰动来源：数据层级与解释指南

2026-11-24：GSE278572、GSE330295 的逐细胞蛋白数据及 PRJDB16517 的聚合蛋白响应已接收，现行 NAS 来源入口为`/mnt/NAS_21T/ProjectData/CellWeft/raw/sc_protein/`；接收和文件验收详情见[当前状态](../project/status.md)。本文维护已接收数据的层级、数值定义和解释边界；来源内响应估计、marker 对应及空间图像映射尚未完成。

## 1. 科学用途与工作边界

三项来源可为后续 Figure 6 或补充分析提供来源内的蛋白响应方向和条件信息。来源数值处于各自 assay 和归一化尺度；本文不把来源响应幅度直接换算为空间图像强度，也不指定具体 marker、剂量或转换关系。

- Figure 5 原有人为指定空间编辑全部保持不变。
- 本部分作为 Figure 6 或补充材料的候选来源。
- 来源内对照和作者定义可用于解释扰动方向；具体映射需在后续研究中根据 marker、细胞类型、条件及测量定义另行评估。
- Figure 5 的人为指定编辑保持独立；外部来源不提供本图已确定的空间编辑数值。

## 2. 已接收来源一览

| 来源 | 已接收层级 | 来源内响应信息 |
| --- | --- | --- |
| GSE278572 | 逐细胞 ADT 计数及 guide、target、donor、细胞类型和条件连接 | 可按来源内 NTC 和条件估计响应；本项目尚未完成该响应估计 |
| GSE330295 | 逐细胞蛋白 deviance residual，按批次 z-score，含 guide、target、batch 和细胞记录 | 可按来源内 NTC 和条件估计响应；数值为来源特定的残差尺度 |
| PRJDB16517 | 作者聚合的 88 蛋白 × 227 KO 靶点响应分数 | `score = (mean_perturbed - mean_NTC) / mean_NTC`；仅有聚合分数，无逐细胞或逐供体响应 |

来源响应可以在各自定义内提供方向信息；不同数据层级和数值尺度不支持直接比较绝对幅度或直接赋值为空间图像像素。具体 marker 对应和空间映射未在本文确定。

## 3. 来源数据与数值定义

### 3.1 数据层级与来源内解释

| 层级 | 已核对内容 | 可支持的来源内解释 |
| --- | --- | --- |
| 逐细胞蛋白 | 蛋白数值与身份、细胞键、扰动和对照连接；细胞类型与条件可从逐细胞表或实验设计确定 | 在来源内按扰动、细胞类型和条件估计响应方向与分布 |
| 聚合蛋白响应 | 蛋白×扰动数值、细胞类型／实验条件、对照及作者分数定义 | 保留作者定义解释来源内响应方向；不把分数换算为空间图像强度 |

供体、批次、时间、剂量、guide、QC 标签按来源实际情况保留。药物数据必须有药物及对应剂量；单供体或固定时间点可以由研究级 metadata 描述。没有供体 ID 时记录其缺失，不伪造 donor，也不将实验重复视为供体。供体缺失本身不否定表达—处理连接已明确的数据用途。

### 3.2 数据接收与复核标准

1. 从官方目录、API 或论文数据声明确认文件身份和版本。读取少量实际 payload；HTTP 200、网页标题或 HDF5 文件头本身不足以证明内容可用。
2. 小型 feature、guide、condition、donor 表尽量完整读取。大矩阵用 HTTP Range 或分块访问读取实际数值，记录其方向、维度及数值所在位置。
3. 蛋白必须是生物抗体 features；将 HTO／CellTag／同型对照与蛋白分开。RNA 与蛋白共存时明确取哪个 assay。
4. 至少展示一条真实扰动记录和一条真实对照记录的连接：同一 cell ID 的蛋白数值、marker、perturbation、condition；聚合表则展示对应行列及对照定义。非配对实验无需寻找同一细胞处理前后的配对。
5. 汇总可用范围内各扰动、条件和对照的数量，检查缺失值与重复键；不能用所有原始 droplets 数替代作者 QC 后细胞数。
6. 对已接收文件完整解析结构，检查压缩包可解压、矩阵维度与 metadata 对应，并读取实际涉及的蛋白数据块。稀疏矩阵避免整体 dense 化。

若大文件无法范围读取，允许先完整取得该单个处理后文件以完成复核；此时它属于待确认来源，不先宣布数据通过。TLS／网络超时记录为访问未完成，不据此判断作者缺少数据。

### 3.3 当前来源位置

三项已接收来源位于 NAS `raw/sc_protein/{GSE278572,GSE330295,PRJDB16517}/`。实际文件、验收记录和 8T 副本位置以[当前状态](../project/status.md)为准；本文不再保留接收缓存或临时工作目录作为执行入口。

## 4. GSE278572：逐细胞 ADT 蛋白来源

### 来源

- 论文：[Central control of dynamic gene circuits governs T cell rest and activation](https://www.nature.com/articles/s41586-024-08314-y)。
- [GEO GSE278572](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278572)。
- [GEO supplementary 目录](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278572/suppl/)。
- [作者 Zenodo 13924126](https://zenodo.org/records/13924126)；`data_tables.zip` 中的 `S14_metadata_Treg_Teff_perturbseq.xlsx` 提供逐细胞注释。

### 已实际核实的基线

原代人 CD4 Treg／Teff，CRISPRi 28 个调节因子，2 位供体，静息与再刺激 48 小时。

- features：36,807 项，含 36,601 RNA、65 guide、141 Antibody Capture；后者为 130 个蛋白抗体、7 个同型对照、4 个 HTO。
- matrix：36,807 features × 249,799 barcodes，1,027,798,128 个非零；gzip 大小 3,857,480,145 bytes。完整矩阵和作者筛选细胞的对应关系已核验。
- guide 表：191,620 个唯一 barcode，全部匹配矩阵；140,641 个单 guide。
- 作者 S14：100,087 个唯一细胞，全部匹配矩阵 barcode，sgRNA 全部与 guide 表一致；含 15,525 个 Non-Targeting。
- 两位供体各自在 Treg／Teff × resting／stimulated 四组均有 NTC。
- 实际读取过两个 S14 细胞的蛋白计数，并连接到 guide、target、donor 和 condition。
- `data_tables.zip` 已知大小 57,967,623 bytes；无需为获得 S14 下载 11.74 GB 的分析 pipeline ZIP。

### 当前可用层级

该来源已通过逐细胞蛋白层级验收，作者筛选的 100,087 个细胞、130 个生物蛋白抗体、28 个扰动靶点，以及 guide、donor 和 Treg／Teff × resting／stimulated 条件连接已核对。ADT 计数支持来源内按条件和细胞类型估计相对 NTC 的响应；估计结果尚未完成，计数幅度不直接映射为空间图像强度。

## 5. SCITO-Perturb-seq：大规模 CRISPRa 蛋白来源

### 来源与文件

- [预印本](https://www.biorxiv.org/content/10.64898/2026.08.08.743670v1.full)。
- [GEO GSE330295](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE330295)。
- 完整文件：[GSE330295_deviance_norm.h5ad](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE330nnn/GSE330295/suppl/GSE330295_deviance_norm.h5ad)，4,666,269,128 bytes。
- [作者代码](https://github.com/bjhwang113/SCITO-Perturb-seq)，尤其 `01_sgRNA_LFC_se_zscore.Rnw` 的特征过滤、NTC 定义和响应计算。

### 已实际核实的基线

- `X` 为 float32 dense，3,134,864 cells × 203 features；实际值包括负数，`layers` 为空。
- `var/marker` 有 203 个名称，包括 CD279、CD27、CD38。作者代码移除 IgE 和 IgLightChainL 后使用 201 项；不能仅凭名称将这两项称作 isotype。
- `obs` 的 11 个字段与矩阵行数一致且无缺失；guide→target 无多重映射，所有批次均有 NTC。
- 全部 636,377,392 个 `X` 数值有限；3,134,864 个细胞中有 17,755 个非 NTC 靶点细胞和 26,602 个 NTC。
- 来源是单供体、活化 CD4 T 细胞、单一时间点的 CRISPRa 实验。`Type` 出现 Doublet／Quadruplet 等标签，不能当作生物细胞亚型。
- 数值为技术协变量回归后的 deviance residual，再按批次 z-score；该来源尺度只支持来源内解释。

### 当前可用层级

该来源已通过逐细胞蛋白层级验收：完整 H5AD 的矩阵、marker、guide／target 连接及九批 NTC 覆盖已核对。表达值为批次 z-score 的 deviance residual，可在来源内按批次、扰动和细胞类型解释响应方向；其绝对数值不与 ADT 计数、作者聚合分数或空间图像强度直接换算。

## 6. FOXP3 Perturb-icCITE-seq：聚合蛋白响应来源

### 来源与文件

- [Nature 论文](https://www.nature.com/articles/s41586-025-08795-5)，原代人 CD4 细胞在 iTreg 诱导条件下接受基因 KO。
- [官方补充表 ZIP](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-08795-5/MediaObjects/41586_2025_8795_MOESM5_ESM.zip)，12,645,918 bytes，含 6 个 xlsx。
- [官方补充方法](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-08795-5/MediaObjects/41586_2025_8795_MOESM1_ESM.pdf)，529,176 bytes。
- 原始来源 PRJDB16517；[作者代码](https://github.com/agiguelay/Perturb-icCITEseq)。

### 已实际核实的基线

- Supplementary Table 2 的 `ProteinScore_Targets`：88 个蛋白 × 227 个 KO 靶基因，19,976 个数值，无缺失。
- 蛋白包括 FOXP3、Granzyme B、PD1、CD25、CD27、CD38、CTLA4、Ki67 等；保留特定磷酸化表位的完整名称。
- 方法定义：蛋白 CLR 归一化后，`score = (mean_perturbed - mean_NTC) / mean_NTC`。表中没有单独 NTC 列，因为对照已进入评分。
- Table 5 包含 guide 及 NTC 设计，Table 6 包含抗体 panel／barcode 参考。
- 聚合表没有逐细胞 barcode、逐供体结果或响应分布。没有找到可直接下载的 processed cell×protein 及 cell→guide→donor 完整映射。
- 原始 ADT 与 guide FASTQ 前缀确实可读取，但 ADT 约 152 GB、CROPseq 约 66 GB，另有大量 GEX／hashing 数据。当前用途无需依赖该重建路线。

### 当前可用层级

PRJDB16517 已按聚合蛋白响应层级接收。作者分数基于 CLR 归一化值并以 NTC 为参照；其符号可说明来源内的相对响应方向，幅度保持为作者定义的相对分数。该表没有逐细胞、逐供体或响应分布信息，不据此推断这些层级。

## 7. GSE300551：本轮排除

GSE300551 未进入本轮接收范围；官方入口访问和正式 vehicle 连接未能闭合，排除理由与实际访问结果见[当前状态](../project/status.md)。

## 8. 响应尺度与空间映射边界

1. GSE278572 的逐细胞 ADT 计数可在其来源内按已连接的扰动、对照、细胞类型及条件估计响应方向；本项目尚未执行该估计。
2. GSE330295 的 deviance residual 经批次 z-score，PRJDB16517 的分数则按作者定义相对 NTC。二者的响应方向限于来源自身定义和比较，数值幅度保留来源尺度。
3. 三种来源的 assay、归一化和统计层级不同；来源间幅度不可直接比较，也没有从这些数值到空间图像通道强度的已验证换算。
4. 后续可在明确的 marker、细胞类型和条件对应后将来源响应方向作为候选编辑方向。本文不指定跨来源转换、空间像素赋值、dose 或 marker 模板；这些需由单独分析确定。

来源文件、读取验收和路径以[当前状态](../project/status.md)为准。
