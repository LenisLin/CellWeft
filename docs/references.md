# CellWeft — 核心文献与资源索引

原始包记录的核验日期为 2026-10-17；下文“本轮”“当前”和已核验描述均指该次来源记录，不表示本次文档整理重新核验。R01–R18 保持稳定编号；论文、代码、数据门户和候选资源分别记录，公开可浏览不等于允许下载、再分发或使用全部权重。实际集成时的版本、许可、访问要求与来源配置按[工程规范](development/engineering.md)维护。

## 一、直接进入 V1 设计的参考

### R01 · Geneformer

- Theodoris et al. *Transfer learning enables predictions in network biology*. Nature (2023).
- 论文：https://www.nature.com/articles/s41586-023-06139-9
- 本轮直接核验的官方操作说明：https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html
- 借鉴点：在已训练的模型中执行定义明确的输入编辑，再重编码。官方 `delete` 从 rank-value encoding 中删除 gene；`overexpress` 将 gene 移到序列前端。V1 借鉴操作范式，不复制 RNA rank-value 编码为蛋白测量接口。

### R02 · VirTues

- *The Virtual Tissues foundation model resolves spatial proteomics across scales*. Nature (2026).
- 论文：https://www.nature.com/articles/s41586-026-10884-y
- 官方代码：https://github.com/bunnelab/virtues
- 早期公开预印本：https://arxiv.org/abs/2501.06039
- 本轮核验依据：官方仓库的论文标题、期刊说明、预计算 ESM-2 marker embeddings、数据接口和训练入口。期刊全文在本轮直连受到站点跳转限制，因此不重新引用未经读取的细节参数。
- 借鉴点：不同 marker panels 的 identity-aware 表征、多通道图像、空间蛋白预训练与 spora 接口。我们的 per-position marker fusion → DINOv3 spatial tokens 是新工程设计，不声称逐层复制 VirTues。

### R03 · KRONOS

- *A Foundation Model for Spatial Proteomics*. 2025 preprint.
- 论文：https://arxiv.org/abs/2506.03373
- 官方代码：https://github.com/mahmoodlab/KRONOS
- 核验点：官方说明包含 shared channel-wise stem、sinusoidal marker-identity embeddings，以及基于 DINOv2 的实现。
- 用途：multiplex image backbone 与评估基线。不是 V1 的第二套训练主工程。
- 许可提示：代码和权重存在非商业学术使用限制；代码重用与论文引用分开管理。

### R04 · DINOv3

- Siméoni et al. *DINOv3*. Technical report (2025).
- 论文：https://arxiv.org/abs/2508.10104
- HTML：https://arxiv.org/html/2508.10104v1
- 官方代码：https://github.com/facebookresearch/dinov3
- 官方基础配置：https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/configs/ssl_default_config.yaml
- 官方环境：https://raw.githubusercontent.com/facebookresearch/dinov3/main/conda.yaml
- 已核验基础配置：ViT patch size 16；local crop 数量 8；基础 YAML 中 global size 224、local size 96；两个 heads 均为 65,536 prototypes、hidden 2,048、bottleneck 256；DINO、iBOT、teacher EMA 及可选 Gram 配置均存在。
- CellWeft 正式结构与训练覆盖由[模型规范](architecture/model.md)和[训练规范](architecture/training.md)维护；本条保留上游来源信息。
- 注意：官方公布的 86M ViT-B 权重属于其发布模型；我们使用 B 架构从头进行领域预训练，不将其公布权重的训练历史与我们的流程混为一谈。

### R05 · TERRA

- 官方代码：https://github.com/Lotfollahi-lab/terra
- 官方文档：https://terra-st.readthedocs.io
- 官方引用元数据：https://raw.githubusercontent.com/Lotfollahi-lab/terra/main/CITATION.cff
- 核验点：官方 README 说明 gene/cell/neighborhood 层级、JEPA latent prediction 和空间邻域序列化。
- 采用点：TERRA-inspired index cell、邻居距离 rank 和层级组织方式。具体 tokenizer 版本应在本地读取代码并锁定。
- 文献状态：本轮官方 CITATION.cff 仍写 `Manuscript in preparation`。此前讨论的 bioRxiv 地址 https://www.biorxiv.org/content/10.64898/2026.07.29.741565v1.full 本轮返回 403；不能据此固定发表状态、作者顺序或最终 DOI。正式 bibliography 以之后可核验的 manuscript 为准。

### R06 · CellWorld

- Liu et al. *CellWorld: From Gene-Level Reconstruction to Latent Cell Prediction in Spatial Transcriptomics Foundation Models*. 2026 preprint.
- 论文：https://arxiv.org/abs/2608.06659
- HTML：https://arxiv.org/html/2608.06659v1
- 官方代码：https://github.com/UoM-HealthAI/CellWorld
- 核验点：student 编码可见细胞，EMA target 编码完整上下文，predictor 预测被遮盖细胞的 latent；partial-expression hints 仅提供给 predictor。
- 采用点：masked-cell contextual latent prediction，而非其全部基因 tokenizer、organ/platform 输入或坐标编码。CellWeft 使用蛋白输入、TERRA-inspired cell-rank，joint teacher 扩展为双分支，是本项目设计。

### R07 · ST-Align

- 公开预印本标题：*ST-Align: A Multimodal Foundation Model for Image-Gene Alignment in Spatial Transcriptomics*.
- 预印本：https://arxiv.org/abs/2411.16793
- HTML：https://arxiv.org/html/2411.16793
- 用户提供的版本：https://openreview.net/forum?id=EWzPy228cM
- 核验依据：本轮读取 arXiv 全文；OpenReview 出现浏览器验证页面。正式题名/版本以用户最终论文和可读取的正式记录为准。
- 采用点：节选 Neighborhood–Region 欧氏对齐，以及双向 cross-attention 思路；不照搬其所有 alignment 项。

### R08 · HyperST

- Zhang et al. *HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction*.
- 预印本：https://arxiv.org/abs/2511.22107
- HTML：https://arxiv.org/html/2511.22107
- 用户提供的 CVPR 页面：https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_HyperST_Hierarchical_Hyperbolic_Learning_for_Spatial_Transcriptomics_Prediction_CVPR_2026_paper.html
- 核验依据：本轮读取 arXiv；CVF 页面未直接成功获取。
- 采用状态：问题演化与研究背景参考；V1 不叠加双曲空间、entailment loss 或另一套 hierarchy regularization。

## 二、Protein knowledge pretraining

### R09 · MAP / MAP-KG

- Feng et al. *A knowledge-driven framework for predicting single-cell responses for unprofiled drugs*. Nature Machine Intelligence (2026), published 26 August 2026.
- 论文：https://www.nature.com/articles/s42256-026-01286-w
- 官方代码（论文给出）：https://github.com/MAGIC-AI4Med/MAP
- MAP-KG（论文给出）：https://huggingface.co/datasets/RainGate/MAP-KG
- 用途：候选 molecular knowledge representation；可评估直接导出 protein/gene embeddings。可访问的论文确认其结合蛋白序列、药物结构与机制文本。并不在初始化时假定任意 UniProt/marker 都已有现成权重。
- 实现前检查：实体 ID、embedding 维数、可单独导出的 encoder、覆盖率、模型许可、gene→protein 映射、复合靶点。

### R10 · SIGNOR 4.0

- Lo Surdo et al. *SIGNOR 4.0: the 2025 update with focus on phosphorylation data*. Nucleic Acids Research 54(D1), D682–D690 (2026); online 17 November 2025.
- 论文：https://academic.oup.com/nar/article/54/D1/D682/8324960
- DOI：https://doi.org/10.1093/nar/gkaf1237
- 数据：https://signor.uniroma2.it/
- 下载：https://signor.uniroma2.it/downloads.php
- 用途：人工整理的方向、正负调控、机制及 PTM 知识；score 是证据置信度，不当作干预效应幅度。关系未收录应标记 unknown。

### R11 · OmniPath

- 门户：https://omnipathdb.org/
- 论文：Türei et al. *Integrated intra- and intercellular signaling knowledge for multicellular omics analysis*. Molecular Systems Biology (2021), 17:e9923.
- 用途：互作、调控、enzyme–PTM、复合物与注释。按原始资源保留 provenance 和许可，不把重复聚合的证据误作独立证据。

## 三、V1 数据与 benchmark

### R12 · spora

- 当前入口：https://spora.bioimagearchive.org/projects/spora
- 文档：https://spora.epfl.ch/docs/
- VirTues 官方仓库中也给出下载/转换指引。
- 本轮官方门户统计：13,546 multiplex images、5,477 patients、35 cohorts、5 technologies。
- 使用方式：manifest 优先；逐 cohort 检查 images、marker metadata、cell masks、annotations、访问规则。不是假设全部样本同时具有每一种注释。
- 与 CellWeft 15,000 张、40 markers、1,000×1,000 pixels 的规划数字分开保存；后者是计算预算假设，不是门户事实。

### R13 · HTAN

- 门户：https://humantumoratlas.org/
- 文档：https://docs.humantumoratlas.org/
- 用途：补充不同肿瘤和组织队列的 multiplex imaging。下载前按 assay 选择，不能把全部组学数据都视作空间蛋白图像。

### R14 · HuBMAP

- 当前门户：https://portal.hubmapconsortium.org/
- 用途：补充组织与 CODEX 等成像资源。当前门户可解析正文有限，具体 cohort 清单和文件类型在接入时核验。

### R15 · spora-bench

- 入口：从 https://spora.epfl.ch/docs/ 进入 spora-bench 文档。
- 用途：细胞、亚细胞、组织层面的评估协议。不要假定它已经覆盖本项目所有 synchronized deletion 任务；后者单独实现。

## 四、V2 后续扩展，不进入 V1 初始预训练

### R16 · CellOT / 4i data

- Bunne et al. *Learning single-cell perturbation responses using neural optimal transport*. Nature Methods 20, 1759–1768 (2023).
- DOI：https://doi.org/10.1038/s41592-023-01969-x
- 代码：https://github.com/bunnech/cellot
- 处理后数据：https://www.research-collection.ethz.ch/entities/researchdata/7964ba4a-92c7-4031-80f4-358f681c5056
- 用途：寻找蛋白成像 perturbation 数据与非配对分布监督思路；V2 不必直接采用完整 CellOT predictor。

### R17 · Single-cell CyTOF perturbation

- Bodenmiller et al. *Multiplexed mass cytometry profiling of cellular states perturbed by small-molecule regulators*. Nature Biotechnology 30, 858–867 (2012).
- DOI：https://doi.org/10.1038/nbt.2317
- 全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC3627543/
- 用途：真实单细胞蛋白/磷酸化测量；当前下载位置、条件元数据与再使用许可在 V2 数据接入时进一步检查。

### R18 · Perturb-CITE-seq

- Frangieh et al. *Multimodal pooled Perturb-CITE-seq screens in patient models define mechanisms of cancer immune evasion*. Nature Genetics 53, 332–341 (2021).
- DOI：https://doi.org/10.1038/s41588-021-00779-1
- 官方代码：https://github.com/klarman-cell-observatory/Perturb-CITE-seq
- 处理矩阵：https://singlecell.broadinstitute.org/single_cell/study/SCP1064/multi-modal-pooled-perturb-cite-seq-screens-in-patient-models-define-novel-mechanisms-of-cancer-immune-evasion
- 原始数据：官方代码说明指向 DUOS-000124。
- 用途：遗传扰动与 RNA/ADT protein readouts；不标记成现成的小分子 drug-treated IMC 数据。

## 五、引用与实现规则

1. 论文思想、具体代码、预训练权重和数据，分别记录来源与许可。
2. 不把本文件中的 `main` URL 作为最终可复现实验版本：初始化时记录 commit SHA、配置校验和和模型/数据 manifest hash。
3. 尚未读到完整正式文稿的记录只使用已核验的官方仓库或预印本元数据；不编造最终刊物、作者顺序、卷期或 DOI。
4. 所有参考算法是设计依据，不把 CellWeft 的新双分支 mask/fusion 机制描述成这些论文已完成的功能。
