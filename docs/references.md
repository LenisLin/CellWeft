# CellWeft — 核心文献与资源索引

本页维护现行方法和科学依据的来源。论文、代码、权重及数据分别记录。采用版本与许可见[来源配置](../configs/sources.yaml)，设计选择及继任关系见[决策记录](project/decisions.md)。

## 一、模型方法与设计背景

### R01 · Geneformer

- Theodoris et al. *Transfer learning enables predictions in network biology*. Nature (2023).
- 论文：https://www.nature.com/articles/s41586-023-06139-9
- 本轮直接核验的官方操作说明：https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html
- 借鉴点：在已训练的模型中执行定义明确的输入编辑，再重编码。官方 `delete` 从 rank-value encoding 中删除 gene；`overexpress` 将 gene 移到序列前端。V1 在蛋白图像及实例视图上执行输入编辑后重新编码。

### R02 · VirTues

- *The Virtual Tissues foundation model resolves spatial proteomics across scales*. Nature (2026).
- 论文：https://www.nature.com/articles/s41586-026-10884-y
- 官方代码：https://github.com/bunnelab/virtues
- 早期公开预印本：https://arxiv.org/abs/2501.06039
- 借鉴点：不同 marker panels 的 identity-aware 表征、多通道图像与空间蛋白接口。CellWeft 的 protein-aware tokenizer 与 marker 条件化像素 decoder 按[模型规范](architecture/model.md)适配；冻结编码器后训练的阶段安排由[训练规范](architecture/training.md)维护。

### R03 · KRONOS

- *A Foundation Model for Spatial Proteomics*. 2025 preprint.
- 论文：https://arxiv.org/abs/2506.03373
- 官方代码：https://github.com/mahmoodlab/KRONOS
- 核验点：官方说明包含 shared channel-wise stem、sinusoidal marker-identity embeddings，以及基于 DINOv2 的实现。
- 用途：multiplex image 表征和 panel 处理的方法背景，保留来源与许可记录。
- 许可提示：代码和权重存在非商业学术使用限制；代码重用与论文引用分开管理。

## 二、Protein knowledge pretraining

### R09 · MAP / MAP-KG

- Feng et al. *A knowledge-driven framework for predicting single-cell responses for unprofiled drugs*. Nature Machine Intelligence (2026), published 26 August 2026.
- 论文：https://www.nature.com/articles/s42256-026-01286-w
- 官方代码（论文给出）：https://github.com/MAGIC-AI4Med/MAP
- MAP-KG（论文给出）：https://huggingface.co/datasets/RainGate/MAP-KG
- 当前用途：固定 MAP gene-projector 表示已部署，供各 study 按 S1 导出 marker bank；实际数值验证与资源身份见[资源部署](development/resources.md)。
- 实际 panel 的 canonical identity、measured form 与映射仍按[数据合同](architecture/data_contract.md)及[数据准备](data/preparation.md)逐项处理。

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

## 三、数据与评价协议参考

### R12 · spora

- 当前入口：https://spora.bioimagearchive.org/projects/spora
- 文档：https://spora.epfl.ch/docs/
- VirTues 官方仓库中也给出下载/转换指引。
- 用途：按实际 manifest 逐 cohort 核对图像、marker、实例、标签及访问范围。

### R13 · HTAN

- 门户：https://humantumoratlas.org/
- 文档：https://docs.humantumoratlas.org/
- 用途：补充不同肿瘤和组织队列的 multiplex imaging。下载前按 assay 选择，不能把全部组学数据都视作空间蛋白图像。

### R14 · HuBMAP

- 当前门户：https://portal.hubmapconsortium.org/
- 用途：补充组织与 CODEX 等成像资源。当前门户可解析正文有限，具体 cohort 清单和文件类型在接入时核验。

### R15 · spora-bench

- 入口：从 https://spora.epfl.ch/docs/ 进入 spora-bench 文档。
- 用途：参考其细胞、亚细胞、组织层面的评估协议；CellWeft 输入编辑及重编码按[编辑规范](architecture/editing.md)实现。

## 四、训练和实例读出
### R20 · DINOv2

- 论文：https://arxiv.org/abs/2304.07193
- 官方实现：https://github.com/facebookresearch/dinov2
- 配置：https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml
- 视图：https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py
- 训练及EMA：https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py
- 当前用途：DINO CLS蒸馏、iBOT patch蒸馏、KoLeo、多视图及EMA更新；CellWeft使用自有蛋白tokenizer、ViT-B/8与whole-cell监督。

### R21 · Mask-Adapter

- 论文：[Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Mask-Adapter_The_Devil_is_in_the_Masks_for_Open-Vocabulary_Segmentation_CVPR_2025_paper.html)。
- 方法全文：https://arxiv.org/html/2412.04533v2
- 当前用途：mask编码与图像特征融合、ConvNeXt空间权重预测及实例加权读出。CellWeft的8×8 mask降采样、768维输出和whole-cell MSE是项目适配，首版参数见[模型规范](architecture/model.md)。

## 五、复现记录

发布使用的来源版本、配置和模型／数据资产身份按[工程规范](development/engineering.md)登记。沿用已记录的资产完整性标识。
