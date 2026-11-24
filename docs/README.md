# CellWeft 文档入口

## 研究与论文

| 职责 | 唯一维护位置 |
| --- | --- |
| 科学范围、图版主线与里程碑 | [研究计划](project/research_plan.md) |
| 已验证资产、实施进度与下一步 | [当前状态](project/status.md) |
| 接受的决定与依据 | [决策记录](project/decisions.md) |
| 论文 Methods、补充表与 Figure 4／5 协议 | [论文写作材料](paper/README.md) |
| Figure 2／3 的读出、成员规则与评价指标 | [表征评价](evaluation/representation.md) |
| Figure 4 的冻结编码器、panel 与特征提取 | [临床编码器工作流](evaluation/clinical/frozen_encoders.md) |
| Figure 4／5 的 TRG 建模、解释与编辑实施 | [临床实施工作流](development/clinical_figure4_5_workflow.md) |
| 输入编辑的响应、对照与验证 | [扰动评价](evaluation/perturbation.md) |
| 外部蛋白扰动的来源、测量与对照 | [来源指南](development/protein_perturbation_download_guide.md) |
| 方法出处、官方资源与许可 | [参考索引](references.md) |

## 数据准备与核验

| 职责 | 唯一维护位置 |
| --- | --- |
| 数据来源、训练就绪层级与患者划分 | [数据目录](data/catalog.md) |
| 公开数据收集、候选维护与资源表 | [收集规范](data/collection.md) |
| 公开患者和样本连接的逐 study 证据 | [患者 Meta 审计](data/patient_metadata_public_scope_audit_20261111.md) |
| 来源解析、sample 与 panel 对接、原生输入交付 | [输入工作流](data/study_input.md) |
| 输入和临床字段定义 | [CSV 模板说明](data/templates/study_input/README.md) · [Excel 副本](data/templates/study_input_metadata_template.xlsx) |
| 独立 D01 校对、分层抽样与异常扩查 | [输入校对](data/study_input_review.md) |
| Marker、强度、分割、细胞定量与切窗 | [数据准备](data/preparation.md) |
| 准备入口、参数及恢复命令 | [准备实现说明](../scripts/data/preparation/README.md) |
| 私有膀胱癌队列批次范围及验收 | [队列准备计划](development/imc_20240725_63_preparation.md) |

## 技术与工程

| 职责 | 唯一维护位置 |
| --- | --- |
| 代码模块归属与配置组织 | [架构概览](architecture/overview.md) |
| 数据记录、batch、索引与空间对应 | [数据合同](architecture/data_contract.md) |
| 单图像主干、实例读出与像素 decoder | [模型规范](architecture/model.md) |
| 训练视图、loss、EMA 与 provisional 配置 | [训练规范](architecture/training.md) |
| EditSpec、重编码与输出对齐 | [编辑规范](architecture/editing.md) |
| SSD／NAS 产物归属、发布与接收 | [存储规范](architecture/storage.md) |
| 实现顺序、必要工程检查与目标 CLI | [工程规范](development/engineering.md) |
| 固定资源、版本配置与本地部署 | [资源部署](development/resources.md) |
| 8T 训练盘交付和服务器恢复 | [搬运规范](development/training_disk8t_transfer.md) |
| 开发和 Git 工作流 | [开发流程](development/workflow.md) |

## 实测结果

- [分割探索](experiments/segmentation_pilot_20261019.md) · [蛋白通道定量](experiments/protein_quantification_pilot_20261020.md) · [非零 P99 试验](experiments/nonzero_p99_pilot_20261020.md)
