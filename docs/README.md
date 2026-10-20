# CellWeft 文档入口

现行内容按职责维护。开始工作先读[当前状态](project/status.md)，再读对应规范。文档规格不代表代码、训练或生物学验证已经完成。

| 职责 | 唯一维护位置 |
| --- | --- |
| 科学目标、动机、范围与里程碑 | [研究计划](project/research_plan.md) |
| 已验证进度与下一步 | [当前状态](project/status.md) |
| 接受决定及依据 | [决策记录](project/decisions.md) |
| 模块归属与配置组织 | [架构概览](architecture/overview.md) |
| 记录字段、batch、索引与空间对应 | [数据合同](architecture/data_contract.md) |
| Protein bank、双基座、Fusion 与 V2 接口 | [模型规范](architecture/model.md) |
| Loss、masking、EMA、冻结与梯度 | [训练规范](architecture/training.md) |
| EditSpec、三尺度编辑与比较输出 | [编辑规范](architecture/editing.md) |
| 数据来源、训练就绪与患者划分 | [数据目录](data/catalog.md) |
| 公开数据收集、候选维护、资源表与 agent 流程 | [收集规范](data/collection.md) |
| 全局marker、study准备与逐image处理 | [数据准备](data/preparation.md) |
| 准备模块、dataset入口、参数与运行命令 | [准备实现说明](../scripts/data/preparation/README.md) |
| 表征任务、基线与消融 | [表征评价](evaluation/representation.md) |
| 删除响应、对照与独立验证 | [扰动评价](evaluation/perturbation.md) |
| 预算、集成、实施步骤、35 项工程验收与目标 CLI | [工程规范](development/engineering.md) |
| SSD/NAS 路径与产物归属 | [存储规范](architecture/storage.md) |
| 开发与英文 Git 提交流程 | [开发流程](development/workflow.md) |
| MAP 固定表示、DINOv3 权重与来源版本 | [资源部署](development/resources.md) |
| 来源链接、历史核验与许可 | [参考索引](references.md) |
| 12 源图、5 组合的预处理与分割实测 | [分割探索结果](experiments/segmentation_pilot_20261019.md) |
| 12 个 CP_B 固定区域的蛋白通道定量 | [定量实测与稀疏通道缺口](experiments/protein_quantification_pilot_20261020.md) |
| 12 个固定区域的全部/非零像素 P99 比较 | [非零 P99 实测与信号限制](experiments/nonzero_p99_pilot_20261020.md) |

具体实验的设计与结果摘要归 experiments/；未决讨论归 project/discussions/。只有开展对应工作时才建立记录，不填充占位文件。
图像准备从[准备规范](data/preparation.md)进入；字段与文件布局分别见[数据合同](architecture/data_contract.md)和[存储规范](architecture/storage.md)。[多机安排](project/discussions/image_preparation_workflow_20261020.md)保留资源信息，[Crop来源核对](project/discussions/crop_sampling_sources_20261020.md)保留方法依据。
各规范维护当前要求，decisions 记录依据，status 记录实况；其他文档使用链接引用。新规范替代历史要求时保留继任关系。
