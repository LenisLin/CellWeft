# 工程实施与验收

本页维护资源测量、集成、实施顺序和工程验收。结构、训练、输入及编辑分别见[模型规范](../architecture/model.md)、[训练规范](../architecture/training.md)、[数据合同](../architecture/data_contract.md)与[编辑规范](../architecture/editing.md)。协作要求见[AGENTS](../../AGENTS.md)，实施进度见[当前状态](../project/status.md)。

## 1. 资源与实现策略

按实际 manifest 统计患者、图像、panel、细胞及可采样窗口。基础切窗见[准备规范§7](../data/preparation.md#7-crop-与训练视图)，bootstrap 资源和优化配置见[训练规范](../architecture/training.md)。

实测资源预算覆盖各 student／EMA 模块、优化器、全部视图、反向激活与实例权重图，按实际实例化统计参数。单独测量冻结编码器后 decoder 训练的显存及吞吐，记录训练硬件与准备节点。恢复合同见[训练规范§4](../architecture/training.md#4-梯度ema-与导出)。

## 2. 来源与实现参考

DINOv2 提供多视图、DINO／iBOT heads、KoLeo、teacher targets 及 EMA；Mask-Adapter 提供实例条件化权重读出；VirTues 提供蛋白遮盖与 Transformer 像素 decoder。具体来源见[参考索引](../references.md)。CellWeft 的多蛋白 tokenizer、8×8 网格、whole-cell MSE、配套 adapter EMA 和冻结读出阶段由本项目规范定义。

实际集成时在现有 `configs/sources.yaml` 登记所用 revision、URL、role、许可与访问日期。采用明确的本地依赖路径，缺少依赖时报告具体缺口。参考源码存在、文档规格完成和运行接线完成分别记录。

## 3. 实施顺序

1. 读取当前规范与工作区变更，接入已有 image／mask／whole-cell 目标及 MAP 资产；核对 patient split、marker 顺序和来源有效域。
2. 接入 batch 与训练视图，核对[空间对应](../architecture/data_contract.md#4-空间对应)及网格边界。
3. 接入图像主干和实例读出，执行 §4 的张量和实例检查。
4. 接入[联合目标与 EMA](../architecture/training.md)，核对梯度流、更新频率和恢复。
5. 完成 M3 短程训练接线验收：检查几何、loss 归约、梯度、EMA 与恢复，并测量候选配置的显存、吞吐及 I/O。
6. 完成 M6 正式基座预训练与 checkpoint 交付，登记实际训练范围、配置和资源结果。
7. 在 M7 接入共享像素 decoder、cell 注释、image／patient MIL 和固定表征检索。按[表征评价](../evaluation/representation.md)核对任务成员及评分域。
8. 在 M7 接入 Figure 5 输入编辑与固定 MIL，按[编辑规范](../architecture/editing.md)和[临床实施流程](clinical_figure4_5_workflow.md)验收。

## 4. 必要工程检查

1. 完整有效 224/8 输入输出 784×768 patch features 和一个 768 维 CLS；位置编码随 DINO 视图网格一致变化。
2. Marker 与图像通道成对重排保持输出一致；未测量和 padding 不作为真实信号，真实零与缺失由观测状态区分。单层 summary-query pooling 与其完整单层 summary 输出实现对照一致。
3. Instance mask 只进入主干后的 adapter；保持图像不变而改变 mask 时 CLS 不变，cell 读出按新实例重新计算。每 crop 一次图像编码，实例分块与整批读出结果及梯度一致。
4. Centroid 在半开窗口内决定细胞归属，跨 patch 的实例通过 mask 聚合；截断实例保持 whole-cell 监督。没有有效细胞的 crop 仍提供 DINO／iBOT 表征，cell loss 按有效 crop／cell／marker 归约。
5. 按[训练规范](../architecture/training.md)验证 iBOT 几何匹配、whole-cell 梯度、固定参数边界、配套 EMA 及恢复连续性。
6. 验证共享 decoder 的像素重组、编码器冻结、M4 目标全局留出及既定隐藏有效像素评分。
7. 验证编辑 no-op、观测状态、受影响窗口覆盖和完整 bag 更新，比较增量结果与同配置全量重编码。
8. CLI 使用显式配置，tracker 默认 offline／disabled。短程运行记录有限数值、吞吐和显存。

按本次实施范围执行检查，实际验证见[当前状态](../project/status.md)。

## 5. 目标 CLI 与交付

待实现模型入口统一为 `cellweft pretrain`；数据检查、蛋白 bank、smoke、export-embeddings、apply-edit、compare-embeddings 和 profile 按职责提供入口。现有数据脚本继续按[准备实现说明](../../scripts/data/preparation/README.md)使用。

交付列出实际文件改动、可执行命令、检查结果、依赖与参数版本、短程资源结果及下一步；模型与训练完成状态以实际运行结果为准。
