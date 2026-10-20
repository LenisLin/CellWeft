# 初版项目架构

采用 SSD 代码与文档工作区、NAS 数据与产物工作区。目录集合由 [layout.json](../../configs/layout.json) 定义；空目录通过 [bootstrap_layout.py](../../scripts/bootstrap_layout.py) 恢复，不使用占位文件。

## 代码职责

| 路径 | 职责 |
| --- | --- |
| `src/cellweft/data/` | 数据读取、标准化、稳定 ID、空间索引、几何变换及 batch 组装 |
| `src/cellweft/models/` | Protein-aware PatchEmbed、Intrinsic/Neighborhood encoders、Fusion 与 readout |
| `src/cellweft/objectives/` | Marker prediction、masked-cell latent prediction 和单一 alignment 目标 |
| `src/cellweft/optimization/` | 优化器、调度和可训练参数组 |
| `src/cellweft/training/` | Cell/joint 训练生命周期、冻结策略、EMA 与 checkpoint 恢复 |
| `src/cellweft/interventions/` | Marker/cell/niche 编辑定义、选择与双视图一致编辑 |
| `src/cellweft/inference/` | 表征导出、baseline/edited 重新前向与序列化 |
| `src/cellweft/metrics/` | 可复用的指标数学与归约 |
| `src/cellweft/evaluation/` | 表征/删除 benchmark 组织、统计与汇总 |
| `src/cellweft/integrations/` | 官方 DINOv3、spora 和参考模型适配 |
| `scripts/` | 数据来源和实验特定入口；复用源码实现 |
| `tests/` | 单元、空间/编辑合同和外部集成验证；fixtures 仅放小型合成输入 |

图像准备的共享实现位于`src/cellweft/data/{io,preparation}/`，记录与读取接口由`data/contracts.py`定义。Dataset入口位于`scripts/data/preparation/`，按顶层顺序解释来源、建立study index并调用单图pipeline；算法模块执行计算，pipeline统一决定复用和依赖重算，writer负责保存与发布。函数职责、配置及命令见[准备入口](../../scripts/data/preparation/README.md)。

源代码不反向导入 scripts。模型 forward 不负责改写原始数据或发布产物。图像预训练调用锁定的 DINOv3 官方工程；cell/joint 训练独立组织，不重新实现一套简化 DINO。

## 配置与接口

`configs/data/`、`model/`、`train/`、`optimization/`、`evaluation/`、`edits/` 分别维护对应选择；`experiments/` 组合实验设置。本机路径由不提交的 `paths.local.yaml` 维护。正式执行保存解析后的配置。

后续按需实现 Batch、ModelOutput、EditSpec 等类型与 CLI。字段与几何见[data_contract.md](data_contract.md)，结构与 V2 接口见[model.md](model.md)，目标与生命周期见[training.md](training.md)，输入编辑与输出见[editing.md](editing.md)。实施、验收及目标 CLI 见[engineering.md](../development/engineering.md)；文档定义不表示接口已实现。

普通消融通过配置实现；不兼容架构候选出现时再建立独立实验模块和 checkpoint 身份。模型变体、运行 ID 和 checkpoint schema 分别管理。
