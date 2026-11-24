# 项目架构与目录职责

采用 SSD 代码与文档工作区、NAS 数据与产物工作区。目录集合由 [layout.json](../../configs/layout.json) 定义；空目录通过 [bootstrap_layout.py](../../scripts/bootstrap_layout.py) 恢复，不使用占位文件。

## 代码职责

下表定义代码模块职责。实施进度见[当前状态](../project/status.md)。

| 路径 | 职责 |
| --- | --- |
| `src/cellweft/data/` | 数据读取、标准化、稳定 ID、空间索引、几何变换及 batch 组装 |
| `src/cellweft/models/` | Protein-aware tokenizer、ViT-B/8、Mask-Adapter 细胞读出、marker 条件化表达头及共享像素 decoder |
| `src/cellweft/objectives/` | DINO、iBOT、KoLeo、whole-cell MSE 及冻结编码器后的像素重建 MSE |
| `src/cellweft/optimization/` | 优化器、调度和可训练参数组 |
| `src/cellweft/training/` | 联合预训练与冻结读出训练生命周期、DINO TaskViews、配套 EMA 与 checkpoint 恢复 |
| `src/cellweft/interventions/` | Marker/cell/niche 选择、图像像素与实例视图编辑 |
| `src/cellweft/inference/` | 按需表征导出、baseline/edited 重新前向与序列化 |
| `src/cellweft/metrics/` | 可复用的指标数学与归约 |
| `src/cellweft/evaluation/` | 固定模型的表征/删除评价、患者级统计与汇总 |
| `src/cellweft/integrations/` | DINOv2、Mask-Adapter、VirTues 参考实现及 spora 数据接口 |
| `scripts/` | 数据来源和实验特定入口；复用源码实现 |
| `tests/` | 单元、空间/编辑合同和外部集成验证；fixtures 仅放小型合成输入 |

图像准备的共享实现位于`src/cellweft/data/{io,preparation}/`，记录与读取接口由`data/contracts.py`定义。Dataset入口位于`scripts/data/preparation/`，按顶层顺序解释来源、建立study index并调用单图pipeline；算法模块执行计算，pipeline统一决定复用和依赖重算，writer负责保存与发布。函数职责、配置及命令见[准备入口](../../scripts/data/preparation/README.md)。

源代码复用方向为 scripts 调用源码。模型 forward 负责表示计算，原始数据编辑和产物发布由相应模块编排。

## 配置与接口

`configs/data/`、`model/`、`train/`、`optimization/`、`evaluation/`、`edits/` 分别维护对应选择；`experiments/` 组合实验设置。本机路径记录可保存于 Git 忽略的 `paths.local.yaml`；实际执行按入口参数或环境变量传入，并保存解析后的配置。
Batch、ModelOutput 和 EditSpec 的职责分别见[数据合同](data_contract.md)、[模型规范](model.md)和[编辑规范](editing.md)。目标与生命周期见[训练规范](training.md)，实施及验收见[工程规范](../development/engineering.md)。运行 ID、checkpoint schema 和下游任务配置分别管理，保存实际接口版本。
