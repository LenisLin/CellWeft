# CellWeft（细胞织域）

CellWeft 以空间蛋白图像为主要输入，以具有稳定身份和空间位置的细胞实例为读出与编辑对象。图像主干产生空间 patch 特征和 crop 表征，实例适配器读取细胞表征。MIL 将同一图像的 crop 向量汇总为图像表征，或按患者和任务汇总为患者表征。项目通过蛋白恢复、细胞注释、组织任务、胃癌治疗响应分析和空间编辑研究这些表征的用途。外部实测蛋白扰动指导的编辑作为独立扩展，见[研究计划](docs/project/research_plan.md#31-figure-15-研究主线与输入输出)。

已建立研究/工程规范，并实现可复用图像预处理、mask 检查、细胞分割、蛋白定量及索引保存，使用方式见[数据准备入口](scripts/data/preparation/README.md)。模型和联合训练仍待代码接线；代码实现、真实训练和数据处理验证范围见[当前状态](docs/project/status.md)。

## 开始工作

1. 阅读 [AGENTS.md](AGENTS.md) 和 [当前状态](docs/project/status.md)。
2. 按 [文档索引](docs/README.md) 阅读相关规范。
3. 从新 checkout 恢复目录（Python 3.10+）：

```bash
python3 scripts/bootstrap_layout.py
# 同时创建 NAS 目录时显式指定根路径：
python3 scripts/bootstrap_layout.py --data-root /mnt/NAS_21T/ProjectData/CellWeft
```

Git 不保存空目录。[目录清单](configs/layout.json)和创建脚本用于重建结构。需要记录本机路径时，可复制 [paths.example.yaml](configs/paths.example.yaml) 为 Git 忽略的 `configs/paths.local.yaml` 后调整。当前入口使用各自的参数或环境变量，按相应入口说明传入路径。

## 组织方式

| 位置 | 用途 |
| --- | --- |
| `docs/` | 现行研究计划、架构、决策、实验记录与验证状态 |
| `configs/` | 数据、模型、训练、评价和编辑配置 |
| `src/cellweft/` | 可复用数据处理实现；模型与训练模块尚待接线 |
| `scripts/` | 数据准备、启动、实验与绘图入口 |
| `tests/` | 单元、合同、集成验证和小型合成 fixtures |
| `reference/` | 本地参考论文、固定源码及 reader 使用的 vendor，整个目录排除 Git |
| `/mnt/NAS_21T/ProjectData/CellWeft` | 数据、表征、权重、中间结果和运行产物 |

研究目标见 [research_plan.md](docs/project/research_plan.md)，专项规范见[文档入口](docs/README.md)，模块和物理保存见[架构概览](docs/architecture/overview.md)与[存储规范](docs/architecture/storage.md)。

## 版本管理

当前开发版本为 `0.2.0`，版本源为 `pyproject.toml`，发布标签为 `v0.2.0`。

远程仓库：[LenisLin/CellWeft](https://github.com/LenisLin/CellWeft)。只管理 SSD 项目内容，NAS 不纳入 Git。提交标题采用英文 `type(scope): description`，正文使用英文并按不同文件批次分别说明，规则见 [开发流程](docs/development/workflow.md)。

仓库沿用远程已有 [LICENSE](LICENSE)。第三方论文、代码、权重和数据遵循各自许可。
