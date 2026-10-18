# CellWeft（细胞织域）

CellWeft 研究空间蛋白的多尺度表征，以及 marker、细胞和细胞群删除后目标细胞与微环境的模型响应。设计包含固定蛋白知识表示、DINOv3 图像基座、Intrinsic/Neighborhood 细胞基座和递归 Fusion。

当前交付为项目架构与协作基础：已整理研究文档并建立 SSD/NAS 目录。模型、数据管线和训练尚未实现。

## 开始工作

1. 阅读 [AGENTS.md](AGENTS.md) 和 [当前状态](docs/project/status.md)。
2. 按 [文档索引](docs/README.md) 阅读相关规范。
3. 从新 checkout 恢复目录（Python 3.10+）：

```bash
python3 scripts/bootstrap_layout.py
# 同时创建 NAS 目录时显式指定根路径：
python3 scripts/bootstrap_layout.py --data-root /mnt/NAS_21T/ProjectData/CellWeft
```

Git 不保存空目录；[目录清单](configs/layout.json)和创建脚本用于重建结构，不添加占位文件。本机路径从 [paths.example.yaml](configs/paths.example.yaml) 复制到 `configs/paths.local.yaml` 后调整，实际文件不提交。

## 组织方式

| 位置 | 用途 |
| --- | --- |
| `docs/` | 现行研究计划、架构、决策和实验记录 |
| `configs/` | 数据、模型、训练、评价和编辑配置 |
| `src/cellweft/` | 后续可复用实现 |
| `scripts/` | 数据准备、启动、实验与绘图入口 |
| `tests/` | 单元、合同、集成验证和小型合成 fixtures |
| `reference/` | 本地参考论文和代码，整个目录排除 Git |
| `third_party/` | 后续锁定的运行依赖，本地 checkout 排除 Git |
| `/mnt/NAS_21T/ProjectData/CellWeft` | 数据、表征、权重、中间结果和运行产物 |

详细分工见 [架构概览](docs/architecture/overview.md)和 [存储规范](docs/architecture/storage.md)。研究方案见 [research_plan.md](docs/project/research_plan.md)，原始初始化 Prompt 已[归档](docs/archive/README.md)。

## 版本管理

远程仓库：[LenisLin/CellWeft](https://github.com/LenisLin/CellWeft)。只管理 SSD 项目内容，NAS 不纳入 Git。提交标题采用英文 `type(scope): description`，正文使用英文并按不同文件批次分别说明，规则见 [开发流程](docs/development/workflow.md)。

仓库沿用远程已有 [LICENSE](LICENSE)。第三方论文、代码、权重和数据遵循各自许可。
