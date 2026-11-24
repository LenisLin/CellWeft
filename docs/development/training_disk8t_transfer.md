# 8T 训练数据交付与恢复

2026-11-22 已交付 123 个未压缩 POSIX/PAX tar，共 4,967,427,051,520 字节。全包结构核验覆盖 221,260 个归档成员，内容抽样覆盖 11 包 68 个文件。尚未执行全量内容比对或训练服务器恢复验收。

包内包含 `processed/`、`raw/shared/`、来源 metadata 和 manifest、纳入清单的 `results/` 以及项目代码、配置和本地参考依赖。逐项范围以盘内 `transfer_plan.json` 和 `manifests/` 为准。

2026-11-24 的 Feng 补充沿用单份数据加成员清单的布局：`processed__Feng2026_IMC_BladderCancer.tar` 为保留的 515 图；`processed___catalog_update_20261124.tar` 更新当前实验成员；`feng2026_source_metadata_20261124.tar` 保存来源元数据、当前来源 mask 和精确验收证据；`project_code_update_20261124.tar` 为当前代码、配置和文档。补充完成状态、各包大小及抽样范围读取 `manifests/feng2026_supplement_20261124.json`，各包沿用 `.copied.json` 回执。

Feng 的训练／内部验证分别为 50 位患者／421 图与 12 位患者／94 图，当前 `experiments.json` 的 `foundation_pretraining.datasets.Feng2026_IMC_BladderCancer.members` 只选择训练 split，`validation_members` 只选择验证 split。图像保存单份，全部 ROI 和窗口继承患者分组；组织采样仍为 `pending`。原始 MCD、TXT、RDS 留在 NAS。

1. 用 UUID `FABB-DA77` 确认目标盘及实际挂载根。本机实查为 `/mnt/disk_8T`。打开该根下的 `CellWeft/RESTORE.md`，将示例中的 `bundle` 改为当前挂载根下的 `CellWeft`。
2. 在训练服务器上选择与主 NAS 隔离、支持 Linux 文件属性的本地文件系统；解包前用 `findmnt -T <恢复根>` 核对挂载，主 NAS 共享卷不得作为解包目标。原数据根 `/mnt/NAS_21T/ProjectData/CellWeft` 仅在它指向独立本地卷时沿用；采用新根时，完成正式路径引用和版本绑定迁移。
3. 先恢复 `project_code.tar`，再覆盖代码更新包；按 `manifests/retired_project_paths_20261124.json` 移除旧包中已退役的项目文件。盘内 `RESTORE.md` 给出完整顺序。后续修改再由当前 SSD 工作树提供。
4. 恢复 `processed__*.tar` 时按文件名字典序执行，使 `processed___catalog_update_20261124.tar` 覆盖旧 catalog；后续再与主 NAS 现行 `processed/_catalog/public/experiments.json` 比对。当前配置应为配套 EMA adapter 的 768 维细胞读出，S4 query／gallery 均为 224 步长、末端 padding。按[表征评价](../evaluation/representation.md)核对恢复配置。
5. 读取公共实验成员，抽查图像、细胞产物和当前路径绑定。NAS 源数据保留至恢复验收完成。
