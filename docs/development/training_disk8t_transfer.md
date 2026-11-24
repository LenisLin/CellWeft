# 8T 训练数据交付与恢复

2026-11-22 已交付 123 个未压缩 POSIX/PAX tar，共 4,967,427,051,520 字节。全包结构核验覆盖 221,260 个归档成员，内容抽样覆盖 11 包 68 个文件。尚未执行全量内容比对或训练服务器恢复验收。

包内包含 `processed/`、`raw/shared/`、来源 metadata 和 manifest、纳入清单的 `results/` 以及项目代码、配置和本地参考依赖。逐项范围以盘内 `transfer_plan.json` 和 `manifests/` 为准。

1. 用 UUID `FABB-DA77` 确认目标盘及实际挂载根。本机实查为 `/mnt/disk_8T`。打开该根下的 `CellWeft/RESTORE.md`，将示例中的 `bundle` 改为当前挂载根下的 `CellWeft`。
2. 在训练服务器上选择与主 NAS 隔离、支持 Linux 文件属性的本地文件系统；解包前用 `findmnt -T <恢复根>` 核对挂载，主 NAS 共享卷不得作为解包目标。原数据根 `/mnt/NAS_21T/ProjectData/CellWeft` 仅在它指向独立本地卷时沿用；采用新根时，完成正式路径引用和版本绑定迁移。
3. 将当前 SSD 工作树的代码、配置和文档同步到服务器项目根，按当前文件清单移除已退役入口与旧讨论稿。盘内 `project_code.tar` 为交付时快照，后续修改由当前工作树提供。
4. 从主 NAS 同步现行 `processed/_catalog/public/experiments.json` 到恢复数据根的同一路径。盘内目录包仍含旧版 960 维细胞读出与检索网格；当前配置应为配套 EMA adapter 的 768 维细胞读出，S4 query／gallery 均为 224 步长、末端 padding。按[表征评价](../evaluation/representation.md)核对恢复配置。
5. 读取公共实验成员，抽查图像、细胞产物和当前路径绑定。NAS 源数据保留至恢复验收完成。
