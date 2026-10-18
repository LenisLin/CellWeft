# 存储与产物规范

## 根目录

- SSD：`/home/lenislin/Experiment/projects/CellWeft`，维护代码、配置、测试、当前文档和参考资料。
- NAS：`/mnt/NAS_21T/ProjectData/CellWeft`，维护真实数据、运行产物与大型资源；不初始化 Git。
- `.cache/` 是 SSD 可清理工作缓存，NAS `cache/` 是可重建的大型缓存。缓存不是唯一正式副本。

路径模板为 [paths.example.yaml](../../configs/paths.example.yaml)，实际设置保存为忽略的 `configs/paths.local.yaml`。数据配置采用相对 NAS 根目录的路径。尚未实现运行时配置读取器。

## NAS 目录所有权

| 路径 | 内容与身份 |
| --- | --- |
| `raw/{public,private}/<dataset_id>/<source_release>/` | 保留来源语义的原始发布文件，不原位覆盖 |
| `metadata/{public,private}/` | 来源/样本清单、marker 对应表和标识映射 |
| `metadata/splits/<split_version>/` | 固定的患者划分和成员清单 |
| `interim/<dataset_id>/<preparation_id>/` | 数据准备中间产物 |
| `processed/{public,private}/<dataset_id>/<data_version>/` | 标准化图像、mask、cell tables、索引、归一化和 manifest |
| `resources/protein_banks/<bank_version>/` | 固定蛋白表示及其来源与映射 |
| `resources/knowledge/<resource>/<release>/` | 知识资源快照 |
| `resources/pretrained/<provider>/<model_version>/` | 外部预训练权重 |
| `training/prepared/<preparation_id>/` | 训练采样索引与准备产物 |
| `training/runs/<experiment_id>/<run_id>/` | 解析配置、运行记录、checkpoints、日志和 profiling |
| `embeddings/<checkpoint_id>/<dataset_version>/<export_id>/` | 可复用的正式细胞、邻域和区域表征导出 |
| `results/<experiment_id>/<analysis_run_id>/` | 编辑集合、编辑后表征、delta、指标、图件和分析日志 |

spora 已有原生格式可由 manifest 直接引用，补充 CellWeft 专用索引，不为符合示意布局重复复制大型图像。真实私有成员和患者级结果不进入 Git；public/private 目录名称不代替实际文件访问权限。

## 运行与编辑结果

一个 `EXP-YYYY-NNN` 对应一个研究问题；多个 seeds、模型变体和执行各有 run ID。数据版本、模型变体、run ID 和 checkpoint 身份互相引用，不混为一个版本号。

正式运行关联代码版本、解析配置、数据/split 版本、protein bank、初始化 checkpoint 和实际命令。未提交改动参与正式运行时保存对应差异快照。相同运行恢复使用原 run；改变实验配置另建 run。

删除实验引用 baseline 导出前核对模型、输入、归一化、空间范围及推理设置。编辑输入重新执行前向，不使用删除前的 contextual/image 缓存。实际 cell ID 集合、编辑参数和输出 ID 对齐记录保存在 NAS。区域为空时保留稳定 region ID 和 empty flag。

SSD 实验文档保存设计、结论和 NAS 结果入口。完整数组、患者级结果、权重、日志及正式图件保留在 NAS。已完成运行不原位覆盖。

## Git 跟踪范围

GitHub 只管理 SSD 工作区中代码、规则、文档和小型非敏感配置。整个 `reference/` 按用户要求排除；`third_party/` 的本地源码副本排除，来源和锁定版本在后续 sources 配置中记录。本机路径、环境、缓存、权重和数据均排除。根目录 LICENSE 来自既有远程仓库，不改变第三方资源自身许可。
