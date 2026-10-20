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
| `processed/<study>/` | 共享归一化图像、可复用或生成的 mask、cell tables、索引、归一化参数和 manifest |
| `resources/protein_banks/<bank_version>/` | 固定蛋白表示及其来源与映射 |
| `resources/knowledge/<resource>/` | 按资源名称组织的知识表及配套文件 |
| `resources/pretrained/<provider>/<model_version>/` | 外部预训练权重 |
| `training/prepared/<preparation_id>/` | 训练采样索引与准备产物 |
| `training/runs/<experiment_id>/<run_id>/` | 解析配置、运行记录、checkpoints、日志和 profiling |
| `embeddings/<checkpoint_id>/<dataset_version>/<export_id>/` | 可复用的正式细胞、邻域和区域表征导出 |
| `results/<experiment_id>/<analysis_run_id>/` | 编辑集合、编辑后表征、delta、指标、图件和分析日志 |

来源强度与[数据准备](../data/preparation.md)兼容且支持局部读取的 TIFF 可直接引用；布局转换保持数值无损。来源文件保留，正式像素使用 TIFF，表格与元数据使用 Parquet、CSV、JSON。核/膜合成信号及重叠 crop 从共享图像按需读取。Study 命名和来源路径见[收集规范](../data/collection.md)。真实私有成员和患者级结果保留在 NAS，访问按数据权限控制。

用户指定的已有 NAS 数据位于项目外目录时，可在 raw 中建立目录符号链接，并在 metadata 记录原始绝对路径；原文件保留原位，收集流程只读核验。

## 准备产物文件布局

Study 是处理和 index 组织单位，`<study>` 沿用已收集数据的稳定 study 标识。正式目录为 `processed/<study>/`。

| 相对路径 | 内容 |
| --- | --- |
| `manifest.json` | study、schema、处理规则与产物身份、registry/split 引用；bank 映射完成后补齐引用 |
| `images.parquet` | 每 study 一份 index；初始七字段、每 image 一行，image_path/output_path 为目录 |
| `panel_mapping.csv` | 全部 panel 的有序来源通道、marker_id、去向及核/膜组；由读取配置选择对应 panel |
| `images/<image_id>/metadata.json` | H/W、µm/px、实际参数、mask来源/类型/检查及身份、产物引用、224/112网格规则与完成项 |
| `images/<image_id>/channels.parquet` | 全部来源通道、marker_id、q995、数值去向/原因、TIFF page 与表达列对应 |
| `images/<image_id>/image.tiff` | tiled BigTIFF float32，逻辑CYX，各通道为独立灰度page；保留有效核染和蛋白 |
| `images/<image_id>/mask.tiff` | tiled BigTIFF uint32 YX，0背景、正整数实例标签；合格来源TIFF可由metadata等价引用 |
| `images/<image_id>/cells.parquet` | cell_index、mask_label、稳定cell_id、centroid、面积、bbox、crop_origins |
| `images/<image_id>/expression.parquet` | cell_index/mask_label及有效蛋白marker_id列，float32整细胞均值 |
| `images/<image_id>/neighbors.parquet` | cell_index和有序邻居index；建议n01…n20整数列，不足填-1 |

全局语义表为 NAS `metadata/marker_registry.csv`，各 study 的 panel_mapping 引用同一 marker_id。文件级来源、完成信息及产物身份记录于 JSON；study index 保持导入入口职责。`cell_index → cells.mask_label / crop_origins` 将表达、mask及窗口关联起来；crop像素从全图TIFF读取。

TIFF 使用无损压缩，窗口读取通过页/tile偏移只解码相交tile。存储tile、分割计算块和224训练窗口分别配置；压缩、tile大小与Parquet行组经吞吐测试后确定。实例像素由bbox窗口内的mask_label定位。每图仅登记实际完成的产物，训练可用层级按[数据目录](../data/catalog.md)检查。

## 共享资源命名

1. CellWeft 自建资源目录采用小写英文、数字和下划线，按上表的资源类别组织；同一模型统一名称，例如 `dinov3`。
2. 预训练资源采用 `pretrained/<provider>/<model_version>/`。使用可识别的模型版本或结构与预训练集组合，例如 `map/mapkg_v3`、`dinov3/vitb16_lvd1689m`。同一模型的封装格式分别放在 `pytorch/`、`huggingface/`。
3. 知识资源直接使用资源名称组织，例如 `knowledge/map_kg/`；来源版本和获取日期保存在 sources 配置及元数据中。
4. 本地导出表示库使用 `<provider>_<representation>_vN`，例如 `protein_banks/map_gene_projector_v1`。该编号为 CellWeft 本地资产版本，记录在 manifest；改变输入或导出定义时创建新版本。
5. 上游发布文件保留原名；本地元数据使用 `manifest.json`、`verification.json` 等稳定名称。重命名目录时同步来源配置、manifest 输入路径、运行脚本和文档引用，保持权重与表示文件的内容一致。

## 运行与编辑结果

一个 `EXP-YYYY-NNN` 对应一个研究问题；多个 seeds、模型变体和执行各有 run ID。数据版本、模型变体、run ID 和 checkpoint 身份互相引用，不混为一个版本号。

正式运行关联代码版本、解析配置、数据/split 版本、protein bank、初始化 checkpoint 和实际命令。未提交改动参与正式运行时保存对应差异快照。相同运行恢复使用原 run；改变实验配置另建 run。

删除实验的前向、ID 对齐和空区域语义由[editing.md](editing.md)维护，缓存版本规则见[engineering.md](../development/engineering.md)。实际 cell ID 集合、编辑参数、表征和比较产物保存在 NAS 的相应 results 目录；baseline 可按其版本身份引用正式 embeddings 导出。

SSD 实验文档保存设计、结论和 NAS 结果入口。完整数组、患者级结果、权重、日志及正式图件保留在 NAS。已完成运行不原位覆盖。

## Git 跟踪范围

GitHub 只管理 SSD 工作区中代码、规则、文档和小型非敏感配置。整个 `reference/` 按用户要求排除；`third_party/` 的本地源码副本排除，来源和锁定版本在后续 sources 配置中记录。本机路径、环境、缓存、权重和数据均排除。根目录 LICENSE 来自既有远程仓库，不改变第三方资源自身许可。
