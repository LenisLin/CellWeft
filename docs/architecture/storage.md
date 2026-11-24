# 存储与产物规范

任务结束后，将必要完成事实记入项目状态或所属规范，删除结束的检查报告、日志、可视化和过程快照。保留正式数据及恢复、接收或复现仍依赖的记录。

## 根目录

- SSD `/home/lenislin/Experiment/projects/CellWeft` 维护代码、配置、测试、文档、参考资料及可清理的 `.cache/`。
- NAS `/mnt/NAS_21T/ProjectData/CellWeft` 的 `raw/` 维护来源和外部资源，`processed/` 维护正式产物、项目资源及必要接收证据。NAS 不初始化 Git。
- 临时计算和中转使用显式配置的工作根。共享运行状态及 I/O 锁按需保留，待任务结束及读写者退出后收尾。正式产物由主 NAS 持有。

路径模板见 [paths.example.yaml](../../configs/paths.example.yaml)，本机记录保存在忽略的 `configs/paths.local.yaml`。各入口通过参数或环境变量读取路径，按入口说明传入；数据配置默认采用相对 NAS 根的路径。

## 迁移与收尾

1. 冻结精确 study、图像、成员和适用版本，列明旧路径、目标、当前读者和锁域，沿用已验证的接收及依赖证据。关闭或协调受影响的下载、在途发布及冻结读者。主 NAS inputs、D01 和权威元数据按各自合同更新，接收下游产物时保留权威表行和原始下载。
2. 同文件系统的已核定同源文件可用硬链接，跨文件系统的唯一文件先复制并读回。校验正式目标、接收记录、引用和当前 reader 后才退役确实已覆盖的原副本。保留项目外原件与未接收的唯一内容。已接收且读者已切换的副本可独立退出。
3. 涉及已冻结记录时，保留原接收回执和所需索引，以新旧版本和当前 study 汇总建立迁移接收绑定。处理绑定相同可复用科学产物和核验，处理依赖变化按实际影响使完成项失效。切换失败保留原入口和必要状态，退役后使用当前正式入口。
4. 来源 range 接收默认拒绝新建或续做需临时开放 `0666` 的发布。已完成 journal 可重放校验 manifest、目标身份和权限，未完成事务仍依赖原来源。保留活动锁的路径与身份，待使用者退出后按精确范围收尾。

## 来源 metadata 单文件归并

将迁移纳入的来源附件归入 `raw/public/<study>/metadata.json` 或 `raw/private/<study>/metadata.json`，保持作者原始文件、项目登记和科学产物各自的来源语义。
合并文件使用 `cellweft.source_metadata.v1`：JSON 保留结构化文档及不同来源命名空间，UTF-8 表格／文本保留完整文本，Parquet、PDF、工作簿及来源片段保留原始字节的 base64；索引保存来源长度、mtime及原始成员键。按原字节等价去重，不能以同名或长度相同推断重复。写入、持久化、逐成员字节读回并完成当前读者切换后，才精确退役旧文件和空目录；不留兼容软链。

唯一公共 marker 表归 `raw/shared/metadata.json#shared/marker_registry.csv`。跨 study 检索／候选证据归 `raw/public/_catalog/metadata.json`；私有共享表与病例分组规则归 `raw/private/_shared/metadata.json`，私有成员不进入 public。历史命名空间只有 metadata 时，owner 明确记录原图是否存在、是否仍在资源表；不由 metadata 迁移恢复下载、训练资格或此前撤回的数据。

来源读取由 `src/cellweft/data/source_metadata.py` 统一处理：按索引读出成员，既有文件型 reader 需要目录时只在 SSD 临时展开，保留原字节及 mtime，进程结束删除临时目录。正式来源引用使用 `raw/.../metadata.json#<成员键>`，来源跟进写回通过同一合并接口发布；不重新创建 NAS `metadata/<study>/`。既有不可变验收及图像绑定中的旧路径作为历史来源保留，通过成员键和归并回执承接。

## 公共原始来源目录

`raw/public` 的未压缩目录与 `tar.gz` 使用相同内容层级。

```text
raw/public/<canonical_study>/
  images/       原始影像、影像归档及不可拆分的混合发布包
  masks/        作者发布的原始mask及其归档
  metadata/     作者发布的panel、样本表、临床附件和说明
  manifest.json 来源身份、成员对应及迁移关系
```

仅建立有内容的类别。类别下按必要的采集／样本结构组织；多个真实来源、不同发布版本或重名文件需要区分时，再增加稳定的`source_id`／`release`层，并在manifest解释其含义。保留原始文件名、原始发布包及Zarr等格式所需内部结构，不为目录统一展开压缩包或重写原始数据。尚未完成的来源片段放在相应来源下的`incomplete/`，记录实际覆盖和未完成状态，不能冒充完整原始文件。

metadata-only 迁移使用不可覆盖的 completion successor 链。首代为 `.successor.json`，后续用 `.successor.2.json` 等，最多 32 代。发布显式绑定前一版本，在发布锁内完成 CAS。读取核对逐代前驱、原图像范围和当前 scope／summary。重试只接受已发布的同一内容。

## 原始来源归档与恢复

1. 先核对 study 的来源身份、当前引用和完整成员集合，归并原始图像、作者 mask 和来源附件。原发布包准确覆盖的可还原副本去重。归档收入已核定的实际文件内容，保留原件及格式必需的目录结构。
2. 在写入结束并冻结成员后生成临时 `tar.gz`。`manifest.json` 保存 canonical study、来源发布身份、原路径到成员的映射、成员字节数及 `mtime_ns`、必要采集／格式信息和接收证据。PAX 保存原始时间精度。完整归档及冻结成员通过读回后发布 `raw/public/<study>.tar.gz` 并核对当前读者，之后再退役覆盖的展开目录。未闭合来源继续按其实际范围登记。
3. 按 manifest 在指定 scratch 恢复所需成员，核对路径、大小和 PAX 时间，恢复纳秒 `mtime_ns`，并验证现有版本绑定与读者。中断时保留尚未闭合的原始输入和恢复状态，继续使用已发布的权威入口。
## NAS 目录所有权

| 路径 | 内容与身份 |
| --- | --- |
| `raw/public/<study>/{images,masks,metadata}/` | 按上述统一层级保留公共原始发布文件；来源版本按需分层，不原位覆盖 |
| `raw/private/<dataset_id>/<source_release>/` | 保持私有来源权限和来源语义，不混入公共整理范围 |
| `raw/{public,private}/<study>/metadata.json` | 每数据集唯一合并来源 metadata，保留来源命名空间及附件原字节 |
| `raw/shared/metadata.json`、`raw/public/_catalog/metadata.json`、`raw/private/_shared/metadata.json` | 分别承接公共 marker、跨study候选／检索事实及私有共享规则 |
| `processed/<study>/` | 两级输入metadata、panel映射、独立输入图像及后续共享图像、mask、cell tables、索引、处理参数与study向量 |
| `processed/_catalog/public/` | 现行公共study、候选及交接登记表 |
| `processed/_catalog/public/mask_bindings/` | 现行mask登记与必要来源质控事实；沿用原唯一发布锁身份 |
| `processed/_resources/protein_banks/<bank_version>/` | 项目导出的共享基因／蛋白表示、来源及行键 |
| `raw/shared/knowledge/<resource>/` | 外部知识表及必要配套文件 |
| `raw/shared/pretrained/<provider>/<model_version>/` | 外部预训练权重及必要来源信息 |
| `cache/`、`runtime/` | 临时接收、共享锁与恢复状态，仅在实际事务或依赖存续时使用 |

来源强度与[数据准备](../data/preparation.md)兼容且支持局部读取的 TIFF 可直接引用；布局转换保持数值无损。来源文件保留，正式像素使用TIFF，表格与元数据使用Parquet、CSV、JSON，study固定向量使用`marker_embeddings.pt`。核/膜合成信号及重叠 crop 从共享图像按需读取。Study 命名和来源路径见[收集规范](../data/collection.md)。真实私有成员和患者级结果保留在 NAS，访问按数据权限控制。

## 准备产物文件布局

Study 是处理和 index 组织单位，`<study>` 沿用已收集数据的稳定 study 标识。正式目录为 `processed/<study>/`。

| 相对路径 | 内容 |
| --- | --- |
| `images.parquet` | 完整image metadata及处理入口；每个独立ROI／TMA core／FOV一行，包含区域注释及sample/panel关联 |
| `sample_metadata.csv` | 每份生物学取样一行，包含患者信息与来源明确的临床摘要 |
| `patient_metadata.csv` | 当前两个胃癌队列的精简英文冻结患者全集；仅保留一份CSV，承接尚不能全部由既有sample表达的患者，不增加重复Parquet或来源记录槽 |
| `panel_mapping.csv` | 每study一份；按panel_id保存全部来源行、marker身份、核/膜选用、keep_channel及输出位置；后续向量阶段补embedding_row |
| `field_mapping.csv` | 来源字段、连接与转换规则；扩展字段定义及类型、单位和时点 |
| `review_results.csv` | 独立校对、修补轮次、修改动作、复查及最终验收结论 |
| `images/<image_id>/input.tiff` | 每image单独保存筛选后CYX TIFF，默认方形；明确批准去除人工补边的原生矩形按input规范的canvas_policy声明。保留来源解码dtype、强度及尺度，供后续图像预处理 |
| `images/<image_id>/valid_pixels.tiff` | 来源框内存在未采集位置时保存uint8 YX有效域；实测为1，未采集及padding为0 |
| `images/<image_id>/sample_region.tiff` | 矩形框包含相邻sample时保存uint8 YX处理区域；本sample区域1，邻sample及padding为0；由image扩展列sample_region_path引用 |
| `marker_embeddings.pt` | 通道筛选后的独立步骤生成；float32向量、marker_ids、mapping_expressions及来源规则metadata |
| `images/<image_id>/metadata.json` | 图像几何、实测有效域依据、实际处理参数、mask来源／转换／检查及最终身份、产物引用、网格及采样规则、可采样数量、完成项与缺口 |
| `images/<image_id>/channels.parquet` | 全部来源通道及三层位置、marker_id、实际用途、q995、数值去向／原因、共享TIFF page和表达列 |
| `images/<image_id>/image.tiff` | tiled BigTIFF float32，逻辑CYX；保存数值有效的downstream及所选分割专用通道，实测区域外为0 |
| `images/<image_id>/mask.tiff` | tiled BigTIFF uint32 YX，0背景、正整数实例标签；与input局部坐标一致，实测区域外为0。无需转换且符合最终合同的来源TIFF可由metadata引用，其余生成派生TIFF并保留原文件 |
| `images/<image_id>/cells.parquet` | cell_index、mask_label、稳定cell_id、centroid、面积、bbox、crop_origins |
| `images/<image_id>/expression.parquet` | cell_index/mask_label及有效蛋白marker_id列，float32整细胞均值 |
| `images/<image_id>/neighbors.parquet` | cell_index和有序邻居index；建议n01…n20整数列，不足填-1 |
| `annotations/source_cells.parquet` | 已取得的作者逐细胞注释，保留来源细胞身份、原始类别和连接字段 |
| `annotations/cell_labels.parquet` | 按`image_id`与当前`cell_id`／`mask_label`连接的细胞标签；原始类别、映射方法及未赋值项随表保留 |
| `annotations/metadata.json` | 注释来源文件、字段与连接方法、表格行数和本次实际匹配范围 |

全局marker表位于`raw/shared/metadata.json#shared/marker_registry.csv`，由S1及来源metadata派生，study的panel_mapping引用同一marker_id。输入文件按[逐study流程](../data/study_input.md)生成，字段由[CSV模板](../data/templates/study_input/README.md)维护；image_metadata.csv是字段模板，实际image表保存为images.parquet。后续处理参数、产物引用及完成项写入逐图metadata.json，cell_index通过cells.mask_label及crop_origins关联表达、mask和窗口。

作者细胞标签按实际取得和对应情况建立`annotations/`中的文件；已完成当前实例连接的记录用于细胞注释任务。原始附件保存在对应`raw/public/<study>/metadata/<source_release>/`并由来源manifest登记。来源细胞表与当前实例表分开保存，避免将作者源细胞数量当作已匹配数量。

公共 image metadata 以 study 的 `images.parquet` 按 `(study,image_id)` 逐行维护。排除图像时更新对应行、当前索引及必要计数；保留图的实际输入、参数和科学产物未变时，复用已有处理与核验证据。共享表文件版本、组成员或交付引用变化本身不触发保留图重算或整批重新验收。运行时派生清单不能自行增加用户已确认范围之外的验收任务。
`processing_summary.json` 汇总当前来源入口及版本、冻结图像范围、正式技术产物、接收及核验证据、未检查事项与待解决缺口。必要的当前接收索引、批准及迁移回执保存在 `processed/`，并由实际接收及恢复引用定位。
Study向量在输入通道筛选后由本机CPU单独导出。基础库版本与路径、S1路径、资产身份、marker数量、维度、dtype及聚合／归一化规则附于marker_embeddings.pt的metadata，映射表达式随向量保存。由本机主控单写并按以下顺序发布：

1. 临时生成向量及带embedding_row的panel映射，关闭后重新加载，核对marker顺序、表达式与全部非空行号。
2. 更新期间暂停该study向量的并发消费；更新已有资产时，先撤下正式向量文件，避免旧向量与新行号混用。
3. 替换已核对的panel映射，最后发布新向量文件；重新加载核对对应后恢复消费。中断后缺少正式向量的study保持向量待完成，输入metadata及图像产物保留。

向量依赖本study采用的downstream marker身份、映射表达式、基础库版本及聚合／归一化规则；仅向量或embedding_row更新时保留有效像素、mask、表达及几何。输入实测有效域改变时，重新计算受影响通道的q995和共享图像，更新组织估计与采样结果，并沿实际依赖更新分割合成信号、预测mask及表达。最终mask或其几何／实测域约束改变时，更新mask身份并重建cells、expression、neighbors及crop对应。仅采样阈值或组织估计方法改变时，更新窗口选择、缓存及采样metadata，保留完整crop对应和细胞产物。来源像素、通道去向、marker身份或分割配置变化时，重建实际受影响的产物；完成记录按当前依赖核对。

TIFF使用无损压缩，窗口读取只解码相交tile；存储tile、分割计算块和训练crop分别配置，布局参数按吞吐验收确定。全图保存像素，crop与训练任务视图按需读取／生成；cells保存完整crop_origins，组织比例和采样选择可缓存，不新增正式逐crop审计表。每图先关闭并核对临时结果及其直接依赖。替换已发布产物前，由本机先使受影响的完成项失效并暂停相关消费，待在途读取结束后再替换；本次实际发布的产物重开核对通过后，最后更新正式引用和完成记录，再恢复消费。中断时受影响项保持未完成，未受影响产物继续有效；训练可用层级按[数据目录](../data/catalog.md)检查。

单机／文件交接实现用逐image `.input.lock`协调上游input替换与下游整图读取；使用逐image `.preparation.lock` 保证任务单写，`.publication.lock`保证受控读者与发布者跨进程互斥；study级同名publication锁协调输入表和向量。它们是运行时锁文件，不作为科学产物或完成记录。消费者持有读锁直至所用文件句柄关闭，并核对当前请求版本及completed；逐产物状态为complete／partial／pending。检测依赖变更即撤销受影响完成项；当前segmentation_task.task_id持久化表示有效attempt，错误精度／任务回执不能接续。完成判定另重开核对产物直接关系。

## 实验成员组织（2026-11-19）

科学资产保存在`processed/<study>/`。正式实验入口为`processed/_catalog/public/experiments.json`：datasets保存共同图像分组、固定划分及panel输入marker，tasks保存各项实验的成员、目标marker、标签来源与读出设置；paths保存相对于data_root的资产路径模板。

该文件原位维护当前安排，取消实验与划分的迭代版本编号，患者分配及任务成员直接以文件内容为准。

逐 ROI 质控决定作用于这个实验入口。弃用 ROI 从共同成员及任务显式 images 中退出，原科学资产保留；等待替换 mask 验收的 ROI 同样暂停使用旧分割，保留其原 group／split 以便恢复。保留 ROI 不重新划分。`quality_control.decisions` 指向用户决定，`inactive_rois` 保存退出原因和原划分。

当前公共质控决定统一维护在`processed/_catalog/public/quality_control.json`，由`experiments.json`引用；逐ROI尺寸检查、最终处置及padding验证合并在这份记录中。小图质控PNG使用固定目录`processed/_catalog/public/quality_control/small_fov/`。按用户2026-11-20要求原位维护当前记录，不另存每轮完整manifest、候选清单、重复报告或过程日志；临时执行材料放SSD或`/tmp`，完成后清理。清理保留现有HTML直接引用的附件及原始科学资产。

仅排除某 ROI 的测量通道时，`datasets[study].excluded_markers_by_image[image_id]` 保存 marker_id。实验消费者须在解析逐图可用通道后，对图像输入、细胞表达列、目标候选／valid_targets、隐藏信号、损失与评分统一扣除这些 marker；panel 和其他 ROI 保持。源 TIFF 页、表达列及来源表保留，不能仅凭非空 tiff_page 绕过该规则。保留旧 mask 时明确记录其历史分割通道依赖；当前测量排除不表示已消除该依赖。新 mask 另存候选，验收并重建相应 cells、expression、neighbors、crop 对应后才恢复依赖该 mask 的实验消费。

组织区域标签由`tasks.tissue_classification.labels`维护：`definitions`保存来源代码的中英文含义及各固定划分的图像／患者数，`source`保存原始表、字段及连接键，`spatial_scope`保存标签范围与定位字段。逐图标签按`image_id`读取原study的`images.parquet.region_annotation`；组织检索通过`labels_ref`复用同一份定义。

检索使用 `tasks.tissue_retrieval` 保存固定成员、执行设置及临床字段来源。实际病例值从 NAS 来源表读取。成员消费与算法规则见[表征评价](../evaluation/representation.md)，恢复使用当前正式 JSON。

## 共享资源命名

1. CellWeft 自建资源目录采用小写英文、数字和下划线，按上表的资源类别组织；同一模型统一名称，例如 `dinov3`。
2. 预训练资源采用 `pretrained/<provider>/<model_version>/`。使用可识别的模型版本或结构与预训练集组合，例如 `map/mapkg_v3`、`dinov3/vitb16_lvd1689m`。同一模型的封装格式分别放在 `pytorch/`、`huggingface/`。
3. 知识资源直接使用资源名称组织，例如 `knowledge/map_kg/`；来源版本和获取日期保存在 sources 配置及元数据中。
4. 本地导出表示库使用 `<provider>_<representation>_vN`，例如 `protein_banks/map_gene_projector_v1`。该编号为 CellWeft 本地资产版本，记录在 manifest；改变输入或导出定义时创建新版本。
5. 上游发布文件保留原名；本地元数据使用 `manifest.json`、`verification.json` 等稳定名称。重命名目录时同步来源配置、manifest 输入路径、运行脚本和文档引用，保持权重与表示文件的内容一致。

## 运行与编辑结果

一个 `EXP-YYYY-NNN` 对应一个研究问题；多个 seeds、模型变体和执行各有 run ID。数据版本、模型变体、run ID 和 checkpoint 身份互相引用，不混为一个版本号。

正式运行关联代码版本、解析配置、数据/split 版本、protein bank、初始化 checkpoint 和实际命令。未提交改动参与正式运行时保存对应差异快照。相同运行恢复使用原 run；改变实验配置另建 run。

删除实验的前向、ID 对齐和空区域语义由[editing.md](editing.md)维护，缓存版本规则见[engineering.md](../development/engineering.md)。实际 cell ID 集合、编辑参数、表征和比较产物保存在 NAS `processed/` 下相应正式目录；baseline 可按其版本身份引用正式表征导出。

SSD 实验文档保存设计、结论和 NAS 结果入口。完整数组、患者级结果、权重、日志及正式图件保留在 NAS。已完成运行不原位覆盖。

## Git 跟踪范围

GitHub 管理 SSD 工作区中的代码、规则、文档和小型非敏感配置。固定源码位于 Git 排除的 `reference/`，来源和锁定版本见 [sources.yaml](../../configs/sources.yaml)。本机路径、环境、缓存、权重和数据均排除。根目录 LICENSE 来自既有远程仓库，第三方资源沿用自身许可。
