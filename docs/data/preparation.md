# 数据准备流程

本页维护全局 marker 统一、study 准备和逐 image 处理的操作规则。字段及空间对应见[数据合同](../architecture/data_contract.md)，正式文件布局见[存储规范](../architecture/storage.md#准备产物文件布局)，实际进度见[项目状态](../project/status.md)。

共享代码、dataset入口、配置及运行命令见[准备实现说明](../../scripts/data/preparation/README.md)。单图pipeline维护复用和部分完成，科学规则以本页为准。

执行根、队列、锁域及续跑入口见[准备脚本](../../scripts/data/preparation/README.md)，产物接收见[存储规范](../architecture/storage.md)。

## 执行与验收合同

按当前已批准范围验收，以 `study + image_id` 核对保留图像的绑定和产物。分别记录完整 study、部分完成的具体缺项及另需批准排除的范围。Mo 与 Radtke 的已确认原生图像框、padding 和有效抽样证据可继续复用。仍在准备的图像按实际状态记录。

| 层级 | 必需交付 |
| --- | --- |
| Study | `images.parquet`、`panel_mapping.csv`、`sample_metadata.csv`、`field_mapping.csv`、`review_results.csv`、`marker_embeddings.pt` |
| Image | `input.tiff`、`image.tiff`、`channels.parquet`、`mask.tiff`、`cells.parquet`、`expression.parquet`、`neighbors.parquet`、`metadata.json` |
| 条件性附件 | 已有或适用时核对`valid_pixels.tiff`、`sample_region.tiff`及其坐标；普通补边以padding字段表达 |
| 实例与 crop | `cells.parquet` 保存 `cell_index`、`mask_label`、稳定 `cell_id` 和 `crop_origins`。按 §7 重建窗口 |

内容检查覆盖来源、通道、图像几何、实测范围、P99.5归一化、mask与细胞索引／表达／邻域／crop的一致性，以及marker向量与panel对应。规定产物齐全、对应正确且正式接收才计技术完成；部分完成精确列出image及缺项。组织／分割质量按既定分层抽样记录实际覆盖范围，已知异常图单独核对。未执行的检查不记通过；临床连接缺失不单独阻断技术处理；采样已执行但合格窗口为0不自动删除图像。

公共metadata逐图维护，study汇总由逐图记录生成。正确产物与有效核验证据直接复用，仅修复具体问题及其依赖；管理字段或组成员变化不触发重算或整批重新验收。每条验收结果至少包含以下字段，技术结果与质量结果分别填写：

`study｜image_id｜owner及实际路径｜纳入状态｜技术结果｜组织/分割质量结果｜具体问题或缺项｜修复结果或排除原因｜证据引用`

问题处置遵循[独立确认、持续修复与批准后排除](study_input_review.md#2-确认问题持续修复与批准后排除)。

## 1. 执行层级与输入

1. 全局：以[S1](../paper/supplementary_tables.md)维护已确认marker语义和MAP-KG映射，派生`raw/shared/metadata.json#shared/marker_registry.csv`；新增study接入同一marker身份体系。
2. Study输入：按[per-study工作流](study_input.md)确认来源和image范围，建立两级metadata、对接panel，保存每image的筛选后TIFF及study级panel文件，经独立校对、修补和验收后进入图像预处理。
3. Image：遍历images.parquet中独立ROI／TMA core／FOV，完成强度统计、mask、定量、neighbors与crop对应；image是计算、恢复和完成检查单位。
4. 保存：原始发布文件按收集记录读取，正式产物保存到 NAS `processed/<study>/`。SSD 用于有界临时缓存。

Study输入的操作顺序由[输入工作流](study_input.md)维护，字段及临床扩展规则由[模板字典](templates/study_input/README.md)维护，有限轮次的校对与修补由[独立校对流程](study_input_review.md)维护，文件位置由[存储规范](../architecture/storage.md#准备产物文件布局)维护。

## 2. Marker 清理、共享强度与分割参数

### Marker 与来源强度

1. Manual mapping 在 study/panel 层完成。以蛋白名称和实际测量形式对齐，gene symbol 作为辅助注释；同义且测量含义等价的通道共享 `marker_id`。CD45、CD45RA、CD45RO 及 total/phospho、完整/切割形式分别建立 marker；UniProt 主条目相同不足以合并测量。结合来源抗体及 UniProt 核对身份，克隆识别范围不明时保留具体缺口。
2. 通道去向以S1及已确认的panel取舍为准。准入marker用于downstream，mapping_pending保留映射缺口；所选核输入及仅为分割保留的重复raw channel可使用segmentation-only。每panel的downstream marker唯一；未选作表达测量但仍用于分割的重复通道保留像素、source_name和marker身份，记录实际用途，不生成重复表达列，其embedding_row为空。其余明确排除的通道使用exclude。核组优先DNA专用通道，也可使用经来源及图像复核的Nuc/Nuclear、总Histone H3/HH3，具体以S1为准；Ki67、FOXP3不作为通用核输入。
3. 家族、组合及修饰通道按实际测量身份整理。先合并等价别名，再按不同 study 统计覆盖；仅见于一个 study 的多靶标混合通道移入排除，多个 panel 不增加 study 数。保留的组合对应一个表达列，如 CK8/18、E/P-cadherin、vWF+CD31；Collagen 采用下述用户确认的聚合分组。已确认映射在输入通道筛选完成后的独立study级步骤查表生成向量。
4. 根据来源说明和实际加载过程确认磁盘强度。已有结果符合下述公式及统计域时直接复用；来源已完成的处理沿用其明确语义，补齐当前归一化。通道映射固定后，per-image 仅执行映射和数值处理。

语义整理阶段将来源及覆盖缺口纳入同一份按 study/panel、原通道顺序排列的 marker 对齐清单。保留全部来源通道与模型原始分类，模型曾判为非蛋白的记录可单独筛选，纠正后仍保留原判断；模型分类作为人工确认的候选依据。

查询表使用常用简名，如 FoxP3、Ki-67、E-Cadherin；蛋白全称冗长时使用 AFP、APOD、CDH11、CTNNA2 等明确缩写或 gene symbol 作为显示名。统一大小写与同义写法，去除无意义的 Protein 前缀。CD20 的 cytoplasmic 标签归入 CD20；PR 的 A/B 合测归入 PR，来源 epitope/clone 注释保留。

用户确认的聚合命名：VEGFA/VEGFC→VEGF；Collagen各类型、COL1A1/COL4A1/COL6A2、Fibronectin（含EDA-FN）及Collagen I/Fibronectin合测→Collagen；Glucose transporter→GLUT1；HLA-A1→HLA-A；HLA-DR/DP/DQ→HLA-DR；TOX/TOX2及TOX2→TOX；pH3(S28)→pH3。聚合组保留来源靶标和修饰位点，跨蛋白分组的canonical protein ID可空，向量映射以S1为准。其他按同义测量归并，ADAMTS1/5、ATP1A1/ATP5A、CA1/CA2与vWF、CD31、vWF+CD31保持区别。同图同组的多通道取舍在panel清理中确定，名称归并不隐含像素求和。

通道名中的alpha/beta/gamma等采用α/β/γ，如Integrin β、β-Tubulin、TNFα、RORγt；原始别名保持。用户指定泛Cytokeratin条目的显示名为Cytokeration。Reolysin按用户要求排除，其来源含义为病毒相关测量。明确突变／缺失形式本阶段排除；当前已确认并更新EGFRvIII、H3K27M、IDH1 R132H，来源通道记录保留。用户进一步确认排除H3K27、H3K4、H3K27M3、H3 (Lys28) trimethylate四种修饰或位点存疑的来源名称。甲基化、磷酸化的整体训练筛选仍在讨论，当前保留映射；pH3已按上述要求合并。

Biotin、明确 RNA 探针、来源已确认失败的通道，以及 Seg1/2/3、分割 mask、未指定蛋白靶标的技术通道列入排除；DNA/DAPI 等核染保留用于分割。RNA/蛋白联合成像按通道区分测量类型。同一原始别名在不同来源确指不同靶标或通道去向时，在对齐清单及查询表中使用带 study 限定的别名，保留原始名称与来源关系。覆盖低于3个study的条目作为后续筛选讨论对象，当前不按频次自动排除。

CD15、CD57、PNAD和H3K27me3明确保留；SSEA-1归入CD15，MECA-79归入PNAD。H3K4me2、H3K9me3、5hmC、CA19-9、GD2、hyaluronan、m6A、O-GlcNAc、WGA按用户决定排除。CD15/CD57的糖链表位不能直接等同于FUT4/B3GAT1酶蛋白。当前基础向量按S1的已确认映射查表；CD15、CD57、PNAD的自定义编码及H3K27me3的修饰描述作为后续扩展讨论。

### Study级marker向量查表

执行时序：先完成[study输入整理](study_input.md)的panel对接与通道筛选，再单独导出本study实际采用marker的向量；输入交付不依赖向量文件或embedding_row。

输入：NAS根目录下的`processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv`与`processed/_resources/protein_banks/map_gene_projector_v1/`。向量步骤读取S1的五个marker映射字段：`Study、Original marker、Unified marker、MAP-KG mapping、UniProt accession`；另列`Segmentation channel`由下文“新分割”使用，不参与向量计算。

1. 确定本批获准准备的study及平台范围。当前公开study从NAS `processed/_catalog/public/studies.csv`读取，结合[收集规范](collection.md)和来源metadata选择实际保留的panel；S1中的历史study记录不直接生成处理任务。确认：study、平台和panel均属于本批范围。
2. 读取输入阶段已核对的panel_mapping，按keep_channel及逐图实际采用通道确定本study的marker集合。以`Study + Original marker`核对S1；名称差异只使用来源明确的对应关系。复用marker身份及panel取舍。确认：来源行和输出位置可追溯，所选marker来自筛选后保留集合。
3. 为保留且MAP-KG mapping非空的downstream marker在所属下游节点CPU查表：单基因直接取值，`mean(...)`对成员向量等权平均，`proxy(...)`取指定基因向量；沿用原始projector输出，不追加L2归一化。确认：全部成员命中，结果为有限的float32向量；维度读取基础库manifest，当前为1,024。
4. 每study按`marker_id`排序建立唯一向量行，同一marker跨panel复用一行；不同marker即使向量相同也保留各自身份。仅分割、排除及映射待确认通道的`embedding_row`为空。确认：`panel_mapping.csv`行号与向量文件对应，来源通道顺序保留。
5. 按[向量发布步骤](../architecture/storage.md#准备产物文件布局)保存`marker_embeddings.pt`并更新panel映射；来源和规则附在向量载荷metadata中。确认：重新加载后的marker列表、映射表达式、向量及行号一致。
6. Study 加载层按 `marker_id` 匹配当版 study 向量的 `marker_ids`，并核对当版 panel 的 `embedding_row`；batch 接收已解析的 `marker_embeddings`，图像 tokenizer 与 marker 条件化表达头共用。确认：逐图数值筛选后通道仍与marker及向量对应；向量更新仅重建受影响study的向量和引用，依赖范围按存储规范判断。

### 共享强度

逐独立 image、逐保留通道，以全部实测像素（含背景）计算 linear 定义的精确 P99.5，记为 `q995`：

```python
def normalize_channel(block, q995):
    return clip(block / q995, 0, 1)  # 调用前确认 q995 > 0 且来源像素有限
```

1. 固定零下界；`q995 > 0` 时生成 float32 共享通道。分割、蛋白定量、训练和推理引用同一表示及基线统计。
2. `q995 <= 0` 的通道从当前 image 数值产物中排除，记录原因并保留来源映射。人工排除通道的 `q995` 为空；逐图排除保持全局 marker 身份不变。
3. 读取器 padding 和已知缺失采集填充值位于统计域之外。非有限输入记通道失败，阻止其依赖产物标为完成；有限负值按公式截到零。正值常数按公式得到全一，可用于记录测量，但缺乏分割结构。
4. 精确统计可采用整数直方图或 SSD 外存顺序统计；统计域覆盖完整独立图像或真实采集 ROI。按通道、按块读写，以限定内存占用。当前`uint8/uint16`采用`int64`整数计数和累计计数，按原rank及浮点运算顺序插值；其他dtype继续使用原float64外存partition。有效域、背景零、失败原因、float32输出和请求身份保持，正确归一化继续复用；整数路径不创建逐像素`pixels.bin`。
目录型来源须给出 `source_revision`。适配器仅对已确认归一化公式、统计域和通道 q 值一致的输入声明 `SourceSpec.normalized_q995`。共享处理核对值有限且在 [0,1]，复用像素及统计量。

### 新分割

分割修补可调整实际分割通道、分割专用预处理及模型参数。Cui 可将已核验总 Histone H3 纳入核组。保持共享 P99.5 图像，记录实际输入、参数、任务及原生边界检查。结合实例与边界证据判断分割质量，分别记录执行或传输错误、科学失败和正式产物接收状态。无法修复的图像记录证据及已尝试方法，获准后才排除。

2026-10-23更新：各study/panel的分割选择以raw channel为单位，使用[S1](../paper/supplementary_tables.md)的`Segmentation channel`列记录（1核、2膜／胞质、空值未选入）。核组选1–2个输入，优先DNA专用通道，允许经来源及本地图像复核的Nuc/Nuclear复合信号或总Histone H3/HH3；Ki67、FOXP3和磷酸化H3不作为通用核输入；膜／胞质组选择4–5个候选，兼顾信号强度和该组织中不同细胞类型的覆盖。来源panel决定实际通道位置；统一marker名称不改变分割输入身份。缺少可用核输入、候选不足或跨panel标记冲突需显式保留，候选的预期表达不能表述为已完成图像强度验证。核输入用途与DNA染色身份分别记录，组蛋白修饰通道的条件性备选不自动写为核组。Amitay沿用作者mask路径，后续完成来源mask检查后复用，本轮不修改其分割通道配置或mask。此处更新选择规则，现有代码的S1接入与多panel适配另行修改。

| 项目 | 操作或参数 |
| --- | --- |
| 核信号 | 对 study/panel 指定的有效共享核通道等权取均值 |
| 膜／胞质信号 | 对指定的有效共享膜／胞质通道等权取均值 |
| 输入 | HWC `[body, nucleus, 0]`；直接使用上述均值 |
| 模型 | 本地 `cpsam_v2`；已验证试验环境 `cellpose==4.2.1.1`；`normalize=False` |
| 试验起始参数 | `diameter=None, bsize=256, tile_overlap=0.1, flow_threshold=0.4, cellprob_threshold=0, min_size=15, max_size_fraction=0.4, resample=True, augment=False, niter=None, do_3D=False` |
| 节点执行参数 | batch_size、精度及并发按实际部署显式配置，并记入不可变请求、分割任务及结果 |

已有可用mask直接进入细胞信息建立。新分割需要有效、具有空间变化的核信号；核／膜信号的空间变化只在实测区域内判断，padding和未采集位置不参与。缺少核信号时记录mask缺口并保存已完成图像。膜／胞质信号缺失或在实测区域内恒定时可用零通道作为起点，并通过一次整图概览判断whole-cell范围。上述模型数值为工程起点，study/panel通道配置固定后按图执行。

## 3. 已有 mask 的一次检查

2026-11-02用户授权：作者未提供、访问受限或供应状态暂不明确的mask不再单独阻断技术预处理。优先复用已确认可用的作者whole-cell mask；当前无法取得可用实例mask时，允许按现有S1所选raw channel运行`cpsam_v2`，同时保留作者来源的实际状态和自行分割标记，不能把unknown伪写为已证实absent。已有归一化继续复用。缺少有效核输入、实测核信号为零或当前图像几何不成立仍需明确记录，不能据此宣称mask或细胞完成。此授权也适用于Amitay在作者mask不可用的明确范围；原始作者mask和input保持只读。

沿用来源提供的image-level配对和已有检查记录。按已确认的来源坐标关系，将mask转换到与input.tiff相同的裁剪框、朝向和padding，保留原始mask文件及转换记录；在同一独立image坐标系中执行以下四项：

1. **Whole-cell 类型**：根据来源说明大概确认 whole-cell 或 nuclear；说明不足时结合第4项判断。
2. **尺寸与空间范围**：比较 image/mask 的 H/W、范围及明显整体偏移。正常触及图像边缘视为界内。
3. **Index 正常性**：确认非负整数实例标签，背景0、细胞为正整数；允许编号不连续。
4. **Study级概览抽样**：按同来源mask类型、坐标转换和panel分组，查看代表image/mask整图叠加缩略图，记录明显整体错位、全空或类型疑问。起始3张及异常扩查沿用[抽样规则](study_input_review.md#3-study级分层抽样)；已知异常图像单独查看。同组未查看图的概览状态保留未查看，并引用组级证据。

在image metadata记录这四项结果与具体问题。可用则复用；缺失或明确不可用时进入新分割。当前逐图实现支持带非空原因和来源证据的 unusable 声明，无需复制被拒绝 mask；未做的几何、整数与概览检查记录为未判定。必要概览尚无判断时保留待确认状态及已完成图像。来源转换或预测拼接完成后，将实测区域外的实例像素置0，再冻结最终mask身份并建立细胞编号和表达对应。

## 4. 完整操作与逐步确认

本节维护输入验收后的固定操作顺序。按study组织、按image计算和恢复；字段、文件与节点执行分别遵循数据合同、存储规范和多机执行规范。

1. 接收输入。读取已验收的images.parquet、panel_mapping.csv、sample_metadata.csv及逐图input.tiff，保留来源、sample/patient关联和临床字段。按实际panel解析三层通道位置，确认已选分割raw channel仍有输入像素。确认：来源行、input通道及独立image身份可对应；缺失分割输入定位到上游修补。
2. 导出study向量。输入通道取舍固定后，按§2“Study级marker向量查表”在所属下游节点CPU独立执行，可与逐图处理并行。确认：marker_embeddings.pt与panel_mapping的embedding_row一致；仅分割行不产生表达列或向量引用，逐图处理不等待向量。
3. 生成共享图像。由四侧padding和可选valid_pixels.tiff恢复实测区域，按§2计算全实测域P99.5；实测背景和零值参与统计，区域外共享像素为0。确认：image.tiff与channels.parquet对应，逐通道q995、数值去向及失败原因可查询。
4. 检查来源mask。按来源image配对读取mask，以与input相同的裁剪框和padding转换到独立image坐标，按§3完成一次检查。确认：可用则复用；必要概览尚无判断时记录待确认，缺失或明确不可用时进入新分割。Amitay核对后复用已本地化作者mask。
5. 执行必要的新分割。按raw channel配置合成有效核／膜信号并调用cpsam_v2；使用既定节点参数和大图拼接方案。确认：实际输入组、模型、参数及结果身份一致；核信号缺失时保留mask缺口，零body分支完成whole-cell概览判断。
6. 固定mask和细胞索引。完成来源几何转换或预测拼接，将实测区域外的实例像素置0，再按最终正标签升序建立连续cell_index、mask_label、稳定cell_id、centroid、area和bbox。确认：mask前景全部位于实测区域，标签与cells一一对应；边缘实例保留其图内实测部分，后续步骤使用该固定索引。
7. 完成蛋白定量。在最终mask内对每个有效downstream通道求全部实例像素均值，包含真实零值；crop前逐细胞计算一次，float64累计、float32保存。确认：expression显式连接cell_index及mask_label，分母等于最终实例的实测像素数，分割专用通道无表达列。结果表示图像内归一化平均信号，边缘实例仅代表保留部分。
8. 建立同图邻域。按完整独立image的centroid建立cKDTree，按ID排除自身，按距离保存最多20个邻居。确认：索引引用同图cells，不足填-1，不按crop重新计算。
9. 建立crop对应和采样规则。按§7分别建立训练224／112与验证／测试224／224网格，以实际实例像素建立对应；训练采样另按有效组织比例筛选。确认：所有实例／窗口相交关系保留，采样过滤不删细胞或表达，图像、mask及实测有效域可同步读取。
10. 核对并发布。按存储规范先写临时结果，重开并检查本次产物及直接依赖，再由所属下游节点发布到自己的 NAS，最后写逐图完成记录。确认：完整输入metadata保留，正式产物引用所属节点持久目录，cell→expression/mask/crop可查询；缺口和实际完成项分别记录。最终归并主 NAS 时再次核对当前原始 D01、输入和请求并重开结果。

缺mask时保留共享图像、通道记录及网格规则；无有效蛋白时保留可用mask、几何和邻域；无合格crop时保留全部准备产物并记录采样数量为0。训练准入按[数据目录](catalog.md)核对实际输入和所用marker向量，部分产物完成不表示整条流程或训练已就绪。

## 5. Per-study任务模板

任务引用本页科学规则；模型起始参数及节点预算由各自维护位置读取。输入验收范围和允许处理的image明确列入任务。

```yaml
study: <稳定study标识>
input_root: processed/<study>/
image_ids: [<本批已验收image_id>]
panel_ids: [<本批实际panel_id>]
source_mask_bindings: <来源配对记录或已确认无来源mask>
workflow: docs/data/preparation.md
data_contract: docs/architecture/data_contract.md
storage: docs/architecture/storage.md
execution: scripts/data/preparation/README.md
output_root: processed/<study>/
```

## 6. IO、资源与完成检查

1. NAS 读取来源，SSD 缓存当前任务；统计、归一化、定量和索引按 image/通道/块推进。符合强度及窗口读取要求的 TIFF 可直接引用，布局转换保持数值无损。
2. 分割、定量和索引使用显式工作根及资源预算。执行和恢复参数见[准备脚本](../../scripts/data/preparation/README.md)。
3. 分割任务保留核/膜合成TIFF、任务JSON、mask及实际参数；同一image的前后处理和study index按实际执行配置协调，正式发布至主NAS并核对当前来源，缓存继续受容量限制。
4. 大图外部分块需实现 halo、重叠实例匹配、重复去除和全图 label 合并；以接缝实例一致性验收后扩大运行。分割计算块、TIFF 存储 tile 和训练 crop 分别配置。
5. 每图先写临时结果；检查本次实际生成的产物及其依赖，通过后发布，并在metadata记录完成项与缺口。规则和输入未变的完成结果复用；变更后的产物重建范围按[存储规范](../architecture/storage.md#准备产物文件布局)确定。
6. 最小验收覆盖已有 mask、新分割、稀疏标签、退化通道、跨 TIFF tile 窗口、必要的大图接缝，以及 cell/mask/expression/crop/view 查询闭环。完成情况见[项目状态](../project/status.md)。
7. **独立重核采用 study 级分层抽样。** 起始每 study 3 张并覆盖不同 panel、作者 mask 来源／坐标转换、新分割及拼接分支，新增分支补选代表图。样本核对 P99.5 及归一化、mask 空间对应、细胞索引、蛋白均值、邻域和训练 crop 闭环。逐图检查当前 D01、来源与请求版本、文件非空、文件头或表 schema、完成状态。完整标签／面积、外键及 crop 生成在产物计算时核对。抽样证据保存在 `study_qc`。相同分支和版本复用样本，受影响的证据随依赖失效并按分支补查。选图和异常扩查见[输入抽样规则](study_input_review.md#3-study级分层抽样)。
8. 分别统计持久产物、逐图核对、抽样覆盖及实际数值抽样图数。组织方法按组织与方法分层核验，未检组保持 sampling pending。逐窗口继续自动执行组织比例和全黑排除。作者 mask 类型未知、缺身份连接和已定位问题记录实际影响范围。

## 7. Crop 与训练视图

1. 按patient→image／ROI／FOV→crop组织输入；基础窗口为原生224×224像素。训练步长112，保留既有末端length-224回移并去重规则；短轴不足224时起点为0、右／下padding。验证／测试／S4步长224、窗口无重叠，每轴从0开始按224生成固定起点，末窗不足224时右／下padding，不回移起点。实测范围由输入padding、可选valid_pixels.tiff及窗口越界共同确定。
2. 物理视野为224乘对应轴的µm/px，像素尺度逐图核验。基础窗口直接读取图像、mask和实测有效域的相同范围。
3. 全图TIFF保存像素，cells.crop_origins保存每个cell实际相交的全部窗口左上角，允许部分实例；网格由H/W、用途与本节规则重建。表达保持最终mask内的整图实例均值，训练采样过滤不删除完整对应关系。
4. 实测区域包含真实背景和零值；训练筛选另估计组织区域。工程起点为共享图像中有效、在实测区域内非恒定的核／蛋白通道最大投影，在实测区域估计Otsu阈值，再做3×3二值opening/closing各一次并限制到实测区域。组织估计不使用细胞mask覆盖率或简单非零判断替代；按study/panel代表图核查后使用，无可用空间信号时记录不可估计，不自动准入。
5. 基础训练crop须存在有效downstream蛋白通道，且有效组织像素数／(224×224)严格大于0.50。分母包含完整窗口；真实背景、padding及未采集位置不计入分子，恰好50%不通过。训练实际读取的downstream通道在窗口实测区域内全部为零时不通过，核／分割专用信号不能单独使此类窗口准入；部分padding窗口按同一面积条件判断。此规则取代原30%阈值和默认无padding限制。
6. 小图／窄图或其他image无合格窗口时，保留图像、mask、细胞、表达、邻域及完整crop对应，记录可采样窗口数为0。组织比例及窗口选择可按需计算、缓存，不新增正式逐crop审计表；实际方法、参数、数量及缺口记入image metadata。

组织方法尚未核验时，新计算延后全图组织估计，metadata明确记录声明的方法／参数、`evaluated=false`、`count=null`及`status=pending`，不将待核验写成零个合格窗口。同一image与sampling request已有的已评估结果继续复用；组织核验通过后仍执行原组织估计、严格>50%及downstream非全黑筛选。该延后只影响训练采样评估，不影响完整细胞crop对应、mask、定量及邻域产物。实现与远端部署状态分别由[当前状态](../project/status.md)记录。
7. 训练从原生224×224基础窗口生成DINOv2多视图，image、instance mask、实测有效域和坐标共享裁剪、缩放及翻转／旋转变换；离线whole-cell目标保持原值，实例标签采用离散插值。视图尺寸和增强由[训练规范](../architecture/training.md#2-dinov2-多视图与遮盖)维护。全部视图和重叠crop继承来源患者／图像split。
8. 网络内部采用8×8 patch；实例mask在主干后用于细胞读出。当前crop以centroid确定细胞归属，允许实例截断，表达监督保持离线whole-cell目标；离线crop_origins继续按实际实例相交保存，不用它替代centroid归属。结构与像素观测状态分别由[模型规范](../architecture/model.md)和[数据合同](../architecture/data_contract.md)维护。
9. 推理与删除比较读取固定基线224窗口，在对应实例像素和编辑实例视图上执行操作，重新编码与改动像素相交的crop，并保持归一化统计、原始mask身份和比较几何固定。步骤由[编辑规范](../architecture/editing.md)维护。

```python
grid_origins((height, width))  # train: edge-aligned 224/112
grid_origins((height, width), purpose="val")  # val: fixed 224 starts
grid_origins((height, width), purpose="test")
grid_origins((height, width), purpose="s4")
```

未指定用途时`grid_origins`沿用训练网格，因此既有`cells.crop_origins`及历史metadata、采样结果的224／112末端回移解释不变。验证／测试／S4按用途生成224／224固定起点网格；`read_crop`对越出图像画布的末窗在右／下补零并标为无效。Figure 3 S4的query／gallery冻结导出采用共同评分网格，各模型在同中心读取原生输入；细节由[检索协议](../evaluation/representation.md#46-s4相似组织检索与metadata描述)维护。

## 8. 方法依据

强度与分割的试验依据见[分割探索](../experiments/segmentation_pilot_20261019.md)、[固定 mask 定量](../experiments/protein_quantification_pilot_20261020.md)和[非零 P99 比较](../experiments/nonzero_p99_pilot_20261020.md)。

| 参考 | 来源做法与采用范围 |
| --- | --- |
| [KRONOS v1](../references.md#r03--kronos) | 以 256×256 基础块和 50% 重叠组织数据，训练再随机裁至 224×224，生成 2 个全局、8 个局部视图。CellWeft 借鉴重叠与多视图，直接使用 §7 的原生 224 基础块。 |
| [KRONOS2 模型卡](https://huggingface.co/MahmoodLab/KRONOS2/blob/main/README.md)与 [CORAL 源码](https://github.com/mahmoodlab/CORAL/tree/main/src/coral/patch) | KRONOS2 发布模型参考块为 256×256，推荐 CORAL。CORAL 支持原生尺度、可配步长或重叠、末端贴边及组织比例索引，短轴不足一块时网格为空。CellWeft 采用原生坐标及训练末端贴边，短轴补边并按 §7 筛选，重叠和验证网格使用项目自身配置。 |
| [VirTues](../references.md#r02--virtues) | MAE 采用 256 基础块，过滤组织比例低于 30% 的块，训练随机裁取 128，组织检测使用最大投影、Otsu 及开闭运算。CellWeft 借鉴组织估计及分层裁剪，按 §7 的实测域、代表图核验和组织比例执行。 |
| [DINOv2 官方增强](../references.md#r20--dinov2) | 使用独立的 global／local RandomResizedCrop 及图像 bicubic 插值。CellWeft 采用独立多视图，同步空间对应，视图与强度规则由[训练规范](../architecture/training.md)维护。 |

按原生像素直接组窗保留来源几何。固定 224 像素的物理视野由逐轴 µm/px 决定，跨平台尺度影响仍待验证。现有证据尚不足以支持额外引入 256 基础块。保留完整实例相交索引和 whole-cell 定量，按 §7 决定窗口是否用于训练。
