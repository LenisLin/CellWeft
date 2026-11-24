# 输入、任务视图与空间对应

采用 dataclasses 或同类 typed schemas。稳定实体 ID、测量状态和几何对应是模型/编辑公共边界；物理保存位置见[storage.md](storage.md)。

逐图逐通道归一化采用固定零下界和完整实测域内全部像素的 P99.5，含实测背景零值。q995 非正时排除该图对应通道并保留来源映射。强度处理和基础切窗见[数据准备](../data/preparation.md)。实施进度见[当前状态](../project/status.md)。

## 1. 实体与记录

主实体：cohort_id、hospital_id、patient_key、slide_id、roi_id、image_id、crop_id、cell_id、marker_id。Dataset row index 与稳定 cell_id 分开。image_id在study内唯一，跨study使用(study,image_id)；来源短文件名不单独充当跨study身份。查询绑定metadata中的图像及mask产物身份，cell_index在该身份内有效。
Study输入metadata：每study一份images.parquet，承担image metadata与处理入口职责，每个独立ROI／TMA core／FOV一行。主键为(study,image_id)，sample_id连接生物学取样宽表sample_metadata.csv，panel_id连接本study的panel_mapping.csv。患者信息并入sample表，区域注释保存在image的region_annotation。字段及扩展规则由[CSV模板及字典](../data/templates/study_input/README.md)维护，准入、连接及拆分由[输入流程](../data/study_input.md)维护。

可选image扩展列sample_region_path为NAS相对路径，指向同尺寸uint8 YX sample_region.tiff；1表示本sample处理区域，0表示邻sample或padding。该区域与实测有效域取交集后用于强度统计、分割、定量和crop；input原始像素保持。边界依据及扩展字段定义记录在field_mapping。Image、sample及patient身份的完成范围分别验收。

source_channel_count记录原始采集通道数，channel_count记录input.tiff保存通道数。来源框使用选定采集的原生像素坐标及半开区间；保存来源尺度、dtype、四侧padding及采集有效域信息。input_image_path直接定位逐image文件，路径相对NAS项目根目录；panel及study目录按固定布局解析。

SlideRecord：cohort/platform、pseudonymous patient key、slide/ROI、image/mask 路径、marker 表、pixel_size_x/y、H/W、channel 数、licensing/access、split、source DOI/accession、normalization_version及已有来源完整性记录（若有；不新增常规全量hash）。此为后续扩展记录，不作为初始导入表的必填字段集合。
MarkerRecord：marker_id、display_name、可空canonical protein ID、可空gene symbol、measured_form、可空phosphosite/epitope/clone、channel role、study上下文中可空的embedding_row、mapping confidence。Canonical protein ID在能明确对应单一蛋白时填写；家族／组合保留marker_id和来源测量身份。已确认向量在输入通道筛选后的独立步骤生成；训练组装前检查实际使用通道的向量完整性。

display_name 使用常用的蛋白／测量简名（如 FoxP3、Ki-67、EGFR），用于通道命名；可采用 AFP、APOD、CDH11、CTNNA2 等简洁、明确的缩写或 gene symbol。蛋白全称和 gene symbol 保留为数据库注释。marker_id 区分实际测量形式：CD45、CD45RA、CD45RO 使用独立 ID；共享 UniProt 主 accession 不表示测量等价。磷酸化、切割、突变及明确异构体同样按来源测量区分。用户确认的聚合命名及pH3位点归并由[数据准备](../data/preparation.md)维护；聚合组保留来源靶标与实际修饰位点，跨蛋白组的canonical protein ID可空，向量映射以S1为准。其他按同义测量归并。草稿对齐清单保留模型原始分类与非蛋白判定，确认后才发布正式 registry。
Total/phospho 即使对应同一 canonical protein 也不合并 marker identity；不同 clone 可共享基础向量，但保留测量 metadata。DNA/DAPI/isotype 不伪造 protein ID。
全项目从S1及来源metadata派生统一marker registry。已确认downstream marker首次派生时，以S1的`Unified marker`原文作为`marker_id`，相同名称使用同一ID；MAP-KG mapping作为向量映射表达式，不参与ID命名。名称调整若改变已发布ID，需明确迁移身份并同步关联产物。来源表位、位点及clone注释继续保留；分割专用通道沿用其measurement身份。在study/panel阶段生成按来源通道位置排列的映射和segmentation/downstream/exclude去向；per-image阶段应用已确认映射。操作规则由[数据准备](../data/preparation.md)维护，图像 tokenizer、表达头与像素 decoder 使用同一 marker 身份；tokenizer 与表达头读取固定 study 向量，decoder 的目标条件编码见[模型规范](model.md#7-图像-marker-重建读出)。
家族／组合按[数据准备](../data/preparation.md)的 study 覆盖规则决定保留；保留的组合维持独立测量身份和强度，向量按S1确认的映射生成，不复制成多个独立蛋白测量。跨 study 同名异义或通道去向不同的来源标签使用 study 限定别名，查询表与 study 对齐清单保持同一对应关系。
CellRecord/CellTable：全局稳定cell_id与cell_index、mask_label分开；包含image_id、slide_id、独立image局部像素centroid_x/y、area、bbox、crop_origins、mask类型及来源／版本、marker measurements和assayed mask。来源mask的几何转换或预测拼接、实测区域外清零全部完成后，固定最终mask身份及CellTable。几何和蛋白均值仅使用最终实例的实测像素，包含真实零值；图像边缘实例按保留部分统计。背景label 0不建细胞或表达行。来源表达行按mask_label显式关联；mask改变时不将新旧标签直接视为同一细胞。
Crop坐标归入CellTable的crop_origins：每个cell对应全部实际相交网格窗口的局部左上角(x0,y0)，窗口为[x0,x0+224)×[y0,y0+224)，允许部分实例；bbox仅用于筛候选。现有 crop_origins 与 metadata 描述训练网格。验证和测试按[准备规范](../data/preparation.md#7-crop-与训练视图)生成独立网格，并以原图 centroid 确定归属。H/W确定图像外边界，padding及可选valid_pixels.tiff共同确定实测范围。训练采样过滤不删除完整crop_origins，随机view的几何变换在加载时维护。

StudyMarkerEmbeddings：`marker_embeddings.pt`保存`embeddings: FloatTensor[M_study, R_p]`、按行排列的`marker_ids: list[str]`及`mapping_expressions: list[str]`；两列表长度均为M_study，marker_id在study内唯一，向量dtype为float32。metadata另保存study及实际panel范围、构建时panel输入身份（不含embedding_row）、基础库/S1引用、资产身份及聚合／归一化规则。读取向量时同时核对expected_study、当前panel身份及行号，避免同marker改绑S1后消费旧向量。R_p读取基础库manifest，当前为1,024。该产物在输入通道筛选后单独导出。

输入阶段panel映射按[模板](../data/templates/study_input/panel_mapping.csv)保留全部来源行。mapping_status记录S1状态，role记录实际处理用途，keep_channel及output_channel_index记录像素保留情况；同一marker可有多个保留来源行，但每panel最多一个downstream行。其余已选分割行使用segmentation-only，保留marker身份及nuclear/body配置，表达列和embedding_row为空。后续向量步骤增加可空整数embedding_row；非空时满足0 <= embedding_row < M_study，且marker_id与向量文件对应。同一marker跨panel复用一行，不同marker即使向量相同也保留身份；全局registry不保存study行号，映射待确认和排除行不分配向量引用。

通道位置分三层：source_channel_index定位原始采集，output_channel_index定位筛选后的input.tiff，后续channels.parquet中的TIFF page定位共享image.tiff。读取筛选后输入必须使用output_channel_index；记录与核对继续保留原始source_channel_index。逐图数值排除不改写来源身份和输入位置。

逐图metadata.json和channels.parquet保存共享图像引用、全部来源通道的三层位置、q995、数值去向／原因及表达列。metadata记录mask_origin、最终mask_identity、mask类型、来源引用和一次检查结果；来源mask另记录读取对象及到input局部坐标的裁剪／padding变换，新分割另记录实际核／膜组、模型参数及交接输入／权重身份。记录实测有效域的来源、网格与采样方法／参数、可采样窗口数、完成项及缺口；输入images.parquet保持完整来源和身份字段。
共享图像保存数值有效的downstream通道及已选分割专用通道，训练图像和表达只读取downstream；分割合成信号按raw channel用途组装。图像及表格格式为TIFF/Parquet/CSV/JSON，study向量使用.pt载荷；文件布局见[存储规范](storage.md#准备产物文件布局)，TIFF窗口性能和生产接口仍需实际验收。

## 2. 张量、观测状态与训练视图

单个 image 为 M,H,W；基础 crop 输入为 B,M_max,224,224。U 为本 microbatch 中按 (study,image_id,cell_index) 去重的细胞数；同一细胞可以由多个训练 crop 引用。Whole-cell 定量值作为训练目标及分析数据，不作为图像主干输入。坐标使用原图像素 (x,y)，运行时记录到 crop 和各视图的变换；物理坐标按逐图 µm/px 派生。

| 字段 | 语义与形状 |
| --- | --- |
| images | FloatTensor[B,M_max,224,224]，下游蛋白通道；多视图按训练规范的对应分辨率组织 |
| marker_ids / marker_embeddings / marker_valid | 稳定测量身份；FloatTensor[B,M_max,R_p]；BoolTensor[B,M_max]。按 marker_id 匹配当版 study 向量的 marker_ids 后组装；当版 panel 的 embedding_row 须与其一致，R_p 当前为 1,024 |
| whole_cell_targets / cell_marker_assayed | FloatTensor[U,M_max] 与 BoolTensor[U,M_max]；完整 image 上最终实例的蛋白均值及原始实测状态；作为训练目标和分析数据 |
| cell_keys / cell_valid | (study,image_id,cell_index) 到 unique-cell 行号的映射及有效性；padding 不参与细胞读出或 loss |
| coordinates / cell_to_crop | 原图 centroid、变换后的局部 centroid 与对应 crop／view、原始 cell 行号的映射；只选择 centroid 落入该视图的细胞用于其读出 |
| instance_masks | 与相应图像同步裁剪与变换的整数实例 labels；label→cell_index，按需为每个细胞取二值 mask |
| spatial_valid | BoolTensor[B,1,224,224]，原始实测域；真实背景／零值为 true，padding／未采集为 false；各视图随相同几何变换 |
| pixel_observed | 当前图像输入可用观测 O；可共享时为 B,1,224,224，有 marker-specific 缺口时为 B,M_max,224,224 |
| ibot_patch_mask | iBOT student 的空间 patch 遮盖及对应 teacher 位置；按该视图 patch 网格组织，与原始实测域和逐 marker 的 pixel_observed 分别维护 |
| crop_ids / transform_records / provenance | Crop origin、同步几何、study／向量资产身份、patient／split 和任务输入来源 |

Unique-cell 行的 marker 列按来源 image／panel 对齐，并通过映射取得对应 marker identity 和 embeddings。真实测量零值对应 assayed=true、O=1；padding、未采集及按任务人工隐藏的像素设为 O=0，tokenizer 使用 O⊙I。Spatial_valid 保留原始采集含义，pixel_observed 随输入视图变化；DINO／iBOT 的空间 token 遮盖按训练规范处理。图像、instance mask、centroid 与实测域共享每个视图的几何变换；离散标签使用保持实例身份的变换。

每个 crop 使用 marker／cell validity mask 表示可变数量。有效 crop 可无可读细胞，此时仍输出图像 patch 和 CLS。输出结构见[模型规范](model.md)，有效 cell／marker 的损失归约见[训练规范§3](training.md#3-联合训练目标)。

## 3. 索引

必须可查询 slide／image → cells／images／markers／split；cell → marker row／mask label／centroid／area／top-20 近邻及反向邻居引用；crop → 实际相交实例及 centroid 归属细胞；cell → overlapping crops／patch footprints。预处理已有同 image top-20 centroid 近邻可供邻域分析及任务使用。每张完整独立 image 的近邻由 cKDTree 按 centroid 欧氏距离计算，不跨 image_id，排除自身并按距离升序排列；现行 neighbor_indices[N,20] 不足部分填 -1，Parquet 列为 n01…n20，距离不落盘。近邻使用原图像素 centroid；若发现非等宽像素，再讨论尺度修正。

查询主键为 (study,image_id,cell_index)，cell_index 按最终 mask 的正标签升序从 0 分配，是该 image 固定 CellTable 的连续行号；表达表通过显式 cell_index／marker 列关联，不能依赖任意 Parquet 物理行顺序。Crop／view 不重新编号或重新计算 whole-cell measurements。组装 batch 时映射到 batch 行，采用 -1 表示 padding 时不得将其用于有效 gather。

## 4. 空间对应

1. 同坐标读取原图、instance mask 和实测域。`cells.crop_origins` 按资产版本解释。窗口外 padding 的 label=0、`spatial_valid=false`。基础窗口按[准备规范§7](../data/preparation.md#7-crop-与训练视图)生成。
2. 图像、mask、实测域与 centroid 同步变换并保存原图到视图的变换及其逆映射。实例标签用最近邻插值保留离散 ID。训练视图参数见[训练规范§2](training.md#2-dinov2-多视图与遮盖)。
3. 按半开窗口内的变换后 centroid 选择可读细胞。用正标签反查原图 `cell_index`，以视图内可见实例像素读出。无有效实例像素时跳过本视图的细胞读出并记录覆盖。保留原 ID 与离线 whole-cell 目标，允许边缘截断。全图边界实例以图内实测部分为目标。训练重叠窗口可多次读取同一细胞，验证／测试按非重叠窗口归属。
4. Figure 2／3 的 ROI 级遮盖同步映射到各 crop 像素和观测状态。任务流程见[表征评价](../evaluation/representation.md)。

表达矩阵、mask 和 crop 通过同一 cell 索引直接对应：expression.parquet 的 cell_index → cells.parquet 同键行 → mask_label 与 crop_origins。离线 crop_origins 按实例 mask 与窗口实际相交保存，一个 cell 可以关联多个 crops；读取时同范围加载 image 和 mask，再依据 centroid 选择本视图需读出的细胞。图像、mask 和各自有效域始终对齐。反向查询某 cell 可从 crop_origins 得到全部相交图像块；低组织占比筛选不删除这一关系。

例如训练网格资产中 cell_index=42、mask_label=107、crop_origins=[(0,0),(112,0)]：表达向量取 expression 中 cell_index=42 的行；两块 mask 分别取 mask[0:224,0:224] 和 mask[0:224,112:336]，其中值 107 的像素对应该细胞。若 centroid 落在两个重叠窗口，训练时两窗均可读取同一 whole-cell 目标；验证／测试按非重叠窗口确定唯一归属，不拼接相邻 crop 特征。


## 5. Figure 5 编辑视图设计（待实现）

细胞替换和 marker 数值／状态编辑使用独立编辑视图。以下定义字段语义，具体 typed schema 与操作接口在实施前确定。输入操作见[编辑规范§6](editing.md#6-替换与状态编辑)。

1. Baseline 的 CellTable、cell_index、实例、whole-cell measurements、原始 assayed 和 spatial_valid 保持不变。
2. 编辑图像与实例视图是编辑输入源；marker数值编辑／细胞替换后，可依baseline相同whole-cell均值规则从编辑像素重算受影响细胞值以描述编辑，记录修改位置。图像的pixel_observed表示当前输入可用性；原始assayed不因数值编辑改写，编辑值另标记来源和操作。
3. 替换细胞使用 edit-scoped 身份，并关联移出的原细胞、真实供体、编辑实例及坐标；供体测量支持单独记录。运行时 gather 按编辑身份映射，不改写原始 mask_label/cell_index。
4. 编辑视图记录 edit_id、来源／供体、几何和尺度变换、被改 marker／位置及适用分析策略。实际患者和细胞成员按存储规范进入 NAS，跟踪文档保存规则。
