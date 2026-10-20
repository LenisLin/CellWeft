# 数据准备流程

本页维护全局 marker 统一、study 准备和逐 image 处理的操作规则。字段及空间对应见[数据合同](../architecture/data_contract.md)，正式文件布局见[存储规范](../architecture/storage.md#准备产物文件布局)，实际进度见[项目状态](../project/status.md)。

共享代码、dataset入口、配置及运行命令见[准备实现说明](../../scripts/data/preparation/README.md)。单图pipeline维护复用和部分完成，科学规则以本页为准。

## 1. 执行层级与输入

1. 全局：汇总全部已收集 study/panel 的 marker，人工确认统一语义，形成 `metadata/marker_registry.csv`。后续新增 study 接入同一 registry。
2. Study：完成 panel 通道映射和分割通道组配置，再建立一份 `images.parquet`；study 是任务组织与 index 汇总单位。
3. Image：遍历 study index，独立完成强度统计、mask、定量、neighbors 与 crop 对应；image 是计算、恢复和完成检查单位。
4. 保存：原始发布文件按收集记录读取，正式产物保存到 NAS `processed/<study>/`。SSD 用于有界临时缓存。

Study index 初始字段为 `study、image_id、channel_count、pixel_size_um、image_path、marker_list_path、output_path`。`channel_count` 是来源通道数；`image_path` 为来源目录，`output_path` 为 study 输出目录，`marker_list_path` 为确认后的 panel 映射文件。Study 读取配置结合 `image_id` 定位实际文件或容器 ROI，并选择对应 panel；像素尺度落实到每张图像，多尺度或非等宽像素分别记录。

## 2. Marker 清理、共享强度与分割参数

### Marker 与来源强度

1. Manual mapping 在 study/panel 层完成。同义且测量含义等价的通道共享 `marker_id`；蛋白身份与 measured form 分开记录，total/phospho 分别保留。
2. 通道去向：实际蛋白用于 downstream；DNA/DAPI 等非蛋白核染用于 segmentation-only；其他非蛋白、身份未决及明确重复通道标记 exclude。Downstream 蛋白也可参与核、膜／胞质分割信号；Ki67、FOXP3 按蛋白保留。
3. 保留有明确测量含义的家族、组合及修饰通道，如 panCK、CK8/18、E/P-cadherin、vWF+CD31、HLA-ABC、Collagen、S100、pErk1/2。一个来源测量对应一个表达列；组合成员尚未明确时保留来源/panel 测量身份。Protein bank 编码在全部数据准备完成后统一完成。
4. 根据来源说明和实际加载过程确认磁盘强度。已有结果符合下述公式及统计域时直接复用；来源已完成的处理沿用其明确语义，补齐当前归一化。通道映射固定后，per-image 仅执行映射和数值处理。

### 共享强度

逐独立 image、逐保留通道，以全部实测像素（含背景）计算 linear 定义的精确 P99.5，记为 `q995`：

```python
def normalize_channel(block, q995):
    return clip(block / q995, 0, 1)  # 调用前确认 q995 > 0 且来源像素有限
```

1. 固定零下界；`q995 > 0` 时生成 float32 共享通道。分割、蛋白定量、训练和推理引用同一表示及基线统计。
2. `q995 <= 0` 的通道从当前 image 数值产物中排除，记录原因并保留来源映射。人工排除通道的 `q995` 为空；逐图排除保持全局 marker 身份不变。
3. 读取器 padding 和已知缺失采集填充值位于统计域之外。非有限输入记通道失败，阻止其依赖产物标为完成；有限负值按公式截到零。正值常数按公式得到全一，可用于记录测量，但缺乏分割结构。
4. 精确统计可采用整数直方图或 SSD 外存顺序统计；统计域覆盖完整独立图像或真实采集 ROI。按通道、按块读写，以限定内存占用。

### 新分割

| 项目 | 操作或参数 |
| --- | --- |
| 核信号 | 对 study/panel 指定的有效共享核通道等权取均值 |
| 膜／胞质信号 | 对指定的有效共享膜／胞质通道等权取均值 |
| 输入 | HWC `[body, nucleus, 0]`；直接使用上述均值 |
| 模型 | 本地 `cpsam_v2`；已验证试验环境 `cellpose==4.2.1.1`；`normalize=False` |
| 试验起始参数 | `diameter=None, bsize=256, batch_size=4, tile_overlap=0.1, flow_threshold=0.4, cellprob_threshold=0, min_size=15, max_size_fraction=0.4, resample=True, augment=False, niter=None, do_3D=False` |

已有可用 mask 直接进入细胞信息建立。新分割需要有效、具有空间变化的核信号；缺少核信号时记录 mask 缺口并保存已完成图像。膜／胞质信号缺失时可用零通道作为起点，并通过一次整图概览判断 whole-cell 范围。上述模型数值为工程起点，study/panel 通道配置固定后按图执行。

## 3. 已有 mask 的一次检查

沿用来源提供的 image-level 配对和已有检查记录，执行以下四项：

1. **Whole-cell 类型**：根据来源说明大概确认 whole-cell 或 nuclear；说明不足时结合第4项判断。
2. **尺寸与空间范围**：比较 image/mask 的 H/W、范围及明显整体偏移。正常触及图像边缘视为界内。
3. **Index 正常性**：确认非负整数实例标签，背景0、细胞为正整数；允许编号不连续。
4. **必要时一次整图快速概览**：查看整张 image/mask 叠加缩略图，记录明显整体错位、全空或类型疑问。

在 image metadata 记录这四项结果与具体问题。可用则复用；缺失或明确不可用时进入新分割。细胞编号和表达对应在后续步骤建立。

## 4. 完整操作与逐步确认

| 步骤 | 输入与核心操作 | 完成时确认 |
| --- | --- | --- |
| 1. 全局 marker 统一 | 汇总各 study/panel 来源列表，按 §2 确认测量身份 | 单一 registry 和明确的等价关系 |
| 2. Study 准备 | 固定通道映射、核/膜组，建立七字段 study index | 每个 image 可解析到来源、panel、尺度和输出目录 |
| 3. 图像处理 | 读取对应图像，复用兼容结果或计算共享 P99.5 图像 | 全部来源通道有去向；保留通道有统计与 TIFF page 对应 |
| 4. 获得 mask | 按 §3 检查已有 mask；需要时按 §2 新分割 | 原图坐标下的 whole-cell mask，或明确的 mask 缺口 |
| 5. 细胞表与定量 | 固定 mask 后建立 cell_index、mask_label、cell_id、centroid、面积和 bbox；在全部实例像素上逐蛋白通道求均值 | cells 与 expression 通过显式 cell_index 对应 |
| 6. Neighbors | 完整独立 image 内建立 centroid cKDTree；查询并按 ID 排除自身，按距离保留最多20个邻居 | 邻居均引用同图 CellTable 的 cell_index |
| 7. Crop 对应 | 按 §7 生成224/112网格；bbox筛候选，再用实际实例像素判断相交 | 每个cell的 crop_origins覆盖所有实际相交窗口 |
| 8. Study 发布 | 检查各图像引用及对应关系，主控更新同一 study index 和 manifest | NAS正式目录可完成 cell→expression/mask/crop 正反查询 |

定量采用最终 whole-cell mask 内共享蛋白图像的算术均值，包含实例内零像素；在全图坐标下、crop 前计算一次。按块累计 float64 通道和与像素数，结果存 float32。稀疏 mask 标签先映射到连续 cell_index，再进行累计。背景0属于细胞集合之外。所得数值表示图像内归一化平均信号，跨 study 的解释仍受来源及归一化尺度影响。

Mask 固定后，cell_index 在该图像的 CellTable 内保持不变；表达、neighbors、crop 均引用该编号。无有效蛋白时保留已有几何和邻域；缺 mask 时保留共享图像及网格规则。训练准入依据[数据目录](catalog.md)检查实际输入，bank 映射在数据准备之后补齐。

## 5. 核心功能伪代码

以下是职责与数据依赖，供生产实现使用；大数组以可分块读取的引用传递。`publish_image` 检查实际产物及其依赖，并将完成项与缺口写入 image metadata；study index 保持七字段。

```python
def prepare_studies(studies, policies):
    registry = confirm_global_marker_semantics(studies)
    for study in studies:
        mapping = finalize_panel_mapping(study, registry)
        index = build_study_index(study, mapping)
        for row in index:
            source = open_image(row.image_path, row.image_id, study.reader)
            shared, channels = prepare_or_reuse_shared(source, mapping, policies)
            mask = obtain_whole_cell_mask(source, shared, mapping, policies)
            products = dict(image=shared, channels=channels, mask=mask)
            if mask is not None:
                cells = freeze_cell_table(study, row.image_id, mask)
                products["cells"] = cells
                if shared.has_protein_channels:
                    products["expression"] = mean_by_instance(shared.proteins, mask, cells)
                products["neighbors"] = top20_ckdtree(cells.centroids, exclude_self_by_id=True)
                cells.crop_origins = intersecting_origins(mask, cells, size=224, stride=112)
            publish_image(row.output_path, row.image_id, products)
        publish_study(index, registry)
```

## 6. IO、资源与完成检查

1. NAS 读取来源，SSD 缓存当前任务；统计、归一化、定量和索引按 image/通道/块推进。符合强度及窗口读取要求的 TIFF 可直接引用，布局转换保持数值无损。
2. GPU 负责需要新 mask 的图像；主控负责 NAS IO、定量和索引。主机分工及并发起点见[资源安排](../project/discussions/image_preparation_workflow_20261020.md#5-多机资源与任务分配)。
3. 主控向远端发送核/膜合成 TIFF 与参数 JSON，接收 mask TIFF 及实际参数；远端缓存有容量上限。Study index 由主控单写，图像任务独立领取和恢复。
4. 大图外部分块需实现 halo、重叠实例匹配、重复去除和全图 label 合并；以接缝实例一致性验收后扩大运行。分割计算块、TIFF 存储 tile 和训练 crop 分别配置。
5. 每图先写临时结果；检查本次实际生成的产物及其依赖，通过后发布，并在 metadata 记录完成项与缺口。规则和输入未变的完成结果复用；规则或 mask 改变时更新 metadata 中的产物身份，并同步重建相关细胞索引及依赖产物。
6. 最小验收覆盖已有 mask、新分割、稀疏标签、退化通道、跨 TIFF tile 窗口、必要的大图接缝，以及 cell/mask/expression/crop/view 查询闭环。完成情况由[项目状态](../project/status.md)记录。

## 7. Crop 与训练视图

1. 基础窗口为原生 **224×224像素**，常规步长 **112**。每轴从0开始，规则网格后追加末端 `length-224` 起点以贴边，去除重复；末段重叠可超过50%。短轴不足224时起点为0，在右/下补零，真实范围由H/W记录，默认训练采样选择无padding块。
2. 物理视野为224乘对应轴的µm/px，像素尺度逐图核验。基础窗口直接读取图像与mask的相同范围。
3. 全图TIFF保存像素，`cells.crop_origins`保存每个cell实际相交的窗口左上角列表，允许部分实例。网格由H/W及上述规则重建。表达向量保持整细胞均值。
4. 训练组织筛选的工程起点：有效、非恒定蛋白/核通道最大投影→Otsu→3×3二值opening/closing各一次；组织占比至少0.30且存在有效蛋白。组织比例可按需计算及缓存，筛选只影响采样；完整cell/crop对应保留。无可用空间信号时记录组织占比不可估计。
5. Global224/local98在加载时生成，图像、实例mask、坐标共用几何参数；缩放mask使用最近邻，图像使用带抗混叠的bicubic并裁回[0,1]。实际可见标签映射回原图cell_index；每个14×14空间token的细胞对应由变换后mask确定。全部重叠crop继承来源患者/图像split。
6. 图像预训练候选为2个global、8个local，随机裁剪面积比例分别[0.48,1.0]和[0.16,0.48]，宽高比[3/4,4/3]；同步水平/垂直翻转各概率0.5，四种90度倍数旋转等概率。这些是训练候选，阶段安排见[训练规范](../architecture/training.md#4-joint-双视图与-ema)。
7. 推理与删除比较读取基线224窗口；删除先修改原图实例像素，再以同一几何参数生成所有受影响views。归一化统计和基线mask保持固定，编辑依赖由[编辑规范](../architecture/editing.md)维护。

```python
def axis_starts(length, size=224, stride=112):
    if length < size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts
```

## 8. 方法依据

强度与分割的试验依据见[分割探索](../experiments/segmentation_pilot_20261019.md)、[固定mask定量](../experiments/protein_quantification_pilot_20261020.md)和[非零P99比较](../experiments/nonzero_p99_pilot_20261020.md)；各报告保留当时配方及结果。Crop文献依据和CellWeft适配见[来源核对](../project/discussions/crop_sampling_sources_20261020.md)。当前生产实现、工程验收及完整DINOv2文档迁移的状态统一见[项目状态](../project/status.md)。
