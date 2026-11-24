# 表征评价

本页维护 Figure 2／3 的任务、读出及评分。模型接口见[模型规范](../architecture/model.md)，数据划分见[数据目录](../data/catalog.md#3-私有医院与划分)，实施进度见[当前状态](../project/status.md)。

## 1. 任务与报告

| 任务 | 输入与报告 |
| --- | --- |
| Image / patch marker reconstruction | Figure 2 M1–M4：由遮盖输入预测目标marker信号，见§3 |
| Cell expression readout | 共享图像patch表征与实例mask经cell adapter及marker条件化表达头，预测whole-cell测量 |
| Cell annotation / state | Figure 2 C1/C2：逐细胞表征接分类头；其他状态读出按实际标签选择 |
| Spatial inpainting / niche masking | Figure 3 S1：整区域全panel遮盖后的信号恢复，见§4.3 |
| Patch similarity retrieval | Figure 3 S4：完整组织patch检索、临床匹配与分子组成评价，见§4.6 |
| Tissue region classification | Figure 3 S2：image内crop bag经MIL输出ROI类别，见§4.4 |
| Extracellular / tissue interface | 使用有可靠标记和参考的胞外区域、ECM pattern或组织界面读出 |
| ROI / tissue / clinical task | Figure 3 S3：患者级MIL分类或生存预测，见§4.5 |

## 2. 共同评价规则

1. Figure 2／3 共用预训练前固定的患者划分。同一患者的所有 ROI、时间点、切片和 panel 同侧，已确认的跨来源重复共同分组。患者身份缺失时依次采用已确认 sample、物理 slide 或 image 分组。部分已连接来源的未连接记录保留 pending。内部测试与整项外部测试在预训练前留出。开发训练与验证可参加无标签预训练，监督训练拟合开发训练，开发验证选参。比较来源与固定 VirTues 重合时，按作者公开 SPORA 患者划分对齐：仅含 test ROI 的患者用于共同比较，含 train ROI 的患者归开发。Hoch 对 CellWeft 整项留出，共同评分限作者留出患者。PASCAL 传导式预测与 GMM 评分门按 §3.7 和 §3.9 单独执行。
2. 按[训练规范](../architecture/training.md)使用配套 EMA 与共享像素 decoder。每个模型的任务共用一个预训练 checkpoint，表征任务冻结编码器及实例读出，仅训练任务头。
3. 各方法共享固定成员、标签及评价真值。任务头使用可比容量和训练预算，像素恢复遵循 §3.3。先在 ROI 坐标遮盖再提取 crop，同一 ROI 共享遮盖，每个原始位置计分一次。基础窗口及边缘有效域见[准备规范§7](../data/preparation.md#7-crop-与训练视图)，S4 外部模型按共同中心读取各自原生输入。
4. 各dataset分别报告结果。空间遮盖回补按§4.3在dataset内对ROI等权汇总；其他任务按任务所需的marker、ROI、类别或患者计算，再以患者或实际独立分组单位汇总。需要估计不确定性时按患者或实际独立分组重采样；图像、patch和细胞数另列。Pearson无法定义时记为不可计算，并报告有效数量。

**固定成员与数据连接。** 读取 NAS `processed/_catalog/public/experiments.json` 的 `tasks[task].datasets[study].members`，每组用 `splits` 的并集或显式 `images` 之一展开成员。按 `use` 区分用途，按 `label_field` 和 `evaluation_group` 确定终点与评价组。外部主评价使用 `external_test`，Hoch 的 `control_description` 单列。用 `(study,image_id)` 连接图像，ROI 标签按 `image_id` 连接，病例按 `sample_id` 连接，细胞按 `cell_id` 连接。标签按任务 `source` 和 `label_field` 读取，类别保留来源字符串（Cords 复发为 `0.0`／`1.0`），生存时间及事件按数值解析。使用 `definitions`、`source`、`spatial_scope` 和 `labels_ref` 定位标签并解释共享组织标签。跨类别患者在各类独立去重。

**逐图观测与目标。** 用 `images.parquet` 连接 panel，由 `channels.parquet` 的有效 `tiff_page` 确定实际测量，按 `marker_id` 连接当版 `marker_embeddings.pt` 的 `marker_ids`。当 `valid_targets` 包含当前 `image_id` 时，以该图列表为准，显式空列表表示该图无目标；缺少该图键时取 panel `targets`。`excluded_markers_by_image` 同时作用于实际输入、目标、损失和评分。按固定任务成员读取，报告逐目标有效患者数及 ROI×目标实际覆盖。HuBMAP 聚合来源保持 `aggregate_pending`，明确原始 study 归属与重复关系后再归并。尚未闭合的成员及来源连接见[数据目录](../data/catalog.md#7-尚未闭合的来源)。

**固定划分的生成规则。** 运行直接消费上述 JSON。复现生成时，对 G≥2 个合格独立分组按 `max(1, min(G-1, floor(0.2*G+0.5)))` 确定测试配额，使用种子 20261118、稳定排序及 512 个候选分配。METABRIC 先兼顾 Ali／Danenberg 各自配额，再保留至少两组支持的 panel、区域和终点类别的两侧覆盖，随后平衡标签比例、ROI 总量与患者 ROI 数分布，以稳定排序解平局。开发内按 80:20 划分，开发不足两组的来源仅用于基础模型。测试标签仅用于预定分层。内部蛋白恢复按患者与目标 panel 以稳定 image_id 取一张 ROI，ForoughiPour 先固定 13／12 panel 配额，Karimi 按实际 panel。Piyadasa 外部按患者×panel family 取一图并按患者汇总。内部空间回补每患者一图，优先复用恢复 ROI。分类、MIL 与检索读取各任务全部有效成员。

## 3. Figure 2 蛋白恢复与细胞类型注释

### 3.1 研究问题与命名

工作标题：**Protein signal recovery and cross-dataset cell annotation**（蛋白信号恢复与跨数据集细胞注释）。

Figure 2先比较四项marker恢复条件：局部可见→当前ROI整通道隐藏→新dataset中的已见marker→全局未见marker；再评价内部与跨dataset的细胞注释，并以标注量曲线比较监督需求。

前两项沿用[VirTues](https://www.nature.com/articles/s41586-026-10884-y)的Independent masking和Marker masking命名，后两项按跨来源及未见marker预测命名。

### 3.2 Marker恢复的四项条件

训练可见性覆盖预训练及读出训练中的测量与预测监督。固定 marker 身份和知识向量可用于查询。M4 目标资格按开发来源的 canonical 蛋白核对保留测量，检查范围包括分割专用通道。VirTues 按公开训练 panel 报告可见性。

| 编号与名称 | 测试图像 | 目标marker的训练观测 | 当前测试ROI的目标marker观测 | 测试操作与研究问题 |
| --- | --- | --- | --- | --- |
| M1 Independent masking：局部蛋白回补 | 来源dataset的既定内部测试分组 | 同dataset开发图像中见过 | 部分空间位置可见 | 各通道分别遮盖部分patch；评价局部空间信息及其他marker支持的恢复 |
| M2 Marker masking：整通道蛋白回补 | 同一来源dataset的既定内部测试分组 | 同dataset开发图像中见过 | 整个ROI隐藏 | 遮盖目标marker完整通道；评价marker关联支持的预测 |
| M3 Cross-dataset marker prediction：跨dataset已见蛋白预测 | 整项外部测试dataset | 在其他训练dataset中见过 | 整个ROI隐藏 | 遮盖目标marker完整通道；评价新dataset及panel背景中的迁移 |
| M4 Unseen-marker prediction：全局未见marker预测 | 整项外部测试dataset | 所有训练来源均无该marker的测量与预测监督 | 整个ROI隐藏 | 以固定marker身份／知识查询目标通道；评价未见marker的推广 |

1. M1在当前ROI保留目标marker的部分观测，每个计分位置保留至少一个其他marker；采用满足该条件的按通道随机遮盖，并给出实际遮盖比例。
2. M2–M4每次将一个目标通道在整个ROI及其全部crop中遮盖。各方法使用相同的目标与可见通道集合。
3. 评价目标均须有实测真值。M3/M4可在同一外部dataset中分别选择训练已见和未见marker；marker身份沿用[数据合同](../architecture/data_contract.md)的marker_id及测量形式，评价目标读取正式任务清单。

### 3.3 共同预测流程与图像读出

| 方法 | 预测规则 | 参数与适用范围 |
| --- | --- | --- |
| CellWeft | 遮盖图像→冻结EMA图像encoder→共享marker条件化像素decoder | decoder在开发训练分组拟合、开发验证分组选参；Figure 2四项与Figure 3 S1共用一组权重 |
| VirTues | 使用公开`virtues-sp32`的encoder及原生像素decoder | 直接使用固定权重；患者划分依据作者公开队列与SPORA train/test |
| 均值基线 | 目标通道仍有可见像素时，用其可见像素均值；整通道隐藏时，取各可见蛋白通道均值的等权平均 | 使用共同评分域中的可见蛋白通道 |
| 线性基线：空间插值 | 用目标通道可见位置及其坐标预测隐藏像素 | 用于局部蛋白回补；插值方式及边界处理在开发验证部分确定 |
| 线性基线：Ridge | 以其他蛋白的同位置强度预测目标蛋白 | 用于训练已见目标的整通道及跨dataset预测；回归系数、截距和正则强度使用开发数据确定 |

Ridge依据开发与测试panel的可用marker为每个目标确定输入集合，不同panel可使用各自的输入集合；使用同时实测目标与该输入集合的开发像素拟合。训练采样预算与正则强度写入运行配置。

1. **读取任务。** 从正式`experiments.json`读取任务成员、panel输入marker、目标marker及固定的患者与开发训练／验证划分；结合逐图索引和通道表确定实际panel与有效通道，并应用逐图通道排除。各方法共享评价ROI、目标和遮盖种子。
2. **确定信号尺度。** 各模型保留原生输入处理；预测与真值按§3.4转换至既有P99.5 image.tiff评分域。CellWeft蛋白图像按[数据合同](../architecture/data_contract.md)读取；DNA等segmentation-only通道仅用于预先分割。whole-cell均值和实例mask用于细胞读出及监督，蛋白恢复输入为遮盖图像。
3. **生成遮盖。** 在原ROI坐标中先遮盖图像再提取crop。局部回补按marker和位置独立遮盖；整通道回补隐藏目标marker在ROI内的全部像素。遮盖在图像tokenizer汇总marker之前执行，保留观测状态以区分隐藏值和实测零值；重叠crop使用相同ROI遮盖。
4. **预测目标。** CellWeft通过冻结EMA图像encoder及已训练的共享marker条件化像素decoder输出目标像素，接口见[模型规范](../architecture/model.md)；VirTues通过固定原生encoder／decoder预测；简单基线按上表计算。M4目标的测量及预测监督在预训练、whole-cell目标和共享decoder训练中均不可出现，固定MAP身份仍可用于目标查询。
5. **合成ROI。** 将crop预测按固定覆盖权重合成到原ROI像素网格，输出转换到共同评分量纲；同一原始像素只参与一次评分。
6. **评分与汇总。** 隐藏实测值作为评分真值，按§3.4产生ROI×marker指标并汇总。结果记录任务、方法、dataset、panel、患者或实际分组、ROI、marker、遮盖设置、有效像素数、Pearson与MSE；保存实际使用的运行配置。

### 3.4 Marker恢复的评价

设`Ω`为某ROI内某目标marker被遮盖、具有有效实测值且位于规定评价覆盖范围的像素集合。有效实测背景参与评分，padding不参与。

共同评分域为P99.5归一化的`[0,1]`强度：`T(x)=clip(x/q995,0,1)`，其中`x`为线性强度，`q995`取该ROI×marker既有channels.parquet中的值，各方法共享。真值`y_i`直接读取既有image.tiff；CellWeft与基线在该评分域预测。VirTues输出`z`先按`expm1(z*(std+1e-8)+mean)`转换至线性强度，再经`T`获得`ŷ_i`。整通道隐藏时，模型从可见蛋白通道和目标 marker 身份预测。目标通道的 ROI q995 仅供评分端定义归一化真值或换算输出。Pearson 与 MSE 在同一 `Ω` 和归一化强度域上计算。

- **Pearson r ↑：** `Σ(ŷ_i−mean(ŷ))(y_i−mean(y)) / sqrt[Σ(ŷ_i−mean(ŷ))² × Σ(y_i−mean(y))²]`，均值与求和均在`Ω`上计算；衡量隐藏位置的空间信号变化是否一致。
- **MSE ↓：** `Σ(ŷ_i−y_i)² / |Ω|`；衡量共同尺度下隐藏位置的强度误差。
- 遮盖由共同ROI坐标mask定义，输出与评分保留原像素强度；共同遮盖网格与模型内部token分别记录。预测或真值方差为零等导致Pearson无法定义时记为不可计算，并报告有效数量；仍计算MSE。

1. **逐ROI与marker计算。** 各方法使用相同`Ω`；局部回补与整通道回补使用配对ROI和目标，按遮盖方式及比例分别报告。改变M1遮盖比例，比较局部观测减少时的恢复表现。
2. **逐患者汇总。** 同一marker及遮盖条件下，先平均患者内各ROI指标，再平均dataset内患者指标；无患者身份时使用正式划分的独立分组。
3. **逐dataset汇总。** 保留各marker结果，并对有效marker等权汇总；跨dataset总览对dataset等权，避免图像多或panel大的来源主导结果。各层级同时列出有效患者、ROI、marker与像素数量。
4. **报告变异。** 按患者或实际独立分组重采样估计不确定性；不同遮盖种子下的方法使用配对设置，次数由运行配置指定。
5. **解释迁移。** 跨dataset已见与全局未见蛋白分别报告，附目标panel重叠及训练可见性；后者按CellWeft开发测量与预测监督定义，VirTues按其公开训练panel说明对应目标的可见性。

### 3.5 细胞类型注释：任务、标签与比较方法

本轮比较CellWeft、固定virtues-sp32、KRONOS2、MAPS及PASCAL。三种FM冻结编码器后训练统一logistic regression；MAPS训练原生注释模型；PASCAL使用参考细标签作为传播种子。

| 任务 | 参考集及标签 | 待预测数据与GT |
| --- | --- | --- |
| Jackson内部注释 | 287位开发患者、571张ROI；source_cell_type，26类 | 70位内部测试患者、137张ROI；GT为source_cell_type |
| Danenberg内部注释 | 573位开发患者、632张ROI；meta_description，32类 | 144位内部测试患者、161张ROI；GT为meta_description |
| Xiao内部注释 | 19位开发患者、109张ROI；celltype20，20类 | 6位内部测试患者、43张ROI；GT为celltype20 |
| Jackson→Keren | 38位开发患者、39张TNBC ROI；source_cell_type | Keren整项外部测试40位患者、41张ROI；GT为source_label |
| Danenberg→Keren | 555位开发患者、598张肿瘤ROI；meta_description | Keren整项外部测试40位患者、41张ROI；GT为source_label |
| Xiao→Hoch | 19位开发患者、109张ROI；celltype20 | Hoch固定7位外部患者、9张ROI；GT为cell_type |

Annotation作为分类监督；三种FM的输入与表征提取沿用§3.7。Pred保留参考集原始细类，GT保留目标原始注释；类别编码器由参考训练标签建立。分类评分将已确认同义的原始标签归入同一语义类别，完整热图使用原始名称。正式experiments.json维护患者、图像、来源筛选和同义关系；作者原始类别及其细胞连接继续保存在各study已有annotations/cell_labels.parquet与metadata.json。 Jackson→Keren的来源限定为ER/PR/HER2阴性、术前无治疗的肿瘤ROI；Danenberg→Keren限定作者isTumour=True的ROI。

### 3.6 标注量曲线

1. 固定开发训练／验证患者、测试成员、模型读出与完整可用panel，按来源细类分层抽取20%、50%、80%的开发标注；训练及验证标签共同计入预算。
2. 每档抽样3次，种子42、43、44。同一重复按每类同一随机顺序取不同比例前缀；每个开发角色内取floor(fraction×n)，存在该类时至少取1个。
3. 五种方法共享训练标注细胞和验证标注细胞；PASCAL仅以其中的训练标注作为种子。每档记录各类实际训练／验证标签数。
4. 三种FM的完整输入表征每个checkpoint提取一次；各档只重新拟合标准化器和分类器。MAPS各档重新训练，PASCAL各档重新拟合LDA并传播。

### 3.7 各方法输入与实现流程

**CellWeft。** 采用预训练时联合学习的Mask-Adapter式细胞读出：224×224蛋白图像经EMA ViT-B／8得到784个768维空间patch features，结合与图像同步裁切的逐实例mask，由配套EMA cell adapter输出每细胞768维contextual embedding，再训练统一logistic regression。当前crop读取centroid落在crop范围内的细胞；边缘实例仅使用该crop内的mask及图像特征。验证／测试使用224／224无重叠网格、末端padding不回移，按centroid唯一归属；不跨crop拼接。whole-cell均值在完整image上预先定量，用于训练目标和评价，不作为独立细胞编码输入。逐细胞记录沿用正式cell_id及cell_index。

**VirTues virtues-sp32。** 使用公开固定checkpoint及其ESM marker表示，从input.tiff走作者标准化流程。调用官方SporaVirTuesWrapper.compute_cell_tokens：图像及实例mask先四周补120像素，再由compute_cell_tokens使用128窗口、8像素内部patch、stride=42提取空间summary tokens。按细胞与token的重叠像素数加权，跨窗口合并得到512维cell embedding。Mask用于表征汇总；返回的cell_ids连接正式实例身份。[官方wrapper](https://github.com/bunnelab/spora-bench/blob/main/models/virtues/virtues_model.py)及[细胞汇总函数](https://github.com/bunnelab/virtues/blob/main/virtues/utils/cell_tokens.py)。

**KRONOS2。** 沿CORAL原生细胞流程：从既有mask确定centroid，提取64×64细胞中心窗口，将窗口内目标细胞以外的像素置零。以既有P99.5 image.tiff作为[0,1]上游输入，执行官方marker名称适配及model.preprocess，再在FP32下前向，每个细胞读取768维CLS。首版使用对应队列完整可用蛋白panel；marker资源准备复用既有冻结编码器规范。[官方细胞教程](https://github.com/mahmoodlab/CORAL/blob/main/tutorials/3-Cell-Segmentation-and-Feature-Extraction.ipynb)。

**三种FM的共同分类头。** 用本档参考训练细胞拟合StandardScaler，再变换验证／测试特征；采用scikit-learn多类logistic regression，L-BFGS、L2、C=1、tol=1e-4、max_iter=1000、class_weight=None。每档保存来源细类顺序及预测概率。VirTues原生linear probe提供方法依据；上述显式数值为本项目统一运行设置。[官方linear probe](https://github.com/bunnelab/spora-bench/blob/main/spora_bench/benchmarks/run_cell_probe_tasks.py)。

**MAPS。** 以input.tiff和既有mask计算whole-cell marker均值，沿Spora runner按dataset逐marker做z-score。内部注释使用完整共同可用蛋白；跨dataset使用已保留同名marker_id的交集并固定列顺序，组合测量维持原身份。训练／验证使用既定患者和本档细标签；通过maps.cell_phenotyping.Trainer训练、Predictor预测，batch_size=128、max_epochs=300、min_epochs=100、patience=20，其余训练参数沿原生默认。类别编码只来自参考训练集，目标原始GT单独保留。[官方MAPS runner](https://github.com/bunnelab/spora-bench/blob/main/spora_bench/benchmarks/run_maps.py)。

**PASCAL。** 使用 whole-cell marker 均值，经 `arcsinh(x/5)` 形成 cells×markers 矩阵。内部使用本队列完整共同可用蛋白，跨 dataset 使用与 MAPS 相同的 marker 交集及顺序。以本档参考训练细标签为种子，其余待预测细胞编码为 `-1`。LDA 仅拟合种子细胞，维数为 `min(种子类别数−1,marker数)`。传导式 SNN 图使用参考种子与当前目标评价集合的无标签特征，目标真值用于评分。执行官方 LabelSpreading，`k=10`、`alpha=0.9`、`max_iter=300`、`tol=1e-15`、`min_shared=1`、`euclidean`。用官方 Unknown 阈值处理前最高传播分对应的参考类别作为 Pred，保留各类原始传播分。见[官方传播实现](https://github.com/xmuyulab/PASCAL-XMBD/blob/main/labelPropagation.py)及[运行入口](https://github.com/xmuyulab/PASCAL-XMBD/blob/main/run.py)。

### 3.8 逐细胞标签、分类指标与热图

1. 使用(study,image_id,cell_id)连接正式细胞记录、作者GT、方法Pred及实测表达。mask_label与cell_index按现有CellTable对应；共同评价集合由任务成员、明确cell_id和非空GT预先固定，各方法为该集合输出Pred；保留作者Unidentified、unknown等原始非空类别。
2. 标签读取各study已有annotations/cell_labels.parquet。Jackson使用source_cell_type（26类），Danenberg使用meta_description（32类），Xiao使用celltype20（20类），Keren使用source_label（16类），Hoch使用cell_type（10类）；完整类别及来源连接由原annotation文件和metadata.json维护。
作者标签连接复用已接收结果：Jackson按core及来源细胞编号连接；Danenberg按ImageNumber和ObjectNumber连接；Xiao使用已有唯一cell_id坐标映射；Keren使用作者类型栅格与现有实例至少90%的单类型覆盖；Hoch使用作者细胞编号或已确认的质心映射。具体匹配方法及来源字段保存在原annotation metadata。

3. 同义关系按下表固定。内部任务使用本dataset原始类别；跨dataset将同义Pred映射至目标GT类别用于评分，原始Pred与GT另行保留。

| 方向 | 参考原始细类 | 同义目标原始类别 |
| --- | --- | --- |
| Jackson→Keren | B cell；Macrophage；Endothelial | B cells；Macrophages；Endothelial |
| Danenberg→Keren | B cells；CD4^{+} T cells；CD8^{+} T cells；Macrophages；Endothelial | B cells；CD4 t cells；CD8 T cells；Macrophages；Endothelial |
| Xiao→Hoch | CD8 T cells | CD8+ T cell |

4. 将同义标签归一后，对每个目标语义类别c计算TP_c、FP_c、FN_c，precision_c=TP_c/(TP_c+FP_c)、recall_c=TP_c/(TP_c+FN_c)、F1_c=2TP_c/(2TP_c+FP_c+FN_c)。分类指标使用zero_division=0；未匹配的Pred按原类别参与计数，对真实类别产生FN。
5. macro-F1为目标测试集实际存在的语义类别集合C上的F1等权平均。C在任务内固定，所有方法、标注预算及重复共用。先用完整测试集混淆矩阵计算每个dataset的指标；患者作为不确定性重采样单位。
6. 热图使用完整Pred×GT矩阵：行是参考集预测细类，列是目标原始GT。保存计数N_pg及列比例H_pg=N_pg/sum_p(N_pg)，每列注明GT细胞数；无测试细胞的列显示NA。类别顺序固定；已确认同义的格子高亮，色阶表示预测比例。

### 3.9 GMM阳性判定

1. 读取固定测试成员的共同实测细胞，使用现有expression.parquet中P99.5 whole-cell marker均值x_im。逐图通道有效性及marker排除沿用正式selection规则。
2. 以dataset×panel×marker为单位汇集这些细胞；内部任务使用固定内部测试成员，跨dataset任务使用固定目标外部测试成员。GMM输入为该marker的实测强度。
3. 对z_im=log(1+x_im)拟合一维双组分GMM。均值较高的组分定义为阳性组分，b_im=1[P(阳性组分|z_im)>=0.5]，否则为0。
4. 每个dataset/panel/marker拟合一次。所有方法、20%/50%/80%标注预算及三个重复共享阳性判定；Jackson→Keren与Danenberg→Keren共享Keren的拟合结果。保存组分权重、均值、方差及拟合细胞数。
5. 阳性门用于评价实测蛋白表型；分类头与标签传播沿用§3.7的输入及训练流程。双组分GMM的蛋白阳性判定参考[IMC原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9059779/)；本节的P99.5输入、log1p变换和共同测试集合为本项目确定的适配。

### 3.10 Marker共表达与互斥

每个pair按原始细胞类别分别计算，表中的细胞范围说明其生物学解释。J、D、X、K、H分别指Jackson、Danenberg、Xiao、Keren、Hoch；JSON使用完整study名称及实际marker_id。

| 关系 | Marker pair与适用dataset | 原作者表型支持的解释范围 |
| --- | --- | --- |
| 互斥 | CD20—CD3e：J/D/K/X | B细胞与T细胞类别 |
| 共表达 | CD68—CD163：D/H | CD163表型的巨噬细胞 |
| 共表达 | CD14—CD68：X | 髓系细胞，重点MC6 |
| 共表达 | αSMA—Vimentin：J/K/X/H | 收缩性基质细胞 |
| 共表达 | αSMA—CD140b：D/X | 对应的基质及血管周表型；CD140b为本项目PDGFRβ测量名称 |
| 互斥 | J：vWF/CD31—PanCK + Keratin Epithelial；D：vWF/CD31—PanCK；K：CD31—PanCK [AE1/AE3] | 血管与上皮类别 |
| 共表达 | J：PanCK + Keratin Epithelial—CK8/18；D：PanCK—CK8/18 | 上皮类别 |
| 共表达 | ERα—CXCL12：D | 作者定义的相关肿瘤状态 |
| 共表达 | SOX10—Melan-A：H | 黑色素瘤类别 |

1. 对每个pair，选取两种marker均有有效实测值的细胞，分别按Pred原始细类和GT原始类别分组；同义关系沿用§3.8，为对应类别计算差值。
2. 在每个分组中，n11为两个marker均阳性的细胞数，n10、n01分别为仅A或仅B阳性的细胞数。两种marker均阴性的细胞计入类别总细胞数。
3. 共同阳性比例R_AB=n11/(n11+n10+n01)。共表达组将其作为共同阳性比例；互斥组将其作为MECR，表示互斥marker的双阳性比例。分母为0时记为NA。
4. 对应Pred与GT类别分别报告R_Pred、R_GT、Delta_R=R_Pred−R_GT及abs(Delta_R)，以GT表型为参照。其他原始类别保留其各自pair比例和计数。
5. 在每个dataset/panel/pair/类别内，同一患者合并全部有效ROI的n11、n10、n01后计算比例；dataset内对有效患者等权汇总。Delta_R使用Pred与GT均可计算的患者配对统计，并列出有效患者数。
6. 保留类别总细胞数、至少一个marker阳性的细胞数及双阳性细胞数；结果按dataset、panel、原始类别、pair、方法、标注预算和重复种子组织。不同panel的marker身份及单独比例保留。
7. [cellAdmix](https://www.nature.com/articles/s41588-025-02497-4)与[Xenium SPLIT研究](https://www.nature.com/articles/s41592-026-03089-8)提供自身类型信号与异类信号混入的评价依据；双阳性/至少一个阳性的MECR公式来自[Hartman与Satija原始研究](https://elifesciences.org/reviewed-preprints/96949)。本项目以原作者表型及上述九组蛋白pair评价注释一致性。

### 3.11 共同输出与实施顺序

1. 读取正式任务成员、开发训练/验证角色、标签字段及共同有效测试细胞，连接既有mask、表达和作者注释。
2. 三种FM提取固定逐细胞表征；按§3.6的共同标注细胞训练linear probe或MAPS，PASCAL按相同参考训练标签建立种子并传播。每个方法、预算和重复输出原始Pred及逐类概率或传播分。
3. 按§3.9拟合并固定GMM阳性判定；按§3.8计算分类指标和完整热图，按§3.10计算共表达与互斥指标。
4. 分类结果保存原始Pred、GT、语义评分标签、参考类别顺序、各类precision/recall/F1及macro-F1。Marker结果保存pair身份、panel、分组、计数、R_Pred、R_GT、Delta_R、abs(Delta_R)及有效患者数。
5. 各次重复单独评分，再报告三次重复的均值和标准差；需要区间时以患者为单位重采样，各方法使用配对患者样本。无患者身份时使用正式划分的独立分组。
6. 以上为已确定的评价协议；实施依次完成输入/标签接线、五种方法的小规模推理、固定表征导出、标注量实验及统一评分。

## 4. Figure 3 组织结构理解与表型应用

### 4.1 研究问题与命名

工作标题：**Spatial recovery, tissue classification, patient phenotype prediction and tissue retrieval**（空间信号恢复、组织分类、患者表型预测与组织检索）。

主线为恢复组织信号→识别局部结构→汇总整体表型→检索相似组织。四项独立评价共享预训练checkpoint；S2–S4输入完整实测组织。

| 编号与名称 | 输入及读出 | 研究问题与主指标 |
| --- | --- | --- |
| S1 Niche masking / spatial inpainting：全panel空间回补 | 当前image patch内随机网格全marker遮盖→逐像素重建 | 可见组织能否恢复隐藏像素强度；Pearson r、MSE |
| S2 Tissue region classification：局部组织识别 | 完整image／ROI→crop bag→Mean pooling／gated ABMIL／TransMIL→ROI类别 | 能否识别ROI组织类别；macro-F1 |
| S3 Tissue phenotype prediction：整体组织表型预测 | 患者组织patch→MIL→分类／生存读出 | 能否汇总患者表型；macro-F1／AUROC、C-index |
| S4 Tissue similarity retrieval：相似组织检索 | 完整组织patch固定表征→同study全部gallery排序 | 母ROI标签一致性、逐字段临床匹配及SWD；Precision@1／5／10、mAP@10 |

空间遮盖回补沿用[VirTues](https://www.nature.com/articles/s41586-026-10884-y)的niche masking术语，采用各通道同步的随机空间网格遮盖。Figure 2局部蛋白回补在目标位置保留其他marker；空间遮盖回补在该位置隐藏整个输入panel，具体流程见§4.3。

### 4.2 数据使用

沿用 §2 的划分。开发分组用于 S1 共享 decoder、S2 image 级 MIL、S3 患者级 MIL 及 S4 参考库。内部测试分组用于四项评价，整项外部 dataset 用于 S1 跨来源回补。S4 的留出查询、位置和图库见 §4.6。

局部组织识别与整体组织表型预测共用内部患者留出及开发训练／验证划分。开发患者可参加无标签预训练；内部测试患者在CellWeft预训练前留出。两项任务比较CellWeft、固定VirTues、固定KRONOS2、固定Eva及§4.7的marker均值基线；各来源分别训练任务读出。

**共同特征提取。** 每个方法为一个完整输入窗口生成一条向量；模型窗口与物理视野沿用[Figure 4冻结编码器流程](clinical/frozen_encoders.md#1-已确认的整体流程)，按原图坐标记录实际覆盖。

| 方法 | 资源与输入适配 | 窗口及读出 |
| --- | --- | --- |
| CellWeft | 同一预训练checkpoint的EMA图像主干；完整image.tiff及对应marker身份和观测状态 | 原生224×224；内部8×8 patch构成28×28＝784个空间tokens，每crop读取一个768维CLS向量 |
| VirTues | bunnelab/virtues的virtues-sp32/model.safetensors；按UniProt准备640维ESM-2表及同序channel_ids；input.tiff经1 µm/pixel适配、P99截断、log1p、固定逐marker标准化及所选滤波 | 128×128；encoder.forward_list([tile], [channel_ids], multiplex_mask=None)，平均patch_summary_tokens[0]的16×16网格，得到512维 |
| KRONOS2 | MahmoodLab/KRONOS2及marker metadata；image.tiff按逐图tiff_page读取，同序marker_names；按官方接口登记必要的新marker，model.preprocess后执行模型前向 | 原生256×256；模型返回768维CLS |
| Eva | yandrewl/Eva的Eva_model.ckpt、官方config.yaml、名称字典及GenePT资源；image.tiff与可映射biomarkers同步选择，保持相同通道顺序 | 原生224×224；extract_features(cls=False, channel_mode="full")平均非CLS空间token，得到768维 |

1. **读取成员与图像。** 从experiments.json读取对应任务、用途、标签及患者分组；结合images.parquet、sample_metadata、panel_mapping.csv与逐图channels.parquet连接真实输入。input.tiff按output_channel_index读取，image.tiff按tiff_page读取；应用既有逐图通道排除。每个来源按实际有序marker列表分批，保留每个模型的输入通道及未映射通道记录。
2. **准备资源。** VirTues从同一ESM目录构建bank及channel_ids，以MultiplexVirtues(prior_bias_embeddings=bank)初始化后加载固定state_dict；KRONOS2使用AutoModel.from_pretrained及Kronos2Extractor，新marker沿novel_markers→prepare_slide／pool_marker_stats→register_additional_markers准备；Eva在源码根目录定位marker_embeddings/GenePT_embedding.pkl，实例化前设mask_ratio=0，再用load_from_checkpoint恢复权重。具体加载代码由[资源流程§8](clinical/frozen_encoders.md#8-模型加载与原生调用的具体接线)维护。
3. **固定统计与推理。** 逐图P99.5 image.tiff直接复用；模型所需固定公开统计可直接使用，需从本项目估计的跨图统计只拟合开发训练患者，验证和测试复用。完整输入的逐图预处理可读取整张当前ROI。所有编码器eval、requires_grad_(False)、inference_mode；VirTues沿官方CUDA autocast后转float32，KRONOS2及Eva前向使用float32。CellWeft沿训练的观测与有效域规则处理输入，读取图像主干CLS；细胞任务另用同步裁切的mask及匹配EMA cell adapter，详见[模型规范](../architecture/model.md)。推理精度随统一框架配置固定，导出为float32。
4. **切窗与导出。** CellWeft 基础网格见[准备规范§7](../data/preparation.md#7-crop-与训练视图)。外部模型采用各自原生窗口与预处理。复用确认的有效组织范围并记录组织覆盖及 ownership。每行特征保留 feature_row、study、患者／分组、image_id、panel_id、原图窗口坐标、物理范围、有效域和覆盖归属。各编码器保存与索引逐行对应的 float32 特征，维度见本节读出表。
5. **训练范围。** 仅训练输入维度适配层、MIL及分类／生存头。开发训练拟合，开发验证选参、选择checkpoint及二分类阈值；锁定后评价内部测试。S2、S3分别按image bag和患者bag训练，均比较Mean pooling／gated ABMIL／TransMIL及可比任务头；MIL定义见§4.5第2步。各编码器使用相同划分、同类读出容量、相同搜索次数及重复种子预算。

真实窗口特征提取和读出训练见 §4.4、§4.5 及[冻结编码器实施交付](clinical/frozen_encoders.md#94-实施时的完成标志)。

### 4.3 S1：全panel空间回补

空间遮盖回补评价模型能否利用当前image patch内的可见组织，恢复隐藏位置全部目标marker的逐像素强度。比较CellWeft、固定VirTues、固定Eva、均值及空间插值；KRONOS2没有重建decoder，本任务不纳入。

**评价成员与目标。** 唯一成员入口为NAS的processed/_catalog/public/experiments.json，读取tasks.spatial_recovery.datasets中的members、targets及逐图valid_targets，应用现有逐图通道排除。当前11个来源共759张评分ROI，内部133张、整项外部626张；患者划分及marker目标沿用正式入口。目标身份、有效测量与模型名称映射分别处理，模型映射问题不自动改变任务目标。

| 来源 | 评价角色 | 评分ROI数 |
| --- | --- | --- |
| Bandyopadhyay2024_CODEX_BoneMarrow | 整项外部 | 12 |
| ForoughiPour2026_IMC_TNBC | 内部测试 | 25 |
| Hoch2022_IMC_Melanoma | 整项外部中的作者留出患者 | 7 |
| Karimi2023_IMC_BrainTumours | 内部测试 | 37 |
| Lin2026_IMC_CRLM | 内部测试 | 7 |
| Malchiodi2024_IMC_PancreaticCancer | 内部测试 | 10 |
| Piyadasa2025_MIBI_Glioma | 整项外部 | 607 |
| Rahim2023_MIBI_HNSCC_LN | 内部测试 | 6 |
| Risom2022_MIBI_BreastCancer | 内部测试 | 20 |
| Salie2025_IMC_HCC | 内部测试 | 20 |
| Schurch2020_CODEX_ColorectalCancer | 内部测试 | 8 |

**共同输入、遮盖与强度。**

1. 依据images.parquet、panel_mapping.csv和逐图channels.parquet读取实际通道。input.tiff按output_channel_index索引，image.tiff按tiff_page索引；图像、名称及蛋白向量保持同序。
2. 本任务在原ROI坐标读取224×224原生像素image patch。CellWeft训练切窗224／112，验证和测试224／224无重叠，末端padding不回移；既有crop_origins需与相应角色网格核对，不能代替新验证／测试位置。VirTues按输入尺寸构造原生8×8 token网格，Eva配置输入为224；Figure 4的128窗口及1 µm/pixel重采样设置属于其组织特征提取流程，本任务保留原像素网格。物理像素间距、有效域和padding随位置记录。
3. 在原ROI的共同8×8评分／遮盖网格上按运行配置的比例与种子无放回抽取隐藏位置，各方法复用同一mask。CellWeft内部patch也是8×8；实际位置按ROI与crop坐标映射，padding及不完整实测域以像素观测状态表示，隐藏真值不进入图像输入。比例分母为包含有效实测像素的共同网格数，padding不计入；逐窗记录实际隐藏比例。该位置所有输入通道同步隐藏，各窗口保持一致，预测仅使用当前image patch的可见输入。
4. 在各通道token汇总前应用全panel空间遮盖，并保留有效域和观测状态；恢复路径直接读取遮盖图像。
5. 评分真值使用既有P99.5 image.tiff，保留其原像素网格与float32强度。q995沿用既有测量尺度。模型特有预处理所需统计及来源写入运行配置；需要估计的模型统计使用开发分组，按图像计算的输入归一化只读取当前可见像素。本任务关闭高斯等跨像素滤波，不让隐藏真值经滤波或重采样进入可见输入。
6. 所有预测转换到相同的image.tiff强度域并统一截到[0,1]。保留浮点数，不做逐图预测min-max缩放或依据隐藏真值拟合强度变换。缺失的模型统计或身份映射明确记录，不能用任意默认值替代。

**VirTues：固定encoder及原生像素decoder。**

1. 使用bunnelab/virtues的virtues-sp32/model.safetensors。按实际marker映射UniProt，准备官方640维ESM-2表及同序channel_ids；按同一目录的排序构造bank及行号，以MultiplexVirtues(prior_bias_embeddings=bank)初始化后，通过load_state_dict(load_file(weights_path))加载权重。资源构建沿用[冻结模型资源流程](clinical/frozen_encoders.md#82-virtues先建立蛋白表再恢复-checkpoint)。
2. 从input.tiff读取对应原生窗口，执行P99截断、log1p及逐marker标准化；P99按可见实测像素计算，均值／标准差使用已明确来源的固定统计。构建[C,28,28]布尔multiplex_mask，在相同空间网格对全部通道置True。
3. 在eval、requires_grad_(False)及inference_mode下调用model([tile], [channel_ids], multiplex_mask=[mask])，使用decoded_multiplex[0]，其形状为[C,224,224]。encoder和decoder均直接使用公开固定权重；CUDA前向沿用官方重建教程的autocast，输出转float32后适配及评分。
4. 对每个marker执行expm1(prediction * (std + 1e-8) + mean)，回到线性强度后按同一q995映射至评分域。P99截断本身不可逆，不将该转换描述为恢复已被截掉的原始计数。[官方重建接口](https://github.com/bunnelab/virtues/blob/main/virtues/modules/multiplex_virtues.py)及[强度变换](https://github.com/bunnelab/spora-io/blob/main/spora_io/utils/dataset/standardize.py)。

**Eva：固定encoder及原生像素decoder。**

1. 使用yandrewl/Eva的Eva_model.ckpt及官方config.yaml、GenePT_embedding.pkl。初始化前设置mask_ratio=0，保留224输入、8像素token及原生网络结构；以Eva源码根目录为工作目录，使marker_embeddings/GenePT_embedding.pkl可定位；用load_from_checkpoint(checkpoint_path, conf, device)加载，前向使用float32。名称准备与资源路径沿用[冻结模型资源流程](clinical/frozen_encoders.md#84-eva先准备-genept-路径和推理配置再加载权重)。
2. 读取image.tiff窗口，图像和marker名称同步映射到Eva官方字典。按每窗实际通道构建[C,784]的infer_mask，1表示隐藏，各通道共用空间mask。首版逐窗推理；当前公开源码不能直接接受[B,C,784]的独立mask批次。
3. 在eval、requires_grad_(False)及inference_mode下调用model.recon(image_bhwc, marker_in=[names], marker_out=[names], infer_mask=mask)，输入形状为[1,224,224,C]。使用返回的image_recon，转换为[C,224,224]；另两个返回值不参与强度评分。
4. 该入口直接执行蛋白图像重建，不调用Figure 4的extract_features，也不经会裁掉末三通道的MIF分支。输入名称和输出目标分别核对官方字典与GenePT／checkpoint资源。无法映射的输入通道使该panel当前推理未完成，不静默删通道；无法预测的目标记录缺失及覆盖数，不以相近蛋白或随机向量替代。[官方重建入口](https://github.com/YAndrewL/Eva/blob/main/Eva/eva.py)、[外部mask实现](https://github.com/YAndrewL/Eva/blob/main/Eva/mae.py)及[marker字典](https://github.com/YAndrewL/Eva/blob/main/utils/constant.py)。

**CellWeft：共享marker条件化像素decoder。**

1. 预训练后冻结EMA图像encoder，开发训练分组拟合marker条件化像素decoder，开发验证分组选参；训练覆盖单marker局部遮盖、整通道遮盖及全panel空间遮盖。Figure 2 M1–M4与Figure 3 S1的内部和跨来源测试共用一组decoder权重。
2. 共享 decoder 接收空间 patch 表征、位置和目标 marker 身份，输出与 image.tiff 同量纲的 [M,224,224] 强度图和有效域。结构及首版参数见[模型规范§7](../architecture/model.md#7-图像-marker-重建读出)。
3. 只在任务规定的隐藏有效像素上计算 Pearson 与 MSE。训练损失及有效重建范围见[训练规范§5](../architecture/training.md#5-预训练后的像素-decoder)。

**简单基线：当前image patch内逐marker执行。**

| 方法 | 具体计算 | 参数学习 |
| --- | --- | --- |
| 均值 | 计算该marker可见有效像素的均值，将隐藏位置填为该值 | 无 |
| 空间插值 | 用该marker可见像素的二维坐标及强度作分片线性插值；可见点凸包之外使用最近邻值 | 无 |

两项均使用共同评分强度域及相同mask。空间插值的采样点与三角剖分仅来自可见位置。Figure 2的同位置跨marker Ridge不用于本任务。

**像素评分、汇总与结果字段。**

1. 将同方法窗口预测按原坐标合成；重叠区域对预测等权平均，每个原图像素计一次。固定隐藏mask、真值及评价覆盖；padding、未测量位置及既有排除通道不参与评分，覆盖内有效实测背景参与。
2. 对每个ROI×marker×遮盖条件，在隐藏有效像素集合Ω计算Pearson r及MSE。MSE=sum((prediction-truth)^2)/|Ω|；Pearson的均值、协方差及方差均在同一Ω上计算。常量或有效值不足时Pearson记为不可计算并列有效数量，仍计算可定义的MSE。
3. 先平均同一ROI×marker的遮盖重复，再在ROI内对目标marker等权平均，最后在dataset内对ROI等权平均。各dataset单独汇报；内部／外部、遮盖比例及已见／未见marker分别列示。
4. 模型无法输出的既定ROI×marker记录缺失和原因。完整任务分数要求覆盖相同既定目标；覆盖不齐时保留已完成的逐目标结果及覆盖数，不将不同目标集合的均值直接用于方法排序。
5. 逐ROI×marker记录dataset、panel、ROI、患者或既有分组、方法、遮盖条件及种子、有效隐藏像素数、Pearson、MSE与预测覆盖；另保存dataset汇总和实际运行配置。运行配置包含资源路径、通道适配、模型统计及强度转换、mask比例／种子、窗口覆盖和合成规则，不生成实验或划分迭代版本号。

### 4.4 S2：局部组织识别

研究问题：完整ROI的固定空间表征能否识别其来源组织类别。采用完整image／ROI→crop表征bag→MIL→一个ROI类别；标签及评分单位均为ROI。

**成员与类别。** 读取tasks.tissue_classification的members、evaluation_group及labels。CRLM使用肿瘤核心（TC）、浸润边缘（IM）、瘤周肝组织（PT）三类，selected_tls保持描述；Zhuhuman使用原位腺癌（AIS）、微浸润性腺癌（MIA）、浸润性腺癌（IAC）及正常肺组织（Normal）。当前标签覆盖整张采集ROI，ROI内精细边界未提供。标签来源连接及空间范围由正式labels维护。

| 来源 | 开发训练ROI／患者 | 开发验证ROI／患者 | 内部测试ROI／患者 | 测试ROI类别支持 |
| --- | --- | --- | --- | --- |
| Lin2026_IMC_CRLM | 196／22 | 49／6 | 63／7 | TC 17、IM 24、PT 22 |
| Zhuhuman2025_IMC_Lung | 826／33 | 200／10 | 246／11 | AIS 79、MIA 73、IAC 90、Normal 4 |

上述分类成员分别为开发245／1026张、测试63／246张ROI；CRLM另有16张开发、3张测试TLS ROI作描述。Zhuhuman Normal测试支持为1位患者的4张ROI，该类F1列作描述，不估计独立患者间的类别区间。同患者可含多个类别，患者总数按去重患者统计。

1. **提取完整输入表征。** 按§4.2对每个合格crop提取固定向量h_w，同一image／ROI内全部合格crop平铺为[N_r,d]的bag，保留窗口边界、中心、有效组织范围及coverage／ownership索引。CellWeft每crop使用CLS；外部模型使用原生窗口向量。窗口覆盖和未覆盖范围另列。
2. **训练ROI级MIL。** 每个来源、每种方法分别训练Mean pooling／gated ABMIL／TransMIL及同类分类头，定义与§4.5第2步一致；K分别为3或4，使用交叉熵。患者均匀、患者内ROI均匀采样或使用等价loss权重；分类loss施加于image bag的ROI输出。开发验证macro-F1用于选参，测试输出K类概率及argmax类别。
3. **计算ROI级指标。** 对患者p的R_p个ROI，每个ROI权重为1/R_p；累积加权混淆矩阵C，再计算各类F1_c=2*C_cc/(2*C_cc+FP_c+FN_c)及固定K类的患者等权ROI macro-F1。报告加权混淆矩阵及原始患者、ROI、窗口和各类支持数；各患者总权重相同。
4. **汇总与不确定性。** 两个来源分别汇报。按患者cluster重采样，每次保留其全部ROI、重算权重及指标；方法比较使用相同重采样患者。重复训练种子分别记录，再汇报均值和波动。
5. **后续实施交付。** 每个既定评分ROI有可回溯的crop bag索引、各MIL的类别概率与预测；输出患者等权ROI macro-F1、各类F1及混淆矩阵。无合格窗口或模型输入未完成的ROI记录覆盖与原因；在相同已完成成员上比较方法，并列完整既定成员覆盖。

### 4.5 S3：整体组织表型预测

研究问题：冻结组织窗口表征经统一MIL后能否预测患者的病理类别、分子状态及OS。读取tasks.patient_prediction，按每个终点的既定图像成员建立患者bag，每患者输出一个结果。开发训练、开发验证及内部测试保持正式患者划分；各终点分别训练读出。

**当前终点与有效患者。** 下表括号内为按类别顺序的患者支持；人数由任务图像经sample_id连接sample_metadata.patient_key去重，并依据正式患者组确定划分。

| 来源与终点 | 类别顺序 | 开发训练 | 开发验证 | 内部测试 |
| --- | --- | --- | --- | --- |
| Cords2024_IMC_Lung组织学类型 | 腺癌／鳞癌 | 655（396／259） | 162（97／65） | 200（124／76） |
| Cords2024_IMC_Lung分级 | 1／2／3 | 681（32／315／334） | 169（12／67／90） | 206（18／99／89） |
| Cords2024_IMC_Lung作者复发状态 | 0／1 | 667（338／329） | 165（100／65） | 201（116／85） |
| Danenberg2022_IMC_BreastCancer PAM50 | Luminal A／Luminal B／HER2／Basal／Normal-like | 391（148／103／36／56／48） | 95（34／29／13／11／8） | 123（45／29／12／20／17） |
| Danenberg2022_IMC_BreastCancer ER | 阳性／阴性 | 445（351／94） | 105（85／20） | 140（104／36） |
| Danenberg2022_IMC_BreastCancer HER2 | 阴性／阳性 | 391（346／45） | 95（82／13） | 123（111／12） |
| Danenberg2022_IMC_BreastCancer分级 | 1／2／3 | 444（35／165／244） | 108（11／41／56） | 141（20／44／77） |
| Fischer2023_IMC_Breast OS | 死亡／删失 | 462（122／340） | 112（21／91） | 128（32／96） |

分级保留作者1／2／3三类。Cords复发项使用source_relapse的作者0.0／1.0状态；Danenberg HER2使用source_erbb2_positive的False／True二类。Fischer使用os_time、os_event、os_origin及os_time_unit，当前单位为月、起点为diagnosis，os_event=1为死亡、0为删失。当前入选患者的终点值完整且患者内一致。

1. **建立患者bag。** 按§4.2提取窗口向量，将当前终点成员中同一患者全部合格image／ROI的crop直接平铺为[N_p,d]，不先形成逐image汇总向量。CellWeft使用逐crop CLS，其他模型使用原生窗口向量；同一编码器的三种MIL复用同一bag索引。保留ROI、panel、原图坐标及覆盖归属，袋内次序固定，实际窗口数及覆盖随结果记录。可复用同一模型特征缓存，按各终点成员分别组bag；患者数和事件数按去重患者计算。
2. **复用三种MIL。** Mean pooling对bag内窗口向量等权平均后接任务头；gated ABMIL对各向量计算tanh／sigmoid门控分数，经softmax归一化后加权汇总；TransMIL经输入维度适配、bag内Transformer上下文建模和患者表征读出后接任务头。四种编码器分别接同一MIL实现及可比任务头，输入适配层按d配置并随读出训练。每种encoder×MIL×来源×终点分别报告结果，验证集选择配置后锁定测试。
3. **分类训练与输出。** 使用交叉熵输出K类概率，分级为三分类，PAM50为五分类，其余当前端点为二分类。开发验证macro-F1用于选参；多类用argmax，二分类阈值在开发验证内确定并锁定。测试报告macro-F1、各类F1、混淆矩阵；二分类另用阳性概率计算AUROC。
4. **OS训练与输出。** 生存头输出单一风险值，越高表示死亡风险越高。采用Cox负部分对数似然，并列事件时间使用Efron处理；风险集由完整开发训练队列形成，事件时点t只纳入仍在风险中的训练患者R(t)={j:T_j>=t}。分块计算bag表示时仍须在完整训练风险集上形成损失，验证／测试患者不进入训练风险集。开发验证按Harrell C-index选参，测试以同方向风险值计算Harrell C-index并处理删失。
5. **患者级汇总。** 分类每位患者一个预测，Harrell C-index按可比患者对计算；各来源、终点及MIL读出分别汇报。训练重复和患者bootstrap分别记录，方法间使用相同重采样患者。生存报告有效患者、死亡事件、删失数及可比患者对；不可计算指标保留数量和原因。
6. **后续实施交付。** 每个既定有效患者有回溯到ROI窗口的bag索引，以及类别概率／预测或OS风险值；形成按encoder、MIL、来源及终点分层的指标表，并列实际预测覆盖。当前阶段完成协议与有效人数核对，真实训练及测试结果在实施后登记。

患者隔离沿用[VirTues的公开划分原则](https://www.nature.com/articles/s41586-026-10884-y)及当前正式患者组。双中心胃癌的患者对应、治疗时点、D／V设计和配对比较继续由[Figure 4协议](../paper/figure4_gastric_cancer.md)维护。

### 4.6 S4：相似组织检索与metadata描述

**研究问题与训练范围。** 用完整组织image patch的固定表征，能否从其他患者中检回组织标签一致、临床信息相近及分子组成相近的patch。比较CellWeft、固定VirTues、固定KRONOS2、固定Eva、marker均值和随机排序；四编码器资源、原生输入与向量读出沿用§4.2及[冻结编码器流程](clinical/frozen_encoders.md)。本任务不训练MLP、MIL或检索投影；编码器冻结，需由本项目估计的跨图统计及marker均值标准化只拟合开发训练患者。

1. **读取成员与病例连接。** 从正式`experiments.json`的`tasks.tissue_retrieval`读取query、gallery及`labels_ref`，结合逐study的images、sample与channels表读取实际输入。首版query来自内部测试患者，gallery来自开发训练及开发验证患者；同study全部gallery构成候选库，`evaluation_group`只分层query。组织标签由`tasks.tissue_classification.labels`维护；临床字段按本节来源连接到位置索引，保留未知值。

   | 来源 | Gallery患者／ROI | Query患者／ROI | 评价分层 |
   | --- | --- | --- | --- |
   | Lin2026_IMC_CRLM | 28／261，含TC／IM／PT 245及作者选取TLS 16 | 7／66，含TC／IM／PT 63及TLS 3 | TC／IM／PT主评价；TLS查询涉及2位患者，仅单列描述，TLS gallery保留 |
   | Zhuhuman2025_IMC_Lung | 43／1026 | 11／246 | AIS／MIA／IAC／Normal；Normal查询仅1位患者／4 ROI |

2. **生成query和gallery位置。** 在去除既有padding后的ROI坐标上，query与gallery均使用224×224、步长224的无重叠评分网格；末端以padding补全，不回移。共同中心的224×224框作为`score_box`，用于组织筛选、marker均值、展示及分子评分。各模型在同一中心读取原生128／224／256输入，超出实测ROI的部分padding，并保存实际输入框与有效域。Gallery虽来自开发患者，其冻结检索导出也使用这一网格；训练增强使用的224／112网格与此分开。

   在共同 `score_box` 复用既有组织估计。有效组织像素占完整 224×224 评分框面积的比例须严格大于组织阈值（默认 0.50），且实测 downstream 通道在有效域内有非零信号。Padding 仅计入面积分母。所有方法使用同一批位置。`retrieval_patch_id` 由 study、ROI 及原图中心确定，query／gallery 角色另存，物理框按逐轴像素间距记录。实测背景和零值按组织估计判定。CellWeft 对原生 224 输入提取图像 CLS，旧 `crop_origins` 仅用于解释训练网格。报告通过筛选的实际 patch 数。

3. **提取固定向量并排序。** 按§4.2的实际有序marker及各模型预处理提取向量：CellWeft的768维CLS、VirTues的512维summary均值、KRONOS2的768维CLS、Eva的768维非CLS均值。四模型共享位置ID，分别保存输入框与实际marker。marker均值基线从P99.5 [0,1] image.tiff的共同框内有效组织及实测域计算，按本study固定蛋白身份顺序排列，沿用§4.7的开发训练标准化、缺测填标准化零及presence仅用于缺失处理规则；DNA和辅助通道不纳入蛋白向量。

   每方法内部L2归一化后，以精确余弦相似度降序排列，同分按`retrieval_patch_id`升序排序。临床值不参与向量或排序。非有限或零范数向量无效；任一gallery特征无效时，该方法×study×条件的检索记未完成，不以缩小后的库计分；仅query特征无效时，该query指标不可计算并报告覆盖，方法间配对比较使用共同有效query。同患者不同patch均可命中并计分；另报原始Top-k不同ROI数与不同患者数。随机排序使用同一patch库；病例展示可合并患者，但保留命中排名。

4. **计算组织标签指标。** CRLM以母ROI的TC／IM／PT一致定义相关，Zhuhuman以AIS／MIA／IAC／Normal一致定义相关；这是母ROI标签来源一致性，标签粒度仍为ROI。TLS查询不并入主分数，TLS候选对TC／IM／PT查询为不同来源标签。

   gallery为空时全部评分不可计算并报告实际覆盖；非空时令`k_eff=min(k,N_gallery)`、`r_q(i)`为第i名的相关指示、`R_q`为完整gallery相关patch数，计算`P_q@k=sum(r_q(i),i<=k_eff)/k_eff`与`AP_q@k=sum(P_q@i*r_q(i),i<=k_eff)/min(R_q,k_eff)`。主指标为Precision@1／5／10及mAP@10；完整排序AP以`R_q`作分母，汇总的全排序mAP为辅助结果。非空库中`R_q=0`时P为0、AP不可计算，另报无相关候选及有效查询数。

5. **独立计算临床相近性。** 组织一致性与临床字段相近性分别报告；原始Top-k指各正常／打乱／基线条件自身未经临床字段筛选的排序。对字段已知的query，在原始Top-k中记字段已知候选数为`K_qf`、其中同值数为`M_qf`，报告匹配率`M_qf/K_qf`与metadata覆盖率`K_qf/k_eff`；`K_qf=0`时匹配率不可计算、覆盖为0。query字段未知者不进入该字段评分，另报查询患者／ROI／patch的字段覆盖。年龄仅在原始Top-1及Top-10已知配对中报告平均绝对差，单位为岁，同时报告覆盖。

   仅对分类字段，补充表可从同一完整排序中取该字段已知候选的保序子序列，以同值为相关，按步骤4公式在子库重算`k_eff`、`R_q`及条件P／AP；各模型使用同一已知子库，空子库不可计算，明确标注条件结果。筛选依据是字段已知性，不按query字段值预先筛候选。每字段单列，不合成总临床分。

   | 来源 | 临床比较字段 | 当前患者支持及解释 | 随返回结果展示 |
   | --- | --- | --- | --- |
   | Lin2026_IMC_CRLM | 组织学亚型、记录复发状态、KRAS、分化程度；年龄绝对差 | gallery28／query7均有原表支持；复发／未记录复发13／15对5／2；KRAS野生型／突变15／13对5／2；中／低分化17／11对2／5。query性别均为男性，性别匹配作描述并列机会水平 | 性别、RFS月数及事件／删失、术后方案、原始分期及负荷等已连接病例信息 |
   | Zhuhuman2025_IMC_Lung | 吸烟史、性别；手术年龄绝对差；复发及生命状态作描述性同值比较 | 性别／吸烟／年龄／生命状态gallery40／43、query10／11；复发已知gallery39／43、query10／11，gallery另有1位UNKNOWN。query复发及死亡各1位，保留实际支持 | ROI诊断、来源ROI_Location；对应病灶大小、T／N／M、原始stage及位置；原始MULTIFOCAL |

   CRLM源为`raw/public/Lin2027_IMC_CRLM/zenodo_22300873/Supplementary_Tables_public.xlsx`的Supplementary Table 1，按Patient ID精确连接当前`sample_id=patient_key`；新增年龄、KRAS及分化等从原表按患者读取。Zhuhuman源为`raw/public/Zhuhuman2025_IMC_Lung/metadata/table_s1.xlsx`的Table S1A，仅向下填充Case；按既有SPORA `tissue_id`／`old_tissue_id`桥接，Path Acc按`^[A-Za-z]+(?=[0-9]{4}-)`去字母前缀后，与`original_patient_identifier + "-" + lesion`精确连接。患者字段沿已核对的Case连接；大小、TNM、stage和位置按该accession连接对应病灶sample，不从患者第一行广播。Table S1B的ROI_Diag及ROI_Location按ROI_ID桥接，评分标签沿`labels_ref`，Table S1A的Histology不覆盖ROI诊断；MULTIFOCAL属于Table S1A的Case，保留原码。只规范确定的大小写及首尾空格，UNKNOWN保持未知，未释义编码按原值展示。原表新字段尚未全部进入正式sample表，执行时按上述来源接线。

   复发／生命状态比较描述来源记录是否同值。CRLM的`rfs_event=0`是实际随访内未记录复发，RFS自原发灶及肝转移灶切除完成起算；时间与事件／删失成对展示。Zhuhuman缺乏统一观察时点及生存随访时长，本任务不据其生命状态计算OS。术后方案表示治疗暴露，两来源当前均不生成治疗响应标签。

6. **计算分子组成相似性。** 所有条件从未打乱完整组织的共同`score_box`读取既有P99.5 `[0,1]` float32 `image.tiff`，保留有效实测背景及零值，排除padding和未测量像素。每study固定全体当前任务ROI共有的有效蛋白集合：逐图`role=downstream`、`tiff_page`非空且`q995>0`，并应用既有逐图marker排除。集合及固定顺序由正式任务元数据维护；名义panel交集和编码器词表不代替它。

   当前CRLM共有32项；Zhuhuman共有8项：CD11b、CD14、CD274、CD45RO、CD73、FoxP3、HLA-DR、αSMA。SWD结果限于这些共同有效marker的组成；该集合不反向缩减编码器输入。按固定marker顺序同步读取多通道像素，以运行配置的`SWD_seed`派生study级投影流及study＋`retrieval_patch_id`级像素流；每patch稳定无放回抽取`min(max_pixels,N_valid)`个多marker像素向量，独立标准高斯方向经L2单位化后作为固定投影，计算投影上一维经验Wasserstein-1距离的均值`SWD_1`。两边样本数不同时按各自经验权重积分，不截断一侧排序数组。各方法及打乱条件共享像素抽样与投影，仅需计算原始Top-1及Top-10实际返回配对，并以分块方式复用投影或距离。每query报告Top-1距离和Top-10平均距离，越低越接近。

7. **运行组成、随机及空间对照。** marker均值与随机排序使用同一位置与候选库。编码前空间打乱沿§4.7既定入口处理，query与gallery两侧都生成对应打乱表征，再在同一候选ID上重新排序；原始组织标签、临床记录及完整图像`score_box`真值保持，用同一套指标比较。冻结编码器，不拟合读出。VirTues和CellWeft入口沿用§4.7；Eva／KRONOS2的实际token化与位置编码hook仍待实施；未完成hook的条件明确记录待实现，不将正常编码输出当作打乱结果。

8. **按患者汇总并导出。** 各query先计算指标，同ROI内query patch等权平均、同患者内有效ROI等权平均、同study内有效查询患者等权平均；不可计算值与其支持数分别报告，不填零。组织分类别先选该类query，再使用同一层级汇总，另报固定类别的宏平均；某类无可计算查询时该类及完整类别宏平均不可计算，并列类别覆盖。两个dataset分别报告，TLS查询单列描述。

   Dataset患者等权主指标的区间按查询患者cluster bootstrap计算，保留抽到患者的全部ROI及patch；gallery固定，各模型与基线共享患者重采样索引。每个临床字段使用其有效查询患者作为分母，另报有效配对数、去重后的有效查询／候选患者数及缺失比例；这些支持数不取代患者等权分母。随机排序及空间打乱重复先在每query内平均再做同层级汇总。逐类结果列支持数，单患者类别不估计患者间区间。随机重复、空间打乱重复及患者不确定性分别记录。

   导出NAS结果：①共享位置索引与每方法float32 `[N,d]`特征，含患者／分组、ROI／sample、panel、中心、`score_box`、实际输入框、像素尺度、有效域、组织覆盖、标签与临床原值／规范值／缺失；②各条件原始Top-10及各query全排序AP，含候选ID、余弦、组织相关性、临床已知／同值、SWD与患者／ROI身份，条件临床Top-10仅在生成分类字段补表时导出；③逐query、ROI、患者和dataset结果、类别宏平均、有效支持、缺失覆盖、Top-k患者多样性、随机机会水平及区间。病例展示使用匿名患者标识、共同框缩略图、比例尺及该dataset已连接metadata。

**检索配置。** query／gallery均为224评分框、步长224、末端padding；组织比例阈值默认0.50。其余既有runtime_defaults保持：SWD像素上限4096、投影128、seed=0，随机排序种子0–9，空间打乱种子0／1／2，患者bootstrap 2000次、seed=0、95% percentile区间。测试标签不用于选择参数。任务确定的k=1／5／10及AP定义从指标字段读取。

实施时先用少量真实窗口检查通道、形状、坐标和评分，再按同一配置导出固定成员的特征与实际覆盖。

### 4.7 空间与组成对照

S2–S4采用marker汇总或细胞组成特征作为组成信息对照，沿用对应任务的划分与评价，比较空间表征带来的增益。空间打乱用于比较改变输入空间安排后的结果。

各任务的组成与空间对照按以下定义执行：

1. **Marker均值基线。** 在各方法已确定的窗口及有效组织范围内计算各实测输入marker均值，构成逐窗口向量；局部组织识别在image／ROI内组bag，整体组织表型预测将患者全部image的窗口平铺组bag，分别接同类MIL及任务头。CellWeft窗口按§4.2训练／验证／测试网格，外部模型保留原生流程。每来源按固定marker身份顺序组织特征，跨panel或逐图未测量项附measurement-presence mask，仅用于缺失处理及统计，不作为独立预测特征。标准化仅拟合开发训练的实测项；未测量项在标准化空间填0，真实测量零值作为实测保留。MIL输入为固定marker顺序的均值向量。记录实际通道与窗口覆盖。
2. **现有细胞组成。** 已有固定细胞类型注释的来源可补充类型比例，沿用相同成员和读出，并列注释覆盖。统一组成基线使用上面的marker均值。
3. **空间打乱条件。** 在局部单元构造完成、空间上下文编码前，对每窗完整实测的有效空间单元作按种子固定的随机置换，位置编码留在目标位置；padding、部分实测及无效位置保留原位。CellWeft在protein-aware tokenizer生成8×8 patch tokens后、ViT加入空间位置编码前置换tokens；VirTues在token化前同步置换全部通道原生8×8图像块、再按目标网格编码。
4. **公开接口与训练。** Eva／KRONOS2实施时核对实际加载版本的token化、上下文attention及位置编码顺序，在相同语义的入口接入置换；当前已核对的是窗口级提取接口。局部组织识别与整体组织表型预测的正常及打乱条件分别处理开发训练、开发验证和测试，以相同结构、成员及预算独立拟合读出，再分别评分。置换种子和配置随特征记录，正常与打乱结果配对报告。
5. **相似组织检索。** Marker均值、随机排序及query／gallery双侧空间打乱按§4.6步骤3及7执行。各条件使用同一位置ID及原始评分真值；固定编码器直接重新排序，结果按相同患者层级汇总。

## 5. 实施接入

| 项目 | 已定协议与接入工作 |
| --- | --- |
| 归一化与遮盖顺序 | Figure 2的共同P99.5评分域及转换按§3.4执行；空间遮盖回补按§4.3执行，实施时对接各模型统计资源及输入／输出适配 |
| 空间信号输出 | Figure 2及空间遮盖回补使用同一marker条件化decoder输出8×8 patch内逐像素强度，预训练后冻结encoder，在开发数据拟合decoder；模型与训练流程见[模型规范](../architecture/model.md) |

运行配置记录模型资源、正式成员入口、输入变换与评分量纲、遮盖比例及种子、crop覆盖和合成权重、读出训练预算及统计重复设置。空间遮盖回补的像素评分、空间插值和ROI等权规则按§4.3执行；Figure 2另记录插值方式及边界、Ridge输入marker及拟合预算／正则强度。局部组织识别与整体组织表型预测另记录模型原生切窗、组织覆盖与ownership、image内及患者平铺bag、MIL结构和优化预算、阈值、Cox风险集与Efron并列事件处理、Harrell C-index方向、患者重采样及空间打乱配置；相似组织检索按§4.6记录固定k、共享位置及原生输入框、临床来源与已知子库、实际SWD marker／抽样／投影、随机与打乱重复及患者重采样配置。
