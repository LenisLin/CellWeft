# Figure 4／5：TRG 主线实施工作流

实施与验证进度见[当前状态](../project/status.md)。

科学范围由 [Figure 4](../paper/figure4_gastric_cancer.md) 和 [Figure 5](../paper/figure5_spatial_editing.md) 维护；外部模型的原生接口由[临床编码器工作流](../evaluation/clinical/frozen_encoders.md)维护。本文拥有实施顺序、模块职责、运行输入输出及首版参数建议。

## 1. 当前交付范围与依赖

1. 用 Lin 开发、IMC_GC 独立验证；本轮范围为 4a–d、4f–h 的治疗前 TRG 分析：响应预测、表征变化、重要区域及细胞解释；4e OS 留待后续批次。
2. 从上述区域开展 Figure 5 的删除／减少、CAF→T 替换、指定 marker 数值编辑，保存距离、真实检索和患者级 R 类概率及其逐步变化。
3. Figure 4 比较冻结 CellWeft、VirTues、KRONOS2、Eva，各接 Mean pooling、gated ABMIL、TransMIL。Figure 5 首版建议使用固定的 CellWeft＋gated ABMIL 临床读出。
4. CellWeft 使用训练完成的配套 EMA 编码器，外部方法使用发布权重。

临床主模型采用 R／NR 直接二分类。数值设置及 CellWeft＋gated ABMIL 解释通路为首版建议。运行配置记录所用设置，沿用既定研究分组与临床记录。

## 2. 建立成员、标签与索引

1. 读取两个 study 的 patient_metadata.csv、sample_metadata.csv、images.parquet、panel_mapping.csv、field_mapping.csv，沿现有连接建立患者和时间点。
2. D 为 Lin 中心／队列 12 人；V 为 IMC_GC 中心／队列，排除 11 位重叠患者后有 41 位治疗前患者，其中 40 位有前后配对。实际成员保存在 NAS，图版报告两中心的实际组成。
3. 原始 TRG 取 tumor_regression_grade_reported。R＝0／1，NR＝2／3；Lin 实际为 0 级 2 人、1 级 5 人、3 级 5 人，即 R 7 人、NR 5 人。
4. 同患者全部 panel、时间点、ROI 和 crop 继承同一折。首版按 R／NR 分层的三折患者级划分为实施建议；patient_key 分组是硬性要求。D 为每位患者生成未使用该患者任何数据训练的折外预测，完整 D 重训后向 V 预测。OS 事件数不作为本轮分折依据或阻断条件。
5. 纯化疗／化免保留为分析字段，在 V 内报告响应、表征变化和预测表现。D 内无变异的协变量不拟合组别系数。

本轮 V 的治疗前分析范围为 41 人，治疗前后配对范围为 40 人；纯化疗／化免分别为 10／31 人和 10／30 人。每项统计的分母为符合该项输入、时间点、标签和质量条件的唯一 patient_key 数，不以 ROI、crop、细胞或区域数替代患者数，并报告实际有效 n、中心／治疗组构成和标签覆盖。

交付：members.parquet、splits.parquet、标签和按子分析列明的有效患者人数。保留原始完整队列表，分析成员通过引用建立。

## 3. 四种冻结编码器

所有方法采用 eval、requires_grad_(False) 和 inference_mode。批次按实际有序通道列表组织；编码输出统一保存 float32，每行对应一个组织窗口。

### 3.1 CellWeft

1. 读取 image.tiff、逐图 channels、完整 image 的实例 mask、crop 索引及该 study 的 MAP bank。蛋白图像使用 downstream 蛋白通道；DNA 等 segmentation-only 通道仅参与分割。实例 mask 和 centroid 用于细胞归属与读出，whole-cell 均值用于对应分析与监督。
2. 临床冻结提取按样本用途读取：训练使用 224×224、stride 112；验证／测试使用 224×224、stride 224 无重叠网格，末端 padding、不回移；切窗由[数据准备](../data/preparation.md#7-crop-与训练视图)维护。按[模型规范](../architecture/model.md)使用 protein-aware 图像 token 与 Spatial ViT-B 编码。
3. 内部 patch 为 8×8 像素，28×28＝784 个空间 token，各为 768 维；每 crop 的 768 维 CLS 直接作为 MIL 输入。
4. 另按需保存内部 token、稳定细胞 ID 和实例 mask 读取的 cell embedding；临床 MIL 使用 crop CLS。用于细胞读出的 adapter 与导出主干配套，具体 checkpoint 规则见训练规范。

### 3.2 VirTues

1. 从实际通道对应的 UniProt 构建官方 640 维 ESM-2 表及同序 channel_ids；用 MultiplexVirtues 加载 virtues-sp32。
2. 读取 input.tiff，按官方 P99 截断、log1p、marker 标准化及选定滤波处理，准备 1 µm/pixel 图像。
3. 切 128×128；调用 encoder.forward_list，完整输入使用 multiplex_mask=None。返回 16×16 个 512 维 spatial summary token。
4. 按窗口平均为空间窗口的 512 维向量；保留重采样坐标变换。CUDA 推理沿官方 wrapper 使用 FP16 autocast，平均结果转 float32。

### 3.3 KRONOS2

1. 用 AutoModel.from_pretrained 加载 MahmoodLab/KRONOS2。来源名显式适配，例如 CD274→PDL1、CD279→PD1、CD8a→CD8、αSMA→A-SMA。
2. 对照实际 checkpoint 词表运行 novel_markers；新增蛋白按官方 prepare_slide→pool_marker_stats→register_additional_markers 流程准备文本、mean/std。统计来源和使用范围在提取前记录。
3. 读取现有 P99.5 image.tiff，按 tiff_page 同序取通道；原生分辨率切 256×256，准备 NumPy float32 BCHW。
4. 调用 model.preprocess(patches, marker_names, preferred_dapi=...)，再调用模型前向，得到每窗 768 维 CLS。首版采用官方 FP32。

### 3.4 Eva

1. 用官方 marker_to_gene key 适配通道，例如 PDL1、PD1、Ki67、aSMA、CD8、GranzymeB、IFNg、ECad。
2. 按官方字典同步筛选图像通道与名称。当前四个技术 panel 的静态可映射蛋白数为 26／27／27／27，具体名单由编码器工作流维护。
3. 将 GenePT 文件放到 Eva 根目录下 marker_embeddings/GenePT_embedding.pkl；设置 mask_ratio=0 后加载官方配置与 checkpoint。
4. 读取 P99.5 image.tiff，原生分辨率切 224×224，准备 float32 BHWC。调用 extract_features，设置 cls=False、channel_mode="full"。
5. 官方 helper 平均 784 个非 CLS token，得到每窗 768 维。纯蛋白输入采用 full，保留全部已映射蛋白通道。

四种方法各保留原生窗口和实际物理范围。每个模型交付 features.npy 和 patch_index.parquet；索引至少含 feature_row、study、patient_key、timepoint、panel_id、image_id、ROI 坐标、像素尺度及有效覆盖。

编码器所需跨图统计须记录拟合范围；使用 checkpoint 固定统计或当前图像自身统计时分别登记。若某项统计由训练图像估计，折外提取使用对应训练折统计，V 使用完整 D 阶段锁定的处理方案。

## 4. Bag 与三种 MIL

1. 每位患者、每个时间点单独将全部合格 images／ROI／FOV 的窗口向量平铺为 N×d bag，再执行患者级 MIL；不同患者及治疗前后时间点不合并。首版使用完整 bag。按 image_id、y、x 排序并保存行号，三个 MIL 共用同一方法的 bag。
2. 不同编码器的 d 为 512 或 768。每个模型独立训练 Linear(d,256)＋ReLU 输入适配层，随后进入下表 pooling。
3. Mean：对窗口向量取均值，输出 256 维患者表示。
4. gated ABMIL：a_j＝softmax_j{wᵀ[tanh(Vu_j)⊙sigmoid(Uu_j)]}，z＝Σa_j u_j；门控宽度建议 128，输出患者向量和窗口 attention。
5. TransMIL：输入适配后接 CLS、第一层 Nyström attention、PPEG、第二层 attention 和 LayerNorm；PPEG 保留 3／5／7 卷积，宽度 256、8 heads、128 landmarks。
6. TransMIL 沿官方方式复制前部真实特征补成近方阵，补入位置仅用于模型内部计算；输出 CLS 作为患者向量。PPEG 的序列网格与原图坐标分别保留。

MIL 返回 patient_embedding、任务输出，以及适用时的 attention。Figure 4f–h 和 Figure 5 首版建议固定使用 CellWeft＋gated ABMIL 的折外模型作为解释读出；每位 D 患者的 Figure 5 编辑始终使用未见过该患者的对应折外模型，并保持模型、任务头和 pooling 规则固定。

实现参考：[gated ABMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py)、[TransMIL](https://github.com/szc19990412/TransMIL/blob/main/models/TransMIL.py)。256 维为本项目的小样本适配，保留各方法聚合机制。

## 5. 病理响应二分类读出、训练与评价

主模型直接预测患者级 R／NR，R 为正类；原始 TRG 等级保留用于分组与描述性展示。Figure 5 复用同一响应概率读出。

1. 患者表示接单 logit 二分类头 Linear(256,1)，输出 logit_R；以 sigmoid 得到响应概率 p_R。标签 y_R＝1 对应 TRG 0／1，y_R＝0 对应 TRG 2／3。
2. 训练采用 BCEWithLogits(logit_R,y_R)，每位患者等权；输出 p_R 及预测 R／NR 类别。
3. 主指标为患者级 R／NR AUROC；补充 AUPRC（R 为正类）、Brier、以 p_R≥0.5 得到的 balanced accuracy。
4. 首版训练建议：AdamW，lr=1e-4，weight_decay=1e-4，100 epochs，seed=2026；各组合使用相同训练预算，V 不参与训练或方案选择。
5. D 汇总三折折外预测；再用完整 D 按同配置训练，并生成 V 的患者级预测。预测表记录模型、fold、logit_R、p_R、R／NR 标签及原始 TRG。每个患者仅有一个对应的折外预测；bootstrap 和模型比较均以患者为单位。
6. 以患者为单位 bootstrap 2,000 次估计区间；模型间指标差使用同一组患者重采样。不可计算的重采样记录原因和有效次数。
7. 主要响应模型使用组织表征；临床基线取治疗前年龄、性别、cT、cN 等可用字段，采用带 L2 正则的二分类 logistic regression 和相同划分。训练折内固定编码、缺失值处理和非恒定字段，随后原样应用于折外患者及 V。

## 6. Figure 4 的分析与图件

1. 4a：从成员和运行配置生成队列流程，标注 D／V、治疗前／配对人数、panel 和两种治疗组。
2. 4b：CellWeft 窗口特征逐向量 L2 归一化，患者平衡采样；仅用 D 的真实表征拟合一次 UMAP。建议 n_neighbors=15、min_dist=0.1，邻居数按实际样本数限制；按 R／NR 分面，先对每位患者各时间点的分布归一化，再按患者等权汇总前后密度，并绘制患者代表位置连线。
3. 4c：在原高维特征上计算每位配对患者的前后分布差异。首版使用去偏 Sinkhorn，每时间点最多固定采样 128 个窗口；不足时使用全部窗口，各分布总权重为 1。
4. 距离首版建议使用 GeomLoss `SamplesLoss(loss="sinkhorn", p=2, blur=0.05, debias=True)`。若采用 GeomLoss `p=2`，运输代价为 \(\lVert x-y\rVert^2/2\)。采样位置和参数固定。每位患者贡献一个距离，比较 R／NR 的患者级效应和区间。
5. 4d：展示 R／NR 标签、p_R、原始 TRG 及 D 折外和 V 独立预测指标。治疗组分层结果与有效人数一起汇报。
6. 4f：attention 定位候选，将相邻窗口按固定规则形成区域。定义 s(H)=p_R(H)，C_R=s(H)−s(H去除区域R)；完整重跑 MIL，并与等面积参照区域比较；正贡献表示移除该区域后模型的 R 类概率下降。
7. 4g：先复用可用细胞标签；需要建立标签时，以 CD3/CD4/CD8、PanCK/E-Cadherin、FAP/αSMA/Vimentin、CD68/CD14/CD163 等实测组合定义 T、tumor、CAF、Mac，并核对代表细胞。输出组成、密度、同类细胞 embedding 和实测 marker 状态。
8. 4h：从原始坐标和实例生成 T–tumor 邻近、接触及 CAF/Mac 界面占有；先在患者内汇总，再比较患者。距离阈值以 µm 写入配置；首版邻近半径建议 20 µm。
9. 为 Figure 5 导出 region_id、坐标、窗口成员、贡献方向、原图引用、细胞标签和对应折外模型。4e 的 OS 建模留待后续批次。

表达变化先记录区室、肿瘤含量及取样量的影响。[GeomLoss 官方参数](https://www.kernel-operations.io/geomloss/api/pytorch-api.html)说明 p=2 的具体代价与去偏设置；图注按实际算法命名。

## 7. Figure 5：从区域到可执行编辑

1. 接收 Figure 4 的 NR 相关区域，首版以现有 224 窗口作为候选结构区域；聚类采用患者平衡采样、kNN 图和 Leiden，medoid 按原始表征确定。簇数依据数据，不强制三簇。
2. 用实测蛋白和空间指标归因，优先研究 T 细胞耗竭相关状态、CAF 界面和 CAF–Mac 界面；单个 PD-1 不单独承担耗竭判定。
3. 固定匹配的治疗前 R 参考、R／NR 混合检索库和真实供体库。参考匹配区室、视野及测量条件，先固定目标亚型，参考患者等权、患者内区域及 token 均分权重；排除查询患者，涉及供体时在检索中另排除供体患者。
4. 先用一个典型区域完成 no-op、编辑、固定位置重编码和检索，再执行该类型的全部合格区域。Figure 5 不引入模型或方法间 benchmark。

### 7.1 删除／减少

1. 连接保留 T 细胞与其邻近肿瘤边界，首版候选距离上限建议 100 µm、连接带宽 10 µm。
2. 按实例与连接带的交集选取 CAF/Mac，以界面占有排序；按 0、25%、50%、75%、100% 构造嵌套删除集合，保存实际细胞数及面积。
3. 调用既定 depletion：目标实例蛋白通道输入置零且 pixel_observed=0，spatial_valid 保留，并从编辑实例视图移除对应身份。
4. CAF–Mac 类型分别删除 CAF、Mac 及两者联合；保留匹配的非界面同类细胞和随机操作对照，随机对照首版建议 20 次。
5. 重新编码图像像素或实例读出成员受影响的全部 crop，读取固定 patch 位置和同一批保留 T 的 cell embedding。

### 7.2 CAF→T 替换

1. 每次移出一个 CAF、置入一个真实 T，供体限定相同 panel、预定 T 亚型／状态和可比测量条件。
2. 按尺寸和形状匹配候选，首版依次检查前 20 个供体；对齐质心、按预设尺度变换供体实例及全通道像素，拒绝覆盖保留实例或超出有效域的候选。
3. 原 CAF 支持中未由供体覆盖的部分保持缺失；在 edited image＋instance view 中建立新细胞身份，在同一 image.tiff 强度域生成新像素，并按原定量规则记录需要分析的受影响细胞 whole-cell 测量。
4. 首版替换采用相同几何处理的 CAF→CAF 供体替换作同类对照；选择和物理空间关系按实验规则记录。
5. 原有保留 T 与新置入 T 分别读出；保存供体、变换、替换数、面积和成功／无合格供体的结果。

### 7.3 指定 marker 数值编辑

1. 首版候选包括 PD-1 降低、GZMB 提高及二者组合；IFNγ 可作为追加候选，均以实际 panel 覆盖为依据。
2. 从真实 R 相关同亚型 T 细胞取得指定 marker 模板，按形状对齐；保持未选 marker、实例外像素和其他细胞测量。
3. 在唯一 edited image 的 image.tiff 强度域执行 I_m(α)=(1−α)I_m(0)+αT_m，建议 α=0、0.25、0.5、0.75、1；按原 whole-cell 定量规则重算受影响细胞的指定 marker 均值。
4. 编辑值维持可见，原始 assayed 与来源保持记录；读取目标 T 和其他保留细胞的 cell embedding。

沿用 Figure 4 对应的固定 bag 和原图位置。编辑输入、归一化、受影响窗口与实例定量按[编辑规范](../architecture/editing.md)及[数据合同](../architecture/data_contract.md)处理。

## 8. 每个编辑节点的量化、检索与路径

1. Patch：固定区域有效位置、ownership 和权重，以内部 768 维 token 形成分布 P；读出其固定加权均值 z。删除前后使用相同位置，保留直接操作区之外的额外读出。
2. 5e 比较每个区域的有效内部 token 分布。沿用第 6 节建议的特征处理和求解设置，每区域固定采样位置、至多 128 个 token。方向及编辑幅度按[Figure 5 §7](../paper/figure5_spatial_editing.md#7-距离方向与二维路径)定义。
3. Patch retrieval：按区域向量召回最近 50 个真实候选，以 token 分布距离重排，输出 Top-5 和全部距离。排序后标注 R／NR；主图显示 Top-1 和其在真实区域检索距离中的位置。
4. Cell：邻居删除时读取同一批保留 T 的 cell embedding；编辑自身 marker 时读取目标 T 的 cell embedding。在同类／同亚型真实细胞中检索 Top-5，匹配自身状态并展示邻域。
5. 蛋白解释：比较真实原始、R 参考和检索对象的细胞组成、实测蛋白及界面指标；记录人为修改通道。当前主分析使用这些实测证据。
6. 策略：每类先枚举单操作及剂量，再枚举少量两步组合；按各类自己的细胞数、面积、替换次数或 marker 改变量比较，保留全部方向；输出同操作类别的成本—距离改善前沿，标出同等改善下成本较小的候选。
7. 累积临床读出：按基线 NR 贡献强度固定患者内区域顺序，每步加入一个区域编辑；在累积 edited image＋instance view 上重编码像素或实例读出成员受影响的全部 crop，以新 CLS 替换完整患者 bag 中对应行，重跑同一折外 MIL。
8. 记录每步 p_R、局部距离与检索；曲线横轴为累计区域数／面积，纵轴为 R 类概率，并报告相对基线的概率变化。区域效应先按患者汇总，再做配对区间；概率变化不表述为实际治疗疗效变化。
9. 可视化固定为 5a 流程、5b–c 典型案例、5d 分型归因、5e 三类多路径＋逐节点检索、5f 类型×操作矩阵及患者累积曲线。UMAP 只拟合真实区域一次，编辑节点使用 transform。

## 9. 实现职责与产物

下列为拟新增内部接口职责；正式代码按实施批次展开。

| 位置 | 职责与接口 |
| --- | --- |
| src/cellweft/integrations/clinical_encoders.py | 原生模型 adapter；读取已解析通道和窗口，返回 features[N,d] 及窗口索引 |
| src/cellweft/inference/clinical_features.py | 分批提取与索引写出；按患者、时间点建立 bag 引用 |
| src/cellweft/models/mil.py | Mean、GatedABMIL、TransMIL；输入单个 N×d bag，返回患者表示及适用的 attention |
| src/cellweft/objectives/clinical.py | R／NR 标签、单 logit 二分类读出及 BCEWithLogits 损失；输出 logit_R、p_R 和预测类别 |
| src/cellweft/evaluation/clinical.py | 三折训练／折外预测、完整 D 重训、V 预测及患者级指标 |
| src/cellweft/interventions/clinical_edits.py | 实例删除、供体替换、marker 编辑视图及受影响范围 |
| src/cellweft/evaluation/spatial_editing.py | 固定参考、距离、检索、有限策略枚举及累积患者 bag |
| scripts/evaluation/、scripts/figures/ | 按配置编排运行和作图，复用 src 中行为 |
| configs/evaluation/、configs/edits/ | 非敏感选择规则、方法设置、首版参数和编辑操作配置 |

公共函数的 docstring 说明轴、维度、坐标、输入强度域和返回字段。图像页与 marker 列表不一致、缺少必要资源或患者 bag 为空时明确报错；Eva 的官方未映射子集选择按已记录规则执行。

正式运行产物置于 NAS processed/downstream/gastric_clinical/<run_id>/：members／splits、各模型 features 与 patch_index、bag 引用、任务头、predictions、metrics、regions、编辑 nodes／retrieval／patient_curves 和图件。实际患者、细胞和供体名单均在 NAS；SSD 维护代码、配置、规则及不含私人记录的结果说明。

## 10. 分批完成标准

1. 成员与配置：D 12、V pre 41、V paired 40 的选择规则可重建，Lin R／NR 为 7／5，所有时间点继承患者分组。
2. 编码器接线：每个实际 panel 各跑少量窗口，验证 512／768 输出、有限数值、冻结参数和回到原图的坐标；再批量提取。
3. MIL／TRG：先跑通首版建议的 CellWeft＋gated ABMIL；得到 D 折外和 V 锁定预测后，按同配置完成其余组合。
4. 指标与解释：交付患者级性能、4a–d／f–h、真实区域及生物学归因；按各分析的有效 patient_key 数报告分母，OS 的 4e 留待后续批次。
5. 编辑引擎：依次完成删除、替换、marker 编辑；验证 no-op、实例外输入保留、保留细胞身份对齐与直接操作范围外原始像素不变及正确的全范围重编码。
6. 检索和探索：每个节点可回到编辑输入、距离和真实对象；完成全类型局部分析、患者汇总及 R 类概率变化曲线，生成 5a–f。概率变化表示固定读出的预测，不表述为实际治疗疗效变化。
7. 收尾：代码检查覆盖真实行为，包括二分类标签与响应概率、bag 标签隔离、指标方向、编辑身份、固定读出和空间关系规则；运行结果登记到 docs/experiments，并按实际完成状态更新 status。科学操作变更分别回写其规范。
