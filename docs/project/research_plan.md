# CellWeft：空间蛋白基础模型项目规划

版本：Research Plan v0.1

整理日期：2026-10-17

工作名：**CellWeft（细胞织域）**

英文研究题名：**CellWeft: A Cell-Centric, Knowledge-Informed Spatial Proteomics Foundation Model for Multiscale In Silico Perturbation**

本文保留 2026-10-17 已确认的研究设计。[参考索引](../references.md)中的 R01–R18 是文献与资源索引；[原始初始化规格](../archive/Initialization_Prompt_zh.md)已归档。当前工程组织见[架构概览](../architecture/overview.md)，实际进度见[项目状态](status.md)。项目名为工作名，不作为商标或全网唯一性声明。


## 2026-10-18 研究目标澄清

V1 将 Geneformer 式输入编辑扩展到空间细胞层次：删除 marker、单个细胞或细胞集合后，重新编码双视图输入，研究保留 target cell 与微环境的表示响应。优先研究细胞及细胞群删除，关注影响强度、参考状态方向、空间背景依赖及组合效应。

保留细胞的自身测量固定，其 intrinsic embedding 不因邻居删除改变；主要响应读出为 neighbor-contextual、fused-contextual 和固定区域表示。区域整体变化与固定保留细胞集合的变化分别分析。表示 delta 是模型的扰动预测读出，其生物学解释需要独立评价和验证。

全局移除 marker 观测与模拟蛋白功能丧失采用明确区分的操作定义。蛋白级空间响应读出尚未锁定，不因存在 intrinsic marker tokens 就宣称已具备该能力。

## 0. 一页定义

CellWeft 以细胞为可索引、可编辑的基本单位，把两种同源视图共同建模：一条从蛋白知识 embedding 和细胞 marker 测量出发，编码细胞自身及邻近细胞；另一条从完整 multiplex images 出发，编码局部组织。两个基座独立预训练，再在冻结或少量解冻的条件下训练双向递归 fusion。最终输出 marker、intrinsic cell、contextual cell、neighborhood、region/niche 和 ROI/tissue 的表示，支持 marker、cell、niche 的双视图同步删除与重新编码。

当前主模型为 ViT-B/16，不采用 ViT-L。图像训练完整复用锁定版本的 DINOv3 工程；不主动缩减 local views、prototypes 或 heads。细胞基座同时包含 Intrinsic Cell Encoder 和 Neighborhood Encoder。邻域目标是 masked-cell latent prediction，不改成恢复整条表达谱。联合阶段主要训练 Fusion、Predictor 和 Alignment Heads，不重复每步完整 DINO 多视图训练。

科学方向分成三个能力层：

1. **Protein knowledge pretraining**：固定、跨样本共享的 identity/function embeddings。
2. **Spatial FM**：V1 的研究主体；多尺度表征与原生 deletion。
3. **Perturbation extension**：V2；以后用非空间单细胞蛋白扰动数据学习 intrinsic-cell delta。

优化流程只有两个大阶段：

- **预训练**：两基座分别训练 → 联合预训练。
- **下游微调**：医院数据适配、临床任务、后续扰动模块。

Protein knowledge embedding 可以作为独立前置资产生成，不要求它和 Spatial FM 端到端共同训练。

## 1. Motivation

### 1.1 从单细胞状态转向空间上下文

现有单细胞表征通常描述一个细胞内部的分子组合；本项目关注同一个细胞在不同周围细胞、不同组织结构中，如何形成不同的上下文表示。肿瘤免疫微环境提供了直接的问题场景：当目标 T cell 的自身 marker measurements 固定时，其邻近 macrophages、CAF 或其他细胞的存在与空间组织，如何影响该 T cell 及区域层面的模型表示？

我们不是首先选择某一癌种的 R/NR 任务，而是使用多癌种、多中心、不同 panel 的共同结构来构建可迁移的空间模型。临床信息进入后续应用与 case studies，不决定预训练的唯一目标。预训练不使用药物条件，不等于把所有观测都重新标记为“治疗前”；原始 treatment/timepoint metadata 仍保留用于划分和后续分析。

### 1.2 为什么同时需要 image view 和 cell view

Image view 保留局部形态、亚细胞分布、组织边界及胞外信号；cell view 提供稳定的实体索引、分子测量、邻近关系及直接删除的操作对象。只使用图像，cell depletion 需要实例级修改；只使用细胞平均表达，则会舍弃一部分完整图像信息。因此保留两条强分支，而不是将图像降为一个固定胞外背景特征。

### 1.3 基础模型应该原生支持输入编辑

Geneformer 的启发是“对模型本来使用的输入对象做明确操作，再重编码”[R01]。在 CellWeft 中，cell token 是最自然的可操作单位；niche deletion 可以还原成一组 cell IDs 的删除；marker deletion 同时影响细胞分子输入和图像对应通道。模型的 tokenizer、索引和 forward 路径必须为这些操作原生设计。

## 2. 科学问题的迭代与最终取舍

| 讨论阶段 | 当时的问题 | 最后保留的内容 | 不进入当前 V1 的内容 |
|---|---|---|---|
| 泛癌种空间数据 | 如何发挥多个医院、癌种、panel 的优势？ | 以公共多平台数据为主训练统一 FM；两家私有医院做独立应用 | 不把首篇主线收敛为单一临床分类任务 |
| Virtual cell → virtual tissue | 如何把药物与空间组织联系起来？ | 未来药物模块需要共享 molecular/cell interface | 当前不建立任意药物的组织生成器 |
| Drug–protein graph | 药物可能改变功能连接，而非删除蛋白 | 知识进入 protein identity/function pretraining；保留机制和方向 | 不用任意 KG latent delta 直接加到空间 hidden state |
| Geneformer 操作类比 | 是否能编辑 FM 已经认识的输入？ | marker/cell/niche deletion → 同一模型重编码 | 不要求先生成完整的治疗后测量矩阵 |
| 单图像 vs 细胞 token | 怎样同时保留图像和可操作实体？ | 双分支；cell 是基本操作单位，image 保留完整组织 | 不采用仅 cell pooling、仅胞外 patch 或强制 composable-image teacher |
| 多尺度交互 | 小邻域与大区域如何连接？ | TERRA-inspired neighbors；ST-Align-inspired 双向交互和单一欧氏对齐 | 不同时叠加 HyperST 与多个 hierarchy losses |
| 自监督目标 | 细胞侧如何学习空间上下文？ | marker measurement prediction + masked-cell contextual latent prediction | 不将邻域任务统一退回 BERT/MAE 式原始表达恢复 |
| 计算预算 | 能否在持续可用的八卡节点上组织研究？ | Image L→B；联合阶段冻结/少量解冻基座；复用 checkpoints 做消融 | 不默认缩小 DINO heads/views，也不取消 fusion |

VirTues 与 KRONOS 是空间蛋白图像基座和跨 panel 参考[R02,R03]；TERRA 与 CellWorld 提供邻域组织和 latent prediction 参考[R05,R06]；ST-Align 提供本版选用的跨尺度交互/对齐思想[R07]。HyperST 保留为科学问题演化的相关工作，不进入主损失[R08]。

## 3. V1 要回答的科学问题

### Q1. 蛋白知识是否提高跨 panel 表征迁移？

比较固定 PLM-only 与 PLM+regulatory knowledge embeddings，在相同空间训练预算下测试 marker prediction、未知 marker/panel 泛化、跨 cohort cell annotation。知识表征应提供分子身份和功能先验，不包含特定 slide 的测量。

### Q2. 邻域上下文是否为细胞表征带来独立信息？

输出和评估 intrinsic cell、neighborhood-contextual cell、fused contextual cell 三类表示。比较 cell-only、neighbor-aware、image-only、late-fusion、recursive-fusion，并测试邻域 shuffle、距离 rank 消融。

### Q3. 双向交互是否优于两个基座的后期拼接？

保持相同的 image/cell checkpoints，比对无 fusion、late concatenation、一轮交互、多轮交互以及有/无单一 alignment。只针对这些模块的变化重训 joint stage。

### Q4. 输入删除是否成为稳定、可追踪的多尺度操作？

对固定的 target cell 和 ROI 执行 marker、neighbor-cell、niche deletion，跟踪各层 embedding delta。用 no-op、随机同数量细胞、面积匹配、位置匹配、类型匹配、仅删除一个分支等对照，剖析结果对输入内容和操作范围的依赖。

### Q5. 公共预训练能否迁移到未参加表征训练的医院？

至少一整家医院保留为不参与预训练与适配的外部数据。另一家可承担有限适配或探索性分析，具体角色按授权、规模和临床资料完整度锁定。纵向患者的不同时间点必须处于同一数据划分。

## 4. 数据规划

### 4.1 公共预训练主数据

- **spora**：主入口。2026-10-17 官方页面列出 13,546 图像、5,477 患者、35 cohorts。图像、metadata 及可用时的 masks/annotations 使用统一接口[R12]。
- **HTAN**：补充癌种与临床背景，按 multiplex imaging assay 选择，不将全部 HTAN 组学记录算作空间蛋白图像[R13]。
- **HuBMAP**：补充组织及 CODEX 等成像数据，逐 dataset 核查文件和 segmentation[R14]。
- **原始研究仓库**：补充独特 panel 或质量较高的图像。必须通过 patient/sample IDs、文献、checksum 和图像指纹去重，避免与 spora/HTAN 重复计数。

按可用输入分层，而不是要求所有样本都有所有标签：

| 数据层 | 必要内容 | 可参与训练 |
|---|---|---|
| Image-ready | 多通道图像、marker 对应表、尺度信息 | 图像基座 |
| Cell-ready | Image-ready + cell masks、坐标、marker measurements | 细胞基座 |
| Joint-ready | 两视图可严格对齐，索引/QC 完整 | 联合预训练 |
| Annotated | 以上任一层 + 细胞/组织/患者标签 | 独立 benchmark 或下游微调 |

不需要原始仪器文件才能开展 V1；可使用已配准、多通道强度语义可追溯的图像。DNA/DAPI-only 分割得到的是何种 mask 必须标注，不能将 nuclear segmentation 悄悄当成 whole-cell segmentation。

### 4.2 私有数据

九家医院不全部纳入首轮预训练。选择约两家最有价值的数据用于独立外部应用：一组大规模、panel 完整的数据；一组具有治疗前后及临床资料的数据。两家数据的用途分别写入 manifest；不要在代码里把所有 in-house 数据自动加入公共预训练集。

私有数据在本地可控路径，默认不写入代码仓库、不上传公共 tracker、不写患者直接标识。允许使用 pseudonymous patient keys 进行 split 管理。

### 4.3 规模和计算假设

用户提供的预算：15,000 张 1,000×1,000 图像、平均 40 markers、每图 500–2,000 cells。

由此计算：

- 细胞总量约 750 万–3,000 万。
- 256×256 crop 面积占原图 6.5536%，平均约 33–131 cells/crop；这是面积平均，不是密集区域的上限。
- 15,000 张图像约 22.9 万个面积等价 256 crops；随机重叠采样不产生同等数量的独立患者。
- 6,000 亿强度值，16-bit 未压缩约 1.2 TB，float32 约 2.4 TB。

实际 manifest 必须保存尺寸、marker 数、pixel size 和细胞数，不用上述预算数覆盖真实数据。

## 5. 第一阶段：固定 Protein Knowledge Embeddings

使用已确认形式：

`e_p = Fuse(e_p^PLM, e_p^regulatory)`

embedding 对同一分子身份共享，不依赖当前细胞的表达、空间位置或治疗标签。候选来源为 PLM、MAP/MAP-KG、SIGNOR、OmniPath[R09–R11]。优先评估现成知识 encoder 是否可以独立导出并覆盖所需实体，而非立即重建大型 KG 模型。

工程接口是版本化 `ProteinEmbeddingBank`：

- `protein_id / marker_id → vector`；维数 `R_p` 由资产 manifest 读取，不写死为 1,280。
- 保存来源、模型版本、KG snapshot、ID mapping、embedding checksum。
- 区分 canonical protein 和 measured molecular form；total protein、phosphosite、epitope、clone 信息尽量保留。
- DNA/DAPI、isotype 等非蛋白通道使用明确的 special-marker 描述，不伪造 UniProt ID。
- 未测量 marker、低表达 marker、训练时遮盖、推理时删除分别标识。
- 允许 ESM-only 作为透明的基线/暂时缺失知识对照；不将随机向量说成已训练的知识 embedding。

V1 使用固定导出的表，仅训练各分支的 projection 和 measurement encoders。知识关系 score 作为证据置信度；不当成药理效应幅度。源资源重复证据与许可分别记录。

## 6. 图像基座：DINOv3-B/16 + Protein-aware PatchEmbed

### 6.1 输入与形状

统一使用 PyTorch `B,D,H,W` 张量约定。单个 global crop 为 `D,256,256`；内部 patch size 为 16。因此 global view 有 16×16=256 个空间位置。

在每个空间位置内，对每个 marker 的 16×16 像素进行共享线性投影，加对应固定 protein embedding 的可训练投影，再做位置内 marker fusion：

`[B,D,256,256] → [B,256,D,d_marker_image] → [B,256,768]`

D 不固定为 40。按 batch padding 时使用显式 valid-marker mask；marker fusion 不依赖任意 channel 顺序。输入像素本身就是图像侧 marker 测量，不额外拼一份 cell mean。

每个位置只输出一个 ViT spatial token，不能未经批准变为 `256×D` 的主干长序列。空间 RoPE 沿用 DINOv3。global/local view 的 grid 动态计算，不能把 256 hardcode 到全部调用中。

### 6.2 DINOv3 训练配方

使用官方代码，不新写一个外观相似的“简易 DINO”代替。锁定 commit、环境和基础配置，再记录覆盖项。

明确覆盖：

- `ViT-B/16`；
- global crop 256；
- 多通道 protein-aware PatchEmbed；
- dataset、metadata、domain-appropriate augmentation 和硬件 batch 设置。

保持所选官方默认的 local views、heads、teacher EMA、DINO/iBOT 与其他配置。2026-10-17 基础 YAML 为 8 local views、local size 96、两 heads 各 65,536 prototypes / hidden 2,048 / bottleneck 256。基础 YAML 的 Gram loss 为 false；是否使用额外 Gram recipe 由明确实验配置控制，而不是把“官方完整训练工程”理解为必须训练原论文 7B teacher 或全部后训练阶段。

不要默认套 ImageNet RGB mean/std、color jitter 和灰度增强。几何增强同步更新细胞/segmentation metadata；强度增强按可追溯的 marker 统计定义。iBOT 的空间 masking 在每位置 marker fusion 后、跨空间主干前执行。

## 7. 细胞基座：Intrinsic + Neighborhood，二者都预训练

### 7.1 Intrinsic Cell Encoder

输入为某细胞的各 marker identity embeddings 与表达值。编码后产生：

- per-marker hidden tokens；
- intrinsic cell embedding `c_i`。

这个阶段不读取邻居，因此相同保留细胞在 neighbor deletion 前后的 intrinsic 表示应相同。该性质通过测试检查。

训练使用 masked marker-value prediction。保持 marker identity，遮盖其 value，用预测头恢复实际测量值。辅助 decoder 应至少有一条路径经过 pooled intrinsic cell embedding，使聚合向量获得训练。未测量 marker 不参加监督。

### 7.2 Neighborhood Encoder

使用全局稳定 cell IDs、kNN indices 与以 index cell 为参考的距离 rank；index cell 为 rank 0。近邻数量配置化，初始 `k=32` 仅为候选，不是论文继承的锁定值。

输入不是邻居的所有 marker 长序列，而是独立编码后的 cell tokens。保留以下输出：

- index cell 的 contextual embedding；
- neighborhood summary；
- 可需要时读取邻域内各 token states。

同一 cell 可出现在多个 neighborhoods；输入 ID 不作为可学习的实体词表，避免记忆患者/细胞编号。不同中心下的 cell-rank 是相对 index cell 的关系属性，不将任意 row order 当成生物顺序。

### 7.3 Masked-cell latent prediction

Student 只处理可见细胞；EMA target encoder 处理完整 neighborhood，并生成 masked targets 的 contextual embeddings。Predictor 使用目标位置/rank 查询，从 student context 预测相应 latent。

可保留有限 marker hints，但只能进入 predictor；不能进入 student context encoder，不能在 cell token 删除后把整条真实 marker profile 绕路交给 predictor。hint ratio 配置化，明确实现和消融；不直接照抄 RNA 场景比例。

Loss：`L_cell = L_marker + lambda_context * L_cell_latent`。

不加入必须恢复整个被遮盖细胞表达谱的 neighborhood decoder；不强行统一为传统 MAE。

## 8. 联合预训练：固定两个基座，主要学习递归 Fusion

### 8.1 冻结表

| 模块 | 初始联合阶段 |
|---|---|
| Protein embeddings | 固定 |
| Protein-aware Image PatchEmbed | 固定 |
| Image ViT-B | 固定，必要时仅解冻末端 |
| Intrinsic Cell Encoder | 固定，必要时仅解冻末端 |
| Neighborhood Encoder | 固定，必要时仅解冻末端 |
| Fusion attention / projections | 训练 |
| Masked-cell predictor / alignment heads | 训练 |

独立细胞预训练已经使用 neighbors。冻结 Neighborhood Encoder 不表示删除它，而是每次对编辑后的邻域重新 forward。

### 8.2 双向交互

冻结基座输出 cell/neighborhood tokens 和 dense image tokens。可训练模块进行多轮：

`Cell → Image update → Image → Cell update → Region readout`

每轮使用更新后的状态，不是重复拼接相同向量。原生维数可以不同，在交互处投影。轮数和是否共享权重写入配置；当前未额外批准通过少轮数/共享权重缩减模型，不默默采取这种削减。

Tissue 暂时是 region/ROI pooling 或 MIL，不要求独立的全切片大 Transformer。cell-centered neighborhoods 可重叠，ROI readout 使用明确的 region coverage/ownership，避免重复计权。

### 8.3 梯度目标

本版只保留一个 ST-Align-inspired Neighborhood–Region alignment，使用欧氏投影、归一化和 multi-positive contrastive objective。相同 region 的多个 neighborhoods 都可为正样本；cross-rank negatives 按明确 patient/crop/overlap 规则组织。

alignment 在融合前时，完全冻结基座后只更新 projection heads，不宣称它单独训练了 Fusion。Fusion 的主要梯度来自融合后的 masked-cell latent prediction：

`L_joint = L_joint_cell_latent + lambda_align * L_alignment`

联合阶段不重复每步执行完整 DINO 多视图训练；使用空间一致的 crop 及其 masked/full views。已经冻结基座的 marker/image 原生 pretraining losses 不必继续每步计算。

### 8.4 Full-view teacher 与 masked-view student

Student：先对 image/cell 双视图同步遮盖目标细胞，再运行基座和 fusion；Predictor 在目标位置输出 latent。

Teacher：读取完整的同一区域，运行基座和 EMA fusion，抽取目标 cell 的 latent，stop-gradient。冻结的图像/细胞基座可以共享权重对象，但完整输入和编辑输入仍需分别 forward；不重复维护无用的大型冻结副本。

EMA 覆盖当前可训练的 fusion，以及后来被解冻的基座部分。使用预训练表示的 residual 路径初始化 fusion，监测 latent variance、norm、teacher/student dispersion 和梯度，不自动加入未经决定的新损失。

### 8.5 no_grad 边界

冻结基座只作为前端时可在 `no_grad()` 下前向，输出作为普通输入进入可训练 Fusion。若任何可训练表示之后又穿过冻结 block，则该 block 需要保留对输入的梯度，不可整段 `no_grad()`。初始方案把递归放在 Fusion 内，减少这种复杂度。

冻结模块 `eval()` 与参数冻结分别处理；不能用一次 `model.train()` 让冻结基座的 dropout 等状态意外变化。每次保存阶段状态表和实际 trainable parameter report。

## 9. 索引系统与双视图数据合同

### 9.1 主实体

- `cohort_id, hospital_id, patient_key, slide_id, roi_id, crop_id, cell_id, marker_id`。
- dataset row index 与 cell_id 分开，禁止用 tensor 行号代替稳定 ID。
- 图像用 `D,H,W`；坐标使用 `(x,y)`，pixel 与 micrometer 坐标都保存；每个 crop 记录 affine transform。

### 9.2 必须可查询的关系

- `slide → cells / images / markers / split`；
- `cell → mask label / centroid / area / marker row`；
- `cell → neighbor IDs / distances / ranks`；
- `cell → reverse-neighbor IDs`；
- `crop → visible cells / context halo cells / image token grid`；
- `cell → overlapping crops / image-token footprint`。

不存 `N×N` 全距离矩阵，用空间索引构建 kNN。邻域允许从同一 slide crop 外读取 halo cells，不跨患者/slide；用于 image alignment 的目标需有明确 crop 覆盖。crop 外邻域影响与 crop 内视觉范围在 manifest 中标注，不通过错误 padding 伪造细胞。

### 9.3 图像和细胞的坐标一致性

所有旋转、翻转、裁剪必须有 transform ledger，支持原图→view 和 inverse mapping。多 crop 的对应 targets 按共同覆盖定义；instance mask 变换采用离散 label 安全方式。不得独立随机裁剪两条分支后仍按同一 ID 直接对齐。

## 10. 三种编辑操作与 masking/deletion 的区别

### 10.1 Marker deletion

显式提供作用范围：整张 ROI/slide 或指定细胞集合。全局 marker deletion 同步移除/禁用细胞侧 marker tokens 和图像侧该通道；局部 marker edit 同步作用于所选细胞 measurement mask 与相同实例的对应通道像素。不能只修改 feature bank 中的 identity。

### 10.2 Cell depletion

由一个全局 edit specification 同时执行：

1. 从 cell set 删除目标 IDs，以及所有邻域和 region membership 中的引用。
2. 在全部图像通道上对相同 instance pixels 做一致的背景替换；包含核染通道。
3. 不删除与该细胞相交的整个 16×16 image token，保留邻近细胞和胞外区域。
4. 不重新计算整张 slide 的归一化统计；保留 baseline normalization。
5. 对所有重叠 crops、context halos 应用相同删除集合。
6. 从编辑后的输入重新执行完整 forward，不能复用旧的 contextual/image embeddings。
7. 默认固定 baseline kNN 候选，不补入更远的细胞；rank 是否重新连续编号设置为显式配置，默认保留距离顺序后连续编号，并另存原始 rank。

背景替换值必须在 marker normalization 对应的背景坐标中定义；不能认为所有归一化空间的数值 0 都表示背景。训练 image masking 与删除 pixel edit 通过不同操作类型管理。

### 10.3 Niche depletion

niche selection 先解析成 cell ID 集合，再调用同一 cell-depletion engine。不是只删除 niche summary 而保留其 constituent cells。

### 10.4 训练遮盖与推理删除

训练：masked cells 在 student 中不可见，完整 teacher 和 targets 保留；Predictor 查询并学习恢复 latent。

推理：被删除 cells 不再有 prediction queries，也不调用 predictor 补回它们。只读取保留细胞、regions 和 ROI 的 embeddings。

### 10.5 标准输出

保存 baseline、edited、delta、cosine distance、normalized L2、编辑范围、参数/输入 hashes。按照相同 surviving cell_id 和相同固定 region_id 比较；输出 intrinsic、neighbor、fused-contextual、region、ROI 各层读出。删除后的 region 即使为空也保留稳定 ID，并给出 empty flag。

## 11. 评价与论文实验组织

### 11.1 基座评价

- Marker reconstruction，按 marker 和 cohort 汇报，不只平均。
- Cell annotation：intrinsic vs neighbor vs fused；macro-F1 和混淆矩阵。
- Neighborhood/region retrieval：相同区域跨视图检索。
- Niche clustering/domain annotation，在有标注的独立数据上使用 ARI/NMI 等。
- ROI/tissue 任务：器官、亚型、分级、选定临床任务；任务头与 backbone 适配分开记录。

### 11.2 Deletion benchmark

- No-op edit；
- 指定 cell type 的 neighbor depletion；
- 与删除数、面积、位置匹配的随机 cell removal；
- 仅 cell branch / 仅 image branch 的不一致编辑对照；
- Marker、cell、niche 三尺度；
- 相同 target phenotype、不同 neighbors 的分层比较；
- 多次随机种子与 held-out cohorts；
- 固定全部视图和 normalization，完整重新 forward。

不以“所有 embedding 都应当大幅变化”作为单元测试。工程测试检查信息流与一致性，研究评价比较变化的分布、选择性和跨 cohort 复现。

### 11.3 节省计算的消融树

两基座各预训练一次，复用 checkpoints 对比 late fusion、单轮/多轮、alignment 开关、masking 策略和少量末端解冻。只有改变 protein embedding、image tokenizer 或 cell pretraining 本身的消融，才需要相应重新训练基座。不要对所有因素做全组合从头训练。

## 12. 资源与工程预算

目标为持续可用的一台 8×80GB 或 8×96GB 节点。预训练 Image-B 可使用全节点；Cell pretraining、Fusion 消融与下游任务轮转或分卡运行。联合阶段不需要第二套完整图像优化器。

当前主模型可暂按约 1.1–1.3 亿参数估算；保留默认 DINO/iBOT heads 时，图像预训练额外约 4,600 万 head 参数。最终数量由实例化代码逐模块报告，不把上一轮缩小 heads 后的约 1.25 亿训练参数错误沿用到当前配置。

必须分别统计：inference backbone、active trainable parameters、frozen parameters、EMA copies、optimizer states、activation peak memory。冻结参数不从“模型总参数量”中抹掉。

工程策略：BF16、按 cell 数分桶、marker padding masks、同 forward 内 intrinsic cell 去重、局部 kNN 索引、crop-wise image I/O、activation checkpointing、阶段性 FSDP/DDP 选择、真实 benchmark 后确定 microbatch。累积梯度的 batch 与每个 contrastive/Sinkhorn forward 的 batch 分开计算。

先在 synthetic data 和小型公开 manifest 上运行 50–200 steps profiler。输出 samples/s、cells/s、tokens/s、data wait、GPU peak memory、各模块参数数和训练比例；不预设固定 GPU-days。

## 13. V2 预留但不实现的扩展

Protein embeddings 继续固定共享；把当前 intrinsic-cell encoder 暴露为无 spatial 输入即可调用的接口。后续使用 CITE-seq ADT、CyTOF、4i 的 control/perturbed 数据对齐到同一 molecular space，训练 drug-conditioned intrinsic-cell delta：

`h_cell_edited = h_cell_baseline + Adapter(h_cell_baseline, drug, assay_context)`

随后通过 Neighborhood/Fusion/ROI 路径重新编码。优先在共享的 protein identity 和 cell space 中微调，不另建完全独立的 transcriptome decoder。实际 supervision 按数据的 genetic/chemical 类型区分；CITE-seq 不一概当作小分子药物数据[R16–R18]。

V1 仅保留 `PerturbationAdapter` protocol、latent 注入点与 state-dict version，不引入 drug-conditioned loss、扰动数据下载或额外跨模态大规模预训练。

## 14. 实现里程碑与验收

| 里程碑 | 交付 | 验收重点 |
|---|---|---|
| M0 规格锁定 | architecture、config、sources、decision ledger | 锁定 B、256/16、DINO 默认、冻结表 |
| M1 数据闭环 | manifests、marker registry、cell/kNN/crop indices、synthetic fixtures | 坐标、实体对应、patient split、去重 |
| M2 双基座前向 | protein-aware image adapter、cell/neighborhood encoders | variable panels、dynamic grids、validity masks |
| M3 两基座训练 | DINO upstream adapter、marker/latent trainer | EMA、目标遮盖、loss 与梯度路径 |
| M4 Joint | frozen encoders、recursive fusion、predictor、single alignment | 冻结权重不变、fusion 参数变化、无目标泄漏 |
| M5 编辑引擎 | marker/cell/niche edits、baseline/edited embedding export | 双视图一致、overlap/halo、一致 normalization |
| M6 规模化 | profiler、8-GPU launcher、resume、checkpoints | 吞吐、显存、分布式恢复和 padding |
| M7 研究评估 | 基线、消融、两医院应用 | 完整 split、可复现实验记录 |

## 15. 尚未锁定而应保留为配置的项目

- Protein embedding 的最终 provider、维度与 knowledge fusion 方式。
- Marker/Neighborhood/Fusion 具体宽度、层数、递归轮数和共享权重。
- k、target masking ratio、partial-hint ratio。
- 两家医院的实际身份及独立验证/适配分工。
- 每平台的强度标准化、pixel-size 分层策略、cell boundary 处理。
- 联合阶段是否以及何时少量解冻末端层。
- Image head/prototype/local-view 数量不是上述可任意削减项：当前锁定为所选 DINOv3 官方配方，改变需显式决策。

## 16. 最终主线

**CellWeft = 固定的蛋白知识表示 + DINOv3-B 图像基座 + 具备邻域预训练的细胞基座 + 冻结基座上的递归融合 + 双视图同步输入编辑。**

第一篇工作的重点是多尺度 spatial representation 与可操作的 cell/niche 结构，公共数据支撑基座，独立医院数据支撑应用。未来药物模块使用同一 intrinsic-cell 接口，不迫使当前 V1 为尚未建立的扰动数据链条支付计算成本。
