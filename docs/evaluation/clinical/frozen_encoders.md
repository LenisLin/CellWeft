# Figure 4 冻结编码器的 panel 适配与 patch 特征提取

确认日期：2026-11-20。本文维护 VirTues、KRONOS2、Eva 在下游临床队列上的输入与特征提取流程；临床图版、患者分组、终点及解释分析由 [Figure 4 协议](../../paper/figure4_gastric_cancer.md)维护。

## 1. 已确认的整体流程

> 下游队列图像 → 各模型的 panel 与强度适配 → 按各自推荐尺寸切窗 → 冻结编码 → 每窗一个 embedding → 按患者及时间点汇集 → MIL → TRG 定义的 R／NR 响应或随访预测。

VirTues、KRONOS2、Eva 使用各自的原生 marker 输入方式，均作为冻结的蛋白图像编码器。Panel A、Panel B 分别执行特征提取，并沿用 Figure 4 的开发 D／独立验证 V 分工；同一模型在两队列使用相同 checkpoint 和读出定义。

Figure 4 的比较接口固定为组织窗口的 patch embedding。每个方法按下表准备输入，不强制统一窗口像素数或物理视野；保留每个窗口的原图坐标和像素间距，以便解释其实际组织范围。

| 方法 | 图像入口 | 首版窗口 | panel 信息 | 每窗读出 |
| --- | --- | --- | --- | --- |
| VirTues `virtues-sp32` | `input.tiff`，另走官方强度处理 | `128×128`，准备为 `1 µm/pixel` | UniProt 对应的 ESM-2 向量表与 `channel_ids` | 平均 `16×16` 空间 summary token，`512` 维 |
| KRONOS2 | 现有 P99.5 `image.tiff` | 原始分辨率 `256×256` | 按图像通道顺序排列的 `marker_names` | 模型直接返回 CLS，`768` 维 |
| Eva | 现有 P99.5 `image.tiff` | 原始分辨率 `224×224` | 按图像通道顺序排列的 `biomarkers` | `cls=False`，平均非 CLS 空间 token，`768` 维 |

冻结编码阶段使用 `eval()`、关闭参数梯度并在 inference mode 下运行。后续训练 Mean pooling、gated ABMIL、TransMIL 及相应任务头，编码器参数保持冻结。

## 2. 共同读取的文件与强度约定

### 2.1 每个下游 study 的读取内容

路径均相对 NAS 项目根目录 `/mnt/NAS_21T/ProjectData/CellWeft`。

| 文件 | 读取用途 |
| --- | --- |
| `processed/<study>/images.parquet` | 定位图像、panel、ROI 几何与像素间距，并连接患者及时间点 |
| `processed/<study>/panel_mapping.csv` | 来源通道、实际 marker 身份及 panel 对应；按 `utf-8-sig` 读取现有 CSV |
| `processed/<study>/images/<image_id>/input.tiff` | 保留来源解码强度、dtype 和原生尺度的多通道图像，供 VirTues 重新预处理 |
| `processed/<study>/images/<image_id>/image.tiff` | 已按逐图逐通道 P99.5 缩放至 `[0,1]` 的 `float32` 图像，供 KRONOS2、Eva 适配 |
| `processed/<study>/images/<image_id>/channels.parquet` | 逐图实际可用通道、三层通道位置、marker 身份与 `q995` |
| 对应逐图 metadata、有效域及组织范围 | 识别实测像素、补边和有效组织，筛选窗口并保存坐标变换 |

读取 `input.tiff` 使用 `output_channel_index`；读取 `image.tiff` 使用逐图 `channels.parquet` 的 `tiff_page`。`source_channel_index` 定位原始采集，不能直接用来索引筛选后的 TIFF。图像通道选择与传入模型的 marker 列表须同步执行。字段定义见[数据合同](../../architecture/data_contract.md)。

### 2.2 KRONOS2 与 Eva 的已接受适配

用户已确认：KRONOS2 与 Eva 使用现有 `image.tiff` 作为 `[0,1]` 输入。其每个通道的公式为：

```python
normalized = clip(raw / q995, 0, 1).astype("float32")
```

`q995` 已由整张图像的全部实测像素计算，包含实测背景和零值；直接复用既有图像，不在每个 patch 中重新计算分位数，也不再次除以 `255` 或 `65535`。公式及无效通道处理由[共享强度规范](../../data/preparation.md#共享强度)维护。

P99.5 是本项目已接受的上游适配；其后分别保留 KRONOS2 的官方 marker 标准化和 Eva 的官方特征读出。官方 CORAL 对浮点输入的要求是已经处于 `[0,1]`，Eva 教程也从已归一化 MIF 图像开始。[CORAL dtype 规则](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/dtypes.py) · [Eva 数据教程](https://github.com/YAndrewL/Eva/blob/main/tutorials/01_data_loading.ipynb)

## 3. VirTues 工作流

### 3.1 Panel 准备：每套 panel 执行一次

> 实际 marker → UniProt ID → 官方 ESM-2 蛋白向量 → 蛋白向量表及 `channel_ids`。

1. 根据测量身份把蛋白通道映射到 UniProt accession，保持与所选图像通道相同的顺序。
2. 优先复用对应的官方蛋白向量；需补充时，按官方脚本使用 `esm2_t30_150M_UR50D`，取第 30 层氨基酸 token 的均值并排除首尾特殊 token，得到每个蛋白 `640` 维向量。
3. 用官方加载函数构建向量表，作为模型的 `prior_bias_embeddings`；`channel_ids[c]` 指向第 `c` 个图像通道在该向量表中的行号。它不是 CellWeft MAP 行号。

无法确定 UniProt 身份的通道须先落实蛋白身份；模型输入的图像通道与向量引用始终一一对应。[官方 ESM-2 生成脚本](https://github.com/bunnelab/virtues/blob/main/virtues/utils/compute_esm_embeddings.py) · [官方资源加载与用法](https://github.com/bunnelab/virtues/blob/main/README.md)

### 3.2 读取与预处理：每张 ROI 执行

> `input.tiff` → 选择已完成蛋白映射的通道 → 准备 `1 µm/pixel` 图像 → 逐通道 P99 截断 → `log1p` → 逐 marker 均值／标准差标准化 → 按配置滤波 → 切窗。

采用官方 Spora IMC 组织任务的处理路线：`quantile_clipping_log1p/uq_0.99_image`、`use_mean_std: True`；均值／标准差表随数据适配准备。整张 ROI 先处理，再切 `128×128` 窗口。Cords IMC 配置启用高斯滤波，其他来源配置存在差异，实际使用的配置及统计量来源随运行设置固定。

Spora 的相关组织任务配置使用 `1.0 µm/pixel`。若原图间距不同，在模型外重采样，并保存回到原 ROI 的坐标变换。[官方 IMC 配置](https://github.com/bunnelab/spora-bench/blob/main/configs/datasets/virtues/cords2024cancer.yaml) · [官方标准化实现](https://github.com/bunnelab/spora-io/blob/main/spora_io/utils/dataset/standardize.py) · [组织特征提取入口](https://github.com/bunnelab/spora-bench/blob/main/spora_bench/tools/compute_tile_tokens.py)

### 3.3 冻结编码与窗口读出

> `[C,128,128]` 图像＋`channel_ids` → `encoder.forward_list` → `[16,16,512]` 空间 summary token → 空间平均 → `[512]`。

以下为已加载模型和已准备张量的接口示意：

```python
with torch.inference_mode():
    output = model.encoder.forward_list([tile], [channel_ids])
    grid = output.patch_summary_tokens[0]  # [16, 16, 512]
    embedding = grid.float().mean(dim=(0, 1))  # [512]
```

每条向量对应 `128×128` 像素、约 `128×128 µm` 的完整输入窗口。内部每个 token 定位到 `8×8` 像素，但包含所在输入窗口的上下文。冻结特征提取直接调用 encoder。

窗口内平均有作者论文无监督 niche 分析的依据；将窗口向量再接入患者 MIL 是本项目的 Figure 4 适配。作者官方 Spora 组织分类实现把每窗的 256 个空间 token 直接拼入 attention bag，两者读出粒度不同。[官方编码器](https://github.com/bunnelab/virtues/blob/main/virtues/modules/multiplex_virtues.py) · [官方组织任务 wrapper](https://github.com/bunnelab/spora-bench/blob/main/models/virtues/virtues_model.py) · [论文 Methods](https://www.nature.com/articles/s41586-026-10884-y)

## 4. KRONOS2 工作流

### 4.1 “官方可识别名称”的具体含义

KRONOS2 接收的是与图像通道同序的字符串列表 `marker_names`。已见蛋白的名称及均值／标准差来自所用 checkpoint 的 `marker_metadata.csv`，模型卡标明预训练词表为 268 项。

官方 `marker_match_key` 将名称转为小写并去掉分隔符后精确匹配，例如 `CD-8`、`CD_8`、`CD8` 对应同一个 key。`Ki-67`／`KI67`、`HLA-DR`／`HLA_DR` 也属于这种写法差异。生物学别名需要我们在输入适配表中明确处理，例如 `CD274→PDL1`、`CD279→PD1`、`CD8a→CD8`；这些对应不会由模型自动完成。[官方模型卡：Marker handling](https://huggingface.co/MahmoodLab/KRONOS2#marker-handling)

针对当前胃癌 panel，按以下具体名称准备。右列字符串已在作者 CORAL 的公开 canonical registry 中核对；加载 KRONOS2 后，再依据该 checkpoint 的词表确认是否已有对应统计量。CORAL 公共注册表与 KRONOS2 的 268 项预训练词表分别承担名称整理与模型归一化职责。

| 我们的 marker 标签 | 准备传入的名称 | 对应方式 |
| --- | --- | --- |
| `CD4`、`CD14`、`CD20`、`CD27`、`CD31`、`CD38`、`CD45`、`CD68`、`CD163` | 保持同名 | 直接使用已确认的测量身份 |
| `CD11c` | `CD11C` | 大小写整理 |
| `CD66b` | `CD66B` | 大小写整理 |
| `CD3e` | `CD3e` | 保留已确认的亚基身份 |
| `FoxP3` | `FOXP3` | 大小写整理 |
| `Vimentin` | `VIMENTIN` | 大小写整理 |
| `E-Cadherin` | `E-CADHERIN` | 大小写整理 |
| `HLA-DR` | `HLA_DR` | 分隔符整理 |
| `Ki-67` | `KI67` | 大小写及分隔符整理 |
| `PanCK` | `panCK` | 大小写整理，保留泛角蛋白测量含义 |
| `CD274`／来源 `CD274_PDL1` | `PDL1` | 明确的生物学别名对应 |
| `CD279`／来源 `CD279_PD1` | `PD1` | 明确的生物学别名对应 |
| `CD8a` | `CD8` | 依据现有 CD8A 测量身份显式对应 |
| `CD197`／来源 `CCR7` | `CCR7` | 明确的生物学别名对应 |
| `IFNγ`／来源 `INF_gamma`、`INFγ` | `IFNG` | 根据已确认来源身份对应干扰素 γ |
| `Granzyme B`／来源 `GZMB` | `GZMB` | 明确的生物学别名对应 |
| `Osteopontin`／来源 `SPP1` | `SPP1` | 明确的生物学别名对应 |
| `αSMA`／来源 `alpha_SMA` | `A-SMA` | 将希腊字母写法显式对应到作者名称 |
| `CA9`／来源 `CAIX` | `CA9` | 保留碳酸酐酶 IX 测量身份 |
| `SOX9`、`FAP`、`FOLR2`、`MUC6` | 保持同名 | 直接使用已确认的测量身份 |

名称依据：[作者 canonical registry](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/markers/data/registry_v1.csv)。来源标签与 marker 身份以各 study 的 `panel_mapping.csv` 为准，适配记录保留原始名称。

当前 panel 另有 `LRP1`、`CD80`、`TRPA1`、`HMMR`、`HSP70`、`TTF1`、`VCAN`、`NFIB`、`GPNMB`、`CD74`、`LIPF`、`MYH11`、`Periostin` 等名称，需要在加载的 KRONOS2 checkpoint 中逐项查表。测量身份明确时保留对应名称；HSP70 等家族级标签按实际来源测量说明处理。

本轮已核对公开名称表和匹配规则；受访问控制的 KRONOS2 `marker_metadata.csv` 尚未逐项读取。因此，上表明确的是名称适配，不将各项预先标记为已在该 checkpoint 中命中。

### 4.2 Panel 准备及未匹配蛋白处理

> 来源 marker → 上述显式名称适配 → 按图像通道排序 → checkpoint 词表匹配 → 对未匹配蛋白完成登记。

1. 为每个实际 panel 生成与所选 TIFF 通道同序的 `marker_names`；不同 panel 分批编码。
2. 用官方 CORAL `Kronos2Extractor.novel_markers(names, nuclear_marker=...)` 判断哪些名称尚未在模型中匹配。
3. 对未匹配蛋白准备 `additional_markers.csv`，字段沿用作者合同：`marker_name`、`compartment`、`family`、`compartment_desc`、`family_desc`、`marker_full_name`、`mean`、`std`、`pretraining`。生物学说明依实际蛋白填写；均值／标准差从与输入一致的 P99.5 `[0,1]` 图像估计，统计域及来源随运行配置固定。
4. 调用 `register_additional_markers(csv_path)`。模型使用这些信息准备归一化统计及 BioLinkBERT 文本先验；整个流程不需要外部 MAP 向量，也不更新编码器参数。
5. 若纳入核染通道，保留实际核染名称，以 `preferred_dapi=nuclear_name` 指定。存在 DNA1、DNA2 时，所选核染及图像通道顺序随 panel 适配固定。

新增 marker 接口依据：[官方 KRONOS2 提取器](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/features/kronos2.py) · [官方新增 marker 字段与统计合同](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/markers/additional.py)

### 4.3 读取、预处理与冻结编码

> P99.5 `image.tiff` → 按 `tiff_page` 选择通道 → 原始分辨率切 `256×256` → `model.preprocess` → 模型前向 → `[B,768]`。

使用官方模型加载入口 `AutoModel.from_pretrained("MahmoodLab/KRONOS2", trust_remote_code=True)`。首版按官方 CORAL 教程采用 `256×256` 网格窗口；`256` 是作者参考尺寸。保留原生像素间距，模型 wrapper 不额外 resize。

```python
# patches: numpy float32 [B, C, 256, 256]，来自现有 [0,1] image.tiff
normalized = model.preprocess(
    patches, marker_names, preferred_dapi=nuclear_name
)
with torch.inference_mode():
    embeddings = model(torch.from_numpy(normalized).to(device), marker_names)
# embeddings: [B, 768]，每窗一条 CLS
```

`model.preprocess` 继续执行官方逐 marker 标准化；每个完整窗口直接输出一条 `768` 维 CLS，无需平均内部 token。原图像素间距为 `r_x,r_y` 时，每条向量对应约 `256r_x×256r_y µm`。[官方模型卡及调用示例](https://huggingface.co/MahmoodLab/KRONOS2) · [官方切块与特征教程](https://github.com/mahmoodlab/CORAL/blob/main/tutorials/2-Step-by-Step-Patch-Feature-Extraction.ipynb)

## 5. Eva 工作流

### 5.1 Panel 准备：每套 panel 执行一次

> 实际 marker → Eva `marker_to_gene` 对应名称 → 按图像通道排序的 `biomarkers` → 模型内部 GenePT 信息。

使用 Eva 原作者名称字典，按其实际 key 对接，例如：

| 我们的 marker 标签 | Eva 输入名称 | 字典对应基因 |
| --- | --- | --- |
| `CD274` | `PDL1` | `CD274` |
| `CD279` | `PD1` | `PDCD1` |
| `Ki-67` | `Ki67` | `MKI67` |
| `αSMA` | `aSMA` | `ACTA2` |
| `CD8a` | `CD8` | `CD8A` |
| `Granzyme B` | `GranzymeB` | `GZMB` |
| `CD3e` | `CD3e` | `CD3E` |
| `IFNγ` | `IFNg` | `IFNG` |
| `E-Cadherin` | `ECad` | `CDH1` |

字典 key 的大小写需与官方一致。逐个核对其余实际通道，在图像和名称两侧同步选择能够完成官方映射的通道；模型内部调用 GenePT 资源，不向 Eva 传入 CellWeft MAP。未映射通道需要先落实官方可用映射，不能仅改为相近蛋白名称。[官方 marker 字典](https://github.com/YAndrewL/Eva/blob/main/utils/constant.py)

2026-11-20 已将两个实际 study 的 `panel_mapping.csv` 与官方字典的 197 个 key 做静态对照。按上表显式别名，其余标签使用原名；下表只统计 `role=downstream` 的蛋白通道，逐图数值可用性还要读取 `channels.parquet`。

| study／技术 panel | panel 蛋白通道 | 名称可映射 | 当前字典无对应 key 的 marker |
| --- | --- | --- | --- |
| IMC_GC／`panel_02` | 35 | 26 | `LRP1`、`CD80`、`TRPA1`、`HMMR`、`HSP70`、`VCAN`、`Osteopontin`、`NFIB`、`GPNMB` |
| IMC_GC／`panel_03` | 35 | 27 | `LRP1`、`CD80`、`TRPA1`、`HMMR`、`HSP70`、`VCAN`、`Osteopontin`、`GPNMB` |
| IMC_GC／`panel_03_backup37` | 35 | 27 | 与 `panel_03` 相同 |
| Lin2026_IMC_GastricCancer／`Lin2026_IMC_GastricCancer__native41` | 35 | 27 | `MYH11`、`CD80`、`TRPA1`、`LIPF`、`FOLR2`、`Osteopontin`、`MUC6`、`GPNMB` |

原生字典流程使用已映射通道子集，名称和图像同步筛选。`MarkerEmbeddingGenePT` 对“字典中有 key、但 GenePT 中无对应项”的 marker 使用 checkpoint 内相应的 learned embedding；对“字典本身无 key”的名称会在 `marker_to_gene[m]` 处报错。这两种情况需要区分。上表的名称匹配数量尚未经过 GenePT 文件和真实权重推理验证。[官方 marker embedding 实现](https://github.com/YAndrewL/Eva/blob/main/Eva/layers.py)

### 5.2 读取与预处理：每张 ROI 执行

> P99.5 `image.tiff` → 按 `tiff_page` 选择通道 → 原始分辨率切 `224×224` → `float32` → `[B,224,224,C]`。

使用现有 `[0,1]` 图像作为已确认的上游适配。官方数据教程读取的是已归一化、已切好的 MIF 窗口；推理 helper 不负责从整张 ROI 切窗。原作者发布配置使用 `224×224` 输入，首版按此尺寸准备，保留原始像素间距。

Eva 论文的上游 MIF 处理描述为按直方图阈值截断并缩放至 `[0,1]`，教程未展开阈值算法；本项目已选择现有 P99.5 适配，后续仍采用作者的编码及读出接口。[官方数据教程](https://github.com/YAndrewL/Eva/blob/main/tutorials/01_data_loading.ipynb) · [作者论文 Methods](https://www.alexandrotrevino.com/assets/pdfs/Liu%20et%20al.%20-%202025%20-%20Eva.pdf)

### 5.3 冻结编码与窗口读出

> `[B,224,224,C]`＋逐样本 `biomarkers` → `extract_features(cls=False, channel_mode="full")` → `[B,768]`。

按原作者基础教程加载 checkpoint 与 GenePT 资源，推理时设置 `mask_ratio=0`，保持编码器冻结。

```python
with torch.inference_mode():
    embeddings = extract_features(
        patch=patches,
        bms=[biomarkers] * len(patches),
        model=model,
        device=device,
        cls=False,
        channel_mode="full",
    )
# embeddings: [B, 768]
```

`channel_mode="full"` 编码调用者传入的全部多通道蛋白图像；按作者教程的 `cls=False`，平均 784 个非 CLS 空间 token，得到每个 `224×224` 窗口的一条 `768` 维向量。原图像素间距为 `r_x,r_y` 时，物理范围约为 `224r_x×224r_y µm`。[官方基础教程](https://github.com/YAndrewL/Eva/blob/main/tutorials/02_basic.ipynb) · [官方特征提取 helper](https://github.com/YAndrewL/Eva/blob/main/Eva/utils.py)

## 6. 特征交付与患者级 MIL

每个方法分别保存 patch 特征矩阵及其索引。每一行对应一个组织窗口，索引保留 `study`、患者键、时间点、panel、ROI／图像键、原图坐标、窗口覆盖范围及像素间距；实际患者身份和成员表保存在 NAS。

同一患者同一时间点的全部合格 images／ROI／FOV 窗口向量直接平铺为一个 `N×d` bag，再执行患者级 MIL。各模型的 `N` 可因原生切窗而不同，`d` 分别为 `512` 或 `768`；输入侧适配层随各自 MIL 训练，使用相同患者划分及可比的训练预算。

| 下游任务 | 特征进入 MIL 的方式 |
| --- | --- |
| 病理响应（TRG 分组） | 治疗前 patch bag → Mean pooling／gated ABMIL／TransMIL → 患者级 R／NR 二分类响应概率 p_R |
| 治疗前随访预测 | 治疗前 patch bag → MIL → 生存头 |
| 术后更新的治疗前／治疗后单独预测 | 两个时间点分别形成 bag，按 Figure 4 的同一术后起点进行比较 |
| 治疗前后联合随访预测 | 两个时间点分别 pooling 到维度一致的患者表示 → 融合患者表示 → 生存头 |

联合方案先分别汇聚两个时间点，避免手术样本 patch 数更多而直接主导混合 bag。D 内完成任务头训练与方案选择，V 使用锁定方案预测；患者重叠与所有时间点的分组沿用 [Figure 4 协议](../../paper/figure4_gastric_cancer.md)。

## 7. 实施顺序

1. 读取下游 study 的图像及 panel 对应，为三个方法分别准备通道顺序和原生蛋白资源。
2. VirTues 准备 ESM-2 表与 ROI 级强度统计；KRONOS2 核对 checkpoint 词表并完成必要的新 marker 登记；Eva 核对名称字典和 GenePT 资源。
3. 按本文分别读取 `input.tiff` 或 P99.5 `image.tiff`，切出各模型窗口，保存回到原图的坐标。
4. 冻结编码器，先用少量实际窗口确认输入通道顺序、返回形状及每行特征的坐标对应，再批量提取下游队列特征。
5. 按患者及时间点建立 bag，连接 Mean pooling、gated ABMIL、TransMIL 和两项临床任务。

运行记录须包含实际权重、panel 适配、强度统计、窗口覆盖及 MIL 配置。Embeddings 和运行结果保存在 NAS。路径与进度见[存储规范](../../architecture/storage.md)及[当前状态](../../project/status.md)。

## 8. 模型加载与原生调用的具体接线

本节中的路径、`device`、通道名称及图像张量由提取运行配置提供；代码片段说明官方调用顺序。它们是待接入项目提取程序的接口，不表示项目已经实现对应 CLI。

### 8.1 一次加载的资源

| 方法 | 初始化前需要的内容 | 官方加载入口 | 资源如何进入模型 |
| --- | --- | --- | --- |
| VirTues | `virtues-sp32/model.safetensors`；ESM-2 `<UniProtID>.pt` 目录；ROI P99 和逐 marker 标准化统计 | `MultiplexVirtues(...)`＋`safetensors.torch.load_file`＋`load_state_dict` | ESM 表先传给构造函数，权重随后加载；统计作用于图像预处理 |
| KRONOS2 | `MahmoodLab/KRONOS2` 的模型配置、代码、权重及 marker metadata；必要时增加完整的 `additional_markers.csv` 和官方文本编码资源 | `AutoModel.from_pretrained(..., trust_remote_code=True)` | marker metadata 随模型加载；新 marker 在批量图像前向之前登记 |
| Eva | 官方 `config.yaml`、`Eva_model.ckpt`、官方名称字典、`marker_embeddings/GenePT_embedding.pkl` | `Eva.utils.load_from_checkpoint` 或 `load_from_hf` | 构造模型时先读取 GenePT，再恢复 checkpoint；配置控制输入窗口和推理 mask |

VirTues 官方 checkpoint 加载见[重建教程](https://github.com/bunnelab/virtues/blob/main/notebooks/1_demo_reconstruction.ipynb)，KRONOS2 见[模型卡示例](https://huggingface.co/MahmoodLab/KRONOS2)，Eva 见[加载 helper](https://github.com/YAndrewL/Eva/blob/main/Eva/utils.py)。这些模型资源独立于各 study 的 CellWeft `marker_embeddings.pt`。

### 8.2 VirTues：先建立蛋白表，再恢复 checkpoint

```python
import torch
from safetensors.torch import load_file
from virtues.modules.multiplex_virtues import MultiplexVirtues
from virtues.utils.utils import load_marker_embeddings, load_marker_embedding_dict

bank = load_marker_embeddings(esm_dir)  # [M, 640]
row_by_uniprot = load_marker_embedding_dict(esm_dir)
model = MultiplexVirtues(prior_bias_embeddings=bank)
model.load_state_dict(load_file(weights_path))
model = model.to(device).eval().requires_grad_(False)

channel_ids = torch.tensor(
    [row_by_uniprot[u] for u in ordered_uniprot_ids],
    dtype=torch.long,
    device=device,
)
```

两个官方加载函数都按文件名排序，因此应从同一目录一次性构建 bank 和行号；`ordered_uniprot_ids` 的顺序来自实际图像通道。ESM bank 在 encoder 中以 `persistent=False` 注册，checkpoint 不替代这份外部蛋白表。[官方资源加载函数](https://github.com/bunnelab/virtues/blob/main/virtues/utils/utils.py) · [官方 encoder 构造及先验注入](https://github.com/bunnelab/virtues/blob/main/virtues/modules/multiplex_virtues.py)

接下来执行 §3 的 ROI 标准化和切窗，将图像与 `channel_ids` 放到同一设备。完整观测输入使用 `multiplex_mask=None`；作者 Spora wrapper 在 CUDA FP16 autocast 下运行 encoder，窗口内平均读出转回 `float32`。首版实现保留该模型自身的精度设置，不将 KRONOS2 的精度设置套到所有模型。

若直接调用 Spora 的 stats-backed standardizer，对应统计资源包含 `image_level_upper_quantiles.parquet`、`global_level_means.parquet`、`global_level_stds.parquet`；采用本项目 TIFF 读取器时，图像端须实现相同的 P99 截断、`log1p`、标准化及所选滤波。上述统计量与处理后的强度域对应，不能直接拿 P99.5 的 `q995` 作为 P99 或标准化统计。[官方 standardizer](https://github.com/bunnelab/spora-io/blob/main/spora_io/utils/dataset/standardize.py)

### 8.3 KRONOS2：新 marker 准备完成后冻结模型

```python
import torch
from transformers import AutoModel
from coral.features.kronos2 import Kronos2Extractor

model = AutoModel.from_pretrained(
    "MahmoodLab/KRONOS2", trust_remote_code=True
)
extractor = Kronos2Extractor(model=model)
novel = extractor.novel_markers(marker_names, nuclear_marker=nuclear_name)
if novel:
    # additional_csv 已覆盖 novel，并具有完整文本信息与 mean/std
    extractor.register_additional_markers(additional_csv)
model = model.float().to(device).eval().requires_grad_(False)
```

为新 marker 补齐统计时，官方 CORAL 的顺序是：

> `novel_markers` → 准备文本 CSV → `configure_stats` → 各 ROI 的 `prepare_slide` → 逐 marker 汇总 `last_slide_stats` → `pool_marker_stats` → 填写 CSV → `register_additional_markers` → 图像提取。

`prepare_slide` 使用组织 mask 内、已缩放至 `[0,1]` 的像素，记录每个 marker 的像素数、均值和样本方差；`pool_marker_stats` 按像素数量汇总，并包含 ROI 间均值差异，而不是直接平均各 ROI 的标准差。对本项目，统计输入就是 P99.5 `image.tiff`。使用的统计图像范围、组织 mask 和冻结方式沿用 Figure 4 的开发／验证规则，在批量提取前指定。[官方 prepare 流程](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/features/prepare.py) · [官方统计汇总实现](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/markers/additional.py)

之后调用 §4.3 的 `model.preprocess` 和 `model(...)`。官方 extractor 采用 FP32，参考 batch size 为 16；首版按此准备，不启用 FP16/BF16 autocast。Batch size 随实际运行记录，显存不足时调整并实际核对特征输出。[官方 KRONOS2 extractor](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/features/kronos2.py)

### 8.4 Eva：先准备 GenePT 路径和推理配置，再加载权重

官方 README 指定从 GenePT Zenodo 资源取得 `GenePT_gene_protein_embedding_model_3_text.pickle`，保存为 `marker_embeddings/GenePT_embedding.pkl`。`Eva/mae.py` 在模型构造时直接按这个相对路径打开文件，因此工作目录应明确设为 Eva 源码根目录；仅传入 checkpoint 的绝对路径仍不足以定位 GenePT。[官方资源说明](https://github.com/YAndrewL/Eva/blob/main/README.md) · [模型构造中的 GenePT 读取](https://github.com/YAndrewL/Eva/blob/main/Eva/mae.py)

```python
import os
from pathlib import Path
from omegaconf import OmegaConf
from Eva.utils import load_from_checkpoint

eva_root = Path(eva_root).resolve()
checkpoint_path = Path(checkpoint_path).resolve()
os.chdir(eva_root)  # 此独立提取进程的工作目录
conf = OmegaConf.load(eva_root / "config.yaml")
conf.ds.mask_ratio = 0.0
model = load_from_checkpoint(str(checkpoint_path), conf, device=device)
model = model.float().eval().requires_grad_(False)
```

保留官方配置的 `ds.patch_size=224`、`ds.token_size=8`、`ds.marker_dim=3072` 和 `pm.dim=768`。`mask_ratio=0` 必须在实例化之前设置，因为模型构造时就创建 masker。编码使用 §5.3 的 standalone `extract_features`，其内部完成 BHWC→BCHW、连续内存和 `float32` 转换。[官方配置](https://github.com/YAndrewL/Eva/blob/main/config.yaml) · [官方特征 helper](https://github.com/YAndrewL/Eva/blob/main/Eva/utils.py)

纯蛋白图像固定 `channel_mode="full"`。官方 helper 的 `"MIF"` 分支假设输入末尾还附有 H&E 三通道，会直接执行 `x[:, :-3]` 并裁去最后三个 marker；将其用于我们的纯蛋白输入会少读三个真实蛋白。`cls=False` 返回 encoder 的非 CLS token 均值，读出为 768 维；配置中的 512 维重建投影不用于这项特征提取。

## 9. 从现有 TIFF 到患者 bag 的数据接线

### 9.1 逐图确定实际通道

> `images.parquet` 取得 `panel_id` 和图像引用 → 连接该 panel 的 `panel_mapping.csv` → 读取逐图 `channels.parquet` → 应用对应模型的名称／蛋白映射 → 同序选择 TIFF 页。

`panel_id` 用于找到来源通道定义；最终批次由实际输入的有序 marker 列表决定。即使属于同一 panel，逐图数值可用通道也可能不同。KRONOS2 的一个 batch 共享 `marker_names`；Eva 的 dense tensor 要求同批图像具有相同通道数。首版按相同有序 marker 列表分批，分别编码后再按 patch 索引还原顺序。

对于 VirTues，选择 `output_channel_index` 并生成对应的 `channel_ids`；对于 KRONOS2／Eva，选择非空 `tiff_page` 并生成各自名称列表。核染作为单独的输入选择记录，遵循 §4 的 `preferred_dapi` 规则；§5.1 的 Eva 数量统计只描述蛋白通道。

### 9.2 每个模型读取和交付什么

| 步骤 | VirTues | KRONOS2 | Eva |
| --- | --- | --- | --- |
| 原图读取 | `input.tiff`，CYX | `image.tiff`，CYX、`float32 [0,1]` | `image.tiff`，CYX、`float32 [0,1]` |
| ROI 级处理 | 按 §3／§8 标准化，并准备 `1 µm/pixel` 图像 | 复用现有强度与原生尺度 | 复用现有强度与原生尺度 |
| 模型前的 batch | `list[Tensor[C,128,128]]`＋对应 `list[LongTensor[C]]` | NumPy `float32 [B,C,256,256]`＋`list[str]` | Torch `float32 [B,224,224,C]`＋`list[list[str]]` |
| 模型内关键调用 | `encoder.forward_list(..., multiplex_mask=None)` | `preprocess` 后调用模型前向 | `extract_features(..., cls=False, channel_mode="full")` |
| 每窗输出 | 空间 summary 平均后的 `512` 维 | `768` 维 CLS | `768` 维非 CLS 均值 |
| 保存对象 | 原生窗口读出的 `float32` 特征＋patch 索引 | 同左 | 同左 |

各自从 ROI 按输入尺寸生成窗口；已有 CellWeft `crop_origins` 不能直接当作三种方法共同的窗口索引。VirTues 官方组织 wrapper 以 128 为步长并在边界贴边，窗口组织覆盖要求严格大于 30%；KRONOS2 官方切块教程以 256 网格为起点；Eva 数据教程读取预切好的 224 窗口。实际提取设置需明确各自的步长、边界处理、组织覆盖及有效域，窗口选择复用已确认的组织范围。[VirTues 窗口实现](https://github.com/bunnelab/spora-bench/blob/main/models/virtues/virtues_model.py) · [KRONOS2 切块教程](https://github.com/mahmoodlab/CORAL/blob/main/tutorials/2-Step-by-Step-Patch-Feature-Extraction.ipynb) · [Eva 预切窗口教程](https://github.com/YAndrewL/Eva/blob/main/tutorials/01_data_loading.ipynb)

### 9.3 按现有字段连接患者及时间点

两个胃癌 study 的样本表均有 `sample_id`、`patient_key`、`timepoint_label`、`treatment_status_at_collection`。先在 study 内按 `sample_id` 连接样本表，取得已确认的患者与取样时点；IMC_GC 图像表还含 `patient_key` 及连接状态，Lin 图像表通过样本表取得患者键。使用现有冻结连接及 `field_mapping.csv` 的时点解释。

每个窗口对应一个唯一特征行，patch 索引保留 `feature_row`、`image_id`、`panel_id`、患者键、时间点、ROI 坐标及物理范围。坐标明确属于 `input.tiff` 的 ROI 坐标系；VirTues 重采样后的窗口通过所保存的坐标变换回到该 ROI。

完成连接的窗口按患者及时间点直接平铺，形成 §6 的 `N×d` bag。图像编码与临床连接分步执行：可先为已纳入图像提取特征，患者／时间点对应明确后再建立临床 bag。治疗前、治疗后分别汇集和 pooling，再形成前后联合患者表示。

### 9.4 实施时的完成标志

| 阶段 | 可观察的交付 |
| --- | --- |
| Panel 对接 | 每个模型、实际 panel 及逐图可用输入都有有序 TIFF 页与原生名称／蛋白引用 |
| 资源加载 | checkpoint、模型配置及蛋白资源成功加载，新 marker 准备完成，编码器冻结 |
| 少量窗口推理 | 实际窗口分别返回 `B×512`／`B×768`／`B×768`，特征行可准确回到输入窗口 |
| 队列特征提取 | 每个方法的特征矩阵、patch 索引及实际 panel／切窗配置对应 |
| 临床接线 | 患者及时间点连接明确，bag 可用于冻结特征上的 MIL 训练 |
