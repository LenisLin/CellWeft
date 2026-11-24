# 数据来源与训练准入

本页维护来源角色、训练就绪层级和患者划分。公开数据收集见[collection.md](collection.md)，链接、许可与方法依据见[参考索引](../references.md)。

## 1. 公共来源

| 来源 | 角色与接入要求 |
| --- | --- |
| spora（R12） | 主入口；逐 cohort 检查图像、marker metadata、可用 masks/annotations 和访问要求。 |
| HTAN（R13） | 补充癌种与临床背景，按 multiplex imaging assay 筛选，不把全部组学记录计作图像。 |
| HuBMAP（R14） | 补充组织和 CODEX 等成像数据，逐 dataset 核对文件、尺度及 segmentation。 |
| 原始研究仓库 | 补充独特 panel 或高质量图像；按[收集规范](collection.md)确认 study 覆盖并补齐缺失内容。 |

## 2. 训练就绪层级

| 层级 | 必需输入 | 使用角色 |
| --- | --- | --- |
| Image-ready | 符合准备规范的共享蛋白图像、marker 映射、尺度信息 | 图像输入产物就绪 |
| Cell-ready | 上述输入及通过检查的 whole-cell masks、固定 CellTable/坐标、marker measurements、top-20 邻域 | 实例读出与 whole-cell 表达监督就绪 |
| Joint-ready | Cell-ready 且图像/细胞严格对齐、crop/实例索引和所需检查完成 | 单图像主干联合预训练与实例监督就绪 |
| Annotated | 对应任务所需就绪层级及细胞/组织/患者标签 | 固定模型评价与下游任务 |

训练准入按实际输入检查：使用的每个marker须能关联本study已发布的固定向量。已确认marker在输入通道筛选后按[数据准备](preparation.md)单独查表生成；映射待确认项保留具体缺口。图像、mask、表达和索引按各自产物完成情况发布，向量映射更新时保留仍有效的像素与几何产物。

已配准且强度语义可追溯的图像可接入，不要求原始仪器文件；收集完成不代表已满足上述准备层级。已有 whole-cell mask 按[一次检查流程](preparation.md)优先复用，核 mask 不替代 whole-cell mask。Mask 或定量缺失时保留已完成的图像产物并记录相应输入缺口；mask 与索引完成也不等于蛋白定量完成。

## 3. 私有医院与划分

首轮不自动纳入九家医院；约两家用于独立应用，候选分别为大规模/panel 完整数据及具有治疗前后和临床资料的数据。
至少一整家医院不参与预训练和适配；另一家可作有限适配或探索，具体分工按授权、规模和资料完整度锁定并写入 manifest。
公共benchmark与私有应用的数据角色在预训练前确定；私有case study沿用§4规则。Figure 2/3及比较模型共用以下划分：

1. 统一患者别名及重复发布记录，确定来源datasets、整项外部测试datasets；外部测试与开发来源按患者隔离。
2. 来源dataset原则上按患者分组为80%开发集与20%内部测试集；当前比较任务中与固定VirTues模型重合的来源采用作者公开SPORA患者train/test划分，具体来源见§5及experiments.json的split_source。同一作者链接患者只要含任一train ROI，整位患者进入开发侧；仅含test ROI的患者用于内部测试。缺患者身份时采用可追溯的最高生物学样本层级，最低为图像，并按实际分组层级命名。
3. 同一患者或可追溯样本的全部ROI、时间点、切片及panel保持同组，crop继承所属分组；使用去标识患者键，实际成员清单保存于NAS。
4. 开发图像用于无标签预训练，任务头使用其中可用标签；训练、适配、验证与选参在开发部分完成。内部测试及整项外部测试图像用于评分，始终保留在训练之外；选参验证从开发80%内部划分。
5. 跨样本拟合的归一化、标签驱动采样及模型选择使用开发数据；逐图预处理由[准备规范](preparation.md)维护，蛋白恢复的评分域见[Figure 2](../evaluation/representation.md#34-marker恢复的评价)，全 panel 空间遮盖的强度处理见[Figure 3 S1](../evaluation/representation.md#43-s1全panel空间回补)。
保留原始 treatment/timepoint；不使用药物条件不代表所有观测都属于治疗前。实际私有清单与数据保存在 NAS，不上传患者直接标识或公共 tracker。
准备步骤见[preparation.md](preparation.md)，字段合同见[data_contract.md](../architecture/data_contract.md)。

## 4. 用户指定本地原始数据与 private case study（2026-10-22）

2026-11-24 登记私有膀胱癌 IMC 队列 `Feng2026_IMC_BladderCancer`，数据所有者 Feng，年份 2026，来源别名 `IMC_20240725_63`。本批处理来源索引的 520 个 ROI，复用经核验的来源实例 mask。[队列工作流](../development/imc_20240725_63_preparation.md)

GC—TiRank、Melanoma 和 CRLM 的既有原始接入范围为 MCD 与来源 ROI／clinical／panel 表；这些批次沿用原接入选择。混合来源目录按文件建立符号链接，从冷存储恢复的 MCD 保存到 NAS raw/private。Feng2026 队列按上述已确认实例 mask 复用流程执行。公开发布物及原来源保留。

GC—TiRank 的完整队列与公开发布子集沿用同一 study 身份 `Lin2026_IMC_GastricCancer`，public/private 入口引用同一原始文件；来源别名、裁剪版本按独立采集去重。Melanoma 的本地 MCD 补入原 Xiao2022 study。CRLM 保持既有原始 MCD 入口，本地分割与分析结果不接入。

CRLM 的正式处理目录于 2026-11-18 更名为 `processed/Lin2026_IMC_CRLM/`；来源别名 `zenodo_22300873`、原始发布记录和稳定 ID 保留，当前路径引用及总览已同步。

私有队列为 GC—TiRank、IMC_GC 独立 panel 和 IMC_20240725_91 肝癌。原始入口见 NAS `raw/private/<dataset_id>/`；来源、样本、患者、批次、时间点、ROI 与原始有序通道归并于 `raw/private/<dataset_id>/metadata.json` 的 `private/<dataset_id>/local_intake_v1/` 成员。总表、蛋白内容和说明归并于 `raw/private/_shared/metadata.json` 的原始私有成员键。

三个队列暂保留为 case study，不参与预训练与适配；TiRank 的公开对应来源同时排除。跨 GC 队列的同病理编号联系及共同 split groups 保存于 NAS `raw/private/_shared/metadata.json` 的 `private/case_study_v1/` 成员。正式训练入口接入时必须消费这些排除与分组规则；本次没有启动训练。医院身份及至少一整家医院的 holdout 继续按§3锁定。患者数未知或映射冲突保持明确缺口，不以 MCD 数、项目名称或临床母队列人数替代。

## 5. Figure 2/3 来源范围

逐来源任务、目标蛋白、实际图像成员及固定切分见 `processed/_catalog/public/experiments.json`。公共来源按已固定成员或具体分组缺口登记，4 项整项外部测试为 Bandyopadhyay、Piyadasa、Keren 和 Hoch。胃癌独立处理，HuBMAP 聚合来源待明确原始 study 归属及重复关系。开发部分内部划分任务训练与选参验证。

Moldoveanu、Rigamonti、Monkman、Malchiodi、Salie、Xiao、Cords2024、Danenberg、Fischer、Jackson、Schurch及Zhuhuman均按作者SPORA患者划分；Ali中共有METABRIC患者同步继承测试及开发训练／验证角色。Fischer的28个患者组同时含作者train/test ROI，按上述患者规则整体归入开发。原pending图像继续pending；开发训练／验证患者名单已固定，选参患者约占开发患者的20%。其余来源保留原划分。

Hoch整项50张图像继续位于CellWeft训练之外；与固定VirTues共同比较时，空间遮盖回补使用7位作者留出患者的7张ROI，跨dataset细胞注释使用这7位患者的9张ROI，另2张对照保持描述用途。空间遮盖回补现为11来源759张评分ROI，其中内部133张、外部626张，Schurch为8位患者各1张。Zhuhuman组织分类及检索查询共用11位留出患者的246张ROI，开发侧为43位患者1026张ROI；正常类测试仅1位患者／4张ROI，按实际支持数解释结果。

局部蛋白回补与整通道回补保持19来源的261张评分ROI；跨dataset已见蛋白预测保持3来源643张ROI，全局未见蛋白预测保持2来源619张ROI。Schurch不参加前两项比较。Lin胃癌仍属独立胃癌数据，选定24张ROI补充跨dataset已见蛋白预测，整项不参与CellWeft预训练，专属临床划分沿用原安排。下游分类、MIL、检索及跨dataset细胞注释来源成员随患者划分同步重建，终点资格和标签映射保持。

原 study 资产保持单份，按正式清单执行已弃用或暂停的 ROI 与逐图通道排除。获准保留的小图按原生像素补边，ROI 技术接受与组织估计验收分别记录。成员与有效目标读取规则见[表征评价](../evaluation/representation.md)。

**来源与分组例外。** Ali 和 Danenberg 已确认的共享 METABRIC 患者共同分组，其他已知跨发布患者或重复样本沿用同一原则。Celltune 按物理切片分组。ForoughiPour 按作者 `source_subject_id` 跨 panel 分组，冲突取样不补生物学 sample。Campbell 的 native subject 与临床 specimen subject 冲突记录保留源值，任务采用正式清单中已确认连接。CRLM 用 `source_files`、`source_acquisition_id` 和 `source_image_id` 联合识别采集，单独 ROI 编号可能跨 MCD 重复。Piyadasa 使用实际 immune／tumor panel family，病例连接未确认的图像保留待定。Hoch 对照图使用共同 sample 组，原作者患者键缺失时留空。Keren 的作者同患者关系写入 `source_patient_key`，未确认的 sample 仍为空。

## 6. 细胞注释来源与连接

作者标签存于各 study 的 `annotations/`，读取 `cell_labels.parquet` 时使用已对应的 `image_id` 与 `cell_id`，缺少对应或作者类型的记录保留来源状态。原始附件和连接证据按 study 来源 manifest 定位。

| 来源 | 连接及边界 |
| --- | --- |
| Jackson | 作者标签 ZIP 按 core 精确对应，依据作者说明将顺序 CellId 连到 mask_label 的升序秩。位置附件未完整取得，类型缺失时保留已有 PhenoGraph；严格 TNBC 跨来源主评仅使用无术前治疗的肿瘤 ROI。 |
| Danenberg | 作者 `SingleCells.fst` 的 ImageNumber 经 FullStack 文件名连接图像，ObjectNumber 匹配 mask_label，保留 `meta_description` 的混合及状态类。 |
| Xiao | 作者 `sc_data.csv` 的患者键与 ROI 号定位候选图像，交换 Position 两轴后在 0.01 像素容差内唯一匹配质心。缺图像或坐标支持、无法唯一连接的记录继续保留。Acral、Mucosal、Cutaneous 表示原发亚型。 |
| Keren | 将 `TNBCcellTypes` 的作者栅格叠合当前 whole-cell mask，仅单一来源类型覆盖至少 90% 的实例赋值。保留覆盖比例、Unidentified 与无明确对应状态。 |
| Hoch | 用 RNA-panel SCE 图像号及来源文件名建立图像对应，作者 mask 按 cell_number 与 mask_label 连接，两张重分割对照图按来源质心定位。保留原始粗细标签及实际连接方法，取样部位与原发亚型分别记录。 |

细胞注释的固定比较、各任务标签和标注量曲线见[表征评价](../evaluation/representation.md#35-细胞类型注释任务标签与比较方法)。

## 7. 尚未闭合的来源

- MartinezCasals 的 metadata-only successor 已发布，样本及字段映射表的版本绑定仍待闭合；原始范围覆盖与成年标本连接按来源记录核对。Johnson TIFF 片段保留，待确认 DICOM 对其独有内容的覆盖。Walker 已接收的两个不完整 range 按实际覆盖登记。Parsons 已完成技术接收，病例连接及训练准入仍按现有缺口记录。
- Ranek 按已核验子集与各图实际依赖消费。未闭合的原生来源范围及单图分割依据按来源记录继续保留。
- `IMC_20240725_91` 暂缓纳入训练及搬运。空图像登记已归入 `raw/private/IMC_20240725_91/unprocessed_registration/`，保持原始来源及 manifest。
- Meylan（`S-BIAD1921`）保留撤回前的原生核验证据。首个完整 stitched ROI 的 48 个原生平面已按 cycle/channel 对应官方清单。历史蛋白输入按原生顺序从这 48 个平面中选取 26 个蛋白测量。核分割所用 DAPI、物理尺度及 image 到 sample 的桥接仍缺确认。公开附件可支持 sample 到患者的连接。已核验范围、通道顺序及限制见其来源包的 `withdrawn_native_acceptance_facts.json` 和 `source_marker_order_evidence.json`。
- Allam 的患者及 vendor tissue ID 与文件夹连接仍未知。保留 H3K27me3 测量及映射缺口，所选发布无可配对 whole-cell mask。Sorin 公开图像保留作者 mask／量化信号语义，作者 MAT 为核分割来源，validation 中未获患者 crosswalk 的图像继续留空。
- `biad3612_codex_hnscc` 保留已确认坏 tile 证据及尚未建立的患者连接。RDS 和 `core_metadata` 版本对应的语义证据见 `author_metadata_semantic_probe.json`。Amitay 沿用经过明确配对的作者 mask。
