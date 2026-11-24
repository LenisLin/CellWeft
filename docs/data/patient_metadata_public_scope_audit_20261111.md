# 患者级 Meta 公开范围审计（2026-11-14 更新）

## 范围与判定

本次范围为 NAS `processed/` 中同时有对应 `raw/public/<study>/` 目录的 116 个公开 study。私有队列 `IMC_GC`、`IMC_20240725_91` 和目录型资源不纳入公开范围判定。每个公开 study 检查 sample 表是否存在、行数和 `patient_key` 覆盖；不读取或记录患者级具体取值。

M01 核对公开来源查找、image→sample→patient 连接和覆盖；M02 核对已填字段的语义、来源、转换及缺失。状态按每 study `review_results.csv` 中 `scope=study` 与 `scope=sample` 的最新 `(scope, record_id, check_id)` 结果归并。M01 通过且 M02 通过或不适用的 study 标为“公开范围内闭合”；M02 不适用仅表示没有已填 sample 值可核，不代表临床 Meta 完整。

已落盘的来源字段映射、连接键和引用见对应 NAS `processed/<study>/field_mapping.csv`；逐项核对回执见 `processed/<study>/review_results.csv`；实际样本记录见 `processed/<study>/sample_metadata.csv`。本次重新读取了全部公开 sample 表，并对空表、缺表及新增样本映射进行了针对性来源核查；患者和样本具体标识不写入本报告。

## 汇总

- 公开 study：116
- 有 sample 表：115；其中非空 109，空表 6；另有 1 个 study 缺表
- M01 通过且 M02 通过或不适用：52 个；有表但未闭合：63 个；缺 sample 表：1 个
- 公开 sample 表共 12065 行；`patient_key` 非空 10531 行（87.3%）。Gerber 的 16 个键由 Table S1 行号生成，并非来源患者 ID，不视为已验证患者键
- 已闭合的 52 个 study 共 4796 行，全部有 `patient_key`
- 私有数据另有 2 个目录，不作为公开数据缺口计入上述数字
- 11月14日核对的当前 NAS 表：Ohara 的 3 个作者 T-cell 来源字段共 34 个非空单元格，14/14 行逐值对应；Piyadasa 的 70 个来源扩展字段共 45,640 个单元格，652/778 行精确连接并逐值对应。Piyadasa 另有 126 行无 source key、42 个来源行未被 processed 表覆盖；标准 grade 字段按已记录规则转换且 0/652 行不符。两项 M02 仍未决，整体闭合数不变
- 已由独立来源复核确认公开范围内未发布个体患者/样本 Meta 表：Amitay2023_MIBI_Melanoma、BenUri2025_MultiPlatform_MultiDisease、Cai2026_IMC_HER2_BreastCancer、Liu2023_MIBI_LymphNode。Cai 与 Liu 的 image→sample→patient 连接仍未决
- CellTuneDepot 已由独立复核检查官方正式论文附件与当前十个 FOV：发布范围内无患者/样本表或交叉键，底层原始 melanoma 研究身份仍未确认。Aoki 官方 JCO 正文确认线上 Appendix Tables A1–A7 被引用且患者特征列于附件；另一任务身份独立复核后，官方 Europe PMC supplementaryFiles ZIP 仅含 12 个 JPG/GIF 图像、无表格/文档，ASCO 补充页及 PDF 请求返回 403 challenge。`sample_metadata.csv` 仍为 0 行，299 张图像的 `sample_id` 均为空；附件内容未取得，M01 未决，不能判为公开数据不存在。MartinezMorilla 由不同任务身份独立复核目标 raw 目录为空、processed 无 sample 表或 field mapping；Guo 2020 的相似胎儿/成人肠道 IMC 论文没有目标来源键，PMC 列出的 Data_Sheet_1.docx 内容仍未核实，因此目标身份未决，不能判为公开数据不存在。Hu2023 的 Supplementary Table 4 与 TissueArray 产品页均已取得：7 个本地 core 位置与厂商 position 精确匹配，新增 11 个来源字段、77 个原始值和对应 mapping；来源 Age/TNM/Stage 保留为扩展字段，不写入时间语义未明的标准化字段。厂商页未提供 core→case/患者交叉键，`patient_key` 与 `source_subject_id` 仍为空，M01 未决。Bortolomeazzi 的官方 Supplementary Data 1 已归档，含 6 条 CLN 样本记录及 stage/MMR/tissue/site 字段，但与 Zenodo `raw_file_metadata.csv` 的 CRC_1–6 标签无公开交叉键；独立复核确认不能按数字后缀转移字段，M01 未决。两项均不是“公开数据不存在”的结论
- Ohara：PMC 列出的 Source Data 工作簿和 Supplementary Data 1 CSV 已按原名归档至 NAS `raw/public/Ohara2024_IMC_UTUC/metadata/`。Source Data `Fig. 1b` 有 44 个 WCM sample.ID；独立复核全部 18 个工作表后，未找到与本地 author/sample/image/source keys 的精确交叉键。作者代码 ZIP 内的 `samples.csv` 有 14 行，与现有 author sample keys 一致，但与 WCM Source Data IDs 和 Supplementary Data 1 的 case/sample IDs 均无精确交集。Supplementary Data 1 有 14 列、14,612 条 WES 突变/CNV 记录，含 `case.ID` 和 `sample.ID`，与本地键的精确交集均为 0，也不能作为 IMC 样本交叉表。未将这些字段转入 `sample_metadata.csv`；14 个 IMC 样本来自 6 位患者与作者表 7 个 PMID 的差异仍未解，sample/study M02 未决
- Bollhagen：公开论文 Methods 描述一份来自单名 HGSOC 患者的术后组织，并在 3D 培养中比较 DMSO 与化疗条件；现有 `sample_metadata.csv` 已填 1 行/1 个生成的 donor key，公开字段为 sample 描述，不含来源 donor ID、年龄、分期或结局。论文列出的 Supplementary Table 1 仅为抗体信息；官方 Zenodo v1 只发布 `Zenodo.zip`，论文 Data Availability 指向该影像数据及按请求提供的数据。独立 M02 复核确认已填值和缺失语义，M02 通过。两轮不同任务身份的独立 M01 复核确认 Figure 3 的 12 张图连到该单一标本/供者；Figure 4 的 3 张 in-situ 图在论文、Zenodo 包和作者 SCE 中均无生物学标本/患者交叉键，M01 保持未决。该结论限定为公开材料未提供其连接与临床字段，不表示按请求数据不存在。[论文与补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695663/)，[Zenodo 17077712](https://zenodo.org/records/17077712)，[作者代码](https://github.com/BodenmillerGroup/HR_IMC)
- Kucukkose：独立复核作者 ZIP 的 `ROI_info.csv` 后，27/27 个图像键可精确连接到 9 个 archive `Patient_ID`，且每个患者对应 3 个 core；当前 9/9 sample rows 均有唯一非空患者键。PMC 正文及归档 Supplementary Table 1 未说明 `pt ID` 与 archive `Patient_ID`/TMA core 的映射；9 个同号项中仅 3 个组别一致、6 个组别相反。按组或数字重排无法唯一匹配临床行，故年龄、性别、分期和部位仍空；M01 通过、M02 未决
- Malchiodi2024：作者工作簿已由 `metadata.json` canonical member 读取；164/164 个 SPoRA 来源键匹配且 17 个源字段无差异。110/110 入选图像连接到 56 个样本组，56/56 行有患者键；6 条仅见于作者工作簿的记录中，5 条对应已有 Case+Block 组且不增加字段值，1 条 Block 为空而未分配。Stage 的两个来源写法仅有空白差异。独立复核后 M01、M02 均通过

63 个有表但未闭合的 study 中有 6 个空 sample 表；另有 1 个 study 缺表。其余包含患者连接未决、部分 source→sample 映射未决、临床字段校对未决或失败。逐 study 状态见表。公开来源未发布的字段不应通过论文队列汇总值推算到个体记录。

## 逐 Study 结果

| 数据集 | sample 行 | patient_key 非空 | M01 | M02 | 结论 |
| --- | ---: | ---: | --- | --- | --- |
| Ajaib2025_IMC_GBM_Longitudinal | 10 | 10/10 | 通过 | 通过 | 公开范围内闭合 |
| Ali2020_IMC_BreastCancer | 483 | 483/483 | 通过 | 通过 | 公开范围内闭合 |
| Allam2022_IMC_Lung | 12 | 12/12 | 通过 | 通过 | 公开范围内闭合 |
| Allam2024_IMC_NormalSLO | 6 | 6/6 | 未决 | 通过 | 6 个组织限定 donor 键已填；跨组织患者等价未证实 |
| Amitay2023_MIBI_Melanoma | 0 | 0/0 | 通过 | 不适用 | 公开范围内闭合；未公开个体患者/样本 Meta 表，独立来源复核确认 |
| Andrade2025_IMC_pLGG_MAPK | 120 | 120/120 | 通过 | 通过 | 公开范围内闭合 |
| Aoki2024_IMC_RelapsedHodgkin | 0 | 0/0 | 未决 | 不适用 | 独立复核官方 Europe PMC 包仅有 12 个图像文件，JCO 附件仍受 403 challenge；299 张图像均无 sample 外键，不能判为公开数据不存在 |
| Bandyopadhyay2024_CODEX_BoneMarrow | 20 | 20/20 | 通过 | 通过 | 公开范围内闭合 |
| Barone2026_IMC_LungAdenocarcinoma | 10 | 10/10 | 通过 | 通过 | 公开范围内闭合 |
| BenUri2025_MultiPlatform_MultiDisease | 0 | 0/0 | 未决 | 不适用 | 独立复核检查论文、7份补充表、作者仓库和本地来源库存，未发现患者/样本 Meta 表；TissueArray 仅能确认阵列资源，实验图像到 core/患者的连接仍无公开键 |
| Bever2025_IMC_PDAC_CXCR4Trial | 25 | 25/25 | 未决 | 通过 | 25 个作者组已填；图像连接与 3 个 block 覆盖未闭合 |
| Bollhagen2025_IMC_HGSOC | 1 | 1/1 | 未决 | 通过 | 表中已填单一 ex-vivo donor 可公开信息；Supplementary Table 1 为抗体表，无临床字段；独立复核确认 3 张 in-situ 图无公开样本/患者交叉键 |
| Borner2025_CODEX_MultiTissue | 80 | 80/80 | 通过 | 通过 | 公开范围内闭合；80 个 SPoRA 样本映射至 15 个患者，无临床协变量 |
| Bortolomeazzi2022_IMC_NormalColon_SIMPLI | 6 | 0/6 | 未决 | 通过 | 未闭合 |
| Bowen2023_IMC_Colorectal | 36 | 36/36 | 通过 | 未决 | 未闭合 |
| Cai2026_IMC_HER2_BreastCancer | 0 | 0/0 | 未决 | 不适用 | 独立复核确认公开材料仅有 ROI→Group 表、无患者/生物样本记录；空表下 M02 不适用，图像到生物样本连接仍未决 |
| Campbell2026_MIBI_MelanomaS1616 | 45 | 45/45 | 未决 | 未决 | 作者样本表已取得并填充；7 张图像的内部 FOV 标签与 2 个发布样本组的 subject 键冲突 |
| Chakraborty2026_IMC_Melanoma_TAM | 13 | 13/13 | 未决 | 未决 | 13 行/6 位患者的公开患者级字段已填且独立逐值复核通过；17/17 非空 sample 字段有 mapping。22 个来源 `Tissue_ID` 无图像交叉键，不能确认到具体 specimen，故 M01/M02 未闭合 |
| Chan2021_MIBI_SCLC | 37 | 37/37 | 通过 | 通过 | 公开范围内闭合 |
| Cho2025_IMC_PancreaticCancer | 16 | 16/16 | 未决 | 未决 | 官方文章 Supplement 2 工作簿已从 raw metadata bundle 取得：64/64 core 键与作者 specimen 表对应，12/12 患者 ID 经去连字符后匹配现表；现有 16 行年龄、治疗和性别映射均核对一致，58/58 图像 core 与 sample patient 一致。M02 待独立复核；M01 的图像源身份检查仍未决 |
| Colombo2022_IMC_DLBCL | 26 | 26/26 | 未决 | 通过 | 作者患者表按 `Run ID → Patient ID` 精确连接 26/26；5 张图像没有公开键，且缺少 core/specimen crosswalk |
| Cords2023_IMC_Breast | 12 | 12/12 | 通过 | 通过 | 公开范围内闭合 |
| Cords2024_IMC_Lung | 1068 | 1068/1068 | 未决 | 未决 | 公开患者临床表与当前 1,068 行精确连接；91 张图像没有可确认 sample 外键，临床字段语义仍待复核 |
| CuiZhou2022_IMC_PDAC | 27 | 27/27 | 未决 | 通过 | Supplementary Table 1 的 27 个非键字段已填入 27 行/9 位患者；729 个来源单元格逐值一致，独立 M02 通过。Table 1 有 32 个患者键而论文报告 31 位患者；7 张图像的 sample 身份冲突仍未决 |
| Danenberg2022_IMC_BreastCancer | 718 | 718/718 | 通过 | 通过 | 公开范围内闭合 |
| Ehret2025_IMC_ColorectalCancer | 99 | 99/99 | 通过 | 通过 | 公开范围内闭合 |
| Einhaus2023_IMC_OSCC | 48 | 48/48 | 通过 | 通过 | 公开范围内闭合 |
| Eling2025_IMC_MultiCancer | 179 | 179/179 | 通过 | 通过 | 公开范围内闭合 |
| Enninful2026_MultiPlatform_MultiTissue | 5 | 0/5 | 未决 | 未决 | 官方 Zenodo 80 MB CSV 已取得并按 MD5 核对；108,388 行、48 列，无 patient/sample/file/image/source 键，两列匿名数字索引与当前 sample/image 键无精确交集。不能作为 sample bridge；患者 Meta 结论待独立复核 |
| Fischer2023_IMC_Breast | 1229 | 1229/1229 | 未决 | 未决 | 当前样本键与归档 Table S1 精确匹配 1,229/1,229；sample 层 M02 历史结果为 2 pass、2 unresolved，study M02 暂未汇总通过。13 个 ROI 缺作者 core 身份，另 1 行样本无图像 |
| ForoughiPour2026_IMC_TNBC | 149 | 149/149 | 未决 | 未决 | 公开患者表与 ROI→PID 键表可用；一行 sample 对应两条互相冲突的临床记录，未选择冲突值；另有未分配图像 |
| Greenbaum2023_MIBI_MaternalFetalInterface | 66 | 66/66 | 未决 | 未决 | 当前表含66个患者样本；19个患者来源字段和2个block组织学字段均已映射。基于暂存Strict OOXML表的精确键复算覆盖1,288个非缺失来源单元格，目标空缺0、有效值差异0。定向CLI未生成终稿，且其中“66个目标空缺”聚合未被复算支持；最新正式样本M02仍为fail、研究M02为unresolved，未改写审核记录。M01仍有80/228图像缺显式patient桥、6份block未连图 |
| Greenwald2024_CODEX_Glioblastoma | 12 | 9/12 | 未决 | 未决 | 未闭合 |
| Greenwald2026_MIBI_Breast | 356 | 356/356 | 未决 | 通过 | 未闭合 |
| Gupta2026_IMC_Nodisease | 527 | 527/527 | 通过 | 通过 | 公开范围内闭合 |
| Haist2026_CODEX_HeadNeck | 389 | 389/389 | 未决 | 通过 | 未闭合 |
| Haley2024_IMC_Glioblastoma | 8 | 8/8 | 通过 | 通过 | 公开范围内闭合 |
| Hickey2023_CODEX_Intestine | 72 | 72/72 | 未决 | 通过 | 未闭合 |
| Hiltbrunner2023_IMC_NSCLC_Resistance | 14 | 14/14 | 通过 | 通过 | 公开范围内闭合 |
| Hoch2022_IMC_Melanoma | 34 | 33/34 | 未决 | 通过 | 未闭合 |
| Hoefflin2026_CODEX_IDHMutantGlioma | 37 | 37/37 | 未决 | 未决 | 未闭合 |
| Hu2023_IMC_Lung | 7 | 0/7 | 未决 | 通过 | 7/7 图像与 core sample 相连；Supplementary Table 4 和产品页的 core 字段已核对，尚无 patient/case crosswalk；Excel 端点返回 Cloudflare challenge，不据此判定数据不存在 |
| Iglesia2024_CODEX_BreastCancer | 48 | 48/48 | 未决 | 通过 | 未闭合 |
| Jackson2020_IMC_BreastCancer | 502 | 502/502 | 通过 | 通过 | 公开范围内闭合 |
| Johnson2022_MultiPlatform_MetastaticBreastCancer | 3 | 3/3 | 通过 | 通过 | 公开范围内闭合 |
| Karimi2023_IMC_BrainTumours | 388 | 388/388 | 通过 | 通过 | 公开范围内闭合 |
| Keren2018_MIBI_TNBC | 40 | 40/40 | 未决 | 通过 | 未闭合 |
| Khlebus2025_IMC_OvarianAGCT | 23 | 23/23 | 未决 | 通过 | 未闭合 |
| Kim2023_IMC_EsophagealAdenocarcinoma | 1 | 0/1 | 未决 | 通过 | 未闭合 |
| Klughammer2024_MIBI_MetastaticBreastCancer | 14 | 14/14 | 通过 | 通过 | 公开范围内闭合 |
| Krop2022_IMC_HealthyDecidua | 13 | 0/13 | 未决 | 未决 | 官方 Table 1 的13个SampleID均与MCD archive stem对应；sample表现有13行，覆盖当前12个成像样本和1个无processed图像的公开样本。补行晚于最新M02失败记录，待独立复核；来源未给跨样本patient key |
| Kucukkose2024_IMC_ColorectalCancer | 9 | 9/9 | 通过 | 未决 | 图像到 9 个样本/患者键完整。Table S1 `pt ID` 与 ROI `Patient_ID` 数值标签 9/9 重合，但归档冲突表中仅 3/9 个 MMR/疾病分组一致；相同编号不足以确认临床记录与图像患者同源，未转入临床字段，M02 待独立复核 |
| Kuett2021_IMC_BreastCancer | 3 | 3/3 | 通过 | 通过 | 公开范围内闭合 |
| Lin2026_IMC_GastricCancer | 24 | 24/24 | 未决 | 通过 | 10/10 公开记录已填，14行仅有本地私有来源；公开 image→sample 桥未证实 |
| Liu2022_IMC_FollicularLymphoma | 39 | 26/39 | 未决 | 通过 | 26/39 个切片按 Table S2 Number 精确匹配；13 个 B 系列标签与 A 系列数字后缀一一重合，但论文和 Table S2 未定义该患者 crosswalk，独立复核后仍留空 |
| Liu2023_MIBI_LymphNode | 0 | 0/0 | 未决 | 不适用 | 未发现公开个体 sample/patient Meta 表；独立复核确认；M01连接仍未决 |
| Lu2023_IMC_BoneMarrow | 27 | 0/27 | 未决 | 通过 | 未闭合 |
| Ly2025_COMET_AML | 16 | 16/16 | 通过 | 通过 | 公开范围内闭合 |
| Lyman2025_IMC_PanIN | 5 | 4/5 | 未决 | 通过 | 未闭合 |
| Malchiodi2024_IMC_PancreaticCancer | 56 | 56/56 | 通过 | 通过 | 公开范围内闭合 |
| MartinezCasals2026_CODEX_Pancreas | 11 | 11/11 | 未决 | 通过 | 未闭合 |
| MartinezMorilla2025_IMC_FetalAdultTissue | NA | — | 未决 | 不适用 | 独立复核未能确认目标来源身份；候选附件内容仍待核 |
| Matusiak2024_CODEX_MultiDisease | 48 | 48/48 | 通过 | 通过 | 公开范围内闭合 |
| Merkus2026_IMC_cSCC | 21 | 21/21 | 通过 | 通过 | 公开范围内闭合 |
| Meyer2025_IMC_BreastCancer | 450 | 450/450 | 通过 | 通过 | 公开范围内闭合；450 个组织样本映射至 215 个患者，来源字段和 `stage_basis` 映射经独立复核确认 |
| Meylan2026_CODEX_RecurrentGBM | 32 | 32/32 | 未决 | 通过 | 未闭合 |
| Mo2024_CODEX_MultiCancer | 7 | 7/7 | 通过 | 通过 | 公开范围内闭合 |
| Moldoveanu2022_IMC_Melanoma | 90 | 90/90 | 通过 | 通过 | 公开范围内闭合 |
| Monkman2024_CODEX_NSCLC | 42 | 42/42 | 通过 | 通过 | 公开范围内闭合 |
| Nawrocki2023_IMC_MultipleMyeloma | 12 | 12/12 | 通过 | 通过 | 公开范围内闭合 |
| Ohara2024_IMC_UTUC | 14 | 14/14 | 通过 | 未决 | 独立复核确认现表含 3 个作者 T-cell 来源字段，14/14 行逐值对应，非空数分别为 12、11、11；58/58 图像键及既有样本映射核对通过。18 个 Source Data 工作表及 14,612 条补充 CSV 记录均无 author/sample/image/source key 交叉；作者表有 7 个 PMID、论文报告 IMC cohort 为 6 位患者的差异仍未解 |
| Parsons2025_IMC_TNBC | 98 | 0/98 | 未决 | 通过 | 未闭合 |
| Patel2024_MIBI_Neuroblastoma | 11 | 11/11 | 通过 | 通过 | 公开范围内闭合 |
| Phillips2021_CODEX_CutaneousTCellLymphoma | 33 | 33/33 | 通过 | 通过 | 公开范围内闭合 |
| Piyadasa2025_MIBI_Glioma | 778 | 652/778 | 通过 | 未决 | 当前 NAS 表 141 列；70 个来源扩展字段的 45,640 个单元格与官方来源逐值一致，652 行精确连接，126 行无 source key 且扩展列为空，42 个来源行未被 processed 表覆盖。标准 `grade` 按已记录规则映射，652 行零差异。独立 CLI 复核基于发布前快照的候选表，当前表独立复核待补；患者性别桥及未匹配记录使 M02 保持未决 |
| Plattner2023_IMC_CRC | 5 | 5/5 | 通过 | 通过 | 公开范围内闭合 |
| Radtke2024_CellDive_FL | 7 | 5/7 | 未决 | 通过 | sample 表已填；图像覆盖与 nLN donor 连接未闭合 |
| Rahim2023_MIBI_HNSCC_LN | 96 | 96/96 | 通过 | 通过 | 公开范围内闭合 |
| Ranek2026_MIBI_TNBC | 1025 | 406/1025 | 通过 | 未决 | 1,025 个 sample 行均与来源 FOV 清单对应；患者桥覆盖 406 行、619 行无患者键，已填桥为 246 个键配对。清单另有 21 个 FOV 未映射（5 个 assigned、16 个 deferred unresolved alias）；补充表 3/4 是队列汇总，不可分配个体值，来源覆盖和患者桥均未闭合 |
| Ravi2022_IMC_Glioblastoma | 6 | 6/6 | 通过 | 通过 | 公开范围内闭合 |
| Rigamonti2024_IMC_NSCLC | 83 | 83/83 | 通过 | 通过 | 公开范围内闭合 |
| Risom2022_MIBI_BreastCancer | 125 | 125/125 | 通过 | 通过 | 公开范围内闭合 |
| Roelands2023_IMC_CRC | 8 | 8/8 | 通过 | 通过 | 公开范围内闭合 |
| Roider2024_CODEX_Lymphoma | 7 | 7/7 | 通过 | 通过 | 公开范围内闭合 |
| Roseth2026_IMC_MultipleMyeloma | 65 | 65/65 | 通过 | 通过 | 公开范围内闭合 |
| Ruf2023_CODEX_HepatocellularCarcinoma | 17 | 15/17 | 未决 | 未决 | 未闭合 |
| Salie2025_IMC_HCC | 96 | 96/96 | 通过 | 通过 | 公开范围内闭合 |
| Schurch2020_CODEX_ColorectalCancer | 102 | 35/102 | 未决 | 通过 | 未闭合；独立复核后清除3个来源 `pN` 为空的 `tnm_n` 值，其他样本字段未变 |
| Shaban2024_MultiPlatform_cHL | 21 | 18/21 | 未决 | 通过 | 未闭合 |
| Sorin2023_IMC_Lung | 536 | 416/536 | 未决 | 未决 | 未闭合 |
| Stankiewicz2023_CODEX_Thymus | 7 | 7/7 | 通过 | 通过 | 公开范围内闭合 |
| Sun2025_CODEX_HPA_NormalTissue | 121 | 0/121 | 未决 | 通过 | 未闭合 |
| Sussman2024_IMC_PDAC_VascularNiche | 9 | 9/9 | 通过 | 通过 | 公开范围内闭合 |
| Tanoue2024_IMC_EsophagealCancer | 33 | 33/33 | 通过 | 通过 | 公开范围内闭合 |
| Tietscher2023_IMC_BreastCancer | 12 | 12/12 | 通过 | 通过 | 公开范围内闭合 |
| Tu2026_PhenoCycler_Neuroblastoma | 29 | 0/29 | 通过 | 未决 | 此前作者临床表曾与 29/29 sample_id 精确连接；本次独立 CLI 复核发现映射指定的临床表与其他非图像来源路径当前不可读，故不能复核来源值或据此判断来源不存在。29 个 core 与 27 位患者的计数差额不足以确定归并，`source_subject_id` 29/29 唯一、`patient_key` 0/29；M02 未决 |
| VanKrimpen2026_IMC_MelanomaTDLN | 24 | 23/24 | 未决 | 通过 | 未闭合 |
| VanOost2025_IMC_UPSMFS | 7 | 7/7 | 通过 | 通过 | 公开范围内闭合 |
| Walker2026_MIBI_PDAC | 58 | 58/58 | 通过 | 通过 | 公开范围内闭合 |
| Wu2025_IMC_ESCC_Serplulimab | 20 | 20/20 | 未决 | 未决 | 患者级临床记录已补入；sample_id为项目记录键，未证实生物样本及图像对应 |
| Xi2026_IMC_NSCLC_LymphNode | 29 | 29/29 | 通过 | 通过 | 公开范围内闭合 |
| Xiao2022_IMC_Melanoma | 26 | 26/26 | 未决 | 通过 | 未闭合 |
| Xu2024_IMC_HGSOC | 228 | 0/228 | 未决 | 未决 | 未闭合 |
| Yakubovsky2026_PhenoCycler_Liver | 6 | 5/6 | 未决 | 通过 | Fig. 1b 可用临床记录未覆盖全部图像样本；P17E→P17 无明示交叉键 |
| Yamauchi2023_IMC_HepatocellularCarcinoma | 30 | 30/30 | 通过 | 通过 | 公开范围内闭合 |
| Yeo2026_mIF_DLBCL | 73 | 73/73 | 未决 | 通过 | 141 条 S11 记录聚为 73 个 TB# 样本；20/47 张图像已连接；其余 27 张无公开连接键 |
| Yeung2025_PhenoCycler_MultiCancer | 8 | 6/8 | 未决 | 通过 | 未闭合 |
| Zhu2025_IMC_TNBC_Disparity | 10 | 0/10 | 未决 | 通过 | 未闭合 |
| Zhuhuman2025_IMC_Lung | 123 | 123/123 | 通过 | 未决 | 123 行包含 62 个患者组；`source_subject_id` 与 SPoRA 来源标识一一对应，但 62 个 `patient_key` 均与来源键不匹配，映射未解释其转换。`old_tissue_id` 到 sample 的规则无法复现，映射指定的 Table S1 精确路径不可读；来源语义与覆盖仍待核验 |
| biad3612_codex_hnscc | 28 | 0/28 | 未决 | 通过 | 未闭合 |
| celltune_melanoma_oct2522 | 0 | 0/0 | 未决 | 不适用 | 独立复核确认 Depot 发布范围内无患者/样本表；底层原始研究身份待确认 |
| gerber_biad2091_imc_mimic | 16 | 16/16 | 未决 | 未决 | Table S1 患者级内容已转录；项目行号键不是来源患者/样本 ID，无法连接 43 个核心图像 |
| hcc_tls_imc_2024 | 10 | 10/10 | 通过 | 通过 | 公开范围内闭合 |
| hubmap2019human | 37 | 37/37 | 通过 | 通过 | 公开范围内闭合；37 个样本映射到 15 个患者，公开来源未给临床协变量 |
| meijer_biad833_imc_osteosarcoma | 27 | 27/27 | 未决 | 通过 | 未闭合 |
| zenodo_22300873 | 35 | 35/35 | 未决 | 通过 | 未闭合 |

## 公开来源抽查

- Cho2025 附件追加检查：NCBI OA 文件清单对 `PMC12988722` 返回 HTTP 404；PMC 直链响应为 `text/html` 的 “Preparing to download” proof-of-work 页面。外部直链虽未返回 XLSX，但 NAS `raw/public/Cho2025_IMC_PancreaticCancer/metadata.json` 已归档官方 article XML 和 `media-2.xlsx`，因此最终补充表可在本地核验，无需重复下载。样本与患者连接仍按已归档来源和现有映射复核。
- Liu2023：Zenodo v1/v2 只发布 Point/FOV 图像、分割和分析源数据；论文仅报告队列级6人和12个FOV。补充材料及作者代码均未提供 Point/FOV→生物标本→患者交叉表。独立复核读取公开附件和作者代码后确认，不能把 Point 编号写成 sample/patient 行；sample 表维持0行，M01继续表示当前图像到患者连接不可建立。[论文](https://www.nature.com/articles/s41467-023-40068-5)，[Zenodo数据](https://zenodo.org/records/8118725)，[作者代码](https://github.com/angelolab/publications/tree/main/2022-Liu_etal_Pixie)
- Liu2022：出版社正式补充 DOCX 已下载并归档至 NAS `raw/public/Liu2022_IMC_FollicularLymphoma/metadata/13045_2022_1326_MOESM1_ESM.docx`。Table S2 提供 26 个 A/C 编号的患者临床记录；Zenodo v1 仅列 `raw_data.rar`，压缩包有 39 个切片目录及 ROI marker summary，没有另外的患者 crosswalk。当前 sample 表 26/39 行由临床 `Number` 精确匹配；另 13 个 B 编号的数字部分与 A 系列样本一一重合，且正文/方法报告 13 对诊断与 POD24 样本，但未找到作者明确说明该命名构成患者 crosswalk 的材料。B 系列患者键和具体采样时点保持空值；M01 未决；独立复核确认该后缀模式只是强推断，不足以赋患者键或时点，13 个 B 行继续留空。26 行临床值有既有 M02 独立复核通过记录。[论文](https://link.springer.com/article/10.1186/s13045-022-01326-z)，[Zenodo v1](https://zenodo.org/records/7086346)，[官方 Supplementary DOCX](https://static-content.springer.com/esm/art%3A10.1186%2Fs13045-022-01326-z/MediaObjects/13045_2022_1326_MOESM1_ESM.docx)
- Krop2022：官方 Table 1 标题为患者特征及测量详情，按 `SampleID` 与当前 processed 表精确连接 12/12 行；胎龄、孕期、分娩/流产方式及来源 maternal age 已填，来源 age 为 unknown 的 8 行留空，4 行保留报告数值。Table 1 和 MCD 归档均没有跨样本患者 ID；不把 `SampleID` 重写成 `patient_key`。M02 既有独立逐值复核通过，M01 仍因没有来源患者键而未闭合。[PMC 论文与 Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9257341/)
- Enninful2026：两份 Zenodo 正式记录发布图像/组织学文件；论文 Supplementary Information 为 Tables 1–8，Supplementary Data 1 仅为抗体来源与成像周期信息。现有 5 行与作者 `Multiplexed_IF_Metadata.xlsx` 对应的样本来源记录已填，未见患者 ID 或患者临床表。此前不同任务身份的独立 M01 复核检查两个 Zenodo 版本、论文及补充材料、Reporting Summary、作者仓库和 GSE308167 全部 10 个 GSM；当时未检查被字段映射引用的 80 MB CSV。该 CSV 本轮已从官方 Zenodo 下载并按 MD5 核对，包含 108,388 行、48 列，无 patient/sample/file/image/source 键，两列未命名索引均为逐行唯一数字，且与当前 sample/image 键无精确交集。它不提供 sample→patient crosswalk；最终公开患者 Meta 结论和既有 M02 `pass` 状态待独立复核，M01/M02 保持未决。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13541625/)，[Zenodo 17153112](https://zenodo.org/records/17153112)，[Zenodo 17153187](https://zenodo.org/records/17153187)
- Yakubovsky2026：官方 Figure 1b 临床记录当前可精确对应 5 个 cohort ID；官方 Source Data Fig. 1 工作簿仅有 Fig. 1d 与 Fig. 1e，未提供额外患者临床表，Supplementary Table 5 是抗体表。Zenodo PhenoCycler ZIP 有 6 个样本 TIFF，其中 `P17E`；论文 Methods 列 `P17`，但没有公开说明 E 后缀对应关系，因此 P17E 的患者键及临床字段保持空值。既有不同任务身份独立复核确认发布页、论文方法、QuPath 工程和 ZIP 内无明示 suffix→patient crosswalk；M01 未决，M02 通过。[PMC 论文与补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC13216088/)，[Zenodo 17735558](https://zenodo.org/records/17735558)
- Cords2023 与 Jackson：Cords2023 的 12/12 sample 行和 109/109 图像外键通过 M01/M02；20 个原始 SPoRA 扩展字段共 240 个非空值精确匹配，`source_date` 的事件/时间起点保持未知。Jackson 的 502 行对应 357 个唯一患者键，21 个扩展字段的 5949 个非空值精确匹配，4593 个源缺失单元留空；M01/M02 均通过。
- Cords2024：从 Zenodo v1.0 的 `Patient Stratification.zip` 将 `metacluster` 作为原始来源扩展精确连接到 975/1068 个 sample 行；不解释为临床亚型。样本字段值的独立核对通过，但另一条 study-scope `source_clinical_context` M02 仍未决；M01 也未决，因为 91 张图像没有可确认的生物样本外键（90 个作者 ROI 是 Control，另 1 个缺作者 ROI 记录）。[Zenodo v1.0](https://zenodo.org/records/7961844)，[Zenodo v3](https://zenodo.org/records/15399046)
- Fischer 与 Hoch：Fischer 的 1229 个 sample 行保留 1229 个已确认患者键，新增 `source_menop`（999）、`source_mib1_pr`（803）、`source_tumorsize`（1151）；M02 通过，M01 因 13 个非test ROI 缺作者 core/sample 身份及 1 行缺图像而未决。Hoch 新增 `source_patient_age` 31/34 行，保留来源原字符串且不推断单位/时点；M02 通过，M01 仍未决。
- Bortolomeazzi：官方 Nature Supplementary Data 1 已下载并保存为 NAS `raw/public/Bortolomeazzi2022_IMC_NormalColon_SIMPLI/metadata/41467_2022_28470_MOESM4_ESM.xlsx`。Sheet1 的 CLN1–CLN6 有 stage、MMR by IHC、tissue 和 anatomical site；Zenodo 原始归档表使用 CRC_1–CRC_6，二者没有明确 alias/crosswalk。独立复核检查论文、补充表、Zenodo 描述及归档 `raw_file_metadata.csv` 后确认不能凭相同数字后缀连接；当前 6 个 CRC 行不接收这些 CLN 字段，患者键仍为空。M01 未决，M02 通过仅覆盖当前已填字段和 mapping，不代表 Supplementary Data 1 已合并。[论文](https://www.nature.com/articles/s41467-022-28470-x)，[Zenodo 5545882](https://zenodo.org/records/5545882)，[Supplementary Data 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-28470-x/MediaObjects/41467_2022_28470_MOESM4_ESM.xlsx)
- Hu 与 Moldoveanu：Hu 的 7/7 图像均精确连接 sample 与 SPoRA tissue；作者 Supplementary Table 4 的 7 个 core 位置填入 sex、病理诊断、grade、stage 原值和 type，既有 98 个已填单元及 14/14 非空字段 mapping 经独立 M02 复核通过。新增官方 TissueArray BS04081a 产品页快照后，7/7 core 位置精确匹配厂商 position，并新增 11 个 `source_vendor_*` 字段、77 个来源值及 11 条 mapping；年龄、TNM、grade、stage、sex 等原值保留，标准化年龄/TNM/index-stage 字段仍空。产品页列出 21 cases/63 cores，但不提供本研究 core 到 case/患者的键；Table 4 同行可包含年龄和 TNM 不同的 core（如 B5 与 F7），不能把其合并显示当患者组。`patient_key`/`source_subject_id` 仍空；M01 未决、M02 通过。[Hu 论文](https://www.nature.com/articles/s41467-023-43917-5)，[TissueArray BS04081a](https://www.tissuearray.com/tissue-arrays/Lung/BS04081a)。Moldoveanu 的 90/90 sample、image 与 tissue 精确连接，90 行患者键均采用 `study:patient_id`；`source_sample_id` 已按 SPoRA 定义改为 `original_file_name`，`source_subject_id` 保留 `original_patient_identifier`，30 条 mapping 覆盖 23/23 个非空字段；sample/study 两层 M01/M02 均通过。[SPoRA 格式](https://spora.epfl.ch/docs/io/format.html)
- Hoefflin：从官方 Zenodo v2 获取 Table S1，筛选出 37 条 CODEX 样本记录，连接到 30 个公开患者键；当前 M02 等待独立复查。M01 仍未决：sample→patient 已连接，但本地无 image 表，公开资源中尚未核实原始图像名→SampleID 的键。[Zenodo v2](https://zenodo.org/records/21335411)，[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13589474/)，[BioImage Archive S-BIAD2840](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2840)
- Bever：作者发布的两张 workbook 与 annotated RDS 相互核对支持 25 个 Case×Timepoint 组、14 个患者键；63 个 FCS 文件 ID 与主 workbook 完全一致。79 张 processed 图像分属 26 个 MCD stem；按作者 FCS 记录，63 张有显式 acquisition ID，另 16 张没有。每个 stem 下的作者记录均落到唯一 Case×Timepoint 组，因此 stem 归组只能作为待核候选，不能覆盖当前“不推断未列 ROI 样本身份”的来源规则。论文报告 28 个 FFPE block，作者表只支持 25 组；PMC 正文列出 Supplementary Table 6，但本次未取得附件。M02 通过，图像/specimen 连接与 block 覆盖未闭合，M01 保持未决。[Zenodo 11122889](https://zenodo.org/records/11122889)，[论文 Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360203/)
- Bandyopadhyay：出版社 Table S1 与作者最终 R 映射支持 20 个 specimen、18 个患者键；20/20 图像外键与交叉表一致，21/21 个非空样本字段均有 mapping。唯一后缀冲突行由作者明确的患者与时间点唯一定位，冲突保留在 `metadata_notes`；独立复核确认 20/20 唯一来源行及 380 项字段/值比较，sample/study M01、M02 和 D01 均通过。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11162340/)，[出版社 Table S1](https://ars.els-cdn.com/content/image/1-s2.0-S0092867424004082-mmc1.xlsx)，[作者最终映射](https://github.com/tanlabcode/spatial-bonemarrow-atlas/blob/main/CODEX_Analysis/Step6_AML_ReferenceMapping/RefMap_AML_ForManuscript_Final.R)，[Figshare 发布](https://figshare.com/s/1b3d22ae14f4a448f21c)
- Lin：公开 Zenodo 记录说明 UnionH-3 为 10 位患者并发布 `IMC clinical information.csv`；该 CSV 与出版社 Table S4 均为 10×20 且逐格一致。现有 sample 表的10条公开 `PathNum` 均对应 PatientID；17个原始临床字段170/170值一致，20个公开来源字段已有 mapping。其余14行仅来自本地私有 intake。Table S6 的47条 ROI 可连10个样本键，但 ROI 名称与当前 processed image 键0匹配；相关74张图像的来源路径均指向本地私有 MCD，故公开 image→sample 桥仍未证实。独立复核据此判 M01 未决、M02 通过；M02 仅代表原始公开字段的来源与值保真，不表示临床编码、单位或时间起点已解释。[Zenodo 14864800](https://zenodo.org/records/14864800)，[Genome Medicine 论文](https://link.springer.com/article/10.1186/s13073-026-01604-2)
- 个体患者/样本表缺失的来源结论：Amitay 的 Mendeley v2 发布有 cell/image/channel 标注但无患者或生物样本表；BenUri 的 BioImage 记录是 specimen/file 级，不含可连接到患者的表；Cai 的 Zenodo 发布 157 个 ROI 及 Group 标签，论文报告来源为 47 位患者但没有 ROI→患者交叉键，其他实验数据为按请求提供；Liu 的 Zenodo v1/v2、论文、补充材料和作者代码均没有 Point/FOV→生物样本→患者交叉表。四项均经独立来源复核确认当前公开发布未提供个体患者/样本 Meta 表；Cai、Liu 的 M01 仍表示当前图像关联无法建立。[Amitay Mendeley](https://data.mendeley.com/datasets/zcz8743fcv/2)，[BenUri 论文](https://www.nature.com/articles/s41587-025-02585-0)，[Cai 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13589474/)，[Liu Zenodo](https://zenodo.org/records/8118725)，[Liu 论文](https://www.nature.com/articles/s41467-023-40068-5)
- CellTune：独立复核检查官方 CellTuneDepot-v0.2 页面、本地 10 个 FOV 与官方 2026 论文的四份正式附件。CaltechAUTHORS 正式记录与本地归档附件的 MD5 四项全部匹配；附件内容为补充信息、reporting summary 和 source data 工作簿，未发现患者/样本键、患者级表头或与本地 FOV 键匹配的字段。Depot 页面描述的发布表为 Channels、CellTypes、Labels/Populations 和 CellTable；本地 10 个 FOV 均对应发布键，`sample_metadata.csv` 为0行、图像 `sample_id` 为空。独立 reviewer 的 study/sample M01 均为 unresolved、M02 均为 not_applicable：已确认当前 Depot 发布范围内没有患者 Meta 表，但原始 melanoma acquisition 身份仍未建立，不能外推为别处不存在患者表。[CellTuneDepot](https://celltune.org/celltunedepot.html)，[CellTune 正式论文](https://www.nature.com/articles/s41592-026-03162-2)，[CaltechAUTHORS 正式附件记录](https://authors.library.caltech.edu/records/8dgsd-e8p07)
- Yeo：公开 Supplementary Table S11 的 141 条 TMA/Seq 记录按 TB# 聚为 73 个来源样本/患者行；73/73 行有来源 PT# 患者键。独立复核重建确认 73 行与来源一致，sample/study M02 均通过；Age 与 EBV 字段 12 个字面 `NA` 按来源原样保留，不解释为实际值。精确 TMA+core/Seq 连接支持 20/47 张图像到 18 个样本，其余 27 张无可验证公开键，M01 仍未决。[Zenodo](https://zenodo.org/records/18379156)，[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13091187/)
- Campbell：官方作者 Supplementary Table S1/S2 已取得；45 个 MIBI specimen、26 个 source subject 已写入现有 sample 表，45/45 行有 patient key。arm、histology、response、prior MAPKi、response group 和 specimen/timepoint 均按来源映射；论文报告的 baseline 与 on-therapy 采集窗口也已保留在 `metadata_notes`。所查作者表未含个体年龄、性别、分期或生存字段。202 张处理图像均有 `sample_id`；按 ROI 目录键精确连接 S2 的 `mibi.id` 可匹配 202/202 张图像并与当前 sample 外键一致，但其中 7 张的 native FOV subject label 与 2 个发布 specimen 组不一致。作者公开代码引用 `mibi_roi_map.tsv`，归档的 GitHub main tree 与 S1/S2 附件未包含该 crosswalk，因此目录键匹配未消除来源冲突。对应 sample M01/M02 保持未决；这不是“公开患者表不存在”的结论。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13274726/)，[作者补充表](https://ndownloader.figshare.com/files/68090540)，[作者代码](https://github.com/kcampbel/s1616-public)
- Bowen：作者 `meta_sample.tsv` 与 SPoRA tissues 的 58 个 ROI 键及 Subject、Visit、Treatment Arm 字段经独立复核一致；当前 36 行覆盖 18 个患者键，所有非空样本字段均有 mapping。官方 Table 1 与 ROI 表存在 6 个编号组别冲突，GEO SOFT 的数值标签也与 ROI 表 12/18 组别不一致，FlowJo 仅给出 ROI 文件名，不能建立患者剂量交叉键。样本表已填，但 sample M02 因该患者级临床交叉未决。
- Ohara：作者 `samples.csv` 含 14 个 `sample_name`，58/58 张 TIFF 按目录名连接；作者表提供 7 个原样 PMID 和 14 个 PMID_block。`sample_metadata.csv` 保留这 14 行及来源键，未合并 PMID。新增 mapping 覆盖公开 source manifest 的 `study`、作者 `sample_name` 到 `sample_id`，以及论文明确报告的诊断与 tumor tissue 属性；独立复核确认 14/14 行的样本值与作者表一致，58/58 张图像外键可解析，10/10 个非空 sample 字段均有 mapping。PMC 正文的 Associated Data 列出 Supplementary Data 1 CSV、Source Data XLSX，下载在当前工具渠道触发 reCAPTCHA/cache miss，附件仍待取得。论文报告 14 个 IMC 样本来自 6 位患者，与作者表的 7 个 PMID 冲突；未找到可解释差异的公开交叉表，M02 保持未决，不能将表述为患者级临床 Meta 已齐全。[论文与官方附件列表](https://pmc.ncbi.nlm.nih.gov/articles/PMC10948878/)，[Zenodo 作者代码与样本表](https://zenodo.org/records/5719188)
- Chakraborty：Zenodo `Melanoma_metadata.xlsx` 与论文 Supplementary Material 4 的 `Patient_info` 按 UKF 患者伪名连接；13/13 图像关联行有 6 位患者键和可用治疗、BMI、原发/转移材料、治疗初治状态及 ICI 反应信息。论文称组织在 ICI 前三个月内采集，窗口只映射到 `pre_treatment`，更精确日数仍未知。独立复核重建确认 13/13 行患者级来源值一致，补齐 `study` 映射后 17/17 个非空 sample 字段均有 mapping。补充表中的 22 个 `Tissue_ID` 没有与 13 张 IMC 图像后缀的 crosswalk，故 `source_sample_id`、`biospecimen_group_id` 为空，M01/M02 仍未决；不能将患者级材料/部位信息解释为已确认的图像 specimen 信息。来源未说明 response criteria/version，`response_criteria` 留空。Supplementary Material 4 已归档至 NAS `raw/public/Chakraborty2026_IMC_Melanoma_TAM/metadata/`。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13737523/)，[Zenodo](https://zenodo.org/records/15124735)
- Gerber：官方 Hepatology H992 Table S1 含 16 条 NASH-HCC 患者临床记录，当前已逐行转录，18 个 source extension 与原表逐值匹配。Table S1 的 `No.` 是行号而非来源患者或生物样本 ID；当前 sample_id/patient_key 为项目生成行号键，不作有效来源身份。43 张本地图像的 `sample_id` 全空，公开附件没有 TMA core/acquisition/image→患者交叉键；因此公开患者 Meta 存在，但无法形成符合生物样本粒度的已验证 sample 表，M01/M02 未决。[论文](https://www.nature.com/articles/s42003-026-09961-9)，[BioImage Archive S-BIAD2091](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2091)，[Zenodo](https://zenodo.org/records/15641167)，[Hepatology Supplement](https://cdn-links.lww.com/permalink/hep/h/hep_2023_09_06_yu_hep-23-0611r1_sdc2.pdf)
- Wu：官方 Source Data 工作簿的 Supplementary Table 1 列出20名用于IMC的患者（10名PCR、10名non-PCR）；Patient ID在六张工作表内均可精确匹配。现已据六表填入20条患者级临床记录，20/20行有有效 `patient_key`，67个来源扩展列均有映射，1,340个来源单元逐项读回一致。项目 `sample_id` 是患者记录键，因为公开表未给source sample/biospecimen ID；21张MCD acquisition仍有0张可连接该表，M01未决，M02等待独立复核。已填可直接支持的基线年龄、BMI、ECOG、临床TNM/分期、RECIST固定时点评估及定义明确的OS/DFS；未将客观缓解或pCR改写为 `best_overall_response`。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12827353/)，[Figshare数据](https://doi.org/10.6084/m9.figshare.30466160.v1)
- Aoki：官方 JCO 正文确认最终 Data Supplement 含线上 Appendix Tables A1–A7，并明确 Appendix Table A1 汇总 validation cohort 的患者特征。bioRxiv v4 Supplemental Tables 只有队列汇总和 marker panel，不能替代最终附件。独立复核再次确认 ASCO supplement landing/direct routes 不可取得，PMC 页面没有附加文件，Zenodo 7963681 仅列 8 个 TMA ZIP。当前 `sample_metadata.csv` 为0行；299张图像的 `sample_id` 均为空。reviewer 最新 study/sample M01 均为 unresolved、M02 均为 not_applicable。最终附件仍是公开来源线索，其内容和患者/样本 ID、acquisition/core 交叉键未核实，不能据访问失败断言数据不存在。[Aoki Zenodo](https://zenodo.org/records/7963681)，[JCO 正文](https://ascopubs.org/doi/full/10.1200/JCO.23.01115)，[Aoki EGA](https://ega-archive.org/datasets/EGAD00001010892)，[bioRxiv v4 Supplemental Tables](https://www.biorxiv.org/content/biorxiv/early/2023/06/01/2023.05.24.542195/DC1/embed/media-1.xlsx?download=true)
- Ajaib：论文摘要报告 5 位患者的原发与复发配对样本；本地 M01/M02 通过，映射记录 10 个 biological samples 与 5 位患者。[PubMed 论文记录](https://pubmed.ncbi.nlm.nih.gov/40898767/)
- Ali：IDR 官方 annotation 的 483 个 Metabric ID 与 483 个 sample group 一一对应；548/548 张图像的 sample 外键可解析。12 个组无 Table 5 或 cBioPortal 临床行，保留 IDR 身份键，未填缺失临床字段；15,728/15,728 个已发布 Table 5/6 非空映射值一致。独立复核确认 study/sample M01、M02 全通过。[IDR 官方数据页](https://idr.openmicroscopy.org/study/idr0076/)，[论文](https://www.nature.com/articles/s43018-020-0026-6)
- Ruf：TCIA 官方页列出 15 位受试者及临床 XLSX；本地 processed 表有 17 行，其中 15 行连接到 HCC 患者，另外 2 行是匿名验证标本，M01/M02 仍有未决记录。[TCIA 数据页](https://www.cancerimagingarchive.net/collection/codex-imaging-of-hcc/)
- Meylan：BioImage Archive 与 Nature 补充表支持 32 条样本记录连接到 16 个患者键；字段映射和已填值经独立 M02 复核通过。M01 仍未决：本地没有 `images.parquet`，image→sample 覆盖无法核实。[BioImage Archive S-BIAD1921](https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD1921)，[论文](https://doi.org/10.1038/s41586-023-06623-2)
- HuBMAP：NAS 原始 SPoRA `tissues.parquet` 有 80 个 CODEX tissue 行、15 个患者键及 specimen/原始文件名字段。当前 processed 的 40 张图像 `source_image_id` 与 40 个 `tissue_id` 精确匹配；34 张图像逐一对应 tissue，另有 3 个 HuBMAP source-sample 组各包含两张 raster。独立复核确认 37 个样本行均连到 15 个患者，image sample 外键 40/40；`field_mapping.csv` 已覆盖所有非空字段，M01/M02 均通过。公开来源未提供可填的年龄、性别等临床协变量。[SPoRA HuBMAP 队列](https://spora.bioimagearchive.org/bioimage-archive/projects/spora/dataset)，[SPoRA tissues 格式](https://spora.epfl.ch/docs/io/format.html)，[HuBMAP APIs](https://docs.hubmapconsortium.org/apis)，[HuBMAP sample registration](https://docs.hubmapconsortium.org/data-submission/Section2)
- Allam2024：PNAS Nexus 补充表2的六个组织限定 donor 标签已精确对应六个 source group；`source_subject_id` 与 tissue-qualified `patient_key` 均填 6/6，年龄/性别值填 4/6，来源标为 n/a 的两行留空。该键不跨组织合并同名 donor；图像 sample 外键 6/6、sample M02 通过，但跨组织患者等价和 vendor tissue ID→source folder 仍无公开交叉键，M01 未决。[论文](https://academic.oup.com/pnasnexus/article/3/8/pgae285/7717711)，[补充材料](https://academic.oup.com/pnasnexus/article/3/8/pgae285/7717711#supplementary-data)
- Meyer：SPoRA `tissues.parquet` 的 450 个组织记录精确映射到 450 个 sample 行和 215 个患者键，450/450 图像外键可解析。独立复核确认 66/66 个非空来源字段有映射、68 个直接映射共 30,600/30,600 个值一致，45 个自定义扩展均有类型定义；`stage_basis` 450/450 为 pathological。冲突 pM 留空并保留组织级原值，年龄/OS/DFS 时间起点未定义故不填入标准化时间字段。study/sample M01、M02 均通过。[SPoRA 队列](https://bodenmillergroup.github.io/imcdatasets/reference/Meyer_2025_TripleNegativeBreastCancer.html)，[Zenodo v3](https://zenodo.org/records/20414755)
- Parsons：BioImage Archive 官方文件清单与 eLife 论文/作者仓库已查。98 行中19个 `Matched with?` 值精确指向现有 `sample_id`，另1个外部目标只保留在 `source_matched_with`；19个连接只表示样本间来源引用，不表示同患者。字段说明已修正，独立 sample-scope M02 通过；患者键仍为0/98，680/803图像sample外键可解析，M01仍未决。作者README提到的独立 `CleanCohort_Metadata` 未在公开仓库跟踪，公开来源中未找到可复现的患者交叉表。[BioImage Archive S-BIAD2027](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2027)，[eLife 论文](https://elifesciences.org/reviewed-preprints/109177)，[作者仓库](https://github.com/Schumacher-group/ML4SpatialAnalysis)
- Xu：论文补充表1包含42位患者的临床/人口学资料，论文另报告110个肿瘤组织样本；当前228行由MCD member/acquisition构成，属于未确认的ROI级键，不能按项目sample字典当作生物学取样。228行均无 `patient_key`，Table S1与当前MCD core/ROI没有已验证的公共连接键；独立复核确认sample-scope M01/M02均未决。该结论是“公开患者表存在但无法连到当前样本行”，不是“患者数据不存在”。[论文与补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC11023532/)，[Zenodo 7618945](https://zenodo.org/records/7618945)
- Zhu2025：10份 BSW-Validation whole-section sample 行已补充公开文件名、队列/组织属性及来源定义的 AA/EA 群体标签；数值后缀未用作患者身份，患者键仍为0/10。57行 `clinical_attributes.txt` 属于 BSW-Discovery；独立复核检查论文、补充信息、Source Data 1/2、SFig2，均未发现 BSW2 个体临床表或可将 Discovery 临床行连接到当前 acquisition 的键。Source Data 的 `Case#_ROI#` 不能连接本地5个 TMA-8 acquisition；故 sample/study M01 未决，补齐 `study` 字段映射后 sample M02 通过。[Zenodo](https://zenodo.org/records/15242478)，[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12271405/)，[补充信息](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-61034-3/MediaObjects/41467_2025_61034_MOESM1_ESM.pptx)，[Source Data 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-61034-3/MediaObjects/41467_2025_61034_MOESM4_ESM.zip)，[Source Data 2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-61034-3/MediaObjects/41467_2025_61034_MOESM5_ESM.zip)
- Hiltbrunner：Zenodo 8041882公开14个response/resistance MCD样本、作者分析脚本和159.8 MB单细胞RDS。Supplementary Table 3按患者编号及时间点提供性别、组织学、标本部位和时间点；与MCD basename及论文定义的7组配对样本交叉匹配后，14/14行有患者键、性别、标本部位和治疗状态，组织学12/14行一致填入，P4冲突项留空。作者脚本引用的 `Image.csv` 未在公开发布中；RDS的numeric PatientID/SampleID可内部配对，但不映射到MCD basename，故未用作连接。公开年龄为队列汇总，分期为诊断时点且与索引时点语义不一致，二者未写入个体行。独立复核确认study/sample M01、M02均通过。[论文](https://www.nature.com/articles/s41467-023-40745-5)，[Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-40745-5/MediaObjects/41467_2023_40745_MOESM1_ESM.pdf)，[Zenodo 8041882](https://zenodo.org/records/8041882)
- Börner：本次按 processed 中全部公开目录纳入审计；经 official SPoRA `tissues.parquet` 和 BioImage Archive namespace receipt 确认，已填 80 行并映射到 15 个患者。独立复核确认 80/80 行精确对应 `tissue_id`，13 个非空字段均有映射；study/sample M01、M02 通过。该目录没有本地 `images.parquet`，不表示图像外键已在 processed 表中核实。[SPoRA 队列](https://spora.bioimagearchive.org/bioimage-archive/projects/spora/dataset)，[SPoRA 格式](https://spora.epfl.ch/docs/io/format.html)，[Börner 论文](https://www.nature.com/articles/s41592-024-02563-5)
- MartinezMorilla：当前 processed 目录只有 review 记录，没有 sample 表、field mapping 或图像；对应 raw/public 目录为空且无 `metadata.json`，公开 study catalog 也没有该 study 行。catalog 中唯一精确作者候选是 2021 melanoma 研究，已被标记为排除；fetal/adult intestinal IMC 对应 Guo 等人 2020 年的 34-marker panel 研究，作者/年份不同且没有 target-linked sample 表。本地 review 记录确认目录来源身份未建立；因此不能把两篇相似论文并入，也不能据此判定目标来源患者数据不存在。[Martinez-Morilla 2021 PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/33504554/)，[Guo et al. 2020](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2020.01466/full)

- Radtke2024 CellDive FL：官方 Table S1 共 33 行，其中 7 行按 Sample ID 精确匹配当前处理的 7 个样本。11 个来源值字段逐行核对 77/77 项通过；S1 的明确缺失标记按原文保留，并已记录与 IDR 空值的优先级。age 原值保留为未锚定文本，baseline Lugano 字段按 Methods 记录。5 个 FL 标签保留为 study-local 匿名患者键；两个 nLN 样本无公开 donor crosswalk。独立复核确认 sample/study M02 通过；由于 nLN donor 关系未公开且 `images.parquet` 仅覆盖 3/7 个样本，M01 仍未决。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10966827/)，[官方 Table S1](https://pmc.ncbi.nlm.nih.gov/articles/instance/10966827/bin/NIHMS1967473-supplement-2.xlsx)

## 下一步

- Cho2025：最终论文称 Table S1 提供 12 位患者的特征和 core 数；NAS raw metadata bundle 中的 article XML 将 `media-2.xlsx` 列为官方 Supplement 2。工作簿有 64 个 core 行和 12 个患者 ID，`ID` 去连字符后与现有 12 个 `source_subject_id` 精确匹配；`Specimen designation.xlsx` 的 64 个 `Name` 与 64 个 clinical `core` 精确对应。现有 sample_metadata 16/16 行的年龄、治疗字段逐字匹配，性别按既有 M/F→male/female 规则映射一致；images.parquet 的 58/58 行 core 与 sample patient 一致。此前 M02 因未取得最终附件而未决，现待独立 reviewer 重审；M01 仍受原图像源身份检查限制。外部 PMC/Figshare 直链仍返回 proof-of-work/WAF 响应，但归档附件已满足本地核验。[最终论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12988722/)，[Table S1 官方 Figshare 记录](https://aacr.figshare.com/articles/dataset/Table_S1_from_Computational_Modeling_of_Cellular_Influence_Delineates_Functionally_Relevant_and_Distinct_Cellular_Neighborhoods_in_Primary_and_Metastatic_Pancreatic_Ductal_Adenocarcinoma/32728909)

上一批已复核 Aoki、Allam、Bever、Bortolomeazzi、Bowen、Campbell、Chakraborty 七项；6 项已有非空 sample 表，Aoki 官方最终 Data Supplement 及患者/样本交叉键仍未取得，不能判为公开数据不存在。当前批次为 Bowen2023、Kucukkose2024、Ohara2024、Piyadasa2025、Ranek2026、Tu2026、Zhuhuman2025；已有表及既有 CLI 报告显示七项 M02 均未闭合。本轮重新启动的七个只读 `gpt-6-luna/max` CLI 在模型响应流多次断开后停止，未产生新报告，CLI 未写 NAS。Piyadasa 的候选表按 652 个显式样本键连接，准备保留 74 个来源扩展字段；Ohara 的候选表按 14/14 个作者样本键连接，准备补入 3 个来源字段并修正 1 条过时 mapping。两份候选与对应 review 更新仍在 `/tmp`，正式 NAS 写入租约排队中；当前 NAS 表格未因此改变，M02 仍未决。取得租约并发布后，再独立复核这两项；Bowen 的临床剂量交叉、Kucukkose 的 pt ID crosswalk、Ranek 的 patient bridge、Tu 的 patient grouping、Zhuhuman 的 M02 证据继续按已有缺口处理。若公开发布中确实没有个体患者/样本表，由独立任务身份复核并限定结论范围；患者表存在但无法连到当前生物样本时，不推断 sample 身份。更新同步至 NAS 对应 `sample_metadata.csv`、`field_mapping.csv` 与 `review_results.csv`。
