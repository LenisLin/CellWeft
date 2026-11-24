# 补充表

## Supplementary Table S1. 跨研究 marker 名称与 MAP-KG 编码对应

正式维护表：[supplementary_table_s1_marker_mapping.csv](/mnt/NAS_21T/ProjectData/CellWeft/processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv)。更新日期：2026-11-19。

| 列 | 定义 |
|---|---|
| Study | 来源研究名称，沿用来源清单中的 study 名称 |
| Original marker | 原研究 panel 的 marker 标签，去除查询表附加的 study 消歧后缀 |
| Unified marker | 整合后统一 marker 名称；同时标示排除、仅分割或映射待确认状态 |
| MAP-KG mapping | 对应 gene symbol、成员集合的 `mean(...)`，或指定替代蛋白的 `proxy(...)` |
| UniProt accession | 对应编码成员的参考 accession，顺序与上一列一致 |
| Segmentation channel | 原始通道的分割用途：`1` 为核信号，`2` 为膜／胞质信号，空值为未选入 |

表注：`mean(...)` 表示先取得各成员的 MAP-KG 向量，再进行等权平均；单个 gene symbol 或 `proxy(gene)` 对应一个向量。`proxy(...)` 使用指定蛋白的向量，其 UniProt accession 也是该指定蛋白的参考条目。多个 gene 的 accession 用 ` | ` 分隔，同一 gene 对应的多个参考条目用 `/` 分隔。UniProt 用于身份核对。

`Excluded` 表示从下游 marker 映射排除；`(segmentation only)` 表示仅用于分割；`(mapping pending)` 表示当前未确定编码映射。这三类行的 MAP-KG mapping 和 UniProt accession 均留空。相同 study 中完全相同的原名与映射跨 panel 仅列一次。分割选择独立依据 raw channel，不因统一 marker 名称相同而合并输入通道；`Excluded` 本身不替代原始分割信号的适用性判断。

分割通道按实际 study/panel 规划：核信号优先选择1–2个DNA专用通道，也可选入已结合来源和本地图像复核的Nuc/Nuclear复合信号或总Histone H3/HH3；分割用途与染色身份分别记录，不选Ki67、FOXP3或磷酸化H3等功能／亚群通道作为通用核输入；膜／胞质信号选择4–5个候选，考虑较强信号及免疫、上皮／肿瘤、间质等适合该组织的细胞覆盖。预期信号强度与实际图像验证分别记录。单列标记通过原始panel的通道列表取交集后使用，不能将study通道并集视作一个panel。缺少合适通道或单列无法表达panel差异时，保留具体缺口。

疑似缺少DNA通道时，核对本地原始文件的通道头、采集元数据、数组通道轴及独立核染文件。整理后的panel表或S1没有DNA标签，不足以确认缺失；本地发布版本不含DNA通道，也不等于原实验未检测DNA。原文件不可访问或染色身份不明确时，记录待核查。

对只有同位素标签的Ir通道，结合本地原图小范围像素及核／胞质参照核查空间定位。原生Ir标签与清晰核形态支持其作为核分割候选时，可记为`1`，但需在审核记录中区分“图像支持的核定位”与“来源明确的DNA染料身份”；前者仍含染料身份推断。主要呈背景、不能显示稳定核定位的通道不选入。核蛋白或未知组成的核信号组合不因形态相似而改称DNA专用通道。当前标记是输入候选方案，尚未完成逐图强度和分割效果验证。

表头及内容使用英文；`<Unnamed channel>` 表示原始空通道名的英文占位符，替换此前的中文占位符，不表示新增测量。

当前版本覆盖125个study，共6,634行：5,444行已映射、318行仅分割、819行排除、53行映射待确认（涉及40类marker）。`Segmentation channel`中214行标为`1`、700行标为`2`、其余5,720行留空；用途与编码分类独立。2026-11-19新增IMC_GC的79个来源名称，保留两个原生41通道panel及37通道备份的独立顺序；HSP70保持家族级mapping pending。全局marker registry按现行S1同步为679个身份，原668个身份逐值保持；其中3个新身份来自IMC_GC，另8个此前已在S1中。本次只完成来源身份与候选用途对接，没有授予IMC_GC图像导出或分割质量准入。

逐study候选、实际panel通道索引、局部图像及未决问题见[正式矩阵](/mnt/NAS_21T/ProjectData/CellWeft/processed/_shared/paper/segmentation_channel_review/segmentation_channel_review.md)。初轮审核覆盖287个panel变体；本轮按原始通道映射重新计算，其中262个满足1–2核和4–5膜／胞质的数量目标，包含历史panel，不能据此推定当前准入或图像可用性。数量达标不等于图像质量或分割效果验收。经[原始图像复核](/mnt/NAS_21T/ProjectData/CellWeft/processed/_shared/paper/segmentation_channel_review/nuclear_panel_recheck/root_review.md)并获用户确认，10个study的Nuc/Nuclear或总H3核候选已写入，6个study的膜／胞质选择同步调整；[写回验证](/mnt/NAS_21T/ProjectData/CellWeft/processed/_shared/paper/segmentation_channel_review/nuclear_panel_recheck/s1_writeback_validation.json)记录26处用途变更及当前panel索引。Greenwald、Ranek的组蛋白修饰核候选继续留空；Meylan等待原始数据核验。Amitay本轮表格与作者mask均保持不变，后续核对并复用已本地化的16个作者mask。投稿前根据实际纳入的研究及最终训练marker集合定稿。

## 表格维护与使用

本表作为跨研究marker映射的唯一维护表，通过Study与Original marker查询统一名称、MAP编码成员和UniProt accession。后续修订直接更新本表，并同步本页统计。向量计算遵循上述表注与[Methods](methods.md)。

各study的panel组成、通道顺序及图像对应从来源metadata读取。
