# Feng2026_IMC_BladderCancer 膀胱癌队列数据准备

2026-11-24，用户确认数据所有者为 Feng，登记年份为 2026，癌种为膀胱癌。正式 study ID 为 `Feng2026_IMC_BladderCancer`，处理结果位于主 NAS `processed/Feng2026_IMC_BladderCancer/`。原始来源为 `/mnt/public/lyx/horizontalProject/IMC_20240725_63`，原目录名保留为来源别名。首批处理来源索引中的 520 个 ROI；同日按用户要求删除其中 5 个患者连接未确认 ROI 的处理目录和当前图像索引，正式保留 515 图，稳定图像 ID 不重新编号。来源及历史验收证据保留。

## 已核对的范围

| 对象 | 实查结果 | 对后续处理的影响 |
| --- | --- | --- |
| `steinbock/images.csv` | 520 个唯一图像名，518 行唯一对应 MCD 文件与 acquisition ID，原生宽高全部相符。另 2 行各自唯一定位到一份原始 TXT，两份表头均为 45 列 | 按这 520 行处理，两份 TXT 单独验证 XY、强度列、方向及与 MCD 的重复关系 |
| 原始 MCD | 82 个容器共 714 个采集。533 个为同一套 39 通道，181 个为另一套 3 通道。首批索引覆盖其中 518 个 39 通道采集 | 索引外有 15 个 39 通道采集，其中 8 个数据起止偏移相等，另 7 个有正的数据跨度。后者的组织身份、像素完整性和唯一性待核实 |
| panel 与导出图像 | `panel.csv` 有 39 行，来源保留 35 行。520 个导出图像均为索引指定的 35 通道及宽高。同名 mask 均为同宽高的整型单平面 | S1 映射从来源 39 通道开始。对齐并验收来源 Steinbock 实例 mask，通过者复用，缺失或确认不合格者按现行规范重分割 |
| 患者连接 | 两版来源 RDS 均可按 ROI 和尺寸连接全部 520 图。结合分组表，515 图确认对应 62 个患者键，5 图的患者连接待定 | `images.parquet` 的 `source_patient_key` 保存直接确认的连接。来源未提供独立生物学取样的标识，`sample_metadata.csv` 保留表头 |

已接收 606 份来源文件，包括 82 份 MCD、2 份 TXT、520 张来源 mask 和两版 RDS。520 个 ROI 的来源引用已切换到主 NAS。独立 D01 复核覆盖全部输入结构，并抽查 7 个 MCD ROI 和 2 个 TXT ROI 的原始像素。原始 panel 共 39 通道，输入保留 35 个，其中 31 个蛋白已映射到固定向量，2 个核通道仅用于分割，另 2 个待映射。

520 个来源 mask 通过逐图几何、实例标签和面积核对。20 张空间叠加抽查覆盖 17 个分层，包含两份 TXT。同一批次的组织区域估计在 3 张代表图上覆盖约 7.9%–14.4% 的画布，最大连通区域仅占估计面积的 6.3%–11.7%。训练窗口的组织筛选保留为 `pending`。逐细胞 `crop_origins` 保存已计算的窗口相交关系。

## 实施与验收

原始 520 个 ROI 全部完成技术处理，共 3,695,878 个细胞，20 个代表 ROI 通过数值复核。删除 5 图后，保留 515 图、3,659,156 个细胞，代表图中仍保留 19 图。独立 D01 已按当前 515 图重新绑定，逐图结构及输入版本重核，沿用范围内 6 个 MCD 和 2 个 TXT 的原始像素抽查。现行回执为主 NAS `processed/Feng2026_IMC_BladderCancer/processing_summary.json`。

## 训练与内部验证

| 用途 | 正式 split | 患者数 | ROI 数 |
| --- | --- | ---: | ---: |
| 训练 | `development_train` | 50 | 421 |
| 内部验证与选参 | `development_validation` | 12 | 94 |
| 合计 | | 62 | 515 |

先对确认的患者键排序，再用 Python `random.Random(20261124).shuffle` 打乱，前 `round(62×0.2)=12` 位用于验证。全部切片、ROI 和窗口继承患者分组。正式成员保存在 study 的 `training_split.json`，并加入主实验训练目录；仅训练 split 参与拟合。31 个已映射且绑定固定向量的蛋白作为有效输入，2 个待映射蛋白维持缺口。训练组织筛选仍为 `pending`，待完成组织估计核验后开放窗口采样。

删除及划分入口为 `feng2026_training_split.py --retire`；随后运行 `feng2026_review.py --publish`，再执行 `feng2026_training_split.py --publish` 和 `feng2026_summary.py`。历史 520 图证据用于来源追溯，当前消费者以正式 515 图索引及新 D01 为准。

## 处理步骤

1. **锁定来源范围。** 核对与既有私有 study 的重复采集，按本批选定范围在 `raw/private/Feng2026_IMC_BladderCancer/` 登记原始文件与来源表。记录索引外采集及处置理由。原始来源保留在原位置。
2. **整理 study 输入。** 用原始 MCD 或核对通过的 TXT 读取独立 ROI。核对方向、像素尺度、非矩形实测区域与四侧补边。按 S1 对接有序 panel，记录表达、仅分割、排除、映射待定的通道和原始索引。生成 `images.parquet`、`panel_mapping.csv`、`sample_metadata.csv`、`field_mapping.csv`、`review_results.csv` 与逐图 `input.tiff`。患者键仅按来源证据填写。
3. **独立验收输入。** 全量核对来源绑定、表间引用和文件头，代表图从 3 张起步，覆盖 MCD、TXT、几何和分割差异。比对来源像素与导出像素，核对实测域和补边。对通过的逐图输入记录 D01，按同一 panel 生成 `marker_embeddings.pt`。
4. **生成逐图产物。** 在实测域按通道计算 P99.5，输出 `image.tiff`、`channels.parquet`。按选定策略生成最终 `mask.tiff`，继而发布 `cells.parquet`、`expression.parquet`、`neighbors.parquet`、`metadata.json`；有不规则有效域时提供 `valid_pixels.tiff`。whole-cell 表达在切窗前计算，邻域为同图最多 20 个细胞，`crop_origins` 沿用现有 224／112 训练网格。
5. **发布并对账。** 在正式目录重开 TIFF 和表，逐图检查 mask 标签与 cell 索引、表达行、邻域及 marker 向量引用。代表图核对 mask 叠加、实例均值和边缘窗口。汇总完成、部分完成、患者连接和可采样窗口范围。后续临床或训练用途按私有队列角色另行选取。

执行依据见 [study 输入](../data/study_input.md)、[独立校对](../data/study_input_review.md)、[数据准备](../data/preparation.md)和[存储布局](../architecture/storage.md)。
