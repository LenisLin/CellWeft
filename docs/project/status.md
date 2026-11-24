# 当前状态

## 研究设计与实施

- V1 以单图像 Spatial ViT-B/8 编码 patch 和 CLS，以实例 adapter 读出细胞，以任务 MIL 汇总图像或患者窗口。Figure 2／3 评价蛋白与组织表征，Figure 4 首批实施胃癌 TRG，Figure 5 保持原人工编辑，Figure 6／补充材料探索外部实测蛋白响应指导的编辑。[研究计划](research_plan.md)
- CellWeft 模型及训练代码待实现。基座预训练、冻结像素 decoder 和下游评价待运行。[工程规范](../development/engineering.md)
- 正式 NAS `processed/_catalog/public/experiments.json` 已同步 CellWeft 读出、训练／评价窗口及 S4 query／gallery 网格，任务成员与患者划分已比对。[表征评价](../evaluation/representation.md)
- 现行规范、任务 IO 和阅读入口已归并，旧讨论稿、结束的执行计划及退役入口已清理。文档链接、配置格式与源码语法检查通过。

## 已验证数据资产

- 私有膀胱癌 `Feng2026_IMC_BladderCancer` 原处理 520 图；按用户要求删除 5 个患者连接未确认 ROI 后，正式保留 515 图、62 位患者、3,659,156 个细胞，独立 D01 及逐图输入绑定已更新。全部复用核验通过的来源实例 mask，当前 19 张代表图通过数值复核。训练／内部验证固定为 50 位患者／421 图及 12 位患者／94 图，已加入正式 `experiments.json`，验证图不参与拟合。交付见 study 的 `processing_summary.json` 和 `training_split.json`。[队列工作流](../development/imc_20240725_63_preparation.md)

- 主 NAS 已有逐图蛋白图像、实例 mask、细胞索引、whole-cell 定量、邻域、窗口对应和 study 级固定 marker 向量。2026-11-11 回执覆盖特定回传批次的 103 个 study、19,931 张唯一图像。当前数据量和各用途范围按正式目录、逐图记录与实验成员核定。技术接收、组织采样、患者连接及训练准入分别核验。[数据目录](../data/catalog.md) · [公开患者来源审计](../data/patient_metadata_public_scope_audit_20261111.md)
- 胃癌 `IMC_GC` 保留范围的 958 图已完成技术处理并正式接收。两条 MCD 采集缺完整像素，未入接收范围。患者与时间点连接以现行临床表为准，各分析按 Figure 4 协议选取患者。[Figure 4 协议](../paper/figure4_gastric_cancer.md)
- 蛋白扰动候选归入 NAS `raw/sc_protein/`。GSE278572、GSE330295 已核读逐细胞蛋白测量及对照连接，PRJDB16517 已核读作者聚合响应表。8T 交付副本保留。GSE300551 于 2026-11-24 复试官方 Figshare 入口仍返回 403，未取得 processed H5AD 及正式 vehicle 连接，已退出本轮并清理接收与临时复核材料。[来源指南](../development/protein_perturbation_download_guide.md)
- 8T 训练盘已搬运 123 个 tar，共 4,967,427,051,520 字节（约 4.97 TB）。全部包的 221,260 条成员通过结构核验，11 包的 68 个样本文件与源一致，其中 40 文件全字节比较，其余比较头、中、尾各 64 KiB。未执行全盘逐字节比较或全图解码。NAS 原件保留。训练服务器恢复待完成，解包后同步当前 SSD 代码、规范和主 NAS 实验配置，按[恢复流程](../development/training_disk8t_transfer.md)验收。
- 项目外辅助来源 `lungST` 的默认清单已接收。来源映射、验收及未交付范围见 8T 盘 `lungST/supplement_manifest.json` 与 `lungST/supplement_complete.json`，后续配准需补 GSE311609 的 H&E–Xenium 变换。

## 进行中与未闭合来源

- Feng 队列的 2 个通道待映射。3 张代表图的组织估计较碎，保留 515 图的训练窗口组织筛选仍为 `pending`；成员入表后仍须闭合窗口准入。[队列工作流](../development/imc_20240725_63_preparation.md)
- MartinezCasals 的 metadata-only successor 已落盘，现行接收读取因样本与字段映射表版本不符仍未通过。按该 study 的 `source_handoff.json`、正式接收绑定及公开患者审计核对归档残段覆盖和成年标本连接。[公开患者来源审计](../data/patient_metadata_public_scope_audit_20261111.md)
- Johnson 的正式接收读回通过。独立 TIFF 是否由 DICOM 完整覆盖仍待确认，按 `raw/public/Johnson2022_MultiPlatform_MetastaticBreastCancer/` 的 manifest 与来源 metadata 核对并保留未证实可替代的内容。
- Walker 已接收并读回的两段来源是 `raw/public/Walker2026_MIBI_PDAC/images/zenodo_21108550/incomplete/` 下的 tar 前缀，完整范围按原 manifest 核对。正式产物接收读取另有表版本不一致待闭合。
- Parsons 已正式接收 803 图，患者连接与训练用途仍待核验。其他公开队列的缺失来源和连接见逐 study 审计及各来源 metadata。[公开患者来源审计](../data/patient_metadata_public_scope_audit_20261111.md)

## 下一步

1. 按正式清单接入图像、实例与表达监督，运行短程数值、EMA 导出／恢复及显存验证，再进入基座预训练。[工程规范](../development/engineering.md)
2. 按正式成员落地 Figure 4 的 TRG 任务，OS 随访按后续批次实施。[Figure 4 协议](../paper/figure4_gastric_cancer.md)
3. 按三项蛋白来源各自的测量定义和对照完成 marker、细胞类型及条件对应，再估计来源内响应。[来源指南](../development/protein_perturbation_download_guide.md)
