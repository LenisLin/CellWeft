# 表征评价与消融

本页维护研究评价，工程正确性检查见[工程验收](../development/engineering.md)。数据成员遵循[来源与划分](../data/catalog.md)。

## 1. 任务与报告

- Marker reconstruction 按 marker 和 cohort 报告，不只给总体均值。
- Cell annotation 比较 intrinsic、neighbor-contextual 和 fused 表示，报告 macro-F1 与混淆矩阵。
- Neighborhood/region retrieval 评价同一区域跨视图检索。
- Niche clustering/domain annotation 在独立有标注数据上报告 ARI/NMI 等指标。
- ROI/tissue 评价器官、亚型、分级及选定临床任务；任务头训练与 backbone 适配分别记录。

## 2. 比较设计

- 蛋白知识：相同空间训练预算比较 PLM-only 与 PLM+regulatory，评价未知 marker/panel 与跨 cohort annotation。
- 上下文：cell-only、neighbor-aware、image-only、late fusion、recursive fusion；包含 neighborhood shuffle 和 distance-rank 消融。
- 融合：固定 image/cell checkpoints，比较无 fusion、late concatenation、单轮/多轮和单一 alignment 开关。
- 联合阶段复用基座比较 masking 策略与少量末端解冻；改变 protein bank、image tokenizer 或 cell pretraining 时重训相应基座。
- 不对全部因素做全组合从头训练；完整记录初始化、训练预算、split、seed 和适配范围。
- 至少一整家医院作为不参加预训练与适配的外部评价；另一医院的适配/探索结果单独说明。

参考基线与 spora-bench 的角色见[参考索引](../references.md) R02、R03、R15；同步删除任务采用独立[扰动评价](perturbation.md)。
