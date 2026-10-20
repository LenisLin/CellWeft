# 删除响应评价

目标是检验 marker、单细胞和多细胞/niche 删除后保留目标细胞与微环境的响应；编辑操作和输出由[editing.md](../architecture/editing.md)维护。

## 1. 读出

固定 target cell、区域和推理条件，比较 intrinsic、neighbor-contextual、fused-contextual、region 与 ROI。
评价影响强度、参考状态方向、空间背景依赖和组合效应；状态方向需要独立定义的参考状态或经过验证的读出。
分别报告完整区域变化与固定保留细胞集合变化，避免将组成改变全部解释为剩余细胞响应。
移除 marker 观测与模拟功能丧失分开评价；不将未建立的蛋白级空间读出描述为现有能力。

## 2. 对照与复现

- No-op。
- 指定 cell type 的 neighbor depletion。
- 删除数量、面积、位置及类型匹配的随机/对照 cell removal。
- 仅 cell branch、仅 image branch 的不一致编辑诊断。
- Marker、cell、niche 三尺度，以及相同 target phenotype、不同邻域的分层比较。
- 多个随机种子和 held-out cohorts；固定视图、归一化和模型参数，完整重新 forward。

背景替换、mask 范围与 rank policy 随实验记录，使操作差异可追踪；本次拆分不新增其默认算法。

## 3. 解释与验证

工程测试检验信息流与一致性；研究评价检验变化分布、选择性和跨 cohort 复现。不能以所有 embedding 变化越大越好作为验收标准。
用独立生物学证据评价响应解释；自然邻域差异提供支持，真实干预实验可进一步检验干预预测。未验证方向或组合效应仍记为待检验假设。
每个具体实验在 docs/experiments/ 中记录问题、成员、对照、读出、判断依据和结果入口；实际 cell IDs、数组、图件与日志写入 NAS。
