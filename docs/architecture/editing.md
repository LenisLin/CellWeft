# 输入编辑与输出语义

本文维护编辑行为，字段基础见[data_contract.md](data_contract.md)，研究对照见[扰动评价](../evaluation/perturbation.md)。

## 1. EditSpec 与模式

不可变 EditSpec 至少包含 edit_id、mode、target_cell_ids、target_marker_ids、scope、fill_policy、seed、baseline_manifest_hash；使用指定空间关系选取细胞时记录选择规则。

`train_mask` 的图像遮盖、目标和视图由[训练规范](training.md)维护。`inference_delete` 在原图坐标构造编辑图像和实例视图后重新前向，读取保留细胞与固定位置／区域表征。

原始图像、实例和实测域保持记录；`pixel_observed` 表示当前视图的可用观测，定义见[数据合同](data_contract.md)。

## 2. Marker 观测删除

显式指定 ROI／slide 全局范围或选定细胞集合。全局删除目标图像通道的观测；局部删除仅作用于目标实例和指定 marker 的像素，遮挡像素的输入值置 0、`pixel_observed=0`。原始实测记录保留，protein identity bank 不变。真实测量零值和缺失观测分别记录；观测删除不自动等同于功能敲除。

## 3. Cell depletion

1. 将删除对象解析为去重 stable cell IDs，定位其原图实例 mask。确认目标和操作范围可返回原始数据。
2. 在全部 downstream 蛋白通道中将目标实例像素置 0 并设 `pixel_observed=0`，从编辑实例视图移除目标身份。保留原始实测域和实例外像素。
3. 按 §5 重编码受影响窗口，以相同 stable cell ID 对齐保留细胞。分别比较完整区域与固定保留细胞集合。

## 4. Niche depletion

将 niche／neighborhood／region selection 解析为去重 cell ID 集合，调用同一 cell-depletion engine。选择与邻近关系由坐标和实例确定；保留所选规则和身份。

## 5. 比较与保存

固定基线归一化统计、空间网格、原图位置、coverage／ownership、模型参数与推理配置，关闭随机 augmentation／dropout。以 edited image 和 instance view 为输入，重编码全部图像像素、观测状态或实例读出成员发生改变的 crop，包括跨窗实例涉及的窗口。其余窗口可复用同输入版本和同配置的基线特征。按 stable cell ID、固定 patch 位置与 region ID 对齐。

报告完整区域、固定保留细胞集合和操作区外的变化。导出 cell embedding、spatial patch、crop CLS 和固定区域／ROI 的 baseline、edited、delta、cosine distance 与 normalized L2。读出接口见[模型规范](model.md)。MIL 使用更新后的完整 bag 重新计算 attention。

保存 EditSpec、范围、实际剩余 IDs、模型／输入／checkpoint／normalization 身份、crop transform 和相关参数。空 region 保留稳定 ID 和 empty flag。No-op 应在数值容差内一致；有方向的状态评分使用独立固定的 probe 或参考，评价见[扰动评价](../evaluation/perturbation.md)。

## 6. 替换与状态编辑

Figure 5 的候选操作和剂量见[Figure 5 协议](../paper/figure5_spatial_editing.md)。读出统一按 §5 执行。

### 6.1 细胞替换

1. 一次替换移除一个原细胞并置入一个真实供体细胞；按具体实验预先确定供体选择和同状态对照，并记录来源、替换数、面积及几何变换。
2. 原实例移除区域按 cell depletion 遮挡；经配准的供体实例提供编辑图像像素与新实例，未覆盖的原实例区域保持不可观测，供体支持范围不得覆盖保留实例。
3. 使用独立编辑视图承接新身份、实例及来源，按[数据合同](data_contract.md)保留 baseline；不重编号原始 cell_index，也不借用原身份作为供体身份。
4. 按预先确定的目标尺度转换供体像素，使用查询图的基线归一化参数。
5. 按 §5 更新受影响窗口。按需以基线定量规则从编辑像素计算供体或其他受影响实例的 whole-cell 均值，用于编辑记录和分析。

### 6.2 Marker 数值／状态编辑

1. 仅修改指定细胞和 marker；状态模板、编辑程度及保留的身份通道由实验定义。确认：实例外像素、未选 marker 和其他细胞的原始测量保持固定。
2. 构造编辑图像与实例视图，保留原始 assayed 和编辑来源。按需要使用 baseline 相同 whole-cell 均值规则记录受影响细胞的编辑后数值。
3. 按 §5 重编码后比较目标细胞与空间读出。

替换及状态编辑的供体、模板／尺度、几何处理、指定数值、实际成本和来源写入编辑记录；原始记录及编辑视图分别保存。外部扰动来源的响应幅度不直接转换为空间图像强度，方向性输入及解释边界见[扰动评价](../evaluation/perturbation.md)；具体编辑参数仍由 Figure 5 协议在实施前锁定。
