# 输入编辑与输出语义

本文维护编辑行为，字段基础见[data_contract.md](data_contract.md)，研究对照见[扰动评价](../evaluation/perturbation.md)。

## 1. EditSpec 与模式

不可变 EditSpec 至少包含 edit_id、mode、target_cell_ids、target_marker_ids、scope、fill_policy、neighbor_policy、seed、baseline_manifest_hash。
train_mask：保留 full-view teacher targets 和 predictor queries，student 看不到目标原始内容。
inference_delete：删除目标不再产生 prediction queries，不运行 predictor 补回，只读取 surviving cells、regions 和 ROI。
训练 image masking 与推理 pixel edit 分别管理，不能混同为一个数值操作。

## 2. Marker 编辑

显式指定 ROI/slide 全局范围或选定 cell 集合。全局删除同步禁用细胞侧 marker 输入与图像通道。
局部编辑同步改变选中 cell 的 measurement visibility 和相同实例的对应通道像素；不能仅修改 protein identity bank。
缺失输入、实际测量零值和蛋白功能丧失使用不同定义；观测删除不自动等同于功能敲除。

## 3. Cell depletion

1. 将删除对象解析为稳定 cell IDs，从 cell set、marker row 可见性、所有 neighborhoods 和 region membership 中移除。
2. 原子性更新 reverse-neighbor、crop halo、cell–image footprint/membership 和所有 overlapping crops。
3. 在所有通道（含核染）中只处理对应实例像素，以规范背景值替换；保留实例外像素及同 patch 的其他细胞/胞外区域。
4. 不删除整个 16×16 image token，不只删除 pooled embedding。
5. 固定 baseline normalization、原始坐标和 surviving-cell measurements；不随删除重新估计整张 slide 统计。
6. 默认固定 baseline kNN 候选，不补入更远细胞。Neighbor policy 和 rank policy 显式配置；默认按剩余近邻距离连续编号并保留 original rank。
7. 对编辑后的输入完整重新 forward；不复用旧 contextual/image embeddings，初版不采用局部更新优化。

Fill value 必须在对应 normalization 的背景坐标中定义，数值 0 不一定表示背景。

## 4. Niche depletion

将 niche/neighborhood/region selection 解析为去重 cell ID 集合，调用同一 cell-depletion engine。不能仅移除 summary 而保留 constituent cells。

## 5. 比较与保存

使用同一模型参数和推理配置，关闭随机 augmentation/dropout：baseline full forward → edited full forward → 按 ID 对齐比较。

引用已有 baseline 导出前，核对模型、输入、归一化、空间范围及推理设置是否满足同一比较条件。编辑后的输入仍须完整重新 forward。

按相同 surviving cell_id 和固定 region_id 导出 intrinsic、neighbor-contextual、fused-contextual、region、ROI 的 baseline、edited、delta、cosine distance 和 normalized L2。
保存 EditSpec、范围、实际剩余 IDs、模型/输入与 checkpoint hashes、normalization hash、crop transform 和相关参数。文件归属见[storage.md](storage.md)。
删除 cell 不出现在 surviving 输出；空 region 保留稳定 ID 和 empty flag。No-op 应在数值容差内一致；邻居删除后 surviving intrinsic 保持不变。
完整区域读出与固定保留细胞集合的响应分别评价；蛋白级空间响应读出仍属待定接口。
