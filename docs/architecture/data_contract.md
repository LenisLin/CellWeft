# 双视图数据合同

采用 dataclasses 或同类 typed schemas。稳定实体 ID、测量状态和几何对应是模型/编辑公共边界；物理保存位置见[storage.md](storage.md)。

强度采用固定零下界的逐图逐通道全像素 P99.5，非正分母按图排除。流程与crop参数由[数据准备](../data/preparation.md)维护；本页定义身份及对应，当前实现与验证范围见[项目状态](../project/status.md)。

## 1. 实体与记录

主实体：cohort_id、hospital_id、patient_key、slide_id、roi_id、image_id、crop_id、cell_id、marker_id。Dataset row index 与稳定 cell_id 分开。image_id在study内唯一，跨study使用(study,image_id)；来源短文件名不单独充当跨study身份。查询绑定metadata中的图像及mask产物身份，cell_index在该身份内有效。
Study index：每个study维护一份images.parquet，在全局marker语义统一和study/panel通道清理之后建立，内部按image_id逐条登记。初始字段为study、image_id、channel_count、pixel_size_um、image_path、marker_list_path、output_path；channel_count为来源通道数。image_path是输入目录，output_path是NAS processed/<study>/目录；具体文件/容器ROI由study读取配置与image_id解析，最终图像产物位于output_path/images/<image_id>/。marker_list_path指向study的panel_mapping.csv，读取配置结合image_id选择对应panel。pixel_size_um为µm/px，非等宽像素用x/y分量，多尺度/panel按图登记。
SlideRecord：cohort/platform、pseudonymous patient key、slide/ROI、image/mask 路径、marker 表、pixel_size_x/y、H/W、channel 数、licensing/access、split、source DOI/accession、normalization_version及已有来源完整性记录（若有；不新增常规全量hash）。此为后续扩展记录，不作为初始导入表的必填字段集合。
MarkerRecord：marker_id、display_name、可空 canonical protein ID、可空 gene symbol、measured_form、可空 phosphosite/epitope/clone、channel role、可空 protein-bank row、mapping confidence。Canonical protein ID 在能明确对应单一蛋白时填写；家族／组合保留 marker_id 和来源测量身份。Bank 编码在全部数据准备完成后统一处理，空行号不阻塞 processed 发布；模型输入组装前再检查两分支共享映射。
Total/phospho 即使对应同一 canonical protein 也不合并 marker identity；不同 clone 可共享基础向量，但保留测量 metadata。DNA/DAPI/isotype 不伪造 protein ID。
全项目先建立统一marker registry，再在study/panel阶段完成Manual marker mapping，产生按来源通道位置排列的映射和segmentation/downstream/exclude去向；per-image阶段只应用该映射，准入规则由[数据准备](../data/preparation.md)维护；两分支的下游蛋白输入使用同一映射。
已明确为蛋白家族或组合的来源通道保留测量身份和强度，不因无法直接对应单一 bank 行而删除；其具体 bank 编码待定，不复制成多个独立蛋白测量。
CellRecord/CellTable：全局稳定 cell_id，与 cell_index、mask_label 分开；包含 image_id、slide_id、原图像素 centroid_x/y、area、bbox、crop_origins、nuclear/whole-cell mask 类型及分割来源/版本、marker measurements、assayed mask。微米坐标按需派生，不作为近邻查询必需输入；Annotation 仅作可选评价字段。Mask 复用检查和 cpsam_v2 新分割流程由[数据准备](../data/preparation.md)维护；蛋白定量采用共享 P99.5 归一化图像在 whole-cell mask 内逐通道的全部像素均值（含零值），在 crop/训练重采样前计算；背景 label 0 不建表达行。Mask 固定后冻结 CellTable，来源表达行按 mask_label 显式关联；重新分割形成新 mask 产物身份，不把旧标签与新标签直接视为同一细胞。
Crop坐标直接归入CellTable的crop_origins：每个cell对应一组原图左上角(x0,y0)，实际窗口为[x0,x0+224)×[y0,y0+224)。列表包含全部与实例像素相交的网格窗口，允许部分细胞；bbox仅用于筛候选。网格224/112、贴边规则及H/W保存在image metadata，短图有效范围由H/W计算。随机view的几何变换在加载时维护。

准备产物的最小附加信息：共享图像引用、保留通道顺序、全部来源通道的位置、P99.5（人工排除时为空）、保留/排除原因及统计域/估计方法；排除通道的 TIFF page/表达列为空；mask 引用、类型和基础检查结果及具体问题；后续 indexing 建立 mask_label → cell_index 及区域位置、表达行的对应。metadata的mask_origin区分来源与cpsam_v2，mask_identity在同一正式mask复用时保持；source_mask_check保留一次来源判断。新分割另保存实际核/膜组、模型参数及交接输入/权重身份；来源 mask 复用已有来源信息。这些信息属于准备结果，不扩展study index的七个初始字段。
共享图像包含有效核染和 downstream 蛋白，训练读取 downstream 通道，分割合成信号按需组装。正式格式为 TIFF/Parquet/CSV/JSON，具体文件内容见[存储规范](storage.md#准备产物文件布局)。TIFF tile和压缩吞吐仍待实现验收。

## 2. 张量与状态

图像单样本 D,H,W，batch B,D,H,W；坐标 (x,y)，以原图像素坐标为基础，微米坐标按需派生。
Batch 至少包含：

- images: FloatTensor[B,D_max,H,W]。
- marker_ids: 下游通道的 marker identity；protein_bank_indices: 对应共享 bank 行号。
- marker_embeddings: FloatTensor[B,D_max,R_p]，或共享表与 indices；两分支使用同一固定 bank 及版本。
- marker_valid: BoolTensor[B,D_max]。
- cell_values: FloatTensor[B,N_max,D_max]。
- cell_marker_assayed / visible: BoolTensor[B,N_max,D_max]。
- cell_ids、cell_valid、coordinates。
- neighbor_indices、neighbor_ranks、neighbor_valid；采用下述建议表示时，rank 和 valid 可由排序索引及 -1 推导。
- instance_masks，或按需加载的原图 label maps。
- crop_ids、view_ids、view_kind、transform_records、patient/split provenance。

未测量、真实零/低表达、padding、train mask 和 deletion 分别表示；不能用一个数值兼任。
D 和 N 可变，按 cell 数分桶；padding 不参与融合、attention 或 loss。零细胞 crop 可用于 image-only；joint 明确 skip/empty handling。

## 3. 索引

必须可查询 slide/image → cells/images/markers/split；cell → marker row/mask label/centroid/area；cell → baseline top-20 neighbor indices；cell → reverse-neighbor references；crop → visible/halo cells/grid；cell → overlapping crops/patch footprints。
每张完整独立图像使用 cKDTree 计算 centroid 欧氏近邻，不跨 image_id；排除自身并按距离升序排列，仅保存邻居 index。当前实现采用neighbor_indices[N,20]、不足部分填-1，Parquet列为n01…n20；距离不落盘。
近邻使用原图像素 centroid；若发现非等宽像素，再讨论尺度修正。有效邻居列依次对应 rank 1...20，中心 cell 的 rank 0 由模型单独加入；反向邻居关系从索引派生。
查询主键为 (study, image_id, cell_index)，cell_index 按最终mask的正标签升序从0分配，是该图像固定 CellTable 的连续行号；表达表通过显式 cell_index/marker列关联，不能依赖任意Parquet物理行顺序。crop/view 不重新编号或重新计算邻域。组装 batch 时映射到 batch 行，采用 -1 表示时不得将其用于有效 gather。
Halo 可来自同图像的 crop 外邻域；image alignment target 必须有明确 crop 覆盖。记录 crop 外邻域与视觉范围差异，不用 padding 伪造细胞。删除后的候选与 rank 更新按[editing.md](editing.md)，保留候选的距离次序可由原排序索引取得。

## 4. 空间对应

所有裁剪、旋转和翻转记录 affine/transform ledger，支持原图→view 及坐标 inverse mapping。
Image、instance mask、坐标和 footprint 同步变换；label map 缩放使用最近邻。
跨 crop targets 由共同覆盖定义；两分支不能独立随机裁剪后仅凭同 ID 对齐。
基础块按全图TIFF窗口读取，坐标由cells.crop_origins或image网格规则取得，与实例mask同范围对应；基础 mask 中实际保留的前景标签沿用原图 mask_label，包含边界处部分实例，并关联原图 cell_index。
Global/local views 在训练加载时从配对基础图像块生成，图像与 mask 共用几何参数；小实例可能在缩放后消失，可见 cell 集合由各 view 的实际 mask 标签确定，cell_id/cell_index 保持不变。Local view 不单独落盘；网络输出尺寸与基础 crop 范围见[数据准备](../data/preparation.md)，训练数量及阶段见[training.md](training.md)。

表达矩阵、mask和crop通过同一cell索引直接对应：expression.parquet的cell_index → cells.parquet同键行 → mask_label与crop_origins。加载任一crop时，同范围读取image和mask，以mask中实际正标签反查cell_index，再取得完整表达向量；反向查询某cell则从crop_origins得到全部对应图像块。低组织占比筛选不删除这一关系。

例如cell_index=42、mask_label=107、crop_origins=[(0,0),(112,0)]：表达向量取expression中cell_index=42的行；两块mask分别取mask[0:224,0:224]和mask[0:224,112:336]，其中值107的像素对应该细胞；图像读取完全相同的窗口。

一个cell与多个crop/14×14空间token可以相交；token对应按view变换后的实例像素计算。随机view关系运行时建立，不把原图cell_index改成局部编号；batch局部行号另行映射。删除先修改原图实例像素，再以相同插值生成相关views，避免插值后残留信号。
