# 工程实施与验收

本文维护资源预算、集成步骤和工程验收；模型与训练的科学约束见[模型](../architecture/model.md)、[训练](../architecture/training.md)，协作边界见[AGENTS](../../AGENTS.md)。

2026-10-19 图像基座按[模型规范](../architecture/model.md)开头的最新决定执行；本页旧 DINOv3 集成、256 crop 面积预算及 96 local 验收待独立迁移重核。图像准备与索引以[数据准备](../data/preparation.md)、[数据合同](../architecture/data_contract.md)为准：旧 Special markers 输入要求由三去向规则取代，近邻范围收紧到独立 image_id。基础crop已固定为原生224×224、步长112；面积预算按实际µm/px计算。

## 1. 预算与实现策略

预算假设为 15,000 张 1,000×1,000 图像、平均 40 markers、每图 500–2,000 cells：约 750 万–3,000 万 cells。
256×256 crop 面积占比 6.5536%，面积平均约 33–131 cells，不是密集区上限；约 22.9 万面积等价 crops，重叠采样不增加独立患者数。
约 6,000 亿强度值，未压缩 16-bit 约 1.2 TB、float32 约 2.4 TB。实际 manifest 和 batch 使用真实尺寸、panel 与 cell 数，不硬编码 512 cells/crop。
资源目标为持续可用的一台八卡 80GB/96GB 节点（原包候选 H800/RTX PRO 6000）；初始化先 CPU/单卡 smoke。图像可用全节点，cell/joint/下游轮转或分卡。
主模型约 1.1–1.3 亿参数、默认 DINO/iBOT heads 额外约 4,600 万，均为待实例化核对的估算。
分别统计 inference backbone、active trainable、frozen、EMA、optimizer states 与 activation peak；冻结参数仍属于模型总量。
BF16 可配置，CPU smoke 用 float32；使用动态 masks/bucketing、unique-cell gather、crop-wise I/O、局部 kNN、efficient attention 和 activation checkpointing，按阶段选择 FSDP/DDP。
真实 profiler 决定 microbatch；分别记录 gradient accumulation batch 与对比/Sinkhorn 每次 forward 的有效 batch。
在 synthetic 和小型公开 manifest 上运行 50–200 steps，分别统计 image/cell/joint 的 samples/s、cells/s、tokens/s、data wait、GPU peak memory、参数与训练比例；不预设 GPU-days。
Checkpoint 恢复包含 optimizer、EMA、scheduler、RNG、sampler、config/source hashes。Cache key 包含模型版本、输入 hash、normalization、edit 和 geometry transform；编辑后不得命中旧 contextual/image 特征。

## 2. 来源与集成

参考入口统一见[references.md](../references.md)。实际集成时建立 configs/sources.yaml，记录 URL、访问日期、role、commit/hash、许可和核验状态；本次整理不生成虚构锁定值。
论文、代码、权重和数据分别记录许可。main URL 不替代固定版本，import 不自动下载；缺依赖报告 offline/缺失状态。
第三方代码放置遵循[架构](../architecture/overview.md)和[存储](../architecture/storage.md)，原 Prompt 的旧文件名和 outputs/.gitkeep 不再作为创建要求。

## 3. 实施顺序

1. 检查 Git、已有文件、Python/PyTorch/CUDA/GPU；确认环境事实和已有工作。
2. 明确研究/来源/数据合同及配置；确认正式值与 provisional 候选分开。
3. 构建不同 panels、masks、坐标、背景与重叠 crops 的 synthetic fixtures；确认几何对应。
4. 实现 bank、PatchEmbed、Intrinsic/Neighborhood、Fusion、pooling 和 edit engine 的最小完整前向；确认公共 IO。
5. 接入官方 DINOv3，先验 metadata 和 forward 再接训练；缺依赖时报告阻塞及可执行的下一步。
6. 实现 marker/latent/alignment 目标、冻结与 EMA，运行少量 synthetic steps；确认梯度路径。
7. 运行 baseline/deleted forward，导出表征、差异和编辑记录；确认双视图一致及 ID 对齐。
8. 运行对应 pytest，报告通过/失败/跳过的数量及原因；不把 skip 记作集成成功。
9. 生成实际参数报告和短 profiler；仅报告实际测得的吞吐。
10. 提供八卡 torchrun 模板；初始化不自动启动长训练。

## 4. 工程验收清单

以下保留原 Prompt 的 N1–N5 / 1–35 对应编号，属于后续验收要求，不是已通过结果。

1. D=20/40/60 global 输入均输出 256 spatial tokens，特殊 tokens 单列。
2. 96×96 local 输出 36 spatial positions。
3. Marker-image 成对重排在 eval 下保持一致。
4. Padding/未测量 marker 不影响合法输出，真实零值不作 missing。
5. Special markers 可运行，无虚构 protein 映射。
6. Cell ID 与行号分离，重排行不改变邻域含义。
7. kNN 不跨 slide，不足 k 和零邻居行为明确。
8. 裁剪/旋转/翻转后 image、mask、coordinate、footprint 一致。
9. Overlap crops 和 halo 共享 global edit set。
10. No-op 输出一致。
11. 删除目标从所有 student neighborhoods 消失。
12. 全通道实例像素被处理，实例外像素不变。
13. 不删除整 patch，不改变同 patch 其他细胞。
14. 全局 marker edit 同步两分支。
15. Niche deletion 等于同一去重 cell 集合 deletion。
16. Neighbor deletion 不改变 surviving intrinsic。
17. 编辑后不读旧 contextual/image cache。
18. Train masking 不泄漏目标内容，teacher targets 完整。
19. Inference deletion 不恢复或重插目标 cells。
20. 编辑不重估 normalization。
21. Cell pretraining 为 Intrinsic 和 Neighborhood 提供对应梯度。
22. Frozen joint 中基座不变，Fusion/predictor 得到梯度并更新。
23. 双方向 cross-attention 均有通向 loss 的梯度。
24. Teacher 无梯度，EMA 仅更新指定模块。
25. Partial unfreeze 仅更新指定末端，autograd 连通。
26. Hints 仅进入 predictor。
27. 顶层 train() 不意外改变冻结 eval 状态。
28. 固定 seed 恢复一步可复现，save/load 保持 edit semantics。
29. Alignment 在单/多 positive、padding、无合法 negatives 时不产生 NaN。
30. 可记录 latent variance/norm 和每项 loss 的统计，不将塌缩零 loss 视为成功。
31. 正式配置实际调用官方 DINOv3 adapter。
32. 缺 upstream 的集成检查明确 skip reason。
33. 正式 local views/prototypes/head sizes 与锁定来源和批准覆盖一致。
34. Mock bank 标记 synthetic，不用于生物结论。
35. CLI 不默认启动 TB 下载、tracker 上传或长训练。

## 5. 待实现 CLI 与交付

目标命令：cellweft validate-data、build-indices、prepare-protein-bank、smoke-test、pretrain-image、pretrain-cell、pretrain-joint、export-embeddings、apply-edit、compare-embeddings、profile。这些是接口规划，尚未实现。
交付报告列出实际创建/修改文件与当前目录树、已实现设计、命令与测试、依赖/upstream/config/许可状态、实际参数/冻结/梯度证据、未完成项和下一条可执行命令。未测得项明确说明，不用空类或 toy 替代完成声明。
