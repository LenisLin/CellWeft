# Methods

## 跨研究 panel 整合与 marker 向量

我们从各研究的 panel 文件中收集原始 marker 名称，并整合为统一的跨研究 marker 集合。原始名称、统一名称、对应 gene symbol 及 UniProt accession 列于 Supplementary Table S1。对于基因 \(g\)，采用 MAP 官方提供的 ESM-2 表示 \(\mathbf{x}_g\) 和预训练 MAP-KG 蛋白编码器 \(f_{\mathrm{MAP}}\)，获得固定向量：

\[
\mathbf{e}_g=f_{\mathrm{MAP}}(\mathbf{x}_g)\in\mathbb{R}^{1024}.
\]

对于 marker \(m\)，令 \(G_m\) 为 Supplementary Table S1 中列出的对应基因集合，其向量定义为：

\[
\mathbf{z}_m=\frac{1}{|G_m|}\sum_{g\in G_m}\mathbf{e}_g
\in\mathbb{R}^{1024}.
\]

单基因 marker 使用相应基因向量，家族和多靶标组合使用所列成员向量的均值。图像 tokenizer 和细胞表达头使用固定 marker 向量。目标 marker 条件的像素读出接口见[模型规范§7](../architecture/model.md#7-图像-marker-重建读出)。

## 图像编码、实例读出及任务读出（拟实施）

完整 image 预先分割并提取 whole-cell marker 均值，按实例 mask 与 centroid 对齐细胞目标。蛋白图像输入及表达目标从有效蛋白通道读取。基础切窗和同步几何见[数据准备](../data/preparation.md#7-crop-与训练视图)及[数据合同](../architecture/data_contract.md)。

固定 MAP marker 向量为 protein-aware tokenizer 提供蛋白身份。单一 ViT-B/8 输出空间 patch 表征与 crop CLS，主干后的 Mask-Adapter 式读出提供细胞表征。以 DINO、iBOT、KoLeo 和 whole-cell MSE 联合预训练，细胞监督来自未施加 iBOT 遮盖的 student global view。目标是检验预测和离线定量的一致性。结构与训练见[模型规范](../architecture/model.md)及[训练规范](../architecture/training.md)。

基座预训练后冻结配套 EMA。Figure 2／3 共用 marker 条件化像素 decoder，细胞注释以冻结细胞表征拟合分类器，图像及患者预测以 crop CLS bag 训练 MIL。任务和评分见[表征评价](../evaluation/representation.md)。
