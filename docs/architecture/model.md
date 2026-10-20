# 模型结构与表示接口

本文维护模型结构，训练目标见[training.md](training.md)，输入字段见[data_contract.md](data_contract.md)，编辑见[editing.md](editing.md)。以下为待实现规格。

2026-10-19 更新：当前图像基座采用官方 RGB 预训练 DINOv2 ViT-B/14，后续接入 Protein-aware PatchEmbed；patch=14×14，global/local 网络视图输出为 224×224/98×98。下文及相关旧文中的 DINOv3、从头初始化、16×16 patch、256/96 输入和 RoPE 规格已被此决定取代，完整结构与训练迁移另行执行。图像准备按[数据准备](../data/preparation.md)，决定依据见[决策记录](../project/decisions.md)；基础crop已确认直接取原生224×224像素、常规步长112（50%重叠），不预先统一物理尺度。

## 1. 固定蛋白知识表示

当前用户指定 MAP 为 provider：e_p = MAP_gene_projector(ESM2(p))，使用官方知识训练后的蛋白分支导出固定表示；同一分子身份跨样本共享，不依赖当前细胞测量、空间位置或治疗标签。
Provider 保持可插拔。MAP 分支输入 5,120 维、输出 1,024 维，具体资产维度由 manifest 读取；当前库以 gene symbol 为键，来源marker身份在数据接入时清理；bank编码在全部数据准备完成后统一处理、模型训练前完成。来源、覆盖与部署状态见[资源部署](../development/resources.md)，其他知识来源见[参考索引](../references.md) R09–R11。
ProteinEmbeddingBank 提供 protein_id/marker_id → vector；R_p 从资产 manifest 读取，不写死为 1,280。V1 固定导出表，只训练各分支 projection 和 measurement encoders。
保存来源、模型版本、KG snapshot、ID mapping 与 embedding checksum。Canonical protein 与 measured form 分离，total/phospho、位点、epitope、clone 信息保留；当前下游仅接入经 Manual marker mapping 确认保留的实际蛋白通道，核染等通道的用途见[数据准备](../data/preparation.md)。
ESM-only 可作透明基线或知识缺失对照；随机 bank 仅用于标明 synthetic 的测试。关系 score 表示证据置信度，不解释为药理幅度；重复证据和许可分别记录。

## 2. Protein-aware 图像基座

使用官方 DINOv3 ViT-B/16，global 输入 B,D,256,256，D 可变。每位置融合 marker 后输出一个 width=768 的空间 token：
[B,D,256,256] → [B,256,D,d_marker_image] → [B,256,768]。

1. 对每个 marker 的 16×16 像素使用共享线性投影。
2. 加固定 protein embedding 的可训练投影。
3. 位置内 marker attention/fusion 使用 valid-marker mask。
4. Learnable summary query/readout 输出一个 spatial token，再进入 DINOv3。
5. 空间位置沿用原生 RoPE；iBOT masking 前不进行跨空间传播。

图像像素本身提供测量，不另拼 cell mean；不采用伪 RGB 或把 256×D tokens 直接送入主干。
不使用 channel-index positional embedding；marker-image 成对重排后输出在 eval 容差内一致。Padding/未测量 marker 不参加融合。
Global grid 为 16×16；96×96 local grid 为 6×6。Grid 动态计算，CLS/register tokens 单独计数。
Marker metadata 显式贯穿 student/teacher 与 global/local 调用，不保存在 forward 的全局 mutable 状态。接入前检查上游 patch_embed layout、grid、patch size 和 flatten 属性。
MockImageEncoder 仅在显式 smoke/test 配置可用；正式路径缺依赖时报明缺失，不静默降级。

## 3. Intrinsic 与 Neighborhood

IntrinsicCellEncoder 对可用 marker 构造 identity+value tokens，经 attention 输出 per-marker tokens 和 intrinsic c_i。默认不输入 patient/study ID 或 supervised cell type。
Intrinsic 只依赖自身测量；邻居删除不改变保留细胞的 intrinsic 输入和输出。Marker 测量状态定义见数据合同。
NeighborhoodEncoder 输入独立编码的 cell tokens；index cell 为 rank 0，近邻按相对该 index 的距离 rank 组织，使用[数据合同](data_contract.md)规定的排序邻居索引。
输出 index contextual embedding、neighborhood summary，并按需导出 token states。相同表达、不同邻域的细胞使用各自 context。
Cell ID 仅用于索引，不学习实体词表；任意行顺序不是生物顺序。不足 k 时使用有效性标识，不复制最后一个细胞补齐。
同一 forward 先编码 unique cells 再 gather；不把邻居全部 marker 序列送入 NeighborhoodEncoder。

## 4. 递归 Fusion 与 readout

对 cell/neighborhood tokens 和 dense image tokens 进行 Cell → Image update → Image → Cell update → Region readout。
每轮使用上一轮更新状态，保留 residual、norm 和必要的 Q/K/V 投影；基座输出只作初始条件，不每轮覆盖回来。两分支原生维度可不同。
轮数与权重共享方式分别配置；缩减轮数或改为共享参数需要显式决策。Predictor 可读取融合后两类 context。
Region/ROI 使用明确的 coverage/ownership 和 pooling/MIL；重叠 neighborhoods 不重复计权。V1 不要求全切片大型 Transformer。

## 5. Provisional 结构数值

| 模块 | Bootstrap 候选，非正式锁定参数 |
| --- | --- |
| Image marker fusion | width 192，2 blocks，6 heads，输出 768 |
| IntrinsicCellEncoder | width 256，6 blocks，8 heads |
| NeighborhoodEncoder | width 384，6 blocks，6 heads；邻居数量按已确认的数据合同 |
| Fusion | width 384，3 rounds，unshared weights |
| Predictor | 较小 Transformer，投影到 target 维度，具体宽度/层数未定 |

正式数值在配置中记录；维度不满足 heads 整除时显式修订配置和决策，不静默更改模型。

## 6. V2 预留接口

PerturbationAdapter(intrinsic_cell_state, drug_features, assay_context) → intrinsic_cell_delta。
h_cell_edited = h_cell_baseline + Adapter(h_cell_baseline, drug, assay_context)，随后进入 Neighborhood/Fusion/ROI。
暴露无 spatial 输入即可调用的 intrinsic encoder，保留共享 protein bank、latent 注入点和 state-dict version。
未来按 genetic/chemical 实际监督类型接入 CITE-seq ADT、CyTOF、4i；不将 CITE-seq 全部视作小分子药物数据，不另建独立 transcriptome decoder。
V1 只预留 protocol。可选 apply_intrinsic_delta 标为未训练接口；没有 trained adapter 时需显式 delta 或报告缺模型，不生成随机药效。
