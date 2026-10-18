# CellWeft：本地项目初始化 Prompt

以下内容可整体粘贴给本地 coding agent。它是一份执行规格，不要求预先读取聊天记录。初始化任务是搭建可运行、可测试的项目基础，不是自动开展大规模预训练。

---

你是一名负责空间组学与深度学习基础模型的研究工程师。请在当前工作目录初始化项目 **CellWeft**，Python 包名 `cellweft`。

英文研究题名：
**CellWeft: A Cell-Centric, Knowledge-Informed Spatial Proteomics Foundation Model for Multiscale In Silico Perturbation**。

你的任务是：检查当前环境和已有文件，建立研究规格、配置、代码结构、数据合同、模型接口、最小可运行实现及测试。不要只再写一份概念方案；不要自动下载 TB 级数据或启动完整预训练。对缺失的外部依赖和权重明确报告，不伪装成已经完成集成或训练。

## A. 操作边界与决策优先级

1. 先检查当前目录、Git 状态、已有 README/AGENTS/配置以及 Python/PyTorch/CUDA/GPU。若已有工程，增量修改，不覆盖已有研究代码，不删除用户文件。
2. 当前目录是授权工作范围；不要改写全局环境，不安装 system-level 软件，不使用 sudo。
3. 可核验公开代码和文献，并在本项目下建立锁定版本的外部依赖；不要自动下载模型大权重或数据集。没有网络时保留显式 offline 状态。
4. 不上传私有医院数据、路径中敏感信息、患者标识或 embeddings；tracker 默认 offline/disabled。
5. 下面“已经锁定”内容高于你根据个人偏好提出的架构优化。不能悄悄换成 RGB ViT、DINOv2、ViT-L、纯 cell 模型、MAE-only neighborhood 或无 fusion 模型。
6. 未给定的超参数可以选择合理的 smoke/bootstrap 初值，写进 `docs/DECISIONS.md` 的“provisional”栏目，不包装成用户已确认。
7. 不调用自动任务，不承诺后台完成。实际完成什么、运行什么测试、哪些依赖未准备好，都在本轮结果里准确报告。
8. README/研究文档用中文，标识符和代码用英文；保持类型标注、合理错误信息和适量注释。

## B. 科学目标与范围

V1 使用 CODEX/IMC/MIBI 等公开 multiplex image 数据，结合固定 protein knowledge embeddings，训练：

`Marker → Intrinsic Cell → Neighborhood ↔ Image Region → ROI/Tissue`

Cell 是基本可索引、可编辑实体。Image branch 保留完整组织图像，包括细胞内和胞外内容。两个分支通过双向递归交互学习多尺度表示，原生支持 marker/cell/niche 的同步删除和重新 forward。

三个科学能力层是：

- Protein knowledge pretraining：PLM + regulatory/drug–protein knowledge，离线生成固定身份/功能向量。
- Spatial FM：当前实现主体。
- Drug/perturbation extension：V2 预留接口，当前不训练。

优化流程只有两个大阶段：

- **Pretraining**：独立训练 image/cell 两个基座，再 joint pretraining。
- **Downstream fine-tuning**：私有医院适配、任务头、未来非空间单细胞扰动数据接入。

不要把 CellOT、CITE-seq、药物浓度或 drug→expression predictor 塞进 V1。

## C. 已锁定的架构与训练决策

### C1. Protein embeddings

- `e_p = Fuse(e_p_plm, e_p_regulatory)`。
- 不依赖具体 slide 的表达量、治疗信息或细胞位置。
- 可以评估 MAP 等已有知识 encoder，或使用 ESM + SIGNOR/OmniPath 的融合；provider 需可插拔。
- V1 加载版本化的 fixed embedding bank；没有明确配置就不更新该表。
- Protein embedding 的维数由资产 manifest 决定，不默认固定 1,280。
- 严格区分 canonical protein 和 marker measured form，例如 total/phospho、epitope、clone。DNA/DAPI/isotype 使用明确 special-marker metadata，不伪造蛋白 ID。

### C2. Image foundation encoder

- 使用 **DINOv3 官方训练工程**。
- Backbone 为 **ViT-B/16**，不是 L。
- Global region 输入为 `256×256×D`，tensor 使用 `B,D,H,W`。
- 内部 patch 为 `16×16`；global view 有 256 个空间位置。
- 使用 protein-aware multichannel PatchEmbed，最终每个空间位置输出一个 token。
- 不能把 DINOv3 主干输入改成未经融合的 `256×D` 超长序列。
- 不套用 RGB 3-channel PatchEmbed；不压成伪 RGB。
- 不主动减少 local views、prototypes、head hidden dimensions 或其他已选择的官方配方。
- 官方默认配置需要锁定 commit，明确记录 delta。2026-10-17 基础 YAML 显示 local views=8、local size=96、两 heads 各 prototypes=65536、hidden=2048、bottleneck=256；global default=224，由本项目明确覆盖为256；arch 由 L 覆盖为 B。
- 维持原生 DINO/iBOT、EMA、调度和可选 Gram 机制。Gram 是否启用由锁定 recipe 决定；不将“使用完整工程”扩展成必须训练官方 7B teacher、权重蒸馏或所有高分辨率阶段。

### C3. Cell foundation encoder

完整细胞基座包含两个模块，二者都参与独立预训练：

1. `IntrinsicCellEncoder`：只用细胞自身 marker values + fixed marker embeddings。
2. `NeighborhoodEncoder`：使用 intrinsic cell tokens、kNN indices、TERRA-inspired cell-rank。

Losses：

- `masked_marker_loss`：恢复被遮盖 marker 的实际测量值。
- `masked_cell_latent_loss`：可见 cells 的 student context → 预测完整视图 EMA teacher 的目标 contextual cell embeddings。

不要把第二项换成恢复整个细胞表达谱，也不要重新提议一个 BERT/MAE-only 空间训练框架。

CellWorld 提供 latent prediction 的参考；TERRA 提供邻域组织/rank 思路。不是复制它们全部 tokenizer、metadata 输入、原始表达分布或精确采样参数。

### C4. Fusion and alignment

- 保留 cell/neighborhood tokens 与 dense image tokens 的双向递归 cross-attention。
- 原生 embedding dimensions 可以不同，在 query/key/value 投影时映射。
- 只采用一项 **ST-Align-inspired Neighborhood–Region 欧氏对齐**。
- 不加入 HyperST/hyperbolic/entailment/hierarchy regularization。
- Tissue/ROI 初期 pooling/MIL，不建全切片巨型 Transformer。
- Fusion round count、是否共享参数是独立配置项。此次资源调整没有授权默默降低轮数或改成共享权重。

### C5. Joint pretraining freeze policy

| Module | Initial joint policy |
|---|---|
| Protein embedding bank | frozen |
| Protein-aware Image PatchEmbed | frozen |
| Image ViT-B | frozen; optional unfreeze last blocks |
| IntrinsicCellEncoder | frozen; optional unfreeze last blocks |
| NeighborhoodEncoder | frozen; optional unfreeze last blocks |
| Fusion | trainable |
| Masked-cell predictor | trainable |
| Alignment/projection heads | trainable |

联合阶段不再次每步运行完整 DINO multi-crop pretraining。使用与 cells 空间对应的 crop 和 masked/full views，为 Fusion 提供已训练的基座特征。

冻结不代表复用删除前的 embeddings。输入改变后始终重新 forward。

## D. 初始化前核验的参考与代码

建立 `docs/SOURCES.md` 和机器可读的 `configs/sources.yaml`，记录 URL、访问日期、role、commit/hash、许可、核验状态。

关键入口：

- DINOv3: https://github.com/facebookresearch/dinov3
- DINOv3 paper: https://arxiv.org/abs/2508.10104
- DINOv3 config: https://raw.githubusercontent.com/facebookresearch/dinov3/main/dinov3/configs/ssl_default_config.yaml
- VirTues: https://github.com/bunnelab/virtues
- KRONOS: https://github.com/mahmoodlab/KRONOS
- TERRA: https://github.com/Lotfollahi-lab/terra
- CellWorld paper: https://arxiv.org/abs/2608.06659
- CellWorld code: https://github.com/UoM-HealthAI/CellWorld
- ST-Align: https://arxiv.org/abs/2411.16793
- 用户提供的 ST-Align 版本：https://openreview.net/forum?id=EWzPy228cM
- HyperST（背景，不进入主模型）：https://arxiv.org/abs/2511.22107
- MAP: https://www.nature.com/articles/s42256-026-01286-w
- MAP code: https://github.com/MAGIC-AI4Med/MAP
- MAP-KG: https://huggingface.co/datasets/RainGate/MAP-KG
- SIGNOR: https://signor.uniroma2.it/
- OmniPath: https://omnipathdb.org/
- spora: https://spora.bioimagearchive.org/projects/spora
- spora docs: https://spora.epfl.ch/docs/
- HTAN: https://humantumoratlas.org/
- HuBMAP: https://portal.hubmapconsortium.org/

论文、代码和权重许可分别记录。不要因为参考了一个方法就复制其受限代码；可独立实现方法并保留引用。不要擅自给整个混合工程授予覆盖上游资产的宽松许可。TERRA 的最终 manuscript/DOI 若尚未核实，只记录官方仓库和已核验元数据。

## E. 目录结构

可根据当前仓库轻微调整，但至少覆盖以下功能：

```text
cellweft/
  README.md
  AGENTS.md
  pyproject.toml
  .gitignore
  configs/
    sources.yaml
    protein_bank.yaml
    data/public.yaml
    data/private.example.yaml
    model/base.yaml
    model/smoke.yaml
    train/image_pretrain.yaml
    train/cell_pretrain.yaml
    train/joint_frozen.yaml
    train/joint_partial_unfreeze.yaml
    train/downstream.yaml
    edits/noop.yaml
    edits/cell_deletion.yaml
  docs/
    PROJECT_SPEC.md
    DECISIONS.md
    ARCHITECTURE.md
    DATA_CONTRACT.md
    TRAINING.md
    DELETION_SEMANTICS.md
    ENGINEERING.md
    SOURCES.md
    THIRD_PARTY_NOTICES.md
    V2_INTERFACE.md
  src/cellweft/
    types.py
    cli.py
    data/
      manifests.py
      marker_registry.py
      protein_bank.py
      datasets.py
      collate.py
      normalization.py
      indices.py
      spatial_transforms.py
      synthetic.py
    models/
      image_patch_embed.py
      dinov3_adapter.py
      marker_encoder.py
      neighborhood_encoder.py
      cell_predictor.py
      fusion.py
      pooling.py
      cellweft.py
    objectives/
      marker_prediction.py
      latent_prediction.py
      region_alignment.py
    training/
      freeze_policy.py
      ema.py
      pretrain_cell.py
      pretrain_joint.py
      downstream.py
      checkpoints.py
    interventions/
      specs.py
      edit_engine.py
      compare.py
    evaluation/
      embeddings.py
      metrics.py
    engineering/
      profiling.py
      reproducibility.py
  scripts/
    inspect_environment.py
    prepare_manifest.py
    build_indices.py
    export_protein_bank.py
    make_synthetic_dataset.py
    smoke_test.py
    profile_step.py
    report_parameters.py
    run_deletion.py
    launch_image_pretrain.sh
    launch_cell_pretrain.sh
    launch_joint.sh
  tests/
    test_patch_tokens.py
    test_channel_permutation.py
    test_marker_missingness.py
    test_indices_and_transforms.py
    test_masked_cell_context.py
    test_dual_view_deletion.py
    test_gradient_routes.py
    test_freeze_and_ema.py
    test_checkpoint_resume.py
    test_region_alignment.py
    test_upstream_integration.py
  third_party/README.md
  data/README.md
  outputs/.gitkeep
```

`data/` 不提交真实数据；`outputs/` 不提交 checkpoints 或患者级结果。上游代码采用明确的 submodule/vendor/外部依赖策略并锁定 SHA，不通过 import 自动下载。

## F. 数据合同与类型

优先使用 dataclasses 或同类 typed schemas。至少定义以下结构。

### F1. SlideRecord

包含 cohort、platform、pseudonymous patient key、slide_id、ROI、图像路径、mask 路径、marker 表、pixel_size_x/y、H/W、channel 数、licensing/access、split、source DOI/accession、normalization_version、checksum。

### F2. MarkerRecord

包含 marker_id、display_name、canonical protein ID、gene symbol（可空）、measured_form、phosphosite/epitope（可空）、clone（可空）、channel role、protein-bank row、mapping confidence、special-marker flag。

不能因 total/phospho 对应同一 canonical protein 就合并 marker identity。不同 clone 可以共享基础 protein vector，但 measurement metadata 不丢弃。

### F3. CellRecord / CellTable

- 全局稳定 `cell_id` 与 `mask_label` 分开。
- slide_id、centroid_x/y_pixels、centroid_x/y_um、area、bbox。
- mask 表示 nuclear/whole-cell，记录 segmentation source/version。
- marker measurements、assayed mask。
- annotation 只作可选 evaluation 字段，不默认进入无监督模型输入。

### F4. CropRecord

- crop_id、source slide/ROI、坐标范围、native physical FOV、resampling policy、transform ledger。
- 256 global input view 对应原图范围。
- visible cell IDs、context/halo IDs、boundary flags。
- image-token grid / cell footprint 对应。
- normalization snapshot 和 source checksum。

### F5. 索引

- cell_id → expression row / mask label / coordinates。
- cell_id → baseline kNN candidates / distances / ranks。
- cell_id → reverse-neighbor references。
- crop_id → constituent cells。
- cell_id → overlapping crops / image-patch footprints。

使用 cKDTree、KDTree 或等价空间索引，不计算持久化的 N×N 矩阵。各 slide 独立构建，不跨患者。

### F6. Batch

至少覆盖：

- `images: FloatTensor[B,D_max,H,W]`
- `marker_embeddings: FloatTensor[B,D_max,R_p]` 或共享表＋indices
- `marker_valid: BoolTensor[B,D_max]`
- `cell_values: FloatTensor[B,N_max,D_max]`
- `cell_marker_assayed / visible: BoolTensor[B,N_max,D_max]`
- `cell_ids, cell_valid, coordinates`
- `neighbor_indices, neighbor_ranks, neighbor_valid`
- `instance_masks` 或按需要加载的原图 label maps
- `crop_ids, transform_records, patient/split provenance`

marker 未测量、marker 测量为 0、padding、train mask 和 deletion 是不同状态，不能用同一个数值替代全部语义。

动态 batch 按 cell 数分桶。图像 D 也可不同；mask padding 不参与 marker fusion、attention 或 losses。零细胞 crop 可用于 image-only 阶段；joint 有明确 skip/empty handling。

## G. Protein-aware 图像 tokenization 的精确要求

对每个 `16×16` 空间位置 s、marker p：

1. 读取该 marker 的 256 个像素，经过共享线性投影。
2. 加对应固定 protein knowledge embedding 的投影。
3. 在同一空间位置内进行 marker attention/fusion，使用有效 marker mask。
4. 用 learnable summary query/readout 得到一个空间 token，输出 width 768。
5. 交给官方 DINOv3；空间位置在原生 RoPE 中处理。

重要实现约束：

- 不在 iBOT masking 之前进行跨空间信息传播。
- 不使用任意 channel-index positional embedding；同一 marker-image 成对重排应保持最终空间 token 不变（数值容差内）。
- 不在 `forward()` 中把某个 batch 的 marker metadata 保存为全局 mutable 状态；必须显式贯穿 student/teacher、global/local view 调用。
- 检查官方 patch_embed 的实际输出 layout 和 attributes（例如 grid、patch size、flatten 路径），写 adapter 而非假设接口。
- 96×96 local view 对应 6×6=36 个空间位置；global 256 对应256。额外 CLS/register tokens 不算入空间位置数量。
- 单个 marker dropout/deletion 需通过输入有效性处理，不把 D 固定到40。
- 如果外部 DINOv3 不可用，允许 tiny `MockImageEncoder` 仅用于标注为 smoke/test 的配置；正式配置必须明确报依赖缺失，不得静默降级成 mock。

## H. 细胞与邻域编码的精确要求

### H1. IntrinsicCellEncoder

- 对每个 cell 的 available markers 构造 identity+value tokens。
- attention 汇总为 intrinsic cell token，同时可导出 marker tokens。
- 默认不输入 patient ID、study ID 或 supervised cell type。
- intrinsic 只依赖自身 measurements；neighbor edit 不得改变其输入。
- masked marker prediction 保留 marker identity，只遮盖 value；重建 loss 只针对 actual assayed targets。

### H2. NeighborhoodEncoder

- 输入 index cell + 有序 kNN cell tokens，rank0 为 index。
- `k` 配置化，不要求所有 neighborhoods 满k；不能为补齐复制最后一个真实细胞。
- cell ID 只用于查找，不学习 ID embedding。
- 表达相同而 neighbors 不同的 index cells 必须通过不同 neighborhood context 前向。
- 同一个 forward 内先对 unique cells 编码，再 index gather，避免重复 intrinsic computation。

### H3. Cell latent pretraining

- 先确定 visible/target cell sets，再编码 student；目标细胞在所有 neighborhood、overlapping context、聚合统计路径中移除。
- Teacher 读取完整细胞上下文，stop-gradient，并按 EMA 更新。
- Predictor 接收 visible context、target location/rank query 和可选有限 hints。
- hints 只进 predictor；不得把完整 target profile 或 teacher latent 作为 student 输入。
- 可用 normalized MSE/cosine 的一个配置化目标；实现一个默认，不同时堆多个损失。
- 监测表示均值、std、norm、teacher/student 方差；禁止通过最小化所有 embeddings 为零来“通过”测试。

## I. Fusion 与联合训练

### I1. 数据输入

联合阶段读取一个空间一致的 crop 和 cell neighborhood。Student 使用同步 masked views，teacher 使用完整视图。DINO 独立训练的8个local crops不在每个 joint step 重复运行。

### I2. Frozen forward

- image/cell/neighborhood 基座固定并在适当 eval 状态 forward。
- `requires_grad=False` 与 `eval()` 分别管理。
- 基座前端可用 `torch.no_grad()`，输出在离开该上下文后进入可训练 Fusion。不要默认用 `inference_mode()` 生成会被后续可训练层保存用于 backward 的张量；如使用，需正确转成普通 tensor 并测试。
- 若配置解冻末端，必须保留对应 autograd 路径，优化器仅接收相应参数。
- 若可训练 latent 需要穿过冻结 block，不能断开对其输入的梯度；当前默认递归只在 Fusion 内完成。

### I3. Fusion architecture

- 两个 directions：cell/neighborhood→image 和 image→cell/neighborhood。
- 每轮使用更新后的 tokens；保留 residual、norm 和投影。
- 冻结基座 feature 仅作为这一轮的初始条件，不在每轮强制覆盖回旧 embedding。
- Predictor 可读取融合后的两类 context；必须保证两个 attention directions 都获得梯度。
- tissue 采用明确 region pooling/MIL。niche embedding 可以是 cell-centered summary；不要不加权地将高度重叠 neighborhoods 当成独立面积相加。

### I4. Teacher / target

- 冻结基座在 student/teacher 之间共享相同权重即可，但对两份不同输入分别计算。
- 可训练 Fusion 使用 EMA target counterpart；少量解冻后将被解冻模块纳入 EMA 管理。
- target latent 来自完整视图中对应 cell，停止梯度。
- 不需要第二套 giant VirTues teacher，也不要在 joint 阶段持有未使用的 DINO prototype heads/optimizer states。

### I5. Loss routes

- `L_joint_cell_latent`：fusion 后 predictor 与 full-view target 匹配，负责训练 Fusion。
- `L_neighborhood_region_align`：一个欧氏、multi-positive 对比目标，作用于对应小邻域和大图像区域的 projection。
- 当两基座固定且 alignment 位于 fusion 前时，它只训练 projection heads；文档、参数梯度报告不能说成它独自训练 Fusion。
- 不增加另一个 hyperbolic/hierarchy objective。

## J. Masking 与 Deletion Engine

实现不可变的 `EditSpec`，至少包括：

`edit_id, mode, target_cell_ids, target_marker_ids, scope, fill_policy, neighbor_policy, seed, baseline_manifest_hash`。

两种 mode：

- `train_mask`：保留 teacher targets 和 predictor queries。
- `inference_delete`：不创建被删 cell 的恢复 queries，不运行 predictor补回。

### J1. Cell edit

由唯一的 cell ID 集合原子性地修改：

1. Cell token set 和相关 marker rows 的可见性。
2. 所有 neighborhoods 中的引用，包括 reverse index 和 crop halo。
3. 所有重叠 image crops 中相同实例的全部 marker pixels。
4. cell–image footprint/membership 表。

不得只删除最终 pooled cell embedding。不得保留 image branch 的旧特征。

图像 edit 使用精确 instance mask，而不是整个16×16 image token。fill value 来自原始/标准化背景规范；固定 baseline normalization，不编辑后重新归一化剩余像素。

默认不补入更远 kNN；配置明确记录 rank policy。建议默认按剩余近邻距离连续重排，并保存 original rank 便于分析。原始坐标和 surviving-cell expression 不变。

### J2. Marker edit

全局 marker deletion 同步禁用细胞侧 marker 输入和图像侧通道。局部 edit 同步对应 cell measurement visibility 和同一实例/通道像素。不能把 token 缺失简单混同为真实观测零值。

### J3. Niche edit

先解析 neighborhood/region selection 得到一组 cell IDs，去重，再调用同一 cell edit。不能只删 niche summary token。

### J4. 比较输出

同一参数、相同推理配置：baseline full forward → edited full forward → ID 对齐比较。

输出 intrinsic cell、neighbor contextual、fused cell、region、ROI embeddings 与 deltas；保存 edit spec、checkpoint hash、normalization hash、crop transform、剩余 IDs。no-op 必须接近零变化；被删除 cells 不出现在 surviving-cell 输出中。

所有随机 augmentation/dropout 在此关闭。初始版本做完整 forward，不先优化成局部更新。

## K. 数据划分与研究评价

- patient-level split 优先于 cell/crop split。纵向同一患者所有时间点在同一组。保留原始 treatment/timepoint metadata；不使用药物条件不等于把已治疗观测伪装成治疗前。
- Public studies 跨来源去重，spora/HTAN/原始研究不能重复泄漏到 train/test。
- 至少一整家私有医院预留外部验证；不默认把九家医院加入预训练。
- 无标签公开图像可进入 image pretraining；没有 segmentation 不伪造 cell targets。
- 初始化只生成 manifests/模板，不下载真实大数据。
- 实现 baseline指标接口：marker prediction、cell annotation、region retrieval、niche/domain annotation、ROI tasks、deletion embedding comparison。
- deletion controls 包括 no-op、count/area-matched random removals、single-branch edit diagnostic；研究结果不强制所有 delta 越大越好。
- 消融复用两基座 checkpoints，不默认从头重训每个组合。

## L. 工程与预算约束

资源目标：一台持续可用的8卡节点，单卡 H800 80GB 或 RTX PRO 6000 96GB。初始化优先 CPU/single-GPU smoke tests，不启动8卡长任务。

预算数据：15,000 images；每张1000×1000、约40 markers、500–2000 cells。约750万–3000万 cells，平均约33–131 cells/global crop。真实 batch 使用实际 cell 数，不能硬编码512 cells/crop。

必须实现：

- BF16 configurable；CPU smoke用float32。
- tensor masks、动态cell/marker数量和bucketed batches。
- unique-cell encoding + index gather。
- efficient attention、checkpointing 选项。
- stage-aware optimizer、EMA、参数统计。
- checkpoint恢复包含 optimizer、EMA、scheduler、RNG、sampler/config/source hashes。
- 缓存key包含模型版本、输入hash、normalization、edit、geometry transform；禁止编辑后命中原始特征cache。
- profile image/cell/joint 三种 step，分别汇报计算和data wait。
- gradient accumulation batch 与 contrastive实际all-gather batch分开记录。
- 保存 nominal参数量、可训练参数、frozen和EMA参数，不编造GPU-hours或benchmark。

DINO默认大heads保留，仅在明确 `smoke` 配置可减小测试尺寸；测试配置不得成为正式训练默认值。

## M. 初始可配置数值：不是额外锁定的科学决策

为便于实例化，可以使用以下 provisional values，并全部写入配置：

- image marker fusion：width192、2 blocks、6 heads，输出768。
- IntrinsicCellEncoder：width256、6 blocks、8 heads。
- NeighborhoodEncoder：width384、6 blocks、6 heads；k32。
- Fusion：width384、3 rounds、unshared weights作为bootstrap；不得把它描述成唯一已确认值。
- predictor：较小的Transformer，维数投影到target维度。
- marker mask ratio、cell target ratio、hint ratio配置化；初始可用0.3/0.5/0.1作smoke候选，明确不是已确认正式超参数。
- 主训练的DINO字段由upstream recipe继承，不在这里重设。

如本地尺寸不满足heads整除等条件，先修正配置并记入DECISIONS，而非静默改变模型。

## N. 必须通过的测试

### N1. Shape / panel

1. 输入 D=20/40/60 的global crop均输出256个spatial tokens，额外特殊tokens单独计数。
2. 96×96 local view输出36个spatial positions，不写死global长度。
3. 相同(image, marker identity)成对重排后，PatchEmbed在eval下保持一致。
4. padding marker和未测量marker不影响合法输出；真实零值不是missing。
5. DNA/DAPI等special-marker可运行，无虚构protein映射。

### N2. 索引与空间

6. cell ID与row index分离，重排行顺序后邻域含义不变。
7. kNN不跨slide；不足k和零邻居有合法行为。
8. crop旋转/翻转后，image、mask、coordinate和footprint严格一致。
9. 重叠crops和halo共享同一个global edit set。

### N3. Masking / deletion

10. no-op edit前后输出一致。
11. 删除cell从所有student neighborhood输入中消失。
12. 被删cell全部通道instance pixels被处理，mask外像素保持不变。
13. 不删除整个16×16块，不改动同块其他cell的像素。
14. 全局marker edit同时影响两分支。
15. niche deletion等于相同去重cell集合的cell deletion。
16. surviving-cell intrinsic embedding不因neighbor deletion变化。
17. 删除目标后不从baseline cache读取其旧image/contextual feature。
18. train masking中student不读取目标原始内容；teacher targets仍然完整。
19. inference_delete不执行target recovery predictor，也不重新插入被删cell。
20. 归一化参数不随edit重新估计。

### N4. 训练和梯度

21. 独立cell pretraining使Intrinsic和Neighborhood参数都得到对应loss的梯度。
22. joint frozen模式下，protein/image/cell/neighborhood权重不变；Fusion/predictor在有效batch上得到梯度并更新。
23. 两个方向的cross-attention都存在通向loss的梯度。
24. teacher无梯度；EMA按配置只更新目标模块。
25. partial-unfreeze只更新指定末端层，不能因错误no_grad而失效。
26. hints只进入predictor，不进入student encoder。
27. 冻结eval状态不会被顶层train()意外覆盖。
28. checkpoint恢复后固定seed的一步结果可复现；save/load不会改变edit semantics。
29. multi-positive alignment在单positive/多positive、padding、无合法negative等情况下不会NaN。
30. latent variance/norm与每个loss统计可记录；不把表示塌缩后的零loss当作成功训练。

### N5. 环境与来源

31. 正式配置必须确认调用官方DINOv3 adapter，不静默使用mock。
32. 缺少upstream依赖的测试清楚标记skip reason，不谎报integration passed。
33. 正式配置的local views/prototypes/head sizes与锁定source一致；仅批准override存在。
34. mock embedding bank明确标synthetic，不用于任何正式生物结果。
35. CLI不会默认启动TB下载、上传tracker或长训练。

## O. 初始化的实施顺序与验收产物

按顺序执行，不先追求全量训练：

1. 环境检查与已有目录保护；记录当前Git状态。
2. 创建研究规格、来源、决策表、数据合同和配置。
3. 构建少量synthetic multiplex images：具有不同marker panels、可复现cell masks、坐标、背景和overlapping crops。
4. 实现protein bank、PatchEmbed、cell/neighborhood最小完整前向、fusion、pooling和edit engine。
5. 接入官方DINOv3。先验证metadata传递与forward contract，再接训练入口；如依赖缺失，文档明确下一条命令和blocker。
6. 实现marker/latent/alignment loss和冻结/EMA管理；运行一至数个synthetic训练step验证梯度。
7. 运行baseline/deleted forward，导出示例embeddings、diff、edit日志。
8. 执行pytest；输出通过/失败/跳过数量及原因。
9. 生成参数报告和可选短profiling脚本，不给出伪造吞吐。
10. 提供8卡torchrun启动模板，但不要自动启动。

CLI命令建议统一成：

- `cellweft validate-data ...`
- `cellweft build-indices ...`
- `cellweft prepare-protein-bank ...`
- `cellweft smoke-test ...`
- `cellweft pretrain-image ...`
- `cellweft pretrain-cell ...`
- `cellweft pretrain-joint ...`
- `cellweft export-embeddings ...`
- `cellweft apply-edit ...`
- `cellweft compare-embeddings ...`
- `cellweft profile ...`

这些命令是需要你实现的目标接口，不要在实际实现前宣称现已存在。

## P. V2 extension contract

只建立protocol和文档，不训练：

`PerturbationAdapter(intrinsic_cell_state, drug_features, assay_context) -> intrinsic_cell_delta`

随后由已训练的spatial modules进行forward。保留同一protein bank和intrinsic cell接口；后续通过实际protein perturbation数据微调，不在V1加入CITE-seq预训练。

可选 `apply_intrinsic_delta` 路径应标为未训练接口，不返回伪造药效。没有trained adapter时，要求显式提供delta或报缺失模型，不用随机向量替代。

## Q. 最终交付报告格式

本轮完成后报告：

- 已创建/修改的文件与当前目录树。
- 哪些核心设计已经按规格实现。
- 已运行的命令与真实测试结果。
- 本地依赖、upstream commit/config、许可核验状态。
- 各模块实际参数数量、freeze policy与梯度验证结果。
- 尚未实现或缺外部数据的部分，不能混入已完成项目。
- 下一次能直接运行的最小命令；不自动启动训练。

目标是建立一个可以连续演化的研究工程，而不是写满空类、TODO和不可运行的shell命令。对external DINOv3集成等未完成的环节，明确边界并让正式路径失败得清楚；不要用一个自写toy ViT冒充已经复用完整DINOv3。

现在请开始检查当前目录和环境，按照上述顺序初始化 CellWeft。
