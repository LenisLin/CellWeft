# 模型资源与实测

当前模型使用固定 MAP bank，其余可训练组件按模型规范随机初始化。DINOv2 提供自监督训练方法，Mask-Adapter 提供实例读出结构，VirTues 提供蛋白遮盖与像素解码参考。见[模型规范](../architecture/model.md)和[参考索引](../references.md)。

本地 MAP 与 DINOv3 checkout、固定 revision、来源许可和 NAS 相对资源路径由 [sources.yaml](../../configs/sources.yaml) 维护。DINOv3 权重访问仅有 2026-10-18 的申请与未取得权重记录。当前资源目录归属见[存储规范](../architecture/storage.md)。

## MAP 表示的含义与边界

NAS `processed/_resources/protein_banks/map_gene_projector_v1/` 含向量、gene symbol 行键、manifest 和验证记录。官方 5,120 维 ESM2 向量输入知识训练后的 residual/gated `gene_projector`，导出对比损失执行 L2 normalization 前的 19,790 × 1,024 float32 表示。MAP-KG 基因表的 19,528 个唯一 ESM 键均被覆盖，另有 262 个表外键。isoform、磷酸化位点、clone 与特殊标记依实际 panel 映射。见[数据合同](../architecture/data_contract.md)和[数据准备](../data/preparation.md#study级marker向量查表)。

MAP 代码许可为 MIT，MAP-KG 数据许可为 Apache-2.0，已核验发布页未单列权重许可。导出脚本用 Python 3.11+、PyTorch 和 NumPy 2.x 在 CPU 前向计算，接受对应 checkpoint、ESM 表、基因元数据及尚不存在的输出目录。见[导出脚本](../../scripts/resources/export_map_bank.py)。

2026-10-18 验证中，官方 projector 严格加载通过，行键唯一且输出全为有限值。抽取 64 个基因与固定版本官方 `ResidualProjector` 对比，最大绝对误差约 `1.43e-6`（`rtol=1e-5, atol=1e-6`），官方 eval 重复前向完全一致。见库内 `verification.json`。

2026-11-18 已核对 Figure 2 的 12 个全局未见目标均有正式 MAP 键。Bandyopadhyay 和 Piyadasa 的 study 向量均与相应 bank 行逐值相同，为有限、非零的 1,024 维 float32。CD70 使用正式键 `CD70`，`TNFSF7` 保留为别名。该核对覆盖表示的可用性，训练测量覆盖与任务成员见[表征评估](../evaluation/representation.md)。

## 本地工具依赖

项目 Hugging Face 入口使用 `.venv-hf/`，凭据位于 Git 忽略的 SSD `.cache/huggingface/`，模型下载缓存位于 NAS。见[入口](../../scripts/resources/hf.sh)和[依赖](../../configs/huggingface-cli.requirements.txt)。

项目 Cellpose-SAM 入口使用 `.venv-cellpose-sam/`，分割调用显式传入 `CELLPOSE_SAM_MODEL` 指向的本地权重。`cpsam_v2` 为官方 `7c61431b5fbb078f3296754bd15d9f51b320f837` 的 `sam_vitl` checkpoint，大小 1,233,586,851 字节，实测 SHA256 与官方 LFS 标识一致。见[入口](../../scripts/resources/cellpose-sam.sh)和[依赖](../../configs/cellpose-sam.requirements.txt)。

2026-10-19 已在 12 张源图上探索 Cellpose-SAM、InstanSeg 和 StarDist 的预训练推理。具体模型版本、环境、输入及结果见[分割探索记录](../experiments/segmentation_pilot_20261019.md)。

## 远端预处理环境（2026-10-22）

五节点该批项目服务于 2026-11-10 退役。验证环境为 Python 3.12.15、PyTorch 2.7.1+cu126、Cellpose 4.2.1.1，依赖记录见 [remote_preprocessing_environments.json](../../configs/remote_preprocessing_environments.json)。

| 被测节点 | GPU | 精度 | Radtke 1536 × 1536，batch 4 | 峰值 PyTorch 分配显存 |
| --- | --- | --- | --- | --- |
| .72 | 2080 Ti | FP32 | 7.22 s | 2.04 GiB |
| .73 | 3090 | BF16 | 3.83 s | 1.15 GiB |
| .226 | 4090 | BF16 | 1.37 s | 1.15 GiB |

每节点选一张 GPU 使用固定 `cpsam_v2` 实测 Risom MIBI、Schurch CODEX、Radtke IBEX 的三个固定公开区域，其余 GPU 逐卡验证 256 × 256。测试区域采用 ROI 内 P99.5。`.72` 的 batch 1 与 4 输出逐像素相同。上表为预热后单次分割，计时排除传输、模型加载与 TIFF I/O。多 GPU 并发吞吐、全图分块接缝和正式 study 吞吐需按任务另行验收。

2026-10-24 执行器在 `.226`、`.73`、`.72` 合计 8 GPU 验证 `cpsam_v2`，对应 BF16/batch 32、BF16/batch 16、FP32/batch 8。另验证 `.81`、`.89` 的 CPU 后处理、结果返回与本机发布。现行执行和验收合同见[数据准备](../data/preparation.md#执行与验收合同)。

2026-10-22 对 PTM-Mamba 固定源码 `659724c3a108f9c904cfec2f35a582082cfdb1d3` 核对词表，包含 S/T/Y 磷酸化、赖氨酸乙酰化，未收录赖氨酸甲基化。加载与推理未验证。检查与许可记录位于 NAS `raw/shared/knowledge/ptm_mamba/`。
