# 预训练资源部署

核验日期：2026-10-18。机器可读来源与版本见 [sources.yaml](../../configs/sources.yaml)，存储归属见[存储规范](../architecture/storage.md)。下表中的路径相对 NAS 根目录 `/mnt/NAS_21T/ProjectData/CellWeft`。

| 资源 | NAS 路径 | 当前状态 |
| --- | --- | --- |
| MAP 官方知识编码器 | `resources/pretrained/map/mapkg_v3/mapkg_encoder_v3.pt` | 已下载，1,126,763,160 bytes；可读取，蛋白 projector 严格加载通过 |
| MAP 官方 ESM2 输入表 | 同目录 `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 已下载，410,886,729 bytes；19,790 个 gene symbols，每个 5,120 维 |
| MAP-KG 基因节点表 | `resources/knowledge/map_kg/` | 已下载基因表及 README；用于核对实体覆盖 |
| MAP 固定表示库 | `resources/protein_banks/map_gene_projector_v1/` | 已导出 19,790 × 1,024 float32 表示，附行键、manifest 和验证结果 |
| DINOv3 ViT-B/16，LVD-1689M | `resources/pretrained/dinov3/vitb16_lvd1689m/pytorch/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 原生格式目标路径，权重尚未下载；HF 登录已验证，模型访问待批准 |
| DINOv3 ViT-L/16，LVD-1689M | `resources/pretrained/dinov3/vitl16_lvd1689m/pytorch/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | 原生格式目标路径，权重尚未下载；HF 登录已验证，模型访问待批准 |

官方源码已分别部署到 SSD 的 `third_party/map/` 和 `third_party/dinov3/`，版本固定在 sources 配置中。两处 checkout 均受 Git 忽略规则保护；原有 `reference/` 用作阅读资料。

资源目录按[共享资源命名规范](../architecture/storage.md)统一为可读名称。`mapkg_v3` 对应官方知识编码器版本，`map_kg` 标识知识资源，`map_gene_projector_v1` 为 CellWeft 本地导出版本；完整来源 revision 保存在 sources 和 manifest 中。

DINOv3 的 HF 封装分别使用 `resources/pretrained/dinov3/vitb16_lvd1689m/huggingface/` 和 `resources/pretrained/dinov3/vitl16_lvd1689m/huggingface/`。当前仅有下载过程中取得的官方 README、LICENSE 和下载元数据；权重仍待审批。原生 `.pth` 的 `pytorch/` 路径在实际下载时创建。

## MAP 表示的含义与边界

本轮用户指定 MAP 为 protein embedding provider。当前导出的是官方知识编码器的 `gene_projector` 分支：固定 ESM2 向量 → MAP 知识训练后的 residual/gated projector → 1,024 维表示。输出取自对比损失执行 L2 normalization 之前的分支结果，不另行拼接调控向量。

官方向量以 gene symbol 为键。MAP-KG 基因表的 19,528 个唯一 ESM 键均有输入向量；表示库另含 262 个该表未列出的键。这一覆盖核对不代表所有蛋白 isoforms、phosphosites 或实际 marker panels 均已映射。接入数据时，仍按[数据合同](../architecture/data_contract.md)处理 canonical protein、measured form、clone 和 special markers。

官方权重、输入表、行顺序、维度、来源和输出校验和保存在 NAS manifest。代码仓库声明 MIT，MAP-KG 数据卡声明 Apache-2.0；所检查的权重发布页未单列权重许可，sources 中保留这一状态。

## 离线加载与重新导出

本轮实际运行环境为已有 Python 3.12.3、PyTorch 2.7.1+cu126、NumPy 2.5.2；计算使用 CPU。没有修改全局环境或安装完整 MAP 药物预测依赖，也未建立 CellWeft 正式训练环境。

加载已导出的库只需 PyTorch 和 Python 标准库：

```python
import json
from pathlib import Path
import torch

bank = Path('/mnt/NAS_21T/ProjectData/CellWeft/resources/protein_banks/map_gene_projector_v1')
gene_symbols = json.loads((bank / 'gene_symbols.json').read_text())
embeddings = torch.load(bank / 'embeddings.pt', map_location='cpu', weights_only=True)
row_by_gene = {gene: row for row, gene in enumerate(gene_symbols)}
protein_embedding = embeddings[row_by_gene['TP53']]
```

[导出脚本](../../scripts/resources/export_map_bank.py)按官方 projector 的 eval 前向计算，输入/输出维度来自训练权重。官方 checkpoint 带 NumPy scalar 元数据；脚本保留 `weights_only=True`，仅允许实际观察到的 NumPy 类型。脚本使用 Python 3.11+ 的文件摘要接口和 NumPy 2.x。

需要生成新版本时，提供一个尚不存在的输出目录，以下使用本地版本 `v2`：

```bash
python scripts/resources/export_map_bank.py \
  --checkpoint /mnt/NAS_21T/ProjectData/CellWeft/resources/pretrained/map/mapkg_v3/mapkg_encoder_v3.pt \
  --esm-table /mnt/NAS_21T/ProjectData/CellWeft/resources/pretrained/map/mapkg_v3/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --gene-metadata /mnt/NAS_21T/ProjectData/CellWeft/resources/knowledge/map_kg/GENE_tahoe_filtered.csv \
  --output /mnt/NAS_21T/ProjectData/CellWeft/resources/protein_banks/map_gene_projector_v2
```

实际验证：严格加载官方 projector 的全部参数；库中行键唯一，全部输出有限；均匀抽取 64 个基因与固定版本官方 `ResidualProjector` 比较，最大绝对误差 `1.430511474609375e-06`，通过 `rtol=1e-5, atol=1e-6`；官方 eval 重复前向完全一致。结果保存在库内 `verification.json`。这属于数值与部署验证，不是生物学有效性验证。

## Hugging Face 登录与配置

CellWeft 使用独立的 HF 2.2.0 客户端，安装在 SSD 的 `.venv-hf/`；版本见[客户端依赖](../../configs/huggingface-cli.requirements.txt)。通过[项目入口](../../scripts/resources/hf.sh)调用，避免继承当前 shell 中指向 OSMOSIS 的 HF_HOME。

| 设置 | CellWeft 路径或行为 |
| --- | --- |
| 登录凭据及 OAuth 刷新信息 | SSD `.cache/huggingface/`；目录权限 700，新建文件权限 600；Git 忽略 |
| 模型下载缓存 | NAS `cache/huggingface/hub/` |
| Xet 与其他资源缓存 | NAS `cache/huggingface/xet/`、`cache/huggingface/assets/` |
| 本机代理 | SSD `.cache/huggingface/network.env`，仅对项目入口进程生效 |
| 服务地址 | `https://huggingface.co` |

```bash
# 在项目根目录执行。浏览器登录会给出短期验证码，无需把 token 粘贴到聊天。
scripts/resources/hf.sh auth login --no-add-to-git-credential
scripts/resources/hf.sh auth whoami
```

需要重建客户端时，先创建 `.venv-hf`，再用其 Python 执行 `pip install -r configs/huggingface-cli.requirements.txt`。NAS 根目录可通过 `CELLWEFT_DATA_ROOT` 覆盖。入口只设置当前进程的环境，不修改 shell 启动文件、全局 Git 凭据或其他项目的登录配置。

账号登录与模型访问审批分别验证：登录成功后，仍需账号对 B/L 模型具备访问权限。授权码过期时重新执行登录命令，按新码完成浏览器授权。

2026-10-18 已完成官方 OAuth 浏览器授权，`auth whoami` 验证通过；实际凭据文件权限为 600，目录为 700。随后对 B/L 固定 revision 发起下载，两者均返回 `requires approval`，尚未取得权重。

## DINOv3 待完成步骤

官方 GitHub 列出 B/16、L/16 的 LVD-1689M distilled checkpoints。初次未登录时，普通官方下载请求返回 HTTP 403，HF 文件请求返回 HTTP 401。当前 CellWeft 的 HF 登录已验证；两个模型的账号访问权限仍待批准。

用户已确认申请处于等待审核状态；后续由用户发起权限复查。

1. 使用上面的项目入口完成 Hugging Face 浏览器登录，并在官方 B/L 模型页确认访问审批；也可将 Meta 批准邮件中的 B/L 下载链接保存到被忽略的 `.cache/resource-deployment/dinov3-urls.local.json`，键为 `vitb16`、`vitl16`。签名链接和 token 不写入 sources 或 Git。
2. 将权重下载到上表目标目录；核对文件身份。原生 `.pth` 的官方文件名包含 SHA-256 前缀，B 为 `73cec8be`，L 为 `8aa4cbdd`。HF 的 `model.safetensors` 是另一种封装，应按其格式加载并记录来源 revision。
3. 通过固定版本官方代码严格加载原生权重，在 eval 模式运行前向；256×256 RGB 输入应提供 256 个 patch tokens，B/L 宽度分别为 768/1,024。记录实际结果后才将 sources 状态改为已验证。

下载原生权重后的调用形式：

```python
model = torch.hub.load(
    '/home/lenislin/Experiment/projects/CellWeft/third_party/dinov3',
    'dinov3_vitb16',  # L uses dinov3_vitl16
    source='local',
    weights='/mnt/NAS_21T/ProjectData/CellWeft/resources/pretrained/dinov3/vitb16_lvd1689m/pytorch/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
).eval()
```

资源部署不自动改变[训练规范](../architecture/training.md)中的领域预训练方案。L 权重作为用户要求的可用资源部署，V1 主架构仍按当前模型规范维护；采用这些权重初始化 multiplex 分支时需另行明确 PatchEmbed 适配和参数加载规则。

## 分割模型资源（2026-10-19）

用户授权的12源图探索已部署并实测以下三个预训练模型；环境均为项目局部 venv，基础环境只读继承。没有训练或微调。参数、真实返回与资源测量见[分割探索记录](../experiments/segmentation_pilot_20261019.md)。

| 模型 | 本地环境 / 包版本 | NAS resources/pretrained 下的权重 | 实测范围 |
| --- | --- | --- | --- |
| Cellpose-SAM cpsam_v2 | `.venv-cellpose-sam` / cellpose 4.2.1.1 | `cellpose/cpsam_v2/checkpoint` | RTX4090，A/B各12图 |
| InstanSeg fluorescence_nuclei_and_cells 0.1.1 | `.venv-instanseg` / instanseg-torch 0.1.2 | `instanseg/fluorescence_nuclei_and_cells/0.1.1/instanseg.pt` | RTX4090，A/B各12图，核与细胞分别保存 |
| StarDist 2D_versatile_fluo | `.venv-stardist` / stardist 0.9.2、TF 2.12.0 | `stardist/2D_versatile_fluo/` | CPU，B核输入12图 |

依赖记录分别位于 `configs/cellpose-sam.requirements.txt`、`configs/instanseg.requirements.txt`、`configs/stardist.requirements.txt`。前两环境继承 OSMOSIS_V2 的 PyTorch 2.7.1+cu126；StarDist继承现有 cellvit_env 的Python3.10/TF2.12。它们不是完全隔离的可移植锁文件。

cpsam_v2来自[官方 Hugging Face 固定revision](https://huggingface.co/mouseland/cellpose-sam/tree/7c61431b5fbb078f3296754bd15d9f51b320f837)，实际大小1,233,586,851字节，下载后文件SHA256与官方LFS标识一致，显式加载为sam_vitl、bfloat16。项目入口 `scripts/resources/cellpose-sam.sh` 将模型/cache/Matplotlib目录指向NAS；显式权重变量为 `CELLPOSE_SAM_MODEL`，调用模型时传入该文件路径。

InstanSeg来自[官方模型release](https://github.com/instanseg/instanseg/releases/tag/instanseg_models_v0.1.1)，模型pixel_size为0.5µm/px；StarDist来自[官方预训练模型release](https://github.com/stardist/stardist-models/releases/tag/v0.1)。均使用显式本地文件加载，无推理时自动下载。StarDist的`KERAS_HOME`不保证改变`get_file`默认缓存，本项目直接下载官方ZIP到NAS并从本地目录构造模型。

本轮模型名称中的cell表示输出目标，不等于每个study已验证whole-cell范围。Mesmer未部署，本轮因旧TF环境成本省略。DINOv2全面文档迁移仍独立待执行。
