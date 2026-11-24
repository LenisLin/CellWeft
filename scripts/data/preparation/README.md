# 数据准备入口

科学规则由[准备规范](../../../docs/data/preparation.md)维护，字段和坐标由[数据合同](../../../docs/architecture/data_contract.md)维护。Dataset脚本负责来源解释和study循环，共享单图处理在`src/cellweft/data/preparation/`。

进入多机入口前配置实际运行根、共享队列、锁域及 checkpoint，见[分布式入口与恢复](#分布式入口与恢复)。

## 环境与顺序

在项目根运行，使用现有`.venv-data`或安装`pyproject.toml`声明的依赖；脚本通过`PYTHONPATH=src`读取共享包。本地分割使用`scripts/resources/cellpose-sam.sh`对应的项目环境（实际命令见下方）。

## Per-study输入整理

以下入口按[输入工作流](../../../docs/data/study_input.md)读取完整来源，查询S1，导出逐image的原生强度`input.tiff`、完整`images.parquet`及study级panel、sample和字段映射。独立校对按[校对与修补流程](../../../docs/data/study_input_review.md)执行。

```bash
PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/01_aoki2024_study_input.py
.venv-data/bin/python scripts/data/preparation/tu2026_clinical_metadata.py
PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/02_tu2026_study_input.py
```

- `Allam2024_IMC_NormalSLO.py --metadata-only` 生成来源清单和 S1 提案。`--output-root <候选父目录> --s1 <候选S1>` 导出 TIFF，`--verify-only` 读回结构和像素窗口。
- `Sorin2023_IMC_Lung.py --output-root <候选study目录> --s1 <候选S1> --limit 0` 刷新输入表。`--shard-count` 和 `--shard-index` 划分源成员。Allam 与 Sorin 在正式 S1 已登记该 study 时默认复用当前行。
- `biad3612_codex_hnscc.py --job <任务目录> --wait-seconds 90` 复用版本匹配的产物并处理新到齐 core。已确认损坏成员按 inode、大小和 mtime／ctime 复用跨会话记录。`--metadata-only` 在原 writer 退出后补录字段规则和语义追查，并保留 input 像素。

来源包脚本使用 `metadata_source_directory` 临时还原读入成员，`metadata_source_relative` 生成 bundle/member 引用，来源跟进通过 `publish_metadata_sources` 写回。来源包格式和目录见[存储规范](../../../docs/architecture/storage.md#来源-metadata-单文件归并)。

## input.tiff 后处理入口

本机用 `run_local_preparation.py --study <study> --data-root <NAS根> --run-root <SSD运行根>` 读取 D01、在 SSD 计算并逐图发布。`--image-ids-json` 可限定代表批。续跑使用同一 `--run-root`，复用当前绑定一致的完整产物。`--sampling-reviewed` 只用于已核验组织估计的范围，`--allow-unknown-mask-segmentation` 只用于已明确获准的范围。

通用入口消费 [D01 当前输入绑定](../../../docs/data/study_input_review.md#5-程序消费的验收绑定)。运行前将 `STUDY_ROOT` 设为实际 study 目录。

```bash
PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/run_study_preparation.py \
  --study-root "$STUDY_ROOT" \
  --data-root /mnt/NAS_21T/ProjectData/CellWeft \
  --mask-bindings /path/to/reviewed_mask_bindings.json \
  --cache-dir /path/to/local/cache --node local \
  --weights /mnt/NAS_21T/ProjectData/CellWeft/raw/shared/pretrained/cellpose/cpsam_v2/checkpoint

PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/export_study_marker_embeddings.py \
  --study-root "$STUDY_ROOT" \
  --data-root /mnt/NAS_21T/ProjectData/CellWeft \
  --s1 /mnt/NAS_21T/ProjectData/CellWeft/processed/_shared/paper/supplementary_tables/supplementary_table_s1_marker_mapping.csv \
  --bank /mnt/NAS_21T/ProjectData/CellWeft/processed/_resources/protein_banks/map_gene_projector_v1
```

两个入口支持重复`--image-id`，选定范围必须与D01绑定一致。逐图入口执行CPU处理及文件交接，不自动派发远端任务；节点选项`226/73/72/local`分别固定BF16/32、BF16/16、FP32/8、BF16/16。向量入口仅导出选定已验收image实际panel的marker，范围记录在载荷metadata；读取使用published_study_embeddings并给出expected_study，同时核对载荷中的完整panel身份。向量入口使用现有PyTorch CPU环境；依赖声明见`tensor_data`可选组。

Mask绑定JSON按image_id索引，每图显式给`status: unknown/absent/bound`。unknown保留缺口；absent表示已确认无作者mask；bound同时给`path`（NAS相对路径）、`frame: source/input`及`mask_type`。来源坐标先执行声明的`flip_x/flip_y/quarter_turns`，再用输入来源框裁剪、添加同一padding；input坐标不重复转换。TIFF指定page，NPZ指定`format: npz`、`npz_key`及selection（整数选轴、null保留轴）。原始mask只读。必要概览沿用overview_required／overview_judgment。Amitay使用本地作者mask的明确配对。

`--tissue-reviewed-panel`只在该panel代表图已核验组织估计后传入；未核验时采样状态为pending、count=null。已核验而无合格窗口时count=0。窗口选择可按需通过src的sampling接口重建，完整cells.crop_origins保留。

通过 `00_prepare_marker_registry.py --s1 <最终S1> --output <NAS>/raw/shared/metadata.json` 派生共享 registry。

## 当前代码的CSV输入接口

Registry CSV列：`marker_id,display_name,canonical_protein_id,measured_form`。家族/组合的canonical protein ID可空；非蛋白核染也有明确measurement marker_id。

旧来源适配器的基础Panel CSV列：`panel_id,source_channel_index,source_name,marker_id,role,nuclear,body,exclusion_reason`。每panel按全部来源通道从0连续编号；role为`downstream`、`segmentation-only`或`exclude`；nuclear/body用`true/false`，downstream蛋白可兼任分割通道。人工排除的marker_id可空。运行时逐图数值排除保留来源行，输出TIFF page和表达列由`channels.parquet`给出。

## 产物读取与恢复

使用 `published_image` 按当前 `expected_request` 和 `required` 读取正式产物，在上下文内关闭文件句柄。向量使用 `published_study_embeddings` 并给出 `expected_study`。原任务恢复复用当前依赖有效的完整项，按 `product_status` 接续。依赖及发布顺序见[存储规范](../../../docs/architecture/storage.md#准备产物文件布局)。

## 分割任务交接

通用逐study入口生成文件任务。对应image metadata的`segmentation_task`给出signal、task和result所在缓存目录，按需复制到GPU主机。

```bash
PYTHONPATH=src scripts/resources/cellpose-sam.sh scripts/data/preparation/run_segmentation_worker.py \
  /path/to/task.json --device cuda:0 --precision bf16 \
  --input /remote/signal.tiff --weights /remote/cpsam_v2/checkpoint \
  --result /remote/result.json
```

将 `result.json` 和同目录 `mask.tiff` 复制回任务指定的 result 目录，重跑原入口。命令的 `--precision`、`--core-size`、`--halo`、`--match-iou` 与提交任务一致，运行时 checkpoint 与 signal 通过任务身份核对。一个 worker 可顺序处理多个 task。缺任务 JSON 时从 image metadata 恢复。如果结果引用的 mask 尚不可读，保留待处理。仍有有效 signal 时可重算，否则重建任务。待整图复核的结果将 `whole_cell_confirmed` 填入 result JSON 后接续。

## 控制器与接管

- `run_study_input_controller.py --run-root <已有运行根> --workers <数量>` 读取已有计划和阶段状态。`--status` 只读查询。同一 study 保留一个负责人，上下游共用 `CELLWEFT_NAS_LOCK_ROOT`，在实际操作内领取 I/O 名额。
- `run_preparation_batch.py --config <runtime.json>` 从配置读取 `run_root`、当前 D01、mask 绑定、节点和资源。`--continuous` 持续接收实际新增及变化范围。`run_preparation_controller.py --plan <plan.json>` 消费已冻结计划。
- `run_remote_worker.py` 执行单个配置的 GPU／CPU 槽，`run_cpu_worker.py --task <task.json>` 执行独立 CPU 派生任务，`run_transfer.py` 执行受限文件交接。运行前对照各入口的 `--help`，确认各节点可见的 payload、队列、运行根和实际写权限。
- `export_accepted_study_embeddings.py --plan <plan.json> --s1 <final.csv> --bank <bank-dir>` 按已验收范围导出。`--continuous` 接续新范围，使用向量专属运行根，状态写入 `embedding_status.json`。
- 批处理控制器在运行根读取 `STOP_ADMISSION` 或 `STOP` 后排空已准入工作。连续发现循环以 `STOP` 或受控信号结束。重启前解除对应停止标志，使用原运行根、计划及接收回执接续未完成范围。

## 分布式入口与恢复

配置显式源数据根、目标逻辑 owner 根、物理执行根、共享队列及 I/O 锁根。下列参数使用实际部署的 Python、主机和 checkpoint。
本机交付入口须指定目标project root、目标gateway、Python及已有checkpoint；`--max-deliveries 1`限制本轮交付数：

```bash
PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/run_study_handover.py \
  --node 72 --owners /path/to/study_owners.json --state-dir /path/to/run_state \
  --data-root /path/to/source_project --mask-bindings-root /path/to/mask_bindings \
  --code-root /path/to/target_code --node-root /path/to/owner_project \
  --gateway-host target-host --gateway-python /path/to/target_python \
  --weights /path/to/target_checkpoint --max-deliveries 1
```

目标节点的完整下游入口须指定owner根及交付队列；`--packet <交付目录> --once`可限定一次处理：

```bash
PYTHONPATH=src /path/to/target_python scripts/data/preparation/run_study_node.py \
  --node 72 --device cuda:0 --node-root /path/to/owner_project \
  --queue-dir /path/to/owner_project/transfer/study_deliveries \
  --packet /path/to/owner_project/transfer/study_deliveries/study/delivery --once
```

拆分CPU与GPU阶段时，生产者和消费者须指定同一共享queue根；handover还须传同一`--stage-queue-root`，以发布CPU交付事件：

```bash
PYTHONPATH=src /path/to/target_python scripts/data/preparation/run_study_cpu_stage.py \
  --owner 72 --node 72 --node-root /path/to/owner_project \
  --queue-root /path/to/shared_queue --worker-id cpu0 --once
PYTHONPATH=src /path/to/target_python scripts/data/preparation/run_study_segmentation_queue.py \
  --node 72 --device cuda:0 --node-root /path/to/physical_executor_project \
  --queue-root /path/to/shared_queue --once
```

1. CPU 的 `--node-root` 为逻辑 owner，GPU 队列的 `--node-root` 为物理执行节点。同一物理 GPU 的 direct 和 queue worker 共用物理根及 `locks/cuda_N.lock` 和 `locks/cuda_N.worker.lock`。生产与消费共用同一 `--queue-root`，交付端将其传给 `--stage-queue-root`。需要覆盖 GPU 运行时 checkpoint 路径时，传 `--weights` 并保持任务要求的内容身份。
2. `run_study_node_embeddings.py` 必须给 `--node` 和 `--node-root`。可用 `--queue-dir` 覆盖默认的 `<node-root>/transfer/study_deliveries`。`dispatch_study_delivery.py` 接收 `--study-root` 或 `--scan-root`，以及 `--data-root`、`--mask-bindings-root`、`--owner-map`、`--node`、目标 `--node-root`、`--settings`、`--s1`、`--bank-dir`、`--weights`、`--gateway-host`、`--gateway-python`、`--gateway-code-root` 和 `--lock-root`。
3. CPU 消费分割完成事件前核对当前任务及结果可见性。事件早于结果文件时保留待处理，并在后续轮次重新读取同一事件。已完成且绑定相同的 mask 直接复用。
4. 交付恢复核对 `same_processing_binding`。仅接收管理字段改变且处理绑定相同，保留交付版本和既有 pending 包。实际处理绑定变化时按受影响范围重新交付。待接收图像为当前范围扣除已接收图像，且已接收项须匹配当前绑定。
5. 受控停止时等当前发布结束、worker 释放锁。恢复读取原计划、完成事件及接收回执，接续未完成范围。
