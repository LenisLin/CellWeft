# 数据准备入口

科学规则由[准备规范](../../../docs/data/preparation.md)维护，字段和坐标由[数据合同](../../../docs/architecture/data_contract.md)维护。Dataset脚本负责来源解释和study循环，共享单图处理在`src/cellweft/data/preparation/`。

## 环境与顺序

在项目根运行，使用现有`.venv-data`或安装`pyproject.toml`声明的依赖；脚本通过`PYTHONPATH=src`读取共享包。本地分割使用`scripts/resources/cellpose-sam.sh`对应的项目环境（实际命令见下方）。

1. 人工确认全局marker语义及各来源panel映射。
2. 运行`00_prepare_marker_registry.py`保存registry。
3. 运行对应dataset入口，生成study index和逐图产物。
4. 使用文件交接时，运行worker并将结果送回任务指定目录；重跑dataset接续定量和索引。

```bash
PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/00_prepare_marker_registry.py \
  --confirmed /path/to/confirmed_markers.csv \
  --output /mnt/NAS_21T/ProjectData/CellWeft/metadata/marker_registry.csv

PYTHONPATH=src .venv-data/bin/python scripts/data/preparation/risom2022_mibi_breast_cancer.py \
  --image-id Point2203_31606

PYTHONPATH=src scripts/resources/cellpose-sam.sh \
  scripts/data/preparation/risom2022_mibi_breast_cancer.py \
  --image-id Point2203_31606 --device cuda:0
```

当前提供Risom发布TIFF的dataset入口。其已确认panel CSV由研究者准备在配置指定的NAS路径；尚未生成项目全量人工映射，因此上述命令的真实执行以该输入到位为前提。

## 两张人工确认表

Registry CSV列：`marker_id,display_name,canonical_protein_id,measured_form`。家族/组合的canonical protein ID可空；非蛋白核染也有明确measurement marker_id。

Panel CSV列：`panel_id,source_channel_index,source_name,marker_id,role,nuclear,body,exclusion_reason`。每panel按全部来源通道从0连续编号；role为`downstream`、`segmentation-only`或`exclude`；nuclear/body用`true/false`，downstream蛋白可兼任分割通道。人工排除的marker_id可空。运行时逐图数值排除保留来源行，输出TIFF page和表达列由`channels.parquet`给出。

## 共享模块与脚本范围

|层|职责|入口|
|---|---|---|
|Dataset脚本|枚举来源、panel/尺度/ROI解释、study循环|`discover_images`、顶层顺序代码|
|读取|TIFF page/多文件、来源Zarr、IMS、MCD窗口|`io.readers.open_source`|
|强度|全实测域精确P99.5、float32共享图像|`intensity.prepare_shared_image`|
|Mask|四项粗检、必要的整图概览|`masks.check_existing_mask`|
|分割|有效核/膜均值、cpsam_v2、core/halo拼接|`segmentation.CellposeExecutor`|
|细胞|稳定标签对应、全实例均值、邻居、crop关系|`indexing`、`quantification`、`neighbors`、`crops`|
|流程|复用/失效、部分完成、分割结果接收|`pipeline.prepare_image`|
|落盘|TIFF块流、Parquet、JSON、完成记录|`io.writers`|

`SourceSpec`明确原文件和ROI位置，所有计算使用独立image左上角为0的坐标。TIFF读取只解码相交tile/strip；最大解码内存仍受来源单段大小约束。IMS和CZYX来源需显式z_index；MCD缺失采集位置由其记录坐标识别，避免作为实测零参与P99.5。

兼容的上游归一化由dataset适配显式确认相同公式、统计域及通道q值，再设置`SourceSpec.normalized_q995`。共享模块检查有限值与[0,1]范围并直接写出，不重复缩放；仅有“已处理”描述不足以声明兼容。

来源mask的`overview_judgment`取true/false/null：false否定复用；确需整图概览时设置`overview_required=True`，等待一次判断，期间保存共享图像及缺口，不启动新分割；判断通过则复用，明确否定后才进入新分割。可信来源默认无需概览。所有判断仍属于既定四项粗检。

## 产物与恢复

正式位置`processed/<study>/`，文件清单见[存储规范](../../../docs/architecture/storage.md#准备产物文件布局)。`cell_index`显式连接expression、mask_label和cells内crop_origins；邻居固定20列、不足填-1。Crop membership直接按每个窗口中的实际实例标签建立，允许部分细胞。

Pipeline依据输入选择、文件size/mtime、panel、相关参数与mask身份决定复用。目录型来源需提供明确source_revision。来源mask绑定自身文件与图像空间选择；预测mask另绑定分割输入、权重和参数，metadata的mask_origin区分两者。Mask变化重建CellTable及下游；缺失表达等下游文件时直接复用完整的正式mask。数值排除仅影响对应image/channel。缺mask保存共享图像；缺蛋白列保存已有几何和邻域。

每个image只允许一个writer。发布时先撤销旧完成记录，再替换已关闭的文件，最后原子写metadata；中断后缺完成记录的image重算。更新期间停止该image的并发消费。读取产物以metadata引用为准，目录中的旧文件名不代表当前完成项。

## 分割任务交接

Dataset入口未指定`--device`时生成文件任务。对应image metadata的`segmentation_task`给出signal、task和result所在缓存目录，按需复制到GPU主机。

```bash
PYTHONPATH=src scripts/resources/cellpose-sam.sh scripts/data/preparation/run_segmentation_worker.py \
  /path/to/task.json --device cuda:0 \
  --input /remote/signal.tiff --weights /remote/cpsam_v2/checkpoint \
  --result /remote/result.json
```

将`result.json`和同目录`mask.tiff`一起复制回任务指定result目录，再重跑dataset。同步执行器和远端worker都使用已提交的参数；worker核对实际信号与权重的内容摘要，以及两页float32/CYX信号形状；回执记录实际身份。接收端核对task、实际内容身份、参数与拼接设置。内容摘要只用于分割交接文件和权重，不扫描原始数据集。多个task可由单worker顺序处理并复用模型。待处理任务缺失的task JSON可从image metadata补回；result JSON引用的mask不存在时保持待处理，有效signal用于重算，signal也缺失则重建任务。已完成图像直接读取正式产物，分割缓存可清理。

起始外层core=1024、halo=128、重叠区域IoU≥0.5进行一对一实例匹配，core像素写回全图。该拼接规则通过合成接缝测试后仍需按study检查真实接缝质量。缺有效body时使用零body，结果需一次whole-cell整图判断；在result JSON中记录`whole_cell_confirmed`后接续。多机任务分配目前由操作者组织，worker不自动SSH或复制数据。

## 验证

```bash
.venv-data/bin/python -m pytest -q tests/data
```

测试覆盖精确统计、已知缺失像素、通道退化、稀疏实例、空间索引、部分完成与恢复、分割交接和合成拼接。真实GPU冒烟和实际来源格式验收情况以[项目状态](../../../docs/project/status.md)为准。
