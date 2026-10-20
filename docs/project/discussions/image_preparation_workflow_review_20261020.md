# TIFF / Parquet 工作流独立复核

## 本次落盘后的独立复核（2026-10-20）

用户明确授权发送8个技术文件及批准计划后，codex-cli以`gpt-6-astra`、`model_reasoning_effort="max"`、只读沙箱完成独立审阅，退出码0。[原始报告](/mnt/NAS_21T/ProjectData/CellWeft/results/workflow_document_review_20261020/astra_review.md)的结论为“有条件通过，限于规范审查”，提出两项修订：

1. 部分产物发布：删除无条件将image登记为完成的伪代码；`publish_image`检查实际产物及依赖，将完成项与缺口写入metadata，study index保持七字段。
2. 家族／组合身份：canonical protein ID明确可空；能对应单一蛋白时填写，家族／组合保留marker_id和来源测量身份。

上述两项已由主代理修正。同时采纳简洁性建议：并发参数集中在多机安排，删除表达查询示例中重复的Parquet行序提醒。独立审阅确认现行规则与候选参数、四项mask检查、三方索引、目录身份和定量解释边界基本自洽，目前没有需要用户重新决定的问题。

当前规范见[数据准备](../../data/preparation.md)和[数据合同](../../architecture/data_contract.md)。修订后的本地检查由主代理执行，未再次送审；生产读取、全图拼接及实际分割效果仍属于实现验收。

## 早期方案复核记录

以下保留当时的结论与行号；其中待讨论选择由后续用户决定和当前规范接替。

2026-10-20。通过 codex-cli 显式指定 `gpt-6-astra`、`model_reasoning_effort="max"`，只读沙箱执行，CLI 退出码为 0。以下保留独立审查结论；送审行号对应冻结输入，不是复核后修订稿。主代理已将最小工程修正吸收至[当前讨论方案](image_preparation_workflow_20261020.md)，四项科学/产品选择继续待讨论。本次未实施生产处理。

**结论：可作为讨论基线，不能直接生产执行。** [目标稿](/home/lenislin/Experiment/projects/CellWeft/.cache/workflow_review_20261020/input.md:3)正确承接了最新归一化、定量和格式要求，也标明尚未实施。主要缺口是读取器、异常输入、就绪判定及删除验收的合同闭合；旧试验不能作为新 P99.5 流程已验证的证据。

按严重度，需要以下最小修正：

- **高：TIFF 直接复用条件不足。** 目标稿 L31 允许强度规则兼容的 TIFF 直接引用，L108 又要求只解码窗口相交 tile；压缩 strip TIFF 不一定满足后者。现有 [pilot_io.py:117](/home/lenislin/Experiment/projects/CellWeft/scripts/experiments/segmentation_pilot/pilot_io.py:117)大图路径实际使用 `aszarr`，读取范围还受 1536px 限制。最小修正：直接复用同时要求空间布局支持规定的窗口读取；否则保持像素值无损转存 tiled `.tiff`，不再次归一化。验收除像素一致外，还须观察实际读取字节、解码 tile 数及峰值内存。

- **高：产物完成与训练就绪尚未明确连接。** 目标稿 L23 允许 bank 未明确时保存通道产物，L116 的发布检查却未落到 [catalog.md:20](/home/lenislin/Experiment/projects/CellWeft/docs/data/catalog.md:20)的分层准入。最小修正：在同一主索引记录实际产物和可用层级，明确尺度、保留通道映射、mask、表达及 crop 对齐的依赖。`channels.parquet` 应覆盖全部来源通道；排除通道保留原因，page/表达列引用为空。缺 mask、缺 bank 或蛋白全部排除，不能仅凭目录完整声明 training-ready，也不能填零补齐。

- **高：no-op 无法验收真实删除对应。** 目标稿 L118 只有 no-op 删除闭环，不能证明 L80 所述跨 crop 删除正确。[editing.md:20](/home/lenislin/Experiment/projects/CellWeft/docs/architecture/editing.md:20)还要求更新反向邻居、halo 和 membership。最小修正：增加一个跨重叠 crop、具有反向邻居的真实删除用例，检查保留 ID、表达和实例外原图像素不变，邻域引用同步移除。先编辑原图实例像素，再按相同插值生成视图；仅在重采样后的最近邻 mask 内置零可能留下插值混入的信号。

- **中：归一化有效与分割信号有效需要分开。** 目标稿 L29、35、41 尚未闭合非有限值、常数和缺失合成组的行为。最小修正：非有限通道记输入失败，不发布依赖它的完成产物，不忽略异常像素改算分位数；正值常数通道按已确认公式得到全 1，不能沿用旧 P1/P99 的“常数即排除”。但无空间变化的核合成信号不能据此认定可分割。无核且有合格 mask 可定量；无有效核不新分割。缺 body 可沿用 [preparation.md:37](/home/lenislin/Experiment/projects/CellWeft/docs/data/preparation.md:37)的零 body 起点及一次 whole-cell 概览，无需重开讨论。

- **中：来源兼容问题不宜整体重新提问。** 目标稿 L126 将“已处理但规则不同”列为待讨论；[preparation.md:11](/home/lenislin/Experiment/projects/CellWeft/docs/data/preparation.md:11)已允许强度语义明确的发布像素补齐当前归一化。最小修正：沿用该原则，落实已配准、真实统计范围及已知缺失填充值的边界；仅对语义不明或只有显示用途的具体来源提出例外讨论。不得把读取器填零当实测背景，也不必重新审计数据库已给出的 mask 配对。

**A 类仍需用户讨论的事项，可收敛为四个主题：**

1. **基础 crop 的空间与交付方式。** 选择统一物理 FOV 还是原生像素范围，并确定抽样、边缘/空区原则及是否接受虚拟 crop。推荐有可靠尺度者统一物理范围，以全图 TIFF 加索引交付；224µm/112µm仍是候选，不能从网络224/98推导。选择会改变细胞尺度、组织上下文和样本权重。
2. **部分可见细胞如何参与直接对齐。** 推荐所有实际出现的实例保留索引和上下文，初版直接对齐只用完整可见实例；代价是对齐目标减少。整细胞表达继续固定，不为局部视图重算。
3. **家族、组合及 measured form 如何进入固定 bank。** 推荐按明确靶标集合和测量形式统一跨 study 身份；明确组合可讨论固定成员向量池化，同名 cocktail 不自动合并。还需保证测量形式能进入两分支：仅使用不同 marker_id、却输入相同向量，并不必然提供可辨信息。未知靶标继续保存通道产物，不臆造映射。
4. **是否取消分割合成信号的二次 P1/P99。** 推荐将取消作为候选，以少量固定图验证后锁定。即使保留二次适配，新全图 P99.5 也不是旧 CP_B 复现：旧 [01_prepare.py:81](/home/lenislin/Experiment/projects/CellWeft/scripts/experiments/segmentation_pilot/01_prepare.py:81)先做区域 P1/P99，随后再次适配合成信号。

**B 类可以由工程师选择起点并实现验收：**

- TIFF 压缩、tile、缓存、Parquet 行组和精确分位数算法；验证完整独立 image 统计、跨 tile 窗口及有界暂存，不能用 pilot ROI 代替全图。
- 固定文件/page/marker/表达列与 mask_label/cell_index/crop 的键关系；稀疏标签先映射到连续索引再累计，避免照搬旧脚本按最大标签分配 `bincount` 数组。覆盖空表、邻居并列及 `-1` 禁止有效 gather。
- 大图 core/halo、实例匹配阈值、GPU batch/精度和并发；验收接缝后冻结全图 mask，再建立表达与邻域。接收结果须匹配任务版本、有效 attempt 和图像/tile 范围，关闭并核验完整文件后发布。
- 按最新会话同步规范和就绪字段，保留历史实验定义。远端环境、挂载、吞吐和缓存上限属于实施验证，不要求用户逐参数批准。

**C 类已确认规则应保留，不再提问：**逐独立 image/channel、全部实测像素 P99.5、固定零下界及 `clip(x/q,0,1)`；`q<=0` 仅排除当前图像下游通道，接受稀疏信号损失并保留来源。Whole-cell 内全部像素均值含零、不加 arcsinh；已有 mask 仅四项检查并优先复用；完整图像原像素 centroid top-20；baseline 统计固定；224/98、patch14；新产物使用 `.tiff/.parquet/.csv/.json`，不新用 Zarr，正式数据归 NAS。

实际读取了目标稿、AGENTS.md、status 相关末尾及指定五份规范；另读 editing、01/02/05、非零 P99 报告，以及文档入口、research_plan、decisions、pilot_io 和 model/training 相关内容。**未读取图像像素，未运行处理或测试、安装、SSH，未调用其他代理。** TIFF 新读取路径、P99.5 实际输出、拼接质量及远端运行能力均未验证；资源情况沿用会话已核实记录。