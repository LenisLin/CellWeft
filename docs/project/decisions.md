# 已接受的项目决定

2026-11-24 用户确认私有膀胱癌队列 `Feng2026_IMC_BladderCancer` 的数据所有者为 Feng，登记年份为 2026，来源别名为 `IMC_20240725_63`。本批按来源 `steinbock/images.csv` 收入 520 个 ROI。验收通过的来源实例 mask 用于细胞定量和空间邻域，缺失或确认不合格者按现行准备规范重分割。[队列工作流](../development/imc_20240725_63_preparation.md)

同日用户要求删除其中 5 个患者连接未确认的 ROI，并纳入训练表及 8T 交付。正式保留 515 图／62 位患者，采用约 80%／20% 的患者级训练／内部验证两分方案，固定种子 `20261124`，得到训练 50 位患者／421 图、验证 12 位患者／94 图。验证图不参与训练拟合；以直接确认的患者键分组，独立 sample 缺口继续保留。该授权仅适用于 Feng 队列，组织采样核验仍待完成。[数据目录](../data/catalog.md)

本页记录现行选择及理由，技术细节由对应规范维护，实施进度见[当前状态](status.md)。

## 研究范围与来源准入

- V1 收集人类肿瘤含原位癌、配对癌旁或正常对照，以及独立正常组织。混合来源只纳入身份明确的合格人类成像范围。其他癌前病变暂缓，以其他非肿瘤疾病为主的研究排除。独立患者／供者、图像或 TMA 物理 core、各 panel 的不同蛋白各自计数。按可读原生强度和准确通道对应接收合格子集，以来源清单记录缺失与排除范围。指定 panel 例外、来源撤回与范围门槛由收集规范维护。[研究计划](research_plan.md) · [收集规范](../data/collection.md)
- 来源接收、技术产物、患者连接与任务准入各自判定。来源及 panel 明确的图像允许先完成技术处理。仅保留可证实的 sample 与患者连接，缺口约束相应下游用途。部分接收按实际图像和产物登记。区域来源值保留，冲突标签在任务选择时排除待核成员。[数据目录](../data/catalog.md) · [输入工作流](../data/study_input.md) · [公开患者来源审计](../data/patient_metadata_public_scope_audit_20261111.md)

## Marker、强度与空间资产

- Marker 按实测身份与测量形式对齐。等价别名共用身份，isoform、total／phospho、完整／切割等不同测量独立。确认的聚合命名保留原始靶点和修饰信息，组合实测保留单列。S1 维护实际去向、排除项、分割用途及 `mean`／`proxy` 映射，近似向量映射不改变实测身份。固定 MAP 表示在输入通道筛选后按 study 查表，按有效通道绑定行顺序。[数据准备](../data/preparation.md) · [补充表 S1](../paper/supplementary_tables.md)
- 原输入保留来源强度和采集几何。共享图像按 ROI×marker 的全部实测域计算精确 P99.5，包含实测背景与零值。明示无效域、padding 与未测量状态，保留实际 `q995`，重用公式与实测域一致的结果。归一化或坐标真正变化时更新全量关联产物。原始几何可靠才补边，保持原生像素；获批的原生矩形及小图例外按逐图记录执行。组织估计待核验时保留 training sampling pending。[数据准备](../data/preparation.md) · [数据合同](../architecture/data_contract.md)
- 作者 mask 复用要求实例类型、源图身份、坐标和质量有据。未取得合格 mask 的图像可按批准范围使用新分割，实际 mask 来源与作者状态各自记录。最终实例 mask 定义 cell index、标签、质心、完整实例均值和同图邻域，先定量及建索引再切窗。排除的 ROI 或测量记录在成员和通道层，保留范围内依赖及产物一致时沿用原处理。[数据准备](../data/preparation.md) · [数据合同](../architecture/data_contract.md)
- D01 绑定精确输入版本和认可范围，所有图像完成适用计算与自动结构／版本检查。来源人工对照及独立数值重核按 study 分层抽样，异质分组补选、异常扩查，明确区分逐图直接证据与组级沿用。派生分组变化只触发受影响范围的核对。[独立校对](../data/study_input_review.md) · [数据准备](../data/preparation.md)

- 分割通道以原始 panel 角色、来源抗体或染料、图像复核和 S1 最终去向确定。核输入优先 DNA 专用信号，复核通过后允许复合核信号和总 Histone H3。Greenwald／Ranek 的组蛋白修饰候选用途留空。核染和仅用于分割的重复测量保留来源身份及像素，标记为 `segmentation-only`，从表达列和固定向量中排除。[数据准备](../data/preparation.md)

## 模型、训练与评价

- 图像主干采用单一 ViT-B/8，crop 读 CLS，细胞由主干后的 Mask-Adapter 读取。实例按质心归属窗口，截断实例仍使用原完整 image 的 whole-cell 目标。Whole-cell 表达用于监督与分析。该设计以一次图像编码支持不同尺度的读出。[模型规范](../architecture/model.md) · [数据合同](../architecture/data_contract.md)
- 基座采用 DINOv2 配套视图、DINO、iBOT、KoLeo 与 whole-cell MSE，联合训练图像主干、adapter 和表达头，配套 EMA 更新与导出。基座训练后冻结 EMA 图像编码器，在开发分组训练并选取 Figure 2 M1–M4／Figure 3 S1 共用的 marker 条件化像素 decoder。M4 全局未见 marker 从基座与 decoder 训练的观测和监督中留出。[训练规范](../architecture/training.md) · [表征评价](../evaluation/representation.md)
- 训练窗口采用 224／112，验证、测试和 S4 冻结 query／gallery 导出采用 224／224 无重叠网格，末端 padding 不回移。外部模型同中心读取原生输入。图像与患者 MIL 分别使用对应范围的 CLS window bag。共同窗口与索引使任务输出及编辑前后可以对齐。[数据准备](../data/preparation.md) · [表征评价](../evaluation/representation.md)
- 预训练前固定独立患者组、内部留出和整项外部验证。患者的全部时间点、panel、图像及窗口继承同组，同患者跨来源的重复身份先解析。来源队列一般按患者作 80% 开发与 20% 内部测试，指定比较沿用符合病例隔离的作者 SPORA 划分。正式成员与协议共用 `processed/_catalog/public/experiments.json`，分母按任务独立单位计算。CRLM 使用正式名 `Lin2026_IMC_CRLM`，来源别名和稳定标识可追溯，TC／IM／PT 按 ROI 记录，TLS 保留来源描述。[数据目录](../data/catalog.md) · [表征评价](../evaluation/representation.md)
- Figure 2／3 蛋白恢复用共同 P99.5 [0,1] 域，在同一隐藏有效像素计算 Pearson r 与 MSE。空间遮盖同步隐藏全 panel，按观测掩码隐藏图像值，评分域由原始实测有效域与遮盖位置共同确定。细胞注释保留已确认的任务、细标签、固定比较方法和标注量曲线。局部组织识别按 ROI 输出，整体表型按患者输出，S4 分开报告组织、分子与临床一致性。基准仅使用符合来源和划分约束的成员。[表征评价](../evaluation/representation.md)
- 首版资源预算为 4 或 8×H20，batch、精度、优化、损失权重、EMA 和 decoder 参数属于 provisional 起点，按小规模真实训练的显存、吞吐与数值结果确认。[训练规范](../architecture/training.md) · [工程规范](../development/engineering.md)

## 临床与编辑

- Figure 4 采用 Lin 开发与 IMC_GC 独立验证，跨 panel 重叠患者在验证范围剔除。同一患者的图像继承同一分组与交叉验证折，V 使用 D 锁定的预测协议。治疗前 TRG 人群与治疗前后配对人群分别报告，比较纯化疗与化免并按患者汇总。IMC_GC 当前临床主表、重叠 OS 同步规则、队列数字和治疗编码由协议及正式来源维护。[Figure 4 协议](../paper/figure4_gastric_cancer.md)
- 先实施治疗前 TRG 二分类，保留原始等级。R 为 TRG 0／1，NR 为 2／3，阳性为 R。Figure 5 使用 D 折外锁定的 `p_R` 解释编辑响应。OS 留作后续 4e，保留随访输入时点、事件和删失条件。来源字段语义清楚时采用标准字段，未知定义保留 reported 字段和空值。[Figure 4 协议](../paper/figure4_gastric_cancer.md) · [字段模板](../data/templates/study_input/README.md)
- Figure 5 保持原人工编辑与 CellWeft 内比较，包含细胞删除或减少、CAF→T 替换、指定 marker 编辑、组合及 `p_R` 累积路径。以编辑图像和实例视图为来源，固定 q995、空间覆盖与模型配置，重新编码受影响窗口并重算患者 MIL。用匹配参照、留出患者与真实检索解释方向，生物学验证另需独立证据。[Figure 5 协议](../paper/figure5_spatial_editing.md) · [编辑规范](../architecture/editing.md)
- Figure 6／补充材料单列外部蛋白响应指导的编辑。采用 GSE278572、GSE330295 的逐细胞数据和 PRJDB16517 的作者聚合响应，按来源内对照及测量尺度解释。GSE300551 因正式数据与 vehicle 连接未闭合退出当前候选。正式来源归 NAS `raw/sc_protein/`，保留 8T 交付副本。[来源指南](../development/protein_perturbation_download_guide.md) · [扰动评价](../evaluation/perturbation.md)

## 存储、发布与维护

- SSD 保存代码、配置和研究文档，NAS 保存原始来源、正式产物、私有清单和资源。每 study 保持单份正式科学产物与权威表。公开 metadata 按稳定 study／image 键更新，临床信息和来源依据归对应元数据。纯位置或管理引用变更复用版本一致的科学验收，保留精确绑定。[存储规范](../architecture/storage.md)
- 来源和产物先持久发布并按绑定读回，再切换现行读者，最后按实际覆盖退出副本。报告、日志、快照和可重建输出在其任务与消费者闭合后清理，必要事实与依据归现有状态或对应规范。运行命令由准备脚本 README 维护。[存储规范](../architecture/storage.md) · [准备实现](../../scripts/data/preparation/README.md)
- 固定 MAP 权重、基因顺序和来源版本按配置与资源清单维护。项目使用项目本地环境，训练 tracker 默认离线或关闭。SSD 代码和文档由 Git 管理，`reference/`、私有记录及运行资产留在本地。英文 Conventional Commit 和提交范围按开发流程执行。[资源部署](../development/resources.md) · [工程规范](../development/engineering.md) · [开发流程](../development/workflow.md)
