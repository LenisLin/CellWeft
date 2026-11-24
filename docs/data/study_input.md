# Per-study输入整理工作流

本流程以独立image为处理单位，从数据集内容确认，经sample关联、panel对接、区域拆分、方形补边和通道筛选，交付每image的独立input.tiff及配套metadata。input.tiff保留来源强度，P99.5归一化在后续图像预处理阶段生成image.tiff。sample是一份生物学取样，image是独立ROI／TMA core／FOV；患者信息并入sample表。

字段和类型见[CSV模板](templates/study_input/README.md)，产物位置见[存储规范](../architecture/storage.md#准备产物文件布局)，实现进度见[准备入口](../../scripts/data/preparation/README.md)。

## 1. 逐步操作

### 第1步：准备来源

读取study、平台、panel范围、发布版本及来源路径，核对实际图像、归档、有序panel、core map／ROI表和临床表。查看归档成员及大小，按可用空间解压到SSD项目的`.cache/cellweft/runtime/study_input/<study>/`，复用已有完整成员并保留原归档。原始来源及作者附件归`raw`，当前来源索引和生成映射归`metadata`，必要正式核验证据归`results`；已接受结果直接复用。

整理来源版本与归档成员对应。source_files引用原始来源；source_image_id定位原始对象或完整归档成员，结合采集及层选择字段定位读入对象。复制适用模板，按字典设置类型。

确认：来源范围明确，文件与配套表可打开，归档成员可定位。

### 第2步：确认读取方式

逐文件枚举采集，记录轴、H/W、dtype、有序通道、像素尺度和强度说明。MCD逐acquisition登记；多文件通道明确文件／page对应；Z或时间轴明确所选层。未知尺度留空。

每种实际布局读取局部块确认读入方式。按来源用途识别组织与测试／校准采集，分别统计文件、采集和候选image数量；未纳入的来源对象及理由在S01核对记录中对账。

确认：每个候选image可定位到原始对象，读取方式已验证。

### 第3步：建立两级metadata

使用来源取样／标本ID连接区域表、样本表与临床表，再核对patient_key和时点。每份生物学取样分配sample_id，每个独立区域分配image_id；同一取样多图共享sample_id，同一患者多份取样共享patient_key。同一物理core的多视图共用core_id；重复发布的同一观测选定一份输入。

按[sample模板](templates/study_input/sample_metadata.csv)填写患者、取样和临床摘要。区域注释写入image的region_annotation，保留来源原值。来源只确认区域身份时保留区域ID，尚未确认的sample关联留空并说明。

来源的可选生物学区域分类互相矛盾时，`region_annotation`仍保存原始来源值，并为受影响image增加`region_annotation_status=source_conflict`及`region_annotation_evidence_path`；field_mapping定义扩展列、准确来源连接和冲突处理，证据文件记录双方来源、受影响image及禁止使用该分类的分析范围。只有已独立确认ROI／sample／patient身份、采集框及相邻sample分离，且分类矛盾不影响这些物理输入条件时，I04才可分别确认来源注释转录与区域几何。生物学分类保持unresolved，分类分层、条件采样及相关结论须排除该未决标签。冲突影响实际身份、区域边界或有效域时，I04继续阻断受影响输入；修补后须独立复查并重新绑定D01。

逐study主动核查image／core→生物学取样→患者的连接：先读本地metadata、归档附件和采集注释；公开study再核对原始发布页、论文补充临床表及作者仓库的sample/core map，下载相关小型附件到本study缓存。Private study核对本地临床表、标本清单和采集对应表。将实际查阅来源、连接键、已连接image数、sample数及patient连接覆盖写入field_mapping和M01核对记录。已有来源标本身份即可建立sample行，其他临床字段按证据填写；空sample表或未确认连接保留为明确metadata缺口，继续补查来源。

按[临床规则](templates/study_input/clinical_rules.csv)选用或扩展study字段；扩展字段沿用命名、类型、单位和时间语义规范，在field_mapping中记录定义与来源。未知可选值留空并说明。同一连接／转换规则填写一次。

2026-11-20用户确认，私有IMC_GC的当前临床主表采用`/mnt/public/lyx/clinical_master.csv`，其已填值优先于旧表，旧值与连接依据归入既有raw metadata；主表缺失的非终点原有观测可保留。OS、DFS、RFS的状态／时间按同一主表成对更新，并分别保存。治疗前资格及随访按已确认的当前记录执行，Lin 随访时间沿用来源显示精度。患者键与已有图像科学产物复用，患者／样本／来源时点变化同步更新关联表、D01及逐图metadata。完整队列表保留全部已确认患者；Figure 4记录11位跨panel重叠，IMC_GC独立分析使用41位治疗前、40位配对患者。治疗编码0为纯化疗、1为化免，两组均纳入比较。Lin中11位重叠患者的OS状态和时间按IMC_GC当前记录更新，另一位沿用其既有样本中一致的OS记录；旧值与对应关系保存在Lin原私有metadata.json的clinical_os_sync命名空间。科学分析分工及图版由[Figure 4协议](../paper/figure4_gastric_cancer.md)维护。

保存sample_metadata.csv及field_mapping.csv，建立image身份与来源记录；几何和输出信息在第5–6步补齐。

确认：主键唯一，非空sample_id可连接，连接后的image数符合来源区域数，已填值及区域注释有来源依据。

分别报告已导出image范围、sample连接覆盖和patient连接覆盖。图像／panel核对通过不等于sample信息完整；M01及D01按实际缺口记录，已确认图像可继续逐图处理。患者划分和临床分析使用已确认身份及对应字段。

临床连接与技术预处理分别判断。缺少 patient 或生物学 sample 跨图桥接时保留空值和连接缺口；图像来源、panel、采集框、有效域和物理拆分已确认的范围可独立建立技术 D01。M01 记录 metadata 连接缺口，I01–I05 检查技术输入条件；患者划分、临床分析、配对和跨图归并按所需身份依据准入。IMC_GC 的新分割已按限定范围获批使用 cpsam_v2，作者 mask 状态为 unknown 时如实保留来源状态。组织估计未经代表图核验时 sampling 保持 pending，私有身份及公开训练成员排除规则沿用[数据目录](catalog.md)。队列处理进展见[当前状态](../project/status.md)。

### 第4步：对接实际panel

按来源通道顺序登记全部来源行，source_channel_index从0编号；通道集合或顺序不同的采集使用不同panel_id。以`Study + Original marker`查询[S1](../paper/supplementary_tables.md)，别名按该study已确认的来源对应连接。

保留source_name，按[数据合同](../architecture/data_contract.md)派生marker_id，填写mapping_status和role。按S1分割标记与来源证据填写nuclear/body，核输入保留其具体测量身份。

来源作者的核通道标签与当前 S1 不一致时，单独保留来源证据并提出具体冲突；不能将 S1 空白用途或明确排除的行自动提升为当前分割输入。[已接受决定](../project/decisions.md#marker强度与空间资产)明确 Greenwald/Ranek 的组蛋白修饰候选用途仍留空，因此 H3K27me3/H3K9ac 的作者核标签不能自行改变当前 S1 或形成核组。输入取舍按现行 S1 执行，缺少获准核输入保留具体分割缺口；后续作者 whole-cell mask 的核验复用独立判断。

重复downstream marker按来源QC或测量资料确定表达通道，每panel同一marker只保留一个downstream测量。另一个raw channel若仍被选作核／膜分割输入，保留其像素并将实际role设为segmentation-only，保留S1 mapping_status、marker_id和source_name，在keep_reason说明用途。未匹配标签定位到具体来源行并完成身份核实。

确认：每个image对应实际panel，来源通道数、顺序及S1连接一致。

### 第5步：确定独立image范围

沿用来源独立ROI／FOV／core的完整采集框，保留组织与背景；包含多个sample时依据core map／区域标注拆分。拆分区域使用完整包围框，全量自动检查区域标签及框；按[study级分层抽样](study_input_review.md#3-study级分层抽样)结合来源概览确认归属、完整性及相邻sample分离。

同一生物ROI对应多个来源原生采集、而片段拼接关系尚未证实时，已独立确认来源、panel、原生像素及有效域的完整单次采集可按明确image范围进行技术处理。保留acquisition身份及其局部ROI性质，不宣称完整生物ROI覆盖、片段相邻／无重叠或新增独立生物学sample；其他有信号片段继续在S01逐项记未处理范围。hcc的OT7 acquisition 9按此规则接收764×800原生框；2026-11-05用户明确本轮仅保留该采集段，36×800的acquisition 7不纳入本轮处理，来源原件保留。不得用未经证实的拼接补齐现有input。

在原生像素坐标中记录半开框`[x0,x1) × [y0,y1)`，保持分辨率与朝向，所有通道共用同一框和padding：

```text
w = x1 - x0; h = y1 - y0; L = max(w, h)
pad_left = (L - w) // 2; pad_right = L - w - pad_left
pad_top = (L - h) // 2; pad_bottom = L - h - pad_top
x_source = x_image - pad_left + x0
y_source = y_image - pad_top + y0
```

将来源框和四侧padding写入image记录。来源框内存在未采集位置时，随图保存同尺寸uint8 valid_pixels.tiff：实测位置为1，未采集位置及padding为0；其余图像的补边由padding字段表达。

2026-11-20用户批准IMC_GC两张已复核ROI仅沿y去除既有零值方形补边，x与完整原生采集框保持。此类明确批准的输入以扩展列`canvas_policy=user_approved_native_rectangle`声明原生矩形，四侧padding均为0；该列由field_mapping定义并进入D01。input、image和mask同步裁剪，细胞坐标和完整224/112 crop对应同步更新，逐图metadata记录到旧画布的坐标变换。默认输入继续按上式补方；矩形例外不改变训练窗口的224×224尺寸。

来源原生逻辑框作为默认处理范围。2026-11-06用户明确：已有抽样确认正常核／蛋白信号、未发现具体像素或几何错误的图像正常放行；不因缺少tile／stage日志或全框实测声明另设技术阻塞。黑色或低信号本身不判为未采集，不据此制造有效域mask。Mo四图及Radtke FL4／FL6沿用已确认原生框与padding，复用已有抽样证据进入原流程；实际已知的padding、未采集区域及原有用途注释继续保留。

矩形框包含相邻sample时，结合来源区域标签／坐标及图像确认边界，保存同尺寸uint8 `sample_region.tiff`：本sample允许处理的区域为1，邻sample及padding为0。input.tiff保持原像素；image扩展列sample_region_path引用该文件，field_mapping记录边界依据。区域内可包含组织和背景；目标完整性及邻sample分离按拆分类型抽样核对，已知异常区域单独核对。后续强度统计、分割和crop同时应用sample区域与实测有效域。

Tu 以作者临床表的 `CODEX=Y` 和 `TMAMap` 连接 29 份标本与 Parent 区域。以 QPTIFF 概览和作者 Parent 细胞坐标确认边界，仅导出 `core_bounds.csv` 中已确认的区域。临床交叉表、坐标证据与限制按来源包 `raw/public/Tu2026_PhenoCycler_Neuroblastoma/metadata.json` 的 `input_facts/` 成员定位。病例连接未确认时留空，技术准入见[Tu29 限定技术准入](study_input_review.md#tu29限定技术准入2026-10-29用户批准)。

确认：每个image至多关联一个sample，目标区域完整，H=W=L，来源坐标和采集范围可恢复。

### 第6步：筛选通道并保存

保留S1已选核输入、通过panel取舍的准入marker，以及仅用于分割的重复raw channel；其余明确exclude的通道不写入。mapped和mapping_pending按准入处理，实际用途由role区分。对全部来源行填写keep_channel及理由，保留行按来源顺序分配output_channel_index=0..K-1，其余行输出位置留空。确认所有已选分割通道均可定位到input.tiff像素。

按image框分块读取并保存，同时核对整图准入marker信号。所有准入marker在整图均为0的image排除；有marker信号的组织image纳入。按实际格式、panel和拆分方式进行study级分层概览抽样；已知异常图像单独查看，新增同类图像复用既有规则和抽样证据。纳入决定按image记录到S01核对结果。

每个image独立保存无损CYX input.tiff，保持来源解码dtype、强度及像素尺度；大图按块读写BigTIFF。按output_channel_index写入标准化display_name，核输入保留具体测量名称，原始标签在source_name中保留。

公共导出器在本地TIFF及可选valid_pixels均关闭后，按正式destination绝对路径保存发布checkpoint和最后写入的complete.json。正式写名额、复制、fsync、关闭、替换、companion或重开失败时，保留完整本地文件并在原异常中给出恢复位置。同一原生SourceReader的完整SourceSpec、实际来源及有效域文件版本、shape/dtype、有序panel和tile_size完全一致时，可仅重试逐image发布；目录来源另须显式source_revision。自定义reader保留闭合结果供核对，不自动推断复用身份。来源版本读取失败不删除旧缓存；来源或科学依赖改变时重新计算，旧缓存保留至新计算闭合。读名额在逐图正式写名额前释放；必要的input及valid companion都须在各自写入句柄上按64 MiB窗口flush/fsync并成功关闭，复核完整字节数和强版本后才开始正式替换。正式头重开、来源重核及目标目录fsync全部通过后，才清理本次缓存并返回成功。checkpoint不替代整理者自检、独立复核或D01。

若正式目录已有input或valid_pixels中间文件，仅接受本地checkpoint内对应的durable.json证明：该证明必须在完整写入、同句柄fsync和成功关闭之后记录缓存及中间文件的dev/inode/size/纳秒mtime/ctime，并完成证明文件及本地目录fsync。证明缺失、不完整或版本漂移时重新复制，不以缓存读取相等授权复用。对证明有效且同大小的候选，先领取大读名额、再取得该image共享锁，按8 MiB块比较完整编码字节，并核对两端强版本稳定。释放读名额和共享锁后，按原顺序取得正式写名额及image独占锁；仅当持久证明、缓存和中间文件版本仍与完整相等证明一致时跳过复制。所有必要文件在替换前再次重核持久证明及强版本。不同、截断或变化的文件沿用复制路径；首次发布不增加完整比较。完整比较不占正式写名额，异常保留原错误、具体阶段及闭合缓存位置。旧正式产物的持久性不能由本修补或合成检查追认。

重开文件核对尺寸、dtype、通道名称与顺序，回填image记录。每study保存一份panel_mapping.csv，由panel_id区分实际panel；读取筛选后图像时使用keep_channel=true的行及output_channel_index。保存完整images.parquet、sample_metadata.csv和field_mapping.csv，路径相对NAS项目根目录。

确认：每个image的来源、sample、panel和TIFF可相互定位；来源数、纳入数及排除理由可对账。

### 第7步：独立校对、修补和验收

整理者自检并关闭输出文件，提交来源范围和产物。独立复核者按[校对与修补流程](study_input_review.md)执行检查，主控核实差异后派发修补，整理者修补后由独立复核者复查，最后主控验收。

确认：交付已确认的image、对应panel、两级metadata及review_results.csv，汇总实际数量和核对范围。

## 2. 交付位置

```text
/mnt/NAS_21T/ProjectData/CellWeft/processed/<study>/
  images.parquet
  sample_metadata.csv
  panel_mapping.csv
  field_mapping.csv
  review_results.csv
  images/<image_id>/
    input.tiff
    valid_pixels.tiff     # 来源框内存在未采集位置时保存
    sample_region.tiff    # 方形来源框需要区分相邻sample时保存
```

输入验收后，按[完整操作与逐步确认](preparation.md#4-完整操作与逐步确认)开展共享强度、mask、细胞定量、邻域、crop及发布；study向量在通道筛选后的独立CPU步骤导出。后续处理沿用完整image/sample metadata，保留来源坐标和实测有效域。

## 3. Subagent任务参数

每个study对应`scripts/data/preparation/`中的一个独立执行脚本；已有入口复用，其余以study ID命名。脚本负责来源解释、sample连接、panel、拆分及发布，共用读写行为放在src/cellweft。分批执行合并本study既有索引，复用已确认且输入未变化的图像。一个study同一时间由一个整理者写入；独立复核在该study交付关闭后执行。

```yaml
study: <稳定study标识>
platforms: [<纳入平台>]
panels: [<本批panel范围>]
releases: [<纳入发布版本>]
source_locations:
  images_or_archives: [<来源图像或归档>]
  ordered_panels: [<有序panel资料>]
  region_sample_tables: [<区域与sample对应表>]
  clinical_tables: [<临床来源表>]
output_root: processed/<study>/
extraction_root: .cache/cellweft/runtime/study_input/<study>/  # 相对SSD项目根目录
workflow: docs/data/study_input.md
templates: docs/data/templates/study_input/
```
