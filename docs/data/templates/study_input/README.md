# Study输入CSV模板

字段维护源为本目录CSV，工作流见[输入整理](../../study_input.md)，独立核对见[校对流程](../../study_input_review.md)。[Excel副本](../study_input_metadata_template.xlsx)使用相同字段、字典和词表，便于查看与填写；修订字段时同步两种格式。

## 文件职责

| 模板／规则 | 实际保存与用途 |
| --- | --- |
| [image_metadata.csv](image_metadata.csv) | 字段模板；逐image完整metadata实际保存为`processed/<study>/images.parquet` |
| [sample_metadata.csv](sample_metadata.csv) | 一份生物学取样一行；患者信息并入同表 |
| [panel_mapping.csv](panel_mapping.csv) | 来源panel的全部通道；记录S1状态、核／膜组、取舍和输出通道位置 |
| [field_mapping.csv](field_mapping.csv) | 来源字段、连接键、分类映射、转换和冲突处理规则 |
| [review_results.csv](review_results.csv) | 独立复核的实际逐项结果 |
| [data_dictionary.csv](data_dictionary.csv) | 字段名称、层级、类型、单位、必填性与来源规则 |
| [value_sets.csv](value_sets.csv) | 分类值与含义 |
| [clinical_rules.csv](clinical_rules.csv) | 从临床案例吸收的整理规则与核对要求 |
| [review_checks.csv](review_checks.csv) | 独立核对项、覆盖范围及直接影响的阶段 |

模板数据表只含表头；字典、词表及规则表包含定义。sample表沿用`universal_clinical_cohort_metadata_template.xlsx`的字段风格，重组为生物学取样宽表，按study实际来源选用可选列。

2026-11-02：M01使用`metadata_linkage`，连接缺失保留unresolved和依赖用途限制，临床信息不阻断独立确认的图像技术处理；I01–I05及panel检查继续判断具体输入条件。CSV和Excel副本保持一致，完整规则见[独立校对](../../study_input_review.md)。

D01 包含独立确认、定向修复及复查；当前无法修复项须经用户批准后删除排除。CSV 与 Excel 副本的 `review_checks` 保持一致，逐图交付及验收字段见[执行与验收合同](../../preparation.md#执行与验收合同)。

## 填写顺序

1. 按data_dictionary选择字段并设置类型。ID按字符串读取并保留前导零，未知值留空，真实0按来源填写；CSV使用UTF-8 BOM，Parquet可空字段使用可空类型。
2. 先连接sample与image。主键分别为(study,sample_id)、(study,image_id)；image用sample_id关联生物学取样，用panel_id关联实际panel。region_annotation保存来源区域注释原值。
   可选生物学分类有来源冲突时，另以image扩展列region_annotation_status和region_annotation_evidence_path显式记录冲突与证据，保留原注释。定义及准确连接规则写入field_mapping；适用条件及I04复核范围见[输入流程第3步](../../study_input.md#第3步建立两级metadata)。未决分类不得用于分类分层或条件采样。
3. Sample按来源选用字段，依[clinical_rules.csv](clinical_rules.csv)扩展所需列；扩展字段的定义、类型、单位和时点记录于field_mapping。MCD填写source_acquisition_id，Z／时间层选择填写相应索引。
   Image可保留已定义的扩展列；sample_region_path为可空字符串、NAS相对路径，引用同尺寸sample区域TIFF。sample信息整理须追查来源连接，按M01报告实际行数及image/sample/patient覆盖；空表记录metadata缺口。
4. 每study填写一份panel_mapping，保留全部来源行。source_channel_index按来源从0编号；保留行的output_channel_index从0连续编号，其他行该字段留空。source_channel_count记录来源数，channel_count记录TIFF保存数。
5. 来源连接及转换规则在field_mapping填写一次。source_files在CSV中保存JSON数组文本、在Parquet中保存字符串列表；value_mapping保存JSON对象文本。实际数据路径相对NAS项目根目录。
6. Image记录保存为images.parquet，其余表按上表文件名保存到NAS。提交后按[独立校对与修补](../../study_input_review.md)完成验收；details记录修补轮次、动作和复查结论。
7. 输入通道筛选后，独立向量步骤补充embedding_row。项目内模板维护字段和规则，实际患者及样本记录保存到NAS。


## 临床来源字段保全（2026-11-19）

临床整理须对照原始工作簿／附件的字段全集，不能以已精简的中间CSV代替原始来源核对。字段命名遵循`clinical_rules.csv`的C03：已有语义匹配的字段采用`data_dictionary.csv`名称；扩展属性使用含义明确的英文`lower_snake_case`，定义、类型、单位、时点、来源和中英文值映射统一记录在`field_mapping.csv`。不另建逐study字段字典或列迁移表。样本／患者属性不得以`clinical_source_<原字段>`代替规范命名；单位、编码体系或时点尚未确认的原值使用明确的`reported`属性，不伪装成已标准化终点。cTNM与ypTNM分别保存；TRG原值与其评分体系分别定义。原始`NA`等未知哨兵在来源metadata中原样保存，临床摘要未知值留空，不能算作已知临床分期或有效终点。

患者临床信息按既定两级metadata并入`sample_metadata.csv`。当前两个胃癌队列另外保留一份精简`patient_metadata.csv`以承接已冻结患者全集：IMC_GC的111个患者键尚不能全部由当前39份样本表达，不得为套用表格而虚构sample或丢弃患者。患者表不再重复保存Parquet或数百个来源记录槽；仅保存有定义的患者属性、必要的来源时点摘要和定位引用。主表字段及临床文本使用英文，原始中英文记录、来源行、旧宽表和必要核验证据合入该study既有的私有`raw/.../metadata.json`，正式目录不再散放来源副表、字段字典、迁移表或多轮版本报告。药物原文的英文渲染保留剂量、日期和未确认的简称／商品名，不据文本翻译推定药物身份。样本层只接收有依据的样本／病理号和来源时点连接；冲突摘要留空，原始各值保留于同一来源归档，不擅取第一行。

临床字段扩展且输入图像、panel及文件资产不变时，按现有`same_processing_binding`判断能否复用像素产物；独立复核通过后更新D01的元数据版本，保留既有图像范围，不因临床列扩展重跑分割或定量。私有临床值和成员表继续保存到NAS，不写入版本库。
