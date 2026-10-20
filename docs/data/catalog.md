# 数据来源与训练准入

本页维护来源角色、训练就绪层级和患者划分。公开数据收集见[collection.md](collection.md)，链接、许可与历史核验记录见[参考索引](../references.md)。

## 1. 公共来源

| 来源 | 角色与接入要求 |
| --- | --- |
| spora（R12） | 主入口；逐 cohort 检查图像、marker metadata、可用 masks/annotations 和访问要求。 |
| HTAN（R13） | 补充癌种与临床背景，按 multiplex imaging assay 筛选，不把全部组学记录计作图像。 |
| HuBMAP（R14） | 补充组织和 CODEX 等成像数据，逐 dataset 核对文件、尺度及 segmentation。 |
| 原始研究仓库 | 补充独特 panel 或高质量图像；按[收集规范](collection.md)确认 study 覆盖并补齐缺失内容。 |

spora 的历史规模统计由[参考索引](../references.md) R12 维护；该记录不代表本地数据量或本次重新核验结果。

## 2. 训练就绪层级

| 层级 | 必需输入 | 可参与阶段 |
| --- | --- | --- |
| Image-ready | 符合准备规范的共享蛋白图像、marker 映射、尺度信息 | 图像基座 |
| Cell-ready | 上述输入及通过检查的 whole-cell masks、固定 CellTable/坐标、marker measurements、top-20 邻域 | 细胞基座 |
| Joint-ready | Cell-ready 且图像/细胞视图严格对齐、crop/实例索引和所需检查完成 | 联合预训练 |
| Annotated | 任一层级及细胞/组织/患者标签 | 独立 benchmark 或下游任务 |

以上是模型训练准入，不是processed保存的前置清单。数据准备阶段完成来源marker清理；全部数据准备后统一补齐共享bank映射，再判断训练所需marker输入是否完整。已完成的图像、mask、表达和索引可先发布，不因bank后置重复处理像素。

已配准且强度语义可追溯的图像可接入，不要求原始仪器文件；收集完成不代表已满足上述准备层级。已有 whole-cell mask 按[一次检查流程](preparation.md)优先复用，核 mask 不替代 whole-cell mask。Mask 或定量缺失时保留可用图像阶段，不伪造 cell targets；mask 与索引完成也不等于蛋白定量完成。

## 3. 私有医院与划分

首轮不自动纳入九家医院；约两家用于独立应用，候选分别为大规模/panel 完整数据及具有治疗前后和临床资料的数据。
至少一整家医院不参与预训练和适配；另一家可作有限适配或探索，具体分工按授权、规模和资料完整度锁定并写入 manifest。
以 pseudonymous patient keys 管理 patient-level split，纵向同一患者全部时间点同组。公共来源跨门户去重后划分，避免 train/test 泄漏。同一图像的全部重叠crop继承该患者/图像的split，不按crop独立随机划分。
保留原始 treatment/timepoint；不使用药物条件不代表所有观测都属于治疗前。实际私有清单与数据保存在 NAS，不上传患者直接标识或公共 tracker。
准备步骤见[preparation.md](preparation.md)，字段合同见[data_contract.md](../architecture/data_contract.md)。
