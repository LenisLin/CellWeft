# 已接受的项目决定

日期：2026-10-18。依据为本轮用户确认与现有研究规划；此页维护决定及理由，详细合同由链接文档维护。

| 决定 | 依据与归属 |
| --- | --- |
| V1 研究 marker、单细胞和多细胞输入删除后的目标细胞与微环境响应 | 本轮讨论确认；研究定义见 [research_plan.md](research_plan.md) |
| SSD 保存代码、工作计划和讨论约束，NAS 保存真实数据、大型中间结果、表征及权重 | 用户指定两处存储用途；详见 [storage.md](../architecture/storage.md) |
| 采用目录先行的初版架构，按需实现组件，不填充空文件 | 用户授权本轮目录构建；机器目录清单为 `configs/layout.json` |
| SSD 使用 GitHub 管理，NAS 不加入 Git | 用户指定 `LenisLin/CellWeft`；保留已有 main 初始提交和 LICENSE |
| 整个 reference 目录不加入 GitHub 管理 | 用户本轮明确要求；`.gitignore` 排除整个目录 |
| 提交标题与正文使用英文，标题为 type(scope): description | 用户明确要求英文标题；正文按独立文件批次说明，详见 [workflow.md](../development/workflow.md) |
| 当前规划文档迁入 docs；初始化 Prompt 作为历史规格归档 | 已确认架构的文档承接方案；归档指令不构成当前执行授权 |

现有模型设计约束继续以研究计划为准。本轮没有改变模型宽度、训练目标、冻结策略或临床数据用途。
