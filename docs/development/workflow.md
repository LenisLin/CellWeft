# 开发与 GitHub 工作流

## 开发交接

1. 读取 [status.md](../project/status.md) 和相关现行规范，确定本次范围与可观察验收条件。
2. 检查已有实现与用户修改；按授权实施，保存大型产物到 NAS。
3. 运行最窄的有效验证；记录实际通过、失败、跳过及未验证内容。
4. 更新所属规范、必要的决策依据和状态；将结果链接到实际运行产物。
5. 检查 Git diff、暂存清单和忽略规则，再提交；在授权发布范围内推送并核对远程提交。

目录结构由 `configs/layout.json` 定义。从新 checkout 恢复空目录：

```bash
python3 scripts/bootstrap_layout.py
python3 scripts/bootstrap_layout.py --data-root /mnt/NAS_21T/ProjectData/CellWeft
```

第二条命令同时创建 NAS 基础目录。脚本仅创建目录，可重复运行。当前脚本要求 Python 3.10 或更新版本，模型环境尚未建立。

## 提交约定

标题使用英文 Conventional Commit：`type(scope): description`。类型可用 `feat`、`fix`、`docs`、`refactor`、`test`、`build`、`ci`、`chore`、`perf`。scope 使用简短英文领域，如 `layout`、`data`、`model`、`training`、`edits`、`evaluation`。

正文使用英文。同一提交包含不同文件批次时，每批分别列出文件/目录、变化及目的；验证单独列出。独立变更优先拆分提交；同一目标的跨文件变更可以合并并分组解释。

```text
chore(layout): initialize the SSD and NAS project structure

Documentation batch:
- docs/project/: organize the research plan, decisions, and current status.
- docs/architecture/: define module boundaries and artifact ownership.

Configuration and tooling batch:
- configs/layout.json and scripts/bootstrap_layout.py: restore empty directories.
- configs/paths.example.yaml: document machine-specific storage roots.

Repository policy batch:
- AGENTS.md and .gitignore: define collaboration and local-only assets.

Validation:
- Verify all declared directories exist and repeated creation is idempotent.
- Check local document links and the staged file list.
```

使用明确文件清单暂存，审阅 `git diff --cached` 和 `git diff --cached --check`。多行 commit message 写入临时文件，用 `git commit -F` 提交，保留真实换行。`reference/` 整体不提交，也不添加例外。

## 远程管理

远程为 `https://github.com/LenisLin/CellWeft.git`，主分支 `main`。保留远程原始 LICENSE 和历史，不强推。后续发布前检查远程是否前进，发生分歧先审阅并整合，不覆盖他人提交。完成推送后核对远程 main 与本地提交。

NAS 不使用 Git；不要将 NAS 作为子模块或以数据软链接加入暂存区。本机认证配置不写入仓库。网络设置仅在确需时对单条命令覆盖，不修改全局设置。
