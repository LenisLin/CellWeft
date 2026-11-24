# 分割 ROI 复核

当前公共实验成员维护在NAS `processed/_catalog/public/experiments.json`，最终质控决定、112幅小图尺寸/检查及8幅padding验证统一维护在同目录的`quality_control.json`。小图PNG与联系图固定保存在`quality_control/small_fov/`，不按轮次另存完整清单或重复报告。

`13_check_small_fov_crops.py`重新检查全部短边不足224像素的图像，并原位更新上述质控记录和PNG；保留用户决定，不自动批准sampling。执行：`PYTHONPATH=src scripts/resources/cellpose-sam.sh scripts/experiments/segmentation_roi_review/13_check_small_fov_crops.py`。临时组织估计文件使用本机临时目录，结束即清理。

`16_consolidate_quality_control.py`用于本次单次归并：默认打印精确清理范围，经实际gpt-6.2-sol/medium独立复核后以`--apply --review-file <临时意见文件> --plan-file <已审清单>`执行；保留科学资产、现有HTML及其直接引用附件。01–10和14–15为此前单次复核/发布脚本，其已退役过程文件不再作为当前入口，不再运行旧发布或HTML导出流程。原图库及五study的逐ROI图像观察仍保留于`evidence/segmentation_roi_review_20261119/`和`evidence/segmentation_followup_20261119/`。

IMC_GC 的私有人工确认包由 `11_export_imc_gc_user_review.py` 和 `imc_gc_review_template.html` 生成，位于 `/mnt/NAS_21T/ProjectData/CellWeft/results/imc_gc_user_review_20261119/index.html`。复用既有8张跨panel代表图，另附1张已排除空信号ROI；提供整图轮廓开关、两个224×224原生像素局部和4张已有组织估计叠图。R01/R02优先确认组织范围。页面批量接受仅作用于8张代表图的分割反馈，组织范围和排除处置独立选择；导出的JSON绑定源文件版本，页面不会改写正式结果。私有ROI清单和图像仅保存于忽略的SSD缓存及NAS结果目录。

用户决定由 `12_apply_imc_gc_user_decisions.py --decisions <私有JSON> --stage` 生成有SSD备份的裁剪候选，确认全零补边及保留像素后以 `--publish` 发布。只去除已批准的y人工补边，保持x和原生采集框；更新input/image/mask、细胞坐标与完整crop对应，复用同一统计域的q995、表达和邻域。矩形输入由canvas_policy和field_mapping显式声明；D01及保留范围逐图回执重新绑定，临床表和向量保持。原8图及排除图的判断通过 `applied_user_decisions.json` 记录，组织窗口筛选独立保持pending。执行使用现有 `.venv-data`，设置 `PYTHONPATH=src`。
