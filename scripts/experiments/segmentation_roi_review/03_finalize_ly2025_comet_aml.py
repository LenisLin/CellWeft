"""Merge Ly2025 COMET visual review with matched source/input fidelity evidence."""

import json
from pathlib import Path


OUTPUT = Path("/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence/segmentation_followup_20261119/Ly2025_COMET_AML")
REVIEW_PATH = OUTPUT / "roi_review.json"
SUPPORT_PATH = OUTPUT / "visual_support.json"


FIRST_FOUR = {
    "TMA8_2D": {
        "judgment": "证据不足",
        "findings": (
            "窗口1中可见DAPI阳性核及对应的细胞轮廓；窗口2的DAPI与多个body通道细节较软，"
            "部分mask边界难以凭当前信号逐核确认。两窗未见明确整体XY偏移或跨越大片空白的轮廓。"
        ),
        "likely_cause": (
            "窗口2的边界证据较弱。源OME到input在映射后逐像素一致，input到image仅为q995线性归一；"
            "因此低细节更符合来源局部信号/成像质量，不支持预处理插值造成的模糊或坐标错位。"
        ),
        "next_action": "优先在原始分辨率复核窗口2；在边界仍不清楚前，不据此触发重分割。",
    },
    "TMA8_3C": {
        "judgment": "样例可用",
        "findings": (
            "两个补充窗口的可见DAPI核大多分别对应实例边界，未见明确多核合并或mask跨越暗色空隙。"
            "旧随机图库窗口1亦已按原坐标补看DAPI、5个body通道和mask overlay，边界整体同位；"
            "旧图库窗口2主要为低信号区域。CD34/CD33边界对比偏弱，但未形成明确mask错位证据。"
        ),
        "likely_cause": (
            "源OME到input及input到image均在所查像素上完全一致；通道的弱边界对比属于来源图像信号表现，"
            "不是拼图缩放导致的数据错位。"
        ),
        "next_action": "保留当前分割；旧随机窗口证据见附加图，结论仅覆盖已显示局部。",
    },
    "TMA8_4C": {
        "judgment": "样例可用",
        "findings": (
            "两个密集组织窗口中，清晰DAPI核大多有分别的实例轮廓；mask避开主要暗孔，"
            "未见明确多核合并、跨空白轮廓或整体XY偏移。若干body通道边界较软，但DAPI支持逐核判读。"
        ),
        "likely_cause": "通道图像与来源像素一致；可见的body边界柔和来自原始通道信号，不支持处理阶段平滑。",
        "next_action": "按普通样例保留；本结论仅覆盖两个补充窗口。",
    },
    "TMA8_5C": {
        "judgment": "样例可用",
        "findings": (
            "两个组织窗口中DAPI核与mask轮廓总体逐核对应，组织空隙保持未分割；"
            "未见明确合并、跨越空白或整体XY偏移。部分body通道边界较柔和。"
        ),
        "likely_cause": "源OME到input按padding映射后像素完全一致，input到image为精确线性归一；柔和边界不由预处理插值造成。",
        "next_action": "按普通样例保留；本结论仅覆盖两个补充窗口。",
    },
}


def main() -> None:
    records = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    support = json.loads(SUPPORT_PATH.read_text(encoding="utf-8"))
    support_by_roi = {record["roi"]: record for record in support}
    for record in records:
        roi = record["roi"]
        if roi in support_by_roi:
            visual = support_by_roi[roi]
            for key in ("judgment", "findings", "likely_cause", "next_action",
                        "figures", "windows_xyxy", "visual_scope"):
                record[key] = visual[key]
        else:
            record.update(FIRST_FOUR[roi])
            record["figures"] = [record["figure"]]
            record["visual_scope"] = "仅根据所示补充窗口及TMA8_3C旧随机窗判断，不代表全ROI无问题。"

        fidelity = record["source_input_image_fidelity"]
        channels = fidelity["channels"]
        exact = sum(channel["source_equals_input_on_windows"] for channel in channels)
        max_difference = max(channel["source_input_max_abs_error"] for channel in channels)
        min_correlation = min(channel["source_input_correlation"] for channel in channels
                              if channel["source_input_correlation"] is not None)
        image_error = max(channel["linear_normalization_max_abs_error"] for channel in channels)
        record["source_input_summary"] = {
            "exact_channels": exact,
            "checked_channels": len(channels),
            "max_abs_error": max_difference,
            "minimum_correlation": min_correlation,
            "input_image_max_abs_error": image_error,
            "coordinate_mapping": fidelity["source_coordinate_mapping"],
        }
        if roi in {"TMA7_1C", "TMA7_1D", "TMA8_4E", "TMA8_2D"}:
            record["likely_cause"] += (
                " 源OME到input的6个分割通道在valid_bounds映射后逐像素一致，"
                "input到image的q995线性归一最大误差为0；观察到的柔化因此已存在于来源信号。"
            )
        if roi == "TMA8_3C":
            prior = record["previous_gallery_window"]
            record["figures"] = [record["figure"], prior["figure"]]
            record["previous_gallery_window_fidelity"] = prior["source_input_image_fidelity"]

        record["window_selection_context"] = (
            "本轮两处窗口为mask/DAPI/body支持度辅助选择的补充诊断窗口；原随机图库保留。"
        )

    REVIEW_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(records)


def write_report(records: list) -> None:
    lines = [
        "# Ly2025_COMET_AML 逐ROI分割复核", "",
        "本study有16个当前分割ROI，mask来源均为`cpsam_v2` whole-cell。每张补充图按同一processed XY展示全景、两个局部窗、DAPI、5个body分割通道、实例mask和边界叠加；原随机图库未修改。",
        "补充诊断窗由mask/DAPI/body信号辅助选取，不作为原随机样本替代。每项判断只覆盖图示局部，不能据此断言全ROI无缺陷。",
        "分割输入由DAPI与CD34、CD68、CD3e、CD33、CD45RB五个body通道组成。", "",
        "| ROI | Panel/批次 | Mask来源 | 判断 | 异常类型与代表位置 | 质控图 | 下一步 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        coords = "; ".join("({},{})-({},{})".format(
            window["x0"], window["y0"], window["x1"], window["y1"])
            for window in record["windows"])
        finding = record["findings"].rstrip("。．").replace("|", "／")
        next_action = record["next_action"].replace("|", "／")
        lines.append("| {} | {} | {} | {} | {}；位置：`{}` | `{}` | {} |".format(
            record["roi"], record["panel_id"], record["mask_origin"],
            record["judgment"], finding, coords, record["figure"], next_action))

    lines.extend(["", "## 源图与显示核验", "",
                  "原始source OME与input.tiff比较使用metadata `valid_bounds`和`source.window`反映射；padding不参与源像素比较。此前5C、5E、4A、5A直用processed坐标导致的0/6是假差异；按映射后所有ROI均为6/6通道逐像素完全一致，源→input最大绝对差0、最低相关1.00000。input.tiff到image.tiff与q995线性归一一致，所有通道最大绝对误差为0。",
                  "source映射及padding信息：", "",
                  "| ROI | valid_bounds (XYXY) | padding (L,T,R,B) | 源→input | max abs | min r | input→image max abs |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for record in records:
        h, w = record["shape_yx"]
        valid = record["source_valid_bounds_xyxy"]
        padding = [valid[0], valid[1], w - valid[2], h - valid[3]]
        summary = record["source_input_summary"]
        lines.append("| {} | `{}` | `{}` | {}/{} | {} | {:.5f} | {:.2e} |".format(
            record["roi"], valid, padding, summary["exact_channels"],
            summary["checked_channels"], summary["max_abs_error"],
            summary["minimum_correlation"], summary["input_image_max_abs_error"]))

    old = next(record for record in records if record["roi"] == "TMA8_3C")["previous_gallery_window"]
    lines.extend(["", "## TMA8_3C旧随机窗", "",
                  "旧图库窗口1 `(4387,1534)-(4899,2046)`已按原坐标重新叠加实例mask和边界，图中mask含{}个实例，覆盖率{:.3f}；六个分割通道source→input逐像素一致。旧图库将512×512裁切缩放至480×480仅用于显示。附加图：`{}`。".format(
                      old["mask_labels"], old["mask_foreground_fraction"], old["figure"]),
                  "旧图库窗口2 `(529,1909)-(1041,2421)`主要为低信号区，mask近乎空白；本次优先对照了窗口1的组织内细胞和mask。",
                  "", "## 优先复核清单", "",
                  "- `TMA7_1C`、`TMA7_1D`：两个ROI的CD3e均明显较其他通道柔软、空间细节较低；DAPI与mask总体同位。原始source像素保持到input，优先核查来源CD3e通道本身及其对合并body输入的影响。",
                  "- `TMA8_4E` 窗口2：组织边缘DAPI/body对比低，局部mask正确性仍不确定；先复核原始分辨率。",
                  "- `TMA8_2D` 窗口2：DAPI/body低细节使逐核边界证据不足；先复核原始分辨率。",
                  "", "本轮窗口未发现明确全局坐标错位或跨大片组织空白的系统性mask异常；目前没有足够证据要求立即重分割。若优先复核窗口仍确认漏分/合并，再以对应ROI和原始分辨率局部为单位评估重分割。", ""])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
