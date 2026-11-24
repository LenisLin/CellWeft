"""Merge reviewed Iglesia ROI records into the study evidence bundle."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


STUDY = "Iglesia2024_CODEX_BreastCancer"
EVIDENCE_ROOT = Path(
    "/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence"
    "/segmentation_followup_20261119"
)
STUDY_DIR = EVIDENCE_ROOT / STUDY


FIRST24_NOTES: dict[str, tuple[str, list[str], str, str]] = {
    "HT365B1_S1H1__20230221": (
        "样例可用",
        ["DAPI核信号清楚，mask轮廓大体围绕单个细胞核；PanCK及其他蛋白通道可辨。", "当前窗口未见明显整体XY错位或轮廓跨越大片组织空白。"],
        "当前视野未见可归因于坐标错位的证据。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT397B1-S1H3A1-U22__20230316": (
        "样例可用",
        ["DAPI较清楚，PanCK呈组织岛状信号；mask与可见细胞核大体同位。", "未见明显多核合并、大片漏分或整体XY错位。"],
        "当前视野支持通道与mask坐标相合。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT397B1-S1H3A1-U2__20230315": (
        "样例可用",
        ["DAPI核与PanCK组织信号清晰；彩色实例轮廓大体逐核对应。", "未见明显整体坐标偏移或跨组织空白的轮廓。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT397B1-S1H3A1-U31__20230407": (
        "样例可用",
        ["DAPI及可用蛋白通道信号可辨；多数mask实例与单个核位置相合。", "当前窗口未见清楚的整体XY错位或明显跨空白轮廓。"],
        "当前视野未见明确坐标错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT110B1-S1H4__20221216": (
        "样例可用",
        ["细胞密集，DAPI可辨，Vimentin组织信号明显而PanCK较稀疏；mask总体围绕可见核。", "当前窗口没有显示大范围漏分或整体错位。"],
        "蛋白通道呈现组织/细胞类型差异，未见证据表明由坐标偏移造成。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT243B1-S1H4__20221216": (
        "样例可用",
        ["DAPI核和PanCK相关组织结构可辨；实例轮廓大体覆盖单核细胞。", "未见明显整体XY错位或跨大块无组织区的mask。"],
        "当前视野支持mask与DAPI位置相合。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT305B1-S1H1__20221216": (
        "样例可用",
        ["DAPI与PanCK信号及细胞轮廓可辨；实例mask大体逐核分布。", "当前视野未见整体错位或明显异常外扩。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT397B1-S1H3A1-U22__20230413": (
        "样例可用",
        ["DAPI及PanCK信号清楚，mask大体围绕组织内的单个细胞核。", "未见明显整体XY错位。"],
        "当前视野支持通道和mask坐标相合。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT480B1-S1H2__20220915": (
        "样例可用",
        ["DAPI核和PanCK组织结构较清晰，mask轮廓大体与可见核逐个对应。", "未见明显大范围漏分、跨空白轮廓或整体坐标偏移。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT480B1-S1H2__20220917": (
        "样例可用",
        ["DAPI和PanCK信号清楚；实例轮廓总体覆盖相应单核细胞。", "当前视野未见明显整体XY错位。"],
        "当前视野支持mask与DAPI位置相合。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT206B1-H1__20220625": (
        "样例可用",
        ["DAPI及PanCK信号较强，mask大体围绕可见单核细胞。", "当前窗口未见清楚的整体坐标错位。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT243B1-S1H4A4__20230420": (
        "局部问题",
        ["DAPI及部分蛋白结构偏软，原始input.tiff中对应信号也偏软；mask与可辨细胞核大体同位。", "局部清晰度限制细胞边界判断，但未见明确整体XY错位。"],
        "较支持原始成像/通道清晰度限制；没有观察到处理造成的整体坐标错位。",
        "优先核查原始通道成像质量；若需判断边界准确率，另选更清晰视野复核。",
    ),
    "HT308B1-S1H5A4__20230420": (
        "样例可用",
        ["DAPI和PanCK组织区及实例轮廓清楚，mask多数与单核位置相合。", "未见明显整体XY错位或大范围跨空白轮廓。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "_HT271B1-S1H6A5__20230420": (
        "样例可用",
        ["DAPI、PanCK及mask轮廓均可辨，多数实例围绕对应细胞核。", "未见明显整体坐标偏移或跨组织空白的轮廓。"],
        "当前视野支持通道和mask坐标相合。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT323B1-A1__20220210": (
        "样例可用",
        ["DAPI及PanCK组织信号可见，mask总体围绕细胞核分布。", "当前窗口未见明确整体XY错位或异常跨空白轮廓。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT323B1-H1A4__20220210": (
        "样例可用",
        ["DAPI核及蛋白通道可辨；mask轮廓大体逐核对应。", "未见明显整体坐标偏移。"],
        "当前视野支持mask与细胞核相合。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT323B1-H3__20220211": (
        "样例可用",
        ["DAPI、PanCK及mask均显示组织内细胞结构；轮廓大体与单核细胞同位。", "当前视野未见明显整体错位。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT171B1-S1H9A1-4_left__20230512": (
        "局部问题",
        ["DAPI/PanCK结构可辨，mask大体围绕细胞核；CD68可见横向条带。", "同XY的input.tiff中可见相同CD68条带，支持条带来自源通道/采集而非后处理引入；未见整体mask错位。"],
        "CD68条带更支持原始输入/采集通道异常；不支持坐标错位。",
        "优先标记CD68源通道质量；不建议仅凭该条带重分割mask。",
    ),
    "HT171B1-S1H9A1-4_right__20230512": (
        "局部问题",
        ["DAPI及mask基本同位；PanCK出现横向条带，且同XY input.tiff中存在相同结构。", "条带附近蛋白边界判读受限，未见mask整体平移。"],
        "PanCK条带更支持原始通道/采集伪影，不支持mask坐标错位。",
        "优先复核PanCK源通道质量；无需仅据此重分割。",
    ),
    "HT243B1-S1H4A4_left__20230512": (
        "局部问题",
        ["DAPI和mask在可见细胞区大体同位；CD68上缘可见低幅横向条带。", "同XY input.tiff具有对应条带形态，未见整体坐标偏移。"],
        "更支持CD68源通道存在条带；未见坐标错位证据。",
        "优先核查CD68源通道；无需仅凭该条带重分割。",
    ),
    "HT243B1-S1H4A4_right__20230512": (
        "局部问题",
        ["DAPI核及mask总体同位；CD68上缘有低幅条带，input.tiff同位置可见相似结构。", "未见清楚整体XY错位。"],
        "更支持源CD68通道条带，而非处理引入的坐标偏移。",
        "优先核查CD68源通道；无需仅凭该条带重分割。",
    ),
    "HT271B1-S1H6A5_left__20230512": (
        "局部问题",
        ["DAPI和mask大体逐核对应；PanCK有局部高亮团块，形态在同XY input.tiff中相同。", "未见mask随高亮区域整体偏移。"],
        "亮团块更支持源通道强度异常；当前窗口不支持坐标错位。",
        "优先检查PanCK源通道高亮区域；无需仅据此重分割mask。",
    ),
    "HT271B1-S1H6A5_right__20230512": (
        "样例可用",
        ["DAPI较密，实例边界大体对应单个核；未见清楚整体错位或跨空白轮廓。", "当前窗口支持局部mask与核信号同位。"],
        "当前视野未见明确mask错位。",
        "无需基于此窗口优先重分割；结论仅限所示局部。",
    ),
    "HT308B1-S1H5A4_left__20230512": (
        "局部问题",
        ["DAPI/PanCK及mask在可见组织区大体同位；CD68上部有明显横向条带。", "同XY input.tiff中有相同条带，支持原始通道/采集伪影；未见整体mask坐标偏移。"],
        "CD68条带更支持源图像/采集通道异常，不支持mask错位。",
        "优先核查CD68源通道质量；无需仅凭该条带重分割。",
    ),
}


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    source = read_json(STUDY_DIR / "roi_review.json")
    first24 = read_json(STUDY_DIR / "roi_review_first24.json")
    support = read_json(STUDY_DIR / "visual_support.json")["roi_reviews"]
    replacements = read_json(STUDY_DIR / "replacement_windows.json")

    by_id = {row["roi"]: row for row in source}
    if len(by_id) != 48 or len(first24) != 24 or len(support) != 24:
        raise ValueError("Expected 48 source rows, 24 first-pass reviews, and 24 support reviews")
    if set(row["roi"] for row in first24) & set(row["roi"] for row in support):
        raise ValueError("Review groups overlap")
    if set(by_id) != {row["roi"] for row in first24 + support}:
        raise ValueError("Review ROI membership does not match the 48-row source list")

    for reviewed in first24:
        row = by_id[reviewed["roi"]]
        row.update(reviewed)
        try:
            judgment, findings, cause, action = FIRST24_NOTES[row["roi"]]
        except KeyError as exc:
            raise ValueError(f"Missing first24 visual judgment: {row['roi']}") from exc
        row["judgment"] = judgment
        row["findings"] = findings
        row["likely_cause"] = cause
        row["next_action"] = action
        row["window_limitation"] = "结论仅适用于图示512×512窗口，不能据此判定整个ROI无缺陷。"

    for reviewed in support:
        row = by_id[reviewed["roi"]]
        row.update(
            window_xyxy=reviewed["window_xyxy"],
            judgment=reviewed["judgment"],
            findings=reviewed["findings"],
            likely_cause=reviewed["likely_cause"],
            next_action=reviewed["next_action"],
            window_limitation=reviewed["window_limitation"],
            figures=[reviewed["figure"]],
        )

    replacement_by_id = {row["roi"]: row for row in replacements}
    replacement_notes = {
        "HT565B1-H2__20230427": (
            "局部问题",
            ["替代窗口显示组织和mask；可见实例大体与DAPI细胞位置相合。", "CD68横向条带在同XY input.tiff中同样出现；未见整体mask错位。", "原窗口信号不足，已改用DAPI与mask均非零的窗口。"],
            "CD68条带更支持源通道/采集异常；未见坐标错位证据。",
            "优先核查CD68源通道；无需仅因条带重分割mask。",
        ),
        "HT339B1-H1A1__20220302": (
            "样例可用",
            ["替代窗口有清晰DAPI细胞及PanCK组织区，mask轮廓大体围绕细胞核。", "原窗口信号不足，替代窗口提供了可判读的局部证据；当前未见明显整体错位。"],
            "当前替代视野支持mask与DAPI位置相合。",
            "无需基于此窗口优先重分割；结论仅限替代视野。",
        ),
        "HT171B1-S1H9A1-A4_left__20230526": (
            "局部问题",
            ["替代窗口可见细胞和mask，但DAPI、PanCK及其他通道均偏软；同XY input.tiff也呈相似模糊。", "mask大体与可辨细胞核同位，未见整体坐标平移；细边界判断受源图像清晰度限制。", "原窗口信号不足，已改用DAPI与mask均非零的窗口。"],
            "更支持原始成像/通道清晰度限制；未见处理造成的整体错位。",
            "优先复核原始图像清晰度；如需评估mask细边界，另选清晰组织窗口。",
        ),
        "HT243B1-S1H4A4_left__20230526": (
            "样例可用",
            ["替代窗口DAPI及组织信号可辨，mask多数与细胞核同位。", "原窗口信号不足，替代窗口提供可判读证据；未见明确整体XY错位。"],
            "当前替代视野支持mask与DAPI位置相合。",
            "无需基于此窗口优先重分割；结论仅限替代视野。",
        ),
        "HT271B1-S1H6A5_middle__20230526": (
            "局部问题",
            ["替代窗口DAPI/PanCK及mask可判读，轮廓大体逐核对应。", "CD68下缘横带在同XY input.tiff中出现，限制该通道局部判读；未见整体mask错位。", "原窗口信号不足，已改用DAPI与mask均非零的窗口。"],
            "CD68横带更支持源通道/采集表现；没有坐标错位证据。",
            "优先核查CD68源通道；mask当前无需因该条带重分割。",
        ),
        "HT308B1-S1H5A4_left__20230526": (
            "样例可用",
            ["替代窗口可见细胞组织和mask，轮廓总体围绕DAPI核。", "原窗口信号不足，替代窗口提供可判读证据；未见明显整体坐标偏移。"],
            "当前替代视野支持mask与DAPI位置相合。",
            "无需基于此窗口优先重分割；结论仅限替代视野。",
        ),
        "HT308B1-S1H5A4_right__20230526": (
            "样例可用",
            ["替代窗口DAPI和组织信号可辨，mask大体逐核对应。", "原窗口信号不足，替代窗口显示可判读组织；未见明显整体XY错位。"],
            "当前替代视野支持mask与DAPI位置相合。",
            "无需基于此窗口优先重分割；结论仅限替代视野。",
        ),
        "HT339B2-H2_right__20230701": (
            "样例可用",
            ["替代窗口含较亮、较密组织，DAPI细胞及mask均可见；多数实例轮廓总体贴合细胞簇。", "原窗口信号不足，替代窗口提供可判读证据；高密度边界仍应谨慎解读。"],
            "当前替代视野未见明确整体错位；密集区边界准确率尚不能逐细胞确认。",
            "如用于边界精度判断，优先放大高密度细胞簇复核；当前证据不支持系统性重分割。",
        ),
    }

    for roi, replacement in replacement_by_id.items():
        row = by_id[roi]
        previous_window = row["window_xyxy"]
        previous_figure = row["figures"][0]
        row["previous_window_xyxy"] = previous_window
        row["previous_figure"] = previous_figure
        row["window_xyxy"] = replacement["window_xyxy"]
        row["window_dapi_signal_fraction"] = replacement["window_dapi_fraction_gt_0_2"]
        row["window_mask_foreground_fraction"] = replacement["window_mask_foreground_fraction"]
        row["figures"] = replacement["figures"]
        judgment, findings, cause, action = replacement_notes[roi]
        row["judgment"] = judgment
        row["findings"] = findings
        row["likely_cause"] = cause
        row["next_action"] = action
        row["window_limitation"] = "结论仅适用于图示替代512×512窗口，不能据此判定整个ROI无缺陷。"

    extra_figures = {
        "HT171B1-S1H9A1-4_left__20230512": "figures/source_compare/HT171B1-S1H9A1-4_left__20230512__CD68.jpg",
        "HT171B1-S1H9A1-4_right__20230512": "figures/source_compare/HT171B1-S1H9A1-4_right__20230512__PanCK.jpg",
        "HT243B1-S1H4A4_left__20230512": "figures/source_compare/HT243B1-S1H4A4_left__20230512__CD68.jpg",
        "HT243B1-S1H4A4_right__20230512": "figures/source_compare/HT243B1-S1H4A4_right__20230512__CD68.jpg",
        "HT271B1-S1H6A5_left__20230512": "figures/source_compare/HT271B1-S1H6A5_left__20230512__PanCK.jpg",
        "HT308B1-S1H5A4_left__20230512": "figures/source_compare/HT308B1-S1H5A4_left__20230512__CD68.jpg",
        "HT308B1-S1H5A4_right__20230512": "figures/source_compare/HT308B1-S1H5A4_right__20230512__CD68_source_check.jpg",
        "HT297B1-S1H1_right__20230701": "figures/source_compare/HT297B1-S1H1_right__20230701__CD3e_source_check.jpg",
        "HT565B1-H2__20230427": "figures/source_compare/HT565B1-H2__20230427__CD68_reselected.jpg",
        "HT171B1-S1H9A1-A4_left__20230526": "figures/source_compare/HT171B1-S1H9A1-A4_left__20230526__CD68_reselected.jpg",
        "HT243B1-S1H4A4_left__20230526": "figures/source_compare/HT243B1-S1H4A4_left__20230526__CD68_reselected.jpg",
        "HT271B1-S1H6A5_middle__20230526": "figures/source_compare/HT271B1-S1H6A5_middle__20230526__CD68_reselected.jpg",
        "HT308B1-S1H5A4_left__20230526": "figures/source_compare/HT308B1-S1H5A4_left__20230526__CD68_reselected.jpg",
        "HT308B1-S1H5A4_right__20230526": "figures/source_compare/HT308B1-S1H5A4_right__20230526__CD68_reselected.jpg",
        "HT339B1-H1A1__20220302": "figures/source_compare/HT339B1-H1A1__20220302__CD68_reselected.jpg",
    }
    for roi, figure in extra_figures.items():
        by_id[roi]["figures"].append(figure)

    rows = [by_id[row["roi"]] for row in source]
    expected = {"样例可用", "局部问题", "疑似系统性问题", "证据不足"}
    if len(rows) != 48 or len({row["roi"] for row in rows}) != 48:
        raise ValueError("Final review must contain 48 unique ROI IDs")
    if any(row["judgment"] not in expected for row in rows):
        raise ValueError("Unexpected judgment value")
    if any("待逐图复核" in " ".join(row["findings"]) for row in rows):
        raise ValueError("Placeholder finding remains")

    (STUDY_DIR / "roi_review.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    counts = Counter(row["judgment"] for row in rows)
    review_targets = [
        row["roi"] for row in rows if row["judgment"] in {"局部问题", "疑似系统性问题"}
    ]
    report = [
        f"# {STUDY} 分割复核",
        "",
        "共逐ROI查看48个512×512窗口，核对同坐标DAPI、可用蛋白通道及实例mask。所有判断只适用于图示窗口。8个原低信号窗口已在`valid_bounds`内改选DAPI和mask均有信号的窗口，并保留旧窗口坐标以供追溯。",
        "",
        f"- 样例可用：{counts['样例可用']}；局部问题：{counts['局部问题']}；疑似系统性问题：{counts['疑似系统性问题']}；证据不足：{counts['证据不足']}。",
        "- 主要局部发现：若干ROI的CD68或PanCK存在横向条带/亮区，且同XY `input.tiff` 可见相同表现；HT171B1-S1H9A1-A4_left/right 的通道整体偏软，原始输入也同样模糊。现有证据更支持源图像/通道清晰度或采集表现，不支持预处理造成的整体坐标错位。",
        "- mask方面：多数所见实例与DAPI细胞核大体同位；高密度ROI `HT565B1-S1H2_right__20230701` 中局部轮廓拥挤，可能有挤压或相邻实例边界不清，建议优先放大复核。当前抽查未显示可复现的系统性XY偏移；抽查窗口不能证明未展示区域无缺陷。",
        "",
        "## 建议优先复核",
        "",
        "；".join(review_targets),
        "",
        "源通道条带/模糊ROI应优先核查原始成像质量；仅当边界复核发现mask合并、拆分或漏分时再考虑重分割。完整逐ROI字段与图像路径见`roi_review.json`。",
    ]
    (STUDY_DIR / "report.md").write_text("\n".join(report) + "\n")
    print(f"rows={len(rows)} counts={dict(counts)}")
    print(f"priority_roi_count={len(review_targets)}")
    print(f"json={STUDY_DIR / 'roi_review.json'}")
    print(f"report={STUDY_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
