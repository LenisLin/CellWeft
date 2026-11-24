"""Finalize the manual Bandyopadhyay CODEX ROI review records."""

import json
from pathlib import Path


OUTPUT_ROOT = Path(
    "/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence/"
    "segmentation_followup_20261119/Bandyopadhyay2024_CODEX_BoneMarrow"
)
REVIEW_PATH = OUTPUT_ROOT / "roi_review.json"


REVIEWS = {
    "AML2_183_Scan1": {
        "judgment": "局部问题",
        "findings": "w1组织区DAPI与多数mask大致相配，mask多贴近核；w2 (5144,0)-(5656,512) DAPI近黑、CD144为零且其他蛋白信号弱，仍有多个mask实例；w3近空白。",
        "likely_cause": "当前证据指向源mask与同坐标原始图像内容不一致，或局部源mask假阳性；原始DAPI与当前输入/归一化吻合，不能归因于显示缩放或通道抽取。",
        "next_action": "优先配对核对作者mask与native QPTIFF的原始坐标/帧映射，重点复核w2；暂不据此直接重分割。",
        "window_notes": ["有效组织中多数核有对应轮廓；边界偏核周。", "DAPI/蛋白极弱但存在多个mask，是主要异常。", "DAPI最高候选仍接近空白，不作通过证据。"],
    },
    "H36_Scan1": {
        "judgment": "样例可用",
        "findings": "w1密集DAPI核大多对应mask，未见明显整体平移；w2/w3较稀疏。清晰区mask边界普遍贴近核缘，小于Vimentin/CD45/CD144可见范围。",
        "likely_cause": "局部图像-mask位置大体相符；whole_cell标签与实际轮廓范围存在语义疑点，需与其他同源mask一并确认。",
        "next_action": "确认作者mask的whole_cell定义及边界生成方式，再决定是否满足整细胞分割用途。",
        "window_notes": ["密集组织中DAPI与mask大体对应，轮廓偏核周。", "稀疏边缘窗，证据有限。", "近空窗，不作通过证据。"],
    },
    "H33_Scan1": {
        "judgment": "局部问题",
        "findings": "w1可见细胞簇且mask大体围绕DAPI，但有明显弥散亮斑及横向接缝；w2 (7424,480)-(7936,992) DAPI近黑、CD45/CD144仅边缘有信号，中心仍出现细长mask；w3接近空白。",
        "likely_cause": "w2更支持局部源mask/图像对应或假阳性问题；native QPTIFF、input及归一化像素一致，显示层不能解释该差异。w1另有原图接缝/弥散信号。",
        "next_action": "配对检查w2 native QPTIFF与作者mask坐标，并检查w1接缝两侧通道；暂缓重分割。",
        "window_notes": ["DAPI-mask整体相配，但有弥散亮斑和横向接缝。", "弱DAPI区域出现细长mask，是主要异常。", "近空窗，不作通过证据。"],
    },
    "AML3_1443_Scan1": {
        "judgment": "证据不足",
        "findings": "w1 DAPI与mask大致对应；w2 DAPI低对比、明显发虚且Vimentin背景较高；w3接近空白。",
        "likely_cause": "图像清晰度/背景限制了边界判断；当前局部证据不足以分辨源图像模糊与mask边缘问题。",
        "next_action": "先对照native QPTIFF检查DAPI清晰度和背景，再决定是否需要复核mask。",
        "window_notes": ["组织区DAPI与mask大致对应。", "DAPI发虚、低对比，不能可靠判读边界。", "近空窗，不作通过证据。"],
    },
    "H27_Scan1": {
        "judgment": "样例可用",
        "findings": "w1密集组织中DAPI与mask大体对应，未见明显整体平移；其余窗较稀疏。清晰轮廓偏核周，明显小于广泛的Vimentin/CD45/CD144信号。",
        "likely_cause": "样本内位置匹配可接受；whole_cell语义与边界范围尚未验证，且表现与多个同源ROI相似。",
        "next_action": "将whole_cell语义作为同源作者mask的系统性确认项；未确认前不要把边界范围解释为全细胞。",
        "window_notes": ["密集区核-mask大体对应，边界偏核周。", "稀疏组织窗，证据有限。", "近空窗，不作通过证据。"],
    },
    "H39_Scan1": {
        "judgment": "局部问题",
        "findings": "w1少量可见细胞与mask大体相配；w2 (7904,240)-(8416,752) DAPI及所示蛋白通道均缺少清晰细胞信号，却有3个较大的椭圆mask；w3近空白。",
        "likely_cause": "更支持局部源mask与当前原始图像不匹配或mask假阳性；native DAPI像素与输入/归一化相符，不支持显示缩放造成。",
        "next_action": "配对核查w2的native QPTIFF和原始作者mask坐标/帧映射；再判断是源mask问题还是坐标框架不一致。",
        "window_notes": ["少量DAPI与mask大体对应。", "弱信号区域含多个较大mask，是主要异常。", "近空窗，不作通过证据。"],
    },
    "AML1_191_Scan1": {
        "judgment": "样例可用",
        "findings": "w1密集DAPI区多数核有对应mask，未见整体错位；w2/w3近空或稀疏。密集区mask多贴近核缘。",
        "likely_cause": "所见组织区位置对应；轮廓偏核周的现象与其他作者源mask重复，whole_cell语义仍需统一确认。",
        "next_action": "先确认源mask语义；本轮未覆盖足够组织以判断整幅ROI完整性。",
        "window_notes": ["密集DAPI区mask大体对应、轮廓偏核周。", "低信号/近空窗，不作通过证据。", "近空窗，不作通过证据。"],
    },
    "H32_Scan1": {
        "judgment": "样例可用",
        "findings": "w1密集组织中DAPI与mask大体对应，未见明显整体偏移；Vimentin/CD45信号范围普遍大于mask轮廓。w2/w3为空或低信号。",
        "likely_cause": "局部位置匹配尚可；whole_cell语义/边界范围是同源mask的重复疑点。",
        "next_action": "核实作者mask边界语义后再确认其是否可作整细胞监督。",
        "window_notes": ["密集组织核与mask大致对应，边界偏核周。", "近空窗，不作通过证据。", "低信号窗，不作通过证据。"],
    },
    "H35_Scan1": {
        "judgment": "样例可用",
        "findings": "w1密集组织中多数DAPI核有对应mask；未见明显整体偏移。mask形状贴近核缘，周围Vimentin/CD45/CD144范围更广；w2/w3信号很少。",
        "likely_cause": "局部图像-mask位置大体一致；边界更接近核周而不是明显的全细胞外缘，whole_cell标签需核实。",
        "next_action": "纳入同源mask语义确认；当前局部组织未见优先坐标错位证据。",
        "window_notes": ["密集组织核-mask大体相配，边界偏核周。", "图像与mask均近空。", "DAPI候选窗实际信号很弱，不作通过证据。"],
    },
    "AML3_1329_Scan2": {
        "judgment": "样例可用",
        "findings": "有效组织w1/w2中DAPI与mask大致对应，mask边界偏核周；w3空白。旧gallery两窗分别为y=1111–1623和778–1290，均低于metadata声明的有效图像起点y=2040，位于顶部padding，空白符合padding而非错位证据。",
        "likely_cause": "有效窗未见整体XY平移；旧gallery空窗由坐标落padding解释。作者mask whole_cell语义仍有跨ROI疑点。",
        "next_action": "旧gallery窗不用于分割质量判断；若需评估该ROI，继续在valid_bounds内检查组织区域并核实mask语义。",
        "window_notes": ["有效组织区DAPI与mask大致对应，边界偏核周。", "有效边缘细胞簇有对应mask，外围稀疏核不完整。", "位于有效图像起始边缘但全通道空白。", "旧gallery坐标落顶部padding，黑窗是无效抽样。", "旧gallery坐标落顶部padding，黑窗是无效抽样。"],
    },
    "H38_Scan1": {
        "judgment": "样例可用",
        "findings": "w1组织清晰，DAPI与mask总体同心；mask轮廓贴近核缘，小于Vimentin/CD45信号覆盖。其他窗为空或组织边缘。",
        "likely_cause": "图像-mask局部位置大体对应；whole_cell语义需结合源mask说明复核。",
        "next_action": "纳入同源mask语义确认；本轮未发现支持全局错位的局部证据。",
        "window_notes": ["清晰组织区DAPI-mask大体对应，边界偏核周。", "近空窗，不作通过证据。", "组织边缘/低信号。"],
    },
    "H14_Scan1": {
        "judgment": "局部问题",
        "findings": "主组织窗中DAPI与部分实例对应，mask多数集中在核附近且覆盖小于蛋白信号；另一组织窗可见部分DAPI没有对应轮廓。",
        "likely_cause": "局部漏分/覆盖不足较可疑；未见足以证明全局坐标平移的证据。",
        "next_action": "优先复核有DAPI而缺轮廓的局部及其邻域，确认作者mask是否遗漏弱核/边缘细胞。",
        "window_notes": ["主组织区DAPI与部分mask对应，边界偏核周。", "有局部DAPI未见轮廓，需复核。", "低信号窗，不作通过证据。"],
    },
    "H41_Scan1": {
        "judgment": "证据不足",
        "findings": "多个抽样窗信号很弱；其中一个窗出现mask，但同位置未见清晰DAPI及其他细胞标记，无法判断是真实弱信号细胞还是mask异常。",
        "likely_cause": "图像信号不足，显示局部无法排除低强度/背景影响；没有足够证据确定分割或坐标原因。",
        "next_action": "先核对native输入的DAPI强度与该窗对应的作者mask，再决定是否升级复核。",
        "window_notes": ["弱信号并有mask，缺少清晰DAPI佐证。", "低信号组织窗。", "弱信号窗，不能判定mask正确性。"],
    },
    "H26_Scan1": {
        "judgment": "局部问题",
        "findings": "w1密集组织中DAPI与mask大体对应，边界偏核周；w2边缘窗出现数枚细长mask，同坐标DAPI近黑。w3以有效组织边缘/空区为主。",
        "likely_cause": "w2支持局部mask与图像不一致或源mask假阳性；没有证据支持全局偏移，需与native图像作坐标配对。",
        "next_action": "配对复核w2 native DAPI与原始作者mask；同时核实该来源mask的whole_cell定义。",
        "window_notes": ["主组织区核-mask大体对应，边界偏核周。", "弱DAPI窗有细长mask，是主要异常。", "组织边缘/近空，证据有限。"],
    },
    "H37_Scan1": {
        "judgment": "局部问题",
        "findings": "w1密集组织区mask多紧贴DAPI核；w2/w3近空。旧gallery窗1 (18141,12046)-(18653,12558)几乎无DAPI/mask；旧gallery窗2 (13574,2126)-(14086,2638)有少量DAPI，只有部分有轮廓，窗内多个孤立DAPI无mask。",
        "likely_cause": "局部漏分/覆盖不足较可疑；已对齐的细胞说明并非简单全局平移。whole_cell标签与核周轮廓的语义差异是同源mask的系统性疑点。",
        "next_action": "优先复核旧gallery窗2及相邻区域的孤立DAPI是否确属漏分；再确认源mask的细胞边界语义。",
        "window_notes": ["密集组织核与mask大体相配，但边界偏核周。", "近空窗。", "DAPI候选窗实际近黑，不作通过证据。", "旧gallery窗1接近空白且在valid_bounds内，属于无组织抽样。", "旧gallery窗2有稀疏DAPI但部分核无对应轮廓。"],
    },
    "H10_Scan1": {
        "judgment": "样例可用",
        "findings": "w1主组织区信号清楚，DAPI与mask大致对应；mask边界偏核周，Vimentin/CD45/CD144范围较宽。其他窗信号稀少。",
        "likely_cause": "局部图像-mask位置基本吻合；全细胞语义仍未由当前边界外观证实。",
        "next_action": "确认作者mask语义后再用于整细胞监督；当前样本窗未见明显坐标错位。",
        "window_notes": ["清楚组织区DAPI与mask大致相配，边界偏核周。", "近空窗，不作通过证据。", "稀疏组织窗，不能代表整幅ROI。"],
    },
    "NSM2_1086_Scan1": {
        "judgment": "证据不足",
        "findings": "组织区DAPI明显发虚；Vimentin背景噪声强且可见拼接带；另一窗有单个mask但未见清楚DAPI。",
        "likely_cause": "原始图像模糊/噪声或局部显示质量限制判断；当前证据不足以区分通道质量问题与mask局部异常。",
        "next_action": "先检查native QPTIFF的DAPI清晰度、Vimentin背景和拼接位置，再复核单mask窗口。",
        "window_notes": ["DAPI发虚、Vimentin高背景，边界判读受限。", "单个mask缺少清楚DAPI对应。", "组织边缘/低信号。"],
    },
    "NSM3_1996_Scan1": {
        "judgment": "样例可用",
        "findings": "w1左侧组织边缘DAPI密集，mask多数对应单个DAPI核；Vimentin/CD45较广，轮廓偏核周。w2/w3低信号或近空。",
        "likely_cause": "主组织局部位置大体匹配；whole_cell与核周轮廓的语义仍需确认。",
        "next_action": "纳入同源mask语义确认；本次近空窗口不代表其他ROI范围。",
        "window_notes": ["组织边缘DAPI与mask多数对应，轮廓偏核周。", "近空窗。", "DAPI候选窗只有少数弱亮点，不作通过证据。"],
    },
    "NSM1_1720_Scan2": {
        "judgment": "局部问题",
        "findings": "w1 (15944,0)-(16456,512) 几乎无DAPI、CD45/CD144无清楚细胞信号，却在图像顶部有数个细长/大片mask；w2全空，w3只有弱DAPI亮点且无mask。",
        "likely_cause": "源mask与原始图像局部对应/标注可信度可疑；native QPTIFF DAPI、input与现有归一化像素完全相符，显示或通道抽取不能解释。",
        "next_action": "配对检查w1坐标处的native QPTIFF和作者mask原始帧/坐标；确认mask是否落在无细胞区域。",
        "window_notes": ["DAPI/蛋白信号弱而有多个大片mask，是主要异常。", "全通道空白，无mask。", "仅有弱DAPI点，无可见mask。"],
    },
    "AML1_382_Scan2": {
        "judgment": "证据不足",
        "findings": "当前mask来源为cpsam_v2；官方作者mask因20540×5573与当前native QPTIFF尺寸不符且无可恢复变换而未复用。三个窗口均无清晰细胞群，DAPI/Vimentin呈弥散噪声，mask极稀疏或为空。",
        "likely_cause": "当前图像局部信号/组织可见度不足，且可配对作者mask不可用；无法凭这些窗判断CPSAM分割表现或mask坐标。",
        "next_action": "先核对native QPTIFF和输入边界/强度，定位可见组织后再评估cpsam_v2；不要把当前空窗判作分割通过。",
        "window_notes": ["DAPI/Vimentin弥散噪声，仅见极少mask。", "顶部窗全通道近黑。", "DAPI候选窗仍无清晰细胞或mask。"],
    },
}


NATIVE_DAPI_CHECKS = {
    "AML2_183_Scan1": {
        "bounds_xyxy": [5144, 0, 5656, 512], "native_qptiff_page": 0,
        "source_to_input_exact": True, "normalized_to_input_q995_exact": True,
        "input_crop_max": 8, "input_crop_p99": 5.0, "q995": 88.0,
    },
    "H33_Scan1": {
        "bounds_xyxy": [7424, 480, 7936, 992], "native_qptiff_page": 0,
        "source_to_input_exact": True, "normalized_to_input_q995_exact": True,
        "input_crop_max": 18, "input_crop_p99": 6.0, "q995": 69.0,
    },
    "H39_Scan1": {
        "bounds_xyxy": [7904, 240, 8416, 752], "native_qptiff_page": 0,
        "source_to_input_exact": True, "normalized_to_input_q995_exact": True,
        "input_crop_max": 9, "input_crop_p99": 6.0, "q995": 67.0,
    },
    "NSM1_1720_Scan2": {
        "bounds_xyxy": [15944, 0, 16456, 512], "native_qptiff_page": 0,
        "source_to_input_exact": True, "normalized_to_input_q995_exact": True,
        "input_crop_max": 7, "input_crop_p99": 5.0, "q995": 107.0,
    },
}


def finalize() -> None:
    records = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    if {record["roi"] for record in records} != set(REVIEWS):
        raise ValueError("Review annotations must cover exactly the current ROI inventory")

    for record in records:
        review = REVIEWS[record["roi"]]
        if len(review["window_notes"]) != len(record["windows"]):
            raise ValueError("Window notes do not match figures for {}".format(record["roi"]))
        record.update({key: value for key, value in review.items() if key != "window_notes"})
        record["reviewer"] = "人工逐窗视觉复核；对4处低DAPI异常窗追加native QPTIFF像素配对检查"
        record["reviewed_all_windows"] = True
        record["systemic_semantic_risk"] = (
            "作者源mask metadata标whole_cell；多个独立ROI的清晰组织窗中轮廓反复贴近DAPI核，"
            "明显小于可见胞体/膜标记范围。该重复现象支持whole-cell边界语义需系统性确认，"
            "但当前证据不足以断言所有源mask为核分割。"
            if record["mask_origin"] == "source" else None
        )
        for window, note in zip(record["windows"], review["window_notes"]):
            window["visual_review_note"] = note
            check = NATIVE_DAPI_CHECKS.get(record["roi"])
            if check and window["bounds_xyxy"] == check["bounds_xyxy"]:
                window["native_dapi_frame_check"] = check
        record["native_dapi_frame_checks"] = (
            [NATIVE_DAPI_CHECKS[record["roi"]]] if record["roi"] in NATIVE_DAPI_CHECKS else []
        )

    REVIEW_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {}
    for record in records:
        counts[record["judgment"]] = counts.get(record["judgment"], 0) + 1
    source_exact = [window for record in records if record["mask_origin"] == "source"
                    for window in record["windows"] if window["source_mapping_exact"] is True]
    figures = [figure for record in records for figure in record["figures"]]

    lines = [
        "# Bandyopadhyay2024 CODEX Bone Marrow 分割局部复核",
        "",
        "## 范围与证据",
        "",
        "复核当前20个ROI的 {} 个独立512×512窗口；每窗逐一显示DAPI、Vimentin、CD45、CD144、DAPI+mask轮廓及彩色实例mask。".format(len(figures)),
        "判断只针对所列窗口，不代表整幅ROI已全面验证。所有mask原始产物保持未修改。",
        "",
        "19个source mask在所选窗口与其metadata指定的作者源mask裁片像素完全一致（{}个窗口）。这只证明源mask映射/裁切复用正确，不证明它与图像生物学对齐。".format(len(source_exact)),
        "对AML2_183_Scan1、H33_Scan1、H39_Scan1和NSM1_1720_Scan2的mask/弱DAPI异常窗，原始QPTIFF DAPI页→input.tiff像素完全相同，image.tiff也与input/q995完全相同。因此这些异常不是本次通道选择或归一化显示造成；仍需在native图像与作者mask间核实原始帧/坐标对应。",
        "多个独立ROI的清晰组织窗都显示mask轮廓紧贴DAPI核周，而metadata将作者mask标为whole_cell；可见胞体/膜信号范围通常更广。该重复现象构成whole-cell语义/边界范围的系统性风险，但不能仅凭窗口断言源mask的正式定义是核分割。",
        "",
        "## ROI判断",
        "",
        "| ROI | mask来源 | 判断 | 主要观察 | 可能原因 | 下一步 | 证据图 |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        figure_links = " ".join("[w{}](./{})".format(i + 1, figure)
                                for i, figure in enumerate(record["figures"]))
        cells = [record["roi"], record["mask_origin"], record["judgment"],
                 record["findings"], record["likely_cause"], record["next_action"], figure_links]
        lines.append("| " + " | ".join(str(value).replace("|", "／").replace("\n", " ")
                                           for value in cells) + " |")

    lines.extend([
        "",
        "## 结论与优先项",
        "",
        "判断计数：" + "；".join("{} {}个".format(key, value) for key, value in counts.items()) + "。",
        "",
        "优先配对核查的局部问题：AML2_183_Scan1、H33_Scan1、H39_Scan1、H26_Scan1、H14_Scan1、H37_Scan1、NSM1_1720_Scan2。重点复核DAPI弱/无信号处的mask，以及H37旧gallery次窗中未被轮廓覆盖的孤立DAPI。",
        "图像质量/证据不足优先项：AML3_1443_Scan1、H41_Scan1、NSM2_1086_Scan1、AML1_382_Scan2。先核native QPTIFF信号和清晰度；AML1_382_Scan2当前用cpsam_v2，官方作者mask因尺寸不符且缺少精确变换而未复用。",
        "跨ROI优先确认：source mask的whole_cell标记是否与可见边界一致。当前复核未发现可支持全study统一XY平移的证据；局部DAPI-mask不一致更适合先做原始坐标/帧配对检查，暂不直接重分割。",
        "",
        "## 输出",
        "",
        "- 结构化记录：`roi_review.json`",
        "- 独立局部证据图：`figures/`（{}张）".format(len(figures)),
        "- 复核脚本：`scripts/experiments/segmentation_roi_review/02_bandyopadhyay_visual_review.py` 与 `03_finalize_bandyopadhyay_review.py`",
        "",
    ])
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("finalized", len(records), "ROIs", len(figures), "figures", counts)


if __name__ == "__main__":
    finalize()
