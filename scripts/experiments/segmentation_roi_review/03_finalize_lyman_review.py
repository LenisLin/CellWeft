"""Apply visual findings to the generated Lyman ROI evidence inventory."""

import json
from collections import Counter
from pathlib import Path


OUTPUT = Path("/mnt/NAS_21T/ProjectData/CellWeft/processed/_catalog/public/evidence/segmentation_followup_20261119/Lyman2025_IMC_PanIN")


def item(judgment, findings, cause="当前窗口未见明显分割异常", action="保留现有mask；如需全ROI结论，再检查未抽查区域"):
    return {
        "judgment": judgment,
        "findings": findings,
        "likely_cause": cause,
        "next_action": action,
    }


REVIEWED = {
    "lyman_01_a001": item("样例可用", "高核信号窗口中，轮廓大体逐核对应；窗口内未见明显合并或错位。"),
    "lyman_01_a002": item("样例可用", "窗口内核信号与实例轮廓总体同位，密集处仍可辨认逐核实例。"),
    "lyman_01_a003": item("样例可用", "抽查窗口中实例边界围绕核信号分布，未见跨越明显无信号空白。"),
    "lyman_01_a004": item("样例可用", "核信号与mask边界总体一致；该局部未见连续多核并入单实例的明显迹象。"),
    "lyman_01_a005": item("样例可用", "窗口内多数核信号各有对应实例轮廓，未见整体平移。"),
    "lyman_01_a006": item("样例可用", "所见组织窗口的实例边界与核信号大体对应；源视野较方形图像窄，按源尺寸补边。"),
    "lyman_02_a001": item("样例可用", "实例轮廓与窄条形组织中的核信号大体同位；有效组织限于原生66×1000 px acquisition条带，周边为padding。", "原始acquisition为1000×66 px，方形输出由空白padding形成；非重采样或坐标错位", "保留mask；分析时按实际采集条带解释，不把padding计入组织"),
    "lyman_02_a002": item("样例可用", "高信号窗口内核与实例边界总体对应，未见明显跨空白轮廓。"),
    "lyman_02_a003": item("样例可用", "窗口内实例分离与核信号大体相符；未见整体偏移或连续多核合并。"),
    "lyman_02_a004": item("样例可用", "核信号和实例边界大体同位；输出较源视野略宽的部分为padding。"),
    "lyman_03_a001": item("样例可用", "高密度组织及腺样空腔周围的轮廓总体贴合单核分布，未见跨越明显腔隙。"),
    "lyman_03_a002": item("样例可用", "抽查局部中实例轮廓围绕核信号，未见系统性坐标偏移。"),
    "lyman_03_a003": item("样例可用", "密集窗口内可见逐核轮廓，组织间隙未见被大范围连成实例。"),
    "lyman_03_a004": item("样例可用", "窗口中实例轮廓与核信号总体一致；未见明显拆分或合并异常。"),
    "lyman_04_a001": item("样例可用", "抽查窗口中核信号与实例边界大体同位。"),
    "lyman_04_a002": item("样例可用", "密集细胞窗口内边界总体逐核分隔，未见整片轮廓错位。"),
    "lyman_04_a003": item("样例可用", "实例边界与局部核信号大体对应；未见明显跨组织空隙连接。"),
    "lyman_04_a004": item("样例可用", "窗口内核信号和实例边界总体同位；较长输出尺寸由源视野padding解释。"),
    "lyman_09_a001": item("样例可用", "中等密度窗口中实例轮廓总体逐核分布，未见明显错位。"),
    "lyman_09_a002": item("样例可用", "高密度组织与腺样结构边缘可见逐核轮廓；未见轮廓跨越明显黑色腔隙。"),
    "lyman_09_a003": item("样例可用", "高密度核信号区域内实例大体逐核分开，未见显著整体偏移。"),
    "lyman_09_a004": item("样例可用", "暗色腺样空隙周围轮廓沿组织细胞分布，空隙本身未见被大块mask填充。"),
    "lyman_09_a005": item("样例可用", "组织团块及边缘处轮廓总体随单核分布，未见跨越较大无信号间隙。"),
    "lyman_09_a006": item("样例可用", "高密度窗口内核信号与实例边界总体对应；局部密集但未见成片错位。"),
    "lyman_09_a007": item("样例可用", "密集窗口中轮廓总体逐核，组织空白区域未见明显跨越。"),
    "lyman_09_a008": item("样例可用", "不规则组织边缘处轮廓跟随核阳性细胞，空白间隙基本未被连成实例。"),
    "lyman_09_a009": item("样例可用", "高密度区域内实例边界总体与核信号同位，未见系统性偏移。"),
    "lyman_09_a010": item("样例可用", "致密且形态不规则的组织窗口仍可见逐核轮廓，未见明显大范围合并。"),
    "lyman_10_a001": item("证据不足", "原生50×50 px图中mask与少量核信号粗略同位；单像素尺度及稀疏边缘信号不足以判断细胞边界、拆分或合并。", "MCD acquisition原生仅50×50 px，pixel size为1 µm；input/image/mask保持该尺寸，显示放大不能恢复细节", "核实该50×50 acquisition是否应保留；不要据此单图要求重分割，需更高分辨率来源才能判断边界"),
    "lyman_10_a002": item("证据不足", "原生50×50 px窗口中部分实例包围核阳性斑块，但边界细节不足以判定细胞分割准确性。", "原始采集尺寸为50×50 px而非预处理缩小；近邻显示只保留像素，不增加信息", "核实小视野用途；边界质量需更高分辨率图像或人工标注确认"),
    "lyman_10_a003": item("证据不足", "原生50×50 px图中可见少数核信号及对应mask标签，当前像素细节不足以区分单核轮廓与实例拆分/合并。", "原始MCD acquisition为50×50 px；input、image、mask尺寸一致", "核实源采集是否为预期视野；不以插值显示图作细胞边界结论"),
    "lyman_10_a004": item("证据不足", "原生50×50 px图显示稀疏核信号和相应mask，核/胞质边界不可可靠辨认。", "原始MCD acquisition为50×50 px；50×50表示采集视野较小，不能由此判定空间分辨率降低；未发现处理阶段重采样", "需更高分辨率源数据或人工核对后判断是否重分割"),
    "lyman_10_a005": item("样例可用", "窗口内核信号与实例轮廓大体对应；全视野为上下两侧组织条带和中间空隙，mask随组织带分布且未跨越中央空隙。", "MCD原生1050×550 px视野居中padding至1050×1050；图中上下黑区是padding，非模糊或位移", "保留现有mask；分析时排除padding和未采集空白"),
    "lyman_10_a006": item("样例可用", "密集核阳性区域的实例边界总体逐核对应，未见明显错位。"),
    "lyman_10_a007": item("样例可用", "高密度窗口内轮廓大体围绕单个核信号，未见大范围合并。"),
    "lyman_10_a008": item("样例可用", "稀疏且形态较细长的组织区中实例与核阳性结构大体对应，背景间隙基本保持为空。"),
    "lyman_10_a009": item("样例可用", "中等密度局部的实例边界总体贴合核信号分布。"),
    "lyman_10_a010": item("样例可用", "高密度窗口中轮廓总体逐核，未见明显跨空白连接。"),
    "lyman_11_a001": item("样例可用", "密集组织局部的核信号与实例轮廓总体同位。"),
    "lyman_11_a002": item("样例可用", "窄带组织内核和实例轮廓大体对应，条带之外的空白区域与mask一致。", "MCD原生800×191 px条带居中padding至800×800；黑区与所有通道和mask共享", "保留现有mask；按实际191 px采集高度分析"),
    "lyman_11_a003": item("样例可用", "腺样环形空腔周围可见逐核轮廓，未见轮廓跨过主要空腔。"),
    "lyman_11_a004": item("样例可用", "不规则组织边缘和细胞团块中的轮廓总体随核信号分布，未见整体错位。"),
    "lyman_11_a005": item("样例可用", "较稀疏、细长细胞区内仍可见实例与核信号大体一一对应。"),
    "lyman_12_a001": item("局部问题", "抽查窗口和全视野可见多条横向黑带贯穿核、body通道及mask；黑带处没有mask，邻近组织处轮廓大体对应。", "更符合原图/采集中的横向缺失带或拼接空带；不是图像与mask坐标错位", "先核对对应MCD原始通道的横带；定量分析排除缺失带。当前证据不支持因该条带单独重分割"),
    "lyman_12_a002": item("样例可用", "局部核信号与实例总体同位；细胞间存在较多组织空隙，mask大体限于核阳性组织。"),
    "lyman_12_a003": item("样例可用", "异质密度与较细长细胞区域中，实例边界总体随单核信号分布。"),
    "lyman_13_a001": item("样例可用", "核信号与mask轮廓在窄条形组织区域总体同位；源条带之外为padding。", "MCD原生1000×284 px采集视野补边成方形；不是预处理下采样", "保留mask；使用时只分析原生有效采集范围"),
    "lyman_13_a002": item("样例可用", "致密组织局部轮廓总体逐核，未见整体平移或跨越明显组织间隙。"),
    "lyman_13_a003": item("局部问题", "x=[96,352)、y=[96,352)窗口中，大片191Ir/193Ir双核信号缺少实例；邻近分散核仍有轮廓。", "局部漏分候选；团块内可分辨细胞数和边界仍需核实，当前不支持整体XY错位", "优先复核该窗口的核信号团块及同来源邻近ROI；确认真实细胞后局部修复或重分割"),
    "lyman_13_a004": item("样例可用", "较密集局部核与mask边界总体同位；未见明显多个核并成大片实例。"),
    "lyman_13_a005": item("样例可用", "中等密度窗口的实例边界总体贴合核信号。"),
    "lyman_13_a006": item("样例可用", "组织窗口中实例边界总体随核信号分布；上下各42 px为方形padding。", "源视野1000×916 px居中补边至1000×1000；input到image归一化误差极小，不支持坐标偏移", "保留mask；分析时排除上下padding；单个局部窗口不能证明全ROI无问题"),
    "lyman_05_a001": item("样例可用", "密集组织窗口中核对象仍分别有轮廓，未见明显多核合并、大片漏分或图像与mask错位。"),
    "lyman_05_a002": item("样例可用", "不规则腺样空隙及邻近细胞处，核通道与轮廓同位；空隙周边轮廓随单个可见细胞分布。"),
    "lyman_05_a003": item("样例可用", "高密度且有不规则组织间隙；多数可见细胞核各有实例轮廓，未见明显整体偏移或反复大块合并。"),
    "lyman_06_a001": item("样例可用", "局部覆盖细长组织/腔隙边界；轮廓沿边界两侧核对象分布，未见明显坐标偏移。"),
    "lyman_06_a002": item("样例可用", "密集混合组织和腺样结构中，核对象大体各有对应轮廓，图像与mask在窗口内一致。"),
    "lyman_06_a003": item("样例可用", "密集细胞区与不规则间隙内，核对象和轮廓大体同位，未见明显跨空隙连接。"),
    "lyman_06_a004": item("样例可用", "高拥挤度区域的轮廓围绕可见核信号分布；拥挤限制细边界判断，但未见明显整体错位或连续多核实例。", "高细胞密度使局部边界不易区分", "当前不需额外复核；若下游依赖该形态的精细细胞边界，再查看其他局部窗口"),
    "lyman_06_a005": item("样例可用", "致密且细长核对象大体分别有轮廓，mask随信号分布，未见明显XY偏移。"),
    "lyman_06_a006": item("样例可用", "密度变化及腺样空隙区域内，实例轮廓大体对应可见核对象，未见明显空间错位。"),
    "lyman_06_a007": item("样例可用", "活动图像区域内核信号和轮廓一致；上、下空白padding与mask共享。", "1200×1000源视野显示在1200×1200画布上，异常方形显示来自补边，未见窗口级错位", "无需分割复核；解释显示时保留源图与画布尺寸差异"),
    "lyman_06_a008": item("证据不足", "原始与显示尺寸均为50×50 px；最近邻放大后核、蛋白通道和mask仍为粗像素块，无法可靠评价单细胞分离及边界位置。", "所示核/边界信号不足以验证细胞边界；50×50为视野尺寸，pixel size为1 µm；当前图像未显示可确认的坐标错位", "需更高分辨率源图或独立参考标注后再接受或修改该ROI mask"),
    "lyman_06_a009": item("证据不足", "原始与显示尺寸均为50×50 px；当前核和细胞边界细节不足以区分正确单细胞mask与合并/拆分。", "所示核/边界信号不足以验证细胞边界；50×50为视野尺寸，pixel size为1 µm；无法据此确认具体对齐故障", "需更高分辨率源图或独立参考标注后再接受或修改该ROI mask"),
    "lyman_06_a010": item("证据不足", "原始尺寸50×50 px；放大面板中核与轮廓呈像素块，不能验证单细胞边界。", "所示核/边界信号不足以验证细胞边界；50×50为视野尺寸，pixel size为1 µm；无法据此确认具体对齐故障", "需更高分辨率源图或独立参考标注后再接受或修改该ROI mask"),
    "lyman_06_a011": item("证据不足", "原生50×50 px中可见粗略对象与轮廓，但无法判断单细胞边界位置及核与实例是否一一对应。", "所示核/边界信号不足以验证细胞边界；50×50为视野尺寸，pixel size为1 µm；无法据此确认具体对齐故障", "需更高分辨率源图或独立参考标注后再接受或修改该ROI mask"),
    "lyman_07_a001": item("样例可用", "密集且纹理不规则的窗口内，轮廓大体对应单个核对象并与通道同位，未见明显偏移。"),
    "lyman_07_a002": item("样例可用", "腺样结构周围细长核的轮廓沿局部核信号分布，未见明显XY偏移或重复多核合并。"),
    "lyman_07_a003": item("样例可用", "活动图像区域内轮廓沿核信号分布；上下空白带与mask一致。", "1200×1000源视野补边后显示在1200×1200画布，未见窗口级坐标错位", "保留源尺寸与画布尺寸说明；无需针对padding重分割"),
    "lyman_07_a004": item("样例可用", "圆形腺样结构周围的单个核清晰可见，轮廓与DNA信号在局部同位。"),
    "lyman_07_a005": item("样例可用", "较稀疏且部分细长的细胞邻近组织区；轮廓位置有对应DNA对象，未见明显错位。部分区域body标记弱，胞质边界证据有限。", "部分蛋白通道对胞质边界支持较弱；未见明确坐标异常", "弱信号区的精细胞质边界应保留不确定性；当前样例未提示广泛复核"),
    "lyman_07_a006": item("样例可用", "拥挤组织和不规则腺样空隙中，核与mask轮廓总体同位，未见明确整体偏移或重复大块合并。"),
    "lyman_08_a001": item("样例可用", "活动图像范围内核与轮廓对齐；上下空带在图像和mask中一致。", "1200×800源视野显示于1200×1200画布，空带为补边而非坐标错位", "保留源尺寸与画布尺寸说明；无需针对padding重分割"),
    "lyman_08_a002": item("样例可用", "中央高密度区域DNA信号较清晰，逐核轮廓大体对应；部分body标记弱或弥散，限制胞质边界判断，未见明显整体错位。", "body标记较弱或弥散，细胞质边界证据有限；DNA与mask未显示坐标位移", "弱标记区域的细胞质边界保留不确定性；该样例未提示系统性分割复核"),
    "lyman_08_a003": item("样例可用", "致密且形态不规则的局部组织带有间隙；多数可见核各有轮廓，通道与mask在窗口内一致。"),
}


def update_records(records):
    for record in records:
        review = REVIEWED.get(record["roi"])
        if review:
            record.update(review)


def write_report(records):
    counts = Counter(record["judgment"] for record in records)
    rows = []
    for record in records:
        rows.append("| `{roi}` | {source} {acq} | {native} | {current} | {mask} | {judgment} | {findings} | {cause} | {action} | `{figures}` |".format(
            roi=record["roi"], source=record["source_mcd"], acq=record["source_acquisition_id"],
            native=record["native_mcd_shape_yx"], current=record["indexed_shape_yx"],
            mask=record["mask_origin"], judgment=record["judgment"], findings=record["findings"],
            cause=record["likely_cause"], action=record["next_action"],
            figures=", ".join(record["figures"])))
    report = """# Lyman2025_IMC_PanIN 逐 ROI 通道与 mask 复核

## 范围与方法

- 清单共 {n} 个 ROI，mask 来源均记录在逐 ROI JSON；当前本轮人工查看 {reviewed} 个，仍待合并的外部分工条目 {pending} 个。
- 按 ROI 的 `segmentation_task.signal.groups` 显示 191Ir/193Ir 核标记和 8 个 body 通道；窗口使用同一 XY 坐标比较原信号与实例 mask。
- 每个 ROI 检查一个核信号较强的局部窗口，并对照整幅信号/mask概览。单窗口未覆盖整张 ROI，因此“样例可用”只表示抽查窗口没有明显异常。
- 所有局部面板使用最近邻显示。8个50×50 px acquisition只能查看原生粗像素结构，不能据放大图恢复细胞边界。
- `input.tiff` 到 `image.tiff` 的 q995 归一化逐像素误差接近浮点精度；原生尺寸、MCD几何与padding见JSON。该数值核对辅助判断显示/预处理，不替代视觉复核。

## 当前计数

- 样例可用：{usable}
- 局部问题：{local}
- 疑似系统性问题：{systemic}
- 证据不足：{insufficient}
- 待补充：{pending}

8个50×50 ROI（`lyman_06_a008`–`lyman_06_a011`、`lyman_10_a001`–`lyman_10_a004`）为MCD原生 acquisition 尺寸，pixel size记录为1 µm；input/image/mask均保留50×50，未发现预处理把较大视野缩成50×50的证据。该尺寸的细胞边界判断为证据不足。50×50是视野尺寸，所有这些acquisition的pixel size仍为1 µm，不能据尺寸推断空间分辨率降低；证据不足来自所示信号/边界可辨性。当前QC使用最近邻；旧版HTML若用LANCZOS放大，可能额外显得模糊，但这不会改变源图分辨率。

另有窄条ROI通过padding形成方形索引图，例如 `lyman_02_a001`（原生66×1000）、`lyman_10_a005`（550×1050）、`lyman_11_a002`（191×800）及 `lyman_13_a001`（284×1000）；对应空白区与各通道/mask一致，不应当作低分辨率重采样。`lyman_12_a001`可见跨通道一致的横向黑带，当前mask在黑带处为空，优先核对源采集并在分析中排除缺失带。

`lyman_13_a003`经独立CLI与主控再次看图，当前x=[96,352)、y=[96,352)局部可见双核标记强信号团块缺少mask，改为局部问题并优先复核。该发现不支持将整个study直接判为系统性故障。

## 逐 ROI 汇总

| ROI | 源 acquisition | MCD 原生尺寸 Y×X | processed 尺寸 Y×X | mask来源 | 判断 | 异常/观察 | 可能原因 | 下一步 | QC图 |
|---|---|---:|---:|---|---|---|---|---|---|
{rows}
""".format(n=len(records), reviewed=sum(record["judgment"] != "待逐图确认" for record in records),
           pending=counts.get("待逐图确认", 0), usable=counts.get("样例可用", 0),
           local=counts.get("局部问题", 0), systemic=counts.get("疑似系统性问题", 0),
           insufficient=counts.get("证据不足", 0), rows="\n".join(rows))
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


def main():
    path = OUTPUT / "roi_review.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    update_records(records)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(records)
    print("reviewed", len(REVIEWED), "of", len(records), "judgments", dict(Counter(r["judgment"] for r in records)))


if __name__ == "__main__":
    main()
