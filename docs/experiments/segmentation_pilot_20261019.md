# CellWeft 12 源图预处理与分割比较

2026-10-19。**已完成：12 个不同 study × 5 个组合，共 60 次有效预测，无失败或跳过。** IMC、MIBI、CODEX、多重 IF 各3个study；IF具体为COMET、CyCIF、IBEX。每个study一张源图，小图全图，大图固定ROI（最长边1536px），不声称12张整片均完成分割。原始来源保持只读。

[交互叠加图报告](/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/comparison/report.html) · [逐图CSV](/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/comparison/cell_statistics.csv) · [实例比较CSV](/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/comparison/instance_comparisons.csv) · [冻结样本清单](/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/manifest.json) · [最终独立复核](/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01/reviews/cellweft_pilot_final_review.md)

## 结论与后续讨论建议

试验完成时，证据支持优先讨论 B→cpsam_v2 作为新分割起点，并保留按 study/panel 核/膜通道配置。随后用户决定训练与分割共用逐图逐通道 P1/P99 表示，已有合格 whole-cell mask 优先复用；当前生产要求由[数据准备](../data/preparation.md)维护。本页 A/B、CP/IS/SD 定义、冻结参数和数值结果保留为历史实验记录，不表示需要维护两套正式预处理，也不证明 B 普遍更准确。

- Cellpose两路平均对象一致性F1为0.831，范围0.526–0.944。MIBI三图的一致性较高；CODEX均值较低主要受Monkman影响，不能将三个样本外推成技术总体排名。
- Monkman固定ROI：CP_A为1,304个实例，CP_B为2,973个，一致性F1=0.526。已查看固定中心和全ROI，A输入背景更亮，局部部分可见核区域未被A勾出而B有覆盖。这是支持B作为起点的具体观察，尚无人工标注误差率。
- Hoch：IS_A/IS_B分别为10,327/6,366个实例；固定局部中A的细小轮廓更密。数量差别本身不能证明过分割。Cellpose同图A/B为5,111/6,480，提示预处理作用依赖模型。
- Nirmal：局部核信号明显，膜/胞质合成信号对边界的支持较弱，若干cell-target输出贴近核；ROI包含大片低信号区域，部分IS实例位于这些区域。模型输出称为cell不能替代whole-cell范围检查，后续应优先核验该study的膜/胞质组合与组织区域。
- StarDist仅作核对照，Hoch上仅261个核实例，固定局部有大量可见核未被勾出。本轮保持原生像素、未针对StarDist调整尺度，不能据此否定其经适配后的表现。InstanSeg按mpp调整到0.5µm/px，CP/SD保持原生尺度，跨模型比较包含尺度策略差异。
- 已查看全部12张固定中心预览，并查看Monkman、Nirmal、Hoch全ROI叠加图；没有逐细胞人工修订。全部图的全ROI与固定局部已保存，便于一次概览。当前全部reference=null：来源mask的精确配对或compartment证据未齐，不进行准确率或优劣排名。

## 按技术比较

下表为每图对象匹配F1的等权均值，IoU≥0.5最大基数一对一匹配。**F1表示方案一致性，不是相对金标准的准确度；高一致性不说明A或B更优。**

|技术（每组n=3）|CP A/B一致性|IS A/B一致性|CP_B/IS_B一致性|
|---|---:|---:|---:|
|IMC|0.800|0.642|0.672|
|MIBI|0.910|0.829|0.705|
|CODEX|0.736|0.701|0.772|
|IF|0.879|0.787|0.816|

## 全部12源图结果矩阵

CP/IS列为cell-target实例数，不表示whole-cell真实性已验证；SD列为核数。两类不可直接作同一准确度排名。

|Study/技术|CP_A|CP_B|CP A/B F1|IS_A|IS_B|IS A/B F1|SD_B 核|
|---|---:|---:|---:|---:|---:|---:|---:|
|Danenberg2022_imc|1579|1360|0.685|1667|1466|0.633|1196|
|Hoch2022_imc|5111|6480|0.782|10327|6366|0.656|261|
|Jackson2020_imc|853|864|0.933|971|683|0.636|643|
|Schurch2020_codex|2291|2349|0.873|3763|2754|0.703|2378|
|Monkman2024_codex|1304|2973|0.526|2659|3313|0.680|2484|
|Phillips2021_codex|2401|2286|0.807|3134|2678|0.721|2417|
|Risom2022_mibi|628|679|0.927|1072|1076|0.806|644|
|Azulay2025_mibi|1378|1537|0.859|1855|1650|0.771|1215|
|Liu2023_mibi|3650|3969|0.944|4508|4356|0.909|3006|
|Ly2025_comet|550|542|0.927|664|613|0.854|924|
|Nirmal2022_cycif|2374|2400|0.845|2470|2940|0.617|2221|
|Radtke2024_ibex|3849|4096|0.867|4375|4414|0.891|4418|

## 资源实测与执行状态

RTX4090 24GiB、i9-13900K、约125GiB RAM；每进程计算线程上限8，模型串行。推理时间含模型计算及结果传回，排除模型加载、输入读取、输出压缩写盘和图表；首图保留懒初始化开销。RSS是每个进程的历史峰值，显存是PyTorch allocator峰值，非整卡占用。StarDist在CPU运行，不能据此作为同设备速度排名。

|组合|设备|有效结果|12图推理合计(s)|加载(s)|峰值RSS(GiB)|峰值allocated显存(GiB)|
|---|---|---:|---:|---:|---:|---:|
|CP_A|RTX4090|12/12|10.68|2.17|2.12|1.22|
|CP_B|RTX4090|12/12|10.81|2.10|2.10|1.17|
|IS_A|RTX4090|12/12|12.78|0.32|2.18|0.67|
|IS_B|RTX4090|12/12|11.46|0.35|2.17|0.67|
|SD_B|CPU|12/12|14.84|0.50|0.89|—|

12图准备（包含拟合区域读取、数值处理与派生文件写盘）合计86.44秒，准备进程峰值约1.59GiB。NAS源格式/压缩与读写是本次重要开销；这里不是全片吞吐基准，也未将下载和方案复核时间计入推理。

每个组合均完成：任务目标确认→方案规划→独立计划复核PASS→实现及合同检查→12图执行→逐图/技术汇总。未增加第6个组合、未做阈值搜索或密集消融。Mesmer因旧TF环境部署成本未纳入。

## 解释边界与产物

A为读取域Q99截断→log1p→study/panel/channel的pooled-pixel mean/std→3×3 σ1 Gaussian；B为选定通道P1/P99归一化。两路都按固定组等权合成，再各合成通道P1/P99适配到[0,1]，CP/IS内部normalize关闭。因此比较的是这些完整组合，不能分离log、滤波和权重各自的因果贡献。线性均值偏移在最终适配中抵消，std主要改变多通道相对权重。

P99是读取域内精确值，大图并非全图分位数；study统计使用最多3张其他采集的固定区域，Nirmal只有1张同panel拟合图；IBEX只有1份完整文件，使用同Z的3个不相交区域估计，不能称独立采集或cohort统计。全部像素含背景参与统计，没有组织门控。来源是已发布强度图，部分已有清理/缩放，本轮派生实验不代表重复执行这些来源的生产预处理。Hoch来源数值很小，log1p接近线性，不能假设跨来源效应相同。Monkman/Phillips尺度依据发布分辨率合同，原生NGFF的1µm元数据冲突已注明；未独立审计显微镜标定。

NAS中manifest保存精确来源路径、原生通道索引、核/膜分组、mpp证据、ROI和拟合域；prepared保存原始选定区域副本、A/B的HWC双通道及统计；masks保存60份原生ROI坐标uint32二维实例图，另有24份IS核图；model_raw保存模型附加输出；runtime保存逐图状态和资源；comparison保存CSV/JSON、24张PNG与HTML。本轮没有生成训练crop、蛋白定量或全局cell索引；这些输出是隔离试验产物。

可复跑入口：[实验脚本说明](../../scripts/experiments/segmentation_pilot/README.md)，[运行参数](../../configs/experiments/segmentation_pilot_20261019.json)。完整执行前定义及12图核/膜通道表如下。

---

# CellWeft 12 源图分割 pilot 计划

状态：授权范围内的实现修订，待主控以独立 CLI 对最终计划、脚本和 manifest 联合复核；**尚未宣称计划复核 PASS、真实输入验证或适配器实测通过**。本轮只修改 `/tmp/cellweft_pilot_draft/` 的脚本/README 和本计划，交付 patch notes；不处理实际图像、不推理、不部署权重、不修改生产代码/文档或 collection CSV。独立 CLI 可用 gpt-6.1-sol/high。统计和接口均为本次探索设置，不自动成为生产准备合同。

## 1. 问题与组合冻结

每个 IMC/MIBI/CODEX/mIF-CyCIF 技术组各取 3 个不同原始 study，每 study 一张评价目标源图，共 12 个图像单元。镜像、重发布和同 cohort 复用不增加 study 数；跨技术 study 只计一次。COMET、IBEX 在 mIF 组内注明原生技术。大图只评价冻结 ROI，不能称完成 12 张整片分割。所有 12 个样本均保留 CP_A/CP_B 主任务；mpp 未知不取消 CP。

最多 **5 个组合**，不因可选模型不可用而扩增：

| 组合 | 输入/实例语义 | 优先级与资源 |
| --- | --- | --- |
| CP_A | A 输入→明确本地 cpsam_v2；whole-cell | 主任务，torch 单 GPU |
| CP_B | B 输入→同一 cpsam_v2；whole-cell | 主任务，torch 单 GPU |
| IS_A | 与 CP_A 相同准备双通道→fluorescence_nuclei_and_cells；cell 主表、nucleus 另存 | 可选，torch 单 GPU；mpp 未知逐样本跳过 |
| IS_B | 与 CP_B 相同准备双通道→同一 IS 权重 | 可选，与 IS_A 配对；mpp 未知逐样本跳过 |
| SD_B | B 核通道→本地 StarDist 2D_versatile_fluo；仅核 | 可选，CPU-only |

先完成 CP_A/CP_B，再按剩余时间资源运行成对 IS，最后 SD_B；可选权重由其他部署负责人处理，本轮不负责。Mesmer 本轮省略的原因是旧 TensorFlow/Python 环境的额外部署成本；官方匿名权重可用，不将访问 token 写成阻碍。

## 2. Manifest 与输入准入

最终 manifest 由其他 helper 准备于 `/tmp/cellweft_final_manifest.json`，本轮只消费接口、不编辑。**12 行准确 manifest 表由主控补入并核对**，此处不编造文件、通道、study 独立性或已验证状态。运行前冻结每行 study/cohort/论文依据、准确 acquisition 和 panel、marker/measured form、来源原生位置与 selected 顺序、kind/path/indices、轴/Z、原生尺寸、来源强度处理阶段、ROI、mpp 及其证据、拟合集合、参考 mask 语义/方法/几何对应和预定替换顺序。替换只依读取/通道/统计可用性，不能依据预测质量。

- `mpp` 是已核验正数（µm/px）或 `null`；缺少依据写 null。CP/SD 原生像素有效；物理面积及物理 FOV 为 null，不猜测比例尺。IS 明确记录 `skipped: unverified mpp`。
- 拟合兼容键至少包含 study/dataset、panel/测量形式、来源处理阶段、所选 marker 身份；跨文件按 marker 对应，不能仅凭相同列号混 panel。主控核对兼容性，脚本保存实际来源与统计范围；不声称已自动审核这些生物学身份。
- 来源阶段未知允许本探索运行，但逐样本记 `unknown` 并在 HTML 展示；它不能证明处理只执行一次或与正式准备兼容。若已知磁盘已 log/标准化/平滑且不能合理解释重复处理，执行前换合格输入或修订实验定义；未知阶段的结果只能解释成对现存像素应用所声明实验链的比较，不能外推为生产流程重复验证。
- 非有限或负值直接失败，不填补、不截为零；原始数据保持不变。零动态/P1-P99 退化或拟合方差为零的通道在 A/B 同步省略并记录。核组无有效通道或最终核合成 P1-P99 退化时样本失败；膜组空时双路对应膜图固定为零并显式标注。

## 3. 拟合范围、ROI 与完整公式

每个样本最多 **3 个拟合读取域**；每个域两边最多 1536 native px。默认 `fit_scope='independent_files'`：1–3 个不同来源文件，与目标及彼此文件不重合；这只声明文件独立，不推断患者独立。**IBEX 唯一例外**必须逐 sample 显式设置 `fit_scope='same_source_disjoint_windows'`，study/technology 明确含 IBEX，恰好 3 个固定拟合窗口；三者与目标 ROI、彼此均不相交（共边允许），且文件、kind、通道 indices/顺序、Z index/maxproj 完全相同。不能把该例外称独立图像拟合；记录实际 scope、窗口及像素量，HTML 明示。

ROI 由主控在推理前固定 `[x,y,w,h]`，不由预测选择：两边≤1536 时用全图；其余用预定组织坐标，`w=min(W,1536), h=min(H,1536)`，因此窄长图是矩形，不强裁不存在的正方形。人工预选需保存核信号组织覆盖依据、候选/并列取舍及最终坐标。未预选时不得由脚本临时挑好看的窗口。保留原生尺寸和 ROI 定位；已有可用全源缩略图由主控另附，脚本只产出 ROI 预览，不能将它称全片缩略图。mpp 已知时物理视野为 `(w*mpp,h*mpp)`，未知为 null。

所有百分位采用 NumPy 默认 linear 定义，**在各自有界读取域内精确计算**，包含全部背景像素，不做组织阈值过滤。ROI 情况下它们是完整源图百分位的 ROI-derived 近似，不声称精确完整图像/全 cohort 统计。目标域与各拟合域分别求百分位；不先合并窗口求统一 P99。最大投影仅在 manifest 显式要求时逐 Z 平面 ROI 累计。

记来源域通道为 `x_ic`，`Q_p` 在该域全部像素求百分位，定义 `T_ic=log1p(min(x_ic,Q99(x_ic)))`，**不额外除以 P99**。对拟合域 `i=1..k, k≤3`，`N=sum_i n_i`，`mu_c=sum_i sum_p T_icp/N`，`sd_c=sqrt(sum_i sum_p (T_icp-mu_c)^2/N)`。使用 population moments，按 pooled pixels 加权，**不是等图像权重**；目标 ROI 不参与拟合，IBEX 仅用其同文件不相交域。

对目标每通道 `c`：

- `A_c=G_reflect_3x3,sigma1((log1p(min(x_c,Q99(x_c)))-mu_c)/sd_c)`。
- `B_c=clip((x_c-Q1(x_c))/(Q99(x_c)-Q1(x_c)),0,1)`。
- 两路使用共同有效 marker 集合；膜/核分别对组内通道做固定等权算术均值 `C_g^A/C_g^B`。
- 两路每个合成图均固定适配 `U_g=clip((C_g-Q1(C_g))/(Q99(C_g)-Q1(C_g)),0,1)`；退化膜图置零，退化核图失败。准备输出 float32 `H,W,2=[膜,核]`。没有隐藏的 branch 特有归一化。

Gaussian **先裁剪再滤波**，`sigma=1,radius=1,mode='reflect'` 仅 YX，ROI 边界 reflect，无物理 halo；不声称等价整图先滤波再裁剪。记录源图边缘（ROI 与 native 边界重合）、人工 ROI 边缘及模型内部 tile 概念。所有 ROI 触边/截断实例纳入主统计与匹配；`border_fraction` 只看 ROI 外沿，不测 tile seam，不按模型分别剔除实例。

末端百分位适配会抵消单通道线性 z-score，固定合成后的逐通道 mean 项亦合成为可被抵消的常数偏移；多通道 std 相对权重、log、平滑仍可能保留。比较回答完整有效输入链的输出差异，不能识别各步骤贡献或判定 dataset mean/std 是否有效。

`01_prepare` 保存来源每个 selected channel 的 min/p50/p90/p99/max/nonzero_fraction，另存最终 A/B 各双通道同一诊断，均包括背景像素；保存 Q1/Q99、pooled mu/sd/像素数、实际 fit_scope、实际域和通道省略理由，便于解释零动态、来源阶段未知及强度差异。

## 4. 模型接口与资源冻结

- CP：本地 `.venv-cellpose-sam` 源码/metadata 静态确认 cellpose **4.2.1.1**；`eval` 接受 `tile_overlap`，`diameter=None` 对应 rescale=1，`normalize=False` 绕过内部归一化，convert_image 保留所传三通道。显式本地 `cpsam_v2` 文件，不靠默认模型名或下载。输入补为 HWC `[膜,核,0]`，`channel_axis=-1`；固定 `diameter=None,bsize=256,batch_size=4,normalize=False,flow_threshold=.4,cellprob_threshold=0,tile_overlap=.1,resample=True,augment=False,min_size=15,max_size_fraction=.4,niter=None,do_3D=False`。这仅是源码检查，尚未证明权重加载或两路实际推理有效。
- IS：本地源码静态确认 `eval_medium_image` 的 `normalise=False`、双通道 CHW、`batch_size=4,tile_size=512,target='all_outputs',return_image_tensor=False,rescale_output=True`。只允许 nuclei+cells artifact，校验 `model.pixel_size=.5` µm/px。已核验输入 mpp 对应双线性输入重采样（源码约 1% 比例容差内不重采样），标签用 **nearest** 恢复原生坐标和 H/W；无额外归一化。保存 nuclei 与 cells，cell 才进入 whole-cell 比较。适配器及 TorchScript 返回轴仍待真实运行验证。
- SD_B：使用 `.venv-stardist` 继承的 **Python3.10 / TF2.12 CPU** 环境，**不要求 torch**。在 TF import 前 `CUDA_VISIBLE_DEVICES=-1`，关闭 TF GPU，可见设备/加载/预测均 CPU，intra-op 8、inter-op 1；读取已部署本地 `2D_versatile_fluo` 目录，不自动下载。B 核输入、原生像素、`n_tiles=(4,4)`，模型目录阈值保持固定；GPU 资源字段均 null。
- 所有阶段 CPU 计算线程配置最多 8，BLAS/OMP 同设 8；CP/IS 同时仅用一个 GPU，不并发模型，**NAS 无并行读取，一次一个 ROI**。准备每图最多读 3 个拟合域+1 个目标域、每域≤1536²×selected channels，Z 投影流式，禁止整片多通道载入；只有目标及当前拟合域的有界数组/单通道 moments 驻留，不缓存拟合图集合。所有样本共最多 36 个拟合域（不含显式 Z 平面展开）。主机 RSS 以 16 GiB 为执行预算，超预算停止该组合并记限制，不扩大读取/并行预算；外层主控监测，不把该预算误写为脚本硬性 OS 内存限制。
- 分别记录 preparation、model load、input read、inference 用时；CP/IS CUDA 同步后计时并在每 ROI 前 reset GPU allocator peak。计时含传输，不含文件 I/O/模型加载；首图初始化保留，不隐式热身。SD 用同步 CPU wall time。记录 RSS、进程历史峰值 RSS（不是每 ROI 独立峰值）、GPU allocated/reserved；SD 均 null。逐图记录错误；资源不足不私改 batch，改参须新 run 并成对重跑。
- 每次使用 NAS `results/segmentation_pilot_20261019/<run_id>/` 唯一目录，准备目录和每个 scheme 首次创建后不覆盖。改变配置新建 run_id；该草稿不实现原目录续跑。可重复重建 comparison 视图，但不能以旧成功 mask 掩盖新失败。

## 5. 比较指标与空集合

保留每个样本的全部 5 个组合状态：ok/failed/skipped/missing/blocked_preparation，逐技术显示 intended_n=3、manifest_n、n_available 和各状态 n；指标另报 n_defined。缺参考、可选未部署、IS mpp 未知或失败都保留分母背景。主完成计数是 12 study、每技术 3 图、**24 个有效 CP 输出**；否则标 partial，不能复用 study 或隐藏失败凑数。

whole-cell：CP_A vs CP_B、IS_A vs IS_B 为模型内 A/B agreement；另加 **CP_B vs IS_B、CP_A vs IS_A 为跨模型 agreement**，不是 accuracy。按图保存 B−A 的实例数、前景比例、面积和触边差值；按技术仅描述，n=3 个图像，不把细胞当独立重复或做显著性推断。

核：SD_B 另表；已保存的 IS_A/IS_B nuclei 可在同 native 坐标中与 SD_B 比较，明确为可选辅助核 agreement。不得与 whole-cell mask 混用。来源参考只有 acquisition、坐标原点、轴/尺度/裁剪对应已核实（`alignment_verified=true`）、方法已记录且类型明确时使用；shape 相同本身不能证明配准。核参考不进入 whole-cell 表；Xiao 等已知尺寸不符来源不能默认匹配。来源方法偏好随指标记录，不设人造 F1 金标准。

匹配固定：背景=0、非负整数实例，非连续 ID 合法；将正 ID 升序压缩，按稀疏实际重叠对计算 `IoU_ij=I_ij/(a_i+b_j-I_ij)`，只保留 **IoU≥0.5（含等号）**边。CSR 邻居升序、固定 SciPy 实现的最大基数二分匹配计 TP；并列以升序 ID 遍历固定选择，指标只依匹配数。不构造全对象 N×M 稠密矩阵。

以左侧为 reference、右侧为 prediction：`P=TP/N_pred`、`R=TP/N_ref`、`F1=2TP/(N_pred+N_ref)`。空分母为 null；双空 P/R/F1 均 null；仅一侧空时 F1=0，有定义侧覆盖=0，另侧覆盖=null。agreement 的 P/R 解释为方向性覆盖。

前景 IoU=`|F_ref∩F_pred|/|F_ref∪F_pred|`，同一 ROI 全像素，双空=1。count 为 ROI 内所有非零实例数；coverage=非零像素数/ROI像素数；border_fraction=接触 ROI 四边的不同非零实例数/count，count=0 时 null。面积为每实例像素数，报告 min/median/P90；空实例集为 null，已验证 mpp 时 µm²=px²×mpp²，否则所有物理面积字段 null。保留逐图 px² 值，跨分辨率不池化或平均 px² 面积；技术摘要仅在有验证 mpp 时汇总物理面积并报 n_defined。边界仅叠加作**视觉检查**，不生成未定义 boundary score。

HTML 保存固定全 ROI 与同一中心最大 384px 局部，同一 B 背景，清楚区分核/whole-cell、来源阶段、fit_scope 例外、统计近似、失败/缺失及所有 n；无预测后挑图或逐细胞修改参考。

## 6. 阶段入口与交付状态

每个组合保留：任务目标→方案规划→计划复核→实现检查→执行结果→汇总比较。以下是阶段入口条件，**不是已经完成的 PASS 记录，也不新增用户批准门槛**：

1. 主控冻结并补入 12-study 真实 manifest 表及来源/窗口/尺度证据；检查 IBEX 三窗口及其他样本独立文件、未知 phase/mpp 的限制；来源图保持只读。
2. 独立 CLI gpt-6.1-sol/high 联合复核最终计划、manifest、脚本；检查矩阵不超过 5 组合、核/cell 分离、统计域及资源预算。
3. 本轮仅运行合成检查（identity/ID permutation/sparse IDs/split/merge/no overlap/empty/IoU阈值含等号/ROI触边、fit overlap/count/selectors、null mpp），以及 compile/三个 CLI help；真实适配器验证另记。
4. 后续先对冻结输入做实际准备/模型加载和首 ROI 资源测量；使用真实输出验证适配器通道、轴、整数标签和 native 坐标，再串行推进全部 CP。可选组合依部署和预算决定，失败按原状态保留。
5. 收齐 24 CP 有效结果、逐样本/技术描述表、输入诊断、资源记录、全部状态、NAS 唯一运行链接，才记主任务完成；模型训练、生物学验证和生产接口仍未完成。

本轮文件仅为隔离实验 helper：`01_prepare.py`、`02_segment.py`、`03_compare.py`、`pilot_io.py`、`check_synthetic.py`、`README.md`。`pilot_io` 只供本实验共享，不提升生产 API。执行器不改变生产数据接口；实际结果另写实验记录，并更新项目状态与资源部署说明。实际检查结果及未验证项见 `/tmp/cellweft_pilot_patch_notes.md`。

## 固定12源图

|ID|技术|核|膜/胞质|mpp|ROI xywh|fit数量/范围|
|---|---|---|---|---|---|---|
|Danenberg2022_imc|IMC|DNA1,DNA2|panCK,SMA,CD45RA|1.0|[0, 0, 516, 581]|3/independent_files|
|Hoch2022_imc|IMC|DNA1,DNA2|B2M,S100,VIM|1.0|[0, 0, 903, 986]|3/independent_files|
|Jackson2020_imc|IMC|DNA1,DNA2|CD45,E-Cadherin/P-Cadherin,panCK|1.0|[0, 0, 881, 714]|3/independent_files|
|Schurch2020_codex|CODEX|HOECHST2|NA-K-ATPASE,CD45,CYTOKERATIN|0.37744|[192, 0, 1536, 1440]|3/independent_files|
|Monkman2024_codex|CODEX|DAPI2|CD45,PanCK,Vimentin|0.3775|[576, 744, 1536, 1536]|3/independent_files|
|Phillips2021_codex|CODEX|Hoechst|CD45,Cytokeratin,Vimentin|0.37744|[192, 0, 1536, 1440]|3/independent_files|
|Risom2022_mibi|MIBI|HH3|ECAD,CD44,CD45|0.488|[0, 0, 1024, 1024]|3/independent_files|
|Azulay2025_mibi|MIBI|dsDNA|NAKATPASE,ECAD,CD45|0.390625|[0, 0, 1024, 1024]|3/independent_files|
|Liu2023_mibi|MIBI|HH3|CD45,CD3,CD20|0.48828125|[0, 0, 1024, 1024]|3/independent_files|
|Ly2025_comet|IF|DAPI|CD45Rb,CD33Ms,CD68ms|0.23|[3207, 1651, 1536, 1536]|3/independent_files|
|Nirmal2022_cycif|IF|DNA_1|NCAD,S100a,CD45RO,CD68|0.6499999761581421|[3328, 3328, 1536, 1536]|1/independent_files|
|Radtke2024_ibex|IF|Hoechst:Cycle1_Nuclei|AF488:Cycle1_CD20,AF594:Cycle4_CD3,PE_iF594:Cycle9_CD45|0.284|[9728, 2560, 1536, 1536]|3/same_source_disjoint_windows|

精确路径/通道原生索引/尺度证据/统计窗口见本目录 manifest.json。全体12图中心64×64选定通道只读检查已通过，见 /tmp/cellweft_manifest_read_check.json。IBEX按multiplex IF定义纳入（主控已说明该假设），不声称用户逐项批准技术归组。

主控整合修订：manifest fit_scope 枚举统一为 independent_files；IBEX身份同时使用 sample_id/assay，IMS坐标采用valid_shape而非填充存储边界；CP加载官方固定版本显式checkpoint路径，不用文件名猜模型身份。官方LFS SHA256已核对并完成GPU合成冒烟。IS使用channels_last=False内存格式和skimage reader，数值预处理仍normalise=False。

## 执行记录

2026-10-19：最终独立复核 PASS；12项真实manifest集成检查、合成检查、编译和CLI检查通过。随后5组合、60次真实预测均完成；见文首结果。NAS运行目录：`/mnt/NAS_21T/ProjectData/CellWeft/results/segmentation_pilot_20261019/run01`。
