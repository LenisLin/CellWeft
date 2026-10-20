# Crop 与采样：来源核对和采用边界

日期：2026-10-20。当前用户已明确固定224×224 crop、常规步长112（50%重叠）；文献用于解释采样选择，不覆盖这一决定。核对原论文、官方模型卡及公开实现；本页记录依据，执行定义由 [preparation.md §7](../../data/preparation.md#7-crop-与训练视图)维护。未运行生产 crop 管线或新训练实验。

| 来源 | 已核实做法 | CellWeft 采用范围 |
| --- | --- | --- |
| [KRONOS v1 原论文](https://arxiv.org/html/2506.03373v1)，Methods、表 S24 | DINOv2；基础块256×256、50%重叠，Methods明确训练先随机取224×224；2个global、8个local，面积比例分别[0.48,1]、[0.16,0.48] | 借鉴50%重叠及多视图，不照搬256基础块；CellWeft直接224 crop、112步长、224/98输出、token边长14 |
| [KRONOS2 官方模型卡](https://huggingface.co/MahmoodLab/KRONOS2/blob/main/README.md) | marker-aware DINOv2 ViT-B/16，参考图块256×256，推荐CORAL处理 | 256是该发布模型的参考输入，不构成CellWeft的尺寸要求；模型卡未完整披露预训练边缘/空白配方 |
| [CORAL PatchConfig](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/config/patch.py)、[Patcher](https://github.com/mahmoodlab/CORAL/blob/main/src/coral/patch/core.py) | 默认原生mpp；可配置stride/overlap；追加贴边起点；保留全部网格坐标及组织占比；短轴不足块边长时网格为空 | 采用原生网格、贴边覆盖、坐标索引和下游筛选；50%重叠由CellWeft显式配置，非CORAL默认；不引入其Zarr存储 |
| [VirTues 原论文](https://www.nature.com/articles/s41586-026-10884-y)，Data augmentation、Dataset curation | MAE；先采256基础块，过滤组织占比<30%，训练随机128；max-projection/Otsu及二值opening/closing估计组织；小于256图像不纳入预训练 | 借鉴组织筛选和分层裁剪，不沿用其128网络输入、不称其为DINO模型 |
| [DINOv2 官方增强代码](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py) | global/local分别对输入调用RandomResizedCrop；图像bicubic；原实现包含RGB颜色与标准化操作 | 独立生成各view并同步mask；明确覆盖local为98，不照搬RGB强度处理 |

CORAL 的上述行为来自实际源码阅读，不是论文推断；其组织检测还支持核/结构信号联合阈值与组织区域构建，未将这些额外操作加入本项目共享图像。公开工具行为也不等于 KRONOS2 完整预训练配方。

本项目适配有三处需要单独标明：

1. 基础块保留原生像素，因此固定224像素不代表统一µm视野。当前选择不改全图尺度；各图保存真实µm/px，跨平台尺度效果尚未验证。
2. 小于224的图像保留补零的虚拟索引供cell查询，默认不参与DINO训练；这是保留全部细胞关联的工程适配，不是CORAL网格行为或VirTues训练规则。
3. 组织估计使用已有共享图像的非恒定有效通道，3×3二值形态学；常数/无法估计时不自动准入。30%只筛训练块，不删低占比区域中的细胞/表达。不重用VirTues的强度处理、marker合并或核染蛋白替代策略。

直接索引224图像块、训练加载时生成224/98视图，无需把重叠块和local像素另存。随机增强产生的cell–token关系运行时计算，使用原图固定cell_index；部分可见细胞保留对应，表达仍为整细胞均值。每个cell可以对应多个crop和模型token，不能强行一对一。

用户纠正后撤回256准备块：没有本项目证据证明额外扩大基础区域的收益，不因其他模型使用256就引入这一步。KRONOS v1实际上区分256数据块与训练224裁剪，不能概括成“DINO要求256”；本项目直接224不需要复刻该数据准备选择。
