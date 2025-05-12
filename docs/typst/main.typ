#import "@preview/blind-cvpr:0.5.0": cvpr2025
#import "logo.typ": LaTeX, TeX

#show table.cell: set text(size: 6pt)

#let affls = (
  course_project_group: ( // 你可以修改这个ID
    institution: "某某大学 - 机器学习课程项目 (Your University/Course)",
  ),
)

#let authors_list = (
  (name: "你的名字/团队成员名字 (Your Name(s))", affl: ("course_project_group", ), email: "your.email@example.com"),
  // 如果是团队项目，可以添加更多作者
)

#show: cvpr2025.with(
  title: [基于深度学习的人体关键点检测系统实现 #linebreak() #text(smallcaps("Implementation of a Deep Learning Based Human Keypoint Detection System"))],
  authors: (authors_list, affls),
  keywords: ("人体姿态估计", "关键点检测", "深度学习", "课程项目", "MobileNet", "热力图", "PAFs"), // 根据你的项目调整关键词
  abstract: [
    // --- 请在此处重写你的摘要 ---
    // 示例：
    "本报告详细介绍了一个作为机器学习课程项目而设计和实现的人体关键点检测系统。该系统采用自顶向下的策略，利用基于MobileNet改进的卷积神经网络从图像中提取特征，并通过预测关键点热力图 (Heatmaps) 和部分亲和场 (Part Affinity Fields, PAFs) 来定位人体关节点并构建人体姿态。本文将阐述系统的整体架构，包括模型设计 (参考 `src/human_keypoints_detection/models/with_mobilenet.py`)、关键点检测与组合模块 (参考 `src/human_keypoints_detection/modules/`) 的实现细节。我们还将描述系统的训练流程 (参考 `src/human_keypoints_detection/train.py`)、推理管线 (参考 `src/human_keypoints_detection/run.py`)，并展示了在[提及你使用的数据集或测试图片]上的可视化结果。最后，本文总结了项目实现过程中的主要成果、遇到的挑战以及未来可能的改进方向。"
    #v(1em)
    #strong("Abstract:") " This report details the design and implementation of a human keypoint detection system, developed as part of a machine learning course project. The system employs a top-down strategy, utilizing a convolutional neural network based on a modified MobileNet architecture (see `src/human_keypoints_detection/models/with_mobilenet.py`) to extract features from images. It predicts keypoint heatmaps and Part Affinity Fields (PAFs) to localize human joints and construct body poses. This paper outlines the overall system architecture, including a_NEWLINE_NOT_ESCAPED implementation details of the model design and keypoint detection/grouping modules (see `src/human_keypoints_detection/modules/`). We also describe the training process (see `src/human_keypoints_detection/train.py`), inference pipeline (see `src/human_keypoints_detection/run.py`), and present visual results on [mention your dataset or test images]. Finally, this report summarizes the main achievements, challenges encountered, and potential future improvements for the project."
    // --- 摘要结束 ---
  ],
  bibliography: bibliography("main.bib"), // 如果你引用了文献，保留这个
  accepted: false, // 这通常用于CVPR投稿，课程项目可能不需要
  id: none,
)

= 项目引言 (Project Introduction) <sec:intro>
// --- 请在此处重写你的引言 ---
// 示例：
人体姿态估计（Human Pose Estimation, HPE）是计算机视觉领域的一项基础任务，旨在从图像或视频中定位人体的关键关节点，如头、肩、肘等。这项技术是理解人类行为、动作识别等众多应用的基础。本报告围绕一个机器学习课程项目展开，目标是设计并实现一个能够进行2D人体关键点检测的深度学习系统。

我们采用了一种常见的#strong[自顶向下 (Top-Down)] 的两阶段方法：首先假设图像中已有人体边界框（在实际应用中可由一个独立的人体检测器提供），然后在该框内进行单人姿态估计。姿态估计的核心是利用卷积神经网络 (CNN) 学习从图像到关键点位置的映射。具体来说，我们的系统输出每个关键点的#strong[热力图 (Heatmap)] 和连接不同身体部件的#strong[部分亲和场 (Part Affinity Fields, PAFs)]。热力图表示关节点可能出现的位置，而PAFs则帮助将检测到的独立关节点正确组合成属于同一个人的骨架。

本报告的后续章节将详细介绍我们的系统：
- #link(<sec:system_design>)[系统设计与实现]：阐述网络架构、训练配置和推理流程。
- #link(<sec:key_modules>)[关键模块与实现考量]：讨论核心模块的具体实现和设计选择。
- #link(<sec:results_discussion>)[实验结果与讨论]：展示系统在示例图片上的可视化输出，并进行分析。
- #link(<sec:challenges_future_project>)[项目挑战与未来展望]：总结开发过程中遇到的问题及对本项目后续的改进思考。
- #link(<sec:appendix_visuals>)[附录]：包含系统流程图和更多可视化结果。
// --- 引言结束 ---

// --- 原"历史演进"部分内容已大幅修改/替换 ---
= 背景与相关工作 (Background and Related Work) <sec:background>
// 示例：
我们的项目建立在深度学习姿态估计的成熟技术之上。早期的深度学习方法如DeepPose @Toshev2014DeepPose 尝试直接回归关节点坐标，但面临优化难题。随后，基于热力图的检测方法成为主流，如经典的Stacked Hourglass Networks @Newell2016StackedHourglass 和 Convolutional Pose Machines (CPM) @Wei2016CPM。这些方法通过预测每个关节点的概率热力图，在精度和鲁棒性上取得了显著提升。

在多人姿态估计中，OpenPose @Cao2017OpenPose (一个自底向上的方法) 引入了部分亲和场 (PAFs) 来编码身体部件间的关联，有效地将检测到的关节点分组到对应的个体。我们的项目借鉴了热力图和PAFs的思想，并将其应用于单人姿态估计（或假设已完成人体检测的Top-Down流程中）。

在网络架构方面，为了平衡精度和效率，许多研究探索了轻量级的骨干网络。MobileNet系列 @Howard2017MobileNets @Sandler2018MobileNetV2 因其深度可分离卷积等高效设计，常被用作资源受限场景下的特征提取器。我们的模型（详见 `src/human_keypoints_detection/models/with_mobilenet.py`）也采用了类似MobileNet的卷积块来构建特征提取主干，并结合了类似CPM的多阶段处理思想来逐步优化热力图和PAFs的预测。

// 你可以根据你的实际参考和借鉴，简要提及对你项目有启发的工作。
// 不需要像之前那样全面回顾SOTA历史。
// --- 背景与相关工作结束 ---


= 系统设计与实现 (System Design and Implementation) <sec:system_design>
本章节详细介绍我们实现的2D人体关键点检测系统的整体设计，包括网络架构、训练配置和推理流程。相关的核心代码位于 `src` 目录下的 `human_keypoints_detection` 包中。

== 网络架构 (Network Architecture) <sec:network_arch>
我们设计的姿态估计网络（`PoseEstimationWithMobileNet` 类，定义于 `models/with_mobilenet.py`）是一个多阶段的卷积神经网络，旨在从输入图像中预测关键点热力图和部分亲和场 (PAFs)。

#figure(
  image("images/flowcharts/your_network_architecture_diagram_zh.png", width: 70%), // <--- 提示：你需要准备一张你网络结构的示意图
  caption: [我们姿态估计网络的架构示意图。它主要包含一个特征提取主干、一个CPM模块以及多个预测阶段。 (请替换为你的网络结构图和描述)]
) <fig:network_architecture>

该网络主要由以下几个部分组成：
1.  #strong[特征提取主干 (Backbone)]：
    此部分基于轻量级的MobileNet思想构建，采用了一系列标准卷积 (`conv`) 和深度可分离卷积 (`conv_dw`) 模块（定义于 `modules/conv.py`）。它负责从输入图像（例如，256x192x3）中提取初始的图像特征。网络的前几层通过步长为2的卷积进行下采样，以扩大感受野并减少计算量。

2.  #strong[CPM模块 (Convolutional Pose Machine like block)]：
    在特征主干之后，我们引入了一个类似于CPM的模块（`Cpm` 类），它对主干输出的特征进行进一步处理，以适应后续的姿态估计任务。该模块通常包含若干卷积层，旨在整合特征并将其调整到适合初始预测的通道数（例如128通道）。

3.  #strong[初始预测阶段 (Initial Stage)]：
    `InitialStage` 类负责根据CPM模块输出的特征，首次预测19个关键点的热力图（每个关节点一个，加上一个背景通道）和38个PAFs（每个身体连接的x和y方向向量场）。这一阶段的输出维度通常是输入特征图的1/8（例如，对于256x192的输入，热力图和PAFs的空间维度是32x24）。

4.  #strong[精炼阶段 (Refinement Stages)]：
    为了提升预测精度，网络包含一个或多个 `RefinementStage`。每个精炼阶段接收前一阶段预测的热力图、PAFs以及CPM模块输出的特征图作为输入。通过将这些信息拼接（`torch.cat`）并送入一系列 `RefinementStageBlock`，网络能够逐步修正和优化预测结果。每个精炼阶段同样输出新的热力图和PAFs。我们的默认配置使用了一个精炼阶段。

所有卷积模块（除了最终输出层）都使用了ReLU激活函数，并且大部分配备了批归一化 (Batch Normalization) 以稳定训练。最终输出热力图和PAFs的卷积层不使用ReLU，以便可以直接表示概率值或向量分量。

== 训练细节 (Training Details) <sec:training_details>
模型的训练过程在 `train.py` 脚本中定义。
-   #strong[数据集 (Dataset)]：我们使用了[请在此处填写你使用的数据集，例如：COCO 2017训练集的一个子集 / MPII / 或你自定义的数据集]。训练标签需要预处理成目标热力图和目标PAFs。热力图通常通过在关键点真值位置放置一个2D高斯核生成。
-   #strong[数据增强 (Data Augmentation)]：为了提升模型的泛化能力，我们采用了一系列数据增强技术（定义于 `datasets/transformations.py`），包括：
    -   随机缩放 (`Scale`)
    -   随机旋转 (`Rotate`)
    -   随机翻转 (`Flip`)
    -   随机裁剪与填充 (`CropPad`)
    这些变换会同时应用于图像和关键点坐标。
-   #strong[损失函数 (Loss Function)]：对于热力图和PAFs的预测，我们均采用均方误差损失 (L2 Loss)，分别计算预测输出与目标真值之间的差异。损失函数定义在 `modules/loss.py` (虽然它很简单，直接用 `torch.nn.MSELoss` 也可以)。每个阶段（初始阶段和所有精炼阶段）的损失都会被计算并累加，以实现中间监督。
-   #strong[优化器 (Optimizer)]：我们使用了 [请填写你使用的优化器，例如：Adam 优化器 @Kingma2014Adam]。
-   #strong[学习率 (Learning Rate)]：初始学习率为 [请填写]，并可能采用 [请填写学习率衰减策略，例如：MultiStepLR]。
-   #strong[批大小与迭代次数 (Batch Size & Iterations)]：训练时的批大小为 [请填写]，总共训练了 [请填写] 次迭代或轮次。
-   #strong[硬件与环境 (Hardware & Environment)]：训练在 [请填写你的硬件，如NVIDIA GPU型号] 上进行，使用了PyTorch [版本号]框架。

== 推理流程 (Inference Pipeline) <sec:inference_pipeline>
推理过程在 `run.py` 中实现，其核心函数是 `infer_fast` 和 `run_demo`。给定一张输入图像，系统执行以下步骤（如图所示 @fig:inference_overall_flowchart，参考附录中的推理主控流程图）：

1.  #strong[图像预处理 (Preprocessing)]：
    -   将输入图像缩放到网络期望的固定高度（例如256像素），同时保持宽高比。
    -   进行归一化处理（例如，减去均值128，然后除以256）。
    -   对图像进行填充，使其宽度也满足网络步长（例如8的倍数）的要求，便于后续特征图处理。
    这些操作由 `val.py` 中的 `normalize` 和 `pad_width` 函数辅助完成。

2.  #strong[模型前向传播 (Model Inference)]：
    -   将预处理后的图像转换为PyTorch张量，并送入加载了预训练权重的 `PoseEstimationWithMobileNet` 模型。
    -   模型输出最后一个精炼阶段预测的热力图和PAFs。这些通常是下采样后的特征图（例如，原始输入图像的1/8）。
    -   通过双三次插值将热力图和PAFs上采样回接近原始图像1/stride的分辨率（例如，如果stride=8，upsample_ratio=8，则恢复到原图1/1的尺寸，但在项目中通常是上采样到原图的某个固定比例，比如1/8上采样8倍到原图尺寸，或者上采样到输入给后处理的大小）。

3.  #strong[关键点提取与组合 (Keypoint Extraction and Grouping)] (参考 `modules/keypoints.py`)：
    -   #strong[提取候选关键点 (`extract_keypoints`)]：遍历每个关节点类型的上采样热力图，通过寻找局部峰值（并可能应用非极大值抑制）来识别候选的关键点位置和置信度。
    -   #strong[组合关键点 (`group_keypoints`)]：利用上采样的PAFs将提取到的候选关键点组合成属于不同个体的完整人体骨架。PAFs编码了身体不同部位之间的连接方向和强度。通过计算候选关键点对之间的连接得分（基于PAFs的线积分），并使用图匹配或贪心算法（项目中是基于NMS和PAF打分的贪心策略）来找到最优的连接方式，形成多人的姿态（在我们的单人Top-Down设定中，主要是形成一个人的姿态）。

4.  #strong[坐标转换与姿态构建 (Coordinate Transformation and Pose Construction)]：
    -   将提取和组合后得到的关键点坐标从上采样后的特征图空间转换回原始输入图像的坐标空间。这需要考虑预处理阶段的缩放 (scale) 和填充 (pad) 的影响。
    -   将每个人的关键点集合及其置信度封装成一个 `Pose` 对象（定义于 `modules/pose.py`）。

5.  #strong[可视化与输出 (Visualization and Output)]：
    -   在原始输入图像上绘制检测到的关键点和连接它们的骨骼。`Pose` 类中的 `draw` 方法负责此功能。
    -   （可选）可视化中间结果，如热力图和PAFs的叠加图（由 `run.py` 中的 `draw_heatmap` 和 `draw_paf` 函数实现）。
    -   显示结果图像或保存到文件。

以上流程确保了从原始图像到最终姿态估计结果的完整处理。

= 关键模块与实现考量 (Key Modules and Implementation Considerations) <sec:key_modules>
// --- 此部分替换原"关键技术创新" ---
// 你可以根据项目中你认为重要或有特点的模块来写
// 示例：

在实现本系统的过程中，我们对几个核心模块的设计和参数选择进行了一些考量。

== 卷积模块 (`modules/conv.py`)
我们定义了基础的 `conv`（标准卷积+BN+ReLU）和 `conv_dw`（深度可分离卷积+BN+ReLU）模块。选择深度可分离卷积是为了在保持感受野的同时减少参数量和计算量，这对于构建轻量级网络是重要的。`conv_dw_no_bn` 则用于某些不需要BN的特定层。

== 关键点处理 (`modules/keypoints.py`)
-   `extract_keypoints` 函数：我们采用了[简述你提取关键点的策略，例如：简单的局部最大值搜索，或是否有NMS]。
-   `group_keypoints` 函数：此函数的核心是根据PAFs将候选点连接起来。我们的实现基于[简述你的分组策略，例如：遍历所有可能的连接，计算PAF得分，然后进行分配]。我们定义了19条身体部件连接（`BODY_PARTS_PAF_IDS`）和18个关键点类型（`Pose.num_kpts`，不包括背景）。

== 姿态表示与绘制 (`modules/pose.py`)
`Pose` 类封装了一个人体的所有关键点、置信度以及边界框。其 `draw` 方法负责在图像上可视化骨架。我们为左右身体部位定义了不同的颜色，并可以显示关键点ID。在此模块中，`BODY_PARTS_KPT_IDS` 定义了构成身体部件连接的关节点对。

== 损失函数选择 (`modules/loss.py`)
我们选择了L2损失（均方误差）来监督热力图和PAFs的生成。这是一个简单且广泛应用的选择。我们没有尝试更复杂的损失函数如Wing Loss等，主要是考虑到课程项目的时间和实现复杂度。对所有预测阶段（初始和精炼）都施加损失，有助于进行有效的中间监督。

== 数据变换 (`datasets/transformations.py`)
实现了一系列标准的几何增强变换。这些变换在训练时随机应用，以提高模型的泛化能力。在 `generate_flow_docs.py` 中，我们尝试复用这些变换来生成增强效果的可视化示例，但由于该脚本不处理完整的标签数据（如 `label['scale_provided']`, `label['objpos']`），因此这些变换在该脚本中的行为可能与实际训练时有所简化或不同。例如，`Scale` 变换在实际训练中会依赖 `label['scale_provided']` 和 `label['objpos']`，而在可视化脚本中我们主要关注其对图像本身的缩放效果。

// --- 关键模块结束 ---

= 实验结果与讨论 (Experimental Results and Discussion) <sec:results_discussion>
// --- 此部分替换原"SOTA性能分析"的大部分内容 ---
由于这是一个课程项目，我们的主要目标是成功搭建并运行一个基本的人体关键点检测系统，而非追求在标准基准上的SOTA性能。因此，本节主要展示系统在示例图片上的可视化输出，并对结果进行定性分析。

所有的可视化结果均由 `src/generate_flow_docs.py` 脚本自动生成，并保存在 `generated_flow_visualizations` 目录下。该脚本使用了预设的检查点（`../train_checkpoints/1/checkpoint_iter_420000.pth`）和示例图片（`../images/000000000790.jpg`）。

== 数据增强效果
图 @fig:data_augmentation_examples (参考附录) 展示了对原始训练样本应用随机旋转、缩放、翻转和裁剪后的效果。这些增强手段有助于模型学习对不同姿态、尺度和视角的不变性。

== 推理过程可视化
附录中的图 @fig:inference_input_preprocess 到 @fig:inference_final_pose 展示了推理管线中几个关键步骤的可视化结果：
-   #strong[输入与预处理]：原始输入图像和经过缩放、归一化、填充后的图像。
-   #strong[网络中间输出]：模型预测的原始热力图和PAFs。热力图清晰地标示了各个关节点的高响应区域，PAFs则以向量场的形式指示了身体部件间的连接方向。
-   #strong[最终姿态输出]：经过后处理（关键点提取、组合、坐标变换）后，在原始图像上绘制的最终人体骨架，以及在空白背景上绘制的骨架。

从这些可视化结果可以看出，我们的系统能够：
-   [请根据你的实际观察结果填写，例如：基本准确地定位大部分可见的关键点。]
-   [例如：在一定程度上处理简单的自遮挡。]
-   [例如：成功地将检测到的点连接成符合人体结构的骨架。]

然而，也观察到一些不足之处：
-   [例如：对于某些复杂姿态或轻微遮挡的关节点，定位可能不够精确。]
-   [例如：在多人场景（如果你的示例图片包含多人且你的系统是单人的），系统可能只关注一个人，或错误地将不同人的点连接起来（如果未做严格的单人处理）。—— 根据你的实际情况说明]
-   [例如：PAFs的质量对于最终骨架连接的准确性至关重要，若PAFs预测模糊，则可能导致错误的连接。]

这些定性结果为我们理解系统的当前能力和潜在的改进方向提供了依据。

// --- 实验结果结束 ---


= 项目挑战与未来展望 (Project Challenges and Future Outlook) <sec:challenges_future_project>
// --- 此部分替换原"当前挑战与未来方向"的大部分内容 ---

== 项目实现过程中的挑战 <sec:project_challenges>
在本次课程项目中，我们主要遇到了以下挑战：
1.  #strong[环境配置与依赖管理]：搭建PyTorch开发环境，确保所有依赖库（如OpenCV, NumPy, TorchVision）版本兼容是一个初始的障碍。
2.  #strong[理解热力图与PAFs的生成与监督]：理解如何从关键点坐标生成高斯热力图作为监督信号，以及如何生成和利用PAFs进行部件连接，是项目的一个核心难点。
3.  #strong[数据处理与增强的细节]：正确实现数据增强变换，并确保它们能同时作用于图像和关键点坐标，需要细致的坐标变换处理。
4.  #strong[网络调试与参数调整]：由于训练时间有限，我们可能未能充分进行超参数调优（如学习率、优化器参数、损失权重等）。网络不收敛或效果不佳时，定位问题也比较耗时。
5.  #strong[计算资源限制]：[如果你遇到了，例如：由于缺乏高性能GPU，训练时长受限，难以尝试更大模型或更充分的训练。]
6.  #strong[代码理解与复现]：本项目在一定程度上参考了现有开源实现，理解其代码逻辑并适配到我们的课程项目框架中也花费了不少精力。

== 本项目的未来改进方向 <sec:project_future_work>
针对当前系统的实现和遇到的挑战，未来可以从以下几个方面进行改进：
1.  #strong[模型优化]：
    -   尝试更先进的骨干网络（如ResNet、HRNet的轻量版本）或注意力机制，以提升特征提取能力。
    -   探索更有效的损失函数，例如针对前景背景不平衡的Focal Loss，或针对关键点定位更精确的Wing Loss。
2.  #strong[训练策略改进]：
    -   进行更细致的超参数搜索和学习率调度策略优化。
    -   采用更丰富和更针对性的数据增强方法。
    -   如果条件允许，使用更大、更多样化的数据集进行训练或微调。
3.  #strong[后处理优化]：
    -   改进关键点提取算法（例如，更鲁棒的NMS）和部件连接算法，以提高复杂情况下的准确性。
4.  #strong[扩展到多人姿态估计]：
    -   基于当前的单人姿态估计模块，实现一个完整的多人Top-Down系统（集成人体检测器）。
    -   或者尝试实现一个自底向上的方法，学习更强大的关节点分组策略。
5.  #strong[量化评估]：
    -   在标准的公开数据集（如COCO验证集、MPII）上进行定量评估，使用OKS或PCKh等指标来衡量系统性能，并与其他方法进行比较。

通过这些改进，有望进一步提升本系统的准确性、鲁棒性和实用性。

// --- 挑战与未来展望结束 ---

= 结论 (Conclusion) <sec:conclusion_project>
// --- 请在此处重写你的结论 ---
// 示例：
本报告详细介绍了一个作为机器学习课程项目而设计和实现的人体关键点检测系统。我们成功搭建了一个基于MobileNet改进架构的卷积神经网络，该网络能够通过预测热力图和部分亲和场，从输入图像中检测人体关键点并构建姿态骨架。系统的核心模块包括特征提取、多阶段预测、关键点提取与分组、以及姿态表示。

通过本次项目，我们深入学习了深度学习在姿态估计领域的应用，实践了包括数据处理、模型设计、训练调优和结果分析在内的完整流程。可视化结果表明，我们的系统能够对简单场景下的单人姿态进行有效估计。尽管受限于时间和资源，系统在精度和鲁棒性方面与当前SOTA方法存在差距，但项目的完成为我们后续在该领域的学习和研究打下了坚实的基础。我们识别了当前系统的一些局限性，并展望了未来可能的改进方向，包括模型优化、训练策略调整和功能扩展等。

// --- 结论结束 ---

#heading(level: 1, numbering: none, outlined: false)[致谢 (Acknowledgements)]
// 示例：
感谢[课程名称]的[老师姓名]老师在项目过程中的指导。感谢[同学/队友姓名]的合作与讨论（如果是团队项目）。
// 致谢内容 (可选)...

// --- 附录部分保留，但请确保其内容与你正文描述的系统一致 ---
#heading(level: 1, numbering: "A", outlined: true)[附录：流程图与可视化示例 (Appendix: Flowcharts and Visualization Examples)] <sec:appendix_visuals>

本附录展示了我们项目所实现的系统流程图以及通过 `generate_flow_docs.py` 脚本自动生成的部分可视化结果。

== 训练与推理主控流程图 (Main Control Flowcharts)

#figure(
  image("images/flowcharts/training_0_overall_control_flow.png", width: 80%),
  caption: [本项目训练过程主控流程图。]
) <fig:training_overall_flowchart>

#figure(
  image("images/flowcharts/inference_0_overall_control_flow.png", width: 80%),
  caption: [本项目推理过程主控流程图。]
) <fig:inference_overall_flowchart>

== 详细阶段流程图示例 (Detailed Stage Flowchart Examples)
// ... (附录的其余部分，你可以根据需要调整图片和说明文字，确保它们准确反映你的系统) ...

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    image("images/flowcharts/training_1_data_setup.png", width: 100%),
    image("images/flowcharts/inference_1_setup.png", width: 100%),
  ),
  caption: [左：训练阶段一（配置与数据准备）详细流程。右：推理阶段一（设置与准备）详细流程。]
) <fig:detailed_stage_flowcharts_example>


== 自动生成的可视化结果示例 (Examples of Auto-Generated Visualizations)

=== 训练数据增强示例 (Training Data Augmentation Examples)

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 6pt,
    image("images/visualizations/example_train_original.jpg", width: 100%),
    image("images/visualizations/example_train_augmented_rotate_scale.jpg", width: 100%),
    image("images/visualizations/example_train_augmented_flip_crop.jpg", width: 100%),
  ),
  caption: [从左至右：原始训练样本，随机旋转与缩放增强，随机翻转与裁剪增强。]
) <fig:data_augmentation_examples>

=== 推理过程可视化示例 (Inference Process Visualization Examples)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    image("images/visualizations/inference_input_original_example.jpg", width: 100%),
    image("images/visualizations/inference_preprocessed_scaled_example.jpg", width: 100%),
  ),
  caption: [左：推理原始输入图像。右：图像预处理（缩放后）示例。]
) <fig:inference_input_preprocess>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    image("images/visualizations/inference_output_heatmaps_example.png", width: 100%),
    image("images/visualizations/inference_output_pafs_example.png", width: 100%),
  ),
  caption: [左：模型输出Heatmaps可视化。右：模型输出PAFs可视化。]
) <fig:inference_heatmaps_pafs>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    image("images/visualizations/inference_final_output_on_image_example.jpg", width: 100%),
    image("images/visualizations/inference_postprocessed_poses_blank_bg_example.jpg", width: 100%),
  ),
  caption: [左：最终姿态在原图上绘制结果。右：后处理姿态在空白背景上绘制结果。]
) <fig:inference_final_pose>
