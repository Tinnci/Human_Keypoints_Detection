# 人体关键点检测项目

本项目是一个基于深度学习的人体关键点检测系统，旨在识别图像或视频中的人体姿态。它利用了卷积神经网络（特别是提到了基于 MobileNet 的姿态估计模型）来定位人体的关键关节部位。

## 项目简介

该项目包含完整的训练和推理流程：
*   **训练流程**：使用如 COCO 等数据集对模型进行训练，包括数据加载、数据增强、模型初始化、损失计算、反向传播和检查点保存等步骤。
*   **推理流程**：加载训练好的模型，对新的图像或视频进行处理，提取并可视化人体关键点和姿态。

项目还包括对可选的中间特征图（如 Heatmaps 和 PAFs）进行可视化的功能，以及对 OpenVINO 工具套件的潜在支持。

## 主要特性

*   **基于 MobileNet 的高效姿态估计模型**：平衡了速度与精度。
*   **完整的训练和推理管线**：从数据准备到模型部署。
*   **支持 COCO 数据集**：包含数据加载和标签准备脚本。
*   **数据增强**：提升模型的泛化能力。
*   **模块化设计**：代码结构清晰，易于扩展。
*   **详细的流程图**：通过 `inference_flow.py` 和 `trainning_flow.py` 脚本可以生成详细的训练和推理流程图，帮助理解项目逻辑。
    *   整体流程图 (`inference_0_overall_control_flow.png`, `training_0_overall_control_flow.png`)
    *   各阶段详细流程图 (例如: `inference_1_setup.png`, `training_1_data_setup.png` 等)
*   **(可能) 支持 OpenVINO 加速**：`modules/openvino` 目录下包含相关模块。

## 目录结构
D:\HUMAN_KEYPOINTS_DETECTION
│ inference_flow.py # 生成推理流程图的脚本
│ trainning_flow.py # 生成训练流程图的脚本
│ README.md # 项目说明文件
├─src
│ └─human_keypoints_detection
│ │ run.py # 运行推理的入口脚本
│ │ train.py # 运行训练的入口脚本
│ │ val.py # (可能)运行验证的入口脚本
│ │
│ ├─datasets
│ │ │ coco.py # COCO 数据集处理
│ │ │ prepare_train_labels.py # 准备训练标签
│ │ │ transformations.py # 数据增强/转换
│ │ │ init.py
│ │ │
│ │ └─coco
│ │ ├─annotation # COCO 标注文件存放处
│ │ └─images # COCO 图像文件存放处
│ │ └─val2017 # 示例：COCO 验证集图片
│ │ ... (图像文件)
│ │
│ ├─models
│ │ with_mobilenet.py # 基于 MobileNet 的模型定义
│ │ init.py
│ │
│ └─modules
│ │ conv.py # 卷积层模块
│ │ get_parameters.py # 获取模型参数
│ │ keypoints.py # 关键点处理逻辑
│ │ load_state.py # 加载模型状态
│ │ loss.py # 损失函数定义
│ │ pose.py # 姿态数据结构和处理
│ │ init.py
│ │
│ └─openvino # OpenVINO 相关模块
│ conv.py
│ draw.py
│ inference_engine_openvino.py
│ inference_engine_pytorch.py
│ input_reader.py
│ legacy_pose_extractor.py
│ load_state.py
│ one_euro_filter.py
│ parse_poses.py
│ pose.py
└─ ... (生成的流程图图片等)


## 安装与环境配置

1.  **克隆项目**：
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```
2.  **安装依赖**：
    项目可能依赖于 Python, PyTorch, OpenCV, Graphviz (用于生成流程图) 等库。建议创建一个虚拟环境，并根据 `requirements.txt` (如果提供) 安装依赖。
    ```bash
    pip install -r requirements.txt # 如果有此文件
    # 或手动安装主要依赖:
    # pip install torch torchvision torchaudio opencv-python numpy graphviz
    ```
    确保 Graphviz 已正确安装并添加到系统 PATH，以便生成流程图。

3.  **准备数据集**：
    *   下载 COCO 数据集（或其他支持的数据集）。
    *   按照 `src/human_keypoints_detection/datasets/coco/` 下的结构组织图像和标注文件。
    *   运行 `src/human_keypoints_detection/datasets/prepare_train_labels.py` (如果需要预处理标签)。

## 使用说明

### 1. 生成流程图 (可选)

为了更好地理解项目结构，可以首先生成流程图：
```bash
python inference_flow.py
python trainning_flow.py
```
这将在项目根目录下生成多张 `.png` 格式的流程图。

### 2. 模型训练

执行训练脚本，通常需要指定配置文件或命令行参数：
```bash
python src/human_keypoints_detection/train.py --config <path_to_config_file> # 示例命令
```
具体的训练参数和配置请参考 `train.py` 脚本的帮助信息或相关文档。训练好的模型权重（检查点）通常会保存在指定的目录中。

### 3. 模型推理

使用 `run.py` 脚本进行推理。你需要提供训练好的模型路径和输入源（图像、视频或摄像头）。
```bash
python src/human_keypoints_detection/run.py --checkpoint-path <path_to_checkpoint> --images <path_to_image_folder_or_file>
# 或者处理视频:
# python src/human_keypoints_detection/run.py --checkpoint-path <path_to_checkpoint> --video <path_to_video_file_or_camera_id>
```
具体的推理参数请参考 `run.py` 脚本的帮助信息。

### 4. 模型验证 (如果适用)

如果项目包含 `val.py` 脚本，它可能用于在验证集上评估模型性能：
```bash
python src/human_keypoints_detection/val.py --config <path_to_config_file> --checkpoint <path_to_checkpoint> # 示例命令
```

## 贡献

欢迎对本项目进行贡献！如果您有任何改进建议或发现了bug，请提交 Pull Request 或 Issue。

## 许可证

我改的我没权力我分配不了许可证哈