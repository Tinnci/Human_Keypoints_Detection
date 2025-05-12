import os
import sys
import cv2
import numpy as np
import torch
import shutil # 用于清理旧的图片

# --- 自动修正PYTHONPATH，确保可以import human_keypoints_detection下的包 ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_for_paths = os.path.dirname(current_script_dir) # src的父目录，用于构建资源路径
if project_root_for_paths not in sys.path:
    sys.path.insert(0, project_root_for_paths)

# --- 项目相关导入 ---
try:
    import human_keypoints_detection.models.with_mobilenet as with_mobilenet
    import human_keypoints_detection.modules.load_state as load_state_mod
    import human_keypoints_detection.modules.keypoints as keypoints_mod
    import human_keypoints_detection.modules.pose as pose_mod
    import human_keypoints_detection.val as val_mod
    import human_keypoints_detection.run as run_mod
    from torchvision import transforms
    import human_keypoints_detection.datasets.transformations as transf_mod
    NEURAL_NETWORK_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告：导入项目模块失败。自动图像生成功能将受限。错误：{e}")
    print("请确保 generate_flow_docs.py 在 src 目录下，并用 'python generate_flow_docs.py' 运行。")
    NEURAL_NETWORK_MODULES_AVAILABLE = False

# --- 配置常量 ---
# !!! 用户需要根据实际情况修改这些路径 !!!
DEFAULT_CHECKPOINT_PATH = os.path.join(project_root_for_paths, 'train_checkpoints', '1', 'checkpoint_iter_420000.pth')
DEFAULT_SAMPLE_IMAGE_PATH = os.path.join(project_root_for_paths, 'images', '000000000790.jpg')
# 确保文件存在的路径检查
if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
    print(f"警告：检查点文件 {DEFAULT_CHECKPOINT_PATH} 不存在！")
if not os.path.exists(DEFAULT_SAMPLE_IMAGE_PATH):
    print(f"警告：示例图片 {DEFAULT_SAMPLE_IMAGE_PATH} 不存在！")

VIZ_SUBFOLDER = os.path.join(project_root_for_paths, "generated_flow_visualizations") # 存放自动生成的图片
NET_INPUT_HEIGHT_SIZE = 256 # 与 run.py 中一致
CPU_INFERENCE = True # 如果没有GPU或想强制CPU，设为True

# 全局网络实例 (按需加载)
net_instance = None
device_instance = None

def get_model_and_device():
    global net_instance, device_instance
    if not NEURAL_NETWORK_MODULES_AVAILABLE:
        print("错误：核心模块未加载，无法初始化模型。")
        return None, None
        
    if net_instance is None:
        try:
            print(f"正在加载模型检查点: {DEFAULT_CHECKPOINT_PATH}...")
            net_instance = with_mobilenet.PoseEstimationWithMobileNet()
            if not os.path.exists(DEFAULT_CHECKPOINT_PATH):
                print(f"错误：检查点文件 {DEFAULT_CHECKPOINT_PATH} 未找到。请检查路径。")
                return None, None
            checkpoint = torch.load(DEFAULT_CHECKPOINT_PATH, map_location='cpu')
            load_state_mod.load_state(net_instance, checkpoint)
            
            device_instance = torch.device("cuda" if torch.cuda.is_available() and not CPU_INFERENCE else "cpu")
            net_instance = net_instance.eval().to(device_instance)
            print(f"模型已加载到 {device_instance} 并设置为评估模式。")
        except Exception as e:
            print(f"加载模型或检查点失败: {e}")
            net_instance = None
            device_instance = None
    return net_instance, device_instance

def ensure_viz_subfolder():
    if os.path.exists(VIZ_SUBFOLDER):
        print(f"清理旧的 '{VIZ_SUBFOLDER}' 目录...")
        shutil.rmtree(VIZ_SUBFOLDER)
    os.makedirs(VIZ_SUBFOLDER, exist_ok=True)
    print(f"已创建/清理可视化图片输出目录: '{VIZ_SUBFOLDER}'")

# --- 图像生成函数 ---

def generate_training_data_augmentation_images():
    if not NEURAL_NETWORK_MODULES_AVAILABLE: return
    print("正在生成训练数据增强示例图片...")
    if not os.path.exists(DEFAULT_SAMPLE_IMAGE_PATH):
        print(f"错误：示例图片 {DEFAULT_SAMPLE_IMAGE_PATH} 未找到。")
        return

    img_orig = cv2.imread(DEFAULT_SAMPLE_IMAGE_PATH)
    if img_orig is None:
        print(f"错误：无法读取示例图片 {DEFAULT_SAMPLE_IMAGE_PATH}。")
        return

    cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "example_train_original.jpg"), img_orig)

    # 定义一些增强变换 (从 train.py 借鉴)
    # 注意：这些变换可能需要关键点信息才能完整工作，这里仅作图像变换演示
    # 为简化，我们这里不处理关键点，只做图像本身的变换
    transform_rotate_scale = transforms.Compose([
        transf_mod.Scale(min_scale=0.7, max_scale=1.3), # 使用 min_scale 和 max_scale
        transf_mod.Rotate(pad=(128, 128, 128), max_rotate_degree=40), # 修改了参数名并调整了顺序
    ])
    transform_flip_crop = transforms.Compose([
        transf_mod.Flip(prob=1.0), # 强制翻转, 修改了参数名
        transf_mod.CropPad(pad=(128,128,128), crop_x=NET_INPUT_HEIGHT_SIZE, crop_y=NET_INPUT_HEIGHT_SIZE) # 修改了参数
    ])
    
    # 模拟 CocoTrainDataset 的 __getitem__ 返回的字典结构中的 'image'
    # ConvertKeypoints 需要 labels, 我们这里简化处理，直接操作图像
    # 真实的 ConvertKeypoints 会将图像转为 float32 并进行一些处理
    # 为了应用变换，我们先将图像转为 PIL Image (多数 torchvision transform期望此格式)
    # 然后再转回 OpenCV格式保存

    try:
        # 旋转和缩放
        # 变换通常期望 PIL Image 或 Tensor。这里假设它们能处理 numpy array 或我们手动转换
        # 为了简单起见，我们将自己实现简化的变换，或确保变换能处理cv2图像
        img_to_transform = img_orig.copy()
        # 对于Rotate, Scale等，它们在CocoTrainDataset中通常作用于包含 'image' 和 'keypoints'的字典
        # 此处我们仅为示意，直接应用，可能与原项目行为不完全一致
        
        # 简化的旋转和缩放应用 (实际变换可能更复杂)
        # 旋转
        h, w = img_to_transform.shape[:2]
        center = (w // 2, h // 2)
        angle = 30
        scale_sim = 0.8
        M_rotate = cv2.getRotationMatrix2D(center, angle, scale_sim)
        img_aug_rs = cv2.warpAffine(img_to_transform, M_rotate, (w, h), borderValue=(128,128,128))
        cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "example_train_augmented_rotate_scale.jpg"), img_aug_rs)

        # 翻转和裁剪
        img_flipped = cv2.flip(img_orig.copy(), 1) # 水平翻转
        # 简单中心裁剪示例
        target_h, target_w = int(h*0.8), int(w*0.8)
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        img_aug_fc = img_flipped[start_h:start_h+target_h, start_w:start_w+target_w]
        # 如果需要pad回原大小或特定大小
        img_aug_fc_padded = cv2.copyMakeBorder(img_aug_fc, 
                                           (h-target_h)//2 , (h-target_h) - (h-target_h)//2, 
                                           (w-target_w)//2, (w-target_w) - (w-target_w)//2, 
                                           cv2.BORDER_CONSTANT, value=(128,128,128))
        if img_aug_fc_padded.shape[0] > 0 and img_aug_fc_padded.shape[1] > 0 :
             cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "example_train_augmented_flip_crop.jpg"), img_aug_fc_padded)
        else:
            print("警告：数据增强裁剪后图像为空，跳过保存 example_train_augmented_flip_crop.jpg")


    except Exception as e:
        print(f"生成数据增强图片时出错: {e}")


def generate_inference_visualizations():
    if not NEURAL_NETWORK_MODULES_AVAILABLE: return
    net, device = get_model_and_device()
    if not net: return

    print("正在生成推理流程示例图片...")
    if not os.path.exists(DEFAULT_SAMPLE_IMAGE_PATH):
        print(f"错误：示例图片 {DEFAULT_SAMPLE_IMAGE_PATH} 未找到。")
        return

    orig_img = cv2.imread(DEFAULT_SAMPLE_IMAGE_PATH)
    if orig_img is None:
        print(f"错误：无法读取示例图片 {DEFAULT_SAMPLE_IMAGE_PATH}。")
        return
    
    cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "inference_input_original_example.jpg"), orig_img)

    # --- 阶段二: 数据处理 ---
    img_for_infer = orig_img.copy()
    # 从 infer_fast 借鉴预处理逻辑
    height, width, _ = img_for_infer.shape
    scale = NET_INPUT_HEIGHT_SIZE / height
    scaled_img_cv = cv2.resize(img_for_infer, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "inference_preprocessed_scaled_example.jpg"), scaled_img_cv)
    
    normed_img = val_mod.normalize(scaled_img_cv, np.array([128, 128, 128], np.float32), np.float32(1/256))
    min_dims = [NET_INPUT_HEIGHT_SIZE, max(normed_img.shape[1], NET_INPUT_HEIGHT_SIZE)]
    padded_img, pad_vals = val_mod.pad_width(normed_img, 8, (0,0,0), min_dims)
    # padded_img (归一化后) 不适合直接可视化为RGB，跳过保存

    # --- 阶段三: 模型推理 (获取Heatmaps, PAFs) ---
    try:
        print("执行模型推理以获取Heatmaps和PAFs...")
        heatmaps, pafs, _, _ = run_mod.infer_fast(net, orig_img.copy(), NET_INPUT_HEIGHT_SIZE, 8, 8, CPU_INFERENCE) # upsample_ratio=8
        
        # 可视化 Heatmaps (使用 run.py 中的 draw_heatmap)
        heatmap_viz = run_mod.draw_heatmap(heatmaps, orig_img.copy(), 8) # 改为 run_mod
        cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "inference_output_heatmaps_example.png"), heatmap_viz)
        print("Heatmap 可视化已保存。")

        # 可视化 PAFs (使用 run.py 中的 draw_paf)
        paf_viz = run_mod.draw_paf(pafs, orig_img.copy()) # 改为 run_mod
        cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "inference_output_pafs_example.png"), paf_viz)
        print("PAF 可视化已保存。")

        # --- 阶段四 & 五: 后处理和最终结果 ---
        print("执行后处理和姿态绘制...")
        num_keypoints = pose_mod.Pose.num_kpts
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += keypoints_mod.extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = keypoints_mod.group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * 8 / 8 - pad_vals[1]) / scale # stride/upsample_ratio = 1
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * 8 / 8 - pad_vals[0]) / scale

        img_final_pose = orig_img.copy()
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0: continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id_person in range(num_keypoints):
                if pose_entries[n][kpt_id_person] != -1.0:
                    pose_keypoints[kpt_id_person, 0] = int(all_keypoints[int(pose_entries[n][kpt_id_person]), 0])
                    pose_keypoints[kpt_id_person, 1] = int(all_keypoints[int(pose_entries[n][kpt_id_person]), 1])
            pose = pose_mod.Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            pose.draw(img_final_pose) # 在副本上绘制
        
        # 可选：混合原图和骨架
        # img_final_pose = cv2.addWeighted(orig_img.copy(), 0.4, img_final_pose, 0.6, 0)
        cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "inference_final_output_on_image_example.jpg"), img_final_pose)
        print("最终姿态绘制图片已保存。")

        # 对于"后处理在空白背景上"的图，我们可以创建一个黑色背景
        blank_bg = np.zeros_like(orig_img)
        for pose in current_poses:
            pose.draw(blank_bg)
        cv2.imwrite(os.path.join(VIZ_SUBFOLDER, "inference_postprocessed_poses_blank_bg_example.jpg"), blank_bg)
        print("空白背景上的姿态图已保存。")


    except Exception as e:
        print(f"生成推理可视化时出错: {e}")


def check_all_submodule_imports():
    """
    自动化检查 human_keypoints_detection 及其所有子模块的导入情况。
    """
    import importlib
    import pkgutil
    import traceback
    base_pkg = 'human_keypoints_detection'
    print("\n[自动化检查] 开始检查所有子模块导入...")
    failed = []
    try:
        pkg = importlib.import_module(base_pkg)
        for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, base_pkg + "."):
            try:
                importlib.import_module(name)
                print(f"[OK] {name}")
            except Exception as e:
                print(f"[FAIL] {name} 导入失败: {e}")
                print(traceback.format_exc())
                failed.append(name)
    except Exception as e:
        print(f"[FATAL] 无法导入主包 {base_pkg}: {e}")
        print(traceback.format_exc())
        return False
    if not failed:
        print("[自动化检查] 所有子模块导入正常！")
        return True
    else:
        print(f"[自动化检查] 有 {len(failed)} 个子模块导入失败。请检查上方详细信息。")
        return False

if __name__ == "__main__":
    print("自动化流程文档生成开始...")
    print(f"使用的默认检查点: {DEFAULT_CHECKPOINT_PATH}")
    print(f"使用的默认示例图片: {DEFAULT_SAMPLE_IMAGE_PATH}")
    
    # 新增：自动化检查所有子模块导入
    check_all_submodule_imports()
    
    ensure_viz_subfolder() # 创建或清理可视化图片输出目录

    if not NEURAL_NETWORK_MODULES_AVAILABLE:
        print("\n警告：由于项目模块导入失败，自动图像生成功能将无法运行。")
        print("文档将只包含流程图和描述，可视化部分会提示图片缺失。")
    else:
        # 尝试生成图片
        print("\n--- 开始自动生成可视化图片 ---")
        generate_training_data_augmentation_images()
        generate_inference_visualizations()
        print("--- 可视化图片自动生成结束 ---\n")

    print("\n--- 完成 ---")
    print(f"请检查 '{VIZ_SUBFOLDER}/' 目录中的图片。")
    print("确保 `DEFAULT_CHECKPOINT_PATH` 和 `DEFAULT_SAMPLE_IMAGE_PATH` 设置正确，并且相关文件存在。")
