import argparse

import cv2
import numpy as np
import torch
import time
import warnings

from human_keypoints_detection.models.with_mobilenet import PoseEstimationWithMobileNet
from human_keypoints_detection.modules.keypoints import extract_keypoints, group_keypoints
from human_keypoints_detection.modules.load_state import load_state
from human_keypoints_detection.modules.pose import Pose
from human_keypoints_detection.val import normalize, pad_width

from PIL import Image, ImageDraw, ImageFont


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV需要整数来读取摄像头
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def cv2_add_chinese_text(img, text, position, font_path, font_size, text_color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")

    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = tensor_img.to(device)

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def draw_heatmap(heatmaps, img, upsample_ratio):
    heatmap_image = np.zeros_like(img)
    for kpt_idx in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, kpt_idx]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_image = cv2.addWeighted(heatmap_image, 1.0, heatmap, 0.5, 0)

    # 将热力图与原图按50%透明度混合
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_image, alpha, 0)
    return overlay


# def draw_paf(pafs, img, upsample_ratio):
#     paf_image = np.zeros_like(img)
#     for paf_idx in range(pafs.shape[-1] // 2):
#         paf = pafs[:, :, paf_idx * 2:paf_idx * 2 + 2]
#         paf = cv2.resize(paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
#         magnitude = np.sqrt(paf[..., 0] ** 2 + paf[..., 1] ** 2)
#         magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#         paf_image[..., paf_idx % 3] = magnitude.astype(np.uint8)
#     return paf_image


def draw_paf(pafs, orig_img):
    h, w = orig_img.shape[:2]
    # 调整PAF到原图尺寸
    paf_resized = cv2.resize(pafs, (w, h), interpolation=cv2.INTER_CUBIC)
    num_channels = paf_resized.shape[2]
    assert num_channels == 38, "PAF必须包含38个通道"

    # 重塑PAF形状为(19个连接, 2个方向)
    pafs_reshaped = paf_resized.reshape((h, w, 19, 2))

    # 计算每个连接的向量长度
    magnitudes = np.linalg.norm(pafs_reshaped, axis=3)
    # 找到每个像素点响应最强的连接
    max_indices = np.argmax(magnitudes, axis=2)

    # 提取每个像素点的最大响应向量
    rows, cols = np.indices((h, w))
    max_paf = pafs_reshaped[rows, cols, max_indices, :]

    # 计算向量角度和强度
    angles = np.arctan2(max_paf[..., 1], max_paf[..., 0])  # 弧度值
    magnitudes = np.linalg.norm(max_paf, axis=2)

    # 转换为HSV颜色空间（H:方向，S:最大，V:强度）
    hue = (np.degrees(angles) % 180).astype(np.uint8)  # 色调表示方向
    saturation = np.full_like(hue, 255, dtype=np.uint8)  # 饱和度设为最大
    value = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 明度表示强度

    hsv = np.stack([hue, saturation, value], axis=-1)
    paf_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 与原图叠加显示（50%透明度）
    alpha = 0.5
    overlay = cv2.addWeighted(orig_img, 1 - alpha, paf_image, alpha, 0)
    return overlay


def run_demo(net, image_provider, height_size, cpu, smooth, image_path, show_heatmap, show_paf):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    net = net.eval().to(device)
    processing_time_ms = 0

    stride = 8
    upsample_ratio = 8
    num_keypoints = Pose.num_kpts
    delay = 0 if image_path != '' else 1

    fourcc = cv2.VideoWriter_fourcc(*'hev1')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, size)

    cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
    if show_heatmap:
        cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL)
    if show_paf:
        cv2.namedWindow('PAFs', cv2.WINDOW_NORMAL)

    for img in image_provider:
        if img is None or (isinstance(img, np.ndarray) and img.size == 0):
            warnings.warn("图片未找到或无法读取，已跳过。");
            continue
        start_time = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        if show_heatmap:
            heatmap_img = draw_heatmap(heatmaps, orig_img, upsample_ratio)
            cv2.imshow('heatmap', heatmap_img)
        if show_paf:
            paf_img = draw_paf(pafs, orig_img)
            cv2.imshow('PAFs', paf_img)

        cv2.imshow('keypoints', orig_img)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        # Draw poses on the image
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.4, img, 0.8, 0)

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        # Add English annotations
        cv2.putText(img, 'Green: Keypoints', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, 'Red: Right Side', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, 'Yellow: Left Side', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, f'Processing time: {processing_time_ms:.1f} ms', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('keypoints', img)
        out.write(img)

        key = cv2.waitKey(delay)
        if key == 27:  # ESC
            break
        elif key == 112:  # P
            delay = 0 if delay == 1 else 1

    cv2.destroyAllWindows()
    # print(f"处理时间：{processing_time_ms}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人体骨骼关键点检测系统')
    parser.add_argument('--checkpoint-path', type=str,
                        default='train_checkpoints/1/checkpoint_iter_420000.pth',
                        help='模型参数路径')
    parser.add_argument('--height-size', type=int, default=256,
                        help='网络输入高度尺寸')
    parser.add_argument('--video', type=str, default='',
                        help='视频文件路径或摄像头ID')
    parser.add_argument('--images', nargs='+', type=str, default='',
                        help='输入图片路径列表')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU进行计算')
    parser.add_argument('--smooth', type=int, default=1,
                        help='人体关键点平滑参数')
    parser.add_argument('--show-heatmap', action='store_true',
                        help='热力图可视化')
    parser.add_argument('--show-paf', action='store_true',
                        help='PAF向量场可视化')

    args = parser.parse_args()

    if not args.video and not args.images:
        raise ValueError("必须提供视频或图片输入")

    # 初始化网络
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    # # 初始化输入源
    # if args.images:
    #     frame_provider = ImageReader(args.images)
    # else:
    #     frame_provider = VideoReader(args.video)

    global size
    frame_provider = ImageReader(args.images)
    # 读取图片
    if args.images != '':
        image_path = args.images[0]
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        size = (width, height)
    # 是否读取视频
    if args.video != '':
        frame_provider = VideoReader(args.video)
        capture = cv2.VideoCapture(args.video)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    run_demo(net, frame_provider, args.height_size, args.cpu, args.smooth, args.images, args.show_heatmap, args.show_paf)
