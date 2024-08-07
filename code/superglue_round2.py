import os
import shutil
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

# 将SuperGlue代码目录添加到系统路径
import sys

sys.path.append('./SuperGluePretrainedNetwork-master')

from models.matching import Matching
from models.utils import frame2tensor


# 增强对比度
def enhance_contrast(image):
    # 应用CLAHE（对比度受限的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# 锐化图像
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# 使用Canny边缘检测将图像转换为黑色背景和白色边缘
def canny_edge_detection(image, low_threshold=30, high_threshold=100):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    # 将边缘转换为白色，背景为黑色
    edges_inv = cv2.bitwise_not(edges)  # 将边缘转换为白色，背景为黑色
    return edges_inv


# 处理目录中的所有图像
def process_images_in_directory(base_dir, output_dir, low_threshold=30, high_threshold=100):
    # 获取所有jpg图像文件，包括子目录
    image_paths = glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True)
    print(f"Total images found: {len(image_paths)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error reading image {img_path}")
            continue

        # 增强对比度
        img_contrast = enhance_contrast(img)

        # 锐化图像
        img_sharpen = sharpen_image(img_contrast)

        # 边缘检测并转换为黑色背景和白色边缘
        img_edges = canny_edge_detection(img_sharpen, low_threshold, high_threshold)

        # 将二值图像转换为三通道图像以保存为JPEG
        img_edges_color = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)

        # 生成新的文件路径
        new_img_path = os.path.join(output_dir, os.path.relpath(img_path, base_dir))
        new_img_dir = os.path.dirname(new_img_path)
        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)

        # 保存处理后的图像
        cv2.imwrite(new_img_path, img_edges_color)


# 初始化SuperGlue模型
def initialize_superglue():
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().cuda()
    return matching


# 调整图像尺寸为固定尺寸
def resize_images_to_fixed_size(img1, img2, size=(640, 480)):
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    return img1, img2


# 使用SuperGlue进行图像匹配
def superglue_match(img1_path, img2_path, matcher):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Error reading images {img1_path} or {img2_path}")
        return float('inf')

    # 调整图像尺寸为固定尺寸
    img1, img2 = resize_images_to_fixed_size(img1, img2)

    # 将图像转换为张量并转移到GPU
    img1_tensor = frame2tensor(img1, 'cuda')
    img2_tensor = frame2tensor(img2, 'cuda')

    # 进行匹配
    with torch.no_grad():
        pred = matcher({'image0': img1_tensor, 'image1': img2_tensor})

        try:
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()

            valid = matches > -1
            valid = valid[:min(len(kpts0), len(matches))]  # 确保索引在有效范围内
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
        except Exception as e:
            print(f"Error converting tensors to numpy: {e}")
            return float('inf')

    if len(mkpts0) == 0 or len(mkpts1) == 0:
        return float('inf')

    # 计算平均距离
    distances = np.linalg.norm(mkpts0 - mkpts1, axis=1)
    return np.mean(distances)


# 路径和目录设置
simulated_photos_path = r'../file/cels'
top30_results_path = r'../file/resullts_frames_extract'
processed_simulated_photos_path = r'../file/processed_cels'
processed_top30_results_path = r'../file/processed_frames_extract'
final_results_base_path = r'../file/resullts_final'

# 处理图像并保存到新的目录中
process_images_in_directory(simulated_photos_path, processed_simulated_photos_path)
process_images_in_directory(top30_results_path, processed_top30_results_path)

# 删除并重新创建结果目录
if os.path.exists(final_results_base_path):
    shutil.rmtree(final_results_base_path)
os.makedirs(final_results_base_path)

# 初始化SuperGlue
matcher = initialize_superglue()

folders = sorted(glob.glob(os.path.join(processed_top30_results_path, '*')))
print(f"Total folders found: {len(folders)}")
for folder in folders:
    print(f"Processing folder: {folder}")

total_images = 0
correct_matches = 0
incorrect_matches = []

# 创建字典以保存处理前图像路径和处理后图像路径的映射
original_to_processed = {}


# 递归遍历目录，构建原始路径到处理后路径的映射
def build_original_to_processed_mapping(original_dir, processed_dir):
    for root, _, files in os.walk(original_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                original_path = os.path.join(root, file)
                relative_path = os.path.relpath(original_path, original_dir)
                processed_path = os.path.join(processed_dir, relative_path)
                original_to_processed[processed_path] = original_path


# 构建原始路径到处理后路径的映射
build_original_to_processed_mapping(simulated_photos_path, processed_simulated_photos_path)
build_original_to_processed_mapping(top30_results_path, processed_top30_results_path)

for img_path in tqdm(glob.glob(os.path.join(processed_simulated_photos_path, '**', '*.jpg'), recursive=True),
                     desc="Processing SuperGlue similarity"):
    img_name = os.path.basename(img_path)
    img_name_wo_ext = os.path.splitext(img_name)[0]
    img_folder = os.path.join(processed_top30_results_path, img_name_wo_ext)

    if not os.path.exists(img_folder):
        print(f"Folder for {img_name} does not exist, skipping.")
        continue

    candidate_images = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith('.jpg')]
    print(f"Folder {img_folder} contains {len(candidate_images)} images. Debug: {candidate_images}")
    if not candidate_images:
        print(f"No images found in folder {img_folder}, skipping.")
        continue

    distances = []
    for candidate_img_path in candidate_images:
        try:
            distance = superglue_match(img_path, candidate_img_path, matcher)
            distances.append((candidate_img_path, distance))
            print(f"Matching {img_name} with {os.path.basename(candidate_img_path)}, distance: {distance}")
        except Exception as e:
            print(f"Error processing {candidate_img_path}: {e}")

    # 按距离排序
    distances.sort(key=lambda x: x[1])

    # 选择前30个匹配
    top_matches = distances[:30]

    result_folder = os.path.join(final_results_base_path, img_name_wo_ext)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for rank, (best_match_image, _) in enumerate(top_matches, start=1):
        original_best_match_image = original_to_processed[best_match_image]
        output_image_name = f"rank{rank}_{os.path.basename(original_best_match_image)}"
        shutil.copy(original_best_match_image, os.path.join(result_folder, output_image_name))
        print(f"Best match for {img_name} is {output_image_name}")

    total_images += 1

print("All images processed and SuperGlue results saved.")
