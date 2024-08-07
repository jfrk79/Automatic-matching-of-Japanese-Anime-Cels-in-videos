import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


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
def process_images_in_directory(base_dir, low_threshold=30, high_threshold=100):
    # 获取所有jpg图像文件，包括子目录
    image_paths = glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True)
    print(f"Total images found: {len(image_paths)}")

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

        # 保存处理后的图像
        cv2.imwrite(img_path, img_edges_color)


if __name__ == "__main__":
    # 目录路径设置
    base_dir = '.'  # 当前目录

    # 处理目录中的所有图像，调整阈值以提取更多边缘
    process_images_in_directory(base_dir, low_threshold=70, high_threshold=160)
