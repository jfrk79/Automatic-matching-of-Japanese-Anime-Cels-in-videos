import os
import torch
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import open_clip

# 加载OpenAI CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')  # 使用 ResNet-50 版本
model.to(device)  # 将模型移动到指定设备

def extract_features(img_path, model, preprocess, device):
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)  # 将图像移动到指定设备
    with torch.no_grad():
        features = model.encode_image(img)
    return features.cpu().numpy().squeeze()

# 获取当前文件的目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 手动输入路径
dataset_path = '../file/frames_seg/video1'  # 数据集路径
features_dir = '../file/features_file'  # 保存特征文件的路径

# 提取保存文件名
dataset_name = os.path.basename(dataset_path)
save_filename = f'{dataset_name}_clip_resnet.npz'
save_path = os.path.join(features_dir, save_filename)

dataset_features = []
img_paths = []

for img_path in tqdm(glob.glob(f'{dataset_path}/*.jpg'), desc="Extracting features"):
    try:
        features = extract_features(img_path, model, preprocess, device)
        dataset_features.append(features)
        img_paths.append(img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

dataset_features = np.array(dataset_features)

# 确保保存特征文件的目录存在
os.makedirs(features_dir, exist_ok=True)

# 保存特征和路径
np.savez(save_path, features=dataset_features, paths=img_paths)
print(f"Features and paths saved to {save_path}")
