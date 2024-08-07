import os
import torch
from PIL import Image
import numpy as np
import faiss
import shutil
from tqdm import tqdm
import glob
import open_clip

# 解决 OpenMP 问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型（使用 ResNet-50 版本）
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
model.to(device)

def extract_features(image_path, model, preprocess, device):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().squeeze()

def get_video_and_scene_number(image_name):
    parts = image_name.split('_scene_')
    if len(parts) < 2:
        return None, None
    video_name = parts[0]
    scene_number = int(parts[1].split('_')[0])
    return video_name, scene_number

# 特征文件路径（使用 ResNet 提取的特征文件）
features_file = r'../file/features_file/video1_clip_resnet.npz'
new_image_folder = r'../file/frames/video1'
# 加载特征文件
data = np.load(features_file)
features_array = data['features']
image_paths = data['paths']

# 归一化特征向量
features_array_normalized = features_array / np.linalg.norm(features_array, axis=1, keepdims=True)

# 创建图像名称到新路径的字典
image_name_to_path = {os.path.basename(path): os.path.join(new_image_folder, os.path.basename(path)) for path in image_paths}

# 使用FAISS的内积索引
d = features_array_normalized.shape[1]
index = faiss.IndexFlatIP(d)
index.add(features_array_normalized)

# 处理模拟照片
simulated_photos_path =r'../file/cels'
output_results_path = r'../file/withbackground'

# 确保结果文件夹是空的
if os.path.exists(output_results_path):
    shutil.rmtree(output_results_path)
os.makedirs(output_results_path)

for img_path in tqdm(glob.glob(os.path.join(simulated_photos_path, '*.jpg')), desc="Processing simulated photos for CLIP"):
    img_name = os.path.basename(img_path)
    query_features = extract_features(img_path, model, preprocess, device)
    query_features_normalized = query_features / np.linalg.norm(query_features)

    # 计算与npz文件中其他图像的相似度
    query_features_exp = np.expand_dims(query_features_normalized, axis=0)
    distances, indices = index.search(query_features_exp, len(image_paths))

    selected_images = []
    selected_videos_scenes = set()
    saved_scenes = set()

    for idx in indices[0]:
        candidate_image_name = os.path.basename(image_paths[idx])
        candidate_video, candidate_scene = get_video_and_scene_number(candidate_image_name)

        if candidate_video is None or candidate_scene is None:
            continue

        skip_candidate = False
        for selected_video, selected_scene in selected_videos_scenes:
            if candidate_video == selected_video and abs(candidate_scene - selected_scene) <= 1:
                skip_candidate = True
                break

        if skip_candidate:
            continue

        # 跳过已保存的相同场景编号
        if candidate_scene in saved_scenes:
            continue

        selected_images.append((candidate_image_name, float(distances[0][idx])))
        selected_videos_scenes.add((candidate_video, candidate_scene))
        saved_scenes.add(candidate_scene)

        if len(selected_images) == 30:
            break

    # 为每张待检测的图片创建一个文件夹
    query_image_folder = os.path.join(output_results_path, os.path.splitext(img_name)[0])
    if not os.path.exists(query_image_folder):
        os.makedirs(query_image_folder)

    # 保存前30个相似图片到对应文件夹中
    for top_image_name, _ in selected_images:
        top_image_path = image_name_to_path[top_image_name]
        shutil.copy(top_image_path, os.path.join(query_image_folder, top_image_name))

    print(f"Processed {img_name}, saved top 30 matches to {query_image_folder}")

print("All images processed and results saved.")
