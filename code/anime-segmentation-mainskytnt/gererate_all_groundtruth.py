import os
import cv2
import torch
import numpy as np
import glob
from torch.cuda import amp
from tqdm import tqdm
from train import AnimeSegmentation, net_names

def get_mask(model, input_img, use_amp=True, s=640):
    input_img = (input_img / 255).astype(np.float32)
    h, w = h0, w0 = input_img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))
        return pred

def process_folder(model, folder_path, img_size, use_amp, threshold):
    # Skip the 'ground_truth' folder
    if 'ground_truth' in folder_path:
        return

    # Create the output directory based on the folder name and threshold
    folder_name = os.path.basename(folder_path)
    output_dir = os.path.join(folder_path, f"{folder_name}_ground_truth_skytnt_{threshold:.1f}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in tqdm(sorted(glob.glob(f"{folder_path}/*.*"))):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Unable to load image {path}. Skipping...")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = get_mask(model, img, use_amp=use_amp, s=img_size)
        mask = mask.squeeze()  # Ensure mask is in 2D if needed

        print(f"Processing {path}: Mask min={np.min(mask)}, max={np.max(mask)}")  # Debug: Output min and max of mask

        # Ensure mask is in binary form
        binary_mask = (mask > threshold).astype(np.uint8) * 255  # Convert to binary and scale to 0-255

        # Create a new image with the same size as the input
        result_img = np.zeros_like(img)
        result_img[binary_mask == 255] = [255, 255, 255]  # White for person
        result_img[binary_mask == 0] = [0, 0, 0]  # Black for background

        # Use original filename for the output
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result_img)
        print(f"Processed image saved to {output_path}")  # Debug: Confirm image save location

if __name__ == "__main__":
    # Set parameters directly in the script
    net = 'isnet_is'
    ckpt = r'G:\project\char_seg\anime-segmentation-mainskytnt\saved_models\isnetis.ckpt'
    data_root = r'G:\project\dataset_classfication'
    img_size = 1024
    device = 'cuda:0'
    use_fp32 = False  # Corresponds to --fp32 flag

    device = torch.device(device)

    model = AnimeSegmentation.try_load(net, ckpt, device, img_size=img_size)
    model.eval()
    model.to(device)

    # Traverse through each folder in the dataset root directory
    for root, dirs, files in os.walk(data_root):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for sub_dir_name in os.listdir(dir_path):
                sub_dir_path = os.path.join(dir_path, sub_dir_name)
                if os.path.isdir(sub_dir_path) and 'ground_truth' not in sub_dir_path:
                    for threshold in np.arange(0.1, 1.0, 0.1):
                        process_folder(model, sub_dir_path, img_size, use_amp=not use_fp32, threshold=threshold)
