import os
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='isnet_is', choices=net_names, help='net name')
    parser.add_argument('--ckpt', type=str, default='saved_models/isnetis.ckpt', help='model checkpoint path')
    parser.add_argument('--data', type=str, help='input data dir')
    parser.add_argument('--out', type=str, default='out', help='output dir')
    parser.add_argument('--img-size', type=int, default=1024, help='hyperparameter, input image size of the net')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')
    parser.add_argument('--fp32', action='store_true', default=False, help='disable mix precision')
    parser.add_argument('--only-matted', action='store_true', default=False, help='only output matted image')

    opt = parser.parse_args()

    device = torch.device(opt.device)

    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, opt.device, img_size=opt.img_size)
    model.eval()
    model.to(device)

    if not os.path.exists(opt.out):
        os.makedirs(opt.out)

    for path in tqdm(sorted(glob.glob(f"{opt.data}/*.*"))):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Unable to load image {path}. Skipping...")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = get_mask(model, img, use_amp=not opt.fp32, s=opt.img_size)
        mask = mask.squeeze()  # Ensure mask is in 2D if needed

        print(f"Processing {path}: Mask min={np.min(mask)}, max={np.max(mask)}")  # Debug: Output min and max of mask

        # Apply mask to set background to white
        img[mask < 0.5] = [255, 255, 255]  # Assuming the threshold for background is 0.5

        # Use original filename for the output
        filename = os.path.basename(path)
        output_path = os.path.join(opt.out, filename)
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Processed image saved to {output_path}")  # Debug: Confirm image save location
