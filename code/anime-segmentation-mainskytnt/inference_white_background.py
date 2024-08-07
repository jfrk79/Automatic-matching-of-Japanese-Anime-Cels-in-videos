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
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
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
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
        return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='isnet_is', choices=net_names, help='net name')
    parser.add_argument('--ckpt', type=str, default='saved_models/isnetis.ckpt', help='model checkpoint path')
    parser.add_argument('--data', type=str, default='../../dataset/anime-seg/test2', help='input data dir')
    parser.add_argument('--out', type=str, default='out', help='output dir')
    parser.add_argument('--img-size', type=int, default=1024, help='hyperparameter, input image size of the net')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')
    parser.add_argument('--fp32', action='store_true', default=False, help='disable mix precision')
    parser.add_argument('--only-matted', action='store_true', default=False, help='only output matted image')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)
    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, opt.device, img_size=opt.img_size)
    model.eval()
    model.to(device)

    if not os.path.exists(opt.out):
        os.mkdir(opt.out)

    for path in tqdm(sorted(glob.glob(f"{opt.data}/*.*"))):
        filename = os.path.basename(path)  # Extracts filename from path
        print(f"Processing file: {path}")
        if not os.path.isfile(path):
            print(f"Warning: File does not exist {path}")
            continue

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Unable to read image at {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = get_mask(model, img, use_amp=not opt.fp32, s=opt.img_size)

        mask_area = np.sum(mask > 0.5)
        img_area = img.shape[0] * img.shape[1]

        if mask_area < 0.1 * img_area:
            print(f"Mask area is less than 10% of the image area for {path}, saving original image.")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{opt.out}/{filename}', img)  # Save original image
        else:
            if opt.only_matted:
                white_background = np.ones_like(img) * 255
                img = mask * img + (1 - mask) * white_background
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{opt.out}/{filename}', img)  # Save using the original filename
            else:
                img = np.concatenate((img, mask * img, mask.repeat(3, 2) * 255), axis=1).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{opt.out}/{filename}', img)  # Save using the original filename
