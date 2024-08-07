import os
import subprocess

def run_inference(net, ckpt, data_dir, out_dir, img_size=1024, device='cuda:0', fp32=False, only_matted=False):
    # 构建命令行参数列表
    command = [
        'python', 'anime-segmentation-mainskytnt/inference_white_background.py',
        '--net', net,
        '--ckpt', ckpt,
        '--data', data_dir,
        '--out', out_dir,
        '--img-size', str(img_size),
        '--device', device
    ]

    if fp32:
        command.append('--fp32')

    if only_matted:
        command.append('--only-matted')

    # 打印即将运行的命令
    print('Running command:', ' '.join(command))

    # 运行命令
    subprocess.run(command)

if __name__ == "__main__":
    # 手动输入路径
    net = 'isnet_is'
    ckpt = 'anime-segmentation-mainskytnt/saved_models/isnetis.ckpt'
    data_dir = '../file/frames/video1'
    out_dir = '../file/frames_seg/video1'
    img_size = 1024
    device = 'cuda:0'
    fp32 = False
    only_matted = True

    run_inference(net, ckpt, data_dir, out_dir, img_size, device, fp32, only_matted)
