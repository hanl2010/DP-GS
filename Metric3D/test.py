import glob
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from mmengine import Config, DictAction
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
import matplotlib
import cv2
from argparse import ArgumentParser


parser = ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--resolution', type=int, default=1)
parser.add_argument('--image_folder', type=str, default="images")
parser.add_argument("--get_normal", action="store_true", default=False)
parser.add_argument("--get_depth", action="store_true", default=False)
args = parser.parse_args()


cfg_large = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.large.py')
model_large = get_configured_monodepth_model(cfg_large, )
model_large, _,  _, _ = load_ckpt('./weight/metric_depth_vit_large_800k.pth', model_large, strict_match=False)
model_large.eval()

# cfg_small = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.small.py')
# model_small = get_configured_monodepth_model(cfg_small)
# model_small, _,  _, _ = load_ckpt('./weight/metric_depth_vit_small_800k.pth', model_small, strict_match=False)
# model_small.eval()

device = "cuda"
model_large.to(device)
# model_small.to(device)
model = model_large
cfg = cfg_large

cmap = matplotlib.colormaps.get_cmap('rainbow')


data_dir = args.data_dir
image_dir = args.image_folder
resolution = args.resolution

images_path = []
exts = [".png", ".jpg", ".jpeg"]
for ext in exts:
    images_path.extend(glob.glob(os.path.join(data_dir, f"{image_dir}/*{ext}")))

normal_output_path = os.path.join(data_dir, "normals_metric3d")
depth_output_path = os.path.join(data_dir, "depths_metric3d")
os.makedirs(normal_output_path, exist_ok=True)
os.makedirs(depth_output_path, exist_ok=True)

assert len(images_path) != 0

for image_file in tqdm(images_path):
    rgb = Image.open(image_file)
    if len(rgb.split()) > 3:
        rgb = np.stack([np.array(im) for im in rgb.split()[:3]], axis=-1)
    else:
        rgb = np.array(rgb)

    if resolution > 1:
        height, width, _ = rgb.shape
        rgb = cv2.resize(rgb, (width//resolution, height//resolution), interpolation=cv2.INTER_AREA)

    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb_tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb_tensor = torch.div((rgb_tensor - mean), std)
    rgb_tensor = rgb_tensor[None, :, :, :].cuda()

    # intrinsic = [fx, fy, rgb.shape[1] / 2, rgb.shape[0] / 2]
    # rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(rgb, intrinsic,
    #                                                                                       cfg.data_basic)

    with torch.no_grad():
        # pred_depth, pred_depth_scale, scale, output, confidence = get_prediction_v2(
        #     model=model,
        #     input=rgb_input,
        #     cam_model=cam_models_stacks,
        #     pad_info=pad,
        #     scale_info=label_scale_factor,
        #     gt_depth=None,
        #     normalize_scale=cfg.data_basic.depth_range[1],
        #     ori_shape=[rgb.shape[0], rgb.shape[1]],
        # )
        # pred_normal = output['normal_out_list'][0][:, :3, :, :]
        # H, W = pred_normal.shape[2:]
        # pred_normal = pred_normal[:, :, pad[0]:H - pad[1], pad[2]:W - pad[3]]

        pred_depth, confidence, output_dict = model.inference({'input': rgb_tensor})

    # if "20.png" in image_file:
    #     print(image_file)

    if args.get_normal:
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        pred_normal = torch.nn.functional.interpolate(pred_normal, [rgb.shape[0], rgb.shape[1]], mode='bilinear').squeeze()
        pred_normal = pred_normal.permute(1, 2, 0)
        pred_normal = pred_normal.cpu().numpy()
        pred_normal_vis = (pred_normal + 1) / 2

        plt.imsave(os.path.join(normal_output_path, os.path.basename(image_file)), pred_normal_vis)
        np.save(os.path.join(normal_output_path, os.path.basename(image_file).split(".")[0]+".npy"), pred_normal)
        # print(os.path.join(normal_output_path, os.path.basename(image_file)))

    if args.get_depth:
        pred_depth = torch.nn.functional.interpolate(pred_depth, [rgb.shape[0], rgb.shape[1]], mode='bilinear').squeeze()
        depth = pred_depth.squeeze().cpu().numpy()
        np.save(os.path.join(depth_output_path, os.path.basename(image_file).split(".")[0] + ".npy"), depth)
        plt.imsave(os.path.join(depth_output_path, os.path.basename(image_file)), depth, cmap="turbo")
        # print(os.path.join(depth_output_path, os.path.basename(image_file)))

