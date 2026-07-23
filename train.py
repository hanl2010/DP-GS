#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp, get_depth_candidate_weights, get_pos
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def training(dataset, opt, pipe, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    # cmd = f'cp ./train.py {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./arguments {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./scene {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./utils {dataset.model_path}/'
    # os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=None)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()


    ps = 5
    pad = (ps - 1) // 2
    center_idx = (ps * ps - 1) // 2

    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    image_height = viewpoint_stack[0].image_height
    image_width = viewpoint_stack[0].image_width
    H = image_height
    W = image_width
    print(f"image shape: ({H}, {W})")


    ix, iy = torch.meshgrid(torch.arange(image_width), torch.arange(image_height), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().cuda()
    all_pixel_indices = torch.arange(image_width * image_height).cuda()

    occ_mask_all = {}
    for view in viewpoint_stack:
        occ_mask_all[view.image_name] = torch.ones(size=(image_height, image_width)).cuda()
    d_mask = torch.ones(size=(image_height, image_width)).cuda().bool()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, app_model=app_model,
                            return_plane=True,
                            return_depth_normal=iteration > opt.single_view_weight_from_iter)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg["plane_depth"]

        intrinsic, extrinsic = viewpoint_cam.get_calib_matrix_nerf()
        gt_normal = viewpoint_cam.gt_normal.cuda()
        gt_normal_edge = viewpoint_cam.gt_normal_edge.cuda()

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            depth = render_pkg["plane_depth"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)

        # multi-view loss
        ncc_confidence_all = None
        if iteration > opt.multi_view_weight_from_iter:
            if len(viewpoint_cam.nearest_id) == 0:
                nearest_cam = None
            else:
                nearest_cam = scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]

            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2

                ## compute geometry consistency mask and loss
                nearest_render_pkg = render(nearest_cam, gaussians, pipe, background, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[3, :3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'],
                                                                        pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                R = nearest_cam.R
                T = nearest_cam.T
                pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,
                                         :3] + viewpoint_cam.world_view_transform[3, :3]
                pts_projections = torch.stack(
                    [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                     pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy], -1).float()

                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0

                ncc_confidence_all = torch.zeros((image_height * image_width, 3), device=d_mask.device, dtype=torch.float32)
                # Cumulative occ_mask
                occ_mask_all[viewpoint_cam.image_name] = occ_mask_all[viewpoint_cam.image_name] * (~d_mask.reshape(image_height, image_width))


                if d_mask.sum() > 0:
                    with torch.no_grad():
                        d_mask = d_mask.reshape(-1)
                        valid_indices = all_pixel_indices[d_mask]

                        if d_mask.sum() > sample_num:
                            index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                            valid_indices = valid_indices[index]

                        weights = weights.reshape(-1)[valid_indices]
                        ## sample ref frame patch
                        pixels_selected = pixels.reshape(-1,2)[valid_indices]
                        offsets = patch_offsets(patch_size, pixels.device)
                        ori_pixels_patch = pixels_selected.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                        pixels_patch = ori_pixels_patch.clone()
                        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (image_width - 1) - 1.0
                        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (image_height - 1) - 1.0
                        # ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                        # ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
                        ref_val = F.grid_sample(gt_image.unsqueeze(0), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                        ref_val = ref_val.reshape(3, -1, total_patch_size)

                        ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                    ## compute Homography
                    ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                    ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                    ref_local_d = render_pkg['rendered_distance'].squeeze()
                    # rays_d = viewpoint_cam.get_rays()
                    # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                    # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                    # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                    H_ref_to_neareast = ref_to_neareast_r[None] - \
                        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1),
                                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                    H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                    H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                    ## compute neareast frame patch
                    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (image_width - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (image_height - 1) - 1.0
                    nearest_image, nearest_image_gray = nearest_cam.get_image()
                    # sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                    # sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                    sampled_val = F.grid_sample(nearest_image[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                    sampled_val = sampled_val.reshape(3, -1, total_patch_size)
                    ref_val = ref_val.permute(1,0,2).reshape(-1, total_patch_size)
                    sampled_val = sampled_val.permute(1,0,2).reshape(-1, total_patch_size)

                    ## compute loss
                    # ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                    ncc, ncc_mask = lncc(ref_val, sampled_val) # ncc_mask=ncc<0.9
                    mask = ncc_mask.reshape(-1, 3)
                    ncc_confidence = (2.0 - ncc.reshape(-1, 3)) / 2.0 * weights[:, None]
                    ncc_confidence_all[valid_indices] = ncc_confidence
                    ncc = ncc.reshape(-1, 3) * weights[:, None]

                    if mask.sum() > 0:
                        ncc_loss = opt.multi_view_ncc_weight * ncc.mean()
                        loss += ncc_loss


        ## normal guided depth propagation
        normal_guided_depth_loss = 0.0
        if iteration > args.normal_guided_start_iter and ncc_confidence_all is not None:
            ncc_confidence_mean = torch.mean(ncc_confidence_all.reshape(image_height, image_width, 3).permute(2, 0, 1), dim=0, keepdim=True)
            depth_mask = (ncc_confidence_mean > args.ncc_confidence_threshold).float()
            intrinsic, extrinsic = viewpoint_cam.get_calib_matrix_nerf()

            if depth_mask.sum()>0:
                pos = get_pos(intrins=intrinsic, W=image_width, H=image_height)
                depth_candidate_weights = get_depth_candidate_weights(normal_prior=gt_normal.permute(2,0,1).unsqueeze(0), pos_arctan=pos.unsqueeze(0), patch_size=ps, pad=pad, center_idx=center_idx)

                with torch.no_grad():
                    depth_pred = depth
                    depth_mask_tmp = depth_mask
                    # start_time = time.time()
                    for i in range(args.depth_propagation_iters):
                        depth_pred = depth_pred * depth_mask_tmp
                        pred_depth_pad = F.pad(depth_pred.unsqueeze(0), pad=(pad, pad, pad, pad), mode='replicate')
                        pred_depth_unfold = F.unfold(pred_depth_pad, (ps, ps), dilation=1, padding=0)  # (B, ps*ps, H*W)
                        pred_depth_unfold = pred_depth_unfold.view(1, ps * ps, H, W)  # (B, ps*ps, H, W)
                        new_depth_candidates = depth_candidate_weights * pred_depth_unfold

                        depth_mask_pad = F.pad(depth_mask_tmp.unsqueeze(0), pad=(pad, pad, pad, pad), mode='replicate')
                        depth_mask_unfold = F.unfold(depth_mask_pad, (ps, ps), dilation=1, padding=0)  # (B, ps*ps, H*W)
                        depth_mask_unfold = depth_mask_unfold.view(1, ps * ps, H, W)  # (B, ps*ps, H, W)
                        depth_mask_candidates = depth_candidate_weights * depth_mask_unfold

                        depth_pred = torch.sum(new_depth_candidates, dim=1, keepdim=False) / (torch.sum(depth_mask_candidates > 0, dim=1, keepdim=False)+1e-6)

                        ## "Normal edge" is used to prevent the "depth" from propagating to areas with significant differences in "normal" values.
                        depth_mask_tmp = torch.mean(depth_mask_candidates * gt_normal_edge.unsqueeze(1), dim=1, keepdim=False)
                        depth_mask_tmp = (depth_mask_tmp > 0.01).float()
                    # print(f"depth propagation time: {time.time()-start_time:.4f} seconds")
                normal_guided_depth_loss = torch.sum(torch.abs(depth - depth_pred) * depth_mask_tmp) / torch.sum(depth_mask_tmp)
                normal_guided_depth_loss = normal_guided_depth_loss * args.normal_guided_depth_weight
                loss += normal_guided_depth_loss


        depth_smooth_loss = 0.0
        if iteration > args.depth_smooth_iter_start and iteration < args.depth_smooth_iter_end:
            depth_rendered = depth.reshape(-1, 1)
            occ_depth_mask = occ_mask_all[viewpoint_cam.image_name]

            ## get mask of depth edge ##
            grad_depth_left = torch.mean(torch.abs(depth[:, 1:-1, 1:-1] - depth[:, 1:-1, :-2]), 0)
            grad_depth_right = torch.mean(torch.abs(depth[:, 1:-1, 1:-1] - depth[:, 1:-1, 2:]), 0)
            grad_depth_top = torch.mean(torch.abs(depth[:, 1:-1, 1:-1] - depth[:, :-2, 1:-1]), 0)
            grad_depth_bottom = torch.mean(torch.abs(depth[:, 1:-1, 1:-1] - depth[:, 2:, 1:-1]), 0)
            max_grad = torch.max(torch.stack([grad_depth_left, grad_depth_right, grad_depth_top, grad_depth_bottom], dim=-1), dim=-1)[0]
            # pad
            depth_edge = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
            depth_edge_mask = (depth_edge > 0.01).float()

            edge_mask = depth_edge_mask * gt_normal_edge.reshape(image_height, image_width)
            edge_mask = edge_mask.reshape(-1)

            # Convert the "valid depth" to points in the camera space
            valid_indices = all_pixel_indices[edge_mask.bool()]
            pixels_valid = pixels.reshape(-1, 2)[valid_indices]
            depth_valid = depth_rendered[valid_indices]
            cam_xyz = torch.cat([pixels_valid * depth_valid, depth_valid], dim=-1)
            cam_xyz = cam_xyz @ torch.inverse(intrinsic).T

            ## Sampling around the "valid pixels"
            random_xy = torch.randint(1, 5, size=(len(valid_indices), 2), device=occ_depth_mask.device)
            pixels_nearest = pixels_valid + random_xy
            pixels_nearest[:, 0] = torch.clip(pixels_nearest[:, 0], 0, image_width-1)
            pixels_nearest[:, 1] = torch.clip(pixels_nearest[:, 1], 0, image_height-1)

            ## Convert the sampling to points in the camera space
            nearest_indices = (pixels_nearest[:, 0] + pixels_nearest[:, 1] * image_width).long()
            depth_nearest = depth_rendered[nearest_indices]
            cam_xyz_nearest = torch.cat([pixels_nearest*depth_nearest, depth_nearest], dim=-1)
            cam_xyz_nearest = cam_xyz_nearest @ torch.inverse(intrinsic).T

            ## Make the sampling points within a single plane.
            gt_normal_sample = gt_normal.reshape(-1, 3)[valid_indices]
            gt_normal_nearest = gt_normal.reshape(-1, 3)[nearest_indices]
            normal_sim=torch.sum(gt_normal_sample * gt_normal_nearest, dim=-1)
            normal_sim = normal_sim*0.5 + 0.5 # [0, 1]
            vectors = F.normalize(cam_xyz - cam_xyz_nearest, dim=-1)
            depth_smooth_loss = torch.mean(normal_sim*normal_sim*torch.abs(torch.sum(vectors * gt_normal_sample, dim=-1)))
            depth_smooth_loss = depth_smooth_loss * args.depth_smooth_weight
            loss += depth_smooth_loss

        gt_normal_loss = 0.0
        if iteration > args.prior_normal_iter:
            normal_local = render_pkg["rendered_normal"]
            gt_normal_loss = args.prior_normal_weight * (1 - torch.mean(torch.sum(normal_local.permute(1,2,0) * gt_normal, dim=-1)))
            loss += gt_normal_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{image_loss:.5f}",
                    "depth_prop": f"{normal_guided_depth_loss:.5f}",
                    "gt_normal": f"{gt_normal_loss:.5f}",
                    "depth_smooth": f"{depth_smooth_loss:.5f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, scene, render, (pipe, background), app_model)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, 0.0, args.test_iterations, scene, render, (pipe, background), app_model)
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if args.opacity_decay and iteration > opt.densify_from_iter and iteration < args.opacity_decay_end_iter:
                gaussians.opacity_decay(factor=args.opacity_decay_factor)
                if args.opacity_decay_end_iter > opt.densify_until_iter:
                    opt.densify_until_iter = args.opacity_decay_end_iter

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if args.opacity_decay:
                        size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)


            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, background, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if args.opacity_decay:
                        pass
                    else:
                        gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])


    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dataset_name", type=str, default="DTU")
    parser.add_argument("--sfm_type", type=str, default="colmap") # colmap or mast3r

    parser.add_argument("--prior_normal_iter", type=int, default=3000)
    parser.add_argument("--prior_normal_weight", type=float, default=0.1)
    parser.add_argument("--opacity_decay", action="store_true", default=False)
    parser.add_argument("--opacity_decay_factor", type=float, default=0.995)
    parser.add_argument("--opacity_decay_end_iter", type=int, default=10000)
    parser.add_argument("--normal_guided_start_iter", type=float, default=4000)
    parser.add_argument("--depth_smooth_iter_start", type=int, default=1000)
    parser.add_argument("--depth_smooth_iter_end", type=int, default=10000)
    parser.add_argument("--depth_smooth_weight", type=float, default=0.03)
    parser.add_argument("--ncc_confidence_threshold", type=float, default=0.6)
    parser.add_argument("--normal_guided_depth_weight", type=float, default=0.5)
    parser.add_argument("--depth_propagation_iters", type=int, default=10)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    args.densify_until_iter = 7000
    args.single_view_weight_from_iter = 3000
    args.multi_view_weight_from_iter = 3000
    args.multi_view_sample_num = 10000
    args.use_multi_view_trim = False


    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
