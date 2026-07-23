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

import torch
import math
import numpy as np
from typing import NamedTuple
import torch.nn.functional as F
import open3d as o3d
from utils.depth_utils import depths_to_points
from utils.loss_utils import lncc


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_x, valid_y = torch.meshgrid(valid_x, valid_y, indexing='xy')
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world

def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape 
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    # xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
    _, xyz_cam = depth2point_cam(depth[None,None,None,...], intrinsic_matrix[None,...]) # (HxW, 3)
    xyz_cam = xyz_cam.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_cam, offset, gt_image)

    return xyz_normal

def normal_from_neareast(normal, offset):
    _, hd, wd = normal.shape 
    left_top_point = normal[..., 0:hd-2, 0:wd-2]
    top_point      = normal[..., 0:hd-2, 1:wd-1]
    right_top_point= normal[..., 0:hd-2, 2:wd]
    left_point   = normal[..., 1:hd-1, 0:wd-2]
    right_point  = normal[..., 1:hd-1, 2:wd]
    left_bottom_point   = normal[..., 2:hd, 0:wd-2]
    bottom_point = normal[..., 2:hd,   1:wd-1]
    right_bottom_point   = normal[..., 2:hd, 2:wd]
    normals = torch.stack((left_top_point,top_point,right_top_point,left_point,right_point,left_bottom_point,bottom_point,right_bottom_point),dim=0)
    new_normal = (normals * offset[:,None,1:-1,1:-1]).sum(0)
    new_normal = torch.nn.functional.normalize(new_normal, p=2, dim=0)
    new_normal = torch.nn.functional.pad(new_normal, (1,1,1,1), mode='constant').permute(1,2,0)
    return new_normal

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t,
                   translate=torch.tensor([.0, .0, .0], device="cuda", dtype=torch.float32),
                   scale=1.0):
    # Rt = np.zeros((4, 4))
    Rt = torch.zeros((4, 4), device="cuda", dtype=torch.float32)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # C2W = np.linalg.inv(Rt)
    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # Rt = np.linalg.inv(C2W)
    Rt = torch.inverse(C2W)
    # return np.float32(Rt)
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device="cuda")

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixCenterShift(znear, zfar, cx, cy, fl_x, fl_y, w, h):
    top = cy / fl_y * znear
    bottom = -(h - cy) / fl_y * znear

    left = -(w - cx) / fl_x * znear
    right = cx / fl_x * znear

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid


### code from IronDepth https://github.com/baegwangbin/IronDepth
def get_depth_candidate_weights(normal_prior, pos_arctan, patch_size, pad, center_idx):
    with torch.no_grad():
        B, _, H, W = normal_prior.shape
        # pred norm down - nghbr
        norm = F.pad(normal_prior, pad=(pad, pad, pad, pad), mode='replicate')
        norm_unfold = F.unfold(norm, (patch_size, patch_size), padding=0)                                     # (B, 3*ps*ps, H*W)
        norm_unfold = norm_unfold.reshape(B, 3, patch_size*patch_size, H, W)                                     # (B, 3, ps*ps, H, W)

        # pos - nghbr
        pos_nghbr = F.pad(pos_arctan, pad=(pad, pad, pad, pad), mode='replicate')
        pos_nghbr_unfold = F.unfold(pos_nghbr, (patch_size, patch_size), padding=0)                                         # (B, 2*ps*ps, H*W)
        pos_nghbr_unfold = pos_nghbr_unfold.view(B, 2, patch_size*patch_size, H, W)                                         # (B, 2, ps*ps, H, W)

        # norm and pos - nghbr
        nx, ny, nz = norm_unfold[:, 0, ...], norm_unfold[:, 1, ...], norm_unfold[:, 2, ...]       # (B, ps*ps, H, W) or (B, 1, H, W)
        pos_u, pos_v = pos_nghbr_unfold[:, 0, ...], pos_nghbr_unfold[:, 1, ...]                   # (B, ps*ps, H, W)

        # pos - center
        pos_u_center = pos_u[:, center_idx, :, :].unsqueeze(1)             # (B, 1, H, W)
        pos_v_center = pos_v[:, center_idx, :, :].unsqueeze(1)             # (B, 1, H, W)

        ddw_num = nx * pos_u + ny * pos_v + nz
        ddw_denom = nx * pos_u_center + ny * pos_v_center + nz
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom                                                                                       # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0               # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0   # inf
    return ddw_weights

### code from IronDepth https://github.com/baegwangbin/IronDepth
def get_pos(intrins, W, H):
    pos = torch.ones((H, W, 2), device=intrins.device)
    x_range = torch.cat([torch.arange(W).reshape(1, W)] * H, dim=0)
    y_range = torch.cat([torch.arange(H).reshape(H, 1)] * W, dim=1)
    pos[:, :, 0] = x_range + 0.5
    pos[:, :, 1] = y_range + 0.5
    pos[:, :, 0] = torch.arctan((pos[:, :, 0] - intrins[0, 2]) / intrins[0, 0])
    pos[:, :, 1] = torch.arctan((pos[:, :, 1] - intrins[1, 2]) / intrins[1, 1])
    # pos = torch.from_numpy(pos.astype(np.float32)).permute(2, 0, 1)
    pos = pos.permute(2, 0, 1)
    return pos


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale],
                                            intrinsic_matrix.to(depth.device),
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref
