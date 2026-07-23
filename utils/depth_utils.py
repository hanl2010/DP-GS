# copy from gaussian-opacity-fields
# copy from 2DGS
import math
import torch
import numpy as np

def depths_to_points(view, depthmap, depth_mask=None):
    # c2w = (view.world_view_transform.T).inverse()
    # we train in camera coordinate
    c2w = torch.eye(4).float().cuda()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    if depth_mask is not None:
        depth_mask = depth_mask.reshape(-1)
        points = depthmap.reshape(-1, 1)[depth_mask] * rays_d[depth_mask] + rays_o
    else:
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points


def depth_to_normal_local(view, depth):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W / 2.],
         [0., fy, H / 2.],
         [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T
    points = depth.reshape(-1, 1) * rays_d
    points = points.reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def map_points_to_image(points, src_poses_inv, focal, center):
    """
    把光线上的点映射到图像上,得到图像上的坐标
    points: 光线上的点 （N, n_p, 3）
    src_poses_inv: 源图像的相机姿态的逆 (n_views, 4, 4)
    focal: 相机焦距 (2, ) (fx, fy)
    center: 图像中心 (2, ) (cx, cy)
    return: uv (n_view, N, n_p, 2)
    """
    # src_poses_inv = torch.inverse(src_poses)  # 得到姿态矩阵的逆
    points_rot = torch.matmul(src_poses_inv[:, None, None, :3, :3], points[None, :, :, :, None])[..., 0]
    points_trans = points_rot + src_poses_inv[:, None, None, :3, 3]  # (n_view, N, n_p, 3)
    uv = points_trans[..., :2] / points_trans[..., 2:]  # 转为Z轴坐标归一化的点， 与get_rays的过程相反
    uv *= focal[None, None, None, :]  # (n_view, N, n_p, 2)
    uv += center[None, None, None, :]  # (n_view, N, n_p, 2) # 得到图像中的像素坐标
    return uv

def point_world2depth(points, intrinsic_matrix, w2c):
    points_rot = torch.matmul(w2c[None, :3, :3], points[:, :, None])[..., 0]
    points_trans = points_rot + w2c[None, :3, 3]  # (n_p, 3)
    points_image = torch.matmul(intrinsic_matrix, points_trans[..., None])[..., 0]
    uv = points_image[:, :2] / points_image[:, 2:]
    depth = points_image[:, 2:]
    # focal = torch.tensor([intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]] ,device="cuda")
    # center = torch.tensor([intrinsic_matrix[0,2], intrinsic_matrix[1,2]] ,device="cuda")
    return uv, depth