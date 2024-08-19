import json, yaml
import numpy as np
import torch
from pycg import vis
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d


from src.utils.graphics_utils import getProjectionMatrix2, focal2fov
from src.gsr.se3.numpy_se3 import transform
from src.gsr.camera import Camera

def read_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    
def read_trajectory(file_path, slam_config, scale = 1., device = "cuda:0"):
    """read trajectory data from plot output to list of Cameras [to use in 3DGS]

    Args:
        file_path (string): path
    """
    trj_data = dict()
    json_data = read_json_data(file_path)
    dataset = slam_config['Dataset']['type']
    
    if json_data:
        # Access the data as needed
        trj_data["trj_id"] = json_data["trj_id"]
        trj_data["trj_est"] = torch.Tensor(json_data["trj_est"]).to(device).float()
        trj_data["trj_gt"] = torch.Tensor(json_data["trj_gt"]).to(device).float()
        # print("Trajectory loaded successfully!")
    
    cam_dict = dict()
    calibration = slam_config["Dataset"]["Calibration"]
    
    proj_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx = calibration["fx"] / scale,
        fy = calibration["fy"] / scale,
        cx = calibration["cx"] / scale,
        cy = calibration["cy"] / scale,
        W = calibration["width"] / scale,
        H = calibration["height"] / scale,
    ).T
    
    fovx = focal2fov(calibration['fx'], calibration['width'])
    fovy = focal2fov(calibration['fy'], calibration['height'])
    for id, est_pose, gt_pose in zip(trj_data["trj_id"], trj_data["trj_est"], trj_data["trj_gt"]):
        T_gt = torch.linalg.inv(gt_pose)
        cam_i = Camera(id, None, None,
                   T_gt, 
                   proj_matrix, 
                   calibration["fx"]/ scale, 
                   calibration["fy"]/ scale, 
                   calibration["cx"]/ scale, 
                   calibration["cy"]/ scale, 
                   fovx, 
                   fovy, 
                   calibration["height"]/ scale, 
                   calibration["width"]/ scale)
        est_T = torch.linalg.inv(est_pose)
        cam_i.R = est_T[:3, :3]
        cam_i.T = est_T[:3, 3]
        
        cam_dict[id] = cam_i

    return cam_dict

def visualize_registration(src_3dgs, tgt_3dgs, pre_tsfm, gt_tsfm):
    """visualize registration in 3D Gaussians

    Args:
        gs3d_src (Gaussian Model): _description_
        gs3d_tgt (Gaussian Model): _description_
        tsfm (torch.Tensor): _description_
    """
    src_pc = src_3dgs.get_xyz().detach().cpu().numpy()
    tgt_pc = tgt_3dgs.get_xyz().detach().cpu().numpy()
    est_src_pc = transform(pre_tsfm.detach().cpu().numpy(), src_pc)
    gt_src_pc = transform(gt_tsfm.detach().cpu().numpy(), src_pc)
    
    src_vis = vis.pointcloud(src_pc[::2], ucid=0, cmap='tab10', is_sphere=True)
    tgt_vis = vis.pointcloud(tgt_pc[::2], ucid=1, cmap='tab10', is_sphere=True)
    
    est_vis = vis.pointcloud(est_src_pc[::2], ucid=0, cmap='tab10', is_sphere=True)
    gt_vis = vis.pointcloud(gt_src_pc[::2], ucid=0, cmap='tab10', is_sphere=True)
    
    try:
        vis.show_3d([src_vis, tgt_vis], [est_vis, tgt_vis], [gt_vis, tgt_vis],  use_new_api=True, show=True)
    except:
        print("estimate is not a good transformation")
        vis.show_3d([src_vis, tgt_vis], [gt_vis, tgt_vis],  use_new_api=True, show=True)
    
def visualize_mv_registration(gaussian_list, pred_rel_tsfms, gt_rel_tsfms):
    vis_pred_list, vis_gt_list = [], []
    pred_abs_tsfm = torch.eye(4).cuda()
    gt_abs_tsfm = torch.eye(4).cuda()
    for i, gaussians in enumerate(gaussian_list[:5]):
        pc = gaussians.get_xyz().detach().cpu().numpy()
        
        pc_pred = transform(pred_abs_tsfm.detach().cpu().numpy(), pc)
        vis_pred_list.append(vis.pointcloud(pc_pred, ucid=i, cmap='tab10', is_sphere=True))
        if i>0: pred_abs_tsfm = pred_abs_tsfm @ pred_rel_tsfms[i-1]
        
        pc_gt = transform(gt_abs_tsfm.detach().cpu().numpy(), pc)
        vis_gt_list.append(vis.pointcloud(pc_gt, ucid=i, cmap='tab10', is_sphere=True))
        if i>0: gt_abs_tsfm = gt_abs_tsfm @ gt_rel_tsfms[i-1]
        
    vis.show_3d(vis_pred_list, vis_gt_list, use_new_api=True, show=True)



def axis_angle_to_rot_mat(axes, thetas):
    """
    Computer a rotation matrix from the axis-angle representation using the Rodrigues formula.
    \mathbf{R} = \mathbf{I} + (sin(\theta)\mathbf{K} + (1 - cos(\theta)\mathbf{K}^2), where K = \mathbf{I} \cross \frac{\mathbf{K}}{||\mathbf{K}||}

    Args:
    axes (numpy array): array of axes used to compute the rotation matrices [b,3]
    thetas (numpy array): array of angles used to compute the rotation matrices [b,1]

    Returns:
    rot_matrices (numpy array): array of the rotation matrices computed from the angle, axis representation [b,3,3]

    borrowed from: https://github.com/zgojcic/3D_multiview_reg/blob/master/lib/utils.py
    """

    R = []
    for k in range(axes.shape[0]):
        K = np.cross(np.eye(3), axes[k,:]/np.linalg.norm(axes[k,:]))
        R.append( np.eye(3) + np.sin(thetas[k])*K + (1 - np.cos(thetas[k])) * np.matmul(K,K))

    rot_matrices = np.stack(R)
    return rot_matrices

def sample_random_trans(pcd, randg=None, rotation_range=360):
    """
    Samples random transformation paramaters with the rotaitons limited to the rotation range

    Args:
    pcd (numpy array): numpy array of coordinates for which the transformation paramaters are sampled [n,3]
    randg (numpy random generator): numpy random generator

    Returns:
    T (numpy array): sampled transformation paramaters [4,4]
    
    borrowed from: https://github.com/zgojcic/3D_multiview_reg/blob/master/lib/utils.py
    """
    if randg == None:
        randg = np.random.default_rng(41)

    # Create 3D identity matrix
    T = np.zeros((4,4))
    idx = np.arange(4)
    T[idx,idx] = 1
    
    axes = np.random.rand(1,3) - 0.5

    angles = rotation_range * np.pi / 180.0 * (np.random.rand(1,1) - 0.5)

    R = axis_angle_to_rot_mat(axes, angles)

    T[:3, :3] = R
    # T[:3, 3] = np.random.rand(3)-0.5
    T[:3, 3]  = np.matmul(R,-np.mean(pcd, axis=0))

    return T

def visualize_mv_registration(data_list, pred_pose_list):
    n_view = len(pred_pose_list)
    vis_init_list, vis_pred_list, vis_gt_list = [], [], []
    for i in range(n_view):
        pc = data_list[i]['gaussians'].get_xyz().detach().cpu().numpy()
        pred_pc = transform(pred_pose_list[i].detach().cpu().numpy(), pc)
        
        gt_tsfm = data_list[0]['gt_tsfm'] @ data_list[i]['gt_tsfm'].inverse()
        gt_pc = transform(gt_tsfm.detach().cpu().numpy(), data_list[i]['gaussians'].get_xyz().detach().cpu().numpy())   
        
        vis_pred_list.append(vis.pointcloud(pred_pc[::2], ucid=i, is_sphere=True))
        vis_gt_list.append(vis.pointcloud(gt_pc[::2], ucid=i, is_sphere=True))
        vis_init_list.append(vis.pointcloud(pc[::2], ucid=i, is_sphere=True))
    
    vis.show_3d(vis_init_list, vis_pred_list, vis_gt_list, use_new_api=True)
    

@torch.no_grad()
def plot_and_save(points, pngname, title='', axlim=None):
    points = points.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:,0], points[:,1], points[:,2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    plt.savefig(pngname)
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def visualize_camera_traj(cam_list):
    vis_cam_list = []
    for cam in cam_list:
        intrinsic = np.array([
            [cam.fx, 0.0, cam.cx],
            [0.0, cam.fy, cam.cy],
            [0.0, 0.0, 1.0]
            ])
        pred_extrinsic = cam.get_T.cpu().numpy()
        gt_extrinsic = cam.get_T_gt.cpu().numpy()
        vis_pred_cam = o3d.geometry.LineSet.create_camera_visualization(
            640, 480, intrinsic, pred_extrinsic, scale=0.1)
        vis_gt_cam = o3d.geometry.LineSet.create_camera_visualization(
            640, 480, intrinsic, gt_extrinsic, scale=0.1)
        
        # Set colors for predicted (blue) and ground truth (green) cameras
        vis_pred_cam.paint_uniform_color([1, 0, 0])
        vis_gt_cam.paint_uniform_color([0, 1, 0])
        
        vis_cam_list.append(vis_pred_cam)
        vis_cam_list.append(vis_gt_cam)
    
    return vis_cam_list