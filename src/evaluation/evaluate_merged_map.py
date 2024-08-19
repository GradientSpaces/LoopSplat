""" This module is responsible for merging submaps. """
from argparse import ArgumentParser

import faiss
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.utils import (batch_search_faiss, get_render_settings,
                             np2ptcloud, render_gaussian_model, torch2np)
from src.utils.gaussian_model_utils import BasicPointCloud


class RenderFrames(Dataset):
    """A dataset class for loading keyframes along with their estimated camera poses and render settings."""
    def __init__(self, dataset, render_poses: np.ndarray, height: int, width: int, fx: float, fy: float, exposures_ab=None):
        self.dataset = dataset
        self.render_poses = render_poses
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.device = "cuda"
        self.stride = 1
        self.exposures_ab = exposures_ab
        if len(dataset) > 1000:
            self.stride = len(dataset) // 1000

    def __len__(self) -> int:
        return len(self.dataset) // self.stride

    def __getitem__(self, idx):
        idx = idx * self.stride
        color = (torch.from_numpy(
            self.dataset[idx][1]) / 255.0).float().to(self.device)
        depth = torch.from_numpy(self.dataset[idx][2]).float().to(self.device)
        estimate_c2w = self.render_poses[idx]
        estimate_w2c = np.linalg.inv(estimate_c2w)
        frame = {
            "frame_id": idx,
            "color": color,
            "depth": depth,
            "render_settings": get_render_settings(
                self.width, self.height, self.dataset.intrinsics, estimate_w2c)
        }
        if self.exposures_ab is not None:
            frame["exposure_ab"] = self.exposures_ab[idx]
        return frame


def merge_submaps(submaps_paths: list, radius: float = 0.0001, device: str = "cuda") -> o3d.geometry.PointCloud:
    """ Merge submaps into a single point cloud, which is then used for global map refinement.
    Args:
        segments_paths (list): Folder path of the submaps.
        radius (float, optional): Nearest neighbor distance threshold for adding a point. Defaults to 0.0001.
        device (str, optional): Defaults to "cuda".

    Returns:
        o3d.geometry.PointCloud: merged point cloud
    """
    pts_index = faiss.IndexFlatL2(3)
    if device == "cuda":
        pts_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 500, faiss.METRIC_L2))
        pts_index.nprobe = 5
    merged_pts = []
    for submap_path in tqdm(submaps_paths, desc="Merging submaps"):
        gaussian_params = torch.load(submap_path)["gaussian_params"]
        current_pts = gaussian_params["xyz"].to(device).float().contiguous()
        pts_index.train(current_pts)
        distances, _ = batch_search_faiss(pts_index, current_pts, 8)
        neighbor_num = (distances < radius).sum(axis=1).int()
        ids_to_include = torch.where(neighbor_num == 0)[0]
        pts_index.add(current_pts[ids_to_include])
        merged_pts.append(current_pts[ids_to_include])
    pts = torch2np(torch.vstack(merged_pts))
    pt_cloud = np2ptcloud(pts, np.zeros_like(pts))

    # Downsampling if the total number of points is too large
    if len(pt_cloud.points) > 1_000_000:
        voxel_size = 0.02
        pt_cloud = pt_cloud.voxel_down_sample(voxel_size)
        print(f"Downsampled point cloud to {len(pt_cloud.points)} points")
    filtered_pt_cloud, _ = pt_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    del pts_index
    return filtered_pt_cloud


def refine_global_map(pt_cloud: o3d.geometry.PointCloud, training_frames: list, max_iterations: int,
                      export_refine_mesh=False, output_dir=".",
                      len_frames=None, o3d_intrinsic=None, enable_sh=True, enable_exposure=False) -> GaussianModel:
    """Refines a global map based on the merged point cloud and training keyframes frames.
    Args:
        pt_cloud (o3d.geometry.PointCloud): The merged point cloud used for refinement.
        training_frames (list): A list of training frames for map refinement.
        max_iterations (int): The maximum number of iterations to perform for refinement.
    Returns:
        GaussianModel: The refined global map as a Gaussian model.
    """
    opt_params = OptimizationParams(ArgumentParser(description="Training script parameters"))

    gaussian_model = GaussianModel(3)
    gaussian_model.active_sh_degree = 0
    if pt_cloud is None:
        output_mesh = output_dir / "mesh" / "cleaned_mesh.ply"
        output_mesh = o3d.io.read_triangle_mesh(str(output_mesh))
        pcd = o3d.geometry.PointCloud()
        pcd.points = output_mesh.vertices
        pcd.colors = output_mesh.vertex_colors
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd = BasicPointCloud(points=np.asarray(pcd.points),
                            colors=np.asarray(pcd.colors))
        gaussian_model.create_from_pcd(pcd, 1.0)
        gaussian_model.training_setup(opt_params)
    else:
        gaussian_model.training_setup(opt_params)
        gaussian_model.add_points(pt_cloud)

    iteration = 0
    for iteration in tqdm(range(max_iterations), desc="Refinement"):
        training_frame = next(training_frames)
        gaussian_model.update_learning_rate(iteration)
        if enable_sh and iteration > 0 and iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()
        gt_color, gt_depth, render_settings = (
            training_frame["color"].squeeze(0),
            training_frame["depth"].squeeze(0),
            training_frame["render_settings"])

        render_dict = render_gaussian_model(gaussian_model, render_settings)
        rendered_color, rendered_depth = (render_dict["color"].permute(1, 2, 0), render_dict["depth"])
        if enable_exposure and training_frame.get("exposure_ab") is not None:
            rendered_color = torch.clamp(
                rendered_color * torch.exp(training_frame["exposure_ab"][0,0]) + training_frame["exposure_ab"][0,1], 0, 1.)

        reg_loss = isotropic_loss(gaussian_model.get_scaling())
        depth_mask = (gt_depth > 0)
        color_loss = (1.0 - opt_params.lambda_dssim) * l1_loss(
            rendered_color[depth_mask, :], gt_color[depth_mask, :]
        ) + opt_params.lambda_dssim * (1.0 - ssim(rendered_color, gt_color))
        depth_loss = l1_loss(
            rendered_depth[:, depth_mask], gt_depth[depth_mask])

        total_loss = color_loss + depth_loss + reg_loss
        total_loss.backward()

        with torch.no_grad():
            if iteration % 500 == 0:
                prune_mask = (gaussian_model.get_opacity() < 0.005).squeeze()
                gaussian_model.prune_points(prune_mask)

            # Optimizer step
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)
        iteration += 1
    
    try:
        if export_refine_mesh:
            output_dir = output_dir / "mesh" / "refined_mesh.ply"
            scale = 1.0
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=5.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for i in tqdm(range(len_frames), desc="Integrating mesh"):  # one cycle
                training_frame = next(training_frames)
                gt_color, gt_depth, render_settings, estimate_w2c = (
                    training_frame["color"].squeeze(0),
                    training_frame["depth"].squeeze(0),
                    training_frame["render_settings"],
                    training_frame["estimate_w2c"])

                render_dict = render_gaussian_model(gaussian_model, render_settings)
                rendered_color, rendered_depth = (
                    render_dict["color"].permute(1, 2, 0), render_dict["depth"])
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

                rendered_color = (
                    torch2np(rendered_color) * 255).astype(np.uint8)
                rendered_depth = torch2np(rendered_depth.squeeze())
                # rendered_depth = filter_depth_outliers(
                #     rendered_depth, kernel_size=20, threshold=0.1)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                    o3d.geometry.Image(rendered_depth),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                volume.integrate(
                    rgbd, o3d_intrinsic, estimate_w2c.squeeze().cpu().numpy().astype(np.float64))

            o3d_mesh = volume.extract_triangle_mesh()
            o3d.io.write_triangle_mesh(str(output_dir), o3d_mesh)
            print(f"Refined mesh saved to {output_dir}")

    except Exception as e:
        print(f"Error export_refine_mesh in refine_global_map:\n {e}")

    return gaussian_model
