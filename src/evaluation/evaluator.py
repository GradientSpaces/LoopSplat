""" This module is responsible for evaluating rendering, trajectory and reconstruction metrics"""
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torchvision
import json
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.evaluation.evaluate_merged_map import (RenderFrames, merge_submaps,
                                                refine_global_map)
from src.evaluation.evaluate_reconstruction import evaluate_reconstruction, clean_mesh
from src.evaluation.evaluate_trajectory import evaluate_trajectory
from src.utils.io_utils import load_config, save_dict_to_json, log_metrics_to_wandb
from src.utils.mapper_utils import calc_psnr
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, setup_seed, 
                             torch2np, filter_depth_outliers)


class Evaluator(object):

    def __init__(self, checkpoint_path, config_path, config=None, save_render=False) -> None:
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        setup_seed(self.config["seed"])

        self.checkpoint_path = Path(checkpoint_path)
        self.use_wandb = self.config["use_wandb"]
        self.device = "cuda"
        self.dataset = get_dataset(self.config["dataset_name"])({**self.config["data"], **self.config["cam"]})
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = np.array(self.dataset.poses)
        self.fx, self.fy = self.dataset.intrinsics[0, 0], self.dataset.intrinsics[1, 1]
        self.cx, self.cy = self.dataset.intrinsics[0,
                                                   2], self.dataset.intrinsics[1, 2]
        self.width, self.height = self.dataset.width, self.dataset.height
        self.save_render = save_render
        if self.save_render:
            self.render_path = self.checkpoint_path / "rendered_imgs"
            self.render_path.mkdir(exist_ok=True, parents=True)

        self.estimated_c2w = torch2np(torch.load(self.checkpoint_path / "estimated_c2w.ckpt", map_location=self.device))
        self.submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*')))
        self.exposures_ab = None
        if (self.checkpoint_path / "exposures_ab.ckpt").exists():
            self.exposures_ab = torch2np(torch.load(self.checkpoint_path / "exposures_ab.ckpt", map_location=self.device))
            print(f"Loaded trained exposures paramters for scene {self.scene_name}")

    def run_trajectory_eval(self):
        """ Evaluates the estimated trajectory """
        print("Running trajectory evaluation...")
        evaluate_trajectory(self.estimated_c2w, self.gt_poses, self.checkpoint_path)

    def run_rendering_eval(self):
        """ Renders the submaps and global splats and evaluates the PSNR, LPIPS, SSIM and depth L1 metrics."""
        print("Running rendering evaluation...")
        psnr, lpips, ssim, depth_l1 = [], [], [], []
        color_transform = torchvision.transforms.ToTensor()
        lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(self.device)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        submaps_paths = sorted(
            list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:

                _, gt_color, gt_depth, _ = self.dataset[keyframe_id]
                gt_color = color_transform(gt_color).to(self.device)
                gt_depth = np2torch(gt_depth).to(self.device)

                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_dict = render_gaussian_model(
                    gaussian_model, get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c))
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"][0].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                if self.save_render:
                    torchvision.utils.save_image(
                        rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                mse_loss = torch.nn.functional.mse_loss(
                    rendered_color, gt_color)
                psnr_value = (-10. * torch.log10(mse_loss)).item()
                lpips_value = lpips_model(
                    rendered_color[None], gt_color[None]).item()
                ssim_value = ms_ssim(
                    rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                depth_l1_value = torch.abs(
                    (rendered_depth - gt_depth)).mean().item()

                psnr.append(psnr_value)
                lpips.append(lpips_value)
                ssim.append(ssim_value)
                depth_l1.append(depth_l1_value)

        num_frames = len(psnr)
        metrics = {
            "psnr": sum(psnr) / num_frames,
            "lpips": sum(lpips) / num_frames,
            "ssim": sum(ssim) / num_frames,
            "depth_l1_train_view": sum(depth_l1) / num_frames,
            "num_renders": num_frames
        }
        save_dict_to_json(metrics, "rendering_metrics.json",
                          directory=self.checkpoint_path)

        x = list(range(len(psnr)))
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].plot(x, psnr, label="PSNR")
        axs[0].legend()
        axs[0].set_title("PSNR")
        axs[1].plot(x, ssim, label="SSIM")
        axs[1].legend()
        axs[1].set_title("SSIM")
        axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
        axs[2].legend()
        axs[2].set_title("Depth L1 Render")
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_path /
                    "rendering_metrics.png"), dpi=300)
        print(metrics)

    def run_reconstruction_eval(self):
        """ Reconstructs the mesh, evaluates it, render novel view depth maps from it, and evaluates them as well """
        print("Running reconstruction evaluation...")

        (self.checkpoint_path / "mesh").mkdir(exist_ok=True, parents=True)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:
                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_dict = render_gaussian_model(
                    gaussian_model, get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c))
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"][0].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

                rendered_color = (
                    torch2np(rendered_color.permute(1, 2, 0)) * 255).astype(np.uint8)
                rendered_depth = torch2np(rendered_depth)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                    o3d.geometry.Image(rendered_depth),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic, estimate_w2c)

        o3d_mesh = volume.extract_triangle_mesh()
        compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                             scale / 512.0, -2.5 * scale / 512.0)
        o3d_mesh = o3d_mesh.translate(compensate_vector)
        o3d_mesh = clean_mesh(o3d_mesh)
        file_name = self.checkpoint_path / "mesh" / "cleaned_mesh.ply"
        o3d.io.write_triangle_mesh(str(file_name), o3d_mesh)
        print(f"Reconstructed mesh saved to {file_name}")
        if self.config["dataset_name"] == "replica":
            evaluate_reconstruction(file_name,
                                    f"data/Replica-SLAM/cull_replica/{self.scene_name}/gt_mesh_cull_virt_cams.ply",
                                    f"data/Replica-SLAM/cull_replica/{self.scene_name}/gt_pc_unseen.npy",
                                    self.checkpoint_path)


    def run_global_map_eval(self, init_from='mesh'):
        """ Merges the map, evaluates it over training and novel views 
        
        Args:
            init_from (str, optional): 'mesh' or 'splats'. Initialization method for the global splats. Defaults to mesh vertices reconstructed before.
        """
        print("Running global map evaluation...")

        training_frames = RenderFrames(
            self.dataset, self.estimated_c2w, self.height, self.width, self.fx, self.fy, self.exposures_ab)
        training_frames = DataLoader(
            training_frames, batch_size=1, shuffle=True)
        len_frames = len(training_frames)
        training_frames = cycle(training_frames)
        merged_cloud = merge_submaps(self.submaps_paths) if init_from == 'splats' else None

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        refined_merged_gaussian_model = refine_global_map(merged_cloud, training_frames, 30000, export_refine_mesh=False,
                                                          output_dir=self.checkpoint_path, len_frames=len_frames, 
                                                          o3d_intrinsic=intrinsic)
        ply_path = self.checkpoint_path / \
            f"{self.config['data']['scene_name']}_global_splats.ply"
        refined_merged_gaussian_model.save_ply(ply_path)
        print(f"Refined global splats saved to {ply_path}")

        if self.config["dataset_name"] == "scannetpp":
            # "NVS evaluation only supported for scannetpp"

            eval_config = deepcopy(self.config)
            print(f"✨ Eval NVS for scene {self.config['data']['scene_name']}...")
            (self.checkpoint_path / "nvs_eval").mkdir(exist_ok=True, parents=True)
            eval_config["data"]["use_train_split"] = False
            test_set = get_dataset(eval_config["dataset_name"])({**eval_config["data"], **eval_config["cam"]})
            test_poses = torch.stack([torch.from_numpy(test_set[i][3]) for i in range(len(test_set))], dim=0)
            test_frames = RenderFrames(test_set, test_poses, self.height, self.width, self.fx, self.fy)

            psnr_list = []
            for i in tqdm(range(len(test_set))):
                gt_color, _, render_settings = (
                    test_frames[i]["color"],
                    test_frames[i]["depth"],
                    test_frames[i]["render_settings"])
                render_dict = render_gaussian_model(refined_merged_gaussian_model, render_settings)
                rendered_color, _ = (render_dict["color"].permute(1, 2, 0), render_dict["depth"],)
                rendered_color = torch.clip(rendered_color, 0, 1)
                save_image(rendered_color.permute(2, 0, 1), self.checkpoint_path / f"nvs_eval/{i:04d}.jpg")
                psnr = calc_psnr(gt_color, rendered_color)
                psnr_list.append(psnr.item())
            print(f"PSNR List: {psnr_list}")
            print(f"Avg. NVS PSNR: {np.array(psnr_list).mean()}")
            with open(self.checkpoint_path / 'nvs_eval' / "results.json", "w") as f:
                data = {"avg_nvs_psnr": np.mean(psnr_list)}
                json.dump(data, f, indent=4)
        
        else: # evaluate rendering performance on the global submap
            print("Running rendering evaluation on global map ...")
            psnr, lpips, ssim, depth_l1 = [], [], [], []
            color_transform = torchvision.transforms.ToTensor()
            lpips_model = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=True).to(self.device)
            
            submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
            for submap_path in tqdm(submaps_paths):
                submap = torch.load(submap_path, map_location=self.device)

                for keyframe_id in submap["submap_keyframes"]:

                    _, gt_color, gt_depth, _ = self.dataset[keyframe_id]
                    gt_color = color_transform(gt_color).to(self.device)
                    gt_depth = np2torch(gt_depth).to(self.device)

                    estimate_c2w = self.estimated_c2w[keyframe_id]
                    estimate_w2c = np.linalg.inv(estimate_c2w)
                    render_dict = render_gaussian_model(
                        refined_merged_gaussian_model, get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c))
                    rendered_color, rendered_depth = render_dict["color"].detach(
                    ), render_dict["depth"][0].detach()
                    rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                    if self.save_render:
                        torchvision.utils.save_image(
                            rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                    mse_loss = torch.nn.functional.mse_loss(
                        rendered_color, gt_color)
                    psnr_value = (-10. * torch.log10(mse_loss)).item()
                    lpips_value = lpips_model(
                        rendered_color[None], gt_color[None]).item()
                    ssim_value = ms_ssim(
                        rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                    depth_l1_value = torch.abs(
                        (rendered_depth - gt_depth)).mean().item()

                    psnr.append(psnr_value)
                    lpips.append(lpips_value)
                    ssim.append(ssim_value)
                    depth_l1.append(depth_l1_value)

            num_frames = len(psnr)
            metrics = {
                "psnr": sum(psnr) / num_frames,
                "lpips": sum(lpips) / num_frames,
                "ssim": sum(ssim) / num_frames,
                "depth_l1_train_view": sum(depth_l1) / num_frames,
                "num_renders": num_frames
            }
            save_dict_to_json(metrics, "rendering_metrics_global.json",
                            directory=self.checkpoint_path)

            x = list(range(len(psnr)))
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].plot(x, psnr, label="PSNR")
            axs[0].legend()
            axs[0].set_title("PSNR")
            axs[1].plot(x, ssim, label="SSIM")
            axs[1].legend()
            axs[1].set_title("SSIM")
            axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
            axs[2].legend()
            axs[2].set_title("Depth L1 Render")
            plt.tight_layout()
            plt.savefig(str(self.checkpoint_path /
                        "rendering_metrics_global.png"), dpi=300)
            print(metrics)

    def run(self):
        """ Runs the general evaluation flow """

        print("Starting evaluation...🍺")

        try:
            self.run_trajectory_eval()
        except Exception:
            print("Could not run trajectory eval")
            traceback.print_exc()

        try:
            self.run_reconstruction_eval()
        except Exception:
            print("Could not run reconstruction eval")
            traceback.print_exc()
            
        try:
            self.run_global_map_eval()
        except Exception:
            print("Could not run global map eval")
            traceback.print_exc()

        if self.use_wandb: 
            evals = ["rendering_metrics.json", "reconstruction_metrics.json", "ate_aligned.json", "nvs_eval/results.json"]
            log_metrics_to_wandb(evals, self.checkpoint_path, "Evaluation")
