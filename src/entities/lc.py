from argparse import ArgumentParser
import os, glob
import copy
import time
import numpy as np
import torch, roma
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.entities.gaussian_model import GaussianModel
from src.entities.arguments import OptimizationParams

from src.gsr.descriptor import GlobalDesc
from src.gsr.camera import Camera
from src.gsr.solver import gaussian_registration as gs_reg
from src.gsr.pcr import (preprocess_point_cloud, execute_global_registration)

from src.utils.utils import np2torch, torch2np
from src.utils.graphics_utils import getProjectionMatrix2, focal2fov
from src.utils.eval_utils import eval_ate

class PGO_Edge:
    def __init__(self, src_id, tgt_id, overlap=0.):
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.overlap_ratio = overlap
        self.success = False
        self.transformation = np.identity(4)
        self.information = np.identity(6)
        self.transformation_gt = np.identity(4)
        
    def __str__(self) -> str:
        return f"source_id : {self.s}, target_id : {self.t}, success : {self.success}, \
            transformation : {self.transformation}, information : {self.information}, \
            overlap_ratio : {self.overlap_ratio}, transformation_gt : {self.transformation_gt}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
class Loop_closure(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Initializes the LC with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the loop closure process.
        """
        self.device = "cuda"
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.netvlad = GlobalDesc()
        self.submap_lc_info = dict()
        self.submap_id = 0
        self.submap_path = None
        self.pgo_count = 0
        self.n_loop_edges = 0
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.proj_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx = self.config["cam"]["fx"],
                fy = self.config["cam"]["fx"],
                cx = self.config["cam"]["cx"],
                cy = self.config["cam"]["cy"],
                W = self.config["cam"]["W"],
                H = self.config["cam"]["H"],
            ).T
        self.fovx = focal2fov(self.config["cam"]["fx"], self.config["cam"]["W"])
        self.fovy = focal2fov(self.config["cam"]["fy"], self.config["cam"]["H"])
        self.min_interval = self.config['lc']['min_interval']
        
        # TODO: rename below
        self.config["Training"] = {"edge_threshold": 4.0}
        self.config["Dataset"] = {"type": "replica"}
        
        self.max_correspondence_distance_coarse = self.config['lc']['voxel_size'] * 15
        self.max_correspondence_distance_fine = self.config['lc']['voxel_size'] * 1.5
        
    def update_submaps_info(self, keyframes_info):
        """Update the submaps_info with current submap 

        Args:
            keyframes_info (dict): a dictionary of all submap information for loop closures
        """
        with torch.no_grad():
            kf_ids, submap_desc = [], []
            for key in keyframes_info.keys():
                submap_desc.append(self.netvlad(np2torch(self.dataset[key][1], self.device).permute(2, 0, 1)[None]/255.0))
            submap_desc = torch.cat(submap_desc)
            self_sim = torch.einsum("id,jd->ij", submap_desc, submap_desc)
            score_min, _ = self_sim.topk(max(int(len(submap_desc) * self.config["lc"]["min_similarity"]), 1))
            
        self.submap_lc_info[self.submap_id] = {
                "submap_id": self.submap_id,
                "kf_id": np.array(sorted(list(keyframes_info.keys()))),
                "kf_desc": submap_desc,
                "self_sim": score_min, # per image self similarity within the submap
            }
    
    def submap_loader(self, id: int):
        """load submap data for loop closure

        Args:
            id (int): submap id to load
        """
        submap_dict = torch.load(self.submap_paths[id], map_location=torch.device(self.device))
        gaussians = GaussianModel(sh_degree=0)
        gaussians.restore_from_params(submap_dict['gaussian_params'], self.opt)
        submap_cams = []
        
        for kf_id in submap_dict['submap_keyframes']:
            _, rgb, depth, c2w_gt = self.dataset[kf_id]
            c2w_est = self.c2ws_est[kf_id]
            T_gt = torch.from_numpy(c2w_gt).to(self.device).inverse()
            T_est = torch.linalg.inv(c2w_est).to(self.device)
            cam_i = Camera(kf_id, None, None,
                   T_gt, 
                   self.proj_matrix, 
                   self.config["cam"]["fx"],
                   self.config["cam"]["fx"],
                   self.config["cam"]["cx"],
                   self.config["cam"]["cy"],
                   self.fovx, 
                   self.fovy, 
                   self.config["cam"]["H"], 
                   self.config["cam"]["W"])
            cam_i.R = T_est[:3, :3]
            cam_i.T = T_est[:3, 3]
            rgb_path = self.dataset.color_paths[kf_id]
            depth_path = self.dataset.depth_paths[kf_id]
            depth = np.array(Image.open(depth_path)) / self.config['cam']['depth_scale']
            cam_i.depth = depth
            cam_i.rgb_path = rgb_path
            cam_i.depth_path = depth_path
            cam_i.config = self.config
            submap_cams.append(cam_i)
        
        data_dict = {
            "submap_id": id,
            "gaussians": gaussians,
            "kf_ids": submap_dict['submap_keyframes'],
            "cameras": submap_cams,
            "kf_desc": self.submap_lc_info[id]['kf_desc']
            }
        
        return data_dict
    
    def detect_closure(self, query_id: int, final=False):
        """detect closure given a submap_id, we only match it to the submaps before it

        Args:
            query_id (int): the submap id used to detect closure

        Returns:
            torch.Tensor: 1d vector of matched submap_id
        """
        n_submaps = self.submap_id + 1
        query_info = self.submap_lc_info[query_id]
        iterator = range(query_id+1, n_submaps) if final else range(query_id)
        db_info_list = [self.submap_lc_info[i] for i in iterator]
        db_desc_map_id = []
        for db_info in db_info_list:
            db_desc_map_id += [db_info['submap_id'] for _ in db_info['kf_desc']]
        db_desc_map_id = torch.Tensor(db_desc_map_id).to(self.device)
        
        query_desc = query_info['kf_desc']
        db_desc = torch.cat([db_info['kf_desc'] for db_info in db_info_list])
        
        with torch.no_grad():
            cross_sim = torch.einsum("id,jd->ij", query_desc, db_desc)
            self_sim = query_info['self_sim']
            matches = torch.argwhere(cross_sim > self_sim[:,[-1]])[:,-1]
            matched_map_ids = db_desc_map_id[matches].long().unique()
        
        # filter out invalid matches
        filtered_mask = abs(matched_map_ids - query_id) > self.min_interval
        matched_map_ids = matched_map_ids[filtered_mask]
                
        return matched_map_ids
    
    def construct_pose_graph(self, final=False):
        """Build the pose graph from detected loops

        Returns:
            _type_: _description_
        """
        n_submaps = self.submap_id + 1
        pose_graph = o3d.pipelines.registration.PoseGraph()
        submap_list = []
        
        # initialize pose graph node from odometry with identity matrix
        for i in range(n_submaps):
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
            submap_list.append(self.submap_loader(i))
        
        # log info for edge analysis
        self.cam_dict = dict()
        self.kf_ids, self.kf_submap_ids = [], []
        for submap in submap_list:
            for cam in submap['cameras']:
                self.kf_submap_ids.append(submap["submap_id"])
                self.kf_ids.append(cam.uid)
                self.cam_dict[cam.uid] = copy.deepcopy(cam)
        self.kf_submap_ids = np.array(self.kf_submap_ids)
        
        odometry_edges, loop_edges = [], []
        new_submap_valid_loop = False
        for source_id in tqdm(reversed(range(1, n_submaps))):
            matches = self.detect_closure(source_id, final)
            iterator = range(source_id+1, n_submaps) if final else range(source_id)
            for target_id in iterator:
                if abs(target_id - source_id)== 1: # odometry edge
                    reg_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "identity")
                    transformation = reg_dict['transformation']
                    information = reg_dict['information']
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation,
                                                                information,
                                                                uncertain=False))
                    # analyse 
                    gt_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "gt")
                    ae = roma.rotmat_geodesic_distance(torch.from_numpy(gt_dict['transformation'][:3,:3]), torch.from_numpy(reg_dict['transformation'][:3, :3])) * 180 /torch.pi
                    te = np.linalg.norm(gt_dict['transformation'][:3,3] - reg_dict["transformation"][:3,3])
                    odometry_edges.append((source_id, target_id, ae.item(), te.item()))
                    # TODO: update odometry edge with the PGO_edge class

                elif target_id in matches: # loop closure edge
                    reg_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "gs_reg")
                    if not reg_dict['successful']: continue
                    
                    if np.isnan(reg_dict["transformation"][:3,3]).any() or reg_dict["transformation"][3,3]!=1.0: continue
                    
                    # analyse 
                    gt_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "gt")
                    ae = roma.rotmat_geodesic_distance(torch.from_numpy(reg_dict['transformation'][:3,:3]), torch.from_numpy(gt_dict['transformation'][:3, :3])) * 180 /torch.pi
                    te = np.linalg.norm(gt_dict['transformation'][:3,3] - reg_dict["transformation"][:3,3])
                    loop_edges.append((source_id, target_id, ae.item(), te.item()))
                    
                    transformation = reg_dict['transformation']
                    information = reg_dict['information']
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation,
                                                                information,
                                                                uncertain=True))
                    new_submap_valid_loop = True
            
            if source_id == self.submap_id and not new_submap_valid_loop:
                break    
                
        return pose_graph, odometry_edges, loop_edges
    
    def loop_closure(self, estimated_c2ws, final=False):
        '''
        Compute loop closure correction
        
        returns: 
            None or the pose correction for each submap
        '''
        
        print("\nDetecting loop closures ...")
        # first see if current submap generates any new edge to the pose graph
        correction_list = []
        self.c2ws_est = estimated_c2ws.detach()        
        self.submap_paths = sorted(glob.glob(str(self.submap_path/"*.ckpt")), key=lambda x: int(x.split('/')[-1][:-5]))
        
        
        if self.submap_id<3 or len(self.detect_closure(self.submap_id)) == 0:
            print(f"\nNo loop closure detected at submap no.{self.submap_id}")
            return correction_list
        
        pose_graph, odometry_edges, loop_edges = self.construct_pose_graph(final)
        
        # save pgo edge analysis result
        
        if len(loop_edges)>0 and len(loop_edges) > self.n_loop_edges: 
            
            print("Optimizing PoseGraph ...")
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=self.max_correspondence_distance_fine,
                edge_prune_threshold=self.config['lc']['pgo_edge_prune_thres'],
                reference_node=0)
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
            
            self.pgo_count += 1
            self.n_loop_edges = len(loop_edges)
            
            for id in range(self.submap_id+1):
                submap_correction = {
                    'submap_id': id,
                    "correct_tsfm": pose_graph.nodes[id].pose}
                correction_list.append(submap_correction)
                
            self.analyse_pgo(odometry_edges, loop_edges, pose_graph)
        
        else:
            print("No valid loop edges or new loop edges. Skipping ...")
            
        return correction_list
    
    def analyse_pgo(self, odometry_edges, loop_edges, pose_graph):
        """analyse the results from pose graph optimization

        Args:
            odometry_edges (list): list of error in odometry edges
            loop_edges (list): list of error in loop edges
        """
        pgo_save_path = Path(self.config["data"]["output_path"])/"pgo"/str(self.pgo_count)
        
        print("Evaluating ATE before pgo ...")
        eval_ate(self.cam_dict, self.kf_ids, str(pgo_save_path/"before"), 100, final=True)
        
        corrected_cams = {}
        corrected_ids = []
        for i, node in enumerate(pose_graph.nodes):
            for kf_id in self.submap_lc_info[i]['kf_id']:
                cam = self.cam_dict[kf_id]
                updated_RT = (cam.get_T @ torch.from_numpy(node.pose).cuda().float().inverse())
                cam.update_RT(updated_RT[:3, :3], updated_RT[:3, 3])
                corrected_cams[cam.uid] = cam
                corrected_ids.append(cam.uid)
        print("Evaluating ATE after pgo ...")
        eval_ate(corrected_cams, self.kf_ids, str(pgo_save_path/"after_gs"), None, True)
        
        submap_cams = {}
        for i in range(len(self.submap_lc_info)):
            global_kf = self.cam_dict[self.submap_lc_info[i]['kf_id'][0]]
            corr_delta = global_kf.get_T.inverse() @ global_kf.get_T_gt
            for kf_id in self.submap_lc_info[i]['kf_id']:
                cam = self.cam_dict[kf_id]
                updated_RT = (cam.get_T @ corr_delta)
                cam.update_RT(updated_RT[:3, :3], updated_RT[:3, 3])
                submap_cams[cam.uid] = cam
        print("Evaluating ATE within each submap ...")
        eval_ate(corrected_cams, self.kf_ids, str(pgo_save_path/"submap"), None, True)
        
        # save edge errors
        odometry_re_errors = [edge[2] for edge in odometry_edges]
        loop_re_errors = [edge[2] for edge in loop_edges]
        odometry_te_errors = [edge[3]*100 for edge in odometry_edges]
        loop_te_errors = [edge[3]*100 for edge in loop_edges]

        # create plot for rotation errors  
        # Combine errors and create labels
        all_errors = odometry_re_errors + loop_re_errors
        all_edges = odometry_edges + loop_edges
        colors = ['blue'] * len(odometry_re_errors) + ['orange'] * len(loop_re_errors)
        labels = ['Odometry'] * len(odometry_re_errors) + ['Loop Closure'] * len(loop_re_errors)

        # Calculate the medians
        median_odometry_error = np.median(odometry_re_errors)
        median_loop_error = np.median(loop_re_errors)

        # Create bar plot
        plt.figure(figsize=(12, 6))

        # Plot each error as a separate bar
        for i in range(len(all_errors)):
            plt.bar(i, all_errors[i], color=colors[i], label=labels[i] if i == 0 or labels[i] != labels[i-1] else "")

        # Plot the median lines
        plt.axhline(y=median_odometry_error, color='blue', linestyle='--', label=f'Median Odometry Error: {median_odometry_error:.2f} degrees')
        plt.axhline(y=median_loop_error, color='orange', linestyle='--', label=f'Median Loop Error: {median_loop_error:.2f} degrees')

        # Add labels, title, and legend
        plt.xlabel('Edges')
        plt.ylabel('Error (degrees)')
        plt.title('Odometry and Loop Closure Edge Errors with Medians')
        plt.legend()

        # Set x-ticks to show edge labels
        plt.xticks(range(len(all_errors)), labels, rotation=90)

        plt.tight_layout()
        plot_filename = pgo_save_path/"submap_all_edge_ae.png"
        plt.savefig(plot_filename)
        
        # create plot for translational errors  
        # Combine errors and create labels
        all_errors = odometry_te_errors + loop_te_errors
        all_edges = odometry_edges + loop_edges
        colors = ['blue'] * len(odometry_re_errors) + ['orange'] * len(loop_re_errors)
        labels = ['Odometry'] * len(odometry_re_errors) + ['Loop Closure'] * len(loop_re_errors)

        # Calculate the medians
        median_odometry_error = np.median(odometry_te_errors)
        median_loop_error = np.median(loop_te_errors)

        # Create bar plot
        plt.figure(figsize=(12, 6))

        # Plot each error as a separate bar
        for i in range(len(all_errors)):
            plt.bar(i, all_errors[i], color=colors[i], label=labels[i] if i == 0 or labels[i] != labels[i-1] else "")

        # Plot the median lines
        plt.axhline(y=median_odometry_error, color='blue', linestyle='--', label=f'Median Odometry Error: {median_odometry_error:.2f} cm')
        plt.axhline(y=median_loop_error, color='orange', linestyle='--', label=f'Median Loop Error: {median_loop_error:.2f} cm')

        # Add labels, title, and legend
        plt.xlabel('Edges')
        plt.ylabel('Error (cm)')
        plt.title('Odometry and Loop Closure Edge Errors with Medians')
        plt.legend()

        # Set x-ticks to show edge labels
        plt.xticks(range(len(all_errors)), labels, rotation=90)

        plt.tight_layout()
        plot_filename = pgo_save_path/"submap_all_edge_te.png"
        plt.savefig(plot_filename)
        return
    
    def submap_to_segment(self, submap):
        segment = {
            "points": submap['gaussians'].get_xyz().detach().cpu(),
            "keyframe": submap['cameras'][0].get_T.detach().cpu(),
            "gt_camera": submap['cameras'][0].get_T_gt.detach().cpu(),
        }
        return segment

    def pairwise_registration(self, submap_source, submap_target, method="gs_reg"):

        segment_source = self.submap_to_segment(submap_source)
        segment_target = self.submap_to_segment(submap_target)
        max_correspondence_distance_coarse = 0.3
        max_correspondence_distance_fine = 0.03

        source_points = segment_source["points"]
        target_points = segment_target["points"]

        # source_colors = segment_source["points_color"]
        # target_colors = segment_target["points_color"]

        cloud_source = o3d.geometry.PointCloud()
        cloud_source.points = o3d.utility.Vector3dVector(np.array(source_points))
        # cloud_source.colors = o3d.utility.Vector3dVector(np.array(source_colors))
        cloud_source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        keyframe_source = segment_source["keyframe"]
        camera_location_source = keyframe_source[:3, 3].cpu().numpy()
        cloud_source.orient_normals_towards_camera_location(
            camera_location=camera_location_source)

        cloud_target = o3d.geometry.PointCloud()
        cloud_target.points = o3d.utility.Vector3dVector(np.array(target_points))
        # cloud_target.colors = o3d.utility.Vector3dVector(np.array(target_colors))
        cloud_target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        keyframe_target = segment_target["keyframe"]
        camera_location_target = keyframe_target[:3, 3].cpu().numpy()
        cloud_target.orient_normals_towards_camera_location(
            camera_location=camera_location_target)
        
        output = dict()
        if method == "gt":
            gt_source = segment_source["gt_camera"]
            gt_target = segment_target["gt_camera"]
            delta_src = gt_source.inverse() @ keyframe_source
            delta_tgt = gt_target.inverse() @ keyframe_target
            delta = delta_tgt.inverse() @ delta_src
            output["transformation"] = np.array(delta)
        elif method == "icp":
            icp_coarse = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_coarse, np.identity(
                    4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            delta = icp_fine.transformation
            output["transformation"] = np.array(delta)
        elif method == "robust_icp":
            voxel_size = 0.04
            sigma = 0.01

            source_down, source_fpfh = preprocess_point_cloud(
                cloud_source, voxel_size, camera_location_source)
            target_down, target_fpfh = preprocess_point_cloud(
                cloud_target, voxel_size, camera_location_target)
            
            tic = time.perf_counter()

            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size)

            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
            
            toc = time.perf_counter()
            delta = icp_fine.transformation

            # compute success to gt delta
            gt_source = segment_source["gt_camera"]
            gt_target = segment_target["gt_camera"]
            rel_gt = gt_source@gt_target.inverse()
            delta_gt = rel_gt@keyframe_target@keyframe_source.inverse()
            output["transformation_gt_mag"] = torch.abs(delta_gt).mean().item()
            output["transformation_mag"] = torch.abs(
                torch.tensor(delta)).mean().item()
            output["transformation"] = np.array(delta)
            output["fitness"] = icp_fine.fitness
            output["inlier_rmse"] = icp_fine.inlier_rmse
            output["registration_time"] = toc-tic

        elif method == "identity":
            delta = np.identity(4)
            output["transformation"] = delta

        elif method == "gs_reg":
            res = gs_reg(submap_source, submap_target, self.config['lc']['registration'])
            delta = res['pred_tsfm'].cpu().numpy()
            output["transformation"] = delta
            output["successful"] = res["successful"]
                
        else:
            raise NotImplementedError("Unknown registration method!")

        output["information"] = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            cloud_source,
            cloud_target,
            max_correspondence_distance_fine,
            np.array(delta)
        )

        output["n_points"] = min(len(cloud_source.points),
                                len(cloud_target.points))
        output['pc_src'] = cloud_source
        output['pc_tgt'] = cloud_target
        return output
    