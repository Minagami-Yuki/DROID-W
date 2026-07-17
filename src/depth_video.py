import numpy as np
import torch
import lietorch
import droid_backends
import src.geom.ba
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F

from src.modules.droid_net import cvx_upsample
import src.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from src.utils.dyn_uncertainty import mapping_utils as map_utils
from src.utils.plot_utils import create_gif_from_directory
from src.utils.omega_prior import OmegaPriorCache
from src.utils.edge_dtf_prior import EdgeDTFPrior
from src.utils.v46_reliability import apply_v46_reliability
import matplotlib.pyplot as plt
import os
import csv
from src.utils.sys_timer import timer
import PIL
import PIL.Image as Image
from sklearn.decomposition import PCA
from tqdm import tqdm

class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.printer = printer
        self.metric_depth_reg = cfg['tracking']['backend']['metric_depth_reg']
        if not self.metric_depth_reg:
            self.printer.print(f"Metric depth for regularization is not activated.",FontColor.INFO)
            self.printer.print(f"This should not happen for WildGS-SLAM unless you are doing ablation study",FontColor.INFO)
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.focal_calibration_cfg = cfg['tracking'].get('focal_calibration', {}) or {}
        self.focal_calibration_enabled = bool(self.focal_calibration_cfg.get('enable', False))
        self._focal_prior = None
        self._focal_ratio = None
        self._intrinsics_prior = None
        self._focal_ba_calls = 0
        self._focal_stable_ba_streak = 0
        self._focal_confidence_shape_scores = []
        self._focal_calibration_rows = []
        self.omega_prior = OmegaPriorCache(cfg, self.device)
        self.edge_dtf_prior = EdgeDTFPrior(cfg, self.device)
        self.down_scale = 8
        self.slice_h = slice(self.down_scale // 2 - 1, ht//self.down_scale*self.down_scale+1, self.down_scale)
        self.slice_w = slice(self.down_scale // 2 - 1, wd//self.down_scale*self.down_scale+1, self.down_scale)
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        # To save gpu ram, we put images to cpu as it is never used
        self.images = torch.zeros(buffer, 3, ht, wd, device='cpu', dtype=torch.float32)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()  # world to camera
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        
        self.omega_uncertainties = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.omega_uncertainty_valid = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        self.edge_dtf_edges = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.edge_dtf_maps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.edge_dtf_valid = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        token_cfg = self._omega_token_buffer_cfg()
        self.omega_token_count = int(token_cfg.get("max_tokens", 17))
        self.omega_token_dim = int(token_cfg.get("token_dim", 2048))
        self.omega_tokens = torch.zeros(buffer, self.omega_token_count, self.omega_token_dim, device=self.device, dtype=torch.float).share_memory_()
        self.omega_token_valid = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        patch_token_cfg = self._omega_patch_token_buffer_cfg()
        self.omega_patch_token_dim = int(patch_token_cfg.get("token_dim", 8))
        self.omega_patch_maps = torch.zeros(buffer, self.omega_patch_token_dim, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.omega_patch_valid = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        self.omega_dense_patch_risk = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.omega_dense_patch_valid = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()
        self._dense_patch_map_update = 0
        self._dense_patch_map_visual_count = 0
        self._patch_token_stats_call = 0
        self._patch_token_stats_header_written = False
        self._low_parallax_support_ema = None
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.debug = cfg['debug']

        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.enable_bidirectional_uncer = cfg['tracking']["uncertainty_params"]['enable_bidirectional_uncer']
        if self.uncertainty_aware:
            n_features = self.cfg["tracking"]["uncertainty_params"]['feature_dim']

            # This check is to ensure the size of self.dino_feats
            if self.cfg["mono_prior"]["feature_extractor"] not in ["dinov2_reg_small_fine", "dinov2_small_fine","dinov2_vits14", "dinov2_vits14_reg", "dinov3_vits16", "dinov3_vits16plus"]:
                raise ValueError("You are using a new feature extractor, make sure the downsample factor is 14")
            if self.cfg["mono_prior"]["feature_extractor"] in ["dinov3_vits16", "dinov3_vits16plus"]:
                self.feature_downsample_factor = 16
            else:
                self.feature_downsample_factor = 14
            
            # The followings are in cpu to save memory
            self.dino_feats = torch.zeros(buffer, ht//self.feature_downsample_factor, wd//self.feature_downsample_factor, n_features, device='cpu', dtype=torch.float).share_memory_()
            self.dino_feats_resize = torch.zeros(buffer, n_features, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            self.uncertainties = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            # use kaiming normal initialization to initialize the affine weights
            # [1, n_features + 1] ：first n_features are weights, last one is bias
            self.affine_weights = torch.empty((1, n_features + 1), dtype=torch.float, device=self.device)
            # use kaiming normal initialization to initialize the affine weights
            torch.nn.init.kaiming_normal_(self.affine_weights[:, :-1], mode='fan_in', nonlinearity='linear')
            # set bias to 0
            self.affine_weights[:, -1].zero_()
            self.affine_weights = self.affine_weights.squeeze(0).share_memory_()
            self.enable_affine_transform = cfg['tracking']['uncertainty_params']['enable_affine_transform']
            self.temp_y_cdot = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        else:
            self.dino_feats = None
            self.dino_feats_resize = None

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1].cpu()
        if self.edge_dtf_prior.enabled:
            images = item[1]
            if images.ndim == 3:
                edge_map, dtf_map = self.edge_dtf_prior.compute_maps(
                    images,
                    (self.ht//self.down_scale, self.wd//self.down_scale),
                )
                self.edge_dtf_edges[index] = edge_map
                self.edge_dtf_maps[index] = dtf_map
                self.edge_dtf_valid[index] = True
            elif images.ndim == 4:
                edge_maps = []
                dtf_maps = []
                for image in images:
                    edge_map, dtf_map = self.edge_dtf_prior.compute_maps(
                        image,
                        (self.ht//self.down_scale, self.wd//self.down_scale),
                    )
                    edge_maps.append(edge_map)
                    dtf_maps.append(dtf_map)
                self.edge_dtf_edges[index] = torch.stack(edge_maps, dim=0)
                self.edge_dtf_maps[index] = torch.stack(dtf_maps, dim=0)
                self.edge_dtf_valid[index] = True
            else:
                raise ValueError(f"Expected image tensor [3,H,W] or [B,3,H,W], got {tuple(images.shape)}")
        else:
            self.edge_dtf_edges[index] = 0.0
            self.edge_dtf_maps[index] = 1.0
            self.edge_dtf_valid[index] = False

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            mono_depth = item[4][self.slice_h,self.slice_w]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            self.mono_disps_up[index] = torch.where(item[4]>0, 1.0/item[4], 0)
            # self.disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)

        if item[5] is not None:
            intrinsic = item[5]
            if self.focal_calibration_enabled:
                # Trajectory filling writes a batch of non-keyframes at once.
                # Those interpolated poses must not contribute to the online
                # Omega confidence statistic used for calibration routing.
                frame_index = item[0]
                if not torch.is_tensor(frame_index) or torch.numel(frame_index) == 1:
                    confidence_score = self.omega_prior.raw_confidence_shape_score(int(frame_index))
                    if confidence_score is not None:
                        self._focal_confidence_shape_scores.append(confidence_score)
                if self._focal_prior is None:
                    self._focal_prior = intrinsic[:2].detach().clone()
                    self._focal_ratio = (intrinsic[1] / intrinsic[0]).detach().clone()
                    self._intrinsics_prior = intrinsic.detach().clone().to(self.device)
                else:
                    # The sequence shares one camera. New keyframes must use
                    # the current calibrated K, not the original dataset input.
                    intrinsic = intrinsic.clone()
                    if self.focal_calibration_cfg.get("intrinsics_mode", "focal") == "full_pinhole":
                        intrinsic[...] = self.intrinsics[0]
                    else:
                        intrinsic[..., :2] = self.intrinsics[0, :2]
            self.intrinsics[index] = intrinsic

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.dino_feats[index] = item[9].cpu()

            if len(item[9].shape) == 3:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(2,0,1).unsqueeze(0),
                                                            self.disps_up.shape[-2:], 
                                                            mode='bilinear').squeeze()[:,self.slice_h,self.slice_w]
                # y_cdot = w * x + b
                # y = log(1 + exp(y_cdot))
                if self.enable_affine_transform:
                    y_cdot = self.dino_feats_resize[index].permute(1,2,0) @ self.affine_weights[:-1] + self.affine_weights[-1]
                    self.temp_y_cdot[index] = y_cdot
                    self.uncertainties[index] = torch.log(1.1 + torch.exp(y_cdot))
            else:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(0,3,1,2),
                                                            self.disps_up.shape[-2:], 
                                                            mode='bilinear')[:,:,self.slice_h,self.slice_w]
                # y_cdot = w * x + b
                # y = log(1 + exp(y_cdot))
                if self.enable_affine_transform:
                    y_cdot = self.dino_feats_resize[index].permute(0, 2, 3, 1) @ self.affine_weights[:-1] + self.affine_weights[-1]
                    self.temp_y_cdot[index] = y_cdot
                    self.uncertainties[index] = torch.log(1.1 + torch.exp(y_cdot))
                    
            # constrain the uncertainty of similar dino feats to be similar (TODO:unused in current implementation)
            # Compute pairwise similarity between all pixels within each frame
            # normalize the dino feats
            dino_feats_normalized = F.normalize(self.dino_feats_resize[index], p=2, dim=-3)
            dino_feats_tmp = dino_feats_normalized  # [1, C, H, W]
            C, H, W = dino_feats_tmp.shape[-3:]

        if len(item) > 10 and item[10] is not None:
            omega_uncertainty = item[10].to(self.device, dtype=torch.float32)
            if tuple(omega_uncertainty.shape[-2:]) != (self.ht, self.wd):
                omega_uncertainty = F.interpolate(
                    omega_uncertainty[None, None],
                    size=(self.ht, self.wd),
                    mode='bilinear',
                    align_corners=False,
                )[0, 0]
            omega_uncertainty_small = omega_uncertainty[self.slice_h, self.slice_w]
            self.omega_uncertainties[index] = omega_uncertainty_small
            self.omega_uncertainty_valid[index] = True
            write_to_droid_uncertainty = (
                self.omega_prior.uncertainty_cfg.get("write_to_droid_uncertainty", False)
                or self.omega_prior.uncertainty_cfg.get("apply_to") == "droid_uncertainty"
            )
            if self.uncertainty_aware and write_to_droid_uncertainty:
                self.uncertainties[index] = omega_uncertainty_small
        else:
            self.omega_uncertainties[index] = 1.0
            self.omega_uncertainty_valid[index] = False

        if len(item) > 11 and item[11] is not None:
            omega_tokens = item[11].to(self.device, dtype=torch.float32)
            if omega_tokens.ndim == 1:
                omega_tokens = omega_tokens[None]
            if omega_tokens.ndim > 2:
                omega_tokens = omega_tokens.reshape(-1, omega_tokens.shape[-1])
            if omega_tokens.ndim != 2:
                raise ValueError(f"Expected Omega tokens [T,C], got {tuple(omega_tokens.shape)}")

            self.omega_tokens[index].zero_()
            token_count = min(self.omega_token_count, omega_tokens.shape[0])
            token_dim = min(self.omega_token_dim, omega_tokens.shape[1])
            self.omega_tokens[index, :token_count, :token_dim] = omega_tokens[:token_count, :token_dim]
            self.omega_token_valid[index] = True
        else:
            self.omega_tokens[index].zero_()
            self.omega_token_valid[index] = False

        if len(item) > 12 and item[12] is not None:
            omega_patch_map = item[12].to(self.device, dtype=torch.float32)
            if omega_patch_map.ndim != 3:
                raise ValueError(f"Expected Omega patch token map [C,H,W] or [H,W,C], got {tuple(omega_patch_map.shape)}")
            if omega_patch_map.shape[0] != self.omega_patch_token_dim and omega_patch_map.shape[-1] == self.omega_patch_token_dim:
                omega_patch_map = omega_patch_map.permute(2, 0, 1).contiguous()
            if omega_patch_map.shape[0] != self.omega_patch_token_dim:
                raise ValueError(
                    f"Expected Omega patch token dim {self.omega_patch_token_dim}, got {tuple(omega_patch_map.shape)}"
                )
            if tuple(omega_patch_map.shape[-2:]) != (self.ht//self.down_scale, self.wd//self.down_scale):
                omega_patch_map = F.interpolate(
                    omega_patch_map[None],
                    size=(self.ht//self.down_scale, self.wd//self.down_scale),
                    mode='bilinear',
                    align_corners=False,
                )[0]
            self.omega_patch_maps[index] = F.normalize(omega_patch_map, p=2, dim=0, eps=1e-6)
            self.omega_patch_valid[index] = True
            self.omega_dense_patch_risk[index].zero_()
            self.omega_dense_patch_valid[index] = False
        else:
            self.omega_patch_maps[index].zero_()
            self.omega_patch_valid[index] = False
            self.omega_dense_patch_risk[index].zero_()
            self.omega_dense_patch_valid[index] = False

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def init_w_mono_disp(self, start_idx, end_idx):
        with self.get_lock():
            self.disps[start_idx:end_idx] = self.mono_disps[start_idx:end_idx]
            self.disps_up[start_idx:end_idx] = self.mono_disps_up[start_idx:end_idx]

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def upsample_weight(self, weight):
        """ upsample weight to the original image size """
        weight_up = F.interpolate(weight.unsqueeze(0), 
                                size=(self.ht, self.wd), mode='bilinear').squeeze()
        return weight_up

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def project_images_with_mask(self, images, pixel_positions, masks=None):
        """ 
            Project images/depths from the input pixel positions using bilinear interpolation.
            This function will automatically return the mask where the given pixel positions are out of the images
        Args:
            images (torch.Tensor): A tensor of shape [B, C, H, W] representing the images/depths.
            pixel_positions (torch.Tensor): A tensor of shape [B, H, W, 2] containing float 
                                            pixel positions for interpolation. Note that [:,:,:,0]
                                            is width and [:,:,:,1] is height.
            masks (torch.Tensor, optional): A boolean tensor of shape [B, H, W]. If provided, 
                                            specifies valid pixels. Default is None, which 
                                            results in all pixels being valid at the begining.
        
        Returns:
            torch.Tensor: A tensor of shape [B, C, H, W] containing the projected images/depths, 
                        where invalid pixels are set to 0.
            torch.Tensor: The combined mask that filters out invalid positions and applies
                      the original mask.
        """
        B, C, H, W = images.shape
        device = images.device

        # If masks are not provided, create a mask of all ones (True) with the same shape as the images
        if masks is None:
            masks = torch.ones(B, H, W, dtype=torch.bool, device=device)
        
        # Normalize pixel positions to range [-1, 1]
        grid = pixel_positions.clone()
        grid[..., 0] = 2.0 * (grid[..., 0] / (W - 1)) - 1.0
        grid[..., 1] = 2.0 * (grid[..., 1] / (H - 1)) - 1.0

        projected_image = F.grid_sample(images, grid, mode='bilinear', align_corners=True)

        # Mask out invalid positions where x or y are out of bounds and combine it with the initial mask
        valid_mask = (pixel_positions[..., 0] >= 0) & (pixel_positions[..., 0] < W - 1) & \
                    (pixel_positions[..., 1] >= 0) & (pixel_positions[..., 1] < H - 1)
        valid_mask &= masks

        # Apply the combined mask: set to 0 where combined mask is False
        projected_image = projected_image.permute(0, 2, 3, 1)  # conver to [B, H, W, C]
        projected_image = projected_image * valid_mask.unsqueeze(-1)
        
        return projected_image.permute(0, 3, 1, 2), valid_mask  # Return to [B, C, H, W]

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, gamma=0.02, tao=0.1, lr=1e-2, weight_decay=2e-4,
           motion_only=False, enable_update_uncer=False, enable_udba=False, visualization_stage=False):      # ii, jj represent all img pairs
        
        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = self.apply_omega_edge_weight(weight, ii, jj)
            if self.omega_prior.uncertainty_enabled and self.omega_prior.uncertainty_cfg.get("freeze_droid_uncertainty_update", False):
                enable_update_uncer = False

            solver = self.focal_calibration_cfg.get("solver", "alternating")
            bootstrap_cfg = self.focal_calibration_cfg.get("schur_bootstrap", {}) or {}
            stability_cfg = bootstrap_cfg.get("trajectory_stability", {}) or {}
            record_base_pose_update = (
                solver in {"droidcalib_schur", "flow3r_schur"}
                and self.focal_calibration_enabled
                and bool(stability_cfg.get("enable", False))
                and not motion_only
            )
            poses_before_base_ba = self.poses[:self.counter.value].clone() if record_base_pose_update else None

            # if there is NaN of inf value for self.affine_weights, assert
            if self.uncertainty_aware:
                assert not torch.isnan(self.affine_weights).any(), "self.affine_weights has NaN value"
                assert not torch.isinf(self.affine_weights).any(), "self.affine_weights has inf value"
            if not self.metric_depth_reg:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,           # the shape of poses is determined by buffer size
                    target, weight, self.uncertainties, 
                    self.temp_y_cdot,
                    self.dino_feats_resize,
                    self.affine_weights,
                    eta, ii, jj, t0, t1, iters, lm, ep, 
                    self.cfg['tracking']['uncertainty_params']['gamma_data'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_prior'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_depth'],
                    lr, weight_decay,
                    motion_only, False, enable_update_uncer,
                    enable_udba, self.enable_affine_transform,
                    self.enable_bidirectional_uncer,
                    self.debug)         # poses: [buffer, 7], disps: [buffer, h, w], 
            else:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.mono_disps,
                    target, weight, self.uncertainties, 
                    self.temp_y_cdot,
                    self.dino_feats_resize,
                    self.affine_weights,
                    eta, ii, jj, t0, t1, iters, lm, ep, 
                    self.cfg['tracking']['uncertainty_params']['gamma_data'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_prior'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_depth'],
                    lr, weight_decay,
                    motion_only, False, enable_update_uncer,
                    enable_udba, self.enable_affine_transform,
                    self.enable_bidirectional_uncer,
                    self.debug)          # t0, t1: window of keyframes for BA
            
            self.disps.clamp_(min=1e-5)
            base_pose_update = self._focal_base_pose_update(poses_before_base_ba, t0, t1)
            if solver in {"droidcalib_schur", "flow3r_schur"}:
                if self.focal_calibration_cfg.get("intrinsics_mode", "focal") == "full_pinhole":
                    self._maybe_flow3r_full_pinhole_schur(
                        target, weight, eta, ii, jj, t0, t1, motion_only, base_pose_update
                    )
                else:
                    self._maybe_flow3r_focal_schur(
                        target, weight, eta, ii, jj, t0, t1, motion_only, base_pose_update
                    )
            else:
                self._maybe_calibrate_focal(target, weight, ii, jj, motion_only)

    def _maybe_flow3r_focal_schur(
        self, target, weight, eta, ii, jj, t0, t1, motion_only, base_pose_update=None
    ):
        from src.utils.flow3r_joint_ba import load_flow3r_joint_backend

        cfg = self.focal_calibration_cfg
        if not self.focal_calibration_enabled or motion_only or self._focal_prior is None:
            return
        self._focal_ba_calls += 1
        bootstrap_cfg = cfg.get("schur_bootstrap", {}) or {}
        bootstrap_active = (
            bool(bootstrap_cfg.get("enable", False))
            and int(bootstrap_cfg.get("start_keyframes", 30)) <= self.counter.value
            and self.counter.value <= int(bootstrap_cfg.get("end_keyframes", 80))
        )
        recovery_cfg = bootstrap_cfg.get("initial_k_recovery", {}) or {}
        initial_focal_px = float(self._focal_prior[0]) * float(self.down_scale)
        focal_recovery_active = (
            bootstrap_active
            and bool(recovery_cfg.get("enable", False))
            and initial_focal_px >= float(recovery_cfg.get("focal_min_px", float("inf")))
            and initial_focal_px <= float(recovery_cfg.get("focal_max_px", float("inf")))
        )
        confidence_recovery_cfg = bootstrap_cfg.get("omega_confidence_recovery", {}) or {}
        confidence_min_samples = int(confidence_recovery_cfg.get("min_samples", 30))
        confidence_score = (
            float(np.median(self._focal_confidence_shape_scores))
            if self._focal_confidence_shape_scores
            else float("nan")
        )
        confidence_recovery_active = (
            bootstrap_active
            and bool(confidence_recovery_cfg.get("enable", False))
            and len(self._focal_confidence_shape_scores) >= confidence_min_samples
            and np.isfinite(confidence_score)
            and confidence_score >= float(confidence_recovery_cfg.get("min_mean_median_ratio", float("inf")))
        )
        recovery_active = focal_recovery_active or confidence_recovery_active
        stability_cfg = bootstrap_cfg.get("trajectory_stability", {}) or {}
        stability_enabled = (
            bootstrap_active
            and bool(stability_cfg.get("enable", False))
            and not recovery_active
        )
        stability_ok = True
        if stability_enabled:
            max_translation = float(stability_cfg.get("max_translation_update", 0.03))
            max_rotation = float(stability_cfg.get("max_rotation_update_rad", 0.025))
            stability_ok = (
                base_pose_update is not None
                and base_pose_update["finite"]
                and base_pose_update["translation_max"] <= max_translation
                and base_pose_update["rotation_max"] <= max_rotation
            )
            if stability_ok:
                self._focal_stable_ba_streak += 1
            else:
                self._focal_stable_ba_streak = 0
        every_n_ba = int(
            bootstrap_cfg.get("every_n_ba", 1) if bootstrap_active else cfg.get("every_n_ba", 8)
        )
        if self._focal_ba_calls % max(1, every_n_ba) != 0:
            return
        if self.counter.value < int(cfg.get("warmup_keyframes", 20)) or int((ii != jj).sum()) < int(cfg.get("min_edges", 12)):
            return

        intrinsics_before = self.intrinsics[0].clone()
        diagnostic_enabled = bool(cfg.get("schur_diagnostics", False))
        min_hessian = (
            bootstrap_cfg.get("min_hessian", cfg.get("schur_min_hessian"))
            if bootstrap_active
            else cfg.get("schur_min_hessian")
        )
        loss_before, support, gradient, hessian = self._focal_schur_observability(
            target, weight, ii, jj, intrinsics_before, diagnostic_enabled or min_hessian is not None
        )

        def write_row(reason, accepted, fx_proposed, fy_proposed, step_proposed, loss_after):
            if not diagnostic_enabled:
                return
            final_step = torch.log((self.intrinsics[0, 0] / intrinsics_before[0]).clamp_min(1e-6))
            final_relative = torch.log((self.intrinsics[0, 0] / self._focal_prior[0]).clamp_min(1e-6))
            self._focal_calibration_rows.append({
                "solver": "droidcalib_schur",
                "phase": "recovery" if recovery_active else ("bootstrap" if bootstrap_active else "tracking"),
                "ba_call": self._focal_ba_calls,
                "keyframes": self.counter.value,
                "fx_before": float(intrinsics_before[0]),
                "fy_before": float(intrinsics_before[1]),
                "fx_proposed": float(fx_proposed),
                "fy_proposed": float(fy_proposed),
                "fx_after": float(self.intrinsics[0, 0]),
                "fy_after": float(self.intrinsics[0, 1]),
                "step_log_proposed": float(step_proposed),
                "step_log_applied": float(final_step),
                "relative_log_after": float(final_relative),
                "loss_before": float(loss_before),
                "loss_after": float(loss_after),
                "support": float(support),
                "gradient": float(gradient),
                "hessian": float(hessian),
                "hessian_threshold": "" if min_hessian is None else float(min_hessian),
                "initial_focal_px": initial_focal_px,
                "initial_k_recovery": int(recovery_active),
                "omega_confidence_shape_score": confidence_score,
                "omega_confidence_recovery": int(confidence_recovery_active),
                "base_pose_translation_max": "" if base_pose_update is None else base_pose_update["translation_max"],
                "base_pose_rotation_max": "" if base_pose_update is None else base_pose_update["rotation_max"],
                "stable_ba_streak": self._focal_stable_ba_streak if stability_enabled else "",
                "accepted": int(accepted),
                "reason": reason,
            })
            self._write_focal_calibration_rows()

        if stability_enabled and (
            not stability_ok
            or self._focal_stable_ba_streak < int(stability_cfg.get("min_consecutive_ba", 3))
        ):
            write_row("unstable_trajectory", False, intrinsics_before[0], intrinsics_before[1], 0.0, loss_before)
            return

        if min_hessian is not None and (
            not torch.isfinite(hessian) or float(hessian) < float(min_hessian)
        ):
            write_row("low_observability", False, intrinsics_before[0], intrinsics_before[1], 0.0, loss_before)
            return

        backend = load_flow3r_joint_backend()
        poses_before = self.poses.clone()
        disps_before = self.disps.clone()
        backend.flow3r_ba(
            self.poses, self.disps, self.intrinsics[0], self.zeros, target, weight, eta,
            ii, jj, t0, t1, 1, 2, 1e-4, 0.1, False, True,
            float(self._focal_prior[0]),
            float(cfg.get("prior_weight", 5.0)) / float(self._focal_prior[0].square().clamp_min(1e-6)),
            1.0, self._intrinsics_prior, torch.zeros_like(self._intrinsics_prior),
        )
        max_deviation = float(
            bootstrap_cfg.get("max_log_deviation", cfg.get("max_log_deviation", 0.15))
            if bootstrap_active
            else cfg.get("max_log_deviation", 0.15)
        )
        relative_log = torch.log((self.intrinsics[0, 0] / self._focal_prior[0]).clamp_min(1e-6))
        step_log = torch.log((self.intrinsics[0, 0] / intrinsics_before[0]).clamp_min(1e-6))
        fx_proposed, fy_proposed, step_proposed = self.intrinsics[0, 0].clone(), self.intrinsics[0, 1].clone(), step_log.clone()
        max_step = float(
            bootstrap_cfg.get("max_log_step", cfg.get("max_log_step", 0.002))
            if bootstrap_active
            else cfg.get("max_log_step", 0.002)
        )
        if not torch.isfinite(relative_log) or not torch.isfinite(step_log) or relative_log.abs() > max_deviation:
            self.poses.copy_(poses_before)
            self.disps.copy_(disps_before)
            self.intrinsics[0].copy_(intrinsics_before)
            write_row("bounds", False, fx_proposed, fy_proposed, step_proposed, loss_before)
            return
        if step_log.abs() > max_step:
            scale = float((max_step / step_log.abs()).clamp(max=1.0))
            self.poses.copy_(poses_before)
            self.disps.copy_(disps_before)
            self.intrinsics[0].copy_(intrinsics_before)
            backend.flow3r_ba(
                self.poses, self.disps, self.intrinsics[0], self.zeros, target, weight, eta,
                ii, jj, t0, t1, 1, 2, 1e-4, 0.1, False, True,
                float(self._focal_prior[0]),
                float(cfg.get("prior_weight", 5.0)) / float(self._focal_prior[0].square().clamp_min(1e-6)),
                scale, self._intrinsics_prior, torch.zeros_like(self._intrinsics_prior),
            )
        with torch.no_grad():
            loss_after, _ = self._focal_data_loss(target, weight, ii, jj, self.intrinsics[0])
        if not torch.isfinite(loss_after) or loss_after > loss_before:
            self.poses.copy_(poses_before)
            self.disps.copy_(disps_before)
            self.intrinsics[0].copy_(intrinsics_before)
            write_row("loss_increase", False, fx_proposed, fy_proposed, step_proposed, loss_after)
            return
        self.intrinsics[:self.counter.value, :2] = self.intrinsics[0, :2]
        write_row("accepted", True, fx_proposed, fy_proposed, step_proposed, loss_after)
        if stability_enabled and bool(stability_cfg.get("reset_after_accept", True)):
            self._focal_stable_ba_streak = 0

    def _maybe_flow3r_full_pinhole_schur(
        self, target, weight, eta, ii, jj, t0, t1, motion_only, base_pose_update=None
    ):
        """Constrained four-parameter pinhole Schur update.

        The CUDA backend already supports model_id=0 (fx, fy, cx, cy).  This
        wrapper keeps that larger update inside the existing acceptance path by
        applying per-component trust regions and a prior-aware loss check.
        """
        from src.utils.flow3r_joint_ba import load_flow3r_joint_backend

        cfg = self.focal_calibration_cfg
        if (
            not self.focal_calibration_enabled
            or motion_only
            or self._intrinsics_prior is None
        ):
            return
        # Intrinsics are scene-global.  In dynamic sequences, allowing this
        # block to keep adapting after the early well-observed segment can
        # couple late moving-object residuals back into the shared camera.
        # The optional freeze keeps ordinary pose/depth BA active.
        freeze_after = cfg.get("freeze_after_keyframes")
        if freeze_after is not None and self.counter.value > int(freeze_after):
            return
        self._focal_ba_calls += 1
        bootstrap_cfg = cfg.get("schur_bootstrap", {}) or {}
        bootstrap_active = (
            bool(bootstrap_cfg.get("enable", False))
            and int(bootstrap_cfg.get("start_keyframes", 30)) <= self.counter.value
            and self.counter.value <= int(bootstrap_cfg.get("end_keyframes", 80))
        )
        confidence_cfg = bootstrap_cfg.get("omega_confidence_recovery", {}) or {}
        confidence_score = (
            float(np.median(self._focal_confidence_shape_scores))
            if self._focal_confidence_shape_scores
            else float("nan")
        )
        confidence_active = (
            bootstrap_active
            and bool(confidence_cfg.get("enable", False))
            and len(self._focal_confidence_shape_scores) >= int(confidence_cfg.get("min_samples", 30))
            and np.isfinite(confidence_score)
            and confidence_score >= float(confidence_cfg.get("min_mean_median_ratio", float("inf")))
        )
        stability_cfg = bootstrap_cfg.get("trajectory_stability", {}) or {}
        stability_enabled = (
            bootstrap_active
            and bool(stability_cfg.get("enable", False))
            and not confidence_active
        )
        stability_ok = True
        if stability_enabled:
            stability_ok = (
                base_pose_update is not None
                and base_pose_update["finite"]
                and base_pose_update["translation_max"] <= float(stability_cfg.get("max_translation_update", 0.03))
                and base_pose_update["rotation_max"] <= float(stability_cfg.get("max_rotation_update_rad", 0.025))
            )
            self._focal_stable_ba_streak = self._focal_stable_ba_streak + 1 if stability_ok else 0

        every_n_ba = int(bootstrap_cfg.get("every_n_ba", 1) if bootstrap_active else cfg.get("every_n_ba", 8))
        if self._focal_ba_calls % max(1, every_n_ba) != 0:
            return
        if self.counter.value < int(cfg.get("warmup_keyframes", 20)) or int((ii != jj).sum()) < int(cfg.get("min_edges", 12)):
            return

        before = self.intrinsics[0].clone()
        diagnostic_enabled = bool(cfg.get("schur_diagnostics", False))
        calibration_weight, calibration_observation = self._calibration_observation_weights(
            weight, ii, jj
        )
        loss_before, support, hessian_diag = self._full_intrinsics_observability(
            target, calibration_weight, ii, jj, before
        )
        pp_cfg = cfg.get("principal_point", {}) or {}
        # Model-id 0 uses independent fx/fy/cx/cy coordinates, so its
        # curvature scale is not comparable to the focal-only scalar model.
        focal_min_hessian = float(cfg.get("full_focal_min_hessian", 1.0))
        principal_min_hessian = float(pp_cfg.get("min_hessian", 0.0))

        def write_row(reason, accepted, proposed, loss_after):
            if not diagnostic_enabled:
                return
            self._focal_calibration_rows.append({
                "solver": "droidcalib_schur_full_pinhole",
                "phase": "confidence_recovery" if confidence_active else ("bootstrap" if bootstrap_active else "tracking"),
                "ba_call": self._focal_ba_calls,
                "keyframes": self.counter.value,
                "fx_before": float(before[0]), "fy_before": float(before[1]),
                "cx_before": float(before[2]), "cy_before": float(before[3]),
                "fx_proposed": float(proposed[0]), "fy_proposed": float(proposed[1]),
                "cx_proposed": float(proposed[2]), "cy_proposed": float(proposed[3]),
                "fx_after": float(self.intrinsics[0, 0]), "fy_after": float(self.intrinsics[0, 1]),
                "cx_after": float(self.intrinsics[0, 2]), "cy_after": float(self.intrinsics[0, 3]),
                "loss_before": float(loss_before), "loss_after": float(loss_after),
                "support": float(support),
                "calibration_observation_coverage": calibration_observation["coverage"],
                "calibration_observation_retained_edges": calibration_observation["retained_edges"],
                "calibration_observation_total_edges": calibration_observation["total_edges"],
                "hessian_diag": ";".join(f"{float(value):.6g}" for value in hessian_diag),
                "omega_confidence_shape_score": confidence_score,
                "omega_confidence_recovery": int(confidence_active),
                "base_pose_translation_max": "" if base_pose_update is None else base_pose_update["translation_max"],
                "base_pose_rotation_max": "" if base_pose_update is None else base_pose_update["rotation_max"],
                "stable_ba_streak": self._focal_stable_ba_streak if stability_enabled else "",
                "accepted": int(accepted), "reason": reason,
            })
            self._write_focal_calibration_rows()

        if stability_enabled and (
            not stability_ok or self._focal_stable_ba_streak < int(stability_cfg.get("min_consecutive_ba", 3))
        ):
            write_row("unstable_trajectory", False, before, loss_before)
            return
        if (
            not torch.isfinite(hessian_diag).all()
            or float(hessian_diag[:2].abs().min()) < focal_min_hessian
            or float(hessian_diag[2:].abs().min()) < principal_min_hessian
        ):
            write_row("low_observability", False, before, loss_before)
            return

        backend = load_flow3r_joint_backend()
        poses_before, disps_before = self.poses.clone(), self.disps.clone()
        observation_cfg = cfg.get("calibration_observation", {}) or {}
        # The hard-mask ablation needs state isolation.  Soft calibration
        # weighting deliberately preserves the joint pose-depth-K coupling.
        use_observation_subset = (
            bool(observation_cfg.get("enable", False))
            and observation_cfg.get("mode", "hard") == "hard"
        )
        # The filtered calibration solve proposes K on copies.  This prevents
        # its restricted factor set from changing the ordinary BA pose/depth
        # state that the experiment is meant to hold fixed.
        poses_work = self.poses.clone() if use_observation_subset else self.poses
        disps_work = self.disps.clone() if use_observation_subset else self.disps
        intrinsics_work = before.clone() if use_observation_subset else self.intrinsics[0]
        prior_weight = torch.as_tensor(
            cfg.get("intrinsics_prior_weight", (0.01, 0.01, 0.005, 0.005)),
            dtype=self.intrinsics.dtype, device=self.device,
        ).flatten()
        if prior_weight.numel() != 4:
            raise ValueError("full_pinhole intrinsics_prior_weight must contain four values.")

        def solve(scale):
            backend.flow3r_ba(
                poses_work, disps_work, intrinsics_work, self.zeros, target, calibration_weight, eta,
                ii, jj, t0, t1, 1, 0, 1e-4, 0.1, False, True, 0.0, 0.0, scale,
                self._intrinsics_prior, prior_weight,
            )

        def reset_work():
            poses_work.copy_(poses_before)
            disps_work.copy_(disps_before)
            intrinsics_work.copy_(before)

        solve(1.0)
        proposed = intrinsics_work.clone()
        scale = self._full_intrinsics_update_scale(before, proposed, bootstrap_cfg, pp_cfg)
        if scale <= 0.0:
            reset_work()
            write_row("bounds", False, proposed, loss_before)
            return
        if scale < 0.999:
            reset_work()
            solve(scale)
        after = intrinsics_work.clone()
        if self._full_intrinsics_update_scale(before, after, bootstrap_cfg, pp_cfg) < 0.999:
            reset_work()
            write_row("bounds", False, proposed, loss_before)
            return
        with torch.no_grad():
            loss_after, _ = self._full_intrinsics_objective(
                target, calibration_weight, ii, jj, intrinsics_work
            )
        if not torch.isfinite(loss_after) or loss_after > loss_before:
            reset_work()
            write_row("loss_increase", False, proposed, loss_after)
            return
        if use_observation_subset:
            self.intrinsics[0].copy_(intrinsics_work)
        self.intrinsics[:self.counter.value] = self.intrinsics[0]
        write_row("accepted", True, proposed, loss_after)
        if stability_enabled and bool(stability_cfg.get("reset_after_accept", True)):
            self._focal_stable_ba_streak = 0

    def _calibration_observation_weights(self, weight, ii, jj):
        """Build the K-only observation subset without changing the main BA graph.

        Intrinsics are a shared scene-global variable, while short-baseline or
        high-uncertainty pixels are often explained equally well by local pose
        and depth changes.  This mask is used only by the extra joint Schur
        calibration solve and its acceptance objective; the ordinary DROID BA
        receives the original weights unchanged.
        """
        cfg = self.focal_calibration_cfg.get("calibration_observation", {}) or {}
        total_edges = int(ii.numel())
        if not bool(cfg.get("enable", False)):
            return weight, {
                "coverage": 1.0,
                "retained_edges": total_edges,
                "total_edges": total_edges,
            }

        mode = cfg.get("mode", "hard")
        if mode not in {"hard", "soft"}:
            raise ValueError(
                "tracking.focal_calibration.calibration_observation.mode must be 'hard' or 'soft'"
            )
        edge_span = (ii - jj).abs().float()
        edge_keep = edge_span >= int(cfg.get("min_keyframe_span", 0))
        pixel_keep = torch.ones_like(weight[:, 0], dtype=torch.bool)
        soft_scale = torch.ones_like(weight[:, 0])

        if bool(cfg.get("require_omega_uncertainty", True)):
            source_valid = self.omega_uncertainty_valid[ii].view(-1, 1, 1)
            if mode == "hard":
                max_uncertainty = float(cfg.get("omega_max_uncertainty", 0.93))
                source_static = self.omega_uncertainties[ii] <= max_uncertainty
                pixel_keep &= source_valid & source_static
            else:
                low = float(cfg.get("soft_omega_certain", 0.85))
                high = float(cfg.get("soft_omega_uncertain", 1.00))
                min_scale = float(cfg.get("soft_min_pixel_scale", 0.60))
                alpha = torch.clamp(
                    (self.omega_uncertainties[ii] - low) / max(high - low, 1e-6),
                    0.0,
                    1.0,
                )
                omega_scale = 1.0 - (1.0 - min_scale) * alpha
                soft_scale *= torch.where(source_valid, omega_scale, torch.ones_like(omega_scale))

        if bool(cfg.get("use_dense_patch_risk", False)):
            dense_valid = self.omega_dense_patch_valid[ii].view(-1, 1, 1)
            max_dense_risk = float(cfg.get("max_dense_patch_risk", 0.25))
            pixel_keep &= dense_valid & (self.omega_dense_patch_risk[ii] <= max_dense_risk)

        if mode == "hard":
            pixel_keep &= edge_keep.view(-1, 1, 1)
            calibration_weight = weight * pixel_keep[:, None].to(weight.dtype)
            retained_edges = int(edge_keep.sum().item())
        else:
            span_start = float(cfg.get("soft_span_start", 1.0))
            span_full = float(cfg.get("soft_span_full", 4.0))
            min_edge_scale = float(cfg.get("soft_min_edge_scale", 0.75))
            span_alpha = torch.clamp(
                (edge_span - span_start) / max(span_full - span_start, 1e-6), 0.0, 1.0
            )
            edge_scale = min_edge_scale + (1.0 - min_edge_scale) * span_alpha
            calibration_weight = weight * (soft_scale * edge_scale.view(-1, 1, 1))[:, None]
            retained_edges = total_edges
        total_mass = weight[:, 0].abs().sum().clamp_min(1e-6)
        retained_mass = calibration_weight[:, 0].abs().sum()
        return calibration_weight.contiguous(), {
            "coverage": float((retained_mass / total_mass).detach()),
            "retained_edges": retained_edges,
            "total_edges": total_edges,
        }

    def _full_intrinsics_update_scale(self, before, proposed, bootstrap_cfg, pp_cfg):
        """Return the largest safe fraction of a backend pinhole update."""
        prior = self._intrinsics_prior.to(before.device)
        max_log_step = float(
            bootstrap_cfg.get("max_log_step", self.focal_calibration_cfg.get("max_log_step", 0.002))
        )
        max_log_deviation = float(
            bootstrap_cfg.get("max_log_deviation", self.focal_calibration_cfg.get("max_log_deviation", 0.15))
        )
        max_pp_step = float(pp_cfg.get("max_step", 0.10))
        max_pp_deviation = float(pp_cfg.get("max_deviation", 1.50))
        scale = 1.0
        for index in (0, 1):
            before_value = torch.log((before[index] / prior[index]).clamp_min(1e-6)).item()
            proposed_value = torch.log((proposed[index] / prior[index]).clamp_min(1e-6)).item()
            delta = proposed_value - before_value
            if not np.isfinite(proposed_value) or abs(delta) > max_log_step:
                scale = min(scale, max_log_step / max(abs(delta), 1e-12))
            if abs(proposed_value) > max_log_deviation:
                target = np.copysign(max_log_deviation, delta if delta else proposed_value)
                scale = min(scale, max(0.0, (target - before_value) / delta) if delta else 0.0)
        for index in (2, 3):
            before_value = (before[index] - prior[index]).item()
            proposed_value = (proposed[index] - prior[index]).item()
            delta = proposed_value - before_value
            if not np.isfinite(proposed_value) or abs(delta) > max_pp_step:
                scale = min(scale, max_pp_step / max(abs(delta), 1e-12))
            if abs(proposed_value) > max_pp_deviation:
                target = np.copysign(max_pp_deviation, delta if delta else proposed_value)
                scale = min(scale, max(0.0, (target - before_value) / delta) if delta else 0.0)
        return float(np.clip(scale, 0.0, 1.0))

    def _full_intrinsics_observability(self, target, weight, ii, jj, intrinsics):
        """Autograd Hessian diagonals for log-focal and principal-point updates."""
        with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
            current = intrinsics.detach().clone()
            delta = torch.zeros(4, device=self.device, dtype=torch.float32, requires_grad=True)
            candidate = current.clone()
            candidate[0] = current[0] * torch.exp(delta[0])
            candidate[1] = current[1] * torch.exp(delta[1])
            candidate[2] = current[2] + delta[2]
            candidate[3] = current[3] + delta[3]
            loss, support = self._focal_data_loss(target, weight, ii, jj, candidate)
            gradient = torch.autograd.grad(loss, delta, create_graph=True)[0]
            hessian_diag = torch.stack([
                torch.autograd.grad(gradient[index], delta, retain_graph=index < 3)[0][index]
                for index in range(4)
            ])
        return loss.detach(), support.detach(), hessian_diag.detach()

    def _full_intrinsics_objective(self, target, weight, ii, jj, intrinsics):
        loss, support = self._focal_data_loss(target, weight, ii, jj, intrinsics)
        prior = self._intrinsics_prior.to(intrinsics.device)
        pp_cfg = self.focal_calibration_cfg.get("principal_point", {}) or {}
        pp_scale = float(pp_cfg.get("max_deviation", 1.50))
        focal_penalty = 0.05 * torch.log((intrinsics[:2] / prior[:2]).clamp_min(1e-6)).square().sum()
        principal_penalty = float(pp_cfg.get("prior_weight", 0.05)) * ((intrinsics[2:] - prior[2:]) / max(pp_scale, 1e-6)).square().sum()
        return loss + focal_penalty + principal_penalty, support

    def _focal_base_pose_update(self, poses_before, t0, t1):
        """Summarize the ordinary BA pose change before allowing a focal update."""
        if poses_before is None:
            return None
        count = min(self.counter.value, poses_before.shape[0])
        start = max(0, min(int(t0), count))
        end = min(count, int(t1) if t1 is not None else count)
        if end <= start:
            return {"translation_max": float("inf"), "rotation_max": float("inf"), "finite": False}

        before = poses_before[start:end]
        after = self.poses[start:end]
        translation = torch.linalg.vector_norm(after[:, :3] - before[:, :3], dim=-1)
        before_q = F.normalize(before[:, 3:7], dim=-1)
        after_q = F.normalize(after[:, 3:7], dim=-1)
        cosine = (before_q * after_q).sum(dim=-1).abs().clamp(0.0, 1.0)
        rotation = 2.0 * torch.acos(cosine)
        finite = bool(torch.isfinite(translation).all() and torch.isfinite(rotation).all())
        return {
            "translation_max": float(translation.max()) if translation.numel() else float("inf"),
            "rotation_max": float(rotation.max()) if rotation.numel() else float("inf"),
            "finite": finite,
        }

    def _focal_schur_observability(self, target, weight, ii, jj, intrinsics, compute_derivatives):
        """Measure focal information in the post-gating reprojection objective."""
        if not compute_derivatives:
            with torch.no_grad():
                loss, support = self._focal_data_loss(target, weight, ii, jj, intrinsics)
            nan = torch.full((), float("nan"), device=loss.device)
            return loss, support, nan, nan

        with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
            current = intrinsics.detach().clone()
            delta = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)
            candidate = current.clone()
            focal_scale = torch.exp(delta)
            candidate[0] = current[0] * focal_scale
            candidate[1] = current[1] * focal_scale
            loss, support = self._focal_data_loss(target, weight, ii, jj, candidate)
            gradient = torch.autograd.grad(loss, delta, create_graph=True)[0]
            hessian = torch.autograd.grad(gradient, delta)[0]
        return loss.detach(), support.detach(), gradient.detach(), hessian.detach()

    def _focal_data_loss(self, target, weight, ii, jj, intrinsics):
        """Weighted reprojection objective for the shared focal calibration block."""
        count = self.counter.value
        poses = lietorch.SE3(self.poses[:count].detach()[None])
        disps = self.disps[:count].detach()[None]
        intrinsics = intrinsics[None, None].expand(1, count, -1)
        coords, valid = pops.projective_transform(poses, disps, intrinsics, ii, jj)

        target_xy = target.permute(0, 2, 3, 1)[None]
        pixel_weight = weight.permute(0, 2, 3, 1).mean(dim=-1)[None]
        edge_mask = (ii != jj).float()[None, :, None, None]
        mask = valid[..., 0] * edge_mask
        numerator = (pixel_weight * mask * (target_xy - coords).square().sum(dim=-1)).sum()
        denominator = (pixel_weight * mask).sum().clamp_min(1e-6)
        return numerator / denominator, denominator

    def _write_focal_calibration_rows(self):
        if not self._focal_calibration_rows:
            return
        os.makedirs(self.output, exist_ok=True)
        path = os.path.join(self.output, "focal_calibration.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._focal_calibration_rows[0].keys())
            writer.writeheader()
            writer.writerows(self._focal_calibration_rows)

    def _maybe_calibrate_focal(self, target, weight, ii, jj, motion_only):
        """Alternating focal-only joint BA, adapted from flow3r's focal model.

        The DROID-W CUDA backend remains responsible for the pose/depth Schur
        step. This block optimizes the single shared focal increment from the
        same post-gating factors, then feeds that K into the next BA iteration.
        """
        cfg = self.focal_calibration_cfg
        if not self.focal_calibration_enabled or motion_only or self._focal_prior is None:
            return
        self._focal_ba_calls += 1
        every = max(1, int(cfg.get("every_n_ba", 8)))
        if self._focal_ba_calls % every != 0:
            return
        if self.counter.value < int(cfg.get("warmup_keyframes", 20)):
            return
        if int((ii != jj).sum().item()) < int(cfg.get("min_edges", 12)):
            return

        calibration_weight = weight
        if not bool(cfg.get("use_reliable_weights", True)):
            calibration_weight = torch.ones_like(weight)

        with torch.enable_grad(), torch.amp.autocast('cuda', enabled=False):
            current = self.intrinsics[0].detach().clone()
            relative_log = torch.log((current[0] / self._focal_prior[0]).clamp_min(1e-6))
            delta = torch.zeros((), device=self.device, dtype=torch.float32, requires_grad=True)

            def candidate_intrinsics(update):
                scale = torch.exp(update)
                candidate = current.clone()
                candidate[0] = current[0] * scale
                candidate[1] = current[1] * scale
                return candidate

            before, support = self._focal_data_loss(target, calibration_weight, ii, jj, candidate_intrinsics(delta))
            min_support = float(cfg.get("min_weighted_pixels", 1024.0))
            if float(support.detach()) < min_support:
                return
            prior_weight = float(cfg.get("prior_weight", 20.0))
            objective = before + prior_weight * (relative_log + delta).square()
            gradient = torch.autograd.grad(objective, delta, create_graph=True)[0]
            hessian = torch.autograd.grad(gradient, delta)[0]
            min_hessian = float(cfg.get("min_hessian", 1e-3))
            if not torch.isfinite(hessian) or float(hessian.detach()) < min_hessian:
                return

            damping = float(cfg.get("gn_damping", 1e-3))
            max_step = float(cfg.get("max_log_step", 0.002))
            step = torch.clamp(-gradient / (hessian + damping), -max_step, max_step).detach()
            max_deviation = float(cfg.get("max_log_deviation", 0.15))
            proposed_total = torch.clamp(relative_log + step, -max_deviation, max_deviation)
            accepted_step = proposed_total - relative_log
            proposed = candidate_intrinsics(accepted_step)
            with torch.no_grad():
                after, _ = self._focal_data_loss(target, calibration_weight, ii, jj, proposed)
            tolerance = float(cfg.get("max_loss_increase", 0.0))
            accepted = bool(torch.isfinite(after) and after <= before.detach() * (1.0 + tolerance))

        row = {
            "ba_call": self._focal_ba_calls,
            "keyframes": self.counter.value,
            "fx_before": float(current[0]),
            "fy_before": float(current[1]),
            "step_log_f": float(accepted_step),
            "hessian": float(hessian.detach()),
            "loss_before": float(before.detach()),
            "loss_after": float(after),
            "accepted": int(accepted),
        }
        if accepted:
            self.intrinsics[:self.counter.value, 0] = proposed[0]
            self.intrinsics[:self.counter.value, 1] = proposed[1]
            row["fx_after"] = float(proposed[0])
            row["fy_after"] = float(proposed[1])
        else:
            row["fx_after"] = float(current[0])
            row["fy_after"] = float(current[1])
        self._focal_calibration_rows.append(row)
        self._write_focal_calibration_rows()

    def apply_omega_edge_weight(self, weight, ii, jj):
        if not self.omega_prior.uncertainty_enabled:
            return weight

        mode = self.omega_prior.uncertainty_cfg.get("apply_to", "edge_weight")
        if mode in [False, None, "none", "droid_uncertainty"]:
            return weight

        source_uncertainty = self.omega_uncertainties[ii]
        source_valid = self.omega_uncertainty_valid[ii].view(-1, 1, 1)
        source_weight = self.omega_prior.edge_weight_from_uncertainty(source_uncertainty)
        source_weight = torch.where(source_valid, source_weight, torch.ones_like(source_weight))
        return (weight * source_weight[:, None]).contiguous()

    def apply_edge_dtf_edge_weight(self, weight, ii, jj):
        if not self.edge_dtf_prior.enabled:
            return weight

        coords, valid_mask = self.reproject(ii, jj)
        source_edge, residual_dtf = self.edge_dtf_residual_from_coords(
            ii,
            jj,
            coords,
            valid_mask=valid_mask,
        )
        source_edge = self.apply_edge_dtf_cycle_gate(
            ii,
            jj,
            source_edge,
            residual_dtf,
            coords=coords,
            valid_mask=valid_mask,
        )
        calibration = self.edge_dtf_token_calibration(ii, jj, source_edge)
        edge_weight = self.edge_dtf_prior.edge_weight(source_edge, residual_dtf, calibration=calibration)
        edge_weight = edge_weight * self.edge_dtf_token_dynamic_suppression(ii, jj, edge_weight)
        edge_weight = edge_weight * self.edge_dtf_token_spatial_suppression(
            ii,
            jj,
            source_edge,
            residual_dtf,
            edge_weight,
        )
        edge_weight = edge_weight * self.edge_dtf_patch_token_uncertainty(
            ii,
            jj,
            coords,
            valid_mask,
            source_edge,
            residual_dtf,
            edge_weight,
        )
        edge_weight = edge_weight * self.edge_dtf_per_edge_covariance(
            ii,
            jj,
            source_edge,
            residual_dtf,
            edge_weight,
        )
        return (weight * edge_weight[:, None]).contiguous()

    def edge_dtf_weight_from_coords(self, ii, jj, coords, graph_metadata=None):
        if not self.edge_dtf_prior.enabled:
            return torch.ones_like(coords[..., :1])

        source_edge, residual_dtf = self.edge_dtf_residual_from_coords(ii, jj, coords)
        source_edge = self.apply_edge_dtf_cycle_gate(ii, jj, source_edge, residual_dtf, coords=coords)
        calibration = self.edge_dtf_token_calibration(ii, jj, source_edge)
        edge_weight = self.edge_dtf_prior.edge_weight(source_edge, residual_dtf, calibration=calibration)
        edge_weight = edge_weight * self.edge_dtf_token_dynamic_suppression(ii, jj, edge_weight)
        edge_weight = edge_weight * self.edge_dtf_token_spatial_suppression(
            ii,
            jj,
            source_edge,
            residual_dtf,
            edge_weight,
        )
        edge_weight = edge_weight * self.edge_dtf_patch_token_uncertainty(
            ii,
            jj,
            coords,
            None,
            source_edge,
            residual_dtf,
            edge_weight,
            graph_metadata=graph_metadata,
        )
        edge_weight = edge_weight * self.edge_dtf_per_edge_covariance(
            ii,
            jj,
            source_edge,
            residual_dtf,
            edge_weight,
        )
        return edge_weight[None, ..., None].contiguous()

    def _omega_token_buffer_cfg(self):
        token_cfg = {}
        for key in ("token_calibration", "token_dynamic_suppression", "token_spatial_suppression", "per_edge_covariance"):
            cfg = self.edge_dtf_prior.cfg.get(key, {}) or {}
            if cfg:
                token_cfg.update({
                    "max_tokens": cfg.get("max_tokens", token_cfg.get("max_tokens", 17)),
                    "token_dim": cfg.get("token_dim", token_cfg.get("token_dim", 2048)),
                })
        return token_cfg

    def _omega_patch_token_buffer_cfg(self):
        edge_cfg = self.edge_dtf_prior.cfg.get("patch_token_uncertainty", {}) or {}
        model_cfg = ((self.cfg.get("omega_prior", {}) or {}).get("model", {}) or {}).get("patch_tokens", {}) or {}
        return {
            "token_dim": edge_cfg.get("token_dim", model_cfg.get("dim", 8)),
        }

    def edge_dtf_token_calibration(self, ii, jj, reference):
        token_cfg = self.edge_dtf_prior.cfg.get("token_calibration", {}) or {}
        if not bool(token_cfg.get("enable", False)):
            return None

        token_distance, valid = self.edge_dtf_token_distance(ii, jj, token_cfg)
        if token_distance is None:
            return torch.ones_like(reference)

        min_distance = float(token_cfg.get("min_distance", 0.02))
        max_distance = float(token_cfg.get("max_distance", 0.20))
        denom = max(max_distance - min_distance, 1e-6)
        calibrated = torch.clamp((token_distance - min_distance) / denom, 0.0, 1.0)

        strength = float(token_cfg.get("strength", 0.10))
        mode = token_cfg.get("mode", "amplify")
        if mode == "attenuate":
            scale = 1.0 - strength * calibrated
        elif mode == "amplify":
            scale = 1.0 + strength * calibrated
        else:
            raise ValueError(f"edge_dtf_prior.token_calibration.mode must be 'amplify' or 'attenuate', got {mode}")
        scale = torch.clamp(
            scale,
            min=float(token_cfg.get("min_scale", 1.0)),
            max=float(token_cfg.get("max_scale", 1.15)),
        ).view(-1, 1, 1)

        scale = torch.where(valid, scale, torch.ones_like(scale))
        return torch.ones_like(reference) * scale

    def edge_dtf_token_dynamic_suppression(self, ii, jj, reference):
        suppress_cfg = self.edge_dtf_prior.cfg.get("token_dynamic_suppression", {}) or {}
        if not bool(suppress_cfg.get("enable", False)):
            return torch.ones_like(reference)

        token_distance, valid = self.edge_dtf_token_distance(ii, jj, suppress_cfg)
        if token_distance is None:
            return torch.ones_like(reference)

        min_distance = float(suppress_cfg.get("min_distance", 0.02))
        max_distance = float(suppress_cfg.get("max_distance", 0.20))
        denom = max(max_distance - min_distance, 1e-6)
        dynamic_score = torch.clamp((token_distance - min_distance) / denom, 0.0, 1.0)

        strength = float(suppress_cfg.get("strength", 0.05))
        scale = 1.0 - strength * dynamic_score
        scale = torch.clamp(
            scale,
            min=float(suppress_cfg.get("min_scale", 0.95)),
            max=float(suppress_cfg.get("max_scale", 1.0)),
        ).view(-1, 1, 1)

        hard_threshold = suppress_cfg.get("hard_threshold", None)
        if hard_threshold is not None:
            hard_scale = float(suppress_cfg.get("hard_scale", scale.min().item()))
            hard_mask = (token_distance >= float(hard_threshold)).view(-1, 1, 1)
            scale = torch.where(hard_mask, torch.full_like(scale, hard_scale), scale)

        scale = torch.where(valid, scale, torch.ones_like(scale))
        return torch.ones_like(reference) * scale

    def edge_dtf_token_spatial_suppression(self, ii, jj, source_edge, residual_dtf, reference):
        spatial_cfg = self.edge_dtf_prior.cfg.get("token_spatial_suppression", {}) or {}
        if not bool(spatial_cfg.get("enable", False)):
            return torch.ones_like(reference)

        token_distance, valid = self.edge_dtf_token_distance(ii, jj, spatial_cfg)
        if token_distance is None:
            return torch.ones_like(reference)

        min_distance = float(spatial_cfg.get("min_distance", 0.02))
        max_distance = float(spatial_cfg.get("max_distance", 0.20))
        denom = max(max_distance - min_distance, 1e-6)
        token_score = torch.clamp((token_distance - min_distance) / denom, 0.0, 1.0)
        token_score = torch.where(valid.view(-1), token_score, torch.zeros_like(token_score)).view(-1, 1, 1)

        pixel_score = torch.zeros_like(reference)
        total = 0.0

        if bool(spatial_cfg.get("use_edge_residual", True)):
            edge_component = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            edge_weight = float(spatial_cfg.get("edge_weight", 0.5))
            pixel_score = pixel_score + edge_weight * edge_component
            total += edge_weight

        if bool(spatial_cfg.get("use_omega_uncertainty", True)):
            omega_valid = (self.omega_uncertainty_valid[ii] & self.omega_uncertainty_valid[jj])
            if omega_valid.any():
                omega_uncertainty = 0.5 * (self.omega_uncertainties[ii] + self.omega_uncertainties[jj])
                min_uncertainty = float(spatial_cfg.get("omega_min_uncertainty", 0.78))
                max_uncertainty = float(spatial_cfg.get("omega_max_uncertainty", 1.0))
                denom = max(max_uncertainty - min_uncertainty, 1e-6)
                omega_score = torch.clamp((omega_uncertainty - min_uncertainty) / denom, 0.0, 1.0)
                omega_score = torch.where(
                    omega_valid.view(-1, 1, 1),
                    omega_score,
                    torch.zeros_like(omega_score),
                )
                omega_weight = float(spatial_cfg.get("omega_weight", 1.0))
                pixel_score = pixel_score + omega_weight * omega_score
                total += omega_weight

        if total <= 0.0:
            return torch.ones_like(reference)

        if bool(spatial_cfg.get("normalize_components", True)):
            pixel_score = pixel_score / max(total, 1e-6)

        dynamic_score = token_score * torch.clamp(pixel_score, 0.0, 1.0)
        scale = 1.0 - float(spatial_cfg.get("strength", 0.03)) * dynamic_score
        return torch.clamp(
            scale,
            min=float(spatial_cfg.get("min_scale", 0.97)),
            max=float(spatial_cfg.get("max_scale", 1.0)),
        )

    def edge_dtf_patch_token_uncertainty(
        self, ii, jj, coords, valid_mask, source_edge, residual_dtf, reference, graph_metadata=None
    ):
        patch_cfg = self.edge_dtf_prior.cfg.get("patch_token_uncertainty", {}) or {}
        if not bool(patch_cfg.get("enable", False)):
            return torch.ones_like(reference)
        if coords is None:
            return torch.ones_like(reference)

        valid_pair = (self.omega_patch_valid[ii] & self.omega_patch_valid[jj]).view(-1, 1, 1)
        if not valid_pair.any():
            return torch.ones_like(reference)

        coords_ = coords.squeeze(0)
        valid_mask_ = None
        if valid_mask is not None:
            valid_mask_ = valid_mask.squeeze(0).squeeze(-1).bool()

        target_patch_maps = self.omega_patch_maps[jj]
        sampled_target, sample_valid = self.project_images_with_mask(target_patch_maps, coords_, valid_mask_)
        source_patch = self.omega_patch_maps[ii]

        source_patch = F.normalize(source_patch, p=2, dim=1, eps=1e-6)
        sampled_target = F.normalize(sampled_target, p=2, dim=1, eps=1e-6)
        token_distance = 1.0 - (source_patch * sampled_target).sum(dim=1)
        token_distance = torch.clamp(token_distance, min=0.0)

        valid_pixel = sample_valid & valid_pair
        token_distance = torch.where(valid_pixel, token_distance, torch.zeros_like(token_distance))

        min_distance = float(patch_cfg.get("min_distance", 0.05))
        max_distance = float(patch_cfg.get("max_distance", 0.60))
        denom = max(max_distance - min_distance, 1e-6)
        risk = torch.clamp((token_distance - min_distance) / denom, 0.0, 1.0)
        total = 1.0

        if bool(patch_cfg.get("combine_with_omega_uncertainty", True)):
            omega_valid = (self.omega_uncertainty_valid[ii] & self.omega_uncertainty_valid[jj]).view(-1, 1, 1)
            if omega_valid.any():
                omega_uncertainty = 0.5 * (self.omega_uncertainties[ii] + self.omega_uncertainties[jj])
                min_uncertainty = float(patch_cfg.get("omega_min_uncertainty", 0.78))
                max_uncertainty = float(patch_cfg.get("omega_max_uncertainty", 1.0))
                omega_score = torch.clamp(
                    (omega_uncertainty - min_uncertainty) / max(max_uncertainty - min_uncertainty, 1e-6),
                    0.0,
                    1.0,
                )
                omega_weight = float(patch_cfg.get("omega_weight", 0.25))
                risk = risk + omega_weight * torch.where(omega_valid, omega_score, torch.zeros_like(omega_score))
                total += omega_weight

        if bool(patch_cfg.get("combine_with_edge_residual", False)):
            edge_weight = float(patch_cfg.get("edge_weight", 0.25))
            risk = risk + edge_weight * torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            total += edge_weight

        if bool(patch_cfg.get("normalize_components", True)):
            risk = risk / max(total, 1e-6)

        remap_debug = None

        def apply_reliability_remap(risk_in, remap_cfg):
            nonlocal remap_debug
            edge_residual = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            min_residual = float(remap_cfg.get("min_residual", 0.04))
            max_residual = float(remap_cfg.get("max_residual", 0.28))
            residual_score = torch.clamp(
                (edge_residual - min_residual) / max(max_residual - min_residual, 1e-6),
                0.0,
                1.0,
            )
            min_risk = float(remap_cfg.get("min_risk", 0.0))
            max_risk = float(remap_cfg.get("max_risk", 0.25))
            confidence_gap = 1.0 - torch.clamp(
                (torch.clamp(risk_in, 0.0, 1.0) - min_risk) / max(max_risk - min_risk, 1e-6),
                0.0,
                1.0,
            )
            mismatch = torch.pow(residual_score, float(remap_cfg.get("residual_weight", 1.0)))
            mismatch = mismatch * torch.pow(
                confidence_gap,
                float(remap_cfg.get("confidence_gap_weight", 1.0)),
            )

            pixel_alpha = torch.clamp(
                (
                    mismatch
                    - float(remap_cfg.get("min_mismatch", 0.02))
                )
                / max(
                    float(remap_cfg.get("max_mismatch", 0.35))
                    - float(remap_cfg.get("min_mismatch", 0.02)),
                    1e-6,
                ),
                0.0,
                1.0,
            )
            mode = remap_cfg.get("mode", "smoothstep")
            if mode == "smoothstep":
                pixel_alpha = pixel_alpha * pixel_alpha * (3.0 - 2.0 * pixel_alpha)
            elif mode == "sigmoid":
                center = float(remap_cfg.get("sigmoid_center", 0.5))
                temperature = max(float(remap_cfg.get("sigmoid_temperature", 0.15)), 1e-6)
                pixel_alpha = torch.sigmoid((pixel_alpha - center) / temperature)
            elif mode == "linear":
                pass
            else:
                raise ValueError(
                    f"edge_dtf_prior.patch_token_uncertainty.reliability_remap.mode "
                    f"must be 'smoothstep', 'sigmoid', or 'linear', got {mode}"
                )

            valid = valid_pixel.detach().bool()
            flat_valid = valid.reshape(valid.shape[0], -1)
            denom = flat_valid.sum(dim=1).clamp(min=1).float()
            flat_mismatch = mismatch.detach().reshape(mismatch.shape[0], -1)
            edge_mismatch = (flat_mismatch * flat_valid.float()).sum(dim=1) / denom
            flat_edge_residual = edge_residual.detach().reshape(edge_residual.shape[0], -1)
            edge_residual_mean = (flat_edge_residual * flat_valid.float()).sum(dim=1) / denom
            edge_alpha = torch.clamp(
                (
                    edge_mismatch
                    - float(remap_cfg.get("edge_min_mismatch", 0.02))
                )
                / max(
                    float(remap_cfg.get("edge_max_mismatch", 0.10))
                    - float(remap_cfg.get("edge_min_mismatch", 0.02)),
                    1e-6,
                ),
                0.0,
                1.0,
            )
            if mode == "smoothstep":
                edge_alpha = edge_alpha * edge_alpha * (3.0 - 2.0 * edge_alpha)

            selective_alpha = torch.ones_like(edge_alpha)
            residual_coverage = torch.zeros_like(edge_alpha)
            mismatch_coverage = torch.zeros_like(edge_alpha)
            selective_cfg = remap_cfg.get("selective", {}) or {}
            if bool(selective_cfg.get("enable", False)):
                full_pixels = max(float(valid.shape[-2] * valid.shape[-1]), 1.0)
                valid_fraction = flat_valid.float().sum(dim=1) / full_pixels

                min_valid_fraction = float(selective_cfg.get("min_valid_fraction", 0.0))
                if min_valid_fraction > 0.0:
                    selective_alpha = torch.where(
                        valid_fraction >= min_valid_fraction,
                        selective_alpha,
                        torch.zeros_like(selective_alpha),
                    )

                min_edge_residual_mean = float(selective_cfg.get("min_edge_residual_mean", 0.0))
                if min_edge_residual_mean > 0.0:
                    max_edge_residual_mean = float(
                        selective_cfg.get("max_edge_residual_mean", min_edge_residual_mean)
                    )
                    residual_alpha = torch.clamp(
                        (edge_residual_mean - min_edge_residual_mean)
                        / max(max_edge_residual_mean - min_edge_residual_mean, 1e-6),
                        0.0,
                        1.0,
                    )
                    residual_alpha = residual_alpha * residual_alpha * (3.0 - 2.0 * residual_alpha)
                    selective_alpha = selective_alpha * residual_alpha

                min_residual_coverage = float(selective_cfg.get("min_residual_coverage", 0.0))
                if min_residual_coverage > 0.0:
                    residual_coverage = (
                        ((edge_residual.detach() >= float(selective_cfg.get("residual_coverage_threshold", 0.10))) & valid)
                        .reshape(edge_residual.shape[0], -1)
                        .float()
                        .sum(dim=1)
                        / denom
                    )
                    max_residual_coverage = float(
                        selective_cfg.get("max_residual_coverage", min_residual_coverage)
                    )
                    residual_coverage_alpha = torch.clamp(
                        (residual_coverage - min_residual_coverage)
                        / max(max_residual_coverage - min_residual_coverage, 1e-6),
                        0.0,
                        1.0,
                    )
                    residual_coverage_alpha = (
                        residual_coverage_alpha
                        * residual_coverage_alpha
                        * (3.0 - 2.0 * residual_coverage_alpha)
                    )
                    selective_alpha = selective_alpha * residual_coverage_alpha

                min_mismatch_coverage = float(selective_cfg.get("min_mismatch_coverage", 0.0))
                if min_mismatch_coverage > 0.0:
                    mismatch_coverage = (
                        ((mismatch.detach() >= float(selective_cfg.get("mismatch_coverage_threshold", 0.08))) & valid)
                        .reshape(mismatch.shape[0], -1)
                        .float()
                        .sum(dim=1)
                        / denom
                    )
                    max_mismatch_coverage = float(
                        selective_cfg.get("max_mismatch_coverage", min_mismatch_coverage)
                    )
                    mismatch_coverage_alpha = torch.clamp(
                        (mismatch_coverage - min_mismatch_coverage)
                        / max(max_mismatch_coverage - min_mismatch_coverage, 1e-6),
                        0.0,
                        1.0,
                    )
                    mismatch_coverage_alpha = (
                        mismatch_coverage_alpha
                        * mismatch_coverage_alpha
                        * (3.0 - 2.0 * mismatch_coverage_alpha)
                    )
                    selective_alpha = selective_alpha * mismatch_coverage_alpha

                if bool(selective_cfg.get("protect_low_residual_edges", False)):
                    low_min = float(selective_cfg.get("low_residual_protect_min", 0.0))
                    low_max = float(selective_cfg.get("low_residual_protect_max", 0.03))
                    protect_alpha = torch.clamp(
                        (edge_residual_mean - low_min) / max(low_max - low_min, 1e-6),
                        0.0,
                        1.0,
                    )
                    protect_alpha = protect_alpha * protect_alpha * (3.0 - 2.0 * protect_alpha)
                    selective_alpha = selective_alpha * protect_alpha

                edge_alpha = edge_alpha * selective_alpha

            if bool((patch_cfg.get("debug_stats", {}) or {}).get("enable", False)):
                flat_pixel_alpha = pixel_alpha.detach().reshape(pixel_alpha.shape[0], -1)
                flat_gain_alpha = (edge_alpha.view(-1, 1, 1) * pixel_alpha).detach().reshape(pixel_alpha.shape[0], -1)
                remap_debug = {
                    "edge_alpha": edge_alpha.detach().view(-1, 1, 1).expand_as(risk_in),
                    "selective_alpha": selective_alpha.detach().view(-1, 1, 1).expand_as(risk_in),
                    "pixel_alpha": pixel_alpha.detach(),
                    "gain_alpha": (edge_alpha.view(-1, 1, 1) * pixel_alpha).detach(),
                    "edge_mismatch": edge_mismatch.detach().view(-1, 1, 1).expand_as(risk_in),
                    "edge_residual_mean": edge_residual_mean.detach().view(-1, 1, 1).expand_as(risk_in),
                    "valid_fraction": (flat_valid.float().sum(dim=1) / max(float(valid.shape[-2] * valid.shape[-1]), 1.0))
                    .detach()
                    .view(-1, 1, 1)
                    .expand_as(risk_in),
                    "residual_coverage": residual_coverage.detach().view(-1, 1, 1).expand_as(risk_in),
                    "mismatch_coverage": mismatch_coverage.detach().view(-1, 1, 1).expand_as(risk_in),
                    "pixel_alpha_mean": ((flat_pixel_alpha * flat_valid.float()).sum(dim=1) / denom)
                    .detach()
                    .view(-1, 1, 1)
                    .expand_as(risk_in),
                    "gain_alpha_mean": ((flat_gain_alpha * flat_valid.float()).sum(dim=1) / denom)
                    .detach()
                    .view(-1, 1, 1)
                    .expand_as(risk_in),
                }

            edge_min_coverage = float(remap_cfg.get("edge_min_coverage", 0.0))
            if edge_min_coverage > 0.0:
                coverage_threshold = float(remap_cfg.get("coverage_threshold", 0.20))
                coverage = (
                    ((mismatch.detach() >= coverage_threshold) & valid)
                    .reshape(mismatch.shape[0], -1)
                    .float()
                    .sum(dim=1)
                    / denom
                )
                edge_alpha = torch.where(
                    coverage >= edge_min_coverage,
                    edge_alpha,
                    torch.zeros_like(edge_alpha),
                )

            gain = float(remap_cfg.get("weight", 0.04)) * edge_alpha.view(-1, 1, 1) * pixel_alpha
            risk_out = torch.clamp(
                risk_in + gain * (1.0 - torch.clamp(risk_in, 0.0, 1.0)),
                0.0,
                1.0,
            )
            return torch.where(valid_pixel, risk_out, risk_in)

        remap_cfg = patch_cfg.get("reliability_remap", {}) or {}
        if bool(remap_cfg.get("enable", False)) and not bool(remap_cfg.get("apply_after_gate", False)):
            risk = apply_reliability_remap(risk, remap_cfg)

        mismatch_cfg = patch_cfg.get("calibration_mismatch_boost", {}) or {}
        if bool(mismatch_cfg.get("enable", False)) and not bool(mismatch_cfg.get("apply_after_gate", False)):
            edge_residual = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            min_residual = float(mismatch_cfg.get("min_residual", 0.05))
            max_residual = float(mismatch_cfg.get("max_residual", 0.40))
            residual_score = torch.clamp(
                (edge_residual - min_residual) / max(max_residual - min_residual, 1e-6),
                0.0,
                1.0,
            )
            min_risk = float(mismatch_cfg.get("min_risk", 0.0))
            max_risk = float(mismatch_cfg.get("max_risk", 0.35))
            confidence_gap = 1.0 - torch.clamp(
                (risk - min_risk) / max(max_risk - min_risk, 1e-6),
                0.0,
                1.0,
            )
            mismatch = residual_score * confidence_gap
            min_pair_risk_mean = float(mismatch_cfg.get("min_pair_risk_mean", 0.0))
            if min_pair_risk_mean > 0.0:
                valid = valid_pixel.detach().bool()
                flat_valid = valid.reshape(valid.shape[0], -1)
                denom = flat_valid.sum(dim=1).clamp(min=1).float()
                flat_risk = torch.clamp(risk.detach(), 0.0, 1.0).reshape(risk.shape[0], -1)
                pair_risk_mean = (flat_risk * flat_valid.float()).sum(dim=1) / denom
                active_pair = (pair_risk_mean >= min_pair_risk_mean).view(-1, 1, 1)
                mismatch = torch.where(active_pair, mismatch, torch.zeros_like(mismatch))
            risk = torch.clamp(risk + float(mismatch_cfg.get("weight", 0.10)) * mismatch, 0.0, 1.0)

        low_parallax_debug = None
        low_parallax_alpha = None
        low_parallax_effective_alpha = None
        low_parallax_cfg = patch_cfg.get("low_parallax_adaptive", {}) or {}
        if bool(low_parallax_cfg.get("enable", False)):
            with torch.no_grad():
                beta = float(low_parallax_cfg.get("beta", self.cfg["tracking"].get("beta", 0.75)))
                edge_distance = self.distance(
                    ii.contiguous(),
                    jj.contiguous(),
                    beta=beta,
                    bidirectional=True,
                ).detach().float().view(-1)
                min_distance = float(low_parallax_cfg.get("min_distance", 0.0))
                max_distance = float(low_parallax_cfg.get("max_distance", 4.0))
                distance_alpha = 1.0 - torch.clamp(
                    (edge_distance - min_distance) / max(max_distance - min_distance, 1e-6),
                    0.0,
                    1.0,
                )
                if low_parallax_cfg.get("mode", "smoothstep") == "smoothstep":
                    distance_alpha = distance_alpha * distance_alpha * (3.0 - 2.0 * distance_alpha)
                elif low_parallax_cfg.get("mode", "smoothstep") == "linear":
                    pass
                else:
                    raise ValueError(
                        f"edge_dtf_prior.patch_token_uncertainty.low_parallax_adaptive.mode "
                        f"must be 'smoothstep' or 'linear', got {low_parallax_cfg.get('mode')}"
                    )
                valid = valid_pixel.detach().bool()
                flat_valid = valid.reshape(valid.shape[0], -1)
                denom = flat_valid.sum(dim=1).clamp(min=1).float()
                edge_residual = torch.clamp(source_edge.detach() * residual_dtf.detach(), 0.0, 1.0)
                residual_mean = (
                    edge_residual.reshape(edge_residual.shape[0], -1) * flat_valid.float()
                ).sum(dim=1) / denom
                min_residual = float(low_parallax_cfg.get("min_residual_mean", 0.0))
                max_residual = float(low_parallax_cfg.get("max_residual_mean", 0.08))
                residual_alpha = torch.clamp(
                    (residual_mean - min_residual) / max(max_residual - min_residual, 1e-6),
                    0.0,
                    1.0,
                )
                if bool(low_parallax_cfg.get("use_residual_gate", True)):
                    low_parallax_alpha = distance_alpha * residual_alpha
                else:
                    low_parallax_alpha = distance_alpha
                low_parallax_alpha = torch.clamp(low_parallax_alpha, 0.0, 1.0)

                support_cfg = low_parallax_cfg.get("graph_support", {}) or {}
                support_gate = torch.ones((), device=low_parallax_alpha.device)
                active_coverage = torch.ones((), device=low_parallax_alpha.device)
                support_signal = torch.ones((), device=low_parallax_alpha.device)
                if bool(support_cfg.get("enable", False)):
                    activation_threshold = float(support_cfg.get("activation_threshold", 0.05))
                    active_coverage = (low_parallax_alpha > activation_threshold).float().mean()
                    support_signal_name = support_cfg.get("signal", "active_coverage")
                    if support_signal_name == "active_coverage":
                        support_signal = active_coverage
                        min_support = float(support_cfg.get("min_coverage", 0.03))
                        max_support = float(support_cfg.get("max_coverage", 0.10))
                    elif support_signal_name == "mean_alpha_ema":
                        current_mean = low_parallax_alpha.mean().detach()
                        ema_decay = float(support_cfg.get("ema_decay", 0.80))
                        ema_decay = min(max(ema_decay, 0.0), 0.9999)
                        if self._low_parallax_support_ema is None:
                            self._low_parallax_support_ema = current_mean
                        else:
                            self._low_parallax_support_ema = (
                                ema_decay * self._low_parallax_support_ema
                                + (1.0 - ema_decay) * current_mean
                            )
                        support_signal = self._low_parallax_support_ema
                        min_support = float(support_cfg.get("min_mean_alpha", 0.006))
                        max_support = float(support_cfg.get("max_mean_alpha", 0.016))
                    else:
                        raise ValueError(
                            "edge_dtf_prior.patch_token_uncertainty.low_parallax_adaptive."
                            "graph_support.signal must be 'active_coverage' or 'mean_alpha_ema', "
                            f"got {support_signal_name}"
                        )
                    support_gate = torch.clamp(
                        (support_signal - min_support) / max(max_support - min_support, 1e-6),
                        0.0,
                        1.0,
                    )
                    support_mode = support_cfg.get("mode", "smoothstep")
                    if support_mode == "smoothstep":
                        support_gate = support_gate * support_gate * (3.0 - 2.0 * support_gate)
                    elif support_mode != "linear":
                        raise ValueError(
                            "edge_dtf_prior.patch_token_uncertainty.low_parallax_adaptive."
                            f"graph_support.mode must be 'smoothstep' or 'linear', got {support_mode}"
                        )
                low_parallax_effective_alpha = low_parallax_alpha * support_gate

            risk_gain = float(low_parallax_cfg.get("risk_gain", 0.0))
            if risk_gain > 0.0:
                alpha = low_parallax_effective_alpha.view(-1, 1, 1)
                risk = torch.clamp(risk + risk_gain * alpha * (1.0 - torch.clamp(risk, 0.0, 1.0)), 0.0, 1.0)

            if bool((patch_cfg.get("debug_stats", {}) or {}).get("enable", False)):
                low_parallax_debug = {
                    "alpha": low_parallax_alpha.detach().view(-1, 1, 1).expand_as(risk),
                    "effective_alpha": low_parallax_effective_alpha.detach().view(-1, 1, 1).expand_as(risk),
                    "support_gate": support_gate.detach().view(1, 1, 1).expand_as(risk),
                    "support_signal": support_signal.detach().view(1, 1, 1).expand_as(risk),
                    "active_coverage": active_coverage.detach().view(1, 1, 1).expand_as(risk),
                    "distance": edge_distance.detach().view(-1, 1, 1).expand_as(risk),
                    "residual_mean": residual_mean.detach().view(-1, 1, 1).expand_as(risk),
                }

        gate_for_adaptive = None
        gate_cfg = patch_cfg.get("conditional_gate", {}) or {}
        if bool(gate_cfg.get("enable", False)):
            gate = torch.zeros_like(reference)
            gate_total = 0.0

            if bool(gate_cfg.get("use_omega_uncertainty", True)):
                omega_valid = (self.omega_uncertainty_valid[ii] & self.omega_uncertainty_valid[jj]).view(-1, 1, 1)
                if omega_valid.any():
                    omega_uncertainty = 0.5 * (self.omega_uncertainties[ii] + self.omega_uncertainties[jj])
                    min_uncertainty = float(gate_cfg.get("omega_min_uncertainty", patch_cfg.get("omega_min_uncertainty", 0.82)))
                    max_uncertainty = float(gate_cfg.get("omega_max_uncertainty", patch_cfg.get("omega_max_uncertainty", 1.0)))
                    omega_gate = torch.clamp(
                        (omega_uncertainty - min_uncertainty) / max(max_uncertainty - min_uncertainty, 1e-6),
                        0.0,
                        1.0,
                    )
                    omega_weight = float(gate_cfg.get("omega_weight", 1.0))
                    gate = gate + omega_weight * torch.where(omega_valid, omega_gate, torch.zeros_like(omega_gate))
                    gate_total += omega_weight

            if bool(gate_cfg.get("use_edge_residual", True)):
                edge_residual = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
                min_residual = float(gate_cfg.get("edge_min_residual", 0.05))
                max_residual = float(gate_cfg.get("edge_max_residual", 0.40))
                edge_gate = torch.clamp(
                    (edge_residual - min_residual) / max(max_residual - min_residual, 1e-6),
                    0.0,
                    1.0,
                )
                edge_weight = float(gate_cfg.get("edge_weight", 1.0))
                gate = gate + edge_weight * edge_gate
                gate_total += edge_weight

            if gate_total <= 0.0:
                return torch.ones_like(reference)

            if bool(gate_cfg.get("normalize_components", True)):
                gate = gate / max(gate_total, 1e-6)

            min_gate_value = float(gate_cfg.get("min_gate", 0.0))
            min_gate = torch.full_like(gate[:, :1, :1], min_gate_value)
            fallback_min_gate_override = None

            adaptive_gate_cfg = gate_cfg.get("adaptive", {}) or {}
            if bool(adaptive_gate_cfg.get("enable", False)):
                signal_name = adaptive_gate_cfg.get("signal", "risk_mean")
                valid = valid_pixel.detach().bool()
                flat_valid = valid.reshape(valid.shape[0], -1)
                denom = flat_valid.sum(dim=1).clamp(min=1).float()

                def masked_pair_mean(tensor):
                    flat = tensor.reshape(tensor.shape[0], -1)
                    return (flat * flat_valid.float()).sum(dim=1) / denom

                def masked_pair_max(tensor):
                    flat = tensor.reshape(tensor.shape[0], -1)
                    return torch.where(flat_valid, flat, torch.zeros_like(flat)).max(dim=1).values

                risk_for_signal = torch.clamp(risk.detach(), 0.0, 1.0)
                edge_residual_for_signal = torch.clamp(source_edge.detach() * residual_dtf.detach(), 0.0, 1.0)
                gate_for_signal = torch.clamp(gate.detach(), 0.0, 1.0)
                risk_residual_for_signal = risk_for_signal * edge_residual_for_signal
                calibration_mismatch_for_signal = edge_residual_for_signal * (1.0 - risk_for_signal)
                if signal_name == "risk_mean":
                    adaptive_signal = masked_pair_mean(risk_for_signal)
                elif signal_name == "risk_max":
                    adaptive_signal = masked_pair_max(risk_for_signal)
                elif signal_name == "edge_residual_mean":
                    adaptive_signal = masked_pair_mean(edge_residual_for_signal)
                elif signal_name == "edge_residual_max":
                    adaptive_signal = masked_pair_max(edge_residual_for_signal)
                elif signal_name == "gate_mean":
                    adaptive_signal = masked_pair_mean(gate_for_signal)
                elif signal_name == "gate_max":
                    adaptive_signal = masked_pair_max(gate_for_signal)
                elif signal_name == "risk_residual_mean":
                    adaptive_signal = masked_pair_mean(risk_residual_for_signal)
                elif signal_name == "risk_residual_max":
                    adaptive_signal = masked_pair_max(risk_residual_for_signal)
                elif signal_name == "calibration_mismatch_mean":
                    adaptive_signal = masked_pair_mean(calibration_mismatch_for_signal)
                elif signal_name == "calibration_mismatch_max":
                    adaptive_signal = masked_pair_max(calibration_mismatch_for_signal)
                else:
                    raise ValueError(
                        f"edge_dtf_prior.patch_token_uncertainty.conditional_gate.adaptive.signal "
                        f"must be one of risk_mean, risk_max, edge_residual_mean, edge_residual_max, "
                        f"gate_mean, gate_max, risk_residual_mean, risk_residual_max, "
                        f"calibration_mismatch_mean, calibration_mismatch_max; got {signal_name}"
                    )

                min_signal = float(adaptive_gate_cfg.get("min_signal", 0.10))
                max_signal = float(adaptive_gate_cfg.get("max_signal", 0.20))
                alpha = torch.clamp(
                    (adaptive_signal - min_signal) / max(max_signal - min_signal, 1e-6),
                    0.0,
                    1.0,
                ).view(-1, 1, 1)

                min_multiplier = float(adaptive_gate_cfg.get("min_multiplier", 1.0))
                max_multiplier = float(adaptive_gate_cfg.get("max_multiplier", 1.0))
                multiplier = min_multiplier + (max_multiplier - min_multiplier) * alpha
                gate = torch.clamp(gate * multiplier, 0.0, 1.0)

                min_gate_low = adaptive_gate_cfg.get("min_gate_low", None)
                min_gate_high = adaptive_gate_cfg.get("min_gate_high", None)
                if min_gate_low is not None or min_gate_high is not None:
                    low = min_gate_value if min_gate_low is None else float(min_gate_low)
                    high = min_gate_value if min_gate_high is None else float(min_gate_high)
                    min_gate = low + (high - low) * alpha

                fallback_low = adaptive_gate_cfg.get("fallback_min_gate_low", None)
                fallback_high = adaptive_gate_cfg.get("fallback_min_gate_high", None)
                if fallback_low is not None or fallback_high is not None:
                    fallback_base = float(
                        (gate_cfg.get("evidence_floor", {}) or {}).get("fallback_min_gate", 0.0)
                    )
                    low = fallback_base if fallback_low is None else float(fallback_low)
                    high = fallback_base if fallback_high is None else float(fallback_high)
                    fallback_min_gate_override = low + (high - low) * alpha

            evidence_floor_cfg = gate_cfg.get("evidence_floor", {}) or {}
            if bool(evidence_floor_cfg.get("enable", False)) and float(min_gate.max().detach().cpu()) > 0.0:
                signal_name = evidence_floor_cfg.get("signal", "gate_max")
                if signal_name == "gate_max":
                    floor_signal = gate.amax(dim=(1, 2), keepdim=True)
                elif signal_name == "gate_mean":
                    floor_signal = gate.mean(dim=(1, 2), keepdim=True)
                else:
                    raise ValueError(
                        f"edge_dtf_prior.patch_token_uncertainty.conditional_gate."
                        f"evidence_floor.signal must be 'gate_max' or 'gate_mean', got {signal_name}"
                    )
                if fallback_min_gate_override is None:
                    fallback_min_gate = torch.full_like(
                        min_gate,
                        float(evidence_floor_cfg.get("fallback_min_gate", 0.0)),
                    )
                else:
                    fallback_min_gate = fallback_min_gate_override
                threshold = float(evidence_floor_cfg.get("threshold", 0.90))
                floor = torch.where(
                    floor_signal >= threshold,
                    min_gate,
                    fallback_min_gate,
                )
                gate = torch.maximum(gate, floor)
                gate = torch.clamp(gate, max=1.0)
            else:
                gate = torch.maximum(gate, min_gate)
                gate = torch.clamp(gate, max=1.0)
            gate_for_adaptive = gate
            mode = gate_cfg.get("mode", "multiply")
            if mode == "multiply":
                risk = risk * gate
            elif mode == "hard":
                risk = torch.where(
                    gate >= float(gate_cfg.get("hard_threshold", 0.5)),
                    risk,
                    torch.zeros_like(risk),
                )
            else:
                raise ValueError(
                    f"edge_dtf_prior.patch_token_uncertainty.conditional_gate.mode "
                    f"must be 'multiply' or 'hard', got {mode}"
                )

        if bool(remap_cfg.get("enable", False)) and bool(remap_cfg.get("apply_after_gate", False)):
            risk = apply_reliability_remap(risk, remap_cfg)

        if bool(mismatch_cfg.get("enable", False)) and bool(mismatch_cfg.get("apply_after_gate", False)):
            edge_residual = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            min_residual = float(mismatch_cfg.get("min_residual", 0.05))
            max_residual = float(mismatch_cfg.get("max_residual", 0.40))
            residual_score = torch.clamp(
                (edge_residual - min_residual) / max(max_residual - min_residual, 1e-6),
                0.0,
                1.0,
            )
            min_risk = float(mismatch_cfg.get("min_risk", 0.0))
            max_risk = float(mismatch_cfg.get("max_risk", 0.35))
            confidence_gap = 1.0 - torch.clamp(
                (risk - min_risk) / max(max_risk - min_risk, 1e-6),
                0.0,
                1.0,
            )
            mismatch = residual_score * confidence_gap
            min_pair_risk_mean = float(mismatch_cfg.get("min_pair_risk_mean", 0.0))
            if min_pair_risk_mean > 0.0:
                valid = valid_pixel.detach().bool()
                flat_valid = valid.reshape(valid.shape[0], -1)
                denom = flat_valid.sum(dim=1).clamp(min=1).float()
                flat_risk = torch.clamp(risk.detach(), 0.0, 1.0).reshape(risk.shape[0], -1)
                pair_risk_mean = (flat_risk * flat_valid.float()).sum(dim=1) / denom
                active_pair = (pair_risk_mean >= min_pair_risk_mean).view(-1, 1, 1)
                mismatch = torch.where(active_pair, mismatch, torch.zeros_like(mismatch))
            risk = torch.clamp(risk + float(mismatch_cfg.get("weight", 0.10)) * mismatch, 0.0, 1.0)

        strength = float(patch_cfg.get("strength", 0.03))
        adaptive_cfg = patch_cfg.get("adaptive_strength", {}) or {}
        if bool(adaptive_cfg.get("enable", False)):
            signal_name = adaptive_cfg.get("signal", "gate_mean")
            if signal_name == "gate_mean" and gate_for_adaptive is not None:
                signal = gate_for_adaptive.mean(dim=(1, 2), keepdim=True)
            elif signal_name == "risk_mean":
                signal = torch.clamp(risk, 0.0, 1.0).mean(dim=(1, 2), keepdim=True)
            else:
                raise ValueError(
                    f"edge_dtf_prior.patch_token_uncertainty.adaptive_strength.signal "
                    f"must be 'gate_mean' or 'risk_mean', got {signal_name}"
                )

            min_signal = float(adaptive_cfg.get("min_signal", 0.35))
            max_signal = float(adaptive_cfg.get("max_signal", 0.70))
            signal = torch.clamp(
                (signal - min_signal) / max(max_signal - min_signal, 1e-6),
                0.0,
                1.0,
            )
            min_multiplier = float(adaptive_cfg.get("min_multiplier", 0.55))
            max_multiplier = float(adaptive_cfg.get("max_multiplier", 1.10))
            strength = strength * (min_multiplier + (max_multiplier - min_multiplier) * signal)

        if low_parallax_effective_alpha is not None:
            strength_boost = float(low_parallax_cfg.get("strength_boost", 0.0))
            if strength_boost > 0.0:
                strength = strength * (1.0 + strength_boost * low_parallax_effective_alpha.view(-1, 1, 1))

        scale = 1.0 - strength * torch.clamp(risk, 0.0, 1.0)
        scale = torch.clamp(
            scale,
            min=float(patch_cfg.get("min_scale", 0.97)),
            max=float(patch_cfg.get("max_scale", 1.0)),
        )
        filter_cfg = patch_cfg.get("edge_residual_filter", {}) or {}
        filter_signal = None
        filter_active = None
        if bool(filter_cfg.get("enable", False)):
            valid = valid_pixel.detach().bool()
            flat_valid = valid.reshape(valid.shape[0], -1)
            denom = flat_valid.sum(dim=1).clamp(min=1).float()
            edge_residual = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            flat_residual = edge_residual.reshape(edge_residual.shape[0], -1)
            if filter_cfg.get("signal", "mean") == "max":
                residual_signal = torch.where(flat_valid, flat_residual, torch.zeros_like(flat_residual)).max(dim=1).values
            elif filter_cfg.get("signal", "mean") == "mean":
                residual_signal = (flat_residual * flat_valid.float()).sum(dim=1) / denom
            else:
                raise ValueError(
                    f"edge_dtf_prior.patch_token_uncertainty.edge_residual_filter.signal "
                    f"must be 'mean' or 'max', got {filter_cfg.get('signal')}"
                )
            filter_signal = residual_signal

            flat_risk = torch.clamp(risk, 0.0, 1.0).reshape(risk.shape[0], -1)
            risk_mean = (flat_risk * flat_valid.float()).sum(dim=1) / denom
            active = (
                (residual_signal >= float(filter_cfg.get("min_residual", 0.0)))
                & (risk_mean >= float(filter_cfg.get("min_risk_mean", 0.0)))
            ).view(-1, 1, 1)
            filter_active = active
            scale = torch.where(active, scale, torch.ones_like(scale))

        dense_cfg = patch_cfg.get("dense_map", {}) or {}
        dense_temporal_debug = None
        if bool(dense_cfg.get("enable", False)):
            temporal_cfg = dense_cfg.get("temporal_agreement", {}) or {}
            if (
                bool(temporal_cfg.get("enable", False))
                and self._dense_patch_map_update >= int(temporal_cfg.get("warmup_updates", 50))
            ):
                prior_dense_valid = self.omega_dense_patch_valid[ii].view(-1, 1, 1)
                prior_dense_risk = torch.where(
                    prior_dense_valid,
                    torch.clamp(self.omega_dense_patch_risk[ii], 0.0, 1.0),
                    torch.zeros_like(scale),
                )
                agreement = (
                    prior_dense_valid
                    & (prior_dense_risk >= float(temporal_cfg.get("min_dense_risk", 0.15)))
                    & (torch.clamp(risk, 0.0, 1.0) >= float(temporal_cfg.get("min_current_risk", 0.10)))
                )
                agreement_mode = temporal_cfg.get("mode", "mask")
                if agreement_mode == "mask":
                    scale = torch.where(agreement, scale, torch.ones_like(scale))
                elif agreement_mode == "boost":
                    boost_scale = 1.0 - float(temporal_cfg.get("boost_strength", 0.003)) * prior_dense_risk
                    boost_scale = torch.clamp(
                        boost_scale,
                        min=float(temporal_cfg.get("boost_min_scale", 0.997)),
                        max=1.0,
                    )
                    scale = torch.where(agreement, scale * boost_scale, scale)
                elif agreement_mode == "adaptive_boost":
                    edge_residual = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
                    residual_score = torch.clamp(
                        (
                            edge_residual
                            - float(temporal_cfg.get("residual_min", 0.05))
                        )
                        / max(
                            float(temporal_cfg.get("residual_max", 0.40))
                            - float(temporal_cfg.get("residual_min", 0.05)),
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                    risk_score = torch.clamp(
                        (
                            torch.clamp(risk, 0.0, 1.0)
                            - float(temporal_cfg.get("risk_min", 0.0))
                        )
                        / max(
                            float(temporal_cfg.get("risk_max", 0.35))
                            - float(temporal_cfg.get("risk_min", 0.0)),
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                    if temporal_cfg.get("adaptive_signal", "calibration_mismatch") == "residual":
                        adaptive_signal = residual_score
                    elif temporal_cfg.get("adaptive_signal", "calibration_mismatch") == "calibration_mismatch":
                        adaptive_signal = residual_score * (1.0 - risk_score)
                    else:
                        raise ValueError(
                            f"edge_dtf_prior.patch_token_uncertainty.dense_map."
                            f"temporal_agreement.adaptive_signal must be 'calibration_mismatch' "
                            f"or 'residual', got {temporal_cfg.get('adaptive_signal')}"
                        )
                    adaptive_alpha = torch.clamp(
                        (
                            adaptive_signal
                            - float(temporal_cfg.get("adaptive_min_signal", 0.05))
                        )
                        / max(
                            float(temporal_cfg.get("adaptive_max_signal", 0.35))
                            - float(temporal_cfg.get("adaptive_min_signal", 0.05)),
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                    min_strength = float(temporal_cfg.get("adaptive_min_strength", 0.0))
                    max_strength = float(temporal_cfg.get("adaptive_max_strength", 0.006))
                    adaptive_strength = min_strength + (max_strength - min_strength) * adaptive_alpha
                    active_edge = None
                    activation_cfg = temporal_cfg.get("activation", {}) or {}
                    if bool(activation_cfg.get("enable", False)):
                        valid = valid_pixel.detach().bool()
                        active = (agreement & valid).detach()
                        flat_valid = valid.reshape(valid.shape[0], -1)
                        flat_active = active.reshape(active.shape[0], -1)
                        valid_denom = flat_valid.sum(dim=1).clamp(min=1).float()
                        active_count = flat_active.sum(dim=1).clamp(min=1).float()
                        coverage = flat_active.sum(dim=1).float() / valid_denom

                        signal_name = activation_cfg.get("signal", "calibration_mismatch_mean")
                        flat_mismatch = (residual_score.detach() * (1.0 - risk_score.detach())).reshape(
                            residual_score.shape[0],
                            -1,
                        )
                        flat_residual = residual_score.detach().reshape(residual_score.shape[0], -1)
                        if signal_name == "calibration_mismatch_mean":
                            activation_signal = (flat_mismatch * flat_active.float()).sum(dim=1) / active_count
                        elif signal_name == "calibration_mismatch_max":
                            activation_signal = torch.where(
                                flat_active,
                                flat_mismatch,
                                torch.zeros_like(flat_mismatch),
                            ).max(dim=1).values
                        elif signal_name == "residual_mean":
                            activation_signal = (flat_residual * flat_active.float()).sum(dim=1) / active_count
                        elif signal_name == "residual_max":
                            activation_signal = torch.where(
                                flat_active,
                                flat_residual,
                                torch.zeros_like(flat_residual),
                            ).max(dim=1).values
                        else:
                            raise ValueError(
                                f"edge_dtf_prior.patch_token_uncertainty.dense_map."
                                f"temporal_agreement.activation.signal must be one of "
                                f"calibration_mismatch_mean, calibration_mismatch_max, "
                                f"residual_mean, residual_max; got {signal_name}"
                            )

                        active_edge = coverage >= float(activation_cfg.get("min_agreement_coverage", 0.02))
                        activation_mode = activation_cfg.get("mode", "hard")
                        if activation_mode == "hard":
                            active_edge = (
                                active_edge
                                & (activation_signal >= float(activation_cfg.get("min_signal", 0.05)))
                            ).view(-1, 1, 1)
                            adaptive_strength = torch.where(
                                active_edge,
                                adaptive_strength,
                                torch.zeros_like(adaptive_strength),
                            )
                        elif activation_mode == "soft":
                            min_signal = float(activation_cfg.get("min_signal", 0.05))
                            max_signal = float(activation_cfg.get("max_signal", 0.10))
                            alpha = torch.clamp(
                                (activation_signal - min_signal) / max(max_signal - min_signal, 1e-6),
                                0.0,
                                1.0,
                            )
                            min_multiplier = float(activation_cfg.get("min_multiplier", 0.0))
                            max_multiplier = float(activation_cfg.get("max_multiplier", 1.0))
                            multiplier = min_multiplier + (max_multiplier - min_multiplier) * alpha
                            active_edge = active_edge.view(-1, 1, 1)
                            multiplier = multiplier.view(-1, 1, 1)
                            adaptive_strength = torch.where(
                                active_edge,
                                adaptive_strength * multiplier,
                                torch.zeros_like(adaptive_strength),
                            )
                        else:
                            raise ValueError(
                                f"edge_dtf_prior.patch_token_uncertainty.dense_map."
                                f"temporal_agreement.activation.mode must be 'hard' or 'soft', "
                                f"got {activation_mode}"
                            )
                    boost_scale = 1.0 - adaptive_strength * prior_dense_risk
                    boost_scale = torch.clamp(
                        boost_scale,
                        min=float(temporal_cfg.get("boost_min_scale", 0.997)),
                        max=1.0,
                    )
                    dense_temporal_debug = {
                        "agreement": agreement.detach().float(),
                        "prior_dense_risk": prior_dense_risk.detach().float(),
                        "adaptive_signal": adaptive_signal.detach().float(),
                        "adaptive_strength": adaptive_strength.detach().float(),
                        "boost_scale": boost_scale.detach().float(),
                    }
                    if active_edge is not None:
                        dense_temporal_debug["active_edge"] = active_edge.detach().float().expand_as(scale)
                    scale = torch.where(agreement, scale * boost_scale, scale)
                else:
                    raise ValueError(
                        f"edge_dtf_prior.patch_token_uncertainty.dense_map."
                        f"temporal_agreement.mode must be 'mask', 'boost', or 'adaptive_boost', "
                        f"got {agreement_mode}"
                    )

            self._update_dense_patch_token_uncertainty_map(dense_cfg, ii, valid_pixel, risk)
            if bool(dense_cfg.get("apply_to_scale", False)):
                dense_valid = self.omega_dense_patch_valid[ii].view(-1, 1, 1)
                dense_risk = torch.where(
                    dense_valid,
                    torch.clamp(self.omega_dense_patch_risk[ii], 0.0, 1.0),
                    torch.zeros_like(scale),
                )
                dense_scale = 1.0 - float(dense_cfg.get("strength", 0.01)) * dense_risk
                dense_scale = torch.clamp(
                    dense_scale,
                    min=float(dense_cfg.get("min_scale", 0.99)),
                    max=float(dense_cfg.get("max_scale", 1.0)),
                )
                scale = torch.clamp(scale * dense_scale, min=0.0, max=1.0)

        v46_cfg = patch_cfg.get("v46_reliability", {}) or {}
        v46_debug = None
        if bool(v46_cfg.get("enable", False)):
            temporal_cfg = v46_cfg.get("temporal", {}) or {}
            prior_risk = None
            prior_valid = None
            if bool(temporal_cfg.get("enable", False)):
                prior_valid = self.omega_dense_patch_valid[ii].view(-1, 1, 1).expand_as(risk)
                prior_risk = self.omega_dense_patch_risk[ii]

            scale, v46_debug = apply_v46_reliability(
                scale=scale,
                risk=risk,
                edge_residual=torch.clamp(source_edge * residual_dtf, 0.0, 1.0),
                valid=valid_pixel,
                cfg=v46_cfg,
                graph_metadata=graph_metadata,
                prior_risk=prior_risk,
                prior_valid=prior_valid,
            )

            if bool(temporal_cfg.get("enable", False)) and not bool(dense_cfg.get("enable", False)):
                self._update_dense_patch_token_uncertainty_map(
                    {
                        "ema": temporal_cfg.get("ema", 0.80),
                        "save_visuals": False,
                    },
                    ii,
                    valid_pixel,
                    risk,
                )

        self._write_patch_token_uncertainty_stats(
            patch_cfg,
            ii,
            jj,
            valid_pixel,
            token_distance,
            risk,
            gate_for_adaptive,
            source_edge,
            residual_dtf,
            scale,
            strength,
            filter_signal,
            filter_active,
            dense_temporal_debug,
            remap_debug,
            low_parallax_debug,
            v46_debug,
        )
        return scale

    @torch.no_grad()
    def _update_dense_patch_token_uncertainty_map(self, dense_cfg, ii, valid_pixel, risk):
        if risk.numel() == 0:
            return

        self._dense_patch_map_update += 1
        ema = float(dense_cfg.get("ema", 0.80))
        ema = min(max(ema, 0.0), 0.999)

        risk_detached = torch.clamp(risk.detach().float(), 0.0, 1.0)
        valid_detached = valid_pixel.detach().bool()
        unique_indices = torch.unique(ii.detach())

        for frame_idx_tensor in unique_indices:
            frame_idx = int(frame_idx_tensor.item())
            edge_mask = ii == frame_idx_tensor
            if not bool(edge_mask.any()):
                continue

            valid_edges = valid_detached[edge_mask]
            count = valid_edges.float().sum(dim=0)
            if not bool((count > 0).any()):
                continue

            summed = (risk_detached[edge_mask] * valid_edges.float()).sum(dim=0)
            dense_risk = torch.where(count > 0, summed / count.clamp(min=1.0), torch.zeros_like(summed))

            previous = self.omega_dense_patch_risk[frame_idx]
            if bool(self.omega_dense_patch_valid[frame_idx]):
                updated = torch.where(count > 0, ema * previous + (1.0 - ema) * dense_risk, previous)
            else:
                updated = torch.where(count > 0, dense_risk, previous)

            self.omega_dense_patch_risk[frame_idx] = torch.clamp(updated, 0.0, 1.0)
            self.omega_dense_patch_valid[frame_idx] = True

            self._save_dense_patch_token_uncertainty_visual(dense_cfg, frame_idx)

    @torch.no_grad()
    def _save_dense_patch_token_uncertainty_visual(self, dense_cfg, frame_idx):
        if not bool(dense_cfg.get("save_visuals", False)):
            return
        every_n_updates = max(int(dense_cfg.get("every_n_updates", 200)), 1)
        if self._dense_patch_map_update % every_n_updates != 0:
            return
        max_visuals = max(int(dense_cfg.get("max_visuals", 64)), 1)
        if self._dense_patch_map_visual_count >= max_visuals:
            return

        out_dir = os.path.join(
            self.output,
            dense_cfg.get("output_dir", "dense_patch_token_uncertainty"),
        )
        os.makedirs(out_dir, exist_ok=True)

        risk_np = self.omega_dense_patch_risk[frame_idx].detach().float().cpu().numpy()
        risk_np = np.nan_to_num(risk_np, nan=0.0, posinf=1.0, neginf=0.0)
        risk_np = np.clip(risk_np, 0.0, 1.0)
        colored = (plt.get_cmap("magma")(risk_np)[..., :3] * 255.0).astype(np.uint8)
        timestamp = int(self.timestamp[frame_idx].detach().cpu()) if frame_idx < self.counter.value else frame_idx
        stem = f"dense_patch_uncertainty_kf_{frame_idx:03d}_ts_{timestamp:05d}_u_{self._dense_patch_map_update:06d}"
        Image.fromarray(colored).save(os.path.join(out_dir, f"{stem}.png"))
        if bool(dense_cfg.get("save_npy", False)):
            np.save(os.path.join(out_dir, f"{stem}.npy"), risk_np.astype(np.float32))
        self._dense_patch_map_visual_count += 1

    def _write_patch_token_uncertainty_stats(
        self,
        patch_cfg,
        ii,
        jj,
        valid_pixel,
        token_distance,
        risk,
        gate,
        source_edge,
        residual_dtf,
        scale,
        strength,
        filter_signal=None,
        filter_active=None,
        dense_temporal_debug=None,
        remap_debug=None,
        low_parallax_debug=None,
        v46_debug=None,
    ):
        stats_cfg = patch_cfg.get("debug_stats", {}) or {}
        if not bool(stats_cfg.get("enable", False)):
            return

        self._patch_token_stats_call += 1
        every_n_calls = max(int(stats_cfg.get("every_n_calls", 1)), 1)
        if self._patch_token_stats_call % every_n_calls != 0:
            return

        max_edges = max(int(stats_cfg.get("max_edges_per_call", 64)), 1)
        count = min(int(risk.shape[0]), max_edges)
        if count <= 0:
            return

        out_dir = os.path.join(self.output, "debug")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, stats_cfg.get("output_name", "patch_token_uncertainty_stats.csv"))

        values = {
            "token_distance": token_distance[:count].detach().float(),
            "risk": torch.clamp(risk[:count].detach().float(), 0.0, 1.0),
            "source_edge": source_edge[:count].detach().float(),
            "residual_dtf": residual_dtf[:count].detach().float(),
            "edge_residual": torch.clamp(
                source_edge[:count].detach().float() * residual_dtf[:count].detach().float(),
                0.0,
                1.0,
            ),
            "scale": scale[:count].detach().float(),
        }
        if gate is not None:
            values["gate"] = gate[:count].detach().float()
        if filter_active is not None:
            values["filter_active"] = filter_active[:count].detach().float().expand_as(values["scale"])
        if dense_temporal_debug is not None:
            for name, tensor in dense_temporal_debug.items():
                values[f"dense_temporal_{name}"] = tensor[:count].detach().float()
        if remap_debug is not None:
            for name, tensor in remap_debug.items():
                values[f"remap_{name}"] = tensor[:count].detach().float()
        if low_parallax_debug is not None:
            for name, tensor in low_parallax_debug.items():
                values[f"low_parallax_{name}"] = tensor[:count].detach().float()
        if v46_debug is not None:
            for name, tensor in v46_debug.items():
                values[f"v46_{name}"] = tensor[:count].detach().float()

        mask = valid_pixel[:count].detach().bool()
        flat_mask = mask.reshape(count, -1)
        denom = flat_mask.sum(dim=1).clamp(min=1).float()
        valid_ratio = flat_mask.float().mean(dim=1)

        def masked_mean(name):
            val = values[name].reshape(count, -1)
            return (val * flat_mask.float()).sum(dim=1) / denom

        def masked_max(name):
            val = values[name].reshape(count, -1)
            return torch.where(flat_mask, val, torch.zeros_like(val)).max(dim=1).values

        ii_cpu = ii[:count].detach().cpu().tolist()
        jj_cpu = jj[:count].detach().cpu().tolist()
        rows = []
        strength_tensor = strength.detach().flatten() if torch.is_tensor(strength) else None
        for k in range(count):
            if strength_tensor is not None:
                strength_value = float(strength_tensor[min(k, strength_tensor.numel() - 1)].cpu())
            else:
                strength_value = float(strength)
            row = {
                "call": self._patch_token_stats_call,
                "edge_local_index": k,
                "ii": int(ii_cpu[k]),
                "jj": int(jj_cpu[k]),
                "valid_ratio": float(valid_ratio[k].cpu()),
                "token_distance_mean": float(masked_mean("token_distance")[k].cpu()),
                "token_distance_max": float(masked_max("token_distance")[k].cpu()),
                "risk_mean": float(masked_mean("risk")[k].cpu()),
                "risk_max": float(masked_max("risk")[k].cpu()),
                "source_edge_mean": float(masked_mean("source_edge")[k].cpu()),
                "residual_dtf_mean": float(masked_mean("residual_dtf")[k].cpu()),
                "edge_residual_pixel_mean": float(masked_mean("edge_residual")[k].cpu()),
                "scale_mean": float(masked_mean("scale")[k].cpu()),
                "scale_min": float(torch.where(flat_mask[k], values["scale"][k].reshape(-1), torch.ones_like(values["scale"][k].reshape(-1))).min().cpu()),
                "strength": strength_value,
            }
            if filter_signal is not None and filter_active is not None:
                row["filter_signal"] = float(filter_signal[min(k, filter_signal.numel() - 1)].detach().cpu())
                row["filter_active"] = float(filter_active[min(k, filter_active.shape[0] - 1)].detach().float().cpu().mean())
            else:
                row["filter_signal"] = ""
                row["filter_active"] = ""
            if "gate" in values:
                row["gate_mean"] = float(masked_mean("gate")[k].cpu())
                row["gate_max"] = float(masked_max("gate")[k].cpu())
            else:
                row["gate_mean"] = ""
                row["gate_max"] = ""
            if dense_temporal_debug is not None:
                row["dense_temporal_agreement_ratio"] = float(masked_mean("dense_temporal_agreement")[k].cpu())
                row["dense_temporal_prior_risk_mean"] = float(masked_mean("dense_temporal_prior_dense_risk")[k].cpu())
                row["dense_temporal_adaptive_signal_mean"] = float(masked_mean("dense_temporal_adaptive_signal")[k].cpu())
                row["dense_temporal_adaptive_signal_max"] = float(masked_max("dense_temporal_adaptive_signal")[k].cpu())
                row["dense_temporal_strength_mean"] = float(masked_mean("dense_temporal_adaptive_strength")[k].cpu())
                row["dense_temporal_boost_scale_min"] = float(
                    torch.where(
                        flat_mask[k],
                        values["dense_temporal_boost_scale"][k].reshape(-1),
                        torch.ones_like(values["dense_temporal_boost_scale"][k].reshape(-1)),
                    ).min().cpu()
                )
                if "dense_temporal_active_edge" in values:
                    row["dense_temporal_active_edge"] = float(
                        values["dense_temporal_active_edge"][k].detach().float().mean().cpu()
                    )
                else:
                    row["dense_temporal_active_edge"] = ""
            else:
                row["dense_temporal_agreement_ratio"] = ""
                row["dense_temporal_prior_risk_mean"] = ""
                row["dense_temporal_adaptive_signal_mean"] = ""
                row["dense_temporal_adaptive_signal_max"] = ""
                row["dense_temporal_strength_mean"] = ""
                row["dense_temporal_boost_scale_min"] = ""
                row["dense_temporal_active_edge"] = ""
            if remap_debug is not None:
                row["remap_edge_alpha"] = float(masked_mean("remap_edge_alpha")[k].cpu())
                row["remap_selective_alpha"] = float(masked_mean("remap_selective_alpha")[k].cpu())
                row["remap_pixel_alpha_mean"] = float(masked_mean("remap_pixel_alpha")[k].cpu())
                row["remap_pixel_alpha_max"] = float(masked_max("remap_pixel_alpha")[k].cpu())
                row["remap_gain_alpha_mean"] = float(masked_mean("remap_gain_alpha")[k].cpu())
                row["remap_gain_alpha_max"] = float(masked_max("remap_gain_alpha")[k].cpu())
                row["remap_edge_mismatch"] = float(masked_mean("remap_edge_mismatch")[k].cpu())
                row["remap_edge_residual_mean"] = float(masked_mean("remap_edge_residual_mean")[k].cpu())
                row["remap_valid_fraction"] = float(masked_mean("remap_valid_fraction")[k].cpu())
                row["remap_residual_coverage"] = float(masked_mean("remap_residual_coverage")[k].cpu())
                row["remap_mismatch_coverage"] = float(masked_mean("remap_mismatch_coverage")[k].cpu())
            else:
                row["remap_edge_alpha"] = ""
                row["remap_selective_alpha"] = ""
                row["remap_pixel_alpha_mean"] = ""
                row["remap_pixel_alpha_max"] = ""
                row["remap_gain_alpha_mean"] = ""
                row["remap_gain_alpha_max"] = ""
                row["remap_edge_mismatch"] = ""
                row["remap_edge_residual_mean"] = ""
                row["remap_valid_fraction"] = ""
                row["remap_residual_coverage"] = ""
                row["remap_mismatch_coverage"] = ""
            if low_parallax_debug is not None:
                row["low_parallax_alpha"] = float(masked_mean("low_parallax_alpha")[k].cpu())
                row["low_parallax_effective_alpha"] = float(masked_mean("low_parallax_effective_alpha")[k].cpu())
                row["low_parallax_support_gate"] = float(masked_mean("low_parallax_support_gate")[k].cpu())
                row["low_parallax_support_signal"] = float(masked_mean("low_parallax_support_signal")[k].cpu())
                row["low_parallax_active_coverage"] = float(masked_mean("low_parallax_active_coverage")[k].cpu())
                row["low_parallax_distance"] = float(masked_mean("low_parallax_distance")[k].cpu())
                row["low_parallax_residual_mean"] = float(masked_mean("low_parallax_residual_mean")[k].cpu())
            else:
                row["low_parallax_alpha"] = ""
                row["low_parallax_effective_alpha"] = ""
                row["low_parallax_support_gate"] = ""
                row["low_parallax_support_signal"] = ""
                row["low_parallax_active_coverage"] = ""
                row["low_parallax_distance"] = ""
                row["low_parallax_residual_mean"] = ""
            if v46_debug is not None:
                for name in (
                    "prior_score",
                    "geometry_score",
                    "agreement",
                    "temporal_score",
                    "observability",
                    "endpoint_degree",
                    "reverse_support",
                    "edge_span",
                    "budget_multiplier",
                    "candidate_scale",
                    "output_scale",
                ):
                    row[f"v46_{name}_mean"] = float(masked_mean(f"v46_{name}")[k].cpu())
            else:
                for name in (
                    "prior_score",
                    "geometry_score",
                    "agreement",
                    "temporal_score",
                    "observability",
                    "endpoint_degree",
                    "reverse_support",
                    "edge_span",
                    "budget_multiplier",
                    "candidate_scale",
                    "output_scale",
                ):
                    row[f"v46_{name}_mean"] = ""
            rows.append(row)

        fieldnames = [
            "call",
            "edge_local_index",
            "ii",
            "jj",
            "valid_ratio",
            "token_distance_mean",
            "token_distance_max",
            "risk_mean",
            "risk_max",
            "gate_mean",
            "gate_max",
            "source_edge_mean",
            "residual_dtf_mean",
            "edge_residual_pixel_mean",
            "scale_mean",
            "scale_min",
            "strength",
            "filter_signal",
            "filter_active",
            "dense_temporal_agreement_ratio",
            "dense_temporal_prior_risk_mean",
            "dense_temporal_adaptive_signal_mean",
            "dense_temporal_adaptive_signal_max",
            "dense_temporal_strength_mean",
            "dense_temporal_boost_scale_min",
            "dense_temporal_active_edge",
            "remap_edge_alpha",
            "remap_selective_alpha",
            "remap_pixel_alpha_mean",
            "remap_pixel_alpha_max",
            "remap_gain_alpha_mean",
            "remap_gain_alpha_max",
            "remap_edge_mismatch",
            "remap_edge_residual_mean",
            "remap_valid_fraction",
            "remap_residual_coverage",
            "remap_mismatch_coverage",
            "low_parallax_alpha",
            "low_parallax_effective_alpha",
            "low_parallax_support_gate",
            "low_parallax_support_signal",
            "low_parallax_active_coverage",
            "low_parallax_distance",
            "low_parallax_residual_mean",
            "v46_prior_score_mean",
            "v46_geometry_score_mean",
            "v46_agreement_mean",
            "v46_temporal_score_mean",
            "v46_observability_mean",
            "v46_endpoint_degree_mean",
            "v46_reverse_support_mean",
            "v46_edge_span_mean",
            "v46_budget_multiplier_mean",
            "v46_candidate_scale_mean",
            "v46_output_scale_mean",
        ]
        file_exists = os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists and not self._patch_token_stats_header_written:
                writer.writeheader()
            writer.writerows(rows)
        self._patch_token_stats_header_written = True

    def edge_dtf_per_edge_covariance(self, ii, jj, source_edge, residual_dtf, reference):
        cov_cfg = self.edge_dtf_prior.cfg.get("per_edge_covariance", {}) or {}
        if not bool(cov_cfg.get("enable", False)):
            return torch.ones_like(reference)

        risk = torch.zeros_like(reference)
        total = 0.0

        if bool(cov_cfg.get("use_edge_residual", True)):
            edge_component = torch.clamp(source_edge * residual_dtf, 0.0, 1.0)
            edge_weight = float(cov_cfg.get("edge_weight", 0.5))
            risk = risk + edge_weight * edge_component
            total += edge_weight

        if bool(cov_cfg.get("use_token_distance", True)):
            token_distance, valid = self.edge_dtf_token_distance(ii, jj, cov_cfg)
            if token_distance is not None:
                min_distance = float(cov_cfg.get("min_distance", 0.02))
                max_distance = float(cov_cfg.get("max_distance", 0.20))
                denom = max(max_distance - min_distance, 1e-6)
                token_score = torch.clamp((token_distance - min_distance) / denom, 0.0, 1.0)
                token_score = torch.where(valid.view(-1), token_score, torch.zeros_like(token_score))
                token_weight = float(cov_cfg.get("token_weight", 1.0))
                risk = risk + token_weight * token_score.view(-1, 1, 1)
                total += token_weight

        if bool(cov_cfg.get("use_omega_uncertainty", True)):
            omega_valid = (self.omega_uncertainty_valid[ii] & self.omega_uncertainty_valid[jj])
            if omega_valid.any():
                omega_uncertainty = 0.5 * (self.omega_uncertainties[ii] + self.omega_uncertainties[jj])
                min_uncertainty = float(cov_cfg.get("omega_min_uncertainty", 0.78))
                max_uncertainty = float(cov_cfg.get("omega_max_uncertainty", 1.0))
                denom = max(max_uncertainty - min_uncertainty, 1e-6)
                omega_score = torch.clamp((omega_uncertainty - min_uncertainty) / denom, 0.0, 1.0)
                omega_score = torch.where(
                    omega_valid.view(-1, 1, 1),
                    omega_score,
                    torch.zeros_like(omega_score),
                )
                omega_weight = float(cov_cfg.get("omega_weight", 1.0))
                risk = risk + omega_weight * omega_score
                total += omega_weight

        if total <= 0.0:
            return torch.ones_like(reference)

        if bool(cov_cfg.get("normalize_components", True)):
            risk = risk / max(total, 1e-6)

        risk = torch.clamp(risk, 0.0, 1.0)
        strength = float(cov_cfg.get("strength", 0.04))
        reliable_boost = float(cov_cfg.get("reliable_boost", 0.0))
        scale = 1.0 - strength * risk
        if reliable_boost > 0.0:
            scale = scale + reliable_boost * (1.0 - risk)
        return torch.clamp(
            scale,
            min=float(cov_cfg.get("min_scale", 0.96)),
            max=float(cov_cfg.get("max_scale", 1.0)),
        )

    def edge_dtf_token_distance(self, ii, jj, token_cfg):
        valid = (self.omega_token_valid[ii] & self.omega_token_valid[jj]).view(-1, 1, 1)
        if not valid.any():
            return None, valid

        tokens_i = self.omega_tokens[ii].float()
        tokens_j = self.omega_tokens[jj].float()

        if bool(token_cfg.get("exclude_camera_token", False)) and tokens_i.shape[1] > 1:
            tokens_i = tokens_i[:, 1:]
            tokens_j = tokens_j[:, 1:]

        pooling = token_cfg.get("pooling", "mean")
        if pooling == "max":
            pooled_i = tokens_i.max(dim=1).values
            pooled_j = tokens_j.max(dim=1).values
        elif pooling == "mean":
            pooled_i = tokens_i.mean(dim=1)
            pooled_j = tokens_j.mean(dim=1)
        else:
            raise ValueError(f"Omega token pooling must be 'mean' or 'max', got {pooling}")

        pooled_i = F.normalize(pooled_i, p=2, dim=-1, eps=1e-6)
        pooled_j = F.normalize(pooled_j, p=2, dim=-1, eps=1e-6)
        token_distance = 1.0 - (pooled_i * pooled_j).sum(dim=-1)
        return torch.clamp(token_distance, min=0.0), valid

    def edge_dtf_residual_from_coords(self, ii, jj, coords, valid_mask=None):
        coords_ = coords.squeeze(0)
        target_dtf = self.edge_dtf_maps[jj].unsqueeze(1)
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(0).squeeze(-1).bool()
        sampled_dtf, sample_valid = self.project_images_with_mask(target_dtf, coords_, valid_mask)
        sampled_dtf = sampled_dtf.squeeze(1)

        valid_pair = (self.edge_dtf_valid[ii] & self.edge_dtf_valid[jj]).view(-1, 1, 1)
        residual_dtf = torch.where(
            sample_valid & valid_pair,
            sampled_dtf,
            torch.zeros_like(sampled_dtf),
        )
        source_edge = torch.where(
            valid_pair,
            self.edge_dtf_edges[ii] * self.edge_dtf_static_gate(ii),
            torch.zeros_like(residual_dtf),
        )
        return source_edge, residual_dtf

    def apply_edge_dtf_cycle_gate(self, ii, jj, source_edge, residual_dtf, coords=None, valid_mask=None):
        cycle_cfg = self.edge_dtf_prior.cfg.get("cycle_consistency", {}) or {}
        if not bool(cycle_cfg.get("enable", False)):
            return source_edge

        mode = cycle_cfg.get("mode", "pair")
        if mode == "pixel":
            return self.apply_edge_dtf_pixel_cycle_gate(
                ii,
                jj,
                source_edge,
                residual_dtf,
                coords,
                valid_mask,
                cycle_cfg,
            )

        reverse_coords, reverse_valid = self.reproject(jj, ii)
        reverse_edge, reverse_dtf = self.edge_dtf_residual_from_coords(
            jj,
            ii,
            reverse_coords,
            valid_mask=reverse_valid,
        )

        fwd_mean, fwd_mass = self.edge_dtf_mean_residual(source_edge, residual_dtf)
        rev_mean, rev_mass = self.edge_dtf_mean_residual(reverse_edge, reverse_dtf)

        max_asym = float(cycle_cfg.get("max_asymmetry", 0.08))
        min_residual = float(cycle_cfg.get("min_mean_residual", 0.0))
        max_residual = float(cycle_cfg.get("max_mean_residual", 1.0))
        min_edge_mass = float(cycle_cfg.get("min_edge_mass", 16.0))
        failed_scale = float(cycle_cfg.get("failed_scale", 0.0))

        pair_static = (
            (torch.abs(fwd_mean - rev_mean) <= max_asym)
            & (fwd_mean >= min_residual)
            & (rev_mean >= min_residual)
            & (fwd_mean <= max_residual)
            & (rev_mean <= max_residual)
            & (fwd_mass >= min_edge_mass)
            & (rev_mass >= min_edge_mass)
        )
        gate = torch.where(
            pair_static,
            torch.ones_like(fwd_mean),
            torch.full_like(fwd_mean, failed_scale),
        )
        return source_edge * gate.view(-1, 1, 1)

    def edge_dtf_mean_residual(self, source_edge, residual_dtf):
        edge_mass = source_edge.sum(dim=(1, 2))
        residual_sum = (source_edge * residual_dtf).sum(dim=(1, 2))
        mean_residual = residual_sum / edge_mass.clamp(min=1e-6)
        return mean_residual, edge_mass

    def apply_edge_dtf_pixel_cycle_gate(self, ii, jj, source_edge, residual_dtf, coords, valid_mask, cycle_cfg):
        if coords is None:
            return source_edge

        coords_ = coords.squeeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(0).squeeze(-1).bool()

        target_edge = (self.edge_dtf_edges[jj] * self.edge_dtf_static_gate(jj)).unsqueeze(1)
        sampled_target_edge, sample_valid = self.project_images_with_mask(target_edge, coords_, valid_mask)
        sampled_target_edge = sampled_target_edge.squeeze(1)

        valid_pair = (self.edge_dtf_valid[ii] & self.edge_dtf_valid[jj]).view(-1, 1, 1)
        source_dtf = torch.where(
            valid_pair,
            self.edge_dtf_maps[ii],
            torch.zeros_like(residual_dtf),
        )
        reverse_like_residual = torch.where(
            sample_valid & valid_pair,
            sampled_target_edge * source_dtf,
            torch.zeros_like(residual_dtf),
        )

        forward_residual = source_edge * residual_dtf
        asymmetry = torch.abs(forward_residual - reverse_like_residual)

        max_asym = float(cycle_cfg.get("max_pixel_asymmetry", cycle_cfg.get("max_asymmetry", 0.12)))
        min_residual = float(cycle_cfg.get("min_mean_residual", 0.0))
        max_residual = float(cycle_cfg.get("max_mean_residual", 1.0))
        pixel_keep = (
            (asymmetry <= max_asym)
            & (forward_residual >= min_residual)
            & (forward_residual <= max_residual)
            & sample_valid
            & valid_pair
        )
        return source_edge * pixel_keep.to(source_edge.dtype)

    def edge_dtf_static_gate(self, ii):
        gate_cfg = self.edge_dtf_prior.cfg.get("gate", {}) or {}
        gate = torch.ones_like(self.edge_dtf_edges[ii])

        if bool(gate_cfg.get("omega_uncertainty", False)):
            omega_valid = self.omega_uncertainty_valid[ii].view(-1, 1, 1)
            omega_static = self.omega_uncertainties[ii] <= float(gate_cfg.get("omega_max_uncertainty", 0.86))
            gate = gate * (omega_valid & omega_static).to(gate.dtype)

        if bool(gate_cfg.get("droid_uncertainty", False)) and self.uncertainty_aware:
            droid_static = self.uncertainties[ii] <= float(gate_cfg.get("droid_max_uncertainty", 0.90))
            gate = gate * droid_static.to(gate.dtype)

        return gate

    @torch.no_grad()
    def visualize_uncertainty(self, target, weight, ii, jj, frame_choice="nearest", mode="Before"):
        """ 
        visualize the uncertainty before and after optimization, reprojection error, and the weight prediction
        """
        # for ind, i, j in zip(range(ii.shape[0]), ii, jj):
        i = ii.max().item()
        mask = (ii == i)                       # bool mask, mark all the same max ii
        idx_nd = mask.nonzero(as_tuple=False)  # [K, ndim]

        # 2) get all candidates of j
        j_candidates = jj[mask]                # [K]

        # 3) select the max j from the candidates and get the relative index
        if frame_choice == "nearest":
            j, rel = j_candidates.max(dim=0)       # rel is the index in j_candidates
        elif frame_choice == "farthest":
            j, rel = j_candidates.min(dim=0)
        elif frame_choice == "random":
            rand_idx = torch.randint(0, j_candidates.shape[0], (1,))
            j = j_candidates[rand_idx].item()
            rel = rand_idx
        rel = rel.item()

        # 4) output the max j and the corresponding original index
        ind = idx_nd[rel].item()

        weight_pred = self.upsample_weight(weight[ind].squeeze()).cpu().numpy()

        img_i = self.images[i].permute(1,2,0).numpy()
        img_j = self.images[j].permute(1,2,0).numpy()

        # compute the reprojection error between img_i and img_j
        reprojected_coords, valid_mask = self.reproject(i, j)
        reprojection_error = target[ind].permute(1,2,0) - reprojected_coords.squeeze()
        reprojection_error_x = reprojection_error[:,:,0].abs()
        reprojection_error_y = reprojection_error[:,:,1].abs()
        reprojection_error_norm = torch.norm(reprojection_error, dim=-1)

        reprojection_error_x = self.upsample_weight(reprojection_error_x.unsqueeze(0)).cpu().numpy()
        reprojection_error_y = self.upsample_weight(reprojection_error_y.unsqueeze(0)).cpu().numpy()
        reprojection_error_norm = self.upsample_weight(reprojection_error_norm.unsqueeze(0)).cpu().numpy()

        disp_i = self.upsample_weight(self.disps[i].unsqueeze(0)).cpu().numpy()
        mono_disp_i = self.upsample_weight(self.mono_disps[i].unsqueeze(0)).cpu().numpy()

        # compute DINO features similarity between img_i and img_j
        img_w = self.wd // self.down_scale
        img_h = self.ht // self.down_scale
        dino_feats_i = F.normalize(self.dino_feats_resize[i], p=2, dim=1).unsqueeze(0)   # [1, C, H, W]
        dino_feats_j = F.normalize(self.dino_feats_resize[j], p=2, dim=1).unsqueeze(0)   # [1, C, H, W]
        dino_feats_reproj, valid_mask = self.project_images_with_mask(dino_feats_j, reprojected_coords.resize(1, img_h, img_w, 2))
        dino_feats_reproj = F.normalize(dino_feats_reproj, p=2, dim=1)  # Normalize features for cosine similarity
        dino_feats_similarity_reproj = (dino_feats_i * dino_feats_reproj).sum(dim=1)  # [1, H, W]
        dino_feats_similarity_reproj = self.upsample_weight(dino_feats_similarity_reproj).cpu().numpy()

        # Create the figure WITH constrained layout
        fig_height = 12
        fig_width = 13
        aspect_ratio = img_i.shape[1] / img_i.shape[0]
        fig_width = fig_width * aspect_ratio
        fig, axes = plt.subplots(3, 4, figsize=(fig_width, fig_height), constrained_layout=True)

        # vis image i
        axes[0,0].imshow(img_i)
        axes[0,0].set_title("Image i Visualization")
        # visualize img_j
        axes[0,1].imshow(img_j)
        axes[0,1].set_title("Image j Visualization")

        # visualize the weight (X/Y)
        wmax = weight_pred.max()
        axes[0,2].imshow(img_i)
        im1 = axes[0,2].imshow(weight_pred[0], cmap='jet', alpha=0.7, vmin=0, vmax=wmax)
        axes[0,2].set_title("X Weight Pred")
        fig.colorbar(im1, ax=axes[0,2], fraction=0.046, pad=0.04)

        axes[0,3].imshow(img_i)
        im2 = axes[0,3].imshow(weight_pred[1], cmap='jet', alpha=0.7, vmin=0, vmax=wmax)
        axes[0,3].set_title("Y Weight Pred")
        fig.colorbar(im2, ax=axes[0,3], fraction=0.046, pad=0.04)


        uncer_pred_i = self.uncertainties[i]
        uncer_rescaled = torch.clamp(45.0 * uncer_pred_i - 35.0, min=0.1)
        mask_i = torch.clamp(1.0 / uncer_rescaled, min=0.0, max=1.0)
        im9 = axes[1,0].imshow(mask_i.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[1,0].set_title("Optimized Dynamic Mask of Frame i")

        # vis DINO feature similarity at (2, 2)
        im8 = axes[1,1].imshow(dino_feats_similarity_reproj, cmap='jet', vmin=0, vmax=1)
        axes[1,1].set_title("DINO Feature Similarity Between i and j")
        fig.colorbar(im8, ax=axes[1,1], fraction=0.046, pad=0.04)

        # save dino feats similarity
        os.makedirs(f"{self.output}/intermediate_results", exist_ok=True)
        dino_feats_similarity_reproj_vis = plt.get_cmap('viridis')(dino_feats_similarity_reproj)
        Image.fromarray((dino_feats_similarity_reproj_vis * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/dino_feats_similarity_reproj_{i:03d}_vs_{j:03d}.png")

        def visualize_dino_feature_pca(features, save_path=None, scale_each=False):
            """
            Visualize DINO features (C, H, W) or (B, C, H, W) as RGB image using PCA projection.
            Args:
                features: torch.Tensor or np.ndarray, shape [C,H,W] or [B,C,H,W]
                save_path: optional, if provided, save the RGB visualization
                scale_each: if True, scale per image when B>1
            """
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()

            # Handle batch
            if features.ndim == 4:
                feats_list = []
                for f in features:
                    feats_list.append(_pca_to_rgb_single(f, scale_each))
                vis = np.concatenate(feats_list, axis=1)  # horizontal concat
            elif features.ndim == 3:
                vis = _pca_to_rgb_single(features, scale_each)
            else:
                raise ValueError("Feature tensor must be [C,H,W] or [B,C,H,W]")

            plt.figure(figsize=(6,6))
            plt.imshow(vis)
            plt.axis('off')
            plt.title("DINO feature PCA→RGB projection")
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            # plt.show()
            return vis


        def _pca_to_rgb_single(feature, scale_each=False):
            """Helper: apply PCA→RGB"""
            C, H, W = feature.shape
            feat = feature.reshape(C, -1).T  # [HW, C]
            feat -= feat.mean(0, keepdims=True)

            pca = PCA(n_components=3)
            feat_rgb = pca.fit_transform(feat)  # [HW, 3]
            feat_rgb = feat_rgb.reshape(H, W, 3)

            # Normalize to [0,1]
            if scale_each:
                for i in range(3):
                    feat_rgb[..., i] = (feat_rgb[..., i] - feat_rgb[..., i].min()) / (feat_rgb[..., i].max() - feat_rgb[..., i].min() + 1e-6)
            else:
                feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min() + 1e-6)

            return feat_rgb

        dino_feats_i_vis = visualize_dino_feature_pca(dino_feats_i)
        dino_feats_j_vis = visualize_dino_feature_pca(dino_feats_j)
        # from [H, W, C] to [C, H, W] for numpy array.
        dino_feats_i_vis = dino_feats_i_vis.transpose(2,0,1)
        dino_feats_j_vis = dino_feats_j_vis.transpose(2,0,1)
        dino_feats_i_vis_upsampled = F.interpolate(torch.from_numpy(dino_feats_i_vis).unsqueeze(0), size=(self.ht, self.wd), mode='bilinear', align_corners=False)
        dino_feats_j_vis_upsampled = F.interpolate(torch.from_numpy(dino_feats_j_vis).unsqueeze(0), size=(self.ht, self.wd), mode='bilinear', align_corners=False)
        # from [C, H, W] to [H, W, C]
        dino_feats_i_vis_upsampled = dino_feats_i_vis_upsampled.squeeze(0).cpu().numpy().transpose(1,2,0)
        dino_feats_j_vis_upsampled = dino_feats_j_vis_upsampled.squeeze(0).cpu().numpy().transpose(1,2,0)
        Image.fromarray((dino_feats_i_vis_upsampled * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/dino_feats_i_{i:03d}.png")
        Image.fromarray((dino_feats_j_vis_upsampled * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/dino_feats_j_{j:03d}.png")

        # save high-resolution mono disparity
        mono_disp_i_high_res = self.mono_disps_up[i]
        mono_disp_j_high_res = self.mono_disps_up[j]
        mono_disp_i_high_res = torch.clamp(mono_disp_i_high_res, min=0.1, max=1)
        i_min = mono_disp_i_high_res.min()
        i_max = mono_disp_i_high_res.max()
        mono_disp_i_high_res = (mono_disp_i_high_res - i_min) / (i_max - i_min)
        mono_disp_j_high_res = torch.clamp(mono_disp_j_high_res, min=0.1, max=1)
        j_min = mono_disp_j_high_res.min()
        j_max = mono_disp_j_high_res.max()
        mono_disp_j_high_res = (mono_disp_j_high_res - j_min) / (j_max - j_min)
        mono_disp_i_high_res_vis = plt.get_cmap('viridis')(mono_disp_i_high_res.cpu().numpy())
        mono_disp_j_high_res_vis = plt.get_cmap('viridis')(mono_disp_j_high_res.cpu().numpy())
        Image.fromarray((mono_disp_i_high_res_vis * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/mono_disp_i_{i:03d}.png")
        Image.fromarray((mono_disp_j_high_res_vis * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/mono_disp_j_{j:03d}.png")

        # vis mono disparity map of image i
        im6 = axes[1,2].imshow(mono_disp_i, cmap='viridis')
        axes[1,2].set_title("Mono Disparity of Image i")
        fig.colorbar(im6, ax=axes[1,2], fraction=0.046, pad=0.04)

        # vis depth map of image i at (2, 1)
        im7 = axes[1,3].imshow(disp_i, cmap='viridis')
        axes[1,3].set_title("Optimized Disparity of Image i")
        fig.colorbar(im7, ax=axes[1,3], fraction=0.046, pad=0.04)

        # vis optimized uncertainty of image i with contours
        im_uncer = axes[2,0].imshow(uncer_pred_i.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[2,0].set_title("Optimized Uncertainty of Image i")
        fig.colorbar(im_uncer, ax=axes[2,0], fraction=0.046, pad=0.04)
        contours = axes[2,0].contour(uncer_pred_i.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
        axes[2,0].clabel(contours, inline=True, fontsize=8)
        axes[2,1].imshow(uncer_rescaled.cpu().numpy(), cmap='jet', vmin=0, vmax=10.0)
        axes[2,1].set_title("Rescaled Uncertainty of Frame i")

        # visualize photometric error
        # downsample the image to the same size as the reprojected coordinates
        img_i_downsampled = F.interpolate(self.images[i].unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False)
        img_j_downsampled = F.interpolate(self.images[j].unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False)
        img_j_warp, valid_mask = self.project_images_with_mask(img_j_downsampled.to(self.device), reprojected_coords.resize(1, img_h, img_w, 2))
        photometric_error = (img_i_downsampled.to(self.device) - img_j_warp) * valid_mask
        photometric_error = photometric_error.abs().squeeze(0)
        photometric_error = photometric_error.permute(1,2,0).cpu().numpy()        # [H, W, 3]
        axes[2,2].imshow(photometric_error)
        axes[2,2].set_title("Photometric Error")

        axes[2,3].imshow(img_i)
        im5 = axes[2,3].imshow(reprojection_error_norm, cmap='jet', alpha=0.7)
        mean_reprojection_error_norm = reprojection_error_norm.mean()
        axes[2,3].set_title(f"Reprojection Error Norm (Mean: {mean_reprojection_error_norm:.4f})")
        fig.colorbar(im5, ax=axes[2,3], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.axis("off")

        fig.suptitle(f"Keyframe {i:03d} vs {j:03d}", fontsize=16)

        os.makedirs(f"{self.output}/intermediate_results", exist_ok=True)
        fig.savefig(f"{self.output}/intermediate_results/ts_{int(self.timestamp[i]):05d}_kf_{i:03d}_vs_kf_{j:03d}_{mode}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def visualize_all_opt_params(self, out_directory=None, iteration="final"):
        """ 
        visualize the uncertainty before and after optimization, disparity map
        """

        plot_dir = os.path.join(out_directory, "plots_" + iteration)
        # add tqdm progress bar
        for idx in tqdm(range(self.counter.value), desc="Visualizing all optimized parameters", total=self.counter.value):
            img_i = self.images[idx].permute(1,2,0).cpu().numpy()

            uncer_pred = self.uncertainties[idx]
            uncer_rescaled = torch.clamp(45.0 * uncer_pred - 35.0, min=0.1)

            mask_i = torch.clamp(1.0 / uncer_rescaled, min=0.0, max=1.0)

            # Create the figure WITH constrained layout
            plot_cols = 4
            plot_rows = 2
            fig_height = plot_rows * 4 + 0.5    # leave space for the title
            fig_width = plot_cols * 4
            aspect_ratio = img_i.shape[1] / img_i.shape[0]
            fig_width = fig_width * aspect_ratio
            fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(fig_width, fig_height), constrained_layout=True)

            # vis image i
            axes[0,0].imshow(img_i)
            axes[0,0].set_title("Input Image i")

            # visualize uncertainty heatmap with contours at (row 1, col 0)
            im_unc = axes[1,0].imshow(uncer_pred.cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)
            contours = axes[1,0].contour(uncer_pred.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
            axes[1,0].clabel(contours, inline=True, fontsize=8)
            axes[1,0].set_title("Uncertainty")

            # rescaled uncertainty
            axes[0,1].imshow(uncer_rescaled.cpu().numpy(), cmap='jet', vmin=0, vmax=10.0)
            axes[0,1].set_title("Rescaled Uncertainty")

            # visualize high-resolution scaled uncertainty
            # uncer_rescaled_high_res = self.upsample_weight(uncer_rescaled.unsqueeze(0))
            uncer_pred_high_res = self.upsample_weight(uncer_pred.unsqueeze(0))
            uncer_rescaled_high_res = torch.clamp(45.0 * uncer_pred_high_res - 35.0, min=0.1)
            axes[1,1].imshow(uncer_rescaled_high_res.cpu().numpy(), cmap='jet', vmin=0, vmax=10.0)
            axes[1,1].set_title("High-Resolution Scaled Uncertainty")

            # vis mono disparity map of image i
            mono_disp = self.mono_disps[idx].cpu().numpy()
            vmax = min(mono_disp.max().item(), 5.0)
            im2 = axes[0,2].imshow(mono_disp, cmap='viridis', vmin=0, vmax=vmax)
            axes[0,2].set_title("Mono Disparity")

            # visualize high-resolution mono disparity
            mono_disp_high_res = self.mono_disps_up[idx]
            vmax = min(mono_disp_high_res.max().item(), 5.0)
            axes[1,2].imshow(mono_disp_high_res.cpu().numpy(), cmap='viridis', vmin=0, vmax=vmax)
            axes[1,2].set_title("High-Resolution Mono Disparity")

            # vis depth map of image i at (2, 1)
            droid_disp = self.disps[idx].cpu().numpy()
            vmax = min(droid_disp.max().item(), 5.0)
            im3 = axes[0,3].imshow(droid_disp, cmap='viridis', vmin=0, vmax=vmax)
            axes[0,3].set_title("Optimized Disparity")

            # visualize high-resolution optimized disparity
            droid_disp_high_res = self.disps_up[idx]
            vmax = min(droid_disp_high_res.max().item(), 5.0)
            axes[1,3].imshow(droid_disp_high_res.cpu().numpy(), cmap='viridis', vmin=0, vmax=vmax)
            axes[1,3].set_title("High-Resolution Optimized Disparity")

            for ax in axes.ravel():
                ax.axis("off")

            fig.suptitle(f"Keyframe idx {idx:03d}, Frame idx {int(self.timestamp[idx]):05d}", fontsize=20)

            os.makedirs(f"{plot_dir}", exist_ok=True)
            fig.savefig(f"{plot_dir}/video_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # save the input image, scaled uncertainty, uncertainty with contours, and mask individually
            # Create separate directories for each image type
            input_dir = os.path.join(plot_dir, "input_images")
            uncer_dir = os.path.join(plot_dir, "scaled_uncertainty")
            uncer_contour_dir = os.path.join(plot_dir, "uncertainty_contours")
            high_res_uncer_dir = os.path.join(plot_dir, "high_res_uncertainty")
            
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(uncer_dir, exist_ok=True)
            os.makedirs(uncer_contour_dir, exist_ok=True)
            os.makedirs(high_res_uncer_dir, exist_ok=True)

            def color_map(tensor, cmap='jet', vmin=0, vmax=1):
                return (plt.get_cmap(cmap)(tensor.cpu().numpy() / vmax)[:, :, :3] * 255.0).astype(np.uint8)
            
            # Save input image
            Image.fromarray((img_i * 255.0).astype(np.uint8)).save(f"{input_dir}/input_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png")

            # Save scaled uncertainty
            uncer_rescaled_colored = color_map(uncer_rescaled, vmax=10.0)
            Image.fromarray(uncer_rescaled_colored).save(f"{uncer_dir}/uncertainty_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png")
            
            # Save uncertainty with contours
            fig_contour, ax_contour = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            ax_contour.imshow(uncer_pred.cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)
            contours = ax_contour.contour(uncer_pred.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
            ax_contour.clabel(contours, inline=True, fontsize=8)
            ax_contour.axis("off")
            ax_contour.set_position([0, 0, 1, 1])  # Remove margins
            fig_contour.savefig(f"{uncer_contour_dir}/uncertainty_contour_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png", dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig_contour)
            
            # Save high-resolution scaled uncertainty
            uncer_rescaled_high_res_colored = color_map(uncer_rescaled_high_res, vmax=10.0)
            Image.fromarray(uncer_rescaled_high_res_colored).save(f"{high_res_uncer_dir}/high_res_uncertainty_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png")
            
        # Create gif
        create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)

    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            if self.metric_depth_reg:
                est_disp = self.mono_disps_up[index].clone().to(device)  # [h, w]
                est_depth = torch.where(est_disp>0.0, 1.0 / (est_disp), 0.0)
                depth_mask = torch.ones_like(est_disp,dtype=torch.bool).to(device)
                c2w = self.get_pose(index,device)
            else:
                est_disp = self.disps_up[index].clone().to(device)  # [h, w]
                est_depth = 1.0 / (est_disp)
                depth_mask = self.valid_depth_mask[index].clone().to(device)
                c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            poses.append(pose)
        poses = torch.stack(poses,dim=0).numpy()

        timestamps = self.timestamp[:self.counter.value].cpu().numpy()
        images = self.images[:self.counter.value].cpu().numpy()
        tum_poses = self.poses[:self.counter.value].cpu().numpy()
        mono_disps = self.mono_disps_up[:self.counter.value].cpu().numpy()
        droid_disps_up = self.disps_up[:self.counter.value].cpu().numpy()
        droid_disps = self.disps[:self.counter.value].cpu().numpy()
        intrinsics = self.intrinsics[:self.counter.value].cpu().numpy()
        uncertainties = self.uncertainties[:self.counter.value].cpu().numpy()
        omega_tokens = self.omega_tokens[:self.counter.value].cpu().numpy()
        omega_token_valid = self.omega_token_valid[:self.counter.value].cpu().numpy()
        np.savez(path,
            timestamps=timestamps,
            images=images,
            poses=poses,tum_poses=tum_poses,
            mono_disps=mono_disps,
            droid_disps_up=droid_disps_up,
            droid_disps=droid_disps,
            intrinsics=intrinsics,
            uncertainties=uncertainties,
            omega_tokens=omega_tokens,
            omega_token_valid=omega_token_valid)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)

    def save_poses(self,path:str):
        poses = []
        timestamps = []
        for i in range(self.counter.value):
            _, _, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            timestamps.append(timestamp)
        poses = torch.stack(poses,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy()   
        np.savez(path,poses=poses, timestamps=timestamps)
        self.printer.print(f"Saved poses and timestamp: {path}", FontColor.INFO)

    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        """This is from splat-slam, not used in WildGS-SLAM
        """
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()
