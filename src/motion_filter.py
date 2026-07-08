import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.datasets import load_metric_depth, load_img_feature
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor
from src.utils.omega_prior import OmegaPriorCache
from src.utils.omega_predictor import OmegaOnlinePredictor
from src.utils.omega_visualization import OmegaUncertaintyVisualizer

class MotionFilter:
    """ This class is used to filter incoming frames and extract features 
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0
        self.input_frame_counter = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        self.omega_prior = OmegaPriorCache(cfg, device)
        self.omega_predictor = OmegaOnlinePredictor(cfg, device)
        self.omega_uncertainty_visualizer = OmegaUncertaintyVisualizer(cfg, self.save_dir)
        self.use_metric_depth_estimator = self._should_use_metric_depth_estimator()
        self.metric_depth_estimator = get_metric_depth_estimator(cfg) if self.use_metric_depth_estimator else None
        if cfg['mapping']["uncertainty_params"]['activate']:
            # If mapping needs dino features, we still need feature extractor
            self.feat_extractor = get_feature_extractor(cfg)

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    def _should_use_metric_depth_estimator(self):
        omega_cfg = self.cfg.get('omega_prior', {}) or {}
        depth_cfg = omega_cfg.get('depth', {}) or {}
        if not omega_cfg.get('enable', False) or not depth_cfg.get('enable', False):
            return True
        if depth_cfg.get('mode', 'replace') != 'replace':
            return True
        return bool(depth_cfg.get('fallback_to_mono', False))

    def _predict_depth_prior(self, tstamp, image):
        if self.use_metric_depth_estimator:
            return predict_metric_depth(
                self.metric_depth_estimator,
                tstamp,
                image,
                self.cfg,
                self.device,
                save_depth=(self.cfg['mono_prior']['save_depth'] or self.cfg['mapping']["enable"]),
            )
        return torch.zeros(image.shape[-2:], device=self.device, dtype=torch.float32)

    def _apply_omega_prior(self, tstamp, mono_depth, image, frame_idx=None, keyframe_idx=None):
        image_hw = tuple(image.shape[-2:])
        omega_cfg = self.cfg.get('omega_prior', {}) or {}
        depth_cfg = omega_cfg.get('depth', {}) or {}
        uncertainty_cfg = omega_cfg.get('uncertainty', {}) or {}
        depth_enabled = bool(omega_cfg.get('enable', False) and depth_cfg.get('enable', False))
        uncertainty_enabled = bool(omega_cfg.get('enable', False) and uncertainty_cfg.get('enable', False))
        source = omega_cfg.get('source', 'cache')
        if source == 'auto':
            source = 'cache' if omega_cfg.get('cache_dir') else 'model'

        omega_depth = None
        omega_uncertainty = None
        omega_tokens = None
        omega_patch_map = None
        tokens_enabled = self._omega_tokens_enabled()
        patch_tokens_enabled = self._omega_patch_tokens_enabled()
        cache_cfg = omega_cfg.get("cache", {}) or {}
        cache_write_enabled = bool(cache_cfg.get("write", False))
        patch_model_cfg = (omega_cfg.get("model", {}) or {}).get("patch_tokens", {}) or {}
        tokens_needed = tokens_enabled or (cache_write_enabled and bool(cache_cfg.get("save_tokens", True)))
        patch_tokens_needed = patch_tokens_enabled or (
            cache_write_enabled
            and bool(cache_cfg.get("save_patch_tokens", True))
            and bool(patch_model_cfg.get("enable", False))
        )
        if source in ['model', 'online'] and (depth_enabled or uncertainty_enabled or tokens_needed or patch_tokens_needed or cache_write_enabled):
            if tokens_needed and patch_tokens_needed:
                omega_depth, omega_confidence, omega_tokens, omega_patch_map = self.omega_predictor.predict_frame(
                    image,
                    return_tokens=True,
                    return_patch_tokens=True,
                )
            elif tokens_needed:
                omega_depth, omega_confidence, omega_tokens = self.omega_predictor.predict_frame(image, return_tokens=True)
            elif patch_tokens_needed:
                omega_depth, omega_confidence, omega_patch_map = self.omega_predictor.predict_frame(image, return_patch_tokens=True)
            else:
                omega_depth, omega_confidence = self.omega_predictor.predict_frame(image)
            if uncertainty_enabled or bool(cache_cfg.get("save_uncertainty", True)):
                omega_uncertainty = self.omega_prior.confidence_to_uncertainty(omega_confidence)
            self.omega_prior.save_for_frame(
                int(tstamp),
                depth=omega_depth,
                confidence=omega_confidence,
                uncertainty=omega_uncertainty,
                tokens=omega_tokens,
                patch_map=omega_patch_map,
            )
            if not uncertainty_enabled:
                omega_uncertainty = None
            if not tokens_enabled:
                omega_tokens = None
            if not patch_tokens_enabled:
                omega_patch_map = None
        else:
            omega_depth, omega_uncertainty = self.omega_prior.load_for_frame(int(tstamp), image_hw)
            if tokens_enabled:
                omega_tokens = self.omega_prior.load_tokens_for_frame(int(tstamp))
            if patch_tokens_enabled:
                omega_patch_map = self.omega_prior.load_patch_map_for_frame(int(tstamp))
            if not depth_enabled:
                omega_depth = None
            if not uncertainty_enabled:
                omega_uncertainty = None

        if depth_enabled and omega_depth is not None:
            mode = self.omega_prior.depth_cfg.get("mode", "replace")
            if mode == "blend":
                omega_depth = self._align_omega_depth_to_mono(omega_depth, mono_depth)
                alpha = float(self.omega_prior.depth_cfg.get("blend_alpha", 1.0))
                valid = omega_depth > 0
                mono_depth = torch.where(valid, alpha * omega_depth + (1.0 - alpha) * mono_depth, mono_depth)
            else:
                mono_depth = torch.where(omega_depth > 0, omega_depth, mono_depth)

        self._save_omega_uncertainty_visual(frame_idx, tstamp, omega_uncertainty, keyframe_idx)
        return mono_depth, omega_uncertainty, omega_tokens, omega_patch_map

    def _omega_tokens_enabled(self):
        edge_cfg = self.cfg.get("edge_dtf_prior", {}) or {}
        token_cfg = edge_cfg.get("token_calibration", {}) or {}
        suppress_cfg = edge_cfg.get("token_dynamic_suppression", {}) or {}
        spatial_cfg = edge_cfg.get("token_spatial_suppression", {}) or {}
        covariance_cfg = edge_cfg.get("per_edge_covariance", {}) or {}
        return bool(
            edge_cfg.get("enable", False)
            and (
                token_cfg.get("enable", False)
                or suppress_cfg.get("enable", False)
                or spatial_cfg.get("enable", False)
                or (
                    covariance_cfg.get("enable", False)
                    and covariance_cfg.get("use_token_distance", True)
                )
            )
        )

    def _omega_patch_tokens_enabled(self):
        edge_cfg = self.cfg.get("edge_dtf_prior", {}) or {}
        patch_edge_cfg = edge_cfg.get("patch_token_uncertainty", {}) or {}
        patch_model_cfg = ((self.cfg.get("omega_prior", {}) or {}).get("model", {}) or {}).get("patch_tokens", {}) or {}
        return bool(
            edge_cfg.get("enable", False)
            and patch_edge_cfg.get("enable", False)
            and patch_model_cfg.get("enable", False)
        )

    def _omega_source(self):
        omega_cfg = self.cfg.get('omega_prior', {}) or {}
        source = omega_cfg.get('source', 'cache')
        if source == 'auto':
            source = 'cache' if omega_cfg.get('cache_dir') else 'model'
        return source

    def _predict_omega_uncertainty_for_visualization(self, tstamp, image):
        if not self.omega_uncertainty_visualizer.enabled or not self.omega_prior.uncertainty_enabled:
            return None

        image_hw = tuple(image.shape[-2:])
        source = self._omega_source()
        if source in ['model', 'online']:
            _, omega_confidence = self.omega_predictor.predict_frame(image)
            return self.omega_prior.confidence_to_uncertainty(omega_confidence)

        _, omega_uncertainty = self.omega_prior.load_for_frame(int(tstamp), image_hw)
        return omega_uncertainty

    def _save_omega_uncertainty_visual(self, frame_idx, tstamp, omega_uncertainty, keyframe_idx=None):
        if frame_idx is None:
            return
        self.omega_uncertainty_visualizer.save(
            omega_uncertainty,
            frame_idx=int(frame_idx),
            timestamp=tstamp,
            keyframe_idx=keyframe_idx,
        )

    def _align_omega_depth_to_mono(self, omega_depth, mono_depth):
        align_mode = self.omega_prior.depth_cfg.get("align_to_mono", "none")
        if align_mode in [False, None, "none"]:
            return omega_depth

        if align_mode != "scale":
            raise ValueError("omega_prior.depth.align_to_mono currently supports only 'none' or 'scale'")

        min_depth = float(self.omega_prior.depth_cfg.get("min_depth", 1e-4))
        valid = (
            torch.isfinite(omega_depth)
            & torch.isfinite(mono_depth)
            & (omega_depth > min_depth)
            & (mono_depth > min_depth)
        )
        if valid.sum() < int(self.omega_prior.depth_cfg.get("align_min_pixels", 256)):
            return omega_depth

        ratio = mono_depth[valid] / omega_depth[valid]
        ratio = ratio[torch.isfinite(ratio)]
        if ratio.numel() == 0:
            return omega_depth

        trim = float(self.omega_prior.depth_cfg.get("align_trim", 0.05))
        if ratio.numel() > 20 and trim > 0.0:
            lo = torch.quantile(ratio, trim)
            hi = torch.quantile(ratio, 1.0 - trim)
            ratio = ratio[(ratio >= lo) & (ratio <= hi)]
            if ratio.numel() == 0:
                return omega_depth

        scale = torch.median(ratio)
        min_scale = float(self.omega_prior.depth_cfg.get("align_min_scale", 0.2))
        max_scale = float(self.omega_prior.depth_cfg.get("align_max_scale", 5.0))
        scale = torch.clamp(scale, min=min_scale, max=max_scale)
        return omega_depth * scale

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None):
        """ main update operation - run on every frame in video """

        frame_idx = self.input_frame_counter
        self.input_frame_counter += 1

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)       # [1, 128, 45, 80]

        force_to_add_keyframe = False

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            mono_depth = self._predict_depth_prior(tstamp, image)
            mono_depth, omega_uncertainty, omega_tokens, omega_patch_map = self._apply_omega_prior(
                tstamp, mono_depth, image, frame_idx=frame_idx, keyframe_idx=self.video.counter.value)
            if self.uncertainty_aware:
                dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=self.cfg['mono_prior']['save_feature'])
            else:
                dino_features = None
                if self.cfg['mapping']["uncertainty_params"]['activate']:
                    # If mapping needs dino features, we predict here and store the value in local disk
                    _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=True)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0], dino_features, omega_uncertainty, omega_tokens, omega_patch_map)
        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            if self.cfg['tracking']['force_keyframe_every_n_frames'] > 0:
                # Actually, tstamp is the frame idx
                last_tstamp = self.video.timestamp[self.video.counter.value-1]
                force_to_add_keyframe = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']


            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or force_to_add_keyframe:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                mono_depth = self._predict_depth_prior(tstamp, image)
                mono_depth, omega_uncertainty, omega_tokens, omega_patch_map = self._apply_omega_prior(
                    tstamp, mono_depth, image, frame_idx=frame_idx, keyframe_idx=self.video.counter.value)
                if self.uncertainty_aware:
                    dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=self.cfg['mono_prior']['save_feature'])
                else:
                    dino_features = None
                    if self.cfg['mapping']["uncertainty_params"]['activate']:
                        # if mapping needs dino features, we predict here and store the value in local disk
                        _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=True)
                # add new frame to video, all params
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0], dino_features, omega_uncertainty, omega_tokens, omega_patch_map)     # video.counter += 1
                # gmap: torch.Size([1, 128, 45, 80]) net[0]: [128, 45, 80] inp: [1, 128, 45, 80], dino_features: [25, 45, 384]
            else:
                self.count += 1
                if self.omega_uncertainty_visualizer.should_save(frame_idx):
                    omega_uncertainty = self._predict_omega_uncertainty_for_visualization(tstamp, image)
                    self._save_omega_uncertainty_visual(
                        frame_idx, tstamp, omega_uncertainty, keyframe_idx=max(self.video.counter.value - 1, 0))

        return force_to_add_keyframe

    @torch.no_grad()
    def get_img_feature(self, tstamp, image, suffix=''):
        dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,suffix=suffix,save_feat=self.cfg['mono_prior']['save_feature'])
        return dino_features
