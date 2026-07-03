import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.datasets import load_metric_depth, load_img_feature
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor
from src.utils.omega_prior import OmegaPriorCache
from src.utils.omega_predictor import OmegaOnlinePredictor

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

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        
        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        self.omega_prior = OmegaPriorCache(cfg, device)
        self.omega_predictor = OmegaOnlinePredictor(cfg, device)
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

    def _apply_omega_prior(self, tstamp, mono_depth, image):
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
        if source in ['model', 'online'] and (depth_enabled or uncertainty_enabled):
            omega_depth, omega_confidence = self.omega_predictor.predict_frame(image)
            if uncertainty_enabled:
                omega_uncertainty = self.omega_prior.confidence_to_uncertainty(omega_confidence)
        else:
            omega_depth, omega_uncertainty = self.omega_prior.load_for_frame(int(tstamp), image_hw)
            if not depth_enabled:
                omega_depth = None
            if not uncertainty_enabled:
                omega_uncertainty = None

        if depth_enabled and omega_depth is not None:
            mode = self.omega_prior.depth_cfg.get("mode", "replace")
            if mode == "blend":
                alpha = float(self.omega_prior.depth_cfg.get("blend_alpha", 1.0))
                valid = omega_depth > 0
                mono_depth = torch.where(valid, alpha * omega_depth + (1.0 - alpha) * mono_depth, mono_depth)
            else:
                mono_depth = torch.where(omega_depth > 0, omega_depth, mono_depth)

        return mono_depth, omega_uncertainty

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None):
        """ main update operation - run on every frame in video """

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
            mono_depth, omega_uncertainty = self._apply_omega_prior(tstamp, mono_depth, image)
            if self.uncertainty_aware:
                dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=self.cfg['mono_prior']['save_feature'])
            else:
                dino_features = None
                if self.cfg['mapping']["uncertainty_params"]['activate']:
                    # If mapping needs dino features, we predict here and store the value in local disk
                    _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=True)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0], dino_features, omega_uncertainty)
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
                mono_depth, omega_uncertainty = self._apply_omega_prior(tstamp, mono_depth, image)
                if self.uncertainty_aware:
                    dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=self.cfg['mono_prior']['save_feature'])
                else:
                    dino_features = None
                    if self.cfg['mapping']["uncertainty_params"]['activate']:
                        # if mapping needs dino features, we predict here and store the value in local disk
                        _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=True)
                # add new frame to video, all params
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0], dino_features, omega_uncertainty)     # video.counter += 1
                # gmap: torch.Size([1, 128, 45, 80]) net[0]: [128, 45, 80] inp: [1, 128, 45, 80], dino_features: [25, 45, 384]
            else:
                self.count += 1

        return force_to_add_keyframe

    @torch.no_grad()
    def get_img_feature(self, tstamp, image, suffix=''):
        dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,suffix=suffix,save_feat=self.cfg['mono_prior']['save_feature'])
        return dino_features
