import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class OmegaOnlinePredictor:
    """Thin runtime wrapper around facebookresearch/vggt-omega."""

    def __init__(self, cfg: Dict, device: str):
        self.cfg = cfg.get("omega_prior", {}) or {}
        self.model_cfg = self.cfg.get("model", {}) or {}
        self.device = self.model_cfg.get("device") or device
        self.repo_path = self.model_cfg.get("repo_path", "thirdparty/vggt-omega")
        self.checkpoint = self.model_cfg.get("checkpoint", "/data1/czy/Output/DROID-omega/vggt_omega_1b_512.pt")
        self.image_resolution = int(self.model_cfg.get("image_resolution", 512))
        self.preprocess_mode = self.model_cfg.get("preprocess_mode", "balanced")
        self.patch_size = int(self.model_cfg.get("patch_size", 16))
        self.model = None

    def enabled(self) -> bool:
        return bool(self.cfg.get("enable", False)) and self.cfg.get("source", "cache") in ["model", "online"]

    def predict_frame(self, image: torch.Tensor, return_tokens: bool = False, return_patch_tokens: bool = False):
        """Predict metric depth and confidence for one DROID-W frame.

        Args:
            image: Tensor in [0, 1], shape [1, 3, H, W] or [3, H, W].

        Returns:
            depth: [H, W] float tensor on self.device.
            confidence: [H, W] float tensor on self.device.
            tokens: optional [T, C] camera/register token tensor when return_tokens is True.
            patch_map: optional [D, H_patch, W_patch] tensor when return_patch_tokens is True.
        """

        if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            raise RuntimeError("VGGT-Omega online prediction requires a CUDA device.")

        self._ensure_model()

        image = image.detach().float()
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [3,H,W] or [1,3,H,W], got {tuple(image.shape)}")

        _, src_h, src_w = image.shape
        image_in = self._preprocess_image(image.to(self.device))

        with torch.inference_mode():
            predictions = self.model(image_in)

        depth = predictions["depth"][0, 0, ..., 0].float()
        confidence = predictions["depth_conf"][0, 0].float()

        depth = F.interpolate(depth[None, None], size=(src_h, src_w), mode="bilinear", align_corners=False)[0, 0]
        confidence = F.interpolate(confidence[None, None], size=(src_h, src_w), mode="bilinear", align_corners=False)[0, 0]
        tokens = None
        if return_tokens:
            tokens = predictions.get("camera_and_register_tokens")
            if tokens is not None:
                tokens = tokens[0, 0].float().detach()
        patch_map = None
        if return_patch_tokens:
            patch_map = self._project_patch_tokens(predictions)

        if return_tokens and return_patch_tokens:
            return depth, confidence, tokens, patch_map
        if return_tokens:
            return depth, confidence, tokens
        if return_patch_tokens:
            return depth, confidence, patch_map
        return depth, confidence

    def _project_patch_tokens(self, predictions: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        patch_cfg = self.model_cfg.get("patch_tokens", {}) or {}
        patch_tokens = predictions.get("patch_tokens")
        patch_grid_size = predictions.get("patch_grid_size")
        if patch_tokens is None or patch_grid_size is None:
            return None

        tokens = patch_tokens[0, 0].float().detach()
        if tokens.ndim != 2:
            return None

        grid_h = int(patch_grid_size[0].item())
        grid_w = int(patch_grid_size[1].item())
        if grid_h * grid_w != tokens.shape[0]:
            return None

        dim = int(patch_cfg.get("dim", 8))
        if dim <= 0:
            return None

        projection = patch_cfg.get("projection", "group_mean")
        if projection == "first":
            if tokens.shape[1] < dim:
                projected = F.pad(tokens, (0, dim - tokens.shape[1]))
            else:
                projected = tokens[:, :dim]
        elif projection == "group_mean":
            pad = (-tokens.shape[1]) % dim
            if pad:
                tokens = F.pad(tokens, (0, pad))
            projected = tokens.reshape(tokens.shape[0], dim, -1).mean(dim=-1)
        else:
            raise ValueError("omega_prior.model.patch_tokens.projection must be 'group_mean' or 'first'")

        if bool(patch_cfg.get("normalize", True)):
            projected = F.normalize(projected, p=2, dim=-1, eps=1e-6)

        return projected.reshape(grid_h, grid_w, dim).permute(2, 0, 1).contiguous()

    def _ensure_model(self):
        if self.model is not None:
            return

        repo_path = os.path.abspath(self.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from vggt_omega.models import VGGTOmega

        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {self.checkpoint}")

        model = VGGTOmega().eval()
        state_dict = torch.load(self.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        self.model = model.to(self.device)

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        target_h, target_w = self._target_shape(height, width)
        image = F.interpolate(
            image[None],
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        return image

    def _target_shape(self, height: int, width: int) -> Tuple[int, int]:
        if self.image_resolution <= 0:
            raise ValueError("omega_prior.model.image_resolution must be positive")
        if self.image_resolution % self.patch_size != 0:
            raise ValueError("omega_prior.model.image_resolution must be divisible by patch_size")

        aspect_ratio = height / max(width, 1)
        if self.preprocess_mode == "balanced":
            token_number = (self.image_resolution // self.patch_size) ** 2
            width_patches = (token_number / aspect_ratio) ** 0.5
            height_patches = token_number / width_patches
            width_patches = max(1, int(round(width_patches)))
            height_patches = max(1, int(round(height_patches)))
            return height_patches * self.patch_size, width_patches * self.patch_size

        if self.preprocess_mode == "max_size":
            if aspect_ratio >= 1.0:
                target_h = self.image_resolution
                target_w = self._round_to_patch_multiple(self.image_resolution / aspect_ratio)
            else:
                target_w = self.image_resolution
                target_h = self._round_to_patch_multiple(self.image_resolution * aspect_ratio)
            return target_h, target_w

        raise ValueError("omega_prior.model.preprocess_mode must be 'balanced' or 'max_size'")

    def _round_to_patch_multiple(self, value: float) -> int:
        return max(self.patch_size, int(round(float(value) / self.patch_size)) * self.patch_size)
