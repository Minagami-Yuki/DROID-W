import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class OmegaPriorCache:
    """Read precomputed VGGT/VGGT-Omega priors without changing baseline flow."""

    def __init__(self, cfg: Dict, device: str):
        self.cfg = cfg.get("omega_prior", {}) or {}
        self.device = device
        self.cache_dir = self.cfg.get("cache_dir")
        self.enabled = bool(self.cfg.get("enable", False))
        self.depth_cfg = self.cfg.get("depth", {}) or {}
        self.uncertainty_cfg = self.cfg.get("uncertainty", {}) or {}
        self._warned = set()

    @property
    def depth_enabled(self) -> bool:
        return self.enabled and bool(self.depth_cfg.get("enable", False))

    @property
    def uncertainty_enabled(self) -> bool:
        return self.enabled and bool(self.uncertainty_cfg.get("enable", False))

    def load_for_frame(
        self,
        frame_idx: int,
        image_hw: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (depth, uncertainty) tensors resized to image_hw, or None."""

        if not self.enabled:
            return None, None

        if not self.cache_dir:
            self._warn_once("missing_cache_dir", "Omega prior enabled but cache_dir is empty; falling back to baseline priors.")
            return None, None

        depth = self._load_depth(frame_idx, image_hw) if self.depth_enabled else None
        uncertainty = self._load_uncertainty(frame_idx, image_hw) if self.uncertainty_enabled else None
        return depth, uncertainty

    def _load_depth(self, frame_idx: int, image_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        path = self._find_prior_file(frame_idx, "depth")
        if path is None:
            self._handle_missing("depth", frame_idx)
            return None

        depth = self._load_array(path)
        depth = self._resize_2d(depth, image_hw, mode="bilinear")
        depth = depth.to(self.device, dtype=torch.float32)
        depth = depth * float(self.depth_cfg.get("scale", 1.0))

        min_depth = float(self.depth_cfg.get("min_depth", 1e-4))
        max_depth = self.depth_cfg.get("max_depth", None)
        depth = torch.where(torch.isfinite(depth) & (depth > min_depth), depth, torch.zeros_like(depth))
        if max_depth is not None:
            depth = torch.where(depth <= float(max_depth), depth, torch.zeros_like(depth))
        return depth

    def _load_uncertainty(self, frame_idx: int, image_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        conf_path = self._find_prior_file(frame_idx, "confidence")
        uncer_path = self._find_prior_file(frame_idx, "uncertainty")

        if conf_path is None and uncer_path is None:
            self._handle_missing("confidence/uncertainty", frame_idx)
            return None

        if uncer_path is not None:
            uncertainty = self._load_array(uncer_path)
            uncertainty = self._resize_2d(uncertainty, image_hw, mode="bilinear")
            uncertainty = uncertainty.to(self.device, dtype=torch.float32)
            return self._sanitize_uncertainty(uncertainty)

        confidence = self._load_array(conf_path)
        confidence = self._resize_2d(confidence, image_hw, mode="bilinear")
        confidence = confidence.to(self.device, dtype=torch.float32)
        return self._confidence_to_uncertainty(confidence)

    def edge_weight_from_uncertainty(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Convert DROID-W-style uncertainty to a static weight in [0, 1]."""

        scale = torch.clamp(
            float(self.uncertainty_cfg.get("droid_scale", 45.0)) * uncertainty
            - float(self.uncertainty_cfg.get("droid_shift", 35.0)),
            min=float(self.uncertainty_cfg.get("droid_min_scale", 0.1)),
        )
        return torch.clamp(1.0 / scale, 0.0, 1.0)

    def confidence_to_uncertainty(self, confidence: torch.Tensor) -> torch.Tensor:
        return self._confidence_to_uncertainty(confidence)

    def _confidence_to_uncertainty(self, confidence: torch.Tensor) -> torch.Tensor:
        conf = confidence.float()
        if bool(self.uncertainty_cfg.get("normalize_confidence", True)):
            finite = torch.isfinite(conf)
            if finite.any():
                lo = conf[finite].amin()
                hi = conf[finite].amax()
                conf = (conf - lo) / (hi - lo + 1e-6)

        conf = torch.clamp(conf, 0.0, 1.0)
        min_conf = float(self.uncertainty_cfg.get("min_confidence", 0.0))
        if min_conf > 0.0:
            conf = torch.where(conf >= min_conf, conf, torch.zeros_like(conf))

        certain = float(self.uncertainty_cfg.get("certain_value", 0.78))
        uncertain = float(self.uncertainty_cfg.get("uncertain_value", 1.0))
        uncertainty = certain + (1.0 - conf) * (uncertain - certain)
        return self._sanitize_uncertainty(uncertainty)

    def _sanitize_uncertainty(self, uncertainty: torch.Tensor) -> torch.Tensor:
        fallback = float(self.uncertainty_cfg.get("missing_value", 1.0))
        uncertainty = torch.where(torch.isfinite(uncertainty), uncertainty, torch.full_like(uncertainty, fallback))
        return torch.clamp(
            uncertainty,
            min=float(self.uncertainty_cfg.get("min_value", 0.1)),
            max=float(self.uncertainty_cfg.get("max_value", 10.0)),
        )

    def _find_prior_file(self, frame_idx: int, kind: str) -> Optional[str]:
        names = self._candidate_names(frame_idx, kind)
        subdirs = self._subdirs(kind)

        for subdir in subdirs:
            base = os.path.join(self.cache_dir, subdir) if subdir else self.cache_dir
            for name in names:
                path = os.path.join(base, name)
                if os.path.isfile(path):
                    return path
        return None

    def _candidate_names(self, frame_idx: int, kind: str):
        idx = f"{int(frame_idx):05d}"
        raw_idx = str(int(frame_idx))
        aliases = {
            "depth": ["depth", "omega_depth", "vggt_depth"],
            "confidence": ["confidence", "conf", "omega_confidence", "vggt_confidence"],
            "uncertainty": ["uncertainty", "uncer", "omega_uncertainty"],
        }[kind]
        names = [f"{idx}.npy", f"{raw_idx}.npy"]
        for alias in aliases:
            names.extend([
                f"{idx}_{alias}.npy",
                f"{alias}_{idx}.npy",
                f"{raw_idx}_{alias}.npy",
                f"{alias}_{raw_idx}.npy",
            ])
        return names

    def _subdirs(self, kind: str):
        configured = self.cfg.get(f"{kind}_subdir")
        if configured:
            return [configured, ""]
        if kind == "depth":
            return ["depths", "depth", "omega_depths", ""]
        if kind == "confidence":
            return ["confidences", "confidence", "conf", "omega_confidences", ""]
        return ["uncertainties", "uncertainty", "omega_uncertainties", ""]

    def _load_array(self, path: str) -> torch.Tensor:
        array = np.load(path)
        tensor = torch.from_numpy(np.asarray(array)).float().squeeze()
        if tensor.ndim > 2:
            tensor = tensor[..., 0]
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D Omega prior at {path}, got shape {tuple(tensor.shape)}")
        return tensor

    def _resize_2d(self, tensor: torch.Tensor, image_hw: Tuple[int, int], mode: str) -> torch.Tensor:
        h, w = image_hw
        if tuple(tensor.shape[-2:]) == (h, w):
            return tensor
        return F.interpolate(tensor[None, None], size=(h, w), mode=mode, align_corners=False)[0, 0]

    def _handle_missing(self, kind: str, frame_idx: int) -> None:
        policy = self.cfg.get("missing_policy", "warn")
        message = f"Missing Omega {kind} prior for frame {frame_idx}; falling back to baseline for that frame."
        if policy == "error":
            raise FileNotFoundError(message)
        if policy == "warn":
            self._warn_once(f"missing_{kind}", message)

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[OmegaPrior] {message}")
