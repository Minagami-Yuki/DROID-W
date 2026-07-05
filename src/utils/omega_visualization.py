import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


class OmegaUncertaintyVisualizer:
    """Save periodic color visualizations of VGGT-Omega uncertainty maps."""

    def __init__(self, cfg, save_dir: str):
        omega_cfg = cfg.get("omega_prior", {}) or {}
        self.cfg = omega_cfg.get("visualization", {}) or {}
        self.uncertainty_cfg = omega_cfg.get("uncertainty", {}) or {}
        self.enabled = bool(omega_cfg.get("enable", False) and self.cfg.get("enable", False))
        self.interval = int(self.cfg.get("interval", 100))
        self.save_first = bool(self.cfg.get("save_first", True))
        self.output_dir = self.cfg.get("output_dir") or os.path.join(save_dir, "omega_uncertainty_vis")
        self.colormap = self.cfg.get("colormap", "magma")
        self.save_raw_npy = bool(self.cfg.get("save_raw_npy", False))

    def should_save(self, frame_idx: int) -> bool:
        if not self.enabled or self.interval <= 0:
            return False
        frame_idx = int(frame_idx)
        if frame_idx == 0:
            return self.save_first
        return frame_idx % self.interval == 0

    def save(
        self,
        uncertainty: Optional[torch.Tensor],
        frame_idx: int,
        timestamp=None,
        keyframe_idx: Optional[int] = None,
    ) -> Optional[str]:
        if uncertainty is None or not self.should_save(frame_idx):
            return None

        uncertainty_np = self._to_numpy_2d(uncertainty)
        if uncertainty_np is None:
            return None

        os.makedirs(self.output_dir, exist_ok=True)
        stem = self._filename_stem(frame_idx, timestamp, keyframe_idx)
        image_path = os.path.join(self.output_dir, f"{stem}.png")
        Image.fromarray(self._colorize(uncertainty_np)).save(image_path)

        if self.save_raw_npy:
            np.save(os.path.join(self.output_dir, f"{stem}.npy"), uncertainty_np.astype(np.float32))

        return image_path

    def _to_numpy_2d(self, uncertainty: torch.Tensor) -> Optional[np.ndarray]:
        uncertainty = uncertainty.detach().float().squeeze()
        if uncertainty.ndim != 2:
            return None
        return uncertainty.cpu().numpy()

    def _colorize(self, uncertainty: np.ndarray) -> np.ndarray:
        finite = np.isfinite(uncertainty)
        if not finite.any():
            return np.zeros((*uncertainty.shape, 3), dtype=np.uint8)

        vmin = self.cfg.get("min_value", None)
        vmax = self.cfg.get("max_value", None)
        if vmin is None:
            vmin = self.uncertainty_cfg.get("certain_value", None)
        if vmax is None:
            vmax = self.uncertainty_cfg.get("uncertain_value", None)

        if vmin is None or vmax is None or float(vmax) <= float(vmin):
            valid = uncertainty[finite]
            vmin = float(valid.min())
            vmax = float(valid.max())
            if vmax <= vmin:
                vmax = vmin + 1e-6
        else:
            vmin = float(vmin)
            vmax = float(vmax)

        normalized = np.clip((uncertainty - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        normalized = np.where(finite, normalized, 0.0)
        colored_bgr = cv2.applyColorMap(
            (normalized * 255.0).astype(np.uint8),
            self._opencv_colormap(),
        )
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        colored_rgb[~finite] = 0
        return colored_rgb.astype(np.uint8)

    def _opencv_colormap(self):
        colormap = str(self.colormap).lower()
        table = {
            "magma": cv2.COLORMAP_MAGMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "plasma": cv2.COLORMAP_PLASMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "turbo": cv2.COLORMAP_TURBO,
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
        }
        return table.get(colormap, cv2.COLORMAP_MAGMA)

    def _filename_stem(self, frame_idx: int, timestamp, keyframe_idx: Optional[int]) -> str:
        parts = [f"frame_{int(frame_idx):06d}"]
        if timestamp is not None:
            parts.append(f"ts_{self._format_token(timestamp)}")
        if keyframe_idx is not None:
            parts.append(f"kf_{int(keyframe_idx):06d}")
        return "_".join(parts)

    def _format_token(self, value) -> str:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value).replace(os.sep, "_")
        if value.is_integer():
            return f"{int(value):06d}"
        return f"{value:.6f}".replace(".", "p").replace("-", "m")
