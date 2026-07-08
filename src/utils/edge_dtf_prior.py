from typing import Dict, Tuple

import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None


class EdgeDTFPrior:
    """EPO-inspired edge-to-distance-field prior for BA edge reweighting."""

    def __init__(self, cfg: Dict, device: str):
        self.cfg = cfg.get("edge_dtf_prior", {}) or {}
        self.enabled = bool(self.cfg.get("enable", False))
        self.device = device

    def compute_maps(self, image: torch.Tensor, out_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return soft edge map and normalized distance transform at BA resolution."""

        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [3,H,W], got {tuple(image.shape)}")

        image = image.to(self.device, dtype=torch.float32)
        if image.max() > 2.0:
            image = image / 255.0

        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gray = F.interpolate(
            gray[None, None],
            size=out_hw,
            mode="bilinear",
            align_corners=False,
        )

        grad = self._sobel_magnitude(gray)[0, 0]
        finite = torch.isfinite(grad)
        if not finite.any():
            edge = torch.zeros(out_hw, device=self.device, dtype=torch.float32)
            return edge, torch.ones_like(edge)

        grad = torch.where(finite, grad, torch.zeros_like(grad))
        grad = grad / (grad.amax() + 1e-6)

        quantile = float(self.cfg.get("edge_quantile", 0.80))
        quantile = min(max(quantile, 0.0), 1.0)
        threshold = torch.quantile(grad.flatten(), quantile)
        threshold = torch.clamp(threshold, min=float(self.cfg.get("min_edge_threshold", 0.05)))

        edge = torch.clamp((grad - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)
        edge = F.max_pool2d(edge[None, None], kernel_size=3, stride=1, padding=1)[0, 0]

        dtf = self._distance_transform(edge > 0.0)
        return edge.contiguous(), dtf.contiguous()

    def edge_weight(
        self,
        source_edge: torch.Tensor,
        sampled_target_dtf: torch.Tensor,
        calibration: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = torch.clamp(source_edge * sampled_target_dtf, 0.0, 1.0)
        if calibration is not None:
            residual = residual * calibration.to(device=residual.device, dtype=residual.dtype)
            residual = torch.clamp(residual, 0.0, 1.0)

        power = float(self.cfg.get("residual_power", 1.0))
        if power != 1.0:
            residual = torch.pow(torch.clamp(residual, min=1e-6), power)

        strength = float(self.cfg.get("edge_weight_strength", 0.20))
        strength = min(max(strength, 0.0), 1.0)
        weight = 1.0 - strength * residual
        return torch.clamp(
            weight,
            min=float(self.cfg.get("min_edge_weight", 0.5)),
            max=float(self.cfg.get("max_edge_weight", 1.0)),
        )

    def _sobel_magnitude(self, gray: torch.Tensor) -> torch.Tensor:
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(-1, -2)
        grad_x = F.conv2d(gray, kernel_x, padding=1)
        grad_y = F.conv2d(gray, kernel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

    def _distance_transform(self, edge_mask: torch.Tensor) -> torch.Tensor:
        max_distance = float(self.cfg.get("max_distance_px", 8.0))
        if distance_transform_edt is not None:
            dtf = distance_transform_edt((~edge_mask).detach().cpu().numpy())
            dtf = torch.from_numpy(dtf).to(self.device, dtype=torch.float32)
        else:
            dtf = self._chamfer_distance(edge_mask)

        return torch.clamp(dtf / max(max_distance, 1e-6), 0.0, 1.0)

    def _chamfer_distance(self, edge_mask: torch.Tensor) -> torch.Tensor:
        h, w = edge_mask.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        edge_yx = torch.stack([yy[edge_mask], xx[edge_mask]], dim=-1).float()
        if edge_yx.numel() == 0:
            return torch.ones((h, w), device=self.device, dtype=torch.float32) * float(self.cfg.get("max_distance_px", 8.0))

        coords = torch.stack([yy, xx], dim=-1).float().view(-1, 2)
        chunks = []
        for chunk in coords.split(4096, dim=0):
            dist = torch.cdist(chunk, edge_yx)
            chunks.append(dist.amin(dim=1))
        return torch.cat(chunks, dim=0).view(h, w)
