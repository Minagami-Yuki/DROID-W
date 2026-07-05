# Omega Prior Phase 1 Experiments

This file records commands and results for the minimal VGGT/VGGT-Omega prior path in DROID-W.

## Cache Layout

Set `omega_prior.cache_dir` to a directory containing any of the following `.npy` patterns:

- Depth: `depths/00000.npy`, `depth/00000.npy`, `00000_depth.npy`, `depth_00000.npy`
- Confidence: `confidences/00000.npy`, `confidence/00000.npy`, `00000_confidence.npy`, `confidence_00000.npy`
- Uncertainty: `uncertainties/00000.npy`, `00000_uncertainty.npy`

Depth arrays are metric depth maps. Confidence arrays are normalized to `[0, 1]` by default and converted to DROID-W-style uncertainty.

## Online VGGT-Omega

The online path uses the official `facebookresearch/vggt-omega` implementation as a submodule:

```bash
git submodule update --init --recursive thirdparty/vggt-omega
```

Checkpoint used on this server:

```text
/data1/czy/Output/DROID-W/vggt_omega_1b_512.pt
```

Example configs:

```bash
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon_omega_online.yaml
conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_crowd_omega_online.yaml
```

Runtime Omega uncertainty visualization can be enabled per experiment config:

```yaml
omega_prior:
  visualization:
    enable: True
    interval: 100
```

The default output directory is `<output>/<scene>/omega_uncertainty_vis`. The saver writes `frame_*.png` every `interval` input frames, using the current VGGT-Omega uncertainty map. If the scheduled input frame is not selected as a DROID-W keyframe, the saver performs an extra Omega uncertainty prediction only for visualization.

## Ablations

Baseline:

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd2.yaml
```

Omega depth only:

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_depth.yaml
```

Omega uncertainty only:

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_uncertainty.yaml
```

Omega depth + uncertainty:

```bash
python run.py --config configs/Dynamic/Bonn/bonn_crowd2_omega_depth_uncertainty.yaml
```

## Results

| Date | Command | Cache | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-02 | `python -m compileall src` | N/A | pass | Syntax smoke test for all `src` Python files |
| 2026-07-02 | inline config load for `configs/Dynamic/Bonn/bonn_crowd2.yaml` | N/A | pass | Inherited `omega_prior.enable`, `depth.enable`, and `uncertainty.enable` are all `False` |
| 2026-07-02 | `python -m py_compile src/depth_video.py src/motion_filter.py src/factor_graph.py src/utils/omega_prior.py` | N/A | pass | Focused syntax check after integration edits |
| 2026-07-02 | inline `OmegaPriorCache` dummy-cache test | `/tmp/omega_prior_*` | pass | Loaded depth/confidence, resized to `(4, 6)`, uncertainty range `[0.78, 1.0]`, weight range `[0.1, 1.0]` |
| 2026-07-03 | `git clone --depth 1 git@github.com:facebookresearch/vggt-omega.git thirdparty/vggt-omega` | N/A | pass | Installed official VGGT-Omega at commit `39a0cb8` |
| 2026-07-03 | `conda run -n droid-w python -m py_compile src/utils/omega_predictor.py src/utils/omega_prior.py src/motion_filter.py src/depth_video.py src/factor_graph.py` | N/A | pass | Syntax check in the actual DROID-W env |
| 2026-07-03 | inline `OmegaOnlinePredictor` single-frame forward on Bonn balloon | `/data1/czy/Output/DROID-W/vggt_omega_1b_512.pt` | pass | Depth shape `(480, 640)`, mean `1.0424`; confidence mean `28.0436`; uncertainty range `[0.78, 1.0]` |
| 2026-07-03 | `conda run -n droid-w python run.py --config configs/Dynamic/Bonn/bonn_balloon_omega_smoke.yaml` | `/data1/czy/Output/DROID-W/vggt_omega_1b_512.pt` | pass | 120 frames, 16 keyframes; KF ATE RMSE `0.02138`, full ATE RMSE `0.01889`; final BA disabled |
| 2026-07-05 | `python3 -m py_compile src/motion_filter.py src/utils/omega_visualization.py` | N/A | pass | Syntax check after adding periodic Omega uncertainty visualization |
| 2026-07-05 | inline `OmegaUncertaintyVisualizer` dummy tensor save | `/tmp/omega_vis_*` | pass | Wrote `frame_000100_ts_000100_kf_000003.png` using OpenCV `magma` colormap |
