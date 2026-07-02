# Omega Prior Phase 1 Experiments

This file records commands and results for the minimal VGGT/VGGT-Omega prior path in DROID-W.

## Cache Layout

Set `omega_prior.cache_dir` to a directory containing any of the following `.npy` patterns:

- Depth: `depths/00000.npy`, `depth/00000.npy`, `00000_depth.npy`, `depth_00000.npy`
- Confidence: `confidences/00000.npy`, `confidence/00000.npy`, `00000_confidence.npy`, `confidence_00000.npy`
- Uncertainty: `uncertainties/00000.npy`, `00000_uncertainty.npy`

Depth arrays are metric depth maps. Confidence arrays are normalized to `[0, 1]` by default and converted to DROID-W-style uncertainty.

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
