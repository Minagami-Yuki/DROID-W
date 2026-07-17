# Confidence Main v1

This is the frozen main method for unknown-intrinsics tracking.

- Initialize one shared pinhole K from the median of VGGT-Omega predictions over 12 frames sampled every 5 input frames.
- Keep the normal DROID-W tracking graph and weights unchanged. Omega depth and uncertainty do not reweight the pose/depth BA in this preset.
- Optimize only the shared focal scale (`fx`, `fy` with fixed Omega aspect ratio) jointly with pose and depth through `droidcalib_schur`.
- Use a focal prior weight of 5, 20-keyframe warm-up, an information gate of 100, bounded log updates (0.002 per regular update, 0.004 during bootstrap), and full rollback on rejected joint updates.
- During keyframes 30--80, allow early high-frequency recovery only after three stable pose updates. A raw Omega confidence mean/median score of at least 1.05 over 30 samples enables the confidence-recovery route.

The optional full-pinhole (`fx, fy, cx, cy`) and calibration-observation branches remain disabled. The full-pinhole soft-weight control was neutral on TUM and worse than this focal preset on Bonn, so it is retained as an experimental branch rather than the default method.

The exact machine-readable parameters are in `configs/Experiments/confidence_main_v1.yaml`.
