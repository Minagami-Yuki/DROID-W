#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v7_softfloor.yaml
  configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v7_softfloor.yaml
  configs/Dynamic/Bonn/bonn_crowd_omega_patch_token_uncertainty_v7_softfloor.yaml
  configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v7_softfloor.yaml
  configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_patch_token_uncertainty_v7_softfloor.yaml
  configs/Dynamic/Bonn/bonn_person_tracking2_omega_patch_token_uncertainty_v7_softfloor.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "[$(date '+%F %T')] Running ${cfg}"
  conda run -n droid-w python run.py --config "${cfg}"
done
