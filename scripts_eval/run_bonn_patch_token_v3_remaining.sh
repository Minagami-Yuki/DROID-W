#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_config() {
  local config="$1"
  echo "[$(date '+%F %T')] RUN $config"
  conda run -n droid-w python run.py --config "$config"
}

ensure_cache() {
  local scene="$1"
  local config="$2"
  local cache_dir="/data1/czy/Output/DROID-omega/cache/Bonn/${scene}"
  if [[ -d "${cache_dir}/patch_tokens" ]] && find "${cache_dir}/patch_tokens" -type f -name '*.npy' -print -quit | grep -q .; then
    echo "[$(date '+%F %T')] CACHE OK $scene"
  else
    run_config "$config"
  fi
}

ensure_cache bonn_balloon configs/Dynamic/Bonn/bonn_balloon_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_balloon_omega_patch_token_uncertainty_v3.yaml

ensure_cache bonn_balloon2 configs/Dynamic/Bonn/bonn_balloon2_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_balloon2_omega_patch_token_uncertainty_v3.yaml

ensure_cache bonn_crowd configs/Dynamic/Bonn/bonn_crowd_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_crowd_omega_patch_token_uncertainty_v3.yaml

ensure_cache bonn_moving_nonobstructing_box configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_patch_token_uncertainty_v3.yaml

ensure_cache bonn_moving_nonobstructing_box2 configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_patch_token_uncertainty_v3.yaml

ensure_cache bonn_person_tracking configs/Dynamic/Bonn/bonn_person_tracking_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_person_tracking_omega_patch_token_uncertainty_v3.yaml

ensure_cache bonn_person_tracking2 configs/Dynamic/Bonn/bonn_person_tracking2_omega_cache_write.yaml
run_config configs/Dynamic/Bonn/bonn_person_tracking2_omega_patch_token_uncertainty_v3.yaml
