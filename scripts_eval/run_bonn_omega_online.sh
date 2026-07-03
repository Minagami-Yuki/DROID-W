#!/usr/bin/env bash
set -u

CONFIGS=(
  "configs/Dynamic/Bonn/bonn_balloon_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_balloon2_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_crowd_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_crowd2_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_moving_nonobstructing_box_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_person_tracking_omega_online.yaml"
  "configs/Dynamic/Bonn/bonn_person_tracking2_omega_online.yaml"
)

LOG_DIR="Outputs/Bonn/omega_online_logs"
mkdir -p "${LOG_DIR}"

for cfg in "${CONFIGS[@]}"; do
  scene="$(basename "${cfg}" .yaml)"
  log="${LOG_DIR}/${scene}.log"
  echo "START ${scene} $(date)" | tee "${log}"
  conda run -n droid-w python run.py --config "${cfg}" >> "${log}" 2>&1
  status=$?
  echo "DONE ${scene} status=${status} $(date)" | tee -a "${log}"
done
