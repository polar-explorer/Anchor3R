#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT CHECKPOINT OUTPUT_ROOT [SEQUENCES]" >&2
  echo "Example: DATASET_TYPE=kitti $0 /data/kitti-evc checkpoints/Anchor3R.pt outputs/kitti 00,01" >&2
  echo "DATASET_TYPE: kitti uses intrinsics crop; vbr or generic (default) uses center crop." >&2
  exit 2
fi

DATASET_TYPE=${DATASET_TYPE:-generic}
case "${DATASET_TYPE,,}" in
  kitti) PREPROCESS_FLAG=--camera-preprocess ;;
  vbr|generic) PREPROCESS_FLAG=--no-camera-preprocess ;;
  *) echo "Unsupported DATASET_TYPE: ${DATASET_TYPE}; expected kitti, vbr or generic." >&2; exit 2 ;;
esac

DATA_ROOT=$(realpath "$1")
CHECKPOINT=$(realpath "$2")
OUTPUT_ROOT=$(realpath -m "$3")
SEQUENCES=${4:-}
CAMERA=${CAMERA:-02}
INFER_MODE=${INFER_MODE:-online}
MAX_FRAMES=${MAX_FRAMES:-}
CONFIG=${CONFIG:-infer.yaml}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

if [[ -n "${SEQUENCES}" ]]; then
  IFS=',' read -r -a scenes <<< "${SEQUENCES}"
elif [[ -f "${DATA_ROOT}/data_roots.txt" ]]; then
  mapfile -t scenes < <(sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "${DATA_ROOT}/data_roots.txt")
else
  mapfile -t scenes < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

for scene in "${scenes[@]}"; do
  image_dir="${DATA_ROOT}/${scene}/images/${CAMERA}"
  if [[ ! -d "${image_dir}" ]]; then
    echo "[WARN] skipping ${scene}: missing ${image_dir}" >&2
    continue
  fi

  python scripts/validate_evc_dataset.py "${image_dir}"
  command=(
    python infer.py
    --config "${CONFIG}"
    --images "${image_dir}"
    --sequence-name "${scene}"
    "${PREPROCESS_FLAG}"
    --checkpoint "${CHECKPOINT}"
    --output "${OUTPUT_ROOT}"
    --infer-mode "${INFER_MODE}"
  )
  if [[ -n "${MAX_FRAMES}" ]]; then
    command+=(--max-frames "${MAX_FRAMES}")
  fi
  "${command[@]}"

  sequence_dir="${OUTPUT_ROOT}/${scene}"
  python scripts/visualize_results.py "${sequence_dir}"

  pose_dir="poses"
  if [[ "${INFER_MODE}" == "offline" ]]; then
    pose_dir="poses_offline"
  fi

  gt_poses="${DATA_ROOT}/${scene}/cameras/${CAMERA}/extri.yml"
  if [[ -f "${gt_poses}" ]]; then
    for trajectory in online offline; do
      pose_pred="${sequence_dir}/${pose_dir}/${trajectory}_pose.txt"
      if [[ ! -f "${pose_pred}" ]]; then
        continue
      fi
      python scripts/evaluate_trajectory.py \
        --pose-pred "${pose_pred}" \
        --images "${image_dir}" \
        --gt-poses "${gt_poses}" \
        --gt-format vbr-extri \
        --alignment sim3 \
        --output-dir "${sequence_dir}/pose_eval_${trajectory}" \
        --sequence-name "${scene}"
    done
  fi
done
