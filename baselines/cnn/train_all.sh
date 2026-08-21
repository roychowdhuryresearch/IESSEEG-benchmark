#!/usr/bin/env bash
# From-scratch vision baselines on Morlet time-frequency cubes.
#
# One implementation for both architectures, selected by ARCH:
#   ARCH=cnn  -> 3D ResNet-18   (train_cnn_all.sh)
#   ARCH=vit  -> 3D ViT         (train_vit_all.sh)
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a checkpoint.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

ARCH="${ARCH:-cnn}"
case "${ARCH}" in
  cnn) ARCH_DISPLAY="3D ResNet-18"; TRAIN_ENTRY="train_cnn_baseline.py" ;;
  vit) ARCH_DISPLAY="3D ViT";       TRAIN_ENTRY="train_vit_baseline.py" ;;
  *)   echo "ARCH must be 'cnn' or 'vit', got '${ARCH}'" >&2; exit 1 ;;
esac

STAGE_NAME="${ARCH_DISPLAY} train"
TRAIN_DATA_DIR="$(model_data_dir cnn_resnet)"

cnn_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local ckpt_dir="ckpts/${ARCH}"
  local res_dir="result/${ARCH}/train/${tag}"
  mkdir -p "${ckpt_dir}" "${res_dir}"

  if [ -f "${ckpt_dir}/${tag}.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a checkpoint, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" "${TRAIN_ENTRY}" \
    --train_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --epoch_length 30 \
    --sfreq 200 \
    --test_size 0.2 \
    --val_size 0.2 \
    --model_out "${ckpt_dir}/${tag}.pth" \
    --confusion_fig "${res_dir}/confusion_matrix.png" \
    --metrics_out "${res_dir}/metrics.txt" \
    --patience 5 \
    --max_epochs 100 \
    --label_key "${label_key}" \
    --cuda "${gpu}"
}

for_each_task_fold cnn_train
