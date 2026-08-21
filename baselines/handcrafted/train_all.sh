#!/usr/bin/env bash
# GBDT + Clinical Prior: train for every task x fold. CPU only.
#
# Set FORCE_RETRAIN=1 to retrain folds that already have a model.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="GBDT train"
TRAIN_DATA_DIR="$(model_data_dir handcrafted)"

# Feature extraction dominates this baseline's runtime, and the same
# windows are re-read across folds, so the cache is shared across the
# whole sweep rather than per fold.
FEATURE_CACHE="${IESSEEG_FEATURE_CACHE:-$(pwd)/feature_cache/regional}"

handcrafted_train () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="result/train/${tag}"
  mkdir -p ckpts "${res_dir}" "${FEATURE_CACHE}"

  if [ -f "ckpts/${tag}.pth" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "${tag} already has a model, skipping (FORCE_RETRAIN=1 to redo)"
    return 0
  fi

  "${PYTHON_BIN}" train_handcrafted_baseline.py \
    --train_meta_csv "$(split_csv "${task}" "${fold}" train)" \
    --data_root "${TRAIN_DATA_DIR}" \
    --epoch_length 30 \
    --sfreq 200 \
    --test_size 0.1 \
    --model_out "ckpts/${tag}.pth" \
    --cv_folds 1 \
    --confusion_fig "${res_dir}/confusion_matrix.png" \
    --metrics_out "${res_dir}/metrics.txt" \
    --label_key "${label_key}" \
    --feature_cache_dir "${FEATURE_CACHE}"
}

# This baseline never touches a GPU; skip the picker so it can run on a
# machine with none, and alongside GPU jobs without competing for them.
CUDA_DEVICE="${CUDA_DEVICE:-cpu}" for_each_task_fold handcrafted_train
