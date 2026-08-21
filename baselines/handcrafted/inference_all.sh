#!/usr/bin/env bash
# GBDT + Clinical Prior: clip-level inference for every task x fold.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

STAGE_NAME="GBDT inference"
TEST_DATA_DIR="${IESSEEG_DATA_ROOT}/baseline_test"
FEATURE_CACHE="${IESSEEG_FEATURE_CACHE:-$(pwd)/feature_cache/regional}"

handcrafted_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="result/inference/${tag}"
  mkdir -p "${res_dir}" "${FEATURE_CACHE}"

  "${PYTHON_BIN}" inference.py \
    --model_file "ckpts/${tag}.pth" \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --epoch_length 30 \
    --sfreq 200 \
    --label_key "${label_key}" \
    --out_csv "${res_dir}/inference_results.csv" \
    --feature_cache_dir "${FEATURE_CACHE}"
}

CUDA_DEVICE="${CUDA_DEVICE:-cpu}" for_each_task_fold handcrafted_inference
