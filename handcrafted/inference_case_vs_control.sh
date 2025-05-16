#!/usr/bin/env bash
set -euo pipefail

for set_id in "A" "B"; do
    SET_TO_TEST="${set_id}_case_vs_control"
    DATA_DIR="../../data"
    SPLIT_META="../../data/case_control_train_test_split"
    RES_DIR="result/inference/${SET_TO_TEST}"

    mkdir -p ${RES_DIR}

    python inference.py \
      --model_file "ckpts/${SET_TO_TEST}.pth" \
      --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
      --data_root "${DATA_DIR}/baseline_test" \
      --epoch_length 10 \
      --sfreq 200 \
      --out_csv "${RES_DIR}/inference_results.csv"
done