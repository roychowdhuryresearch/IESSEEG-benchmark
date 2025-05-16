#!/usr/bin/env bash
set -euo pipefail

for set_id in "A" "B"; do
    LABEL_KEY="immediate_responder"
    SET_TO_TEST="${set_id}_${LABEL_KEY}"
    DATA_DIR="../../data"
    SPLIT_META="../../data/immediate_treatment_response_train_test_split"
    RES_DIR="result/inference/${SET_TO_TEST}"

    mkdir -p ${RES_DIR}

    python inference.py \
      --model_file "ckpts/${SET_TO_TEST}.pth" \
      --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
      --data_root "${DATA_DIR}/baseline_test" \
      --epoch_length 10 \
      --sfreq 200 \
      --out_csv "${RES_DIR}/inference_results.csv" \
      --label_key "${LABEL_KEY}"
done