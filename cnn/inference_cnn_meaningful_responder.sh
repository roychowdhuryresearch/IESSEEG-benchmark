#!/usr/bin/env bash
set -euo pipefail

for set_id in "A" "B"; do
    LABEL_KEY="meaningful_responder"
    SET_TO_TEST="${set_id}_${LABEL_KEY}"
    DATA_DIR="../../data"
    SPLIT_META="../../data/meaningful_treatment_response_train_test_split"
    CKPT_DIR="ckpts/cnn"
    RES_DIR="result/cnn/inference/${SET_TO_TEST}"

    mkdir -p ${RES_DIR}

    python inference_cnn.py \
        --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
        --data_root "${DATA_DIR}/baseline_test" \
        --model_file "${CKPT_DIR}/${SET_TO_TEST}.pth" \
        --out_csv "${RES_DIR}/${SET_TO_TEST}_inference_results.csv" \
        --epoch_length 10 \
        --sfreq 200 \
        --label_key "${LABEL_KEY}" \
        --cuda 1
done