#!/usr/bin/env bash
set -e

for set_id in "A" "B"; do
    LABEL_KEY="immediate_responder"
    SET_TO_TEST="${set_id}_${LABEL_KEY}"
    DATA_DIR="../../data"
    SPLIT_META="../../data/immediate_treatment_response_train_test_split"
    CKPT_DIR="ckpts/vit"
    RES_DIR="result/vit/inference/${SET_TO_TEST}"

    mkdir -p ${RES_DIR}

    python inference_vit.py \
        --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
        --data_root "${DATA_DIR}/baseline_test" \
        --model_file "${CKPT_DIR}/${SET_TO_TEST}.pth" \
        --out_csv "${RES_DIR}/inference_results.csv" \
        --epoch_length 10 \
        --sfreq 200 \
        --label_key ${LABEL_KEY} \
        --cuda 1 \
        "$@"
done