#!/usr/bin/env bash
set -e

for set_id in "A" "B"; do
    # Name of the split and location of data
    LABEL_KEY="case_control_label"
    SET_TO_TEST="${set_id}_case_vs_control"
    DATA_DIR="../../data"
    SPLIT_META="../../data/case_control_train_test_split"

    # Where the trained model checkpoints are stored
    CKPT_DIR="ckpts/${SET_TO_TEST}"

    # Where we save final inference outputs
    RES_DIR="result/inference/${SET_TO_TEST}"
    mkdir -p "${RES_DIR}"

    # Inference script
    python inference.py \
      --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
      --data_root "${DATA_DIR}/biot_test" \
      --model_ckpt "${CKPT_DIR}/best_BIOT.ckpt" \
      --out_csv "${RES_DIR}/inference_results.csv" \
      --epoch_length 30 \
      --sfreq 200 \
      --batch_size 64 \
      --cuda 2 \
      --label_key ${LABEL_KEY}
done