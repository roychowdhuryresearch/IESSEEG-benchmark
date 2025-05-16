#!/usr/bin/env bash
set -euo pipefail

LABEL_KEY="immediate_responder"
SET_TO_TRAIN="B_${LABEL_KEY}"
DATA_DIR="../../data"
CKPT_DIR="ckpts/vit/"
RES_DIR="result/vit/train/${SET_TO_TRAIN}"

# Create checkpoints directory if not exists
mkdir -p ${CKPT_DIR}
mkdir -p ${RES_DIR}

python train_vit_baseline.py \
    --train_csv "${DATA_DIR}/immediate_treatment_response_train_test_split/train_${SET_TO_TRAIN}.csv" \
    --data_root "${DATA_DIR}/scalp_eeg_data_200HZ_np_format" \
    --epoch_length 10 \
    --sfreq 200 \
    --test_size 0.1 \
    --model_out "${CKPT_DIR}/${SET_TO_TRAIN}.pth"  \
    --confusion_fig "${RES_DIR}/confusion_matrix.png" \
    --metrics_out "${RES_DIR}/metrics.txt" \
    --label_key ${LABEL_KEY} \
    --patience 5 \
    --max_epochs 100 \
    --cuda 2
