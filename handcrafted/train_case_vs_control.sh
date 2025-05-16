#!/usr/bin/env bash
set -euo pipefail

SET_TO_TRAIN="A_case_vs_control"
DATA_DIR="../../data"
CKPT_DIR="ckpts"
RES_DIR="result/train/${SET_TO_TRAIN}"

# Create checkpoints directory if not exists
mkdir -p ${CKPT_DIR}
mkdir -p ${RES_DIR}

python train_handcrafted_baseline.py \
    --train_meta_csv "${DATA_DIR}/case_control_train_test_split/train_${SET_TO_TRAIN}.csv" \
    --data_root "${DATA_DIR}/scalp_eeg_data_200HZ_np_format" \
    --epoch_length 10 \
    --sfreq 200 \
    --test_size 0.1 \
    --model_out "${CKPT_DIR}/${SET_TO_TRAIN}.pth"  \
    --cv_folds 1 \
    --confusion_fig "${RES_DIR}/confusion_matrix.png" \
    --metrics_out "${RES_DIR}/metrics.txt" 
