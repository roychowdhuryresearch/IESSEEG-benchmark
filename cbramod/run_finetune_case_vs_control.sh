#!/usr/bin/env bash
set -e

LABEL_KEY="case_control_label"
SET_TO_TRAIN="B_case_vs_control"
DATA_DIR="../../data"
CKPT_DIR="ckpts/${SET_TO_TRAIN}"
RES_DIR="result/train/${SET_TO_TRAIN}"
PREDS_DIR="result/val_preds/${SET_TO_TRAIN}"

# Create checkpoints directory if not exists
mkdir -p ${CKPT_DIR}
mkdir -p ${RES_DIR}
mkdir -p ${PREDS_DIR}

python finetune_main.py \
  --train_csv       "${DATA_DIR}/case_control_train_test_split/train_${SET_TO_TRAIN}.csv" \
  --data_root       "${DATA_DIR}/scalp_eeg_data_200HZ_np_format_cbramod" \
  --epoch_length    30            \
  --sfreq           200           \
  --test_size       0.2           \
  --epochs          50            \
  --batch_size      64            \
  --lr              3e-5          \
  --optimizer       AdamW         \
  --weight_decay    5e-2          \
  --clip_value      1.0           \
  --dropout         0.1           \
  --num_workers     8             \
  --num_of_classes  2             \
  --model_dir       "$CKPT_DIR"   \
  --model_out       "$CKPT_DIR/best.pth" \
  --cuda            0             \
  --use_pretrained_weights \
  --foundation_dir pretrained_weights/pretrained_weights.pth \
  --save_preds_dir "$PREDS_DIR" \
  --label_key ${LABEL_KEY} \
  "$@"