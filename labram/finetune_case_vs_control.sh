#!/usr/bin/env bash
set -e

LABEL_KEY="case_control_label"
SET_TO_TRAIN="B_case_vs_control"
DATA_DIR="../../data"

python run_class_finetuning.py \
    --train_csv       "${DATA_DIR}/case_control_train_test_split/train_${SET_TO_TRAIN}.csv" \
    --data_root       "${DATA_DIR}/scalp_eeg_data_200HZ_np_format_labram" \
    --output_dir ./checkpoints/${SET_TO_TRAIN} \
    --log_dir ./log/finetune_${SET_TO_TRAIN}_base \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 50 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset ${SET_TO_TRAIN} \
    --disable_qkv_bias \
    --seed 42 \
    --device cuda:3 \
    --label_key ${LABEL_KEY} \
    "$@"