#!/usr/bin/env bash
set -e

for set_id in "A" "B"; do
    LABEL_KEY="case_control_label"
    SET_TO_TEST="${set_id}_case_vs_control"
    DATA_DIR="../../data"
    SPLIT_META="../../data/case_control_train_test_split"
    RES_DIR="result/inference/${SET_TO_TEST}"

    mkdir -p ${RES_DIR}

    python inference.py \
        --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
        --data_root "${DATA_DIR}/labram_test" \
        --model_ckpt "./checkpoints/${SET_TO_TEST}/checkpoint-best.pth" \
        --out_csv "${RES_DIR}/inference_results.csv" \
        --epoch_length 10 \
        --sfreq 200 \
        --nb_classes 2 \
        --device cuda:1 \
        --disable_qkv_bias \
        --abs_pos_emb \
        --disable_rel_pos_bias \
        --label_key ${LABEL_KEY} \
        "$@"
done