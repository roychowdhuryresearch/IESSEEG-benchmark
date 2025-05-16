#!/usr/bin/env bash

for set_id in "A" "B"; do
    LABEL_KEY="case_control_label"
    SET_TO_TEST="${set_id}_case_vs_control"
    DATA_DIR="../../data"
    SPLIT_META="../../data/case_control_train_test_split"
    CKPT_DIR="ckpts/${SET_TO_TEST}"
    RES_DIR="result/inference/${SET_TO_TEST}"

    mkdir -p ${RES_DIR}

    python inference_cbramod.py \
        --inference_csv "${SPLIT_META}/test_${SET_TO_TEST}.csv" \
        --data_root "${DATA_DIR}/cbramod_test" \
        --model_file "${CKPT_DIR}/best.pth" \
        --out_csv "${RES_DIR}/inference_results.csv" \
        --epoch_length 30 \
        --sfreq 200 \
        --label_key ${LABEL_KEY} \
        --cuda 2
done

# We can also inference on the other training set, because it also satisfies patient wise data exclusion principle.
# SET_TO_TEST="A"
# DATA_DIR="../../data"
# SPLIT_META="../../data/case_control_train_test_split"
# CKPT_DIR="ckpts/${SET_TO_TEST}"
# RES_DIR="result/inference_on_other_train/${SET_TO_TEST}"

# mkdir -p ${RES_DIR}

# python inference_cbramod.py \
#     --inference_csv "${SPLIT_META}/trainB.csv" \
#     --data_root "${DATA_DIR}/scalp_eeg_data_200HZ_np_format_cbramod" \
#     --model_file "${CKPT_DIR}/best.pth" \
#     --out_csv "${RES_DIR}/inference_results.csv" \
#     --epoch_length 30 \
#     --sfreq 200