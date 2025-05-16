#!/usr/bin/env bash
# -------------------------------------------------------------------------
# run_finetune.sh -- one-liner to launch finetune_main.py with our dataset
# -------------------------------------------------------------------------
#  • assumes finetune_main.py is in the current folder
#  • SET_TAG lets you pick train-split “A/B/C …” (matches CSV naming)
#  • change MODEL_NAME to the backbone you want (e.g. BIOT, SPaRCNet …)
#  • set PRETRAIN_PATH only if the backbone can load released weights
# -------------------------------------------------------------------------
set -euo pipefail

# ----------------- user-adjustable knobs ----------------------------------
SET_TAG="B_case_vs_control"                                   # A / B
MODEL_NAME="BIOT"                             # SPaRCNet | BIOT | …
PRETRAIN_PATH="pretrained-models/EEG-six-datasets-18-channels.ckpt"  # "" for none
GPU_ID=3                                       # which CUDA device
EPOCHS=50
BATCH=64
LR=3e-5
WORKERS=8
# -------------------------------------------------------------------------

DATA_DIR="../../data"
CSV_FILE="${DATA_DIR}/case_control_train_test_split/train_${SET_TAG}.csv"
NPZ_ROOT="${DATA_DIR}/scalp_eeg_data_200HZ_np_format_biot"

CKPT_DIR="ckpts/${SET_TAG}"
PREDS_DIR="result/val_preds/${SET_TAG}"
mkdir -p "${CKPT_DIR}" "${PREDS_DIR}"

echo "=== Finetuning ${MODEL_NAME} on split ${SET_TAG} ==="

python run_binary_supervised.py \
  --train_csv "${CSV_FILE}" \
  --data_root "${NPZ_ROOT}" \
  --seed 42 \
  --test_size 0.2 \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH} \
  --lr ${LR} \
  --weight_decay 5e-2 \
  --num_workers ${WORKERS} \
  --dataset "CASE_CTRL" \
  --model "${MODEL_NAME}" \
  --in_channels 18 \
  --sample_length 30 \
  --sfreq 200 \
  --token_size 200 \
  --hop_length 100 \
  --pretrain_model_path "${PRETRAIN_PATH}" \
  --model_dir "${CKPT_DIR}" \
  --model_out "best_${MODEL_NAME}" \
  --save_preds_dir "${PREDS_DIR}" \
  --cuda ${GPU_ID} \
  --label_key "case_control_label" \
  "$@"

echo "=== Done.  Best checkpoint is in ${CKPT_DIR} ==="
