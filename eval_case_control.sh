#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODEL_GROUPS=(cbramod biot labram handcrafted cnn)

for MODEL in "${MODEL_GROUPS[@]}"; do
  RESULTS_ROOT="$MODEL/result"
  [[ -d "$RESULTS_ROOT" ]] || { echo ">> $MODEL: no $RESULTS_ROOT; skipping."; continue; }

  # find every *_inference_results.csv under this model (handles cnn/vit/train/... too)
  while IFS= read -r -d '' PRED_CSV; do
    SAVE_DIR="$(dirname "$PRED_CSV")"
    echo -e "\n=== Evaluating $(realpath --relative-to=. "$PRED_CSV") ==="
    python evaluate_case_control.py \
      --prediction_csv "$PRED_CSV" \
      --output_folder  "$SAVE_DIR"
  done < <(find "$RESULTS_ROOT" -type f -name '*inference_results.csv' -print0)

done

shopt -u nullglob
