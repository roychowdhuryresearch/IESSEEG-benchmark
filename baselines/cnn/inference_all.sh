#!/usr/bin/env bash
# Clip-level inference for the from-scratch vision baselines.
#
# One implementation for both architectures, selected by ARCH:
#   ARCH=cnn  -> 3D ResNet-18   (inference_cnn_all.sh)
#   ARCH=vit  -> 3D ViT         (inference_vit_all.sh)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib/common.sh"

ARCH="${ARCH:-cnn}"
case "${ARCH}" in
  cnn) ARCH_DISPLAY="3D ResNet-18"; INFER_ENTRY="inference_cnn.py" ;;
  vit) ARCH_DISPLAY="3D ViT";       INFER_ENTRY="inference_vit.py" ;;
  *)   echo "ARCH must be 'cnn' or 'vit', got '${ARCH}'" >&2; exit 1 ;;
esac

STAGE_NAME="${ARCH_DISPLAY} inference"
TEST_DATA_DIR="$(model_test_dir cnn_resnet)"

cnn_inference () {
  local task="$1" fold="$2" label_key="$3" gpu="$4"
  local tag="${task}_fold${fold}"
  local res_dir="result/${ARCH}/inference/${tag}"
  mkdir -p "${res_dir}"

  "${PYTHON_BIN}" "${INFER_ENTRY}" \
    --inference_csv "$(split_csv "${task}" "${fold}" test)" \
    --data_root "${TEST_DATA_DIR}" \
    --model_file "ckpts/${ARCH}/${tag}.pth" \
    --out_csv "${res_dir}/${tag}_inference_results.csv" \
    --epoch_length 30 \
    --sfreq 200 \
    --label_key "${label_key}" \
    --cuda "${gpu}"
}

for_each_task_fold cnn_inference
