#!/usr/bin/env bash
# 3D ResNet-18 inference: thin wrapper selecting ARCH=cnn. See inference_all.sh.
exec env ARCH=cnn bash "$(dirname "${BASH_SOURCE[0]}")/inference_all.sh" "$@"
