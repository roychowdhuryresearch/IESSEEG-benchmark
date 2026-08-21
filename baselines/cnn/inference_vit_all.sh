#!/usr/bin/env bash
# 3D ViT inference: thin wrapper selecting ARCH=vit. See inference_all.sh.
exec env ARCH=vit bash "$(dirname "${BASH_SOURCE[0]}")/inference_all.sh" "$@"
