#!/usr/bin/env bash
# 3D ViT training: thin wrapper selecting ARCH=vit. See train_all.sh.
exec env ARCH=vit bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
