#!/usr/bin/env bash
# 3D ResNet-18 training: thin wrapper selecting ARCH=cnn. See train_all.sh.
exec env ARCH=cnn bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
