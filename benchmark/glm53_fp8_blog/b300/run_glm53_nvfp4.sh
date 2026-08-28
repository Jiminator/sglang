#!/usr/bin/env bash
# GLM-5.3 NVFP4 — both B300 panels, then plot this curve alone.
#
# The NVFP4 curve is optional: when RadixArk/GLM-5.3-NVFP4 is not in the local
# HF cache this script SKIPS (exit 0) so b300/run_all.sh still completes with
# the FP8 curve alone. Pre-stage the checkpoint (or set FORCE_DOWNLOAD=1 to
# let the server download it) to include the curve.
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh
if [ "${FORCE_DOWNLOAD:-0}" != "1" ] && ! model_cached RadixArk/GLM-5.3-NVFP4; then
    echo "SKIPPING NVFP4 curve: RadixArk/GLM-5.3-NVFP4 is not in the local HF cache"
    echo "(pre-stage the checkpoint, or FORCE_DOWNLOAD=1 to download it here)"
    exit 0
fi
./run_glm53_nvfp4_tp8.sh
python3 ../plot_figure.py b300 --curves glm53_nvfp4 --out ../figure_b300_glm53_nvfp4.png
