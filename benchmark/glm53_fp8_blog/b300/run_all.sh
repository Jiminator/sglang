#!/usr/bin/env bash
# Everything for B300: both curves (FP8 first, then NVFP4), then the
# TP=8 figure with both curves.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm53_fp8.sh
./run_glm53_nvfp4.sh
python3 ../plot_figure.py b300 --out ../figure_b300.png
