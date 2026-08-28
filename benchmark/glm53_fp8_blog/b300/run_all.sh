#!/usr/bin/env bash
# Everything for B300: both curves (FP8 first, then NVFP4), then the full
# two-panel figure (TP=8, TEP=8) with both curves in each panel.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm53_fp8.sh
./run_glm53_nvfp4.sh
python3 ../plot_figure.py b300 --out ../figure_b300.png
