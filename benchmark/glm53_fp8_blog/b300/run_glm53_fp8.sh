#!/usr/bin/env bash
# GLM-5.3 fp8 — both B300 panels, then plot this curve alone.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm53_fp8_tp8.sh
python3 ../plot_figure.py b300 --curves glm53_fp8 --out ../figure_b300_glm53_fp8.png
