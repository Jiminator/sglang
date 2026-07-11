#!/usr/bin/env bash
# GLM-5.1-NVFP4 on release/v0.5.15 — both GB300 panels, then plot this curve.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm51_v0515_tp4.sh
./run_glm51_v0515_tep4.sh
python3 ../plot_figure.py gb300 --curves glm51_v0515 \
    --out ../figure_gb300_glm51_v0515.png
