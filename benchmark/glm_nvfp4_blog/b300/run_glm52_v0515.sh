#!/usr/bin/env bash
# GLM-5.2-NVFP4 on release/v0.5.15 — both B300 panels, then plot this curve.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm52_v0515_tp8.sh
./run_glm52_v0515_tep8.sh
python3 ../plot_figure.py b300 --curves glm52_v0515 \
    --out ../figure_b300_glm52_v0515.png
