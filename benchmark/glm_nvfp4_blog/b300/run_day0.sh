#!/usr/bin/env bash
# GLM-5.2-NVFP4 on the day-0 snapshot — both B300 panels, then plot this curve.
set -euo pipefail
cd "$(dirname "$0")"
./run_day0_tp8.sh
./run_day0_tep8.sh
python3 ../plot_figure.py b300 --curves day0 \
    --out ../figure_b300_day0.png
