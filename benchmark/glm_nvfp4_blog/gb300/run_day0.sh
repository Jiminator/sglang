#!/usr/bin/env bash
# GLM-5.2-NVFP4 on the day-0 snapshot — both GB300 panels, then plot this curve.
set -euo pipefail
cd "$(dirname "$0")"
./run_day0_tp4.sh
./run_day0_tep4.sh
python3 ../plot_figure.py gb300 --curves day0 \
    --out ../figure_gb300_day0.png
