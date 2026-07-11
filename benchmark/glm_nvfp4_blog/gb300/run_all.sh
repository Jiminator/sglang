#!/usr/bin/env bash
# Everything for GB300: all three curves (day-0 first), then the full
# two-panel figure (TP=4, TEP=4) with all six curves.
set -euo pipefail
cd "$(dirname "$0")"
./run_day0.sh
./run_glm52_v0515.sh
./run_glm51_v0515.sh
python3 ../plot_figure.py gb300 --out ../figure_gb300.png
