#!/usr/bin/env bash
# Everything for B300: all three curves (day-0 first), then the full
# two-panel figure (TP=8, TEP=8) with all six curves.
set -euo pipefail
cd "$(dirname "$0")"
./run_day0.sh
./run_glm52_v0515.sh
./run_glm51_v0515.sh
python3 ../plot_figure.py b300 --out ../figure_b300.png
