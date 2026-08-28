#!/usr/bin/env bash
# Everything for GB300: the TP4 and TEP4 sweeps, then the single-panel figure
# with both curves.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm53_tp4.sh
./run_glm53_tep4.sh
python3 ../plot_figure.py gb300 --out ../figure_gb300.png
