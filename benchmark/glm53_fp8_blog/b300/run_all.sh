#!/usr/bin/env bash
# Everything for B300: the TP8 and TEP8 sweeps, then the single-panel figure
# with both curves.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm53_tp8.sh
./run_glm53_tep8.sh
python3 ../plot_figure.py b300 --out ../figure_b300.png
