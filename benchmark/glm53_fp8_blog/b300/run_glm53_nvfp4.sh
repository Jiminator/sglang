#!/usr/bin/env bash
# GLM-5.3 nvfp4 — both B300 panels, then plot this curve alone.
set -euo pipefail
cd "$(dirname "$0")"
./run_glm53_nvfp4_tp8.sh
./run_glm53_nvfp4_tep8.sh
python3 ../plot_figure.py b300 --curves glm53_nvfp4 --out ../figure_b300_glm53_nvfp4.png
