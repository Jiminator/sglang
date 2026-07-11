#!/usr/bin/env bash
# One command for the whole GB300 ISL ablation: set up the client and the day-0
# checkout, run both c=1 ISL ladders (day-0 first, so the run ends on v0.5.15),
# and render the grouped-bar figure. Safe to re-run — each rung skips itself once
# its summary exists, so an interrupted ladder just resumes.
set -euo pipefail
cd "$(dirname "$0")/.."           # glm_nvfp4_blog/: common.sh + evalscope-deps live here
source common.sh

ensure_evalscope                  # pinned evalscope client + datasets>=4.0 for the build
ensure_day0_checkout              # ../sglang-day0 @ 22dce5720
export DAY0_SGLANG                # the day-0 ladder reads it from the server's environment

isl_ablation/run_isl_client.sh day0
isl_ablation/run_isl_client.sh v0515
python3 isl_ablation/plot_isl_figure.py
