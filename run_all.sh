#!/usr/bin/env bash
# Full run on one 48 GB GPU. Usage: bash run_all.sh   (CB_QUICK=1 for a 5-minute smoke test)
set -euo pipefail
cd "$(dirname "$0")"
pip install -q -r requirements.txt
python make_data.py
python cb.py base
python cb.py stage1 --seed 0
python cb.py prep
CB_SAVE_M2=1 python cb.py stage2 --methods naive,replay,kl,sdft,sdft_replay --seeds 0,1,2
python cb.py stage3 --methods naive,sdft_replay --seeds 0,1,2
python summarize.py
