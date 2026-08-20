#!/usr/bin/env bash
set -euo pipefail

python estimate_depth.py
python fuse_depth.py
python simulate.py

echo "MonoNav live demo complete."
