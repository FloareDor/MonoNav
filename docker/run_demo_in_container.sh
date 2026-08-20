#!/usr/bin/env bash
set -euo pipefail

Xvfb :99 -screen 0 1280x720x24 -nolisten tcp -ac &
xvfb_pid=$!
trap 'kill "${xvfb_pid}" 2>/dev/null || true' EXIT
export DISPLAY=:99
sleep 1

python estimate_depth.py
python fuse_depth.py
python simulate.py

/usr/bin/ffmpeg -y \
  -framerate 8 -start_number 0 \
  -i data/demo_hallway/crazyflie-rgb-images/crazyflie_frame-%06d.rgb.jpg \
  -framerate 8 -start_number 0 \
  -i data/demo_hallway/kinect-depth-images/kinect_frame-%06d.depth.jpg \
  -filter_complex '[0:v][1:v]hstack=inputs=2[v]' \
  -map '[v]' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -shortest data/demo_hallway/mononav_rgb_depth.mp4
