#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
image_name="${MONONAV_DOCKER_IMAGE:-mononav-demo:2.7.1-cu128}"
network_name="${AIRSTACK_DOCKER_NETWORK:-airstack_airstack_network}"
server_url="${MONONAV_AIRSTACK_SERVER:-http://airstack-robot-desktop-1:8765}"

if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
  echo "Docker image ${image_name} is missing. Build it with ./docker/build_image.sh."
  exit 1
fi

if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
  echo "AirStack network ${network_name} is missing. Start AirStack first."
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set. Run from the graphical desktop session."
  exit 1
fi

xauthority="${XAUTHORITY:-${HOME}/.Xauthority}"
if [[ ! -f "${xauthority}" ]]; then
  echo "Xauthority file not found: ${xauthority}"
  exit 1
fi

execute_args=()
if [[ "${MONONAV_EXECUTE:-false}" == "true" ]]; then
  execute_args+=(--execute)
fi

docker run --rm --name mononav-airstack --gpus all --ipc=host \
  --network "${network_name}" \
  --env DISPLAY="${DISPLAY}" \
  --env XAUTHORITY=/tmp/mononav.xauth \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "${xauthority}:/tmp/mononav.xauth:ro" \
  --volume "${repo_dir}:/workspace/MonoNav" \
  --volume mononav-torch-cache:/root/.cache/torch \
  --workdir /workspace/MonoNav \
  "${image_name}" \
  python mononav_airstack.py --server "${server_url}" "${execute_args[@]}" "$@"
