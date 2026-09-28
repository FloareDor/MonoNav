#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
image_name="${MONONAV_DOCKER_IMAGE:-mononav-demo:2.7.1-cu128}"

docker build \
  --tag "${image_name}" \
  --file "${script_dir}/Dockerfile.demo" \
  "${repo_dir}"
