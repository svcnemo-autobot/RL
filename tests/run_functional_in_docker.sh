#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..")
set -eou pipefail
if ! command -v docker &> /dev/null; then echo "Error: Docker is not installed or not in PATH."; exit 1; fi
if [[ -z "${CONTAINER:-}" ]]; then echo "Error: CONTAINER environment variable is not set."; echo "Usage: CONTAINER=<docker-container> $0 <script to run, e.g., functional/grpo.sh>"; exit 1; fi
if [[ $# -ne 1 ]]; then echo "Error: Did not provide functional test script to run."; echo "Usage: CONTAINER=<docker-container> $0 <script to run, e.g., functional/grpo.sh>"; exit 1; fi
TEST_SCRIPT=$(realpath "$1")
CONTAINER=${CONTAINER}
export HF_HOME=${HF_HOME:-$(realpath "$SCRIPT_DIR/../hf_home")}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$(realpath "$SCRIPT_DIR/../hf_datasets_cache")}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
INTERACTIVE_FLAG=""
if [[ "${CI:-false}" != "true" ]]; then INTERACTIVE_FLAG="-it"; fi
source "$SCRIPT_DIR/scripts/detect_rdma.sh"
RDMA_FLAGS=()
if rdma_device_available; then RDMA_FLAGS=(--device=/dev/infiniband --cap-add=IPC_LOCK); fi
# Use the caller identity and an executable writable home while reusing the baked environment.
docker run --user "$(id -u):$(id -g)" $INTERACTIVE_FLAG "${RDMA_FLAGS[@]}" \
  --ulimit memlock=-1 --ulimit stack=67108864 --rm --gpus '"device=0,1"' \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" -v "$HF_HOME:/hf_home" -v "$HF_DATASETS_CACHE:/hf_datasets_cache" \
  --tmpfs "/home/nemo-rl:rw,exec,nosuid,nodev,mode=0700,uid=$(id -u),gid=$(id -g)" \
  -e WANDB_API_KEY -e HF_TOKEN -e HF_HOME=/hf_home -e HF_DATASETS_CACHE=/hf_datasets_cache \
  -e HOME=/home/nemo-rl -e UV_CACHE_DIR=/home/nemo-rl/.cache/uv \
  -w "$SCRIPT_DIR" "$CONTAINER" -- bash -x -c 'uv run --no-sync bash -x "$1"' bash "$TEST_SCRIPT"
