#!/usr/bin/env bash
# Build the Blackwell-capable AirSLAM dev image.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-airslam:blackwell}"

docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
echo "Built image: $IMAGE"
