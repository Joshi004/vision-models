#!/usr/bin/env bash
# download_weights.sh - Download Qwen-Image-2.1 (base, "quality" mode) and
# Qwen-Image-2.1-Turbo ("fast" mode) into $VISION_DATA_DIR/weights/qwen_image21/.
#
# Neither repository is gated (checked 2026-10-10), so no token is needed.
# Each checkpoint is about 33 GB, so expect about 66 GB in total.
# License: Qwen Research License (non-commercial use only).
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/qwen_image21/<this file>
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
HF="$DATA_DIR/envs/qwen_image21/bin/hf"
DEST="$DATA_DIR/weights/qwen_image21"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- build the env first: scripts/build_env.sh qwen_image21" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== Base: Qwen/Qwen-Image-2.1 =="
"$HF" download Qwen/Qwen-Image-2.1 --local-dir "$DEST/base"

echo "== Turbo: Qwen/Qwen-Image-2.1-Turbo =="
"$HF" download Qwen/Qwen-Image-2.1-Turbo --local-dir "$DEST/turbo"

echo "== Done. Sizes: =="
du -sh "$DEST/base" "$DEST/turbo"
