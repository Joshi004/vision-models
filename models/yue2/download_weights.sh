#!/usr/bin/env bash
# download_weights.sh - Download YuE2 into <vision-data>/weights/yue2/:
#   YuE2-3B   the song language model (~7.3 GB)
#   YuE2-Vae  the audio VAE           (~0.5 GB)
#
# The repos are public; no token is needed. The model licence is CC BY-NC 4.0
# (non-commercial) -- see docs/models/yue2/README.md.
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/yue2/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
HF="$DATA_DIR/envs/yue2/bin/hf"
DEST="$DATA_DIR/weights/yue2"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- run scripts/build_env.sh yue2 first." >&2
  exit 1
fi

mkdir -p "$DEST"

# The assets/ and examples/ folders hold demo media; the runner does not need them.
echo "== m-a-p/YuE2-3B =="
"$HF" download m-a-p/YuE2-3B --exclude "assets/*" "examples/*" "*.whl" --local-dir "$DEST/YuE2-3B"

echo "== m-a-p/YuE2-Vae =="
"$HF" download m-a-p/YuE2-Vae --exclude "assets/*" --local-dir "$DEST/YuE2-Vae"

echo "== Done. Contents: =="
du -sh "$DEST"/*
