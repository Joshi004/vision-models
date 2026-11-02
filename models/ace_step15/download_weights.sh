#!/usr/bin/env bash
# download_weights.sh - Download the ACE-Step 1.5 checkpoint folder into
# <vision-data>/weights/ace_step15/checkpoints/:
#   acestep-v15-xl-sft          the XL (4B) SFT diffusion transformer   (~20 GB)
#   acestep-5Hz-lm-4B           the 4B language model (planner)          (~8.4 GB)
#   vae, Qwen3-Embedding-0.6B   shared audio VAE and text encoder        (~1.5 GB)
#
# The repos are public; no token is needed. Folder names must stay exactly as
# they are -- ACE-Step looks them up by name.
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/ace_step15/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
HF="$DATA_DIR/envs/ace_step15/bin/hf"
DEST="$DATA_DIR/weights/ace_step15/checkpoints"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- run scripts/build_env.sh ace_step15 first." >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== Shared VAE + text encoder: ACE-Step/Ace-Step1.5 =="
"$HF" download ACE-Step/Ace-Step1.5 \
  --include "vae/*" "Qwen3-Embedding-0.6B/*" \
  --local-dir "$DEST"

echo "== Diffusion transformer: ACE-Step/acestep-v15-xl-sft =="
"$HF" download ACE-Step/acestep-v15-xl-sft --local-dir "$DEST/acestep-v15-xl-sft"

echo "== Language model: ACE-Step/acestep-5Hz-lm-4B =="
"$HF" download ACE-Step/acestep-5Hz-lm-4B --local-dir "$DEST/acestep-5Hz-lm-4B"

echo "== Done. Contents: =="
du -sh "$DEST"/*
