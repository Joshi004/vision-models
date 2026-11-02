#!/usr/bin/env bash
# download_weights.sh - Download Stable Audio 3 Medium into the private Hugging
# Face cache <vision-data>/weights/stable_audio3/hf (~10.4 GB). The upstream
# loader resolves its files through the Hugging Face cache, and the repo also
# carries the T5Gemma text encoder, so one download is enough. The runner sets
# HF_HOME to this folder and runs offline.
#
# stabilityai/stable-audio-3-medium is gated. Before running this once:
#   1. Accept the terms on https://huggingface.co/stabilityai/stable-audio-3-medium
#   2. Log in with a Read token:  <vision-data>/envs/stable_audio3/bin/hf auth login
#      (or export HF_TOKEN for this one command)
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/stable_audio3/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
HF="$DATA_DIR/envs/stable_audio3/bin/hf"
DEST="$DATA_DIR/weights/stable_audio3/hf"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- run scripts/build_env.sh stable_audio3 first." >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== stabilityai/stable-audio-3-medium (gated) =="
HF_HOME="$DEST" "$HF" download stabilityai/stable-audio-3-medium

echo "== Done. Contents: =="
du -sh "$DEST"/*
