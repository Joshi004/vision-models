#!/usr/bin/env bash
# download_weights.sh - Download Stable Audio Open 1.0 into
# <vision-data>/weights/stable_audio_open/:
#   model/      model_config.json + model.safetensors of the gated repo (~4.9 GB)
#   hf/         Hugging Face cache holding the T5 text encoder ("t5-base") that
#               the model loads by name; the runner sets HF_HOME to this folder
#
# stabilityai/stable-audio-open-1.0 is gated. Before running this once:
#   1. Accept the terms on https://huggingface.co/stabilityai/stable-audio-open-1.0
#   2. Log in with a Read token:  <vision-data>/envs/stable_audio_open/bin/hf auth login
#      (or export HF_TOKEN for this one command)
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/stable_audio_open/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
HF="$DATA_DIR/envs/stable_audio_open/bin/hf"
DEST="$DATA_DIR/weights/stable_audio_open"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- run scripts/build_env.sh stable_audio_open first." >&2
  exit 1
fi

mkdir -p "$DEST/model" "$DEST/hf"

echo "== stabilityai/stable-audio-open-1.0 (gated) =="
"$HF" download stabilityai/stable-audio-open-1.0 model_config.json model.safetensors \
  --local-dir "$DEST/model"

echo "== T5 text encoder: t5-base (into the private HF cache) =="
# xet is switched off: the legacy "t5-base" name has no xet token endpoint (404 seen on 2026-10-23).
HF_HUB_DISABLE_XET=1 HF_HOME="$DEST/hf" "$HF" download t5-base \
  config.json generation_config.json model.safetensors spiece.model tokenizer.json tokenizer_config.json

echo "== Done. Contents: =="
du -sh "$DEST"/*
