#!/usr/bin/env bash
# download_weights.sh - Download MiniMax H3 (the diffusers-format copy in
# MiniMaxAI/MiniMax-H3) into $VISION_DATA_DIR/weights/h3/MiniMax-H3/
# (about 210 GB, 196 GiB):
#
#   transformer/      66.3 GB  bf16, the FL2VA partition (text / first / last frame -> video + audio)
#   transformer_ref/  66.3 GB  bf16, the Ref2VA partition (image / video / audio references -> video + audio)
#   text_encoder/     66.7 GB  bf16, Qwen3-VL-32B conditioner (shared by both partitions)
#   vae/              10.4 GB  video VAE (shared)
#   audio_vae/         0.6 GB  audio VAE (shared)
#   scheduler/, audio_scheduler/, tokenizer/, processor/, the two index files,
#   LICENSE, README.md and the prompt-writing guides under docs/
#
# NOT downloaded: the FL2VA/ and Ref2VA/ folders of the same repo. They hold the
# original (non-diffusers) copy of the same weights, a second copy of the text
# encoder and VAE each, and add about 290 GB that nothing here reads.
#
# These are the complete released bf16 weights: no pruning, no quantization.
#
# Before the first run, read the license. It is NOT a click-through gate on
# Hugging Face (the repo is public, checked 2026-10-13, so no token is needed),
# but it is a real agreement:
#   https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
#   https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md
# In short (read the text, not this summary): the "Applicable Territory" is the
# world EXCLUDING the United States, the European Union, the United Kingdom and
# South Korea; using or hosting the weights or their outputs there needs
# separate written authorization from MiniMax (api@minimax.io); and commercial
# products above 20 million USD yearly revenue also need authorization.
#
# Build the environment first:  scripts/build_env.sh h3
#
# The revision is pinned (MiniMax's own self-hosting guide asks for it), so a
# later change to the repository cannot silently change what the API runs.
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/h3/<this file>
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
HF="$DATA_DIR/envs/h3/bin/hf"
DEST="$DATA_DIR/weights/h3/MiniMax-H3"
REPO_ID="MiniMaxAI/MiniMax-H3"
REVISION="42ed227ee7df40d41602854ae760620d6eb651fe"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- build the env first: scripts/build_env.sh h3" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== MiniMax H3 (diffusers layout): $REPO_ID @ $REVISION -> $DEST =="
"$HF" download "$REPO_ID" \
  --revision "$REVISION" \
  --include "transformer/*" \
  --include "transformer_ref/*" \
  --include "text_encoder/*" \
  --include "vae/*" \
  --include "audio_vae/*" \
  --include "scheduler/*" \
  --include "audio_scheduler/*" \
  --include "tokenizer/*" \
  --include "processor/*" \
  --include "model_index.json" \
  --include "modular_model_index.json" \
  --include "LICENSE" \
  --include "README.md" \
  --include "docs/*" \
  --local-dir "$DEST"

echo "== Checking the files the pipeline needs =="
missing=0
for f in \
  modular_model_index.json \
  transformer/config.json \
  transformer/diffusion_pytorch_model.safetensors.index.json \
  transformer_ref/config.json \
  transformer_ref/diffusion_pytorch_model.safetensors.index.json \
  text_encoder/config.json \
  text_encoder/model.safetensors.index.json \
  vae/config.json \
  vae/diffusion_pytorch_model.safetensors.index.json \
  audio_vae/config.json \
  audio_vae/diffusion_pytorch_model.safetensors \
  scheduler/scheduler_config.json \
  audio_scheduler/scheduler_config.json \
  tokenizer/tokenizer.json \
  processor/preprocessor_config.json; do
  if [ ! -f "$DEST/$f" ]; then
    echo "MISSING: $f" >&2
    missing=1
  fi
done

# Every shard an index names must exist (a half-finished download shows up here).
for d in transformer transformer_ref text_encoder vae; do
  index="$(ls "$DEST/$d"/*.safetensors.index.json 2>/dev/null | head -1)"
  [ -n "$index" ] || continue
  while read -r shard; do
    if [ ! -f "$DEST/$d/$shard" ]; then
      echo "MISSING shard: $d/$shard" >&2
      missing=1
    fi
  done < <(grep -o '"[^"]*\.safetensors"' "$index" | tr -d '"' | sort -u)
done

if [ "$missing" -ne 0 ]; then
  echo "Download incomplete -- re-run this script." >&2
  exit 1
fi

echo "== Done. Size: =="
du -sh "$DEST"
for d in transformer transformer_ref text_encoder vae audio_vae; do
  du -sh "$DEST/$d"
done
