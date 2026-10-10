#!/usr/bin/env bash
# download_weights.sh - Download the LTX-2.5 distilled split pack (plus the DFR
# detailing IC-LoRA) into ltx-2.5/models/ltx-2.5/.
#
# Both Hugging Face repos are gated. Before running this once:
#   1. Accept the terms on https://huggingface.co/Lightricks/LTX-2.5
#      and https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler
#   2. Log in with a Read token:  ltx-2.5/LTX-2/.venv/bin/hf auth login
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/ltx25/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
HF="$DATA_DIR/envs/ltx25/bin/hf"
DEST="$DATA_DIR/weights/ltx25/ltx-2.5"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- build the venv first (see model-api/STATUS.md, LTX-2.5 section)." >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== Main pack: Lightricks/LTX-2.5 =="
"$HF" download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  model_patches/ltx-2.5-duration-head-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --local-dir "$DEST"

echo "== DFR detailing IC-LoRA: Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler =="
"$HF" download Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler \
  ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
  --local-dir "$DEST/loras"

echo "== Done. Files: =="
find "$DEST" -name '*.safetensors' -printf '%8s bytes  %P\n' | sort -k3
du -sh "$DEST"
