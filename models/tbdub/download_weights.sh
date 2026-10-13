#!/usr/bin/env bash
# download_weights.sh - Download everything TBDub (30-step Teacher) needs into
# $VISION_DATA_DIR/weights/tbdub/ (about 15 GB):
#   * the Teacher DiT, its config.json manifest and the empty-prompt embedding
#     (TaoLiveAIGC/TBDub)
#   * the Wan2.2 VAE (KlingTeam/X-Dub)
#   * the HuBERT-large audio encoder (facebook/hubert-large-ll60k)
#   * MediaPipe's Face Landmarker, checked against the checksum TBDub documents
#
# None of the repositories is gated (checked 2026-10-12), so no token is needed.
# Licenses: TBDub and X-Dub are Apache-2.0; HuBERT and MediaPipe have their own.
# Build the environment first:  scripts/build_env.sh tbdub
#
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/tbdub/<this file>
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
HF="$DATA_DIR/envs/tbdub/bin/hf"
DEST="$DATA_DIR/weights/tbdub"

FACE_LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_SHA256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- build the env first: scripts/build_env.sh tbdub" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== TBDub Teacher: TaoLiveAIGC/TBDub =="
"$HF" download TaoLiveAIGC/TBDub \
  config.json null_prompt_emb.pt tbdub_teacher.safetensors \
  --local-dir "$DEST"

echo "== Wan2.2 VAE: KlingTeam/X-Dub =="
"$HF" download KlingTeam/X-Dub Wan2.2_VAE.safetensors --local-dir "$DEST"

echo "== HuBERT: facebook/hubert-large-ll60k =="
"$HF" download facebook/hubert-large-ll60k \
  config.json preprocessor_config.json pytorch_model.bin \
  --local-dir "$DEST/hubert-large-ll60k"

echo "== MediaPipe Face Landmarker =="
LANDMARKER="$DEST/face_landmarker.task"
if [ ! -f "$LANDMARKER" ]; then
  curl -fL "$FACE_LANDMARKER_URL" -o "$LANDMARKER"
fi
echo "$FACE_LANDMARKER_SHA256  $LANDMARKER" | sha256sum -c -

echo "== Done. Size: =="
du -sh "$DEST"
