#!/usr/bin/env bash
# download_weights.sh - Download Qwen3.8-27B-FP8 (Qwen's official block-FP8
# checkpoint) into $VISION_DATA_DIR/weights/qwen38/Qwen3.8-27B-FP8/
# (about 31 GB, 81 files). Apache-2.0, public, no token needed.
#
# It is the full checkpoint (text layers, the vision tower and the MTP head);
# the server runs it text-only, but the vision weights are part of the same
# checkpoint layout and vLLM expects them to be present.
#
# Build the environment first:  scripts/build_env.sh qwen38
#
# The revision is pinned (checked 2026-10-18), so a later change to the
# repository cannot silently change what the API serves.
# Safe to re-run: `hf download` skips files that are already complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"  # <repo>/models/qwen38/<this file>
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
HF="${QWEN38_HF_CLI:-$DATA_DIR/envs/qwen38/bin/hf}"
DEST="$DATA_DIR/weights/qwen38/Qwen3.8-27B-FP8"
REPO_ID="Qwen/Qwen3.8-27B-FP8"
REVISION="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"

if [ ! -x "$HF" ]; then
  echo "hf CLI not found at $HF -- build the env first: scripts/build_env.sh qwen38" >&2
  echo "(or point QWEN38_HF_CLI at any working 'hf' executable)" >&2
  exit 1
fi

mkdir -p "$DEST"

echo "== Qwen3.8-27B-FP8: $REPO_ID @ $REVISION -> $DEST =="
"$HF" download "$REPO_ID" --revision "$REVISION" --local-dir "$DEST"

echo "== Checking the files the server needs =="
missing=0
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja \
         model.safetensors.index.json outside.safetensors mtp.safetensors; do
  if [ ! -f "$DEST/$f" ]; then
    echo "MISSING: $f" >&2
    missing=1
  fi
done

# Every shard the index names must exist (a half-finished download shows up here).
if [ -f "$DEST/model.safetensors.index.json" ]; then
  while read -r shard; do
    if [ ! -f "$DEST/$shard" ]; then
      echo "MISSING shard: $shard" >&2
      missing=1
    fi
  done < <(python3 -c "
import json, sys
print('\n'.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))
" "$DEST/model.safetensors.index.json")
fi

if [ "$missing" -ne 0 ]; then
  echo "Download incomplete; re-run this script." >&2
  exit 1
fi
echo "== Done: $DEST ($(du -sh "$DEST" | cut -f1)) =="
