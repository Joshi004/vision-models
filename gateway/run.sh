#!/usr/bin/env bash
# run.sh - Launch the shared model-api server inside a tmux session so it
# survives an SSH disconnect (the same pattern generate.sh recommends for
# long jobs).
#
# Safe to re-run: attaches to the existing session instead of starting a
# second one if it's already up.
#
# Env overrides:
#   MODEL_API_HOST, MODEL_API_PORT        (default 0.0.0.0:8012)
#   MODEL_API_FALLBACK_PARTITION          (default "background")
#   MODEL_API_MAX_INFLIGHT                (optional soft concurrency cap, global across backends)
#   MODEL_API_RETENTION_DAYS              (finished job cleanup, default 7)
#
#   VISION_DATA_DIR                       (default <repo>/../vision-data: envs/, weights/, jobs/, jobs.db live here)
#
#   LTX_MEM_PER_GPU, LTX_CPUS, LTX_API_TIME   (LTX-2.3 backend's own Slurm allocation shape)
#   LTX25_MEM_PER_GPU, LTX25_CPUS, LTX25_API_TIME   (LTX-2.5 backend's own Slurm allocation shape)
#   LTX25_VIDEO_VAE                           ("conv" (default, reliable at every length tried) or "diffusion" (see STATUS.md: corrupts the tail of long clips on this cluster))
#   LTX25_ENHANCER_GEMMA_DIR                  (Gemma instruct dir used only by enhance_prompt; default: LTX-2.3's gemma-3-12b)
#   WAN_ANIMATE_MEM_PER_GPU, WAN_ANIMATE_CPUS, WAN_ANIMATE_API_TIME   (Wan-Animate v1's own Slurm allocation shape)
#
#   PARAKEET_MODEL_PATH                               (default <data dir>/weights/parakeet/parakeet-tdt-0.6b-v3.nemo)
#   PARAKEET_MEM_PER_GPU, PARAKEET_CPUS, PARAKEET_JOB_TIME   (Parakeet backend's own Slurm allocation shape)
#
#   RVC_VOICES_DIR                                    (default <data dir>/weights/rvc/voices)
#
#   BREEZE_TTS_MODEL_DIR                              (default <data dir>/weights/breeze_tts/breeze-tts-2)
#   BREEZE_TTS_MEM_PER_GPU, BREEZE_TTS_CPUS, BREEZE_TTS_JOB_TIME   (Breeze TTS 2 backend's own Slurm allocation shape)
#   BREEZE_TTS_MAX_TEXT_CHARS                         (default 1000)
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$API_DIR/.." && pwd)"
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
SESSION="model-api"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' is already running. Attaching (Ctrl-b d to detach without stopping it)..."
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "export VISION_DATA_DIR='$DATA_DIR'; cd '$API_DIR' && '$DATA_DIR/envs/gateway/bin/uvicorn' server:app --host \"\${MODEL_API_HOST:-0.0.0.0}\" --port \"\${MODEL_API_PORT:-8012}\"; exec bash"

echo "Started in tmux session '$SESSION'."
echo "Attach:  tmux attach -t $SESSION   (Ctrl-b d to detach without stopping it)"
echo "Stop:    tmux kill-session -t $SESSION"
if [ -d "$API_DIR/ui/dist" ]; then
  echo "Web UI:  http://localhost:${MODEL_API_PORT:-8012}/ui/  (after tunneling, e.g. ssh -L 8012:localhost:8012 ...)"
else
  echo "Web UI:  not built yet -- see ui/README.md (cd ui && npm install && npm run build), then restart this script."
fi
