#!/usr/bin/env bash
# Slurm job entrypoint for the shared Qwen3.8-27B-FP8 server (vLLM, one GPU).
# Submitted by the gateway (gateway/services/qwen38/server_manager.py), which
# is also the only client of the server it starts.
#
# What it does:
#   1. Picks a free port and starts `vllm serve` on it, protected by an API
#      key. The key is read from <state-dir>/api_key (mode 0600, written by the
#      gateway) and handed to vLLM through the VLLM_API_KEY environment
#      variable -- never on the command line, where `ps` would show it.
#   2. Waits until vLLM answers /health, then writes <state-dir>/server.json
#      (host, port, Slurm job id, ready time) atomically. The gateway treats the
#      server as ready only once that file exists and names this Slurm job.
#   3. Removes server.json again when vLLM exits for any reason.
#
# Usage: serve.sh STATE_DIR MODEL_DIR ENV_DIR CACHE_DIR MAX_MODEL_LEN MAX_NUM_SEQS SERVED_NAME
set -euo pipefail

if [ $# -ne 7 ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 2
fi

STATE_DIR="$1"
MODEL_DIR="$2"
ENV_DIR="$3"
CACHE_DIR="$4"
MAX_MODEL_LEN="$5"
MAX_NUM_SEQS="$6"
SERVED_NAME="$7"

STATE_FILE="$STATE_DIR/server.json"
KEY_FILE="$STATE_DIR/api_key"

[ -f "$KEY_FILE" ] || { echo "Missing API key file: $KEY_FILE" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "Missing model: $MODEL_DIR (run models/qwen38/download_weights.sh)" >&2; exit 1; }
[ -x "$ENV_DIR/bin/vllm" ] || { echo "Missing environment: $ENV_DIR (run scripts/build_env.sh qwen38)" >&2; exit 1; }

# The compute node's driver supports CUDA 12.8 at most. Put this environment's
# own CUDA 12.x libraries ahead of the system toolkit (which can be newer than
# the driver), same fix as models/parakeet/runner/run_transcribe.sh.
SITE="$(echo "$ENV_DIR"/lib/python3*/site-packages)"
NVIDIA_LIBS=""
for d in cuda_runtime cublas cuda_cupti cuda_nvrtc cudnn cufft curand cusolver cusparse cusparselt nccl nvjitlink nvtx; do
  [ -d "$SITE/nvidia/$d/lib" ] && NVIDIA_LIBS="$NVIDIA_LIBS:$SITE/nvidia/$d/lib"
done
export LD_LIBRARY_PATH="${NVIDIA_LIBS#:}:${LD_LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:$PATH"

mkdir -p "$CACHE_DIR"
export VLLM_CACHE_ROOT="$CACHE_DIR"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
VLLM_API_KEY="$(cat "$KEY_FILE")"
export VLLM_API_KEY

rm -f "$STATE_FILE"

PORT="$("$ENV_DIR/bin/python" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
HOST="$(hostname -f 2>/dev/null || hostname)"
echo "== vLLM on $HOST:$PORT (model $MODEL_DIR, max_model_len $MAX_MODEL_LEN) =="

# Prompts and answers are not written to the log: request logging stays off
# (vLLM's default), and nothing below turns it on.
vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0}' &
VLLM_PID=$!

cleanup() {
  rm -f "$STATE_FILE"
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# Wait for the engine (weights, compile, warm-up). /health is the one route
# vLLM serves without the API key.
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before it became ready." >&2
    wait "$VLLM_PID" || true
    exit 1
  fi
  sleep 3
done

TMP="$STATE_FILE.tmp.$$"
printf '{"host": "%s", "port": %s, "slurm_job_id": "%s", "ready_at": "%s"}\n' \
  "$HOST" "$PORT" "${SLURM_JOB_ID:-}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TMP"
mv "$TMP" "$STATE_FILE"
echo "== ready: $STATE_FILE =="

wait "$VLLM_PID"
