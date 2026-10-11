#!/usr/bin/env bash
# run.sh - (Re)start the shared model-api server inside a tmux session so it
# survives an SSH disconnect.
#
# Always serves on port 8012, and always restarts: re-running it stops the
# running gateway first and starts a fresh one with the current code, then
# waits until GET /v1/health answers.
#   * The `model-api` tmux session on this node is stopped.
#   * Any other process of yours still listening on port 8012 here is stopped.
#     If someone else's process holds the port, the script stops with an error.
#   * A gateway started on a different login node (tmux sessions are per node;
#     recorded in <data dir>/gateway.host) is stopped over ssh if possible;
#     otherwise you get the exact command to run there. Two gateways would
#     share one jobs.db, and your tunnel may point at the old one.
# Restarting does not affect Slurm jobs already submitted: the new gateway
# picks their status up from jobs.db.
#
# Env overrides:
#   MODEL_API_HOST                        (default 0.0.0.0; the port is always 8012)
#   MODEL_API_FALLBACK_PARTITION          (default "background")
#   MODEL_API_MAX_INFLIGHT                (optional soft concurrency cap, global across backends)
#   MODEL_API_RETENTION_DAYS              (finished job cleanup, default 7; a request sent with "incognito": true is the
#                                          exception: its inputs are deleted when its job ends, its result right after its
#                                          first complete download, and it keeps no logs)
#   MODEL_API_UNUSED_UPLOAD_TTL_MINUTES   (delete uploads that no job references after this long, default 60)
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
#
#   QWEN_IMAGE21_BASE_DIR, QWEN_IMAGE21_TURBO_DIR     (default <data dir>/weights/qwen_image21/{base,turbo})
#   QWEN_IMAGE21_MEM_PER_GPU, QWEN_IMAGE21_CPUS, QWEN_IMAGE21_JOB_TIME   (Qwen-Image-2.1 backend's own Slurm allocation shape)
#   QWEN_IMAGE21_MAX_PROMPT_CHARS                     (default 8000)
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$API_DIR/.." && pwd)"
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
SESSION="model-api"
PORT=8012
HOST="${MODEL_API_HOST:-0.0.0.0}"
THIS_NODE="$(hostname)"
NODE_FILE="$DATA_DIR/gateway.host"

if [ -n "${MODEL_API_PORT:-}" ] && [ "$MODEL_API_PORT" != "$PORT" ]; then
  echo "Note: ignoring MODEL_API_PORT=$MODEL_API_PORT -- the gateway always runs on port $PORT."
fi

# 1. A gateway last started on another login node.
if [ -f "$NODE_FILE" ]; then
  OTHER_NODE="$(head -n 1 "$NODE_FILE" | cut -d' ' -f1)"
  if [ -n "$OTHER_NODE" ] && [ "$OTHER_NODE" != "$THIS_NODE" ] \
     && curl -s -m 3 "http://$OTHER_NODE:$PORT/v1/health" | grep -q '"status":"ok"'; then
    echo "A gateway is running on $OTHER_NODE; stopping it..."
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$OTHER_NODE" "tmux kill-session -t $SESSION" 2>/dev/null; then
      echo "Stopped the gateway on $OTHER_NODE."
    else
      echo "WARNING: could not stop it automatically. Run this on $OTHER_NODE:"
      echo "           tmux kill-session -t $SESSION"
      echo "         Until then two gateways share one jobs.db."
    fi
  fi
fi

# 2. This node: the tmux session, then anything of ours still on the port.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Stopping the running '$SESSION' session on $THIS_NODE..."
  tmux kill-session -t "$SESSION"
fi

port_in_use() { [ -n "$(ss -Hltn "sport = :$PORT")" ]; }

for _ in $(seq 20); do port_in_use || break; sleep 0.5; done
if port_in_use; then
  PIDS="$(lsof -a -t -iTCP:"$PORT" -sTCP:LISTEN -u "$(id -u)" 2>/dev/null || true)"
  if [ -n "$PIDS" ]; then
    echo "Stopping your process(es) on port $PORT: $(echo $PIDS)"
    kill $PIDS 2>/dev/null || true
    for _ in $(seq 20); do port_in_use || break; sleep 0.5; done
    port_in_use && { kill -9 $PIDS 2>/dev/null || true; sleep 1; }
  fi
fi
if port_in_use; then
  echo "ERROR: port $PORT on $THIS_NODE is still in use by another user's process:" >&2
  ss -Hltn "sport = :$PORT" >&2
  exit 1
fi

# 3. Start, then wait for the health check.
tmux new-session -d -s "$SESSION" \
  "export VISION_DATA_DIR='$DATA_DIR'; cd '$API_DIR' && '$DATA_DIR/envs/gateway/bin/uvicorn' server:app --host '$HOST' --port $PORT; exec bash"
echo "$THIS_NODE $PORT" > "$NODE_FILE"

echo -n "Starting on $THIS_NODE:$PORT "
HEALTH=""
for _ in $(seq 60); do
  HEALTH="$(curl -s -m 2 "http://localhost:$PORT/v1/health" || true)"
  echo "$HEALTH" | grep -q '"status":"ok"' && break
  echo -n "."; sleep 1
done
echo
if ! echo "$HEALTH" | grep -q '"status":"ok"'; then
  echo "ERROR: the gateway did not answer within 60 s. Last lines of its output:" >&2
  tmux capture-pane -t "$SESSION" -p | grep -v '^$' | tail -n 20 >&2
  exit 1
fi

echo "Running:  $HEALTH"
echo "Node:     $THIS_NODE   (forward port $PORT from THIS node)"
echo "Tunnel:   ssh -N -L $PORT:localhost:$PORT <you>@$THIS_NODE   (or let Cursor/VS Code forward port $PORT)"
echo "Attach:   tmux attach -t $SESSION   (Ctrl-b d to detach without stopping it)"
echo "Stop:     tmux kill-session -t $SESSION"
if [ -d "$API_DIR/ui/dist" ]; then
  echo "Web UI:   http://localhost:$PORT/ui/"
else
  echo "Web UI:   not built yet -- see ui/README.md (cd ui && npm install && npm run build), then re-run this script."
fi
