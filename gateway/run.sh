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
#   * Every gateway of yours on another login node that uses the same data
#     directory (tmux sessions are per node) is stopped over ssh if possible;
#     otherwise you get the exact command to run there. Two gateways would
#     share one jobs.db, and your tunnel may point at the old one. Gateways
#     using a different data directory (another checkout) are left alone.
#     Login nodes are the login-<N> names that resolve here, plus the node
#     recorded in <data dir>/gateway.host.
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
#   MODEL_API_LOGIN_NODES                 (space-separated login nodes to check for other gateways;
#                                          default: every login-<N> name, N = 0..31, that resolves)
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
#   TBDUB_MEM_PER_GPU, TBDUB_CPUS, TBDUB_API_TIME     (TBDub lip-sync backend's own Slurm allocation shape; defaults 64G, 8, 02:00:00)
#   TBDUB_LIPSYNC_EXTRA_SECONDS                       (default 1800: time added to an LTX audio-to-video job's limit when it asks for
#                                                      lip sync, and the most the lip-sync step may then take before it is stopped)
#
#   H3_MODEL_DIR                                      (default <data dir>/weights/h3/MiniMax-H3)
#   H3_MEM_PER_GPU, H3_CPUS, H3_API_TIME              (MiniMax H3 backend's own Slurm allocation shape; every job holds 2 GPUs;
#                                                      defaults 112G per GPU, 16 CPUs in total, 01:30:00 -- estimates, not yet measured)
#   H3_ATTENTION_BACKEND                              (diffusers attention backend for the H3 transformer; default _flash_3_hub
#                                                      = FlashAttention-3 on Hopper. Set it to an empty string to use diffusers'
#                                                      default instead. Needs the `kernels` package in envs/h3 and the
#                                                      flash-attn3 kernel in the Hugging Face cache, because jobs run offline;
#                                                      see docs/models/h3/README.md)
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$API_DIR/.." && pwd)"
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
SESSION="model-api"
PORT=8012
HOST="${MODEL_API_HOST:-0.0.0.0}"
THIS_NODE="$(hostname)"
NODE_FILE="$DATA_DIR/gateway.host"
# `-` (not `:-`): unset -> FlashAttention-3; set but empty -> diffusers' default backend.
H3_ATTENTION_BACKEND="${H3_ATTENTION_BACKEND-_flash_3_hub}"

if [ -n "${MODEL_API_PORT:-}" ] && [ "$MODEL_API_PORT" != "$PORT" ]; then
  echo "Note: ignoring MODEL_API_PORT=$MODEL_API_PORT -- the gateway always runs on port $PORT."
fi

# 1. Gateways on other login nodes that share this data directory.
LOGIN_NODES="${MODEL_API_LOGIN_NODES:-$(for i in $(seq 0 31); do if getent hosts "login-$i" >/dev/null; then echo "login-$i"; fi; done)}"
if [ -f "$NODE_FILE" ]; then
  LOGIN_NODES="$LOGIN_NODES $(head -n 1 "$NODE_FILE" | cut -d' ' -f1)"
fi

# Runs on the other node (data dir as $1): stops each of your gateways there
# whose VISION_DATA_DIR is this data directory, by killing the tmux session
# whose pane started it (or the process itself if it isn't in tmux).
read -r -d '' STOP_REMOTE_GATEWAYS <<'EOF' || true
data_dir="$1"
for pid in $(pgrep -u "$(id -u)" -f 'uvicorn server:app'); do
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qxF "VISION_DATA_DIR=$data_dir" || continue
  parent="$(ps -o ppid= -p "$pid" | tr -d ' ')"
  session="$(tmux list-panes -a -F '#{pane_pid} #{session_name}' 2>/dev/null | awk -v p="$parent" '$1 == p { print $2; exit }')"
  if [ -n "$session" ]; then
    tmux kill-session -t "$session" && echo "  stopped tmux session '$session' (gateway pid $pid)"
  else
    kill "$pid" && echo "  stopped gateway pid $pid"
  fi
done
EOF

for NODE in $(printf '%s\n' $LOGIN_NODES | sort -u); do
  [ "$NODE" = "$THIS_NODE" ] && continue
  OTHER_HEALTH="$(curl -s -m 3 "http://$NODE:$PORT/v1/health" || true)"
  [[ "$OTHER_HEALTH" == *'"status":"ok"'* ]] || continue
  echo "A gateway is answering on $NODE:$PORT; stopping it if it uses $DATA_DIR..."
  # accept-new: login nodes not yet in known_hosts are added on first use
  # (that failure is what used to leave old gateways running); a node whose
  # key has changed is still refused.
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
       "$NODE" "bash -s -- $(printf '%q' "$DATA_DIR")" <<< "$STOP_REMOTE_GATEWAYS"; then
    echo "WARNING: could not check $NODE automatically. On $NODE, stop the gateway that uses"
    echo "         $DATA_DIR (usually: tmux kill-session -t $SESSION)."
    echo "         Until then two gateways share one jobs.db."
  fi
done

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
  "export VISION_DATA_DIR='$DATA_DIR'; export H3_ATTENTION_BACKEND=$(printf '%q' "$H3_ATTENTION_BACKEND"); cd '$API_DIR' && '$DATA_DIR/envs/gateway/bin/uvicorn' server:app --host '$HOST' --port $PORT; exec bash"
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
echo "H3 attention backend: ${H3_ATTENTION_BACKEND:-default (diffusers)}"
echo "Node:     $THIS_NODE   (forward port $PORT from THIS node)"
echo "Tunnel:   ssh -N -L $PORT:localhost:$PORT <you>@$THIS_NODE   (or let Cursor/VS Code forward port $PORT)"
echo "Attach:   tmux attach -t $SESSION   (Ctrl-b d to detach without stopping it)"
echo "Stop:     tmux kill-session -t $SESSION"
if [ -d "$API_DIR/ui/dist" ]; then
  echo "Web UI:   http://localhost:$PORT/ui/"
else
  echo "Web UI:   not built yet -- see ui/README.md (cd ui && npm install && npm run build), then re-run this script."
fi
