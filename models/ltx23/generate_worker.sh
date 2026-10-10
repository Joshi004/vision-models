#!/usr/bin/env bash
# generate_worker.sh - Runs inside ONE Slurm task of a batch/variations job (one GPU
# per task). Invoked by generate.sh via `srun --ntasks=N ... bash generate_worker.sh
# ...`; not meant to be run directly.
#
# Args: BATCH_DIR TODO_FILE MARKER_FILE ITEM_LABEL PYTHON_BIN PIPELINE_ARGS...
#
#   BATCH_DIR    Directory holding the saved prompt record, <label>_NNN.mp4 outputs,
#                and logs/.
#   TODO_FILE    Lines of "NNN<TAB>SEED<TAB>prompt text" for items not yet done or
#                failed. SEED is empty for a normal batch scene, or an integer for a
#                variations run. Every task reads the same file and claims every
#                SLURM_NTASKS-th line (round robin by SLURM_PROCID) so no coordination
#                between tasks is needed.
#   MARKER_FILE  Touched immediately so the launcher can tell "workers started" apart
#                from "srun could not get an allocation" even if every item fails.
#   ITEM_LABEL   "scene" (batch mode) or "variant" (variations mode) -- the output
#                filename prefix, e.g. scene_001.mp4 or variant_001.mp4.
#   PYTHON_BIN   Path to the venv's python.
#   PIPELINE_ARGS... The pipeline module + model/common flags (--prompt, --output-path,
#                and, for variations, --seed are appended per item below).
#
# Per item: runs the pipeline writing to <label>_NNN.partial.mp4, then renames to
# <label>_NNN.mp4 on success. On failure, deletes the partial file and touches
# <label>_NNN.failed. On preemption (SIGTERM/SIGKILL), deletes the partial file and
# exits without creating a .failed marker, leaving the item for a later attempt.

set -uo pipefail  # no -e: one item's failure must not stop the rest of this task's items

BATCH_DIR="$1"
TODO_FILE="$2"
MARKER_FILE="$3"
ITEM_LABEL="$4"
PYTHON_BIN="$5"
shift 5
PIPELINE_ARGS=("$@")

LOG_DIR="$BATCH_DIR/logs"
TASK_ID="${SLURM_PROCID:-0}"
NTASKS="${SLURM_NTASKS:-1}"

# Signals a "started" mark for the launcher before doing any real work, so a job that
# starts but crashes immediately is still distinguishable from one that never started.
mkdir -p "$LOG_DIR" 2>/dev/null || true
touch "$MARKER_FILE" 2>/dev/null || true

log() {
  echo "[task $TASK_ID @ $(hostname)] $*"
}

# Slurm delivers the preemption signal to every process in the job step, so the python
# child below normally dies on its own (caught by the exit-code check in the loop). This
# trap is the backstop for the window between items (or during the final `mv`) when no
# child is running to catch the signal directly.
trap 'log "received termination signal, stopping (item in flight, if any, is left for a later attempt)"; exit 143' TERM INT

log "starting: $NTASKS task(s) total, this task handles every ${NTASKS}th ${ITEM_LABEL} starting at index $TASK_ID. GPU(s): ${CUDA_VISIBLE_DEVICES:-unknown}"

mapfile -t TODO_LINES < "$TODO_FILE"

for (( i = TASK_ID; i < ${#TODO_LINES[@]}; i += NTASKS )); do
  line="${TODO_LINES[$i]}"
  [[ -n "$line" ]] || continue

  num="${line%%$'\t'*}"
  rest="${line#*$'\t'}"
  seed="${rest%%$'\t'*}"
  prompt="${rest#*$'\t'}"
  item="${ITEM_LABEL}_${num}"

  final="$BATCH_DIR/${item}.mp4"
  if [[ -e "$final" ]]; then
    # Defensive re-check: a stale todo line from a previous attempt already resolved.
    log "${item}: already done, skipping"
    continue
  fi

  partial="$BATCH_DIR/${item}.partial.mp4"
  failed_marker="$BATCH_DIR/${item}.failed"
  log_file="$LOG_DIR/${item}.log"
  rm -f "$partial" "$failed_marker"

  item_args=("${PIPELINE_ARGS[@]}" --prompt "$prompt" --output-path "$partial")
  if [[ -n "$seed" ]]; then
    # Appended after the shared pipeline args, so it overrides any global --seed
    # (argparse keeps the last occurrence of a repeated flag).
    item_args+=(--seed "$seed")
  fi

  log "${item}: starting"
  "$PYTHON_BIN" "${item_args[@]}" > "$log_file" 2>&1
  rc=$?

  if (( rc == 0 )); then
    mv -f "$partial" "$final"
    log "${item}: done -> ${final}"
  elif (( rc == 137 || rc == 143 )); then
    # SIGKILL / SIGTERM -- almost certainly preemption, not a real failure.
    rm -f "$partial"
    log "${item}: interrupted (exit $rc), leaving for a later attempt"
    exit "$rc"
  else
    rm -f "$partial"
    touch "$failed_marker"
    log "${item}: FAILED (exit $rc, see $log_file)"
  fi
done

log "no more ${ITEM_LABEL}s assigned to this task"
