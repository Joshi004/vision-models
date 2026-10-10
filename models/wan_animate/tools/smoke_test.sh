#!/bin/bash
# One-off smoke test for run_replace.py, run via sbatch on the Wan2.2 repo's
# own official replace-mode example inputs. Not part of the shipped
# pipeline -- see the setup plan's smoke-test step. Samples GPU memory
# alongside the run so real VRAM usage (not just sacct's coarser MaxRSS)
# is available afterward.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"  # <repo>/models/wan_animate/tools/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
cd "$REPO_ROOT"

WORK_DIR="$DATA_DIR/smoke_test/wan_animate"
mkdir -p "$WORK_DIR"

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

(
  while true; do
    printf '%s ' "$(date +%s)"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 5
  done
) > "$WORK_DIR/vram_sampler.log" &
SAMPLER_PID=$!

START=$(date +%s)
"$DATA_DIR/envs/wan_animate/bin/python" models/wan_animate/runner/run_replace.py \
  --video third_party/Wan2.2/examples/wan_animate/replace/video.mp4 \
  --image third_party/Wan2.2/examples/wan_animate/replace/image.jpeg \
  --work-dir "$WORK_DIR" \
  --output-path "$WORK_DIR/output.mp4" \
  --seed 10
RC=$?
END=$(date +%s)
echo "run_replace.py exit code: $RC, total wall time: $((END - START))s"

kill "$SAMPLER_PID" 2>/dev/null
wait "$SAMPLER_PID" 2>/dev/null

echo "--- peak VRAM used (MiB) ---"
awk '{print $2}' "$WORK_DIR/vram_sampler.log" | sort -n | tail -1

exit $RC
