#!/usr/bin/env bash
# Thin wrapper so core.py's own cwd-relative internal lookups (e.g. its
# module-level load of rvc/lib/tools/tts_voices.json, which uses a bare
# relative path rather than one resolved against its own __file__) work
# correctly regardless of what directory the Slurm job actually starts in
# -- discovered via a real end-to-end test through the API, not assumed.
#
# Every argv element after this script's own path is passed through to
# core.py unchanged (as "$@", not a re-joined string), so model-api's
# common/run_pipeline_job.py can still find "--output-path" as an exact,
# standalone argument and rewrite it -- see services/rvc/dispatch.py.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"  # <repo>/models/rvc/runner/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
cd "$REPO_ROOT/third_party/Applio"
exec "$DATA_DIR/envs/rvc/bin/python" core.py "$@"
