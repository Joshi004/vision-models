#!/usr/bin/env bash
# build_env.sh - Build one Python environment into $VISION_DATA_DIR/envs/<name>
# from its frozen lock file.
#
# Usage:  scripts/build_env.sh <name>
#   <name> is one of: gateway ltx23 ltx25 wan_animate rvc parakeet breeze_tts
#
# Lock file (the single source of truth for that environment):
#   gateway          -> gateway/requirements.lock.txt
#   everything else  -> models/<name>/env/requirements.lock.txt
# Its first line must be:  # python: <major.minor>
# Placeholders substituted before install:
#   @REPO_ROOT@  the repository root (used for editable installs from third_party/)
#   @DATA_DIR@   $VISION_DATA_DIR (used for local wheels)
#
# Optional hook: models/<name>/env/post_install.sh is run after the install,
# with REPO_ROOT, DATA_DIR and ENV_DIR exported.
#
# Safety: refuses to touch an environment that already exists. To rebuild one,
# move or delete it yourself first.
set -euo pipefail

if [ $# -ne 1 ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 2
fi

NAME="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
ENV_DIR="$DATA_DIR/envs/$NAME"

if [ "$NAME" = "gateway" ]; then
  LOCK="$REPO_ROOT/gateway/requirements.lock.txt"
  HOOK=""
else
  LOCK="$REPO_ROOT/models/$NAME/env/requirements.lock.txt"
  HOOK="$REPO_ROOT/models/$NAME/env/post_install.sh"
fi

[ -f "$LOCK" ] || { echo "No lock file: $LOCK" >&2; exit 1; }
if [ -e "$ENV_DIR" ]; then
  echo "Refusing to overwrite existing environment: $ENV_DIR" >&2
  echo "Move or delete it first if you really want to rebuild it." >&2
  exit 1
fi

PYTHON_VERSION="$(sed -n 's/^# python: *//p' "$LOCK" | head -1)"
[ -n "$PYTHON_VERSION" ] || { echo "First line of $LOCK must be '# python: X.Y'" >&2; exit 1; }

RESOLVED_LOCK="$(mktemp)"
trap 'rm -f "$RESOLVED_LOCK"' EXIT
sed "s#@REPO_ROOT@#$REPO_ROOT#g; s#@DATA_DIR@#$DATA_DIR#g" "$LOCK" > "$RESOLVED_LOCK"

echo "== Building '$NAME' (python $PYTHON_VERSION) in $ENV_DIR"
mkdir -p "$DATA_DIR/envs"
# Some locks add a PyTorch CUDA wheel index next to PyPI; with several indexes,
# look for each exact pin in all of them instead of only the first index that
# has the package name.
export UV_INDEX_STRATEGY=unsafe-best-match
uv venv --python "$PYTHON_VERSION" "$ENV_DIR"
# --no-deps: the lock is a complete freeze, so install exactly what it lists
# and let `uv pip check` below prove it is consistent.
uv pip install --python "$ENV_DIR/bin/python" --no-deps -r "$RESOLVED_LOCK"

if [ -n "$HOOK" ] && [ -f "$HOOK" ]; then
  echo "== Running post-install hook: $HOOK"
  REPO_ROOT="$REPO_ROOT" DATA_DIR="$DATA_DIR" ENV_DIR="$ENV_DIR" bash "$HOOK"
fi

echo "== Consistency check"
uv pip check --python "$ENV_DIR/bin/python"
echo "== Done: $ENV_DIR"
