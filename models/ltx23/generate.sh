#!/usr/bin/env bash
# generate.sh - Generate videos (synchronized video + audio) using LTX-2.3.
#
# Four ways to run it:
#
#   Interactive (no arguments):
#     ./generate.sh
#     Walks you through the prompt, an optional first-frame image, orientation,
#     duration, quality, and GPUs/variations, then shows a summary to confirm
#     before submitting anything.
#
#   Single video:
#     ./generate.sh "PROMPT" [--quality] [--output-path FILE] [PIPELINE_FLAGS...]
#     ./generate.sh --prompt-file FILE [--quality] [--output-path FILE] [PIPELINE_FLAGS...]
#     (A positional PROMPT must not start with "--", or it will be parsed as a flag.)
#
#   Variations -- the same prompt, multiple seeds, to compare outputs:
#     ./generate.sh "PROMPT" --variations N [--base-seed S] [--gpus G] [--quality] [--output-dir DIR] [PIPELINE_FLAGS...]
#     ./generate.sh --prompt-file FILE --variations N ...
#     Seeds used are S, S+1, ..., S+N-1 (default base seed 10). --gpus controls
#     whether they run one at a time (1, default) or in parallel (up to 8).
#
#   Batch -- many different scenes, one worker per GPU:
#     ./generate.sh --prompts-file FILE --gpus N [--quality] [--output-dir DIR] [PIPELINE_FLAGS...]
#
# Pipeline choice (all modes):
#   (default)   Fast distilled pipeline (ltx_pipelines.distilled): 8-step, lowest latency.
#   --quality   Production-quality two-stage pipeline (ltx_pipelines.ti2vid_two_stages):
#               full dev transformer + distilled LoRA refinement. Slower, higher quality.
#
# Video shape (all modes):
#   --duration SECONDS    Length of the video, converted to the nearest valid frame
#                         count (--num-frames, which must be 8*K + 1) at --frame-rate
#                         (default 24fps). Cannot be combined with --num-frames.
#   --orientation X       "landscape" (1920x1088, default) or "portrait" (1088x1920) --
#                         YouTube/Shorts-style sizes. Cannot be combined with
#                         --height/--width; pass those directly for any other size.
#   --image-fit X         How an --image is fit to the video frame: "crop" (default)
#                         center-crops it to fill the frame; "pad" scales it down to
#                         fit within the frame and adds black bars on the sides that
#                         would otherwise be cropped off. Only matters if --image is
#                         also given. Every --image is also fully decoded up front
#                         (not just its header), so a truncated/corrupt file is
#                         caught before anything is submitted to Slurm.
#
# Prompt-file flags:
#   --prompt-file FILE    A single prompt, for one video or --variations. The whole
#                         file is read and its whitespace (including newlines)
#                         collapsed into one paragraph, so you can wrap a long prompt
#                         across editor lines for readability.
#   --prompts-file FILE   Many scenes for batch mode. Scenes are separated by one or
#                         more BLANK LINES; each scene's own lines are joined the same
#                         way as --prompt-file. Lines starting with '#' are comments,
#                         dropped wherever they appear (including inside a scene).
#
# Variations and batch share these flags:
#   --gpus N              Workers to run in parallel, 1-8 (default: 1). Each worker is
#                         an independent process on its own GPU -- this is parallel
#                         single-GPU generation, not multi-GPU inference for one video.
#                         The launcher tries to fit all N workers on one node first
#                         (faster model loading via the shared page cache), then
#                         spreads across nodes if that isn't available.
#   --output-dir DIR      Resume a previous batch or variations run: only unfinished
#                         items are (re)run. Requires the same prompt(s) as the
#                         original run, so numbering lines up (--variations may differ
#                         between the original run and a resume).
#
#   Output: outputs/<name>-<timestamp>/scene_NNN.mp4 (batch) or variant_NNN.mp4
#   (variations), plus logs/<item>.log per item and a saved record of the prompt(s)
#   used (prompts.txt for batch; prompt.txt + base_seed.txt for variations).
#
#   `background` is a low-priority partition: any job here can be preempted at any
#   time for higher-priority work, with no grace period. Batch/variations mode
#   handles this itself: on preemption it retries only the unfinished items, up to
#   LTX_MAX_ATTEMPTS times, and prints the exact resume command if items are still
#   unfinished when it gives up. Items that fail with a real error (not preemption)
#   are not retried automatically -- rerunning a broken prompt three times wastes GPU
#   time. Resuming manually (via --output-dir) does give failed items a fresh
#   attempt, in case the cause was transient.
#
# Any flag this wrapper does not recognize is passed straight through to the underlying
# `python -m ltx_pipelines.*` CLI. Useful examples:
#   --seed 7
#   --num-frames 241                  (must be 8*K + 1)
#   --height 1024 --width 1536        (must be divisible by 64)
#   --image path/to/image.jpg 0 1.0   (image-to-video conditioning: PATH FRAME_IDX STRENGTH)
#   --enhance-prompt
#   --quantization fp8-cast --offload cpu   (lower GPU memory use)
#
# Paths: a relative path given to --prompt-file, --prompts-file, --image,
# --output-path, or --output-dir (or typed into the interactive wizard) is resolved
# from this project's root directory (where this script lives), not from wherever
# you happened to run it from. Absolute paths and ~/... are unaffected.
#
# Run with the underlying CLI's own --help to see the full flag set, e.g.:
#   LTX-2/.venv/bin/python -m ltx_pipelines.distilled --help
#
# Environment overrides for the Slurm allocation:
#   LTX_PARTITION      (default: background)
#   LTX_TIME           (default: 01:00:00 for a single video; auto-computed per attempt
#                       in batch/variations mode from items remaining and worker count)
#   LTX_MEM_PER_GPU    (default: 96G)
#   LTX_CPUS           (default: 8, per GPU)
#   LTX_SAME_NODE_WAIT (default: 60, seconds to wait for a single free node)
#   LTX_MAX_ATTEMPTS   (default: 3, batch/variations mode retries after preemption)
#
# This script blocks until generation finishes (single video), or until every item is
# done, permanently failed, or attempts run out (batch/variations). For long runs,
# launch it inside `tmux` so it survives an SSH disconnect.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

# Every relative path this script (or the wizard, or the underlying pipeline job)
# resolves from here on -- --prompt-file, --prompts-file, --image, --output-path,
# --output-dir, and anything else -- is relative to the project root, not to
# whatever directory the caller happened to be in.
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # <repo>/models/ltx23/<this file>
DATA_DIR="${VISION_DATA_DIR:-$REPO_ROOT/../vision-data}"
LTX_REPO="$REPO_ROOT/third_party/ltx-2.3"
MODEL_DIR="$DATA_DIR/weights/ltx23/ltx-2.3"
GEMMA_DIR="$DATA_DIR/weights/ltx23/gemma-3-12b"
OUTPUT_DIR="$DATA_DIR/outputs/ltx23"
PYTHON="$DATA_DIR/envs/ltx23/bin/python"
WORKER="$SCRIPT_DIR/generate_worker.sh"
INTERACTIVE="$SCRIPT_DIR/generate_interactive.sh"

PARTITION="${LTX_PARTITION:-background}"
TIME_LIMIT_OVERRIDE="${LTX_TIME:-}"
MEM_PER_GPU="${LTX_MEM_PER_GPU:-96G}"
CPUS="${LTX_CPUS:-8}"
SAME_NODE_WAIT="${LTX_SAME_NODE_WAIT:-60}"
MAX_ATTEMPTS="${LTX_MAX_ATTEMPTS:-3}"

DEFAULT_FRAME_RATE=24
LANDSCAPE_WIDTH=1920
LANDSCAPE_HEIGHT=1088
PORTRAIT_WIDTH=1088
PORTRAIT_HEIGHT=1920

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
  # $SELF (always absolute), not $0: after the `cd` above, a relative $0 (e.g. this
  # script called as "ltx-2.3/generate.sh" from one directory up) would no longer
  # point at this file.
  awk '/^# generate.sh/{p=1} /^set -euo pipefail/{exit} p' "$SELF" | sed 's/^# \{0,1\}//'
}

abs_path() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")")
  fi
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Error: expected model file missing: $1" >&2
    exit 1
  fi
}

# Expands a leading "~" or "~/..." to $HOME and strips one matching pair of
# surrounding quotes (some terminals/file managers add quotes when a path is typed
# or dragged in). Used for path-like values typed into the interactive wizard, and
# defensively for CLI flag values too. Does NOT require the path to exist.
expand_user_path() {
  local p="$1"
  if [[ ${#p} -ge 2 ]]; then
    if [[ "${p:0:1}" == '"' && "${p: -1}" == '"' ]] || [[ "${p:0:1}" == "'" && "${p: -1}" == "'" ]]; then
      p="${p:1:-1}"
    fi
  fi
  case "$p" in
    "~") p="$HOME" ;;
    "~/"*) p="$HOME/${p#\~/}" ;;
  esac
  printf '%s\n' "$p"
}

# Best-effort absolute-looking path for an error message. Unlike abs_path, this
# does not require the path (or its parent directory) to exist.
display_path() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$SCRIPT_DIR" "$p"
  fi
}

# Collapses all whitespace (including newlines) in $1 to single spaces and trims
# the ends. Used to turn a multi-line prompt file, or a multi-line batch scene block,
# into one flowing paragraph.
collapse_whitespace() {
  local -a words
  read -r -d '' -a words <<< "$1" || true
  echo "${words[*]}"
}

read_prompt_file() {
  local file
  file="$(expand_user_path "$1")"
  if [[ ! -f "$file" ]]; then
    echo "Error: prompt file not found: $(display_path "$file")" >&2
    echo "(Relative paths are resolved from the project root: $SCRIPT_DIR)" >&2
    exit 1
  fi
  collapse_whitespace "$(cat "$file")"
}

# Splits $1 (already CRLF-stripped) into prompts, one per blank-line-separated block.
# Lines starting with '#' (leading whitespace tolerated) are dropped wherever they
# appear; the remaining lines of each block are joined into one paragraph via
# collapse_whitespace. A block that is empty after comment-stripping (e.g. it was only
# comments) produces no output. Prints one resulting prompt per line on stdout.
parse_prompt_blocks() {
  local text="$1"
  local -a lines block_lines
  local line joined
  mapfile -t lines <<< "$text"
  block_lines=()
  for line in "${lines[@]}" ""; do  # the trailing "" flushes the final block
    if [[ "$line" =~ ^[[:space:]]*# ]]; then
      continue
    fi
    if [[ -z "${line//[[:space:]]/}" ]]; then
      if (( ${#block_lines[@]} > 0 )); then
        joined="$(collapse_whitespace "${block_lines[*]}")"
        printf '%s\n' "$joined"
        block_lines=()
      fi
      continue
    fi
    block_lines+=("$line")
  done
}

# Compares two arrays (by variable NAME, via nameref) element-by-element. Used to
# verify a --output-dir resume is against the exact same prompt(s) as the original run.
prompts_match() {
  local -n arr1="$1" arr2="$2"
  if [[ "${#arr1[@]}" != "${#arr2[@]}" ]]; then
    return 1
  fi
  local i
  for i in "${!arr1[@]}"; do
    if [[ "${arr1[$i]}" != "${arr2[$i]}" ]]; then
      return 1
    fi
  done
  return 0
}

# Matches a plain non-negative decimal (no sign, no exponent): "5", "5.0", ".5".
is_nonnegative_number() {
  [[ "$1" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]]
}

# Same, but rejects zero -- for values that must be strictly positive (durations,
# frame rates).
is_positive_number() {
  is_nonnegative_number "$1" && awk -v n="$1" 'BEGIN { exit !(n > 0) }'
}

# SECONDS FPS -> nearest valid LTX-2 frame count (8*K + 1, K >= 1). Prints just the
# frame count.
frames_for_duration() {
  local seconds="$1" fps="$2"
  awk -v s="$seconds" -v r="$fps" 'BEGIN {
    k = int(s * r / 8 + 0.5)
    if (k < 1) k = 1
    print 8 * k + 1
  }'
}

# Rounds an arbitrary frame count to the nearest valid 8*K + 1 (K >= 1).
snap_frames() {
  local frames="$1"
  awk -v f="$frames" 'BEGIN {
    k = int((f - 1) / 8 + 0.5)
    if (k < 1) k = 1
    print 8 * k + 1
  }'
}

# PATH -> "portrait" or "landscape" on stdout. Prints nothing and returns failure if
# the image can't be read (missing Pillow, unsupported format, etc.) -- callers
# treat that as "unknown" rather than a hard error, since this is only used for an
# advisory warning.
image_orientation() {
  "$PYTHON" -c '
import sys
from PIL import Image
with Image.open(sys.argv[1]) as im:
    w, h = im.size
print("portrait" if h > w else "landscape")
' "$1" 2>/dev/null
}

# Fully decodes PATH to catch a truncated/corrupt file that a header read alone
# would miss (Image.open() only parses the header lazily; a cut-off file can still
# report valid-looking dimensions). Prints nothing on success. On failure, prints
# "ExceptionType: message" on stdout and returns 1 -- callers fold that into their
# own error message.
check_image_decodes() {
  "$PYTHON" -c '
import sys
from PIL import Image
try:
    with Image.open(sys.argv[1]) as im:
        im.load()
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)
' "$1"
}

# Scales SRC down (if needed) to fit within WIDTHxHEIGHT preserving aspect ratio, then
# pads it with black bars to exactly WIDTHxHEIGHT -- the "pad" alternative to the
# pipeline's own center-crop, for --image-fit pad. The original file is never
# modified: the result is written under outputs/prepared-images/, named from SRC's
# own basename plus a content hash and the target size, so a different upload under
# the same filename can't collide with a copy an earlier run may still be using.
# Written via a temp file + atomic rename, so a job can never read a half-written
# copy. Prints the new file's path on stdout; returns non-zero (nothing printed) if
# it could not be created.
pad_image_to_fit() {
  local src="$1" width="$2" height="$3"
  local prepared_dir base hash dst tmp
  prepared_dir="$OUTPUT_DIR/prepared-images"
  mkdir -p "$prepared_dir"
  base="$(basename "$src")"
  base="${base%.*}"
  hash="$(sha256sum "$src" | cut -c1-8)"
  dst="$prepared_dir/${base}-${hash}-${width}x${height}-pad.png"
  tmp="$dst.tmp.$$"
  "$PYTHON" -c '
import os
import sys
from PIL import Image

src, tmp, dst, width, height = sys.argv[1:6]
width, height = int(width), int(height)
with Image.open(src) as source:
    image = source.convert("RGB")
scale = min(width / image.width, height / image.height)
new_size = (round(image.width * scale), round(image.height * scale))
image = image.resize(new_size, Image.Resampling.LANCZOS)
canvas = Image.new("RGB", (width, height), (0, 0, 0))
canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
canvas.save(tmp, format="PNG")
os.replace(tmp, dst)
' "$src" "$tmp" "$dst" "$width" "$height" || return 1
  printf '%s\n' "$dst"
}

# Whether EXTRA_ARGS contains flag NAME (e.g. "--num-frames") anywhere.
extra_args_has() {
  local name="$1" i
  for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

# Prints the value following the LAST occurrence of flag NAME in EXTRA_ARGS (empty
# if absent). "Last wins" matches argparse's behavior for a repeated flag.
extra_arg_value() {
  local name="$1" i value=""
  for ((i = 0; i < ${#EXTRA_ARGS[@]} - 1; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "$name" ]]; then
      value="${EXTRA_ARGS[$((i + 1))]}"
    fi
  done
  printf '%s' "$value"
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

PROMPT=""
PROMPT_FILE_ARG=""
PROMPTS_FILE=""
MODE="fast"
OUTPUT=""
OUTPUT_DIR_ARG=""
GPUS=1
VARIATIONS=1
BASE_SEED=10
DURATION=""
ORIENTATION=""
IMAGE_FIT="crop"
EXTRA_ARGS=()

if [[ $# -eq 0 ]]; then
  # shellcheck source=generate_interactive.sh
  source "$INTERACTIVE"
else
  if [[ "$1" != --* ]]; then
    PROMPT="$1"
    shift
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --quality)
        MODE="quality"
        shift
        ;;
      --duration)
        if [[ $# -lt 2 ]]; then
          echo "Error: --duration requires a value." >&2
          exit 1
        fi
        DURATION="$2"
        shift 2
        ;;
      --orientation)
        if [[ $# -lt 2 ]]; then
          echo "Error: --orientation requires a value." >&2
          exit 1
        fi
        ORIENTATION="$2"
        shift 2
        ;;
      --image-fit)
        if [[ $# -lt 2 ]]; then
          echo "Error: --image-fit requires a value." >&2
          exit 1
        fi
        IMAGE_FIT="$2"
        shift 2
        ;;
      --output-path)
        if [[ $# -lt 2 ]]; then
          echo "Error: --output-path requires a value." >&2
          exit 1
        fi
        OUTPUT="$2"
        shift 2
        ;;
      --prompt-file)
        if [[ $# -lt 2 ]]; then
          echo "Error: --prompt-file requires a value." >&2
          exit 1
        fi
        PROMPT_FILE_ARG="$2"
        shift 2
        ;;
      --prompts-file)
        if [[ $# -lt 2 ]]; then
          echo "Error: --prompts-file requires a value." >&2
          exit 1
        fi
        PROMPTS_FILE="$2"
        shift 2
        ;;
      --variations)
        if [[ $# -lt 2 ]]; then
          echo "Error: --variations requires a value." >&2
          exit 1
        fi
        VARIATIONS="$2"
        shift 2
        ;;
      --base-seed)
        if [[ $# -lt 2 ]]; then
          echo "Error: --base-seed requires a value." >&2
          exit 1
        fi
        BASE_SEED="$2"
        shift 2
        ;;
      --gpus)
        if [[ $# -lt 2 ]]; then
          echo "Error: --gpus requires a value." >&2
          exit 1
        fi
        GPUS="$2"
        shift 2
        ;;
      --output-dir)
        if [[ $# -lt 2 ]]; then
          echo "Error: --output-dir requires a value." >&2
          exit 1
        fi
        OUTPUT_DIR_ARG="$2"
        shift 2
        ;;
      *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
  done
fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Expand a leading "~/" and strip accidental surrounding quotes on user-supplied
# paths. --prompt-file goes through read_prompt_file(), which does this itself.
[[ -n "$PROMPTS_FILE" ]] && PROMPTS_FILE="$(expand_user_path "$PROMPTS_FILE")"
[[ -n "$OUTPUT" ]] && OUTPUT="$(expand_user_path "$OUTPUT")"
[[ -n "$OUTPUT_DIR_ARG" ]] && OUTPUT_DIR_ARG="$(expand_user_path "$OUTPUT_DIR_ARG")"

PROMPT_SOURCES=0
[[ -n "$PROMPT" ]] && PROMPT_SOURCES=$((PROMPT_SOURCES + 1))
[[ -n "$PROMPT_FILE_ARG" ]] && PROMPT_SOURCES=$((PROMPT_SOURCES + 1))
[[ -n "$PROMPTS_FILE" ]] && PROMPT_SOURCES=$((PROMPT_SOURCES + 1))

if (( PROMPT_SOURCES > 1 )); then
  echo "Error: provide exactly one of: a prompt, --prompt-file, or --prompts-file." >&2
  exit 1
fi
if (( PROMPT_SOURCES == 0 )); then
  echo "Error: provide a prompt, --prompt-file FILE, or --prompts-file FILE." >&2
  exit 1
fi

if [[ -n "$PROMPT_FILE_ARG" ]]; then
  PROMPT="$(read_prompt_file "$PROMPT_FILE_ARG")"
  if [[ -z "$PROMPT" ]]; then
    echo "Error: $PROMPT_FILE_ARG is empty." >&2
    exit 1
  fi
fi

if [[ ! "$GPUS" =~ ^[0-9]+$ ]] || (( GPUS < 1 || GPUS > 8 )); then
  echo "Error: --gpus must be an integer from 1 to 8 (got: $GPUS)." >&2
  exit 1
fi
if [[ ! "$VARIATIONS" =~ ^[0-9]+$ ]] || (( VARIATIONS < 1 )); then
  echo "Error: --variations must be a positive integer (got: $VARIATIONS)." >&2
  exit 1
fi
if [[ ! "$BASE_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "Error: --base-seed must be an integer (got: $BASE_SEED)." >&2
  exit 1
fi

BATCH_MODE=false
VARIATIONS_MODE=false
if [[ -n "$PROMPTS_FILE" ]]; then
  BATCH_MODE=true
elif (( VARIATIONS > 1 )); then
  VARIATIONS_MODE=true
fi

if $BATCH_MODE && (( VARIATIONS > 1 )); then
  echo "Error: --variations is not valid with --prompts-file (variations apply to a single prompt)." >&2
  exit 1
fi

if $BATCH_MODE || $VARIATIONS_MODE; then
  if [[ -n "$OUTPUT" ]]; then
    echo "Error: --output-path is not valid with --prompts-file or --variations; use --output-dir instead." >&2
    exit 1
  fi
else
  if (( GPUS > 1 )); then
    echo "Error: --gpus > 1 requires --prompts-file or --variations (a single video only uses one GPU)." >&2
    exit 1
  fi
  if [[ -n "$OUTPUT_DIR_ARG" ]]; then
    echo "Error: --output-dir is only valid with --prompts-file or --variations." >&2
    exit 1
  fi
fi

if $BATCH_MODE && [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Error: prompts file not found: $(display_path "$PROMPTS_FILE")" >&2
  echo "(Relative paths are resolved from the project root: $SCRIPT_DIR)" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Error: $PYTHON not found." >&2
  echo "Run 'uv sync --frozen --python 3.12' inside $LTX_REPO first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Video shape: --duration -> --num-frames, --orientation -> --height/--width, and
# --image path checking/absolutizing (existence, full decode, --image-fit).
# Everything here appends to EXTRA_ARGS, so single-video mode, batch mode,
# variations mode, and the resume command printed at the end all pick it up
# automatically via the existing `PIPELINE_ARGS+=("${EXTRA_ARGS[@]}")` below -- no
# further plumbing needed.
# ---------------------------------------------------------------------------

if [[ -n "$DURATION" ]] && ! is_positive_number "$DURATION"; then
  echo "Error: --duration must be a positive number of seconds (got: $DURATION)." >&2
  exit 1
fi
if [[ -n "$ORIENTATION" && "$ORIENTATION" != "landscape" && "$ORIENTATION" != "portrait" ]]; then
  echo "Error: --orientation must be 'landscape' or 'portrait' (got: $ORIENTATION)." >&2
  exit 1
fi
if [[ "$IMAGE_FIT" != "crop" && "$IMAGE_FIT" != "pad" ]]; then
  echo "Error: --image-fit must be 'crop' or 'pad' (got: $IMAGE_FIT)." >&2
  exit 1
fi

FRAME_RATE="$(extra_arg_value --frame-rate)"
FRAME_RATE="${FRAME_RATE:-$DEFAULT_FRAME_RATE}"
if ! is_positive_number "$FRAME_RATE"; then
  echo "Error: --frame-rate must be a positive number (got: $FRAME_RATE)." >&2
  exit 1
fi

if [[ -n "$DURATION" ]] && extra_args_has --num-frames; then
  echo "Error: use either --duration or --num-frames, not both." >&2
  exit 1
fi
if [[ -n "$ORIENTATION" ]] && { extra_args_has --height || extra_args_has --width; }; then
  echo "Error: use either --orientation or --height/--width, not both." >&2
  exit 1
fi

# --- Frames ---------------------------------------------------------------

if [[ -n "$DURATION" ]]; then
  FRAMES="$(frames_for_duration "$DURATION" "$FRAME_RATE")"
  EXTRA_ARGS+=(--num-frames "$FRAMES")
  ACTUAL_SECONDS="$(awk -v f="$FRAMES" -v r="$FRAME_RATE" 'BEGIN { printf "%.2f", (f - 1) / r }')"
  echo "Length:     ${DURATION}s at ${FRAME_RATE}fps -> $FRAMES frames (${ACTUAL_SECONDS}s)"
  EFFECTIVE_NUM_FRAMES="$FRAMES"
elif extra_args_has --num-frames; then
  REQUESTED_FRAMES="$(extra_arg_value --num-frames)"
  if [[ ! "$REQUESTED_FRAMES" =~ ^[0-9]+$ ]] || (( (REQUESTED_FRAMES - 1) % 8 != 0 )) || (( REQUESTED_FRAMES < 9 )); then
    SNAPPED_FRAMES="$(snap_frames "$REQUESTED_FRAMES")"
    echo "Note: --num-frames $REQUESTED_FRAMES is not 8*K+1; using $SNAPPED_FRAMES instead." >&2
    for ((i = 0; i < ${#EXTRA_ARGS[@]} - 1; i++)); do
      if [[ "${EXTRA_ARGS[$i]}" == "--num-frames" ]]; then
        EXTRA_ARGS[$((i + 1))]="$SNAPPED_FRAMES"
      fi
    done
    REQUESTED_FRAMES="$SNAPPED_FRAMES"
  fi
  EFFECTIVE_NUM_FRAMES="$REQUESTED_FRAMES"
else
  EFFECTIVE_NUM_FRAMES=121  # the pipelines' own default (LTX_2_3_PARAMS.num_frames)
fi

if [[ -n "$DURATION" ]] && awk -v d="$DURATION" 'BEGIN { exit !(d > 20) }'; then
  echo "Note: ${DURATION}s is longer than has been tested on this setup; if generation" >&2
  echo "      fails with an out-of-memory error, try adding --quantization fp8-cast." >&2
fi

# --- Size -------------------------------------------------------------------

HW_INJECTED=false
if ! extra_args_has --height && ! extra_args_has --width; then
  EFFECTIVE_ORIENTATION="${ORIENTATION:-landscape}"
  if [[ "$EFFECTIVE_ORIENTATION" == "portrait" ]]; then
    SHAPE_WIDTH="$PORTRAIT_WIDTH"
    SHAPE_HEIGHT="$PORTRAIT_HEIGHT"
  else
    SHAPE_WIDTH="$LANDSCAPE_WIDTH"
    SHAPE_HEIGHT="$LANDSCAPE_HEIGHT"
  fi
  EXTRA_ARGS+=(--width "$SHAPE_WIDTH" --height "$SHAPE_HEIGHT")
  echo "Size:       $EFFECTIVE_ORIENTATION ${SHAPE_WIDTH}x${SHAPE_HEIGHT}"
  HW_INJECTED=true
fi
# else: the conflict check above already guarantees --orientation wasn't also
# given, so the user is asking for an explicit, possibly non-landscape/portrait
# size -- leave --height/--width exactly as given.

# --- Image(s): existence check + absolute path + decode check + fit ---------

# Target size for --image-fit pad, if any --image is present below. Known directly
# when generate.sh injected --width/--height itself (HW_INJECTED); otherwise
# whatever the user passed via --width/--height directly, which may be incomplete
# (only one of the two) -- pad mode checks for that itself, below.
if $HW_INJECTED; then
  IMG_TARGET_WIDTH="$SHAPE_WIDTH"
  IMG_TARGET_HEIGHT="$SHAPE_HEIGHT"
else
  IMG_TARGET_WIDTH="$(extra_arg_value --width)"
  IMG_TARGET_HEIGHT="$(extra_arg_value --height)"
fi

for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[$i]}" == "--image" ]]; then
    img_idx=$((i + 1))
    if (( img_idx >= ${#EXTRA_ARGS[@]} )) || [[ "${EXTRA_ARGS[$img_idx]}" == --* ]]; then
      echo "Error: --image requires a path, frame index, and strength: --image PATH FRAME_IDX STRENGTH [CRF]." >&2
      exit 1
    fi
    img_path="$(expand_user_path "${EXTRA_ARGS[$img_idx]}")"
    if [[ ! -f "$img_path" ]]; then
      echo "Error: image not found: $(display_path "$img_path")" >&2
      echo "(Relative paths are resolved from the project root: $SCRIPT_DIR)" >&2
      exit 1
    fi
    img_abs="$(abs_path "$img_path")"

    # Image.open() alone only reads the header, which survives truncation -- fully
    # decode it here so a corrupt/incomplete file (e.g. an interrupted upload) is
    # caught in a second on the login node, not after a Slurm job has already spent
    # minutes loading models before reaching this same file.
    decode_error="$(check_image_decodes "$img_abs" || true)"
    if [[ -n "$decode_error" ]]; then
      echo "Error: can't read image $(display_path "$img_abs") ($decode_error)." >&2
      echo "The file is probably incomplete or corrupt (e.g. an interrupted upload/copy). Re-upload it and try again." >&2
      exit 1
    fi

    if [[ "$IMAGE_FIT" == "pad" ]]; then
      if [[ -z "$IMG_TARGET_WIDTH" || -z "$IMG_TARGET_HEIGHT" ]]; then
        echo "Error: --image-fit pad needs a known target size; pass --orientation, or both --height and --width, alongside --image." >&2
        exit 1
      fi
      if ! img_abs="$(pad_image_to_fit "$img_abs" "$IMG_TARGET_WIDTH" "$IMG_TARGET_HEIGHT")"; then
        echo "Error: failed to create a padded copy of $(display_path "$img_abs")." >&2
        exit 1
      fi
      echo "Image:      padded to fit ${IMG_TARGET_WIDTH}x${IMG_TARGET_HEIGHT}: $img_abs"
    fi

    EXTRA_ARGS[$img_idx]="$img_abs"
    # Padding already made the image exactly the target size, so the pipeline's own
    # center-crop (below) has nothing left to crop -- only warn about it in crop mode.
    if $HW_INJECTED && [[ "$IMAGE_FIT" == "crop" ]]; then
      img_orient="$(image_orientation "$img_abs" || true)"
      if [[ -n "$img_orient" && "$img_orient" != "$EFFECTIVE_ORIENTATION" ]]; then
        echo "Warning: $img_abs looks $img_orient but the video is $EFFECTIVE_ORIENTATION ($SHAPE_WIDTH x $SHAPE_HEIGHT); it will be center-cropped to fit." >&2
      fi
    fi
  fi
done

# ---------------------------------------------------------------------------
# Model files (same for all modes)
# ---------------------------------------------------------------------------

DISTILLED_CKPT="$MODEL_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
DEV_CKPT="$MODEL_DIR/ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA="$MODEL_DIR/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
UPSCALER="$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

require_file "$UPSCALER"
require_file "$GEMMA_DIR/config.json"

if [[ "$MODE" == "quality" ]]; then
  require_file "$DEV_CKPT"
  require_file "$DISTILLED_LORA"
  PIPELINE_ARGS=(-m ltx_pipelines.ti2vid_two_stages
    --checkpoint-path "$DEV_CKPT"
    --distilled-lora "$DISTILLED_LORA" 0.8)
else
  require_file "$DISTILLED_CKPT"
  PIPELINE_ARGS=(-m ltx_pipelines.distilled
    --distilled-checkpoint-path "$DISTILLED_CKPT")
fi

PIPELINE_ARGS+=(
  --spatial-upsampler-path "$UPSCALER"
  --gemma-root "$GEMMA_DIR"
)
PIPELINE_ARGS+=("${EXTRA_ARGS[@]}")

# ---------------------------------------------------------------------------
# Single-video mode
# ---------------------------------------------------------------------------

if ! $BATCH_MODE && ! $VARIATIONS_MODE; then
  if [[ -z "$OUTPUT" ]]; then
    TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
    OUTPUT="$OUTPUT_DIR/${TIMESTAMP}-${MODE}.mp4"
  fi
  mkdir -p "$(dirname "$OUTPUT")"

  TIME_LIMIT="${TIME_LIMIT_OVERRIDE:-01:00:00}"
  SINGLE_ARGS=("${PIPELINE_ARGS[@]}" --prompt "$PROMPT" --output-path "$OUTPUT")

  echo "Mode:       $MODE"
  echo "Output:     $OUTPUT"
  echo "Slurm:      partition=$PARTITION  time=$TIME_LIMIT  mem-per-gpu=$MEM_PER_GPU  cpus=$CPUS  gres=gpu:1"
  echo "Submitting srun job (this blocks until the video is done)..."
  echo

  HF_HUB_OFFLINE=1 srun \
    --partition="$PARTITION" \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS" \
    --mem-per-gpu="$MEM_PER_GPU" \
    --time="$TIME_LIMIT" \
    --job-name="ltx23-$MODE" \
    "$PYTHON" "${SINGLE_ARGS[@]}"

  echo
  if [[ -f "$OUTPUT" ]]; then
    echo "Done. Video written to: $OUTPUT"
    exit 0
  else
    echo "srun finished but $OUTPUT was not created. Check the output above for errors." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Batch / variations mode setup (shared engine)
# ---------------------------------------------------------------------------

ITEM_LABEL=""
RAW_PROMPTS=()
PER_ITEM_SEEDS=()

if $BATCH_MODE; then
  ITEM_LABEL="scene"
  PROMPTS_FILE="$(abs_path "$PROMPTS_FILE")"

  FILE_TEXT="$(sed 's/\r$//' "$PROMPTS_FILE")"
  mapfile -t RAW_PROMPTS < <(parse_prompt_blocks "$FILE_TEXT")
  NUM_SCENES=${#RAW_PROMPTS[@]}
  if (( NUM_SCENES == 0 )); then
    echo "Error: no prompts found in $PROMPTS_FILE (after skipping blank lines and '#' comments)." >&2
    exit 1
  fi
  for (( i = 0; i < NUM_SCENES; i++ )); do
    PER_ITEM_SEEDS+=("")
  done

  if [[ -n "$OUTPUT_DIR_ARG" ]]; then
    BATCH_DIR="$(abs_path "$OUTPUT_DIR_ARG")"
    if [[ ! -d "$BATCH_DIR" ]]; then
      echo "Error: --output-dir does not exist: $BATCH_DIR" >&2
      exit 1
    fi
    if [[ ! -f "$BATCH_DIR/prompts.txt" ]]; then
      echo "Error: $BATCH_DIR/prompts.txt not found; --output-dir must point at a batch directory created by a previous --prompts-file run." >&2
      exit 1
    fi
    mapfile -t SAVED_PROMPTS < "$BATCH_DIR/prompts.txt"
    if ! prompts_match RAW_PROMPTS SAVED_PROMPTS; then
      echo "Error: $PROMPTS_FILE does not match $BATCH_DIR/prompts.txt from the original run." >&2
      echo "Resuming requires the exact same prompts (after parsing) so scene numbers line up." >&2
      echo "Edit the original prompts file in place, or start a fresh batch by omitting --output-dir." >&2
      exit 1
    fi
    echo "Resuming batch: $BATCH_DIR"
    # A deliberate resume gives previously-failed items a fresh attempt too, in case
    # the cause was transient. Automatic retries within a single invocation do not.
    rm -f "$BATCH_DIR"/*.failed
  else
    BATCH_NAME="$(basename "$PROMPTS_FILE")"
    BATCH_NAME="${BATCH_NAME%.*}"
    TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
    BATCH_DIR="$OUTPUT_DIR/${BATCH_NAME}-${TIMESTAMP}"
    mkdir -p "$BATCH_DIR"
    printf '%s\n' "${RAW_PROMPTS[@]}" > "$BATCH_DIR/prompts.txt"
    echo "New batch:  $BATCH_DIR"
  fi

elif $VARIATIONS_MODE; then
  ITEM_LABEL="variant"
  NUM_SCENES=$VARIATIONS
  for (( i = 0; i < NUM_SCENES; i++ )); do
    RAW_PROMPTS+=("$PROMPT")
    PER_ITEM_SEEDS+=("$(( BASE_SEED + i ))")
  done

  if [[ -n "$OUTPUT_DIR_ARG" ]]; then
    BATCH_DIR="$(abs_path "$OUTPUT_DIR_ARG")"
    if [[ ! -d "$BATCH_DIR" ]]; then
      echo "Error: --output-dir does not exist: $BATCH_DIR" >&2
      exit 1
    fi
    if [[ ! -f "$BATCH_DIR/prompt.txt" || ! -f "$BATCH_DIR/base_seed.txt" ]]; then
      echo "Error: $BATCH_DIR does not look like a --variations run (missing prompt.txt/base_seed.txt)." >&2
      exit 1
    fi
    SAVED_PROMPT="$(cat "$BATCH_DIR/prompt.txt")"
    SAVED_BASE_SEED="$(cat "$BATCH_DIR/base_seed.txt")"
    if [[ "$PROMPT" != "$SAVED_PROMPT" || "$BASE_SEED" != "$SAVED_BASE_SEED" ]]; then
      echo "Error: prompt or --base-seed does not match the original run in $BATCH_DIR." >&2
      echo "Resuming requires the same prompt and --base-seed so seeds line up (--variations may differ)." >&2
      exit 1
    fi
    echo "Resuming variations: $BATCH_DIR"
    rm -f "$BATCH_DIR"/*.failed
  else
    TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
    BATCH_DIR="$OUTPUT_DIR/variations-${TIMESTAMP}"
    mkdir -p "$BATCH_DIR"
    printf '%s\n' "$PROMPT" > "$BATCH_DIR/prompt.txt"
    printf '%s\n' "$BASE_SEED" > "$BATCH_DIR/base_seed.txt"
    echo "New variations run: $BATCH_DIR"
  fi
fi
mkdir -p "$BATCH_DIR/logs"

SCENE_DIGITS=3
if (( NUM_SCENES > 999 )); then
  SCENE_DIGITS=${#NUM_SCENES}
fi

# Writes "NNN<TAB>SEED<TAB>prompt" to $1 for every item with neither a finished .mp4
# nor a permanent .failed marker. SEED is empty for batch scenes.
compute_todo() {
  local out="$1" i num
  : > "$out"
  for (( i = 1; i <= NUM_SCENES; i++ )); do
    printf -v num "%0${SCENE_DIGITS}d" "$i"
    if [[ -e "$BATCH_DIR/${ITEM_LABEL}_${num}.mp4" || -e "$BATCH_DIR/${ITEM_LABEL}_${num}.failed" ]]; then
      continue
    fi
    printf '%s\t%s\t%s\n' "$num" "${PER_ITEM_SEEDS[$((i - 1))]}" "${RAW_PROMPTS[$((i - 1))]}" >> "$out"
  done
}

# ceil(todo / workers) generation rounds, ~10 min each at the default ~5s (121-frame)
# length -- scaled by ceil(EFFECTIVE_NUM_FRAMES / 121) for longer --duration/
# --num-frames clips, since a 10s clip takes roughly twice as long to denoise as a
# 5s one -- plus 15 min fixed overhead for model loading. Returns plain minutes (a
# valid Slurm time format on its own). This is a rough estimate for sizing the Slurm
# time limit, not a precise timing model; LTX_TIME still overrides it.
compute_batch_time_limit() {
  local todo="$1" workers="$2" rounds frame_rounds
  rounds=$(( (todo + workers - 1) / workers ))
  frame_rounds=$(( (EFFECTIVE_NUM_FRAMES + 120) / 121 ))
  echo $(( rounds * frame_rounds * 10 + 15 ))
}

# Runs one srun step of $WORKERS tasks (one GPU each); "$@" are extra placement flags
# (e.g. --nodes=1 --immediate=60, or nothing). Touched by every task that actually
# starts, so this tells "workers started" apart from "srun could not get an allocation"
# even if the run then fails or is preempted immediately.
launch_workers() {
  local marker="$BATCH_DIR/.started-$ATTEMPT-$1"
  shift
  HF_HUB_OFFLINE=1 srun "${SRUN_ARGS[@]}" "$@" \
    bash "$WORKER" "$BATCH_DIR" "$TODO_FILE" "$marker" "$ITEM_LABEL" "$PYTHON" "${PIPELINE_ARGS[@]}" || true
  [[ -e "$marker" ]]
}

ATTEMPT=0
while (( ATTEMPT < MAX_ATTEMPTS )); do
  ATTEMPT=$(( ATTEMPT + 1 ))
  TODO_FILE="$BATCH_DIR/.todo-$ATTEMPT"
  compute_todo "$TODO_FILE"
  TODO_COUNT="$(wc -l < "$TODO_FILE")"
  TODO_COUNT=${TODO_COUNT//[[:space:]]/}

  if (( TODO_COUNT == 0 )); then
    break
  fi

  WORKERS=$(( GPUS < TODO_COUNT ? GPUS : TODO_COUNT ))
  if [[ -n "$TIME_LIMIT_OVERRIDE" ]]; then
    ATTEMPT_TIME="$TIME_LIMIT_OVERRIDE"
  else
    ATTEMPT_TIME="$(compute_batch_time_limit "$TODO_COUNT" "$WORKERS")"
  fi

  echo
  echo "=== Attempt $ATTEMPT/$MAX_ATTEMPTS on partition '$PARTITION': $TODO_COUNT ${ITEM_LABEL}(s) remaining, $WORKERS worker(s), time limit ${ATTEMPT_TIME} ==="

  SRUN_ARGS=(
    --partition="$PARTITION"
    --ntasks="$WORKERS"
    --gpus-per-task=1
    --cpus-per-task="$CPUS"
    --mem-per-gpu="$MEM_PER_GPU"
    --time="$ATTEMPT_TIME"
    --job-name="ltx23-$ITEM_LABEL"
    --label
  )

  STARTED=false
  if (( WORKERS > 1 )); then
    echo "Trying to fit all $WORKERS workers on one node (waiting up to ${SAME_NODE_WAIT}s)..."
    if launch_workers node --nodes=1 --immediate="$SAME_NODE_WAIT"; then
      STARTED=true
    else
      echo "No single free node with $WORKERS free GPUs within ${SAME_NODE_WAIT}s; spreading across nodes..."
    fi
  fi
  if ! $STARTED; then
    if launch_workers any; then
      STARTED=true
    fi
  fi

  if ! $STARTED; then
    echo "Error: could not start any workers on partition '$PARTITION'." >&2
    break
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

DONE_COUNT=0
FAILED_COUNT=0
FAILED_ITEMS=()
for (( i = 1; i <= NUM_SCENES; i++ )); do
  printf -v num "%0${SCENE_DIGITS}d" "$i"
  if [[ -e "$BATCH_DIR/${ITEM_LABEL}_${num}.mp4" ]]; then
    DONE_COUNT=$(( DONE_COUNT + 1 ))
  elif [[ -e "$BATCH_DIR/${ITEM_LABEL}_${num}.failed" ]]; then
    FAILED_COUNT=$(( FAILED_COUNT + 1 ))
    FAILED_ITEMS+=("${ITEM_LABEL}_${num}  (log: $BATCH_DIR/logs/${ITEM_LABEL}_${num}.log)")
  fi
done
UNFINISHED_COUNT=$(( NUM_SCENES - DONE_COUNT - FAILED_COUNT ))

echo
echo "=== Summary: $BATCH_DIR ==="
echo "Done:       $DONE_COUNT / $NUM_SCENES"
echo "Failed:     $FAILED_COUNT"
if (( FAILED_COUNT > 0 )); then
  printf '  %s\n' "${FAILED_ITEMS[@]}"
fi
echo "Unfinished: $UNFINISHED_COUNT"

if (( UNFINISHED_COUNT > 0 )); then
  if $BATCH_MODE; then
    RESUME_CMD=("$SELF" --prompts-file "$PROMPTS_FILE" --gpus "$GPUS" --output-dir "$BATCH_DIR")
  else
    RESUME_CMD=("$SELF" --prompt-file "$BATCH_DIR/prompt.txt" --variations "$VARIATIONS" --base-seed "$BASE_SEED" --gpus "$GPUS" --output-dir "$BATCH_DIR")
  fi
  if [[ "$MODE" == "quality" ]]; then
    RESUME_CMD+=(--quality)
  fi
  RESUME_CMD+=("${EXTRA_ARGS[@]}")
  echo
  echo "Attempts exhausted with items still unfinished (most likely preempted). Resume with:"
  printf '  '
  printf '%q ' "${RESUME_CMD[@]}"
  printf '\n'
  exit 1
elif (( FAILED_COUNT > 0 )); then
  echo
  echo "All remaining items finished, but some failed outright -- see the logs above."
  exit 1
else
  echo
  echo "All done."
  exit 0
fi
