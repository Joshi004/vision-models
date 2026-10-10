#!/usr/bin/env bash
# generate_interactive.sh - Interactive wizard for generate.sh.
#
# Sourced (not executed) by generate.sh when it is run with no arguments. Runs in
# generate.sh's own shell, so it can see and use the variables/functions generate.sh
# has already defined (SCRIPT_DIR, SELF, PARTITION, TIME_LIMIT_OVERRIDE,
# read_prompt_file, collapse_whitespace, expand_user_path, display_path,
# is_positive_number, is_nonnegative_number, frames_for_duration, image_orientation,
# check_image_decodes, PYTHON, abs_path, LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT,
# PORTRAIT_WIDTH, PORTRAIT_HEIGHT, DEFAULT_FRAME_RATE). It populates the same
# variables the flag parser would have set (PROMPT, MODE, GPUS, VARIATIONS,
# BASE_SEED, DURATION, ORIENTATION, IMAGE_FIT, EXTRA_ARGS, ...), then returns
# control to generate.sh, which continues exactly as if those had been passed as
# flags. generate.sh's own video-shape step (--duration -> --num-frames,
# --orientation -> --height/--width, --image path checking/decoding/fit) runs
# afterward on whatever this wizard puts in EXTRA_ARGS/DURATION/ORIENTATION/
# IMAGE_FIT, so this file only needs to collect those, not act on them.
#
# Covers single-video and --variations only (not --prompts-file batch mode, which
# needs a prepared file of several different scenes rather than one prompt).

echo "=== LTX-2.3 interactive generation ==="
echo "(Ctrl-C at any point exits without submitting anything.)"
echo

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT=""
while [[ -z "$PROMPT" ]]; do
  echo "How do you want to provide the prompt?"
  echo "  1) Load from a file"
  echo "  2) Type it now"
  read -rp "Choice [1-2, Enter = 2]: " prompt_source

  # A path pasted directly at this prompt (instead of picking 1 first, then being
  # asked for the path) is accepted too, rather than rejected as an invalid choice.
  prompt_path=""
  prompt_source_expanded="$(expand_user_path "$prompt_source")"
  if [[ "$prompt_source" != "1" && "$prompt_source" != "2" \
        && -n "$prompt_source_expanded" && -f "$prompt_source_expanded" ]]; then
    prompt_source="1"
    prompt_path="$prompt_source_expanded"
  fi

  case "$prompt_source" in
    1)
      while [[ -z "$prompt_path" ]]; do
        read -erp "Path to prompt file: " prompt_path_in
        prompt_path_in="$(expand_user_path "$prompt_path_in")"
        if [[ -z "$prompt_path_in" ]]; then
          break
        elif [[ -f "$prompt_path_in" ]]; then
          prompt_path="$prompt_path_in"
        else
          echo "File not found: $(display_path "$prompt_path_in")"
          echo "(Relative paths are resolved from the project root: $SCRIPT_DIR. Tab-complete works from there too.)"
        fi
      done
      if [[ -z "$prompt_path" ]]; then
        continue
      fi
      PROMPT="$(read_prompt_file "$prompt_path")"
      if [[ -z "$PROMPT" ]]; then
        echo "File exists but is empty (0 bytes): $(display_path "$prompt_path")"
        echo "Add your prompt text to that file, then try again -- or choose option 2 to type it now."
      fi
      ;;
    2 | "")
      echo "(one line only -- for a long or multi-line prompt, save it to a file and choose option 1 above)"
      read -rp "Enter your prompt: " typed_prompt
      PROMPT="$(collapse_whitespace "$typed_prompt")"
      if [[ -z "$PROMPT" ]]; then
        echo "That produced an empty prompt -- let's try again."
      fi
      ;;
    *)
      echo "Please choose 1 or 2."
      ;;
  esac
done

echo
echo "Prompt to use:"
echo "  $PROMPT"
echo
read -rp "Looks right? [Y/n]: " prompt_ok
if [[ "$prompt_ok" =~ ^[Nn] ]]; then
  echo "Run the script again to retry."
  exit 0
fi

# ---------------------------------------------------------------------------
# Image (optional first frame)
# ---------------------------------------------------------------------------

echo
read -rp "Use an image as the first frame? [y/N]: " use_image_in
IMAGE_PATH=""
IMAGE_ARGS=()
if [[ "$use_image_in" =~ ^[Yy] ]]; then
  while true; do
    read -erp "Path to image (JPG/PNG; on this filesystem, not /tmp -- compute nodes can't see /tmp; blank to skip): " image_path_in
    image_path_in="$(expand_user_path "$image_path_in")"
    if [[ -z "$image_path_in" ]]; then
      echo "No image will be used."
      break
    elif [[ -f "$image_path_in" ]]; then
      # Image.open() alone only reads the header, which survives truncation --
      # fully decode it here so a corrupt/incomplete file (e.g. an interrupted
      # upload) is caught now, not after a Slurm job has already spent minutes
      # loading models before reaching this same file. Skipped if the venv isn't
      # set up yet (generate.sh's own check for that runs after this wizard); that
      # case surfaces its own clear error once the wizard finishes.
      decode_error=""
      if [[ -x "$PYTHON" ]]; then
        decode_error="$(check_image_decodes "$image_path_in" || true)"
      fi
      if [[ -n "$decode_error" ]]; then
        echo "Can't read that image ($decode_error)."
        echo "It's probably incomplete or corrupt (e.g. an interrupted upload) -- re-upload it and try again, or pick a different file."
      else
        IMAGE_PATH="$(abs_path "$image_path_in")"
        break
      fi
    else
      echo "File not found: $(display_path "$image_path_in")"
      echo "(Relative paths are resolved from the project root: $SCRIPT_DIR. Tab-complete works from there too.)"
    fi
  done

  if [[ -n "$IMAGE_PATH" ]]; then
    read -rp "Image strength, 0-1 [1.0]: " image_strength_in
    IMAGE_STRENGTH="${image_strength_in:-1.0}"
    if ! is_nonnegative_number "$IMAGE_STRENGTH" || ! awk -v s="$IMAGE_STRENGTH" 'BEGIN { exit !(s <= 1) }'; then
      echo "Error: strength must be a number between 0 and 1." >&2
      exit 1
    fi
    IMAGE_ARGS=(--image "$IMAGE_PATH" 0 "$IMAGE_STRENGTH")
  fi
fi

# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

echo
DEFAULT_ORIENTATION="landscape"
if [[ -n "$IMAGE_PATH" ]]; then
  detected_orientation="$(image_orientation "$IMAGE_PATH" || true)"
  if [[ -n "$detected_orientation" ]]; then
    DEFAULT_ORIENTATION="$detected_orientation"
  fi
fi
if [[ "$DEFAULT_ORIENTATION" == "portrait" ]]; then
  DEFAULT_ORIENTATION_CHOICE=2
else
  DEFAULT_ORIENTATION_CHOICE=1
fi
echo "Orientation?"
echo "  1) Landscape (${LANDSCAPE_WIDTH}x${LANDSCAPE_HEIGHT}, YouTube)"
echo "  2) Portrait (${PORTRAIT_WIDTH}x${PORTRAIT_HEIGHT}, Shorts/Reels)"
read -rp "Choice [1-2, Enter = $DEFAULT_ORIENTATION_CHOICE]: " orientation_in
orientation_in="${orientation_in:-$DEFAULT_ORIENTATION_CHOICE}"
case "$orientation_in" in
  1) ORIENTATION="landscape" ;;
  2) ORIENTATION="portrait" ;;
  *)
    echo "Error: please choose 1 or 2." >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Image fit (only relevant if an image was chosen above)
# ---------------------------------------------------------------------------

IMAGE_FIT="crop"
if [[ -n "$IMAGE_PATH" ]]; then
  echo
  echo "How should the image be fit to the video frame?"
  echo "  1) Crop -- fills the frame, trimming whatever doesn't fit (default)"
  echo "  2) Pad -- keeps the whole image, adding black bars where it doesn't fill the frame"
  read -rp "Choice [1-2, Enter = 1]: " image_fit_in
  image_fit_in="${image_fit_in:-1}"
  case "$image_fit_in" in
    1) IMAGE_FIT="crop" ;;
    2) IMAGE_FIT="pad" ;;
    *)
      echo "Error: please choose 1 or 2." >&2
      exit 1
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

echo
read -rp "Video length in seconds [5]: " duration_in
DURATION="${duration_in:-5}"
if ! is_positive_number "$DURATION"; then
  echo "Error: that's not a valid number of seconds." >&2
  exit 1
fi
DURATION_FRAMES="$(frames_for_duration "$DURATION" "$DEFAULT_FRAME_RATE")"
DURATION_ACTUAL="$(awk -v f="$DURATION_FRAMES" -v r="$DEFAULT_FRAME_RATE" 'BEGIN { printf "%.2f", (f - 1) / r }')"
echo "That's $DURATION_FRAMES frames at ${DEFAULT_FRAME_RATE}fps (${DURATION_ACTUAL}s)."

# ---------------------------------------------------------------------------
# Variations
# ---------------------------------------------------------------------------

echo
read -rp "How many variations of this prompt (different seeds)? [1]: " variations_in
VARIATIONS="${variations_in:-1}"
if [[ ! "$VARIATIONS" =~ ^[0-9]+$ ]] || (( VARIATIONS < 1 )); then
  echo "Error: that's not a valid number." >&2
  exit 1
fi

BASE_SEED=10
GPUS=1
if (( VARIATIONS > 1 )); then
  read -rp "Base seed (variations use base, base+1, ...)? [10]: " base_seed_in
  BASE_SEED="${base_seed_in:-10}"
  if [[ ! "$BASE_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "Error: that's not a valid integer." >&2
    exit 1
  fi

  echo
  echo "How many GPUs? 1 runs the variations one at a time (series); more than 1"
  echo "runs them at the same time (parallel), up to 8."
  read -rp "GPUs [1]: " gpus_in
  GPUS="${gpus_in:-1}"
  if [[ ! "$GPUS" =~ ^[0-9]+$ ]] || (( GPUS < 1 || GPUS > 8 )); then
    echo "Error: --gpus must be 1-8." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

echo
echo "Fast or quality mode?"
echo "  1) Fast -- 8-step distilled pipeline, lowest latency"
echo "  2) Quality -- two-stage pipeline, slower, higher quality"
read -rp "Choice [1-2, Enter = 1]: " mode_in
mode_in="${mode_in:-1}"
case "$mode_in" in
  1) MODE="fast" ;;
  2) MODE="quality" ;;
  *)
    echo "Error: please choose 1 or 2." >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Extra flags
# ---------------------------------------------------------------------------

echo
echo "Anything else to pass through (--seed, --enhance-prompt, --quantization, etc.)?"
echo "Simple space-separated flags only -- for anything needing quotes, use the"
echo "non-interactive command line instead. Leave blank for defaults."
echo "(Length, orientation, and image were already asked above -- no need to repeat"
echo "--num-frames, --height/--width, or --image here.)"
read -rp "Extra flags: " extra_in
EXTRA_ARGS=()
if [[ -n "$extra_in" ]]; then
  read -ra EXTRA_ARGS <<< "$extra_in"
fi
# IMAGE_ARGS is merged in further down, after the summary is printed -- the
# summary already shows the image on its own "Image:" line, so folding it into
# EXTRA_ARGS this early would make the summary's "Extra flags:" line repeat it.

# ---------------------------------------------------------------------------
# Time / partition
# ---------------------------------------------------------------------------

echo
read -rp "Slurm partition? [$PARTITION]: " partition_in
PARTITION="${partition_in:-$PARTITION}"

read -rp "Time limit (blank = default: ${TIME_LIMIT_OVERRIDE:-auto}): " time_in
if [[ -n "$time_in" ]]; then
  TIME_LIMIT_OVERRIDE="$time_in"
fi

# ---------------------------------------------------------------------------
# Summary + confirm
# ---------------------------------------------------------------------------

echo
echo "=== Summary ==="
echo "Prompt:      $PROMPT"
if [[ -n "$IMAGE_PATH" ]]; then
  echo "Image:       $IMAGE_PATH (strength $IMAGE_STRENGTH, fit $IMAGE_FIT)"
else
  echo "Image:       none"
fi
if [[ "$ORIENTATION" == "portrait" ]]; then
  echo "Orientation: portrait (${PORTRAIT_WIDTH}x${PORTRAIT_HEIGHT})"
else
  echo "Orientation: landscape (${LANDSCAPE_WIDTH}x${LANDSCAPE_HEIGHT})"
fi
echo "Duration:    ${DURATION}s ($DURATION_FRAMES frames)"
echo "Mode:        $MODE"
echo "Variations:  $VARIATIONS"
if (( VARIATIONS > 1 )); then
  echo "Base seed:   $BASE_SEED"
fi
echo "GPUs:        $GPUS"
echo "Partition:   $PARTITION"
echo "Time limit:  ${TIME_LIMIT_OVERRIDE:-auto}"
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  echo "Extra flags: ${EXTRA_ARGS[*]}"
fi
echo
read -rp "Proceed? [y/N]: " proceed
if [[ ! "$proceed" =~ ^[Yy] ]]; then
  echo "Cancelled -- nothing submitted."
  exit 0
fi

# Now that the summary has been shown (and confirmed), fold the image flags into
# EXTRA_ARGS -- generate.sh's own --image handling (path checking/absolutizing,
# the orientation-mismatch warning) picks it up from there, same as if --image
# had been passed on the command line.
EXTRA_ARGS+=("${IMAGE_ARGS[@]}")

# ---------------------------------------------------------------------------
# Equivalent non-interactive command, for next time
# ---------------------------------------------------------------------------

EQUIVALENT_CMD=("$SELF" "$PROMPT" --orientation "$ORIENTATION" --duration "$DURATION")
if (( VARIATIONS > 1 )); then
  EQUIVALENT_CMD+=(--variations "$VARIATIONS" --base-seed "$BASE_SEED" --gpus "$GPUS")
fi
if [[ "$MODE" == "quality" ]]; then
  EQUIVALENT_CMD+=(--quality)
fi
if [[ "$IMAGE_FIT" == "pad" ]]; then
  EQUIVALENT_CMD+=(--image-fit pad)
fi
EQUIVALENT_CMD+=("${EXTRA_ARGS[@]}")

echo
echo "Equivalent non-interactive command for next time:"
printf '  '
printf '%q ' "${EQUIVALENT_CMD[@]}"
printf '\n'
echo
