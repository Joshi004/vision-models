#!/usr/bin/env python3
"""Runs Wan2.2-Animate-14B ("v1") in *replace* mode end to end: preprocessing
(pose/face/background/mask extraction), generation, then re-muxing the
source video's original audio onto the model's (silent) output.

This script is the single unit of work model-api's Slurm job submits --
see model-api/services/wan_animate/dispatch.py, which builds the argv below
and model-api/common/run_pipeline_job.py, which actually invokes it on the
compute node (that wrapper owns the partial-path-rename-on-success /
.failed-marker-on-failure convention; this script only needs to write
`--output-path` and exit non-zero on any real failure).

Must be run with *this project's* venv interpreter (v1/.venv/bin/python),
never the system python3 -- everything below (torch, flash-attn, sam2,
decord, onnxruntime, ...) only exists in that venv. Paths to the Wan2.2
repo and the model checkpoint are resolved relative to this script's own
location (v1/), not the caller's current directory, since model-api invokes
this script from its own working directory, not from inside v1/.

Usage:
    run_replace.py --video SRC.mp4 --image REF.jpg --work-dir DIR \\
        --output-path OUT.mp4 [--seed 10] [--no-relighting-lora] [--no-audio]

--video       The existing footage to edit (the "driving" video). Official
              preprocessing warning: single-person videos only -- mask
              extraction is not designed for multi-person frames.
--image       A single clear reference photo of the character to swap in.
--work-dir    Scratch directory for preprocessing output and the silent
              render (reused as-is; not cleaned up by this script -- the
              caller, model-api, owns that directory's lifecycle).
--output-path Final MP4 path. Written directly (this script does not do
              its own "partial" renaming -- see run_pipeline_job.py).

Exit code 0 only if --output-path was actually written; non-zero on any
failure (a non-zero exit from either Wan2.2 step, a missing expected file,
or a failed audio mux).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo>/models/wan_animate/runner/<this file>
# Same data directory the gateway uses (gateway/common/config.py): VISION_DATA_DIR,
# defaulting to a "vision-data" folder next to the repository.
DATA_DIR = Path(os.environ.get("VISION_DATA_DIR", str(REPO_ROOT.parent / "vision-data")))
WAN22_REPO = REPO_ROOT / "third_party" / "Wan2.2"
CKPT_DIR = DATA_DIR / "weights" / "wan_animate" / "Wan2.2-Animate-14B"
PROCESS_CKPT_DIR = CKPT_DIR / "process_checkpoint"
PREPROCESS_SCRIPT = WAN22_REPO / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = WAN22_REPO / "generate.py"

# Matches the official README's documented replace-mode preprocessing
# flags exactly (see UserGuider.md). Not exposed as request-level options
# today -- see the setup plan's "deliberately excluded" list.
RESOLUTION_AREA = ("1280", "720")
MASK_ITERATIONS = "3"
MASK_K = "7"
MASK_W_LEN = "1"
MASK_H_LEN = "1"

# WanAnimate.generate() hard-asserts refert_num in {1, 5}; the CLI's own
# --refert_num default (77) fails that assert, so it must always be passed
# explicitly. 1 is the officially-documented single-GPU example value.
REFERT_NUM = "1"


def _run_step(name: str, argv: list[str], cwd: Path) -> None:
    print(f"=== {name}: starting ===", flush=True)
    print("argv:", argv, flush=True)
    start = time.monotonic()
    subprocess.run(argv, cwd=str(cwd), check=True)
    elapsed = time.monotonic() - start
    print(f"=== {name}: finished in {elapsed:.1f}s ===", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Source/driving video (the existing footage to edit).")
    parser.add_argument("--image", required=True, help="Reference image of the character to swap in.")
    parser.add_argument("--work-dir", required=True, help="Scratch directory for preprocessing output and the silent render.")
    parser.add_argument("--output-path", required=True, help="Final MP4 path to write.")
    parser.add_argument("--seed", type=int, default=10, help="Random seed (base_seed).")
    parser.add_argument(
        "--no-relighting-lora",
        action="store_true",
        help="Skip the relighting LoRA (on by default -- matches the swapped-in character's lighting/tone to the original scene).",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip re-muxing the source video's audio; write the model's silent output directly.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    python = sys.executable  # must already be v1/.venv/bin/python -- see module docstring
    video_path = Path(args.video).resolve()
    image_path = Path(args.image).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_path = Path(args.output_path).resolve()
    process_results = work_dir / "process_results"
    silent_video = work_dir / "generated_silent.mp4"

    work_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"ERROR: source video not found: {video_path}", file=sys.stderr)
        return 1
    if not image_path.exists():
        print(f"ERROR: reference image not found: {image_path}", file=sys.stderr)
        return 1

    # Nothing here needs network access (every checkpoint path below is
    # already local) -- avoid any accidental hub round-trip slowing down a
    # cold start, same defensive habit LTX's own dispatch already uses.
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # --- 1. Preprocessing: pose/face/background/mask extraction (replace mode) ---
    _run_step(
        "preprocess",
        [
            python, str(PREPROCESS_SCRIPT),
            "--ckpt_path", str(PROCESS_CKPT_DIR),
            "--video_path", str(video_path),
            "--refer_path", str(image_path),
            "--save_path", str(process_results),
            "--resolution_area", *RESOLUTION_AREA,
            "--iterations", MASK_ITERATIONS,
            "--k", MASK_K,
            "--w_len", MASK_W_LEN,
            "--h_len", MASK_H_LEN,
            "--replace_flag",
        ],
        cwd=WAN22_REPO,
    )
    for required in ("src_ref.png", "src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4"):
        if not (process_results / required).exists():
            print(f"ERROR: preprocessing finished but did not produce {required}", file=sys.stderr)
            return 1

    # --- 2. Generation (replace mode, single GPU) ---
    generate_argv = [
        python, str(GENERATE_SCRIPT),
        "--task", "animate-14B",
        "--ckpt_dir", str(CKPT_DIR),
        "--src_root_path", str(process_results),
        "--refert_num", REFERT_NUM,
        "--replace_flag",
        "--base_seed", str(args.seed),
        "--save_file", str(silent_video),
    ]
    if not args.no_relighting_lora:
        generate_argv.append("--use_relighting_lora")
    _run_step("generate", generate_argv, cwd=WAN22_REPO)

    if not silent_video.exists():
        # generate.py's own save_video() catches and logs write errors
        # instead of raising (a real gap in the upstream code), so a clean
        # exit code alone doesn't guarantee the file exists -- check
        # explicitly rather than letting a confusing ffmpeg error below be
        # the first sign anything went wrong.
        print(
            f"ERROR: generate.py exited successfully but {silent_video} does not exist. "
            "Check the run log above for a 'save_video failed' message.",
            file=sys.stderr,
        )
        return 1

    # --- 3. Re-mux the source video's original audio ---
    # The model itself produces silent video (see the service's openapi
    # docs). `-map 1:a:0?` is an *optional* stream map: ffmpeg simply omits
    # it if the source has no audio track, so one invocation handles both
    # cases without a separate probe step.
    if args.no_audio:
        shutil.move(str(silent_video), str(output_path))
        return 0

    import imageio_ffmpeg  # local import: only this venv has it installed

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mux_argv = [
        ffmpeg_exe, "-y",
        "-i", str(silent_video),
        "-i", str(video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    _run_step("mux_audio", mux_argv, cwd=work_dir)

    if not output_path.exists():
        print(f"ERROR: ffmpeg exited successfully but did not produce {output_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
