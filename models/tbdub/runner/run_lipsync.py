#!/usr/bin/env python3
"""Lip-syncs one video to one audio track with TBDub (30-step Teacher).

Used two ways, both as one Slurm job step:
  * the standalone `POST /v1/tbdub/videos/lip-sync` job
    (gateway/services/tbdub/dispatch.py), and
  * the optional second step after LTX audio-to-video, chained by
    gateway/common/run_pipeline_job.py (see its `post_step.json` support).

Must be run with the TBDub venv interpreter (`$VISION_DATA_DIR/envs/tbdub`):
TBDub's dependencies (numpy 1.x, mediapipe, transformers 5.1) conflict with
the LTX environments, which is why this is a separate process.

What this wrapper adds around TBDub's own `inference.py`:

  1. TBDub assumes every video frame lasts 1/25 s and never checks. A video at
     another frame rate is re-encoded to 25 fps here (same duration), so the
     mouth stays in sync with the audio.
  2. The audio is cut to `[audio_start_time, + video duration]`, padded with
     silence if the file ends early. TBDub fills a video that is *shorter*
     than its audio by playing it backwards, so the audio must never outlast
     the video. It is padded by half a frame so rounding cannot drop a frame.
  3. TBDub calls a bare `ffmpeg` command, but the only ffmpeg on this cluster
     is the one bundled with `imageio-ffmpeg`. A directory containing an
     `ffmpeg` link to it is put first on PATH for the TBDub process.
  4. TBDub muxes the audio it was given as 16 kHz mono AAC. The final file is
     instead re-muxed with the audio at its original sample rate and channels.

Usage:
    run_lipsync.py --video FILE --audio FILE [--audio-start-time SECONDS]
                   --output-path FILE --work-dir DIR --tbdub-repo DIR
                   --checkpoint-dir DIR [--seed N]
                   [--audio-cfg-scale X] [--ref-cfg-scale X]

Exit code 0 means `--output-path` was written. Anything else is a failure; the
reason is printed to stderr (the caller keeps the log).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

TBDUB_FPS = 25
SAMPLE_NAME = "lipsync"  # TBDub names its output "<sample name>_tbdub.mp4"
STDERR_TAIL_CHARS = 2000


class LipSyncError(RuntimeError):
    """A step of the lip-sync run failed; the message says which and why."""


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(argv: list[str], what: str, env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = proc.stderr.strip()[-STDERR_TAIL_CHARS:]
        raise LipSyncError(f"{what} failed (exit {proc.returncode}): {tail}")


def _video_fps(video: Path) -> float:
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        meta = next(reader)  # the first item is the metadata dict
    finally:
        reader.close()
    return float(meta["fps"])


def _video_seconds(video: Path) -> tuple[int, float]:
    """(frame count, duration) of the video stream, from a fast stream copy."""
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    return int(frames), float(seconds)


def ensure_25fps(video: Path, work_dir: Path) -> Path:
    """The video itself if it is already 25 fps, else a 25 fps re-encode with
    the same duration (frames are repeated or dropped, not stretched)."""
    fps = _video_fps(video)
    if abs(fps - TBDUB_FPS) < 0.01:
        return video
    print(f"Video is {fps:.3f} fps; converting to {TBDUB_FPS} fps (same duration).", flush=True)
    converted = work_dir / "video_25fps.mp4"
    _run(
        [
            _ffmpeg(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(video.resolve()),
            "-map", "0:v:0", "-an",
            "-vf", f"fps={TBDUB_FPS}",
            "-c:v", "libx264", "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p",
            str(converted),
        ],
        "25 fps conversion",
    )
    return converted


def cut_audio(audio: Path, start_time: float, video_seconds: float, work_dir: Path) -> Path:
    """`[start_time, start_time + video_seconds + half a frame]` of the audio
    as a WAV, original sample rate and channels, padded with silence at the end
    if the file is shorter than that."""
    length = video_seconds + 0.5 / TBDUB_FPS
    cut = work_dir / "audio_cut.wav"
    _run(
        [
            _ffmpeg(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_time:.6f}",
            "-i", str(audio.resolve()),
            "-map", "0:a:0", "-vn",
            "-af", "apad",
            "-t", f"{length:.6f}",
            "-c:a", "pcm_s16le",
            str(cut),
        ],
        "audio cut",
    )
    return cut


def run_tbdub(
    video: Path,
    audio: Path,
    tbdub_repo: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    shim_dir: Path,
    args: argparse.Namespace,
) -> Path:
    shim_dir.mkdir(parents=True, exist_ok=True)
    link = shim_dir / "ffmpeg"
    link.unlink(missing_ok=True)
    link.symlink_to(_ffmpeg())
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"

    argv = [
        sys.executable, str(tbdub_repo / "inference.py"),
        "--video", str(video.resolve()),
        "--audio", str(audio.resolve()),
        "--inference-mode", "teacher",
        "--checkpoint-dir", str(checkpoint_dir),
        "--output-dir", str(out_dir),
        "--sample-name", SAMPLE_NAME,
        "--seed", str(args.seed),
    ]
    if args.audio_cfg_scale is not None:
        argv += ["--audio-cfg-scale", str(args.audio_cfg_scale)]
    if args.ref_cfg_scale is not None:
        argv += ["--ref-cfg-scale", str(args.ref_cfg_scale)]

    print("Running TBDub:", " ".join(argv), flush=True)
    # stdout/stderr go straight to this process's log; the caller keeps it.
    proc = subprocess.run(argv, env=env, cwd=str(tbdub_repo))
    if proc.returncode != 0:
        raise LipSyncError(f"TBDub inference exited with code {proc.returncode} (see the log above).")
    result = out_dir / f"{SAMPLE_NAME}_tbdub.mp4"
    if not result.is_file():
        raise LipSyncError(f"TBDub finished but did not write {result}.")
    return result


def remux_original_audio(tbdub_video: Path, audio: Path, output: Path) -> None:
    """TBDub's video stream (copied, not re-encoded) plus the cut audio at its
    original quality, ending with whichever is shorter."""
    _run(
        [
            _ffmpeg(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(tbdub_video),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output),
        ],
        "final audio mux",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path, help="Video to re-sync (any frame rate).")
    parser.add_argument("--audio", required=True, type=Path, help="Audio (or video with audio) to sync to.")
    parser.add_argument("--audio-start-time", type=float, default=0.0, help="Seconds into --audio to start at.")
    parser.add_argument("--output-path", required=True, type=Path, help="Where to write the lip-synced MP4.")
    parser.add_argument("--work-dir", required=True, type=Path, help="Scratch directory (inside the job folder).")
    parser.add_argument("--tbdub-repo", required=True, type=Path, help="Checkout of TBDub (third_party/TBDub).")
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="TBDub weights folder (has config.json).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-cfg-scale", type=float, default=None, help="Default: TBDub's own (config.json).")
    parser.add_argument("--ref-cfg-scale", type=float, default=None, help="Default: TBDub's own (config.json).")
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args(sys.argv[1:])
    for label, path in (("video", args.video), ("audio", args.audio)):
        if not path.is_file():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 1

    work_dir = args.work_dir / "tbdub_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        video = ensure_25fps(args.video, work_dir)
        frames, seconds = _video_seconds(video)
        print(f"Video: {frames} frames, {seconds:.3f} s at {TBDUB_FPS} fps.", flush=True)
        audio = cut_audio(args.audio, args.audio_start_time, seconds, work_dir)
        tbdub_video = run_tbdub(
            video, audio, args.tbdub_repo, args.checkpoint_dir, work_dir / "tbdub_out", work_dir / "bin", args
        )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        remux_original_audio(tbdub_video, audio, args.output_path)
    except LipSyncError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    shutil.rmtree(work_dir, ignore_errors=True)  # only reached on success; kept on failure for debugging
    print(f"Lip-synced video written to {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
