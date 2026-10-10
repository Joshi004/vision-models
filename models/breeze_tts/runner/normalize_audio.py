"""Decodes an input audio file to a float32 WAV using the ffmpeg binary
bundled with this project's own `.venv` (via the `imageio-ffmpeg` package).

Copied from ModelService_RVC-Applio/normalize_audio.py. Exists because
Breeze TTS 2's own reference-audio loader (`breeze_infer/audio.py`) only
ever calls `soundfile.read()`, and this deployment's libsndfile (1.2.2)
can't decode AAC/M4A/MP4/WMA and similar formats -- the same gap RVC hit
with real uploads (iPhone-recorded `.m4a` voice notes). No system ffmpeg
exists on this cluster; `imageio_ffmpeg.get_ffmpeg_exe()` is the only
source of one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


class AudioDecodeError(RuntimeError):
    """ffmpeg could not decode the given input file -- a corrupt/truncated
    file, or a genuinely unsupported codec."""


def to_wav(src: Path, dst: Path) -> None:
    """Decodes `src` (any format ffmpeg can read) to a float32 PCM WAV at
    `dst`, preserving the source's sample rate and channel count (Breeze's
    own loader downmixes to mono and resamples itself).

    Only the first audio stream is used (`-map 0:a:0`). `dst`'s parent
    directory is created if missing. Raises AudioDecodeError (with
    ffmpeg's own stderr) if ffmpeg exits non-zero or doesn't produce `dst`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        # Absolute path: always starts with "/", so a colon inside a
        # filename can never be misread as an ffmpeg "scheme:" prefix.
        "-i", str(src.resolve()),
        "-map", "0:a:0",
        "-c:a", "pcm_f32le",
        str(dst.resolve()),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file():
        raise AudioDecodeError(
            f"ffmpeg could not decode {src} (exit {proc.returncode}): {proc.stderr.strip()}"
        )
