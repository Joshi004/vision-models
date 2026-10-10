"""Decodes an input audio file to a float32 WAV using the ffmpeg binary
already bundled with this project's own `.venv` (via the `imageio-ffmpeg`
package) -- the same binary `ModelService_Wan-Animate-2/v1/run_replace.py`
already uses on this cluster's compute nodes for muxing video audio. No
system ffmpeg exists on this cluster (checked directly) --
`imageio_ffmpeg.get_ffmpeg_exe()` is the only source for a real ffmpeg
binary here.

Exists because Applio's own `load_audio_infer()`/`convert_audio_batch()`
(rvc/lib/utils.py, rvc/infer/infer.py) only ever call `soundfile.read()`,
and this deployment's libsndfile (1.2.2) can decode WAV/MP3/FLAC/OGG/Opus/
AIFF but not AAC/M4A/MP4/WMA/... -- real uploads (iPhone-recorded `.m4a`
voice notes) hit exactly this gap, failing every real `/convert` request
that used one. See `model-api/services/rvc/config.py`'s `INPUT_EXTENSIONS`
for the full accepted list this is meant to cover.

Used by both `run_convert.py` (one file) and `run_batch.py` (many files) --
neither imports the other, so this is the one shared piece between them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


class AudioDecodeError(RuntimeError):
    """ffmpeg could not decode the given input file -- a corrupt/truncated
    file, or a genuinely unsupported codec (rare: ffmpeg's own format
    coverage is far broader than libsndfile's alone)."""


def to_wav(src: Path, dst: Path) -> None:
    """Decodes `src` (any format ffmpeg can read) to a float32 PCM WAV at
    `dst`, preserving the source's original sample rate and channel count
    -- Applio's own `load_audio()`/`load_audio_infer()` already do their
    own mono downmix and resampling on whatever soundfile hands them, so
    this only needs to get soundfile a format it can actually open, not
    replicate that logic itself.

    Only the first audio stream is used (`-map 0:a:0`) -- matches
    Wan-Animate's own `run_replace.py` convention for the same reason: a
    video file with an audio track has exactly one stream worth converting
    here.

    `dst`'s parent directory is created if missing. Raises
    AudioDecodeError (with ffmpeg's own stderr) if ffmpeg exits non-zero or
    doesn't actually produce `dst`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    argv = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        # Absolute paths: at least one real upload on this deployment is
        # named "Soi Sukhumvit 48:3 3.m4a" -- resolving to an absolute
        # path (always starts with "/", which can never match ffmpeg's own
        # "scheme:" protocol-prefix syntax) sidesteps any risk of a colon
        # inside a filename ever being misread as a protocol prefix.
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
