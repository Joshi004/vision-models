#!/usr/bin/env python3
"""Runs LTX-2.3's audio-to-video pipeline (`ltx_pipelines.a2vid_two_stage`)
for one `/v1/ltx/videos/audio-to-video` request, after first converting the
input audio to a stereo WAV. See `gateway/services/ltx/dispatch.py`, which
builds the argv below, and `gateway/common/run_pipeline_job.py`, which
invokes this on the compute node.

Why: the LTX-2.3 audio VAE encoder only accepts 2-channel (stereo) audio,
and the final MP4 mux rejects anything else too, but upstream's
`decode_audio_from_file` keeps whatever channel count the file has. A mono
file (e.g. Breeze TTS output, most voice recordings) therefore fails with
"expected input[1, 1, ...] to have 2 channels". Mono is copied into both
channels at full level, surround is downmixed. The sample rate is kept as-is:
the pipeline resamples for the encoder itself and muxes AAC at the source
rate.

Must be run with LTX-2.3's own venv interpreter -- `ltx_pipelines` and
`imageio-ffmpeg` (the only ffmpeg on this cluster) only exist there.

Usage:
    run_a2vid.py --audio-path FILE --work-dir DIR -- <a2vid_two_stage args>

Everything after the first "--" is passed to `ltx_pipelines.a2vid_two_stage`
verbatim (it contains user text such as --prompt, so it is never parsed
here); it must not include --audio-path, which this script supplies.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import av
import imageio_ffmpeg


class AudioDecodeError(RuntimeError):
    """ffmpeg could not decode the input file (corrupt, truncated, no audio
    stream, or an unsupported codec)."""


def _channel_count(src: Path) -> int:
    """Channel count of the first audio stream of `src`."""
    try:
        with av.open(str(src)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            return stream.codec_context.channels
    except StopIteration:
        raise AudioDecodeError(f"No audio stream in {src}") from None
    except av.error.FFmpegError as error:
        raise AudioDecodeError(f"Could not open {src}: {error}") from error


def to_stereo_wav(src: Path, dst: Path) -> None:
    """Decodes the first audio stream of `src` to a 2-channel float32 PCM WAV
    at `dst`, keeping the source sample rate. Raises AudioDecodeError (with
    ffmpeg's own stderr) if ffmpeg fails or doesn't produce `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _channel_count(src) == 1:
        # ffmpeg's own mono -> stereo upmix (-ac 2) attenuates by 3 dB.
        channel_args = ["-af", "pan=stereo|c0=c0|c1=c0"]
    else:
        channel_args = ["-ac", "2"]
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
        *channel_args,
        "-c:a", "pcm_f32le",
        str(dst.resolve()),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file():
        raise AudioDecodeError(f"ffmpeg could not decode {src} (exit {proc.returncode}): {proc.stderr.strip()}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-path", required=True, help="Input audio (any format ffmpeg can decode).")
    parser.add_argument("--work-dir", required=True, help="Directory for the converted audio_stereo.wav.")
    return parser.parse_args(argv)


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        print("usage: run_a2vid.py --audio-path FILE --work-dir DIR -- <a2vid_two_stage args>", file=sys.stderr)
        return 2
    separator = argv.index("--")
    args = _parse_args(argv[:separator])
    pipeline_args = argv[separator + 1:]

    source = Path(args.audio_path)
    if not source.is_file():
        print(f"ERROR: audio file not found: {source}", file=sys.stderr)
        return 1

    stereo_wav = Path(args.work_dir) / "audio_stereo.wav"
    print(f"Converting '{source}' to stereo -> '{stereo_wav}'...", flush=True)
    try:
        to_stereo_wav(source, stereo_wav)
    except AudioDecodeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    pipeline_argv = [
        sys.executable, "-m", "ltx_pipelines.a2vid_two_stage",
        "--audio-path", str(stereo_wav),
        *pipeline_args,
    ]
    return subprocess.run(pipeline_argv).returncode


if __name__ == "__main__":
    sys.exit(main())
