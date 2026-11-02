"""Checks on audio files that a music request refers to (an upload's asset_id).

Shared by the music backends so a bad file gets an immediate 400 from the API
instead of failing later on a GPU. Needs PyAV, which the gateway environment has.
"""

from __future__ import annotations

from pathlib import Path

import av

from common import storage

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"})
ACCEPTED_AUDIO_FORMATS = ", ".join(sorted(ext.lstrip(".") for ext in AUDIO_EXTENSIONS))


class AudioInputError(ValueError):
    """The referenced audio can't be used (wrong type, unreadable, or too long)."""


def audio_duration_seconds(path: Path) -> float:
    """Length of the file's first audio stream in seconds, read from its header (not decoded)."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioInputError(f"No audio stream found in {path.name}.")
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return container.duration / av.time_base
    except av.error.FFmpegError as e:
        raise AudioInputError(f"{path.name} could not be read as audio: {e}") from e
    raise AudioInputError(f"The length of {path.name} could not be determined.")


def resolve_audio_asset(asset_id: str, max_seconds: float) -> tuple[Path, float]:
    """The uploaded audio's path and duration, or an error before any job is created.

    Raises KeyError / FileNotFoundError (unknown or missing asset, as storage does) and
    AudioInputError (unsupported extension, unreadable, empty, or longer than max_seconds).
    """
    path = storage.resolve_asset_path(asset_id)
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise AudioInputError(
            f"Unsupported file extension {path.suffix!r} (accepted: {ACCEPTED_AUDIO_FORMATS}): {path.name}"
        )
    duration = audio_duration_seconds(path)
    if duration <= 0:
        raise AudioInputError(f"{path.name} is empty.")
    if duration > max_seconds:
        raise AudioInputError(f"{path.name} is {duration:.0f}s long; the limit for this endpoint is {max_seconds:g}s.")
    return path, duration
