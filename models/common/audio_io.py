"""Small audio helpers shared by the music runners (needs PyAV and numpy).

Every music runner that accepts an input recording decodes it the same way, so a
file that one model accepts decodes identically for the others.
"""

from __future__ import annotations

from pathlib import Path


def decode_audio(path: Path):
    """Decode any audio (or video) file's first audio stream.

    Returns (sample_rate, samples) where samples is float32 with shape
    [channels, frames], always stereo (mono input is duplicated to both channels).
    """
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {path}")
        stream = container.streams.audio[0]
        sample_rate = stream.rate
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError(f"No audio samples decoded from {path}")
    return sample_rate, np.concatenate(chunks, axis=1).astype(np.float32, copy=False)


def duration_seconds(path: Path) -> float:
    """Length of the first audio stream in seconds, without decoding it."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {path}")
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return container.duration / av.time_base
    raise ValueError(f"Cannot determine the duration of {path}")


def write_flac(path: Path, samples, sample_rate: int) -> None:
    """Write float samples shaped [frames, channels] as 24-bit FLAC."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, format="FLAC", subtype="PCM_24")
