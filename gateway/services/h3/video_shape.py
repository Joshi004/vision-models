"""Video shape helpers for H3: duration -> frame count, orientation -> canvas.

H3 always runs at 24 fps and its video VAE decodes only frame counts of the form
17 * n + 5, so a requested length is rounded up onto that grid and kept inside
the 5-15 second window the model generates (which, on the grid, is 124 to 345
frames). The runner (models/h3/runner/run_h3.py::snap_num_frames) applies the
same rule on its side; the gateway cannot import that file.
"""

from __future__ import annotations

from services.h3 import config


def snap_frames(frames: int) -> int:
    """Round up to the next 17 * n + 5, then clamp to [MIN_FRAMES, MAX_FRAMES]."""
    snapped = max(frames, 1)
    while snapped % config.FRAMES_PER_CHUNK != config.FRAMES_OFFSET:
        snapped += 1
    return min(max(snapped, config.MIN_FRAMES), config.MAX_FRAMES)


def resolve_num_frames(duration_seconds: float | None, num_frames: int | None) -> int:
    """The final --num-frames value. Callers (schemas.py) already guarantee at
    most one of duration_seconds / num_frames is set; with neither, the model's
    own default length (124 frames, 5.17 s)."""
    if duration_seconds is not None:
        return snap_frames(round(duration_seconds * config.FPS))
    if num_frames is not None:
        return snap_frames(num_frames)
    return config.DEFAULT_FRAMES


def resolve_canvas(orientation: str | None, height: int | None, width: int | None) -> tuple[int, int] | None:
    """The final (height, width), or None to let H3 choose: its own 16:9 canvas
    for text-only and reference requests, and the first keyframe's aspect ratio
    for a keyframe request. Callers already guarantee at most one of
    orientation / (height and width) is given."""
    if height is not None and width is not None:
        return height, width
    if orientation == "portrait":
        return config.PORTRAIT_HEIGHT, config.PORTRAIT_WIDTH
    if orientation == "landscape":
        return config.LANDSCAPE_HEIGHT, config.LANDSCAPE_WIDTH
    return None
