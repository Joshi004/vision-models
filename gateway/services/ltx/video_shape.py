"""Video shape helpers: duration<->frames, orientation<->height/width.

Mirrors the exact formulas generate.sh already uses (frames_for_duration,
snap_frames) so a given duration/orientation resolves to the same frame
count/resolution whether requested through generate.sh or through this API.
"""

from __future__ import annotations

from services.ltx import config


def frames_for_duration(seconds: float, fps: float) -> int:
    """SECONDS, FPS -> nearest valid LTX-2 frame count (8*K + 1, K >= 1)."""
    k = max(1, round(seconds * fps / 8))
    return 8 * k + 1


def snap_frames(frames: int) -> int:
    """Round an arbitrary frame count to the nearest valid 8*K + 1 (K >= 1)."""
    k = max(1, round((frames - 1) / 8))
    return 8 * k + 1


def resolve_num_frames(duration_seconds: float | None, num_frames: int | None, frame_rate: float) -> int:
    """Resolve the final --num-frames value.

    Callers (schemas.py) already guarantee at most one of duration_seconds /
    num_frames is set; if neither is given, falls back to the pipelines' own
    default (LTX_2_3_PARAMS.num_frames == 121, ~5s @ 24fps).
    """
    if duration_seconds is not None:
        return frames_for_duration(duration_seconds, frame_rate)
    if num_frames is not None:
        if (num_frames - 1) % 8 != 0 or num_frames < 9:
            return snap_frames(num_frames)
        return num_frames
    return config.DEFAULT_NUM_FRAMES


def resolve_dimensions(orientation: str | None, height: int | None, width: int | None) -> tuple[int, int]:
    """Resolve the final (height, width). Callers already guarantee at most
    one of orientation / (height and width) is given."""
    if height is not None and width is not None:
        return height, width
    if orientation == "portrait":
        return config.PORTRAIT_HEIGHT, config.PORTRAIT_WIDTH
    return config.LANDSCAPE_HEIGHT, config.LANDSCAPE_WIDTH
