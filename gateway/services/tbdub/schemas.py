"""Pydantic request model for POST /v1/tbdub/videos/lip-sync, plus the
pieces the LTX audio-to-video schemas share to offer `lip_sync: true`.

File inputs are referenced by an asset_id obtained from POST /v1/uploads (see
services/ltx/schemas.py for why), or -- to re-sync an LTX audio-to-video job
that already ran -- by that job's id, so nothing has to be uploaded again.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from common.schemas import IncognitoOptionMixin, PartitionOptionMixin
from services.tbdub import config

# The `lip_sync` field of both LTX audio-to-video requests (2.3 and 2.5).
LIP_SYNC_FIELD_DESCRIPTION = (
    "If true, after the video is generated a dedicated lip-sync model (TBDub) redraws the speaker's "
    "mouth to match the audio exactly, in the same job (no second upload or request). This forces "
    f"`frame_rate` to {config.LIP_SYNC_FRAME_RATE:g} (the rate the lip-sync model works at), and an explicit "
    "`num_frames` is converted to the same duration at your frame rate and re-rounded at "
    f"{config.LIP_SYNC_FRAME_RATE:g} fps -- the request as stored shows the values actually used. Lip sync only changes the "
    "mouth, jaw and the skin around them: anything else the speaker should do (eyebrows, a laugh, nods) "
    "must be described in `prompt`. It adds several minutes to the job. If lip sync fails the job still "
    "succeeds and returns the video WITHOUT lip sync; check `lip_sync.applied` / `lip_sync.error` on the "
    "job, then retry just the sync with POST /v1/tbdub/videos/lip-sync and this job's id as `source_job_id`. "
    "One speaker with a visible face is expected."
)


def apply_lip_sync_shape(req: Any) -> None:
    """For a request with `lip_sync` set, force the frame rate the lip-sync
    model needs, keeping the clip's duration.

    `req` is an LTX audio-to-video request (VideoShapeMixin fields). An exact
    `num_frames` is turned into the duration it has at the caller's frame rate,
    then the usual duration -> frame-count rule (8k+1 at the new rate) applies
    -- so the clip does not get shorter just because the frame rate went up.
    A request that gives neither keeps the pipeline's own default length, and
    one with `audio_max_duration` ("match the audio") already follows the audio.
    """
    if not req.lip_sync:
        return
    if req.num_frames is not None:
        req.duration_seconds = req.num_frames / req.frame_rate
        req.num_frames = None
    req.frame_rate = config.LIP_SYNC_FRAME_RATE


class LipSyncRequest(PartitionOptionMixin, IncognitoOptionMixin):
    """Request body for POST /v1/tbdub/videos/lip-sync.

    Two ways to say what to sync: a video and an audio file you uploaded
    (`video_asset_id` + `audio_asset_id`), or an LTX audio-to-video job that
    already succeeded (`source_job_id`), whose video and audio are reused."""

    video_asset_id: str | None = Field(
        None,
        description=(
            "asset_id (from POST /v1/uploads) of the video to re-sync. Any frame rate: it is converted to "
            "25 fps first, keeping its duration. Its own audio track is ignored. Needs one speaker whose "
            "face is visible. Give either this (with `audio_asset_id`) or `source_job_id`, not both."
        ),
        examples=[None],
    )
    audio_asset_id: str | None = Field(
        None,
        description=(
            "asset_id (from POST /v1/uploads) of the audio, or of a video whose audio track is used, that "
            "the mouth should follow. Required with `video_asset_id`; not allowed with `source_job_id`."
        ),
        examples=[None],
    )
    audio_start_time: float | None = Field(
        None,
        ge=0,
        description=(
            "Seconds into the audio file where it lines up with the start of the video. Defaults to 0. "
            "Only with `video_asset_id`: with `source_job_id` the source job's own `audio_start_time` is used."
        ),
        examples=[None],
    )
    source_job_id: str | None = Field(
        None,
        description=(
            "job_id of a succeeded LTX audio-to-video job (`ltx:audio-to-video` or `ltx25:audio-to-video`) "
            "to re-sync -- typically one that returned `lip_sync.applied: false`. Its un-synced video "
            "(or, if lip sync was not requested, its result) and its audio are reused; nothing needs "
            "uploading. Not available once that job was deleted, was an `incognito` job, or lost its audio "
            "upload; then upload the video and audio instead."
        ),
        examples=["6b54a0bd6cef4b04abe50a8f81f55a80"],
    )
    seed: int = Field(10, description="Random seed; same inputs + settings + seed = same result.", examples=[10])
    audio_cfg_scale: float | None = Field(
        None,
        ge=1.0,
        le=20.0,
        description=(
            "How strongly the mouth follows the audio (\"follow the voice\"). 1.0 = no extra push. Raise it "
            "if the mouth under-articulates or barely moves; too high gives exaggerated mouth shapes and "
            "odd teeth. Default (omit): TBDub's own setting, currently 6."
        ),
        examples=[None],
    )
    ref_cfg_scale: float | None = Field(
        None,
        ge=1.0,
        le=20.0,
        description=(
            "How strictly the face stays faithful to the original video (\"stay like the source\"). 1.0 = no "
            "extra push. Raise it if skin, teeth or face shape drift from the original; too high copies "
            "the original mouth's own motion. Default (omit): TBDub's own setting, currently 2."
        ),
        examples=[None],
    )

    @model_validator(mode="after")
    def _check_inputs(self):
        if (self.video_asset_id is None) == (self.source_job_id is None):
            raise ValueError("Give exactly one of 'video_asset_id' (with 'audio_asset_id') or 'source_job_id'.")
        if self.video_asset_id is not None and self.audio_asset_id is None:
            raise ValueError("'audio_asset_id' is required with 'video_asset_id'.")
        if self.source_job_id is not None and (self.audio_asset_id is not None or self.audio_start_time is not None):
            raise ValueError(
                "'audio_asset_id' and 'audio_start_time' are not allowed with 'source_job_id' "
                "(the source job's own audio is used)."
            )
        return self
