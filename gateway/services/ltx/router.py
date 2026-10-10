"""LTX-2.3's own generation endpoints, mounted by server.py under /v1/ltx.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
endpoints that are specific to LTX-2.3's five generation recipes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common import openapi_docs as common_docs
from common.schemas import JobSubmitResponse
from services.ltx import dispatch, openapi_docs as ltx_docs
from services.ltx.schemas import (
    AudioToVideoRequest,
    KeyframeInterpolationRequest,
    RetakeRequest,
    TextToAudioRequest,
    TextToVideoRequest,
)

router = APIRouter()


def _submit(dispatch_fn, req) -> JobSubmitResponse:
    try:
        job_id = dispatch_fn(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, f"Invalid asset reference: {e}") from e
    except dispatch.DispatchError as e:
        raise HTTPException(502, f"Could not submit Slurm job: {e}") from e
    return JobSubmitResponse(job_id=job_id, status="queued")


@router.post(
    "/videos/generate",
    tags=["Video generation"],
    summary="Text/image-to-video (fast or quality)",
    description=ltx_docs.GENERATE_VIDEO,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def generate_video(
    req: Annotated[TextToVideoRequest, Body(openapi_examples=ltx_docs.GENERATE_VIDEO_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_text_to_video, req)


@router.post(
    "/videos/keyframe-interpolation",
    tags=["Video generation"],
    summary="Interpolate between >=2 keyframe images",
    description=ltx_docs.KEYFRAME_INTERPOLATION,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def keyframe_interpolation(
    req: Annotated[KeyframeInterpolationRequest, Body(openapi_examples=ltx_docs.KEYFRAME_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_keyframe_interpolation, req)


@router.post(
    "/videos/audio-to-video",
    tags=["Video generation"],
    summary="Generate video conditioned on a given audio track",
    description=ltx_docs.AUDIO_TO_VIDEO,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def audio_to_video(
    req: Annotated[AudioToVideoRequest, Body(openapi_examples=ltx_docs.AUDIO_TO_VIDEO_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_audio_to_video, req)


@router.post(
    "/videos/retake",
    tags=["Video generation"],
    summary="Regenerate a time window of an existing video",
    description=ltx_docs.RETAKE,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def retake(
    req: Annotated[RetakeRequest, Body(openapi_examples=ltx_docs.RETAKE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_retake, req)


@router.post(
    "/audio/generate",
    tags=["Audio generation"],
    summary="Text-to-audio (no video)",
    description=ltx_docs.GENERATE_AUDIO,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def generate_audio(
    req: Annotated[TextToAudioRequest, Body(openapi_examples=ltx_docs.GENERATE_AUDIO_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_text_to_audio, req)
