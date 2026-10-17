"""MiniMax H3's own generation endpoints, mounted by server.py under /v1/h3.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
endpoints that are specific to H3's two recipes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common import openapi_docs as common_docs
from common.schemas import JobSubmitResponse
from services.h3 import dispatch
from services.h3 import openapi_docs as h3_docs
from services.h3.schemas import H3AudioToVideoRequest, H3GenerateRequest

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
    summary="Text, first-frame or last-frame to video with sound on MiniMax H3",
    description=h3_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def generate_video(
    req: Annotated[H3GenerateRequest, Body(openapi_examples=h3_docs.GENERATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_generate, req)


@router.post(
    "/videos/audio-to-video",
    tags=["Video generation"],
    summary="Video with sound from image/video references and an optional audio reference (MiniMax H3)",
    description=h3_docs.AUDIO_TO_VIDEO,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def audio_to_video(
    req: Annotated[H3AudioToVideoRequest, Body(openapi_examples=h3_docs.AUDIO_TO_VIDEO_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_audio_to_video, req)
