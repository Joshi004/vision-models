"""LTX-2.5's own generation endpoints, mounted by server.py under /v1/ltx25.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
endpoints that are specific to LTX-2.5's three recipes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common import openapi_docs as common_docs
from common.schemas import JobSubmitResponse
from services.ltx25 import dispatch, openapi_docs as ltx25_docs
from services.ltx25.schemas import Ltx25InterpolateRequest, Ltx25RetakeRequest, Ltx25TextToVideoRequest

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
    summary="Text/image-to-video on LTX-2.5 (fast or quality)",
    description=ltx25_docs.GENERATE_VIDEO,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def generate_video(
    req: Annotated[Ltx25TextToVideoRequest, Body(openapi_examples=ltx25_docs.GENERATE_VIDEO_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_text_to_video, req)


@router.post(
    "/videos/interpolate",
    tags=["Video generation"],
    summary="Generate the motion between a first and a last frame (LTX-2.5)",
    description=ltx25_docs.INTERPOLATE,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def interpolate(
    req: Annotated[Ltx25InterpolateRequest, Body(openapi_examples=ltx25_docs.INTERPOLATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_interpolate, req)


@router.post(
    "/videos/retake",
    tags=["Video generation"],
    summary="Regenerate a time window of an existing video (LTX-2.5)",
    description=ltx25_docs.RETAKE,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def retake(
    req: Annotated[Ltx25RetakeRequest, Body(openapi_examples=ltx25_docs.RETAKE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_retake, req)
