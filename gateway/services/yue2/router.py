"""YuE2's song endpoints, mounted by server.py under /v1/yue2.

Shared infrastructure (job status/result/cancel, uploads) lives at the top
level in server.py / common/, not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.yue2 import dispatch
from services.yue2 import openapi_docs as yue2_docs
from services.yue2.schemas import Yue2CoverRequest, Yue2GenerateRequest

router = APIRouter()


def _submit(submit, req) -> JobSubmitResponse:
    """Run one dispatch function and translate its errors to HTTP errors."""
    try:
        submission = submit(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except dispatch.DispatchError as e:
        raise HTTPException(502, f"Could not submit Slurm job: {e}") from e
    return JobSubmitResponse(job_id=submission.job_id, status=submission.status, reused=submission.reused)


@router.post(
    "/songs/generate",
    tags=["Music generation"],
    summary="Generate a full song with sung vocals from a style and lyrics",
    description=yue2_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=yue2_docs.GENERATE_RESPONSES,
)
def generate(
    req: Annotated[Yue2GenerateRequest, Body(openapi_examples=yue2_docs.GENERATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_generate, req)


@router.post(
    "/songs/cover",
    tags=["Music generation"],
    summary="Generate a song that follows a melody score (ABC) you supply",
    description=yue2_docs.COVER,
    response_model=JobSubmitResponse,
    responses=yue2_docs.COVER_RESPONSES,
)
def cover(
    req: Annotated[Yue2CoverRequest, Body(openapi_examples=yue2_docs.COVER_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_cover, req)
