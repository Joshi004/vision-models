"""Stable Audio 3 Medium's audio endpoints, mounted by server.py under /v1/stable-audio3.

Shared infrastructure (job status/result/cancel, uploads) lives at the top
level in server.py / common/, not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.stable_audio3 import dispatch
from services.stable_audio3 import openapi_docs as sa3_docs
from services.stable_audio3.schemas import StableAudio3GenerateRequest, StableAudio3InpaintRequest, StableAudio3TransformRequest

router = APIRouter()


def _submit(submit, req) -> JobSubmitResponse:
    """Run one dispatch function and translate its errors to HTTP errors."""
    try:
        submission = submit(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except dispatch.InputError as e:
        raise HTTPException(400, str(e)) from e
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, f"Invalid asset reference: {e}") from e
    except dispatch.DispatchError as e:
        raise HTTPException(502, f"Could not submit Slurm job: {e}") from e
    return JobSubmitResponse(job_id=submission.job_id, status=submission.status, reused=submission.reused)


@router.post(
    "/audio/generate",
    tags=["Music generation"],
    summary="Generate instrumental music, a stem, an instrument or a sound effect from text",
    description=sa3_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=sa3_docs.GENERATE_RESPONSES,
)
def generate(
    req: Annotated[StableAudio3GenerateRequest, Body(openapi_examples=sa3_docs.GENERATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_generate, req)


@router.post(
    "/audio/transform",
    tags=["Music generation"],
    summary="Transform an existing recording from a prompt (audio-to-audio)",
    description=sa3_docs.TRANSFORM,
    response_model=JobSubmitResponse,
    responses=sa3_docs.AUDIO_INPUT_RESPONSES,
)
def transform(
    req: Annotated[StableAudio3TransformRequest, Body(openapi_examples=sa3_docs.TRANSFORM_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_transform, req)


@router.post(
    "/audio/inpaint",
    tags=["Music generation"],
    summary="Regenerate part of a recording, or extend it",
    description=sa3_docs.INPAINT,
    response_model=JobSubmitResponse,
    responses=sa3_docs.AUDIO_INPUT_RESPONSES,
)
def inpaint(
    req: Annotated[StableAudio3InpaintRequest, Body(openapi_examples=sa3_docs.INPAINT_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_inpaint, req)
