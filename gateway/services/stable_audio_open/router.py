"""Stable Audio Open 1.0's audio endpoints, mounted by server.py under /v1/stable-audio-open.

Shared infrastructure (job status/result/cancel, uploads) lives at the top
level in server.py / common/, not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.stable_audio_open import dispatch
from services.stable_audio_open import openapi_docs as sao_docs
from services.stable_audio_open.schemas import StableAudioOpenGenerateRequest, StableAudioOpenTransformRequest

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
    summary="Generate a sound effect, loop or short clip (up to 47 s) from text",
    description=sao_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=sao_docs.GENERATE_RESPONSES,
)
def generate(
    req: Annotated[StableAudioOpenGenerateRequest, Body(openapi_examples=sao_docs.GENERATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_generate, req)


@router.post(
    "/audio/transform",
    tags=["Music generation"],
    summary="Re-imagine an existing recording from a prompt (audio-to-audio)",
    description=sao_docs.TRANSFORM,
    response_model=JobSubmitResponse,
    responses=sao_docs.TRANSFORM_RESPONSES,
)
def transform(
    req: Annotated[StableAudioOpenTransformRequest, Body(openapi_examples=sao_docs.TRANSFORM_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_transform, req)
