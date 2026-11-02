"""ACE-Step 1.5's music endpoints, mounted by server.py under /v1/ace-step15.

Shared infrastructure (job status/result/cancel, uploads) lives at the top
level in server.py / common/, not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.ace_step15 import dispatch
from services.ace_step15 import openapi_docs as ace_docs
from services.ace_step15.schemas import AceStepCoverRequest, AceStepGenerateRequest, AceStepRepaintRequest

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
    "/music/generate",
    tags=["Music generation"],
    summary="Generate a song or instrumental from a text description and lyrics",
    description=ace_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=ace_docs.GENERATE_RESPONSES,
)
def generate(
    req: Annotated[AceStepGenerateRequest, Body(openapi_examples=ace_docs.GENERATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_generate, req)


@router.post(
    "/music/cover",
    tags=["Music generation"],
    summary="Restyle an existing recording, keeping its structure",
    description=ace_docs.COVER,
    response_model=JobSubmitResponse,
    responses=ace_docs.AUDIO_INPUT_RESPONSES,
)
def cover(
    req: Annotated[AceStepCoverRequest, Body(openapi_examples=ace_docs.COVER_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_cover, req)


@router.post(
    "/music/repaint",
    tags=["Music generation"],
    summary="Regenerate one time window of an existing recording",
    description=ace_docs.REPAINT,
    response_model=JobSubmitResponse,
    responses=ace_docs.AUDIO_INPUT_RESPONSES,
)
def repaint(
    req: Annotated[AceStepRepaintRequest, Body(openapi_examples=ace_docs.REPAINT_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_repaint, req)
