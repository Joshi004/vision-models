"""Parakeet's own transcription endpoint, mounted by server.py under
/v1/parakeet.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
endpoint specific to Parakeet's own recipe.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.parakeet import dispatch
from services.parakeet import openapi_docs as parakeet_docs
from services.parakeet.schemas import TranscribeRequest

router = APIRouter()


def _submit(dispatch_fn, req) -> JobSubmitResponse:
    try:
        job_id = dispatch_fn(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except dispatch.InputError as e:
        raise HTTPException(400, str(e)) from e
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, f"Invalid asset reference: {e}") from e
    except dispatch.DispatchError as e:
        raise HTTPException(502, f"Could not submit Slurm job: {e}") from e
    return JobSubmitResponse(job_id=job_id, status="queued")


@router.post(
    "/transcribe",
    tags=["Transcription"],
    summary="Transcribe an audio or video file, with word- and segment-level timestamps",
    description=parakeet_docs.TRANSCRIBE,
    response_model=JobSubmitResponse,
    responses=parakeet_docs.TRANSCRIBE_RESPONSES,
)
def transcribe(
    req: Annotated[TranscribeRequest, Body(openapi_examples=parakeet_docs.TRANSCRIBE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_transcribe, req)
