"""Breeze TTS 2's text-to-speech endpoint, mounted by server.py under
/v1/breeze-tts.

Shared infrastructure (job status/result/cancel, uploads) lives at the top
level in server.py / common/, not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.breeze_tts import dispatch
from services.breeze_tts import openapi_docs as breeze_docs
from services.breeze_tts.schemas import SynthesizeRequest

router = APIRouter()


@router.post(
    "/synthesize",
    tags=["Text to speech"],
    summary="Synthesize speech from text (voice design, clone, or direction)",
    description=breeze_docs.SYNTHESIZE,
    response_model=JobSubmitResponse,
    responses=breeze_docs.SYNTHESIZE_RESPONSES,
)
def synthesize(
    req: Annotated[SynthesizeRequest, Body(openapi_examples=breeze_docs.SYNTHESIZE_EXAMPLES)],
) -> JobSubmitResponse:
    try:
        job_id = dispatch.dispatch_synthesize(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except dispatch.InputError as e:
        raise HTTPException(400, str(e)) from e
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, f"Invalid asset reference: {e}") from e
    except dispatch.DispatchError as e:
        raise HTTPException(502, f"Could not submit Slurm job: {e}") from e
    return JobSubmitResponse(job_id=job_id, status="queued")
