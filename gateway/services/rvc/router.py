"""RVC/Applio's own voice-conversion endpoints, mounted by server.py under
/v1/rvc.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
endpoints specific to RVC's own recipes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.rvc import config, dispatch
from services.rvc import openapi_docs as rvc_docs
from services.rvc.schemas import BatchConvertRequest, ConvertRequest, VoiceListResponse

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


@router.get(
    "/voices",
    tags=["Voice conversion"],
    summary="List available voice models",
    description=rvc_docs.VOICES_LIST,
    response_model=VoiceListResponse,
)
def list_voices() -> VoiceListResponse:
    return VoiceListResponse(voices=sorted(config.VOICE_REGISTRY))


@router.post(
    "/convert",
    tags=["Voice conversion"],
    summary="Convert a recording to one of the installed voices",
    description=rvc_docs.CONVERT,
    response_model=JobSubmitResponse,
    responses=rvc_docs.CONVERT_RESPONSES,
)
def convert_voice(
    req: Annotated[ConvertRequest, Body(openapi_examples=rvc_docs.CONVERT_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_convert, req)


@router.post(
    "/batch-convert",
    tags=["Voice conversion"],
    summary="Convert many recordings to one installed voice in a single job",
    description=rvc_docs.BATCH_CONVERT,
    response_model=JobSubmitResponse,
    responses=rvc_docs.BATCH_CONVERT_RESPONSES,
)
def batch_convert_voice(
    req: Annotated[BatchConvertRequest, Body(openapi_examples=rvc_docs.BATCH_CONVERT_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_batch_convert, req)
