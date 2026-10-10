"""Wan-Animate v1's own generation endpoint, mounted by server.py under
/v1/wan-animate.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
one endpoint specific to Wan-Animate v1's replace-mode recipe.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common import openapi_docs as common_docs
from common.schemas import JobSubmitResponse
from services.wan_animate import dispatch, openapi_docs as wan_animate_docs
from services.wan_animate.schemas import ReplaceRequest

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
    "/videos/replace",
    tags=["Video generation"],
    summary="Swap a character into an existing video (replace mode)",
    description=wan_animate_docs.REPLACE,
    response_model=JobSubmitResponse,
    responses=common_docs.SUBMIT_RESPONSES,
)
def replace_character(
    req: Annotated[ReplaceRequest, Body(openapi_examples=wan_animate_docs.REPLACE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_replace, req)
