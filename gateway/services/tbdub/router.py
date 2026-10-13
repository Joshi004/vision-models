"""TBDub's lip-sync endpoint, mounted by server.py under /v1/tbdub.

Shared infrastructure (job status/result/cancel, uploads) lives at
the top level in server.py / common/, not here -- this router only has the
endpoint specific to lip sync.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.tbdub import dispatch
from services.tbdub import openapi_docs as tbdub_docs
from services.tbdub.schemas import LipSyncRequest

router = APIRouter()


def _submit(dispatch_fn, req) -> JobSubmitResponse:
    try:
        job_id = dispatch_fn(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except dispatch.InputError as e:
        raise HTTPException(400, str(e)) from e
    except dispatch.NotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except dispatch.ConflictError as e:
        raise HTTPException(409, str(e)) from e
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(400, f"Invalid asset reference: {e}") from e
    except dispatch.DispatchError as e:
        raise HTTPException(502, f"Could not submit Slurm job: {e}") from e
    return JobSubmitResponse(job_id=job_id, status="queued")


@router.post(
    "/videos/lip-sync",
    tags=["Lip sync"],
    summary="Lip-sync an existing video to an audio track (TBDub)",
    description=tbdub_docs.LIP_SYNC,
    response_model=JobSubmitResponse,
    responses=tbdub_docs.LIP_SYNC_RESPONSES,
)
def lip_sync(
    req: Annotated[LipSyncRequest, Body(openapi_examples=tbdub_docs.LIP_SYNC_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_lip_sync, req)
