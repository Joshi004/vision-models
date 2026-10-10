"""Qwen-Image-2.1's image endpoints, mounted by server.py under
/v1/qwen-image21.

Shared infrastructure (job status/result/cancel, uploads) lives at the top
level in server.py / common/, not here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from common.schemas import JobSubmitResponse
from services.qwen_image21 import dispatch
from services.qwen_image21 import openapi_docs as qwen_docs
from services.qwen_image21.schemas import QwenImageEditRequest, QwenImageGenerateRequest

router = APIRouter()


def _submit(submit, req) -> JobSubmitResponse:
    """Run one dispatch function and translate its errors to HTTP errors."""
    try:
        job_id = submit(req)
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
    "/images/generate",
    tags=["Image generation"],
    summary="Generate an image from text (including transparent RGBA images)",
    description=qwen_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=qwen_docs.GENERATE_RESPONSES,
)
def generate(
    req: Annotated[QwenImageGenerateRequest, Body(openapi_examples=qwen_docs.GENERATE_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_generate, req)


@router.post(
    "/images/edit",
    tags=["Image generation"],
    summary="Edit an image, or compose a new one from up to 10 reference images",
    description=qwen_docs.EDIT,
    response_model=JobSubmitResponse,
    responses=qwen_docs.EDIT_RESPONSES,
)
def edit(
    req: Annotated[QwenImageEditRequest, Body(openapi_examples=qwen_docs.EDIT_EXAMPLES)],
) -> JobSubmitResponse:
    return _submit(dispatch.dispatch_edit, req)
