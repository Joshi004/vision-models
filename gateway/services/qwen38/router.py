"""Qwen3.8's own endpoints, mounted by server.py under /v1/qwen38.

Shared infrastructure (job status/result/cancel) lives in server.py /
common/. This router has the text endpoint and two small endpoints about the
shared model server behind it (see server_manager.py).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, Header, HTTPException

from common.schemas import JobSubmitResponse
from services.qwen38 import dispatch, server_manager
from services.qwen38 import openapi_docs as qwen38_docs
from services.qwen38.schemas import Qwen38GenerateRequest, ServerStatus

router = APIRouter()


@router.post(
    "/generate",
    tags=["Text model"],
    summary="Ask the text model a question, optionally with a JSON schema for the answer",
    description=qwen38_docs.GENERATE,
    response_model=JobSubmitResponse,
    responses=qwen38_docs.GENERATE_RESPONSES,
)
async def generate(
    req: Annotated[Qwen38GenerateRequest, Body(openapi_examples=qwen38_docs.GENERATE_EXAMPLES)],
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=1,
            max_length=200,
            description=(
                "A key you choose (for example a UUID) for this one request. If a request with the same key "
                "was sent before, no new job is created: the response carries the earlier job's `job_id`, "
                "with `reused: true`, as long as that job is still queued or running, or finished "
                "successfully and its answer is still available. Resend the same request with the same key "
                "after a timeout or a lost response, and it is not run twice. Using the same key for a "
                "different request is refused with 422. Ignored when the request has `incognito: true`."
            ),
        ),
    ] = None,
) -> JobSubmitResponse:
    try:
        submission = await dispatch.submit_generate(req, idempotency_key)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    except dispatch.IdempotencyConflict as e:
        raise HTTPException(422, str(e)) from e
    return JobSubmitResponse(job_id=submission.job_id, status=submission.status, reused=submission.reused)


@router.get(
    "/server",
    tags=["Text model"],
    summary="State of the shared model server",
    description=qwen38_docs.SERVER,
    response_model=ServerStatus,
)
async def server_status() -> ServerStatus:
    return await asyncio.to_thread(server_manager.get_status)


@router.post(
    "/server/warm",
    tags=["Text model"],
    summary="Start the model server now, before the first request",
    description=qwen38_docs.WARM,
    response_model=ServerStatus,
    responses=qwen38_docs.WARM_RESPONSES,
)
async def warm_server() -> ServerStatus:
    try:
        await asyncio.to_thread(server_manager.ensure_server)
    except server_manager.ServerError as e:
        raise HTTPException(502, str(e)) from e
    return await asyncio.to_thread(server_manager.get_status)
