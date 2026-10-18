"""Qwen3.8's own endpoints, mounted by server.py under /v1/qwen38.

Shared infrastructure (job status/result/cancel) lives in server.py /
common/. This router has the text endpoint and two small endpoints about the
shared model server behind it (see server_manager.py).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

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
) -> JobSubmitResponse:
    try:
        job_id = await dispatch.submit_generate(req)
    except dispatch.CapacityError as e:
        raise HTTPException(429, str(e)) from e
    return JobSubmitResponse(job_id=job_id, status="queued")


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
