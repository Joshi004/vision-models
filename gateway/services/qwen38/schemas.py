"""Pydantic models for the Qwen3.8 text endpoint and its server status.

Text only: there is no image, audio or video input (the server runs the
model without its vision tower). Everything the caller sends is in
`messages`, the same shape as a chat API.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from common.schemas import IncognitoOptionMixin
from services.qwen38 import config


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Who wrote the message. `system` is allowed only as the first message."
    )
    content: str = Field(..., min_length=1, description="The message text.")


class Qwen38GenerateRequest(IncognitoOptionMixin):
    """Request body for POST /v1/qwen38/generate."""

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description=(
            "The conversation so far, oldest first. The last message must be from the `user`. "
            f"All messages together may hold at most {config.MAX_PROMPT_CHARS:,} characters."
        ),
        examples=[
            [
                {"role": "system", "content": "You plan video scenes. Answer only with JSON."},
                {"role": "user", "content": "A 20-second ad for a trail running shoe."},
            ]
        ],
    )
    json_schema: dict | None = Field(
        None,
        description=(
            "A JSON Schema (an object) the answer must follow. When set, the server only produces text "
            "that matches the schema, so `json` in the result is always parseable -- unless the answer "
            "was cut off by `max_tokens` (check `finish_reason`). When omitted, the answer is free text."
        ),
        examples=[None],
    )
    thinking: bool = Field(
        True,
        description=(
            "Let the model reason before it answers. Better on hard planning, but it uses `max_tokens` "
            "and takes longer (the reasoning counts toward `max_tokens`). Set false for fast, direct answers."
        ),
    )
    include_reasoning: bool = Field(
        False,
        description="Return the model's reasoning text in the result (`reasoning`). Only meaningful with `thinking: true`.",
    )
    max_tokens: int = Field(
        config.DEFAULT_MAX_TOKENS,
        ge=1,
        le=config.MAX_OUTPUT_TOKENS,
        description=(
            "Upper limit on generated tokens, reasoning included. If the limit is reached the answer is cut "
            "off (`finish_reason: \"length\"`)."
        ),
    )
    temperature: float = Field(0.6, ge=0.0, le=2.0, description="Sampling temperature. 0 makes answers nearly deterministic.")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="Nucleus sampling cutoff.")
    seed: int | None = Field(
        None, ge=0, le=2**31 - 1, description="Seed for repeatable sampling. Same seed and request usually gives the same answer."
    )

    @field_validator("json_schema")
    @classmethod
    def _schema_size(cls, value: dict | None) -> dict | None:
        if value is not None and len(json.dumps(value)) > config.MAX_SCHEMA_BYTES:
            raise ValueError(f"json_schema is larger than {config.MAX_SCHEMA_BYTES} bytes.")
        return value

    @model_validator(mode="after")
    def _check_conversation(self) -> Qwen38GenerateRequest:
        if self.messages[-1].role != "user":
            raise ValueError("The last message must have role 'user'.")
        if any(m.role == "system" for m in self.messages[1:]):
            raise ValueError("A 'system' message is allowed only as the first message.")
        total = sum(len(m.content) for m in self.messages)
        if total > config.MAX_PROMPT_CHARS:
            raise ValueError(f"The messages hold {total:,} characters; the limit is {config.MAX_PROMPT_CHARS:,}.")
        return self


class ServerStatus(BaseModel):
    """Response of GET /v1/qwen38/server and POST /v1/qwen38/server/warm."""

    state: Literal["stopped", "starting", "ready"] = Field(
        ...,
        description=(
            "`stopped`: no server is running; the next request starts one. `starting`: the Slurm job is "
            "queued or loading the model. `ready`: requests are answered right away."
        ),
    )
    slurm_job_id: str | None = Field(None, description="Slurm job id of the server, when one exists.")
    partition: str = Field(..., description="Partition the server runs on (or would run on).")
    requests_in_flight: int = Field(..., description="Requests this gateway is waiting on right now.")
    idle_shutdown_at: str | None = Field(
        None,
        description=(
            "UTC time the server stops if no request arrives before then. Only set while it is `ready` and idle."
        ),
    )
    idle_minutes: float = Field(..., description="Minutes without a request after which the server stops itself.")
    model: str = Field(..., description="The model the server runs.")
