"""Who submitted a job: this web UI, or some other client (curl, a script, an agent).

The web UI sends `X-Job-Source: ui` with every job submission (ui/src/api.js's postJSON). A plain
ASGI middleware (same style as server.py's UploadSizeGuard, so streamed responses are not
affected) turns that header into a context variable for the length of the request, and
job_store.create_job reads it -- so no backend's dispatch code has to pass it along.

The header is only a label, not authentication: any client can send it.
"""

from __future__ import annotations

from contextvars import ContextVar

SOURCE_UI = "ui"
SOURCE_API = "api"
HEADER_NAME = b"x-job-source"

# None outside a request (nothing there creates jobs today); the stored value is then NULL.
_current_source: ContextVar[str | None] = ContextVar("job_source", default=None)


def current_source() -> str | None:
    return _current_source.get()


class JobSourceMiddleware:
    """Sets the context variable to "ui" when the request carries `X-Job-Source: ui`, else "api"."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        header = dict(scope["headers"]).get(HEADER_NAME, b"").strip().lower()
        token = _current_source.set(SOURCE_UI if header == SOURCE_UI.encode() else SOURCE_API)
        try:
            await self.inner(scope, receive, send)
        finally:
            _current_source.reset(token)
