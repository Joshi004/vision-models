"""A FileResponse whose `background` task runs only if the client stayed
connected until the last chunk was handed to the server.

Why not plain FileResponse + `background=`: Starlette runs the background
task after its send loop finishes, but uvicorn silently discards every write
after the client has disconnected, so an aborted download still "finishes"
and the task would run -- here, deleting a result the client never received.
This class listens for the ASGI `http.disconnect` message while it sends, and
skips the background task if the client left before the final chunk started.

What this cannot know: whether the client actually saved the file. Bytes the
server has already handed to the operating system (the last chunk, plus
whatever fits in the socket buffers) count as sent; a client that disconnects
after that point is indistinguishable from one that finished.
"""

from __future__ import annotations

import anyio
from starlette.responses import FileResponse
from starlette.types import Message, Receive, Scope, Send


class OneTimeFileResponse(FileResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        background = self.background
        self.background = None  # run it ourselves below, only if the download really completed
        client_left_early = False
        last_chunk_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal last_chunk_started
            final_body = message["type"] == "http.response.body" and not message.get("more_body", False)
            if final_body or message["type"] == "http.response.pathsend":
                last_chunk_started = True
            await send(message)

        async def watch_for_disconnect() -> None:
            nonlocal client_left_early
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    if not last_chunk_started:
                        client_left_early = True
                    return

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(watch_for_disconnect)
            await super().__call__(scope, receive, tracking_send)
            task_group.cancel_scope.cancel()

        if background is not None and not client_left_early:
            await background()
