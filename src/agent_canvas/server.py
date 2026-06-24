"""FastAPI server + WebSocket endpoint for Agent Canvas.

Serves the static trace viewer and streams trace events over WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as _queue
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent_canvas.storage import InMemoryStore, get_default_store
from agent_canvas.trace import TraceEvent

logger = logging.getLogger("agent_canvas.server")

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def create_app(store: InMemoryStore | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        store: In-memory store. Defaults to the module-level singleton.
    """
    store = store or get_default_store()
    app = FastAPI(title="Agent Canvas", version="0.1.0")

    # ---- REST endpoints ----

    @app.get("/runs")
    async def list_runs() -> list[dict]:
        return store.list_runs()

    @app.get("/runs/{run_id}", response_model=None)
    async def get_run(run_id: str):
        rec = store.get_run(run_id)
        if rec is None:
            return JSONResponse({"error": "run not found"}, status_code=404)
        return rec.to_summary()

    @app.get("/runs/{run_id}/events")
    async def get_run_events(run_id: str) -> list[dict]:
        return [e.to_dict() for e in store.get_events(run_id)]

    # ---- WebSocket ----

    @app.websocket("/ws/trace/{run_id}")
    async def ws_trace(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        sid, queue = store.subscribe(run_id)

        # Send existing events first (for replay / late joiners)
        existing = store.get_events(run_id)
        for event in existing:
            try:
                await websocket.send_text(event.to_json())
            except Exception:
                break

        # Then stream live events
        try:
            while True:
                # Poll the queue with a short timeout
                try:
                    event = await asyncio.wait_for(
                        asyncio.to_thread(queue.get, timeout=0.5),
                        timeout=1.0,
                    )
                    await websocket.send_text(event.to_json())
                except (asyncio.TimeoutError, _queue.Empty):
                    # Send a keepalive ping
                    try:
                        await websocket.send_json({"type": "__ping__"})
                    except Exception:
                        break
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            store.unsubscribe(run_id, sid)

    # ---- Static files ----

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    store: InMemoryStore | None = None,
    open_browser: bool = True,
) -> None:
    """Start the Agent Canvas server.

    Args:
        host: Bind address.
        port: HTTP port.
        store: In-memory store.
        open_browser: Whether to open the browser automatically.
    """
    app = create_app(store=store)
    url = f"http://{host}:{port}"

    if open_browser:
        logger.info("Opening browser at %s", url)
        webbrowser.open(url)

    logger.info("Agent Canvas server starting at %s", url)
    uvicorn.run(app, host=host, port=port, log_level="info")
