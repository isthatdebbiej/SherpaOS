"""Read-only public API for live supervisor evidence."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sherpaos.runtime.live import live_evidence

app = FastAPI(title="SherpaOS Live Evidence API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        x.strip()
        for x in os.environ.get("SHERPA_WEB_ORIGINS", "http://127.0.0.1:3000").split(",")
        if x.strip()
    ],
    allow_methods=["GET"],
    allow_headers=[],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/supervisor/current")
def current() -> dict[str, object]:
    if not live_evidence.events:
        raise HTTPException(404, "no live supervisor evidence")
    return live_evidence.events[-1]


@app.get("/api/supervisor/events")
def events(limit: int = 100) -> list[dict[str, object]]:
    return list(live_evidence.events)[-min(500, max(1, limit)) :]


@app.websocket("/ws/supervisor")
async def socket(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = live_evidence.subscribe()
    try:
        if live_evidence.events:
            await websocket.send_json(live_evidence.events[-1])
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        live_evidence.unsubscribe(queue)
