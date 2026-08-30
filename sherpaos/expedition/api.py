"""HTTP API for real expedition memory uploads and grounded evidence access."""

from __future__ import annotations

import base64
import json
import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sherpaos.expedition.storage import ExpeditionStore
from sherpaos.runtime.live import live_evidence
from sherpaos.voice.tools import ExpeditionVoiceTools

app = FastAPI(title="SherpaOS Expedition Memory API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "SHERPA_WEB_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
store = ExpeditionStore()
voice_tools = ExpeditionVoiceTools(store)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/supervisor/current")
def supervisor_current() -> dict[str, object]:
    if not live_evidence.events:
        raise HTTPException(status_code=404, detail="no live supervisor evidence")
    return live_evidence.events[-1]


@app.get("/api/supervisor/events")
def supervisor_events(limit: int = 100) -> list[dict[str, object]]:
    safe_limit = min(500, max(1, limit))
    return list(live_evidence.events)[-safe_limit:]


@app.websocket("/ws/supervisor")
async def supervisor_socket(socket: WebSocket) -> None:
    await socket.accept()
    queue = live_evidence.subscribe()
    try:
        if live_evidence.events:
            await socket.send_json(live_evidence.events[-1])
        while True:
            await socket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        live_evidence.unsubscribe(queue)


@app.get("/api/expeditions/{expedition_id}/days")
def list_days(expedition_id: str) -> list[dict[str, object]]:
    try:
        return [manifest.to_dict() for manifest in store.list_days(expedition_id)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/expeditions/{expedition_id}/days/{day}")
def get_day(expedition_id: str, day: int) -> dict[str, object]:
    try:
        manifest = store.get_manifest(expedition_id, day)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if manifest is None:
        raise HTTPException(status_code=404, detail="no bag uploaded for this day")
    return manifest.to_dict()


@app.post("/api/expeditions/{expedition_id}/days/{day}/upload", status_code=201)
def upload_day(
    expedition_id: str,
    day: int,
    bag: Annotated[UploadFile, File(description="A genuine .mcap or rosbag2 .db3 recording")],
) -> dict[str, object]:
    try:
        return store.ingest(expedition_id, day, bag.filename or "recording", bag.file).to_dict()
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/voice/tools/day-overview/{expedition_id}/{day}")
def voice_day_overview(expedition_id: str, day: int) -> dict[str, object]:
    try:
        return voice_tools.get_day_overview(expedition_id, day)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/voice/ask/{expedition_id}/{day}")
def ask_by_radio(
    expedition_id: str,
    day: int,
    audio: Annotated[UploadFile, File(description="One completed push-to-talk transmission")],
) -> dict[str, object]:
    """Transcribe one radio turn, answer from verified evidence, and synthesize speech."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    overview = voice_tools.get_day_overview(expedition_id, day)
    if not overview.get("available"):
        raise HTTPException(status_code=409, detail=overview["limitation"])
    try:
        from openai import OpenAI

        client = OpenAI()
        payload = audio.file.read()
        transcript = client.audio.transcriptions.create(
            model="gpt-transcribe",
            file=(audio.filename or "radio.webm", payload, audio.content_type or "audio/webm"),
        ).text
        prompt = _grounded_prompt(transcript, overview)
        response = client.responses.create(
            model=os.environ.get("SHERPA_ANALYSIS_MODEL", "gpt-5.6-luna"),
            input=prompt,
        )
        answer = response.output_text.strip()
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=os.environ.get("SHERPA_VOICE", "coral"),
            input=answer,
            response_format="mp3",
        )
        audio_bytes = speech.read()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"voice service failed: {exc}") from exc
    return {
        "question": transcript,
        "answer": answer,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "audio_mime": "audio/mpeg",
        "evidence": overview["evidence"],
    }


def _grounded_prompt(question: str, overview: dict[str, object]) -> str:
    return (
        "You are Pemba's half-duplex field-radio voice. Answer only from the verified "
        "ROS bag evidence JSON below. Never invent telemetry, events, causes, or values. "
        "If the evidence cannot answer the question, say exactly what is unavailable. "
        "Use at most three short sentences and end with 'Over.'\n\n"
        f"OPERATOR QUESTION:\n{question}\n\n"
        f"VERIFIED BAG EVIDENCE:\n{json.dumps(overview, sort_keys=True)}"
    )
