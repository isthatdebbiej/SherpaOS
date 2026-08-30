"""Read-only public API for live supervisor evidence and grounded radio Q&A."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sherpaos.runtime.live import live_evidence

app = FastAPI(title="SherpaOS Live Evidence API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        x.strip()
        for x in os.environ.get("SHERPA_WEB_ORIGINS", "http://127.0.0.1:3000").split(",")
        if x.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/supervisor/current")
def current() -> dict[str, object]:
    event = live_evidence.current()
    if event is None:
        raise HTTPException(404, "no live supervisor evidence")
    return event


@app.get("/api/supervisor/events")
def events(limit: int = 100) -> list[dict[str, object]]:
    return live_evidence.history(min(500, max(1, limit)))


def _memory(day: int) -> dict[str, object]:
    if day not in range(1, 5):
        raise HTTPException(404, "journal memory is available only for days 1-4")
    root = Path(os.environ.get("SHERPA_MEMORY_DIR", "web/src/data/memories"))
    path = root / f"day-{day:02d}.json"
    if not path.exists():
        raise HTTPException(404, "verified memory file is unavailable")
    return json.loads(path.read_text(encoding="utf-8"))


def _radio_evidence(day: int) -> dict[str, object]:
    memory = _memory(day)
    episodes = memory["episodes"]
    windows = memory["windows"]
    positives = memory["positive_windows"]
    outcomes = memory["physical_outcomes"]
    return {
        "mission_day": memory["day"],
        "scenario_family": memory["scenario_family"],
        "mission_rehearsal": True,
        "route_trials_reviewed": episodes["train"] + episodes["validation"],
        "motion_windows_reviewed": windows["train"] + windows["validation"],
        "mobility_risk_windows": (
            positives["train"]["mobility"] + positives["validation"]["mobility"]
        ),
        "body_risk_windows": (
            positives["train"]["dynamics"] + positives["validation"]["dynamics"]
        ),
        "physical_boundary_outcomes": (
            outcomes["train_falls"] + outcomes["validation_falls"]
        ),
        "guard_context": memory["context"],
        "latest_live_decision": live_evidence.current(),
    }


def _answer(question: str, day: int) -> tuple[str, dict[str, object]]:
    evidence = _radio_evidence(day)
    if "stop" in question.casefold() and evidence["latest_live_decision"] is None:
        return (
            "I have no live stop decision or actuation receipt to explain. Start the live "
            "supervisor run, then ask again with its decision evidence. Over.",
            evidence,
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise HTTPException(503, "HF_TOKEN is not configured")
    from huggingface_hub import InferenceClient

    client = InferenceClient(provider="auto", api_key=token)
    response = client.chat_completion(
        model=os.environ.get("SHERPA_HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Pemba, a robot field-radio assistant. Answer only from supplied JSON. "
                    "Never invent measurements or claim simulation is real-world data. For stop "
                    "questions, prioritize the latest live decision, guard reasons, decision ID, "
                    "and actuation receipt. If evidence is absent, say so. Use no more than three "
                    "short sentences and end with 'Over.'"
                ),
            },
            {"role": "user", "content": f"Question: {question}\nEvidence: {json.dumps(evidence)}"},
        ],
        max_tokens=180,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip(), evidence


@app.get("/api/memory/days")
def memory_days() -> list[dict[str, object]]:
    return [_memory(day) for day in range(1, 5)]


@app.post("/api/radio/ask-text/{day}")
def ask_text(day: int, question: str) -> dict[str, object]:
    answer, evidence = _answer(question, day)
    return {"question": question, "answer": answer, "evidence": evidence}


@app.post("/api/voice/ask/everest-001/{day}")
def ask_voice(
    day: int,
    audio: Annotated[UploadFile, File(description="One push-to-talk transmission")],
) -> dict[str, object]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise HTTPException(503, "HF_TOKEN is not configured")
    from huggingface_hub import InferenceClient

    client = InferenceClient(provider="auto", api_key=token)
    transcript = client.automatic_speech_recognition(
        audio.file.read(),
        model=os.environ.get("SHERPA_HF_ASR_MODEL", "openai/whisper-large-v3"),
    ).text.strip()
    answer, evidence = _answer(transcript, day)
    speech = _synthesize_speech(answer)
    return {
        "question": transcript,
        "answer": answer,
        "audio_base64": base64.b64encode(speech).decode("ascii"),
        "audio_mime": "audio/wav",
        "evidence": evidence,
    }


_TTS_MODEL = None
_TTS_TOKENIZER = None


def _synthesize_speech(text: str) -> bytes:
    """Run a Hugging Face MMS-TTS checkpoint locally on Vultr CPU."""
    global _TTS_MODEL, _TTS_TOKENIZER
    import io

    import numpy as np
    import torch
    from scipy.io import wavfile
    from transformers import AutoTokenizer, VitsModel

    model_id = os.environ.get("SHERPA_HF_TTS_MODEL", "facebook/mms-tts-eng")
    if _TTS_MODEL is None or _TTS_TOKENIZER is None:
        _TTS_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _TTS_MODEL = VitsModel.from_pretrained(model_id).eval()
    inputs = _TTS_TOKENIZER(text, return_tensors="pt")
    with torch.no_grad():
        waveform = _TTS_MODEL(**inputs).waveform[0].cpu().numpy()
    peak = max(float(np.abs(waveform).max()), 1e-6)
    pcm = np.int16(np.clip(waveform / peak, -1.0, 1.0) * 32767)
    output = io.BytesIO()
    wavfile.write(output, int(_TTS_MODEL.config.sampling_rate), pcm)
    return output.getvalue()

@app.websocket("/ws/supervisor")
async def socket(websocket: WebSocket) -> None:
    await websocket.accept()
    last_id: str | None = None
    try:
        while True:
            event = live_evidence.current()
            decision_id = str((event or {}).get("decision", {}).get("decision_id", ""))
            if event is not None and decision_id != last_id:
                await websocket.send_json(event)
                last_id = decision_id
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
