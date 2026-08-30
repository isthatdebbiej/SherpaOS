"""Live decision evidence bus shared by simulator and the Vultr API."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sherpaos.contracts import ActuationReceipt, GuardDecision


class LiveEvidenceBus:
    def __init__(self) -> None:
        self.events: deque[dict[str, Any]] = deque(maxlen=1000)
        self.subscribers: set[asyncio.Queue] = set()
        state_dir = Path(os.environ.get("SHERPA_LIVE_STATE_DIR", "artifacts/live"))
        self.current_file = state_dir / "current.json"
        self.events_file = state_dir / "events.jsonl"

    def publish(
        self,
        decision: GuardDecision,
        receipt: ActuationReceipt,
        *,
        requested_velocity_mps: float,
        applied_velocity_mps: float,
    ) -> dict[str, Any]:
        if decision.decision_id != receipt.decision_id:
            raise ValueError("decision and receipt IDs differ")
        event = {
            "type": "supervisor_decision",
            "decision": asdict(decision),
            "receipt": asdict(receipt),
            "requested_velocity_mps": requested_velocity_mps,
            "applied_velocity_mps": applied_velocity_mps,
        }
        self.events.append(event)
        self._persist(event)
        for queue in tuple(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return event

    def current(self) -> dict[str, Any] | None:
        if self.current_file.exists():
            try:
                return json.loads(self.current_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return self.events[-1] if self.events else None

    def history(self, limit: int) -> list[dict[str, Any]]:
        if self.events_file.exists():
            try:
                lines = self.events_file.read_text(encoding="utf-8").splitlines()
                return [json.loads(line) for line in lines[-limit:]]
            except (OSError, json.JSONDecodeError):
                return []
        return list(self.events)[-limit:]

    def _persist(self, event: dict[str, Any]) -> None:
        self.current_file.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(event, separators=(",", ":"), default=str)
        temporary = self.current_file.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(self.current_file)
        with self.events_file.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)


live_evidence = LiveEvidenceBus()
