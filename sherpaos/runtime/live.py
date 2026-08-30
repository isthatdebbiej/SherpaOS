"""Live decision evidence bus for the Vultr API and simulator."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
from typing import Any

from sherpaos.contracts import ActuationReceipt, GuardDecision


class LiveEvidenceBus:
    def __init__(self) -> None:
        self.events: deque[dict[str, Any]] = deque(maxlen=1000)
        self.subscribers: set[asyncio.Queue] = set()

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
        for queue in tuple(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return event

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)


live_evidence = LiveEvidenceBus()
