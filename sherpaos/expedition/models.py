"""Serializable contracts for uploaded expedition memories."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class DayStatus(enum.StrEnum):
    EMPTY = "EMPTY"
    RECEIVING = "RECEIVING"
    VERIFYING = "VERIFYING"
    READING_TOPICS = "READING_TOPICS"
    BUILDING_TIMELINE = "BUILDING_TIMELINE"
    READY_FOR_QUESTIONS = "READY_FOR_QUESTIONS"
    FAILED = "FAILED"


@dataclass(slots=True, frozen=True)
class TopicSummary:
    name: str
    message_type: str
    message_count: int
    start_time_ns: int | None
    end_time_ns: int | None
    approved: bool
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DayManifest:
    expedition_id: str
    day: int
    upload_id: str
    original_filename: str
    stored_filename: str
    bag_sha256: str
    size_bytes: int
    storage_format: str
    status: DayStatus
    created_at: str
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    message_count: int = 0
    topics: list[TopicSummary] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.start_time_ns is None or self.end_time_ns is None:
            return None
        return max(0.0, (self.end_time_ns - self.start_time_ns) / 1_000_000_000)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["duration_seconds"] = self.duration_seconds
        return value
