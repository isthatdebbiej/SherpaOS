"""Narrow tool surface exposed to a realtime voice model."""

from __future__ import annotations

from typing import Any

from sherpaos.expedition.storage import ExpeditionStore


class ExpeditionVoiceTools:
    def __init__(self, store: ExpeditionStore | None = None) -> None:
        self.store = store or ExpeditionStore()

    def get_day_overview(self, expedition_id: str, day: int) -> dict[str, Any]:
        manifest = self.store.get_manifest(expedition_id, day)
        if manifest is None:
            return {
                "available": False, "expedition_id": expedition_id, "day": day,
                "limitation": "No verified ROS bag has been uploaded for this day.",
            }
        approved = [topic for topic in manifest.topics if topic.approved]
        return {
            "available": True, "expedition_id": expedition_id, "day": day,
            "bag_sha256": manifest.bag_sha256, "storage_format": manifest.storage_format,
            "duration_seconds": manifest.duration_seconds,
            "message_count": manifest.message_count,
            "approved_topics": [topic.to_dict() for topic in approved],
            "limitations": manifest.limitations,
            "evidence": {"kind": "bag_manifest", "day": day,
                         "bag_sha256": manifest.bag_sha256},
        }


REALTIME_TOOL_SCHEMAS = [
    {
        "type": "function", "name": "get_day_overview",
        "description": "Read the verified manifest and approved topic coverage for one day.",
        "parameters": {
            "type": "object",
            "properties": {
                "expedition_id": {"type": "string"},
                "day": {"type": "integer", "minimum": 1, "maximum": 99},
            },
            "required": ["expedition_id", "day"], "additionalProperties": False,
        },
    }
]
