"""Allowlist and leakage controls for ROS topics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True, frozen=True)
class TopicPolicy:
    approved: dict[str, dict[str, object]]
    forbidden_patterns: tuple[str, ...]

    @classmethod
    def load(cls, path: Path | None = None) -> TopicPolicy:
        config_path = path or Path(__file__).parents[2] / "configs" / "ros_topics.yaml"
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(
            approved=dict(raw.get("approved_topics", {})),
            forbidden_patterns=tuple(raw.get("forbidden_name_patterns", [])),
        )

    def classify(self, topic: str) -> tuple[bool, str | None]:
        lowered = topic.lower()
        if any(pattern.lower() in lowered for pattern in self.forbidden_patterns):
            return False, None
        config = self.approved.get(topic)
        return (config is not None, str(config["role"]) if config else None)
