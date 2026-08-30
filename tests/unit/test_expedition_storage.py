from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from sherpaos.expedition.api import _grounded_prompt
from sherpaos.expedition.models import DayStatus
from sherpaos.expedition.storage import ExpeditionStore
from sherpaos.voice.tools import ExpeditionVoiceTools


def _rosbag2_db(tmp_path: Path) -> bytes:
    buffer_path = tmp_path / "memory.db3"
    connection = sqlite3.connect(buffer_path)
    connection.executescript(
        """
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                            serialization_format TEXT, offered_qos_profiles TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER,
                              timestamp INTEGER, data BLOB);
        INSERT INTO topics VALUES(1, '/battery_state',
            'sensor_msgs/msg/BatteryState', 'cdr', '');
        INSERT INTO topics VALUES(2, '/sim/true_friction',
            'std_msgs/msg/Float64', 'cdr', '');
        INSERT INTO messages VALUES(1, 1, 1000000000, X'01');
        INSERT INTO messages VALUES(2, 1, 3000000000, X'02');
        INSERT INTO messages VALUES(3, 2, 2000000000, X'03');
        """
    )
    connection.close()
    return buffer_path.read_bytes()


def test_real_db3_is_hashed_inspected_and_promoted(tmp_path: Path) -> None:
    store = ExpeditionStore(tmp_path / "expeditions")
    manifest = store.ingest("everest-001", 3, "robot.db3", io.BytesIO(_rosbag2_db(tmp_path)))
    assert manifest.status is DayStatus.READY_FOR_QUESTIONS
    assert manifest.message_count == 3
    assert manifest.duration_seconds == 2.0
    assert len(manifest.bag_sha256) == 64
    assert [topic.name for topic in manifest.topics if topic.approved] == ["/battery_state"]
    assert not next(topic for topic in manifest.topics if "true_friction" in topic.name).approved
    assert (tmp_path / "expeditions/everest-001/day-03/raw/day-03.db3").exists()


def test_existing_day_cannot_be_silently_replaced(tmp_path: Path) -> None:
    store = ExpeditionStore(tmp_path / "expeditions")
    payload = _rosbag2_db(tmp_path)
    store.ingest("everest-001", 1, "first.db3", io.BytesIO(payload))
    with pytest.raises(FileExistsError):
        store.ingest("everest-001", 1, "replacement.db3", io.BytesIO(payload))


def test_voice_overview_is_grounded_in_manifest(tmp_path: Path) -> None:
    store = ExpeditionStore(tmp_path / "expeditions")
    manifest = store.ingest("everest-001", 2, "robot.db3", io.BytesIO(_rosbag2_db(tmp_path)))
    result = ExpeditionVoiceTools(store).get_day_overview("everest-001", 2)
    assert result["available"] is True
    assert result["bag_sha256"] == manifest.bag_sha256
    assert result["approved_topics"][0]["name"] == "/battery_state"


def test_path_like_expedition_id_is_rejected(tmp_path: Path) -> None:
    store = ExpeditionStore(tmp_path / "expeditions")
    with pytest.raises(ValueError):
        store.ingest("../escape", 1, "robot.db3", io.BytesIO(_rosbag2_db(tmp_path)))


def test_radio_prompt_requires_verified_evidence() -> None:
    prompt = _grounded_prompt(
        "What data do you have?",
        {"day": 3, "bag_sha256": "abc123", "approved_topics": ["/battery_state"]},
    )
    assert "Never invent telemetry" in prompt
    assert "abc123" in prompt
    assert "What data do you have?" in prompt
