import json

import pytest
from fastapi import HTTPException

from sherpaos.runtime.api import _memory
from sherpaos.runtime.live import LiveEvidenceBus


def test_live_bus_persists_across_process_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERPA_LIVE_STATE_DIR", str(tmp_path))
    writer = LiveEvidenceBus()
    event = {"decision": {"decision_id": "decision-001"}, "receipt": {"accepted": True}}
    writer._persist(event)

    reader = LiveEvidenceBus()
    assert reader.current() == event
    assert reader.history(10) == [event]


def test_journal_memory_is_exactly_days_one_to_four(tmp_path, monkeypatch):
    for day in range(1, 5):
        (tmp_path / f"day-{day:02d}.json").write_text(
            json.dumps({"day": day, "real_world_telemetry": {"episodes": 0}}),
            encoding="utf-8",
        )
    monkeypatch.setenv("SHERPA_MEMORY_DIR", str(tmp_path))

    assert [item["day"] for item in (_memory(day) for day in range(1, 5))] == [1, 2, 3, 4]
    with pytest.raises(HTTPException) as exc:
        _memory(5)
    assert exc.value.status_code == 404
