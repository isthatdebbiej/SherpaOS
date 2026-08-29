"""Unit tests for sherpaos.recorder.queue.StoreAndForwardQueue."""

from __future__ import annotations

from pathlib import Path

from sherpaos.recorder.queue import StoreAndForwardQueue


def _make_bundle_dir(tmp_path: Path, name: str) -> Path:
    bundle_dir = tmp_path / "bundles" / name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "telemetry.jsonl").write_text('{"x": 1}\n')
    return bundle_dir


def test_enqueue_while_offline_leaves_entry_pending(tmp_path: Path) -> None:
    queue = StoreAndForwardQueue(tmp_path / "queue")
    bundle = _make_bundle_dir(tmp_path, "incident-1")

    queue.enqueue(bundle)

    pending = queue.pending()
    assert pending == [bundle.resolve()]

    # "offline": transport always fails
    result = queue.flush(transport=lambda _path: False)
    assert result.attempted == 1
    assert result.sent == 0
    assert result.failed == 1
    assert queue.pending() == [bundle.resolve()]


def test_flush_success_marks_sent_and_clears_pending(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    queue = StoreAndForwardQueue(queue_dir)
    bundle = _make_bundle_dir(tmp_path, "incident-2")
    queue.enqueue(bundle)

    result = queue.flush(transport=lambda _path: True)

    assert result.sent == 1
    assert result.failed == 0
    assert queue.pending() == []
    sent_pointer_files = list((queue_dir / StoreAndForwardQueue.SENT_DIRNAME).glob("*.json"))
    assert len(sent_pointer_files) == 1
    pending_pointer_files = list((queue_dir / StoreAndForwardQueue.PENDING_DIRNAME).glob("*.json"))
    assert pending_pointer_files == []


def test_flush_with_no_pending_entries_is_a_noop(tmp_path: Path) -> None:
    queue = StoreAndForwardQueue(tmp_path / "queue")
    result = queue.flush(transport=lambda _path: True)
    assert result.attempted == 0
    assert result.sent == 0
    assert result.failed == 0


def test_queue_persists_pending_entries_across_reinstantiation(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    bundle_a = _make_bundle_dir(tmp_path, "incident-a")
    bundle_b = _make_bundle_dir(tmp_path, "incident-b")

    queue1 = StoreAndForwardQueue(queue_dir)
    queue1.enqueue(bundle_a)
    queue1.enqueue(bundle_b)
    del queue1  # simulate the process exiting while still "offline"

    queue2 = StoreAndForwardQueue(queue_dir)
    assert set(queue2.pending()) == {bundle_a.resolve(), bundle_b.resolve()}

    result = queue2.flush(transport=lambda _path: True)
    assert result.sent == 2
    assert queue2.pending() == []
    del queue2

    # a third instantiation against the same root sees nothing left pending
    queue3 = StoreAndForwardQueue(queue_dir)
    assert queue3.pending() == []


def test_partial_flush_failure_only_retains_failed_entries(tmp_path: Path) -> None:
    queue = StoreAndForwardQueue(tmp_path / "queue")
    bundle_ok = _make_bundle_dir(tmp_path, "incident-ok")
    bundle_bad = _make_bundle_dir(tmp_path, "incident-bad")
    queue.enqueue(bundle_ok)
    queue.enqueue(bundle_bad)

    def transport(path: Path) -> bool:
        return path == bundle_ok.resolve()

    result = queue.flush(transport)
    assert result.sent == 1
    assert result.failed == 1
    assert queue.pending() == [bundle_bad.resolve()]

    # a subsequent flush with a fully-working transport clears the rest
    result2 = queue.flush(transport=lambda _path: True)
    assert result2.sent == 1
    assert queue.pending() == []
