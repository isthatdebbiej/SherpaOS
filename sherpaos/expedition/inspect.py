"""Read-only inspection of genuine rosbag2 SQLite and MCAP recordings."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from sherpaos.expedition.models import TopicSummary
from sherpaos.expedition.topics import TopicPolicy

MCAP_MAGIC = b"\x89MCAP0\r\n"
SQLITE_MAGIC = b"SQLite format 3\x00"


class UnsupportedBagError(ValueError):
    """Raised when an upload is not a supported, readable bag."""


def detect_storage_format(path: Path) -> str:
    with path.open("rb") as stream:
        prefix = stream.read(16)
    if prefix.startswith(MCAP_MAGIC):
        return "mcap"
    if prefix.startswith(SQLITE_MAGIC):
        return "sqlite3"
    raise UnsupportedBagError("upload is neither an MCAP nor rosbag2 SQLite recording")


def inspect_bag(path: Path, policy: TopicPolicy | None = None) -> dict[str, object]:
    active_policy = policy or TopicPolicy.load()
    storage_format = detect_storage_format(path)
    if storage_format == "sqlite3":
        return _inspect_sqlite(path, active_policy)
    return _inspect_mcap(path, active_policy)


def _inspect_sqlite(path: Path, policy: TopicPolicy) -> dict[str, object]:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"topics", "messages"}.issubset(tables):
            raise UnsupportedBagError("SQLite file does not contain rosbag2 topics/messages tables")
        rows = connection.execute(
            """
            SELECT t.name, t.type, COUNT(m.id) AS message_count,
                   MIN(m.timestamp) AS start_time_ns, MAX(m.timestamp) AS end_time_ns
            FROM topics t LEFT JOIN messages m ON m.topic_id = t.id
            GROUP BY t.id, t.name, t.type ORDER BY t.name
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise UnsupportedBagError(f"unreadable rosbag2 SQLite recording: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    topics = []
    for row in rows:
        approved, role = policy.classify(row["name"])
        topics.append(
            TopicSummary(
                name=row["name"], message_type=row["type"],
                message_count=row["message_count"], start_time_ns=row["start_time_ns"],
                end_time_ns=row["end_time_ns"], approved=approved, role=role,
            )
        )
    return _summarize("sqlite3", topics)


def _inspect_mcap(path: Path, policy: TopicPolicy) -> dict[str, object]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise UnsupportedBagError(
            "MCAP support requires: uv sync --extra expedition"
        ) from exc
    counts: dict[int, int] = defaultdict(int)
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    channels: dict[int, tuple[str, str]] = {}
    try:
        with path.open("rb") as stream:
            for schema, channel, message in make_reader(stream).iter_messages():
                counts[channel.id] += 1
                starts[channel.id] = min(starts.get(channel.id, message.log_time), message.log_time)
                ends[channel.id] = max(ends.get(channel.id, message.log_time), message.log_time)
                channels[channel.id] = (
                    channel.topic,
                    schema.name if schema is not None else channel.message_encoding,
                )
    except Exception as exc:
        raise UnsupportedBagError(f"unreadable MCAP recording: {exc}") from exc
    topics = []
    for channel_id, (name, message_type) in sorted(channels.items(), key=lambda item: item[1][0]):
        approved, role = policy.classify(name)
        topics.append(
            TopicSummary(
                name=name, message_type=message_type, message_count=counts[channel_id],
                start_time_ns=starts.get(channel_id), end_time_ns=ends.get(channel_id),
                approved=approved, role=role,
            )
        )
    return _summarize("mcap", topics)


def _summarize(storage_format: str, topics: list[TopicSummary]) -> dict[str, object]:
    starts = [topic.start_time_ns for topic in topics if topic.start_time_ns is not None]
    ends = [topic.end_time_ns for topic in topics if topic.end_time_ns is not None]
    return {
        "storage_format": storage_format,
        "topics": topics,
        "message_count": sum(topic.message_count for topic in topics),
        "start_time_ns": min(starts) if starts else None,
        "end_time_ns": max(ends) if ends else None,
    }
