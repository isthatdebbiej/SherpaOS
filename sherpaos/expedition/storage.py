"""Immutable local storage for expedition days."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from sherpaos.expedition.inspect import inspect_bag
from sherpaos.expedition.models import DayManifest, DayStatus, TopicSummary

SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SUPPORTED_SUFFIXES = {".mcap", ".db3"}


class ExpeditionStore:
    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("SHERPA_EXPEDITION_STORE")
        default_root = Path(configured) if configured else Path("var/expeditions")
        self.root = (root or default_root).resolve()

    def ingest(self, expedition_id: str, day: int, filename: str, source: BinaryIO) -> DayManifest:
        self._validate_identity(expedition_id, day)
        safe_filename = Path(filename).name
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("only .mcap and rosbag2 .db3 uploads are supported")
        upload_id = uuid.uuid4().hex
        incoming_root = self.root.parent / "incoming"
        incoming_root.mkdir(parents=True, exist_ok=True)
        upload_dir = Path(tempfile.mkdtemp(prefix=f"{upload_id}-", dir=incoming_root))
        temporary = upload_dir / f"recording{suffix}"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as destination:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            inspection = inspect_bag(temporary)
            manifest = DayManifest(
                expedition_id=expedition_id, day=day, upload_id=upload_id,
                original_filename=safe_filename, stored_filename=f"day-{day:02d}{suffix}",
                bag_sha256=digest.hexdigest(), size_bytes=size,
                storage_format=str(inspection["storage_format"]),
                status=DayStatus.READY_FOR_QUESTIONS,
                created_at=datetime.now(UTC).isoformat(),
                start_time_ns=inspection["start_time_ns"],  # type: ignore[arg-type]
                end_time_ns=inspection["end_time_ns"],  # type: ignore[arg-type]
                message_count=int(inspection["message_count"]),
                topics=list(inspection["topics"]),  # type: ignore[arg-type]
            )
            self._promote(temporary, manifest)
            return manifest
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    def get_manifest(self, expedition_id: str, day: int) -> DayManifest | None:
        self._validate_identity(expedition_id, day)
        path = self._day_dir(expedition_id, day) / "manifest.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("duration_seconds", None)
        raw["status"] = DayStatus(raw["status"])
        raw["topics"] = [TopicSummary(**topic) for topic in raw["topics"]]
        return DayManifest(**raw)

    def list_days(self, expedition_id: str) -> list[DayManifest]:
        if not SAFE_ID.fullmatch(expedition_id):
            raise ValueError("invalid expedition id")
        return [
            manifest
            for day in range(1, 6)
            if (manifest := self.get_manifest(expedition_id, day))
        ]

    def _promote(self, temporary: Path, manifest: DayManifest) -> None:
        day_dir = self._day_dir(manifest.expedition_id, manifest.day)
        if day_dir.exists():
            raise FileExistsError(
                f"day {manifest.day} already has an authoritative bag; replacement is not implicit"
            )
        raw_dir = day_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=False)
        try:
            shutil.move(str(temporary), raw_dir / manifest.stored_filename)
            (day_dir / "manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(day_dir, ignore_errors=True)
            raise

    def _day_dir(self, expedition_id: str, day: int) -> Path:
        return self.root / expedition_id / f"day-{day:02d}"

    @staticmethod
    def _validate_identity(expedition_id: str, day: int) -> None:
        if not SAFE_ID.fullmatch(expedition_id):
            raise ValueError("expedition id must be lowercase letters, numbers, and hyphens")
        if not 1 <= day <= 99:
            raise ValueError("day must be between 1 and 99")
