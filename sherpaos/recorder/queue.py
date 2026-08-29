"""Durable local store-and-forward queue for evidence bundles.

Purpose: if the link to durable/off-box storage is down when an incident
happens, the reference to its evidence bundle must queue locally (surviving
a process crash/restart) and flush once connectivity returns. This module
only implements the durable local queue + retry/flush mechanics -- the
actual "upload" is a caller-supplied `transport` callable (in tests/demo,
a local copy-to-another-directory stand-in), never a real network client.

Layout under `root`:

    root/_pending/<uuid>.json   -- one pointer file per un-sent bundle
    root/_sent/<uuid>.json      -- moved here once `transport` succeeds

Each pointer file is `{"bundle_path": "<absolute path to an evidence bundle
directory>"}`. Pointer files (not the bundles themselves) are what move
between `_pending` and `_sent`; the bundle directories written by
`evidence.bundle.write_evidence_bundle` are untouched by this module.

This class deliberately keeps no in-memory index of what's pending -- every
call to `pending()`/`flush()` re-reads `_pending/` from disk. That is what
makes "queued while offline, flushed after reconnect" survive a process
restart: a freshly constructed `StoreAndForwardQueue` pointed at the same
`root` sees exactly the same pending set a previous instance would have.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True, frozen=True)
class FlushResult:
    """Outcome of one `StoreAndForwardQueue.flush()` call."""

    attempted: int
    sent: int
    failed: int
    sent_paths: list[Path] = field(default_factory=list)
    failed_paths: list[Path] = field(default_factory=list)


class StoreAndForwardQueue:
    """Filesystem-backed queue of references to already-written evidence bundles."""

    PENDING_DIRNAME = "_pending"
    SENT_DIRNAME = "_sent"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.pending_dir = self.root / self.PENDING_DIRNAME
        self.sent_dir = self.root / self.SENT_DIRNAME
        # Re-instantiating against an existing root is exactly how this
        # queue demonstrates surviving a process restart: these dirs (and
        # whatever pointer files already sit in _pending/) already exist
        # from the prior process, and nothing below re-derives state from
        # anywhere other than the filesystem.
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.sent_dir.mkdir(parents=True, exist_ok=True)

    def _pointer_files(self) -> list[Path]:
        return sorted(self.pending_dir.glob("*.json"))

    def enqueue(self, bundle_path: Path) -> None:
        """Record a durable reference to an already-written evidence bundle.

        Written via write-to-temp-then-rename so a crash mid-write can
        never leave a half-written pointer file behind in `_pending/`.
        """
        bundle_path = Path(bundle_path)
        resolved = bundle_path.resolve()
        pointer_name = f"{uuid.uuid4().hex}.json"
        pointer_path = self.pending_dir / pointer_name
        tmp_path = self.pending_dir / f"{pointer_name}.tmp"
        tmp_path.write_text(json.dumps({"bundle_path": str(resolved)}))
        tmp_path.replace(pointer_path)

    def pending(self) -> list[Path]:
        """Return the evidence bundle paths currently queued (unsent)."""
        result: list[Path] = []
        for pointer_path in self._pointer_files():
            try:
                payload = json.loads(pointer_path.read_text())
                result.append(Path(payload["bundle_path"]))
            except (OSError, json.JSONDecodeError, KeyError):
                # A malformed pointer file shouldn't crash enumeration;
                # it will simply also fail in flush() and stay pending.
                continue
        return result

    def flush(self, transport: Callable[[Path], bool]) -> FlushResult:
        """Attempt to send every pending entry via `transport`.

        `transport(bundle_path)` should return `True` on success, `False`
        (or raise) on failure. Entries that succeed move to `_sent/`;
        entries that fail (or raise, or have unreadable pointer files)
        stay in `_pending/` for a future flush() call.
        """
        sent_paths: list[Path] = []
        failed_paths: list[Path] = []

        for pointer_path in self._pointer_files():
            try:
                payload = json.loads(pointer_path.read_text())
                bundle_path = Path(payload["bundle_path"])
            except (OSError, json.JSONDecodeError, KeyError):
                failed_paths.append(pointer_path)
                continue

            try:
                success = bool(transport(bundle_path))
            except Exception:
                success = False

            if success:
                destination = self.sent_dir / pointer_path.name
                pointer_path.replace(destination)
                sent_paths.append(destination)
            else:
                failed_paths.append(pointer_path)

        return FlushResult(
            attempted=len(sent_paths) + len(failed_paths),
            sent=len(sent_paths),
            failed=len(failed_paths),
            sent_paths=sent_paths,
            failed_paths=failed_paths,
        )
