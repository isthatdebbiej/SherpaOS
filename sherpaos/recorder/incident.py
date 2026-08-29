"""Pre/post-event incident capture.

`IncidentRecorder.observe(telemetry, decision, receipt=None)` is meant to be
called once per control step. It keeps a bounded ring buffer of recent
`(telemetry, decision, receipt)` samples as pre-event context, and whenever
the policy's action leaves `PASS` or changes (e.g. `LIMIT_SPEED` ->
`REQUEST_HOLD` escalation), it opens an "incident": the ring buffer content
at that instant becomes the pre-event window, and the next `post_event_window`
observations become the post-event window. Once both windows are complete the
whole thing is handed to `evidence.bundle.write_evidence_bundle` and written
under `<output_dir>/<incident_id>/`.

Incidents can overlap (a second trigger firing while the first incident's
post-event window is still being collected) -- every currently open incident
gets each new sample appended to its own post-event window and is finalized
independently, so a second incident starting before the first one closes
never drops or overwrites it.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sherpaos.contracts import ActuationReceipt, GuardAction, GuardDecision, RobotTelemetry
from sherpaos.evidence.bundle import write_evidence_bundle
from sherpaos.recorder.queue import StoreAndForwardQueue

# Pre/post window sizes are in *samples*, not seconds -- deliberately, so
# this module has no notion of the control loop's actual rate. At a rough
# 20-50 Hz control loop, 50 pre-event samples is on the order of 1-2.5s of
# context before the trigger, and 25 post-event samples is ~0.5-1.25s after
# it: enough to show the estimator/score trending up before an intervention
# and the immediate outcome after it, without an unbounded bundle for a
# long-running incident. Callers running at a very different rate should
# override both explicitly.
DEFAULT_PRE_EVENT_WINDOW = 50
DEFAULT_POST_EVENT_WINDOW = 25

Sample = tuple[RobotTelemetry, GuardDecision, ActuationReceipt | None]


@dataclass
class _ActiveIncident:
    incident_id: str
    pre_window: list[Sample]
    post_window: list[Sample] = field(default_factory=list)
    post_target: int = 0  # number of *additional* samples to collect after the trigger


class IncidentRecorder:
    """Detects incidents from a telemetry/decision stream and writes evidence bundles."""

    def __init__(
        self,
        output_dir: Path | str = Path("artifacts/incidents"),
        pre_event_window: int = DEFAULT_PRE_EVENT_WINDOW,
        post_event_window: int = DEFAULT_POST_EVENT_WINDOW,
        queue: StoreAndForwardQueue | None = None,
        on_incident: Callable[[Path], None] | None = None,
    ) -> None:
        if pre_event_window < 1:
            raise ValueError("pre_event_window must be >= 1")
        if post_event_window < 0:
            raise ValueError("post_event_window must be >= 0")

        self.output_dir = Path(output_dir)
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window
        # Optional: auto-enqueue each finalized bundle for store-and-forward,
        # and/or notify a caller-supplied callback. Both are opt-in so unit
        # tests can exercise IncidentRecorder in isolation.
        self.queue = queue
        self.on_incident = on_incident

        self._ring: deque[Sample] = deque(maxlen=pre_event_window)
        self._active: dict[str, _ActiveIncident] = {}
        self._last_action: GuardAction | None = None
        self._finalized_bundle_paths: list[Path] = []

    @property
    def finalized_bundle_paths(self) -> list[Path]:
        """Paths of every incident bundle finalized so far, in finalization order."""
        return list(self._finalized_bundle_paths)

    def observe(
        self,
        telemetry: RobotTelemetry,
        decision: GuardDecision,
        receipt: ActuationReceipt | None = None,
    ) -> str | None:
        """Feed one control-step sample in. Returns the new incident id if this
        sample triggered one, else `None`."""
        sample: Sample = (telemetry, decision, receipt)

        # Trigger condition: the action isn't PASS, and it's not simply a
        # continuation of the same non-PASS action we were already in
        # (that would otherwise re-trigger a fresh incident on every single
        # step of one ongoing intervention). An escalation, e.g.
        # LIMIT_SPEED -> REQUEST_HOLD, does count as a new trigger.
        is_trigger = decision.action != GuardAction.PASS and decision.action != self._last_action

        triggered_id: str | None = None
        if is_trigger:
            incident_id = decision.decision_id or uuid.uuid4().hex
            while incident_id in self._active:
                incident_id = f"{incident_id}-{uuid.uuid4().hex[:8]}"
            self._active[incident_id] = _ActiveIncident(
                incident_id=incident_id,
                pre_window=list(self._ring),
                post_target=self.post_event_window,
            )
            triggered_id = incident_id

        # Every in-flight incident (including one that just triggered on
        # this very sample) absorbs the current sample as post-event
        # context. The just-triggered incident's post_window therefore
        # starts with the triggering sample itself, followed by
        # `post_event_window` further samples before it's finalized.
        finalized_ids = []
        for incident_id, active in self._active.items():
            active.post_window.append(sample)
            if len(active.post_window) >= active.post_target + 1:
                finalized_ids.append(incident_id)

        for incident_id in finalized_ids:
            self._finalize(self._active.pop(incident_id))

        self._ring.append(sample)
        self._last_action = decision.action
        return triggered_id

    def flush_active_incidents(self) -> list[Path]:
        """Force-finalize any incidents still mid-post-window.

        Call this at the end of a run/episode so a trigger near the very
        end of a stream (which would never accumulate a full post-event
        window) is still written out rather than silently dropped.
        """
        finalized_paths = []
        for incident_id in list(self._active.keys()):
            finalized_paths.append(self._finalize(self._active.pop(incident_id)))
        return finalized_paths

    def _finalize(self, active: _ActiveIncident) -> Path:
        combined = active.pre_window + active.post_window
        telemetry = [s[0] for s in combined]
        decisions = [s[1] for s in combined]
        receipts = [s[2] for s in combined if s[2] is not None]

        bundle_dir = self.output_dir / active.incident_id
        write_evidence_bundle(bundle_dir, telemetry, decisions, receipts)

        self._finalized_bundle_paths.append(bundle_dir)
        if self.queue is not None:
            self.queue.enqueue(bundle_dir)
        if self.on_incident is not None:
            self.on_incident(bundle_dir)
        return bundle_dir
