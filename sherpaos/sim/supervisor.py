"""Adapter from the five-guard runtime to MuJoCo's guard callback."""

from __future__ import annotations

from sherpaos.contracts import (
    ActuationReceipt,
    GuardAction,
    GuardDecision,
    MissionContext,
    RobotTelemetry,
)
from sherpaos.policy.guards import FiveGuardSupervisor


class SimulationSupervisorAdapter:
    """Callable guard adapter that retains decisions and actuation receipts."""

    def __init__(
        self,
        mission_context: MissionContext | None,
        supervisor: FiveGuardSupervisor | None = None,
    ) -> None:
        self.mission_context = mission_context
        self.supervisor = supervisor if supervisor is not None else FiveGuardSupervisor()
        self.decisions: list[GuardDecision] = []
        self.receipts: list[ActuationReceipt] = []

    def __call__(self, history: list[RobotTelemetry]) -> tuple[float, bool]:
        # The simulator asks for a command before it has produced its first
        # observation. Allow exactly this bootstrap step; every later command
        # is based on the most recent telemetry sample.
        if not history:
            return 1.0, False

        sample = history[-1]
        now = float(sample.monotonic_time)
        decision = self.supervisor.decide(sample, self.mission_context, now)

        if decision.action == GuardAction.REQUEST_HOLD:
            speed_scale, hold = 0.0, True
        elif decision.action == GuardAction.LIMIT_SPEED:
            speed_scale = float(decision.requested_speed_limit or 0.0)
            hold = False
        else:
            speed_scale, hold = 1.0, False

        receipt = ActuationReceipt(
            decision_id=decision.decision_id,
            requested_action=decision.action,
            applied_action=decision.action,
            accepted=True,
            rejection_reason=None,
            adapter_timestamp=now,
            acknowledgement_source="mujoco-supervisor-adapter",
        )
        self.decisions.append(decision)
        self.receipts.append(receipt)
        return speed_scale, hold
