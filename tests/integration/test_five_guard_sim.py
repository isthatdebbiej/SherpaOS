from __future__ import annotations

import pytest

from sherpaos.contracts import GuardName
from sherpaos.geography.terrain import build_mission_context, load_route
from sherpaos.sim.runner import run_episode
from sherpaos.sim.scenario import nominal_scenario
from sherpaos.sim.supervisor import SimulationSupervisorAdapter

pytestmark = pytest.mark.integration


def test_five_guard_supervisor_drives_sim_and_receipts_every_decision():
    route = load_route()
    mission_context = build_mission_context(route, now=0.0, waypoint_name="Lukla")
    adapter = SimulationSupervisorAdapter(mission_context)
    result = run_episode(
        nominal_scenario(seed=21),
        seed=21,
        guard_fn=adapter,
        max_steps=30,
    )

    assert result.steps_survived == 30
    assert len(adapter.decisions) == 29  # one bootstrap step has no telemetry yet
    assert len(adapter.receipts) == len(adapter.decisions)
    assert all(
        {report.guard for report in decision.guard_reports} == set(GuardName)
        for decision in adapter.decisions
    )
    assert all(
        receipt.decision_id == decision.decision_id
        for decision, receipt in zip(adapter.decisions, adapter.receipts, strict=True)
    )
