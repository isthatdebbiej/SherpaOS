"""Hypothesis property tests for sherpaos.contracts.

Telemetry is built via a small local helper rather than the `make_telemetry`
fixture from tests/conftest.py: mixing a function-scoped pytest fixture with
`@given` re-uses the same fixture instance across every generated example,
which Hypothesis flags (HealthCheck.function_scoped_fixture). Since the
construction logic here is trivial and stateless, duplicating a minimal
version locally avoids that entirely.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sherpaos.contracts import GuardAction, ReasonCode, RobotTelemetry, TelemetrySource

pytestmark = pytest.mark.property

# Bounded away from +/-inf (excluded anyway) to keep subtraction well-behaved.
_FLOATS = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12)


def _minimal_telemetry(monotonic_time: float) -> RobotTelemetry:
    """Smallest valid RobotTelemetry needed to exercise age_seconds."""
    return RobotTelemetry(
        monotonic_time=monotonic_time,
        source_time=monotonic_time,
        sequence=0,
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        joint_effort=np.zeros(29),
        base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_angular_velocity=np.zeros(3),
        base_linear_acceleration=np.zeros(3),
    )


@given(monotonic_time=_FLOATS, now=_FLOATS)
def test_age_seconds_is_never_negative(monotonic_time: float, now: float):
    telemetry = _minimal_telemetry(monotonic_time)
    assert telemetry.age_seconds(now) >= 0.0


@given(monotonic_time=_FLOATS, now=_FLOATS)
def test_age_seconds_matches_elapsed_time_or_clamps_to_zero(monotonic_time: float, now: float):
    telemetry = _minimal_telemetry(monotonic_time)
    age = telemetry.age_seconds(now)
    if now >= monotonic_time:
        # Same subtraction as age_seconds itself -- must match exactly.
        assert age == now - monotonic_time
    else:
        assert age == 0.0


@given(action=st.sampled_from(list(GuardAction)))
def test_guard_action_value_roundtrips(action: GuardAction):
    assert isinstance(action, str)
    assert action == action.value
    assert GuardAction(action.value) is action


@given(source=st.sampled_from(list(TelemetrySource)))
def test_telemetry_source_value_roundtrips(source: TelemetrySource):
    assert isinstance(source, str)
    assert source == source.value
    assert TelemetrySource(source.value) is source


@given(reason=st.sampled_from(list(ReasonCode)))
def test_reason_code_value_roundtrips(reason: ReasonCode):
    assert isinstance(reason, str)
    assert reason == reason.value
    assert ReasonCode(reason.value) is reason
