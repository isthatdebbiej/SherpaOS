"""Conservative adapter for MuJoCo Playground policy observations.

Playground observations are task-specific: some environments expose a mapping,
while others expose one flat policy vector.  This adapter therefore requires an
explicit layout instead of guessing indices.  Only fields that have a plausible
on-robot equivalent are copied into :class:`RobotTelemetry`.  Extra observation
entries (including simulator ground truth) are deliberately ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from sherpaos.contracts import RobotTelemetry, TelemetrySource

ObservationSelector: TypeAlias = str | slice
Observation: TypeAlias = Mapping[str, object] | np.ndarray


@dataclass(frozen=True, slots=True)
class PlaygroundObservationLayout:
    """Selectors for observable fields in a Playground policy observation.

    String selectors address mapping observations.  Slice selectors address a
    flat numeric vector.  A layout must not select privileged simulator values
    such as friction, contact labels, injected faults, or fall state.
    """

    joint_position: ObservationSelector
    joint_velocity: ObservationSelector
    base_orientation: ObservationSelector
    base_angular_velocity: ObservationSelector
    base_linear_acceleration: ObservationSelector
    joint_effort: ObservationSelector | None = None
    commanded_velocity: ObservationSelector | None = None


class PlaygroundTelemetryAdapter:
    """Map an already-produced Playground observation to ``RobotTelemetry``."""

    def __init__(self, layout: PlaygroundObservationLayout, *, joint_count: int = 29) -> None:
        if joint_count <= 0:
            raise ValueError("joint_count must be positive")
        self._layout = layout
        self._joint_count = joint_count
        _validate_layout(layout)

    def sample(
        self,
        observation: Observation,
        *,
        sequence: int,
        monotonic_time: float,
        source_time: float | None = None,
        gait_mode: str | None = None,
        valid: bool = True,
    ) -> RobotTelemetry:
        """Return a sample, marking malformed observations invalid.

        Missing, wrongly-shaped, or non-finite required fields are represented
        by NaN arrays and ``valid=False``.  This lets the telemetry-health guard
        choose a conservative action without an adapter exception taking down
        the runtime loop.
        """

        provenance: dict[str, str] = {}
        sample_valid = bool(valid)

        def required(name: str, selector: ObservationSelector, size: int) -> np.ndarray:
            nonlocal sample_valid
            value, ok = _read_vector(observation, selector, size)
            provenance[name] = _provenance(selector) if ok else "missing_or_invalid"
            sample_valid = sample_valid and ok
            return value

        def optional(
            name: str, selector: ObservationSelector | None, size: int | None = None
        ) -> np.ndarray | None:
            nonlocal sample_valid
            if selector is None:
                provenance[name] = "unavailable"
                return None
            value, ok = _read_vector(observation, selector, size)
            provenance[name] = _provenance(selector) if ok else "missing_or_invalid"
            sample_valid = sample_valid and ok
            return value

        joint_position = required(
            "joint_position", self._layout.joint_position, self._joint_count
        )
        joint_velocity = required(
            "joint_velocity", self._layout.joint_velocity, self._joint_count
        )
        base_orientation = required("base_orientation", self._layout.base_orientation, 4)
        base_angular_velocity = required(
            "base_angular_velocity", self._layout.base_angular_velocity, 3
        )
        base_linear_acceleration = required(
            "base_linear_acceleration", self._layout.base_linear_acceleration, 3
        )
        joint_effort = optional("joint_effort", self._layout.joint_effort, self._joint_count)
        commanded_velocity = optional(
            "commanded_velocity", self._layout.commanded_velocity
        )

        return RobotTelemetry(
            monotonic_time=float(monotonic_time),
            source_time=float(monotonic_time if source_time is None else source_time),
            sequence=int(sequence),
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            base_orientation=base_orientation,
            base_angular_velocity=base_angular_velocity,
            base_linear_acceleration=base_linear_acceleration,
            commanded_velocity=commanded_velocity,
            gait_mode=gait_mode,
            source=TelemetrySource.SIM,
            valid=sample_valid,
            field_provenance=provenance,
        )


def _read_vector(
    observation: Observation,
    selector: ObservationSelector,
    expected_size: int | None,
) -> tuple[np.ndarray, bool]:
    try:
        if isinstance(selector, str):
            if not isinstance(observation, Mapping) or selector not in observation:
                raise KeyError(selector)
            raw = observation[selector]
        else:
            if isinstance(observation, Mapping):
                raise TypeError("slice selector cannot address a mapping observation")
            raw = np.asarray(observation)[selector]
        value = np.asarray(raw, dtype=float).reshape(-1).copy()
        if expected_size is not None and value.size != expected_size:
            raise ValueError("unexpected observation field size")
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("observation field is empty or non-finite")
        return value, True
    except (KeyError, TypeError, ValueError, IndexError):
        size = 1 if expected_size is None else expected_size
        return np.full(size, np.nan, dtype=float), False


def _validate_layout(layout: PlaygroundObservationLayout) -> None:
    forbidden = ("friction", "contact", "fault", "fall", "ground_truth", "privileged")
    for name, selector in (
        ("joint_position", layout.joint_position),
        ("joint_velocity", layout.joint_velocity),
        ("base_orientation", layout.base_orientation),
        ("base_angular_velocity", layout.base_angular_velocity),
        ("base_linear_acceleration", layout.base_linear_acceleration),
        ("joint_effort", layout.joint_effort),
        ("commanded_velocity", layout.commanded_velocity),
    ):
        if selector is None:
            continue
        if isinstance(selector, str) and any(token in selector.lower() for token in forbidden):
            raise ValueError(f"{name} selector appears to reference simulator-only truth")


def _provenance(selector: ObservationSelector) -> str:
    if isinstance(selector, str):
        return f"playground_observation:{selector}"
    return f"playground_observation:slice({selector.start},{selector.stop},{selector.step})"
