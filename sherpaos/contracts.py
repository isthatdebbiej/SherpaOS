"""Frozen runtime contracts shared by every SherpaOS module.

These shapes are the interface boundary described in docs/CONTRACTS.md.
The simulator adapter, dump-replay adapter, and live-G1 adapter all
produce `RobotTelemetry`; nothing downstream may depend on how it was
produced. Do not change field names/types without recording the change
in docs/DECISIONS.md — every other lane imports this module.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class TelemetrySource(enum.StrEnum):
    SIM = "sim"
    DUMP = "dump"
    G1_LIVE = "g1_live"


class GuardAction(enum.StrEnum):
    PASS = "PASS"
    LIMIT_SPEED = "LIMIT_SPEED"
    REQUEST_HOLD = "REQUEST_HOLD"


class ReasonCode(enum.StrEnum):
    """Shared vocabulary of estimator/policy trigger reasons.

    Add new codes here rather than inventing ad-hoc strings elsewhere,
    so evidence bundles and the demo UI can render a stable legend.
    """

    NOMINAL = "NOMINAL"
    SLIP_RISK_HIGH = "SLIP_RISK_HIGH"
    SLIP_RISK_ELEVATED = "SLIP_RISK_ELEVATED"
    BODY_ANOMALY = "BODY_ANOMALY"
    ASYMMETRY_DETECTED = "ASYMMETRY_DETECTED"
    ORIENTATION_INSTABILITY = "ORIENTATION_INSTABILITY"
    STALE_TELEMETRY = "STALE_TELEMETRY"
    MISSING_FIELD = "MISSING_FIELD"
    NAN_OR_INVALID = "NAN_OR_INVALID"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HYSTERESIS_HOLD = "HYSTERESIS_HOLD"
    RECOVERY_CONFIRMED = "RECOVERY_CONFIRMED"

    # Battery-margin guard
    BATTERY_MARGIN_LOW = "BATTERY_MARGIN_LOW"
    BATTERY_VOLTAGE_SAG = "BATTERY_VOLTAGE_SAG"
    BATTERY_COLD_DERATED = "BATTERY_COLD_DERATED"
    BATTERY_DATA_UNAVAILABLE = "BATTERY_DATA_UNAVAILABLE"

    # Geographic-risk guard
    GEOGRAPHIC_STEEP_SLOPE = "GEOGRAPHIC_STEEP_SLOPE"
    GEOGRAPHIC_HIGH_EXPOSURE = "GEOGRAPHIC_HIGH_EXPOSURE"
    GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT = "GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT"
    GEOGRAPHIC_CONTEXT_UNAVAILABLE = "GEOGRAPHIC_CONTEXT_UNAVAILABLE"
    GEOGRAPHIC_CONTEXT_STALE = "GEOGRAPHIC_CONTEXT_STALE"


class GuardName(enum.StrEnum):
    """The five guard families. See docs/CONTRACTS.md 'Guard families'."""

    MOBILITY = "mobility"
    DYNAMICS = "dynamics"
    TELEMETRY_HEALTH = "telemetry_health"
    BATTERY = "battery"
    GEOGRAPHIC = "geographic"


@dataclass(slots=True, frozen=True)
class RobotTelemetry:
    """One timestamped observation of onboard-observable robot state.

    `valid` and `field_provenance` let a conservative producer (e.g. a
    dump replay hitting a gap, or a live adapter losing a sensor) mark
    a sample or field as untrustworthy without inventing data. Do not
    add simulator-only ground truth fields here (true friction, true
    contact class, injected-fault label, true fall state) — those
    belong to evaluator-only structures in sherpaos/evaluation.
    """

    monotonic_time: float
    source_time: float
    sequence: int

    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_effort: np.ndarray | None

    base_orientation: np.ndarray  # quaternion, wxyz
    base_angular_velocity: np.ndarray
    base_linear_acceleration: np.ndarray

    commanded_velocity: np.ndarray | None = None
    gait_mode: str | None = None

    battery_fraction: float | None = None
    battery_voltage: float | None = None
    battery_current_a: float | None = None
    battery_temperature_c: float | None = None

    source: TelemetrySource = TelemetrySource.SIM
    valid: bool = True
    field_provenance: dict[str, str] = field(default_factory=dict)

    def age_seconds(self, now_monotonic: float) -> float:
        return max(0.0, now_monotonic - self.monotonic_time)


@dataclass(slots=True, frozen=True)
class GuardReport:
    """Per-guard output before fusion, from one of the five guard families.

    Each guard (mobility, dynamics, telemetry-health, battery, geographic)
    emits its own score/confidence/reasons/recommended action independently.
    The policy fuses these conservatively into one GuardDecision — a
    high-severity guard must not be hidden by averaging it with unrelated
    low-severity guards. See docs/CONTRACTS.md 'Guard families'.
    """

    guard: GuardName
    score: float  # 0..1
    confidence: float  # 0..1
    reason_codes: tuple[ReasonCode, ...]
    recommended_action: GuardAction
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GuardDecision:
    """Output of the SherpaOS runtime for one telemetry sample."""

    decision_id: str
    action: GuardAction
    score: float  # 0..1 aggregate risk score
    confidence: float  # 0..1
    reason_codes: tuple[ReasonCode, ...]

    input_age_seconds: float
    requested_speed_limit: float | None  # fraction of nominal, 0..1

    timestamp: float
    rules_version: str
    model_version: str | None = None
    guard_reports: tuple[GuardReport, ...] = ()


@dataclass(slots=True, frozen=True)
class ActuationReceipt:
    """Confirms whether a requested action was actually applied."""

    decision_id: str
    requested_action: GuardAction
    applied_action: GuardAction
    accepted: bool
    rejection_reason: str | None

    adapter_timestamp: float
    acknowledgement_source: str


@dataclass(slots=True, frozen=True)
class RunManifest:
    """Reproducibility record for one simulation/evaluation run."""

    run_id: str
    commit_sha: str
    config_hash: str
    controller_hash: str
    model_hash: str | None
    data_hash: str | None

    dependency_lock_hash: str
    container_identity: str | None

    seed: int
    scenario_name: str
    runtime_identity: str
    hardware_identity: str

    artifact_checksums: dict[str, str] = field(default_factory=dict)
    created_at: str = ""


@dataclass(slots=True, frozen=True)
class MissionContext:
    """Offline-preprocessed geographic/route context for the geographic-risk
    guard. Runtime must not query the internet for this — it is loaded once
    from a locally packaged terrain artifact (see docs/RUNBOOK.md). Missing,
    out-of-bounds, low-resolution, or stale context must be represented
    explicitly (`valid=False` / low `resolution_m` / old `lookup_timestamp`)
    and must lower the geographic guard's confidence rather than inventing
    terrain.
    """

    latitude: float | None
    longitude: float | None
    elevation_m: float | None
    slope_deg: float | None
    route_segment: str | None
    distance_to_safe_waypoint_m: float | None

    terrain_source: str
    terrain_version: str
    coordinate_reference_system: str

    lookup_timestamp: float
    valid: bool
    resolution_m: float | None
    provenance: str

    wind_mps: float | None = None
    temperature_c: float | None = None
