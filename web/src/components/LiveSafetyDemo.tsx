"use client";

import { useEffect, useMemo, useState } from "react";
import "./LiveSafetyDemo.css";

type Report = {
  guard: string;
  score: number;
  confidence: number;
  reason_codes: string[];
  recommended_action: "PASS" | "LIMIT_SPEED" | "REQUEST_HOLD";
  provenance: Record<string, string>;
};
type Event = {
  decision: {
    decision_id: string;
    action: "PASS" | "LIMIT_SPEED" | "REQUEST_HOLD";
    score: number;
    confidence: number;
    timestamp: number;
    model_version?: string;
    guard_reports: Report[];
  };
  receipt: { accepted: boolean; applied_action: string; acknowledgement_source: string };
  requested_velocity_mps: number;
  applied_velocity_mps: number;
};

const API = process.env.NEXT_PUBLIC_SHERPA_API ?? "http://127.0.0.1:8000";
const WS = API.replace(/^http/, "ws");
const actionLabel = { PASS: "GO", LIMIT_SPEED: "CAUTION", REQUEST_HOLD: "NO-GO" };
const reasonLabel: Record<string, string> = {
  NOMINAL: "Within configured range",
  SLIP_RISK_HIGH: "Strong traction-loss signature",
  SLIP_RISK_ELEVATED: "Elevated traction-loss signature",
  BODY_ANOMALY: "Unexpected body response",
  ASYMMETRY_DETECTED: "Left/right motion imbalance",
  ORIENTATION_INSTABILITY: "IMU orientation instability",
  STALE_TELEMETRY: "Sensor data is stale",
  MISSING_FIELD: "Required sensor field missing",
  NAN_OR_INVALID: "Invalid sensor value",
  OUT_OF_ORDER: "Sensor sequence out of order",
  FUTURE_DATED_TELEMETRY: "Sensor timestamp is ahead of runtime",
  BATTERY_MARGIN_LOW: "Cold-adjusted charge margin is low",
  BATTERY_VOLTAGE_SAG: "Pack voltage is below the expected level under load",
  BATTERY_COLD_DERATED: "Cold temperature reduces usable battery capacity",
  BATTERY_DATA_UNAVAILABLE: "Battery measurement unavailable",
  GEOGRAPHIC_STEEP_SLOPE: "Route grade exceeds the configured limit",
  GEOGRAPHIC_HIGH_EXPOSURE: "Route exposure is high",
  GEOGRAPHIC_FAR_FROM_SAFE_WAYPOINT: "Long return distance to a safe waypoint",
  GEOGRAPHIC_CONTEXT_UNAVAILABLE: "Route position or terrain context unavailable",
  GEOGRAPHIC_CONTEXT_STALE: "Route position is stale",
  ENVIRONMENT_HIGH_WIND: "Wind exceeds the configured operating threshold",
  ENVIRONMENT_EXTREME_COLD: "Ambient temperature is extremely cold",
};

const present = (value?: string) => value && value !== "unavailable";
const percent = (value?: string) => present(value) ? `${Math.round(Number(value) * 100)}%` : "unavailable";
const fixed = (value: string | undefined, unit: string, digits = 1) =>
  present(value) ? `${Number(value).toFixed(digits)} ${unit}` : "unavailable";

function guardView(report: Report) {
  const p = report.provenance ?? {};
  const samples = p.sample_count ?? p.window_sample_count ?? "0";
  const views: Record<string, { title: string; covers: string; evidence: string[] }> = {
    mobility: {
      title: "Traction stability",
      covers: "Leg-joint motion compared with the command and IMU body response",
      evidence: [
        `Traction anomaly: ${percent(p.slip_proxy)}`,
        `${samples} joint + IMU samples`,
        `Latest sample age: ${fixed(p.input_age_seconds, "s", 3)}`,
      ],
    },
    dynamics: {
      title: "Body & IMU stability",
      covers: "IMU roll, pitch and angular rate; joint residuals; left/right balance",
      evidence: [
        `IMU roll / pitch: ${fixed(p.imu_roll_deg, "°")} / ${fixed(p.imu_pitch_deg, "°")}`,
        `IMU angular rate: ${fixed(p.imu_angular_rate_rad_s, "rad/s", 2)}`,
        `Body response / asymmetry: ${percent(p.body_component)} / ${percent(p.asymmetry_component)}`,
      ],
    },
    telemetry_health: {
      title: "Sensor stream integrity",
      covers: "Joint position, velocity and effort; IMU; command; validity, order and freshness",
      evidence: [
        `${samples} consecutive samples checked`,
        `Latest sample age: ${fixed(p.input_age_seconds, "s", 3)}`,
        `Missing fields: ${p.missing_fields || "none"}`,
      ],
    },
    battery: {
      title: "Battery operating margin",
      covers: "Charge, cold derating, pack voltage/current and discharge trend",
      evidence: [
        `Charge / cold-adjusted: ${percent(p.battery_fraction)} / ${percent(p.effective_fraction)}`,
        `Pack: ${fixed(p.battery_voltage_v, "V")} · ${fixed(p.battery_current_a, "A")}`,
        `Temperature / estimated time: ${fixed(p.battery_temperature_c, "°C")} / ${fixed(p.estimated_remaining_min, "min")}`,
      ],
    },
    geographic: {
      title: "Route & environment",
      covers: "Route grade, elevation, exposure, distance to safety, wind and temperature",
      evidence: [
        `Elevation / route grade: ${fixed(p.elevation_m, "m", 0)} / ${fixed(p.route_slope_deg, "°")}`,
        `Distance to safety: ${present(p.distance_to_safe_waypoint_m) ? `${(Number(p.distance_to_safe_waypoint_m) / 1000).toFixed(1)} km` : "unavailable"}`,
        `Wind / air temperature: ${fixed(p.wind_mps, "m/s")} / ${fixed(p.temperature_c, "°C")}`,
      ],
    },
  };
  return views[report.guard] ?? {
    title: report.guard.replaceAll("_", " "),
    covers: "Guard evidence",
    evidence: [],
  };
}

export function LiveSafetyDemo() {
  const [event, setEvent] = useState<Event | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<Event[]>([]);
  const [streamAttempt, setStreamAttempt] = useState(0);
  const [streamFailed, setStreamFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnect: ReturnType<typeof setTimeout> | null = null;
    const receive = (value: Event) => {
      setEvent(value);
      setEvents(items => items[0]?.decision.decision_id === value.decision.decision_id
        ? items
        : [value, ...items].slice(0, 8));
    };
    const connect = () => {
      if (cancelled) return;
      socket = new WebSocket(`${WS}/ws/supervisor`);
      socket.onopen = () => setConnected(true);
      socket.onmessage = message => receive(JSON.parse(message.data) as Event);
      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) reconnect = setTimeout(connect, 1500);
      };
      socket.onerror = () => socket?.close();
    };
    const poll = async () => {
      try {
        const response = await fetch(`${API}/api/supervisor/current`, { cache: "no-store" });
        if (response.ok) receive(await response.json() as Event);
      } catch { /* WebSocket reconnect and the next poll remain available. */ }
    };
    connect();
    void poll();
    const poller = setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      if (reconnect) clearTimeout(reconnect);
      clearInterval(poller);
      socket?.close();
    };
  }, []);
  useEffect(() => {
    if (!streamFailed) return;
    const retry = setTimeout(() => {
      setStreamAttempt(value => value + 1);
      setStreamFailed(false);
    }, 1500);
    return () => clearTimeout(retry);
  }, [streamFailed]);
  const guards = useMemo(() => event?.decision.guard_reports ?? [], [event]);
  const verdict = event ? actionLabel[event.decision.action] : "NO LIVE EVIDENCE";

  return <section className="liveDemo" aria-label="Live SherpaOS safety supervisor">
    <header><div><span>VULTR · LIVE SUPERVISOR</span><h1>Robot safety decision</h1><p>The robot replay and live supervisor are separate evidence streams. Every live verdict requires a matching actuation receipt.</p></div><b className={connected ? "connected" : "offline"}>{connected ? "STREAM CONNECTED" : "RUNTIME OFFLINE"}</b></header>
    <div className="liveGrid">
      <article className="robotFeed"><div className="feedLabel">QUALIFIED G1 SIMULATION REPLAY</div><img key={streamAttempt} className={streamFailed ? "streamError" : ""} src={`${API}/stream/robot?attempt=${streamAttempt}`} alt="Qualified MuJoCo G1 simulation replay" onLoad={() => setStreamFailed(false)} onError={() => setStreamFailed(true)}/><div className="feedEmpty">{streamFailed ? "Reconnecting to Vultr robot stream…" : "Waiting for Vultr robot stream"}</div></article>
      <aside className={`verdict ${event?.decision.action.toLowerCase() ?? "unknown"}`}><small>SHERPAOS POLICY</small><strong>{verdict}</strong>{event && <><p>Requested <b>{event.requested_velocity_mps.toFixed(2)} m/s</b> · Applied <b>{event.applied_velocity_mps.toFixed(2)} m/s</b></p><dl><div><dt>Decision ID</dt><dd>{event.decision.decision_id}</dd></div><div><dt>Receipt</dt><dd>{event.receipt.accepted ? "ACCEPTED" : "REJECTED"}</dd></div><div><dt>Decision source</dt><dd>{event.decision.model_version ?? "five deterministic guards"}</dd></div></dl></>}</aside>
    </div>
    <div className="guardGrid">{guards.map(report => { const view = guardView(report); return <article key={report.guard}><div className="guardHeading"><span>{view.title}</span><div><b>{Math.round(report.score * 100)}%</b><em>risk</em></div></div><p>{view.covers}</p><meter min="0" max="1" value={report.score}/><ul>{view.evidence.map(line => <li key={line}>{line}</li>)}</ul><small>{report.reason_codes.map(code => reasonLabel[code] ?? code.replaceAll("_", " ").toLowerCase()).join(" · ")}</small><i>{Math.round(report.confidence * 100)}% evidence confidence</i></article>; })}</div>
    <section className="evidenceLog"><h2>Immutable decision evidence</h2>{events.length ? events.map(item => <div key={item.decision.decision_id}><time>{item.decision.timestamp.toFixed(2)}s</time><b>{actionLabel[item.decision.action]}</b><code>{item.decision.decision_id}</code><span>{item.requested_velocity_mps.toFixed(2)} → {item.applied_velocity_mps.toFixed(2)} m/s</span><i>{item.receipt.accepted ? "RECEIPT MATCHED" : "NO RECEIPT"}</i></div>) : <p>No decision has been received. The UI will not manufacture demo data.</p>}</section>
  </section>;
}