# SherpaOS risk inputs and episode data requirements

## Deterministic guards (3, 4, and 5)

### Telemetry health

Inputs are the timestamp, sequence, producer-valid flag, provenance, joint position,
joint velocity, optional joint effort, IMU orientation quaternion, IMU angular velocity,
IMU linear acceleration, and optional commanded velocity. The guard checks shape,
finite values, freshness, future timestamps, ordering/duplicate sequence numbers,
producer validity, and missing risk-relevant optional fields. A malformed stream never
becomes a confident `PASS`.

### Battery margin

Inputs are state of charge (`battery_fraction`), pack voltage, signed current, pack
temperature, timestamp, and source provenance. A rolling window checks effective cold-
derated charge margin, discharge rate/time remaining, and voltage sag under confirmed
load. Missing or malformed battery data produces a conservative low-confidence report.

### Geographic and environmental risk

Inputs are offline-resolved latitude/longitude, elevation, slope, route segment,
exposure class, distance to a safe waypoint, artifact version/CRS/resolution,
lookup timestamp, validity/provenance, and optional wind speed and ambient temperature.
The runtime performs no network lookup. It checks terrain availability/staleness, steep
slope, exposure, remoteness, high wind, and extreme cold. Severe terrain/environment
conditions can request a hold without being averaged down by other guards.

Depth images are not currently a guard input. Raw depth pixels need a separate,
timestamped adapter contract with calibration, frame ID, range, validity, and freshness.
After that exists, derived obstacle clearance/traversability can feed a deterministic
perception guard or the mobility model. Until then, depth may be recorded as auxiliary
evidence but must not silently influence the safety decision.

## Why generate 200 episodes for guards 1 and 2?

An episode is one continuous, seeded robot rollout with a controller, command trace,
environment, telemetry, and evaluator-only ground truth. The configured 200 episodes
(50 nominal, 50 mobility, 50 dynamics, 50 combined) are the minimum first experiment:
enough independent scenario groups to split by episode/group, exercise both labels, and
compare a learned model against deterministic rules. They are not enough to claim field
validation. If the first audit is healthy, expand scenario diversity before merely
adding more near-duplicate windows.

Training converts each episode into 2-second observation windows sampled at 50 Hz and
asks whether mobility or dynamics risk occurs in the following 1 second. With a stride
of 10 samples, adjacent examples overlap, so they are correlated. Splitting must happen
by scenario group/episode—not by window—to prevent near-duplicate train/test leakage.

## Episode deliverables

The numerical telemetry and label artifacts train/evaluate the model. Video does not
train the current TCN. Video is synchronized qualitative evidence used to verify that
the injected event visibly happened, labels align with it, the robot remains visible,
and a reviewer can distinguish nominal, slip/disturbance, recovery, and fall behavior.

Each episode must contain:

- immutable episode ID, scenario-group ID, seed, controller and simulator versions;
- 50 Hz monotonic telemetry with no unexplained gaps or duplicate timestamps;
- 29 joint positions, velocities, optional efforts, IMU orientation/angular velocity/
  linear acceleration, commanded velocity, validity, and field provenance;
- evaluator-only friction, slope, planted-foot slip, actuator health, disturbance timing,
  tilt, and fall outcome stored separately from observations;
- command/activity phase, hazard onset/offset, and episode termination reason;
- synchronized video timestamp/timebase and camera metadata when video is captured;
- checksum, source manifest, units, coordinate frames, and generation configuration.

## Quality gates before training

1. Stable nominal behavior: nominal episodes do not fall or constantly trigger hazards.
2. Physical response: low friction changes planted-foot slip/body response; impulses and
   actuator degradation measurably change dynamics.
3. Label alignment: future-horizon targets change at the correct timestamps and never
   appear in observation tensors.
4. Coverage: speeds, turns, holds, slopes, friction levels, disturbance directions,
   payload/actuator cases, sensor noise, and combined hazards vary across groups.
5. Hard negatives: aggressive but safe commands, rough motion without unsafe outcomes,
   and cold/windy contexts without traction failure are represented.
6. Balance: every output has adequate positive and negative episodes; report rates by
   episode and window rather than hiding imbalance in one aggregate.
7. Independence: train/validation/test groups share no seed, episode, command trace, or
   near-identical environment configuration.
8. Sensor realism: units/ranges match the intended G1 interface; noise, latency,
   missingness, reordering, and saturation are explicit and provenance-tagged.
9. Visual audit: the G1 is visible, motion is not occluded, event timing is recognizable,
   and video agrees with telemetry/labels. Snow texture alone is not evidence of ice.
10. Reproducibility: rerunning a seed/config produces the same artifact hashes, excluding
    explicitly documented nondeterministic rendering bytes.

Do not train if the 20-episode pilot fails any of gates 1–4. Do not scale to thousands of
episodes until held-out results show that added scenario diversity improves performance.

## Five-day expedition timeline (separate from model training)

Short training episodes and hours-long expedition memory serve different purposes. The
five demo days should be stored as one or more immutable ROS bags per day, plus a derived
timeline. A day is not ready for reflection merely because a bag file exists.

Each day needs an explicit calendar date, IANA time zone (normally `Asia/Kathmandu` for
the expedition story), UTC start/end instants, and a declared timestamp basis for every
source (`unix_epoch`, `ros_sim_time`, or `monotonic`). Never infer a calendar date from
monotonic/simulation time. Preserve both source time and receive/monotonic time when
available so delay and reordering remain measurable.

Before a day becomes `READY_FOR_QUESTIONS`, the derived pipeline must:

1. decode approved topics into typed rows rather than expose only topic counts;
2. normalize units and map each sample to `timestamp_utc_ns` plus `elapsed_day_s`;
3. record source clock, frame ID, sequence, validity, and provenance per row;
4. align IMU/joints/command at the control rate without pretending interpolated values
   were observed; battery/weather/geography may use slower last-known-valid joins with
   a recorded age;
5. detect gaps, duplicates, out-of-order samples, clock resets, rate drift, and intervals
   where required topics do not overlap;
6. create deterministic event intervals (risk onset, action, receipt, recovery), each
   referencing the exact source rows and bag hash;
7. publish coverage by topic and session, including downtime—not one misleading span
   from the day's first message to its last;
8. verify that the configured day/date agrees with epoch-based bag timestamps, or refuse
   promotion when it cannot be proven.

Current implementation status: bag ingestion preserves nanosecond start/end times and
per-topic coverage, but value decoding, common-clock alignment, gap/rate audit, explicit
date/time-zone mapping, and derived event generation are not yet implemented. Therefore
the existing bags are evidence-preserving inputs, not yet a complete five-day temporal
dataset for grounded reflection.
