import day01 from "@/data/memories/day-01.json";
import day02 from "@/data/memories/day-02.json";
import day03 from "@/data/memories/day-03.json";
import day04 from "@/data/memories/day-04.json";
import type { ExpeditionDay } from "@/types/expedition";

type Memory = typeof day01 | typeof day02 | typeof day03 | typeof day04;

const reflection = (
  summary: string,
  challenge: string,
  recovery: string,
  proudMoment: string,
  lesson: string,
  tomorrowIntent: string,
) => ({ summary, challenge, recovery, proudMoment, lesson, tomorrowIntent });

function makeDay(memory: Memory, mood: ExpeditionDay["mood"], motion: ExpeditionDay["motion"], progress: number): ExpeditionDay {
  const family = memory.scenario_family;
  const train = memory.episodes.train;
  const validation = memory.episodes.validation;
  const noGo = memory.context.geographic_labels_low_slope_only.NO_GO;
  const falls = memory.physical_outcomes.train_falls + memory.physical_outcomes.validation_falls;
  const mobility = memory.positive_windows.train.mobility + memory.positive_windows.validation.mobility;
  const dynamics = memory.positive_windows.train.dynamics + memory.positive_windows.validation.dynamics;
  const sample = {
    nominal: "episode-013 · t=0.020 s · CAUTION/LIMIT_SPEED · wind 5.635 m/s · forecast 8.05 m/s · route slope 4.17°",
    mobility: "episode-052 · t=0.020 s · CAUTION/LIMIT_SPEED · wind 6.895 m/s · forecast 9.85 m/s · route slope 4.26°",
    dynamics: "episode-104 · t=0.020 s · NO-GO/REQUEST_HOLD · wind 8.000 m/s · forecast 24.85 m/s · route slope 3.18°",
    combined: "episode-162 · t=0.020 s · NO-GO/REQUEST_HOLD · wind 8.000 m/s · forecast 25.05 m/s · route slope 4.26°",
  }[family] ?? "No representative transition is available.";
  return {
    day: memory.day as 1 | 2 | 3 | 4,
    date: "SIMULATION DATASET",
    camp: memory.title,
    altitude: 0,
    weather: `${memory.context.current_wind_mps.min}–${memory.context.current_wind_mps.max} m/s wind`,
    reflectionTimestamp: "IMMUTABLE",
    timezone: "UTC",
    mood,
    moodCaption: validation ? `${train} train · ${validation} validation` : `${train} train · validation gap`,
    motion,
    trailProgress: progress,
    assets: {},
    diary: reflection(
      `${family} evidence: ${train} training and ${validation} validation episodes, ${memory.windows.train + memory.windows.validation} observation windows, and ${memory.context.control_steps.toLocaleString()} synchronized context steps.`,
      `${falls} physical falls were recorded as evaluator-only truth. Mobility-positive windows: ${mobility}; dynamics-positive windows: ${dynamics}.`,
      noGo
        ? `${noGo.toLocaleString()} low-slope context steps carried a geographic NO-GO label; these labels did not alter controller-only generation.`
        : "No geographic NO-GO labels were recorded for the low-slope context represented here.",
      "Observation arrays remained separate from privileged simulator truth and deterministic guard context.",
      validation
        ? "This is the only scenario family represented in validation."
        : "This scenario family has no validation members, so its validation performance is currently unmeasured.",
      memory.real_world_telemetry.statement,
    ),
    events: [
      { id: `d${memory.day}-split`, time: "SPLIT", title: "Immutable membership", description: `${train} train · ${validation} validation episodes`, kind: "milestone", evidence: memory.provenance.package },
      { id: `d${memory.day}-risk`, time: "LABELS", title: "Risk outcomes", description: `${mobility} mobility-positive and ${dynamics} dynamics-positive windows; ${falls} physical falls.`, kind: "challenge", evidence: "privileged labels / selected train+validation IDs" },
      { id: `d${memory.day}-decision`, time: "t=0.020 s", title: "Representative guard decision", description: sample, kind: noGo ? "challenge" : "observation", evidence: `low-slope context · memory/day-${String(memory.day).padStart(2, "0")}.json` },
      { id: `d${memory.day}-real`, time: "FIELD", title: "Real-world telemetry", description: memory.real_world_telemetry.statement, kind: "observation", evidence: "UNAVAILABLE · 0 hardware episodes" },
    ],
  };
}

export const trailDays: ExpeditionDay[] = [
  makeDay(day01, "calm", "breathe", 10),
  makeDay(day02, "brave", "zombie", 36),
  makeDay(day03, "tired", "tantrum", 63),
  makeDay(day04, "proud", "victory", 90),
];
