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
  const trials = train + validation;
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
    date: "MISSION REHEARSAL",
    camp: ({ nominal: "Base Camp Route Qualification", mobility: "Icy Traction Corridor", dynamics: "Exposed Wind Face", combined: "Storm Decision Boundary" } as Record<string, string>)[family],
    altitude: 0,
    weather: `${memory.context.current_wind_mps.min}–${memory.context.current_wind_mps.max} m/s wind`,
    reflectionTimestamp: "IMMUTABLE",
    timezone: "UTC",
    mood,
    moodCaption: validation ? `${trials} route trials reviewed` : `${trials} route trials reviewed`,
    motion,
    trailProgress: progress,
    assets: {},
    diary: reflection(
      `I reviewed ${trials} route trials, ${memory.windows.train + memory.windows.validation} motion windows, and ${memory.context.control_steps.toLocaleString()} synchronized context steps to understand when this route was safe and when I should stop.`,
      `I observed ${mobility} mobility-risk windows and ${dynamics} body-risk windows. ${falls} trials reached the physical fall boundary when walking continued.`,
      noGo
        ? `I identified ${noGo.toLocaleString()} NO-GO context steps. The correct operational response was to hold before the physical boundary, even while I was still standing.`
        : "I did not identify a NO-GO route state here; GO or CAUTION remained sufficient, with CAUTION requiring reduced speed.",
      "I based the decision on motion evidence and deterministic route, telemetry, and battery context, keeping outcome truth separate from what I could observe.",
      noGo
        ? "The route exposure consumed my remaining safety margin, so I should hold position."
        : "The remaining margin supported GO or a speed-limited CAUTION.",
      "On the next route segment I will attach each decision to its reason, timestamp, decision ID, and actuation receipt.",
    ),
    events: [
      { id: `d${memory.day}-split`, time: "MISSION", title: "Route evidence reviewed", description: `${trials} route trials reviewed`, kind: "milestone", evidence: memory.provenance.package },
      { id: `d${memory.day}-risk`, time: "MOTION", title: "Why I judged the route risky", description: `${mobility} mobility-risk and ${dynamics} body-risk windows; ${falls} physical falls.`, kind: "challenge", evidence: "sealed motion evidence and physical outcomes" },
      { id: `d${memory.day}-decision`, time: "t=0.020 s", title: "Representative guard decision", description: sample, kind: noGo ? "challenge" : "observation", evidence: `sealed route context / memory/day-${String(memory.day).padStart(2, "0")}.json` },
      { id: `d${memory.day}-real`, time: "NEXT", title: "My next operational intention", description: "On the next route segment I will attach each decision to its reason, timestamp, decision ID, and actuation receipt.", kind: "observation", evidence: "decision reasons · timestamp · receipt" },
    ],
  };
}

export const trailDays: ExpeditionDay[] = [
  makeDay(day01, "calm", "breathe", 10),
  makeDay(day02, "brave", "zombie", 36),
  makeDay(day03, "tired", "tantrum", 63),
  makeDay(day04, "proud", "victory", 90),
];
