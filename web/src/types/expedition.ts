export type DayMood = "tired" | "brave" | "calm" | "joyful" | "proud";
export type RobotMotion = "tantrum" | "zombie" | "breathe" | "dance" | "victory";
export type ContentKind = "fact" | "expressive_reflection";

export type ExpeditionEvent = {
  id: string;
  time: string;
  title: string;
  description: string;
  kind: "challenge" | "recovery" | "milestone" | "observation";
  evidence: string;
};

export type DiaryEntry = {
  summary: string;
  challenge: string;
  recovery: string;
  proudMoment: string;
  lesson: string;
  tomorrowIntent: string;
};

export type RobotAssets = {
  walking?: string;
  arrival?: string;
  emotion?: string;
  poster?: string;
};

export type ExpeditionDay = {
  day: 1 | 2 | 3 | 4 | 5;
  date: string;
  camp: string;
  altitude: number;
  weather: string;
  reflectionTimestamp: string;
  timezone: string;
  mood: DayMood;
  moodCaption: string;
  motion: RobotMotion;
  trailProgress: number;
  assets: RobotAssets;
  diary: DiaryEntry;
  events: ExpeditionEvent[];
};
