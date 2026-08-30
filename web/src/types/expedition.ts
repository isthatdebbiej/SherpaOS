export type DayStatus = "locked" | "waiting" | "processing" | "complete";
export type PembaMood = "curious" | "brave" | "tired" | "hopeful" | "proud";

export type ExpeditionEvent = {
  id: string;
  time: string;
  title: string;
  description: string;
  kind: "discovery" | "challenge" | "recovery" | "milestone";
};

export type DiaryEntry = {
  title: string;
  body: string;
  proudMoment: string;
  lesson: string;
  tomorrowIntent: string;
};

export type ExpeditionDay = {
  day: 1 | 2 | 3 | 4 | 5;
  camp: string;
  altitude: number;
  status: DayStatus;
  sherpaPlan: string[];
  progress: number;
  mood?: PembaMood;
  weather: string;
  diary?: DiaryEntry;
  events: ExpeditionEvent[];
  narrationUrl?: string;
};
