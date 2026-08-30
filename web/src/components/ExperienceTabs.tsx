"use client";

export type ExperienceTab = "diary" | "radio";

type ExperienceTabsProps = {
  active: ExperienceTab;
  onChange: (tab: ExperienceTab) => void;
  radioReady: boolean;
};

export function ExperienceTabs({ active, onChange, radioReady }: ExperienceTabsProps) {
  return (
    <nav className="experienceTabs" aria-label="Field Journal sections">
      <button type="button" className={active === "diary" ? "active" : ""} onClick={() => onChange("diary")} aria-pressed={active === "diary"}>
        <span className="tabIcon">✎</span>
        <span><strong>Diary</strong><small>Trail & memories</small></span>
      </button>
      <button type="button" className={active === "radio" ? "active" : ""} onClick={() => onChange("radio")} aria-pressed={active === "radio"}>
        <span className="tabIcon radioIcon">◉</span>
        <span><strong>Radio</strong><small>Talk with Pemba</small></span>
        <i className={radioReady ? "signal live" : "signal"} aria-hidden="true" />
      </button>
    </nav>
  );
}
