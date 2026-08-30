"use client";

import { useEffect, useState, type PointerEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import { expeditionDays } from "@/data/days";
import type { ExpeditionDay } from "@/types/expedition";
import { MemoryRadio } from "@/components/MemoryRadio";

const currentDay = expeditionDays.find((day) => day.status === "waiting") ?? expeditionDays[0];

export default function Home() {
  const [selectedDay, setSelectedDay] = useState<ExpeditionDay>(currentDay);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [pembaReaction, setPembaReaction] = useState("waiting for today’s memories…");
  const [collectedMemories, setCollectedMemories] = useState<string[]>([]);
  const [lanternOn, setLanternOn] = useState(true);

  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  function selectDay(day: ExpeditionDay) {
    if (day.status === "locked") return;
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
    setSelectedDay(day);
  }

  function moveMountain(event: PointerEvent<HTMLElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
    event.currentTarget.style.setProperty("--pointer-x", x.toFixed(2));
    event.currentTarget.style.setProperty("--pointer-y", y.toFixed(2));
  }

  function pokePemba() {
    const reactions = [
      "oh! hello down there 👋",
      "my antenna says you are friendly",
      "do you think snow tastes cold?",
      "I am practicing my brave face!",
    ];
    const currentIndex = reactions.indexOf(pembaReaction);
    setPembaReaction(reactions[(currentIndex + 1) % reactions.length]);
  }

  function collectMemory(memory: string) {
    setCollectedMemories((memories) => memories.includes(memory) ? memories : [...memories, memory]);
  }

  function speakCurrentThought() {
    if (!("speechSynthesis" in window)) return;
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }
    const thought = new SpeechSynthesisUtterance(
      "I am waiting at the Western Cwm. The valley is very bright today. I keep thinking about the blue ice behind me, and the long trail ahead. I hope today's memories arrive soon.",
    );
    thought.rate = 0.88;
    thought.pitch = 1.12;
    thought.onend = () => setIsSpeaking(false);
    thought.onerror = () => setIsSpeaking(false);
    setIsSpeaking(true);
    window.speechSynthesis.speak(thought);
  }

  const isCurrent = selectedDay.day === currentDay.day;

  return (
    <main className="shell">
      <div className="stars" aria-hidden="true" />
      <header className="masthead">
        <div>
          <p className="eyebrow">Robot Everest · Expedition 001</p>
          <h1>Pemba&apos;s Field Journal</h1>
        </div>
        <div className="altitude">
          <strong>{selectedDay.altitude.toLocaleString()} m</strong>
          <span>Day {selectedDay.day} of 5</span>
        </div>
      </header>

      <section className={`mountain ${lanternOn ? "lanternOn" : ""}`} aria-label="Five-day Himalayan expedition trail" onPointerMove={moveMountain}>
        <div className="aurora" aria-hidden="true" />
        <div className="cloud cloudOne" aria-hidden="true" />
        <div className="cloud cloudTwo" aria-hidden="true" />
        <div className="snow" aria-hidden="true">{Array.from({ length: 22 }, (_, index) => <i key={index} />)}</div>
        <svg className="terrainRelief" viewBox="0 0 1200 700" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id="terrainBase" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#151c26" />
              <stop offset=".48" stopColor="#505866" />
              <stop offset="1" stopColor="#0a1019" />
            </linearGradient>
            <linearGradient id="snowFace" x1="0" y1="0" x2=".8" y2="1">
              <stop offset="0" stopColor="#d3d8dc" stopOpacity=".75" />
              <stop offset=".42" stopColor="#78818d" stopOpacity=".58" />
              <stop offset="1" stopColor="#171e29" stopOpacity=".3" />
            </linearGradient>
            <filter id="softRelief"><feGaussianBlur stdDeviation="7" /></filter>
            <filter id="routeGlow"><feGaussianBlur stdDeviation="5" /></filter>
            <pattern id="contours" width="84" height="36" patternUnits="userSpaceOnUse" patternTransform="rotate(-7)">
              <path d="M-20 18 Q20 -3 62 17 T146 18" fill="none" stroke="#dce5ee" strokeOpacity=".075" strokeWidth="1" />
              <path d="M-20 28 Q20 7 62 27 T146 28" fill="none" stroke="#dce5ee" strokeOpacity=".045" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="1200" height="700" fill="#070c14" />
          <path d="M0 700V428L100 386L196 427L283 343L368 427L454 307L525 355L610 194L667 267L728 92L785 202L845 293L910 230L980 355L1066 275L1200 410V700Z" fill="#171f2b" />
          <path d="M163 700L455 307L369 427L282 343L0 700Z" fill="#39414d" />
          <path d="M308 700L610 194L667 267L728 92L704 312L614 450Z" fill="url(#snowFace)" />
          <path d="M728 92L785 202L846 293L804 453L704 312Z" fill="#111822" />
          <path d="M728 92L745 159L704 220L674 265Z" fill="#dce2e644" />
          <path d="M560 700L704 312L728 92L803 454L930 700Z" fill="#252d38" />
          <path d="M770 700L910 230L980 355L1066 275L1200 410V700Z" fill="#121923" />
          <path d="M0 700V428L100 386L196 427L283 343L368 427L454 307L525 355L610 194L667 267L728 92L785 202L845 293L910 230L980 355L1066 275L1200 410V700Z" fill="url(#contours)" />
          <ellipse cx="600" cy="650" rx="540" ry="95" fill="#02050a" opacity=".72" filter="url(#softRelief)" />
        </svg>
        <svg className="trail" viewBox="0 0 1000 520" role="img" aria-label="Route from Base Camp to the summit">
          <path className="trailShadow" d="M500 495 C485 455 535 430 510 392 C484 353 541 330 526 289 C513 254 563 231 546 194 C531 162 566 137 558 102 C552 77 570 58 575 38" />
          <path className="trailFuture" d="M500 495 C485 455 535 430 510 392 C484 353 541 330 526 289 C513 254 563 231 546 194 C531 162 566 137 558 102 C552 77 570 58 575 38" />
          <path className="trailDone" d="M500 495 C485 455 535 430 510 392 C484 353 541 330 526 289" />
        </svg>

        <button className="pemba" type="button" aria-label="Say hello to Pemba" onClick={pokePemba}>
          <span className="antenna" />
          <span className="head"><i /></span>
          <span className="body" />
          <span className="feet" />
          <AnimatePresence mode="wait">
            <motion.span className="thought" key={isSpeaking ? "speaking" : pembaReaction} initial={{ opacity: 0, scale: .9, y: 5 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: .9 }}>
              {isSpeaking ? "telling you what I am thinking…" : pembaReaction}
            </motion.span>
          </AnimatePresence>
          {isSpeaking && <span className="voiceWaves" aria-hidden="true"><i /><i /><i /></span>}
        </button>

        <div className="memoryCharms" aria-label="Collect Pemba's trail memories">
          {[
            ["snowflake", "❄", "A perfectly tiny snowflake"],
            ["footprint", "⌁", "A brave footprint"],
            ["star", "✦", "A mountain wish"],
          ].map(([id, icon, label]) => (
            <button key={id} type="button" className={collectedMemories.includes(id) ? "collected" : ""} onClick={() => collectMemory(id)} aria-label={label} title={label}>
              <span>{collectedMemories.includes(id) ? "✓" : icon}</span>
            </button>
          ))}
        </div>

        <div className="prayerFlags" aria-hidden="true"><i /><i /><i /><i /><i /></div>

        <nav className="camps" aria-label="Expedition days">
          {expeditionDays.map((day) => (
            <button
              key={day.day}
              className={`camp camp-${day.day} ${day.status} ${selectedDay.day === day.day ? "selected" : ""}`}
              type="button"
              disabled={day.status === "locked"}
              onClick={() => selectDay(day)}
              aria-pressed={selectedDay.day === day.day}
              aria-label={`${day.status === "complete" ? "Open diary for" : day.status === "waiting" ? "Hear Pemba's thoughts for" : "Locked"} day ${day.day}, ${day.camp}`}
            >
              <span className="campDot" />
              <span className="campLabel">Day {day.day}<strong>{day.camp}</strong></span>
            </button>
          ))}
        </nav>

        <AnimatePresence mode="wait">
        <motion.aside key={selectedDay.day} className={`storyCard ${isCurrent ? "currentThoughts" : "pastDiary"}`} initial={{ opacity: 0, x: 24, rotate: 2 }} animate={{ opacity: 1, x: 0, rotate: isCurrent ? .5 : -.6 }} exit={{ opacity: 0, x: 18, scale: .96 }} transition={{ type: "spring", stiffness: 220, damping: 22 }}>
          {isCurrent ? (
            <>
              <p className="eyebrow">Live from the trail · Today</p>
              <h2>{selectedDay.camp}</h2>
              <p className="currentPrompt">Pemba is still collecting today&apos;s memories. Ask what is on its mind right now.</p>
              <ul>{selectedDay.sherpaPlan.map((item) => <li key={item}>{item}</li>)}</ul>
              <button type="button" onClick={speakCurrentThought} className={isSpeaking ? "speaking" : ""}>
                <span>{isSpeaking ? "Stop Pemba" : "Hear Pemba's thoughts"}</span><span>{isSpeaking ? "■" : "▶"}</span>
              </button>
            </>
          ) : (
            <>
              <p className="eyebrow">Written at {selectedDay.camp}</p>
              <h2>{selectedDay.diary?.title}</h2>
              <p className="diaryBody">{selectedDay.diary?.body}</p>
              <div className="diaryNote"><span>Proud moment</span>{selectedDay.diary?.proudMoment}</div>
              <button type="button" onClick={() => selectDay(currentDay)}><span>Return to today</span><span>↗</span></button>
            </>
          )}
        </motion.aside>
        </AnimatePresence>

        <button className="lanternSwitch" type="button" onClick={() => setLanternOn((value) => !value)} aria-pressed={lanternOn}>
          <span>{lanternOn ? "✦" : "☾"}</span>{lanternOn ? "Lantern on" : "Night mode"}
        </button>

        <div className="memoryPouch" aria-live="polite">
          <span>Memory pouch</span><strong>{collectedMemories.length}/3</strong>
        </div>
      </section>

      <footer className="dayRail" aria-label="Select an expedition day">
        {expeditionDays.map((day) => (
          <button key={day.day} type="button" className={`${day.status} ${selectedDay.day === day.day ? "selected" : ""}`} onClick={() => selectDay(day)} disabled={day.status === "locked"}>
            <i /> Day {day.day}<small>{day.status === "complete" ? "Diary" : day.status === "waiting" ? "Today" : "Locked"}</small>
          </button>
        ))}
      </footer>
      <MemoryRadio initialDay={selectedDay.day} />
    </main>
  );
}
