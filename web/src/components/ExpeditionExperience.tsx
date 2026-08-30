"use client";

import { AnimatePresence, animate, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { trailDays } from "@/data/trail-days";
import type { ExpeditionDay } from "@/types/expedition";
import { MemoryRadio } from "@/components/MemoryRadio";
import { LiveSafetyDemo } from "@/components/LiveSafetyDemo";

type RobotState = "idle" | "travelling" | "arriving" | "reacting" | "journal-open";
const PATH = "M 70 340 C 190 292, 220 330, 320 282 S 510 250, 585 274 S 740 220, 930 166";
const STOP_Y = [17, 27, 36, 49];

function Robot({ day, state, direction }: { day: ExpeditionDay; state: RobotState; direction: 1 | -1 }) {
  const [videoFailed, setVideoFailed] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const useVideo = Boolean(day.assets.emotion) && !videoFailed && state !== "travelling";
  useEffect(() => { setVideoFailed(false); setVideoReady(false); }, [day.day]);
  return <div className={`trailRobot ${state} mood-${day.motion} ${direction < 0 ? "facesLeft" : "facesRight"} ${day.trailProgress > 75 ? "edgeRight" : ""}`} aria-label={`Pemba: ${day.moodCaption}`}>
    {useVideo && <video key={day.day} className={`robotVideo ${videoReady ? "ready" : ""}`} autoPlay loop muted playsInline preload="metadata" poster={day.assets.poster} onCanPlay={() => setVideoReady(true)} onError={() => setVideoFailed(true)}><source src={day.assets.emotion} type="video/webm" /></video>}
    <div className={`cssRobot ${videoReady ? "videoPending" : ""}`} aria-hidden="true"><span className="g1Head"><i /></span><span className="g1Torso"><b>G1</b></span><span className="g1Arm left"/><span className="g1Arm right"/><span className="g1Leg left"/><span className="g1Leg right"/></div>
    <span className="robotShadow" aria-hidden="true" />
    {state !== "travelling" && <motion.span className="moodBubble" initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}>{day.moodCaption}<small>REFLECTIVE MOOD</small></motion.span>}
  </div>;
}

function JournalDrawer({ day, onClose, onStep }: { day: ExpeditionDay; onClose: () => void; onStep: (delta: -1 | 1) => void }) {
  const side = day.trailProgress > 55 ? "left" : "right";
  return <motion.aside className={`journalDrawer ${side}`} role="dialog" aria-modal="false" aria-labelledby="journal-title" initial={{ x: side === "left" ? "-102%" : "102%", opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: side === "left" ? "-102%" : "102%", opacity: 0 }} transition={{ type: "spring", stiffness: 250, damping: 28 }}>
    <header className="journalHeader"><div><span>PEMBA MISSION LOG · DAY 0{day.day}</span><h2 id="journal-title">{day.camp}</h2><p>{day.date} · {day.reflectionTimestamp} {day.timezone}</p></div><button onClick={onClose} aria-label="Close journal">×</button></header>
    <div className="journalFacts"><span><small>ALTITUDE</small>{day.altitude ? `${day.altitude.toLocaleString()} m` : "SIMULATED"}</span><span><small>WEATHER</small>{day.weather}</span><span><small>MOOD</small>{day.moodCaption}</span></div>
    <section className="operatorNarrative"><span className="sectionLabel">OPERATOR ASSESSMENT · SEALED MISSION EVIDENCE</span><h3>Evidence scope</h3><p>{day.diary.summary}</p><h3>Why the scenario was risky or safe</h3><p>{day.diary.challenge}</p><p>{day.diary.recovery}</p><h3>Interpretation</h3><p>{day.diary.lesson}</p><h3>Next operational intention</h3><p>{day.diary.tomorrowIntent}</p></section>
    <section className="operatorTimeline"><span className="sectionLabel">TIMESTAMPED EVIDENCE LOG</span><div className="eventTimeline">{day.events.map(item => <article key={item.id}><time>{item.time}</time><div><h3>{item.title}</h3><p>{item.description}</p><code>{item.evidence}</code></div></article>)}</div></section>
    <footer className="journalNav"><button disabled={day.day === 1} onClick={() => onStep(-1)}>← Previous day</button><span>{day.day} / 4</span><button disabled={day.day === 4} onClick={() => onStep(1)}>Next day →</button></footer>
  </motion.aside>;
}

export function ExpeditionExperience() {
  const [active, setActive] = useState<"journal" | "radio" | "live">("live");
  const [day, setDay] = useState(trailDays[0]);
  const [target, setTarget] = useState(trailDays[0]);
  const [progress, setProgress] = useState(day.trailProgress);
  const [state, setState] = useState<RobotState>("reacting");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [direction, setDirection] = useState<1 | -1>(1);
  const pathRef = useRef<SVGPathElement>(null);
  const robotRef = useRef<HTMLDivElement>(null);
  const sequence = useRef(0);
  const travelControl = useRef<{ stop: () => void } | null>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    const path = pathRef.current, robot = robotRef.current;
    if (!path || !robot) return;
    const point = path.getPointAtLength(path.getTotalLength() * progress / 100);
    const scene = path.ownerSVGElement?.getBoundingClientRect();
    if (!scene) return;
    robot.style.transform = `translate(${point.x * scene.width / 1000}px, ${point.y * scene.height / 410}px)`;
  }, [progress]);

  function selectDay(next: ExpeditionDay) {
    if (next.day === day.day && state !== "travelling") { setDrawerOpen(value => !value); setState(drawerOpen ? "reacting" : "journal-open"); return; }
    const run = ++sequence.current;
    setTarget(next); setDrawerOpen(false); setDirection(next.trailProgress >= progress ? 1 : -1); setState("travelling");
    if (reduceMotion) { setProgress(next.trailProgress); setDay(next); setState("reacting"); window.setTimeout(() => { if (sequence.current === run) { setDrawerOpen(true); setState("journal-open"); } }, 80); return; }
    travelControl.current?.stop();
    const controls = animate(progress, next.trailProgress, { type: "spring", stiffness: 90, damping: 18, duration: .72, onUpdate: setProgress, onComplete: () => {
      if (sequence.current !== run) return; setDay(next); setState("arriving");
      window.setTimeout(() => { if (sequence.current !== run) return; setState("reacting"); window.setTimeout(() => { if (sequence.current === run) { setDrawerOpen(true); setState("journal-open"); } }, 250); }, 130);
    }});
    travelControl.current = controls;
  }

  function step(delta: -1 | 1) { const next = trailDays.find(item => item.day === day.day + delta); if (next) selectDay(next); }
  const displayDay = state === "travelling" ? day : target;

  return <main className={`fieldApp ${active}View ${drawerOpen ? "drawerVisible" : ""}`}>
    <header className="appHeader"><a className="brand" href="#"><span>▲</span><div><strong>ROBOT EVEREST</strong><small>Pemba’s field journal</small></div></a><nav aria-label="Main navigation"><button className={active === "live" ? "active" : ""} onClick={() => { setActive("live"); setDrawerOpen(false); }}>Live Demo <i /></button><button className={active === "radio" ? "active" : ""} onClick={() => { setActive("radio"); setDrawerOpen(false); }}>Radio <i /></button><button className={active === "journal" ? "active" : ""} onClick={() => setActive("journal")}>Journal</button></nav><div className="liveBadge"><i /> EXPEDITION ARCHIVE</div></header>
    {active === "live" && <LiveSafetyDemo />}
    <div className="skyGlow"/><div className="cloud cloudOne"/><div className="cloud cloudTwo"/>
    <svg className="mountainBackdrop" viewBox="0 0 1440 790" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="nearMountain" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#9bb4c6"/><stop offset="1" stopColor="#172a3d"/></linearGradient><linearGradient id="farMountain" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#829cb2"/><stop offset="1" stopColor="#263b4e"/></linearGradient></defs><path d="M0 510L170 286l99 108 170-247 138 207 166-278 137 231 112-155 120 208 110-126 208 282v274H0Z" fill="url(#farMountain)" opacity=".58"/><path d="M0 620l187-187 118 94 162-254 124 168 127-226 160 238 136-164 127 206 110-115 189 187v223H0Z" fill="url(#nearMountain)"/><path d="M187 433l52 41 66 53-71-21-47 40-44-19Zm280-160 55 75 69 93-91-73-33 52-39 13Zm251-58 58 87 102 151-115-113-45 57-49 22Zm296 74 63 102 64 104-92-74-35 44-48 4Z" fill="#eef6f3" opacity=".86"/><path d="M0 618q250-52 480 12t480-9 480 17v152H0Z" fill="#e9f0e9"/></svg>
    {active === "journal" ? <section className="trailScene" aria-label="Four-day mission rehearsal journal"><div className="sceneTitle"><span>SHERPAOS · FOUR-DAY MISSION REHEARSAL</span><h1>Every scenario left<br/>measurable evidence.</h1><p>Choose a day to hear what I encountered, why I judged the route risky, and what I decided.</p></div><svg className="trailSvg" viewBox="0 0 1000 410" preserveAspectRatio="none"><path className="trailBase" d={PATH}/><path className="trailComplete" d={PATH} pathLength="100" strokeDasharray={`${progress} ${100-progress}`}/><path ref={pathRef} d={PATH} fill="none" stroke="transparent"/></svg><nav className="dayStops" aria-label="Journal days">{trailDays.map((item,index) => <button key={item.day} style={{ "--stop": `${item.trailProgress}%`, "--stop-y": `${STOP_Y[index]}%` } as React.CSSProperties} className={`${item.day === target.day ? "selected" : ""} mood-${item.mood}`} onClick={() => selectDay(item)} aria-pressed={item.day === target.day}><i/><span><b>DAY {item.day}</b>{item.camp}<small>{item.moodCaption}</small></span></button>)}</nav><div ref={robotRef} className="robotPosition"><Robot day={displayDay} state={state} direction={direction}/></div><div className="trailHint">CLICK A DAY · OPEN ITS FIELD REFLECTION</div></section> : <section className="radioScene"><div className="radioIntro"><span>BASE CAMP RADIO · HALF DUPLEX</span><h1>A conversation<br/>between companions.</h1><p>Hold to talk. Pemba answers only from verified daily and recent episode summaries.</p></div><MemoryRadio initialDay={day.day} embedded /></section>}
    <AnimatePresence>{active === "journal" && drawerOpen && <JournalDrawer key={day.day} day={day} onClose={() => { setDrawerOpen(false); setState("reacting"); }} onStep={step}/>}</AnimatePresence>
    <footer className="appFooter"><span>EVEREST · 27.9881° N</span><span>MISSION REFLECTIONS · DECISIONS LINK TO EVIDENCE</span></footer>
  </main>;
}
