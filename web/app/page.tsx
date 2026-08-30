"use client";

import { ExpeditionExperience } from "@/components/ExpeditionExperience";

export default function Home() {
  return <ExpeditionExperience />;
}

/* Legacy composition retained temporarily for reference.

import { useEffect, useState, type PointerEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import { expeditionDays } from "@/data/days";
import type { ExpeditionDay } from "@/types/expedition";
import { MemoryRadio } from "@/components/MemoryRadio";
import { ExperienceTabs, type ExperienceTab } from "@/components/ExperienceTabs";

const today = expeditionDays.find(day => day.status === "waiting") ?? expeditionDays[0];

export default function Home() {
  const [selectedDay, setSelectedDay] = useState<ExpeditionDay>(today);
  const [activeTab, setActiveTab] = useState<ExperienceTab>("diary");
  const [speaking, setSpeaking] = useState(false);
  useEffect(() => () => window.speechSynthesis?.cancel(), []);

  function selectDay(day: ExpeditionDay) { if (day.status === "locked") return; window.speechSynthesis?.cancel(); setSpeaking(false); setSelectedDay(day); }
  function parallax(event: PointerEvent<HTMLElement>) { const rect=event.currentTarget.getBoundingClientRect(); event.currentTarget.style.setProperty("--mx",String((event.clientX-rect.left)/rect.width-.5)); event.currentTarget.style.setProperty("--my",String((event.clientY-rect.top)/rect.height-.5)); }
  function speak() { if (!("speechSynthesis" in window)) return; if(speaking){window.speechSynthesis.cancel();setSpeaking(false);return} const line=new SpeechSynthesisUtterance("The Western Cwm is bright and quiet today. I am saving my battery, watching the blue shadows, and waiting for my memories to arrive from base.");line.rate=.87;line.pitch=1.08;line.onend=line.onerror=()=>setSpeaking(false);setSpeaking(true);window.speechSynthesis.speak(line); }
  const current=selectedDay.day===today.day;

  return <main className={`expedition ${activeTab}Mode ${current?"":"archiveMode"}`} onPointerMove={parallax}>
    <div className="grain" aria-hidden="true"/>
    <header className="topbar">
      <a className="wordmark" href="#"><span>▲</span><strong>ROBOT EVEREST</strong><small>Pemba field journal</small></a>
      <ExperienceTabs active={activeTab} onChange={setActiveTab} radioReady/>
      <div className="baseStatus"><i/> EXPEDITION <strong>LIVE</strong></div>
    </header>
    <section className="heroCopy"><p className="kicker">EXPEDITION 001 <span>·</span> LIVE FROM EVEREST</p><h1>PEMBA&apos;S<br/><em>JOURNAL</em></h1><p className="intro">Five days. One careful robot.<br/>A mountain full of memories.</p></section>

    <section className="everestStage" aria-label="Interactive Everest expedition route">
      <div className="summitMist"/>
      <svg className="ridgeFace" viewBox="0 0 1000 760" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="ridgeIce" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#8fa5ba"/><stop offset=".42" stopColor="#31445a"/><stop offset="1" stopColor="#08111e"/></linearGradient><pattern id="ridgeLines" width="15" height="15" patternUnits="userSpaceOnUse" patternTransform="rotate(19)"><path d="M0 0V15" stroke="#dbe8ef" strokeOpacity=".1" strokeWidth="2"/></pattern></defs><path d="M0 760V704L105 664 205 626 305 570 405 548 505 489 610 459 710 405 792 350 858 282 916 378 1000 451V760Z" fill="url(#ridgeIce)"/><path d="M0 760V704L105 664 205 626 305 570 405 548 505 489 610 459 710 405 792 350 858 282 916 378 1000 451V760Z" fill="url(#ridgeLines)"/><path d="M858 282L916 378 1000 451V760H730L792 556 821 416Z" fill="#07101c" opacity=".72"/><path d="M858 282L821 416 792 556 730 760" fill="none" stroke="#d7e4ec" strokeOpacity=".18" strokeWidth="4"/></svg>
      <svg className="mountainFace" viewBox="0 0 1000 760" preserveAspectRatio="xMidYMax slice" aria-hidden="true"><defs><linearGradient id="rock" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#8996a5"/><stop offset=".38" stopColor="#263549"/><stop offset="1" stopColor="#050a12"/></linearGradient><linearGradient id="ice" x1="0" y1="0" x2=".8" y2="1"><stop stopColor="#e6eef2" stopOpacity=".86"/><stop offset=".45" stopColor="#66809b" stopOpacity=".55"/><stop offset="1" stopColor="#152235" stopOpacity=".12"/></linearGradient><pattern id="hatch" width="13" height="13" patternUnits="userSpaceOnUse" patternTransform="rotate(18)"><path d="M0 0V13" stroke="#dce7ed" strokeOpacity=".09" strokeWidth="2"/></pattern><filter id="blur"><feGaussianBlur stdDeviation="18"/></filter></defs><path d="M10 760L150 655 251 602 330 505 403 457 467 345 532 287 590 166 641 57 700 126 748 248 811 329 867 456 1000 592V760Z" fill="url(#rock)"/><path d="M641 57L590 166 532 287 467 345 403 457 330 505 251 602 150 655 10 760H500L595 619 650 471 679 294Z" fill="url(#ice)"/><path d="M641 57L700 126 748 248 811 329 867 456 1000 592V760H500L595 619 650 471 679 294Z" fill="#07101d" opacity=".82"/><path d="M641 57L679 294 650 471 595 619 500 760" fill="none" stroke="#b7c8d4" strokeOpacity=".18" strokeWidth="4"/><path d="M10 760L150 655 251 602 330 505 403 457 467 345 532 287 590 166 641 57 700 126 748 248 811 329 867 456 1000 592V760Z" fill="url(#hatch)"/><ellipse cx="720" cy="150" rx="260" ry="80" fill="#9bb9d0" opacity=".13" filter="url(#blur)"/></svg>
      <svg className="routeLine" viewBox="0 0 1000 760" preserveAspectRatio="none" aria-label="Route to the summit"><path className="routeShadow" d="M92 676C175 647 229 613 307 582S425 556 507 509 624 473 704 422 804 354 858 294"/><path className="routeFuture" d="M92 676C175 647 229 613 307 582S425 556 507 509 624 473 704 422 804 354 858 294"/><path className="routeDone" d="M92 676C175 647 229 613 307 582S425 556 507 509"/></svg>
      <nav className="campStops" aria-label="Expedition camps">{expeditionDays.map(day=><button key={day.day} type="button" className={`stop stop${day.day} ${day.status} ${selectedDay.day===day.day?"active":""}`} disabled={day.status==="locked"} onClick={()=>selectDay(day)}><i/><span><b>0{day.day}</b>{day.camp}<small>{day.altitude.toLocaleString()} M</small></span></button>)}</nav>
      <motion.button className="robotPin" type="button" aria-label="Pemba at Western Cwm" animate={{y:[0,-5,0]}} transition={{duration:2.8,repeat:Infinity,ease:"easeInOut"}} onClick={()=>activeTab==="diary"&&speak()}><span className="robotHead"><i/></span><span className="robotBody"/><span className="robotLegs"/></motion.button>
    </section>

    <AnimatePresence mode="wait">{activeTab==="diary"?<motion.aside key={`diary-${selectedDay.day}`} className={`contentPanel diaryPanel ${current?"livePanel":"pastPanel"}`} initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-20}}><div className="panelMeta"><span>{current?"TODAY · LIVE":"FIELD NOTE · ARCHIVED"}</span><b>DAY 0{selectedDay.day}</b></div><h2>{current?selectedDay.camp:selectedDay.diary?.title}</h2>{current?<><p className="lead">Today&apos;s entry is still being written. Pemba can tell you what it is thinking right now.</p><div className="plan"><span>SHERPA PLAN</span>{selectedDay.sherpaPlan.map((item,index)=><p key={item}><b>0{index+1}</b>{item}</p>)}</div><button className="primaryAction" onClick={speak}>{speaking?"STOP TRANSMISSION":"HEAR PEMBA'S THOUGHTS"}<span>{speaking?"■":"▶"}</span></button></>:<><p className="diaryText">{selectedDay.diary?.body}</p><blockquote><span>PROUD MOMENT</span>{selectedDay.diary?.proudMoment}</blockquote><button className="textAction" onClick={()=>selectDay(today)}>RETURN TO TODAY →</button></>}</motion.aside>:<motion.aside key="radio" className="contentPanel radioPanel" initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-20}}><MemoryRadio initialDay={today.day} embedded/></motion.aside>}</AnimatePresence>
    <footer className="progressFooter"><span>BASE CAMP</span><div><i style={{width:`${((selectedDay.day-1)/4)*100}%`}}/></div><strong>{selectedDay.altitude.toLocaleString()} M</strong><span>SUMMIT · 8,849 M</span></footer>
  </main>;
}
*/
