"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./MemoryRadio.module.css";

type Manifest = {
  day: number; status: string; original_filename: string; bag_sha256: string;
  duration_seconds: number | null; message_count: number;
  topics: Array<{name:string; message_count:number; approved:boolean; role:string|null}>;
};

const API = process.env.NEXT_PUBLIC_SHERPA_API ?? "http://127.0.0.1:8000";

export function MemoryRadio({initialDay=3}:{initialDay?:number}) {
  const [open,setOpen]=useState(false),[day,setDay]=useState(initialDay);
  const [manifest,setManifest]=useState<Manifest|null>(null),[file,setFile]=useState<File|null>(null);
  const [status,setStatus]=useState("No verified memory loaded"),[busy,setBusy]=useState(false);
  const [transmitting,setTransmitting]=useState(false),[question,setQuestion]=useState("");
  const [answer,setAnswer]=useState("");
  const recorder=useRef<MediaRecorder|null>(null),chunks=useRef<Blob[]>([]),stream=useRef<MediaStream|null>(null);

  useEffect(()=>setDay(initialDay),[initialDay]);
  useEffect(()=>{void loadDay(day);return()=>stream.current?.getTracks().forEach(track=>track.stop());},[day]);
  async function loadDay(selected:number){setManifest(null);setStatus("Checking expedition memory…");try{const response=await fetch(`${API}/api/expeditions/everest-001/days/${selected}`);if(response.status===404){setStatus("No verified bag for this day");return}if(!response.ok)throw new Error("Memory API unavailable");const value=await response.json() as Manifest;setManifest(value);setStatus("Channel clear · hold to ask")}catch(error){setStatus(error instanceof Error?error.message:"Memory API unavailable")}}
  async function upload(){if(!file)return;setBusy(true);setStatus("Receiving and verifying real ROS bag…");const form=new FormData();form.append("bag",file);try{const response=await fetch(`${API}/api/expeditions/everest-001/days/${day}/upload`,{method:"POST",body:form});const value=await response.json();if(!response.ok)throw new Error(value.detail??"Upload failed");setManifest(value as Manifest);setStatus("Memory verified · channel clear")}catch(error){setStatus(error instanceof Error?error.message:"Upload failed")}finally{setBusy(false)}}
  async function begin(){if(!manifest||busy)return;try{stream.current=await navigator.mediaDevices.getUserMedia({audio:true});chunks.current=[];const next=new MediaRecorder(stream.current);next.ondataavailable=event=>{if(event.data.size)chunks.current.push(event.data)};next.onstop=()=>void ask(new Blob(chunks.current,{type:next.mimeType||"audio/webm"}));recorder.current=next;next.start();setTransmitting(true);setStatus("Operator transmitting · release to send")}catch{setStatus("Microphone permission is required")}}
  function end(){if(!transmitting)return;setTransmitting(false);recorder.current?.stop();stream.current?.getTracks().forEach(track=>track.stop());setStatus("Transmission received · consulting bag evidence")}
  async function ask(blob:Blob){setBusy(true);const form=new FormData();form.append("audio",blob,"radio.webm");try{const response=await fetch(`${API}/api/voice/ask/everest-001/${day}`,{method:"POST",body:form});const value=await response.json();if(!response.ok)throw new Error(value.detail??"Voice request failed");setQuestion(value.question);setAnswer(value.answer);setStatus("Pemba transmitting");const audio=new Audio(`data:${value.audio_mime};base64,${value.audio_base64}`);audio.onended=()=>setStatus("Channel clear · hold to ask");await audio.play()}catch(error){setStatus(error instanceof Error?error.message:"Voice request failed")}finally{setBusy(false)}}
  if(!open)return <button className={styles.launcher} onClick={()=>setOpen(true)}>◉ Call Pemba</button>;
  return <aside className={styles.panel} aria-label="Pemba field radio">
    <div className={styles.top}><div className={styles.title}><strong>Channel 01 · Pemba</strong><small>Real ROS bag evidence</small></div><button className={styles.close} onClick={()=>setOpen(false)} aria-label="Close radio">×</button></div>
    <div className={styles.body}>
      <label className={styles.label} htmlFor="radio-day">Active expedition memory</label><select id="radio-day" className={styles.daySelect} value={day} onChange={event=>setDay(Number(event.target.value))}>{[1,2,3,4,5].map(value=><option key={value} value={value}>Day {value}</option>)}</select>
      {manifest?<div className={styles.memory}><strong>Day {day} memory verified</strong><p>{manifest.original_filename} · {manifest.message_count.toLocaleString()} messages · {manifest.topics.filter(topic=>topic.approved).length} approved topics</p><code>SHA-256 {manifest.bag_sha256.slice(0,16)}…</code><div className={styles.constellation} aria-label="Topics found in the real recording">{manifest.topics.slice(0,12).map(topic=><span className={`${styles.topic} ${topic.approved?"":styles.topicBlocked}`} key={topic.name} title={`${topic.message_count.toLocaleString()} messages · ${topic.approved?topic.role:"not exposed to voice"}`}><i/>{topic.name.split("/").filter(Boolean).at(-1)}</span>)}</div></div>:<div className={styles.memory}><strong>Add this day’s robot memory</strong><p>The bag stays on the SherpaOS base station. Answers remain locked until verification succeeds.</p><label className={styles.upload}><input type="file" accept=".mcap,.db3" onChange={event=>setFile(event.target.files?.[0]??null)}/></label><button className={styles.uploadButton} disabled={!file||busy} onClick={()=>void upload()}>Upload and verify real bag</button></div>}
      <div className={styles.status} aria-live="polite">{status}</div><div className={`${styles.wave} ${(busy||transmitting)?styles.active:""}`} aria-hidden="true"><i/><i/><i/><i/><i/><i/><i/></div>
      <button className={`${styles.ptt} ${transmitting?styles.transmitting:""}`} disabled={!manifest||busy} onPointerDown={()=>void begin()} onPointerUp={end} onPointerCancel={end}>{transmitting?"Transmitting…":"Hold to talk"}</button>
      {(question||answer)&&<div className={styles.conversation}>{question&&<div className={styles.burst}><small>Operator</small>{question}</div>}{answer&&<div className={`${styles.burst} ${styles.answer}`}><small>Pemba · grounded in Day {day}</small>{answer}</div>}</div>}
    </div>
  </aside>;
}
