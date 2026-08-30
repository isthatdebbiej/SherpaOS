"use client";

import { useRef, useState, type FormEvent } from "react";
import styles from "./MemoryRadio.module.css";

const API=process.env.NEXT_PUBLIC_SHERPA_API??"http://127.0.0.1:8000";
type Message={from:"operator"|"pemba";text:string};

export function MemoryRadio({initialDay=3,embedded=false}:{initialDay?:number;embedded?:boolean}){
  const [draft,setDraft]=useState(""),[messages,setMessages]=useState<Message[]>([{from:"pemba",text:"Base camp, this is Pemba. I can hear you."}]);
  const [busy,setBusy]=useState(false),[transmitting,setTransmitting]=useState(false),[pembaSpeaking,setPembaSpeaking]=useState(false);
  const recorder=useRef<MediaRecorder|null>(null),chunks=useRef<Blob[]>([]),stream=useRef<MediaStream|null>(null);
  async function ask(blob:Blob){setBusy(true);if(process.env.NEXT_PUBLIC_SHERPA_VOICE_MODE!=="api"){await new Promise(resolve=>setTimeout(resolve,850));setBusy(false);setPembaSpeaking(true);setMessages(items=>[...items,{from:"operator",text:"How did the route feel today?"},{from:"pemba",text:"The route demanded patience, but the verified day summary shows a stable finish and a safe return. Over."}]);setTimeout(()=>setPembaSpeaking(false),3200);return}const form=new FormData();form.append("audio",blob,"radio.webm");try{const response=await fetch(`${API}/api/voice/ask/everest-001/${initialDay}`,{method:"POST",body:form});const value=await response.json();if(!response.ok)throw new Error();setMessages(items=>[...items,{from:"operator",text:value.question},{from:"pemba",text:value.answer}]);setPembaSpeaking(true);const audio=new Audio(`data:${value.audio_mime};base64,${value.audio_base64}`);audio.onended=()=>setPembaSpeaking(false);await audio.play()}catch{setMessages(items=>[...items,{from:"pemba",text:"The signal faded. Please call me again."}]);setPembaSpeaking(false)}finally{setBusy(false)}}
  async function begin(){if(busy)return;try{stream.current=await navigator.mediaDevices.getUserMedia({audio:true});chunks.current=[];const next=new MediaRecorder(stream.current);next.ondataavailable=e=>{if(e.data.size)chunks.current.push(e.data)};next.onstop=()=>void ask(new Blob(chunks.current,{type:next.mimeType||"audio/webm"}));recorder.current=next;next.start();setTransmitting(true)}catch{setMessages(items=>[...items,{from:"pemba",text:"I need microphone permission to hear you."}])}}
  function end(){if(!transmitting)return;setTransmitting(false);recorder.current?.stop();stream.current?.getTracks().forEach(track=>track.stop())}
  function send(event:FormEvent){event.preventDefault();const text=draft.trim();if(!text)return;setMessages(items=>[...items,{from:"operator",text},{from:"pemba",text:"I heard you. Ask me by voice and I’ll answer from today’s memories."}]);setDraft("")}
  return <aside className={`${styles.panel} ${embedded?styles.embedded:""}`} data-radio-panel={embedded ? "embedded" : undefined} aria-label="Radio Pemba">
    <div className={styles.top}><div className={styles.title}><strong>RADIO PEMBA</strong><small>Day 0{initialDay}</small></div><div className={styles.onAir}><i/> ON AIR</div></div>
    <div className={styles.simpleBody}>
      <div className={`${styles.companions} ${transmitting?styles.humanSpeaking:""} ${pembaSpeaking?styles.robotSpeaking:""} ${busy?styles.thinking:""}`} aria-label={transmitting?"You are speaking; Pemba is listening":pembaSpeaking?"Pemba is speaking; the Sherpa is listening":"Radio companions waiting"}>
        <div className={styles.sherpa}><span className={styles.sherpaHead}><i/></span><span className={styles.sherpaBody}/><b>YOU</b></div>
        <div className={styles.radioSignal}><i/><i/><i/></div>
        <div className={styles.radioRobot}><span className={styles.radioRobotHead}><i/></span><span className={styles.radioRobotBody}/><span className={styles.radioRobotLegs}/><b>PEMBA</b></div>
      </div>
      <div className={styles.contextStrip}><span>MEMORY CONTEXT</span><b>DAY 0{initialDay} SUMMARY</b><b>3 RECENT EPISODES</b></div>
      <div className={styles.chatLog} aria-live="polite">{messages.map((message,index)=><div key={index} className={`${styles.chatBubble} ${message.from==="pemba"?styles.pembaBubble:styles.operatorBubble}`}><small>{message.from==="pemba"?"Pemba":"You"}</small>{message.text}</div>)}</div>
      <form className={styles.composer} onSubmit={send}><input value={draft} onChange={e=>setDraft(e.target.value)} placeholder="Ask Pemba something…"/><button disabled={!draft.trim()} aria-label="Send">↗</button></form>
      <div className={`${styles.wave} ${(busy||transmitting)?styles.active:""}`} aria-hidden="true"><i/><i/><i/><i/><i/><i/><i/></div>
      <button className={`${styles.ptt} ${transmitting?styles.transmitting:""}`} disabled={busy} onPointerDown={()=>void begin()} onPointerUp={end} onPointerCancel={end}>{busy?"PEMBA IS THINKING…":transmitting?"RELEASE TO SEND":"HOLD TO TALK"}</button>
    </div>
  </aside>
}
