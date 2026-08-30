"use client";

import { useEffect, useMemo, useState } from "react";
import "./LiveSafetyDemo.css";

type Report={guard:string;score:number;confidence:number;reason_codes:string[]};
type Event={decision:{decision_id:string;action:"PASS"|"LIMIT_SPEED"|"REQUEST_HOLD";score:number;confidence:number;timestamp:number;model_version?:string;guard_reports:Report[]};receipt:{accepted:boolean;applied_action:string;acknowledgement_source:string};requested_velocity_mps:number;applied_velocity_mps:number};
const API=process.env.NEXT_PUBLIC_SHERPA_API??"http://127.0.0.1:8000";
const WS=API.replace(/^http/,"ws");
const label={PASS:"GO",LIMIT_SPEED:"CAUTION",REQUEST_HOLD:"NO-GO"};

export function LiveSafetyDemo(){
 const [event,setEvent]=useState<Event|null>(null);
 const [connected,setConnected]=useState(false);
 const [events,setEvents]=useState<Event[]>([]);
 useEffect(()=>{const socket=new WebSocket(`${WS}/ws/supervisor`);socket.onopen=()=>setConnected(true);socket.onclose=()=>setConnected(false);socket.onmessage=message=>{const value=JSON.parse(message.data) as Event;setEvent(value);setEvents(items=>[value,...items].slice(0,8))};return()=>socket.close()},[]);
 const guards=useMemo(()=>event?.decision.guard_reports??[],[event]);
 const verdict=event?label[event.decision.action]:"NO LIVE EVIDENCE";
 return <section className="liveDemo" aria-label="Live SherpaOS safety supervisor">
  <header><div><span>VULTR · LIVE SUPERVISOR</span><h1>Robot safety decision</h1><p>Video and decisions are independent streams. Every verdict requires a matching actuation receipt.</p></div><b className={connected?"connected":"offline"}>{connected?"STREAM CONNECTED":"RUNTIME OFFLINE"}</b></header>
  <div className="liveGrid">
   <article className="robotFeed"><div className="feedLabel">RAW ROBOT STREAM · NO DECISION OVERLAY</div><img src={`${API}/stream/robot`} alt="Raw live MuJoCo G1 stream" onError={e=>{e.currentTarget.style.display="none"}}/><div className="feedEmpty">Waiting for Vultr robot stream</div></article>
   <aside className={`verdict ${event?.decision.action.toLowerCase()??"unknown"}`}><small>SHERPAOS POLICY</small><strong>{verdict}</strong>{event&&<><p>Requested <b>{event.requested_velocity_mps.toFixed(2)} m/s</b> · Applied <b>{event.applied_velocity_mps.toFixed(2)} m/s</b></p><dl><div><dt>Decision ID</dt><dd>{event.decision.decision_id}</dd></div><div><dt>Receipt</dt><dd>{event.receipt.accepted?"ACCEPTED":"REJECTED"}</dd></div><div><dt>Model</dt><dd>{event.decision.model_version??"deterministic guards"}</dd></div></dl></>}</aside>
  </div>
  <div className="guardGrid">{guards.map(report=><article key={report.guard}><span>{report.guard.replaceAll("_"," ")}</span><b>{Math.round(report.score*100)}%</b><meter min="0" max="1" value={report.score}/><small>{report.reason_codes.join(" · ")}</small></article>)}</div>
  <section className="evidenceLog"><h2>Immutable decision evidence</h2>{events.length?events.map(item=><div key={item.decision.decision_id}><time>{item.decision.timestamp.toFixed(2)}s</time><b>{label[item.decision.action]}</b><code>{item.decision.decision_id}</code><span>{item.requested_velocity_mps.toFixed(2)} → {item.applied_velocity_mps.toFixed(2)} m/s</span><i>{item.receipt.accepted?"RECEIPT MATCHED":"NO RECEIPT"}</i></div>):<p>No decision has been received. The UI will not manufacture demo data.</p>}</section>
 </section>
}
