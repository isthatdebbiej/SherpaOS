"use client";

import { useRef, useState, type FormEvent } from "react";
import styles from "./MemoryRadio.module.css";

const API = process.env.NEXT_PUBLIC_SHERPA_API ?? "http://127.0.0.1:8000";
type Message = { from: "operator" | "pemba"; text: string };

export function MemoryRadio({ initialDay = 1, embedded = false }: { initialDay?: number; embedded?: boolean }) {
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { from: "pemba", text: "Base camp, this is Pemba. Ask about Days 1–4 or my latest live safety decision. Over." },
  ]);
  const [busy, setBusy] = useState(false);
  const [transmitting, setTransmitting] = useState(false);
  const [pembaSpeaking, setPembaSpeaking] = useState(false);
  const [voiceAudio, setVoiceAudio] = useState<HTMLAudioElement | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stream = useRef<MediaStream | null>(null);
  const pressed = useRef(false);

  function addAnswer(question: string, answer: string) {
    setMessages((items) => [...items, { from: "operator", text: question }, { from: "pemba", text: answer }]);
  }

  async function ask(blob: Blob) {
    setBusy(true);
    const form = new FormData();
    form.append("audio", blob, "radio.webm");
    try {
      const response = await fetch(`${API}/api/voice/ask/everest-001/${initialDay}`, { method: "POST", body: form });
      const value = await response.json();
      if (!response.ok) throw new Error(value.detail ?? "Radio inference failed");
      addAnswer(value.question, value.answer);
      setPembaSpeaking(true);
      const audio = new Audio(`data:${value.audio_mime};base64,${value.audio_base64}`);
      audio.onended = () => setPembaSpeaking(false);
      await audio.play();
    } catch (error) {
      const message = error instanceof Error ? error.message : "The radio inference service is unavailable.";
      setMessages((items) => [...items, { from: "pemba", text: `${message} No answer was fabricated.` }]);
      setPembaSpeaking(false);
    } finally {
      setBusy(false);
    }
  }

  async function askText(question: string) {
    setBusy(true);
    try {
      const url = new URL(`${API}/api/radio/ask-text/${initialDay}`);
      url.searchParams.set("question", question);
      const response = await fetch(url, { method: "POST" });
      const value = await response.json();
      if (!response.ok) throw new Error(value.detail ?? "Radio inference failed");
      addAnswer(question, value.answer);
    } catch (error) {
      const message = error instanceof Error ? error.message : "The radio inference service is unavailable.";
      setMessages((items) => [...items, { from: "operator", text: question }, { from: "pemba", text: `${message} No answer was fabricated.` }]);
    } finally {
      setBusy(false);
    }
  }

  async function begin() {
    if (busy) return;
    pressed.current = true;
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = media;
      if (!pressed.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      chunks.current = [];
      const next = new MediaRecorder(media);
      next.ondataavailable = (event) => { if (event.data.size) chunks.current.push(event.data); };
      next.onstop = () => {
        const payload = new Blob(chunks.current, { type: next.mimeType || "audio/webm" });
        if (payload.size) void ask(payload);
      };
      recorder.current = next;
      next.start();
      setTransmitting(true);
    } catch {
      pressed.current = false;
      setMessages((items) => [...items, { from: "pemba", text: "I need microphone permission to hear you." }]);
    }
  }

  function end() {
    pressed.current = false;
    setTransmitting(false);
    if (recorder.current?.state === "recording") recorder.current.stop();
    stream.current?.getTracks().forEach((track) => track.stop());
  }

  function send(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    void askText(text);
  }

  return <aside className={`${styles.panel} ${embedded ? styles.embedded : ""}`} data-radio-panel={embedded ? "embedded" : undefined} aria-label="Radio Pemba">
    <div className={styles.top}><div className={styles.title}><strong>RADIO PEMBA</strong><small>Day 0{initialDay}</small></div><div className={styles.onAir}><i /> HF INFERENCE</div></div>
    <div className={styles.simpleBody}>
      <div className={`${styles.companions} ${transmitting ? styles.humanSpeaking : ""} ${pembaSpeaking ? styles.robotSpeaking : ""} ${busy ? styles.thinking : ""}`}>
        <div className={styles.sherpa}><span className={styles.sherpaHead}><i /></span><span className={styles.sherpaBody}/><b>YOU</b></div>
        <div className={styles.radioSignal}><i/><i/><i/></div>
        <div className={styles.radioRobot}><span className={styles.radioRobotHead}><i/></span><span className={styles.radioRobotBody}/><span className={styles.radioRobotLegs}/><b>PEMBA</b></div>
      </div>
      <div className={styles.contextStrip}><span>VERIFIED MEMORY</span><b>DAYS 01–04</b><b>LATEST LIVE RECEIPT</b></div>
      <div className={styles.chatLog} aria-live="polite">{messages.map((message, index) => <div key={index} className={`${styles.chatBubble} ${message.from === "pemba" ? styles.pembaBubble : styles.operatorBubble}`}><small>{message.from === "pemba" ? "Pemba" : "You"}</small>{message.text}</div>)}</div>
      <form className={styles.composer} onSubmit={send}><input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Why did you stop?"/><button disabled={!draft.trim() || busy} aria-label="Send">↗</button></form>
      <div className={`${styles.wave} ${(busy || transmitting) ? styles.active : ""}`} aria-hidden="true"><i/><i/><i/><i/><i/><i/><i/></div>
      <button className={`${styles.ptt} ${transmitting ? styles.transmitting : ""}`} disabled={busy} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); void begin(); }} onPointerUp={end} onPointerCancel={end}>{busy ? "HF INFERENCE RUNNING…" : transmitting ? "RELEASE TO SEND" : "PRESS AND HOLD TO TALK"}</button>{voiceAudio && !pembaSpeaking && <button type="button" className={styles.ptt} onClick={() => { setPembaSpeaking(true); voiceAudio.currentTime = 0; void voiceAudio.play(); }}>PLAY PEMBA VOICE</button>}
    </div>
  </aside>;
}
