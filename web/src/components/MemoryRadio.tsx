"use client";

import { useRef, useState } from "react";
import styles from "./MemoryRadio.module.css";

const API = process.env.NEXT_PUBLIC_SHERPA_API ?? "http://127.0.0.1:8000";

export function MemoryRadio({ initialDay = 1, embedded = false }: { initialDay?: number; embedded?: boolean }) {
  const [busy, setBusy] = useState(false);
  const [transmitting, setTransmitting] = useState(false);
  const [pembaSpeaking, setPembaSpeaking] = useState(false);
  const [failed, setFailed] = useState(false);
  const [voiceAudio, setVoiceAudio] = useState<HTMLAudioElement | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const stream = useRef<MediaStream | null>(null);
  const pressed = useRef(false);

  async function ask(blob: Blob) {
    setBusy(true);
    setFailed(false);
    const form = new FormData();
    form.append("audio", blob, "radio.webm");
    try {
      const response = await fetch(`${API}/api/voice/ask/everest-001/${initialDay}`, {
        method: "POST",
        body: form,
      });
      const value = await response.json();
      if (!response.ok || !value.audio_base64) throw new Error("Radio inference failed");
      const audio = new Audio(`data:${value.audio_mime};base64,${value.audio_base64}`);
      audio.onended = () => setPembaSpeaking(false);
      audio.onerror = () => { setPembaSpeaking(false); setFailed(true); };
      setVoiceAudio(audio);
      setPembaSpeaking(true);
      try {
        await audio.play();
      } catch {
        setPembaSpeaking(false);
      }
    } catch {
      setFailed(true);
      setPembaSpeaking(false);
    } finally {
      setBusy(false);
    }
  }

  async function begin() {
    if (busy) return;
    pressed.current = true;
    setFailed(false);
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
      setFailed(true);
    }
  }

  function end() {
    pressed.current = false;
    setTransmitting(false);
    if (recorder.current?.state === "recording") recorder.current.stop();
    stream.current?.getTracks().forEach((track) => track.stop());
  }

  function replay() {
    if (!voiceAudio) return;
    setPembaSpeaking(true);
    voiceAudio.currentTime = 0;
    void voiceAudio.play().catch(() => {
      setPembaSpeaking(false);
      setFailed(true);
    });
  }

  const label = busy
    ? "PEMBA IS THINKING…"
    : transmitting
      ? "RELEASE TO SEND"
      : failed
        ? "RADIO FAILED — PRESS TO RETRY"
        : "PRESS AND HOLD TO TALK";

  return <aside className={`${styles.panel} ${embedded ? styles.embedded : ""}`} data-radio-panel={embedded ? "embedded" : undefined} aria-label="Voice-only Radio Pemba">
    <div className={styles.top}><div className={styles.title}><strong>RADIO PEMBA</strong><small>Day 0{initialDay}</small></div><div className={styles.onAir}><i /> ON AIR</div></div>
    <div className={styles.simpleBody}>
      <div className={`${styles.companions} ${transmitting ? styles.humanSpeaking : ""} ${pembaSpeaking ? styles.robotSpeaking : ""} ${busy ? styles.thinking : ""}`} aria-live="polite" aria-label={transmitting ? "Operator speaking" : busy ? "Pemba thinking" : pembaSpeaking ? "Pemba speaking" : "Radio ready"}>
        <div className={styles.sherpa}><span className={styles.sherpaHead}><i /></span><span className={styles.sherpaBody}/><b>YOU</b></div>
        <div className={styles.radioSignal}><i/><i/><i/></div>
        <div className={styles.radioRobot}><span className={styles.radioRobotHead}><i/></span><span className={styles.radioRobotBody}/><span className={styles.radioRobotLegs}/><b>PEMBA</b></div>
      </div>
      <div className={`${styles.wave} ${(busy || transmitting || pembaSpeaking) ? styles.active : ""}`} aria-hidden="true"><i/><i/><i/><i/><i/><i/><i/></div>
      <button className={`${styles.ptt} ${transmitting ? styles.transmitting : ""}`} disabled={busy || pembaSpeaking} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); void begin(); }} onPointerUp={end} onPointerCancel={end}>{label}</button>
      {voiceAudio && !pembaSpeaking && <button type="button" className={styles.ptt} onClick={replay}>PLAY PEMBA VOICE</button>}
    </div>
  </aside>;
}
