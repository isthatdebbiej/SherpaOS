import { expeditionDays } from "@/data/days";

export default function Home() {
  const current = expeditionDays.find((day) => day.status === "waiting") ?? expeditionDays[0];

  return (
    <main className="shell">
      <div className="stars" aria-hidden="true" />
      <header className="masthead">
        <div>
          <p className="eyebrow">Robot Everest · Expedition 001</p>
          <h1>Pemba&apos;s Field Journal</h1>
        </div>
        <div className="altitude">
          <strong>{current.altitude.toLocaleString()} m</strong>
          <span>Day {current.day} of 5</span>
        </div>
      </header>

      <section className="mountain" aria-label="Five-day Himalayan expedition trail">
        <div className="moon" />
        <div className="peak peakBack" />
        <div className="peak peakFront" />
        <svg className="trail" viewBox="0 0 1000 520" role="img" aria-label="Route from Base Camp to the summit">
          <path className="trailFuture" d="M110 430 C235 370 230 320 360 300 S510 260 565 220 S700 150 770 112 S850 72 900 42" />
          <path className="trailDone" d="M110 430 C235 370 230 320 360 300 S480 272 520 245" />
        </svg>

        <div className="pemba" aria-label="Pemba is waiting at Western Cwm">
          <span className="antenna" />
          <span className="head"><i /></span>
          <span className="body" />
          <span className="feet" />
          <span className="thought">waiting for today&apos;s memories…</span>
        </div>

        <nav className="camps" aria-label="Expedition days">
          {expeditionDays.map((day) => (
            <button key={day.day} className={`camp camp-${day.day} ${day.status}`} type="button" disabled={day.status === "locked"}>
              <span className="campDot" />
              <span className="campLabel">Day {day.day}<strong>{day.camp}</strong></span>
            </button>
          ))}
        </nav>

        <aside className="planCard">
          <p className="eyebrow">Today&apos;s Sherpa plan</p>
          <h2>{current.camp}</h2>
          <ul>{current.sherpaPlan.map((item) => <li key={item}>{item}</li>)}</ul>
          <button type="button">Receive today&apos;s memory <span>↗</span></button>
        </aside>
      </section>

      <footer className="dayRail">
        {expeditionDays.map((day) => <span key={day.day} className={day.status}><i /> Day {day.day}</span>)}
      </footer>
    </main>
  );
}
