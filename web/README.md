# Pemba's Field Journal

Interactive five-day Robot Everest diary experience. It presents an original animated
Everest relief with a central ascent route, clickable camps, written past-day entries,
spoken current-day thoughts, locked future days, and playful Pemba interactions.

The current UX milestone runs from `src/data/days.ts`. The repository already has a
Python expedition-memory API, but this redesigned page is not connected to it yet.

## Run locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

Production check:

```bash
npm run build
```

## Deployment

Vultr is the only deployment target. The intended production layout places this Next.js
server behind nginx/Caddy on the same host as `sherpaos.expedition.api`. The reverse
proxy should route `/api/` to the Python service and other requests to Next.js.

`OPENAI_API_KEY` is server-only and must never use a `NEXT_PUBLIC_` prefix. The current
browser voice is a temporary UX fixture; generated narration will be produced by the
backend and persisted beside the day's diary artifact.

## Current behavior

- Days 1–2: open persisted-looking mock diary pages.
- Day 3: current-day plan and temporary browser-spoken thoughts.
- Days 4–5: locked until their data exists.
- Mountain: code-drawn SVG/CSS relief with route animation and no copied site assets.
