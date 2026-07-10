# Pitwall — CLAUDE.md

Real-time F1 lap-time forecasting app: per-compound lap forecasts with uncertainty bands during
live races, live timing UI, and a Monte-Carlo strategy simulator. Personal, non-commercial —
never brand as official F1.

## Stack (pinned)

- **Backend** (`backend/`): Python 3.12 via `uv`, FastAPI + uvicorn, polars (pandas only where
  FastF1 requires it), lightgbm, numpy, scipy, fastf1, httpx, paho-mqtt, pydantic v2, pytest, ruff.
- **Frontend** (`frontend/`): Next.js App Router + TypeScript strict, Tailwind, Zustand,
  uPlot (streaming charts), Recharts (simulator distributions), Framer Motion.
- **Storage**: parquet (training data, `data/`), SQLite (live event log), in-process state. No DB servers.
- **Deploy**: Docker; backend Fly.io, frontend Cloudflare Pages, GitHub Actions CI + weekly retrain.

## Commands (run from repo root)

| Command | What it does |
|---|---|
| `make setup` | `uv sync` backend + `npm install` frontend |
| `make test` | backend pytest |
| `make lint` | ruff check + format check |
| `make data` | FastF1 historical pull → `data/*.parquet` |
| `make train` | train LightGBM quantile models → `models/YYYYMMDD/` |
| `make backtest` | walk-forward replay evaluation → `reports/` |
| `make replay RACE=2025_monza SPEED=60` | replay a historical race through the live pipeline |
| `make dev` | run backend (uvicorn) + frontend (next dev) |

## Conventions

- Commits: conventional prefixes (`feat(ingest): …`, `test(features): …`, `fix(sim): …`);
  one logical change per commit; push after every commit; `phase-N: <summary>` closes a phase.
- Never commit: credentials, `data/`, FastF1 cache, trained artifacts (`models/*/`).
- All magic numbers (fuel constants, thresholds, priors) live in `backend/app/config.py`,
  commented as tunable approximations.
- Never invent API fields: probe the real API, save raw samples to `backend/tests/fixtures/`,
  parse against fixtures. If reality differs from the brief, trust reality and note it here.
- Leakage rule: any feature or reference value at lap L uses only laps < L (tested explicitly).
- `make test` and `make lint` must be green before a phase is declared done.

## Phase checklist

- [x] **Phase 0 — Scaffold**: git+origin, gitignore, Makefile, backend uv project, frontend
      Next.js scaffold, CI (ruff + pytest + frontend build), CLAUDE.md.
- [ ] **Phase 1 — Historical data pipeline**: FastF1 pull (≥120 races 2019–2026) → parquet with
      tested schema; lap classifier; computed `tracks.yaml` fields (pit_loss_s, sc_hazard_per_lap).
- [ ] **Phase 2 — Features + normalisation**: reference pace (golden tests incl. cold start +
      rain hold), feature matrix builder, explicit leakage test.
- [ ] **Phase 3 — Models**: LightGBM quantile ×3 (monotone in tyre_age), versioned artifacts,
      degradation priors per (track, compound), online bias module.
- [ ] **Phase 4 — Replay harness + backtest**: chronological replay through the live code path;
      backtest report meeting acceptance gates (≥20% vs persistence, ≥8% vs rolling-median,
      coverage 80±7).
- [ ] **Phase 5 — Live service**: FastAPI WS `/ws/live` + REST; OpenF1 MQTT client (token refresh,
      reconnect, `_key` dedupe, `_id` ordering) + REST fallback; `DATA_SOURCE=replay|live`.
- [ ] **Phase 6 — Frontend**: live board, driver view, strategy lab, replay banner;
      Lighthouse ≥85; screenshots in `reports/ui/`.
- [ ] **Phase 7 — Strategy simulator**: vectorised Monte Carlo, 2000 sims <2s, seeded
      determinism, sanity check vs a fixture race.
- [ ] **Phase 8 — Deploy + ops**: Dockerfile, fly.toml, CF Pages, retrain.yml with regression
      gate, README runbook.

## Decisions

- 2026-07-10: Repo root is `pitwall/` (pre-existing GitHub repo `karacayir/pitwall`), not
  `f1-pitwall/` as in the brief — origin already existed, kept as-is.
- 2026-07-10: `models/*/` artifacts are gitignored (binary boosters bloat git); retrain workflow
  publishes them as CI artifacts instead.
- 2026-07-10: `gh` CLI not installed; remote was already configured so not needed for Phase 0.

## Environment gotchas

- Repo path contains a space (`Tech Stuff`) — quote paths in scripts.
- System python is 3.9; always go through `uv` (backend/.python-version pins 3.12).
- Env vars: `OPENF1_USERNAME`, `OPENF1_PASSWORD`, `DATA_SOURCE=replay|live`, `REPLAY_RACE`,
  `REPLAY_SPEED`, `MODEL_DIR`, `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_WS_URL`.
