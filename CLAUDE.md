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
- [x] **Phase 1 — Historical data pipeline**: 161 races 2019–2026 in parquet, schema tests,
      lap classifier (golden counts on 2 fixture races), computed `tracks.yaml`.
- [x] **Phase 2 — Features + normalisation**: reference pace with golden tests, feature matrix,
      two explicit leakage tests (lap data + weather boundaries).
- [x] **Phase 3 — Models**: LightGBM quantile ×3, split-conformal calibration, versioned
      artifacts, degradation priors, online bias (readout only — see Decisions), monotonicity
      enforced at inference.
- [x] **Phase 4 — Replay harness + backtest**: replay through the live engine; walk-forward
      backtest over 57 held-out races (2024–2026, retrain-per-3): clean-air next-lap MAE
      **0.489s**, +58.5% vs spec persistence (gate ≥20%), +10.6% vs rolling-median(5)
      (gate ≥8%), coverage 73.4% (gate 80±7) — ALL GATES PASS
      (`reports/backtest_2026-07-11.md`). Wet compounds remain the weak regime (1.58s MAE),
      per the non-goals.
- [x] **Phase 5 — Live service**: FastAPI WS `/ws/live` (+resume via `?since=`), REST incl.
      `/api/history` + `/api/simulate`; OpenF1 client (MQTT + REST fallback, `_id`/`_key`
      MessageLog) with mapper fixture-tested against real API samples.
- [x] **Phase 6 — Frontend**: live board / driver view / strategy lab / replay banner;
      Lighthouse performance 99 (`reports/lighthouse.md`); screenshots in `reports/ui/`.
- [x] **Phase 7 — Strategy simulator**: shared-random-environment Monte Carlo; 36 strategies
      × 2000 sims in ~0.9s; determinism + physics + fixture sanity tests.
- [x] **Phase 8 — Deploy + ops**: backend/Dockerfile, fly.toml, retrain.yml (MAE regression
      gate <5%), README runbook.

## Decisions

- 2026-07-10: Repo root is `pitwall/` (pre-existing GitHub repo `karacayir/pitwall`), not
  `f1-pitwall/` as in the brief — origin already existed, kept as-is.
- 2026-07-10: `models/*/` artifacts are gitignored (binary boosters bloat git); retrain workflow
  publishes them as CI artifacts instead.
- 2026-07-10: `gh` CLI not installed; remote was already configured so not needed for Phase 0.
- 2026-07-10: results.parquet added as a 4th consolidated file (grid/finish/status/team colours);
  the brief listed three but also asked for results to be persisted.
- 2026-07-10: Sprint races are NOT ingested (GP races only) — simpler stint semantics; revisit
  if more current-season data is needed.
- 2026-07-10: FastF1 rate-limits at 500 API calls/hour; `make data` is resumable per race and
  must be re-run (or looped) until complete. ~35 races per hour.
- 2026-07-11: LightGBM's quantile objective REJECTS monotone_constraints (hard error).
  Monotonicity in tyre_age is enforced at inference: isotonic cummax along age sweeps in
  `PaceModel.predict_quantiles(age_groups=)`.
- 2026-07-11: fastf1 renamed circuit locations across seasons (monte_carlo/monaco,
  miami_gardens/miami, marina_bay/singapore) — canonicalised in `ingest.fastf1_pull.CANON_TRACKS`.
- 2026-07-11: raw LightGBM quantile intervals undercover (64%); split-conformal calibration
  factors (s_lo/s_hi in meta.json) fitted on the validation fold restore ~80%.
- 2026-07-11: **Backtest #1 failed** — persistence beat the curve-only model. Added
  autoregressive features (`last_green_ratio`, `rolling_ratio_3`, strictly laps < L);
  validation MAE 0.73 → 0.50s.
- 2026-07-11: with AR features, adding the online bias to forecast quantiles double-adapts and
  hurts (A/B: +0.07s MAE, −8pts coverage). Bias is now a UI calibration readout + sim input
  only, NOT added to quantiles (deviation from brief, measured).
- 2026-07-11: persistence gate baseline is the spec-literal "last lap" (any class); the harder
  last-GREEN-lap variant is also reported as a diagnostic.
- 2026-07-11: sim positions must be computed from lap-synchronised clocks (cars sit on different
  laps); `to_sim_setup` projects every car to the chosen driver's current lap.

## Environment gotchas

- Repo path contains a space (`Tech Stuff`) — quote paths in scripts.
- System python is 3.9; always go through `uv` (backend/.python-version pins 3.12).
- Env vars: `OPENF1_USERNAME`, `OPENF1_PASSWORD`, `DATA_SOURCE=replay|live`, `REPLAY_RACE`,
  `REPLAY_SPEED`, `MODEL_DIR`, `NEXT_PUBLIC_API_BASE`, `NEXT_PUBLIC_WS_URL`.
