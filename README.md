# Pitwall

Real-time Formula 1 lap-time forecasting: during live races, Pitwall predicts each driver's
next laps for every tyre compound (with uncertainty bands), streams live timing to a
"midnight pit wall" UI, and Monte-Carlo-simulates the rest of the race to rank strategies.

**Unofficial personal project — not associated with Formula 1, the FIA, or any team.**

## How it works

- **Historical data** (FastF1, 2019→today, 161+ races) trains three LightGBM quantile models
  (P10/P50/P90) on a normalised target: `lap_time / reference_pace`, where the reference is a
  causal, fuel-corrected, clean-air rolling median — so tyre/driver/weather effects transfer
  across circuits and seasons. Split-conformal calibration widens the intervals to honest coverage.
- **Live** (OpenF1 MQTT/REST), every completed lap flows through the same `RaceEngine` that the
  replay and backtest use: reference update → feature build → quantile forecast → per-compound
  counterfactuals ("pace if they pitted now") → WebSocket broadcast.
- **Strategy lab** vectorises 2,000 race-ending simulations per strategy (safety cars, traffic,
  degradation curves shrunk to track priors) and ranks pit plans by expected finish position.

## Local dev

```bash
make setup            # uv sync (backend) + npm install (frontend)
make test             # backend pytest (~80 tests)
make lint             # ruff check + format check
make data             # pull historical races -> data/*.parquet (hours; resumable;
                      # FastF1 rate-limits at 500 calls/h — rerun until complete)
cd backend && uv run python -m ingest.tracks_build   # regenerate tracks.yaml
make train            # train models -> models/YYYYMMDD/
make backtest         # walk-forward evaluation -> reports/backtest_<date>.md
make replay RACE=2025_monza SPEED=60                 # headless replay (no frontend)
make dev              # backend :8000 (replay mode) + frontend :3000
```

Replay mode is the default dev loop: `DATA_SOURCE=replay REPLAY_RACE=2025_monza REPLAY_SPEED=25`
streams a historical race through the exact live code path.

Smoke-test the WebSocket contract against a running backend:

```bash
cd backend && uv run python scripts/ws_smoke.py --seconds 20
```

## Race-day live checklist

1. `fly secrets set OPENF1_USERNAME=... OPENF1_PASSWORD=...` (paid OpenF1 account).
2. Ensure the latest model artifact is on the `pitwall_models` volume (`models/YYYYMMDD/`).
3. Deploy with `DATA_SOURCE=live` (default in fly.toml). The backend polls `/v1/sessions`
   until the race session appears, then prefers MQTT (`mqtt.openf1.org:8883`, topics
   `v1/<endpoint>`) and falls back to REST polling every 4s automatically.
   *NOTE: the MQTT auth handshake needs a paid account and could not be exercised in dev —
   verify on first race day; the REST fallback is fixture-tested.*
4. Watch `/api/health` — it reports the attached session and current lap.
5. Frontend: `NEXT_PUBLIC_API_BASE` / `NEXT_PUBLIC_WS_URL` must point at the Fly app.

## Deploy

- **Backend** (Fly.io, persistent process for MQTT):
  `fly deploy` (uses `fly.toml` + `backend/Dockerfile`).
- **Frontend** (Cloudflare Pages, static export): connect the repo in the Pages dashboard with
  root directory `frontend`, build command `npm run build`, output directory `out`. Production
  branch `main` deploys to `<project>.pages.dev`; every PR gets its own preview URL. Set
  `NEXT_PUBLIC_API_BASE=https://<backend-host>` and `NEXT_PUBLIC_WS_URL=wss://<backend-host>/ws/live`
  as Pages build env vars once a backend is deployed (they are baked in at build time).
- **Weekly retrain**: `.github/workflows/retrain.yml` (Sundays 23:00 UTC) pulls the weekend's
  race, retrains, runs the backtest gate, and publishes the artifact only if clean-air MAE
  regressed by less than 5%.

## Repo map

`backend/` FastAPI + engine + models + sim (see `CLAUDE.md` for conventions) ·
`frontend/` Next.js app · `reports/` backtest reports + UI screenshots ·
`models/` trained artifacts (gitignored) · `data/` parquet + FastF1 cache (gitignored).
