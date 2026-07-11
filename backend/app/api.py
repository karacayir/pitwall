"""FastAPI app: WebSocket /ws/live + REST /api/*.

DATA_SOURCE=replay streams a historical race through the RaceEngine at
REPLAY_SPEED; DATA_SOURCE=live attaches the OpenF1 client. Both feed the same
SessionRuntime, which fans completed-lap snapshots out to WebSocket clients.
"""

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import config, schemas
from app.state import RaceEngine
from ingest.replay import event_stream, load_session_meta, resolve_race
from models.predict import PaceModel

log = logging.getLogger("pitwall.api")


class SessionRuntime:
    """Holds the engine, the latest snapshot, a sequence counter and the
    WebSocket fan-out queues. One instance per running session."""

    def __init__(self, engine: RaceEngine, data_source: str, replay_speed: float | None = None):
        self.engine = engine
        self.data_source = data_source
        self.replay_speed = replay_speed
        self.latest: schemas.LapUpdate | None = None
        self.seq = 0
        self.buffer: list[tuple[int, str]] = []  # (seq, json) ring buffer for resume
        self.buffer_size = 512
        self.queues: set[asyncio.Queue] = set()
        self.finished = False

    def publish(self, update: schemas.LapUpdate) -> None:
        self.latest = update
        self.seq += 1
        payload = update.model_dump_json()
        msg = json.loads(payload)
        msg["_id"] = self.seq
        text = json.dumps(msg)
        self.buffer.append((self.seq, text))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        for q in list(self.queues):
            q.put_nowait(text)

    def subscribe(self, since: int | None = None) -> tuple[asyncio.Queue, list[str]]:
        q: asyncio.Queue = asyncio.Queue()
        self.queues.add(q)
        backlog = [text for seq, text in self.buffer if since is None or seq > since]
        if since is None and backlog:
            backlog = backlog[-1:]  # new client: just the latest snapshot
        return q, backlog

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)


async def replay_task(runtime: SessionRuntime, race_dir: Path, speed: float) -> None:
    """Async twin of ingest.replay.run_replay feeding the runtime."""
    log.info("replay starting: %s at %sx", race_dir.name, speed)
    prev_t: float | None = None
    started = False  # fast-forward the pre-race dead air (weather-only feed)
    for t, kind, payload in event_stream(race_dir):
        started = started or kind == "lap"
        if started and speed > 0 and prev_t is not None and t > prev_t:
            await asyncio.sleep(min((t - prev_t) / speed, 60.0))
        if started:
            prev_t = t
        if kind == "weather":
            runtime.engine.on_weather(payload)
            continue
        update = runtime.engine.on_lap(payload)
        if update is not None:
            runtime.publish(update)
        await asyncio.sleep(0)  # let WS writers run at speed<=0
    runtime.finished = True
    log.info("replay finished: %s", race_dir.name)


def create_app(
    race_dir: Path | None = None,
    speed: float | None = None,
    model: PaceModel | None | str = "auto",
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        the_model: PaceModel | None
        if model == "auto":
            try:
                the_model = PaceModel.latest()
                log.info("loaded model %s", the_model.model_dir)
            except FileNotFoundError:
                log.warning("no model artifacts — running without forecasts")
                the_model = None
        else:
            the_model = model  # already a PaceModel or None

        tasks = []
        if config.DATA_SOURCE == "live" and race_dir is None:
            from ingest.openf1 import OpenF1LiveClient

            runtime = SessionRuntime(engine=None, data_source="live")  # type: ignore[arg-type]
            app.state.runtime = runtime
            client = OpenF1LiveClient(runtime, the_model)
            tasks.append(asyncio.create_task(client.run()))
        else:
            rd = race_dir or resolve_race(config.REPLAY_RACE)
            spd = config.REPLAY_SPEED if speed is None else speed
            meta = load_session_meta(rd)
            engine = RaceEngine(meta, the_model)
            runtime = SessionRuntime(engine, "replay", replay_speed=spd)
            app.state.runtime = runtime
            tasks.append(asyncio.create_task(replay_task(runtime, rd, spd)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Pitwall", version="0.1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # personal project; tighten if it ever grows auth
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def runtime() -> SessionRuntime:
        rt = getattr(app.state, "runtime", None)
        if rt is None or rt.engine is None:
            raise HTTPException(503, "no active session")
        return rt

    @app.get("/api/health")
    async def health():
        rt = getattr(app.state, "runtime", None)
        return {
            "status": "ok",
            "data_source": config.DATA_SOURCE if rt is None else rt.data_source,
            "session": rt.engine.meta.session_id if rt and rt.engine else None,
            "lap": rt.engine.max_lap if rt and rt.engine else None,
        }

    @app.get("/api/session", response_model=schemas.SessionInfo)
    async def session():
        rt = runtime()
        meta = rt.engine.meta
        return schemas.SessionInfo(
            session_id=meta.session_id,
            event_name=meta.event_name,
            track_id=meta.track_id,
            laps_total=meta.laps_total,
            data_source=rt.data_source,
            replay_speed=rt.replay_speed,
        )

    @app.get("/api/state", response_model=schemas.LapUpdate)
    async def state():
        rt = runtime()
        if rt.latest is None:
            raise HTTPException(503, "no laps processed yet")
        return rt.latest

    @app.get("/api/history")
    async def lap_history():
        """Per-driver lap series so late-joining clients can backfill charts."""
        rt = runtime()
        return {str(d.driver_number): d.lap_log for d in rt.engine.drivers.values()}

    @app.get("/api/predictions")
    async def predictions():
        rt = runtime()
        return {
            str(d.driver_number): d.forecast.model_dump() if d.forecast else None
            for d in rt.engine.drivers.values()
        }

    @app.post("/api/simulate", response_model=schemas.SimulateResponse)
    async def simulate(req: schemas.SimulateRequest):
        rt = runtime()
        from sim.montecarlo import simulate_strategies

        try:
            setup = rt.engine.to_sim_setup(req.driver_number)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        strategies = None
        if req.strategies is not None:
            strategies = [list(s) for s in req.strategies]
        loop = asyncio.get_running_loop()
        baseline, results = await loop.run_in_executor(
            None, lambda: simulate_strategies(setup, strategies, req.n_sims)
        )
        return schemas.SimulateResponse(
            driver_number=req.driver_number,
            from_lap=setup.from_lap,
            n_sims=req.n_sims,
            baseline=baseline,
            strategies=results,
        )

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        rt = getattr(app.state, "runtime", None)
        if rt is None:
            await ws.close(code=1013)
            return
        since = ws.query_params.get("since")
        q, backlog = rt.subscribe(int(since) if since else None)
        try:
            for text in backlog:
                await ws.send_text(text)
            while True:
                text = await q.get()
                await ws.send_text(text)
        except WebSocketDisconnect:
            pass
        finally:
            rt.unsubscribe(q)

    return app


app = create_app()
